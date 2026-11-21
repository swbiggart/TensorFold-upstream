//! W > 1, prompt chunks: a layer's rank-order sum (cuda_tp.zig) by rows instead of whole. Rank r owns rows
//! [r*S, r*S + n_r) of the forward (S = ceil(R / W)); every rank sends each other rank that rank's rows of its fp32
//! partial (one NCCL send/recv group), sums its own rows over the ranks in rank order (res.rankSumRows: the same adds
//! in the same order as res.rankSum, so every row's bits are those of the whole sum), and the bf16 rows are
//! all-gathered in rank order, S rows a rank, straight into the sum's output (rows R .. W*S - 1 of it are scratch).
//!
//! Bytes a rank receives for one sum: (W-1) * S * H * 4 + (W-1) * S * H * 2, against (W-1) * R * H * 4 for the
//! all-gather of whole partials: at W = 4 and 2048 rows 28.3 + 14.2 MB against 113 MB (2.7x less); at W = 2,
//! 18.9 + 9.4 against 37.7 (1.3x less). Decode-sized forwards keep the whole all-gather (one collective, no
//! send/recv group): below `min_rows`.
//!
//! Opt-in (TF_INK2_XROWS=1): the send/recv group connects every pair of ranks, which a switch carries but the
//! four-Spark switchless ring does not (its non-neighbours have no direct route; NCCL's own collectives follow the
//! ring). The bits are the same either way, but the NCCL calls are not: set it on every rank or none. At W = 2
//! over PCIe on one host it gains nothing (1.33x fewer bytes against a second collective); the Sparks at W = 4 on a
//! switch are where it is meant to pay.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const res = @import("cuda_res.zig");
const tp = @import("cuda_tp.zig");
const weights = @import("cuda_weights.zig");

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;

/// The fewest rows a forward sums by rows.
pub const min_rows = 64;

var enabled: ?bool = null;

/// TF_INK2_XROWS=1: prompt chunks sum by rows (see the header); unset or 0, the whole all-gather at every row count.
pub fn wanted() bool {
    if (enabled == null) enabled = if (std.c.getenv("TF_INK2_XROWS")) |v| v[0] == '1' else false;
    return enabled.?;
}

/// Rows a rank owns of a forward of R rows: S = ceil(R / W) each, the last ranks fewer (or none).
pub fn share(R: usize, world: usize, rank: usize) [2]usize {
    const S = (R + world - 1) / world;
    const lo = @min(R, rank * S);
    return .{ lo, @min(R, lo + S) };
}

/// Whether a forward of R rows sums by rows: real ranks, R >= min_rows, and the output holds W * S rows.
pub fn applies(x: *const tp.Exchange, R: usize, out_rows: usize) bool {
    if (x.group == null or x.world < 2 or R < min_rows or R > x.parts.cap or !wanted()) return false;
    const S = (R + x.world - 1) / x.world;
    return S * x.world <= out_rows;
}

/// Sum `s` of a forward of R rows (fp32 partials) into `out` (bf16, at least W * S rows from row 0; rows 0 .. R - 1
/// get the sum). Every rank's partial is in its slot of the sum's buffer (tp.Exchange.mine), not yet gathered.
pub fn sum(x: *const tp.Exchange, c: *const Ctx, s: tp.Sum, R: usize, out: Mat) !void {
    if (!tp.Timing.enabled()) return run(x, c, s, R, out);
    try c.stream.synchronize();
    const t = tp.Timing.now();
    try run(x, c, s, R, out);
    try c.stream.synchronize();
    tp.Timing.ns[@intFromEnum(s)] += tp.Timing.now() - t;
    tp.Timing.calls[@intFromEnum(s)] += 1;
}

fn run(x: *const tp.Exchange, c: *const Ctx, s: tp.Sum, R: usize, out: Mat) !void {
    const g = x.group orelse return error.NoGroup;
    const W: usize = x.world;
    const r: usize = x.rank;
    const S = (R + W - 1) / W;
    if (!out.is(out.rows, H, .bf16) or !out.dense() or out.rows < W * S) return error.BadExchangeShape;
    const row = H * 4;
    const base = x.parts.ptr[@intFromEnum(s)];
    const slot = R * row; // rank q's partial [R][H] at base + q * slot (tp.Exchange.mine)
    const own = share(R, W, r);
    // rank q's rows of this rank's partial to q; this rank's rows of q's partial into q's slot, at their own rows
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        for (0..W) |q| {
            if (q == r) continue;
            const theirs = share(R, W, q);
            if (theirs[1] > theirs[0]) g.send(c.stream, base + r * slot + theirs[0] * row, (theirs[1] - theirs[0]) * row, q) catch |e| break :post e;
            if (own[1] > own[0]) g.recv(c.stream, base + q * slot + own[0] * row, (own[1] - own[0]) * row, q) catch |e| break :post e;
        }
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    if (own[1] > own[0]) try res.rankSumRows(c, base + own[0] * row, W, slot, out.slice(own[0], own[1] - own[0]));
    // every rank's S rows (the last ranks' padded with scratch rows) into out in rank order
    try g.allGather(c.stream, out.row(r * S), out.ptr, S * H * 2);
}

test "inkling2 cuda xrows: shares" {
    try std.testing.expectEqual([2]usize{ 0, 512 }, share(2048, 4, 0));
    try std.testing.expectEqual([2]usize{ 1536, 2048 }, share(2048, 4, 3));
    // 1967 rows: 492 a rank, the last 491
    try std.testing.expectEqual([2]usize{ 984, 1476 }, share(1967, 4, 2));
    try std.testing.expectEqual([2]usize{ 1476, 1967 }, share(1967, 4, 3));
    // 65 rows at W = 4: 17, 17, 17, 14
    try std.testing.expectEqual([2]usize{ 51, 65 }, share(65, 4, 3));
    var n: usize = 0;
    for (0..4) |q| {
        const x = share(1967, 4, q);
        n += x[1] - x[0];
    }
    try std.testing.expectEqual(@as(usize, 1967), n);
}
