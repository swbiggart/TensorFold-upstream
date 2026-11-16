//! Where the embedding table [201,024, 4,608] bf16 (1.73 GiB) lives at W > 1: ranks 1 .. W-1 hold it by vocabulary
//! rows (canon's rule over W - 1 ranks: 67,008 rows, 0.58 GiB a rank at W = 4), rank 0 none: rank 0, the rank with
//! the least room (the HTTP side, the sampler, the head's logits), gets 1.73 GiB back. At W = 2 rank 1 holds the whole
//! table. W = 1 and --solo (no group): rank 0 holds the whole table and embeds a forward's rows itself.
//!
//! Start-up (`spread`): the rank folders are unchanged (the plan still places the table on rank 0, whose digests check
//! it). Rank 0 loads the whole table and its norm, then broadcasts both in 64 MiB slices over the group; each rank
//! keeps the rows it holds and rank 0 frees the table, before any room is measured.
//!
//! A forward (`rows`): every rank has the forward's token ids on the device (a captured step reads the step header's
//! copy, the drafter's walk copied into it on every rank as on rank 0; an eager forward of at most `graph_rows` rows
//! carries its ids in its header, a prompt chunk gets rank 0's ids by one broadcast, 4 B a row; the drafter's anchors
//! come in the header). Each rank copies the rows it holds into its slot of [W][R][H] bf16 (zeros for the rest), the
//! slots are all-gathered in rank order (the group's gather: the RoCE wire for a decode step's rows, NCCL for a
//! prompt chunk) and every rank copies each row from its owner's slot, then norms the rows with the kernel rank 0
//! uses at W = 1: every rank holds the bits rank 0's own embedding gives. No floating-point sum combines the shares
//! (-0 + +0 would change a bit).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ops = @import("cuda_ops.zig");
const head = @import("cuda_head.zig");
const weights = @import("cuda_weights.zig");

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;
const Span = ink.plan.Span;

/// The first rank holding rows (the owners are ranks 1 .. world - 1).
const first: u32 = 1;

/// The rows rank `rank` of `world` holds (vocabulary entries; the whole table on rank 0 at W = 1).
pub fn rowsOf(vocab: u32, rank: u32, world: u32) Span {
    if (world < 2) return if (rank == 0) .{ .lo = 0, .hi = vocab } else .{ .lo = 0, .hi = 0 };
    if (rank < first) return .{ .lo = 0, .hi = 0 };
    return ink.plan.canon(vocab, rank - first, world - first);
}

/// The rank holding entry `id` at W > 1 (tf_ink2_head_embed_pick's rule).
pub fn owner(vocab: u32, world: u32, id: u32) u32 {
    const n = world - first;
    const base = vocab / n;
    const extra = vocab % n;
    const edge = extra * (base + 1);
    const q = if (id < edge) id / (base + 1) else extra + (id - edge) / base;
    return first + @min(q, n - 1);
}

/// A rank's view of the table held by vocabulary rows (Engine.shared; null at W = 1 and --solo).
pub const Shared = struct {
    rank: u32,
    world: u32,
    vocab: u32,
    /// The gather's slots, bf16 [world][R][H] for R up to the exchange's capacity (the attention partials' buffer,
    /// idle outside a layer's exchange).
    slots: u64,
    group: ?*const cuda.tp_group.Group,

    /// `out` bf16 [R, H] = the normed embedding of `ids` (i32 [R] on the device, the same on every rank), on every
    /// rank; `raw` bf16 [R, H] is scratch (the looked-up rows before the norm).
    pub fn rows(s: *const Shared, c: *const Ctx, w: *const weights.Weights, ids: u64, raw: Mat, out: Mat) !void {
        const R = out.rows;
        if (raw.rows != R or raw.cols != H or out.cols != H) return error.BadHeadShape;
        const mine = s.slots + @as(u64, s.rank) * R * H * 2;
        if (w.embed_rows.len() > 0) try head.embedShare(c, w, ids, Mat.of(mine, R, H, .bf16));
        if (s.group) |g| try g.allGather(c.stream, mine, s.slots, R * H * 2);
        try head.embedPick(c, ids, s.slots, first, s.world - first, s.vocab, raw);
        try head.embedNorm(c, w, raw, out);
    }
};

/// A slice of the start-up broadcast (rows of the table).
const slice_bytes: usize = 64 << 20;

/// Start-up, every rank of `g` (W > 1; after the load and its digest check, before any room is read): rank 0's table
/// and norm broadcast in slices; each rank keeps its rows (w.embed, w.embed_rows) and the norm; rank 0 frees the
/// table. Waits for the device.
pub fn spread(d: *const cuda.Driver, s: cuda.Stream, g: *const cuda.tp_group.Group, w: *weights.Weights, rank: u32, world: u32) !void {
    if (world < 2) return error.BadRank;
    const vocab = w.vocab;
    const row: usize = H * 2;
    const mine = rowsOf(vocab, rank, world);
    const leader = rank == 0;
    if (leader and (w.embed == 0 or w.embed_rows.len() != vocab or w.embed_norm == 0)) return error.EmbedNotLoaded;
    if (!leader) w.embed_norm = try w.alloc(H * 4);
    try g.lib.check(g.lib.api.ncclBroadcast(w.embed_norm, w.embed_norm, H * 4, .u8, 0, g.comm, s.handle), "ncclBroadcast");
    const share: u64 = if (mine.len() == 0) 0 else try w.alloc(@as(usize, mine.len()) * row);
    var staging: ?cuda.DeviceBuffer = if (leader) null else try cuda.DeviceBuffer.alloc(d, slice_bytes);
    defer if (staging) |*b| b.free();
    const per: u32 = @intCast(slice_bytes / row);
    var lo: u32 = 0;
    while (lo < vocab) : (lo += per) {
        const hi = @min(vocab, lo + per);
        const src: u64 = if (leader) w.embed + @as(u64, lo) * row else staging.?.ptr;
        try g.lib.check(g.lib.api.ncclBroadcast(src, src, @as(usize, hi - lo) * row, .u8, 0, g.comm, s.handle), "ncclBroadcast");
        const a = @max(lo, mine.lo);
        const b = @min(hi, mine.hi);
        if (a < b) {
            const view: cuda.DeviceBuffer = .{ .d = d, .ptr = share + @as(u64, a - mine.lo) * row, .len = @as(usize, b - a) * row, .borrowed = true };
            try view.copyFrom(0, src + @as(u64, a - lo) * row, @as(usize, b - a) * row, s.handle);
        }
    }
    try s.synchronize();
    if (leader) {
        const whole = w.embed;
        const bytes = @as(u64, vocab) * row;
        if (!w.release(whole)) return error.EmbedNotLoaded;
        w.released -= bytes; // not a repacked source (the log's "dropped" figure)
    }
    w.embed = share;
    w.embed_rows = mine;
}

test "inkling2 cuda embed: every entry has one owner, the rank whose rows hold it" {
    const vocab: u32 = 201024;
    for ([_]u32{ 2, 3, 4, 8 }) |world| {
        var total: u32 = 0;
        for (0..world) |r| total += rowsOf(vocab, @intCast(r), world).len();
        try std.testing.expectEqual(vocab, total);
        var id: u32 = 0;
        while (id < vocab) : (id += 1) {
            const o = owner(vocab, world, id);
            try std.testing.expect(o >= 1 and o < world);
            const span = rowsOf(vocab, o, world);
            try std.testing.expect(id >= span.lo and id < span.hi);
        }
    }
    try std.testing.expectEqual(@as(u32, 0), rowsOf(vocab, 0, 4).len());
    try std.testing.expectEqual(@as(u32, 67008), rowsOf(vocab, 3, 4).len());
    try std.testing.expectEqual(vocab, rowsOf(vocab, 1, 2).len());
    try std.testing.expectEqual(vocab, rowsOf(vocab, 0, 1).len());
}
