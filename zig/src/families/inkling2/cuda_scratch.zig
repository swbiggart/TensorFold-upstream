//! The op scratch a forward takes (cuda_ops.zig Scratch: a bump arena the forward resets before every op), counted
//! from the ops' own sizes so a step can be checked before its header goes out: a short scratch found after the header
//! stops every rank (the others wait in a collective), one found before refuses only the step.
//!
//! The takes, op by op (each op from an empty scratch; cuda_forward.zig resets before every op and, in a forward of
//! several segments, before each segment's attention core):
//!   attn.project  the lane matmul's split-K partials of the layer's projections (cuda_attn.zig linear), in one op
//!   attn.core     the split form's pairs, partials, kept scores and folded pairs (cuda_attn.zig Plan.scratch), a
//!                 segment at a time: its rows, its first position, and in a captured step the slot's capacity
//!   attn.out      the output projection's partials
//!   moe.experts   checkpoint mode: cuda_moe.zig ckScratchBytes; full mode: scratchBytes; split experts: splitUp's
//!                 quantized rows and route bitmaps (moe.down: splitDown's quantized activations and route rows)
//! All but the core depend only on the forward's rows; the core on each segment's position. The largest at a slot's
//! context over every forward the engine runs (`startupNeed`: up to prompt_rows rows eager, up to a verify window as
//! a captured step) is checked once at start-up; `fits` checks one forward.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const attn = @import("cuda_attn.zig");
const moe = @import("cuda_moe.zig");
const weights = @import("cuda_weights.zig");

const Kernels = ops.Kernels;
const H = weights.hidden;

/// The op scratch a rank holds by default, GiB (TF_INK2_SCRATCH_GIB, whole GiB, in the settings the ranks agree on).
/// Measured need at W = 4: about 0.2 GiB at a 131,072-position slot with 2,048-row prompt chunks (--exp scratch);
/// the start-up check (`checkStartup`) refuses a context whose need passes it.
pub const default_gib: usize = 1;

/// TF_INK2_SCRATCH_GIB, else `default_gib`.
pub fn gib() usize {
    const v = std.c.getenv("TF_INK2_SCRATCH_GIB") orelse return default_gib;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default_gib;
}

/// One segment of a forward: `rows` rows from position `pos0`; `positions` the slot's capacity when the forward runs
/// as a captured step (the core's scratch then spans the slot), 0 eager.
pub const Part = struct { pos0: usize, rows: usize, positions: usize = 0 };

/// The kind of forward `parts` make (cuda_ops.zig Scratch.kind, for the peaks by kind).
pub fn kindOf(parts: []const Part) ops.Scratch.Kind {
    if (parts.len > 1) return .batched;
    if (parts.len == 0) return .other;
    const R = parts[0].rows;
    return if (R == 1) .decode else if (R <= 8) .window else .prompt;
}

/// Takes in order, each 256-byte aligned as Scratch.take aligns: the op's high-water mark.
fn takes(list: []const usize) usize {
    var at: usize = 0;
    for (list) |b| if (b > 0) {
        at = std.mem.alignForward(usize, at, 256) + b;
    };
    return at;
}

/// The lane matmul's partials of a projection (0 for a layer not packed).
fn part(k: *const Kernels, l: *const weights.Layer, p: attn.Packed, R: usize) usize {
    if (l.extra[0] == 0) return 0;
    return k.attn.lane.partBytes(R, attn.lanes(l, p));
}

/// The most any op but the attention core takes in a forward of R rows (all segments together).
pub fn rowsNeed(k: *const Kernels, w: *const weights.Weights, R: usize) usize {
    if (R == 0) return 0;
    var most: usize = 0;
    const routes = R * moe.topk;
    const I = moe.intermediate;
    // layers alike in these give the same takes (2-4 classes in the model): count each class once
    var seen: [8][5]u32 = undefined;
    var nseen: usize = 0;
    layers: for (w.layers) |*l| {
        const E0 = &l.experts;
        const key = [5]u32{ @intFromBool(l.global), @intFromEnum(E0.format), E0.split, E0.count(), E0.n2 };
        for (seen[0..nseen]) |x| if (std.mem.eql(u32, &x, &key)) continue :layers;
        if (nseen < seen.len) {
            seen[nseen] = key;
            nseen += 1;
        }
        // attn.project: q, rel, gate, sink, k, v (+ the global layers' k and v window projections), project's order
        var pl: [8]usize = undefined;
        var n: usize = 0;
        for ([_]attn.Packed{ .q, .rel, .gate, .sink, .k, .v }) |p| {
            pl[n] = part(k, l, p, R);
            n += 1;
        }
        if (l.global) for ([_]attn.Packed{ .k_wproj, .v_wproj }) |p| {
            pl[n] = part(k, l, p, R);
            n += 1;
        };
        most = @max(most, takes(pl[0..n]));
        most = @max(most, part(k, l, .o, R));
        const E = &l.experts;
        if (E.count() == 0) continue;
        const nv = E.format == .nvfp4;
        if (E.split > 1) {
            // splitUp: the rows quantized (codes, scales, amax), then ckPlan's bitmaps over every expert
            most = @max(most, takes(&.{ if (nv) R * H / 2 else R * H, if (nv) R * H / 16 else R * H / 32, if (nv) R * 4 else 0, 2 * moe.experts_total * ((R + 31) / 32) * 4 }));
            // splitDown: the activations quantized, the route rows
            most = @max(most, takes(&.{ if (nv) routes * I / 2 else routes * I, if (nv) routes * I / 16 else routes * I / 32, if (nv) routes * 4 else 0, routes * @as(usize, E.n2) * 2 }));
        } else if (k.moe.mode == .checkpoint) {
            most = @max(most, moe.ckScratchBytes(R, E.count(), k.moe.shape(routes, nv, 0), E.format));
        } else most = @max(most, moe.scratchBytes(R, E.count(), moe.rowTiles(routes, E.count())));
    }
    return most;
}

/// The attention core's take for one segment (sliding and global layers; the rank's heads).
pub fn coreNeed(k: *const Kernels, hq: usize, hk: usize, g: Part) usize {
    var most: usize = 0;
    for ([_]usize{ 0, attn.window }) |win| {
        const kv: attn.Kv = .{ .k = 0, .v = 0, .slots = if (win == 0) @max(g.positions, g.pos0 + g.rows) else win - 1 + g.rows, .heads = hk, .window = win };
        const dev: ?attn.Dev = if (g.positions > 0) .{ .posp = 1, .positions = g.positions } else null;
        most = @max(most, attn.plan(&k.attn, g.rows, g.pos0, dev, kv, hq, hk).scratch(g.rows, hq));
    }
    return most;
}

/// The op scratch a forward of `parts` takes at most: its largest op's.
pub fn need(k: *const Kernels, w: *const weights.Weights, parts: []const Part) usize {
    if (w.layers.len == 0) return 0;
    const l0 = &w.layers[0];
    var R: usize = 0;
    var most: usize = 0;
    for (parts) |g| {
        R += g.rows;
        most = @max(most, coreNeed(k, l0.hq, l0.hk, g));
    }
    return @max(most, rowsNeed(k, w, R));
}

/// Before a step's header: whether `s` holds the forward of `parts` (every rank's scratch is the same size and the
/// ranks' shapes are alike, so rank 0's answer is every rank's). error.ScratchShort, logged with the need, when not.
pub fn fits(s: *const ops.Scratch, k: *const Kernels, w: *const weights.Weights, parts: []const Part) error{ScratchShort}!void {
    const n = need(k, w, parts);
    if (n <= s.buf.len) return;
    std.log.err("inkling2: a forward of {d} segment(s) ({d} rows, the first at position {d}) needs {d:.1} MiB of op scratch, {d} MiB held (TF_INK2_SCRATCH_GIB): refused before its header", .{ parts.len, rowsOf(parts), if (parts.len > 0) parts[0].pos0 else 0, ops.mib(n), s.buf.len >> 20 });
    return error.ScratchShort;
}

/// `fits` for the engine's segments (anything with `pos0` and `rows`): `positions` the slot's capacity when the
/// forward runs as a captured step, null eager.
pub fn fitsSegs(s: *const ops.Scratch, k: *const Kernels, w: *const weights.Weights, segs: anytype, positions: ?usize) error{ScratchShort}!void {
    var ps: [16]Part = undefined;
    if (segs.len > ps.len) return error.ScratchShort;
    for (segs, 0..) |g, i| ps[i] = .{ .pos0 = g.pos0, .rows = g.rows, .positions = positions orelse 0 };
    return fits(s, k, w, ps[0..segs.len]);
}

fn rowsOf(parts: []const Part) usize {
    var R: usize = 0;
    for (parts) |g| R += g.rows;
    return R;
}

/// The most op scratch any forward at a slot of `context` positions takes: up to `rows` rows (eager, any position)
/// and up to `graph_rows` as a captured step (cuda_attn.zig scratchNeed for the core), and every row count's other ops.
pub fn startupNeed(k: *const Kernels, w: *const weights.Weights, context: usize, rows: usize, graph_rows: usize) struct { total: usize, core: usize, other: usize } {
    if (w.layers.len == 0) return .{ .total = 0, .core = 0, .other = 0 };
    const l0 = &w.layers[0];
    const core = attn.scratchNeed(&k.attn, l0.hq, l0.hk, context, rows, graph_rows);
    var other: usize = 0;
    for (1..rows + 1) |R| other = @max(other, rowsNeed(k, w, R));
    return .{ .total = @max(core, other), .core = core, .other = other };
}

/// Start-up, on every rank, after the weights are loaded and prepared (the projections' lane forms): the op scratch
/// must hold `startupNeed`, so no forward of a slot of `context` positions can be short. error.ScratchShort with the
/// need logged; else the need is logged.
pub fn checkStartup(s: *const ops.Scratch, k: *const Kernels, w: *const weights.Weights, context: usize, rows: usize, graph_rows: usize, rank: u32) !void {
    const n = startupNeed(k, w, context, rows, graph_rows);
    if (n.total > s.buf.len) {
        std.log.err("inkling2 rank {d}: the op scratch ({d} MiB) is short of {d:.1} MiB at {d} positions a slot (attention core {d:.1}, other ops {d:.1}; raise TF_INK2_SCRATCH_GIB or lower the context)", .{ rank, s.buf.len >> 20, ops.mib(n.total), context, ops.mib(n.core), ops.mib(n.other) });
        return error.ScratchShort;
    }
    std.log.info("inkling2 rank {d}: op scratch need {d:.1} MiB of {d} MiB at {d} positions a slot (attention core {d:.1}, other ops {d:.1} at up to {d} rows)", .{ rank, ops.mib(n.total), s.buf.len >> 20, context, ops.mib(n.core), ops.mib(n.other), rows });
}

test "inkling2 cuda scratch: takes, kinds" {
    try std.testing.expectEqual(@as(usize, 0), takes(&.{}));
    try std.testing.expectEqual(@as(usize, 256 + 10), takes(&.{ 1, 0, 10 }));
    try std.testing.expectEqual(@as(usize, 512 + 4), takes(&.{ 300, 4 }));
    try std.testing.expectEqual(ops.Scratch.Kind.decode, kindOf(&.{.{ .pos0 = 9, .rows = 1 }}));
    try std.testing.expectEqual(ops.Scratch.Kind.window, kindOf(&.{.{ .pos0 = 9, .rows = 8 }}));
    try std.testing.expectEqual(ops.Scratch.Kind.prompt, kindOf(&.{.{ .pos0 = 0, .rows = 2048 }}));
    try std.testing.expectEqual(ops.Scratch.Kind.batched, kindOf(&.{ .{ .pos0 = 9, .rows = 1 }, .{ .pos0 = 3, .rows = 1 } }));
}
