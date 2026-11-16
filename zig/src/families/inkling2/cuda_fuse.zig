//! G-fuse: stateless neighbours of the forward as one launch each, and their one-row latency forms. Kernel image:
//! inkling2_fuse.cu. Each entry point runs the same expressions in the same order as the ops it replaces, so its
//! outputs are the same bits (checked by `tf-inkling2-fuse`, each form against the unfused sequence at 1, 2, 8 and
//! 2048 rows), and it writes every tensor those ops wrote (the mix outputs too), so later ops read the same memory.
//!
//!   attnIn  = res.mix (attn_input) + res.norm (attn_norm, or q/k/v_norm)       2 launches (4 global) -> 1
//!   mlpIn   = res.partial + state.pushBlock + res.mix (mlp_input) + res.norm     3 (4 at a boundary)   -> 1
//!   route   = moe.route's logits, then select + slot order (checkpoint mode)    3 -> 2
//!
//! Every form is row-invariant (a CTA a row, or a warp a row), so it also holds under any row count.
//!
//! Two forms of each (TF_INK2_FUSE, `Mode`), never while the forward's hooks record or feed ops:
//!   - the latency forms (tf_ink2_lat_*; the default, for calls of up to `lat_rows` rows: decode steps, verify
//!     windows): 576 threads a row, every source loaded once up front and kept in registers, a barrier a phase, the
//!     softmax on one warp's lanes; select's 8 rounds on two warp reductions each. At one row on GB10 (48 layers'
//!     worth, tf-inkling2-fuse) the three cost 27 us a layer against the release's 43; in the W = 4 decode step on
//!     the Sparks they save 1.23 +- 0.05 ms a step (tf-inkling2-decode --exp check,fuse, against an A/A control of
//!     -0.08 +- 0.06). Prompt chunks keep the release's ops.
//!   - the stage-1 fusions (tf_ink2_fuse_*, TF_INK2_FUSE=1): one launch each, but the same latency as the ops they
//!     replace (one CTA of 1,024 threads, one dependent round trip a source a pass); 0.18 ms slower than the
//!     release's ops at W = 4 (graph nodes cost about 0.15 us on GB10, so fewer nodes buy little).
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const res = @import("cuda_res.zig");
const moe = @import("cuda_moe.zig");

const Ctx = ops.Ctx;
const Mat = ops.Mat;
const Blocks = ops.Blocks;
const H = res.width;
/// Threads a row: norms.cu's tf_rms_rows block (the norm's sums are its order).
const row_threads = 1024;
const most_blocks = 8;
const most_queries = 3;

/// Which of the forward's stateless ops run in this file's forms (TF_INK2_FUSE): `fused` the stage-1 fusions (attnIn,
/// mlpIn, route on the tf_ink2_fuse_* kernels), the rest the latency forms one by one. All false: the release's ops.
pub const Mode = struct {
    fused: bool = false,
    attn: bool = false,
    mlp: bool = false,
    route: bool = false,

    pub const none: Mode = .{};
    pub const lat: Mode = .{ .attn = true, .mlp = true, .route = true };

    /// The forward's attention input runs attnIn (fused or latency form).
    pub fn attnIn(m: Mode) bool {
        return m.fused or m.attn;
    }

    /// The forward's MLP input runs mlpIn (fused or latency form).
    pub fn mlpIn(m: Mode) bool {
        return m.fused or m.mlp;
    }

    /// The forward's router runs `route` here.
    pub fn router(m: Mode) bool {
        return m.fused or m.route;
    }

    /// The forms a call of R rows runs: the latency forms only up to `lat_rows` rows (the stage-1 fusions at any).
    pub fn forRows(m: Mode, R: usize) Mode {
        return if (R <= lat_rows) m else .{ .fused = m.fused };
    }

    /// TF_INK2_FUSE: unset or lat every latency form (the default); 0 none (the release's ops); 1 the stage-1
    /// fusions; else a comma list of attn, mlp, route.
    pub fn parse(v: []const u8) !Mode {
        if (std.mem.eql(u8, v, "0")) return .none;
        if (std.mem.eql(u8, v, "1")) return .{ .fused = true };
        if (v.len == 0 or std.mem.eql(u8, v, "lat")) return .lat;
        var m: Mode = .{};
        var it = std.mem.splitScalar(u8, v, ',');
        while (it.next()) |w| {
            if (std.mem.eql(u8, w, "attn")) m.attn = true else if (std.mem.eql(u8, w, "mlp")) m.mlp = true else if (std.mem.eql(u8, w, "route")) m.route = true else return error.BadInk2Fuse;
        }
        return m;
    }
};

/// The most rows a call runs the latency forms at (decode steps, verify windows, a few streams' rows); prompt chunks
/// run the release's ops.
pub const lat_rows = 64;

/// inkling2_fuse.cu's functions, and the switch.
pub const Kernels = struct {
    g: @import("cuda_kernels.zig").Group("fuse"),
    attn_in: cuda.Function,
    mlp_in: cuda.Function,
    route_fn: cuda.Function,
    lat_attn_in: cuda.Function,
    lat_mlp_in: cuda.Function,
    lat_route: cuda.Function,
    /// TF_INK2_FUSE (default the latency forms): the forms the forward runs.
    mode: Mode = .lat,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var g = try @import("cuda_kernels.zig").Group("fuse").load(d);
        errdefer g.unload();
        const k: Kernels = .{
            .g = g,
            .attn_in = try g.mod.function("tf_ink2_fuse_attn_in"),
            .mlp_in = try g.mod.function("tf_ink2_fuse_mlp_in"),
            .route_fn = try g.mod.function("tf_ink2_fuse_route"),
            .lat_attn_in = try g.mod.function("tf_ink2_lat_attn_in"),
            .lat_mlp_in = try g.mod.function("tf_ink2_lat_mlp_in"),
            .lat_route = try g.mod.function("tf_ink2_lat_route"),
            .mode = try fromEnv(),
        };
        try k.lat_attn_in.allowDynamicShared(lat_staged * most_queries);
        try k.lat_mlp_in.allowDynamicShared(lat_staged);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.g.unload();
    }
};

/// TF_INK2_FUSE (see `Mode.parse`).
pub fn fromEnv() !Mode {
    return Mode.parse(std.mem.span(std.c.getenv("TF_INK2_FUSE") orelse return .lat));
}

/// The latency forms' threads a row (the mix's 576) and dynamic shared bytes a staged row (fp32).
const lat_threads = H / 8;
const lat_staged: u32 = H * 4;

fn mixScale() f32 {
    return @floatCast(1.0 / @sqrt(@as(f64, H)));
}

fn rowsOk(m: Mat, R: usize, dt: ops.Dt) bool {
    return m.is(R, H, dt) and m.dense() and m.ptr % 16 == 0;
}

fn blocksOk(b: Blocks, R: usize, most: usize) bool {
    return b.count <= most and b.hidden == H and (b.count == 0 or b.rows == R) and b.stride % 16 == 0 and b.ptr % 16 == 0;
}

/// res.attn_input then its norms: xs[j] fp32 = Mix(queries j; blocks, h_in) for j < nq (cuda_res.zig `mix`), then
/// ys[j] bf16 = norm(xs[nq == 1 ? 0 : j], gains[j]) for j < ys.len (cuda_res.zig `norm`).
pub fn attnIn(c: *const Ctx, blocks: Blocks, h_in: Mat, queries: u64, nq: usize, xs: []const Mat, gains: []const u64, ys: []const Mat) !void {
    const R = h_in.rows;
    const nn = ys.len;
    if (nq == 0 or nq > most_queries or xs.len < nq or queries == 0 or nn == 0 or nn > most_queries or gains.len < nn) return error.BadFuseShape;
    if (nq != 1 and nq != nn) return error.BadFuseShape;
    if (!blocksOk(blocks, R, most_blocks) or !rowsOk(h_in, R, .bf16)) return error.BadFuseShape;
    for (xs[0..nq]) |x| if (!rowsOk(x, R, .f32)) return error.BadFuseShape;
    for (ys) |y| if (!rowsOk(y, R, .bf16)) return error.BadFuseShape;
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(blocks.ptr);
    a.add(@as(i64, @intCast(blocks.stride / 2)));
    a.add(@as(i32, @intCast(blocks.count)));
    a.add(h_in.ptr);
    a.add(queries);
    a.add(@as(i32, @intCast(nq)));
    for (0..most_queries) |j| a.add(if (j < nq) xs[j].ptr else @as(u64, 0));
    a.add(res.eps);
    a.add(mixScale());
    a.add(@as(i32, @intCast(nn)));
    for (0..most_queries) |j| a.add(if (j < nn) gains[j] else @as(u64, 0));
    for (0..most_queries) |j| a.add(if (j < nn) ys[j].ptr else @as(u64, 0));
    a.add(res.eps);
    if (c.k.fuse.mode.attn) return cuda.launch.launch(c.k.fuse.lat_attn_in, .{ .grid = .{ .x = @intCast(R) }, .block = .{ .x = lat_threads }, .shared = lat_staged * @as(u32, @intCast(nq)) }, c.stream, &a);
    try cuda.launch.launch(c.k.fuse.attn_in, .{ .grid = .{ .x = @intCast(R) }, .block = .{ .x = row_threads } }, c.stream, &a);
}

/// res.partial, the boundary push, res.mlp_input and res.mlp_norm (cuda_forward.zig mlpStage):
///   part fp32 = boundary ? f32(ac) : f32(h_in) + f32(ac)
///   boundary: push (the next block, bf16 [R, H]) = h_in; the mix's sources are then `blocks` and h_in
///   mx fp32 = Mix(query; sources, part); m bf16 = norm(mx, gain)
/// `blocks` are the completed blocks before the push; the caller counts the pushed one.
pub fn mlpIn(c: *const Ctx, h_in: Mat, ac: Mat, boundary: bool, part: Mat, blocks: Blocks, push: u64, query: u64, mx: Mat, gain: u64, m: Mat) !void {
    const R = h_in.rows;
    if (!rowsOk(h_in, R, .bf16) or !rowsOk(ac, R, .bf16) or !rowsOk(part, R, .f32) or !rowsOk(mx, R, .f32) or !rowsOk(m, R, .bf16)) return error.BadFuseShape;
    if (!blocksOk(blocks, R, if (boundary) most_blocks - 1 else most_blocks) or query == 0) return error.BadFuseShape;
    if (boundary and (push == 0 or push % 16 != 0)) return error.BadFuseShape;
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(h_in.ptr);
    a.add(ac.ptr);
    a.add(@as(i32, @intFromBool(boundary)));
    a.add(part.ptr);
    a.add(blocks.ptr);
    a.add(@as(i64, @intCast(blocks.stride / 2)));
    a.add(@as(i32, @intCast(blocks.count)));
    a.add(if (boundary) push else @as(u64, 0));
    a.add(query);
    a.add(mx.ptr);
    a.add(res.eps);
    a.add(mixScale());
    a.add(gain);
    a.add(m.ptr);
    a.add(res.eps);
    if (c.k.fuse.mode.mlp) return cuda.launch.launch(c.k.fuse.lat_mlp_in, .{ .grid = .{ .x = @intCast(R) }, .block = .{ .x = lat_threads }, .shared = lat_staged }, c.stream, &a);
    try cuda.launch.launch(c.k.fuse.mlp_in, .{ .grid = .{ .x = @intCast(R) }, .block = .{ .x = row_threads } }, c.stream, &a);
}

/// moe.route in this file's forms: the same logits launch, then in checkpoint mode select and the slot order as one
/// launch (the latency form with `route`, else the stage-1 fusion); full mode keeps select's ascending order, so it
/// runs moe.route itself. The same ids, weights and logits as moe.route.
pub fn route(c: *const Ctx, l: *const @import("cuda_weights.zig").Layer, m: Mat, ids: Mat, wts: Mat, logits: Mat) !void {
    if (c.k.moe.mode != .checkpoint) return moe.route(c, l, m, ids, wts, logits);
    const R = m.rows;
    if (!m.is(R, H, .bf16) or !ids.is(R, moe.topk, .i32) or !wts.is(R, moe.topk, .f32) or !logits.is(R, moe.experts_total, .f32)) return error.BadMoeShape;
    if (!m.dense() or !ids.dense() or !wts.dense() or !logits.dense()) return error.BadMoeShape;
    if (R == 0) return;
    try moe.routeLogits(c, l, m, logits);
    try selectSlot(c, logits.ptr, l.router_bias, ids, wts);
}

/// select + slot order as one launch: ids, wts [R, 8] in slot order from logits fp32 [R, 384] and the selection bias
/// (tf_ink2_lat_route with `route`, else tf_ink2_fuse_route).
pub fn selectSlot(c: *const Ctx, logits: u64, bias: u64, ids: Mat, wts: Mat) !void {
    const R = ids.rows;
    if (R == 0) return;
    var b: cuda.launch.Args = .{};
    b.add(logits);
    b.add(bias);
    b.add(ids.ptr);
    b.add(wts.ptr);
    b.add(@as(i32, @intCast(R)));
    const f = if (c.k.fuse.mode.route) c.k.fuse.lat_route else c.k.fuse.route_fn;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((R + 7) / 8) }, .block = .{ .x = 256 } }, c.stream, &b);
}

test "inkling2 cuda fuse: the switch" {
    try std.testing.expect(H == 4608);
    try std.testing.expect(H / 8 <= row_threads);
    try std.testing.expectEqual(Mode.lat, try Mode.parse(""));
    try std.testing.expectEqual(Mode.lat, try Mode.parse("lat"));
    try std.testing.expectEqual(Mode.none, try Mode.parse("0"));
    try std.testing.expectEqual(Mode{ .fused = true }, try Mode.parse("1"));
    try std.testing.expectEqual(Mode{ .attn = true, .route = true }, try Mode.parse("attn,route"));
    try std.testing.expectError(error.BadInk2Fuse, Mode.parse("attn,x"));
    try std.testing.expectError(error.BadInk2Fuse, Mode.parse("logits"));
    try std.testing.expect(!(Mode{ .route = true }).mlpIn() and (Mode{ .route = true }).router());
    try std.testing.expectEqual(Mode.lat, Mode.lat.forRows(lat_rows));
    try std.testing.expectEqual(Mode.none, Mode.lat.forRows(lat_rows + 1));
    try std.testing.expectEqual(Mode{ .fused = true }, (Mode{ .fused = true }).forRows(2048));
}
