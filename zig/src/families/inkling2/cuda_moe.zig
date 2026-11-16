//! G-moe: the router (sigmoid scores, selection by score + bias, weights normalised and scaled by 8) and the
//! grouped experts in `full` mode: stored weights decoded in the kernel, bf16 activations (W4A16 for the NVFP4
//! layers 0-39, W8A16 for the MXFP8 layers 40-47), softcapped SwiGLU, the weighted fp32 sum in expert-id order.
//! Kernel image: inkling2_moe.cu. Math: Transformers modeling_inkling2.py (Apache-2.0), Inkling2TopKRouter,
//! Inkling2Experts, Inkling2SoftcappedSwiGLU; expert decode as tools/zig/inkling2_ref.py `dequant_rows`.
//!
//! `route` (moe.route, L19), input m bf16 [R, 4608] (the mlp norm's output):
//!   logits fp32 [R, 384] = f32(m) . router^T (router fp32 [384, 4608]; a true fp32 product, no TF32)
//!   s = sigmoid(logits); pick the 8 largest of s + router_bias (fp32 [384]; selection only; ties to the lower id)
//!   w_k = s_k / (sum_8 s + 1e-20) * 8.0 (fp32)
//!   ids i32 [R, 8] in ascending expert id, wts fp32 [R, 8] in the same order.
//!   Oracle: route.ids.<L>, route.w.<L> (every row), route.logits.<L> (kept rows); op.<L>.moe.route.out.*.
//!   Kernels: tf_ink2_moe_logits (4 rows a block, a warp an expert: a lane's fma chain over its 144 k, then the
//!   warp's butterfly), tf_ink2_moe_select (a warp a row; the sum of the 8 picks in ascending id). Neither order
//!   is cuBLAS's or torch's, so logits and weights may differ from the oracle in the last fp32 bits.
//!
//! `experts` (moe.experts, L20, full mode) -> out bf16 [R, 4608]:
//!   weights decoded to bf16 exactly as the oracle does (cuda_weights.zig `Experts`):
//!     NVFP4: W = bf16(f32(e2m1 * e4m3) * f32(amax[e] / den))   (amax / den one fp32 division a expert)
//!     MXFP8: W = bf16(e4m3 * 2^(ue8m0 - 127))                  (exact)
//!   w13 rows interleaved: gate_i = row 2i, up_i = row 2i + 1 (i < 2304).
//!   per route (t, e): gu = bf16(m_t . W13_e^T) [4608]; g, u = its gate and up halves (fp32)
//!     act = bf16((tanh(g / 4) * 4) * sigmoid(g) * (tanh(u / 25) * 25))     (sigmoid of the raw gate)
//!     d = bf16(act . W2_e^T) [4608]
//!   out_t = bf16(sum over t's routes in ascending expert id of f32(d) * w)  (fp32 accumulation from 0)
//!   (the oracle manifest's expert_sum_order for full mode). At W > 1 (Phase 2) a rank sums only its experts
//!   [l.experts.lo, hi) into an fp32 partial and the ranks' partials are summed in rank order.
//!   Oracle: op.<L>.moe.experts.out.y; feeds the post-MLP conv, whose chain ends in h.<L>.
//!   Kernels, all on the op stream, no host round trip: tf_ink2_moe_plan groups the R * 8 routes by expert
//!   (deterministic: a thread an expert, routes in order) into items of at most 16 * RT routes;
//!   tf_ink2_moe_{up,down}_{nvfp4,mxfp8}_r{1,4} run every (item, 32-column block) as a warp (W4A16 / W8A16 with
//!   the weights decoded to the oracle's bf16 in registers, bf16 m16n8k16 MMAs; each 64-k step's four MMAs start
//!   from zero and are added to the fp32 sum with one IEEE add, the full K in one fixed order, so the tensor core's
//!   truncating accumulation spans 64 k, not 4608: measured against fp64-summed GEMMs this is closer than cuBLAS);
//!   the up kernel's epilogue applies the capped SwiGLU to the (gate, up) column pairs and writes
//!   act [route, 2304], the down kernel writes d [route, 4608] bf16; tf_ink2_moe_combine sums each row's routes in
//!   ascending id (it sorts the ids itself, so torch's route order is fine under --feed). RT (row tiles a warp:
//!   1 for decode windows and short chunks, 4 for long prompt chunks) changes only which warp computes a row, never
//!   a row's arithmetic, so a row's bits do not depend on R.
//!   Scratch: R * 8 * (2304 + 4608) * 2 bytes plus the plan (216 MiB at R = 2048).
//!   Oracle note: torch's bf16 GEMMs run with cuBLAS's reduced-precision (bf16) split-K reduction allowed (torch's
//!   default), which puts the oracle's own experts output further from exact sums than these kernels; differences
//!   against the oracle are mostly the oracle's (tools/zig/inkling2_moe_check.py's "floor" lines).
//!
//! `checkpoint` mode (TF_INK2_MODE=checkpoint or the runner's --mode checkpoint; `full` is the default): the experts
//! as the checkpoint was trained and is served (TokenSpeed's Torch reference, tokenspeed-kernel ops/moe/
//! block_scaled.py, MIT; the oracle's `--mode checkpoint`), on the block-scaled tensor cores (inkling2_ck.cu):
//!   route: the same picks and weights, put in TokenSpeed's slot order (descending sigmoid + bias, ties to the lower
//!     id), the order its fold sums them; ids and wts leave `route` in that order.
//!   NVFP4 layers: m rows to NVFP4 per token (amax / 1536 outer, an e4m3 scale a 16 by the four/six rule), FP4 x FP4
//!     mma against w13 as stored; gu = bf16(sum * (amax_row * w13_amax[e]) / (1536 * den13)).
//!   MXFP8 layers: m rows to MXFP8 (ue8m0 a 32, rounded up), FP8 x FP8 mma against w13 as stored; gu = bf16(sum).
//!   act = bf16(capped_swiglu(gate, up) * w) (the routing weight inside, before the store), re-quantized per route
//!   in the layer's format, the down projection the same way: d = bf16(sum [* (amax_act * w2_amax[e]) / (1536 *
//!   den2)]); out_t = bf16(fp32 sum of t's d rows in slot order) (no weight: it is inside act). At W > 1 the rank's
//!   fp32 partial over its experts, summed in rank order as in full mode (TokenSpeed all-reduces bf16 rank sums).
//!   Kernels: tf_ink2_ck_quant_{nvfp4,mxfp8} (a block a row; activation rows only for this rank's routes),
//!   tf_ink2_ck_{up,down}_{nvfp4,mxfp8}_s{0,1,2} (a block a unit (item, column block): 16 rows by 8 columns for
//!   decode, 16 by 32, or 64 by 32; its four warps sum K's four quarters, each mma into a fresh fragment and one IEEE
//!   add, and the quarters are added in order), _s3 (from three rows: 64 rows by 128 columns a block of 8 warps, A
//!   and B staged in shared memory 128 bytes a row at a time, each warp (32 by 32) walking all of K with the same
//!   quarter sums), _s4 and _s5 (GB10 windows of 2-8 rows: 16 rows by 64 columns, 384- or 256-byte stages, the
//!   scales staged too; per format and GEMM, Kernels.ck_rows), tf_ink2_ck_fold. The shape changes which block computes
//!   an output, never its arithmetic.
//!   Scratch at R rows: R * (H/2 + H/16 + 4) or R * (H + H/32) for the quantized rows, plus per route 2304 * 2
//!   (act) + 1152 + 144 + 4 or 2304 + 72 (act quantized) + 4608 * 2 (d), plus the plan: 258 MiB at R = 2048.
//!
//! Split experts (plan.Layout.split, checkpoint mode only; W ranks, rank r holding w13 rows [r*2I/W, ..) and w2
//! output rows [r*H/W, ..) of every expert): one layer's experts in two halves around the exchanges (cuda_tp.zig):
//!   `splitUp`: the plan (kept in the exchange's buffer for the second half), x quantized per token, the up GEMM over
//!     the rank's n13 = 2I/W gate/up rows of each route's expert, the capped SwiGLU times the routing weight: h, the
//!     rank's I/W columns of act [R * 8, I/W] bf16.
//!   MoE-A: h all-gathered in rank order ([W][R * 8][I/W]).
//!   `splitDown`: each route's whole act row (its W segments) re-quantized per token over all I = 2304 values, as at
//!     W = 1; the down GEMM over the rank's n2 = H/W output rows with all of K; each row's routes folded in slot
//!     order (fp32, one bf16 rounding): y, the rank's columns [R, H/W] bf16.
//!   MoE-B: y all-gathered in rank order and laid out as rows (`splitColumns`).
//! Every output element is computed whole on one rank by the same kernels in the same order as at W = 1 (a row's bits
//! never depend on the call's shape, see above; a column's never on which columns share its call), so the layer's
//! output is bit-equal to W = 1 checkpoint mode, on every rank.
//!
//! Test alone: the runner with --feed <capture> --dump-ops (route and experts get torch's m, ids and weights),
//! then inkling2_compare.py --ops; or inkling2_op_ref.py check on a --dump-ops dump. tf-inkling2-moe
//! (cuda_moe_main.zig) runs both ops on any rows (2,048 and more) at several chunk sizes and compares a row's
//! bits across them; tools/zig/inkling2_moe_check.py builds its input from captures and checks it against torch.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");

const Ctx = ops.Ctx;
const Mat = ops.Mat;
const Layer = weights.Layer;
const H = weights.hidden;

pub const experts_total = 384;
pub const topk = 8;
pub const intermediate = 2304;
pub const route_scale: f32 = 8.0;
pub const gate_cap: f32 = 4.0;
pub const up_cap: f32 = 25.0;

/// The experts' arithmetic: `full` (bf16 activations, the Transformers reference) or `checkpoint` (activations
/// quantized as the checkpoint requires, TokenSpeed's numerics, on the block-scaled tensor cores).
pub const Mode = enum { full, checkpoint };

/// TF_INK2_MODE (full when unset or unknown).
pub fn envMode() Mode {
    const v = std.c.getenv("TF_INK2_MODE") orelse return .full;
    return std.meta.stringToEnum(Mode, std.mem.span(v)) orelse .full;
}

/// inkling2_moe.cu's launch shapes (keep in step with the kernel's constants).
const logits_rows = 4;
const logits_split = 24;
const select_rows = 8;
const gemm_warps = 4;
const gemm_cols = 32;

/// inkling2_moe.cu's functions.
pub const Kernels = struct {
    mod: cuda.Module,
    probe: cuda.Function,
    logits: cuda.Function,
    select: cuda.Function,
    plan: cuda.Function,
    combine: cuda.Function,
    combine_f32: cuda.Function,
    /// [format: nvfp4, mxfp8][up, down][row tiles: 1, 4]
    gemm: [2][2][2]cuda.Function,
    /// checkpoint mode (inkling2_ck.cu)
    mode: Mode,
    ck_mod: cuda.Module,
    ck_slot: cuda.Function,
    /// the route plan without per-expert scans (tf_ink2_moe_plan's output)
    ck_plan: cuda.Function,
    /// [format: nvfp4, mxfp8]
    ck_quant: [2]cuda.Function,
    /// [format][up, down][shape: 16x8, 16x32, 64x32 a warp's unit; 64x128 a block, staged; 4.. window tiles (ckGeo)]
    ck_gemm: [2][2][ck_shapes]cuda.Function,
    ck_fold: cuda.Function,
    ck_fold_f32: cuda.Function,
    /// split experts: MoE-B's gathered columns laid out as rows
    ck_columns: cuda.Function,
    /// the unit shape of calls of 1 .. 8 rows in place of ckShape's (null: ckShape's): TF_INK2_CK_SHAPE1, else the
    /// GB10 table on an integrated GPU (windowShapes); [nvfp4, mxfp8][up, down][rows - 1]
    ck_rows: Rows,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_moe);
        errdefer mod.unload();
        var ck = try cuda.Module.load(d, cuda.kernels.inkling2_ck);
        errdefer ck.unload();
        var k: Kernels = .{
            .mode = envMode(),
            .ck_rows = try windowShapes(d),
            .ck_mod = ck,
            .ck_slot = try ck.function("tf_ink2_ck_slot_order"),
            .ck_plan = try ck.function("tf_ink2_ck_plan"),
            .ck_quant = .{ try ck.function("tf_ink2_ck_quant_nvfp4"), try ck.function("tf_ink2_ck_quant_mxfp8") },
            .ck_gemm = undefined,
            .ck_fold = try ck.function("tf_ink2_ck_fold"),
            .ck_fold_f32 = try ck.function("tf_ink2_ck_fold_f32"),
            .ck_columns = try ck.function("tf_ink2_ck_columns"),
            .mod = mod,
            .probe = try mod.function("tf_ink2_moe_probe"),
            .logits = try mod.function("tf_ink2_moe_logits"),
            .select = try mod.function("tf_ink2_moe_select"),
            .plan = try mod.function("tf_ink2_moe_plan"),
            .combine = try mod.function("tf_ink2_moe_combine"),
            .combine_f32 = try mod.function("tf_ink2_moe_combine_f32"),
            .gemm = undefined,
        };
        const names = [2][2][2][:0]const u8{
            .{ .{ "tf_ink2_moe_up_nvfp4_r1", "tf_ink2_moe_up_nvfp4_r4" }, .{ "tf_ink2_moe_down_nvfp4_r1", "tf_ink2_moe_down_nvfp4_r4" } },
            .{ .{ "tf_ink2_moe_up_mxfp8_r1", "tf_ink2_moe_up_mxfp8_r4" }, .{ "tf_ink2_moe_down_mxfp8_r1", "tf_ink2_moe_down_mxfp8_r4" } },
        };
        for (0..2) |f| for (0..2) |p| for (0..2) |r| {
            k.gemm[f][p][r] = try mod.function(names[f][p][r]);
        };
        const fmts = [2][]const u8{ "nvfp4", "mxfp8" };
        const passes = [2][]const u8{ "up", "down" };
        @setEvalBranchQuota(200_000);
        inline for (0..2) |f| inline for (0..2) |p| inline for (0..ck_shapes) |sh| {
            k.ck_gemm[f][p][sh] = try ck.function(std.fmt.comptimePrint("tf_ink2_ck_{s}_{s}_s{d}", .{ passes[p], fmts[f], sh }));
            const geo = ckGeo(sh, f == 0, if (p == 0) H else intermediate);
            if (geo.fits and geo.smem > 0) try k.ck_gemm[f][p][sh].allowDynamicShared(geo.smem);
        };
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.ck_mod.unload();
        k.mod.unload();
    }

    /// The unit shape of a checkpoint-mode GEMM (`pass` 0 up, 1 down) over `routes` (R * 8) of a layer of this
    /// format: ckShape's, or ck_rows' for 1 .. 8 rows.
    pub fn shape(k: *const Kernels, routes: usize, nvfp4: bool, pass: usize) u32 {
        const R = routes / topk;
        if (R >= 1 and R <= window_rows) if (k.ck_rows[if (nvfp4) 0 else 1][pass][R - 1]) |s| return s;
        return ckShape(routes);
    }

    /// A call's up and down shapes (`force` both); the down GEMM reads the up GEMM's plan, whose items hold at most
    /// the up shape's tile of routes and never more than R (a row routes to an expert once), so the down shape's tile
    /// must hold min(R, that).
    pub fn shapes(k: *const Kernels, routes: usize, nvfp4: bool, force: ?u32) ![2]u32 {
        const s: [2]u32 = if (force) |x| ckUnpair(x) else .{ k.shape(routes, nvfp4, 0), k.shape(routes, nvfp4, 1) };
        if (s[0] >= ck_shapes or s[1] >= ck_shapes) return error.BadMoeShape;
        if (ckGeo(s[1], nvfp4, intermediate).tile < @min(routes / topk, ckGeo(s[0], nvfp4, H).tile)) return error.BadMoeShape;
        return s;
    }
};

/// Window shapes: [nvfp4, mxfp8][up, down][rows - 1], null where ckShape's applies.
pub const Rows = [2][2][window_rows]?u32;

/// Calls of 1 .. window_rows rows (decode steps, DSpark windows, batched steps) may take their own unit shape.
pub const window_rows = 8;

/// The GB10 shapes of calls of 1 .. 8 rows: [nvfp4, mxfp8][up, down][rows - 1] (see ckShape for the measurements).
pub const gb10_rows = [2][2][window_rows]u32{
    .{ .{ 4, 4, 4, 4, 4, 5, 5, 5 }, .{ 4, 4, 4, 4, 4, 4, 4, 4 } },
    .{ .{ 5, 5, 5, 5, 5, 5, 5, 5 }, .{ 5, 5, 5, 5, 5, 5, 5, 5 } },
};

/// The checkpoint-mode unit shapes of calls of 1 .. 8 rows, where not ckShape's (null): TF_INK2_CK_SHAPE1 (a debug
/// switch, the ranks agree on it in the settings fingerprint): one shape S for one- and two-row calls (rows 3 .. 8
/// ckShape's 3: TF_INK2_CK_SHAPE1=3 is inkling2-release 116b8232f's choice on a GB10), or a list for 1, 2, .. rows, each
/// S (both GEMMs) or U:D (up and down), for both formats; else gb10_rows on an integrated GPU (GB10: its LPDDR5x
/// streams the staged shapes' long runs a weight row far better than shape 0's 32-byte steps), else none. Every shape
/// gives a row the same bits.
pub fn windowShapes(d: *const cuda.Driver) !Rows {
    if (std.c.getenv("TF_INK2_CK_SHAPE1")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0) return parseShapes(v);
    }
    var out: Rows = @splat(@splat(@splat(null)));
    if (try cuda.download_stage.integrated(d)) {
        for (0..2) |f| for (0..2) |p| for (0..window_rows) |r| {
            out[f][p][r] = gb10_rows[f][p][r];
        };
    }
    return out;
}

fn parseShapes(v: []const u8) !Rows {
    var out: Rows = @splat(@splat(@splat(null)));
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, v, ',');
    while (it.next()) |x| : (n += 1) {
        if (n == window_rows) return error.BadInk2CkShape1;
        const colon = std.mem.indexOfScalar(u8, x, ':');
        const up = std.fmt.parseInt(u32, x[0 .. colon orelse x.len], 10) catch return error.BadInk2CkShape1;
        const down = if (colon) |c| std.fmt.parseInt(u32, x[c + 1 ..], 10) catch return error.BadInk2CkShape1 else up;
        if (up >= ck_shapes or down >= ck_shapes) return error.BadInk2CkShape1;
        for (0..2) |f| {
            out[f][0][n] = up;
            out[f][1][n] = down;
        }
    }
    if (n == 1) for (0..2) |f| for (0..2) |p| {
        out[f][p][1] = out[f][p][0];
    };
    return out;
}

fn launch(c: *const Ctx, f: cuda.Function, grid: [2]u32, threads: u32, a: *cuda.launch.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1] }, .block = .{ .x = threads } }, c.stream, a);
}

/// L19: logits, the picks (ascending id) and their weights (see the header).
pub fn route(c: *const Ctx, l: *const Layer, m: Mat, ids: Mat, wts: Mat, logits: Mat) !void {
    const R = m.rows;
    if (!m.is(R, H, .bf16) or !ids.is(R, topk, .i32) or !wts.is(R, topk, .f32) or !logits.is(R, experts_total, .f32)) return error.BadMoeShape;
    if (!m.dense() or !ids.dense() or !wts.dense() or !logits.dense()) return error.BadMoeShape;
    if (R == 0) return;
    const k = &c.k.moe;
    try routeLogits(c, l, m, logits);
    var b: cuda.launch.Args = .{};
    b.add(logits.ptr);
    b.add(l.router_bias);
    b.add(ids.ptr);
    b.add(wts.ptr);
    b.add(@as(i32, @intCast(R)));
    try launch(c, k.select, .{ @intCast(ceilDiv(R, select_rows)), 1 }, 32 * select_rows, &b);
    if (k.mode == .checkpoint) {
        var o: cuda.launch.Args = .{};
        o.add(logits.ptr);
        o.add(l.router_bias);
        o.add(ids.ptr);
        o.add(wts.ptr);
        o.add(@as(i32, @intCast(R)));
        try launch(c, k.ck_slot, .{ @intCast(ceilDiv(R, 128)), 1 }, 128, &o);
    }
}

/// route's first launch: logits fp32 [R, 384] = f32(m) . router^T (cuda_fuse.zig route runs it too).
pub fn routeLogits(c: *const Ctx, l: *const Layer, m: Mat, logits: Mat) !void {
    const R = m.rows;
    if (R == 0) return;
    var a: cuda.launch.Args = .{};
    a.add(m.ptr);
    a.add(l.router);
    a.add(logits.ptr);
    a.add(@as(i32, @intCast(R)));
    try launch(c, c.k.moe.logits, .{ @intCast(ceilDiv(R, logits_rows)), logits_split }, 256, &a);
}

/// L20: the routed experts' weighted sum in full mode (see the header). `out` bf16 [R, H], or fp32 at W > 1: the
/// rank's partial over its experts [lo, hi), unrounded, for the rank-order sum after the exchange.
pub fn experts(c: *const Ctx, l: *const Layer, m: Mat, ids: Mat, wts: Mat, out: Mat) !void {
    if (l.experts.split > 1) return error.SplitExpertsTakeTwoHalves;
    if (c.k.moe.mode == .checkpoint) return expertsCk(c, l, m, ids, wts, out, null);
    return expertsTiled(c, l, m, ids, wts, out, null);
}

/// Checkpoint mode's unit shape for a call's `routes` (all of them, R * 8) on a discrete GPU: 0 (16 rows by 8 columns
/// a block of four K-quarter warps) for one or two rows, where a block's few routes want the most blocks reading
/// weights at once; 3 (64 rows by 128 columns, A and B staged through shared memory) from three rows on. Measured at
/// the rank-local shape (W = 4, 96 experts) on sm_120: one row 0.054 ms (0) against 0.060 (3) on NVFP4, 0.046 against
/// 0.069 on MXFP8; 8 rows 0.254 against 0.174 and 0.217 against 0.188; 3 wins at every size above. On a GB10 (the
/// Sparks, split experts at W = 4, cold weights) 3 wins at one row too: NVFP4 up 113 us against 184 (211 against 130
/// GB/s), down 61 against 82, MXFP8 up 204 against 214, down 100 against 108; and at two rows (the whole block 0.33 ms
/// against 0.53 on NVFP4), so one-row calls took 3 there before the window tiles (Kernels.ck_rows, gb10_rows). Shapes 1 (16 by 32) and 2 (64
/// by 32 unstaged) stay for tests (tf-inkling2-moe --force): every shape gives a row the same bits.
/// Calls of 2-8 rows on a GB10 (DSpark verify windows, batched steps; Kernels.ck_rows, gb10_rows) take the window
/// tiles 4 and 5: shape 3's staged tile cut to 16 rows by 64 columns, with longer runs of a weight row a stage (384
/// bytes, or 256 in 3 stages) and the block scales staged with them. Spark (W = 4, rank 1, cold weights, 256 real rows
/// in calls of R rows, the block's CUDA-event time a call) against shape 3, NVFP4 layer 0 / MXFP8 layer 40:
///   rows 2: 0.332 against 0.346 ms / 0.502 against 0.519; 4: 0.558 / 0.578, 0.807 / 0.837; 8: 0.959 / 0.991,
///   1.309 / 1.354 (the GEMMs at 217-230 GB/s against 207-220; a plain read of the same pattern reaches ~240).
/// One-row calls (decode steps) take them too: 0.205 against 0.210 ms, 0.318 against 0.333 (~0.35 ms a W = 4 step).
pub fn ckShape(routes: usize) u32 {
    return if (routes <= 2 * topk) 0 else 3;
}

/// Checkpoint-mode unit shapes: 0-2 (ck_body), 3 (ck_tile), 4 .. ck_shapes - 1 (ck_stage, the window tiles).
pub const ck_shapes = 4 + ck_stages.len;

/// A window tile (inkling2_ck.cu's TF_INK2_CK_STAGE table, in the same order from shape 4): 16-row tiles a warp, warps
/// down and across (16 trt wm rows by 32 wn columns a block), stages of `rb` bytes a row, scales staged with them.
const Stage = struct { trt: u32, wm: u32, wn: u32, nstg: u32, rb: u32, ss: bool };
const ck_stages = [_]Stage{
    .{ .trt = 1, .wm = 1, .wn = 2, .nstg = 2, .rb = 384, .ss = true },
    .{ .trt = 1, .wm = 1, .wn = 2, .nstg = 3, .rb = 256, .ss = true },
};

/// The most dynamic shared memory a block may take (sm_120 / sm_121: 99 KiB).
const smem_max = 99 << 10;

/// A unit shape's launch: rows an item (the plan's tile), columns a block, threads a block, dynamic shared bytes.
pub const CkGeo = struct {
    tile: usize,
    cols: usize,
    threads: u32,
    smem: u32,
    /// false: the shape cannot run this GEMM (its stages do not divide a weight row, or its shared memory is short)
    fits: bool = true,
};

/// A forced shape (the `force_shape` of expertsCk, splitUp, splitDown): S for both GEMMs, or ckPair(up, down).
pub fn ckPair(up: u32, down: u32) u32 {
    return up | (down + 1) << 16;
}

pub fn ckUnpair(x: u32) [2]u32 {
    return if (x >> 16 == 0) .{ x, x } else .{ x & 0xFFFF, (x >> 16) - 1 };
}

/// Whether a forced shape (S or ckPair) runs both GEMMs of a layer of this format (see CkGeo.fits).
pub fn ckFits(force: u32, nvfp4: bool) bool {
    const s = ckUnpair(force);
    return s[0] < ck_shapes and s[1] < ck_shapes and ckGeo(s[0], nvfp4, H).fits and ckGeo(s[1], nvfp4, intermediate).fits;
}

/// The launch of `shape` for a GEMM over K (H for up, intermediate for down).
pub fn ckGeo(shape: usize, nvfp4: bool, K: usize) CkGeo {
    return switch (shape) {
        0 => .{ .tile = 16, .cols = 8, .threads = 32 * gemm_warps, .smem = 0 },
        1 => .{ .tile = 16, .cols = 32, .threads = 32 * gemm_warps, .smem = 0 },
        2 => .{ .tile = 64, .cols = 32, .threads = 32 * gemm_warps, .smem = 0 },
        3 => .{ .tile = 64, .cols = 128, .threads = 256, .smem = 0 },
        else => {
            const g = ck_stages[shape - 4];
            const tm = 16 * g.trt * g.wm;
            const tn = 32 * g.wn;
            // scale bytes a row a stage: a step is 32 bytes and 4 scales (NVFP4) or 64 bytes and 2 (MXFP8)
            const ssb: u32 = if (!g.ss) 0 else if (nvfp4) g.rb / 8 else g.rb / 32;
            const smem = g.nstg * (tm + tn) * (g.rb + ssb);
            const rowb = if (nvfp4) K / 2 else K;
            return .{ .tile = tm, .cols = tn, .threads = 32 * g.wm * g.wn, .smem = smem, .fits = rowb % g.rb == 0 and smem <= smem_max };
        },
    };
}

/// The device scratch checkpoint-mode `experts` takes for R rows.
pub fn ckScratchBytes(R: usize, count: usize, shape: u32, format: weights.Format) usize {
    const routes = R * topk;
    const tile = ckGeo(shape, format == .nvfp4, H).tile;
    const items = maxItems(routes, count, tile);
    const xq = if (format == .nvfp4) R * (H / 2 + H / 16 + 4) else R * (H + H / 32);
    const aq = if (format == .nvfp4) routes * (intermediate / 2 + intermediate / 16 + 4) else routes * (intermediate + intermediate / 32);
    return 16 * 256 + routes * 4 + items * 3 * 4 + 2 * count * ceilDiv(R, 32) * 4 + xq + routes * intermediate * 2 + aq + routes * H * 2;
}

/// Checkpoint mode's experts (see the header) with the warp shape forced (null: ckShape); a row's bits are the same
/// for every shape. ids and wts in slot order (route's output in checkpoint mode).
pub fn expertsCk(c: *const Ctx, l: *const Layer, m: Mat, ids: Mat, wts: Mat, out: Mat, force_shape: ?u32) !void {
    const R = m.rows;
    if (!m.is(R, H, .bf16) or !ids.is(R, topk, .i32) or !wts.is(R, topk, .f32) or out.rows != R or out.cols != H or (out.dt != .bf16 and out.dt != .f32)) return error.BadMoeShape;
    if (!m.dense() or !ids.dense() or !wts.dense() or !out.dense()) return error.BadMoeShape;
    if (R == 0) return;
    const E = &l.experts;
    if (E.split > 1) return error.SplitExpertsTakeTwoHalves;
    const count: usize = E.count();
    if (count == 0 or count > 1024) return error.BadMoeShape;
    const routes = R * topk;
    const k = &c.k.moe;
    const nv = E.format == .nvfp4;
    const sh = try k.shapes(routes, nv, force_shape);
    const geo_up = ckGeo(sh[0], nv, H);
    const geo_down = ckGeo(sh[1], nv, intermediate);
    const tile = geo_up.tile;
    const items_max = maxItems(routes, count, tile);
    const f: usize = if (nv) 0 else 1;
    const nitems = try c.tmp(4);
    const members = try c.tmp(routes * 4);
    const items = try c.tmp(items_max * 3 * 4);
    // the tokens' rows quantized: codes / values, block scales, row amax (NVFP4)
    const xq = try c.tmp(if (nv) R * H / 2 else R * H);
    const xs = try c.tmp(if (nv) R * H / 16 else R * H / 32);
    const xa = if (nv) try c.tmp(R * 4) else 0;
    const act = try c.tmp(routes * intermediate * 2);
    const aq = try c.tmp(if (nv) routes * intermediate / 2 else routes * intermediate);
    const asc = try c.tmp(if (nv) routes * intermediate / 16 else routes * intermediate / 32);
    const aa = if (nv) try c.tmp(routes * 4) else 0;
    const dmat = try c.tmp(routes * H * 2);
    var prof = try Prof.init(c);
    defer prof.deinit();
    try prof.mark(c);
    try ckPlan(c, ids.ptr, R, E.lo, count, tile, members, items, nitems);
    try prof.mark(c);
    try ckQuant(c, k.ck_quant[f], nv, m.ptr, R, H, H, 0, 0, 0, 0, xq, xs, xa);
    try prof.mark(c);
    const units = items_max * (H / geo_up.cols);
    const rcp13: f32 = if (nv) 1.0 / (1536.0 * E.den13) else 1;
    const rcp2: f32 = if (nv) 1.0 / (1536.0 * E.den2) else 1;
    // up: the tokens' quantized rows (member / 8), w13, capped SwiGLU times the routing weight into act
    try ckGemm(c, k.ck_gemm[f][0][sh[0]], try gemmBlocks(geo_up, units), geo_up, xq, xs, xa, 8, H, E.w13, E.w13_scale, E.w13_amax, rcp13, wts.ptr, items, nitems, members, 2 * intermediate, intermediate, act);
    try prof.mark(c);
    // act re-quantized, only this rank's routes
    try ckQuant(c, k.ck_quant[f], nv, act, routes, intermediate, intermediate, 0, ids.ptr, E.lo, E.hi, aq, asc, aa);
    try prof.mark(c);
    // down: act rows (member), w2, bf16 route rows into d
    try ckGemm(c, k.ck_gemm[f][1][sh[1]], try gemmBlocks(geo_down, items_max * (H / geo_down.cols)), geo_down, aq, asc, aa, 1, intermediate, E.w2, E.w2_scale, E.w2_amax, rcp2, 0, items, nitems, members, H, H, dmat);
    try prof.mark(c);
    var a: cuda.launch.Args = .{};
    a.add(dmat);
    a.add(ids.ptr);
    a.add(@as(i32, @intCast(E.lo)));
    a.add(@as(i32, @intCast(E.hi)));
    a.add(@as(i32, H));
    a.add(out.ptr);
    try launch(c, if (out.dt == .f32) k.ck_fold_f32 else k.ck_fold, .{ @intCast(R), 1 }, 256, &a);
    try prof.mark(c);
    try prof.report(R, sh[0]);
}

/// The route plan a split layer keeps between its two halves (the exchange holds it): nitems, members [R * 8] and
/// items [at most maxItems * 3], i32, for forwards of at most `cap` rows.
pub fn splitPlanBytes(cap: usize) usize {
    const routes = cap * topk;
    return 256 + std.mem.alignForward(usize, routes * 4, 256) + maxItems(routes, experts_total, 16) * 3 * 4;
}

const PlanPtrs = struct { nitems: u64, members: u64, items: u64 };

fn planPtrs(plan: u64, R: usize) PlanPtrs {
    const members = plan + 256;
    return .{ .nitems = plan, .members = members, .items = members + std.mem.alignForward(usize, R * topk * 4, 256) };
}

fn splitCheck(c: *const Ctx, E: *const weights.Experts) !void {
    if (c.k.moe.mode != .checkpoint) return error.SplitExpertsNeedCheckpointMode;
    if (E.split < 2 or E.lo != 0 or E.count() != experts_total) return error.NotSplitExperts;
    if (E.n13 * E.split != 2 * intermediate or E.n2 * E.split != H or E.n13 % 128 != 0 or (E.n13 / 2) % 32 != 0 or E.n2 % 128 != 0) return error.BadMoeShape;
}

/// A split layer's first half (see the header): `h` bf16 [R * 8, I/W] (this rank's columns of act: its slot of the
/// MoE-A gather); `plan` (splitPlanBytes) is kept for `splitDown`. `force_shape` as in expertsCk, the same in both
/// halves.
pub fn splitUp(c: *const Ctx, l: *const Layer, m: Mat, ids: Mat, wts: Mat, h: u64, plan: u64, force_shape: ?u32) !void {
    const R = m.rows;
    if (!m.is(R, H, .bf16) or !ids.is(R, topk, .i32) or !wts.is(R, topk, .f32)) return error.BadMoeShape;
    if (!m.dense() or !ids.dense() or !wts.dense()) return error.BadMoeShape;
    if (R == 0) return;
    const E = &l.experts;
    try splitCheck(c, E);
    const routes = R * topk;
    const k = &c.k.moe;
    const nv = E.format == .nvfp4;
    const f: usize = if (nv) 0 else 1;
    const shape = (try k.shapes(routes, nv, force_shape))[0];
    const geo = ckGeo(shape, nv, H);
    const tile = geo.tile;
    const pp = planPtrs(plan, R);
    const xq = try c.tmp(if (nv) R * H / 2 else R * H);
    const xs = try c.tmp(if (nv) R * H / 16 else R * H / 32);
    const xa = if (nv) try c.tmp(R * 4) else 0;
    var prof = try Prof.init(c);
    defer prof.deinit();
    try prof.mark(c);
    try ckPlan(c, ids.ptr, R, 0, experts_total, tile, pp.members, pp.items, pp.nitems);
    try prof.mark(c);
    try ckQuant(c, k.ck_quant[f], nv, m.ptr, R, H, H, 0, 0, 0, 0, xq, xs, xa);
    try prof.mark(c);
    const rcp13: f32 = if (nv) 1.0 / (1536.0 * E.den13) else 1;
    try ckGemm(c, k.ck_gemm[f][0][shape], try gemmBlocks(geo, splitUnits(routes, geo, E.n13)), geo, xq, xs, xa, 8, H, E.w13, E.w13_scale, E.w13_amax, rcp13, wts.ptr, pp.items, pp.nitems, pp.members, E.n13, E.n13 / 2, h);
    try prof.mark(c);
    try prof.reportAs("split up", &.{ "plan", "quant x", "up" }, R, shape);
}

/// A split layer's second half (see the header), after MoE-A: `h_all` bf16 [W][R * 8][I/W] (every rank's h in rank
/// order), `plan` as `splitUp` left it; `y` bf16 [R, H/W], this rank's columns of the layer's output (its slot of the
/// MoE-B gather).
pub fn splitDown(c: *const Ctx, l: *const Layer, R: usize, h_all: u64, plan: u64, y: u64, force_shape: ?u32) !void {
    if (R == 0) return;
    const E = &l.experts;
    try splitCheck(c, E);
    const routes = R * topk;
    const k = &c.k.moe;
    const nv = E.format == .nvfp4;
    const f: usize = if (nv) 0 else 1;
    const shape = (try k.shapes(routes, nv, force_shape))[1];
    const geo = ckGeo(shape, nv, intermediate);
    const pp = planPtrs(plan, R);
    const seg: usize = E.n13 / 2;
    const aq = try c.tmp(if (nv) routes * intermediate / 2 else routes * intermediate);
    const asc = try c.tmp(if (nv) routes * intermediate / 16 else routes * intermediate / 32);
    const aa = if (nv) try c.tmp(routes * 4) else 0;
    const dmat = try c.tmp(routes * E.n2 * 2);
    var prof = try Prof.init(c);
    defer prof.deinit();
    try prof.mark(c);
    // every route's act row from its W segments, quantized over the whole row
    try ckQuant(c, k.ck_quant[f], nv, h_all, routes, intermediate, seg, routes * seg, 0, 0, 0, aq, asc, aa);
    try prof.mark(c);
    const rcp2: f32 = if (nv) 1.0 / (1536.0 * E.den2) else 1;
    try ckGemm(c, k.ck_gemm[f][1][shape], try gemmBlocks(geo, splitUnits(routes, geo, E.n2)), geo, aq, asc, aa, 1, intermediate, E.w2, E.w2_scale, E.w2_amax, rcp2, 0, pp.items, pp.nitems, pp.members, E.n2, E.n2, dmat);
    try prof.mark(c);
    var a: cuda.launch.Args = .{};
    a.add(dmat);
    a.add(@as(u64, 0)); // no ids: every route is this rank's
    a.add(@as(i32, 0));
    a.add(@as(i32, experts_total));
    a.add(@as(i32, @intCast(E.n2)));
    a.add(y);
    try launch(c, k.ck_fold, .{ @intCast(R), 1 }, 256, &a);
    try prof.mark(c);
    try prof.reportAs("split down", &.{ "quant act", "down", "fold" }, R, shape);
}

/// MoE-B's layout: `out` bf16 [R, H] from `parts` bf16 [W][R][H/W] (rank w's columns w * H/W ..), a plain copy.
pub fn splitColumns(c: *const Ctx, parts: u64, world: u32, out: Mat) !void {
    const R = out.rows;
    if (!out.is(R, H, .bf16) or !out.dense() or world < 2 or H % (2 * world) != 0) return error.BadMoeShape;
    if (R == 0) return;
    var a: cuda.launch.Args = .{};
    a.add(parts);
    a.add(@as(i32, @intCast(world)));
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(H / world)));
    a.add(out.ptr);
    try launch(c, c.k.moe.ck_columns, .{ @intCast(R), 1 }, 256, &a);
}

/// The most units (item, column block) a split GEMM's call can have.
fn splitUnits(routes: usize, geo: CkGeo, N: usize) usize {
    return maxItems(routes, experts_total, geo.tile) * (N / geo.cols);
}

/// A GEMM's grid: a block a unit (at most `units`; the ones past the call's own units return at once); a shape that
/// does not fit the GEMM is refused.
fn gemmBlocks(geo: CkGeo, units: usize) !u32 {
    if (!geo.fits) return error.BadMoeShape;
    return @intCast(@min(units, 1 << 20));
}


/// TF_INK2_CK_PROFILE=1: each checkpoint-mode kernel's device time (events between launches, a sync a call), printed
/// a call: plan, quantize rows, up, quantize act, down, fold. Off: nothing recorded.
const Prof = struct {
    ev: [8]?cuda.Event = @splat(null),
    n: usize = 0,

    fn init(c: *const Ctx) !Prof {
        var p: Prof = .{};
        const on = if (std.c.getenv("TF_INK2_CK_PROFILE")) |v| v[0] == '1' else false;
        if (!on) return p;
        for (&p.ev) |*e| e.* = try cuda.Event.init(c.d, true);
        return p;
    }

    fn deinit(p: *Prof) void {
        for (&p.ev) |*e| if (e.*) |*x| x.deinit();
    }

    fn mark(p: *Prof, c: *const Ctx) !void {
        if (p.ev[0] == null or p.n >= p.ev.len) return;
        try p.ev[p.n].?.record(c.stream);
        p.n += 1;
    }

    fn report(p: *Prof, R: usize, shape: u32) !void {
        return p.reportAs("ck profile", &.{ "plan", "quant x", "up", "quant act", "down", "fold" }, R, shape);
    }

    fn reportAs(p: *Prof, what: []const u8, names: []const []const u8, R: usize, shape: u32) !void {
        if (p.ev[0] == null or p.n < 2) return;
        try p.ev[p.n - 1].?.synchronize();
        std.debug.print("{s} R {d} shape {d}:", .{ what, R, shape });
        for (1..p.n) |i| std.debug.print(" {s} {d:.3}", .{ names[i - 1], try cuda.Event.elapsedMs(p.ev[i - 1].?, p.ev[i].?) });
        std.debug.print(" ms\n", .{});
    }
};

/// The route plan (tf_ink2_ck_plan: the same members and items as tf_ink2_moe_plan) for R rows' routes over experts
/// [lo, lo + count), with its row bitmaps in op scratch.
fn ckPlan(c: *const Ctx, ids: u64, R: usize, lo: u32, count: usize, tile: usize, members: u64, items: u64, nitems: u64) !void {
    const bm = try c.tmp(2 * count * ceilDiv(R, 32) * 4);
    var a: cuda.launch.Args = .{};
    a.add(ids);
    a.add(@as(i32, @intCast(R * topk)));
    a.add(@as(i32, @intCast(lo)));
    a.add(@as(i32, @intCast(count)));
    a.add(@as(i32, @intCast(tile)));
    a.add(members);
    a.add(items);
    a.add(nitems);
    a.add(bm);
    try launch(c, c.k.moe.ck_plan, .{ 1, 1 }, 1024, &a);
}

/// A quantizer's threads a row: a thread a scale block (16 values NVFP4, 32 MXFP8), whole warps, at most 512 (a block
/// a row; the kernels loop when a row has more blocks). 128 before: the same bits (the amax a max, each block alone).
fn quantThreads(K: usize, nvfp4: bool) u32 {
    const blocks = K / @as(usize, if (nvfp4) 16 else 32);
    return @intCast(@min(512, (blocks + 31) / 32 * 32));
}

/// Quantizes `rows` rows of K values; a row is in segments of `seg` values `seg_stride` values apart (seg = K: dense).
fn ckQuant(c: *const Ctx, f: cuda.Function, nv: bool, x: u64, rows: usize, K: usize, seg: usize, seg_stride: usize, ids: u64, lo: u32, hi: u32, q: u64, s: u64, amax: u64) !void {
    var a: cuda.launch.Args = .{};
    a.add(x);
    a.add(@as(i32, @intCast(K)));
    a.add(@as(i32, @intCast(seg)));
    a.add(@as(i64, @intCast(seg_stride)));
    a.add(ids);
    a.add(@as(i32, @intCast(lo)));
    a.add(@as(i32, @intCast(hi)));
    a.add(q);
    a.add(s);
    if (nv) a.add(amax);
    try launch(c, f, .{ @intCast(rows), 1 }, quantThreads(K, nv), &a);
}

/// One GEMM pass over N weight rows an expert (w13: N/2 gate/up pairs into act columns; w2: N output rows), output
/// rows `ostride` values apart.
fn ckGemm(c: *const Ctx, f: cuda.Function, blocks: u32, geo: CkGeo, x: u64, xs: u64, xa: u64, x_div: i32, K: usize, w: u64, s: u64, amax: u64, rcp: f32, gamma: u64, items: u64, nitems: u64, members: u64, N: usize, ostride: usize, out: u64) !void {
    var a: cuda.launch.Args = .{};
    a.add(x);
    a.add(xs);
    a.add(xa);
    a.add(x_div);
    a.add(@as(i32, @intCast(K)));
    a.add(w);
    a.add(s);
    a.add(amax);
    a.add(rcp);
    a.add(gamma);
    a.add(items);
    a.add(nitems);
    a.add(members);
    a.add(@as(i32, @intCast(N)));
    a.add(@as(i32, @intCast(ostride)));
    a.add(out);
    if (N % geo.cols != 0) return error.BadMoeShape;
    try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = geo.threads }, .shared = geo.smem }, c.stream, &a);
}

/// Row tiles a warp for `routes` routes over `count` experts: 4 once the mean exceeds 32 routes an expert.
pub fn rowTiles(routes: usize, count: usize) u32 {
    return if (routes >= 32 * count) 4 else 1;
}

/// The device scratch `experts` takes for R rows.
pub fn scratchBytes(R: usize, count: usize, rt: u32) usize {
    const routes = R * topk;
    const items = maxItems(routes, count, 16 * rt);
    return 4 * 256 + routes * 4 + items * 3 * 4 + routes * (intermediate + H) * 2;
}

fn maxItems(routes: usize, count: usize, tile: usize) usize {
    return @min(routes, count) + routes / tile;
}

/// `experts` with the row tiles a warp forced (null: rowTiles); a row's bits are the same for either.
pub fn expertsTiled(c: *const Ctx, l: *const Layer, m: Mat, ids: Mat, wts: Mat, out: Mat, force_rt: ?u32) !void {
    const R = m.rows;
    if (!m.is(R, H, .bf16) or !ids.is(R, topk, .i32) or !wts.is(R, topk, .f32) or out.rows != R or out.cols != H or (out.dt != .bf16 and out.dt != .f32)) return error.BadMoeShape;
    if (!m.dense() or !ids.dense() or !wts.dense() or !out.dense()) return error.BadMoeShape;
    if (R == 0) return;
    const E = &l.experts;
    const count: usize = E.count();
    if (count == 0 or count > 1024) return error.BadMoeShape;
    const routes = R * topk;
    const rt: u32 = force_rt orelse rowTiles(routes, count);
    if (rt != 1 and rt != 4) return error.BadMoeShape;
    const tile = 16 * rt;
    const items_max = maxItems(routes, count, tile);
    const k = &c.k.moe;
    const nitems = try c.tmp(4);
    const members = try c.tmp(routes * 4);
    const items = try c.tmp(items_max * 3 * 4);
    const act = try c.tmp(routes * intermediate * 2);
    const dmat = try c.tmp(routes * H * 2);
    {
        var a: cuda.launch.Args = .{};
        a.add(ids.ptr);
        a.add(@as(i32, @intCast(routes)));
        a.add(@as(i32, @intCast(E.lo)));
        a.add(@as(i32, @intCast(count)));
        a.add(@as(i32, @intCast(tile)));
        a.add(members);
        a.add(items);
        a.add(nitems);
        try launch(c, k.plan, .{ 1, 1 }, @intCast(std.mem.alignForward(usize, count, 32)), &a);
    }
    const f: usize = if (E.format == .nvfp4) 0 else 1;
    const r: usize = if (rt == 1) 0 else 1;
    const units = items_max * (H / gemm_cols);
    const blocks: u32 = @intCast(@min(ceilDiv(units, gemm_warps), 1 << 20));
    const den13: f32 = if (E.format == .nvfp4) E.den13 else 1;
    const den2: f32 = if (E.format == .nvfp4) E.den2 else 1;
    // up: m rows (member / 8), w13, capped SwiGLU into act
    try gemm(c, k.gemm[f][0][r], blocks, m.ptr, 8, H, E.w13, E.w13_scale, E.w13_amax, den13, items, nitems, members, act);
    // down: act rows (member), w2, bf16 into d
    try gemm(c, k.gemm[f][1][r], blocks, act, 1, intermediate, E.w2, E.w2_scale, E.w2_amax, den2, items, nitems, members, dmat);
    var a: cuda.launch.Args = .{};
    a.add(dmat);
    a.add(ids.ptr);
    a.add(wts.ptr);
    a.add(@as(i32, @intCast(E.lo)));
    a.add(@as(i32, @intCast(E.hi)));
    a.add(out.ptr);
    try launch(c, if (out.dt == .f32) k.combine_f32 else k.combine, .{ @intCast(R), 1 }, 256, &a);
}

fn gemm(c: *const Ctx, f: cuda.Function, blocks: u32, x: u64, x_div: i32, K: usize, w: u64, s: u64, amax: u64, den: f32, items: u64, nitems: u64, members: u64, out: u64) !void {
    var a: cuda.launch.Args = .{};
    a.add(x);
    a.add(x_div);
    a.add(@as(i32, @intCast(K)));
    a.add(w);
    a.add(s);
    a.add(amax);
    a.add(den);
    a.add(items);
    a.add(nitems);
    a.add(members);
    a.add(out);
    try launch(c, f, .{ blocks, 1 }, 32 * gemm_warps, &a);
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Runs once per loaded layer (cuda_weights.zig `Prepare`). The kernels read the experts as stored, so there is
/// nothing to repack.
pub fn prepare(w: *weights.Weights, l: *Layer) anyerror!void {
    _ = .{ w, l };
}

test "inkling2 cuda moe: row tiles, item bounds and scratch" {
    try std.testing.expectEqual(@as(u32, 1), rowTiles(8, 384));
    try std.testing.expectEqual(@as(u32, 1), rowTiles(256 * 8, 384));
    try std.testing.expectEqual(@as(u32, 4), rowTiles(2048 * 8, 384));
    try std.testing.expectEqual(@as(u32, 4), rowTiles(2048 * 8, 96));
    // every expert's last tile may be partial: at most one more item an expert than routes / tile
    try std.testing.expectEqual(@as(usize, 8), maxItems(8, 384, 16));
    try std.testing.expectEqual(@as(usize, 384 + 256), maxItems(16384, 384, 64));
    try std.testing.expect(scratchBytes(2048, 384, 4) < 240 << 20);
    try std.testing.expectEqual(@as(usize, 0), H % gemm_cols);
    try std.testing.expectEqual(@as(usize, 0), H % 64);
    try std.testing.expectEqual(@as(usize, 0), intermediate % 64);
    try std.testing.expectEqual(@as(u32, 0), ckShape(8));
    try std.testing.expectEqual(@as(u32, 0), ckShape(2 * 8));
    try std.testing.expectEqual(@as(u32, 3), ckShape(3 * 8));
    try std.testing.expectEqual(@as(u32, 3), ckShape(2048 * 8));
    try std.testing.expect(ckScratchBytes(2048, 384, 3, .nvfp4) < 270 << 20);
    try std.testing.expect(ckScratchBytes(2048, 384, 3, .mxfp8) < 280 << 20);
    try std.testing.expectEqual(@as(usize, 0), H % 128);
    // the split plan's buffer holds a forward's members and every item at the smallest tile
    try std.testing.expect(planPtrs(0, 2048).items + maxItems(2048 * 8, 384, 16) * 12 <= splitPlanBytes(2048));
    try std.testing.expect(planPtrs(0, 1).items + maxItems(8, 384, 16) * 12 <= splitPlanBytes(1));
    // the window tiles: every one divides the split and whole expert widths, its plan tile is one the split plan's
    // buffer holds, and its shared memory fits a GB10 / sm_120 block (99 KiB)
    for (0..ck_shapes) |sh| for ([2]bool{ true, false }) |nv| {
        const g = ckGeo(sh, nv, H);
        for ([_]usize{ 2 * intermediate / 4, H / 4, 2 * intermediate / 2, H / 2, 2 * intermediate, H }) |n| try std.testing.expectEqual(@as(usize, 0), n % g.cols);
        try std.testing.expect(g.tile >= 16 and g.threads <= 1024);
    };
    try std.testing.expect(!ckGeo(5, true, intermediate).fits); // 256-byte stages do not divide a 1152-byte row
    try std.testing.expect(ckGeo(5, true, H).fits and ckGeo(4, true, intermediate).fits);
    try std.testing.expectEqual(@as(u32, 2 * 80 * (384 + 48)), ckGeo(4, true, H).smem);
    try std.testing.expectEqual(@as(u32, 3 * 80 * (256 + 8)), ckGeo(5, false, H).smem);
    try std.testing.expectEqual([2]u32{ 5, 9 }, ckUnpair(ckPair(5, 9)));
    try std.testing.expectEqual([2]u32{ 3, 3 }, ckUnpair(3));
    try std.testing.expectEqual(@as(u32, 288), quantThreads(H, true));
    try std.testing.expectEqual(@as(u32, 160), quantThreads(intermediate, true));
    try std.testing.expectEqual(@as(u32, 96), quantThreads(intermediate, false));
    try std.testing.expect(ckFits(ckPair(5, 4), true) and !ckFits(5, true));
    try std.testing.expectEqual(@as(?u32, 4), (try parseShapes("4"))[1][1][1]);
    try std.testing.expectEqual(@as(?u32, null), (try parseShapes("4"))[0][0][2]);
    try std.testing.expectEqual(@as(?u32, 5), (try parseShapes("3,3,5"))[0][1][2]);
    try std.testing.expectEqual(@as(?u32, 4), (try parseShapes("3,5:4"))[1][1][1]);
    try std.testing.expectEqual(@as(?u32, 5), (try parseShapes("3,5:4"))[0][0][1]);
    try std.testing.expectError(error.BadInk2CkShape1, parseShapes("3,5:"));
    try std.testing.expectError(error.BadInk2CkShape1, parseShapes("3,99"));
    try std.testing.expectError(error.BadInk2CkShape1, parseShapes("3,3,3,3,3,3,3,3,3"));
}
