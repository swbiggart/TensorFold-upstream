//! G-res: the block attention residual (6-layer blocks) and the norms around it. Kernel image: inkling2_res.cu.
//!
//! Math (Transformers modeling_inkling2.py, Apache-2.0: Inkling2BlockAttentionResidual, the decoder layers'
//! forward, Inkling2RMSNorm). H = 4608, R = the forward's rows, every row independent.
//!
//! `mix` (layer ops res.attn_input, res.mlp_input; model op res.final_input):
//!   sources s_0 .. s_{nb-1} = blocks (bf16 [R, H] each), s_nb = cand (bf16 or fp32 [R, H]); all taken as fp32.
//!   n_i = s_i * rsqrt(mean_H(s_i^2) + 1e-6)                       (Inkling2UnweightedRMSNorm on fp32: no gain)
//!   for query j (queries fp32 [nq, H]):  score_i = (q_j . n_i) * H^-0.5,  p = softmax_i(score)
//!   out[j][r] = sum_i p_i * n_i                                    (fp32 [R, H]; torch SDPA, fp32 throughout)
//!   attn_input of layer L > 0: blocks = the completed blocks below L, cand = h_in (bf16), nq = 1 (sliding) or 3
//!     (global: out[0] feeds the q norm, out[1] the k norm, out[2] the v norm). Layer 0 has none (x = h_in).
//!   mlp_input of layer L: blocks after the boundary push (see `partial`), cand = partial (fp32), nq = 1.
//!   final_input: the 8 blocks (inputs of layers 0, 6, ..., 42), cand = h_out(47) (bf16), nq = 1.
//!   Oracle tensors: op.<L>.res.attn_input.out.x0/x1/x2, op.<L>.res.mlp_input.out.x, op.model.res.final_input.out.x
//!   (inkling2_op_ref.py capture); the layer's h.<L> and final at the end of the chain.
//!   Kernel (tf_ink2_res_mix_bf16 / _f32, by the candidate's dtype): a CTA a row, 576 threads of 8 columns. Pass 1
//!   per source: the sum of squares (square, then add), mean as a product with fp32(1/H), rsqrtf, n_i elementwise,
//!   the nq dots with fp32 FMAs; pass 2 rereads the sources and accumulates p_ji * n_i in source order. Every sum
//!   is a thread's 8 columns in order, a warp xor tree, then the 18 warps in order: a row's bits do not depend on R.
//!   Against torch (SDPA's own order): rel 1.4e-7 to 3.2e-7, max abs <= 2.7e-5, 99.97% equal after a bf16 round.
//!
//! `norm` (res.attn_norm, res.q_norm / k_norm / v_norm, res.mlp_norm, head.final_norm, the embed norm):
//!   y = bf16(gain * (x * rsqrt(mean_H(x^2) + 1e-6))), x bf16 or fp32, gain fp32 [H] (stored bf16, converted
//!   exactly). Implemented on norms.cu's tf_rms_rows (the scaffold's slow path: fp32 math, gain after; a block a
//!   row, its bits independent of the row count).
//!
//! `partial` (res.partial, decoder layer: `residual = hidden_states = conv_out.float()`, then on a non-boundary
//!   layer `block_residual.float() + hidden_states`):
//!   out fp32 [R, H] = boundary ? f32(a) : f32(h_in) + f32(a)     (a = the post-attention conv's bf16 output)
//!   On a boundary layer (L % 6 == 0) the scaffold pushes h_in onto the blocks before mlp_input (a device copy).
//!
//! `output` (res.output, `(residual + hidden_states).to(bf16)`):
//!   out bf16 [R, H] = bf16(partial + f32(mconv))                 (mconv = the post-MLP conv's bf16 output)
//!   `partial` and `output` are elementwise kernels (tf_ink2_res_partial / _output), bit-equal to torch.
//!
//! State: none across tokens; the blocks live with the forward's rows (cuda_state.zig `Act.blocks`).
//! Test alone: `tf-inkling2-cuda ... --feed <capture> --dump-ops` then `inkling2_compare.py --ops`, or
//! `inkling2_op_ref.py check` on a --dump-ops dump (see docs in tools/zig/inkling2_op_ref.py).
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");

const Ctx = ops.Ctx;
const Mat = ops.Mat;
const Blocks = ops.Blocks;

pub const eps: f32 = 1e-6;
/// inkling2_res.cu's row width and threads a row (8 columns a thread).
pub const width = 4608;
const mix_threads = width / 8;
const mix_most_blocks = 8;
const mix_most_queries = 3;
const elem_threads = 256;

/// inkling2_res.cu's functions.
pub const Kernels = struct {
    g: @import("cuda_kernels.zig").Group("res"),
    mix_bf16: cuda.Function,
    mix_f32: cuda.Function,
    partial_fn: cuda.Function,
    output_fn: cuda.Function,
    rank_sum_fn: cuda.Function,
    rank_sum_rows_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var g = try @import("cuda_kernels.zig").Group("res").load(d);
        errdefer g.unload();
        return .{
            .g = g,
            .mix_bf16 = try g.mod.function("tf_ink2_res_mix_bf16"),
            .mix_f32 = try g.mod.function("tf_ink2_res_mix_f32"),
            .partial_fn = try g.mod.function("tf_ink2_res_partial"),
            .output_fn = try g.mod.function("tf_ink2_res_output"),
            .rank_sum_fn = try g.mod.function("tf_ink2_res_rank_sum"),
            .rank_sum_rows_fn = try g.mod.function("tf_ink2_res_rank_sum_rows"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.g.unload();
    }
};

/// out[j] fp32 [R, H] for j < nq; see the header. One CTA a row: a row's bits do not depend on R.
pub fn mix(c: *const Ctx, blocks: Blocks, cand: Mat, queries: u64, nq: usize, out: []const Mat) !void {
    const R = cand.rows;
    if (nq == 0 or nq > mix_most_queries or out.len < nq or queries == 0) return error.BadMixShape;
    if (blocks.count > mix_most_blocks or blocks.hidden != width or (blocks.count > 0 and blocks.rows != R)) return error.BadMixShape;
    if (blocks.stride % 2 != 0 or blocks.ptr % 16 != 0 or blocks.stride % 16 != 0) return error.Misaligned;
    if (cand.cols != width or !cand.dense() or (cand.dt != .bf16 and cand.dt != .f32) or cand.ptr % 16 != 0) return error.BadMixShape;
    for (out[0..nq]) |o| if (!o.is(R, width, .f32) or !o.dense() or o.ptr % 16 != 0) return error.BadMixShape;
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(blocks.ptr);
    a.add(@as(i64, @intCast(blocks.stride / 2)));
    a.add(@as(i32, @intCast(blocks.count)));
    a.add(cand.ptr);
    a.add(queries);
    a.add(@as(i32, @intCast(nq)));
    for (0..mix_most_queries) |j| a.add(if (j < nq) out[j].ptr else @as(u64, 0));
    a.add(eps);
    a.add(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, width)))));
    const f = if (cand.dt == .f32) c.k.res.mix_f32 else c.k.res.mix_bf16;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(R) }, .block = .{ .x = mix_threads } }, c.stream, &a);
}

/// y bf16 = gain-weighted RMS norm of x (bf16 or fp32).
pub fn norm(c: *const Ctx, x: Mat, gain: u64, y: Mat) !void {
    try c.rms(x, gain, y, eps);
}

/// out fp32 = boundary ? f32(a) : f32(h_in) + f32(a).
pub fn partial(c: *const Ctx, h_in: Mat, a: Mat, boundary: bool, out: Mat) !void {
    const R = a.rows;
    if (!a.is(R, width, .bf16) or !h_in.is(R, width, .bf16) or !out.is(R, width, .f32)) return error.BadResidualShape;
    if (!a.dense() or !h_in.dense() or !out.dense()) return error.BadResidualShape;
    try elementwise(c, c.k.res.partial_fn, .{ h_in.ptr, a.ptr, out.ptr, @as(i32, @intFromBool(boundary)) }, R * width);
}

/// out bf16 = bf16(partial + f32(mconv)).
pub fn output(c: *const Ctx, partial_: Mat, mconv: Mat, out: Mat) !void {
    const R = partial_.rows;
    if (!partial_.is(R, width, .f32) or !mconv.is(R, width, .bf16) or !out.is(R, width, .bf16)) return error.BadResidualShape;
    if (!partial_.dense() or !mconv.dense() or !out.dense()) return error.BadResidualShape;
    try elementwise(c, c.k.res.output_fn, .{ partial_.ptr, mconv.ptr, out.ptr }, R * width);
}

/// W > 1: out bf16 [R, H] = bf16(sum over ranks in rank order of parts[r]), parts fp32 [world][R][H] at `parts`.
pub fn rankSum(c: *const Ctx, parts: u64, world: usize, out: Mat) !void {
    const R = out.rows;
    if (!out.is(R, width, .bf16) or !out.dense() or world == 0) return error.BadResidualShape;
    try elementwise(c, c.k.res.rank_sum_fn, .{ parts, @as(i32, @intCast(world)), out.ptr }, R * width);
}

/// rankSum over some rows: out bf16 [n, H] = bf16(sum over ranks in rank order of the fp32 [n, H] rows at
/// `parts + r * stride` bytes); the same kernel arithmetic as rankSum, so a row's bits are the whole sum's.
pub fn rankSumRows(c: *const Ctx, parts: u64, world: usize, stride: usize, out: Mat) !void {
    const R = out.rows;
    if (!out.is(R, width, .bf16) or !out.dense() or world == 0 or stride % 4 != 0) return error.BadResidualShape;
    try elementwise(c, c.k.res.rank_sum_rows_fn, .{ parts, @as(i32, @intCast(world)), @as(i64, @intCast(stride / 4)), out.ptr }, R * width);
}

/// An 8-elements-a-thread kernel over n elements: `args` then n.
fn elementwise(c: *const Ctx, f: cuda.Function, args: anytype, n: usize) !void {
    if (n == 0) return;
    if (n % 8 != 0) return error.BadResidualShape;
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    a.add(@as(i64, @intCast(n)));
    const threads = n / 8;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((threads + elem_threads - 1) / elem_threads) }, .block = .{ .x = elem_threads } }, c.stream, &a);
}

/// Runs once per loaded layer (cuda_weights.zig `Prepare`): repack what the group's kernels want.
pub fn prepare(w: *@import("cuda_weights.zig").Weights, l: *@import("cuda_weights.zig").Layer) anyerror!void {
    _ = .{ w, l };
}
