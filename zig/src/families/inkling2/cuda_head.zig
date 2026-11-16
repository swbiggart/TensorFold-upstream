//! G-head: the embedding and its norm at the input; the final norm, the unembed, the final logit softcap (30) and
//! the unpadded vocabulary at the output. Kernel image: inkling2_head.cu. Math: Transformers modeling_inkling2.py
//! (Apache-2.0), Inkling2NormedEmbedding, Inkling2TextModel.forward's end, Inkling2ForCausalLM.forward.
//!
//! `embed` (head.embed, M1; IMPLEMENTED with embed_rows.cu + norms.cu): ids i32 [R] (device) ->
//!   raw = embed[ids] (bf16 [R, 4608]); e = bf16(embed_norm * (f32(raw) * rsqrt(mean(raw^2) + 1e-6))).
//!   Oracle: `embed` (layer 0's input and block 0).
//! `finalNorm` (head.final_norm, M5; IMPLEMENTED): bf16(norm * (x * rsqrt(mean(x^2) + 1e-6))), x the final mix
//!   (fp32 [R, 4608]). Oracle: `final`.
//! `unembed` (head.unembed, M6; inkling2_head.cu tf_ink2_head_unembed_<RT>): logits bf16 [n, 201024] =
//!   bf16(y . unembed^T) (Transformers' bf16 lm_head; the checkpoint's manifest declares an fp32 output, which
//!   TokenSpeed keeps: checkpoint mode). A warp an output column, fp32 sums in dense_rows.cu's fixed order, so a
//!   row's logits do not depend on how many rows are unembedded together (cuBLASLt's do: it picks by row count).
//! `softcap` (head.softcap, M7; tf_ink2_head_softcap): out fp32 [n, 200058] = tanh(f32(l) * fp32(1/30)) * 30 over
//!   the first 200,058 columns (torch divides by the Python scalar 30.0 as a product with its fp32 reciprocal:
//!   measured, x / 30.0 equals x * fp32(1/30) on every value). Oracle: logits.full at logits.rows (the runner
//!   unembeds those rows only), topk.ids / topk.vals.
//! Every op here is a block (or a warp) a row: a row's bits do not depend on the row count.
//! Test alone: the runner with --head (and --embed for layer 0) against the oracle (inkling2_compare.py reports
//! final and logits), or --feed <capture> --dump-ops and inkling2_compare.py --ops.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");
const res = @import("cuda_res.zig");

const Ctx = ops.Ctx;
const Mat = ops.Mat;
const Weights = weights.Weights;

pub const softcap_value: f32 = 30.0;

/// The unembed's row tiles (rows sharing each weight row's loads; the sums do not depend on them).
const tiles = [_]usize{ 1, 4, 16, 32 };
const unembed_warps = 8;
const softcap_threads = 256;

/// inkling2_head.cu's functions.
pub const Kernels = struct {
    g: @import("cuda_kernels.zig").Group("head"),
    unembed_fns: [tiles.len]cuda.Function,
    softcap_fn: cuda.Function,
    argmax_fn: cuda.Function,
    embed_share_fn: cuda.Function,
    embed_pick_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var g = try @import("cuda_kernels.zig").Group("head").load(d);
        errdefer g.unload();
        var k: Kernels = .{ .g = g, .unembed_fns = undefined, .softcap_fn = try g.mod.function("tf_ink2_head_softcap"), .argmax_fn = try g.mod.function("tf_ink2_head_argmax"), .embed_share_fn = try g.mod.function("tf_ink2_head_embed_share"), .embed_pick_fn = try g.mod.function("tf_ink2_head_embed_pick") };
        inline for (tiles, 0..) |t, i| k.unembed_fns[i] = try g.mod.function(std.fmt.comptimePrint("tf_ink2_head_unembed_{d}", .{t}));
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.g.unload();
    }
};

/// M1: the normed embedding of `ids` (i32 [rows] on the device); `raw` bf16 [rows, 4608] is scratch.
pub fn embed(c: *const Ctx, w: *const Weights, ids: u64, raw: Mat, out: Mat) !void {
    if (w.embed == 0) return error.EmbedNotLoaded;
    try c.k.embed.bf16(c.stream, ids, w.embed, raw.ptr, raw.rows, raw.cols);
    try c.rms(raw, w.embed_norm, out, res.eps);
}

/// M1 by vocabulary rows (cuda_embed.zig): `out` bf16 [rows, H] = the table rows of `ids` (i32 [rows] on the device)
/// that this rank holds (entries w.embed_rows; its `w.embed` holds those rows only), zeros for the others. Bit copies.
pub fn embedShare(c: *const Ctx, w: *const Weights, ids: u64, out: Mat) !void {
    if (w.embed == 0 or w.embed_rows.len() == 0) return error.EmbedNotLoaded;
    if (out.dt != .bf16 or !out.dense() or out.cols % 8 != 0 or out.ptr % 16 != 0) return error.BadHeadShape;
    if (out.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(ids);
    a.add(w.embed);
    a.add(out.ptr);
    a.add(@as(i32, @intCast(out.cols)));
    a.add(@as(i32, @intCast(w.embed_rows.lo)));
    a.add(@as(i32, @intCast(w.embed_rows.hi)));
    try cuda.launch.launch(c.k.head.embed_share_fn, .{ .grid = .{ .x = @intCast(out.rows), .y = @intCast((out.cols / 8 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, c.stream, &a);
}

/// After the shares' all-gather (`parts` bf16 [world][rows][H], rank order): out[r] = the slot of the rank holding
/// ids[r], the owners being ranks `first` .. first + owners - 1 by canon's rule over `vocab` (cuda_embed.zig). Bit
/// copies; then `embedNorm`.
pub fn embedPick(c: *const Ctx, ids: u64, parts: u64, first: u32, owners: u32, vocab: u32, out: Mat) !void {
    if (out.dt != .bf16 or !out.dense() or out.cols % 8 != 0 or out.ptr % 16 != 0 or parts % 16 != 0 or owners == 0 or vocab < owners) return error.BadHeadShape;
    if (out.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(ids);
    a.add(parts);
    a.add(out.ptr);
    a.add(@as(i32, @intCast(out.cols)));
    a.add(@as(i32, @intCast(out.rows)));
    a.add(@as(i32, @intCast(first)));
    a.add(@as(i32, @intCast(owners)));
    a.add(@as(i32, @intCast(vocab)));
    try cuda.launch.launch(c.k.head.embed_pick_fn, .{ .grid = .{ .x = @intCast(out.rows), .y = @intCast((out.cols / 8 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, c.stream, &a);
}

/// M1's norm of looked-up rows `raw` (bf16 [rows, H]) into `out`: the second half of `embed`, the same kernel.
pub fn embedNorm(c: *const Ctx, w: *const Weights, raw: Mat, out: Mat) !void {
    if (w.embed_norm == 0) return error.EmbedNotLoaded;
    try c.rms(raw, w.embed_norm, out, res.eps);
}

/// M5: the final norm of the final mix.
pub fn finalNorm(c: *const Ctx, w: *const Weights, x: Mat, out: Mat) !void {
    try c.rms(x, w.norm, out, res.eps);
}

/// M6: bf16 logits of the rank's share of the padded vocabulary, entries w.vocab_rows (all of them at W = 1; rows
/// of y any ld; out [n, w.vocab_rows.len()] any ld). A column's sum does not depend on which rows a rank holds, so
/// the shares gathered in rank order are the single head's logits bit for bit.
pub fn unembed(c: *const Ctx, w: *const Weights, y: Mat, out: Mat) !void {
    if (w.unembed == 0) return error.HeadNotLoaded;
    const n = y.rows;
    const V: usize = w.vocab_rows.len();
    if (y.dt != .bf16 or out.dt != .bf16 or out.rows != n or out.cols != V or y.cols % 8 != 0 or y.ld % 8 != 0) return error.BadHeadShape;
    if (y.ptr % 16 != 0 or w.unembed % 16 != 0) return error.Misaligned;
    if (n == 0) return;
    const i: usize = if (n == 1) 0 else if (n <= 4) 1 else if (n <= 16) 2 else 3;
    var a: cuda.Args = .{};
    a.add(y.ptr);
    a.add(@as(i64, @intCast(y.ld)));
    a.add(w.unembed);
    a.add(out.ptr);
    a.add(@as(i64, @intCast(out.ld)));
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(V)));
    a.add(@as(i32, @intCast(y.cols)));
    const grid: cuda.Dim3 = .{ .x = @intCast((V + unembed_warps - 1) / unembed_warps), .y = @intCast((n + tiles[i] - 1) / tiles[i]), .z = 1 };
    try cuda.launch.launch(c.k.head.unembed_fns[i], .{ .grid = grid, .block = .{ .x = unembed_warps * 32 } }, c.stream, &a);
}

/// M7: softcapped fp32 logits of the unpadded vocabulary (see the header): out [n, cols] from l [n, >= cols].
pub fn softcap(c: *const Ctx, l: Mat, out: Mat) !void {
    const n = l.rows;
    if (l.dt != .bf16 or out.dt != .f32 or out.rows != n or out.cols > l.cols or out.cols == 0) return error.BadHeadShape;
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(l.ptr);
    a.add(@as(i64, @intCast(l.ld)));
    a.add(out.ptr);
    a.add(@as(i64, @intCast(out.ld)));
    a.add(@as(i32, @intCast(out.cols)));
    a.add(softcap_value);
    const grid: cuda.Dim3 = .{ .x = @intCast((out.cols + softcap_threads - 1) / softcap_threads), .y = @intCast(n), .z = 1 };
    try cuda.launch.launch(c.k.head.softcap_fn, .{ .grid = grid, .block = .{ .x = softcap_threads } }, c.stream, &a);
}

/// M7 over the ranks' shares as gathered: `parts` holds `world` blocks of bf16 [n, share], block q being rank q's
/// logits for entries q * share .. (an all-gather's rank order; canon's rule splits the padded vocabulary evenly);
/// out fp32 [n, unpadded] gets the softcap of every entry below `out.cols`. One row's blocks are one contiguous row.
pub fn softcapParts(c: *const Ctx, parts: u64, world: u32, share: usize, out: Mat) !void {
    const n = out.rows;
    if (out.dt != .f32 or share == 0 or share * world < out.cols) return error.BadHeadShape;
    if (n == 1) return softcap(c, Mat.of(parts, 1, share * world, .bf16), out);
    var q: usize = 0;
    while (q < world and q * share < out.cols) : (q += 1) {
        var o = out;
        o.ptr = out.ptr + q * share * 4;
        o.cols = @min(share, out.cols - q * share);
        try softcap(c, Mat.of(parts + q * n * share * 2, n, share, .bf16), o);
    }
}

/// Greedy rows on the device (B2 C4): out (u32 [x.rows], device) = each row's lanes/gpu_rule.zig argmax of x (fp32
/// [n, cols], the softcapped logits): the first index of the largest value; a NaN at column 0 wins, a NaN elsewhere
/// never does, all -inf gives 0. A block a row (tf_ink2_head_argmax).
pub fn argmax(c: *const Ctx, x: Mat, out: u64) !void {
    if (x.dt != .f32 or x.cols == 0 or x.cols > std.math.maxInt(i32)) return error.BadHeadShape;
    if (x.rows == 0) return;
    var a: cuda.Args = .{};
    a.add(x.ptr);
    a.add(@as(i64, @intCast(x.ld)));
    a.add(@as(i32, @intCast(x.cols)));
    a.add(out);
    try cuda.launch.launch(c.k.head.argmax_fn, .{ .grid = .{ .x = @intCast(x.rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, c.stream, &a);
}

/// Runs once per loaded layer (cuda_weights.zig `Prepare`).
pub fn prepare(w: *Weights, l: *weights.Layer) anyerror!void {
    _ = .{ w, l };
}
