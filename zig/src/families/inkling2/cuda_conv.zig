//! G-conv: the four short depthwise 4-tap causal convolutions of a layer: K and V (static taps in sliding layers;
//! static plus per-token taps in global layers), after attention and after the MLP. Kernel image: inkling2_conv.cu.
//! Math: Transformers modeling_inkling2.py (Apache-2.0), Inkling2ShortConvolution and
//! Inkling2DynamicShortConvolution; their conv weights are kept in fp32 (stored bf16, exact).
//!
//! Every convolution reads raw bf16 input rows x_t (t = pos0 .. pos0 + R - 1) and, for t - j < pos0, the stream's
//! state: bf16 [3, C], row 0 = the input at pos0 - 3, row 2 = at pos0 - 1 (zeros before position 0). Entry points
//! only read the state; the scaffold commits it afterwards (cuda_state.zig `convCommit`: the last 3 input rows).
//! All arithmetic fp32, one bf16 rounding of the output.
//!
//! Static form (sliding K and V: C = hk*128, taps `l.k_conv` / `l.v_conv` [C, 4]; post-attention and post-MLP:
//!   C = 4608, taps `l.attn_conv` / `l.mlp_conv` [C, 4]; tap 3 multiplies the current row):
//!   y_t[c] = (x_t[c] + sum_{j=0..3} w[c][j] * x_{t-3+j}[c]) / sqrt(5)
//! Dynamic form (global K and V: C = hk*128, static taps `l.k_conv` / `l.v_conv` [4, C] with lag 0 = current row,
//!   per-token taps dk / dv bf16 [R, hk*4] from attn.project, row t, column head*4 + lag, head = c / 128):
//!   y_t[c] = (x_t[c] + sum_{lag=0..3} (static[lag][c] + f32(d_t[c/128*4 + lag])) * x_{t-lag}[c]) / sqrt(9)
//! The division is torch's on CUDA: a tensor divided by a Python scalar is multiplied by the scalar's fp32
//! reciprocal (measured: 1.4 M of 4.2 M fp32 quotients by 3 differ from true division), so the kernels multiply by
//! `inv(5)` / `inv(9)`. The sums are torch's in fp32 (inkling2_conv.cu): the static taps oldest row first (bf16 x
//! bf16 products are exact in fp32, so only the order counts; cuDNN's conv1d matched at rows 1 to 4000), the dynamic
//! lags as (p0 + p2) + (p1 + p3), the order torch's sum(-1) takes at every row count tried (1 to 4000).
//!
//! `kv` (conv.kv, L4): kc = conv(k), vc = conv(v), both bf16 [R, hk*128]; global layers use dk, dv.
//! `post` (conv.attn L16 with `l.attn_conv`, conv.mlp L21 with `l.mlp_conv`): out bf16 [R, 4608].
//! A row's bits depend only on its own four input rows and taps (one thread walks rows of two channels), so they
//! do not depend on the forward's row count or on how a prompt is chunked.
//! Oracle tensors: op.<L>.conv.<kv|attn|mlp>.out.* (inkling2_op_ref.py capture). Test alone: the runner with
//! --feed <capture> --dump-ops, then inkling2_compare.py --ops; a chunked run (--chunk) checks the state seam, and
//! inkling2_compare.py with another run's dump as --ops compares two chunkings' op outputs bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");

const Ctx = ops.Ctx;
const Mat = ops.Mat;
const Layer = weights.Layer;

/// Rows of state a stream keeps a convolution.
pub const state_rows = 3;

/// inkling2_conv.cu's threads a block (two channels a thread) and rows a thread walks.
const threads = 128;
const rows_a_thread = 16;

/// fp32(1) / fp32(sqrt(n)): what torch multiplies by for `t / math.sqrt(n)` on a CUDA tensor.
pub fn inv(n: f64) f32 {
    const s: f32 = @floatCast(@sqrt(n));
    return 1.0 / s;
}

/// inkling2_conv.cu's functions.
pub const Kernels = struct {
    group: Group,
    static_fn: cuda.Function,
    dynamic_fn: cuda.Function,
    commit_fn: cuda.Function,
    /// A verify window's commit keeping its inputs, and `keep`'s rollback of a whole slot (cuda_state.zig).
    commit_spec_fn: cuda.Function,
    keep_fn: cuda.Function,

    const Group = @import("cuda_kernels.zig").Group("conv");

    pub fn load(d: *const cuda.Driver) !Kernels {
        var g = try Group.load(d);
        errdefer g.unload();
        return .{
            .group = g,
            .static_fn = try g.mod.function("tf_ink2_conv_static"),
            .dynamic_fn = try g.mod.function("tf_ink2_conv_dynamic"),
            .commit_fn = try g.mod.function("tf_ink2_conv_commit"),
            .commit_spec_fn = try g.mod.function("tf_ink2_conv_commit_spec"),
            .keep_fn = try g.mod.function("tf_ink2_conv_keep"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.group.unload();
        k.* = undefined;
    }
};

/// One convolution's rows: x [R, C] in, y [R, C] out (bf16, any even ld), the stream's state [3, C], static taps,
/// and in the dynamic form the per-token taps d [R, C/128*4].
const Conv = struct {
    x: Mat,
    y: Mat,
    state: u64,
    taps: u64,
    d: ?Mat = null,
};

fn bf16Rows(m: Mat, rows: usize, cols: usize) bool {
    return m.ptr != 0 and m.is(rows, cols, .bf16) and m.ld >= cols and m.ld % 2 == 0 and m.ptr % 4 == 0;
}

fn run(c: *const Ctx, v: Conv) !void {
    const R = v.x.rows;
    const C = v.x.cols;
    if (C == 0 or C % 2 != 0 or !bf16Rows(v.x, R, C) or !bf16Rows(v.y, R, C)) return error.BadConvShape;
    if (v.state == 0 or v.state % 4 != 0 or v.taps == 0) return error.BadConvShape;
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(v.x.ptr);
    a.add(@as(i64, @intCast(v.x.ld)));
    a.add(v.state);
    a.add(v.taps);
    const f = if (v.d) |d| blk: {
        if (C % weights.head_dim != 0 or v.taps % 4 != 0) return error.BadConvShape;
        const width = C / weights.head_dim * weights.taps;
        if (d.ptr == 0 or !d.is(R, width, .bf16) or d.ld < width) return error.BadConvShape;
        a.add(d.ptr);
        a.add(@as(i64, @intCast(d.ld)));
        break :blk c.k.conv.dynamic_fn;
    } else c.k.conv.static_fn;
    a.add(v.y.ptr);
    a.add(@as(i64, @intCast(v.y.ld)));
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(C)));
    a.add(inv(if (v.d != null) 2 * weights.taps + 1 else weights.taps + 1));
    const grid: cuda.Dim3 = .{
        .x = @intCast(std.math.divCeil(usize, C, 2 * threads) catch unreachable),
        .y = @intCast(std.math.divCeil(usize, R, rows_a_thread) catch unreachable),
        .z = 1,
    };
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, c.stream, &a);
}

/// L4: the K and V convolutions (see the header). dk, dv are empty in sliding layers.
pub fn kv(c: *const Ctx, l: *const Layer, k: Mat, v: Mat, dk: Mat, dv: Mat, state_k: u64, state_v: u64, kc: Mat, vc: Mat) !void {
    const C = @as(usize, l.hk) * weights.head_dim;
    if (k.cols != C or v.cols != C) return error.BadConvShape;
    try run(c, .{ .x = k, .y = kc, .state = state_k, .taps = l.k_conv, .d = if (l.global) dk else null });
    try run(c, .{ .x = v, .y = vc, .state = state_v, .taps = l.v_conv, .d = if (l.global) dv else null });
}

/// L16 / L21: a static 4-tap convolution of 4608 channels with `taps` [4608, 4] (see the header).
pub fn post(c: *const Ctx, taps: u64, x: Mat, state: u64, out: Mat) !void {
    if (x.cols != weights.hidden) return error.BadConvShape;
    try run(c, .{ .x = x, .y = out, .state = state, .taps = taps });
}

/// Runs once per loaded layer (cuda_weights.zig `Prepare`): the taps are used as stored.
pub fn prepare(w: *weights.Weights, l: *Layer) anyerror!void {
    _ = .{ w, l };
}

test "inkling2 cuda conv: torch's reciprocals of sqrt(5) and sqrt(9)" {
    // fp32(sqrt(5)) = 2.236068 (0x400F1BBD); its fp32 reciprocal 0x3EE4F92E; 1 / fp32(3) = 0x3EAAAAAB
    try std.testing.expectEqual(@as(u32, 0x3EE4F92E), @as(u32, @bitCast(inv(5))));
    try std.testing.expectEqual(@as(u32, 0x3EAAAAAB), @as(u32, @bitCast(inv(9))));
}
