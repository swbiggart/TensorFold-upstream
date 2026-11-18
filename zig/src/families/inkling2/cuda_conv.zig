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
pub const rows_a_thread = 16;

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
    /// conv.kv and the K and V commits as one launch (kvCommit)
    kv_commit_fn: cuda.Function,
    /// The same for every segment of a batched forward in one launch (kvCommitSegs, postCommitSegs)
    kv_commit_segs_fn: cuda.Function,

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
            .kv_commit_fn = try g.mod.function("tf_ink2_conv_kv_commit"),
            .kv_commit_segs_fn = try g.mod.function("tf_ink2_conv_kv_commit_segs"),
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

/// inkling2_conv.cu KvCommit: K's and V's convolution and commit (index 0 K, 1 V).
const KvCommitArgs = extern struct {
    x: [2]u64,
    ldx: [2]i64,
    s: [2]u64,
    w: [2]u64,
    d: [2]u64 = .{ 0, 0 },
    ldd: [2]i64 = .{ 0, 0 },
    y: [2]u64,
    ldy: [2]i64,
    h: [2]u64 = .{ 0, 0 },
};

/// `kv` and then the K and V state commits (cuda_state.zig convCommit, or convCommitSpec where `hist_k` / `hist_v`
/// are not 0) as one launch, for a forward of up to `rows_a_thread` rows (one thread walks all of its channels'
/// rows, so it can commit them after it read them): the same output bits and the same state and history bytes.
pub fn kvCommit(c: *const Ctx, l: *const Layer, k: Mat, v: Mat, dk: Mat, dv: Mat, state_k: u64, state_v: u64, hist_k: u64, hist_v: u64, kc: Mat, vc: Mat) !void {
    const R = k.rows;
    const p = try kvArgs(l, k, v, dk, dv, state_k, state_v, hist_k, hist_v, kc, vc);
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(p);
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(k.cols)));
    a.add(inv(if (l.global) 2 * weights.taps + 1 else weights.taps + 1));
    const grid: cuda.Dim3 = .{ .x = @intCast(std.math.divCeil(usize, k.cols, 2 * threads) catch unreachable), .y = 1, .z = 2 };
    try cuda.launch.launch(c.k.conv.kv_commit_fn, .{ .grid = grid, .block = .{ .x = threads } }, c.stream, &a);
}

/// kvCommit's checks and arguments.
fn kvArgs(l: *const Layer, k: Mat, v: Mat, dk: Mat, dv: Mat, state_k: u64, state_v: u64, hist_k: u64, hist_v: u64, kc: Mat, vc: Mat) !KvCommitArgs {
    const C = @as(usize, l.hk) * weights.head_dim;
    const R = k.rows;
    if (k.cols != C or v.cols != C or v.rows != R or R > rows_a_thread) return error.BadConvShape;
    if ((hist_k != 0 or hist_v != 0) and R > 8) return error.BadConvShape;
    for ([_]Mat{ k, v, kc, vc }) |m| if (!bf16Rows(m, R, C)) return error.BadConvShape;
    for ([_]u64{ state_k, state_v }) |st| if (st == 0 or st % 4 != 0) return error.BadConvShape;
    if (C % 2 != 0 or l.k_conv == 0 or l.v_conv == 0) return error.BadConvShape;
    var p: KvCommitArgs = .{
        .x = .{ k.ptr, v.ptr },
        .ldx = .{ @intCast(k.ld), @intCast(v.ld) },
        .s = .{ state_k, state_v },
        .w = .{ l.k_conv, l.v_conv },
        .y = .{ kc.ptr, vc.ptr },
        .ldy = .{ @intCast(kc.ld), @intCast(vc.ld) },
        .h = .{ hist_k, hist_v },
    };
    if (l.global) {
        const width = C / weights.head_dim * weights.taps;
        for ([_]Mat{ dk, dv }, 0..) |d, i| {
            if (d.ptr == 0 or !d.is(R, width, .bf16) or d.ld < width) return error.BadConvShape;
            p.d[i] = d.ptr;
            p.ldd[i] = @intCast(d.ld);
        }
    }
    return p;
}

/// `post` and then its state commit (convCommit, or convCommitSpec where `hist` is not 0) as one launch, for a
/// forward of up to `rows_a_thread` rows (tf_ink2_conv_kv_commit with one convolution): the same bits and bytes.
pub fn postCommit(c: *const Ctx, taps: u64, x: Mat, state: u64, hist: u64, out: Mat) !void {
    const C = weights.hidden;
    const R = x.rows;
    const p = try postArgs(taps, x, state, hist, out);
    if (R == 0) return;
    var a: cuda.Args = .{};
    a.add(p);
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(C)));
    a.add(inv(weights.taps + 1));
    const grid: cuda.Dim3 = .{ .x = @intCast(std.math.divCeil(usize, C, 2 * threads) catch unreachable), .y = 1, .z = 1 };
    try cuda.launch.launch(c.k.conv.kv_commit_fn, .{ .grid = grid, .block = .{ .x = threads } }, c.stream, &a);
}

/// postCommit's checks and arguments (index 0).
fn postArgs(taps: u64, x: Mat, state: u64, hist: u64, out: Mat) !KvCommitArgs {
    const C = weights.hidden;
    const R = x.rows;
    if (x.cols != C or R > rows_a_thread or (hist != 0 and R > 8)) return error.BadConvShape;
    if (!bf16Rows(x, R, C) or !bf16Rows(out, R, C) or state == 0 or state % 4 != 0 or taps == 0) return error.BadConvShape;
    return .{
        .x = .{ x.ptr, 0 },
        .ldx = .{ @intCast(x.ld), 0 },
        .s = .{ state, 0 },
        .w = .{ taps, 0 },
        .y = .{ out.ptr, 0 },
        .ldy = .{ @intCast(out.ld), 0 },
        .h = .{ hist, 0 },
    };
}

/// Segments a segs launch carries at most (inkling2_conv.cu MS; cuda_forward.zig max_segs).
pub const max_segs = 8;

/// One stream's rows of a batched forward for kvCommitSegs / postCommitSegs: rows row0 .. row0 + rows of the
/// forward's matrices, the stream's state (index 0 K, 1 V; postCommitSegs uses index 0) and its history, or 0.
pub const Seg = struct {
    row0: usize,
    rows: usize,
    state: [2]u64,
    hist: [2]u64 = .{ 0, 0 },
};

/// inkling2_conv.cu KvCommitSegs: the shared matrices of KvCommitArgs, then each segment's rows, states and histories.
const KvCommitSegsArgs = extern struct {
    x: [2]u64,
    ldx: [2]i64,
    w: [2]u64,
    d: [2]u64 = .{ 0, 0 },
    ldd: [2]i64 = .{ 0, 0 },
    y: [2]u64,
    ldy: [2]i64,
    s: [max_segs][2]u64 = @splat(.{ 0, 0 }),
    h: [max_segs][2]u64 = @splat(.{ 0, 0 }),
    row0: [max_segs]i32 = @splat(0),
    rows: [max_segs]i32 = @splat(0),

    /// Segment j's entry from the arguments kvCommit or postCommit would take for its rows alone.
    fn put(t: *KvCommitSegsArgs, j: usize, g: Seg, q: KvCommitArgs) void {
        t.s[j] = q.s;
        t.h[j] = q.h;
        t.row0[j] = @intCast(g.row0);
        t.rows[j] = @intCast(g.rows);
    }
};

/// The segments in row order, not overlapping, each inside `R` rows, at most max_segs; the empty ones left out.
fn segsFit(segs: []const Seg, R: usize) !void {
    if (segs.len > max_segs) return error.BadConvShape;
    var end: usize = 0;
    for (segs) |g| {
        if (g.rows == 0) continue;
        if (g.row0 < end or g.row0 + g.rows > R) return error.BadConvShape;
        end = g.row0 + g.rows;
    }
}

/// kvCommit for every segment of a batched forward in one launch (blockIdx.y a segment): k, v, dk, dv, kc and vc
/// are the whole forward's rows, and each segment gets kvCommit's checks on its own rows, so each stream's output
/// and state bytes are those of its own kvCommit. Segments never share a state (a slot runs once a forward) and
/// their rows are disjoint, so the one launch does what the launches a segment do, in any order.
pub fn kvCommitSegs(c: *const Ctx, l: *const Layer, k: Mat, v: Mat, dk: Mat, dv: Mat, kc: Mat, vc: Mat, segs: []const Seg) !void {
    const R = @min(k.rows, v.rows, kc.rows, vc.rows);
    try segsFit(segs, if (l.global) @min(R, dk.rows, dv.rows) else R);
    var t: KvCommitSegsArgs = .{
        .x = .{ k.ptr, v.ptr },
        .ldx = .{ @intCast(k.ld), @intCast(v.ld) },
        .w = .{ l.k_conv, l.v_conv },
        .y = .{ kc.ptr, vc.ptr },
        .ldy = .{ @intCast(kc.ld), @intCast(vc.ld) },
    };
    if (l.global) {
        t.d = .{ dk.ptr, dv.ptr };
        t.ldd = .{ @intCast(dk.ld), @intCast(dv.ld) };
    }
    var n: usize = 0;
    for (segs) |g| {
        if (g.rows == 0) continue;
        const r = g.row0;
        const sdk = if (l.global) dk.slice(r, g.rows) else dk;
        const sdv = if (l.global) dv.slice(r, g.rows) else dv;
        t.put(n, g, try kvArgs(l, k.slice(r, g.rows), v.slice(r, g.rows), sdk, sdv, g.state[0], g.state[1], g.hist[0], g.hist[1], kc.slice(r, g.rows), vc.slice(r, g.rows)));
        n += 1;
    }
    if (n == 0) return;
    const C = k.cols;
    var a: cuda.Args = .{};
    a.add(t);
    a.add(@as(i32, @intCast(C)));
    a.add(inv(if (l.global) 2 * weights.taps + 1 else weights.taps + 1));
    const grid: cuda.Dim3 = .{ .x = @intCast(std.math.divCeil(usize, C, 2 * threads) catch unreachable), .y = @intCast(n), .z = 2 };
    try cuda.launch.launch(c.k.conv.kv_commit_segs_fn, .{ .grid = grid, .block = .{ .x = threads } }, c.stream, &a);
}

/// postCommit for every segment of a batched forward in one launch (see kvCommitSegs): x and out are the whole
/// forward's rows, each segment's state and history at index 0.
pub fn postCommitSegs(c: *const Ctx, taps: u64, x: Mat, out: Mat, segs: []const Seg) !void {
    try segsFit(segs, @min(x.rows, out.rows));
    var t: KvCommitSegsArgs = .{
        .x = .{ x.ptr, 0 },
        .ldx = .{ @intCast(x.ld), 0 },
        .w = .{ taps, 0 },
        .y = .{ out.ptr, 0 },
        .ldy = .{ @intCast(out.ld), 0 },
    };
    var n: usize = 0;
    for (segs) |g| {
        if (g.rows == 0) continue;
        t.put(n, g, try postArgs(taps, x.slice(g.row0, g.rows), g.state[0], g.hist[0], out.slice(g.row0, g.rows)));
        n += 1;
    }
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(t);
    a.add(@as(i32, @intCast(weights.hidden)));
    a.add(inv(weights.taps + 1));
    const grid: cuda.Dim3 = .{ .x = @intCast(std.math.divCeil(usize, weights.hidden, 2 * threads) catch unreachable), .y = @intCast(n), .z = 1 };
    try cuda.launch.launch(c.k.conv.kv_commit_segs_fn, .{ .grid = grid, .block = .{ .x = threads } }, c.stream, &a);
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

test "inkling2 cuda conv: the segs launch's arguments" {
    // inkling2_conv.cu KvCommitSegs: 7 pairs of 8 bytes, 2 x 8 pointer pairs, 2 x 8 ints; well under the 4 KB of
    // kernel parameters
    try std.testing.expectEqual(@as(usize, 432), @sizeOf(KvCommitSegsArgs));
    try std.testing.expectEqual(@as(usize, 112), @offsetOf(KvCommitSegsArgs, "s"));
    try std.testing.expectEqual(@as(usize, 368), @offsetOf(KvCommitSegsArgs, "row0"));
    const ok = [_]Seg{ .{ .row0 = 0, .rows = 2, .state = .{ 4, 4 } }, .{ .row0 = 2, .rows = 0, .state = .{ 4, 4 } }, .{ .row0 = 2, .rows = 3, .state = .{ 4, 4 } } };
    try segsFit(&ok, 5);
    try std.testing.expectError(error.BadConvShape, segsFit(&ok, 4));
    const overlap = [_]Seg{ .{ .row0 = 0, .rows = 2, .state = .{ 4, 4 } }, .{ .row0 = 1, .rows = 1, .state = .{ 4, 4 } } };
    try std.testing.expectError(error.BadConvShape, segsFit(&overlap, 8));
}

test "inkling2 cuda conv: torch's reciprocals of sqrt(5) and sqrt(9)" {
    // fp32(sqrt(5)) = 2.236068 (0x400F1BBD); its fp32 reciprocal 0x3EE4F92E; 1 / fp32(3) = 0x3EAAAAAB
    try std.testing.expectEqual(@as(u32, 0x3EE4F92E), @as(u32, @bitCast(inv(5))));
    try std.testing.expectEqual(@as(u32, 0x3EAAAAAB), @as(u32, @bitCast(inv(9))));
}
