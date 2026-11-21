//! Inkling-2's vision tower on one GPU (FABLE-IMAGE-INPUT-DESIGN 1.3, 2.2): RGB bytes and (H, W) to the T rows of
//! 4,608 the text model takes at the image's tokens. The ops are inkling2_vision.cu's (patches, position sum, class
//! row, LayerNorm, rotary, epilogues, the reference attention core) around the lane matmul (cuda/qmmf.zig, BF16 mode)
//! for the 95 packed projections (patch embedding, 23 x fused q|k|v, out, fc1, fc2, the merger's two).
//!
//! Rounding points are torch's: each projection rounds once to bf16 after its bias; layer scale and residual are two
//! bf16 roundings; GELU is torch's erf form in fp32. Every projection and epilogue runs on at most `row_block` rows a
//! call and the attention core on at most `q_block` query rows a call; neither changes a bit (the lane's sums do not
//! depend on the row count, the core's on the query block). A tower run can be cut into steps of whole layers
//! (begin, step, finish) with the state held on the device between them.
//!
//! Engine mode (Options.own_work false, the serving engine's rank 0): the tower keeps only its weights and the
//! residual stream `x` of the image in flight; every other buffer is borrowed per unit (`bindSlice`: begin, a run of
//! layers, the merger) from a region the caller lends, the engine's op scratch between forwards, which holds nothing
//! across units: the rotary tables are recomputed at each layer unit (the same kernel on the same inputs), and the
//! merger writes the rows where the caller says (`finishInto`).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const st = ink.st;
const Weight = cuda.qmmf.Weight;
const Stream = cuda.Stream;

pub const hidden = 1024;
pub const heads = 16;
pub const head_dim = 64;
pub const layers = 23;
pub const inter = 4096;
pub const proj_hidden = 4096;
pub const out_width = 4608;
pub const block = 40; // pixels a token side
pub const patch = 20;
pub const features = 3 * patch * patch; // 1200
pub const features_pad = 1216; // the lane's K: a multiple of 64
pub const grid_side = 32; // the position table's side
pub const eps: f32 = 1e-5;
pub const tensors = 422;
const prefix = "model.visual.";

/// torch's inv_freq (compute_axial_rope_parameters: 1 / 10000 ** (arange(0, 32, 2) / 32), fp32 on the CPU), as bits.
pub const inv_freq_bits = [16]u32{
    0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2, 0x3d0186e3, 0x3c91ad39,
    0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528, 0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753,
};

/// The 16 rotary frequencies as torch computes them: pow in fp32 (correctly rounded here through f64), then 1 / x
/// in fp32. Checked against `inv_freq_bits` when a tower loads.
pub fn invFreq() [16]f32 {
    var out: [16]f32 = undefined;
    for (&out, 0..) |*f, j| {
        const e: f32 = @as(f32, @floatFromInt(2 * j)) / 32.0;
        const p: f32 = @floatCast(std.math.pow(f64, 10000.0, @as(f64, e)));
        f.* = 1.0 / p;
    }
    return out;
}

/// The processor's fp32 value of a byte, (x - 127.5) / 127.5, and at 256 the padding's, (-1 - 127.5) / 127.5.
pub fn pixelTable() [257]f32 {
    var t: [257]f32 = undefined;
    for (0..256) |b| {
        const x: f32 = @floatFromInt(b);
        t[b] = (x - 127.5) / 127.5;
    }
    const pad: f32 = -1.0;
    t[256] = (pad - 127.5) / 127.5;
    return t;
}

/// Tokens a side: ceil(H / 40) rows, ceil(W / 40) cols.
pub const Grid = struct {
    rows: usize,
    cols: usize,

    pub fn of(h: usize, w: usize) Grid {
        return .{ .rows = (h + block - 1) / block, .cols = (w + block - 1) / block };
    }
    pub fn tokens(g: Grid) usize {
        return g.rows * g.cols;
    }
    pub fn patches(g: Grid) usize {
        return 4 * g.tokens();
    }
    /// Tower rows: the class row, then the patches.
    pub fn towerRows(g: Grid) usize {
        return 4 * g.tokens() + 1;
    }
};

/// One axis of transformers' vision_utils _interpolation_axis_taps_weights (bilinear, align_corners False, border):
/// src = (i + 0.5) * 32 / size - 0.5 in fp32, taps floor and floor + 1 clamped, weights max(0, 1 - |src - floor - o|).
fn axisTaps(i: usize, size: usize, taps: *[2]i32, w: *[2]f32) void {
    const side: f32 = grid_side;
    const src: f32 = ((@as(f32, @floatFromInt(i)) + 0.5) * side) / @as(f32, @floatFromInt(size)) - 0.5;
    const fl = @floor(src);
    const base: i64 = @intFromFloat(fl);
    for (0..2) |o| {
        taps[o] = @intCast(std.math.clamp(base + @as(i64, @intCast(o)), 0, grid_side - 1));
        const dist = @abs((src - fl) - @as(f32, @floatFromInt(o)));
        w[o] = @max(1.0 - dist, 0.0);
    }
}

/// get_vision_interpolation_indices_and_weights(merge 2) for one image: patch p (block order) gets 4 table rows and
/// 4 fp32 weights in tap order (h0 w0, h0 w1, h1 w0, h1 w1). idx and w hold 4 * patches each.
pub fn positionTaps(g: Grid, idx: []i32, w: []f32) void {
    const ph = 2 * g.rows;
    const pw = 2 * g.cols;
    for (0..g.patches()) |p| {
        const t = p / 4;
        const in = p % 4;
        const row = 2 * (t / g.cols) + in / 2;
        const col = 2 * (t % g.cols) + in % 2;
        var ht: [2]i32 = undefined;
        var hw: [2]f32 = undefined;
        var wt: [2]i32 = undefined;
        var ww: [2]f32 = undefined;
        axisTaps(row, ph, &ht, &hw);
        axisTaps(col, pw, &wt, &ww);
        for (0..2) |a| for (0..2) |b| {
            idx[4 * p + 2 * a + b] = ht[a] * grid_side + wt[b];
            w[4 * p + 2 * a + b] = hw[a] * ww[b];
        };
    }
}

pub const Core = enum { ref, tc };

pub const Options = struct {
    /// rows a projection or epilogue call takes at most
    row_block: usize = 2048,
    /// query rows an attention core call takes at most
    q_block: usize = 2048,
    core: Core = .tc,
    /// the longest image side the buffers hold
    max_side: usize = 4096,
    /// the tower allocates every buffer of an image (the tool); false: engine mode, only `x` resident and the rest
    /// borrowed per unit (`bindSlice`)
    own_work: bool = true,
};

/// What `Tower.load` takes beside the file: the content-digest manifest's text (each of the 422 tensors checked by
/// name, len and xxh64 against `rank`'s lines, digests.checkNamed), and a device region the projections' staging may
/// borrow (36 MiB; the engine's op scratch, idle at start-up) instead of allocating its own.
pub const LoadArgs = struct {
    manifest: ?[]const u8 = null,
    rank: u32 = 0,
    stage: ?cuda.DeviceBuffer = null,
};

/// An engine-mode unit's borrowed buffers (`bindSlice`): E1 .. E5, a run of layers, or the merger.
pub const Phase = enum { begin, layer, merger };

/// inkling2_vision.cu's kernels; `attn_tc` when the image holds inkling2_vision_attn.cuh's tf_iv_attn.
pub const Kernels = struct {
    mod: cuda.Module,
    patchify: cuda.Function,
    embed: cuda.Function,
    class: cuda.Function,
    layer_norm: cuda.Function,
    rope_table: cuda.Function,
    rotary_fn: cuda.Function,
    bias: cuda.Function,
    bias_gelu: cuda.Function,
    bias_ls_res: cuda.Function,
    ls_res: cuda.Function,
    gelu_fn: cuda.Function,
    gather: cuda.Function,
    attn_ref: cuda.Function,
    attn_tc: ?cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_vision);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .patchify = try mod.function("tf_iv_patchify"),
            .embed = try mod.function("tf_iv_embed"),
            .class = try mod.function("tf_iv_class"),
            .layer_norm = try mod.function("tf_iv_layer_norm"),
            .rope_table = try mod.function("tf_iv_rope_table"),
            .rotary_fn = try mod.function("tf_iv_rotary"),
            .bias = try mod.function("tf_iv_bias"),
            .bias_gelu = try mod.function("tf_iv_bias_gelu"),
            .bias_ls_res = try mod.function("tf_iv_bias_ls_res"),
            .ls_res = try mod.function("tf_iv_ls_res"),
            .gelu_fn = try mod.function("tf_iv_gelu"),
            .gather = try mod.function("tf_iv_gather"),
            .attn_ref = try mod.function("tf_iv_attn_ref"),
            .attn_tc = optionalFunction(d, mod, "tf_iv_attn"),
        };
    }

    /// A kernel the image may lack (the tensor-core core): null when cuModuleGetFunction does not find it.
    fn optionalFunction(d: *const cuda.Driver, mod: cuda.Module, name: [:0]const u8) ?cuda.Function {
        var f: cuda.abi.Function = null;
        if (d.api.cuModuleGetFunction(&f, mod.handle, name.ptr) != cuda.abi.success) return null;
        return .{ .d = d, .handle = f };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    fn go(f: cuda.Function, s: Stream, gx: usize, gy: usize, threads: u32, a: *cuda.Args) !void {
        if (gx == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = threads } }, s, a);
    }

    /// E1: patches p0 .. p0 + n of an H x W image (rgb bytes on the device) as bf16 rows [n][1216].
    pub fn patchRows(k: *const Kernels, s: Stream, rgb: u64, table: u64, h: usize, w: usize, cols: usize, p0: usize, n: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(rgb);
        a.add(table);
        for ([_]usize{ h, w, cols, p0 }) |v| a.add(@as(i32, @intCast(v)));
        a.add(out);
        try go(k.patchify, s, n, 1, 256, &a);
    }

    /// E3 + E4: x[1 + p] for p = p0 .. p0 + n from e2 rows [n][1024] and the position taps.
    pub fn embedRows(k: *const Kernels, s: Stream, e2: u64, table: u64, idx: u64, w: u64, p0: usize, n: usize, x: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ e2, table, idx, w }) |p| a.add(p);
        a.add(@as(i32, @intCast(p0)));
        a.add(x);
        try go(k.embed, s, n, 1, 256, &a);
    }

    pub fn classRow(k: *const Kernels, s: Stream, cls: u64, cls_pos: u64, x: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ cls, cls_pos, x }) |p| a.add(p);
        try go(k.class, s, 1, 1, 256, &a);
    }

    /// y [n][1024] = LayerNorm(x) * w + b (eps 1e-5); x and y must not overlap.
    pub fn layerNorm(k: *const Kernels, s: Stream, x: u64, w: u64, b: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, b, y }) |p| a.add(p);
        a.add(eps);
        try go(k.layer_norm, s, n, 1, 256, &a);
    }

    /// cos, sin [S][32] of rows 0 .. S - 1 for a grid `cols` tokens wide; inv: 16 fp32 on the device.
    pub fn ropeTable(k: *const Kernels, s: Stream, inv: u64, cols: usize, rows: usize, cs: u64, sn: u64) !void {
        var a: cuda.Args = .{};
        a.add(inv);
        a.add(@as(i32, @intCast(cols)));
        a.add(@as(i32, 0));
        a.add(cs);
        a.add(sn);
        try go(k.rope_table, s, rows, 1, 32, &a);
    }

    /// L3 on rows r0 .. r0 + n of qkv (rows `ld` apart, q at +0, k at +1024) in place.
    pub fn rotary(k: *const Kernels, s: Stream, qkv: u64, ld: usize, cs: u64, sn: u64, r0: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(@as(i32, @intCast(ld)));
        a.add(cs);
        a.add(sn);
        a.add(@as(i32, @intCast(r0)));
        try go(k.rotary_fn, s, n, 1, 256, &a);
    }

    /// o[r * ldo + c] = bf16(acc[r * n + c] + bias[c]) for `rows` rows.
    pub fn biasRows(k: *const Kernels, s: Stream, acc: u64, bias: u64, o: u64, rows: usize, n: usize, ldo: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ acc, bias, o }) |p| a.add(p);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(ldo)));
        try go(k.bias, s, rows, (n + 255) / 256, 256, &a);
    }

    pub fn biasGelu(k: *const Kernels, s: Stream, acc: u64, bias: u64, o: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ acc, bias, o }) |p| a.add(p);
        a.add(@as(i32, @intCast(n)));
        try go(k.bias_gelu, s, rows, (n + 255) / 256, 256, &a);
    }

    /// x [rows][1024] = bf16(x + bf16(ls * bf16(acc + bias))) in place.
    pub fn biasLsRes(k: *const Kernels, s: Stream, acc: u64, bias: u64, ls: u64, x: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ acc, bias, ls, x }) |p| a.add(p);
        try go(k.bias_ls_res, s, rows, hidden / 256, 256, &a);
    }

    /// x [rows][1024] = bf16(x + bf16(ls * y)) in place, y bf16.
    pub fn lsRes(k: *const Kernels, s: Stream, y: u64, ls: u64, x: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ y, ls, x }) |p| a.add(p);
        try go(k.ls_res, s, rows, hidden / 256, 256, &a);
    }

    /// o [count] = bf16(gelu(x)); x may be o.
    pub fn gelu(k: *const Kernels, s: Stream, x: u64, o: u64, count: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(o);
        a.add(@as(i64, @intCast(count)));
        try go(k.gelu_fn, s, (count + 255) / 256, 1, 256, &a);
    }

    /// out [n][1024] = x's rows at rows[0 .. n] (i32 on the device).
    pub fn gatherRows(k: *const Kernels, s: Stream, x: u64, rows: u64, n: usize, out: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, rows, out }) |p| a.add(p);
        try go(k.gather, s, n, 1, 256, &a);
    }

    /// L4 on query rows q0 .. q0 + qn of an S-row image: q, k, v (row r, head h, d) at base + r * ld + h * 64 + d;
    /// out row r at out + (r - q0) * 1024 + h * 64 + d. `ref`: tf_iv_attn_ref; `tc`: tf_iv_attn (64 rows a CTA).
    pub fn core(k: *const Kernels, s: Stream, which: Core, q: u64, kk: u64, v: u64, ld: usize, out: u64, rows: usize, q0: usize, qn: usize) !void {
        if (qn == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kk, v }) |p| a.add(p);
        a.add(@as(i32, @intCast(ld)));
        a.add(out);
        for ([_]usize{ rows, q0, qn }) |x| a.add(@as(i32, @intCast(x)));
        switch (which) {
            .ref => try go(k.attn_ref, s, (qn + 127) / 128, heads, 128, &a),
            .tc => try go(k.attn_tc orelse return error.TcCoreAbsent, s, (qn + 63) / 64, heads, 128, &a),
        }
    }
};

/// One encoder layer's device tensors: norms, layer scales and biases (bf16), and its packed projections.
pub const Layer = struct {
    ln1_w: u64 = 0,
    ln1_b: u64 = 0,
    ln2_w: u64 = 0,
    ln2_b: u64 = 0,
    ls1: u64 = 0,
    ls2: u64 = 0,
    b_qkv: u64 = 0, // [3072]: q, k, v
    b_o: u64 = 0,
    b_fc1: u64 = 0,
    b_fc2: u64 = 0,
    qkv: Weight = undefined, // [3072, 1024]
    o: Weight = undefined,
    fc1: Weight = undefined,
    fc2: Weight = undefined,
};

/// The device rows of a finished image: [tokens][4608] bf16.
pub const Rows = struct { ptr: u64, tokens: usize };

/// A small tensor's place in the vector buffer: bf16 elements, by checkpoint name.
const Small = struct { name: []const u8, len: usize };

/// Each layer's small tensors, in the vector buffer's order.
const per_layer = [_]Small{
    .{ .name = "ln_1.weight", .len = hidden },          .{ .name = "ln_1.bias", .len = hidden },
    .{ .name = "ln_2.weight", .len = hidden },          .{ .name = "ln_2.bias", .len = hidden },
    .{ .name = "ls_1.gamma", .len = hidden },           .{ .name = "ls_2.gamma", .len = hidden },
    .{ .name = "self_attn.q_proj.bias", .len = hidden }, .{ .name = "self_attn.k_proj.bias", .len = hidden },
    .{ .name = "self_attn.v_proj.bias", .len = hidden }, .{ .name = "self_attn.out_proj.bias", .len = hidden },
    .{ .name = "mlp.c_fc.bias", .len = inter },         .{ .name = "mlp.c_proj.bias", .len = hidden },
};
/// The backbone's small tensors before the position table.
const top = [_]Small{
    .{ .name = "backbone.embeddings.class_embedding", .len = hidden },
    .{ .name = "backbone.embeddings.cls_position_embedding", .len = hidden },
    .{ .name = "backbone.ln_pre.weight", .len = hidden },
    .{ .name = "backbone.ln_pre.bias", .len = hidden },
};

/// The vector buffer's bytes: the small tensors, the position table, the 257 pixel values and the 16 frequencies.
fn vectorBytes() usize {
    var per: usize = 0;
    for (per_layer) |x| per += x.len;
    return (top.len * hidden + grid_side * grid_side * hidden + layers * per) * 2 + 257 * 4 + 16 * 4;
}

fn laneBytes(n: usize, k: usize) usize {
    return cuda.modelopt.padded(n) * k * 2;
}

/// The packed projections' bytes.
fn projBytes() usize {
    const per = laneBytes(3 * hidden, hidden) + laneBytes(hidden, hidden) + laneBytes(inter, hidden) + laneBytes(hidden, inter);
    return laneBytes(hidden, features_pad) + layers * per + laneBytes(proj_hidden, 4 * hidden) + laneBytes(out_width, proj_hidden);
}

/// Device bytes an engine-mode tower holds for images of `opt.max_side` a side (weights, tables and `x`), before
/// loading: the start-up estimate.
pub fn residentBytes(opt: Options) usize {
    return vectorBytes() + projBytes() + Grid.of(opt.max_side, opt.max_side).towerRows() * hidden * 2;
}

/// The RGB bytes an image of grid g holds at most (whole blocks a side).
fn rgbBytes(g: Grid) usize {
    return g.rows * block * g.cols * block * 3;
}

pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    opt: Options,
    k: Kernels,
    lane: cuda.qmmf.Lane,
    vectors: cuda.DeviceBuffer, // every bf16 vector and table, then the 257 pixel values and the 16 frequencies
    proj: cuda.DeviceBuffer, // the lane's projections, one after another
    work: cuda.DeviceBuffer, // the image's buffers (engine mode: `x` alone)
    /// load: each tensor's name, len and xxh64 as read (with a digest manifest)
    sums: ?*std.ArrayList(ink.digests.Named) = null,
    /// the lane's split-K partials at most, over every row count a call takes
    part_bytes: usize = 0,
    // weights
    class: u64 = 0,
    cls_pos: u64 = 0,
    pos_table: u64 = 0, // [1024][1024]
    ln_pre_w: u64 = 0,
    ln_pre_b: u64 = 0,
    pixel_table: u64 = 0, // 257 fp32
    inv: u64 = 0, // 16 fp32
    patch_w: Weight = undefined, // [1024, 1216]
    layer: [layers]Layer = @splat(.{}),
    m1: Weight = undefined, // [4096, 4096]
    m2: Weight = undefined, // [4608, 4096]
    // buffers (sizes for max_side x max_side)
    max_tokens: usize,
    x: u64 = 0, // [S][1024] the residual stream
    h: u64 = 0, // [S][1024] normed rows; the embedding before ln_pre
    qkv: u64 = 0, // [S][3072]
    ao: u64 = 0, // [S][1024] the core's output
    acc: u64 = 0, // [row_block][4096] fp32
    g: u64 = 0, // [row_block][4096] bf16: fc1's GELU, the merger's middle, E2's rows
    patches: u64 = 0, // [row_block][1216]
    rgb: u64 = 0,
    idx: u64 = 0, // [P][4] i32
    wts: u64 = 0, // [P][4] f32
    cs: u64 = 0, // [S][32] f32
    sn: u64 = 0,
    out: u64 = 0, // [T][4608]
    part: u64 = 0,
    rgb_cap: usize = 0,
    host_idx: []i32 = &.{},
    host_w: []f32 = &.{},
    // the image in flight
    grid: Grid = .{ .rows = 0, .cols = 0 },
    next: usize = 0, // the next layer step() runs

    /// The 422 model.visual.* tensors from the safetensors file `path` (names, BF16, shapes checked; with a manifest
    /// their contents, `LoadArgs`), the projections packed for the lane, and buffers for an image of opt.max_side a side
    /// (engine mode: `x` alone). The context must be current.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, opt: Options, args: LoadArgs) !Tower {
        if (opt.row_block == 0 or opt.q_block == 0) return error.BadVisionOptions;
        const inv = invFreq();
        for (inv, inv_freq_bits) |f, b| if (@as(u32, @bitCast(f)) != b) return error.InvFreqDiffers;
        var r = try st.Reader.open(gpa, io, path);
        defer r.close();
        var count: usize = 0;
        for (r.entries) |e| if (std.mem.startsWith(u8, e.name, prefix)) {
            count += 1;
        };
        if (count != tensors) {
            std.log.err("{s}: {d} model.visual.* tensors, expected {d}", .{ path, count, tensors });
            return error.UnexpectedVisionTensors;
        }
        var k = try Kernels.load(d);
        errdefer k.unload();
        if (opt.core == .tc and k.attn_tc == null) return error.TcCoreAbsent;
        var lane = try cuda.qmmf.Lane.load(d, (try d.currentCapability()) / 10);
        errdefer lane.unload();
        try lane.loadPrompt(d);
        const g = Grid.of(opt.max_side, opt.max_side);
        var t: Tower = .{ .gpa = gpa, .d = d, .opt = opt, .k = k, .lane = lane, .vectors = undefined, .proj = undefined, .work = undefined, .max_tokens = g.tokens() };
        var sums: std.ArrayList(ink.digests.Named) = .empty;
        defer sums.deinit(gpa);
        if (args.manifest != null) {
            try sums.ensureTotalCapacity(gpa, tensors);
            t.sums = &sums;
        }
        try t.loadVectors(&r);
        errdefer t.vectors.free();
        try t.loadProjections(gpa, &r, args.stage);
        errdefer t.proj.free();
        t.sums = null;
        if (args.manifest) |text| {
            const c = try ink.digests.checkNamed(gpa, text, args.rank, prefix, sums.items);
            if (c.differ + c.missing > 0 or c.pieces != tensors) {
                std.log.err("inkling2 rank {d}: vision tower: {d} of {d} tensors differ from the digest manifest and {d} have no matching line (first: {s})", .{ args.rank, c.differ, c.pieces, c.missing, c.first orelse "none" });
                return error.WeightBytesDiffer;
            }
            std.log.info("inkling2 rank {d}: vision tower: {d} tensors match the digest manifest (xxh64 {x:0>16})", .{ args.rank, c.pieces, c.loaded });
        }
        t.part_bytes = t.partNeed();
        if (opt.own_work) try t.allocWork(g) else try t.allocResident(g);
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.gpa.free(t.host_idx);
        t.gpa.free(t.host_w);
        t.work.free();
        t.proj.free();
        t.vectors.free();
        t.lane.unload();
        t.k.unload();
        t.* = undefined;
    }

    /// Tensor model.visual.`name` (BF16, `shape`) into `out`, its xxh64 noted when `sums` is set.
    fn read(t: *Tower, r: *const st.Reader, name: []const u8, shape: []const u64, out: []u8) !void {
        var nb: [128]u8 = undefined;
        const full = try std.fmt.bufPrint(&nb, prefix ++ "{s}", .{name});
        const e = r.find(full) orelse {
            std.log.err("the tower file has no {s}", .{full});
            return error.MissingVisionTensor;
        };
        var ok = e.dtype == .bf16 and e.rank == shape.len;
        if (ok) for (shape, 0..) |n, i| {
            if (e.dim(i) != n) ok = false;
        };
        if (!ok or out.len != e.bytes()) {
            std.log.err("{s}: {s} {any}, expected BF16 {any}", .{ full, e.dtype.name(), e.shape[0..e.rank], shape });
            return error.UnexpectedVisionTensor;
        }
        try r.readInto(e, 0, out);
        if (t.sums) |list| try list.append(t.gpa, .{ .name = e.name, .len = out.len, .xxh64 = std.hash.XxHash64.hash(0, out) });
    }

    fn loadVectors(t: *Tower, r: *const st.Reader) !void {
        const bytes = vectorBytes();
        const host = try t.gpa.alloc(u8, bytes);
        defer t.gpa.free(host);
        var at: usize = 0;
        var offs: [top.len]usize = undefined;
        for (top, 0..) |s, i| {
            try t.read(r, s.name, &.{s.len}, host[at..][0 .. s.len * 2]);
            offs[i] = at;
            at += s.len * 2;
        }
        const table_at = at;
        try t.read(r, "backbone.embeddings.position_embedding", &.{ grid_side * grid_side, hidden }, host[at..][0 .. grid_side * grid_side * hidden * 2]);
        at += grid_side * grid_side * hidden * 2;
        var lofs: [layers][per_layer.len]usize = undefined;
        for (0..layers) |l| for (per_layer, 0..) |s, i| {
            var nb: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "backbone.layers.{d}.{s}", .{ l, s.name });
            try t.read(r, name, &.{s.len}, host[at..][0 .. s.len * 2]);
            lofs[l][i] = at;
            at += s.len * 2;
        };
        const px_at = at;
        const px = pixelTable();
        @memcpy(host[at..][0 .. 257 * 4], std.mem.sliceAsBytes(&px));
        at += 257 * 4;
        const inv_at = at;
        const inv = invFreq();
        @memcpy(host[at..][0 .. 16 * 4], std.mem.sliceAsBytes(&inv));
        at += 16 * 4;
        std.debug.assert(at == bytes);
        t.vectors = try cuda.DeviceBuffer.fromHost(t.d, host);
        const base = t.vectors.ptr;
        t.class = base + offs[0];
        t.cls_pos = base + offs[1];
        t.ln_pre_w = base + offs[2];
        t.ln_pre_b = base + offs[3];
        t.pos_table = base + table_at;
        t.pixel_table = base + px_at;
        t.inv = base + inv_at;
        for (&t.layer, lofs) |*l, o| {
            l.ln1_w = base + o[0];
            l.ln1_b = base + o[1];
            l.ln2_w = base + o[2];
            l.ln2_b = base + o[3];
            l.ls1 = base + o[4];
            l.ls2 = base + o[5];
            l.b_qkv = base + o[6]; // q, k, v biases are adjacent: [3072]
            if (o[7] != o[6] + hidden * 2 or o[8] != o[7] + hidden * 2) return error.UnexpectedVisionLayout;
            l.b_o = base + o[9];
            l.b_fc1 = base + o[10];
            l.b_fc2 = base + o[11];
        }
    }

    /// The projections packed for the lane; `lent`, when it holds the largest source, stages them instead of a buffer
    /// of their own.
    fn loadProjections(t: *Tower, gpa: std.mem.Allocator, r: *const st.Reader, lent: ?cuda.DeviceBuffer) !void {
        const total = projBytes();
        t.proj = try cuda.DeviceBuffer.alloc(t.d, total);
        errdefer t.proj.free();
        const most = out_width * proj_hidden * 2; // the largest source: proj_out
        const host = try gpa.alloc(u8, most);
        defer gpa.free(host);
        var stage = if (lent) |b| (if (b.len >= most) view(t.d, b.ptr, most) else try cuda.DeviceBuffer.alloc(t.d, most)) else try cuda.DeviceBuffer.alloc(t.d, most);
        defer stage.free();
        var packer = try cuda.modelopt.Packer.load(t.d);
        defer packer.unload();
        var s = try Stream.init(t.d, true);
        defer s.deinit();
        var at: usize = 0;
        const P = struct {
            fn pack(tw: *Tower, pk: *const cuda.modelopt.Packer, ss: Stream, stg: *cuda.DeviceBuffer, src: []const u8, n: usize, k: usize, off: *usize) !Weight {
                if (k % cuda.qmmf.group != 0) return error.UnexpectedVisionLayout;
                try ss.synchronize(); // the stage is free
                try stg.upload(0, src);
                const dst = tw.proj.ptr + off.*;
                try pk.bf16Lane(ss, stg.ptr, dst, n, k);
                off.* += laneBytes(n, k);
                return cuda.modelopt.bf16Weight(dst, n, k);
            }
        };
        // E2: [1024, 1200] zero-padded to K = 1216
        {
            const raw = try gpa.alloc(u8, hidden * features * 2);
            defer gpa.free(raw);
            try t.read(r, "backbone.embeddings.patch_embedding.weight", &.{ hidden, features }, raw);
            const dst = host[0 .. hidden * features_pad * 2];
            @memset(dst, 0);
            for (0..hidden) |n| @memcpy(dst[n * features_pad * 2 ..][0 .. features * 2], raw[n * features * 2 ..][0 .. features * 2]);
            t.patch_w = try P.pack(t, &packer, s, &stage, dst, hidden, features_pad, &at);
        }
        for (&t.layer, 0..) |*l, i| {
            var nb: [96]u8 = undefined;
            const wb = hidden * hidden * 2;
            for ([_][]const u8{ "q", "k", "v" }, 0..) |p, j| {
                try s.synchronize();
                try t.read(r, try std.fmt.bufPrint(&nb, "backbone.layers.{d}.self_attn.{s}_proj.weight", .{ i, p }), &.{ hidden, hidden }, host[j * wb ..][0..wb]);
            }
            l.qkv = try P.pack(t, &packer, s, &stage, host[0 .. 3 * wb], 3 * hidden, hidden, &at);
            try s.synchronize();
            try t.read(r, try std.fmt.bufPrint(&nb, "backbone.layers.{d}.self_attn.out_proj.weight", .{i}), &.{ hidden, hidden }, host[0..wb]);
            l.o = try P.pack(t, &packer, s, &stage, host[0..wb], hidden, hidden, &at);
            try s.synchronize();
            try t.read(r, try std.fmt.bufPrint(&nb, "backbone.layers.{d}.mlp.c_fc.weight", .{i}), &.{ inter, hidden }, host[0 .. inter * hidden * 2]);
            l.fc1 = try P.pack(t, &packer, s, &stage, host[0 .. inter * hidden * 2], inter, hidden, &at);
            try s.synchronize();
            try t.read(r, try std.fmt.bufPrint(&nb, "backbone.layers.{d}.mlp.c_proj.weight", .{i}), &.{ hidden, inter }, host[0 .. inter * hidden * 2]);
            l.fc2 = try P.pack(t, &packer, s, &stage, host[0 .. inter * hidden * 2], hidden, inter, &at);
        }
        try s.synchronize();
        try t.read(r, "proj_in.weight", &.{ proj_hidden, 4 * hidden }, host[0 .. proj_hidden * 4 * hidden * 2]);
        t.m1 = try P.pack(t, &packer, s, &stage, host[0 .. proj_hidden * 4 * hidden * 2], proj_hidden, 4 * hidden, &at);
        try s.synchronize();
        try t.read(r, "proj_out.weight", &.{ out_width, proj_hidden }, host[0 .. out_width * proj_hidden * 2]);
        t.m2 = try P.pack(t, &packer, s, &stage, host[0 .. out_width * proj_hidden * 2], out_width, proj_hidden, &at);
        try s.synchronize();
        std.debug.assert(at == total);
    }

    /// The lane's split-K partials at most: every projection shape at every row count a call takes (up to row_block;
    /// a tail block's count may need more than a whole block's).
    fn partNeed(t: *const Tower) usize {
        var part: usize = 0;
        const ws = [_]Weight{ t.patch_w, t.layer[0].qkv, t.layer[0].o, t.layer[0].fc1, t.layer[0].fc2, t.m1, t.m2 };
        for (1..t.opt.row_block + 1) |m| for (ws) |w| {
            part = @max(part, t.lane.partBytes(m, w));
        };
        return part;
    }

    fn allocWork(t: *Tower, g: Grid) !void {
        const S = g.towerRows();
        const P = g.patches();
        const T = g.tokens();
        const rb = t.opt.row_block;
        const part = t.part_bytes;
        t.rgb_cap = t.opt.max_side * t.opt.max_side * 3;
        const sizes = [_]usize{
            S * hidden * 2, // x
            S * hidden * 2, // h
            S * 3 * hidden * 2, // qkv
            S * hidden * 2, // ao
            rb * inter * 4, // acc
            rb * inter * 2, // g
            rb * features_pad * 2, // patches
            t.rgb_cap, // rgb
            P * 4 * 4, // idx
            P * 4 * 4, // wts
            S * 32 * 4, // cs
            S * 32 * 4, // sn
            T * out_width * 2, // out
            part,
        };
        var total: usize = 0;
        for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
        t.work = try cuda.DeviceBuffer.alloc(t.d, total);
        errdefer t.work.free();
        var at: usize = t.work.ptr;
        var ptrs: [sizes.len]u64 = undefined;
        for (sizes, &ptrs) |n, *p| {
            p.* = if (n == 0) 0 else at;
            at += std.mem.alignForward(usize, n, 256);
        }
        t.x, t.h, t.qkv, t.ao, t.acc, t.g, t.patches, t.rgb, t.idx, t.wts, t.cs, t.sn, t.out, t.part = ptrs;
        t.host_idx = try t.gpa.alloc(i32, P * 4);
        errdefer t.gpa.free(t.host_idx);
        t.host_w = try t.gpa.alloc(f32, P * 4);
    }

    /// Engine mode: `x` alone on the device (and the host's position taps).
    fn allocResident(t: *Tower, g: Grid) !void {
        t.work = try cuda.DeviceBuffer.alloc(t.d, g.towerRows() * hidden * 2);
        errdefer t.work.free();
        t.x = t.work.ptr;
        t.host_idx = try t.gpa.alloc(i32, g.patches() * 4);
        errdefer t.gpa.free(t.host_idx);
        t.host_w = try t.gpa.alloc(f32, g.patches() * 4);
    }

    /// Device bytes the tower holds (weights, tables and its own buffers; engine mode: resident bytes only).
    pub fn deviceBytes(t: *const Tower) usize {
        return t.vectors.len + t.proj.len + t.work.len;
    }

    // ------------------------------------------------------------------------------------------- engine mode

    /// The buffers phase `p` borrows for grid g, in `bindSlice`'s order (0: none).
    fn phaseSizes(t: *const Tower, g: Grid, p: Phase) [8]usize {
        const S = g.towerRows();
        const P = g.patches();
        const rb = t.opt.row_block;
        return switch (p) {
            // rgb, patches, E2's rows (g), idx, wts, h, part
            .begin => .{ rgbBytes(g), rb * features_pad * 2, rb * hidden * 2, P * 16, P * 16, S * hidden * 2, t.part_bytes, 0 },
            // h, qkv, ao, acc, g, cs, sn, part
            .layer => .{ S * hidden * 2, S * 3 * hidden * 2, S * hidden * 2, rb * inter * 4, rb * inter * 2, S * 32 * 4, S * 32 * 4, t.part_bytes },
            // g, part
            .merger => .{ rb * proj_hidden * 2, t.part_bytes, 0, 0, 0, 0, 0, 0 },
        };
    }

    fn takes(sizes: [8]usize) usize {
        var at: usize = 0;
        for (sizes) |n| if (n > 0) {
            at = std.mem.alignForward(usize, at, 256) + n;
        };
        return at;
    }

    /// The bytes an engine-mode unit borrows at most for an image of grid g (each phase's buffers 256-aligned).
    pub fn sliceNeed(t: *const Tower, g: Grid) usize {
        var most: usize = 0;
        for ([_]Phase{ .begin, .layer, .merger }) |p| most = @max(most, takes(t.phaseSizes(g, p)));
        return most;
    }

    /// Engine mode, before a unit on stream s: phase p's buffers laid out in [base, base + len) (256-aligned; nothing
    /// in it is read before the unit writes it). A layer unit recomputes the rotary tables there first. `g` is the
    /// image's grid (begin: the image about to start; else the one in flight).
    pub fn bindSlice(t: *Tower, s: Stream, base: u64, len: usize, p: Phase, g: Grid) !void {
        if (t.opt.own_work) return error.TowerOwnsItsBuffers;
        if (g.tokens() == 0 or g.tokens() > t.max_tokens) return error.ImageSize;
        if (p != .begin and (g.rows != t.grid.rows or g.cols != t.grid.cols)) return error.NotTheImageInFlight;
        if (base % 256 != 0) return error.BadSlice;
        const sizes = t.phaseSizes(g, p);
        if (takes(sizes) > len) return error.ScratchShort;
        var ptrs: [8]u64 = @splat(0);
        var at: usize = 0;
        for (sizes, &ptrs) |n, *q| if (n > 0) {
            at = std.mem.alignForward(usize, at, 256);
            q.* = base + at;
            at += n;
        };
        t.h, t.qkv, t.ao, t.acc, t.g, t.patches, t.rgb, t.idx, t.wts, t.cs, t.sn, t.part = @as([12]u64, @splat(0));
        t.rgb_cap = 0;
        switch (p) {
            .begin => {
                t.rgb, t.patches, t.g, t.idx, t.wts, t.h, t.part = ptrs[0..7].*;
                t.rgb_cap = sizes[0];
            },
            .layer => {
                t.h, t.qkv, t.ao, t.acc, t.g, t.cs, t.sn, t.part = ptrs;
                try t.k.ropeTable(s, t.inv, g.cols, g.towerRows(), t.cs, t.sn);
            },
            .merger => {
                t.g = ptrs[0];
                t.part = ptrs[1];
            },
        }
    }

    // ------------------------------------------------------------------------------------------- projections

    /// y [n][w.n] bf16 (rows `ldy` apart) = bf16(x W^T + bias), x rows `ldx` apart; row blocks of row_block.
    pub fn linearBias(t: *const Tower, s: Stream, x: u64, ldx: usize, n: usize, w: Weight, bias: u64, y: u64, ldy: usize) !void {
        var r0: usize = 0;
        while (r0 < n) : (r0 += t.opt.row_block) {
            const m = @min(t.opt.row_block, n - r0);
            try t.lane.matmulF32(s, x + r0 * ldx * 2, ldx, m, w, t.acc, t.part);
            try t.k.biasRows(s, t.acc, bias, y + r0 * ldy * 2, m, w.n, ldy);
        }
    }

    /// y [n][w.n] = bf16(gelu(bf16(x W^T + bias))).
    pub fn linearBiasGelu(t: *const Tower, s: Stream, x: u64, ldx: usize, n: usize, w: Weight, bias: u64, y: u64) !void {
        var r0: usize = 0;
        while (r0 < n) : (r0 += t.opt.row_block) {
            const m = @min(t.opt.row_block, n - r0);
            try t.lane.matmulF32(s, x + r0 * ldx * 2, ldx, m, w, t.acc, t.part);
            try t.k.biasGelu(s, t.acc, bias, y + r0 * w.n * 2, m, w.n);
        }
    }

    /// y [n][w.n] bf16 = x W^T, one rounding (no bias).
    pub fn linear(t: *const Tower, s: Stream, x: u64, ldx: usize, n: usize, w: Weight, y: u64) !void {
        var r0: usize = 0;
        while (r0 < n) : (r0 += t.opt.row_block) {
            const m = @min(t.opt.row_block, n - r0);
            try t.lane.matmulOut(s, x + r0 * ldx * 2, ldx, m, w, y + r0 * w.n * 2, .bf16, t.part);
        }
    }

    /// The core over every query row of an S-row image in blocks of q_block; out [S][1024].
    pub fn attention(t: *const Tower, s: Stream, which: Core, q: u64, k: u64, v: u64, ld: usize, rows: usize, out: u64) !void {
        var q0: usize = 0;
        while (q0 < rows) : (q0 += t.opt.q_block) {
            const qn = @min(t.opt.q_block, rows - q0);
            try t.k.core(s, which, q, k, v, ld, out + q0 * hidden * 2, rows, q0, qn);
        }
    }

    // -------------------------------------------------------------------------------------------- the tower

    /// E1 .. E5 and the rotary tables of an H x W RGB image (`rgb` [H][W][3] on the host); then step() its layers.
    pub fn begin(t: *Tower, s: Stream, rgb: []const u8, h: usize, w: usize) !void {
        if (h == 0 or w == 0 or h > t.opt.max_side or w > t.opt.max_side) return error.ImageSize;
        if (rgb.len != h * w * 3) return error.ImageBytes;
        const g = Grid.of(h, w);
        if (!t.opt.own_work and (t.rgb == 0 or rgb.len > t.rgb_cap)) return error.NoBeginSlice;
        try t.prepareGrid(s, g);
        try view(t.d, t.rgb, t.rgb_cap).upload(0, rgb);
        const P = g.patches();
        var p0: usize = 0;
        while (p0 < P) : (p0 += t.opt.row_block) {
            const n = @min(t.opt.row_block, P - p0);
            try t.k.patchRows(s, t.rgb, t.pixel_table, h, w, g.cols, p0, n, t.patches);
            try t.linear(s, t.patches, features_pad, n, t.patch_w, t.g);
            try t.k.embedRows(s, t.g, t.pos_table, t.idx, t.wts, p0, n, t.h);
        }
        try t.k.classRow(s, t.class, t.cls_pos, t.h);
        try t.layerNormRows(s, t.h, t.ln_pre_w, t.ln_pre_b, t.x, g.towerRows());
    }

    fn view(d: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
        return .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
    }

    /// An image's grid without its pixels: the position taps (host, then uploaded) and the rotary tables (engine mode:
    /// at each layer unit instead); the next step() runs layer 0. Waits for the stream first (the buffers are then
    /// free).
    pub fn prepareGrid(t: *Tower, s: Stream, g: Grid) !void {
        if (g.tokens() == 0 or g.tokens() > t.max_tokens) return error.ImageSize;
        try s.synchronize();
        t.grid = g;
        t.next = 0;
        const P = g.patches();
        positionTaps(g, t.host_idx[0 .. 4 * P], t.host_w[0 .. 4 * P]);
        try view(t.d, t.idx, P * 16).upload(0, std.mem.sliceAsBytes(t.host_idx[0 .. 4 * P]));
        try view(t.d, t.wts, P * 16).upload(0, std.mem.sliceAsBytes(t.host_w[0 .. 4 * P]));
        if (t.opt.own_work) try t.k.ropeTable(s, t.inv, g.cols, g.towerRows(), t.cs, t.sn);
    }

    /// y = LayerNorm(x) over n rows in row blocks.
    pub fn layerNormRows(t: *const Tower, s: Stream, x: u64, w: u64, b: u64, y: u64, n: usize) !void {
        var r0: usize = 0;
        while (r0 < n) : (r0 += t.opt.row_block) {
            const m = @min(t.opt.row_block, n - r0);
            try t.k.layerNorm(s, x + r0 * hidden * 2, w, b, y + r0 * hidden * 2, m);
        }
    }

    /// Layer `i` on the image's rows (t.x in place).
    pub fn runLayer(t: *const Tower, s: Stream, i: usize) !void {
        const l = &t.layer[i];
        const S = t.grid.towerRows();
        const rb = t.opt.row_block;
        var r0: usize = 0;
        while (r0 < S) : (r0 += rb) {
            const m = @min(rb, S - r0);
            try t.k.layerNorm(s, t.x + r0 * hidden * 2, l.ln1_w, l.ln1_b, t.h + r0 * hidden * 2, m);
            try t.lane.matmulF32(s, t.h + r0 * hidden * 2, hidden, m, l.qkv, t.acc, t.part);
            try t.k.biasRows(s, t.acc, l.b_qkv, t.qkv + r0 * 3 * hidden * 2, m, 3 * hidden, 3 * hidden);
            try t.k.rotary(s, t.qkv, 3 * hidden, t.cs, t.sn, r0, m);
        }
        try t.attention(s, t.opt.core, t.qkv, t.qkv + hidden * 2, t.qkv + 2 * hidden * 2, 3 * hidden, S, t.ao);
        r0 = 0;
        while (r0 < S) : (r0 += rb) {
            const m = @min(rb, S - r0);
            try t.lane.matmulF32(s, t.ao + r0 * hidden * 2, hidden, m, l.o, t.acc, t.part);
            try t.k.biasLsRes(s, t.acc, l.b_o, l.ls1, t.x + r0 * hidden * 2, m);
        }
        r0 = 0;
        while (r0 < S) : (r0 += rb) {
            const m = @min(rb, S - r0);
            try t.k.layerNorm(s, t.x + r0 * hidden * 2, l.ln2_w, l.ln2_b, t.h + r0 * hidden * 2, m);
            try t.lane.matmulF32(s, t.h + r0 * hidden * 2, hidden, m, l.fc1, t.acc, t.part);
            try t.k.biasGelu(s, t.acc, l.b_fc1, t.g, m, inter);
            try t.lane.matmulF32(s, t.g, inter, m, l.fc2, t.acc, t.part);
            try t.k.biasLsRes(s, t.acc, l.b_fc2, l.ls2, t.x + r0 * hidden * 2, m);
        }
    }

    /// The next `n` layers (fewer at the end); returns the layers left.
    pub fn step(t: *Tower, s: Stream, n: usize) !usize {
        const end = @min(layers, t.next + n);
        while (t.next < end) : (t.next += 1) try t.runLayer(s, t.next);
        return layers - t.next;
    }

    /// The merger on the finished layers: patch rows 1 .. S - 1 as [T][4096], linear_1, GELU, linear_2.
    pub fn finish(t: *Tower, s: Stream) !Rows {
        if (t.out == 0) return error.TowerOwnsNoRows;
        return t.finishInto(s, t.out);
    }

    /// The merger on the finished layers into `out` [T][4608] bf16 (engine mode: after bindSlice(.merger)).
    pub fn finishInto(t: *Tower, s: Stream, out: u64) !Rows {
        if (t.next != layers) return error.TowerNotDone;
        const T = t.grid.tokens();
        try t.merger(s, t.x + hidden * 2, T, out);
        return .{ .ptr = out, .tokens = T };
    }

    /// M2 on [n][4096] rows `x` (rows 4096 apart) into out [n][4608].
    pub fn merger(t: *const Tower, s: Stream, x: u64, n: usize, out: u64) !void {
        var r0: usize = 0;
        while (r0 < n) : (r0 += t.opt.row_block) {
            const m = @min(t.opt.row_block, n - r0);
            try t.lane.matmulOut(s, x + r0 * 4 * hidden * 2, 4 * hidden, m, t.m1, t.g, .bf16, t.part);
            try t.k.gelu(s, t.g, t.g, m * proj_hidden);
            try t.lane.matmulOut(s, t.g, proj_hidden, m, t.m2, out + r0 * out_width * 2, .bf16, t.part);
        }
    }

    /// The whole tower: begin, every layer, finish.
    pub fn run(t: *Tower, s: Stream, rgb: []const u8, h: usize, w: usize) !Rows {
        try t.begin(s, rgb, h, w);
        _ = try t.step(s, layers);
        return t.finish(s);
    }
};

// ----------------------------------------------------------------------------------------------- the serving engine

const vision_store = ink.vision_store;

/// Images a request may hold at most: the spans an image chunk splices (cuda_vision_store.zig max_images).
pub const max_images: u32 = vision_store.max_images;
comptime {
    std.debug.assert(vision_store.tower_layers == layers);
}
/// Image tokens a request by default: one 4,096 x 4,096 image (image_offer.zig Limits.default_image_tokens).
pub const default_image_tokens: u32 = 10_816;
/// The tower's file in the model folder when neither Vision.weights nor TF_INK2_VISION_WEIGHTS names one.
pub const file_name = "inkling2-vision.safetensors";

/// The serving engine's vision (cuda_engine.zig Options.vision; --vision's limits, or TF_INK2_VISION=1 in the tools):
/// rank 0 loads the tower and a row store of `image_tokens` rows; the other ranks hold nothing.
pub const Vision = struct {
    /// image tokens a request at most, and the row store's rows
    image_tokens: u32 = default_image_tokens,
    /// images a request at most (--vision-max-images): an image chunk splices at most `max_images` (16)
    max_images: u32 = 4,
    /// the tower's file; null: TF_INK2_VISION_WEIGHTS, else <model dir>/inkling2-vision.safetensors
    weights: ?[]const u8 = null,
};

/// Every rank, before any GPU work: images a request 1 .. `max_images` (error.VisionMaxImages; the server words it
/// with both numbers) and a store of at least one row.
pub fn checkLimits(v: Vision) error{ VisionMaxImages, BadVisionOptions }!void {
    if (v.max_images == 0 or v.max_images > max_images) return error.VisionMaxImages;
    if (v.image_tokens == 0) return error.BadVisionOptions;
}

/// Device bytes rank 0's vision takes beside the text weights, before loading (the start-up estimate): the
/// engine-mode tower (weights, tables, `x` at max_side) and a row store of `image_tokens` rows.
pub fn rank0Bytes(image_tokens: u32) usize {
    return residentBytes(.{ .own_work = false }) + @as(usize, image_tokens) * vision_store.row_bytes;
}

/// Rank 0's vision in the serving engine (cuda_engine.zig Engine.vision): the tower in engine mode (weights and `x`
/// resident; each unit borrows the op scratch, idle between forwards), the image rows it made (`rows`, bf16
/// [image_tokens][4608]) and their host index (`store`, whose compaction copies go on the engine's stream).
pub const Rank0 = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    /// the engine's stream: the tower's units and the store's copies are ordered with the forwards
    stream: Stream,
    limits: Vision,
    tower: Tower,
    rows: cuda.DeviceBuffer,
    store: vision_store.Store,
    /// the tower's job in flight and its pace beside decoders (cuda_lanes.zig: units under a chunk row cap)
    pace: vision_store.Pace = .{},

    /// The tower's file: `v.weights`, else TF_INK2_VISION_WEIGHTS, else <dir>/inkling2-vision.safetensors (written to
    /// `buf` when built from `dir`).
    pub fn path(buf: []u8, dir: []const u8, v: Vision) ![]const u8 {
        if (v.weights) |p| return p;
        if (std.c.getenv("TF_INK2_VISION_WEIGHTS")) |p| return std.mem.span(p);
        return std.fmt.bufPrint(buf, "{s}/" ++ file_name, .{dir});
    }

    /// Rank 0 at start-up, after the text weights (the context current): the tower from its file, its 422 tensors
    /// checked against `manifest`'s rank-0 lines by name, len and xxh64 when there is one, the projections staged in
    /// `stage` (the op scratch, idle at load), and the row store. error.NoVisionTower when the file is absent.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, v: Vision, stream: Stream, stage: cuda.DeviceBuffer, manifest: ?[]const u8) !*Rank0 {
        try checkLimits(v);
        var pb: [std.fs.max_path_bytes]u8 = undefined;
        const p = try path(&pb, dir, v);
        const r = try gpa.create(Rank0);
        errdefer gpa.destroy(r);
        r.* = .{ .gpa = gpa, .d = d, .stream = stream, .limits = v, .tower = undefined, .rows = undefined, .store = undefined };
        r.tower = Tower.load(gpa, io, d, p, .{ .own_work = false }, .{ .manifest = manifest, .rank = 0, .stage = stage }) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("inkling2 rank 0: --vision: no vision tower at {s} (the recipe's weights step writes <model dir>/" ++ file_name ++ " with VISION=1; TF_INK2_VISION_WEIGHTS names another file)", .{p});
                return error.NoVisionTower;
            },
            else => return err,
        };
        errdefer r.tower.deinit();
        if (manifest == null) std.log.warn("inkling2 rank 0: vision tower contents not checked (no digest manifest)", .{});
        r.rows = try cuda.DeviceBuffer.alloc(d, @as(usize, v.image_tokens) * vision_store.row_bytes);
        errdefer r.rows.free();
        r.store = try vision_store.Store.init(gpa, r.rows.ptr, v.image_tokens, vision_store.row_bytes, .{ .ctx = r, .run = copyRows, .done = copiesDone });
        std.log.info("inkling2 rank 0: vision tower from {s}: {d:.1} MiB resident (weights and tables {d:.1}, x {d:.1}), the image row store {d} rows ({d:.1} MiB), at most {d} images a request", .{ p, mib(r.deviceBytes()), mib(r.tower.vectors.len + r.tower.proj.len), mib(r.tower.work.len), v.image_tokens, mib(r.rows.len), v.max_images });
        return r;
    }

    pub fn destroy(r: *Rank0) void {
        const gpa = r.gpa;
        r.store.deinit();
        r.rows.free();
        r.tower.deinit();
        gpa.destroy(r);
    }

    /// Device bytes held: the tower's resident bytes and the row store.
    pub fn deviceBytes(r: *const Rank0) usize {
        return r.tower.deviceBytes() + r.rows.len;
    }

    /// The op scratch a unit borrows at most: the largest image (max_side a side) in its largest phase.
    pub fn scratchNeed(r: *const Rank0) usize {
        return r.tower.sliceNeed(Grid.of(r.tower.opt.max_side, r.tower.opt.max_side));
    }

    /// One image's whole tower in one unit (the decode tool): begin, every layer, then the merger into `out` ([T][4608]
    /// bf16: the image's store rows), each phase's buffers laid out in [base, base + len) (at least `tower.sliceNeed`
    /// of its grid, borrowed from the op scratch). On the engine's stream; nothing waits for it but `begin`'s grid
    /// (prepareGrid synchronizes first).
    pub fn whole(r: *Rank0, base: u64, len: usize, rgb: []const u8, h: usize, w: usize, out: u64) !Rows {
        try r.beginUnit(base, len, rgb, h, w);
        _ = try r.layersUnit(base, len, layers);
        return r.mergeUnit(base, len, out);
    }

    /// A unit's start: an H x W image's grid (prepareGrid waits for the stream), then E1 .. E5 into the resident `x`,
    /// the begin phase's buffers in [base, base + len).
    pub fn beginUnit(r: *Rank0, base: u64, len: usize, rgb: []const u8, h: usize, w: usize) !void {
        try r.tower.bindSlice(r.stream, base, len, .begin, Grid.of(h, w));
        try r.tower.begin(r.stream, rgb, h, w);
    }

    /// The image in flight's next `n` layers (the rotary tables recomputed in [base, base + len) first); the layers
    /// left.
    pub fn layersUnit(r: *Rank0, base: u64, len: usize, n: usize) !usize {
        try r.tower.bindSlice(r.stream, base, len, .layer, r.tower.grid);
        return r.tower.step(r.stream, n);
    }

    /// The image in flight's merger into `out` once its layers are done.
    pub fn mergeUnit(r: *Rank0, base: u64, len: usize, out: u64) !Rows {
        try r.tower.bindSlice(r.stream, base, len, .merger, r.tower.grid);
        return r.tower.finishInto(r.stream, out);
    }

    /// The op scratch a unit of the image in flight borrows (an image of grid g: `tower.sliceNeed(g)`).
    pub fn unitNeed(r: *const Rank0) usize {
        return r.tower.sliceNeed(r.tower.grid);
    }

    /// The store's compaction copy, on the engine's stream.
    fn copyRows(ctx: *anyopaque, dst: u64, src: u64, bytes: usize) anyerror!void {
        const r: *Rank0 = @ptrCast(@alignCast(ctx));
        try Tower.view(r.d, dst, bytes).copyFrom(0, src, bytes, r.stream.handle);
    }

    /// The store's compaction copies end: the engine's stream drained, so a copy's device fault surfaces here.
    fn copiesDone(ctx: *anyopaque) anyerror!void {
        const r: *Rank0 = @ptrCast(@alignCast(ctx));
        try r.stream.synchronize();
    }
};

fn mib(n: usize) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 20);
}

test "inkling2 vision: rank 0's start-up estimate is the engine-mode tower and a store of image_tokens rows (about 803 MiB)" {
    const M: usize = 1 << 20;
    const base = rank0Bytes(default_image_tokens);
    // weights and tables 625.06 MiB, x [S][1024] at 4096 a side 82.88 MiB, store [10,816][4608] bf16 95.06 MiB
    try std.testing.expect(base > 802 * M and base < 804 * M);
    try std.testing.expectEqual(@as(usize, 10_816 * 9216), base - residentBytes(.{ .own_work = false }));
    try std.testing.expectEqual(base + 9216, rank0Bytes(default_image_tokens + 1));
    try std.testing.expectEqual(@as(usize, 10_609), Grid.of(4096, 4096).tokens());
}

test "inkling2 vision: images a request 1 .. 16, the splice's spans, and a store of at least one row" {
    try checkLimits(.{});
    try checkLimits(.{ .max_images = 1 });
    try checkLimits(.{ .max_images = 16 });
    try std.testing.expectError(error.VisionMaxImages, checkLimits(.{ .max_images = 17 }));
    try std.testing.expectError(error.VisionMaxImages, checkLimits(.{ .max_images = 64 }));
    try std.testing.expectError(error.VisionMaxImages, checkLimits(.{ .max_images = 0 }));
    try std.testing.expectError(error.BadVisionOptions, checkLimits(.{ .image_tokens = 0 }));
    try std.testing.expectEqual(@as(u32, 16), max_images);
    const v: Vision = .{};
    try std.testing.expectEqual(@as(u32, 4), v.max_images);
    try std.testing.expectEqual(@as(u32, 10_816), v.image_tokens);
}

test "inkling2 vision: the tower's file is Vision.weights, else <dir>/inkling2-vision.safetensors" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("/w/t.safetensors", try Rank0.path(&buf, "/m", .{ .weights = "/w/t.safetensors" }));
    if (std.c.getenv("TF_INK2_VISION_WEIGHTS") != null) return error.SkipZigTest;
    try std.testing.expectEqualStrings("/m/inkling2-vision.safetensors", try Rank0.path(&buf, "/m", .{}));
}

test "inkling2 vision: an engine unit at the largest image fits the default op scratch with room for the partials" {
    var t: Tower = .{ .gpa = std.testing.allocator, .d = undefined, .opt = .{ .own_work = false }, .k = undefined, .lane = undefined, .vectors = undefined, .proj = undefined, .work = undefined, .max_tokens = Grid.of(4096, 4096).tokens() };
    const M: usize = 1 << 20;
    const need = t.sliceNeed(Grid.of(4096, 4096));
    // the layer phase: h 82.9, qkv 248.7, ao 82.9, acc 32, g 16, cs/sn 10.4 MiB (the partials borrowed beside)
    try std.testing.expect(need > 470 * M and need < 476 * M);
    t.part_bytes = 64 * M;
    try std.testing.expect(t.sliceNeed(Grid.of(4096, 4096)) <= need + 64 * M + 256);
    try std.testing.expect(t.sliceNeed(Grid.of(4096, 4096)) < 1 << 30);
}

test "inkling2 vision: the tool's buffers give back the device work and the host index when the host weights fail" {
    const F = struct {
        var mem: [16]u8 align(256) = undefined;
        var allocs: usize = 0;
        var frees: usize = 0;
        fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
            out.* = @intFromPtr(&mem);
            allocs += 1;
            return 0;
        }
        fn free(_: u64) callconv(.c) cuda.abi.Result {
            frees += 1;
            return 0;
        }
    };
    var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
    d.api.cuMemAlloc_v2 = F.alloc;
    d.api.cuMemFree_v2 = F.free;
    // the host index is the first host allocation, the weights the second (they fail)
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    const g = Grid.of(64, 64);
    var t: Tower = .{ .gpa = failing.allocator(), .d = &d, .opt = .{ .max_side = 64 }, .k = undefined, .lane = undefined, .vectors = undefined, .proj = undefined, .work = undefined, .max_tokens = g.tokens() };
    try std.testing.expectError(error.OutOfMemory, t.allocWork(g));
    try std.testing.expect(F.allocs == 1 and F.frees == 1); // the device work went back
    try std.testing.expectEqual(failing.allocations, failing.deallocations); // and the host index
}
