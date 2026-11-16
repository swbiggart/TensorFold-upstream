//! tf-inkling2-fuse: the op check of G-fuse (cuda_fuse.zig). Each fused op, in both forms (the stage-1 fusion and the
//! latency form), against the unfused sequence it replaces, on the same pseudo-random inputs at each row count, every
//! output compared byte for byte (attnIn: the mix outputs and the normed rows; mlpIn: the partial, the pushed block,
//! the mix output and the normed rows; route: ids and weights in slot order, with rows of tied scores and NaN
//! logits). Then the timing at one row: for each op, a CUDA graph of 48 layers' worth of each form (the release's
//! ops, the fusion, the latency form), replayed in rotation, us a layer; then all three ops a layer, each form.
//! Exit status: 0 every output equal, 1 otherwise or failed, 2 usage. No weights: one GPU (the first visible).
const std = @import("std");
const cuda = @import("cuda");
const ic = @import("inkling2_cuda");

const ops = ic.ops;
const res = ic.res;
const fuse = ic.fuse;
const Mat = ops.Mat;
const H = res.width;
const NE = ic.moe.experts_total;
const K8 = ic.moe.topk;

const usage =
    \\usage: tf-inkling2-fuse [--rows N[,N...]] [--reps N] [--layers N]
    \\  --rows    row counts to check (default 1,2,8,2048)
    \\  --reps    graph replays a variant for the timing (default 200; 0: no timing)
    \\  --layers  layers a timed graph holds (default 48; a global layer every 6th)
    \\  --blocks  the timed attention and MLP inputs' block count (default: the layer's, 0 .. 7 by depth)
    \\
;

const Opts = struct {
    rows: [8]usize = .{ 1, 2, 8, 2048, 0, 0, 0, 0 },
    nrows: usize = 4,
    reps: usize = 200,
    layers: usize = 48,
    blocks: ?usize = null,
};

fn parse(args: []const [:0]const u8) !?Opts {
    var o: Opts = .{};
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--rows")) {
            o.nrows = 0;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| {
                if (o.nrows == o.rows.len) return null;
                o.rows[o.nrows] = try std.fmt.parseInt(usize, x, 10);
                if (o.rows[o.nrows] == 0 or o.rows[o.nrows] > max_rows) return null;
                o.nrows += 1;
            }
        } else if (std.mem.eql(u8, k, "--reps")) o.reps = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--layers")) o.layers = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--blocks")) {
            o.blocks = try std.fmt.parseInt(usize, v, 10);
            if (o.blocks.? > 7) return null;
        } else return null;
    }
    if (i != args.len or o.nrows == 0 or o.layers == 0) return null;
    return o;
}

const max_rows = 2048;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const o = (parse(args) catch null) orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    return run(init.gpa, o) catch |e| {
        std.debug.print("FAIL tf-inkling2-fuse: {t}\n", .{e});
        return 1;
    };
}

/// bf16 bits of x, rounded to nearest even.
fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    return @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// Values like the forward's: per-row scale (so the norms differ by row), mostly normal, a few large.
fn fillRows(rng: std.Random, out: []f32, cols: usize) void {
    var r: usize = 0;
    while (r * cols < out.len) : (r += 1) {
        const scale = @exp(rng.floatNorm(f32) * 1.5);
        for (out[r * cols ..][0..cols]) |*v| {
            v.* = rng.floatNorm(f32) * scale;
            if (rng.uintLessThan(u32, 512) == 0) v.* *= 40;
        }
    }
}

const Bufs = struct {
    d: *const cuda.Driver,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: std.mem.Allocator,

    fn alloc(b: *Bufs, n: usize) !u64 {
        try b.own.append(b.gpa, try cuda.DeviceBuffer.alloc(b.d, n));
        return b.own.items[b.own.items.len - 1].ptr;
    }

    fn deinit(b: *Bufs) void {
        for (b.own.items) |*x| x.free();
        b.own.deinit(b.gpa);
    }
};

fn upload(d: *const cuda.Driver, ptr: u64, bytes: []const u8) !void {
    const view: cuda.DeviceBuffer = .{ .d = d, .ptr = ptr, .len = bytes.len, .borrowed = true };
    try view.upload(0, bytes);
}

fn uploadBf16(gpa: std.mem.Allocator, d: *const cuda.Driver, ptr: u64, xs: []const f32) !void {
    const h = try gpa.alloc(u16, xs.len);
    defer gpa.free(h);
    for (h, xs) |*y, x| y.* = bf16(x);
    try upload(d, ptr, std.mem.sliceAsBytes(h));
}

/// Device bytes [ptr, ptr + n) of x and y equal (the stream synchronized first); prints the first difference.
fn same(gpa: std.mem.Allocator, c: *const ops.Ctx, what: []const u8, x: u64, y: u64, n: usize) !bool {
    try c.stream.synchronize();
    const hx = try gpa.alloc(u8, n);
    defer gpa.free(hx);
    const hy = try gpa.alloc(u8, n);
    defer gpa.free(hy);
    try cuda.memory.download(c.d, x, hx);
    try cuda.memory.download(c.d, y, hy);
    if (std.mem.eql(u8, hx, hy)) return true;
    var at: usize = 0;
    while (hx[at] == hy[at]) at += 1;
    var nd: usize = 0;
    for (hx, hy) |p, q| nd += @intFromBool(p != q);
    std.debug.print("    {s}: {d} of {d} bytes differ, first at byte {d}\n", .{ what, nd, n, at });
    return false;
}

const Inputs = struct {
    blocks: u64, // bf16 [8][max_rows][H], the reference
    blocks_u: u64, // the unfused run's copy (pushBlock writes it)
    blocks_f: u64, // the fused run's copy
    h_in: u64,
    ac: u64,
    queries: u64, // fp32 [3][H]
    gains: [3]u64,
    mlp_gain: u64,
    mlp_query: u64,
    logits: u64, // fp32 [max_rows][NE]
    bias: u64,
    stride: usize, // bytes between blocks
};

const Outs = struct {
    x: [3]u64,
    y: [3]u64,
    part: u64,
    mx: u64,
    m: u64,
    ids: u64,
    wts: u64,
};

fn outsAlloc(b: *Bufs) !Outs {
    var o: Outs = undefined;
    for (&o.x) |*p| p.* = try b.alloc(max_rows * H * 4);
    for (&o.y) |*p| p.* = try b.alloc(max_rows * H * 2);
    o.part = try b.alloc(max_rows * H * 4);
    o.mx = try b.alloc(max_rows * H * 4);
    o.m = try b.alloc(max_rows * H * 2);
    o.ids = try b.alloc(max_rows * K8 * 4);
    o.wts = try b.alloc(max_rows * K8 * 4);
    return o;
}

fn blocksOf(in: *const Inputs, base: u64, n: usize, R: usize) ops.Blocks {
    return .{ .ptr = base, .stride = in.stride, .count = n, .rows = R, .hidden = H };
}

fn f32Rows(p: u64, R: usize) Mat {
    return Mat.of(p, R, H, .f32);
}

fn bf16Rows(p: u64, R: usize) Mat {
    return Mat.of(p, R, H, .bf16);
}

/// The unfused attention input: res.mix then nn norms (cuda_forward.zig attnStage).
fn attnUnfused(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, nb: usize, nq: usize, nn: usize, R: usize) !void {
    const xs = [3]Mat{ f32Rows(out.x[0], R), f32Rows(out.x[1], R), f32Rows(out.x[2], R) };
    try res.mix(c, blocksOf(in, in.blocks, nb, R), bf16Rows(in.h_in, R), in.queries, nq, xs[0..nq]);
    for (0..nn) |j| try res.norm(c, xs[if (nq == 1) 0 else j], in.gains[j], bf16Rows(out.y[j], R));
}

fn attnFused(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, nb: usize, nq: usize, nn: usize, R: usize) !void {
    const xs = [3]Mat{ f32Rows(out.x[0], R), f32Rows(out.x[1], R), f32Rows(out.x[2], R) };
    const ys = [3]Mat{ bf16Rows(out.y[0], R), bf16Rows(out.y[1], R), bf16Rows(out.y[2], R) };
    try fuse.attnIn(c, blocksOf(in, in.blocks, nb, R), bf16Rows(in.h_in, R), in.queries, nq, xs[0..nq], in.gains[0..nn], ys[0..nn]);
}

/// The unfused MLP input: res.partial, pushBlock's copy, res.mix, res.norm (cuda_forward.zig mlpInput).
fn mlpUnfused(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, nb: usize, boundary: bool, R: usize) !void {
    const part = f32Rows(out.part, R);
    try res.partial(c, bf16Rows(in.h_in, R), bf16Rows(in.ac, R), boundary, part);
    if (boundary) try c.copyRows(bf16Rows(in.blocks_u + nb * in.stride, R), bf16Rows(in.h_in, R));
    const n = nb + @intFromBool(boundary);
    try res.mix(c, blocksOf(in, in.blocks_u, n, R), part, in.mlp_query, 1, &.{f32Rows(out.mx, R)});
    try res.norm(c, f32Rows(out.mx, R), in.mlp_gain, bf16Rows(out.m, R));
}

fn mlpFused(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, nb: usize, boundary: bool, R: usize) !void {
    try fuse.mlpIn(c, bf16Rows(in.h_in, R), bf16Rows(in.ac, R), boundary, f32Rows(out.part, R), blocksOf(in, in.blocks_f, nb, R), in.blocks_f + nb * in.stride, in.mlp_query, f32Rows(out.mx, R), in.mlp_gain, bf16Rows(out.m, R));
}

/// The unfused router tail in checkpoint mode: select, then the slot order (cuda_moe.zig route).
fn routeUnfused(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, R: usize) !void {
    const k = &c.k.moe;
    var b: cuda.launch.Args = .{};
    b.add(in.logits);
    b.add(in.bias);
    b.add(out.ids);
    b.add(out.wts);
    b.add(@as(i32, @intCast(R)));
    try cuda.launch.launch(k.select, .{ .grid = .{ .x = @intCast((R + 7) / 8) }, .block = .{ .x = 256 } }, c.stream, &b);
    var s: cuda.launch.Args = .{};
    s.add(in.logits);
    s.add(in.bias);
    s.add(out.ids);
    s.add(out.wts);
    s.add(@as(i32, @intCast(R)));
    try cuda.launch.launch(k.ck_slot, .{ .grid = .{ .x = @intCast((R + 127) / 128) }, .block = .{ .x = 128 } }, c.stream, &s);
}

fn routeFused(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, R: usize) !void {
    try fuse.selectSlot(c, in.logits, in.bias, Mat.of(out.ids, R, K8, .i32), Mat.of(out.wts, R, K8, .f32));
}

/// The forms an op is checked and timed in: the release's ops, the stage-1 fusion, the latency form.
const Form = enum { release, fused, lat };

fn modeOf(form: Form) fuse.Mode {
    return switch (form) {
        .release => .none,
        .fused => .{ .fused = true },
        .lat => .lat,
    };
}

fn poison(c: *const ops.Ctx, out: *const Outs) !void {
    for (out.x) |p| try c.zero(p, max_rows * H * 4);
    for (out.y) |p| try c.zero(p, max_rows * H * 2);
    try c.zero(out.part, max_rows * H * 4);
    try c.zero(out.mx, max_rows * H * 4);
    try c.zero(out.m, max_rows * H * 2);
    try c.zero(out.ids, max_rows * K8 * 4);
    try c.zero(out.wts, max_rows * K8 * 4);
}

fn run(gpa: std.mem.Allocator, o: Opts) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try ic.kernels.Kernels.load(&driver);
    defer k.unload();
    var gemms: ops.Gemms = .{ .gpa = gpa };
    defer gemms.deinit();
    var scratch = try ops.Scratch.init(&driver, 64 << 20);
    defer scratch.deinit();
    const c: ops.Ctx = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch };
    var bufs: Bufs = .{ .d = &driver, .gpa = gpa };
    defer bufs.deinit();

    // inputs
    var prng = std.Random.DefaultPrng.init(0x1f05e5);
    const rng = prng.random();
    const stride = max_rows * H * 2;
    var in: Inputs = undefined;
    in.stride = stride;
    in.blocks = try bufs.alloc(8 * stride);
    in.blocks_u = try bufs.alloc(8 * stride);
    in.blocks_f = try bufs.alloc(8 * stride);
    in.h_in = try bufs.alloc(max_rows * H * 2);
    in.ac = try bufs.alloc(max_rows * H * 2);
    in.queries = try bufs.alloc(3 * H * 4);
    for (&in.gains) |*g| g.* = try bufs.alloc(H * 4);
    in.mlp_gain = try bufs.alloc(H * 4);
    in.mlp_query = try bufs.alloc(H * 4);
    in.logits = try bufs.alloc(max_rows * NE * 4);
    in.bias = try bufs.alloc(NE * 4);
    {
        const t = try gpa.alloc(f32, 8 * max_rows * H);
        defer gpa.free(t);
        fillRows(rng, t, H);
        try uploadBf16(gpa, &driver, in.blocks, t);
        fillRows(rng, t[0 .. max_rows * H], H);
        try uploadBf16(gpa, &driver, in.h_in, t[0 .. max_rows * H]);
        fillRows(rng, t[0 .. max_rows * H], H);
        try uploadBf16(gpa, &driver, in.ac, t[0 .. max_rows * H]);
        // queries and the MLP query: the checkpoint's scale (|q| about 1 a column), gains about 1
        for (t[0 .. 3 * H]) |*v| v.* = rng.floatNorm(f32) * 0.5;
        try upload(&driver, in.queries, std.mem.sliceAsBytes(t[0 .. 3 * H]));
        for (t[0..H]) |*v| v.* = rng.floatNorm(f32) * 0.5;
        try upload(&driver, in.mlp_query, std.mem.sliceAsBytes(t[0..H]));
        for ([4]u64{ in.gains[0], in.gains[1], in.gains[2], in.mlp_gain }) |g| {
            for (t[0..H]) |*v| v.* = @as(f32, @bitCast(@as(u32, bf16(1 + rng.floatNorm(f32) * 0.3)) << 16));
            try upload(&driver, g, std.mem.sliceAsBytes(t[0..H]));
        }
        // router logits: rows of plain draws, rows with many tied scores, rows with NaN logits
        const lg = t[0 .. max_rows * NE];
        for (0..max_rows) |r| {
            const row = lg[r * NE ..][0..NE];
            switch (r % 8) {
                3 => for (row) |*v| {
                    v.* = @floatFromInt(rng.uintLessThan(u32, 3));
                },
                5 => for (row, 0..) |*v, e| {
                    v.* = if (e % 37 == 0) std.math.nan(f32) else rng.floatNorm(f32) * 3;
                },
                6 => @memset(row, 0.25),
                else => for (row) |*v| {
                    v.* = rng.floatNorm(f32) * 3;
                },
            }
        }
        try upload(&driver, in.logits, std.mem.sliceAsBytes(lg));
        for (t[0..NE], 0..) |*v, e| v.* = if (e % 5 == 0) 0 else rng.floatNorm(f32) * 0.1;
        try upload(&driver, in.bias, std.mem.sliceAsBytes(t[0..NE]));
    }
    const u = try outsAlloc(&bufs);
    const f = try outsAlloc(&bufs);

    var bad: usize = 0;
    var checks: usize = 0;
    for (o.rows[0..o.nrows]) |R| for ([_]Form{ .fused, .lat }) |form| {
        k.fuse.mode = modeOf(form);
        defer k.fuse.mode = .none;
        const fname = @tagName(form);
        // attnIn: sliding (nq 1, one norm) and global (nq 3, three norms), at several block counts
        for ([_]usize{ 0, 1, 3, 8 }) |nb| for ([_][2]usize{ .{ 1, 1 }, .{ 3, 3 }, .{ 1, 3 } }) |qn| {
            try poison(&c, &u);
            try poison(&c, &f);
            try attnUnfused(&c, &in, &u, nb, qn[0], qn[1], R);
            try attnFused(&c, &in, &f, nb, qn[0], qn[1], R);
            var ok = true;
            for (0..qn[0]) |j| ok = try same(gpa, &c, "mix out", u.x[j], f.x[j], R * H * 4) and ok;
            for (0..qn[1]) |j| ok = try same(gpa, &c, "norm out", u.y[j], f.y[j], R * H * 2) and ok;
            std.debug.print("  {s} {s:<5} attn_in  rows {d}, blocks {d}, queries {d}, norms {d}\n", .{ if (ok) "equal " else "DIFFER", fname, R, nb, qn[0], qn[1] });
            bad += @intFromBool(!ok);
            checks += 1;
        };
        // mlpIn: with and without the boundary push
        for ([_]usize{ 0, 2, 7, 8 }) |nb| for ([_]bool{ false, true }) |boundary| {
            if (boundary and nb == 8) continue;
            try poison(&c, &u);
            try poison(&c, &f);
            for ([_]u64{ in.blocks_u, in.blocks_f }) |dst| try c.copy(dst, in.blocks, 8 * stride);
            try mlpUnfused(&c, &in, &u, nb, boundary, R);
            try mlpFused(&c, &in, &f, nb, boundary, R);
            var ok = try same(gpa, &c, "partial", u.part, f.part, R * H * 4);
            ok = try same(gpa, &c, "blocks", in.blocks_u, in.blocks_f, 8 * stride) and ok;
            ok = try same(gpa, &c, "mix out", u.mx, f.mx, R * H * 4) and ok;
            ok = try same(gpa, &c, "norm out", u.m, f.m, R * H * 2) and ok;
            std.debug.print("  {s} {s:<5} mlp_in   rows {d}, blocks {d}, boundary {s}\n", .{ if (ok) "equal " else "DIFFER", fname, R, nb, if (boundary) "yes" else "no" });
            bad += @intFromBool(!ok);
            checks += 1;
        };
        // route: select + slot order
        {
            try poison(&c, &u);
            try poison(&c, &f);
            try routeUnfused(&c, &in, &u, R);
            try routeFused(&c, &in, &f, R);
            var ok = try same(gpa, &c, "ids", u.ids, f.ids, R * K8 * 4);
            ok = try same(gpa, &c, "wts", u.wts, f.wts, R * K8 * 4) and ok;
            std.debug.print("  {s} {s:<5} route    rows {d} (ties every 8th row from 3 and 6, NaN logits from 5)\n", .{ if (ok) "equal " else "DIFFER", fname, R });
            bad += @intFromBool(!ok);
            checks += 1;
        }
    };
    std.debug.print("{s} tf-inkling2-fuse: {d}/{d} fused ops bit-equal to the unfused sequence (rows", .{ if (bad == 0) "PASS" else "FAIL", checks - bad, checks });
    for (o.rows[0..o.nrows]) |R| std.debug.print(" {d}", .{R});
    std.debug.print(")\n", .{});
    if (o.reps > 0) {
        for ([_]Op{ .empty, .attn, .attn_global, .mlp, .route, .layer }) |op| try timing(&c, &k, &in, &u, o, op);
    }
    return if (bad == 0) 0 else 1;
}

/// What a timed graph holds, `layers` times at one row: one op (in each form), or all of a layer's (`layer`), or an
/// empty kernel (`empty`, the floor a launch costs).
const Op = enum { empty, attn, attn_global, mlp, route, layer };

/// Layer li's launches of `op` in `form` (k.fuse.mode set by the caller), as the forward runs them at one row: a
/// global layer every 6th, a boundary every 6th (layer 0 has no attention input).
fn opLayer(c: *const ops.Ctx, in: *const Inputs, out: *const Outs, op: Op, form: Form, li: usize, blocks: ?usize) !void {
    const global = li % 6 == 5;
    const boundary = li % 6 == 0;
    const nb = blocks orelse @min(li / 6, 7);
    const unf = form == .release;
    switch (op) {
        .empty => {
            var a: cuda.launch.Args = .{};
            a.add(@as(u64, 0));
            try cuda.launch.launch(c.k.fuse.g.probe, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, c.stream, &a);
        },
        .attn => if (unf) try attnUnfused(c, in, out, nb, 1, 1, 1) else try attnFused(c, in, out, nb, 1, 1, 1),
        .attn_global => if (unf) try attnUnfused(c, in, out, nb, 3, 3, 1) else try attnFused(c, in, out, nb, 3, 3, 1),
        .mlp => if (unf) try mlpUnfused(c, in, out, nb, boundary, 1) else try mlpFused(c, in, out, nb, boundary, 1),
        .route => if (unf) try routeUnfused(c, in, out, 1) else try routeFused(c, in, out, 1),
        .layer => {
            const nq: usize = if (global) 3 else 1;
            if (li > 0) try (if (unf) attnUnfused(c, in, out, nb, nq, nq, 1) else attnFused(c, in, out, nb, nq, nq, 1));
            try (if (unf) mlpUnfused(c, in, out, nb, boundary, 1) else mlpFused(c, in, out, nb, boundary, 1));
            try (if (unf) routeUnfused(c, in, out, 1) else routeFused(c, in, out, 1));
        },
    }
}

/// One CUDA graph a form of `layers` layers' worth of `op` at one row, replayed in rotation `reps` times: launches
/// and us a layer each form.
fn timing(c: *const ops.Ctx, k: *ic.kernels.Kernels, in: *const Inputs, out: *const Outs, o: Opts, op: Op) !void {
    const forms: []const Form = switch (op) {
        .empty => &.{.release},
        else => &.{ .release, .fused, .lat },
    };
    var ex: [3]cuda.graph.Exec = undefined;
    var nodes: [3]usize = undefined;
    var made: usize = 0;
    defer for (ex[0..made]) |*x| x.deinit();
    for (forms, 0..) |form, v| {
        k.fuse.mode = modeOf(form);
        defer k.fuse.mode = .none;
        try cuda.graph.beginCapture(c.stream, .relaxed);
        const captured: anyerror!void = blk: {
            for (0..o.layers) |li| opLayer(c, in, out, op, form, li, if (op == .layer) null else o.blocks) catch |e| break :blk e;
        };
        var g = cuda.graph.endCapture(c.stream) catch |e| {
            captured catch |first| return first;
            return e;
        };
        defer g.deinit();
        try captured;
        nodes[v] = try g.nodeCount();
        ex[v] = try g.instantiate();
        made += 1;
        try ex[v].upload(c.stream);
    }
    var e0 = try cuda.Event.init(c.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(c.d, true);
    defer e1.deinit();
    var ms: [3]f64 = .{ 0, 0, 0 };
    const n = forms.len;
    for (0..o.reps + 10) |i| {
        for (0..n) |j| {
            const v = (i + j) % n;
            try e0.record(c.stream);
            try ex[v].launchOn(c.stream);
            try e1.record(c.stream);
            try e1.synchronize();
            if (i >= 10) ms[v] += try cuda.Event.elapsedMs(e0, e1);
        }
    }
    const reps: f64 = @floatFromInt(o.reps);
    const L: f64 = @floatFromInt(o.layers);
    std.debug.print("timing {s:<11} {d} layers at one row, {d} replays:", .{ @tagName(op), o.layers, o.reps });
    for (forms, 0..) |form, v| std.debug.print(" {s} {d} nodes {d:.2} us a layer;", .{ @tagName(form), nodes[v], ms[v] / reps / L * 1e3 });
    if (n == 3) std.debug.print(" lat saves {d:.2} us a layer ({d:.3} ms over the graph)", .{ (ms[0] - ms[2]) / reps / L * 1e3, (ms[0] - ms[2]) / reps });
    std.debug.print("\n", .{});
}

