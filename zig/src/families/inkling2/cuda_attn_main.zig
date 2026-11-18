//! tf-inkling2-attn: G-attn's ops alone on one GPU (cuda_attn.zig `heads` and `core`) over a whole prompt at several
//! forward sizes, from an inkling2_op_ref.py / inkling2_attn_check.py capture of the prompt from position 0.
//!
//! For layer L the capture gives torch's op inputs and outputs: attn.project in (x, or xq, xk, xv), attn.heads in
//! (q, kc, vc, rel) and out (q, k, v, rel), attn.core in (sink, gate), attn.out in (o). Each forward size N (and each
//! form of `core`) gets a fresh stream: forwards of N rows run `project` and `output` on torch's inputs, `heads` on
//! torch's inputs, write torch's k', v' to the cache (cuda_state.zig `kvWrite`) and run `core` on torch's q', rel',
//! sink and gate, so each op is measured alone and every run sees the same inputs: a row's outputs must have the
//! same bits at every N and form. Outputs go to OUT as <op>.<form>.<N>.<tensor> (op project, heads, core, out),
//! compared by tools/zig/inkling2_attn_check.py compare. --pos-offset P runs `heads` alone at
//! positions P .. P + T - 1 (log scaling active past 128,000); `core` needs a stream from position 0.
//! --op-check and --bench (cuda_attn_check.zig): `core`'s paths against each other on pseudo-random inputs.
//! --oracle (cuda_attn_oracle.zig): `core`'s forms and kx against relbias_attention_ref.zig on synthetic inputs;
//! --oracle --capture (cuda_attn_oracle_real.zig): the same on one real layer's inputs (a core-inputs file).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");
const check = @import("cuda_attn_check.zig");
const oracle = @import("cuda_attn_oracle.zig");
const oracle_real = @import("cuda_attn_oracle_real.zig");

const attn = ic.attn;
const Mat = ic.ops.Mat;
const D = ic.weights.head_dim;

const usage =
    \\usage: tf-inkling2-attn <model dir> <capture.safetensors> <out.safetensors> --layer L [options]
    \\  --chunks N,N,...   forward sizes (default 1,8,16,2048)
    \\  --forms F,F        core's forms: kx (the default), v01 (v0.1's core, its own plan), fused, split (v0.1's forced)
    \\  --rows T           the prompt's first T rows (default all the capture holds)
    \\  --pos-offset P     positions P .. P + T - 1 (default 0)
    \\  --scratch-gib G    device scratch an op may take (default 2)
    \\The GPU is the first visible device (set CUDA_VISIBLE_DEVICES).
    \\
;

const Opts = struct {
    dir: []const u8,
    cap: []const u8,
    out: []const u8,
    layer: ?u32 = null,
    chunks: []const usize = &.{ 1, 8, 16, 2048 },
    forms: []const attn.Form = &.{.kx},
    rows: usize = 0,
    pos_offset: usize = 0,
    scratch_gib: usize = 2,
};

fn parse(a: std.mem.Allocator, args: []const [:0]const u8) !?Opts {
    if (args.len < 4) return null;
    var o: Opts = .{ .dir = args[1], .cap = args[2], .out = args[3] };
    var i: usize = 4;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--layer")) {
            o.layer = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--chunks")) {
            var list: std.ArrayList(usize) = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(usize, x, 10));
            o.chunks = list.items;
        } else if (std.mem.eql(u8, k, "--forms")) {
            var list: std.ArrayList(attn.Form) = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| try list.append(a, std.meta.stringToEnum(attn.Form, x) orelse return null);
            o.forms = list.items;
        } else if (std.mem.eql(u8, k, "--rows")) {
            o.rows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--pos-offset")) {
            o.pos_offset = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--scratch-gib")) {
            o.scratch_gib = try std.fmt.parseInt(usize, v, 10);
        } else return null;
    }
    if (i != args.len or o.layer == null) return null;
    for (o.chunks) |n| if (n == 0) return null;
    return o;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--oracle") and oracle_real.wanted(args)) {
        const ro = (oracle_real.parse(a, args) catch null) orelse {
            std.debug.print("{s}", .{oracle_real.usage});
            return 2;
        };
        return oracle_real.run(init.gpa, init.io, ro) catch |e| {
            std.debug.print("FAIL tf-inkling2-attn --oracle --capture: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--oracle")) {
        const oo = (oracle.parse(a, args) catch null) orelse {
            std.debug.print("{s}", .{oracle.usage});
            return 2;
        };
        return oracle.run(init.gpa, init.io, oo) catch |e| {
            std.debug.print("FAIL tf-inkling2-attn --oracle: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len >= 2 and (std.mem.eql(u8, args[1], "--op-check") or std.mem.eql(u8, args[1], "--bench") or std.mem.eql(u8, args[1], "--limb"))) {
        const co = (check.parse(a, args) catch null) orelse {
            std.debug.print("{s}", .{check.usage});
            return 2;
        };
        return check.run(init.gpa, init.io, co) catch |e| {
            std.debug.print("FAIL tf-inkling2-attn: {t}\n", .{e});
            return 1;
        };
    }
    const o = (parse(a, args) catch null) orelse {
        std.debug.print("{s}{s}{s}{s}", .{ usage, check.usage, oracle.usage, oracle_real.usage });
        return 2;
    };
    return run(a, init.gpa, init.io, o) catch |e| {
        std.debug.print("FAIL tf-inkling2-attn: {t}\n", .{e});
        return 1;
    };
}

/// A whole-prompt device tensor read from the capture (its first T rows).
fn upload(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *const ink.st.Reader, name: []const u8, T: usize, cols: usize, own: *std.ArrayList(cuda.DeviceBuffer)) !Mat {
    const e = r.find(name) orelse {
        std.log.err("the capture has no {s}", .{name});
        return error.MissingTensor;
    };
    if (e.dtype != .bf16 or e.rank != 2 or e.dim(1) != cols or e.dim(0) < T) return error.UnexpectedCaptureTensor;
    const host = try gpa.alloc(u8, T * cols * 2);
    defer gpa.free(host);
    try r.readInto(e, 0, host);
    var b = try cuda.DeviceBuffer.fromHost(d, host);
    errdefer b.free();
    try own.append(gpa, b);
    return Mat.of(b.ptr, T, cols, .bf16);
}

fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, rows: usize, cols: usize, own: *std.ArrayList(cuda.DeviceBuffer)) !Mat {
    var b = try cuda.DeviceBuffer.alloc(d, rows * cols * 2);
    errdefer b.free();
    try b.fill8(0, null);
    try own.append(gpa, b);
    return Mat.of(b.ptr, rows, cols, .bf16);
}

fn rowsOf(r: *const ink.st.Reader, name: []const u8) ?usize {
    const e = r.find(name) orelse return null;
    return @intCast(e.dim(0));
}

fn run(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    const clock = std.Io.Clock.awake;
    const L = o.layer.?;
    var cap = try ink.st.Reader.open(gpa, io, o.cap);
    defer cap.close();
    var d: ink.Diag = .{};
    var model = ink.describe.Model.fromDir(gpa, io, o.dir, .{ .world = 1 }, &d) catch |e| {
        std.debug.print("FAIL tf-inkling2-attn: {t}: {s}\n", .{ e, d.text() });
        return 1;
    };
    defer model.deinit();
    var nb: [96]u8 = undefined;
    const avail = rowsOf(&cap, try std.fmt.bufPrint(&nb, "op.{d}.attn.heads.in.q", .{L})) orelse {
        std.debug.print("the capture has no layer {d} attention\n", .{L});
        return 2;
    };
    const T = if (o.rows == 0) avail else @min(o.rows, avail);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try ic.kernels.Kernels.load(&driver);
    defer k.unload();
    var gemms = try ic.ops.Gemms.init(gpa, &driver);
    defer gemms.deinit();
    var scratch = try ic.ops.Scratch.init(&driver, o.scratch_gib << 30);
    defer scratch.deinit();
    var t0 = clock.now(io).toNanoseconds();
    var w = try ic.weights.load(gpa, io, &driver, o.dir, &model, .{ .lo = L, .hi = L + 1 }, &ic.forward.prepare);
    defer w.deinit();
    const l = w.layer(L);
    std.debug.print("layer {d} ({s}) loaded in {d:.1} s; {d} rows at positions {d}..{d}\n", .{ L, if (l.global) "global" else "sliding", secs(clock.now(io).toNanoseconds() - t0), T, o.pos_offset, o.pos_offset + T - 1 });
    const hq: usize = l.hq;
    const hk: usize = l.hk;
    const dr = ic.weights.d_rel;
    var own: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (own.items) |*b| b.free();
        own.deinit(gpa);
    }
    const p = "op.{d}.attn.{s}.{s}.{s}";
    const in_q = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "in", "q" }), T, hq * D, &own);
    const in_kc = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "in", "kc" }), T, hk * D, &own);
    const in_vc = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "in", "vc" }), T, hk * D, &own);
    const in_rel = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "in", "rel" }), T, hq * dr, &own);
    const t_q = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "out", "q" }), T, hq * D, &own);
    const t_k = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "out", "k" }), T, hk * D, &own);
    const t_v = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "out", "v" }), T, hk * D, &own);
    const t_rel = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "heads", "out", "rel" }), T, hq * dr, &own);
    const sink = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "core", "in", "sink" }), T, hq, &own);
    const gate = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "core", "in", "gate" }), T, hq * D, &own);
    const hd: attn.Heads = .{ .q = try alloc(gpa, &driver, T, hq * D, &own), .k = try alloc(gpa, &driver, T, hk * D, &own), .v = try alloc(gpa, &driver, T, hk * D, &own), .rel = try alloc(gpa, &driver, T, hq * dr, &own) };
    const o_out = try alloc(gpa, &driver, T, hq * D, &own);
    // the projections: attn.project on torch's normed sources, attn.out on torch's o
    const H = ic.weights.hidden;
    const src_names: []const []const u8 = if (l.global) &.{ "xq", "xk", "xv" } else &.{"x"};
    var xs: [3]Mat = undefined;
    for (src_names, 0..) |n, i| xs[i] = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "project", "in", n }), T, H, &own);
    if (!l.global) xs[1] = xs[0];
    if (!l.global) xs[2] = xs[0];
    const in_o = try upload(gpa, &driver, &cap, try std.fmt.bufPrint(&nb, p, .{ L, "out", "in", "o" }), T, hq * D, &own);
    const taps = ic.weights.taps;
    const pj: attn.Proj = .{
        .q = try alloc(gpa, &driver, T, hq * D, &own),
        .k = try alloc(gpa, &driver, T, hk * D, &own),
        .v = try alloc(gpa, &driver, T, hk * D, &own),
        .rel = try alloc(gpa, &driver, T, hq * dr, &own),
        .gate = try alloc(gpa, &driver, T, hq * D, &own),
        .sink = try alloc(gpa, &driver, T, hq, &own),
        .dk = if (l.global) try alloc(gpa, &driver, T, hk * taps, &own) else .{},
        .dv = if (l.global) try alloc(gpa, &driver, T, hk * taps, &own) else .{},
    };
    const ao = try alloc(gpa, &driver, T, H, &own);
    // the cache: room for every position (global) or the ring the forward needs (sliding), as cuda_state.zig sizes it
    const kv_max = try alloc(gpa, &driver, T, hk * D, &own);
    const vv_max = try alloc(gpa, &driver, T, hk * D, &own);
    const c: ic.ops.Ctx = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch };
    var wr = ink.st.Writer.init(gpa);
    defer wr.deinit();
    for (o.forms) |form| {
        k.attn.form = form;
        for (o.chunks) |n0| {
            const N = @min(n0, T);
            const slots = if (l.global) T else @min(T, attn.window - 1 + N);
            const kv: attn.Kv = .{ .k = kv_max.ptr, .v = vv_max.ptr, .slots = slots, .heads = hk, .window = if (l.global) 0 else attn.window };
            try c.zero(kv.k, T * hk * D * 2);
            try c.zero(kv.v, T * hk * D * 2);
            try c.zero(o_out.ptr, o_out.bytes());
            try stream.synchronize();
            var heads_ns: i96 = 0;
            var proj_ns: i96 = 0;
            var core_ns: i96 = 0;
            var start: usize = 0;
            var forwards: usize = 0;
            while (start < T) : (start += N) {
                const R = @min(N, T - start);
                const pos0 = o.pos_offset + start;
                scratch.reset();
                t0 = clock.now(io).toNanoseconds();
                try attn.heads(&c, l, pos0, null, in_q.slice(start, R), in_kc.slice(start, R), in_vc.slice(start, R), in_rel.slice(start, R), .{ .q = hd.q.slice(start, R), .k = hd.k.slice(start, R), .v = hd.v.slice(start, R), .rel = hd.rel.slice(start, R) });
                try stream.synchronize();
                const t1 = clock.now(io).toNanoseconds();
                heads_ns += t1 - t0;
                const sl = struct {
                    fn of(m: Mat, a0: usize, n: usize) Mat {
                        return if (m.rows == 0) m else m.slice(a0, n);
                    }
                };
                try attn.project(&c, l, xs[0].slice(start, R), xs[1].slice(start, R), xs[2].slice(start, R), .{
                    .q = pj.q.slice(start, R),
                    .k = pj.k.slice(start, R),
                    .v = pj.v.slice(start, R),
                    .rel = pj.rel.slice(start, R),
                    .gate = pj.gate.slice(start, R),
                    .sink = pj.sink.slice(start, R),
                    .dk = sl.of(pj.dk, start, R),
                    .dv = sl.of(pj.dv, start, R),
                });
                try attn.output(&c, l, in_o.slice(start, R), ao.slice(start, R));
                try stream.synchronize();
                proj_ns += clock.now(io).toNanoseconds() - t1;
                scratch.reset();
                forwards += 1;
                // a stream at an offset has no keys before it: only `heads` (the log scaling) is tested there
                if (o.pos_offset > 0) continue;
                try ic.state.kvWrite(&c, kv, pos0, null, t_k.slice(start, R), t_v.slice(start, R));
                try stream.synchronize();
                const t2 = clock.now(io).toNanoseconds();
                try attn.core(&c, l, pos0, null, t_q.slice(start, R), t_rel.slice(start, R), sink.slice(start, R), gate.slice(start, R), kv, o_out.slice(start, R));
                try stream.synchronize();
                core_ns += clock.now(io).toNanoseconds() - t2;
            }
            const used = attn.formOf(form, @min(N, T), attn.splits(kv, o.pos_offset + T - @min(N, T), @min(N, T)).count, hq);
            std.debug.print("form {t} (last forward {t}), forwards of {d}: {d} forwards; project+out {d:.3} ms, heads {d:.3} ms, core {d:.3} ms ({d:.3} ms a forward); scratch peak {d} MiB\n", .{ form, used, N, forwards, ms(proj_ns), ms(heads_ns), ms(core_ns), ms(core_ns) / @as(f64, @floatFromInt(forwards)), scratch.peak >> 20 });
            const outs = [_]struct { []const u8, []const u8, Mat }{
                .{ "heads", "q", hd.q },       .{ "heads", "k", hd.k },         .{ "heads", "v", hd.v },         .{ "heads", "rel", hd.rel },
                .{ "core", "o", o_out },       .{ "project", "q", pj.q },       .{ "project", "k", pj.k },       .{ "project", "v", pj.v },
                .{ "project", "rel", pj.rel }, .{ "project", "gate", pj.gate }, .{ "project", "sink", pj.sink }, .{ "project", "dk", pj.dk },
                .{ "project", "dv", pj.dv },   .{ "out", "ao", ao },
            };
            for (outs) |x| {
                if (o.pos_offset > 0 and std.mem.eql(u8, x[0], "core")) continue;
                if (x[2].rows == 0) continue;
                const host = try gpa.alloc(u8, x[2].bytes());
                try cuda.memory.download(&driver, x[2].ptr, host);
                const name = try std.fmt.allocPrint(a, "{s}.{t}.{d}.{s}", .{ x[0], form, N, x[1] });
                try wr.addOwned(name, .bf16, &.{ x[2].rows, x[2].cols }, host);
            }
        }
    }
    try wr.setMeta("tool", "tf-inkling2-attn");
    try wr.setMeta("layer", try std.fmt.allocPrint(a, "{d}", .{L}));
    try wr.setMeta("rows", try std.fmt.allocPrint(a, "{d}", .{T}));
    try wr.setMeta("pos_offset", try std.fmt.allocPrint(a, "{d}", .{o.pos_offset}));
    try wr.setMeta("capture", std.fs.path.basename(o.cap));
    try wr.write(io, o.out);
    std.debug.print("wrote {s}\n", .{o.out});
    k.attn.printExceptions(&driver);
    return 0;
}

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-9;
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}
