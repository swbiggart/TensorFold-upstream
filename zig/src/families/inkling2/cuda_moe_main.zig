//! tf-inkling2-moe: G-moe's two ops (moe.route, moe.experts; cuda_moe.zig) alone on one layer's real weights and
//! any number of input rows, at several forward sizes. It checks that a row's bits do not depend on how many rows
//! a call holds (a decode row, 8- and 16-row windows, a whole 2,048-row prompt chunk) or on the kernels' row
//! tiles, and writes the outputs for tools/zig/inkling2_moe_check.py, which compares them with torch.
//!
//! Input (safetensors): m bf16 [N, 4608] (the mlp norm's output rows); optional ids i32 [N, 8] and w f32 [N, 8],
//! torch's routes, which a second experts pass takes (the experts op on torch's exact inputs, as under --feed).
//! Output: ids, w, logits, y (our routes, our experts on them) and y_fed (our experts on torch's routes), from the
//! first chunk size; the other chunk sizes are compared with it here, bit for bit.
//!
//! --mode checkpoint runs the checkpoint-mode experts (cuda_moe.zig; ids and w then leave in slot order), --world W
//! --rank R loads that rank's slice of the layer (its experts [lo, hi): y is then the rank's partial, bf16), and
//! --reps N repeats every forward size N times and reports the fastest pass, timed by CUDA events around each
//! experts call (route not included).
//!
//! --split W (checkpoint mode): split experts (plan.Layout.split), all W ranks' slices of the layer on this card as
//! virtual ranks: every rank's splitUp, then (MoE-A: the shared buffer holds every rank's h) every rank's splitDown,
//! then the columns (MoE-B) into y. The time reported is one rank's own work (after that pass, rank 0's two halves and
//! the columns layout again, alone between two CUDA events), the exchanges excluded; with --slowest 1 every rank's is
//! timed alone and the slowest rank's time a call is reported too (whole experts, --world W: every rank's slice is
//! loaded and its experts call timed alone; y stays rank 0's partial). --against FILE compares y (and y_fed) with
//! another run's output (W = 1 checkpoint mode: the split must be bit-equal to it). --split W --only R loads only rank
//! R's slice (a Spark's rank folder holds only its own) and times its two halves and the columns alone, once a call,
//! its weights cold as in a decode step; the other ranks' h and y are zero, so y is not the layer's output (it still
//! has a row's bits fixed for the invariance checks). --rows N takes the input's first N rows.
//!
//! --force takes a list (--force 0,1,3): every forward size runs with each shape in turn. Every pass of a repeat runs
//! each configuration once, in order, so the variants alternate and drift on the card spreads over all of them.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");

const moe = ic.moe;
const Mat = ic.ops.Mat;
const H = ic.weights.hidden;
const K8 = moe.topk;
const NE = moe.experts_total;

const usage =
    \\usage: tf-inkling2-moe <model dir> <input.safetensors> <out.safetensors> --layer L [options]
    \\  --chunks A,B,..   forward sizes to run (0: all rows in one; default 0,1,8,16); the first is written
    \\  --rt both         also run the whole input with every warp shape (full: 1 and 4 row tiles; checkpoint:
    \\                    16x8, 16x32, 64x32, 64x128 staged and the window tiles 4..): they must agree
    \\  --mode M          full (default: TF_INK2_MODE, else full) or checkpoint
    \\  --world W         load rank --rank's slice of W (default 1, rank 0)
    \\  --rank R
    \\  --reps N          each forward size N times, the fastest pass reported (default 1)
    \\  --force S,..      every forward size with row tiles S (full) or unit shape S (checkpoint: 0-5, or U:D for the
    \\                    up and down GEMMs); a list runs
    \\                    each in turn, alternating within every repeat
    \\  --split W         split experts over W virtual ranks (checkpoint mode; every rank's slice loaded)
    \\  --only R          with --split: only rank R's slice, run as every virtual rank (its time; y not the layer's)
    \\  --rows N          the input's first N rows (default all)
    \\  --fed 0           ignore the input's torch routes (no y_fed pass)
    \\  --against FILE    compare y (and y_fed) row by row with FILE's (another run's output), bit for bit
    \\  --slowest 1       time every rank's work alone (--split W, or --world W --rank 0) and report the slowest
    \\  --scratch-gib G   device scratch (default 2)
    \\  --k1 1|both       split experts in K1's forms (cuda_moe.zig Form.k1: one launch for the plan and the tokens'
    \\                    quantization, the fold inside the down GEMM, K1's shapes) for calls of 1-8 rows
    \\The GPU is the first visible device (set CUDA_VISIBLE_DEVICES).
    \\
;

const Opts = struct {
    dir: []const u8,
    input: []const u8,
    out: []const u8,
    layer: ?u32 = null,
    chunks: []const usize = &.{ 0, 1, 8, 16 },
    rt_both: bool = false,
    scratch_gib: usize = 2,
    mode: ?moe.Mode = null,
    world: u32 = 1,
    rank: u32 = 0,
    reps: usize = 1,
    /// every forward size with each of these row-tile counts (full) or shapes (checkpoint); empty: the default
    force: []const u32 = &.{},
    /// split experts over this many virtual ranks (0: whole experts)
    split: u32 = 0,
    /// --split: only this rank's slice, as every virtual rank
    only: ?u32 = null,
    /// --k1 1: the split experts' K1 forms; 2 (both): each configuration also in K1's forms
    k1: u8 = 0,
    /// the input's first rows (null: all)
    rows: ?usize = null,
    /// run the input's torch routes too when it has them
    fed: bool = true,
    against: ?[]const u8 = null,
    slowest: bool = false,
};

fn parse(a: std.mem.Allocator, args: []const [:0]const u8) !?Opts {
    if (args.len < 4) return null;
    var o: Opts = .{ .dir = args[1], .input = args[2], .out = args[3] };
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
            if (list.items.len == 0) return null;
            o.chunks = list.items;
        } else if (std.mem.eql(u8, k, "--rt")) {
            o.rt_both = std.mem.eql(u8, v, "both");
        } else if (std.mem.eql(u8, k, "--scratch-gib")) {
            o.scratch_gib = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--mode")) {
            o.mode = std.meta.stringToEnum(moe.Mode, v) orelse return null;
        } else if (std.mem.eql(u8, k, "--world")) {
            o.world = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--rank")) {
            o.rank = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--force")) {
            var list: std.ArrayList(u32) = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| {
                // checkpoint: U:D forces the up and down GEMMs' shapes apart
                const colon = std.mem.indexOfScalar(u8, x, ':');
                const up = try std.fmt.parseInt(u32, x[0 .. colon orelse x.len], 10);
                try list.append(a, if (colon) |c| moe.ckPair(up, try std.fmt.parseInt(u32, x[c + 1 ..], 10)) else up);
            }
            o.force = list.items;
        } else if (std.mem.eql(u8, k, "--split")) {
            o.split = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--k1")) {
            o.k1 = if (std.mem.eql(u8, v, "both")) 2 else if (std.mem.eql(u8, v, "1")) 1 else if (std.mem.eql(u8, v, "0")) 0 else return error.BadK1;
        } else if (std.mem.eql(u8, k, "--only")) {
            o.only = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--fed")) {
            o.fed = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "--rows")) {
            o.rows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--against")) {
            o.against = v;
        } else if (std.mem.eql(u8, k, "--slowest")) {
            o.slowest = std.mem.eql(u8, v, "1");
        } else if (std.mem.eql(u8, k, "--reps")) {
            o.reps = @max(1, try std.fmt.parseInt(usize, v, 10));
        } else return null;
    }
    if (i != args.len or o.layer == null or o.rank >= o.world or (o.only != null and (o.split < 2 or o.only.? >= o.split or o.slowest or o.against != null)) or o.split == 1 or (o.split > 1 and o.world != 1) or (o.slowest and o.split == 0 and (o.world == 1 or o.rank != 0))) return null;
    return o;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const o = (parse(a, args) catch null) orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    return run(a, init.gpa, init.io, o) catch |e| {
        std.debug.print("FAIL tf-inkling2-moe: {t}\n", .{e});
        return 1;
    };
}

/// One run's outputs on the host.
const Host = struct {
    ids: []u8,
    w: []u8,
    logits: []u8,
    y: []u8,
    yf: []u8,

    fn alloc(a: std.mem.Allocator, n: usize, fed: bool) !Host {
        return .{
            .ids = try a.alloc(u8, n * K8 * 4),
            .w = try a.alloc(u8, n * K8 * 4),
            .logits = try a.alloc(u8, n * NE * 4),
            .y = try a.alloc(u8, n * H * 2),
            .yf = if (fed) try a.alloc(u8, n * H * 2) else &.{},
        };
    }
};

/// Device buffers of N rows.
const Bufs = struct {
    m: Mat,
    ids: Mat,
    w: Mat,
    logits: Mat,
    y: Mat,
    fids: Mat,
    fw: Mat,
    yf: Mat,
};

/// The mean number of distinct experts a call of C rows routes to (ids i32 [n, 8]): the weights a call reads.
fn distinct(ids: []const u8, n: usize, C: usize) f64 {
    var total: usize = 0;
    var calls: usize = 0;
    var start: usize = 0;
    while (start < n) : (start += C) {
        var seen: std.StaticBitSet(NE) = .empty;
        for (start..@min(start + C, n)) |r| for (0..K8) |j| {
            const e = std.mem.readInt(i32, ids[(r * K8 + j) * 4 ..][0..4], .little);
            if (e >= 0 and e < NE) seen.set(@intCast(e));
        };
        total += seen.count();
        calls += 1;
    }
    return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(calls));
}

fn rowsDiffer(x: []const u8, y: []const u8, n: usize) usize {
    const per = x.len / n;
    var bad: usize = 0;
    for (0..n) |r| bad += @intFromBool(!std.mem.eql(u8, x[r * per ..][0..per], y[r * per ..][0..per]));
    return bad;
}

fn run(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    const clock = std.Io.Clock.awake;
    const L = o.layer.?;
    var input = try ink.st.Reader.open(gpa, io, o.input);
    defer input.close();
    const em = input.find("m") orelse return error.InputHasNoM;
    if (em.dtype != .bf16 or em.rank != 2 or em.dim(1) != H) return error.BadInput;
    const N: usize = @min(@as(usize, @intCast(em.dim(0))), o.rows orelse std.math.maxInt(usize));
    if (N == 0) return error.BadInput;
    const fed = o.fed and input.find("ids") != null and input.find("w") != null;
    var d: ink.Diag = .{};
    var model = ink.describe.Model.fromDir(gpa, io, o.dir, if (o.split > 1) .{ .world = o.split, .layout = .split } else .{ .world = o.world }, &d) catch |e| {
        std.debug.print("FAIL tf-inkling2-moe: {t}: {s}\n", .{ e, d.text() });
        return 1;
    };
    defer model.deinit();
    if (L >= model.c.text.layers) return error.BadLayer;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try ic.kernels.Kernels.load(&driver);
    defer k.unload();
    if (o.mode) |md| k.moe.mode = md;
    const ckm = k.moe.mode == .checkpoint;
    if (o.split > 1 and !ckm) {
        std.debug.print("--split runs in checkpoint mode only (--mode checkpoint)\n", .{});
        return 2;
    }
    var gemms: ic.ops.Gemms = .{ .gpa = gpa }; // no cuBLASLt: these ops do not use it
    defer gemms.deinit();
    var scratch = try ic.ops.Scratch.init(&driver, o.scratch_gib << 30);
    defer scratch.deinit();
    var t0 = clock.now(io).toNanoseconds();
    const prep = [_]ic.weights.Prepare{moe.prepare};
    var w = try ic.weights.load(gpa, io, &driver, o.dir, &model, .{ .rank = o.only orelse o.rank, .lo = L, .hi = L + 1 }, &prep);
    defer w.deinit();
    const l = w.layer(L);
    // --split or --slowest: ranks 1 .. W-1's slices too (rank 0's is `w`)
    const nranks: u32 = if (o.split > 1) o.split else if (o.slowest) o.world else 1;
    // --split W --only R: rank R's slice is every virtual rank
    const one_rank = o.only != null;
    var others: [ink.plan.max_world]ic.weights.Weights = undefined;
    var nothers: usize = 0;
    defer for (others[0..nothers]) |*x| x.deinit();
    var split_layers: [ink.plan.max_world]*const ic.weights.Layer = undefined;
    split_layers[0] = l;
    for (1..nranks) |r| {
        if (one_rank) {
            split_layers[r] = l;
            continue;
        }
        others[r - 1] = try ic.weights.load(gpa, io, &driver, o.dir, &model, .{ .rank = @intCast(r), .lo = L, .hi = L + 1 }, &prep);
        nothers += 1;
        split_layers[r] = others[r - 1].layer(L);
    }
    var held: u64 = w.bytes;
    for (others[0..nothers]) |*x| held += x.bytes;
    std.debug.print("layer {d} ({s}, {s} experts {d}..{d}, rank {d} of {d}{s}, {s} mode) loaded: {d:.2} GiB in {d:.1} s; {d} input rows{s}\n", .{ L, if (l.global) "global" else "sliding", @tagName(l.experts.format), l.experts.lo, l.experts.hi - 1, o.only orelse o.rank, @max(o.world, o.split), if (o.split > 1) try std.fmt.allocPrint(a, ", split over {d} virtual ranks (w13 rows {d}, w2 rows {d} of each expert a rank){s}", .{ o.split, l.experts.n13, l.experts.n2, if (one_rank) "; only this rank's slice, as every virtual rank" else "" }) else "", @tagName(k.moe.mode), @as(f64, @floatFromInt(held)) / (1 << 30), secs(clock.now(io).toNanoseconds() - t0), N, if (fed) ", torch routes given" else "" });
    var ev0 = try cuda.Event.init(&driver, true);
    defer ev0.deinit();
    var ev1 = try cuda.Event.init(&driver, true);
    defer ev1.deinit();

    // device buffers
    var own: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer for (own.items) |*b| b.free();
    const sizes = [_]usize{ N * H * 2, N * K8 * 4, N * K8 * 4, N * NE * 4, N * H * 2, N * K8 * 4, N * K8 * 4, N * H * 2 };
    var ptr: [sizes.len]u64 = undefined;
    for (sizes, 0..) |n, i| {
        try own.append(a, try cuda.DeviceBuffer.alloc(&driver, n));
        ptr[i] = own.items[i].ptr;
    }
    const B: Bufs = .{
        .m = Mat.of(ptr[0], N, H, .bf16),
        .ids = Mat.of(ptr[1], N, K8, .i32),
        .w = Mat.of(ptr[2], N, K8, .f32),
        .logits = Mat.of(ptr[3], N, NE, .f32),
        .y = Mat.of(ptr[4], N, H, .bf16),
        .fids = Mat.of(ptr[5], N, K8, .i32),
        .fw = Mat.of(ptr[6], N, K8, .f32),
        .yf = Mat.of(ptr[7], N, H, .bf16),
    };
    {
        const host = try gpa.alloc(u8, N * H * 2);
        defer gpa.free(host);
        try input.readInto(em, 0, host);
        try own.items[0].upload(0, host);
        if (fed) {
            for ([_][]const u8{ "ids", "w" }, [_]usize{ 5, 6 }, [_]ink.st.DType{ .i32, .f32 }) |name, bi, dt| {
                const e = input.find(name).?;
                if (e.dtype != dt or e.rank != 2 or e.dim(0) < N or e.dim(1) != K8) return error.BadInput;
                const hb = try gpa.alloc(u8, N * K8 * 4);
                defer gpa.free(hb);
                try input.readInto(e, 0, hb);
                try own.items[bi].upload(0, hb);
            }
        }
    }
    const c: ic.ops.Ctx = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch };
    var sp: ?Virt = null;
    if (nranks > 1) {
        const split_sizes = [4]usize{ if (o.split > 1) N * K8 * moe.intermediate * 2 else 0, N * H * 2, if (o.split > 1) moe.splitPlanBytes(N) * nranks else 0, N * H * 2 };
        var sb: [4]u64 = undefined;
        for (split_sizes, &sb) |n, *x| {
            try own.append(a, try cuda.DeviceBuffer.alloc(&driver, n));
            x.* = own.items[own.items.len - 1].ptr;
        }
        sp = .{ .world = nranks, .split = o.split > 1, .layers = split_layers[0..nranks], .h = sb[0], .y = sb[1], .plan = sb[2], .plan_bytes = if (o.split > 1) moe.splitPlanBytes(N) else 0, .y_rerun = sb[3], .one = one_rank };
    }
    var ev2 = try cuda.Event.init(&driver, true);
    defer ev2.deinit();
    var ev3 = try cuda.Event.init(&driver, true);
    defer ev3.deinit();
    const evs = [4]*cuda.Event{ &ev0, &ev1, &ev2, &ev3 };

    var first: ?Host = null;
    var bad_total: usize = 0;
    const Cfg = struct { chunk: usize, rt: ?u32, k1: bool = false, route_ns: i96 = 0, exp_ns: i96 = 0, max_ns: i96 = 0, fed_ns: i96 = 0 };
    var configs: std.ArrayList(Cfg) = .empty;
    // a checkpoint shape that cannot run this layer's GEMMs (moe.ckFits) is skipped
    const nvl = l.experts.format == .nvfp4;
    for (o.force) |sh| if (ckm and !moe.ckFits(sh, nvl)) std.debug.print("shape {d} skipped: it does not fit this layer's GEMMs\n", .{sh});
    for (o.chunks) |ch| {
        if (o.force.len == 0) try configs.append(a, .{ .chunk = ch, .rt = null });
        for (o.force) |sh| if (!ckm or moe.ckFits(sh, nvl)) try configs.append(a, .{ .chunk = ch, .rt = sh });
    }
    if (o.rt_both) {
        if (ckm) {
            for (0..moe.ck_shapes) |sh| if (moe.ckFits(@intCast(sh), nvl)) try configs.append(a, .{ .chunk = 0, .rt = @intCast(sh) });
        } else {
            try configs.append(a, .{ .chunk = 0, .rt = 1 });
            try configs.append(a, .{ .chunk = 0, .rt = 4 });
        }
    }
    // --k1: every configuration in K1's forms (1), or each followed by its K1 twin (2)
    if (o.k1 == 1) {
        for (configs.items) |*cfg| cfg.k1 = true;
    } else if (o.k1 == 2) {
        var both: std.ArrayList(Cfg) = .empty;
        for (configs.items) |cfg| {
            try both.append(a, cfg);
            var t = cfg;
            t.k1 = true;
            try both.append(a, t);
        }
        configs = both;
    }
    // every repeat runs each configuration once, in order (the variants alternate); the last one checks the outputs
    for (0..o.reps) |rep| for (configs.items) |*cfg| {
        const C = if (cfg.chunk == 0) N else @min(cfg.chunk, N);
        var r_ns: i96 = 0;
        var e_ns: i96 = 0;
        var x_ns: i96 = 0;
        var f_ns: i96 = 0;
        var start: usize = 0;
        while (start < N) : (start += C) {
            const R = @min(C, N - start);
            scratch.reset();
            t0 = clock.now(io).toNanoseconds();
            try moe.route(&c, l, B.m.slice(start, R), B.ids.slice(start, R), B.w.slice(start, R), B.logits.slice(start, R));
            try stream.synchronize();
            r_ns += clock.now(io).toNanoseconds() - t0;
            scratch.reset();
            const t = try timed(&c, evs, l, sp, B.m.slice(start, R), B.ids.slice(start, R), B.w.slice(start, R), B.y.slice(start, R), cfg.rt, .{ .k1 = cfg.k1 });
            e_ns += t[0];
            x_ns += t[1];
            if (fed) {
                scratch.reset();
                f_ns += (try timed(&c, evs, l, sp, B.m.slice(start, R), B.fids.slice(start, R), B.fw.slice(start, R), B.yf.slice(start, R), cfg.rt, .{ .k1 = cfg.k1 }))[0];
            }
        }
        if (rep == 0 or e_ns < cfg.exp_ns) {
            cfg.route_ns = r_ns;
            cfg.exp_ns = e_ns;
            cfg.fed_ns = f_ns;
        }
        if (rep == 0 or x_ns < cfg.max_ns) cfg.max_ns = x_ns;
        if (rep + 1 < o.reps) continue;
        const h = try Host.alloc(a, N, fed);
        try own.items[1].download(0, h.ids);
        try own.items[2].download(0, h.w);
        try own.items[3].download(0, h.logits);
        try own.items[4].download(0, h.y);
        if (fed) try own.items[7].download(0, h.yf);
        const calls = (N + C - 1) / C;
        const shape_names = [_][]const u8{ ", 16x8 a warp", ", 16x32 a warp", ", 64x32 a warp", ", 64x128 staged" };
        const tag = if (cfg.rt) |r| (if (ckm) (if (r >> 16 != 0) try std.fmt.allocPrint(a, ", shapes {d}:{d}", .{ moe.ckUnpair(r)[0], moe.ckUnpair(r)[1] }) else if (r < shape_names.len) shape_names[r] else try std.fmt.allocPrint(a, ", shape {d}", .{r})) else if (r == 1) ", 1 row tile" else ", 4 row tiles") else "";
        std.debug.print("forwards of {d} rows ({d} calls{s}{s}): route {d:.3} ms, experts {d:.4} ms a call (mean, CUDA events, best of {d}); scratch peak {d} MiB; {d:.2} distinct experts a call", .{ C, calls, tag, if (cfg.k1) ", k1" else "", ms(cfg.route_ns) / @as(f64, @floatFromInt(calls)), ms(cfg.exp_ns) / @as(f64, @floatFromInt(calls)), o.reps, scratch.peak >> 20, distinct(h.ids, N, C) });
        if (o.slowest) std.debug.print(", the slowest of {d} ranks {d:.4} ms a call", .{ nranks, ms(cfg.max_ns) / @as(f64, @floatFromInt(calls)) });
        if (fed) std.debug.print(", experts on torch's routes {d:.3} ms", .{ms(cfg.fed_ns) / @as(f64, @floatFromInt(calls))});
        std.debug.print("\n", .{});
        if (first) |f| {
            const bi = rowsDiffer(h.ids, f.ids, N);
            const bw = rowsDiffer(h.w, f.w, N);
            const bl = rowsDiffer(h.logits, f.logits, N);
            const by = rowsDiffer(h.y, f.y, N);
            const bf = if (fed) rowsDiffer(h.yf, f.yf, N) else 0;
            bad_total += bi + bw + bl + by + bf;
            std.debug.print("  against the first: rows differing ids {d}, w {d}, logits {d}, y {d}, y_fed {d} (of {d}) {s}\n", .{ bi, bw, bl, by, bf, N, if (bi + bw + bl + by + bf == 0) "BIT-EQUAL" else "DIFFER" });
        } else first = h;
    };
    const f = first.?;
    if (o.against) |path| {
        var ref = try ink.st.Reader.open(gpa, io, path);
        defer ref.close();
        for ([_][]const u8{ "ids", "w", "y", "y_fed" }, [_][]u8{ f.ids, f.w, f.y, f.yf }) |name, mine| {
            if (mine.len == 0) continue;
            const e = ref.find(name) orelse return error.AgainstHasNoTensor;
            if (e.dim(0) != N) return error.AgainstRows;
            const theirs = try a.alloc(u8, mine.len);
            try ref.readInto(e, 0, theirs);
            const bad = rowsDiffer(mine, theirs, N);
            bad_total += bad;
            std.debug.print("against {s}: {s} rows differing {d} of {d} {s}\n", .{ std.fs.path.basename(path), name, bad, N, if (bad == 0) "BIT-EQUAL" else "DIFFER" });
        }
    }
    var out = ink.st.Writer.init(gpa);
    defer out.deinit();
    try out.add("ids", .i32, &.{ N, K8 }, f.ids);
    try out.add("w", .f32, &.{ N, K8 }, f.w);
    try out.add("logits", .f32, &.{ N, NE }, f.logits);
    try out.add("y", .bf16, &.{ N, H }, f.y);
    if (fed) try out.add("y_fed", .bf16, &.{ N, H }, f.yf);
    try out.setMeta("tool", "tf-inkling2-moe");
    try out.setMeta("layer", try std.fmt.allocPrint(a, "{d}", .{L}));
    try out.setMeta("format", @tagName(l.experts.format));
    try out.setMeta("mode", @tagName(k.moe.mode));
    try out.setMeta("world", try std.fmt.allocPrint(a, "{d}", .{o.world}));
    try out.setMeta("split", try std.fmt.allocPrint(a, "{d}", .{o.split}));
    try out.setMeta("rank", try std.fmt.allocPrint(a, "{d}", .{o.rank}));
    try out.setMeta("experts", try std.fmt.allocPrint(a, "{d}:{d}", .{ l.experts.lo, l.experts.hi }));
    try out.setMeta("chunk", try std.fmt.allocPrint(a, "{d}", .{if (o.chunks[0] == 0) N else o.chunks[0]}));
    try out.write(io, o.out);
    std.debug.print("{s}: {d} rows written to {s}; row invariance{s} {s}\n", .{ if (bad_total == 0) "PASS" else "FAIL", N, o.out, if (o.against != null) " and the --against rows" else "", if (bad_total == 0) "bit-equal" else "BROKEN" });
    return if (bad_total == 0) 0 else 1;
}

/// Virtual ranks: every rank's layer slice; split experts' shared exchange buffers (h [W][R * 8][I/W], y [W][R][H/W]),
/// or whole experts' spare output (y, the other ranks' partials).
const Virt = struct {
    world: u32,
    split: bool,
    layers: []const *const ic.weights.Layer,
    h: u64,
    y: u64,
    /// each virtual rank's route plan (cuda_moe.zig splitPlanBytes(N) apart, as each rank's exchange holds its own:
    /// the K1 forms keep their fold counters there)
    plan: u64,
    plan_bytes: usize = 0,
    /// the timing pass's columns (its result is not the one compared)
    y_rerun: u64 = 0,
    /// --only: one rank's slice as every rank; only virtual rank 0 runs, once, between the events (its weights cold,
    /// as in a decode step); the call's h ([W][R * 8][I/W]) and y ([W][R][H/W]) are zeroed first, so y (not the
    /// layer's) still has a row's bits fixed
    one: bool = false,
};

fn since(ev: [4]*cuda.Event) !i96 {
    try ev[1].synchronize();
    return @intFromFloat(@as(f64, try cuda.Event.elapsedMs(ev[0].*, ev[1].*)) * 1e6);
}

/// One experts call (the layer's mode; `force` the full mode's row tiles or the checkpoint mode's warp shape) between
/// two CUDA events: its device time in ns, rank 0's and the slowest rank's. Split experts: every rank's two halves and
/// the columns (the result), then each rank's own work again alone between the events (the same values rewritten).
/// Whole experts over virtual ranks: each rank's call alone (rank 0's into y).
fn timed(c: *const ic.ops.Ctx, ev: [4]*cuda.Event, l: *const ic.weights.Layer, v: ?Virt, m: Mat, ids: Mat, w: Mat, y: Mat, force: ?u32, form: moe.Form) ![2]i96 {
    const s = v orelse {
        try ev[0].record(c.stream);
        if (c.k.moe.mode == .checkpoint) {
            try moe.expertsCk(c, l, m, ids, w, y, force);
        } else {
            try moe.expertsTiled(c, l, m, ids, w, y, force);
        }
        try ev[1].record(c.stream);
        const t = try since(ev);
        return .{ t, t };
    };
    const R = m.rows;
    var out: [2]i96 = .{ 0, 0 };
    if (!s.split) {
        for (s.layers, 0..) |lr, r| {
            c.scratch.reset();
            const yr = if (r == 0) y else Mat.of(s.y, R, H, .bf16);
            try ev[0].record(c.stream);
            if (c.k.moe.mode == .checkpoint) try moe.expertsCk(c, lr, m, ids, w, yr, force) else try moe.expertsTiled(c, lr, m, ids, w, yr, force);
            try ev[1].record(c.stream);
            const t = try since(ev);
            if (r == 0) out[0] = t;
            out[1] = @max(out[1], t);
        }
        return out;
    }
    const seg = moe.intermediate / s.world;
    if (s.one) {
        try c.d.check(c.d.api.cuMemsetD8Async(s.h, 0, R * K8 * moe.intermediate * 2, c.stream.handle), "cuMemsetD8Async");
        try c.d.check(c.d.api.cuMemsetD8Async(s.y, 0, R * H * 2, c.stream.handle), "cuMemsetD8Async");
        c.scratch.reset();
        try ev[0].record(c.stream);
        try moe.splitUp(c, s.layers[0], m, ids, w, s.h, s.plan, force, form);
        c.scratch.reset();
        try moe.splitDown(c, s.layers[0], R, s.h, s.plan, s.y, force, form);
        try moe.splitColumns(c, s.y, s.world, y);
        try ev[1].record(c.stream);
        const t = try since(ev);
        return .{ t, t };
    }
    // every rank's slices poisoned first (0xFF bytes: NaN bf16), so a slice a rank never writes cannot pass
    try c.d.check(c.d.api.cuMemsetD8Async(s.h, 0xFF, R * K8 * moe.intermediate * 2, c.stream.handle), "cuMemsetD8Async");
    try c.d.check(c.d.api.cuMemsetD8Async(s.y, 0xFF, R * H * 2, c.stream.handle), "cuMemsetD8Async");
    for (s.layers, 0..) |lr, r| {
        c.scratch.reset();
        try moe.splitUp(c, lr, m, ids, w, s.h + r * R * K8 * seg * 2, s.plan + r * s.plan_bytes, force, form);
    }
    for (s.layers, 0..) |lr, r| {
        c.scratch.reset();
        try moe.splitDown(c, lr, R, s.h, s.plan + r * s.plan_bytes, s.y + r * R * (H / s.world) * 2, force, form);
    }
    // the result compared: this first all-up, all-down pass's columns (the timing pass below writes y_rerun)
    try moe.splitColumns(c, s.y, s.world, y);
    // each rank's own work alone: the same values again
    for (s.layers, 0..) |lr, r| {
        c.scratch.reset();
        try ev[0].record(c.stream);
        try moe.splitUp(c, lr, m, ids, w, s.h + r * R * K8 * seg * 2, s.plan + r * s.plan_bytes, force, form);
        c.scratch.reset();
        try moe.splitDown(c, lr, R, s.h, s.plan + r * s.plan_bytes, s.y + r * R * (H / s.world) * 2, force, form);
        try moe.splitColumns(c, s.y, s.world, Mat.of(s.y_rerun, R, H, .bf16));
        try ev[1].record(c.stream);
        const t = try since(ev);
        if (r == 0) out[0] = t;
        out[1] = @max(out[1], t);
    }
    return out;
}

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-9;
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}
