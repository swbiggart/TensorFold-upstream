//! tf-inkling2-attn --oracle: the fork's attention cores (v0.1 eager as `fused`, `rows` and `prev`; kx's forms as `kx`
//! (default dispatch), `kx-c1` (column split 1), `kx-split` (kx-split wherever it fits) and `kx-walk`) on synthetic
//! inputs at the real model's shapes against relbias_attention_ref.zig (oracle spec section 6). Every (row, query
//! head) gets the reference's float64 point and the range the fp32 form allows (replayForm), kx's own form's range
//! (kxForm sliding, kxSplitForm at --kx-s global, q'' with subnormal elements as +0 as kx loads them) and
//! kxFallbackForm's, and the fork's eager core replayed on the host in its own order
//! (k_exp 1, 1 +- 2^-22). Each forward runs at 48/8 (W = 1) and as the four 12/2 rank slices (W = 4), on the engine's
//! layouts: sliding layers a ring of 1023 + prompt_rows slots, global layers a slot a position.
//!
//! Per case (kind, profile, p0, R) a line per run (implementation x layout): outputs, inside replayForm, inside kx's
//! forms, the range's decided / ranged / nonfinite, the most bf16 steps and fp32 ulps outside and from the point, the
//! NaN / Inf class mismatches, outputs equal to the eager replay and the most steps from it (head-scale steps: bf16
//! steps at max over columns of sigmoid(gate) * sum p |v|, the head's output before cancellation); then the byte equality
//! of runs whose plans differ, slices against unsplit, and (kx-fallback) rows without the plant against kx-safe.
//! The masked-value panel puts a NaN value row at three places and reports which rows each form changes.
//! Exit 1 when eager runs with different plans differ, an implementation's rank slices differ from its unsplit run, kx's
//! forms differ from each other, kx is outside its own form, or the reference's escape fires or it refuses a head.
const std = @import("std");
const cuda = @import("cuda");
const ic = @import("inkling2_cuda");
const host = @import("cuda_attn_oracle_host.zig");
const rb = @import("core").relbias_attention_ref;

const attn = ic.attn;
const Mat = ic.ops.Mat;
const D = host.D;
const HQ = host.HQ;
const HK = host.HK;
const Profile = host.Profile;
const Kind = host.Kind;
const Acc = host.Acc;

pub const usage =
    \\usage: tf-inkling2-attn --oracle [options]   (synthetic inputs against relbias_attention_ref.zig, no model)
    \\  --profiles P,...   clip, diffuse, planted, kx-safe, kx-fallback (default all; kx-fallback needs kx-safe before it)
    \\  --kinds K,...      sliding, global (default both)
    \\  --positions P,...  forwards' first positions (default 0,1,31,255,256,1022,1023,1024,1025,2047,4095,16384,131072)
    \\  --rows R,...       forward rows, at most 64 (default 1,8,16,64)
    \\  --threads N        host threads for the ranges (default 32)
    \\  --panel 0|1        the masked-value panel (default 1)
    \\  --kx-s S           kx's key split (default cuda_attn.zig kx_split_keys)
    \\The GPU is the first visible device (CUDA_VISIBLE_DEVICES).
    \\
;

pub const Opts = struct {
    profiles: []const Profile = &.{ .clip, .diffuse, .planted, .@"kx-safe", .@"kx-fallback" },
    kinds: []const Kind = &.{ .sliding, .global },
    positions: []const usize = &.{ 0, 1, 31, 255, 256, 1022, 1023, 1024, 1025, 2047, 4095, 16384, 131072 },
    rows: []const usize = &.{ 1, 8, 16, 64 },
    threads: usize = 32,
    panel: bool = true,
    s: usize = attn.kx_split_keys,
};

fn enums(comptime T: type, a: std.mem.Allocator, v: []const u8) !?[]T {
    var out: std.ArrayList(T) = .empty;
    var it = std.mem.splitScalar(u8, v, ',');
    while (it.next()) |x| try out.append(a, std.meta.stringToEnum(T, x) orelse return null);
    return out.items;
}

pub fn ints(a: std.mem.Allocator, v: []const u8) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, v, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseInt(usize, x, 10));
    return out.items;
}

pub fn parse(a: std.mem.Allocator, args: []const [:0]const u8) !?Opts {
    var o: Opts = .{};
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        const eql = std.mem.eql;
        if (eql(u8, k, "--profiles")) o.profiles = (try enums(Profile, a, v)) orelse return null else if (eql(u8, k, "--kinds")) o.kinds = (try enums(Kind, a, v)) orelse return null else if (eql(u8, k, "--positions")) o.positions = try ints(a, v) else if (eql(u8, k, "--rows")) o.rows = try ints(a, v) else if (eql(u8, k, "--threads")) o.threads = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--panel")) o.panel = !eql(u8, v, "0") else if (eql(u8, k, "--kx-s")) o.s = try std.fmt.parseInt(usize, v, 10) else return null;
    }
    if (i != args.len or o.threads == 0 or std.mem.indexOfScalar(usize, &attn.kx_splits, o.s) == null) return null;
    for (o.rows) |R| if (R == 0 or R > max_rows) return null;
    return o;
}

const max_rows = 64;
pub const ring = attn.window - 1 + ic.engine.prompt_rows;
comptime {
    std.debug.assert(host.sliding_split == attn.kx_sliding_split);
}
/// Layouts: 0 the unsplit 48/8, 1 + r rank r's 12/2 (q heads 12r .. 12r + 11, kv heads 2r, 2r + 1).
pub const layouts = 5;
pub const Impl = enum { fused, rows, prev, kx, @"kx-c1", @"kx-split", @"kx-walk" };

/// kx's forms (each must give kx's bytes).
pub fn isKx(i: Impl) bool {
    return @backingInt(i) >= @backingInt(Impl.kx);
}

pub fn ix(i: Impl) usize {
    return @backingInt(i);
}
pub const impls = std.enums.values(Impl);
/// Runs: an implementation on the unsplit layout (2i) and on the four slices put together (2i + 1).
const runs = 2 * impls.len;

pub fn shape(li: usize) [3]usize {
    return if (li == 0) .{ HQ, HK, 0 } else .{ HQ / 4, HK / 4, li - 1 };
}

/// The device side: a layout's inputs, caches and an output per implementation.
const Dev = struct {
    q: [layouts]Mat,
    rel: [layouts]Mat,
    sink: [layouts]Mat,
    gate: [layouts]Mat,
    k: [layouts]u64,
    v: [layouts]u64,
    out: [layouts][impls.len]Mat,
    proj: u64,
    slots: usize,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn buf(x: *Dev, gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        try b.fill8(0, null);
        try x.own.append(gpa, b);
        return b.ptr;
    }

    fn mat(x: *Dev, gpa: std.mem.Allocator, d: *const cuda.Driver, cols: usize) !Mat {
        return Mat.of(try x.buf(gpa, d, max_rows * cols * 2), max_rows, cols, .bf16);
    }

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, slots: usize) !Dev {
        var x: Dev = undefined;
        x.own = .empty;
        errdefer x.deinit(gpa);
        x.slots = slots;
        x.proj = try x.buf(gpa, d, 16 * host.E * 2);
        for (0..layouts) |li| {
            const s = shape(li);
            x.q[li] = try x.mat(gpa, d, s[0] * D);
            x.rel[li] = try x.mat(gpa, d, s[0] * 16);
            x.sink[li] = try x.mat(gpa, d, s[0]);
            x.gate[li] = try x.mat(gpa, d, s[0] * D);
            x.k[li] = try x.buf(gpa, d, @max(slots, ring) * s[1] * D * 2);
            x.v[li] = try x.buf(gpa, d, @max(slots, ring) * s[1] * D * 2);
            for (&x.out[li]) |*m| m.* = try x.mat(gpa, d, s[0] * D);
        }
        return x;
    }

    fn deinit(x: *Dev, gpa: std.mem.Allocator) void {
        for (x.own.items) |*b| b.free();
        x.own.deinit(gpa);
    }

    fn kv(x: *const Dev, li: usize, kind: Kind) attn.Kv {
        return .{ .k = x.k[li], .v = x.v[li], .slots = if (kind == .global) x.slots else ring, .heads = shape(li)[1], .window = if (kind == .global) 0 else attn.window };
    }
};

pub fn put(d: *const cuda.Driver, dst: u64, bytes: []const u8) !void {
    const view: cuda.DeviceBuffer = .{ .d = d, .ptr = dst, .len = bytes.len, .borrowed = true };
    try view.upload(0, bytes);
}

/// The rows of `src` ([HK][cap][D], positions from ..) for layout li's kv heads into its cache's slots (ring or a
/// slot a position), through `stage`.
pub fn putCache(d: *const cuda.Driver, in: *const host.Inputs, src: []const u16, dst: u64, kv: attn.Kv, li: usize, stage: []u16) !void {
    const s = shape(li);
    const hk = s[1];
    const n = in.npos;
    for (0..n) |i| for (0..hk) |lg| {
        const g = s[2] * hk + lg;
        @memcpy(stage[(i * hk + lg) * D ..][0..D], src[(g * in.cap + i) * D ..][0..D]);
    };
    const row = hk * D * 2;
    var i: usize = 0;
    while (i < n) { // contiguous runs of slots
        const slot = (in.from + i) % kv.slots;
        const len = @min(n - i, kv.slots - slot);
        try put(d, dst + slot * row, std.mem.sliceAsBytes(stage[i * hk * D ..][0 .. len * hk * D]));
        i += len;
    }
}

/// Layout li's q'', r'', sink and gate (its query heads' columns) through `stage`.
fn putRows(d: *const cuda.Driver, in: *const host.Inputs, x: *const Dev, li: usize, stage: []u16) !void {
    const s = shape(li);
    const h0 = s[2] * s[0];
    for ([_]struct { []const u16, usize, Mat }{ .{ in.q, D, x.q[li] }, .{ in.rel, 16, x.rel[li] }, .{ in.sink, 1, x.sink[li] }, .{ in.gate, D, x.gate[li] } }) |t| {
        const w = s[0] * t[1];
        for (0..in.R) |r| @memcpy(stage[r * w ..][0..w], t[0][(r * HQ + h0) * t[1] ..][0..w]);
        try put(d, t[2].ptr, std.mem.sliceAsBytes(stage[0 .. in.R * w]));
    }
}

/// Implementation `impl` (its `attn.core` form, set as `Ctx.runOn` sets it; kx's at key split S) on layout li's shape
/// for the forward of q.rows rows at p0, with the bias projection at `proj`; its plan.
pub fn runCore(c: *const ic.ops.Ctx, k: *ic.kernels.Kernels, scratch: *ic.ops.Scratch, impl: Impl, S: usize, li: usize, kind: Kind, proj: u64, kv: attn.Kv, p0: usize, q: Mat, rel: Mat, sink: Mat, gate: Mat, out: Mat) !?attn.Plan {
    const ka = &k.attn;
    ka.rows = if (impl == .prev) 0 else attn.max_rows;
    ka.form = switch (impl) {
        .fused => .fused,
        .rows, .prev => .v01,
        else => .kx,
    };
    ka.kxo = switch (impl) {
        .@"kx-c1" => .{ .s = S, .c = 1 },
        .@"kx-split" => .{ .s = S, .global = .split },
        .@"kx-walk" => .{ .s = S, .global = .walk },
        else => .{ .s = S },
    };
    ka.grid = attn.grid_splits;
    ka.fold_min = attn.fold_splits;
    const s = shape(li);
    var l: ic.weights.Layer = .{ .index = 0, .global = kind == .global, .boundary = false, .hq = @intCast(s[0]), .hk = @intCast(s[1]), .nq = 1, .experts = .{ .format = .nvfp4, .lo = 0, .hi = 0 } };
    l.rel_proj = proj;
    scratch.reset();
    try attn.core(c, &l, p0, null, q, rel, sink, gate, kv, out);
    return attn.plan(ka, q.rows, p0, null, kv, s[0], s[1]);
}

const Gpu = struct {
    c: ic.ops.Ctx,
    k: *ic.kernels.Kernels,
    scratch: *ic.ops.Scratch,
    x: *Dev,
    kx: bool,
    s: usize,

    /// Implementation `impl` on layout li for the forward of R rows at p0; its plan.
    fn run(g: *const Gpu, impl: Impl, li: usize, kind: Kind, p0: usize, R: usize) !?attn.Plan {
        const x = g.x;
        return runCore(&g.c, g.k, g.scratch, impl, g.s, li, kind, x.proj, x.kv(li, kind), p0, x.q[li].slice(0, R), x.rel[li].slice(0, R), x.sink[li].slice(0, R), x.gate[li].slice(0, R), x.out[li][ix(impl)].slice(0, R));
    }

    /// Every implementation on every layout; outputs into outs[run] as [R][HQ][D] (slices put together).
    fn all(g: *const Gpu, in: *const host.Inputs, plans: *[runs]?attn.Plan, outs: *[runs][]u16, tmp: []u16) !void {
        for (0..layouts) |li| for (impls) |impl| {
            if (isKx(impl) and !g.kx) continue;
            const p = try g.run(impl, li, in.kind, in.p0, in.R);
            if (li <= 1) plans[2 * ix(impl) + @min(li, 1)] = p;
        };
        try g.c.stream.synchronize();
        for (0..layouts) |li| for (impls) |impl| {
            if (isKx(impl) and !g.kx) continue;
            const s = shape(li);
            const w = s[0] * D;
            try cuda.memory.download(g.c.d, g.x.out[li][ix(impl)].ptr, std.mem.sliceAsBytes(tmp[0 .. in.R * w]));
            const o = outs[2 * ix(impl) + @min(li, 1)];
            for (0..in.R) |r| @memcpy(o[(r * HQ + s[2] * s[0]) * D ..][0..w], tmp[r * w ..][0..w]);
        };
    }
};

/// A pool of host threads over `n` items: f(ctx, item, worker).
pub fn parallel(threads: usize, n: usize, ctx: anytype, comptime f: anytype) !void {
    const W = struct {
        fn go(next: *std.atomic.Value(usize), total: usize, c: @TypeOf(ctx), w: usize) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= total) return;
                f(c, i, w);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    var ts: [128]std.Thread = undefined;
    const m = @min(threads, ts.len, @max(n, 1));
    for (ts[0..m], 0..) |*t, w| t.* = try std.Thread.spawn(.{}, W.go, .{ &next, n, ctx, w });
    for (ts[0..m]) |t| t.join();
}

/// The host check of one forward: each (row, head) item's ranges against every run's outputs, per worker.
const Check = struct {
    in: *const host.Inputs,
    outs: *const [runs][]u16,
    have: [runs]bool,
    kx: bool,
    s: usize,
    work: []rb.Work,
    accs: [][runs]Acc,

    fn item(c: *const Check, i: usize, w: usize) void {
        const t = i / HQ;
        const h = i % HQ;
        var r: host.Ref = .{};
        r.make(c.in.head(t, h), c.in.geo(), c.kx, c.s, &c.work[w]);
        for (0..runs) |ri| if (c.have[ri]) {
            r.check(c.outs[ri][(t * HQ + h) * D ..][0..D], isKx(impls[ri / 2]), &c.accs[w][ri]);
        };
    }
};

fn fillKv(in: *const host.Inputs, i: usize, _: usize) void {
    in.fillKv(i);
}

fn fillRow(in: *const host.Inputs, i: usize, _: usize) void {
    in.fillRow(i);
}

fn label(ri: usize) [2][]const u8 {
    return .{ @tagName(impls[ri / 2]), if (ri % 2 == 0) "48/8" else "12/2x4" };
}

pub fn planText(buf: []u8, p: ?attn.Plan) []const u8 {
    const x = p orelse return "kx";
    if (x.form == .kx) return switch (x.kx) {
        .tile => std.fmt.bufPrint(buf, "kx-{s}{d}", .{ if (x.c == 1) "tile/c" else "dec/c", x.c }) catch "kx-tile",
        .split => std.fmt.bufPrint(buf, "kx-split/S{d}/c{d}/{d}", .{ x.s, x.c, x.span }) catch "kx-split",
        .walk => std.fmt.bufPrint(buf, "kx-walk/S{d}/{d}", .{ x.s, x.span }) catch "kx-walk",
        .none => "kx",
    };
    if (x.form != .split) return @tagName(x.form);
    return std.fmt.bufPrint(buf, "split{s}{s}/{d}", .{ if (x.rows) "+rows" else "", if (x.fold) "+fold" else "", x.span }) catch "split";
}

fn line(in: *const host.Inputs, ri: usize, plan: ?attn.Plan, a: Acc, eq: []const u8) void {
    const lb = label(ri);
    var pb: [48]u8 = undefined;
    const pt = planText(&pb, plan);
    std.debug.print("CASE\t{s}\t{s}\t{t}\t{s}\t{t}\t{d}\t{d}\t{d}\t{d}\t{d}/{d}/{d}\t{d}/{d}/{d}\t{d}\t{d:.2}\t{d:.2}\t{d}\t{d:.1}\t{d:.2}\t{d}\t{d}\t{d}\t{d:.2}\t{d}\t{s}\n", .{ lb[0], pt, in.kind, lb[1], in.profile, in.p0, in.R, a.outputs, a.inside, a.kx_in[0], a.kx_in[1], a.kx_in[2], a.stats.decided, a.stats.ranged, a.stats.nonfinite, a.out_steps, a.out_ulps, a.hs_out, a.pt_steps, a.pt_ulps, a.hs_pt, a.class_bad, a.eager_eq, a.eager_steps, a.eager_hs, a.escape, eq });
    std.debug.print("JSON {{\"impl\":\"{s}\",\"form\":\"{s}\",\"kind\":\"{t}\",\"heads\":\"{s}\",\"profile\":\"{t}\",\"p0\":{d},\"R\":{d},\"outputs\":{d},\"inside_replay\":{d},\"inside_kxform\":{d},\"inside_kxfallback\":{d},\"inside_kx_either\":{d},\"decided\":{d},\"ranged\":{d},\"nonfinite\":{d},\"max_steps_outside\":{d},\"max_ulps32_outside\":{d:.3},\"max_head_steps_outside\":{d:.3},\"max_steps_from_point\":{d},\"max_ulps32_from_point\":{d:.3},\"max_head_steps_from_point\":{d:.3},\"class_mismatch\":{d},\"got_nonfinite\":{d},\"eager_equal\":{d},\"eager_max_steps\":{d},\"eager_max_head_steps\":{d:.3},\"escape\":{d},\"refused\":{d},", .{ lb[0], pt, in.kind, lb[1], in.profile, in.p0, in.R, a.outputs, a.inside, a.kx_in[0], a.kx_in[1], a.kx_in[2], a.stats.decided, a.stats.ranged, a.stats.nonfinite, a.out_steps, a.out_ulps, a.hs_out, a.pt_steps, a.pt_ulps, a.hs_pt, a.class_bad, a.got_nonfinite, a.eager_eq, a.eager_steps, a.eager_hs, a.escape, a.refused });
    std.debug.print("\"hist_out_head_steps\":[{d},{d},{d},{d},{d},{d}],\"bytes_equal\":\"{s}\"}}\n", .{ a.hist[0], a.hist[1], a.hist[2], a.hist[3], a.hist[4], a.hist[5], eq });
}

/// Host and device state of a run of the tool.
const State = struct {
    gpa: std.mem.Allocator,
    o: Opts,
    gpu: Gpu,
    in: host.Inputs,
    stage: []u16,
    outs: [runs][]u16,
    keep: [runs][]u16,
    work: []rb.Work,
    accs: [][runs]Acc,
    sum: [runs][2][5]Acc = @splat(@splat(@splat(.{}))),
    failed: usize = 0,
    /// DIGEST oracle: sha256 a run (implementation x layout) over every case's and panel step's outputs, in run order
    digest: [runs]std.crypto.hash.sha2.Sha256 = @splat(std.crypto.hash.sha2.Sha256.init(.{})),
    /// the forward whose kx-safe runs `keep` holds (null: none, or the panel's)
    kept: ?host.Forward = null,

    /// Inputs for (profile, kind, p0, R) on the host, then on every layout.
    fn prepare(s: *State, profile: Profile, kind: Kind, p0: usize, R: usize) !void {
        const in = &s.in;
        in.profile = profile;
        in.kind = kind;
        in.p0 = p0;
        in.R = R;
        const lo = rb.keyLo(in.geo(), p0);
        in.from = if (kind == .global) 0 else lo / 64 * 64; // kx loads 64-key tiles from its first row's window
        in.npos = p0 + R - in.from;
        try parallel(s.o.threads, 2 * HK, in, fillKv);
        in.plant();
        try parallel(s.o.threads, R * HQ, in, fillRow);
        const x = s.gpu.x;
        try put(s.gpu.c.d, x.proj, std.mem.sliceAsBytes(in.proj));
        for (0..layouts) |li| {
            const kv = x.kv(li, kind);
            try putCache(s.gpu.c.d, in, in.k, kv.k, kv, li, s.stage);
            try putCache(s.gpu.c.d, in, in.v, kv.v, kv, li, s.stage);
            try putRows(s.gpu.c.d, in, x, li, s.stage);
        }
    }

    /// The runs' outputs of this forward (R rows) into their digests.
    fn hash(s: *State, R: usize) void {
        for (&s.digest, 0..) |*h, ri| if (s.gpu.kx or !isKx(impls[ri / 2])) {
            h.update(std.mem.sliceAsBytes(s.outs[ri][0 .. R * HQ * D]));
        };
    }

    /// The checks of rows `rows` of the current outputs; per-run totals.
    fn check(s: *State, rows: usize, only_unsplit: bool) ![runs]Acc {
        var have: [runs]bool = undefined;
        for (&have, 0..) |*h, ri| h.* = (s.gpu.kx or !isKx(impls[ri / 2])) and (!only_unsplit or ri % 2 == 0);
        for (s.accs) |*a| a.* = @splat(.{});
        const c: Check = .{ .in = &s.in, .outs = &s.outs, .have = have, .kx = s.gpu.kx, .s = s.gpu.s, .work = s.work, .accs = s.accs };
        try parallel(s.o.threads, rows * HQ, &c, Check.item);
        var tot: [runs]Acc = @splat(.{});
        for (s.accs) |a| for (&tot, a) |*t, x| t.add(x);
        return tot;
    }

    fn case(s: *State, profile: Profile, kind: Kind, p0: usize, R: usize) !void {
        try s.prepare(profile, kind, p0, R);
        var plans: [runs]?attn.Plan = @splat(null);
        try s.gpu.all(&s.in, &plans, &s.outs, s.stage);
        s.hash(R);
        const tot = try s.check(R, false);
        // byte equality: eager runs pairwise (trivial where layout and plan agree), each slice run against unsplit
        var eb: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&eb);
        const n = R * HQ * D;
        var differ: usize = 0;
        var equal: usize = 0;
        var trivial: usize = 0;
        for (0..6) |a| for (a + 1..6) |b| {
            if (a % 2 == b % 2 and std.meta.eql(plans[a], plans[b])) {
                trivial += 1;
            } else if (std.mem.eql(u16, s.outs[a][0..n], s.outs[b][0..n])) equal += 1 else differ += 1;
        };
        w.print("eager pairs: {d} equal, {d} differ, {d} same plan; slices=unsplit:", .{ equal, differ, trivial }) catch {};
        var slices: [impls.len]usize = @splat(0);
        for (impls, 0..) |impl, i| if (!isKx(impl) or s.gpu.kx) {
            slices[i] = host.vectorsDiffer(s.outs[2 * i], s.outs[2 * i + 1], R * HQ);
            w.print(" {t} {d}", .{ impl, slices[i] }) catch {};
        };
        // kx's forms against kx's unsplit run: the same bytes
        var kx_differ: usize = 0;
        if (s.gpu.kx) {
            for (impls, 0..) |impl, i| if (isKx(impl)) {
                kx_differ += host.vectorsDiffer(s.outs[2 * ix(.kx)], s.outs[2 * i], R * HQ);
            };
            w.print("; kx forms=kx: {d} differ", .{kx_differ}) catch {};
            if (kx_differ > 0) s.failed += 1;
        }
        if (profile == .@"kx-fallback" and R > 1) { // rows without the plant against kx-safe's run of them
            try host.checkBaseline(s.kept, .{ .kind = kind, .p0 = p0, .R = R });
            w.print("; rows coupled to the plant:", .{}) catch {};
            for (impls, 0..) |impl, i| if (!isKx(impl) or s.gpu.kx) {
                var moved: usize = 0;
                for (0..R) |t| if (t % 16 != 0) {
                    moved += @intFromBool(!std.mem.eql(u16, s.outs[2 * i][t * HQ * D ..][0 .. HQ * D], s.keep[2 * i][t * HQ * D ..][0 .. HQ * D]));
                };
                w.print(" {t} {d}/{d}", .{ impl, moved, R - (R + 15) / 16 }) catch {};
            };
        }
        s.failed += host.byteFailures(differ, &slices);
        for (tot, 0..) |a, ri| if (a.outputs > 0) {
            line(&s.in, ri, plans[ri], a, w.buffered());
            s.sum[ri][@backingInt(kind)][@backingInt(profile)].add(a);
            if (a.escape > 0 or a.refused > 0) s.failed += 1;
            // kx's own form for every profile (a subnormal q'' loads as +0, so kx-fallback's rows stay on the tiles)
            if (isKx(impls[ri / 2]) and a.kx_in[0] != a.outputs) s.failed += 1;
        };
        if (profile == .@"kx-safe") {
            for (s.keep, s.outs) |k, o| @memcpy(k[0..n], o[0..n]);
            s.kept = .{ .kind = kind, .p0 = p0, .R = R };
        }
    }

    /// The masked-value panel: sliding, p0 1024, 8 rows, kx-safe, 48/8: a NaN value row (every kv head) at a stale
    /// ring slot, at position 0 (before every window, inside kx's first tile), at position 1028 (rows 0-3 cannot see it).
    fn panel(s: *State) !void {
        try s.prepare(.@"kx-safe", .sliding, 1024, 8);
        var plans: [runs]?attn.Plan = @splat(null);
        try s.gpu.all(&s.in, &plans, &s.outs, s.stage);
        const n = 8 * HQ * D;
        for (s.keep, s.outs) |k, o| @memcpy(k[0..n], o[0..n]);
        s.kept = null;
        const row = HK * D;
        const nan = try s.gpa.alloc(u16, row);
        defer s.gpa.free(nan);
        @memset(nan, 0x7fc0);
        const old = try s.gpa.alloc(u16, row);
        defer s.gpa.free(old);
        const places = [_]struct { []const u8, usize, ?u64 }{ .{ "stale slot 2000", 2000, null }, .{ "position 0", 0, 0 }, .{ "position 1028", 1028, 1028 } };
        for (places) |pl| {
            const at = s.gpu.x.v[0] + pl[1] * row * 2;
            try cuda.memory.download(s.gpu.c.d, at, std.mem.sliceAsBytes(old));
            try put(s.gpu.c.d, at, std.mem.sliceAsBytes(nan));
            if (pl[2]) |p| for (0..HK) |g| @memset(s.in.vRow(g, p), 0x7fc0);
            try s.gpu.all(&s.in, &plans, &s.outs, s.stage);
            s.hash(8);
            const tot = try s.check(8, true);
            for (impls, 0..) |impl, i| if (!isKx(impl) or s.gpu.kx) {
                var changed: [8]u8 = @splat('.');
                for (0..8) |t| {
                    const got = s.outs[2 * i][t * HQ * D ..][0 .. HQ * D];
                    if (!std.mem.eql(u16, got, s.keep[2 * i][t * HQ * D ..][0 .. HQ * D])) changed[t] = 'c';
                    for (got) |x| if (x & 0x7f80 == 0x7f80 and x & 0x7f != 0) {
                        changed[t] = 'N';
                        break;
                    };
                }
                var pb: [48]u8 = undefined;
                const a = tot[2 * i];
                std.debug.print("PANEL\t{s}\t{t}\t{s}\trows {s}\tinside {d}/{d}\tclass mismatch {d}\tnonfinite points {d}\n", .{ pl[0], impl, planText(&pb, plans[2 * i]), &changed, a.inside, a.outputs, a.class_bad, a.stats.nonfinite });
            };
            try put(s.gpu.c.d, at, std.mem.sliceAsBytes(old));
            if (pl[2] != null) for (0..HK) |g| s.in.fillKv(HK + g); // the host's values again
        }
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    try host.checkProfiles(o.profiles); // before any GPU work: kx-fallback compares with kx-safe's run of its forward
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
    var top: usize = 1024 + 8;
    var most: usize = 1;
    for (o.positions) |p| for (o.rows) |R| {
        top = @max(top, p + R);
        most = @max(most, R);
    };
    const kx = k.attn.kx != null;
    const form = k.attn.form;
    k.attn.form = .v01;
    var need = @max(attn.kx_split_cap, attn.kxNeed(.{ .s = o.s }, HQ, top, max_rows, 0), attn.kxNeed(.{ .s = o.s }, HQ / 4, top, max_rows, 0));
    for ([_][2]usize{ .{ HQ, HK }, .{ HQ / 4, HK / 4 } }) |s| need = @max(need, attn.scratchNeed(&k.attn, s[0], s[1], top, max_rows, 0));
    k.attn.form = form;
    var scratch = try ic.ops.Scratch.init(&driver, need + (16 << 20));
    defer scratch.deinit();
    var x = try Dev.init(gpa, &driver, top);
    defer x.deinit(gpa);
    var st: State = .{ .gpa = gpa, .o = o, .gpu = .{ .c = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch }, .k = &k, .scratch = &scratch, .x = &x, .kx = kx, .s = o.s }, .in = try host.Inputs.alloc(gpa, top, max_rows), .stage = undefined, .outs = undefined, .keep = undefined, .work = undefined, .accs = undefined };
    defer st.in.free(gpa);
    st.stage = try gpa.alloc(u16, @max(top * HK * D, max_rows * HQ * D));
    defer gpa.free(st.stage);
    for (&st.outs, &st.keep) |*a, *b| {
        a.* = try gpa.alloc(u16, max_rows * HQ * D);
        b.* = try gpa.alloc(u16, max_rows * HQ * D);
    }
    defer for (st.outs, st.keep) |a, b| {
        gpa.free(a);
        gpa.free(b);
    };
    const nt = @min(o.threads, 128);
    st.work = try gpa.alloc(rb.Work, nt);
    defer gpa.free(st.work);
    for (st.work) |*wk| wk.* = try rb.Work.init(gpa, top + 1);
    defer for (st.work) |*wk| wk.deinit(gpa);
    st.accs = try gpa.alloc([runs]Acc, nt);
    defer gpa.free(st.accs);
    st.o.threads = nt;
    k.attn.kxo = .{ .s = o.s };
    var cb: [160]u8 = undefined;
    std.debug.print("{s}\n", .{k.attn.census(&cb)});
    std.debug.print("oracle: kx {s} (S {d}); slots {d} (global), {d} (sliding ring); scratch {d:.1} MiB; {d} host threads\n", .{ if (kx) "loaded" else "not built", o.s, top, ring, @as(f64, @floatFromInt(need)) / (1 << 20), nt });
    std.debug.print("CASE\timpl\tform\tkind\theads\tprofile\tp0\tR\toutputs\tinside_replay\tinside_kxform/kxfallback/either\tdecided/ranged/nonfinite\tmax_steps_out\tmax_ulps32_out\tmax_head_steps_out\tmax_steps_pt\tmax_ulps32_pt\tmax_head_steps_pt\tclass_bad\teager_equal\teager_max_steps\teager_max_head_steps\tescape\tbytes_equal\n", .{});
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(io).toNanoseconds();
    if (o.panel) try st.panel();
    for (o.kinds) |kind| for (o.positions) |p0| for (o.rows) |R| for (o.profiles) |pf| try st.case(pf, kind, p0, R);
    std.debug.print("SUM\timpl\theads\tkind\tprofile\toutputs\tinside_replay\tpct\tinside_kxform\tinside_kxfallback\tinside_kx_either\tmax_steps_out\tmax_ulps32_out\tmax_head_steps_out\tmax_steps_pt\tmax_head_steps_pt\teager_equal_pct\teager_max_steps\teager_max_head_steps\tclass_bad\tescape\toutside_by_head_steps(<=.5,1,2,4,16,more)\n", .{});
    for (st.sum, 0..) |per, ri| for (per, 0..) |pk, ki| for (pk, 0..) |a, pi| if (a.outputs > 0) {
        const lb = label(ri);
        const f: f64 = @floatFromInt(a.outputs);
        std.debug.print("SUM\t{s}\t{s}\t{t}\t{t}\t{d}\t{d}\t{d:.3}\t{d}\t{d}\t{d}\t{d}\t{d:.2}\t{d:.2}\t{d}\t{d:.2}\t{d:.3}\t{d}\t{d:.2}\t{d}\t{d}\t{d},{d},{d},{d},{d},{d}\n", .{ lb[0], lb[1], @as(Kind, @fromBackingInt(@intCast(ki))), @as(Profile, @fromBackingInt(@intCast(pi))), a.outputs, a.inside, 100 * @as(f64, @floatFromInt(a.inside)) / f, a.kx_in[0], a.kx_in[1], a.kx_in[2], a.out_steps, a.out_ulps, a.hs_out, a.pt_steps, a.hs_pt, 100 * @as(f64, @floatFromInt(a.eager_eq)) / f, a.eager_steps, a.eager_hs, a.class_bad, a.escape, a.hist[0], a.hist[1], a.hist[2], a.hist[3], a.hist[4], a.hist[5] });
    };
    for (&st.digest, 0..) |*h, ri| if (kx or !isKx(impls[ri / 2])) {
        var dg: [32]u8 = undefined;
        h.final(&dg);
        const lb = label(ri);
        std.debug.print("DIGEST oracle\t{s}\t{s}\tsha256 {s}\n", .{ lb[0], lb[1], std.fmt.bytesToHex(dg, .lower) });
    };
    k.attn.printExceptions(&driver);
    std.debug.print("{s} oracle: {d} failed cases; {d:.0} s\n", .{ if (st.failed == 0) "PASS" else "FAIL", st.failed, @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-9 });
    return if (st.failed == 0) 0 else 1;
}
