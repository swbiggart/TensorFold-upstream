//! tf-inkling2-attn --oracle, the host half: the synthetic inputs (contract_relbias_attention_host.zig's generator at
//! the real model, a forward's rows sharing one cache), the fork's eager core replayed in its own order, and each
//! output's check against relbias_attention_ref.zig's ranges. No GPU.
const std = @import("std");
const core = @import("core");
const rb = core.relbias_attention_ref;
const ar = core.attention_range;
const bfr = core.bf16_round;

pub const D = 128;
pub const E = 1024;
pub const HQ = 48;
pub const HK = 8;
pub const NREP = HQ / HK;

pub const Profile = enum { clip, diffuse, planted, @"kx-safe", @"kx-fallback" };
pub const Kind = enum { sliding, global };
const Tag = enum(u64) { q = 1, k, v, rel, proj, gate, sink, dir, sign };

fn mix(x0: u64) u64 {
    var x = x0 +% 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) *% 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) *% 0x94d049bb133111eb;
    return x ^ (x >> 31);
}

/// A value stream per (seed, tensor, position, global head): a rank slice draws exactly the unsplit layout's values.
fn stream(seed: u64, tag: Tag, pos: u64, head: u64) std.Random.DefaultPrng {
    return std.Random.DefaultPrng.init(mix(mix(mix(seed ^ @backingInt(tag)) ^ pos) ^ head));
}

fn bits(x: f64) u16 {
    return bfr.bits(@floatCast(x));
}

/// Gaussians of deviation sd, scaled to `norm` when given, rounded to bf16.
fn normal(seed: u64, tag: Tag, pos: u64, head: u64, norm: ?f64, sd: f64, out: []u16) void {
    var prng = stream(seed, tag, pos, head);
    var buf: [D]f64 = undefined;
    var ss: f64 = 0;
    for (buf[0..out.len]) |*x| {
        x.* = prng.random().floatNorm(f64) * sd;
        ss += x.* * x.*;
    }
    const k = if (norm) |nm| nm / @sqrt(ss) else 1;
    for (out, buf[0..out.len]) |*o, x| o.* = bits(x * k);
}

/// One forward's inputs: keys and values of positions from .. from + npos - 1 for every kv head ([HK][cap][D]), and
/// per row t (position p0 + t) and query head: q'', r'', gate [R][HQ][..], sink [R][HQ].
pub const Inputs = struct {
    profile: Profile,
    kind: Kind,
    p0: u64,
    R: usize,
    from: u64,
    npos: usize,
    cap: usize,
    k: []u16,
    v: []u16,
    q: []u16,
    rel: []u16,
    gate: []u16,
    sink: []u16,
    proj: []u16,

    pub fn alloc(gpa: std.mem.Allocator, cap: usize, rows: usize) !Inputs {
        var in: Inputs = undefined;
        in.cap = cap;
        in.k = try gpa.alloc(u16, HK * cap * D);
        errdefer gpa.free(in.k);
        in.v = try gpa.alloc(u16, HK * cap * D);
        errdefer gpa.free(in.v);
        in.q = try gpa.alloc(u16, rows * HQ * D);
        errdefer gpa.free(in.q);
        in.rel = try gpa.alloc(u16, rows * HQ * rb.d_rel);
        errdefer gpa.free(in.rel);
        in.gate = try gpa.alloc(u16, rows * HQ * D);
        errdefer gpa.free(in.gate);
        in.sink = try gpa.alloc(u16, rows * HQ);
        errdefer gpa.free(in.sink);
        in.proj = try gpa.alloc(u16, rb.d_rel * E);
        return in;
    }

    pub fn free(in: *Inputs, gpa: std.mem.Allocator) void {
        for ([_][]u16{ in.k, in.v, in.q, in.rel, in.gate, in.sink, in.proj }) |x| gpa.free(x);
    }

    pub fn geo(in: *const Inputs) rb.Geo {
        return .{ .window = if (in.kind == .sliding) 1024 else 0, .extent = E };
    }

    /// The shape's seed: contract_relbias_attention_host.zig's at the real model; kx-fallback draws kx-safe's values.
    fn seed(in: *const Inputs) u64 {
        const p: u64 = if (in.profile == .@"kx-fallback") 3 else @backingInt(in.profile);
        return mix(p << 16 | @as(u64, @backingInt(in.kind)) << 8 | HQ);
    }

    fn norm(in: *const Inputs) f64 {
        return switch (in.profile) {
            .clip, .planted => 22.6,
            .diffuse => 3,
            .@"kx-safe", .@"kx-fallback" => 12,
        };
    }

    pub fn kRow(in: *const Inputs, g: usize, pos: u64) []u16 {
        return in.k[(g * in.cap + @as(usize, @intCast(pos - in.from))) * D ..][0..D];
    }

    pub fn vRow(in: *const Inputs, g: usize, pos: u64) []u16 {
        return in.v[(g * in.cap + @as(usize, @intCast(pos - in.from))) * D ..][0..D];
    }

    /// Row t, query head h as the reference takes it (keys from the row's window start).
    pub fn head(in: *const Inputs, t: usize, h: usize) rb.Head {
        const pos = in.p0 + t;
        const lo = rb.keyLo(in.geo(), pos);
        const g = h / NREP;
        const a = (g * in.cap + @as(usize, @intCast(lo - in.from))) * D;
        const n: usize = @intCast(pos - lo + 1);
        const i = t * HQ + h;
        return .{ .pos = pos, .base = lo, .q = in.q[i * D ..][0..D], .k = in.k[a..][0 .. n * D], .v = in.v[a..][0 .. n * D], .rel = in.rel[i * rb.d_rel ..][0..rb.d_rel], .proj = in.proj, .sink = in.sink[i], .gate = in.gate[i * D ..][0..D] };
    }

    /// Keys and values of kv head g (item g) or the values (item HK + g), positions from .. from + npos - 1.
    pub fn fillKv(in: *const Inputs, item: usize) void {
        const g = item % HK;
        const s = in.seed();
        for (0..in.npos) |i| {
            const j = in.from + i;
            if (item < HK) normal(s, .k, j, g, in.norm(), 1, in.kRow(g, j)) else normal(s, .v, j, g, null, 1, in.vRow(g, j));
        }
    }

    /// planted: keys at distances 1022 .. 1025 from the first row share the group's direction with its q heads
    /// (values s_g twice, then -s_g); the bias columns, as the host test sets them.
    pub fn plant(in: *const Inputs) void {
        const s = in.seed();
        var col: [rb.d_rel]u16 = undefined;
        for (0..E) |c| {
            normal(s, .proj, c, 0, if (in.norm() == 12) 0.5 else null, 0.25, &col); // kx-safe: |bias| <= 4
            for (col, 0..) |x, i| in.proj[i * E + c] = x;
        }
        if (in.profile != .planted) return;
        const cols: []const usize = if (in.kind == .sliding) &.{ 1022, 1023 } else &.{E - 1};
        for (cols) |c| for (0..rb.d_rel) |i| {
            in.proj[i * E + c] = if (in.kind == .sliding) 0 else bits(6);
        };
        for (0..HK) |g| {
            var sign = stream(s, .sign, in.p0, g);
            var sg: [D]f64 = undefined;
            for (&sg) |*x| x.* = if (sign.random().boolean()) 4 else -4;
            for ([_]u64{ 1022, 1023, 1024, 1025 }) |dist| if (in.p0 >= dist and in.p0 - dist >= in.from) {
                const j = in.p0 - dist;
                normal(s, .dir, in.p0, g, 22.6, 1, in.kRow(g, j));
                for (in.vRow(g, j), sg) |*o, x| o.* = bits(if (dist < 1024) x else -x);
            };
        }
    }

    /// Row t, head h: q'', r'', gate, then the sink from the row's float64 max (keys, values and proj filled).
    pub fn fillRow(in: *const Inputs, item: usize) void {
        const t = item / HQ;
        const h = item % HQ;
        const i = t * HQ + h;
        const pos = in.p0 + t;
        const s = in.seed();
        const q = in.q[i * D ..][0..D];
        if (in.profile == .planted) normal(s, .dir, in.p0, h / NREP, 22.6, 1, q) else normal(s, .q, pos, h, in.norm(), 1, q);
        if (in.profile == .@"kx-fallback" and t % 16 == 0) q[0] = 0x0001; // a subnormal: kx's 16-row block falls back
        normal(s, .rel, pos, h, 8, 1, in.rel[i * rb.d_rel ..][0..rb.d_rel]);
        var gs = stream(s, .gate, pos, h);
        for (in.gate[i * D ..][0..D]) |*x| x.* = bits(gs.random().float(f64) * 8 - 4);
        const hd = in.head(t, h);
        var m: f64 = -std.math.inf(f64);
        for (0..@as(usize, @intCast(pos - hd.base + 1))) |j| {
            var sc: f64 = 0;
            for (hd.q, hd.k[j * D ..][0..D]) |x, y| sc += @as(f64, bfr.value(x)) * bfr.value(y);
            m = @max(m, sc * rb.scale32 + rb.bias(hd.rel, hd.proj, E, pos - hd.base - j));
        }
        var ss = stream(s, .sink, pos, h);
        in.sink[i] = bits(if (in.profile == .planted and h % 2 == 0) m + 1 else m + ss.random().float(f64) * 6 - 4);
    }
};

fn expk(x: f32, k_exp: f64) f32 {
    return @floatCast(@exp(@as(f64, x)) * k_exp);
}

/// The fork's eager core (inkling2_attn.cu) for one (row, head) in its own order, expf as fp32(exp64(x) * k_exp):
/// six bf16 points; 32-key absolute tiles with a rescale at each tile's max, 256-key splits folded from (sink, 1)
/// in order; P = bf16(expf(s - m) / z); o a fmaf chain a split, the splits added in order; out = bf16(bf16(o) * G).
pub fn eagerFork(h: rb.Head, g: rb.Geo, k_exp: f64, s: []f32, out: []u16) void {
    const lo = rb.keyLo(g, h.pos);
    const n: usize = @intCast(h.pos - lo + 1);
    const at: usize = @intCast(lo - h.base);
    for (s[0..n], 0..) |*sj, j| {
        var acc: f32 = 0;
        for (h.q, h.k[(at + j) * D ..][0..D]) |x, y| acc = @mulAdd(f32, bfr.value(x), bfr.value(y), acc);
        var sc = bfr.round32(bfr.round32(acc) * rb.scale32);
        const dist = n - 1 - j;
        if (dist < g.extent) sc = bfr.round32(sc + rb.bias(h.rel, h.proj, g.extent, dist));
        sj.* = sc;
    }
    var m: f32 = bfr.value(h.sink);
    var z: f32 = 1;
    var u = lo / 256;
    while (u <= h.pos / 256) : (u += 1) {
        var ms = -std.math.inf(f32);
        var zs: f32 = 0;
        var kt = @max(u * 256, lo / 32 * 32);
        while (kt <= @min(u * 256 + 255, h.pos)) : (kt += 32) {
            const a: usize = @intCast(@max(kt, lo) - lo);
            const b: usize = @intCast(@min(kt + 31, h.pos) - lo + 1);
            var tm = -std.math.inf(f32);
            for (s[a..b]) |x| tm = fmax(tm, x);
            if (tm == -std.math.inf(f32)) continue;
            const mn = fmax(ms, tm);
            zs = zs * expk(ms - mn, k_exp);
            for (s[a..b]) |x| zs = zs + expk(x - mn, k_exp);
            ms = mn;
        }
        if (zs > 0) {
            const mn = fmax(m, ms);
            z = z * expk(m - mn, k_exp) + zs * expk(ms - mn, k_exp);
            m = mn;
        }
    }
    for (s[0..n]) |*x| x.* = bfr.round32(expk(x.* - m, k_exp) / z);
    for (out[0..D], h.gate, 0..) |*o, gb, e| {
        var acc: f32 = 0;
        u = lo / 256;
        while (u <= h.pos / 256) : (u += 1) {
            var os: f32 = 0;
            for (@max(u * 256, lo)..@min(u * 256 + 255, h.pos) + 1) |p| {
                const j: usize = @intCast(p - lo);
                os = @mulAdd(f32, s[j], bfr.value(h.v[(at + j) * D + e]), os);
            }
            acc = acc + os;
        }
        o.* = bfr.bits(bfr.round32(acc) * (1.0 / (1.0 + expk(-bfr.value(gb), k_exp))));
    }
}

fn fmax(a: f32, b: f32) f32 {
    return if (std.math.isNan(a)) b else if (std.math.isNan(b)) a else @max(a, b);
}

/// bf16 bits in value order, -0 just below +0.
fn key(b: u16) i32 {
    return if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) - 1 else b;
}

fn isNan16(b: u16) bool {
    return b & 0x7f80 == 0x7f80 and b & 0x7f != 0;
}

/// bf16 steps between two outputs in value order (NaN against NaN 0, NaN against a value maxInt).
pub fn steps(a: u16, b: u16) u32 {
    if (isNan16(a) or isNan16(b)) return if (isNan16(a) and isNan16(b)) 0 else std.math.maxInt(u32);
    return @abs(key(a) - key(b));
}

/// One fp32 ulp at |x| (2^-149 below the normal range).
fn ulp32(x: f64) f64 {
    if (x == 0 or !std.math.isFinite(x)) return 0x1p-149;
    return @max(std.math.ldexp(@as(f64, 1), std.math.frexp(x).exponent - 24), 0x1p-149);
}

/// What one run (an implementation on a layout) gave, over its outputs.
pub const Acc = struct {
    outputs: usize = 0,
    inside: usize = 0,
    stats: ar.Stats = .{},
    class_bad: usize = 0,
    got_nonfinite: usize = 0,
    out_steps: u32 = 0,
    out_ulps: f64 = 0,
    pt_steps: u32 = 0,
    pt_ulps: f64 = 0,
    /// |got - point| (outside: past the range) in bf16 steps at the head's largest |point| (the head's scale)
    hs_out: f64 = 0,
    hs_pt: f64 = 0,
    /// inside kx's own form (kxForm sliding, kxSplitForm global; its subnormal q'' as +0), inside kxFallbackForm,
    /// inside either
    kx_in: [3]usize = .{ 0, 0, 0 },
    eager_eq: usize = 0,
    eager_steps: u32 = 0,
    eager_hs: f64 = 0,
    /// outputs outside by head-scale steps past the range: <= 0.5, 1, 2, 4, 16, more
    hist: [hist_edges.len + 1]usize = @splat(0),
    escape: usize = 0,
    /// (row, head) vectors whose replayForm range (or, for kx, either own form's) core refused (error.OutsideDomain)
    refused: usize = 0,
    /// finite outputs by |got - point| in head-scale steps, bins of 1/pt_scale up to pt_bins / pt_scale (the real-layer check's
    /// p50 and p99), the last bin everything above
    pt_hist: [pt_bins]u32 = @splat(0),
    /// the largest finite |score| any (row, head) had (the real-layer check's precondition)
    smax: f64 = 0,

    pub fn add(a: *Acc, b: Acc) void {
        inline for (.{ "outputs", "inside", "class_bad", "got_nonfinite", "eager_eq", "escape", "refused" }) |f| @field(a, f) += @field(b, f);
        inline for (.{ "decided", "ranged", "nonfinite" }) |f| @field(a.stats, f) += @field(b.stats, f);
        inline for (.{ "out_steps", "out_ulps", "pt_steps", "pt_ulps", "hs_out", "hs_pt", "eager_steps", "eager_hs" }) |f| @field(a, f) = @max(@field(a, f), @field(b, f));
        for (&a.kx_in, b.kx_in) |*x, y| x.* += y;
        for (&a.hist, b.hist) |*x, y| x.* += y;
        for (&a.pt_hist, b.pt_hist) |*x, y| x.* += y;
        a.smax = @max(a.smax, b.smax);
    }

    /// The q-quantile (0 .. 1) of |got - point| in head-scale steps (a bin's upper edge).
    pub fn quantile(a: *const Acc, q: f64) f64 {
        var total: u64 = 0;
        for (a.pt_hist) |x| total += x;
        if (total == 0) return 0;
        const want: u64 = @intFromFloat(@ceil(q * @as(f64, @floatFromInt(total))));
        var run: u64 = 0;
        for (a.pt_hist, 0..) |x, i| {
            run += x;
            if (run >= @max(want, 1)) return @as(f64, @floatFromInt(i + 1)) / pt_scale;
        }
        return @as(f64, @floatFromInt(pt_bins)) / pt_scale;
    }
};

pub const hist_edges = [_]f64{ 0.5, 1, 2, 4, 16 };
pub const pt_scale = 512.0;
pub const pt_bins = 2048;

/// (row, head) vectors of D outputs whose bytes differ between the outputs `a` and `b` of two runs.
pub fn vectorsDiffer(a: []const u16, b: []const u16, vectors: usize) usize {
    var n: usize = 0;
    for (0..vectors) |v| n += @intFromBool(!std.mem.eql(u16, a[v * D ..][0..D], b[v * D ..][0..D]));
    return n;
}

/// Failed checks among a case's byte comparisons: eager run pairs whose layout or plan differ and whose bytes do
/// (one), and any implementation, kx included, whose slices put together differ from its unsplit run (one): the
/// rank slices must give the unsplit heads' bytes whatever ranges the outputs sit in.
pub fn byteFailures(eager_differ: usize, slices: []const usize) usize {
    var slice_differ: usize = 0;
    for (slices) |x| slice_differ += x;
    return @as(usize, @intFromBool(eager_differ > 0)) + @intFromBool(slice_differ > 0);
}

/// kx-fallback's coupling check reads kx-safe's run of the same forward, which the case loop (profiles innermost)
/// has made only when kx-safe comes earlier in the list: every kx-fallback needs a kx-safe before it.
pub fn checkProfiles(profiles: []const Profile) error{FallbackNeedsKxSafe}!void {
    var safe = false;
    for (profiles) |p| switch (p) {
        .@"kx-safe" => safe = true,
        .@"kx-fallback" => if (!safe) return error.FallbackNeedsKxSafe,
        else => {},
    };
}

/// A forward: the kx-safe run a coupling check reads must be of this exact one.
pub const Forward = struct { kind: Kind, p0: u64, R: usize };

/// The saved kx-safe run (`saved`, null when none or when the panel overwrote it) must be of forward `now`.
pub fn checkBaseline(saved: ?Forward, now: Forward) error{FallbackNeedsKxSafe}!void {
    if (!std.meta.eql(saved orelse return error.FallbackNeedsKxSafe, now)) return error.FallbackNeedsKxSafe;
}

/// max over columns of sigmoid(gate) * sum_j p_j |v_j| / sum p, from the logit-form weights core left in `w`.
fn magnitude(h: rb.Head, lo: u64, n: usize, w: []const f64) f64 {
    const at: usize = @intCast(lo - h.base);
    const sink = bfr.value(h.sink) != -std.math.inf(f32);
    var l: f64 = 0;
    for (w[0 .. n + @intFromBool(sink)]) |x| l += x;
    var col: [D]f64 = @splat(0);
    for (0..n) |j| for (&col, h.v[(at + j) * D ..][0..D]) |*c, x| {
        c.* += w[j] * @abs(@as(f64, bfr.value(x)));
    };
    var top: f64 = 0;
    for (col, h.gate) |c, gb| top = @max(top, c / l / (1 + @exp(-@as(f64, bfr.value(gb)))));
    return top;
}

/// kx's sliding-layer key split (cuda_attn.zig kx_sliding_split): absolute 256-key splits from the window start's.
pub const sliding_split: u64 = 256;

/// Ranges and replays of one (row, head), shared by every run's outputs of it.
pub const Ref = struct {
    pt: [3][D]f64 = undefined,
    hf: [3][D]f64 = undefined,
    eager: [3][D]u16 = undefined,
    escape: [3]bool = .{ false, false, false },
    /// core refused the form (error.OutsideDomain: an fp32 sum may overflow, or a class is undecided)
    refused: [3]bool = .{ false, false, false },
    /// the head's output scale before cancellation: max over columns of sigmoid(gate) * sum_j p_j |v_j| (float64)
    scale: f64 = 0,
    /// the largest finite |score| of the head (replayForm's pass)
    smax: f64 = 0,

    /// replayForm (0) always; kx's own form (1: kxSplitForm at S global, at sliding_split sliding, on q'' with its
    /// subnormal elements as +0, as kx loads them) and kxFallbackForm (2) when `kx`; the eager replays at k_exp 1,
    /// 1 +- 2^-22.
    pub fn make(r: *Ref, h: rb.Head, g: rb.Geo, kx: bool, S: u64, wk: *rb.Work) void {
        const lo = rb.keyLo(g, h.pos);
        const n: usize = @intCast(h.pos - lo + 1);
        const forms = [3]rb.Form{ rb.replayForm(n, D), rb.kxSplitForm(lo, h.pos, D, if (g.window == 0) S else sliding_split), rb.kxFallbackForm(n, D) };
        var qf: [D]u16 = undefined;
        for (&qf, h.q[0..D]) |*o, x| o.* = if (x & 0x7f80 == 0 and x & 0x7f != 0) 0 else x;
        var hk = h;
        hk.q = &qf;
        for (forms, 0..) |f, i| {
            if (i > 0 and !kx) break;
            rb.core(if (i == 1) hk else h, g, f, wk, &r.pt[i], &r.hf[i]) catch {
                r.refused[i] = true; // no range: the outputs meet NaN points and the case fails on `refused`
                @memset(&r.pt[i], std.math.nan(f64));
                @memset(&r.hf[i], std.math.inf(f64));
                continue;
            };
            if (i == 0) {
                r.scale = magnitude(h, lo, n, wk.w);
                r.smax = wk.smax;
            }
            rb.gated(h.gate, &r.pt[i], &r.hf[i]);
            for (r.pt[i], r.hf[i]) |p, s| if (std.math.isFinite(p) and std.math.isInf(s)) {
                r.escape[i] = true;
            };
        }
        for ([_]f64{ 1, 1 + 0x1p-22, 1 - 0x1p-22 }, &r.eager) |k, *o| eagerFork(h, g, k, wk.e, o);
    }

    /// `got` (D outputs of an implementation) into `a`; `kx` also counts kx's own forms.
    pub fn check(r: *const Ref, got: []const u16, kx: bool, a: *Acc) void {
        if (r.escape[0]) a.escape += 1;
        if (r.refused[0] or (kx and (r.refused[1] or r.refused[2]))) a.refused += 1;
        a.smax = @max(a.smax, r.smax);
        var top: f64 = r.scale;
        if (!(top > 0 and std.math.isFinite(top))) for (r.pt[0]) |p| if (std.math.isFinite(p)) {
            top = @max(top, @abs(p));
        };
        const hstep = ulp32(top) * 0x1p16; // a bf16 step at the head's scale
        for (got, 0..) |x, e| {
            const p = r.pt[0][e];
            const s = r.hf[0][e];
            a.outputs += 1;
            if (ar.within(p, s, x, &a.stats)) a.inside += 1;
            const st = rb.outside(p, s, x);
            if (st == std.math.maxInt(u32)) a.class_bad += 1 else a.out_steps = @max(a.out_steps, st);
            const v: f64 = bfr.value(x);
            if (!std.math.isFinite(v)) a.got_nonfinite += 1;
            if (std.math.isFinite(p) and std.math.isFinite(v)) {
                const u = ulp32(p);
                // past the range's ends rounded to bf16 (the output's own rounding is allowed, as ar.within allows it)
                const lo = bfr.bf16(p - s);
                const hi = bfr.bf16(p + s);
                const past = if (std.math.isFinite(s)) @max(0, lo - v, v - hi) else 0;
                a.out_ulps = @max(a.out_ulps, past / u);
                a.pt_ulps = @max(a.pt_ulps, @abs(v - p) / u);
                a.hs_out = @max(a.hs_out, past / hstep);
                if (past > 0) a.hist[
                    for (hist_edges, 0..) |edge, b| {
                        if (past / hstep <= edge) break b;
                    } else hist_edges.len
                ] += 1;
                a.hs_pt = @max(a.hs_pt, @abs(v - p) / hstep);
                const hs = @abs(v - p) / hstep * pt_scale;
                if (hs >= 0) a.pt_hist[@intFromFloat(@min(@as(f64, pt_bins - 1), @floor(hs)))] += 1;
                a.pt_steps = @max(a.pt_steps, steps(x, bfr.bits(@floatCast(p))));
            }
            if (kx) {
                var st2: ar.Stats = .{};
                const in = [2]bool{ ar.within(r.pt[1][e], r.hf[1][e], x, &st2), ar.within(r.pt[2][e], r.hf[2][e], x, &st2) };
                for (in, 0..) |b, i| a.kx_in[i] += @intFromBool(b);
                a.kx_in[2] += @intFromBool(in[0] or in[1]);
            }
            var near: u32 = std.math.maxInt(u32);
            var far: f64 = std.math.inf(f64);
            for (r.eager) |ev| {
                near = @min(near, steps(x, ev[e]));
                far = @min(far, @abs(v - bfr.value(ev[e])));
            }
            if (near == 0) a.eager_eq += 1 else if (near != std.math.maxInt(u32)) a.eager_steps = @max(a.eager_steps, near);
            if (std.math.isFinite(far)) a.eager_hs = @max(a.eager_hs, far / hstep);
        }
    }
};

test "the fork-order eager replay equals the reference's eager replay where one tile and one split hold every key" {
    const gpa = std.testing.allocator;
    var in = try Inputs.alloc(gpa, 64, 1);
    defer in.free(gpa);
    in.profile = .clip;
    in.kind = .global;
    in.p0 = 20;
    in.R = 1;
    in.from = 0;
    in.npos = 21;
    for (0..2 * HK) |i| in.fillKv(i);
    in.plant();
    var wk = try rb.Work.init(gpa, 64);
    defer wk.deinit(gpa);
    var y: [D]f32 = undefined;
    var a: [D]u16 = undefined;
    var b: [D]u16 = undefined;
    var differ: usize = 0;
    for (0..HQ) |h| {
        in.fillRow(h);
        const hd = in.head(0, h);
        rb.replay(hd, in.geo(), .eager, 1, &wk, &y);
        rb.round(&y, &a);
        eagerFork(hd, in.geo(), 1, wk.e, &b);
        for (a, b) |x, z| differ += @intFromBool(x != z);
    }
    // one tile: the replay's z starts at the sink's weight and adds the keys; the fork's folds (sink, 1) with the
    // tile's pair, so z may differ by a rounding and a P by a step: few outputs only
    try std.testing.expect(differ * 100 < HQ * D);
}

test "a head the reference refuses counts as refused, not as a range" {
    const gpa = std.testing.allocator;
    var q: [D]u16 = @splat(0);
    var v: [2 * D]u16 = @splat(0x7f00); // 2^127: two such values overflow the fp32 sum
    var k: [2 * D]u16 = @splat(0);
    var rel: [rb.d_rel]u16 = @splat(0);
    const proj = try gpa.alloc(u16, rb.d_rel * E);
    defer gpa.free(proj);
    @memset(proj, 0);
    var gate: [D]u16 = @splat(0);
    const h: rb.Head = .{ .pos = 1, .base = 0, .q = &q, .k = &k, .v = &v, .rel = &rel, .proj = proj, .sink = 0xff80, .gate = &gate };
    var wk = try rb.Work.init(gpa, 4);
    defer wk.deinit(gpa);
    var r: Ref = .{};
    r.make(h, .{ .window = 0, .extent = E }, true, 1024, &wk);
    try std.testing.expect(r.refused[0] and r.refused[1] and r.refused[2]);
    var a: Acc = .{};
    r.check(&r.eager[0], false, &a);
    try std.testing.expectEqual(@as(usize, 1), a.refused);
    try std.testing.expectEqual(@as(usize, 0), a.inside);
    @memset(v[D..], 0x3f80); // the second value 1: the sums stay finite, and the reference gives a range
    r = .{};
    r.make(h, .{ .window = 0, .extent = E }, true, 1024, &wk);
    try std.testing.expect(!r.refused[0] and !r.refused[1] and !r.refused[2]);
    a = .{};
    r.check(&r.eager[0], false, &a);
    try std.testing.expectEqual(@as(usize, 0), a.refused);
}

test "a slice run one bf16 step off its unsplit run fails the case" {
    var unsplit: [2 * HQ * D]u16 = @splat(0x3f80);
    var split = unsplit;
    split[HQ * D + 5] = 0x3f81; // row 1, head 0, column 5: one bf16 step, a value a wide range may well hold
    const slices = [4]usize{ 0, 0, 0, vectorsDiffer(&unsplit, &split, 2 * HQ) }; // fused, rows, prev, kx
    try std.testing.expectEqual(@as(usize, 1), slices[3]);
    try std.testing.expectEqual(@as(usize, 1), byteFailures(0, &slices));
    try std.testing.expectEqual(@as(usize, 1), byteFailures(0, &.{ 1, 0, 0, 0 }));
    try std.testing.expectEqual(@as(usize, 2), byteFailures(3, &slices));
    try std.testing.expectEqual(@as(usize, 1), byteFailures(3, &.{ 0, 0, 0, 0 }));
    try std.testing.expectEqual(@as(usize, 0), byteFailures(0, &.{ 0, 0, 0, 0 }));
}

test "kx-fallback is refused unless kx-safe comes before it in the profile list" {
    try checkProfiles(&.{ .clip, .diffuse, .planted, .@"kx-safe", .@"kx-fallback" });
    try checkProfiles(&.{ .@"kx-safe", .clip, .@"kx-fallback" });
    try checkProfiles(&.{ .clip, .@"kx-safe" });
    try std.testing.expectError(error.FallbackNeedsKxSafe, checkProfiles(&.{ .@"kx-fallback", .@"kx-safe" }));
    try checkProfiles(&.{ .@"kx-safe", .@"kx-fallback", .clip, .@"kx-fallback" }); // clip leaves kx-safe's run
    try std.testing.expectError(error.FallbackNeedsKxSafe, checkProfiles(&.{.@"kx-fallback"}));
}

test "the coupling check takes kx-safe's run of the same forward only" {
    const f: Forward = .{ .kind = .sliding, .p0 = 1024, .R = 8 };
    try checkBaseline(f, f);
    try std.testing.expectError(error.FallbackNeedsKxSafe, checkBaseline(null, f));
    try std.testing.expectError(error.FallbackNeedsKxSafe, checkBaseline(.{ .kind = .global, .p0 = 1024, .R = 8 }, f));
    try std.testing.expectError(error.FallbackNeedsKxSafe, checkBaseline(.{ .kind = .sliding, .p0 = 1023, .R = 8 }, f));
    try std.testing.expectError(error.FallbackNeedsKxSafe, checkBaseline(.{ .kind = .sliding, .p0 = 1024, .R = 16 }, f));
}
