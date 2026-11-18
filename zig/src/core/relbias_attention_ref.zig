//! Inkling-2's relative-bias attention: float64 at the fp32 form's points, the range fp32 allows, a fixed-order replay.
//! Copy of the standalone float64 reference with v0.2's kx forms added, a deviation from it:
//! kxSplitForm (absolute S-key splits and their ordered merge: three more roundings in num and den a merge, one more
//! rescale on every weight), the P-limb flush term of both kx forms (3 * 2^-126 * sum_j |v_j| / l, absolute), kx's
//! forms refusing subnormal q, K or V by name (kx loads them as +0: error.SubnormalInput), and Work.smax (the largest
//! finite |score|, for the real-layer check's precondition).
const std = @import("std");
const bfr = @import("bf16_round");
const ar = @import("attention_range.zig");

const u: f64 = 0x1p-24;
const lim: f64 = 0x1p127; // the domain's bound on a sum's terms: with gamma < 1/3 no fp32 partial sum reaches 2^128
const inf = std.math.inf(f64);
const isNan = std.math.isNan;
const isFinite = std.math.isFinite;

fn gamma(n: f64) f64 {
    return n * u / (1 - n * u);
}

fn val(b: u16) f64 {
    return bfr.value(b);
}

/// The widest head dimension the stack buffers hold.
pub const max_d = 256;
pub const d_rel = 16;
pub const scale32: f32 = @bitCast(@as(u32, 0x3db504f3));
pub const qk_thr32: f32 = @bitCast(@as(u32, 0x41b504f3));
pub const rel_thr32: f32 = 8.0;
pub const eps32: f32 = @bitCast(@as(u32, 0x358637bd));
pub const inv_floor32: f32 = @bitCast(@as(u32, 0x3703126f));
pub const alpha32: f32 = @bitCast(@as(u32, 0x3dcccccd));

/// A layer kind's keys: `window` keys a row sees (0: every earlier key), `extent` the bias's distances, `d` head dim.
pub const Geo = struct { window: u32, extent: u32, d: u32 = 128 };

/// The first key position a row at `pos` sees.
pub fn keyLo(g: Geo, pos: u64) u64 {
    return if (g.window == 0 or pos < g.window) 0 else pos - g.window + 1;
}

/// One (row, query head) as bf16 bits: k, v [count][d] hold positions base .. pos (base <= keyLo); proj [16][extent].
pub const Head = struct { pos: u64, base: u64, q: []const u16, k: []const u16, v: []const u16, rel: []const u16, proj: []const u16, sink: u16, gate: []const u16 };

fn chain(rel: []const u16, proj: []const u16, extent: u32, dist: usize, terms: usize) f32 {
    var acc: f32 = 0;
    for (0..terms) |i| acc = @mulAdd(f32, bfr.value(rel[i]), bfr.value(proj[i * extent + dist]), acc);
    return acc;
}

/// The bias's fp32 chain i = 0 .. 15 from +0 rounded to bf16 (fixed bytes); +0 at dist >= extent.
pub fn bias(rel: []const u16, proj: []const u16, extent: u32, dist: u64) f32 {
    if (dist >= extent) return 0;
    return bfr.round32(chain(rel, proj, extent, @intCast(dist), d_rel));
}

/// error.OracleDrift when bias's fp32 chain is not within gamma(16) of its float64 sum (the products are exact).
pub fn biasSane(rel: []const u16, proj: []const u16, extent: u32, dist: u64) error{OracleDrift}!void {
    if (dist >= extent) return;
    return drift(chain(rel, proj, extent, @intCast(dist), d_rel), rel, proj, extent, @intCast(dist));
}

fn drift(c: f32, rel: []const u16, proj: []const u16, extent: u32, dist: usize) error{OracleDrift}!void {
    var s: f64 = 0;
    var a: f64 = 0;
    for (0..d_rel) |i| {
        const p = val(rel[i]) * val(proj[i * extent + dist]);
        s += p;
        a += @abs(p);
    }
    if (!isFinite(a) and !isFinite(c)) return; // NaN or Inf inputs: no rounding to measure
    // each of the 15 adds rounds within u of its sum, or within 2^-150 below the normal range
    if (!(@abs(@as(f64, c) - s) <= gamma(d_rel) * a + d_rel * 0x1p-150)) return error.OracleDrift;
}

pub const Sink = enum { logit, multiplier };

/// A form's fp32 steps: attention_range's counts (scale: inner.score_round), the bias add, the sink, online rescales;
/// `split` 1: one more rescale on every weight (its S-key split's merge factor); `flush`: the P limbs flushed below
/// 2^-126 (kx), and subnormal q, K or V outside the domain.
pub const Form = struct { inner: ar.Form, bias_round: f64 = 1, sink: Sink = .logit, tile: u32 = 0, log_ulps: f64 = 2, split: u32 = 0, flush: bool = false };

/// The fixed-order replay's form over n keys at head dim d (sequential fp32 chains, two passes).
pub fn replayForm(n: usize, d: usize) Form {
    const c: f64 = @floatFromInt(n + 1);
    return .{ .inner = .{ .score = @floatFromInt(d), .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = c, .den = c, .nan_max = true } };
}

/// kortexa e2614fe's fast path over keys lo .. pos (mma Q.K, 64-key absolute tiles, three-limb P x V, multiplier sink).
pub fn kxForm(lo: u64, pos: u64, d: usize) Form {
    const t: f64 = @floatFromInt(pos / 64 - lo / 64 + 1);
    const steps: f64 = @floatFromInt(d / 16);
    return .{ .inner = .{ .score = steps * ar.mma_step, .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = t * (12 * ar.mma_step + 1), .den = 8 + 3 + 2 * t, .nan_max = true }, .sink = .multiplier, .tile = 64, .flush = true };
}

/// v0.2's kx over keys lo .. pos in absolute S-key splits merged in order: kxForm plus, for each of the
/// L = pos / S - lo / S merges, three roundings in num and den (O * a + o * b; Z * a + z * b) and one more rescale on
/// every weight (its split's factor b; the later merges' a factors are the tile max moves kxForm already counts).
/// With one split (L = 0) it is kxForm.
pub fn kxSplitForm(lo: u64, pos: u64, d: usize, S: u64) Form {
    var f = kxForm(lo, pos, d);
    const L: f64 = @floatFromInt(pos / S - lo / S);
    if (L == 0) return f;
    f.inner.num += 3 * L;
    f.inner.den += 3 * L;
    f.split = 1;
    return f;
}

/// kortexa e2614fe's per-row fallback over n keys (block-tree dot, per-key online softmax, multiplier sink).
pub fn kxFallbackForm(n: usize, d: usize) Form {
    const c: f64 = @floatFromInt(n);
    const tree: f64 = @floatFromInt(5 + std.math.log2_int(usize, d / 32)); // a warp's 5 levels, then the warps'
    return .{ .inner = .{ .score = tree, .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = 3 * c, .den = 2 * c, .nan_max = true }, .sink = .multiplier, .tile = 1 };
}

/// Scratch for up to n keys plus the sink.
pub const Work = struct {
    w: []f64,
    tau: []f64,
    rescales: []f64,
    e: []f32,
    /// the last core's largest finite |score| (bias included)
    smax: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, n: usize) !Work {
        const w = try gpa.alloc(f64, n + 1);
        errdefer gpa.free(w);
        const tau_s = try gpa.alloc(f64, n + 1);
        errdefer gpa.free(tau_s);
        const rescales = try gpa.alloc(f64, n + 1);
        errdefer gpa.free(rescales);
        return .{ .w = w, .tau = tau_s, .rescales = rescales, .e = try gpa.alloc(f32, n + 1) };
    }

    pub fn deinit(wk: *Work, gpa: std.mem.Allocator) void {
        gpa.free(wk.w);
        gpa.free(wk.tau);
        gpa.free(wk.rescales);
        gpa.free(wk.e);
    }
};

/// point[d] float64's o at the fp32 form's points, half[d] f's fp32 spread, before the gate; refuses overflow.
pub fn core(h: Head, g: Geo, f: Form, wk: *Work, point: []f64, half: []f64) error{ OutsideDomain, SubnormalInput }!void {
    const r = try pass(h, g, f, f.sink == .logit, wk, point, half);
    if (f.sink == .multiplier) try multiplier(val(h.sink), r, f, point[0..g.d], half[0..g.d]);
}

/// kx: o * sigma(m + logf(z) - sink), the logit form's point in real arithmetic; the range holds every fp32 factor.
fn multiplier(sink: f64, r: Pass, f: Form, point: []f64, half: []f64) error{OutsideDomain}!void {
    if (sink == -inf) return; // absent (rule 3): kx's 1 / (1 + expf(-Inf)) is exactly 1, and the product exact
    const lse = r.m + @log(r.l);
    const arg = lse - sink;
    const phi = 1 / (1 + @exp(-arg));
    // the fp32 argument's error: z, logf, the add and the subtract; at a +Inf sink it is -Inf in fp32 too, factor +0
    const da = if (sink == inf) 0 else (r.tmax / (1 - r.tmax) + gamma(f.inner.den) + f.log_ulps * 0x1p-23 * @abs(@log(r.l)) + u * @abs(lse) + u * @abs(arg)) * (1 + 0x1p-20);
    const lo = sigmoid32(arg - da, false);
    const hi = sigmoid32(arg + da, true);
    for (point, half) |*p, *hf| {
        if (std.math.isInf(p.*)) { // fp32's factor is +0 where expf overflows: Inf * 0 = NaN
            if (hi == 0 or isNan(phi)) p.* = std.math.nan(f64) else if (!(lo > 0)) return error.OutsideDomain;
        } else if (isFinite(p.*) and isFinite(hf.*)) {
            const c = p.* * phi;
            var w: f64 = if (lo >= 0 and hi >= lo) 0 else inf; // a NaN end: no range
            var top: f64 = 0;
            for ([4]f64{ (p.* - hf.*) * lo, (p.* - hf.*) * hi, (p.* + hf.*) * lo, (p.* + hf.*) * hi }) |x| {
                w = @max(w, @abs(x - c));
                top = @max(top, @abs(x));
            }
            hf.* = (w + u * top) * (1 + 0x1p-20); // and the product's rounding; its subnormal 2^-150 is gated's 2^-149
        }
        p.* *= phi;
    }
}

/// fp32's 1 / (1 + expf(-x)) at its lowest or (`up`) highest: expf within 2^-22 and overflowing to +Inf, two RNs.
fn sigmoid32(x: f64, up: bool) f64 {
    if (isNan(x)) return x; // @max and @min below would drop it
    const ex = @exp(-x);
    const cut = 0x1p128 * (1 - 0x1p-25); // RN to fp32 is +Inf from FLT_MAX + half an ulp
    if (up) return if (ex * (1 - 0x1p-21) >= cut) 0 else @min(1, (1 + 0x1p-21) / (1 + @max(0, ex * (1 - 0x1p-21) - 0x1p-149)) + 0x1p-149);
    return if (ex * (1 + 0x1p-21) >= cut) 0 else @max(0, (1 - 0x1p-21) / (1 + ex * (1 + 0x1p-21)) - 0x1p-149);
}

const Pass = struct { l: f64, m: f64, tmax: f64 };

/// attention_range.head's arithmetic with the bias, the sink as a zero-value weight, and online rescales.
fn subnormal(b: u16) bool {
    return b & 0x7f80 == 0 and b & 0x7f != 0;
}

fn pass(h: Head, g: Geo, f: Form, with_sink: bool, wk: *Work, point: []f64, half: []f64) error{ OutsideDomain, SubnormalInput }!Pass {
    const d: usize = g.d;
    for ([_]f64{ f.inner.score, f.inner.num, f.inner.den }) |c| if (!(c * u <= 0.25)) return error.OutsideDomain;
    const lo = keyLo(g, h.pos);
    const n: usize = @intCast(h.pos - lo + 1);
    const at: usize = @intCast(lo - h.base);
    if (f.flush) for ([_][]const u16{ h.q[0..d], h.k[at * d .. (at + n) * d], h.v[at * d .. (at + n) * d] }) |xs| for (xs) |x| {
        if (subnormal(x)) return error.SubnormalInput; // kx loads it as +0: outside these forms
    };
    wk.smax = 0;
    const scale: f64 = scale32;
    const w = wk.w;
    const tau_s = wk.tau;
    var m: f64 = -inf;
    var dmax: f64 = 0;
    for (0..n) |j| {
        const kj = h.k[(at + j) * d ..][0..d];
        var s: f64 = 0;
        var a: f64 = 0;
        var af: f64 = 0; // the finite products' magnitudes: an infinite one sets the score's class in both
        for (0..d) |e| {
            const p = val(h.q[e]) * val(kj[e]);
            s += p;
            a += @abs(p);
            if (isFinite(p)) af += @abs(p);
        }
        const t = s * scale;
        const b: f64 = bias(h.rel, h.proj, g.extent, n - 1 - j);
        if (af > lim or af * scale + (if (isFinite(b)) @abs(b) else 0) > lim) return error.OutsideDomain; // dot, t + b
        w[j] = if (b != 0) t + b else t; // t + (+-0) is exact in fp32: not charged
        if (isFinite(w[j])) wk.smax = @max(wk.smax, @abs(w[j]));
        tau_s[j] = f.inner.score * u * a * @abs(scale) + f.inner.score_round * u * @abs(t);
        if (b != 0) tau_s[j] += f.bias_round * u * @abs(w[j]);
        m = if (isNan(w[j]) or isNan(m)) std.math.nan(f64) else @max(m, w[j]);
        if (isFinite(tau_s[j])) dmax = @max(dmax, tau_s[j]);
    }
    var nw = n;
    if (with_sink and val(h.sink) != -inf) {
        w[n] = val(h.sink);
        tau_s[n] = 0;
        m = if (isNan(w[n]) or isNan(m)) std.math.nan(f64) else @max(m, w[n]);
        nw = n + 1;
    }
    rescaleCounts(f.tile, lo, n, nw > n, dmax, w, wk.rescales);
    if (f.split != 0) for (wk.rescales[0..nw]) |*r| {
        r.* += 1; // the split's merge factor b
    };
    var l: f64 = 0;
    var tmax: f64 = 0;
    var vmax: f64 = 0;
    for (w[0..nw], tau_s[0..nw], wk.rescales[0..nw], 0..) |*p, *t, r, j| {
        const x = (m - p.*) + 2 * dmax;
        var e = t.* + u * x + if (f.inner.exp_ulps) |c| (c[0] + c[1] * x) * 0x1p-23 else if (f.inner.fast_exp) (3 + 1.2 * x) * 0x1p-23 else 0x1p-22;
        if (f.inner.chunk_exp) e += 0x1p-22 + u * x;
        if (r > 0) e += r * (0x1p-22 + u * x); // each later rescale: its expf and the subtract under it
        // fp32's weight is within exp(+-(e + dmax)) of exp(-gap): +0 below 2^-151 or at a -Inf score, kept over 2^-126
        const gap = m - p.*;
        if (isFinite(m) and (p.* == -inf or gap - e - dmax > 151 * std.math.ln2)) {
            p.* = 0; // exactly +0 in fp32 too: no spread, and 0 * Inf = NaN in the point as in fp32
            t.* = 0;
            continue;
        }
        if (isFinite(m) and j < n and !(gap + e + dmax < 126 * std.math.ln2)) for (h.v[(at + j) * d ..][0..d]) |vb| {
            if (std.math.isInf(val(vb))) return error.OutsideDomain; // Inf times a weight fp32 may flush: Inf or NaN
        };
        p.* = @exp(p.* - m);
        l += p.*;
        t.* = std.math.expm1(e);
        if (!isNan(t.*)) tmax = @max(tmax, t.*); // an infinite spread on a kept weight takes the escape below
    }
    var cs: [max_d]f64 = @splat(0); // each column's sum of finite |v|: P x V's partial sums stay under it
    for (h.v[at * d .. (at + n) * d], 0..) |x, i| if (isFinite(val(x))) {
        vmax = @max(vmax, @abs(val(x)));
        cs[i % d] += @abs(val(x));
    };
    for (cs[0..d]) |c| if (c > lim) return error.OutsideDomain;
    const big = tmax >= 0.5; // scores too far apart for the spread's first order: any finite value
    const kappa = (1 + tmax) / (1 - tmax);
    const tiny = @as(f64, @floatFromInt(nw + 8)) * 0x1p-120 * (1 + vmax);
    var rs: [max_d]f64 = @splat(0);
    var dv: [max_d]f64 = @splat(0);
    @memset(point[0..d], 0);
    for (0..n) |j| for (point[0..d], h.v[(at + j) * d ..][0..d]) |*o, x| { // row-major, each column's sum in j order
        o.* += w[j] * val(x);
    };
    for (point[0..d]) |*o| o.* /= l;
    for (0..n) |j| for (point[0..d], h.v[(at + j) * d ..][0..d], rs[0..d], dv[0..d]) |o, x, *r, *v| {
        r.* += w[j] * @abs(val(x));
        v.* += w[j] * tau_s[j] * @abs(val(x) - o);
    };
    for (point[0..d], half[0..d], rs[0..d], dv[0..d], cs[0..d]) |o, *hf, r_, dv_, c_| {
        if (!isFinite(o) or big) {
            hf.* = if (big) inf else 0;
            continue;
        }
        var v = dv_;
        if (nw > n) v += w[n] * tau_s[n] * @abs(0 - o); // the sink's value is 0
        const r = r_ / l;
        v /= l;
        const e1 = v / (1 - tmax);
        const en = (gamma(f.inner.num) + f.inner.p_rel) * r * kappa;
        const base = @abs(o) + e1 + en;
        hf.* = (e1 + en + (gamma(f.inner.den) + f.inner.div * u) * base) * (1 + 0x1p-20) + tiny;
        // kx: each weight's P limbs flushed below 2^-126 move its P.V term by at most 3 * 2^-126 * |v|
        if (f.flush) hf.* += 3 * 0x1p-126 * c_ / l;
    }
    return .{ .l = l, .m = m, .tmax = tmax };
}

/// R_j: the later absolute tiles whose max may move the running max (more than 2 dmax under it is exact alpha = 1).
fn rescaleCounts(tile: u32, lo: u64, n: usize, sink: bool, dmax: f64, w: []const f64, out: []f64) void {
    if (tile == 0) return @memset(out[0 .. n + 1], 0);
    var run: f64 = if (sink) w[n] else -inf;
    var total: f64 = 0;
    var j: usize = 0;
    while (j < n) {
        const end: usize = @min(n, @as(usize, @intCast(((lo + j) / tile + 1) * tile - lo)));
        var tm: f64 = -inf;
        for (w[j..end]) |x| tm = @max(tm, x);
        if (tm > run - 2 * dmax) total += 1;
        run = @max(run, tm);
        @memset(out[j..end], total);
        j = end;
    }
    for (out[0..n]) |*r| r.* = total - r.*;
    out[n] = total;
}

/// The gate: point and half of o * sigmoid(gate) after its fp32 steps, before the output's bf16 rounding.
pub fn gated(gate: []const u16, point: []f64, half: []f64) void {
    for (gate, point[0..gate.len], half[0..gate.len]) |gb, *p, *hf| {
        const x = val(gb);
        const sig = if (x < -88.75) 0 else 1 / (1 + @exp(-x)); // fp32 expf(-x) is +Inf for a bf16 x <= -89: G = +0
        const a = @abs(p.*) + hf.*;
        if (!isFinite(p.*)) hf.* = 0 else if (isFinite(hf.*)) hf.* = (hf.* * sig + a * sig * 0x1p-21 + u * a * sig + a * 0x1p-126) * (1 + 0x1p-20) + 0x1p-149;
        p.* *= sig;
    }
}

/// bf16 bits in value order, -0 just below +0.
fn key(b: u16) i32 {
    return if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) - 1 else b;
}

fn bitsOf(v: f64) u16 {
    return bfr.bits(@floatCast(bfr.bf16(v)));
}

/// bf16 bits against point +- half: 0 inside, else the bf16 steps past the nearer end (value order, -0 below +0).
pub fn outside(point: f64, half: f64, got: u16) u32 {
    const nan = got & 0x7f80 == 0x7f80 and got & 0x7f != 0;
    if (isNan(point)) return if (nan) 0 else std.math.maxInt(u32);
    if (std.math.isInf(point)) return if (got == bitsOf(point)) 0 else std.math.maxInt(u32);
    if (nan) return std.math.maxInt(u32);
    const lo = key(bitsOf(point - half));
    const hi = key(bitsOf(point + half));
    const k = key(got);
    return @intCast(if (k < lo) lo - k else if (k > hi) k - hi else 0);
}

/// An fp32 value against point +- half in float64; NaN for NaN, infinities equal; counted as ar.within counts.
pub fn withinF32(point: f64, half: f64, got: f32, stats: *ar.Stats) bool {
    if (isNan(point)) {
        stats.nonfinite += 1;
        return isNan(got);
    }
    if (std.math.isInf(point)) {
        stats.nonfinite += 1;
        return got == point;
    }
    const lo = point - half;
    const hi = point + half;
    if (@as(f32, @floatCast(lo)) == @as(f32, @floatCast(hi))) stats.decided += 1 else stats.ranged += 1;
    return !isNan(got) and lo <= got and got <= hi;
}

/// The replay's arithmetic and the planted controls (each changes one thing, section 5.6).
pub const Variant = enum { form, eager, window_early, bias_minus, bias_plus, no_sink, bias_past_extent, p_bf16, nan_skipped };

fn expk(x: f32, k_exp: f64) f32 {
    return @floatCast(@exp(@as(f64, x)) * k_exp);
}

fn nanMax(a: f32, b: f32) f32 {
    return if (isNan(a) or isNan(b)) std.math.nan(f32) else @max(a, b);
}

fn fmax(a: f32, b: f32) f32 {
    return if (isNan(a)) b else if (isNan(b)) a else @max(a, b);
}

/// Fixed-order fp32 replay of `v`, every expf as fp32(exp64(x) * k_exp); y[d] fp32 before the output's bf16 rounding.
pub fn replay(h: Head, g: Geo, v: Variant, k_exp: f64, wk: *Work, y: []f32) void {
    const d: usize = g.d;
    var lo = keyLo(g, h.pos);
    if (v == .window_early and g.window > 0 and lo > 0) lo -= 1;
    const n: usize = @intCast(h.pos - lo + 1);
    const at: usize = @intCast(lo - h.base);
    const s = wk.e[0..n];
    const eager = v == .eager;
    for (s, 0..) |*sj, j| {
        const kj = h.k[(at + j) * d ..][0..d];
        var acc: f32 = 0;
        for (0..d) |e| acc = @mulAdd(f32, bfr.value(h.q[e]), bfr.value(kj[e]), acc); // products exact, d - 1 roundings
        if (eager) acc = bfr.round32(acc);
        var t = acc * scale32;
        if (eager) t = bfr.round32(t);
        const dist: i64 = @intCast(n - 1 - j);
        const dd = switch (v) {
            .bias_minus => dist - 1,
            .bias_plus => dist + 1,
            else => dist,
        };
        var sc = t;
        if (v == .bias_past_extent) {
            sc = t + bias(h.rel, h.proj, g.extent, @intCast(@min(dist, g.extent - 1)));
        } else if (dd >= 0 and dd < g.extent) sc = t + bias(h.rel, h.proj, g.extent, @intCast(dd));
        sj.* = if (eager) bfr.round32(sc) else sc;
    }
    const sink = bfr.value(h.sink);
    const skip = v == .nan_skipped;
    const drop_sink = v == .no_sink or (skip and isNan(sink));
    var m: f32 = if (drop_sink) -std.math.inf(f32) else sink;
    for (s) |x| m = if (skip) fmax(m, x) else nanMax(m, x);
    var z: f32 = if (drop_sink) 0 else expk(sink - m, k_exp); // the sink first, as the fork's (sink, 1) seed
    for (s) |*x| {
        const e = if (skip and isNan(x.*)) 0 else expk(x.* - m, k_exp);
        x.* = e;
        z = z + e;
    }
    const normalised = v == .p_bf16 or eager; // HF's P point: bf16(e_j / z), no later division
    if (normalised) for (s) |*x| {
        x.* = bfr.round32(x.* / z);
    };
    @memset(y[0..d], 0);
    for (s, 0..) |p, j| for (y[0..d], h.v[(at + j) * d ..][0..d]) |*o, x| { // row-major, each column's chain in j order
        o.* = @mulAdd(f32, p, bfr.value(x), o.*);
    };
    for (y[0..d], h.gate[0..d]) |*o, gb| {
        var yy = if (normalised) o.* else o.* / z;
        if (eager) yy = bfr.round32(yy);
        const gs: f32 = 1.0 / (1.0 + expk(-bfr.value(gb), k_exp));
        o.* = yy * gs;
    }
}

/// The output's bf16 bits of y.
pub fn round(y: []const f32, out: []u16) void {
    for (y, out[0..y.len]) |x, *o| o.* = bfr.bits(x);
}

/// The lane order's sum of squares of n fp32 values (n = 128 or 16).
pub fn sumSquares(x: []const f32) f32 {
    const per = @max(1, x.len / 32);
    var lane: [32]f32 = @splat(0);
    for (&lane, 0..) |*s, i| for (i * per..(i + 1) * per) |c| {
        if (c < x.len) s.* += x[c] * x[c];
    };
    var off: usize = 16;
    while (off > 0) : (off /= 2) {
        var next: [32]f32 = undefined;
        for (&next, 0..) |*o, i| o.* = lane[i] + lane[i ^ off];
        lane = next;
    }
    return lane[0];
}

/// Clip (P1-P7), then tau's rounding (P11) when tau is not null; out bf16 bits.
pub fn clip(x: []const u16, scale: ?[]const u16, thr: f32, tau_v: ?f32, out: []u16) void {
    var buf: [max_d]f32 = undefined;
    const xs = buf[0..x.len];
    for (xs, x, 0..) |*o, b, i| o.* = if (scale) |sc| bfr.value(b) * bfr.value(sc[i]) else bfr.value(b);
    const t = @sqrt(sumSquares(xs)) + eps32;
    var f = (1.0 / t) * thr;
    if (f > 1) f = 1; // a NaN passes
    for (out[0..x.len], xs) |*o, xv| {
        var yv = bfr.round32(xv * f);
        if (tau_v) |tv| yv = bfr.round32(yv * tv);
        o.* = bfr.bits(yv);
    }
}

/// V norm (P12-P14) with 1/sqrtf, out bf16 bits.
pub fn vNorm(x: []const u16, out: []u16) void {
    var buf: [max_d]f32 = undefined;
    const xs = buf[0..x.len];
    for (xs, x) |*o, b| o.* = bfr.value(b);
    const ms = sumSquares(xs) * (1.0 / @as(f32, @floatFromInt(x.len)));
    const rv = 1.0 / @sqrt(ms + eps32);
    for (out[0..x.len], xs) |*o, xv| o.* = bfr.bits(xv * rv);
}

/// Tau (P8-P10) with logf as fp32(log64(r) * k_log); exactly 1 for pos <= 127999.
pub fn tau(pos: u64, k_log: f64) f32 {
    const p1: f32 = @floatFromInt(pos + 1);
    var r = p1 * inv_floor32;
    if (r < 1) r = 1;
    const lg: f32 = @floatCast(@log(@as(f64, r)) * k_log);
    return 1 + lg * alpha32;
}

fn randBf16(rng: std.Random, sd: f64) u16 {
    return bfr.bits(@floatCast(rng.floatNorm(f64) * sd));
}

fn testHead(rng: std.Random, buf: []u16, n: usize, d: usize, sink: u16) Head {
    for (buf) |*x| x.* = randBf16(rng, 1.5);
    const q = buf[0..d];
    const kv = buf[d..][0 .. 2 * n * d];
    for (buf[d + 2 * n * d ..][0..d_rel]) |*x| x.* = 0; // rel: every bias +0
    return .{ .pos = n - 1, .base = 0, .q = q, .k = kv[0 .. n * d], .v = kv[n * d ..], .rel = buf[d + 2 * n * d ..][0..d_rel], .proj = buf[0 .. d_rel * n], .sink = sink, .gate = q };
}

test "core matches attention_range.head bit for bit without bias, sink and rescales" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(71);
    const rng = prng.random();
    const d = 128;
    var wk = try Work.init(gpa, 64);
    defer wk.deinit(gpa);
    const buf = try gpa.alloc(u16, d + 2 * 64 * d + d_rel);
    defer gpa.free(buf);
    var q64: [d]f64 = undefined;
    const kv64 = try gpa.alloc(f64, 2 * 64 * d);
    defer gpa.free(kv64);
    var pt: [2][d]f64 = undefined;
    var hf: [2][d]f64 = undefined;
    for (0..200) |i| {
        const n = 1 + rng.uintLessThan(usize, 64);
        const h = testHead(rng, buf[0 .. d + 2 * n * d + d_rel], n, d, 0xff80);
        for (&q64, h.q) |*o, x| o.* = val(x);
        for (kv64[0 .. 2 * n * d], buf[d..][0 .. 2 * n * d]) |*o, x| o.* = val(x);
        const inner = switch (i % 3) {
            0 => replayForm(n, d).inner,
            1 => ar.decode(d, n),
            else => ar.prompt(d, n, true),
        };
        try core(h, .{ .window = 0, .extent = @intCast(n) }, .{ .inner = inner }, &wk, &pt[0], &hf[0]);
        ar.head(&q64, kv64[0 .. n * d], kv64[n * d ..], n, d, scale32, inner, wk.w, wk.tau, &pt[1], &hf[1]);
        for (pt[0], pt[1], hf[0], hf[1]) |a, b, c, e| {
            try std.testing.expectEqual(@as(u64, @bitCast(b)), @as(u64, @bitCast(a)));
            try std.testing.expectEqual(@as(u64, @bitCast(e)), @as(u64, @bitCast(c)));
        }
    }
}

fn unfused(rel: []const u16, proj: []const u16) f32 {
    var acc: f32 = 0;
    for (0..d_rel) |i| acc = acc + bfr.value(rel[i]) * bfr.value(proj[i]);
    return acc;
}

fn exact(a: u16, b: u16) bool {
    const p = val(a) * val(b);
    return @as(f64, @as(f32, @floatCast(p))) == p;
}

/// A bf16 with a random significand and sign, at 2^e.
fn atExp(rng: std.Random, e: i32) u16 {
    const v = std.math.ldexp(1 + @as(f64, @floatFromInt(rng.uintLessThan(u32, 128))) / 128, e);
    return bfr.bits(@floatCast(if (rng.boolean()) v else -v));
}

test "bias chains: fused and unfused agree when every bf16 product is exact" {
    var prng = std.Random.DefaultPrng.init(72);
    const rng = prng.random();
    var rel: [d_rel]u16 = undefined;
    var proj: [d_rel]u16 = undefined;
    var near: usize = 0;
    var below: usize = 0;
    for (0..1_000_000) |row| {
        for (&rel, &proj) |*a, *b| {
            if (row % 8 != 0) {
                a.* = atExp(rng, rng.intRangeAtMost(i32, -8, 4));
                b.* = atExp(rng, rng.intRangeAtMost(i32, -8, 4));
                continue;
            }
            // tiny rows: products near 2^-134 (exponent sum >= -135, always exact) and exact ones below it
            const ea = rng.intRangeAtMost(i32, -72, -60);
            const one = rng.boolean();
            a.* = atExp(rng, ea);
            if (one) a.* = bfr.bits(@floatCast(std.math.ldexp(@as(f64, 1), ea)));
            b.* = atExp(rng, (if (one) rng.intRangeAtMost(i32, -142, -136) else rng.intRangeAtMost(i32, -135, -125)) - ea);
        }
        for (rel, proj) |a, b| try std.testing.expect(exact(a, b));
        for (rel, proj) |a, b| {
            const p = @abs(val(a) * val(b));
            if (p < 0x1p-134) below += 1;
            if (p >= 0x1p-135 and p < 0x1p-130) near += 1;
        }
        try std.testing.expectEqual(@as(u32, @bitCast(unfused(&rel, &proj))), @as(u32, @bitCast(chain(&rel, &proj, 1, 0, d_rel))));
    }
    try std.testing.expect(near > 1000 and below > 100);
    // control: an inexact product (lowest bit under 2^-149, so under 2^-134) splits the two chains
    var found = false;
    for (0..100_000) |_| {
        @memset(&rel, 0);
        @memset(&proj, 0);
        rel[0] = atExp(rng, -75);
        proj[0] = atExp(rng, -74);
        rel[1] = atExp(rng, -75);
        proj[1] = atExp(rng, -75);
        if (exact(rel[1], proj[1])) continue;
        if (chain(&rel, &proj, 1, 0, d_rel) != unfused(&rel, &proj)) found = true;
        if (found) break;
    }
    try std.testing.expect(found);
}

test "biasSane accepts the chain and rejects a chain missing its last term" {
    var prng = std.Random.DefaultPrng.init(73);
    const rng = prng.random();
    var rel: [d_rel]u16 = undefined;
    var proj: [d_rel * 3]u16 = undefined;
    for (0..10_000) |_| {
        for (&rel) |*x| x.* = randBf16(rng, 2);
        for (&proj) |*x| x.* = randBf16(rng, 0.25);
        for (0..3) |dist| try biasSane(&rel, &proj, 3, dist);
        try biasSane(&rel, &proj, 3, 3); // past the extent: +0
        try std.testing.expectEqual(@as(f32, 0), bias(&rel, &proj, 3, 7));
    }
    rel[d_rel - 1] = bfr.bits(4);
    proj[(d_rel - 1) * 3 + 1] = bfr.bits(1);
    try std.testing.expectError(error.OracleDrift, drift(chain(&rel, &proj, 3, 1, d_rel - 1), &rel, &proj, 3, 1));
}

test "kxSplitForm: one split is kxForm; each merge adds three roundings to num and den and one rescale" {
    const a = kxForm(0, 1000, 128);
    try std.testing.expectEqualDeep(a, kxSplitForm(0, 1000, 128, 1024));
    try std.testing.expectEqualDeep(kxForm(0, 4095, 128), kxSplitForm(0, 4095, 128, 4096));
    const b = kxSplitForm(0, 4100, 128, 1024);
    const c = kxForm(0, 4100, 128);
    try std.testing.expectEqual(c.inner.num + 12, b.inner.num);
    try std.testing.expectEqual(c.inner.den + 12, b.inner.den);
    try std.testing.expect(b.split == 1 and b.flush and c.flush);
}

test "kx forms: the limb flush term, the split's extra rescale, and subnormal inputs refused by name" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const n = 40;
    const d = 128;
    var buf: [d + 2 * n * d + d_rel]u16 = undefined;
    var proj: [d_rel * 1024]u16 = @splat(0);
    var h = testHead(rng, &buf, n, d, 0xff80);
    h.proj = &proj;
    const g: Geo = .{ .window = 0, .extent = 1024 };
    var wk = try Work.init(std.testing.allocator, n + 1);
    defer wk.deinit(std.testing.allocator);
    var p0: [d]f64 = undefined;
    var h0: [d]f64 = undefined;
    var p1: [d]f64 = undefined;
    var h1: [d]f64 = undefined;
    // the logit form with and without the flush: the same point, the half larger by 3 * 2^-126 * sum |v| / l
    var f: Form = replayForm(n, d);
    _ = try pass(h, g, f, true, &wk, &p0, &h0);
    f.flush = true;
    const r = try pass(h, g, f, true, &wk, &p1, &h1);
    try std.testing.expect(wk.smax > 0);
    for (0..d) |c| {
        var cs: f64 = 0;
        for (0..n) |j| cs += @abs(val(h.v[j * d + c]));
        try std.testing.expectEqual(p0[c], p1[c]);
        try std.testing.expectEqual(h0[c] + 3 * 0x1p-126 * cs / r.l, h1[c]);
    }
    // a split adds one rescale a weight: the half grows
    f.split = 1;
    _ = try pass(h, g, f, true, &wk, &p0, &h0);
    for (0..d) |c| try std.testing.expect(h0[c] >= h1[c]);
    // a subnormal q, K or V element: refused by kx's forms, not by the replay's
    buf[3] = 0x0001;
    try std.testing.expectError(error.SubnormalInput, core(h, g, kxForm(0, n - 1, d), &wk, &p0, &h0));
    try core(h, g, replayForm(n, d), &wk, &p0, &h0);
    buf[3] = 0;
    buf[d + 5 * d + 7] = 0x8001; // key 5's element 7
    try std.testing.expectError(error.SubnormalInput, core(h, g, kxSplitForm(0, n - 1, d, 256), &wk, &p0, &h0));
}
