//! tf-inkling2-attn --oracle --capture, the host half: a real layer's keys and values into the oracle's layout, the
//! exception tests of kx's fast path (inkling2_kx_attention.cuh) predicted from the inputs, and kx's path read from
//! its outputs. No GPU.
//!
//! kx runs a query head over a 16-row block of a forward. The block leaves the fast path for the per-row fallback when
//! (1) a q element of its rows is not finite or is subnormal, (2) a key or value of its loaded 64-key tiles (from the
//! first row's window start rounded down to 64, to its last row) is, or (3) a weight p = expf(s - m) of a tile, m the
//! row's running max over the tiles so far, does not split exactly into three bf16 limbs: p nonzero below 2^-126
//! always fails; below 2^-102 it fails or not by its low bits; at or above 2^-102 it never fails. (1) and (2) are
//! decided from the inputs' bits, (2) on what the tiles' cache slots hold when the forward runs; (3) from float64
//! scores, as a band.
const std = @import("std");
const host = @import("cuda_attn_oracle_host.zig");
const rb = @import("core").relbias_attention_ref;
const bfr = @import("core").bf16_round;

const D = host.D;
const HQ = host.HQ;

/// A capture's [T][HK * D] rows (positions 0 .. T - 1) into `dst` as the oracle's [HK][cap][D].
pub fn transpose(src: []const u16, T: usize, cap: usize, dst: []u16) void {
    for (0..T) |p| for (0..host.HK) |g| {
        @memcpy(dst[(g * cap + p) * D ..][0..D], src[(p * host.HK + g) * D ..][0..D]);
    };
}

/// The engine's rows a forward (cuda_engine.zig `prompt_rows`; cuda_attn_oracle_real.zig asserts they agree). The
/// sliding ring keeps 1023 + prompt_rows slots and a forward's keys and values are all written before its first query
/// runs, so a forward of more rows overwrites keys that its own rows still attend: --capture refuses such forwards.
pub const prompt_rows = 2048;

/// A forward size --chunks takes: 1 to prompt_rows rows.
pub fn chunkOk(n: usize) bool {
    return n > 0 and n <= prompt_rows;
}

/// A bf16 that kx's tests reject: not finite, or subnormal.
pub fn odd(b: u16) bool {
    const e = (b >> 7) & 0xff;
    return e == 0xff or (e == 0 and b & 0x7f != 0);
}

/// Test (3) for one weight: sure (p nonzero below 2^-126 at every probe), maybe (some probe below 2^-102), none.
pub const Band = enum(u2) { none, maybe, sure };

/// x = s - m, its fp32 uncertainty d: p = fp32(exp(x)) at x - d, x, x + d.
pub fn band(x: f64, d: f64) Band {
    var sub: usize = 0;
    var low: usize = 0;
    for ([3]f64{ x - d, x, x + d }) |y| {
        const p: f32 = @floatCast(@exp(y));
        if (p != 0 and p < 0x1p-126) sub += 1 else if (p != 0 and p < 0x1p-102) low += 1;
    }
    return if (sub == 3) .sure else if (sub + low > 0) .maybe else .none;
}

/// kx's tests for one (row, query head) as its own 16-row block would see them; a block takes the OR of its rows
/// (their loaded tiles are the union of the rows' ranges). `xmin`: the lowest s - m over the row's keys.
pub const Pred = struct {
    q_bad: bool = false,
    kv_bad: bool = false,
    limb: Band = .none,
    xmin: f64 = 0,

    pub fn sure(p: Pred) bool {
        return p.q_bad or p.kv_bad or p.limb == .sure;
    }

    pub fn join(a: *Pred, b: Pred) void {
        a.q_bad = a.q_bad or b.q_bad;
        a.kv_bad = a.kv_bad or b.kv_bad;
        a.limb = @fromBackingInt(@intCast(@max(@backingInt(a.limb), @backingInt(b.limb))));
        a.xmin = @min(a.xmin, b.xmin);
    }
};

/// The position whose key and value slot `j % slots` holds once positions 0 .. end - 1 are written in order (j < end):
/// j itself on a slot a position, and on the sliding ring a later position of the same forward for a key that a
/// forward writes over (only keys before every window of the forward, when the forward is at most prompt_rows).
pub fn owner(j: usize, end: usize, slots: usize) usize {
    return j + (end - 1 - j) / slots * slots;
}

/// Row t (position in.p0 + t, keys from in.from <= the row's first tile) and query head h, the prompt (in.p0 ..
/// in.p0 + in.R - 1, in.p0 a multiple of n) run in forwards of n rows on a cache of `slots` slots; `s` holds a score a
/// key. Tests (1) and (2) read what the loaded tiles hold when the row's forward runs: a key or value slot holds its
/// owner's (`owner`, the forward's last position + 1 as `end`); the scores take the window's keys by position.
pub fn kxRow(in: *const host.Inputs, t: usize, h: usize, s: []f64, n: usize, slots: usize) Pred {
    const pos = in.p0 + t;
    const win: u64 = if (in.kind == .sliding) 1023 else 0; // kx's window is the largest distance (Kv.window - 1)
    const first = if (win > 0 and pos > win) pos - win else 0;
    const g = h / host.NREP;
    const i = t * HQ + h;
    var p: Pred = .{};
    const q = in.q[i * D ..][0..D];
    for (q) |x| p.q_bad = p.q_bad or odd(x);
    const end = @min(in.p0 + (t / n + 1) * n, in.p0 + in.R);
    for (first / 64 * 64..pos + 1) |j| {
        const o = owner(j, end, slots);
        for (in.kRow(g, o)) |x| p.kv_bad = p.kv_bad or odd(x);
        for (in.vRow(g, o)) |x| p.kv_bad = p.kv_bad or odd(x);
    }
    const rel = in.rel[i * rb.d_rel ..][0..rb.d_rel];
    for (first..pos + 1) |j| {
        var dot: f64 = 0;
        for (q, in.kRow(g, j)) |a, b| dot += @as(f64, bfr.value(a)) * bfr.value(b);
        const dist = pos - j;
        s[j - first] = dot * rb.scale32 + (if (dist < host.E) rb.bias(rel, in.proj, host.E, dist) else 0);
    }
    var m = -std.math.inf(f64);
    var b = first / 64 * 64;
    while (b <= pos) : (b += 64) {
        const lo = @max(b, first);
        const hi = @min(b + 63, pos);
        for (lo..hi + 1) |j| m = @max(m, s[j - first]);
        for (lo..hi + 1) |j| {
            const x = s[j - first] - m;
            p.xmin = @min(p.xmin, x);
            const w = band(x, 1e-4 * (1 + @abs(s[j - first]) + @abs(m)));
            p.limb = @fromBackingInt(@intCast(@max(@backingInt(p.limb), @backingInt(w))));
        }
    }
    return p;
}

/// kx's path for a 16-row block, read from its outputs.
pub const Seen = enum(u2) { fast, fallback, unseen };

/// Block b (rows 16b .. of a prompt of T rows run in forwards of a multiple of 16 from position 0) and query head h,
/// from kx's run `natural` and two runs whose rows t % 16 == v (v = 0, 1) hold a subnormal q element, which sends
/// every block to the per-row fallback. The fallback gives a row the same bytes whatever its block holds, so: fast
/// when a row that variant leaves alone has other bytes in `natural`; fallback when every such row has the same
/// bytes (or the fast path happens to give them); unseen when no row can be compared.
pub fn seen(natural: []const u16, forced: [2][]const u16, T: usize, b: usize, h: usize) Seen {
    const r0 = 16 * b;
    const r1 = @min(r0 + 16, T);
    var compared: usize = 0;
    for (r0..r1) |t| {
        const v: usize = if (t % 16 == 0) 1 else 0;
        if (r0 + v >= r1) continue; // the variant plants nothing in this block
        const at = (t * HQ + h) * D;
        compared += 1;
        if (!std.mem.eql(u16, natural[at..][0..D], forced[v][at..][0..D])) return .fast;
    }
    return if (compared == 0) .unseen else .fallback;
}

/// The q of a run that sends every block to the fallback: rows t % 16 == v get a subnormal first element per head.
pub fn plant(q: []u16, T: usize, v: usize) void {
    var t = v;
    while (t < T) : (t += 16) for (0..HQ) |h| {
        q[(t * HQ + h) * D] = 0x0001;
    };
}

test "test (3)'s band: subnormal weights always fail, small normal ones may, the rest and zero never" {
    try std.testing.expectEqual(Band.none, band(0, 1e-4));
    try std.testing.expectEqual(Band.none, band(-60, 1e-4));
    try std.testing.expectEqual(Band.maybe, band(-75, 1e-4)); // 2^-108: normal, low bits below 2^-126
    try std.testing.expectEqual(Band.sure, band(-95, 1e-4)); // 2^-137: an fp32 subnormal
    try std.testing.expectEqual(Band.maybe, band(-103.97, 1e-2)); // the last subnormal or zero
    try std.testing.expectEqual(Band.none, band(-200, 1e-4)); // p = 0: weight 0, no limb
    try std.testing.expect(odd(0x0001) and odd(0x8001) and odd(0x7f80) and odd(0xff80) and odd(0x7fc0));
    try std.testing.expect(!odd(0) and !odd(0x8000) and !odd(0x0080) and !odd(0x3f80));
}

fn tiny(gpa: std.mem.Allocator, kind: host.Kind, n: usize) !host.Inputs {
    var in = try host.Inputs.alloc(gpa, n, n);
    in.kind = kind;
    in.profile = .clip;
    in.p0 = 0;
    in.R = n;
    in.from = 0;
    in.npos = n;
    for ([_][]u16{ in.k, in.v, in.q, in.rel, in.gate, in.sink, in.proj }) |x| @memset(x, 0);
    return in;
}

test "kxRow: q, a key or value of a loaded tile, and a score 90 under the max each send the row to the fallback" {
    const gpa = std.testing.allocator;
    var in = try tiny(gpa, .global, 4);
    defer in.free(gpa);
    var s: [8]f64 = undefined;
    try std.testing.expect(!kxRow(&in, 3, 7, &s, in.R, in.R).sure());
    in.q[(3 * HQ + 7) * D] = 0x4200; // 32
    in.kRow(1, 2)[0] = 0x4200; // q.k = 1024 at key 2: score 90.5, the other keys 0
    const p = kxRow(&in, 3, 7, &s, in.R, in.R);
    try std.testing.expectEqual(Band.sure, p.limb);
    try std.testing.expect(p.xmin < -90 and !p.q_bad and !p.kv_bad);
    try std.testing.expectEqual(Band.none, kxRow(&in, 2, 7, &s, in.R, in.R).limb); // row 2's q is 0: every score 0
    in.kRow(1, 2)[0] = 0;
    in.q[(3 * HQ + 7) * D + 5] = 0x0001;
    try std.testing.expect(kxRow(&in, 3, 7, &s, in.R, in.R).q_bad and !kxRow(&in, 3, 6, &s, in.R, in.R).q_bad and !kxRow(&in, 3, 0, &s, in.R, in.R).q_bad);
    in.q[(3 * HQ + 7) * D + 5] = 0;
    in.vRow(1, 0)[9] = 0x7f80;
    try std.testing.expect(kxRow(&in, 3, 7, &s, in.R, in.R).kv_bad and kxRow(&in, 0, 6, &s, in.R, in.R).kv_bad and !kxRow(&in, 3, 0, &s, in.R, in.R).kv_bad);
}

test "kxRow: a sliding row's loaded tiles start at its window start rounded down to 64, before the window" {
    const gpa = std.testing.allocator;
    var in = try tiny(gpa, .sliding, 1101);
    defer in.free(gpa);
    const s = try gpa.alloc(f64, 1101);
    defer gpa.free(s);
    in.kRow(0, 70)[3] = 0x7f80; // row 1100's window starts at 77, its first tile at 64
    try std.testing.expect(kxRow(&in, 1100, 0, s, in.R, in.R).kv_bad);
    try std.testing.expectEqual(@as(u64, 77), rb.keyLo(in.geo(), 1100)); // the reference's window leaves key 70 out
    try std.testing.expect(!kxRow(&in, 1100, 6, s, in.R, in.R).kv_bad); // kv head 1
    in.kRow(0, 70)[3] = 0;
    in.kRow(0, 63)[3] = 0x7f80; // before the first tile
    try std.testing.expect(!kxRow(&in, 1100, 0, s, in.R, in.R).kv_bad);
}

test "seen: a row that differs from the forced run means the fast path; a block of one row cannot be read" {
    const gpa = std.testing.allocator;
    const T = 17;
    const n = T * HQ * D;
    const nat = try gpa.alloc(u16, n);
    defer gpa.free(nat);
    const f0 = try gpa.alloc(u16, n);
    defer gpa.free(f0);
    const f1 = try gpa.alloc(u16, n);
    defer gpa.free(f1);
    @memset(nat, 0x3f80);
    @memset(f0, 0x3f80);
    @memset(f1, 0x3f80);
    try std.testing.expectEqual(Seen.fallback, seen(nat, .{ f0, f1 }, T, 0, 3));
    f0[(5 * HQ + 3) * D + 100] = 0x3f81; // row 5 is compared with variant 0
    try std.testing.expectEqual(Seen.fast, seen(nat, .{ f0, f1 }, T, 0, 3));
    try std.testing.expectEqual(Seen.fallback, seen(nat, .{ f0, f1 }, T, 0, 2));
    f1[(5 * HQ + 3) * D + 100] = 0x3f81; // variant 1 is not read for row 5
    f0[(5 * HQ + 3) * D + 100] = 0x3f80;
    try std.testing.expectEqual(Seen.fallback, seen(nat, .{ f0, f1 }, T, 0, 3));
    f1[(0 * HQ + 3) * D] = 0x3f81; // row 0 is read through variant 1
    try std.testing.expectEqual(Seen.fast, seen(nat, .{ f0, f1 }, T, 0, 3));
    try std.testing.expectEqual(Seen.unseen, seen(nat, .{ f0, f1 }, T, 1, 3)); // rows 16 .. 16: variant 1 plants row 17
    plant(f0, T, 1);
    try std.testing.expectEqual(@as(u16, 0x0001), f0[(1 * HQ + 47) * D]);
    try std.testing.expectEqual(@as(u16, 0x3f80), f0[(2 * HQ + 47) * D]);
}

/// Whether every row of a T-row prompt run in forwards of n rows from position 0 finds each key of its sliding window
/// (distances 0 .. 1023) in its slot of a ring of `slots`, a forward's keys written before any of its rows runs.
fn ringKeeps(gpa: std.mem.Allocator, T: usize, n: usize, slots: usize) !bool {
    const held = try gpa.alloc(usize, slots);
    defer gpa.free(held);
    @memset(held, std.math.maxInt(usize));
    var p0: usize = 0;
    while (p0 < T) : (p0 += n) {
        const end = @min(p0 + n, T);
        for (p0..end) |p| held[p % slots] = p;
        for (p0..end) |pos| for ((if (pos > 1023) pos - 1023 else 0)..pos + 1) |j| {
            if (held[j % slots] != j) return false;
        };
    }
    return true;
}

test "--chunks: a forward the sliding ring cannot hold whole is refused" {
    const gpa = std.testing.allocator;
    const slots = 1023 + prompt_rows;
    try std.testing.expect(!chunkOk(0));
    for ([_]usize{ 1, 64, 2047, 2048, 2049, 3071, 4096 }) |n| {
        // two whole forwards of every n: the second forward of 2049 writes position 4097 over key 1026, which its
        // first row (2049) attends; a forward of 4096 writes position 3071 over key 0 before row 0 runs
        try std.testing.expectEqual(try ringKeeps(gpa, 2 * slots, n, slots), chunkOk(n));
    }
}

test "kxRow: a loaded slot the forward overwrote is tested as it is held, not by its position" {
    const gpa = std.testing.allocator;
    const T = 4096;
    const slots = 1023 + prompt_rows;
    var in = try tiny(gpa, .sliding, T);
    defer in.free(gpa);
    const s = try gpa.alloc(f64, 1100);
    defer gpa.free(s);
    in.vRow(0, 4095)[5] = 0x0001; // a subnormal value at position 4095, in slot 4095 % 3071 = 1024
    // forwards of 2048: the second writes 2048 .. 4095, then block 128 (rows 2048 .. 2063, first key 1025) loads the
    // tile from 1024, whose slot holds position 4095
    var p: Pred = .{};
    for (2048..2064) |t| p.join(kxRow(&in, t, 0, s, 2048, slots));
    try std.testing.expect(p.kv_bad and kxRow(&in, 2048, 0, s, 2048, slots).kv_bad);
    try std.testing.expect(!kxRow(&in, 2048, 6, s, 2048, slots).kv_bad); // kv head 1
    try std.testing.expect(!kxRow(&in, 2047, 0, s, 2048, slots).kv_bad); // the first forward: slot 1024 holds 1024
    try std.testing.expect(!kxRow(&in, 2048, 0, s, 64, slots).kv_bad); // forwards of 64: 4095 not yet written
    try std.testing.expect(kxRow(&in, 4095, 0, s, 64, slots).kv_bad); // its own row
    try std.testing.expect(!kxRow(&in, 2048, 0, s, 2048, T).kv_bad); // a slot a position
    try std.testing.expectEqual(@as(usize, 4095), owner(1024, 4096, slots));
}

test "owner: the position a slot holds, against a ring written forward by forward" {
    const gpa = std.testing.allocator;
    const slots = 1023 + prompt_rows;
    const held = try gpa.alloc(usize, slots);
    defer gpa.free(held);
    for ([_]usize{ 1, 64, 2048 }) |n| {
        @memset(held, std.math.maxInt(usize));
        var end: usize = 0;
        while (end < 7000) {
            const next = @min(end + n, 7000);
            for (end..next) |q| held[q % slots] = q;
            end = next;
            var j = end -| 3 * slots;
            while (j < end) : (j += 7) try std.testing.expectEqual(held[j % slots], owner(j, end, slots));
        }
    }
}
