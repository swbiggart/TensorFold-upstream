//! E4M3 codes with one UE8M0 byte per 32 inputs, repacked losslessly for the lane matmul.
const std = @import("std");
const fp8 = @import("fp8.zig");
const qmmf = @import("qmmf.zig");

pub const group = 32;
pub const padded = fp8.padded;
pub const packCodes = fp8.packCodes;

/// Row-major [n, k/32] bytes to [npad/64][k/64][64][2]; padded rows have scale 1.
pub fn packScales(out: []u8, scales: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / qmmf.group;
    std.debug.assert(k % qmmf.group == 0 and out.len == npad * (k / group) and scales.len == n * (k / group));
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |r| for (0..2) |h| {
        const row = t * 64 + r;
        out[((t * kg + g) * 64 + r) * 2 + h] = if (row < n) scales[row * (k / group) + g * 2 + h] else 127;
    };
}

pub fn weight(codes: u64, scales: u64, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .mxfp8, .codes = codes, .scales = scales, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

test "every scale byte lands in its independent 32-input group, including padded rows" {
    const a = std.testing.allocator;
    const n = 130;
    const k = 256;
    const scales = try a.alloc(u8, n * (k / group));
    defer a.free(scales);
    for (scales, 0..) |*v, i| v.* = @truncate(i * 13 + 7);
    const out = try a.alloc(u8, padded(n) * (k / group));
    defer a.free(out);
    packScales(out, scales, n, k);
    for (0..padded(n)) |row| for (0..k / group) |g| {
        const at = (((row / 64) * (k / 64) + g / 2) * 64 + row % 64) * 2 + g % 2;
        try std.testing.expectEqual(if (row < n) scales[row * (k / group) + g] else @as(u8, 127), out[at]);
    };
    const w = weight(1, 2, n, k);
    try std.testing.expect(w.mode == .mxfp8 and w.npad == 256 and w.scale == 1.0);
}

/// The value of e4m3 code `c` (sign bit 7, exponent bits 6-3 biased 7, mantissa bits 2-0; 0x7F and 0xFF NaN).
pub fn e4m3Value(c: u8) f32 {
    if (c & 0x7F == 0x7F) return std.math.nan(f32);
    const e: i32 = (c >> 3) & 15;
    const m: f32 = @floatFromInt(c & 7);
    const v: f32 = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(8.0 + m, e - 10);
    return if (c & 0x80 != 0) -v else v;
}

/// The finite positive codes' values, increasing with the code (0 .. 0x7E).
const e4m3_values: [0x7F]f32 = blk: {
    @setEvalBranchQuota(10000);
    var t: [0x7F]f32 = undefined;
    for (&t, 0..) |*v, c| v.* = e4m3Value(c);
    break :blk t;
};

/// x's nearest e4m3 code (ties to the even code), saturating at +-448; x finite.
pub fn e4m3Round(x: f32) u8 {
    const a = @abs(x);
    const sign: u8 = if (x < 0) 0x80 else 0;
    if (!(a < 448.0)) return sign | 0x7E;
    // the last code whose value is <= a
    var lo: u8 = 0;
    var hi: u8 = 0x7E;
    while (lo < hi) {
        const mid: u8 = lo + (hi - lo + 1) / 2;
        if (e4m3_values[mid] <= a) lo = mid else hi = mid - 1;
    }
    const below = e4m3_values[lo];
    const above = e4m3_values[lo + 1];
    const c: u8 = if (a - below < above - a) lo else if (a - below > above - a) lo + 1 else if (lo & 1 == 0) lo else lo + 1;
    return sign | c;
}

/// bf16 [n, k] (little-endian bytes) to e4m3 codes [n, k] and one UE8M0 scale byte a 32 inputs [n, k/32] (row-major,
/// packScales' input): the scale is the smallest power of two 2^s with amax / 2^s <= 448 (no input saturates), byte
/// s + 127 (127: an all-zero group); each code the nearest e4m3 of x / 2^s. Weight-only, for a drafter's projections.
/// A non-finite input (bf16 infinity or NaN) is refused. Groups are independent: worker threads take contiguous
/// ranges of them, so every byte equals one serial pass's.
pub fn quantize(codes: []u8, scales: []u8, w: []const u8, n: usize, k: usize) Error!void {
    return quantizeOn(std.Thread.getCpuCount() catch 1, codes, scales, w, n, k);
}

pub const Error = error{NonFiniteMxfp8Input};
const max_workers = 16;

fn quantizeOn(cpus: usize, codes: []u8, scales: []u8, w: []const u8, n: usize, k: usize) Error!void {
    std.debug.assert(k % group == 0 and w.len == n * k * 2 and codes.len == n * k and scales.len == n * (k / group));
    const ng = scales.len;
    const workers = @min(max_workers, @max(1, cpus), @max(1, ng / 4096));
    const per = (ng + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |t| threads[t] = std.Thread.spawn(.{}, run, .{ &results[t], codes, scales, w, @min(ng, t * per), @min(ng, (t + 1) * per) }) catch null;
    run(&results[0], codes, scales, w, 0, @min(ng, per));
    for (1..workers) |t| if (threads[t]) |th| th.join() else run(&results[t], codes, scales, w, @min(ng, t * per), @min(ng, (t + 1) * per));
    for (results[0..workers]) |r| try r;
}

fn run(out: *Error!void, codes: []u8, scales: []u8, w: []const u8, first: usize, last: usize) void {
    out.* = groups(codes, scales, w, first, last);
}

/// Groups [first, last) of the row-major [n, k/32] scale order; group i holds inputs [32 i, 32 i + 32).
fn groups(codes: []u8, scales: []u8, w: []const u8, first: usize, last: usize) Error!void {
    for (first..last) |gi| {
        var amax: f32 = 0;
        for (0..group) |i| {
            const bits = std.mem.readInt(u16, w[2 * (gi * group + i) ..][0..2], .little);
            if (bits & 0x7F80 == 0x7F80) return error.NonFiniteMxfp8Input;
            amax = @max(amax, @abs(@as(f32, @bitCast(@as(u32, bits) << 16))));
        }
        var s: i32 = 0;
        if (amax > 0) {
            s = @as(i32, @intFromFloat(@ceil(std.math.log2(amax / 448.0))));
            while (s < 127 and std.math.ldexp(amax, -s) > 448.0) s += 1;
            while (s > -127 and std.math.ldexp(amax, -(s - 1)) <= 448.0) s -= 1;
        }
        s = std.math.clamp(s, -127, 127);
        scales[gi] = @intCast(s + 127);
        const inv = std.math.ldexp(@as(f32, 1.0), -s);
        for (0..group) |i| {
            const at = gi * group + i;
            const bits = std.mem.readInt(u16, w[2 * at ..][0..2], .little);
            codes[at] = e4m3Round(@as(f32, @bitCast(@as(u32, bits) << 16)) * inv);
        }
    }
}

test "e4m3 round: every finite code is its own value's code, halfway ties go to the even code, saturation at 448" {
    for (0..0x7F) |c| {
        const v = e4m3Value(@intCast(c));
        try std.testing.expectEqual(@as(u8, @intCast(c)), e4m3Round(v));
        try std.testing.expectEqual(@as(u8, @intCast(c)) | 0x80, e4m3Round(-v) | if (c == 0) @as(u8, 0x80) else 0);
    }
    try std.testing.expectEqual(@as(f32, 448), e4m3Value(0x7E));
    try std.testing.expectEqual(@as(f32, -448), e4m3Value(0xFE));
    try std.testing.expect(std.math.isNan(e4m3Value(0x7F)) and std.math.isNan(e4m3Value(0xFF)));
    try std.testing.expectEqual(@as(u8, 0x7E), e4m3Round(1000));
    // 1.0 is 0x38, 1.125 is 0x39: 1.0625 ties to the even 0x38; 1.1875 (between 0x39 and 0x3A) to 0x3A
    try std.testing.expectEqual(@as(u8, 0x38), e4m3Round(1.0625));
    try std.testing.expectEqual(@as(u8, 0x3A), e4m3Round(1.1875));
}

test "quantize: scales cover each group's largest input, codes within half an e4m3 step, zeros stay zero" {
    const a = std.testing.allocator;
    const n = 3;
    const k = 64;
    const w = try a.alloc(u8, n * k * 2);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(5);
    for (0..n * k) |i| {
        const x: f32 = if (i / k == 2 and i % k < 32) 0 else (prng.random().float(f32) - 0.5) * 0.08;
        std.mem.writeInt(u16, w[2 * i ..][0..2], @truncate(@as(u32, @bitCast(x)) >> 16), .little);
    }
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    const scales = try a.alloc(u8, n * (k / group));
    defer a.free(scales);
    try quantize(codes, scales, w, n, k);
    try std.testing.expectEqual(@as(u8, 127), scales[2 * 2]);
    for (0..n) |r| for (0..k / group) |g| {
        const sc = std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, scales[r * 2 + g]) - 127)));
        var amax: f32 = 0;
        for (0..group) |i| {
            const at = r * k + g * group + i;
            const x: f32 = @bitCast(@as(u32, std.mem.readInt(u16, w[2 * at ..][0..2], .little)) << 16);
            amax = @max(amax, @abs(x));
            const back = e4m3Value(codes[at]) * sc;
            // relative step of e4m3 is 2^-3; half of it, or the subnormal step's half
            try std.testing.expect(@abs(back - x) <= @max(@abs(x) * 0.0625, sc * std.math.pow(f32, 2, -10)) * 1.0001);
        }
        if (amax > 0) {
            try std.testing.expect(amax / sc <= 448 and amax / sc > 224);
        }
    };
}

test "quantize: worker threads give one serial pass's bytes" {
    const a = std.testing.allocator;
    const n = 1000;
    const k = 512;
    const w = try a.alloc(u8, n * k * 2);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(11);
    for (0..n * k) |i| {
        const x: f32 = (prng.random().float(f32) - 0.5) * std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast(i / 4096 % 40)) - 30);
        std.mem.writeInt(u16, w[2 * i ..][0..2], @truncate(@as(u32, @bitCast(x)) >> 16), .little);
    }
    const c1 = try a.alloc(u8, n * k);
    defer a.free(c1);
    const s1 = try a.alloc(u8, n * (k / group));
    defer a.free(s1);
    try quantizeOn(1, c1, s1, w, n, k);
    for ([_]usize{ 2, 3, 7, 16 }) |cpus| {
        const c2 = try a.alloc(u8, n * k);
        defer a.free(c2);
        const s2 = try a.alloc(u8, n * (k / group));
        defer a.free(s2);
        try quantizeOn(cpus, c2, s2, w, n, k);
        try std.testing.expectEqualSlices(u8, c1, c2);
        try std.testing.expectEqualSlices(u8, s1, s2);
    }
}

test "quantize: infinities and NaNs of either sign are refused, in any worker's range" {
    const a = std.testing.allocator;
    const n = 512;
    const k = 512;
    const w = try a.alloc(u8, n * k * 2);
    defer a.free(w);
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    const scales = try a.alloc(u8, n * (k / group));
    defer a.free(scales);
    for ([_]u16{ 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81 }) |bad| for ([_]usize{ 0, n * k / 2 + 5, n * k - 1 }) |at| {
        @memset(w, 0);
        std.mem.writeInt(u16, w[2 * at ..][0..2], bad, .little);
        for ([_]usize{ 1, 7 }) |cpus| try std.testing.expectError(error.NonFiniteMxfp8Input, quantizeOn(cpus, codes, scales, w, n, k));
    };
    // 8,192 groups: two workers of 4,096 groups, so worker one is a spawned thread; its range holds the only
    // invalid weight, and its error reaches the caller
    const ng = n * (k / group);
    try std.testing.expect(@min(max_workers, ng / 4096) == 2);
    for ([_]usize{ ng / 2, ng / 2 + 1234, ng - 1 }) |gi| {
        @memset(w, 0);
        std.mem.writeInt(u16, w[2 * (gi * group + 17) ..][0..2], 0xFFC0, .little);
        for ([_]usize{ 2, 16 }) |cpus| try std.testing.expectError(error.NonFiniteMxfp8Input, quantizeOn(cpus, codes, scales, w, n, k));
    }
    // the largest finite bf16, (2 - 2^-7) 2^127, is accepted: scale 2^120, code 256 (255 rounded)
    @memset(w, 0);
    std.mem.writeInt(u16, w[0..2], 0x7F7F, .little);
    try quantize(codes, scales, w, n, k);
    try std.testing.expectEqual(@as(u8, 127 + 120), scales[0]);
    try std.testing.expectEqual(@as(u8, 0x78), codes[0]);
}
