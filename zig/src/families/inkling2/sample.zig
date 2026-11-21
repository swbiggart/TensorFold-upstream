//! Rank 0's sampled draw over a row of fp32 logits (cuda_lanes.zig): the lane core's keyed fp64 sampler
//! (lanes.sampling.choose: the temperature floored at 1e-6, top_k, top_p over the top k, min_p, Gumbel keys at the
//! row's position) over the whole vocabulary, without sorting the whole row. A draw only ever reads a prefix of the
//! row's order (value descending, then the lower id), and only that prefix is sorted:
//! - top_k below the vocabulary: the values at or above a cutoff that at least k of them reach. chooseOrdered reads
//!   the first k of an order alone, so the draw is choose's own.
//! - top_p alone: the values at or above a cutoff whose mass reaches top_p of the row's total (fp64, every token).
//!   The nucleus walk runs over that prefix against the row's total; a walk that does not end inside the prefix
//!   takes the whole sorted row instead.
//! - neither: choose's one pass over the row.
//! Temperature 0 never comes here: greedy is the caller's argmax.
//! A row may hold -inf (a grammar's disallowed tokens, cuda_lanes.zig drawMasked): such a value has no mass and sorts
//! after every finite one, so the paths above draw what choose draws with those tokens in the row.
const std = @import("std");
const lanes = @import("lanes");

const sampling = lanes.sampling;
const Sampling = lanes.Sampling;
const Candidate = sampling.Candidate;

extern "c" fn exp(x: f64) f64;

/// Histogram buckets over a value's distance below the row's best: `per` a unit, the last one holds the rest.
const buckets = 256;
const per = 8.0;

pub const Path = enum { top_k, top_p, race, full };

pub const Sampler = struct {
    gpa: std.mem.Allocator,
    order: std.ArrayList(Candidate) = .empty,
    values: std.ArrayList(f64) = .empty,
    ids: std.ArrayList(u64) = .empty,
    /// The last draw's path (tests).
    path: Path = .race,

    pub fn deinit(self: *Sampler) void {
        self.order.deinit(self.gpa);
        self.values.deinit(self.gpa);
        self.ids.deinit(self.gpa);
    }

    /// The token `s` draws from `logits` at `position` (s.temperature > 0).
    pub fn draw(self: *Sampler, logits: []const f32, s: Sampling, position: u64) !u32 {
        const V = logits.len;
        if (V == 0) return error.NoAllowedToken;
        var m = -std.math.inf(f32);
        var finite = true;
        for (logits) |l| {
            finite = finite and (std.math.isFinite(l) or l == -std.math.inf(f32));
            m = @max(m, l);
        }
        // NaN, +inf, a row of -inf alone, and temperature 0 keep choose's own handling
        if (!finite or !std.math.isFinite(m) or !(s.temperature > 0)) return self.whole(logits, s, position, .full);
        const k: usize = if (s.top_k == 0) V else @min(V, s.top_k);
        if (k < V) return self.topK(logits, m, k, s, position);
        if (s.top_p > 0.0 and s.top_p < 1.0) return self.topP(logits, m, s, position);
        return self.whole(logits, s, position, .race);
    }

    /// choose over the whole row (its one-pass race when no cut applies, else its full sort).
    fn whole(self: *Sampler, logits: []const f32, s: Sampling, position: u64, path: Path) !u32 {
        self.path = path;
        try self.values.resize(self.gpa, logits.len);
        try self.ids.resize(self.gpa, logits.len);
        for (self.values.items, self.ids.items, logits, 0..) |*v, *id, l, i| {
            v.* = l;
            id.* = i;
        }
        return @intCast(try sampling.choose(self.gpa, self.values.items, self.ids.items, position, s));
    }

    /// The bucket of logit `l` at `top - l / t` below the best (`per` buckets a unit, the last one for the rest).
    /// It only grows as l falls: buckets 0 ..= b hold every value at or above their lowest one, ties included.
    fn bucket(top: f64, l: f32, t: f64) usize {
        const at = (top - @as(f64, l) / t) * per;
        return if (at >= buckets - 1) buckets - 1 else @intFromFloat(at);
    }

    /// The row's values in buckets 0 ..= last, sorted: the sorted row's first entries.
    fn prefix(self: *Sampler, logits: []const f32, top: f64, t: f64, last: usize) ![]const Candidate {
        self.order.clearRetainingCapacity();
        for (logits, 0..) |l, i| {
            if (bucket(top, l, t) <= last) try self.order.append(self.gpa, .{ .value = l, .id = i });
        }
        std.mem.sort(Candidate, self.order.items, {}, byValueThenId);
        return self.order.items;
    }

    /// top_k below the vocabulary: a prefix of at least k values, then chooseOrdered (it reads the first k).
    fn topK(self: *Sampler, logits: []const f32, m: f32, k: usize, s: Sampling, position: u64) !u32 {
        var count: [buckets]usize = @splat(0);
        const top: f64 = m;
        for (logits) |l| count[bucket(top, l, 1.0)] += 1;
        var last: usize = 0;
        var n: usize = count[0];
        while (n < k) {
            last += 1;
            n += count[last];
        }
        if (last == buckets - 1) return self.whole(logits, s, position, .full);
        self.path = .top_k;
        const order = try self.prefix(logits, top, 1.0, last);
        return @intCast(try sampling.chooseOrdered(self.gpa, order, position, s));
    }

    /// top_p over the whole row: the row's total in fp64, a prefix that holds top_p of it (one bucket more than the
    /// histogram asks, for its rounding), the nucleus walked over it (as chooseOrdered walks), then chooseOrdered
    /// over the nucleus alone (min_p and the keyed draw).
    fn topP(self: *Sampler, logits: []const f32, m: f32, s: Sampling, position: u64) !u32 {
        const t = @max(s.temperature, 1e-6);
        const x0 = @as(f64, m) / t;
        var mass: [buckets]f64 = @splat(0.0);
        var total: f64 = 0.0;
        for (logits) |l| {
            const p = exp(@as(f64, l) / t - x0);
            mass[bucket(x0, l, t)] += p;
            total += p;
        }
        const want = s.top_p * total;
        var last: usize = 0;
        var held: f64 = mass[0];
        while (held < want and last < buckets - 1) {
            last += 1;
            held += mass[last];
        }
        last = @min(last + 1, buckets - 1);
        if (last == buckets - 1) return self.whole(logits, s, position, .full);
        const order = try self.prefix(logits, x0, t, last);
        // chooseOrdered's walk: keep the candidates whose running share stays below top_p, and the one that reaches it
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (order) |c| {
            cum += exp(c.value / t - x0) / total;
            if (cum < s.top_p) below += 1 else break;
        }
        if (below == order.len) return self.whole(logits, s, position, .full); // the walk did not end inside the prefix
        self.path = .top_p;
        var nucleus = s;
        nucleus.top_k = @intCast(below + 1);
        nucleus.top_p = 1.0;
        return @intCast(try sampling.chooseOrdered(self.gpa, order[0 .. below + 1], position, nucleus));
    }
};

fn byValueThenId(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

// Host tests: draws against the lane core's choose over the whole row, and the drawn frequencies against the exact
// distribution on small synthetic rows.

const testing = std.testing;

/// choose over the whole row, as the reference.
fn reference(a: std.mem.Allocator, logits: []const f32, s: Sampling, position: u64) !u32 {
    const values = try a.alloc(f64, logits.len);
    defer a.free(values);
    const ids = try a.alloc(u64, logits.len);
    defer a.free(ids);
    for (values, ids, logits, 0..) |*v, *id, l, i| {
        v.* = l;
        id.* = i;
    }
    return @intCast(try sampling.choose(a, values, ids, position, s));
}

/// The exact probabilities of `s` over `logits`: softmax at max(T, 1e-6), the top k (ties to the lower id), the
/// nucleus over the top k's mass, min_p against the best.
fn exact(a: std.mem.Allocator, logits: []const f32, s: Sampling) ![]f64 {
    const V = logits.len;
    const order = try a.alloc(Candidate, V);
    defer a.free(order);
    for (order, logits, 0..) |*c, l, i| c.* = .{ .value = l, .id = i };
    std.mem.sort(Candidate, order, {}, byValueThenId);
    const t = @max(s.temperature, 1e-6);
    const k = if (s.top_k == 0) V else @min(V, s.top_k);
    var z: f64 = 0.0;
    for (order[0..k]) |c| z += @exp((c.value - order[0].value) / t);
    var keep = k;
    if (s.top_p > 0.0 and s.top_p < 1.0) {
        var cum: f64 = 0.0;
        for (order[0..k], 0..) |c, j| {
            cum += @exp((c.value - order[0].value) / t) / z;
            if (cum >= s.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (s.min_p > 0.0) {
        var n: usize = 0;
        while (n < keep and @exp((order[n].value - order[0].value) / t) >= s.min_p) n += 1;
        keep = n;
    }
    const p = try a.alloc(f64, V);
    @memset(p, 0.0);
    var kept: f64 = 0.0;
    for (order[0..keep]) |c| kept += @exp((c.value - order[0].value) / t);
    for (order[0..keep]) |c| p[c.id] = @exp((c.value - order[0].value) / t) / kept;
    return p;
}

const settings = [_]Sampling{
    .{ .seed = 7, .temperature = 0.7, .top_k = 0, .top_p = 1.0 },
    .{ .seed = 8, .temperature = 1.0, .top_k = 5, .top_p = 1.0 },
    .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.8 },
    .{ .seed = 10, .temperature = 1.3, .top_k = 6, .top_p = 0.9 },
    .{ .seed = 11, .temperature = 0.5, .top_k = 0, .top_p = 0.95, .min_p = 0.05 },
    .{ .seed = 12, .temperature = 2.0, .top_k = 0, .top_p = 1.0, .min_p = 0.1 },
};

test "inkling2 sampler: every path draws what choose draws over the whole row" {
    const a = testing.allocator;
    var rng = std.Random.DefaultPrng.init(42);
    const r = rng.random();
    const logits = try a.alloc(f32, 5000);
    defer a.free(logits);
    for (logits) |*l| l.* = @floatCast(r.floatNorm(f64) * 3.0);
    logits[1234] = logits[77]; // a tie
    var smp: Sampler = .{ .gpa = a };
    defer smp.deinit();
    const more = [_]Sampling{
        .{ .seed = 1, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 2, .temperature = 1e-9, .top_k = 0, .top_p = 0.9 }, // under the floor
        .{ .seed = 3, .temperature = 1.0, .top_k = 2000, .top_p = 0.99 },
        .{ .seed = 4, .temperature = 1.0, .top_k = 0, .top_p = 0.999999 },
    };
    var seen: [4]usize = @splat(0);
    for (settings ++ more) |s| {
        for (0..60) |position| {
            const got = try smp.draw(logits, s, position);
            try testing.expectEqual(try reference(a, logits, s, position), got);
            seen[@intFromEnum(smp.path)] += 1;
        }
    }
    try testing.expect(seen[@intFromEnum(Path.top_k)] > 0 and seen[@intFromEnum(Path.top_p)] > 0 and seen[@intFromEnum(Path.race)] > 0);
}

test "inkling2 sampler: a masked row (disallowed tokens at -inf) draws what choose draws, never a disallowed token" {
    const a = testing.allocator;
    var rng = std.Random.DefaultPrng.init(43);
    const r = rng.random();
    const logits = try a.alloc(f32, 5000);
    defer a.free(logits);
    var smp: Sampler = .{ .gpa = a };
    defer smp.deinit();
    const more = [_]Sampling{
        .{ .seed = 1, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 3, .temperature = 1.0, .top_k = 2000, .top_p = 0.99 },
        .{ .seed = 4, .temperature = 1.0, .top_k = 0, .top_p = 0.999999 },
        .{ .seed = 5, .temperature = 1.0, .top_k = 4000, .top_p = 1.0 }, // past the allowed tokens: the -inf ones too
    };
    var seen: [4]usize = @splat(0);
    // most allowed (a JSON string's body), a few (a key's choice), one (a forced token)
    for ([_]f64{ 0.9, 0.01, 0.0 }) |keep| {
        for (logits) |*l| l.* = if (r.float(f64) < keep) @floatCast(r.floatNorm(f64) * 3.0) else -std.math.inf(f32);
        logits[4321] = 1.5; // at least one allowed
        for (settings ++ more) |s| {
            for (0..40) |position| {
                const got = try smp.draw(logits, s, position);
                try testing.expect(logits[got] != -std.math.inf(f32));
                try testing.expectEqual(try reference(a, logits, s, position), got);
                seen[@intFromEnum(smp.path)] += 1;
            }
        }
    }
    try testing.expect(seen[@intFromEnum(Path.top_k)] > 0 and seen[@intFromEnum(Path.top_p)] > 0 and seen[@intFromEnum(Path.race)] > 0);
}

test "inkling2 sampler: the drawn frequencies follow the exact distribution" {
    const a = testing.allocator;
    const logits = [_]f32{ 2.0, 1.5, 1.5, 0.3, -0.2, 0.9, -1.7, 3.1, 0.0, -4.0, 1.1, 2.6 };
    var smp: Sampler = .{ .gpa = a };
    defer smp.deinit();
    const n = 40_000;
    for (settings) |s| {
        const p = try exact(a, &logits, s);
        defer a.free(p);
        var counts: [logits.len]usize = @splat(0);
        for (0..n) |position| counts[try smp.draw(&logits, s, position)] += 1;
        // chi-square over the kept tokens; a dropped token is never drawn
        var chi: f64 = 0.0;
        var df: usize = 0;
        for (p, counts) |q, c| {
            if (q == 0.0) {
                try testing.expectEqual(@as(usize, 0), c);
                continue;
            }
            const e = q * n;
            chi += (@as(f64, @floatFromInt(c)) - e) * (@as(f64, @floatFromInt(c)) - e) / e;
            df += 1;
        }
        // at most 11 degrees of freedom: chi-square past 45 has a chance below 1e-5
        if (chi > 45.0) {
            std.debug.print("seed {d}: chi-square {d:.1} over {d} tokens\n", .{ s.seed, chi, df });
            return error.DistributionDiffers;
        }
    }
}

test "inkling2 sampler: no 1024-candidate cap: a flat row and a nucleus past 1024 tokens are drawn whole" {
    const a = testing.allocator;
    const logits = try a.alloc(f32, 2048);
    defer a.free(logits);
    @memset(logits, 0.5);
    var smp: Sampler = .{ .gpa = a };
    defer smp.deinit();
    // the whole row: half the draws land past id 1023
    var past: usize = 0;
    const n = 4000;
    for (0..n) |position| {
        if (try smp.draw(logits, .{ .seed = 5, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, position) >= 1024) past += 1;
    }
    try testing.expect(past > n / 2 - 160 and past < n / 2 + 160); // 5 standard deviations
    // top_p 0.9 of a flat row keeps ids 0 .. 1843 (ties go to the lower id): draws reach past 1024, none past 1843
    var high: u32 = 0;
    for (0..n) |position| {
        const id = try smp.draw(logits, .{ .seed = 6, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, position);
        try testing.expect(id <= 1843);
        high = @max(high, id);
    }
    try testing.expectEqual(Path.top_p, smp.path);
    try testing.expect(high > 1800);
}
