//! Shared rows by landing probability and forward costs between timed widths (Python allocate.py, extend_costs).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;

/// A batched head's extra drafts cost the deepest stream's steps; FZ_PRICE_BATCHED=0 prices each stream's.
pub fn pricedBatched() bool {
    const v = std.c.getenv("FZ_PRICE_BATCHED") orelse return true;
    return v[0] != '0';
}

pub const Cost = struct { width: u32, ms: f64 };

/// Draft prefixes per stream with the most expected tokens a ms; fixed rows always run. Caller frees.
pub fn allocate(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64) ![]u32 {
    return allocateDrafts(gpa, fixed, probs, costs, overhead_ms, max_rows, .{});
}

/// What drafting the counted nodes costs: a head step each, or the deepest stream's steps when one batch drafts all.
pub const HeadCost = struct {
    step_ms: f64 = 0.0,
    batched: bool = false,
    free_first: bool = false,
    row_ms: f64 = 0.0, // what each row past a stream's first costs past the table on this machine (below it: less)
    least_rows: u64 = 0, // a measuring round reaches this many rows, the likeliest nodes first, whatever they cost
    pass_ms: f64 = 0.0, // a drafter's own pass: once a round any stream drafts in (the round without drafts has none)
    first_ms: f64 = 0.0, // and its share a drafting stream (charged at the stream's first draft)
    margin: f64 = 1.0, // drafts must beat the round without them by this factor (estimates near the line are noise)
    /// Per stream (as `probs`), what its round costs past the shallow price at c drafts (`depth[s][c]`, c from 0 to its
    /// chain's length): a stream deep in its context pays for its window's rows and its draft over that context
    /// (depth.zig Rule.depthMs). Empty: none (the allocation as before). With it, the next node is the one with the
    /// most chance a ms it adds (the table's next row, its depth, a stream's first its share of the pass), the likeliest
    /// among equal ms.
    depth: []const []const f64 = &.{},
};

/// allocate, each stream's drafts past its first costing one more head step.
pub fn allocatePriced(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64, step_ms: f64) ![]u32 {
    return allocateDrafts(gpa, fixed, probs, costs, overhead_ms, max_rows, .{ .step_ms = step_ms, .free_first = true });
}

/// allocate, each count also priced by the head steps that draft it.
pub fn allocateDrafts(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64, head: HeadCost) ![]u32 {
    const counts = try gpa.alloc(u32, probs.len);
    defer gpa.free(counts);
    const best_counts = try gpa.alloc(u32, probs.len);
    @memset(counts, 0);
    @memset(best_counts, 0);
    var rows: u64 = 0;
    for (fixed) |f| rows += f;
    var expected: f64 = @floatFromInt(rows);
    var drafts: u64 = 0;
    var deepest: u32 = 0;
    var firsts: u64 = 0;
    const timed = !costs.empty();
    var live: u64 = 0;
    for (fixed) |f| live += @intFromBool(f > 0);
    const least = @min(head.least_rows, max_rows);
    const deep = head.depth.len == probs.len;
    var depth_ms: f64 = 0; // the streams' depth at their counts (head.depth)
    if (deep) for (head.depth) |d| {
        depth_ms += d[0];
    };
    var best = if (rows >= least) head.margin * rate(costs, timed, rows, expected, overhead_ms + depth_ms + rowTerm(costs, timed, head, rows, live)) else -std.math.inf(f64);
    while (rows < max_rows) {
        // the likeliest next node, ties to the lowest stream (Python's heapq of (-p, stream)); with depth, the most
        // chance a ms it adds (the table's next row is the same for every stream)
        const row: f64 = if (!deep) 0 else if (timed) @max(1e-3, (costs.get(@intCast(rows + 1)) orelse 0) - (costs.get(@intCast(rows)) orelse 0) + head.row_ms) else 1.0;
        var pick: ?usize = null;
        var pick_p: f64 = 0.0;
        var pick_ms: f64 = 0.0;
        for (probs, 0..) |p, s| {
            if (counts[s] >= p.len) continue;
            const v = p[counts[s]];
            const ms = if (deep) row + @max(0, head.depth[s][counts[s] + 1] - head.depth[s][counts[s]]) + (if (counts[s] == 0) head.first_ms else 0) else 0;
            const better = if (ms == pick_ms) v > pick_p else v * pick_ms > pick_p * ms;
            if (pick == null or better) {
                pick = s;
                pick_p = v;
                pick_ms = ms;
            }
        }
        const s = pick orelse break;
        if (deep) depth_ms += head.depth[s][counts[s] + 1] - head.depth[s][counts[s]];
        counts[s] += 1;
        firsts += @intFromBool(counts[s] == 1);
        rows += 1;
        drafts += 1;
        deepest = @max(deepest, counts[s]);
        expected += pick_p;
        const steps: f64 = @floatFromInt(if (head.batched) deepest -| @intFromBool(head.free_first) else drafts - (if (head.free_first) firsts else 0));
        if (rows < least) continue;
        const pass = head.pass_ms + head.first_ms * @as(f64, @floatFromInt(firsts));
        const now = rate(costs, timed, rows, expected, overhead_ms + depth_ms + head.step_ms * steps + pass + rowTerm(costs, timed, head, rows, live));
        if (now > best or (!timed and head.step_ms == 0)) {
            best = now;
            @memcpy(best_counts, counts);
        }
    }
    // a measuring round whose width is out of reach takes every node on offer
    if (best == -std.math.inf(f64)) @memcpy(best_counts, counts);
    return best_counts;
}

/// head.row_ms for the rows past the streams' first; a fall below the table stops at the streams' plain width.
pub fn rowTerm(costs: *const Table, timed: bool, head: HeadCost, rows: u64, live: u64) f64 {
    const term = head.row_ms * @as(f64, @floatFromInt(rows -| live));
    if (term >= 0 or !timed) return term;
    const at = costs.get(@intCast(rows)) orelse return term;
    const plain = costs.get(@intCast(@max(live, 1))) orelse return term;
    return @max(term, plain - at);
}

fn rate(costs: *const Table, timed: bool, total: u64, expected: f64, overhead_ms: f64) f64 {
    var cost: f64 = 1.0;
    if (timed) cost = costs.get(@intCast(total)) orelse return -1.0;
    return expected / (cost + overhead_ms);
}

/// A chain's nodes' chances from per-depth acceptance (the j-th lands if every earlier one did). Caller frees.
pub fn chainProbabilities(gpa: Allocator, rates: []const f64, count: usize) ![]f64 {
    const out = try gpa.alloc(f64, count);
    var reach: f64 = 1.0;
    for (out, 0..) |*o, j| {
        reach *= if (j < rates.len) rates[j] else rates[rates.len - 1];
        o.* = reach;
    }
    return out;
}

/// Costs for 1..rows: timed widths as measured, interpolated between them, the widest's ms a row past it.
pub fn extendCosts(gpa: Allocator, costs: []const Cost, rows: u32) !Table {
    var out: Table = .{};
    if (costs.len == 0) return out;
    const measured = try gpa.dupe(Cost, costs);
    defer gpa.free(measured);
    std.mem.sort(Cost, measured, {}, struct {
        fn less(_: void, a: Cost, b: Cost) bool {
            return a.width < b.width;
        }
    }.less);
    const widest = measured[measured.len - 1];
    var total: u32 = 1;
    while (total <= rows) : (total += 1) {
        var value: f64 = undefined;
        if (find(measured, total)) |ms| {
            value = ms;
        } else if (total > widest.width) {
            value = widest.ms * @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(widest.width));
        } else {
            var above: Cost = undefined;
            for (measured) |c| {
                if (c.width > total) {
                    above = c;
                    break;
                }
            }
            var below: ?Cost = null;
            for (measured) |c| {
                if (c.width < total) below = c;
            }
            if (below) |b| {
                const span: f64 = @floatFromInt(above.width - b.width);
                value = b.ms + (above.ms - b.ms) * @as(f64, @floatFromInt(total - b.width)) / span;
            } else value = above.ms;
        }
        try out.put(gpa, total, value);
    }
    return out;
}

fn find(costs: []const Cost, width: u32) ?f64 {
    for (costs) |c| if (c.width == width) return c.ms;
    return null;
}

test "a priced drafter pass: a round drafts only where its streams' drafts pay for the pass and their shares" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer costs.deinit(gpa);
    var likely: [8][]const f64 = undefined;
    var some_land: [8][]const f64 = undefined;
    for (&likely, &some_land) |*l, *s| {
        l.* = &.{ 0.9, 0.85, 0.8, 0.75 };
        s.* = &.{ 0.6, 0.4, 0.2 };
    }
    const ones: [8]u32 = @splat(1);
    const head: HeadCost = .{ .pass_ms = 8.0, .first_ms = 4.6 };
    // eight streams whose drafts land some of the time: priced free every stream drafts, priced the plain round wins
    const free = try allocateDrafts(gpa, &ones, &some_land, &costs, 10.0, 32, .{});
    defer gpa.free(free);
    for (free) |c| try std.testing.expect(c >= 1);
    const none = try allocateDrafts(gpa, &ones, &some_land, &costs, 10.0, 32, head);
    defer gpa.free(none);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, none);
    // likely drafts pay for the pass
    const some = try allocateDrafts(gpa, &ones, &likely, &costs, 10.0, 32, head);
    defer gpa.free(some);
    for (some) |c| try std.testing.expect(c >= 1);
}

test "allocate takes the widest round when no cost is timed" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const probs = [_][]const f64{ &.{ 0.9, 0.5 }, &.{0.8} };
    const counts = try allocate(gpa, &.{ 1, 1 }, &probs, &empty, 0.0, 8);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, counts);
}

test "a priced draft step keeps a stream's later drafts out when they land too seldom to pay" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 12.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.6, 0.3, 0.15 }, &.{ 0.6, 0.3, 0.15 } };
    const free = try allocate(gpa, &.{ 1, 1 }, &probs, &costs, 2.0, 8);
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, free);
    const priced = try allocatePriced(gpa, &.{ 1, 1 }, &probs, &costs, 2.0, 8, 1.5);
    defer gpa.free(priced);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, priced);
}

test "a priced draft step leaves each stream's first draft free" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 10.0 } }, 8);
    defer flat.deinit(gpa);
    const probs = [_][]const f64{&.{0.5}};
    // charged, the draft would cost 100 ms against 0.5 tokens: it stays because the first is free
    const counts = try allocatePriced(gpa, &.{1}, &probs, &flat, 0.0, 8, 100.0);
    defer gpa.free(counts);
    try std.testing.expectEqualSlices(u32, &.{1}, counts);
}

test "a priced draft step cuts the stream whose later drafts land seldom and keeps the other's" {
    const gpa = std.testing.allocator;
    var flat = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 10.0 } }, 8);
    defer flat.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.9, 0.85, 0.8 }, &.{ 0.9, 0.2, 0.1 } };
    const priced = try allocatePriced(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8, 1.0);
    defer gpa.free(priced);
    try std.testing.expectEqualSlices(u32, &.{ 3, 1 }, priced);
    const free = try allocate(gpa, &.{ 1, 1 }, &probs, &flat, 0.0, 8);
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, free);
}

test "a zero draft step prices rounds as allocate does" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 1, .ms = 10.0 }, .{ .width = 8, .ms = 12.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.9, 0.85, 0.8 }, &.{ 0.9, 0.2, 0.1 }, &.{0.4} };
    const priced = try allocatePriced(gpa, &.{ 1, 1, 1 }, &probs, &costs, 2.0, 8, 0.0);
    defer gpa.free(priced);
    const free = try allocate(gpa, &.{ 1, 1, 1 }, &probs, &costs, 2.0, 8);
    defer gpa.free(free);
    try std.testing.expectEqualSlices(u32, free, priced);
}

test "a measuring round reaches its rows, then keeps only rows that pay; out of reach it takes every node" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 2, .ms = 20.0 }, .{ .width = 8, .ms = 80.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.5, 0.25 }, &.{ 0.5, 0.25 } };
    // 10 ms a row against 10 tokens a 100 ms: no draft pays
    const none = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 8, .{});
    defer gpa.free(none);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0 }, none);
    const four = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 8, .{ .least_rows = 4 });
    defer gpa.free(four);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, four);
    const all = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 8, .{ .least_rows = 7 });
    defer gpa.free(all);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2 }, all);
}

test "rows cheaper than the table make drafts pay, never pricing a round under its plain width" {
    const gpa = std.testing.allocator;
    var costs = try extendCosts(gpa, &.{ .{ .width = 2, .ms = 20.0 }, .{ .width = 8, .ms = 80.0 } }, 8);
    defer costs.deinit(gpa);
    const probs = [_][]const f64{ &.{ 0.5, 0.25 }, &.{ 0.5, 0.25 } };
    // measured: each row costs 8 ms less than the table's 10: 2 ms a row, so first drafts pay
    const cheap = try allocateDrafts(gpa, &.{ 1, 1 }, &probs, &costs, 0.0, 8, .{ .row_ms = -8.0 });
    defer gpa.free(cheap);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, cheap);
    // a fall past the table's own slope is bounded: a round never costs less than its plain width
    const head: HeadCost = .{ .row_ms = -25.0 };
    try std.testing.expectApproxEqAbs(@as(f64, -60.0), rowTerm(&costs, true, head, 8, 2), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -16.0), rowTerm(&costs, true, .{ .row_ms = -8.0 }, 4, 2), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), rowTerm(&costs, true, .{ .row_ms = 3.0 }, 4, 2), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -16.0), rowTerm(&costs, false, .{ .row_ms = -8.0 }, 4, 2), 1e-9);
}

test "extendCosts interpolates and extrapolates" {
    const gpa = std.testing.allocator;
    var t = try extendCosts(gpa, &.{ .{ .width = 4, .ms = 8.0 }, .{ .width = 2, .ms = 4.0 } }, 6);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(?f64, 4.0), t.get(1));
    try std.testing.expectEqual(@as(?f64, 6.0), t.get(3));
    try std.testing.expectEqual(@as(?f64, 12.0), t.get(6));
}
