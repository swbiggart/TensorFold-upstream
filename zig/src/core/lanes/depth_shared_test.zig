//! Context costs in shared rounds: a stream deep in its context pays for its rows and its draft there, so it drafts
//! less or not at all beside shallow streams, which draft as they did without context costs; and the A/B modes.
const std = @import("std");
const alloc = @import("allocate.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const a = std.testing.allocator;

const No = struct {
    fn copy(_: *anyopaque, _: *const depth.Who) bool {
        return false;
    }
};
var unused: u8 = 0;
const probe: depth.Probe = .{ .ptr = &unused, .copy = No.copy };

// a window's ms by rows and the draft at a shallow, a flat and a deep context (the Sparks' shape, four widths)
const nodes = [_]depth.ContextCost{
    .{ .context = 600, .window_ms = &.{ 34.1, 42.4, 46.9, 51.7 }, .draft_ms = 8.4 },
    .{ .context = 32_768, .window_ms = &.{ 35.1, 42.2, 47.7, 52.6 }, .draft_ms = 9.4 },
    .{ .context = 524_000, .window_ms = &.{ 63.2, 72.0, 82.7, 99.9 }, .draft_ms = 39.7 },
};
const pass: depth.DraftPass = .{ .base_ms = 3.8, .stream_ms = 4.6, .drafted_ms = 4.2, .fair_above = 4 };

fn rule(shared: *const Table, empty: *const Table, by_context: bool, prior: []const f64) depth.Rule {
    return .{ .gpa = a, .prior = prior, .most_drafts = 3, .family_costs = empty, .shared_costs = shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32, .batched_heads = true, .draft_pass = pass, .context_costs = if (by_context) &nodes else &.{}, .depth_from = 32_768 };
}

test "context costs: depthMs is the price past depth_from, none at or below it" {
    var shared = try alloc.extendCosts(a, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer shared.deinit(a);
    const empty: Table = .{};
    var r = rule(&shared, &empty, true, &.{0.84});
    defer r.deinit();
    try std.testing.expectEqual(@as(f64, 0), r.depthMs(0, 600));
    try std.testing.expectEqual(@as(f64, 0), r.depthMs(2, 32_768));
    try std.testing.expectApproxEqAbs(@as(f64, 63.2 - 35.1), r.depthMs(0, 524_000), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 72.0 + 39.7 - 42.2 - 9.4), r.depthMs(1, 524_000), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, (72.0 + 39.7 - 42.2 - 9.4) / 2.0), r.depthMs(1, (32_768 + 524_000) / 2), 1e-6);
    var none = rule(&shared, &empty, false, &.{0.84});
    defer none.deinit();
    try std.testing.expectEqual(@as(f64, 0), none.depthMs(1, 524_000));
}

test "context costs in shared rounds: shallow streams draft as without them; a deep one beside them goes plain" {
    var shared = try alloc.extendCosts(a, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer shared.deinit(a);
    const empty: Table = .{};
    for ([_][]const f64{ &.{ 0.84, 0.81, 0.81 }, &.{ 0.6, 0.4, 0.3 } }) |prior| {
        // every stream within depth_from: the same budgets as a rule without context costs, round after round
        var plain_rule = rule(&shared, &empty, false, prior);
        defer plain_rule.deinit();
        var ctx_rule = rule(&shared, &empty, true, prior);
        defer ctx_rule.deinit();
        var s1: [4]?depth.State = @splat(null);
        defer for (&s1) |*s| if (s.*) |*x| x.deinit(a);
        var s2: [4]?depth.State = @splat(null);
        defer for (&s2) |*s| if (s.*) |*x| x.deinit(a);
        var w1: [4]depth.Who = undefined;
        var w2: [4]depth.Who = undefined;
        for (&w1, &w2, &s1, &s2, 0..) |*x, *y, *p, *q, i| {
            const ctx: u64 = 600 + 10_000 * i; // up to 30,600: flat
            x.* = .{ .state = p, .draft_room = 100, .finished = false, .forced = false, .context = ctx };
            y.* = .{ .state = q, .draft_room = 100, .finished = false, .forced = false, .context = ctx };
        }
        var o1: [4]i64 = undefined;
        var o2: [4]i64 = undefined;
        for (0..12) |round| {
            try plain_rule.budgets(&w1, probe, false, &o1);
            try ctx_rule.budgets(&w2, probe, false, &o2);
            try std.testing.expectEqualSlices(i64, &o1, &o2);
            const rows: u64 = 4 + @as(u64, @intCast(o1[0] + o1[1] + o1[2] + o1[3]));
            const ms = shared.get(@intCast(rows)).? + 6.0 + @as(f64, @floatFromInt(round % 3));
            try plain_rule.observeOverhead(4, rows, ms, false, plain_rule.headMs(&o1));
            try ctx_rule.observeOverheadAt(4, rows, ms, false, ctx_rule.headMs(&o2), 0);
        }
    }
    // one of the four at 524,000: likely drafts, the shallow streams draft, the deep one does not (its first draft
    // costs ~32 ms more than at the shallow price); without context costs it drafts as the others do
    {
        const prior = &[_]f64{ 0.84, 0.81, 0.81 };
        var plain_rule = rule(&shared, &empty, false, prior);
        defer plain_rule.deinit();
        var ctx_rule = rule(&shared, &empty, true, prior);
        defer ctx_rule.deinit();
        var s1: [4]?depth.State = @splat(null);
        defer for (&s1) |*s| if (s.*) |*x| x.deinit(a);
        var s2: [4]?depth.State = @splat(null);
        defer for (&s2) |*s| if (s.*) |*x| x.deinit(a);
        var w1: [4]depth.Who = undefined;
        var w2: [4]depth.Who = undefined;
        for (&w1, &w2, &s1, &s2, 0..) |*x, *y, *p, *q, i| {
            const ctx: u64 = if (i == 2) 524_000 else 3_000;
            x.* = .{ .state = p, .draft_room = 100, .finished = false, .forced = false, .context = ctx };
            y.* = .{ .state = q, .draft_room = 100, .finished = false, .forced = false, .context = ctx };
        }
        var o1: [4]i64 = undefined;
        var o2: [4]i64 = undefined;
        try plain_rule.budgets(&w1, probe, false, &o1);
        try ctx_rule.budgets(&w2, probe, false, &o2);
        for (o1) |d| try std.testing.expect(d >= 1);
        try std.testing.expectEqual(@as(i64, 0), o2[2]);
        for (o2, 0..) |d, i| if (i != 2) try std.testing.expect(d >= 1);
        // a deep stream whose drafts nearly always land, beside one whose drafts seldom do: it pays for its depth
        // and drafts (the round's tokens a ms past plain's), the shallow one does not; beside a shallow stream whose
        // drafts land as often, the shallow one's drafts are the cheaper tokens and the deep one goes plain
        var sure = rule(&shared, &empty, true, &.{ 0.99, 0.99, 0.99 });
        defer sure.deinit();
        var s3: [2]?depth.State = @splat(null);
        defer for (&s3) |*s| if (s.*) |*x| x.deinit(a);
        var w3 = [_]depth.Who{
            .{ .state = &s3[0], .draft_room = 100, .finished = false, .forced = false, .context = 524_000 },
            .{ .state = &s3[1], .draft_room = 100, .finished = false, .forced = false, .context = 3_000 },
        };
        var o3: [2]i64 = undefined;
        try sure.budgets(&w3, probe, false, &o3);
        try std.testing.expectEqualSlices(i64, &.{ 0, 3 }, &o3);
        @memcpy(s3[1].?.p, &[_]f64{ 0.2, 0.1, 0.1 });
        try sure.budgets(&w3, probe, false, &o3);
        try std.testing.expectEqual(@as(i64, 0), o3[0]); // a deep chain starts at one draft (ramp): one does not pay
        s3[0].?.reach = 4; // its chains have landed three deep
        try sure.budgets(&w3, probe, false, &o3);
        try std.testing.expect(o3[0] >= 2);
        try std.testing.expectEqual(@as(i64, 0), o3[1]);
    }
}

test "context costs in shared rounds: the streams' depth is no part of the overhead a round teaches" {
    var shared = try alloc.extendCosts(a, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer shared.deinit(a);
    const empty: Table = .{};
    var r = rule(&shared, &empty, true, &.{0.84});
    defer r.deinit();
    const deep = r.depthMs(0, 524_000);
    try r.observeOverheadAt(4, 4, shared.get(4).? + 10.0 + deep, false, 0, deep); // a plain round, one stream deep
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), r.fixedMs(4, false), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), r.plain_overhead.items[0].ms, 1e-9);
}

test "context costs: A/B modes take turns (by context, by the table, plain) and price as their mode says" {
    var fc: Table = .{};
    defer fc.deinit(a);
    for (nodes[0].window_ms, 0..) |ms, i| try fc.put(a, @intCast(i + 1), ms + if (i > 0) nodes[0].draft_ms else 0);
    const shared: Table = .{};
    // drafts that land seldom, at 524,000: by context plain (then a probe), by the table one draft, plain none
    var r: depth.Rule = .{ .gpa = a, .prior = &.{ 0.3, 0.1, 0.05 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = &nodes, .ab_modes = true };
    defer r.deinit();
    var states: [4]?depth.State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(a);
    var whos: [4]depth.Who = undefined;
    for (&whos, &states) |*w, *s| w.* = .{ .state = s, .draft_room = 100, .finished = false, .forced = false, .context = 524_000 };
    try std.testing.expectEqual(@as(i64, 0), try r.depth(whos[0]));
    try std.testing.expectEqual(@as(i64, 1), try r.depth(whos[1]));
    try std.testing.expectEqual(@as(i64, 0), try r.depth(whos[2]));
    try std.testing.expectEqual(@as(i64, 0), try r.depth(whos[3])); // the fourth: by context again
    try std.testing.expectEqual(depth.Mode.rule, states[0].?.mode);
    try std.testing.expectEqual(depth.Mode.table, states[1].?.mode);
    try std.testing.expectEqual(depth.Mode.plain, states[2].?.mode);
    try std.testing.expectEqual(depth.Mode.rule, states[3].?.mode);
    // the table stream's times are its own (raw), the others' past their prices
    try r.observeCost(1, 70.0, false, whos[1]);
    try std.testing.expectApproxEqAbs(@as(f64, 70.0), r.ab_table_ms.get(1).?, 1e-12);
    try std.testing.expectEqual(@as(?f64, null), r.context_ms.get(1));
    try r.observeCost(0, 64.2, false, whos[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), r.context_ms.get(0).?, 1e-9);
    // plain stays plain, likely drafts or not; a table stream prices as if shallow (no depth in shared rounds)
    @memcpy(states[2].?.p, &[_]f64{ 0.99, 0.99, 0.99 });
    for (0..40) |_| try std.testing.expectEqual(@as(i64, 0), try r.depth(whos[2]));
    try std.testing.expectEqual(@as(f64, 0), try r.streamDepthMs(whos[1], 1));
    try std.testing.expect(try r.streamDepthMs(whos[2], 0) > 0);
}

test "allocate with depth: a costly stream's nodes come after cheaper ones of near chance, and none without depth as before" {
    var costs = try alloc.extendCosts(a, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer costs.deinit(a);
    const probs = [_][]const f64{ &.{ 0.9, 0.8, 0.7 }, &.{ 0.85, 0.7, 0.6 } };
    const ones = [_]u32{ 1, 1 };
    const flat = [_][]const f64{ &.{ 0, 0, 0, 0 }, &.{ 0, 0, 0, 0 } };
    const before = try alloc.allocateDrafts(a, &ones, &probs, &costs, 8.0, 32, .{});
    defer a.free(before);
    const zero = try alloc.allocateDrafts(a, &ones, &probs, &costs, 8.0, 32, .{ .depth = &flat });
    defer a.free(zero);
    try std.testing.expectEqualSlices(u32, before, zero);
    // the first stream deep: its first draft 30 ms dearer, so it drafts none; the second as before or more
    const deep = [_][]const f64{ &.{ 28, 58, 63, 75 }, &.{ 0, 0, 0, 0 } };
    const priced = try alloc.allocateDrafts(a, &ones, &probs, &costs, 8.0, 32, .{ .depth = &deep });
    defer a.free(priced);
    try std.testing.expectEqual(@as(u32, 0), priced[0]);
    try std.testing.expect(priced[1] >= before[1]);
}

test "context costs: a lone stream at or below depth_from drafts as without them; past it plain competes" {
    var fc: Table = .{};
    defer fc.deinit(a);
    for (nodes[0].window_ms, 0..) |ms, i| try fc.put(a, @intCast(i + 1), ms + if (i > 0) nodes[0].draft_ms else 0);
    const shared: Table = .{};
    // drafts that land seldom: without context costs one draft at least, whatever the context
    var r: depth.Rule = .{ .gpa = a, .prior = &.{ 0.3, 0.1, 0.05 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = &nodes, .depth_from = 32_768 };
    defer r.deinit();
    var states: [3]?depth.State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(a);
    const ctx = [_]u64{ 3_000, 32_768, 524_000 };
    for (&states, ctx) |*s, c| {
        const w: depth.Who = .{ .state = s, .draft_room = 100, .finished = false, .forced = false, .context = c };
        try std.testing.expectEqual(@as(i64, @intFromBool(c <= 32_768)), try r.depth(w));
    }
    // a shallow stream's round times are its own (raw, the table rule's), a deep one's past its price
    const shallow: depth.Who = .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false, .context = 3_000 };
    const deep: depth.Who = .{ .state = &states[2], .draft_room = 100, .finished = false, .forced = false, .context = 524_000 };
    try r.observeCost(1, 52.0, false, shallow);
    try std.testing.expectApproxEqAbs(@as(f64, 52.0), r.greedy_ms.get(1).?, 1e-12);
    try r.observeCost(1, 113.7, false, deep);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), r.context_ms.get(1).?, 1e-9);
}

test "context costs: a deep stream's chain starts at one draft and reaches one further each time a chain lands" {
    var fc: Table = .{};
    defer fc.deinit(a);
    for (nodes[0].window_ms, 0..) |ms, i| try fc.put(a, @intCast(i + 1), ms + if (i > 0) nodes[0].draft_ms else 0);
    const shared: Table = .{};
    var r: depth.Rule = .{ .gpa = a, .prior = &.{ 0.99, 0.99, 0.99 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = &nodes, .depth_from = 32_768 };
    defer r.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(a);
    const w: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .context = 524_000 };
    try std.testing.expectEqual(@as(i64, 0), try r.depth(w)); // plain until a plain round is timed past depth_from
    try r.observeCost(0, 63.2, false, w);
    try std.testing.expectEqual(@as(i64, 1), try r.depth(w));
    try r.observeDepth(w, 1, 1);
    try std.testing.expectEqual(@as(i64, 2), try r.depth(w));
    try r.observeDepth(w, 2, 2);
    try std.testing.expectEqual(@as(i64, 3), try r.depth(w));
    try r.observeDepth(w, 3, 0); // a chain that missed at once: the reach stays, the estimates decide below it
    try std.testing.expectEqual(@as(i64, 3), try r.depth(w));
    for (0..12) |_| try r.observeDepth(w, 1, 0); // its drafts stop landing: plain
    try std.testing.expectEqual(@as(i64, 0), try r.depth(w));
    // a shallow stream with the same estimates takes its whole chain at once
    var s2: ?depth.State = null;
    defer if (s2) |*s| s.deinit(a);
    try std.testing.expectEqual(@as(i64, 3), try r.depth(.{ .state = &s2, .draft_room = 100, .finished = false, .forced = false, .context = 3_000 }));
}

test "context costs: rounds far cheaper than their prices (another machine) still choose by what they took" {
    var fc: Table = .{};
    defer fc.deinit(a);
    for (nodes[0].window_ms, 0..) |ms, i| try fc.put(a, @intCast(i + 1), ms + if (i > 0) nodes[0].draft_ms else 0);
    const shared: Table = .{};
    var r: depth.Rule = .{ .gpa = a, .prior = &.{ 0.99, 0.99, 0.99 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = &nodes, .depth_from = 32_768 };
    defer r.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(a);
    const w: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .context = 524_000 };
    try r.observeCost(1, 20.0, false, w); // a drafted round took 20 ms (priced 111.7): no plain round timed yet
    try std.testing.expectEqual(@as(i64, 0), try r.depth(w));
    try r.observeCost(0, 4.6, false, w); // plain took 4.6 (priced 63.2): plain's 1 token in 4.6 ms wins
    for (0..7) |_| try std.testing.expectEqual(@as(i64, 0), try r.depth(w));
}
