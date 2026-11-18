//! lone_cut (depth.zig Rule.cut, windows.zig cutLone): a lone greedy stream at or below depth_from verifies the prefix
//! of its held drafts that the head's chances price best; shared rounds, sampled, grammar and deeper streams are not
//! touched, and every stream commits the same tokens as without it.
const std = @import("std");
const lanes = @import("lanes.zig");
const fake = lanes.fake;
const depth = lanes.depth;
const win = lanes.windows;
const engine_test = @import("engine_test.zig");

const gpa = std.testing.allocator;

/// A head's chances as the hook gives them (each draft's own; the hook returns their running product), and what the
/// test saw: calls, calls outside a lone round, the deepest context a call came at, the streams it was asked for.
var hook: struct {
    chance: f64 = 0.0,
    give: bool = true,
    calls: usize = 0,
    shared_calls: usize = 0,
    deepest: u64 = 0,
    engine: ?*lanes.Engine = null,
} = .{};

fn chances(_: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    hook.calls += 1;
    if (hook.engine) |e| hook.shared_calls += @intFromBool(!e.alone);
    hook.deepest = @max(hook.deepest, s.cache_len);
    if (!hook.give) return false;
    var product: f64 = 1;
    for (out) |*o| {
        product *= hook.chance;
        o.* = product;
    }
    return true;
}

/// Prices by context (two nodes: windows rising 0.8 ms a row over a 6 ms base, a 2 ms draft) and `depth_from`.
const window_ms = [_]f64{ 5.8, 6.6, 7.4, 8.2, 9.0, 9.8, 10.6, 11.4 };
const nodes = [_]depth.ContextCost{
    .{ .context = 8, .window_ms = &window_ms, .draft_ms = 2.0 },
    .{ .context = 1 << 20, .window_ms = &window_ms, .draft_ms = 2.0 },
};

const Run = struct {
    tokens: []u32,
    by_drafts: [lanes.stream.depth_buckets]u32, // rounds by drafts verified (Stream.depth_rounds), every stream's
};

/// The cases admitted together and stepped until done, with the hook on the fake target and `cut` the fact.
fn run(cases: []const engine_test.Case, cut: bool, depth_from: u64) ![]Run {
    return runWith(cases, cut, depth_from, .{});
}

/// `run`'s options: every stream in the cut arm (Stream.cut_arm, A/B tooling), and a look at the engine once done.
const With = struct {
    cut_arm: bool = false,
    done: ?*const fn (*lanes.Engine) anyerror!void = null,
};

fn runWith(cases: []const engine_test.Case, cut: bool, depth_from: u64, with: With) ![]Run {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .context_costs = &nodes, .depth_from = depth_from, .lone_cut = cut }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var b = target.backend();
    var vt = b.vtable.*;
    vt.probabilities = chances;
    b.vtable = &vt;
    var clock: fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, b, clock.clock());
    defer engine.deinit();
    hook.engine = &engine;
    defer hook.engine = null;
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    const classes = try gpa.alloc(engine_test.Classes, cases.len);
    defer gpa.free(classes);
    const constraints = try gpa.alloc(lanes.grammar.Constraint, cases.len);
    defer gpa.free(constraints);
    for (cases, streams, classes, constraints) |c, *s, *g, *k| {
        if (c.classes) |given| {
            g.* = given;
            k.* = .init(g.rules(), (fake.vocab + 31) / 32, given.after);
        }
        s.* = try lanes.Stream.init(gpa, .{ .id = "cut", .prompt = c.prompt, .max_new = c.max_new, .eos = c.eos, .sampling = c.sampling, .drafts = c.drafts, .grammar = if (c.classes != null) k else null });
    }
    defer for (streams) |*s| s.deinit(gpa);
    for (streams) |*s| s.cut_arm = with.cut_arm;
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    if (with.done) |look| try look(&engine);
    const out = try gpa.alloc(Run, cases.len);
    for (out, streams) |*o, *s| o.* = .{ .tokens = try gpa.dupe(u32, s.emitted()), .by_drafts = s.depth_rounds };
    return out;
}

fn free(runs: []Run) void {
    for (runs) |r| gpa.free(r.tokens);
    gpa.free(runs);
}

fn reset(chance: f64, give: bool) void {
    hook = .{ .chance = chance, .give = give };
}

/// Rounds that verified more than one draft.
fn wide(r: Run) u32 {
    var n: u32 = 0;
    for (r.by_drafts[2..]) |x| n += x;
    return n;
}

const long_prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };

test "lone cut: off, no hook call; on, the same tokens with only the drafts that pay verified" {
    const cases = [_]engine_test.Case{.{ .prompt = &.{ 3, 1, 4 }, .max_new = 120, .eos = &.{} }};
    reset(0.0, true);
    const off = try run(&cases, false, 1 << 15);
    defer free(off);
    try std.testing.expectEqual(@as(usize, 0), hook.calls);
    try std.testing.expect(wide(off[0]) > 0); // the depth rule holds chains of several drafts
    reset(0.0, true); // no draft looks likely: the cut keeps one, never none
    const low = try run(&cases, true, 1 << 15);
    defer free(low);
    try std.testing.expect(hook.calls > 0);
    try std.testing.expectEqualSlices(u32, off[0].tokens, low[0].tokens);
    try std.testing.expectEqual(@as(u32, 0), wide(low[0]));
    try std.testing.expect(low[0].by_drafts[1] > 0);
    reset(1.0, true); // every draft certain: nothing cut
    const high = try run(&cases, true, 1 << 15);
    defer free(high);
    try std.testing.expectEqualSlices(u32, off[0].tokens, high[0].tokens);
    try std.testing.expect(wide(high[0]) > 0);
    reset(0.0, false); // a hook without chances: the window stays whole, the rounds as without the fact
    const none = try run(&cases, true, 1 << 15);
    defer free(none);
    try std.testing.expect(hook.calls > 0);
    try std.testing.expectEqualSlices(u32, off[0].tokens, none[0].tokens);
    try std.testing.expectEqualSlices(u32, &off[0].by_drafts, &none[0].by_drafts);
}

test "lone cut: shared rounds never ask the hook, and every stream commits what it commits without the fact" {
    const cases = [_]engine_test.Case{ .{ .prompt = &.{ 3, 1, 4 }, .max_new = 60, .eos = &.{} }, .{ .prompt = &.{ 2, 7, 1, 8 }, .max_new = 60, .eos = &.{} }, .{ .prompt = &.{ 5, 5, 5 }, .max_new = 60, .eos = &.{} } };
    reset(0.0, true);
    const off = try run(&cases, false, 1 << 15);
    defer free(off);
    reset(0.0, true);
    const on = try run(&cases, true, 1 << 15);
    defer free(on);
    try std.testing.expectEqual(@as(usize, 0), hook.shared_calls);
    for (off, on) |x, y| try std.testing.expectEqualSlices(u32, x.tokens, y.tokens);
}

test "lone cut: sampled, grammar and deeper streams never ask the hook" {
    reset(0.0, true);
    const sampled = [_]engine_test.Case{.{ .prompt = &.{ 3, 1, 4 }, .max_new = 60, .sampling = .{ .temperature = 0.8, .seed = 7 }, .eos = &.{} }};
    const s = try run(&sampled, true, 1 << 15);
    defer free(s);
    try std.testing.expectEqual(@as(usize, 0), hook.calls);
    const shaped = [_]engine_test.Case{.{ .prompt = &.{ 3, 1, 4 }, .max_new = 30, .classes = .{} }};
    const g = try run(&shaped, true, 1 << 15);
    defer free(g);
    try std.testing.expectEqual(@as(usize, 0), hook.calls);
    const deep = [_]engine_test.Case{.{ .prompt = &long_prompt, .max_new = 60, .eos = &.{} }};
    const d = try run(&deep, true, long_prompt.len - 1);
    defer free(d);
    try std.testing.expectEqual(@as(usize, 0), hook.calls);
    reset(0.0, true);
    const plain = try run(&deep, false, long_prompt.len - 1);
    defer free(plain);
    try std.testing.expectEqualSlices(u32, plain[0].tokens, d[0].tokens);
    try std.testing.expectEqualSlices(u32, &plain[0].by_drafts, &d[0].by_drafts);
}

test "lone cut: a stream crossing depth_from mid-reply is cut up to it and not past it" {
    const cases = [_]engine_test.Case{.{ .prompt = &long_prompt, .max_new = 120, .eos = &.{} }};
    const from = long_prompt.len + 40;
    reset(0.0, true);
    const off = try run(&cases, false, from);
    defer free(off);
    reset(0.0, true);
    const on = try run(&cases, true, from);
    defer free(on);
    try std.testing.expect(hook.calls > 0);
    try std.testing.expect(hook.deepest <= from);
    try std.testing.expectEqualSlices(u32, off[0].tokens, on[0].tokens);
    try std.testing.expect(wide(on[0]) > 0); // past depth_from the rule's chains are verified whole again
}

fn rule(nodes_: []const depth.ContextCost) depth.Rule {
    const T = @typeInfo(@FieldType(depth.Rule, "family_costs")).pointer.child;
    const empty = struct {
        var t: T = .{};
    };
    return .{ .gpa = gpa, .prior = &.{0.8}, .most_drafts = 7, .family_costs = &empty.t, .shared_costs = &empty.t, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = nodes_, .depth_from = 1 << 15 };
}

test "lone cut: the cut verifies one draft at least and never more than it holds" {
    var r = rule(&nodes);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 1), r.cut(&.{ 0, 0, 0, 0, 0, 0, 0 }, 100));
    try std.testing.expectEqual(@as(usize, 7), r.cut(&.{ 1, 1, 1, 1, 1, 1, 1 }, 100));
    try std.testing.expectEqual(@as(usize, 3), r.cut(&.{ 1, 1, 1 }, 100));
    try std.testing.expectEqual(@as(usize, 1), r.cut(&.{0.0}, 100));
    // a first draft at 0.95 and the second at 0.10 of it: one more row (0.8 ms) is worth 0.095 tokens, under the
    // round's rate (1.95 / 8.6 ms = 0.23 a ms)
    try std.testing.expectEqual(@as(usize, 1), r.cut(&.{ 0.95, 0.095, 0.01 }, 100));
    try std.testing.expectEqual(@as(usize, 2), r.cut(&.{ 0.95, 0.9, 0.05 }, 100));
    var unpriced = rule(&.{});
    defer unpriced.deinit();
    try std.testing.expectEqual(@as(usize, 4), unpriced.cut(&.{ 0, 0, 0, 0 }, 100)); // no prices: nothing cut
}

test "lone cut: prices are the windows' line in rows plus the draft, and one slow round moves them under 1 ms" {
    // a noisy node: rows 2 .. 8 lie on 40 + 4 r except the 6-row window, 3 ms dear
    const noisy = [_]f64{ 30, 48, 52, 56, 60, 67, 68, 72 };
    const n = [_]depth.ContextCost{.{ .context = 600, .window_ms = &noisy, .draft_ms = 8 }};
    var r = rule(&n);
    defer r.deinit();
    const line = struct {
        fn at(d: f64) f64 {
            return 40 + 4 * (d + 1) + 8 + 3.0 / 7.0 + (3.0 * (5 - 4) * (d - 4)) / 28.0; // least squares through the bump
        }
    }.at;
    for (1..8) |d| try std.testing.expectApproxEqAbs(line(@floatFromInt(d)), r.cutPrice(@intCast(d), 600).?, 1e-9);
    try std.testing.expectEqual(@as(?f64, null), r.cutPrice(0, 600));
    const before = r.cutPrice(4, 600).?;
    r.observeCut(4, 3 * before, 600);
    const after = r.cutPrice(4, 600).?;
    try std.testing.expect(after - before > 0 and after - before < 1.0);
    for (1..8) |d| try std.testing.expectApproxEqAbs(after - before, r.cutPrice(@intCast(d), 600).? - line(@floatFromInt(d)), 1e-9);
    // rounds 1.5 ms past their price: the figure settles there
    for (0..80) |i| r.observeCut(@intCast(1 + i % 7), line(@floatFromInt(1 + i % 7)) + 1.5, 600);
    try std.testing.expectApproxEqAbs(1.5, r.cutPrice(3, 600).? - line(3), 1e-6);
}

test "lone cut a/b: each arm's round times price its own streams alone" {
    var fc: lanes.table.Table = .{};
    defer fc.deinit(gpa);
    for (1..9) |w| try fc.put(gpa, @intCast(w), 10 + @as(f64, @floatFromInt(w)));
    var r: depth.Rule = .{ .gpa = gpa, .prior = &.{ 0.8, 0.8, 0.8 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &fc, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
    defer r.deinit();
    var states: [4]?depth.State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    const whole: depth.Who = .{ .state = &states[0], .draft_room = 8, .finished = false, .forced = false };
    const whole_sampled: depth.Who = .{ .state = &states[1], .draft_room = 8, .finished = false, .forced = false, .sampled = true };
    const cut: depth.Who = .{ .state = &states[2], .draft_room = 8, .finished = false, .forced = false, .cut_arm = true };
    const cut_sampled: depth.Who = .{ .state = &states[3], .draft_room = 8, .finished = false, .forced = false, .sampled = true, .cut_arm = true };
    // nothing timed: every stream prices two drafts by the table (a window of 3 rows, 13 ms)
    for ([_]depth.Who{ whole, whole_sampled, cut, cut_sampled }) |w| try std.testing.expectEqual(@as(?f64, 13), try r.roundCost(2, w));
    // a cut-arm round (its draft flushed early: slower) prices its arm's streams, greedy and sampled, and no other
    try r.observeCost(2, 40, false, cut);
    try std.testing.expectEqual(@as(?f64, 40), try r.roundCost(2, cut));
    try std.testing.expectEqual(@as(?f64, 40), try r.roundCost(2, cut_sampled));
    try std.testing.expectEqual(@as(?f64, 13), try r.roundCost(2, whole));
    try std.testing.expectEqual(@as(?f64, 13), try r.roundCost(2, whole_sampled));
    // and a whole-arm round its own, as a stream outside the A/B prices
    try r.observeCost(2, 20, false, whole);
    try std.testing.expectEqual(@as(?f64, 20), try r.roundCost(2, whole));
    try std.testing.expectEqual(@as(?f64, 20), try r.roundCost(2, whole_sampled));
    try std.testing.expectEqual(@as(?f64, 40), try r.roundCost(2, cut));
    try std.testing.expectEqual(@as(?f64, 20), try r.roundCost(2, null));
}

test "lone cut a/b: a cut round crossing depth_from leaves the whole arm's deep prices as they were" {
    // a cut-arm stream from below depth_from to just past it: cut rounds up to depth_from, the last of them ending
    // past it (timed at the context it ends at, so priced by context), then a few deep rounds
    const from = long_prompt.len + 40;
    const cases = [_]engine_test.Case{.{ .prompt = &long_prompt, .max_new = 46, .eos = &.{} }};
    const look = struct {
        fn done(e: *lanes.Engine) !void {
            // a whole-arm stream's next price past depth_from: the price at its context, nothing learned on top
            var st: ?depth.State = null;
            defer if (st) |*x| x.deinit(gpa);
            const whole: depth.Who = .{ .state = &st, .draft_room = 8, .finished = false, .forced = false, .context = from + 8 };
            var d: i64 = 0;
            while (d < 7) : (d += 1) try std.testing.expectEqual(depth.priceAt(&nodes, d, from + 8), try e.rule.roundCost(d, whole));
            // the whole arm's deep table untouched; the cut arm's rounds past depth_from timed in its own
            for (e.rule.context_ms.slots) |x| try std.testing.expectEqual(@as(?f64, null), x);
            var timed: usize = 0;
            for (e.rule.cut_arm_context_ms.slots) |x| timed += @intFromBool(x != null);
            try std.testing.expect(timed > 0);
        }
    };
    reset(0.0, true);
    const runs = try runWith(&cases, true, from, .{ .cut_arm = true, .done = look.done });
    defer free(runs);
    try std.testing.expect(hook.calls > 0 and hook.deepest <= from); // cut up to depth_from
    try std.testing.expect(long_prompt.len + runs[0].tokens.len > from + 1); // and past it
}
