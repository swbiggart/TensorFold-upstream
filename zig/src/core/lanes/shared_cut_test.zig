//! shared_cut (windows.zig sharedChances, Engine.deepened): a shared round's greedy streams' held drafts priced by this
//! round's chances (Backend.probabilities) in place of the learned rates; `deepen` drafts each drafting greedy stream's
//! whole block. Off: no chances asked; lone, sampled and grammar streams never ask; every stream's tokens are the same.
const std = @import("std");
const lanes = @import("lanes.zig");
const fake = lanes.fake;
const depth = lanes.depth;
const SharedCut = lanes.config.SharedCut;
const t = std.testing;
const a = t.allocator;

var hook: struct { chance: f64 = 0.5, calls: usize = 0, sampled_calls: usize = 0, lone_calls: usize = 0, wide_calls: usize = 0, deny_after: ?usize = null, engine: ?*lanes.Engine = null } = .{};

fn chances(_: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
    hook.calls += 1;
    if (s.sampling != null) hook.sampled_calls += 1;
    if (hook.engine) |e| hook.lone_calls += @intFromBool(e.alone);
    if (hook.engine) |e| hook.wide_calls += @intFromBool(e.activeCount() > 4);
    if (hook.deny_after) |n| if (hook.calls > n) return false; // the chances lost (as a prefill's flush drops them)
    var product: f64 = 1;
    for (out) |*o| {
        product *= hook.chance;
        o.* = product;
    }
    return true;
}

/// Transitions the runs take: stream 1's reply length (null: 48 like the others), a stream that joins after this many
/// steps (null: none), and a prior that makes the rule's budgets short (one or two drafts) while the chances are high.
var short_for_test: ?u32 = null;
var join_for_test: ?usize = null;
var low_prior_for_test = false;
/// The runs timed by the fake clock (rounds learn their overheads), else untimed.
var timed_for_test = false;
/// Windows of rounds that do not price by chances (lone, or 5 or more live streams) and how many verified more held
/// drafts than the stream's budget (`Stream.asked`): what v0.2 would not have verified there.
var watch: struct { orig: ?*const fn (ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void = null, checked: usize = 0, past_budget: usize = 0, checked_unpriced: usize = 0, past_budget_unpriced: usize = 0 } = .{};
/// From this many chance reads on, every read gives none (null: never).
var deny_for_test: ?usize = null;

fn verifyWatched(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
    if (hook.engine) |e| {
        const denied = hook.deny_after != null and hook.calls > hook.deny_after.?;
        for (windows) |w| {
            if (w.stream.asked == 0 or w.held == 0) continue;
            if (!e.shared_open) {
                watch.checked += 1;
                watch.past_budget += @intFromBool(w.held > w.stream.asked);
            } else if (denied) {
                // a chance-open round whose chances are gone: its streams' blocks priced by the rates, within the budget
                watch.checked_unpriced += 1;
                watch.past_budget_unpriced += @intFromBool(w.held > w.stream.asked);
            }
        }
    }
    return watch.orig.?(ptr, windows, out);
}

/// The backend's view of each round (Backend.shared_round), and the chance reads a backend that runs left-out streams'
/// deferred drafts would make, mirrored from cuda_lanes.zig's verifyFn: it reads them only when the round is open.
var told: struct { open: usize = 0, closed: usize = 0, wide_open: usize = 0, lone_open: usize = 0, left_out_reads: usize = 0 } = .{};

fn sharedRound(_: *anyopaque, open: bool) void {
    if (open) told.open += 1 else told.closed += 1;
    const e = hook.engine orelse return;
    if (open and e.live_streams > 4) told.wide_open += 1;
    if (open and e.alone) told.lone_open += 1;
    // the CUDA flush of a left-out stream's deferred draft reads its chances exactly when `open` (leftOutChances)
    if (open and e.live_streams > e.cfg.batch_streams) told.left_out_reads += 1;
}

/// The rounds' stream limit the runs take (null: the model's).
var turns_for_test: ?u32 = null;

const prompts = [_][]const u32{ &.{ 3, 1, 4 }, &.{ 1, 5, 9 }, &.{ 2, 6, 5 }, &.{ 3, 5, 8 }, &.{ 9, 7, 9 }, &.{ 3, 2, 3 } };
const Run = struct { tokens: [6][]u32, record: [32]u8 = undefined, drafted: u64, rounds: u64, calls: usize, sampled_calls: usize, lone_calls: usize, wide_calls: usize = 0, rate: f64 = -1, prior: f64 = 0, plain_overheads: usize = 0 };

/// `count` streams together (stream `sampled_at` sampled), stepped until done, the fact `cut`, the hook's `chance`.
fn run(count: usize, cut: SharedCut, chance: f64, sampled_at: ?usize) !Run {
    return runAt(count, cut, chance, sampled_at, null);
}

/// `run` with context prices and `depth_from` (null: none).
fn runAt(count: usize, cut: SharedCut, chance: f64, sampled_at: ?usize, depth_from: ?u64) !Run {
    const window_ms = [_]f64{ 5.8, 6.6, 7.4, 8.2, 9.0, 9.8, 10.6, 11.4 };
    const nodes = [_]depth.ContextCost{
        .{ .context = 8, .window_ms = &window_ms, .draft_ms = 2.0 },
        .{ .context = 1 << 20, .window_ms = &window_ms, .draft_ms = 2.0 },
    };
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    const pass: depth.DraftPass = .{ .base_ms = 1.0, .stream_ms = 0.5, .drafted_ms = 0.5, .fair_above = 4 };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .draft_pass = pass, .shared_cut = cut, .context_costs = if (depth_from != null) &nodes else &.{}, .depth_from = depth_from orelse 0 }, 16, 15);
    defer cfg.deinit(a);
    cfg.timed = timed_for_test;
    if (turns_for_test) |n| cfg.batch_streams = n;
    if (low_prior_for_test) for (cfg.depth_prior, 0..) |*x, i| {
        x.* = if (i == 0) 0.9 else 0.15;
    };
    try t.expectEqual(cut, cfg.shared_cut);
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var b = target.backend();
    var vt = b.vtable.*;
    vt.probabilities = chances;
    vt.shared_round = sharedRound;
    told = .{};
    watch = .{ .orig = vt.verify };
    vt.verify = verifyWatched;
    b.vtable = &vt;
    var clock: fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, b, clock.clock());
    defer e.deinit();
    hook = .{ .chance = chance, .engine = &e, .deny_after = deny_for_test };
    defer hook.engine = null;
    var streams: [6]lanes.Stream = undefined;
    var log = lanes.events.Log.init(a);
    defer log.deinit();
    e.log = &log;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    const joined = join_for_test != null;
    for (streams[0..count], prompts[0..count], 0..) |*s, p, i| {
        const max_new: u32 = if (i == 1 and short_for_test != null) short_for_test.? else 48;
        s.* = try lanes.Stream.init(a, .{ .id = "s", .prompt = p, .max_new = max_new, .sampling = if (sampled_at == i) .{ .seed = 11 } else null });
        made += 1;
        if (!(joined and i == count - 1)) try e.addStream(s);
    }
    var rate: f64 = -1; // stream 0's first-draft rate, last seen while it decodes
    var steps: usize = 0;
    while (e.activeCount() > 0) {
        try e.step();
        steps += 1;
        if (joined and steps == join_for_test.?) try e.addStream(&streams[count - 1]);
        if (streams[0].depth) |st| rate = st.p[0];
    }
    var out: Run = .{ .tokens = undefined, .drafted = e.drafted, .rounds = e.shared_rounds, .calls = hook.calls, .sampled_calls = hook.sampled_calls, .lone_calls = hook.lone_calls, .wide_calls = hook.wide_calls, .rate = rate, .prior = cfg.depth_prior[0], .plain_overheads = e.rule.plain_overhead.items.len };
    for (&out.tokens, 0..) |*x, i| x.* = try a.dupe(u32, if (i < count) streams[i].emitted() else &.{});
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..log.events.items.len) |i| {
        const line = try log.line(a, i);
        defer a.free(line);
        hash.update(line);
        hash.update("\n");
    }
    out.record = hash.finalResult();
    return out;
}

fn free(r: Run) void {
    for (r.tokens) |x| a.free(x);
}

test "shared cut: off asks no chances; on, every stream's tokens are as without it" {
    var off = try run(4, .off, 0.5, null);
    defer free(off);
    try t.expectEqual(@as(usize, 0), off.calls);
    for ([_]SharedCut{ .cut, .deepen }) |mode| for ([_]f64{ 0.05, 0.5, 0.99 }) |chance| {
        const on = try run(4, mode, chance, null);
        defer free(on);
        try t.expect(on.calls > 0);
        try t.expectEqual(@as(usize, 0), on.lone_calls);
        for (off.tokens, on.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
    };
    _ = &off;
}

test "shared cut: drafts unlikely to land are verified less, likely ones more when deepened" {
    const off = try run(4, .off, 0.5, null);
    defer free(off);
    const low = try run(4, .cut, 0.02, null);
    defer free(low);
    try t.expect(low.drafted < off.drafted);
    const high = try run(4, .deepen, 0.999, null);
    defer free(high);
    const high_cut = try run(4, .cut, 0.999, null);
    defer free(high_cut);
    try t.expect(high.drafted >= high_cut.drafted);
}

test "shared cut: rates learn only from chains that verified the rule's depth (cut chains feed nothing)" {
    const off = try run(4, .off, 0.3, null);
    defer free(off);
    try t.expect(off.rate >= 0 and off.rate != off.prior); // the rates move with what lands
    const cut = try run(4, .cut, 0.3, null);
    defer free(cut);
    try t.expect(cut.rate >= 0 and cut.drafted < off.drafted);
    // every chain was cut short of the rule's depth: nothing fed, the rate is the prior's
    try t.expectEqual(cut.prior, cut.rate);
}

test "shared cut: past depth_from no stream asks chances or drafts deeper than its budget" {
    const off = try runAt(4, .off, 0.99, null, 1);
    defer free(off);
    const deep = try runAt(4, .deepen, 0.99, null, 1);
    defer free(deep);
    try t.expectEqual(@as(usize, 0), deep.calls);
    try t.expectEqual(off.drafted, deep.drafted);
    for (off.tokens, deep.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
    const shallow = try runAt(4, .deepen, 0.99, null, 1 << 20);
    defer free(shallow);
    try t.expect(shallow.calls > 0);
}

test "shared cut: rounds of more than the pass's fair_above streams ask no chances (v0.2's rounds), tokens as off" {
    for ([_]usize{ 5, 6 }) |n| {
        const off = try run(n, .off, 0.99, null);
        defer free(off);
        const on = try run(n, .deepen, 0.99, null);
        defer free(on);
        try t.expectEqual(@as(usize, 0), on.wide_calls);
        for (off.tokens, on.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
    }
}

test "shared cut: five live streams in rounds of four ask no chances (the live count gates, not the round's)" {
    turns_for_test = 4;
    defer turns_for_test = null;
    const on = try run(5, .deepen, 0.99, null);
    defer free(on);
    try t.expectEqual(@as(usize, 0), on.wide_calls);
    turns_for_test = null;
    const off = try run(5, .off, 0.99, null);
    defer free(off);
    for (off.tokens, on.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
}

test "shared cut: open at 2 .. the pass's fair_above streams, never without the priced pass" {
    var shared = try lanes.allocate.extendCosts(a, &.{ .{ .width = 1, .ms = 30.0 }, .{ .width = 8, .ms = 70.0 }, .{ .width = 32, .ms = 160.0 } }, 32);
    defer shared.deinit(a);
    const empty: lanes.table.Table = .{};
    var r: depth.Rule = .{ .gpa = a, .prior = &.{0.8}, .most_drafts = 1, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32, .draft_pass = .{ .base_ms = 3.8, .stream_ms = 4.6, .fair_above = 4 } };
    defer r.deinit();
    try t.expect(!r.sharedOpen(1) and r.sharedOpen(2) and r.sharedOpen(3) and r.sharedOpen(4) and !r.sharedOpen(5) and !r.sharedOpen(8));
    // TF_INK2_DRAFT_FAIR=8 moves the plain-share rule's threshold, not the shared cut's 2 .. 4
    var fair8 = r;
    fair8.draft_pass.?.fair_above = 8;
    try t.expect(fair8.sharedOpen(4) and !fair8.sharedOpen(5) and !fair8.sharedOpen(8));
    var none: depth.Rule = .{ .gpa = a, .prior = &.{0.8}, .most_drafts = 1, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32 };
    defer none.deinit();
    try t.expect(!none.sharedOpen(2));
}

test "shared cut: a sampled stream and a lone stream ask no chances" {
    const mixed = try run(4, .deepen, 0.5, 2);
    defer free(mixed);
    try t.expectEqual(@as(usize, 0), mixed.sampled_calls);
    try t.expectEqual(@as(usize, 0), mixed.lone_calls);
    const lone = try run(1, .deepen, 0.5, null);
    defer free(lone);
    try t.expectEqual(@as(usize, 0), lone.calls);
}

test "shared cut: the fact needs the priced pass, drafts and shared forwards" {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var none = try lanes.Config.init(a, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .drafts = 7, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .shared_cut = .cut }, 16, 15);
    defer none.deinit(a);
    try t.expectEqual(SharedCut.off, none.shared_cut);
}

test "shared cut: a deepened block is verified past its budget only in a chance-priced round (2 -> 1)" {
    low_prior_for_test = true;
    defer low_prior_for_test = false;
    // 2 -> 1: stream 1 ends early; stream 0's next rounds are lone
    short_for_test = 6;
    const lone_off = try run(2, .off, 0.999, null);
    defer free(lone_off);
    const lone = try run(2, .deepen, 0.999, null);
    defer free(lone);
    try t.expect(watch.checked > 0); // lone windows with a held block were seen
    try t.expectEqual(@as(usize, 0), watch.past_budget);
    for (lone_off.tokens, lone.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
    short_for_test = null;
}

test "shared cut: a deepened block is verified past its budget only in a chance-priced round (4 -> 5)" {
    low_prior_for_test = true;
    defer low_prior_for_test = false;
    // a fifth stream joins; rounds of five are v0.2's
    join_for_test = 6;
    defer join_for_test = null;
    const wide_off = try run(5, .off, 0.999, null);
    defer free(wide_off);
    const wide = try run(5, .deepen, 0.999, null);
    defer free(wide);
    try t.expect(watch.checked > 0);
    try t.expectEqual(@as(usize, 0), watch.past_budget);
    for (wide_off.tokens, wide.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
}

test "shared cut: a round whose drafts were all cut after the pass ran is not timed as a plain round" {
    timed_for_test = true;
    defer timed_for_test = false;
    // chances so low that every held draft is cut: the pass ran (for the chances), no draft was verified
    const cut_all = try run(3, .deepen, 0.0001, null);
    defer free(cut_all);
    try t.expect(cut_all.calls > 0);
    try t.expectEqual(@as(usize, 0), cut_all.plain_overheads);
    // without the shared cut the same rounds draft and nothing is cut by chances: timing as before
    const off = try run(3, .off, 0.0001, null);
    defer free(off);
    for (off.tokens, cut_all.tokens) |x, y| try t.expectEqualSlices(u32, x, y);
}

test "shared cut: a deepened block whose chances are gone is planned to its budget in a shared round" {
    // the allocation of a 2-stream shared round (open: 2 .. 4 live) whose chances are gone (as when a joining prompt's
    // flush drops them): each stream holds a deepened block of 7 against a budget of 2; the rates alone would keep more
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    const pass: depth.DraftPass = .{ .base_ms = 1.0, .stream_ms = 0.5, .drafted_ms = 0.5, .fair_above = 4 };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .draft_pass = pass, .shared_cut = .deepen }, 16, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var b = target.backend();
    var vt = b.vtable.*;
    vt.probabilities = chances;
    b.vtable = &vt;
    var clock: fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, b, clock.clock());
    defer e.deinit();
    hook = .{ .chance = 0.999, .engine = &e, .deny_after = 0 }; // every read gives none
    defer hook.engine = null;
    var s1 = try lanes.Stream.init(a, .{ .id = "a", .prompt = &.{ 3, 1, 4 }, .max_new = 48 });
    defer s1.deinit(a);
    var s2 = try lanes.Stream.init(a, .{ .id = "b", .prompt = &.{ 1, 5, 9 }, .max_new = 48 });
    defer s2.deinit(a);
    for ([_]*lanes.Stream{ &s1, &s2 }) |s| {
        s.asked = 2;
        s.deepened = true;
    }
    e.live_streams = 2;
    e.shared_open = true;
    defer e.shared_open = false;
    var plans = [_]lanes.windows.Plan{
        .{ .stream = &s1, .position = 3, .kind = .head, .held = 7 },
        .{ .stream = &s2, .position = 3, .kind = .head, .held = 7 },
    };
    try lanes.windows.allocate(&e, &plans);
    try t.expect(hook.calls > 0); // the chances were asked for, and none came
    for (plans) |p| {
        try t.expect(!p.chance_priced);
        try t.expect(p.count() <= 2);
    }
    // the same round with chances: the deepened blocks are priced by them, not cut to the budget
    hook = .{ .chance = 0.999, .engine = &e };
    var priced = [_]lanes.windows.Plan{
        .{ .stream = &s1, .position = 3, .kind = .head, .held = 7 },
        .{ .stream = &s2, .position = 3, .kind = .head, .held = 7 },
    };
    try lanes.windows.allocate(&e, &priced);
    var most: usize = 0;
    for (priced) |p| {
        try t.expect(p.chance_priced);
        most = @max(most, p.count());
    }
    try t.expect(most > 2);
}

test "shared cut: a new stream's first block feeds the rates when its chance-priced round verifies it whole" {
    // two streams admitted together, chances high: the first shared round verifies each stream's first block whole,
    // and the rates learn from it (the first block's budget is its request's depth)
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    const pass: depth.DraftPass = .{ .base_ms = 1.0, .stream_ms = 0.5, .drafted_ms = 0.5, .fair_above = 4 };
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .draft_pass = pass, .shared_cut = .deepen }, 16, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var b = target.backend();
    var vt = b.vtable.*;
    vt.probabilities = chances;
    b.vtable = &vt;
    var clock: fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, b, clock.clock());
    defer e.deinit();
    hook = .{ .chance = 0.999, .engine = &e };
    defer hook.engine = null;
    var s1 = try lanes.Stream.init(a, .{ .id = "a", .prompt = &.{ 3, 1, 4 }, .max_new = 48 });
    defer s1.deinit(a);
    var s2 = try lanes.Stream.init(a, .{ .id = "b", .prompt = &.{ 1, 5, 9 }, .max_new = 48 });
    defer s2.deinit(a);
    try e.addStream(&s1);
    try e.addStream(&s2);
    try t.expect(s1.asked > 0 and s2.asked > 0); // the first block's budget is kept
    // step until the first shared round has verified both streams
    var steps: usize = 0;
    while (e.shared_rounds == 0 and steps < 8) : (steps += 1) try e.step();
    try t.expect(e.shared_rounds > 0);
    var moved = false;
    for ([_]*lanes.Stream{ &s1, &s2 }) |s| if (s.depth) |st| {
        for (st.p, cfg.depth_prior[0..st.p.len]) |x, prior| moved = moved or x != prior;
    };
    try t.expect(moved); // fed: some rate left the prior
}

test "shared cut: the backend is told a round is open only at 2 .. 4 live streams (no left-out chance reads at 1 or 5+)" {
    // 5 live streams in rounds of 4 (one left out each round), and 2 -> 1 with a peer ending early
    turns_for_test = 4;
    const wide = try run(5, .deepen, 0.99, null);
    turns_for_test = null;
    defer free(wide);
    try t.expect(told.closed > 0);
    try t.expectEqual(@as(usize, 0), told.wide_open);
    try t.expectEqual(@as(usize, 0), told.left_out_reads);
    short_for_test = 6;
    defer short_for_test = null;
    const lone = try run(2, .deepen, 0.99, null);
    defer free(lone);
    try t.expect(told.open > 0 and told.closed > 0); // shared rounds of 2 open, lone rounds after the peer ended closed
    try t.expectEqual(@as(usize, 0), told.lone_open);
}
