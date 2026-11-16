//! Draft depth by expected tokens a millisecond (Python family_depth.py DraftDepth and FamilyRounds._head_depth).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");
pub const pricedBatched = alloc.pricedBatched;

pub const Constants = @import("depth_constants.zig").Constants;

pub const HoldCost = @import("hold_price.zig").HoldCost;
const holdMs = @import("hold_price.zig").holdMs;

/// How a stream's rounds are priced under `Rule.ab_modes` (A/B tooling; `rule` otherwise): by its context
/// (context_costs), by the shallow prices whatever its context (as without context_costs), or never drafted.
pub const Mode = enum { rule, table, plain };

/// A stream's acceptance by depth and probe counters (Python `_depth_state[stream]`).
pub const State = struct {
    p: []f64,
    rounds: u64 = 0,
    plain: ?u64 = null,
    wait: ?u64 = null,
    ms: ?Table = null, // plain_guard: the stream's own round times
    mode: Mode = .rule,
    reach: i64 = 1, // context_costs: the deepest chain a deep stream may draft (one past the most drafts a round kept)

    pub fn deinit(st: *State, gpa: Allocator) void {
        gpa.free(st.p);
        if (st.ms) |*t| t.deinit(gpa);
    }
};

/// What the rule reads of a stream when it decides.
pub const Who = struct {
    state: *?State,
    draft_room: i64,
    finished: bool,
    forced: bool,
    granted: ?u32 = null,
    grant_short: bool = false, // the last grant left some held drafts unverified (the width did not bind)
    gain: ?f64 = null, // the drafts its last grant expected to keep (null: none priced)
    stream: ?*anyopaque = null,
    sampled: bool = false, // keyed draws: host work a row that a greedy round does not have
    context: u64 = 0, // the stream's context length (positions held): what its round costs at depth (ContextCost)
};

/// A lone round's prices at one context length: the window's ms by rows (`window_ms[w]`: w + 1 rows, the head over
/// every row) and the drafter's block (a round's fixed cost from two rows on). A model whose window and draft read the
/// whole context (global attention layers) prices a stream's round at its own depth: `priceAt` interpolates between
/// two nodes by context (attention's part grows linearly with it), and past the ends takes the end's.
pub const ContextCost = struct { context: u64, window_ms: []const f64, draft_ms: f64 };

/// A round of `drafts` drafts (a window of drafts + 1 rows, and the draft that made them) at `context`, from nodes in
/// ascending context; null past the widest window priced.
pub fn priceAt(nodes: []const ContextCost, drafts: i64, context: u64) ?f64 {
    if (nodes.len == 0 or drafts < 0) return null;
    const one = struct {
        fn f(n: ContextCost, d: i64) ?f64 {
            const w: usize = @intCast(d);
            if (w >= n.window_ms.len) return null;
            return n.window_ms[w] + (if (d > 0) n.draft_ms else 0);
        }
    }.f;
    if (context <= nodes[0].context) return one(nodes[0], drafts);
    for (nodes[0 .. nodes.len - 1], nodes[1..]) |a, b| {
        if (context > b.context) continue;
        const x = one(a, drafts) orelse return null;
        const y = one(b, drafts) orelse return null;
        const t = @as(f64, @floatFromInt(context - a.context)) / @as(f64, @floatFromInt(b.context - a.context));
        return x + t * (y - x);
    }
    return one(nodes[nodes.len - 1], drafts);
}

/// Whether a copied continuation is ahead of a stream (the round loop proposes; Python `_copy_proposal`).
pub const Probe = struct {
    ptr: *anyopaque,
    copy: *const fn (ptr: *anyopaque, who: *const Who) bool,
};

/// A drafter's own pass over a shared round's drafting streams (one pass for all of them, as a backend whose
/// `draft_streams` drafts them together): `base_ms` once a round any stream drafts in, `stream_ms` a drafting stream.
/// With it, `budgets` prices the pass (allocate.zig HeadCost pass_ms, first_ms) and lets plain rounds compete: a
/// stream drafts only when its drafts' expected tokens a ms beat the round without them (no `@max(1, ...)`), with a
/// probe after a growing run of plain rounds (plain_guard's), and the pass that ran is no part of the overhead the
/// round teaches. `deferred`: the pass runs in the next round's step (with its verify), so that round's time holds
/// it. `most_streams` (0: any): a shared round of more streams drafts none. Lone rounds are not touched.
/// A drafted round costs more than its rows' forward and the pass besides (its forward's shape rarely repeats, so it
/// may run without a captured graph; more host work): greedy shared rounds teach two overheads, the rounds that
/// verified drafts and those that did not, and the difference is priced on the pass (`drafted_ms` until both are
/// measured). Drafts must beat the round without them by Constants.plain_margin. In a round of more than
/// `fair_above` streams (0: every shared round) no stream's expected share may fall below the plain round's (`fair`).
/// `log`: a line a decision and a
/// line an observed round (diagnostics).
pub const DraftPass = struct { base_ms: f64, stream_ms: f64, drafted_ms: f64 = 0, deferred: bool = false, most_streams: u32 = 0, fair_above: u32 = 0, log: bool = false };

/// A shared round's time past its forward: `ms` without its head steps, `head` the steps' modeled time.
pub const Overhead = struct { streams: u32, ms: f64, head: f64 = 0.0 };

pub const Rule = struct {
    gpa: Allocator,
    k: Constants = .{},
    prior: []const f64,
    most_drafts: u32,
    family_costs: *const Table,
    shared_costs: *const Table,
    mtp_step_ms: f64,
    draft_step_ms: f64 = 0, // a shared round's draft past a stream's first (budgets price it)
    /// Lone rounds priced by the stream's context (ContextCost nodes, ascending): its measured round times are kept as
    /// what a round took past its price there (the host's part), and plain rounds compete with drafted depths.
    context_costs: []const ContextCost = &.{},
    /// context_costs apply past this context: a stream at or below it is priced as without them (lone rounds by the
    /// table, always drafting; shared rounds without a depth), and a deeper one by its context, lone (a plain round
    /// competing) and in shared rounds (`depthMs`, what its round costs past the price here).
    depth_from: u64 = 0,
    context_ms: Table = .{}, // context_costs: what deep streams' rounds took past their prices, by drafts
    /// A/B tooling: each new stream takes the next Mode in turn (rule, table, plain), logged as it starts.
    ab_modes: bool = false,
    ab_next: u64 = 0,
    ab_table_ms: Table = .{}, // ab_modes: the table-priced streams' round times (the others keep times past prices)
    plain_guard: bool,
    node_probabilities: bool,
    granted_depth: bool = false, // node_probabilities: a stream drafts its last grant plus draft_slack, lone ones too
    confidence_floor: ?f64 = null,
    batch_rows: u32,
    batched_heads: bool = false, // one batch drafts every stream of a shared round (its steps: the deepest stream's)
    draft_pass: ?DraftPass = null, // a shared round's drafter pass, priced (budgets)
    pass_next: f64 = 0, // draft_pass: the pass the last budgets asked for (ms, its model)
    pass_ran: f64 = 0, // and the one that ran in the last shared round (headMs)
    plain_overhead: std.ArrayList(Overhead) = .empty, // draft_pass: greedy shared rounds that verified no drafts
    drafted_overhead: std.ArrayList(Overhead) = .empty, // and those that did
    row_cost: @import("row_cost.zig").RowCost = .{}, // what a row costs past shared_costs here, under k.learn_rows
    round_ms: Table = .{}, // a round's wall time by drafts, every stream's: what sampled streams price by
    overhead: std.ArrayList(Overhead) = .empty, // a shared round's ms past its forward, by streams (insertion order)
    greedy_ms: Table = .{}, // round_ms from greedy rounds alone: what greedy streams price by
    shared_rounds: u64 = 0,
    greedy_overhead: std.ArrayList(Overhead) = .empty, // overhead from rounds without a sampled stream

    hold_order: []usize = &.{}, // reused hold ordering, at most batch_rows entries

    pub fn deinit(r: *Rule) void {
        r.gpa.free(r.hold_order);
        r.row_cost.deinit(r.gpa);
        r.round_ms.deinit(r.gpa);
        r.ab_table_ms.deinit(r.gpa);
        r.context_ms.deinit(r.gpa);
        r.overhead.deinit(r.gpa);
        r.greedy_ms.deinit(r.gpa);
        r.greedy_overhead.deinit(r.gpa);
        r.plain_overhead.deinit(r.gpa);
        r.drafted_overhead.deinit(r.gpa);
    }

    pub fn rates(r: *Rule, who: Who) ![]f64 {
        if (who.state.* == null) {
            const n = @min(r.prior.len, @max(1, r.most_drafts));
            who.state.* = .{ .p = try r.gpa.dupe(f64, r.prior[0..n]) };
            if (r.ab_modes) {
                const m: Mode = switch (r.ab_next % 3) {
                    0 => .rule,
                    1 => .table,
                    else => .plain,
                };
                std.log.info("lanes depth a/b: stream {d} prices {s}", .{ r.ab_next, @tagName(m) });
                who.state.*.?.mode = m;
                r.ab_next += 1;
            }
        }
        return who.state.*.?.p;
    }

    fn modeOf(r: *Rule, who: Who) !Mode {
        _ = try r.rates(who);
        return who.state.*.?.mode;
    }

    /// Whether a stream's rounds are priced by its context: context_costs, past depth_from, not a table-priced A/B stream.
    fn byContext(r: *Rule, who: Who) !bool {
        return r.context_costs.len > 0 and who.context > r.depth_from and try r.modeOf(who) == .rule;
    }

    /// What a round of `drafts` drafts costs past the same round at `depth_from` (its window's and its draft's
    /// attention over a deeper context), from context_costs; 0 at or below `depth_from` or without them.
    pub fn depthMs(r: *const Rule, drafts: i64, context: u64) f64 {
        if (r.context_costs.len == 0 or context <= r.depth_from) return 0;
        const at = priceAt(r.context_costs, drafts, context) orelse return 0;
        const base = priceAt(r.context_costs, drafts, r.depth_from) orelse return 0;
        return @max(0, at - base);
    }

    /// depthMs for a stream (0 for a table-priced A/B stream, which prices as if it were shallow).
    pub fn streamDepthMs(r: *Rule, who: Who, drafts: i64) !f64 {
        const mode: Mode = if (who.state.*) |st| st.mode else .rule;
        return if (mode != .table) r.depthMs(drafts, who.context) else 0;
    }

    /// Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves toward whether it was kept.
    pub fn observeDepth(r: *Rule, who: Who, proposed: i64, accepted: i64) !void {
        const p = try r.rates(who);
        const n = @min(@max(proposed, 0), @as(i64, @intCast(p.len)));
        var j: i64 = 0;
        while (j < n) : (j += 1) {
            if (accepted < j) break;
            const hit: f64 = if (accepted > j) 1.0 else 0.0;
            const at: usize = @intCast(j);
            p[at] += r.k.depth_rate * (hit - p[at]);
        }
        if (proposed > 0) who.state.*.?.reach = @max(who.state.*.?.reach, @min(accepted, proposed) + 1);
    }

    /// context_costs: a deep stream drafts at most one more than the most drafts a round of it kept (one at first).
    /// Its estimates start from the drafter's prior, which at depth would buy whole chains of dear drafts before a
    /// stream whose drafts seldom land learns it; a chain that lands lets the next reach one further, and a miss does
    /// not take that back (the estimates decide below it).
    fn ramp(who: Who) i64 {
        return who.state.*.?.reach;
    }

    /// Round times by depth: the stream's own under `plain_guard`, else greedy rounds' for a greedy stream.
    fn costs(r: *Rule, who: ?Who) !*Table {
        const w = who orelse return &r.round_ms;
        if (try r.byContext(w)) return &r.context_ms;
        if (r.ab_modes and try r.modeOf(w) == .table) return &r.ab_table_ms;
        if (!r.plain_guard) return if (w.sampled) &r.round_ms else &r.greedy_ms;
        _ = try r.rates(w);
        const st = &w.state.*.?;
        if (st.ms == null) st.ms = .{};
        return &st.ms.?;
    }

    /// A stream's first round carries the prefill-to-decode switch: its time stays out of the costs. Under
    /// context_costs a round's time is kept as what it took past its price at the stream's context (plain rounds too).
    pub fn observeCost(r: *Rule, drafts: i64, ms: f64, initializing: bool, who: ?Who) !void {
        const priced = if (who) |w| try r.byContext(w) else false;
        if (drafts < 0 or (drafts == 0 and !r.plain_guard and !priced) or initializing) return;
        const table = try r.costs(who);
        const v = if (priced) ms - (priceAt(r.context_costs, drafts, who.?.context) orelse return) else ms;
        try r.blend(table, drafts, v);
        if (table == &r.greedy_ms) try r.blend(&r.round_ms, drafts, v); // a greedy round counts for every stream too
    }

    fn blend(r: *Rule, table: *Table, drafts: i64, ms: f64) !void {
        const before = table.get(drafts);
        try table.put(r.gpa, drafts, if (before) |b| b + r.k.cost_rate * (ms - b) else ms);
    }

    /// Measured round time when there is one, else the load-time forward cost plus head steps. Under context_costs:
    /// the price at the stream's context plus what such rounds took past their price.
    pub fn roundCost(r: *Rule, drafts: i64, who: ?Who) !?f64 {
        if (who) |w| if (try r.byContext(w)) return r.contextCost(drafts, w);
        const table = try r.costs(who);
        if (table.get(drafts)) |v| return v;
        const forward = r.family_costs.get(drafts + 1) orelse return null;
        var modeled = forward + r.mtp_step_ms * @as(f64, @floatFromInt(drafts));
        if (r.plain_guard) {
            // untimed: its model plus the least time a row of a timed round took beyond its model
            var least: ?f64 = null;
            for (table.slots, 0..) |slot, d| {
                const ms = slot orelse continue;
                const fc = r.family_costs.get(@intCast(d + 1)) orelse continue;
                const beyond = (ms - fc - r.mtp_step_ms * @as(f64, @floatFromInt(d))) / @as(f64, @floatFromInt(d + 1));
                least = if (least) |l| (if (beyond < l) beyond else l) else beyond;
            }
            const floor = least orelse 0.0;
            modeled += @as(f64, @floatFromInt(drafts + 1)) * (if (floor > 0.0) floor else 0.0);
        }
        return modeled;
    }

    /// context_costs: the round's price at the stream's context plus what such rounds took past their price; a plain
    /// round not timed yet takes the least any timed depth took past its price (a round's host work).
    fn contextCost(r: *Rule, drafts: i64, who: Who) !?f64 {
        const model = priceAt(r.context_costs, drafts, who.context) orelse return null;
        const table = try r.costs(who);
        if (table.get(drafts)) |x| return model + x;
        if (drafts > 0) return model;
        var least: ?f64 = null;
        for (table.slots) |slot| {
            const x = slot orelse continue;
            least = if (least) |l| @min(l, x) else x;
        }
        return model + @max(0, least orelse 0);
    }

    /// The most expected tokens a ms, probing one depth farther every `depth_probe_every` rounds. With plain_guard or
    /// context_costs a plain round competes (a draft must beat it by plain_margin), with a probe after a run of them.
    pub fn depth(r: *Rule, who: Who) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        if (most <= 0) return 0;
        if (r.confidence_floor != null and !who.sampled and !r.k.floor_rates) return most;
        const p = try r.rates(who);
        if (who.state.*.?.mode == .plain) {
            who.state.*.?.rounds += 1;
            return 0;
        }
        const by_context = try r.byContext(who);
        // past depth_from plain rounds compete by what they took there: until one is timed, a deep stream runs plain
        // (on a machine whose rounds the prices miss by far, an untimed plain round's guess decides badly)
        if (by_context and (try r.costs(who)).get(0) == null) {
            who.state.*.?.rounds += 1;
            return 0;
        }
        if (r.family_costs.empty() and !by_context) {
            const want: i64 = if (p[0] < 0.8) 1 else if (p[0] < 0.9) 2 else 3;
            return @max(1, @min(most, want));
        }
        var best: i64 = 1;
        var best_rate: f64 = -1.0;
        if (r.plain_guard or by_context) {
            if (try r.roundCost(0, who)) |plain| {
                if (plain > 0.0) { // a draft must beat plain by plain_margin: estimates near the line are noise
                    best = 0;
                    best_rate = r.k.plain_margin / plain;
                }
            }
        }
        var expected: f64 = 1.0;
        var run: f64 = 1.0;
        var d: i64 = 1;
        while (d <= most) : (d += 1) {
            const cost = (try r.roundCost(d, who)) orelse break;
            const at: usize = @intCast(d - 1);
            run *= if (at < p.len) p[at] else p[p.len - 1];
            expected += run;
            if (expected / cost > best_rate) {
                best = d;
                best_rate = expected / cost;
            }
        }
        const st = &who.state.*.?;
        st.rounds += 1;
        if (best == 0) return @intFromBool(r.probePlain(st));
        st.plain = 0;
        st.wait = r.k.depth_probe_every;
        if (best < most and st.rounds % r.k.depth_probe_every == 0) best += 1;
        if (by_context) best = @min(best, ramp(who));
        return best;
    }

    /// One draft after `wait` plain rounds in a row; each probe doubles the wait until drafting wins again.
    fn probePlain(r: *Rule, st: *State) bool {
        st.plain = (st.plain orelse 0) + 1;
        const wait = st.wait orelse r.k.depth_probe_every;
        if (st.plain.? < wait) return false;
        st.plain = 0;
        st.wait = @min(2 * wait, r.k.plain_wait_most);
        return true;
    }

    /// Drafts past a grant: none when it stopped short of the held drafts (the width did not bind), else grant_slack.
    fn slack(r: *const Rule, who: Who) i64 {
        return if (who.grant_short) r.k.short_slack else r.k.grant_slack;
    }

    /// The last grant plus its slack (the deepest chain before any); after a grant of none, a probe of draft_slack.
    fn grantDepth(r: *Rule, who: Who) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        const granted = who.granted orelse return most;
        _ = try r.rates(who);
        const st = &who.state.*.?;
        if (granted == 0) return if (!r.plain_guard or r.probePlain(st)) @min(most, r.k.draft_slack) else 0;
        st.plain = 0;
        st.wait = r.k.depth_probe_every;
        return @min(most, @as(i64, granted) + r.slack(who));
    }

    /// Python `_head_depth`: the next round's head drafts for a stream (0 when forced, 1 or none with a copy ahead).
    pub fn headDepth(r: *Rule, who: *const Who, budget: ?i64, probe: Probe) !i64 {
        const d: i64 = if (who.finished) 0 else if (budget) |b| b else if (r.granted_depth)
            try r.grantDepth(who.*)
        else if (r.node_probabilities and !((who.sampled or r.k.floor_rates) and r.confidence_floor != null))
            @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1))
        else
            try r.depth(who.*);
        if (who.forced) return 0;
        if (d != 0 and probe.copy(probe.ptr, who)) return @intFromBool(!r.k.copy_first);
        return d;
    }

    /// A shared hold pads every block to its deepest: depths cut so the drafting streams' blocks fit `rows` (one pass).
    pub fn capPass(out: []i64, rows: u32) void {
        var n: i64 = 0;
        for (out) |d| n += @intFromBool(d > 0);
        if (n == 0) return;
        const most = @max(1, @divFloor(@as(i64, rows), n));
        for (out) |*d| d.* = @min(d.*, most);
    }

    /// Granted holds kept, most expected drafts a row first, while they raise the next round's expected tokens a ms.
    pub fn payHolds(r: *Rule, whos: []const Who, sampled: bool, out: []i64) !void {
        if (whos.len > r.batch_rows) return error.BadHoldStreamCount;
        if (r.hold_order.len == 0) r.hold_order = try r.gpa.alloc(usize, r.batch_rows);
        const order = r.hold_order[0..whos.len];
        var m: usize = 0;
        var probes: usize = 0; // a probe after a grant of none keeps its hold: it measures the stream again
        var deepest: i64 = 0;
        for (whos, out, 0..) |w, d, i| {
            if (d <= 0) continue;
            if (w.granted == @as(?u32, 0)) {
                probes += 1;
                deepest = @max(deepest, d);
            } else {
                order[m] = i;
                m += 1;
            }
        }
        if (m == 0) return;
        const Ctx = struct {
            r: *Rule,
            whos: []const Who,
            fn less(c: @This(), x: usize, y: usize) bool {
                const gx = c.r.gainOf(c.whos[x]) catch 0;
                const gy = c.r.gainOf(c.whos[y]) catch 0;
                return gx * rowsOf(c.whos[y]) > gy * rowsOf(c.whos[x]) or (gx * rowsOf(c.whos[y]) == gy * rowsOf(c.whos[x]) and x < y);
            }
        };
        std.mem.sort(usize, order[0..m], Ctx{ .r = r, .whos = whos }, Ctx.less);
        const fixed = r.fixedMs(whos.len, sampled);
        var rows: f64 = @floatFromInt(whos.len);
        var tokens = rows;
        const plain = r.shared_costs.get(@intCast(whos.len)) orelse return;
        const probe_ms = if (probes > 0) holdMs(r.k.hold_costs, probes, deepest) else 0;
        var best = tokens / (plain + fixed + probe_ms);
        var keep: usize = 0;
        for (order[0..m], 1..) |i, k| {
            rows += rowsOf(whos[i]);
            tokens += try r.gainOf(whos[i]);
            deepest = @max(deepest, out[i]);
            const verify = r.shared_costs.get(@intFromFloat(@ceil(rows))) orelse break;
            const extra = r.row_cost.extraMs(whos.len, sampled) * (rows - @as(f64, @floatFromInt(whos.len)));
            const now = tokens / (verify + (if (extra < 0) @max(extra, plain - verify) else extra) + fixed + holdMs(r.k.hold_costs, k + probes, deepest));
            if (now > best) {
                best = now;
                keep = k;
            }
        }
        for (order[keep..m]) |i| out[i] = 0;
    }

    /// The drafts a granted stream's next round is expected to keep: its last grant's, or its first draft's rate.
    fn gainOf(r: *Rule, w: Who) !f64 {
        if (w.gain) |g| if (w.granted != null) return g;
        return (try r.rates(w))[0];
    }

    fn rowsOf(w: Who) f64 {
        return @floatFromInt(@max(1, w.granted orelse 1));
    }

    /// Head drafts for the streams of a shared round, sized by allocation over their chain chances.
    pub fn budgets(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64) !void {
        try r.budgetsInner(whos, probe, sampled, out);
        const p = r.draft_pass orelse return;
        var n: usize = 0;
        for (out) |d| n += @intFromBool(d > 0);
        const now = if (n > 0) p.base_ms + p.stream_ms * @as(f64, @floatFromInt(n)) else 0;
        r.pass_ran = if (p.deferred) r.pass_next else now;
        r.pass_next = now;
    }

    fn budgetsInner(r: *Rule, whos: []Who, probe: Probe, sampled: bool, out: []i64) !void {
        if (r.draft_pass) |p| if (p.most_streams > 0 and whos.len > p.most_streams) {
            for (whos, out) |*w, *o| o.* = try r.headDepth(w, 0, probe);
            return;
        };
        if (r.confidence_floor != null) {
            for (whos, out) |*w, *o| o.* = try r.headDepth(w, null, probe);
            return;
        }
        if (r.node_probabilities) {
            for (whos, out) |*w, *o| {
                const inner = try r.headDepth(w, null, probe);
                const granted: i64 = if (w.granted) |g| g else r.most_drafts;
                o.* = if (r.granted_depth) inner else try r.headDepth(w, @min(inner, granted + r.k.draft_slack), probe);
            }
            if (r.granted_depth) capPass(out, r.batch_rows);
            if (r.granted_depth and r.k.hold_costs.len > 0) try r.payHolds(whos, sampled, out);
            return;
        }
        const most = try r.gpa.alloc(i64, whos.len);
        defer r.gpa.free(most);
        const probs = try r.gpa.alloc([]f64, whos.len);
        defer r.gpa.free(probs);
        var made: usize = 0;
        defer for (probs[0..made]) |p| r.gpa.free(p);
        for (whos, most, probs) |w, *m, *p| {
            m.* = @min(@as(i64, r.most_drafts), @max(1, w.draft_room - 1));
            if (r.ab_modes and try r.modeOf(w) == .plain) m.* = 0;
            if (try r.byContext(w)) m.* = @min(m.*, ramp(w));
            p.* = try alloc.chainProbabilities(r.gpa, try r.rates(w), @intCast(m.*));
            made += 1;
        }
        // context_costs: what each stream's round costs past the shallow price at each count (allocate.zig
        // HeadCost.depth): a deep stream's rows and draft read its whole context
        const depth_rows = try r.gpa.alloc([]f64, if (r.context_costs.len > 0) whos.len else 0);
        defer r.gpa.free(depth_rows);
        var depth_made: usize = 0;
        defer for (depth_rows[0..depth_made]) |d| r.gpa.free(d);
        for (depth_rows, whos[0..depth_rows.len], most[0..depth_rows.len]) |*d, w, m| {
            d.* = try r.gpa.alloc(f64, @intCast(m + 1));
            depth_made += 1;
            for (d.*, 0..) |*x, c| x.* = try r.streamDepthMs(w, @intCast(c));
        }
        const ones = try r.gpa.alloc(u32, whos.len);
        defer r.gpa.free(ones);
        @memset(ones, 1);
        const pass = r.draft_pass orelse DraftPass{ .base_ms = 0, .stream_ms = 0 };
        // draft_pass: the round without drafts at its own overhead; a drafted one's extra on the pass
        const measured_plain = if (r.draft_pass != null and !sampled) nearestIn(r.plain_overhead.items, whos.len) else null;
        const fixed = if (measured_plain) |o| o.ms else r.fixedMs(whos.len, sampled);
        const drafted = if (r.draft_pass != null and !sampled) nearestIn(r.drafted_overhead.items, whos.len) else null;
        const extra_ms = if (drafted) |d| @max(0, d.ms - fixed) else pass.drafted_ms;
        const head: alloc.HeadCost = .{ .step_ms = r.sharedStepMs(), .batched = r.batched_heads, .row_ms = r.row_cost.extraMs(whos.len, sampled), .least_rows = r.row_cost.leastRows(whos.len, sampled), .pass_ms = pass.base_ms + extra_ms, .first_ms = pass.stream_ms, .margin = if (r.draft_pass != null) r.k.plain_margin else 1.0, .depth = depth_rows };
        const counts = if (r.draft_step_ms != 0)
            try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, r.overheadMs(whos.len, sampled), r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), .{ .step_ms = r.draft_step_ms, .batched = r.batched_heads and pricedBatched(), .free_first = true, .row_ms = r.row_cost.extraMs(whos.len, sampled), .least_rows = head.least_rows, .depth = depth_rows })
        else
            try alloc.allocateDrafts(r.gpa, ones, probs, r.shared_costs, fixed, r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), head);
        defer r.gpa.free(counts);
        if (r.draft_pass) |p| if (p.log) {
            var n: usize = 0;
            var d: u64 = 0;
            var lo: f64 = 1;
            var hi: f64 = 0;
            for (counts, probs) |c, pr| {
                n += @intFromBool(c > 0);
                d += c;
                if (pr.len > 0) {
                    lo = @min(lo, pr[0]);
                    hi = @max(hi, pr[0]);
                }
            }
            const plain = r.shared_costs.get(@intCast(whos.len)) orelse 0;
            std.log.info("lanes pass: {d} streams{s}: plain {d:.1} + overhead {d:.1} ms ({d:.4} tokens a ms); pass {d:.1} + {d:.1} a stream (drafted rounds' extra {d:.1}{s}); first drafts' chances {d:.2}..{d:.2}; {d} streams draft {d}", .{ whos.len, if (sampled) " (sampled)" else "", plain, fixed, @as(f64, @floatFromInt(whos.len)) / (plain + fixed), pass.base_ms, pass.stream_ms, extra_ms, if (drafted == null) ", prior" else "", lo, hi, n, d });
        };
        r.probeShared(whos, most, counts);
        if (r.draft_pass) |p| if (whos.len > p.fair_above) r.fair(probs, counts, .{ .fixed = fixed, .pass = pass.base_ms + extra_ms, .stream = pass.stream_ms, .rows = r.row_cost.capRows(whos.len, sampled, @max(whos.len, r.batch_rows)), .depth = depth_rows });
        if (r.plain_guard or r.draft_pass != null) {
            // no draft where none pays; a probe after a growing run of plain rounds
            for (whos, counts) |w, *c| {
                const st = &w.state.*.?;
                st.rounds += 1;
                if (c.* != 0) {
                    st.plain = 0;
                    st.wait = r.k.depth_probe_every;
                } else c.* = @intFromBool(r.probePlain(st));
            }
            for (whos, most, counts, out) |*w, m, c, *o| o.* = try r.headDepth(w, @min(m, @as(i64, c)), probe);
            return;
        }
        for (whos, most, counts, out) |*w, m, c, *o| o.* = try r.headDepth(w, @min(m, @max(1, @as(i64, c))), probe);
    }

    /// A shared round's price as `fair` reads it: the round's overhead past its forward, the pass (with the drafted
    /// rounds' extra) and its share a drafting stream, the rows at most.
    const Price = struct { fixed: f64, pass: f64, stream: f64, rows: u64, depth: []const []const f64 = &.{} };

    /// The round's ms and each stream's expected tokens under `counts`; null when past the rows or unpriced.
    fn fairRate(r: *const Rule, probs: []const []f64, counts: []const u32, price: Price) ?struct { ms: f64, least: f64, tokens: f64 } {
        var rows: u64 = probs.len;
        var n: usize = 0;
        var least: f64 = std.math.inf(f64);
        var tokens: f64 = 0;
        var depth_ms: f64 = 0;
        if (price.depth.len == probs.len) for (price.depth, counts) |d, c| {
            depth_ms += d[@min(c, d.len - 1)];
        };
        for (probs, counts) |pr, c| {
            rows += c;
            n += @intFromBool(c > 0);
            var e: f64 = 1;
            for (pr[0..@min(c, pr.len)]) |x| e += x;
            least = @min(least, e);
            tokens += e;
        }
        if (rows > price.rows) return null;
        const fwd = r.shared_costs.get(@intCast(rows)) orelse return null;
        const ms = fwd + price.fixed + depth_ms + (if (n > 0) price.pass + price.stream * @as(f64, @floatFromInt(n)) else 0);
        return .{ .ms = ms, .least = least, .tokens = tokens };
    }

    /// draft_pass (Scott's rule: drafting never slower than plain): no stream of a shared round may expect fewer
    /// tokens a ms than in the round without drafts (one token over its time), or the other streams' drafts would
    /// slow it below plain. The allocation's counts when every stream's share holds; else the uniform depth (every
    /// stream drafting it, within the rows) with the most tokens a ms among those where it holds; else no drafts.
    fn fair(r: *const Rule, probs: []const []f64, counts: []u32, price: Price) void {
        if (probs.len == 0) return;
        const zeros: [64]u32 = @splat(0);
        if (probs.len > zeros.len) return;
        const plain = r.fairRate(probs, zeros[0..probs.len], price) orelse return;
        const holds = struct {
            fn f(x: anytype, base: f64) bool {
                return x.least * base >= x.ms;
            }
        }.f;
        var any = false;
        for (counts) |c| any = any or c > 0;
        if (!any) return;
        if (r.fairRate(probs, counts, price)) |x| if (holds(x, plain.ms)) return;
        var deepest: usize = 0;
        for (probs) |pr| deepest = @max(deepest, pr.len);
        var best_d: usize = 0;
        var best: f64 = 0;
        var trial: [64]u32 = undefined;
        for (1..deepest + 1) |d| {
            for (probs, trial[0..probs.len]) |pr, *t| t.* = @intCast(@min(d, pr.len));
            const x = r.fairRate(probs, trial[0..probs.len], price) orelse continue;
            if (!holds(x, plain.ms) or x.tokens / x.ms <= best) continue;
            best = x.tokens / x.ms;
            best_d = d;
        }
        for (probs, counts) |pr, *c| c.* = @intCast(@min(best_d, pr.len));
    }

    // One extra offered row, rotating streams; the normal window allocator still prices and trims it.
    fn probeShared(r: *Rule, whos: []const Who, most: []const i64, counts: []u32) void {
        r.shared_rounds +%= 1;
        const every = r.k.depth_probe_every;
        if (whos.len == 0 or every == 0 or r.shared_rounds % every != 0) return;
        var rows: u64 = whos.len;
        for (counts) |c| rows += c;
        if (rows >= @max(whos.len, r.batch_rows)) return;
        const first: usize = @intCast((r.shared_rounds / every -% 1) % whos.len);
        for (0..whos.len) |offset| {
            const i = (first + offset) % whos.len;
            if (whos[i].finished or whos[i].forced or counts[i] == 0 or counts[i] >= most[i]) continue;
            counts[i] += 1;
            return;
        }
    }

    /// A shared round's ms beyond its forward at this many streams (the nearest measured count, first seen on ties).
    pub fn overheadMs(r: *const Rule, streams: usize, sampled: bool) f64 {
        const o = r.nearest(streams, sampled) orelse return 8.0;
        return o.ms + o.head;
    }

    /// overheadMs without the head steps the measured rounds drafted.
    pub fn fixedMs(r: *const Rule, streams: usize, sampled: bool) f64 {
        const o = r.nearest(streams, sampled) orelse return 8.0;
        return o.ms;
    }

    fn nearest(r: *const Rule, streams: usize, sampled: bool) ?Overhead {
        return nearestIn(if (sampled) r.overhead.items else r.greedy_overhead.items, streams);
    }

    /// Shared pricing can retain a family policy while lone rounds use a measured head step.
    pub fn sharedStepMs(r: *const Rule) f64 {
        return r.k.shared_step_ms orelse r.mtp_step_ms;
    }

    /// The modeled time of a shared round's head steps for the depths it drafted.
    pub fn headMs(r: *const Rule, depths: []const i64) f64 {
        if (r.draft_pass != null) return r.pass_ran; // the drafter's pass, as budgets priced it
        var steps: i64 = 0;
        for (depths) |d| steps = if (r.batched_heads) @max(steps, d) else steps + d;
        return r.sharedStepMs() * @as(f64, @floatFromInt(steps));
    }

    /// `sampled`: the round drew keyed samples, so it counts for sampled streams' overhead alone; `head_ms` its drafts.
    pub fn observeOverhead(r: *Rule, streams: u32, rows: u64, ms: f64, sampled: bool, head_ms: f64) !void {
        return r.observeOverheadAt(streams, rows, ms, sampled, head_ms, 0);
    }

    /// observeOverhead for a round whose streams' depth (`depthMs` at their rows, summed) took `depth_ms` of it: what
    /// the round took past its forward at the shallow prices is its overhead.
    pub fn observeOverheadAt(r: *Rule, streams: u32, rows: u64, ms_all: f64, sampled: bool, head_ms: f64, depth_ms: f64) !void {
        const ms = ms_all - depth_ms;
        const forward = r.shared_costs.get(@intCast(rows)) orelse return;
        if (r.k.learn_rows) try r.row_cost.observe(r.gpa, streams, rows, ms - forward - head_ms, sampled);
        // rows the table underprices are no fixed cost: called fixed, they would make the next draft look cheaper
        const row_ms = r.row_cost.extraMs(streams, sampled);
        // no fit yet: a wide round's excess may be its rows, and adding it to the overhead draws the next round wider
        if (r.k.learn_rows and streams > 1 and row_ms == 0 and rows > streams) return;
        const paid = alloc.rowTerm(r.shared_costs, true, .{ .row_ms = row_ms }, rows, streams);
        const extra = if (ms - forward - paid > 0.0) ms - forward - paid else 0.0;
        const fixed = if (extra - head_ms > 0.0) extra - head_ms else 0.0;
        try r.addOverhead(&r.overhead, streams, fixed, extra - fixed);
        if (!sampled) try r.addOverhead(&r.greedy_overhead, streams, fixed, extra - fixed);
        if (r.draft_pass) |p| if (streams > 1 and !sampled) {
            try r.addOverhead(if (rows > streams) &r.drafted_overhead else &r.plain_overhead, streams, fixed, extra - fixed);
            if (p.log) std.log.info("lanes pass: round of {d} streams, {d} rows: {d:.1} ms, forward {d:.1}, pass {d:.1}, overhead {d:.1} ({s})", .{ streams, rows, ms, forward, head_ms, fixed, if (rows > streams) "drafted" else "plain" });
        };
    }

    fn addOverhead(r: *Rule, list: *std.ArrayList(Overhead), streams: u32, fixed: f64, head: f64) !void {
        for (list.items) |*o| {
            if (o.streams == streams) {
                o.ms = o.ms + 0.2 * (fixed - o.ms);
                o.head = o.head + 0.2 * (head - o.head);
                return;
            }
        }
        try list.append(r.gpa, .{ .streams = streams, .ms = fixed, .head = head });
    }
};

/// The overhead measured at the nearest stream count (first seen on ties).
fn nearestIn(list: []const Overhead, streams: usize) ?Overhead {
    if (list.len == 0) return null;
    var best = list[0];
    for (list[1..]) |o| {
        if (distance(o.streams, streams) < distance(best.streams, streams)) best = o;
    }
    return best;
}

fn distance(a: u32, b: usize) usize {
    const x: usize = a;
    return if (x > b) x - b else b - x;
}

test "a priced drafter pass: streams whose drafts land seldom go plain, a probe after a run of plain rounds" {
    const gpa = std.testing.allocator;
    var shared = try alloc.extendCosts(gpa, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer shared.deinit(gpa);
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    for ([_]bool{ false, true }) |priced| {
        var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.3, 0.1, 0.05 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32, .batched_heads = true, .draft_pass = if (priced) .{ .base_ms = 8.0, .stream_ms = 4.6, .deferred = true } else null };
        defer rule.deinit();
        var states: [8]?State = @splat(null);
        defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
        var whos: [8]Who = undefined;
        for (&whos, &states) |*w, *st| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false };
        var out: [8]i64 = undefined;
        var drafted: usize = 0;
        for (0..8) |_| {
            try rule.budgets(&whos, probe, false, &out);
            for (out) |d| drafted += @intFromBool(d > 0);
        }
        if (!priced) {
            try std.testing.expectEqual(@as(usize, 64), drafted); // every stream drafts at least one, every round
            try std.testing.expectEqual(@as(f64, 0), rule.headMs(&out));
        } else {
            // plain rounds but the probe after the run of plain rounds (its 8th round): one draft a stream then
            try std.testing.expectEqual(@as(usize, 8), drafted);
            try std.testing.expectApproxEqAbs(@as(f64, 0), rule.headMs(&out), 1e-12); // deferred: the 7th round's pass
            try std.testing.expectApproxEqAbs(@as(f64, 8.0 + 8 * 4.6), rule.pass_next, 1e-9);
            try rule.budgets(&whos, probe, false, &out);
            try std.testing.expectApproxEqAbs(@as(f64, 8.0 + 8 * 4.6), rule.headMs(&out), 1e-9);
        }
    }
    // a cap on drafting streams: a round of more streams drafts none
    var capped: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.9, 0.9 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32, .draft_pass = .{ .base_ms = 0, .stream_ms = 0, .most_streams = 4 } };
    defer capped.deinit();
    var cstates: [5]?State = @splat(null);
    defer for (&cstates) |*s| if (s.*) |*x| x.deinit(gpa);
    var cwhos: [5]Who = undefined;
    for (&cwhos, &cstates) |*w, *st| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false };
    var cout: [5]i64 = undefined;
    try capped.budgets(&cwhos, probe, false, &cout);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0 }, &cout);
    try capped.budgets(cwhos[0..4], probe, false, cout[0..4]);
    for (cout[0..4]) |d| try std.testing.expect(d >= 1);
}

test "a priced drafter pass: drafted rounds' own overhead past the plain rounds' goes on the pass" {
    const gpa = std.testing.allocator;
    var shared = try alloc.extendCosts(gpa, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer shared.deinit(gpa);
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.84, 0.81, 0.81 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32, .batched_heads = true, .draft_pass = .{ .base_ms = 3.8, .stream_ms = 4.6 } };
    defer rule.deinit();
    var states: [8]?State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    var whos: [8]Who = undefined;
    for (&whos, &states) |*w, *st| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false };
    var out: [8]i64 = undefined;
    try rule.budgets(&whos, probe, false, &out);
    for (out) |d| try std.testing.expect(d > 0); // likely drafts, nothing measured: every stream drafts
    try rule.observeOverhead(8, 8, 77.0 + 20.0, false, 0); // a plain round: 20 ms past its forward
    try rule.observeOverhead(8, 32, 161.0 + 220.0, false, 3.8 + 8 * 4.6); // a drafted round: 220 past its forward
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), nearestIn(rule.plain_overhead.items, 8).?.ms, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 220.0 - 3.8 - 8 * 4.6), nearestIn(rule.drafted_overhead.items, 8).?.ms, 1e-9);
    try rule.budgets(&whos, probe, false, &out);
    for (out) |d| try std.testing.expectEqual(@as(i64, 0), d); // drafted rounds cost too much here: plain
}

test "a priced drafter pass: no stream's share falls below plain's, so a stream whose drafts land seldom keeps the round plain" {
    const gpa = std.testing.allocator;
    var shared = try alloc.extendCosts(gpa, &.{ .{ .width = 1, .ms = 34.0 }, .{ .width = 8, .ms = 77.0 }, .{ .width = 32, .ms = 161.0 } }, 32);
    defer shared.deinit(gpa);
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.95, 0.95, 0.95 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 32, .batched_heads = true, .draft_pass = .{ .base_ms = 3.8, .stream_ms = 4.6 } };
    defer rule.deinit();
    var states: [2]?State = @splat(null);
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    var whos: [2]Who = undefined;
    for (&whos, &states) |*w, *st| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false };
    var out: [2]i64 = undefined;
    try rule.budgets(&whos, probe, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 3, 3 }, &out); // both land often: both draft
    @memcpy(states[1].?.p, &[_]f64{ 0.3, 0.1, 0.05 }); // the second stream's drafts now land seldom
    try rule.budgets(&whos, probe, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out); // the first's drafts would slow it below plain
}

test "a depth's acceptance estimate moves by the configured weight of the newest round" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .k = .{ .depth_rate = 0.08 }, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try rule.observeDepth(who, 2, 1); // the first draft landed, the second did not
    try std.testing.expectApproxEqAbs(@as(f64, 0.54), st.?.p[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.46), st.?.p[1], 1e-12);
}

test "depth picks the most tokens a ms and probes deeper" {
    const gpa = std.testing.allocator;
    var fc: Table = .{};
    defer fc.deinit(gpa);
    for (1..5) |w| try fc.put(gpa, @intCast(w), 5.0 + @as(f64, @floatFromInt(w)));
    const shared: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.9, 0.9 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 3), try rule.depth(who));
}

test "under a confidence floor a greedy stream drafts the most, and a sampled one keeps its rates' depth" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5, 0.5 }, .most_drafts = 3, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .confidence_floor = 0.5, .batch_rows = 8 };
    defer rule.deinit();
    var states: [2]?State = .{ null, null };
    defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
    const greedy: Who = .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false };
    const sampled: Who = .{ .state = &states[1], .draft_room = 100, .finished = false, .forced = false, .sampled = true };
    try std.testing.expectEqual(@as(i64, 3), try rule.headDepth(&greedy, null, probe));
    try std.testing.expectEqual(@as(i64, 1), try rule.headDepth(&sampled, null, probe)); // a first rate under 0.8: one draft
}

test "a shared round's drafts come out the same whichever depths its overhead was measured at" {
    const gpa = std.testing.allocator;
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..17) |w| try shared.put(gpa, @intCast(w), 10.0 + 2.0 * @as(f64, @floatFromInt(w)));
    const empty: Table = .{};
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = No.copy };
    var got: [2][2]i64 = undefined;
    for ([_][2]i64{ .{ 1, 1 }, .{ 3, 3 } }, &got) |measured, *out| {
        var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.8, 0.7, 0.6 }, .most_drafts = 4, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 3.0, .plain_guard = false, .node_probabilities = false, .batch_rows = 16 };
        defer rule.deinit();
        // two streams' round: their pending rows and the drafts verified, 3 ms of its own, then each drafted step
        const rows: u64 = @intCast(2 + measured[0] + measured[1]);
        const head = rule.headMs(&measured);
        try rule.observeOverhead(2, rows, shared.get(@intCast(rows)).? + 3.0 + head, false, head);
        var states: [2]?State = .{ null, null };
        defer for (&states) |*s| if (s.*) |*x| x.deinit(gpa);
        var whos = [_]Who{ .{ .state = &states[0], .draft_room = 100, .finished = false, .forced = false }, .{ .state = &states[1], .draft_room = 100, .finished = false, .forced = false } };
        try rule.budgets(&whos, probe, false, out);
    }
    try std.testing.expectEqualSlices(i64, &got[0], &got[1]);
}

test "granted depths in a shared round fit one hold pass: every block is as deep as the deepest" {
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = std.testing.allocator, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    var states: [8]?State = @splat(null);
    defer for (&states) |*st| if (st.*) |*x| x.deinit(std.testing.allocator);
    var whos: [8]Who = undefined;
    for (&whos, &states, 0..) |*w, *st, i| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false, .granted = @intCast(i) };
    const No = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    var out: [8]i64 = undefined;
    try rule.budgets(&whos, .{ .ptr = &unused, .copy = No.copy }, false, &out);
    // a grant of none probes two; grants 1..7 that took every held draft, plus one; cut to 32 rows over eight blocks
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 3, 4, 4, 4, 4, 4 }, &out);
    try rule.budgets(whos[0..2], .{ .ptr = &unused, .copy = No.copy }, false, out[0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, out[0..2]); // two blocks fit whole
    for (whos[0..2]) |*w| w.grant_short = true;
    try rule.budgets(whos[0..2], .{ .ptr = &unused, .copy = No.copy }, false, out[0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 2, 1 }, out[0..2]); // a grant short of the held drafts: no slack
}

test "a copy ahead: one head draft beside it, or none when the head's drafts cost a pass of their own" {
    const empty: Table = .{};
    const Yes = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return true;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = Yes.copy };
    var st: ?State = null;
    defer if (st) |*x| x.deinit(std.testing.allocator);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .granted = 3 };
    for ([_]bool{ false, true }) |first| {
        var rule: Rule = .{ .gpa = std.testing.allocator, .k = .{ .copy_first = first }, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
        defer rule.deinit();
        try std.testing.expectEqual(@as(i64, @intFromBool(!first)), try rule.headDepth(&who, null, probe));
        var out: [1]i64 = undefined;
        var whos = [_]Who{who};
        try rule.budgets(&whos, probe, false, &out);
        try std.testing.expectEqual(@as(i64, @intFromBool(!first)), out[0]);
    }
}

test "a granted hold stays only while its stream's expected drafts pay for the verify rows and the hold pass" {
    const gpa = std.testing.allocator;
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..33) |w| try shared.put(gpa, @intCast(w), 10.0 + @as(f64, @floatFromInt(w)));
    const empty: Table = .{};
    const cheap = [_]HoldCost{ .{ .streams = 1, .depth = 1, .ms = 1.0 }, .{ .streams = 8, .depth = 1, .ms = 4.5 } };
    var states: [4]?State = @splat(null);
    defer for (&states) |*st| if (st.*) |*x| x.deinit(gpa);
    const granted = [_]?u32{ 2, 2, 0, 1 };
    const gains = [_]?f64{ 1.8, 0.2, null, 0.9 };
    var whos: [4]Who = undefined;
    for (&whos, &states, granted, gains) |*w, *st, g, x| w.* = .{ .state = st, .draft_room = 100, .finished = false, .forced = false, .granted = g, .gain = x };
    var rule: Rule = .{ .gpa = gpa, .k = .{ .hold_costs = &cheap }, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &shared, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    // 8 ms overhead: plain 4 / 22; stream 0 then 3 raise the rate (5.8 / 25, 6.7 / 26.5), stream 1 would lower it
    var out = [_]i64{ 4, 4, 0, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 4, 0, 0, 3 }, &out);
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    rule.gpa = failing.allocator();
    out = .{ 4, 4, 0, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectEqualSlices(i64, &.{ 4, 0, 0, 3 }, &out);
    rule.gpa = gpa;
    const dear = [_]HoldCost{.{ .streams = 1, .depth = 1, .ms = 20.0 }};
    rule.k.hold_costs = &dear;
    out = .{ 4, 4, 0, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, &out); // no gain pays a 20 ms pass
    whos[2].granted = 0;
    out = .{ 4, 4, 2, 3 };
    try rule.payHolds(&whos, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2, 0 }, &out); // a probe after a grant of none keeps its hold
}

test {
    _ = @import("shared_step_test.zig");
    _ = @import("depth_shared_test.zig");
}

test "context prices: interpolated by context, the ends' past them, none past the widest window" {
    const nodes = [_]ContextCost{
        .{ .context = 1000, .window_ms = &.{ 30, 40, 50 }, .draft_ms = 8 },
        .{ .context = 3000, .window_ms = &.{ 50, 80, 110 }, .draft_ms = 28 },
    };
    try std.testing.expectApproxEqAbs(@as(f64, 30), priceAt(&nodes, 0, 10).?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 48), priceAt(&nodes, 1, 1000).?, 1e-12); // window 2 rows + draft
    try std.testing.expectApproxEqAbs(@as(f64, 40), priceAt(&nodes, 0, 2000).?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (58.0 + 138.0) / 2.0), priceAt(&nodes, 2, 2000).?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 138), priceAt(&nodes, 2, 1 << 30).?, 1e-12);
    try std.testing.expectEqual(@as(?f64, null), priceAt(&nodes, 3, 2000));
    try std.testing.expectEqual(@as(?f64, null), priceAt(&.{}, 0, 2000));
}

test "context prices: a stream at depth_from drafts as the table rule does; a deep one with seldom-landing drafts goes plain and probes" {
    const gpa = std.testing.allocator;
    // the window's ms by rows and the draft at a shallow and a deep context (the deep round's draft and window grow)
    const shallow_ms = [_]f64{ 34.1, 42.4, 46.9, 51.7 };
    const nodes = [_]ContextCost{
        .{ .context = 600, .window_ms = &shallow_ms, .draft_ms = 8.4 },
        .{ .context = 524_000, .window_ms = &.{ 62.5, 90.0, 105.0, 120.0 }, .draft_ms = 39.9 },
    };
    var fc: Table = .{};
    defer fc.deinit(gpa);
    for (shallow_ms, 0..) |ms, i| try fc.put(gpa, @intCast(i + 1), ms + if (i > 0) @as(f64, 8.4) else 0);
    const shared: Table = .{};
    for ([_][]const f64{ &.{ 0.84, 0.81, 0.81 }, &.{ 0.7, 0.6, 0.5 } }) |prior| {
        var table: Rule = .{ .gpa = gpa, .prior = prior, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
        defer table.deinit();
        var by_context: Rule = table; // copied before either has measured anything (nothing shared)
        by_context.context_costs = &nodes;
        by_context.depth_from = 600;
        defer by_context.deinit();
        var s1: ?State = null;
        defer if (s1) |*s| s.deinit(gpa);
        var s2: ?State = null;
        defer if (s2) |*s| s.deinit(gpa);
        const a: Who = .{ .state = &s1, .draft_room = 100, .finished = false, .forced = false, .context = 600 };
        const b: Who = .{ .state = &s2, .draft_room = 100, .finished = false, .forced = false, .context = 600 };
        // the same depth as the table rule, untimed and after the same timed rounds (the host's 2 ms a round)
        for (0..20) |i| {
            const da = try table.depth(a);
            try std.testing.expectEqual(da, try by_context.depth(b));
            const ms = (try table.roundCost(da, a)).? + 2.0 + @as(f64, @floatFromInt(i % 3));
            try table.observeCost(da, ms, false, a);
            try by_context.observeCost(da, ms, false, b);
        }
    }
    // shallow, drafts that land seldom: the table rule drafts one at least; by context plain competes (1.3 tokens in
    // 50.8 ms against 1 in 34.1) past depth_from, and at it drafts as the table rule does
    {
        var table: Rule = .{ .gpa = gpa, .prior = &.{ 0.3, 0.1, 0.05 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
        defer table.deinit();
        var by_context: Rule = table;
        by_context.context_costs = &nodes;
        defer by_context.deinit();
        var s1: ?State = null;
        defer if (s1) |*s| s.deinit(gpa);
        var s2: ?State = null;
        defer if (s2) |*s| s.deinit(gpa);
        var s3: ?State = null;
        defer if (s3) |*s| s.deinit(gpa);
        try std.testing.expectEqual(@as(i64, 1), try table.depth(.{ .state = &s1, .draft_room = 100, .finished = false, .forced = false, .context = 600 }));
        try std.testing.expectEqual(@as(i64, 0), try by_context.depth(.{ .state = &s2, .draft_room = 100, .finished = false, .forced = false, .context = 601 }));
        by_context.depth_from = 600;
        try std.testing.expectEqual(@as(i64, 1), try by_context.depth(.{ .state = &s3, .draft_room = 100, .finished = false, .forced = false, .context = 600 }));
    }
    // at 524,000 with drafts that land seldom: plain rounds, one draft after 8 of them, then after 16
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.3, 0.1, 0.05 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = &nodes };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    var deep: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .context = 524_000 };
    try std.testing.expectEqual(@as(i64, 0), try rule.depth(deep)); // plain until a plain round is timed there
    try rule.observeCost(0, 62.5, false, deep);
    st.?.rounds = 0;
    var drafted: [32]i64 = undefined;
    for (&drafted) |*d| d.* = try rule.depth(deep);
    for (drafted, 1..) |d, i| try std.testing.expectEqual(@as(i64, @intFromBool(i == 8 or i == 24)), d);
    // the same stream shallow again, with likely drafts: it drafts
    @memcpy(st.?.p, &[_]f64{ 0.84, 0.81, 0.81 });
    deep.context = 600;
    rule.depth_from = 600;
    try std.testing.expect(try rule.depth(deep) >= 2);
    // and deep with likely drafts at these prices: three drafts' 3.2 tokens in 160 ms beat plain's 1 in 62.5 ms
    deep.context = 524_000;
    try std.testing.expect(try rule.depth(deep) >= 1);
}

test "context prices: a round's time is kept past its price; untimed plain takes the least excess" {
    const gpa = std.testing.allocator;
    const nodes = [_]ContextCost{
        .{ .context = 0, .window_ms = &.{ 30, 40 }, .draft_ms = 10 },
        .{ .context = 100_000, .window_ms = &.{ 60, 90 }, .draft_ms = 30 },
    };
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{0.5}, .most_drafts = 1, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .context_costs = &nodes };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    var who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false, .context = 1 };
    try std.testing.expectApproxEqAbs(@as(f64, 30), (try rule.roundCost(0, who)).?, 1e-2); // nothing timed
    try rule.observeCost(1, 53.0, false, who); // 3 ms past 40 + 10
    try std.testing.expectApproxEqAbs(@as(f64, 3), rule.context_ms.get(1).?, 1e-2);
    who.context = 100_000; // the stream deeper: its price there plus the same 3 ms
    try std.testing.expectApproxEqAbs(@as(f64, 123), (try rule.roundCost(1, who)).?, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f64, 63), (try rule.roundCost(0, who)).?, 1e-2); // plain untimed: least excess
    try rule.observeCost(0, 61.0, false, who); // a plain round timed: 1 ms past its price
    try std.testing.expectApproxEqAbs(@as(f64, 61), (try rule.roundCost(0, who)).?, 1e-2);
    try rule.observeCost(1, 99.0, true, who); // a first round (prefill to decode): not kept
    try std.testing.expectApproxEqAbs(@as(f64, 3), rule.context_ms.get(1).?, 1e-2);
}
