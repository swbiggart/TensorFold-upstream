//! Ranks over processes: rank 0's `Leader` broadcasts each step's header (kind, slot, rows, first position) over the
//! ranks' NCCL group before running it; every other rank `follow`s, running the same forward (its embedded rows come
//! by the gather of the ranks' table rows inside the forward, cuda_embed.zig). A header can ask for the head over
//! some of the forward's rows after it, or over rows of the last forward alone (`head`): every rank unembeds its
//! share of the vocabulary (cuda_engine.zig `headRows`). Rank 0 alone samples: nothing about tokens or sampling
//! crosses. At W = 1 the leader just runs its steps.
//!
//! Rank 0 does everything in a step that can fail on its own (the rows, the slot's state, the positions) before the
//! header goes out, so such a failure is the request's alone and no rank has moved. A failure after the header, or
//! in sending it, leaves the other ranks waiting in a collective rank 0 will not reach: every rank stops then.
//!
//! With the ranks' liveness lines open (cuda.tp_watch, served ranks), a rank that dies stops the others, and while no
//! stream is left rank 0 sends `idle`: the other ranks then sleep on their lines (neither CUDA nor NCCL spins) until
//! rank 0 wakes them before its next step. Followers start asleep.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const scratch = @import("cuda_scratch.zig");

/// `drafts`: several slots' keeps and drafter blocks in one pass (`Leader.draftMany`): segment i is (slot, rows to
/// keep, the anchor token in pos0). `save` .. `capture`: the prompt cache's copies (cuda_snap.zig). `mark` / `rewind`:
/// tooling (tf-inkling2-decode --exp deep), a slot's convolution states and position kept aside and put back.
/// `synth` / `synth_check`: tooling (--exp deep --fill synth), a slot's positions 0 .. pos0 - 1 written with synthetic
/// values (cuda_synth.zig), and the global layers' rows read back against them (each rank logs its own).
/// `pages`: a slot's stream opened on the KV page pool (cuda_pages.zig): rank 0's page ids for it (the slot's table
/// broadcast after the header; `rows` of them name its pages) on every rank.
pub const Kind = enum(u32) { forward = 1, end = 2, idle = 3, reserve = 4, head = 5, keep = 6, draft = 7, drafts = 8, save = 9, restore = 10, arena = 11, move = 12, capture = 13, mark = 14, rewind = 15, synth = 16, synth_check = 17, pages = 18 };

/// Rows lo .. lo + n of a forward to turn into logits.
pub const Rows = struct {
    lo: usize,
    n: usize,
};

/// TF_INK2_GRAPH (rank 0): `0` runs every forward eagerly; anything else (or unset) runs decode steps and verify
/// windows as captured CUDA graphs (`Leader.graphOk`). The timing switches that wait on the device inside a step
/// (TF_INK2_TIME_EXCHANGE, TF_INK2_CK_PROFILE) run eagerly.
fn graphs() bool {
    for ([_][*:0]const u8{ "TF_INK2_TIME_EXCHANGE", "TF_INK2_CK_PROFILE" }) |n| if (std.c.getenv(n)) |v| if (v[0] == '1') return false;
    const v = std.c.getenv("TF_INK2_GRAPH") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

/// TF_INK2_GRAPH_SEGS (rank 0): `1` runs only one stream's forwards as graphs (several streams' rows eagerly, as
/// stage 2b did; for A/B); anything else (or unset) runs several streams' too, on a key's second sighting.
fn multiGraphs() bool {
    const v = std.c.getenv("TF_INK2_GRAPH_SEGS") orelse return true;
    return !(v[0] == '1' and v[1] == 0);
}

/// TF_INK2_SERVE_FAULT (tests): `slot:N` fails rank 0's N-th forward step at its slot, before its header (as a failed
/// state allocation would); `step:N` fails it just after its header.
const Fault = struct {
    kind: enum { none, slot, step } = .none,
    at: u64 = 0,
    var steps: u64 = 0;

    fn read() Fault {
        const v = std.mem.span(std.c.getenv("TF_INK2_SERVE_FAULT") orelse return .{});
        const colon = std.mem.indexOfScalar(u8, v, ':') orelse return .{};
        const at = std.fmt.parseInt(u64, v[colon + 1 ..], 10) catch return .{};
        const name = v[0..colon];
        return if (std.mem.eql(u8, name, "slot")) .{ .kind = .slot, .at = at } else if (std.mem.eql(u8, name, "step")) .{ .kind = .step, .at = at } else .{};
    }

    /// This step's induced fault, if any (counts rank 0's forward steps).
    fn next() @FieldType(Fault, "kind") {
        const f = read();
        if (f.kind == .none) return .none;
        steps += 1;
        if (steps != f.at) return .none;
        std.log.warn("inkling2 rank 0: TF_INK2_SERVE_FAULT fails step {d} ({s} its header)", .{ steps, if (f.kind == .slot) "before" else "after" });
        return f.kind;
    }
};

pub const Leader = struct {
    e: *engine.Engine,
    /// Tooling: decode steps run eagerly whatever TF_INK2_GRAPH says (graph-against-eager timing in one process).
    eager: bool = false,

    /// The shared embedding (cuda_embed.zig): an eager forward of at most `graph_rows` rows carries its ids in its
    /// header (`token` 1), so the other ranks need no broadcast of them (cuda_engine.zig `tokens`, `body`).
    fn idsInHeader(x: Leader, h: *engine.Header, ids: []const u32) !void {
        x.e.ids_in_header = false;
        if (x.e.shared == null or ids.len == 0 or ids.len > engine.graph_rows) return;
        try h.setTokens(ids);
        h.token = 1;
        x.e.ids_in_header = true;
    }

    /// A failure in sending a header stops every rank (whether it went out is not known).
    pub fn send(x: Leader, h: engine.Header) !void {
        if (x.e.asleep) {
            cuda.tp_watch.wake();
            x.e.asleep = false;
        }
        x.e.post(h) catch |err| return x.stop(err);
    }

    /// W > 1: a step failed after its header went out, so the other ranks wait in a collective this rank will not
    /// reach. Every rank stops: the others hear this one's error on their lines and exit. W = 1: the error returns.
    pub fn stop(x: Leader, err: anyerror) anyerror {
        if (x.e.o.world < 2 or x.e.o.group == null) return err; // W = 1, or one rank alone (--solo)
        std.log.err("inkling2 rank 0: a step failed after its header went out ({s}); every rank stops", .{@errorName(err)});
        cuda.tp_watch.fatal(@errorName(err));
        std.c._exit(1);
    }

    /// Whether a forward of `segs` (R rows) with the head over rows `hr` runs as a captured graph (`graphStep`), rank
    /// 0's choice (the header carries it): TF_INK2_GRAPH not 0, no forced-eager tooling (`eager`) or profiled pass
    /// (its marks are events), every segment past its prompt with at most 8 rows (a decode step or a verify window),
    /// at most `graph_rows` in all, and the head over every row. One segment always; several (one forward of several
    /// streams' rows, keyed by their slots and rows in order) from the key's second sighting (`Engine.graphWanted`;
    /// the first runs eagerly), unless TF_INK2_GRAPH_SEGS=1. Called once a forward (it counts the sighting). With the
    /// drafter its taps and context writes are in the graph.
    pub fn graphOk(x: Leader, segs: []const engine.Segment, R: usize, hr: Rows) bool {
        if (!graphs() or x.eager or x.e.prof_on or segs.len == 0 or R > engine.graph_rows) return false;
        if (hr.lo != 0 or hr.n != R) return false;
        for (segs) |g| if (g.pos0 == 0 or g.rows > engine.graph_seg_rows) return false;
        if (segs.len == 1) return true;
        if (!multiGraphs()) return false;
        return x.e.graphWanted(x.e.keyOf(segs) catch return false);
    }

    /// Before a forward's header: the op scratch holds it (cuda_scratch.zig need; a captured step's attention spans
    /// the slot's capacity), else error.ScratchShort fails the request and no rank moves.
    fn fits(x: Leader, segs: []const engine.Segment, graph: bool) !void {
        try scratch.fitsSegs(&x.e.scratch, &x.e.k, &x.e.w, segs, if (graph) x.e.capacity() else null);
    }

    /// `ids` (R rows) of slot `slot` from position pos0, on every rank; then, if asked, the head over rows `logits`
    /// of it (rank 0's softcapped logits: `e.logits()`). A decode step or a verify window (past the prompt, at most
    /// 8 rows, the head over all of them) runs as a captured graph (`graphOk`).
    pub fn forward(x: Leader, slot: u32, pos0: usize, ids: []const u32, logits: ?Rows) !void {
        const fault = Fault.next();
        if (fault == .slot) return error.OutOfMemory;
        try x.toolPages(slot, pos0);
        _ = try x.e.ready(slot, pos0, ids.len);
        const hr = logits orelse Rows{ .lo = 0, .n = 0 };
        if (hr.n > engine.head_rows or hr.lo + hr.n > ids.len) return error.BadRows;
        const seg = [1]engine.Segment{.{ .slot = slot, .pos0 = pos0, .rows = ids.len }};
        const graph = x.graphOk(&seg, ids.len, hr);
        try x.fits(&seg, graph);
        if (graph) return x.graphForward(&seg, ids, fault);
        var h: engine.Header = .{ .kind = @intFromEnum(Kind.forward), .slot = slot, .rows = @intCast(ids.len), .pos0 = pos0, .head_lo = @intCast(hr.lo), .head_n = @intCast(hr.n) };
        h.pad[1] = @intFromBool(x.e.prof_on);
        try x.idsInHeader(&h, ids);
        try x.send(h);
        if (fault == .step) return x.stop(error.InducedFault);
        x.e.forward(slot, pos0, ids.len, ids) catch |err| return x.stop(err);
        if (hr.n > 0) x.e.headRows(hr.lo, hr.n) catch |err| return x.stop(err);
    }

    /// A forward of `segs` with the head over all its rows as a captured graph on every rank: the segments and the
    /// token ids go out in the header (the graph reads them on the device).
    fn graphForward(x: Leader, segs: []const engine.Segment, ids: []const u32, fault: @FieldType(Fault, "kind")) !void {
        var fs: [engine.max_segs]@import("cuda_forward.zig").Seg = undefined;
        const R = try x.e.graphReady(segs, &fs);
        if (ids.len != R) return error.BadRows;
        var h: engine.Header = .{ .kind = @intFromEnum(Kind.forward), .rows = @intCast(R), .head_lo = 0, .head_n = @intCast(R), .graph = 1 };
        try h.setSegs(segs);
        try h.setTokens(ids);
        try x.send(h);
        if (fault == .step) return x.stop(error.InducedFault);
        x.e.graphStep(segs, ids) catch |err| return x.stop(err);
    }

    /// Several streams' rows in one forward (batched decode, cuda_engine.zig forwardMany: segment i is segs[i].rows
    /// rows of slot segs[i].slot from segs[i].pos0, `ids` in segment order), on every rank; then, if asked, the head
    /// over rows `logits` of it. One segment is `forward`; several run as a captured graph when `graphOk` says so,
    /// else eagerly.
    pub fn forwardMany(x: Leader, segs: []const engine.Segment, ids: []const u32, logits: ?Rows) !void {
        if (segs.len == 1) return x.forward(segs[0].slot, segs[0].pos0, ids, logits);
        const fault = Fault.next();
        if (fault == .slot) return error.OutOfMemory;
        for (segs) |g| try x.toolPages(g.slot, g.pos0);
        var fs: [engine.max_segs]@import("cuda_forward.zig").Seg = undefined;
        const R = try x.e.readyMany(segs, &fs);
        if (ids.len != R) return error.BadRows;
        const hr = logits orelse Rows{ .lo = 0, .n = 0 };
        if (hr.n > engine.head_rows or hr.lo + hr.n > R) return error.BadRows;
        const graph = x.graphOk(segs, R, hr);
        try x.fits(segs, graph);
        if (graph) return x.graphForward(segs, ids, fault);
        var h: engine.Header = .{ .kind = @intFromEnum(Kind.forward), .rows = @intCast(R), .head_lo = @intCast(hr.lo), .head_n = @intCast(hr.n) };
        try h.setSegs(segs);
        h.pad[1] = @intFromBool(x.e.prof_on);
        try x.idsInHeader(&h, ids);
        try x.send(h);
        if (fault == .step) return x.stop(error.InducedFault);
        x.e.forwardMany(segs, ids) catch |err| return x.stop(err);
        if (hr.n > 0) x.e.headRows(hr.lo, hr.n) catch |err| return x.stop(err);
    }

    /// A verify forward whose drafts are made in the same step, with one header (the deferred draft, B2 C3 S2): each
    /// segment i with keeps[i].n > 0 first keeps that many rows of its slot's last forward, and with keeps[i].draft
    /// its drafter block runs from the window's first row (its anchor, ids[r0]); then the forward of `segs` and the
    /// head over all its rows. Every drafting segment's window rows 1 .. are its drafts, made on the device: their
    /// entries in `ids` are not read (rank 0 copies the drafter's walk into the forward's ids; `e.proposals` gives
    /// them after). The same kernels in the same order as `draftMany` and then `forwardMany` (a graph when it
    /// applies), so the same bits; one header and no wait between the draft and the window.
    pub fn forwardDrafted(x: Leader, segs: []const engine.Segment, ids: []const u32, keeps: []const Keep) !void {
        const e = x.e;
        if (e.drafter == null) return error.NoDrafter;
        if (segs.len == 0 or segs.len > engine.max_segs or keeps.len != segs.len) return error.BadSegments;
        for (segs) |g| try x.toolPages(g.slot, g.pos0);
        // every check a rank could fail alone, before the header: the keeps, each window at its kept position, the
        // drafts' rows, then the forward's own (as readyMany, at the positions the keeps leave)
        var R: usize = 0;
        var items: [engine.max_segs]engine.DraftItem = undefined;
        var walks: [engine.max_segs]engine.WalkIds = undefined;
        var m: usize = 0;
        var packed_keeps: u64 = 0;
        var kept: [engine.max_segs]Keep = undefined;
        for (segs, keeps, kept[0..segs.len], 0..) |g, k, *kk, i| {
            for (segs[0..i]) |o| if (o.slot == g.slot) return error.SlotTwiceInAForward;
            if (g.rows == 0 or g.pos0 == 0) return error.BadRows;
            kk.* = k;
            if (k.n > 0) {
                const st = try e.keepReady(g.slot, k.n);
                if (g.pos0 != st.last_pos0 + k.n or g.pos0 + g.rows > st.limit()) return error.OutOfStep;
                if (st.pos == g.pos0) kk.n = 0 else if (k.n > 0x7f) return error.BadKeep; // kept already: nothing to send
            }
            if (kk.n == 0) _ = try e.ready(g.slot, g.pos0, g.rows);
            if (k.draft) {
                if (g.rows - 1 > @import("cuda_dspark.zig").drafts or R >= ids.len) return error.BadRows;
                items[m] = .{ .slot = g.slot, .anchor = ids[R] };
                walks[m] = .{ .row = @intCast(R), .y = @intCast(m), .n = @intCast(g.rows - 1) };
                m += 1;
            }
            packed_keeps |= @as(u64, @intCast(kk.n | (@as(usize, @intFromBool(k.draft)) << 7))) << @intCast(8 * i);
            R += g.rows;
        }
        if (ids.len != R) return error.BadRows;
        if (packed_keeps == 0) return x.forwardMany(segs, ids, .{ .lo = 0, .n = R });
        if (R > engine.graph_rows) return error.BadRows;
        const fault = Fault.next();
        if (fault == .slot) return error.OutOfMemory;
        const hr: Rows = .{ .lo = 0, .n = R };
        const graph = x.graphOk(segs, R, hr);
        try x.fits(segs, graph);
        var h: engine.Header = .{ .kind = @intFromEnum(Kind.forward), .rows = @intCast(R), .head_lo = 0, .head_n = @intCast(R), .graph = @intFromBool(graph) };
        try h.setSegs(segs);
        try h.setTokens(ids); // the anchors (each drafting window's first row) reach the other ranks here
        h.pad[0] = packed_keeps;
        h.pad[1] = @intFromBool(e.prof_on);
        if (!graph and e.shared != null) h.token = 1; // and the eager forward's ids (`idsInHeader`)
        e.ids_in_header = h.token == 1;
        try x.send(h);
        if (fault == .step) return x.stop(error.InducedFault);
        x.keepsDrafts(segs, kept[0..segs.len], items[0..m]) catch |err| return x.stop(err);
        e.setWalkIds(walks[0..m]) catch |err| return x.stop(err);
        if (graph) {
            e.graphStep(segs, ids) catch |err| return x.stop(err);
            return;
        }
        e.forwardMany(segs, ids) catch |err| return x.stop(err);
        e.headRows(0, R) catch |err| return x.stop(err);
    }

    fn keepsDrafts(x: Leader, segs: []const engine.Segment, keeps: []const Keep, items: []const engine.DraftItem) !void {
        for (segs, keeps) |g, k| if (k.n > 0) try x.e.keep(g.slot, k.n);
        if (items.len > 0) try x.e.draftMany(items, null);
    }

    /// A segment's keep (rows of its slot's last forward; 0: none) and whether its drafter block runs after it
    /// (`forwardDrafted`).
    pub const Keep = struct { n: usize = 0, draft: bool = false };

    /// The head over rows `r` of the last forward, on every rank: rank 0's softcapped logits [r.n, vocab] (host
    /// memory, valid until the next forward or head).
    pub fn head(x: Leader, r: Rows) ![]const f32 {
        if (r.n == 0 or r.n > engine.head_rows or r.lo + r.n > x.e.rows) return error.BadRows;
        try x.send(.{ .kind = @intFromEnum(Kind.head), .head_lo = @intCast(r.lo), .head_n = @intCast(r.n) });
        x.e.headRows(r.lo, r.n) catch |err| return x.stop(err);
        return x.e.logits();
    }

    /// The slot keeps the first `n` rows of its last forward (a verify window), on every rank.
    pub fn keep(x: Leader, slot: u32, n: usize) !void {
        const st = try x.e.keepReady(slot, n);
        if (st.pos == st.last_pos0 + n) return;
        try x.send(.{ .kind = @intFromEnum(Kind.keep), .slot = slot, .rows = @intCast(n) });
        x.e.keep(slot, n) catch |err| return x.stop(err);
    }

    /// `keep` of n rows, then the drafter's block at the slot's new position from `anchor`, on every rank; rank 0's
    /// 7 drafts into `out`.
    pub fn draft(x: Leader, slot: u32, n: usize, anchor: u32, out: *[@import("cuda_dspark.zig").drafts]u32) !void {
        if (x.e.drafter == null) return error.NoDrafter;
        _ = try x.e.keepReady(slot, n);
        var h: engine.Header = .{ .kind = @intFromEnum(Kind.draft), .slot = slot, .rows = @intCast(n) };
        h.pad[0] = anchor;
        try x.send(h);
        x.e.keep(slot, n) catch |err| return x.stop(err);
        x.e.draft(slot, anchor, out) catch |err| return x.stop(err);
    }

    /// Several slots' drafts in one pass, on every rank: each slot keeps its `keeps[i]` rows (its last verify's
    /// accepted ones), then one drafter pass makes every slot's block (cuda_engine.zig draftMany: each slot's drafts
    /// the bits of its own `draft`); rank 0's 7 drafts of slot i into out[i]. One header for all of them.
    pub fn draftMany(x: Leader, items: []const engine.DraftItem, keeps: []const usize, out: []const *[@import("cuda_dspark.zig").drafts]u32) !void {
        if (x.e.drafter == null) return error.NoDrafter;
        if (items.len != keeps.len or items.len != out.len or items.len == 0 or items.len > engine.max_segs) return error.BadSegments;
        // everything a rank could fail on alone, before the header: each keep, then each block's taps after it
        var segs: [engine.max_segs]engine.Segment = undefined;
        for (items, keeps, segs[0..items.len], 0..) |it, n, *g, i| {
            for (items[0..i]) |b| if (b.slot == it.slot) return error.SlotTwiceInADraft;
            const st = try x.e.keepReady(it.slot, n);
            if (n == 0 or n > st.last_rows) return error.BadKeep;
            g.* = .{ .slot = it.slot, .pos0 = it.anchor, .rows = n };
        }
        var h: engine.Header = .{ .kind = @intFromEnum(Kind.drafts) };
        try h.setSegs(segs[0..items.len]);
        try x.send(h);
        for (items, keeps) |it, n| x.e.keep(it.slot, n) catch |err| return x.stop(err);
        x.e.draftMany(items, out) catch |err| return x.stop(err);
    }

    /// Tooling: slot `slot`'s state where it stands kept aside on every rank (cuda_engine.zig `mark`).
    pub fn mark(x: Leader, slot: u32) !void {
        if (x.e.pos(slot) == 0) return error.UnknownSlot;
        try x.send(.{ .kind = @intFromEnum(Kind.mark), .slot = slot });
        x.e.mark(slot) catch |err| return x.stop(err);
    }

    /// Tooling: slot `slot` back to its `mark` on every rank.
    pub fn rewind(x: Leader, slot: u32) !void {
        const m = x.e.marked orelse return error.NothingMarked;
        if (m.slot != slot) return error.NothingMarked;
        try x.send(.{ .kind = @intFromEnum(Kind.rewind), .slot = slot });
        x.e.rewind(slot) catch |err| return x.stop(err);
    }

    /// Tooling: slot `slot` as if positions 0 .. positions - 1 had run, on every rank (cuda_engine.zig `synth`).
    pub fn synth(x: Leader, slot: u32, positions: usize) !void {
        if (positions == 0 or positions >= x.e.capacity()) return error.PastContext;
        _ = try x.e.bind(slot); // (and its draft caches) a failed allocation moves no rank
        try x.toolPages(slot, 0);
        try x.send(.{ .kind = @intFromEnum(Kind.synth), .slot = slot, .pos0 = positions });
        x.e.synth(slot, positions) catch |err| return x.stop(err);
    }

    /// Tooling: every rank reads slot `slot`'s global layers' positions 0 .. positions - 1 back against the fill and
    /// logs it; true only if every rank's rows are equal (the verdicts gathered in Engine.synthCheck).
    pub fn synthCheck(x: Leader, slot: u32, positions: usize) !bool {
        if (slot >= x.e.slots.items.len or x.e.slots.items[slot] == null) return error.UnknownSlot;
        try x.send(.{ .kind = @intFromEnum(Kind.synth_check), .slot = slot, .pos0 = positions });
        const got = x.e.synthCheck(slot, positions) catch |err| return x.stop(err);
        return got.bad == null and got.ranks_bad == 0;
    }

    /// Every rank's states for `n` streams (slots 0 .. n - 1), made at start-up, on pages the pool of rank 0's
    /// planned pages first (cuda_native.zig count): rank 0's first, so its failure sends nothing; a rank that fails
    /// after stops every rank. The other ranks then sleep again.
    pub fn reserve(x: Leader, n: u32) !void {
        const npages: usize = if (x.e.plan) |p| p.pages else 0;
        try x.e.reserve(n, npages);
        if (x.e.o.world < 2) return;
        try x.send(.{ .kind = @intFromEnum(Kind.reserve), .slot = n, .pos0 = npages });
        try x.idle();
    }

    /// Slot `slot`'s stream holds pages `ids` of the pool (rank 0's allocation; the first `shared` a kept state's,
    /// adopted), on every rank: the header, then the slot's table broadcast (Engine.mapPages). Checked before the
    /// header: the slot is on the pool and the ids fit its table.
    pub fn mapPages(x: Leader, slot: u32, ids: []const u32, shared: usize) !void {
        const p = x.e.pool orelse return error.NoPool;
        if (slot >= p.slots or ids.len == 0 or shared > ids.len) return error.NoSlot;
        const st = try x.e.bind(slot);
        if (ids.len > st.ids.len) return error.PastContext;
        for (ids) |id| if (id >= p.total()) return error.BadPage;
        try x.send(.{ .kind = @intFromEnum(Kind.pages), .slot = slot, .rows = @intCast(ids.len), .pos0 = shared });
        x.e.mapPages(slot, ids.len, ids, shared) catch |err| return x.stop(err);
    }

    /// Rank 0: slot `slot`'s stream takes `n` pages of the pool, the first `adopt.len` a kept state's (Pool.open),
    /// and every rank maps them (`mapPages`). LaneMemory: not now; nothing was sent.
    pub fn openPages(x: Leader, slot: u32, n: usize, adopt: []const u32) ![]const u32 {
        const p = x.e.pool orelse return error.NoPool;
        _ = try x.e.bind(slot);
        const ids = try p.open(slot, n, adopt);
        try x.mapPages(slot, ids, adopt.len);
        return ids;
    }

    /// Tooling on the tool pool (Engine.toolPool): before a forward of slot `slot` from `pos0`, the slot holds its
    /// whole context's pages: taken when it first runs, and taken anew when a new sequence (pos0 = 0) would rewrite
    /// pages a kept state holds.
    pub fn toolPages(x: Leader, slot: u32, pos0: usize) !void {
        if (!x.e.tool_pool) return;
        const p = x.e.pool.?;
        const st = try x.e.bind(slot);
        const a = &(p.alloc orelse return error.NotLeader);
        if (st.mapped == p.entries and !(pos0 == 0 and a.anyKept(st.ids[0..st.mapped]))) return;
        _ = try x.openPages(slot, p.entries, &.{});
    }

    /// No stream is left: the other ranks sleep on their lines until the next step (when the lines are open).
    pub fn idle(x: Leader) !void {
        if (!x.e.asleep) x.e.scratch.report(x.e.o.rank);
        if (x.e.o.world < 2 or x.e.asleep or !cuda.tp_watch.open()) return;
        try x.send(.{ .kind = @intFromEnum(Kind.idle) });
        x.e.asleep = true;
    }

    /// The other ranks leave their loops.
    pub fn end(x: Leader) !void {
        if (x.e.o.world < 2) return;
        cuda.tp_watch.ended();
        try x.send(.{ .kind = @intFromEnum(Kind.end) });
    }
};

/// A forward header's keeps and drafts (Leader.forwardDrafted, pad[0]: segment i's keep in byte i, its draft flag
/// in that byte's bit 7; the anchors are the token ids of the drafting windows' first rows). With the shared
/// embedding the window rows the drafts fill are set as on rank 0 (a captured forward reads this rank's header copy).
fn followDrafts(e: *engine.Engine, h: *const engine.Header) !void {
    var buf: [engine.max_segs]engine.Segment = undefined;
    const segs = try h.getSegs(&buf);
    var items: [engine.max_segs]engine.DraftItem = undefined;
    var walks: [engine.max_segs]engine.WalkIds = undefined;
    var m: usize = 0;
    var r0: usize = 0;
    for (segs, 0..) |g, i| {
        const b: u8 = @truncate(h.pad[0] >> @intCast(8 * i));
        const n = b & 0x7f;
        if (n > 0) try e.keep(g.slot, n);
        if (b & 0x80 != 0) {
            if (r0 >= engine.graph_rows) return error.BadStep;
            if (g.rows == 0 or g.rows - 1 > @import("cuda_dspark.zig").drafts) return error.BadStep;
            items[m] = .{ .slot = g.slot, .anchor = @intCast(h.tokens[r0]) };
            walks[m] = .{ .row = @intCast(r0), .y = @intCast(m), .n = @intCast(g.rows - 1) };
            m += 1;
        }
        r0 += g.rows;
    }
    if (m > 0) try e.draftMany(items[0..m], null);
    if (e.shared != null) try e.setWalkIds(walks[0..m]);
}

/// A rank other than 0: rank 0's steps until it ends.
pub fn follow(e: *engine.Engine) !void {
    var steps: u64 = 0;
    cuda.tp_watch.sleep(e.io); // returns at once without lines
    while (true) {
        const h = try e.receive();
        switch (std.enums.fromInt(Kind, h.kind) orelse return error.BadStep) {
            .forward => {
                e.prof_on = h.pad[1] == 1; // rank 0's budget pass
                if (h.pad[0] != 0) try followDrafts(e, &h); // Leader.forwardDrafted: the keeps and drafts first
                if (h.graph == 1) {
                    var buf: [engine.max_segs]engine.Segment = undefined;
                    if (h.head_lo != 0 or h.head_n != h.rows) return error.BadStep;
                    try e.graphStep(try h.getSegs(&buf), null);
                } else if (h.nseg > 0) {
                    var buf: [engine.max_segs]engine.Segment = undefined;
                    e.ids_in_header = h.token == 1;
                    try e.forwardMany(try h.getSegs(&buf), null);
                    if (h.head_n > 0) try e.headRows(h.head_lo, h.head_n);
                } else {
                    e.ids_in_header = h.token == 1;
                    try e.forward(h.slot, @intCast(h.pos0), h.rows, null);
                    if (h.head_n > 0) try e.headRows(h.head_lo, h.head_n);
                }
            },
            .head => try e.headRows(h.head_lo, h.head_n),
            .reserve => try e.reserve(h.slot, @intCast(h.pos0)),
            .pages => try e.mapPages(h.slot, h.rows, null, @intCast(h.pos0)),
            .mark => try e.mark(h.slot),
            .rewind => try e.rewind(h.slot),
            .synth => try e.synth(h.slot, @intCast(h.pos0)),
            .synth_check => _ = try e.synthCheck(h.slot, @intCast(h.pos0)),
            .keep => try e.keep(h.slot, h.rows),
            .draft => {
                try e.keep(h.slot, h.rows);
                try e.draft(h.slot, @intCast(h.pad[0]), null);
            },
            .save, .restore, .arena, .capture => try @import("cuda_snap.zig").follow(e, h), // the prompt cache's copies
            .move => return error.BadStep, // the copies' compaction (stage 1) is gone
            .drafts => {
                var buf: [engine.max_segs]engine.Segment = undefined;
                const segs = try h.getSegs(&buf);
                var items: [engine.max_segs]engine.DraftItem = undefined;
                for (segs, items[0..segs.len]) |g, *it| {
                    try e.keep(g.slot, g.rows);
                    it.* = .{ .slot = g.slot, .anchor = @intCast(g.pos0) };
                }
                try e.draftMany(items[0..segs.len], null);
            },
            .idle => {
                e.scratch.report(e.o.rank);
                cuda.tp_watch.sleep(e.io);
            },
            .end => {
                cuda.tp_watch.ended();
                std.log.info("inkling2 rank {d}: rank 0 ended after {d} steps", .{ e.o.rank, steps });
                return;
            },
        }
        steps += 1;
    }
}
