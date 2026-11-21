//! Inkling-2's prompt cache on the KV page pool (v0.2 stage 2d; the copies of stage 1 are gone): a kept state at N is
//! the stream's whole pages [0, N / 256) by reference, and a copy of the slot's fixed part and the tail rows in one
//! slot of a small device arena a rank made at start-up; a resume adopts those pages and copies the fixed part back.
//! Rank 0 decides every save and restore inside its lane thread and sends it as a step (kinds save and restore,
//! cuda_follow.zig), so every rank copies the same bytes at the same point of the step sequence, at the same arena
//! offset; rank 0 alone counts the kept states' page references (cuda_pages.zig retain / release), the other ranks
//! map whatever ids rank 0's `pages` step sends.
//!
//! What a state at N holds, per rank (nothing else is read by a row at or past N):
//!   global layers   K and V of positions [0, B), B = N - N % 256: the stream's pages ids[0 .. N / 256), referenced
//!                   (never written again: a stream writes only at or past its position, and a stream that resumes
//!                   the state writes only past N), and of [B, N) (the tail rows) copied into the slot
//!   sliding layers  K and V of the last 1,023 positions [N - 1023, N) in position order (restored to p % slots)
//!   every layer     its four convolutions' 3 input rows (k, v, post-attention, post-MLP)
//!   DSpark          the drafter's context K and V: its global layer as the target's (pages and tail, stage 2c), its
//!                   local layers [N - 1024, N) (restored into their rings at p % ring, the rows below 1,024 mirrored)
//! A resume at N: the stream's table is the state's pages and fresh pages after them (cuda_pages.zig Pool.open with
//! `adopt`, in place when the next pages are free), and the tail rows land in the first fresh page. Every slot of
//! the arena is one size (a state's largest copy), so it never fragments: there is no compaction.
//! A state is taken where a forward ended at N, or (stage 1b) at a mark N inside a prompt chunk [a, a + n): rank 0
//! sends a `capture` step before the chunk's forward, and each convolution commit of that forward first writes what
//! the convolution's state would be at N (the commit kernel over the old state and rows a .. N - 1) into a capture
//! area; the save after the forward takes those rows, and the rest of the state is still intact (global rows [0, N)
//! are never overwritten; the rings hold 1,023 + 2,048 (target) and 1,024 + 2,048 (drafter) positions, so
//! [N - 1,024, N) survives a chunk of at most 2,048 rows past N). Every op is row-invariant, so the bytes do not
//! depend on how [0, N) was chunked and a resume followed by any chunking of the rest gives the cold pass's bits.
//!
//! Rank 0 keeps the arena's free extents (slots of one size); every rank keeps what each extent holds (a tag a save),
//! and a restore whose tag the follower does not hold stops every rank. Followers never allocate after start-up, so
//! a save cannot fail on one rank alone for want of memory. The window layout (TF_INK2_PAGES=0) keeps no prompts.
//!
//! TF_INK2_SNAP_FAULT (tests): `restore:N` refuses rank 0's N-th restore before its header (the lane falls back to a
//! cold pass); `tag:N` makes a follower's N-th restore find another tag (every rank stops).
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const weights = @import("cuda_weights.zig");
const attn = @import("cuda_attn.zig");
const conv = @import("cuda_conv.zig");
const dspark = @import("cuda_dspark.zig");
const ops = @import("cuda_ops.zig");
const follow_ = @import("cuda_follow.zig");
const pages = @import("cuda_pages.zig");

const Engine = engine.Engine;
const Mat = ops.Mat;
const Header = engine.Header;
const Leader = follow_.Leader;
const Kind = follow_.Kind;
const H = weights.hidden;
const D = weights.head_dim;

/// The arena's unit: every kept state takes a whole number of them.
pub const granule: u64 = 2 << 20;
/// Positions a sliding layer's kept state holds: a row reads the 1,023 before its own.
pub const ring_keep: usize = attn.window - 1;
/// Positions a local drafter layer's kept state holds: a draft at p reads [p - 1024, p), and p > N after a resume.
pub const draft_keep: usize = dspark.window;

/// Rank 0's record of one kept state: its arena slot and the pages it references (rank 0's allocation, owned).
pub const Kept = struct {
    tag: u32,
    at: u32,
    off: u64,
    bytes: u64,
    pages: []u32 = &.{},
    /// The images of the kept prefix (rank 0's host only; cuda_vision_store.zig prefixKey over the images before
    /// `at`): all zeros for a prefix without images. A resume refuses a state whose images differ (cuda_lanes.zig).
    images: [32]u8 = @splat(0),
};

/// Stage 1b: marks one prompt chunk captures inside it at most (more end chunks, as stage 1 did).
pub const max_caps = 8;

/// A prompt chunk whose forward captures the convolution rows of the marks inside it (every rank: rank 0 sets it,
/// followers take it from the `capture` step). `ran`: the forward took it; a save inside the chunk reads capture i.
pub const Capture = struct {
    slot: u32 = 0,
    pos0: usize = 0,
    rows: usize = 0,
    n: usize = 0,
    marks: [max_caps]u32 = undefined,
    armed: bool = false,
    ran: bool = false,
};

/// The convolution rows a state holds (every layer's four 3-row states), in the copies' order.
pub fn convBytes(layers: anytype) u64 {
    var n: u64 = 0;
    for (layers) |l| n += 2 * conv.state_rows * @as(u64, l.hk) * D * 2 + 2 * conv.state_rows * H * 2;
    return n;
}

/// The capture area a rank allocates with its arena (out of the cache's budget).
pub fn capBytes(e: *const Engine) u64 {
    return max_caps * convBytes(e.w.layers);
}

/// Marks strictly inside the chunk [at, at + n), ascending, into `out` (at most max_caps); the rows the chunk may run
/// so that no mark past those is inside it (a chunk with more marks ends at the next one).
pub const Inside = struct { n: usize, rows: usize };

pub fn inside(marks: []const u32, at: usize, n: usize, out: *[max_caps]u32) Inside {
    var k: usize = 0;
    for (marks) |m| if (m > at and m < at + n) {
        if (k == max_caps) return .{ .n = k, .rows = m - at };
        out[k] = m;
        k += 1;
    };
    return .{ .n = k, .rows = n };
}

/// TF_INK2_SNAP_SPLIT=1 (rank 0, A/B): every mark ends a chunk (stage 1), no capture.
pub fn splitAtMarks() bool {
    const v = std.c.getenv("TF_INK2_SNAP_SPLIT") orelse return false;
    return v[0] == '1' and v[1] == 0;
}

/// The capture hook for the forward of `R` rows of `slot` from `pos0` (cuda_engine.zig forward), when the capture
/// step armed it for that chunk; it is taken once.
pub fn forwardHook(e: *Engine, slot: u32, pos0: usize, R: usize) ?@import("cuda_forward.zig").Capture {
    const p = &e.snap.capture;
    if (!p.armed) return null;
    p.armed = false;
    if (p.slot != slot or p.pos0 != pos0 or p.rows != R) return null;
    p.ran = true;
    return .{ .ctx = e, .conv = captureConv };
}

/// Before the commit of convolution state `conv_state` with rows x (the chunk's, from pos0): each mark m's state,
/// the commit kernel over (the old state, rows pos0 .. m - 1), into capture i at the convolution's place. A chunk in
/// parts (cuda_xover.zig: x the rows from part.pos0, the state the one before them): only the marks this part's rows
/// reach (above part.pos0 and at most part.pos0 + x.rows, or at part.pos0 in the first part), from part.pos0.
fn captureConv(ptr: *anyopaque, c: *const ops.Ctx, conv_state: u64, x: Mat, part: ?@import("cuda_forward.zig").Capture.Part) anyerror!void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const a = &e.snap;
    const p = &a.capture;
    const cap = a.cap orelse return error.NoArena;
    const st = e.slots.items[p.slot] orelse return error.UnknownSlot;
    var off: u64 = 0;
    for (st.layers, e.w.layers) |ls, l| {
        const kv: u64 = conv.state_rows * @as(u64, l.hk) * D * 2;
        const hh: u64 = conv.state_rows * H * 2;
        for ([4]u64{ ls.conv_k, ls.conv_v, ls.conv_attn, ls.conv_mlp }, [4]u64{ kv, kv, hh, hh }) |q, n| {
            if (q == conv_state) {
                for (p.marks[0..p.n], 0..) |m, i| {
                    const dst = cap.ptr + i * a.cap_stride + off;
                    const lo = if (part) |pt| pt.pos0 else p.pos0;
                    if (part) |pt| if (m < pt.pos0 or m > pt.pos0 + x.rows or (m == pt.pos0 and !pt.first)) continue;
                    const k = m - lo;
                    if (k < conv.state_rows) try c.copy(dst, conv_state, @intCast(n)); // rows before the chunk stay
                    try state.convCommit(c, dst, x.slice(0, k));
                }
                return;
            }
            off += n;
        }
    }
}

/// Rank 0, before the forward of rows [at, at + n) of `slot`: the marks inside it captured by that forward on every
/// rank.
pub fn capture(x: Leader, slot: u32, at: usize, n: usize, marks: []const u32) !void {
    const e = x.e;
    if (e.snap.cap == null or marks.len == 0 or marks.len > max_caps) return error.BadCapture;
    var h: Header = .{ .kind = @intFromEnum(Kind.capture), .slot = slot, .pos0 = at, .rows = @intCast(n), .nseg = @intCast(marks.len) };
    for (marks, 0..) |m, i| h.tokens[i] = @intCast(m);
    try x.send(h);
    setCapture(e, h);
}

fn setCapture(e: *Engine, h: Header) void {
    const p = &e.snap.capture;
    p.* = .{ .slot = h.slot, .pos0 = h.pos0, .rows = h.rows, .n = h.nseg, .armed = true };
    for (p.marks[0..p.n], h.tokens[0..p.n]) |*m, v| m.* = @intCast(v);
}

/// The capture that holds slot `slot`'s state at `at` (a mark inside the last forward, which took the capture), or
/// null.
fn capturedAt(e: *const Engine, slot: u32, at: usize) ?usize {
    const p = &e.snap.capture;
    if (!p.ran or p.slot != slot) return null;
    const st = e.slots.items[slot] orelse return null;
    if (st.last_pos0 != p.pos0 or st.last_rows != p.rows or st.pos != p.pos0 + p.rows) return null; // another forward since
    for (p.marks[0..p.n], 0..) |m, i| if (m == at) return i;
    return null;
}

/// A save's flag (Header.head_lo): the convolution rows come from capture (low byte).
const cap_flag: u32 = 0x200;

/// One layer as the copies see it (cuda_weights.zig Layer has the same two fields).
pub const LayerShape = struct { hk: u32, global: bool };

/// The global layers' rows a state at `at` copies: the tail past its whole pages.
pub fn tailRows(at: usize) usize {
    return at % pages.page_rows;
}

/// The bytes a state at `at` copies, unrounded: `layers` (anything with hk and global), the drafter's key heads (0:
/// no drafter); the global layers' rows below the tail are the pages it references.
pub fn rawBytes(layers: anytype, drafter_hk: usize, at: usize) u64 {
    var n: u64 = 0;
    for (layers) |l| {
        const row: u64 = @as(u64, l.hk) * D * 2;
        n += 2 * conv.state_rows * row + 2 * conv.state_rows * H * 2;
        n += 2 * row * @as(u64, if (l.global) tailRows(at) else @min(at, ring_keep));
    }
    if (drafter_hk > 0) {
        const row: u64 = @as(u64, drafter_hk) * D * 2;
        for (dspark.local) |loc| n += 2 * row * @as(u64, if (loc) @min(at, draft_keep) else tailRows(at));
    }
    return n;
}

pub fn rounded(n: u64) u64 {
    return (n + granule - 1) / granule * granule;
}

/// A position whose copy is the largest: full rings (>= 1,024) and the longest tail (255 rows).
pub const widest_at: usize = 4 * pages.page_rows + pages.page_rows - 1;

/// An arena slot's bytes: the largest copy any state takes (every rank the same: equal head shares), rounded.
pub fn slotBytes(layers: anytype, drafter_hk: usize) u64 {
    return rounded(rawBytes(layers, drafter_hk, widest_at));
}

/// The arena bytes a state at `at` takes on this engine: one slot, whatever `at`.
pub fn bytes(e: *const Engine, at: usize) u64 {
    _ = at;
    return slotBytes(e.w.layers, if (e.drafter) |dr| dr.hk else 0);
}

/// Free extents of the arena (rank 0): best fit, holes kept sorted and merged. Room for every hole is reserved up
/// front, so giving an extent back never allocates.
pub const Extents = struct {
    pub const Span = struct { off: u64, len: u64 };
    holes: std.ArrayList(Span) = .empty,
    size: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, size: u64) !Extents {
        var x: Extents = .{ .size = size };
        try x.holes.ensureTotalCapacity(gpa, @intCast(size / granule / 2 + 2));
        if (size > 0) x.holes.appendAssumeCapacity(.{ .off = 0, .len = size });
        return x;
    }

    pub fn deinit(x: *Extents, gpa: std.mem.Allocator) void {
        x.holes.deinit(gpa);
    }

    /// The smallest hole that holds `n` bytes (the lowest such on a tie); its first `n` bytes.
    pub fn take(x: *Extents, n: u64) ?u64 {
        if (n == 0) return null;
        var best: ?usize = null;
        for (x.holes.items, 0..) |h, i| if (h.len >= n and (best == null or h.len < x.holes.items[best.?].len)) {
            best = i;
        };
        const i = best orelse return null;
        const h = &x.holes.items[i];
        const off = h.off;
        if (h.len == n) _ = x.holes.orderedRemove(i) else {
            h.off += n;
            h.len -= n;
        }
        return off;
    }

    /// Bytes [off, off + n) free again, merged with the holes beside them.
    pub fn give(x: *Extents, off: u64, n: u64) void {
        var i: usize = 0;
        while (i < x.holes.items.len and x.holes.items[i].off < off) i += 1;
        const prev = i > 0 and x.holes.items[i - 1].off + x.holes.items[i - 1].len == off;
        const next = i < x.holes.items.len and off + n == x.holes.items[i].off;
        if (prev and next) {
            x.holes.items[i - 1].len += n + x.holes.items[i].len;
            _ = x.holes.orderedRemove(i);
        } else if (prev) {
            x.holes.items[i - 1].len += n;
        } else if (next) {
            x.holes.items[i].off = off;
            x.holes.items[i].len += n;
        } else x.holes.insertAssumeCapacity(i, .{ .off = off, .len = n });
    }

    pub fn free(x: *const Extents) u64 {
        var n: u64 = 0;
        for (x.holes.items) |h| n += h.len;
        return n;
    }

    pub fn largest(x: *const Extents) u64 {
        var n: u64 = 0;
        for (x.holes.items) |h| n = @max(n, h.len);
        return n;
    }
};

/// What an extent of the arena holds (every rank): the save that wrote it.
pub const Held = struct { off: u64, len: u64, tag: u32, at: u32 };

/// One rank's arena: its device bytes, what each extent holds, and (rank 0) the free extents and the last tag.
pub const Arena = struct {
    buf: ?cuda.DeviceBuffer = null,
    size: u64 = 0,
    extents: Extents = .{},
    held: std.ArrayList(Held) = .empty,
    /// Rank 0: the kept states' records (their pages: the reclaim chain's count, cuda_lanes.zig fits).
    live: std.ArrayList(*Kept) = .empty,
    tag: u32 = 0,
    saves: u64 = 0,
    restores: u64 = 0,
    /// Stage 1b: `max_caps` states' convolution rows captured inside a prompt chunk (every rank), and the chunk.
    cap: ?cuda.DeviceBuffer = null,
    cap_stride: u64 = 0,
    capture: Capture = .{},
    captured: u64 = 0,

    pub fn deinit(a: *Arena, gpa: std.mem.Allocator) void {
        if (a.buf) |*b| b.free();
        if (a.cap) |*b| b.free();
        a.extents.deinit(gpa);
        a.held.deinit(gpa);
        a.live.deinit(gpa);
        a.* = .{};
    }

    /// A save wrote [off, off + len): the states it overlapped are gone.
    pub fn note(a: *Arena, off: u64, len: u64, tag: u32, at: u32) void {
        var i: usize = 0;
        while (i < a.held.items.len) {
            const h = a.held.items[i];
            if (h.off < off + len and off < h.off + h.len) _ = a.held.swapRemove(i) else i += 1;
        }
        a.held.appendAssumeCapacity(.{ .off = off, .len = len, .tag = tag, .at = at });
    }

    /// The extent at `off` holds save `tag` of `at` tokens.
    pub fn holds(a: *const Arena, off: u64, tag: u32, at: u32) bool {
        for (a.held.items) |h| if (h.off == off) return h.tag == tag and h.at == at;
        return false;
    }

    fn forget(a: *Arena, off: u64) void {
        for (a.held.items, 0..) |h, i| if (h.off == off) {
            _ = a.held.swapRemove(i);
            return;
        };
    }
};

/// This rank's arena of `n` bytes (start-up; every rank the same size).
pub fn arenaInit(e: *Engine, n: u64) !void {
    if (e.snap.buf != null) return error.ArenaTwice;
    if (n == 0 or n % granule != 0) return error.BadArena;
    var a: Arena = .{ .size = n };
    errdefer a.deinit(e.gpa);
    a.extents = try Extents.init(e.gpa, n);
    try a.held.ensureTotalCapacity(e.gpa, @intCast(n / granule + 1));
    try a.live.ensureTotalCapacity(e.gpa, @intCast(n / granule + 1));
    a.buf = try cuda.DeviceBuffer.alloc(e.d, @intCast(n));
    a.cap_stride = convBytes(e.w.layers);
    a.cap = try cuda.DeviceBuffer.alloc(e.d, @intCast(max_caps * a.cap_stride));
    e.snap = a;
    std.log.info("inkling2 rank {d}: prompt cache arena {d:.2} GiB: {d} kept states of {d} MiB (each its pages by reference, its fixed part and tail rows copied)", .{ e.o.rank, gib(n), n / bytes(e, 0), bytes(e, 0) >> 20 });
}

pub const Dir = enum { out, in };

/// Positions [lo, hi) of a store of `slots` slots (position p at slot p % slots) as at most two runs of slots, in
/// position order (hi - lo <= slots).
pub const Runs = struct {
    pub const Run = struct { slot: usize, len: usize };
    r: [2]Run = undefined,
    n: usize = 0,
};

pub fn ringRuns(slots: usize, lo: usize, hi: usize) Runs {
    var out: Runs = .{};
    var p = lo;
    while (p < hi and out.n < 2) {
        const s = p % slots;
        const n = @min(hi - p, slots - s);
        out.r[out.n] = .{ .slot = s, .len = n };
        out.n += 1;
        p += n;
    }
    return out;
}

/// Rows of a prompt chunk from `at`: at most `max`, and it ends at the first of `marks` (ascending) past `at`.
pub fn rowsTo(marks: []const u32, at: usize, max: usize) usize {
    for (marks) |m| if (m > at) return @min(max, m - at);
    return max;
}

/// The copies between a slot and an arena extent, in one fixed order (so out and in agree), counted without a
/// context.
pub const Walk = struct {
    c: ?*const ops.Ctx,
    dir: Dir,
    base: u64,
    off: u64 = 0,

    fn move(w: *Walk, live: u64, n: u64) !void {
        if (n == 0) return;
        if (w.c) |c| switch (w.dir) {
            .out => try c.copy(w.base + w.off, live, @intCast(n)),
            .in => try c.copy(live, w.base + w.off, @intCast(n)),
        };
        w.off += n;
    }

    /// Positions [lo, hi) of a store of `slots` slots, `row` bytes each, in position order.
    pub fn rows(w: *Walk, ptr: u64, slots: usize, row: usize, lo: usize, hi: usize) !void {
        if (hi < lo or hi - lo > slots) return error.SnapRing;
        const runs = ringRuns(slots, lo, hi);
        for (runs.r[0..runs.n]) |r| try w.move(ptr + r.slot * row, r.len * row);
    }

    /// `rows` for a mirrored ring (cuda_dspark.zig Caches: position p at row p % slots, and rows below `mirror` again
    /// at row + slots): a restore also writes each restored row below `mirror` to its mirror, from the same arena
    /// bytes (the arena holds each position once).
    pub fn mirrored(w: *Walk, ptr: u64, slots: usize, row: usize, lo: usize, hi: usize, mirror: usize) !void {
        const off0 = w.off;
        try w.rows(ptr, slots, row, lo, hi);
        const c = w.c orelse return;
        if (w.dir != .in) return;
        var p = lo;
        while (mirrorRun(slots, mirror, p, hi)) |m| {
            try c.copy(ptr + (m.row + slots) * row, w.base + off0 + (m.p - lo) * row, @intCast(m.n * row));
            p = m.p + m.n;
        }
    }
};

/// The next run of positions in [p, hi) whose ring rows (p % slots) lie below `mirror`: its first position, row and
/// length, or null.
pub fn mirrorRun(slots: usize, mirror: usize, p0: usize, hi: usize) ?struct { p: usize, row: usize, n: usize } {
    var p = p0;
    while (p < hi) {
        const r = p % slots;
        if (r >= mirror) {
            p += slots - r;
            continue;
        }
        return .{ .p = p, .row = r, .n = @min(hi - p, mirror - r) };
    }
    return null;
}

/// `conv_src` (a save inside the last forward): the convolution rows come from that capture, laid out in this order.
fn walk(e: *Engine, c: ?*const ops.Ctx, dir: Dir, slot: u32, at: usize, base: u64, conv_src: ?u64) !u64 {
    if (slot >= e.slots.items.len) return error.UnknownSlot;
    const st = e.slots.items[slot] orelse return error.UnknownSlot;
    if (at == 0 or at > st.context) return error.BadSnapPosition;
    if (conv_src != null and dir != .out) return error.BadCapture;
    var w: Walk = .{ .c = c, .dir = dir, .base = base };
    var cp: u64 = conv_src orelse 0;
    for (st.layers, e.w.layers) |ls, l| {
        const cols: usize = @as(usize, l.hk) * D;
        for ([4]u64{ ls.conv_k, ls.conv_v, ls.conv_attn, ls.conv_mlp }, [4]usize{ cols, cols, H, H }) |q, n| {
            const len = conv.state_rows * n * 2;
            try w.move(if (conv_src != null) cp else q, len);
            cp += len;
        }
        if (l.global) {
            // a global layer on pages: the tail rows [B, at) (the pages below B are the state's by reference)
            if (!ls.kv.paged()) return error.CacheNeedsPages;
            if (at > st.limit()) return error.PastPages;
            const lo = at - tailRows(at);
            var keys = state.RowRuns.of(ls.kv, st.ids, lo, at);
            while (try keys.next()) |r| try w.move(r.k, r.n * cols * 2);
            var values = state.RowRuns.of(ls.kv, st.ids, lo, at);
            while (try values.next()) |r| try w.move(r.v, r.n * cols * 2);
        } else {
            const lo = at -| ring_keep;
            try w.rows(ls.kv.k, ls.kv.slots, cols * 2, lo, at);
            try w.rows(ls.kv.v, ls.kv.slots, cols * 2, lo, at);
        }
    }
    if (e.drafter) |dr| {
        if (slot >= dr.caches.items.len) return error.UnknownSlot;
        const cs = &(dr.caches.items[slot] orelse return error.UnknownSlot);
        if (at > dr.context) return error.BadSnapPosition;
        // a local layer is a mirrored ring of cs.wrap(i) positions (lane M), or position-indexed (wrap 0)
        for (dspark.local, 0..) |loc, i| {
            if (!loc) {
                // the global layer on pages (stage 2c) as the target's: the tail rows, the rest by reference
                const kv = cs.pagedKv(st.ids.len) orelse return error.CacheNeedsPages;
                if (at > st.limit()) return error.PastPages;
                const lo = at - tailRows(at);
                var keys = state.RowRuns.of(kv, st.ids, lo, at);
                while (try keys.next()) |r| try w.move(r.k, r.n * cs.row);
                var values = state.RowRuns.of(kv, st.ids, lo, at);
                while (try values.next()) |r| try w.move(r.v, r.n * cs.row);
                continue;
            }
            const lo = at -| draft_keep;
            const ring = cs.wrap(i);
            if (ring > 0) {
                try w.mirrored(cs.keys(i), ring, cs.row, lo, at, dspark.window);
                try w.mirrored(cs.values(i), ring, cs.row, lo, at, dspark.window);
            } else {
                try w.rows(cs.keys(i), cs.context, cs.row, lo, at);
                try w.rows(cs.values(i), cs.context, cs.row, lo, at);
            }
        }
    }
    return w.off;
}

fn ctx(e: *Engine) ops.Ctx {
    return .{ .d = e.d, .stream = e.stream, .k = &e.k, .gemms = &e.gemms, .scratch = &e.scratch };
}

/// The copies of `at` tokens' state in direction `dir` between slot `slot` and the arena at `off` (on the engine's
/// stream, after the slot's forwards and before its next).
fn copy(e: *Engine, dir: Dir, slot: u32, at: usize, off: u64, cap_i: ?usize) !void {
    const conv_src: ?u64 = if (cap_i) |i| (e.snap.cap orelse return error.NoArena).ptr + i * e.snap.cap_stride else null;
    const buf = e.snap.buf orelse return error.NoArena;
    const n = try walk(e, null, dir, slot, at, 0, null);
    if (n != rawBytes(e.w.layers, if (e.drafter) |dr| dr.hk else 0, at)) return error.SnapLayout;
    if (off % granule != 0 or off + rounded(n) > e.snap.size) return error.SnapOutside;
    const c = ctx(e);
    _ = try walk(e, &c, dir, slot, at, buf.ptr + off, conv_src);
}

/// Slot `slot` (its forwards so far ended at `at`) copied into the arena at `off`.
pub fn copyOut(e: *Engine, slot: u32, at: usize, off: u64) !void {
    try copy(e, .out, slot, at, off, null);
}

/// Slot `slot`'s state at `at` into the arena at `off`: its forwards ended at `at`, or (`cap_i`) `at` is a mark inside
/// the last forward and capture cap_i holds its convolution rows.
pub fn copyOutFrom(e: *Engine, slot: u32, at: usize, off: u64, cap_i: ?usize) !void {
    try copy(e, .out, slot, at, off, cap_i);
}

/// The arena at `off` copied into slot `slot`, which then stands at `at` as if its own forwards had ended there.
pub fn copyIn(e: *Engine, slot: u32, at: usize, off: u64) !void {
    try copy(e, .in, slot, at, off, null);
    try standAt(e, slot, at);
}

/// A state's fixed part at device address `base` (the NVMe tier's staging: mapped host memory, the bytes of an arena
/// slot in the walk's order) copied into slot `slot`, which then stands at `at` as after `copyIn`.
pub fn copyInFrom(e: *Engine, slot: u32, at: usize, base: u64) !void {
    const n = try walk(e, null, .in, slot, at, 0, null);
    if (n != rawBytes(e.w.layers, if (e.drafter) |dr| dr.hk else 0, at)) return error.SnapLayout;
    const c = ctx(e);
    _ = try walk(e, &c, .in, slot, at, base, null);
    try standAt(e, slot, at);
    e.snap.restores += 1;
}

/// The slot's scalars after its state at `at` was copied in (its forwards as if they had ended there).
fn standAt(e: *Engine, slot: u32, at: usize) !void {
    const st = e.slots.items[slot].?;
    st.pos = at;
    st.last_pos0 = at;
    st.last_rows = 0; // no keep or draft before the next forward
    st.kept_inputs = false;
    st.last_row0 = 0;
    st.last_fwd = 0; // the taps are not this slot's
    e.rows = 0; // no head over another slot's rows
    e.head_n = 0;
    try control(e, slot, at);
}

/// TF_INK2_SNAP_CONTROL (negative controls of the resume gate, every rank): after each restore, one piece of the
/// state it restored is overwritten (finite bytes), so the warm run must differ: `ring` the oldest kept ring position
/// of every sliding layer (at >= 1023), `conv` every layer's oldest post-MLP convolution row (at >= 3), `draft` the
/// drafter's global layer's keys at positions [0, 16).
fn control(e: *Engine, slot: u32, at: usize) !void {
    const v = std.mem.span(std.c.getenv("TF_INK2_SNAP_CONTROL") orelse return);
    const c = ctx(e);
    const st = e.slots.items[slot].?;
    const byte: u8 = 0x3c;
    const view = struct {
        fn fill(cx: *const ops.Ctx, p: u64, n: usize, b: u8) !void {
            const d: cuda.DeviceBuffer = .{ .d = cx.d, .ptr = p, .len = n, .borrowed = true };
            try d.fill8(b, cx.stream.handle);
        }
    };
    if (std.mem.eql(u8, v, "ring") and at >= ring_keep) {
        for (st.layers, e.w.layers) |ls, l| if (!l.global) {
            const row: usize = @as(usize, l.hk) * D * 2;
            try view.fill(&c, ls.kv.k + ((at - ring_keep) % ls.kv.slots) * row, row, byte);
        };
    } else if (std.mem.eql(u8, v, "conv") and at >= conv.state_rows) {
        for (st.layers) |ls| try view.fill(&c, ls.conv_mlp, H * 2, byte);
    } else if (std.mem.eql(u8, v, "draft")) {
        // on pages positions [0, 16) are the first rows of the slot's first page
        if (e.drafter) |dr| if (dr.caches.items[slot]) |cs| try view.fill(&c, if (cs.pg) |pg| pg.k + @as(u64, st.ids[0]) * pg.page * 2 else cs.keys(2), @min(at, 16) * cs.row, 0x47); // bf16 51,000: large, finite
    }
}

/// Tests: every byte of slot `slot`'s state (and its drafter caches) set to `byte`; not its rollback table (the
/// slot's own pointers, made once at its start), nor its first `shared` pages (a kept state's, adopted: other
/// streams and states read them).
pub fn poison(e: *Engine, slot: u32, byte: u8, shared: usize) !void {
    const st = e.slots.items[slot] orelse return error.UnknownSlot;
    for (st.own.items) |b| if (b.ptr != st.keep_table and b.ptr != st.table) try b.fill8(byte, e.stream.handle);
    // on the pool: the slot's own mapped pages in every region
    if (e.pool) |p| if (st.paged() and st.mapped > shared) try pages.fillIn(e.d, e.stream, p.place(), p.regions, st.ids[shared..st.mapped], byte);
    if (e.drafter) |dr| if (dr.caches.items[slot]) |cs| try cs.buf.fill8(byte, e.stream.handle);
}

/// A restore's test flag (Header.head_lo): poison the slot with the low byte first.
const poison_flag: u32 = 0x100;

/// What rank 0 checks before a save's header: the slot's forwards ended at `at` and none is rolled back pending.
pub fn savable(e: *Engine, slot: u32, at: usize) !*state.State {
    if (slot >= e.slots.items.len) return error.UnknownSlot;
    const st = e.slots.items[slot] orelse return error.UnknownSlot;
    if (at == 0 or st.pos != at or st.last_pos0 + st.last_rows != st.pos or st.last_rows == 0) return error.NotAtMark;
    return st;
}

const Fault = struct {
    var restores: u64 = 0;
    var tags: u64 = 0;

    /// TF_INK2_SNAP_FAULT=`name`:N names this call (the N-th of its kind).
    fn hit(comptime name: []const u8, count: *u64) bool {
        const v = std.mem.span(std.c.getenv("TF_INK2_SNAP_FAULT") orelse return false);
        const colon = std.mem.indexOfScalar(u8, v, ':') orelse return false;
        if (!std.mem.eql(u8, v[0..colon], name)) return false;
        const n = std.fmt.parseInt(u64, v[colon + 1 ..], 10) catch return false;
        count.* += 1;
        if (count.* != n) return false;
        std.log.warn("inkling2: TF_INK2_SNAP_FAULT {s}:{d} induced", .{ name, n });
        return true;
    }
};

/// Rank 0, start-up: the arena of `budget` bytes less the capture area (rounded down to a whole number of slots) on
/// every rank; the other ranks sleep again after. Returns the arena's bytes (the store's budget). The window layout
/// (TF_INK2_PAGES=0) keeps none: 0.
pub fn arena(x: Leader, budget: u64) !u64 {
    if (!x.e.paged) {
        std.log.info("inkling2: no prompt cache: the window layout (TF_INK2_PAGES=0) keeps no prompts (they live on the KV page pool)", .{});
        return 0;
    }
    const slot = bytes(x.e, 0);
    const n = (budget -| capBytes(x.e)) / slot * slot;
    if (n == 0) return 0;
    try arenaInit(x.e, n);
    if (x.e.o.world > 1) {
        try x.send(.{ .kind = @intFromEnum(Kind.arena), .pos0 = n });
        try x.idle();
    }
    return n; // the store's budget: the capture area and the bytes past the last granule are set aside
}

/// Rank 0: slot `slot`'s state at `at` kept into `k`, on every rank (`k` stays registered until `drop`): its whole
/// pages referenced (rank 0's count), its fixed part and tail rows copied into a free slot of the arena;
/// error.CacheFull before any rank moves.
pub fn save(x: Leader, slot: u32, at: usize, k: *Kept) !void {
    const e = x.e;
    const a = &e.snap;
    if (a.buf == null) return error.NoArena;
    const p = e.pool orelse return error.CacheNeedsPages;
    const al = if (p.alloc) |*al| al else return error.NotLeader;
    const ci = capturedAt(e, slot, at);
    const st = if (ci == null) try savable(e, slot, at) else e.slots.items[slot].?;
    if (at > st.limit()) return error.PastPages;
    const n = bytes(e, at);
    const off = a.extents.take(n) orelse return error.CacheFull;
    errdefer a.extents.give(off, n);
    const held = try e.gpa.dupe(u32, st.ids[0 .. at / pages.page_rows]);
    errdefer e.gpa.free(held);
    try a.live.ensureUnusedCapacity(e.gpa, 1);
    a.tag +%= 1;
    if (a.tag == 0) a.tag = 1;
    k.* = .{ .tag = a.tag, .at = @intCast(at), .off = off, .bytes = n, .pages = held };
    try x.send(.{ .kind = @intFromEnum(Kind.save), .slot = slot, .rows = k.at, .pos0 = off, .token = k.tag, .head_lo = if (ci) |i| cap_flag | @as(u32, @intCast(i)) else 0 });
    copyOutFrom(e, slot, at, off, ci) catch |err| return x.stop(err);
    for (held) |id| al.retain(id);
    if (ci != null) a.captured += 1;
    a.note(off, n, k.tag, k.at);
    a.live.appendAssumeCapacity(k);
    a.saves += 1;
}

/// What rank 0 checks before a restore's header: the extent still holds `k`, the slot's table starts with its pages
/// (adopted: Pool.open), and the slot can stand at its position.
pub fn restorable(e: *Engine, slot: u32, k: Kept) !void {
    if (!e.snap.holds(k.off, k.tag, k.at)) return error.SnapshotStale;
    const st = try e.bind(slot);
    if (k.at == 0 or k.at >= st.context) return error.SnapshotNotPrefix;
    if (k.at >= st.limit()) return error.PastPages; // on the pool: the slot's stream holds no page past the state
    if (st.mapped < k.pages.len or !std.mem.eql(u32, st.ids[0..k.pages.len], k.pages)) return error.NotAdopted;
    if (Fault.hit("restore", &Fault.restores)) return error.InducedFault;
}

/// Rank 0: slot `slot` set to kept state `k` on every rank (its next forward starts at k.at): its table must start
/// with the state's pages (the lanes' open adopts them; tooling on the tool pool: adopted here, with fresh pages to
/// the slot's whole context). `poison_byte` (tests): the slot's every byte but the adopted pages set to it first.
pub fn restore(x: Leader, slot: u32, k: Kept, poison_byte: ?u8) !void {
    if (x.e.tool_pool) {
        if (!x.e.snap.holds(k.off, k.tag, k.at)) return error.SnapshotStale;
        _ = try x.openPages(slot, x.e.pool.?.entries, k.pages);
    }
    try restorable(x.e, slot, k);
    const h: Header = .{ .kind = @intFromEnum(Kind.restore), .slot = slot, .rows = k.at, .pos0 = k.off, .token = k.tag, .head_lo = if (poison_byte) |b| poison_flag | b else 0 };
    try x.send(h);
    apply(x.e, h) catch |err| return x.stop(err);
}

fn apply(e: *Engine, h: Header) !void {
    _ = try e.bind(h.slot); // a follower's slot may not have run yet
    if (h.head_lo & poison_flag != 0) try poison(e, h.slot, @truncate(h.head_lo), h.rows / pages.page_rows);
    try copyIn(e, h.slot, h.rows, h.pos0);
    e.snap.restores += 1;
}

/// Rank 0: kept state `k` dropped: its extent free for later saves, its pages' references gone (a page no stream
/// holds is free again). Nothing is sent: a follower's record of it is overwritten by the next save there, and a
/// restore is only ever sent for a state rank 0 holds.
pub fn drop(x: Leader, k: *Kept) void {
    const e = x.e;
    const a = &e.snap;
    a.forget(k.off);
    a.extents.give(k.off, k.bytes);
    if (e.pool) |p| if (p.alloc) |*al| {
        for (k.pages) |id| al.release(id);
        p.show(); // /health's counts (B3): an evicted state's pages may be free now
    };
    if (k.pages.len > 0) e.gpa.free(k.pages);
    k.pages = &.{};
    for (a.live.items, 0..) |q, i| if (q == k) {
        _ = a.live.swapRemove(i);
        break;
    };
}

/// Rank 0: the pages of kept states the reclaim chain may not drop (`borrowed`: a resume in flight), for
/// Pool.reclaimable; at most `out.len`.
pub fn protectedPages(e: *const Engine, ctx_: *anyopaque, borrowed: *const fn (*anyopaque, *anyopaque) bool, out: [][]const u32) [][]const u32 {
    var n: usize = 0;
    for (e.snap.live.items) |k| {
        if (n == out.len) break;
        if (borrowed(ctx_, k)) {
            out[n] = k.pages;
            n += 1;
        }
    }
    return out[0..n];
}

/// A rank other than 0: the save, restore, arena and capture steps.
pub fn follow(e: *engine.Engine, h: Header) !void {
    switch (std.enums.fromInt(Kind, h.kind) orelse return error.BadStep) {
        .arena => try arenaInit(e, h.pos0),
        .capture => setCapture(e, h),
        .save => {
            const n = bytes(e, h.rows);
            try copyOutFrom(e, h.slot, h.rows, h.pos0, if (h.head_lo & cap_flag != 0) @as(usize, h.head_lo & 0xff) else null);
            e.snap.note(h.pos0, n, h.token, h.rows);
            e.snap.saves += 1;
        },
        .restore => {
            const tag = if (Fault.hit("tag", &Fault.tags)) h.token +% 1 else h.token;
            if (!e.snap.holds(h.pos0, tag, h.rows)) {
                std.log.err("inkling2 rank {d}: rank 0 restores save {d} ({d} tokens) at arena offset {d}, which this rank does not hold", .{ e.o.rank, h.token, h.rows, h.pos0 });
                return error.SnapTagMismatch;
            }
            try apply(e, h);
        },
        else => return error.BadStep,
    }
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// Every field of a slot's state is either copied (state) or rewritten before it is read again (scratch), named
/// here; a new field fails the build until it is classified (a resume must not silently miss state).
const Census = struct { state: []const []const u8, scratch: []const []const u8 };

const census_state: Census = .{
    .state = &.{ "layers", "pos", "last_pos0", "last_rows", "kept_inputs" },
    // the slot's allocation, the same in every slot (keep_table: its rollback table, of this slot's buffers); the last
    // forward's taps row and count, rewritten by the next forward (a restore leaves no taps to draft from)
    // place, table, ids, mapped: where the slot's global layers live (on pages: the stream's own pages, mapped when it
    // opened; a state's rows are copied into them, never its page ids); run, ring_on_run, last_one, moves, ones, flip:
    // whether those pages are one run and which kernels a one-row forward takes on them (the slot's, not the state's)
    .scratch = &.{ "gpa", "first", "own", "bytes", "context", "keep_table", "keep_entries", "last_row0", "last_fwd", "place", "table", "ids", "mapped", "run", "ring_on_run", "last_one", "moves", "ones", "flip" },
};
const census_layer: Census = .{
    .state = &.{ "kv", "conv_k", "conv_v", "conv_attn", "conv_mlp" },
    // one verify window's convolution inputs, rewritten by the next forward (a restore sets last_rows = 0: no keep)
    .scratch = &.{"hist"},
};
const census_caches: Census = .{
    .state = &.{"buf"},
    // the slot's shape: positions, a local layer's ring (mirrored below dspark.window), bytes a position; on pages
    // where its global layer lives (the slot's pages, walked as the target's global layers are)
    .scratch = &.{ "context", "ring", "row", "pg" },
};

fn censusCheck(comptime T: type, comptime c: Census) void {
    @setEvalBranchQuota(10000);
    inline for (@typeInfo(T).@"struct".field_names) |name| {
        var known = false;
        for (c.state) |n| known = known or std.mem.eql(u8, n, name);
        for (c.scratch) |n| known = known or std.mem.eql(u8, n, name);
        if (!known) @compileError("inkling2 prompt cache: " ++ @typeName(T) ++ "." ++ name ++ " is neither copied nor named scratch (cuda_snap.zig census)");
    }
}

comptime {
    censusCheck(state.State, census_state);
    censusCheck(state.LayerState, census_layer);
    censusCheck(dspark.Caches, census_caches);
}

test {
    _ = @import("cuda_snap_test.zig");
}
