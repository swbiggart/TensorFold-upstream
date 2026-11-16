//! Inkling-2's KV page pool (v0.2 stage 2b): the global layers' keys and values of every served stream in one arena a
//! rank, cut into pages of `page_rows` positions (one attention split), and rank 0's allocator of those pages.
//! Sliding layers keep their rings outside the pool (and the drafter its own caches until stage 2c); a slot keeps its
//! fixed part (rings, convolution states, the rollback history) and a page table at a fixed address, so no captured
//! graph is recaptured when a stream's pages change.
//!
//! The arena is laid out by region (the bench's `soa`): region 2g holds global layer g's keys, region 2g + 1 its
//! values, each `pages` runs of page_rows x row bytes; page id i is run i of every region, and the last page is the
//! scrub page that unmapped table entries name (never handed out). A stream on consecutive page ids therefore has
//! each layer's rows in one contiguous run, as a v0.1 window did (G2 on GB10: scattered ids cost one-row attention
//! 3-8 %, consecutive ones within 0.5 % at 32k-131k).
//!
//! The allocator (rank 0 owns the pool; the other ranks map the ids its `pages` step sends, cuda_follow.zig):
//! - admission promises a request's pages (its prompt, max_tokens and one verify window) or it waits (`fits`);
//! - a stream's pages are taken as ONE run of consecutive ids when a free run holds them: the smallest free run that
//!   does (best fit, the lowest on a tie), from its start;
//! - only when no free run holds them are they taken scattered, from the longest free runs first (fewest pieces);
//!   each such cover is counted and logged as a fallback;
//! - growth (`grow`: a stream that needs more pages than it holds) takes the pages right after its last one when
//!   they are free (in place), else a best-fit run of the rest, else scattered (a fallback).
//! Pages go back when the stream ends. The counts an idle server must show (`census`): nothing promised, no stream's
//! page taken, every page free or a kept state's (free + kept = total).
//! The prompt cache on pages (stage 2d, cuda_snap.zig): a kept state at N holds a reference to the stream's whole
//! pages [0, N / page_rows) (rank 0's count a page: `retain` / `release`; the kept states hold a page once in the
//! pool, however many share it), never written again; a stream resuming it adopts those pages (`open` with `adopt`:
//! one more holder each, first in its table) and takes fresh pages after them, in place after the last adopted one
//! when they are free (the consecutive-run rule), else a best-fit run, else scattered.
//! TF_KV_POISON=1 (tests, every rank): a stream's pages are set to NaN when they are mapped, so a row read before it
//! is written shows in the output.
const std = @import("std");
const cuda = @import("cuda");
const attn = @import("cuda_attn.zig");

/// Positions a page holds: one attention split, so a block reads its page id once a split.
pub const page_rows: usize = attn.split_keys;

/// Pages `positions` rows take.
pub fn pagesFor(positions: usize) usize {
    return (positions + page_rows - 1) / page_rows;
}

/// TF_INK2_PAGES (tooling, every rank; part of the settings the ranks agree on): `0` keeps every global layer in a
/// window a slot (v0.1's layout, for A/B measurements only: the pool is the one serving layout, and a server started
/// with it says so); anything else (or unset) puts them on pages.
pub fn enabled() bool {
    const v = std.c.getenv("TF_INK2_PAGES") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

/// TF_INK2_RUN_RING (every rank; agreed): `0` gives every forward of a stream on pages the paged kernels; anything
/// else (or unset) gives a stream's one-row forwards the v0.1 ring kernels on its pages when they are one run of
/// consecutive ids (a contiguous window of each region), as fast as v0.1's windows.
pub fn ringOnRun() bool {
    const v = std.c.getenv("TF_INK2_RUN_RING") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

/// TF_KV_POISON=1: new pages start as NaN (G4).
pub fn poisoned() bool {
    const v = std.c.getenv("TF_KV_POISON") orelse return false;
    return v[0] == '1' and v[1] == 0;
}

/// TF_INK2_PAGE_ORDER (tooling, a slot with its own pages): `shuffle` gives its table a pseudo-random permutation of
/// its pages (the worst case for locality), anything else consecutive ids.
pub fn shuffled() bool {
    const v = std.c.getenv("TF_INK2_PAGE_ORDER") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "shuffle");
}

/// TF_INK2_PAGE_ORDER=move (tooling, a slot with its own pages): an arena of two contexts, each new sequence on
/// another run of consecutive ids (a captured step replayed on a moved run), and every other 5 one-row forwards on
/// the page table instead of the run (mode switches mid-stream).
pub fn moving() bool {
    const v = std.c.getenv("TF_INK2_PAGE_ORDER") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "move");
}

/// Where a slot's global layers live: an arena of `pages` runs a region (the scrub page last), `run` bytes a run.
pub const Place = struct {
    base: u64,
    pages: usize,
    run: u64,

    /// Region r's first byte (2g: global layer g's keys, 2g + 1 its values).
    pub fn region(p: Place, r: usize) u64 {
        return p.base + @as(u64, r) * p.pages * p.run;
    }

    /// The scrub page: what an unmapped table entry names.
    pub fn scrub(p: Place) u32 {
        return @intCast(p.pages - 1);
    }

    /// The page stride as the paged kernels take it (Kv.page): elements of 2 bytes.
    pub fn elems(p: Place) usize {
        return @intCast(p.run / 2);
    }
};

/// The device bytes of an arena of `pages` pages (the scrub page included) for `regions` runs of `run` bytes.
pub fn arenaBytes(pages: usize, regions: usize, run: u64) u64 {
    return @as(u64, pages) * regions * run;
}

/// A maximal run of free pages.
pub const Run = struct { start: u32, len: u32 };

/// A stream's reservation: pages promised and not yet taken, and the pages it holds in position order (its table).
pub const Lease = struct {
    promised: usize = 0,
    ids: std.ArrayList(u32) = .empty,
};

/// What an idle pool must show: nothing promised, no stream's page taken, every page free or a kept state's (`kept`:
/// pages kept states hold; `taken`: pages any holder holds).
pub const Census = struct { total: usize, free: usize, promised: usize, taken: usize, kept: usize = 0 };

/// Rank 0's counts for /health (B3), in pages: the pool, free, free and not promised, kept by the prompt cache, the
/// need of a request held back for pages (0: none) and the requests held back since start.
pub const Shown = struct { pages: usize = 0, free: usize = 0, available: usize = 0, kept: usize = 0, waiting: usize = 0, waits: u64 = 0 };

fn shownOf(a: *const Alloc, waiting: ?usize, waits: u64) Shown {
    const c = a.census();
    return .{
        .pages = c.total,
        .free = c.free,
        .available = a.available(),
        .kept = if (@hasField(Census, "kept")) c.kept else 0, // the prompt cache on pages (stage 2d)
        .waiting = waiting orelse 0,
        .waits = waits,
    };
}

/// How streams' pages were placed since start.
pub const Stats = struct {
    covers: u64 = 0, // streams' first covers
    one_run: u64 = 0, // of those, one run of consecutive ids
    fallbacks: u64 = 0, // covers and growths taken scattered (no free run held them)
    fallback_pieces: u64 = 0, // the runs those took
    grown: u64 = 0, // growths
    grown_in_place: u64 = 0, // of those, the pages right after the stream's last
    grown_run: u64 = 0, // of those, a best-fit run elsewhere
    resumed: u64 = 0, // streams that adopted a kept state's pages (the prompt cache on pages)
    resumed_pages: u64 = 0, // the pages they adopted
    resumed_one_run: u64 = 0, // of those streams, a table of one run (adopted and fresh pages consecutive)
};

/// Rank 0's page allocator over `total` pages (ids 0 .. total - 1; the arena's scrub page is not one of them).
pub const Alloc = struct {
    gpa: std.mem.Allocator,
    refs: []u32, // holders a page (0: free); the kept states count as one holder of a page they hold
    kept: []u32, // kept states holding a page (the prompt cache on pages)
    nfree: usize,
    nkept: usize = 0, // pages with kept > 0
    promised: usize = 0,
    stats: Stats = .{},
    runs: std.ArrayList(Run) = .empty, // scratch for the scattered fallback

    pub fn init(gpa: std.mem.Allocator, count: usize) !Alloc {
        const refs = try gpa.alloc(u32, count);
        errdefer gpa.free(refs);
        @memset(refs, 0);
        const kept = try gpa.alloc(u32, count);
        @memset(kept, 0);
        return .{ .gpa = gpa, .refs = refs, .kept = kept, .nfree = count };
    }

    pub fn deinit(a: *Alloc) void {
        a.gpa.free(a.refs);
        a.gpa.free(a.kept);
        a.runs.deinit(a.gpa);
        a.* = undefined;
    }

    pub fn total(a: *const Alloc) usize {
        return a.refs.len;
    }

    /// Pages free and not promised.
    pub fn available(a: *const Alloc) usize {
        return a.nfree - a.promised;
    }

    /// Whether `n` more pages could be promised now.
    pub fn fits(a: *const Alloc, n: usize) bool {
        return a.available() >= n;
    }

    pub fn census(a: *const Alloc) Census {
        return .{ .total = a.total(), .free = a.nfree, .promised = a.promised, .taken = a.total() - a.nfree, .kept = a.nkept };
    }

    /// Pages a stream holds (more holders than the kept states' one).
    pub fn leased(a: *const Alloc) usize {
        var n: usize = 0;
        for (a.refs, a.kept) |r, k| n += @intFromBool(r > @intFromBool(k > 0));
        return n;
    }

    /// Nothing promised and no stream's page: every page free or a kept state's (free + kept = total).
    pub fn idle(a: *const Alloc) bool {
        return a.promised == 0 and a.leased() == 0 and a.nfree + a.nkept == a.total();
    }

    /// A kept state takes a reference to page `id`, a stream's (taken): the kept states hold it once in the pool.
    pub fn retain(a: *Alloc, id: u32) void {
        if (a.refs[id] == 0) std.debug.panic("inkling2 pages: page {d} kept while free", .{id});
        if (a.kept[id] == 0) {
            a.refs[id] += 1;
            a.nkept += 1;
        }
        a.kept[id] += 1;
    }

    /// A kept state's reference to page `id` goes; the page is free once nothing holds it.
    pub fn release(a: *Alloc, id: u32) void {
        if (a.kept[id] == 0) std.debug.panic("inkling2 pages: page {d} released by a kept state that does not hold it", .{id});
        a.kept[id] -= 1;
        if (a.kept[id] > 0) return;
        a.nkept -= 1;
        a.refs[id] -= 1;
        if (a.refs[id] == 0) a.nfree += 1;
    }

    /// The pages kept states hold that no stream does and no state in `protect` holds (the reclaim chain could free
    /// them by dropping the other states); `mark`: scratch, a byte a page.
    pub fn reclaimable(a: *const Alloc, protect: []const []const u32, mark: []u8) usize {
        if (a.nkept == 0) return 0;
        @memset(mark, 0);
        for (protect) |ids| for (ids) |id| {
            mark[id] = 1;
        };
        var n: usize = 0;
        for (a.refs, a.kept, mark) |r, k, m| n += @intFromBool(k > 0 and r == 1 and m == 0);
        return n;
    }

    /// Whether a kept state holds any of `ids`.
    pub fn anyKept(a: *const Alloc, ids: []const u32) bool {
        for (ids) |id| if (a.kept[id] > 0) return true;
        return false;
    }

    /// `l` (its table empty) adopts pages `ids` (a kept state's, taken): one more holder each, first in its table.
    pub fn adopt(a: *Alloc, l: *Lease, ids: []const u32) !void {
        if (l.ids.items.len != 0) return error.LeaseTaken;
        try l.ids.ensureUnusedCapacity(a.gpa, ids.len);
        for (ids) |id| {
            if (id >= a.refs.len or a.refs[id] == 0) return error.AdoptFree;
            a.refs[id] += 1;
            l.ids.appendAssumeCapacity(id);
        }
    }

    /// `n` more pages promised to `l`; LaneMemory when they are not available now.
    pub fn promise(a: *Alloc, l: *Lease, n: usize) error{LaneMemory}!void {
        if (a.available() < n) return error.LaneMemory;
        a.promised += n;
        l.promised += n;
    }

    /// What `l` was promised and has not taken goes back.
    pub fn unpromise(a: *Alloc, l: *Lease) void {
        a.promised -= l.promised;
        l.promised = 0;
    }

    /// The smallest free run of at least `n` pages (the lowest on a tie), or null.
    pub fn bestFit(a: *const Alloc, n: usize) ?Run {
        if (n == 0) return null;
        var best: ?Run = null;
        var i: usize = 0;
        while (i < a.refs.len) {
            if (a.refs[i] != 0) {
                i += 1;
                continue;
            }
            const s = i;
            while (i < a.refs.len and a.refs[i] == 0) i += 1;
            const len = i - s;
            if (len >= n and (best == null or len < best.?.len)) {
                best = .{ .start = @intCast(s), .len = @intCast(len) };
                if (len == n) break; // exact: nothing smaller can hold n
            }
        }
        return best;
    }

    /// Page `id` taken by `l` from its promise.
    fn takeId(a: *Alloc, l: *Lease, id: u32) void {
        std.debug.assert(a.refs[id] == 0 and l.promised > 0);
        a.refs[id] = 1;
        a.nfree -= 1;
        a.promised -= 1;
        l.promised -= 1;
        l.ids.appendAssumeCapacity(id);
    }

    /// Pages [start, start + n) taken by `l`.
    fn takeRun(a: *Alloc, l: *Lease, start: u32, n: usize) void {
        for (0..n) |k| a.takeId(l, start + @as(u32, @intCast(k)));
    }

    /// `n` pages for `l` as one best-fit run when one holds them, else scattered from the longest free runs first
    /// (a fallback). Returns whether one run held them.
    fn place(a: *Alloc, l: *Lease, n: usize) !bool {
        if (n == 0) return true;
        if (l.promised < n) return error.PastReservation;
        try l.ids.ensureUnusedCapacity(a.gpa, n);
        if (a.bestFit(n)) |r| {
            a.takeRun(l, r.start, n);
            return true;
        }
        // no free run holds n: the free runs, longest first (the lowest start on a tie), until n are taken
        a.runs.clearRetainingCapacity();
        var i: usize = 0;
        while (i < a.refs.len) {
            if (a.refs[i] != 0) {
                i += 1;
                continue;
            }
            const s = i;
            while (i < a.refs.len and a.refs[i] == 0) i += 1;
            try a.runs.append(a.gpa, .{ .start = @intCast(s), .len = @intCast(i - s) });
        }
        std.mem.sort(Run, a.runs.items, {}, longerFirst);
        var left = n;
        var pieces: u64 = 0;
        for (a.runs.items) |r| {
            if (left == 0) break;
            const k = @min(left, r.len);
            a.takeRun(l, r.start, k);
            left -= k;
            pieces += 1;
        }
        if (left > 0) return error.PastReservation; // a promise always has its pages free
        a.stats.fallbacks += 1;
        a.stats.fallback_pieces += pieces;
        return false;
    }

    fn longerFirst(_: void, x: Run, y: Run) bool {
        return x.len > y.len or (x.len == y.len and x.start < y.start);
    }

    /// `l`'s first `n` pages (its table empty), from its promise: one run when one holds them (see the top).
    pub fn cover(a: *Alloc, l: *Lease, n: usize) !void {
        if (l.ids.items.len != 0) return error.LeaseTaken;
        a.stats.covers += 1;
        const one = try a.place(l, n);
        if (one) a.stats.one_run += 1;
    }

    /// `k` more pages after `l`'s last, from its promise: in place when the pages right after its last one are free,
    /// else a best-fit run of the k, else scattered (a fallback).
    pub fn grow(a: *Alloc, l: *Lease, k: usize) !void {
        if (k == 0) return;
        if (l.ids.items.len == 0) return a.cover(l, k);
        if (l.promised < k) return error.PastReservation;
        a.stats.grown += 1;
        const last = l.ids.items[l.ids.items.len - 1];
        const from: usize = @as(usize, last) + 1;
        if (from + k <= a.refs.len and std.mem.allEqual(u32, a.refs[from..][0..k], 0)) {
            try l.ids.ensureUnusedCapacity(a.gpa, k);
            a.takeRun(l, @intCast(from), k);
            a.stats.grown_in_place += 1;
            return;
        }
        if (try a.place(l, k)) a.stats.grown_run += 1;
    }

    /// The stream ended: its promise and its pages go back (a page a kept state holds stays with it).
    pub fn finish(a: *Alloc, l: *Lease) void {
        a.unpromise(l);
        for (l.ids.items) |id| {
            if (a.refs[id] == 0) std.debug.panic("inkling2 pages: page {d} given back while free", .{id});
            a.refs[id] -= 1;
            if (a.refs[id] == 0) a.nfree += 1;
        }
        l.ids.deinit(a.gpa);
        l.* = .{};
    }

    /// The runs of consecutive ids in `ids` (a stream's table), for the log.
    pub fn runsOf(ids: []const u32) usize {
        if (ids.len == 0) return 0;
        var n: usize = 1;
        for (ids[1..], ids[0 .. ids.len - 1]) |b, p| n += @intFromBool(b != p + 1);
        return n;
    }
};

/// Pages a request takes: its prompt, its reply and one verify window (`window` rows past its last token), at most
/// the slot's table (`entries`).
pub fn need(prompt: usize, max_new: usize, window: usize, entries: usize) usize {
    return @min(pagesFor(prompt + max_new + window), entries);
}

/// The served streams' pool on one rank: the arena (every rank), and on rank 0 the allocator and each slot's lease.
pub const Pool = struct {
    gpa: std.mem.Allocator,
    arena: cuda.DeviceBuffer,
    place_: Place,
    regions: usize,
    slots: u32,
    entries: usize, // a slot's table: pages(context)
    leader: bool,
    alloc: ?Alloc = null, // rank 0's
    leases: []Lease = &.{}, // rank 0's, a slot each
    mark: []u8 = &.{}, // rank 0's scratch, a byte a page (`reclaimable`)
    poison: bool,
    /// Rank 0: requests the lane host held back for want of pages (each counted once), and the need of the last one.
    waits: u64 = 0,
    waiting: ?usize = null,
    /// Rank 0: the counts /health reads (`shownNow`, any thread), copied by the engine thread (`show`) at each open,
    /// close and admission check, under `shown_lock`.
    shown: Shown = .{},
    shown_lock: std.atomic.Value(bool) = .init(false),

    /// `total` pages (and the scrub page) for `slots` slots of `entries` table entries, `regions` runs of `run` bytes.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, count: usize, regions: usize, run: u64, slots: u32, entries: usize, leader: bool) !*Pool {
        if (count == 0 or regions == 0 or run == 0 or slots == 0) return error.BadPool;
        if (count >= std.math.maxInt(i32)) return error.BadPool;
        const p = try gpa.create(Pool);
        errdefer gpa.destroy(p);
        var arena = try cuda.DeviceBuffer.alloc(d, @intCast(arenaBytes(count + 1, regions, run)));
        errdefer arena.free();
        p.* = .{ .gpa = gpa, .arena = arena, .place_ = .{ .base = arena.ptr, .pages = count + 1, .run = run }, .regions = regions, .slots = slots, .entries = entries, .leader = leader, .poison = poisoned() };
        // the scrub page: zeros (NaN under TF_KV_POISON); never written by a forward, never read
        for (0..regions) |r| {
            const view: cuda.DeviceBuffer = .{ .d = d, .ptr = p.place_.region(r) + count * run, .len = @intCast(run), .borrowed = true };
            try view.fill8(if (p.poison) 0xff else 0, null);
        }
        if (leader) {
            p.alloc = try Alloc.init(gpa, count);
            errdefer p.alloc.?.deinit();
            p.leases = try gpa.alloc(Lease, slots);
            errdefer gpa.free(p.leases);
            @memset(p.leases, .{});
            p.mark = try gpa.alloc(u8, count);
            p.show();
        }
        return p;
    }

    /// Rank 0's engine thread, after a change: the allocator's counts and the admission's wait into `shown`.
    pub fn show(p: *Pool) void {
        const s = shownOf(&(p.alloc orelse return), p.waiting, p.waits);
        while (p.shown_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        p.shown = s;
        p.shown_lock.store(false, .release);
    }

    /// Any thread: the counts `show` last copied.
    pub fn shownNow(p: *Pool) Shown {
        while (p.shown_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        const s = p.shown;
        p.shown_lock.store(false, .release);
        return s;
    }

    pub fn deinit(p: *Pool) void {
        if (p.alloc) |*a| {
            for (p.leases) |*l| a.finish(l);
            a.deinit();
        }
        if (p.leases.len > 0) p.gpa.free(p.leases);
        if (p.mark.len > 0) p.gpa.free(p.mark);
        p.arena.free();
        p.gpa.destroy(p);
    }

    pub fn place(p: *const Pool) Place {
        return p.place_;
    }

    /// Pages handed out (the scrub page not counted).
    pub fn total(p: *const Pool) usize {
        return p.place_.pages - 1;
    }

    /// Rank 0: whether a request needing `n` pages fits now.
    pub fn fits(p: *const Pool, n: usize) bool {
        const a = p.alloc orelse return false;
        return a.fits(n);
    }

    /// Rank 0: slot `slot` takes `n` pages (promised and covered: one run when one holds them); its ids, in position
    /// order. `adopt`: a kept state's pages (the prompt cache on pages), the first of the n, adopted (one more holder
    /// each) and the rest taken after them (`grow`: in place when free). LaneMemory: not now (the request waits);
    /// RequestOverPool: never (more than the pool).
    pub fn open(p: *Pool, slot: u32, n: usize, adopt: []const u32) ![]const u32 {
        const a = &(p.alloc orelse return error.NotLeader);
        defer p.show();
        if (slot >= p.slots) return error.NoSlot;
        if (n == 0 or n > p.entries or adopt.len > n) return error.PastContext;
        if (n > a.total()) return error.RequestOverPool;
        const l = &p.leases[slot];
        const fb = a.stats.fallbacks;
        if (adopt.len > 0) {
            // the stream resumes a kept state: what the slot held goes back (a kept page stays with its state)
            a.finish(l);
            try a.promise(l, n - adopt.len);
            errdefer a.finish(l);
            try a.adopt(l, adopt);
            try a.grow(l, n - adopt.len);
            a.stats.resumed += 1;
            a.stats.resumed_pages += adopt.len;
            if (Alloc.runsOf(l.ids.items) == 1) a.stats.resumed_one_run += 1;
            if (a.stats.fallbacks > fb) logFallback(p, slot, l.ids.items);
            return l.ids.items;
        }
        if (l.ids.items.len > 0 and a.anyKept(l.ids.items)) a.finish(l); // a kept state holds some: never rewritten
        if (l.ids.items.len > 0) {
            // the slot's stream opens again (its pages kept): what it holds serves, grown past it when it needs more
            if (l.ids.items.len >= n) return l.ids.items;
            try a.promise(l, n - l.ids.items.len);
            errdefer a.unpromise(l);
            try a.grow(l, n - l.ids.items.len);
            if (a.stats.fallbacks > fb) logFallback(p, slot, l.ids.items);
            return l.ids.items;
        }
        a.unpromise(l);
        try a.promise(l, n);
        errdefer a.finish(l);
        try a.cover(l, n);
        if (a.stats.fallbacks > fb) logFallback(p, slot, l.ids.items);
        return l.ids.items;
    }

    fn logFallback(p: *const Pool, slot: u32, ids: []const u32) void {
        if (@import("builtin").is_test) return;
        const a = p.alloc.?;
        std.log.warn("inkling2 pages: slot {d}'s {d} pages taken scattered in {d} runs (no free run held them; {d} free in all): fallback {d} since start", .{ slot, ids.len, Alloc.runsOf(ids), a.nfree, a.stats.fallbacks });
    }

    /// Rank 0: the stream on slot `slot` ended; its pages go back. Logs the idle census when no slot holds any.
    pub fn close(p: *Pool, slot: u32) void {
        const a = &(p.alloc orelse return);
        if (slot >= p.slots) return;
        a.finish(&p.leases[slot]);
        p.show();
        for (p.leases) |l| if (l.ids.items.len > 0 or l.promised > 0) return;
        if (@import("builtin").is_test) return;
        const c = a.census();
        const s = a.stats;
        std.log.info("inkling2 pages idle{s}: promised {d}, taken {d} (kept states' {d}), free {d} + kept {d} == total {d}; streams' covers {d}, one run {d}, scattered (fallback) {d} in {d} runs, grown {d} ({d} in place); resumed on kept pages {d} ({d} pages adopted, {d} tables one run); requests that waited for pages {d}", .{ if (a.idle()) "" else " (UNEVEN)", c.promised, c.taken, c.kept, c.free, c.kept, c.total, s.covers, s.one_run, s.fallbacks, s.fallback_pieces, s.grown, s.grown_in_place, s.resumed, s.resumed_pages, s.resumed_one_run, p.waits });
    }

    /// Rank 0: the pages kept states hold that no stream does and no state in `protect` holds (the reclaim chain could
    /// free them by dropping the other states).
    pub fn reclaimable(p: *Pool, protect: []const []const u32) usize {
        const a = &(p.alloc orelse return 0);
        return a.reclaimable(protect, p.mark);
    }

    /// Every rank: `ids`' pages set to NaN in every region (TF_KV_POISON), on `stream`.
    pub fn poisonPages(p: *const Pool, d: *const cuda.Driver, stream: cuda.Stream, ids: []const u32) !void {
        if (!p.poison) return;
        try poisonIn(d, stream, p.place_, p.regions, ids);
    }
};

/// `ids`' pages of the arena at `pl` set to NaN in each of `regions` (a run of consecutive ids at once).
pub fn poisonIn(d: *const cuda.Driver, stream: cuda.Stream, pl: Place, regions: usize, ids: []const u32) !void {
    return fillIn(d, stream, pl, regions, ids, 0xff);
}

/// `ids`' pages of the arena at `pl` set to `byte` in each of `regions`, on `stream` (tests and poison).
pub fn fillIn(d: *const cuda.Driver, stream: cuda.Stream, pl: Place, regions: usize, ids: []const u32, byte: u8) !void {
    var i: usize = 0;
    while (i < ids.len) {
        var j = i + 1;
        while (j < ids.len and ids[j] == ids[j - 1] + 1) j += 1;
        for (0..regions) |r| {
            const at = pl.region(r) + @as(u64, ids[i]) * pl.run;
            try d.check(d.api.cuMemsetD8Async(at, byte, @intCast((j - i) * pl.run), stream.handle), "cuMemsetD8Async");
        }
        i = j;
    }
}

test "inkling2 pages: a stream takes one best-fit run, the lowest on a tie" {
    const gpa = std.testing.allocator;
    var a = try Alloc.init(gpa, 32);
    defer a.deinit();
    var l: [4]Lease = @splat(.{});
    defer for (&l) |*x| a.finish(x);
    try a.promise(&l[0], 8);
    try a.cover(&l[0], 8);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, l[0].ids.items);
    try a.promise(&l[1], 4);
    try a.cover(&l[1], 4);
    try std.testing.expectEqual(@as(u32, 8), l[1].ids.items[0]);
    try a.promise(&l[2], 10);
    try a.cover(&l[2], 10);
    try std.testing.expectEqual(@as(u32, 12), l[2].ids.items[0]); // 12 .. 21; 22 .. 31 free
    // free 0 .. 7 (a hole of 8) beside 22 .. 31 (10): a 6-page stream takes the smaller hole
    a.finish(&l[0]);
    try a.promise(&l[0], 6);
    try a.cover(&l[0], 6);
    try std.testing.expectEqual(@as(u32, 0), l[0].ids.items[0]);
    try std.testing.expectEqual(@as(usize, 1), Alloc.runsOf(l[0].ids.items));
    try std.testing.expectEqual(@as(u64, 0), a.stats.fallbacks);
    try std.testing.expectEqual(@as(u64, 4), a.stats.one_run);
}

test "inkling2 pages: no run holds them: scattered from the longest runs first, counted; growth in place" {
    const gpa = std.testing.allocator;
    var a = try Alloc.init(gpa, 16);
    defer a.deinit();
    var l: [6]Lease = @splat(.{});
    defer for (&l) |*x| a.finish(x);
    // holes of 3 (0 .. 2), 2 (5 .. 6) and 4 (12 .. 15) between taken pages
    for ([_][2]usize{ .{ 0, 3 }, .{ 1, 2 }, .{ 2, 2 }, .{ 3, 5 } }) |x| {
        try a.promise(&l[x[0]], x[1]);
        try a.cover(&l[x[0]], x[1]);
    }
    a.finish(&l[0]); // 0 .. 2
    a.finish(&l[2]); // 5 .. 6
    try std.testing.expectEqual(@as(usize, 5 + 4), a.nfree); // and 12 .. 15, never taken
    try a.promise(&l[5], 8);
    try a.cover(&l[5], 8); // no run of 8: 12 .. 15, then 0 .. 2, then 5
    try std.testing.expectEqualSlices(u32, &.{ 12, 13, 14, 15, 0, 1, 2, 5 }, l[5].ids.items);
    try std.testing.expectEqual(@as(u64, 1), a.stats.fallbacks);
    try std.testing.expectEqual(@as(u64, 3), a.stats.fallback_pieces);
    // growth: in place when the next page is free (6 after 5), else elsewhere
    try a.promise(&l[5], 1);
    try a.grow(&l[5], 1);
    try std.testing.expectEqual(@as(u32, 6), l[5].ids.items[8]);
    try std.testing.expectEqual(@as(u64, 1), a.stats.grown_in_place);
    try std.testing.expectError(error.LaneMemory, a.promise(&l[0], 1)); // every page taken
    a.finish(&l[5]);
    a.finish(&l[1]);
    a.finish(&l[3]);
    try std.testing.expect(a.idle());
}

test "inkling2 pages: a promise holds pages for its stream; finish gives back what was not taken" {
    const gpa = std.testing.allocator;
    var a = try Alloc.init(gpa, 8);
    defer a.deinit();
    var x: Lease = .{};
    var y: Lease = .{};
    try a.promise(&x, 6);
    try std.testing.expect(!a.fits(3));
    try std.testing.expectError(error.LaneMemory, a.promise(&y, 3));
    try a.cover(&x, 4);
    try std.testing.expectEqual(@as(usize, 2), x.promised);
    try std.testing.expectEqual(Census{ .total = 8, .free = 4, .promised = 2, .taken = 4 }, a.census());
    a.finish(&x);
    try std.testing.expect(a.idle());
    try std.testing.expectEqual(@as(usize, 3), need(500, 200, 8, 100));
    try std.testing.expectEqual(@as(usize, 2), need(500, 200, 8, 2));
}

test "inkling2 pages: churn keeps the counts whole and most covers in one run" {
    const gpa = std.testing.allocator;
    var a = try Alloc.init(gpa, 2052);
    defer a.deinit();
    var ls: [4]Lease = @splat(.{});
    defer for (&ls) |*x| a.finish(x);
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (0..4000) |_| {
        const s = r.uintLessThan(usize, 4);
        if (ls[s].ids.items.len > 0) {
            a.finish(&ls[s]);
            continue;
        }
        const n = 1 + r.uintLessThan(usize, 513);
        a.promise(&ls[s], n) catch continue;
        try a.cover(&ls[s], n);
        var held: usize = 0;
        for (ls) |x| held += x.ids.items.len;
        try std.testing.expectEqual(a.total() - a.nfree, held);
    }
    try std.testing.expect(a.stats.one_run * 10 >= a.stats.covers * 9);
}

/// G5 (host): a kept state as rank 0 holds it, its pages' ids.
const KeptPages = struct { ids: []const u32 };

fn keepPages(a: *Alloc, ids: []const u32) KeptPages {
    const own = a.gpa.dupe(u32, ids) catch @panic("test allocation");
    for (own) |id| a.retain(id);
    return .{ .ids = own };
}

fn dropPages(a: *Alloc, k: KeptPages) void {
    for (k.ids) |id| a.release(id);
    a.gpa.free(k.ids);
}

test "inkling2 pages: kept states share a stream's pages; two streams resume one state; a third keeps a longer one on the same pages" {
    const gpa = std.testing.allocator;
    var a = try Alloc.init(gpa, 64);
    defer a.deinit();
    var l: [4]Lease = @splat(.{});
    defer for (&l) |*x| a.finish(x);
    // stream 0 runs 20 pages and keeps states at 9 and 14 whole pages
    try a.promise(&l[0], 20);
    try a.cover(&l[0], 20);
    const s0 = try gpa.dupe(u32, l[0].ids.items);
    defer gpa.free(s0);
    const k9 = keepPages(&a, s0[0..9]);
    const k14 = keepPages(&a, s0[0..14]);
    try std.testing.expectEqual(@as(usize, 14), a.nkept);
    a.finish(&l[0]); // the stream ends: its 6 other pages free, the kept 14 stay
    try std.testing.expectEqual(@as(usize, 64 - 14), a.nfree);
    try std.testing.expect(a.idle()); // free + kept == total, nothing leased
    // two streams resume k9 at once: each adopts its 9 pages and takes 7 more; pages 9 .. 13 are k14's, so neither
    // continues in place: each takes a best-fit run elsewhere (14 .. 20, then 21 .. 27)
    for (l[1..3]) |*x| {
        try a.promise(x, 7);
        try a.adopt(x, k9.ids);
        try a.grow(x, 7);
    }
    try std.testing.expectEqualSlices(u32, s0[0..9], l[1].ids.items[0..9]);
    try std.testing.expectEqualSlices(u32, s0[0..9], l[2].ids.items[0..9]);
    try std.testing.expectEqual(@as(u32, 14), l[1].ids.items[9]);
    try std.testing.expectEqual(@as(u32, 21), l[2].ids.items[9]);
    try std.testing.expect(!a.idle());
    // a third keeps a longer state on stream 1's pages (12: the 9 shared and 14, 15, 16)
    const k12 = keepPages(&a, l[1].ids.items[0..12]);
    for (k9.ids) |id| try std.testing.expectEqual(@as(u32, 3), a.kept[id]); // k9, k14 and k12
    for (k9.ids) |id| try std.testing.expectEqual(@as(u32, 3), a.refs[id]); // the kept states once, two streams
    a.finish(&l[1]);
    a.finish(&l[2]);
    try std.testing.expect(a.idle());
    try std.testing.expectEqual(@as(usize, 14 + 3), a.nkept);
    try std.testing.expectEqual(a.total(), a.nfree + a.nkept);
    dropPages(&a, k14);
    dropPages(&a, k12);
    dropPages(&a, k9);
    try std.testing.expect(a.idle() and a.nfree == 64 and a.nkept == 0);
}

test "inkling2 pages: kept states evicted in every order leave the pool idle, free + kept = total at each step" {
    const gpa = std.testing.allocator;
    const orders = [_][3]usize{ .{ 0, 1, 2 }, .{ 0, 2, 1 }, .{ 1, 0, 2 }, .{ 1, 2, 0 }, .{ 2, 0, 1 }, .{ 2, 1, 0 } };
    for (orders) |ord| {
        var a = try Alloc.init(gpa, 32);
        defer a.deinit();
        var l: [2]Lease = @splat(.{});
        defer for (&l) |*x| a.finish(x);
        try a.promise(&l[0], 12);
        try a.cover(&l[0], 12);
        // states at 4, 8 and 11 pages of stream 0; stream 1 resumes the 8 and keeps 10 (2 of its own)
        var ks: [3]KeptPages = undefined;
        ks[0] = keepPages(&a, l[0].ids.items[0..4]);
        ks[1] = keepPages(&a, l[0].ids.items[0..8]);
        try a.promise(&l[1], 6);
        try a.adopt(&l[1], ks[1].ids);
        try a.grow(&l[1], 6);
        ks[2] = keepPages(&a, l[1].ids.items[0..10]);
        a.finish(&l[0]);
        a.finish(&l[1]);
        try std.testing.expect(a.idle());
        for (ord) |i| {
            dropPages(&a, ks[i]);
            try std.testing.expect(a.idle());
            try std.testing.expectEqual(a.total(), a.nfree + a.nkept);
        }
        try std.testing.expectEqual(@as(usize, 32), a.nfree);
    }
}

test "inkling2 pages: a resumed stream's table continues its adopted pages when the next ones are free" {
    const gpa = std.testing.allocator;
    var a = try Alloc.init(gpa, 40);
    defer a.deinit();
    var l: [3]Lease = @splat(.{});
    defer for (&l) |*x| a.finish(x);
    try a.promise(&l[0], 10);
    try a.cover(&l[0], 10); // 0 .. 9
    const k = keepPages(&a, l[0].ids.items[0..6]);
    defer dropPages(&a, k);
    a.finish(&l[0]);
    try a.promise(&l[1], 10);
    try a.adopt(&l[1], k.ids);
    try a.grow(&l[1], 10); // 6 .. 15 in place
    try std.testing.expectEqual(@as(usize, 1), Alloc.runsOf(l[1].ids.items));
    try std.testing.expectEqual(@as(u64, 1), a.stats.grown_in_place);
    // a second resume while the first runs: its fresh pages go elsewhere (two runs)
    try a.promise(&l[2], 4);
    try a.adopt(&l[2], k.ids);
    try a.grow(&l[2], 4);
    try std.testing.expectEqual(@as(usize, 2), Alloc.runsOf(l[2].ids.items));
    try std.testing.expectEqual(@as(usize, 40 - 16 - 4), a.nfree);
    a.finish(&l[1]);
    a.finish(&l[2]);
    try std.testing.expect(a.idle());
}

test "inkling2 pages: /health's counts follow the allocator and the admission's wait" {
    var a = try Alloc.init(std.testing.allocator, 64);
    defer a.deinit();
    try std.testing.expectEqual(Shown{ .pages = 64, .free = 64, .available = 64 }, shownOf(&a, null, 0));
    var l: Lease = .{};
    try a.promise(&l, 20);
    try a.cover(&l, 12);
    const s = shownOf(&a, 40, 3);
    try std.testing.expectEqual(@as(usize, 52), s.free);
    try std.testing.expectEqual(@as(usize, 44), s.available); // 8 still promised to the stream
    try std.testing.expectEqual(@as(usize, 40), s.waiting);
    try std.testing.expectEqual(@as(u64, 3), s.waits);
    a.finish(&l);
    try std.testing.expectEqual(Shown{ .pages = 64, .free = 64, .available = 64, .waits = 1 }, shownOf(&a, null, 1));
}
