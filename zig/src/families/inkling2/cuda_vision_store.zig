//! Rank 0's image rows for the vision tower in the engine: the host side of one device buffer of `rows` tower output
//! rows (bf16 [rows][4608], 9,216 B a row), indexed by each image's full 256-bit content digest.
//!
//! `Store` places an image's rows as one extent: first fit, else the least recently used unpinned entries go, and when
//! the free rows are enough but split, the entries move down (compaction) in gap-sized device copies through `Copy`,
//! so this file needs no GPU and the host tests run it over fake memory. A request reserves its images' rows when it
//! opens; while the reservations sum to at most `rows`, `alloc` cannot fail: every pinned entry is a reserved image.
//! At most one entry is building (the tower holds one image's state); it has no rows until the tower finishes it, and
//! it is aborted when its last pin drops. A compaction copy that fails breaks the store (`broken`), at its enqueue or
//! at the compaction's wait for its copies: an entry it was moving may hold other rows, so from then on no image is
//! served, stored or begun (`StoreBroken`) and only pins and reservations drop; the engine needs a restart for images
//! again.
//!
//! `Need` is a prompt pass's view of its images: which ones it still needs (`init` skips those that end at or before
//! the resume point), what the next tower unit is (`plan`), the chunk's spans of stored rows (`spans`), and the pins it
//! drops as the chunk cursor passes them (`passed`, `release`). `prefixKey` names the images in a kept prompt prefix.
const std = @import("std");

const Sha256 = std.crypto.hash.sha2.Sha256;

/// The bytes of one tower output row: 4,608 bf16.
pub const row_bytes = 4608 * 2;
/// Images a request may hold (a chunk's spans are at most this many).
pub const max_images = 16;

pub const Digest = [32]u8;

/// One image of a prompt: rows [at, at + tokens) of the prompt are its `rows` x `cols` tower rows, and `digest` names
/// its pixels (SHA-256 of the decoded image; never all zeros).
pub const Image = struct { at: u32, tokens: u32, rows: u32, cols: u32, digest: Digest };

/// A device copy of `bytes` from `src` to `dst`, ordered on the tower's stream (`run`; it may return before the copy
/// ends), and the wait for every copy run so far (`done`: a copy's device fault surfaces here). The two ranges never
/// overlap.
pub const Copy = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, dst: u64, src: u64, bytes: usize) anyerror!void,
    done: *const fn (ctx: *anyopaque) anyerror!void,
};

pub const Entry = struct {
    digest: Digest,
    row0: u32,
    tokens: u32,
    pins: u32 = 0,
    ready: bool = false, // false: the tower is building it, and its rows hold nothing yet
    used: u64 = 0, // the store's clock at the last pin or finish
};

/// Prompt rows [first, first + n) of a chunk take the stored rows at `src`.
pub const Span = struct { first: u32, n: u32, src: u64 };

pub const Store = struct {
    gpa: std.mem.Allocator,
    buf: u64,
    rows: u32,
    row_bytes: usize,
    copy: Copy,
    entries: std.ArrayList(Entry) = .empty, // by row0; extents never overlap; capacity `rows` from `init`
    committed: u32 = 0, // rows reserved by open requests
    clock: u64 = 0,
    evicted: u64 = 0,
    moved: u64 = 0, // rows compaction copied
    /// a compaction copy failed part way: no entry's rows are trusted (`check`)
    broken: bool = false,

    /// The index's room for its most entries (one a row) is taken here, at start-up: nothing later allocates.
    pub fn init(gpa: std.mem.Allocator, buf: u64, rows: u32, bytes_per_row: usize, copy: Copy) !Store {
        var s: Store = .{ .gpa = gpa, .buf = buf, .rows = rows, .row_bytes = bytes_per_row, .copy = copy };
        try s.entries.ensureTotalCapacityPrecise(gpa, rows);
        return s;
    }

    pub fn deinit(s: *Store) void {
        s.entries.deinit(s.gpa);
    }

    /// A request's rows, at its opening: `LaneMemory` (the request waits) when the open requests already hold the rest.
    pub fn reserve(s: *Store, n: u32) error{LaneMemory}!void {
        if (n > s.rows - s.committed) return error.LaneMemory;
        s.committed += n;
    }

    pub fn unreserve(s: *Store, n: u32) void {
        std.debug.assert(n <= s.committed);
        s.committed -= n;
    }

    /// StoreBroken once a compaction copy failed: every lookup, splice and new entry is refused from then on.
    pub fn check(s: *const Store) error{StoreBroken}!void {
        if (s.broken) return error.StoreBroken;
    }

    fn index(s: *const Store, d: *const Digest) ?usize {
        for (s.entries.items, 0..) |*e, i| if (std.mem.eql(u8, &e.digest, d)) return i;
        return null;
    }

    /// The entry with all 256 bits of `d`; the pointer lasts until the next alloc, abort or unpin.
    pub fn find(s: *const Store, d: *const Digest) ?*Entry {
        const i = s.index(d) orelse return null;
        return &s.entries.items[i];
    }

    /// The building entry (the tower's job), if any.
    pub fn job(s: *const Store) ?*Entry {
        for (s.entries.items) |*e| if (!e.ready) return e;
        return null;
    }

    /// The device address of row `row` of the buffer.
    pub fn addr(s: *const Store, row: u64) u64 {
        return s.buf + row * s.row_bytes;
    }

    pub fn pin(s: *Store, d: *const Digest) void {
        const e = s.find(d).?;
        e.pins += 1;
        s.clock += 1;
        e.used = s.clock;
    }

    /// Drops one pin; true when that aborted a building entry (its tower job must be dropped too).
    pub fn unpin(s: *Store, d: *const Digest) bool {
        const i = s.index(d).?;
        const e = &s.entries.items[i];
        std.debug.assert(e.pins > 0);
        e.pins -= 1;
        if (e.pins > 0 or e.ready) return false;
        _ = s.entries.orderedRemove(i);
        return true;
    }

    /// The tower wrote the building entry's rows.
    pub fn finish(s: *Store, d: *const Digest) void {
        const e = s.find(d).?;
        std.debug.assert(!e.ready);
        e.ready = true;
        s.clock += 1;
        e.used = s.clock;
    }

    /// Drops a building entry whatever its pins (the tower failed); its extent is free again.
    pub fn abort(s: *Store, d: *const Digest) void {
        const i = s.index(d).?;
        std.debug.assert(!s.entries.items[i].ready);
        _ = s.entries.orderedRemove(i);
    }

    /// Rows no entry holds.
    pub fn free(s: *const Store) u32 {
        var used: u32 = 0;
        for (s.entries.items) |e| used += e.tokens;
        return s.rows - used;
    }

    const Fit = struct { pos: usize, row0: u32 };

    fn firstFit(s: *const Store, tokens: u32) ?Fit {
        var next: u32 = 0;
        for (s.entries.items, 0..) |e, i| {
            if (e.row0 - next >= tokens) return .{ .pos = i, .row0 = next };
            next = e.row0 + e.tokens;
        }
        if (s.rows - next >= tokens) return .{ .pos = s.entries.items.len, .row0 = next };
        return null;
    }

    fn lru(s: *const Store) ?usize {
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (e.pins != 0 or !e.ready) continue;
            if (best == null or e.used < s.entries.items[best.?].used) best = i;
        }
        return best;
    }

    /// Moves `n` rows from `src` down to `dst` in pieces of at most the gap, so no copy overlaps itself.
    fn move(s: *Store, dst: u32, src: u32, n: u32) !void {
        const gap = src - dst;
        var off: u32 = 0;
        while (off < n) {
            const k = @min(gap, n - off);
            try s.copy.run(s.copy.ctx, s.addr(dst + off), s.addr(src + off), @as(usize, k) * s.row_bytes);
            off += k;
        }
        s.moved += n;
    }

    /// Every entry down to the lowest rows, in order, the moves waited for (`Copy.done`, once) before it returns. A
    /// copy error breaks the store, whether its enqueue or its completion reports it: the entry being moved may already
    /// have rows of its own written over at its old address (pieces of the gap), and it is still ready there.
    fn compact(s: *Store) !void {
        errdefer {
            s.broken = true;
            std.log.warn("inkling2 rank 0: vision: an image row store compaction copy failed: no image is served until the engine restarts", .{});
        }
        var next: u32 = 0;
        var moved = false;
        for (s.entries.items) |*e| {
            std.debug.assert(e.row0 >= next);
            if (e.row0 != next) {
                if (e.ready) {
                    try s.move(next, e.row0, e.tokens);
                    moved = true;
                }
                e.row0 = next;
            }
            next += e.tokens;
        }
        // the copies are asynchronous: a device fault surfaces at their completion, here, under the errdefer
        if (moved) try s.copy.done(s.copy.ctx);
    }

    /// A building entry of `tokens` rows for `d`, pinned once; returns its first row. First fit; else unpinned
    /// entries go, least recently used first, until one fits or the free rows suffice; then compaction. No host
    /// allocation: entries hold a row or more each, so with `tokens` rows free there are fewer than `rows` of them.
    pub fn alloc(s: *Store, d: *const Digest, tokens: u32) !u32 {
        try s.check();
        if (tokens == 0 or tokens > s.rows) return error.ImageRows;
        if (s.index(d) != null) return error.ImageStored;
        if (s.job() != null) return error.TowerBusy;
        const fit = fit: while (true) {
            if (s.firstFit(tokens)) |f| break :fit f;
            if (s.free() >= tokens) {
                try s.compact();
                break :fit s.firstFit(tokens).?;
            }
            const i = s.lru() orelse return error.StoreFull; // pinned rows beyond the reservations
            _ = s.entries.orderedRemove(i);
            s.evicted += 1;
        };
        s.clock += 1;
        std.debug.assert(s.entries.items.len < s.entries.capacity);
        s.entries.insertAssumeCapacity(fit.pos, .{ .digest = d.*, .row0 = fit.row0, .tokens = tokens, .pins = 1, .used = s.clock });
        return fit.row0;
    }
};

/// What a prompt pass does before a chunk: `ready` (every image in it is stored), `work` (run the tower job's next
/// slice: this pass's image or another pass's, whose rows it may need first), or `begin` (images[i] has a new building
/// entry, pinned: start the tower on it).
pub const Step = union(enum) { ready, work, begin: u32 };

/// A prompt pass's images still to splice, and its pins and reservation in the store. The image slice is passed to
/// each call (it is the request's), in prompt order.
pub const Need = struct {
    count: u32,
    first: u32, // images before it end at or before the cursor: no tower, no splice
    pinned: std.bit_set.IntegerBitSet(max_images) = .empty,
    reserved: u32 = 0,

    /// Skips the images that end at or before `from` (a kept prefix holds their rows). Refuses images out of prompt
    /// order or overlapping, a grid that is not `tokens`, an all-zero digest, and more than `max_images`.
    pub fn init(images: []const Image, from: u32) !Need {
        if (images.len > max_images) return error.TooManyImages;
        var end: u64 = 0;
        for (images) |m| {
            if (m.tokens == 0 or @as(u64, m.rows) * m.cols != m.tokens) return error.ImageGrid;
            if (std.mem.allEqual(u8, &m.digest, 0)) return error.ImageDigestDiffers;
            if (m.at < end) return error.ImageOrder;
            end = @as(u64, m.at) + m.tokens;
        }
        var first: u32 = 0;
        while (first < images.len and @as(u64, images[first].at) + images[first].tokens <= from) first += 1;
        return .{ .count = @intCast(images.len), .first = first };
    }

    /// Rows of the images still needed: what the pass reserves.
    pub fn tokens(n: *const Need, images: []const Image) u32 {
        var t: u32 = 0;
        for (images[n.first..]) |m| t += m.tokens;
        return t;
    }

    pub fn reserve(n: *Need, s: *Store, images: []const Image) error{LaneMemory}!void {
        std.debug.assert(images.len == n.count and n.reserved == 0);
        const t = n.tokens(images);
        try s.reserve(t);
        n.reserved = t;
    }

    /// The next tower unit for chunk [at, at + len): pins each stored image it meets, a hit costs no tower.
    pub fn plan(n: *Need, s: *Store, images: []const Image, at: u32, len: u32) !Step {
        std.debug.assert(images.len == n.count);
        try s.check(); // before any pin: a broken store's entries are no hits
        const end = @as(u64, at) + len;
        for (images[n.first..], n.first..) |*m, i| {
            if (m.at >= end) break;
            if (@as(u64, m.at) + m.tokens <= at) continue;
            if (s.find(&m.digest)) |e| {
                const ready = e.ready;
                if (!n.pinned.isSet(i)) {
                    s.pin(&m.digest);
                    n.pinned.set(i);
                }
                if (ready) continue;
                return .work; // the job is this image (another pass's or ours)
            }
            if (s.job() != null) return .work; // another image holds the tower: help it finish
            _ = try s.alloc(&m.digest, m.tokens);
            n.pinned.set(i);
            return .{ .begin = @intCast(i) };
        }
        return .ready;
    }

    /// Whether every image chunk [at, at + len) meets is stored and ready, so `plan` would return `ready` with no
    /// tower unit. Pins nothing, allocates nothing.
    pub fn stored(n: *const Need, s: *const Store, images: []const Image, at: u32, len: u32) bool {
        std.debug.assert(images.len == n.count);
        const end = @as(u64, at) + len;
        for (images[n.first..]) |*m| {
            if (m.at >= end) break;
            if (@as(u64, m.at) + m.tokens <= at) continue;
            const e = s.find(&m.digest) orelse return false;
            if (!e.ready) return false;
        }
        return true;
    }

    /// The stored rows chunk [at, at + len) splices, in prompt order, into `out`; returns the count. Every image the
    /// chunk meets must be pinned by this pass and ready (`plan` returned `ready`).
    pub fn spans(n: *const Need, s: *const Store, images: []const Image, at: u32, len: u32, out: []Span) !usize {
        std.debug.assert(images.len == n.count);
        try s.check(); // a pass pinned before the store broke splices nothing either
        const end = @as(u64, at) + len;
        var k: usize = 0;
        for (images[n.first..], n.first..) |*m, i| {
            if (m.at >= end) break;
            const m_end = @as(u64, m.at) + m.tokens;
            if (m_end <= at) continue;
            if (!n.pinned.isSet(i)) return error.ImageNotReady;
            const e = s.find(&m.digest) orelse return error.ImageNotReady;
            if (!e.ready) return error.ImageNotReady;
            if (k == out.len) return error.TooManySpans;
            const lo: u32 = @max(at, m.at);
            const hi: u32 = @intCast(@min(end, m_end));
            out[k] = .{ .first = lo - at, .n = hi - lo, .src = s.addr(@as(u64, e.row0) + (lo - m.at)) };
            k += 1;
        }
        return k;
    }

    /// Whether the pass pins the image named `d` (its own, or the same pixels another pass began).
    pub fn holds(n: *const Need, images: []const Image, d: *const Digest) bool {
        var it = n.pinned.iterator(.{});
        while (it.next()) |i| if (std.mem.eql(u8, &images[i].digest, d)) return true;
        return false;
    }

    fn drop(n: *Need, s: *Store, images: []const Image, i: usize) bool {
        if (!n.pinned.isSet(i)) return false;
        n.pinned.unset(i);
        return s.unpin(&images[i].digest);
    }

    /// The cursor reached `at`: the images that end at or before it are unpinned and no longer needed. True when that
    /// aborted the tower job.
    pub fn passed(n: *Need, s: *Store, images: []const Image, at: u32) bool {
        std.debug.assert(images.len == n.count);
        var aborted = false;
        while (n.first < n.count and @as(u64, images[n.first].at) + images[n.first].tokens <= at) : (n.first += 1) {
            if (n.drop(s, images, n.first)) aborted = true;
        }
        return aborted;
    }

    /// The pass ended or was dropped: every pin and the reservation go. True when that aborted the tower job.
    pub fn release(n: *Need, s: *Store, images: []const Image) bool {
        std.debug.assert(images.len == n.count);
        var aborted = false;
        for (0..n.count) |i| {
            if (n.drop(s, images, i)) aborted = true;
        }
        s.unreserve(n.reserved);
        n.reserved = 0;
        n.first = n.count;
        return aborted;
    }
};

/// SHA-256 over (at, tokens, rows, cols as u32 little-endian, then the digest) of each image with `img.at < at`, in
/// order: the images a kept prefix of `at` rows holds. All zeros, with no hashing, when there is none.
pub fn prefixKey(images: []const Image, at: u32) Digest {
    var h: ?Sha256 = null;
    for (images) |m| {
        if (m.at >= at) continue;
        if (h == null) h = Sha256.init(.{});
        var b: [16]u8 = undefined;
        std.mem.writeInt(u32, b[0..4], m.at, .little);
        std.mem.writeInt(u32, b[4..8], m.tokens, .little);
        std.mem.writeInt(u32, b[8..12], m.rows, .little);
        std.mem.writeInt(u32, b[12..16], m.cols, .little);
        h.?.update(&b);
        h.?.update(&m.digest);
    }
    var out: Digest = @splat(0);
    if (h) |*x| x.final(&out);
    return out;
}

/// Tower layers in one unit: all `left` when chunks are unbounded (`max_rows` 0); 1 while a layer's time is unknown;
/// else what a chunk of `max_rows` rows costs (`ns_per_row` each) over a layer's `layer_ns`, clamped to [1, left].
pub fn sliceLayers(max_rows: u32, ns_per_row: u64, layer_ns: u64, left: u32) u32 {
    if (left == 0) return 0;
    if (max_rows == 0) return left;
    if (layer_ns == 0 or ns_per_row == 0) return 1;
    const budget = @as(u128, max_rows) * ns_per_row;
    return @intCast(std.math.clamp(budget / layer_ns, 1, left));
}

/// The tower's layers (cuda_vision.zig layers).
pub const tower_layers: u32 = 23;

/// The tower's pace beside decoders (a prompt chunk capped at `max_rows` rows: --decode-share): the job in flight
/// (its image's digest, the layers it has left and the units it took), what one of its layers takes (timed on the
/// host in its first unit), what a settled prompt chunk's row costs (an EMA), and the help other passes gave.
pub const Pace = struct {
    /// the building entry the tower holds the state of; null: none, or its state is lost (a unit failed)
    job: ?Digest = null,
    left: u32 = 0,
    units: u32 = 0,
    layer_ns: u64 = 0,
    /// a settled prompt chunk's ns over its rows (cuda_lanes.zig chunk), 1/4 the newest
    chunk_ns_per_row: u64 = 0,
    /// since start: units a pass ran of another image's job, and those in a step nothing billed to decoders (no chunk
    /// row cap and a step the lane host did not time: a capped unit is side work the lane host bills, Stream.side_ns)
    help_units: u64 = 0,
    unbilled_help: u64 = 0,

    /// A settled prompt chunk of `rows` rows took `ns`.
    pub fn chunked(p: *Pace, ns: u64, rows: u64) void {
        if (rows == 0) return;
        const v = @max(1, ns / rows);
        p.chunk_ns_per_row = if (p.chunk_ns_per_row == 0) v else (3 * p.chunk_ns_per_row + v) / 4;
    }

    pub fn holds(p: *const Pace, d: *const Digest) bool {
        return if (p.job) |*j| std.mem.eql(u8, j, d) else false;
    }

    fn start(p: *Pace, d: *const Digest) void {
        p.job = d.*;
        p.left = tower_layers;
        p.units = 1;
        p.layer_ns = 0;
    }

    /// `k` layers of the job (none only when none is left), then the merger into its rows when none is left: the store
    /// holds the image. A failure leaves the tower's state unknown: the job is no longer held (the caller drops it).
    fn run(p: *Pace, s: *Store, t: Units, k: u32) !void {
        errdefer p.job = null;
        std.debug.assert(k <= p.left and (k > 0 or p.left == 0));
        if (k > 0) try t.layers(t.ctx, k);
        p.left -= k;
        if (p.left > 0) return;
        const d = s.job().?.digest;
        try t.merge(t.ctx, s.find(&d).?.row0);
        s.finish(&d);
        p.job = null;
    }
};

/// The tower's work on the device, a unit at a time (cuda_lanes.zig: Rank0's engine-mode units on the engine's stream,
/// each borrowing the op scratch; the host tests' fakes).
pub const Units = struct {
    ctx: *anyopaque,
    /// the pass's image `i` (the building entry just allocated): its grid, then E1 .. E5 into the tower's rows
    begin: *const fn (ctx: *anyopaque, i: u32) anyerror!void,
    /// the job's next `k` layers (k >= 1)
    layers: *const fn (ctx: *anyopaque, k: u32) anyerror!void,
    /// the merger into the store's rows from `row0`
    merge: *const fn (ctx: *anyopaque, row0: u32) anyerror!void,
    /// waits for the device
    settle: *const fn (ctx: *anyopaque) anyerror!void,
    /// the host's clock, ns
    now: *const fn (ctx: *anyopaque) u64,
};

/// A pass's units of other images' jobs (help), and those in steps nothing billed to decoders (`unit`'s `settled`).
pub const Help = struct { units: u32 = 0, layers: u32 = 0, unbilled: u32 = 0 };

pub const Unit = enum { ready, worked };

/// Before a pass's chunk [at, at + len): `ready` when every image of the chunk is stored (the chunk's forward goes
/// out), else `worked` after one tower unit (the chunk waits for a later call). Every `worked` ran a layer or more,
/// or the merger, so a caller that calls again until `ready` makes progress each call.
/// - `max_rows` 0 (chunks without a cap): each job whole (begin, every layer, the merger), then the chunk: never
///   `worked`.
/// - else a new job's first unit is its begin and one layer, timed on the host between two settles (`layer_ns`);
///   a later unit runs `sliceLayers(max_rows, chunk_ns_per_row, layer_ns, left)` layers, and the merger with the last.
/// - another image's job in flight (`plan`'s `work` for an image not this pass's): its next unit runs here (help),
///   counted in `help` and the pace; `settled`: this step's time is billed to decoders (the lane host times it, or
///   bills a capped unit as side work).
pub fn unit(n: *Need, s: *Store, pace: *Pace, images: []const Image, at: u32, len: u32, max_rows: u32, settled: bool, t: Units, help: *Help) !Unit {
    while (true) switch (try n.plan(s, images, at, len)) {
        .ready => return .ready,
        .begin => |i| {
            const d = &images[i].digest;
            pace.start(d);
            errdefer pace.job = null;
            t.begin(t.ctx, i) catch |err| {
                s.abort(d);
                n.pinned.unset(i);
                return err;
            };
            if (max_rows == 0) {
                try pace.run(s, t, pace.left);
                continue;
            }
            try t.settle(t.ctx); // the begin's work done: the layer alone is timed
            const t0 = t.now(t.ctx);
            try pace.run(s, t, 1);
            try t.settle(t.ctx);
            pace.layer_ns = @max(1, t.now(t.ctx) -| t0);
            return .worked;
        },
        .work => {
            const j = s.job().?.digest; // plan works only for the building entry
            if (!pace.holds(&j)) return error.TowerJobLost;
            const k = sliceLayers(max_rows, pace.chunk_ns_per_row, pace.layer_ns, pace.left);
            if (!n.holds(images, &j)) {
                help.units += 1;
                help.layers += k;
                pace.help_units += 1;
                if (!settled) {
                    help.unbilled += 1;
                    pace.unbilled_help += 1;
                }
            }
            pace.units += 1;
            try pace.run(s, t, k);
            if (max_rows == 0) continue;
            return .worked;
        },
    };
}

// ---------------------------------------------------------------------------------------------------------------------
// Host tests over fake device memory.

const testing = std.testing;

const Fake = struct {
    mem: []u8,
    base: u64,
    copies: u32 = 0,
    /// the copy after this many fails (a device fault in the middle of a compaction)
    fail_at: ?u32 = null,
    /// every copy is taken and lost on the device, and the wait for them fails (a fault that surfaces at completion)
    fail_done: bool = false,
    dones: u32 = 0,

    fn run(ctx: *anyopaque, dst: u64, src: u64, bytes: usize) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_at) |n| if (f.copies == n) return error.FakeFault;
        if (f.fail_done) {
            f.copies += 1;
            return;
        }
        if (dst < f.base or src < f.base) return error.OutOfRange;
        const d: usize = @intCast(dst - f.base);
        const s: usize = @intCast(src - f.base);
        if (d + bytes > f.mem.len or s + bytes > f.mem.len) return error.OutOfRange;
        if (d < s + bytes and s < d + bytes) return error.Overlap;
        @memcpy(f.mem[d..][0..bytes], f.mem[s..][0..bytes]);
        f.copies += 1;
    }

    fn done(ctx: *anyopaque) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.dones += 1;
        if (f.fail_done) return error.FakeFault;
    }

    fn copy(f: *Fake) Copy {
        return .{ .ctx = f, .run = run, .done = done };
    }

    fn at(f: *const Fake, a: u64) []const u8 {
        return f.mem[@intCast(a - f.base)..];
    }
};

/// The bytes a tower writes for row `r` of the image named `d`.
fn pattern(d: *const Digest, r: u64, b: u64) u8 {
    return d[@intCast((r * 7 + b) % 32)] ^ @as(u8, @truncate(r * 31 + b * 3 + 1));
}

fn write(f: *Fake, s: *const Store, d: *const Digest) void {
    const e = s.find(d).?;
    var r: u64 = 0;
    while (r < e.tokens) : (r += 1) {
        const a: usize = @intCast(s.addr(e.row0 + r) - f.base);
        for (0..s.row_bytes) |b| f.mem[a + b] = pattern(d, r, @intCast(b));
    }
}

/// Whether the rows from device row `row0` hold image `d`'s rows [r0, r0 + n).
fn holds(f: *const Fake, s: *const Store, d: *const Digest, row0: u64, r0: u64, n: u64) bool {
    var r = r0;
    while (r < r0 + n) : (r += 1) {
        const m = f.at(s.addr(row0 + (r - r0)));
        for (0..s.row_bytes) |b| if (m[b] != pattern(d, r, @intCast(b))) return false;
    }
    return true;
}

fn digestOf(k: u32) Digest {
    var d: Digest = undefined;
    Sha256.hash(std.mem.asBytes(&k), &d, .{});
    return d;
}

fn mk(at: u32, rows: u32, cols: u32, k: u32) Image {
    return .{ .at = at, .tokens = rows * cols, .rows = rows, .cols = cols, .digest = digestOf(k) };
}

fn checkStore(f: *const Fake, s: *const Store) !void {
    var next: u32 = 0;
    var building: u32 = 0;
    for (s.entries.items) |*e| {
        try testing.expect(e.row0 >= next);
        try testing.expect(e.tokens > 0);
        next = e.row0 + e.tokens;
        if (!e.ready) {
            building += 1;
            try testing.expect(e.pins > 0);
        } else try testing.expect(holds(f, s, &e.digest, e.row0, 0, e.tokens));
    }
    try testing.expect(next <= s.rows);
    try testing.expect(building <= 1);
}

test "reserve refuses rows past the store, unreserve returns them" {
    var mem: [64]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 16, 4, f.copy());
    defer s.deinit();
    try s.reserve(10);
    try s.reserve(6);
    try testing.expectError(error.LaneMemory, s.reserve(1));
    s.unreserve(6);
    try testing.expectError(error.LaneMemory, s.reserve(7));
    try s.reserve(6);
    try testing.expectEqual(@as(u32, 16), s.committed);
}

test "find compares all 256 bits of the digest" {
    var mem: [64]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 16, 4, f.copy());
    defer s.deinit();
    const d = digestOf(1);
    _ = try s.alloc(&d, 4);
    for (0..256) |bit| {
        var x = d;
        x[bit / 8] ^= @as(u8, 1) << @intCast(bit % 8);
        try testing.expect(s.find(&x) == null);
    }
    try testing.expect(s.find(&d) != null);
    try testing.expectError(error.ImageStored, s.alloc(&d, 4));
    try testing.expectError(error.TowerBusy, s.alloc(&digestOf(2), 4));
    try testing.expectError(error.ImageRows, s.alloc(&digestOf(2), 17));
}

test "alloc: first fit, then the least recently used unpinned entry, then compaction in gap-sized copies" {
    const rb = 4;
    var mem: [16 * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 16, rb, f.copy());
    defer s.deinit();
    // A [0,2) B [2,4) C [4,14), 2 rows free at the top
    const a = digestOf(10);
    const b = digestOf(11);
    const c = digestOf(12);
    for ([_]struct { d: *const Digest, n: u32, row0: u32 }{ .{ .d = &a, .n = 2, .row0 = 0 }, .{ .d = &b, .n = 2, .row0 = 2 }, .{ .d = &c, .n = 10, .row0 = 4 } }) |x| {
        try testing.expectEqual(x.row0, try s.alloc(x.d, x.n));
        write(&f, &s, x.d);
        s.finish(x.d);
    }
    try testing.expect(!s.unpin(&a));
    try testing.expect(!s.unpin(&b));
    s.pin(&b); // B is now newer than A
    try testing.expect(!s.unpin(&b));
    // E (4): 2 free -> A goes (oldest) -> 4 free but split -> B 2->0, C 4->2 in 2-row pieces, E at 12
    const e = digestOf(13);
    try testing.expectEqual(@as(u32, 12), try s.alloc(&e, 4));
    try testing.expectEqual(@as(u64, 1), s.evicted);
    try testing.expect(s.find(&a) == null);
    try testing.expectEqual(@as(u32, 0), s.find(&b).?.row0);
    try testing.expectEqual(@as(u32, 2), s.find(&c).?.row0);
    try testing.expectEqual(@as(u64, 12), s.moved);
    try testing.expectEqual(@as(u32, 1 + 5), f.copies);
    write(&f, &s, &e);
    s.finish(&e);
    try checkStore(&f, &s);
    // a first fit takes a hole without moving anything: drop B, then 2 rows fit at 0
    try testing.expect(!s.unpin(&e));
    _ = s.entries.orderedRemove(0); // B evicted by hand
    const g = digestOf(14);
    try testing.expectEqual(@as(u32, 0), try s.alloc(&g, 2));
    try testing.expectEqual(@as(u32, 1 + 5), f.copies);
    // nothing unpinned and too few rows: the reservations were broken
    write(&f, &s, &g);
    s.finish(&g);
    s.pin(&e);
    try testing.expectError(error.StoreFull, s.alloc(&digestOf(15), 4));
    try checkStore(&f, &s);
}

test "split free rows are compacted before any entry is evicted" {
    const rb = 3;
    var mem: [12 * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x40 };
    var s: Store = try .init(testing.allocator, f.base, 12, rb, f.copy());
    defer s.deinit();
    const ds = [_]Digest{ digestOf(1), digestOf(2), digestOf(3), digestOf(4) };
    for (&ds) |*d| {
        _ = try s.alloc(d, 3);
        write(&f, &s, d);
        s.finish(d);
        _ = s.unpin(d);
    }
    // holes at [0,3) and [6,9): free 6, no hole of 5
    _ = s.entries.orderedRemove(2);
    _ = s.entries.orderedRemove(0);
    const n = digestOf(5);
    try testing.expectEqual(@as(u32, 6), try s.alloc(&n, 5));
    try testing.expectEqual(@as(u64, 0), s.evicted);
    try testing.expectEqual(@as(u32, 0), s.find(&ds[1]).?.row0);
    try testing.expectEqual(@as(u32, 3), s.find(&ds[3]).?.row0);
    s.abort(&n);
    try checkStore(&f, &s);
}

test "a compaction copy that fails part way breaks the store: no hit, no span (pinned passes too), no alloc; pins still drop" {
    const rb = 4;
    var mem: [8 * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 8, rb, f.copy());
    defer s.deinit();
    // A [0,1) unpinned, B [1,5) pinned by a pass, C [5,7) pinned; one row free at the top
    const a = digestOf(1);
    _ = try s.alloc(&a, 1);
    write(&f, &s, &a);
    s.finish(&a);
    try testing.expect(!s.unpin(&a));
    const ib = [_]Image{mk(3, 2, 2, 2)};
    var pb: Need = try .init(&ib, 0);
    try pb.reserve(&s, &ib);
    try testing.expectEqual(Step{ .begin = 0 }, try pb.plan(&s, &ib, 0, 8));
    write(&f, &s, &ib[0].digest);
    s.finish(&ib[0].digest);
    try testing.expectEqual(Step.ready, try pb.plan(&s, &ib, 0, 8));
    const c = digestOf(3);
    try testing.expectEqual(@as(u32, 5), try s.alloc(&c, 2));
    write(&f, &s, &c);
    s.finish(&c);
    // D (2 rows): A goes, the free rows [0,1) and [7,8) are split, B moves 1 -> 0 in 1-row pieces: the third fails
    // after two pieces overwrote B's first row at its old address
    f.fail_at = f.copies + 2;
    try testing.expectError(error.FakeFault, s.alloc(&digestOf(4), 2));
    try testing.expectEqual(@as(u32, 1), s.find(&ib[0].digest).?.row0);
    try testing.expect(!holds(&f, &s, &ib[0].digest, 1, 0, 4));
    f.fail_at = null;
    // the next lookup of B's pixels is no hit, and the pass that pinned B splices nothing
    var q: Need = try .init(&ib, 0);
    try testing.expectError(error.StoreBroken, q.plan(&s, &ib, 0, 8));
    try testing.expect(q.pinned.count() == 0);
    var sp: [max_images]Span = undefined;
    try testing.expectError(error.StoreBroken, pb.spans(&s, &ib, 0, 8, &sp));
    try testing.expectError(error.StoreBroken, pb.plan(&s, &ib, 0, 8));
    // C was not moved but nothing is served; no new entry either, and the tower's units refuse
    const ic = [_]Image{mk(0, 1, 2, 3)};
    var qc: Need = try .init(&ic, 0);
    try testing.expectError(error.StoreBroken, qc.plan(&s, &ic, 0, 2));
    try testing.expectError(error.StoreBroken, s.alloc(&digestOf(5), 1));
    var ft: FakeTower = .{ .f = &f, .s = &s };
    var pace: Pace = .{};
    var help: Help = .{};
    const ie = [_]Image{mk(0, 1, 1, 6)};
    var qe: Need = try .init(&ie, 0);
    try testing.expectError(error.StoreBroken, unit(&qe, &s, &pace, &ie, 0, 1, 512, true, ft.units(), &help));
    try testing.expect(ft.work == 0 and s.job() == null);
    // the passes still end: their pins and reservations go
    try testing.expect(!pb.release(&s, &ib));
    try testing.expect(!s.unpin(&c));
    try testing.expectEqual(@as(u32, 0), s.committed);
    for (s.entries.items) |e| try testing.expectEqual(@as(u32, 0), e.pins);
}

test "a compaction whose copies are all taken but fail on the device breaks the store: no hit, no span of a pinned pass" {
    const rb = 4;
    var mem: [8 * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 8, rb, f.copy());
    defer s.deinit();
    // A [0,1) unpinned, B [1,5) pinned by a pass, C [5,7) pinned; one row free at the top: no compaction yet
    const a = digestOf(1);
    _ = try s.alloc(&a, 1);
    write(&f, &s, &a);
    s.finish(&a);
    try testing.expect(!s.unpin(&a));
    const ib = [_]Image{mk(3, 2, 2, 2)};
    var pb: Need = try .init(&ib, 0);
    try pb.reserve(&s, &ib);
    try testing.expectEqual(Step{ .begin = 0 }, try pb.plan(&s, &ib, 0, 8));
    write(&f, &s, &ib[0].digest);
    s.finish(&ib[0].digest);
    try testing.expectEqual(Step.ready, try pb.plan(&s, &ib, 0, 8));
    const c = digestOf(3);
    try testing.expectEqual(@as(u32, 5), try s.alloc(&c, 2));
    write(&f, &s, &c);
    s.finish(&c);
    try testing.expectEqual(@as(u32, 0), f.dones);
    // D (2 rows): A goes and B and C move down; every copy is enqueued without error, and the device fault shows only
    // when the moves complete, once for the compaction, before alloc returns
    f.fail_done = true;
    try testing.expectError(error.FakeFault, s.alloc(&digestOf(4), 2));
    try testing.expectEqual(@as(u32, 1), f.dones);
    try testing.expect(f.copies > 0);
    f.fail_done = false;
    // a new lookup of B's pixels is no hit, and the pass that pinned B before the compaction splices nothing
    var q: Need = try .init(&ib, 0);
    try testing.expectError(error.StoreBroken, q.plan(&s, &ib, 0, 8));
    try testing.expect(q.pinned.count() == 0);
    var sp: [max_images]Span = undefined;
    try testing.expectError(error.StoreBroken, pb.spans(&s, &ib, 0, 8, &sp));
    try testing.expectError(error.StoreBroken, s.alloc(&digestOf(5), 1));
    // the passes still end
    try testing.expect(!pb.release(&s, &ib));
    try testing.expect(!s.unpin(&c));
    try testing.expectEqual(@as(u32, 0), s.committed);
}

test "no host allocation after start-up: cold fills to one entry a row, replacement in a full store, compaction" {
    const rb = 4;
    const rows = 6;
    var mem: [rows * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var fa: std.testing.FailingAllocator = .init(testing.allocator, .{});
    var s: Store = try .init(fa.allocator(), f.base, rows, rb, f.copy());
    defer s.deinit();
    fa.fail_index = fa.alloc_index; // every allocation from here on fails
    fa.resize_fail_index = fa.resize_index;
    // cold fills, one 1-row image each, until every row is an entry; the first through a pass
    const im0 = [_]Image{mk(2, 1, 1, 0)};
    var p: Need = try .init(&im0, 0);
    try p.reserve(&s, &im0);
    try testing.expectEqual(Step{ .begin = 0 }, try p.plan(&s, &im0, 0, 4));
    write(&f, &s, &im0[0].digest);
    s.finish(&im0[0].digest);
    _ = p.release(&s, &im0);
    for (1..rows) |k| {
        const d = digestOf(@intCast(k));
        try testing.expectEqual(@as(u32, @intCast(k)), try s.alloc(&d, 1));
        write(&f, &s, &d);
        s.finish(&d);
        try testing.expect(!s.unpin(&d));
    }
    try testing.expectEqual(@as(usize, rows), s.entries.items.len);
    try testing.expectEqual(@as(u32, 0), s.free());
    // a full store: a 1-row image replaces the least recently used entry in its row
    const n1 = digestOf(100);
    try testing.expectEqual(@as(u32, 0), try s.alloc(&n1, 1));
    try testing.expect(s.find(&digestOf(0)) == null and s.evicted == 1);
    write(&f, &s, &n1);
    s.finish(&n1);
    try testing.expect(!s.unpin(&n1));
    // a 3-row image replaces three: n1 [0,1) n3 [1,4) d4 d5
    const n3 = digestOf(101);
    try testing.expectEqual(@as(u32, 1), try s.alloc(&n3, 3));
    write(&f, &s, &n3);
    s.finish(&n3); // n3 stays pinned
    s.pin(&digestOf(5));
    // 2 rows: d4 then n1 go, the free rows [0,1) and [4,5) are split: n3 and d5 move down, the image at 4
    const n2 = digestOf(102);
    try testing.expectEqual(@as(u32, 4), try s.alloc(&n2, 2));
    try testing.expect(s.moved == 4 and s.evicted == 6);
    write(&f, &s, &n2);
    s.finish(&n2);
    try checkStore(&f, &s);
    try testing.expect(!fa.has_induced_failure);
    try testing.expectEqual(fa.fail_index, fa.alloc_index);
}

test "a building entry is aborted when its last pin drops, and not before" {
    var mem: [64]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 16, 4, f.copy());
    defer s.deinit();
    const ims = [_]Image{mk(4, 2, 3, 7)};
    var p: Need = try .init(&ims, 0);
    var q: Need = try .init(&ims, 0);
    try p.reserve(&s, &ims);
    try q.reserve(&s, &ims);
    try testing.expectEqual(Step{ .begin = 0 }, try p.plan(&s, &ims, 0, 8));
    try testing.expectEqual(Step.work, try q.plan(&s, &ims, 0, 8)); // the same image: q pins it and helps
    try testing.expectEqual(@as(u32, 2), s.find(&ims[0].digest).?.pins);
    try testing.expect(!p.release(&s, &ims));
    try testing.expect(s.job() != null);
    try testing.expect(q.release(&s, &ims));
    try testing.expect(s.job() == null);
    try testing.expectEqual(@as(u32, 16), s.free());
    try testing.expectEqual(@as(u32, 0), s.committed);
}

test "help: a pass whose image waits on another image's tower job runs that job's slices" {
    var mem: [64 * 4]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 64, 4, f.copy());
    defer s.deinit();
    const big = [_]Image{mk(10, 5, 6, 1)};
    const small = [_]Image{mk(3, 1, 2, 2)};
    var a: Need = try .init(&big, 0);
    var b: Need = try .init(&small, 0);
    try a.reserve(&s, &big);
    try b.reserve(&s, &small);
    try testing.expectEqual(Step{ .begin = 0 }, try a.plan(&s, &big, 0, 64));
    try testing.expectEqual(Step.work, try b.plan(&s, &small, 0, 64)); // the holder's slice, unbilled
    try testing.expectEqual(Step.work, try b.plan(&s, &small, 0, 64));
    try testing.expectEqual(@as(u32, 1), s.find(&big[0].digest).?.pins); // b never pins a's image
    write(&f, &s, &big[0].digest);
    s.finish(&big[0].digest);
    try testing.expectEqual(Step{ .begin = 0 }, try b.plan(&s, &small, 0, 64));
    try testing.expectEqual(Step.ready, try a.plan(&s, &big, 0, 64)); // a needs nothing of b's
    try testing.expectEqual(Step.work, try b.plan(&s, &small, 0, 64));
    write(&f, &s, &small[0].digest);
    s.finish(&small[0].digest);
    try testing.expectEqual(Step.ready, try b.plan(&s, &small, 0, 64));
    // a later pass with the same pixels is a hit: no tower
    var c: Need = try .init(&big, 0);
    try c.reserve(&s, &big);
    try testing.expectEqual(Step.ready, try c.plan(&s, &big, 0, 64));
    try testing.expectEqual(@as(u32, 2), s.find(&big[0].digest).?.pins);
    _ = a.release(&s, &big);
    _ = b.release(&s, &small);
    _ = c.release(&s, &big);
    try testing.expectEqual(@as(u32, 0), s.committed);
    try checkStore(&f, &s);
}

test "Need.init skips the images a kept prefix holds and refuses bad image lists" {
    const ims = [_]Image{ mk(10, 4, 5, 1), mk(35, 2, 3, 2) }; // [10,30) [35,41)
    try testing.expectEqual(@as(u32, 0), (try Need.init(&ims, 0)).first);
    try testing.expectEqual(@as(u32, 0), (try Need.init(&ims, 29)).first); // inside the first image
    try testing.expectEqual(@as(u32, 1), (try Need.init(&ims, 30)).first);
    try testing.expectEqual(@as(u32, 1), (try Need.init(&ims, 35)).first);
    try testing.expectEqual(@as(u32, 1), (try Need.init(&ims, 40)).first);
    try testing.expectEqual(@as(u32, 2), (try Need.init(&ims, 41)).first);
    const n = try Need.init(&ims, 30);
    try testing.expectEqual(@as(u32, 6), n.tokens(&ims));
    try testing.expectError(error.ImageOrder, Need.init(&.{ ims[1], ims[0] }, 0));
    try testing.expectError(error.ImageOrder, Need.init(&.{ ims[0], mk(29, 1, 1, 3) }, 0));
    _ = try Need.init(&.{ ims[0], mk(30, 1, 1, 3) }, 0); // adjacent is fine
    var g = ims[0];
    g.tokens = 21;
    try testing.expectError(error.ImageGrid, Need.init(&.{g}, 0));
    var z = ims[0];
    z.digest = @splat(0);
    try testing.expectError(error.ImageDigestDiffers, Need.init(&.{z}, 0));
    var many: [max_images + 1]Image = undefined;
    for (&many, 0..) |*m, i| m.* = mk(@intCast(2 * i), 1, 1, @intCast(i));
    _ = try Need.init(many[0..max_images], 0);
    try testing.expectError(error.TooManyImages, Need.init(&many, 0));
}

test "stored: true only when every image the chunk meets is ready in the store; it pins and allocates nothing" {
    const rb = 4;
    var mem: [16 * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 16, rb, f.copy());
    defer s.deinit();
    const ims = [_]Image{ mk(4, 1, 2, 1), mk(10, 2, 2, 2) }; // [4, 6), [10, 14)
    var n: Need = try .init(&ims, 0);
    try testing.expect(n.stored(&s, &ims, 0, 4)); // no image in the chunk
    try testing.expect(!n.stored(&s, &ims, 0, 8)); // image 0 not stored
    try testing.expect(s.entries.items.len == 0 and n.pinned.count() == 0);
    _ = try s.alloc(&ims[0].digest, 2);
    try testing.expect(!n.stored(&s, &ims, 0, 8)); // building, not ready
    s.finish(&ims[0].digest);
    try testing.expect(!s.unpin(&ims[0].digest));
    try testing.expect(n.stored(&s, &ims, 0, 8));
    try testing.expect(!n.stored(&s, &ims, 0, 12)); // image 1 not stored
    try testing.expect(n.stored(&s, &ims, 6, 4)); // between the images
    try testing.expect(n.pinned.count() == 0 and s.find(&ims[0].digest).?.pins == 0);
    try testing.expectEqual(Step.ready, try n.plan(&s, &ims, 0, 8)); // as plan finds it
    try testing.expect(!n.release(&s, &ims));
}

test "spans: inside one image, two images in a chunk, a cut inside an image, a short tail, a resume offset" {
    const rb = 5;
    var mem: [64 * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x2000 };
    var s: Store = try .init(testing.allocator, f.base, 64, rb, f.copy());
    defer s.deinit();
    // a stored image first, so the prompt's images do not start at row 0
    const other = digestOf(99);
    _ = try s.alloc(&other, 3);
    write(&f, &s, &other);
    s.finish(&other);
    const ims = [_]Image{ mk(10, 4, 5, 1), mk(35, 2, 3, 2) }; // [10,30) [35,41), prompt 50
    var n: Need = try .init(&ims, 0);
    try n.reserve(&s, &ims);
    while (true) switch (try n.plan(&s, &ims, 0, 50)) {
        .ready => break,
        .work => unreachable,
        .begin => |i| {
            write(&f, &s, &ims[i].digest);
            s.finish(&ims[i].digest);
        },
    };
    const r0 = s.find(&ims[0].digest).?.row0;
    const r1 = s.find(&ims[1].digest).?.row0;
    try testing.expectEqual(@as(u32, 3), r0);
    var sp: [max_images]Span = undefined;
    const Want = struct { first: u32, n: u32, image: u32, from: u32 };
    const cases = [_]struct { at: u32, len: u32, want: []const Want }{
        .{ .at = 0, .len = 10, .want = &.{} }, // text only
        .{ .at = 12, .len = 5, .want = &.{.{ .first = 0, .n = 5, .image = 0, .from = 2 }} }, // inside one image
        .{ .at = 0, .len = 48, .want = &.{ .{ .first = 10, .n = 20, .image = 0, .from = 0 }, .{ .first = 35, .n = 6, .image = 1, .from = 0 } } },
        .{ .at = 0, .len = 15, .want = &.{.{ .first = 10, .n = 5, .image = 0, .from = 0 }} }, // cut inside: the head
        .{ .at = 15, .len = 20, .want = &.{.{ .first = 0, .n = 15, .image = 0, .from = 5 }} }, // and the rest, ending at I1.at
        .{ .at = 29, .len = 7, .want = &.{ .{ .first = 0, .n = 1, .image = 0, .from = 19 }, .{ .first = 6, .n = 1, .image = 1, .from = 0 } } },
        .{ .at = 39, .len = 2, .want = &.{.{ .first = 0, .n = 2, .image = 1, .from = 4 }} }, // a 2-row tail inside an image
        .{ .at = 30, .len = 32, .want = &.{.{ .first = 5, .n = 6, .image = 1, .from = 0 }} }, // a 32-row chunk
        .{ .at = 41, .len = 9, .want = &.{} },
    };
    for (cases) |c| {
        const k = try n.spans(&s, &ims, c.at, c.len, &sp);
        try testing.expectEqual(c.want.len, k);
        for (c.want, sp[0..k]) |w, got| {
            const row0 = if (w.image == 0) r0 else r1;
            try testing.expectEqual(w.first, got.first);
            try testing.expectEqual(w.n, got.n);
            try testing.expectEqual(s.addr(row0 + w.from), got.src);
            try testing.expect(holds(&f, &s, &ims[w.image].digest, row0 + w.from, w.from, w.n));
        }
    }
    try testing.expectError(error.TooManySpans, n.spans(&s, &ims, 0, 48, sp[0..1]));
    // resume at 17 (a mark inside image 0): its span starts at offset 17 - 10
    var m: Need = try .init(&ims, 17);
    try m.reserve(&s, &ims);
    try testing.expectEqual(Step.ready, try m.plan(&s, &ims, 17, 2048));
    try testing.expectEqual(@as(usize, 2), try m.spans(&s, &ims, 17, 33, &sp));
    try testing.expectEqual(Span{ .first = 0, .n = 13, .src = s.addr(r0 + 7) }, sp[0]);
    try testing.expectEqual(Span{ .first = 18, .n = 6, .src = s.addr(r1) }, sp[1]);
    // passed: image 0 unpinned at 30, not at 29
    try testing.expectEqual(@as(u32, 2), s.find(&ims[0].digest).?.pins);
    try testing.expect(!n.passed(&s, &ims, 29));
    try testing.expectEqual(@as(u32, 2), s.find(&ims[0].digest).?.pins);
    try testing.expect(!n.passed(&s, &ims, 30));
    try testing.expectEqual(@as(u32, 1), s.find(&ims[0].digest).?.pins);
    try testing.expectEqual(@as(u32, 1), n.first);
    try testing.expectEqual(@as(usize, 1), try n.spans(&s, &ims, 30, 20, &sp));
    _ = n.release(&s, &ims);
    _ = m.release(&s, &ims);
    try testing.expectEqual(@as(u32, 0), s.find(&ims[1].digest).?.pins);
    try testing.expectEqual(@as(u32, 0), s.committed);
    // an image that is not stored or not ready gives no spans
    var q: Need = try .init(&ims, 0);
    try testing.expectError(error.ImageNotReady, q.spans(&s, &ims, 0, 48, &sp));
}

test "prefixKey: zeros with no image before the cut, and every field and digest bit changes it" {
    const ims = [_]Image{ mk(10, 4, 5, 1), mk(35, 2, 3, 2) };
    const zero: Digest = @splat(0);
    try testing.expectEqual(zero, prefixKey(&.{}, 100));
    try testing.expectEqual(zero, prefixKey(&ims, 0));
    try testing.expectEqual(zero, prefixKey(&ims, 10)); // the core's `<`: an image at the cut is not in it
    const k11 = prefixKey(&ims, 11);
    try testing.expect(!std.mem.eql(u8, &zero, &k11));
    try testing.expectEqual(k11, prefixKey(&ims, 35));
    try testing.expectEqual(k11, prefixKey(ims[0..1], 1000));
    const k36 = prefixKey(&ims, 36);
    try testing.expect(!std.mem.eql(u8, &k11, &k36));
    var b: [16]u8 = undefined;
    std.mem.writeInt(u32, b[0..4], 10, .little);
    std.mem.writeInt(u32, b[4..8], 20, .little);
    std.mem.writeInt(u32, b[8..12], 4, .little);
    std.mem.writeInt(u32, b[12..16], 5, .little);
    var h = Sha256.init(.{});
    h.update(&b);
    h.update(&ims[0].digest);
    try testing.expectEqual(h.finalResult(), k11);
    var seen: std.AutoHashMapUnmanaged(Digest, void) = .empty;
    defer seen.deinit(testing.allocator);
    try seen.put(testing.allocator, k36, {});
    for (0..2) |which| for (0..256) |bit| {
        var x = ims;
        x[which].digest[bit / 8] ^= @as(u8, 1) << @intCast(bit % 8);
        const k = prefixKey(&x, 36);
        try testing.expect(!seen.contains(k));
        try seen.put(testing.allocator, k, {});
    };
    inline for (.{ "at", "tokens", "rows", "cols" }) |field| {
        var x = ims;
        @field(x[1], field) += 1;
        try testing.expect(!seen.contains(prefixKey(&x, 37)));
        try testing.expect(!std.mem.eql(u8, &k36, &prefixKey(&x, 37)));
    }
}

test "sliceLayers: whole tower unbounded, 1 layer until timed, the budget clamped to [1, left]" {
    try testing.expectEqual(@as(u32, 23), sliceLayers(0, 1000, 5_000_000, 23));
    try testing.expectEqual(@as(u32, 0), sliceLayers(512, 1000, 5_000_000, 0));
    try testing.expectEqual(@as(u32, 1), sliceLayers(512, 1000, 0, 23));
    try testing.expectEqual(@as(u32, 1), sliceLayers(512, 0, 5_000_000, 23));
    try testing.expectEqual(@as(u32, 1), sliceLayers(512, 1000, 5_000_000, 23)); // 0.512 ms budget, 5 ms a layer
    try testing.expectEqual(@as(u32, 4), sliceLayers(2048, 10_000, 5_000_000, 23)); // 20.48 ms / 5 ms
    try testing.expectEqual(@as(u32, 3), sliceLayers(2048, 10_000, 5_000_000, 3));
    try testing.expectEqual(@as(u32, 23), sliceLayers(std.math.maxInt(u32), std.math.maxInt(u64), 1, 23));
    var prng = std.Random.DefaultPrng.init(0x5713);
    const r = prng.random();
    for (0..10_000) |_| {
        const left = r.intRangeAtMost(u32, 1, 23);
        const k = sliceLayers(r.uintLessThan(u32, 4096), r.uintLessThan(u64, 1 << 20), r.uintLessThan(u64, 1 << 30), left);
        try testing.expect(k >= 1 and k <= left); // every unit runs a layer
    }
}

test "random passes: reservations within the store mean alloc never fails, and every move keeps the rows" {
    const gpa = testing.allocator;
    const rows = 48;
    const rb = 4;
    var mem: [rows * rb]u8 = undefined;
    @memset(&mem, 0xaa);
    var f: Fake = .{ .mem = &mem, .base = 0x8000 };
    var s: Store = try .init(gpa, f.base, rows, rb, f.copy());
    defer s.deinit();
    const pool = 14;
    var sizes: [pool][2]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x1a2b3c);
    const r = prng.random();
    for (&sizes) |*z| z.* = .{ r.intRangeAtMost(u32, 1, 4), r.intRangeAtMost(u32, 1, 6) };

    const Pass = struct {
        live: bool = false,
        ims: [4]Image = undefined,
        len: u32 = 0,
        count: usize = 0,
        need: Need = undefined,
        cursor: u32 = 0,
    };
    var passes: [4]Pass = @splat(.{});
    var job: ?struct { digest: Digest, left: u32 } = null;
    var stats: struct { opened: u32 = 0, waited: u32 = 0, begun: u32 = 0, hits: u32 = 0, helped: u32 = 0, chunks: u32 = 0, done: u32 = 0, dropped: u32 = 0, aborted: u32 = 0 } = .{};
    var sp: [max_images]Span = undefined;

    for (0..40_000) |_| {
        const p = &passes[r.uintLessThan(usize, passes.len)];
        if (!p.live) {
            // open: 1-4 images (repeats allowed) with text between, resumed from 0 or a random point
            p.count = r.intRangeAtMost(usize, 1, 4);
            var at: u32 = r.uintLessThan(u32, 4);
            for (p.ims[0..p.count]) |*m| {
                const k = r.uintLessThan(u32, pool);
                m.* = mk(at, sizes[k][0], sizes[k][1], k);
                at += m.tokens + r.uintLessThan(u32, 3);
            }
            p.len = at + 1;
            const from = if (r.boolean()) 0 else r.uintLessThan(u32, p.len);
            p.need = try .init(p.ims[0..p.count], from);
            p.need.reserve(&s, p.ims[0..p.count]) catch {
                stats.waited += 1;
                continue;
            };
            p.cursor = from;
            p.live = true;
            stats.opened += 1;
        } else if (r.uintLessThan(u32, 16) == 0) {
            if (p.need.release(&s, p.ims[0..p.count])) {
                job = null;
                stats.aborted += 1;
            }
            p.live = false;
            stats.dropped += 1;
        } else {
            // one prefill step: a tower unit, or the chunk
            const ims = p.ims[0..p.count];
            const n = @min(r.intRangeAtMost(u32, 1, 12), p.len - p.cursor);
            const hits_before = s.clock;
            switch (try p.need.plan(&s, ims, p.cursor, n)) {
                .begin => |i| {
                    try testing.expect(job == null);
                    job = .{ .digest = ims[i].digest, .left = r.intRangeAtMost(u32, 0, 2) };
                    stats.begun += 1;
                },
                .work => {
                    const j = if (job) |*x| x else return error.NoJob;
                    const own = for (ims, 0..) |m, i| {
                        if (p.need.pinned.isSet(i) and std.mem.eql(u8, &m.digest, &j.digest)) break true;
                    } else false;
                    if (!own) stats.helped += 1;
                    if (j.left == 0) {
                        write(&f, &s, &j.digest);
                        s.finish(&j.digest);
                        job = null;
                    } else j.left -= 1;
                },
                .ready => {
                    if (s.clock != hits_before) stats.hits += 1;
                    const k = try p.need.spans(&s, ims, p.cursor, n, &sp);
                    var covered: u32 = 0;
                    for (sp[0..k]) |x| covered += x.n;
                    var want: u32 = 0;
                    for (ims) |m| {
                        const lo = @max(p.cursor, m.at);
                        const hi = @min(p.cursor + n, m.at + m.tokens);
                        if (hi > lo) want += hi - lo;
                    }
                    try testing.expectEqual(want, covered);
                    for (sp[0..k]) |x| {
                        const pos = p.cursor + x.first;
                        const m = for (ims) |mm| {
                            if (pos >= mm.at and pos < mm.at + mm.tokens) break mm;
                        } else return error.SpanOutsideImage;
                        try testing.expect(x.first + x.n <= n);
                        const e = s.find(&m.digest).?;
                        try testing.expectEqual(s.addr(e.row0 + (pos - m.at)), x.src);
                        try testing.expect(holds(&f, &s, &m.digest, e.row0 + (pos - m.at), pos - m.at, x.n));
                    }
                    p.cursor += n;
                    try testing.expect(!p.need.passed(&s, ims, p.cursor));
                    stats.chunks += 1;
                    if (p.cursor == p.len) {
                        try testing.expect(!p.need.release(&s, ims));
                        p.live = false;
                        stats.done += 1;
                    }
                },
            }
        }
        // the store's invariants after every step
        try checkStore(&f, &s);
        try testing.expectEqual(job != null, s.job() != null);
        var committed: u32 = 0;
        var pins: std.AutoHashMapUnmanaged(Digest, u32) = .empty;
        defer pins.deinit(gpa);
        for (&passes) |*q| if (q.live) {
            committed += q.need.reserved;
            var it = q.need.pinned.iterator(.{});
            while (it.next()) |i| (try pins.getOrPutValue(gpa, q.ims[i].digest, 0)).value_ptr.* += 1;
        };
        try testing.expectEqual(committed, s.committed);
        var pinned_rows: u32 = 0;
        for (s.entries.items) |e| {
            try testing.expectEqual(pins.get(e.digest) orelse 0, e.pins);
            if (e.pins > 0) pinned_rows += e.tokens;
        }
        try testing.expect(pinned_rows <= s.committed);
    }
    // the run went through every path
    try testing.expect(stats.done > 1000 and stats.waited > 100 and stats.begun > 500 and stats.hits > 500);
    try testing.expect(stats.helped > 100 and stats.aborted > 10 and s.evicted > 100 and s.moved > 100);
}

/// The tower's units over fake memory: `begin` and `layers` advance a clock by `layer_ns` a layer (begin a quarter of
/// one), `merge` writes the job's rows (`pattern`); every call is counted as device work.
const FakeTower = struct {
    f: *Fake,
    s: *Store,
    clock: u64 = 1_000,
    layer_ns: u64 = 5_000_000,
    begins: u32 = 0,
    layers_run: u32 = 0,
    merges: u32 = 0,
    settles: u32 = 0,
    reads: u32 = 0,
    work: u32 = 0, // begin, layers and merge calls
    ks: [64]u32 = undefined,
    nk: usize = 0,
    fail: enum { none, begin, layers } = .none,

    fn of(ctx: *anyopaque) *FakeTower {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(ctx: *anyopaque, _: u32) anyerror!void {
        const t = of(ctx);
        if (t.fail == .begin) return error.FakeFault;
        t.begins += 1;
        t.work += 1;
        t.clock += t.layer_ns / 4;
    }
    fn layers(ctx: *anyopaque, k: u32) anyerror!void {
        const t = of(ctx);
        if (t.fail == .layers) return error.FakeFault;
        try testing.expect(k >= 1);
        t.layers_run += k;
        t.work += 1;
        t.clock += k * t.layer_ns;
        if (t.nk < t.ks.len) t.ks[t.nk] = k;
        t.nk += 1;
    }
    fn merge(ctx: *anyopaque, row0: u32) anyerror!void {
        const t = of(ctx);
        const j = t.s.job().?;
        try testing.expectEqual(j.row0, row0);
        write(t.f, t.s, &j.digest);
        t.merges += 1;
        t.work += 1;
    }
    fn settle(ctx: *anyopaque) anyerror!void {
        of(ctx).settles += 1;
    }
    fn now(ctx: *anyopaque) u64 {
        const t = of(ctx);
        t.reads += 1;
        return t.clock;
    }
    fn units(t: *FakeTower) Units {
        return .{ .ctx = t, .begin = begin, .layers = layers, .merge = merge, .settle = settle, .now = now };
    }
};

test "unit: without a row cap each image's tower runs whole and the chunk goes in the same call, with no settle or clock" {
    var mem: [64 * 4]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 64, 4, f.copy());
    defer s.deinit();
    var ft: FakeTower = .{ .f = &f, .s = &s };
    var pace: Pace = .{};
    var help: Help = .{};
    const ims = [_]Image{ mk(2, 2, 3, 1), mk(9, 1, 4, 2) };
    var n: Need = try .init(&ims, 0);
    try n.reserve(&s, &ims);
    try testing.expectEqual(Unit.ready, try unit(&n, &s, &pace, &ims, 0, 20, 0, true, ft.units(), &help));
    try testing.expect(ft.begins == 2 and ft.layers_run == 2 * tower_layers and ft.merges == 2);
    try testing.expectEqual(@as(usize, 2), ft.nk);
    try testing.expect(ft.ks[0] == tower_layers and ft.ks[1] == tower_layers);
    try testing.expect(ft.settles == 0 and ft.reads == 0 and pace.layer_ns == 0);
    try testing.expect(pace.job == null and s.job() == null);
    try testing.expectEqual(Help{}, help);
    var sp: [max_images]Span = undefined;
    try testing.expectEqual(@as(usize, 2), try n.spans(&s, &ims, 0, 20, &sp));
    try checkStore(&f, &s);
    _ = n.release(&s, &ims);
}

test "unit: under a row cap a job's first unit is begin and one timed layer, then the k rule a unit and the merger with the last; each call works until ready" {
    var mem: [64 * 4]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 64, 4, f.copy());
    defer s.deinit();
    const ims = [_]Image{mk(4, 5, 6, 1)};
    // k = max_rows * chunk_ns_per_row / layer_ns: 512 rows at 20 us is 10.24 ms over 5 ms, 2 layers a unit; with no
    // chunk timed yet, 1
    for ([_]struct { ns_per_row: u64, k: u32 }{ .{ .ns_per_row = 20_000, .k = 2 }, .{ .ns_per_row = 0, .k = 1 }, .{ .ns_per_row = 1 << 30, .k = tower_layers - 1 } }) |c| {
        var ft: FakeTower = .{ .f = &f, .s = &s };
        var pace: Pace = .{};
        if (c.ns_per_row > 0) pace.chunked(c.ns_per_row * 1024, 1024);
        var help: Help = .{};
        var n: Need = try .init(&ims, 0);
        try n.reserve(&s, &ims);
        try testing.expectEqual(Unit.worked, try unit(&n, &s, &pace, &ims, 0, 40, 512, false, ft.units(), &help));
        try testing.expect(ft.begins == 1 and ft.layers_run == 1 and ft.merges == 0);
        try testing.expect(ft.settles == 2 and ft.reads == 2); // settled before and after the timed layer
        try testing.expectEqual(ft.layer_ns, pace.layer_ns);
        try testing.expectEqual(tower_layers - 1, pace.left);
        var calls: u32 = 1;
        while (true) {
            const work = ft.work;
            const r = try unit(&n, &s, &pace, &ims, 0, 40, 512, false, ft.units(), &help);
            if (r == .ready) break;
            calls += 1;
            try testing.expect(ft.work > work); // every call makes progress
            try testing.expect(calls <= tower_layers + 1);
        }
        try testing.expectEqual(ft.work, ft.begins + @as(u32, @intCast(ft.nk)) + ft.merges);
        const want: u32 = 1 + (tower_layers - 1 + c.k - 1) / c.k;
        try testing.expectEqual(want, calls);
        try testing.expectEqual(want, pace.units);
        for (ft.ks[1..ft.nk], 1..) |k, i| try testing.expectEqual(if (i == ft.nk - 1) (tower_layers - 1) - c.k * @as(u32, @intCast(ft.nk - 2)) else c.k, k);
        try testing.expect(ft.layers_run == tower_layers and ft.merges == 1 and ft.settles == 2);
        try testing.expect(pace.job == null and s.find(&ims[0].digest).?.ready);
        try testing.expectEqual(Help{}, help);
        _ = n.release(&s, &ims);
        try checkStore(&f, &s);
        s.entries.clearRetainingCapacity(); // the next case's tower runs again
    }
    // the chunk EMA: a quarter the newest
    var p: Pace = .{};
    p.chunked(1000, 0);
    try testing.expectEqual(@as(u64, 0), p.chunk_ns_per_row);
    p.chunked(4000, 100);
    try testing.expectEqual(@as(u64, 40), p.chunk_ns_per_row);
    p.chunked(8000, 100);
    try testing.expectEqual(@as(u64, 50), p.chunk_ns_per_row);
    p.chunked(0, 100);
    try testing.expectEqual(@as(u64, 37), p.chunk_ns_per_row);
}

test "unit: help: a pass whose image waits on another image's job runs its units (unbilled unless settled), the same image's pass is not helping" {
    var mem: [64 * 4]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 64, 4, f.copy());
    defer s.deinit();
    var ft: FakeTower = .{ .f = &f, .s = &s };
    var pace: Pace = .{};
    pace.chunked(20_000, 1); // 2 layers a unit at 512 rows
    const big = [_]Image{mk(10, 5, 6, 1)};
    const small = [_]Image{mk(3, 1, 2, 2)};
    var a: Need = try .init(&big, 0);
    var b: Need = try .init(&small, 0);
    var c: Need = try .init(&big, 0); // the same pixels as a's
    var ha: Help = .{};
    var hb: Help = .{};
    var hc: Help = .{};
    try a.reserve(&s, &big);
    try b.reserve(&s, &small);
    try c.reserve(&s, &big);
    try testing.expectEqual(Unit.worked, try unit(&a, &s, &pace, &big, 0, 64, 512, true, ft.units(), &ha));
    try testing.expectEqual(Unit.worked, try unit(&b, &s, &pace, &small, 0, 64, 512, false, ft.units(), &hb));
    try testing.expectEqual(Unit.worked, try unit(&b, &s, &pace, &small, 0, 64, 512, true, ft.units(), &hb));
    try testing.expectEqual(Help{ .units = 2, .layers = 4, .unbilled = 1 }, hb);
    try testing.expect(!b.holds(&small, &big[0].digest) and s.find(&big[0].digest).?.pins == 1); // b pins none of a's
    try testing.expectEqual(Unit.worked, try unit(&c, &s, &pace, &big, 0, 64, 512, false, ft.units(), &hc));
    try testing.expectEqual(Help{}, hc); // c's own image: c pins it, no help
    try testing.expectEqual(@as(u32, 2), s.find(&big[0].digest).?.pins);
    try testing.expectEqual(tower_layers - 7, pace.left);
    // b drives the job to its end; then its own image begins
    var calls: u32 = 0;
    while (s.find(&big[0].digest).?.ready == false) : (calls += 1) {
        try testing.expectEqual(Unit.worked, try unit(&b, &s, &pace, &small, 0, 64, 512, false, ft.units(), &hb));
    }
    try testing.expectEqual(@as(u32, 8), calls);
    try testing.expectEqual(Help{ .units = 10, .layers = 20, .unbilled = 9 }, hb);
    try testing.expect(pace.help_units == 10 and pace.unbilled_help == 9);
    try testing.expectEqual(ft.begins, 1);
    try testing.expectEqual(Unit.worked, try unit(&b, &s, &pace, &small, 0, 64, 512, false, ft.units(), &hb));
    try testing.expectEqual(ft.begins, 2);
    try testing.expect(pace.holds(&small[0].digest));
    try testing.expectEqual(Unit.ready, try unit(&a, &s, &pace, &big, 0, 64, 512, true, ft.units(), &ha)); // a needs nothing of b's
    try testing.expectEqual(Help{}, ha);
    _ = a.release(&s, &big);
    _ = c.release(&s, &big);
    try testing.expect(b.release(&s, &small)); // b's job aborted with its last pin
    try testing.expectEqual(@as(u32, 0), s.committed);
    try checkStore(&f, &s);
}

test "unit: a failed begin drops its entry; a failed later unit leaves the job unheld, refused until its pins drop" {
    var mem: [64 * 4]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x1000 };
    var s: Store = try .init(testing.allocator, f.base, 64, 4, f.copy());
    defer s.deinit();
    var ft: FakeTower = .{ .f = &f, .s = &s, .fail = .begin };
    var pace: Pace = .{};
    var help: Help = .{};
    const ims = [_]Image{mk(4, 2, 2, 1)};
    var n: Need = try .init(&ims, 0);
    try n.reserve(&s, &ims);
    try testing.expectError(error.FakeFault, unit(&n, &s, &pace, &ims, 0, 16, 512, false, ft.units(), &help));
    try testing.expect(s.entries.items.len == 0 and n.pinned.count() == 0 and pace.job == null);
    ft.fail = .none;
    try testing.expectEqual(Unit.worked, try unit(&n, &s, &pace, &ims, 0, 16, 512, false, ft.units(), &help));
    ft.fail = .layers;
    try testing.expectError(error.FakeFault, unit(&n, &s, &pace, &ims, 0, 16, 512, false, ft.units(), &help));
    try testing.expect(pace.job == null and s.job() != null);
    ft.fail = .none;
    try testing.expectError(error.TowerJobLost, unit(&n, &s, &pace, &ims, 0, 16, 512, false, ft.units(), &help));
    try testing.expect(n.release(&s, &ims));
    try testing.expect(s.job() == null and s.committed == 0);
}

test "unit: random passes under random caps: every worked call did device work, and every pass reaches its end" {
    const gpa = testing.allocator;
    const rows = 40;
    const rb = 4;
    var mem: [rows * rb]u8 = undefined;
    var f: Fake = .{ .mem = &mem, .base = 0x8000 };
    var s: Store = try .init(gpa, f.base, rows, rb, f.copy());
    defer s.deinit();
    var ft: FakeTower = .{ .f = &f, .s = &s };
    var pace: Pace = .{};
    var prng = std.Random.DefaultPrng.init(0x9e37);
    const r = prng.random();
    const Pass = struct { live: bool = false, ims: [3]Image = undefined, count: usize = 0, len: u32 = 0, need: Need = undefined, cursor: u32 = 0, calls: u32 = 0, help: Help = .{} };
    var passes: [3]Pass = @splat(.{});
    var stats: struct { done: u32 = 0, worked: u32 = 0, helped: u64 = 0 } = .{};
    for (0..20_000) |_| {
        const p = &passes[r.uintLessThan(usize, passes.len)];
        if (!p.live) {
            p.count = r.intRangeAtMost(usize, 1, 3);
            var at: u32 = r.uintLessThan(u32, 3);
            for (p.ims[0..p.count]) |*m| {
                const k = r.uintLessThan(u32, 40); // more images than the store holds: misses and evictions
                m.* = mk(at, 1 + k % 3, 1 + k % 4, k);
                at += m.tokens + r.uintLessThan(u32, 3);
            }
            p.len = at + 1;
            p.need = try .init(p.ims[0..p.count], 0);
            p.need.reserve(&s, p.ims[0..p.count]) catch continue;
            p.* = .{ .live = true, .ims = p.ims, .count = p.count, .len = p.len, .need = p.need };
            continue;
        }
        const ims = p.ims[0..p.count];
        const n = @min(r.intRangeAtMost(u32, 1, 8), p.len - p.cursor);
        if (r.uintLessThan(u32, 8) == 0) pace.chunked(r.uintLessThan(u64, 1 << 24), r.intRangeAtMost(u64, 1, 2048));
        const cap: u32 = switch (r.uintLessThan(u32, 3)) {
            0 => 0,
            1 => 32,
            else => 512,
        };
        const work = ft.work;
        p.calls += 1;
        try testing.expect(p.calls < 1000); // a pass's calls are bounded: each one works or goes on
        switch (try unit(&p.need, &s, &pace, ims, p.cursor, n, cap, r.boolean(), ft.units(), &p.help)) {
            .worked => {
                try testing.expect(cap > 0 and ft.work > work);
                stats.worked += 1;
            },
            .ready => {
                var sp: [max_images]Span = undefined;
                _ = try p.need.spans(&s, ims, p.cursor, n, &sp);
                p.cursor += n;
                if (p.need.passed(&s, ims, p.cursor)) pace.job = null;
                if (p.cursor == p.len) {
                    if (p.need.release(&s, ims)) pace.job = null;
                    stats.helped += p.help.units;
                    p.live = false;
                    stats.done += 1;
                }
            },
        }
        try checkStore(&f, &s);
        try testing.expect(s.job() == null or pace.holds(&s.job().?.digest));
    }
    try testing.expect(stats.done > 1000 and stats.worked > 1000 and stats.helped > 500 and s.evicted > 100);
    for (passes) |q| if (q.live) {
        stats.helped += q.help.units;
    };
    try testing.expectEqual(stats.helped, pace.help_units); // each pass's help is the pace's
}
