//! Where the embedding table [201,024, 4,608] bf16 (1.73 GiB) lives at W > 1: ranks 1 .. W-1 hold it by vocabulary
//! rows (canon's rule over W - 1 ranks: 67,008 rows, 0.58 GiB a rank at W = 4), rank 0 none: rank 0, the rank with
//! the least room (the HTTP side, the sampler, the head's logits), gets 1.73 GiB back. At W = 2 rank 1 holds the whole
//! table. W = 1 and --solo (no group): rank 0 holds the whole table and embeds a forward's rows itself.
//!
//! Start-up (`spread`): the rank folders are unchanged (the plan still places the table on rank 0, whose digests check
//! it). Rank 0 loads the whole table and its norm, then broadcasts both in 64 MiB slices over the group; each rank
//! keeps the rows it holds and rank 0 frees the table, before any room is measured.
//!
//! A forward (`rows`): every rank has the forward's token ids on the device (a captured step reads the step header's
//! copy, the drafter's walk copied into it on every rank as on rank 0; an eager forward of at most `graph_rows` rows
//! carries its ids in its header, a prompt chunk gets rank 0's ids by one broadcast, 4 B a row; the drafter's anchors
//! come in the header). Each rank copies the rows it holds into its slot of [W][R][H] bf16 (zeros for the rest), the
//! slots are all-gathered in rank order (the group's gather: the RoCE wire for a decode step's rows, NCCL for a
//! prompt chunk) and every rank copies each row from its owner's slot, then norms the rows with the kernel rank 0
//! uses at W = 1: every rank holds the bits rank 0's own embedding gives. No floating-point sum combines the shares
//! (-0 + +0 would change a bit).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ops = @import("cuda_ops.zig");
const head = @import("cuda_head.zig");
const weights = @import("cuda_weights.zig");
const vision_store = ink.vision_store;

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;
const Span = ink.plan.Span;

/// The first rank holding rows (the owners are ranks 1 .. world - 1).
const first: u32 = 1;

/// The rows rank `rank` of `world` holds (vocabulary entries; the whole table on rank 0 at W = 1).
pub fn rowsOf(vocab: u32, rank: u32, world: u32) Span {
    if (world < 2) return if (rank == 0) .{ .lo = 0, .hi = vocab } else .{ .lo = 0, .hi = 0 };
    if (rank < first) return .{ .lo = 0, .hi = 0 };
    return ink.plan.canon(vocab, rank - first, world - first);
}

/// The rank holding entry `id` at W > 1 (tf_ink2_head_embed_pick's rule).
pub fn owner(vocab: u32, world: u32, id: u32) u32 {
    const n = world - first;
    const base = vocab / n;
    const extra = vocab % n;
    const edge = extra * (base + 1);
    const q = if (id < edge) id / (base + 1) else extra + (id - edge) / base;
    return first + @min(q, n - 1);
}

/// A rank's view of the table held by vocabulary rows (Engine.shared; null at W = 1 and --solo).
pub const Shared = struct {
    rank: u32,
    world: u32,
    vocab: u32,
    /// The gather's slots, bf16 [world][R][H] for R up to the exchange's capacity (the attention partials' buffer,
    /// idle outside a layer's exchange).
    slots: u64,
    group: ?*const cuda.tp_group.Group,

    /// `out` bf16 [R, H] = the normed embedding of `ids` (i32 [R] on the device, the same on every rank), on every
    /// rank; `raw` bf16 [R, H] is scratch (the looked-up rows before the norm).
    pub fn rows(s: *const Shared, c: *const Ctx, w: *const weights.Weights, ids: u64, raw: Mat, out: Mat) !void {
        const R = out.rows;
        if (raw.rows != R or raw.cols != H or out.cols != H) return error.BadHeadShape;
        const mine = s.slots + @as(u64, s.rank) * R * H * 2;
        if (w.embed_rows.len() > 0) try head.embedShare(c, w, ids, Mat.of(mine, R, H, .bf16));
        if (s.group) |g| try g.allGather(c.stream, mine, s.slots, R * H * 2);
        try head.embedPick(c, ids, s.slots, first, s.world - first, s.vocab, raw);
        try head.embedNorm(c, w, raw, out);
    }
};

/// An image chunk's splice (cuda_follow.zig Leader.forwardImages, kind `image_forward`): rows [lo, hi) of the forward
/// hold image rows. Rank 0 also holds `spans`: span i's rows [first, first + n) take its vision store's rows at `src`
/// (cuda_vision_store.zig Need.spans, computed after any compaction). The other ranks hold `lo` and `hi` alone (the
/// header's `imageRows`). The default (all zero) is a text forward's: nothing is spliced.
pub const Splice = struct {
    /// The spans an image chunk splices at most: the images a request may hold (--vision-max-images is refused
    /// above it at start-up, cuda_vision.zig checkLimits).
    pub const max_spans = vision_store.max_images;
    pub const Piece = vision_store.Span;

    spans: [max_spans]Piece = undefined,
    count: u32 = 0,
    lo: u32 = 0,
    hi: u32 = 0,

    comptime {
        std.debug.assert(max_spans == 16 and vision_store.row_bytes == H * 2);
    }

    /// A text forward's (the default): nothing to splice.
    pub fn empty(s: *const Splice) bool {
        return s.count == 0 and s.lo == 0 and s.hi == 0;
    }

    /// The rows of a forward of R rows: lo < hi <= R (every rank; the other ranks hold no spans).
    pub fn checkRows(s: *const Splice, R: usize) !void {
        if (s.lo >= s.hi or s.hi > R) return error.BadSplice;
    }

    /// Rank 0, before the header: 1 .. max_spans spans of at least one row, in row order without overlap, the
    /// first at `lo` and the last ending at `hi` <= R (text rows may sit between two spans).
    pub fn check(s: *const Splice, R: usize) !void {
        try s.checkRows(R);
        if (s.count == 0 or s.count > max_spans) return error.BadSplice;
        var at: u64 = s.lo;
        for (s.spans[0..s.count]) |p| {
            if (p.n == 0 or p.src == 0 or p.first < at) return error.BadSplice;
            at = @as(u64, p.first) + p.n;
        }
        if (s.spans[0].first != s.lo or at != s.hi) return error.BadSplice;
    }

    /// Rank 0: every span's rows lie inside the store's `bytes` from `base`.
    pub fn within(s: *const Splice, base: u64, bytes: u64) !void {
        for (s.spans[0..@min(s.count, max_spans)]) |p| {
            const len = @as(u64, p.n) * vision_store.row_bytes;
            if (p.src < base or p.src - base > bytes or len > bytes - (p.src - base)) return error.BadSplice;
        }
    }
};

/// head.image (cuda_forward.zig Forward.images), after the embedding's norm (the tower's rows are never normed): rank
/// 0 copies each span's stored rows into `out` (bf16 [R, H], the forward's embedded rows); with a group every rank
/// then takes rows [lo, hi) from rank 0 by one in-place broadcast (text rows inside it already hold rank 0's bits, so
/// it changes none of them). W = 1 and --solo: the copies alone.
pub fn splice(c: *const Ctx, g: ?*const cuda.tp_group.Group, rank: u32, sp: *const Splice, out: Mat) !void {
    if (out.cols != H or out.dt != .bf16 or !out.dense()) return error.BadHeadShape;
    if (rank == 0) try sp.check(out.rows) else try sp.checkRows(out.rows);
    if (rank != 0 and sp.count != 0) return error.BadSplice;
    const row: usize = H * 2;
    if (rank == 0) for (sp.spans[0..sp.count]) |p| try c.copy(out.row(p.first), p.src, @as(usize, p.n) * row);
    if (g) |gr| {
        const at = out.row(sp.lo);
        try gr.lib.check(gr.lib.api.ncclBroadcast(at, at, @as(usize, sp.hi - sp.lo) * row, .u8, 0, gr.comm, c.stream.handle), "ncclBroadcast");
    }
}

/// A slice of the start-up broadcast (rows of the table).
const slice_bytes: usize = 64 << 20;

/// Start-up, every rank of `g` (W > 1; after the load and its digest check, before any room is read): rank 0's table
/// and norm broadcast in slices; each rank keeps its rows (w.embed, w.embed_rows) and the norm; rank 0 frees the
/// table. Waits for the device.
pub fn spread(d: *const cuda.Driver, s: cuda.Stream, g: *const cuda.tp_group.Group, w: *weights.Weights, rank: u32, world: u32) !void {
    if (world < 2) return error.BadRank;
    const vocab = w.vocab;
    const row: usize = H * 2;
    const mine = rowsOf(vocab, rank, world);
    const leader = rank == 0;
    if (leader and (w.embed == 0 or w.embed_rows.len() != vocab or w.embed_norm == 0)) return error.EmbedNotLoaded;
    if (!leader) w.embed_norm = try w.alloc(H * 4);
    try g.lib.check(g.lib.api.ncclBroadcast(w.embed_norm, w.embed_norm, H * 4, .u8, 0, g.comm, s.handle), "ncclBroadcast");
    const share: u64 = if (mine.len() == 0) 0 else try w.alloc(@as(usize, mine.len()) * row);
    var staging: ?cuda.DeviceBuffer = if (leader) null else try cuda.DeviceBuffer.alloc(d, slice_bytes);
    defer if (staging) |*b| b.free();
    const per: u32 = @intCast(slice_bytes / row);
    var lo: u32 = 0;
    while (lo < vocab) : (lo += per) {
        const hi = @min(vocab, lo + per);
        const src: u64 = if (leader) w.embed + @as(u64, lo) * row else staging.?.ptr;
        try g.lib.check(g.lib.api.ncclBroadcast(src, src, @as(usize, hi - lo) * row, .u8, 0, g.comm, s.handle), "ncclBroadcast");
        const a = @max(lo, mine.lo);
        const b = @min(hi, mine.hi);
        if (a < b) {
            const view: cuda.DeviceBuffer = .{ .d = d, .ptr = share + @as(u64, a - mine.lo) * row, .len = @as(usize, b - a) * row, .borrowed = true };
            try view.copyFrom(0, src + @as(u64, a - lo) * row, @as(usize, b - a) * row, s.handle);
        }
    }
    try s.synchronize();
    if (leader) {
        const whole = w.embed;
        const bytes = @as(u64, vocab) * row;
        if (!w.release(whole)) return error.EmbedNotLoaded;
        w.released -= bytes; // not a repacked source (the log's "dropped" figure)
    }
    w.embed = share;
    w.embed_rows = mine;
}

test "inkling2 cuda embed: every entry has one owner, the rank whose rows hold it" {
    const vocab: u32 = 201024;
    for ([_]u32{ 2, 3, 4, 8 }) |world| {
        var total: u32 = 0;
        for (0..world) |r| total += rowsOf(vocab, @intCast(r), world).len();
        try std.testing.expectEqual(vocab, total);
        var id: u32 = 0;
        while (id < vocab) : (id += 1) {
            const o = owner(vocab, world, id);
            try std.testing.expect(o >= 1 and o < world);
            const span = rowsOf(vocab, o, world);
            try std.testing.expect(id >= span.lo and id < span.hi);
        }
    }
    try std.testing.expectEqual(@as(u32, 0), rowsOf(vocab, 0, 4).len());
    try std.testing.expectEqual(@as(u32, 67008), rowsOf(vocab, 3, 4).len());
    try std.testing.expectEqual(vocab, rowsOf(vocab, 1, 2).len());
    try std.testing.expectEqual(vocab, rowsOf(vocab, 0, 1).len());
}

test "inkling2 cuda embed: an image chunk's splice is refused out of bounds or out of order" {
    const S = Splice;
    const empty: S = .{};
    try std.testing.expect(empty.empty());
    // two images in one chunk of 64 rows, text rows 20 .. 30 between them
    var sp: S = .{ .count = 2, .lo = 5, .hi = 40 };
    sp.spans[0] = .{ .first = 5, .n = 15, .src = 0x1000 };
    sp.spans[1] = .{ .first = 30, .n = 10, .src = 0x1000 + 15 * vision_store.row_bytes };
    try sp.check(64);
    try sp.check(40);
    try std.testing.expect(!sp.empty());
    try sp.within(0x1000, 25 * vision_store.row_bytes);
    try std.testing.expectError(error.BadSplice, sp.within(0x1000, 25 * vision_store.row_bytes - 1)); // past the end
    try std.testing.expectError(error.BadSplice, sp.within(0x1001, 25 * vision_store.row_bytes)); // below the base
    try std.testing.expectError(error.BadSplice, sp.within(0, 0x1000)); // starting at the end
    try std.testing.expectError(error.BadSplice, sp.check(39)); // hi past R
    var bad = sp;
    bad.count = 0;
    try std.testing.expectError(error.BadSplice, bad.check(64));
    bad = sp;
    bad.count = S.max_spans + 1;
    try std.testing.expectError(error.BadSplice, bad.check(64));
    bad = sp;
    bad.lo = 40;
    try std.testing.expectError(error.BadSplice, bad.check(64)); // empty rows
    bad = sp;
    bad.lo = 4;
    try std.testing.expectError(error.BadSplice, bad.check(64)); // lo is not the first span's row
    bad = sp;
    bad.hi = 41;
    try std.testing.expectError(error.BadSplice, bad.check(64)); // hi is not the last span's end
    bad = sp;
    bad.spans[1].first = 19;
    bad.hi = 29;
    try std.testing.expectError(error.BadSplice, bad.check(64)); // overlapping
    bad = sp;
    bad.spans[0] = sp.spans[1];
    bad.spans[1] = sp.spans[0];
    try std.testing.expectError(error.BadSplice, bad.check(64)); // out of order
    bad = sp;
    bad.spans[1].n = 0;
    try std.testing.expectError(error.BadSplice, bad.check(64));
    bad = sp;
    bad.spans[0].src = 0;
    try std.testing.expectError(error.BadSplice, bad.check(64));
    // the most spans a chunk takes: 16 one-row images, one row apart
    var many: S = .{ .count = S.max_spans, .lo = 0, .hi = 2 * S.max_spans - 1 };
    for (many.spans[0..S.max_spans], 0..) |*p, i| p.* = .{ .first = @intCast(2 * i), .n = 1, .src = 0x100 + i * vision_store.row_bytes };
    try many.check(2048);
    // a follower holds rows alone: lo < hi <= R
    const rows: S = .{ .lo = 0, .hi = 2048 };
    try rows.checkRows(2048);
    try std.testing.expectError(error.BadSplice, rows.checkRows(2047));
    try std.testing.expectError(error.BadSplice, (S{ .lo = 7, .hi = 7 }).checkRows(64));
    try std.testing.expectError(error.BadSplice, rows.check(2048)); // rank 0 needs its spans
    // splice: every refusal comes before the first launch (no device here)
    const c: Ctx = undefined;
    var buf: [1]u8 = undefined;
    const out = Mat.of(@intFromPtr(&buf), 64, H, .bf16);
    try std.testing.expectError(error.BadSplice, splice(&c, null, 1, &sp, out)); // a follower holds no spans
    try std.testing.expectError(error.BadSplice, splice(&c, null, 0, &rows, out));
    try std.testing.expectError(error.BadSplice, splice(&c, null, 0, &sp, Mat.of(out.ptr, 39, H, .bf16)));
    try std.testing.expectError(error.BadHeadShape, splice(&c, null, 0, &sp, Mat.of(out.ptr, 64, H - 1, .bf16)));
    try std.testing.expectError(error.BadHeadShape, splice(&c, null, 0, &sp, Mat.of(out.ptr, 64, H, .f32)));
}
