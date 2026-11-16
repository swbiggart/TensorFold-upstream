//! A forward's activation buffers (`Act`, sized for the widest forward) and one stream's state across forwards
//! (`State`: each layer's keys and values and its four convolutions' last 3 input rows), with the scaffold's own
//! state steps: the cache write, the convolution state commits (a forward's, a verify window's keeping its inputs,
//! and `keep`'s rollback, one launch each) and the block push.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");
const attn = @import("cuda_attn.zig");
const conv = @import("cuda_conv.zig");
const pages = @import("cuda_pages.zig");

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;
const D = weights.head_dim;

/// Completed block sources at most (layers 0, 6, ..., 42).
pub const max_blocks = 8;
pub const topk = 8;
pub const experts = 384;

/// Every activation of one forward, `cap` rows each (a forward uses the first R).
pub const Act = struct {
    gpa: std.mem.Allocator,
    cap: usize,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,
    /// Layer input and output, swapped after each layer.
    h: [2]Mat = undefined,
    x: [3]Mat = undefined, // fp32 mix outputs (attention input queries)
    xn: [3]Mat = undefined, // bf16 normed sources
    q: Mat = undefined,
    k: Mat = undefined,
    v: Mat = undefined,
    rel: Mat = undefined,
    gate: Mat = undefined,
    sink: Mat = undefined,
    dk: Mat = undefined,
    dv: Mat = undefined,
    kc: Mat = undefined,
    vc: Mat = undefined,
    qh: Mat = undefined,
    kh: Mat = undefined,
    vh: Mat = undefined,
    relh: Mat = undefined,
    o: Mat = undefined,
    ao: Mat = undefined,
    a: Mat = undefined,
    partial: Mat = undefined,
    mx: Mat = undefined,
    m: Mat = undefined,
    ids: Mat = undefined,
    wts: Mat = undefined,
    logits: Mat = undefined,
    y: Mat = undefined,
    cm: Mat = undefined,
    /// Token ids (i32) for the embedding.
    tokens: Mat = undefined,
    /// The blocks: max_blocks matrices [cap, H] bf16, `nblocks` of them complete.
    blocks: u64 = 0,
    nblocks: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, hq: usize, hk: usize) !Act {
        var a: Act = .{ .gpa = gpa, .cap = cap };
        errdefer a.deinit();
        for (&a.h) |*m| m.* = try a.mat(d, H, .bf16);
        for (&a.x) |*m| m.* = try a.mat(d, H, .f32);
        for (&a.xn) |*m| m.* = try a.mat(d, H, .bf16);
        a.q = try a.mat(d, hq * D, .bf16);
        a.k = try a.mat(d, hk * D, .bf16);
        a.v = try a.mat(d, hk * D, .bf16);
        a.rel = try a.mat(d, hq * weights.d_rel, .bf16);
        a.gate = try a.mat(d, hq * D, .bf16);
        a.sink = try a.mat(d, hq, .bf16);
        a.dk = try a.mat(d, hk * weights.taps, .bf16);
        a.dv = try a.mat(d, hk * weights.taps, .bf16);
        a.kc = try a.mat(d, hk * D, .bf16);
        a.vc = try a.mat(d, hk * D, .bf16);
        a.qh = try a.mat(d, hq * D, .bf16);
        a.kh = try a.mat(d, hk * D, .bf16);
        a.vh = try a.mat(d, hk * D, .bf16);
        a.relh = try a.mat(d, hq * weights.d_rel, .bf16);
        a.o = try a.mat(d, hq * D, .bf16);
        a.ao = try a.mat(d, H, .bf16);
        a.a = try a.mat(d, H, .bf16);
        a.partial = try a.mat(d, H, .f32);
        a.mx = try a.mat(d, H, .f32);
        a.m = try a.mat(d, H, .bf16);
        a.ids = try a.mat(d, topk, .i32);
        a.wts = try a.mat(d, topk, .f32);
        a.logits = try a.mat(d, experts, .f32);
        a.y = try a.mat(d, H, .bf16);
        a.cm = try a.mat(d, H, .bf16);
        a.tokens = try a.mat(d, 1, .i32);
        a.blocks = try a.raw(d, max_blocks * a.blockStride());
        return a;
    }

    pub fn deinit(a: *Act) void {
        for (a.own.items) |*b| b.free();
        a.own.deinit(a.gpa);
        a.* = undefined;
    }

    fn raw(a: *Act, d: *const cuda.Driver, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        try a.own.append(a.gpa, b);
        a.bytes += bytes;
        return b.ptr;
    }

    fn mat(a: *Act, d: *const cuda.Driver, cols: usize, dt: ops.Dt) !Mat {
        return Mat.of(try a.raw(d, a.cap * cols * dt.size()), a.cap, cols, dt);
    }

    pub fn blockStride(a: *const Act) usize {
        return a.cap * H * 2;
    }

    /// The completed blocks for a forward of `rows` rows.
    pub fn blockView(a: *const Act, rows: usize) ops.Blocks {
        return .{ .ptr = a.blocks, .stride = a.blockStride(), .count = a.nblocks, .rows = rows, .hidden = H };
    }

    /// Block `i` (complete or not) for `rows` rows.
    pub fn block(a: *const Act, i: usize, rows: usize) Mat {
        return Mat.of(a.blocks + i * a.blockStride(), rows, H, .bf16);
    }
};

/// Rows of a verify window at most: a forward this narrow on a slot with drafts keeps each convolution's inputs, so
/// `keep` can take the state back to any of its rows (DSpark: the anchor and 7 drafts).
pub const spec_rows = 8;

/// One layer's state for one stream.
pub const LayerState = struct {
    kv: attn.Kv,
    /// bf16 [3, C] each: K and V (C = hk*128), post-attention and post-MLP (C = 4608).
    conv_k: u64,
    conv_v: u64,
    conv_attn: u64,
    conv_mlp: u64,
    /// With drafts: each convolution's [state before the forward; the forward's input rows], bf16 [3 + spec_rows, C]
    /// (k, v, attn, mlp); 0 without.
    hist: [4]u64 = @splat(0),
};

/// One stream's state over the loaded layers.
pub const State = struct {
    gpa: std.mem.Allocator,
    first: u32,
    layers: []LayerState,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,
    /// Positions held (the next forward starts here).
    pos: usize = 0,
    context: usize,
    /// The last forward's first position and rows, and whether it kept its convolution inputs (a verify window).
    last_pos0: usize = 0,
    last_rows: usize = 0,
    kept_inputs: bool = false,
    /// The last forward's row this stream's rows started at (a batched forward's segment; the drafter's taps).
    last_row0: usize = 0,
    /// The engine's forward count at this stream's last forward (cuda_engine.zig Engine.forwards): the drafter's
    /// taps are this stream's while no other forward came after it.
    last_fwd: u64 = 0,
    /// With the convolutions' input history: `keepAll`'s table on the device, a `KeepEntry` for each convolution of
    /// each layer (4 a layer), and its entries.
    keep_table: u64 = 0,
    keep_entries: u32 = 0,
    /// Global layers on pages (cuda_pages.zig): where they live, the slot's page table (i32 ids on the device at a
    /// fixed address, `entries` of them; unmapped entries name the scrub page), its host copy, and the entries that
    /// name the slot's own pages (positions [0, mapped * page_rows) may run). Null place: a window a layer (v0.1).
    place: ?pages.Place = null,
    table: u64 = 0,
    ids: []u32 = &.{},
    mapped: usize = 0,
    /// On pages: the first page when the slot's mapped pages are one run of consecutive ids (the allocator's normal
    /// case), else null. Its one-row forwards then read and write that run through the v0.1 ring kernels (`kvFor`;
    /// TF_INK2_RUN_RING=0: never), every other forward through the page table.
    run: ?u32 = null,
    ring_on_run: bool = true,
    /// The mode of this stream's last one-row forward (`noteOneRow`: counted when it changes).
    last_one: enum { none, ring, paged } = .none,
    /// Tooling (own pages, TF_INK2_PAGE_ORDER=move): each new sequence's run (`restart`), and the one-row forwards
    /// counted so far; while `flip` is set, every other `flip` one-row forwards take the page table (mode switches
    /// mid-stream).
    moves: u32 = 0,
    ones: u64 = 0,
    flip: u32 = 0,

    /// Where a slot's global layers live: a window each (v0.1), the served streams' shared pool, or pages of its own
    /// (tooling: the slot's whole context mapped at once, consecutive ids or, with `shuffle`, a permutation; with
    /// `move`, an arena of two contexts in which each new sequence takes another run, see `restart`).
    /// `drafter_hk` (own pages): the drafter's key heads when its global layer shares the slot's pages (stage 2c: two
    /// more regions; 0: no drafter).
    pub const Storage = union(enum) { window, pool: pages.Place, own: struct { shuffle: bool = false, move: bool = false, drafter_hk: usize = 0 } };

    /// Room for `context` positions, forwards of at most `max_rows` rows; `spec`: the convolutions' input history.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, context: usize, max_rows: usize, spec: bool) !State {
        return initOn(gpa, d, w, context, max_rows, spec, .window);
    }

    /// `init` with the global layers on `storage`.
    pub fn initOn(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, context: usize, max_rows: usize, spec: bool, storage: Storage) !State {
        var s: State = .{ .gpa = gpa, .first = w.first, .layers = try gpa.alloc(LayerState, w.layers.len), .context = context, .ring_on_run = pages.ringOnRun() };
        errdefer s.deinit();
        if (storage != .window) try s.pageTable(gpa, d, w, context, storage);
        var g: usize = 0;
        for (w.layers, s.layers) |l, *st| {
            const cols = l.hk * D;
            const slots = if (l.global) context else @min(context, attn.window - 1 + max_rows);
            var kv: attn.Kv = undefined;
            if (l.global and s.place != null) {
                // global layer g: keys in region 2g, values in 2g + 1, position p at row p % page_rows of page ids[p / page_rows]
                const pl = s.place.?;
                kv = .{ .k = pl.region(2 * g), .v = pl.region(2 * g + 1), .slots = s.ids.len * pages.page_rows, .heads = l.hk, .window = 0, .pages = s.table, .page = pl.elems() };
                g += 1;
            } else kv = .{ .k = try s.zeroed(d, slots * cols * 2), .v = try s.zeroed(d, slots * cols * 2), .slots = slots, .heads = l.hk, .window = if (l.global) 0 else attn.window };
            st.* = .{
                .kv = kv,
                .conv_k = try s.zeroed(d, conv.state_rows * cols * 2),
                .conv_v = try s.zeroed(d, conv.state_rows * cols * 2),
                .conv_attn = try s.zeroed(d, conv.state_rows * H * 2),
                .conv_mlp = try s.zeroed(d, conv.state_rows * H * 2),
            };
            if (spec) for (&st.hist, [4]usize{ cols, cols, H, H }) |*h, c| {
                h.* = try s.zeroed(d, (conv.state_rows + spec_rows) * c * 2);
            };
        }
        if (spec) {
            const table = try gpa.alloc(KeepEntry, 4 * w.layers.len);
            defer gpa.free(table);
            for (s.layers, w.layers, 0..) |st, l, i| {
                const kvc: u32 = @intCast(@as(usize, l.hk) * D);
                for (st.hist, [4]u64{ st.conv_k, st.conv_v, st.conv_attn, st.conv_mlp }, [4]u32{ kvc, kvc, H, H }, 0..) |h, x, c, j| table[4 * i + j] = .{ .state = x, .hist = h, .cols = c };
            }
            var b = try cuda.DeviceBuffer.alloc(d, table.len * @sizeOf(KeepEntry));
            errdefer b.free();
            try b.upload(0, std.mem.sliceAsBytes(table));
            try s.own.append(gpa, b);
            s.bytes += b.len;
            s.keep_table = b.ptr;
            s.keep_entries = @intCast(table.len);
        }
        return s;
    }

    pub fn deinit(s: *State) void {
        for (s.own.items) |*b| b.free();
        s.own.deinit(s.gpa);
        s.gpa.free(s.layers);
        if (s.ids.len > 0) s.gpa.free(s.ids);
        s.* = undefined;
    }

    /// The page table (every entry on the scrub page), and with `own` the slot's own arena, every entry mapped.
    fn pageTable(s: *State, gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const weights.Weights, context: usize, storage: Storage) !void {
        const g = (try pagedShapeFor(w, if (storage == .own) storage.own.drafter_hk else 0)) orelse return error.NoGlobalLayers;
        const entries = pages.pagesFor(context);
        s.place = switch (storage) {
            .pool => |pl| pl,
            .own => |o| .{ .base = try s.raw(d, pages.arenaBytes(ownPages(entries, o.move), g.regions, g.run)), .pages = ownPages(entries, o.move), .run = g.run },
            .window => unreachable,
        };
        s.ids = try gpa.alloc(u32, entries);
        @memset(s.ids, s.place.?.scrub());
        s.table = try s.raw(d, entries * 4);
        if (storage == .own) {
            for (s.ids, 0..) |*v, i| v.* = @intCast(i);
            if (storage.own.shuffle) {
                var prng = std.Random.DefaultPrng.init(0x1d2b_2024);
                prng.random().shuffle(u32, s.ids);
            }
            s.mapped = entries;
            s.run = runStart(s.ids);
            if (storage.own.move) s.flip = move_flip;
            // NaN under TF_KV_POISON, so a row read before its write shows (G4); the scrub page too
            const pl = s.place.?;
            const view: cuda.DeviceBuffer = .{ .d = d, .ptr = pl.base, .len = @intCast(pages.arenaBytes(pl.pages, g.regions, g.run)), .borrowed = true };
            try view.fill8(if (pages.poisoned()) 0xff else 0, null);
        }
        try s.upload(d);
    }

    /// The host table to the device (blocking; start-up and tooling).
    fn upload(s: *const State, d: *const cuda.Driver) !void {
        const view: cuda.DeviceBuffer = .{ .d = d, .ptr = s.table, .len = s.ids.len * 4, .borrowed = true };
        try view.upload(0, std.mem.sliceAsBytes(s.ids));
    }

    /// Whether the global layers are on pages.
    pub fn paged(s: *const State) bool {
        return s.place != null;
    }

    /// Positions the slot may run to: its table's mapped pages (on pages), else its context.
    pub fn limit(s: *const State) usize {
        return if (s.place != null) @min(s.mapped * pages.page_rows, s.context) else s.context;
    }

    /// Rank 0 and the followers alike: the slot's pages are `ids` (position order); the rest of its table the scrub
    /// page. The host copy now; the device table by `enqueueTable` (on the stream, after every earlier forward).
    pub fn mapIds(s: *State, ids: []const u32) !void {
        if (s.place == null) return error.NotPaged;
        if (ids.len > s.ids.len) return error.PastContext;
        const scrub = s.place.?.scrub();
        for (ids) |id| if (id >= scrub) return error.BadPage;
        @memcpy(s.ids[0..ids.len], ids);
        @memset(s.ids[ids.len..], scrub);
        s.mapped = ids.len;
        s.run = runStart(ids);
    }

    /// No pages mapped (the slot's stream ended): no forward of it until it opens again.
    pub fn unmap(s: *State) void {
        s.mapped = 0;
        s.run = null;
        s.last_one = .none;
    }

    /// Layer `l`'s keys and values for a forward of `rows` of this stream's rows: a one-row forward of a global layer
    /// on pages that are one run (`run`) gets the ring view of the run (base its first page, a slot a position, no
    /// table read: v0.1's kernels and window); every other forward the layer's own Kv (paged for a global layer on
    /// pages). Both address the same bytes, and the ring and paged kernels give the same bits (G1).
    pub fn kvFor(s: *const State, l: u32, rows: usize) attn.Kv {
        const kv = s.layer(l).kv;
        if (rows != 1 or !kv.paged()) return kv;
        const start = s.ringRun() orelse return kv;
        const off = @as(u64, start) * kv.page * 2;
        return .{ .k = kv.k + off, .v = kv.v + off, .slots = s.context, .heads = kv.heads, .window = 0, .run = start };
    }

    /// The run a one-row forward reads through the ring kernels, or null (not on pages, not one run, or switched off;
    /// tooling: a `flip` phase on the page table).
    pub fn ringRun(s: *const State) ?u32 {
        if (!s.ring_on_run or s.mapped == 0) return null;
        if (s.flip > 0 and (s.ones / s.flip) % 2 == 1) return null;
        return s.run;
    }

    /// Tooling (TF_INK2_PAGE_ORDER=move): the slot's own pages moved to the next run of its two-context arena (a new
    /// sequence's: rows written before it are never read), the device table queued on `c`'s stream.
    fn moveRun(s: *State, c: *const Ctx) !void {
        const pl = s.place orelse return;
        const entries = s.ids.len;
        if (pl.pages != ownPages(entries, true)) return; // not a moving arena
        s.moves += 1;
        const start: u32 = @intCast((@as(usize, s.moves) * (entries / 2 + 1)) % (entries + 1));
        for (s.ids, 0..) |*v, i| v.* = start + @as(u32, @intCast(i));
        s.mapped = entries;
        s.run = start;
        s.ones = 0;
        s.last_one = .none;
        try s.enqueueTable(c.d, c.stream);
    }

    /// The host table queued to the device on `stream` (a pinned copy: it lands after the stream's earlier work).
    pub fn enqueueTable(s: *const State, d: *const cuda.Driver, stream: cuda.Stream) !void {
        try cuda.copy_source.enqueue(d, s.table, std.mem.sliceAsBytes(s.ids), stream);
    }

    fn raw(s: *State, d: *const cuda.Driver, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        try s.own.append(s.gpa, b);
        s.bytes += bytes;
        return b.ptr;
    }

    fn zeroed(s: *State, d: *const cuda.Driver, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        try b.fill8(0, null);
        try s.own.append(s.gpa, b);
        s.bytes += bytes;
        return b.ptr;
    }

    pub fn layer(s: *const State, l: u32) *const LayerState {
        return &s.layers[l - s.first];
    }

    /// A new sequence: every convolution's state back to zeros (the keys and values need no clearing: a forward
    /// reads only positions it or an earlier forward of the same sequence wrote).
    pub fn restart(s: *State, c: *const Ctx) !void {
        if (s.flip > 0) try s.moveRun(c);
        for (s.layers) |st| {
            const kvc = st.kv.heads * D * 2;
            try c.zero(st.conv_k, conv.state_rows * kvc);
            try c.zero(st.conv_v, conv.state_rows * kvc);
            try c.zero(st.conv_attn, conv.state_rows * H * 2);
            try c.zero(st.conv_mlp, conv.state_rows * H * 2);
        }
        s.pos = 0;
    }
};

/// Tooling, TF_INK2_PAGE_ORDER=move: one-row forwards a phase (ring on the run, then the page table, in turn).
pub const move_flip: u32 = 5;

/// The pages of a slot's own arena (the scrub page last): its context's, or two contexts' when it moves.
fn ownPages(entries: usize, move: bool) usize {
    return if (move) 2 * entries + 1 else entries + 1;
}

/// The first id when `ids` are one run of consecutive ids, else null.
pub fn runStart(ids: []const u32) ?u32 {
    if (ids.len == 0 or pages.Alloc.runsOf(ids) != 1) return null;
    return ids[0];
}

/// The device bytes `State.init` takes for these weights, `context` positions and forwards of `max_rows` rows.
pub fn stateBytes(w: *const weights.Weights, context: usize, max_rows: usize) usize {
    var n: usize = 0;
    for (w.layers) |l| {
        const cols: usize = @as(usize, l.hk) * D;
        const slots = if (l.global) context else @min(context, attn.window - 1 + max_rows);
        n += 2 * slots * cols * 2 + 2 * conv.state_rows * cols * 2 + 2 * conv.state_rows * H * 2;
    }
    return n;
}

/// The pages' shape on these weights: regions (2 a global layer: keys, values) and a run's bytes (page_rows rows of
/// a global layer); null without a global layer. Every global layer has the same key heads.
/// `drafter`: the drafter's global layer has the last two regions (2g: keys, 2g + 1: values; stage 2c).
pub const PagedShape = struct {
    regions: usize,
    run: u64,
    globals: usize,
    drafter: bool = false,

    /// The drafter's keys' (0) and values' (1) regions.
    pub fn drafterRegion(g: PagedShape, which: usize) usize {
        std.debug.assert(g.drafter and which < 2);
        return 2 * g.globals + which;
    }
};

pub fn pagedShape(w: *const weights.Weights) ?PagedShape {
    var n: usize = 0;
    var hk: usize = 0;
    for (w.layers) |l| if (l.global) {
        n += 1;
        hk = l.hk;
    };
    if (n == 0) return null;
    return .{ .regions = 2 * n, .run = pages.page_rows * hk * D * 2, .globals = n };
}

/// `pagedShape` with the drafter's global layer in two more regions when `drafter_hk` > 0 (its key heads: a page's
/// run of it must be a target global layer's, as it is at every W: 8 / W both).
pub fn pagedShapeFor(w: *const weights.Weights, drafter_hk: usize) !?PagedShape {
    var g = pagedShape(w) orelse return null;
    if (drafter_hk == 0) return g;
    if (pages.page_rows * drafter_hk * D * 2 != g.run) return error.DraftPageShape;
    g.regions += 2;
    g.drafter = true;
    return g;
}

/// Bytes a page takes on this rank (every region's run of it).
pub fn pageBytes(w: *const weights.Weights) u64 {
    const g = pagedShape(w) orelse return 0;
    return g.regions * g.run;
}

/// `pageBytes` with the drafter's two runs when `drafter_hk` > 0.
pub fn pageBytesFor(w: *const weights.Weights, drafter_hk: usize) u64 {
    const g = (pagedShapeFor(w, drafter_hk) catch return 0) orelse return 0;
    return g.regions * g.run;
}

/// The device bytes a slot on pages takes beside its pages: every layer's but the global layers' keys and values,
/// and its page table.
pub fn fixedBytes(w: *const weights.Weights, context: usize, max_rows: usize) usize {
    var n: usize = 0;
    for (w.layers) |l| {
        const cols: usize = @as(usize, l.hk) * D;
        const slots = if (l.global) 0 else @min(context, attn.window - 1 + max_rows);
        n += 2 * slots * cols * 2 + 2 * conv.state_rows * cols * 2 + 2 * conv.state_rows * H * 2;
    }
    return n + pages.pagesFor(context) * 4;
}

/// Positions [lo, hi) of a global layer's keys and values as runs of contiguous rows, in position order: one run a
/// window, or a run a page (consecutive page ids merged) on pages.
pub const RowRuns = struct {
    kv: attn.Kv,
    ids: []const u32,
    p: usize,
    hi: usize,
    row: usize,

    pub const Run = struct { pos: usize, n: usize, k: u64, v: u64 };

    pub fn of(kv: attn.Kv, ids: []const u32, lo: usize, hi: usize) RowRuns {
        return .{ .kv = kv, .ids = ids, .p = lo, .hi = hi, .row = kv.heads * D * 2 };
    }

    pub fn next(r: *RowRuns) !?Run {
        if (r.p >= r.hi) return null;
        const p = r.p;
        if (!r.kv.paged()) {
            if (r.kv.window != 0 or r.hi > r.kv.slots) return error.KvWindow; // a slot a position only
            r.p = r.hi;
            return .{ .pos = p, .n = r.hi - p, .k = r.kv.k + p * r.row, .v = r.kv.v + p * r.row };
        }
        const stride: u64 = @as(u64, r.kv.page) * 2;
        var j = p / pages.page_rows;
        if (j >= r.ids.len) return error.PastContext;
        const off = p % pages.page_rows;
        const id = r.ids[j];
        var end = @min(r.hi, (j + 1) * pages.page_rows);
        // consecutive ids are one run when a page's rows are a whole run of the region (the arena's layout)
        if (stride == pages.page_rows * r.row) while (end < r.hi and j + 1 < r.ids.len and r.ids[j + 1] == r.ids[j] + 1) {
            j += 1;
            end = @min(r.hi, (j + 1) * pages.page_rows);
        };
        r.p = end;
        const at = @as(u64, id) * stride + off * r.row;
        return .{ .pos = p, .n = end - p, .k = r.kv.k + at, .v = r.kv.v + at };
    }
};

/// The device bytes the convolutions' input history adds (State.init with `spec`).
pub fn histBytes(w: *const weights.Weights) usize {
    var n: usize = 0;
    for (w.layers) |l| n += (conv.state_rows + spec_rows) * (2 * @as(usize, l.hk) * D + 2 * H) * 2;
    return n;
}

/// A convolution's state after a verify window's input rows x [R, C] (R <= spec_rows), keeping its history:
/// hist = [state; x], then state = hist[R .. R + 3), one launch (inkling2_conv.cu tf_ink2_conv_commit_spec: the same
/// bytes as `convCommitSpecCopies`' three copies).
pub fn convCommitSpec(c: *const Ctx, state: u64, hist: u64, x: Mat) !void {
    const R = x.rows;
    if (R > spec_rows or x.dt != .bf16) return error.BadSpecWindow;
    if (R == 0) return;
    const C = x.cols;
    var a: cuda.Args = .{};
    a.add(state);
    a.add(hist);
    a.add(x.ptr);
    a.add(@as(i64, @intCast(x.ld)));
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(C)));
    try cuda.launch.launch(c.k.conv.commit_spec_fn, .{ .grid = .{ .x = @intCast((C + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, c.stream, &a);
}

/// `convCommitSpec` as three device copies (the form it replaced; the op check compares the two).
pub fn convCommitSpecCopies(c: *const Ctx, state: u64, hist: u64, x: Mat) !void {
    const n = conv.state_rows;
    const row = x.cols * x.dt.size();
    const R = x.rows;
    if (R > spec_rows or !x.dense()) return error.BadSpecWindow;
    try c.copy(hist, state, n * row);
    try c.copy(hist + n * row, x.ptr, R * row);
    try c.copy(state, hist + R * row, n * row);
}

/// One convolution's place in `State.keep_table` (inkling2_conv.cu KeepEntry).
pub const KeepEntry = extern struct {
    state: u64,
    hist: u64,
    cols: u32,
    pad: u32 = 0,
};

/// Every convolution of the slot as if only its last window's first `keep` rows had run: state = hist[keep ..
/// keep + 3) for each, one launch over `keep_table` (inkling2_conv.cu tf_ink2_conv_keep: the same bytes as a
/// `convKeep` copy a convolution).
pub fn keepAll(c: *const Ctx, st: *const State, keep: usize) !void {
    if (st.keep_table == 0) return error.RollbackNotKept;
    if (keep == 0 or keep > spec_rows) return error.BadKeep;
    var a: cuda.Args = .{};
    a.add(st.keep_table);
    a.add(@as(i32, @intCast(keep)));
    try cuda.launch.launch(c.k.conv.keep_fn, .{ .grid = .{ .x = 8, .y = st.keep_entries, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, c.stream, &a);
}

/// The state as if only the window's first `keep` rows had run: hist[keep .. keep + 3) (one convolution, one copy;
/// `keepAll` does every convolution in one launch, and the op check compares the two).
pub fn convKeep(c: *const Ctx, state: u64, hist: u64, cols: usize, keep: usize) !void {
    const row = cols * 2;
    try c.copy(state, hist + keep * row, conv.state_rows * row);
}

/// k', v' rows [R, hk*128] of positions pos0 .. pos0 + R - 1 into their slots, one launch
/// (inkling2_attn.cu tf_ink2_attn_kv_write); `dev`: pos0 read on the device at run time (a captured decode step).
/// A paged Kv (a global layer in the KV page pool) takes tf_ink2_attn_kv_write_pg: each row to its page's row.
pub fn kvWrite(c: *const Ctx, kv: attn.Kv, pos0: usize, dev: ?attn.Dev, k: Mat, v: Mat) !void {
    const R = k.rows;
    if (R > kv.slots or !k.dense() or !v.dense()) return error.KvWindow;
    if (R == 0) return;
    const row = kv.heads * D * 2;
    if (k.cols * 2 != row or v.cols * 2 != row or row % 16 != 0 or kv.slots > std.math.maxInt(i32) or pos0 + R > std.math.maxInt(i32)) return error.KvWindow;
    // paged: every row inside the table, whole 16-byte words a page
    if (kv.paged() and (kv.window != 0 or kv.page % 8 != 0 or pos0 + R > kv.slots)) return error.KvWindow;
    var a: cuda.Args = .{};
    for ([_]u64{ k.ptr, v.ptr, kv.k, kv.v }) |p| a.add(p);
    a.add(@as(i32, @intCast(row / 16)));
    a.add(@as(i32, @intCast(kv.slots)));
    a.add(@as(i32, @intCast(pos0)));
    a.add(if (dev) |x| x.posp else @as(u64, 0));
    if (kv.paged()) kv.addPaged(&a);
    const f = if (kv.paged()) c.k.attn.pg.kv_write else c.k.attn.kv_write;
    try c.launchOn(kv.run, kv.k, kv.v, f, .{ .grid = .{ .x = @intCast(R), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, &a);
}

/// The cached rows of positions lo .. hi - 1 gathered into k, v [hi - lo, hk*128] (for dumps); `ids` a paged
/// layer's host table.
pub fn kvGather(c: *const Ctx, kv: attn.Kv, ids: []const u32, lo: usize, hi: usize, k: Mat, v: Mat) !void {
    if (kv.paged()) {
        var runs = RowRuns.of(kv, ids, lo, hi);
        while (try runs.next()) |r| {
            try c.copy(k.row(r.pos - lo), r.k, r.n * runs.row);
            try c.copy(v.row(r.pos - lo), r.v, r.n * runs.row);
        }
        return;
    }
    const row = kv.heads * D * 2;
    var p = lo;
    while (p < hi) {
        const slot = p % kv.slots;
        const n = @min(hi - p, kv.slots - slot);
        try c.copy(k.row(p - lo), kv.k + slot * row, n * row);
        try c.copy(v.row(p - lo), kv.v + slot * row, n * row);
        p += n;
    }
}

/// A convolution's state after a forward's input rows x [R, C] (bf16): the last 3 rows of [state; x], one launch
/// (inkling2_conv.cu tf_ink2_conv_commit).
pub fn convCommit(c: *const Ctx, state: u64, x: Mat) !void {
    comptime std.debug.assert(conv.state_rows == 3);
    if (x.rows == 0) return;
    if (x.dt != .bf16) return error.BadConvShape;
    // only the last three rows can stay
    const xs = if (x.rows > conv.state_rows) x.slice(x.rows - conv.state_rows, conv.state_rows) else x;
    const C = x.cols;
    var a: cuda.Args = .{};
    a.add(state);
    a.add(xs.ptr);
    a.add(@as(i64, @intCast(xs.ld)));
    a.add(@as(i32, @intCast(xs.rows)));
    a.add(@as(i32, @intCast(C)));
    try cuda.launch.launch(c.k.conv.commit_fn, .{ .grid = .{ .x = @intCast((C + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, c.stream, &a);
}

/// h_in [R, H] bf16 as the next completed block.
pub fn pushBlock(c: *const Ctx, a: *Act, h_in: Mat) !void {
    if (a.nblocks >= max_blocks) return error.TooManyBlocks;
    try c.copyRows(a.block(a.nblocks, h_in.rows), h_in);
    a.nblocks += 1;
}

test "inkling2 cuda state: ring slots of a sliding layer cover every row's window" {
    // a forward of R rows at pos0 writes slots of pos0 .. pos0+R-1; row pos0 still needs pos0 - 1023
    const R = 300;
    const slots = attn.window - 1 + R;
    const pos0 = 5000;
    // the oldest position still needed is pos0 - 1023; the newest written is pos0 + R - 1
    try std.testing.expect((pos0 + R - 1) - (pos0 - (attn.window - 1)) < slots);
}

test "inkling2 cuda state: a slot's pages are one run only when every id follows the one before" {
    try std.testing.expectEqual(@as(?u32, 7), runStart(&.{ 7, 8, 9 }));
    try std.testing.expectEqual(@as(?u32, 0), runStart(&.{0}));
    try std.testing.expectEqual(@as(?u32, null), runStart(&.{ 7, 8, 10 }));
    try std.testing.expectEqual(@as(?u32, null), runStart(&.{ 8, 7 }));
    try std.testing.expectEqual(@as(?u32, null), runStart(&.{}));
}
