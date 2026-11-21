//! Inkling-2-Small behind the native CUDA registry (native/cuda.zig), `full` mode: `tensorfold-native serve MODEL
//! --backend cuda [--tp W --rank R]`. Rank 0 serves HTTP; ranks 1 .. W-1 (the same command with --rank R) load
//! their share, then run rank 0's forwards until it ends (cuda_follow.zig) and exit. The ranks meet over NCCL at
//! NCCL_COMM_ID (rank 0's address:port; from --master/--master-port when unset).
//! TF_INK2_LAYERS=N serves only layers [0, N) (debug; the same N on every rank).
//! TF_INK2_MODE and TF_INK2_DIGESTS (cuda_engine.zig): the experts' mode and the content-digest manifest, the same
//! on every rank. TF_INK2_DSPARK=1 loads the checkpoint's DSpark drafter split over the ranks (the same on every rank;
//! TF_INK2_DSPARK_FOLD names its head folds).
//! Streams: --context sets how many run at once (`count`). After every rank's load, rank 0 gathers each rank's room
//! (the server's reading after the reserve; the weights, the drafter and the op scratch are in already) and its bytes
//! a stream at --context (the drafter's caches counted when it is loaded); `--parallel auto` (the server's default)
//! serves as many streams as fit on every rank beside the prompt cache (--prompt-cache-gib, one arena a rank) and the
//! graph cache's reserve for that many streams (`graphBytes`), at most TF_INK2_MAX_STREAMS (rank 0's; 1 .. 8, default
//! 8: the most the exact gates ran at once; the recipe sets 4). None fits: the server refuses to start and names the
//! largest context that does. `--parallel N` serves N (the cap does not apply), refused when they do not fit. Every rank makes the slots' states after the count (`reserve`, from rank 0's number).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");
const ink_lanes = @import("cuda_lanes.zig");
const image_offer = @import("image_offer.zig");

const engine = ic.engine;

pub const model_type = ink.config.model_type;
pub const formats: []const []const u8 = &.{"inkling2"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
pub const explains_open = true;
/// Positions a slot holds past --context: a request of --context tokens in all (prompt plus max_tokens) fits.
const context_slack = 16;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: Inkling-2 on CUDA serves plain decode so far (its DSpark head is not built yet)";

/// The server's reading of this rank's room after its load; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    tp: u32 = 1,
    rank: u32 = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29551,
    /// What --parallel and --prompt-cache-gib ask (native/cuda_options.zig): `count`'s inputs.
    streams_asked: Asked = .{},
    /// --vision: image input's limits (native/cuda_options.zig); null without --vision, and nothing about images loads.
    vision: ?image_offer.Limits = null,
};

/// --vision's image tokens a request when --vision-image-tokens is not given: one 4096 x 4096 image (owner decision D2).
pub const default_image_tokens: u32 = image_offer.Limits.default_image_tokens;

/// --parallel's count (`fixed`: a number, else auto up to `n`) and --prompt-cache-gib (null: not given).
pub const Asked = struct { n: u32 = most_streams, fixed: bool = false, cache_gib: ?f64 = null };

/// Streams `--parallel auto` serves at most: the most the exact gates have run at once (and the step header's
/// segments, a batched forward's streams).
pub const most_streams: u32 = @min(8, engine.max_segs);

/// TF_INK2_MAX_STREAMS (rank 0): the cap on `--parallel auto`'s count, 1 .. most_streams; unset: most_streams.
fn maxStreams() !u32 {
    const v = std.mem.span(std.c.getenv("TF_INK2_MAX_STREAMS") orelse return most_streams);
    return parseMaxStreams(v) orelse error.BadMaxStreams;
}

fn parseMaxStreams(v: []const u8) ?u32 {
    const n = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return null;
    return if (n >= 1 and n <= most_streams) n else null;
}

/// The smallest room of ranks 1 .. W-1, in rank 0's stream bytes, and its rank.
pub const Peer = struct { bytes: u64, rank: usize };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    /// No stream's state exists before `reserve` makes the admitted ones on every rank.
    free_streams: u32 = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// The server's admitted streams: their states on every rank, made once.
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    /// Rank 0: the server admits streams by this when it is smaller than rank 0's own room.
    peer_room: ?Peer = null,
    /// The checkpoint's own draft head when loaded (TF_INK2_DSPARK=1): "DSpark".
    draft_kind: ?[]const u8 = null,
    /// The prompt cache keeps this rank group's states (snap_*).
    target: *anyopaque,
    /// Rank 0: the streams `count` found at --context (the server serves exactly these; null: no room reading).
    streams: ?u32 = null,
    /// On the KV page pool: the lane host admits a request only when its pages fit (`fits`).
    admits: bool = false,
    /// Rank 0 with --vision: the image offer (native/cuda_vision_offer.zig), alive until `close`.
    vision: ?*image_offer.Vision = null,
};

/// --context's ceiling: the checkpoint's model_max_length (1,048,576), so --context 0 serves it and a larger one is
/// refused by name (the text config names no max_position_embeddings, which the server reads otherwise). An
/// unreadable config names none here (0): the family's open then refuses it with its own reason.
pub fn contextCeiling(a: std.mem.Allocator, io: std.Io, dir: []const u8, rope_yarn: ?f64) !i64 {
    _ = rope_yarn; // no rope in the target's layers
    const c = ink.config.Config.read(a, io, dir, null) catch return 0;
    return c.text.max_positions;
}

/// Rank 0 with --vision (native/cuda_vision_offer.zig weightBytes): the vision tower in engine mode and its image
/// row store at the default image tokens, beside the text weights (about 803 MiB). A larger --vision-image-tokens
/// store is in the room each rank reads after its load.
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    _ = a;
    _ = io;
    _ = dir;
    return ic.vision.rank0Bytes(default_image_tokens);
}

/// --vision's limits as the engine takes them (every rank; only rank 0 loads anything).
fn engineVision(o: Options) ?engine.Vision {
    const l = o.vision orelse return null;
    return .{ .image_tokens = l.image_tokens, .max_images = l.max_images };
}

/// Every rank, before any GPU work: --vision-max-images within the spans an image chunk splices, the refusal's
/// words (both numbers) in `refusal`.
fn checkVision(o: Options) !void {
    const v = engineVision(o) orelse return;
    ic.vision.checkLimits(v) catch |err| {
        if (err == error.VisionMaxImages) refusal = std.fmt.bufPrint(&refusal_buf, "--vision-max-images {d}: an image chunk splices at most {d} images (give 1 to {d})", .{ v.max_images, ic.vision.max_images, ic.vision.max_images }) catch "--vision-max-images: an image chunk splices at most 16 images";
        return err;
    };
}

/// This rank's weights: its heads and experts of the served layers, rank 0's embed and head.
pub fn placementWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, tp: u32, rank: u32, drafts: bool) !u64 {
    _ = drafts;
    var m = ink.describe.Model.fromDir(a, io, dir, .{ .world = @max(tp, 1) }, null) catch return 0;
    defer m.deinit();
    if (rank >= m.p.ranks.len) return 0;
    return engine.rankWeightBytes(&m, rank, engine.layersFromEnv() catch 0);
}

const Owned = struct {
    gpa: std.mem.Allocator,
    group: ?cuda.tp_group.Group = null,
    e: *engine.Engine,
    lanes: ink_lanes.Lanes,
    /// TF_INK2_MAX_STREAMS as `count` read it (0: unreadable), for /health.
    max_streams: u32 = 0,
    /// Rank 0 with --vision: the processor the server calls on request threads (image_offer.zig).
    vision: ?image_offer.Vision = null,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    const world: u32 = @max(o.tp, 1);
    if (o.rank >= world) return error.NoRank;
    if (world > ink.plan.max_world) return error.NoRank;
    if (world == 1) try checkVision(o);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = undefined, .lanes = undefined, .max_streams = maxStreams() catch 0 };
    if (world > 1) {
        try commId(gpa, o);
        // the ranks' liveness lines (NCCL_COMM_ID's port + 1): a rank that fails or dies stops the others
        try cuda.tp_watch.start(io, o.rank, world);
        // --vision-max-images past the splice: every rank refuses before the group opens, the others hear the reason
        checkVision(o) catch |err| {
            cuda.tp_watch.fail(@errorName(err));
            return err;
        };
        own.group = cuda.tp_group.Group.open(ctx.d, o.rank, world) catch |err| {
            cuda.tp_watch.fail(@errorName(err)); // the others stop now instead of waiting in NCCL's init
            return err;
        };
        // one vote of every rank on the transport settings, then the transports (cuda_wire.zig openAll)
        ic.wire.openAll(gpa, io, ctx.d, &own.group.?) catch |err| {
            own.group.?.abort(); // nothing may wait on a peer that stopped; the process ends next
            return err;
        };
    }
    errdefer if (own.group) |*g| g.close();
    // A/B tooling, one experiment at a time: rank 0's refusal stops the other ranks with its reason
    if (o.rank == 0) ink_lanes.abSwitches() catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
    own.e = engine.Engine.init(gpa, io, ctx.d, dir, .{ .rank = o.rank, .world = world, .context = o.context + context_slack, .group = if (own.group) |*g| g else null, .scratch_gib = scratchGib(), .vision = engineVision(o) }) catch |err| {
        if (err == error.ScratchShort and ic.scratch.tower_short.need > 0) refusal = std.fmt.bufPrint(&refusal_buf, "--vision: a vision tower unit borrows {d:.1} MiB of rank 0's op scratch, which holds {d} MiB: raise TF_INK2_SCRATCH_GIB on every rank", .{ @as(f64, @floatFromInt(ic.scratch.tower_short.need)) / (1 << 20), ic.scratch.tower_short.have >> 20 }) catch "";
        cuda.tp_watch.fail(@errorName(err)); // the other ranks stop with this rank's reason (a no-op at W = 1)
        return err;
    };
    errdefer own.e.deinit(); // Engine.init checked every forward's op scratch at this context (cuda_scratch.zig)
    // rank 0's refusal (no stream fits, or --parallel N does not) stops the other ranks with its reason
    const counted = (if (own.group) |*g| rooms(own.e, g, o) else alone(own.e, o)) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
    own.e.asleep = world > 1 and cuda.tp_watch.open(); // the followers start asleep
    if (o.rank == 0 and !own.e.paged) std.log.warn("inkling2: TF_INK2_PAGES=0 serves v0.1's fixed windows, a tooling setting for A/B measurements (the KV page pool is the serving layout; /health reports no kv)", .{});
    if (o.rank != 0) {
        std.log.info("inkling2 rank {d} of {d}: following rank 0", .{ o.rank, world });
        ic.follow.follow(own.e) catch |err| {
            std.log.err("inkling2 rank {d}: following rank 0 stopped ({s})", .{ o.rank, @errorName(err) });
            cuda.tp_watch.fail(@errorName(err));
            std.c._exit(1);
        };
        release(own);
        std.process.exit(0);
    }
    own.lanes = .{ .x = .{ .e = own.e }, .gpa = gpa };
    // request threads call prepare at once: decode samples in malloc (thread-safe), freed before prepare returns
    if (o.vision) |l| own.vision = image_offer.Vision.init(io, std.heap.c_allocator, own.e.model.c.image_token, l);
    own.lanes.loadScript(io) catch |err| {
        std.log.err("inkling2: TF_INK2_SCRIPT: {s}", .{@errorName(err)});
        return err;
    };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = own.lanes.widest(), // the widest verify window (the lane config's drafts; 2 without the drafter, so the lane core may share rounds)
        .stream_bytes = own.e.streamBytes(),
        .ctx = own,
        .deinit = close,
        .reserve = reserve,
        .peer_room = if (counted) |c| c.peer else null,
        .draft_kind = if (own.e.drafter != null) "DSpark" else null,
        .target = own,
        .streams = if (counted) |c| c.streams else null,
        .admits = own.e.paged,
        .vision = if (own.vision) |*v| v else null,
    };
}

/// /health's `kv` (B3; native/cuda_startup_phase.zig converts it): rank 0's KV page pool as its engine thread last
/// copied it (cuda_pages.zig `show`), the streams and their cap. Null without a pool (TF_INK2_PAGES=0, tooling).
pub const KvCounts = struct { page_tokens: u32, pages: u64, free: u64, available: u64, kept: u64, request_tokens: u64, streams: u32, max_streams: u32, waiting_pages: u64, page_waits: u64 };

pub fn kv_source(ptr: *anyopaque) ?KvCounts {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const p = own.e.pool orelse return null;
    const s = p.shownNow();
    return .{
        .page_tokens = @intCast(ic.pages.page_rows),
        .pages = s.pages,
        .free = s.free,
        .available = s.available,
        .kept = s.kept,
        .request_tokens = @min(s.pages * ic.pages.page_rows, o_context(own.e)),
        .streams = p.slots,
        .max_streams = own.max_streams,
        .waiting_pages = s.waiting,
        .page_waits = s.waits,
    };
}

/// The lane host's admission on the KV page pool (Loaded.admits): a slot free and the request's pages available.
pub fn fits(ptr: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return own.lanes.fits(prompt, max_tokens);
}

// The prompt cache on the KV page pool (stage 2d; cuda_snap.zig). A state at N holds rows [0, N) (no head lookahead):
// its whole pages by reference, its fixed part and tail rows in one arena slot; it is kept where the lane cut a chunk,
// at any row: no grid, so the store is not "planned". The fork rule keeps one where a prompt parts from the longest
// kept one; TF_INK2_PROMPT_TAIL=1 keeps a raw prompt's last row too. The store's budget is the arena's slots; the pages
// are the pool's, given back by the store when a stream's pages do not fit (`snap_store`: the reclaim chain).
pub const snap_lookahead: u32 = 0;
pub const snap_fork = true;
pub const snap_tail = "TF_INK2_PROMPT_TAIL";
/// A new shared system prompt's end is planned with the endpoint and may evict, so concurrent requests on it wait for
/// one prefill and resume there even when the arena is full (stage 1 on the Sparks: DSpark's larger states left no
/// free room, the cuts were trimmed and 0 of 3 waited).
pub const snap_shared_first = true;

/// The arena bytes a state at `at` takes (the same on every rank): one slot.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    return ic.snap.bytes(own.e, at);
}

/// The store's budget becomes one arena on every rank, made now (nothing is allocated while serving); the bytes past
/// its last granule are set aside.
pub fn snap_ready(ptr: *anyopaque, budget: u64) anyerror!u64 {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const n = try ic.snap.arena(own.lanes.x, budget);
    // the NVMe tier under it (TF_INK2_NVME): every rank's directory read, rank 0's index made
    if (n > 0) try ic.nvme.start(own.lanes.x);
    if (n > 0) std.log.info("inkling2: prompt cache arena {d:.2} GiB on every rank: {d} kept states of {d:.0} MiB (their pages the pool's)", .{ gib(n), n / ic.snap.bytes(own.e, 0), @as(f64, @floatFromInt(ic.snap.bytes(own.e, 0))) / (1 << 20) });
    return budget - n;
}

/// The server's store (native/cuda.zig, after it is made): the lanes ask it to give kept states' pages back before a
/// stream's pages do not fit (the reclaim chain), and whether a state is borrowed (a resume in flight).
pub fn snap_store(ptr: *anyopaque, store: *anyopaque, give_back: *const fn (*anyopaque, ?*anyopaque, *anyopaque, *const fn (*anyopaque) bool, bool) bool, borrowed: *const fn (*anyopaque, *anyopaque) bool) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.lanes.store = .{ .ctx = store, .give_back = give_back, .borrowed = borrowed };
}

/// The prompt pass of `owner` (a lane stream) stands at `at`, a mark at a chunk start: every rank copies its slot
/// into the arena at one offset. Rank 0 names the prefix's images in the record (Kept.images; zeros for text).
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.UnknownStream));
    const slot = own.lanes.slotOf(s) orelse return error.UnknownStream;
    const images = try ink_lanes.imageKey(s.images, at); // zeros without images (no hashing)
    const k = try own.gpa.create(ic.snap.Kept);
    errdefer own.gpa.destroy(k);
    try ic.snap.save(own.lanes.x, slot, at, k);
    if (s.images.len > 0) k.images = images;
    ic.nvme.wanted(own.e, k, s.prompt(), s.images.len); // written behind to the NVMe tier, when on
    return k;
}

/// A restore outside a prompt pass would leave the other ranks behind: states resume inside it (Lanes.resumeFrom).
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = .{ ptr, owner, saved };
    return error.RestoredInPromptPass;
}

/// Rank 0 frees the state's extent; nothing goes to the other ranks.
pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const k: *ic.snap.Kept = @ptrCast(@alignCast(saved));
    ic.nvme.dropping(own.lanes.x, k); // not written now (a write of it under way is given up)
    ic.snap.drop(own.lanes.x, k);
    own.gpa.destroy(k);
}

/// A rank's room after its load (the server's reading, after the reserve) and its bytes a stream at --context; on
/// pages, a slot's fixed part and a page's bytes (`stream` is then the fixed part and a whole context's pages).
pub const RankRoom = extern struct { room: u64, stream: u64, fixed: u64 = 0, page: u64 = 0 };

/// What `count` found (rank 0): the streams, and the smallest room less the graph cache's reserve, in rank 0's
/// stream bytes, for the server's admission (which then admits exactly `streams` and gives the prompt cache the rest).
const Counted = struct { streams: u32, peer: Peer };

/// On pages: the most pool pages every rank holds beside `n` slots' fixed parts, the caches for n streams and the
/// scrub page (0: none; a rank without a reading holds any number).
pub fn poolPages(all: []const RankRoom, cache: u64, n: u32) u64 {
    var most: u64 = std.math.maxInt(u64);
    for (all) |r| {
        if (r.room == std.math.maxInt(u64) or r.page == 0) continue;
        const rest = r.room -| (cache + graphBytes(n) + r.page + @as(u64, n) * r.fixed);
        most = @min(most, rest / r.page);
    }
    return if (most == std.math.maxInt(u64)) 0 else most;
}

/// The largest page's bytes over the ranks (the scrub page every rank keeps beside its pool; 0: not on pages).
fn scrubBytes(all: []const RankRoom) u64 {
    var b: u64 = 0;
    for (all) |r| b = @max(b, r.page);
    return b;
}

/// Device bytes kept out of the streams' room for the captured forwards of `streams` slots (cuda_engine.zig
/// graphUnits units of graph_budget_bytes, on every rank).
pub fn graphBytes(streams: u32) u64 {
    return @as(u64, engine.graphUnits(streams)) * engine.graph_budget_bytes;
}

/// The prompt cache's arena a rank: --prompt-cache-gib (not given: none counted; the server sizes the cache from
/// what the streams leave).
fn cacheBytes(asked: Asked) u64 {
    const g = asked.cache_gib orelse return 0;
    return if (g > 0) @intFromFloat(g * (1 << 30)) else 0;
}

/// Room left on rank r for `n` streams: its room less the prompt cache and the graph cache for n (0: short).
fn left(r: RankRoom, cache: u64, n: u32) u64 {
    return r.room -| (cache + graphBytes(n));
}

/// Whether `n` streams fit on every rank (a rank without a reading, room maxInt, fits anything).
pub fn fitsAll(all: []const RankRoom, cache: u64, n: u32) bool {
    for (all) |r| {
        if (r.room == std.math.maxInt(u64)) continue;
        if (r.room < cache + graphBytes(n) or left(r, cache, n) / @max(r.stream, 1) < n) return false;
    }
    return true;
}

/// The most streams (1 .. `most`) that fit on every rank, 0 if not even one does.
pub fn fitStreams(all: []const RankRoom, cache: u64, most: u32) u32 {
    var n = most;
    while (n > 0) : (n -= 1) if (fitsAll(all, cache, n)) return n;
    return 0;
}

/// The rank whose room holds the fewest of its streams beside the caches for `n` streams.
fn binding(all: []const RankRoom, cache: u64, n: u32) usize {
    var best: usize = 0;
    var most: u128 = std.math.maxInt(u128);
    for (all, 0..) |r, q| {
        if (r.room == std.math.maxInt(u64)) continue;
        const k = @as(u128, left(r, cache, n)) * (1 << 20) / @max(r.stream, 1);
        if (k < most) {
            most = k;
            best = q;
        }
    }
    return best;
}

/// The largest --context (a multiple of 1,024 tokens, at most `ceiling`) whose one stream fits on every rank beside
/// the caches; 0 if none. Rank r's stream at a context scales as rank 0's (`model.at(positions)`: rank 0's bytes for
/// a slot of that many positions; `stream0` its bytes at this start's --context).
pub fn largestContext(all: []const RankRoom, cache: u64, stream0: u64, model: anytype, ceiling: usize) usize {
    var lo: usize = 0;
    var hi: usize = ceiling / 1024;
    while (lo < hi) {
        const mid = (lo + hi + 1) / 2;
        const b0 = model.at(mid * 1024 + context_slack);
        var ok = true;
        for (all) |r| {
            if (r.room == std.math.maxInt(u64)) continue;
            const b = @as(u128, b0) * r.stream / @max(stream0, 1);
            if (r.room < cache + graphBytes(1) or b > left(r, cache, 1)) ok = false;
        }
        if (ok) lo = mid else hi = mid - 1;
    }
    return lo * 1024;
}

/// The refusal `explain` gives for `count`'s errors (rank 0 writes it before failing).
var refusal_buf: [768]u8 = undefined;
var refusal: []const u8 = "";

/// Rank 0's bytes a slot of `positions` positions, for `largestContext` and `logTable`.
const At = struct {
    e: *const engine.Engine,
    fn at(x: @This(), positions: usize) usize {
        return x.e.streamBytesAt(positions);
    }
};

/// Rank 0: the streams at --context from every rank's room (see the top). Logs the count; refuses when none fits
/// (auto) or --parallel N does not.
fn count(e: *engine.Engine, all: []const RankRoom, o: Options) !Counted {
    const a = o.streams_asked;
    // on pages the scrub page sits beside the pool on every rank: counted with the prompt cache
    const cache = cacheBytes(a) + scrubBytes(all);
    const cap = maxStreams() catch |err| {
        refusal = "TF_INK2_MAX_STREAMS (the recipe's MAX_STREAMS): a stream count from 1 to 8";
        std.log.err("inkling2: {s}", .{refusal});
        return err;
    };
    // auto: as many as fit, at most the cap; --parallel N may ask for more (each is checked against the rooms)
    const fit = fitStreams(all, cache, if (a.fixed) @max(a.n, most_streams) else most_streams);
    const window = o.context;
    const stream = e.streamBytes();
    // on pages, --parallel N past the whole windows that fit: N slots share the pages the room holds beside them, and
    // a request waits until its pages are free (at least a page a slot)
    if (e.paged and a.fixed and a.n > fit) {
        const p = poolPages(all, cache + pool_margin, a.n);
        if (p >= a.n) return planned(e, all, cache, a.n, p, "--parallel", fit);
    }
    if (a.fixed and a.n > fit and fit > 0) {
        const q = binding(all, cache, a.n);
        refusal = std.fmt.bufPrint(&refusal_buf, "--parallel {d} at --context {d} needs {d:.2} GiB a rank ({d} streams of {d:.2} GiB, the prompt cache's {d:.2} GiB, the graph cache's {d:.2} GiB), and rank {d} has {d:.2} GiB after its load and the reserve: {d} fit (--parallel auto serves as many as fit), or lower --context", .{ a.n, window, gib(@as(u64, a.n) * all[q].stream + cache + graphBytes(a.n)), a.n, gib(all[q].stream), gib(cache), gib(graphBytes(a.n)), q, gib(all[q].room), fit }) catch "--parallel asks for more streams than fit at --context";
        std.log.err("inkling2: {s}", .{refusal});
        return error.ParallelDoesNotFit;
    }
    if (fit == 0) {
        const q = binding(all, cache, 1);
        const most = largestContext(all, cache, stream, At{ .e = e }, window);
        refusal = std.fmt.bufPrint(&refusal_buf, "--context {d} fits no stream: one takes {d:.2} GiB a rank{s}, and rank {d} has {d:.2} GiB after its load and the reserve, {d:.2} GiB beside the prompt cache ({d:.2} GiB) and the graph cache ({d:.2} GiB); the largest context that fits is {d} tokens (--context {d}), or lower --prompt-cache-gib or TENSORFOLD_MEMORY_RESERVE_GIB", .{ window, gib(all[q].stream), if (e.drafter != null) " with the drafter" else "", q, gib(all[q].room), gib(left(all[q], cache, 1)), gib(cache), gib(graphBytes(1)), most, most }) catch "--context fits no stream: lower --context";
        std.log.err("inkling2: {s}", .{refusal});
        return error.ContextFitsNoStream;
    }
    const n = if (a.fixed) a.n else @min(fit, cap);
    // on pages the pool takes the room beside the n slots (stage 2d): the pages past n whole contexts serve longer
    // requests and the prompt cache's kept states; the streams stay n (whole contexts, capped)
    if (e.paged) return planned(e, all, cache, n, @max(poolPages(all, cache + pool_margin, n), @as(u64, n) * ic.pages.pagesFor(e.o.context)), if (a.fixed) "--parallel" else "auto", fit);
    const q = binding(all, cache, n);
    var cap_buf: [24]u8 = undefined;
    const capped = if (a.fixed) "" else std.fmt.bufPrint(&cap_buf, " (cap {d})", .{cap}) catch "";
    std.log.info("inkling2: context {d} -> {d} stream{s}{s} ({s}: {d} fit; rank {d} binds, {d:.2} GiB after its load and the reserve, less the prompt cache's {d:.2} GiB and the graph cache's {d:.2} GiB ({d} units), holds {d} of {d:.2} GiB{s}; at most {d})", .{
        window, n, if (n == 1) "" else "s", capped, if (a.fixed) "--parallel" else "auto", fit, q, gib(all[q].room), gib(cache), gib(graphBytes(n)), engine.graphUnits(n), left(all[q], cache, n) / @max(all[q].stream, 1), gib(all[q].stream), if (e.drafter != null) " with the drafter" else "", most_streams,
    });
    logTable(e, all, cache);
    // the server's admission: the smallest room less the graph cache's reserve, in rank 0's stream bytes
    var peer: Peer = .{ .bytes = std.math.maxInt(u64), .rank = 0 };
    for (all, 0..) |r, i| {
        if (r.room == std.math.maxInt(u64)) continue;
        const wide = @as(u128, r.room -| graphBytes(n)) * stream / @max(r.stream, 1);
        const bytes: u64 = @intCast(@min(wide, std.math.maxInt(u64)));
        if (bytes < peer.bytes) peer = .{ .bytes = bytes, .rank = i };
    }
    return .{ .streams = n, .peer = peer };
}

/// Rank 0, on pages: `n` slots on a pool of `npages` pages (`e.plan`, what `reserve` makes on every rank). Logs the
/// count, the pool and the most pages the room would hold beside the slots; the server's admission gets the smallest
/// room less the graph cache's reserve, in rank 0's bytes a slot (its fixed part and its share of the pool).
fn planned(e: *engine.Engine, all: []const RankRoom, cache: u64, n: u32, planned_pages: u64, how: []const u8, fit: u32) !Counted {
    const npages = if (poolCap()) |c| @min(planned_pages, @max(c, n)) else planned_pages;
    const room_pages = poolPages(all, cache, n);
    const page_rows = ic.pages.page_rows;
    const q = binding(all, cache, n);
    std.log.info("inkling2: context {d} -> {d} stream{s} on a KV page pool ({s}: {d} whole contexts fit; rank {d} binds): {d} pages of {d} positions ({d} positions, {d:.2} GiB a rank; {d:.2} GiB fixed a slot; {d} whole contexts' {d} pages and {d} more for longer requests and kept prompts), the largest request {d} positions; the room holds {d} pages ({d} positions) beside {d} slots, the prompt cache's {d:.2} GiB and the graph cache's {d:.2} GiB{s}", .{
        o_context(e), n, if (n == 1) "" else "s", how, fit, q, npages, page_rows, npages * page_rows, gib(npages * all[0].page), gib(all[0].fixed), @min(n, npages / ic.pages.pagesFor(e.o.context)), @min(npages, @as(u64, n) * ic.pages.pagesFor(e.o.context)), npages -| @as(u64, n) * ic.pages.pagesFor(e.o.context), @min(npages * page_rows, o_context(e)), room_pages, room_pages * page_rows, n, gib(cache -| scrubBytes(all)), gib(graphBytes(n)), if (e.drafter != null) " (with the drafter)" else "",
    });
    logTable(e, all, cache);
    e.plan = .{ .slots = n, .pages = @intCast(npages) }; // after the table: e.streamBytes() is then a slot's share
    // a slot's bytes as the server counts them: its fixed part and its share of the pool, on each rank
    const mine = all[0].fixed + (npages * all[0].page + n - 1) / n;
    var peer: Peer = .{ .bytes = std.math.maxInt(u64), .rank = 0 };
    for (all, 0..) |r, i| {
        if (r.room == std.math.maxInt(u64)) continue;
        const each = r.fixed + (npages * r.page + n - 1) / n;
        const wide = @as(u128, r.room -| graphBytes(n)) * mine / @max(each, 1);
        const bytes: u64 = @intCast(@min(wide, std.math.maxInt(u64)));
        if (bytes < peer.bytes) peer = .{ .bytes = bytes, .rank = i };
    }
    return .{ .streams = n, .peer = peer };
}

/// On pages, room left out of the pool beside the slots and the caches: the room readings' drift between the count
/// and the server's own placement (the pool is sized to the room after the reserve, not to whole contexts).
const pool_margin: u64 = 256 << 20;

/// TF_INK2_POOL_PAGES=N (tests, rank 0): the pool at most N pages (a page a slot at least), so admission's waits and
/// refusals can be reached on any host.
fn poolCap() ?u64 {
    const v = std.c.getenv("TF_INK2_POOL_PAGES") orelse return null;
    const n = std.fmt.parseInt(u64, std.mem.span(v), 10) catch return null;
    std.log.warn("inkling2: TF_INK2_POOL_PAGES={d} caps the KV page pool (a test setting)", .{n});
    return n;
}

/// The served positions a slot holds (the table's capacity is a whole number of pages past it).
fn o_context(e: *const engine.Engine) usize {
    return e.o.context - context_slack;
}

/// The README's contexts, for `logTable`.
const table_contexts = [_]usize{ 32768, 131072, 262144, 524288, 1048576 };

/// Rank 0 (log only): the streams `--parallel auto` would serve at each of `table_contexts` beside the same caches,
/// from these rooms (each rank's bytes a stream scaled as rank 0's at that context; the rooms are this start's, at
/// --context: the drafter's attention buffers and the op scratch's need grow a little with the context).
fn logTable(e: *const engine.Engine, all: []const RankRoom, cache: u64) void {
    var buf: [320]u8 = undefined;
    var at: usize = 0;
    const stream0 = e.streamBytesAt(e.o.context); // a whole context's bytes (on pages: before the pool's share)
    for (table_contexts) |c| {
        var scaled: [ink.plan.max_world]RankRoom = undefined;
        const b0 = (At{ .e = e }).at(c + context_slack);
        for (all, scaled[0..all.len]) |r, *x| x.* = .{ .room = r.room, .stream = @intCast(@min(@as(u128, b0) * r.stream / @max(stream0, 1), std.math.maxInt(u64))) };
        const n = fitStreams(scaled[0..all.len], cache, most_streams);
        const t = std.fmt.bufPrint(buf[at..], " {d} -> {d} ({d:.2} GiB a stream);", .{ c, n, gib(b0) }) catch break;
        at += t.len;
    }
    std.log.info("inkling2: --parallel auto by --context at these rooms, before the cap{s}:{s}", .{ if (e.drafter != null) " (with the drafter)" else "", buf[0..at] });
}

/// W = 1: this rank's room, counted as `count` does (null: no reading).
fn alone(e: *engine.Engine, o: Options) !?Counted {
    const r = o.room orelse return null;
    const b = r.read(r.ctx) orelse return null;
    std.log.info("inkling2 rank 0: {d:.2} GiB of room after its load ({d:.2} GiB a stream)", .{ gib(b), gib(e.streamBytes()) });
    const all = [1]RankRoom{mineOf(e, b)};
    return try count(e, &all, o);
}

/// Every rank's room after its load (the server's reading: on a unified-memory GPU such as GB10 that is
/// /proc/meminfo's MemAvailable, not cuMemGetInfo) and its stream bytes, gathered over the group; rank 0 counts the
/// streams from them (`count`) and gets the smallest room less the graph cache's reserve, in its own stream bytes,
/// for the server's admission. The other ranks get null (rank 0's count reaches them in `reserve`).
/// TF_INK2_FAKE_ROOM=R:GIB (tests): rank R reports GIB GiB.
/// This rank's room reading and its bytes a stream (on pages: a slot's fixed part and a page's too).
fn mineOf(e: *const engine.Engine, room: u64) RankRoom {
    return .{ .room = room, .stream = e.streamBytes(), .fixed = if (e.paged) e.fixedBytesAt(e.o.context) else 0, .page = e.pageBytes() };
}

fn rooms(e: *engine.Engine, g: *const cuda.tp_group.Group, o: Options) !?Counted {
    var mine: RankRoom = mineOf(e, if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64));
    if (fakeRoom(e.o.rank)) |b| {
        std.log.warn("inkling2 rank {d}: TF_INK2_FAKE_ROOM reports {d:.2} GiB of room (read {d:.2})", .{ e.o.rank, gib(b), gib(mine.room) });
        mine.room = b;
    }
    var all: [ink.plan.max_world]RankRoom = undefined;
    try g.gather(RankRoom, e.d, e.stream, mine, all[0..e.o.world]);
    if (e.o.rank != 0) return null;
    var any = false;
    for (all[0..e.o.world], 0..) |r, q| {
        if (r.room == std.math.maxInt(u64)) continue; // no reading
        any = true;
        std.log.info("inkling2 rank 0: rank {d} has {d:.2} GiB of room after its load ({d:.2} GiB a stream there)", .{ q, gib(r.room), gib(r.stream) });
    }
    if (!any) return null;
    return try count(e, all[0..e.o.world], o);
}

fn fakeRoom(rank: u32) ?u64 {
    const v = std.mem.span(std.c.getenv("TF_INK2_FAKE_ROOM") orelse return null);
    const colon = std.mem.indexOfScalar(u8, v, ':') orelse return null;
    const r = std.fmt.parseInt(u32, v[0..colon], 10) catch return null;
    const g = std.fmt.parseFloat(f64, v[colon + 1 ..]) catch return null;
    if (r != rank or !(g >= 0)) return null;
    return @intFromFloat(g * (1 << 30));
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

fn reserve(ptr: *anyopaque, streams: u32) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    try own.lanes.x.reserve(streams);
}

/// NCCL_COMM_ID from --master and --master-port when the environment has none.
fn commId(gpa: std.mem.Allocator, o: Options) !void {
    if (std.c.getenv("NCCL_COMM_ID") != null) return;
    const host = o.master orelse return error.NoCommId;
    const v = try std.fmt.allocPrintSentinel(gpa, "{s}:{d}", .{ host, o.master_port }, 0);
    defer gpa.free(v);
    if (setenv("NCCL_COMM_ID", v.ptr, 1) != 0) return error.NoCommId;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// TF_INK2_SCRATCH_GIB: the op scratch a forward may take (cuda_scratch.zig: default 1).
fn scratchGib() usize {
    return ic.scratch.gib();
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.lanes.x.end() catch |err| std.log.err("inkling2: ending the other ranks: {s}", .{@errorName(err)});
    own.lanes.deinit();
    release(own);
}

fn release(own: *Owned) void {
    own.e.deinit();
    if (own.group) |*g| g.close();
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "Inkling-2 at TP: --tp W (1, 2 or 4) and --rank R below W on every rank (rank 0 serves)",
        error.NoCommId => "Inkling-2's ranks meet over NCCL: set NCCL_COMM_ID to rank 0's address:port on every rank (or --master and --master-port)",
        error.CapturedKernels => "Inkling-2 runs its own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.BadInk2Layers => "TF_INK2_LAYERS: a layer count from 1 to the model's 48",
        error.RankMismatch => "the ranks differ in executable, checkpoint or settings (TF_INK2_LAYERS, --context, TF_INK2_*): the log names which",
        error.TransportSettingsDiffer => "the ranks differ in their transport settings (TF_ROCE, TF_INK2_ROCE_MULTI, TF_TP_*, the kernel image's contents): the log names the setting and the rank",
        error.TransportNotReady => "a rank cannot open its transports (TF_TP_KERNEL_IMAGE missing or unreadable, or an invalid TF_TP_* or TF_ROCE_MAX_KB value): the log names the rank and why",
        error.BuiltWithoutKernels => "this build has no CUDA kernels for Inkling-2 (build with -Dnvcc and the GPU's -Dsm)",
        error.ContextFitsNoStream, error.ParallelDoesNotFit, error.BadMaxStreams => if (refusal.len > 0) refusal else "the streams do not fit at --context: lower --context",
        error.RequestOverPool => "the request (its prompt, max_tokens and a verify window) needs more KV pages than this server's whole page pool holds: shorten the prompt or lower max_tokens (the startup log names the pool's positions)",
        error.PastPages => "the stream ran past the KV pages it reserved (its prompt, max_tokens and a verify window)",
        error.VisionMaxImages => if (refusal.len > 0) refusal else "--vision-max-images: an image chunk splices at most 16 images",
        error.NoVisionTower => "--vision: rank 0's folder has no inkling2-vision.safetensors (the recipe's weights step adds it with VISION=1)",
        error.ScratchShort => if (refusal.len > 0) refusal else null,
        error.ImageDigestDiffers => "an image's rows are not the image the request named (its digest differs, or is all zeros)",
        error.NotInkling2Image => "an image the request carries was not prepared by Inkling-2's image processor (its record does not read)",
        error.TooManyImages => "the request holds more images than --vision-max-images allows",
        error.ImageTokens => "the request's images hold more image tokens than --vision-image-tokens allows",
        error.ImageOutsidePrompt => "an image must end before the prompt's last token (the chat template puts text after it)",
        error.ImageOrder, error.ImageGrid => "the request's images overlap, are out of prompt order, or their grids differ from their tokens",
        error.UnexpectedVisionTensors => "--vision: the vision tower's file does not hold Inkling-2's 422 model.visual.* tensors (the log names the count)",
        error.TcCoreAbsent => "--vision: this build's kernels lack the vision tower's tensor-core attention (rebuild with -Dnvcc for the GPU's -Dsm)",
        else => null,
    };
}

test "inkling2 native: refusals explain themselves" {
    try std.testing.expect(explain(null, error.NoCommId) != null);
    try std.testing.expect(explain(null, error.NoRank) != null);
    try std.testing.expect(explain(null, error.TransportSettingsDiffer) != null);
    try std.testing.expect(explain(null, error.TransportNotReady) != null);
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
    try std.testing.expect(explain(null, error.NoVisionTower) != null);
}

test "inkling2 native: --vision-max-images above the splice's 16 spans is refused by name with both numbers" {
    defer refusal = "";
    const base: Options = .{ .context = 4096, .drafts = false };
    try checkVision(base); // no --vision: nothing to check
    var o = base;
    o.vision = .{};
    try checkVision(o);
    try std.testing.expectEqual(@as(?engine.Vision, .{ .image_tokens = 10_816, .max_images = 4 }), engineVision(o));
    o.vision.?.max_images = 16;
    try checkVision(o);
    o.vision.?.max_images = 20;
    try std.testing.expectError(error.VisionMaxImages, checkVision(o));
    const why = explain(null, error.VisionMaxImages).?;
    try std.testing.expect(std.mem.indexOf(u8, why, "--vision-max-images 20") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "at most 16 images") != null);
    // the family default is the engine's
    try std.testing.expectEqual(ic.vision.default_image_tokens, default_image_tokens);
    try std.testing.expectEqual(@as(u32, (engine.Vision{}).max_images), (image_offer.Limits{}).max_images);
}

test "inkling2 native: rank 0's start-up estimate with --vision is the tower and its row store, about 803 MiB" {
    const b = try visionBytes(std.testing.allocator, undefined, "unused");
    try std.testing.expect(b > 802 << 20 and b < 804 << 20);
}

test "inkling2 native: TF_INK2_MAX_STREAMS takes 1 .. 8" {
    try std.testing.expectEqual(@as(?u32, 4), parseMaxStreams("4"));
    try std.testing.expectEqual(@as(?u32, 1), parseMaxStreams("1"));
    try std.testing.expectEqual(@as(?u32, most_streams), parseMaxStreams("8"));
    try std.testing.expectEqual(@as(?u32, null), parseMaxStreams("0"));
    try std.testing.expectEqual(@as(?u32, null), parseMaxStreams("9"));
    try std.testing.expectEqual(@as(?u32, null), parseMaxStreams("auto"));
    try std.testing.expectEqual(@as(?u32, null), parseMaxStreams(""));
}

test "inkling2 native: --parallel auto counts the streams every rank holds beside the prompt and graph caches" {
    const G: u64 = 1 << 30;
    if (std.c.getenv("TF_INK2_GRAPHS_MAX") != null) return error.SkipZigTest;
    // one stream: 8 units (0.25 GiB); more: 64 units (2 GiB)
    try std.testing.expectEqual(@as(u64, G / 4), graphBytes(1));
    try std.testing.expectEqual(@as(u64, 2 * G), graphBytes(2));
    const cache = 3 * G / 2;
    // rank 1 binds: 10 GiB less 1.5 and 2 leaves 6.5, six 1 GiB streams
    var all = [_]RankRoom{ .{ .room = 12 * G, .stream = G }, .{ .room = 10 * G, .stream = G }, .{ .room = std.math.maxInt(u64), .stream = G } };
    try std.testing.expectEqual(@as(u32, 6), fitStreams(&all, cache, most_streams));
    try std.testing.expectEqual(@as(usize, 1), binding(&all, cache, 6));
    // no cache: 8 (the cap), not 8.0 / 1
    try std.testing.expectEqual(@as(u32, 8), fitStreams(&all, 0, most_streams));
    // one long stream: 9 GiB does not fit 10 - 1.5 - 0.25 = 8.25; 8 GiB does
    all[0].stream = 9 * G;
    all[1].stream = 9 * G;
    try std.testing.expectEqual(@as(u32, 0), fitStreams(&all, cache, most_streams));
    all[0].stream = 8 * G;
    all[1].stream = 8 * G;
    try std.testing.expectEqual(@as(u32, 1), fitStreams(&all, cache, most_streams));
    // two streams of 4 GiB take the 2 GiB graph reserve: 10 - 1.5 - 2 = 6.5 < 8
    all[0].stream = 4 * G;
    all[1].stream = 4 * G;
    try std.testing.expectEqual(@as(u32, 1), fitStreams(&all, cache, most_streams));
    // the largest context for one stream: 0.12 GiB + 8 KiB a position on rank 0 (rank 1's the same) in rank 1's
    // 9 - 1.5 - 0.25 = 7.25 GiB: about 0.93M positions, below the 1M asked
    const Lin = struct {
        fn at(_: @This(), positions: usize) usize {
            return (G * 12 / 100) + positions * 8192;
        }
    };
    all[0].stream = 1;
    all[1].stream = 1;
    all[1].room = 9 * G;
    const most = largestContext(&all, cache, 1, Lin{}, 1 << 20);
    try std.testing.expect(most % 1024 == 0 and most > 0 and most < 1 << 20);
    try std.testing.expect(Lin.at(.{}, most + context_slack) <= left(all[1], cache, 1));
    try std.testing.expect(Lin.at(.{}, most + 1024 + context_slack) > left(all[1], cache, 1));
    // rank 1 with no room for the caches: nothing fits, no context either
    all[1].room = G;
    try std.testing.expectEqual(@as(u32, 0), fitStreams(&all, cache, most_streams));
    try std.testing.expectEqual(@as(usize, 0), largestContext(&all, cache, 1, Lin{}, 1 << 20));
}
