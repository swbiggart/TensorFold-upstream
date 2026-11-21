//! Inkling-2's CUDA lane backend: a slot a stream (its state on every rank), prompts in 2048-row chunks, then decode
//! windows. Rank 0 draws on the host from the softcapped fp32 logits rows: greedy, or the request's sampling keyed at
//! each row's position over the whole vocabulary (sample.zig). Without the drafter a window is the pending row alone.
//! With it (TF_INK2_DSPARK=1, cuda_dspark.zig) a window is the pending row and up to 7 held drafts in one forward;
//! `keep` takes the slot back to the accepted rows (cuda_engine.zig keep: the convolution states; keys and values
//! are rewritten) and `draft` runs the drafter's block from the accepted rows' taps. The keep goes to the ranks with
//! the next draft (or before the next window when no draft came between).
//! Several streams' windows run as one forward (cuda_engine.zig forwardMany: a segment a stream, each stream's rows
//! with the bits of a forward of their own), so the lane core shares a round between streams (`hidden_rows`).
//! TF_INK2_BATCH=0 (rank 0) keeps a forward a stream (the lane core's per-stream rounds).
//! Structured output (a stream with a grammar): the lane core gives each row's allowed tokens (`Window.masks`, and
//! the first token's to `first_masked`); rank 0 sets the others to -inf in a copy of the row and draws it by the same
//! rule (`drawMasked`). The prompt's last row is kept for a grammar stream until its first token is drawn. A greedy
//! row takes the device's argmax when its mask allows it (the first largest value is then the first largest allowed
//! one), else its logits come down then and the largest allowed value wins; only sampled rows bring theirs down first.
//! With the drafter, a grammar stream's held drafts are made in its verify's own step, in the round's one drafter
//! pass, as every stream's are (Backend.late_masks): the window goes out unread, and its rows' masks are walked
//! (`Constraint.late`) once the step's drafts are on the host, before its draws. TF_INK2_LATE_MASKS=0 (rank 0) reads
//! them back first instead (a drafter pass of the grammar streams' own before their windows; for A/B).
//! TF_INK2_MASK_DUMP=PATH (rank 0, debug): every masked draw appends its row and mask to PATH (`dumpRow`).
//! The lane plumbing follows minimax_m3/cuda_lanes.zig and glm/cuda_lanes.zig.
const std = @import("std");
const lanes = @import("lanes");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");
const sample = @import("sample.zig");
const image = @import("image.zig");

const be = lanes.backend;
const engine = ic.engine;
const vs = ink.vision_store;

/// Drawn tokens a handle names.
const ring = 1024;

const Lane = struct {
    slot: u32,
    first: ?u64 = null,
    fill: ?usize = null, // a prompt going a chunk at a time: where its next chunk starts
    from: usize = 0, // where its prompt pass started: a kept state's position (the prompt cache), else 0
    kept: ?usize = null, // rows of the last window `keep` accepted, not yet sent to the ranks
    held: [ic.dspark.drafts]u32 = undefined, // the drafter's block for the next window
    held_n: usize = 0,
    /// The lone cut (TF_INK2_DRAFT_CUT): whether the stream's held chain is cut by its chances, and the chances of
    /// its held block (each draft's pick probability, the drafter's own), read when its draft ran: in `flushWhere`
    /// (deferred) or right after its eager pass (`eagerPass`, when a stream of the pass can use them).
    cut: bool = false,
    chances: [ic.dspark.drafts]f32 = undefined,
    chances_ok: bool = false,
    /// The draft `draft` asked for, not run yet (TF_INK2_DRAFT_DEFER): it runs with the stream's next verify, in its
    /// step (Leader.forwardDrafted: one header, the keep, the block, then the window), its drafts read back after
    /// the head. Any other forward first runs it on its own (`flush`), since the block reads the taps of the
    /// stream's last forward.
    pending: ?Pending = null,
    /// A grammar stream's prompt's last row (the softcapped fp32 logits, owned) until its first token is drawn:
    /// `first_masked` draws it under the grammar's mask (or `first` without one).
    first_row: []f32 = &.{},
    /// An image prompt's pass (rank 0 with the vision tower): its images as the row store names them, and the pass's
    /// pins and reservation there. Released when the pass ends, is dropped, or the stream opens again.
    img: ?Images = null,

    fn dropRow(l: *Lane, gpa: std.mem.Allocator) void {
        if (l.first_row.len > 0) gpa.free(l.first_row);
        l.first_row = &.{};
    }
};

/// A prompt pass's images (cuda_vision_store.zig): the stream's, in prompt order, as the row store names them
/// (`storeImage`), its `Need` (what it still needs, its pins and its reservation of store rows), and the units it ran
/// of other images' tower jobs (logged once when the pass ends).
const Images = struct {
    list: [vs.max_images]vs.Image = undefined,
    count: u32 = 0,
    need: vs.Need = undefined,
    help: vs.Help = .{},

    fn images(p: *const Images) []const vs.Image {
        return p.list[0..p.count];
    }
};

/// A stream's image as the row store names it: the prompt rows it fills, its grid (the record's: `image.record`
/// checks the stream's gh, gw and tokens against it) and the full digest of its pixels.
fn storeImage(m: lanes.stream.Image) vs.Image {
    return .{ .at = m.at, .tokens = m.tokens, .rows = m.gh, .cols = m.gw, .digest = m.digest };
}

/// The images of a kept prompt prefix of `at` rows (cuda_vision_store.zig prefixKey; cuda_snap.zig Kept.images): all
/// zeros, with no hashing, for a prompt without images.
pub fn imageKey(images: []const lanes.stream.Image, at: u32) error{TooManyImages}!vs.Digest {
    if (images.len == 0) return @splat(0);
    if (images.len > vs.max_images) return error.TooManyImages;
    var list: [vs.max_images]vs.Image = undefined;
    for (images, list[0..images.len]) |m, *o| o.* = storeImage(m);
    return vs.prefixKey(list[0..images.len], at);
}

/// Rank 0, an image prompt's opening, before anything moves (cuda_vision_store.zig Need.init's rules, and): each
/// image's record read back and agreeing with the stream's image (image.zig record: sides, grid and tokens), its
/// digest the one the server sealed in the record and never all zeros, each ending before the prompt's last row (the
/// last row is text: the first token's row and the drafter's tap), at most `v.max_images` images, and their tokens at
/// most the row store's rows. The images in `out`; returns their tokens.
fn checkImages(images: []const lanes.stream.Image, prompt: usize, v: ic.vision.Vision, out: *Images) !u32 {
    if (images.len > v.max_images or images.len > vs.max_images) return error.TooManyImages;
    var tokens: u64 = 0;
    for (images, out.list[0..images.len]) |m, *o| {
        const r = try image.record(m);
        if (!std.mem.eql(u8, &r.rec.digest, &m.digest) or std.mem.allEqual(u8, &m.digest, 0)) return error.ImageDigestDiffers;
        if (@as(u64, m.at) + m.tokens >= prompt) return error.ImageOutsidePrompt;
        o.* = storeImage(m);
        tokens += m.tokens;
    }
    out.count = @intCast(images.len);
    _ = try vs.Need.init(out.images(), 0); // order, overlap, grids and digests
    if (tokens > v.image_tokens) return error.ImageTokens;
    return @intCast(tokens);
}

/// Rank 0's tower units for a prompt pass (cuda_vision_store.zig Units): the engine-mode tower (cuda_vision.zig
/// Rank0) on the engine's stream, each unit's buffers borrowed from the op scratch (idle between forwards: every op
/// takes its own and the forward resets it before each; the tower's state between units is its resident `x`).
const TowerUnits = struct {
    e: *engine.Engine,
    v: *ic.vision.Rank0,
    s: *const lanes.Stream,
    list: []const vs.Image,

    fn units(u: *TowerUnits) vs.Units {
        return .{ .ctx = u, .begin = begin, .layers = layers, .merge = merge, .settle = settle, .now = now };
    }

    fn of(ctx: *anyopaque) *TowerUnits {
        return @ptrCast(@alignCast(ctx));
    }

    /// The op scratch from its start: `len` bytes for this unit alone.
    fn borrow(u: *TowerUnits, len: usize) !u64 {
        u.e.scratch.reset();
        u.e.scratch.kind = .other;
        return u.e.scratch.take(len);
    }

    fn begin(ctx: *anyopaque, i: u32) anyerror!void {
        const u = of(ctx);
        const r = try image.record(u.s.images[i]);
        const len = u.v.tower.sliceNeed(ic.vision.Grid.of(r.rec.height, r.rec.width));
        try u.v.beginUnit(try u.borrow(len), len, r.rgb, r.rec.height, r.rec.width);
        const m = u.list[i];
        std.log.info("inkling2 rank 0: vision: image {d} of the prompt ({d}x{d}, {d} tokens at {d}) to the tower", .{ i, r.rec.width, r.rec.height, m.tokens, m.at });
    }

    fn layers(ctx: *anyopaque, k: u32) anyerror!void {
        const u = of(ctx);
        const len = u.v.unitNeed();
        _ = try u.v.layersUnit(try u.borrow(len), len, k);
    }

    fn merge(ctx: *anyopaque, row0: u32) anyerror!void {
        const u = of(ctx);
        const len = u.v.unitNeed();
        const out = try u.v.mergeUnit(try u.borrow(len), len, u.v.store.addr(row0));
        const p = &u.v.pace;
        const st = &u.v.store;
        std.log.info("inkling2 rank 0: vision: the tower's image ({d} tokens) into store rows [{d}, {d}) in {d} unit(s){s}; the store: {d} entries, {d} evicted, {d} rows moved", .{ out.tokens, row0, row0 + out.tokens, p.units, if (p.layer_ns > 0) " (sliced beside decoders)" else "", st.entries.items.len, st.evicted, st.moved });
    }

    fn settle(ctx: *anyopaque) anyerror!void {
        try of(ctx).v.stream.synchronize();
    }

    fn now(ctx: *anyopaque) u64 {
        return @intCast(std.Io.Clock.awake.now(of(ctx).e.io).toNanoseconds());
    }
};

const Pending = struct { n: usize, anchor: u32, stream: *lanes.Stream };

/// The prompt cache's store as the lanes see it (native/cuda.zig: its giveBack and borrowed over the store).
pub const StoreHooks = struct {
    ctx: *anyopaque,
    /// Kept states dropped, least valuable first, never `restoring` nor a borrowed one, and when `guarded` never one a
    /// queued or admitted request will resume (the lane host's protection), until `fits(left)` (false: none left to
    /// drop).
    give_back: *const fn (ctx: *anyopaque, restoring: ?*anyopaque, left: *anyopaque, fits: *const fn (*anyopaque) bool, guarded: bool) bool,
    borrowed: *const fn (ctx: *anyopaque, saved: *anyopaque) bool,
};

/// What the reclaim chain did (`reclaimChain`): whether `want` pages are available after it, and whether it went
/// past the lane host's protection.
pub const Reclaimed = struct { ok: bool, unguarded: bool = false };

/// The reclaim chain over rank 0's allocator `a`: kept states give their pages back (the store's order: least
/// valuable first, never `restoring` nor a borrowed one) until `want` pages are available. First guarded: a state a
/// queued or admitted request will resume stays. When that leaves too few and the stream is `alone` (no other stream
/// runs, so waiting brings no page back), unguarded: admission (`Lanes.fits`) counted every page no borrowed state
/// holds, and the lane host fails a request that finds no memory with nothing else running, so the protected states
/// go instead of the request (a queued request that would have resumed one runs its prompt from the start: the same
/// bits). Engine thread.
pub fn reclaimChain(a: *const ic.pages.Alloc, hooks: StoreHooks, restoring: ?*anyopaque, want: usize, alone: bool) Reclaimed {
    if (a.available() >= want) return .{ .ok = true };
    const Left = struct {
        a: *const ic.pages.Alloc,
        want: usize,
        fn fits(ptr: *anyopaque) bool {
            const l: *const @This() = @ptrCast(@alignCast(ptr));
            return l.a.available() >= l.want;
        }
    };
    var left: Left = .{ .a = a, .want = want };
    if (hooks.give_back(hooks.ctx, restoring, &left, Left.fits, true)) return .{ .ok = true };
    if (!alone) return .{ .ok = false };
    return .{ .ok = hooks.give_back(hooks.ctx, restoring, &left, Left.fits, false), .unguarded = true };
}

/// A lone window's ms by rows at TP4 on four GB10s (switch fabric with the RoCE mesh wire, checkpoint mode, experts
/// split, all 48 layers, the drafter loaded, release defaults), measured by `tf-inkling2-decode --exp check,price` (20
/// a width, two runs, the cells' medians) at a 600-token prompt (the server's requests' depth) on v0.2 with the decode
/// launch merges (one-launch projections, the bf16 router, the convolutions with their commits) and the drafter's
/// packed global attention: the window as a captured graph with every row's head logits, plus the keep that rolls
/// back the rejected rows; rows 2-8 on their least-squares line in rows (cells scatter 1-5 ms run to run).
const tp4_window_ms = [_]f64{ 29.9, 38.6, 43.0, 47.5, 51.9, 56.3, 60.7, 65.1 };
/// The drafter's block (with its keep) at TP4, measured the same way: one pass whatever its depth (its 7 drafts at
/// once), so a round's draft is a fixed cost, not a cost a draft.
const tp4_draft_ms = 8.2;
/// What the lane core weighs depths by: a round of a window of R rows (R - 1 drafts) is the window and, from two
/// rows on, the draft that made its drafts (the head steps are priced at 0: `mtp_step_ms`).
const tp4_costs = costs: {
    var c: [tp4_window_ms.len]lanes.config.Cost = undefined;
    for (&c, tp4_window_ms, 0..) |*x, ms, i| x.* = .{ .width = i + 1, .ms = ms + if (i > 0) tp4_draft_ms else 0 };
    break :costs c;
};
/// A lone round's prices by the stream's context (lanes/depth.zig ContextCost): the drafter's global layer and the
/// window's global layers read the whole context, so at depth the draft and every window cost more, a window's rows
/// more than one row. The first node is `tp4_window_ms`/`tp4_draft_ms` (the server's shallow requests); the others were
/// measured as those were, on the same Sparks and build: 3,040 and 32,768 by `--exp check,price` (20 a width, two runs)
/// and `--exp deep-draft --price-depths` (10 a width, one run in one prefill of real text, slot 524,304), the cells'
/// medians; 131,072 .. 524,000 by the deep run alone, 1,048,000 by `--price-deep` on a synthetic fill (its draft 35 %
/// above 524,000's, so a 1M stream is no longer priced by the 524,000 node); the window as a captured graph with
/// every row's head logits plus the keep, the draft with its keep; rows 2-8 on their line at every node. A round is
/// priced at its stream's own
/// context (interpolated between two nodes), and a plain round competes with the drafted depths (lanes/depth.zig
/// `depth`), so a stream deep enough that its drafts do not pay decodes plainly. In a shared round each stream pays
/// what its rows and its draft cost past `depth_from` (lanes/depth.zig `depthMs`): a deep stream drafts less or not at
/// all beside shallow streams. Up to `depth_from` (32,768: the nodes there are one price within noise) a stream is
/// priced as v0.1 prices it (lone rounds by `tp4_costs`, always drafting; shared rounds without depth): on short
/// prompts a plain round competing lost 4-14 % on bursty acceptance (it went plain in a stretch of misses and waited).
const depth_costs = [_]lanes.depth.ContextCost{
    .{ .context = 600, .window_ms = &tp4_window_ms, .draft_ms = tp4_draft_ms },
    .{ .context = 3_040, .window_ms = &.{ 30.1, 39.5, 44.0, 48.6, 53.1, 57.7, 62.2, 66.8 }, .draft_ms = 8.4 },
    .{ .context = 32_768, .window_ms = &.{ 31.3, 40.1, 44.1, 48.2, 52.2, 56.3, 60.3, 64.4 }, .draft_ms = 8.7 },
    .{ .context = 131_072, .window_ms = &.{ 36.2, 45.6, 49.9, 54.2, 58.5, 62.9, 67.2, 71.5 }, .draft_ms = 10.8 },
    .{ .context = 262_144, .window_ms = &.{ 44.3, 52.2, 57.1, 62.0, 66.9, 71.8, 76.7, 81.6 }, .draft_ms = 12.1 },
    .{ .context = 524_000, .window_ms = &.{ 58.0, 63.9, 69.6, 75.4, 81.2, 87.0, 92.7, 98.5 }, .draft_ms = 15.5 },
    .{ .context = 1_048_000, .window_ms = &.{ 80.2, 90.9, 98.5, 106.0, 113.6, 121.2, 128.7, 136.3 }, .draft_ms = 20.9 },
};

/// TF_INK2_DRAFT_CUT=0's prices: v0.2's own tables as it shipped them (measured on v0.2's kernels), so that `0`
/// schedules as v0.2 does, the same depths from the same rates; `tp4_costs` and `depth_costs` serve the cut and `ab`.
const v02_window_ms = [_]f64{ 34.1, 42.4, 46.9, 51.7, 57.6, 60.2, 65.5, 67.3 };
const v02_draft_ms = 8.4;
const v02_costs = costs: {
    var c: [v02_window_ms.len]lanes.config.Cost = undefined;
    for (&c, v02_window_ms, 0..) |*x, ms, i| x.* = .{ .width = i + 1, .ms = ms + if (i > 0) v02_draft_ms else 0 };
    break :costs c;
};
const v02_depth_costs = [_]lanes.depth.ContextCost{
    .{ .context = 600, .window_ms = &v02_window_ms, .draft_ms = v02_draft_ms },
    .{ .context = 3_040, .window_ms = &.{ 33.9, 41.1, 47.1, 50.9, 54.3, 61.4, 63.6, 68.2 }, .draft_ms = 8.5 },
    .{ .context = 32_768, .window_ms = &.{ 35.1, 42.2, 47.7, 52.6, 56.9, 65.4, 66.9, 72.2 }, .draft_ms = 9.4 },
    .{ .context = 131_072, .window_ms = &.{ 41.1, 48.1, 54.5, 62.2, 69.7, 73.5, 82.2, 90.3 }, .draft_ms = 16.1 },
    .{ .context = 262_144, .window_ms = &.{ 48.3, 56.1, 65.3, 74.7, 86.8, 91.3, 103.9, 115.9 }, .draft_ms = 23.0 },
    .{ .context = 524_000, .window_ms = &.{ 63.2, 72.0, 82.7, 99.9, 110.1, 121.7, 140.7, 150.3 }, .draft_ms = 39.7 },
};

const depth_from: u64 = 32_768;

/// TF_INK2_DRAFT_DEPTH (rank 0; A/B tooling for the depth measurements, not a setting of the release: the recipe never
/// sets it): `0` prices a round by `tp4_costs` whatever the stream's context (v0.1's prices); `ab` prices new streams
/// in turn by their context, by `tp4_costs`, and plain (never drafted), each logged as it starts ("lanes depth a/b");
/// anything else (or unset) by the stream's context (`depth_costs`), the release's behaviour. `ab` with
/// TF_INK2_DRAFT_CUT=ab is refused at startup (`abSwitches`).
fn depthPriced() bool {
    const v = std.c.getenv("TF_INK2_DRAFT_DEPTH") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

fn depthAb() bool {
    const v = std.c.getenv("TF_INK2_DRAFT_DEPTH") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "ab");
}

/// The two A/B switches one at a time (rank 0, at startup): with TF_INK2_DRAFT_DEPTH=ab and TF_INK2_DRAFT_CUT=ab
/// together a table-priced stream falls in either cut arm, and both arms' rounds price it by one table
/// (lanes/depth.zig Rule.ab_table_ms), so neither comparison would hold; refused, by name.
pub fn abSwitches() !void {
    abPair(depthAb(), draftCut()) catch |err| {
        std.log.err("inkling2: TF_INK2_DRAFT_DEPTH=ab and TF_INK2_DRAFT_CUT=ab together are refused: the two A/B experiments would share one timing table; run one at a time", .{});
        return err;
    };
}

fn abPair(depth_ab: bool, cut: Cut) error{DraftAbTogether}!void {
    if (depth_ab and cut == .ab) return error.DraftAbTogether;
}

/// A shared forward's ms by its total rows at TP4 (several streams' rows as a captured graph, B2 C2; the same Sparks
/// and settings), measured by `tf-inkling2-decode --exp shared-price` (medians of 10, after the key's eager first
/// sighting and its capture) at prompts of 417 .. 600 tokens: R rows as the fewest streams of at most 8 rows each (at
/// least two; one row: a lone graph step), with every row's head logits.
/// Rows between two measured counts are interpolated (lanes allocate.zig extendCosts). The lane core weighs a shared
/// round's rows against them; what a round costs past them (its streams' draft, the host) it learns as the round's
/// overhead, by streams.
const shared_ms = [_]struct { u32, f64 }{
    .{ 1, 33.8 },  .{ 2, 44.7 },   .{ 3, 53.5 },   .{ 4, 56.7 },   .{ 5, 63.0 },   .{ 6, 66.1 },   .{ 7, 74.4 },
    .{ 8, 77.3 }, .{ 12, 90.4 }, .{ 16, 105.3 }, .{ 20, 126.5 }, .{ 24, 135.6 }, .{ 28, 153.5 }, .{ 32, 161.1 },
};
const shared_costs = costs: {
    var c: [shared_ms.len]lanes.config.Cost = undefined;
    for (&c, shared_ms) |*x, m| x.* = .{ .width = m[0], .ms = m[1] };
    break :costs c;
};
/// A shared forward's streams and rows at most: the step header's segments; 32 rows keeps a round's forward under the
/// by-rows exchange's 64 (cuda_xrows.zig) and the head's 128.
const batch_streams: u32 = engine.max_segs;
const batch_rows: u32 = 32;

/// TF_INK2_BATCH (rank 0): `0` keeps a forward a stream; anything else (or unset) shares rounds.
fn batching() bool {
    return !ic.dspark.switchedOff("TF_INK2_BATCH");
}

/// TF_INK2_LATE_MASKS (rank 0): `0` reads a grammar stream's held drafts back before its window (a drafter pass of
/// the grammar streams' own a round) so the lane core walks them first; anything else (or unset) leaves them in the
/// verify's step with every other stream's and walks them after (Backend.late_masks: the same draws, one pass).
fn lateMasks() bool {
    const v = std.c.getenv("TF_INK2_LATE_MASKS") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

/// TF_INK2_DRAFT_DEFER (rank 0): `0` runs a stream's draft when the lane core asks for it (its own header, then a
/// wait for its drafts); anything else (or unset) runs it in the step of the stream's next verify (one header a
/// round, no wait between the draft and the window: the host gap between them measured 0.0-0.02 ms, so overlapping
/// the draft with the host would gain nothing).
fn deferDrafts() bool {
    const v = std.c.getenv("TF_INK2_DRAFT_DEFER") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

/// TF_INK2_DRAFT_STREAMS (rank 0): `0` drafts each stream of a shared round in a pass (and a header) of its own;
/// anything else (or unset) drafts them all in one pass (Leader.draftMany: each stream's drafts the same bits).
fn draftStreams() bool {
    return !ic.dspark.switchedOff("TF_INK2_DRAFT_STREAMS");
}

/// A shared round's drafter pass at TP4 (one pass for every drafting stream, Leader.draftMany), measured by the
/// streams gate's pass C (one pass against the streams' own: 8.4 ms for one stream, 21.4 for 3.7, 35.8 for 6.9 at
/// prompts of 173 .. 600 tokens, 36.6 for 7.1 at 3k): about 3.8 ms plus 4.6 a drafting stream. A drafted shared
/// round also runs its forward eagerly far more often than a plain one (its key, the rows a stream, rarely repeats;
/// `shared_ms` are graph prices): 3.5-6.4 ms more at 8-32 rows, priced as 4.2 ms on the pass until the lane core has
/// measured drafted and plain rounds' overheads apart (lanes/depth.zig DraftPass).
/// In rounds of more than 4 streams no stream's share may fall below plain's (lanes/depth.zig `fair`): at 8 streams on
/// distinct prompts the allocation alone left the streams whose drafts land seldom slower than plain (DSpark 71.0-71.3
/// against plain 74.4-74.6 aggregate; with the rule 74.0 against 74.8), while at 4 streams the rule cost 71 -> 66.
const shared_pass: lanes.depth.DraftPass = .{ .base_ms = 3.8, .stream_ms = 4.6, .drafted_ms = 4.2, .fair_above = 4 };

/// TF_INK2_DRAFT_PRICE (rank 0): `0` prices a shared round's drafts free and every stream of it drafts at least one
/// (stage 2c, for A/B); anything else (or unset) prices the pass (`shared_pass`), so a stream drafts only when its
/// drafts' expected tokens a ms beat the round without them, and a round where none pays runs plain (no pass).
/// TF_INK2_DRAFT_MOST=N (with the price): a shared round of more than N streams drafts none. TF_INK2_DRAFT_FAIR=N:
/// the plain-share rule in rounds of more than N streams (0: every shared round; default 4). TF_INK2_DRAFT_LOG=1: the
/// lane core logs each shared round's decision and observed overhead (diagnostics).
fn draftPass() ?lanes.depth.DraftPass {
    if (ic.dspark.switchedOff("TF_INK2_DRAFT_PRICE")) return null;
    var p = shared_pass;
    p.deferred = deferDrafts();
    if (std.c.getenv("TF_INK2_DRAFT_LOG")) |v| p.log = v[0] == '1';
    if (std.c.getenv("TF_INK2_DRAFT_FAIR")) |v| p.fair_above = std.fmt.parseInt(u32, std.mem.span(v), 10) catch p.fair_above;
    if (std.c.getenv("TF_INK2_DRAFT_MOST")) |v| p.most_streams = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    return p;
}

/// TF_INK2_SHARED_CUT (rank 0, read when the lanes start; lanes/windows.zig sharedChances): unset (the default) or
/// `deepen` (or `2`): a shared round of 2 .. 4 streams (the pass's fair_above; one stream and 5 or more keep v0.2's
/// rounds) prices its greedy streams' held drafts by the drafter's chances of this round through the lone cut's
/// calibration map (`chainChances`; the deferred drafts then run before the verify, their chances read), and every
/// drafting greedy stream drafts its whole block, so the rows go to the drafts likeliest to land, a confident stream's
/// deeper ones too; `cut` (or `1`) the chances only, within the budget; `0`, `off` or `false` (any case) price by each
/// stream's learned rates, as v0.2 does. Every rank hashes the value as resolved (cuda_dspark.zig sharedCutEffective).
/// Greedy streams without a grammar in the cut arm (TF_INK2_DRAFT_CUT=ab's whole arm has no chances read: neither
/// priced nor deepened, Stream.no_chances); off with TF_INK2_DRAFT_CUT=0 or TF_INK2_DRAFT_DEPTH=0 (v0.2's rounds).
/// Outputs are the same either way. GB10, v0.2's final kernels and tables (hold A6), two servers a setting (`0`, the
/// default), CONTEXT=32768 MAX_STREAMS=8, the frozen bench prompts, the time to the last request's end: 2 streams
/// +5.9%, 4 streams +5.5%, no other level slower beyond its spread; replayed on texts whose drafts seldom land (a first
/// draft kept about 3 times in 10), 20-26% slower than with it off at 2 to 4 streams.
const shared_cut_default: lanes.config.SharedCut = .deepen;

fn parseSharedCut(v: ?[]const u8) lanes.config.SharedCut {
    return switch (ic.dspark.parseSharedCut(v)) {
        .off => .off,
        .cut => .cut,
        .deepen => shared_cut_default,
    };
}

/// Whether a flush of streams a forward leaves out reads their chances (for their next shared window): only in a round
/// the lane core prices by chances (`open`: 2 .. 4 live streams, Lanes.round_open) and only when shared rounds use
/// chances in effect, which TF_INK2_DRAFT_CUT=0 turns off (the drafter then makes no chances, and asking for them
/// fails the request), as `factsCut` does; else v0.2's flush, proposals only (no wait, no download).
fn leftOutChances(open: bool, cut: Cut, shared: lanes.config.SharedCut) bool {
    return open and cut != .off and shared != .off;
}

fn sharedRoundFn(ptr: *anyopaque, open: bool) void {
    const self: *Lanes = @ptrCast(@alignCast(ptr));
    self.round_open = open;
}

fn sharedCut() lanes.config.SharedCut {
    return parseSharedCut(if (std.c.getenv("TF_INK2_SHARED_CUT")) |v| std.mem.span(v) else null);
}

/// TF_INK2_DRAFT_CUT (rank 0 cuts; every rank reads it, cuda_dspark.zig draftCut: off, the drafter makes no chances):
/// unset (or anything else) cuts a lone greedy stream's held chain (at or below `depth_from`, without a grammar) to the
/// drafts whose chances pay (lanes/depth.zig Rule.cut): the stream's draft runs before its window (`probabilities`: the
/// undeferred draft) so its seven chances are on the host; `0` verifies the rule's whole chain and prices rounds by
/// v0.2's tables (`v02_costs`, `v02_depth_costs`), v0.2's behaviour; `ab` cuts every other new stream (logged as it
/// opens: "inkling2: draft cut a/b"), for a same-server A/B: each arm's rounds are timed and priced apart, deep ones
/// too (lanes/depth.zig Rule.cut_arm_ms, cut_arm_context_ms). Outputs are the same either way (a window's rows are
/// exact at any width); only the rows verified change. GB10, one server in `ab` mode (before the arms were timed
/// apart), cut against whole (median tok/s): fibonacci-raw +12.6%, gpu-chat-no-think +5.9%, code-lru-chat +3.6%,
/// tool-call-weather +6.6% (30 requests an arm), thinking-high +3.2%.
const Cut = ic.dspark.Cut;
const draftCut = ic.dspark.draftCut;

/// The drafter's pick probability as the chance its draft lands (given the drafts before it landed): knots (pick
/// probability, landed share) of the device's own chances on the chain's path, binned and pooled to rise
/// (tf-inkling2-draft --chances over 12 replies of the five bench workloads and a long chat, 3,163 anchors), linear
/// between knots, through 0 below the first and flat past the last. Fitted on the FP8 'mlp' drafter (the draft MLP's
/// projections as MXFP8 weights, TF_INK2_DSPARK_FP8=mlp).
const cut_map = [_][2]f64{ .{ 0.1659, 0.0000 }, .{ 0.2588, 0.1638 }, .{ 0.3553, 0.2903 }, .{ 0.4527, 0.3650 }, .{ 0.5510, 0.4632 }, .{ 0.6480, 0.4866 }, .{ 0.7521, 0.5960 }, .{ 0.8539, 0.6487 }, .{ 0.9272, 0.7179 }, .{ 0.9745, 0.8149 }, .{ 0.9961, 0.9354 }, .{ 0.9999, 0.9886 } };

fn landing(p: f64) f64 {
    if (!(p > cut_map[0][0])) return if (p > 0) cut_map[0][1] * p / cut_map[0][0] else 0;
    for (cut_map[0 .. cut_map.len - 1], cut_map[1..]) |a, b| if (p <= b[0]) return a[1] + (p - a[0]) / (b[0] - a[0]) * (b[1] - a[1]);
    return cut_map[cut_map.len - 1][1];
}

/// A lane's held drafts' chances of landing with every draft before them (the running product of `landing`); false
/// when its block's chances were not read or it holds fewer drafts than asked.
fn chainChances(l: *const Lane, out: []f64) bool {
    if (!l.chances_ok or out.len > l.held_n) return false;
    var run: f64 = 1;
    for (out, l.chances[0..out.len]) |*o, c| {
        run *= landing(c);
        o.* = run;
    }
    return true;
}

/// The drafter's acceptance by depth (offline, the checkpoint oracle's greedy replies on four prompts: each depth's
/// chance given the one before it landed).
const draft_prior = [_]f64{ 0.84, 0.81, 0.81, 0.81, 0.83, 0.84, 0.86 };

pub const Lanes = struct {
    x: ic.follow.Leader,
    gpa: std.mem.Allocator,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    /// TF_INK2_SCRIPT's tokens (debug), owned.
    script: []u32 = &.{},
    sampler: ?sample.Sampler = null,
    /// When the last `draft` that drafted returned (ns, the awake clock): the next verify measures the host gap
    /// from it (the done line's `gap`; the draft overlap's case), then clears it.
    draft_end: ?i96 = null,
    /// Since start: the gaps measured, their sum (ns), and drafts made.
    gaps: u64 = 0,
    /// shared_cut: whether the current round prices its drafts by chances (the lane core's Backend.shared_round)
    round_open: bool = false,
    gap_ns: u64 = 0,
    drafts_made: u64 = 0,
    /// Since start: verify rounds, their rows, and the ns their draw loops took (the host tail after the head).
    draw_rounds: u64 = 0,
    draw_rows: u64 = 0,
    draw_ns: u64 = 0,
    /// Structured output: a row with its disallowed tokens at -inf (rank 0's draw under a mask), and since start the
    /// masked draws and the ns they took (the copy, the mask and the draw).
    masked: []f32 = &.{},
    masked_rows: u64 = 0,
    masked_ns: u64 = 0,
    /// Greedy masked rows whose device argmax the mask refused (their logits brought down after the head).
    masked_misses: u64 = 0,
    /// Since start: late windows (Backend.late_masks), and their rows past a draft the grammar refused or its stop
    /// (drawn unconstrained, never on an accepted path).
    late_windows: u64 = 0,
    late_free_rows: u64 = 0,
    /// TF_INK2_MASK_DUMP (debug): 0 not read yet, 1 off, 2 on; records left (TF_INK2_MASK_DUMP_ROWS, default 4096);
    /// the flags of the window being drawn (8: late masks).
    dump_state: u8 = 0,
    dump_left: u64 = 0,
    dump_flags: u32 = 0,
    /// The prompt cache on pages (stage 2d): the server's store (cuda_native.zig snap_store), asked to give kept
    /// states' pages back before a stream's pages do not fit (the reclaim chain), and whether a state is borrowed
    /// (a resume in flight: its pages are not reclaimable).
    store: ?StoreHooks = null,
    /// TF_INK2_DRAFT_CUT=ab: the streams opened so far (every other one is cut).
    cut_streams: u64 = 0,
    /// The NVMe tier's write-behind has had an idle turn (logged once).
    idle_seen: bool = false,

    fn now(self: *const Lanes) i96 {
        return std.Io.Clock.awake.now(self.x.e.io).toNanoseconds();
    }

    pub fn deinit(self: *Lanes) void {
        var it = self.lanes.valueIterator();
        while (it.next()) |l| l.dropRow(self.gpa);
        self.lanes.deinit(self.gpa);
        if (self.masked.len > 0) self.gpa.free(self.masked);
        if (self.sampler) |*smp| smp.deinit();
        if (self.script.len > 0) self.gpa.free(self.script);
    }

    /// TF_INK2_SCRIPT: a file of token ids (whitespace or commas between them) every stream's reply starts with.
    pub fn loadScript(self: *Lanes, io: std.Io) !void {
        const path = std.c.getenv("TF_INK2_SCRIPT") orelse return;
        const text = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(path), self.gpa, .limited(1 << 20));
        defer self.gpa.free(text);
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(self.gpa);
        var it = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
        while (it.next()) |t| try ids.append(self.gpa, try std.fmt.parseInt(u32, t, 10));
        self.script = try ids.toOwnedSlice(self.gpa);
        std.log.warn("inkling2: TF_INK2_SCRIPT: every reply starts with the script's {d} tokens (debug)", .{self.script.len});
    }

    /// Chunk-invariant: a prompt's state and first token are the same bits however it is cut into forwards (every op
    /// is row-invariant; tf-inkling2-decode --exp chunks checks 2,048 .. 1-row cuts and seeded random ones against
    /// the decode check's hash), so the lane host may cap a long prompt's chunks beside live decoders
    /// (--decode-share: 512 rows a step, the share's debt between them).
    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = &vt, .chunk_invariant = true, .late_masks = lateMasks() };
    }

    const vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .prefill_step = prefillStepFn,
        .cancel_prefill = cancelPrefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .held = heldFn,
        .probabilities = probabilitiesFn,
        .shared_round = sharedRoundFn,
        .first_masked = firstMaskedFn,
        .release = releaseFn,
        .idle_pending = idlePendingFn,
        .idle = idleFn,
    };

    /// One row a window without the drafter (two for a forced token; any width is exact); with it, the pending row
    /// and up to 7 held drafts. Several streams' windows share a forward unless TF_INK2_BATCH=0.
    pub fn facts(self: *const Lanes) lanes.Model {
        return self.factsCut(draftCut());
    }

    /// `facts` under TF_INK2_DRAFT_CUT `cut`.
    fn factsCut(self: *const Lanes, cut: Cut) lanes.Model {
        const shared = batching();
        if (self.x.e.drafter != null) return .{
            .copy_drafts = false,
            .lone_copies = false,
            .root_copies = false,
            .context_copies = false,
            .exact_width = ic.state.spec_rows,
            .gpu_tokens = false,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .drafts = ic.dspark.drafts,
            .draft_prior = &draft_prior,
            .mtp_step_ms = 0, // the draft is a round's fixed cost, in tp4_costs (lone) or the round's overhead (shared)
            .window_costs = if (cut == .off) &v02_costs else &tp4_costs,
            .window_costs_lone = shared,
            .context_costs = if (!depthPriced()) &.{} else if (cut == .off) &v02_depth_costs else &depth_costs,
            .depth_from = depth_from,
            .depth_ab = depthAb(),
            .lone_cut = cut != .off,
            .draft_streams = shared and draftStreams(), // a shared round's drafts in one call: one pass for all
            .draft_pass = if (shared and draftStreams()) draftPass() else null,
            // the shared cut needs the context prices, as the lone cut does (lanes/config.zig lone_cut): with
            // TF_INK2_DRAFT_DEPTH=0 an eager pass reads no chances (`nextCut`) and the chances' map is for shallow
            // drafts, so shared rounds are v0.2's whether drafts are deferred or not (cuda_dspark.zig sharedCutResolved)
            .shared_cut = if (cut == .off or !depthPriced()) .off else sharedCut(),
            .hidden_rows = shared,
            .batch_rows = batch_rows,
            .max_streams = batch_streams,
            .shared_costs = if (shared) &shared_costs else &.{},
        };
        return .{
            .copy_drafts = false,
            .lone_copies = false,
            .root_copies = false,
            .context_copies = false,
            .exact_width = if (shared) 2 else 1,
            .gpu_tokens = false,
            .mtp = false,
            .speculate = false,
            .speculate_early = false,
            .drafts = 0,
            .draft_streams = false,
            .hidden_rows = shared,
            .batch_rows = batch_rows,
            .max_streams = batch_streams,
            .shared_costs = if (shared) &shared_costs else &.{},
        };
    }

    /// The widest window `verify` takes (the server's row limit).
    pub fn widest(self: *const Lanes) u32 {
        return if (self.x.e.drafter != null) ic.state.spec_rows else self.facts().exact_width;
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn freeSlot(self: *const Lanes) u32 {
        var slot: u32 = 0;
        while (true) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
    }

    /// The row's draw on the host: greedy, or the stream's sampling keyed at `position` over the whole vocabulary
    /// (sample.zig: the lane core's keyed sampler, its temperature floor, exact top_k and top_p). TF_INK2_SCRIPT
    /// (debug): a stream's k-th draw is the script's k-th token while it lasts (the forwards still run; for the text
    /// layer's checks).
    fn draw(self: *Lanes, logits: []const f32, s: *const lanes.Stream, position: u64) !u32 {
        const k = position -| s.prompt_len;
        if (k < self.script.len) return self.script[k];
        const v = s.sampling orelse return lanes.gpu_rule.argmax(logits);
        if (!(v.temperature > 0)) return lanes.gpu_rule.argmax(logits);
        if (self.sampler == null) self.sampler = .{ .gpa = self.gpa };
        return self.sampler.?.draw(logits, v, position);
    }

    /// Whether the row at `position` of `s` is drawn greedily with nothing else read from its logits: no script token
    /// there, no logprobs, greedy sampling. A forward whose every head row is such a row needs only the device's
    /// argmax rows (Engine.want_logits, B2 C4); a grammar's row among them brings its logits down only when its mask
    /// refuses the argmax (`drawAt`).
    fn greedyRow(self: *const Lanes, s: *const lanes.Stream, position: u64) bool {
        if (s.logprobs != null or s.prompt_logprobs != null) return false;
        return self.greedyDraw(s, position);
    }

    /// Whether the draw at `position` of `s` is greedy (no script token there, greedy sampling).
    fn greedyDraw(self: *const Lanes, s: *const lanes.Stream, position: u64) bool {
        if (position -| s.prompt_len < self.script.len) return false;
        const v = s.sampling orelse return true;
        return !(v.temperature > 0);
    }

    /// The draw of head row `r` of the last forward at `position` of `s` (`mask`: the grammar's allowed tokens there;
    /// empty: none): a greedy row from the device's argmax (each the host argmax of its row: B2 C4's op check), else
    /// `draw` over the row's logits, under the mask when there is one (`drawMasked`).
    fn drawAt(self: *Lanes, r: usize, s: *const lanes.Stream, position: u64, mask: []const u32) !u32 {
        const t = try self.drawRow(r, s, position, mask);
        if (mask.len > 0 and self.dumping()) {
            const dev: ?u32 = if (self.x.e.argmax_on) (try self.x.e.argmaxRows())[r] else null;
            const refused = if (dev) |d| !lanes.grammar.allows(mask, d) else false;
            const flags = @as(u32, @intFromBool(refused)) | @as(u32, @intFromBool(!self.greedyDraw(s, position))) << 1 | self.dump_flags;
            self.dumpRow(try self.x.e.fetchRow(r), mask, s, position, dev, t, flags);
        }
        return t;
    }

    fn drawRow(self: *Lanes, r: usize, s: *const lanes.Stream, position: u64, mask: []const u32) !u32 {
        if (self.greedyRow(s, position) and self.x.e.argmax_on) {
            const t = (try self.x.e.argmaxRows())[r];
            if (mask.len == 0 or lanes.grammar.allows(mask, t)) {
                if (mask.len > 0) self.masked_rows += 1;
                return t;
            }
            // the row's best is not allowed: its logits (still on the device), and the largest allowed value
            self.masked_misses += 1;
            return self.drawMasked(try self.x.e.fetchRow(r), mask, s, position);
        }
        const row = try self.x.e.logitsRow(r);
        if (mask.len == 0) return self.draw(row, s, position);
        return self.drawMasked(row, mask, s, position);
    }

    /// The row's draw under a grammar mask: `draw`'s rule over the row with every token the mask does not allow at
    /// -inf (ids past the mask's words too). Greedy: the first largest allowed value, read in place (`argmaxMasked`);
    /// sampled: the stream's keyed sampler over a masked copy of the row.
    fn drawMasked(self: *Lanes, logits: []const f32, mask: []const u32, s: *const lanes.Stream, position: u64) !u32 {
        const t0 = self.now();
        defer {
            self.masked_rows += 1;
            self.masked_ns += @intCast(@max(0, self.now() - t0));
        }
        if (self.greedyDraw(s, position)) return argmaxMasked(logits, mask);
        if (self.masked.len < logits.len) {
            if (self.masked.len > 0) self.gpa.free(self.masked);
            self.masked = &.{};
            self.masked = try self.gpa.alloc(f32, logits.len);
        }
        const row = self.masked[0..logits.len];
        try maskRow(row, logits, mask);
        return self.draw(row, s, position);
    }

    /// TF_INK2_MASK_DUMP=PATH (rank 0, debug) is set: masked draws are recorded (`dumpRow`).
    fn dumping(self: *Lanes) bool {
        if (self.dump_state == 0) {
            self.dump_state = 1;
            if (std.c.getenv("TF_INK2_MASK_DUMP") != null) {
                self.dump_state = 2;
                self.dump_left = 4096;
                if (std.c.getenv("TF_INK2_MASK_DUMP_ROWS")) |v| self.dump_left = std.fmt.parseInt(u64, std.mem.span(v), 10) catch self.dump_left;
                std.log.warn("inkling2: TF_INK2_MASK_DUMP: the first {d} masked draws' rows and masks are appended to {s} (debug)", .{ self.dump_left, std.mem.span(std.c.getenv("TF_INK2_MASK_DUMP").?) });
            }
        }
        return self.dump_state == 2 and self.dump_left > 0;
    }

    /// One masked draw appended to TF_INK2_MASK_DUMP (little-endian): u32 magic 0x444d4654 ("TFMD"), version 1,
    /// position (u64), the stream's prompt length (u64), the device argmax (0xffffffff: none), the pick, the row's
    /// width U, the mask's words W, flags (1: the mask refused the device argmax, 2: sampled, 4: the first token from
    /// the prompt's kept row, 8: a late window's row), 0; then the W mask words (u32) and the U logits (f32) the draw
    /// read. Errors are logged, never returned (a debug aid).
    fn dumpRow(self: *Lanes, logits: []const f32, mask: []const u32, s: *const lanes.Stream, position: u64, dev: ?u32, pick: u32, flags: u32) void {
        self.dump_left -= 1;
        const path = std.c.getenv("TF_INK2_MASK_DUMP") orelse return;
        const f = std.c.fopen(path, "ab") orelse {
            std.log.err("inkling2: TF_INK2_MASK_DUMP: cannot open {s}", .{std.mem.span(path)});
            return;
        };
        defer _ = std.c.fclose(f);
        const pl: u64 = s.prompt_len;
        const head = [_]u32{ 0x444d4654, 1, @truncate(position), @truncate(position >> 32), @truncate(pl), @truncate(pl >> 32), dev orelse 0xffff_ffff, pick, @intCast(logits.len), @intCast(mask.len), flags, 0 };
        const hb = std.mem.asBytes(&head);
        var n = std.c.fwrite(hb, 1, hb.len, f);
        n += std.c.fwrite(std.mem.sliceAsBytes(mask).ptr, 1, mask.len * 4, f);
        n += std.c.fwrite(std.mem.sliceAsBytes(logits).ptr, 1, logits.len * 4, f);
        if (n != hb.len + mask.len * 4 + logits.len * 4) std.log.err("inkling2: TF_INK2_MASK_DUMP: short write", .{});
    }

    /// Whether a new stream's held chains are cut (TF_INK2_DRAFT_CUT); a stream opening again keeps its choice.
    fn cutting(self: *Lanes, s: *const lanes.Stream, again: bool) bool {
        if (again) if (self.lanes.getPtr(s)) |l| return l.cut;
        switch (draftCut()) {
            .off => return false,
            .on => return true,
            .ab => {
                const on = self.cut_streams % 2 == 1;
                std.log.info("inkling2: draft cut a/b: stream {d} (prompt {d} tokens) {s}", .{ self.cut_streams, s.prompt().len, if (on) "cut" else "whole" });
                self.cut_streams += 1;
                return on;
            },
        }
    }

    /// A prompt begins: the stream's slot, and on the KV page pool its pages (`pagesFor`). An image prompt (rank 0's
    /// vision tower) is checked first (`checkImages`) and reserves its images' rows in the row store; LaneMemory (the
    /// open requests hold the rows) before anything moves.
    fn opening(self: *Lanes, s: *lanes.Stream) !*Lane {
        const ids = s.prompt();
        if (ids.len == 0) return error.EmptyPrompt;
        const vision = if (s.images.len > 0) self.x.e.vision orelse return error.NoVision else null;
        if (ids.len + s.max_new + 1 > self.x.e.capacity()) return error.PromptTooLong;
        var imgs: Images = .{};
        const img_tokens = if (vision) |v| try checkImages(s.images, ids.len, v.limits, &imgs) else 0;
        // its slot, or the lowest one no other stream holds
        const again = self.lanes.getPtr(s) != null;
        const slot = if (self.lanes.getPtr(s)) |l| had: {
            l.dropRow(self.gpa);
            self.dropImages(l); // its last pass's pins and rows
            break :had l.slot;
        } else self.freeSlot();
        // every image's rows: where the pass starts (a kept state) is known only below, then the rows of the images
        // it skips go back
        var reserved: u32 = 0;
        if (vision) |v| {
            try v.store.reserve(img_tokens);
            reserved = img_tokens;
        }
        errdefer if (reserved > 0) vision.?.store.unreserve(reserved);
        try self.lanes.put(self.gpa, s, .{ .slot = slot, .cut = self.cutting(s, again) });
        errdefer {
            _ = self.lanes.remove(s);
            self.x.e.unmapPages(slot);
        }
        s.cached = 0;
        // without the drafter no row past a window's first is rolled back (the convolutions keep their inputs for a
        // keep only with it), but a lone grammar stream's window may take the grammar's forced text as drafts, which
        // its masked draw can refuse (another spelling of the text): such a stream takes no drafts (forced tokens one
        // a round)
        if (self.x.e.drafter == null and s.grammar != null) s.drafts = false;
        const l = self.lanes.getPtr(s).?;
        s.cut_arm = l.cut and draftCut() == .ab; // `ab`: each arm's rounds timed apart from the other's
        // a stream not cut (TF_INK2_DRAFT_CUT=ab's whole arm) has no chances read (`probabilitiesFn`): the lane core
        // neither deepens its drafts nor prices them by chances (Stream.no_chances), so its rounds are v0.2's
        s.no_chances = !l.cut;
        // a kept state's pages adopted and its fixed part restored (the prompt cache on pages), or a longer state read
        // from the NVMe tier into fresh pages, else the slot's pages
        l.from = 0;
        var disk = false; // its pages opened by a read from the NVMe tier (never an image prompt's: `fromDisk`)
        if (s.prompt_logprobs == null) {
            if (try self.fromDisk(s, slot, ids, again)) |at| {
                l.from = at; // its pages are open (0: a rank refused the file; a cold pass on them)
                disk = true;
            } else l.from = try self.resumeFrom(s, slot, ids.len, again);
        }
        if (!disk and l.from == 0) try self.pagesFor(s, slot, ids.len, s.max_new, again, null);
        if (vision) |v| {
            // the images a kept prefix holds need no tower and no splice (Need.init's skip rule)
            imgs.need = try .init(imgs.images(), @intCast(l.from));
            const t = imgs.need.tokens(imgs.images());
            v.store.unreserve(img_tokens - t);
            imgs.need.reserved = t;
            reserved = 0;
            l.img = imgs;
        }
        return l;
    }

    /// The NVMe tier (cuda_nvme.zig): when it holds a prefix of the prompt at least `min_gain` tokens past the
    /// device cache's resume point, the slot opens on fresh pages and every rank reads its file into them; the pass
    /// starts there (or at 0 on the same pages when a rank refuses the file). Null: not from disk (nothing opened).
    fn fromDisk(self: *Lanes, s: *lanes.Stream, slot: u32, ids: []const u32, again: bool) !?usize {
        const e = self.x.e;
        if (!e.nvme.on or s.images.len > 0) return null;
        const ram: usize = if (s.reuse.saved != null) s.reuse.at else 0;
        const i = ic.nvme.pick(e, ids, ram) orelse return null;
        // fresh pages for the whole prompt; with a borrowed device state, only when they can be had (free, or from
        // states the reclaim may give back), so a disk attempt never gives back states for an open that then fails:
        // the device resume needs only the suffix's pages
        if (s.reuse.saved != null) if (e.pool) |p| {
            const n = ic.pages.need(ids.len, s.max_new, ic.state.spec_rows, p.entries);
            if (n > p.total() or !(p.fits(n) or p.fits(n -| self.reclaimable()))) return null;
        };
        self.pagesFor(s, slot, ids.len, s.max_new, again, null) catch |err| switch (err) {
            error.LaneMemory => if (s.reuse.saved != null) return null else return err,
            else => return err,
        };
        const ok = ic.nvme.load(self.x, slot, i, null) catch |err| switch (err) {
            error.NvmePages => false, // refused before its step (no rank moved): a cold pass on these pages
            else => return err,
        };
        if (!ok) return 0;
        s.cached = ic.nvme.entryAt(e, i);
        return s.cached;
    }

    /// The lane's image pass ends: its pins and its reserved rows go back to the row store (a tower job only it held
    /// is dropped), and the help it gave other images' towers is logged.
    fn dropImages(self: *Lanes, l: *Lane) void {
        if (l.img) |*p| {
            const v = self.x.e.vision.?;
            if (p.need.release(&v.store, p.images())) v.pace.job = null;
            if (p.help.units > 0) std.log.info("inkling2 rank 0: vision: a prompt pass ran {d} units ({d} layers) of other images' towers, {d} in steps the decode share did not bill; since start {d} such units, {d} unbilled", .{ p.help.units, p.help.layers, p.help.unbilled, v.pace.help_units, v.pace.unbilled_help });
            l.img = null;
        }
    }

    /// What an image prompt's step does before its chunk's forward: a tower unit (the chunk waits for a later call),
    /// nothing (`held`: a capped unit was due while the lane host held side work, Stream.side_held), or the chunk's
    /// splice (null: no image row in it).
    const Chunked = union(enum) { worked, held, ready: ?ic.embed.Splice };

    /// Rank 0, before an image prompt's chunk [at, at + n): the tower's units for the chunk's images the row store
    /// does not hold (a hit pins the stored rows; no tower), then the chunk's spans of stored rows.
    /// - `max_rows` 0 (chunks without a cap): each tower whole in this step, then the chunk.
    /// - else (--decode-share beside decoders) one unit a step, sized to about a capped chunk's time
    ///   (cuda_vision_store.zig unit): `worked`. Another image's tower in flight is helped to its end first. While
    ///   the lane host holds side work (decode time owed) no unit runs: `held`, unless every image of the chunk is
    ///   stored (a hit is served).
    fn imageChunk(self: *Lanes, s: *const lanes.Stream, p: *Images, at: usize, n: usize, max_rows: usize) !Chunked {
        var u: TowerUnits = .{ .e = self.x.e, .v = self.x.e.vision.?, .s = s, .list = p.images() };
        return self.imageChunkOn(s, p, at, n, max_rows, u.units());
    }

    fn imageChunkOn(self: *Lanes, s: *const lanes.Stream, p: *Images, at: usize, n: usize, max_rows: usize, t: vs.Units) !Chunked {
        const v = self.x.e.vision.?;
        const list = p.images();
        const before = p.need.pinned;
        var stored: @TypeOf(before) = .empty; // not pinned yet, and the store holds its rows: a hit if pinned now
        for (list, 0..) |*m, i| if (!before.isSet(i)) if (v.store.find(&m.digest)) |e| if (e.ready) stored.set(i);
        const cap: u32 = @intCast(@min(max_rows, std.math.maxInt(u32)));
        if (cap > 0 and s.side_held and !p.need.stored(&v.store, list, @intCast(at), @intCast(n))) return .held;
        // billed: a step the lane host times, or a capped unit (`worked`: side work it bills, lane_host_fill.zig sided)
        const r = vs.unit(&p.need, &v.store, &v.pace, list, @intCast(at), @intCast(n), cap, s.settle_step or cap > 0, t, &p.help) catch |err| {
            self.dropJob(v);
            return err;
        };
        if (r == .worked) return .worked;
        var hits = p.need.pinned.differenceWith(before).intersectWith(stored).iterator(.{});
        while (hits.next()) |i| {
            std.log.info("inkling2 rank 0: vision: image {d} of the prompt ({d} tokens at {d}) is in the row store (no tower)", .{ i, list[i].tokens, list[i].at });
        }
        var sp: ic.embed.Splice = .{};
        const k = try p.need.spans(&v.store, list, @intCast(at), @intCast(n), sp.spans[0..]);
        if (k == 0) return .{ .ready = null };
        sp.count = @intCast(k);
        sp.lo = sp.spans[0].first;
        sp.hi = sp.spans[k - 1].first + sp.spans[k - 1].n;
        return .{ .ready = sp };
    }

    /// A tower unit failed: the tower's state is unknown, so its job (the building entry) goes from the store and
    /// from every pass's pins; the passes that need the image begin it again.
    fn dropJob(self: *Lanes, v: *ic.vision.Rank0) void {
        const j = v.store.job() orelse return;
        if (v.pace.holds(&j.digest)) return;
        const d = j.digest;
        var it = self.lanes.valueIterator();
        while (it.next()) |l| if (l.img) |*p| {
            for (p.images(), 0..) |m, i| if (std.mem.eql(u8, &m.digest, &d)) p.need.pinned.unset(i);
        };
        v.store.abort(&d);
        std.log.warn("inkling2 rank 0: vision: a tower unit failed: its image's job is dropped from the row store and every pass", .{});
    }

    /// On the KV page pool (cuda_pages.zig): slot `slot` takes the pages of a prompt of `prompt` tokens, its reply and
    /// one verify window (one run of consecutive pages when one is free), and every rank maps them; `k`: a kept state
    /// it resumes, whose pages are the first of them (adopted) and the rest taken after them. Kept states give their
    /// pages back first when the pages do not fit (`reclaim`). LaneMemory: not now (the lane host queues the request
    /// again); RequestOverPool: never (refused by name).
    /// `again`: the stream opens again on its slot: the pages it holds serve (none a kept state holds), grown in place
    /// when it needs more; a new stream gives back whatever the slot still held first.
    fn pagesFor(self: *Lanes, s: *const lanes.Stream, slot: u32, prompt: usize, max_new: usize, again: bool, k: ?*const ic.snap.Kept) !void {
        const p = self.x.e.pool orelse return;
        if (!again) self.x.e.unmapPages(slot);
        const n = ic.pages.need(prompt, max_new, ic.state.spec_rows, p.entries);
        const adopt: []const u32 = if (k) |kk| kk.pages else &.{};
        if (n <= p.total()) self.reclaim(s, n -| adopt.len);
        _ = try self.x.openPages(slot, n, adopt);
    }

    /// The reclaim chain (`reclaimChain`): kept states give their pages back until `want` pages are available, or
    /// none is left to give; past the lane host's protection only when no other stream holds a lane (this stream's
    /// is in `lanes` already).
    fn reclaim(self: *Lanes, s: *const lanes.Stream, want: usize) void {
        const p = self.x.e.pool orelse return;
        const a = &(p.alloc orelse return);
        if (a.available() >= want) return;
        const hooks = self.store orelse return;
        const before = a.nkept;
        const r = reclaimChain(a, hooks, s.reuse.saved, want, self.lanes.count() <= 1);
        if (@import("builtin").is_test) return;
        std.log.info("inkling2 pages: the prompt cache gave back kept pages for a stream needing {d}: {d} kept pages before, {d} after, {d} available{s}{s}", .{ want, before, a.nkept, a.available(), if (r.unguarded) " (no other stream runs: states queued requests would resume given back too)" else "", if (r.ok) "" else " (not enough: the request waits)" });
    }

    /// The lane host's admission on the KV page pool: a slot is free and the request's pages are available now, or
    /// once kept states no resume borrows give theirs back (the reclaim chain at its open: guarded, then with no other
    /// stream running unguarded, so on an idle server the open frees every page counted here). A request larger than the
    /// whole pool is let through, so its open refuses it by name rather than it waiting.
    pub fn fits(self: *Lanes, prompt: usize, max_tokens: usize) bool {
        const p = self.x.e.pool orelse return true;
        const n = ic.pages.need(prompt, max_tokens, ic.state.spec_rows, p.entries);
        if (n > p.total()) return true;
        const slot_free = self.lanes.count() < p.slots;
        const ok = slot_free and (p.fits(n) or p.fits(n -| self.reclaimable()));
        defer p.show(); // /health's counts (B3): the pool and the wait as admission saw them
        if (ok) {
            p.waiting = null;
        } else if (slot_free and p.waiting != n) {
            // a new request waiting for pages (logged once, however often the lane host asks again)
            p.waiting = n;
            p.waits += 1;
            const a = p.alloc.?;
            std.log.info("inkling2 pages: a request needing {d} pages ({d} positions) waits for them: {d} of {d} pages available ({d} kept states'), {d} of {d} slots taken; waits since start {d}", .{ n, n * ic.pages.page_rows, a.available(), a.total(), a.nkept, self.lanes.count(), p.slots, p.waits });
        }
        return ok;
    }

    /// Pages the reclaim chain could free: those only kept states hold, but for states a resume borrows.
    fn reclaimable(self: *Lanes) usize {
        const p = self.x.e.pool orelse return 0;
        const hooks = self.store orelse return 0;
        var buf: [256][]const u32 = undefined;
        const protect = ic.snap.protectedPages(self.x.e, hooks.ctx, hooks.borrowed, &buf);
        return p.reclaimable(protect);
    }

    /// The stream's kept state resumed in its slot on every rank (the prompt cache on pages): its pages adopted
    /// (`pagesFor`) and its fixed part restored; where its pass starts. A state refused before any rank moved is the
    /// stream's alone: its pass starts at 0 (on fresh pages) and the store drops it. LaneMemory (its pages do not fit
    /// now) returns, so the request waits with its state.
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, slot: u32, len: usize, again: bool) !usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const ic.snap.Kept = @ptrCast(@alignCast(saved));
        const refused: anyerror = if (k.at == 0 or k.at >= len) error.SnapshotNotPrefix else if (k.at != s.reuse.at) error.SnapshotOutOfStep else if (!sameImages(s, k)) error.ImageDigestDiffers else ok: {
            self.pagesFor(s, slot, len, s.max_new, again, k) catch |err| switch (err) {
                error.LaneMemory, error.RequestOverPool => return err,
                else => break :ok err,
            };
            ic.snap.restore(self.x, slot, k.*, null) catch |err| break :ok err;
            s.cached = k.at;
            return k.at;
        };
        s.reuse_failed = true;
        s.reuse_error = refused;
        return 0;
    }

    /// Whether a kept state's prefix holds the stream's images there (the prompt cache's keys compare them already;
    /// this is the family's own check): zeros against zeros for text, with no hashing.
    fn sameImages(s: *const lanes.Stream, k: *const ic.snap.Kept) bool {
        const want = imageKey(s.images, k.at) catch return false;
        return std.mem.eql(u8, &k.images, &want);
    }

    /// The prompt pass passed `at` (a chunk's end, or a mark captured inside it): the store keeps the state there when
    /// it marked it.
    fn mark(s: *lanes.Stream, at: usize) void {
        const hook = s.reuse.hook orelse return;
        if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) == null) return;
        hook.at(hook.ptr, s, @intCast(at));
    }

    /// The stream's slot (the prompt cache's saves find it by the stream).
    pub fn slotOf(self: *const Lanes, s: *const lanes.Stream) ?u32 {
        return if (self.lanes.get(s)) |l| l.slot else null;
    }

    /// The chunk of the prompt from `at`; true when it ends the prompt (the first token drawn).
    fn chunk(self: *Lanes, s: *lanes.Stream, l: *Lane, at: usize, max_rows: usize) !bool {
        const ids = s.prompt();
        if (s.isCancelled()) return error.Cancelled; // decided before the step goes out: every rank runs what it is sent
        // the prompt cache's marks: those inside the chunk are captured by its forward (cuda_snap.zig, stage 1b; more
        // than max_caps end it at the next), or with TF_INK2_SNAP_SPLIT=1 every mark ends a chunk (stage 1); each
        // mark the forward passed is kept after it
        const want = @min(ids.len - at, if (max_rows == 0) engine.prompt_rows else @min(max_rows, engine.prompt_rows));
        var caps: [ic.snap.max_caps]u32 = undefined;
        const split = ic.snap.splitAtMarks() or self.x.e.snap.cap == null;
        const in: ic.snap.Inside = if (split) .{ .n = 0, .rows = ic.snap.rowsTo(s.reuse.marks, at, want) } else ic.snap.inside(s.reuse.marks, at, want, &caps);
        const n = in.rows;
        const last = at + n == ids.len;
        // rank 0's tower beside decoders: a settled chunk's time a row paces its units (host clock only)
        const t0: ?i96 = if (s.settle_step and self.x.e.vision != null) self.now() else null;
        try self.flush(); // a deferred draft reads its stream's taps: before this forward writes them
        // an image prompt: the towers of the chunk's images not stored yet, then its spliced rows (rank 0)
        const side0: i96 = if (l.img != null) self.now() else 0;
        const sp: ?ic.embed.Splice = if (l.img) |*p| switch (try self.imageChunk(s, p, at, n, max_rows)) {
            .ready => |x| x,
            .worked => {
                // a tower unit was this step's work (it ran a layer or more, or the merger, under a chunk row cap):
                // the chunk waits for the next call, which does not open the stream again. The unit ends on the
                // device and its time goes to the lane host as side work (Stream.side_ns), which bills it to
                // decoders and lets their round go before the prompt's next call, a short prompt's too
                l.fill = at;
                try self.x.e.stream.synchronize();
                s.side_ns = @intCast(@max(1, self.now() - side0));
                return false;
            },
            .held => {
                // decode time is owed (another prompt's unit or a long step): no unit, nothing on the device. The
                // lane host calls again once the decoders' round has paid it (Stream.side_wait)
                l.fill = at;
                s.side_wait = true;
                return false;
            },
        } else null;
        if (in.n > 0 and s.reuse.hook != null) try ic.snap.capture(self.x, l.slot, at, n, caps[0..in.n]);
        self.x.e.want_logits = !(last and self.greedyRow(s, ids.len) and s.grammar == null); // a grammar's row is kept
        defer self.x.e.want_logits = true;
        const hr: ?ic.follow.Rows = if (last) .{ .lo = n - 1, .n = 1 } else null;
        if (sp) |*x| try self.x.forwardImages(l.slot, at, ids[at..][0..n], hr, x) else try self.x.forward(l.slot, at, ids[at..][0..n], hr);
        // the images the chunk finished: their pins go (their rows stay stored until evicted); all of them at the end
        if (l.img) |*p| {
            const v = self.x.e.vision.?;
            if (p.need.passed(&v.store, p.images(), @intCast(at + n))) v.pace.job = null;
        }
        if (last) self.dropImages(l);
        if (!last) {
            keepMarks(s, at, n);
            l.fill = at + n;
            // the lane host bills this step's time to decoders (--decode-share): end it on the device, or its tail
            // lands in the next decode round's time and the share's debt sees only the launches
            if (s.settle_step) try self.x.e.stream.synchronize();
            if (t0) |t| self.x.e.vision.?.pace.chunked(@intCast(self.now() - t), n);
            return false;
        }
        if (s.grammar != null) {
            // structured output: the row is kept until the lane core asks for the first token, under its mask
            // (`first_masked`) or without one (`first`)
            l.dropRow(self.gpa);
            l.first_row = try self.gpa.dupe(f32, try self.x.e.logitsRow(0));
            l.first = null;
        } else l.first = self.take(try self.drawAt(0, s, ids.len, &.{}));
        l.fill = null;
        keepMarks(s, at, n);
        return true;
    }

    /// The store keeps the states at its marks in (at, at + n], the rows the last forward ran.
    fn keepMarks(s: *lanes.Stream, at: usize, n: usize) void {
        for (s.reuse.marks) |m| if (m > at and m <= at + n) mark(s, m);
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const l = try self.opening(s);
        errdefer self.dropLane(s);
        var at: usize = l.from;
        while (!try self.chunk(s, l, at, 0)) at = l.fill.?;
    }

    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const l = if (going == null) try self.opening(s) else self.lanes.getPtr(s).?;
        errdefer self.dropLane(s); // a later pass of the stream opens it again (the lane core releases only a cancel)
        return self.chunk(s, l, going orelse l.from, max_rows);
    }

    /// A stream whose prompt pass failed leaves its slot, and its pages go back to the pool (`release` may not come).
    fn dropLane(self: *Lanes, s: *lanes.Stream) void {
        if (self.lanes.getPtr(s)) |l| {
            l.dropRow(self.gpa);
            self.dropImages(l);
            self.x.e.unmapPages(l.slot);
        }
        _ = self.lanes.remove(s);
    }

    fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const l = of(ptr).lanes.getPtr(s) orelse return;
        l.fill = null;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (l.first == null and l.first_row.len > 0) {
            // a grammar stream whose first token the grammar leaves free: the kept row's own draw
            defer l.dropRow(self.gpa);
            l.first = self.take(try self.draw(l.first_row, s, position));
        }
        return l.first orelse error.NoFirstToken;
    }

    /// The first token drawn among `mask`'s allowed tokens from the prompt's kept last row (a grammar stream's).
    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (l.first_row.len == 0) return error.NoFirstToken;
        defer l.dropRow(self.gpa);
        const t = try self.drawMasked(l.first_row, mask, s, position);
        if (self.dumping()) self.dumpRow(l.first_row, mask, s, position, null, t, 4 | @as(u32, @intFromBool(!self.greedyDraw(s, position))) << 1);
        l.first = self.take(t);
        return l.first.?;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// The stream's accepted rows sent to the ranks, if a keep is still pending (no draft came after it).
    fn settle(self: *Lanes, l: *Lane) !void {
        const n = l.kept orelse return;
        l.kept = null;
        try self.x.keep(l.slot, n);
    }

    /// Every deferred draft run now, in one pass (each its keep, then the blocks; their drafts to the host).
    fn flush(self: *Lanes) !void {
        return self.flushWhere(.all, false);
    }

    /// `flush` of every deferred draft, or of grammar streams' alone; `chances`: each stream's chances of the pass
    /// to its lane too (the lone cut: a lane's chances are always its own block's).
    fn flushWhere(self: *Lanes, which: enum { all, grammar }, chances: bool) !void {
        var items: [engine.max_segs]engine.DraftItem = undefined;
        var keeps: [engine.max_segs]usize = undefined;
        var outs: [engine.max_segs]*[ic.dspark.drafts]u32 = undefined;
        var lns: [engine.max_segs]*Lane = undefined;
        var m: usize = 0;
        var it = self.lanes.valueIterator();
        while (it.next()) |l| {
            const p = l.pending orelse continue;
            if (which == .grammar and p.stream.grammar == null) continue;
            l.pending = null;
            self.stale(l.slot, p.stream);
            items[m] = .{ .slot = l.slot, .anchor = p.anchor };
            keeps[m] = p.n;
            outs[m] = &l.held;
            lns[m] = l;
            m += 1;
            if (m == engine.max_segs) {
                try deferredPass(&self.x, self.x.e, items[0..m], keeps[0..m], outs[0..m], lns[0..m], chances);
                m = 0;
            }
        }
        if (m > 0) try deferredPass(&self.x, self.x.e, items[0..m], keeps[0..m], outs[0..m], lns[0..m], chances);
    }

    /// One pass of `flushWhere`'s deferred drafts (one stream: `draft`, several: `draftMany`; `x` the leader, `e` its
    /// engine), and with `chances` the pass's chances to its lanes, in its order, only when one of them is cut: a pass
    /// of lanes none of which is cut (TF_INK2_DRAFT_CUT=ab's whole arm) has no use for them, so no wait and no download,
    /// as an eager pass of them (`eagerPass`).
    fn deferredPass(x: anytype, e: anytype, items: []const engine.DraftItem, keeps: []const usize, outs: []const *[ic.dspark.drafts]u32, lns: []const *Lane, chances: bool) !void {
        if (lns.len == 1) try x.draft(items[0].slot, keeps[0], items[0].anchor, outs[0]) else try x.draftMany(items, keeps, outs);
        if (!chances) return;
        for (lns) |l| if (l.cut) return readChances(e, lns);
    }

    /// The last drafter pass's chances, its streams' in its order, each to its own lane (`e`: the engine).
    fn readChances(e: anytype, lns: []const *Lane) !void {
        var outs: [engine.max_segs]*[ic.dspark.drafts]f32 = undefined;
        for (outs[0..lns.len], lns) |*o, l| o.* = &l.chances;
        try e.chances(outs[0..lns.len]);
        for (lns) |l| l.chances_ok = true;
    }

    /// An eager drafter pass (TF_INK2_DRAFT_DEFER=0) of `lns`' blocks, `streams` theirs (one stream: `draft`,
    /// several: `draftMany`; `x` the leader, `e` its engine). When one of its streams can use its chances at its next
    /// window (`nextCut`), the pass's chances go to its lanes now, in the pass's order, before another pass
    /// overwrites them: the lone cut reads them there (`probabilitiesFn`). Otherwise none are read (no wait).
    fn eagerPass(x: anytype, e: anytype, items: []const engine.DraftItem, keeps: []const usize, lns: []const *Lane, streams: []const *const lanes.Stream) !void {
        var outs: [engine.max_segs]*[ic.dspark.drafts]u32 = undefined;
        for (outs[0..lns.len], lns) |*o, l| o.* = &l.held;
        if (lns.len == 1) try x.draft(items[0].slot, keeps[0], items[0].anchor, outs[0]) else try x.draftMany(items, keeps, outs[0..lns.len]);
        for (lns, streams) |l, s| if (nextCut(l, s)) return readChances(e, lns);
    }

    /// Whether the lone cut can take a lane's chances at its stream's next window (lanes/windows.zig cutLone): the cut
    /// on (it needs the context prices: not with TF_INK2_DRAFT_DEPTH=0), a stream in the cut arm, greedy, without a
    /// grammar, at or below `depth_from` (its context is the same at its next window as when its draft runs).
    fn nextCut(l: *const Lane, s: *const lanes.Stream) bool {
        return l.cut and depthPriced() and s.sampling == null and s.grammar == null and s.cache_len <= depth_from;
    }

    /// A draft about to run from taps another forward wrote since its stream's last one (it changes no output, only
    /// what it proposes): counted.
    fn stale(self: *Lanes, slot: u32, s: *lanes.Stream) void {
        const st = self.x.e.slots.items[slot] orelse return;
        if (st.last_fwd == self.x.e.forwards) return;
        s.stale_drafts += 1;
        self.x.e.stale_drafts += 1;
    }

    /// Every window in one forward (a segment each, in slot order: the captured forward's key is the same whatever
    /// order the lane core sends the streams in, and every op is row-invariant, so a stream's bits do not depend on
    /// where its rows sit): the pending row, then its held drafts (or host drafts); every row's draw.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (self.draft_end) |t| {
            // the host gap: from the last draft's return to this verify (each drafted window's stream counts it)
            const gap: u64 = @intCast(@max(0, self.now() - t));
            self.draft_end = null;
            self.gaps += 1;
            self.gap_ns += gap;
            for (windows) |w| if (w.held > 0) {
                w.stream.draft_gap_ns += gap;
                w.stream.draft_gaps += 1;
            };
        }
        if (windows.len > engine.max_segs) return error.TooManyWindows;
        // deferred drafts of streams this forward does not verify run first (it overwrites the taps they read)
        {
            var it = self.lanes.iterator();
            const others = while (it.next()) |kv| {
                if (kv.value_ptr.pending == null) continue;
                for (windows) |w| {
                    if (w.stream == kv.key_ptr.*) break;
                } else break true;
            } else false;
            if (others) try self.flushWhere(.all, leftOutChances(self.round_open, draftCut(), sharedCut())); // shared_cut: their chances for their next window, in a C2 round only
        }
        // the windows in slot order (ord[i]: the window of segment i); row0[j]: window j's first row in the forward
        var ord: [engine.max_segs]usize = undefined;
        var row0: [engine.max_segs]usize = undefined;
        {
            var slot_of: [engine.max_segs]u32 = undefined;
            for (windows, 0..) |w, j| {
                slot_of[j] = (self.lanes.getPtr(w.stream) orelse return error.UnknownStream).slot;
                var k = j;
                while (k > 0 and slot_of[ord[k - 1]] > slot_of[j]) : (k -= 1) ord[k] = ord[k - 1];
                ord[k] = j;
            }
        }
        var segs: [engine.max_segs]engine.Segment = undefined;
        var ids: [engine.max_segs * ic.state.spec_rows]u32 = undefined;
        var keeps: [engine.max_segs]ic.follow.Leader.Keep = @splat(.{});
        var outs: [engine.max_segs]*[ic.dspark.drafts]u32 = undefined;
        var m: usize = 0; // windows whose drafts their own step makes (in segment order, as the step's walk is)
        var R: usize = 0;
        for (ord[0..windows.len], 0..) |wj, i| {
            const w = windows[wj];
            const rows = w.rows();
            if (w.parents != null) return error.TreesNotServed;
            if (rows > self.widest()) return error.DraftsNotServed;
            _ = try maskWords(w, self.x.e.vocab());
            const l = self.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            try self.settle(l);
            const st = self.x.e.slots.items[l.slot] orelse return error.UnknownStream;
            // a deferred draft keeps its rows first: the window starts past them
            const pos = if (l.pending) |p| st.last_pos0 + p.n else self.x.e.pos(l.slot);
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            if (pos + rows > self.x.e.capacity()) return error.PastContext;
            ids[R] = w.pending;
            if (l.pending) |p| {
                if (p.anchor != w.pending) return error.AnchorMismatch;
                keeps[i] = .{ .n = p.n, .draft = w.held > 0 and w.tokens.len == 0 };
                if (keeps[i].draft) {
                    self.stale(l.slot, p.stream);
                    @memset(ids[R + 1 ..][0..w.held], 0); // made on the device in the step
                    outs[m] = &l.held;
                    m += 1;
                } else @memcpy(ids[R + 1 ..][0..w.tokens.len], w.tokens);
            } else {
                const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
                @memcpy(ids[R + 1 ..][0..drafts.len], drafts);
            }
            segs[i] = .{ .slot = l.slot, .pos0 = pos, .rows = rows };
            row0[wj] = R;
            R += rows;
        }
        if (R > engine.head_rows) return error.BadRows;
        var deferred = false;
        for (keeps[0..windows.len]) |k| deferred = deferred or k.n > 0 or k.draft;
        // the rows drawn greedily from the argmax alone bring down no logits (B2 C4); the others (sampled, logprobs,
        // a grammar's) bring down their own
        var need: u32 = 0;
        for (windows, row0[0..windows.len]) |w, r0| for (w.positions, 0..) |p, r| {
            if (!self.greedyRow(w.stream, p)) need |= @as(u32, 1) << @intCast(r0 + r);
        };
        self.x.e.want_logits = need != 0;
        self.x.e.want_rows = need;
        defer {
            self.x.e.want_logits = true;
            self.x.e.want_rows = engine.all_rows;
        }
        if (deferred) {
            try self.x.forwardDrafted(segs[0..windows.len], ids[0..R], keeps[0..windows.len]);
            for (windows) |w| self.lanes.getPtr(w.stream).?.pending = null;
            if (m > 0) try self.x.e.proposals(outs[0..m]); // the drafts the step made, after its head
        } else try self.x.forwardMany(segs[0..windows.len], ids[0..R], .{ .lo = 0, .n = R });
        const t0 = self.now();
        var late: std.heap.ArenaAllocator = .init(self.gpa);
        defer late.deinit();
        defer self.dump_flags = 0;
        for (windows, out, row0[0..windows.len]) |w, *o, r0| {
            const l = self.lanes.getPtr(w.stream).?;
            const rows = w.rows();
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            var masks = w.masks;
            var masked = rows; // rows with a mask (when there are masks): from row 0
            if (w.late_masks) {
                // the grammar walks the pending token and the drafts the step made (on the host since `proposals`)
                if (w.masks.len > 0 or w.tokens.len > 0 or w.parents != null or w.held == 0) return error.LateWindowShape;
                const g = w.stream.grammar orelse return error.LateWindowShape;
                const chain = try late.allocator().alloc(u32, rows);
                chain[0] = w.pending;
                @memcpy(chain[1..], drafts);
                const gw = try g.late(late.allocator(), chain);
                masks = gw.masks;
                masked = gw.rows.len;
                self.late_windows += 1;
                self.late_free_rows += rows - masked;
            }
            const words = if (masks.len == 0) 0 else masks.len / masked;
            if (words > 0 and (masks.len != masked * words or words * 32 < self.x.e.vocab())) return error.InvalidGrammarMasks;
            self.dump_flags = if (w.late_masks) 8 else 0;
            for (0..rows) |r| o.sampled[r] = try self.drawAt(r0 + r, w.stream, w.positions[r], if (words == 0 or r >= masked) &.{} else masks[r * words ..][0..words]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.held_n = 0;
            l.chances_ok = false;
        }
        self.draw_rounds += 1;
        self.draw_rows += R;
        self.draw_ns += @intCast(@max(0, self.now() - t0));
    }

    /// The accepted rows of each window (a chain: rows 0 .. n - 1), sent with the stream's next draft or window.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotServed;
            const l = self.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (self.x.e.drafter == null) {
                if (path.len != w.rows()) return error.RollbackNotServed;
                continue; // windows of one row: nothing to keep
            }
            if (l.pending) |p| {
                if (p.n != path.len) return error.KeepMismatch;
                continue; // the deferred draft keeps them
            }
            l.kept = path.len;
        }
    }

    /// The drafter's block for each stream from its accepted rows (or its prompt's last row): the anchor is the token
    /// after the last kept row; the drafts are held for the stream's next window.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (self.x.e.drafter == null) return error.NoDraftHead;
        var drafted = false;
        defer if (drafted) {
            self.draft_end = self.now();
        };
        if (requests.len > engine.max_segs) return error.TooManyWindows;
        var d: Dispatch = .{ .how = if (deferDrafts()) .deferred else if (!draftStreams()) .each else .one };
        for (requests) |r| {
            if (r.lanes != null) return error.TreesNotServed;
            const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
            const slot_state = self.x.e.slots.items[l.slot] orelse return error.UnknownStream;
            const anchor, const n = if (r.rows) |rows| a: {
                for (rows, 0..) |row, j| if (row != j) return error.TreesNotServed;
                if (rows.len == 0 or r.follow.len < rows.len) return error.EmptyPath;
                break :a .{ r.follow[rows.len - 1], rows.len };
            } else .{ self.value(r.first orelse return error.NoFirstToken), slot_state.last_rows };
            if (r.position != slot_state.last_pos0 + n + 1) return error.PositionMismatch;
            l.kept = null;
            if (r.depth == 0) {
                try self.x.keep(l.slot, n);
                l.held_n = 0;
                continue;
            }
            if (slot_state.last_pos0 + n + ic.dspark.block > self.x.e.capacity()) {
                try self.x.keep(l.slot, n); // too near the end of the context for a block: plain rounds
                l.held_n = 0;
                continue;
            }
            if (l.pending != null) try self.flush(); // (not expected: a verify consumes it)
            if (d.how != .deferred) self.stale(l.slot, r.stream); // deferred: counted where it runs
            try d.add(self.x, self.x.e, l, .{ .slot = l.slot, .anchor = anchor }, n, r.stream, r.depth);
            drafted = true;
            self.drafts_made += 1;
        }
        try d.finish(self.x, self.x.e);
    }

    /// `draftFn`'s blocks (`x` the leader, `e` its engine): deferred to each stream's next verify
    /// (TF_INK2_DRAFT_DEFER), else eager: a pass a stream (TF_INK2_DRAFT_STREAMS=0) or one pass of every drafting
    /// stream after the last (`finish`: one header). A lane's chances are its old block's until its draft: they stop
    /// counting before it runs, so that an eager pass's read (`eagerPass`) is what holds after it.
    const Dispatch = struct {
        how: enum { deferred, each, one },
        items: [engine.max_segs]engine.DraftItem = undefined,
        keeps: [engine.max_segs]usize = undefined,
        lns: [engine.max_segs]*Lane = undefined,
        streams: [engine.max_segs]*const lanes.Stream = undefined,
        m: usize = 0,

        fn add(d: *Dispatch, x: anytype, e: anytype, l: *Lane, item: engine.DraftItem, n: usize, s: *lanes.Stream, depth: u32) !void {
            l.chances_ok = false;
            switch (d.how) {
                .deferred => l.pending = .{ .n = n, .anchor = item.anchor, .stream = s },
                .each => try eagerPass(x, e, &.{item}, &.{n}, &.{l}, &.{s}),
                .one => {
                    d.items[d.m] = item;
                    d.keeps[d.m] = n;
                    d.lns[d.m] = l;
                    d.streams[d.m] = s;
                    d.m += 1;
                },
            }
            l.held_n = @min(depth, ic.dspark.drafts);
        }

        fn finish(d: *const Dispatch, x: anytype, e: anytype) !void {
            if (d.m > 0) try eagerPass(x, e, d.items[0..d.m], d.keeps[0..d.m], d.lns[0..d.m], d.streams[0..d.m]);
        }
    };

    /// The host tail since start (one log line): the heads' downloads (the logits, or the argmax rows alone) and the
    /// verify rounds' draw loops, ms each and rows on average.
    fn tail(self: *const Lanes) void {
        const c = self.x.e.counts;
        const per = struct {
            fn ms(ns: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(n));
            }
            fn rows(r: u64, n: u64) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(r)) / @as(f64, @floatFromInt(n));
            }
        };
        std.log.info("inkling2: head tail since start: {d} heads with logits {d:.3} ms ({d:.2} rows), {d} argmax only {d:.3} ms ({d:.2} rows); {d} verify draws {d:.3} ms ({d:.2} rows); {d} masked rows ({d} brought down for a refused argmax), {d:.3} ms each; {d} late-masked windows ({d} rows past the grammar's)", .{ c.heads[1], per.ms(c.head_ns[1], c.heads[1]), per.rows(c.head_rows[1], c.heads[1]), c.heads[0], per.ms(c.head_ns[0], c.heads[0]), per.rows(c.head_rows[0], c.heads[0]), self.draw_rounds, per.ms(self.draw_ns, self.draw_rounds), per.rows(self.draw_rows, self.draw_rounds), self.masked_rows, self.masked_misses, per.ms(self.masked_ns, self.masked_rows), self.late_windows, self.late_free_rows });
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The drafter's block for the stream's next window, as held (the lane core walks a grammar stream's drafts on
    /// the host before its window). A grammar stream's deferred draft runs now, with every other grammar stream's
    /// (their walks come next); the other streams' deferred drafts still run in their verify's step.
    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const self = of(ptr);
        const l0 = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (l0.pending != null) try self.flushWhere(if (s.grammar != null) .grammar else .all, false);
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.held_n) return error.NoHeldDrafts;
        @memcpy(out, l.held[0..out.len]);
    }

    /// The lone cut (lanes/windows.zig cutLone): the chance that each of the first `out.len` held drafts lands with
    /// every draft before it, from the drafter's pick probabilities of the stream's own block through `cut_map`. A
    /// deferred draft runs now (with any other's), its chances read with its drafts; false for a stream not cut or a
    /// block whose chances were not read (its draft ran in a verify's step).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l0 = self.lanes.getPtr(s) orelse return false;
        if (!l0.cut or out.len == 0 or out.len > ic.dspark.drafts) return false;
        if (l0.pending != null) try self.flushWhere(.all, true);
        const l = self.lanes.getPtr(s) orelse return false;
        return chainChances(l, out);
    }

    /// The NVMe tier's write-behind waits (lane host: while no stream is live and nothing waits).
    fn idlePendingFn(ptr: *anyopaque) bool {
        return ic.nvme.pending(of(ptr).x.e);
    }

    /// One unit of it (a begin, a chunk or a commit, on every rank); the other ranks sleep again when none is left.
    fn idleFn(ptr: *anyopaque) void {
        const self = of(ptr);
        if (!self.idle_seen) std.log.info("inkling2: NVMe prompt cache: the write-behind's first idle turn", .{});
        self.idle_seen = true;
        _ = ic.nvme.step(self.x) catch |err| std.log.err("inkling2: NVMe prompt cache: a write step ({s})", .{@errorName(err)});
        if (!ic.nvme.pending(self.x.e) and self.lanes.count() == 0) self.x.idle() catch |err| std.log.err("inkling2: the other ranks' idle step: {s}", .{@errorName(err)});
    }

    /// The stream left: its slot is free for the next stream (its state stays allocated on every rank).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.lanes.getPtr(s)) |l| {
            l.dropRow(self.gpa);
            self.dropImages(l); // a pass cancelled before its end still holds its images' pins and rows
            self.x.e.unmapPages(l.slot); // its pages back to the pool (rank 0's; nothing is sent)
        }
        _ = self.lanes.remove(s);
        if (self.lanes.count() == 0 and self.drafts_made > 0)
            std.log.info("inkling2: drafts since start {d}, stale {d}; host gap from a draft to the next verify {d:.3} ms (mean of {d})", .{ self.drafts_made, self.x.e.stale_drafts, if (self.gaps > 0) @as(f64, @floatFromInt(self.gap_ns)) / 1e6 / @as(f64, @floatFromInt(self.gaps)) else 0, self.gaps });
        if (self.lanes.count() == 0) self.tail();
        if (self.lanes.count() == 0) self.x.idle() catch |err| std.log.err("inkling2: the other ranks' idle step: {s}", .{@errorName(err)});
    }
};

/// A window's mask words a row (0: no masks), refused before its forward goes out: a mask a row, each covering the
/// logits' `vocab` columns.
fn maskWords(w: be.Window, vocab: usize) !usize {
    if (w.masks.len == 0) return 0;
    const rows = w.rows();
    if (w.masks.len % rows != 0) return error.InvalidGrammarMasks;
    const words = w.masks.len / rows;
    if (words * 32 < vocab) return error.InvalidGrammarMasks;
    return words;
}

/// The first largest value among `mask`'s allowed tokens: `argmax` of the row with the others at -inf (maskRow).
fn argmaxMasked(logits: []const f32, mask: []const u32) !u32 {
    if (mask.len * 32 < logits.len) return error.InvalidGrammarMasks;
    var best: ?usize = null;
    for (mask, 0..) |bits, w| {
        var b = bits;
        while (b != 0) : (b &= b - 1) {
            const i = w * 32 + @ctz(b);
            if (i >= logits.len) break;
            if (best == null or logits[i] > logits[best.?]) best = i;
        }
    }
    return @intCast(best orelse return error.NoAllowedToken);
}

/// `out` = `logits` with every token `mask` does not allow at -inf; error.NoAllowedToken when it allows none of them.
fn maskRow(out: []f32, logits: []const f32, mask: []const u32) !void {
    if (mask.len * 32 < logits.len) return error.InvalidGrammarMasks;
    const ninf = -std.math.inf(f32);
    var any: u32 = 0;
    var i: usize = 0;
    for (mask) |bits| {
        if (i >= logits.len) break;
        const n = @min(32, logits.len - i);
        any |= bits & if (n == 32) std.math.maxInt(u32) else (@as(u32, 1) << @intCast(n)) - 1;
        for (0..n) |b| out[i + b] = if (bits >> @intCast(b) & 1 == 1) logits[i + b] else ninf;
        i += n;
    }
    if (any == 0) return error.NoAllowedToken;
}

test "inkling2 lanes: facts serve no drafts and share rounds between streams" {
    var e: engine.Engine = undefined;
    e.drafter = null;
    var l: Lanes = .{ .x = .{ .e = &e }, .gpa = std.testing.allocator };
    const f = l.facts();
    try std.testing.expect(!f.copy_drafts and !f.mtp and f.drafts == 0);
    // the lane core's shared rounds: width 2 or more, streams exact, several streams' rows in one forward
    try std.testing.expect(f.exact_width >= 2 and f.streams_exact and f.hidden_rows and f.max_streams == engine.max_segs);
    try std.testing.expect(f.batch_rows * 1 <= engine.head_rows and f.shared_costs.len > 0);
    var c = try lanes.Config.init(std.testing.allocator, f, l.widest(), l.widest() - 1);
    defer c.deinit(std.testing.allocator);
    try std.testing.expect(c.family_streams and c.batch_streams == engine.max_segs and !c.family_mtp and c.max_copy == 0);
}

test "inkling2 lanes: with the drafter, windows of the anchor and 7 drafts, priced and primed" {
    try std.testing.expectEqual(@as(usize, ic.state.spec_rows), tp4_costs.len);
    try std.testing.expectEqual(@as(usize, ic.dspark.drafts), draft_prior.len);
    for (tp4_costs[1..], tp4_costs[0 .. tp4_costs.len - 1]) |a, b| try std.testing.expect(a.ms > b.ms and a.width == b.width + 1);
    for (shared_costs[1..], shared_costs[0 .. shared_costs.len - 1]) |a, b| try std.testing.expect(a.ms > b.ms and a.width > b.width);
    // the draft is priced once a round: two rows cost the draft more than one, then only the window's own rows
    try std.testing.expect(tp4_costs[1].ms - tp4_costs[0].ms >= tp4_draft_ms);
    try std.testing.expectApproxEqAbs(tp4_window_ms[7] - tp4_window_ms[1], tp4_costs[7].ms - tp4_costs[1].ms, 1e-9);
    // a shared round's widest forward: every stream's widest window, within the head's rows
    try std.testing.expect(batch_rows <= engine.head_rows and batch_rows >= ic.state.spec_rows);
    // the context prices: ascending, every window width, the first node today's lone prices; from 32k on every
    // window and the draft dearer with depth (below that the nodes are within run-to-run noise of each other)
    try std.testing.expectEqualSlices(f64, &tp4_window_ms, depth_costs[0].window_ms);
    try std.testing.expectEqual(tp4_draft_ms, depth_costs[0].draft_ms);
    for (depth_costs[1..], depth_costs[0 .. depth_costs.len - 1]) |a, b| {
        try std.testing.expect(a.context > b.context and a.window_ms.len == ic.state.spec_rows);
        if (b.context < 32_768) continue;
        try std.testing.expect(a.draft_ms > b.draft_ms);
        for (a.window_ms, b.window_ms) |x, y| try std.testing.expect(x > y);
    }
    for (0..ic.state.spec_rows) |d| try std.testing.expectApproxEqAbs(tp4_costs[d].ms, lanes.depth.priceAt(&depth_costs, @intCast(d), 600).?, 1e-9);
    // shared rounds price depth past a node: up to it a stream's round is the shallow price
    try std.testing.expect(for (depth_costs) |n| {
        if (n.context == depth_from) break true;
    } else false);
}

test "inkling2 lanes: TF_INK2_DRAFT_CUT=0 prices rounds as v0.2 did: at the 0.84 / 0.20 boundary it picks v0.2's width" {
    const gpa = std.testing.allocator;
    // v0.2's tables as it shipped them: the shallow windows and draft, and the context nodes
    const window = [_]f64{ 34.1, 42.4, 46.9, 51.7, 57.6, 60.2, 65.5, 67.3 };
    const nodes = [_]lanes.depth.ContextCost{
        .{ .context = 600, .window_ms = &window, .draft_ms = 8.4 },
        .{ .context = 3_040, .window_ms = &.{ 33.9, 41.1, 47.1, 50.9, 54.3, 61.4, 63.6, 68.2 }, .draft_ms = 8.5 },
        .{ .context = 32_768, .window_ms = &.{ 35.1, 42.2, 47.7, 52.6, 56.9, 65.4, 66.9, 72.2 }, .draft_ms = 9.4 },
        .{ .context = 131_072, .window_ms = &.{ 41.1, 48.1, 54.5, 62.2, 69.7, 73.5, 82.2, 90.3 }, .draft_ms = 16.1 },
        .{ .context = 262_144, .window_ms = &.{ 48.3, 56.1, 65.3, 74.7, 86.8, 91.3, 103.9, 115.9 }, .draft_ms = 23.0 },
        .{ .context = 524_000, .window_ms = &.{ 63.2, 72.0, 82.7, 99.9, 110.1, 121.7, 140.7, 150.3 }, .draft_ms = 39.7 },
    };
    var costs: [window.len]lanes.config.Cost = undefined;
    for (&costs, window, 0..) |*c, ms, i| c.* = .{ .width = @intCast(i + 1), .ms = ms + if (i > 0) @as(f64, 8.4) else 0 };
    var e: engine.Engine = undefined;
    e.drafter = @ptrFromInt(@alignOf(ic.dspark.Drafter)); // never read: facts only ask whether a drafter is loaded
    const l: Lanes = .{ .x = .{ .e = &e }, .gpa = gpa };
    const off = l.factsCut(.off);
    var base = off;
    base.window_costs = &costs;
    base.context_costs = &nodes;
    // the tables `0` installs are v0.2's, cell for cell
    try std.testing.expectEqualDeep(base.window_costs, off.window_costs);
    try std.testing.expectEqualDeep(base.context_costs, off.context_costs);
    try std.testing.expect(!off.lone_cut);
    // `0` plans shared rounds as v0.2 did too: no chances, no deepened drafts (whatever TF_INK2_SHARED_CUT says)
    try std.testing.expectEqual(lanes.config.SharedCut.off, off.shared_cut);
    // the cut on: shared rounds by TF_INK2_SHARED_CUT, `deepen` when unset
    if (std.c.getenv("TF_INK2_SHARED_CUT") == null) try std.testing.expectEqual(lanes.config.SharedCut.deepen, l.factsCut(.on).shared_cut);
    // a shallow greedy stream's round with room for two drafts, nothing timed yet and no probe, its first two drafts
    // landing at 0.84 and 0.20 given the ones before: 1.84 and 2.008 expected tokens; v0.2's 50.8 and 55.3 ms take
    // two drafts, the re-measured 46.8 and 51.2 ms one
    const width = struct {
        fn of(m: lanes.Model) !i64 {
            var cfg = try lanes.Config.init(gpa, m, ic.state.spec_rows, ic.state.spec_rows - 1);
            defer cfg.deinit(gpa);
            var lane_engine = lanes.Engine.init(gpa, &cfg, undefined, undefined);
            defer lane_engine.deinit();
            var st: ?lanes.depth.State = null;
            defer if (st) |*x| x.deinit(gpa);
            const who: lanes.depth.Who = .{ .state = &st, .draft_room = 3, .finished = false, .forced = false, .context = 600 };
            const p = try lane_engine.rule.rates(who);
            p[0] = 0.84;
            p[1] = 0.20;
            return lane_engine.rule.depth(who);
        }
    };
    try std.testing.expectEqual(@as(i64, 2), try width.of(base));
    try std.testing.expectEqual(try width.of(base), try width.of(off));
    try std.testing.expectEqual(@as(i64, 1), try width.of(l.factsCut(.on)));
    try std.testing.expectEqual(@as(i64, 1), try width.of(l.factsCut(.ab)));
}

test "inkling2 lanes: TF_INK2_DRAFT_DEPTH=ab with TF_INK2_DRAFT_CUT=ab is refused, either A/B alone is not" {
    try std.testing.expectError(error.DraftAbTogether, abPair(true, .ab));
    for ([_]Cut{ .off, .on }) |cut| try abPair(true, cut);
    for ([_]Cut{ .off, .on, .ab }) |cut| try abPair(false, cut);
}

test "inkling2 lanes: TF_INK2_SHARED_CUT: `deepen` unset, `cut`, `0` off" {
    const t = std.testing;
    try t.expectEqual(lanes.config.SharedCut.deepen, parseSharedCut(null)); // the default
    // a flush of left-out streams reads chances only when shared rounds use them in effect: never with the cut off
    try t.expect(!leftOutChances(true, .off, .deepen) and !leftOutChances(true, .off, .cut) and !leftOutChances(true, .off, .off));
    try t.expect(leftOutChances(true, .on, .deepen) and leftOutChances(true, .ab, .cut) and !leftOutChances(true, .on, .off));
    // a lone round or one of 5 or more live streams (not open): v0.2's proposals-only flush whatever the switches say
    try t.expect(!leftOutChances(false, .on, .deepen) and !leftOutChances(false, .ab, .cut));
    try t.expectEqual(lanes.config.SharedCut.off, parseSharedCut("0"));
    for ([_][]const u8{ "off", "false", "OFF", "False" }) |v| try t.expectEqual(lanes.config.SharedCut.off, parseSharedCut(v));
    try t.expectEqual(lanes.config.SharedCut.deepen, parseSharedCut("yes"));
    try t.expectEqual(lanes.config.SharedCut.deepen, parseSharedCut("00")); // exactly `0`: other digits stay the default
    try t.expectEqual(lanes.config.SharedCut.cut, parseSharedCut("cut"));
    try t.expectEqual(lanes.config.SharedCut.cut, parseSharedCut("1"));
    try t.expectEqual(lanes.config.SharedCut.deepen, parseSharedCut("deepen"));
    try t.expectEqual(lanes.config.SharedCut.deepen, parseSharedCut("2"));
    // the fact reaches the lane core's config only with the priced shared pass and the drafter
    const m: lanes.Model = .{ .exact_width = ic.state.spec_rows, .gpu_tokens = false, .mtp = true, .speculate = true, .drafts = ic.dspark.drafts, .draft_prior = &draft_prior, .window_costs = &tp4_costs, .window_costs_lone = true, .hidden_rows = true, .batch_rows = batch_rows, .max_streams = batch_streams, .shared_costs = &shared_costs, .draft_streams = true, .draft_pass = shared_pass, .shared_cut = .deepen };
    var c = try lanes.config.Config.init(t.allocator, m, ic.state.spec_rows, ic.state.spec_rows - 1);
    defer c.deinit(t.allocator);
    try t.expectEqual(lanes.config.SharedCut.deepen, c.shared_cut);
}

test "inkling2 lanes: the settings fingerprint's shared cut is the one the lane core's config resolves" {
    // cuda_dspark.zig sharedCutEffective (every rank's fingerprint) against the lanes' own resolution (factsCut, then
    // lanes/config.zig), for every combination of the switches either reads
    const names = [_][*:0]const u8{ "TF_INK2_DRAFT_CUT", "TF_INK2_BATCH", "TF_INK2_DRAFT_STREAMS", "TF_INK2_DRAFT_PRICE", "TF_INK2_DRAFT_DEPTH", "TF_INK2_SHARED_CUT" };
    for (names) |n| if (std.c.getenv(n) != null) return error.SkipZigTest;
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
        fn put(name: [*:0]const u8, v: ?[*:0]const u8) void {
            if (v) |x| _ = setenv(name, x, 1) else _ = unsetenv(name);
        }
    };
    defer for (names) |n| env.put(n, null);
    const gpa = std.testing.allocator;
    var e: engine.Engine = undefined;
    e.drafter = @ptrFromInt(@alignOf(ic.dspark.Drafter)); // never read: facts only ask whether a drafter is loaded
    const l: Lanes = .{ .x = .{ .e = &e }, .gpa = gpa };
    var seen = [_]bool{ false, false, false };
    for ([_]?[*:0]const u8{ null, "0", "ab" }) |cut| for ([_]?[*:0]const u8{ null, "0" }) |batch| for ([_]?[*:0]const u8{ null, "0" }) |one| for ([_]?[*:0]const u8{ null, "0" }) |priced| for ([_]?[*:0]const u8{ null, "0", "ab" }) |dep| for ([_]?[*:0]const u8{ null, "0", "off", "false", "OFF", "cut", "1", "deepen", "2", "x" }) |shared| {
        if (dep != null and cut != null and std.mem.eql(u8, std.mem.span(dep.?), "ab") and std.mem.eql(u8, std.mem.span(cut.?), "ab")) continue; // refused at start (abSwitches)
        env.put("TF_INK2_DRAFT_DEPTH", dep);
        env.put("TF_INK2_DRAFT_CUT", cut);
        env.put("TF_INK2_BATCH", batch);
        env.put("TF_INK2_DRAFT_STREAMS", one);
        env.put("TF_INK2_DRAFT_PRICE", priced);
        env.put("TF_INK2_SHARED_CUT", shared);
        var cfg = try lanes.Config.init(gpa, l.facts(), ic.state.spec_rows, ic.state.spec_rows - 1);
        defer cfg.deinit(gpa);
        const hashed = ic.dspark.sharedCutEffective();
        try std.testing.expectEqualStrings(@tagName(cfg.shared_cut), @tagName(hashed));
        seen[@intFromEnum(hashed)] = true;
    };
    try std.testing.expect(seen[0] and seen[1] and seen[2]); // off, cut and deepen each reached
}

test "inkling2 lanes: the lone cut's chances: a rising map under 1, each lane's own block, none unread" {
    var last: f64 = 0;
    var p: f64 = 0;
    while (p <= 1.0) : (p += 0.0005) {
        const y = landing(p);
        try std.testing.expect(y >= last and y < 1);
        last = y;
    }
    try std.testing.expectEqual(@as(f64, 0), landing(0));
    for (cut_map) |k| try std.testing.expectApproxEqAbs(k[1], landing(k[0]), 1e-12);
    // two streams' blocks: each lane gives its own chances; a block whose chances were not read gives none
    var a: Lane = .{ .slot = 0, .held_n = 7, .cut = true, .chances_ok = true };
    var b: Lane = .{ .slot = 1, .held_n = 3, .cut = true };
    a.chances = @splat(0.999);
    b.chances = @splat(0.3);
    var out: [7]f64 = undefined;
    try std.testing.expect(chainChances(&a, &out));
    for (out, 1..) |x, j| try std.testing.expectApproxEqAbs(std.math.pow(f64, landing(@as(f32, 0.999)), @floatFromInt(j)), x, 1e-12);
    try std.testing.expect(!chainChances(&b, out[0..3]));
    b.chances_ok = true;
    try std.testing.expect(chainChances(&b, out[0..3]));
    for (out[0..3], 1..) |x, j| try std.testing.expectApproxEqAbs(std.math.pow(f64, landing(@as(f32, 0.3)), @floatFromInt(j)), x, 1e-12);
    try std.testing.expect(!chainChances(&b, out[0..4])); // more than it holds
}

/// A drafter for `eagerPass` (host tests): each pass writes its streams' drafts and chances (slot-dependent) into its
/// own buffer a stream, as the device does, so a later pass overwrites an earlier one's.
const FakeDrafter = struct {
    last: [engine.max_segs][ic.dspark.drafts]f32 = undefined,
    passes: [2]usize = .{ 0, 0 }, // one-stream (`draft`) and several-stream (`draftMany`) passes
    reads: usize = 0,

    fn chance(slot: u32, j: usize) f32 {
        return @as(f32, @floatFromInt(slot + 1)) / 16 + @as(f32, @floatFromInt(j)) / 128;
    }

    fn fill(d: *FakeDrafter, s: usize, slot: u32, out: *[ic.dspark.drafts]u32) void {
        for (out, &d.last[s], 0..) |*t, *c, j| {
            t.* = slot * 100 + @as(u32, @intCast(j));
            c.* = chance(slot, j);
        }
    }

    fn draft(d: *FakeDrafter, slot: u32, n: usize, anchor: u32, out: *[ic.dspark.drafts]u32) !void {
        _ = .{ n, anchor };
        d.passes[0] += 1;
        d.fill(0, slot, out);
    }

    fn draftMany(d: *FakeDrafter, items: []const engine.DraftItem, keeps: []const usize, out: []const *[ic.dspark.drafts]u32) !void {
        _ = keeps;
        d.passes[1] += 1;
        for (items, out, 0..) |it, o, s| d.fill(s, it.slot, o);
    }

    fn chances(d: *FakeDrafter, out: []const *[ic.dspark.drafts]f32) !void {
        d.reads += 1;
        for (out, 0..) |o, s| o.* = d.last[s];
    }

    fn expectOwn(l: *const Lane) !void {
        try std.testing.expect(l.chances_ok);
        for (l.chances, 0..) |c, j| try std.testing.expectEqual(chance(l.slot, j), c);
        for (l.held, 0..) |t, j| try std.testing.expectEqual(l.slot * 100 + @as(u32, @intCast(j)), t);
    }
};

/// A stream for `eagerPass` (host tests): greedy, no grammar, its context `context`.
fn testStream(id: []const u8, context: u64) !lanes.Stream {
    var s = try lanes.Stream.init(std.testing.allocator, .{ .id = id, .prompt = &.{ 1, 2, 3 }, .max_new = 8 });
    s.cache_len = context;
    return s;
}

test "inkling2 lanes: an eager draft keeps its chances, each lane its own, one stream a pass or several in one" {
    var d: FakeDrafter = .{};
    var s = try testStream("shallow", 600); // every lane's stream: greedy, at 600
    defer s.deinit(std.testing.allocator);
    var a: Lane = .{ .slot = 2, .cut = true };
    var b: Lane = .{ .slot = 5 };
    var c: Lane = .{ .slot = 7, .cut = true };
    // TF_INK2_DRAFT_STREAMS=0: a pass a stream, one after the other; the cut lane's chances are read before the next
    // pass overwrites them, a lane not cut reads none
    try Lanes.eagerPass(&d, &d, &.{.{ .slot = a.slot, .anchor = 1 }}, &.{1}, &.{&a}, &.{&s});
    try Lanes.eagerPass(&d, &d, &.{.{ .slot = b.slot, .anchor = 1 }}, &.{1}, &.{&b}, &.{&s});
    try std.testing.expectEqual([2]usize{ 2, 0 }, d.passes);
    try std.testing.expectEqual(@as(usize, 1), d.reads);
    try FakeDrafter.expectOwn(&a);
    try std.testing.expect(!b.chances_ok);
    // the cut reads them: the chain's chances of the lane's own block
    a.held_n = ic.dspark.drafts;
    var out: [ic.dspark.drafts]f64 = undefined;
    try std.testing.expect(chainChances(&a, &out));
    try std.testing.expectApproxEqAbs(landing(FakeDrafter.chance(a.slot, 0)), out[0], 1e-12);
    // the drafting streams in one pass (one stream: `draft`, several: `draftMany`), each lane its own stream's chances
    a.chances_ok = false;
    try Lanes.eagerPass(&d, &d, &.{ .{ .slot = b.slot, .anchor = 1 }, .{ .slot = c.slot, .anchor = 1 }, .{ .slot = a.slot, .anchor = 1 } }, &.{ 1, 1, 1 }, &.{ &b, &c, &a }, &.{ &s, &s, &s });
    try std.testing.expectEqual([2]usize{ 2, 1 }, d.passes);
    try std.testing.expectEqual(@as(usize, 2), d.reads);
    for ([_]*const Lane{ &a, &b, &c }) |l| try FakeDrafter.expectOwn(l);
    c.chances_ok = false;
    try Lanes.eagerPass(&d, &d, &.{.{ .slot = c.slot, .anchor = 1 }}, &.{1}, &.{&c}, &.{&s});
    try std.testing.expectEqual([2]usize{ 3, 1 }, d.passes);
    try FakeDrafter.expectOwn(&c);
    // no lane cut (TF_INK2_DRAFT_CUT=0: the drafter makes no chances): nothing read
    var e: Lane = .{ .slot = 3 };
    var f: Lane = .{ .slot = 4 };
    try Lanes.eagerPass(&d, &d, &.{ .{ .slot = e.slot, .anchor = 1 }, .{ .slot = f.slot, .anchor = 1 } }, &.{ 1, 1 }, &.{ &e, &f }, &.{ &s, &s });
    try std.testing.expectEqual(@as(usize, 3), d.reads);
    try std.testing.expect(!e.chances_ok and !f.chances_ok);
}

test "inkling2 lanes: an eager pass reads no chances that no next window can use, and every lane's in order when one can" {
    const gpa = std.testing.allocator;
    var d: FakeDrafter = .{};
    var deep = try testStream("deep", depth_from + 1);
    defer deep.deinit(gpa);
    var sampled = try testStream("sampled", 600);
    defer sampled.deinit(gpa);
    sampled.sampling = .{ .seed = 1 };
    var walked = try testStream("grammar", 600);
    defer walked.deinit(gpa);
    var k: lanes.grammar.Constraint = undefined; // never read: only whether the stream has one
    walked.grammar = &k;
    defer walked.grammar = null;
    var at = try testStream("at depth_from", depth_from);
    defer at.deinit(gpa);
    var a: Lane = .{ .slot = 1, .cut = true };
    var b: Lane = .{ .slot = 2, .cut = true };
    var c: Lane = .{ .slot = 3, .cut = true };
    var w: Lane = .{ .slot = 4 }; // not in the cut arm (TF_INK2_DRAFT_CUT=ab's whole arm)
    var g: Lane = .{ .slot = 6, .cut = true };
    // none can: a cut lane past depth_from, sampled or with a grammar, and a lane not cut at a context the cut takes;
    // a pass a stream (TF_INK2_DRAFT_STREAMS=0) or all in one, nothing is read
    const lns = [_]*Lane{ &a, &b, &c, &w };
    const streams = [_]*const lanes.Stream{ &deep, &sampled, &walked, &at };
    for (lns, streams) |l, s| try Lanes.eagerPass(&d, &d, &.{.{ .slot = l.slot, .anchor = 1 }}, &.{1}, &.{l}, &.{s});
    const items = [_]engine.DraftItem{ .{ .slot = a.slot, .anchor = 1 }, .{ .slot = b.slot, .anchor = 1 }, .{ .slot = c.slot, .anchor = 1 }, .{ .slot = w.slot, .anchor = 1 }, .{ .slot = g.slot, .anchor = 1 } };
    try Lanes.eagerPass(&d, &d, items[0..4], &.{ 1, 1, 1, 1 }, &lns, &streams);
    try std.testing.expectEqual([2]usize{ 4, 1 }, d.passes);
    try std.testing.expectEqual(@as(usize, 0), d.reads);
    for (lns) |l| try std.testing.expect(!l.chances_ok);
    // a cut lane at depth_from, greedy, without a grammar: its pass's chances are read, every lane's in the pass's
    // order (the batch's order kept, each its own stream's), alone or among the others
    try Lanes.eagerPass(&d, &d, items[4..], &.{1}, &.{&g}, &.{&at});
    try std.testing.expectEqual(@as(usize, 1), d.reads);
    try FakeDrafter.expectOwn(&g);
    g.chances_ok = false;
    try Lanes.eagerPass(&d, &d, &.{ items[0], items[4], items[1], items[2], items[3] }, &.{ 1, 1, 1, 1, 1 }, &.{ &a, &g, &b, &c, &w }, &.{ &deep, &at, &sampled, &walked, &at });
    try std.testing.expectEqual(@as(usize, 2), d.reads);
    for ([_]*const Lane{ &a, &g, &b, &c, &w }) |l| try FakeDrafter.expectOwn(l);
    // TF_INK2_DRAFT_DEPTH=0: no context prices, so the lane core cuts nothing (lanes/config.zig lone_cut): none read
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    _ = env.setenv("TF_INK2_DRAFT_DEPTH", "0", 1);
    defer _ = env.unsetenv("TF_INK2_DRAFT_DEPTH");
    g.chances_ok = false;
    try Lanes.eagerPass(&d, &d, items[4..], &.{1}, &.{&g}, &.{&at});
    try std.testing.expectEqual(@as(usize, 2), d.reads);
    try std.testing.expect(!g.chances_ok);
}

test "inkling2 lanes: a deferred pass reads its chances only when one of its lanes is cut" {
    // flushWhere's passes (the omitted streams' flush in a C2 round asks chances): lanes none of which is cut
    // (TF_INK2_DRAFT_CUT=ab's whole arm) read nothing, one pass or several; a mixed pass reads every lane's, in order
    var d: FakeDrafter = .{};
    var a: Lane = .{ .slot = 1 };
    var b: Lane = .{ .slot = 2 };
    var c: Lane = .{ .slot = 3, .cut = true };
    const items = [_]engine.DraftItem{ .{ .slot = a.slot, .anchor = 1 }, .{ .slot = b.slot, .anchor = 1 }, .{ .slot = c.slot, .anchor = 1 } };
    try Lanes.deferredPass(&d, &d, items[0..1], &.{1}, &.{&a.held}, &.{&a}, true);
    try Lanes.deferredPass(&d, &d, items[0..2], &.{ 1, 1 }, &.{ &a.held, &b.held }, &.{ &a, &b }, true);
    try std.testing.expectEqual([2]usize{ 1, 1 }, d.passes);
    try std.testing.expectEqual(@as(usize, 0), d.reads);
    try std.testing.expect(!a.chances_ok and !b.chances_ok);
    // with `chances` false nothing is read, cut or not
    try Lanes.deferredPass(&d, &d, items[2..3], &.{1}, &.{&c.held}, &.{&c}, false);
    try std.testing.expectEqual(@as(usize, 0), d.reads);
    // a pass with a cut lane: every lane's chances, each its own stream's
    try Lanes.deferredPass(&d, &d, &items, &.{ 1, 1, 1 }, &.{ &a.held, &b.held, &c.held }, &.{ &a, &b, &c }, true);
    try std.testing.expectEqual(@as(usize, 1), d.reads);
    for ([_]*const Lane{ &a, &b, &c }) |l| try FakeDrafter.expectOwn(l);
}

test "inkling2 lanes: once draftFn's dispatch returns a lane's chances are its new block's, or none" {
    const gpa = std.testing.allocator;
    var d: FakeDrafter = .{};
    var s = try testStream("shallow", 600);
    defer s.deinit(gpa);
    var deep = try testStream("deep", depth_from + 1);
    defer deep.deinit(gpa);
    // each lane holds its last block's chances, read
    var a: Lane = .{ .slot = 2, .cut = true, .chances_ok = true };
    var b: Lane = .{ .slot = 5, .cut = true, .chances_ok = true };
    a.chances = @splat(0.5);
    b.chances = @splat(0.5);
    // a pass a stream (TF_INK2_DRAFT_STREAMS=0): a lane whose stream can be cut at its next window holds the new
    // block's chances; one whose next window cannot use them holds none (not its last block's)
    var each: Lanes.Dispatch = .{ .how = .each };
    try each.add(&d, &d, &a, .{ .slot = a.slot, .anchor = 1 }, 1, &s, ic.dspark.drafts);
    try FakeDrafter.expectOwn(&a);
    try each.add(&d, &d, &b, .{ .slot = b.slot, .anchor = 1 }, 1, &deep, ic.dspark.drafts);
    try std.testing.expect(!b.chances_ok);
    try each.finish(&d, &d);
    try std.testing.expectEqual([2]usize{ 2, 0 }, d.passes);
    var out: [ic.dspark.drafts]f64 = undefined;
    try std.testing.expect(chainChances(&a, &out) and !chainChances(&b, &out));
    // one pass of every drafting stream: none count until it runs, then each lane's own
    a.chances_ok = true;
    b.chances_ok = true;
    var one: Lanes.Dispatch = .{ .how = .one };
    try one.add(&d, &d, &a, .{ .slot = a.slot, .anchor = 1 }, 1, &s, ic.dspark.drafts);
    try one.add(&d, &d, &b, .{ .slot = b.slot, .anchor = 1 }, 1, &s, ic.dspark.drafts);
    try std.testing.expect(!a.chances_ok and !b.chances_ok);
    try one.finish(&d, &d);
    try std.testing.expectEqual([2]usize{ 2, 1 }, d.passes);
    try FakeDrafter.expectOwn(&a);
    try FakeDrafter.expectOwn(&b);
    // deferred (the default): no pass now; the last block's chances stop counting, the draft waits for the verify
    var later: Lanes.Dispatch = .{ .how = .deferred };
    try later.add(&d, &d, &a, .{ .slot = a.slot, .anchor = 1 }, 1, &s, ic.dspark.drafts);
    try later.finish(&d, &d);
    try std.testing.expect(!a.chances_ok and a.pending != null and a.held_n == ic.dspark.drafts);
    try std.testing.expectEqual([2]usize{ 2, 1 }, d.passes);
    try std.testing.expectEqual(@as(usize, 2), d.reads);
}

test "inkling2 lanes: a mask sets every disallowed token to -inf, the row's last word partial, and refuses a row it empties" {
    const a = std.testing.allocator;
    const U = 70; // 2 words and 6 bits, as 200,058 = 6,251 words and 26 bits
    var logits: [U]f32 = undefined;
    for (&logits, 0..) |*l, i| l.* = @floatFromInt(i % 7);
    var out: [U]f32 = undefined;
    // allowed: 3, 40, 69, and bits past the row (70 .. 95, ids with no logits) that change nothing
    var mask = [_]u32{ 1 << 3, 1 << 8, (1 << 5) | 0xffff_ffc0 };
    try maskRow(&out, &logits, &mask);
    for (out, logits, 0..) |o, l, i| {
        if (i == 3 or i == 40 or i == 69) try std.testing.expectEqual(l, o) else try std.testing.expect(o == -std.math.inf(f32));
    }
    // a mask that allows only ids past the row allows none of the row's tokens
    mask = .{ 0, 0, 0xffff_ffc0 };
    try std.testing.expectError(error.NoAllowedToken, maskRow(&out, &logits, &mask));
    // a mask shorter than the row is refused
    try std.testing.expectError(error.InvalidGrammarMasks, maskRow(&out, &logits, mask[0..2]));
    // a window's masks: a row's words each, covering the vocabulary
    const masks = try a.alloc(u32, 3 * 3);
    defer a.free(masks);
    var s: lanes.Stream = undefined;
    const w: be.Window = .{ .stream = &s, .pending = 1, .held = 0, .tokens = &.{ 2, 3 }, .parents = null, .positions = &.{ 4, 5, 6 }, .masks = masks };
    try std.testing.expectEqual(@as(usize, 3), try maskWords(w, U));
    try std.testing.expectError(error.InvalidGrammarMasks, maskWords(w, 97)); // 3 words cover 96 tokens
    var short = w;
    short.masks = masks[0..8];
    try std.testing.expectError(error.InvalidGrammarMasks, maskWords(short, U));
    short.masks = &.{};
    try std.testing.expectEqual(@as(usize, 0), try maskWords(short, U));
}

test "inkling2 lanes: the masked argmax read in place is the argmax of the masked row" {
    const a = std.testing.allocator;
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();
    const U = 1000;
    const logits = try a.alloc(f32, U);
    defer a.free(logits);
    const out = try a.alloc(f32, U);
    defer a.free(out);
    var mask: [(U + 31) / 32]u32 = undefined;
    for (0..200) |k| {
        // few distinct values (ties), masks from dense to a single token
        for (logits) |*x| x.* = @floatFromInt(r.intRangeLessThan(i32, -4, 5));
        for (&mask) |*m| m.* = if (k % 4 == 0) std.math.maxInt(u32) else r.int(u32) & r.int(u32) & if (k % 4 == 3) r.int(u32) else std.math.maxInt(u32);
        mask[mask.len - 1] |= 1 << 7; // at least one allowed: id 999
        try maskRow(out, logits, &mask);
        try std.testing.expectEqual(lanes.gpu_rule.argmax(out), try argmaxMasked(logits, &mask));
    }
    @memset(&mask, 0);
    mask[mask.len - 1] = 0xffff_ff00; // ids past the row only
    try std.testing.expectError(error.NoAllowedToken, argmaxMasked(logits, &mask));
}

/// A Lanes for the host draws alone (no engine forward runs).
fn hostLanes(e: *engine.Engine) Lanes {
    e.drafter = null;
    e.io = std.testing.io;
    return .{ .x = .{ .e = e }, .gpa = std.testing.allocator };
}

test "inkling2 lanes: a masked draw is the row's own rule over the masked row: greedy the first largest allowed, sampled the sampler's" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    const U = 3000;
    const logits = try a.alloc(f32, U);
    defer a.free(logits);
    for (logits) |*x| x.* = @floatCast(@abs(r.floatNorm(f64)) * 4.0);
    var mask: [(U + 31) / 32]u32 = undefined;
    for (&mask) |*m| m.* = r.int(u32) & r.int(u32); // a quarter allowed
    // a tie between two allowed tokens above the rest of the allowed ones, below the row's best (not allowed)
    logits[1777] = 40.0;
    logits[2999] = 40.0;
    logits[123] = 50.0;
    mask[1777 / 32] |= @as(u32, 1) << @intCast(1777 % 32);
    mask[2999 / 32] |= @as(u32, 1) << @intCast(2999 % 32);
    mask[123 / 32] &= ~(@as(u32, 1) << @intCast(123 % 32));
    const best = lanes.gpu_rule.argmax(logits);
    try std.testing.expectEqual(@as(u32, 123), best);
    var want_best: ?usize = null;
    for (logits, 0..) |x, i| if (lanes.grammar.allows(&mask, i) and (want_best == null or x > logits[want_best.?])) {
        want_best = i;
    };
    try std.testing.expectEqual(@as(?usize, 1777), want_best);
    var s = try lanes.Stream.init(a, .{ .id = "masked", .prompt = &.{ 1, 2 }, .max_new = 8 });
    defer s.deinit(a);
    try std.testing.expectEqual(@as(u32, @intCast(want_best.?)), try l.drawMasked(logits, &mask, &s, 2));
    // sampled: the sampler over the masked row (disallowed at -inf), never a disallowed token
    const masked = try a.alloc(f32, U);
    defer a.free(masked);
    try maskRow(masked, logits, &mask);
    var ref: sample.Sampler = .{ .gpa = a };
    defer ref.deinit();
    for ([_]lanes.Sampling{ .{ .seed = 3, .temperature = 0.8, .top_k = 0, .top_p = 0.95 }, .{ .seed = 4, .temperature = 1.0, .top_k = 40, .top_p = 1.0 }, .{ .seed = 5, .temperature = 1.0, .top_k = 0, .top_p = 1.0 } }) |v| {
        s.sampling = v;
        for (0..50) |p| {
            const got = try l.drawMasked(logits, &mask, &s, 2 + p);
            try std.testing.expect(lanes.grammar.allows(&mask, got));
            try std.testing.expectEqual(try ref.draw(masked, v, 2 + p), got);
        }
    }
    // a mask that allows every token draws what the row draws unmasked
    @memset(&mask, std.math.maxInt(u32));
    s.sampling = null;
    try std.testing.expectEqual(best, try l.drawMasked(logits, &mask, &s, 2));
    try std.testing.expect(l.masked_rows == 2 + 150);
}

test "inkling2 lanes: a grammar stream's first token comes from the kept prompt row, masked or not, and frees it" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "first", .prompt = &.{ 1, 2, 3 }, .max_new = 8 });
    defer s.deinit(a);
    const row = try a.alloc(f32, 100);
    for (row, 0..) |*x, i| x.* = @floatFromInt(i);
    try l.lanes.put(a, &s, .{ .slot = 0, .first_row = row });
    // the lane core's first_masked: among 7 and 41 the larger
    var mask: [4]u32 = @splat(0);
    mask[0] = 1 << 7;
    mask[1] = 1 << 9;
    try std.testing.expectError(error.PositionMismatch, Lanes.firstMaskedFn(&l, &s, 2, &mask));
    const h = try Lanes.firstMaskedFn(&l, &s, 3, &mask);
    try std.testing.expectEqual(@as(u32, 41), try Lanes.readFn(&l, h));
    try std.testing.expectEqual(@as(usize, 0), l.lanes.get(&s).?.first_row.len);
    try std.testing.expectEqual(h, try Lanes.firstFn(&l, &s, 3)); // the same handle after
    try std.testing.expectError(error.NoFirstToken, Lanes.firstMaskedFn(&l, &s, 3, &mask)); // the row is gone
    // a grammar that leaves the first token free: `first` draws the kept row's own token
    const row2 = try a.alloc(f32, 100);
    for (row2, 0..) |*x, i| x.* = -@as(f32, @floatFromInt(i));
    l.lanes.getPtr(&s).?.* = .{ .slot = 0, .first_row = row2 };
    try std.testing.expectEqual(@as(u32, 0), try Lanes.readFn(&l, try Lanes.firstFn(&l, &s, 3)));
    // a row still kept when the lanes go is freed with them (the testing allocator checks)
    l.lanes.getPtr(&s).?.first_row = try a.dupe(f32, row2[0..10]);
}

test "inkling2 lanes: the backend's steps compile" {
    std.testing.refAllDecls(Lanes); // the prompt chunk's image path (chunk, imageChunk, tower) runs only on a GPU
}

/// Rank 0's vision for the lanes' host tests: a row store of `rows` rows over no memory (nothing here runs the tower
/// or compacts), no tower. The engine around it serves at `context` with no page pool.
const HostVision = struct {
    r: ic.vision.Rank0,

    fn noCopy(_: *anyopaque, _: u64, _: u64, _: usize) anyerror!void {
        return error.NoDeviceInHostTests;
    }

    fn noDone(_: *anyopaque) anyerror!void {
        return error.NoDeviceInHostTests;
    }

    fn init(h: *HostVision, e: *engine.Engine, rows: u32, max_images: u32) !void {
        h.r = undefined;
        h.r.limits = .{ .image_tokens = rows, .max_images = max_images };
        h.r.store = try .init(std.testing.allocator, 0x10_0000, rows, vs.row_bytes, .{ .ctx = h, .run = noCopy, .done = noDone });
        h.r.pace = .{};
        e.vision = &h.r;
        e.pool = null;
        e.o.context = 1 << 16;
    }

    fn deinit(h: *HostVision) void {
        h.r.store.deinit();
    }
};

/// An image as the server hands it to the lanes (image.zig carry and seal: the record, then the RGB), `h` x `w`
/// pixels at prompt row `at`.
const HostImage = struct {
    floats: []f32,
    img: lanes.stream.Image,

    fn init(h: u32, w: u32, at: u32, seed: u8) !HostImage {
        const c = try image.carry(std.testing.allocator, h, w);
        for (c.rgb, 0..) |*b, i| b.* = @truncate(i *% 31 +% seed);
        const d = image.seal(c.floats);
        const g = image.grid(h, w);
        return .{ .floats = c.floats, .img = .{ .pixels = c.floats, .gh = g[0], .gw = g[1], .at = at, .tokens = g[0] * g[1], .digest = d } };
    }

    fn deinit(x: *HostImage) void {
        std.testing.allocator.free(x.floats);
    }
};

test "inkling2 lanes: a text stream opens as without the vision tower: no images, nothing reserved, a zero image key" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var hv: HostVision = undefined;
    try hv.init(&e, 64, 4);
    defer hv.deinit();
    const ids = [_]u32{ 5, 6, 7, 8 };
    var s = try lanes.Stream.init(a, .{ .id = "text", .prompt = &ids, .max_new = 8 });
    defer s.deinit(a);
    const lane = try l.opening(&s);
    try std.testing.expect(lane.img == null);
    try std.testing.expectEqual(@as(usize, 0), lane.from);
    const st = &hv.r.store;
    try std.testing.expect(st.committed == 0 and st.clock == 0 and st.entries.items.len == 0);
    // the prompt cache's image key of a text prefix: zeros with no hashing, as a kept state's default
    const zero: vs.Digest = @splat(0);
    try std.testing.expectEqual(zero, try imageKey(s.images, 3));
    const k: ic.snap.Kept = .{ .tag = 1, .at = 3, .off = 0, .bytes = 0 };
    try std.testing.expect(Lanes.sameImages(&s, &k));
    // without the tower the same: a text stream never reads Engine.vision
    e.vision = null;
    var t = try lanes.Stream.init(a, .{ .id = "text2", .prompt = &ids, .max_new = 8 });
    defer t.deinit(a);
    try std.testing.expect((try l.opening(&t)).img == null);
    l.dropLane(&t);
    l.dropLane(&s);
    try std.testing.expectEqual(@as(u32, 0), l.lanes.count());
}

test "inkling2 lanes: an image prompt's opening refuses bad images before anything moves, and reserves its rows" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var hv: HostVision = undefined;
    try hv.init(&e, 20, 2);
    defer hv.deinit();
    const st = &hv.r.store;
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i + 1);
    var im0 = try HostImage.init(80, 120, 10, 1); // 2 x 3 = 6 tokens at [10, 16)
    defer im0.deinit();
    var im1 = try HostImage.init(120, 120, 20, 2); // 3 x 3 = 9 tokens at [20, 29)
    defer im1.deinit();
    const Case = struct { images: []const lanes.stream.Image, prompt: usize = 64, want: anyerror };
    var bad_digest = im0.img;
    bad_digest.digest[31] ^= 1;
    var zero_digest = try HostImage.init(40, 40, 10, 3);
    defer zero_digest.deinit();
    @memset(std.mem.sliceAsBytes(zero_digest.floats)[@offsetOf(image.Record, "digest")..][0..32], 0);
    zero_digest.img.digest = @splat(0);
    var not_ours = im0.img;
    not_ours.pixels = &.{ 0, 0, 0 };
    var wrong_grid = im0.img;
    wrong_grid.gw = 2;
    wrong_grid.tokens = 4;
    var overlap = im1.img;
    overlap.at = 15;
    var late = im1.img;
    late.at = 55; // [55, 64): its last row is the prompt's
    var big = try HostImage.init(200, 200, 30, 4); // 25 tokens: past the store's 20
    defer big.deinit();
    const cases = [_]Case{
        .{ .images = &.{not_ours}, .want = error.NotInkling2Image },
        .{ .images = &.{wrong_grid}, .want = error.NotInkling2Image },
        .{ .images = &.{bad_digest}, .want = error.ImageDigestDiffers },
        .{ .images = &.{zero_digest.img}, .want = error.ImageDigestDiffers },
        .{ .images = &.{ im0.img, overlap }, .want = error.ImageOrder },
        .{ .images = &.{ im1.img, im0.img }, .want = error.ImageOrder },
        .{ .images = &.{late}, .want = error.ImageOutsidePrompt },
        .{ .images = &.{im0.img}, .prompt = 16, .want = error.ImageOutsidePrompt },
        .{ .images = &.{ im0.img, im1.img, im0.img }, .want = error.TooManyImages }, // --vision-max-images 2
        .{ .images = &.{big.img}, .want = error.ImageTokens },
    };
    for (cases) |c| {
        var s = try lanes.Stream.init(a, .{ .id = "img", .prompt = ids[0..c.prompt], .max_new = 8, .images = c.images });
        defer s.deinit(a);
        try std.testing.expectError(c.want, l.opening(&s));
        try std.testing.expect(l.lanes.count() == 0 and st.committed == 0 and st.entries.items.len == 0);
    }
    // without the tower an image prompt is refused by name
    e.vision = null;
    var nv = try lanes.Stream.init(a, .{ .id = "img", .prompt = &ids, .max_new = 8, .images = &.{im0.img} });
    defer nv.deinit(a);
    try std.testing.expectError(error.NoVision, l.opening(&nv));
    e.vision = &hv.r;
    // two good images: their 15 rows reserved; another such request waits (LaneMemory) with nothing moved
    var s = try lanes.Stream.init(a, .{ .id = "img", .prompt = &ids, .max_new = 8, .images = &.{ im0.img, im1.img } });
    defer s.deinit(a);
    const lane = try l.opening(&s);
    const p = &lane.img.?;
    try std.testing.expectEqual(@as(u32, 15), st.committed);
    try std.testing.expectEqual(@as(u32, 15), p.need.reserved);
    try std.testing.expectEqual(@as(u32, 2), p.count);
    try std.testing.expectEqual(vs.Image{ .at = 20, .tokens = 9, .rows = 3, .cols = 3, .digest = im1.img.digest }, p.images()[1]);
    var t = try lanes.Stream.init(a, .{ .id = "img2", .prompt = &ids, .max_new = 8, .images = &.{im1.img} });
    defer t.deinit(a);
    try std.testing.expectError(error.LaneMemory, l.opening(&t));
    try std.testing.expect(l.lanes.count() == 1 and st.committed == 15);
    // the stream opening again (a new pass on its slot) gives its last pass's rows back first: no wait on itself
    const again = try l.opening(&s);
    try std.testing.expectEqual(@as(u32, 15), st.committed);
    try std.testing.expectEqual(lane.slot, again.slot);
    // a drop gives them back
    l.dropLane(&s);
    try std.testing.expect(l.lanes.count() == 0 and st.committed == 0);
}

test "inkling2 lanes: an image chunk's splice comes from the stored rows, pins go as chunks pass, and the lane's release frees all" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var hv: HostVision = undefined;
    try hv.init(&e, 40, 4);
    defer hv.deinit();
    const st = &hv.r.store;
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i + 1);
    var im0 = try HostImage.init(80, 120, 10, 1); // [10, 16)
    defer im0.deinit();
    var im1 = try HostImage.init(120, 120, 20, 2); // [20, 29)
    defer im1.deinit();
    // both images already stored (an earlier request's towers): hits, no tower
    const hold = [_]vs.Image{ storeImage(im0.img), storeImage(im1.img) };
    for (&hold) |*m| {
        _ = try st.alloc(&m.digest, m.tokens);
        st.finish(&m.digest);
        _ = st.unpin(&m.digest);
    }
    const r0 = st.find(&hold[0].digest).?.row0;
    const r1 = st.find(&hold[1].digest).?.row0;
    // a text stream beside it holds a lane, so releasing the image stream leaves the lanes busy (no idle step)
    var txt = try lanes.Stream.init(a, .{ .id = "text", .prompt = ids[0..4], .max_new = 8 });
    defer txt.deinit(a);
    _ = try l.opening(&txt);
    var s = try lanes.Stream.init(a, .{ .id = "img", .prompt = &ids, .max_new = 8, .images = &.{ im0.img, im1.img } });
    defer s.deinit(a);
    const lane = try l.opening(&s);
    const p = &lane.img.?;
    // a text chunk before the images: no splice, nothing pinned
    try std.testing.expect((try l.imageChunk(&s, p, 0, 10, 0)).ready == null);
    try std.testing.expectEqual(@as(u32, 0), st.find(&hold[0].digest).?.pins);
    // a chunk cut inside image 0
    const c1 = (try l.imageChunk(&s, p, 8, 5, 0)).ready.?;
    try std.testing.expect(c1.count == 1 and c1.lo == 2 and c1.hi == 5);
    try std.testing.expectEqual(vs.Span{ .first = 2, .n = 3, .src = st.addr(r0) }, c1.spans[0]);
    try c1.check(5);
    try std.testing.expect(!p.need.passed(st, p.images(), 13));
    try std.testing.expectEqual(@as(u32, 1), st.find(&hold[0].digest).?.pins);
    // the rest of image 0, text, and image 1 whole: two spans, text rows between them
    const c2 = (try l.imageChunk(&s, p, 13, 32, 512)).ready.?;
    try std.testing.expect(c2.count == 2 and c2.lo == 0 and c2.hi == 16);
    try std.testing.expectEqual(vs.Span{ .first = 0, .n = 3, .src = st.addr(r0 + 3) }, c2.spans[0]);
    try std.testing.expectEqual(vs.Span{ .first = 7, .n = 9, .src = st.addr(r1) }, c2.spans[1]);
    try c2.check(32);
    try c2.within(st.buf, @as(usize, st.rows) * vs.row_bytes);
    _ = p.need.passed(st, p.images(), 45);
    try std.testing.expect(st.find(&hold[0].digest).?.pins == 0 and st.find(&hold[1].digest).?.pins == 0);
    try std.testing.expectEqual(@as(u32, 2), p.need.first);
    try std.testing.expect((try l.imageChunk(&s, p, 45, 19, 0)).ready == null);
    // the store's rows stay reserved until the pass ends; the stream's release (a cancel, or the end) frees them
    try std.testing.expectEqual(@as(u32, 15), st.committed);
    Lanes.releaseFn(&l, &s);
    try std.testing.expect(st.committed == 0 and l.lanes.count() == 1);
    // a pass dropped mid-image: its pins and rows go with the lane
    const again = try l.opening(&s);
    _ = (try l.imageChunk(&s, &again.img.?, 12, 2, 0)).ready.?;
    try std.testing.expectEqual(@as(u32, 1), st.find(&hold[0].digest).?.pins);
    l.dropLane(&s);
    try std.testing.expect(st.committed == 0 and st.find(&hold[0].digest).?.pins == 0);
    // a building entry whose tower state is not held (no unit began it): refused, and the entry dropped
    var im2 = try HostImage.init(40, 80, 30, 9);
    defer im2.deinit();
    const other = storeImage(im2.img);
    _ = try st.alloc(&other.digest, other.tokens);
    var cold = try HostImage.init(40, 40, 30, 7);
    defer cold.deinit();
    var w = try lanes.Stream.init(a, .{ .id = "img4", .prompt = &ids, .max_new = 8, .images = &.{cold.img} });
    defer w.deinit(a);
    const lw = try l.opening(&w);
    try std.testing.expectError(error.TowerJobLost, l.imageChunk(&w, &lw.img.?, 28, 4, 512));
    try std.testing.expect(st.job() == null and st.find(&other.digest) == null);
    l.dropLane(&w);
    try std.testing.expectEqual(@as(u32, 0), st.committed);
    l.dropLane(&txt);
}

/// Tower units for the lanes' host tests: device work counted, a clock that a layer advances, and a failure on demand.
const HostUnits = struct {
    work: u32 = 0,
    begins: u32 = 0,
    layers_run: u32 = 0,
    merges: u32 = 0,
    clock: u64 = 1,
    fail: bool = false,

    fn of(ctx: *anyopaque) *HostUnits {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(ctx: *anyopaque, _: u32) anyerror!void {
        of(ctx).begins += 1;
        of(ctx).work += 1;
    }
    fn layers(ctx: *anyopaque, k: u32) anyerror!void {
        const h = of(ctx);
        if (h.fail) return error.FakeFault;
        h.layers_run += k;
        h.work += 1;
        h.clock += 4_000_000 * @as(u64, k);
    }
    fn merge(ctx: *anyopaque, _: u32) anyerror!void {
        of(ctx).merges += 1;
        of(ctx).work += 1;
    }
    fn settle(_: *anyopaque) anyerror!void {}
    fn now(ctx: *anyopaque) u64 {
        return of(ctx).clock;
    }
    fn units(h: *HostUnits) vs.Units {
        return .{ .ctx = h, .begin = begin, .layers = layers, .merge = merge, .settle = settle, .now = now };
    }
};

test "inkling2 lanes: under a chunk row cap an image step is one tower unit (no forward), each step works, another stream helps, and a failed unit drops the job from every lane" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var hv: HostVision = undefined;
    try hv.init(&e, 64, 4);
    defer hv.deinit();
    const st = &hv.r.store;
    const pace = &hv.r.pace;
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i + 1);
    var im0 = try HostImage.init(80, 120, 10, 1); // [10, 16)
    defer im0.deinit();
    var im1 = try HostImage.init(120, 120, 20, 2); // [20, 29)
    defer im1.deinit();
    var hu: HostUnits = .{};
    var sa = try lanes.Stream.init(a, .{ .id = "a", .prompt = &ids, .max_new = 8, .images = &.{im0.img} });
    defer sa.deinit(a);
    var sb = try lanes.Stream.init(a, .{ .id = "b", .prompt = &ids, .max_new = 8, .images = &.{im1.img} });
    defer sb.deinit(a);
    const la = try l.opening(&sa);
    const lb = try l.opening(&sb);
    const pa = &la.img.?;
    const pb = &lb.img.?;
    // a's first step: begin and one layer, timed (4 ms); no chunk goes out
    try std.testing.expect(try l.imageChunkOn(&sa, pa, 8, 8, 512, hu.units()) == .worked);
    try std.testing.expect(hu.begins == 1 and hu.layers_run == 1 and pace.layer_ns == 4_000_000);
    // b's step meets a's job: it runs a's next unit (help, billed: a capped unit is side work the lane host bills,
    // though b's step was not settled)
    try std.testing.expect(try l.imageChunkOn(&sb, pb, 16, 8, 512, hu.units()) == .worked);
    try std.testing.expectEqual(vs.Help{ .units = 1, .layers = 1, .unbilled = 0 }, pb.help);
    try std.testing.expect(!pb.need.pinned.isSet(0));
    // a's settled chunks paced at 20 us a row: 512 rows are 10.24 ms, 2 layers a unit
    pace.chunked(20_000 * 512, 512);
    var calls: u32 = 0;
    while (true) : (calls += 1) {
        const work = hu.work;
        const r = try l.imageChunkOn(&sa, pa, 8, 8, 512, hu.units());
        if (r == .ready) {
            const sp = r.ready.?;
            try std.testing.expect(sp.count == 1 and sp.lo == 2 and sp.hi == 8);
            break;
        }
        try std.testing.expect(hu.work > work); // every step that is not the chunk ran a layer or the merger
    }
    try std.testing.expectEqual(@as(u32, 11), calls); // 21 layers left, 2 a unit, the merger with the last
    try std.testing.expect(hu.layers_run == vs.tower_layers and hu.merges == 1);
    // b's own image now: begin and a timed layer, then units until its chunk goes
    sb.settle_step = true; // the lane host bills these steps
    while (try l.imageChunkOn(&sb, pb, 16, 8, 512, hu.units()) == .worked) {}
    sb.settle_step = false;
    try std.testing.expect(hu.begins == 2 and hu.merges == 2 and st.find(&im1.img.digest).?.ready);
    try std.testing.expectEqual(vs.Help{ .units = 1, .layers = 1, .unbilled = 0 }, pb.help);
    // without a cap the whole tower and the chunk go in one step
    var im2 = try HostImage.init(40, 80, 40, 3); // [40, 42)
    defer im2.deinit();
    var sc = try lanes.Stream.init(a, .{ .id = "c", .prompt = &ids, .max_new = 8, .images = &.{im2.img} });
    defer sc.deinit(a);
    const lc = try l.opening(&sc);
    const work = hu.work;
    const rc = try l.imageChunkOn(&sc, &lc.img.?, 32, 32, 0, hu.units());
    try std.testing.expect(rc.ready != null and hu.work == work + 3 and hu.begins == 3);
    l.dropLane(&sc);
    // a failed unit: the job leaves the store and every lane's pins (d pins the same image), and d begins it again
    var im3 = try HostImage.init(80, 80, 40, 4);
    defer im3.deinit();
    var sd = try lanes.Stream.init(a, .{ .id = "d", .prompt = &ids, .max_new = 8, .images = &.{im3.img} });
    defer sd.deinit(a);
    var se = try lanes.Stream.init(a, .{ .id = "e", .prompt = &ids, .max_new = 8, .images = &.{im3.img} });
    defer se.deinit(a);
    const ld = try l.opening(&sd);
    const le = try l.opening(&se);
    try std.testing.expect(try l.imageChunkOn(&sd, &ld.img.?, 32, 32, 512, hu.units()) == .worked);
    try std.testing.expect(try l.imageChunkOn(&se, &le.img.?, 32, 32, 512, hu.units()) == .worked);
    try std.testing.expectEqual(@as(u32, 2), st.job().?.pins);
    hu.fail = true;
    try std.testing.expectError(error.FakeFault, l.imageChunkOn(&sd, &ld.img.?, 32, 32, 512, hu.units()));
    try std.testing.expect(st.job() == null and pace.job == null);
    try std.testing.expect(ld.img.?.need.pinned.count() == 0 and le.img.?.need.pinned.count() == 0);
    hu.fail = false;
    const begins = hu.begins;
    try std.testing.expect(try l.imageChunkOn(&se, &le.img.?, 32, 32, 512, hu.units()) == .worked);
    try std.testing.expectEqual(begins + 1, hu.begins);
    // the passes end: pins and rows back, the job dropped with its last pin
    for ([_]*lanes.Stream{ &sa, &sb, &sd, &se }) |x| l.dropLane(x);
    try std.testing.expect(st.committed == 0 and st.job() == null and l.lanes.count() == 0);
    try std.testing.expect(pace.help_units == 1 and pace.unbilled_help == 0);
}

test "inkling2 lanes: while the lane host holds side work an image step runs no capped unit and takes no rows; a stored image is served" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var hv: HostVision = undefined;
    try hv.init(&e, 64, 4);
    defer hv.deinit();
    const st = &hv.r.store;
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i + 1);
    var im0 = try HostImage.init(80, 120, 10, 1); // [10, 16)
    defer im0.deinit();
    var hu: HostUnits = .{};
    var sa = try lanes.Stream.init(a, .{ .id = "a", .prompt = &ids, .max_new = 8, .images = &.{im0.img} });
    defer sa.deinit(a);
    var sb = try lanes.Stream.init(a, .{ .id = "b", .prompt = &ids, .max_new = 8, .images = &.{im0.img} });
    defer sb.deinit(a);
    const la = try l.opening(&sa);
    const lb = try l.opening(&sb);
    // decode time is owed: a cold image's capped step runs nothing, allocates no rows and pins nothing
    sa.side_held = true;
    const r = try l.imageChunkOn(&sa, &la.img.?, 8, 8, 512, hu.units());
    try std.testing.expect(r != .worked and r != .ready);
    try std.testing.expect(hu.work == 0 and st.job() == null and st.find(&im0.img.digest) == null);
    try std.testing.expectEqual(@as(u32, 0), la.img.?.need.pinned.count());
    // without a chunk row cap nothing is held: the whole tower and the chunk in one step
    const rw = try l.imageChunkOn(&sa, &la.img.?, 8, 8, 0, hu.units());
    try std.testing.expect(rw == .ready and hu.begins == 1 and hu.merges == 1);
    // a stored image is served while held: a hit, no tower
    sb.side_held = true;
    const work = hu.work;
    const rb = try l.imageChunkOn(&sb, &lb.img.?, 8, 8, 512, hu.units());
    try std.testing.expect(rb == .ready and rb.ready.?.count == 1 and hu.work == work);
    for ([_]*lanes.Stream{ &sa, &sb }) |x| l.dropLane(x);
    try std.testing.expect(st.committed == 0 and st.job() == null and l.lanes.count() == 0);
}

test "inkling2 lanes: a kept state's image key names the images before its cut, and a resume refuses another image's state" {
    const a = std.testing.allocator;
    var e: engine.Engine = undefined;
    var l = hostLanes(&e);
    defer l.deinit();
    var hv: HostVision = undefined;
    try hv.init(&e, 40, 4);
    defer hv.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i + 1);
    var im0 = try HostImage.init(80, 120, 10, 1); // [10, 16)
    defer im0.deinit();
    const imgs = [_]lanes.stream.Image{im0.img};
    const zero: vs.Digest = @splat(0);
    // the key is the store's prefixKey: zeros up to the image's first row, then its own
    try std.testing.expectEqual(zero, try imageKey(&imgs, 10));
    const key = try imageKey(&imgs, 11);
    try std.testing.expectEqual(vs.prefixKey(&.{storeImage(im0.img)}, 11), key);
    try std.testing.expect(!std.mem.eql(u8, &zero, &key));
    try std.testing.expectEqual(key, try imageKey(&imgs, 40));
    var many: [vs.max_images + 1]lanes.stream.Image = @splat(im0.img);
    try std.testing.expectError(error.TooManyImages, imageKey(&many, 40));
    // a state kept at 40 by a request with another image (one digest bit), or with none: refused before any rank
    // moves, so the pass starts at 0 with every image's rows reserved
    var other = im0.img;
    other.digest[0] ^= 0x80;
    for ([_]vs.Digest{ zero, try imageKey(&.{other}, 40) }) |kept_key| {
        var k: ic.snap.Kept = .{ .tag = 1, .at = 40, .off = 0, .bytes = 0, .images = kept_key };
        var s = try lanes.Stream.init(a, .{ .id = "img", .prompt = &ids, .max_new = 8, .images = &imgs, .reuse = .{ .saved = &k, .at = 40 } });
        defer s.deinit(a);
        try std.testing.expect(!Lanes.sameImages(&s, &k));
        const lane = try l.opening(&s);
        try std.testing.expect(s.reuse_failed);
        try std.testing.expectEqual(@as(?anyerror, error.ImageDigestDiffers), s.reuse_error);
        try std.testing.expectEqual(@as(usize, 0), lane.from);
        try std.testing.expectEqual(@as(u32, 6), hv.r.store.committed);
        l.dropLane(&s);
    }
    // the same images: the key agrees (the resume then goes on to its pages and restore, on the device)
    const k: ic.snap.Kept = .{ .tag = 1, .at = 40, .off = 0, .bytes = 0, .images = key };
    var s = try lanes.Stream.init(a, .{ .id = "img", .prompt = &ids, .max_new = 8, .images = &imgs });
    defer s.deinit(a);
    try std.testing.expect(Lanes.sameImages(&s, &k));
    // a text request may resume an image request's state kept before its first image (both keys zeros)
    const before: ic.snap.Kept = .{ .tag = 1, .at = 10, .off = 0, .bytes = 0, .images = try imageKey(&imgs, 10) };
    var t = try lanes.Stream.init(a, .{ .id = "text", .prompt = &ids, .max_new = 8 });
    defer t.deinit(a);
    try std.testing.expect(Lanes.sameImages(&t, &before));
}

/// A store of kept states over rank 0's allocator for `reclaimChain` (host tests): the states in the store's order
/// (least valuable first), each holding its pages once; `protected`: one a queued request will resume (the lane
/// host's guard); `borrowed`: a resume in flight.
const FakeStore = struct {
    a: *ic.pages.Alloc,
    states: []State,

    const State = struct { ids: []const u32, protected: bool = false, borrowed: bool = false, gone: bool = false };

    fn giveBack(ctx: *anyopaque, restoring: ?*anyopaque, left: *anyopaque, fits: *const fn (*anyopaque) bool, guarded: bool) bool {
        const f: *FakeStore = @ptrCast(@alignCast(ctx));
        for (f.states) |*st| {
            if (fits(left)) return true;
            if (st.gone or st.borrowed or (guarded and st.protected)) continue;
            if (restoring) |r| if (@as(*anyopaque, st) == r) continue;
            for (st.ids) |id| f.a.release(id);
            st.gone = true;
        }
        return fits(left);
    }

    fn borrowedFn(_: *anyopaque, saved: *anyopaque) bool {
        const st: *const State = @ptrCast(@alignCast(saved));
        return st.borrowed;
    }

    fn hooks(f: *FakeStore) StoreHooks {
        return .{ .ctx = f, .give_back = giveBack, .borrowed = borrowedFn };
    }

    /// Admission's protection (`Lanes.reclaimable`): the borrowed states' pages.
    fn protect(f: *const FakeStore, out: [][]const u32) [][]const u32 {
        var n: usize = 0;
        for (f.states) |st| if (!st.gone and st.borrowed) {
            out[n] = st.ids;
            n += 1;
        };
        return out[0..n];
    }
};

test "inkling2 lanes: on an idle server the open frees every page admission counted, states queued requests would resume too (A1 audit D1)" {
    const gpa = std.testing.allocator;
    var a = try ic.pages.Alloc.init(gpa, 36);
    defer a.deinit();
    // three streams ran (12, 12 and 4 pages) and each kept a state on all its pages, then ended: A will be resumed by
    // a queued request (the lane host's guard protects it), B nobody needs, C a resume in flight borrows
    var l: [4]ic.pages.Lease = @splat(.{});
    defer for (&l) |*x| a.finish(x);
    var ids: [3][]u32 = undefined;
    for (&ids, [_]usize{ 12, 12, 4 }, l[0..3]) |*x, n, *lease| {
        try a.promise(lease, n);
        try a.cover(lease, n);
        x.* = try gpa.dupe(u32, lease.ids.items);
        for (x.*) |id| a.retain(id);
    }
    defer for (ids) |x| gpa.free(x);
    for (l[0..3]) |*x| a.finish(x);
    try std.testing.expect(a.idle());
    try std.testing.expectEqual(@as(usize, 8), a.available());
    var states = [_]FakeStore.State{ .{ .ids = ids[0], .protected = true }, .{ .ids = ids[1] }, .{ .ids = ids[2], .borrowed = true } };
    var store: FakeStore = .{ .a = &a, .states = &states };
    const mark = try gpa.alloc(u8, a.total());
    defer gpa.free(mark);

    // a new request needing 24 pages: admission lets it in (8 free + 24 that only A and B hold; C's 4 protected)
    const n: usize = 24;
    var buf: [4][]const u32 = undefined;
    const reclaimable = a.reclaimable(store.protect(&buf), mark);
    try std.testing.expectEqual(@as(usize, 24), reclaimable);
    try std.testing.expect(a.fits(n -| reclaimable));

    // other streams running: the guarded chain gives B back and stops short (A stays for its queued request; the
    // lane host requeues this one until a running stream ends)
    const busy = reclaimChain(&a, store.hooks(), null, n, false);
    try std.testing.expect(!busy.ok and !busy.unguarded);
    try std.testing.expect(states[1].gone and !states[0].gone and !states[2].gone);
    try std.testing.expectEqual(@as(usize, 20), a.available());
    // D1: before the fix this was the open's end on an idle server: LaneMemory, and the lane host, with nothing else
    // running, failed the request as out of memory
    try std.testing.expectError(error.LaneMemory, a.promise(&l[3], n));

    // no other stream runs: the chain goes past the guard and A gives its pages back; C (borrowed) never does
    const idle = reclaimChain(&a, store.hooks(), null, n, true);
    try std.testing.expect(idle.ok and idle.unguarded);
    try std.testing.expect(states[0].gone and !states[2].gone);
    try std.testing.expectEqual(@as(usize, 32), a.available());
    try a.promise(&l[3], n);
    try a.cover(&l[3], n);
    try std.testing.expectEqual(n, l[3].ids.items.len);
    for (l[3].ids.items) |id| try std.testing.expect(std.mem.indexOfScalar(u32, ids[2], id) == null); // C's pages untouched
    a.finish(&l[3]);
    for (ids[2]) |id| a.release(id);
    try std.testing.expect(a.idle() and a.available() == a.total());
}

test "inkling2 lanes: the reclaim chain never gives back the state the stream restores, guarded or not" {
    const gpa = std.testing.allocator;
    var a = try ic.pages.Alloc.init(gpa, 16);
    defer a.deinit();
    var l: ic.pages.Lease = .{};
    defer a.finish(&l);
    try a.promise(&l, 10);
    try a.cover(&l, 10);
    const ids = try gpa.dupe(u32, l.ids.items);
    defer gpa.free(ids);
    for (ids) |id| a.retain(id);
    a.finish(&l);
    var states = [_]FakeStore.State{.{ .ids = ids, .protected = true }};
    var store: FakeStore = .{ .a = &a, .states = &states };
    const r = reclaimChain(&a, store.hooks(), &states[0], 12, true);
    try std.testing.expect(!r.ok and r.unguarded and !states[0].gone);
    try std.testing.expectEqual(@as(usize, 6), a.available());
    for (ids) |id| a.release(id);
}
