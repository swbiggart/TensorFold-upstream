//! One rank's Inkling-2 serving engine (`full` mode): the rank's weights of layers [0, N), one activation set for
//! forwards of up to `prompt_rows` rows, and a pool of per-stream states (slots) carried across forwards: each
//! layer's keys and values (global layers a slot a position, sliding layers a ring of 1023 + prompt_rows), its four
//! convolutions' last 3 input rows. The block residual is per row and lives in the forward's activations only.
//!
//! A forward is R rows of one slot at positions pos0 .. pos0 + R - 1 (a prompt chunk of up to 2048 rows, or one
//! decode row). pos0 = 0 starts a new sequence on the slot (its convolution states are zeroed). At W > 1 ranks 1 ..
//! W-1 hold the embedding by vocabulary rows and every rank gets the forward's embedded rows by one gather
//! (cuda_embed.zig; at W = 1 rank 0 holds the table and embeds the rows itself); every rank runs the same layers
//! with the two rank-order sums a layer (cuda_tp.zig). The head is split by vocabulary (plan.zig, vocab_rows):
//! every rank turns the chosen rows into bf16 logits for its share, the shares are all-gathered in rank order, and
//! rank 0 softcaps them and copies them to the host (`headRows`, `logits`); a column's sum does not depend on the
//! split, so these are the single head's logits bit for bit. The followers' side (what step comes next) is
//! cuda_follow.zig.
//!
//! TF_INK2_LAYERS=N (debug): only layers [0, N), then the head over the blocks completed so far (the text is
//! garbage, but a decode step's logits must still equal the same row of a whole-prompt forward).
//! TF_INK2_MODE=full|checkpoint: the experts' arithmetic (cuda_moe.zig); part of the settings every rank agrees on.
//! TF_INK2_DIGESTS=FILE|none: the content-digest manifest (digests.zig, `tf-inkling2-plan --digests`) the loaded
//! bytes must match; default <model dir>/inkling2-digests-w<W>.tsv, which W > 1 needs unless `none`. The ranks
//! agree on its text before loading; each hashes its bytes as they upload and checks them against its own lines.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ops = @import("cuda_ops.zig");
const kernels = @import("cuda_kernels.zig");
const ic_attn = @import("cuda_attn.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const res = @import("cuda_res.zig");
const head = @import("cuda_head.zig");
const tp = @import("cuda_tp.zig");
const dspark = @import("cuda_dspark.zig");
const moe = @import("cuda_moe.zig");
const fuse = @import("cuda_fuse.zig");
const scratch = @import("cuda_scratch.zig");
const conv_rows = @import("cuda_conv.zig").state_rows;
const synth_fill = @import("cuda_synth.zig");
const pages = @import("cuda_pages.zig");
const embed = @import("cuda_embed.zig");

const Mat = ops.Mat;
const H = weights.hidden;

/// Rows of a prompt chunk (the widest forward).
pub const prompt_rows: usize = 2048;
/// Rows `headRows` turns into logits at once: a shared round's widest forward (cuda_lanes.zig batch_rows) and a
/// captured one's (`graph_rows`); rank 0's fp32 logits and their host copy are this many rows (0.15 GiB less than
/// at 128 on the binding rank).
pub const head_rows: usize = 32;
/// Every head row (`Engine.want_rows`).
pub const all_rows: u32 = std.math.maxInt(u32);
comptime {
    std.debug.assert(head_rows <= 32); // a bit a row in `want_rows`
}
/// Segments (streams) a batched forward carries at most (`forwardMany`, the step header's room).
pub const max_segs = fwd.max_segs;
/// Rows a captured forward carries at most (the step header's token ids), and a segment of one at most (a verify
/// window: the anchor and 7 drafts).
pub const graph_rows: usize = 32;
pub const graph_seg_rows: usize = state.spec_rows;
/// Device bytes of one unit of the graph cache (`Key.units`), which admission counts for every unit the cache may
/// hold (cuda_native.zig rooms). Measured on GB10 (cuMemGetInfo around a capture's instantiate and upload): a forward
/// of one stream's rows 3-24 MiB (1,738-1,792 nodes), of 4 streams' 26-35 MiB (3,724), of 8 streams' 45-58 MiB
/// (6,294); so a key of n segments takes 1 + n / 4 units.
pub const graph_budget_bytes: usize = 32 << 20;

/// TF_INK2_GRAPHS_MAX's default: every one-stream key at 8 slots (verify windows of 1 to 8 rows a slot), so a lone
/// stream's windows never recapture at --parallel 8; several streams' keys take the least recently used places.
pub const graphs_default: usize = 8 * graph_seg_rows;

/// TF_INK2_GRAPHS_MAX: the graph cache's size in units (`Key.units`; the least recently used graphs replaced; default
/// `graphs_default`, 2 GiB). Part of the settings the ranks agree on, so every rank's cache holds the same graphs.
pub fn graphsMax() usize {
    const v = std.c.getenv("TF_INK2_GRAPHS_MAX") orelse return graphs_default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch graphs_default;
}

/// The graph cache's units for a server of `streams` slots (`Engine.reserve`; admission keeps that many units out of
/// the streams' room, cuda_native.zig): TF_INK2_GRAPHS_MAX when set; else, for one stream, its every key (slot 0 with
/// a verify window of 1 to graph_seg_rows rows, one unit each, so nothing is ever replaced), and `graphs_default`
/// for more.
pub fn graphUnits(streams: u32) usize {
    if (std.c.getenv("TF_INK2_GRAPHS_MAX") != null) return @max(graphsMax(), 1);
    return if (streams <= 1) graph_seg_rows else graphs_default;
}

/// Keys of several segments rank 0 remembers having run eagerly (`Engine.graphWanted`: the second sighting captures).
const seen_max: usize = 64;

pub const Options = struct {
    rank: u32 = 0,
    world: u32 = 1,
    /// Layers [0, layers); 0: every layer (TF_INK2_LAYERS when 0 here).
    layers: u32 = 0,
    /// Positions a slot holds.
    context: usize,
    scratch_gib: usize = scratch.default_gib,
    /// W > 1: the ranks' NCCL group (owned by the caller).
    group: ?*const cuda.tp_group.Group = null,
    /// The content-digest manifest: a path, "none" (contents not checked), or null: TF_INK2_DIGESTS, else
    /// <dir>/inkling2-digests-w<W>.tsv (W > 1 needs one).
    digests: ?[]const u8 = null,
    /// Whole experts a rank, or split (a 1/W row slice of every expert, checkpoint mode only); null: TF_INK2_EXPERTS.
    experts: ?ink.plan.Layout = null,
    /// The DSpark drafter (mtp.safetensors) split over the ranks; TF_INK2_DSPARK=1 when false here.
    drafter: bool = false,
    /// Tooling (tf-inkling2-decode --solo): rank `rank` of a W-rank plan alone, no group. Its weights and kernels
    /// are the real rank's; its exchanges move nothing, so its outputs are not the model's (kernel timing only).
    solo: bool = false,
};

/// TF_INK2_SYNC_UPLOADS=1 (rank 0): the step header and a forward's token ids go up after a wait for the device and
/// with a blocking copy (the stage-2a way, for A/B); else as pinned copies queued on the stream. The same bytes land
/// before the same kernels either way.
fn syncUploads() bool {
    const v = std.c.getenv("TF_INK2_SYNC_UPLOADS") orelse return false;
    return v[0] == '1';
}

/// TF_INK2_GPU_ARGMAX (rank 0): `0` runs no argmax on the device (every head's fp32 logits go to the host, as
/// stage 2c did; for A/B); anything else (or unset) adds it after the softcap in every head (B2 C4), and a head whose
/// rows are all drawn greedily (`Engine.want_logits` false) downloads only the argmax rows.
fn gpuArgmax() bool {
    const v = std.c.getenv("TF_INK2_GPU_ARGMAX") orelse return true;
    return !(v[0] == '0' and v[1] == 0);
}

/// TF_INK2_DSPARK=1: load the drafter.
pub fn drafterFromEnv() bool {
    const v = std.c.getenv("TF_INK2_DSPARK") orelse return false;
    return v[0] == '1';
}

/// TF_INK2_MODE, refusing a name that is not a mode (cuda_moe's envMode would read it as full).
pub fn modeFromEnv() !moe.Mode {
    const v = std.c.getenv("TF_INK2_MODE") orelse return .full;
    return std.meta.stringToEnum(moe.Mode, std.mem.span(v)) orelse {
        std.log.err("inkling2: TF_INK2_MODE={s}: give full or checkpoint", .{std.mem.span(v)});
        return error.BadInk2Mode;
    };
}

/// The manifest's text (see Options.digests), or null when contents go unchecked.
fn readManifest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !?[]u8 {
    const given: ?[]const u8 = o.digests orelse if (std.c.getenv("TF_INK2_DIGESTS")) |v| std.mem.span(v) else null;
    if (given) |p| {
        if (std.mem.eql(u8, p, "none")) {
            std.log.warn("inkling2 rank {d}: weight contents not checked (digests: none)", .{o.rank});
            return null;
        }
        return std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 28)) catch |err| {
            std.log.err("inkling2 rank {d}: digest manifest {s}: {t}", .{ o.rank, p, err });
            return err;
        };
    }
    const p = try std.fmt.allocPrint(gpa, "{s}/inkling2-digests-w{d}.tsv", .{ dir, o.world });
    defer gpa.free(p);
    return std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(1 << 28)) catch |err| switch (err) {
        error.FileNotFound => {
            if (o.world == 1) return null;
            std.log.err("inkling2 rank {d}: no content-digest manifest at {s}: set TF_INK2_DIGESTS=FILE (tf-inkling2-plan --dir <source> --world {d} --digests FILE) or TF_INK2_DIGESTS=none", .{ o.rank, p, o.world });
            return error.NoDigestManifest;
        },
        else => return err,
    };
}

/// TF_INK2_LAYERS as a layer count (0: unset, every layer).
pub fn layersFromEnv() !u32 {
    const v = std.c.getenv("TF_INK2_LAYERS") orelse return 0;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch error.BadInk2Layers;
}

/// What every rank must agree on before loading: the executable, the checkpoint and the engine's settings.
const Fingerprint = extern struct {
    program: [32]u8,
    checkpoint: [32]u8,
    settings: [32]u8,
    rank: u32,
    world: u32,
};

/// One step as it crosses to the other ranks (cuda_follow.zig): 256 bytes. `head_n` > 0: the head over rows
/// head_lo .. head_lo + head_n of the forward (after it, or of the last one), on every rank. `nseg` > 0: a batched
/// forward (`forwardMany`) of segs[0 .. nseg) in that row order (slot, rows, pos0 unused). `graph` 1: a forward of
/// segs[0 .. nseg) run as a captured CUDA graph (`graphStep`) with the head over all its rows; rank 0 decides, so
/// every rank captures and replays the same forwards. A graph reads its inputs from the header's device copy: each
/// segment's first position (`segPosAt`) and the forward's token ids (`tokens`, rank 0's embedding). `seq` counts
/// the headers rank 0 has sent; a follower checks it against its own count.
pub const Header = extern struct {
    kind: u32,
    slot: u32 = 0,
    rows: u32 = 0,
    magic: u32 = magic_word,
    pos0: u64 = 0,
    head_lo: u32 = 0,
    head_n: u32 = 0,
    graph: u32 = 0,
    token: u32 = 0,
    nseg: u32 = 0,
    seq: u32 = 0,
    pad: [2]u64 = @splat(0),
    segs: [max_segs]Seg = @splat(.{}),
    /// A graph step's token ids in row order (rank 0's), read on the device by its embedding.
    tokens: [graph_rows]i32 = @splat(0),

    /// A batched forward's segment as it crosses (positions stay below 2^31, slots below 2^16).
    pub const Seg = extern struct {
        slot: u16 = 0,
        rows: u16 = 0,
        pos0: u32 = 0,
    };

    /// Byte offsets a captured forward reads on the device: segment i's first position (an i32: positions stay
    /// below 2^31) and the token ids.
    pub fn segPosAt(i: usize) u64 {
        return @offsetOf(Header, "segs") + i * @sizeOf(Seg) + @offsetOf(Seg, "pos0");
    }
    pub const tokens_at = @offsetOf(Header, "tokens");

    pub const magic_word: u32 = 0x324b4e49; // "INK2"

    comptime {
        std.debug.assert(@sizeOf(Header) == 256 and segPosAt(0) == 64 + 4 and segPosAt(7) == 64 + 56 + 4 and tokens_at == 128);
        std.debug.assert(@import("builtin").cpu.arch.endian() == .little);
    }

    /// A graph step's token ids (rows in segment order) into the header.
    pub fn setTokens(h: *Header, ids: []const u32) !void {
        if (ids.len == 0 or ids.len > graph_rows) return error.BadRows;
        for (ids, h.tokens[0..ids.len]) |t, *o| o.* = @intCast(t);
    }

    /// A batched forward's segments into the header.
    pub fn setSegs(h: *Header, segs: []const Segment) !void {
        if (segs.len == 0 or segs.len > fwd.max_segs) return error.BadSegments;
        for (segs, h.segs[0..segs.len]) |g, *o| {
            if (g.slot > std.math.maxInt(u16) or g.rows > std.math.maxInt(u16) or g.pos0 > std.math.maxInt(i32)) return error.BadSegments;
            o.* = .{ .slot = @intCast(g.slot), .rows = @intCast(g.rows), .pos0 = @intCast(g.pos0) };
        }
        h.nseg = @intCast(segs.len);
    }

    /// The header's segments (`nseg` of them) into `out`.
    pub fn getSegs(h: *const Header, out: *[fwd.max_segs]Segment) ![]const Segment {
        if (h.nseg == 0 or h.nseg > fwd.max_segs) return error.BadStep;
        for (h.segs[0..h.nseg], out[0..h.nseg]) |g, *o| o.* = .{ .slot = g.slot, .rows = g.rows, .pos0 = g.pos0 };
        return out[0..h.nseg];
    }
};

/// Rows `row + 1 .. row + n` of a forward's token ids: stream `y`'s drafts 1 .. n of the last `draftMany` (its
/// window's first row, `row`, is the anchor, on the host).
pub const WalkIds = struct { row: u32, y: u32, n: u32 };

/// One stream's draft in `draftMany`: its slot (kept already) and the anchor token at its position.
pub const DraftItem = struct {
    slot: u32,
    anchor: u32,
};

/// One stream's rows of a batched forward (`forwardMany`): `rows` rows of slot `slot` from position pos0.
pub const Segment = struct {
    slot: u32,
    pos0: usize,
    rows: usize,
};

/// What a captured forward is replayed for: its segments' slots and rows in row order (the positions and tokens
/// are read on the device). Unused entries are zero.
pub const Key = struct {
    n: u8 = 0,
    slots: [max_segs]u16 = @splat(0),
    rows: [max_segs]u8 = @splat(0),
    /// Bit i: segment i is one row of a stream whose pages are one run, so its global layers run the ring kernels on
    /// that run (State.kvFor); a graph is captured for each mode (`Engine.keyOf`).
    ring: u32 = 0,

    pub fn of(segs: []const Segment) !Key {
        if (segs.len == 0 or segs.len > max_segs) return error.BadSegments;
        var k: Key = .{ .n = @intCast(segs.len) };
        for (segs, 0..) |g, i| {
            if (g.slot > std.math.maxInt(u16) or g.rows == 0 or g.rows > graph_seg_rows) return error.BadSegments;
            k.slots[i] = @intCast(g.slot);
            k.rows[i] = @intCast(g.rows);
        }
        return k;
    }

    /// The cache units (graph_budget_bytes each) a captured forward of this key takes: 1 + segments / 4.
    pub fn units(k: Key) usize {
        return 1 + @as(usize, k.n) / 4;
    }

    pub fn eql(a: Key, b: Key) bool {
        return a.n == b.n and a.ring == b.ring and std.mem.eql(u16, &a.slots, &b.slots) and std.mem.eql(u8, &a.rows, &b.rows);
    }

    pub fn format(k: Key, w: *std.Io.Writer) std.Io.Writer.Error!void {
        for (0..k.n) |i| try w.print("{s}slot {d} x{d}{s}", .{ if (i > 0) ", " else "", k.slots[i], k.rows[i], if ((k.ring >> @intCast(i)) & 1 != 0) " on its run" else "" });
    }
};

/// The served streams' slots and the pool's pages (cuda_native.zig count, rank 0).
pub const PoolPlan = struct { slots: u32, pages: usize };

/// `Engine.mark`'s copy: the convolution states (device) and the slot's state as it stood.
pub const Marked = struct { buf: cuda.DeviceBuffer, slot: u32, st: state.State };

/// Forwards run since the engine started, by kind (rank 0 prints them; every rank counts the same).
pub const Counts = struct {
    graph: u64 = 0,
    eager: u64 = 0,
    captured: u64 = 0,
    evicted: u64 = 0,
    /// Device bytes the captures took (each measured around its instantiate and upload; cuMemGetInfo).
    graph_bytes: i64 = 0,
    capture_ms: f64 = 0,
    /// Streams on pages, one-row forwards (a stream's row each): through the ring kernels on the stream's run, or
    /// through the page table because its pages are not one run (the fallback); a stream's one-row mode changing
    /// between two of them; captured nodes pointed at a stream's new run on replay, and the replays that needed it.
    one_ring: u64 = 0,
    one_paged: u64 = 0,
    switches: u64 = 0,
    repointed: u64 = 0,
    repoint_replays: u64 = 0,
    /// Rank 0's head tail (`headDone`, after the device is done): heads, their rows, and the ns their downloads took
    /// (fp32 logits, or the argmax rows alone), by whether the logits came down.
    heads: [2]u64 = .{ 0, 0 },
    head_rows: [2]u64 = .{ 0, 0 },
    head_ns: [2]u64 = .{ 0, 0 },
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    o: Options,
    layers: u32,
    model: ink.describe.Model,
    stream: cuda.Stream,
    k: kernels.Kernels,
    gemms: ops.Gemms,
    scratch: ops.Scratch,
    w: weights.Weights,
    act: state.Act,
    parts: ?tp.Parts = null,
    x: tp.Exchange = undefined,
    slots: std.ArrayList(?*state.State) = .empty,
    /// The head's rows and the ranks' bf16 logit shares as gathered ([world][n][share]); rank 0: the softcapped
    /// logits and their host copy.
    hrows: Mat = .{},
    hparts: u64 = 0,
    hout: Mat = .{},
    hbufs: [4]?cuda.DeviceBuffer = .{ null, null, null, null },
    host: []f32 = &.{},
    /// Rank 0 (B2 C4, TF_INK2_GPU_ARGMAX): each head row's argmax on the device (hbufs[3], u32 [head_rows], written
    /// after the softcap in every head, graphs included) and its host copy.
    argmax_on: bool = false,
    host_max: [head_rows]u32 = undefined,
    /// Rank 0: whether the next head's fp32 logits go to the host (`logits`); false (a forward whose every row is
    /// drawn greedily, set by the lanes for it) downloads only the argmax rows (`argmaxRows`). Followers never
    /// download, so this is rank 0's alone and changes no step.
    want_logits: bool = true,
    /// Rank 0: with `want_logits`, the head rows whose fp32 logits go to the host (bit r: row r; the lanes clear the
    /// rows they draw from the device's argmax, so a structured stream's rows bring down only their own logits).
    want_rows: u32 = all_rows,
    /// Rows of the last head (rank 0's host copy holds them; 0: none since the last forward), whether any of its
    /// logits are on the host, and which rows' (`logitsRow`).
    head_n: usize = 0,
    head_logits: bool = false,
    head_have: u32 = 0,
    hdr: cuda.DeviceBuffer,
    /// The last forward's rows (headRows reads them).
    rows: usize = 0,
    load_s: f64 = 0,
    dropped_bytes: u64 = 0,
    /// Rank 0: the other ranks sleep on their liveness lines (cuda_follow.zig); the next step wakes them first.
    asleep: bool = false,
    /// The DSpark drafter (every rank its share), when loaded.
    drafter: ?*dspark.Drafter = null,
    /// Captured forwards (`graphStep`), at most `graph_max` units of them (`Key.units`; least recently used replaced).
    graphs: std.ArrayList(Graph) = .empty,
    graph_max: usize = graphs_default,
    graph_units: usize = 0,
    graph_tick: u64 = 0,
    /// Rank 0: the last `seen_max` keys of several segments run eagerly on their first sighting (a ring).
    seen: [seen_max]Key = undefined,
    seen_n: usize = 0,
    counts: Counts = .{},
    /// Tooling (`mark` / `rewind`): one slot's convolution states and position kept aside.
    marked: ?Marked = null,
    /// Rank 0: headers sent (`post`); ranks above: headers received (`receive`).
    seq: u32 = 0,
    /// Forwards run (each writes the drafter's taps), and drafts made from another forward's taps than their
    /// slot's last one (stale: such a draft changes no output, only what it proposes).
    forwards: u64 = 0,
    stale_drafts: u64 = 0,
    /// Rank 0: rows of the next forward's token ids that the last `draftMany` made on the device (a verify window
    /// drafted in the same step, Leader.forwardDrafted): copied from the drafter's walk into the ids the forward
    /// reads (`tokens`, or the step header's for a graph) once the host's ids are there; then cleared.
    walk_ids: [max_segs]WalkIds = undefined,
    walk_n: usize = 0,
    /// Tooling (tf-inkling2-decode's budget): the forwards run while `prof_on` (eager ones; rank 0's step header
    /// carries it to every rank) add their per-op stream times to `prof` (cuda_forward.zig Prof). Events only: the
    /// same kernels, the same bits.
    prof: ?*fwd.Prof = null,
    prof_on: bool = false,
    /// The prompt cache's arena (cuda_snap.zig), made at start-up when the server keeps prompts.
    snap: @import("cuda_snap.zig").Arena = .{},
    /// Global layers on pages (TF_INK2_PAGES, on by default; every rank the same): the served streams' pool, made by
    /// `reserve` (every slot on it), or without one each slot's own pages (tooling).
    paged: bool = false,
    pool: ?*pages.Pool = null,
    /// Tooling (tf-inkling2-decode, `toolPool`): the pool is the tool's, each slot taking its whole context's pages
    /// when it first runs (rank 0's Leader.toolPages), so kept prompt states can be resumed on another slot.
    tool_pool: bool = false,
    /// Rank 0, after the stream count (cuda_native.zig): the slots and the pool's pages `reserve` makes.
    plan: ?PoolPlan = null,
    /// W > 1 with a group: the embedding by vocabulary rows over ranks 1 .. W-1 (cuda_embed.zig); null: rank 0 holds
    /// the whole table (W = 1, --solo).
    shared: ?embed.Shared = null,
    /// The shared embedding: the next eager forward's token ids are in the step header on every rank (Leader sets
    /// them for a forward of at most `graph_rows` rows and the header's `token` says so), so no broadcast of them.
    ids_in_header: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, opts: Options) !*Engine {
        var o = opts;
        if (o.world == 0 or o.rank >= o.world) return error.BadRank;
        if (o.world > 1 and o.group == null and !o.solo) return error.NoGroup;
        if (o.layers == 0) o.layers = try layersFromEnv();
        if (!o.drafter) o.drafter = drafterFromEnv();
        const mode = try modeFromEnv();
        const spread = o.world > 1 and o.group != null; // the embedding off rank 0 (cuda_embed.zig)
        const manifest = try readManifest(gpa, io, dir, o);
        defer if (manifest) |t| gpa.free(t);
        if (o.experts == null) o.experts = ink.plan.Layout.env();
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var dg: ink.Diag = .{};
        var model = ink.describe.Model.fromDir(gpa, io, dir, .{ .world = o.world, .layout = o.experts.? }, &dg) catch |err| {
            std.log.err("inkling2: {s}: {s}", .{ dir, dg.text() });
            return err;
        };
        errdefer model.deinit();
        const total = model.c.text.layers;
        if (o.layers == 0) o.layers = total;
        if (o.layers > total) return error.BadInk2Layers;
        if (o.context == 0) return error.BadContext;
        var stream = try cuda.Stream.init(d, true);
        errdefer stream.deinit();
        var hdr = try cuda.DeviceBuffer.alloc(d, @sizeOf(Header));
        errdefer hdr.free();
        e.* = .{ .gpa = gpa, .io = io, .d = d, .o = o, .layers = o.layers, .model = model, .stream = stream, .k = undefined, .gemms = undefined, .scratch = undefined, .w = undefined, .act = undefined, .hdr = hdr, .graph_max = @max(graphsMax(), 1), .paged = pages.enabled() };
        if (o.group) |g| try e.agree(g, dir, manifest, mode);
        e.k = try kernels.Kernels.load(d);
        errdefer e.k.unload();
        {
            var cb: [160]u8 = undefined;
            std.debug.print("{s}\n", .{e.k.attn.census(&cb)}); // the form census, on every rank (gates grep it)
        }
        e.k.moe.mode = mode; // the mode the ranks agreed on
        try e.checkExperts();
        // cuBLASLt for ops.Ctx.linear, which no forward calls (the projections take the lane matmul); not required
        e.gemms = ops.Gemms.init(gpa, d) catch ops.Gemms{ .gpa = gpa };
        errdefer e.gemms.deinit();
        e.scratch = try ops.Scratch.init(d, o.scratch_gib << 30);
        errdefer e.scratch.deinit();
        const clock = std.Io.Clock.awake;
        const t0 = clock.now(io).toNanoseconds();
        const leader = o.rank == 0;
        var got: std.ArrayList(ink.digests.Digest) = .empty;
        defer got.deinit(gpa);
        e.w = try weights.load(gpa, io, d, dir, &e.model, .{ .rank = o.rank, .lo = 0, .hi = o.layers, .embed = leader, .head = true, .digests = if (manifest != null) &got else null }, &prepare_dropping);
        errdefer e.w.deinit();
        // every forward's op scratch at this context (cuda_scratch.zig: the attention core and the other ops' takes,
        // eager up to prompt_rows rows and captured up to a verify window), on every rank, before a step can run
        try scratch.checkStartup(&e.scratch, &e.k, &e.w, o.context, prompt_rows, graph_seg_rows, o.rank);
        if (o.drafter) {
            const vn = @as(usize, e.w.vocab) / o.world;
            const same = e.w.unembed != 0 and e.w.vocab_rows.lo == o.rank * vn and e.w.vocab_rows.hi == (o.rank + 1) * vn;
            e.drafter = try dspark.load(gpa, io, d, dir, &e.model, .{ .rank = o.rank, .world = o.world, .group = o.group, .context = o.context + dspark.block, .cap = prompt_rows, .lane = &e.k.attn.lane, .unembed = if (same) e.w.unembed else 0 });
            e.drafter.?.paged = e.paged; // its global layer on the slots' pages (stage 2c)
        }
        errdefer if (e.drafter) |dr| dr.deinit();
        e.load_s = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-9;
        if (manifest) |text| {
            const c = try ink.digests.check(gpa, text, &e.model, o.rank, got.items);
            if (c.differ + c.missing > 0) {
                std.log.err("inkling2 rank {d}: {d} of {d} loaded pieces differ from the digest manifest and {d} have no matching line (first: {s})", .{ o.rank, c.differ, c.pieces, c.missing, c.first.? });
                return error.WeightBytesDiffer;
            }
            std.log.info("inkling2 rank {d}: weight contents match the digest manifest ({d} pieces, loaded xxh64 {x:0>16})", .{ o.rank, c.pieces, c.loaded });
        }
        // the embedding off rank 0: rank 0's checked table to the ranks that hold it, before any room is read
        if (spread) try embed.spread(d, e.stream, o.group.?, &e.w, o.rank, o.world);
        e.dropped_bytes = e.w.released;
        const l0 = e.w.layers[0];
        e.act = try state.Act.init(gpa, d, prompt_rows, l0.hq, l0.hk);
        errdefer e.act.deinit();
        if (o.world > 1) {
            e.parts = try tp.Parts.initFor(d, o.world, prompt_rows, o.experts.?);
            e.x = .{ .rank = o.rank, .world = o.world, .parts = &e.parts.?, .group = o.group };
            if (spread) e.shared = .{ .rank = o.rank, .world = o.world, .vocab = e.w.vocab, .slots = e.parts.?.ptr[0], .group = o.group };
        }
        errdefer if (e.parts) |*p| p.deinit();
        errdefer e.freeHead();
        {
            const share: usize = e.w.vocab_rows.len();
            e.hbufs[0] = try cuda.DeviceBuffer.alloc(d, head_rows * H * 2);
            e.hbufs[1] = try cuda.DeviceBuffer.alloc(d, @as(usize, o.world) * head_rows * share * 2);
            e.hrows = Mat.of(e.hbufs[0].?.ptr, head_rows, H, .bf16);
            e.hparts = e.hbufs[1].?.ptr;
            if (leader) {
                const U: usize = e.w.unpadded_vocab;
                e.hbufs[2] = try cuda.DeviceBuffer.alloc(d, head_rows * U * 4);
                e.hout = Mat.of(e.hbufs[2].?.ptr, head_rows, U, .f32);
                e.host = try gpa.alloc(f32, head_rows * U);
                e.argmax_on = gpuArgmax();
                if (e.argmax_on) e.hbufs[3] = try cuda.DeviceBuffer.alloc(d, head_rows * 4);
            }
        }
        std.log.info("inkling2 rank {d} of {d}: layers 0..{d}, embed rows {d}..{d}, unembed rows {d}..{d}: q heads {d}, kv heads {d}, experts {d}..{d}{s} ({t} mode): {d:.2} GiB of weights in {d:.1} s ({d:.2} GiB of bf16 attention copies dropped); a stream's state {d:.2} GiB at {d} positions", .{
            o.rank, o.world, o.layers - 1, e.w.embed_rows.lo, @as(i64, e.w.embed_rows.hi) - 1, e.w.vocab_rows.lo, e.w.vocab_rows.hi - 1, l0.hq, l0.hk, l0.experts.lo, l0.experts.hi - 1, if (l0.experts.split > 1) " (split: a row slice of each)" else "", e.k.moe.mode, gib(e.w.bytes), e.load_s, gib(e.dropped_bytes), gib(e.streamBytes()), o.context,
        });
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.marked) |*m| m.buf.free();
        for (e.graphs.items) |*g| g.deinit(e.gpa);
        e.graphs.deinit(e.gpa);
        e.snap.deinit(e.gpa);
        for (e.slots.items) |s| if (s) |st| {
            st.deinit();
            e.gpa.destroy(st);
        };
        e.slots.deinit(e.gpa);
        if (e.pool) |p| p.deinit();
        e.freeHead();
        if (e.drafter) |dr| dr.deinit();
        if (e.parts) |*p| p.deinit();
        e.act.deinit();
        e.w.deinit();
        e.scratch.deinit();
        e.gemms.deinit();
        e.k.attn.printExceptions(e.d); // kx's exception counters over the engine's life (real weights: rows 0)
        e.k.unload();
        e.hdr.free();
        e.stream.deinit();
        e.model.deinit();
        e.gpa.destroy(e);
    }

    fn freeHead(e: *Engine) void {
        for (&e.hbufs) |*b| if (b.*) |*x| {
            x.free();
            b.* = null;
        };
        if (e.host.len > 0) e.gpa.free(e.host);
        e.host = &.{};
    }

    /// Device bytes one more stream (slot) takes: at --context, or once the pool is planned a slot's fixed part and
    /// its share of the pool's pages.
    pub fn streamBytes(e: *const Engine) usize {
        if (e.plan) |p| return e.fixedBytesAt(e.o.context) + (p.pages * e.pageBytes() + p.slots - 1) / p.slots;
        return e.streamBytesAt(e.o.context);
    }

    /// Device bytes a stream would take in a slot of `context` positions (the drafter's caches hold a block more:
    /// dspark.load), for admission's largest context that fits (cuda_native.zig): on pages, its fixed part and the
    /// pages of a whole context.
    pub fn streamBytesAt(e: *const Engine, context: usize) usize {
        if (e.paged) return e.fixedBytesAt(context) + pages.pagesFor(context) * e.pageBytes();
        const extra: usize = if (e.drafter) |dr| dr.slotBytesAt(context + dspark.block) + state.histBytes(&e.w) else 0;
        return state.stateBytes(&e.w, context, prompt_rows) + extra;
    }

    /// On pages: a slot's bytes beside its pages at `context` positions (its rings, convolution states, rollback
    /// history, page table, and the drafter's caches).
    pub fn fixedBytesAt(e: *const Engine, context: usize) usize {
        const extra: usize = if (e.drafter) |dr| dr.slotBytesAt(context + dspark.block) + state.histBytes(&e.w) else 0;
        return state.fixedBytes(&e.w, context, prompt_rows) + extra;
    }

    /// On pages: a page's bytes on this rank, the drafter's global layer's two runs included (0 when not on pages).
    pub fn pageBytes(e: *const Engine) usize {
        return if (e.paged) state.pageBytesFor(&e.w, e.drafterHk()) else 0;
    }

    /// The drafter's key heads on this rank (0: no drafter).
    pub fn drafterHk(e: *const Engine) usize {
        return if (e.drafter) |dr| dr.hk else 0;
    }

    /// The pages' shape on this engine: the global layers' regions, and the drafter's two when it is loaded.
    pub fn pagedShape(e: *const Engine) !state.PagedShape {
        return (try state.pagedShapeFor(&e.w, e.drafterHk())) orelse error.NoGlobalLayers;
    }

    /// Positions a slot holds.
    pub fn capacity(e: *const Engine) usize {
        return e.o.context;
    }

    pub fn vocab(e: *const Engine) usize {
        return e.w.unpadded_vocab;
    }

    /// Slot `slot`'s state, made on first use: its global layers on the pool's pages (the served slots, none mapped
    /// until a stream opens), on pages of its own (tooling, every entry mapped), or in a window (TF_INK2_PAGES=0).
    pub fn bind(e: *Engine, slot: u32) !*state.State {
        if (slot >= e.slots.items.len) try e.slots.appendNTimes(e.gpa, null, slot + 1 - e.slots.items.len);
        if (e.slots.items[slot]) |s| return s;
        const storage: state.State.Storage = if (e.pool) |p| (if (slot < p.slots) .{ .pool = p.place() } else return error.NoSlot) else if (e.paged) .{ .own = .{ .shuffle = pages.shuffled(), .move = pages.moving(), .drafter_hk = e.drafterHk() } } else .window;
        if (storage == .own and slot == 0) std.log.info("inkling2 rank {d}: global layers on pages of their own a slot ({d} pages, {s} ids{s})", .{ e.o.rank, pages.pagesFor(e.o.context), if (storage.own.shuffle) "shuffled" else if (storage.own.move) "consecutive, a new run each sequence in two contexts' pages, every other 5 one-row forwards on the page table" else "consecutive", if (pages.poisoned()) ", NaN until written" else "" });
        const s = try e.gpa.create(state.State);
        errdefer e.gpa.destroy(s);
        s.* = try state.State.initOn(e.gpa, e.d, &e.w, e.o.context, prompt_rows, e.drafter != null, storage);
        e.slots.items[slot] = s;
        if (e.drafter) |dr| {
            // on pages the drafter's global layer is in the last two regions, at the slot's pages (its table)
            const pg: ?dspark.Caches.Paged = if (s.place) |pl| pg: {
                const g = try e.pagedShape();
                break :pg .{ .k = pl.region(g.drafterRegion(0)), .v = pl.region(g.drafterRegion(1)), .table = s.table, .page = pl.elems() };
            } else null;
            _ = try dr.bindOn(slot, pg);
        }
        return s;
    }

    /// States for slots 0 .. n - 1, made now (the server's admitted streams), so no later step allocates one; on
    /// pages, the pool of `npages` pages first (rank 0's `plan`, the others' from its reserve step), every slot on it.
    /// The graph cache then holds `graphUnits(n)` units, what admission kept for it (every rank gets the same n, so
    /// every rank's cache holds the same graphs).
    pub fn reserve(e: *Engine, n: u32, npages: usize) !void {
        if (e.paged and npages > 0 and e.pool == null) {
            for (e.slots.items) |s| if (s != null) return error.SlotTaken; // the pool comes before any slot
            const g = try e.pagedShape();
            e.pool = try pages.Pool.init(e.gpa, e.d, npages, g.regions, g.run, n, pages.pagesFor(e.o.context), e.o.rank == 0);
            std.log.info("inkling2 rank {d}: KV page pool {d} pages of {d} positions ({d} positions, {d:.2} GiB; {d} KiB a page, {d} global layers' keys and values{s}) for {d} slots, the largest request {d} positions{s}", .{ e.o.rank, npages, pages.page_rows, npages * pages.page_rows, gib(npages * e.pageBytes()), e.pageBytes() >> 10, g.globals, if (g.drafter) " and the drafter's" else "", n, @min(npages * pages.page_rows, e.o.context), if (e.pool.?.poison) "; TF_KV_POISON: a stream's pages NaN when mapped" else "" });
        }
        var slot: u32 = 0;
        while (slot < n) : (slot += 1) _ = try e.bind(slot);
        if (e.graphs.items.len == 0) e.graph_max = graphUnits(n);
        const each = if (e.pool != null) e.fixedBytesAt(e.o.context) else e.streamBytes();
        std.log.info("inkling2 rank {d}: {d} streams' states made ({d:.2} GiB{s}); the graph cache holds {d} units", .{ e.o.rank, n, gib(@as(u64, n) * each), if (e.pool != null) " beside the pool" else "", e.graph_max });
    }

    /// Tooling (every rank, before any slot): the pool for `slots` slots of the decode tool, each slot's whole context
    /// and `spare` pages more (what kept prompt states hold past their streams). No pool for the window layout or the
    /// own-pages orders (TF_INK2_PAGE_ORDER shuffle / move: a slot's own arena, no prompt cache).
    pub fn toolPool(e: *Engine, slots: u32, spare: usize) !void {
        if (!e.paged or e.pool != null or pages.shuffled() or pages.moving()) return;
        for (e.slots.items) |s| if (s != null) return error.SlotTaken;
        const g = try e.pagedShape();
        const entries = pages.pagesFor(e.o.context);
        const n = @as(usize, slots) * entries + spare;
        e.pool = try pages.Pool.init(e.gpa, e.d, n, g.regions, g.run, slots, entries, e.o.rank == 0);
        e.tool_pool = true;
        if (e.o.rank == 0) std.log.info("inkling2 rank 0: global layers on a tool pool: {d} pages for {d} slots of {d} pages and {d} spare ({d:.2} GiB, {d} KiB a page{s}){s}", .{ n, slots, entries, spare, gib(n * e.pageBytes()), e.pageBytes() >> 10, if (g.drafter) ", the drafter's global layer too" else "", if (e.pool.?.poison) "; TF_KV_POISON: a slot's own pages NaN when mapped" else "" });
    }

    /// Slot `slot` (on the pool) holds `ids` (position order; rank 0's open; the first `shared` adopted from a kept
    /// state): its host table, its device table queued on the stream, at W > 1 broadcast to the other ranks' same slot
    /// (whose host copies come back here), and under TF_KV_POISON its own pages set to NaN. Every rank calls it in the
    /// same step (`pages`); a follower passes null ids and `n` from the header.
    pub fn mapPages(e: *Engine, slot: u32, n: usize, ids: ?[]const u32, shared: usize) !void {
        const p = e.pool orelse return error.NoPool;
        if (slot >= p.slots) return error.NoSlot;
        const st = try e.bind(slot); // the served slots are made at start-up; a tool pool's on their first pages
        if (n == 0 or n > st.ids.len) return error.BadStep;
        if (ids) |x| {
            if (x.len != n) return error.BadStep;
            try st.mapIds(x);
            try st.enqueueTable(e.d, e.stream);
        }
        if (e.o.group) |g| try g.lib.check(g.lib.api.ncclBroadcast(st.table, st.table, st.ids.len * 4, .u8, 0, g.comm, e.stream.handle), "ncclBroadcast");
        if (ids == null) {
            // a follower: rank 0's table, then its host copy (the prompt cache's copies walk it)
            try e.stream.synchronize();
            const got = try e.gpa.alloc(u32, st.ids.len);
            defer e.gpa.free(got);
            try cuda.memory.download(e.d, st.table, std.mem.sliceAsBytes(got));
            try st.mapIds(got[0..n]);
            for (got[n..]) |v| if (v != p.place().scrub()) return error.BadStep;
        }
        if (shared > n) return error.BadStep;
        try p.poisonPages(e.d, e.stream, st.ids[shared..n]);
    }

    /// Rank 0: slot `slot`'s stream ended; its pages go back to the pool (nothing is sent: a follower's table is
    /// rewritten by the slot's next `pages` step before any forward of it).
    pub fn unmapPages(e: *Engine, slot: u32) void {
        const p = e.pool orelse return;
        if (slot < e.slots.items.len) if (e.slots.items[slot]) |st| {
            st.unmap(); // no forward of the slot until it opens again
        };
        p.close(slot);
        if (p.alloc) |a| if (a.idle()) {
            const c = e.counts;
            std.log.info("inkling2 pages idle: one-row stream forwards on the run's ring kernels {d}, on the page table (pages not one run) {d}, mode switches {d}; captured nodes pointed at a moved run {d} in {d} replays", .{ c.one_ring, c.one_paged, c.switches, c.repointed, c.repoint_replays });
        };
    }

    /// What `forward` checks before its first collective: the rows, the slot's state (made now if new) and the
    /// positions. Rank 0 runs it before a step's header goes out, so a failure here moves no rank.
    pub fn ready(e: *Engine, slot: u32, pos0: usize, R: usize) !*state.State {
        if (R == 0 or R > prompt_rows) return error.BadRows;
        const st = try e.bind(slot);
        if (pos0 + R > st.context) return error.PastContext;
        if (pos0 + R > st.limit()) return error.PastPages; // on the pool: past the pages the slot's stream holds
        if (pos0 != 0 and pos0 != st.pos) return error.OutOfStep;
        return st;
    }

    /// Tooling (tf-inkling2-decode --exp deep, every rank by a `mark` header): slot `slot`'s state where it stands
    /// kept aside, every layer's four convolution states and the slot's position and last forward, so `rewind` can
    /// take the slot back here after later forwards (one long prefill, then decode steps and a prompt forward over the
    /// same positions). The keys and values need nothing: a forward writes its positions' rows before any read, and
    /// rows past the mark are read only after they are written again.
    pub fn mark(e: *Engine, slot: u32) !void {
        if (slot >= e.slots.items.len) return error.UnknownSlot;
        const st = e.slots.items[slot] orelse return error.UnknownSlot;
        const bytes = convBytes(st);
        if (e.marked) |*m| if (m.buf.len < bytes) {
            m.buf.free();
            e.marked = null;
        };
        if (e.marked == null) e.marked = .{ .buf = try cuda.DeviceBuffer.alloc(e.d, bytes), .slot = slot, .st = st.* };
        const m = &e.marked.?;
        m.slot = slot;
        m.st = st.*;
        try convCopy(e, st, m.buf.ptr, true);
    }

    /// Tooling: slot `slot` back to where `mark` left it (see there).
    pub fn rewind(e: *Engine, slot: u32) !void {
        const m = e.marked orelse return error.NothingMarked;
        if (m.slot != slot or slot >= e.slots.items.len) return error.NothingMarked;
        const st = e.slots.items[slot] orelse return error.UnknownSlot;
        try convCopy(e, st, m.buf.ptr, false);
        const keep_layers = st.layers;
        const keep_own = st.own;
        st.* = m.st;
        st.layers = keep_layers;
        st.own = keep_own;
        st.last_rows = 0; // no keep before the next forward (its input history is a later forward's)
        st.kept_inputs = false;
        e.forwards += 1; // nor a draft: the taps are a later forward's
    }

    /// Tooling (tf-inkling2-decode --exp deep --fill synth, every rank by a `synth` header): slot `slot` as if
    /// positions 0 .. positions - 1 had run, its state written with synthetic values and no forward (cuda_synth.zig);
    /// the next forward starts at `positions`.
    pub fn synth(e: *Engine, slot: u32, positions: usize) !void {
        const st = try e.bind(slot);
        if (positions == 0 or positions >= st.context) return error.PastContext;
        const c = e.ctx();
        const caches: ?*const dspark.Caches = if (e.drafter) |dr| try dr.bind(slot) else null;
        try synth_fill.fill(e.gpa, &c, st, caches, e.o.rank, positions);
        st.pos = positions;
        st.last_pos0 = positions;
        st.last_rows = 0; // no keep before the next forward, nor a draft from earlier taps
        st.kept_inputs = false;
        e.forwards += 1;
    }

    /// Tooling: slot `slot`'s global layers' positions 0 .. positions - 1 read back against the `synth` fill (this
    /// rank's; logged), then every rank's verdict gathered (`ranks_bad`, the same on every rank), so a follower's
    /// unequal row fails rank 0's gate too.
    pub fn synthCheck(e: *Engine, slot: u32, positions: usize) !synth_fill.Checked {
        if (slot >= e.slots.items.len) return error.UnknownSlot;
        const st = e.slots.items[slot] orelse return error.UnknownSlot;
        const c = e.ctx();
        var got = try synth_fill.check(e.gpa, &c, st, e.o.rank, positions);
        if (got.bad) |p| {
            std.log.err("inkling2 rank {d}: FAIL synthetic fill: layer {d}'s row at position {d} differs from the fill ({d} global layers, {d:.2} GiB read)", .{ e.o.rank, got.bad_layer, p, got.layers, gib(got.bytes) });
        } else std.log.info("inkling2 rank {d}: PASS synthetic fill intact: positions 0 .. {d} of {d} global layers equal the fill ({d:.2} GiB read)", .{ e.o.rank, positions - 1, got.layers, gib(got.bytes) });
        got.ranks_bad = @intFromBool(got.bad != null);
        if (e.o.group) |g| {
            var all: [ink.plan.max_world]u32 = undefined;
            try g.gather(u32, e.d, e.stream, @intFromBool(got.bad != null), all[0..g.world]);
            got.ranks_bad = 0;
            for (all[0..g.world]) |b| got.ranks_bad += b;
        }
        if (got.ranks_bad > 0) std.log.err("inkling2 rank {d}: FAIL synthetic fill: {d} rank(s) found a row unequal", .{ e.o.rank, got.ranks_bad });
        return got;
    }

    fn convBytes(st: *const state.State) usize {
        var n: usize = 0;
        for (st.layers) |l| n += 2 * conv_rows * (l.kv.heads * weights.head_dim * 2 + H * 2);
        return n;
    }

    /// Every layer's four convolution states to (`out`) or from `buf`, on the engine's stream.
    fn convCopy(e: *Engine, st: *const state.State, buf: u64, out: bool) !void {
        const c = e.ctx();
        var at: u64 = buf;
        for (st.layers) |l| {
            const kvc = l.kv.heads * weights.head_dim * 2;
            for ([4]u64{ l.conv_k, l.conv_v, l.conv_attn, l.conv_mlp }, [4]usize{ kvc, kvc, H * 2, H * 2 }) |p, w| {
                const n = conv_rows * w;
                if (out) try c.copy(at, p, n) else try c.copy(p, at, n);
                at += n;
            }
        }
        try e.stream.synchronize();
    }

    /// Where slot `slot`'s sequence stands (positions held).
    pub fn pos(e: *Engine, slot: u32) usize {
        if (slot >= e.slots.items.len) return 0;
        return if (e.slots.items[slot]) |s| s.pos else 0;
    }

    fn ctx(e: *Engine) ops.Ctx {
        return .{ .d = e.d, .stream = e.stream, .k = &e.k, .gemms = &e.gemms, .scratch = &e.scratch };
    }

    /// The op context on the engine's stream (op checks between steps).
    pub fn opCtx(e: *Engine) ops.Ctx {
        return e.ctx();
    }

    /// R rows of slot `slot` at positions pos0 .. pos0 + R - 1 through every layer, on every rank (each rank calls it
    /// with the same slot, pos0 and R; rank 0 with the token ids). pos0 = 0 starts the slot's sequence over.
    pub fn forward(e: *Engine, slot: u32, pos0: usize, R: usize, ids: ?[]const u32) !void {
        const st = try e.ready(slot, pos0, R);
        e.scratch.kind = scratch.kindOf(&.{.{ .pos0 = pos0, .rows = R }});
        var f: fwd.Forward = .{ .c = e.ctx(), .w = &e.w, .st = st, .a = &e.act, .x = if (e.parts != null) &e.x else null, .spec = e.drafter != null and R <= state.spec_rows, .prof = if (e.prof_on) e.prof else null };
        f.capture = @import("cuda_snap.zig").forwardHook(e, slot, pos0, R); // the prompt cache's marks inside a chunk
        if (pos0 == 0) try st.restart(&f.c);
        try e.tokens(ids, R);
        try e.body(&f, R, pos0);
        if (f.prof) |p| try p.end();
        const one = [1]fwd.Seg{.{ .st = st, .pos0 = pos0, .rows = R }};
        const sg = [1]Segment{.{ .slot = slot, .pos0 = pos0, .rows = R }};
        try e.drafterRows(&f, R, &one, &sg);
        e.noteOneRow(&one);
        e.counts.eager += 1;
        e.forwards += 1;
        st.last_fwd = e.forwards;
        st.pos = pos0 + R;
        st.last_pos0 = pos0;
        st.last_rows = R;
        st.last_row0 = 0;
        st.kept_inputs = f.spec;
        e.rows = R;
        e.head_n = 0;
    }

    /// What `forwardMany` checks before its first collective (as `ready`, for every segment): the segments' forward
    /// (each its stream's state, its first row and whether it keeps its convolution inputs) into `out`; the rows.
    pub fn readyMany(e: *Engine, segs: []const Segment, out: *[fwd.max_segs]fwd.Seg) !usize {
        if (segs.len == 0 or segs.len > fwd.max_segs) return error.BadSegments;
        var R: usize = 0;
        for (segs, 0..) |g, i| {
            for (segs[0..i]) |o| if (o.slot == g.slot) return error.SlotTwiceInAForward;
            const st = try e.ready(g.slot, g.pos0, g.rows);
            out[i] = .{ .st = st, .pos0 = g.pos0, .row0 = R, .rows = g.rows, .spec = e.drafter != null and g.rows <= state.spec_rows };
            R += g.rows;
        }
        if (R > prompt_rows) return error.BadRows;
        return R;
    }

    /// Several streams' rows in one forward, on every rank (batched decode): segment i is segs[i].rows rows of slot
    /// segs[i].slot from position segs[i].pos0, after the rows of the segments before it (rank 0's `ids` in that
    /// order). Every op that does not touch a stream's state runs once over all the rows; the 13 that do run once a
    /// segment on its rows (cuda_forward.zig Seg), so each stream's rows get the bits of a forward of their own.
    /// One segment is `forward`.
    pub fn forwardMany(e: *Engine, segs: []const Segment, ids: ?[]const u32) !void {
        if (segs.len == 1) return e.forward(segs[0].slot, segs[0].pos0, segs[0].rows, ids);
        var fs: [fwd.max_segs]fwd.Seg = undefined;
        const R = try e.readyMany(segs, &fs);
        const gs = fs[0..segs.len];
        e.scratch.kind = .batched;
        var f: fwd.Forward = .{ .c = e.ctx(), .w = &e.w, .st = gs[0].st, .a = &e.act, .x = if (e.parts != null) &e.x else null, .segs = gs, .prof = if (e.prof_on) e.prof else null };
        for (gs) |g| if (g.pos0 == 0) try g.st.restart(&f.c);
        try e.tokens(ids, R);
        try e.body(&f, R, segs[0].pos0);
        if (f.prof) |p| try p.end();
        try e.drafterRows(&f, R, gs, segs);
        e.noteOneRow(gs);
        e.counts.eager += 1;
        e.forwards += 1;
        for (gs) |g| {
            g.st.last_fwd = e.forwards;
            g.st.pos = g.pos0 + g.rows;
            g.st.last_pos0 = g.pos0;
            g.st.last_rows = g.rows;
            g.st.last_row0 = g.row0;
            g.st.kept_inputs = g.spec;
        }
        e.rows = R;
        e.head_n = 0;
    }

    /// With the drafter: tap 7 (the final normed rows) over the forward's R rows, then each segment's draft context
    /// at its positions (rows a verify rejects are rewritten by the next one before a draft can read them). A
    /// captured forward's segments carry `dev`: the context lands at the positions its header holds at replay.
    fn drafterRows(e: *Engine, f: *const fwd.Forward, R: usize, gs: []const fwd.Seg, segs: []const Segment) !void {
        const dr = e.drafter orelse return;
        const c = f.c;
        const a = &e.act;
        e.scratch.reset();
        const x = a.x[0].slice(0, R);
        try res.mix(&c, a.blockView(R), a.h[0].slice(0, R), e.w.output_query, 1, &.{x});
        try head.finalNorm(&c, &e.w, x, Mat.of(dr.tap(dspark.ntaps - 1), R, H, .bf16));
        for (gs, segs) |g, sg| try dr.absorb(&c, sg.slot, g.pos0, g.rows, g.row0, if (g.dev) |d| d.posp else 0);
    }

    /// Rank 0: the forward's token ids to the device, queued on the stream after its earlier work (a pinned copy
    /// the stream uploads in order: no wait for the device here; TF_INK2_SYNC_UPLOADS=1 waits and copies as before).
    /// Ranks above 0 with the shared embedding: the header's ids when it carries them (`ids_in_header`; the drafted
    /// rows from the drafter's walk, as rank 0 does), else rank 0's ids come by a broadcast in `body`.
    fn tokens(e: *Engine, ids: ?[]const u32, R: usize) !void {
        if (e.o.rank != 0) {
            if (e.shared != null and e.ids_in_header) {
                if (R > graph_rows) return error.BadRows;
                const c = e.ctx();
                try c.copy(e.act.tokens.ptr, e.hdr.ptr + Header.tokens_at, R * 4);
                try e.walkIds(e.act.tokens.ptr, R);
            } else e.walk_n = 0;
            return;
        }
        const t = ids orelse return error.NoIds;
        if (t.len != R) return error.BadRows;
        var host: [prompt_rows]i32 = undefined;
        for (host[0..R], t) |*h, v| h.* = @intCast(v);
        if (!syncUploads()) try cuda.copy_source.enqueue(e.d, e.act.tokens.ptr, std.mem.sliceAsBytes(host[0..R]), e.stream) else {
            try e.stream.synchronize();
            const view: cuda.DeviceBuffer = .{ .d = e.d, .ptr = e.act.tokens.ptr, .len = R * 4, .borrowed = true };
            try view.upload(0, std.mem.sliceAsBytes(host[0..R]));
        }
        try e.walkIds(e.act.tokens.ptr, R);
    }

    /// Rank 0: the rows `walk_ids` names, from the drafter's walk into the forward's ids at `ids` (R of them), after
    /// the host's ids (queued on the stream); then none are pending.
    fn walkIds(e: *Engine, ids: u64, R: usize) !void {
        if (e.walk_n == 0) return;
        defer e.walk_n = 0;
        const dr = e.drafter orelse return error.NoDrafter;
        const c = e.ctx();
        for (e.walk_ids[0..e.walk_n]) |w| {
            if (w.row + 1 + w.n > R or w.n > dspark.drafts) return error.BadRows;
            try c.copy(ids + (w.row + 1) * 4, dr.walk + (w.y * dspark.block + 1) * 4, w.n * 4);
        }
    }

    /// Rank 0: the next forward's rows that `walkIds` fills from the last `draftMany` (Leader.forwardDrafted); every
    /// rank with the shared embedding (`shared`: a captured forward's ids are each rank's header copy, cuda_follow.zig
    /// followDrafts; the drafter's walk is on every rank).
    pub fn setWalkIds(e: *Engine, ws: []const WalkIds) !void {
        if (e.o.rank != 0 and e.shared == null) return;
        if (ws.len > max_segs) return error.BadSegments;
        @memcpy(e.walk_ids[0..ws.len], ws);
        e.walk_n = ws.len;
    }

    /// Rank 0: the last `draftMany`'s drafts of its streams (synchronizes the stream).
    pub fn proposals(e: *Engine, out: []const *[dspark.drafts]u32) !void {
        const dr = e.drafter orelse return error.NoDrafter;
        const c = e.ctx();
        try dr.proposalsMany(&c, out);
    }

    /// The last draft pass's pick probabilities of each of its streams, in its order (rank 0; the lone cut).
    pub fn chances(e: *Engine, out: []const *[dspark.drafts]f32) !void {
        const dr = e.drafter orelse return error.NoDrafter;
        const c = e.ctx();
        try dr.chancesMany(&c, out);
    }

    /// A forward's launches, after its inputs are on the device: the embedding (W > 1: an eager forward's ids from the
    /// header or broadcast from rank 0, then every rank's rows gathered; W = 1: rank 0's own), every layer. Nothing
    /// here waits on the device, so a decode step's body can be captured (`step`).
    fn body(e: *Engine, f: *fwd.Forward, R: usize, pos0: usize) !void {
        const a = &e.act;
        defer e.ids_in_header = false;
        if (e.shared) |*sh| {
            if (f.ids == 0 and !e.ids_in_header) {
                if (e.o.group) |g| try g.lib.check(g.lib.api.ncclBroadcast(a.tokens.ptr, a.tokens.ptr, R * 4, .u8, 0, g.comm, e.stream.handle), "ncclBroadcast");
            }
            try f.embedShared(sh, R, pos0);
        } else if (e.o.rank == 0) try f.embed(R, pos0);
        a.nblocks = 0;
        var li: u32 = 0;
        while (li < e.layers) : (li += 1) {
            f.layer(li, R, pos0) catch |err| {
                if (f.stopped) |s| std.log.err("inkling2: layer {f} op {s}: {t}", .{ s.scope, s.op, s.err });
                return err;
            };
            if (e.drafter) |dr| if (dspark.tapOf(li)) |t| try f.c.copy(dr.tap(t), a.h[0].ptr, R * H * 2);
        }
    }

    /// What `graphStep` checks before any rank moves (rank 0 runs it before the header): `readyMany`, and every
    /// segment past its prompt (pos0 > 0) with at most `graph_seg_rows` rows, at most `graph_rows` in all.
    pub fn graphReady(e: *Engine, segs: []const Segment, out: *[fwd.max_segs]fwd.Seg) !usize {
        const R = try e.readyMany(segs, out);
        if (R > graph_rows) return error.BadRows;
        for (segs) |g| if (g.pos0 == 0 or g.pos0 > std.math.maxInt(i32) or g.rows > graph_seg_rows) return error.OutOfStep;
        return R;
    }

    /// Rank 0, the second-sighting rule for a forward of several segments: true when its key's graph is held or the
    /// key ran eagerly once before (it is then captured); else the key is remembered and the forward runs eagerly. A
    /// key seen once (a stream set that never repeats) costs no capture.
    pub fn graphWanted(e: *Engine, key: Key) bool {
        if (key.units() > e.graph_max) return false; // larger than the whole cache: eager
        for (e.graphs.items) |g| if (g.key.eql(key)) return true;
        for (e.seen[0..@min(e.seen_n, seen_max)]) |k| if (k.eql(key)) return true;
        e.seen[e.seen_n % seen_max] = key;
        e.seen_n += 1;
        return false;
    }

    /// The graph key of a forward of `segs`: their slots and rows, and which one-row segments are on their run
    /// (State.ringRun: such a segment's global layers take the ring kernels on the run, so it is a different graph).
    pub fn keyOf(e: *const Engine, segs: []const Segment) !Key {
        var k = try Key.of(segs);
        for (segs, 0..) |g, i| {
            if (g.rows != 1 or g.slot >= e.slots.items.len) continue;
            const st = e.slots.items[g.slot] orelse continue;
            if (st.ringRun() != null) k.ring |= @as(u32, 1) << @intCast(i);
        }
        return k;
    }

    /// Counts a forward's one-row segments on pages by the kernels their global layers take (Counts.one_ring /
    /// one_paged) and each stream's change of mode (switches).
    fn noteOneRow(e: *Engine, gs: []const fwd.Seg) void {
        for (gs) |g| {
            if (g.rows != 1 or !g.st.paged()) continue;
            const ring = g.st.ringRun() != null;
            if (ring) e.counts.one_ring += 1 else e.counts.one_paged += 1;
            const now: @TypeOf(g.st.last_one) = if (ring) .ring else .paged;
            if (g.st.last_one != .none and g.st.last_one != now) e.counts.switches += 1;
            g.st.last_one = now;
            g.st.ones += 1;
        }
    }

    /// A captured forward's nodes that `log` recorded (launches on a segment's run), found among the graph's kernel
    /// nodes by kernel and base argument; error.RunNodes unless every recorded launch is found once.
    fn runNodes(e: *Engine, graph: cuda.graph.Graph, log: *const ops.RunLog, gs: []const fwd.Seg, count: usize) ![]Graph.RunNode {
        if (log.items.items.len == 0) return &.{};
        const out = try e.gpa.alloc(Graph.RunNode, log.items.items.len);
        errdefer e.gpa.free(out);
        for (out, log.items.items) |*o, x| {
            o.* = .{ .node = null, .seg = 0, .at = x };
            var seg: ?usize = null;
            for (gs, 0..) |g, i| if (g.rows == 1 and g.st.ringRun() == x.run) {
                seg = i;
            };
            o.seg = @intCast(seg orelse return error.RunNodes);
        }
        const all = try e.gpa.alloc(cuda.graph.Node, count);
        defer e.gpa.free(all);
        const nodes = try graph.nodes(all);
        var found: usize = 0;
        for (nodes) |node| {
            var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
            if (e.d.api.cuGraphKernelNodeGetParams_v2(node, &p) != cuda.abi.success) continue; // not a kernel node
            const params = p.params orelse continue;
            for (out) |*o| {
                if (o.node != null or o.at.f.handle != p.func) continue;
                const i = o.at.at[0];
                const got = @as(*align(1) const u64, @ptrCast(params[i] orelse continue)).*;
                if (got != o.at.args.integer(i).?) continue;
                o.node = node;
                found += 1;
                break;
            }
        }
        if (found != out.len) {
            std.log.err("inkling2 rank {d}: {d} of {d} launches on a stream's run found in the captured graph", .{ e.o.rank, found, out.len });
            return error.RunNodes;
        }
        return out;
    }

    /// A forward of segments `segs` (each past its prompt, a decode row or a verify window) and then the head over
    /// all its rows, on every rank, as one captured CUDA graph: captured the first time its key (`Key`: slots and
    /// rows) is seen and replayed after, each segment's first position and the token ids read on the device from the
    /// step header's copy (`post` and `receive` leave it there; at W = 1 there is no broadcast and this writes it),
    /// so the replay moves nothing to the device besides the header and holds one launch instead of thousands. The
    /// same kernels in the same order as `forwardMany` (or `forward`) then `headRows(0, R)`, the drafter's taps and
    /// context writes included, so the same bits. Every rank runs it for the same forwards (the header's `graph`),
    /// so every rank's cache holds the same graphs with the same collectives. Rank 0 gets the R rows' logits.
    pub fn graphStep(e: *Engine, segs: []const Segment, ids: ?[]const u32) !void {
        var fs: [fwd.max_segs]fwd.Seg = undefined;
        const R = try e.graphReady(segs, &fs);
        const gs = fs[0..segs.len];
        e.scratch.kind = if (segs.len > 1) .batched else scratch.kindOf(&.{.{ .pos0 = segs[0].pos0, .rows = segs[0].rows }});
        if (e.o.group == null) {
            var h: Header = .{ .kind = 0 };
            try h.setSegs(segs);
            try h.setTokens(ids orelse return error.NoIds);
            try e.stream.synchronize();
            try e.hdr.upload(0, std.mem.asBytes(&h));
        }
        if (e.o.rank == 0 or e.shared != null) try e.walkIds(e.hdr.ptr + Header.tokens_at, R); // after the header's upload (post, receive, or above)
        const g = try e.graphOf(try e.keyOf(segs), gs, segs, R);
        const moved = try g.follow(gs);
        if (moved > 0) {
            e.counts.repointed += moved;
            e.counts.repoint_replays += 1;
        }
        e.noteOneRow(gs);
        const a = &e.act;
        a.h = g.h_start;
        try g.exec.launchOn(e.stream);
        a.h = g.h_end;
        a.nblocks = g.nblocks;
        e.counts.graph += 1;
        e.forwards += 1;
        for (gs) |x| {
            x.st.last_fwd = e.forwards;
            x.st.pos = x.pos0 + x.rows;
            x.st.last_pos0 = x.pos0;
            x.st.last_rows = x.rows;
            x.st.last_row0 = x.row0;
            x.st.kept_inputs = x.spec;
        }
        e.rows = R;
        e.head_n = 0;
        try e.headDone(R);
    }

    /// One decode step of slot `slot` (one row at pos0 > 0, then its head) as a captured forward (`graphStep`).
    pub fn step(e: *Engine, slot: u32, pos0: usize, token: ?u32) !void {
        const t = [1]u32{token orelse 0};
        return e.graphStep(&.{.{ .slot = slot, .pos0 = pos0, .rows = 1 }}, if (token != null) &t else null);
    }

    /// The graph for `key`, captured now if the cache has none (the least recently used one replaced when full).
    /// Every rank calls it for the same keys in the same order, so every rank's cache stays the same.
    fn graphOf(e: *Engine, key: Key, gs: []const fwd.Seg, segs: []const Segment, R: usize) !*Graph {
        e.graph_tick += 1;
        for (e.graphs.items) |*g| if (g.key.eql(key)) {
            g.used = e.graph_tick;
            return g;
        };
        if (key.units() > e.graph_max) return error.GraphOverCache; // Leader.graphOk sends no such key
        while (e.graphs.items.len > 0 and e.graph_units + key.units() > e.graph_max) {
            var lru: usize = 0;
            for (e.graphs.items, 0..) |g, i| if (g.used < e.graphs.items[lru].used) {
                lru = i;
            };
            var old = e.graphs.swapRemove(lru);
            e.graph_units -= old.key.units();
            old.deinit(e.gpa); // a replay in flight completes; the driver frees it after
            e.counts.evicted += 1;
        }
        const a = &e.act;
        var g: Graph = .{ .key = key, .exec = undefined, .h_start = a.h, .h_end = undefined, .nblocks = 0, .used = e.graph_tick };
        var dsegs: [fwd.max_segs]fwd.Seg = undefined;
        for (gs, dsegs[0..gs.len], 0..) |x, *y, i| {
            y.* = x;
            y.dev = .{ .posp = e.hdr.ptr + Header.segPosAt(i), .positions = x.st.context };
        }
        var f: fwd.Forward = .{ .c = e.ctx(), .w = &e.w, .st = gs[0].st, .a = a, .x = if (e.parts != null) &e.x else null, .segs = dsegs[0..gs.len], .ids = e.hdr.ptr + Header.tokens_at };
        // the launches on a segment's run (key.ring), so a replay can follow the stream's run when it moves
        var log: ops.RunLog = .{ .gpa = e.gpa };
        defer log.deinit();
        if (key.ring != 0) f.c.runs = &log;
        const free0 = e.memFree();
        const t0 = Timing.now();
        try cuda.graph.beginCapture(e.stream, .relaxed);
        const captured: anyerror!void = run: {
            e.body(&f, R, gs[0].pos0) catch |err| break :run err;
            e.drafterRows(&f, R, dsegs[0..gs.len], segs) catch |err| break :run err;
            e.headBody(0, R) catch |err| break :run err;
        };
        var graph = cuda.graph.endCapture(e.stream) catch |err| {
            a.h = g.h_start;
            captured catch |first| return first;
            return err;
        };
        var kept = false;
        defer if (!kept) graph.deinit();
        captured catch |err| {
            a.h = g.h_start;
            return err;
        };
        const nodes = try graph.nodeCount();
        g.runs = try e.runNodes(graph, &log, gs, nodes);
        errdefer e.gpa.free(g.runs);
        g.exec = try graph.instantiate();
        errdefer g.exec.deinit();
        if (g.runs.len > 0) {
            g.tmpl = graph;
            kept = true;
        }
        errdefer if (kept) {
            kept = false;
            g.tmpl = null;
        };
        try g.exec.upload(e.stream);
        g.h_end = a.h;
        g.nblocks = a.nblocks;
        a.h = g.h_start;
        try e.graphs.append(e.gpa, g);
        e.graph_units += key.units();
        const ms = @as(f64, @floatFromInt(Timing.now() - t0)) * 1e-6;
        const bytes: i64 = @as(i64, @intCast(free0)) - @as(i64, @intCast(e.memFree()));
        e.counts.captured += 1;
        e.counts.graph_bytes += bytes;
        e.counts.capture_ms += ms;
        std.log.info("inkling2 rank {d}: forward ({f}) captured as one CUDA graph ({d} nodes, {d} on a stream's run) in {d:.1} ms, {d:.1} MiB; {d} held ({d} of {d} units), {d:.1} MiB in all", .{ e.o.rank, key, nodes, g.runs.len, ms, @as(f64, @floatFromInt(bytes)) / (1 << 20), e.graphs.items.len, e.graph_units, e.graph_max, @as(f64, @floatFromInt(e.counts.graph_bytes)) / (1 << 20) });
        return &e.graphs.items[e.graphs.items.len - 1];
    }

    /// Free device memory (cuMemGetInfo; on GB10 the system's), to measure a capture's cost.
    fn memFree(e: *const Engine) usize {
        var free: usize = 0;
        var total: usize = 0;
        e.d.check(e.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo") catch return 0;
        return free;
    }

    /// What `keep` checks before any rank moves: the slot's last forward was a verify window holding `n` rows.
    pub fn keepReady(e: *Engine, slot: u32, n: usize) !*state.State {
        if (slot >= e.slots.items.len) return error.UnknownSlot;
        const st = e.slots.items[slot] orelse return error.UnknownSlot;
        if (n == 0 or n > st.last_rows or st.pos != st.last_pos0 + st.last_rows and st.pos != st.last_pos0 + n) return error.BadKeep;
        if (n < st.last_rows and !st.kept_inputs) return error.RollbackNotKept;
        return st;
    }

    /// The slot keeps the first `n` rows of its last forward (a verify window): every convolution's state goes back
    /// to after row n - 1 and the next forward starts at its position + 1. The keys and values of the rows past it
    /// need no undoing: the next forward rewrites those positions before anything reads them.
    pub fn keep(e: *Engine, slot: u32, n: usize) !void {
        const st = try e.keepReady(slot, n);
        const want = st.last_pos0 + n;
        if (st.pos == want) return;
        const c = e.ctx();
        try state.keepAll(&c, st, n); // one launch for every convolution of every layer
        st.pos = want;
    }

    /// The drafter's block at the slot's position (after `keep`): `anchor` is the token there (the target's last
    /// draw); rank 0 gets the 7 drafts in `out`. Every rank runs it (and embeds the anchor: `draftMany`).
    pub fn draft(e: *Engine, slot: u32, anchor: u32, out: ?*[dspark.drafts]u32) !void {
        const one = [1]DraftItem{.{ .slot = slot, .anchor = anchor }};
        if (out) |o| return e.draftMany(&one, &.{o});
        return e.draftMany(&one, null);
    }

    /// What `draftMany` checks before any rank moves (after each slot's keep, or of slots that kept already): every
    /// slot's position is just past a row of its last forward (whose taps the block reads), no slot twice, at most
    /// `dspark.most_streams` of them; their blocks into `out`.
    pub fn draftReady(e: *Engine, items: []const DraftItem, out: *[dspark.most_streams]dspark.Item) ![]const dspark.Item {
        if (e.drafter == null) return error.NoDrafter;
        if (items.len == 0 or items.len > dspark.most_streams) return error.BadSegments;
        for (items, out[0..items.len], 0..) |it, *o, i| {
            for (items[0..i]) |b| if (b.slot == it.slot) return error.SlotTwiceInADraft;
            if (it.slot >= e.slots.items.len) return error.UnknownSlot;
            const st = e.slots.items[it.slot] orelse return error.UnknownSlot;
            const p = st.pos;
            if (p == 0 or p - 1 < st.last_pos0 or p - 1 >= st.last_pos0 + st.last_rows) return error.DraftWithoutTaps;
            o.* = .{ .slot = it.slot, .p = p, .anchor = it.anchor, .feat_row = st.last_row0 + p - 1 - st.last_pos0 }; // the taps' row of position p - 1
        }
        return out[0..items.len];
    }

    /// Several slots' drafter blocks in one pass (cuda_dspark.zig draftMany), each at its slot's position (after its
    /// `keep`) from its `anchor`: each slot's drafts are the bits of its own `draft`. Every rank runs it (W > 1: the
    /// anchors' embedding rows gathered; W = 1: rank 0 embeds them); rank 0 gets slot i's 7 drafts in out[i].
    pub fn draftMany(e: *Engine, items: []const DraftItem, out: ?[]const *[dspark.drafts]u32) !void {
        const dr = e.drafter orelse return error.NoDrafter;
        var buf: [dspark.most_streams]dspark.Item = undefined;
        const its = try e.draftReady(items, &buf);
        const S = its.len;
        const c = e.ctx();
        e.scratch.reset();
        if (e.o.rank == 0 or e.shared != null) {
            var ids: [dspark.most_streams]i32 = undefined;
            for (items, ids[0..S]) |it, *v| v.* = @intCast(it.anchor);
            try cuda.copy_source.enqueue(e.d, e.act.tokens.ptr, std.mem.sliceAsBytes(ids[0..S]), e.stream);
        }
        if (e.shared) |*sh| {
            try sh.rows(&c, &e.w, e.act.tokens.ptr, e.act.y.slice(0, S), Mat.of(dr.anchors, S, H, .bf16)); // every rank has the anchors
        } else if (e.o.rank == 0) try head.embed(&c, &e.w, e.act.tokens.ptr, e.act.y.slice(0, S), Mat.of(dr.anchors, S, H, .bf16));
        try dr.draftMany(&c, its);
        if (out) |o| try dr.proposalsMany(&c, o);
    }

    /// The head over rows lo .. lo + n of the last forward, on every rank (each calls it with the same rows): the
    /// final mix over the blocks completed so far, the final norm, the unembed of the rank's share of the
    /// vocabulary, the shares all-gathered in rank order; rank 0 then softcaps them and copies them to the host
    /// (`logits`).
    pub fn headRows(e: *Engine, lo: usize, n: usize) !void {
        if (n == 0 or n > head_rows or lo + n > e.rows) return error.BadRows;
        try e.headBody(lo, n);
        try e.headDone(n);
    }

    /// The head's launches (see headRows); nothing here waits on the device.
    fn headBody(e: *Engine, lo: usize, n: usize) !void {
        const c = e.ctx();
        const a = &e.act;
        e.scratch.reset();
        const blocks: ops.Blocks = .{ .ptr = a.blocks + lo * H * 2, .stride = a.blockStride(), .count = a.nblocks, .rows = n, .hidden = H };
        const x = a.x[0].slice(0, n);
        try res.mix(&c, blocks, a.h[0].slice(lo, n), e.w.output_query, 1, &.{x});
        const y = e.hrows.slice(0, n);
        try head.finalNorm(&c, &e.w, x, y);
        const share: usize = e.w.vocab_rows.len();
        const mine = e.hparts + @as(u64, e.o.rank) * n * share * 2;
        try head.unembed(&c, &e.w, y, Mat.of(mine, n, share, .bf16));
        if (e.o.group) |g| try g.allGather(e.stream, mine, e.hparts, n * share * 2);
        e.head_n = 0;
        if (e.o.rank != 0) return;
        try head.softcapParts(&c, e.hparts, e.o.world, share, e.hout.slice(0, n));
        if (e.argmax_on) try head.argmax(&c, e.hout.slice(0, n), e.hbufs[3].?.ptr);
    }

    /// Rank 0: waits for the head and copies its n rows of logits to the host (`logits`), or, when the lanes asked
    /// for none (`want_logits` false), only the rows' argmax (`argmaxRows`).
    fn headDone(e: *Engine, n: usize) !void {
        if (e.o.rank != 0) return;
        try e.stream.synchronize();
        if (e.o.group) |g| try g.check();
        const t0 = Timing.now();
        const full = e.want_logits or !e.argmax_on;
        const every = rowsBelow(n);
        // without the device's argmax every row's logits come down; with it, the rows the lanes asked for
        const have: u32 = if (!full) 0 else if (!e.argmax_on) every else e.want_rows & every;
        const U = e.w.unpadded_vocab;
        if (e.argmax_on) try cuda.memory.download(e.d, e.hbufs[3].?.ptr, std.mem.sliceAsBytes(e.host_max[0..n]));
        if (have == every) {
            try cuda.memory.download(e.d, e.hout.ptr, std.mem.sliceAsBytes(e.host[0 .. n * U]));
        } else {
            // each run of wanted rows in one copy
            var r: usize = 0;
            while (r < n) {
                if (have >> @intCast(r) & 1 == 0) {
                    r += 1;
                    continue;
                }
                var end = r + 1;
                while (end < n and have >> @intCast(end) & 1 == 1) end += 1;
                try cuda.memory.download(e.d, e.hout.ptr + r * U * 4, std.mem.sliceAsBytes(e.host[r * U .. end * U]));
                r = end;
            }
        }
        const k = @intFromBool(full);
        e.counts.heads[k] += 1;
        e.counts.head_rows[k] += if (full) @popCount(have) else n;
        e.counts.head_ns[k] += Timing.now() -| t0;
        e.head_n = n;
        e.head_logits = full;
        e.head_have = have;
    }

    /// Bits of head rows 0 .. n - 1.
    fn rowsBelow(n: usize) u32 {
        return if (n >= 32) all_rows else (@as(u32, 1) << @intCast(n)) - 1;
    }

    /// Rank 0: the softcapped fp32 logits [n, unpadded vocab] of the last head's rows (host memory, valid until the
    /// next forward or head). error.NoLogits when the head brought down only its argmax rows.
    pub fn logits(e: *const Engine) ![]const f32 {
        if (e.o.rank != 0) return error.NotRankZero;
        if (e.head_n == 0 or !e.head_logits or e.head_have != rowsBelow(e.head_n)) return error.NoLogits;
        return e.host[0 .. e.head_n * e.w.unpadded_vocab];
    }

    /// Rank 0: row `r` of the last head's logits, brought down now if the head left it on the device (`want_rows`;
    /// the head's buffer holds it until the next forward).
    pub fn fetchRow(e: *Engine, r: usize) ![]const f32 {
        if (e.o.rank != 0) return error.NotRankZero;
        if (r >= e.head_n) return error.NoLogits;
        const U = e.w.unpadded_vocab;
        if (e.head_have >> @intCast(r) & 1 == 0) {
            try cuda.memory.download(e.d, e.hout.ptr + r * U * 4, std.mem.sliceAsBytes(e.host[r * U ..][0..U]));
            e.head_have |= @as(u32, 1) << @intCast(r);
            e.head_logits = true;
        }
        return e.host[r * U ..][0..U];
    }

    /// Rank 0: row `r` of the last head's logits, when it came down (`want_rows`).
    pub fn logitsRow(e: *const Engine, r: usize) ![]const f32 {
        if (e.o.rank != 0) return error.NotRankZero;
        if (r >= e.head_n or !e.head_logits or e.head_have >> @intCast(r) & 1 == 0) return error.NoLogits;
        const U = e.w.unpadded_vocab;
        return e.host[r * U ..][0..U];
    }

    /// Rank 0: the last head's rows' argmax from the device (each the host argmax of its row's logits), when
    /// TF_INK2_GPU_ARGMAX is on.
    pub fn argmaxRows(e: *const Engine) ![]const u32 {
        if (e.o.rank != 0) return error.NotRankZero;
        if (e.head_n == 0 or !e.argmax_on) return error.NoArgmax;
        return e.host_max[0..e.head_n];
    }

    /// Split experts run in checkpoint mode only, and every rank must hold the experts the same way and run the same
    /// mode (a gather of both, its own collective, so a mismatch stops every rank before loading).
    fn checkExperts(e: *Engine) !void {
        const layout = e.o.experts.?;
        if (layout == .split and e.k.moe.mode != .checkpoint) {
            std.log.err("inkling2: split experts (TF_INK2_EXPERTS=split) run in checkpoint mode only: set TF_INK2_MODE=checkpoint", .{});
            return error.SplitExpertsNeedCheckpointMode;
        }
        const g = e.o.group orelse return;
        const mine: [2]u32 = .{ @intFromEnum(layout), @intFromEnum(e.k.moe.mode) };
        var all: [ink.plan.max_world][2]u32 = undefined;
        try g.gather([2]u32, e.d, e.stream, mine, all[0..e.o.world]);
        for (all[0..e.o.world], 0..) |x, r| if (!std.mem.eql(u32, &x, &mine)) {
            std.log.err("inkling2 rank {d}: rank {d} runs experts {t} in {t} mode, this rank {t} in {t} mode", .{ e.o.rank, r, @as(ink.plan.Layout, @enumFromInt(x[0])), @as(@TypeOf(e.k.moe.mode), @enumFromInt(x[1])), layout, e.k.moe.mode });
            return error.RankMismatch;
        };
    }

    /// Rank 0: a step's header to every other rank (W > 1): its upload to the device copy (a pinned copy queued on
    /// the stream, so it lands after every kernel already queued, a replay's reads of the last header included) and
    /// the broadcast, without waiting for the device (TF_INK2_SYNC_UPLOADS=1: wait, then a blocking copy, as before).
    pub fn post(e: *Engine, h: Header) !void {
        const g = e.o.group orelse return;
        var x = h;
        x.seq = e.seq;
        e.seq +%= 1;
        if (syncUploads()) {
            try e.stream.synchronize();
            try e.hdr.upload(0, std.mem.asBytes(&x));
        } else try cuda.copy_source.enqueue(e.d, e.hdr.ptr, std.mem.asBytes(&x), e.stream);
        try g.lib.check(g.lib.api.ncclBroadcast(e.hdr.ptr, e.hdr.ptr, @sizeOf(Header), .u8, 0, g.comm, e.stream.handle), "ncclBroadcast");
    }

    /// Ranks above 0: rank 0's next step.
    pub fn receive(e: *Engine) !Header {
        const g = e.o.group orelse return error.NoGroup;
        try g.lib.check(g.lib.api.ncclBroadcast(e.hdr.ptr, e.hdr.ptr, @sizeOf(Header), .u8, 0, g.comm, e.stream.handle), "ncclBroadcast");
        try e.stream.synchronize();
        var h: Header = undefined;
        try e.hdr.download(0, std.mem.asBytes(&h));
        if (h.magic != Header.magic_word) return error.BadStep;
        if (h.seq != e.seq) {
            std.log.err("inkling2 rank {d}: header {d} from rank 0, {d} expected", .{ e.o.rank, h.seq, e.seq });
            return error.BadStep;
        }
        e.seq +%= 1;
        return h;
    }

    /// Every rank must have the same executable, checkpoint (with its digest manifest's text, or none) and settings
    /// (with the experts' mode) before loading.
    fn agree(e: *Engine, g: *const cuda.tp_group.Group, dir: []const u8, manifest: ?[]const u8, mode: moe.Mode) !void {
        const Sha = std.crypto.hash.sha2.Sha256;
        const o = e.o;
        var fp: Fingerprint = .{ .program = undefined, .checkpoint = undefined, .settings = undefined, .rank = o.rank, .world = o.world };
        {
            var h = Sha.init(.{});
            try hashFile(e.gpa, e.io, &h, "/proc/self/exe");
            h.final(&fp.program);
        }
        {
            var h = Sha.init(.{});
            for ([_][]const u8{ "config.json", "hf_quant_config.json", "model.safetensors.index.json" }) |name| {
                const path = try std.fs.path.join(e.gpa, &.{ dir, name });
                defer e.gpa.free(path);
                try hashFile(e.gpa, e.io, &h, path);
            }
            const m = &e.model;
            for (m.inv.tensors) |t| {
                h.update(t.name);
                h.update(std.mem.asBytes(&[3]u64{ t.file, m.inv.offset(t), t.bytes() }));
            }
            // the expected contents: every rank's piece digests (each rank checks its own after loading)
            h.update(manifest orelse "(no digest manifest)");
            h.final(&fp.checkpoint);
        }
        {
            var h = Sha.init(.{});
            var buf: [768]u8 = undefined;
            var ab: [256]u8 = undefined;
            // the switches hashed on their own (`settingsHash`): TF_INK2_DRAFT_CUT=0 and TF_INK2_UNEMBED_WIDE=0
            const draft_cut_off = o.drafter and dspark.draftCut() == .off;
            // every switch that changes a rank's arithmetic, its collectives' shapes or whether decode steps run as
            // graphs: a rank set differently would hang a collective or mix arithmetic, so the ranks must agree
            const base = try std.fmt.bufPrint(&buf, "world {d} layers {d} context {d} scratch {d} rows {d} mode {t} {s} TF_INK2_XROWS={s} TF_INK2_TIME_EXCHANGE={s} TF_INK2_CK_PROFILE={s} TF_INK2_CK_SHAPE1={s} TF_INK2_GRAPHS_MAX={d} dspark {} TF_INK2_DSPARK_FP8={t} TF_INK2_ROCE_MULTI={s} TF_ROCE={s} TF_INK2_PAGES={} TF_KV_POISON={} TF_INK2_PAGE_ORDER={s} TF_INK2_RUN_RING={}", .{ o.world, o.layers, o.context, o.scratch_gib, prompt_rows, mode, ic_attn.fingerprintText(&ab), env("TF_INK2_XROWS"), env("TF_INK2_TIME_EXCHANGE"), env("TF_INK2_CK_PROFILE"), env("TF_INK2_CK_SHAPE1"), graphsMax(), o.drafter, dspark.fp8Mode(), env("TF_INK2_ROCE_MULTI"), env("TF_ROCE"), pages.enabled(), pages.poisoned(), env("TF_INK2_PAGE_ORDER"), pages.ringOnRun() });
            // TF_INK2_SHARED_CUT is not hashed: rank 0 alone picks the shared rounds' widths and sends the segments
            // (cuda_follow.zig), so ranks set differently still agree on every forward; the bytes stay v0.2's
            settingsHash(&h, base, draft_cut_off, @import("cuda_head.zig").wideOn());
            // the kernels' launch choices (fusion forms, the experts' shapes and grids): the same bits whatever they
            // are, but a rank set differently launches different kernels, so the ranks must agree too
            hashLaunchChoices(&h, .{ envOpt("TF_INK2_FUSE"), envOpt("TF_INK2_CK_SHAPEK1"), envOpt("TF_INK2_CK_BLOCKS"), envOpt("TF_INK2_CK_SHAPEW") }, moe.windowShapes(e.d) catch @splat(@splat(@splat(null))));
            h.final(&fp.settings);
        }
        var all: [ink.plan.max_world]Fingerprint = undefined;
        try g.gather(Fingerprint, e.d, e.stream, fp, all[0..o.world]);
        var ok = true;
        for (all[0..o.world], 0..) |y, r| {
            const why: ?[]const u8 = if (y.rank != r or y.world != o.world) "rank or world" else if (!std.mem.eql(u8, &y.program, &fp.program)) "the executable" else if (!std.mem.eql(u8, &y.checkpoint, &fp.checkpoint)) "the checkpoint or its digest manifest (TF_INK2_DIGESTS)" else if (!std.mem.eql(u8, &y.settings, &fp.settings)) "the settings (layers, --context, TF_INK2_SCRATCH_GIB, TF_INK2_MODE, TF_INK2_ATTN*, TF_INK2_XROWS, TF_INK2_TIME_EXCHANGE, TF_INK2_CK_PROFILE, TF_INK2_CK_SHAPE1, TF_INK2_CK_SHAPEK1, TF_INK2_CK_SHAPEW, TF_INK2_CK_BLOCKS, TF_INK2_FUSE, TF_INK2_GRAPHS_MAX, TF_INK2_DSPARK, TF_INK2_DSPARK_FP8, TF_INK2_UNEMBED_WIDE, TF_INK2_ROCE_MULTI, TF_ROCE, TF_INK2_PAGES, TF_KV_POISON, TF_INK2_PAGE_ORDER, TF_INK2_RUN_RING)" else null;
            if (why) |w| {
                std.log.err("inkling2 rank {d}: rank {d} differs from this rank in {s}", .{ o.rank, r, w });
                if (o.drafter and std.mem.startsWith(u8, w, "the settings")) std.log.err("inkling2 rank {d}: the settings also cover TF_INK2_DRAFT_CUT (0 on some ranks only: their drafter makes no chances)", .{o.rank});
                ok = false;
            }
        }
        if (!ok) return error.RankMismatch;
        std.log.info("inkling2 rank {d} of {d}: fingerprint agreed (executable {x}, checkpoint and digest manifest {x}, settings {x}; {t} mode, {s})", .{ o.rank, o.world, fp.program[0..6], fp.checkpoint[0..6], fp.settings[0..6], mode, if (manifest != null) "contents checked" else "contents NOT checked" });
    }
};

/// The settings part of the ranks' fingerprint (`Engine.agree`): `base`, the formatted settings (v0.2's text and 768-byte
/// buffer), with the switches hashed on their own, each only when it is off its default. So a rank at the defaults has
/// v0.2's digest, and a switch never decides whether `base` fits its buffer: ranks set differently all reach the gather
/// and refuse at the comparison (`error.RankMismatch`) instead of one rank failing before it while the others wait.
fn settingsHash(h: *std.crypto.hash.sha2.Sha256, base: []const u8, draft_cut_off: bool, wide: bool) void {
    // TF_INK2_DRAFT_CUT=0: the drafter's argmax-only picks (cuda_dspark.zig draftCut); nothing is added when the
    // drafter makes its chances, so that fingerprint is unchanged
    if (draft_cut_off) h.update("TF_INK2_DRAFT_CUT=0 ");
    h.update(base);
    // TF_INK2_UNEMBED_WIDE=0: the unembed's row tiles at every row count (cuda_head.wideOn; the same bits)
    if (!wide) h.update(" TF_INK2_UNEMBED_WIDE=0");
}

/// A captured forward (Engine.graphStep): the activations' ping-pong buffers as the capture found and left them,
/// the blocks it completed, and when it was last replayed (the cache's clock).
const Graph = struct {
    key: Key,
    exec: cuda.graph.Exec,
    h_start: [2]Mat,
    h_end: [2]Mat,
    nblocks: usize,
    used: u64,
    /// The nodes that read or write a ring view of a segment's run of pool pages (key.ring), each with the run it
    /// points at now; the captured graph is kept while they are (node handles are the template's).
    runs: []RunNode = &.{},
    tmpl: ?cuda.graph.Graph = null,

    const RunNode = struct { node: cuda.graph.Node, seg: u8, at: ops.RunLog.Launch };

    fn deinit(g: *Graph, gpa: std.mem.Allocator) void {
        g.exec.deinit();
        if (g.tmpl) |*t| t.deinit();
        gpa.free(g.runs);
    }

    /// Before a replay: every recorded node of a segment on its run pointed at the stream's run now (the bases move
    /// by whole pages: the same rows of a region, the run's first page on). Returns the nodes changed.
    fn follow(g: *Graph, gs: []const fwd.Seg) !usize {
        var n: usize = 0;
        for (g.runs) |*r| {
            const st = gs[r.seg].st;
            const now = st.ringRun() orelse return error.RunGone; // the key's ring bit says it is on a run
            if (now == r.at.run) continue;
            const step: i64 = @intCast(st.place.?.run);
            const delta = (@as(i64, now) - @as(i64, r.at.run)) * step;
            for (r.at.at[0..r.at.n]) |i| {
                const was = r.at.args.integer(i).?;
                const to: u64 = @intCast(@as(i64, @intCast(was)) + delta);
                @memcpy(r.at.args.storage[r.at.args.offsets[i]..][0..8], std.mem.asBytes(&to));
            }
            try g.exec.setKernel(r.node, r.at.f, r.at.cfg, &r.at.args);
            r.at.run = now;
            n += 1;
        }
        return n;
    }
};

const Timing = tp.Timing;

/// The load's prepare hooks, then one that frees the layer's bf16 projection copies once they are packed into
/// l.extra[0] (only the lane matmul reads the projections): layer by layer, so the
/// load's peak never holds both forms of every layer. The pointers stay set (their shapes size the packed form).
/// cuda_attn's prepare already frees the packed sources; this catches the rest (w.drop skips freed pointers).
const prepare_dropping = fwd.prepare ++ [_]weights.Prepare{dropCopies};

fn dropCopies(w: *weights.Weights, l: *weights.Layer) anyerror!void {
    if (l.extra[0] == 0) return; // not packed: keep them
    for ([_]u64{ l.wq, l.wk, l.wv, l.wr, l.wg, l.wsink, l.wo, l.k_wproj, l.v_wproj }) |p| w.drop(p);
}

fn env(name: [*:0]const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else "";
}

fn hashFile(gpa: std.mem.Allocator, io: std.Io, h: *std.crypto.hash.sha2.Sha256, path: []const u8) !void {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(2 << 30)) catch |err| switch (err) {
        error.FileNotFound => return h.update("(none)"),
        else => return err,
    };
    defer gpa.free(bytes);
    h.update(bytes);
}

/// Bytes of the weights rank `rank` of `world` loads for layers [0, layers) (0: all), with rank 0's embed (the whole
/// table, as loaded before `embed.spread`), the other ranks' rows of it (cuda_embed.zig) and every rank's share of
/// the head.
pub fn rankWeightBytes(m: *const ink.describe.Model, rank: u32, layers: u32) u64 {
    const n = if (layers == 0) m.c.text.layers else layers;
    var total: u64 = 0;
    for (m.p.ranks[rank].pieces) |piece| {
        const t = m.inv.tensors[piece.tensor];
        if (!std.mem.startsWith(u8, t.name, "model.llm.")) continue;
        if (m.roles[piece.tensor].layer) |li| {
            if (li < n) total += piece.run.bytes();
        } else total += piece.run.bytes();
    }
    const world = m.p.split.world;
    if (rank > 0 and world > 1) {
        const rows = embed.rowsOf(m.c.text.vocab, rank, world);
        if (rows.len() > 0) total += @as(u64, rows.len()) * H * 2 + H * 4;
    }
    return total;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

test "inkling2 cuda engine: the settings fingerprint hashes the wide unembed's switch beside v0.2's settings text" {
    const Sha = std.crypto.hash.sha2.Sha256;
    const digest = struct {
        fn of(base: []const u8, cut_off: bool, wide: bool) [32]u8 {
            var h = Sha.init(.{});
            settingsHash(&h, base, cut_off, wide);
            var d: [32]u8 = undefined;
            h.final(&d);
            return d;
        }
    }.of;
    // a settings text that fills v0.2's 768-byte buffer to the last byte (the review's case: a long TF_INK2_CK_PROFILE)
    var long: [768]u8 = undefined;
    @memset(&long, '0');
    for ([_][]const u8{ "world 4 layers 48", long[0..742], long[0..760], long[0..768] }) |base| {
        for ([_]bool{ false, true }) |cut_off| {
            // at the default (wide on) the digest is v0.2's: the draft cut's text, then the settings text, nothing else
            var v = Sha.init(.{});
            if (cut_off) v.update("TF_INK2_DRAFT_CUT=0 ");
            v.update(base);
            var dv: [32]u8 = undefined;
            v.final(&dv);
            try std.testing.expectEqualSlices(u8, &dv, &digest(base, cut_off, true));
            // the switch set differently changes the digest (a refusal at the comparison), whatever the text's length
            try std.testing.expect(!std.mem.eql(u8, &digest(base, cut_off, true), &digest(base, cut_off, false)));
        }
    }
}

fn envOpt(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// The kernels' launch choices as the kernels will make them (TF_INK2_FUSE, TF_INK2_CK_SHAPEK1, TF_INK2_CK_BLOCKS,
/// TF_INK2_CK_SHAPEW, in that order; null: unset), parsed, resolved and hashed in a fixed binary form, each only where
/// it differs from the resolved built-in default (a spelled-out default adds no byte either): equivalent spellings (unset or `lat`, flags in any order, 0 or unset blocks, an
/// empty table) hash alike and an all-default run adds no byte, while two ranks whose choices differ cannot hash alike.
/// A value the kernels would refuse is hashed as refused (its name), so ranks that differ in it stop at the agreement
/// and ranks that agree on it all stop at the kernels' load.
/// `win`: the window shapes a K1 row left unset falls back to (Kernels.ck_rows: TF_INK2_CK_SHAPE1, else the GB10 table
/// on an integrated GPU), else ckShape's.
fn hashLaunchChoices(h: anytype, v: [4]?[]const u8, win: moe.Rows) void {
    if (v[0]) |x| {
        if (fuse.Mode.parse(x)) |m| {
            if (!std.meta.eql(m, fuse.Mode.lat)) {
                h.update("fuse:");
                inline for (@typeInfo(fuse.Mode).@"struct".field_names) |name| h.update(if (@field(m, name)) "1" else "0");
                h.update(" ");
            }
        } else |_| h.update("fuse:refused ");
    }
    if (v[1]) |x| {
        if (moe.parseK1(x)) |t| {
            var r = t;
            for (&r, 0..) |*pf, f| for (pf, 0..) |*pp, p| for (pp, 0..) |*e, i| {
                if (e.* == null) e.* = win[f][p][i] orelse moe.ckShape(moe.topk * (i + 1));
            };
            if (!std.meta.eql(r, moe.k1Rows())) hashTable(h, "k1:", &r);
        } else |_| h.update("k1:refused ");
    }
    if (v[2]) |x| {
        if (moe.parseBlockCap(x)) |n| {
            if (n != moe.block_cap_default) {
                var b: [8]u8 = undefined;
                std.mem.writeInt(u64, &b, n, .little);
                h.update("blocks:");
                h.update(&b);
            }
        } else |_| h.update("blocks:refused ");
    }
    if (v[3]) |x| {
        if (moe.parseWide(x)) |t| {
            // resolved: a band left to the default takes ckShape's shape 3 (calls of 9-64 rows), so an entry of 3:3
            // and no entry are the same choice
            var r = t;
            for (&r) |*pf| for (pf) |*pp| for (pp) |*e| {
                if (e.* == null) e.* = moe.ckShape(moe.topk * moe.window_rows + moe.topk);
            };
            var d: moe.Wide = undefined;
            for (&d) |*pf| for (pf) |*pp| for (pp) |*e| {
                e.* = moe.ckShape(moe.topk * moe.window_rows + moe.topk);
            };
            if (!std.meta.eql(r, d)) hashTable(h, "wide:", &r);
        } else |_| h.update("wide:refused ");
    }
}

/// A shape table ([format][pass][entry] ?u32) as a tag and one byte an entry (255: none; shapes are below 255).
fn hashTable(h: anytype, tag: []const u8, t: anytype) void {
    h.update(tag);
    for (t) |pf| for (pf) |pp| for (pp) |e| h.update(&[_]u8{if (e) |x| @intCast(x) else 255});
}

test "inkling2 cuda engine: the launch choices hash by what they choose" {
    const Sha = std.crypto.hash.sha2.Sha256;
    // the GB10 window table (an integrated GPU, TF_INK2_CK_SHAPE1 unset)
    const gb10: moe.Rows = comptime blk: {
        var t: moe.Rows = undefined;
        for (0..2) |f| for (0..2) |p| for (0..moe.window_rows) |r| {
            t[f][p][r] = moe.gb10_rows[f][p][r];
        };
        break :blk t;
    };
    const Lc = struct {
        fn of(v: [4]?[]const u8) [32]u8 {
            return ofw(v, gb10);
        }
        fn ofw(v: [4]?[]const u8, win: moe.Rows) [32]u8 {
            var h = Sha.init(.{});
            hashLaunchChoices(&h, v, win);
            var out: [32]u8 = undefined;
            h.final(&out);
            return out;
        }
    };
    const none = Lc.of(.{ null, null, null, null });
    var empty = Sha.init(.{});
    try std.testing.expectEqual(none, empty.finalResult()); // all default: no byte added
    // the raw-string form 1c442ae55 hashed (name=value, set ones only) got both of the review's examples wrong:
    const Raw = struct {
        fn of(v: [4]?[]const u8) [32]u8 {
            var h = Sha.init(.{});
            for ([_][]const u8{ "TF_INK2_FUSE", "TF_INK2_CK_SHAPEK1", "TF_INK2_CK_BLOCKS", "TF_INK2_CK_SHAPEW" }, v) |n, x| if (x) |y| {
                h.update(n);
                h.update("=");
                h.update(y);
                h.update(" ");
            };
            var out: [32]u8 = undefined;
            h.final(&out);
            return out;
        }
    };
    const ex1a: [4]?[]const u8 = .{ null, null, "1 TF_INK2_CK_SHAPEW=4,4,4,4,4", null }; // cap: default, shapes: 3
    const ex1b: [4]?[]const u8 = .{ null, null, "1", "4,4,4,4,4" }; // cap 1, shapes 4
    const ex2a: [4]?[]const u8 = .{ null, null, null, null };
    const ex2b: [4]?[]const u8 = .{ "lat", null, null, null }; // the same forms
    try std.testing.expectEqual(Raw.of(ex1a), Raw.of(ex1b)); // different choices, one hash: agreed wrongly
    try std.testing.expect(!std.mem.eql(u8, &Raw.of(ex2a), &Raw.of(ex2b))); // the same choices: refused wrongly
    try std.testing.expect(!std.mem.eql(u8, &Lc.of(ex1a), &Lc.of(ex1b)));
    try std.testing.expectEqual(Lc.of(ex2a), Lc.of(ex2b));
    // equivalent spellings agree
    try std.testing.expectEqual(none, Lc.of(.{ "lat", null, null, null }));
    try std.testing.expectEqual(none, Lc.of(.{ "attn,mlp,route,proj,conv", "", "0", "" }));
    try std.testing.expectEqual(Lc.of(.{ "conv,moe,attn", null, null, null }), Lc.of(.{ "attn,conv,moe", null, null, null }));
    try std.testing.expectEqual(Lc.of(.{ null, null, "48", null }), Lc.of(.{ null, null, "048", null }));
    // different choices differ, however their strings line up (one rank's BLOCKS swallowing the next setting)
    try std.testing.expect(!std.mem.eql(u8, &Lc.of(.{ null, null, "1 TF_INK2_CK_SHAPEW=4,4,4,4,4", null }), &Lc.of(.{ null, null, "1", "4,4,4,4,4" })));
    try std.testing.expect(!std.mem.eql(u8, &Lc.of(.{ null, null, null, "3,3,3,3,3" }), &Lc.of(.{ null, null, null, "3,3,3,3,10" })));
    try std.testing.expect(!std.mem.eql(u8, &none, &Lc.of(.{ "attn,mlp,route,proj,conv,segs", null, null, null })));
    try std.testing.expect(!std.mem.eql(u8, &none, &Lc.of(.{ null, "4", null, null })));
    try std.testing.expect(!std.mem.eql(u8, &none, &Lc.of(.{ null, null, "x", null }))); // refused
    // each setting spelled as its built-in default agrees with it unset
    try std.testing.expectEqual(none, Lc.of(.{ "attn,mlp,route,proj,conv", null, null, null }));
    try std.testing.expectEqual(none, Lc.of(.{ null, "", null, null }));
    // K1 spelled out as its table (per format: its NVFP4 rows; the MXFP8 rows differ, so a one-list spelling of
    // shape 10 everywhere is a different choice), and a K1 row left unset resolving to the window shape it takes
    try std.testing.expect(!std.mem.eql(u8, &none, &Lc.of(.{ null, "10", null, null })));
    const all4: moe.Rows = @splat(@splat(@splat(4)));
    try std.testing.expectEqual(Lc.ofw(.{ null, "10,10,4", null, null }, all4), Lc.ofw(.{ null, "10,10", null, null }, all4));
    try std.testing.expect(!std.mem.eql(u8, &Lc.of(.{ null, "10,10,4", null, null }), &Lc.of(.{ null, "10,10", null, null }))); // MXFP8's row 3 is 5 on GB10
    try std.testing.expectEqual(none, Lc.of(.{ null, null, "1048576", null }));
    try std.testing.expectEqual(none, Lc.of(.{ null, null, "99999999", null })); // above the cap's ceiling: no cap
    try std.testing.expectEqual(none, Lc.of(.{ null, null, null, "3,3,3,3,3" }));
    try std.testing.expectEqual(none, Lc.of(.{ null, null, null, "3:3,3,3,3,3;3,3,3,3,3:3" }));
    try std.testing.expect(!std.mem.eql(u8, &none, &Lc.of(.{ null, null, null, "3,3,3,3,10" })));
}

test "inkling2 cuda engine: a graph's key is its segments' slots and rows" {
    const k = try Key.of(&.{ .{ .slot = 2, .pos0 = 3000, .rows = 1 }, .{ .slot = 0, .pos0 = 77, .rows = 8 } });
    try std.testing.expectEqual(@as(u8, 2), k.n);
    try std.testing.expect(k.eql(try Key.of(&.{ .{ .slot = 2, .pos0 = 1, .rows = 1 }, .{ .slot = 0, .pos0 = 5, .rows = 8 } })));
    try std.testing.expect(!k.eql(try Key.of(&.{ .{ .slot = 0, .pos0 = 77, .rows = 8 }, .{ .slot = 2, .pos0 = 3000, .rows = 1 } })));
    try std.testing.expect(!k.eql(try Key.of(&.{.{ .slot = 2, .pos0 = 3000, .rows = 1 }})));
    try std.testing.expectError(error.BadSegments, Key.of(&.{.{ .slot = 0, .pos0 = 1, .rows = 9 }}));
    var on_run = k;
    on_run.ring = 1; // segment 0 (one row) on its run: another graph
    try std.testing.expect(!k.eql(on_run));
    // cache units: 1 + segments / 4
    try std.testing.expectEqual(@as(usize, 1), k.units());
    var eight: [max_segs]Segment = undefined;
    for (&eight, 0..) |*g, i| g.* = .{ .slot = @intCast(i), .pos0 = 1, .rows = 1 };
    try std.testing.expectEqual(@as(usize, 3), (try Key.of(&eight)).units());
    try std.testing.expectEqual(@as(usize, 2), (try Key.of(eight[0..4])).units());
    var h: Header = .{ .kind = 1 };
    try h.setTokens(&.{ 7, 200057 });
    const bytes = std.mem.asBytes(&h);
    try std.testing.expectEqual(@as(i32, 200057), std.mem.readInt(i32, bytes[Header.tokens_at + 4 ..][0..4], .little));
    try h.setSegs(&.{ .{ .slot = 1, .pos0 = 3071, .rows = 8 }, .{ .slot = 3, .pos0 = 12345, .rows = 1 } });
    try std.testing.expectEqual(@as(i32, 12345), std.mem.readInt(i32, bytes[@intCast(Header.segPosAt(1))..][0..4], .little));
}

test "inkling2 cuda engine: a step's header is 256 bytes and keeps its fields" {
    var h: Header = .{ .kind = 1, .slot = 3, .rows = 2048, .pos0 = 1 << 40 };
    try h.setSegs(&.{ .{ .slot = 2, .pos0 = 3000, .rows = 1 }, .{ .slot = 0, .pos0 = 77, .rows = 8 } });
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), std.mem.asBytes(&h));
    try std.testing.expectEqual(h, back);
    try std.testing.expectEqual(Header.magic_word, back.magic);
    var buf: [fwd.max_segs]Segment = undefined;
    const segs = try back.getSegs(&buf);
    try std.testing.expectEqual(@as(usize, 2), segs.len);
    try std.testing.expectEqual(Segment{ .slot = 0, .pos0 = 77, .rows = 8 }, segs[1]);
    try std.testing.expectError(error.BadSegments, h.setSegs(&.{.{ .slot = 1 << 16, .pos0 = 0, .rows = 1 }}));
}

test "inkling2 cuda engine: the graph cache holds every one-stream key at 8 slots" {
    try std.testing.expect(graphs_default >= 8 * graph_seg_rows and graph_rows <= head_rows);
}

test "inkling2 cuda engine: at the defaults the settings fingerprint is v0.2's digest of its settings text" {
    // Engine.agree's settings part, composed as it calls it: the draft cut's and the wide unembed's switches
    // (settingsHash), then the kernels' launch choices (hashLaunchChoices), each read from this process's environment.
    // With every one of them at its default (none of TF_INK2_DRAFT_CUT, TF_INK2_UNEMBED_WIDE, TF_INK2_FUSE,
    // TF_INK2_CK_SHAPEK1, TF_INK2_CK_BLOCKS, TF_INK2_CK_SHAPEW set) the digest must be v0.2's: sha256 of the settings
    // text alone, as v0.2 hashed it (TF_INK2_SHARED_CUT is never hashed).
    for ([_][*:0]const u8{ "TF_INK2_DRAFT_CUT", "TF_INK2_UNEMBED_WIDE", "TF_INK2_FUSE", "TF_INK2_CK_SHAPEK1", "TF_INK2_CK_BLOCKS", "TF_INK2_CK_SHAPEW" }) |n|
        if (std.c.getenv(n) != null) return error.SkipZigTest;
    const Sha = std.crypto.hash.sha2.Sha256;
    const base = "world 4 layers 48 context 131072 scratch 0 rows 2048 mode default attn kx TF_INK2_XROWS= TF_INK2_TIME_EXCHANGE=";
    for ([_]bool{ true, false }) |drafter| {
        var h = Sha.init(.{});
        settingsHash(&h, base, drafter and dspark.draftCut() == .off, @import("cuda_head.zig").wideOn());
        hashLaunchChoices(&h, .{ envOpt("TF_INK2_FUSE"), envOpt("TF_INK2_CK_SHAPEK1"), envOpt("TF_INK2_CK_BLOCKS"), envOpt("TF_INK2_CK_SHAPEW") }, @splat(@splat(@splat(null))));
        var got: [32]u8 = undefined;
        h.final(&got);
        var v02: [32]u8 = undefined;
        Sha.hash(base, &v02, .{});
        try std.testing.expectEqualSlices(u8, &v02, &got);
    }
}
