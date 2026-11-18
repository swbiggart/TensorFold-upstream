//! Inkling-2-Small's DSpark drafter (the checkpoint's mtp.safetensors: 5 dense layers, a block of 8, a recurrent Markov
//! head of rank 256) on the ranks' group, split as the target is: rank r of W holds query heads 48r/W.., key heads
//! 8r/W.., a W-th of the MLP's 18,432 and a W-th of the vocabulary (its unembed rows and the head folds). Every rank
//! runs every step on the same rows; the two sums a layer are fp32 partials gathered and added in rank order, and the
//! Markov head's pick gathers each rank's best. The torch reference is tools/zig/inkling2_dspark_ref.py.
//!
//! Math (TokenSpeed's native draft, models/inkling2_mtp.py and execution/drafter/inkling2.py, MIT):
//!   taps       the target's partial entering layers 5, 11, ..., 41 (layer 4, 10, ..., 40's output rows) and the
//!              final normed rows: 8 taps of 4608 a position (`Taps`, captured by every forward)
//!   context    draft layer i, every target position: mix_i = dflare_norm_i(sum_t tap_t * alpha[i, t]) with
//!              alpha = bf16(softmax(fuse_weight)); K = rope(k_norm_ctx(mix_i Wkc^T)), V = mix_i Wvc^T, stored by
//!              position, the local layers' in mirrored rings of 1024 + 2048 positions (`Caches`; `absorb`, in every forward:
//!              rows past the kept ones are rewritten by the next verify)
//!   block      anchor p (the token the target drew last): rows normed-embed(anchor), mask_embed x 7 at positions
//!              p..p+7; per layer: pre-norm, q / k head norms and rope, v raw; every row sees context [0, p) (local
//!              layers 0, 1, 3, 4: the last 1024) and the whole block; o, add-norm, SwiGLU 18,432, add-norm
//!   head       rows 1..7: base = row . unembed^T (bf16, uncapped); state 256 from zeros, previous = the anchor,
//!              f = Markov mix of the taps at p - 1; per step: projected = bf16(rnn_proj [state, w1[prev], row, f]),
//!              state = g * state + (1 - g) tanh(c), latent = tanh(out); logits = softcap30(base + bf16(fold_b +
//!              latent fold_w^T)) over the valid vocabulary; the greedy pick is the next step's previous token
//!   folds      fold_w = unembed . hidden_out, fold_b = unembed . hidden_out_bias in fp64, stored bf16: precomputed by
//!              `tf-inkling2-fold` (fold.zig) or `inkling2_dspark_ref.py fold` (TF_INK2_DSPARK_FOLD, default
//!              <model>/inkling2-dspark-fold.safetensors)
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");
const tp = @import("cuda_tp.zig");

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;
const D = weights.head_dim;

pub const nlayers = 5;
pub const block = 8;
pub const drafts = block - 1;
/// Streams one pass drafts at most (`draftMany`: the step header's segments).
pub const most_streams = 8;
pub const ntaps = 8;
pub const window = 1024;
pub const mrank = 256;
const theta: f32 = 1e6;
const eps: f32 = 1e-6;
const cap: f32 = 30.0;
const heads_all = 48;
const kv_all = 8;
const inter_all = 18432;
const split_keys = 256;
/// Query heads a key head serves, the packed global attention's group (inkling2_dspark.cu PK_G).
const pack_group = 6;
/// The packed attention's dynamic shared bytes (inkling2_dspark.cu pk_shared).
const pack_shared = 48 * 128 * 4;
/// Splits from which the global layer's block attention is packed by key head (TF_INK2_DSPARK_PACK=N; 0 never). The
/// packed form has the bits of the unpacked one, so the switch changes speed only. GB10, W=4 rank (12 q heads, 2 key
/// heads, 8 rows), attention ms unpacked -> packed: 12 splits 0.138 -> 0.151, 17 0.192 -> 0.154, 129 1.03 -> 0.44,
/// 513 5.81 -> 1.49, 2,047 22.7 -> 5.40.
pub const pack_from_default: usize = 16;

/// TF_INK2_DSPARK_FP8: the draft layers' projections as MXFP8 weights (bf16 rounded to e4m3 with a power-of-two scale
/// a 32 inputs, `cuda.mxfp8.quantize`, at load): unset or "mlp" gate/up and down; "all" also qkv and o; "0" bf16 as
/// stored (v0.2's drafter). The context projections (kv) stay bf16. Drafter only: proposals may change, outputs never
/// (drafted = plain holds whatever the drafter proposes). Part of the settings the ranks agree on: a draft is a sum
/// over ranks. "mlp" is the default because "all" lost accepted tokens past the bar on 16 texts at W=1 (pooled -0.56%,
/// code -1.07%; "mlp" -0.32% and -0.79%).
pub const Fp8 = enum { none, mlp, all };

pub fn fp8Mode() Fp8 {
    const v = std.mem.span(std.c.getenv("TF_INK2_DSPARK_FP8") orelse return .mlp);
    if (std.mem.eql(u8, v, "0")) return .none;
    if (std.mem.eql(u8, v, "all")) return .all;
    return .mlp;
}

fn packFrom() usize {
    const v = std.c.getenv("TF_INK2_DSPARK_PACK") orelse return pack_from_default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch pack_from_default;
}
const pick_stride = 16;
const logit_blocks = 512;

/// Target layers whose output rows are taps 0..6 (the partial entering layers 5, 11, ..., 41); tap 7 is the final norm.
pub const tap_layers = [7]u32{ 4, 10, 16, 22, 28, 34, 40 };
pub const local = [nlayers]bool{ true, true, false, true, true };

pub fn tapOf(layer: u32) ?usize {
    for (tap_layers, 0..) |l, i| if (l == layer) return i;
    return null;
}

/// TF_INK2_DRAFT_CUT (the lone cut, cuda_lanes.zig): `0` off, `ab` cut against whole by turns, anything else (or
/// unset) on. Every rank reads it: off, the drafter makes no chances and its pick runs v0.2's argmax-only kernels (no
/// exponentials, sums or chance stores); on and `ab`, each pick also writes its probability. The ranks agree on it
/// (cuda_engine.zig's settings fingerprint), so every rank's pick runs the same kernels.
pub const Cut = enum { off, on, ab };

pub fn draftCut() Cut {
    const v = std.mem.span(std.c.getenv("TF_INK2_DRAFT_CUT") orelse return .on);
    if (std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "ab")) return .ab;
    return .on;
}

/// TF_INK2_SHARED_CUT, normalized (cuda_lanes.zig reads it on rank 0 for its shared rounds; every rank puts the
/// effective value in the settings fingerprint, cuda_engine.zig): unset (and any unread spelling) `deepen`, `cut` or
/// `1`, `deepen` or `2`, `0` off.
pub const SharedCut = enum { off, cut, deepen };

pub fn parseSharedCut(v: ?[]const u8) SharedCut {
    const x = v orelse return .deepen;
    if (std.mem.eql(u8, x, "cut") or std.mem.eql(u8, x, "1")) return .cut;
    if (std.mem.eql(u8, x, "deepen") or std.mem.eql(u8, x, "2")) return .deepen;
    if (std.mem.eql(u8, x, "0")) return .off;
    return .deepen;
}

pub fn sharedCut() SharedCut {
    return parseSharedCut(if (std.c.getenv("TF_INK2_SHARED_CUT")) |v| std.mem.span(v) else null);
}

/// A switch set to `0` (TF_INK2_BATCH, TF_INK2_DRAFT_STREAMS, TF_INK2_DRAFT_PRICE: each turns its feature off).
pub fn switchedOff(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return v[0] == '0' and v[1] == 0;
}

pub const Kernels = struct {
    mod: cuda.Module,
    mix: cuda.Function,
    addnorm: cuda.Function,
    rope: cuda.Function,
    rows: cuda.Function,
    attn: cuda.Function,
    merge: cuda.Function,
    swiglu: cuda.Function,
    mstep: cuda.Function,
    mlogits: cuda.Function,
    mpick: cuda.Function,
    mmerge: cuda.Function,
    mcat: cuda.Function,
    /// v0.2's argmax-only pick (`Drafter.chances_on` false)
    mlogits_argmax: cuda.Function,
    mpick_argmax: cuda.Function,
    mmerge_argmax: cuda.Function,
    /// the global layer's cache sites in the KV page pool (inkling2_dspark.cu `_pg`: rope's and rows' writes, the
    /// block attention's reads), one source with the table forms above
    rope_pg: cuda.Function,
    rows_pg: cuda.Function,
    attn_pg: cuda.Function,
    /// the global layer's block attention packed by key head (inkling2_dspark.cu `attn_pk_body`: attn's bits, each key
    /// and value read once for the group's query heads), table and paged forms
    attn_pk: cuda.Function,
    attn_pk_pg: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_dspark);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (.{ "mix", "addnorm", "rope", "rows", "attn", "merge", "swiglu", "mstep", "mlogits", "mpick", "mmerge", "mcat", "mlogits_argmax", "mpick_argmax", "mmerge_argmax" }) |n| {
            @field(k, n) = try mod.function(if (comptime std.mem.eql(u8, n, "merge")) "tf_ink2_dspark_attn_merge" else "tf_ink2_dspark_" ++ n);
        }
        inline for (.{ "rope", "rows", "attn" }) |n| @field(k, n ++ "_pg") = try mod.function("tf_ink2_dspark_" ++ n ++ "_pg");
        k.attn_pk = try mod.function("tf_ink2_dspark_attn_pk");
        k.attn_pk_pg = try mod.function("tf_ink2_dspark_attn_pk_pg");
        return k;
    }
};

pub const Layer = struct {
    in_norm: u64 = 0,
    post_norm: u64 = 0,
    dflare: u64 = 0,
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    k_norm_ctx: u64 = 0,
    qkv: cuda.qmmf.Weight = undefined, // [q (hq*D) | k (hk*D) | v (hk*D)] x H
    kv: cuda.qmmf.Weight = undefined, // the context's [k | v] x H
    o: cuda.qmmf.Weight = undefined, // H x hq*D (the rank's columns)
    gu: cuda.qmmf.Weight = undefined, // [gate | up] (the rank's I rows each) x H
    down: cuda.qmmf.Weight = undefined, // H x I (the rank's columns)
};

/// Positions a local layer's ring holds: the `window` positions a draft reads before its anchor, plus one forward's
/// rows (at most `chunk`: the drafter's `cap`, the taps' rows), as the target's sliding layers hold window - 1 + 2048
/// (cuda_state.zig). A position is evicted only by a write `ring` positions later; writes reach at most a verify
/// window (8 rows) past an anchor in serving and one forward's rows in any case, so no write evicts a position a later
/// draft reads. A slot of fewer positions keeps them all (no ring). TF_INK2_DSPARK_RINGS=0 (A/B): never a ring.
pub fn ringRows(context: usize, chunk: usize) usize {
    if (std.c.getenv("TF_INK2_DSPARK_RINGS")) |v| if (v[0] == '0') return context;
    return @min(context, window + chunk);
}

/// Rows a local layer takes: a ring of `ringRows` positions plus its first `window` rows mirrored after it (each
/// write to row r < window also goes to row r + ring), so the `window` positions a draft reads are always consecutive
/// rows from (p - window) % ring and the block attention reads them as it reads a position-indexed table; or the
/// slot's positions when it needs no ring.
pub fn localRows(context: usize, chunk: usize) usize {
    const ring = ringRows(context, chunk);
    return if (ring < context) ring + window else context;
}

/// A slot's draft caches: each layer's context keys and values (the rank's key heads), the global layer's by
/// position, each local layer's in a mirrored ring (`localRows`; the last `window` positions are all a draft reads
/// there). Layer l's keys, then its values, then layer l + 1's. On the KV page pool (stage 2c, `pg`) the global
/// layer's keys and values live in the pool's last two regions at the slot's own pages (the target's page table:
/// position p at row p % 256 of page ids[p / 256]), and `buf` holds the local layers alone.
pub const Caches = struct {
    buf: cuda.DeviceBuffer,
    context: usize, // positions a slot holds
    ring: usize, // positions a local layer's ring holds (ringRows; = context: no ring)
    row: usize, // bytes a position
    pg: ?Paged = null,

    /// The global layer on pages: its keys' and values' regions (page 0's first row), the slot's device page table
    /// (i32 ids) and a page's stride in elements (the paged kernels' `page`; bytes = 2 x).
    pub const Paged = struct { k: u64, v: u64, table: u64, page: usize };

    /// Rows layer l's keys (and its values) take in `buf` (0: the global layer on pages).
    pub fn rows(c: *const Caches, l: usize) usize {
        if (!local[l] and c.pg != null) return 0;
        return if (local[l] and c.ring < c.context) c.ring + window else c.context;
    }

    /// The ring layer l is indexed by (position p at row p % ring, rows below `window` mirrored at + ring), or 0:
    /// position-indexed.
    pub fn wrap(c: *const Caches, l: usize) usize {
        return if (local[l] and c.ring < c.context) c.ring else 0;
    }

    /// Layer l's keys: its rows in `buf`, or on pages the region's page 0 (a row's address takes its page from the
    /// table).
    pub fn keys(c: *const Caches, l: usize) u64 {
        if (!local[l]) if (c.pg) |p| return p.k;
        var at: usize = 0;
        for (0..l) |j| at += 2 * c.rows(j) * c.row;
        return c.buf.ptr + at;
    }

    pub fn values(c: *const Caches, l: usize) u64 {
        if (!local[l]) if (c.pg) |p| return p.v;
        return c.keys(l) + c.rows(l) * c.row;
    }

    /// The global layer on pages as a paged Kv (the target's form: the prompt cache's and the synthetic fill's row
    /// walks), or null.
    pub fn pagedKv(c: *const Caches, entries: usize) ?@import("cuda_attn.zig").Kv {
        const p = c.pg orelse return null;
        return .{ .k = p.k, .v = p.v, .slots = entries * split_keys, .heads = @intCast(c.row / (D * 2)), .window = 0, .pages = p.table, .page = p.page };
    }

    /// A slot's bytes: the global layer's keys and values at every position (none in the slot on pages), the local
    /// layers' at `localRows`.
    pub fn bytes(hk: usize, context: usize, chunk: usize, paged: bool) usize {
        var n: usize = 0;
        for (local) |loc| n += if (loc) localRows(context, chunk) else if (paged) 0 else context;
        return 2 * n * hk * D * 2;
    }
};

/// One stream's block in `draftMany`: its slot (its caches), anchor position p, the anchor token (walk's first) and
/// the taps' row of position p - 1 (its Markov features).
pub const Item = struct { slot: u32, p: usize, anchor: u32, feat_row: usize };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: Kernels,
    rank: u32,
    world: u32,
    group: ?*const cuda.tp_group.Group,
    hq: usize,
    hk: usize,
    inter: usize,
    vlo: usize,
    vn: usize,
    valid: usize, // valid columns of the rank's slice
    context: usize,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,
    L: [nlayers]Layer = @splat(.{}),
    alpha: u64 = 0, // fp32 [6][8]
    mask: u64 = 0, // bf16 [H]
    norm: u64 = 0, // fp32 [H]
    mdflare: u64 = 0, // fp32 [H]
    w1: u64 = 0, // bf16 [vocab][256]
    rnn_w: u64 = 0, // bf16 [768][9728]
    rnn_b: u64 = 0, // bf16 [768]
    rnn_inv: cuda.qmmf.Weight = undefined, // rnn_w[:, 512:] (row and features) packed
    unembed: u64 = 0, // bf16 [vn][H]
    fold_w: u64 = 0, // bf16 [vn][256]
    fold_b: u64 = 0, // bf16 [vn]
    // the forward's taps (bf16 [8][cap][H]) and the work buffers, for S <= most_streams streams' blocks at once
    // (stream s's block rows 8s .. 8s + 7)
    cap: usize,
    taps: u64 = 0,
    mixed: u64 = 0, // bf16 [cap][H]
    kvo: u64 = 0, // bf16 [cap][2 hk D]
    anchors: u64 = 0, // bf16 [S][H]: each block's row 0, the anchor's normed embedding (every rank)
    mask7: u64 = 0, // bf16 [7][H]: mask_embed x 7, a block's rows 1 .. 7
    x: u64 = 0, // bf16 [8S][H]: the blocks' residual
    hn: u64 = 0, // bf16 [8S][H]
    hb: u64 = 0, // bf16 [8S][H]: a sum's result
    qkv: u64 = 0, // bf16 [8S][(hq + 2hk) D]
    q: u64 = 0, // bf16 [8S][hq D]
    bk: u64 = 0, // bf16 [8S][hk D]
    att: u64 = 0, // bf16 [8S][hq D]
    gu: u64 = 0, // bf16 [8S][2I]
    act: u64 = 0, // bf16 [8S][I]
    part: u64 = 0, // fp32 [world][8S][H]: the sums' partials
    ml: u64 = 0, // one stream's block attention at a time
    acc: u64 = 0,
    nsplit_max: usize = 0,
    rows7: u64 = 0, // bf16 [7S][H]: each block's rows 1 .. 7 after the final norm
    base: u64 = 0, // bf16 [7S][vn]
    mx: u64 = 0, // bf16 [7S][2H]
    inv: u64 = 0, // fp32 [S][7][768]
    state: u64 = 0, // bf16 [S][256]
    latent: u64 = 0, // bf16 [S][256]
    best_v: u64 = 0, // [S][logit_blocks]
    best_i: u64 = 0,
    best_s: u64 = 0, // fp32 [S][logit_blocks]: each CTA's sum of exp(capped - cap) (the pick's probability)
    pairs: u64 = 0, // [world][S][pick_stride]: (value, id, exp sum) a rank's pick
    walk: u64 = 0, // i32 [S][8]
    chances: u64 = 0, // fp32 [S][8]: chances[8s + 1 + j] = the head's probability of walk's pick j (`chancesMany`)
    chances_on: bool = true, // `draftCut` at load: each pick also makes its chance; false: the argmax-only kernels
    lane_part: u64 = 0,
    lane_part_bytes: usize = 0,
    caches: std.ArrayList(?Caches) = .empty,
    /// The global layer of every slot on the KV page pool (the engine's `paged`; stage 2c): a slot's caches hold the
    /// local layers alone and `bindOn` names its pages.
    paged: bool = false,
    /// `packFrom` at load: the global layer's attention is packed by key head from this many splits (0 never).
    pack_from: usize = pack_from_default,

    pub fn deinit(dr: *Drafter) void {
        for (dr.caches.items) |*c| if (c.*) |*x| x.buf.free();
        dr.caches.deinit(dr.gpa);
        for (dr.own.items) |*b| b.free();
        dr.own.deinit(dr.gpa);
        dr.k.mod.unload();
        dr.gpa.destroy(dr);
    }

    fn alloc(dr: *Drafter, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(dr.d, @max(n, 256));
        errdefer b.free();
        try dr.own.append(dr.gpa, b);
        dr.bytes += n;
        return b.ptr;
    }

    fn put(dr: *Drafter, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(dr.d, bytes);
        errdefer b.free();
        try dr.own.append(dr.gpa, b);
        dr.bytes += bytes.len;
        return b.ptr;
    }

    /// The slot's caches, made on first use (on pages the engine's `bindOn` makes them first).
    pub fn bind(dr: *Drafter, slot: u32) !*Caches {
        if (dr.paged and (slot >= dr.caches.items.len or dr.caches.items[slot] == null)) return error.DraftPagesUnbound;
        return dr.bindOn(slot, null);
    }

    /// The slot's caches, made on first use; `pg`: its global layer on the KV page pool (required when `paged`).
    pub fn bindOn(dr: *Drafter, slot: u32, pg: ?Caches.Paged) !*Caches {
        if (slot >= dr.caches.items.len) try dr.caches.appendNTimes(dr.gpa, null, slot + 1 - dr.caches.items.len);
        if (dr.caches.items[slot] == null) {
            if (dr.paged != (pg != null)) return error.DraftPagesUnbound;
            const n = Caches.bytes(dr.hk, dr.context, dr.cap, dr.paged);
            var b = try cuda.DeviceBuffer.alloc(dr.d, n);
            errdefer b.free();
            try b.fill8(0, null);
            dr.caches.items[slot] = .{ .buf = b, .context = dr.context, .ring = ringRows(dr.context, dr.cap), .row = dr.hk * D * 2, .pg = pg };
        }
        return &dr.caches.items[slot].?;
    }

    /// Bytes a slot's caches take.
    pub fn slotBytes(dr: *const Drafter) usize {
        return Caches.bytes(dr.hk, dr.context, dr.cap, dr.paged);
    }

    /// A slot's draft caches at `context` positions (admission's largest context that fits, cuda_native.zig).
    pub fn slotBytesAt(dr: *const Drafter, context: usize) usize {
        return Caches.bytes(dr.hk, context, dr.cap, dr.paged);
    }

    /// The global layer's rope into the pool's pages (inkling2_dspark.cu tf_ink2_dspark_rope_pg): row r of x lands at
    /// position pos0 + r (pos0 read at `posp` when not 0); the same arithmetic as `rope`.
    fn ropePaged(dr: *const Drafter, c: *const Ctx, x: u64, x_ld: usize, heads: usize, rows: usize, gain: u64, pg: Caches.Paged, out_ld: usize, pos0: usize, posp: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(x_ld)));
        a.add(gain);
        a.add(pg.k);
        a.add(@as(i64, @intCast(out_ld)));
        a.add(@as(i64, @intCast(pos0)));
        a.add(theta);
        a.add(eps);
        a.add(posp);
        a.add(pg.table);
        a.add(@as(i64, @intCast(pg.page)));
        try dr.launch(c, dr.k.rope_pg, .{ heads, rows, 1 }, 32, &a);
    }

    /// The global layer's values into the pool's pages (tf_ink2_dspark_rows_pg; `rowsAt`'s bytes).
    fn rowsPaged(dr: *const Drafter, c: *const Ctx, pg: Caches.Paged, dst_ld: usize, src: u64, src_ld: usize, width: usize, rows: usize, pos0: usize, posp: u64) !void {
        if (rows == 0) return;
        if (width % 16 != 0 or src % 16 != 0 or dst_ld % 16 != 0 or src_ld % 16 != 0 or pg.v % 16 != 0) return error.DraftRowsAlign;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(i64, @intCast(src_ld)));
        a.add(pg.v);
        a.add(@as(i64, @intCast(dst_ld)));
        a.add(@as(i32, @intCast(width / 16)));
        a.add(@as(i64, @intCast(pos0)));
        a.add(posp);
        a.add(pg.table);
        a.add(@as(i64, @intCast(pg.page * 2)));
        try dr.launch(c, dr.k.rows_pg, .{ rows, 1, 1 }, 128, &a);
    }

    pub fn tap(dr: *const Drafter, t: usize) u64 {
        return dr.taps + t * dr.cap * H * 2;
    }

    fn launch(dr: *const Drafter, c: *const Ctx, f: cuda.Function, grid: [3]usize, threads: usize, a: *cuda.Args) !void {
        return dr.launchShared(c, f, grid, threads, 0, a);
    }

    fn launchShared(dr: *const Drafter, c: *const Ctx, f: cuda.Function, grid: [3]usize, threads: usize, shared: u32, a: *cuda.Args) !void {
        _ = dr;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(threads) }, .shared = shared }, c.stream, a);
    }

    fn lin(dr: *Drafter, c: *const Ctx, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, out: u64, f32out: bool) !void {
        const lane = &c.k.attn.lane;
        const need = lane.partBytes(rows, w);
        if (need > dr.lane_part_bytes) return error.DraftLanePartShort;
        try lane.matmulOut(c.stream, x, ldx, rows, w, out, if (f32out) .f32 else .bf16, dr.lane_part);
    }

    fn mixRows(dr: *Drafter, c: *const Ctx, row0: usize, rows: usize, stage: usize, gain: u64, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(dr.taps + row0 * H * 2);
        a.add(@as(i64, @intCast(dr.cap * H)));
        a.add(@as(i32, ntaps));
        a.add(dr.alpha + stage * ntaps * 4);
        a.add(gain);
        a.add(out);
        a.add(eps);
        try dr.launch(c, dr.k.mix, .{ rows, 1, 1 }, H / 8, &a);
    }

    /// `ring` not 0: `out` is a mirrored ring's base (`Caches`), row r landing at out + ((pos0 + r) % ring) * out_ld
    /// and, below `window`, again `ring` rows on; else `posp` not 0:
    /// `out` is a position-indexed table's base (the rows land at out + pos0 * out_ld, as the host form's `out`
    /// points); else `out` as given. `posp` not 0: pos0 is read there on the device. The arithmetic is the same in
    /// every form.
    pub fn rope(dr: *const Drafter, c: *const Ctx, x: u64, x_ld: usize, heads: usize, rows: usize, gain: u64, out: u64, out_ld: usize, pos0: usize, posp: u64, ring: usize) !void {
        if (ring != 0 and rows > ring) return error.DraftRing;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(x_ld)));
        a.add(gain);
        a.add(out);
        a.add(@as(i64, @intCast(out_ld)));
        a.add(@as(i64, @intCast(pos0)));
        a.add(theta);
        a.add(eps);
        a.add(posp);
        a.add(@as(i64, @intCast(ring)));
        a.add(@as(i64, window));
        try dr.launch(c, dr.k.rope, .{ heads, rows, 1 }, 32, &a);
    }

    /// `rows` rows of `width` bytes (a multiple of 16) from src (src_ld bytes apart) to rows pos0 .. of the
    /// position-indexed table `dst` (dst_ld bytes a row), or with `ring` not 0 to rows (pos0 + r) % ring of a mirrored
    /// ring (`Caches`);
    /// `posp` not 0: pos0 read there on the device.
    pub fn rowsAt(dr: *const Drafter, c: *const Ctx, dst: u64, dst_ld: usize, src: u64, src_ld: usize, width: usize, rows: usize, pos0: usize, posp: u64, ring: usize) !void {
        if (rows == 0) return;
        if (width % 16 != 0 or dst % 16 != 0 or src % 16 != 0 or dst_ld % 16 != 0 or src_ld % 16 != 0) return error.DraftRowsAlign;
        if (ring != 0 and rows > ring) return error.DraftRing;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(i64, @intCast(src_ld)));
        a.add(dst);
        a.add(@as(i64, @intCast(dst_ld)));
        a.add(@as(i32, @intCast(width / 16)));
        a.add(@as(i64, @intCast(pos0)));
        a.add(posp);
        a.add(@as(i64, @intCast(ring)));
        a.add(@as(i64, window));
        try dr.launch(c, dr.k.rows, .{ rows, 1, 1 }, 128, &a);
    }

    fn addnorm(dr: *Drafter, c: *const Ctx, h: u64, res: u64, gain: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(h);
        a.add(res);
        a.add(gain);
        a.add(out);
        a.add(eps);
        try dr.launch(c, dr.k.addnorm, .{ rows, 1, 1 }, H / 8, &a);
    }

    /// Context rows pos0 .. pos0 + R - 1 from the taps' rows row0 .. row0 + R - 1 into the slot's caches (every
    /// layer); row0 > 0: a segment of a batched forward. `posp` not 0 (a captured forward): pos0 is read there on
    /// the device, so the keys' rope and the values land at the replay's positions (pos0 is then the capture's,
    /// checked only).
    pub fn absorb(dr: *Drafter, c: *const Ctx, slot: u32, pos0: usize, R: usize, row0: usize, posp: u64) !void {
        if (row0 + R > dr.cap or pos0 + R > dr.context) return error.DraftContext;
        const cs = try dr.bind(slot);
        const kvw = dr.hk * D;
        for (&dr.L, 0..) |*l, i| {
            try dr.mixRows(c, row0, R, i, l.dflare, dr.mixed);
            try dr.lin(c, dr.mixed, H, R, l.kv, dr.kvo, false);
            if (!local[i]) if (cs.pg) |pg| {
                // the global layer on the pool's pages (stage 2c): each row to its page's row, the same bytes
                try dr.ropePaged(c, dr.kvo, 2 * kvw, dr.hk, R, l.k_norm_ctx, pg, kvw, pos0, posp);
                try dr.rowsPaged(c, pg, cs.row, dr.kvo + kvw * 2, 2 * kvw * 2, kvw * 2, R, pos0, posp);
                continue;
            };
            const ring = cs.wrap(i);
            if (posp != 0 or ring != 0) {
                try dr.rope(c, dr.kvo, 2 * kvw, dr.hk, R, l.k_norm_ctx, cs.keys(i), kvw, pos0, posp, ring);
            } else try dr.rope(c, dr.kvo, 2 * kvw, dr.hk, R, l.k_norm_ctx, cs.keys(i) + pos0 * cs.row, kvw, pos0, 0, 0);
            try dr.rowsAt(c, cs.values(i), cs.row, dr.kvo + kvw * 2, 2 * kvw * 2, kvw * 2, R, pos0, posp, ring);
        }
    }

    /// A sum's partials (this rank's fp32 [8][H] in `part`) gathered and added in rank order: bf16 [8][H] in hb.
    fn sum(dr: *Drafter, c: *const Ctx, rows: usize) !void {
        const mine = dr.part + @as(u64, dr.rank) * rows * H * 4;
        if (dr.group) |g| try g.allGather(c.stream, mine, dr.part, rows * H * 4);
        try @import("cuda_res.zig").rankSum(c, dr.part, dr.world, Mat.of(dr.hb, rows, H, .bf16));
    }

    fn partOut(dr: *const Drafter, rows: usize) u64 {
        return if (dr.world > 1) dr.part + @as(u64, dr.rank) * rows * H * 4 else dr.hb;
    }

    /// One stream's block (`draftMany` of one) with its row 0, the anchor's normed embedding, in x[0] on every rank.
    pub fn draft(dr: *Drafter, c: *const Ctx, slot: u32, p: usize, anchor: u32, feat_row: usize) !void {
        try c.copy(dr.anchors, dr.x, H * 2);
        try dr.draftMany(c, &.{.{ .slot = slot, .p = p, .anchor = anchor, .feat_row = feat_row }});
    }

    /// The blocks of S streams in one pass, each at its anchor position p (its context [0, p) absorbed): every
    /// stream's 7 greedy drafts for p + 1 .. p + 7 on the device (walk[8s + 1 .. 8s + 8]). The anchors' normed
    /// embeddings must be in `anchors` (row s: stream s's) on every rank. The ops that read a stream's own state (its
    /// rope positions, its block attention over its context, its Markov features and recurrence) run on its rows
    /// alone; the rest (norms, projections, the sums, the SwiGLU, the unembed) run once over all streams' rows and
    /// are row-invariant, so each stream's drafts are the bits of a pass of its own.
    pub fn draftMany(dr: *Drafter, c: *const Ctx, items: []const Item) !void {
        const S = items.len;
        if (S == 0 or S > most_streams) return error.DraftStreams;
        for (items) |it| if (it.p == 0 or it.p + block > dr.context) return error.DraftContext;
        const T = block;
        const N = T * S;
        try copy2d(c, dr.x, T * H * 2, dr.anchors, H * 2, H * 2, S);
        for (0..S) |s| try c.copy(dr.x + (s * T + 1) * H * 2, dr.mask7, drafts * H * 2);
        const hq = dr.hq;
        const hk = dr.hk;
        const qw = (hq + 2 * hk) * D;
        for (&dr.L, 0..) |*l, i| {
            if (i == 0) try dr.addnorm(c, 0, dr.x, l.in_norm, dr.hn, N) else try dr.addnorm(c, dr.hb, dr.x, l.in_norm, dr.hn, N);
            try dr.lin(c, dr.hn, H, N, l.qkv, dr.qkv, false);
            for (items, 0..) |it, s| {
                const cs = try dr.bind(it.slot);
                const p = it.p;
                const r0 = s * T;
                const qkv = dr.qkv + r0 * qw * 2;
                const q = dr.q + r0 * hq * D * 2;
                const bk = dr.bk + r0 * hk * D * 2;
                try dr.rope(c, qkv, qw, hq, T, l.q_norm, q, hq * D, p, 0, 0);
                try dr.rope(c, qkv + hq * D * 2, qw, hk, T, l.k_norm, bk, hk * D, p, 0, 0);
                // attention over context [lo, p) and the block; in a mirrored ring positions lo .. p - 1 are rows
                // lo % ring .. on, consecutive (nctx <= window), so the kernel reads them as a table's
                const lo: usize = if (local[i]) p -| window else 0;
                const nctx = p - lo;
                const ring = cs.wrap(i);
                if (ring != 0 and nctx > window) return error.DraftRing;
                const nsplit = (nctx + T + split_keys - 1) / split_keys;
                if (nsplit > dr.nsplit_max) return error.DraftContext;
                // the global layer packed by key head (the same bits): one CTA a (split, key head)
                const pk = !local[i] and hq == pack_group * hk and dr.pack_from != 0 and nsplit >= dr.pack_from;
                if (!local[i] and cs.pg != null) {
                    // the global layer on pages (lo = 0: split s is page s; its id read once a split)
                    const pg = cs.pg.?;
                    var a: cuda.Args = .{};
                    a.add(q);
                    a.add(pg.k);
                    a.add(pg.v);
                    a.add(@as(i64, @intCast(hk * D)));
                    a.add(@as(i32, @intCast(nctx)));
                    a.add(bk);
                    a.add(qkv + (hq + hk) * D * 2);
                    a.add(@as(i64, @intCast(qw)));
                    a.add(@as(i32, @intCast(T)));
                    a.add(@as(i32, @intCast(hq)));
                    if (!pk) a.add(@as(i32, @intCast(hq / hk)));
                    a.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
                    a.add(dr.ml);
                    a.add(dr.acc);
                    a.add(pg.table);
                    a.add(@as(i64, @intCast(pg.page)));
                    if (pk) try dr.launchShared(c, dr.k.attn_pk_pg, .{ nsplit, hk, 1 }, 256, pack_shared, &a) else try dr.launch(c, dr.k.attn_pg, .{ nsplit, hq, 1 }, 256, &a);
                } else {
                    var a: cuda.Args = .{};
                    a.add(q);
                    a.add(cs.keys(i));
                    a.add(cs.values(i));
                    a.add(@as(i64, @intCast(hk * D)));
                    a.add(@as(i64, @intCast(if (ring != 0) lo % ring else lo)));
                    a.add(@as(i32, @intCast(nctx)));
                    a.add(bk);
                    a.add(qkv + (hq + hk) * D * 2);
                    a.add(@as(i64, @intCast(qw)));
                    a.add(@as(i32, @intCast(T)));
                    a.add(@as(i32, @intCast(hq)));
                    if (!pk) a.add(@as(i32, @intCast(hq / hk)));
                    a.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
                    a.add(dr.ml);
                    a.add(dr.acc);
                    if (pk) try dr.launchShared(c, dr.k.attn_pk, .{ nsplit, hk, 1 }, 256, pack_shared, &a) else try dr.launch(c, dr.k.attn, .{ nsplit, hq, 1 }, 256, &a);
                }
                {
                    var a: cuda.Args = .{};
                    a.add(dr.ml);
                    a.add(dr.acc);
                    a.add(@as(i32, @intCast(nsplit)));
                    a.add(@as(i32, @intCast(T)));
                    a.add(@as(i32, @intCast(hq)));
                    a.add(dr.att + r0 * hq * D * 2);
                    try dr.launch(c, dr.k.merge, .{ hq, T, 1 }, D, &a);
                }
            }
            try dr.lin(c, dr.att, hq * D, N, l.o, dr.partOut(N), dr.world > 1);
            if (dr.world > 1) try dr.sum(c, N);
            try dr.addnorm(c, dr.hb, dr.x, l.post_norm, dr.hn, N);
            try dr.lin(c, dr.hn, H, N, l.gu, dr.gu, false);
            {
                var a: cuda.Args = .{};
                a.add(dr.gu);
                a.add(@as(i32, @intCast(dr.inter)));
                a.add(dr.act);
                try dr.launch(c, dr.k.swiglu, .{ (dr.inter + 255) / 256, N, 1 }, 256, &a);
            }
            try dr.lin(c, dr.act, dr.inter, N, l.down, dr.partOut(N), dr.world > 1);
            if (dr.world > 1) try dr.sum(c, N);
        }
        try dr.addnorm(c, dr.hb, dr.x, dr.norm, dr.hn, N);
        // the Markov head over each block's rows 1..7 (gathered: stream s's at rows7 + 7s)
        try copy2d(c, dr.rows7, drafts * H * 2, dr.hn + H * 2, T * H * 2, drafts * H * 2, S);
        try unembedRows(c, dr.rows7, H, dr.unembed, dr.vn, dr.base, drafts * S);
        // f = the Markov mix of the taps at p - 1 (stream s's into hb row s)
        for (items, 0..) |it, s| try dr.mixRows(c, it.feat_row, 1, 5, dr.mdflare, dr.hb + s * H * 2);
        for (0..S) |s| {
            var a: cuda.Args = .{};
            a.add(dr.rows7 + s * drafts * H * 2);
            a.add(@as(i64, H));
            a.add(dr.hb + s * H * 2);
            a.add(dr.mx + s * drafts * 2 * H * 2);
            try dr.launch(c, dr.k.mcat, .{ drafts, 1, 1 }, H / 8, &a);
        }
        try dr.lin(c, dr.mx, 2 * H, drafts * S, dr.rnn_inv, dr.inv, true);
        try c.zero(dr.state, S * mrank * 2);
        {
            var head_ids: [most_streams * block]i32 = @splat(0);
            for (items, 0..) |it, s| head_ids[s * block] = @intCast(it.anchor);
            try cuda.copy_source.enqueue(dr.d, dr.walk, std.mem.sliceAsBytes(head_ids[0 .. S * block]), c.stream);
        }
        const pair_ys: usize = pick_stride; // a stream's pair within a rank's
        const rank_pairs: usize = S * pick_stride; // a rank's pairs
        for (0..drafts) |s| {
            {
                var a: cuda.Args = .{};
                a.add(dr.inv + s * 3 * mrank * 4);
                a.add(dr.rnn_b);
                a.add(dr.rnn_w);
                a.add(@as(i64, 2 * mrank + 2 * H));
                a.add(dr.w1);
                a.add(dr.walk);
                a.add(@as(i32, @intCast(s)));
                a.add(dr.state);
                a.add(dr.latent);
                a.add(@as(i64, drafts * 3 * mrank));
                try dr.launch(c, dr.k.mstep, .{ 1, S, 1 }, 3 * mrank, &a);
            }
            {
                var a: cuda.Args = .{};
                a.add(dr.base + s * dr.vn * 2);
                a.add(dr.latent);
                a.add(dr.fold_w);
                a.add(dr.fold_b);
                a.add(@as(i32, @intCast(dr.vn)));
                a.add(@as(i32, @intCast(dr.valid)));
                a.add(cap);
                a.add(dr.best_v);
                a.add(dr.best_i);
                a.add(@as(i64, @intCast(drafts * dr.vn)));
                if (dr.chances_on) a.add(dr.best_s);
                try dr.launch(c, if (dr.chances_on) dr.k.mlogits else dr.k.mlogits_argmax, .{ logit_blocks, S, 1 }, 256, &a);
            }
            const mine = dr.pairs + @as(u64, dr.rank) * rank_pairs;
            {
                var a: cuda.Args = .{};
                a.add(dr.best_v);
                a.add(dr.best_i);
                a.add(@as(i32, logit_blocks));
                a.add(@as(i32, @intCast(dr.vlo)));
                a.add(mine);
                a.add(if (dr.world > 1) @as(u64, 0) else dr.walk);
                a.add(@as(i32, @intCast(s)));
                a.add(@as(i32, @intCast(pair_ys)));
                if (dr.chances_on) {
                    a.add(dr.best_s);
                    a.add(cap);
                    a.add(dr.chances);
                }
                try dr.launch(c, if (dr.chances_on) dr.k.mpick else dr.k.mpick_argmax, .{ 1, S, 1 }, 32, &a);
            }
            if (dr.world > 1) {
                if (dr.group) |g| try g.allGather(c.stream, mine, dr.pairs, rank_pairs);
                var a: cuda.Args = .{};
                a.add(dr.pairs);
                a.add(@as(i32, @intCast(dr.world)));
                a.add(@as(i32, @intCast(rank_pairs)));
                a.add(dr.walk);
                a.add(@as(i32, @intCast(s)));
                a.add(@as(i32, @intCast(pair_ys)));
                if (dr.chances_on) {
                    a.add(cap);
                    a.add(dr.chances);
                }
                try dr.launch(c, if (dr.chances_on) dr.k.mmerge else dr.k.mmerge_argmax, .{ 1, S, 1 }, 32, &a);
            }
        }
    }

    /// The last draft's 7 tokens (synchronizes the stream).
    pub fn proposals(dr: *const Drafter, c: *const Ctx, out: *[drafts]u32) !void {
        return dr.proposalsMany(c, &.{out});
    }

    /// The last `draftMany`'s 7 tokens of each of its streams, in its order (synchronizes the stream).
    pub fn proposalsMany(dr: *const Drafter, c: *const Ctx, out: []const *[drafts]u32) !void {
        if (out.len == 0 or out.len > most_streams) return error.DraftStreams;
        try c.stream.synchronize();
        var got: [most_streams * block]i32 = undefined;
        try cuda.memory.download(dr.d, dr.walk, std.mem.sliceAsBytes(got[0 .. out.len * block]));
        for (out, 0..) |o, s| for (o, got[s * block + 1 ..][0..drafts]) |*t, g| {
            t.* = @intCast(g);
        };
    }

    /// The last `draftMany`'s pick probabilities (each of its 7 drafts' softmax over the valid vocabulary, uncalibrated)
    /// of each of its streams, in its order (synchronizes the stream).
    pub fn chancesMany(dr: *const Drafter, c: *const Ctx, out: []const *[drafts]f32) !void {
        if (!dr.chances_on) return error.NoDraftChances; // TF_INK2_DRAFT_CUT=0
        if (out.len == 0 or out.len > most_streams) return error.DraftStreams;
        try c.stream.synchronize();
        var got: [most_streams * block]f32 = undefined;
        try cuda.memory.download(dr.d, dr.chances, std.mem.sliceAsBytes(got[0 .. out.len * block]));
        for (out, 0..) |o, s| @memcpy(o, got[s * block + 1 ..][0..drafts]);
    }
};

/// rows [n] of x (ld elements) . w[v]^T for v < vn: out bf16 [n][vn] (cuda_head's row-invariant unembed kernels; from
/// cuda_head.wide_from rows a wide form, the same bits).
fn unembedRows(c: *const Ctx, x: u64, ld: usize, w: u64, vn: usize, out: u64, n: usize) !void {
    const head = @import("cuda_head.zig");
    if (c.k.head.wide) if (head.wideForm(n)) |f| return head.unembedWide(c, f, x, ld, w, out, vn, n, vn, H);
    const tiles = [_]usize{ 1, 4, 16, 32 };
    const i: usize = if (n == 1) 0 else if (n <= 4) 1 else if (n <= 16) 2 else 3;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(ld)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(vn)));
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(vn)));
    a.add(@as(i32, H));
    const grid: cuda.Dim3 = .{ .x = @intCast((vn + 7) / 8), .y = @intCast((n + tiles[i] - 1) / tiles[i]), .z = 1 };
    try cuda.launch.launch(c.k.head.unembed_fns[i], .{ .grid = grid, .block = .{ .x = 8 * 32 } }, c.stream, &a);
}

/// `rows` rows of `width` bytes from src (src_pitch bytes apart) to dst (dst_pitch apart).
pub fn copy2d(c: *const Ctx, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, rows: usize) !void {
    if (rows == 0) return;
    const p: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = src_pitch, .dst = dst, .dst_pitch = dst_pitch, .width = width, .height = rows };
    try c.d.check(c.d.api.cuMemcpy2DAsync_v2(&p, c.stream.handle), "cuMemcpy2DAsync");
}

pub const Options = struct {
    rank: u32,
    world: u32,
    group: ?*const cuda.tp_group.Group,
    context: usize,
    cap: usize,
    lane: *const cuda.qmmf.Lane,
    /// The target's unembed rows of this rank's slice (bf16 [V/W, H]) when it holds exactly them (the vocab-split
    /// head), shared instead of read again; 0: read from the checkpoint.
    unembed: u64 = 0,
};

/// The rank's share of the drafter from `dir` (mtp.safetensors, the target's unembed rows, the folds file).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, m: *const ink.describe.Model, o: Options) !*Drafter {
    const W: usize = o.world;
    if (heads_all % W != 0 or kv_all % W != 0 or inter_all % W != 0) return error.BadDraftSplit;
    const V: usize = m.c.text.vocab;
    if (V % W != 0) return error.BadDraftSplit;
    const dr = try gpa.create(Drafter);
    errdefer gpa.destroy(dr);
    dr.* = .{ .gpa = gpa, .d = d, .k = try Kernels.load(d), .rank = o.rank, .world = o.world, .group = o.group, .hq = heads_all / W, .hk = kv_all / W, .inter = inter_all / W, .vn = V / W, .vlo = o.rank * (V / W), .valid = 0, .context = o.context, .cap = o.cap, .pack_from = packFrom() };
    errdefer {
        for (dr.own.items) |*b| b.free();
        dr.own.deinit(gpa);
        dr.k.mod.unload();
    }
    const U: usize = m.c.text.unpadded_vocab;
    dr.valid = if (U > dr.vlo) @min(U - dr.vlo, dr.vn) else 0;
    const path = try std.fs.path.join(gpa, &.{ dir, "mtp.safetensors" });
    defer gpa.free(path);
    var r = try ink.st.Reader.open(gpa, io, path);
    defer r.close();
    var packer = try cuda.modelopt.Packer.load(d);
    defer packer.unload();
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var P: Pack = .{ .dr = dr, .r = &r, .packer = &packer, .s = s };
    const hq = dr.hq;
    const hk = dr.hk;
    const I = dr.inter;
    const q0 = o.rank * hq * D;
    const k0 = o.rank * hk * D;
    const m0 = o.rank * I;
    const fp8 = fp8Mode();
    if (o.rank == 0) std.log.info("inkling2 dspark: draft projections {s} (TF_INK2_DSPARK_FP8={t})", .{ if (fp8 == .none) "bf16" else "as MXFP8 weights", fp8 });
    for (&dr.L, 0..) |*l, i| {
        var nb: [8][96]u8 = undefined;
        const n = struct {
            fn of(b: *[96]u8, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ li, leaf });
            }
        }.of;
        l.in_norm = try P.gain(try n(&nb[0], i, "input_layernorm.weight"));
        l.post_norm = try P.gain(try n(&nb[0], i, "post_attention_layernorm.weight"));
        l.dflare = try P.gain(try n(&nb[0], i, "dflare_norm.weight"));
        l.q_norm = try P.gain(try n(&nb[0], i, "self_attn.q_norm.weight"));
        l.k_norm = try P.gain(try n(&nb[0], i, "self_attn.k_norm.weight"));
        l.k_norm_ctx = try P.gain(try n(&nb[0], i, "self_attn.k_norm_ctx.weight"));
        P.mx = fp8 == .all;
        l.qkv = try P.rows(&.{ .{ try n(&nb[1], i, "self_attn.q_proj.weight"), q0, hq * D }, .{ try n(&nb[2], i, "self_attn.k_proj.weight"), k0, hk * D }, .{ try n(&nb[3], i, "self_attn.v_proj.weight"), k0, hk * D } }, H);
        l.o = try P.cols(try n(&nb[1], i, "self_attn.o_proj.weight"), H, heads_all * D, q0, hq * D);
        P.mx = fp8 != .none;
        l.gu = try P.rows(&.{ .{ try n(&nb[1], i, "mlp.gate_proj.weight"), m0, I }, .{ try n(&nb[2], i, "mlp.up_proj.weight"), m0, I } }, H);
        l.down = try P.cols(try n(&nb[1], i, "mlp.down_proj.weight"), H, inter_all, m0, I);
        P.mx = false;
        l.kv = try P.rows(&.{ .{ try n(&nb[1], i, "self_attn.k_proj_ctx.weight"), k0, hk * D }, .{ try n(&nb[2], i, "self_attn.v_proj_ctx.weight"), k0, hk * D } }, H);
    }
    {
        const fw = try r.alloc(gpa, "fuse_weight", .bf16);
        defer gpa.free(fw);
        if (fw.len != 6 * ntaps * 2) return error.BadDraftCheckpoint;
        // alpha = bf16(softmax(fp32 fuse_weight)) per stage, kept as fp32
        var al: [6 * ntaps]f32 = undefined;
        for (0..6) |st| {
            var x: [ntaps]f32 = undefined;
            var mx: f32 = -std.math.inf(f32);
            for (&x, 0..) |*v, t| {
                v.* = ink.st.bf16ToF32(std.mem.readInt(u16, fw[2 * (st * ntaps + t) ..][0..2], .little));
                mx = @max(mx, v.*);
            }
            var z: f32 = 0;
            for (&x) |*v| {
                v.* = @exp(v.* - mx);
                z += v.*;
            }
            for (x, 0..) |v, t| al[st * ntaps + t] = ink.st.bf16ToF32(ink.st.f32ToBf16(v / z));
        }
        dr.alpha = try dr.put(std.mem.sliceAsBytes(&al));
    }
    dr.mask = try P.raw("mask_embed", H * 2);
    dr.norm = try P.gain("norm.weight");
    dr.mdflare = try P.gain("markov_head.dflare_norm.weight");
    dr.w1 = try P.raw("markov_head.markov_w1.weight", V * mrank * 2);
    dr.rnn_w = try P.raw("markov_head.rnn_proj.weight", 3 * mrank * (2 * mrank + 2 * H) * 2);
    dr.rnn_b = try P.raw("markov_head.rnn_proj.bias", 3 * mrank * 2);
    dr.rnn_inv = try P.colsPacked("markov_head.rnn_proj.weight", 3 * mrank, 2 * mrank + 2 * H, 2 * mrank, 2 * H);
    // the target's unembed rows of the rank's slice (shared with the target's head when it holds them) and the folds
    if (o.unembed != 0) dr.unembed = o.unembed else {
        const t = m.inv.find("model.llm.unembed.weight") orelse return error.MissingInkling2Tensor;
        const up = try std.fs.path.join(gpa, &.{ dir, m.inv.files[t.file].name });
        defer gpa.free(up);
        const f = try std.Io.Dir.cwd().openFile(io, up, .{});
        defer f.close(io);
        const n = dr.vn * H * 2;
        const host = try gpa.alloc(u8, n);
        defer gpa.free(host);
        if (try f.readPositionalAll(io, host, m.inv.offset(t) + dr.vlo * H * 2) != n) return error.ShortRead;
        dr.unembed = try dr.put(host);
    }
    {
        const fpath = if (std.c.getenv("TF_INK2_DSPARK_FOLD")) |v| try gpa.dupe(u8, std.mem.span(v)) else try std.fs.path.join(gpa, &.{ dir, "inkling2-dspark-fold.safetensors" });
        defer gpa.free(fpath);
        var fr = ink.st.Reader.open(gpa, io, fpath) catch |err| {
            std.log.err("inkling2 dspark: the head folds {s}: {t} (make them with tf-inkling2-fold MODEL OUT --rank R --world W)", .{ fpath, err });
            return err;
        };
        defer fr.close();
        const ew = fr.find("fold_w") orelse return error.BadDraftFolds;
        const eb = fr.find("fold_b") orelse return error.BadDraftFolds;
        if (ew.dtype != .bf16 or ew.dim(0) != V or ew.dim(1) != mrank or eb.dim(0) != V) return error.BadDraftFolds;
        const hw = try gpa.alloc(u8, dr.vn * mrank * 2);
        defer gpa.free(hw);
        try fr.readInto(ew, dr.vlo * mrank * 2, hw);
        dr.fold_w = try dr.put(hw);
        const hb = try gpa.alloc(u8, dr.vn * 2);
        defer gpa.free(hb);
        try fr.readInto(eb, dr.vlo * 2, hb);
        dr.fold_b = try dr.put(hb);
    }
    // work buffers
    const ncap = o.cap;
    dr.taps = try dr.alloc(ntaps * ncap * H * 2);
    try (cuda.DeviceBuffer{ .d = d, .ptr = dr.taps, .len = ntaps * ncap * H * 2, .borrowed = true }).fill8(0, null);
    dr.mixed = try dr.alloc(ncap * H * 2);
    dr.kvo = try dr.alloc(ncap * 2 * hk * D * 2);
    const SB = most_streams * block; // block rows of every stream a pass drafts
    const SD = most_streams * drafts;
    dr.anchors = try dr.alloc(most_streams * H * 2);
    dr.mask7 = try dr.alloc(drafts * H * 2);
    for (0..drafts) |j| try (cuda.DeviceBuffer{ .d = d, .ptr = dr.mask7, .len = drafts * H * 2, .borrowed = true }).copyFrom(j * H * 2, dr.mask, H * 2, null);
    dr.x = try dr.alloc(SB * H * 2);
    dr.hn = try dr.alloc(SB * H * 2);
    dr.hb = try dr.alloc(SB * H * 2);
    dr.qkv = try dr.alloc(SB * (hq + 2 * hk) * D * 2);
    dr.q = try dr.alloc(SB * hq * D * 2);
    dr.bk = try dr.alloc(SB * hk * D * 2);
    dr.att = try dr.alloc(SB * hq * D * 2);
    dr.gu = try dr.alloc(SB * 2 * I * 2);
    dr.act = try dr.alloc(SB * I * 2);
    dr.part = try dr.alloc(W * SB * H * 4);
    dr.nsplit_max = (o.context + block + split_keys - 1) / split_keys;
    dr.ml = try dr.alloc(hq * dr.nsplit_max * block * 2 * 4);
    dr.acc = try dr.alloc(hq * dr.nsplit_max * block * D * 4);
    dr.rows7 = try dr.alloc(SD * H * 2);
    dr.base = try dr.alloc(SD * dr.vn * 2);
    dr.mx = try dr.alloc(SD * 2 * H * 2);
    dr.inv = try dr.alloc(SD * 3 * mrank * 4);
    dr.state = try dr.alloc(most_streams * mrank * 2);
    dr.latent = try dr.alloc(most_streams * mrank * 2);
    dr.best_v = try dr.alloc(most_streams * logit_blocks * 4);
    dr.best_i = try dr.alloc(most_streams * logit_blocks * 4);
    dr.best_s = try dr.alloc(most_streams * logit_blocks * 4);
    dr.pairs = try dr.alloc(W * most_streams * pick_stride);
    dr.walk = try dr.alloc(most_streams * block * 4);
    dr.chances = try dr.alloc(most_streams * block * 4);
    dr.chances_on = draftCut() != .off;
    if (o.rank == 0 and !dr.chances_on) std.log.info("inkling2 dspark: no chances (TF_INK2_DRAFT_CUT=0): v0.2's argmax-only picks", .{});
    // the lane matmul's slice partials: the largest any call takes (one stream's block up to every stream's)
    {
        var most: usize = 0;
        for (&dr.L) |*l| {
            for (1..most_streams + 1) |n| for ([_]cuda.qmmf.Weight{ l.qkv, l.o, l.gu, l.down }) |w| {
                most = @max(most, o.lane.partBytes(n * block, w));
            };
            for (1..ncap + 1) |rn| most = @max(most, o.lane.partBytes(rn, l.kv));
        }
        for (1..most_streams + 1) |n| most = @max(most, o.lane.partBytes(n * drafts, dr.rnn_inv));
        dr.lane_part_bytes = most;
        dr.lane_part = try dr.alloc(most);
    }
    try s.synchronize();
    std.log.info("inkling2 dspark rank {d} of {d}: {d} q heads, {d} kv heads, MLP {d}, vocabulary {d}..{d}: {d:.2} GiB; a slot's caches {d:.3} GiB at {d} positions (the local layers' rings: {d} positions, {d} rows mirrored; {d} bytes a position)", .{ o.rank, o.world, hq, hk, I, dr.vlo, dr.vlo + dr.vn, @as(f64, @floatFromInt(dr.bytes)) / (1 << 30), @as(f64, @floatFromInt(dr.slotBytes())) / (1 << 30), o.context, ringRows(o.context, o.cap), localRows(o.context, o.cap), 2 * hk * D * 2 * @as(usize, if (ringRows(o.context, o.cap) < o.context) 1 else nlayers) });
    return dr;
}

/// Reads checkpoint tensors (bf16) and packs the rank's cuts for the lane matmul.
const Pack = struct {
    dr: *Drafter,
    r: *ink.st.Reader,
    packer: *cuda.modelopt.Packer,
    s: cuda.Stream,
    /// pack() quantizes to MXFP8 (TF_INK2_DSPARK_FP8)
    mx: bool = false,

    fn entry(P: *Pack, name: []const u8) !ink.st.Entry {
        const e = P.r.find(name) orelse {
            std.log.err("inkling2 dspark: mtp.safetensors has no {s}", .{name});
            return error.BadDraftCheckpoint;
        };
        if (e.dtype != .bf16) return error.BadDraftCheckpoint;
        return e;
    }

    /// A bf16 tensor as stored (its byte count checked).
    fn raw(P: *Pack, name: []const u8, n: usize) !u64 {
        const e = try P.entry(name);
        if (e.bytes() != n) return error.BadDraftCheckpoint;
        const host = try P.dr.gpa.alloc(u8, n);
        defer P.dr.gpa.free(host);
        try P.r.readInto(e, 0, host);
        return P.dr.put(host);
    }

    /// A bf16 gain vector as fp32.
    fn gain(P: *Pack, name: []const u8) !u64 {
        const e = try P.entry(name);
        const n: usize = @intCast(e.count());
        const host = try P.dr.gpa.alloc(u8, n * 2);
        defer P.dr.gpa.free(host);
        try P.r.readInto(e, 0, host);
        const out = try P.dr.gpa.alloc(f32, n);
        defer P.dr.gpa.free(out);
        for (out, 0..) |*o, i| o.* = ink.st.bf16ToF32(std.mem.readInt(u16, host[2 * i ..][0..2], .little));
        return P.dr.put(std.mem.sliceAsBytes(out));
    }

    /// Row cuts (name, first row, rows) of [*, k] tensors stacked and packed: a lane weight [sum rows, k].
    fn rows(P: *Pack, cuts: []const struct { []const u8, usize, usize }, k: usize) !cuda.qmmf.Weight {
        var n: usize = 0;
        for (cuts) |c| n += c[2];
        const host = try P.dr.gpa.alloc(u8, n * k * 2);
        defer P.dr.gpa.free(host);
        var at: usize = 0;
        for (cuts) |c| {
            const e = try P.entry(c[0]);
            if (e.rank != 2 or e.dim(1) != k or c[1] + c[2] > e.dim(0)) return error.BadDraftCheckpoint;
            try P.r.readInto(e, c[1] * k * 2, host[at..][0 .. c[2] * k * 2]);
            at += c[2] * k * 2;
        }
        return P.pack(host, n, k);
    }

    /// Columns c0 .. c0 + kc of a [n, k] tensor, packed as a lane weight [n, kc].
    fn cols(P: *Pack, name: []const u8, n: usize, k: usize, c0: usize, kc: usize) !cuda.qmmf.Weight {
        const e = try P.entry(name);
        if (e.rank != 2 or e.dim(0) != n or e.dim(1) != k) return error.BadDraftCheckpoint;
        const all = try P.dr.gpa.alloc(u8, n * k * 2);
        defer P.dr.gpa.free(all);
        try P.r.readInto(e, 0, all);
        const host = try P.dr.gpa.alloc(u8, n * kc * 2);
        defer P.dr.gpa.free(host);
        for (0..n) |i| @memcpy(host[i * kc * 2 ..][0 .. kc * 2], all[(i * k + c0) * 2 ..][0 .. kc * 2]);
        return P.pack(host, n, kc);
    }

    fn colsPacked(P: *Pack, name: []const u8, n: usize, k: usize, c0: usize, kc: usize) !cuda.qmmf.Weight {
        return P.cols(name, n, k, c0, kc);
    }

    fn pack(P: *Pack, host: []const u8, n: usize, k: usize) !cuda.qmmf.Weight {
        if (k % cuda.qmmf.group != 0) return error.BadDraftCheckpoint;
        if (P.mx) {
            const mx = cuda.mxfp8;
            const gpa = P.dr.gpa;
            const codes = try gpa.alloc(u8, n * k);
            defer gpa.free(codes);
            const sc = try gpa.alloc(u8, n * (k / mx.group));
            defer gpa.free(sc);
            mx.quantize(codes, sc, host, n, k) catch return error.BadDraftCheckpoint;
            const pc = try gpa.alloc(u8, mx.padded(n) * k);
            defer gpa.free(pc);
            mx.packCodes(pc, codes, n, k);
            const psc = try gpa.alloc(u8, mx.padded(n) * (k / mx.group));
            defer gpa.free(psc);
            mx.packScales(psc, sc, n, k);
            return mx.weight(try P.dr.put(pc), try P.dr.put(psc), n, k);
        }
        var src = try cuda.DeviceBuffer.fromHost(P.dr.d, host);
        defer src.free();
        const out = try P.dr.alloc(cuda.modelopt.padded(n) * k * 2);
        try P.packer.bf16Lane(P.s, src.ptr, out, n, k);
        try P.s.synchronize();
        return cuda.modelopt.bf16Weight(out, n, k);
    }
};

test "inkling2 dspark: taps by target layer, local layers" {
    try std.testing.expectEqual(@as(?usize, 0), tapOf(4));
    try std.testing.expectEqual(@as(?usize, 6), tapOf(40));
    try std.testing.expectEqual(@as(?usize, null), tapOf(5));
    try std.testing.expect(!local[2] and local[0] and local[4]);
    // a slot of fewer positions than a ring: every layer holds them all
    try std.testing.expectEqual(@as(usize, 5 * 2 * 100 * 2 * 128 * 2), Caches.bytes(2, 100, 2048, false));
    // W = 4 (2 key heads) at the release's 131,072 + 16: the global layer at every position, four mirrored rings of
    // 3,072 positions in 4,096 rows
    const ctx: usize = 131_088;
    try std.testing.expectEqual(@as(usize, 3072), ringRows(ctx, 2048));
    try std.testing.expectEqual(@as(usize, 4096), localRows(ctx, 2048));
    try std.testing.expectEqual(@as(usize, 2 * (ctx + 4 * 4096) * 2 * 128 * 2), Caches.bytes(2, ctx, 2048, false));
    // on the KV page pool the global layer is in the pool: the slot holds the four rings alone
    try std.testing.expectEqual(@as(usize, 2 * (4 * 4096) * 2 * 128 * 2), Caches.bytes(2, ctx, 2048, true));
}

test "inkling2 dspark: on pages the global layer is the pool's, the rings packed without it" {
    const row = 2 * 128 * 2;
    const pg: Caches.Paged = .{ .k = 7 << 30, .v = 9 << 30, .table = 3 << 20, .page = 256 * 2 * 128 };
    const cs: Caches = .{ .buf = .{ .d = undefined, .ptr = 1 << 20, .len = Caches.bytes(2, 10_000, 2048, true), .borrowed = true }, .context = 10_000, .ring = ringRows(10_000, 2048), .row = row, .pg = pg };
    try std.testing.expectEqual(@as(usize, 0), cs.rows(2));
    try std.testing.expectEqual(pg.k, cs.keys(2));
    try std.testing.expectEqual(pg.v, cs.values(2));
    try std.testing.expectEqual(cs.keys(0) + 2 * 2 * 4096 * row, cs.keys(3));
    try std.testing.expectEqual(cs.buf.ptr + cs.buf.len, cs.values(4) + 4096 * row);
    const kv = cs.pagedKv(40).?;
    try std.testing.expect(kv.paged() and kv.slots == 40 * split_keys and kv.heads == 2 and kv.page == pg.page);
}

test "inkling2 dspark: the caches' layout and the mirrored ring" {
    const row = 2 * 128 * 2;
    const cs: Caches = .{ .buf = .{ .d = undefined, .ptr = 1 << 20, .len = Caches.bytes(2, 10_000, 2048, false), .borrowed = true }, .context = 10_000, .ring = ringRows(10_000, 2048), .row = row };
    // layers 0, 1 rings, 2 global, 3, 4 rings; keys then values, packed
    try std.testing.expectEqual(@as(u64, 1 << 20), cs.keys(0));
    try std.testing.expectEqual(cs.keys(0) + 4096 * row, cs.values(0));
    try std.testing.expectEqual(cs.keys(0) + 2 * 2 * 4096 * row, cs.keys(2));
    try std.testing.expectEqual(cs.keys(2) + 10_000 * row, cs.values(2));
    try std.testing.expectEqual(cs.keys(2) + 2 * 10_000 * row, cs.keys(3));
    try std.testing.expectEqual(cs.buf.ptr + cs.buf.len, cs.values(4) + 4096 * row);
    try std.testing.expectEqual(@as(usize, 3072), cs.wrap(0));
    try std.testing.expectEqual(@as(usize, 0), cs.wrap(2));
    // a draft's context (positions p - window .. p - 1) stays in the ring while the writes after it reach no further
    // than one forward of `cap` rows from the anchor (in serving: a verify window, 8 rows), and sits in consecutive
    // rows from (p - window) % ring, inside the ring and its mirror
    var p: usize = window;
    while (p < 9_000) : (p += 997) {
        const newest = p + 2048 - 1;
        for (p - window..p) |x| try std.testing.expect(newest - x < cs.ring);
        try std.testing.expect((p - window) % cs.ring + window <= cs.rows(0));
    }
}
