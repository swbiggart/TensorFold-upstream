//! G-attn: sliding-window (1024) and global attention with a relative-position bias, a sink logit, L2-clipped
//! Q / K / relative states, RMS-normed V without gain, log-scaled Q in global layers and a sigmoid output gate.
//! Kernel image: inkling2_attn.cu. Math: Transformers modeling_inkling2.py (Apache-2.0), Inkling2SlidingAttention /
//! Inkling2FullAttention, eager_attention_forward, Inkling2L2Clip, Inkling2ScaledL2Clip, Inkling2RelativeLogits.
//!
//! Shapes on a rank: hq query heads, hk key heads (W = 1: 48 and 8; group hq / hk = 6), D = 128, d_rel = 16,
//! H = 4608, R rows at positions pos0 .. pos0 + R - 1. Query head h reads key head h / 6.
//!
//! `project` (attn.project, L3): from the q source xq (bf16 [R, H], the normed attention input): q = xq Wq^T
//!   [R, hq*D], rel = xq Wr^T [R, hq*16], gate = xq Wg^T [R, hq*D], sink = xq Wsink^T [R, hq]; k = xk Wk^T
//!   [R, hk*D]; v = xv Wv^T [R, hk*D]; global layers also dk = xk Wkproj^T [R, hk*4], dv = xv Wvproj^T [R, hk*4]
//!   (the dynamic conv taps, row head*4 + lag). Sliding layers pass xq == xk == xv and leave dk, dv empty. All
//!   bf16, fp32 sums, one rounding. Projections run on upstream's lane matmul (cuda/qmmf.zig, BF16 mode: K16 MMA
//!   steps in a fixed order and K slices chosen from the weight's shape, so a row's bits do not depend on the row
//!   count), its weights packed once per layer by `prepare` (cuda/modelopt.zig Packer.bf16Lane). The lane matmul is
//!   the only projection path (cuBLASLt picks its algorithm by row count and is not row-invariant). Measured: the
//!   lane's outputs are 99.9% bit-equal to bf16(fp64 x W^T) and to torch with fp32 reductions; the oracle's own
//!   (torch's default allow_bf16_reduced_precision_reduction) differs from both in about 30% of values (rel 2.3e-3
//!   against fp64, the lane's 1.66e-3 being bf16 rounding alone).
//!
//! `heads` (attn.heads, L5-L9) -> q', k', v', rel' (bf16, same shapes as their inputs):
//!   q' = clip(q * q_scale[h], 22.627416997969522)  per head (fp32: x * min(1, thr / (||x||_2 + 1e-6))), bf16
//!   k' = clip(kc, 22.627416997969522)               (kc = G-conv's K output; no scale)
//!   v' = bf16(vc * rsqrt(mean_D(vc^2) + 1e-6))      (fp32, per head, no gain)
//!   rel' = clip(rel * rel_scale[h], 8.0) over d_rel = 16, bf16
//!   global layers only: tau(p) = 1 + 0.1 * ln(max(1, (p + 1) / 128000)) (fp32);
//!     q' = bf16(f32(q') * tau), rel' = bf16(f32(rel') * tau) (after their clips' bf16 rounding; K not scaled).
//!   The scaffold then writes k', v' into the stream's cache (cuda_state.zig `kvWrite`) before `core`.
//!
//! `core` (attn.core, L10 + L12-L14) -> o bf16 [R, hq*D]:
//!   bias[t, h, k] = bf16(rel'[t, h, :] . rel_proj[:, d]) at d = pos_t - pos_k when 0 <= d < 1024, else 0
//!   s[t, h, k] = q'[t,h] . k'[k, h/6] / sqrt(128) + bias, over keys k with 0 <= pos_t - pos_k (global) and also
//!     pos_t - pos_k < 1024 (sliding window, self included)
//!   p = softmax over those keys plus one sink logit sink[t, h] (zero value); o = sum_k p_k v'[k]
//!   (equivalently o = o_nosink * sigmoid(lse - sink)); then o = bf16(f32(o) * sigmoid(f32(gate))).
//!   Transformers' eager form rounds to bf16 at q.k, * scale, + bias, and the probabilities before p.v; a flash
//!   form in fp32 differs from it by bf16 steps only (G1 tolerance). Keys come from `kv` (`Kv` below).
//!
//! `output` (attn.out, L15; the lane matmul as `project`): out bf16 [R, H] = o Wo^T (Wo the rank's column cut).
//!   At W > 1 (Phase 2) this becomes an fp32 partial summed over ranks in rank order.
//!
//! Kernels (inkling2_attn.cu, `glue` flags: no FMA contraction, IEEE division and sqrt). `heads` is one launch, a
//! warp a (row, head), torch's op order (the clip's `thr / t` is t.reciprocal() * thr; tau's `/ 128000` a multiply
//! by the reciprocal). `core` keeps Transformers' eager roundings (q.k, * scale, + bias, the probabilities, p.v)
//! with fp32 products and sums in a fixed order: keys in 32-position tiles and 256-position splits aligned to
//! absolute positions, the softmax sums and the p.v sums folded tiles-within-a-split, then splits in order. It runs
//! as one launch over all of a row block's splits (`fused`: prompt chunks) or a block a split plus an ordered merge
//! (`split`: decode windows over long contexts). Both forms give the same bits, and a row's bits never depend on
//! the forward's other rows (its window, chunk or batch). TF_INK2_ATTN=fused|split forces a form (tests). The split
//! form of 1 .. 8 rows (decode steps, DSpark verify windows, a batched step's segments) runs on the row passes
//! (tf_ink2_attn_rstats / _rpv): a block a (row, key head, split) instead of one 8-row block, the split's scores
//! kept from the stats pass for p.v, the value rows staged in shared memory; the same expressions in the same order,
//! so the same bits (tf-inkling2-attn --op-check). On GB10 at the W = 4 rank shape a sliding layer's core takes
//! 28-39 us at 1-8 rows against 35 (one row) and 143-166 us (2-8 rows) before. TF_INK2_ATTN_ROWS=0 restores the
//! previous kernels.
//! Long contexts: a forward over more than `fold_splits` splits folds each row's softmax pairs once
//! (tf_ink2_attn_fold) between the passes, where every split's block of the second pass folded them again (quadratic
//! in the depth); a captured step's global layers launch at most `grid_splits` splits, each block walking its splits
//! up to the device position (so a step costs what its depth needs, not what the slot holds). Both keep every
//! expression and its order. `scratchNeed` is the split form's largest scratch at a slot's context; the engine checks
//! it at start-up (`checkScratch`).
//! The arithmetic is the scalar fmaf chains only (no tensor-core core, no bf16 partials: they changed bits).
//!
//! KV storage: `Kv` below, written by the scaffold (cuda_state.zig `kvWrite`) and read in place by `core`: sliding
//! layers a ring of at least 1023 + R slots, global layers a slot a position, for decode and prompt alike.
//! Paged global layers (the shared KV page pool, v0.2): a global layer's rows may instead live in pool pages of
//! `split_keys` positions, found through the slot's page table (`Kv.pages`). Every kernel that addresses keys or
//! values (core's fused, split, one-row and row passes, the cache write) is built twice from
//! one source (inkling2_attn.cu `Ring` / `Paged`), and the host takes the `_pg` form for a paged Kv: a split is a page,
//! so a block reads its page id once a split and a key's address has no table read and no branch. Sliding layers keep
//! their rings and their kernels. The bits are the same (`tf-inkling2-attn --op-check --paged`).
//!
//! Oracle tensors: op.<L>.attn.<project|heads|core|out>.out.* (inkling2_op_ref.py capture), whose chain ends in
//! the layer's h.<L>. Test alone: run the runner with --feed <capture> --dump-ops and inkling2_compare.py --ops
//! (each op gets torch's inputs), or inkling2_op_ref.py check on a --dump-ops dump; tools/zig/inkling2_attn_check.py
//! adds long-prompt captures and the bit comparison of runs at different chunk sizes.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");

const Ctx = ops.Ctx;
const Mat = ops.Mat;
const Layer = weights.Layer;

pub const qk_clip: f32 = 22.627416997969522;
pub const rel_clip: f32 = 8.0;
pub const window = 1024;
pub const rel_extent = 1024;

/// Key positions a tile and a split of `core` (inkling2_attn.cu KT, SPLIT).
pub const tile_keys = 32;
pub const split_keys = 256;
/// Rows a block of `core`, and query heads a key head (inkling2_attn.cu RB, G).
pub const block_rows = 8;
pub const group = 6;
/// The widest forward the row passes (tf_ink2_attn_rstats / _rpv) take by default: a DSpark window, a segment.
pub const max_rows = 8;
/// A captured step's global layers on the row passes launch at most this many splits (32,768 positions); a block takes
/// split z, z + grid_splits, ... up to its row's last, so a deep row's splits run in rounds chosen on the device.
pub const grid_splits = 128;
/// Forwards over more than this many splits (16,384 positions) fold the rows' softmax pairs once in tf_ink2_attn_fold.
pub const fold_splits = 64;
/// The split form's fp32 partials stay under this (`formOf`); above it the forward takes the fused form.
pub const partial_cap: usize = 512 << 20;

pub const Form = enum { auto, fused, split };

fn env(name: [*:0]const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else "";
}

/// inkling2_attn.cu's functions and the lane matmul.
pub const Kernels = struct {
    mod: cuda.Module,
    probe: cuda.Function,
    heads: cuda.Function,
    core: cuda.Function,
    stats: cuda.Function,
    pv: cuda.Function,
    merge: cuda.Function,
    /// the split form's passes for one row (decode): the same bits as `stats`/`pv`, laid out for one row
    stats1: cuda.Function,
    pv1: cuda.Function,
    /// the split form's passes for forwards of 1 .. `rows` rows (decode steps, verify windows, a batched step's
    /// segments): a block a (row, key head, split), the q.k of a split taken once (the same bits as stats1 / pv1)
    rstats: cuda.Function,
    rpv: cuda.Function,
    /// each row's softmax pairs folded once between the passes (forwards over more than `fold_min` splits)
    fold: cuda.Function,
    /// a captured global layer's grid on the row passes, in splits (`grid_splits`; the op check narrows it)
    grid: usize = grid_splits,
    /// the split count above which the fold kernel runs (`fold_splits`; the op check moves it)
    fold_min: usize = fold_splits,
    /// the widest forward the row passes take (TF_INK2_ATTN_ROWS, default 8; 0: never, the previous kernels:
    /// stats1 / pv1 for one row, the general passes for more, a captured step over the slot's every split)
    rows: usize,
    /// TF_INK2_ATTN_AB=1 (measurement only, eager forwards): forwards at an odd first position take the previous
    /// kernels, at an even one the row passes; every rank sees the same positions, so one process times both
    ab: bool,
    kv_write: cuda.Function,
    /// the same kernels' paged forms (`Kv.pages` set: a global layer in the KV page pool)
    pg: Paged = undefined,
    lane: cuda.qmmf.Lane,
    form: Form,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_attn);
        errdefer mod.unload();
        var lane = try cuda.qmmf.Lane.load(d, (try d.currentCapability()) / 10);
        errdefer lane.unload();
        try lane.loadPrompt(d);
        return .{
            .mod = mod,
            .probe = try mod.function("tf_ink2_attn_probe"),
            .heads = try mod.function("tf_ink2_attn_heads"),
            .core = try mod.function("tf_ink2_attn_core"),
            .stats = try mod.function("tf_ink2_attn_stats"),
            .pv = try mod.function("tf_ink2_attn_pv"),
            .merge = try mod.function("tf_ink2_attn_merge"),
            .stats1 = try mod.function("tf_ink2_attn_stats1"),
            .pv1 = try mod.function("tf_ink2_attn_pv1"),
            .rstats = try mod.function("tf_ink2_attn_rstats"),
            .rpv = try mod.function("tf_ink2_attn_rpv"),
            .fold = try mod.function("tf_ink2_attn_fold"),
            .rows = std.fmt.parseInt(usize, env("TF_INK2_ATTN_ROWS"), 10) catch max_rows,
            .ab = std.mem.eql(u8, env("TF_INK2_ATTN_AB"), "1"),
            .kv_write = try mod.function("tf_ink2_attn_kv_write"),
            .pg = .{
                .core = try mod.function("tf_ink2_attn_core_pg"),
                .stats = try mod.function("tf_ink2_attn_stats_pg"),
                .pv = try mod.function("tf_ink2_attn_pv_pg"),
                .stats1 = try mod.function("tf_ink2_attn_stats1_pg"),
                .pv1 = try mod.function("tf_ink2_attn_pv1_pg"),
                .rstats = try mod.function("tf_ink2_attn_rstats_pg"),
                .rpv = try mod.function("tf_ink2_attn_rpv_pg"),
                .kv_write = try mod.function("tf_ink2_attn_kv_write_pg"),
            },
            .lane = lane,
            .form = std.meta.stringToEnum(Form, env("TF_INK2_ATTN")) orelse .auto,
        };
    }

    pub fn unload(k: *Kernels) void {
        k.lane.unload();
        k.mod.unload();
    }

    /// The paged forms (inkling2_attn.cu `_pg`): each takes the ring form's arguments, then the slot's page table and
    /// the pool's page stride in elements.
    pub const Paged = struct {
        core: cuda.Function,
        stats: cuda.Function,
        pv: cuda.Function,
        stats1: cuda.Function,
        pv1: cuda.Function,
        rstats: cuda.Function,
        rpv: cuda.Function,
        kv_write: cuda.Function,
    };
};

/// The layer's projections packed for the lane matmul (`prepare`), in one allocation at `l.extra[0]`.
pub const Packed = enum { q, k, v, rel, gate, sink, o, k_wproj, v_wproj };

fn shapeOf(l: *const Layer, p: Packed) struct { src: u64, n: usize, k: usize } {
    const D = weights.head_dim;
    const H = weights.hidden;
    const hq: usize = l.hq;
    const hk: usize = l.hk;
    return switch (p) {
        .q => .{ .src = l.wq, .n = hq * D, .k = H },
        .k => .{ .src = l.wk, .n = hk * D, .k = H },
        .v => .{ .src = l.wv, .n = hk * D, .k = H },
        .rel => .{ .src = l.wr, .n = hq * weights.d_rel, .k = H },
        .gate => .{ .src = l.wg, .n = hq * D, .k = H },
        .sink => .{ .src = l.wsink, .n = hq, .k = H },
        .o => .{ .src = l.wo, .n = H, .k = hq * D },
        .k_wproj => .{ .src = l.k_wproj, .n = hk * weights.taps, .k = H },
        .v_wproj => .{ .src = l.v_wproj, .n = hk * weights.taps, .k = H },
    };
}

fn packedBytes(l: *const Layer, p: Packed) usize {
    const s = shapeOf(l, p);
    return if (s.src == 0) 0 else cuda.modelopt.padded(s.n) * s.k * 2;
}

/// The lane matmul's view of projection `p` of a prepared layer.
pub fn lanes(l: *const Layer, p: Packed) cuda.qmmf.Weight {
    var at = l.extra[0];
    for (std.enums.values(Packed)) |x| {
        if (x == p) break;
        at += packedBytes(l, x);
    }
    const s = shapeOf(l, p);
    return cuda.modelopt.bf16Weight(at, s.n, s.k);
}

/// y [R, n] bf16 = x [R, k] bf16 times projection `p` (see the header).
fn linear(c: *const Ctx, l: *const Layer, p: Packed, x: Mat, out: Mat) !void {
    const s = shapeOf(l, p);
    if (x.dt != .bf16 or (out.dt != .bf16 and out.dt != .f32) or !out.dense() or x.cols != s.k or out.rows != x.rows or out.cols != s.n) return error.BadAttnShape;
    if (x.rows == 0) return;
    if (l.extra[0] == 0) return error.AttnNotPrepared;
    const w = lanes(l, p);
    const lane = &c.k.attn.lane;
    const part_bytes = lane.partBytes(x.rows, w);
    const part = if (part_bytes > 0) try c.tmp(part_bytes) else 0;
    try lane.matmulOut(c.stream, x.ptr, x.ld, x.rows, w, out.ptr, if (out.dt == .f32) .f32 else .bf16, part);
}

/// One stream's keys and values of a layer: k', v' bf16 [slots, hk*D]; position p lives at slot p % slots.
/// Global layers: slots >= the context (every position kept); sliding layers: a ring with slots >= 1023 + the
/// widest forward's rows, so the window of every row of a forward is still there after the forward's writes.
/// `window` is 1024 for sliding layers, 0 for global (all positions 0 .. pos).
/// Paged (`pages` not 0; global layers only): position p lives at row p % split_keys of pool page pages[p / split_keys]
/// (i32 page ids on the device, one entry a split), pages `page` elements apart; k, v are this layer's rows in page 0
/// (the pool's base plus the layer's runs), and `slots` is the positions the table covers.
pub const Kv = struct {
    k: u64,
    v: u64,
    slots: usize,
    heads: usize,
    window: usize,
    pages: u64 = 0,
    page: usize = 0,
    /// A ring view of a stream's run of pool pages (State.kvFor: its pages one run of consecutive ids): the run's
    /// first page. k, v are the run's first rows and the ring kernels read it as a v0.1 window; recorded while a
    /// forward is captured (Ctx.launchOn) so a replay can follow the run.
    run: ?u32 = null,

    pub fn paged(kv: Kv) bool {
        return kv.pages != 0;
    }

    /// The page table and stride, after a paged form's ring arguments.
    pub fn addPaged(kv: Kv, a: *cuda.Args) void {
        a.add(kv.pages);
        a.add(@as(i64, @intCast(kv.page)));
    }

    /// The first position row `pos` attends to.
    pub fn lo(kv: Kv, pos: usize) usize {
        return if (kv.window == 0) 0 else pos -| (kv.window - 1);
    }
};

pub const Proj = struct { q: Mat, k: Mat, v: Mat, rel: Mat, gate: Mat, sink: Mat, dk: Mat = .{}, dv: Mat = .{} };
pub const Heads = struct { q: Mat, k: Mat, v: Mat, rel: Mat };

/// L3: the layer's projections (see the header).
pub fn project(c: *const Ctx, l: *const Layer, xq: Mat, xk: Mat, xv: Mat, out: Proj) !void {
    try linear(c, l, .q, xq, out.q);
    try linear(c, l, .rel, xq, out.rel);
    try linear(c, l, .gate, xq, out.gate);
    try linear(c, l, .sink, xq, out.sink);
    try linear(c, l, .k, xk, out.k);
    try linear(c, l, .v, xv, out.v);
    if (l.global) {
        try linear(c, l, .k_wproj, xk, out.dk);
        try linear(c, l, .v_wproj, xv, out.dv);
    }
}

/// A captured decode step's positions (the kernels read the first position from the device, inkling2_attn.cu):
/// `posp` an i32 on the device holding pos0 at replay, `positions` the slot's capacity (the split scratch covers it;
/// sliding layers on the row passes launch only the `windowSplits` a window can touch, the first derived on the device,
/// and global layers at most `grid_splits`, each block walking its splits up to the device position).
pub const Dev = struct {
    posp: u64,
    positions: usize,
};

/// L5-L9: clips, value norm, log scaling (see the header). `dev`: pos0 is read on the device at run time.
pub fn heads(c: *const Ctx, l: *const Layer, pos0: usize, dev: ?Dev, q: Mat, kc: Mat, vc: Mat, rel: Mat, out: Heads) !void {
    const R = q.rows;
    const D = weights.head_dim;
    const hq: usize = l.hq;
    const hk: usize = l.hk;
    const dr = weights.d_rel;
    if (!q.is(R, hq * D, .bf16) or !kc.is(R, hk * D, .bf16) or !vc.is(R, hk * D, .bf16) or !rel.is(R, hq * dr, .bf16) or
        !out.q.is(R, hq * D, .bf16) or !out.k.is(R, hk * D, .bf16) or !out.v.is(R, hk * D, .bf16) or !out.rel.is(R, hq * dr, .bf16))
        return error.BadAttnShape;
    for ([_]Mat{ q, kc, vc, rel, out.q, out.k, out.v, out.rel }) |m| if (!m.dense()) return error.BadAttnShape;
    if (R == 0) return;
    if (pos0 + R > std.math.maxInt(i32)) return error.AttnPositionTooLarge;
    const items = 2 * hq + 2 * hk;
    var a: cuda.Args = .{};
    for ([_]u64{ q.ptr, kc.ptr, vc.ptr, rel.ptr, l.q_scale, l.rel_scale, out.q.ptr, out.k.ptr, out.v.ptr, out.rel.ptr }) |p| a.add(p);
    a.add(@as(i32, @intCast(hq)));
    a.add(@as(i32, @intCast(hk)));
    a.add(@as(i32, @intCast(pos0)));
    a.add(@as(i32, @intFromBool(l.global)));
    a.add(if (dev) |x| x.posp else @as(u64, 0));
    try cuda.launch.launch(c.k.attn.heads, grid(@intCast(R), (items + 7) / 8, 1, 256), c.stream, &a);
}

fn grid(x: u32, y: usize, z: usize, threads: u32) cuda.launch.Config {
    return .{ .grid = .{ .x = x, .y = @intCast(y), .z = @intCast(z) }, .block = .{ .x = threads, .y = 1, .z = 1 } };
}

pub const Splits = struct { first: usize, count: usize };

/// The splits `core` visits for a forward of R rows at pos0: the first and how many.
pub fn splits(kv: Kv, pos0: usize, R: usize) Splits {
    const first = kv.lo(pos0) / split_keys;
    return .{ .first = first, .count = (pos0 + R - 1) / split_keys - first + 1 };
}

/// The most splits a sliding window of R rows touches: positions lo(pos0) .. pos0 + R - 1, at most 1022 + R apart.
/// A captured step on the row passes launches only these (the first derived on the device from the position).
pub fn windowSplits(R: usize) usize {
    return (window - 2 + R + split_keys - 1) / split_keys + 1;
}

/// The row passes take this forward: up to `rows` rows, the query heads grouped as the kernels assume, and (TF_INK2_ATTN_AB=1) an even first position.
pub fn rowPasses(k: *const Kernels, R: usize, pos0: usize, hq: usize, hk: usize) bool {
    if (k.ab and pos0 % 2 == 1) return false;
    return R <= k.rows and hq == group * hk;
}

/// The form `core` takes: `split` for decode-sized forwards over more than one split while its fp32 partials stay
/// small, else `fused`. Either gives the same bits.
pub fn formOf(forced: Form, R: usize, n_splits: usize, hq: usize) Form {
    if (forced != .auto) return forced;
    const partial_bytes = n_splits * R * hq * weights.head_dim * 4;
    return if (R <= 64 and n_splits > 1 and partial_bytes <= partial_cap) .split else .fused;
}

/// How `core` runs a forward: its form, and in the split form the passes, the first split (-1: derived on the device
/// from the position), the launch's splits, the splits the scratch holds (indexed from the first) and whether the
/// rows' pairs are folded once between the passes.
pub const Plan = struct {
    form: Form,
    rows: bool = false,
    first: i32 = 0,
    grid: usize = 0,
    span: usize = 0,
    fold: bool = false,

    /// Op scratch the split form takes (the takes in `core`'s order, each 256-byte aligned as Scratch.take aligns).
    pub fn scratch(p: Plan, R: usize, hq: usize) usize {
        if (p.form != .split) return 0;
        const n = p.span * R * hq;
        var at: usize = 0;
        for ([_]usize{ n * 8, n * weights.head_dim * 4, if (p.rows) n * split_keys * 2 else 0, if (p.fold) R * hq * 8 else 0 }) |b| {
            if (b > 0) at = std.mem.alignForward(usize, at, 256) + b;
        }
        return at;
    }
};

/// The plan of a forward of R rows at pos0 (`dev`: a captured step, pos0 read on the device).
pub fn plan(k: *const Kernels, R: usize, pos0: usize, dev: ?Dev, kv: Kv, hq: usize, hk: usize) Plan {
    const rk = rowPasses(k, R, pos0, hq, hk);
    if (dev) |x| {
        const all = (x.positions + split_keys - 1) / split_keys;
        // sliding, row passes: the splits a window can touch from the derived first; global, row passes: every split
        // of the slot in the scratch, at most k.grid launched; the previous kernels: every split of the slot
        if (rk and kv.window != 0) {
            const n = @min(windowSplits(R), all);
            return .{ .form = .split, .rows = true, .first = -1, .grid = n, .span = n, .fold = n > k.fold_min };
        }
        return .{ .form = .split, .rows = rk, .first = 0, .grid = if (rk) @min(all, @max(k.grid, 1)) else all, .span = all, .fold = all > k.fold_min };
    }
    const sp = splits(kv, pos0, R);
    if (formOf(k.form, R, sp.count, hq) != .split) return .{ .form = .fused };
    return .{ .form = .split, .rows = rk, .first = @intCast(sp.first), .grid = sp.count, .span = sp.count, .fold = sp.count > k.fold_min };
}

/// The most op scratch `core` takes in any forward of a slot of `context` positions: up to `rows` rows (eager, any
/// position) and up to `graph_rows` as a captured step. The split form's pairs, partials, kept scores and folded pairs
/// grow with the splits a forward spans, so this is the deepest forward's (or, where the partials' cap moves a deep
/// forward to the fused form, the deepest one still split).
pub fn scratchNeed(k: *const Kernels, hq: usize, hk: usize, context: usize, rows: usize, graph_rows: usize) usize {
    var most: usize = 0;
    for ([_]usize{ 0, window }) |win| {
        const kv: Kv = .{ .k = 0, .v = 0, .slots = if (win == 0) context else window - 1 + rows, .heads = hk, .window = win };
        for (1..@min(rows, context) + 1) |R| {
            const deep = splits(kv, context - R, R).count;
            const c_max = if (win == 0) deep else @min(windowSplits(R), (context - 1) / split_keys + 1);
            const c: usize = switch (k.form) {
                .fused => 0,
                .split => c_max,
                .auto => blk: {
                    if (R > 64) break :blk 0;
                    const n = @min(c_max, partial_cap / (R * hq * weights.head_dim * 4));
                    break :blk if (n < 2) 0 else n;
                },
            };
            const rk = R <= k.rows and hq == group * hk;
            if (c > 0) most = @max(most, (Plan{ .form = .split, .rows = rk, .grid = c, .span = c, .fold = c > k.fold_min }).scratch(R, hq));
            if (R <= graph_rows) most = @max(most, plan(k, R, 0, .{ .posp = 0, .positions = context }, kv, hq, hk).scratch(R, hq));
        }
    }
    return most;
}

/// Start-up, on every rank (the ranks agree on the context and the scratch): the op scratch must hold `scratchNeed`
/// at the slot's context, so a short scratch stops every rank here and not one rank's forward after a step's header
/// has gone out (which leaves the other ranks waiting in a collective).
pub fn checkScratch(k: *const Kernels, hq: usize, hk: usize, context: usize, rows: usize, graph_rows: usize, have: usize, rank: u32) !void {
    const need = scratchNeed(k, hq, hk, context, rows, graph_rows);
    const mib = struct {
        fn f(b: usize) f64 {
            return @as(f64, @floatFromInt(b)) / (1 << 20);
        }
    }.f;
    if (need > have) {
        std.log.err("inkling2 rank {d}: the op scratch ({d:.0} MiB) is short of the attention's {d:.1} MiB at {d} positions a slot (raise TF_INK2_SCRATCH_GIB or lower the context)", .{ rank, mib(have), mib(need), context });
        return error.ScratchShort;
    }
    std.log.info("inkling2 rank {d}: the attention's op scratch at {d} positions a slot: {d:.1} MiB of {d:.0} MiB", .{ rank, context, mib(need), mib(have) });
}

/// L10, L12-L14: relative bias, attention with the sink, output gate (see the header). `dev`: pos0 is read on the
/// device at run time and the split form runs as `plan` lays it out: sliding layers on the row passes over
/// `windowSplits(R)` splits from the one the device position gives (s_first -1), global layers on the row passes over
/// at most `grid_splits` (each block walking its splits), the previous kernels over every split of the slot.
pub fn core(c: *const Ctx, l: *const Layer, pos0: usize, dev: ?Dev, q: Mat, rel: Mat, sink: Mat, gate: Mat, kv: Kv, out: Mat) !void {
    const R = q.rows;
    const D = weights.head_dim;
    const hq: usize = l.hq;
    const hk: usize = l.hk;
    if (!q.is(R, hq * D, .bf16) or !rel.is(R, hq * weights.d_rel, .bf16) or !sink.is(R, hq, .bf16) or
        !gate.is(R, hq * D, .bf16) or !out.is(R, hq * D, .bf16))
        return error.BadAttnShape;
    for ([_]Mat{ q, rel, sink, gate, out }) |m| if (!m.dense()) return error.BadAttnShape;
    if (hq != group * hk or kv.heads != hk or kv.window != (if (l.global) @as(usize, 0) else window)) return error.BadAttnShape;
    if (kv.paged() and (kv.window != 0 or kv.page % 8 != 0 or kv.slots % split_keys != 0)) return error.BadAttnShape;
    if (R == 0) return;
    // every key of every row's window must still be in its slot
    if ((pos0 + R - 1) - kv.lo(pos0) >= kv.slots) return error.KvWindow;
    if (kv.slots > std.math.maxInt(i32) or pos0 + R > std.math.maxInt(i32)) return error.AttnPositionTooLarge;
    if (dev) |x| if (pos0 + R > x.positions) return error.KvWindow;
    const k = &c.k.attn;
    const p = plan(k, R, pos0, dev, kv, hq, hk);
    const posp: u64 = if (dev) |x| x.posp else 0;
    const blocks: u32 = @intCast((R + block_rows - 1) / block_rows);
    const geo = [_]i32{ @intCast(R), @intCast(pos0), @intCast(kv.window), @intCast(hq), @intCast(hk), @intCast(kv.slots) };
    // the layer's storage picks the form: ring (or a slot a position), or the pool's pages
    const pg = kv.paged();
    if (p.form != .split) {
        var a: cuda.Args = .{};
        for ([_]u64{ q.ptr, rel.ptr, sink.ptr, gate.ptr, kv.k, kv.v, l.rel_proj, out.ptr }) |x| a.add(x);
        for (geo) |x| a.add(x);
        if (pg) kv.addPaged(&a);
        try c.launchOn(kv.run, kv.k, kv.v, if (pg) k.pg.core else k.core, grid(blocks, hk, 1, 256), &a);
        return;
    }
    // the takes in Plan.scratch's order
    const n = p.span * R * hq;
    const st = try c.tmp(n * 8);
    const op = try c.tmp(n * D * 4);
    const sc = if (p.rows) try c.tmp(n * split_keys * 2) else 0;
    const mz = if (p.fold) try c.tmp(R * hq * 8) else 0;
    if (p.rows) {
        // the row passes: grid (R, hk, splits); the scores of each row's split kept for p.v (bf16)
        var a: cuda.Args = .{};
        for ([_]u64{ q.ptr, rel.ptr, kv.k, l.rel_proj, st, sc }) |x| a.add(x);
        for (geo) |x| a.add(x);
        a.add(p.first);
        a.add(posp);
        if (pg) kv.addPaged(&a);
        try c.launchOn(kv.run, kv.k, kv.v, if (pg) k.pg.rstats else k.rstats, grid(@intCast(R), hk, p.grid, 256), &a);
        if (p.fold) try fold(c, sink, st, mz, geo, p.first, posp);
        var b: cuda.Args = .{};
        for ([_]u64{ sink.ptr, kv.v, st, mz, sc, op }) |x| b.add(x);
        for (geo) |x| b.add(x);
        b.add(p.first);
        b.add(posp);
        if (pg) kv.addPaged(&b);
        try c.launchOn(kv.run, kv.k, kv.v, if (pg) k.pg.rpv else k.rpv, grid(@intCast(R), hk, p.grid, 256), &b);
        try merge(c, op, gate, out, geo, p.first, posp, hq);
        return;
    }
    var a: cuda.Args = .{};
    for ([_]u64{ q.ptr, rel.ptr, kv.k, l.rel_proj, st }) |x| a.add(x);
    for (geo) |x| a.add(x);
    a.add(p.first);
    a.add(posp);
    if (pg) kv.addPaged(&a);
    // one decode row: the one-row passes (the same bits as the general ones)
    const one = R == 1 and hq == group * hk;
    const f_stats = if (pg) (if (one) k.pg.stats1 else k.pg.stats) else if (one) k.stats1 else k.stats;
    try c.launchOn(kv.run, kv.k, kv.v, f_stats, grid(blocks, hk, p.grid, 256), &a);
    if (p.fold) try fold(c, sink, st, mz, geo, p.first, posp);
    var b: cuda.Args = .{};
    for ([_]u64{ q.ptr, rel.ptr, sink.ptr, kv.k, kv.v, l.rel_proj, st, mz, op }) |x| b.add(x);
    for (geo) |x| b.add(x);
    b.add(p.first);
    b.add(posp);
    if (pg) kv.addPaged(&b);
    const f_pv = if (pg) (if (one) k.pg.pv1 else k.pg.pv) else if (one) k.pv1 else k.pv;
    try c.launchOn(kv.run, kv.k, kv.v, f_pv, grid(blocks, hk, p.grid, 256), &b);
    try merge(c, op, gate, out, geo, p.first, posp, hq);
}

/// Between the split form's passes: each row's softmax pair (m, z) a query head, folded once (tf_ink2_attn_fold).
fn fold(c: *const Ctx, sink: Mat, st: u64, mz: u64, geo: [6]i32, first: i32, posp: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ sink.ptr, st, mz }) |x| a.add(x);
    for ([_]i32{ geo[0], geo[1], geo[2], geo[3], first }) |x| a.add(x);
    a.add(posp);
    if (geo[3] > 256) return error.BadAttnShape;
    try cuda.launch.launch(c.k.attn.fold, grid(@intCast(geo[0]), 1, 1, 256), c.stream, &a);
}

/// The split form's ordered sum over a row's splits, then the gate: grid (R, hq), D threads.
fn merge(c: *const Ctx, op: u64, gate: Mat, out: Mat, geo: [6]i32, first: i32, posp: u64, hq: usize) !void {
    var m: cuda.Args = .{};
    for ([_]u64{ op, gate.ptr, out.ptr }) |p| m.add(p);
    for ([_]i32{ geo[0], geo[1], geo[2], geo[3], first }) |x| m.add(x);
    m.add(posp);
    try cuda.launch.launch(c.k.attn.merge, grid(@intCast(geo[0]), hq, 1, weights.head_dim), c.stream, &m);
}

/// L15: the output projection; `out` bf16, or fp32 at W > 1 (the rank's partial over its heads' columns, summed
/// over the ranks in rank order after the exchange).
pub fn output(c: *const Ctx, l: *const Layer, o: Mat, out: Mat) !void {
    try linear(c, l, .o, o, out);
}

/// Runs once per loaded layer (cuda_weights.zig `Prepare`): the projections packed for the lane matmul, one
/// allocation at l.extra[0] (Packed's order). The checkpoint's bf16 copies are then freed (about 2.2 GiB a rank at
/// W = 4); after that the pointers in `l` only tell `lanes` which projections exist.
pub fn prepare(w: *weights.Weights, l: *Layer) anyerror!void {
    if (!cuda.kernels.available) return;
    var total: usize = 0;
    for (std.enums.values(Packed)) |p| total += packedBytes(l, p);
    l.extra[0] = try w.alloc(total);
    var packer = try cuda.modelopt.Packer.load(w.d);
    defer packer.unload();
    var s = try cuda.Stream.init(w.d, true);
    defer s.deinit();
    for (std.enums.values(Packed)) |p| {
        const sh = shapeOf(l, p);
        if (sh.src == 0) continue;
        if (sh.k % cuda.qmmf.group != 0) return error.UnexpectedInkling2Layout;
        try packer.bf16Lane(s, sh.src, lanes(l, p).codes, sh.n, sh.k);
    }
    try s.synchronize();
    for (std.enums.values(Packed)) |p| {
        const src = shapeOf(l, p).src;
        if (src != 0 and !w.release(src)) return error.UnexpectedInkling2Layout;
    }
}

test "inkling2 cuda attn: packed projections laid out one after another" {
    var l: Layer = .{ .index = 5, .global = true, .boundary = false, .hq = 48, .hk = 8, .nq = 3, .experts = .{ .format = .nvfp4, .lo = 0, .hi = 384 } };
    l.wq = 1;
    l.wk = 1;
    l.wv = 1;
    l.wr = 1;
    l.wg = 1;
    l.wsink = 1;
    l.wo = 1;
    l.k_wproj = 1;
    l.v_wproj = 1;
    l.extra[0] = 1 << 20;
    const H = weights.hidden;
    try std.testing.expectEqual(@as(u64, 1 << 20), lanes(&l, .q).codes);
    try std.testing.expectEqual(@as(u64, (1 << 20) + 6144 * H * 2), lanes(&l, .k).codes);
    const sink = lanes(&l, .sink);
    try std.testing.expectEqual(@as(u32, 48), sink.n);
    try std.testing.expectEqual(@as(u32, 128), sink.npad);
    try std.testing.expectEqual(@as(u32, 6144), lanes(&l, .o).k);
    try std.testing.expectEqual(@as(u64, 128 * H * 2), @as(u64, packedBytes(&l, .k_wproj)));
    l.k_wproj = 0;
    try std.testing.expectEqual(@as(usize, 0), packedBytes(&l, .k_wproj));
}

test "inkling2 cuda attn: the splits a forward visits, and the form" {
    const sl: Kv = .{ .k = 0, .v = 0, .slots = 2047, .heads = 8, .window = window };
    // one decode row at 3000: window 1977 .. 3000, splits 7 .. 11
    const a = splits(sl, 3000, 1);
    try std.testing.expectEqual(@as(usize, 7), a.first);
    try std.testing.expectEqual(@as(usize, 5), a.count);
    const g: Kv = .{ .k = 0, .v = 0, .slots = 4096, .heads = 8, .window = 0 };
    const b = splits(g, 0, 25);
    try std.testing.expectEqual(@as(usize, 0), b.first);
    try std.testing.expectEqual(@as(usize, 1), b.count);
    const c = splits(g, 3999, 1);
    try std.testing.expectEqual(@as(usize, 16), c.count);
    try std.testing.expectEqual(Form.fused, formOf(.auto, 25, 1, 48));
    try std.testing.expectEqual(Form.split, formOf(.auto, 1, 16, 48));
    try std.testing.expectEqual(Form.fused, formOf(.auto, 2048, 16, 48));
    try std.testing.expectEqual(Form.split, formOf(.split, 2048, 1, 48));
}

test "inkling2 cuda attn: a captured window's splits cover every split its rows visit" {
    const sl: Kv = .{ .k = 0, .v = 0, .slots = 1023 + 64, .heads = 8, .window = window };
    try std.testing.expectEqual(@as(usize, 5), windowSplits(1));
    try std.testing.expectEqual(@as(usize, 5), windowSplits(2));
    try std.testing.expectEqual(@as(usize, 6), windowSplits(3));
    try std.testing.expectEqual(@as(usize, 6), windowSplits(8));
    for (1..max_rows + 1) |R| {
        var most: usize = 0;
        for (0..6000) |pos0| {
            const sp = splits(sl, pos0, R);
            // the derived first split is the eager one's, and the grid holds all of them
            try std.testing.expectEqual(sl.lo(pos0) / split_keys, sp.first);
            try std.testing.expect(sp.count <= windowSplits(R));
            most = @max(most, sp.count);
        }
        try std.testing.expectEqual(windowSplits(R), most);
    }
}

test "inkling2 cuda attn: the window's first position" {
    const kv: Kv = .{ .k = 0, .v = 0, .slots = 2048, .heads = 8, .window = window };
    try std.testing.expectEqual(@as(usize, 0), kv.lo(5));
    try std.testing.expectEqual(@as(usize, 1), kv.lo(1024));
    const g: Kv = .{ .k = 0, .v = 0, .slots = 4096, .heads = 8, .window = 0 };
    try std.testing.expectEqual(@as(usize, 0), g.lo(4000));
}

test "inkling2 cuda attn: the plan of a captured global layer, the fold, and the scratch need" {
    var k: Kernels = undefined;
    k.rows = max_rows;
    k.ab = false;
    k.form = .auto;
    k.grid = grid_splits;
    k.fold_min = fold_splits;
    const ctx = 131072 + 16;
    const g: Kv = .{ .k = 0, .v = 0, .slots = ctx, .heads = 2, .window = 0 };
    const sl: Kv = .{ .k = 0, .v = 0, .slots = 1023 + 2048, .heads = 2, .window = window };
    const dev: Dev = .{ .posp = 0, .positions = ctx };
    // a captured global layer: the scratch spans the slot, the launch at most grid_splits, the pairs folded once
    const pg = plan(&k, 1, 3000, dev, g, 12, 2);
    try std.testing.expect(pg.rows and pg.first == 0 and pg.grid == grid_splits and pg.span == 513 and pg.fold);
    // a sliding one: the derived window grid, no fold
    const ps = plan(&k, 8, 3000, dev, sl, 12, 2);
    try std.testing.expect(ps.first == -1 and ps.grid == 6 and ps.span == 6 and !ps.fold);
    // a small slot keeps its whole grid and the in-block fold
    const small = plan(&k, 1, 100, .{ .posp = 0, .positions = 3264 }, g, 12, 2);
    try std.testing.expect(small.grid == 13 and small.span == 13 and !small.fold);
    // eager: the fold above fold_splits splits only
    try std.testing.expect(!plan(&k, 1, 64 * 256 - 1, null, g, 12, 2).fold);
    try std.testing.expect(plan(&k, 1, 64 * 256, null, g, 12, 2).fold);
    try std.testing.expectEqual(Form.fused, plan(&k, 2048, 100000, null, g, 12, 2).form);
    // the bytes: pairs, partials, scores, folded pairs, each 256-aligned
    const n: usize = 513 * 12;
    try std.testing.expectEqual(std.mem.alignForward(usize, std.mem.alignForward(usize, n * 8, 256) + n * 512, 256) + n * 512 + 12 * 8, pg.scratch(1, 12));
    // the need at 131k (W = 4): 64-row split forms at the deepest position lead (st + partials over 513 splits)
    const need = scratchNeed(&k, 12, 2, ctx, 2048, 8);
    try std.testing.expect(need >= pg.scratch(8, 12) and need >= 513 * 64 * 12 * 520);
    try std.testing.expect(need < 256 << 20);
    // every eager forward and every captured one up to 8 rows fits in it, at any position
    var R: usize = 1;
    while (R <= 2048) : (R = if (R < 70) R + 1 else R * 2) {
        var pos0: usize = 0;
        while (pos0 + R <= ctx) : (pos0 += 997) {
            try std.testing.expect(plan(&k, R, pos0, null, g, 12, 2).scratch(R, 12) <= need);
            try std.testing.expect(plan(&k, R, pos0, null, sl, 12, 2).scratch(R, 12) <= need);
            if (R <= 8) try std.testing.expect(plan(&k, R, pos0, dev, g, 12, 2).scratch(R, 12) <= need);
        }
    }
    // at 1M a slot, the partials' cap moves deep 64-row forwards to the fused form; the need stays under 1 GiB
    try std.testing.expect(scratchNeed(&k, 12, 2, 1 << 20, 2048, 8) < 1 << 30);
    // TF_INK2_ATTN=split at 2048 rows needs gigabytes: the start-up check refuses it
    k.form = .split;
    try std.testing.expect(scratchNeed(&k, 12, 2, ctx, 2048, 8) > 2 << 30);
}
