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

/// The attention core: `kx` (the default: kortexa-ai's fp32 tile arithmetic, inkling2_kx.cu, `Kx` and `coreKx`), or
/// v0.1's eager-rounded core (`v01`, TF_INK2_ATTN=v01, hidden: its own plan picks fused or split; `fused` / `split`
/// force one of its forms, for tools).
pub const Form = enum { kx, v01, fused, split };

/// kx: the global layers' key split S (keys a split, absolute positions sS .. sS + S - 1; bits depend on it). A
/// compile-time constant of the engine, never `split_keys` (the pool's page); tools take the others (`KxOpt.s`).
/// 256: chosen on speed alone, with the packed C 1 kx-split.
pub const kx_split_keys: usize = 256;
/// The S every image holds (tf_kx_split / tf_kx_walk / tf_kx_merge instances, inkling2_kx_fp32.cuh).
pub const kx_splits = [_]usize{ 256, 512, 1024, 2048, 4096 };
/// kx-split's launch: at most this many splits a CTA column (a CTA walks splits z, z + G, ...; bit-neutral).
pub const kx_grid: usize = 128;
/// Eager global forwards take kx-split (partials + merge) up to this many rows while its scratch stays under
/// `kx_split_cap`, else kx-split-walk (the merge in registers, no partials). Bit-neutral.
pub const kx_split_rows: usize = 64;
pub const kx_split_cap: usize = 512 << 20;
/// Eager sliding forwards take kx-split up to this many rows (over two or more splits), else kx-split-walk: from four
/// 16-row blocks a head on, a block's walk over its window (about 17 tiles) costs less than the partials and the merge
/// (GB10). Bit-neutral.
pub const kx_sliding_split_rows: usize = 48;
/// kx: the sliding layers' key split (absolute 256-key splits from the window start's split, merged in order as the
/// global layers' are; bits depend on it). Fixed: tools' `KxOpt.s` moves the global layers' S only.
pub const kx_sliding_split: usize = 256;
/// Rows a kx CTA; forwards up to this many rows take the column split C = 4 (kx-dec), wider ones C = 1. Bit-neutral.
pub const kx_rows: usize = 16;
/// Global-layer forwards up to this many rows pack the query heads of a kv head into a CTA's rows (GQA packing, design
/// 5.4: a key and value tile read once for hq / hk heads). Sliding layers stay unpacked and take kx-split's C 1: a
/// sliding split is at most 4 tiles, so the CTA count bounds the time, and unpacked C 1 measured fastest (sm_120, 1 and
/// 8 rows). Bit-neutral.
pub const kx_pack_rows: usize = 16;

/// kx's bit-neutral launch choices (tools and the core bench move them; the engine keeps the defaults).
pub const KxOpt = struct {
    /// the key split S (one of kx_splits)
    s: usize = kx_split_keys,
    /// a forced column split C (1, 2 or 4; split forms 1 or 4), else C = 4 up to kx_rows rows and 1 above
    c: ?usize = null,
    /// kx-split's grid in splits
    g: usize = kx_grid,
    /// the default plan, or kx-split wherever its scratch fits, kx-split-walk always, or the tile kernel over every key
    /// of the window (no split: equal to kx-split where the keys lie in one split)
    global: enum { auto, split, walk, tile } = .auto,
    /// a forced GQA packing (on or off), else global layers packed up to kx_pack_rows rows
    pack: ?bool = null,
};

fn env(name: [*:0]const u8) []const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else "";
}

/// TF_INK2_ATTN's value as a form: unset, empty or `kx` -> kx; `v01`, `fused`, `split`; anything else null.
pub fn parseForm(v: []const u8) ?Form {
    if (v.len == 0) return .kx;
    return std.meta.stringToEnum(Form, v);
}

/// The process's form (TF_INK2_ATTN), refused by name when unknown (a typo must never run the other arithmetic), and
/// v0.1's switches refused under kx (TF_INK2_ATTN_AB mixes v0.1's forms by position, TF_INK2_ATTN_ROWS picks them).
pub fn envForm() !Form {
    const v = env("TF_INK2_ATTN");
    const f = parseForm(v) orelse {
        std.log.err("TF_INK2_ATTN={s} is not an attention form (kx, v01)", .{v});
        return error.UnknownAttnForm;
    };
    if (f == .kx and env("TF_INK2_ATTN_AB").len > 0) {
        std.log.err("TF_INK2_ATTN_AB={s} needs TF_INK2_ATTN=v01 (it alternates v0.1's forms)", .{env("TF_INK2_ATTN_AB")});
        return error.AttnAbNeedsV01;
    }
    if (f == .kx and env("TF_INK2_ATTN_ROWS").len > 0) {
        std.log.err("TF_INK2_ATTN_ROWS={s} needs TF_INK2_ATTN=v01 (it picks v0.1's forms)", .{env("TF_INK2_ATTN_ROWS")});
        return error.AttnRowsNeedsV01;
    }
    return f;
}

/// The settings fingerprint's attention part (cuda_engine.zig): the resolved form (unset and `kx` agree; an unknown
/// value as given, refused later at Kernels.load) and v0.1's two switches.
pub fn fingerprintText(buf: []u8) []const u8 {
    const v = env("TF_INK2_ATTN");
    const form = if (parseForm(v)) |f| @tagName(f) else v;
    return std.fmt.bufPrint(buf, "attn {s} TF_INK2_ATTN_ROWS={s} TF_INK2_ATTN_AB={s}", .{ form, env("TF_INK2_ATTN_ROWS"), env("TF_INK2_ATTN_AB") }) catch "attn ?";
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
    /// inkling2_kx.cu (loaded whenever the image holds it: the tools run both cores in one process)
    kx: ?Kx = null,
    kxo: KxOpt = .{},
    /// tf_kx_counts when the kernels were loaded (`kxExceptions` prints the change)
    kx_counts0: [3]u64 = @splat(0),

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const form = try envForm();
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_attn);
        errdefer mod.unload();
        var lane = try cuda.qmmf.Lane.load(d, (try d.currentCapability()) / 10);
        errdefer lane.unload();
        try lane.loadPrompt(d);
        var k: Kernels = .{
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
            .form = form,
        };
        if (cuda.kernels.inkling2_kx.len > 0) {
            k.kx = try Kx.load(d);
            k.kx_counts0 = try k.kx.?.counts(d);
        } else if (form == .kx) return error.KxNotBuilt;
        return k;
    }

    /// The form census a rank prints at start-up (the exactness gates grep it).
    pub fn census(k: *const Kernels, buf: []u8) []const u8 {
        return switch (k.form) {
            .kx => std.fmt.bufPrint(buf, "attn: sliding kx-split S={d}, global kx-split S={d} G={d}, fp32 (kortexa-ai e2614fe tiles)", .{ kx_sliding_split, k.kxo.s, k.kxo.g }) catch "attn: kx",
            .v01 => "attn: v01 (eager bf16 roundings)",
            .fused => "attn: v01 forced fused",
            .split => "attn: v01 forced split",
        };
    }

    /// kx's exception counters since load (call with the streams idle): (row, head) recomputes, (CTA, tile) pairs with a
    /// flushed P limb, with a flushed subnormal input.
    pub fn kxExceptions(k: *const Kernels, d: *const cuda.Driver) ![3]u64 {
        const x = k.kx orelse return @splat(0);
        const now = try x.counts(d);
        var out: [3]u64 = undefined;
        for (&out, now, k.kx_counts0) |*o, n, z| o.* = n -% z;
        return out;
    }

    /// The line every run that loads the kernels prints at its end (the gates require `rows 0` on real weights).
    pub fn printExceptions(k: *const Kernels, d: *const cuda.Driver) void {
        const e = k.kxExceptions(d) catch |err| {
            std.debug.print("attn exceptions: unread ({t})\n", .{err});
            return;
        };
        std.debug.print("attn exceptions: rows {d}, limb-flush tiles {d}, subnormal-input tiles {d}\n", .{ e[0], e[1], e[2] });
    }

    pub fn unload(k: *Kernels) void {
        if (k.kx) |*x| x.mod.unload();
        k.lane.unload();
        k.mod.unload();
    }

    /// inkling2_kx.cu: the kx forms (inkling2_kx_fp32.cuh), each ring and paged (`_pg`), the relative bias, the
    /// exception counters, and kortexa-ai/TensorFold e2614fe's tf_ik_attention as the tools' control.
    pub const Kx = struct {
        mod: cuda.Module,
        relative: cuda.Function,
        e2614fe: cuda.Function,
        /// [paged][C 1, 2, 4][packed]
        tile: [2][3][2]cuda.Function,
        /// [S][C 1, 4][paged][packed]
        split: [kx_splits.len][2][2][2]cuda.Function,
        /// [S][paged]
        merge: [kx_splits.len][2]cuda.Function,
        /// [S][paged][packed]
        walk: [kx_splits.len][2][2]cuda.Function,
        /// tf_kx_counts (3 x u64)
        counts_at: u64,

        fn load(d: *const cuda.Driver) !Kx {
            var mod = try cuda.Module.load(d, cuda.kernels.inkling2_kx);
            errdefer mod.unload();
            var x: Kx = undefined;
            x.mod = mod;
            x.relative = try mod.function("tf_kx_relative");
            x.e2614fe = try mod.function("tf_ik_attention_e2614fe");
            var buf: [64]u8 = undefined;
            for (0..2) |pg| {
                const sfx: []const u8 = if (pg == 1) "_pg" else "";
                for (0..2) |pk| {
                    const pks: []const u8 = if (pk == 1) "_pk" else "";
                    for ([_]usize{ 1, 2, 4 }, 0..) |c, ci| x.tile[pg][ci][pk] = try mod.function(try std.fmt.bufPrintSentinel(&buf, "tf_kx_tile_c{d}{s}{s}", .{ c, pks, sfx }, 0));
                    for (kx_splits, 0..) |s, si| {
                        for ([_]usize{ 1, 4 }, 0..) |c, ci| x.split[si][ci][pg][pk] = try mod.function(try std.fmt.bufPrintSentinel(&buf, "tf_kx_split_s{d}_c{d}{s}{s}", .{ s, c, pks, sfx }, 0));
                        x.walk[si][pg][pk] = try mod.function(try std.fmt.bufPrintSentinel(&buf, "tf_kx_walk_s{d}{s}{s}", .{ s, pks, sfx }, 0));
                    }
                }
                for (kx_splits, 0..) |s, si| x.merge[si][pg] = try mod.function(try std.fmt.bufPrintSentinel(&buf, "tf_kx_merge_s{d}{s}", .{ s, sfx }, 0));
            }
            x.counts_at = (try mod.global("tf_kx_counts")).ptr;
            return x;
        }

        /// tf_kx_counts now (the stream idle).
        pub fn counts(x: Kx, d: *const cuda.Driver) ![3]u64 {
            var v: [3]u64 = undefined;
            try cuda.memory.download(d, x.counts_at, std.mem.asBytes(&v));
            return v;
        }
    };

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
/// The order puts each source's projections side by side (xq's four, then xk's two, then xv's two), so `projectOne`
/// runs each source's group as one packed weight.
pub const Packed = enum { q, rel, gate, sink, k, k_wproj, v, v_wproj, o };

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

/// A layer's one-row projection groups (`projectOne`), each one packed weight read by one launch: a sliding layer's
/// six from xq; a global layer's four from xq, k and its window projection from xk, v and its from xv.
const sliding_group = [_]Packed{ .q, .rel, .gate, .sink, .k, .v };
const global_groups = [3][]const Packed{ &.{ .q, .rel, .gate, .sink }, &.{ .k, .k_wproj }, &.{ .v, .v_wproj } };

/// The columns `projectOne` writes for a layer of hq query and hk key heads: every member's padded rows, o excluded.
pub fn oneWidth(hq: usize, hk: usize) usize {
    const D = weights.head_dim;
    const p = cuda.modelopt.padded;
    return 2 * p(hq * D) + 2 * p(hk * D) + p(hq * weights.d_rel) + p(hq) + 2 * p(hk * weights.taps);
}

/// Projections that sit one after another at l.extra[0] (each padded to whole 128-row blocks, packed 64-row tile
/// first) are one packed weight; `members`' as one, its K slices pinned to the members' own. Null when a member is
/// missing, apart, or takes other slices (then a row's sums would not be the members' sums).
fn groupWeight(l: *const Layer, members: []const Packed) ?cuda.qmmf.Weight {
    const first = lanes(l, members[0]);
    const sk = cuda.qmmf.slices(first);
    var rows: usize = 0;
    for (members) |p| {
        const s = shapeOf(l, p);
        if (s.src == 0 or s.k != weights.hidden or cuda.qmmf.slices(lanes(l, p)) != sk) return null;
        if (lanes(l, p).codes != first.codes + rows * weights.hidden * 2) return null;
        rows += cuda.modelopt.padded(s.n);
    }
    const last = shapeOf(l, members[members.len - 1]).n;
    const w = cuda.modelopt.bf16Weight(first.codes, rows - cuda.modelopt.padded(last) + last, weights.hidden);
    return cuda.qmmf.pinned(w, sk) catch null;
}

/// A layer's one-row groups: weights, sources (0 xq, 1 xk, 2 xv) and first columns; null if any group is.
pub const Groups = struct {
    n: usize,
    w: [3]cuda.qmmf.Weight,
    src: [3]u2,
    col: [3]usize,
};

pub fn oneGroups(l: *const Layer) ?Groups {
    if (l.extra[0] == 0) return null;
    var g: Groups = .{ .n = 0, .w = undefined, .src = undefined, .col = undefined };
    const list: []const []const Packed = if (l.global) &global_groups else &.{&sliding_group};
    for (list, 0..) |members, i| {
        g.w[i] = groupWeight(l, members) orelse return null;
        g.src[i] = @intCast(if (l.global) i else 0);
        g.col[i] = oneColumn(l, members[0]);
        g.n += 1;
    }
    return g;
}

/// Member `p`'s first column in `projectOne`'s output row.
pub fn oneColumn(l: *const Layer, p: Packed) usize {
    return @intCast((lanes(l, p).codes - l.extra[0]) / (weights.hidden * 2));
}

/// `project`'s outputs as views of `projectOne`'s row `y` (bf16 [1, oneWidth]).
pub fn oneViews(l: *const Layer, y: u64) Proj {
    const v = struct {
        fn of(ll: *const Layer, yy: u64, p: Packed) Mat {
            return Mat.of(yy + oneColumn(ll, p) * 2, 1, shapeOf(ll, p).n, .bf16);
        }
    };
    return .{
        .q = v.of(l, y, .q),
        .k = v.of(l, y, .k),
        .v = v.of(l, y, .v),
        .rel = v.of(l, y, .rel),
        .gate = v.of(l, y, .gate),
        .sink = v.of(l, y, .sink),
        .dk = if (l.global) v.of(l, y, .k_wproj) else .{},
        .dv = if (l.global) v.of(l, y, .v_wproj) else .{},
    };
}

/// The most rows `projectOne` takes: one 16-row tile, where every member plans as at one row.
pub const one_rows = 16;

/// The split's members: a ProjSplit (inkling2_fuse.cu) of up to 8 (source, destination, columns, source stride).
const ProjSplit = extern struct { src: [8]u64 = @splat(0), dst: [8]u64 = @splat(0), n: [8]i32 = @splat(0), ld: [8]i32 = @splat(0) };

/// `project` of R <= one_rows rows as one launch a source (TF_INK2_FUSE proj, cuda_fuse.zig): each group's packed
/// weight (`oneGroups`) times its source. At one row into y's columns (bf16 [1, oneWidth]; `out` is then
/// `oneViews(l, y)`, nothing copied); at 2 or more into y as each group's dense [R, n] at R times its first column,
/// then one launch copies each member's columns to its own rows in `out`. Each member's values are the same bits as
/// its own launch (the same K slices meet in the same order; the 32- and 64-column tiles sum alike; a row's sums do
/// not depend on the row count).
pub fn projectOne(c: *const Ctx, l: *const Layer, g: Groups, xq: Mat, xk: Mat, xv: Mat, y: u64, out: Proj) !void {
    const lane = &c.k.attn.lane;
    const R = xq.rows;
    if (R == 0 or R > one_rows) return error.BadAttnShape;
    for (0..g.n) |i| {
        const x = switch (g.src[i]) {
            0 => xq,
            1 => xk,
            else => xv,
        };
        const w = g.w[i];
        if (x.dt != .bf16 or x.rows != R or x.cols != w.k) return error.BadAttnShape;
        const part_bytes = lane.partBytes(R, w);
        const part = if (part_bytes > 0) try c.tmp(part_bytes) else 0;
        try lane.matmulOut(c.stream, x.ptr, x.ld, R, w, y + R * g.col[i] * 2, .bf16, part);
    }
    if (R == 1) return;
    var sp: ProjSplit = .{};
    var m: usize = 0;
    for (0..g.n) |i| {
        const members: []const Packed = if (l.global) global_groups[i] else &sliding_group;
        for (members) |p| {
            const d = switch (p) {
                .q => out.q,
                .rel => out.rel,
                .gate => out.gate,
                .sink => out.sink,
                .k => out.k,
                .v => out.v,
                .k_wproj => out.dk,
                .v_wproj => out.dv,
                .o => unreachable,
            };
            const n = shapeOf(l, p).n;
            if (!d.is(R, n, .bf16) or !d.dense()) return error.BadAttnShape;
            sp.src[m] = y + (R * g.col[i] + oneColumn(l, p) - g.col[i]) * 2;
            sp.dst[m] = d.ptr;
            sp.n[m] = @intCast(n);
            sp.ld[m] = @intCast(g.w[i].n);
            m += 1;
        }
    }
    var a: cuda.Args = .{};
    a.add(sp);
    try cuda.launch.launch(c.k.fuse.proj_split, grid(@intCast(R), m, 1, 256), c.stream, &a);
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
    if (forced != .v01) return forced;
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
    /// kx: the form, its column split C and key split S (kx-split, kx-split-walk; span counts S-key splits from 0 and
    /// grid is kx-split's launch in CTAs along z, C x min(span, G))
    kx: KxForm = .none,
    c: usize = 1,
    s: usize = 0,
    /// kx: GQA packing (grid x over R * hq / hk slots, y over the kv heads)
    pack: bool = false,

    /// Op scratch the split form takes (the takes in `core`'s order, each 256-byte aligned as Scratch.take aligns).
    pub fn scratch(p: Plan, R: usize, hq: usize) usize {
        if (p.form == .kx) return kxScratch(p.kx, p.span, R, hq);
        if (p.form != .split) return 0;
        const n = p.span * R * hq;
        var at: usize = 0;
        for ([_]usize{ n * 8, n * weights.head_dim * 4, if (p.rows) n * split_keys * 2 else 0, if (p.fold) R * hq * 8 else 0 }) |b| {
            if (b > 0) at = std.mem.alignForward(usize, at, 256) + b;
        }
        return at;
    }
};

/// kx's forms (inkling2_kx_fp32.cuh): kx-tile / kx-dec (`tile`, C 1 / 4), kx-split + tf_kx_merge, kx-split-walk.
pub const KxForm = enum { none, tile, split, walk };

/// kx's op scratch (takes in `coreKx`'s order, each 256-byte aligned): the bias table, then kx-split's pairs (m, z),
/// flags and fp32 partials over `span` splits.
pub fn kxScratch(f: KxForm, span: usize, R: usize, hq: usize) usize {
    const n = span * R * hq;
    var at: usize = 0;
    for ([_]usize{ kxBiasBytes(R, hq), if (f == .split) n * 8 else 0, if (f == .split) n * 4 else 0, if (f == .split) n * weights.head_dim * 4 else 0 }) |b| {
        if (b > 0) at = std.mem.alignForward(usize, at, 256) + b;
    }
    return at;
}

/// The most kx_sliding_split splits a sliding window of R rows touches (positions lo(pos0) .. pos0 + R - 1): a captured
/// step launches these from the one its device position gives.
pub fn kxWindowSplits(R: usize) usize {
    return (window - 2 + R + kx_sliding_split - 1) / kx_sliding_split + 1;
}

/// kx's plan (sliding layers as global ones, over kx_sliding_split splits from the window start's): kx-split
/// when captured (the grid over the slot's splits, global: each CTA column walking z, z + G, ... up to the device
/// position; sliding: the window's splits from the device position's first), eager kx-split up to kx_split_rows rows
/// (sliding: kx_sliding_split_rows) over two or more splits while its scratch fits kx_split_cap, else kx-split-walk. The choice reads only the layer
/// kind, R, the span and hq (equal on every rank); every form gives the same bits.
fn kxPlan(o: KxOpt, R: usize, pos0: usize, dev: ?Dev, kv: Kv, hq: usize) Plan {
    const c = o.c orelse if (R <= kx_rows) @as(usize, 4) else 1;
    const sliding = kv.window != 0;
    const pk = o.pack orelse (!sliding and R <= kx_pack_rows);
    if (o.global == .tile) return .{ .form = .kx, .kx = .tile, .c = c, .grid = c, .pack = pk };
    const S = if (sliding) kx_sliding_split else o.s;
    // global: splits 0 .. the last; sliding: the window start's split .. the last (eager), or the most a window of R
    // rows touches (captured)
    const span = if (sliding)
        (if (dev) |x| @min(kxWindowSplits(R), (x.positions + S - 1) / S) else (pos0 + R - 1) / S - kv.lo(pos0) / S + 1)
    else if (dev) |x| (x.positions + S - 1) / S else (pos0 + R + S - 1) / S;
    // kx-split's column split: packed forwards and sliding layers take C 1 (a C-split CTA repeats the split's whole
    // Q.K; packed, the keys' reads and the Q.K bound the time, not P.V), unpacked global ones C 4 up to kx_rows rows
    const cs: usize = if (o.c) |x| (if (x >= 4) 4 else 1) else if (pk or sliding) 1 else (if (c >= 4) 4 else 1);
    const split: Plan = .{ .form = .kx, .kx = .split, .c = cs, .s = S, .span = span, .grid = cs * @min(span, @max(o.g, 1)), .pack = pk };
    if (dev != null) return split;
    const walk: Plan = .{ .form = .kx, .kx = .walk, .c = 1, .s = S, .span = span, .grid = 1, .pack = pk };
    const fits = split.scratch(R, hq) <= kx_split_cap;
    return switch (o.global) {
        .walk => walk,
        .split => if (fits) split else walk,
        else => if (R <= (if (sliding) kx_sliding_split_rows else kx_split_rows) and span >= 2 and fits) split else walk,
    };
}

/// The plan of a forward of R rows at pos0 (`dev`: a captured step, pos0 read on the device).
pub fn plan(k: *const Kernels, R: usize, pos0: usize, dev: ?Dev, kv: Kv, hq: usize, hk: usize) Plan {
    if (k.form == .kx) return kxPlan(k.kxo, R, pos0, dev, kv, hq);
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
    if (k.form == .kx) return kxNeed(k.kxo, hq, context, rows, graph_rows);
    var most: usize = 0;
    for ([_]usize{ 0, window }) |win| {
        const kv: Kv = .{ .k = 0, .v = 0, .slots = if (win == 0) context else window - 1 + rows, .heads = hk, .window = win };
        for (1..@min(rows, context) + 1) |R| {
            const deep = splits(kv, context - R, R).count;
            const c_max = if (win == 0) deep else @min(windowSplits(R), (context - 1) / split_keys + 1);
            const c: usize = switch (k.form) {
                .fused, .kx => 0,
                .split => c_max,
                .v01 => blk: {
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

/// kx's `scratchNeed`: every eager forward of 1 .. rows rows (sliding and global) at any position up to the slot's
/// context, and every captured one of 1 .. graph_rows rows over the slot. An eager kx-split's partials grow with the
/// span until the cap moves the forward to kx-split-walk, so a row count's largest is its deepest span still under
/// the cap (or, past every such span, the bias table alone).
pub fn kxNeed(o: KxOpt, hq: usize, context: usize, rows: usize, graph_rows: usize) usize {
    var most: usize = 0;
    const S = o.s;
    for (1..@min(rows, context) + 1) |R| {
        most = @max(most, kxBiasBytes(R, hq));
        const dev: Dev = .{ .posp = 0, .positions = context };
        const g: Kv = .{ .k = 0, .v = 0, .slots = context, .heads = 1, .window = 0 };
        const sl: Kv = .{ .k = 0, .v = 0, .slots = window - 1 + rows, .heads = 1, .window = window };
        if (R <= graph_rows) most = @max(most, kxPlan(o, R, 0, dev, g, hq).scratch(R, hq));
        if (R <= graph_rows) most = @max(most, kxPlan(o, R, 0, dev, sl, hq).scratch(R, hq));
        // eager sliding: at most the window's splits (kx-split up to kx_sliding_split_rows rows, or forced)
        if (o.global == .split or (o.global == .auto and R <= kx_sliding_split_rows))
            most = @max(most, kxScratch(.split, @min(kxWindowSplits(R), (context + kx_sliding_split - 1) / kx_sliding_split), R, hq));
        // eager global: the deepest span (the forward's last position below context) whose plan is kx-split
        var span = (context + S - 1) / S;
        while (span >= 2 and span * S >= R) : (span -= 1) {
            const p = kxPlan(o, R, @min(span * S, context) - R, null, g, hq);
            if (p.kx == .split) {
                most = @max(most, p.scratch(R, hq));
                break;
            }
            if (o.global == .walk or (o.global == .auto and R > kx_split_rows)) break;
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
    if (k.form == .kx) return coreKx(c, l, pos0, dev, q, rel, sink, gate, kv, out);
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

/// e2614fe's tf_ik_attention argument (kortexa's `InklingAttention`, passed by value), for its control kernel
/// (tf_ik_attention_e2614fe, tools only): `window` is the largest distance a key may be from its row (1023 sliding;
/// 0: every key up to the row), `capacity` the slots (key p at slot p % capacity).
pub const KxAttention = extern struct {
    q: u64,
    k: u64,
    v: u64,
    relative: u64,
    gate: u64,
    sink: u64,
    out: u64,
    rows: i32,
    pos: i32,
    hq: i32,
    hk: i32,
    capacity: i32,
    window: i32,
};

comptime {
    std.debug.assert(@sizeOf(KxAttention) == 80);
}

/// The kx core's relative bias table: bf16 [R, hq, rel_extent].
pub fn kxBiasBytes(R: usize, hq: usize) usize {
    return R * hq * rel_extent * 2;
}

/// kx's `window` argument: the largest distance a key may be from its row (1023 sliding; 0 global: every key up to
/// the row). Kv.window counts keys (1024, self included).
fn kxWindow(kv: Kv) i32 {
    return if (kv.window == 0) 0 else @intCast(kv.window - 1);
}

/// The relative bias table of R rows (tf_kx_relative, one launch: grid x a (row, head) pair, grid y 128 of its 1024
/// keys) into `bias`.
pub fn kxRelative(c: *const Ctx, l: *const Layer, rel: Mat, bias: u64) !void {
    const x = c.k.attn.kx orelse return error.KxNotLoaded;
    var a: cuda.Args = .{};
    for ([_]u64{ rel.ptr, l.rel_proj, bias }) |p| a.add(p);
    try cuda.launch.launch(x.relative, grid(@intCast(rel.rows * @as(usize, l.hq)), 1024 / 128, 1, 128), c.stream, &a);
}

/// e2614fe's tf_ik_attention (tools' control: ring keys, eager) on the bias table `bias` made by kxRelative.
pub fn kxControl(c: *const Ctx, l: *const Layer, pos0: usize, q: Mat, bias: u64, sink: Mat, gate: Mat, kv: Kv, out: Mat) !void {
    const x = c.k.attn.kx orelse return error.KxNotLoaded;
    if (kv.paged() or kv.run != null) return error.BadAttnShape;
    var b: cuda.Args = .{};
    b.add(KxAttention{ .q = q.ptr, .k = kv.k, .v = kv.v, .relative = bias, .gate = gate.ptr, .sink = sink.ptr, .out = out.ptr, .rows = @intCast(q.rows), .pos = @intCast(pos0), .hq = @intCast(l.hq), .hk = @intCast(l.hk), .capacity = @intCast(kv.slots), .window = kxWindow(kv) });
    try cuda.launch.launch(x.e2614fe, grid(@intCast((q.rows + kx_rows - 1) / kx_rows), l.hq, 1, 128), c.stream, &b);
}

fn kxIndex(s: usize) !usize {
    return std.mem.indexOfScalar(usize, &kx_splits, s) orelse error.KxSplitNotBuilt;
}

/// kx (the default core): the relative bias table [R, hq, 1024] (tf_kx_relative), then the plan's form
/// (inkling2_kx_fp32.cuh): kx-tile / kx-dec over 16-row blocks a query head (grid z the column split C), kx-split's
/// per-split partials then tf_kx_merge's ordered merge, or kx-split-walk. Scalar arguments q, bias, k, v, gate, sink,
/// out, [st, fl, op,] R, pos0, hq, hk, slots, window, posp[, pages, page]: kv.k and kv.v are 8-byte slots, so
/// Ctx.launchOn records the launches on a ring view of a run. `dev`: pos0 read on the device (every kernel).
fn coreKx(c: *const Ctx, l: *const Layer, pos0: usize, dev: ?Dev, q: Mat, rel: Mat, sink: Mat, gate: Mat, kv: Kv, out: Mat) !void {
    const R = q.rows;
    const hq: usize = l.hq;
    const hk: usize = l.hk;
    const k = &c.k.attn;
    const x = k.kx orelse return error.KxNotLoaded;
    const p = kxPlan(k.kxo, R, pos0, dev, kv, hq);
    const pg: usize = @intFromBool(kv.paged());
    // the takes in kxScratch's order: the bias table, then kx-split's pairs, flags and partials
    const bias = try c.tmp(kxBiasBytes(R, hq));
    try kxRelative(c, l, rel, bias);
    const n = p.span * R * hq;
    const st = if (p.kx == .split) try c.tmp(n * 8) else 0;
    const fl = if (p.kx == .split) try c.tmp(n * 4) else 0;
    const op = if (p.kx == .split) try c.tmp(n * weights.head_dim * 4) else 0;
    var a: cuda.Args = .{};
    for ([_]u64{ q.ptr, bias, kv.k, kv.v, gate.ptr, sink.ptr, out.ptr }) |v| a.add(v);
    if (p.kx == .split) for ([_]u64{ st, fl, op }) |v| a.add(v);
    for ([_]usize{ R, pos0, hq, hk, kv.slots }) |v| a.add(@as(i32, @intCast(v)));
    a.add(kxWindow(kv));
    a.add(if (dev) |d| d.posp else @as(u64, 0));
    if (kv.paged()) kv.addPaged(&a);
    // a CTA's 16 slots: rows of one query head, or (packed) the (row, head) pairs of one kv head's query heads
    if (p.pack and (hq % hk != 0 or hq / hk > kx_rows)) return error.BadAttnShape;
    const slots = if (p.pack) R * (hq / hk) else R;
    const blocks: u32 = @intCast((slots + kx_rows - 1) / kx_rows);
    const gy = if (p.pack) hk else hq;
    const pk: usize = @intFromBool(p.pack);
    switch (p.kx) {
        .tile => {
            const ci: usize = switch (p.c) {
                1 => 0,
                2 => 1,
                4 => 2,
                else => return error.BadAttnShape,
            };
            try c.launchOn(kv.run, kv.k, kv.v, x.tile[pg][ci][pk], grid(blocks, gy, p.c, 128), &a);
        },
        .walk => try c.launchOn(kv.run, kv.k, kv.v, x.walk[try kxIndex(p.s)][pg][pk], grid(blocks, gy, 1, 128), &a),
        .split => {
            const si = try kxIndex(p.s);
            try c.launchOn(kv.run, kv.k, kv.v, x.split[si][if (p.c == 4) 1 else 0][pg][pk], grid(blocks, gy, p.grid, 128), &a);
            try c.launchOn(kv.run, kv.k, kv.v, x.merge[si][pg], grid(@intCast(R), hq, 1, 128), &a);
        },
        .none => unreachable,
    }
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
    try std.testing.expectEqual(@as(u64, (1 << 20) + (6144 + 768 + 6144 + 128) * H * 2), lanes(&l, .k).codes); // after q, rel, gate, sink
    const sink = lanes(&l, .sink);
    try std.testing.expectEqual(@as(u32, 48), sink.n);
    try std.testing.expectEqual(@as(u32, 128), sink.npad);
    try std.testing.expectEqual(@as(u32, 6144), lanes(&l, .o).k);
    try std.testing.expectEqual(@as(u64, 128 * H * 2), @as(u64, packedBytes(&l, .k_wproj)));
    l.k_wproj = 0;
    try std.testing.expectEqual(@as(usize, 0), packedBytes(&l, .k_wproj));
}

test "inkling2 cuda attn: a layer's one-row projections as one packed weight a source" {
    // the W = 4 rank shape: 12 query heads, 2 key heads
    var l: Layer = .{ .index = 0, .global = false, .boundary = false, .hq = 12, .hk = 2, .nq = 1, .experts = .{ .format = .nvfp4, .lo = 0, .hi = 96 } };
    l.wq = 1;
    l.wk = 1;
    l.wv = 1;
    l.wr = 1;
    l.wg = 1;
    l.wsink = 1;
    l.wo = 1;
    l.extra[0] = 1 << 20;
    var g = oneGroups(&l).?;
    try std.testing.expectEqual(@as(usize, 1), g.n);
    try std.testing.expectEqual(@as(u32, 3968), g.w[0].n); // v last, sink's padding in between
    try std.testing.expectEqual(@as(u32, 3968), g.w[0].npad);
    try std.testing.expectEqual(@as(u32, 8), g.w[0].sk);
    try std.testing.expectEqual(@as(u32, 4), cuda.qmmf.splitK(g.w[0].n, g.w[0].k)); // unpinned it would sum 4 slices
    try std.testing.expectEqual(@as(u64, 1 << 20), g.w[0].codes);
    for (sliding_group) |p| try std.testing.expectEqual(@as(u32, 8), cuda.qmmf.slices(lanes(&l, p)));
    for ([_]Packed{ .q, .rel, .gate, .sink, .k, .v }, [_]usize{ 0, 1536, 1792, 3328, 3456, 3712 }) |p, col| try std.testing.expectEqual(col, oneColumn(&l, p));
    var v = oneViews(&l, 1 << 30);
    try std.testing.expectEqual(@as(u64, (1 << 30) + 1792 * 2), v.gate.ptr);
    try std.testing.expect(v.sink.is(1, 12, .bf16) and v.rel.is(1, 192, .bf16) and v.dk.rows == 0);
    // a global layer: xq's four, xk's two, xv's two
    l.global = true;
    l.k_wproj = 1;
    l.v_wproj = 1;
    g = oneGroups(&l).?;
    try std.testing.expectEqual(@as(usize, 3), g.n);
    try std.testing.expectEqualSlices(u2, &.{ 0, 1, 2 }, g.src[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3456, 3840 }, g.col[0..3]);
    try std.testing.expectEqual(@as(u32, 3340), g.w[0].n);
    try std.testing.expectEqual(@as(u32, 264), g.w[1].n);
    try std.testing.expectEqual(@as(u32, 384), g.w[1].npad);
    try std.testing.expectEqual(g.w[0].codes + 3456 * weights.hidden * 2, g.w[1].codes);
    for (g.w[0..3]) |w| try std.testing.expectEqual(@as(u32, 8), w.sk);
    try std.testing.expectEqual(oneWidth(12, 2), @as(usize, 4224));
    v = oneViews(&l, 0);
    try std.testing.expect(v.dk.is(1, 8, .bf16) and v.dk.ptr == 3712 * 2 and v.dv.ptr == 4096 * 2);
    // W = 1 (48, 8): q takes 2 slices, k 8: none
    l.global = false;
    l.k_wproj = 0;
    l.v_wproj = 0;
    l.hq = 48;
    l.hk = 8;
    try std.testing.expect(oneGroups(&l) == null);
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
    try std.testing.expectEqual(Form.fused, formOf(.v01, 25, 1, 48));
    try std.testing.expectEqual(Form.split, formOf(.v01, 1, 16, 48));
    try std.testing.expectEqual(Form.fused, formOf(.v01, 2048, 16, 48));
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

test "inkling2 cuda attn: the kx core's argument and bias table" {
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(KxAttention));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(KxAttention, "rows"));
    try std.testing.expectEqual(@as(usize, 76), @offsetOf(KxAttention, "window"));
    try std.testing.expectEqual(@as(usize, 48 << 20), kxBiasBytes(2048, 12));
    try std.testing.expectEqual(Form.kx, parseForm("").?);
    try std.testing.expectEqual(Form.kx, parseForm("kx").?);
    try std.testing.expectEqual(Form.v01, parseForm("v01").?);
    try std.testing.expectEqual(Form.fused, parseForm("fused").?);
    try std.testing.expect(parseForm("auto") == null and parseForm("kxx") == null and parseForm("KX") == null);
}

test "inkling2 cuda attn: kx's dispatch reads the layer kind, R, the span and nothing else" {
    const o: KxOpt = .{ .s = 1024 };
    const sl: Kv = .{ .k = 0, .v = 0, .slots = 1023 + 2048, .heads = 2, .window = window };
    const g: Kv = .{ .k = 0, .v = 0, .slots = 1 << 20, .heads = 2, .window = 0 };
    // sliding: 256-key splits from the window start's whatever S; eager kx-split up to 48 rows over 2+ splits, else
    // kx-split-walk; captured kx-split over the most splits a window of R rows touches
    for ([_]usize{ 1, 8, 16, 48 }) |R| {
        const p = kxPlan(o, R, 3000, null, sl, 12);
        try std.testing.expect(p.kx == .split and p.s == kx_sliding_split and p.span == (3000 + R - 1) / 256 - (3000 - 1023) / 256 + 1);
    }
    try std.testing.expectEqual(@as(usize, 5), kxPlan(o, 1, 3000, null, sl, 12).span);
    try std.testing.expectEqual(@as(usize, 4), kxPlan(o, 1, 1279, null, sl, 12).span);
    try std.testing.expectEqual(KxForm.walk, kxPlan(o, 1, 255, null, sl, 12).kx);
    try std.testing.expect(kxPlan(o, 49, 3000, null, sl, 12).kx == .walk and kxPlan(o, 64, 3000, null, sl, 12).kx == .walk);
    try std.testing.expect(kxPlan(o, 65, 3000, null, sl, 12).kx == .walk and kxPlan(o, 2048, 0, null, sl, 12).kx == .walk);
    try std.testing.expect(kxWindowSplits(1) == 5 and kxWindowSplits(2) == 5 and kxWindowSplits(3) == 6 and kxWindowSplits(16) == 6);
    const sc = kxPlan(o, 4, 3000, .{ .posp = 1, .positions = 131072 }, sl, 12);
    try std.testing.expect(sc.kx == .split and sc.span == 6 and sc.s == kx_sliding_split);
    try std.testing.expectEqual(@as(usize, 2), kxPlan(o, 1, 5, .{ .posp = 1, .positions = 300 }, sl, 12).span);
    try std.testing.expectEqual(KxForm.tile, kxPlan(.{ .global = .tile }, 1, 3000, null, sl, 12).kx);
    // global, eager: one split (p < S) -> walk; 2+ splits up to 64 rows -> split; wider -> walk
    try std.testing.expectEqual(KxForm.walk, kxPlan(o, 1, 1000, null, g, 12).kx);
    const sp = kxPlan(o, 1, 1024, null, g, 12);
    try std.testing.expect(sp.kx == .split and sp.span == 2 and sp.c == 1 and sp.grid == 2 and sp.s == 1024 and sp.pack);
    try std.testing.expect(kxPlan(.{ .s = 1024, .pack = false }, 1, 1024, null, g, 12).c == 4 and kxPlan(.{ .s = 1024, .pack = false }, 1, 1024, null, g, 12).grid == 8);
    try std.testing.expect(kxPlan(o, 64, 100000, null, g, 12).kx == .split and kxPlan(o, 64, 100000, null, g, 12).c == 1);
    try std.testing.expectEqual(KxForm.walk, kxPlan(o, 65, 100000, null, g, 12).kx);
    // the cap: 64 rows at hq 24 over 1025 splits (~806 MiB of partials) -> walk
    try std.testing.expectEqual(KxForm.walk, kxPlan(o, 64, (1 << 20) - 64, null, g, 24).kx);
    // captured: always split, the grid C x min(span, G) (packed: C 1)
    const cp = kxPlan(o, 8, 5, .{ .posp = 1, .positions = 1048592 }, g, 12);
    try std.testing.expect(cp.kx == .split and cp.span == 1025 and cp.grid == kx_grid and cp.c == 1);
    try std.testing.expect(kxPlan(.{ .c = 4 }, 8, 5, .{ .posp = 1, .positions = 1048592 }, g, 12).grid == 4 * kx_grid);
    // forced forms (tools)
    try std.testing.expectEqual(KxForm.walk, kxPlan(.{ .global = .walk }, 1, 5000, null, g, 12).kx);
    try std.testing.expectEqual(KxForm.tile, kxPlan(.{ .global = .tile }, 1, 5000, null, g, 12).kx);
    try std.testing.expectEqual(KxForm.split, kxPlan(.{ .global = .split }, 2048, 5000, null, g, 12).kx);
    try std.testing.expectEqual(@as(usize, 7), kxPlan(.{ .g = 7, .c = 1 }, 1, 100000, null, g, 12).grid);
    // GQA packing: global layers up to kx_pack_rows rows (eager and captured); sliding layers unpacked, C 1, unless forced
    try std.testing.expect(kxPlan(o, 1, 100000, null, g, 12).pack and kxPlan(o, 16, 100000, null, g, 12).pack and !kxPlan(o, 17, 100000, null, g, 12).pack);
    try std.testing.expect(kxPlan(o, 8, 5, .{ .posp = 1, .positions = 131072 }, g, 12).pack);
    try std.testing.expect(!kxPlan(o, 1, 3000, null, sl, 12).pack and kxPlan(.{ .pack = true }, 1, 3000, null, sl, 12).pack);
    try std.testing.expect(kxPlan(o, 1, 3000, null, sl, 12).c == 1 and kxPlan(o, 1, 3000, null, sl, 12).grid == 5);
    try std.testing.expect(kxPlan(.{ .c = 4 }, 1, 3000, null, sl, 12).c == 4 and kxPlan(.{ .c = 4 }, 1, 3000, null, sl, 12).grid == 20);
    try std.testing.expect(!kxPlan(.{ .pack = false }, 1, 100000, null, g, 12).pack);
}

test "inkling2 cuda attn: kx's scratch need against design 5.6, at most 512 MiB up to 1,048,592 positions" {
    const mib = struct {
        fn f(b: usize) f64 {
            return @as(f64, @floatFromInt(b)) / (1 << 20);
        }
    }.f;
    // rows the design's table gives (eager: the deepest forward; MiB, one decimal)
    const Row = struct { hq: usize, ctx: usize, s: usize, mib: f64 };
    for ([_]Row{
        .{ .hq = 12, .ctx = 3384, .s = 256, .mib = 48.0 },
        .{ .hq = 12, .ctx = 131128, .s = 1024, .mib = 51.0 },
        .{ .hq = 12, .ctx = 131128, .s = 512, .mib = 100.1 },
        .{ .hq = 24, .ctx = 131128, .s = 2048, .mib = 96.0 },
        .{ .hq = 24, .ctx = 524304, .s = 4096, .mib = 102.0 },
    }) |r| try std.testing.expectApproxEqAbs(r.mib, mib(kxNeed(.{ .s = r.s }, r.hq, r.ctx, 2048, 8)), 0.05);
    // every S at W = 4 and W = 2: the peak under the cap (bias included) and the 1 GiB arena; captured alone as 5.6
    for ([_]usize{ 12, 24 }) |hq| for (kx_splits) |s| {
        const need = kxNeed(.{ .s = s }, hq, 1048592, 2048, 8);
        try std.testing.expect(need <= kx_split_cap and need < 1 << 30);
        const g: Kv = .{ .k = 0, .v = 0, .slots = 1048592, .heads = 2, .window = 0 };
        const cap = kxPlan(.{ .s = s }, 8, 0, .{ .posp = 1, .positions = 1048592 }, g, hq).scratch(8, hq);
        try std.testing.expect(cap <= need);
    };
    try std.testing.expectApproxEqAbs(@as(f64, 393.5), mib(kxPlan(.{ .s = 256 }, 8, 0, .{ .posp = 1, .positions = 1048592 }, .{ .k = 0, .v = 0, .slots = 1048592, .heads = 4, .window = 0 }, 24).scratch(8, 24)), 0.05);
    // every eager forward fits the need, at any position (a sweep)
    const ctx = 131128;
    const need = kxNeed(.{}, 12, ctx, 2048, 8);
    const g: Kv = .{ .k = 0, .v = 0, .slots = ctx, .heads = 2, .window = 0 };
    var R: usize = 1;
    while (R <= 2048) : (R = if (R < 70) R + 1 else R * 2) {
        var pos0: usize = 0;
        while (pos0 + R <= ctx) : (pos0 += 1013) try std.testing.expect(kxPlan(.{}, R, pos0, null, g, 12).scratch(R, 12) <= need);
    }
    var k: Kernels = undefined;
    k.form = .kx;
    k.kxo = .{};
    try std.testing.expectEqual(need, scratchNeed(&k, 12, 2, ctx, 2048, 8));
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
    k.form = .v01;
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
