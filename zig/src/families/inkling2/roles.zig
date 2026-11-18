//! Every checkpoint tensor's role: what it is, its storage format, its expected dtype and shape, and how a rank of a
//! tensor-parallel (attention) and expert-parallel (MoE) split takes it. A tensor this file does not name, one the
//! manifest does not list, or one whose dtype or shape is not what the config implies refuses the checkpoint.
//!
//! Layouts (Transformers' modeling_inkling2.py, Apache-2.0): projections are [out, in] row-major; query heads own
//! consecutive rows of wq_du, gate_attn, wr_du (16 a head), attn_sink_proj and the clip scales, and consecutive columns
//! of wo_ud; key heads own consecutive rows of wk_dv, wv_dv, the static convolutions' weights, and of the dynamic
//! convolutions' weight_proj (4 taps a head), and columns of their static_w [taps, channels]. Query head h reads key
//! head h / (heads / kv_heads). Experts are fused along axis 0, so a run of experts is one contiguous byte range.
const std = @import("std");
const inventory = @import("inventory.zig");
const quant = @import("quant.zig");
const Config = @import("config.zig").Config;
const Diag = @import("diag.zig").Diag;
const fail = @import("diag.zig").fail;
const Tensor = inventory.Tensor;
const DType = inventory.DType;

/// How ranks take a tensor.
pub const Placement = enum {
    /// Rows by query head: `unit` rows a head.
    q_rows,
    /// Rows by key head.
    kv_rows,
    /// Columns of a 2-D tensor by query head (the output projection).
    q_cols,
    /// Columns of a 2-D tensor by key head (a global layer's static convolution taps).
    kv_cols,
    /// Axis 0 by expert.
    experts,
    /// Whole on every rank.
    replicated,
    /// Rows by vocabulary entry, canon's rule over the padded vocabulary (the unembed: each rank's logits for its
    /// share of the vocabulary, gathered in rank order).
    vocab_rows,
    /// Whole on rank 0 only (embed, vision tower).
    first,
    /// Whole on the drafter's rank only.
    drafter,

    pub fn sliced(p: Placement) bool {
        return switch (p) {
            .q_rows, .kv_rows, .q_cols, .kv_cols, .experts, .vocab_rows => true,
            .replicated, .first, .drafter => false,
        };
    }
};

/// What a tensor is, for the memory plan's categories.
pub const Kind = enum { attention, router, norm, conv, residual, experts, embed, unembed, vision, drafter };

/// A quantized weight's parts: the weight, its block scales, its per-expert amax, its outer scale denominator.
pub const Part = enum { weight, scale, amax, denominator };

pub const Role = struct {
    kind: Kind,
    placement: Placement,
    format: quant.Format,
    part: Part = .weight,
    layer: ?u16 = null,
    /// Rows or columns a head (1 for experts: axis 0 is the expert).
    unit: u64 = 1,
};

const Shape = []const u64;

const Ctx = struct {
    c: *const Config,
    m: *const quant.Manifest,
    d: ?*Diag,

    fn expect(x: Ctx, t: Tensor, dtype: DType, shape: Shape) !void {
        if (t.is(dtype, shape)) return;
        return fail(x.d, error.UnexpectedInkling2Tensor, "{s}: {s} {any}, expected {s} {any}", .{ t.name, t.dtype.name(), t.shape[0..t.rank], dtype.name(), shape });
    }

    fn unknown(x: Ctx, t: Tensor) error{UnknownInkling2Tensor} {
        return fail(x.d, error.UnknownInkling2Tensor, "{s}: not a tensor the inkling2 family knows", .{t.name});
    }
};

const companions = [_]struct { suffix: []const u8, part: Part }{
    .{ .suffix = "_outer_scale_denominator", .part = .denominator },
    .{ .suffix = "_scale", .part = .scale },
    .{ .suffix = "_amax", .part = .amax },
};

/// The manifest group of `t`, with the part a companion is of its listed weight.
fn listed(x: Ctx, t: Tensor, s: inventory.Section) !struct { group: *const quant.Group, part: Part } {
    if (x.m.lookup(s, t.name)) |g| return .{ .group = g, .part = .weight };
    for (companions) |cmp| if (std.mem.endsWith(u8, t.name, cmp.suffix)) {
        const base = t.name[0 .. t.name.len - cmp.suffix.len];
        if (x.m.lookup(s, base)) |g| {
            const ok = switch (g.format) {
                .nvfp4 => true,
                .mxfp8 => cmp.part == .scale,
                .fp32, .bf16 => false,
            };
            if (ok) return .{ .group = g, .part = cmp.part };
        }
    };
    return fail(x.d, error.UnknownInkling2Tensor, "{s}: not listed in hf_quant_config.json, nor a quantized weight's scale", .{t.name});
}

/// `name` = `prefix` ++ decimal index ++ "." ++ rest: the index and the rest.
fn indexed(name: []const u8, prefix: []const u8) ?struct { i: u32, rest: []const u8 } {
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const tail = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, tail, '.') orelse tail.len;
    const i = std.fmt.parseInt(u32, tail[0..dot], 10) catch return null;
    if (dot > 1 and tail[0] == '0') return null;
    return .{ .i = i, .rest = if (dot < tail.len) tail[dot + 1 ..] else "" };
}

fn oneOf(s: []const u8, names: []const []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, s, n)) return true;
    return false;
}

/// The role of `t`, its dtype and shape checked.
pub fn classify(c: *const Config, m: *const quant.Manifest, inv: *const inventory.Inventory, t: Tensor, d: ?*Diag) !Role {
    const x: Ctx = .{ .c = c, .m = m, .d = d };
    const s = inv.section(t);
    const l = try listed(x, t, s);
    const f = l.group.format;
    if (s == .mtp) return drafter(x, t, f);
    if (std.mem.startsWith(u8, t.name, "model.visual.")) return vision(x, t, f);
    const llm = "model.llm.";
    if (!std.mem.startsWith(u8, t.name, llm)) return x.unknown(t);
    const n = t.name[llm.len..];
    if (indexed(n, "layers.")) |li| return layer(x, t, f, l.part, li.i, li.rest);
    return top(x, t, f, n);
}

fn top(x: Ctx, t: Tensor, f: quant.Format, n: []const u8) !Role {
    const tc = &x.c.text;
    const h: u64 = tc.hidden;
    const v: u64 = tc.vocab;
    if (std.mem.eql(u8, n, "embed.weight")) {
        try x.expect(t, .bf16, &.{ v, h });
        return .{ .kind = .embed, .placement = .first, .format = f };
    }
    if (std.mem.eql(u8, n, "embed_norm.weight")) {
        try x.expect(t, .bf16, &.{h});
        return .{ .kind = .embed, .placement = .first, .format = f };
    }
    if (std.mem.eql(u8, n, "unembed.weight")) {
        try x.expect(t, .bf16, &.{ v, h });
        return .{ .kind = .unembed, .placement = .vocab_rows, .format = f };
    }
    if (std.mem.eql(u8, n, "norm.weight")) {
        try x.expect(t, .bf16, &.{h});
        return .{ .kind = .norm, .placement = .replicated, .format = f };
    }
    if (std.mem.eql(u8, n, "attn_res.output_query")) {
        try x.expect(t, .f32, &.{h});
        return .{ .kind = .residual, .placement = .replicated, .format = f };
    }
    // attention_input of layer i + 1 (layer 0 has none): three queries (q, k, v inputs) before a global layer
    if (indexed(n, "attn_res.attn_queries.")) |q| if (q.rest.len == 0 and q.i + 1 < tc.layers) {
        const l = q.i + 1;
        if (tc.global[l]) try x.expect(t, .f32, &.{ 3, h }) else try x.expect(t, .f32, &.{h});
        return .{ .kind = .residual, .placement = .replicated, .format = f, .layer = @intCast(l) };
    };
    if (indexed(n, "attn_res.mlp_queries.")) |q| if (q.rest.len == 0 and q.i < tc.layers) {
        try x.expect(t, .f32, &.{h});
        return .{ .kind = .residual, .placement = .replicated, .format = f, .layer = @intCast(q.i) };
    };
    return x.unknown(t);
}

fn layer(x: Ctx, t: Tensor, f: quant.Format, part: Part, li: u32, n: []const u8) !Role {
    const tc = &x.c.text;
    if (li >= tc.layers) return x.unknown(t);
    const g = tc.global[li];
    const L: u16 = @intCast(li);
    const h: u64 = tc.hidden;
    const H: u64 = tc.heads;
    const K: u64 = tc.kv_heads;
    const D: u64 = tc.head_dim;
    const R: u64 = tc.d_rel;
    const T: u64 = tc.conv_taps;
    const E: u64 = tc.experts;
    const I: u64 = tc.moe_intermediate;
    const Q = struct { name: []const u8, rows: u64, unit: u64 };
    for ([_]Q{
        .{ .name = "attn.wq_du.weight", .rows = H * D, .unit = D },
        .{ .name = "attn.gate_attn.weight", .rows = H * D, .unit = D },
        .{ .name = "attn.wr_du.weight", .rows = H * R, .unit = R },
        .{ .name = "attn.attn_sink_proj.weight", .rows = H, .unit = 1 },
    }) |q| if (std.mem.eql(u8, n, q.name)) {
        try x.expect(t, .bf16, &.{ q.rows, h });
        return .{ .kind = .attention, .placement = .q_rows, .format = f, .layer = L, .unit = q.unit };
    };
    if (std.mem.eql(u8, n, "attn.q_l2_clip.scale")) {
        try x.expect(t, .bf16, &.{ H, D });
        return .{ .kind = .attention, .placement = .q_rows, .format = f, .layer = L, .unit = 1 };
    }
    if (std.mem.eql(u8, n, "attn.rel_l2_clip.scale")) {
        try x.expect(t, .bf16, &.{ H, R });
        return .{ .kind = .attention, .placement = .q_rows, .format = f, .layer = L, .unit = 1 };
    }
    if (oneOf(n, &.{ "attn.wk_dv.weight", "attn.wv_dv.weight" })) {
        try x.expect(t, .bf16, &.{ K * D, h });
        return .{ .kind = .attention, .placement = .kv_rows, .format = f, .layer = L, .unit = D };
    }
    if (std.mem.eql(u8, n, "attn.wo_ud.weight")) {
        try x.expect(t, .bf16, &.{ h, H * D });
        return .{ .kind = .attention, .placement = .q_cols, .format = f, .layer = L, .unit = D };
    }
    if (std.mem.eql(u8, n, "attn.rel_logits_proj.proj")) {
        try x.expect(t, .bf16, &.{ R, if (g) tc.rel_extent else tc.window });
        return .{ .kind = .attention, .placement = .replicated, .format = f, .layer = L };
    }
    if (!g and oneOf(n, &.{ "attn.k_sconv.weight", "attn.v_sconv.weight" })) {
        try x.expect(t, .bf16, &.{ K * D, 1, T });
        return .{ .kind = .conv, .placement = .kv_rows, .format = f, .layer = L, .unit = D };
    }
    if (g and oneOf(n, &.{ "attn.k_sconv.static_w", "attn.v_sconv.static_w" })) {
        try x.expect(t, .bf16, &.{ T, K * D });
        return .{ .kind = .conv, .placement = .kv_cols, .format = f, .layer = L, .unit = D };
    }
    if (g and oneOf(n, &.{ "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj" })) {
        try x.expect(t, .bf16, &.{ K * T, h });
        return .{ .kind = .conv, .placement = .kv_rows, .format = f, .layer = L, .unit = T };
    }
    if ((!g and std.mem.eql(u8, n, "attn_norm.weight")) or (g and oneOf(n, &.{ "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight" })) or
        std.mem.eql(u8, n, "mlp_norm.weight"))
    {
        try x.expect(t, .bf16, &.{h});
        return .{ .kind = .norm, .placement = .replicated, .format = f, .layer = L };
    }
    if (oneOf(n, &.{ "attn_sconv.weight", "mlp_sconv.weight" })) {
        try x.expect(t, .bf16, &.{ h, 1, T });
        return .{ .kind = .conv, .placement = .replicated, .format = f, .layer = L };
    }
    if (std.mem.eql(u8, n, "mlp.gate.weight")) {
        try x.expect(t, .bf16, &.{ E, h });
        return .{ .kind = .router, .placement = .replicated, .format = f, .layer = L };
    }
    if (std.mem.eql(u8, n, "mlp.gate.bias")) {
        try x.expect(t, .f32, &.{E});
        return .{ .kind = .router, .placement = .replicated, .format = f, .layer = L };
    }
    const ex = "mlp.experts.";
    if (std.mem.startsWith(u8, n, ex)) {
        const w = n[ex.len..];
        // gate and up stacked: [E, 2I, hidden]; down: [E, hidden, I]
        var rows: u64 = h;
        var cols: u64 = I;
        if (std.mem.startsWith(u8, w, "w13_weight")) {
            rows = 2 * I;
            cols = h;
        } else if (!std.mem.startsWith(u8, w, "w2_weight")) return x.unknown(t);
        const role: Role = .{ .kind = .experts, .placement = .experts, .format = f, .part = part, .layer = L };
        switch (part) {
            .weight => switch (f) {
                .nvfp4 => try x.expect(t, .u8, &.{ E, rows, cols / 2 }),
                .mxfp8 => try x.expect(t, .f8_e4m3, &.{ E, rows, cols }),
                else => return x.unknown(t),
            },
            .scale => switch (f) {
                .nvfp4 => try x.expect(t, .f8_e4m3, &.{ E, rows, cols / 16 }),
                .mxfp8 => try x.expect(t, .u8, &.{ E, rows, cols / 32 }),
                else => return x.unknown(t),
            },
            .amax => try x.expect(t, .f32, &.{E}),
            .denominator => {
                try x.expect(t, .f32, &.{1});
                var r = role;
                r.placement = .replicated;
                return r;
            },
        }
        return role;
    }
    return x.unknown(t);
}

/// A tensor known by name, with its shape (BF16, `rank` of `shape`'s dims).
const Named = struct { name: []const u8, shape: [2]u64, rank: u8 = 2 };

/// Checks `t` against the entry named `n`, or returns false when none is.
fn named(x: Ctx, t: Tensor, n: []const u8, table: []const Named) !bool {
    for (table) |e| if (std.mem.eql(u8, n, e.name)) {
        try x.expect(t, .bf16, e.shape[0..e.rank]);
        return true;
    };
    return false;
}

/// The vision tower (tokenspeed's inkling2_vision.py names; Transformers' Inkling2Vision* shapes): patch embedding
/// [hidden, 3 * patch^2], learned positions [grid^2, hidden], pre-norm, 23 pre-norm layers with layer scale, and the
/// projector [projection_hidden, merge^2 * hidden] then [text hidden, projection_hidden].
fn vision(x: Ctx, t: Tensor, f: quant.Format) !Role {
    const v = &x.c.vision;
    const w: u64 = v.hidden;
    const p: u64 = v.projection_hidden;
    const role: Role = .{ .kind = .vision, .placement = .first, .format = f };
    const n = t.name["model.visual.".len..];
    if (try named(x, t, n, &.{
        .{ .name = "proj_in.weight", .shape = .{ p, @as(u64, v.merge) *| v.merge *| w } },
        .{ .name = "proj_out.weight", .shape = .{ x.c.text.hidden, p } },
        .{ .name = "backbone.ln_pre.weight", .shape = .{ w, 0 }, .rank = 1 },
        .{ .name = "backbone.ln_pre.bias", .shape = .{ w, 0 }, .rank = 1 },
        .{ .name = "backbone.embeddings.class_embedding", .shape = .{ w, 0 }, .rank = 1 },
        .{ .name = "backbone.embeddings.cls_position_embedding", .shape = .{ w, 0 }, .rank = 1 },
        .{ .name = "backbone.embeddings.patch_embedding.weight", .shape = .{ w, 3 *| @as(u64, v.patch) *| v.patch } },
        .{ .name = "backbone.embeddings.position_embedding", .shape = .{ @as(u64, v.posemb_grid) *| v.posemb_grid, w } },
    })) return role;
    if (v.projector_bias and try named(x, t, n, &.{
        .{ .name = "proj_in.bias", .shape = .{ p, 0 }, .rank = 1 },
        .{ .name = "proj_out.bias", .shape = .{ x.c.text.hidden, 0 }, .rank = 1 },
    })) return role;
    if (indexed(n, "backbone.layers.")) |li| if (li.i < v.layers) {
        const i = v.intermediate();
        if (try named(x, t, li.rest, &.{
            .{ .name = "ln_1.weight", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "ln_1.bias", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "ln_2.weight", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "ln_2.bias", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "ls_1.gamma", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "ls_2.gamma", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "mlp.c_fc.weight", .shape = .{ i, w } },
            .{ .name = "mlp.c_fc.bias", .shape = .{ i, 0 }, .rank = 1 },
            .{ .name = "mlp.c_proj.weight", .shape = .{ w, i } },
            .{ .name = "mlp.c_proj.bias", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "self_attn.q_proj.weight", .shape = .{ w, w } },
            .{ .name = "self_attn.q_proj.bias", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "self_attn.k_proj.weight", .shape = .{ w, w } },
            .{ .name = "self_attn.k_proj.bias", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "self_attn.v_proj.weight", .shape = .{ w, w } },
            .{ .name = "self_attn.v_proj.bias", .shape = .{ w, 0 }, .rank = 1 },
            .{ .name = "self_attn.out_proj.weight", .shape = .{ w, w } },
            .{ .name = "self_attn.out_proj.bias", .shape = .{ w, 0 }, .rank = 1 },
        })) return role;
    };
    return x.unknown(t);
}

/// The DSpark drafter (tokenspeed's inkling2_mtp.py): fuse_weight [layers + 1, targets], the mask embedding, the
/// final norm, the Markov head (markov_w1 [vocab, rank]; with the rnn_hidden head, rnn_proj [3 rank, 2 rank +
/// 2 hidden] and hidden_out [hidden, rank]) and its layers (Q [heads * head_dim, hidden], K and V and their context
/// copies [kv_heads * head_dim, hidden], O [hidden, heads * head_dim], the dense SwiGLU, head_dim-wide QK norms).
fn drafter(x: Ctx, t: Tensor, f: quant.Format) !Role {
    const m = &x.c.draft;
    const h: u64 = m.hidden;
    const r: u64 = m.markov_rank;
    const role: Role = .{ .kind = .drafter, .placement = .drafter, .format = f };
    const n = t.name;
    if (try named(x, t, n, &.{
        .{ .name = "fuse_weight", .shape = .{ @as(u64, m.layers) + 1, m.target_count } },
        .{ .name = "mask_embed", .shape = .{ h, 0 }, .rank = 1 },
        .{ .name = "norm.weight", .shape = .{ h, 0 }, .rank = 1 },
        .{ .name = "markov_head.markov_w1.weight", .shape = .{ x.c.text.vocab, r } },
    })) return role;
    if (m.markov_rnn_hidden and try named(x, t, n, &.{
        .{ .name = "markov_head.dflare_norm.weight", .shape = .{ h, 0 }, .rank = 1 },
        .{ .name = "markov_head.hidden_out.weight", .shape = .{ h, r } },
        .{ .name = "markov_head.hidden_out.bias", .shape = .{ h, 0 }, .rank = 1 },
        .{ .name = "markov_head.rnn_proj.weight", .shape = .{ 3 * r, 2 * r + 2 * h } },
        .{ .name = "markov_head.rnn_proj.bias", .shape = .{ 3 * r, 0 }, .rank = 1 },
    })) return role;
    if (indexed(n, "layers.")) |li| if (li.i < m.layers) {
        const D: u64 = m.head_dim;
        const q = m.heads * D;
        const kv = m.kv_heads * D;
        const I: u64 = m.intermediate;
        if (try named(x, t, li.rest, &.{
            .{ .name = "dflare_norm.weight", .shape = .{ h, 0 }, .rank = 1 },
            .{ .name = "input_layernorm.weight", .shape = .{ h, 0 }, .rank = 1 },
            .{ .name = "post_attention_layernorm.weight", .shape = .{ h, 0 }, .rank = 1 },
            .{ .name = "mlp.gate_proj.weight", .shape = .{ I, h } },
            .{ .name = "mlp.up_proj.weight", .shape = .{ I, h } },
            .{ .name = "mlp.down_proj.weight", .shape = .{ h, I } },
            .{ .name = "self_attn.q_proj.weight", .shape = .{ q, h } },
            .{ .name = "self_attn.k_proj.weight", .shape = .{ kv, h } },
            .{ .name = "self_attn.v_proj.weight", .shape = .{ kv, h } },
            .{ .name = "self_attn.k_proj_ctx.weight", .shape = .{ kv, h } },
            .{ .name = "self_attn.v_proj_ctx.weight", .shape = .{ kv, h } },
            .{ .name = "self_attn.o_proj.weight", .shape = .{ h, q } },
            .{ .name = "self_attn.q_norm.weight", .shape = .{ D, 0 }, .rank = 1 },
            .{ .name = "self_attn.k_norm.weight", .shape = .{ D, 0 }, .rank = 1 },
            .{ .name = "self_attn.k_norm_ctx.weight", .shape = .{ D, 0 }, .rank = 1 },
        })) return role;
    };
    return x.unknown(t);
}

/// Tensors a text layer must hold, by suffix after `model.llm.layers.<i>.`.
const local_names = [_][]const u8{ "attn.wq_du.weight", "attn.gate_attn.weight", "attn.wr_du.weight", "attn.attn_sink_proj.weight", "attn.q_l2_clip.scale", "attn.rel_l2_clip.scale", "attn.wk_dv.weight", "attn.wv_dv.weight", "attn.wo_ud.weight", "attn.rel_logits_proj.proj", "attn.k_sconv.weight", "attn.v_sconv.weight", "attn_norm.weight", "attn_sconv.weight", "mlp_norm.weight", "mlp_sconv.weight", "mlp.gate.weight", "mlp.gate.bias", "mlp.experts.w13_weight", "mlp.experts.w13_weight_scale", "mlp.experts.w2_weight", "mlp.experts.w2_weight_scale" };
const global_names = [_][]const u8{ "attn.wq_du.weight", "attn.gate_attn.weight", "attn.wr_du.weight", "attn.attn_sink_proj.weight", "attn.q_l2_clip.scale", "attn.rel_l2_clip.scale", "attn.wk_dv.weight", "attn.wv_dv.weight", "attn.wo_ud.weight", "attn.rel_logits_proj.proj", "attn.k_sconv.static_w", "attn.k_sconv.weight_proj", "attn.v_sconv.static_w", "attn.v_sconv.weight_proj", "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight", "attn_sconv.weight", "mlp_norm.weight", "mlp_sconv.weight", "mlp.gate.weight", "mlp.gate.bias", "mlp.experts.w13_weight", "mlp.experts.w13_weight_scale", "mlp.experts.w2_weight", "mlp.experts.w2_weight_scale" };
const nvfp4_names = [_][]const u8{ "mlp.experts.w13_weight_amax", "mlp.experts.w13_weight_outer_scale_denominator", "mlp.experts.w2_weight_amax", "mlp.experts.w2_weight_outer_scale_denominator" };

/// Every tensor's role (parallel to `inv.tensors`), with every name the manifest lists present and every text
/// layer complete.
pub fn classifyAll(gpa: std.mem.Allocator, c: *const Config, m: *const quant.Manifest, inv: *const inventory.Inventory, d: ?*Diag) ![]Role {
    const out = try gpa.alloc(Role, inv.tensors.len);
    errdefer gpa.free(out);
    var names: std.StringHashMapUnmanaged(u32) = .empty;
    defer names.deinit(gpa);
    try names.ensureTotalCapacity(gpa, @intCast(inv.tensors.len));
    for (inv.tensors, out, 0..) |t, *r, i| {
        r.* = try classify(c, m, inv, t, d);
        names.putAssumeCapacity(t.name, @intCast(i));
    }
    for ([_]inventory.Section{ .model, .mtp }) |s| for (m.groups(s)) |g| for (g.tensors) |n| {
        const i = names.get(n) orelse return fail(d, error.MissingInkling2Tensor, "{s}: listed in hf_quant_config.json, not in the checkpoint", .{n});
        if (inv.section(inv.tensors[i]) != s) return fail(d, error.MissingInkling2Tensor, "{s}: in the wrong file", .{n});
    };
    var buf: [160]u8 = undefined;
    for (0..c.text.layers) |l| {
        const list: []const []const u8 = if (c.text.global[l]) &global_names else &local_names;
        for (list) |n| {
            const full = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.{s}", .{ l, n });
            if (!names.contains(full)) return fail(d, error.MissingInkling2Tensor, "{s}: missing", .{full});
        }
        const w13 = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.w13_weight", .{l});
        if (out[names.get(w13).?].format == .nvfp4) for (nvfp4_names) |n| {
            const full = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.{s}", .{ l, n });
            if (!names.contains(full)) return fail(d, error.MissingInkling2Tensor, "{s}: missing", .{full});
        };
    }
    for ([_][]const u8{ "model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight", "model.llm.norm.weight", "model.llm.attn_res.output_query" }) |n|
        if (!names.contains(n)) return fail(d, error.MissingInkling2Tensor, "{s}: missing", .{n});
    return out;
}

const testing = std.testing;
const config = @import("config.zig");

const Loaded = struct {
    c: Config,
    m: quant.Manifest,
    inv: inventory.Inventory,

    fn init() !Loaded {
        var m = try quant.parse(testing.allocator, quant.fixture, null);
        errdefer m.deinit();
        return .{ .c = try config.parse(testing.allocator, config.fixture, null), .m = m, .inv = try inventory.Inventory.fromText(testing.allocator, inventory.fixture, null) };
    }

    fn deinit(l: *Loaded) void {
        l.m.deinit();
        l.inv.deinit();
    }
};

test "inkling2 roles: every release tensor has one role, the formats the manifest gives" {
    var l = try Loaded.init();
    defer l.deinit();
    const roles = try classifyAll(testing.allocator, &l.c, &l.m, &l.inv, null);
    defer testing.allocator.free(roles);
    var by_format: [4]u64 = @splat(0);
    var by_kind: [@typeInfo(Kind).@"enum".field_names.len]usize = @splat(0);
    for (l.inv.tensors, roles) |t, r| {
        by_format[@backingInt(r.format)] += t.bytes();
        by_kind[@backingInt(r.kind)] += 1;
    }
    // NVFP4 experts of 40 layers with their scales, MXFP8 experts of 8
    try testing.expectEqual(@as(u64, 40 * 384 * 4608 * (2304 + 1152 + 288 + 144) + 40 * 2 * (384 * 4 + 4)), by_format[@backingInt(quant.Format.nvfp4)]);
    try testing.expectEqual(@as(u64, 8 * 384 * 4608 * (4608 + 2304 + 144 + 72)), by_format[@backingInt(quant.Format.mxfp8)]);
    try testing.expectEqual(@as(usize, 40 * 8 + 8 * 4), by_kind[@backingInt(Kind.experts)]);
    try testing.expectEqual(@as(usize, 84), by_kind[@backingInt(Kind.drafter)]);
    try testing.expectEqual(@as(usize, 8 + 23 * 18), by_kind[@backingInt(Kind.vision)]);
    const wo = try classify(&l.c, &l.m, &l.inv, l.inv.find("model.llm.layers.5.attn.wo_ud.weight").?, null);
    try testing.expectEqual(Placement.q_cols, wo.placement);
    const sw = try classify(&l.c, &l.m, &l.inv, l.inv.find("model.llm.layers.5.attn.k_sconv.static_w").?, null);
    try testing.expectEqual(Placement.kv_cols, sw.placement);
    const wp = try classify(&l.c, &l.m, &l.inv, l.inv.find("model.llm.layers.11.attn.v_sconv.weight_proj").?, null);
    try testing.expectEqual(@as(u64, 4), wp.unit);
    const den = try classify(&l.c, &l.m, &l.inv, l.inv.find("model.llm.layers.0.mlp.experts.w2_weight_outer_scale_denominator").?, null);
    try testing.expectEqual(Placement.replicated, den.placement);
    try testing.expectEqual(Part.denominator, den.part);
}

test "inkling2 roles: an unknown, unlisted, misshaped or missing tensor refuses the checkpoint" {
    var l = try Loaded.init();
    defer l.deinit();
    var d: Diag = .{};
    var t = l.inv.find("model.llm.layers.0.attn.wq_du.weight").?;
    t.name = "model.llm.layers.0.attn.wq_new.weight";
    try testing.expectError(error.UnknownInkling2Tensor, classify(&l.c, &l.m, &l.inv, t, &d));
    try testing.expect(std.mem.indexOf(u8, d.text(), "wq_new") != null);
    t = l.inv.find("model.llm.layers.0.attn.wq_du.weight").?;
    t.shape[0] = 6016;
    try testing.expectError(error.UnexpectedInkling2Tensor, classify(&l.c, &l.m, &l.inv, t, &d));
    t = l.inv.find("model.llm.layers.40.mlp.experts.w13_weight_scale").?;
    t.name = "model.llm.layers.40.mlp.experts.w13_weight_amax"; // MXFP8 has no amax
    try testing.expectError(error.UnknownInkling2Tensor, classify(&l.c, &l.m, &l.inv, t, &d));
    // a global layer's tensor on a sliding layer
    t = l.inv.find("model.llm.layers.5.attn.k_sconv.static_w").?;
    t.name = "model.llm.layers.4.attn.k_sconv.static_w";
    try testing.expectError(error.UnknownInkling2Tensor, classify(&l.c, &l.m, &l.inv, t, &d));
    // the inventory without one listed tensor
    const saved = l.inv.tensors;
    defer l.inv.tensors = saved;
    var fewer: std.ArrayList(Tensor) = .empty;
    defer fewer.deinit(testing.allocator);
    for (saved) |x| if (!std.mem.eql(u8, x.name, "model.llm.layers.7.mlp_sconv.weight")) try fewer.append(testing.allocator, x);
    l.inv.tensors = fewer.items;
    try testing.expectError(error.MissingInkling2Tensor, classifyAll(testing.allocator, &l.c, &l.m, &l.inv, &d));
    try testing.expect(std.mem.indexOf(u8, d.text(), "layers.7.mlp_sconv") != null);
}

test "inkling2 roles: a vision or drafter tensor whose shape is not the config's refuses the checkpoint" {
    var l = try Loaded.init();
    defer l.deinit();
    const Case = struct { name: []const u8, axis: u8, to: u64 };
    for ([_]Case{
        .{ .name = "model.visual.proj_in.weight", .axis = 1, .to = 1024 },
        .{ .name = "model.visual.proj_out.weight", .axis = 0, .to = 4096 },
        .{ .name = "model.visual.backbone.embeddings.patch_embedding.weight", .axis = 1, .to = 2400 },
        .{ .name = "model.visual.backbone.embeddings.position_embedding", .axis = 0, .to = 1025 },
        .{ .name = "model.visual.backbone.embeddings.class_embedding", .axis = 0, .to = 1023 },
        .{ .name = "model.visual.backbone.ln_pre.bias", .axis = 0, .to = 4096 },
        .{ .name = "model.visual.backbone.layers.3.mlp.c_fc.weight", .axis = 0, .to = 4097 },
        .{ .name = "model.visual.backbone.layers.3.mlp.c_proj.weight", .axis = 1, .to = 1024 },
        .{ .name = "model.visual.backbone.layers.22.self_attn.q_proj.bias", .axis = 0, .to = 4096 },
        .{ .name = "model.visual.backbone.layers.0.ls_2.gamma", .axis = 0, .to = 4608 },
        .{ .name = "fuse_weight", .axis = 0, .to = 5 },
        .{ .name = "mask_embed", .axis = 0, .to = 1024 },
        .{ .name = "markov_head.markov_w1.weight", .axis = 0, .to = 200058 },
        .{ .name = "markov_head.rnn_proj.weight", .axis = 1, .to = 9216 },
        .{ .name = "markov_head.hidden_out.weight", .axis = 1, .to = 128 },
        .{ .name = "layers.2.self_attn.k_proj.weight", .axis = 0, .to = 6144 },
        .{ .name = "layers.4.self_attn.o_proj.weight", .axis = 1, .to = 4608 },
        .{ .name = "layers.0.self_attn.q_norm.weight", .axis = 0, .to = 4608 },
        .{ .name = "layers.1.mlp.down_proj.weight", .axis = 1, .to = 9216 },
    }) |case| {
        var t = l.inv.find(case.name).?;
        _ = try classify(&l.c, &l.m, &l.inv, t, null);
        t.shape[case.axis] = case.to;
        var d: Diag = .{};
        testing.expectError(error.UnexpectedInkling2Tensor, classify(&l.c, &l.m, &l.inv, t, &d)) catch |e| {
            std.debug.print("{s}: shape {any} passed\n", .{ case.name, t.shape[0..t.rank] });
            return e;
        };
        try testing.expect(std.mem.indexOf(u8, d.text(), case.name) != null);
    }
}
