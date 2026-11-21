//! Inkling-2-Small's shape from the release's config.json: the text model, the vision tower and the DSpark drafter.
//!
//! Every key the release's file holds is read. A key the file leaves out takes the default the reference code uses
//! (Transformers' `configuration_inkling2.py`, Apache-2.0, Copyright 2026 the HuggingFace Team); keys Transformers
//! does not define (the drafter, the arithmetic flags) default to the release's values. Keys this parser does not know
//! are counted in `unknown` and named by `describe`, so a new revision's additions are seen rather than dropped.
const std = @import("std");
const Diag = @import("diag.zig").Diag;
const fail = @import("diag.zig").fail;

pub const model_type = "inkling2";
pub const architecture = "Inkling2ForConditionalGeneration";
pub const max_layers = 64;
pub const max_draft_layers = 16;
pub const max_targets = 16;

pub const Text = struct {
    /// model_max_length (Transformers: max_position_embeddings).
    max_positions: u32 = 1048576,
    hidden: u32 = 4608,
    layers: u32 = 48,
    vocab: u32 = 201024,
    /// Rows of the unembedding that hold real tokens; logits past it are dropped.
    unpadded_vocab: u32 = 200058,
    /// Global (full-attention) layers' heads; the sliding layers' are `swa_*`.
    heads: u32 = 48,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    swa_heads: u32 = 48,
    swa_kv_heads: u32 = 8,
    swa_head_dim: u32 = 128,
    window: u32 = 1024,
    /// Per-head width of the relative states, and the distances the relative bias covers in global layers.
    d_rel: u32 = 16,
    rel_extent: u32 = 1024,
    q_bias: bool = false,
    o_bias: bool = false,
    log_scaling_n_floor: u32 = 128000,
    log_scaling_alpha: f64 = 0.1,
    eps: f64 = 1e-6,
    pre_norm_pre_affine: bool = false,
    final_norm_pre_affine: bool = false,
    value_norm_no_gamma: bool = true,
    qk_clip: f64 = 22.627416997969522,
    rel_clip: f64 = 8.0,
    /// Taps of each short convolution (Transformers: conv_kernel_size).
    conv_taps: u32 = 4,
    logit_softcap: f64 = 30.0,
    res_block: u32 = 6,
    res_normalize: bool = true,
    experts: u32 = 384,
    topk: u32 = 8,
    shared_experts: u32 = 0,
    moe_intermediate: u32 = 2304,
    gate_softcap: f64 = 4.0,
    up_softcap: f64 = 25.0,
    route_scale: f64 = 8.0,
    gate_bias: bool = true,
    /// Not in the release's file: Transformers' defaults.
    scoring_sigmoid: bool = true,
    /// Global layers: Transformers' default puts one at every sixth layer, `(i + 1) % 6 == 0`; the file's
    /// `local_layer_ids` names the sliding ones.
    global: [max_layers]bool = defaultGlobal(),

    fn defaultGlobal() [max_layers]bool {
        var g: [max_layers]bool = @splat(false);
        for (0..max_layers) |i| g[i] = (i + 1) % 6 == 0;
        return g;
    }

    pub fn globalCount(t: *const Text) u32 {
        var n: u32 = 0;
        for (t.global[0..t.layers]) |g| n += @intFromBool(g);
        return n;
    }

    /// Layers whose output joins the block residual's sources (Transformers: layer_idx % 6 == 0).
    pub fn blockBoundary(t: *const Text, layer: u32) bool {
        return layer % t.res_block == 0;
    }
};

pub const Vision = struct {
    temporal_patch: u32 = 2,
    patch: u32 = 20,
    merge: u32 = 2,
    projection_hidden: u32 = 4096,
    projector_bias: bool = false,
    hidden: u32 = 1024,
    layers: u32 = 23,
    heads: u32 = 16,
    mlp_ratio: f64 = 4.0,
    posemb_grid: u32 = 32,
    eps: f64 = 1e-5,

    /// The MLP's width, hidden_size * mlp_ratio (Transformers' intermediate_size), which `validate` holds to a
    /// whole number.
    pub fn intermediate(v: *const Vision) u64 {
        return @intFromFloat(@as(f64, @floatFromInt(v.hidden)) * v.mlp_ratio);
    }
};

pub const Draft = struct {
    hidden: u32 = 4608,
    layers: u32 = 5,
    heads: u32 = 48,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    intermediate: u32 = 18432,
    max_positions: u32 = 1048576,
    q_bias: bool = false,
    o_bias: bool = false,
    eps: f64 = 1e-6,
    rope_theta: f64 = 1e6,
    /// Sliding layers among the drafter's; the others attend to the whole context.
    local: [max_draft_layers]bool = .{ true, true, false, true, true } ++ @as([max_draft_layers - 5]bool, @splat(false)),
    window: u32 = 1024,
    block: u32 = 8,
    markov_rank: u32 = 256,
    markov_rnn_hidden: bool = true,
    /// Target layers whose outputs the drafter reads (48: the last layer's output after the final residual).
    targets: [max_targets]u32 = .{ 5, 11, 17, 23, 29, 35, 41, 48 } ++ @as([max_targets - 8]u32, @splat(0)),
    target_count: u32 = 8,
};

pub const Config = struct {
    eos: u32 = 200006,
    image_token: u32 = 200054,
    text: Text = .{},
    vision: Vision = .{},
    draft: Draft = .{},
    has_vision: bool = true,
    has_draft: bool = true,
    /// Keys the parser does not know, over every section.
    unknown: u32 = 0,
    first_unknown: [64]u8 = undefined,
    first_unknown_len: u8 = 0,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, d: ?*Diag) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch |e|
            return fail(d, e, "cannot read config.json: {t}", .{e});
        defer gpa.free(bytes);
        return parse(gpa, bytes, d);
    }

    pub fn firstUnknown(c: *const Config) []const u8 {
        return c.first_unknown[0..c.first_unknown_len];
    }

    fn noteUnknown(c: *Config, where: []const u8, key: []const u8) void {
        if (c.unknown == 0) {
            const s = std.fmt.bufPrint(&c.first_unknown, "{s}.{s}", .{ where, key }) catch c.first_unknown[0..];
            c.first_unknown_len = @intCast(s.len);
        }
        c.unknown += 1;
    }

    /// Query heads of global layer attention per key head.
    pub fn group(c: *const Config) u32 {
        return c.text.heads / c.text.kv_heads;
    }
};

const Obj = std.json.ObjectMap;

const Reader = struct {
    o: Obj,
    section: []const u8,
    d: ?*Diag,

    fn bad(r: Reader, key: []const u8) error{BadInkling2Config} {
        return fail(r.d, error.BadInkling2Config, "config.json: {s}.{s} has the wrong type or range", .{ r.section, key });
    }

    fn uint(r: Reader, key: []const u8, out: *u32) !void {
        const v = r.o.get(key) orelse return;
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return r.bad(key);
        out.* = @intCast(v.integer);
    }

    fn float(r: Reader, key: []const u8, out: *f64) !void {
        const v = r.o.get(key) orelse return;
        out.* = switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return r.bad(key),
        };
    }

    fn flag(r: Reader, key: []const u8, out: *bool) !void {
        const v = r.o.get(key) orelse return;
        if (v != .bool) return r.bad(key);
        out.* = v.bool;
    }

    fn string(r: Reader, key: []const u8) !?[]const u8 {
        const v = r.o.get(key) orelse return null;
        if (v != .string) return r.bad(key);
        return v.string;
    }

    fn ids(r: Reader, key: []const u8, out: []u32) !?usize {
        const v = r.o.get(key) orelse return null;
        if (v != .array or v.array.items.len > out.len) return r.bad(key);
        for (v.array.items, 0..) |x, i| {
            if (x != .integer or x.integer < 0 or x.integer > std.math.maxInt(u32)) return r.bad(key);
            out[i] = @intCast(x.integer);
        }
        return v.array.items.len;
    }

    fn unknown(r: Reader, c: *Config, known: []const []const u8) void {
        var it = r.o.iterator();
        outer: while (it.next()) |e| {
            for (known) |k| if (std.mem.eql(u8, k, e.key_ptr.*)) continue :outer;
            c.noteUnknown(r.section, e.key_ptr.*);
        }
    }
};

const root_keys = [_][]const u8{ "architectures", "model_type", "eos_token_id", "image_token_id", "text_config", "vision_config", "mtp_config", "transformers_version", "torch_dtype", "dtype" };
const text_keys = [_][]const u8{
    "model_max_length",        "max_position_embeddings", "torch_dtype",                 "dtype",                     "hidden_size",
    "num_hidden_layers",       "vocab_size",              "num_attention_heads",         "num_key_value_heads",       "head_dim",
    "d_rel",                   "rel_extent",              "q_bias",                      "o_bias",                    "log_scaling_n_floor",
    "log_scaling_alpha",       "rms_norm_eps",            "pre_norm_pre_affine",         "final_norm_pre_affine",     "use_value_norm_no_gamma",
    "qk_l2_clip_threshold",    "rel_l2_clip_threshold",   "local_layer_ids",             "layer_types",               "sconv_kernel_size",
    "conv_kernel_size",        "unpadded_vocab_size",     "final_logit_softcapping",     "swa_head_dim",              "swa_num_attention_heads",
    "swa_num_key_value_heads", "sliding_window_size",     "n_layers_per_attn_res_block", "attn_res_normalize_values", "n_routed_experts",
    "num_experts_per_tok",     "n_shared_experts",        "intermediate_size",           "moe_intermediate_size",     "mlp_gate_softcap",
    "mlp_up_softcap",          "route_scale",             "use_gate_bias",               "scoring_func",              "hidden_act",
};
const vision_keys = [_][]const u8{ "temporal_patch_size", "native_patch_size", "patch_size", "adapter_merge_size", "spatial_merge_size", "projection_hidden_dim", "projector_bias", "hidden_size", "num_hidden_layers", "num_attention_heads", "mlp_ratio", "posemb_grid_size", "layer_norm_eps" };
const draft_keys = [_][]const u8{ "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "max_position_embeddings", "torch_dtype", "dtype", "q_bias", "o_bias", "rms_norm_eps", "rope_theta", "local_layer_ids", "sliding_window_size", "block_size", "markov_rank", "markov_head_type", "target_layer_ids" };

fn section(root: Obj, key: []const u8, d: ?*Diag) !?Reader {
    const v = root.get(key) orelse return null;
    if (v != .object) return fail(d, error.BadInkling2Config, "config.json: {s} is not an object", .{key});
    return .{ .o = v.object, .section = key, .d = d };
}

fn dtypeIsBf16(r: Reader) !void {
    for ([_][]const u8{ "torch_dtype", "dtype" }) |k| if (try r.string(k)) |s| {
        if (!std.mem.eql(u8, s, "bfloat16")) return fail(r.d, error.UnsupportedInkling2Config, "config.json: {s}.{s} is {s}, not bfloat16", .{ r.section, k, s });
    };
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, d: ?*Diag) !Config {
    var doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e|
        return fail(d, e, "config.json is not JSON: {t}", .{e});
    defer doc.deinit();
    if (doc.value != .object) return fail(d, error.BadInkling2Config, "config.json is not an object", .{});
    const root: Reader = .{ .o = doc.value.object, .section = "config", .d = d };
    var c: Config = .{};
    const mt = (try root.string("model_type")) orelse "";
    if (!std.mem.eql(u8, mt, model_type)) return fail(d, error.UnsupportedInkling2Config, "config.json: model_type is {s}, not inkling2", .{mt});
    if (root.o.get("architectures")) |v| {
        if (v != .array) return root.bad("architectures");
        var found = false;
        for (v.array.items) |x| {
            if (x != .string) return root.bad("architectures");
            found = found or std.mem.eql(u8, x.string, architecture);
        }
        if (!found) return fail(d, error.UnsupportedInkling2Config, "config.json: architectures does not name " ++ architecture, .{});
    }
    try root.uint("eos_token_id", &c.eos);
    try root.uint("image_token_id", &c.image_token);
    root.unknown(&c, &root_keys);
    if (try section(root.o, "text_config", d)) |r| try readText(&c, r) else return fail(d, error.BadInkling2Config, "config.json lacks text_config", .{});
    if (try section(root.o, "vision_config", d)) |r| try readVision(&c, r) else c.has_vision = false;
    if (try section(root.o, "mtp_config", d)) |r| try readDraft(&c, r) else c.has_draft = false;
    try validate(&c, d);
    return c;
}

fn readText(c: *Config, r: Reader) !void {
    const t = &c.text;
    try dtypeIsBf16(r);
    try r.uint("max_position_embeddings", &t.max_positions);
    try r.uint("model_max_length", &t.max_positions);
    try r.uint("hidden_size", &t.hidden);
    try r.uint("num_hidden_layers", &t.layers);
    try r.uint("vocab_size", &t.vocab);
    try r.uint("unpadded_vocab_size", &t.unpadded_vocab);
    try r.uint("num_attention_heads", &t.heads);
    try r.uint("num_key_value_heads", &t.kv_heads);
    try r.uint("head_dim", &t.head_dim);
    try r.uint("swa_num_attention_heads", &t.swa_heads);
    try r.uint("swa_num_key_value_heads", &t.swa_kv_heads);
    try r.uint("swa_head_dim", &t.swa_head_dim);
    try r.uint("sliding_window_size", &t.window);
    try r.uint("d_rel", &t.d_rel);
    try r.uint("rel_extent", &t.rel_extent);
    try r.flag("q_bias", &t.q_bias);
    try r.flag("o_bias", &t.o_bias);
    try r.uint("log_scaling_n_floor", &t.log_scaling_n_floor);
    try r.float("log_scaling_alpha", &t.log_scaling_alpha);
    try r.float("rms_norm_eps", &t.eps);
    try r.flag("pre_norm_pre_affine", &t.pre_norm_pre_affine);
    try r.flag("final_norm_pre_affine", &t.final_norm_pre_affine);
    try r.flag("use_value_norm_no_gamma", &t.value_norm_no_gamma);
    try r.float("qk_l2_clip_threshold", &t.qk_clip);
    try r.float("rel_l2_clip_threshold", &t.rel_clip);
    try r.uint("conv_kernel_size", &t.conv_taps);
    try r.uint("sconv_kernel_size", &t.conv_taps);
    try r.float("final_logit_softcapping", &t.logit_softcap);
    try r.uint("n_layers_per_attn_res_block", &t.res_block);
    try r.flag("attn_res_normalize_values", &t.res_normalize);
    try r.uint("n_routed_experts", &t.experts);
    try r.uint("num_experts_per_tok", &t.topk);
    try r.uint("n_shared_experts", &t.shared_experts);
    try r.uint("moe_intermediate_size", &t.moe_intermediate);
    try r.uint("intermediate_size", &t.moe_intermediate);
    try r.float("mlp_gate_softcap", &t.gate_softcap);
    try r.float("mlp_up_softcap", &t.up_softcap);
    try r.float("route_scale", &t.route_scale);
    try r.flag("use_gate_bias", &t.gate_bias);
    if (try r.string("scoring_func")) |s| t.scoring_sigmoid = std.mem.eql(u8, s, "sigmoid");
    if (try r.string("hidden_act")) |s| if (!std.mem.eql(u8, s, "silu"))
        return fail(r.d, error.UnsupportedInkling2Config, "config.json: text_config.hidden_act is {s}, not silu", .{s});
    if (t.layers == 0 or t.layers > max_layers) return r.bad("num_hidden_layers");
    var local: [max_layers]u32 = undefined;
    if (try r.ids("local_layer_ids", &local)) |n| {
        t.global = @splat(true);
        for (local[0..n]) |l| {
            if (l >= t.layers or !t.global[l]) return r.bad("local_layer_ids");
            t.global[l] = false;
        }
    } else if (r.o.get("layer_types")) |v| {
        if (v != .array or v.array.items.len != t.layers) return r.bad("layer_types");
        for (v.array.items, 0..) |x, i| {
            if (x != .string) return r.bad("layer_types");
            if (std.mem.eql(u8, x.string, "hybrid")) t.global[i] = true else if (std.mem.eql(u8, x.string, "hybrid_sliding")) t.global[i] = false else return r.bad("layer_types");
        }
    }
    for (t.layers..max_layers) |i| t.global[i] = false;
    r.unknown(c, &text_keys);
}

fn readVision(c: *Config, r: Reader) !void {
    const v = &c.vision;
    try r.uint("temporal_patch_size", &v.temporal_patch);
    try r.uint("patch_size", &v.patch);
    try r.uint("native_patch_size", &v.patch);
    try r.uint("spatial_merge_size", &v.merge);
    try r.uint("adapter_merge_size", &v.merge);
    try r.uint("projection_hidden_dim", &v.projection_hidden);
    try r.flag("projector_bias", &v.projector_bias);
    try r.uint("hidden_size", &v.hidden);
    try r.uint("num_hidden_layers", &v.layers);
    try r.uint("num_attention_heads", &v.heads);
    try r.float("mlp_ratio", &v.mlp_ratio);
    try r.uint("posemb_grid_size", &v.posemb_grid);
    try r.float("layer_norm_eps", &v.eps);
    r.unknown(c, &vision_keys);
}

fn readDraft(c: *Config, r: Reader) !void {
    const m = &c.draft;
    try dtypeIsBf16(r);
    try r.uint("hidden_size", &m.hidden);
    try r.uint("num_hidden_layers", &m.layers);
    try r.uint("num_attention_heads", &m.heads);
    try r.uint("num_key_value_heads", &m.kv_heads);
    try r.uint("head_dim", &m.head_dim);
    try r.uint("intermediate_size", &m.intermediate);
    try r.uint("max_position_embeddings", &m.max_positions);
    try r.flag("q_bias", &m.q_bias);
    try r.flag("o_bias", &m.o_bias);
    try r.float("rms_norm_eps", &m.eps);
    try r.float("rope_theta", &m.rope_theta);
    try r.uint("sliding_window_size", &m.window);
    try r.uint("block_size", &m.block);
    try r.uint("markov_rank", &m.markov_rank);
    if (try r.string("markov_head_type")) |s| m.markov_rnn_hidden = std.mem.eql(u8, s, "rnn_hidden");
    if (m.layers == 0 or m.layers > max_draft_layers) return r.bad("num_hidden_layers");
    var ids: [max_draft_layers]u32 = undefined;
    if (try r.ids("local_layer_ids", &ids)) |n| {
        m.local = @splat(false);
        for (ids[0..n]) |l| {
            if (l >= m.layers) return r.bad("local_layer_ids");
            m.local[l] = true;
        }
    }
    if (try r.ids("target_layer_ids", &m.targets)) |n| {
        m.target_count = @intCast(n);
        for (m.targets[n..]) |*x| x.* = 0;
    }
    r.unknown(c, &draft_keys);
}

/// The arithmetic and shape the family is built for. A config outside it is refused by name, not misread.
pub fn validate(c: *const Config, d: ?*Diag) !void {
    const t = &c.text;
    const Unsupported = error.UnsupportedInkling2Config;
    if (t.q_bias or t.o_bias) return fail(d, Unsupported, "config.json: attention biases (q_bias, o_bias) are not built", .{});
    if (t.pre_norm_pre_affine or t.final_norm_pre_affine) return fail(d, Unsupported, "config.json: pre-affine norms are not built", .{});
    if (!t.value_norm_no_gamma) return fail(d, Unsupported, "config.json: a value norm with a gain is not built", .{});
    if (!t.res_normalize) return fail(d, Unsupported, "config.json: unnormalized attention-residual values are not built", .{});
    if (!t.gate_bias or !t.scoring_sigmoid) return fail(d, Unsupported, "config.json: the router must be sigmoid scores with a selection bias", .{});
    if (t.shared_experts != 0) return fail(d, Unsupported, "config.json: shared experts are not built", .{});
    if (t.heads != t.swa_heads or t.kv_heads != t.swa_kv_heads or t.head_dim != t.swa_head_dim)
        return fail(d, Unsupported, "config.json: sliding and global layers must share their head layout", .{});
    if (t.kv_heads == 0 or t.heads % t.kv_heads != 0 or t.head_dim == 0 or t.hidden % 32 != 0)
        return fail(d, error.BadInkling2Config, "config.json: {d} query heads over {d} key heads", .{ t.heads, t.kv_heads });
    if (t.experts == 0 or t.topk == 0 or t.topk > t.experts or t.moe_intermediate % 32 != 0)
        return fail(d, error.BadInkling2Config, "config.json: {d} experts, {d} a token", .{ t.experts, t.topk });
    if (t.conv_taps < 2 or t.res_block == 0 or t.unpadded_vocab > t.vocab or t.window == 0)
        return fail(d, error.BadInkling2Config, "config.json: convolution taps, block size, vocabulary or window out of range", .{});
    // The attention kernels hold these as constants (inkling2_attn.cu: tau(p) = 1 + 0.1 ln(max(1, (p + 1) / 128000)) in
    // the global layers, a 1024-position sliding window and relative-bias extent; cuda_attn.zig window, rel_extent):
    // another value would be misread, not served.
    if (t.log_scaling_n_floor != 128000 or t.log_scaling_alpha != 0.1 or t.window != 1024 or t.rel_extent != 1024)
        return fail(d, Unsupported, "config.json: log scaling n_floor {d} alpha {d}, sliding window {d}, rel_extent {d}: the kernels are built for 128000, 0.1, 1024 and 1024", .{ t.log_scaling_n_floor, t.log_scaling_alpha, t.window, t.rel_extent });
    if (c.has_vision) {
        const w = @as(f64, @floatFromInt(c.vision.hidden)) * c.vision.mlp_ratio;
        if (!(w >= 1 and w <= std.math.maxInt(u32)) or w != @floor(w))
            return fail(d, error.BadInkling2Config, "config.json: vision_config.mlp_ratio {d} does not give a whole MLP width", .{c.vision.mlp_ratio});
    }
    if (c.has_draft) {
        const m = &c.draft;
        if (m.hidden != t.hidden or m.q_bias or m.o_bias or m.kv_heads == 0 or m.heads % m.kv_heads != 0)
            return fail(d, Unsupported, "config.json: the drafter's width or heads do not fit the target", .{});
        // cuda_dspark.zig window: the drafter's local layers read 1024 positions before the anchor
        if (m.window != 1024) return fail(d, Unsupported, "config.json: mtp_config.sliding_window_size {d}: the drafter is built for 1024", .{m.window});
        for (m.targets[0..m.target_count]) |l| if (l > t.layers)
            return fail(d, error.BadInkling2Config, "config.json: mtp_config.target_layer_ids names layer {d}", .{l});
    }
}

const testing = std.testing;
pub const fixture = @embedFile("fixtures/config.json");

test "inkling2 config: the release's file, every key read, none unknown" {
    const c = try parse(testing.allocator, fixture, null);
    const t = c.text;
    try testing.expectEqual(@as(u32, 0), c.unknown);
    try testing.expectEqual(@as(u32, 200006), c.eos);
    try testing.expectEqual(@as(u32, 200054), c.image_token); // not in the file: Transformers' default
    try testing.expectEqual(@as(u32, 48), t.layers);
    try testing.expectEqual(@as(u32, 4608), t.hidden);
    try testing.expectEqual(@as(u32, 201024), t.vocab);
    try testing.expectEqual(@as(u32, 200058), t.unpadded_vocab);
    try testing.expectEqual(@as(u32, 1048576), t.max_positions);
    try testing.expectEqual(@as(u32, 6), c.group());
    try testing.expectEqual(@as(u32, 8), t.globalCount());
    for (0..48) |i| try testing.expectEqual((i + 1) % 6 == 0, t.global[i]);
    try testing.expectEqual(@as(f64, 22.627416997969522), t.qk_clip);
    try testing.expectEqual(@as(f64, 8.0), t.rel_clip);
    try testing.expectEqual(@as(f64, 1e-6), t.eps);
    try testing.expectEqual(@as(f64, 0.1), t.log_scaling_alpha);
    try testing.expectEqual(@as(u32, 128000), t.log_scaling_n_floor);
    try testing.expectEqual(@as(u32, 1024), t.window);
    try testing.expectEqual(@as(u32, 16), t.d_rel);
    try testing.expectEqual(@as(u32, 4), t.conv_taps);
    try testing.expectEqual(@as(u32, 384), t.experts);
    try testing.expectEqual(@as(u32, 8), t.topk);
    try testing.expectEqual(@as(u32, 2304), t.moe_intermediate);
    try testing.expectEqual(@as(f64, 4.0), t.gate_softcap);
    try testing.expectEqual(@as(f64, 25.0), t.up_softcap);
    try testing.expectEqual(@as(f64, 8.0), t.route_scale);
    try testing.expectEqual(@as(f64, 30.0), t.logit_softcap);
    try testing.expect(t.blockBoundary(0) and t.blockBoundary(6) and !t.blockBoundary(5));
    try testing.expectEqual(@as(u32, 23), c.vision.layers);
    try testing.expectEqual(@as(u32, 20), c.vision.patch);
    try testing.expectEqual(@as(u32, 2), c.vision.merge);
    try testing.expectEqual(@as(u32, 2), c.vision.temporal_patch);
    try testing.expectEqual(@as(f64, 1e-5), c.vision.eps);
    const m = c.draft;
    try testing.expectEqual(@as(u32, 5), m.layers);
    try testing.expectEqual(@as(u32, 18432), m.intermediate);
    try testing.expectEqual(@as(u32, 8), m.block);
    try testing.expectEqual(@as(u32, 256), m.markov_rank);
    try testing.expect(m.markov_rnn_hidden);
    try testing.expectEqualSlices(bool, &.{ true, true, false, true, true }, m.local[0..5]);
    try testing.expectEqualSlices(u32, &.{ 5, 11, 17, 23, 29, 35, 41, 48 }, m.targets[0..m.target_count]);
}

test "inkling2 config: the defaults equal the release's values" {
    const parsed = try parse(testing.allocator, fixture, null);
    const minimal = try parse(testing.allocator,
        \\{"model_type": "inkling2", "text_config": {}, "vision_config": {}, "mtp_config": {}}
    , null);
    try testing.expectEqualDeep(parsed.text, minimal.text);
    try testing.expectEqualDeep(parsed.vision, minimal.vision);
    try testing.expectEqualDeep(parsed.draft, minimal.draft);
}

test "inkling2 config: another model, a changed arithmetic flag or an unknown key" {
    const Case = struct { from: []const u8, to: []const u8, err: anyerror, says: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"model_type\": \"inkling2\"", .to = "\"model_type\": \"inkling\"", .err = error.UnsupportedInkling2Config, .says = "not inkling2" },
        .{ .from = "\"Inkling2ForConditionalGeneration\"", .to = "\"InklingForCausalLM\"", .err = error.UnsupportedInkling2Config, .says = "architectures" },
        .{ .from = "\"q_bias\": false,\n    \"o_bias\"", .to = "\"q_bias\": true,\n    \"o_bias\"", .err = error.UnsupportedInkling2Config, .says = "biases" },
        .{ .from = "\"n_shared_experts\": 0", .to = "\"n_shared_experts\": 1", .err = error.UnsupportedInkling2Config, .says = "shared experts" },
        .{ .from = "\"use_gate_bias\": true", .to = "\"use_gate_bias\": false", .err = error.UnsupportedInkling2Config, .says = "router" },
        .{ .from = "\"swa_num_key_value_heads\": 8", .to = "\"swa_num_key_value_heads\": 4", .err = error.UnsupportedInkling2Config, .says = "head layout" },
        .{ .from = "\"hidden_size\": 4608,\n    \"num_hidden_layers\": 48", .to = "\"hidden_size\": \"4608\",\n    \"num_hidden_layers\": 48", .err = error.BadInkling2Config, .says = "text_config.hidden_size" },
        .{ .from = "      48\n    ]", .to = "      49\n    ]", .err = error.BadInkling2Config, .says = "target_layer_ids" },
        .{ .from = "\"mlp_ratio\": 4.0", .to = "\"mlp_ratio\": 4.0001", .err = error.BadInkling2Config, .says = "mlp_ratio" },
        .{ .from = "\"log_scaling_n_floor\": 128000", .to = "\"log_scaling_n_floor\": 64000", .err = error.UnsupportedInkling2Config, .says = "log scaling" },
        .{ .from = "\"log_scaling_alpha\": 0.1", .to = "\"log_scaling_alpha\": 0.2", .err = error.UnsupportedInkling2Config, .says = "log scaling" },
        .{ .from = "\"rel_extent\": 1024", .to = "\"rel_extent\": 2048", .err = error.UnsupportedInkling2Config, .says = "rel_extent" },
    };
    for (cases) |case| {
        const changed = try std.mem.replaceOwned(u8, testing.allocator, fixture, case.from, case.to);
        defer testing.allocator.free(changed);
        try testing.expect(!std.mem.eql(u8, changed, fixture));
        var d: Diag = .{};
        try testing.expectError(case.err, parse(testing.allocator, changed, &d));
        try testing.expect(std.mem.indexOf(u8, d.text(), case.says) != null);
    }
    const extra = try std.mem.replaceOwned(u8, testing.allocator, fixture, "\"d_rel\": 16,", "\"d_rel\": 16, \"new_knob\": 1,");
    defer testing.allocator.free(extra);
    const c = try parse(testing.allocator, extra, null);
    try testing.expectEqual(@as(u32, 1), c.unknown);
    try testing.expectEqualStrings("text_config.new_knob", c.firstUnknown());
}
