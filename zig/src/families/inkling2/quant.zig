//! The release's quantization manifest (hf_quant_config.json, `quant_method: inkling2`): which tensors are stored in
//! which format, at which compute and output dtypes, and how the activations into them must be quantized.
//!
//! Each parameter group lists its tensors by name. A quantized weight's companions (its block scales, its per-expert
//! amax and its outer scale denominator) are not listed: `roles.zig` attaches them to their weight's group. A group
//! whose settings are not one of the four formats below is refused, as is a key the parser does not know.
const std = @import("std");
const Diag = @import("diag.zig").Diag;
const fail = @import("diag.zig").fail;
const Section = @import("inventory.zig").Section;

/// The storage formats the release uses.
pub const Format = enum {
    /// float32, cast: the attention residual's queries and the router's selection bias.
    fp32,
    /// bfloat16, cast: attention, router weights, norms, convolutions, embed, unembed, vision tower, drafter.
    bf16,
    /// NVFP4: packed E2M1 pairs, an E4M3 scale per 16 values of a row, an fp32 amax per expert and an outer scale
    /// denominator (weight_scale_format amax_denominator). Experts of layers 0-39.
    nvfp4,
    /// MXFP8: E4M3 values, a UE8M0 scale per 32 values of a row. Experts of layers 40-47.
    mxfp8,

    pub fn name(f: Format) []const u8 {
        return switch (f) {
            .fp32 => "FP32",
            .bf16 => "BF16",
            .nvfp4 => "NVFP4",
            .mxfp8 => "MXFP8",
        };
    }
};

/// How activations entering a quantized weight's matmul must be quantized (required_activation_quantization).
pub const ActQuant = struct {
    format: enum { nvfp4_e2m1, mxfp8_e4m3 },
    block: u32,
    scale: enum { e4m3, ue8m0 },
    /// amax_type per_token: the NVFP4 outer scale is chosen per token.
    per_token: bool = false,
    /// scale_selection_rule fouroversix_mae: each 16-block's scale is the better of two candidates (by mean
    /// absolute error), as TokenSpeed's numerics define it. The method is Four Over Six (mit-han-lab/fouroversix,
    /// arXiv:2512.02010).
    two_candidates: bool = false,
};

pub const Group = struct {
    format: Format,
    /// The matmul's output dtype as the manifest gives it (output_dtype; fwd_output_dtype when given, the unembed's
    /// float32 logits).
    output: []const u8,
    fwd_output: ?[]const u8 = null,
    act: ?ActQuant = null,
    /// weight_block_shape: [] for the cast formats, [1, 16] NVFP4, [32, 32] MXFP8 (the stored scales are one per
    /// row and 32 values; see roles.zig's shape checks).
    block: [2]u32 = .{ 0, 0 },
    tensors: []const []const u8,
};

pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    model: []Group,
    mtp: []Group,
    index: [2]std.StringHashMapUnmanaged(u16),

    pub fn deinit(m: *Manifest) void {
        m.arena.deinit();
        m.* = undefined;
    }

    pub fn groups(m: *const Manifest, s: Section) []const Group {
        return switch (s) {
            .model => m.model,
            .mtp => m.mtp,
        };
    }

    /// The group that lists `name` in section `s`, or null when the manifest does not list it.
    pub fn lookup(m: *const Manifest, s: Section, name: []const u8) ?*const Group {
        const i = m.index[@backingInt(s)].get(name) orelse return null;
        return &m.groups(s)[i];
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, d: ?*Diag) !Manifest {
        const path = try std.fs.path.join(gpa, &.{ dir, "hf_quant_config.json" });
        defer gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch |e|
            return fail(d, e, "cannot read hf_quant_config.json: {t}", .{e});
        defer gpa.free(bytes);
        return parse(gpa, bytes, d);
    }
};

const group_keys = [_][]const u8{ "compute_dtype", "output_dtype", "fwd_output_dtype", "required_activation_quantization", "tensors", "weight_block_shape", "weight_transform", "weight_scale_format" };

fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn eql(a: ?[]const u8, b: []const u8) bool {
    return if (a) |x| std.mem.eql(u8, x, b) else false;
}

fn actQuant(v: std.json.Value, d: ?*Diag) !?ActQuant {
    if (v == .null) return null;
    if (v != .object) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: required_activation_quantization is not an object", .{});
    const o = v.object;
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const known = for ([_][]const u8{ "amax_type", "block_size", "format", "rounding", "scale_dtype", "scale_selection_rule" }) |x| {
            if (std.mem.eql(u8, x, k)) break true;
        } else false;
        if (!known) return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: activation quantization key {s} is not known", .{k});
    }
    const block = o.get("block_size") orelse return fail(d, error.BadInkling2Quant, "hf_quant_config.json: activation quantization lacks block_size", .{});
    if (block != .integer) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: block_size is not an integer", .{});
    if (!eql(str(o, "rounding"), "nearest")) return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: activation rounding is not nearest", .{});
    const format = str(o, "format") orelse "";
    const scale = str(o, "scale_dtype") orelse "";
    if (std.mem.eql(u8, format, "nvfp4_e2m1") and block.integer == 16 and std.mem.eql(u8, scale, "float8_e4m3fn") and
        eql(str(o, "amax_type"), "per_token") and eql(str(o, "scale_selection_rule"), "fouroversix_mae"))
        return .{ .format = .nvfp4_e2m1, .block = 16, .scale = .e4m3, .per_token = true, .two_candidates = true };
    if (std.mem.eql(u8, format, "mxfp8_e4m3") and block.integer == 32 and std.mem.eql(u8, scale, "ue8m0") and
        o.get("amax_type") == null and o.get("scale_selection_rule") == null)
        return .{ .format = .mxfp8_e4m3, .block = 32, .scale = .ue8m0 };
    return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: activation quantization {s} at {d} with {s} scales is not built", .{ format, block.integer, scale });
}

fn blockShape(v: ?std.json.Value, d: ?*Diag) ![2]u32 {
    const x = v orelse return .{ 0, 0 };
    if (x != .array or (x.array.items.len != 0 and x.array.items.len != 2)) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: weight_block_shape is not [] or [rows, columns]", .{});
    if (x.array.items.len == 0) return .{ 0, 0 };
    var out: [2]u32 = undefined;
    for (x.array.items, &out) |e, *o| {
        if (e != .integer or e.integer <= 0 or e.integer > 1 << 16) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: weight_block_shape is not [rows, columns]", .{});
        o.* = @intCast(e.integer);
    }
    return out;
}

fn group(a: std.mem.Allocator, v: std.json.Value, d: ?*Diag) !Group {
    if (v != .object) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: a parameter group is not an object", .{});
    const o = v.object;
    var it = o.iterator();
    while (it.next()) |e| {
        const known = for (group_keys) |x| {
            if (std.mem.eql(u8, x, e.key_ptr.*)) break true;
        } else false;
        if (!known) return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: parameter group key {s} is not known", .{e.key_ptr.*});
    }
    const compute = str(o, "compute_dtype") orelse "";
    const transform = str(o, "weight_transform") orelse "";
    const output = str(o, "output_dtype") orelse return fail(d, error.BadInkling2Quant, "hf_quant_config.json: a group lacks output_dtype", .{});
    const act = try actQuant(o.get("required_activation_quantization") orelse .null, d);
    const block = try blockShape(o.get("weight_block_shape"), d);
    const list = o.get("tensors") orelse return fail(d, error.BadInkling2Quant, "hf_quant_config.json: a group lacks tensors", .{});
    if (list != .array) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: tensors is not a list", .{});
    const names = try a.alloc([]const u8, list.array.items.len);
    for (list.array.items, names) |x, *n| {
        if (x != .string) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: a tensor name is not a string", .{});
        n.* = x.string;
    }
    var g: Group = .{ .format = undefined, .output = output, .fwd_output = str(o, "fwd_output_dtype"), .act = act, .block = block, .tensors = names };
    const cast = std.mem.eql(u8, transform, "cast") and act == null and block[0] == 0 and o.get("weight_scale_format") == null;
    if (cast and std.mem.eql(u8, compute, "float32") and std.mem.eql(u8, output, "float32")) {
        g.format = .fp32;
    } else if (cast and std.mem.eql(u8, compute, "bfloat16") and std.mem.eql(u8, output, "bfloat16") and
        (g.fwd_output == null or eql(g.fwd_output, "float32")))
    {
        g.format = .bf16;
    } else if (std.mem.eql(u8, compute, "nvfp4") and std.mem.eql(u8, transform, "nvfp4_nearest") and std.mem.eql(u8, output, "uint8") and
        block[0] == 1 and block[1] == 16 and eql(str(o, "weight_scale_format"), "amax_denominator") and act != null and act.?.format == .nvfp4_e2m1)
    {
        g.format = .nvfp4;
    } else if (std.mem.eql(u8, compute, "mxfp8") and std.mem.eql(u8, transform, "mxfp8_nearest") and std.mem.eql(u8, output, "float8_e4m3fn") and
        block[0] == 32 and block[1] == 32 and o.get("weight_scale_format") == null and act != null and act.?.format == .mxfp8_e4m3)
    {
        g.format = .mxfp8;
    } else return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: a group computing {s} with transform {s} is not one of FP32, BF16, NVFP4 or MXFP8", .{ compute, transform });
    return g;
}

fn groups(a: std.mem.Allocator, root: std.json.ObjectMap, key: []const u8, index: *std.StringHashMapUnmanaged(u16), d: ?*Diag) ![]Group {
    const s = root.get(key) orelse return fail(d, error.BadInkling2Quant, "hf_quant_config.json lacks {s}", .{key});
    if (s != .object) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: {s} is not an object", .{key});
    var it = s.object.iterator();
    while (it.next()) |e| if (!std.mem.eql(u8, e.key_ptr.*, "parameter_groups"))
        return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: {s}.{s} is not known", .{ key, e.key_ptr.* });
    const list = s.object.get("parameter_groups") orelse return fail(d, error.BadInkling2Quant, "hf_quant_config.json: {s} lacks parameter_groups", .{key});
    if (list != .array) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: {s}.parameter_groups is not a list", .{key});
    const out = try a.alloc(Group, list.array.items.len);
    for (list.array.items, out, 0..) |v, *g, i| {
        g.* = try group(a, v, d);
        for (g.tensors) |n| {
            const slot = try index.getOrPut(a, n);
            if (slot.found_existing) return fail(d, error.BadInkling2Quant, "hf_quant_config.json: {s} is in two groups", .{n});
            slot.value_ptr.* = @intCast(i);
        }
    }
    return out;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, d: ?*Diag) !Manifest {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch |e|
        return fail(d, e, "hf_quant_config.json is not JSON: {t}", .{e});
    if (doc != .object) return fail(d, error.BadInkling2Quant, "hf_quant_config.json is not an object", .{});
    const method = str(doc.object, "quant_method") orelse "";
    if (!std.mem.eql(u8, method, "inkling2")) return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: quant_method is {s}, not inkling2", .{method});
    var it = doc.object.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "quant_method") and !std.mem.eql(u8, k, "model") and !std.mem.eql(u8, k, "mtp"))
            return fail(d, error.UnsupportedInkling2Quant, "hf_quant_config.json: {s} is not known", .{k});
    }
    var m: Manifest = .{ .arena = undefined, .model = &.{}, .mtp = &.{}, .index = .{ .empty, .empty } };
    m.model = try groups(a, doc.object, "model", &m.index[0], d);
    m.mtp = if (doc.object.get("mtp") != null) try groups(a, doc.object, "mtp", &m.index[1], d) else &.{};
    m.arena = arena;
    return m;
}

const testing = std.testing;
pub const fixture = @embedFile("fixtures/hf_quant_config.json");

test "inkling2 quant: the release's manifest, five model groups and the drafter's" {
    var m = try parse(testing.allocator, fixture, null);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 5), m.model.len);
    try testing.expectEqual(@as(usize, 1), m.mtp.len);
    var listed: [4]usize = @splat(0);
    for (m.model) |g| listed[@backingInt(g.format)] += g.tensors.len;
    try testing.expectEqualSlices(usize, &.{ 144, 1273 + 1, 80, 16 }, &listed);
    try testing.expectEqual(@as(usize, 84), m.mtp[0].tensors.len);
    const w13 = m.lookup(.model, "model.llm.layers.0.mlp.experts.w13_weight").?;
    try testing.expectEqual(Format.nvfp4, w13.format);
    try testing.expect(w13.act.?.per_token and w13.act.?.two_candidates and w13.act.?.block == 16);
    const mx = m.lookup(.model, "model.llm.layers.40.mlp.experts.w2_weight").?;
    try testing.expectEqual(Format.mxfp8, mx.format);
    try testing.expectEqual(@as(u32, 32), mx.act.?.block);
    try testing.expectEqual([2]u32{ 32, 32 }, mx.block);
    const un = m.lookup(.model, "model.llm.unembed.weight").?;
    try testing.expectEqual(Format.bf16, un.format);
    try testing.expectEqualStrings("float32", un.fwd_output.?);
    try testing.expectEqual(Format.fp32, m.lookup(.model, "model.llm.layers.3.mlp.gate.bias").?.format);
    try testing.expectEqual(Format.bf16, m.lookup(.mtp, "markov_head.markov_w1.weight").?.format);
    try testing.expect(m.lookup(.model, "model.llm.layers.0.mlp.experts.w13_weight_scale") == null); // a companion
    try testing.expect(m.lookup(.mtp, "model.llm.embed.weight") == null);
}

test "inkling2 quant: another method, an unknown setting or a tensor in two groups is refused" {
    const Case = struct { from: []const u8, to: []const u8, err: anyerror };
    const cases = [_]Case{
        .{ .from = "\"quant_method\": \"inkling2\"", .to = "\"quant_method\": \"modelopt\"", .err = error.UnsupportedInkling2Quant },
        .{ .from = "\"scale_selection_rule\": \"fouroversix_mae\"", .to = "\"scale_selection_rule\": \"max\"", .err = error.UnsupportedInkling2Quant },
        .{ .from = "\"weight_scale_format\": \"amax_denominator\"", .to = "\"weight_scale_format\": \"plain\"", .err = error.UnsupportedInkling2Quant },
        .{ .from = "\"weight_transform\": \"mxfp8_nearest\"", .to = "\"weight_transform\": \"mxfp8_stochastic\"", .err = error.UnsupportedInkling2Quant },
        .{ .from = "\"output_dtype\": \"uint8\",", .to = "\"output_dtype\": \"uint8\", \"zero_point\": 0,", .err = error.UnsupportedInkling2Quant },
        .{ .from = "\"model.llm.unembed.weight\"", .to = "\"model.llm.embed.weight\"", .err = error.BadInkling2Quant },
    };
    for (cases) |case| {
        const changed = try std.mem.replaceOwned(u8, testing.allocator, fixture, case.from, case.to);
        defer testing.allocator.free(changed);
        try testing.expect(!std.mem.eql(u8, changed, fixture));
        var d: Diag = .{};
        try testing.expectError(case.err, parse(testing.allocator, changed, &d));
        try testing.expect(d.len > 0);
    }
}
