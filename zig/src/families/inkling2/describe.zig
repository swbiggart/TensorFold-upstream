//! The checkpoint read as one plan: config, manifest, inventory, roles and a rank split checked together, and the
//! per-rank memory table and byte ranges printed from it (tf-inkling2-plan, and the native host's refusal).
const std = @import("std");
const config = @import("config.zig");
const quant = @import("quant.zig");
const inventory = @import("inventory.zig");
const roles_mod = @import("roles.zig");
const plan = @import("plan.zig");
const Diag = @import("diag.zig").Diag;

pub const gib: f64 = 1 << 30;

/// Everything read and checked for one split.
pub const Model = struct {
    gpa: std.mem.Allocator,
    c: config.Config,
    m: quant.Manifest,
    inv: inventory.Inventory,
    roles: []roles_mod.Role,
    p: plan.Plan,
    cov: plan.Coverage,

    /// From the three texts (config.json, hf_quant_config.json, the inventory's text form).
    pub fn fromTexts(gpa: std.mem.Allocator, config_json: []const u8, quant_json: []const u8, inventory_text: []const u8, split: Want, d: ?*Diag) !Model {
        const c = try config.parse(gpa, config_json, d);
        var m = try quant.parse(gpa, quant_json, d);
        errdefer m.deinit();
        var inv = try inventory.Inventory.fromText(gpa, inventory_text, d);
        errdefer inv.deinit();
        return finish(gpa, c, m, inv, split, d);
    }

    /// From a checkpoint folder: its config.json, hf_quant_config.json and safetensors headers.
    pub fn fromDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, split: Want, d: ?*Diag) !Model {
        const c = try config.Config.read(gpa, io, dir, d);
        var m = try quant.Manifest.read(gpa, io, dir, d);
        errdefer m.deinit();
        var inv = try inventory.Inventory.fromDir(gpa, io, dir, d);
        errdefer inv.deinit();
        return finish(gpa, c, m, inv, split, d);
    }

    fn finish(gpa: std.mem.Allocator, c: config.Config, m: quant.Manifest, inv: inventory.Inventory, want: Want, d: ?*Diag) !Model {
        var out: Model = .{ .gpa = gpa, .c = c, .m = m, .inv = inv, .roles = &.{}, .p = undefined, .cov = undefined };
        out.roles = try roles_mod.classifyAll(gpa, &out.c, &out.m, &out.inv, d);
        errdefer gpa.free(out.roles);
        if (want.layout == .split and want.experts.len > 0) return @import("diag.zig").fail(d, error.BadInkling2Split, "split experts take a world size, not expert counts", .{});
        var split = if (want.layout == .split) try plan.Split.sliced(&out.c, want.world, d) else if (want.experts.len > 0) try plan.Split.uneven(&out.c, want.experts, d) else try plan.Split.even(&out.c, want.world, d);
        if (want.drafter) |r| {
            if (r >= split.world) return @import("diag.zig").fail(d, error.BadInkling2Split, "drafter rank {d} is not below world size {d}", .{ r, split.world });
            split.drafter = r;
        }
        out.p = try plan.build(gpa, &out.c, &out.inv, out.roles, split);
        errdefer out.p.deinit();
        out.cov = try plan.verify(gpa, &out.p, &out.inv, out.roles, d);
        return out;
    }

    pub fn deinit(x: *Model) void {
        x.p.deinit();
        x.gpa.free(x.roles);
        x.inv.deinit();
        x.m.deinit();
        x.* = undefined;
    }

    /// Bytes of keys and values one position takes on rank `r` across the global layers (the paged pool).
    pub fn kvPerPosition(x: *const Model, r: u32) u64 {
        const t = &x.c.text;
        return @as(u64, t.globalCount()) * 2 * x.p.ranks[r].kv.len() * t.head_dim * 2;
    }

    /// Bytes of the sliding layers' key and value rings for one stream on rank `r` (a full window each).
    pub fn windowPerStream(x: *const Model, r: u32) u64 {
        const t = &x.c.text;
        return @as(u64, t.layers - t.globalCount()) * t.window * 2 * x.p.ranks[r].kv.len() * t.swa_head_dim * 2;
    }

    /// Bytes of one stream's convolution state on rank `r`: 3 rows of the four convolutions a layer, in fp32.
    pub fn convPerStream(x: *const Model, r: u32) u64 {
        const t = &x.c.text;
        const kv = @as(u64, x.p.ranks[r].kv.len()) * t.head_dim;
        return @as(u64, t.layers) * (t.conv_taps - 1) * (2 * @as(u64, t.hidden) + 2 * kv) * 4;
    }
};

/// The split a caller asks for: a world size (even experts), or explicit expert counts; the drafter's rank.
pub const Want = struct {
    world: u32 = 4,
    experts: []const u32 = &.{},
    drafter: ?u32 = null,
    /// Whole experts a rank, or a 1/W row slice of every expert on every rank (plan.zig `Layout`).
    layout: plan.Layout = .whole,
};

fn g(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / gib;
}

/// The per-rank memory table, as the plan's section 2 gives it.
pub fn report(w: *std.Io.Writer, x: *const Model, positions: u64, streams: u64) !void {
    const t = &x.c.text;
    const s = x.p.split;
    try w.print("Inkling-2-Small: {d} layers ({d} global), {d} experts ({d} a token), {d} tensors in {d} files, {d:.1} GiB\n", .{ t.layers, t.globalCount(), t.experts, t.topk, x.inv.tensors.len, x.inv.files.len, g(x.cov.total) });
    if (x.c.unknown > 0) try w.print("config.json: {d} keys not known to this parser (first: {s})\n", .{ x.c.unknown, x.c.firstUnknown() });
    try w.print("world {d}: experts", .{s.world});
    if (s.layout == .split) try w.print(" split (all {d} on every rank, a 1/{d} row slice of each)", .{ t.experts, s.world }) else for (s.experts[0..s.world], 0..) |n, i| try w.print("{s}{d}", .{ if (i == 0) " " else "/", n });
    try w.print("; drafter on rank {d}; coverage: every byte on one rank, {d:.2} GiB replicated on all\n\n", .{ s.drafter, g(x.cov.replicated) });
    try w.writeAll("rank  q heads  kv heads  experts    | experts  attention  replicated  embed+unembed  vision  drafter | weights  KV pool  total GiB\n");
    for (x.p.ranks) |r| {
        const b = r.bytes;
        const pool = x.kvPerPosition(r.rank) * positions;
        try w.print("{d:>4}  {d:>2}-{d:<2}    {d}-{d}       {d:>3}-{d:<3}    | {d:>7.1}  {d:>9.1}  {d:>10.1}  {d:>13.1}  {d:>6.1}  {d:>7.1} | {d:>7.1}  {d:>7.1}  {d:>9.1}\n", .{
            r.rank,                                 r.q.lo,                                  r.q.hi - 1,                               r.kv.lo,                                    r.kv.hi - 1,
            r.experts.lo,                           r.experts.hi - 1,                        g(b[@backingInt(plan.Category.experts)]), g(b[@backingInt(plan.Category.attention)]), g(b[@backingInt(plan.Category.replicated)]),
            g(b[@backingInt(plan.Category.embed)]), g(b[@backingInt(plan.Category.vision)]), g(b[@backingInt(plan.Category.drafter)]), g(r.total()),                               g(pool),
            g(r.total() + pool),
        });
    }
    const r0 = x.p.ranks[0].rank;
    try w.print("\nKV pool: {d} positions, {d} bytes a position a rank (the {d} global layers' keys and values, bf16)\n", .{ positions, x.kvPerPosition(r0), t.globalCount() });
    try w.print("per stream, rank 0: sliding-window rings {d:.3} GiB, convolution state {d:.1} MiB; {d} streams: {d:.2} GiB\n", .{
        g(x.windowPerStream(r0)), @as(f64, @floatFromInt(x.convPerStream(r0))) / (1 << 20), streams, g((x.windowPerStream(r0) + x.convPerStream(r0)) * streams),
    });
    try w.writeAll("not counted: scratch, the CUDA context and transport buffers (the plan estimates 3 GiB a rank)\n");
}

/// Rank `rank`'s byte ranges, one line a tensor: file, absolute offset, run length, stride, runs, tensor name.
pub fn ranges(w: *std.Io.Writer, x: *const Model, rank: u32) !void {
    try w.writeAll("# file\toffset\tlen\tstride\tcount\tplacement\ttensor\n");
    for (x.p.ranks[rank].pieces) |piece| {
        const t = x.inv.tensors[piece.tensor];
        const r = piece.run;
        try w.print("{s}\t{d}\t{d}\t{d}\t{d}\t{t}\t{s}\n", .{ x.inv.files[r.file].name, r.offset, r.len, r.stride, r.count, x.p.split.placed(x.roles[piece.tensor]), t.name });
    }
}

test "inkling2 describe: the release's table at W = 4 and its ranges" {
    var d: Diag = .{};
    var x = Model.fromTexts(std.testing.allocator, config.fixture, quant.fixture, inventory.fixture, .{ .world = 4 }, &d) catch |e| {
        std.debug.print("{s}\n", .{d.text()});
        return e;
    };
    defer x.deinit();
    try std.testing.expectEqual(@as(u64, 8192), x.kvPerPosition(0)); // 8.2 KB a token a rank
    try std.testing.expectEqual(@as(u64, 4 << 30), x.kvPerPosition(0) * 524288);
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try report(&out.writer, &x, 524288, 8);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "world 4: experts 96/96/96/96; drafter on rank 3") != null);
    out.clearRetainingCapacity();
    try ranges(&out.writer, &x, 1);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "q_cols\tmodel.llm.layers.0.attn.wo_ud.weight") != null);
    var u = try Model.fromTexts(std.testing.allocator, config.fixture, quant.fixture, inventory.fixture, .{ .experts = &.{ 93, 97, 97, 97 }, .drafter = 2 }, null);
    defer u.deinit();
    try std.testing.expectEqual(@as(u32, 93), u.p.ranks[0].experts.len());
    try std.testing.expect(u.p.ranks[2].bytes[@backingInt(plan.Category.drafter)] > 3 << 30);
    var sp = try Model.fromTexts(std.testing.allocator, config.fixture, quant.fixture, inventory.fixture, .{ .world = 4, .layout = .split }, null);
    defer sp.deinit();
    out.clearRetainingCapacity();
    try report(&out.writer, &sp, 524288, 8);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "world 4: experts split (all 384 on every rank, a 1/4 row slice of each)") != null);
    out.clearRetainingCapacity();
    try ranges(&out.writer, &sp, 1);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "replicated\tmodel.llm.layers.0.mlp.experts.w13_weight_amax") != null);
    try std.testing.expectError(error.BadInkling2Split, Model.fromTexts(std.testing.allocator, config.fixture, quant.fixture, inventory.fixture, .{ .experts = &.{ 93, 97, 97, 97 }, .layout = .split }, null));
}
