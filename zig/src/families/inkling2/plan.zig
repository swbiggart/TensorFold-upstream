//! The TP/EP slice plan: for world size W, each rank's query and key heads, its experts, the tensors every rank holds
//! whole, and rank 0's and the drafter rank's own, as byte ranges inside the safetensors files.
//!
//! Attention is tensor-parallel by key head (a key head's query heads go with it: 12 query and 2 key heads a rank at
//! W = 4). The MoE is placed one of two ways (`Layout`):
//!   whole: expert-parallel (96 experts a rank at W = 4, or any split the caller gives, such as 93/97/97/97). Experts
//!     are fused along axis 0 in the release, so a rank's experts are one contiguous byte range of each expert tensor.
//!   split: every rank holds a 1/W row slice of every expert (checkpoint mode only; cuda_moe.zig `splitUp`): rows
//!     [r*n/W, (r+1)*n/W) of each expert's w13 (n = 2I = 4608; the rows interleave gate and up, row 2i = gate_i and
//!     2i + 1 = up_i, as TokenSpeed's `gate_up[:, 0::2]` and Transformers' Interleave converter read them, so the
//!     slice is gate rows r*I/W .. and the matching up rows) and of w2 (n = H = 4608 output rows, all of K), the block
//!     scales the same rows, and the per-expert amax whole on every rank. A slice is one strided run a tensor: 384 runs
//!     of n/W rows, n rows apart.
//! Heads split by canon's rule at equal weights (largest remainder, the first ranks one more), as the mesh layer's
//! `shard.range` does. A query-head column cut (the output projection) is one run a row. The unembed is split by
//! vocabulary entry (canon's rule over the padded vocabulary: 50,256 rows a rank at W = 4) under either expert layout,
//! so every rank computes its share of the logits; the embed stays whole on rank 0.
const std = @import("std");
const inventory = @import("inventory.zig");
const roles_mod = @import("roles.zig");
const Config = @import("config.zig").Config;
const Diag = @import("diag.zig").Diag;
const fail = @import("diag.zig").fail;
const Role = roles_mod.Role;
const Placement = roles_mod.Placement;

pub const max_world = 16;

pub const Span = struct {
    lo: u32,
    hi: u32,

    pub fn len(s: Span) u32 {
        return s.hi - s.lo;
    }
};

/// Rank `rank`'s share of `units` over `world` at equal weights: the first `units % world` ranks one more.
pub fn canon(units: u32, rank: u32, world: u32) Span {
    const base = units / world;
    const extra = units % world;
    const lo = rank * base + @min(rank, extra);
    return .{ .lo = lo, .hi = lo + base + @intFromBool(rank < extra) };
}

/// How ranks hold the experts (see the header).
pub const Layout = enum {
    whole,
    split,

    /// TF_INK2_EXPERTS (whole when unset or unknown).
    pub fn env() Layout {
        const v = std.c.getenv("TF_INK2_EXPERTS") orelse return .whole;
        return std.meta.stringToEnum(Layout, std.mem.span(v)) orelse .whole;
    }
};

pub const Split = struct {
    world: u32,
    /// Experts a rank holds, in rank order: rank r holds the next `experts[r]` after rank r - 1's (whole); all of them
    /// on every rank (split).
    experts: [max_world]u32 = @splat(0),
    /// The rank that holds the drafter (DSpark): the last rank, so rank 0 keeps room for the embed and unembed.
    drafter: u32,
    layout: Layout = .whole,

    /// The even split: experts by canon's rule, the drafter on the last rank.
    pub fn even(c: *const Config, world: u32, d: ?*Diag) !Split {
        if (world == 0 or world > max_world) return fail(d, error.BadInkling2Split, "world size {d} is not from 1 to {d}", .{ world, max_world });
        var s: Split = .{ .world = world, .drafter = world - 1 };
        for (0..world) |r| s.experts[r] = canon(c.text.experts, @intCast(r), world).len();
        return s.checked(c, d);
    }

    /// Experts a rank as given (`counts.len` is the world size).
    pub fn uneven(c: *const Config, counts: []const u32, d: ?*Diag) !Split {
        if (counts.len == 0 or counts.len > max_world) return fail(d, error.BadInkling2Split, "{d} expert counts: give 1 to {d}", .{ counts.len, max_world });
        var s: Split = .{ .world = @intCast(counts.len), .drafter = @intCast(counts.len - 1) };
        @memcpy(s.experts[0..counts.len], counts);
        return s.checked(c, d);
    }

    /// Every expert on every rank, a 1/W row slice of each (see the header), the drafter on the last rank. The slices
    /// must be whole column tiles of the checkpoint-mode kernels (128 rows) and h's slices whole MXFP8 blocks (32).
    pub fn sliced(c: *const Config, world: u32, d: ?*Diag) !Split {
        if (world == 0 or world > max_world) return fail(d, error.BadInkling2Split, "world size {d} is not from 1 to {d}", .{ world, max_world });
        var s: Split = .{ .world = world, .drafter = world - 1, .layout = .split };
        for (0..world) |r| s.experts[r] = c.text.experts;
        const I = c.text.moe_intermediate;
        const ok = (2 * I) % (128 * world) == 0 and c.text.hidden % (128 * world) == 0 and I % (32 * world) == 0;
        if (!ok) return fail(d, error.BadInkling2Split, "split experts at world size {d}: {d} gate/up rows, {d} output rows and {d} h values a rank are not whole tiles (128 rows, 32 values)", .{ world, 2 * I / world, c.text.hidden / world, I / world });
        if (world > c.text.kv_heads) return fail(d, error.BadInkling2Split, "world size {d} is more than the {d} key heads", .{ world, c.text.kv_heads });
        return s;
    }

    fn checked(s: Split, c: *const Config, d: ?*Diag) !Split {
        var sum: u64 = 0;
        for (s.experts[0..s.world], 0..) |n, r| {
            if (n == 0) return fail(d, error.BadInkling2Split, "rank {d} would hold no experts", .{r});
            sum += n;
        }
        if (sum != c.text.experts) return fail(d, error.BadInkling2Split, "the expert counts sum to {d}, not {d}", .{ sum, c.text.experts });
        if (s.world > c.text.kv_heads) return fail(d, error.BadInkling2Split, "world size {d} is more than the {d} key heads", .{ s.world, c.text.kv_heads });
        if (s.drafter >= s.world) return fail(d, error.BadInkling2Split, "drafter rank {d} is not below world size {d}", .{ s.drafter, s.world });
        return s;
    }

    pub fn expertSpan(s: Split, rank: u32) Span {
        if (s.layout == .split) return .{ .lo = 0, .hi = s.experts[rank] };
        var lo: u32 = 0;
        for (s.experts[0..rank]) |n| lo += n;
        return .{ .lo = lo, .hi = lo + s.experts[rank] };
    }

    /// The rows of each expert's `rows` (w13: 2I, w2: H) rank `rank` holds: all of them (whole), its 1/W (split).
    pub fn rowSpan(s: Split, rows: u32, rank: u32) Span {
        return if (s.layout == .split) canon(rows, rank, s.world) else .{ .lo = 0, .hi = rows };
    }

    /// How a rank holds a tensor of `role` under this split: the split layout's per-expert amax is whole on every rank.
    pub fn placed(s: Split, role: Role) Placement {
        if (s.layout == .split and role.placement == .experts and role.part == .amax) return .replicated;
        return role.placement;
    }

    pub fn kvSpan(s: Split, c: *const Config, rank: u32) Span {
        return canon(c.text.kv_heads, rank, s.world);
    }

    pub fn qSpan(s: Split, c: *const Config, rank: u32) Span {
        const kv = s.kvSpan(c, rank);
        const g = c.group();
        return .{ .lo = kv.lo * g, .hi = kv.hi * g };
    }
};

/// `count` runs of `len` bytes, `stride` apart, from `offset` (absolute, from the file's start).
pub const Run = struct {
    file: u16,
    offset: u64,
    len: u64,
    stride: u64,
    count: u64,

    pub fn bytes(r: Run) u64 {
        return r.len * r.count;
    }
};

pub const Piece = struct {
    tensor: u32,
    run: Run,
};

/// The memory plan's columns.
pub const Category = enum {
    experts,
    attention,
    replicated,
    embed,
    vision,
    drafter,

    pub fn of(r: Role) Category {
        return switch (r.placement) {
            .experts => .experts,
            .q_rows, .kv_rows, .q_cols, .kv_cols => .attention,
            .replicated => .replicated,
            .first, .vocab_rows => if (r.kind == .vision) .vision else .embed,
            .drafter => .drafter,
        };
    }
};
pub const categories = @typeInfo(Category).@"enum".field_names.len;

pub const Rank = struct {
    rank: u32,
    q: Span,
    kv: Span,
    experts: Span,
    pieces: []Piece,
    bytes: [categories]u64,

    pub fn total(r: *const Rank) u64 {
        var n: u64 = 0;
        for (r.bytes) |b| n += b;
        return n;
    }
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    split: Split,
    ranks: []Rank,

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }
};

/// Rank `rank`'s run of tensor `t` under `role`, or null when the rank holds none of it.
pub fn run(c: *const Config, inv: *const inventory.Inventory, t: inventory.Tensor, role: Role, split: Split, rank: u32) ?Run {
    const base = inv.offset(t);
    const whole: Run = .{ .file = t.file, .offset = base, .len = t.bytes(), .stride = t.bytes(), .count = 1 };
    const placement = split.placed(role);
    if (placement == .experts and split.layout == .split) {
        // rows [lo, hi) of each expert: one run an expert, `rows` rows apart
        if (t.rank != 3) return null;
        const rows: u32 = @intCast(t.dim(1));
        const span = split.rowSpan(rows, rank);
        if (span.len() == 0) return null;
        if (span.len() == rows) return whole;
        const row = t.bytes() / (t.dim(0) * rows);
        return .{ .file = t.file, .offset = base + @as(u64, span.lo) * row, .len = @as(u64, span.len()) * row, .stride = @as(u64, rows) * row, .count = t.dim(0) };
    }
    const span: Span = switch (placement) {
        .replicated => return whole,
        .first => return if (rank == 0) whole else null,
        .drafter => return if (rank == split.drafter) whole else null,
        .q_rows, .q_cols => split.qSpan(c, rank),
        .kv_rows, .kv_cols => split.kvSpan(c, rank),
        .experts => split.expertSpan(rank),
        .vocab_rows => canon(c.text.vocab, rank, split.world),
    };
    if (span.len() == 0) return null;
    switch (role.placement) {
        .q_rows, .kv_rows, .experts, .vocab_rows => {
            const row = t.bytes() / t.dim(0);
            const off = base + @as(u64, span.lo) * role.unit * row;
            const len = @as(u64, span.len()) * role.unit * row;
            return .{ .file = t.file, .offset = off, .len = len, .stride = len, .count = 1 };
        },
        .q_cols, .kv_cols => {
            const es = t.dtype.size();
            return .{ .file = t.file, .offset = base + @as(u64, span.lo) * role.unit * es, .len = @as(u64, span.len()) * role.unit * es, .stride = t.dim(1) * es, .count = t.dim(0) };
        },
        else => unreachable,
    }
}

pub fn build(gpa: std.mem.Allocator, c: *const Config, inv: *const inventory.Inventory, roles: []const Role, split: Split) !Plan {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const ranks = try a.alloc(Rank, split.world);
    for (ranks, 0..) |*r, i| {
        const rank: u32 = @intCast(i);
        var pieces: std.ArrayList(Piece) = .empty;
        r.* = .{ .rank = rank, .q = split.qSpan(c, rank), .kv = split.kvSpan(c, rank), .experts = split.expertSpan(rank), .pieces = &.{}, .bytes = @splat(0) };
        for (inv.tensors, roles, 0..) |t, role, ti| if (run(c, inv, t, role, split, rank)) |x| {
            try pieces.append(a, .{ .tensor = @intCast(ti), .run = x });
            r.bytes[@backingInt(Category.of(role))] += x.bytes();
        };
        r.pieces = pieces.items;
    }
    return .{ .arena = arena, .split = split, .ranks = ranks };
}

pub const Coverage = struct {
    /// Bytes of every tensor in the checkpoint.
    total: u64,
    /// Bytes of the tensors every rank holds whole (counted once).
    replicated: u64,
    /// Bytes the ranks hold, over all ranks.
    held: u64,
};

/// Checks that every byte of every tensor is held by exactly one rank, except replicated tensors, which every rank
/// holds whole; and that no run leaves its tensor.
pub fn verify(gpa: std.mem.Allocator, p: *const Plan, inv: *const inventory.Inventory, roles: []const Role, d: ?*Diag) !Coverage {
    const Entry = struct { tensor: u32, rank: u32, run: Run };
    var all: std.ArrayList(Entry) = .empty;
    defer all.deinit(gpa);
    var cov: Coverage = .{ .total = inv.totalBytes(), .replicated = 0, .held = 0 };
    for (p.ranks) |r| for (r.pieces) |x| {
        try all.append(gpa, .{ .tensor = x.tensor, .rank = r.rank, .run = x.run });
        cov.held += x.run.bytes();
    };
    std.mem.sort(Entry, all.items, {}, struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            return if (x.tensor != y.tensor) x.tensor < y.tensor else x.rank < y.rank;
        }
    }.lt);
    const Interval = struct { lo: u64, hi: u64 };
    var spans: std.ArrayList(Interval) = .empty;
    defer spans.deinit(gpa);
    var i: usize = 0;
    for (inv.tensors, roles, 0..) |t, role, ti| {
        const lo = inv.offset(t);
        const hi = lo + t.bytes();
        var j = i;
        while (j < all.items.len and all.items[j].tensor == ti) j += 1;
        const mine = all.items[i..j];
        i = j;
        for (mine) |e| if (e.run.file != t.file) return fail(d, error.Inkling2PlanGap, "{s}: rank {d}'s run is in another file", .{ t.name, e.rank });
        if (p.split.placed(role) == .replicated) {
            cov.replicated += t.bytes();
            if (mine.len != p.split.world) return fail(d, error.Inkling2PlanGap, "{s}: replicated on {d} of {d} ranks", .{ t.name, mine.len, p.split.world });
            for (mine, 0..) |e, r| if (e.rank != r or e.run.offset != lo or e.run.bytes() != t.bytes() or e.run.count != 1)
                return fail(d, error.Inkling2PlanGap, "{s}: rank {d} does not hold it whole", .{ t.name, e.rank });
            continue;
        }
        spans.clearRetainingCapacity();
        for (mine) |e| {
            for (0..e.run.count) |k| {
                const s = e.run.offset + k * e.run.stride;
                try spans.append(gpa, .{ .lo = s, .hi = s + e.run.len });
            }
        }
        std.mem.sort(Interval, spans.items, {}, struct {
            fn lt(_: void, x: Interval, y: Interval) bool {
                return x.lo < y.lo;
            }
        }.lt);
        var at = lo;
        for (spans.items) |s| {
            if (s.lo != at) return fail(d, error.Inkling2PlanGap, "{s}: bytes {d} to {d} held {s}", .{ t.name, @min(at, s.lo) - lo, @max(at, s.lo) - lo, if (s.lo < at) "twice" else "by no rank" });
            at = s.hi;
        }
        if (at != hi) return fail(d, error.Inkling2PlanGap, "{s}: its last {d} bytes held by no rank", .{ t.name, hi -| at });
    }
    if (cov.held != cov.total + (p.split.world - 1) * cov.replicated)
        return fail(d, error.Inkling2PlanGap, "the ranks hold {d} bytes, not {d}", .{ cov.held, cov.total + (p.split.world - 1) * cov.replicated });
    return cov;
}

const testing = std.testing;
const config = @import("config.zig");
const quant = @import("quant.zig");

fn fixtures() !struct { c: Config, m: quant.Manifest, inv: inventory.Inventory, roles: []Role } {
    const c = try config.parse(testing.allocator, config.fixture, null);
    var m = try quant.parse(testing.allocator, quant.fixture, null);
    errdefer m.deinit();
    var inv = try inventory.Inventory.fromText(testing.allocator, inventory.fixture, null);
    errdefer inv.deinit();
    const roles = try roles_mod.classifyAll(testing.allocator, &c, &m, &inv, null);
    return .{ .c = c, .m = m, .inv = inv, .roles = roles };
}

const gib: f64 = 1 << 30;

test "inkling2 plan: canon's rule matches the mesh layer's (largest remainder, the first ranks one more)" {
    try testing.expectEqual(Span{ .lo = 0, .hi = 3 }, canon(8, 0, 3));
    try testing.expectEqual(Span{ .lo = 3, .hi = 6 }, canon(8, 1, 3));
    try testing.expectEqual(Span{ .lo = 6, .hi = 8 }, canon(8, 2, 3));
    try testing.expectEqual(Span{ .lo = 288, .hi = 384 }, canon(384, 3, 4));
}

test "inkling2 plan: every byte held once at W = 1, 2, 3 and 4 and at 93/97/97/97, replicated tensors on every rank" {
    var f = try fixtures();
    defer {
        f.m.deinit();
        f.inv.deinit();
        testing.allocator.free(f.roles);
    }
    const uneven = [_]u32{ 93, 97, 97, 97 };
    var replicated: ?u64 = null;
    for ([_]u32{ 1, 2, 3, 4, 0 }) |w| {
        const split = if (w == 0) try Split.uneven(&f.c, &uneven, null) else try Split.even(&f.c, w, null);
        var p = try build(testing.allocator, &f.c, &f.inv, f.roles, split);
        defer p.deinit();
        var d: Diag = .{};
        const cov = verify(testing.allocator, &p, &f.inv, f.roles, &d) catch |e| {
            std.debug.print("world {d}: {s}\n", .{ split.world, d.text() });
            return e;
        };
        try testing.expectEqual(@as(u64, 393_479_869_088), cov.total);
        if (replicated) |r| try testing.expectEqual(r, cov.replicated) else replicated = cov.replicated;
        var experts: u32 = 0;
        var q: u32 = 0;
        for (p.ranks) |r| {
            try testing.expectEqual(experts, r.experts.lo);
            experts = r.experts.hi;
            try testing.expectEqual(q, r.q.lo);
            q = r.q.hi;
            try testing.expectEqual(r.q.len(), 6 * r.kv.len());
        }
        try testing.expectEqual(@as(u32, 384), experts);
        try testing.expectEqual(@as(u32, 48), q);
    }
}

test "inkling2 plan: W = 4 reproduces the plan's table (GiB a rank)" {
    var f = try fixtures();
    defer {
        f.m.deinit();
        f.inv.deinit();
        testing.allocator.free(f.roles);
    }
    var p = try build(testing.allocator, &f.c, &f.inv, f.roles, try Split.even(&f.c, 4, null));
    defer p.deinit();
    for (p.ranks) |r| {
        try testing.expectEqual(@as(u32, 12), r.q.len());
        try testing.expectEqual(@as(u32, 2), r.kv.len());
        try testing.expectEqual(@as(u32, 96), r.experts.len());
        try testing.expectApproxEqAbs(@as(f64, 87.6), @as(f64, @floatFromInt(r.bytes[@backingInt(Category.experts)])) / gib, 0.05);
        try testing.expectApproxEqAbs(@as(f64, 2.2), @as(f64, @floatFromInt(r.bytes[@backingInt(Category.attention)])) / gib, 0.05);
        try testing.expectApproxEqAbs(@as(f64, 0.2), @as(f64, @floatFromInt(r.bytes[@backingInt(Category.replicated)])) / gib, 0.05);
    }
    // rank 0: the embed (1.73 GiB), the vision tower and a quarter of the unembed; every other rank a quarter of the
    // unembed (50,256 rows of 4608 bf16)
    const first = @as(f64, @floatFromInt(p.ranks[0].bytes[@backingInt(Category.embed)] + p.ranks[0].bytes[@backingInt(Category.vision)])) / gib;
    try testing.expectApproxEqAbs(@as(f64, 4.1 - 3 * 0.43), first, 0.05);
    try testing.expectApproxEqAbs(@as(f64, 3.2), @as(f64, @floatFromInt(p.ranks[3].bytes[@backingInt(Category.drafter)])) / gib, 0.05);
    try testing.expectEqual(@as(u64, 0), p.ranks[1].bytes[@backingInt(Category.drafter)]);
    for (1..4) |r| try testing.expectEqual(@as(u64, 50256 * 4608 * 2), p.ranks[r].bytes[@backingInt(Category.embed)]);
    const u = f.inv.find("model.llm.unembed.weight").?;
    var seen = false;
    for (p.ranks[3].pieces) |x| if (std.mem.eql(u8, f.inv.tensors[x.tensor].name, u.name)) {
        try testing.expectEqual(f.inv.offset(u) + 3 * 50256 * 4608 * 2, x.run.offset);
        try testing.expectEqual(@as(u64, 50256 * 4608 * 2), x.run.len);
        seen = true;
    };
    try testing.expect(seen);
    // rank 2's experts 192-287 of layer 0's gate-up are one run: 96 experts of 4608 x 2304 bytes
    const t = f.inv.find("model.llm.layers.0.mlp.experts.w13_weight").?;
    for (p.ranks[2].pieces) |x| if (std.mem.eql(u8, f.inv.tensors[x.tensor].name, t.name)) {
        try testing.expectEqual(@as(u64, 1), x.run.count);
        try testing.expectEqual(f.inv.offset(t) + 192 * 4608 * 2304, x.run.offset);
        try testing.expectEqual(@as(u64, 96 * 4608 * 2304), x.run.len);
    };
}

test "inkling2 plan: split experts at W = 1, 2, 3 and 4 hold every byte once (amax on every rank), W = 4 a quarter of each expert's rows" {
    var f = try fixtures();
    defer {
        f.m.deinit();
        f.inv.deinit();
        testing.allocator.free(f.roles);
    }
    var whole = try build(testing.allocator, &f.c, &f.inv, f.roles, try Split.even(&f.c, 4, null));
    defer whole.deinit();
    for ([_]u32{ 1, 2, 3, 4 }) |w| {
        var p = try build(testing.allocator, &f.c, &f.inv, f.roles, try Split.sliced(&f.c, w, null));
        defer p.deinit();
        var d: Diag = .{};
        _ = verify(testing.allocator, &p, &f.inv, f.roles, &d) catch |e| {
            std.debug.print("split world {d}: {s}\n", .{ w, d.text() });
            return e;
        };
        for (p.ranks) |r| try testing.expectEqual(Span{ .lo = 0, .hi = 384 }, r.experts);
        if (w != 4) continue;
        // the same bytes a rank as whole experts, give or take the amax every rank now holds whole
        for (p.ranks, whole.ranks) |r, x| {
            const a = r.bytes[@backingInt(Category.experts)];
            const b = x.bytes[@backingInt(Category.experts)];
            try testing.expect(a >= b and a - b < 1 << 20);
            try testing.expectEqual(r.total() - a, x.total() - b);
        }
        // rank 2's slice of layer 0's gate-up: rows 2304 .. 3455 of each expert (gate/up pairs 1152 .. 1727)
        const t = f.inv.find("model.llm.layers.0.mlp.experts.w13_weight").?;
        const sc = f.inv.find("model.llm.layers.40.mlp.experts.w2_weight_scale").?;
        const am = f.inv.find("model.llm.layers.0.mlp.experts.w2_weight_amax").?;
        var seen: u32 = 0;
        for (p.ranks[2].pieces) |x| {
            const name = f.inv.tensors[x.tensor].name;
            if (std.mem.eql(u8, name, t.name)) {
                try testing.expectEqual(Run{ .file = t.file, .offset = f.inv.offset(t) + 2304 * 2304, .len = 1152 * 2304, .stride = 4608 * 2304, .count = 384 }, x.run);
                seen += 1;
            } else if (std.mem.eql(u8, name, sc.name)) {
                // MXFP8 w2 scales: 4608 rows of 2304 / 32 a row
                try testing.expectEqual(Run{ .file = sc.file, .offset = f.inv.offset(sc) + 2304 * 72, .len = 1152 * 72, .stride = 4608 * 72, .count = 384 }, x.run);
                seen += 1;
            } else if (std.mem.eql(u8, name, am.name)) {
                try testing.expectEqual(@as(u64, 384 * 4), x.run.bytes());
                seen += 1;
            }
        }
        try testing.expectEqual(@as(u32, 3), seen);
        // the head under split experts too: every rank a quarter of the unembed by vocabulary (50,256 rows of 4608
        // bf16), rank 3's at its offset
        for (1..4) |r| try testing.expectEqual(@as(u64, 50256 * 4608 * 2), p.ranks[r].bytes[@backingInt(Category.embed)]);
        const u = f.inv.find("model.llm.unembed.weight").?;
        var head_seen = false;
        for (p.ranks[3].pieces) |x| if (std.mem.eql(u8, f.inv.tensors[x.tensor].name, u.name)) {
            try testing.expectEqual(Run{ .file = u.file, .offset = f.inv.offset(u) + 3 * 50256 * 4608 * 2, .len = 50256 * 4608 * 2, .stride = 50256 * 4608 * 2, .count = 1 }, x.run);
            head_seen = true;
        };
        try testing.expect(head_seen);
    }
    try testing.expectError(error.BadInkling2Split, Split.sliced(&f.c, 5, null));
    try testing.expectError(error.BadInkling2Split, Split.sliced(&f.c, 8, null));
}

test "inkling2 plan: splits that leave a rank empty, miscount the experts or outnumber the key heads are refused" {
    const c = try config.parse(testing.allocator, config.fixture, null);
    try testing.expectError(error.BadInkling2Split, Split.uneven(&c, &.{ 100, 100, 100, 100 }, null));
    try testing.expectError(error.BadInkling2Split, Split.uneven(&c, &.{ 384, 0 }, null));
    try testing.expectError(error.BadInkling2Split, Split.even(&c, 9, null));
    try testing.expectError(error.BadInkling2Split, Split.even(&c, 0, null));
}

test "inkling2 plan: a replicated tensor's piece in another file is refused" {
    var f = try fixtures();
    defer {
        f.m.deinit();
        f.inv.deinit();
        testing.allocator.free(f.roles);
    }
    var p = try build(testing.allocator, &f.c, &f.inv, f.roles, try Split.even(&f.c, 2, null));
    defer p.deinit();
    var moved: ?[]const u8 = null;
    for (p.ranks[1].pieces) |*x| if (f.roles[x.tensor].placement == .replicated) {
        x.run.file = @intCast((x.run.file + 1) % f.inv.files.len);
        moved = f.inv.tensors[x.tensor].name;
        break;
    };
    var d: Diag = .{};
    try testing.expectError(error.Inkling2PlanGap, verify(testing.allocator, &p, &f.inv, f.roles, &d));
    try testing.expect(std.mem.indexOf(u8, d.text(), moved.?) != null);
    try testing.expect(std.mem.indexOf(u8, d.text(), "another file") != null);
}
