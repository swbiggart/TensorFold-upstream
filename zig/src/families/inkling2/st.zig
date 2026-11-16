//! Safetensors files for the CUDA runner and its tools: the oracle's per-prompt files read by name, the runner's dumps
//! written. Any dtype the format names is read; a dump writes the dtypes the forward holds (BF16, F32, I32, I64, U8).
const std = @import("std");

pub const DType = enum {
    bf16,
    f16,
    f32,
    f64,
    i64,
    i32,
    i16,
    i8,
    u8,
    bool,
    f8_e4m3,

    const names = [_][]const u8{ "BF16", "F16", "F32", "F64", "I64", "I32", "I16", "I8", "U8", "BOOL", "F8_E4M3" };

    pub fn name(d: DType) []const u8 {
        return names[@intFromEnum(d)];
    }

    pub fn parse(text: []const u8) ?DType {
        for (names, 0..) |n, i| if (std.mem.eql(u8, n, text)) return @enumFromInt(i);
        return null;
    }

    pub fn size(d: DType) u64 {
        return switch (d) {
            .f64, .i64 => 8,
            .f32, .i32 => 4,
            .bf16, .f16, .i16 => 2,
            .i8, .u8, .bool, .f8_e4m3 => 1,
        };
    }
};

pub const max_rank = 6;

pub const Entry = struct {
    name: []const u8,
    dtype: DType,
    rank: u8,
    shape: [max_rank]u64,
    /// Bytes from the data region's start.
    begin: u64,
    end: u64,

    pub fn dim(e: Entry, i: usize) u64 {
        return if (i < e.rank) e.shape[i] else 1;
    }

    pub fn count(e: Entry) u64 {
        var n: u64 = 1;
        for (e.shape[0..e.rank]) |x| n *= x;
        return n;
    }

    pub fn bytes(e: Entry) u64 {
        return e.end - e.begin;
    }
};

/// A safetensors file opened for reading by name; its header parsed, its data read on demand.
pub const Reader = struct {
    arena: std.heap.ArenaAllocator,
    io: std.Io,
    file: std.Io.File,
    data_start: u64,
    entries: []Entry,
    /// __metadata__ pairs, as stored.
    meta: []const [2][]const u8,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Reader {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        var r: Reader = .{ .arena = .init(gpa), .io = io, .file = file, .data_start = 0, .entries = &.{}, .meta = &.{} };
        errdefer r.arena.deinit();
        const size = try file.length(io);
        var prefix: [8]u8 = undefined;
        if (size < 8 or try file.readPositionalAll(io, &prefix, 0) != 8) return error.BadSafetensors;
        const n = std.mem.readInt(u64, &prefix, .little);
        if (n > 1 << 30 or n > size - 8) return error.BadSafetensors;
        const text = try r.arena.allocator().alloc(u8, @intCast(n));
        if (try file.readPositionalAll(io, text, 8) != text.len) return error.BadSafetensors;
        r.data_start = 8 + n;
        try r.parse(text, size - 8 - n);
        return r;
    }

    fn parse(r: *Reader, text: []const u8, data_len: u64) !void {
        const a = r.arena.allocator();
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (doc != .object) return error.BadSafetensors;
        var list: std.ArrayList(Entry) = .empty;
        var meta: std.ArrayList([2][]const u8) = .empty;
        var it = doc.object.iterator();
        while (it.next()) |kv| {
            const v = kv.value_ptr.*;
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) {
                if (v != .object) return error.BadSafetensors;
                var mi = v.object.iterator();
                while (mi.next()) |m| if (m.value_ptr.* == .string) try meta.append(a, .{ m.key_ptr.*, m.value_ptr.string });
                continue;
            }
            if (v != .object) return error.BadSafetensors;
            const dt = v.object.get("dtype") orelse return error.BadSafetensors;
            const shape = v.object.get("shape") orelse return error.BadSafetensors;
            const offs = v.object.get("data_offsets") orelse return error.BadSafetensors;
            if (dt != .string or shape != .array or offs != .array or offs.array.items.len != 2) return error.BadSafetensors;
            if (shape.array.items.len > max_rank) return error.BadSafetensors;
            var e: Entry = .{ .name = kv.key_ptr.*, .dtype = DType.parse(dt.string) orelse return error.UnsupportedDType, .rank = @intCast(shape.array.items.len), .shape = @splat(1), .begin = 0, .end = 0 };
            for (shape.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0) return error.BadSafetensors;
                e.shape[i] = @intCast(d.integer);
            }
            for (offs.array.items) |o| if (o != .integer or o.integer < 0) return error.BadSafetensors;
            e.begin = @intCast(offs.array.items[0].integer);
            e.end = @intCast(offs.array.items[1].integer);
            if (e.end < e.begin or e.end > data_len or e.bytes() != e.count() * e.dtype.size()) return error.BadSafetensors;
            try list.append(a, e);
        }
        std.mem.sort(Entry, list.items, {}, struct {
            fn lt(_: void, x: Entry, y: Entry) bool {
                return std.mem.order(u8, x.name, y.name) == .lt;
            }
        }.lt);
        r.entries = list.items;
        r.meta = meta.items;
    }

    pub fn close(r: *Reader) void {
        r.file.close(r.io);
        r.arena.deinit();
        r.* = undefined;
    }

    pub fn find(r: *const Reader, name: []const u8) ?Entry {
        var lo: usize = 0;
        var hi: usize = r.entries.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            switch (std.mem.order(u8, r.entries[mid].name, name)) {
                .eq => return r.entries[mid],
                .lt => lo = mid + 1,
                .gt => hi = mid,
            }
        }
        return null;
    }

    pub fn metaValue(r: *const Reader, key: []const u8) ?[]const u8 {
        for (r.meta) |m| if (std.mem.eql(u8, m[0], key)) return m[1];
        return null;
    }

    /// `out.len` bytes of `e` from byte `offset` within it.
    pub fn readInto(r: *const Reader, e: Entry, offset: u64, out: []u8) !void {
        if (offset > e.bytes() or out.len > e.bytes() - offset) return error.BadSafetensors;
        if (try r.file.readPositionalAll(r.io, out, r.data_start + e.begin + offset) != out.len) return error.BadSafetensors;
    }

    /// The whole of tensor `name`, which must have dtype `dtype`; the caller frees it.
    pub fn alloc(r: *const Reader, gpa: std.mem.Allocator, name: []const u8, dtype: DType) ![]u8 {
        const e = r.find(name) orelse return error.MissingTensor;
        if (e.dtype != dtype) return error.UnexpectedDType;
        const out = try gpa.alloc(u8, @intCast(e.bytes()));
        errdefer gpa.free(out);
        try r.readInto(e, 0, out);
        return out;
    }
};

/// Tensors gathered in memory, then written as one safetensors file (names sorted, data 8-byte aligned).
pub const Writer = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(Item) = .empty,
    meta: std.ArrayList([2][]u8) = .empty,

    const Item = struct { name: []u8, dtype: DType, rank: u8, shape: [max_rank]u64, data: []u8 };

    pub fn init(gpa: std.mem.Allocator) Writer {
        return .{ .gpa = gpa };
    }

    pub fn deinit(w: *Writer) void {
        for (w.items.items) |it| {
            w.gpa.free(it.name);
            w.gpa.free(it.data);
        }
        w.items.deinit(w.gpa);
        for (w.meta.items) |m| {
            w.gpa.free(m[0]);
            w.gpa.free(m[1]);
        }
        w.meta.deinit(w.gpa);
        w.* = undefined;
    }

    pub fn has(w: *const Writer, name: []const u8) bool {
        for (w.items.items) |it| if (std.mem.eql(u8, it.name, name)) return true;
        return false;
    }

    /// A copy of `data` as tensor `name` of `shape`; a name given twice replaces the first.
    pub fn add(w: *Writer, name: []const u8, dtype: DType, shape: []const u64, data: []const u8) !void {
        if (shape.len > max_rank) return error.BadShape;
        var n: u64 = dtype.size();
        for (shape) |d| n *= d;
        if (n != data.len) return error.BadShape;
        const copy = try w.gpa.dupe(u8, data);
        errdefer w.gpa.free(copy);
        try w.addOwned(name, dtype, shape, copy);
    }

    /// As `add`, taking ownership of `data` (allocated with the writer's allocator).
    pub fn addOwned(w: *Writer, name: []const u8, dtype: DType, shape: []const u64, data: []u8) !void {
        var it: Item = .{ .name = try w.gpa.dupe(u8, name), .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .data = data };
        @memcpy(it.shape[0..shape.len], shape);
        for (w.items.items) |*old| if (std.mem.eql(u8, old.name, name)) {
            w.gpa.free(old.name);
            w.gpa.free(old.data);
            old.* = it;
            return;
        };
        errdefer w.gpa.free(it.name);
        try w.items.append(w.gpa, it);
    }

    pub fn setMeta(w: *Writer, key: []const u8, value: []const u8) !void {
        const v = try w.gpa.dupe(u8, value);
        errdefer w.gpa.free(v);
        for (w.meta.items) |*m| if (std.mem.eql(u8, m[0], key)) {
            w.gpa.free(m[1]);
            m[1] = v;
            return;
        };
        const k = try w.gpa.dupe(u8, key);
        errdefer w.gpa.free(k);
        try w.meta.append(w.gpa, .{ k, v });
    }

    pub fn write(w: *Writer, io: std.Io, path: []const u8) !void {
        std.mem.sort(Item, w.items.items, {}, struct {
            fn lt(_: void, x: Item, y: Item) bool {
                return std.mem.order(u8, x.name, y.name) == .lt;
            }
        }.lt);
        var head: std.ArrayList(u8) = .empty;
        defer head.deinit(w.gpa);
        try head.append(w.gpa, '{');
        var first = true;
        if (w.meta.items.len > 0) {
            try head.appendSlice(w.gpa, "\"__metadata__\":{");
            for (w.meta.items, 0..) |m, i| {
                if (i > 0) try head.append(w.gpa, ',');
                try quote(w.gpa, &head, m[0]);
                try head.append(w.gpa, ':');
                try quote(w.gpa, &head, m[1]);
            }
            try head.append(w.gpa, '}');
            first = false;
        }
        var at: u64 = 0;
        for (w.items.items) |it| {
            if (!first) try head.append(w.gpa, ',');
            first = false;
            try quote(w.gpa, &head, it.name);
            var buf: [64]u8 = undefined;
            try head.appendSlice(w.gpa, try std.fmt.bufPrint(&buf, ":{{\"dtype\":\"{s}\",\"shape\":[", .{it.dtype.name()}));
            for (it.shape[0..it.rank], 0..) |d, i| {
                if (i > 0) try head.append(w.gpa, ',');
                try head.appendSlice(w.gpa, try std.fmt.bufPrint(&buf, "{d}", .{d}));
            }
            try head.appendSlice(w.gpa, try std.fmt.bufPrint(&buf, "],\"data_offsets\":[{d},{d}]}}", .{ at, at + it.data.len }));
            at += it.data.len;
        }
        try head.append(w.gpa, '}');
        while ((8 + head.items.len) % 8 != 0) try head.append(w.gpa, ' ');
        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        var prefix: [8]u8 = undefined;
        std.mem.writeInt(u64, &prefix, head.items.len, .little);
        try file.writePositionalAll(io, &prefix, 0);
        try file.writePositionalAll(io, head.items, 8);
        var off: u64 = 8 + head.items.len;
        for (w.items.items) |it| {
            try file.writePositionalAll(io, it.data, off);
            off += it.data.len;
        }
    }
};

fn quote(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(gpa, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(gpa, "\\\""),
        '\\' => try out.appendSlice(gpa, "\\\\"),
        0...0x1f => {
            var buf: [6]u8 = undefined;
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}));
        },
        else => try out.append(gpa, c),
    };
    try out.append(gpa, '"');
}

/// bf16 bits to f32.
pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// f32 to bf16 bits, round to nearest even (NaN kept quiet).
pub fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7fff +% lsb) >> 16);
}

const testing = std.testing;

test "inkling2 st: a written file reads back by name, with metadata, shapes and bytes" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    var w = Writer.init(testing.allocator);
    defer w.deinit();
    const a = [_]u16{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i32{ -7, 9 };
    try w.add("h.3", .bf16, &.{ 2, 3 }, std.mem.sliceAsBytes(&a));
    try w.add("route.ids.3", .i32, &.{ 1, 2 }, std.mem.sliceAsBytes(&b));
    try w.add("odd \"name\"", .u8, &.{3}, "xyz");
    try w.setMeta("layers", "3,4");
    try testing.expectError(error.BadShape, w.add("bad", .f32, &.{2}, "abc"));
    const dir = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(dir);
    const path = try std.fs.path.join(testing.allocator, &.{ dir, "x.safetensors" });
    defer testing.allocator.free(path);
    try w.write(io, path);
    var r = try Reader.open(testing.allocator, io, path);
    defer r.close();
    try testing.expectEqual(@as(usize, 3), r.entries.len);
    try testing.expectEqualStrings("3,4", r.metaValue("layers").?);
    const e = r.find("h.3").?;
    try testing.expectEqual(@as(u64, 3), e.dim(1));
    const got = try r.alloc(testing.allocator, "h.3", .bf16);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&a), got);
    const ids = try r.alloc(testing.allocator, "route.ids.3", .i32);
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(i32, -7), std.mem.bytesToValue(i32, ids[0..4]));
    try testing.expectError(error.UnexpectedDType, r.alloc(testing.allocator, "h.3", .f32));
    try testing.expect(r.find("odd \"name\"") != null);
    try testing.expect(r.find("missing") == null);
}

test "inkling2 st: bf16 rounding is nearest even" {
    try testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(1.0));
    try testing.expectEqual(@as(u16, 0x3f80), f32ToBf16(@bitCast(@as(u32, 0x3f808000)))); // tie to even (down)
    try testing.expectEqual(@as(u16, 0x3f82), f32ToBf16(@bitCast(@as(u32, 0x3f818000)))); // tie to even (up)
    try testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3f80));
}
