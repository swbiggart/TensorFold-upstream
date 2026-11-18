//! Inkling-2-Small's safetensors inventory: every tensor's file, dtype, shape and byte range, from the shard headers
//! of a checkpoint folder or from the text form tools/zig/inkling2_inventory.py writes (fixtures/inventory.tsv).
const std = @import("std");
const Diag = @import("diag.zig").Diag;
const fail = @import("diag.zig").fail;

/// The dtypes the release stores. Any other refuses the checkpoint.
pub const DType = enum {
    bf16,
    f32,
    u8,
    f8_e4m3,

    pub fn size(d: DType) u64 {
        return switch (d) {
            .bf16 => 2,
            .f32 => 4,
            .u8, .f8_e4m3 => 1,
        };
    }

    pub fn parse(text: []const u8) ?DType {
        const names = .{ .{ "BF16", DType.bf16 }, .{ "F32", DType.f32 }, .{ "U8", DType.u8 }, .{ "F8_E4M3", DType.f8_e4m3 } };
        inline for (names) |n| if (std.mem.eql(u8, text, n[0])) return n[1];
        return null;
    }

    pub fn name(d: DType) []const u8 {
        return switch (d) {
            .bf16 => "BF16",
            .f32 => "F32",
            .u8 => "U8",
            .f8_e4m3 => "F8_E4M3",
        };
    }
};

/// Which manifest section a tensor belongs to: the model's files, or the drafter's mtp.safetensors.
pub const Section = enum { model, mtp };

pub const mtp_file = "mtp.safetensors";

pub const File = struct {
    name: []const u8,
    header_len: u64,

    /// Where the data region starts: tensors' begin and end count from here.
    pub fn dataStart(f: File) u64 {
        return 8 + f.header_len;
    }

    pub fn section(f: File) Section {
        return if (std.mem.eql(u8, f.name, mtp_file)) .mtp else .model;
    }
};

pub const max_rank = 4;

pub const Tensor = struct {
    name: []const u8,
    file: u16,
    dtype: DType,
    rank: u8,
    shape: [max_rank]u64,
    /// Bytes from the file's data start (safetensors' data_offsets).
    begin: u64,
    end: u64,

    pub fn dim(t: Tensor, i: usize) u64 {
        return if (i < t.rank) t.shape[i] else 1;
    }

    pub fn bytes(t: Tensor) u64 {
        return t.end - t.begin;
    }

    pub fn is(t: Tensor, dtype: DType, shape: []const u64) bool {
        if (t.dtype != dtype or t.rank != shape.len) return false;
        for (shape, 0..) |d, i| if (t.shape[i] != d) return false;
        return true;
    }
};

pub const Inventory = struct {
    arena: std.heap.ArenaAllocator,
    files: []File,
    tensors: []Tensor,

    pub fn deinit(inv: *Inventory) void {
        inv.arena.deinit();
        inv.* = undefined;
    }

    pub fn section(inv: *const Inventory, t: Tensor) Section {
        return inv.files[t.file].section();
    }

    /// Absolute byte offset of `t`'s first byte in its file.
    pub fn offset(inv: *const Inventory, t: Tensor) u64 {
        return inv.files[t.file].dataStart() + t.begin;
    }

    pub fn find(inv: *const Inventory, name: []const u8) ?Tensor {
        for (inv.tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    pub fn totalBytes(inv: *const Inventory) u64 {
        var n: u64 = 0;
        for (inv.tensors) |t| n += t.bytes();
        return n;
    }

    /// The text form: `file <index> <name> <header_len>` lines, then `t <file> <name> <dtype> <shape> <begin> <end>`.
    pub fn fromText(gpa: std.mem.Allocator, text: []const u8, d: ?*Diag) !Inventory {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var files: std.ArrayList(File) = .empty;
        var tensors: std.ArrayList(Tensor) = .empty;
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trimEnd(u8, raw, "\r");
            if (line.len == 0 or line[0] == '#') continue;
            var f = std.mem.splitScalar(u8, line, '\t');
            const kind = f.next().?;
            if (std.mem.eql(u8, kind, "file")) {
                const i = try std.fmt.parseInt(usize, f.next() orelse return error.BadInventory, 10);
                if (i != files.items.len) return error.BadInventory;
                const name = try a.dupe(u8, f.next() orelse return error.BadInventory);
                const len = try std.fmt.parseInt(u64, f.next() orelse return error.BadInventory, 10);
                _ = std.math.add(u64, 8, len) catch return fail(d, error.BadInventory, "file {s}: header length {d} puts its data past u64", .{ name, len });
                try files.append(a, .{ .name = name, .header_len = len });
            } else if (std.mem.eql(u8, kind, "t")) {
                const file = try std.fmt.parseInt(u16, f.next() orelse return error.BadInventory, 10);
                if (file >= files.items.len) return error.BadInventory;
                var t: Tensor = .{ .name = try a.dupe(u8, f.next() orelse return error.BadInventory), .file = file, .dtype = undefined, .rank = 0, .shape = @splat(1), .begin = 0, .end = 0 };
                const dt = f.next() orelse return error.BadInventory;
                t.dtype = DType.parse(dt) orelse return fail(d, error.UnsupportedDType, "tensor {s} has dtype {s}, not one the release stores", .{ t.name, dt });
                const dims = f.next() orelse return error.BadInventory;
                if (dims.len > 0) {
                    var parts = std.mem.splitScalar(u8, dims, ',');
                    while (parts.next()) |x| {
                        if (t.rank == max_rank) return error.BadInventory;
                        t.shape[t.rank] = try std.fmt.parseInt(u64, x, 10);
                        t.rank += 1;
                    }
                }
                t.begin = try std.fmt.parseInt(u64, f.next() orelse return error.BadInventory, 10);
                t.end = try std.fmt.parseInt(u64, f.next() orelse return error.BadInventory, 10);
                try checkSize(t, d);
                _ = std.math.add(u64, files.items[file].dataStart(), t.end) catch
                    return fail(d, error.BadInventory, "tensor {s} ends past u64 in its file", .{t.name});
                try tensors.append(a, t);
            } else return error.BadInventory;
        }
        if (files.items.len == 0 or tensors.items.len == 0) return error.BadInventory;
        try checkUnique(a, tensors.items, d);
        try checkDisjoint(a, tensors.items, d);
        return .{ .arena = arena, .files = files.items, .tensors = tensors.items };
    }

    /// The headers of `dir`'s shards (model.safetensors.index.json's files) and of its mtp.safetensors, when present.
    pub fn fromDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, d: ?*Diag) !Inventory {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var names = try shardNames(a, io, dir);
        const mtp = try std.fs.path.join(a, &.{ dir, mtp_file });
        if (std.Io.Dir.cwd().access(io, mtp, .{})) |_| {
            try names.append(a, mtp_file);
        } else |_| {}
        var files: std.ArrayList(File) = .empty;
        var tensors: std.ArrayList(Tensor) = .empty;
        for (names.items, 0..) |name, i| {
            const path = try std.fs.path.join(a, &.{ dir, name });
            const file = try std.Io.Dir.cwd().openFile(io, path, .{});
            defer file.close(io);
            const size = try file.length(io);
            var prefix: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &prefix, 0) != prefix.len) return error.BadSafetensors;
            const n = std.mem.readInt(u64, &prefix, .little);
            if (n > 100_000_000 or size < 8 or n > size - 8) return error.BadSafetensors;
            const text = try a.alloc(u8, @intCast(n));
            if (try file.readPositionalAll(io, text, 8) != text.len) return error.BadSafetensors;
            try files.append(a, .{ .name = name, .header_len = n });
            try parseHeader(a, text, @intCast(i), size - 8 - n, &tensors, d);
        }
        try checkUnique(a, tensors.items, d);
        try checkDisjoint(a, tensors.items, d);
        return .{ .arena = arena, .files = files.items, .tensors = tensors.items };
    }
};

/// model.safetensors.index.json's shard names, sorted.
fn shardNames(a: std.mem.Allocator, io: std.Io, dir: []const u8) !std.ArrayList([]const u8) {
    const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (doc != .object) return error.BadIndex;
    const map = doc.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var out: std.ArrayList([]const u8) = .empty;
    var it = map.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .string) return error.BadIndex;
        const f = e.value_ptr.string;
        if (std.mem.indexOfAny(u8, f, "/\\") != null or std.mem.eql(u8, f, "..")) return error.BadIndex;
        if ((try seen.getOrPut(a, f)).found_existing) continue;
        try out.append(a, f);
    }
    std.mem.sort([]const u8, out.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    return out;
}

fn parseHeader(a: std.mem.Allocator, text: []const u8, file: u16, data_len: u64, out: *std.ArrayList(Tensor), d: ?*Diag) !void {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (doc != .object) return error.BadSafetensors;
    var it = doc.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const v = kv.value_ptr.*;
        if (v != .object) return error.BadSafetensors;
        const dt = v.object.get("dtype") orelse return error.BadSafetensors;
        if (dt != .string) return error.BadSafetensors;
        var t: Tensor = .{ .name = kv.key_ptr.*, .file = file, .dtype = undefined, .rank = 0, .shape = @splat(1), .begin = 0, .end = 0 };
        t.dtype = DType.parse(dt.string) orelse return fail(d, error.UnsupportedDType, "tensor {s} has dtype {s}, not one the release stores", .{ t.name, dt.string });
        const shape = v.object.get("shape") orelse return error.BadSafetensors;
        if (shape != .array or shape.array.items.len > max_rank) return error.BadSafetensors;
        for (shape.array.items, 0..) |dim, i| {
            if (dim != .integer or dim.integer < 0) return error.BadSafetensors;
            t.shape[i] = @intCast(dim.integer);
        }
        t.rank = @intCast(shape.array.items.len);
        const offs = v.object.get("data_offsets") orelse return error.BadSafetensors;
        if (offs != .array or offs.array.items.len != 2) return error.BadSafetensors;
        for (offs.array.items) |o| if (o != .integer or o.integer < 0) return error.BadSafetensors;
        t.begin = @intCast(offs.array.items[0].integer);
        t.end = @intCast(offs.array.items[1].integer);
        if (t.end > data_len) return fail(d, error.BadSafetensors, "tensor {s} ends past its file", .{t.name});
        try checkSize(t, d);
        try out.append(a, t);
    }
}

fn checkSize(t: Tensor, d: ?*Diag) !void {
    var n: u64 = t.dtype.size();
    for (t.shape[0..t.rank]) |dim| n = std.math.mul(u64, n, dim) catch return error.BadSafetensors;
    if (t.end < t.begin or t.end - t.begin != n)
        return fail(d, error.BadSafetensors, "tensor {s} spans {d} bytes, its shape {d}", .{ t.name, t.end -| t.begin, n });
}

fn checkUnique(a: std.mem.Allocator, tensors: []const Tensor, d: ?*Diag) !void {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(a);
    for (tensors) |t| if ((try seen.getOrPut(a, t.name)).found_existing)
        return fail(d, error.DuplicateTensor, "tensor {s} is in two files", .{t.name});
}

/// No two tensors of a file share a byte: sorted by file and begin, each starts at or past the one before's end.
fn checkDisjoint(a: std.mem.Allocator, tensors: []const Tensor, d: ?*Diag) !void {
    const order = try a.alloc(u32, tensors.len);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, tensors, struct {
        fn lt(ts: []const Tensor, x: u32, y: u32) bool {
            const p = ts[x];
            const q = ts[y];
            if (p.file != q.file) return p.file < q.file;
            return if (p.begin != q.begin) p.begin < q.begin else p.end < q.end;
        }
    }.lt);
    for (order[0..order.len -| 1], order[@min(1, order.len)..]) |i, j| {
        const p = tensors[i];
        const q = tensors[j];
        if (p.file == q.file and q.begin < p.end)
            return fail(d, error.BadSafetensors, "tensors {s} and {s} overlap in their file", .{ p.name, q.name });
    }
}

pub const fixture = @embedFile("fixtures/inventory.tsv");

test "inkling2 inventory: the release's 1,854 tensors in 25 files, 393.5 GB" {
    var inv = try Inventory.fromText(std.testing.allocator, fixture, null);
    defer inv.deinit();
    try std.testing.expectEqual(@as(usize, 25), inv.files.len);
    try std.testing.expectEqual(@as(usize, 1854), inv.tensors.len);
    try std.testing.expectEqual(@as(u64, 393_479_869_088), inv.totalBytes());
    const w13 = inv.find("model.llm.layers.0.mlp.experts.w13_weight").?;
    try std.testing.expect(w13.is(.u8, &.{ 384, 4608, 2304 }));
    const mx = inv.find("model.llm.layers.47.mlp.experts.w2_weight").?;
    try std.testing.expect(mx.is(.f8_e4m3, &.{ 384, 4608, 2304 }));
    try std.testing.expectEqual(Section.mtp, inv.section(inv.find("markov_head.markov_w1.weight").?));
    try std.testing.expectEqual(Section.model, inv.section(w13));
    try std.testing.expectEqual(@as(u64, 8 + 17504), inv.offset(inv.find("model.llm.attn_res.attn_queries.0").?));
}

test "inkling2 inventory: a size that disagrees with the shape, an unknown dtype or a repeated name is refused" {
    const head = "file\t0\tx.safetensors\t10\n";
    try std.testing.expectError(error.BadSafetensors, Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tBF16\t4\t0\t6\n", null));
    try std.testing.expectError(error.UnsupportedDType, Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tF16\t4\t0\t8\n", null));
    try std.testing.expectError(error.DuplicateTensor, Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tBF16\t4\t0\t8\nt\t0\ta\tBF16\t4\t8\t16\n", null));
    var ok = try Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tBF16\t4\t0\t8\n", null);
    ok.deinit();
}

fn testShard(dir: std.Io.Dir, io: std.Io, name: []const u8, header: []const u8, data: usize) !void {
    var buf: [512]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memset(buf[8 + header.len ..][0..data], 0);
    try dir.writeFile(io, .{ .sub_path = name, .data = buf[0 .. 8 + header.len + data] });
}

test "inkling2 inventory: a folder's shard headers in index order, then mtp.safetensors" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"metadata\":{},\"weight_map\":{\"b\":\"model-00002-of-00002.safetensors\",\"a\":\"model-00001-of-00002.safetensors\"}}" });
    const first = "{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]}}";
    try testShard(tmp.dir, io, "model-00001-of-00002.safetensors", first, 12);
    try testShard(tmp.dir, io, "model-00002-of-00002.safetensors", "{\"b\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[0,4]}}", 4);
    try testShard(tmp.dir, io, mtp_file, "{\"fuse_weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 4);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    var inv = try Inventory.fromDir(std.testing.allocator, io, dir, null);
    defer inv.deinit();
    try std.testing.expectEqual(@as(usize, 3), inv.files.len);
    try std.testing.expectEqualStrings("model-00001-of-00002.safetensors", inv.files[0].name);
    try std.testing.expectEqualStrings(mtp_file, inv.files[2].name);
    const a = inv.find("a").?;
    try std.testing.expect(a.is(.bf16, &.{ 2, 3 }));
    try std.testing.expectEqual(@as(u64, 8 + first.len), inv.offset(a));
    try std.testing.expectEqual(Section.mtp, inv.section(inv.find("fuse_weight").?));
    // a tensor past its file's end
    try testShard(tmp.dir, io, "model-00002-of-00002.safetensors", "{\"b\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[0,4]}}", 2);
    var d: Diag = .{};
    try std.testing.expectError(error.BadSafetensors, Inventory.fromDir(std.testing.allocator, io, dir, &d));
    try std.testing.expect(std.mem.indexOf(u8, d.text(), "ends past its file") != null);
}

test "inkling2 inventory: two tensors on one byte range of a file are refused" {
    const head = "file\t0\tx.safetensors\t10\n";
    var d: Diag = .{};
    // b starts inside a, then the same range twice
    try std.testing.expectError(error.BadSafetensors, Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tBF16\t4\t0\t8\nt\t0\tb\tBF16\t2\t4\t8\n", &d));
    try std.testing.expect(std.mem.indexOf(u8, d.text(), "overlap") != null);
    try std.testing.expectError(error.BadSafetensors, Inventory.fromText(std.testing.allocator, head ++ "t\t0\tb\tBF16\t4\t0\t8\nt\t0\ta\tBF16\t4\t0\t8\n", null));
    // a long tensor that covers a later one
    try std.testing.expectError(error.BadSafetensors, Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tBF16\t8\t0\t16\nt\t0\tb\tBF16\t1\t4\t6\nt\t0\tc\tBF16\t1\t10\t12\n", null));
    // the same range in two files is two tensors
    var ok = try Inventory.fromText(std.testing.allocator, head ++ "file\t1\ty.safetensors\t10\nt\t0\ta\tBF16\t4\t0\t8\nt\t1\tb\tBF16\t4\t0\t8\n", null);
    ok.deinit();
}

test "inkling2 inventory: a header length or byte range whose file offset passes u64 is refused" {
    const head = "file\t0\tx.safetensors\t10\n";
    var d: Diag = .{};
    try std.testing.expectError(error.BadInventory, Inventory.fromText(std.testing.allocator, "file\t0\tx.safetensors\t18446744073709551615\nt\t0\ta\tBF16\t4\t0\t8\n", &d));
    try std.testing.expectError(error.BadInventory, Inventory.fromText(std.testing.allocator, head ++ "t\t0\ta\tBF16\t4\t18446744073709551600\t18446744073709551608\n", &d));
    try std.testing.expect(std.mem.indexOf(u8, d.text(), "past") != null);
}

test "inkling2 inventory: a shard header whose tensors share bytes is refused" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"metadata\":{},\"weight_map\":{\"a\":\"model-00001-of-00001.safetensors\",\"b\":\"model-00001-of-00001.safetensors\"}}" });
    try testShard(tmp.dir, io, "model-00001-of-00001.safetensors", "{\"a\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[4,12]}}", 12);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    var d: Diag = .{};
    try std.testing.expectError(error.BadSafetensors, Inventory.fromDir(std.testing.allocator, io, dir, &d));
    try std.testing.expect(std.mem.indexOf(u8, d.text(), "overlap") != null);
}
