//! The decoder against what 0.6.6 decoded (LFM2's picture goldens and the edge pictures), and against damaged bytes.

const std = @import("std");
const picture = @import("picture.zig");

const Case = struct {
    name: []const u8,
    url: []const u8,
    problem: ?[]const u8 = null,
    width: usize = 0,
    height: usize = 0,
    rgb: []const u8 = "",
};

const Golden = struct { cases: []const Case };

const prefix = ";base64,";

fn goldens(arena: std.mem.Allocator) !Golden {
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/lfm2/images.json", arena, .limited(1 << 22));
    return std.json.parseFromSliceLeaky(Golden, arena, text, .{ .ignore_unknown_fields = true });
}

/// A golden's bytes when its URL is a plain base64 data URL; the URL refusals are the caller's to test.
fn payload(arena: std.mem.Allocator, url: []const u8) !?[]u8 {
    if (!std.mem.startsWith(u8, url, "data:")) return null;
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return null;
    const media = url[5..comma];
    if (!std.mem.endsWith(u8, media, prefix[0 .. prefix.len - 1]) or std.mem.count(u8, media, ";") != 1) return null;
    const b64 = std.base64.standard.Decoder;
    const out = try arena.alloc(u8, b64.calcSizeForSlice(url[comma + 1 ..]) catch return null);
    b64.decode(out, url[comma + 1 ..]) catch return null;
    return out;
}

/// JPEG cases need the pinned library: required if TENSORFOLD_LIBTURBOJPEG names one, else the test reports a skip.
fn jpegReady(a: std.mem.Allocator) !bool {
    const level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = level;
    if (picture.jpeg.available(std.testing.io, a)) return true;
    if (@import("sys").getenv(picture.jpeg.env) != null) return error.PinnedLibraryDidNotLoad;
    return false;
}

fn sha256(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

test "every picture 0.6.6 decoded decodes to the same RGB bytes, and its refusals read the same" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const golden = try goldens(arena);
    const jpeg = try jpegReady(a);
    var checked: [2]usize = .{ 0, 0 }; // decoded, refused
    for (golden.cases) |case| {
        errdefer std.debug.print("case {s}\n", .{case.name});
        const bytes = (try payload(arena, case.url)) orelse continue; // not a base64 picture: the caller's refusal
        if (bytes.len == 0) continue;
        if (picture.sniff(bytes) == .jpeg and !jpeg) continue;
        var rgb = picture.decode(std.testing.io, a, bytes) catch |err| {
            try std.testing.expectEqualStrings(case.problem orelse @errorName(err), picture.problem(err));
            checked[1] += 1;
            continue;
        };
        defer rgb.deinit(a);
        try std.testing.expect(case.problem == null);
        try std.testing.expectEqual(case.width, rgb.width);
        try std.testing.expectEqual(case.height, rgb.height);
        try std.testing.expectEqualStrings(case.rgb, &sha256(rgb.pixels));
        checked[0] += 1;
    }
    try std.testing.expect(checked[0] >= @as(usize, if (jpeg) 29 else 19) and checked[1] >= 4);
    if (!jpeg) return error.SkipZigTest;
}

const Edge = struct { name: []const u8, b64: []const u8, width: usize = 0, height: usize = 0, rgb: ?[]const u8 = null, problem: ?[]const u8 = null };

/// Where this decoder parts from 0.6.6 on purpose: a CMYK JPEG is refused by name; a two-picture MPO gives its first.
const departures = [_]struct { name: []const u8, outcome: []const u8 }{
    .{ .name = "jpeg_cmyk", .outcome = "CMYK JPEG images are unsupported here; send an RGB or grayscale JPEG, or a PNG" },
    .{ .name = "jpeg_mpo_two", .outcome = "40x30 89ec4fbf73b19a585760f7293a178a3f02c38fe177491362e1d84fd3fdd5be70" },
};

test "edge pictures (zig/tests/fixtures/picture/edges.json) get 0.6.6's outcome, but for the named departures" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/picture/edges.json", arena, .limited(1 << 22));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const Edge }, arena, text, .{ .ignore_unknown_fields = true })).cases;
    const jpeg = try jpegReady(a);
    var checked: usize = 0;
    for (cases) |c| {
        errdefer std.debug.print("case {s}\n", .{c.name});
        const b64 = std.base64.standard.Decoder;
        const bytes = try arena.alloc(u8, try b64.calcSizeForSlice(c.b64));
        try b64.decode(bytes, c.b64);
        const ours = if (picture.decode(std.testing.io, a, bytes)) |got| blk: {
            var rgb = got;
            defer rgb.deinit(a);
            break :blk try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ rgb.width, rgb.height, &sha256(rgb.pixels) });
        } else |err| blk: {
            if (!jpeg and (err == error.NoJpegLibrary or err == error.JpegLibraryVersion)) break :blk null;
            break :blk picture.problem(err);
        };
        const got = ours orelse continue; // a JPEG that needs the library this host lacks
        var want = if (c.rgb) |h| try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ c.width, c.height, h }) else c.problem.?;
        for (departures) |d| if (std.mem.eql(u8, d.name, c.name)) {
            want = d.outcome;
        };
        try std.testing.expectEqualStrings(want, got);
        checked += 1;
    }
    try std.testing.expect(checked >= @as(usize, if (jpeg) 76 else 50));
    if (!jpeg) return error.SkipZigTest;
}

/// Small JPEGs built by hand: 8-bit, one quantization table of ones, one-code Huffman tables (DC 0, then end of block).
const crafted = struct {
    const soi = "\xff\xd8";
    const eoi = "\xff\xd9";
    const junk = "\x00"; // a stray byte before a marker: libjpeg's "extraneous bytes" warning
    const dqt = "\xff\xdb\x00\x43\x00" ++ @as([64]u8, @splat(1));
    const dht = "\xff\xc4\x00\x14\x00\x01" ++ @as([15]u8, @splat(0)) ++ "\x00" ++ "\xff\xc4\x00\x14\x10\x01" ++ @as([15]u8, @splat(0)) ++ "\x00";
    const gray8 = "\xff\xc0\x00\x0b\x08\x00\x08\x00\x08\x01\x01\x11\x00";
    const gray16 = "\xff\xc0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00";
    const progressive16 = "\xff\xc2\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00";
    const rgb8 = "\xff\xc0\x00\x11\x08\x00\x08\x00\x08\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00";
    const scan_gray = "\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00";
    const scan_dc = "\xff\xda\x00\x08\x01\x01\x00\x00\x00\x00"; // a progressive first scan: DC only
    const scan_rgb = "\xff\xda\x00\x0c\x03\x01\x00\x02\x00\x03\x00\x00\x3f\x00";

    /// libjpeg stops on a fatal error after a warning, and TurboJPEG reports it as a warning: no pixel is written.
    const unfinished = [_][]const u8{
        soi ++ gray8 ++ junk ++ scan_gray ++ eoi, // no quantization table, refused only as the pixels start
        soi ++ rgb8 ++ junk ++ scan_rgb ++ eoi,
        soi ++ dqt ++ progressive16 ++ dht ++ scan_dc ++ gray16 ++ eoi, // the scan ends at once at a second frame
        soi ++ dqt ++ junk ++ progressive16 ++ dht ++ scan_dc ++ "\x0f" ++ gray16 ++ eoi,
    };

    /// Warned, but libjpeg finishes every row: these still decode, all mid-gray (each block's DC is 0).
    const finished = [_][]const u8{
        soi ++ dqt ++ junk ++ gray16 ++ dht ++ scan_gray ++ "\x00" ++ eoi,
        soi ++ dqt ++ gray16 ++ dht ++ scan_gray ++ eoi, // the scan is empty: its blocks read as zeros
        soi ++ dqt ++ gray16 ++ dht ++ scan_gray ++ gray16 ++ eoi, // a second frame only after the last row
        soi ++ dqt ++ gray16 ++ dht ++ "\xff\xdd\x00\x04\x00\x01" ++ scan_gray ++ "\x00\xff\xd3\x00" ++ eoi, // a wrong restart
    };
};

test "a JPEG libjpeg gives up on after a warning is refused, never returned with pixels it did not write" {
    const a = std.testing.allocator;
    if (!try jpegReady(a)) return error.SkipZigTest;
    var decoded: usize = 0;
    for (crafted.unfinished, 0..) |bytes, i| {
        var rgb = picture.decode(std.testing.io, a, bytes) catch |err| {
            try std.testing.expectEqual(error.Invalid, err);
            continue;
        };
        defer rgb.deinit(a);
        // a safe build paints new memory 0xaa: the bytes the decoder left as it found them
        const painted = std.mem.count(u8, rgb.pixels, "\xaa");
        std.debug.print("case {d}: {d}x{d} decoded, {d} of {d} bytes unwritten\n", .{ i, rgb.width, rgb.height, painted, rgb.pixels.len });
        decoded += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), decoded);
}

test "warned JPEGs libjpeg finishes still decode, every pixel written" {
    const a = std.testing.allocator;
    if (!try jpegReady(a)) return error.SkipZigTest;
    for (crafted.finished, 0..) |bytes, i| {
        errdefer std.debug.print("case {d}\n", .{i});
        var rgb = try picture.decode(std.testing.io, a, bytes);
        defer rgb.deinit(a);
        try std.testing.expectEqual(@as(usize, 16), rgb.width);
        try std.testing.expectEqual(@as(usize, 16), rgb.height);
        for (rgb.pixels) |v| try std.testing.expectEqual(@as(u8, 128), v);
    }
}

test "PNG tRNS keys compare their low bytes, except mode 1 keeps the whole key" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/picture/trns.json", a, .limited(1 << 16));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const Edge }, a, text, .{ .ignore_unknown_fields = true })).cases;
    try std.testing.expectEqual(@as(usize, 5), cases.len);
    for (cases) |c| {
        errdefer std.debug.print("case {s}\n", .{c.name});
        const b64 = std.base64.standard.Decoder;
        const bytes = try a.alloc(u8, try b64.calcSizeForSlice(c.b64));
        try b64.decode(bytes, c.b64);
        var rgb = try picture.decode(std.testing.io, std.testing.allocator, bytes);
        defer rgb.deinit(std.testing.allocator);
        try std.testing.expectEqual(c.width, rgb.width);
        try std.testing.expectEqual(c.height, rgb.height);
        try std.testing.expectEqualStrings(c.rgb.?, &sha256(rgb.pixels));
    }
}

/// WebP cases need the pinned libwebp: required if TENSORFOLD_LIBWEBP names one, else the test reports a skip.
fn webpReady(a: std.mem.Allocator) !bool {
    const level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = level;
    if (picture.webp.available(std.testing.io, a)) return true;
    if (@import("sys").getenv(picture.webp.env) != null) return error.PinnedLibraryDidNotLoad;
    return false;
}

test "WebP pictures (zig/tests/fixtures/picture/webp.json) decode to Pillow's RGB bytes, and its refusals read the same" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/picture/webp.json", arena, .limited(1 << 22));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const Edge }, arena, text, .{ .ignore_unknown_fields = true })).cases;
    const ready = try webpReady(a);
    var checked: usize = 0;
    for (cases) |c| {
        errdefer std.debug.print("case {s}\n", .{c.name});
        const b64 = std.base64.standard.Decoder;
        const bytes = try arena.alloc(u8, try b64.calcSizeForSlice(c.b64));
        try b64.decode(bytes, c.b64);
        try std.testing.expectEqual(picture.Format.webp, picture.sniff(bytes));
        const got = if (picture.decode(std.testing.io, a, bytes)) |d| blk: {
            var rgb = d;
            defer rgb.deinit(a);
            break :blk try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ rgb.width, rgb.height, &sha256(rgb.pixels) });
        } else |err| blk: {
            if (!ready and (err == error.NoWebpLibrary or err == error.WebpLibraryVersion)) continue;
            break :blk picture.problem(err);
        };
        const want = if (c.rgb) |h| try std.fmt.allocPrint(arena, "{d}x{d} {s}", .{ c.width, c.height, h }) else c.problem.?;
        try std.testing.expectEqualStrings(want, got);
        checked += 1;
    }
    try std.testing.expect(checked >= @as(usize, if (ready) 8 else 2)); // without the library: the files refused before it loads
    if (!ready) return error.SkipZigTest;
}

test "damaged and cut pictures are refused or decoded, never trapped" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = std.testing.allocator;
    const golden = try goldens(arena);
    const jpeg = try jpegReady(a);
    var prng = std.Random.DefaultPrng.init(0x70696374);
    const r = prng.random();
    var outcomes: [2]usize = .{ 0, 0 };
    for (golden.cases) |case| {
        if (case.problem != null or case.width * case.height > 40_000) continue;
        const clean = (try payload(arena, case.url)) orelse continue;
        if (picture.sniff(clean) == .jpeg and !jpeg) continue;
        for (0..80) |i| {
            const v = try a.dupe(u8, clean);
            defer a.free(v);
            var bytes = v;
            if (i % 4 == 0) {
                bytes = v[0..r.uintLessThan(usize, v.len)];
            } else for (0..1 + r.uintLessThan(usize, 4)) |_| {
                v[r.uintLessThan(usize, if (r.boolean()) @min(v.len, 600) else v.len)] = r.int(u8);
            }
            if (i % 2 == 1 and picture.sniff(bytes) == .png) refreshCrcs(bytes);
            var rgb = picture.decode(std.testing.io, a, bytes) catch {
                outcomes[0] += 1;
                continue;
            };
            rgb.deinit(a);
            outcomes[1] += 1;
        }
    }
    try std.testing.expect(outcomes[0] > 0 and outcomes[1] > 0);
    if (!jpeg) return error.SkipZigTest;
}

/// Each whole chunk's CRC rewritten, so a damaged body reaches the decoder past its checksum.
fn refreshCrcs(png: []u8) void {
    var at: usize = 8;
    while (png.len >= at + 12) {
        const len = std.mem.readInt(u32, png[at..][0..4], .big);
        if (len > png.len - at - 12) return;
        std.mem.writeInt(u32, png[at + 8 + len ..][0..4], std.hash.Crc32.hash(png[at + 4 .. at + 8 + len]), .big);
        at += 12 + len;
    }
}

test "sizes are judged as 0.6.6 judged them" {
    try picture.fits(8192, 2048);
    try std.testing.expectError(error.Oversized, picture.fits(8193, 1));
    try std.testing.expectError(error.Invalid, picture.fits(0, 5)); // Pillow's open found no image
    try std.testing.expectError(error.Oversized, picture.fits(4097, 4096));
    try std.testing.expectError(error.Invalid, picture.fits(9000, 10000)); // Pillow's bomb bound first
    try std.testing.expectError(error.Invalid, picture.fits(0, 100_000_000));
}

test "explicit decode limits admit a large source while retaining defaults and the absolute bomb bound" {
    const larger: picture.Limits = .{ .pixels = 32 << 20 };
    try std.testing.expectError(error.Oversized, picture.fits(5184, 3456));
    try larger.fits(5184, 3456);
    try std.testing.expectError(error.Oversized, larger.fits(8193, 1));
    try std.testing.expectError(error.Oversized, larger.fits(8192, 4097));
    try std.testing.expectError(error.Invalid, larger.fits(std.math.maxInt(u64), 2));
    try std.testing.expectError(error.Invalid, (picture.Limits{ .side = 100000, .pixels = 100000000 }).fits(9000, 10000));
    try std.testing.expectError(error.Invalid, (picture.Limits{ .pixels = 0 }).fits(1, 1));
}

test "caller decode limits reach every admitted image header before pixel allocation" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const golden = try goldens(arena);
    var checked: usize = 0;
    for (golden.cases) |case| {
        if (case.problem != null or case.width * case.height <= 1) continue;
        const bytes = (try payload(arena, case.url)) orelse continue;
        const format = picture.sniff(bytes);
        if (format == .jpeg and !picture.jpeg.available(std.testing.io, arena)) continue;
        if (format == .webp and !picture.webp.available(std.testing.io, arena)) continue;
        if (format == .unknown) continue;
        try std.testing.expectError(error.Oversized, picture.decodeWithLimits(std.testing.io, std.testing.allocator, bytes, .{ .pixels = 1 }));
        checked += 1;
    }
    try std.testing.expect(checked > 0);
}

test "alpha over white keeps opaque values and whitens clear ones" {
    for (0..256) |c| try std.testing.expectEqual(@as(u8, @intCast(c)), picture.overWhite(@intCast(c), 255));
    try std.testing.expectEqual(@as(u8, 255), picture.overWhite(0, 0));
    try std.testing.expectEqual(@as(u8, 132), picture.overWhite(10, 128));
}

test "orientations transpose as Pillow's do" {
    const a = std.testing.allocator;
    var px: [18]u8 = undefined; // 3 x 2, each pixel its index: 0 1 2 / 3 4 5
    for (0..6) |i| @memset(px[i * 3 ..][0..3], @intCast(i));
    const want = [_][]const u8{
        &.{ 2, 1, 0, 5, 4, 3 }, &.{ 5, 4, 3, 2, 1, 0 }, &.{ 3, 4, 5, 0, 1, 2 }, &.{ 0, 3, 1, 4, 2, 5 },
        &.{ 3, 0, 4, 1, 5, 2 }, &.{ 5, 2, 4, 1, 3, 0 }, &.{ 2, 5, 1, 4, 0, 3 },
    };
    for (want, 2..) |w, o| {
        var out = try picture.finish(a, .{ .width = 3, .height = 2, .channels = 3, .data = &px, .orientation = @intCast(o) });
        defer out.deinit(a);
        for (w, 0..) |v, i| try std.testing.expectEqual(v, out.pixels[i * 3]);
    }
}

/// One PNG chunk with its length and checksum.
fn pngChunk(out: *std.ArrayList(u8), a: std.mem.Allocator, kind: *const [4]u8, body: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(body.len), .big);
    try out.appendSlice(a, &len);
    try out.appendSlice(a, kind);
    try out.appendSlice(a, body);
    var crc = std.hash.Crc32.init();
    crc.update(kind);
    crc.update(body);
    var c: [4]u8 = undefined;
    std.mem.writeInt(u32, &c, crc.final(), .big);
    try out.appendSlice(a, &c);
}

/// An allocator that fails once more than `budget` bytes are live.
const Budget = struct {
    child: std.mem.Allocator,
    budget: usize,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(b: *Budget) std.mem.Allocator {
        return .{ .ptr = b, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const b: *Budget = @ptrCast(@alignCast(ctx));
        if (b.live + n > b.budget) return null;
        const p = b.child.rawAlloc(n, al, ra) orelse return null;
        b.live += n;
        b.peak = @max(b.peak, b.live);
        return p;
    }
    fn resize(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) bool {
        const b: *Budget = @ptrCast(@alignCast(ctx));
        if (n > m.len and b.live + n - m.len > b.budget) return false;
        if (!b.child.rawResize(m, al, n, ra)) return false;
        b.live = b.live + n - m.len;
        b.peak = @max(b.peak, b.live);
        return true;
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, ra: usize) void {
        const b: *Budget = @ptrCast(@alignCast(ctx));
        b.child.rawFree(m, al, ra);
        b.live -= m.len;
    }
};

test "PNG text chunks inflate through one window and buffer: 80 MB of kept zTXt is refused as Pillow refuses it, in bounded memory" {
    const a = std.testing.allocator;
    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(a);
    try png.appendSlice(a, picture.png.signature);
    try pngChunk(&png, a, "IHDR", "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00");
    // 40,000 zero bytes in 61 bytes of zlib: 2,000 of them are 80 MB of text Pillow keeps and counts (Latin-1)
    const zeros = "\x78\xda\xed\xc1\x31\x01\x00\x00\x00\xc2\xa0\xf5\x4f\xed\x65\x0b\xa0" ++ @as([39]u8, @splat(0)) ++ "\x80\x1b\x9c\x40\x00\x01";
    for (0..2000) |_| try pngChunk(&png, a, "zTXt", "note\x00\x00" ++ zeros);
    try pngChunk(&png, a, "IDAT", "\x78\xda\x63\x60\x64\x62\x06\x00\x00\x0e\x00\x07");
    try pngChunk(&png, a, "IEND", "");
    var budget: Budget = .{ .child = a, .budget = 8 << 20 };
    // refused as Pillow refuses past 64 MiB of kept text, with the memory bounded (a window and a 1 MiB buffer),
    // not out of memory from a window and a value per chunk
    try std.testing.expectError(error.Invalid, picture.png.decode(budget.allocator(), png.items));
    try std.testing.expect(budget.peak < 2 << 20);
    // 1,000 such chunks (40 MB) stay under the bound: the picture decodes, in the same memory
    png.clearRetainingCapacity();
    try png.appendSlice(a, picture.png.signature);
    try pngChunk(&png, a, "IHDR", "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00");
    for (0..1000) |_| try pngChunk(&png, a, "zTXt", "note\x00\x00" ++ zeros);
    try pngChunk(&png, a, "IDAT", "\x78\xda\x63\x60\x64\x62\x06\x00\x00\x0e\x00\x07");
    try pngChunk(&png, a, "IEND", "");
    budget = .{ .child = a, .budget = 8 << 20 };
    const s = try picture.png.decode(budget.allocator(), png.items);
    defer budget.allocator().free(s.data);
    try std.testing.expectEqualSlices(u8, "\x01\x02\x03\xff", s.data);
    try std.testing.expect(budget.peak < 2 << 20);
}

test "a token budget is checked on the size the codec decodes, before its pixels" {
    var seen: [2]u64 = .{ 0, 0 };
    const l: picture.Limits = .{ .side = 4096, .pixels = 16 << 20, .tokens = 49, .block = 40, .seen = &seen };
    try l.fits(40 * 49, 40);
    try std.testing.expectError(error.Oversized, l.fits(2000, 20)); // 50 blocks of 40
    try std.testing.expectEqual([2]u64{ 2000, 20 }, seen);
    try (picture.Limits{}).fits(8192, 2048); // no token budget: as before
    // a block near the u64 limit: one block a side, no overflow
    const wide: picture.Limits = .{ .tokens = 1, .block = std.math.maxInt(u64) };
    try wide.fits(4096, 4096);
    try std.testing.expectError(error.Oversized, (picture.Limits{ .tokens = 1, .block = 1 }).fits(2, 1));
}

/// zlib of 1 MiB of 0xff (not UTF-8: Pillow inflates such an iTXt value, then discards it uncounted)
const not_utf8_mib = "\x78\xda\xed\xc1\x31\x01\x00\x00\x00\xc2\xa0\xfe\xa9\x67\x08\x5f\xa0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3e\x03\x8e\x88\xef\x11";
/// zlib of 262,144 two-byte characters (512 KiB of UTF-8)
const xmp_512k = "\x78\xda\xed\xc2\x81\x0c\x00\x00\x00\x02\x30\x7f\x94\x38\x12\xcb\x20\x82\xef\x6f\x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x70\x0d\x2e\x18\x55\x51";

fn itxtPng(a: std.mem.Allocator, discarded: usize, with_xmp: bool) !std.ArrayList(u8) {
    var png: std.ArrayList(u8) = .empty;
    try png.appendSlice(a, picture.png.signature);
    try pngChunk(&png, a, "IHDR", "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00");
    for (0..discarded) |_| try pngChunk(&png, a, "iTXt", "note\x00\x01\x00\x00\x00" ++ not_utf8_mib);
    if (with_xmp) try pngChunk(&png, a, "iTXt", "XML:com.adobe.xmp\x00\x01\x00\x00\x00" ++ xmp_512k);
    try pngChunk(&png, a, "IDAT", "\x78\xda\x63\x60\x64\x62\x06\x00\x00\x0e\x00\x07");
    try pngChunk(&png, a, "IEND", "");
    return png;
}

test "PNG text Pillow discards is never counted: 64 MiB of undecodable iTXt plus 512 KiB of kept XMP decodes (Pillow 12.3 opens it)" {
    const a = std.testing.allocator;
    var png = try itxtPng(a, 64, true);
    defer png.deinit(a);
    var budget: Budget = .{ .child = a, .budget = 8 << 20 };
    const s = try picture.png.decode(budget.allocator(), png.items);
    defer budget.allocator().free(s.data);
    try std.testing.expectEqualSlices(u8, "\x01\x02\x03\xff", s.data);
    try std.testing.expect(budget.peak < 4 << 20); // one window, one buffer, the kept XMP
    // 80 MiB of it, past the 64 MiB bound, decodes too: the bound counts kept text only, as Pillow's does
    var more = try itxtPng(a, 80, false);
    defer more.deinit(a);
    const t = try picture.png.decode(a, more.items);
    a.free(t.data);
}
