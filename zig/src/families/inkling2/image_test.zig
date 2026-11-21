//! Gate V1: Inkling-2's image processor against Transformers' Inkling2ImageProcessor on a fixture set.
//!
//! zig/tests/fixtures/inkling2/images.json is the small set (tools/zig/inkling2_image_fixtures.py, pictures inlined);
//! TF_INK2_IMAGE_FIXTURES names a large set's json (its pictures beside it). For every case: the upright size, the
//! RGB after compositing and the fp32 pixel values (sha256 of their little-endian bytes) equal the processor's, or the
//! picture is refused for the reason the oracle names. JPEG and WebP cases need the pinned libraries: required when
//! TENSORFOLD_LIBTURBOJPEG / TENSORFOLD_LIBWEBP name one, else those cases are counted as skipped.
const std = @import("std");
const picture = @import("picture");
const image = @import("image.zig");
const steps = image.For(picture);

const Case = struct {
    name: []const u8,
    bytes: []const u8,
    b64: ?[]const u8 = null,
    file: ?[]const u8 = null,
    width: ?u32 = null,
    height: ?u32 = null,
    grid: ?[2]u32 = null,
    rgb: ?[]const u8 = null,
    values: ?[]const u8 = null,
    problem: ?[]const u8 = null,
};

fn hex(d: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(d, .lower);
}

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return hex(d);
}

/// Whether a codec's pinned library loads here; an error when the environment names one that does not.
fn ready(comptime codec: type, a: std.mem.Allocator) !bool {
    const level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = level;
    if (codec.available(std.testing.io, a)) return true;
    if (@import("sys").getenv(codec.env) != null) return error.PinnedLibraryDidNotLoad;
    return false;
}

const Tally = struct { checked: usize = 0, skipped: usize = 0, refused: usize = 0 };

fn run(arena: std.mem.Allocator, json_path: []const u8, dir: ?[]const u8) !Tally {
    const a = std.testing.allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, json_path, arena, .limited(1 << 26));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const Case }, arena, text, .{ .ignore_unknown_fields = true })).cases;
    const jpeg = try ready(picture.jpeg, a);
    const webp = try ready(picture.webp, a);
    var t: Tally = .{};
    for (cases) |c| {
        errdefer std.debug.print("case {s}\n", .{c.name});
        const bytes = if (c.b64) |b| blk: {
            const d = std.base64.standard.Decoder;
            const out = try arena.alloc(u8, try d.calcSizeForSlice(b));
            try d.decode(out, b);
            break :blk out;
        } else blk: {
            const p = try std.fs.path.join(arena, &.{ dir.?, c.file.? });
            break :blk try std.Io.Dir.cwd().readFileAlloc(std.testing.io, p, arena, .limited(64 << 20));
        };
        try std.testing.expectEqualStrings(c.bytes, &sha(bytes));
        const fmt = picture.sniff(bytes);
        if ((fmt == .jpeg and !jpeg) or (fmt == .webp and !webp)) {
            t.skipped += 1;
            continue;
        }
        if (c.problem) |p| {
            var why: image.Refusal = .{};
            const got = steps.prepare(std.testing.io, arena, a, bytes, image.request_tokens, image.request_tokens, &why);
            const want: anyerror = if (std.mem.eql(u8, p, "side")) error.ImageTooLarge else if (std.mem.eql(u8, p, "animated") or std.mem.eql(u8, p, "mpo")) error.Animated else if (std.mem.eql(u8, p, "cmyk")) error.CmykJpeg else error.Invalid;
            try std.testing.expectError(want, got);
            t.refused += 1;
            continue;
        }
        const d = try steps.decode(std.testing.io, a, bytes);
        defer a.free(d.rgb);
        try std.testing.expectEqual(c.width.?, d.width);
        try std.testing.expectEqual(c.height.?, d.height);
        try std.testing.expectEqual(c.grid.?, image.grid(d.height, d.width));
        try std.testing.expectEqualStrings(c.rgb.?, &sha(d.rgb));
        const v = try image.pixelValues(a, d.height, d.width, d.rgb);
        defer a.free(v);
        // the processor's float32 bytes are little-endian; so are this host's
        try std.testing.expectEqualStrings(c.values.?, &sha(std.mem.sliceAsBytes(v)));
        // the request path carries the same RGB, its digest and identity
        var why: image.Refusal = .{};
        const p = try steps.prepare(std.testing.io, arena, a, bytes, image.request_tokens, image.request_tokens, &why);
        const back = try image.record(.{ .pixels = p.floats, .gh = p.grid[0], .gw = p.grid[1], .tokens = p.grid[0] * p.grid[1] });
        try std.testing.expectEqualSlices(u8, d.rgb, back.rgb);
        try std.testing.expectEqualSlices(u8, &image.digest(d.height, d.width, d.rgb), &back.rec.digest);
        try std.testing.expectEqualSlices(u8, &p.digest, &back.rec.digest);
        try std.testing.expectEqual(c.grid.?, p.grid);
        t.checked += 1;
    }
    std.debug.print("{s}: {d} cases, {d} bit-equal, {d} refused as the oracle, {d} skipped (no pinned library)\n", .{ json_path, cases.len, t.checked, t.refused, t.skipped });
    return t;
}

test "V1 small set: decoded RGB and pixel values bit-equal to the processor's, refusals as named" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const t = try run(arena_state.allocator(), "zig/tests/fixtures/inkling2/images.json", null);
    try std.testing.expect(t.checked >= 15); // the PNG cases at least, with or without the libraries
    if (t.skipped > 0) return error.SkipZigTest;
}

test "V1 large set (TF_INK2_IMAGE_FIXTURES): the sizes up to 4096 x 4096" {
    const path = @import("sys").getenv("TF_INK2_IMAGE_FIXTURES") orelse return error.SkipZigTest;
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const p = try arena.dupe(u8, path);
    const t = try run(arena, p, std.fs.path.dirname(p) orelse ".");
    try std.testing.expect(t.checked + t.refused > 0);
    if (t.skipped > 0) return error.SkipZigTest;
}

test "the token share is checked on the size the decoder uses (the last header before the data), before inflating" {
    const a = std.testing.allocator;
    // two IHDRs, 10 x 10 then 2000 x 20 (50 tokens), and image data that does not inflate: the decoder's size is
    // the second, and the refusal comes before the data is read (garbage would otherwise be refused as invalid)
    const twice = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x0a\x00\x00\x00\x0a\x08\x02\x00\x00\x00\x02\x50\x58\xea\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x07\xd0\x00\x00\x00\x14\x08\x02\x00\x00\x00\x5a\x84\x31\x02\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    var why: image.Refusal = .{};
    try std.testing.expectError(error.ImageTooManyTokens, steps.prepare(std.testing.io, a, a, twice, 49, 98, &why));
    try std.testing.expectEqualStrings("2000 x 20 pixels is 50 image tokens (1 x 50 blocks of 40 pixels); this request allows 49 an image (98 a request, shared evenly by its images)", why.text());
    // at 50 tokens the size passes and the data is read: refused as invalid bytes
    try std.testing.expectError(error.Invalid, steps.prepare(std.testing.io, a, a, twice, 50, 98, &why));
}

test "a picture past the core's decompression-bomb bound is refused with its numbers, not as invalid bytes" {
    const a = std.testing.allocator;
    const huge = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x27\x10\x00\x00\x27\x10\x08\x02\x00\x00\x00\x35\x2c\xf5\x70\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82"; // 10,000 x 10,000
    var why: image.Refusal = .{};
    try std.testing.expectError(error.ImageTooLarge, steps.prepare(std.testing.io, a, a, huge, image.request_tokens, image.request_tokens, &why));
    try std.testing.expectEqualStrings("it is 10000 x 10000 pixels; at most 4096 pixels a side are accepted (send a smaller image)", why.text());
}

test "a multi-picture JPEG (MPO) is refused as the processor's route refuses it" {
    const a = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // the core decoder's edge fixture: a two-picture MPO, which the core decodes as its first picture
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/fixtures/picture/edges.json", arena, .limited(1 << 22));
    const cases = (try std.json.parseFromSliceLeaky(struct { cases: []const struct { name: []const u8, b64: []const u8 } }, arena, text, .{ .ignore_unknown_fields = true })).cases;
    for (cases) |c| if (std.mem.eql(u8, c.name, "jpeg_mpo_two")) {
        const bytes = try arena.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(c.b64));
        try std.base64.standard.Decoder.decode(bytes, c.b64);
        var why: image.Refusal = .{};
        try std.testing.expectError(error.Animated, steps.prepare(std.testing.io, arena, a, bytes, image.request_tokens, image.request_tokens, &why));
        try std.testing.expectError(error.Animated, steps.decode(std.testing.io, a, bytes));
        return;
    };
    return error.TestUnexpectedResult;
}

test "a refusal names its numbers, before any pixel is decoded" {
    const a = std.testing.allocator;
    // a PNG claiming 5000 x 20 whose image data does not inflate: refused by size, not as invalid bytes
    const head = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x13\x88\x00\x00\x00\x14\x08\x02\x00\x00\x00\xee\x68\x40\xf7\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    var why: image.Refusal = .{};
    try std.testing.expectError(error.ImageTooLarge, steps.prepare(std.testing.io, a, a, head, image.request_tokens, image.request_tokens, &why));
    try std.testing.expectEqualStrings("it is 5000 x 20 pixels; at most 4096 pixels a side are accepted (send a smaller image)", why.text());
    // 2000 x 20 is 50 tokens: over a cap of 49
    const small = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x07\xd0\x00\x00\x00\x14\x08\x02\x00\x00\x00\x5a\x84\x31\x02\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    try std.testing.expectError(error.ImageTooManyTokens, steps.prepare(std.testing.io, a, a, small, 49, 98, &why));
    try std.testing.expectEqualStrings("2000 x 20 pixels is 50 image tokens (1 x 50 blocks of 40 pixels); this request allows 49 an image (98 a request, shared evenly by its images)", why.text());
}

/// A progressive JPEG of one mid-grey, made here: `w` x `h`, three components each sampled `f` x `f`, one DC scan a
/// component in which every difference is 0 (a Huffman table of one 1-bit code, so each scan is zero bytes), no AC
/// scan; `pad` APP2 segments of 65,533 bytes first and `junk` stray bytes before the first scan. The sampling leaves
/// the picture as it is, but libjpeg-turbo rounds each component's coefficient buffer up to whole factors, so 3 x 3
/// gives a 4096 x 4096 picture the largest buffer it can have (3 x 513 x 513 blocks).
fn greyProgressive(a: std.mem.Allocator, w: u16, h: u16, f: u8, pad: usize, junk: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "\xff\xd8");
    for (0..pad) |_| {
        try out.appendSlice(a, "\xff\xe2\xff\xff");
        @memset(try out.addManyAsSlice(a, 65533), 0);
    }
    try out.appendSlice(a, "\xff\xdb\x00\x43\x00"); // one quantization table of ones
    @memset(try out.addManyAsSlice(a, 64), 1);
    try out.appendSlice(a, &.{ 0xff, 0xc2, 0x00, 17, 8, @intCast(h >> 8), @truncate(h), @intCast(w >> 8), @truncate(w), 3 });
    for (1..4) |id| try out.appendSlice(a, &.{ @intCast(id), f << 4 | f, 0 });
    // DC table 0: one 1-bit code (counts 1, 0 x 15), for difference category 0
    try out.appendSlice(a, &.{ 0xff, 0xc4, 0x00, 0x14, 0x00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    @memset(try out.addManyAsSlice(a, junk), 0);
    const blocks = (@as(usize, w) + 7) / 8 * ((@as(usize, h) + 7) / 8); // a component's own blocks, unpadded
    for (1..4) |id| {
        try out.appendSlice(a, &.{ 0xff, 0xda, 0x00, 0x08, 1, @intCast(id), 0x00, 0, 0, 0 });
        @memset(try out.addManyAsSlice(a, (blocks + 7) / 8), 0);
    }
    try out.appendSlice(a, "\xff\xd9");
    return out.toOwnedSlice(a);
}

test "a progressive JPEG past the decoder's memory cap is refused by name, not as out of memory or invalid bytes" {
    const a = std.testing.allocator;
    if (!try ready(picture.jpeg, a)) return error.SkipZigTest;
    // 1024 x 1024 at 4:4:4: 3 x 128 x 128 blocks x 128 B = 6 MiB of coefficients, under 7 MiB with libjpeg's pools
    const bytes = try greyProgressive(a, 1024, 1024, 1, 0, 0);
    defer a.free(bytes);
    const at = struct {
        fn cap(mib: u32) picture.Limits {
            return .{ .side = image.max_side, .pixels = image.max_pixels, .jpeg_memory_mib = mib };
        }
    }.cap;
    try std.testing.expectError(error.JpegDecoderMemory, steps.decodeSamples(std.testing.io, a, bytes, at(6)));
    const s = try steps.decodeSamples(std.testing.io, a, bytes, at(7));
    a.free(s.data);
    // stray bytes before the scan: libjpeg warns, and TurboJPEG then reports the memory error as a warning too
    const warned = try greyProgressive(a, 1024, 1024, 1, 0, 2);
    defer a.free(warned);
    try std.testing.expectError(error.JpegDecoderMemory, steps.decodeSamples(std.testing.io, a, warned, at(6)));
    const t = try steps.decodeSamples(std.testing.io, a, warned, at(7));
    a.free(t.data);
}

test "the largest JPEGs the limits allow decode under the memory cap: 4096 x 4096 progressive, 4:4:4 and 3 x 3, with 9 MB of APP2" {
    const a = std.testing.allocator;
    if (!try ready(picture.jpeg, a)) return error.SkipZigTest;
    for ([_][2]usize{ .{ 1, 0 }, .{ 3, 0 }, .{ 3, 150 } }) |c| {
        const bytes = try greyProgressive(a, 4096, 4096, @intCast(c[0]), c[1], 0);
        defer a.free(bytes);
        try std.testing.expect(bytes.len <= 10 << 20); // the server's bound for one image (server/images.zig)
        var arena_state = std.heap.ArenaAllocator.init(a);
        defer arena_state.deinit();
        var why: image.Refusal = .{};
        const p = try steps.prepare(std.testing.io, arena_state.allocator(), a, bytes, image.request_tokens, image.request_tokens, &why);
        try std.testing.expectEqual([2]u32{ 103, 103 }, p.grid);
        const back = try image.record(.{ .pixels = p.floats, .gh = p.grid[0], .gw = p.grid[1], .tokens = p.grid[0] * p.grid[1] });
        try std.testing.expectEqual(@as(usize, 4096 * 4096 * 3), back.rgb.len);
        for (back.rgb) |b| if (b != 128) return error.TestUnexpectedResult;
    }
}

test "image prepares on two threads never decode at once: one decode in the process at a time" {
    // the probe holds each decode open for up to 500 ms, or until a second decode is inside too: without the
    // serialization the other thread arrives within that time, so the count of decodes in flight reaches 2
    const Probe = struct {
        var inside: std.atomic.Value(u32) = .init(0);
        var most: std.atomic.Value(u32) = .init(0);
        var entered: std.atomic.Value(u32) = .init(0);
        fn hold() void {
            const n = inside.fetchAdd(1, .acq_rel) + 1;
            _ = most.fetchMax(n, .acq_rel);
            _ = entered.fetchAdd(1, .acq_rel);
            var ms: u32 = 0;
            while (inside.load(.acquire) < 2 and ms < 500) : (ms += 1) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            _ = inside.fetchSub(1, .acq_rel);
        }
    };
    image.probe = Probe.hold;
    defer image.probe = null;
    // a 17 x 9 PNG with clear alpha (fixtures/inkling2/images.json png_rgba_clear_17x9)
    const png = "iVBORw0KGgoAAAANSUhEUgAAABEAAAAJCAYAAADU6McMAAABT0lEQVR4nC3SPUjVYRTH8c+Ni9xMu0fFQSgTIxIXCZoabNAQAgkaegEhcJUmt6KXQa3IphoKEl2iQR2C1EWQwiIaDBKiIdJFFLkEUcHlIhL/v89ZHjgcvud7fjwFqso+C3tCk7AqPBDWhS5hU/gpHBN6he/Cc+GQMCXUFJ34i1m8wwusoA6bKKMZ93ANDXiCkfR+RUFRRzcy0AQO46ODOodOFPEFC+jBJfzAIM7ifAZ5hNt4jBYM4zWaMINb+J2296If9biQQG8yyC5OYQ1tGEo27fmWg3qG0WR1H/9wJPW6M8g+dnAVZ9JQCeO4iJuYTyaT2EMt9bbwKYMsYiAFW0VrOukuGnEyab/N1bmDX3ifh8ofBRvLyi4LK0JFKAkPhVfChnBamBPWhG/CuDAhtAnX8+9Q1PESSziOMfRhO+UznevyFFeS4Q0cRSWd+8F/F95Oh7xm2WAAAAAASUVORK5CYII=";
    var bytes: [std.base64.standard.Decoder.calcSizeForSlice(png) catch unreachable]u8 = undefined;
    try std.base64.standard.Decoder.decode(&bytes, png);
    const Worker = struct {
        fn run(go: *const std.atomic.Value(bool), b: []const u8, out: *(anyerror![32]u8)) void {
            while (!go.load(.acquire)) std.atomic.spinLoopHint();
            var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena_state.deinit();
            var why: image.Refusal = .{};
            const p = steps.prepare(std.testing.io, arena_state.allocator(), std.heap.page_allocator, b, image.request_tokens, image.request_tokens, &why);
            out.* = if (p) |q| q.digest else |e| e;
        }
    };
    var go: std.atomic.Value(bool) = .init(false);
    var got: [2](anyerror![32]u8) = .{ error.NotRun, error.NotRun };
    var threads: [2]std.Thread = undefined;
    for (&threads, &got) |*t, *g| t.* = try std.Thread.spawn(.{}, Worker.run, .{ &go, &bytes, g });
    go.store(true, .release);
    for (threads) |t| t.join();
    try std.testing.expectEqual(@as(u32, 2), Probe.entered.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), Probe.most.load(.acquire));
    try std.testing.expectEqualSlices(u8, &(try got[0]), &(try got[1]));
}

test {
    _ = image;
}
