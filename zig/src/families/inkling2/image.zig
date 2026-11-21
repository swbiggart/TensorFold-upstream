//! Inkling-2's image processor on rank 0's host: Transformers' Inkling2ImageProcessor (main 109a7d19), bit for bit.
//!
//! An image's bytes are decoded by TensorFold's codecs (core/picture: PNG its own; JPEG and WebP through
//! libjpeg-turbo 3.1.4.1 and libwebp 1.6.0, the releases Pillow 12.3.0 bundles), turned upright by their orientation
//! tag, and any alpha is composited onto the processor's checkerboard (owner decision D1). There is no resize: an image
//! is ceil(H / 40) x ceil(W / 40) tokens, padded right and bottom, and each byte becomes the processor's fp32 value
//! through one 257-entry table (`values`). Gate V1 (tools/zig/inkling2_image_fixtures.py, image_test.zig) checks the
//! RGB and the pixel values against the processor's on a fixture set.
//!
//! The server's `PreparedImage` carries the upright RGB bytes behind a `Record` (size, grid, digest), not fp32 rows:
//! 4.8 KB a token instead of 19.2 KB, held for the request's life like any request's pixels. The engine reads them
//! back (`record`) and maps each byte through `values` on the device, which gives the processor's `pixel_values`.
const std = @import("std");
const builtin = @import("builtin");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;

/// A token's block of pixels a side; the processor's `size` (40, 40) is this block, not a target size.
pub const block = 40;
/// The recipe's limits (owner decision D2): pixels a side, and image tokens one request may hold (one 4096 x 4096).
pub const max_side = 4096;
pub const max_pixels: u64 = max_side * max_side;
pub const request_tokens = 10_816;

/// libjpeg-turbo's working memory for one JPEG, in MiB (TJPARAM_MAXMEMORY, which it multiplies by 1,048,576): a
/// progressive or multi-scan JPEG decodes through a whole-image buffer of DCT coefficients, 64 x 2 B a block, per
/// component jround_up(width_in_blocks, h) x jround_up(height_in_blocks, v) blocks (jdcoefct.c), width_in_blocks =
/// ceil(W h / (8 h_max)) <= ceil(4096 / 8) = 512, rounded up to the factor (1..4): at most 513, at factor 3. Three
/// components at most (a four-component CMYK or YCCK JPEG is refused before decoding, as the processor's route refuses
/// it): 3 x 513 x 513 x 128 B = 101,056,896 B (96.4 MiB; 4:4:4 is 3 x 512 x 512 x 128 B = 96 MiB). The cap also counts
/// what libjpeg allocated before that buffer (its pools at 4096 wide: under 1 MiB; the pinned 3.1.4.1 decodes the
/// 4096 x 4096 4:4:4 and 3 x 3 pictures at 97 MiB and refuses both at 96), and no marker is kept (TJPARAM_SAVEMARKERS
/// 0: TurboJPEG's default keeps every APP2 segment, up to the file's 10 MiB). Every JPEG these limits allow decodes;
/// one that would need more is refused as error.JpegDecoderMemory, worded by `jpeg_memory_words`.
pub const jpeg_memory_mib = 100;

comptime {
    const blocks = (max_side / 8 + 2) / 3 * 3; // 512 rounded up to a factor of 3
    std.debug.assert(3 * blocks * blocks * 64 * 2 + (1 << 20) <= jpeg_memory_mib << 20);
}

/// The refusal's words for a JPEG past `jpeg_memory_mib`.
pub const jpeg_memory_words = std.fmt.comptimePrint("the JPEG needs more than {d} MiB of decoder memory (a progressive or multi-scan JPEG's whole-image buffer); send a baseline JPEG or a PNG", .{jpeg_memory_mib});

/// One image decode at a time in the process, whatever the requests: each request prepares its images on its own
/// thread (server/server.zig, a thread a connection), so without this k image requests would hold k decodes' working
/// memory at once. The section is CPU and allocation only (the bytes are fetched before it, nothing in it waits on
/// `io`), so its holder always finishes; others wait on the io's futex. Text requests never reach it.
var decoding: std.Io.Mutex = .init;

/// The processor's grid for an upright image: rows, then columns.
pub fn grid(height: u64, width: u64) [2]u32 {
    return .{ @intCast((height + block - 1) / block), @intCast((width + block - 1) / block) };
}

/// An image's tokens: one a block of 40 x 40 pixels (a quarter turn does not change the count).
pub fn tokens(height: u64, width: u64) u64 {
    const g = grid(height, width);
    return @as(u64, g[0]) * g[1];
}

/// The processor's convert_to_rgb for an RGBA pixel at (x, y) of the upright image: integer compositing onto
/// 8-pixel squares of 204 where (y / 8 + x / 8) is even and 153 where odd; alpha 255 keeps the colour exactly.
pub fn overChecker(c: u8, alpha: u8, x: usize, y: usize) u8 {
    const bg: u32 = if ((y / 8 + x / 8) % 2 == 0) 204 else 153;
    return @intCast((@as(u32, c) * alpha + bg * (255 - @as(u32, alpha)) + 127) / 255);
}

/// The index of the padding value in `values`: the processor pads with -1 on the 0-255 scale before it normalizes.
pub const pad = 256;

/// The processor's fp32 value for a byte 0..255, and at `pad` for padding: TorchvisionBackend's fused rescale and
/// normalize, (x - 127.5) / 127.5 with a true fp32 division (mean and std 0.5 times 1 / rescale 255 are 127.5 exactly;
/// x - 127.5 is exact, so each value is one correctly rounded division). A reciprocal multiply differs in 127 values.
pub const values: [257]f32 = blk: {
    @setEvalBranchQuota(10_000);
    var t: [257]f32 = undefined;
    for (&t, 0..) |*v, i| {
        const x: f32 = if (i == pad) -1.0 else @floatFromInt(i);
        v.* = (x - @as(f32, 127.5)) / @as(f32, 127.5);
    }
    break :blk t;
};

/// An upright image after compositing: RGB, three bytes a pixel, row-major.
pub const Decoded = struct {
    height: u32,
    width: u32,
    rgb: []u8,
};

/// Transformers' `pixel_values` for one image: [rows * cols, 40, 40, 3] fp32, tokens in raster order, channel last,
/// padding right and bottom. The engine need not build these; gate V1 compares their bytes.
pub fn pixelValues(a: Allocator, height: u32, width: u32, rgb: []const u8) Allocator.Error![]f32 {
    const g = grid(height, width);
    const n = @as(usize, g[0]) * g[1] * block * block * 3;
    const out = try a.alloc(f32, n);
    var o: usize = 0;
    for (0..g[0]) |r| for (0..g[1]) |c| for (0..block) |dy| for (0..block) |dx| {
        const y = r * block + dy;
        const x = c * block + dx;
        if (y < height and x < width) {
            const px = rgb[(y * width + x) * 3 ..][0..3];
            for (0..3) |ch| out[o + ch] = values[px[ch]];
        } else for (0..3) |ch| {
            out[o + ch] = values[pad];
        }
        o += 3;
    };
    return out;
}

/// The tag hashed before an image's pixels: names this processor's output, so another rule is another digest.
pub const digest_tag = "inkling2-image-1";

/// An image's full identity: SHA-256 of the tag, the upright height and width (u32 little-endian) and the RGB bytes.
/// Two encodings of the same pixels share it; a decoder change that moves a pixel changes it.
pub fn digest(height: u32, width: u32, rgb: []const u8) [32]u8 {
    var h = Sha256.init(.{});
    h.update(digest_tag);
    var hw: [8]u8 = undefined;
    std.mem.writeInt(u32, hw[0..4], height, .little);
    std.mem.writeInt(u32, hw[4..8], width, .little);
    h.update(&hw);
    h.update(rgb);
    var d: [32]u8 = undefined;
    h.final(&d);
    return d;
}

/// The prompt cache's 62-bit identity for a digest and its grid (core/picture/identity.zig's `withGrid`).
pub fn identity(comptime picture: type, d: [32]u8, g: [2]u32) u64 {
    return picture.identity.withGrid(std.mem.readInt(u64, d[0..8], .little), g);
}

/// What a request's image carries in the server's fp32 field (`PreparedImage.pixels`, then the lanes' `Image.pixels`):
/// this record, then the upright RGB bytes, zero-padded to whole floats. The server and the lanes only hold and pass
/// the slice (nothing outside the family reads `pixels`); it lives as long as the request, as any family's pixels do.
pub const Record = extern struct {
    magic: [8]u8 = record_magic.*,
    version: u32 = 1,
    height: u32,
    width: u32,
    rows: u32,
    cols: u32,
    tokens: u32,
    rgb_len: u64,
    digest: [32]u8 = @splat(0),
};

const record_magic = "ink2img\x00";

/// Storage for one image of `height` x `width`: the floats for the server, and the RGB bytes to fill before `seal`.
pub fn carry(a: Allocator, height: u32, width: u32) Allocator.Error!struct { floats: []f32, rgb: []u8 } {
    const n = @as(usize, height) * width * 3;
    const head = @sizeOf(Record);
    const floats = try a.alloc(f32, (head + n + 3) / 4);
    const all = std.mem.sliceAsBytes(floats);
    const g = grid(height, width);
    const r: Record = .{ .height = height, .width = width, .rows = g[0], .cols = g[1], .tokens = g[0] * g[1], .rgb_len = n };
    @memcpy(all[0..head], std.mem.asBytes(&r));
    @memset(all[head + n ..], 0);
    return .{ .floats = floats, .rgb = all[head..][0..n] };
}

/// Writes the digest of the filled RGB into the record and returns it.
pub fn seal(floats: []f32) [32]u8 {
    const all = std.mem.sliceAsBytes(floats);
    var r = std.mem.bytesToValue(Record, all[0..@sizeOf(Record)]);
    r.digest = digest(r.height, r.width, all[@sizeOf(Record)..][0..@intCast(r.rgb_len)]);
    @memcpy(all[0..@sizeOf(Record)], std.mem.asBytes(&r));
    return r.digest;
}

/// An Inkling-2 request image read back on the engine's side: `img` is the lanes' image (or anything with `pixels`,
/// `gh`, `gw` and `tokens`); refuses a slice this file did not write or one whose record disagrees with the image.
pub fn record(img: anytype) error{NotInkling2Image}!struct { rec: Record, rgb: []const u8 } {
    const all = std.mem.sliceAsBytes(img.pixels);
    const head = @sizeOf(Record);
    if (all.len < head) return error.NotInkling2Image;
    const r = std.mem.bytesToValue(Record, all[0..head]);
    // the fixed fields and the sides first: only then are the products below small (at most 4096 x 4096 x 3)
    if (!std.mem.eql(u8, &r.magic, record_magic) or r.version != 1 or r.height == 0 or r.width == 0 or
        r.height > max_side or r.width > max_side)
        return error.NotInkling2Image;
    const g = grid(r.height, r.width);
    const n = @as(u64, r.height) * r.width * 3;
    if (r.rgb_len != n or (head + n + 3) / 4 * 4 != all.len or r.rows != g[0] or r.cols != g[1] or
        r.tokens != g[0] * g[1] or img.gh != r.rows or img.gw != r.cols or img.tokens != r.tokens)
        return error.NotInkling2Image;
    return .{ .rec = r, .rgb = all[head..][0..@intCast(n)] };
}

/// How many pictures JPEG bytes hold, as Pillow 12.3's open judges them (jpeg_factory, _getmp): `.many` when the last
/// APP2 "MPF" index has an image count (0xB001) above 1 and an entry table (0xB002) of that many JPEG entries, unless
/// an APP1 segment marks an Ultra HDR picture (Pillow then opens an MpoImageFile of that many frames, which the
/// processor's route refuses); `.broken` when the table is shorter than the count (Pillow's open raises);
/// `.one` otherwise: no index, an index that does not read, no count or no table, or an entry that is not JPEG
/// (Pillow then warns and opens the base JPEG).
pub const Pictures = enum { one, many, broken };

pub fn pictures(bytes: []const u8) Pictures {
    if (!std.mem.startsWith(u8, bytes, "\xff\xd8\xff")) return .one;
    var mp: ?[]const u8 = null;
    var hdr = false;
    var at: usize = 2;
    while (at + 4 <= bytes.len) {
        if (bytes[at] != 0xff) return .one;
        const m = bytes[at + 1];
        if (m == 0xff) {
            at += 1;
            continue;
        }
        if (m == 0x00 or m == 0x01 or m == 0xc8 or (m >= 0xd0 and m <= 0xd9) or (m >= 0xf0 and m <= 0xfd)) {
            at += 2;
            continue;
        }
        if (m == 0xda) break; // the first scan: Pillow's open reads no further markers
        const len = std.mem.readInt(u16, bytes[at + 2 ..][0..2], .big);
        if (len < 2 or bytes.len - at - 4 < len - 2) return .one;
        const body = bytes[at + 4 ..][0 .. len - 2];
        if (m == 0xe2 and std.mem.startsWith(u8, body, "MPF\x00")) mp = body[4..];
        if (m == 0xe1 and std.mem.indexOf(u8, body, " hdrgm:Version=\"") != null) hdr = true;
        at += 2 + @as(usize, len); // a 65,535-byte segment: the sum is past u16
    }
    return mpIndex(mp orelse return .one, hdr);
}

/// The MP index: a TIFF header (MM or II), its first directory, the count (0xB001, LONG or SHORT) and the entry
/// table (0xB002: 16 bytes an image, its value inline up to 4 bytes, else at an offset from the header).
fn mpIndex(t: []const u8, hdr: bool) Pictures {
    if (t.len < 8) return .one;
    const e: std.builtin.Endian = if (std.mem.startsWith(u8, t, "MM\x00\x2a")) .big else if (std.mem.startsWith(u8, t, "II\x2a\x00")) .little else return .one;
    const ifd = std.mem.readInt(u32, t[4..8], e);
    if (ifd > t.len or t.len - ifd < 2) return .one;
    const count = std.mem.readInt(u16, t[ifd..][0..2], e);
    var n: ?u64 = null;
    var table: ?[]const u8 = null;
    for (0..count) |i| {
        const at = ifd + 2 + i * 12;
        if (at + 12 > t.len) return .one;
        const tag = std.mem.readInt(u16, t[at..][0..2], e);
        const kind = std.mem.readInt(u16, t[at + 2 ..][0..2], e);
        const len = std.mem.readInt(u32, t[at + 4 ..][0..4], e);
        if (tag == 0xb001) n = switch (kind) {
            4 => std.mem.readInt(u32, t[at + 8 ..][0..4], e),
            3 => std.mem.readInt(u16, t[at + 8 ..][0..2], e),
            else => return .one,
        };
        if (tag == 0xb002) {
            if (kind != 7) return .one; // UNDEFINED bytes
            if (len <= 4) {
                table = t[at + 8 ..][0..len];
            } else {
                const off = std.mem.readInt(u32, t[at + 8 ..][0..4], e);
                if (off > t.len or t.len - off < len) return .one;
                table = t[off..][0..len];
            }
        }
    }
    const images = n orelse return .one;
    const entries = table orelse return .one;
    if (entries.len / 16 < images) return .broken;
    for (0..@intCast(images)) |i| {
        const attribute = std.mem.readInt(u32, entries[i * 16 ..][0..4], e);
        if ((attribute >> 24) & 7 != 0) return .one; // not JPEG: Pillow's "unsupported picture format", the base JPEG
    }
    return if (images > 1 and !hdr) .many else .one;
}

/// Why an image was refused, in words with its numbers; each request thread words its own last refusal.
pub const Refusal = struct {
    buf: [240]u8 = undefined,
    len: usize = 0,

    pub fn text(r: *const Refusal) []const u8 {
        return r.buf[0..r.len];
    }

    fn set(r: *Refusal, comptime fmt: []const u8, args: anytype) void {
        r.len = if (std.fmt.bufPrint(&r.buf, fmt, args)) |t| t.len else |_| 0;
    }
};

/// The words for a size the codec refused (its `Limits.fits`, before any pixel): a side over 4,096 (owner decision
/// D2) or more than this request's share of image tokens (`cap`, what the server passes as max_tokens), with the
/// numbers (decision D3); returns when the size meets both (another limit refused it).
pub fn admit(size: [2]u64, cap: u32, request_cap: u32, why: *Refusal) error{ ImageTooLarge, ImageTooManyTokens }!void {
    const w, const h = size;
    if (@max(w, h) > max_side or @as(u64, w) * h > max_pixels) {
        why.set("it is {d} x {d} pixels; at most {d} pixels a side are accepted (send a smaller image)", .{ w, h, max_side });
        return error.ImageTooLarge;
    }
    const t = tokens(h, w);
    if (t > cap) {
        const g = grid(h, w);
        why.set("{d} x {d} pixels is {d} image tokens ({d} x {d} blocks of 40 pixels); this request allows {d} an image ({d} a request, shared evenly by its images)", .{ w, h, t, g[0], g[1], cap, request_cap });
        return error.ImageTooManyTokens;
    }
}

/// Tests only: called inside `prepare`'s decode section, after the samples are read (never set outside tests).
pub var probe: ?*const fn () void = null;

/// The steps that read pixels, over `picture` (core/picture/picture.zig: the core module's, or a test's own).
pub fn For(comptime picture: type) type {
    return struct {
        pub const Error = picture.Error || error{ ImageTooLarge, ImageTooManyTokens };

        /// A request image: refused above a limit by the codec's own size check, on the size it decodes and before it
        /// allocates or inflates any pixel (the side, and this request's share of tokens: `cap`), decoded (temporary
        /// samples in `scratch`), turned upright and composited into `carry` storage in `a`. A multi-picture JPEG is
        /// refused as Pillow's route refuses it (more than one frame).
        pub fn prepare(io: std.Io, a: Allocator, scratch: Allocator, bytes: []const u8, cap: u32, request_cap: u32, why: *Refusal) Error!Prepared {
            switch (pictures(bytes)) {
                .one => {},
                .many => return error.Animated,
                .broken => return error.Invalid,
            }
            var seen: [2]u64 = .{ 0, 0 };
            const limits: picture.Limits = .{ .side = max_side, .pixels = max_pixels, .tokens = cap, .block = block, .seen = &seen, .jpeg_memory_mib = jpeg_memory_mib };
            // the decode, its samples and the codec's working memory, inside the process's one decode section
            const c, const h, const w = decoded: {
                decoding.lockUncancelable(io);
                defer decoding.unlock(io);
                const s = decodeSamples(io, scratch, bytes, limits) catch |err| {
                    // the words, with the numbers: a size over this family's limits, also one the core's decompression
                    // bomb bound (Pillow's, checked first) calls invalid; `seen` is the size the codec judged, or 0 x 0
                    if (err == error.Oversized or err == error.Invalid) try admit(seen, cap, request_cap, why);
                    return err;
                };
                defer scratch.free(s.data);
                if (builtin.is_test) if (probe) |f| f();
                const turned = s.orientation >= 5 and s.orientation <= 8;
                const h: u32 = @intCast(if (turned) s.width else s.height);
                const w: u32 = @intCast(if (turned) s.height else s.width);
                const c = try carry(a, h, w);
                finish(s, c.rgb);
                break :decoded .{ c, h, w };
            };
            const g = grid(h, w);
            const d = seal(c.floats);
            return .{ .floats = c.floats, .height = h, .width = w, .grid = g, .digest = d, .identity = identity(picture, d, g) };
        }

        /// The codecs' samples (RGB or RGBA, as stored) under `limits`.
        pub fn decodeSamples(io: std.Io, a: Allocator, bytes: []const u8, limits: picture.Limits) picture.Error!picture.Samples {
            return switch (picture.sniff(bytes)) {
                .png => try picture.png.decodeWithLimits(a, bytes, limits),
                .jpeg => try picture.jpeg.decodeWithLimits(io, a, bytes, limits),
                .webp => try picture.webp.decodeWithLimits(io, a, bytes, limits),
                .unknown => error.Invalid,
            };
        }

        /// A whole decode for tools and tests: upright, composited RGB in `a`.
        pub fn decode(io: std.Io, a: Allocator, bytes: []const u8) picture.Error!Decoded {
            switch (pictures(bytes)) {
                .one => {},
                .many => return error.Animated,
                .broken => return error.Invalid,
            }
            const s = try decodeSamples(io, a, bytes, .{ .side = max_side, .pixels = max_pixels });
            defer a.free(s.data);
            const turned = s.orientation >= 5 and s.orientation <= 8;
            const h: u32 = @intCast(if (turned) s.width else s.height);
            const w: u32 = @intCast(if (turned) s.height else s.width);
            const rgb = try a.alloc(u8, @as(usize, h) * w * 3);
            finish(s, rgb);
            return .{ .height = h, .width = w, .rgb = rgb };
        }

        /// The orientation's transpose (2-8; any other value keeps the stored picture), then alpha onto the
        /// checkerboard in upright coordinates, into `out` (upright height * width * 3 bytes).
        pub fn finish(s: picture.Samples, out: []u8) void {
            const o = if (s.orientation >= 2 and s.orientation <= 8) s.orientation else 1;
            const w = if (o >= 5) s.height else s.width;
            const h = if (o >= 5) s.width else s.height;
            std.debug.assert(out.len == w * h * 3);
            for (0..h) |y| for (0..w) |x| {
                const sx, const sy = switch (o) {
                    2 => .{ s.width - 1 - x, y },
                    3 => .{ s.width - 1 - x, s.height - 1 - y },
                    4 => .{ x, s.height - 1 - y },
                    5 => .{ y, x },
                    6 => .{ y, s.height - 1 - x },
                    7 => .{ s.width - 1 - y, s.height - 1 - x },
                    8 => .{ s.width - 1 - y, x },
                    else => .{ x, y },
                };
                const px = s.data[(sy * s.width + sx) * s.channels ..][0..s.channels];
                const dst = out[(y * w + x) * 3 ..][0..3];
                if (s.channels == 4) {
                    for (dst, px[0..3]) |*c, v| c.* = overChecker(v, px[3], x, y);
                } else @memcpy(dst, px[0..3]);
            };
        }
    };
}

/// One prepared request image: the RGB in `floats` (`carry`), its upright size and grid, its digest and identity.
pub const Prepared = struct {
    floats: []f32,
    height: u32,
    width: u32,
    grid: [2]u32,
    digest: [32]u8,
    identity: u64,
};

test "the table is the processor's: true fp32 division, black -1, white 1, padding below black" {
    try std.testing.expectEqual(@as(u32, 0xbf800000), @as(u32, @bitCast(values[0])));
    try std.testing.expectEqual(@as(u32, 0x3f800000), @as(u32, @bitCast(values[255])));
    try std.testing.expectEqual(@as(u32, 0xbf810101), @as(u32, @bitCast(values[pad])));
    try std.testing.expectEqual(@as(u32, 0xbf7dfdfe), @as(u32, @bitCast(values[1])));
    try std.testing.expectEqual(@as(u32, 0x3f7dfdfe), @as(u32, @bitCast(values[254])));
    // at runtime, the same division in fp32 for every entry (a comptime fold must not differ)
    var x: f32 = -1;
    for (0..257) |i| {
        const at = if (i == 0) pad else i - 1;
        var m: f32 = 127.5;
        _ = &m;
        try std.testing.expectEqual(@as(u32, @bitCast((x - m) / m)), @as(u32, @bitCast(values[at])));
        x += 1;
    }
    // the reciprocal form differs: the table must not be built that way
    var diff: usize = 0;
    for (0..256) |i| diff += @intFromBool(@as(u32, @bitCast((@as(f32, @floatFromInt(i)) - 127.5) * @as(f32, 1.0 / 127.5))) != @as(u32, @bitCast(values[i])));
    try std.testing.expect(diff > 0);
}

test "grid and tokens: one a 40-pixel block, padded up" {
    try std.testing.expectEqual([2]u32{ 1, 1 }, grid(1, 1));
    try std.testing.expectEqual([2]u32{ 1, 1 }, grid(40, 40));
    try std.testing.expectEqual([2]u32{ 2, 1 }, grid(41, 39));
    try std.testing.expectEqual([2]u32{ 9, 12 }, grid(336, 448));
    try std.testing.expectEqual(@as(u64, 1_296), tokens(1080, 1920));
    try std.testing.expectEqual(@as(u64, 5_184), tokens(2160, 3840));
    try std.testing.expectEqual(@as(u64, 7_676), tokens(3024, 4032));
    try std.testing.expectEqual(@as(u64, 10_609), tokens(4096, 4096));
    try std.testing.expect(tokens(max_side, max_side) <= request_tokens);
}

test "the checkerboard: opaque keeps the colour, clear shows 204 / 153 in 8-pixel squares" {
    for (0..256) |c| try std.testing.expectEqual(@as(u8, @intCast(c)), overChecker(@intCast(c), 255, 3, 5));
    try std.testing.expectEqual(@as(u8, 204), overChecker(0, 0, 0, 0));
    try std.testing.expectEqual(@as(u8, 153), overChecker(0, 0, 8, 0));
    try std.testing.expectEqual(@as(u8, 204), overChecker(0, 0, 8, 8));
    try std.testing.expectEqual(@as(u8, 153), overChecker(255, 0, 7, 15));
    // (100 * 128 + 204 * 127 + 127) // 255 = 152
    try std.testing.expectEqual(@as(u8, 152), overChecker(100, 128, 1, 1));
}

test "pixel values: raster tokens, channel last, padding right and bottom" {
    const a = std.testing.allocator;
    // 41 wide, 1 tall: two tokens; pixel (0, 40) opens the second token
    var rgb: [41 * 3]u8 = undefined;
    for (&rgb, 0..) |*b, i| b.* = @intCast(i % 256);
    const v = try pixelValues(a, 1, 41, &rgb);
    defer a.free(v);
    try std.testing.expectEqual(@as(usize, 2 * 40 * 40 * 3), v.len);
    try std.testing.expectEqual(values[0], v[0]);
    try std.testing.expectEqual(values[120], v[40 * 40 * 3]); // second token's first pixel: x 40, channel 0
    try std.testing.expectEqual(values[pad], v[40 * 3]); // first token, row 1: padding
    try std.testing.expectEqual(values[pad], v[40 * 40 * 3 + 3]); // second token, x 41: padding
}

test "the record round-trips size, grid, digest and RGB through the fp32 field, and refuses what it did not write" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(Record)); // a whole number of floats, so the RGB follows aligned
    const c = try carry(a, 41, 5);
    defer a.free(c.floats);
    for (c.rgb, 0..) |*b, i| b.* = @truncate(i + 1);
    const d = seal(c.floats);
    const Img = struct { pixels: []const f32, gh: u32, gw: u32, tokens: u32 };
    const back = try record(Img{ .pixels = c.floats, .gh = 2, .gw = 1, .tokens = 2 });
    try std.testing.expectEqual(@as(u32, 41), back.rec.height);
    try std.testing.expectEqual(@as(u32, 5), back.rec.width);
    try std.testing.expectEqual(@as(u32, 2), back.rec.rows);
    try std.testing.expectEqualSlices(u8, &d, &back.rec.digest);
    try std.testing.expectEqualSlices(u8, &digest(41, 5, c.rgb), &back.rec.digest);
    try std.testing.expectEqualSlices(u8, c.rgb, back.rgb);
    try std.testing.expectError(error.NotInkling2Image, record(Img{ .pixels = c.floats[0 .. c.floats.len - 1], .gh = 2, .gw = 1, .tokens = 2 }));
    try std.testing.expectError(error.NotInkling2Image, record(Img{ .pixels = c.floats, .gh = 1, .gw = 2, .tokens = 2 }));
    try std.testing.expectError(error.NotInkling2Image, record(Img{ .pixels = &.{ 0.5, 1.5 }, .gh = 1, .gw = 1, .tokens = 1 }));
    // one pixel changed is another digest
    c.rgb[7] +%= 1;
    try std.testing.expect(!std.mem.eql(u8, &d, &seal(c.floats)));
}

test "limits: a side over 4096 and a token share over the cap are refused with their numbers" {
    var why: Refusal = .{};
    try admit(.{ 4096, 4096 }, request_tokens, request_tokens, &why);
    try std.testing.expectError(error.ImageTooLarge, admit(.{ 4097, 10 }, request_tokens, request_tokens, &why));
    try std.testing.expectEqualStrings("it is 4097 x 10 pixels; at most 4096 pixels a side are accepted (send a smaller image)", why.text());
    try std.testing.expectError(error.ImageTooManyTokens, admit(.{ 4096, 4096 }, 5_408, request_tokens, &why));
    try std.testing.expectEqualStrings("4096 x 4096 pixels is 10609 image tokens (103 x 103 blocks of 40 pixels); this request allows 5408 an image (10816 a request, shared evenly by its images)", why.text());
}

test "a multi-picture JPEG (MPO) is found as Pillow finds one: an MPF index of more than one picture, not Ultra HDR" {
    const mpo_two = "\xff\xd8\xff\xe0\x00\x10\x4a\x46\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xe2\x00\x68\x4d\x50\x46\x00\x49\x49\x2a\x00\x08\x00\x00\x00\x03\x00\x00\xb0\x07\x00\x04\x00\x00\x00\x30\x31\x30\x30\x01\xb0\x04\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\xb0\x07\x00\x20\x00\x00\x00\x32\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\xed\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x72\x04\x00\x00\xd1\x04\x00\x00\x00\x00\x00\x00\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\xff\xda\x00\x02";
    const mpo_one = "\xff\xd8\xff\xe0\x00\x10\x4a\x46\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xe2\x00\x68\x4d\x50\x46\x00\x49\x49\x2a\x00\x08\x00\x00\x00\x03\x00\x00\xb0\x07\x00\x04\x00\x00\x00\x30\x31\x30\x30\x01\xb0\x04\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\xb0\x07\x00\x20\x00\x00\x00\x32\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\xed\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x72\x04\x00\x00\xd1\x04\x00\x00\x00\x00\x00\x00\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\xff\xda\x00\x02";
    const mpo_hdr = "\xff\xd8\xff\xe0\x00\x10\x4a\x46\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xe1\x00\x17\x78\x20\x68\x64\x72\x67\x6d\x3a\x56\x65\x72\x73\x69\x6f\x6e\x3d\x22\x31\x2e\x30\x22\xff\xe2\x00\x68\x4d\x50\x46\x00\x49\x49\x2a\x00\x08\x00\x00\x00\x03\x00\x00\xb0\x07\x00\x04\x00\x00\x00\x30\x31\x30\x30\x01\xb0\x04\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\xb0\x07\x00\x20\x00\x00\x00\x32\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\xed\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x72\x04\x00\x00\xd1\x04\x00\x00\x00\x00\x00\x00\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\xff\xda\x00\x02";
    try std.testing.expectEqual(Pictures.many, pictures(mpo_two));
    try std.testing.expectEqual(Pictures.one, pictures(mpo_one));
    try std.testing.expectEqual(Pictures.one, pictures(mpo_hdr)); // Pillow opens an Ultra HDR picture as one JPEG
    try std.testing.expectEqual(Pictures.one, pictures("\xff\xd8\xff\xe2\x00\x08MPF\x00II\xff\xda\x00\x02")); // an index that does not read
    try std.testing.expectEqual(Pictures.one, pictures("\x89PNG\r\n\x1a\n"));
    // a count of 2 with no entry table (the directory's third entry, 0xB002, cut): Pillow opens the base JPEG
    var no_table = mpo_two.*;
    no_table[36] = 2;
    try std.testing.expectEqual(Pictures.one, pictures(&no_table));
    // an entry that is not JPEG (ImageDataFormat 1): Pillow's "unsupported picture format", the base JPEG
    var not_jpeg = mpo_two.*;
    not_jpeg[28 + 0x32 + 3] |= 0x01;
    try std.testing.expectEqual(Pictures.one, pictures(&not_jpeg));
    // a table of one entry for a count of 2: Pillow's open raises (struct.error)
    var short = mpo_two.*;
    short[28 + 8 + 2 + 24 + 4] = 0x10;
    try std.testing.expectEqual(Pictures.broken, pictures(&short));
}

test "record refuses absurd sizes before it multiplies them" {
    const a = std.testing.allocator;
    const floats = try a.alloc(f32, @sizeOf(Record) / 4 + 4);
    defer a.free(floats);
    @memset(floats, 0);
    const r: Record = .{ .height = std.math.maxInt(u32), .width = std.math.maxInt(u32), .rows = 1, .cols = 1, .tokens = 1, .rgb_len = 12 };
    @memcpy(std.mem.sliceAsBytes(floats)[0..@sizeOf(Record)], std.mem.asBytes(&r));
    const Img = struct { pixels: []const f32, gh: u32, gw: u32, tokens: u32 };
    try std.testing.expectError(error.NotInkling2Image, record(Img{ .pixels = floats, .gh = 1, .gw = 1, .tokens = 1 }));
    var w = r;
    w.height = max_side + 1;
    w.width = 1;
    @memcpy(std.mem.sliceAsBytes(floats)[0..@sizeOf(Record)], std.mem.asBytes(&w));
    try std.testing.expectError(error.NotInkling2Image, record(Img{ .pixels = floats, .gh = 1, .gw = 1, .tokens = 1 }));
}
