//! Inkling-2's image input as the server sees it (native/cuda_vision_offer.zig `offer`): the processor (image.zig),
//! the image token and its markers, and the limits. Rank 0 only, host only: `prepare` runs on the request's thread,
//! decodes into the server's request arena and holds no state between requests; one image decodes at a time in the
//! process (image.zig `decoding`), with libjpeg-turbo's working memory capped (image.zig `jpeg_memory_mib`).
//!
//! The chat template writes `<|content_image|><|unused_200054|><|end_message|>` for every image part of a user or
//! tool message (`template_wraps`); the server expands the image token to the image's tokens. Refusals with numbers
//! (decision D3): an image over 4,096 pixels a side, or over its share of the request's image tokens, is refused before
//! its pixels are decoded, and `explain` words it with the numbers `prepare` found on the same thread.
const std = @import("std");
const core = @import("core");
pub const image = @import("image.zig");

const steps = image.For(core.picture);

/// What --vision gives the family (native/cuda_options.zig): image tokens a request (default `default_image_tokens`,
/// one 4096 x 4096 or four 2048 x 2048) and images a request.
pub const Limits = struct {
    image_tokens: u32 = default_image_tokens,
    max_images: u32 = 4,

    pub const default_image_tokens: u32 = image.request_tokens;
};

/// The checkpoint's tokens around an image: `<|content_image|>`, the placeholder `<|unused_200054|>`, `<|end_message|>`.
pub const content_image: u32 = 200005;
pub const end_message: u32 = 200010;

/// One request's image, as the offer passes it to the server: the RGB behind its record (image.zig `carry`) in
/// `pixels`, the grid as gh x gw (rows, columns), one token a block, the 62-bit identity and the full digest.
pub const Prepared = struct {
    pixels: []f32,
    gh: u32,
    gw: u32,
    tokens: u32,
    hash: u64,
    content_size: ?[2]u32,
    digest: [32]u8,
};

pub const Vision = struct {
    io: std.Io,
    /// Temporary decode samples (freed before `prepare` returns); the output lives in the request's arena.
    scratch: std.mem.Allocator,
    limits: Limits,
    marks: struct { image: u32, open: u32 = content_image, close: u32 = end_message },
    mark_ids: [3]u32,

    /// The template writes each image's markers itself: the server adds none.
    pub const template_wraps = true;

    pub fn init(io: std.Io, scratch: std.mem.Allocator, image_token: u32, limits: Limits) Vision {
        return .{ .io = io, .scratch = scratch, .limits = limits, .marks = .{ .image = image_token }, .mark_ids = .{ content_image, image_token, end_message } };
    }

    /// This thread's last refusal in words (`explain` reads it right after `prepare` fails on the same thread).
    threadlocal var why: image.Refusal = .{};

    /// An image's bytes at most `max_tokens` tokens (the server's even share of the request's image tokens).
    pub fn prepare(v: *Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) steps.Error!Prepared {
        why = .{};
        const p = try steps.prepare(v.io, a, v.scratch, bytes, max_tokens, v.limits.image_tokens, &why);
        return .{ .pixels = p.floats, .gh = p.grid[0], .gw = p.grid[1], .tokens = p.grid[0] * p.grid[1], .hash = p.identity, .content_size = .{ p.height, p.width }, .digest = p.digest };
    }

    /// The words for a refusal the server cannot word: the limits with their numbers, and the decoder's own.
    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.ImageTooLarge, error.ImageTooManyTokens => if (why.len > 0) why.text() else null,
            // the decoder's size check, when the header read none: the limit without the image's numbers
            error.Oversized => "it is larger than 4096 pixels a side (16 megapixels); send a smaller image",
            error.JpegDecoderMemory => image.jpeg_memory_words,
            else => core.picture.decoderProblem(err),
        };
    }
};

test "the offer's tokens and limits: the template's markers, 10,816 image tokens by default" {
    var v = Vision.init(std.testing.io, std.testing.allocator, 200054, .{});
    try std.testing.expectEqual(@as(u32, 10_816), v.limits.image_tokens);
    try std.testing.expectEqualSlices(u32, &.{ 200005, 200054, 200010 }, &v.mark_ids);
    try std.testing.expect(Vision.template_wraps);
    // an image over the side limit is refused with its numbers, read back through explain
    const head = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x13\x88\x00\x00\x00\x14\x08\x02\x00\x00\x00\xee\x68\x40\xf7\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    try std.testing.expectError(error.ImageTooLarge, v.prepare(std.testing.allocator, head, 10_816));
    try std.testing.expectEqualStrings("it is 5000 x 20 pixels; at most 4096 pixels a side are accepted (send a smaller image)", Vision.explain(error.ImageTooLarge).?);
    try std.testing.expectEqual(@as(?[]const u8, null), Vision.explain(error.Invalid));
    // a JPEG past the decoder's memory cap: refused by name, with the cap
    try std.testing.expectEqualStrings("the JPEG needs more than 100 MiB of decoder memory (a progressive or multi-scan JPEG's whole-image buffer); send a baseline JPEG or a PNG", Vision.explain(error.JpegDecoderMemory).?);
}
