//! What a CUDA family with an image tower offers the server: its processor, image token, markers and limits.
const std = @import("std");
const api = @import("engine_api");
const qwen35 = @import("qwen35");
const nemotron = @import("nemotron");
const Allocator = std.mem.Allocator;

/// Info.vision for a family's tower (Nemotron Omni's, Qwen's, Flash Next's or GLM's); `open` caps Omni's video counts.
pub fn offer(v: anytype, open: ?api.VisionOpen) ?api.Vision {
    const x = v orelse return null;
    const T = @typeInfo(@TypeOf(x)).pointer.child;
    const prepare = struct {
        fn f(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
            const t: *T = @ptrCast(@alignCast(ctx));
            const p = try t.prepare(a, bytes, max_tokens);
            var out: api.PreparedImage = .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
            // a family that names an image's visible size or full digest passes them on (Inkling-2's)
            if (comptime @hasField(@TypeOf(p), "content_size")) out.content_size = p.content_size;
            if (comptime @hasField(@TypeOf(p), "digest")) out.digest = p.digest;
            return out;
        }
    }.f;
    if (comptime T == qwen35.native.vision.Vision) {
        // the template writes Qwen's vision_start and vision_end around the image token itself
        const image = x.shared.image_token;
        return .{ .ctx = x, .prepare = prepare, .image_token = image, .markers = &x.markers, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = T.explain };
    }
    // a template that writes each image's begin and end itself: the server expands the image token alone
    const wrap: ?[2]u32 = if (@hasDecl(T, "template_wraps")) null else .{ x.marks.open, x.marks.close };
    var out: api.Vision = .{ .ctx = x, .prepare = prepare, .image_token = x.marks.image, .markers = &x.mark_ids, .wrap = wrap, .image_tokens = x.limits.image_tokens, .max_images = x.limits.max_images, .explain = T.explain };
    if (comptime @hasDecl(T, "part_type")) out.part_type = T.part_type; // the template's name for an image part
    if (comptime T == nemotron.native.vision.Vision) {
        // video: the checkpoint's temporal projection, and a reserved token for the template's `<video>`
        if (x.cfg.video_temporal == 2 and x.marks.video != null) if (open) |o| {
            out.video = .{
                .prepare = omniVideo,
                .video_token = x.marks.video.?,
                .video_text = "<video>",
                .group_open = x.marks.open,
                .group_close = x.marks.close,
                .markers = &x.mark_ids,
                .video_tokens = @min(o.video_tokens, x.limits.video_tokens),
                .max_videos = @min(o.max_videos, 16),
                .max_frames = @min(o.max_frames, 256),
            };
        };
        return out;
    }
    if (comptime @hasDecl(T, "prepareVideo")) if (x.video_ids) |*vm| {
        out.video = .{ .prepare = Video(T).prepare, .video_token = vm[1], .group_open = x.marks.open, .group_close = x.marks.close, .markers = vm, .video_tokens = x.limits.video_tokens, .max_videos = x.limits.max_videos, .max_frames = x.limits.max_frames };
    };
    return out;
}

/// A tower's video prepare behind the server's VideoSource: its frames decoded into the family's sink.
fn Video(comptime T: type) type {
    return struct {
        src: api.VideoSource,

        fn decode(c: *anyopaque, indices: []const u32, sink: *T.video.Sink) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(c));
            const into = struct {
                fn take(s: *anyopaque, k: usize, rgb: []const u8, w: u32, h: u32, stride: usize) anyerror!void {
                    const g: *T.video.Sink = @ptrCast(@alignCast(s));
                    try g.take(k, rgb, w, h, stride);
                }
            }.take;
            try self.src.decode(self.src.ctx, indices, .{ .ctx = sink, .take = into });
        }

        fn prepare(ctx: *anyopaque, a: Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
            const t: *T = @ptrCast(@alignCast(ctx));
            var ad: @This() = .{ .src = src };
            const s: T.video.Source = .{ .frames = src.frames, .rate_num = src.rate_num, .rate_den = src.rate_den, .width = src.width, .height = src.height, .ctx = &ad, .decode = decode };
            const p = try t.prepareVideo(a, s, max_tokens, max_frames);
            const groups = try a.alloc(api.PreparedImage, p.groups.len);
            for (groups, p.groups) |*g, q| g.* = .{ .pixels = q.pixels, .gh = q.gh, .gw = q.gw, .tokens = q.tokens, .hash = q.hash };
            return .{ .groups = groups, .times = p.times };
        }
    };
}

/// The server's decoder behind the tower's video source.
const Decoding = struct {
    src: api.VideoSource,

    fn decode(ctx: *anyopaque, indices: []const u32, sink: nemotron.native.vision.video.Sink) anyerror!void {
        const d: *Decoding = @ptrCast(@alignCast(ctx));
        return d.src.decode(d.src.ctx, indices, .{ .ctx = sink.ctx, .take = sink.take });
    }
};

/// Nemotron Omni's video prepare: groups of frames as images, each with its time and the words before it.
fn omniVideo(ctx: *anyopaque, a: Allocator, src: api.VideoSource, max_tokens: u32, max_frames: u32) anyerror!api.PreparedVideo {
    const t: *nemotron.native.vision.Vision = @ptrCast(@alignCast(ctx));
    var d: Decoding = .{ .src = src };
    const p = try t.prepareVideo(a, .{
        .frames = src.frames,
        .rate_num = src.rate_num,
        .rate_den = src.rate_den,
        .width = src.width,
        .height = src.height,
        .ctx = &d,
        .decode = Decoding.decode,
    }, max_tokens, max_frames);
    defer a.free(p.groups);
    errdefer {
        for (p.groups, p.labels) |g, label| {
            a.free(g.pixels);
            a.free(label);
        }
        a.free(p.times);
        a.free(p.labels);
    }
    const groups = try a.alloc(api.PreparedImage, p.groups.len);
    for (groups, p.groups) |*o, g| o.* = .{ .pixels = g.pixels, .gh = g.gh, .gw = g.gw, .tokens = g.tokens, .hash = g.hash };
    return .{ .groups = groups, .times = p.times, .labels = p.labels };
}

/// The device bytes a family's tower takes beside its text weights on this rank, 0 without --vision.
pub fn weightBytes(comptime F: type, a: Allocator, io: std.Io, o: api.Open) !u64 {
    if (comptime @hasDecl(F, "visionBytes")) if (o.vision != null and o.rank == 0) return F.visionBytes(a, io, o.dir);
    return 0;
}

test "a tower whose template writes its own markers gets no wrapper; other towers get theirs" {
    const Marks = struct { open: u32 = 1, image: u32 = 2, close: u32 = 3 };
    const Limits = struct { image_tokens: u32 = 4, max_images: u32 = 2 };
    const Plain = struct {
        marks: Marks = .{},
        mark_ids: [3]u32 = .{ 1, 2, 3 },
        limits: Limits = .{},
        fn prepare(_: *@This(), _: Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
        fn explain(_: anyerror) ?[]const u8 {
            return null;
        }
    };
    const Wraps = struct {
        pub const template_wraps = true;
        pub const part_type = "image";
        marks: Marks = .{},
        mark_ids: [3]u32 = .{ 1, 2, 3 },
        limits: Limits = .{},
        fn prepare(_: *@This(), _: Allocator, _: []const u8, _: u32) !api.PreparedImage {
            return error.Unused;
        }
        fn explain(_: anyerror) ?[]const u8 {
            return null;
        }
    };
    var plain: Plain = .{};
    var own: Wraps = .{};
    try std.testing.expectEqualStrings("image_url", offer(@as(?*Plain, &plain), null).?.part_type);
    try std.testing.expectEqualStrings("image", offer(@as(?*Wraps, &own), null).?.part_type);
    try std.testing.expectEqual([2]u32{ 1, 3 }, offer(@as(?*Plain, &plain), null).?.wrap.?);
    try std.testing.expect(offer(@as(?*Wraps, &own), null).?.wrap == null);
    try std.testing.expectEqual(@as(u32, 2), offer(@as(?*Wraps, &own), null).?.image_token);
}

test "actual offer prepares bounded video and plans mixed rows through the shared Vision owner" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var v: nemotron.native.vision.Vision = undefined;
    v.gpa = a;
    v.spans = &.{};
    v.encoder = null;
    defer v.end();
    v.cfg = .{ .hidden = 1280, .layers = 32, .heads = 16, .ffn = 5120, .eps = 1e-6, .patch = 16, .prefix = 10, .side = 128, .projector_hidden = 20480, .text_hidden = 4096, .final_norm = false, .mean = .{ 0.5, 0.5, 0.5 }, .std = .{ 0.25, 0.25, 0.25 }, .min_patches = 1024, .max_patches = 13312, .max_model_len = 16384, .image_token = 18, .video_temporal = 2 };
    v.marks = .{ .image = 18, .open = 19, .close = 20, .video = 30 };
    v.mark_ids = v.marks.markers();
    v.limits = .{ .image_tokens = 256, .video_tokens = 512, .max_images = 1 };
    const held = offer(@as(?*nemotron.native.vision.Vision, &v), .{ .image_tokens = 256, .video_tokens = 512, .max_frames = 257, .max_videos = 17 }).?;
    try std.testing.expectEqualStrings("image_url", held.part_type);
    try std.testing.expectEqual(@as(u32, 256), held.video.?.max_frames);
    try std.testing.expectEqual(@as(u32, 16), held.video.?.max_videos);
    try std.testing.expectEqual(@as(u32, 512), held.video.?.video_tokens);
    const Fake = struct {
        fn decode(_: *anyopaque, indices: []const u32, sink: api.FrameSink) anyerror!void {
            const pixels: [64 * 64 * 3]u8 = @splat(1);
            for (indices, 0..) |_, i| try sink.take(sink.ctx, i, &pixels, 64, 64, 64 * 3);
        }
    };
    var dummy: u8 = 0;
    const prepared = try held.video.?.prepare(held.ctx, a, .{ .frames = 30, .rate_num = 30, .rate_den = 1, .width = 64, .height = 64, .ctx = &dummy, .decode = Fake.decode }, 512, 256);
    try std.testing.expectEqual(@as(usize, 1), prepared.groups.len);
    const group = prepared.groups[0];
    try std.testing.expectEqual(@as(u32, 256), group.tokens);
    try std.testing.expect(prepared.labels.len == 1 and prepared.times.len == 1);
    v.rows = .{ .d = undefined, .buf = undefined, .hidden = 4096, .limit = try nemotron.native.vision.reservation(v.limits, true) };
    v.tower.c.text_hidden = 1;
    const images = [_]@import("engine_api").Image{
        .{ .pixels = group.pixels, .gh = group.gh, .gw = group.gw, .tokens = group.tokens, .at = 10 },
        .{ .pixels = group.pixels, .gh = group.gh, .gw = group.gw, .tokens = group.tokens, .at = 400 },
    };
    try std.testing.expectError(error.RowShape, v.begin(undefined, &images, 0));
    try std.testing.expectEqual(@as(usize, 2), v.spans.len);
    try std.testing.expectEqual(@as(u32, 256), v.spans[1].row);
    try std.testing.expectError(error.RowShape, v.begin(undefined, &images, 300));
    try std.testing.expectEqual(@as(usize, 1), v.spans.len);
    try std.testing.expectEqual(@as(u32, 1), v.spans[0].image);
    try std.testing.expectEqual(@as(u32, 0), v.spans[0].row);
    try std.testing.expect(offer(@as(?*nemotron.native.vision.Vision, null), null) == null);
}

fn allocationVideo(a: Allocator) !void {
    var v: nemotron.native.vision.Vision = undefined;
    v.cfg = .{ .hidden = 1280, .layers = 32, .heads = 16, .ffn = 5120, .eps = 1e-6, .patch = 1, .prefix = 10, .side = 128, .projector_hidden = 20480, .text_hidden = 4096, .final_norm = false, .mean = .{ 0.5, 0.5, 0.5 }, .std = .{ 0.25, 0.25, 0.25 }, .min_patches = 4, .max_patches = 13312, .max_model_len = 16384, .image_token = 18, .video_temporal = 2, .video_target_patches = 4 };
    const Fake = struct {
        fn decode(_: *anyopaque, indices: []const u32, sink: api.FrameSink) anyerror!void {
            const pixels: [4 * 4 * 3]u8 = @splat(1);
            for (indices, 0..) |_, i| try sink.take(sink.ctx, i, &pixels, 4, 4, 12);
        }
    };
    var dummy: u8 = 0;
    const p = try omniVideo(&v, a, .{ .frames = 30, .rate_num = 30, .rate_den = 1, .width = 4, .height = 4, .ctx = &dummy, .decode = Fake.decode }, 512, 256);
    defer {
        for (p.groups, p.labels) |g, label| {
            a.free(g.pixels);
            a.free(label);
        }
        a.free(p.groups);
        a.free(p.times);
        a.free(p.labels);
    }
    try std.testing.expectEqual(@as(usize, 1), p.groups.len);
}
test "offer conversion unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationVideo, .{});
}

test "actual Qwen offer keeps its image_url type and template-owned markers" {
    var q: qwen35.native.vision.Vision = undefined;
    q.shared.image_token = 57;
    q.markers = .{ 56, 57, 58 };
    q.limits = .{ .image_tokens = 64, .max_images = 2 };
    const held = offer(@as(?*qwen35.native.vision.Vision, &q), null).?;
    try std.testing.expectEqualStrings("image_url", held.part_type);
    try std.testing.expectEqual(@as(u32, 57), held.image_token);
    try std.testing.expectEqualSlices(u32, &q.markers, held.markers);
    try std.testing.expect(held.wrap == null and held.video == null);
}

test "inkling2 on CUDA: the image offer gives the server the RGB record, its size, digest and grid, and refusals with their numbers" {
    const offer_mod = @import("inkling2_serve").image_offer;
    const image = offer_mod.image;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var v = offer_mod.Vision.init(std.testing.io, std.testing.allocator, 200054, .{});
    const x = offer(@as(?*offer_mod.Vision, &v), null).?;
    try std.testing.expectEqual(@as(u32, 200054), x.image_token);
    try std.testing.expectEqualSlices(u32, &.{ 200005, 200054, 200010 }, x.markers);
    try std.testing.expectEqual(@as(?[2]u32, null), x.wrap); // the template writes its own markers
    try std.testing.expectEqual(@as(u32, 10_816), x.image_tokens);
    // a 17 x 9 PNG with clear alpha (fixtures/inkling2/images.json png_rgba_clear_17x9): one token, the checkerboard
    const png = "iVBORw0KGgoAAAANSUhEUgAAABEAAAAJCAYAAADU6McMAAABT0lEQVR4nC3SPUjVYRTH8c+Ni9xMu0fFQSgTIxIXCZoabNAQAgkaegEhcJUmt6KXQa3IphoKEl2iQR2C1EWQwiIaDBKiIdJFFLkEUcHlIhL/v89ZHjgcvud7fjwFqso+C3tCk7AqPBDWhS5hU/gpHBN6he/Cc+GQMCXUFJ34i1m8wwusoA6bKKMZ93ANDXiCkfR+RUFRRzcy0AQO46ODOodOFPEFC+jBJfzAIM7ifAZ5hNt4jBYM4zWaMINb+J2296If9biQQG8yyC5OYQ1tGEo27fmWg3qG0WR1H/9wJPW6M8g+dnAVZ9JQCeO4iJuYTyaT2EMt9bbwKYMsYiAFW0VrOukuGnEyab/N1bmDX3ifh8ofBRvLyi4LK0JFKAkPhVfChnBamBPWhG/CuDAhtAnX8+9Q1PESSziOMfRhO+UznevyFFeS4Q0cRSWd+8F/F95Oh7xm2WAAAAAASUVORK5CYII=";
    const bytes = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(png));
    try std.base64.standard.Decoder.decode(bytes, png);
    const p = try x.prepare(x.ctx, a, bytes, 10_816);
    try std.testing.expectEqual(@as(u32, 1), p.tokens);
    try std.testing.expectEqual(@as(u32, 1), p.gh);
    try std.testing.expectEqual(@as(?[2]u32, .{ 9, 17 }), p.content_size);
    const r = try image.record(.{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens });
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(r.rgb, &h, .{});
    try std.testing.expectEqualStrings("ed09ac3a10d6174370d6d363de943d8ef87cc24d60202670718667e583b74937", &std.fmt.bytesToHex(h, .lower));
    try std.testing.expectEqualSlices(u8, &r.rec.digest, &p.digest);
    try std.testing.expectEqualSlices(u8, &image.digest(9, 17, r.rgb), &p.digest);
    try std.testing.expect(p.hash != 0 and p.hash < 1 << 62);
    // above the side limit: refused from the header, worded with the numbers through explain
    const big = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x13\x88\x00\x00\x00\x14\x08\x02\x00\x00\x00\xee\x68\x40\xf7\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const err = if (x.prepare(x.ctx, a, big, 10_816)) |_| error.TestUnexpectedResult else |e| e;
    try std.testing.expectEqualStrings("it is 5000 x 20 pixels; at most 4096 pixels a side are accepted (send a smaller image)", x.explain.?(err).?);
    // over this image's share of the request's tokens: 2000 x 20 is 50 tokens, with 49 allowed
    const wide = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x07\xd0\x00\x00\x00\x14\x08\x02\x00\x00\x00\x5a\x84\x31\x02\x00\x00\x00\x07\x49\x44\x41\x54\x67\x61\x72\x62\x61\x67\x65\x88\xa3\x30\x37\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";
    const err2 = if (x.prepare(x.ctx, a, wide, 49)) |_| error.TestUnexpectedResult else |e| e;
    try std.testing.expectEqualStrings("2000 x 20 pixels is 50 image tokens (1 x 50 blocks of 40 pixels); this request allows 49 an image (10816 a request, shared evenly by its images)", x.explain.?(err2).?);
}
