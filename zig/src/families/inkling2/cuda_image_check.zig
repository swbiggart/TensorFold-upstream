//! tf-inkling2-decode's image gate (--exp image, --image FILE@AT): an image's tower rows spliced into a prompt by the
//! serving engine's own path (rank 0's tower and row store, Engine.vision; kind 27 image chunks, Leader.forwardImages),
//! at any chunk cut and resume point, against itself and against the tower's V5 pin.
//!
//! FILE is an image oracle input (`*.png.input.safetensors`: rgb, hw, grid); its T tokens sit at prompt positions
//! [AT, AT + T), whose ids become the image token (config image_token). On rank 0:
//!   1. the tower runs whole into the store (one unit on a region borrowed from the op scratch, as a lane's unit does);
//!      --image-poison 1 then changes one stored row (the negative control: the pin must then differ);
//!   2. slot 4 prefills the prompt in 2048-row chunks, an image chunk as a kind-27 forward, then S - 1 greedy decode
//!      steps: the reference decode logits xxh64;
//!   3. slot 5 prefills it cut at every --cuts position (chunks inside the image, a short one among them, carry their
//!      ids in the header), then decodes: the same hash;
//!   4. at each cut c, slot 4 prefills [0, c), keeps its state (cuda_snap.zig), runs on past c; the state is restored
//!      into slot 5 (poisoned first), which prefills [c, T) (a cut inside the image splices from row c - AT of it) and
//!      decodes: the same hash.
//! The head.image rows every rank's forwards splice are recorded (`Tap`, through Engine.image_hooks): each image
//! position's first rows are kept and every later forward's rows there must be the same bytes. Their SHA-256 over
//! [AT, AT + T) is the tower's rows hash, the V5 pin's (--image-sha) at W = 1 and on every rank at W > 1 (the
//! broadcast); the other ranks print theirs at the end.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fwd = @import("cuda_forward.zig");
const embed = @import("cuda_embed.zig");
const vision = @import("cuda_vision.zig");
const snap = @import("cuda_snap.zig");

const vs = ink.vision_store;
const Leader = follow.Leader;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const row_bytes = vs.row_bytes;
/// The reference prefill's slot, the resumes' source; the cut and resumed runs' slot.
pub const ref_slot: u32 = 4;
pub const run_slot: u32 = 5;

/// --image FILE@AT (the last `@` splits them).
pub const Arg = struct { path: []const u8, at: usize };

pub fn parseArg(text: []const u8) ?Arg {
    const i = std.mem.lastIndexOfScalar(u8, text, '@') orelse return null;
    if (i == 0) return null;
    const at = std.fmt.parseInt(usize, text[i + 1 ..], 10) catch return null;
    return .{ .path = text[0..i], .at = at };
}

/// --image-sha HEX: the 32-byte pin.
pub fn parseSha(text: []const u8) ?[32]u8 {
    if (text.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return null;
    return out;
}

/// An image oracle input: the upright RGB bytes, the size and the grid (checked against the size).
pub const Input = struct {
    h: u32,
    w: u32,
    grid: vision.Grid,
    rgb: []u8,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Input {
        var r = try ink.st.Reader.open(gpa, io, path);
        defer r.close();
        const hw = try r.alloc(gpa, "hw", .i32);
        defer gpa.free(hw);
        if (hw.len != 8) return error.UnexpectedInput;
        const h = std.mem.readInt(i32, hw[0..4], .little);
        const w = std.mem.readInt(i32, hw[4..8], .little);
        if (h <= 0 or w <= 0) return error.UnexpectedInput;
        const gb = try r.alloc(gpa, "grid", .i64);
        defer gpa.free(gb);
        if (gb.len < 16) return error.UnexpectedInput;
        const g = vision.Grid.of(@intCast(h), @intCast(w));
        const rows = std.mem.readInt(i64, gb[gb.len - 16 ..][0..8], .little);
        const cols = std.mem.readInt(i64, gb[gb.len - 8 ..][0..8], .little);
        if (rows != g.rows or cols != g.cols) return error.UnexpectedGrid;
        const rgb = try r.alloc(gpa, "rgb", .u8);
        errdefer gpa.free(rgb);
        if (rgb.len != @as(usize, @intCast(h)) * @as(usize, @intCast(w)) * 3) return error.UnexpectedInput;
        return .{ .h = @intCast(h), .w = @intCast(w), .grid = g, .rgb = rgb };
    }

    pub fn deinit(in: *Input, gpa: std.mem.Allocator) void {
        gpa.free(in.rgb);
    }

    pub fn tokens(in: *const Input) usize {
        return in.grid.tokens();
    }

    /// The store's image at prompt position `at`, named by `digest`.
    pub fn image(in: *const Input, at: usize) vs.Image {
        return .{ .at = @intCast(at), .tokens = @intCast(in.tokens()), .rows = @intCast(in.grid.rows), .cols = @intCast(in.grid.cols), .digest = digest(in.h, in.w, in.rgb) };
    }
};

/// The image's content digest by image.zig's rule (`digest`: SHA-256 of "inkling2-image-1", the height and width as
/// u32 little-endian, then the RGB bytes), as V-b's record carries it.
pub fn digest(h: u32, w: u32, rgb: []const u8) [32]u8 {
    var s = Sha256.init(.{});
    s.update("inkling2-image-1");
    var hw: [8]u8 = undefined;
    std.mem.writeInt(u32, hw[0..4], h, .little);
    std.mem.writeInt(u32, hw[4..8], w, .little);
    s.update(&hw);
    s.update(rgb);
    var d: [32]u8 = undefined;
    s.final(&d);
    return d;
}

/// The rows of a prefill chunk at `at`: up to `to`, at most a prompt chunk, ending at the first cut past `at`.
pub fn chunkRows(at: usize, to: usize, cuts: []const usize) usize {
    var n: usize = @min(to - at, engine.prompt_rows);
    for (cuts) |c| if (c > at and c < at + n) {
        n = c - at;
    };
    return n;
}

/// The cuts when --cuts gives none: the image's first row, its second, its middle, its last and the row after it,
/// and the first chunk's end (those in (0, T)); ascending, once each. Returns the count.
pub fn defaultCuts(at: usize, tokens: usize, T: usize, out: *[6]usize) usize {
    const all = [6]usize{ at, at + 1, at + tokens / 2, at + tokens - 1, at + tokens, engine.prompt_rows };
    var buf = all;
    std.mem.sort(usize, &buf, {}, std.sort.asc(usize));
    var n: usize = 0;
    for (buf) |c| if (c > 0 and c < T and (n == 0 or out[n - 1] != c)) {
        out[n] = c;
        n += 1;
    };
    return n;
}

/// Chunk [at, at + n)'s splice from the pass's stored rows (`Need.plan` pins what it meets; every image must be
/// ready), or null when the chunk meets no image.
pub fn spliceOf(need: *vs.Need, s: *vs.Store, images: []const vs.Image, at: usize, n: usize) !?embed.Splice {
    switch (try need.plan(s, images, @intCast(at), @intCast(n))) {
        .ready => {},
        else => return error.ImageNotReady,
    }
    var sp: embed.Splice = .{};
    const k = try need.spans(s, images, @intCast(at), @intCast(n), sp.spans[0..]);
    if (k == 0) return null;
    sp.count = @intCast(k);
    sp.lo = sp.spans[0].first;
    sp.hi = sp.spans[k - 1].first + sp.spans[k - 1].n;
    return sp;
}

/// head.image's rows, as forwards splice them (any rank: the hooks Engine.image_hooks gives that op alone). Rows
/// at the image's positions [at, at + tokens) are kept the first time and compared after; rows outside it (text
/// between two images' spans) are counted only.
pub const Tap = struct {
    d: ?*const cuda.Driver,
    at: usize,
    tokens: usize,
    ref: []u8,
    seen: []bool,
    host: []u8,
    /// image rows recorded, rows recorded again with other bytes (the first such position), rows outside the image
    rows: usize = 0,
    differ: usize = 0,
    first_differ: ?usize = null,
    outside: usize = 0,
    forwards: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: ?*const cuda.Driver, at: usize, tokens: usize) !Tap {
        const ref = try gpa.alloc(u8, tokens * row_bytes);
        errdefer gpa.free(ref);
        const seen = try gpa.alloc(bool, tokens);
        errdefer gpa.free(seen);
        @memset(seen, false);
        const host: []u8 = if (d != null) try gpa.alloc(u8, engine.prompt_rows * row_bytes) else &.{};
        return .{ .d = d, .at = at, .tokens = tokens, .ref = ref, .seen = seen, .host = host };
    }

    pub fn deinit(t: *Tap, gpa: std.mem.Allocator) void {
        gpa.free(t.ref);
        gpa.free(t.seen);
        if (t.host.len > 0) gpa.free(t.host);
    }

    pub fn hooks(t: *Tap) fwd.Hooks {
        return .{ .ctx = t, .record = record };
    }

    fn record(ctx: *anyopaque, scope: fwd.Scope, op: []const u8, dir: fwd.Dir, tensor: fwd.Tensor, row0: usize) anyerror!void {
        _ = scope;
        if (dir != .out or !std.mem.eql(u8, op, "head.image")) return;
        const t: *Tap = @ptrCast(@alignCast(ctx));
        const m = tensor.m;
        if (m.cols * m.dt.size() != row_bytes or !m.dense() or m.rows > engine.prompt_rows) return error.BadHeadShape;
        const bytes = t.host[0 .. m.rows * row_bytes];
        try cuda.memory.download(t.d.?, m.ptr, bytes); // `after` waited for the stream
        t.take(row0, bytes);
    }

    /// Rows from prompt position `row0` (row_bytes each).
    pub fn take(t: *Tap, row0: usize, bytes: []const u8) void {
        t.forwards += 1;
        for (0..bytes.len / row_bytes) |i| {
            const p = row0 + i;
            const row = bytes[i * row_bytes ..][0..row_bytes];
            if (p < t.at or p >= t.at + t.tokens) {
                t.outside += 1;
                continue;
            }
            const j = p - t.at;
            const kept = t.ref[j * row_bytes ..][0..row_bytes];
            t.rows += 1;
            if (!t.seen[j]) {
                @memcpy(kept, row);
                t.seen[j] = true;
            } else if (!std.mem.eql(u8, kept, row)) {
                t.differ += 1;
                if (t.first_differ == null) t.first_differ = p;
            }
        }
    }

    /// Every image position recorded at least once.
    pub fn complete(t: *const Tap) bool {
        return std.mem.indexOfScalar(bool, t.seen, false) == null;
    }

    /// SHA-256 of the image's rows in order (the tower rows hash, as tf-inkling2-vision --hash prints it).
    pub fn sha(t: *const Tap) [32]u8 {
        var d: [32]u8 = undefined;
        Sha256.hash(t.ref, &d, .{});
        return d;
    }

    /// The result line; true when every position was recorded, every repeat was the same bytes and the hash is the
    /// pin (when one is given).
    pub fn report(t: *const Tap, rank: u32, pin: ?[32]u8) bool {
        const full = t.complete();
        const d = t.sha();
        const pinned = if (pin) |p| std.mem.eql(u8, &p, &d) else true;
        const pass = full and t.differ == 0 and pinned and t.rows > 0;
        const hex = std.fmt.bytesToHex(d, .lower);
        std.debug.print("{s} image rank {d}: head.image rows [{d}, {d}) sha256 {s}{s}; {d} rows in {d} forwards, {d} written again with other bytes", .{ if (pass) "PASS" else "FAIL", rank, t.at, t.at + t.tokens, if (full) @as([]const u8, &hex) else "(not every position spliced)", if (pin == null) ", no pin given" else if (pinned) ", EQUAL to the pin" else ", DIFFERS FROM the pin", t.rows, t.forwards, t.differ });
        if (t.first_differ) |p| std.debug.print(" (the first at position {d})", .{p});
        std.debug.print("; {d} text rows between spans\n", .{t.outside});
        return pass;
    }
};

/// A follower's own verdict at the end of tf-inkling2-decode (rank > 0): its head.image rows' report when the run
/// spliced an image; true without one (a follower checks nothing else).
pub fn followerPass(tap: ?*const Tap, rank: u32, pin: ?[32]u8) bool {
    const t = tap orelse return true;
    return t.report(rank, pin);
}

/// A rank's verdict as it goes through tf-inkling2-decode's last barrier (its gather): 0 passed, 1 failed.
pub fn verdictWord(pass: bool) u32 {
    return @intFromBool(!pass);
}

/// Every rank's word back from that gather: the run passed only if every rank's did, so a follower's FAIL fails its
/// own exit status and rank 0's.
pub fn allPassed(words: []const u32) bool {
    for (words) |w| if (w != 0) return false;
    return true;
}

/// What the gate needs beside the engine.
pub const Opts = struct {
    path: []const u8,
    at: usize,
    /// prompt tokens and decode steps (the check's T and S)
    prompt: usize,
    steps: usize,
    /// the cuts for the cut run and the resumes (in (0, T), ascending)
    cuts: []const usize,
    /// --image-poison: one stored row changed before any forward
    poison_row: bool = false,
    /// the byte a restored slot is poisoned with first (--poison)
    poison: ?u8 = 60,
    pin: ?[32]u8 = null,
};

/// A pass over prompt rows [from, to) of `slot` (chunks of up to 2048, cut at `cuts`), an image chunk as a kind-27
/// forward with its spans from the store; the head over the last row when `to` is the prompt's end.
const Pass = struct {
    x: Leader,
    store: *vs.Store,
    images: []const vs.Image,
    seq: []const u32,
    image_chunks: usize = 0,

    fn prefill(p: *Pass, slot: u32, from: usize, to: usize, cuts: []const usize, head_at_end: bool) !void {
        var need = try vs.Need.init(p.images, @intCast(from));
        defer _ = need.release(p.store, p.images);
        var at = from;
        while (at < to) {
            const n = chunkRows(at, to, cuts);
            const hr: ?follow.Rows = if (head_at_end and at + n == to) .{ .lo = n - 1, .n = 1 } else null;
            const ids = p.seq[at..][0..n];
            if (try spliceOf(&need, p.store, p.images, at, n)) |sp| {
                try p.x.forwardImages(slot, at, ids, hr, &sp);
                p.image_chunks += 1;
            } else try p.x.forward(slot, at, ids, hr);
            _ = need.passed(p.store, p.images, @intCast(at + n));
            at += n;
        }
    }

    /// A slot standing at `from`: the prompt's rest and S - 1 greedy decode steps; the decode logits' xxh64 (the
    /// first token's and every step's).
    fn run(p: *Pass, slot: u32, from: usize, cuts: []const usize, T: usize, S: usize, dec: []f32) !u64 {
        const e = p.x.e;
        const U = e.vocab();
        try p.prefill(slot, from, T, cuts, true);
        @memcpy(dec[0..U], try e.logits());
        for (0..S - 1) |k| {
            const tok = [1]u32{argmax(dec[k * U ..][0..U])};
            try p.x.forward(slot, T + k, &tok, .{ .lo = 0, .n = 1 });
            @memcpy(dec[(k + 1) * U ..][0..U], try e.logits());
        }
        return std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(dec[0 .. S * U]));
    }
};

fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

/// One stored row's bf16 values changed in their lowest bit (--image-poison).
fn poisonRow(e: *engine.Engine, addr: u64) !void {
    var buf: [row_bytes]u8 = undefined;
    try e.stream.synchronize();
    try cuda.memory.download(e.d, addr, &buf);
    var i: usize = 0;
    while (i < row_bytes) : (i += 2) buf[i] ^= 1;
    const view: cuda.DeviceBuffer = .{ .d = e.d, .ptr = addr, .len = row_bytes, .borrowed = true };
    try view.upload(0, &buf);
}

/// The gate on rank 0 (the snapshot arena made by the caller; `tap` on Engine.image_hooks). True when it passed.
pub fn run(gpa: std.mem.Allocator, io: std.Io, x: Leader, prompt: []const u32, in: *const Input, o: Opts, tap: *Tap) !bool {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const v = e.vision orelse {
        std.debug.print("FAIL image: rank 0 holds no vision tower (set TF_INK2_VISION=1 on every rank)\n", .{});
        return error.NoVision;
    };
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    const n_img = in.tokens();
    if (o.at + n_img >= T) {
        std.debug.print("FAIL image: the image's {d} tokens at {d} must end before the prompt's last token ({d})\n", .{ n_img, o.at, T });
        return error.ImageOutsidePrompt;
    }
    if (tap.at != o.at or tap.tokens != n_img) return error.TapNotThisImage;
    const seq = try gpa.alloc(u32, T);
    defer gpa.free(seq);
    @memcpy(seq, prompt[0..T]);
    @memset(seq[o.at..][0..n_img], e.model.c.image_token);
    const images = [1]vs.Image{in.image(o.at)};
    const store = &v.store;

    // 1. the tower, whole, into a new store entry (reserved and pinned for the gate)
    var held = try vs.Need.init(&images, 0);
    try held.reserve(store, &images);
    defer _ = held.release(store, &images);
    if (try held.plan(store, &images, 0, @intCast(T)) != .begin) return error.ImageStored;
    const row0 = store.find(&images[0].digest).?.row0;
    const t0 = clock.now(io).toNanoseconds();
    {
        errdefer store.abort(&images[0].digest);
        e.scratch.reset();
        e.scratch.kind = .other;
        const need = v.tower.sliceNeed(in.grid);
        const base = try e.scratch.take(need);
        _ = try v.whole(base, need, in.rgb, in.h, in.w, store.addr(row0));
        try e.stream.synchronize();
    }
    store.finish(&images[0].digest);
    const tower_ms = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-6;
    std.debug.print("image {s}: {d}x{d}, grid {d}x{d}, {d} tokens at positions [{d}, {d}); the tower whole in one unit ({d:.1} MiB of the op scratch) in {d:.1} ms into store rows [{d}, {d})\n", .{ std.fs.path.basename(o.path), in.w, in.h, in.grid.rows, in.grid.cols, n_img, o.at, o.at + n_img, @as(f64, @floatFromInt(v.tower.sliceNeed(in.grid))) / (1 << 20), tower_ms, row0, row0 + n_img });
    if (o.poison_row) {
        const r = n_img / 2;
        try poisonRow(e, store.addr(row0 + r));
        std.debug.print("image: --image-poison: stored row {d} of the image (position {d}) changed (the pin must differ)\n", .{ r, o.at + r });
    }

    const dec = try gpa.alloc(f32, S * U);
    defer gpa.free(dec);
    var p: Pass = .{ .x = x, .store = store, .images = &images, .seq = seq };
    var pass = true;
    // 2. the reference: slot 4 in 2048-row chunks
    const ref = try p.run(ref_slot, 0, &.{}, T, S, dec);
    std.debug.print("  image reference on slot {d} (chunks of {d}, {d} image chunks): decode logits xxh64 {x:0>16}\n", .{ ref_slot, engine.prompt_rows, p.image_chunks, ref });
    var runs: usize = 0;
    // 3. the same prompt cut at every cut
    if (o.cuts.len > 0) {
        p.image_chunks = 0;
        const h = try p.run(run_slot, 0, o.cuts, T, S, dec);
        const same = h == ref;
        pass = pass and same;
        runs += 1;
        std.debug.print("  image cut at every cut on slot {d} ({d} image chunks): decode logits xxh64 {x:0>16} {s}\n", .{ run_slot, p.image_chunks, h, if (same) "equal" else "DIFFERS" });
    }
    // 4. kept at each cut, restored into a poisoned slot, prefilled on
    for (o.cuts) |c| {
        try p.prefill(ref_slot, 0, c, &.{}, false);
        var k: snap.Kept = undefined;
        try snap.save(x, ref_slot, c, &k);
        defer snap.drop(x, &k);
        try p.prefill(ref_slot, c, @min(T, c + engine.prompt_rows), &.{}, false); // the kept copy must not change
        try snap.restore(x, run_slot, k, o.poison);
        p.image_chunks = 0;
        const h = try p.run(run_slot, c, &.{}, T, S, dec);
        const same = h == ref;
        pass = pass and same;
        runs += 1;
        const where: []const u8 = if (c < o.at) "before the image" else if (c == o.at) "at the image's start" else if (c < o.at + n_img) "inside the image" else "after the image";
        std.debug.print("  image resume at {d} ({s}) into slot {d}{s}: {d} image chunks after it; decode logits xxh64 {x:0>16} {s}\n", .{ c, where, run_slot, if (o.poison != null) " (poisoned)" else "", p.image_chunks, h, if (same) "equal" else "DIFFERS" });
    }
    const rows_ok = tap.report(0, o.pin);
    std.debug.print("{s} image: {d} runs (cuts and resumes) against the reference (layers 0..{d}, W={d}, prompt {d}, {d} steps, image {d} tokens at {d}): every decode logits xxh64 {s} {x:0>16}; the head.image rows {s}\n", .{ if (pass and rows_ok) "PASS" else "FAIL", runs, e.layers - 1, e.o.world, T, S, n_img, o.at, if (pass) "equals" else "NOT ALL EQUAL TO", ref, if (rows_ok) "pass" else "FAIL" });
    return pass and rows_ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// Host tests (no GPU).

const testing = std.testing;

test "inkling2 image check: --image FILE@AT and --image-sha" {
    const a = parseArg("dir/scene_448x336.png.input.safetensors@600").?;
    try testing.expectEqualStrings("dir/scene_448x336.png.input.safetensors", a.path);
    try testing.expectEqual(@as(usize, 600), a.at);
    try testing.expectEqualStrings("a@b", parseArg("a@b@7").?.path);
    try testing.expect(parseArg("file") == null);
    try testing.expect(parseArg("@5") == null);
    try testing.expect(parseArg("file@x") == null);
    const pin = parseSha("596d7df20f858b4b8ca4adb80911da78daed482c25a1bc9c601622ea429115e0").?;
    try testing.expectEqual(@as(u8, 0x59), pin[0]);
    try testing.expectEqual(@as(u8, 0xe0), pin[31]);
    try testing.expect(parseSha("596d") == null);
    try testing.expect(parseSha("zz6d7df20f858b4b8ca4adb80911da78daed482c25a1bc9c601622ea429115e0") == null);
}

test "inkling2 image check: chunks of 2048 rows cut at every cut" {
    const cuts = [_]usize{ 1, 600, 601, 700, 2048 };
    var at: usize = 0;
    var got: [8]usize = undefined;
    var n: usize = 0;
    while (at < 3040) : (n += 1) {
        got[n] = chunkRows(at, 3040, &cuts);
        at += got[n];
    }
    try testing.expectEqualSlices(usize, &.{ 1, 599, 1, 99, 1348, 992 }, got[0..n]);
    // no cuts: the prompt in 2048-row chunks; from a resume point inside the image
    try testing.expectEqual(@as(usize, 2048), chunkRows(0, 3040, &.{}));
    try testing.expectEqual(@as(usize, 2048), chunkRows(650, 3040, &.{}));
    try testing.expectEqual(@as(usize, 50), chunkRows(650, 700, &.{}));
    var dc: [6]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 600, 601, 654, 707, 708, 2048 }, dc[0..defaultCuts(600, 108, 3040, &dc)]);
    try testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, dc[0..defaultCuts(0, 4, 4, &dc)]);
}

fn noCopy(ctx: *anyopaque, dst: u64, src: u64, bytes: usize) anyerror!void {
    _ = ctx;
    _ = dst;
    _ = src;
    _ = bytes;
    return error.Unexpected;
}

fn noDone(ctx: *anyopaque) anyerror!void {
    _ = ctx;
    return error.Unexpected;
}

test "inkling2 image check: a chunk's splice is the stored rows of the image rows it holds, from any cut or resume point" {
    var dummy: u8 = 0;
    const base: u64 = 0x10_0000;
    var s = try vs.Store.init(testing.allocator, base, 1000, row_bytes, .{ .ctx = &dummy, .run = noCopy, .done = noDone });
    defer s.deinit();
    var dg: [32]u8 = @splat(7);
    const images = [1]vs.Image{.{ .at = 600, .tokens = 108, .rows = 9, .cols = 12, .digest = dg }};
    // the tower's entry, as the gate makes it (behind 20 rows of another image)
    var other: [32]u8 = @splat(9);
    _ = try s.alloc(&other, 20);
    s.finish(&other);
    var held = try vs.Need.init(&images, 0);
    try held.reserve(&s, &images);
    try testing.expect(try held.plan(&s, &images, 0, 3040) == .begin);
    const row0 = s.find(&dg).?.row0;
    try testing.expectEqual(@as(u32, 20), row0);
    // not ready yet: a pass may not splice it
    {
        var need = try vs.Need.init(&images, 0);
        try testing.expectError(error.ImageNotReady, spliceOf(&need, &s, &images, 0, 2048));
        _ = need.release(&s, &images);
    }
    s.finish(&dg);
    const cuts = [_]usize{ 1, 600, 601, 700, 2048 };
    const Want = struct { at: usize, lo: u32, hi: u32, row: u64 };
    const want = [_]?Want{ null, null, .{ .at = 600, .lo = 0, .hi = 1, .row = 20 }, .{ .at = 601, .lo = 0, .hi = 99, .row = 21 }, .{ .at = 700, .lo = 0, .hi = 8, .row = 120 }, null };
    var need = try vs.Need.init(&images, 0);
    var at: usize = 0;
    var i: usize = 0;
    while (at < 3040) : (i += 1) {
        const n = chunkRows(at, 3040, &cuts);
        const sp = try spliceOf(&need, &s, &images, at, n);
        if (want[i]) |w| {
            try testing.expectEqual(w.at, at);
            const x = sp.?;
            try testing.expectEqual(@as(u32, 1), x.count);
            try testing.expectEqual(w.lo, x.lo);
            try testing.expectEqual(w.hi, x.hi);
            try testing.expectEqual(s.addr(w.row), x.spans[0].src);
            try x.check(n);
            try x.within(base, 1000 * row_bytes);
        } else try testing.expect(sp == null);
        _ = need.passed(&s, &images, @intCast(at + n));
        at += n;
    }
    _ = need.release(&s, &images);
    // the whole prompt in 2048-row chunks: the image in the first chunk at rows 600 .. 708
    var whole = try vs.Need.init(&images, 0);
    const w0 = (try spliceOf(&whole, &s, &images, 0, 2048)).?;
    try testing.expectEqual(@as(u32, 600), w0.lo);
    try testing.expectEqual(@as(u32, 708), w0.hi);
    try testing.expectEqual(s.addr(20), w0.spans[0].src);
    _ = whole.release(&s, &images);
    // resumed inside the image at 650: the splice starts at the image's row 50
    var res = try vs.Need.init(&images, 650);
    const r0 = (try spliceOf(&res, &s, &images, 650, 2048)).?;
    try testing.expectEqual(@as(u32, 0), r0.lo);
    try testing.expectEqual(@as(u32, 58), r0.hi);
    try testing.expectEqual(s.addr(20 + 50), r0.spans[0].src);
    _ = res.release(&s, &images);
    // resumed past the image: nothing to splice
    var past = try vs.Need.init(&images, 708);
    try testing.expect(try spliceOf(&past, &s, &images, 708, 2048) == null);
    _ = past.release(&s, &images);
    // only the gate's pin is left
    try testing.expectEqual(@as(u32, 1), s.find(&dg).?.pins);
    _ = held.release(&s, &images);
    try testing.expectEqual(@as(u32, 0), s.committed);
}

test "inkling2 image check: the head.image tap keeps each image position's rows and counts any other bytes there" {
    const gpa = testing.allocator;
    var t = try Tap.init(gpa, null, 600, 4);
    defer t.deinit(gpa);
    var rows: [6 * row_bytes]u8 = undefined;
    for (0..6) |r| @memset(rows[r * row_bytes ..][0..row_bytes], @intCast(r + 1)); // position 598 + r holds r + 1
    // a forward of positions 598 .. 601: two text rows, then the image's first two
    t.take(598, rows[0 .. 4 * row_bytes]);
    try testing.expect(!t.complete());
    try testing.expectEqual(@as(usize, 2), t.outside);
    // the rest in two forwards, one of them again over 601 with the same bytes
    t.take(601, rows[3 * row_bytes .. 5 * row_bytes]);
    t.take(603, rows[5 * row_bytes .. 6 * row_bytes]);
    try testing.expect(t.complete());
    try testing.expectEqual(@as(usize, 0), t.differ);
    try testing.expectEqual(@as(usize, 5), t.rows);
    var want: [32]u8 = undefined;
    Sha256.hash(rows[2 * row_bytes .. 6 * row_bytes], &want, .{});
    try testing.expectEqualSlices(u8, &want, &t.sha());
    try testing.expect(t.report(0, want));
    try testing.expect(!t.report(0, @splat(0)));
    // a later forward with one changed byte at position 602
    var bad = rows;
    bad[4 * row_bytes + 17] ^= 1;
    t.take(602, bad[4 * row_bytes .. 5 * row_bytes]);
    try testing.expectEqual(@as(usize, 1), t.differ);
    try testing.expectEqual(@as(?usize, 602), t.first_differ);
    try testing.expectEqualSlices(u8, &want, &t.sha()); // the first rows stay the reference
    try testing.expect(!t.report(0, want));
}

test "inkling2 image check: the digest is image.zig's rule" {
    const rgb = [_]u8{ 1, 2, 3, 4, 5, 6 };
    var s = Sha256.init(.{});
    s.update("inkling2-image-1");
    s.update(&[_]u8{ 1, 0, 0, 0, 2, 0, 0, 0 });
    s.update(&rgb);
    var want: [32]u8 = undefined;
    s.final(&want);
    try testing.expectEqualSlices(u8, &want, &digest(1, 2, &rgb));
    try testing.expect(!std.mem.eql(u8, &digest(1, 2, &rgb), &digest(2, 1, &rgb)));
}

test "inkling2 image check: a follower whose head.image rows differ from the pin fails its own status and, through the last barrier, every rank's" {
    const gpa = testing.allocator;
    var t = try Tap.init(gpa, null, 0, 2);
    defer t.deinit(gpa);
    var rows: [2 * row_bytes]u8 = undefined;
    @memset(&rows, 7);
    t.take(0, &rows);
    var want: [32]u8 = undefined;
    Sha256.hash(&rows, &want, .{});
    // rank 1's own verdict: its rows' report (no image spliced: nothing to fail)
    try testing.expect(followerPass(&t, 1, want));
    try testing.expect(!followerPass(&t, 1, @splat(0)));
    try testing.expect(followerPass(null, 1, null));
    // every rank's verdict comes back from the gather: one failed rank fails them all, rank 0 passing or not
    try testing.expect(allPassed(&.{ verdictWord(true), verdictWord(true) }));
    try testing.expect(!allPassed(&.{ verdictWord(true), verdictWord(followerPass(&t, 1, @splat(0))) }));
    try testing.expect(!allPassed(&.{ verdictWord(false), verdictWord(true), verdictWord(true), verdictWord(true) }));
    // the old payload (each rank's number) is not a verdict: a nonzero word is a failed rank
    try testing.expect(!allPassed(&.{ 0, 1 }));
}
