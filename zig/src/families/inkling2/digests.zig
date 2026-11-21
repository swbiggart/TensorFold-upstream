//! Content digests of a rank plan's pieces: xxh64 (seed 0) of each piece's bytes as the loader reads them (a
//! strided run's rows in order). `tf-inkling2-plan --dir SOURCE --world W --digests OUT` hashes every rank's pieces
//! from the source checkpoint into one manifest all ranks share; tf-inkling2-cuda hashes the bytes as it loads them
//! (cuda_weights.zig, no second pass) and checks its own pieces against the manifest's lines for its rank.
//!
//! xxh64, not std's XxHash3: in Zig 0.17 XxHash3's streaming update runs at about 0.3 GB/s, XxHash64's near 9 GB/s.
//!
//! Manifest: a header line, then `rank file offset len stride count tensor xxh64` (tab-separated, xxh64 in hex).
const std = @import("std");
const describe = @import("describe.zig");
const plan = @import("plan.zig");

pub const header = "# inkling2 content digests: xxh64 (seed 0) of each piece's bytes in load order\n# rank\tfile\toffset\tlen\tstride\tcount\ttensor\txxh64\n";

/// A loaded piece (plan.Rank.pieces[piece]) and the xxh64 of its bytes.
pub const Digest = struct { piece: u32, xxh64: u64 };

/// The run's bytes, rows in order, read `buf.len` at a time.
pub fn hashRun(io: std.Io, f: std.Io.File, r: plan.Run, buf: []u8) !u64 {
    var h = std.hash.XxHash64.init(0);
    if (r.count == 1) {
        var done: u64 = 0;
        while (done < r.len) {
            const n: usize = @intCast(@min(r.len - done, buf.len));
            if (try f.readPositionalAll(io, buf[0..n], r.offset + done) != n) return error.ShortRead;
            h.update(buf[0..n]);
            done += n;
        }
    } else {
        if (r.len > buf.len) return error.RunTooLong;
        const per: u64 = @max(1, buf.len / r.len);
        var k: u64 = 0;
        while (k < r.count) {
            const n = @min(per, r.count - k);
            for (0..n) |j| {
                const at: usize = @intCast(j * r.len);
                if (try f.readPositionalAll(io, buf[at..][0..@intCast(r.len)], r.offset + (k + j) * r.stride) != r.len) return error.ShortRead;
            }
            h.update(buf[0..@intCast(n * r.len)]);
            k += n;
        }
    }
    if (comptime @import("builtin").os.tag == .linux) {
        const span = (r.count - 1) * r.stride + r.len;
        _ = std.os.linux.fadvise(f.handle, @intCast(r.offset), @intCast(span), std.os.linux.POSIX_FADV.DONTNEED);
    }
    return h.final();
}

/// Whether piece `p` of the plan is in layers [lo, hi) (null: every piece).
fn wanted(m: *const describe.Model, p: plan.Piece, layers: ?[2]u32) bool {
    const r = layers orelse return true;
    const li = m.roles[p.tensor].layer orelse return false;
    return li >= r[0] and li < r[1];
}

const Job = struct {
    io: std.Io,
    dir: []const u8,
    m: *const describe.Model,
    runs: []const plan.Run,
    out: []u64,
    next: std.atomic.Value(usize) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),

    fn work(j: *Job, gpa: std.mem.Allocator, failed: *?anyerror) void {
        j.loop(gpa) catch |e| {
            failed.* = e;
            j.next.store(j.runs.len, .release);
        };
    }

    fn loop(j: *Job, gpa: std.mem.Allocator) !void {
        const buf = try gpa.alloc(u8, 64 << 20);
        defer gpa.free(buf);
        const files = try gpa.alloc(?std.Io.File, j.m.inv.files.len);
        defer gpa.free(files);
        @memset(files, null);
        defer for (files) |f| if (f) |x| x.close(j.io);
        while (true) {
            const i = j.next.fetchAdd(1, .acq_rel);
            if (i >= j.runs.len) return;
            const r = j.runs[i];
            if (files[r.file] == null) {
                const path = try std.fs.path.join(gpa, &.{ j.dir, j.m.inv.files[r.file].name });
                defer gpa.free(path);
                files[r.file] = try std.Io.Dir.cwd().openFile(j.io, path, .{});
            }
            j.out[i] = try hashRun(j.io, files[r.file].?, r, buf);
            _ = j.bytes.fetchAdd(r.bytes(), .acq_rel);
        }
    }
};

/// Every rank's pieces (in layers [lo, hi) when given) hashed from checkpoint `dir` with `jobs` threads, written as a
/// manifest to `w`; a run several ranks share (a replicated tensor) is read once.
pub fn write(gpa: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, dir: []const u8, m: *const describe.Model, layers: ?[2]u32, jobs: usize) !void {
    var index: std.AutoHashMapUnmanaged(plan.Run, usize) = .empty;
    defer index.deinit(gpa);
    var runs: std.ArrayList(plan.Run) = .empty;
    defer runs.deinit(gpa);
    var total: u64 = 0;
    for (m.p.ranks) |rank| for (rank.pieces) |p| if (wanted(m, p, layers)) {
        const gop = try index.getOrPut(gpa, p.run);
        if (!gop.found_existing) {
            gop.value_ptr.* = runs.items.len;
            try runs.append(gpa, p.run);
            total += p.run.bytes();
        }
    };
    // largest first, so the last runs left are small ones
    const order = try gpa.alloc(usize, runs.items.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, runs.items, struct {
        fn lt(rs: []const plan.Run, a: usize, b: usize) bool {
            return rs[a].bytes() > rs[b].bytes();
        }
    }.lt);
    const sorted = try gpa.alloc(plan.Run, runs.items.len);
    defer gpa.free(sorted);
    for (sorted, order) |*s, o| s.* = runs.items[o];
    const out = try gpa.alloc(u64, runs.items.len);
    defer gpa.free(out);
    var job: Job = .{ .io = io, .dir = dir, .m = m, .runs = sorted, .out = out };
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(io).toNanoseconds();
    std.debug.print("hashing {d} runs, {d:.2} GiB, {d} threads\n", .{ runs.items.len, @as(f64, @floatFromInt(total)) / (1 << 30), jobs });
    const threads = try gpa.alloc(std.Thread, @max(1, jobs));
    defer gpa.free(threads);
    const failed = try gpa.alloc(?anyerror, threads.len);
    defer gpa.free(failed);
    @memset(failed, null);
    for (threads, failed) |*t, *e| t.* = try std.Thread.spawn(.{}, Job.work, .{ &job, gpa, e });
    var last: i96 = t0;
    while (job.next.load(.acquire) < sorted.len) {
        std.Io.sleep(io, .fromMilliseconds(500), .awake) catch {};
        const now = clock.now(io).toNanoseconds();
        if (now - last > 30 * std.time.ns_per_s) {
            last = now;
            const b = job.bytes.load(.acquire);
            std.debug.print("  {d:.1} of {d:.1} GiB, {d:.0} MB/s\n", .{ @as(f64, @floatFromInt(b)) / (1 << 30), @as(f64, @floatFromInt(total)) / (1 << 30), @as(f64, @floatFromInt(b)) / (@as(f64, @floatFromInt(now - t0)) * 1e-3) });
        }
    }
    for (threads) |t| t.join();
    for (failed) |f| if (f) |e| return e;
    const s = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-9;
    std.debug.print("hashed {d:.2} GiB in {d:.1} s ({d:.0} MB/s)\n", .{ @as(f64, @floatFromInt(total)) / (1 << 30), s, @as(f64, @floatFromInt(total)) / s / 1e6 });
    // sorted position of each unique run
    const at = try gpa.alloc(usize, runs.items.len);
    defer gpa.free(at);
    for (order, 0..) |o, i| at[o] = i;
    try w.writeAll(header);
    for (m.p.ranks) |rank| for (rank.pieces) |p| if (wanted(m, p, layers)) {
        const r = p.run;
        try w.print("{d}\t{s}\t{d}\t{d}\t{d}\t{d}\t{s}\t{x:0>16}\n", .{ rank.rank, m.inv.files[r.file].name, r.offset, r.len, r.stride, r.count, m.inv.tensors[p.tensor].name, out[at[index.get(r).?]] });
    };
}

pub const Checked = struct {
    pieces: usize = 0,
    differ: usize = 0,
    missing: usize = 0,
    /// xxh64 over the loaded pieces' digests in plan order: this rank's loaded contents in one number.
    loaded: u64 = 0,
    /// The first piece that differs or is missing.
    first: ?[]const u8 = null,
};

/// Rank `rank`'s loaded digests `got` against manifest `text`: every loaded piece must have a line for this rank with
/// the same file, run and digest. Mismatches are logged; the caller fails on `differ + missing > 0`.
pub fn check(gpa: std.mem.Allocator, text: []const u8, m: *const describe.Model, rank: u32, got: []const Digest) !Checked {
    const Line = struct { file: []const u8, run: [4]u64, xxh64: u64 };
    var map: std.StringHashMapUnmanaged(Line) = .empty;
    defer map.deinit(gpa);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var f: [8][]const u8 = undefined;
        var it = std.mem.splitScalar(u8, line, '\t');
        for (&f) |*x| x.* = it.next() orelse return error.BadDigestManifest;
        if (it.next() != null) return error.BadDigestManifest;
        if ((std.fmt.parseInt(u32, f[0], 10) catch return error.BadDigestManifest) != rank) continue;
        var run: [4]u64 = undefined;
        for (&run, f[2..6]) |*v, s| v.* = std.fmt.parseInt(u64, s, 10) catch return error.BadDigestManifest;
        try map.put(gpa, f[6], .{ .file = f[1], .run = run, .xxh64 = std.fmt.parseInt(u64, f[7], 16) catch return error.BadDigestManifest });
    }
    var out: Checked = .{};
    var all = std.hash.XxHash64.init(0);
    const pieces = m.p.ranks[rank].pieces;
    for (got) |g| {
        const p = pieces[g.piece];
        const name = m.inv.tensors[p.tensor].name;
        all.update(std.mem.asBytes(&g.xxh64));
        out.pieces += 1;
        const e = map.get(name) orelse {
            std.log.err("rank {d}: {s} has no line in the digest manifest", .{ rank, name });
            out.missing += 1;
            if (out.first == null) out.first = name;
            continue;
        };
        const r = p.run;
        if (!std.mem.eql(u8, e.file, m.inv.files[r.file].name) or !std.mem.eql(u64, &e.run, &.{ r.offset, r.len, r.stride, r.count })) {
            std.log.err("rank {d}: {s}: the manifest's run is not this plan's (another world or checkpoint)", .{ rank, name });
            out.missing += 1;
            if (out.first == null) out.first = name;
            continue;
        }
        if (e.xxh64 != g.xxh64) {
            std.log.err("rank {d}: {s}: loaded bytes hash {x:0>16}, the manifest says {x:0>16}", .{ rank, name, g.xxh64, e.xxh64 });
            out.differ += 1;
            if (out.first == null) out.first = name;
        }
    }
    out.loaded = all.final();
    return out;
}

/// A tensor read whole outside the rank plan (the vision tower's file, cuda_vision.zig): its checkpoint name, its
/// length and the xxh64 of its bytes.
pub const Named = struct { name: []const u8, len: u64, xxh64: u64 };

/// Tensors read whole outside the plan (`got`) against manifest `text`'s lines of rank `rank` whose tensor starts with
/// `prefix`, by name, len and xxh64 alone: the file and offset columns are not compared, so a side file holding the
/// checkpoint's bytes under their names matches. Every such line must be one whole run (count 1, stride = len), else
/// error.BadDigestManifest. A tensor without a line, or a line no tensor matched, is missing. Each mismatch is logged as
/// a warning; the caller fails (and logs its error) on `differ + missing > 0`.
pub fn checkNamed(gpa: std.mem.Allocator, text: []const u8, rank: u32, prefix: []const u8, got: []const Named) !Checked {
    const Line = struct { len: u64, xxh64: u64, seen: bool = false };
    var map: std.StringHashMapUnmanaged(Line) = .empty;
    defer map.deinit(gpa);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var f: [8][]const u8 = undefined;
        var it = std.mem.splitScalar(u8, line, '\t');
        for (&f) |*x| x.* = it.next() orelse return error.BadDigestManifest;
        if (it.next() != null) return error.BadDigestManifest;
        if ((std.fmt.parseInt(u32, f[0], 10) catch return error.BadDigestManifest) != rank) continue;
        if (!std.mem.startsWith(u8, f[6], prefix)) continue;
        var run: [4]u64 = undefined; // offset, len, stride, count
        for (&run, f[2..6]) |*v, s| v.* = std.fmt.parseInt(u64, s, 10) catch return error.BadDigestManifest;
        if (run[3] != 1 or run[2] != run[1]) return error.BadDigestManifest;
        const gop = try map.getOrPut(gpa, f[6]);
        if (gop.found_existing) return error.BadDigestManifest;
        gop.value_ptr.* = .{ .len = run[1], .xxh64 = std.fmt.parseInt(u64, f[7], 16) catch return error.BadDigestManifest };
    }
    var out: Checked = .{};
    var all = std.hash.XxHash64.init(0);
    for (got) |g| {
        all.update(std.mem.asBytes(&g.xxh64));
        out.pieces += 1;
        const e = map.getPtr(g.name) orelse {
            std.log.warn("rank {d}: {s} has no line in the digest manifest", .{ rank, g.name });
            out.missing += 1;
            if (out.first == null) out.first = g.name;
            continue;
        };
        e.seen = true;
        if (e.len != g.len or e.xxh64 != g.xxh64) {
            std.log.warn("rank {d}: {s}: {d} bytes hash {x:0>16}, the manifest says {d} bytes {x:0>16}", .{ rank, g.name, g.len, g.xxh64, e.len, e.xxh64 });
            out.differ += 1;
            if (out.first == null) out.first = g.name;
        }
    }
    var it = map.iterator();
    while (it.next()) |kv| if (!kv.value_ptr.seen) {
        std.log.warn("rank {d}: the digest manifest's {s} was not read", .{ rank, kv.key_ptr.* });
        out.missing += 1;
        if (out.first == null) out.first = kv.key_ptr.*;
    };
    out.loaded = all.final();
    return out;
}

test "inkling2 digests: named tensors match by name, len and xxh64; a missing or extra one counts, a strided line refuses" {
    const gpa = std.testing.allocator;
    const level = std.testing.log_level;
    std.testing.log_level = .err; // the mismatches below are warned about, as they should be
    defer std.testing.log_level = level;
    const text = header ++
        "0\tmodel-00024.safetensors\t100\t8\t8\t1\tmodel.visual.a\t00000000000000aa\n" ++
        "0\tmodel-00024.safetensors\t108\t4\t4\t1\tmodel.visual.b\t00000000000000bb\n" ++
        "0\tmodel-00001.safetensors\t0\t2\t4\t3\tmodel.llm.x\t0000000000000011\n" ++ // another prefix: not looked at
        "1\tmodel-00024.safetensors\t100\t8\t8\t1\tmodel.visual.c\t00000000000000cc\n"; // another rank
    const p = "model.visual.";
    const both = [_]Named{ .{ .name = "model.visual.a", .len = 8, .xxh64 = 0xaa }, .{ .name = "model.visual.b", .len = 4, .xxh64 = 0xbb } };
    {
        const c = try checkNamed(gpa, text, 0, p, &both);
        try std.testing.expectEqual(@as(usize, 2), c.pieces);
        try std.testing.expectEqual(@as(usize, 0), c.differ + c.missing);
        try std.testing.expect(c.first == null);
    }
    {
        var one = both;
        one[1].xxh64 = 0xbc;
        const c = try checkNamed(gpa, text, 0, p, &one);
        try std.testing.expectEqual(@as(usize, 1), c.differ);
        try std.testing.expectEqualStrings("model.visual.b", c.first.?);
        one[1] = both[1];
        one[0].len = 9; // the same hash at another length
        try std.testing.expectEqual(@as(usize, 1), (try checkNamed(gpa, text, 0, p, &one)).differ);
    }
    {
        // a line the file lacks, and a tensor with no line
        const c = try checkNamed(gpa, text, 0, p, both[0..1]);
        try std.testing.expectEqual(@as(usize, 1), c.missing);
        try std.testing.expectEqualStrings("model.visual.b", c.first.?);
        const extra = [_]Named{ both[0], both[1], .{ .name = "model.visual.z", .len = 1, .xxh64 = 1 } };
        const d = try checkNamed(gpa, text, 0, p, &extra);
        try std.testing.expectEqual(@as(usize, 1), d.missing);
        try std.testing.expectEqual(@as(usize, 0), d.differ);
        // rank 1's lines are rank 1's: neither tensor has one, and its own line is not read
        try std.testing.expectEqual(@as(usize, 3), (try checkNamed(gpa, text, 1, p, &both)).missing);
    }
    // a prefix line that is not one whole run, or named twice
    try std.testing.expectError(error.BadDigestManifest, checkNamed(gpa, text ++ "0\tf\t0\t2\t2\t2\tmodel.visual.d\t01\n", 0, p, &both));
    try std.testing.expectError(error.BadDigestManifest, checkNamed(gpa, text ++ "0\tf\t0\t2\t4\t1\tmodel.visual.d\t01\n", 0, p, &both));
    try std.testing.expectError(error.BadDigestManifest, checkNamed(gpa, text ++ "0\tf\t9\t8\t8\t1\tmodel.visual.a\t00000000000000aa\n", 0, p, &both));
}

test "inkling2 digests: streaming xxh64 does not depend on the read size" {
    var data: [1000]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i *% 131);
    var h = std.hash.XxHash64.init(0);
    var at: usize = 0;
    while (at < data.len) : (at += 77) h.update(data[at..@min(data.len, at + 77)]);
    try std.testing.expectEqual(std.hash.XxHash64.hash(0, &data), h.final());
}
