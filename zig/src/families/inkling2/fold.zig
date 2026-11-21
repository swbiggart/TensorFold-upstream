//! The DSpark drafter's head folds (cuda_dspark.zig loads them): fold_w = unembed . hidden_out.weight [V, 256] and
//! fold_b = unembed . hidden_out.bias [V], each element a sum over the hidden size H of bf16 x bf16 products in fp64,
//! stored bf16. The products are exact in fp64; the sum runs h = 0 .. H-1 in that fixed order from +0. The store
//! rounds fp64 to fp32 and then fp32 to bf16, each to nearest even: what torch's `.to(torch.bfloat16)` does to an
//! fp64 tensor (its BFloat16 is built from a float), so the result matches `inkling2_dspark_ref.py fold` wherever the
//! fp64 sums agree (they agree whenever the sum is exact in fp64, which a dot product of bf16 values almost always
//! is; an inexact one can still differ from a BLAS's order in its last fp64 bit).
//!
//! Inputs: the checkpoint's `model.llm.unembed.weight` (bf16 [V, H], in the shard model.safetensors.index.json names)
//! and mtp.safetensors' `markov_head.hidden_out.weight` (bf16 or f32 [H, 256]) and `.bias` ([H]). A rank folder holds
//! only its own vocabulary rows; `rows` then names them (the rank's quarter at W = 4) and only those are computed.
const std = @import("std");
const st = @import("st.zig");

pub const rank_dim = 256;
pub const unembed_name = "model.llm.unembed.weight";
pub const weight_name = "markov_head.hidden_out.weight";
pub const bias_name = "markov_head.hidden_out.bias";

const VL = 4; // f64 lanes a vector
const Vec = @Vector(VL, f64);
const tile_rows = 4;
const tile_vecs = 4;
const tile_cols = tile_vecs * VL; // 16 output columns a tile
const cols = rank_dim + 1; // fold_w's 256 and fold_b as column 256
const chunks = (cols + tile_cols - 1) / tile_cols; // 17
const block_rows = 64; // rows a work item

/// hidden_out as fp64, extended by the bias as column 256 and zero padding, laid out chunk by chunk:
/// w[(k H + h) tile_cols + j] = column k tile_cols + j of row h.
pub const Head = struct {
    hidden: usize,
    w: []f64,

    pub fn init(gpa: std.mem.Allocator, hidden: usize, weight: []const f64, bias: []const f64) !Head {
        if (weight.len != hidden * rank_dim or bias.len != hidden) return error.BadFoldInputs;
        const w = try gpa.alloc(f64, chunks * hidden * tile_cols);
        @memset(w, 0);
        for (0..hidden) |h| {
            for (0..cols) |c| {
                const v = if (c < rank_dim) weight[h * rank_dim + c] else bias[h];
                w[((c / tile_cols) * hidden + h) * tile_cols + c % tile_cols] = v;
            }
        }
        return .{ .hidden = hidden, .w = w };
    }

    pub fn deinit(h: *Head, gpa: std.mem.Allocator) void {
        gpa.free(h.w);
    }
};

/// fp64 to bf16 bits as torch rounds an fp64 tensor to bf16: to fp32 (nearest even), then to bf16 (nearest even).
pub fn f64ToBf16(x: f64) u16 {
    return st.f32ToBf16(@floatCast(x));
}

/// The folds of `n` rows: `u` holds them as fp64 [n][H]; out_w gets [n][256] bf16 bits, out_b [n].
pub fn rows(head: *const Head, u: []const f64, n: usize, out_w: []u16, out_b: []u16) void {
    const H = head.hidden;
    var q: usize = 0;
    while (q < n) : (q += tile_rows) {
        const nr: usize = @min(tile_rows, n - q);
        var src: [tile_rows][]const f64 = undefined;
        for (0..tile_rows) |r| src[r] = u[(q + @min(r, nr - 1)) * H ..][0..H];
        for (0..chunks) |k| {
            var acc: [tile_rows][tile_vecs]Vec = @splat(@splat(@splat(0.0)));
            const wk = head.w[k * H * tile_cols ..][0 .. H * tile_cols];
            for (0..H) |h| {
                var wv: [tile_vecs]Vec = undefined;
                inline for (0..tile_vecs) |c| wv[c] = wk[h * tile_cols + c * VL ..][0..VL].*;
                inline for (0..tile_rows) |r| {
                    const s: Vec = @splat(src[r][h]);
                    inline for (0..tile_vecs) |c| acc[r][c] = acc[r][c] + s * wv[c];
                }
            }
            for (0..nr) |r| {
                const row = q + r;
                inline for (0..tile_vecs) |c| {
                    const a: [VL]f64 = acc[r][c];
                    for (0..VL) |l| {
                        const col = k * tile_cols + c * VL + l;
                        if (col < rank_dim) out_w[row * rank_dim + col] = f64ToBf16(a[l]) else if (col == rank_dim) out_b[row] = f64ToBf16(a[l]);
                    }
                }
            }
        }
    }
}

/// Rows [lo, hi) of the unembed, read from `file` at `base` (the tensor's first byte), folded with `jobs` threads into
/// out_w ([hi - lo][256] bf16 bits) and out_b ([hi - lo]). Returns the number of all-zero unembed rows seen.
pub fn run(gpa: std.mem.Allocator, io: std.Io, file: std.Io.File, base: u64, head: *const Head, lo: usize, hi: usize, out_w: []u16, out_b: []u16, jobs: usize) !usize {
    const Job = struct {
        io: std.Io,
        file: std.Io.File,
        base: u64,
        head: *const Head,
        lo: usize,
        hi: usize,
        out_w: []u16,
        out_b: []u16,
        next: std.atomic.Value(usize) = .init(0),
        zero: std.atomic.Value(usize) = .init(0),

        fn work(j: *@This(), g: std.mem.Allocator, failed: *?anyerror) void {
            j.loop(g) catch |e| {
                failed.* = e;
                j.next.store(j.hi, .release);
            };
        }

        fn loop(j: *@This(), g: std.mem.Allocator) !void {
            const H = j.head.hidden;
            const raw = try g.alloc(u8, block_rows * H * 2);
            defer g.free(raw);
            const u = try g.alloc(f64, block_rows * H);
            defer g.free(u);
            while (true) {
                const r0 = j.lo + j.next.fetchAdd(block_rows, .acq_rel);
                if (r0 >= j.hi) return;
                const n: usize = @min(block_rows, j.hi - r0);
                const bytes = raw[0 .. n * H * 2];
                if (try j.file.readPositionalAll(j.io, bytes, j.base + @as(u64, r0) * H * 2) != bytes.len) return error.ShortRead;
                var zero: usize = 0;
                for (0..n) |r| {
                    var any: u16 = 0;
                    for (0..H) |h| {
                        const b = std.mem.readInt(u16, bytes[(r * H + h) * 2 ..][0..2], .little);
                        any |= b & 0x7fff;
                        u[r * H + h] = st.bf16ToF32(b);
                    }
                    if (any == 0) zero += 1;
                }
                _ = j.zero.fetchAdd(zero, .acq_rel);
                const o = r0 - j.lo;
                rows(j.head, u, n, j.out_w[o * rank_dim ..][0 .. n * rank_dim], j.out_b[o..][0..n]);
            }
        }
    };
    if (out_w.len != (hi - lo) * rank_dim or out_b.len != hi - lo) return error.BadFoldInputs;
    var job: Job = .{ .io = io, .file = file, .base = base, .head = head, .lo = lo, .hi = hi, .out_w = out_w, .out_b = out_b };
    const threads = try gpa.alloc(std.Thread, @max(1, jobs));
    defer gpa.free(threads);
    const failed = try gpa.alloc(?anyerror, threads.len);
    defer gpa.free(failed);
    @memset(failed, null);
    var started: usize = 0;
    defer for (threads[0..started]) |t| t.join();
    for (threads, failed) |*t, *e| {
        t.* = try std.Thread.spawn(.{}, Job.work, .{ &job, gpa, e });
        started += 1;
    }
    for (threads[0..started]) |t| t.join();
    started = 0;
    for (failed) |f| if (f) |e| return e;
    return job.zero.load(.acquire);
}

/// The safetensors header of a folds file over `vocab` rows (the bytes torch's save_file writes for
/// {"fold_w", "fold_b"}: names sorted, no spaces, padded with spaces to 8 bytes); `meta` adds __metadata__ pairs.
pub fn header(gpa: std.mem.Allocator, vocab: u64, meta: []const [2][]const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.append(gpa, '{');
    var buf: [256]u8 = undefined;
    if (meta.len > 0) {
        try out.appendSlice(gpa, "\"__metadata__\":{");
        for (meta, 0..) |m, i| {
            if (i > 0) try out.append(gpa, ',');
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "\"{s}\":\"{s}\"", .{ m[0], m[1] }));
        }
        try out.appendSlice(gpa, "},");
    }
    const nb = vocab * 2;
    const nw = vocab * rank_dim * 2;
    try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "\"fold_b\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},", .{ vocab, nb }));
    try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "\"fold_w\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ vocab, rank_dim, nb, nb + nw }));
    while ((8 + out.items.len) % 8 != 0) try out.append(gpa, ' ');
    const full = try gpa.alloc(u8, 8 + out.items.len);
    std.mem.writeInt(u64, full[0..8], out.items.len, .little);
    @memcpy(full[8..], out.items);
    out.deinit(gpa);
    return full;
}

/// `parts`, in order, become the file `out_path`, all or nothing; the one place the folds file is published. A new
/// temporary file beside `out_path` (`.NAME.<16 hex>.tmp`, a fresh random name each try) is created exclusively
/// (O_CREAT|O_EXCL: an existing name, a symlink included, is refused and never opened, so no link is followed and
/// nothing is truncated), written and synced through that one handle, then renamed over `out_path` (a symlink at
/// `out_path` is replaced, not followed). Any failure removes the temporary file and leaves `out_path` as it was.
pub fn publish(gpa: std.mem.Allocator, io: std.Io, out_path: []const u8, parts: []const []const u8) !void {
    return publishNamed(gpa, io, out_path, parts, &randomNonce);
}

/// The folds file's digest record (fold-rank.sh's `.inkling2-dspark-fold.sha256`): `line` and a newline published as
/// `path` the same way as the folds file (`publish`), so no existing name is opened, followed or truncated.
pub fn record(gpa: std.mem.Allocator, io: std.Io, path: []const u8, line: []const u8) !void {
    if (line.len == 0 or std.mem.indexOfScalar(u8, line, '\n') != null) return error.BadRecord;
    return publish(gpa, io, path, &.{ line, "\n" });
}

fn randomNonce(io: std.Io, _: usize) [8]u8 {
    var n: [8]u8 = undefined;
    io.random(&n);
    return n;
}

/// `publish` with the temporary names' nonces from `nonce` (try 0, 1, ...): tests plant a name it will draw.
fn publishNamed(gpa: std.mem.Allocator, io: std.Io, out_path: []const u8, parts: []const []const u8, nonce: *const fn (std.Io, usize) [8]u8) !void {
    const cwd = std.Io.Dir.cwd();
    const dir = std.fs.path.dirname(out_path) orelse ".";
    const base = std.fs.path.basename(out_path);
    if (base.len == 0) return error.BadOutputPath;
    var try_: usize = 0;
    while (true) : (try_ += 1) {
        const tmp = try std.fmt.allocPrint(gpa, "{s}/.{s}.{s}.tmp", .{ dir, base, std.fmt.bytesToHex(nonce(io, try_), .lower) });
        defer gpa.free(tmp);
        const file = cwd.createFile(io, tmp, .{ .exclusive = true }) catch |e| switch (e) {
            error.PathAlreadyExists => if (try_ < 16) continue else return e,
            else => return e,
        };
        errdefer cwd.deleteFile(io, tmp) catch {};
        {
            defer file.close(io);
            var at: u64 = 0;
            for (parts) |part| {
                try file.writePositionalAll(io, part, at);
                at += part.len;
            }
            try file.sync(io);
        }
        try cwd.rename(tmp, cwd, out_path, io);
        return;
    }
}

const testing = std.testing;

test "inkling2 fold: fp64 to bf16 rounds through fp32 (torch's order), nearest even at each step" {
    // 1 + 2^-8 + 2^-30: direct rounding to bf16 goes up (above the halfway point 1 + 2^-8), but fp32 first rounds
    // it to 1 + 2^-8 exactly, the halfway point, and nearest even then goes down to 1.0
    try testing.expectEqual(@as(u16, 0x3f80), f64ToBf16(1.0 + 0x1p-8 + 0x1p-30));
    try testing.expectEqual(@as(u16, 0x3f81), f64ToBf16(1.0 + 0x1p-8 + 0x1p-20));
    try testing.expectEqual(@as(u16, 0xbf80), f64ToBf16(-1.0));
    try testing.expectEqual(@as(u16, 0x0000), f64ToBf16(0.0));
}

test "inkling2 fold: tiles equal the plain fp64 sum in h order for every row and column, odd row counts" {
    const gpa = testing.allocator;
    const H = 37;
    const n = 7;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const bf = struct {
        fn of(r: std.Random) f64 {
            return st.bf16ToF32(st.f32ToBf16((r.float(f32) - 0.5) * 8));
        }
    }.of;
    const w = try gpa.alloc(f64, H * rank_dim);
    defer gpa.free(w);
    const b = try gpa.alloc(f64, H);
    defer gpa.free(b);
    const u = try gpa.alloc(f64, n * H);
    defer gpa.free(u);
    for (w) |*x| x.* = bf(rnd);
    for (b) |*x| x.* = bf(rnd);
    for (u) |*x| x.* = bf(rnd);
    var head = try Head.init(gpa, H, w, b);
    defer head.deinit(gpa);
    var ow: [n * rank_dim]u16 = undefined;
    var ob: [n]u16 = undefined;
    rows(&head, u, n, &ow, &ob);
    for (0..n) |r| {
        for (0..rank_dim + 1) |c| {
            var s: f64 = 0;
            for (0..H) |h| s += u[r * H + h] * (if (c < rank_dim) w[h * rank_dim + c] else b[h]);
            const got = if (c < rank_dim) ow[r * rank_dim + c] else ob[r];
            try testing.expectEqual(f64ToBf16(s), got);
        }
    }
}

test "inkling2 fold: the header is save_file's" {
    const h = try header(testing.allocator, 201024, &.{});
    defer testing.allocator.free(h);
    try testing.expectEqual(@as(u64, 152), std.mem.readInt(u64, h[0..8], .little));
    try testing.expectEqualStrings("{\"fold_b\":{\"dtype\":\"BF16\",\"shape\":[201024],\"data_offsets\":[0,402048]},\"fold_w\":{\"dtype\":\"BF16\",\"shape\":[201024,256],\"data_offsets\":[402048,103326336]}} ", h[8..]);
}

const PublishTest = struct {
    /// Try 0 draws the planted name, later tries ordinary ones.
    fn planted(io: std.Io, try_: usize) [8]u8 {
        return if (try_ == 0) @splat(0xab) else randomNonce(io, try_);
    }

    fn root(tmp: *const testing.TmpDir) ![:0]u8 {
        return tmp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    }

    fn join(dir: []const u8, name: []const u8) ![]u8 {
        return std.fs.path.join(testing.allocator, &.{ dir, name });
    }

    fn expectFile(tmp: *const testing.TmpDir, name: []const u8, want: []const u8) !void {
        const got = try tmp.dir.readFileAlloc(testing.io, name, testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(want, got);
    }

    /// The folder's names other than `keep`: none may be a leftover temporary file.
    fn expectOnly(tmp: *const testing.TmpDir, keep: []const []const u8) !void {
        var d = try tmp.dir.openDir(testing.io, ".", .{ .iterate = true });
        defer d.close(testing.io);
        var it = d.iterate();
        while (try it.next(testing.io)) |e| {
            for (keep) |k| {
                if (std.mem.eql(u8, k, e.name)) break;
            } else {
                std.debug.print("unexpected entry {s}\n", .{e.name});
                return error.TestUnexpectedEntry;
            }
        }
    }

    const Racer = struct {
        out: []const u8,
        body: []const u8,
        err: ?anyerror = null,
        fn run(r: *Racer) void {
            for (0..40) |_| publish(testing.allocator, testing.io, r.out, &.{ r.body[0..3], r.body[3..] }) catch |e| {
                r.err = e;
                return;
            };
        }
    };
};

test "inkling2 fold publish: a symlink planted at the temporary names or at the output is never written through" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    const dir = try PublishTest.root(&tmp);
    defer testing.allocator.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "sentinel", .data = "an original weight shard" });
    const name = "inkling2-dspark-fold.safetensors";
    // the earlier wrapper's and tool's predictable names, and the name try 0 will draw, all link to the sentinel
    try tmp.dir.symLink(io, "sentinel", name ++ ".tmp", .{});
    try tmp.dir.symLink(io, "sentinel", name ++ ".tmp.tmp", .{});
    const drawn = "." ++ name ++ ".abababababababab.tmp";
    try tmp.dir.symLink(io, "sentinel", drawn, .{});
    const out = try PublishTest.join(dir, name);
    defer testing.allocator.free(out);
    try publishNamed(testing.allocator, io, out, &.{ "fold", "s v1" }, &PublishTest.planted);
    try PublishTest.expectFile(&tmp, "sentinel", "an original weight shard");
    try PublishTest.expectFile(&tmp, name, "folds v1");
    try PublishTest.expectOnly(&tmp, &.{ "sentinel", name, name ++ ".tmp", name ++ ".tmp.tmp", drawn });
    // the output itself a link to the sentinel: replaced by the new file, the sentinel unchanged
    try tmp.dir.deleteFile(io, name);
    try tmp.dir.symLink(io, "sentinel", name, .{});
    try publish(testing.allocator, io, out, &.{"folds v2"});
    try PublishTest.expectFile(&tmp, "sentinel", "an original weight shard");
    try PublishTest.expectFile(&tmp, name, "folds v2"); // written through the link, the sentinel would read so
}

test "inkling2 fold publish: two writers at once leave one whole file and no temporary file" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try PublishTest.root(&tmp);
    defer testing.allocator.free(dir);
    const out = try PublishTest.join(dir, "f.safetensors");
    defer testing.allocator.free(out);
    const a_body: [4096]u8 = @splat('a');
    const b_body: [4096]u8 = @splat('b');
    var a: PublishTest.Racer = .{ .out = out, .body = &a_body };
    var b: PublishTest.Racer = .{ .out = out, .body = &b_body };
    const ta = try std.Thread.spawn(.{}, PublishTest.Racer.run, .{&a});
    const tb = try std.Thread.spawn(.{}, PublishTest.Racer.run, .{&b});
    ta.join();
    tb.join();
    try testing.expect(a.err == null and b.err == null);
    const got = try tmp.dir.readFileAlloc(testing.io, "f.safetensors", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(got);
    try testing.expect(std.mem.eql(u8, got, &a_body) or std.mem.eql(u8, got, &b_body));
    try PublishTest.expectOnly(&tmp, &.{"f.safetensors"});
}

test "inkling2 fold record: a symlink planted at the record, its old temporary name or the drawn one is never written through" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = testing.io;
    const dir = try PublishTest.root(&tmp);
    defer testing.allocator.free(dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "sentinel", .data = "an original weight shard" });
    const name = ".inkling2-dspark-fold.sha256";
    const drawn = "." ++ name ++ ".abababababababab.tmp";
    for ([_][]const u8{ name, name ++ ".tmp", name ++ ".AbCd1234", drawn }) |n| try tmp.dir.symLink(io, "sentinel", n, .{});
    const out = try PublishTest.join(dir, name);
    defer testing.allocator.free(out);
    const line = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  rank 3";
    try publishNamed(testing.allocator, io, out, &.{ line, "\n" }, &PublishTest.planted);
    try PublishTest.expectFile(&tmp, "sentinel", "an original weight shard");
    try PublishTest.expectFile(&tmp, name, line ++ "\n");
    try PublishTest.expectOnly(&tmp, &.{ "sentinel", name, name ++ ".tmp", name ++ ".AbCd1234", drawn });
    try record(testing.allocator, io, out, "again  rank 3");
    try PublishTest.expectFile(&tmp, name, "again  rank 3\n");
    try PublishTest.expectFile(&tmp, "sentinel", "an original weight shard");
    try testing.expectError(error.BadRecord, record(testing.allocator, io, out, "two\nlines"));
    try testing.expectError(error.BadRecord, record(testing.allocator, io, out, ""));
    try PublishTest.expectFile(&tmp, name, "again  rank 3\n");
}

test "inkling2 fold publish: a missing folder fails and creates nothing" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try PublishTest.root(&tmp);
    defer testing.allocator.free(dir);
    const out = try PublishTest.join(dir, "absent/f.safetensors");
    defer testing.allocator.free(out);
    try testing.expectError(error.FileNotFound, publish(testing.allocator, testing.io, out, &.{"x"}));
    try PublishTest.expectOnly(&tmp, &.{});
}
