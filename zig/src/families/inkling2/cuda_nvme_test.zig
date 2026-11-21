//! Host tests (no GPU) of the prompt cache's NVMe tier (cuda_nvme.zig): its settings, the file format and names, the
//! payload's pieces, rank 0's index, and files on a real disk (written, read back, refused when damaged or malformed,
//! which files the tier may remove, and what it never follows, blocks on or replaces).
const std = @import("std");
const builtin = @import("builtin");
const nvme = @import("cuda_nvme.zig");
const pages = @import("cuda_pages.zig");

const t = std.testing;

extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;

/// A FIFO named `name` in the tier directory of `id` under `root_path`.
fn fifoIn(root_path: []const u8, id: [32]u8, name: []const u8) !void {
    var db: [32]u8 = undefined;
    var pb: [512]u8 = undefined;
    const p = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}/{s}", .{ root_path, nvme.dirName(&db, id), name }, 0);
    if (mkfifo(p, 0o600) != 0) return error.TestFifo;
}

test "inkling2 nvme: the switch is off unless it names an absolute directory; the cap and floor in GiB" {
    for ([_]?[]const u8{ null, "", "0", "off" }) |v| try t.expect(!(try nvme.configFrom(v, null, null)).on());
    try t.expectError(error.NvmeNeedsAbsoluteDir, nvme.configFrom("1", null, null));
    try t.expectError(error.NvmeNeedsAbsoluteDir, nvme.configFrom("on", null, null));
    try t.expectError(error.NvmeNeedsAbsoluteDir, nvme.configFrom("cache/nvme", null, null));
    const c = try nvme.configFrom("/data/ink2", null, null);
    try t.expect(c.on());
    try t.expectEqual(@as(u64, 64 << 30), c.cap);
    try t.expectEqual(@as(u64, 32 << 30), c.floor);
    const d = try nvme.configFrom("/data/ink2", "0.5", "100");
    try t.expectEqual(@as(u64, 512 << 20), d.cap);
    try t.expectEqual(@as(u64, 100 << 30), d.floor);
    try t.expect(!(try nvme.configFrom("/data/ink2", "0", null)).on()); // a cap of 0 keeps nothing
    try t.expectError(error.BadNvmeSetting, nvme.configFrom("/data/ink2", "-1", null));
    try t.expectError(error.BadNvmeSetting, nvme.configFrom("/data/ink2", "lots", null));
    try t.expectError(error.BadNvmeSetting, nvme.configFrom("/data/ink2", null, "nan"));
    var b: [96]u8 = undefined;
    try t.expectEqualStrings("", nvme.settingsText(.{}, &b));
    try t.expectEqualStrings(" TF_INK2_NVME=on cap 68719476736 format 1", nvme.settingsText(c, &b));
}

test "inkling2 nvme: a setting a rank refuses hashes as refused, so it agrees with no valid setting, on or off" {
    var a: [96]u8 = undefined;
    var b: [96]u8 = undefined;
    const valid_on = nvme.settingsTextOf(nvme.configFrom("/data/ink2", null, null), &a);
    for ([_]anyerror!nvme.Config{ nvme.configFrom("relative/dir", null, null), nvme.configFrom("/data/ink2", "lots", null), nvme.configFrom("/data/ink2", "-1", null) }) |bad| {
        const txt = nvme.settingsTextOf(bad, &b);
        try t.expect(std.mem.startsWith(u8, txt, " TF_INK2_NVME=refused "));
        try t.expect(!std.mem.eql(u8, txt, valid_on));
        try t.expect(txt.len > 0); // off hashes as nothing
    }
    try t.expectEqualStrings("", nvme.settingsTextOf(nvme.configFrom(null, null, null), &b));
    // two refusals for different reasons differ too
    var c: [96]u8 = undefined;
    try t.expect(!std.mem.eql(u8, nvme.settingsTextOf(nvme.configFrom("x", null, null), &b), nvme.settingsTextOf(nvme.configFrom("/d", "nan", null), &c)));
}

const identity: [32]u8 = @splat(0x5a);

fn headFor(at: u32, rank: u32, regions: u32, run: u64, fixed: u64, tokens: []const u32) nvme.Head {
    const npages: u32 = @intCast(at / pages.page_rows);
    var h: nvme.Head = .{ .world = 4, .rank = rank, .at = at, .npages = npages, .regions = regions, .run = run, .fixed = fixed, .payload = fixed + @as(u64, regions) * npages * run, .head_len = nvme.headLen(at, 0), .identity = identity, .name = nvme.nameOf(identity, at, tokens), .content = 0x77 };
    h.meta_hash = nvme.metaHash(tokens);
    h.head_hash = h.hash();
    return h;
}

fn wantFor(rank: u32, regions: u32, run: u64, fixed: ?u64) nvme.Want {
    return .{ .identity = identity, .content = 0x77, .world = 4, .rank = rank, .regions = regions, .run = run, .max_at = 131072, .fixed = fixed, .max_fixed = 64 << 20 };
}

test "inkling2 nvme: the head is 256 bytes, sections are 4 KiB aligned, and every field it states is checked" {
    try t.expectEqual(@as(usize, 256), @sizeOf(nvme.Head));
    try t.expectEqual(@as(u64, 4096), nvme.headLen(1, 0));
    try t.expectEqual(@as(u64, 4096), nvme.headLen(960, 0)); // 256 + 3,840
    try t.expectEqual(@as(u64, 8192), nvme.headLen(961, 0));
    try t.expectEqual(@as(u64, 4096 + 131072 * 4), nvme.headLen(131072, 0));
    const tokens: [3000]u32 = @splat(7);
    const h = headFor(3000, 1, 18, 128 << 10, 1000, &tokens);
    const len = h.head_len + nvme.alignUp(h.payload);
    try nvme.checkHead(h, wantFor(1, 18, 128 << 10, 1000), len);
    try nvme.checkHead(h, wantFor(1, 18, 128 << 10, null), null);
    try nvme.checkTokens(h, &tokens);
    // each field against what this rank expects
    try t.expectError(error.NvmeOtherRank, nvme.checkHead(h, wantFor(2, 18, 128 << 10, 1000), len));
    try t.expectError(error.NvmeLayout, nvme.checkHead(h, wantFor(1, 16, 128 << 10, 1000), len)); // no drafter
    try t.expectError(error.NvmeLayout, nvme.checkHead(h, wantFor(1, 18, 64 << 10, 1000), len)); // another world's run
    try t.expectError(error.NvmeLayout, nvme.checkHead(h, wantFor(1, 18, 128 << 10, 999), len));
    try t.expectError(error.NvmeSize, nvme.checkHead(h, wantFor(1, 18, 128 << 10, 1000), len - 4096)); // cut short
    var w = wantFor(1, 18, 128 << 10, 1000);
    w.identity[0] ^= 1;
    try t.expectError(error.NvmeOtherIdentity, nvme.checkHead(h, w, len));
    // another drafter's bytes (a changed mtp.safetensors or folds on this rank)
    var wc = wantFor(1, 18, 128 << 10, 1000);
    wc.content ^= 1;
    try t.expectError(error.NvmeOtherContent, nvme.checkHead(h, wc, len));
    // the load's position
    var wa = wantFor(1, 18, 128 << 10, 1000);
    wa.at = 3000;
    try nvme.checkHead(h, wa, len);
    wa.at = 2999;
    try t.expectError(error.NvmeLayout, nvme.checkHead(h, wa, len));
    // a changed field without its hash, and a field changed with its hash redone but not agreeing with the rest
    var x = h;
    x.at += 256; // one page more: the head no longer agrees with itself
    try t.expectError(error.NvmeHead, nvme.checkHead(x, wantFor(1, 18, 128 << 10, 1000), len));
    x.head_hash = x.hash();
    try t.expectError(error.NvmeLayout, nvme.checkHead(x, wantFor(1, 18, 128 << 10, 1000), len)); // npages no longer at / 256
    var m = h;
    m.magic[0] = 'X';
    try t.expectError(error.NvmeNotOurs, nvme.checkHead(m, wantFor(1, 18, 128 << 10, 1000), len));
    var v = h;
    v.version = 2;
    try t.expectError(error.NvmeNotOurs, nvme.checkHead(v, wantFor(1, 18, 128 << 10, 1000), len));
    var im = h;
    im.images = 1; // an image section: refused in v0.3 (text states only)
    im.head_hash = im.hash();
    try t.expectError(error.NvmeImages, nvme.checkHead(im, wantFor(1, 18, 128 << 10, 1000), len));
    // the tokens: their hash and the name over them
    var other = tokens;
    other[1234] = 8;
    try t.expectError(error.NvmeMeta, nvme.checkTokens(h, &other));
    var re = h;
    re.meta_hash = nvme.metaHash(&other);
    try t.expectError(error.NvmeName, nvme.checkTokens(re, &other));
    try t.expectError(error.NvmeLayout, nvme.checkTokens(h, tokens[0..2999]));
}

/// `h` with its own hash redone (a malformed head a writer hashed correctly).
fn rehashed(h: nvme.Head) nvme.Head {
    var x = h;
    x.head_hash = x.hash();
    return x;
}

test "inkling2 nvme: a correctly hashed head with oversized or overflowing sizes is refused, never trapped" {
    const tokens: [3000]u32 = @splat(7);
    const h = headFor(3000, 1, 18, 128 << 10, 1000, &tokens);
    const w = wantFor(1, 18, 128 << 10, null);
    // a position past the context (it would size the tokens' allocation)
    var big = h;
    big.at = 131072 + 256;
    big.npages = @intCast(big.at / pages.page_rows);
    big.head_len = nvme.headLen(big.at, 0);
    big.payload = nvme.payloadBytes(big.fixed, big.regions, big.npages, big.run).?;
    try t.expectError(error.NvmeLayout, nvme.checkHead(rehashed(big), w, null));
    var huge = h;
    huge.at = std.math.maxInt(u32);
    huge.npages = @intCast(huge.at / pages.page_rows);
    try t.expectError(error.NvmeLayout, nvme.checkHead(rehashed(huge), w, null));
    // a fixed part past the arena slot (a scan), and one the load's position does not give
    var fx = h;
    fx.fixed = (64 << 20) + 1;
    fx.payload = nvme.payloadBytes(fx.fixed, fx.regions, fx.npages, fx.run).?;
    try t.expectError(error.NvmeLayout, nvme.checkHead(rehashed(fx), w, null));
    // regions x pages x run overflows; fixed + pages overflows; the padding overflows
    var r = h;
    r.run = std.math.maxInt(u64) / 2;
    var wr = w;
    wr.run = r.run;
    try t.expectError(error.NvmeLayout, nvme.checkHead(rehashed(r), wr, null));
    var f = h;
    f.fixed = std.math.maxInt(u64) - 10;
    var wf = w;
    wf.max_fixed = std.math.maxInt(u64);
    try t.expectError(error.NvmeLayout, nvme.checkHead(rehashed(f), wf, null));
    try t.expect(nvme.alignUpChecked(std.math.maxInt(u64) - 5) == null);
    try t.expect(nvme.payloadBytes(std.math.maxInt(u64), 1, 1, 1) == null);
    try t.expect(nvme.payloadBytes(0, std.math.maxInt(u32), std.math.maxInt(u32), std.math.maxInt(u64)) == null);
    // a payload that is not what the sizes give
    var p = h;
    p.payload += 1;
    try t.expectError(error.NvmeLayout, nvme.checkHead(rehashed(p), w, null));
}

test "inkling2 nvme: a state's name is its identity, position and every token" {
    var a = [_]u32{ 1, 2, 3, 4, 5 };
    const n = nvme.nameOf(identity, 5, &a);
    try t.expectEqualSlices(u8, &n, &nvme.nameOf(identity, 5, &a));
    var other_id = identity;
    other_id[31] ^= 1;
    try t.expect(!std.mem.eql(u8, &n, &nvme.nameOf(other_id, 5, &a)));
    try t.expect(!std.mem.eql(u8, &n, &nvme.nameOf(identity, 4, a[0..4])));
    a[4] = 6;
    try t.expect(!std.mem.eql(u8, &n, &nvme.nameOf(identity, 5, &a)));
    // the identity: program, checkpoint, settings, world and layout each move it
    const z: [32]u8 = @splat(0);
    const l: nvme.Layout = .{ .page_rows = 256, .regions = 18, .run = 128 << 10, .slot = 50 << 20 };
    const id0 = nvme.identityOf(z, z, z, 4, l);
    var one = z;
    one[0] = 1;
    for ([_][32]u8{ nvme.identityOf(one, z, z, 4, l), nvme.identityOf(z, one, z, 4, l), nvme.identityOf(z, z, one, 4, l), nvme.identityOf(z, z, z, 2, l), nvme.identityOf(z, z, z, 4, .{ .page_rows = 256, .regions = 16, .run = 128 << 10, .slot = 46 << 20 }) }) |o|
        try t.expect(!std.mem.eql(u8, &id0, &o));
}

test "inkling2 nvme: a payload's pieces cover it once in order, each id run of a region one piece, at its pool address" {
    const gpa = t.allocator;
    const pl: pages.Place = .{ .base = 1 << 40, .pages = 100, .run = 4096 };
    const ids = [_]u32{ 10, 11, 12, 40, 41, 7 }; // runs 10-12, 40-41, 7
    const p: nvme.Plan = .{ .fixed = 10_000, .regions = 3, .run = 4096, .ids = &ids, .pl = pl };
    try t.expectEqual(@as(u64, 10_000 + 3 * 6 * 4096), p.bytes());
    var out: std.ArrayList(nvme.Piece) = .empty;
    defer out.deinit(gpa);
    try p.pieces(gpa, 0, p.bytes(), &out);
    try t.expectEqual(@as(usize, 1 + 3 * 3), out.items.len);
    try t.expect(out.items[0].fixed and out.items[0].off == 0 and out.items[0].len == 10_000);
    // region 1's second run: pages 40, 41 at payload 10,000 + (6 + 3) * 4,096
    const r1 = out.items[1 + 3 + 1];
    try t.expectEqual(@as(u64, 10_000 + 9 * 4096), r1.off);
    try t.expectEqual(@as(u64, 2 * 4096), r1.len);
    try t.expectEqual(pl.region(1) + 40 * 4096, r1.src);
    // any chunking gives the same bytes from the same places
    for ([_]u64{ 1, 4096, 5000, 8 << 20, 10_001, 3 * 4096 + 17 }) |step| {
        var at: u64 = 0;
        var sum: u64 = 0;
        while (at < p.bytes()) {
            const hi = @min(p.bytes(), at + step);
            try p.pieces(gpa, at, hi, &out);
            var cur = at;
            for (out.items) |pc| {
                try t.expectEqual(cur, pc.off); // contiguous, in order
                // the same byte as the whole plan names
                var whole: std.ArrayList(nvme.Piece) = .empty;
                defer whole.deinit(gpa);
                try p.pieces(gpa, pc.off, pc.off + 1, &whole);
                try t.expectEqual(@as(usize, 1), whole.items.len);
                try t.expectEqual(whole.items[0].src, pc.src);
                try t.expectEqual(whole.items[0].fixed, pc.fixed);
                cur += pc.len;
                sum += pc.len;
            }
            try t.expectEqual(hi, cur);
            at = hi;
        }
        try t.expectEqual(p.bytes(), sum);
    }
    try p.pieces(gpa, 5, 5, &out);
    try t.expectEqual(@as(usize, 0), out.items.len);
}

test "inkling2 nvme: rank 0's index picks the longest prefix shorter than the prompt, evicts the least recently used" {
    const gpa = t.allocator;
    var x: nvme.Index = .{};
    defer x.deinit(gpa);
    var prompt: [5000]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i * 7 + 1);
    const n1: [32]u8 = @splat(1);
    const n2: [32]u8 = @splat(2);
    const n3: [32]u8 = @splat(3);
    const n4: [32]u8 = @splat(4);
    try x.add(gpa, n1, try gpa.dupe(u32, prompt[0..2000]), 100);
    try x.add(gpa, n2, try gpa.dupe(u32, prompt[0..4000]), 200);
    var off = try gpa.dupe(u32, prompt[0..4500]);
    off[100] = 0; // not a prefix
    try x.add(gpa, n3, off, 300);
    // room reserved first, then an add that cannot fail (the commit's)
    try x.list.ensureUnusedCapacity(gpa, 1);
    x.addAssumeCapacity(n4, try gpa.dupe(u32, prompt[0..5000]), 400); // the whole prompt: no row left to run
    try t.expectEqual(@as(u64, 1000), x.bytes);
    try t.expectEqual(@as(?usize, 1), x.longest(&prompt, 0));
    try t.expectEqual(@as(?usize, 1), x.longest(&prompt, 4000));
    try t.expectEqual(@as(?usize, null), x.longest(&prompt, 4001));
    try t.expectEqual(@as(?usize, 0), x.longest(prompt[0..3999], 0));
    try t.expectEqual(@as(?usize, null), x.longest(prompt[0..2000], 0));
    try t.expectEqual(@as(?usize, 1), x.find(n2));
    // least recently used: n1 (added first), then after touching n1, n2
    try t.expectEqual(@as(?usize, 0), x.victim());
    x.touch(0);
    try t.expectEqual(@as(?usize, 1), x.victim());
    const gone = x.remove(gpa, 1);
    try t.expectEqualSlices(u8, &n2, &gone.name);
    try t.expectEqual(@as(u64, 800), x.bytes);
    try t.expectEqual(@as(?usize, null), x.find(n2));
    try t.expectEqual(@as(?usize, 0), x.longest(&prompt, 0)); // now the 2,000-token state
}

test "inkling2 nvme: names the tier makes, and only those, parse as its files and directories" {
    var db: [32]u8 = undefined;
    const d = nvme.dirName(&db, identity);
    try t.expectEqualStrings("ink2-5a5a5a5a5a5a5a5a", d);
    try t.expect(nvme.isTierDir(d));
    for ([_][]const u8{ "ink2-5a5a", "ink2-5A5A5A5A5A5A5A5A", "learned", "ink2-5a5a5a5a5a5a5a5g" }) |s| try t.expect(!nvme.isTierDir(s));
    var fb: [64]u8 = undefined;
    const name: [32]u8 = @splat(0xab);
    const f = nvme.fileName(&fb, nvme.stemOf(name), 3, false);
    try t.expectEqualStrings("abababababababababababababababab.r3", f);
    const got = nvme.parseFile(f).?;
    try t.expectEqual(@as(u32, 3), got.rank);
    try t.expect(!got.part);
    try t.expect(nvme.parseFile("abababababababababababababababab.r12.part").?.part);
    try t.expectEqual(@as(u32, 0), nvme.parseFile("abababababababababababababababab.r0").?.rank);
    try t.expectEqual(@as(u32, 100), nvme.parseFile("abababababababababababababababab.r100").?.rank);
    for ([_][]const u8{ "abababababababababababababababab", "abababababababababababababababab.r", "abababababababababababababababab.rx", "ABABABABABABABABABABABABABABABAB.r0", "abababababababababababababababab.r1.tmp", "abababababababababababababababab.r1234", "notes.txt", "abababababababababababababababab.r00", "abababababababababababababababab.r01", "abababababababababababababababab.r00.part", "abababababababababababababababab.r+1", "abababababababababababababababab.r-0" }) |s|
        try t.expect(nvme.parseFile(s) == null);
}

// ---------------------------------------------------------------- files on a real disk

/// A fresh test root without a symlink in its path (macOS: /tmp is one), and its descriptor.
fn tmpRoot(buf: []u8, tag: []const u8) ![:0]const u8 {
    const base = if (builtin.os.tag == .macos) "/private/tmp" else "/tmp";
    const p = try std.fmt.bufPrintSentinel(buf, "{s}/tf-ink2-nvme-{d}-{s}", .{ base, std.c.getpid(), tag }, 0);
    rmTree(p);
    _ = std.c.mkdir(p, 0o700);
    return p;
}

fn openDir(path: [:0]const u8) !c_int {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestOpen;
    return fd;
}

/// A tier directory `ink2-...` of `id` made under `root`; its descriptor (checked as the tier checks it).
fn tierDir(root: c_int, id: [32]u8) !c_int {
    var db: [32]u8 = undefined;
    const n = nvme.dirName(&db, id);
    _ = std.c.mkdirat(root, n, 0o700);
    return nvme.openTierDir(root, n) orelse error.TestOpen;
}

/// A file as the tier writes it (head, tokens, the payload's bytes, padding), at `name` in `dir`.
fn writeState(dir: c_int, name: [:0]const u8, h0: nvme.Head, tokens: []const u32, payload: []const u8) !void {
    const gpa = t.allocator;
    var h = h0;
    h.payload_hash = std.hash.Wyhash.hash(0, payload);
    h.head_hash = h.hash();
    const total = h.head_len + nvme.alignUp(payload.len);
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(4096), total);
    defer gpa.free(buf);
    @memset(buf, 0);
    @memcpy(buf[0..256], std.mem.asBytes(&h));
    @memcpy(buf[256..][0 .. tokens.len * 4], std.mem.sliceAsBytes(tokens));
    @memcpy(buf[h.head_len..][0..payload.len], payload);
    const fd = (try nvme.create(dir, name, false)).fd;
    defer _ = std.c.close(fd);
    try nvme.writeAll(fd, buf, 0);
    try nvme.syncData(fd);
}

fn writeFile(dir: c_int, name: [:0]const u8, bytes: []const u8) !void {
    const fd = std.c.openat(dir, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestWrite;
    defer _ = std.c.close(fd);
    _ = std.c.write(fd, bytes.ptr, bytes.len);
}

fn readFile(dir: c_int, name: [:0]const u8, out: []u8) ![]u8 {
    const fd = std.c.openat(dir, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestOpen;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, out.ptr, out.len);
    if (n < 0) return error.TestRead;
    return out[0..@intCast(n)];
}

fn exists(dir: c_int, name: [:0]const u8) bool {
    var st: [1]u8 = undefined;
    _ = &st;
    const fd = std.c.openat(dir, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd >= 0) {
        _ = std.c.close(fd);
        return true;
    }
    return std.c.errno(fd) != .NOENT; // a symlink (ELOOP) is there too
}

fn rmTree(root: [:0]const u8) void {
    const d = std.c.opendir(root) orelse {
        _ = std.c.unlink(root);
        return;
    };
    while (std.c.readdir(d)) |ent| {
        const base = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, base, ".") or std.mem.eql(u8, base, "..")) continue;
        var b: [1200]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&b, "{s}/{s}", .{ root, base }, 0) catch continue;
        if (std.c.unlink(p) != 0) rmTree(p);
    }
    _ = std.c.closedir(d);
    _ = std.c.rmdir(root);
}

fn nameIn(buf: *[64]u8, name: [32]u8, rank: u32, part: bool) [:0]const u8 {
    return nvme.fileName(buf, nvme.stemOf(name), rank, part);
}

test "inkling2 nvme: a state file reads back whole, and a damaged one is refused" {
    const gpa = t.allocator;
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "rw");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const tokens = try gpa.alloc(u32, 1500);
    defer gpa.free(tokens);
    for (tokens, 0..) |*v, i| v.* = @intCast(i ^ 0x55);
    const run: u64 = 4096;
    const h = headFor(1500, 1, 4, run, 9000, tokens);
    const payload = try gpa.alloc(u8, h.payload);
    defer gpa.free(payload);
    for (payload, 0..) |*b, i| b.* = @truncate(i *% 31);
    var fb: [64]u8 = undefined;
    const name = nameIn(&fb, h.name, 1, false);
    try writeState(dir, name, h, tokens, payload);
    const scratch = try gpa.alignedAlloc(u8, .fromByteUnits(4096), 1 << 20);
    defer gpa.free(scratch);
    const got = try nvme.readHead(gpa, dir, name, wantFor(1, 4, run, 9000), scratch);
    defer gpa.free(got.tokens);
    try t.expectEqualSlices(u32, tokens, got.tokens);
    try t.expectEqual(h.head_len + nvme.alignUp(h.payload), got.bytes);
    // the payload back through aligned reads, its hash as written
    const fd = try nvme.openRead(dir, name);
    const back = try gpa.alignedAlloc(u8, .fromByteUnits(4096), nvme.alignUp(h.payload));
    defer gpa.free(back);
    try nvme.readAll(fd, back, h.head_len);
    _ = std.c.close(fd);
    try t.expectEqualSlices(u8, payload, back[0..payload.len]);
    try t.expectEqual(got.head.payload_hash, std.hash.Wyhash.hash(0, back[0..payload.len]));
    // as another rank's, or another layout's, the same file is refused
    try t.expectError(error.NvmeOtherRank, nvme.readHead(gpa, dir, name, wantFor(2, 4, run, 9000), scratch));
    try t.expectError(error.NvmeLayout, nvme.readHead(gpa, dir, name, wantFor(1, 4, run, 9001), scratch));
    // a token changed on disk: its hash refuses it
    var bad = try gpa.dupe(u32, tokens);
    defer gpa.free(bad);
    bad[700] +%= 1;
    var h2 = h;
    h2.meta_hash = nvme.metaHash(tokens); // the head still names the old tokens
    var fb2: [64]u8 = undefined;
    const name2 = nameIn(&fb2, h.name, 2, false);
    try writeState(dir, name2, withRank(h2, 2), bad, payload);
    try t.expectError(error.NvmeMeta, nvme.readHead(gpa, dir, name2, wantFor(2, 4, run, 9000), scratch));
}

test "inkling2 nvme: a head whose position passes the context is refused before anything is allocated from it" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "big");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    // a sparse file of the size its (hashed) head claims: 4 G tokens
    const tokens: [256]u32 = @splat(1);
    var h = headFor(256, 0, 1, 4096, 100, &tokens);
    h.at = std.math.maxInt(u32) - 255;
    h.npages = @intCast(h.at / pages.page_rows);
    h.head_len = nvme.headLen(h.at, 0);
    h.payload = nvme.payloadBytes(h.fixed, h.regions, h.npages, h.run).?;
    h.head_hash = h.hash();
    var fb: [64]u8 = undefined;
    const name = nameIn(&fb, h.name, 0, false);
    {
        const fd = (try nvme.create(dir, name, false)).fd;
        defer _ = std.c.close(fd);
        var block: [4096]u8 align(4096) = @splat(0);
        @memcpy(block[0..256], std.mem.asBytes(&h));
        try nvme.writeAll(fd, &block, 0);
        _ = std.c.ftruncate(fd, @intCast(h.head_len + nvme.alignUp(h.payload)));
    }
    // an allocator that fails every request: the refusal comes first
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const scratch = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 1 << 16);
    defer t.allocator.free(scratch);
    var w = wantFor(0, 1, 4096, null);
    w.max_fixed = 1 << 20;
    try t.expectError(error.NvmeLayout, nvme.readHead(failing.allocator(), dir, name, w, scratch));
    try t.expectEqual(@as(usize, 0), failing.allocations);
}

fn withRank(h: nvme.Head, rank: u32) nvme.Head {
    var x = h;
    x.rank = rank;
    return x;
}

test "inkling2 nvme: the scan keeps good files, removes bad ones of its own, and leaves strangers alone" {
    const gpa = t.allocator;
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "scan");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const tokens: [600]u32 = @splat(3);
    const h = headFor(600, 0, 2, 4096, 512, &tokens);
    const payload = try gpa.alloc(u8, h.payload); // the fixed part and 2 regions of 2 pages
    defer gpa.free(payload);
    @memset(payload, 9);
    var fb: [6][64]u8 = undefined;
    const good = nameIn(&fb[0], h.name, 0, false);
    try writeState(dir, good, h, &tokens, payload);
    // ours but damaged (the head hash): removed
    var other = tokens;
    other[0] = 4;
    const hb = headFor(600, 0, 2, 4096, 512, &other);
    const damaged = nameIn(&fb[1], hb.name, 0, false);
    try writeState(dir, damaged, hb, &other, payload);
    {
        // writeState redoes the head hash: break it on disk instead
        const fd = std.c.openat(dir, damaged, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
        var byte: [1]u8 = .{0xff};
        _ = std.c.pwrite(fd, &byte, 1, 100);
        _ = std.c.close(fd);
    }
    // a stranger's file with a name the tier would make: not ours (no magic), left alone
    const stranger: [:0]const u8 = "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd.r0";
    try writeFile(dir, stranger, "someone else's bytes, not a state");
    // another rank's file: not this rank's to scan
    const rank1 = nameIn(&fb[2], h.name, 1, false);
    try writeState(dir, rank1, withRank(h, 1), &tokens, payload);
    const five: [600]u32 = @splat(5);
    const six: [600]u32 = @splat(6);
    // a file of another drafter's bytes (its head's content): ours, refused, removed
    var hc = headFor(600, 0, 2, 4096, 512, &other);
    hc.name = nvme.nameOf(identity, 600, &five);
    hc.content = 0x78;
    const stale = nameIn(&fb[3], hc.name, 0, false);
    try writeState(dir, stale, hc, &five, payload);
    // a file named for rank 0 whose head says rank 1: not removed as rank 0's
    const mislabeled = nameIn(&fb[4], nvme.nameOf(identity, 600, &six), 0, false);
    try writeState(dir, mislabeled, withRank(headFor(600, 1, 2, 4096, 512, &six), 1), &six, payload);
    const scratch = try gpa.alignedAlloc(u8, .fromByteUnits(4096), 1 << 20);
    defer gpa.free(scratch);
    var found: std.ArrayList(nvme.Found) = .empty;
    defer {
        for (found.items) |f| gpa.free(f.tokens);
        found.deinit(gpa);
    }
    const dropped = try nvme.scan(gpa, dir, wantFor(0, 2, 4096, null), scratch, &found);
    try t.expectEqual(@as(usize, 1), found.items.len);
    try t.expectEqualSlices(u8, &h.name, &found.items[0].head.name);
    try t.expectEqual(@as(u32, 2), dropped); // the damaged one and the stale drafter's
    try t.expect(exists(dir, good) and !exists(dir, damaged) and exists(dir, stranger) and exists(dir, rank1) and !exists(dir, stale) and exists(dir, mislabeled));
}

test "inkling2 nvme: a scan whose list cannot grow frees what it read" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "oom");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const tokens: [600]u32 = @splat(3);
    const h = headFor(600, 0, 2, 4096, 512, &tokens);
    var payload: [512 + 2 * 2 * 4096]u8 = @splat(9);
    var fb: [64]u8 = undefined;
    try writeState(dir, nameIn(&fb, h.name, 0, false), h, &tokens, &payload);
    const scratch = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 1 << 20);
    defer t.allocator.free(scratch);
    // the tokens' allocation succeeds, the list's fails: the testing allocator reports a leak if they are kept
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 1 });
    var found: std.ArrayList(nvme.Found) = .empty;
    defer found.deinit(failing.allocator());
    try t.expectError(error.OutOfMemory, nvme.scan(failing.allocator(), dir, wantFor(0, 2, 4096, null), scratch, &found));
    try t.expectEqual(@as(usize, 0), found.items.len);
}

test "inkling2 nvme: the start sweep removes this rank's files of other identities and its own .part files, nothing else" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "sweep");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const own = try tierDir(root, identity);
    defer _ = std.c.close(own);
    var old_id = identity;
    old_id[0] = 0x11;
    const old = try tierDir(root, old_id);
    defer _ = std.c.close(old);
    const tokens: [300]u32 = @splat(1);
    const payload: [100]u8 = @splat(2);
    const name = nvme.nameOf(identity, 300, &tokens);
    var b: [8][64]u8 = undefined;
    const r0 = nameIn(&b[0], name, 0, false);
    const r1 = nameIn(&b[1], name, 1, false);
    const p0 = nameIn(&b[2], name, 0, true);
    const p1 = nameIn(&b[3], name, 1, true);
    const h = headFor(300, 0, 1, 4096, 100, &tokens);
    try writeState(old, r0, h, &tokens, &payload);
    try writeState(own, r0, h, &tokens, &payload);
    try writeState(own, p0, h, &tokens, &payload);
    try writeState(old, r1, withRank(h, 1), &tokens, &payload);
    try writeState(own, p1, withRank(h, 1), &tokens, &payload);
    try writeFile(old, "notes.txt", "not the tier's");
    try writeFile(old, "efefefefefefefefefefefefefefefef.r0", "a tier-like name without the magic");
    // a file outside any tier directory, beside them in the root
    try writeFile(root, "abababababababababababababababab.r0", "loose");
    var own_name: [32]u8 = undefined;
    const freed = nvme.sweep(root, nvme.dirName(&own_name, identity), 0);
    try t.expect(freed > 0);
    try t.expect(!exists(old, r0)); // rank 0's, another identity: gone
    try t.expect(!exists(own, p0)); // rank 0's unfinished write: gone
    try t.expect(exists(own, r0)); // rank 0's state of this identity: kept
    try t.expect(exists(old, r1) and exists(own, p1)); // rank 1's: not rank 0's to remove
    try t.expect(exists(old, "notes.txt") and exists(old, "efefefefefefefefefefefefefefefef.r0")); // not the tier's
    try t.expect(exists(root, "abababababababababababababababab.r0")); // not in a tier directory
    var old_name: [32]u8 = undefined;
    try t.expect(exists(root, nvme.dirName(&old_name, old_id))); // still holds files: kept
    // removeOurs refuses a file without the magic, whatever its name
    try t.expect(nvme.removeOurs(old, "efefefefefefefefefefefefefefefef.r0", 0) == .kept);
    try t.expect(exists(old, "efefefefefefefefefefefefefefefef.r0"));
    try t.expect(nvme.removeOurs(old, "efefefefefefefefefefefefefefefee.r0", 0) == .absent);
}

test "inkling2 nvme: a symlinked tier directory is never entered, a symlink in the root's path refuses it, a FIFO is never waited on" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "links");
    defer rmTree(root_path);
    var ob: [256]u8 = undefined;
    const outside_path = try tmpRoot(&ob, "outside");
    defer rmTree(outside_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const own = try tierDir(root, identity);
    defer _ = std.c.close(own);
    // another directory holding rank 0's state files (of this tier's shape), and a link to it named as a tier directory
    const outside = try openDir(outside_path);
    defer _ = std.c.close(outside);
    const tokens: [300]u32 = @splat(1);
    const payload: [100]u8 = @splat(2);
    const h = headFor(300, 0, 1, 4096, 100, &tokens);
    var b: [3][64]u8 = undefined;
    const r0 = nameIn(&b[0], h.name, 0, false);
    try writeState(outside, r0, h, &tokens, &payload);
    var other_id = identity;
    other_id[0] = 0x22;
    var lb: [32]u8 = undefined;
    const link_name = nvme.dirName(&lb, other_id);
    try t.expectEqual(@as(c_int, 0), std.c.symlinkat(outside_path, root, link_name));
    // a FIFO named as rank 0's .part in its own directory, and one named as a state
    const p0 = nameIn(&b[1], h.name, 0, true);
    try fifoIn(root_path, identity, p0);
    var other_name = h.name;
    other_name[0] ^= 1;
    const fifo_state = nameIn(&b[2], other_name, 0, false);
    try fifoIn(root_path, identity, fifo_state);
    var own_name: [32]u8 = undefined;
    _ = nvme.sweep(root, nvme.dirName(&own_name, identity), 0); // returns: no FIFO opened for reading blocks
    try t.expect(exists(outside, r0)); // not reached through the link
    try t.expect(exists(root, link_name)); // the link itself stays (not the tier's)
    try t.expect(exists(own, p0) and exists(own, fifo_state)); // not regular files: not the tier's
    try t.expect(nvme.openTierDir(root, link_name) == null);
    try t.expectError(error.NvmeNotOurs, nvme.openRead(own, fifo_state));
    var found: std.ArrayList(nvme.Found) = .empty;
    defer found.deinit(t.allocator);
    const scratch = try t.allocator.alignedAlloc(u8, .fromByteUnits(4096), 1 << 16);
    defer t.allocator.free(scratch);
    _ = try nvme.scan(t.allocator, own, wantFor(0, 1, 4096, null), scratch, &found);
    try t.expectEqual(@as(usize, 0), found.items.len);
    try t.expect(exists(own, fifo_state));
    // the root's path: a symlink anywhere in it refuses the tier; a real path opens, its missing parts made
    var pb: [512]u8 = undefined;
    const via_link = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}/deeper", .{ root_path, link_name }, 0);
    try t.expectError(error.NvmeDir, nvme.openRoot(via_link));
    var qb: [512]u8 = undefined;
    const fresh = try std.fmt.bufPrintSentinel(&qb, "{s}/made/by/the/tier", .{root_path}, 0);
    const fd = try nvme.openRoot(fresh);
    _ = std.c.close(fd);
    try t.expectError(error.NvmeDir, nvme.openRoot("/no/../dots"));
}

test "inkling2 nvme: a write never replaces a file: creation and publication refuse an existing one and leave it whole" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "collide");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const name: [32]u8 = @splat(0x3c);
    var b: [2][64]u8 = undefined;
    const part = nameIn(&b[0], name, 0, true);
    const dest = nameIn(&b[1], name, 0, false);
    // a stranger's file at the .part name: the create refuses (both tries exclusive), the file is as it was
    try writeFile(dir, part, "FOREIGN PART");
    try t.expectError(error.NvmeCreate, nvme.create(dir, part, false));
    var rd: [64]u8 = undefined;
    try t.expectEqualStrings("FOREIGN PART", try readFile(dir, part, &rd));
    try t.expect(nvme.removeOurs(dir, part, 0) == .kept); // not ours: never removed
    _ = std.c.unlinkat(dir, part, 0); // the test's own cleanup
    // our .part, a stranger's file at the final name: publication refuses, both stay
    const fd = (try nvme.create(dir, part, false)).fd;
    var block: [4096]u8 align(4096) = @splat(0);
    @memcpy(block[0..8], &nvme.magic);
    try nvme.writeAll(fd, &block, 0);
    _ = std.c.close(fd);
    try writeFile(dir, dest, "FOREIGN STATE");
    try t.expectError(error.NvmePublish, nvme.publish(dir, part, dest, false));
    try t.expectEqualStrings("FOREIGN STATE", try readFile(dir, dest, &rd));
    try t.expect(exists(dir, part));
    // with the name free, publication moves it there
    _ = std.c.unlinkat(dir, dest, 0);
    try nvme.publish(dir, part, dest, false);
    try t.expect(exists(dir, dest) and !exists(dir, part));
    try nvme.syncDir(dir);
}

test "inkling2 nvme: the payload hash (Wyhash) over chunks equals the hash of the whole" {
    const gpa = t.allocator;
    const n = (3 << 20) + 12345;
    const bytes = try gpa.alloc(u8, n);
    defer gpa.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
    const whole = std.hash.Wyhash.hash(0, bytes);
    for ([_]usize{ 1, 4096, 8 << 20, 1 << 20, 777777 }) |step| {
        var h: std.hash.Wyhash = .init(0);
        var at: usize = 0;
        while (at < n) {
            const e = @min(n, at + step);
            h.update(bytes[at..e]);
            at = e;
        }
        try t.expectEqual(whole, h.final());
    }
}

test "inkling2 nvme: free space reads for an open directory and fails closed for a bad descriptor" {
    const fd = try openDir("/");
    defer _ = std.c.close(fd);
    try t.expect(nvme.freeBytes(fd) != null);
    try t.expect(nvme.freeBytes(-1) == null);
}

test "inkling2 nvme: one engine a rank a root: its lock is one stable file the tier never removes or writes" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "lock");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    // a lock name there before, with contents of its own: taken, and left whole
    var lb: [24]u8 = undefined;
    const ln0 = nvme.lockName(&lb, 0);
    try writeFile(root, ln0, "FOREIGN LOCK CONTENTS");
    // engine A of rank 0; rank 1 on the same root (one disk for both ranks) has its own
    const a = try nvme.lockRoot(root, 0);
    try t.expectError(error.NvmeBusy, nvme.lockRoot(root, 0)); // a second engine of rank 0
    const r1 = try nvme.lockRoot(root, 1);
    // engine B opens the lock name before A's sweep and only then tries its flock (the interleaving): A's sweep of
    // other identities' directories (an old build's, with an old lock name inside) leaves the root's lock file, so B
    // opened the same inode A holds and its flock fails; once A lets go, B gets that same inode
    var old_id = identity;
    old_id[0] = 0x66;
    const old = try tierDir(root, old_id);
    defer _ = std.c.close(old);
    try writeFile(old, ".lock.r0", "OLD LOCK FILE");
    const b_open = std.c.openat(root, ln0, .{ .ACCMODE = .RDWR, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try t.expect(b_open >= 0);
    defer _ = std.c.close(b_open);
    var own_name: [32]u8 = undefined;
    _ = nvme.sweep(root, nvme.dirName(&own_name, identity), 0);
    try t.expect(flockNb(b_open) != 0); // A still holds this very inode
    const named = std.c.openat(root, ln0, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    defer _ = std.c.close(named);
    try t.expectEqual(inode(b_open), inode(named)); // the name still means the inode B opened
    _ = std.c.close(a);
    try t.expect(flockNb(b_open) == 0); // B now holds the one lock
    try t.expectError(error.NvmeBusy, nvme.lockRoot(root, 0));
    var rd: [64]u8 = undefined;
    try t.expectEqualStrings("FOREIGN LOCK CONTENTS", try readFile(root, ln0, &rd)); // never written or removed
    try t.expectEqualStrings("OLD LOCK FILE", try readFile(old, ".lock.r0", &rd)); // not a name the tier removes
    _ = std.c.close(r1);
}

extern "c" fn flock(fd: c_int, op: c_int) c_int;

fn flockNb(fd: c_int) c_int {
    return flock(fd, 2 | 4);
}

fn inode(fd: c_int) u64 {
    if (builtin.os.tag == .linux) {
        var sx: std.os.linux.Statx = undefined;
        if (std.c.statx(fd, "", 0x1000, .{ .INO = true }, &sx) != 0) return 0;
        return sx.ino;
    }
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return 0;
    return @intCast(st.ino);
}

test "inkling2 nvme: the admission refuses a state any rank can never make room for, before any eviction" {
    const stuck = nvme.stuck_need;
    // W = 4, a reclaimable rank and a stuck one, in both orders; an over-cap leader with a stuck follower
    try t.expectEqual(nvme.Admit.refuse, nvme.admit(&.{ 0, 5 << 20, stuck, 0 }));
    try t.expectEqual(nvme.Admit.refuse, nvme.admit(&.{ 0, stuck, 5 << 20, 0 }));
    try t.expectEqual(nvme.Admit.refuse, nvme.admit(&.{ 7 << 20, stuck }));
    try t.expectEqual(nvme.Admit.refuse, nvme.admit(&.{ stuck, 0, 0, 0 }));
    // only reclaimable: evict (one a turn); everyone fits: write
    try t.expectEqual(nvme.Admit.evict, nvme.admit(&.{ 0, 3, 9, 0 }));
    try t.expectEqual(nvme.Admit.write, nvme.admit(&.{ 0, 0, 0, 0 }));
    try t.expectEqual(nvme.Admit.write, nvme.admit(&.{0}));
    // a rank's answer: its files, those of states every rank holds (`live`), the cap, the state's bytes
    try t.expectEqual(@as(u64, 0), nvme.needOf(10, 10, 100, 90));
    try t.expectEqual(@as(u64, 20), nvme.needOf(30, 30, 100, 90)); // over by 20, all of it evictable
    try t.expectEqual(stuck, nvme.needOf(30, 5, 100, 90)); // 25 bytes no eviction frees + 90 > 100
    try t.expectEqual(@as(u64, 0), nvme.needOf(0, 0, 100, 100));
    try t.expectEqual(stuck, nvme.needOf(0, 0, 100, 101)); // larger than the cap alone
}

test "inkling2 nvme: a publication whose .part cannot be unlinked takes its link back: one name left, the .part" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "pubpart");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const name: [32]u8 = @splat(0x4d);
    var b: [2][64]u8 = undefined;
    const part = nameIn(&b[0], name, 0, true);
    const dest = nameIn(&b[1], name, 0, false);
    const fd = (try nvme.create(dir, part, false)).fd;
    defer _ = std.c.close(fd);
    try t.expectError(error.NvmePublishPart, nvme.publish(dir, part, dest, true));
    try t.expect(exists(dir, part) and !exists(dir, dest));
    try t.expectEqual(@as(u64, 1), nvme.info(fd).?.nlink);
}

test "inkling2 nvme: a write's file has no name until it is published (Linux): a stop right after its creation leaves nothing" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "tmpfile");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const name: [32]u8 = @splat(0x5e);
    var b: [2][64]u8 = undefined;
    const part = nameIn(&b[0], name, 0, true);
    const dest = nameIn(&b[1], name, 0, false);
    // created, and the process "stops" (the descriptor closed) before any header is written: no entry is left
    const m = try nvme.create(dir, part, true);
    try t.expect(m.unnamed);
    try t.expectEqual(@as(u64, 0), nvme.info(m.fd).?.nlink);
    _ = std.c.close(m.fd);
    try t.expect(!exists(dir, part) and !exists(dir, dest));
    try t.expectEqual(@as(u64, 0), try nvme.usedBytes(dir, 0));
    // created and published: one name, the state's, never over another file
    const m2 = try nvme.create(dir, part, true);
    defer _ = std.c.close(m2.fd);
    var block: [4096]u8 align(4096) = @splat(0);
    @memcpy(block[0..8], &nvme.magic);
    try nvme.writeAll(m2.fd, &block, 0);
    try nvme.publishFd(dir, m2.fd, dest);
    try t.expect(exists(dir, dest) and !exists(dir, part));
    try t.expectEqual(@as(u64, 1), nvme.info(m2.fd).?.nlink);
    const m3 = try nvme.create(dir, part, true);
    defer _ = std.c.close(m3.fd);
    try t.expectError(error.NvmePublish, nvme.publishFd(dir, m3.fd, dest)); // the name is taken: no replacement
}

test "inkling2 nvme: the start's count fails rather than undercount (a file of this rank's name it cannot read)" {
    if (std.c.geteuid() == 0) return error.SkipZigTest; // root reads anything
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "count");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const dir = try tierDir(root, identity);
    defer _ = std.c.close(dir);
    const tokens: [300]u32 = @splat(1);
    const payload: [100]u8 = @splat(2);
    const h = headFor(300, 0, 1, 4096, 100, &tokens);
    var fb: [64]u8 = undefined;
    const name = nameIn(&fb, h.name, 0, false);
    try writeState(dir, name, h, &tokens, &payload);
    try t.expect((try nvme.usedBytes(dir, 0)) > 0);
    try t.expectEqual(@as(c_int, 0), std.c.fchmodat(dir, name, 0, 0));
    defer _ = std.c.fchmodat(dir, name, 0o600, 0);
    try t.expectError(error.NvmeCount, nvme.usedBytes(dir, 0));
}

test "inkling2 nvme: a rank's bytes are its tier files in a directory, a .part and a file it could not remove included" {
    var rb: [256]u8 = undefined;
    const root_path = try tmpRoot(&rb, "used");
    defer rmTree(root_path);
    const root = try openDir(root_path);
    defer _ = std.c.close(root);
    const own = try tierDir(root, identity);
    defer _ = std.c.close(own);
    var old_id = identity;
    old_id[0] = 0x55;
    const old = try tierDir(root, old_id);
    defer _ = std.c.close(old);
    try t.expectEqual(@as(u64, 0), try nvme.usedBytes(own, 0));
    const tokens: [300]u32 = @splat(1);
    const payload: [100]u8 = @splat(2);
    const h = headFor(300, 0, 1, 4096, 100, &tokens);
    const len = h.head_len + nvme.alignUp(payload.len);
    var b: [3][64]u8 = undefined;
    try writeState(own, nameIn(&b[0], h.name, 0, false), h, &tokens, &payload);
    try writeState(own, nameIn(&b[1], h.name, 0, true), h, &tokens, &payload); // a stopped write's .part
    try writeState(old, nameIn(&b[2], h.name, 0, false), h, &tokens, &payload); // left under another identity
    try writeState(own, nameIn(&b[0], h.name, 1, false), withRank(h, 1), &tokens, &payload); // rank 1's
    try writeFile(own, "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd.r0", "not the tier's"); // a stranger's: not counted
    try t.expectEqual(2 * len, try nvme.usedBytes(own, 0)); // its state and a stopped write's .part
    try t.expectEqual(len, try nvme.usedBytes(old, 0)); // another identity's: not this directory's cap
    try t.expectEqual(len, try nvme.usedBytes(own, 1));
    try t.expect(nvme.removeOurs(old, nameIn(&b[2], h.name, 0, false), 0) == .removed);
    try t.expectEqual(@as(u64, 0), try nvme.usedBytes(old, 0));
    try t.expectEqual(2 * len, try nvme.usedBytes(own, 0)); // read again: the held descriptor's offset rewound
}
