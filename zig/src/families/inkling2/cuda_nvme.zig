//! Inkling-2's prompt cache on each rank's NVMe (v0.3): a cold tier under the device prompt cache (cuda_snap.zig) that
//! survives restarts and evictions. A kept state the device cache holds is written behind, while no stream runs, as one
//! file a rank: its fixed part (the arena slot's bytes, in the copies' order) and its whole pages' rows of every pool
//! region (the global layers' and the drafter's K and V of positions [0, N - N % 256)). A prompt whose longest prefix is
//! on disk (and not in the device cache) opens on fresh pages, and every rank reads its own file into them and into
//! the slot's fixed part, so the slot stands at N as a resume from the device cache would: the same bytes, so the same
//! bits (cuda_snap.zig's argument; the resume gates compare it).
//!
//! Rank 0 decides everything (what to write, when, what to evict, what to read) and sends it as steps (cuda_follow.zig
//! kinds nvme_*); every rank does the same file work on its own disk, and an all-or-none vote ends each begin, commit,
//! load and start: a short write, a full disk or a bad file on one rank makes every rank give the state up (a load
//! falls back to a cold pass on the same pages), never a stop. Rank 0 prepares everything a step needs before its
//! header goes out (a failure there is its alone: no step); the other ranks take part in every collective of a step
//! whatever their own preparation gave and report it in the vote; a vote gathers into a buffer made at start-up, so
//! no collective waits on an allocation (a rank that cannot make that buffer leaves the tier off on every rank, agreed
//! through the engine's own header buffer; the engine serves on). Followers keep no index. All file work runs in the lane host's idle turns
//! (write-behind) or at a stream's open (a read); a device-cache drop of a state being written only marks the write
//! given up, and the next idle turn removes it on every rank.
//!
//! The switch (every rank; agreed in the settings fingerprint when on, so a one-rank mismatch refuses on every rank,
//! a setting a rank refuses included):
//!   TF_INK2_NVME=<absolute dir>  on, files under <dir>/ink2-<identity>/  (unset, empty, 0 or off: off). The path is
//!                                walked without following a symlink (one in it refuses the tier); missing directories
//!                                are made (0700).
//!   TF_INK2_NVME_GIB=64          the cap on a rank's files in its directory (other identities' go at start)
//!   TF_INK2_NVME_FLOOR_GIB=32    the free space a write must leave on the disk (not agreed: a rank's own disk)
//! The identity (the directory) is the program, the checkpoint and the settings the ranks agreed (the switch itself
//! left out), the world and the layout: a new build, checkpoint or setting opens an empty directory, so deploys start
//! cold; each rank deletes the files it wrote under other identities when it starts. The checkpoint's contents must
//! be checked (a digest manifest: with TF_INK2_DIGESTS=none the tier stays off), and each file's head carries its
//! rank's drafter bytes as loaded (`content`, an xxh64), so a state is read only into the weights it was made with.
//! The tier deletes only files it wrote: names it makes (<32 hex>.r<rank>[.part] in ink2-<16 hex>/, a directory of
//! this user's that no one else may write), regular files whose head has its magic and this rank. Everything under
//! the root goes through directory descriptors held open; nothing below it is reached through a symlink, and a leaf
//! is opened without blocking and must be a regular file before a byte of it is read.
//!
//! File (format 1, every section aligned to 4 KiB so the reads and writes go past the page cache): a 256-byte head
//! (magic, sizes, identity, drafter bytes, name, hashes), the state's tokens (the whole key) and its image section
//! (count 0 in v0.3: text states only), then the payload: the fixed part, then each region's pages in position order.
//! The name is SHA-256 over the identity, N and the tokens; the head, the tokens and the payload each carry a hash
//! checked on read. Every size a head states is checked (no overflow, the position within the context, the fixed
//! part the engine's) before anything is allocated from it.
//!
//! Writing: an admission (every rank answers whether the state fits under the cap, what evictions must free, or that
//! no eviction can make room: then nothing is evicted and the state is not written; otherwise one least recently used
//! state goes a turn, on every rank, and the next turn asks again), a begin (the file made, unnamed on Linux
//! (O_TMPFILE) so a stop leaves nothing, else a `.part` made exclusively; its head and tokens), chunks (each reserving
//! its own space), then a commit on every rank: sealed, published (linked to its name, never over another file), the
//! directory synced, and only then the vote; a no on any rank takes every rank's publication back. Rank 0 indexes a
//! state after a yes only. A restart keeps the states every rank holds whole (rank 0's names, every rank's holdings
//! gathered); the rest are removed, then states go while a rank is over the cap by states every rank holds (a rank
//! over it by files no eviction frees is reported). Each rank counts its own files in its directory (read at start,
//! failing rather than undercounting; then kept as it writes and removes them, by link counts).
//! One engine a rank uses a root: it holds `.ink2-lock.r<rank>` there (flock) for its life, a file the tier never
//! writes or removes; so its start sweep of other identities' directories meets no live writer of its rank.
//! Writers sharing one volume (ranks on one disk) each check the floor alone: together they can pass it by up to
//! their in-flight writes' bytes (a Spark has its own disk and one engine).
//!
//! TF_INK2_NVME_FAULT (tests, one rank): `load:N` fails this rank's N-th load after its reads (the vote refuses it),
//! `write:N` its N-th write at the seal, `publish:N` its N-th publication (after the seal), `flip:N` flips one payload
//! byte of its N-th load before the hash, `prep:N` fails its N-th begin's preparation (as an allocation would),
//! `start:N` its N-th start, `unlink:N` keeps the file of its N-th drop (as a failed unlink would), `votes:N` fails
//! making its vote buffers at engine start-up, `partunlink:N` fails the `.part`'s unlink in its N-th named
//! publication (non-Linux, or no O_TMPFILE).
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const snap = @import("cuda_snap.zig");
const pages = @import("cuda_pages.zig");
const follow_ = @import("cuda_follow.zig");

const Engine = engine.Engine;
const Header = engine.Header;
const Leader = follow_.Leader;
const Kind = follow_.Kind;
const Sha = std.crypto.hash.sha2.Sha256;
const max_world = @import("inkling2").plan.max_world;

pub const format_version: u32 = 1;
/// Every section of a file and every read and write of one: O_DIRECT's unit.
pub const unit: u64 = 4096;
/// A write-behind step's bytes (a few ms of a rank's disk: a request arriving waits at most one).
pub const write_chunk: u64 = 8 << 20;
/// The staging buffer a rank (pinned, mapped): a load's read unit; it holds a whole fixed part.
pub const staging_min: u64 = 64 << 20;
/// States shorter than this are not written (a cold prefill of them is cheap).
pub const min_at: u32 = 1024;
/// A disk state is read only when it holds this many tokens past the device cache's resume point.
pub const min_gain: u32 = 1024;
pub const default_gib: f64 = 64;
pub const default_floor_gib: f64 = 32;

// ---------------------------------------------------------------- settings

pub const Config = struct {
    root: []const u8 = "",
    cap: u64 = @intFromFloat(default_gib * (1 << 30)),
    floor: u64 = @intFromFloat(default_floor_gib * (1 << 30)),

    pub fn on(c: Config) bool {
        return c.root.len > 0;
    }
};

fn gibBytes(v: ?[]const u8, default: f64) !u64 {
    const s = v orelse return @intFromFloat(default * (1 << 30));
    const g = std.fmt.parseFloat(f64, s) catch return error.BadNvmeSetting;
    if (!(g >= 0) or g > 1 << 20) return error.BadNvmeSetting;
    return @intFromFloat(g * (1 << 30));
}

/// The settings from their texts (null: unset).
pub fn configFrom(dir: ?[]const u8, cap_gib: ?[]const u8, floor_gib: ?[]const u8) !Config {
    const d = dir orelse "";
    if (d.len == 0 or std.mem.eql(u8, d, "0") or std.mem.eql(u8, d, "off")) return .{};
    if (d[0] != '/') return error.NvmeNeedsAbsoluteDir; // "1"/"on" name no directory
    if (d.len > 900) return error.BadNvmeSetting;
    var c: Config = .{ .root = d, .cap = try gibBytes(cap_gib, default_gib), .floor = try gibBytes(floor_gib, default_floor_gib) };
    if (c.cap == 0) c.root = ""; // a cap of 0 keeps nothing: off
    return c;
}

fn envSpan(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// This rank's settings (TF_INK2_NVME, TF_INK2_NVME_GIB, TF_INK2_NVME_FLOOR_GIB).
pub fn config() !Config {
    return configFrom(envSpan("TF_INK2_NVME"), envSpan("TF_INK2_NVME_GIB"), envSpan("TF_INK2_NVME_FLOOR_GIB"));
}

/// What the settings fingerprint adds when the tier is on (nothing when off, so the default digest stays v0.2's): the
/// switch and the cap (rank 0 decides evictions; the ranks keep the same files). The directory and the floor are each
/// rank's own.
pub fn settingsText(c: Config, buf: []u8) []const u8 {
    if (!c.on()) return "";
    return std.fmt.bufPrint(buf, " TF_INK2_NVME=on cap {d} format {d}", .{ c.cap, format_version }) catch unreachable;
}

/// The fingerprint's text for this rank's settings as it parsed them: a setting it refuses hashes as refused (with
/// why), so it agrees with no valid one, on or off.
pub fn settingsTextOf(parsed: anyerror!Config, buf: []u8) []const u8 {
    const c = parsed catch |err| return std.fmt.bufPrint(buf, " TF_INK2_NVME=refused {s}", .{@errorName(err)}) catch " TF_INK2_NVME=refused";
    return settingsText(c, buf);
}

// ---------------------------------------------------------------- format

pub const magic = "INK2NVME".*;

/// A file's first 256 bytes.
pub const Head = extern struct {
    magic: [8]u8 = magic,
    version: u32 = format_version,
    world: u32,
    rank: u32,
    at: u32,
    images: u32 = 0,
    npages: u32,
    regions: u32,
    reserved0: u32 = 0,
    run: u64,
    fixed: u64,
    payload: u64,
    head_len: u64,
    payload_hash: u64 = 0,
    meta_hash: u64 = 0,
    head_hash: u64 = 0,
    identity: [32]u8,
    name: [32]u8,
    /// This rank's drafter bytes as loaded (xxh64; 0: no drafter): its K and V and context are of those weights.
    content: u64 = 0,
    reserved: [11]u64 = @splat(0),

    comptime {
        std.debug.assert(@sizeOf(Head) == 256);
    }

    /// Its hash with `head_hash` zero.
    pub fn hash(h: Head) u64 {
        var x = h;
        x.head_hash = 0;
        return std.hash.Wyhash.hash(2, std.mem.asBytes(&x));
    }
};

pub fn alignUp(n: u64) u64 {
    return (n + unit - 1) / unit * unit;
}

/// `alignUp` of a size read from a file: null when it would overflow.
pub fn alignUpChecked(n: u64) ?u64 {
    const r = std.math.add(u64, n, unit - 1) catch return null;
    return r / unit * unit;
}

/// The head region's bytes (head, tokens, image section), aligned: the payload starts there.
pub fn headLen(at: u32, images: u32) u64 {
    return alignUp(@sizeOf(Head) + 4 * @as(u64, at) + 64 * @as(u64, images));
}

/// A file's bytes (sizes the engine made, bounded by its context).
pub fn fileBytes(at: u32, payload: u64) u64 {
    return headLen(at, 0) + alignUp(payload);
}

/// The payload a head's sizes give (the fixed part, then `regions` regions of `npages` pages of `run` bytes): null
/// when it would overflow.
pub fn payloadBytes(fixed: u64, regions: u32, npages: u32, run: u64) ?u64 {
    const pg = std.math.mul(u64, @as(u64, regions) * npages, run) catch return null;
    return std.math.add(u64, fixed, pg) catch null;
}

/// The layout facts that decide a file's bytes beyond the agreed settings.
pub const Layout = struct { page_rows: u32, regions: u32, run: u64, slot: u64 };

/// The directory's identity: the ranks' agreed program, checkpoint and settings (without the switch), the world and
/// the layout. Every rank derives the same one.
pub fn identityOf(program: [32]u8, checkpoint: [32]u8, settings: [32]u8, world: u32, l: Layout) [32]u8 {
    var h = Sha.init(.{});
    h.update("tensorfold inkling2 nvme identity 1\x00");
    h.update(&program);
    h.update(&checkpoint);
    h.update(&settings);
    h.update(std.mem.asBytes(&[4]u64{ world, l.page_rows, l.regions, l.run }));
    h.update(std.mem.asBytes(&l.slot));
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// A state's name: its identity, its position and every token (text states; an image section would follow).
pub fn nameOf(identity: [32]u8, at: u32, tokens: []const u32) [32]u8 {
    var h = Sha.init(.{});
    h.update("tensorfold inkling2 nvme name 1\x00");
    h.update(&identity);
    h.update(std.mem.asBytes(&at));
    h.update(std.mem.sliceAsBytes(tokens));
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

pub fn metaHash(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(tokens));
}

/// What a state's file must say, for this rank, before any byte of it is used.
pub const Want = struct {
    identity: [32]u8,
    /// This rank's drafter bytes (Head.content).
    content: u64 = 0,
    world: u32,
    rank: u32,
    regions: u32,
    run: u64,
    /// The position the file must hold (a load), or null (a scan: any up to `max_at`).
    at: ?u32 = null,
    /// The engine's context: no state holds more.
    max_at: u32,
    /// The fixed part's bytes at `at` (a load), or null (a scan: at most `max_fixed`, the arena slot's bytes; the
    /// start then checks each file's against its own position).
    fixed: ?u64,
    max_fixed: u64,
};

/// The head checked against `w` and its own sizes (every operation on them checked); the name is checked with the
/// tokens (`checkTokens`).
pub fn checkHead(h: Head, w: Want, file_len: ?u64) !void {
    if (!std.mem.eql(u8, &h.magic, &magic) or h.version != format_version) return error.NvmeNotOurs;
    if (h.head_hash != h.hash()) return error.NvmeHead;
    if (h.world != w.world or h.rank != w.rank) return error.NvmeOtherRank;
    if (!std.mem.eql(u8, &h.identity, &w.identity)) return error.NvmeOtherIdentity;
    if (h.content != w.content) return error.NvmeOtherContent;
    if (h.images != 0) return error.NvmeImages; // v0.3 writes text states only
    if (h.at == 0 or h.at > w.max_at) return error.NvmeLayout;
    if (w.at) |a| if (h.at != a) return error.NvmeLayout;
    if (h.npages != h.at / pages.page_rows or h.regions != w.regions or h.run != w.run) return error.NvmeLayout;
    if (w.fixed) |f| {
        if (h.fixed != f) return error.NvmeLayout;
    } else if (h.fixed > w.max_fixed) return error.NvmeLayout;
    if (h.head_len != headLen(h.at, 0)) return error.NvmeLayout; // at <= max_at: no overflow
    const payload = payloadBytes(h.fixed, h.regions, h.npages, h.run) orelse return error.NvmeLayout;
    if (h.payload != payload) return error.NvmeLayout;
    const padded = alignUpChecked(payload) orelse return error.NvmeLayout;
    const total = std.math.add(u64, h.head_len, padded) catch return error.NvmeLayout;
    if (file_len) |n| if (n != total) return error.NvmeSize;
}

pub fn checkTokens(h: Head, tokens: []const u32) !void {
    if (tokens.len != h.at) return error.NvmeLayout;
    if (metaHash(tokens) != h.meta_hash) return error.NvmeMeta;
    if (!std.mem.eql(u8, &nameOf(h.identity, h.at, tokens), &h.name)) return error.NvmeName;
}

// ---------------------------------------------------------------- the payload's pieces

/// One contiguous piece of a payload: `len` bytes at payload offset `off`, from the fixed part (at fixed offset
/// `src`) or the pool (device address `src`).
pub const Piece = struct { off: u64, len: u64, fixed: bool, src: u64 };

/// A state's payload: its fixed part (`fixed` bytes), then each of `regions` regions' pages `ids` (position order),
/// `run` bytes each, at the pool's place `pl`.
pub const Plan = struct {
    fixed: u64,
    regions: u32,
    run: u64,
    ids: []const u32,
    pl: pages.Place,

    pub fn bytes(p: Plan) u64 {
        return p.fixed + @as(u64, p.regions) * p.ids.len * p.run;
    }

    /// The pieces of payload bytes [lo, hi), in payload order, into `out` (each id run of a region one piece).
    pub fn pieces(p: Plan, gpa: std.mem.Allocator, lo: u64, hi: u64, out: *std.ArrayList(Piece)) !void {
        out.clearRetainingCapacity();
        if (lo >= hi) return;
        if (lo < p.fixed) {
            const e = @min(hi, p.fixed);
            try out.append(gpa, .{ .off = lo, .len = e - lo, .fixed = true, .src = lo });
        }
        const n = p.ids.len;
        for (0..p.regions) |r| {
            const rbase = p.fixed + @as(u64, r) * n * p.run;
            if (rbase >= hi) break;
            if (rbase + n * p.run <= lo) continue;
            var i: usize = 0;
            while (i < n) {
                var j = i + 1;
                while (j < n and p.ids[j] == p.ids[j - 1] + 1) j += 1;
                const s0 = rbase + i * p.run;
                const s1 = rbase + j * p.run;
                const a = @max(s0, lo);
                const b = @min(s1, hi);
                if (a < b) try out.append(gpa, .{ .off = a, .len = b - a, .fixed = false, .src = p.pl.region(r) + @as(u64, p.ids[i]) * p.run + (a - s0) });
                i = j;
            }
        }
    }
};

// ---------------------------------------------------------------- rank 0's index

pub const Entry = struct { name: [32]u8, at: u32, tokens: []u32, bytes: u64, used: u64 };

/// Rank 0: the states on disk (this identity), their tokens, bytes and last use.
pub const Index = struct {
    list: std.ArrayList(Entry) = .empty,
    bytes: u64 = 0,
    clock: u64 = 0,

    pub fn deinit(x: *Index, gpa: std.mem.Allocator) void {
        for (x.list.items) |e| gpa.free(e.tokens);
        x.list.deinit(gpa);
        x.* = .{};
    }

    pub fn find(x: *const Index, name: [32]u8) ?usize {
        for (x.list.items, 0..) |e, i| if (std.mem.eql(u8, &e.name, &name)) return i;
        return null;
    }

    /// Adds a state (its tokens owned by the index from here; on failure still the caller's).
    pub fn add(x: *Index, gpa: std.mem.Allocator, name: [32]u8, tokens: []u32, bytes: u64) !void {
        try x.list.ensureUnusedCapacity(gpa, 1);
        x.addAssumeCapacity(name, tokens, bytes);
    }

    /// Adds a state into room reserved before (`list.ensureUnusedCapacity`): it cannot fail.
    pub fn addAssumeCapacity(x: *Index, name: [32]u8, tokens: []u32, bytes: u64) void {
        x.clock += 1;
        x.list.appendAssumeCapacity(.{ .name = name, .at = @intCast(tokens.len), .tokens = tokens, .bytes = bytes, .used = x.clock });
        x.bytes += bytes;
    }

    pub fn remove(x: *Index, gpa: std.mem.Allocator, i: usize) Entry {
        const e = x.list.swapRemove(i);
        x.bytes -= e.bytes;
        gpa.free(e.tokens);
        return e;
    }

    pub fn touch(x: *Index, i: usize) void {
        x.clock += 1;
        x.list.items[i].used = x.clock;
    }

    /// The longest state that is a prefix of `prompt` strictly shorter than it, at least `above` tokens long.
    pub fn longest(x: *const Index, prompt: []const u32, above: usize) ?usize {
        var best: ?usize = null;
        for (x.list.items, 0..) |e, i| {
            if (e.at < above or e.at >= prompt.len) continue;
            if (best) |b| if (x.list.items[b].at >= e.at) continue;
            if (std.mem.eql(u32, e.tokens, prompt[0..e.at])) best = i;
        }
        return best;
    }

    /// The least recently used state.
    pub fn victim(x: *const Index) ?usize {
        var v: ?usize = null;
        for (x.list.items, 0..) |e, i| if (v == null or e.used < x.list.items[v.?].used) {
            v = i;
        };
        return v;
    }
};

// ---------------------------------------------------------------- names

const hex = "0123456789abcdef";

fn hexOf(bytes: []const u8, out: []u8) void {
    for (bytes, 0..) |b, i| {
        out[2 * i] = hex[b >> 4];
        out[2 * i + 1] = hex[b & 15];
    }
}

/// A state's file stem: its name's first 16 bytes (the other ranks get these in a step).
pub fn stemOf(name: [32]u8) [16]u8 {
    return name[0..16].*;
}

/// `ink2-<16 hex of the identity>`: the directory under the root.
pub fn dirName(buf: *[32]u8, identity: [32]u8) [:0]const u8 {
    var h: [16]u8 = undefined;
    hexOf(identity[0..8], &h);
    return std.fmt.bufPrintSentinel(buf, "ink2-{s}", .{&h}, 0) catch unreachable;
}

/// `<32 hex of the stem>.r<rank>` (`.part` while it is written): a state file's name in the directory.
pub fn fileName(buf: *[64]u8, stem: [16]u8, rank: u32, part: bool) [:0]const u8 {
    var h: [32]u8 = undefined;
    hexOf(&stem, &h);
    return std.fmt.bufPrintSentinel(buf, "{s}.r{d}{s}", .{ &h, rank, if (part) ".part" else "" }, 0) catch unreachable;
}

/// A file name this tier makes: its stem (the name's first 16 bytes as hex), its rank and whether it is a `.part`.
pub const FileName = struct { stem: [32]u8, rank: u32, part: bool };

pub fn parseFile(base: []const u8) ?FileName {
    if (base.len < 35 or base[32] != '.' or base[33] != 'r') return null;
    for (base[0..32]) |c| if (std.mem.indexOfScalar(u8, hex, c) == null) return null;
    var rest = base[34..];
    const part = std.mem.endsWith(u8, rest, ".part");
    if (part) rest = rest[0 .. rest.len - 5];
    // the rank as the tier spells it: decimal digits, no leading zero, at most 3
    if (rest.len == 0 or rest.len > 3 or (rest.len > 1 and rest[0] == '0')) return null;
    for (rest) |c| if (c < '0' or c > '9') return null;
    const rank = std.fmt.parseInt(u32, rest, 10) catch return null;
    var f: FileName = .{ .stem = undefined, .rank = rank, .part = part };
    @memcpy(&f.stem, base[0..32]);
    return f;
}

/// `ink2-<16 hex>`: a directory this tier makes.
pub fn isTierDir(base: []const u8) bool {
    if (base.len != 21 or !std.mem.startsWith(u8, base, "ink2-")) return false;
    for (base[5..]) |c| if (std.mem.indexOfScalar(u8, hex, c) == null) return false;
    return true;
}

// ---------------------------------------------------------------- files (through directory descriptors)

const Count = if (builtin.os.tag == .linux) u64 else u32;
const StatVfs = extern struct {
    bsize: u64,
    frsize: u64,
    blocks: Count,
    bfree: Count,
    bavail: Count,
    files: Count,
    ffree: Count,
    favail: Count,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [if (builtin.os.tag == .linux) 24 else 0]u8,
};
extern "c" fn fstatvfs(fd: c_int, result: *StatVfs) c_int;
extern "c" fn posix_fallocate(fd: c_int, offset: i64, len: i64) c_int;

const s_ifmt: u32 = 0o170000;
const s_ifreg: u32 = 0o100000;
const s_ifdir: u32 = 0o040000;

/// An open file's type and permissions, owner and bytes.
pub const Info = struct { mode: u32, uid: u32, size: u64, nlink: u64 = 1 };

pub fn info(fd: c_int) ?Info {
    if (builtin.os.tag == .linux) {
        var sx: std.os.linux.Statx = undefined;
        // AT_EMPTY_PATH (0x1000): the descriptor itself (glibc's fstat binding is not in std for Linux)
        if (std.c.statx(fd, "", 0x1000, .{ .TYPE = true, .MODE = true, .UID = true, .SIZE = true, .NLINK = true }, &sx) != 0) return null;
        return .{ .mode = sx.mode, .uid = sx.uid, .size = sx.size, .nlink = sx.nlink };
    }
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return null;
    return .{ .mode = @intCast(st.mode), .uid = @intCast(st.uid), .size = @intCast(st.size), .nlink = @intCast(st.nlink) };
}

fn openDirAt(dir: c_int, name: [*:0]const u8) c_int {
    return std.c.openat(dir, name, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
}

/// The root (TF_INK2_NVME) opened component by component from `/`, none through a symlink; missing directories are
/// made (0700). Its descriptor.
pub fn openRoot(path: []const u8) !c_int {
    if (path.len == 0 or path[0] != '/') return error.NvmeDir;
    var fd = std.c.open("/", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.NvmeDir;
    var it = std.mem.tokenizeScalar(u8, path, '/');
    while (it.next()) |comp| {
        if (std.mem.eql(u8, comp, ".") or std.mem.eql(u8, comp, "..") or comp.len > 255) {
            _ = std.c.close(fd);
            return error.NvmeDir;
        }
        var nb: [256:0]u8 = @splat(0);
        @memcpy(nb[0..comp.len], comp);
        var sub = openDirAt(fd, &nb);
        if (sub < 0) {
            _ = std.c.mkdirat(fd, &nb, 0o700); // missing: made (a symlink or a file there stays one, and refuses below)
            sub = openDirAt(fd, &nb);
        }
        _ = std.c.close(fd);
        if (sub < 0) return error.NvmeDir; // a symlink, not a directory, or not ours to make
        fd = sub;
    }
    return fd;
}

/// `name` under `dir` opened as one of this tier's directories: a directory (no symlink followed), this user's, that
/// no one else may write; null otherwise.
pub fn openTierDir(dir: c_int, name: [*:0]const u8) ?c_int {
    const fd = openDirAt(dir, name);
    if (fd < 0) return null;
    const fi = info(fd) orelse {
        _ = std.c.close(fd);
        return null;
    };
    if (fi.mode & s_ifmt != s_ifdir or fi.uid != std.c.geteuid() or fi.mode & 0o022 != 0) {
        _ = std.c.close(fd);
        return null;
    }
    return fd;
}

fn openFlags(comptime acc: std.c.O, direct: bool) std.c.O {
    var f = acc;
    f.CLOEXEC = true;
    f.NOFOLLOW = true;
    if (builtin.os.tag == .linux) f.DIRECT = direct;
    return f;
}

fn nocache(fd: c_int) void {
    if (builtin.os.tag == .macos) _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
}

/// A file made by `create`: its descriptor, and whether its writes go past the page cache.
/// `unnamed`: the file has no name until its publication (Linux O_TMPFILE): a stop before then leaves nothing.
pub const Made = struct { fd: c_int, direct: bool, unnamed: bool = false };

/// A new file for a write, mode 0600, then switched to O_DIRECT where the filesystem takes it (Linux; it stays
/// buffered otherwise) or F_NOCACHE (macOS). Linux: an unnamed inode in `dir` (O_TMPFILE), named only when it is
/// published whole (`publishFd`); where the filesystem has no O_TMPFILE (and on macOS), `name` made exclusively
/// (nothing is removed on a failure). `unnamed` = false forces the named form (tests).
pub fn create(dir: c_int, name: [:0]const u8, unnamed: bool) !Made {
    var fd: c_int = -1;
    var tmp = false;
    if (builtin.os.tag == .linux) if (unnamed) {
        // O_TMPFILE is __O_TMPFILE with O_DIRECTORY (the kernel refuses one without the other)
        fd = std.c.openat(dir, ".", .{ .ACCMODE = .WRONLY, .TMPFILE = true, .DIRECTORY = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        tmp = fd >= 0;
        if (fd < 0) switch (std.c.errno(fd)) {
            .OPNOTSUPP, .ISDIR, .INVAL => {}, // no O_TMPFILE here: the named form below
            else => return error.NvmeCreate,
        };
    };
    if (fd < 0) fd = std.c.openat(dir, name, openFlags(.{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, false), @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.NvmeCreate;
    var direct = false;
    if (builtin.os.tag == .linux) {
        const fl = std.c.fcntl(fd, std.c.F.GETFL);
        const bit: c_int = @bitCast(std.c.O{ .DIRECT = true });
        direct = fl >= 0 and std.c.fcntl(fd, std.c.F.SETFL, fl | bit) == 0;
    } else {
        nocache(fd);
        direct = builtin.os.tag == .macos;
    }
    return .{ .fd = fd, .direct = direct, .unnamed = tmp };
}

/// `name` in `dir` for reading: never blocking on it, no symlink, and only a regular file.
pub fn openRead(dir: c_int, name: [:0]const u8) !c_int {
    var fd = std.c.openat(dir, name, openFlags(.{ .ACCMODE = .RDONLY, .NONBLOCK = true }, true), @as(std.c.mode_t, 0));
    if (fd < 0 and builtin.os.tag == .linux and std.c.errno(fd) == .INVAL) fd = std.c.openat(dir, name, openFlags(.{ .ACCMODE = .RDONLY, .NONBLOCK = true }, false), @as(std.c.mode_t, 0));
    if (fd < 0) return error.NvmeMissing;
    const fi = info(fd) orelse {
        _ = std.c.close(fd);
        return error.NvmeRead;
    };
    if (fi.mode & s_ifmt != s_ifreg) {
        _ = std.c.close(fd);
        return error.NvmeNotOurs;
    }
    nocache(fd);
    return fd;
}

pub fn fileLen(fd: c_int) ?u64 {
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    return if (end < 0) null else @intCast(end);
}

/// All of `buf` at `at` (aligned buffer, offset and length for O_DIRECT).
pub fn writeAll(fd: c_int, buf: []const u8, at: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = std.c.pwrite(fd, buf.ptr + done, @min(buf.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.NvmeWrite;
        done += @intCast(n);
    }
}

pub fn readAll(fd: c_int, buf: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = std.c.pread(fd, buf.ptr + done, @min(buf.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.NvmeRead;
        done += @intCast(n);
    }
}

pub fn syncData(fd: c_int) !void {
    const r = if (builtin.os.tag == .linux) std.c.fdatasync(fd) else std.c.fsync(fd);
    if (r != 0) return error.NvmeSync;
}

/// Space for bytes [at, at + n) of the file taken before they are written (Linux): one chunk's, so no idle unit
/// reserves a whole file.
pub fn reserve(fd: c_int, at: u64, n: u64) !void {
    if (builtin.os.tag != .linux) return;
    if (posix_fallocate(fd, @intCast(at), @intCast(n)) != 0) return error.NvmeNoSpace;
}

/// The directory's entries made durable (a publication, a removal).
pub fn syncDir(dir: c_int) !void {
    if (std.c.fsync(dir) != 0) return error.NvmeSync;
}

/// Bytes free to an unprivileged writer on the disk of directory `dir`.
pub fn freeBytes(dir: c_int) ?u64 {
    var s: StatVfs = undefined;
    if (fstatvfs(dir, &s) != 0) return null;
    return std.math.mul(u64, s.frsize, s.bavail) catch null;
}

/// The unnamed file `fd` (O_TMPFILE) published as `dest` in `dir`: a link, which never replaces a file there.
pub fn publishFd(dir: c_int, fd: c_int, dest: [:0]const u8) !void {
    var pb: [40]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&pb, "/proc/self/fd/{d}", .{fd}, 0) catch unreachable;
    const at_fdcwd: c_int = -100;
    const at_symlink_follow: c_uint = 0x400;
    if (std.c.linkat(at_fdcwd, path, dir, dest, at_symlink_follow) != 0) return error.NvmePublish;
}

/// The named `.part` `part` published as `dest` in `dir`: a hard link, which never replaces a file there, then the
/// `.part` unlinked. If that unlink fails (`fail_unlink`: tests) the link is taken back and the publication fails,
/// so the file keeps one name, the `.part`, for the clean-up.
pub fn publish(dir: c_int, part: [:0]const u8, dest: [:0]const u8, fail_unlink: bool) !void {
    if (std.c.linkat(dir, part, dir, dest, 0) != 0) return error.NvmePublish;
    if (!fail_unlink and std.c.unlinkat(dir, part, 0) == 0) return;
    _ = std.c.unlinkat(dir, dest, 0);
    return error.NvmePublishPart;
}

/// The open file `fd` named `name` is this tier's file of rank `rank`: a name it makes for that rank, a regular file
/// of this user's whose head has the magic and the rank. Its bytes then.
fn oursFd(fd: c_int, name: []const u8, rank: u32) ?u64 {
    const f = parseFile(name) orelse return null;
    if (f.rank != rank) return null;
    const fi = info(fd) orelse return null;
    if (fi.mode & s_ifmt != s_ifreg or fi.uid != std.c.geteuid()) return null;
    var hb: [@sizeOf(Head)]u8 = undefined;
    if (std.c.pread(fd, &hb, hb.len, 0) != hb.len) return null;
    const h = std.mem.bytesToValue(Head, &hb);
    if (!std.mem.eql(u8, &h.magic, &magic) or h.rank != rank) return null;
    return fi.size;
}

fn openLeaf(dir: c_int, name: [:0]const u8) c_int {
    return std.c.openat(dir, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(std.c.mode_t, 0));
}

/// Whether `name` in `dir` is this tier's file of rank `rank` (`oursFd`).
pub fn ours(dir: c_int, name: [:0]const u8, rank: u32) bool {
    const fd = openLeaf(dir, name);
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    return oursFd(fd, name, rank) != null;
}

/// What became of a file the tier meant to remove.
pub const Gone = union(enum) { removed: u64, absent, kept };

/// `name` in `dir` unlinked when it is this tier's file of rank `rank` (`oursFd`); its bytes when it went.
pub fn removeOurs(dir: c_int, name: [:0]const u8, rank: u32) Gone {
    const fd = openLeaf(dir, name);
    if (fd < 0) return if (std.c.errno(fd) == .NOENT) .absent else .kept;
    const size = oursFd(fd, name, rank);
    _ = std.c.close(fd);
    const n = size orelse return .kept;
    if (std.c.unlinkat(dir, name, 0) != 0) return .kept;
    return .{ .removed = n };
}

extern "c" fn flock(fd: c_int, op: c_int) c_int;
const lock_ex: c_int = 2;
const lock_nb: c_int = 4;

/// `.ink2-lock.r<rank>`: the file a rank's engine holds locked (flock) in the root, for its life. The tier never
/// removes or writes it (nor any other lock name), so every engine of a rank locks the same inode.
pub fn lockName(buf: *[24]u8, rank: u32) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, ".ink2-lock.r{d}", .{rank}, 0) catch unreachable;
}

/// Rank `rank`'s lock in root `dir` taken (made empty when missing): one engine of a rank on a root at a time, so
/// no other engine of this rank creates, writes or removes files under it while it is held (start-up refuses), and
/// its start sweep of other identities' directories meets no live writer of its rank. Its descriptor.
pub fn lockRoot(dir: c_int, rank: u32) !c_int {
    var lb: [24]u8 = undefined;
    const name = lockName(&lb, rank);
    const fd = std.c.openat(dir, name, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true, .NONBLOCK = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.NvmeLock;
    const fi = info(fd) orelse {
        _ = std.c.close(fd);
        return error.NvmeLock;
    };
    if (fi.mode & s_ifmt != s_ifreg or fi.uid != std.c.geteuid()) {
        _ = std.c.close(fd);
        return error.NvmeLock;
    }
    if (flock(fd, lock_ex | lock_nb) != 0) {
        _ = std.c.close(fd);
        return error.NvmeBusy; // another engine of this rank uses this root
    }
    return fd;
}

/// This rank's bytes in its files in tier directory `dir`: the files themselves (a file it could not remove, or a
/// `.part` of a stopped write, counts while it is there). Read once at start; the tier keeps the count after. It
/// fails rather than undercount: a directory it cannot read, or a file of this rank's name it cannot open or read.
pub fn usedBytes(dir: c_int, rank: u32) !u64 {
    var n: u64 = 0;
    const d = dirStream(dir) orelse return error.NvmeCount;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |fe| {
        const fbase = entName(fe);
        const f = parseFile(fbase) orelse continue;
        if (f.rank != rank) continue;
        const fd = openLeaf(dir, fbase);
        if (fd < 0) {
            if (std.c.errno(fd) == .NOENT) continue; // gone since the listing
            return error.NvmeCount;
        }
        defer _ = std.c.close(fd);
        const fi = info(fd) orelse return error.NvmeCount;
        if (fi.mode & s_ifmt != s_ifreg or fi.uid != std.c.geteuid()) continue; // not the tier's
        var hb: [@sizeOf(Head)]u8 = undefined;
        const got = std.c.pread(fd, &hb, hb.len, 0);
        if (got < 0) return error.NvmeCount;
        if (got != hb.len) continue; // too short to be the tier's
        const h = std.mem.bytesToValue(Head, &hb);
        if (std.mem.eql(u8, &h.magic, &magic) and h.rank == rank) n += fi.size;
    }
    return n;
}

extern "c" fn rewinddir(d: *std.c.DIR) void;

/// A stream over directory `dir`'s entries from the first (a dup shares the held descriptor's offset, which an
/// earlier stream left at the end).
fn dirStream(dir: c_int) ?*std.c.DIR {
    const fd = std.c.dup(dir);
    if (fd < 0) return null;
    const d = std.c.fdopendir(fd) orelse {
        _ = std.c.close(fd);
        return null;
    };
    rewinddir(d);
    return d;
}

/// The entry's name as a sentinel slice (its buffer holds the 0).
fn entName(ent: *const std.c.dirent) [:0]const u8 {
    const base = std.mem.sliceTo(&ent.name, 0);
    return ent.name[0..base.len :0];
}

/// Rank `rank`'s files under other identities' directories of the root `root` (and its `.part` files under its own,
/// `own`) removed: files of this tier only (`removeOurs`), in its directories only (`openTierDir`); other identities'
/// directories go too when empty. The caller holds rank `rank`'s root lock (`lockRoot`). Bytes removed.
pub fn sweep(root: c_int, own: []const u8, rank: u32) u64 {
    var freed: u64 = 0;
    const d = dirStream(root) orelse return 0;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const base = entName(ent);
        if (!isTierDir(base)) continue;
        const is_own = std.mem.eql(u8, base, own);
        const sub = openTierDir(root, base) orelse continue; // a symlink, or not the tier's: not entered
        defer _ = std.c.close(sub);
        if (is_own) {
            freed += sweepDir(sub, rank, true); // this engine holds its lock here
            continue;
        }
        // another identity's: no engine of this rank writes there (the caller holds this rank's root lock)
        freed += sweepDir(sub, rank, false);
        if (!is_own) _ = std.c.unlinkat(root, base, std.c.AT.REMOVEDIR); // only when empty: another rank's or a stranger's file keeps it
    }
    return freed;
}

fn sweepDir(dir: c_int, rank: u32, parts_only: bool) u64 {
    var freed: u64 = 0;
    const d = dirStream(dir) orelse return 0;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const base = entName(ent);
        const f = parseFile(base) orelse continue;
        if (f.rank != rank or (parts_only and !f.part)) continue;
        switch (removeOurs(dir, base, rank)) {
            .removed => |n| freed += n,
            else => {},
        }
    }
    return freed;
}

/// One state file found by `scan`: its head and tokens (owned); `listed`: rank 0 names it (the start's reconciliation).
pub const Found = struct { head: Head, tokens: []u32, bytes: u64, listed: bool = false };

fn freeFound(gpa: std.mem.Allocator, found: *std.ArrayList(Found)) void {
    for (found.items) |f| gpa.free(f.tokens);
    found.deinit(gpa);
}

/// The head and tokens of file `name` in `dir`, checked against `w` before anything is allocated from them (its
/// payload is checked when it is read); `buf` an aligned scratch of at least `unit` bytes.
pub fn readHead(gpa: std.mem.Allocator, dir: c_int, name: [:0]const u8, w: Want, buf: []align(unit) u8) !Found {
    const fd = try openRead(dir, name);
    defer _ = std.c.close(fd);
    const len = fileLen(fd) orelse return error.NvmeRead;
    if (len < unit) return error.NvmeSize;
    try readAll(fd, buf[0..unit], 0);
    const h = std.mem.bytesToValue(Head, buf[0..@sizeOf(Head)]);
    try checkHead(h, w, len);
    const tokens = try gpa.alloc(u32, h.at); // h.at <= w.max_at, the context
    errdefer gpa.free(tokens);
    // the tokens through the aligned scratch, `unit`-aligned spans
    const t_bytes = std.mem.sliceAsBytes(tokens);
    var done: usize = 0;
    var pos: u64 = 0;
    while (done < t_bytes.len) {
        const span = @min(buf.len, h.head_len - pos);
        try readAll(fd, buf[0..span], pos);
        const from: usize = if (pos == 0) @sizeOf(Head) else 0;
        const take = @min(span - from, t_bytes.len - done);
        @memcpy(t_bytes[done..][0..take], buf[from..][0..take]);
        done += take;
        pos += span;
    }
    try checkTokens(h, tokens);
    return .{ .head = h, .tokens = tokens, .bytes = len };
}

/// Every state file of rank `w.rank` in `dir`, checked: good ones into `out`; a bad one of ours is removed.
pub fn scan(gpa: std.mem.Allocator, dir: c_int, w: Want, buf: []align(unit) u8, out: *std.ArrayList(Found)) !u32 {
    var dropped: u32 = 0;
    const d = dirStream(dir) orelse return 0;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const base = entName(ent);
        const f = parseFile(base) orelse continue;
        if (f.rank != w.rank or f.part) continue;
        const got = readHead(gpa, dir, base, w, buf) catch |err| {
            if (err == error.OutOfMemory) return err;
            const gone = removeOurs(dir, base, w.rank);
            if (gone == .removed) dropped += 1;
            std.log.warn("inkling2 rank {d}: NVMe prompt cache: {s} not usable ({s}){s}", .{ w.rank, base, @errorName(err), if (gone == .removed) ", removed" else ", left alone (not this rank's file of the tier)" });
            continue;
        };
        var h: [32]u8 = undefined;
        hexOf(got.head.name[0..16], &h);
        if (!std.mem.eql(u8, &h, &f.stem)) {
            gpa.free(got.tokens);
            if (removeOurs(dir, base, w.rank) == .removed) dropped += 1;
            continue;
        }
        out.append(gpa, got) catch |err| {
            gpa.free(got.tokens);
            return err;
        };
    }
    return dropped;
}

// ---------------------------------------------------------------- the tier on every rank

/// A file being written (every rank) and, on rank 0, the state it is of.
pub const Job = struct {
    fd: c_int = -1,
    name: [32]u8,
    at: u32,
    off: u64,
    tag: u32,
    ids: []u32,
    plan_fixed: u64,
    payload: u64,
    head_len: u64,
    next: u64 = 0,
    hash: std.hash.Wyhash = .init(0),
    failed: ?anyerror = null,
    /// Rank 0: the device cache's state and its tokens (owned; the index takes them at commit). `stale`: the device
    /// cache dropped the state (nothing more is copied from its pages or extent); the next idle turn gives it up.
    kept: ?*snap.Kept = null,
    tokens: []u32 = &.{},
    stale: bool = false,
    /// The file's writes go past the page cache; otherwise each chunk is synced as written (no unit flushes a file).
    direct: bool = true,
    /// The file has no name until it is published (O_TMPFILE); else it is the `.part`.
    unnamed: bool = false,
};

/// Rank 0: a state's files to remove at an idle turn (its stem, its bytes a rank).
pub const Drop = struct { stem: [16]u8, bytes: u64 };

/// Rank 0: a state waiting to be written, and its tokens (owned).
pub const Wanted = struct { kept: *snap.Kept, name: [32]u8, tokens: []u32 };

pub const Stats = struct {
    writes: u64 = 0,
    write_bytes: u64 = 0,
    write_ns: u64 = 0,
    write_fails: u64 = 0,
    aborts: u64 = 0,
    /// States an admission refused (a rank's files no eviction frees): nothing was evicted for them.
    refused: u64 = 0,
    /// The longest idle unit rank 0 ran (ns), and of its begins and commits: what a request arriving waits at most.
    unit_max_ns: u64 = 0,
    begin_max_ns: u64 = 0,
    commit_max_ns: u64 = 0,
    loads: u64 = 0,
    load_bytes: u64 = 0,
    load_ns: u64 = 0,
    load_fails: u64 = 0,
    evicted: u64 = 0,
    skipped: u64 = 0,
};

pub const Tier = struct {
    on: bool = false,
    cfg: Config = .{},
    rank: u32 = 0,
    root_buf: [1024]u8 = undefined,
    root_len: usize = 0,
    root_fd: c_int = -1,
    dir_buf: [32]u8 = undefined,
    dir_len: usize = 0,
    dir_fd: c_int = -1,
    /// The ranks' common identity (the directory's, and in every name).
    identity: [32]u8 = @splat(0),
    /// This rank's drafter bytes (Head.content).
    content: u64 = 0,
    staging: ?cuda.HostBuffer = null,
    staging_dev: u64 = 0,
    xfer: ?cuda.DeviceBuffer = null,
    /// The votes' buffers (device and pinned host), made at engine start-up when the tier is on (`prepare`), kept
    /// when the tier turns off.
    votes: ?cuda.DeviceBuffer = null,
    vote_host: ?cuda.HostBuffer = null,
    /// The head block of the file being written (its hashes filled at commit).
    first: [unit]u8 = undefined,
    job: ?Job = null,
    pieces: std.ArrayList(Piece) = .empty,
    /// This rank's lock in the root (`lockRoot`), held for the tier's life.
    lock_fd: c_int = -1,
    /// Rank 0.
    index: Index = .{},
    queue: std.ArrayList(Wanted) = .empty,
    /// Rank 0: states to remove from every disk at the next idle turns (a read refused while serving), stem and bytes.
    drops: std.ArrayList(Drop) = .empty,
    /// A rank could not make its vote buffers at start-up (`prepare`): the tier stays off on every rank.
    unprepared: bool = false,
    /// W = 1: the admission's answer.
    need_one: [1]u64 = .{0},
    /// This rank's bytes in its files in its directory (`usedBytes` at start, then kept), and of them the files of
    /// states every rank holds (the rest: files it could not remove, which no eviction frees).
    used: u64 = 0,
    live: u64 = 0,
    stats: Stats = .{},
    faults: struct { loads: u64 = 0, writes: u64 = 0, flips: u64 = 0, preps: u64 = 0, publishes: u64 = 0, starts: u64 = 0, unlinks: u64 = 0, preps_votes: u64 = 0, part_unlinks: u64 = 0 } = .{},

    pub fn root(t: *const Tier) []const u8 {
        return t.root_buf[0..t.root_len];
    }

    pub fn dir(t: *const Tier) [:0]const u8 {
        return t.dir_buf[0..t.dir_len :0];
    }

    pub fn deinit(t: *Tier, gpa: std.mem.Allocator) void {
        if (t.job) |*j| {
            abandon(t, j);
            gpa.free(j.ids);
            if (j.tokens.len > 0) gpa.free(j.tokens);
        }
        t.job = null;
        if (t.staging) |*b| b.free();
        if (t.xfer) |*b| b.free();
        if (t.votes) |*b| b.free();
        if (t.vote_host) |*b| b.free();
        if (t.lock_fd >= 0) _ = std.c.close(t.lock_fd); // the lock goes with it
        if (t.dir_fd >= 0) _ = std.c.close(t.dir_fd);
        if (t.root_fd >= 0) _ = std.c.close(t.root_fd);
        t.pieces.deinit(gpa);
        t.index.deinit(gpa);
        for (t.queue.items) |w| gpa.free(w.tokens);
        t.queue.deinit(gpa);
        t.drops.deinit(gpa);
        t.* = .{};
    }

    fn stage(t: *const Tier) []align(unit) u8 {
        return @alignCast(t.staging.?.bytes);
    }

    /// The staging buffer's upper half: a step's ids and tokens on their way to or from the other ranks (the lower
    /// half takes the head and tokens a begin writes).
    fn upper(t: *const Tier) []align(unit) u8 {
        const s = t.stage();
        return @alignCast(s[s.len / 2 ..]);
    }
};

/// The bytes a rank's staging takes when the tier is on (pinned host memory, from the memory reserve).
pub fn stagingBytes(e: *const Engine) u64 {
    return @max(staging_min, alignUp(snap.slotBytes(e.w.layers, if (e.drafter) |dr| dr.hk else 0)));
}

/// The transfer buffer a rank: a begin's page ids and tokens at the agreed context (so every rank's is the same size).
pub fn xferBytes(context: usize) usize {
    return 4 * (context + context / pages.page_rows + 64);
}

fn fixedAt(e: *const Engine, at: u32) u64 {
    return snap.rawBytes(e.w.layers, if (e.drafter) |dr| dr.hk else 0, at);
}

fn maxAt(e: *const Engine) u32 {
    return @intCast(@min(e.o.context, std.math.maxInt(u32)));
}

/// What this rank's files must say (`at`: a load's position, whose fixed part's bytes it fixes; null: any).
fn want(e: *const Engine, t: *const Tier, at: ?u32) Want {
    const p = e.pool.?;
    return .{ .identity = t.identity, .content = t.content, .world = e.o.world, .rank = e.o.rank, .regions = @intCast(p.regions), .run = p.place().run, .at = at, .max_at = maxAt(e), .fixed = if (at) |a| fixedAt(e, a) else null, .max_fixed = snap.bytes(e, 0) };
}

fn layoutOf(e: *const Engine) Layout {
    const p = e.pool.?;
    return .{ .page_rows = @intCast(pages.page_rows), .regions = @intCast(p.regions), .run = p.place().run, .slot = snap.bytes(e, 0) };
}

fn fault(comptime name: []const u8, count: *u64) bool {
    const v = std.mem.span(std.c.getenv("TF_INK2_NVME_FAULT") orelse return false);
    const colon = std.mem.indexOfScalar(u8, v, ':') orelse return false;
    if (!std.mem.eql(u8, v[0..colon], name)) return false;
    const n = std.fmt.parseInt(u64, v[colon + 1 ..], 10) catch return false;
    count.* += 1;
    if (count.* != n) return false;
    std.log.warn("inkling2: TF_INK2_NVME_FAULT {s}:{d} induced", .{ name, n });
    return true;
}

fn nowNs(e: *const Engine) u64 {
    return @intCast(std.Io.Clock.awake.now(e.io).toNanoseconds());
}

/// The vote buffers' bytes: a u64 a rank (admissions) or a u32 (votes).
const votes_bytes = 8 * max_world;

/// Engine start-up, every rank (in step: after the fingerprint's agreement), before any step: the vote buffers when
/// the tier is on (votes never allocate). Whether every rank made them is agreed through the engine's header
/// buffer (made before, for the steps): a rank that could not leaves the tier off on every rank, and the engine
/// serves on without it. Only a failure of that agreement itself (a collective) returns an error.
pub fn prepare(e: *Engine) !void {
    if (e.o.group == null or e.nvme.votes != null) return;
    const c = config() catch return; // refused: the fingerprint refuses it on every rank
    if (!c.on()) return;
    const t = &e.nvme;
    var made = !fault("votes", &t.faults.preps_votes);
    if (made) {
        if (cuda.HostBuffer.allocMapped(e.d, votes_bytes)) |hb| {
            t.vote_host = hb;
        } else |_| made = false;
    }
    if (made) {
        if (cuda.DeviceBuffer.alloc(e.d, votes_bytes)) |db| {
            t.votes = db;
        } else |_| made = false;
    }
    if (try agreeStartup(e, made)) return;
    if (t.votes) |*b| b.free();
    if (t.vote_host) |*b| b.free();
    t.votes = null;
    t.vote_host = null;
    t.unprepared = true;
    std.log.warn("inkling2 rank {d}: NVMe prompt cache off: a rank could not make its vote buffers{s}", .{ e.o.rank, if (made) "" else " (this one)" });
}

/// Every rank's `ok` at start-up, through the engine's header buffer (device) and this stack (pageable: the copies
/// are staged by the driver on the engine's stream, before the gather): true when every rank's is.
fn agreeStartup(e: *Engine, ok: bool) !bool {
    const g = e.o.group orelse return ok;
    const W: usize = e.o.world;
    if (e.hdr.len < 4 * W) return error.NvmeNoVotes;
    const mine: u32 = @intFromBool(ok);
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(e.hdr.ptr + e.o.rank * 4, &mine, 4, e.stream.handle), "cuMemcpyHtoDAsync");
    try g.allGather(e.stream, e.hdr.ptr + e.o.rank * 4, e.hdr.ptr, 4);
    try e.stream.synchronize();
    try g.check();
    var all: [max_world]u32 = undefined;
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.hdr.ptr, 4 * W), "cuMemcpyDtoH");
    for (all[0..W]) |v| if (v != 1) return false;
    return true;
}

/// Host bytes `src` (pinned: the staging or the vote buffer) to device address `dst` on the engine's stream: no
/// staging of its own is taken, so nothing is allocated between a step's collectives.
fn toDev(e: *Engine, dst: u64, src: []const u8) !void {
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, e.stream.handle), "cuMemcpyHtoDAsync");
}

/// Device bytes at `src` to pinned host `dst`, waited for.
fn toHost(e: *Engine, dst: []u8, src: u64) !void {
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
}

/// This rank's tier made ready (its directories, other identities' files swept, its staging and transfer buffers,
/// its own directory read); `found` gets the states there. Every rank; before any tier step.
fn init(e: *Engine, found: *std.ArrayList(Found)) !void {
    const t = &e.nvme;
    const c = try config();
    if (!c.on()) return error.NvmeOff;
    if (e.pool == null or e.snap.buf == null) return error.NvmeNeedsCache;
    const id = e.nvme_identity orelse return error.NvmeNoIdentity;
    // states are kept only for weights whose bytes were checked (a digest manifest): with TF_INK2_DIGESTS=none a
    // changed checkpoint of the same layout would read the old one's states
    if (!id.checked) return error.NvmeNeedsDigests;
    if (fault("start", &t.faults.starts)) return error.OutOfMemory;
    t.cfg = c;
    t.rank = e.o.rank;
    t.identity = identityOf(id.program, id.checkpoint, id.settings, e.o.world, layoutOf(e));
    t.content = if (e.drafter) |dr| dr.content else 0;
    if (c.root.len > t.root_buf.len) return error.BadNvmeSetting;
    @memcpy(t.root_buf[0..c.root.len], c.root);
    t.root_len = c.root.len;
    t.cfg.root = t.root();
    t.root_fd = try openRoot(c.root);
    t.lock_fd = try lockRoot(t.root_fd, e.o.rank); // before any file work: one engine of this rank on this root
    const dn = dirName(&t.dir_buf, t.identity);
    t.dir_len = dn.len;
    _ = std.c.mkdirat(t.root_fd, dn, 0o700); // there already, or made now (checked when opened)
    t.dir_fd = openTierDir(t.root_fd, dn) orelse return error.NvmeDir;
    const swept = sweep(t.root_fd, dn, e.o.rank);
    if (t.staging == null) {
        var sb = try cuda.HostBuffer.allocMapped(e.d, @intCast(stagingBytes(e)));
        errdefer sb.free();
        t.staging_dev = try sb.device();
        t.staging = sb;
    }
    if (t.xfer == null) t.xfer = try cuda.DeviceBuffer.alloc(e.d, xferBytes(e.o.context));
    _ = try scan(e.gpa, t.dir_fd, want(e, t, null), t.stage(), found);
    // each file's fixed part against its own position
    var i: usize = 0;
    while (i < found.items.len) {
        const f = found.items[i];
        if (f.head.fixed == fixedAt(e, f.head.at)) {
            i += 1;
            continue;
        }
        var nb: [64]u8 = undefined;
        _ = removeOurs(t.dir_fd, fileName(&nb, stemOf(f.head.name), e.o.rank, false), e.o.rank);
        e.gpa.free(f.tokens);
        _ = found.swapRemove(i);
    }
    t.used = try usedBytes(t.dir_fd, e.o.rank); // a count it cannot make whole refuses the start (every rank)
    t.live = 0; // the reconciliation adds the states every rank holds
    t.on = true;
    std.log.info("inkling2 rank {d}: NVMe prompt cache on: {s}/{s}, {d} states there, cap {d:.1} GiB, floor {d:.1} GiB free, staging {d} MiB pinned{s}", .{ e.o.rank, t.root(), t.dir(), found.items.len, gib(c.cap), gib(c.floor), stagingBytes(e) >> 20, if (swept > 0) "; other identities' files removed" else "" });
    if (swept > 0) std.log.info("inkling2 rank {d}: NVMe prompt cache: {d:.2} GiB of earlier builds' or settings' states removed (this rank's files only)", .{ e.o.rank, gib(swept) });
}

fn msOf(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / 1e6;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// Every rank's verdict (true: every rank's is); W = 1: its own. An all-gather into the vote buffer made at start-up
/// (nothing allocated), so every rank that reaches it takes part.
fn vote(e: *Engine, ok: bool, why: ?anyerror, what: []const u8) !bool {
    return (try verdict(e, ok, why, what)) == null;
}

/// `vote`, giving the first refusing rank's reason (null: every rank's yes).
fn verdict(e: *Engine, ok: bool, why: ?anyerror, what: []const u8) !?anyerror {
    const mine_err: ?anyerror = if (ok) null else why orelse error.NvmeFailed;
    const g = e.o.group orelse return mine_err;
    const vb = e.nvme.votes orelse return error.NvmeNoVotes; // made at start-up on every rank when the tier is on
    const host = (e.nvme.vote_host orelse return error.NvmeNoVotes).bytes;
    const W: usize = e.o.world;
    const words = std.mem.bytesAsSlice(u32, host[0 .. 4 * W]);
    words[e.o.rank] = if (mine_err) |m| @intFromError(m) else 0;
    try toDev(e, vb.ptr + e.o.rank * 4, host[e.o.rank * 4 ..][0..4]);
    try g.allGather(e.stream, vb.ptr + e.o.rank * 4, vb.ptr, 4);
    try e.stream.synchronize();
    try g.check();
    try toHost(e, host[0 .. 4 * W], vb.ptr);
    var first: ?anyerror = null;
    for (words, 0..) |v, r| if (v != 0) {
        const err = @errorFromInt(@as(@TypeOf(@intFromError(error.NvmeFailed)), @intCast(v)));
        if (first == null) first = err;
        if (e.o.rank == 0) std.log.warn("inkling2: NVMe prompt cache: rank {d} refused the {s} ({s})", .{ r, what, @errorName(err) });
    };
    return first;
}

fn hdrName(h: Header) [16]u8 {
    return @bitCast(h.pad);
}

fn putName(h: *Header, name: [32]u8) void {
    h.pad = @bitCast(name[0..16].*);
}

// ---------------------------------------------------------------- start (every rank)

/// The names a start broadcasts at a time (with every rank's flags for them, they fill the transfer buffer).
fn namesBatch(t: *const Tier, world: u32) u32 {
    const xb = t.xfer orelse return 0;
    if (t.staging == null) return 0;
    return @intCast(@min(@min(xb.len, t.upper().len) / (16 + @as(usize, world)), std.math.maxInt(u32)));
}

/// Rank 0, start-up (after the device cache's arena): every rank's tier ready, then the states every rank holds whole
/// kept (rank 0's index, under the cap) and the rest removed from every disk. A rank that cannot (a directory it
/// cannot write, an unchecked checkpoint, a failed allocation) turns the tier off on every rank, with its reason;
/// serving goes on without it.
pub fn start(x: Leader) !void {
    const e = x.e;
    const c = config() catch |err| {
        std.log.err("inkling2: NVMe prompt cache: {s} (TF_INK2_NVME takes an absolute directory, TF_INK2_NVME_GIB and TF_INK2_NVME_FLOOR_GIB a size in GiB)", .{@errorName(err)});
        return err;
    };
    if (!c.on()) return;
    const t = &e.nvme;
    if (t.unprepared) return; // every rank knows (`prepare`): no step
    var found: std.ArrayList(Found) = .empty;
    defer freeFound(e.gpa, &found);
    var why: ?anyerror = null;
    init(e, &found) catch |err| {
        why = err;
    };
    // rank 0's preparation before its step: the index's room for every state found and the first write
    if (why == null) t.index.list.ensureTotalCapacity(e.gpa, found.items.len + 1) catch |err| {
        why = err;
    };
    const n: u32 = if (why == null) @intCast(found.items.len) else 0;
    const batch = namesBatch(t, e.o.world);
    if (why == null and n > 0 and batch == 0) why = error.NvmeLayout;
    if (e.o.world > 1) try x.send(.{ .kind = @intFromEnum(Kind.nvme_scan), .head_lo = n, .head_n = batch });
    const ok = vote(e, why == null, why, "start") catch |err| return x.stop(err);
    if (!ok) {
        if (why) |w| std.log.warn("inkling2 rank 0: NVMe prompt cache off ({s})", .{@errorName(w)});
        off(e);
        if (e.o.world > 1) try x.idle();
        return;
    }
    reconcile(e, &found, n, batch) catch |err| return x.stop(err);
    // the cap (lowered since the files were written) against every rank's own files: the least recently used go, on
    // every rank, while a rank is over it by states every rank holds; a rank over it by files no eviction frees is
    // reported, and nothing is evicted for it
    while (true) {
        const needs = try admission(x, 0);
        var reclaim: u64 = 0;
        for (needs, 0..) |v, r| {
            if (v == stuck_need) {
                std.log.warn("inkling2: NVMe prompt cache: rank {d}'s files no eviction frees are over the cap alone; it writes nothing until they are gone", .{r});
            } else reclaim = @max(reclaim, v);
        }
        if (reclaim == 0) break;
        const v = t.index.victim() orelse break;
        try dropAt(x, v);
        t.stats.evicted += 1;
    }
    if (e.o.world > 1) try x.idle();
    std.log.info("inkling2: NVMe prompt cache: {d} states ({d:.2} GiB) on every rank's disk at start", .{ t.index.list.items.len, gib(t.index.bytes) });
}

fn off(e: *Engine) void {
    const t = &e.nvme;
    const keep_stats = t.stats;
    const keep_votes = t.votes;
    const keep_host = t.vote_host;
    t.votes = null;
    t.vote_host = null;
    t.deinit(e.gpa);
    t.stats = keep_stats;
    t.votes = keep_votes;
    t.vote_host = keep_host;
}

fn followScan(e: *Engine, h: Header) !void {
    var found: std.ArrayList(Found) = .empty;
    defer freeFound(e.gpa, &found);
    var why: ?anyerror = null;
    init(e, &found) catch |err| {
        why = err;
    };
    const t = &e.nvme;
    const n: u32 = h.head_lo;
    const batch: u32 = h.head_n;
    if (why == null and n > 0) {
        const room = @min(t.xfer.?.len, t.upper().len);
        if (batch == 0 or (16 + @as(usize, e.o.world)) * @as(usize, batch) > room) why = error.NvmeLayout;
    }
    if (!try vote(e, why == null, why, "start")) {
        off(e);
        return;
    }
    try reconcile(e, &found, n, batch);
}

/// Every rank, after a yes at start: rank 0's `n` names in batches through the transfer buffer, every rank's holdings
/// of them gathered; a state every rank holds is kept (rank 0 indexes it, each rank charges its file), the others'
/// files are removed, and so are a follower's files rank 0 does not name. Collectives and this rank's own file work
/// only: nothing allocated.
fn reconcile(e: *Engine, found: *std.ArrayList(Found), n: u32, batch: u32) !void {
    const t = &e.nvme;
    const W: usize = e.o.world;
    const rank: usize = e.o.rank;
    const lead = rank == 0;
    const s = t.upper();
    const flags_at: usize = 16 * @as(usize, batch);
    var removed: u32 = 0;
    var lo: usize = 0;
    while (lo < n) {
        const nb: usize = @min(batch, n - lo);
        const names = s[0 .. 16 * nb];
        if (lead) for (0..nb) |i| @memcpy(names[16 * i ..][0..16], found.items[lo + i].head.name[0..16]);
        const mine = s[flags_at + rank * nb ..][0..nb];
        if (e.o.group) |g| {
            const xb = t.xfer.?;
            if (lead) try toDev(e, xb.ptr, names);
            try g.lib.check(g.lib.api.ncclBroadcast(xb.ptr, xb.ptr, 16 * nb, .u8, 0, g.comm, e.stream.handle), "ncclBroadcast");
            try e.stream.synchronize();
            if (!lead) try toHost(e, names, xb.ptr);
        }
        // this rank's flags: 1 when it holds the state whole (rank 0's are its own scan's)
        for (0..nb) |i| {
            mine[i] = 0;
            if (lead) {
                mine[i] = 1;
                continue;
            }
            for (found.items) |*f| if (std.mem.eql(u8, f.head.name[0..16], names[16 * i ..][0..16])) {
                f.listed = true;
                mine[i] = 1;
                break;
            };
        }
        if (e.o.group) |g| {
            const xb = t.xfer.?;
            try toDev(e, xb.ptr + flags_at + rank * nb, mine);
            try g.allGather(e.stream, xb.ptr + flags_at + rank * nb, xb.ptr + flags_at, nb);
            try e.stream.synchronize();
            try g.check();
            try toHost(e, s[flags_at..][0 .. W * nb], xb.ptr + flags_at);
        }
        for (0..nb) |i| {
            var every = true;
            for (0..W) |r| every = every and s[flags_at + r * nb + i] == 1;
            const stem: [16]u8 = names[16 * i ..][0..16].*;
            // this rank's file of it
            var f: ?*Found = null;
            if (lead) f = &found.items[lo + i] else for (found.items) |*q| if (std.mem.eql(u8, q.head.name[0..16], &stem)) {
                f = q;
                break;
            };
            const mf = f orelse continue; // a follower without it: nothing of its own to keep or remove
            if (every) {
                t.live += mf.bytes;
                if (lead) {
                    t.index.addAssumeCapacity(mf.head.name, mf.tokens, mf.bytes); // room reserved before the step
                    mf.tokens = &.{};
                }
            } else if (removeState(t, stem, mf.bytes, false)) removed += 1;
        }
        lo += nb;
    }
    if (!lead) for (found.items) |f| if (!f.listed) {
        if (removeState(t, stemOf(f.head.name), f.bytes, false)) removed += 1;
    };
    if (removed > 0) std.log.info("inkling2 rank {d}: NVMe prompt cache: {d} states not held whole by every rank removed", .{ e.o.rank, removed });
}

/// This rank's file of `stem` (`bytes`; `live`: a state every rank held) removed; true when it went. One that stays
/// counts against this rank's cap while it is there, and no longer as a live state (no eviction frees it).
fn removeState(t: *Tier, stem: [16]u8, bytes: u64, live: bool) bool {
    var nb: [64]u8 = undefined;
    const gone: Gone = if (fault("unlink", &t.faults.unlinks)) .kept else removeOurs(t.dir_fd, fileName(&nb, stem, t.rank, false), t.rank);
    if (live) t.live -|= bytes;
    switch (gone) {
        .removed => |n| {
            t.used -|= n;
            return true;
        },
        .absent => {
            t.used -|= bytes;
            return false;
        },
        .kept => {
            std.log.warn("inkling2 rank {d}: NVMe prompt cache: a state's file could not be removed; its {d:.2} GiB count against this rank's cap while it is there", .{ t.rank, gib(bytes) });
            return false;
        },
    }
}

// ---------------------------------------------------------------- writing behind (rank 0 decides)

/// Rank 0: the device cache kept `k` (prompt `tokens` [0, k.at)): it waits to be written (nothing when the tier is
/// off, the state is short, has images, or is on disk or waiting already).
pub fn wanted(e: *Engine, k: *snap.Kept, tokens: []const u32, images: usize) void {
    const t = &e.nvme;
    if (!t.on or k.at < min_at or images > 0 or tokens.len < k.at) return;
    _ = enqueue(e, k, tokens) catch {};
}

/// What `enqueue` did.
pub const Queued = enum { queued, on_disk, waiting };

/// `k` waits to be written, whatever its length (tooling calls it directly); nothing when it is on disk or waiting.
pub fn enqueue(e: *Engine, k: *snap.Kept, tokens: []const u32) !Queued {
    const t = &e.nvme;
    if (!t.on) return error.NvmeOff;
    if (tokens.len < k.at) return error.NvmeLayout;
    const name = nameOf(t.identity, k.at, tokens[0..k.at]);
    if (t.index.find(name)) |i| {
        t.index.touch(i);
        return .on_disk;
    }
    // a write of the same state under way may still be given up (its device copy dropped): this one waits behind
    // it, and the begin skips it when the index has it by then
    for (t.queue.items) |w| if (std.mem.eql(u8, &w.name, &name)) return .waiting;
    const owned = try e.gpa.dupe(u32, tokens[0..k.at]);
    t.queue.append(e.gpa, .{ .kept = k, .name = name, .tokens = owned }) catch |err| {
        e.gpa.free(owned);
        return err;
    };
    std.log.info("inkling2: NVMe prompt cache: a state of {d} tokens waits to be written ({d} waiting)", .{ k.at, t.queue.items.len });
    return .queued;
}

/// Tooling: whether this rank's file (or `.part`) of state `name` is in its directory.
pub fn fileThere(e: *const Engine, name: [32]u8, part: bool) bool {
    const t = &e.nvme;
    if (t.dir_fd < 0) return false;
    var nb: [64]u8 = undefined;
    const fd = openLeaf(t.dir_fd, fileName(&nb, stemOf(name), t.rank, part));
    if (fd < 0) return std.c.errno(fd) != .NOENT;
    _ = std.c.close(fd);
    return true;
}

/// Tooling: the index entry of the state of `tokens` [0, at), when it is on disk.
pub fn findAt(e: *const Engine, at: u32, tokens: []const u32) ?usize {
    const t = &e.nvme;
    if (!t.on or tokens.len < at) return null;
    return t.index.find(nameOf(t.identity, at, tokens[0..at]));
}

/// Rank 0: the device cache drops `k` (its pages and extent go back now): not written. A write of it under way only
/// lets go of it here (nothing more is copied from it; no step, no file work while a stream may be waiting); the
/// next idle turn gives the write up on every rank (`step`).
pub fn dropping(x: Leader, k: *snap.Kept) void {
    const t = &x.e.nvme;
    if (!t.on) return;
    var i: usize = 0;
    while (i < t.queue.items.len) {
        if (t.queue.items[i].kept == k) {
            x.e.gpa.free(t.queue.items[i].tokens);
            _ = t.queue.orderedRemove(i);
        } else i += 1;
    }
    if (t.job) |*j| if (j.kept == k) {
        j.kept = null;
        j.stale = true;
        t.stats.skipped += 1;
    };
}

/// Rank 0: work waits (a write under way or waiting, a removal waiting).
pub fn pending(e: *const Engine) bool {
    const t = &e.nvme;
    return t.on and (t.job != null or t.queue.items.len > 0 or t.drops.items.len > 0);
}

/// The file bytes of kept state `k`.
fn bytesOf(e: *const Engine, k: *const snap.Kept) !struct { fixed: u64, npages: u32, payload: u64, file: u64 } {
    const fixed = fixedAt(e, k.at);
    const npages: u32 = @intCast(k.pages.len);
    if (npages != k.at / pages.page_rows) return error.NvmeLayout;
    const p = e.pool.?;
    const payload = fixed + @as(u64, p.regions) * npages * p.place().run;
    return .{ .fixed = fixed, .npages = npages, .payload = payload, .file = fileBytes(k.at, payload) };
}

/// An admission answer: no eviction can make room on this rank.
pub const stuck_need: u64 = std.math.maxInt(u64);

/// This rank's answer for a state of `total` bytes: 0 (it fits under the cap), the bytes evictions must free, or
/// `stuck_need` (its files no eviction frees leave no room).
pub fn needOf(used: u64, live: u64, cap: u64, total: u64) u64 {
    if ((used -| live) + total > cap) return stuck_need;
    return (used + total) -| cap;
}

/// What rank 0 does with every rank's answers: write, evict one state and ask again, or refuse the state (some rank
/// can never make room: nothing is evicted for it).
pub const Admit = enum { write, evict, refuse };

pub fn admit(needs: []const u64) Admit {
    var most: u64 = 0;
    for (needs) |v| {
        if (v == stuck_need) return .refuse;
        most = @max(most, v);
    }
    return if (most > 0) .evict else .write;
}

/// Rank 0: every rank's answer for a state of `total` bytes (an `nvme_admit` step and a gather into the vote
/// buffers: nothing allocated).
fn admission(x: Leader, total: u64) ![]const u64 {
    const e = x.e;
    if (e.o.world > 1) try x.send(.{ .kind = @intFromEnum(Kind.nvme_admit), .pos0 = total });
    return gatherNeeds(e, needOf(e.nvme.used, e.nvme.live, e.nvme.cfg.cap, total)) catch |err| return x.stop(err);
}

/// Every rank's u64 in rank order (W = 1: its own), in the vote buffers' memory.
fn gatherNeeds(e: *Engine, mine: u64) ![]const u64 {
    const t = &e.nvme;
    const W: usize = if (e.o.group != null) e.o.world else 1;
    const host = (t.vote_host orelse {
        t.need_one[0] = mine; // W = 1: no buffers
        return t.need_one[0..1];
    }).bytes;
    const words = std.mem.bytesAsSlice(u64, host[0 .. 8 * W]);
    const r: usize = if (e.o.group != null) e.o.rank else 0;
    words[r] = mine;
    const g = e.o.group orelse return words[0..1];
    const vb = t.votes orelse return error.NvmeNoVotes;
    try toDev(e, vb.ptr + r * 8, host[r * 8 ..][0..8]);
    try g.allGather(e.stream, vb.ptr + r * 8, vb.ptr, 8);
    try e.stream.synchronize();
    try g.check();
    try toHost(e, host[0 .. 8 * W], vb.ptr);
    return words;
}

/// Rank 0, while no stream runs: one unit of the writing (a removal, a write given up, one eviction, a begin, one
/// chunk, or the commit). True when it did one.
pub fn step(x: Leader) !bool {
    const e = x.e;
    const t = &e.nvme;
    if (!t.on) return false;
    const t0 = nowNs(e);
    const did = try oneUnit(x);
    const dt = nowNs(e) - t0;
    // what a request arriving during a unit waits at most (it waits for the engine thread): logged when a unit sets a
    // new longest above 20 ms
    if (did and dt > t.stats.unit_max_ns and dt > 20_000_000) std.log.info("inkling2: NVMe prompt cache: an idle unit took {d:.1} ms (the longest yet)", .{msOf(dt)});
    if (did) t.stats.unit_max_ns = @max(t.stats.unit_max_ns, dt);
    return did;
}

fn oneUnit(x: Leader) !bool {
    const e = x.e;
    const t = &e.nvme;
    if (t.drops.items.len > 0) {
        const d = t.drops.pop().?;
        try sendDrop(x, d.stem, d.bytes);
        return true;
    }
    if (t.job) |*j| {
        if (j.next >= j.payload) {
            try commit(x); // every chunk is copied: the commit reads nothing of the device state
        } else if (j.stale or (j.kept != null and !e.snap.holds(j.off, j.tag, j.at))) {
            try abort(x); // the state left the device cache (or its extent holds another): nothing more is copied
        } else try chunk(x);
        return true;
    }
    if (t.queue.items.len == 0) return false;
    const w = t.queue.items[0];
    const sz = bytesOf(e, w.kept) catch null;
    const on_disk = t.index.find(w.name);
    if (on_disk != null or !e.snap.holds(w.kept.off, w.kept.tag, w.kept.at) or sz == null or sz.?.file > t.cfg.cap) {
        _ = t.queue.orderedRemove(0);
        e.gpa.free(w.tokens);
        if (on_disk) |i| t.index.touch(i) else t.stats.write_fails += 1;
        return true;
    }
    // every rank answers for this state before anything is evicted for it: a rank that can never make room refuses it
    // (no eviction); otherwise one least recently used state goes a turn, on every rank, and the next turn asks again
    switch (admit(try admission(x, sz.?.file))) {
        .refuse => {
            _ = t.queue.orderedRemove(0);
            e.gpa.free(w.tokens);
            t.stats.write_fails += 1;
            t.stats.refused += 1;
            std.log.warn("inkling2: NVMe prompt cache: a state of {d} tokens not written: a rank's files that no eviction frees leave it no room under the cap", .{w.kept.at});
            return true;
        },
        .evict => {
            if (t.index.victim()) |v| {
                try dropAt(x, v);
                t.stats.evicted += 1;
            } else {
                _ = t.queue.orderedRemove(0); // nothing left to evict
                e.gpa.free(w.tokens);
                t.stats.write_fails += 1;
            }
            return true;
        },
        .write => {},
    }
    _ = t.queue.orderedRemove(0);
    const b0 = nowNs(e);
    begin(x, w) catch |err| {
        t.stats.begin_max_ns = @max(t.stats.begin_max_ns, nowNs(e) - b0);
        e.gpa.free(w.tokens);
        t.stats.write_fails += 1;
        if (err != error.NvmeVoted) std.log.warn("inkling2: NVMe prompt cache: a state of {d} tokens not written ({s})", .{ w.kept.at, @errorName(err) });
        return true;
    };
    t.stats.begin_max_ns = @max(t.stats.begin_max_ns, nowNs(e) - b0);
    return true;
}

/// Tooling and a planned stop: every waiting write done now.
pub fn drain(x: Leader) !void {
    while (try step(x)) {}
}

fn begin(x: Leader, w: Wanted) !void {
    const e = x.e;
    const t = &e.nvme;
    const k = w.kept;
    if (!e.snap.holds(k.off, k.tag, k.at)) return error.SnapshotStale;
    const sz = try bytesOf(e, k);
    // everything the step needs, before its header: a failure here is rank 0's alone (no step went out)
    const xn = 4 * (@as(usize, sz.npages) + k.at);
    if (xn > t.xfer.?.len or xn > t.upper().len or headLen(k.at, 0) > t.stage().len / 2) return error.NvmeLayout;
    if (fault("prep", &t.faults.preps)) return error.OutOfMemory;
    try t.index.list.ensureUnusedCapacity(e.gpa, 1); // the commit's room in the index
    const ids = try e.gpa.dupe(u32, k.pages);
    var j: Job = .{ .name = w.name, .at = k.at, .off = k.off, .tag = k.tag, .ids = ids, .plan_fixed = sz.fixed, .payload = sz.payload, .head_len = headLen(k.at, 0), .kept = k, .tokens = w.tokens };
    var h: Header = .{ .kind = @intFromEnum(Kind.nvme_begin), .rows = k.at, .pos0 = k.off, .token = k.tag, .head_lo = sz.npages };
    putName(&h, w.name);
    try x.send(h); // a failure stops every rank (Leader.send)
    // after the header every failure is a vote: the pages' ids, then the tokens, to every rank
    if (e.o.group) |g| {
        const host = std.mem.bytesAsSlice(u32, t.upper()[0..xn]);
        @memcpy(host[0..sz.npages], k.pages);
        @memcpy(host[sz.npages..], w.tokens[0..k.at]);
        const xb = t.xfer.?;
        toDev(e, xb.ptr, t.upper()[0..xn]) catch |err| return x.stop(err);
        g.lib.check(g.lib.api.ncclBroadcast(xb.ptr, xb.ptr, xn, .u8, 0, g.comm, e.stream.handle), "ncclBroadcast") catch |err| return x.stop(err);
        e.stream.synchronize() catch |err| return x.stop(err);
    }
    const why: ?anyerror = if (openJob(e, &j, w.tokens)) null else |err| err;
    const ok = vote(e, why == null, why, "write") catch |err| return x.stop(err);
    if (!ok) {
        abandon(t, &j);
        e.gpa.free(j.ids);
        return error.NvmeVoted; // the tokens: the caller's
    }
    t.job = j;
}

/// Every rank: the job's file made (unnamed until its publication on Linux; else the `.part`, exclusively) and its head
/// and tokens written; its payload's space is reserved a chunk at a time (`writeChunk`).
fn openJob(e: *Engine, j: *Job, tokens: []const u32) !void {
    const t = &e.nvme;
    if (!e.snap.holds(j.off, j.tag, j.at)) return error.SnapshotStale;
    if (!std.mem.eql(u8, &nameOf(t.identity, j.at, tokens), &j.name)) return error.NvmeName;
    const total = j.head_len + alignUp(j.payload);
    var pb: [64]u8 = undefined;
    var qb: [64]u8 = undefined;
    const part = fileName(&pb, stemOf(j.name), t.rank, true);
    const dest = fileName(&qb, stemOf(j.name), t.rank, false);
    // a leftover of a stopped write, a copy no index holds (a stop between ranks' publications): this rank's, by head
    for ([_][:0]const u8{ part, dest }) |n| switch (removeOurs(t.dir_fd, n, t.rank)) {
        .removed => |b| t.used -|= b,
        else => {},
    };
    // this rank's files under the cap (the admission asked before; checked again here, with the file's own bytes)
    if ((t.used -| t.live) + total > t.cfg.cap) return error.NvmeStuck;
    if (t.used + total > t.cfg.cap) return error.NvmeOverCap;
    const free = freeBytes(t.dir_fd) orelse return error.NvmeNoSpace;
    if (free < t.cfg.floor + total) return error.NvmeUnderFloor;
    const made = try create(t.dir_fd, part, true);
    j.fd = made.fd;
    j.direct = made.direct;
    j.unnamed = made.unnamed;
    t.used += total; // its file is there now, whatever follows
    // the head (hashes filled at commit) and the tokens through the staging buffer's lower half
    const s = t.stage();
    var head: Head = .{ .world = e.o.world, .rank = e.o.rank, .at = j.at, .npages = @intCast(j.ids.len), .regions = @intCast(e.pool.?.regions), .run = e.pool.?.place().run, .fixed = j.plan_fixed, .payload = j.payload, .head_len = j.head_len, .identity = t.identity, .name = j.name, .content = t.content };
    head.meta_hash = metaHash(tokens);
    const tb = std.mem.sliceAsBytes(tokens);
    var pos: u64 = 0;
    var done: usize = 0;
    while (pos < j.head_len) {
        const span = @min(s.len / 2, j.head_len - pos);
        @memset(s[0..span], 0);
        const from: usize = if (pos == 0) @sizeOf(Head) else 0;
        const take = @min(span - from, tb.len - done);
        @memcpy(s[from..][0..take], tb[done..][0..take]);
        if (pos == 0) {
            @memcpy(s[0..@sizeOf(Head)], std.mem.asBytes(&head));
            @memcpy(&t.first, s[0..unit]);
        }
        try writeAll(j.fd, s[0..span], pos);
        done += take;
        pos += span;
    }
    j.hash = .init(0);
}

/// Every rank: the job's file let go (a failed or given-up write): its `.part` name removed (an unnamed file has
/// none), then closed; its bytes leave the count when no name of it is left.
fn abandon(t: *Tier, j: *Job) void {
    if (j.fd < 0) return; // nothing made (a refusal before its file)
    var pb: [64]u8 = undefined;
    if (!j.unnamed) _ = std.c.unlinkat(t.dir_fd, fileName(&pb, stemOf(j.name), t.rank, true), 0);
    letGo(t, j);
}

/// The job's file closed; its bytes leave this rank's count when no name of it is left (its link count), else it
/// stays counted (a file this rank could not remove).
fn letGo(t: *Tier, j: *Job) void {
    const links: u64 = if (info(j.fd)) |fi| fi.nlink else 1;
    _ = std.c.close(j.fd);
    j.fd = -1;
    if (links == 0) {
        t.used -|= j.head_len + alignUp(j.payload);
    } else std.log.warn("inkling2 rank {d}: NVMe prompt cache: a given-up write's file could not be removed; it counts against this rank's cap while it is there", .{t.rank});
}

/// One chunk of the payload (`write_chunk` bytes from the job's next) to the file, on every rank.
fn chunk(x: Leader) !void {
    const e = x.e;
    try x.send(.{ .kind = @intFromEnum(Kind.nvme_chunk) });
    writeChunk(e) catch |err| {
        e.nvme.job.?.failed = err;
    };
}

fn writeChunk(e: *Engine) !void {
    const t = &e.nvme;
    const j = &(t.job orelse return error.NvmeNoJob);
    if (j.failed != null) {
        j.next = @min(j.payload, j.next + write_chunk);
        return;
    }
    const lo = j.next;
    const hi = @min(j.payload, lo + write_chunk);
    j.next = hi;
    const plan: Plan = .{ .fixed = j.plan_fixed, .regions = @intCast(e.pool.?.regions), .run = e.pool.?.place().run, .ids = j.ids, .pl = e.pool.?.place() };
    try plan.pieces(e.gpa, lo, hi, &t.pieces);
    const s = t.stage();
    const arena = (e.snap.buf orelse return error.NoArena).ptr;
    for (t.pieces.items) |pc| {
        const src = if (pc.fixed) arena + j.off + pc.src else pc.src;
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(s[pc.off - lo ..].ptr, src, pc.len, e.stream.handle), "cuMemcpyDtoHAsync");
    }
    try e.stream.synchronize();
    const n = hi - lo;
    j.hash.update(s[0..n]);
    const span = alignUp(n);
    @memset(s[n..span], 0);
    try reserve(j.fd, j.head_len + lo, span); // this chunk's space only: no unit reserves a whole file
    try writeAll(j.fd, s[0..span], j.head_len + lo);
    if (!j.direct) try syncData(j.fd); // buffered: this chunk's pages now, so the commit's sync stays short
}

/// The file sealed, published and its directory synced on every rank, then voted: in place on every rank, or on none
/// (a no takes every rank's publication back); rank 0 indexes it after a yes (room reserved at the begin).
fn commit(x: Leader) !void {
    const e = x.e;
    const t = &e.nvme;
    const t0 = nowNs(e);
    try x.send(.{ .kind = @intFromEnum(Kind.nvme_commit) });
    const ok = finish(e) catch |err| return x.stop(err);
    t.stats.commit_max_ns = @max(t.stats.commit_max_ns, nowNs(e) - t0);
    const j = t.job.?;
    t.job = null;
    e.gpa.free(j.ids);
    if (!ok) {
        e.gpa.free(j.tokens);
        t.stats.write_fails += 1;
        return;
    }
    const bytes = fileBytes(j.at, j.payload);
    t.index.addAssumeCapacity(j.name, j.tokens, bytes);
    t.stats.writes += 1;
    t.stats.write_bytes += bytes;
    std.log.info("inkling2: NVMe prompt cache: a state of {d} tokens written on every rank ({d:.2} GiB a rank; {d} states, {d:.2} GiB of {d:.2}; commit {d:.1} ms)", .{ j.at, gib(bytes), t.index.list.items.len, gib(t.index.bytes), gib(t.cfg.cap), msOf(nowNs(e) - t0) });
}

/// Every rank: the job sealed, published and synced, then voted on; true (on every rank) when it is in place on every
/// rank. A no removes this rank's file of it, published or not.
fn finish(e: *Engine) !bool {
    const t = &e.nvme;
    const j = &(t.job orelse return error.NvmeNoJob);
    var pb: [64]u8 = undefined;
    var qb: [64]u8 = undefined;
    const part = fileName(&pb, stemOf(j.name), t.rank, true);
    const dest = fileName(&qb, stemOf(j.name), t.rank, false);
    var why: ?anyerror = j.failed;
    if (why == null and fault("write", &t.faults.writes)) why = error.InducedFault;
    if (why == null) seal(t, j) catch |err| {
        why = err;
    };
    // published while still open (an unnamed file is linked by its descriptor)
    var published = false;
    if (why == null) {
        const r = if (fault("publish", &t.faults.publishes)) error.InducedFault else if (j.unnamed) publishFd(t.dir_fd, j.fd, dest) else publish(t.dir_fd, part, dest, fault("partunlink", &t.faults.part_unlinks));
        if (r) |_| {
            published = true;
            syncDir(t.dir_fd) catch |err| {
                why = err;
            };
        } else |err| why = err;
    }
    const ok = try vote(e, why == null, why, "write");
    if (ok) {
        _ = std.c.close(j.fd);
        j.fd = -1;
        t.live += j.head_len + alignUp(j.payload);
    } else {
        // taken back: every name of this rank's file (the one it just published, a `.part`), then let go
        if (published) _ = std.c.unlinkat(t.dir_fd, dest, 0);
        if (!j.unnamed) _ = std.c.unlinkat(t.dir_fd, part, 0);
        if (published) syncDir(t.dir_fd) catch {};
        letGo(t, j);
    }
    if (e.o.rank != 0) {
        e.gpa.free(j.ids);
        t.job = null;
    }
    return ok;
}

fn seal(t: *Tier, j: *Job) !void {
    var head = std.mem.bytesToValue(Head, t.first[0..@sizeOf(Head)]);
    head.payload_hash = j.hash.final();
    head.head_hash = head.hash();
    @memcpy(t.first[0..@sizeOf(Head)], std.mem.asBytes(&head));
    const s = t.stage(); // free: every chunk is written
    @memcpy(s[0..unit], &t.first);
    try writeAll(j.fd, s[0..unit], 0);
    try syncData(j.fd);
}

/// Rank 0, in an idle turn: the write under way given up on every rank.
fn abort(x: Leader) !void {
    const e = x.e;
    const t = &e.nvme;
    var j = t.job orelse return;
    t.job = null;
    try x.send(.{ .kind = @intFromEnum(Kind.nvme_abort) });
    t.stats.aborts += 1;
    abandon(t, &j);
    e.gpa.free(j.ids);
    e.gpa.free(j.tokens);
}

/// Rank 0: index entry `i` removed from every rank's disk.
fn dropAt(x: Leader, i: usize) !void {
    const t = &x.e.nvme;
    const en = t.index.list.items[i];
    const stem = stemOf(en.name);
    const bytes = en.bytes;
    _ = t.index.remove(x.e.gpa, i);
    try sendDrop(x, stem, bytes);
}

/// Rank 0: state `stem`'s files removed on every rank (its index entry gone already).
fn sendDrop(x: Leader, stem: [16]u8, bytes: u64) !void {
    var name: [32]u8 = @splat(0);
    @memcpy(name[0..16], &stem);
    var h: Header = .{ .kind = @intFromEnum(Kind.nvme_drop), .pos0 = bytes };
    putName(&h, name);
    try x.send(h);
    _ = removeState(&x.e.nvme, stem, bytes, true);
}

// ---------------------------------------------------------------- reading (rank 0 decides)

/// Rank 0: the disk state to read for `prompt`, when it holds at least `min_gain` tokens past `from` (the device
/// cache's resume point): its index entry.
pub fn pick(e: *const Engine, prompt: []const u32, from: usize) ?usize {
    const t = &e.nvme;
    if (!t.on) return null;
    return t.index.longest(prompt, from + min_gain);
}

pub fn entryAt(e: *const Engine, i: usize) u32 {
    return e.nvme.index.list.items[i].at;
}

/// Rank 0: index entry `i` read into slot `slot` on every rank (its pages are the stream's, opened for the prompt):
/// true when every rank's read is whole and checked, and the slot stands at the state's position; false (on every
/// rank) otherwise: the slot's pass starts at 0 on the same pages, and the state is removed from every disk. A write
/// under way waits (the staging buffer holds nothing of it between its units).
/// `poison_byte` (tests): every rank sets the slot's every byte to it first (cuda_snap.zig poison).
pub fn load(x: Leader, slot: u32, i: usize, poison_byte: ?u8) !bool {
    const e = x.e;
    const t = &e.nvme;
    const en = t.index.list.items[i];
    const st = try e.bind(slot);
    if (!st.paged() or st.mapped * pages.page_rows <= en.at) return error.NvmePages;
    var h: Header = .{ .kind = @intFromEnum(Kind.nvme_load), .slot = slot, .rows = en.at, .head_lo = @intCast(en.at / pages.page_rows), .head_n = if (poison_byte) |b| poison_flag | b else 0 };
    putName(&h, en.name);
    const t0 = nowNs(e);
    try x.send(h);
    var why: ?anyerror = null;
    const bytes = readInto(e, h, en.tokens) catch |err| blk: {
        why = err;
        break :blk 0;
    };
    const ok = vote(e, why == null, why, "read") catch |err| return x.stop(err);
    if (ok) {
        t.index.touch(i);
        t.stats.loads += 1;
        t.stats.load_bytes += bytes;
        const ns = nowNs(e) - t0;
        t.stats.load_ns += ns;
        std.log.info("inkling2: NVMe prompt cache: a state of {d} tokens read on every rank in {d:.1} ms ({d:.2} GiB a rank, {d:.2} GB/s)", .{ en.at, msOf(ns), gib(bytes), @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(@max(ns, 1))) });
        return true;
    }
    t.stats.load_fails += 1;
    std.log.warn("inkling2: NVMe prompt cache: a state of {d} tokens refused by a rank; the prompt fills cold and the state is removed at the next idle turn", .{en.at});
    // out of the index now; its files go in an idle turn (no file work while the stream waits), or now if the list
    // cannot take it
    const stem = stemOf(en.name);
    if (t.drops.append(e.gpa, .{ .stem = stem, .bytes = en.bytes })) |_| {
        _ = t.index.remove(e.gpa, i);
    } else |_| try dropAt(x, i);
    return false;
}

/// A load's test flag (Header.head_n): poison the slot with the low byte first.
const poison_flag: u32 = 0x100;

/// Every rank: the file of the step's state read into the slot and its fixed part copied in; its bytes.
fn readInto(e: *Engine, h: Header, rank0_tokens: ?[]const u32) !u64 {
    const t = &e.nvme;
    if (!t.on) return error.NvmeOff;
    const at: u32 = h.rows;
    const st = try e.bind(h.slot);
    const npages: usize = at / pages.page_rows;
    if (h.head_lo != npages or !st.paged() or st.mapped < npages + 1) return error.NvmePages;
    if (h.head_n & poison_flag != 0) try snap.poison(e, h.slot, @truncate(h.head_n), 0);
    var nb: [64]u8 = undefined;
    const name = fileName(&nb, hdrName(h), e.o.rank, false);
    const s = t.stage();
    // the head and tokens checked (identity, drafter bytes, rank, layout, the position, the name against the step's)
    const t0 = nowNs(e);
    const got = try readHead(e.gpa, t.dir_fd, name, want(e, t, at), s);
    defer e.gpa.free(got.tokens);
    if (!std.mem.eql(u8, got.head.name[0..16], &hdrName(h))) return error.NvmeName;
    if (rank0_tokens) |tk| if (!std.mem.eql(u32, tk, got.tokens)) return error.NvmeName;
    const fd = try openRead(t.dir_fd, name);
    defer _ = std.c.close(fd);
    const hd = got.head;
    const plan: Plan = .{ .fixed = hd.fixed, .regions = hd.regions, .run = hd.run, .ids = st.ids[0..npages], .pl = e.pool.?.place() };
    var hash: std.hash.Wyhash = .init(0);
    var lo: u64 = 0;
    var ns: [4]u64 = .{ nowNs(e) - t0, 0, 0, 0 }; // head, reads, hash, copies
    while (lo < hd.payload) {
        const hi = @min(hd.payload, lo + s.len);
        const n = hi - lo;
        const r0 = nowNs(e);
        try readAll(fd, s[0..alignUp(n)], hd.head_len + lo);
        if (lo == 0 and fault("flip", &t.faults.flips)) s[n / 2] ^= 0x10;
        const r1 = nowNs(e);
        hash.update(s[0..n]);
        const r2 = nowNs(e);
        ns[1] += r1 - r0;
        ns[2] += r2 - r1;
        defer ns[3] += nowNs(e) - r2;
        try plan.pieces(e.gpa, lo, hi, &t.pieces);
        for (t.pieces.items) |pc| {
            if (pc.fixed) continue; // the fixed part is copied in by the walk below (it is whole in the first read)
            try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(pc.src, s[pc.off - lo ..].ptr, pc.len, e.stream.handle), "cuMemcpyHtoDAsync");
        }
        if (lo == 0) {
            if (hd.fixed > n) return error.NvmeLayout;
            // the tail rows land in the slot's pages past the state's (the walk's own)
            try snap.copyInFrom(e, h.slot, at, t.staging_dev);
        }
        try e.stream.synchronize();
        lo = hi;
    }
    if (hash.final() != hd.payload_hash) return error.NvmePayload;
    if (fault("load", &t.faults.loads)) return error.InducedFault;
    std.log.info("inkling2 rank {d}: NVMe prompt cache: read {d:.1} MiB: head {d:.1} ms, disk reads {d:.1} ms ({d:.2} GB/s), hash {d:.1} ms, copies {d:.1} ms", .{ e.o.rank, @as(f64, @floatFromInt(got.bytes)) / (1 << 20), msOf(ns[0]), msOf(ns[1]), @as(f64, @floatFromInt(hd.payload)) / @as(f64, @floatFromInt(@max(ns[1], 1))), msOf(ns[2]), msOf(ns[3]) });
    return got.bytes;
}

// ---------------------------------------------------------------- the other ranks

/// A rank other than 0: the tier's steps.
pub fn follow(e: *Engine, h: Header) !void {
    const t = &e.nvme;
    switch (std.enums.fromInt(Kind, h.kind) orelse return error.BadStep) {
        .nvme_scan => try followScan(e, h),
        .nvme_begin => {
            const g = e.o.group orelse return error.NoGroup;
            // the ranks agreed the tier is on (a start that failed turned it off everywhere) and the transfer and
            // staging sizes follow agreed settings: a rank without them cannot take part, and stops
            const xb = t.xfer orelse return error.BadStep;
            const npages: usize = h.head_lo;
            const xn = 4 * (npages + h.rows);
            if (!t.on or xn > xb.len or xn > t.upper().len or headLen(h.rows, 0) > t.stage().len / 2) return error.BadStep;
            // this rank's preparation: reported in the vote, after the broadcast it takes part in whatever it gave
            var why: ?anyerror = null;
            const ids: ?[]u32 = if (fault("prep", &t.faults.preps)) null else e.gpa.alloc(u32, npages) catch null;
            if (ids == null) why = error.OutOfMemory;
            try g.lib.check(g.lib.api.ncclBroadcast(xb.ptr, xb.ptr, xn, .u8, 0, g.comm, e.stream.handle), "ncclBroadcast");
            try e.stream.synchronize();
            const host = std.mem.bytesAsSlice(u32, t.upper()[0..xn]);
            try toHost(e, t.upper()[0..xn], xb.ptr);
            if (ids) |d| @memcpy(d, host[0..npages]);
            const tokens = host[npages..];
            const fixed = fixedAt(e, h.rows);
            const p = e.pool.?;
            var j: Job = .{ .name = nameOf(t.identity, h.rows, tokens), .at = h.rows, .off = h.pos0, .tag = h.token, .ids = ids orelse &.{}, .plan_fixed = fixed, .payload = fixed + @as(u64, p.regions) * npages * p.place().run, .head_len = headLen(h.rows, 0) };
            if (why == null and !std.mem.eql(u8, j.name[0..16], &hdrName(h))) why = error.NvmeName;
            if (why == null) openJob(e, &j, tokens) catch |err| {
                why = err;
            };
            if (!try vote(e, why == null, why, "write")) {
                abandon(t, &j);
                if (ids) |d| e.gpa.free(d);
                return;
            }
            t.job = j;
        },
        .nvme_admit => _ = try gatherNeeds(e, needOf(t.used, t.live, t.cfg.cap, h.pos0)),
        .nvme_chunk => writeChunk(e) catch |err| {
            if (t.job) |*j| j.failed = err;
        },
        .nvme_commit => _ = try finish(e),
        .nvme_abort => if (t.job) |*j| {
            abandon(t, j);
            e.gpa.free(j.ids);
            t.job = null;
        },
        .nvme_drop => if (t.on) {
            _ = removeState(t, hdrName(h), h.pos0, true);
        },
        .nvme_load => {
            var why: ?anyerror = null;
            _ = readInto(e, h, null) catch |err| {
                why = err;
            };
            _ = try vote(e, why == null, why, "read");
        },
        else => return error.BadStep,
    }
}

test {
    _ = @import("cuda_nvme_test.zig");
}
