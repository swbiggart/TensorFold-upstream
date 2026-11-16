//! TF_INK2_ROCE_MULTI=1 at three ranks or more: the layer exchanges' small all-gathers on fabric/tp's multirank RoCE
//! wire instead of NCCL (borrowed as the GLM family borrows it, glm/cuda_roce_multi.zig). The same bytes land in the
//! same rank-order slots, so the rank-order fp32 sums that follow are unchanged; gathers past the wire's ceiling
//! (prompt chunks) stay on NCCL. On the four-Spark switchless ring set TF_TP_ROCE_RELAY=1 with one HCA to each
//! neighbour (TF_TP_ROCE_HCA=rocep1s0f0:3,rocep1s0f1:3), TF_TP_KERNEL_IMAGE (tp_mailbox.fatbin), TF_TP_MASTER (rank
//! 0's LAN address) and a free TF_TP_PORT (its plan link uses TF_TP_PORT + 7).
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const tp = @import("tp");

const Group = cuda.tp_group.Group;

/// Every transport beside NCCL, once the group is open: first one unconditional vote of every rank on the transport
/// settings (agreeTransport), then upstream's RoCE gather (TF_ROCE) and the mesh wire (TF_INK2_ROCE_MULTI), entered
/// by every rank alike. A rank whose settings differ, or that cannot read its own (the kernel image), stops every
/// rank within the vote, naming the setting and the rank; any other failure here is told to the other ranks by name
/// on the liveness lines (when open), so none is left waiting in a collective or a bootstrap. The mesh wire's
/// bootstrap takes the settings and the kernel image's bytes the vote compared (Mesh), never a second read.
pub fn openAll(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group) !void {
    openAgreed(gpa, io, d, g) catch |err| {
        switch (err) {
            // every rank reached the same verdict in the vote and stops on its own; none needs telling
            error.TransportSettingsDiffer, error.TransportNotReady => cuda.tp_watch.ended(),
            else => cuda.tp_watch.fail(@errorName(err)),
        }
        return err;
    };
}

fn openAgreed(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group) !void {
    var mesh = try agreeTransport(gpa, io, d, g);
    defer if (mesh) |*m| m.deinit(gpa);
    try openFast(gpa, io, d, g);
    try open(gpa, io, d, g, wanted(), if (mesh) |*m| m else null);
}

/// What every rank's vote agreed for the mesh wire, kept for its bootstrap: the fabric/tp settings as parsed for the
/// ballot and the kernel image's bytes whose sha256 was compared. Nothing is read again after the vote.
pub const Mesh = struct {
    cfg: tp.Config,
    image: []align(16) u8,

    fn deinit(m: *Mesh, gpa: std.mem.Allocator) void {
        gpa.free(m.image);
    }
};

/// The settings a transport branches on, each compared as its effective value (empty: not in use at this world or
/// with the wire off). TF_TP_ROCE_HCA, TF_TP_ROCE_CPU and TF_TP_ROCE_FAST may differ between hosts and are not here.
const Setting = enum(u8) { roce, roce_max_kb, multi, master, port, mailbox_kb, relay, wire, fallback, tp_rest, image };

fn settingName(s: Setting) []const u8 {
    return switch (s) {
        .roce => "TF_ROCE",
        .roce_max_kb => "TF_ROCE_MAX_KB",
        .multi => "TF_INK2_ROCE_MULTI",
        .master => "TF_TP_MASTER",
        .port => "TF_TP_PORT / TF_TP_PLAN_PORT",
        .mailbox_kb => "TF_TP_MAILBOX_MAX_KB",
        .relay => "TF_TP_ROCE_RELAY / TF_TP_ROCE_RELAY_MAX_KB",
        .wire => "TF_TP_ROCE_WIRE",
        .fallback => "TF_TP_ROCE_FALLBACK",
        .tp_rest => "the other fabric/tp settings (TF_COMM_BACKEND, TF_TP_ROCE_CHUNKED_KB, TF_TP_FAILFAST)",
        .image => "TF_TP_KERNEL_IMAGE's contents (sha256)",
    };
}

const n_settings = @intFromEnum(Setting.image) + 1; // image is the last
const value_len = 40; // a value as shown in the lines (cut to fit)
const full_len = 1024; // the longest value compared (a DNS name is at most 255 bytes)

/// One rank's ballot: its effective transport settings, or why it cannot read them (ready 0). Fixed size: it crosses
/// NCCL as bytes in one all-gather. Each setting is compared by the sha256 of its whole value (`full`); `values`
/// holds it cut to fit, for the lines only.
const Vote = extern struct {
    rank: u32,
    world: u32,
    ready: u32 = 1,
    why: [120]u8 = @splat(0),
    values: [n_settings][value_len]u8 = @splat(@splat(0)),
    full: [n_settings][32]u8 = @splat(@splat(0)),

    fn set(v: *Vote, s: Setting, comptime fmt: []const u8, args: anytype) void {
        const i = @intFromEnum(s);
        var buf: [full_len]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, fmt, args) catch {
            // longer than any valid value: not compared by a prefix, refused (every rank sees which)
            v.refuse("{s} is longer than {d} bytes", .{ settingName(s), full_len });
            return;
        };
        std.crypto.hash.sha2.Sha256.hash(text, &v.full[i], .{});
        const out = &v.values[i];
        @memset(out, 0);
        const n = @min(text.len, value_len - 1);
        @memcpy(out[0..n], text[0..n]);
        if (text.len > n) @memcpy(out[value_len - 4 .. value_len - 1], "...");
    }

    fn value(v: *const Vote, s: Setting) []const u8 {
        return std.mem.sliceTo(&v.values[@intFromEnum(s)], 0);
    }

    /// Whether `v` and `w` hold the same whole value of `s` (not its shown, possibly cut, form).
    fn same(v: *const Vote, w: *const Vote, s: Setting) bool {
        return std.mem.eql(u8, &v.full[@intFromEnum(s)], &w.full[@intFromEnum(s)]);
    }

    fn refuse(v: *Vote, comptime fmt: []const u8, args: anytype) void {
        if (v.ready == 0) return; // the first reason stays
        v.ready = 0;
        _ = std.fmt.bufPrint(v.why[0 .. v.why.len - 1], fmt, args) catch {
            @memcpy(v.why[v.why.len - 4 .. v.why.len - 1], "...");
        };
    }

    fn reason(v: *const Vote) []const u8 {
        return std.mem.sliceTo(&v.why, 0);
    }
};

const Get = *const fn ([]const u8) ?[]const u8;

fn envGet(name: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    return if (std.c.getenv(buf[0..name.len :0])) |v| std.mem.span(v) else null;
}

/// The environment as fabric/tp reads its own settings (tp.Config.fromEnvFor): an empty value is unset.
fn tpEnvGet(name: []const u8) ?[]const u8 {
    const v = envGet(name) orelse return null;
    return if (v.len == 0) null else v;
}

/// openFast's branch: TF_ROCE starting with 1 at two ranks or more.
fn roceOn(get: Get, world: usize) bool {
    const v = get("TF_ROCE") orelse return false;
    return world >= 2 and v.len > 0 and v[0] == '1';
}

/// open's branch: TF_INK2_ROCE_MULTI starting with 1, three ranks or more, on Linux.
fn multiOn(get: Get, world: usize, linux: bool) bool {
    const v = get("TF_INK2_ROCE_MULTI") orelse return false;
    return linux and world >= 3 and v.len > 0 and v[0] == '1';
}

/// The kernel image as this rank found it (read only when the mesh wire is on).
const Image = union(enum) { unset, failed: []const u8, empty, digest: [32]u8 };

/// The image's state for the ballot and, when it was read, its bytes (the caller's): the very bytes hashed.
const ImageRead = struct { image: Image, bytes: ?[]align(16) u8 = null };

fn readImage(gpa: std.mem.Allocator, io: std.Io, get: Get) ImageRead {
    const path = get("TF_TP_KERNEL_IMAGE") orelse return .{ .image = .unset };
    // aligned as the wire's module load takes them (fabric/tp roce_multi)
    const bytes = std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(1 << 26), .@"16", null) catch |err| return .{ .image = .{ .failed = @errorName(err) } };
    if (bytes.len == 0) {
        gpa.free(bytes);
        return .{ .image = .empty };
    }
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return .{ .image = .{ .digest = h }, .bytes = bytes };
}

/// This rank's ballot from `get` (the environment), its place, and its kernel image when the mesh wire is on.
fn ballot(get: Get, world: usize, rank: usize, linux: bool, image: Image) Vote {
    var cfg: ?tp.Config = null;
    return ballotParsed(get, get, world, rank, linux, image, &cfg);
}

/// `ballot`, with the fabric/tp settings read through `tp_get` and kept in `cfg` (when the mesh wire is on and they
/// parse): the settings voted on are the ones the bootstrap uses.
fn ballotParsed(get: Get, tp_get: Get, world: usize, rank: usize, linux: bool, image: Image, cfg_out: *?tp.Config) Vote {
    var v: Vote = .{ .rank = @intCast(rank), .world = @intCast(world) };
    if (roceOn(get, world)) {
        v.set(.roce, "{s}", .{get("TF_ROCE").?});
        const kb = if (get("TF_ROCE_MAX_KB")) |k| std.fmt.parseInt(usize, k, 10) catch blk: {
            v.refuse("TF_ROCE_MAX_KB={s} is not a number", .{k});
            break :blk 0;
        } else 512;
        v.set(.roce_max_kb, "{d}", .{kb});
    } else v.set(.roce, "off", .{});
    if (!multiOn(get, world, linux)) {
        v.set(.multi, "off", .{});
        return v;
    }
    v.set(.multi, "on", .{});
    const cfg = tp.Config.parseFor(tp_get, @intCast(world), @intCast(rank)) catch {
        v.refuse("a fabric/tp setting (TF_TP_*) is invalid: its warning above names it", .{});
        return v;
    };
    cfg_out.* = cfg;
    v.set(.master, "{s}", .{cfg.master});
    v.set(.port, "{d}/{d}", .{ cfg.port, cfg.link_port });
    v.set(.mailbox_kb, "{d}", .{cfg.mailbox_max_bytes / 1024});
    if (cfg.roce_relay) v.set(.relay, "on, {d} KiB", .{cfg.roce_relay_max_bytes / 1024}) else v.set(.relay, "off", .{});
    v.set(.wire, "{t}", .{cfg.roce_wire});
    v.set(.fallback, "{t}", .{cfg.roce_fallback});
    const rest = cfg.digest();
    v.set(.tp_rest, "{x}", .{&rest});
    switch (image) {
        .unset => v.refuse("TF_TP_KERNEL_IMAGE is not set (TF_INK2_ROCE_MULTI=1 needs tp_mailbox.fatbin)", .{}),
        .failed => |e| v.refuse("TF_TP_KERNEL_IMAGE={s}: {s}", .{ get("TF_TP_KERNEL_IMAGE").?, e }),
        .empty => v.refuse("TF_TP_KERNEL_IMAGE={s} is empty", .{get("TF_TP_KERNEL_IMAGE").?}),
        .digest => |h| v.set(.image, "{x}", .{&h}),
    }
    return v;
}

const Verdict = union(enum) {
    agreed,
    not_ready: usize, // the lowest rank that could not read its settings
    differs: struct { rank: usize, setting: ?Setting }, // the first rank and setting unlike `mine` (null: place)
};

/// The same verdict on every rank: every rank reads the same ballots.
fn judge(all: []const Vote, mine: *const Vote) Verdict {
    for (all, 0..) |v, r| if (v.ready != 1) return .{ .not_ready = r };
    for (all, 0..) |v, r| {
        if (v.rank != r or v.world != all.len) return .{ .differs = .{ .rank = r, .setting = null } };
        for (std.enums.values(Setting)) |s| if (!v.same(mine, s)) return .{ .differs = .{ .rank = r, .setting = s } };
    }
    return .agreed;
}

/// Before any transport: every rank's transport ballot over the group's NCCL (the one collective every rank enters
/// unconditionally), then one verdict. Differing settings or a rank not ready stop every rank here, naming them.
/// With the mesh wire on, returns the agreed settings and the image bytes hashed for the ballot (the caller's).
fn agreeTransport(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group) !?Mesh {
    if (g.world < 2) return null;
    const linux = comptime builtin.os.tag == .linux;
    var read: ImageRead = if (multiOn(&envGet, g.world, linux)) readImage(gpa, io, &envGet) else .{ .image = .unset };
    defer if (read.bytes) |b| gpa.free(b);
    var cfg: ?tp.Config = null;
    const mine = ballotParsed(&envGet, &tpEnvGet, g.world, g.rank, linux, read.image, &cfg);
    const all = try gpa.alloc(Vote, g.world);
    defer gpa.free(all);
    {
        var s = try cuda.Stream.init(d, true);
        defer s.deinit();
        try g.gather(Vote, d, s, mine, all);
    }
    switch (judge(all, &mine)) {
        .agreed => {},
        .not_ready => {
            for (all, 0..) |v, r| if (v.ready != 1) std.log.err("inkling2 rank {d}: rank {d} cannot open its transports: {s}; every rank stops", .{ g.rank, r, v.reason() });
            return error.TransportNotReady;
        },
        .differs => {
            for (all, 0..) |v, r| {
                if (v.rank != r or v.world != g.world) std.log.err("inkling2 rank {d}: rank {d} reports rank {d} of {d}", .{ g.rank, r, v.rank, v.world });
                for (std.enums.values(Setting)) |s| if (!v.same(&mine, s))
                    std.log.err("inkling2 rank {d}: rank {d} differs from this rank in {s} (rank {d}: \"{s}\", this rank: \"{s}\"); every rank stops", .{ g.rank, r, settingName(s), r, v.value(s), mine.value(s) });
            }
            return error.TransportSettingsDiffer;
        },
    }
    if (g.rank == 0) std.log.info("inkling2: transport settings agreed on {d} ranks: TF_ROCE {s}, TF_INK2_ROCE_MULTI {s}{s}{s}{s}{s}{s}", .{ g.world, mine.value(.roce), mine.value(.multi), if (mine.value(.mailbox_kb).len > 0) " (TF_TP_MAILBOX_MAX_KB " else "", mine.value(.mailbox_kb), if (mine.value(.image).len > 0) ", kernel image sha256 " else "", mine.value(.image), if (mine.value(.mailbox_kb).len > 0) ")" else "" });
    return keep(&mine, cfg, &read.bytes);
}

/// The mesh wire's part of an agreed vote: its settings and the image bytes (moved out of `bytes`), or null when the
/// wire is off. An agreed ballot with the wire on was ready, so both are there.
fn keep(mine: *const Vote, cfg: ?tp.Config, bytes: *?[]align(16) u8) ?Mesh {
    if (!std.mem.eql(u8, mine.value(.multi), "on")) return null;
    const c = cfg orelse return null;
    const image = bytes.* orelse return null;
    bytes.* = null;
    return .{ .cfg = c, .image = image };
}

/// TF_INK2_ROCE_MULTI=1: every rank must set it alike (it is in the run's settings; the wire's own vote also agrees).
pub fn wanted() bool {
    const v = std.c.getenv("TF_INK2_ROCE_MULTI") orelse return false;
    return v[0] == '1';
}

/// TF_ROCE=1 (switched fabric, every rank reaching every other on the same HCA index): upstream's RoCE small-gather
/// wire (cuda/roce.zig, docs/roce-gather.md) carries gathers of up to TF_ROCE_MAX_KB (default 512) a rank; its connect
/// probe refuses a ring without a switch and every rank keeps NCCL. Off unless set, at any world size.
pub fn openFast(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group) !void {
    const v = std.c.getenv("TF_ROCE") orelse return;
    if (v[0] != '1' or g.world < 2) return;
    const kb = if (std.c.getenv("TF_ROCE_MAX_KB")) |k| try std.fmt.parseInt(usize, std.mem.span(k), 10) else 512;
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    try g.openRoce(gpa, io, d, s, kb * 1024);
    if (g.rank == 0) std.log.info("TF_ROCE=1: gathers up to {d} KiB a rank on {s}", .{ kb, if (g.fast != null) "the RoCE wire" else "NCCL (the wire did not come up)" });
}

/// Every rank opens the wire and lends it to `g` once every probe matched NCCL; one rank failing leaves all on NCCL.
/// `mesh` is what the transport vote agreed (agreeTransport): the settings and kernel image used here, as compared.
pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group, on: bool, mesh: ?*const Mesh) !void {
    if (comptime builtin.os.tag != .linux) return; // fabric/tp's sockets and verbs
    if (!on) return;
    if (g.world < 3) {
        if (g.rank == 0) std.log.info("TF_INK2_ROCE_MULTI=1 takes three ranks or more; {d} keep their own route", .{g.world});
        return;
    }
    // every rank's vote agreed the wire on, so every rank holds its settings and image: no rank enters the bootstrap
    // alone
    const m = mesh orelse return error.TransportNotAgreed;
    _ = io; // nothing is read here: the vote read it
    var stage: Stage = .local;
    const w = borrow(gpa, d, g, m, &stage) catch |err| {
        // TF_TP_ROCE_FALLBACK=error refuses rather than run on NCCL, as does a failure the others may not share
        if (m.cfg.roce_fallback == .@"error" or !fallsBack(err, stage)) return err;
        std.log.warn("rank {d}: TF_INK2_ROCE_MULTI=1, but the wire did not open ({s}); NCCL keeps the small gathers", .{ g.rank, @errorName(err) });
        return;
    };
    g.multi = .{ .ctx = w, .fits = fits, .gather = gather, .check = check, .abort = abort, .deinit = deinit };
}

/// How far a rank's borrow got: `local` until its bootstrap is open (every rank connected), `bootstrap` while every
/// step is voted on the bootstrap, `nccl` once its probe's NCCL gather is enqueued.
const Stage = enum { local, bootstrap, nccl };

/// Whether NCCL keeps the small gathers after `err` at `stage`. Never from `local`: the other ranks are entering the
/// bootstrap, so this rank's failure goes to them (openAll: by name on the liveness lines) and every rank stops,
/// instead of this rank running on while they wait out the bootstrap's timeout. Always within the bootstrap (each
/// step every rank's or none). After the probe's gather is enqueued, only on every rank's verdict (a mismatch).
fn fallsBack(err: anyerror, stage: Stage) bool {
    return switch (stage) {
        .local => false,
        .bootstrap => true,
        .nccl => err == error.ProbeMismatch,
    };
}

/// fabric/tp's bootstrap, ports, wire and kernel, and a probe against the group's NCCL: each step every rank's or none.
fn borrow(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *const Group, m: *const Mesh, stage: *Stage) !*Wire {
    const cfg = m.cfg;
    var b = try tp.Bootstrap.open(cfg);
    defer b.close(); // the wire is connected by the end of init
    stage.* = .bootstrap;
    try b.agree(&cfg.digest(), "fabric/tp settings (TF_TP_MAILBOX_MAX_KB, TF_TP_ROCE_RELAY_MAX_KB, TF_TP_ROCE_WIRE, TF_TP_ROCE_FALLBACK)");
    var specs: [tp.roce_verbs.max_hcas]tp.roce_verbs.HcaSpec = undefined;
    var picked = true;
    const n = if (cfg.roce_wire == .verbs) tp.roce_pick.resolve(cfg, &b, &specs) catch blk: {
        picked = false;
        break :blk 0;
    } else 0;
    if (!try everyRank(&b, picked)) return error.RoceHcaSelectionFailed;
    const r = try tp.roce_multi.Roce.init(d, &b, m.image, wireOptions(cfg, specs[0..n])); // the module copies it at load
    errdefer r.deinit();
    const w = try gpa.create(Wire); // before the last vote: nothing after it can fail
    errdefer gpa.destroy(w);
    w.* = .{ .gpa = gpa, .r = r, .rank = g.rank };
    var p: Probe = .{ .gpa = gpa, .d = d, .g = g, .r = r };
    defer p.deinit();
    try vote(&b, p.steps(), stage);
    var line: [384]u8 = undefined;
    const ceiling = if (r.relay) @min(r.slot_bytes, r.relay_max_bytes) else r.slot_bytes;
    std.log.info("rank {d}: TF_INK2_ROCE_MULTI: gathers up to {d} KiB a rank on the wire, its probe equal to NCCL's: {s}", .{ g.rank, ceiling / 1024, r.describeLink(&line) });
    return w;
}

/// The wire settings used by borrow, kept separate from device setup so the host can check the exact options.
fn wireOptions(cfg: tp.Config, hcas: []const tp.roce_verbs.HcaSpec) tp.roce_multi.Options {
    return .{ .relay = cfg.roce_relay, .relay_max_bytes = cfg.roce_relay_max_bytes, .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = hcas, .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = cfg.mailbox_max_bytes, .timeout_ns = cfg.mailbox_timeout_ns, .fast = cfg.roce_fast, .blocks = cfg.roce_blocks };
}

/// Called before startup or weight loading, using the family's already checked rank/world.
pub fn validate(world: usize, rank: usize, enabled: bool) !void {
    if (enabled) _ = try tp.Config.fromEnvFor(@intCast(world), @intCast(rank));
}

test "inkling2 wire: wire options preserve the opt-in and its slot independently" {
    const o = wireOptions(.{ .world = 4, .roce_relay = true, .roce_relay_max_bytes = 128 * 1024, .mailbox_max_bytes = 256 * 1024 }, &.{});
    try std.testing.expect(o.relay);
    try std.testing.expectEqual(@as(usize, 256 * 1024), o.max_bytes);
    try std.testing.expectEqual(@as(usize, 128 * 1024), o.relay_max_bytes);
    try std.testing.expect(!wireOptions(.{}, &.{}).relay);
}

/// A rank's probe in steps (made ready before any gather, both gathers, the wait, the verdict); a test fails any.
const Steps = struct {
    ctx: *anyopaque,
    prepare: *const fn (*anyopaque) anyerror!void,
    wire: *const fn (*anyopaque) anyerror!void,
    nccl: *const fn (*anyopaque) anyerror!void,
    wait: *const fn (*anyopaque) anyerror!void,
    equal: *const fn (*anyopaque) bool,
};

/// Every rank is ready before any enters a gather, then all enter both; a failed NCCL gather or wait refuses at once.
fn vote(b: *tp.Bootstrap, s: Steps, stage: *Stage) !void {
    const ready = if (s.prepare(s.ctx)) |_| true else |_| false;
    if (!try everyRank(b, ready)) return error.ProbeUnprepared;
    const wired = if (s.wire(s.ctx)) |_| true else |_| false; // NCCL's gather follows it either way
    stage.* = .nccl;
    try s.nccl(s.ctx);
    try s.wait(s.ctx);
    try lendIfAll(b, wired and s.equal(s.ctx));
}

/// One 4 KiB gather in place, on the wire and on the group's NCCL (no wire lent yet): every byte must be equal.
const Probe = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    g: *const Group,
    r: *tp.roce_multi.Roce,
    a: ?cuda.DeviceBuffer = null, // the wire's rows, then NCCL's
    s: ?cuda.Stream = null,

    const n = 4096;

    fn deinit(p: *Probe) void {
        if (p.a) |*a| a.free();
        if (p.s) |*s| s.deinit();
    }

    fn steps(p: *Probe) Steps {
        return .{ .ctx = p, .prepare = prepare, .wire = wire, .nccl = nccl, .wait = wait, .equal = equal };
    }

    fn from(ctx: *anyopaque) *Probe {
        return @ptrCast(@alignCast(ctx));
    }

    fn prepare(ctx: *anyopaque) anyerror!void {
        const p = from(ctx);
        const world = p.g.world;
        p.a = try cuda.DeviceBuffer.alloc(p.d, 2 * world * n);
        p.s = try cuda.Stream.init(p.d, true);
        var mine: [n]u8 = undefined;
        tp.reference.fill(&mine, .bf16, @intCast(p.g.rank), 0x7072_6f62);
        try p.a.?.uploadAsync(p.g.rank * n, &mine, p.s.?.handle);
        try p.a.?.uploadAsync((world + p.g.rank) * n, &mine, p.s.?.handle);
        try p.s.?.synchronize(); // the rows are up before this rank says it is ready
    }

    fn wire(ctx: *anyopaque) anyerror!void {
        const p = from(ctx);
        try p.r.allGatherBytes(p.a.?.ptr + p.g.rank * n, p.a.?.ptr, n, p.s.?.handle);
    }

    fn nccl(ctx: *anyopaque) anyerror!void {
        const p = from(ctx);
        try p.g.allGather(p.s.?, p.a.?.ptr + (p.g.world + p.g.rank) * n, p.a.?.ptr + p.g.world * n, n);
    }

    fn wait(ctx: *anyopaque) anyerror!void {
        try from(ctx).s.?.synchronize();
    }

    fn equal(ctx: *anyopaque) bool {
        const p = from(ctx);
        const world = p.g.world;
        p.r.check() catch return false;
        const got = p.gpa.alloc(u8, 2 * world * n) catch return false;
        defer p.gpa.free(got);
        p.a.?.download(0, got) catch return false;
        const same = std.mem.eql(u8, got[0 .. world * n], got[world * n ..]);
        if (!same) std.log.warn("rank {d}: TF_INK2_ROCE_MULTI's probe: the wire's gather DIFFERS from NCCL's", .{p.g.rank});
        return same;
    }
};

/// The wire is lent only when every rank's probe matched NCCL's bytes: one rank's mismatch keeps every rank on NCCL.
fn lendIfAll(b: *tp.Bootstrap, good: bool) !void {
    if (!try everyRank(b, good)) return error.ProbeMismatch;
}

/// Every rank's vote over the bootstrap, the same answer on every rank.
fn everyRank(b: *tp.Bootstrap, yes: bool) !bool {
    var blobs: [tp.bootstrap.max_ranks][]const u8 = undefined;
    var buf: [tp.bootstrap.max_ranks]u8 = undefined;
    return allSaid(try b.allGather(&.{@intFromBool(yes)}, &buf, &blobs));
}

/// Whether every rank's one-byte vote is yes: every rank reads the same votes, so every rank decides alike.
fn allSaid(votes: []const []const u8) bool {
    for (votes) |v| if (v.len != 1 or v[0] != 1) return false;
    return true;
}

/// The lent wire and what went down it, for its account at the end.
const Wire = struct {
    gpa: std.mem.Allocator,
    r: *tp.roce_multi.Roce,
    rank: usize,
    gathers: std.atomic.Value(u64) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),
};

fn of(ctx: *anyopaque) *Wire {
    return @ptrCast(@alignCast(ctx));
}

fn fits(ctx: *anyopaque, bytes: usize) bool {
    return of(ctx).r.fits(bytes);
}

fn gather(ctx: *anyopaque, s: cuda.abi.Stream, mine: cuda.abi.DevicePtr, all: cuda.abi.DevicePtr, bytes: usize) anyerror!void {
    const w = of(ctx);
    try w.r.allGatherBytes(mine, all, bytes, s);
    // once: the engine's own gathers take the wire (a server's stop exits before any close prints the account)
    if (w.gathers.fetchAdd(1, .monotonic) == 0) std.log.info("rank {d}: TF_INK2_ROCE_MULTI: the first gather is on the wire ({d} bytes a rank)", .{ w.rank, bytes });
    _ = w.bytes.fetchAdd(bytes, .monotonic);
}

fn check(ctx: *anyopaque) anyerror!void {
    return of(ctx).r.check(); // a failed wait's line names the rank and seq it waited on
}

fn abort(ctx: *anyopaque) void {
    const w = of(ctx);
    w.r.abort();
    report(w, "aborted");
}

fn deinit(ctx: *anyopaque) void {
    const w = of(ctx);
    report(w, "closing");
    w.r.deinit();
    w.gpa.destroy(w);
}

/// Gathers, bytes, the proxy's last post and failure, each HCA's QPs: a rank whose posts stop short is the stuck one.
fn report(w: *const Wire, when: []const u8) void {
    var line: [384]u8 = undefined;
    const r = w.r;
    const fmt = "rank {d}: TF_INK2_ROCE_MULTI {s}: {d} gathers, {d} bytes, posted to seq {d}, failed {s}{s}; {s}";
    const args = .{ w.rank, when, w.gathers.load(.monotonic), w.bytes.load(.monotonic), r.last_seq, if (r.failed()) "yes " else "no", r.proxyError(), r.describeLink(&line) };
    if (r.failed()) std.log.warn(fmt, args) else std.log.info(fmt, args);
}

test "TF_INK2_ROCE_MULTI off, or under three ranks: the group keeps its own route and nothing opens" {
    var g: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 3 };
    try open(std.testing.allocator, undefined, undefined, &g, false, null);
    try std.testing.expect(g.multi == null);
    g.world = 2;
    try open(std.testing.allocator, undefined, undefined, &g, true, null);
    try std.testing.expect(g.multi == null);
}

test "inkling2 wire: a failure on one rank after the vote, before its bootstrap is open, never falls back alone" {
    // the vote passed on every rank, then this rank lacks what it agreed (or its bootstrap fails to open): it stops
    // with an error (openAll tells every rank on the liveness lines) instead of returning on NCCL while its peers
    // wait in the bootstrap, whatever the error and TF_TP_ROCE_FALLBACK
    for ([_]anyerror{ error.OutOfMemory, error.FileNotFound, error.ConnectionRefused, error.AddressInUse, error.ProbeMismatch }) |e|
        try std.testing.expect(!fallsBack(e, .local));
    // within the bootstrap every step is every rank's: all fall back together; after the probe's NCCL gather, only a
    // mismatch every rank saw
    try std.testing.expect(fallsBack(error.RoceHcaSelectionFailed, .bootstrap));
    try std.testing.expect(fallsBack(error.ProbeUnprepared, .bootstrap));
    try std.testing.expect(fallsBack(error.ProbeMismatch, .nccl));
    try std.testing.expect(!fallsBack(error.Timeout, .nccl));
    if (comptime builtin.os.tag == .linux) {
        // the wire on at three ranks without the vote's settings and image (a rank that did not pass the vote):
        // refused before any bootstrap, as an error, not a fallback
        var g: Group = .{ .lib = undefined, .comm = null, .rank = 2, .world = 3 };
        try std.testing.expectError(error.TransportNotAgreed, open(std.testing.allocator, undefined, undefined, &g, true, null));
        try std.testing.expect(g.multi == null);
    }
}

test "inkling2 wire: the bootstrap gets the very image bytes the vote hashed, not a second read" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "tp_mailbox.fatbin", .data = "the voted kernel image" });
    const path = try tmp.dir.realPathFileAlloc(io, "tp_mailbox.fatbin", std.testing.allocator);
    defer std.testing.allocator.free(path);
    ImageEnv.path = path;
    var read = readImage(std.testing.allocator, io, &ImageEnv.get);
    defer if (read.bytes) |b| std.testing.allocator.free(b);
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("the voted kernel image", &want, .{});
    try std.testing.expectEqualSlices(u8, &want, &read.image.digest);
    // the file changes (or goes) after the vote: the kept bytes are still the voted ones
    try tmp.dir.writeFile(io, .{ .sub_path = "tp_mailbox.fatbin", .data = "another image" });
    try tmp.dir.deleteFile(io, "tp_mailbox.fatbin");
    const v = ballot(VoteTest.env(&VoteTest.release), 4, 0, true, read.image);
    var cfg: ?tp.Config = tp.Config.parseFor(VoteTest.env(&VoteTest.release), 4, 0) catch unreachable;
    const m = keep(&v, cfg, &read.bytes).?;
    try std.testing.expect(read.bytes == null); // moved into the Mesh
    var got: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(m.image, &got, .{});
    try std.testing.expectEqualSlices(u8, &want, &got);
    try std.testing.expectEqualStrings("198.51.100.1", m.cfg.master);
    std.testing.allocator.free(m.image);
    // with the wire off nothing is kept
    cfg = null;
    const off = ballot(VoteTest.env(&.{}), 4, 0, true, .unset);
    var none: ?[]align(16) u8 = null;
    try std.testing.expect(keep(&off, cfg, &none) == null);
}

const ImageEnv = struct {
    var path: []const u8 = "";
    fn get(name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, "TF_TP_KERNEL_IMAGE")) path else null;
    }
};

test "the wire is lent only when every rank's vote is one byte saying yes" {
    try std.testing.expect(allSaid(&.{ &.{1}, &.{1}, &.{1} }));
    try std.testing.expect(!allSaid(&.{ &.{1}, &.{0}, &.{1} }));
    try std.testing.expect(!allSaid(&.{ &.{1}, &.{}, &.{1} }));
    try std.testing.expect(!allSaid(&.{ &.{1}, &.{ 1, 1 }, &.{1} }));
}

const VoteTest = struct {
    var vars: [8][2][]const u8 = undefined;
    var n: usize = 0;

    fn get(name: []const u8) ?[]const u8 {
        for (vars[0..n]) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
        return null;
    }

    fn env(list: []const [2][]const u8) Get {
        @memcpy(vars[0..list.len], list);
        n = list.len;
        return &get;
    }

    const release = [_][2][]const u8{ .{ "TF_INK2_ROCE_MULTI", "1" }, .{ "TF_TP_MAILBOX_MAX_KB", "640" }, .{ "TF_TP_PORT", "29640" }, .{ "TF_TP_MASTER", "198.51.100.1" }, .{ "TF_TP_KERNEL_IMAGE", "/x/tp_mailbox.fatbin" } };
    const image: Image = .{ .digest = @splat(0x5a) };

    /// Four ballots: the release settings, with rank `odd`'s environment and image replaced.
    fn four(odd: usize, odd_env: []const [2][]const u8, odd_image: Image) [4]Vote {
        var out: [4]Vote = undefined;
        for (&out, 0..) |*v, r| v.* = if (r == odd) ballot(env(odd_env), 4, r, true, odd_image) else ballot(env(&release), 4, r, true, image);
        return out;
    }
};

test "inkling2 transport vote: the release settings alike on four ranks agree" {
    const all = VoteTest.four(9, &.{}, .unset);
    for (&all) |*v| try std.testing.expect(judge(&all, v) == .agreed);
    try std.testing.expectEqualStrings("on", all[0].value(.multi));
    try std.testing.expectEqualStrings("off", all[0].value(.roce));
    try std.testing.expectEqualStrings("640", all[0].value(.mailbox_kb));
    try std.testing.expectEqualStrings("29640/29647", all[0].value(.port));
    try std.testing.expectEqualStrings("5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a...", all[0].value(.image)); // all 32 bytes compared, shown cut
}

test "inkling2 transport vote: TF_ROCE on one follower only is refused on every rank, naming TF_ROCE" {
    const odd = VoteTest.release ++ [_][2][]const u8{.{ "TF_ROCE", "1" }};
    const all = VoteTest.four(1, &odd, VoteTest.image);
    for (&all) |*v| {
        const verdict = judge(&all, v);
        try std.testing.expect(verdict == .differs);
        try std.testing.expectEqual(Setting.roce, verdict.differs.setting.?);
    }
    try std.testing.expectEqualStrings("1", all[1].value(.roce));
    try std.testing.expectEqualStrings("512", all[1].value(.roce_max_kb));
    // the mesh off everywhere: the same refusal
    const plain = [_][2][]const u8{.{ "TF_ROCE", "1" }};
    var off: [4]Vote = undefined;
    for (&off, 0..) |*v, r| v.* = ballot(VoteTest.env(if (r == 2) &plain else &.{}), 4, r, true, .unset);
    try std.testing.expectEqual(Setting.roce, judge(&off, &off[0]).differs.setting.?);
}

test "inkling2 transport vote: one rank without TF_INK2_ROCE_MULTI is refused, naming it" {
    const all = VoteTest.four(3, &.{}, .unset);
    for (&all) |*v| try std.testing.expectEqual(Setting.multi, judge(&all, v).differs.setting.?);
    try std.testing.expectEqualStrings("off", all[3].value(.multi));
}

test "inkling2 transport vote: a missing, unreadable, empty or different kernel image on one rank stops every rank" {
    var all = VoteTest.four(2, &VoteTest.release, .{ .failed = "FileNotFound" });
    for (&all) |*v| try std.testing.expectEqual(@as(usize, 2), judge(&all, v).not_ready);
    try std.testing.expectEqualStrings("TF_TP_KERNEL_IMAGE=/x/tp_mailbox.fatbin: FileNotFound", all[2].reason());
    all = VoteTest.four(0, &VoteTest.release, .empty);
    try std.testing.expectEqual(@as(usize, 0), judge(&all, &all[3]).not_ready);
    all = VoteTest.four(1, VoteTest.release[0..4], .unset); // TF_TP_KERNEL_IMAGE unset
    try std.testing.expectEqual(@as(usize, 1), judge(&all, &all[0]).not_ready);
    all = VoteTest.four(3, &VoteTest.release, .{ .digest = @splat(0x11) });
    try std.testing.expectEqual(Setting.image, judge(&all, &all[0]).differs.setting.?);
}

test "inkling2 transport vote: a different TF_TP_MAILBOX_MAX_KB, an invalid TF_TP_* value or TF_ROCE_MAX_KB is caught" {
    var odd = VoteTest.release;
    odd[1] = .{ "TF_TP_MAILBOX_MAX_KB", "512" };
    var all = VoteTest.four(1, &odd, VoteTest.image);
    try std.testing.expectEqual(Setting.mailbox_kb, judge(&all, &all[0]).differs.setting.?);
    odd[1] = .{ "TF_TP_MAILBOX_MAX_KB", "lots" };
    all = VoteTest.four(1, &odd, VoteTest.image);
    try std.testing.expectEqual(@as(usize, 1), judge(&all, &all[0]).not_ready);
    const bad = [_][2][]const u8{ .{ "TF_ROCE", "1" }, .{ "TF_ROCE_MAX_KB", "x" } };
    var r: [2]Vote = .{ ballot(VoteTest.env(&bad), 2, 0, true, .unset), undefined };
    r[1] = ballot(VoteTest.env(&bad), 2, 1, true, .unset);
    try std.testing.expectEqual(@as(usize, 0), judge(&r, &r[1]).not_ready);
}

test "inkling2 transport vote: long values are compared whole: two master names alike in their first 36 bytes differ" {
    const prefix = "inkling2-rank-zero.a-long-subdomain."; // 36 bytes: what a cut value shows
    comptime std.debug.assert(prefix.len == 36);
    var a = VoteTest.release;
    var b = VoteTest.release;
    a[3] = .{ "TF_TP_MASTER", prefix ++ "lab-one.example.internal" };
    b[3] = .{ "TF_TP_MASTER", prefix ++ "lab-two.example.internal" };
    var all: [4]Vote = undefined;
    for (&all, 0..) |*v, r| v.* = ballot(VoteTest.env(if (r == 3) &b else &a), 4, r, true, VoteTest.image);
    try std.testing.expectEqualStrings(all[0].value(.master), all[3].value(.master)); // shown alike, cut
    try std.testing.expectEqualStrings(prefix ++ "...", all[3].value(.master)[0 .. prefix.len + 3]);
    for (&all) |*v| {
        const verdict = judge(&all, v);
        try std.testing.expect(verdict == .differs);
        try std.testing.expectEqual(Setting.master, verdict.differs.setting.?);
    }
    // the same long name on every rank agrees
    for (&all, 0..) |*v, r| v.* = ballot(VoteTest.env(&a), 4, r, true, VoteTest.image);
    for (&all) |*v| try std.testing.expect(judge(&all, v) == .agreed);
    // any other setting alike in its shown form but not whole: TF_ROCE's value
    const xs: [40]u8 = @splat('x');
    const r1 = [_][2][]const u8{.{ "TF_ROCE", "1" ++ xs ++ "a" }};
    const r2 = [_][2][]const u8{.{ "TF_ROCE", "1" ++ xs ++ "b" }};
    var two: [2]Vote = .{ ballot(VoteTest.env(&r1), 2, 0, true, .unset), undefined };
    two[1] = ballot(VoteTest.env(&r2), 2, 1, true, .unset);
    try std.testing.expectEqualStrings(two[0].value(.roce), two[1].value(.roce));
    try std.testing.expectEqual(Setting.roce, judge(&two, &two[0]).differs.setting.?);
}

test "inkling2 transport vote: a value longer than any valid one is refused, not compared by a prefix" {
    const xs: [1100]u8 = @splat('x');
    const huge = [_][2][]const u8{.{ "TF_ROCE", "1" ++ xs }};
    const v = ballot(VoteTest.env(&huge), 2, 0, true, .unset);
    try std.testing.expect(v.ready == 0);
    try std.testing.expectEqualStrings("TF_ROCE is longer than 1024 bytes", v.reason());
}

test "inkling2 transport vote: under three ranks or off Linux the mesh setting reads off, and a place mismatch is caught" {
    const two = ballot(VoteTest.env(&VoteTest.release), 2, 0, true, .unset);
    try std.testing.expectEqualStrings("off", two.value(.multi));
    try std.testing.expect(two.ready == 1);
    const mac = ballot(VoteTest.env(&VoteTest.release), 4, 0, false, .unset);
    try std.testing.expectEqualStrings("off", mac.value(.multi));
    var all = VoteTest.four(9, &.{}, .unset);
    all[2].rank = 1;
    try std.testing.expect(judge(&all, &all[0]).differs.setting == null);
}
