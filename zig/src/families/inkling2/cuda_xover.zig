//! W > 1, prompt chunks with split experts and by-rows sums (cuda_xrows.zig): the attention side of a layer in two row
//! halves, so each half's by-rows attention sum crosses the wire while the other half computes (TF_INK2_XOVERLAP=1).
//!
//! One layer, halves a = rows [0, Ra) at pos0 and b = rows [Ra, R) at pos0 + Ra, every collective on the overlap's own
//! stream N (one stream for all of the forward's collectives, so the communicator's calls keep one order), every other
//! op on the forward's stream C, events between them:
//!   C  attnStage(a)  attnStage(b)           mlpPre(a) mlpPre(b) splitUp(R)            downStage(R)          outStage(R)
//!   N        sum(a) ......... sum(b) ......                       MoE-A(R) .........        MoE-B(R) ...
//! sum(a) runs under attnStage(b), sum(b) under mlpPre(a). The experts take all R rows at once, so no expert weight is
//! read twice; MoE-A and MoE-B stay where they were (on N, waited for at once).
//!
//! The same bits as the whole-chunk layer: each half is the layer's ops over its rows as a prompt chunk of its own at
//! its positions (the ops are row-invariant and the prompt path chunk-invariant: the attention reads the keys half a
//! wrote, each convolution commit takes its half's rows after the half before it), and the by-rows sum gives each row
//! the whole sum's bits (cuda_xrows.zig). Both halves are above fuse.lat_rows, so they run the chunk's forms. Half b's
//! partials sit after half a's in the attention partial buffer ([W][Ra][H] then [W][Rb][H], within [W][cap][H]), so
//! the next half's output projection never writes what a sum still reads. The prompt cache's captures inside the chunk
//! (cuda_snap.zig) take each mark from the half whose rows reach it.
//!
//! The NCCL calls differ from the whole layer's (two by-rows sums of half size): every rank must set the switch alike
//! (it is in the settings the ranks agree on, cuda_engine.zig agree).
//!
//! TF_INK2_XCODES=1 (prompt chunks of at least min_rows rows, split experts, real ranks; with or without the overlap):
//! MoE-A carries each rank's quantized columns in place of its bf16 ones (cuda_moe.zig splitAmax / splitQuant /
//! splitDownCodes; inkling2_ck.cu "MoE-A as codes"): NVFP4 layers first gather every rank's segment amax of each
//! route (R * 8 fp32), then quantize their own columns with the row amax and gather the codes and scales (0.28x the
//! bf16 bytes); MXFP8 layers quantize and gather (0.52x). The down pass reads the same codes, scales and amax the
//! whole-row quantizer gives, so the bits are unchanged. The gather's slots use the attention partial buffer (nothing
//! live there from the attention sum to the down pass). Every rank must set it alike (agree's settings).
//!
//! TF_INK2_XOVERLAP=s (a test of the halves' bits on one GPU): at W = 1 the same halves with no exchange and no second
//! stream: attnStage(a), attnStage(b), mlpPre(a), mlpPre(b), the experts and outStage over all R rows.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const state = @import("cuda_state.zig");
const tp = @import("cuda_tp.zig");
const xrows = @import("cuda_xrows.zig");
const fwd = @import("cuda_forward.zig");
const moe = @import("cuda_moe.zig");

const Mat = ops.Mat;

/// The fewest rows a chunk splits: halves of at least 288 rows, well above fuse.lat_rows and xrows.min_rows, and every
/// gather of a half above the RoCE mesh wire's ceiling (the release's 640 KiB a rank: a by-rows bf16 gather of 288
/// rows at W = 4 is 648 KiB), so all of a split chunk's collectives stay on NCCL.
pub const min_rows = 576;

var codes_: ?bool = null;

/// TF_INK2_XCODES=1: MoE-A as codes (see the header).
pub fn codesWanted() bool {
    if (codes_ == null) codes_ = if (std.c.getenv("TF_INK2_XCODES")) |v| v[0] == '1' else false;
    return codes_.?;
}

/// Whether forward `f`'s R-row MoE-A goes as codes: the switch, a prompt chunk of at least min_rows rows (decode
/// steps and windows keep the bf16 gather), real ranks with split experts, no op hooks, room in the buffer.
pub fn codes(f: *const fwd.Forward, R: usize) bool {
    if (!codesWanted() or R < min_rows) return false;
    if (f.hooks.record != null or f.hooks.feed != null) return false;
    const x = f.x orelse return false;
    return x.group != null and x.split() != null and x.codesFit(R);
}

/// MoE-A as codes and the down pass, after the up pass left this rank's h (its slot, x.hMine): on the forward's
/// stream, or with `ov` the gathers on its stream N (events between).
pub fn moeA(f: *fwd.Forward, ov: ?*Overlap, li: u32, R: usize, pos0: usize) !void {
    _ = pos0;
    const x = f.x.?;
    const sp = x.split().?;
    const l = f.w.layer(li);
    const nv = l.experts.format == .nvfp4;
    const s: fwd.Scope = .{ .layer = li };
    const C = f.c.stream;
    var cn = f.c;
    if (ov) |o| cn.stream = o.n;
    // NVFP4: every rank's segment amax of each route first
    if (nv) {
        try f.mark(s, "moe.amax");
        try moe.splitAmax(&f.c, l, R, x.world, x.hMine(R), x.amaxMine(R));
        if (ov) |o| try o.handoff(C, o.n, 8);
        try f.mark(s, "tp.split_amax");
        try x.gatherCodes(&cn, .amax, nv, R);
        if (ov) |o| try o.handoff(o.n, C, 9);
    }
    try f.mark(s, "moe.quant");
    try moe.splitQuant(&f.c, l, R, x.world, x.hMine(R), x.amaxAll(R), x.codesMine(nv, R));
    if (ov) |o| try o.handoff(C, o.n, 4);
    try f.mark(s, "tp.split_h");
    try x.gatherCodes(&cn, .codes, nv, R);
    if (ov) |o| try o.handoff(o.n, C, 5);
    try f.mark(s, "moe.down");
    f.c.scratch.reset();
    const ym = x.yMine(R);
    try moe.splitDownCodes(&f.c, l, R, .{ .parts = x.codesAll(), .world = x.world, .amaxes = x.amaxAll(R) }, sp.plan, ym, null, .{}); // >= min_rows rows: never a K1 call
}

pub const Mode = enum { off, on, split };

var mode_: ?Mode = null;

/// TF_INK2_XOVERLAP: 1 prompt chunks overlap their attention sums (see the header), s the halves alone at W = 1 (a
/// test); unset or 0: the whole layer.
pub fn mode() Mode {
    if (mode_ == null) mode_ = if (std.c.getenv("TF_INK2_XOVERLAP")) |v| switch (v[0]) {
        '1' => .on,
        's' => .split,
        else => .off,
    } else .off;
    return mode_.?;
}

/// Whether an engine of `world` ranks (`group`: real ranks) makes the overlap's stream and events.
pub fn wanted(world: u32, group: bool) bool {
    return switch (mode()) {
        .off => false,
        .on => world > 1 and group,
        .split => world == 1,
    };
}

/// TF_INK2_XOVERLAP_SPLIT=N (1..15, default 8): half a takes about N/16 of a chunk's rows (any split gives the same
/// bits; it moves how much compute each half's sum hides under).
pub fn splitSixteenths() usize {
    if (split_ == null) split_ = if (std.c.getenv("TF_INK2_XOVERLAP_SPLIT")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch 8 else 8;
    if (split_.? == 0 or split_.? > 15) split_ = 8;
    return split_.?;
}

var split_: ?u8 = null;

/// Half a's rows of an R-row chunk at W ranks: a multiple of W (so its by-rows sum needs no padding rows) near
/// R * n / 16 (n = splitSixteenths()).
pub fn halfRows(R: usize, world: usize) usize {
    return partRows(R, world, splitSixteenths());
}

fn partRows(R: usize, world: usize, n: usize) usize {
    return world * ((R * n + 16 * world - 1) / (16 * world));
}

/// The activations of rows [row0, cap) of `a` as a forward of their own (every row-major buffer advanced by row0
/// rows; the blocks keep their stride). Owns nothing; `one` (rows of at most attn.one_rows) is left as it is. Every
/// [cap, n] buffer of state.Act is listed (a test checks the count against the struct).
pub fn actFrom(a: *const state.Act, row0: usize) state.Act {
    var v = a.*;
    v.own = .empty;
    for (&v.h) |*m| m.* = from(m.*, row0);
    for (&v.x) |*m| m.* = from(m.*, row0);
    for (&v.xn) |*m| m.* = from(m.*, row0);
    inline for (.{ "q", "k", "v", "rel", "gate", "sink", "dk", "dv", "kc", "vc", "qh", "kh", "vh", "relh", "o", "ao", "a", "partial", "mx", "m", "ids", "wts", "logits", "y", "cm", "tokens" }) |name| {
        @field(v, name) = from(@field(a, name), row0);
    }
    v.blocks = a.blocks + row0 * weights_hidden * 2;
    return v;
}

fn from(m: Mat, row0: usize) Mat {
    return m.slice(row0, m.rows - row0);
}

const weights_hidden = @import("cuda_weights.zig").hidden;

/// The overlap's switches as resolved, for the settings every rank agrees on (cuda_engine.zig agree): bytes only
/// where a switch is on, so a run with every switch off hashes as v0.2's (its default fingerprint test).
/// Every runner of real ranks that can reach Forward.layer must hash it (cuda_engine.zig agree, cuda_run_main.zig).
pub fn hashSettings(h: anytype) void {
    hashOf(h, mode(), splitSixteenths(), codesWanted());
}

/// hashSettings of given switches (resolved): nothing when the overlap is off and the codes are off.
pub fn hashOf(h: anytype, m: Mode, split: usize, with_codes: bool) void {
    if (m != .off) {
        h.update(" xover:");
        h.update(@tagName(m));
        var buf: [16]u8 = undefined;
        h.update(std.fmt.bufPrint(&buf, " split:{d}", .{split}) catch unreachable);
    }
    if (with_codes) h.update(" xcodes:1");
}

pub const Overlap = struct {
    d: *const cuda.Driver,
    /// Layers run in halves (the first one's split is logged).
    layers: u64 = 0,
    /// The collectives' stream (highest priority, so a collective's kernel starts as soon as its inputs are there).
    n: cuda.Stream,
    ev: [10]cuda.Event,

    pub fn init(d: *const cuda.Driver) !Overlap {
        var n = try cuda.Stream.initPriority(d, .highest);
        errdefer n.deinit();
        var ev: [10]cuda.Event = undefined;
        var made: usize = 0;
        errdefer for (ev[0..made]) |*e| e.deinit();
        for (&ev) |*e| {
            e.* = try cuda.Event.init(d, false);
            made += 1;
        }
        return .{ .d = d, .n = n, .ev = ev };
    }

    pub fn deinit(o: *Overlap) void {
        std.log.info("inkling2 xover: {d} layers ran in halves", .{o.layers});
        o.n.synchronize() catch {};
        for (&o.ev) |*e| e.deinit();
        o.n.deinit();
    }

    /// Whether forward `f` runs its R-row layers overlapped: real ranks, split experts, by-rows sums for both halves,
    /// one stream's rows, no op hooks, and room in a.ao for half b's padded sum.
    pub fn applies(o: *const Overlap, f: *const fwd.Forward, R: usize) bool {
        _ = o;
        if (R < min_rows or mode() == .off or f.segs.len != 0) return false;
        if (f.hooks.record != null or f.hooks.feed != null) return false;
        if (mode() == .split) return f.x == null and halfRows(R, 1) > fuse_lat_rows;
        const x = f.x orelse return false;
        if (x.group == null or x.split() == null) return false;
        const W: usize = x.world;
        const Ra = halfRows(R, W);
        const Rb = R - Ra;
        if (Rb < min_rows / 2 or Ra < min_rows / 2) return false; // an uneven split keeps each half above the wire too
        if (!xrows.applies(x, Ra, f.a.ao.rows) or !xrows.applies(x, Rb, f.a.ao.rows - Ra)) return false;
        return R <= x.parts.cap;
    }

    fn mark(o: *Overlap, i: usize, s: cuda.Stream) !void {
        try o.ev[i].record(s);
    }

    /// Stream `dst` waits for what `src` has queued so far (event i).
    fn handoff(o: *Overlap, src: cuda.Stream, dst: cuda.Stream, i: usize) !void {
        try o.ev[i].record(src);
        try dst.wait(o.ev[i]);
    }

    /// Layer `li` of forward `f` (R rows at pos0, `applies`), h[0] in, h[0] out.
    pub fn layer(o: *Overlap, f: *fwd.Forward, li: u32, R: usize, pos0: usize) !void {
        // a failure after a collective is queued: nothing is left running on N past this layer
        errdefer o.n.synchronize() catch {};
        if (o.layers == 0) std.log.info("inkling2 xover: prompt chunks in halves ({t}; first: {d} rows at {d} as {d} + {d})", .{ mode(), R, pos0, halfRows(R, if (f.x) |x| x.world else 1), R - halfRows(R, if (f.x) |x| x.world else 1) });
        o.layers += 1;
        if (f.x == null) return splitLayer(f, li, R, pos0);
        const x = f.x.?;
        const W: usize = x.world;
        const Ra = halfRows(R, W);
        const Rb = R - Ra;
        const C = f.c.stream;
        var cn = f.c;
        cn.stream = o.n;
        // half b's attention partials after half a's (the sum reads [W][Rb][H] from its own base)
        var pb = x.parts.*;
        pb.own = null;
        pb.split = x.parts.split;
        pb.ptr[@intFromEnum(tp.Sum.attn)] += @as(u64, W) * Ra * weights_hidden * 4;
        const xa: tp.Exchange = .{ .rank = x.rank, .world = x.world, .parts = x.parts, .group = x.group };
        const xb: tp.Exchange = .{ .rank = x.rank, .world = x.world, .parts = &pb, .group = x.group };
        var va = actFrom(f.a, 0);
        var vb = actFrom(f.a, Ra);
        var fa = f.*;
        fa.a = &va;
        fa.x = &xa;
        fa.ov = null;
        var fb = f.*;
        fb.a = &vb;
        fb.x = &xb;
        fb.ov = null;
        if (f.capture) |cp| {
            fa.capture = cp.of(pos0, true);
            fb.capture = cp.of(pos0 + Ra, false);
        }
        errdefer {
            if (fa.stopped) |s| f.stopped = s;
            if (fb.stopped) |s| f.stopped = s;
        }
        const s: fwd.Scope = .{ .layer = li };
        // the attention halves; each one's sum on N once its partials are written
        try fa.attnStage(li, Ra, pos0);
        try o.mark(0, C);
        try o.n.wait(o.ev[0]);
        try f.mark(s, "tp.rows");
        try xrows.sum(&xa, &cn, .attn, Ra, va.ao);
        try o.mark(1, o.n);
        try fb.attnStage(li, Rb, pos0 + Ra);
        try o.mark(2, C);
        try o.n.wait(o.ev[2]);
        try xrows.sum(&xb, &cn, .attn, Rb, vb.ao);
        try o.mark(3, o.n);
        // each half from its sum to the router (the summed rows are in a.ao)
        fa.summed[@intFromEnum(tp.Sum.attn)] = true;
        fb.summed[@intFromEnum(tp.Sum.attn)] = true;
        try C.wait(o.ev[1]);
        try fa.mlpPre(li, Ra, pos0);
        try C.wait(o.ev[3]);
        try fb.mlpPre(li, Rb, pos0 + Ra);
        if (va.nblocks != vb.nblocks) return error.OverlapBlocks;
        f.a.nblocks = va.nblocks;
        // the experts over all R rows, MoE-A and MoE-B on N
        try f.mlpExperts(li, R, pos0);
        if (codes(f, R)) try moeA(f, o, li, R, pos0) else {
            try o.mark(4, C);
            try o.n.wait(o.ev[4]);
            try f.mark(s, "tp.split_h");
            try x.gatherSplit(&cn, .h, R);
            try o.mark(5, o.n);
            try C.wait(o.ev[5]);
            try f.downStage(li, R, pos0);
        }
        try o.mark(6, C);
        try o.n.wait(o.ev[6]);
        try f.mark(s, "tp.split_y");
        try x.gatherSplit(&cn, .y, R);
        try o.mark(7, o.n);
        try C.wait(o.ev[7]);
        try f.outStage(li, R, pos0);
    }
};

const fuse_lat_rows = @import("cuda_fuse.zig").lat_rows;

/// TF_INK2_XOVERLAP=s at W = 1: the halves with no exchange (see the header).
fn splitLayer(f: *fwd.Forward, li: u32, R: usize, pos0: usize) !void {
    const Ra = halfRows(R, 1);
    const Rb = R - Ra;
    var va = actFrom(f.a, 0);
    var vb = actFrom(f.a, Ra);
    var fa = f.*;
    fa.a = &va;
    fa.ov = null;
    var fb = f.*;
    fb.a = &vb;
    fb.ov = null;
    if (f.capture) |cp| {
        fa.capture = cp.of(pos0, true);
        fb.capture = cp.of(pos0 + Ra, false);
    }
    errdefer {
        if (fa.stopped) |s| f.stopped = s;
        if (fb.stopped) |s| f.stopped = s;
    }
    try fa.attnStage(li, Ra, pos0);
    try fb.attnStage(li, Rb, pos0 + Ra);
    try fa.mlpPre(li, Ra, pos0);
    try fb.mlpPre(li, Rb, pos0 + Ra);
    if (va.nblocks != vb.nblocks) return error.OverlapBlocks;
    f.a.nblocks = va.nblocks;
    try f.mlpExperts(li, R, pos0);
    try f.outStage(li, R, pos0);
}

test "inkling2 cuda xover: every row buffer of Act is shifted" {
    var n: usize = 0;
    inline for (@typeInfo(state.Act).@"struct".field_names) |name| {
        const T = @FieldType(state.Act, name);
        if (T == Mat) n += 1;
        if (T == [2]Mat) n += 2;
        if (T == [3]Mat) n += 3;
    }
    // 26 named + h (2) + x (3) + xn (3) + `one` (not shifted)
    try std.testing.expectEqual(@as(usize, 26 + 2 + 3 + 3 + 1), n);
}

test "inkling2 cuda xover: the switches' agreement bytes: none when off, distinct when on" {
    const Sha = std.crypto.hash.sha2.Sha256;
    const Case = struct { m: Mode, split: usize, codes: bool };
    const cases = [_]Case{ .{ .m = .off, .split = 8, .codes = false }, .{ .m = .on, .split = 8, .codes = false }, .{ .m = .on, .split = 10, .codes = false }, .{ .m = .off, .split = 8, .codes = true }, .{ .m = .on, .split = 8, .codes = true }, .{ .m = .split, .split = 8, .codes = false } };
    var got: [cases.len][32]u8 = undefined;
    for (cases, 0..) |c, i| {
        var h = Sha.init(.{});
        h.update("base");
        hashOf(&h, c.m, c.split, c.codes);
        h.final(&got[i]);
    }
    var base: [32]u8 = undefined;
    Sha.hash("base", &base, .{});
    // all off: the base's own digest (v0.2's bytes); any switch on: a digest of its own
    try std.testing.expectEqualSlices(u8, &base, &got[0]);
    for (1..cases.len) |i| for (0..i) |j| try std.testing.expect(!std.mem.eql(u8, &got[i], &got[j]));
}

test "inkling2 cuda xover: halves" {
    try std.testing.expectEqual(@as(usize, 1024), partRows(2048, 4, 8));
    try std.testing.expectEqual(@as(usize, 984), partRows(1967, 4, 8));
    try std.testing.expectEqual(@as(usize, 1024), partRows(2048, 2, 8));
    try std.testing.expectEqual(@as(usize, 288), partRows(576, 4, 8));
    try std.testing.expectEqual(@as(usize, 1152), partRows(2048, 4, 9));
    try std.testing.expectEqual(@as(usize, 1280), partRows(2048, 4, 10));
    try std.testing.expectEqual(@as(usize, 496), partRows(992, 1, 8));
    // half b's padded by-rows sum stays inside the chunk's rows rounded up to W
    for ([_]usize{ 576, 577, 1000, 1967, 2045, 2047, 2048 }) |R| for ([_]usize{ 2, 4 }) |W| for ([_]usize{ 8, 9, 10 }) |n| {
        const Ra = partRows(R, W, n);
        try std.testing.expect(Ra % W == 0 and Ra < R);
        const Sb = (R - Ra + W - 1) / W;
        try std.testing.expect(Ra + W * Sb <= W * ((R + W - 1) / W));
    };
}
