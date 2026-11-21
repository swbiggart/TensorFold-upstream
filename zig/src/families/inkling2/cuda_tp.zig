//! W > 1: a rank's two exchanges a layer. Rank r holds query heads [r*48/W, ...), key heads [r*8/W, ...) and experts
//! [r*384/W, ...) (plan.zig); the rest of a layer is replicated. The output projection over the rank's heads and the
//! combine over its experts each leave an fp32 partial [R, H]; every rank's partial is all-gathered in rank order
//! and every rank adds them rank 0 first and rounds to bf16 once (res.rankSum), so every rank holds the same bits
//! after each sum and the router picks the same experts on every rank.
//!
//! Split experts (plan.Layout.split, checkpoint mode): the MoE's rank-order sum gives way to two gathers a layer
//! (cuda_moe.zig `splitUp` / `splitDown`): MoE-A, every rank's h columns [R * 8, I/W] bf16 into [W][R * 8][I/W], and
//! MoE-B, every rank's output columns [R, H/W] bf16 into [W][R][H/W], laid out as rows by `columns`. No rank sums
//! anything another rank computed, so every rank holds the same bits, equal to W = 1's. Each gather's size depends on R
//! only, and nothing waits on the host inside a layer.
//!
//! Real ranks (one process a rank) gather over NCCL (cuda/tp_group.zig, a blocking communicator bootstrapped from
//! NCCL_COMM_ID). Virtual ranks (all ranks in one process, one after another on one card) share the partial
//! buffers, so nothing moves: each writes its own slot and the sum reads them all.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const res = @import("cuda_res.zig");
const weights = @import("cuda_weights.zig");
const moe = @import("cuda_moe.zig");

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;

/// The layer's two sums: after the output projection and after the experts.
pub const Sum = enum(u1) { attn, moe };

/// The partials of both sums: fp32 [world][R][H] each, for forwards of at most `cap` rows. Two buffers, so a virtual
/// rank's MoE partial never lands on an attention partial a later virtual rank has still to add.
pub const Parts = struct {
    ptr: [2]u64,
    world: u32,
    cap: usize,
    own: ?[2]cuda.DeviceBuffer = null,
    /// Split experts: the MoE-A and MoE-B gather buffers and the route plan in place of the MoE partials.
    split: ?Split = null,

    pub const Split = struct {
        /// bf16 [world][cap * 8][I / world]
        h: u64,
        /// bf16 [world][cap][H / world]
        y: u64,
        /// cuda_moe.zig splitPlanBytes(cap)
        plan: u64,
        own: [3]cuda.DeviceBuffer,
        bytes: usize,
    };

    /// `layout` whole: both sums' fp32 partials; split: the attention partials and the split experts' buffers.
    pub fn init(d: *const cuda.Driver, world: u32, cap: usize) !Parts {
        return initFor(d, world, cap, .whole);
    }

    pub fn initFor(d: *const cuda.Driver, world: u32, cap: usize, layout: @import("inkling2").plan.Layout) !Parts {
        const n = @as(usize, world) * cap * H * 4;
        var a = try cuda.DeviceBuffer.alloc(d, n);
        errdefer a.free();
        if (layout == .whole) {
            const b = try cuda.DeviceBuffer.alloc(d, n);
            return .{ .ptr = .{ a.ptr, b.ptr }, .world = world, .cap = cap, .own = .{ a, b } };
        }
        const sizes = [3]usize{ cap * moe.topk * moe.intermediate * 2, cap * H * 2, moe.splitPlanBytes(cap) };
        var own: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (own[0..made]) |*b| b.free();
        for (&own, sizes) |*b, size| {
            b.* = try cuda.DeviceBuffer.alloc(d, size);
            made += 1;
        }
        const empty = try cuda.DeviceBuffer.alloc(d, 0); // no MoE partials: address 0, nothing held
        return .{ .ptr = .{ a.ptr, 0 }, .world = world, .cap = cap, .own = .{ a, empty }, .split = .{ .h = own[0].ptr, .y = own[1].ptr, .plan = own[2].ptr, .own = own, .bytes = sizes[0] + sizes[1] + sizes[2] } };
    }

    pub fn deinit(p: *Parts) void {
        if (p.own) |*o| for (o) |*b| b.free();
        p.own = null;
        if (p.split) |*x| for (&x.own) |*b| b.free();
        p.split = null;
    }

    pub fn bytes(p: *const Parts) usize {
        const whole = @as(usize, p.world) * p.cap * H * 4;
        return if (p.split) |x| whole + x.bytes else 2 * whole;
    }
};

/// TF_INK2_TIME_EXCHANGE=1: every real-rank gather waits for its stream before and after, and its wall time (the
/// transfer plus the wait for the slowest rank) is counted here per sum; the runner prints the totals.
pub const Timing = struct {
    pub var on: ?bool = null;
    pub var ns: [2]u64 = .{ 0, 0 };
    pub var calls: [2]u64 = .{ 0, 0 };

    pub fn enabled() bool {
        if (on == null) on = if (std.c.getenv("TF_INK2_TIME_EXCHANGE")) |v| v[0] == '1' else false;
        return on.?;
    }

    pub fn now() u64 {
        var t: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
        return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
    }
};

pub const Exchange = struct {
    rank: u32,
    world: u32,
    parts: *const Parts,
    /// Real ranks: the group the partials are gathered over; null: virtual ranks (shared buffers).
    group: ?*const cuda.tp_group.Group = null,

    fn base(x: *const Exchange, s: Sum) u64 {
        return x.parts.ptr[@intFromEnum(s)];
    }

    /// This rank's partial of sum `s` for a forward of R rows (its slot of [world][R][H]).
    pub fn mine(x: *const Exchange, s: Sum, R: usize) Mat {
        return Mat.of(x.base(s) + @as(u64, x.rank) * R * H * 4, R, H, .f32);
    }

    /// Every rank's partial into every rank's buffer, rank order (real ranks; virtual ranks share the buffer).
    pub fn gather(x: *const Exchange, c: *const Ctx, s: Sum, R: usize) !void {
        if (R > x.parts.cap) return error.ExchangeTooSmall;
        if (s == .moe and x.parts.split != null) return error.SplitExpertsGatherTwice;
        const g = x.group orelse return;
        if (!Timing.enabled()) return g.allGather(c.stream, x.mine(s, R).ptr, x.base(s), R * H * 4);
        try c.stream.synchronize();
        const t = Timing.now();
        try g.allGather(c.stream, x.mine(s, R).ptr, x.base(s), R * H * 4);
        try c.stream.synchronize();
        Timing.ns[@intFromEnum(s)] += Timing.now() - t;
        Timing.calls[@intFromEnum(s)] += 1;
    }

    /// out bf16 [R, H] = bf16(partial_0 + partial_1 + ...), rank order.
    pub fn sum(x: *const Exchange, c: *const Ctx, s: Sum, out: Mat) !void {
        try res.rankSum(c, x.base(s), x.world, out);
    }

    /// The split experts' buffers (null: whole experts).
    pub fn split(x: *const Exchange) ?Parts.Split {
        return x.parts.split;
    }

    /// Split experts: this rank's h [R * 8, I/W] (its slot of MoE-A's [W][R * 8][I/W]).
    pub fn hMine(x: *const Exchange, R: usize) u64 {
        return x.parts.split.?.h + @as(u64, x.rank) * R * moe.topk * (moe.intermediate / x.world) * 2;
    }

    /// Split experts: this rank's output columns [R, H/W] (its slot of MoE-B's [W][R][H/W]).
    pub fn yMine(x: *const Exchange, R: usize) u64 {
        return x.parts.split.?.y + @as(u64, x.rank) * R * (H / x.world) * 2;
    }

    /// Split experts: MoE-A (`.h`) or MoE-B (`.y`) for a forward of R rows, every rank's slot into every rank's buffer,
    /// rank order (real ranks; virtual ranks share the buffers). Timed with the MoE sum (TF_INK2_TIME_EXCHANGE).
    pub fn gatherSplit(x: *const Exchange, c: *const Ctx, which: enum { h, y }, R: usize) !void {
        if (R > x.parts.cap) return error.ExchangeTooSmall;
        const sp = x.parts.split orelse return error.NotSplitExperts;
        const g = x.group orelse return;
        const own, const all, const n = switch (which) {
            .h => .{ x.hMine(R), sp.h, R * moe.topk * (moe.intermediate / x.world) * 2 },
            .y => .{ x.yMine(R), sp.y, R * (H / x.world) * 2 },
        };
        if (!Timing.enabled()) return g.allGather(c.stream, own, all, n);
        try c.stream.synchronize();
        const t = Timing.now();
        try g.allGather(c.stream, own, all, n);
        try c.stream.synchronize();
        Timing.ns[1] += Timing.now() - t;
        Timing.calls[1] += 1;
    }

    /// Split experts, after MoE-B: out bf16 [R, H], every rank's columns in place.
    pub fn columns(x: *const Exchange, c: *const Ctx, out: Mat) !void {
        try moe.splitColumns(c, x.parts.split.?.y, x.world, out);
    }

    /// MoE-A as codes (TF_INK2_XCODES, cuda_moe.zig splitQuant): the gather's slots [W][codesSlot] and then every
    /// rank's segment amax [W][R * 8] fp32 sit in the attention sum's partial buffer, which holds nothing live from
    /// the attention sum's end to the down pass (the next layer's output projection writes it again). Its size, W * cap
    /// rows of H fp32, holds both for every R <= cap at W >= 2 (codesFit).
    pub fn codesAll(x: *const Exchange) u64 {
        return x.base(.attn);
    }

    pub fn codesMine(x: *const Exchange, nv: bool, R: usize) u64 {
        return x.codesAll() + @as(u64, x.rank) * moe.codesSlot(nv, x.world, R * moe.topk);
    }

    pub fn amaxAll(x: *const Exchange, R: usize) u64 {
        // after the larger (MXFP8) slots, 256-byte aligned: one place for both formats
        const slots = @as(usize, x.world) * moe.codesSlot(false, x.world, R * moe.topk);
        return x.codesAll() + std.mem.alignForward(usize, slots, 256);
    }

    pub fn amaxMine(x: *const Exchange, R: usize) u64 {
        return x.amaxAll(R) + @as(u64, x.rank) * R * moe.topk * 4;
    }

    /// Whether the codes and amaxes of an R-row forward fit the attention partial buffer.
    pub fn codesFit(x: *const Exchange, R: usize) bool {
        const need = (x.amaxAll(R) - x.codesAll()) + @as(usize, x.world) * R * moe.topk * 4;
        return R <= x.parts.cap and need <= @as(usize, x.world) * x.parts.cap * H * 4;
    }

    /// MoE-A as codes: every rank's quantized slot (`nv`: NVFP4 layer) or segment amax (`.amax`) into every rank's
    /// buffer, rank order (real ranks; virtual ranks share the buffer). Timed with the MoE sum (TF_INK2_TIME_EXCHANGE).
    pub fn gatherCodes(x: *const Exchange, c: *const Ctx, which: enum { codes, amax }, nv: bool, R: usize) !void {
        if (!x.codesFit(R)) return error.ExchangeTooSmall;
        const g = x.group orelse return;
        const own, const all, const n = switch (which) {
            .codes => .{ x.codesMine(nv, R), x.codesAll(), moe.codesSlot(nv, x.world, R * moe.topk) },
            .amax => .{ x.amaxMine(R), x.amaxAll(R), R * moe.topk * 4 },
        };
        if (!Timing.enabled()) return g.allGather(c.stream, own, all, n);
        try c.stream.synchronize();
        const t = Timing.now();
        try g.allGather(c.stream, own, all, n);
        try c.stream.synchronize();
        Timing.ns[1] += Timing.now() - t;
        Timing.calls[1] += 1;
    }
};

test "inkling2 cuda tp: MoE-A's codes and amaxes fit the attention partials" {
    const parts: Parts = .{ .ptr = .{ 1 << 30, 0 }, .world = 4, .cap = 2048 };
    for ([_]u32{ 2, 4 }) |W| for ([_]usize{ 1, 576, 2047, 2048 }) |R| {
        var p = parts;
        p.world = W;
        const x: Exchange = .{ .rank = W - 1, .world = W, .parts = &p };
        try std.testing.expect(x.codesFit(R));
        // the last rank's amax ends inside the buffer, after every rank's codes (either format)
        try std.testing.expect(x.amaxMine(R) + R * moe.topk * 4 <= x.codesAll() + @as(usize, W) * p.cap * H * 4);
        try std.testing.expect(x.codesMine(false, R) + moe.codesSlot(false, W, R * moe.topk) <= x.amaxAll(R));
    };
}

test "inkling2 cuda tp: a rank's slot of the partials" {
    const parts: Parts = .{ .ptr = .{ 1 << 30, 2 << 30 }, .world = 4, .cap = 32 };
    const x: Exchange = .{ .rank = 2, .world = 4, .parts = &parts };
    try std.testing.expectEqual(@as(u64, (2 << 30) + 2 * 25 * H * 4), x.mine(.moe, 25).ptr);
    try std.testing.expect(x.mine(.attn, 25).is(25, H, .f32));
    try std.testing.expectEqual(@as(usize, 2 * 4 * 32 * H * 4), parts.bytes());
}
