//! Inkling-2's forward in `full` mode on one rank, written as the sequence of op-group entry points (cuda_res,
//! cuda_attn, cuda_conv, cuda_moe, cuda_head) over the activations of one forward (`Act`: R rows at positions
//! pos0 .. pos0 + R - 1 of one stream) and the stream's state (`State`). Every op runs between `before` and
//! `after`, which give the hooks the op's tensors: --dump-ops records inputs and outputs, --feed replaces every
//! op's outputs with torch's (so each op is tested on torch's inputs) and lets an op without a kernel
//! (error.NotImplemented) pass with torch's outputs. Without --feed such an op stops the forward, named in `stopped`.
//!
//! One decoder layer L (Transformers' Inkling2SlidingDecoderLayer / Inkling2FullDecoderLayer):
//!   res.attn_input  x_j = Mix(attn_queries; blocks, h_in)              (L > 0; 1 query, 3 in global layers)
//!   res.attn_norm   xn = norm(x) (sliding) | res.q_norm, res.k_norm, res.v_norm (global)
//!   attn.project    q, k, v, rel, gate, sink (+ dk, dv)
//!   conv.kv         kc, vc                       then the scaffold commits the K and V conv states
//!   attn.heads      q', k', v', rel'             then the scaffold writes k', v' to the cache
//!   attn.core       o
//!   attn.out        ao                           (W > 1: the rank's fp32 partial; sum 1 in rank order -> ao)
//!   conv.attn       a                            then commits the post-attention conv state
//!   res.partial     partial                      then, at a boundary layer, pushes h_in onto the blocks
//!   res.mlp_input   x = Mix(mlp_query; blocks, partial)
//!   res.mlp_norm    m
//!   moe.route       ids, wts, logits
//!   moe.experts     y                            (W > 1: the rank's fp32 partial; sum 2 in rank order -> y)
//!                   split experts (W > 1, checkpoint mode): h, the rank's act columns; MoE-A gathers them;
//!   moe.down        the rank's columns of y      (split experts only); MoE-B gathers them -> y
//!   conv.mlp        cm                           then commits the post-MLP conv state
//!   res.output      h_out
//! Model ops: head.embed before layer 0; res.final_input, head.final_norm, head.unembed, head.softcap after 47.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const attn = @import("cuda_attn.zig");
const conv = @import("cuda_conv.zig");
const res = @import("cuda_res.zig");
const moe = @import("cuda_moe.zig");
const head = @import("cuda_head.zig");
const tp = @import("cuda_tp.zig");
const xrows = @import("cuda_xrows.zig");
const fuse = @import("cuda_fuse.zig");

const Mat = ops.Mat;
const Ctx = ops.Ctx;
const H = weights.hidden;

/// The `Prepare` hooks of every group, run by the loader per layer.
pub const prepare = [_]weights.Prepare{ attn.prepare, conv.prepare, res.prepare, moe.prepare, head.prepare };

pub const Scope = union(enum) {
    layer: u32,
    model,

    pub fn format(s: Scope, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (s) {
            .layer => |l| try w.print("{d}", .{l}),
            .model => try w.writeAll("model"),
        }
    }
};

pub const Dir = enum { in, out };

/// One named tensor of an op. `whole`: not rows of the forward (a convolution state, gathered cache rows).
pub const Tensor = struct {
    name: []const u8,
    m: Mat,
    whole: bool = false,
};

pub const Hooks = struct {
    ctx: ?*anyopaque = null,
    /// --dump-ops: an op's input (before it runs) or output (after); `row0` is the forward's first row.
    record: ?*const fn (ctx: *anyopaque, scope: Scope, op: []const u8, dir: Dir, t: Tensor, row0: usize) anyerror!void = null,
    /// --feed: put torch's rows row0 .. row0 + R of this output in place; false when the capture has none.
    feed: ?*const fn (ctx: *anyopaque, scope: Scope, op: []const u8, t: Tensor, row0: usize) anyerror!bool = null,
    /// An op without a kernel passed with torch's outputs.
    stub: ?*const fn (ctx: *anyopaque, scope: Scope, op: []const u8) anyerror!void = null,
};

/// The runner's --prof: stream time between marks, summed by layer and label. Each op's `before` marks its start, the
/// scaffold's own copies and the rank sums mark theirs, and a span ends at the next mark (or at `end`), so the spans
/// tile the stream's time from the first mark to `end` (launch gaps count to the span they fall in).
pub const Prof = struct {
    pub const max_marks = 2048;
    pub const max_labels = 48;
    /// Rows of `ms`: layers 0 .. 47, then the model ops.
    pub const rows = 49;

    d: *const cuda.Driver,
    stream: cuda.Stream,
    events: [max_marks]?cuda.Event = @splat(null),
    at: [max_marks][2]u32 = undefined,
    n: usize = 0,
    labels: [max_labels][]const u8 = undefined,
    nlabels: usize = 0,
    ms: [rows][max_labels]f64 = @splat(@splat(0)),
    /// Spans ended (`end` calls with marks): forwards profiled since the last reset.
    ends: usize = 0,

    pub fn deinit(p: *Prof) void {
        for (&p.events) |*e| if (e.*) |*x| x.deinit();
    }

    fn event(p: *Prof) !cuda.Event {
        if (p.n >= max_marks) return error.ProfTooManyMarks;
        if (p.events[p.n] == null) p.events[p.n] = try cuda.Event.init(p.d, true);
        const e = p.events[p.n].?;
        try e.record(p.stream);
        return e;
    }

    pub fn mark(p: *Prof, scope: Scope, label: []const u8) !void {
        const li = for (p.labels[0..p.nlabels], 0..) |x, i| {
            if (std.mem.eql(u8, x, label)) break i;
        } else blk: {
            if (p.nlabels >= max_labels) return error.ProfTooManyLabels;
            p.labels[p.nlabels] = label;
            p.nlabels += 1;
            break :blk p.nlabels - 1;
        };
        _ = try p.event();
        p.at[p.n] = .{ switch (scope) {
            .layer => |l| @min(l, rows - 1),
            .model => rows - 1,
        }, @intCast(li) };
        p.n += 1;
    }

    /// Ends the last span, waits for the stream and adds the spans up.
    pub fn end(p: *Prof) !void {
        if (p.n == 0) return;
        _ = try p.event();
        try p.events[p.n].?.synchronize();
        for (0..p.n) |i| {
            const ms = try cuda.Event.elapsedMs(p.events[i].?, p.events[i + 1].?);
            p.ms[p.at[i][0]][p.at[i][1]] += ms;
        }
        p.n = 0;
        p.ends += 1;
    }

    pub fn reset(p: *Prof) void {
        p.ms = @splat(@splat(0));
        p.ends = 0;
    }
};

/// An A/B of the fused ops in one process (tf-inkling2-decode --ab): forwards on this state run the unfused ops.
pub var unfused_state: ?*const state.State = null;

pub const Stop = struct {
    scope: Scope,
    op: []const u8,
    err: anyerror,
};

/// One stream's rows of a forward (batched decode, cuda_engine.zig forwardMany): rows row0 .. row0 + rows of the
/// forward, at positions pos0 .. pos0 + rows - 1 of that stream, whose state is `st`. `spec`: a verify window on a
/// slot with drafts (each convolution commit keeps its inputs, cuda_state.zig convCommitSpec). `dev`: a captured
/// forward (cuda_engine.zig graphStep) reads the segment's first position on the device (cuda_attn.zig Dev: the
/// heads, the cache write and the attention core), so its graph replays at every position; pos0 is then the
/// capture's.
pub const Seg = struct {
    st: *state.State,
    pos0: usize,
    row0: usize = 0,
    rows: usize,
    spec: bool = false,
    dev: ?attn.Dev = null,
};

/// Segments a forward carries at most (the step header's room, cuda_engine.zig Header).
pub const max_segs = 8;

/// The prompt cache's mid-chunk capture (cuda_snap.zig forwardHook): called before each convolution commit with the
/// state about to be committed and the rows committed into it; it copies what the state would be at a position
/// inside the forward. Reads only; the forward's own bytes do not change.
pub const Capture = struct {
    ctx: *anyopaque,
    conv: *const fn (ctx: *anyopaque, c: *const Ctx, conv_state: u64, x: Mat) anyerror!void,
};

pub const Forward = struct {
    c: Ctx,
    w: *const weights.Weights,
    /// The forward's one stream (rows 0 .. R at pos0), unless `segs` is set.
    st: *state.State,
    a: *state.Act,
    hooks: Hooks = .{},
    stopped: ?Stop = null,
    /// W > 1: the rank's two sums a layer (cuda_tp.zig); null at W = 1.
    x: ?*const tp.Exchange = null,
    /// A verify window on a slot with drafts: each convolution commit keeps its inputs (cuda_state.zig convCommitSpec).
    spec: bool = false,
    /// Several streams' rows in one forward (batched decode): the ops that read or write a stream's state (the
    /// convolutions and their commits, the heads, the cache write, the attention core) run once a segment on its
    /// rows; every other op runs once over all the forward's rows. Each op is row-invariant (a row's bits do not
    /// depend on the forward's other rows), so each stream's rows get the bits of a forward of their own. Empty: the
    /// one stream `st` (with `spec`). The hooks (--dump-ops, --feed) take one stream only.
    segs: []const Seg = &.{},
    /// Not 0: the embedding's token ids (i32, on the device) are read here instead of `a.tokens` (a captured forward
    /// reads its tokens from the step header's device copy).
    ids: u64 = 0,
    /// The runner's --prof.
    prof: ?*Prof = null,
    /// Sums the exchange already left in a.ao / a.y as bf16 (cuda_xrows.zig), so the stage after skips its sum.
    summed: [2]bool = .{ false, false },
    /// The prompt cache's capture of states inside this forward (a one-stream prompt chunk; cuda_snap.zig).
    capture: ?Capture = null,

    fn commit(f: *Forward, spec: bool, conv_state: u64, hist: u64, x: Mat) !void {
        if (f.capture) |cp| try cp.conv(cp.ctx, &f.c, conv_state, x);
        if (spec and hist != 0) return state.convCommitSpec(&f.c, conv_state, hist, x);
        return state.convCommit(&f.c, conv_state, x);
    }

    /// The forward's segments: `segs`, or the one stream's rows 0 .. R at pos0 (in `one`).
    fn segments(f: *const Forward, R: usize, pos0: usize, one: *[1]Seg) ![]const Seg {
        if (f.segs.len == 0) {
            one[0] = .{ .st = f.st, .pos0 = pos0, .rows = R, .spec = f.spec };
            return one;
        }
        if (f.hooks.record != null or f.hooks.feed != null) return error.SegmentsTakeNoHooks;
        return f.segs;
    }

    fn mark(f: *Forward, scope: Scope, label: []const u8) !void {
        if (f.prof) |p| try p.mark(scope, label);
    }

    /// Which stateless ops a call of R rows runs in cuda_fuse.zig's forms (TF_INK2_FUSE; by default the latency
    /// forms up to fuse.lat_rows rows): never while the hooks look at or replace each op's tensors (--dump-ops,
    /// --feed), which need the ops one by one, nor on `unfused_state`'s slot.
    fn fuseMode(f: *const Forward, R: usize) fuse.Mode {
        if (f.hooks.record != null or f.hooks.feed != null) return .none;
        if (unfused_state) |u| if (u == f.st) return .none;
        return f.c.k.fuse.mode.forRows(R);
    }

    fn before(f: *Forward, scope: Scope, op: []const u8, ins: []const Tensor, row0: usize) !void {
        try f.mark(scope, op);
        f.c.scratch.reset();
        const rec = f.hooks.record orelse return;
        try f.c.stream.synchronize();
        for (ins) |t| if (t.m.rows > 0 and t.m.ptr != 0) try rec(f.hooks.ctx.?, scope, op, .in, t, row0);
    }

    fn after(f: *Forward, scope: Scope, op: []const u8, outs: []const Tensor, result: anyerror!void, row0: usize) !void {
        var stub = false;
        result catch |e| {
            if (e != error.NotImplemented or f.hooks.feed == null) {
                f.stopped = .{ .scope = scope, .op = op, .err = e };
                return e;
            }
            stub = true;
        };
        if (stub) {
            for (outs) |t| if (t.m.rows > 0 and t.m.ptr != 0 and !try f.hooks.feed.?(f.hooks.ctx.?, scope, op, t, row0)) {
                std.log.err("inkling2: {f} {s} has no kernel and the capture has no {s} for it", .{ scope, op, t.name });
                f.stopped = .{ .scope = scope, .op = op, .err = error.NotImplemented };
                return error.NotImplemented;
            };
            if (f.hooks.stub) |s| try s(f.hooks.ctx.?, scope, op);
            return;
        }
        if (f.hooks.record) |rec| {
            try f.c.stream.synchronize();
            for (outs) |t| if (t.m.rows > 0 and t.m.ptr != 0) try rec(f.hooks.ctx.?, scope, op, .out, t, row0);
        }
        if (f.hooks.feed) |feed| for (outs) |t| if (t.m.rows > 0 and t.m.ptr != 0) {
            _ = try feed(f.hooks.ctx.?, scope, op, t, row0);
        };
    }

    /// The blocks as op tensors block0 .. block<n-1>.
    fn blockTensors(f: *Forward, R: usize, buf: *[state.max_blocks + 1]Tensor) []Tensor {
        const names = [_][]const u8{ "block0", "block1", "block2", "block3", "block4", "block5", "block6", "block7" };
        for (0..f.a.nblocks) |i| buf[i] = .{ .name = names[i], .m = f.a.block(i, R) };
        return buf[0..f.a.nblocks];
    }

    /// head.embed: the normed embedding of the forward's token ids (already in `a.tokens`) into h[0].
    pub fn embed(f: *Forward, R: usize, pos0: usize) !void {
        const a = f.a;
        const out = a.h[0].slice(0, R);
        try f.before(.model, "head.embed", &.{.{ .name = "ids", .m = a.tokens.slice(0, R) }}, pos0);
        try f.after(.model, "head.embed", &.{.{ .name = "e", .m = out }}, head.embed(&f.c, f.w, if (f.ids != 0) f.ids else a.tokens.ptr, a.y.slice(0, R), out), pos0);
    }

    /// head.embed with the table held by vocabulary rows over the ranks (cuda_embed.zig): every rank's rows of the
    /// forward's token ids (already on every rank: `ids`, or `a.tokens`) gathered, picked and normed into h[0], the
    /// bits `embed` gives on rank 0.
    pub fn embedShared(f: *Forward, sh: *const @import("cuda_embed.zig").Shared, R: usize, pos0: usize) !void {
        const a = f.a;
        const out = a.h[0].slice(0, R);
        try f.before(.model, "head.embed", &.{.{ .name = "ids", .m = a.tokens.slice(0, R) }}, pos0);
        try f.after(.model, "head.embed", &.{.{ .name = "e", .m = out }}, sh.rows(&f.c, f.w, if (f.ids != 0) f.ids else a.tokens.ptr, a.y.slice(0, R), out), pos0);
    }

    /// Layer `li` over the forward's R rows: h[0] in, h[0] out (the buffers swap). At W > 1 the rank's partials are
    /// gathered over its group between the stages (virtual ranks call the stages themselves, every rank's stage in
    /// turn); split experts gather twice in the MoE (MoE-A after mlpStage, MoE-B after downStage).
    pub fn layer(f: *Forward, li: u32, R: usize, pos0: usize) !void {
        try f.attnStage(li, R, pos0);
        if (f.x) |x| try f.exchange(x, .attn, li, R);
        try f.mlpStage(li, R, pos0);
        if (f.split()) {
            try f.mark(.{ .layer = li }, "tp.split_h");
            try f.x.?.gatherSplit(&f.c, .h, R);
            try f.downStage(li, R, pos0);
            try f.mark(.{ .layer = li }, "tp.split_y");
            try f.x.?.gatherSplit(&f.c, .y, R);
        } else if (f.x) |x| try f.exchange(x, .moe, li, R);
        try f.outStage(li, R, pos0);
    }

    /// Split experts (the exchange has their buffers): the MoE runs in two halves around MoE-A.
    pub fn split(f: *const Forward) bool {
        return if (f.x) |x| x.split() != null else false;
    }

    /// Real ranks: sum `s`'s partials over the group, by rows for prompt chunks (cuda_xrows.zig: the sum's bf16 rows
    /// land in a.ao / a.y, the same bits as the whole sum), else the whole all-gather (the next stage sums).
    fn exchange(f: *Forward, x: *const tp.Exchange, s: tp.Sum, li: u32, R: usize) !void {
        const out = if (s == .attn) f.a.ao else f.a.y;
        if (xrows.applies(x, R, out.rows)) {
            try f.mark(.{ .layer = li }, "tp.rows");
            f.c.scratch.reset();
            try xrows.sum(x, &f.c, s, R, out);
            f.summed[@intFromEnum(s)] = true;
            return;
        }
        try f.mark(.{ .layer = li }, "tp.gather");
        try x.gather(&f.c, s, R);
    }

    /// Whether sum `s` still has to run in the stage (and clears the exchange's note).
    fn unsummed(f: *Forward, s: tp.Sum) bool {
        const done = f.summed[@intFromEnum(s)];
        f.summed[@intFromEnum(s)] = false;
        return !done;
    }

    /// The layer up to the output projection: ao bf16 at W = 1, else the rank's fp32 partial (its heads).
    pub fn attnStage(f: *Forward, li: u32, R: usize, pos0: usize) !void {
        const l = f.w.layer(li);
        const a = f.a;
        const c = &f.c;
        const s: Scope = .{ .layer = li };
        const h_in = a.h[0].slice(0, R);
        var bt: [state.max_blocks + 1]Tensor = undefined;
        var ins: [16]Tensor = undefined;
        // attention input (L1) and its norms (L2)
        var src: [3]Mat = .{ h_in, h_in, h_in };
        const norm_ops = if (l.global) [3][]const u8{ "res.q_norm", "res.k_norm", "res.v_norm" } else [3][]const u8{ "res.attn_norm", "", "" };
        const sources: usize = if (l.global) 3 else 1;
        const xn = [3]Mat{ a.xn[0].slice(0, R), a.xn[1].slice(0, R), a.xn[2].slice(0, R) };
        const fuse_in = l.nq > 0 and f.fuseMode(R).attnIn();
        if (fuse_in) {
            // res.attn_input and its norms as one launch
            const xs = [3]Mat{ a.x[0].slice(0, R), a.x[1].slice(0, R), a.x[2].slice(0, R) };
            try f.mark(s, "fuse.attn_in");
            c.scratch.reset();
            try fuse.attnIn(c, a.blockView(R), h_in, l.attn_query, l.nq, xs[0..l.nq], l.in_norm[0..sources], xn[0..sources]);
        } else if (l.nq > 0) {
            const xs = [3]Mat{ a.x[0].slice(0, R), a.x[1].slice(0, R), a.x[2].slice(0, R) };
            const xt = [3]Tensor{ .{ .name = "x0", .m = xs[0] }, .{ .name = "x1", .m = xs[1] }, .{ .name = "x2", .m = xs[2] } };
            const blocks = f.blockTensors(R, &bt);
            @memcpy(ins[0..blocks.len], blocks);
            ins[blocks.len] = .{ .name = "cand", .m = h_in };
            try f.before(s, "res.attn_input", ins[0 .. blocks.len + 1], pos0);
            try f.after(s, "res.attn_input", xt[0..l.nq], res.mix(c, a.blockView(R), h_in, l.attn_query, l.nq, xs[0..l.nq]), pos0);
            src = if (l.nq == 1) .{ xs[0], xs[0], xs[0] } else xs;
        }
        if (!fuse_in) for (0..sources) |j| {
            try f.before(s, norm_ops[j], &.{.{ .name = "x", .m = src[j] }}, pos0);
            try f.after(s, norm_ops[j], &.{.{ .name = "y", .m = xn[j] }}, res.norm(c, src[j], l.in_norm[j], xn[j]), pos0);
        };
        const xq = xn[0];
        const xk = if (l.global) xn[1] else xn[0];
        const xv = if (l.global) xn[2] else xn[0];
        // projections (L3)
        const p: attn.Proj = .{
            .q = a.q.slice(0, R),
            .k = a.k.slice(0, R),
            .v = a.v.slice(0, R),
            .rel = a.rel.slice(0, R),
            .gate = a.gate.slice(0, R),
            .sink = a.sink.slice(0, R),
            .dk = if (l.global) a.dk.slice(0, R) else .{},
            .dv = if (l.global) a.dv.slice(0, R) else .{},
        };
        if (l.global) {
            try f.before(s, "attn.project", &.{ .{ .name = "xq", .m = xq }, .{ .name = "xk", .m = xk }, .{ .name = "xv", .m = xv } }, pos0);
        } else try f.before(s, "attn.project", &.{.{ .name = "x", .m = xq }}, pos0);
        try f.after(s, "attn.project", &.{
            .{ .name = "q", .m = p.q },       .{ .name = "k", .m = p.k },       .{ .name = "v", .m = p.v },
            .{ .name = "rel", .m = p.rel },   .{ .name = "gate", .m = p.gate }, .{ .name = "sink", .m = p.sink },
            .{ .name = "dk", .m = p.dk },     .{ .name = "dv", .m = p.dv },
        }, attn.project(c, l, xq, xk, xv, p), pos0);
        // K and V convolutions (L4), the stream's state a segment
        var one: [1]Seg = undefined;
        const segs = try f.segments(R, pos0, &one);
        const st = segs[0].st.layer(li);
        const kvc = l.hk * weights.head_dim;
        const kc = a.kc.slice(0, R);
        const vc = a.vc.slice(0, R);
        try f.before(s, "conv.kv", &.{
            .{ .name = "k", .m = p.k },
            .{ .name = "v", .m = p.v },
            .{ .name = "dk", .m = p.dk },
            .{ .name = "dv", .m = p.dv },
            .{ .name = "state_k", .m = Mat.of(st.conv_k, conv.state_rows, kvc, .bf16), .whole = true },
            .{ .name = "state_v", .m = Mat.of(st.conv_v, conv.state_rows, kvc, .bf16), .whole = true },
        }, pos0);
        try f.after(s, "conv.kv", &.{ .{ .name = "kc", .m = kc }, .{ .name = "vc", .m = vc } }, convKv(c, l, li, segs, p, kc, vc), pos0);
        try f.mark(s, "state.conv");
        for (segs) |g| {
            const gs = g.st.layer(li);
            try f.commit(g.spec, gs.conv_k, gs.hist[0], p.k.slice(g.row0, g.rows));
            try f.commit(g.spec, gs.conv_v, gs.hist[1], p.v.slice(g.row0, g.rows));
        }
        // heads (L5-L9) and the cache write (L11)
        const hd: attn.Heads = .{ .q = a.qh.slice(0, R), .k = a.kh.slice(0, R), .v = a.vh.slice(0, R), .rel = a.relh.slice(0, R) };
        try f.before(s, "attn.heads", &.{ .{ .name = "q", .m = p.q }, .{ .name = "kc", .m = kc }, .{ .name = "vc", .m = vc }, .{ .name = "rel", .m = p.rel } }, pos0);
        try f.after(s, "attn.heads", &.{ .{ .name = "q", .m = hd.q }, .{ .name = "k", .m = hd.k }, .{ .name = "v", .m = hd.v }, .{ .name = "rel", .m = hd.rel } }, heads(c, l, segs, p.q, kc, vc, p.rel, hd), pos0);
        try f.mark(s, "state.kv");
        for (segs) |g| try state.kvWrite(c, g.st.kvFor(li, g.rows), g.pos0, g.dev, hd.k.slice(g.row0, g.rows), hd.v.slice(g.row0, g.rows));
        // attention (L10, L12-L14)
        const o = a.o.slice(0, R);
        {
            var n: usize = 4;
            ins[0..4].* = .{ .{ .name = "q", .m = hd.q }, .{ .name = "rel", .m = hd.rel }, .{ .name = "sink", .m = p.sink }, .{ .name = "gate", .m = p.gate } };
            if (f.hooks.record != null) {
                const lo = st.kv.lo(pos0);
                const ctx_rows = pos0 + R - lo;
                const kx = Mat.of(try c.tmp(ctx_rows * kvc * 2), ctx_rows, kvc, .bf16);
                const vx = Mat.of(try c.tmp(ctx_rows * kvc * 2), ctx_rows, kvc, .bf16);
                try state.kvGather(c, st.kv, f.st.ids, lo, pos0 + R, kx, vx);
                ins[4] = .{ .name = "k_ctx", .m = kx, .whole = true };
                ins[5] = .{ .name = "v_ctx", .m = vx, .whole = true };
                n = 6;
            }
            try f.before(s, "attn.core", ins[0..n], pos0);
        }
        try f.after(s, "attn.core", &.{.{ .name = "o", .m = o }}, core(c, l, li, segs, hd.q, hd.rel, p.sink, p.gate, o), pos0);
        // output projection (L15)
        const ao = if (f.x) |x| x.mine(.attn, R) else a.ao.slice(0, R);
        try f.before(s, "attn.out", &.{.{ .name = "o", .m = o }}, pos0);
        try f.after(s, "attn.out", &.{.{ .name = "ao", .m = ao }}, attn.output(c, l, o, ao), pos0);
    }

    /// From the attention sum (W > 1) to the experts: y bf16 at W = 1, else the rank's fp32 partial (its experts).
    pub fn mlpStage(f: *Forward, li: u32, R: usize, pos0: usize) !void {
        const l = f.w.layer(li);
        const a = f.a;
        const c = &f.c;
        const s: Scope = .{ .layer = li };
        const h_in = a.h[0].slice(0, R);
        const ao = a.ao.slice(0, R);
        if (f.x) |x| if (f.unsummed(.attn)) {
            try f.mark(s, "tp.sum");
            c.scratch.reset();
            try x.sum(c, .attn, ao);
        };
        // the post-attention conv (L16)
        var one: [1]Seg = undefined;
        const segs = try f.segments(R, pos0, &one);
        const st = segs[0].st.layer(li);
        const ac = a.a.slice(0, R);
        try f.before(s, "conv.attn", &.{ .{ .name = "x", .m = ao }, .{ .name = "state", .m = Mat.of(st.conv_attn, conv.state_rows, H, .bf16), .whole = true } }, pos0);
        try f.after(s, "conv.attn", &.{.{ .name = "y", .m = ac }}, convPost(c, li, segs, l.attn_conv, ao, .attn, ac), pos0);
        try f.mark(s, "state.conv");
        for (segs) |g| {
            const gs = g.st.layer(li);
            try f.commit(g.spec, gs.conv_attn, gs.hist[2], ao.slice(g.row0, g.rows));
        }
        // the block partial (L17), the MLP input (L18) and its norm
        const part = a.partial.slice(0, R);
        const mx = a.mx.slice(0, R);
        const m = a.m.slice(0, R);
        const ids = a.ids.slice(0, R);
        const wts = a.wts.slice(0, R);
        const lg = a.logits.slice(0, R);
        const fm = f.fuseMode(R);
        if (fm.mlpIn()) {
            // res.partial, the boundary push, res.mlp_input and res.mlp_norm as one launch
            try f.mark(s, "fuse.mlp_in");
            c.scratch.reset();
            if (l.boundary and a.nblocks >= state.max_blocks) return error.TooManyBlocks;
            const push = if (l.boundary) a.block(a.nblocks, R).ptr else 0;
            try fuse.mlpIn(c, h_in, ac, l.boundary, part, a.blockView(R), push, l.mlp_query, mx, l.mlp_norm, m);
            if (l.boundary) a.nblocks += 1;
        } else try f.mlpInput(li, R, pos0, h_in, ac, part, mx, m);
        if (fm.router()) {
            // the router: its logits, then select and the slot order as one launch
            try f.mark(s, "fuse.route");
            c.scratch.reset();
            try fuse.route(c, l, m, ids, wts, lg);
        } else {
            // router (L19)
            try f.before(s, "moe.route", &.{.{ .name = "m", .m = m }}, pos0);
            try f.after(s, "moe.route", &.{ .{ .name = "ids", .m = ids }, .{ .name = "w", .m = wts }, .{ .name = "logits", .m = lg } }, moe.route(c, l, m, ids, wts, lg), pos0);
        }
        try f.before(s, "moe.experts", &.{ .{ .name = "m", .m = m }, .{ .name = "ids", .m = ids }, .{ .name = "w", .m = wts } }, pos0);
        if (f.split()) {
            const x = f.x.?;
            const hm = Mat.of(x.hMine(R), R * moe.topk, moe.intermediate / x.world, .bf16);
            return f.after(s, "moe.experts", &.{.{ .name = "h", .m = hm }}, moe.splitUp(c, l, m, ids, wts, hm.ptr, x.split().?.plan, null), pos0);
        }
        const y = if (f.x) |x| x.mine(.moe, R) else a.y.slice(0, R);
        try f.after(s, "moe.experts", &.{.{ .name = "y", .m = y }}, moe.experts(c, l, m, ids, wts, y), pos0);
    }

    /// The unfused ops from the partial to the router's input: res.partial, the boundary push, res.mlp_input,
    /// res.mlp_norm.
    fn mlpInput(f: *Forward, li: u32, R: usize, pos0: usize, h_in: Mat, ac: Mat, part: Mat, mx: Mat, m: Mat) !void {
        const l = f.w.layer(li);
        const a = f.a;
        const c = &f.c;
        const s: Scope = .{ .layer = li };
        var bt: [state.max_blocks + 1]Tensor = undefined;
        var ins: [16]Tensor = undefined;
        try f.before(s, "res.partial", &.{ .{ .name = "h_in", .m = h_in }, .{ .name = "a", .m = ac } }, pos0);
        try f.after(s, "res.partial", &.{.{ .name = "partial", .m = part }}, res.partial(c, h_in, ac, l.boundary, part), pos0);
        if (l.boundary) {
            try f.mark(s, "state.block");
            try state.pushBlock(c, a, h_in);
        }
        // MLP input (L18)
        {
            const blocks = f.blockTensors(R, &bt);
            @memcpy(ins[0..blocks.len], blocks);
            ins[blocks.len] = .{ .name = "cand", .m = part };
            try f.before(s, "res.mlp_input", ins[0 .. blocks.len + 1], pos0);
            try f.after(s, "res.mlp_input", &.{.{ .name = "x", .m = mx }}, res.mix(c, a.blockView(R), part, l.mlp_query, 1, &.{mx}), pos0);
        }
        try f.before(s, "res.mlp_norm", &.{.{ .name = "x", .m = mx }}, pos0);
        try f.after(s, "res.mlp_norm", &.{.{ .name = "y", .m = m }}, res.norm(c, mx, l.mlp_norm, m), pos0);
    }

    /// Split experts, after MoE-A: the rank's output columns of y from every rank's h (cuda_moe.zig splitDown).
    pub fn downStage(f: *Forward, li: u32, R: usize, pos0: usize) !void {
        const x = f.x orelse return error.NotSplitExperts;
        const sp = x.split() orelse return error.NotSplitExperts;
        const l = f.w.layer(li);
        const s: Scope = .{ .layer = li };
        const ym = Mat.of(x.yMine(R), R, H / x.world, .bf16);
        try f.before(s, "moe.down", &.{}, pos0);
        try f.after(s, "moe.down", &.{.{ .name = "y", .m = ym }}, moe.splitDown(&f.c, l, R, sp.h, sp.plan, ym.ptr, null), pos0);
    }

    /// From the MoE sum (W > 1) to the layer's output in h[0].
    pub fn outStage(f: *Forward, li: u32, R: usize, pos0: usize) !void {
        const l = f.w.layer(li);
        const a = f.a;
        const c = &f.c;
        const s: Scope = .{ .layer = li };
        const h_out = a.h[1].slice(0, R);
        const part = a.partial.slice(0, R);
        // split experts: MoE-B's [W][R][H/W] laid out as rows; one row is already [1][H], read in place
        const in_place = if (f.x) |x| x.split() != null and R == 1 else false;
        const y = if (in_place) Mat.of(f.x.?.split().?.y, 1, H, .bf16) else a.y.slice(0, R);
        if (f.x) |x| if (!in_place and (x.split() != null or f.unsummed(.moe))) {
            try f.mark(s, "tp.sum");
            c.scratch.reset();
            if (x.split() != null) try x.columns(c, y) else try x.sum(c, .moe, y);
        };
        // post-MLP conv (L21) and the layer's output (L22)
        var one: [1]Seg = undefined;
        const segs = try f.segments(R, pos0, &one);
        const st = segs[0].st.layer(li);
        const cm = a.cm.slice(0, R);
        try f.before(s, "conv.mlp", &.{ .{ .name = "x", .m = y }, .{ .name = "state", .m = Mat.of(st.conv_mlp, conv.state_rows, H, .bf16), .whole = true } }, pos0);
        try f.after(s, "conv.mlp", &.{.{ .name = "y", .m = cm }}, convPost(c, li, segs, l.mlp_conv, y, .mlp, cm), pos0);
        try f.mark(s, "state.conv");
        for (segs) |g| {
            const gs = g.st.layer(li);
            try f.commit(g.spec, gs.conv_mlp, gs.hist[3], y.slice(g.row0, g.rows));
        }
        try f.before(s, "res.output", &.{ .{ .name = "partial", .m = part }, .{ .name = "mconv", .m = cm } }, pos0);
        try f.after(s, "res.output", &.{.{ .name = "h", .m = h_out }}, res.output(c, part, cm, h_out), pos0);
        std.mem.swap(Mat, &a.h[0], &a.h[1]);
    }

    /// res.final_input and head.final_norm over the forward's rows: the final rows bf16 [R, H] (in a.xn[0]).
    pub fn final(f: *Forward, R: usize, pos0: usize) !Mat {
        const a = f.a;
        const c = &f.c;
        if (a.nblocks != state.max_blocks) return error.FinalNeedsEveryBlock;
        var bt: [state.max_blocks + 1]Tensor = undefined;
        var ins: [state.max_blocks + 1]Tensor = undefined;
        const blocks = f.blockTensors(R, &bt);
        @memcpy(ins[0..blocks.len], blocks);
        const h = a.h[0].slice(0, R);
        ins[blocks.len] = .{ .name = "cand", .m = h };
        const x = a.x[0].slice(0, R);
        try f.before(.model, "res.final_input", ins[0 .. blocks.len + 1], pos0);
        try f.after(.model, "res.final_input", &.{.{ .name = "x", .m = x }}, res.mix(c, a.blockView(R), h, f.w.output_query, 1, &.{x}), pos0);
        const y = a.xn[0].slice(0, R);
        try f.before(.model, "head.final_norm", &.{.{ .name = "x", .m = x }}, pos0);
        try f.after(.model, "head.final_norm", &.{.{ .name = "y", .m = y }}, head.finalNorm(c, f.w, x, y), pos0);
        return y;
    }

    /// head.unembed of `rows` (final rows, bf16 [n, H]) over rank `rank`'s share of the vocabulary: bf16 logits
    /// into its block of `parts` ([world][n][share], an all-gather's rank order; at W = 1 simply [n, vocab]);
    /// `row0` numbers these rows for the hooks (the runner's selection).
    pub fn unembedShare(f: *Forward, rows: Mat, parts: u64, rank: u32, row0: usize) !void {
        const n = rows.rows;
        const share: usize = f.w.vocab_rows.len();
        const lg = Mat.of(parts + @as(u64, rank) * n * share * 2, n, share, .bf16);
        try f.before(.model, "head.unembed", &.{.{ .name = "y", .m = rows }}, row0);
        try f.after(.model, "head.unembed", &.{.{ .name = "logits", .m = lg }}, head.unembed(&f.c, f.w, rows, lg), row0);
    }

    /// head.softcap over every rank's gathered share (`parts`, see unembedShare): `out` fp32 [n, unpadded]. The
    /// hooks see the bf16 logits as one tensor at W = 1 only.
    pub fn softcapShares(f: *Forward, parts: u64, world: u32, out: Mat, row0: usize) !void {
        const n = out.rows;
        const share: usize = f.w.vocab_rows.len();
        const whole: []const Tensor = &.{.{ .name = "logits", .m = Mat.of(parts, n, share, .bf16) }};
        try f.before(.model, "head.softcap", if (world == 1) whole else &.{}, row0);
        try f.after(.model, "head.softcap", &.{.{ .name = "out", .m = out }}, head.softcapParts(&f.c, parts, world, share, out), row0);
    }
};

/// conv.kv a segment: each stream's K and V rows against its own convolution states.
fn convKv(c: *const Ctx, l: *const weights.Layer, li: u32, segs: []const Seg, p: attn.Proj, kc: Mat, vc: Mat) !void {
    for (segs) |g| {
        const gs = g.st.layer(li);
        const dk = if (l.global) p.dk.slice(g.row0, g.rows) else p.dk;
        const dv = if (l.global) p.dv.slice(g.row0, g.rows) else p.dv;
        try conv.kv(c, l, p.k.slice(g.row0, g.rows), p.v.slice(g.row0, g.rows), dk, dv, gs.conv_k, gs.conv_v, kc.slice(g.row0, g.rows), vc.slice(g.row0, g.rows));
    }
}

/// conv.attn / conv.mlp a segment, against each stream's own state.
fn convPost(c: *const Ctx, li: u32, segs: []const Seg, taps: u64, x: Mat, which: enum { attn, mlp }, out: Mat) !void {
    for (segs) |g| {
        const gs = g.st.layer(li);
        try conv.post(c, taps, x.slice(g.row0, g.rows), if (which == .attn) gs.conv_attn else gs.conv_mlp, out.slice(g.row0, g.rows));
    }
}

/// attn.heads a segment (the global layers' tau reads each row's position).
fn heads(c: *const Ctx, l: *const weights.Layer, segs: []const Seg, q: Mat, kc: Mat, vc: Mat, rel: Mat, out: attn.Heads) !void {
    for (segs) |g| {
        const r = g.row0;
        const n = g.rows;
        try attn.heads(c, l, g.pos0, g.dev, q.slice(r, n), kc.slice(r, n), vc.slice(r, n), rel.slice(r, n), .{ .q = out.q.slice(r, n), .k = out.k.slice(r, n), .v = out.v.slice(r, n), .rel = out.rel.slice(r, n) });
    }
}

/// attn.core a segment, over each stream's own keys and values.
fn core(c: *const Ctx, l: *const weights.Layer, li: u32, segs: []const Seg, q: Mat, rel: Mat, sink: Mat, gate: Mat, out: Mat) !void {
    for (segs) |g| {
        const r = g.row0;
        const n = g.rows;
        // each segment's split scratch from the start: the launches are stream-ordered (in a graph too), so a
        // segment reuses the bytes the one before it is done with, and attn.checkScratch's one-forward need holds
        c.scratch.reset();
        try attn.core(c, l, g.pos0, g.dev, q.slice(r, n), rel.slice(r, n), sink.slice(r, n), gate.slice(r, n), g.st.kvFor(li, g.rows), out.slice(r, n));
    }
}

/// The completed blocks entering layer `l` (the boundaries 0, 6, ... below it).
pub fn blocksBefore(l: u32) usize {
    return (l + 5) / 6;
}

test {
    _ = xrows;
}

test "inkling2 cuda forward: blocks entering each layer" {
    try std.testing.expectEqual(@as(usize, 0), blocksBefore(0));
    try std.testing.expectEqual(@as(usize, 1), blocksBefore(1));
    try std.testing.expectEqual(@as(usize, 1), blocksBefore(6));
    try std.testing.expectEqual(@as(usize, 2), blocksBefore(7));
    try std.testing.expectEqual(@as(usize, 8), blocksBefore(47));
    try std.testing.expectEqual(@as(usize, 8), blocksBefore(48));
}
