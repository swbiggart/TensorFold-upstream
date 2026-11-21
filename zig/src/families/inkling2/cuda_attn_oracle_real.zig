//! tf-inkling2-attn --oracle --capture: the fork's attention cores (v0.1 eager as `fused`, `rows` and `prev`; kx's
//! forms `kx`, `kx-c1`, `kx-split`, `kx-walk`) on one real layer's inputs against relbias_attention_ref.zig, with the
//! synthetic run's report (cuda_attn_oracle.zig) a forward size, and the real-layer verdict (the `REF`
//! line): kx inside its own form (kxForm sliding, kxSplitForm global) at 100%, kx's forms, forward sizes and rank
//! slices byte-equal, the most head-scale steps from the float64 point under the bound registered before any data
//! (1.10 sliding, 1.50 global), kx's p50 and p99 no further from it than v0.1's (`fused`), the inputs inside the
//! bound's preconditions, and no kx exception recompute.
//!
//! The input is an inkling2_attn_check.py core-inputs file: torch's q'', r'', sink, gate and the keys and values of
//! positions 0 .. T - 1 (torch's attn.heads and attn.project outputs; no prep kernel runs here, so this checks the core
//! only: the fork's heads match torch's bit for bit on long_4000 L4/L5, but its project outputs, and so its sink and
//! gate, do not), torch's core output o, the layer's rel_logits_proj.proj and its kind.
//! The prompt runs from position 0 in forwards of N rows for each N of --chunks, on the engine's layouts (sliding: a
//! ring of 1023 + prompt_rows slots written forward by forward; global: a slot a position), at 48/8 and as the four
//! 12/2 rank slices. Every (row, head) is checked as the synthetic run checks it, torch's o too. (e2614fe's path a
//! 16-row block, read from planted runs and predicted by cuda_attn_oracle_real_host.zig, no longer applies: v0.2's kx
//! has no block-wide fallback, and only non-finite inputs reach its per-row path.) Every run's bytes get a SHA-256
//! (DIGEST lines) to compare GPUs. Exit 1 on the synthetic run's rules: eager runs that differ (any form, layout or
//! forward size), rank slices that differ from their unsplit run, kx outside both of its forms, an escape or a refusal.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");
const host = @import("cuda_attn_oracle_host.zig");
const real = @import("cuda_attn_oracle_real_host.zig");
const syn = @import("cuda_attn_oracle.zig");
const rb = @import("core").relbias_attention_ref;

const attn = ic.attn;
const Mat = ic.ops.Mat;
const D = host.D;
const HQ = host.HQ;
const HK = host.HK;
const Impl = syn.Impl;
const impls = syn.impls;
const layouts = syn.layouts;

comptime {
    std.debug.assert(real.prompt_rows == ic.engine.prompt_rows and syn.ring == attn.window - 1 + real.prompt_rows);
}

pub const usage =
    \\usage: tf-inkling2-attn --oracle --capture FILE --layer L [options]   (a real layer's inputs, no model)
    \\  FILE               inkling2_attn_check.py core-inputs' output for layer L
    \\  --chunks N,...     forward sizes from position 0, each 1 to 2048 (default 1,64); kx's path is read at the first
    \\                     multiple of 16
    \\  --first T          the prompt's first T rows (default all)
    \\  --threads N        host threads for the ranges (default 32)
    \\  --check 0|1        the ranges and replays on the host (default 1; 0: GPU runs, bytes and digests only)
    \\  --dump FILE        every run's outputs, bf16 [T, 6144] each, to a safetensors file
    \\  --forms F,...      the cores: v01 (fused, rows, prev), kx (kx, kx-c1, kx-split, kx-walk), or any of those
    \\                     names (default v01,kx)
    \\  --kx-s S           kx's key split (default cuda_attn.zig kx_split_keys)
    \\
;

pub const Opts = struct {
    capture: []const u8 = "",
    layer: ?u32 = null,
    chunks: []const usize = &.{ 1, 64 },
    first: usize = 0,
    threads: usize = 32,
    check: bool = true,
    dump: ?[]const u8 = null,
    forms: std.EnumSet(Impl) = .full,
    s: usize = attn.kx_split_keys,
};

/// The --oracle arguments name a capture.
pub fn wanted(args: []const [:0]const u8) bool {
    for (args) |x| if (std.mem.eql(u8, x, "--capture")) return true;
    return false;
}

pub fn parse(a: std.mem.Allocator, args: []const [:0]const u8) !?Opts {
    var o: Opts = .{};
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        const eql = std.mem.eql;
        if (eql(u8, k, "--capture")) {
            o.capture = v;
        } else if (eql(u8, k, "--layer")) {
            o.layer = try std.fmt.parseInt(u32, v, 10);
        } else if (eql(u8, k, "--chunks")) {
            o.chunks = try syn.ints(a, v);
        } else if (eql(u8, k, "--first")) {
            o.first = try std.fmt.parseInt(usize, v, 10);
        } else if (eql(u8, k, "--threads")) {
            o.threads = try std.fmt.parseInt(usize, v, 10);
        } else if (eql(u8, k, "--check")) {
            o.check = !eql(u8, v, "0");
        } else if (eql(u8, k, "--dump")) {
            o.dump = v;
        } else if (eql(u8, k, "--kx-s")) {
            o.s = try std.fmt.parseInt(usize, v, 10);
            if (std.mem.indexOfScalar(usize, &attn.kx_splits, o.s) == null) return null;
        } else if (eql(u8, k, "--forms")) {
            o.forms = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| {
                if (eql(u8, x, "v01")) {
                    for ([_]Impl{ .fused, .rows, .prev }) |m| o.forms.insert(m);
                } else if (eql(u8, x, "kx")) {
                    for (impls) |m| if (syn.isKx(m)) o.forms.insert(m);
                } else o.forms.insert(std.meta.stringToEnum(Impl, x) orelse return null);
            }
        } else return null;
    }
    if (i != args.len or o.capture.len == 0 or o.layer == null or o.threads == 0 or o.chunks.len == 0) return null;
    for (o.chunks) |n| if (!real.chunkOk(n)) {
        std.log.err("--chunks {d}: a forward is 1 to {d} rows (the engine's prompt_rows: the sliding ring keeps 1023 + {d} slots, and a longer forward overwrites keys its own rows attend)", .{ n, real.prompt_rows, real.prompt_rows });
        return null;
    };
    return o;
}

/// The eager plans a run took, as cuda_attn_oracle.zig prints them.
const Plans = struct {
    text: [24][48]u8 = undefined,
    len: [24]usize = undefined,
    n: usize = 0,

    fn add(p: *Plans, s: []const u8) void {
        for (0..p.n) |i| if (std.mem.eql(u8, p.text[i][0..p.len[i]], s)) return;
        if (p.n == p.text.len) return;
        @memcpy(p.text[p.n][0..s.len], s);
        p.len[p.n] = s.len;
        p.n += 1;
    }
};

/// An implementation on the unsplit layout or the four slices put together, at a forward size; or torch's o (impl
/// null, chunk 0), or a kx run with a planted row a block (impl null, `forced`). Outputs [T][HQ][D].
const Run = struct {
    name: []const u8,
    impl: ?Impl,
    split: bool = false,
    chunk: usize,
    forced: ?usize = null,
    out: []u16,
    plans: Plans = .{},
    acc: host.Acc = .{},

    fn checked(r: *const Run) bool {
        return r.forced == null;
    }
};

const Dev = struct {
    q: [layouts]Mat,
    rel: [layouts]Mat,
    sink: [layouts]Mat,
    gate: [layouts]Mat,
    k: [layouts]u64,
    v: [layouts]u64,
    out: [layouts][impls.len]Mat,
    qf: Mat,
    proj: u64,
    slots: usize,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn buf(x: *Dev, gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        try b.fill8(0, null);
        try x.own.append(gpa, b);
        return b.ptr;
    }

    fn mat(x: *Dev, gpa: std.mem.Allocator, d: *const cuda.Driver, rows: usize, cols: usize) !Mat {
        return Mat.of(try x.buf(gpa, d, rows * cols * 2), rows, cols, .bf16);
    }

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, T: usize, kind: host.Kind) !Dev {
        var x: Dev = undefined;
        x.own = .empty;
        errdefer x.deinit(gpa);
        x.slots = if (kind == .global) T else syn.ring;
        x.proj = try x.buf(gpa, d, rb.d_rel * host.E * 2);
        x.qf = try x.mat(gpa, d, T, HQ * D);
        for (0..layouts) |li| {
            const s = syn.shape(li);
            x.q[li] = try x.mat(gpa, d, T, s[0] * D);
            x.rel[li] = try x.mat(gpa, d, T, s[0] * rb.d_rel);
            x.sink[li] = try x.mat(gpa, d, T, s[0]);
            x.gate[li] = try x.mat(gpa, d, T, s[0] * D);
            x.k[li] = try x.buf(gpa, d, x.slots * s[1] * D * 2);
            x.v[li] = try x.buf(gpa, d, x.slots * s[1] * D * 2);
            for (&x.out[li]) |*m| m.* = try x.mat(gpa, d, T, s[0] * D);
        }
        return x;
    }

    fn deinit(x: *Dev, gpa: std.mem.Allocator) void {
        for (x.own.items) |*b| b.free();
        x.own.deinit(gpa);
    }

    fn kv(x: *const Dev, li: usize, kind: host.Kind) attn.Kv {
        return .{ .k = x.k[li], .v = x.v[li], .slots = x.slots, .heads = syn.shape(li)[1], .window = if (kind == .global) 0 else attn.window };
    }
};

/// Layer L's tensor `t` of the core's inputs (or `o`, its output) from the capture: its first T rows into out.
fn read(r: *const ink.st.Reader, L: u32, t: []const u8, T: usize, cols: usize, out: []u16) !void {
    var nb: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, "op.{d}.attn.core.{s}.{s}", .{ L, if (std.mem.eql(u8, t, "o")) "out" else "in", t });
    const e = r.find(name) orelse {
        std.log.err("the capture has no {s} (make it with inkling2_attn_check.py core-inputs)", .{name});
        return error.MissingTensor;
    };
    if (e.dtype != .bf16 or e.rank != 2 or e.dim(1) != cols or e.dim(0) < T) return error.UnexpectedCaptureTensor;
    try r.readInto(e, 0, std.mem.sliceAsBytes(out[0 .. T * cols]));
}

const State = struct {
    gpa: std.mem.Allocator,
    o: Opts,
    T: usize,
    kind: host.Kind,
    in: host.Inputs,
    c: ic.ops.Ctx,
    k: *ic.kernels.Kernels,
    scratch: *ic.ops.Scratch,
    kx: bool,
    dev: Dev,
    stage: []u16,
    tmp: []u16,
    runs: std.ArrayList(Run) = .empty,
    preds: []real.Pred = &.{},
    flags: []u8 = &.{},
    failed: usize = 0,

    fn find(st: *State, impl: Impl, split: bool, chunk: usize) *Run {
        return st.maybe(impl, split, chunk) orelse unreachable;
    }

    fn maybe(st: *State, impl: Impl, split: bool, chunk: usize) ?*Run {
        for (st.runs.items) |*r| if (r.impl == impl and r.split == split and r.chunk == chunk) return r;
        return null;
    }

    /// The run has this implementation (asked for, and kx's built when it is one of kx's forms).
    fn want(st: *const State, impl: Impl) bool {
        return st.o.forms.contains(impl) and (st.kx or !syn.isKx(impl));
    }

    fn forcedRun(st: *State, v: usize) ?*Run {
        for (st.runs.items) |*r| if (r.forced == v) return r;
        return null;
    }

    fn putCols(st: *State, li: usize) !void {
        const s = syn.shape(li);
        const h0 = s[2] * s[0];
        const x = &st.dev;
        for ([_]struct { []const u16, usize, Mat }{ .{ st.in.q, D, x.q[li] }, .{ st.in.rel, rb.d_rel, x.rel[li] }, .{ st.in.sink, 1, x.sink[li] }, .{ st.in.gate, D, x.gate[li] } }) |t| {
            const w = s[0] * t[1];
            for (0..st.T) |r| @memcpy(st.stage[r * w ..][0..w], t[0][(r * HQ + h0) * t[1] ..][0..w]);
            try syn.put(st.c.d, t[2].ptr, std.mem.sliceAsBytes(st.stage[0 .. st.T * w]));
        }
    }

    /// Positions p0 .. p0 + R - 1's keys and values into layout li's cache, as the engine writes a forward's.
    fn putKv(st: *State, li: usize, p0: usize, R: usize) !void {
        var view = st.in;
        view.from = p0;
        view.npos = R;
        const kv = st.dev.kv(li, st.kind);
        try syn.putCache(st.c.d, &view, st.in.k[p0 * D ..], kv.k, kv, li, st.stage);
        try syn.putCache(st.c.d, &view, st.in.v[p0 * D ..], kv.v, kv, li, st.stage);
    }

    /// The prompt in forwards of N rows: every implementation on every layout, or (forced v) kx on 48/8 with qf.
    fn pass(st: *State, N: usize, forced: ?usize) !void {
        const x = &st.dev;
        var p0: usize = 0;
        while (p0 < st.T) : (p0 += @min(N, st.T)) {
            const R = @min(N, st.T - p0);
            try st.c.stream.synchronize(); // the uploads below are not ordered with the stream's kernels
            for (0..layouts) |li| if (forced == null or li == 0) try st.putKv(li, p0, R);
            for (0..layouts) |li| for (impls) |impl| {
                if (!st.want(impl) or (forced != null and (li != 0 or impl != .kx))) continue;
                const q = if (forced != null) x.qf else x.q[li];
                const plan = try syn.runCore(&st.c, st.k, st.scratch, impl, st.o.s, li, st.kind, x.proj, x.kv(li, st.kind), p0, q.slice(p0, R), x.rel[li].slice(p0, R), x.sink[li].slice(p0, R), x.gate[li].slice(p0, R), x.out[li][syn.ix(impl)].slice(p0, R));
                var pb: [48]u8 = undefined;
                if (forced == null) st.find(impl, li > 0, N).plans.add(syn.planText(&pb, plan));
            };
        }
        try st.c.stream.synchronize();
        for (0..layouts) |li| for (impls) |impl| {
            if (!st.want(impl) or (forced != null and (li != 0 or impl != .kx))) continue;
            const s = syn.shape(li);
            const w = s[0] * D;
            try cuda.memory.download(st.c.d, x.out[li][syn.ix(impl)].ptr, std.mem.sliceAsBytes(st.tmp[0 .. st.T * w]));
            const dst = if (forced) |v| st.forcedRun(v).?.out else st.find(impl, li > 0, N).out;
            for (0..st.T) |r| @memcpy(dst[(r * HQ + s[2] * s[0]) * D ..][0..w], st.tmp[r * w ..][0..w]);
        };
    }
};

/// The host check of every (row, head): its ranges against every checked run, kx's own forms and its predicted path.
const Check = struct {
    st: *const State,
    work: []rb.Work,
    scores: [][]f64,
    accs: []host.Acc,
    obs: ?usize,
    n: usize, // the forward size the predictions are for: kx's path read (`obs`)

    fn item(c: *const Check, i: usize, w: usize) void {
        const st = c.st;
        const t = i / HQ;
        const h = i % HQ;
        var r: host.Ref = .{};
        r.make(st.in.head(t, h), st.in.geo(), st.kx, st.o.s, &c.work[w]);
        const n = st.runs.items.len;
        for (st.runs.items, 0..) |x, ri| if (x.checked()) {
            var a: host.Acc = .{};
            r.check(x.out[i * D ..][0..D], if (x.impl) |im| syn.isKx(im) else false, &a);
            if (c.obs == ri) st.flags[i] = @as(u8, @intFromBool(a.kx_in[0] == D)) | @as(u8, @intFromBool(a.kx_in[1] == D)) << 1;
            c.accs[w * n + ri].add(a);
        };
        if (st.kx and c.obs != null) st.preds[i] = real.kxRow(&st.in, t, h, c.scores[w], c.n, st.dev.slots);
    }
};

fn pct(a: usize, b: usize) f64 {
    return if (b == 0) 0 else 100 * @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
}

fn report(r: *const Run, label: []const u8, L: u32, kind: host.Kind) void {
    const a = r.acc;
    var pb: [24 * 49]u8 = undefined;
    var w: std.Io.Writer = .fixed(&pb);
    for (0..r.plans.n) |i| w.print("{s}{s}", .{ if (i > 0) "," else "", r.plans.text[i][0..r.plans.len[i]] }) catch {};
    std.debug.print("RUN\t{s}\t{d}\t{t}\t{s}\t{d}\t{d}\t{d:.3}\t{d:.3}\t{d:.2}\t{d:.2}\t{d:.3}\t{d:.3}\t{d:.2}\t{d}/{d}/{d}\t{d}\t{d}\t{d}\t{s}\n", .{ label, L, kind, r.name, a.outputs, a.inside, pct(a.inside, a.outputs), pct(a.kx_in[2], a.outputs), a.hs_out, a.hs_pt, pct(a.eager_eq, a.outputs), pct(a.stats.ranged, a.outputs), a.eager_hs, a.hist[0] + a.hist[1], a.hist[2] + a.hist[3], a.hist[4] + a.hist[5], a.class_bad, a.escape, a.refused, w.buffered() });
    std.debug.print("JSON {{\"prompt\":\"{s}\",\"layer\":{d},\"kind\":\"{t}\",\"run\":\"{s}\",\"chunk\":{d},\"outputs\":{d},\"inside_replay\":{d},\"inside_kxform\":{d},\"inside_kxfallback\":{d},\"inside_kx_either\":{d},\"decided\":{d},\"ranged\":{d},\"nonfinite\":{d},\"max_steps_outside\":{d},\"max_ulps32_outside\":{d:.3},\"max_head_steps_outside\":{d:.3},\"max_steps_from_point\":{d},\"max_head_steps_from_point\":{d:.3},\"class_mismatch\":{d},\"got_nonfinite\":{d},\"eager_equal\":{d},\"eager_max_steps\":{d},\"eager_max_head_steps\":{d:.3},\"escape\":{d},\"refused\":{d},", .{ label, L, kind, r.name, r.chunk, a.outputs, a.inside, a.kx_in[0], a.kx_in[1], a.kx_in[2], a.stats.decided, a.stats.ranged, a.stats.nonfinite, a.out_steps, a.out_ulps, a.hs_out, a.pt_steps, a.hs_pt, a.class_bad, a.got_nonfinite, a.eager_eq, a.eager_steps, a.eager_hs, a.escape, a.refused });
    std.debug.print("\"hist_out_head_steps\":[{d},{d},{d},{d},{d},{d}],\"plans\":\"{s}\"}}\n", .{ a.hist[0], a.hist[1], a.hist[2], a.hist[3], a.hist[4], a.hist[5], w.buffered() });
}

/// Byte comparisons between runs; the failures they count. Returns the (row, head) vectors of kx's runs (every form,
/// forward size and layout) that differ from kx's unsplit run at the first size.
fn compare(st: *State) usize {
    const T = st.T;
    const n = T * HQ;
    const o = st.o;
    var eref: ?Impl = null;
    for ([_]Impl{ .fused, .rows, .prev }) |m| if (eref == null and st.want(m)) {
        eref = m;
    };
    var differ: usize = 0;
    if (eref) |er| {
        const ref = st.find(er, false, o.chunks[0]).out;
        for (st.runs.items) |r| if (r.impl) |impl| if (!syn.isKx(impl)) {
            const d = host.vectorsDiffer(ref, r.out, n);
            differ += d;
            std.debug.print("BYTES\teager\t{s}\tvs {t}.48x8.{d}\t{d} of {d} vectors differ\n", .{ r.name, er, o.chunks[0], d, n });
        };
    }
    var slices: usize = 0;
    for (impls) |impl| if (st.want(impl)) for (o.chunks) |N| {
        const d = host.vectorsDiffer(st.find(impl, false, N).out, st.find(impl, true, N).out, n);
        slices += d;
        std.debug.print("BYTES\tslices\t{t}.{d}\t12/2x4 vs 48/8\t{d} of {d} vectors differ\n", .{ impl, N, d, n });
    };
    st.failed += host.byteFailures(differ, &.{slices});
    var kx_differ: usize = 0;
    if (st.want(.kx)) for (st.runs.items) |r| if (r.impl) |impl| if (syn.isKx(impl)) {
        const d = host.vectorsDiffer(st.find(.kx, false, o.chunks[0]).out, r.out, n);
        kx_differ += d;
        std.debug.print("BYTES\tkx\t{s}\tvs kx.48x8.{d}\t{d} of {d} vectors differ\n", .{ r.name, o.chunks[0], d, n });
    };
    if (kx_differ > 0) st.failed += 1;
    const torch = st.runs.items[st.runs.items.len - 1].out;
    var eq: usize = 0;
    if (eref) |er| {
        const ref = st.find(er, false, o.chunks[0]).out;
        for (torch, ref) |a, b| eq += @intFromBool(a == b);
        std.debug.print("BYTES\ttorch\ttorch.o\tvs {t}.48x8.{d}\t{d} of {d} vectors differ; {d} of {d} outputs equal\n", .{ er, o.chunks[0], host.vectorsDiffer(ref, torch, n), n, eq, n * D });
    }
    if (st.want(.kx)) {
        const kr = st.find(.kx, false, o.chunks[0]).out;
        eq = 0;
        for (torch, kr) |a, b| eq += @intFromBool(a == b);
        std.debug.print("BYTES\ttorch\ttorch.o\tvs kx.48x8.{d}\t{d} of {d} vectors differ; {d} of {d} outputs equal\n", .{ o.chunks[0], host.vectorsDiffer(kr, torch, n), n, eq, n * D });
    }
    return kx_differ;
}

/// Inputs outside the real-layer bound's domain: subnormal or non-finite q'', r'', sink, gate, keys or values.
fn badInputs(in: *const host.Inputs, T: usize) usize {
    var bad: usize = 0;
    for ([_][]const u16{ in.q[0 .. T * HQ * D], in.rel[0 .. T * HQ * rb.d_rel], in.sink[0 .. T * HQ], in.gate[0 .. T * HQ * D], in.k, in.v }) |xs| for (xs) |x| {
        bad += @intFromBool(x & 0x7f80 == 0x7f80 or (x & 0x7f80 == 0 and x & 0x7f != 0));
    };
    return bad;
}

/// The real-layer verdict for this layer: one `REF` line.
fn refLine(st: *State, L: u32, kx_differ: usize, rows0: bool) void {
    var kxa: host.Acc = .{};
    var nk: usize = 0;
    for (st.runs.items) |r| if (r.impl) |impl| if (syn.isKx(impl)) {
        kxa.add(r.acc);
        nk += 1;
    };
    const kx0 = st.find(.kx, false, st.o.chunks[0]);
    const v01 = st.maybe(.fused, false, st.o.chunks[0]);
    const B: f64 = if (st.kind == .global) 1.50 else 1.10;
    const own = pct(kxa.kx_in[0], kxa.outputs);
    const p50 = [2]f64{ kx0.acc.quantile(0.5), if (v01) |r| r.acc.quantile(0.5) else std.math.inf(f64) };
    const p99 = [2]f64{ kx0.acc.quantile(0.99), if (v01) |r| r.acc.quantile(0.99) else std.math.inf(f64) };
    const bad_in = badInputs(&st.in, st.T);
    const pre = st.T <= 128000 and kxa.smax <= 256 and bad_in == 0;
    const ok = kxa.kx_in[0] == kxa.outputs and kxa.class_bad == 0 and kxa.refused == 0 and kxa.escape == 0 and kxa.hs_pt <= B and p50[0] <= p50[1] and p99[0] <= p99[1] and pre and kx_differ == 0 and rows0;
    if (!ok) st.failed += 1;
    std.debug.print("REF L{d} kx: inside own {d:.2}%, max {d:.3} <= B {d:.2}, p50 {d:.4} <= {d:.4}, p99 {d:.4} <= {d:.4}; inside replayForm {d:.3}%; class mismatches {d}, refused {d}, escapes {d}; {d} kx runs byte-equal ({d} vectors differ); preconditions: positions < {d}, max |s| {d:.1} <= 256, {d} subnormal or non-finite inputs; exception recomputes {s}: {s}\n", .{ L, own, kxa.hs_pt, B, p50[0], p50[1], p99[0], p99[1], pct(kxa.inside, kxa.outputs), kxa.class_bad, kxa.refused, kxa.escape, nk, kx_differ, st.T, kxa.smax, bad_in, if (rows0) "0" else "NOT 0", if (ok) "PASS" else "FAIL" });
}

/// kx's path a (16-row block, head) at forward size N (a multiple of 16): read from the planted runs, predicted
/// from the inputs when the host check ran, and kx inside the form of the path it took.
fn paths(st: *State, N: usize) void {
    const T = st.T;
    const nat = st.find(.kx, false, N).out;
    const f = [2][]const u16{ st.forcedRun(0).?.out, st.forcedRun(1).?.out };
    var seen: [3]usize = @splat(0);
    var cross: [3][4]usize = @splat(@splat(0)); // seen x (q or kv test, limb sure, limb maybe, none)
    var own: [2][2]usize = @splat(@splat(0)); // (fast, fallback) x (vectors, inside the path's form)
    const nb = (T + 15) / 16;
    for (0..nb) |b| for (0..HQ) |h| {
        const s = real.seen(nat, f, T, b, h);
        seen[@backingInt(s)] += 1;
        if (!st.o.check) continue;
        var p: real.Pred = .{};
        for (16 * b..@min(16 * b + 16, T)) |t| p.join(st.preds[t * HQ + h]);
        cross[@backingInt(s)][if (p.q_bad or p.kv_bad) 0 else if (p.limb == .sure) @as(usize, 1) else if (p.limb == .maybe) 2 else 3] += 1;
        if (s == .unseen) continue;
        const k = @backingInt(s);
        for (16 * b..@min(16 * b + 16, T)) |t| {
            own[k][0] += 1;
            own[k][1] += @intFromBool(st.flags[t * HQ + h] >> @intCast(k) & 1 == 1);
        }
    };
    std.debug.print("KXPATH\tforwards of {d}\t(block, head) pairs {d}: fast {d}, fallback (or fast with the fallback's bytes) {d}, unseen {d}\n", .{ N, nb * HQ, seen[0], seen[1], seen[2] });
    if (!st.o.check) return;
    for (cross, [_][]const u8{ "fast", "fallback", "unseen" }) |row, name| std.debug.print("KXPATH\tseen {s}\tpredicted: q/kv test {d}, limb sure {d}, limb maybe {d}, none {d}\n", .{ name, row[0], row[1], row[2], row[3] });
    std.debug.print("KXPATH\town form\tfast blocks: {d}/{d} vectors inside kxForm; fallback blocks: {d}/{d} inside kxFallbackForm\n", .{ own[0][1], own[0][0], own[1][1], own[1][0] });
    var lo: f64 = 0;
    var deep: [5]usize = @splat(0);
    for (st.preds) |p| {
        lo = @min(lo, p.xmin);
        for ([_]f64{ -20, -40, -60, -70.7, -87.3 }, &deep) |edge, *c| c.* += @intFromBool(p.xmin < edge);
    }
    std.debug.print("KXPATH\tscores\tlowest s - m over every (row, head) {d:.2}; (row, head) pairs with a key below m - 20 / 40 / 60 / 70.7 / 87.3: {d} / {d} / {d} / {d} / {d} of {d}\n", .{ lo, deep[0], deep[1], deep[2], deep[3], deep[4], st.preds.len });
    if (st.o.chunks[0] == 1 and N != 1) { // single-row forwards (decode) against the blocks
        const one = st.find(.kx, false, 1).out;
        var moved: [4]usize = @splat(0);
        for (0..T * HQ) |i| if (!std.mem.eql(u16, one[i * D ..][0..D], nat[i * D ..][0..D])) {
            const p = st.preds[i];
            moved[if (p.q_bad or p.kv_bad) 0 else if (p.limb == .sure) @as(usize, 1) else if (p.limb == .maybe) 2 else 3] += 1;
        };
        std.debug.print("KXPATH\tforwards of 1 vs {d}\tvectors that differ, by the row's own prediction: q/kv test {d}, limb sure {d}, limb maybe {d}, none {d}\n", .{ N, moved[0], moved[1], moved[2], moved[3] });
    }
}

fn hex(out: []const u16) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(out), &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    const L = o.layer.?;
    var cap = try ink.st.Reader.open(gpa, io, o.capture);
    defer cap.close();
    const kind = std.meta.stringToEnum(host.Kind, cap.metaValue("kind") orelse "") orelse return error.CaptureHasNoKind;
    if (cap.metaValue("layer")) |l| if (try std.fmt.parseInt(u32, l, 10) != L) return error.CaptureOfAnotherLayer;
    const oracle = cap.metaValue("oracle") orelse "";
    const label = if (std.mem.endsWith(u8, oracle, ".safetensors")) oracle[0 .. oracle.len - 12] else oracle;
    var nb: [96]u8 = undefined;
    const qe = cap.find(try std.fmt.bufPrint(&nb, "op.{d}.attn.core.in.q", .{L})) orelse return error.MissingTensor;
    const T: usize = if (o.first == 0) @intCast(qe.dim(0)) else @min(o.first, @as(usize, @intCast(qe.dim(0))));
    var in = try host.Inputs.alloc(gpa, T, T);
    defer in.free(gpa);
    in.profile = .clip; // unused: the inputs are the capture's
    in.kind = kind;
    in.p0 = 0;
    in.R = T;
    in.from = 0;
    in.npos = T;
    const torch = try gpa.alloc(u16, T * HQ * D);
    defer gpa.free(torch);
    const tmp = try gpa.alloc(u16, T * HQ * D);
    defer gpa.free(tmp);
    try read(&cap, L, "q", T, HQ * D, in.q);
    try read(&cap, L, "rel", T, HQ * rb.d_rel, in.rel);
    try read(&cap, L, "sink", T, HQ, in.sink);
    try read(&cap, L, "gate", T, HQ * D, in.gate);
    try read(&cap, L, "proj", rb.d_rel, host.E, in.proj);
    try read(&cap, L, "o", T, HQ * D, torch);
    try read(&cap, L, "k_ctx@0", T, HK * D, tmp);
    real.transpose(tmp, T, T, in.k);
    try read(&cap, L, "v_ctx@0", T, HK * D, tmp);
    real.transpose(tmp, T, T, in.v);

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var k = try ic.kernels.Kernels.load(&driver);
    defer k.unload();
    var gemms = try ic.ops.Gemms.init(gpa, &driver);
    defer gemms.deinit();
    var most: usize = 1;
    for (o.chunks) |N| most = @max(most, @min(N, T));
    const kx = k.attn.kx != null;
    k.attn.kxo = .{ .s = o.s };
    var cb: [160]u8 = undefined;
    std.debug.print("{s}\n", .{k.attn.census(&cb)});
    const form = k.attn.form;
    k.attn.form = .v01;
    var need = @max(attn.kx_split_cap, attn.kxNeed(.{ .s = o.s }, HQ, T, most, 0), attn.kxNeed(.{ .s = o.s }, HQ / 4, T, most, 0));
    for ([_][2]usize{ .{ HQ, HK }, .{ HQ / 4, HK / 4 } }) |s| need = @max(need, attn.scratchNeed(&k.attn, s[0], s[1], T, most, 0));
    k.attn.form = form;
    var scratch = try ic.ops.Scratch.init(&driver, need + (16 << 20));
    defer scratch.deinit();
    var st: State = .{ .gpa = gpa, .o = o, .T = T, .kind = kind, .in = in, .c = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch }, .k = &k, .scratch = &scratch, .kx = kx, .dev = try Dev.init(gpa, &driver, T, kind), .stage = try gpa.alloc(u16, T * HQ * D), .tmp = tmp };
    defer st.dev.deinit(gpa);
    defer gpa.free(st.stage);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    defer st.runs.deinit(gpa);
    defer for (st.runs.items) |r| if (r.impl != null or r.forced != null) gpa.free(r.out);
    // e2614fe's path reading (planted runs) does not apply to v0.2's kx: no forced runs
    const obs: ?usize = null;
    for (impls) |impl| if (st.want(impl)) for ([_]bool{ false, true }) |split| for (o.chunks) |N| {
        const name = try std.fmt.allocPrint(aa, "{t}.{s}.{d}", .{ impl, if (split) "12x2x4" else "48x8", N });
        try st.runs.append(gpa, .{ .name = name, .impl = impl, .split = split, .chunk = N, .out = try gpa.alloc(u16, T * HQ * D) });
    };
    if (obs) |N| for (0..2) |v| try st.runs.append(gpa, .{ .name = try std.fmt.allocPrint(aa, "kx.forced{d}.{d}", .{ v, N }), .impl = null, .chunk = N, .forced = v, .out = try gpa.alloc(u16, T * HQ * D) });
    try st.runs.append(gpa, .{ .name = "torch.o", .impl = null, .chunk = 0, .out = torch });

    std.debug.print("oracle --capture: {s} layer {d} ({t}), {d} rows; kx {s} (S {d}); chunks", .{ label, L, kind, T, if (kx) "loaded" else "not built", o.s });
    for (o.chunks) |N| std.debug.print(" {d}", .{N});
    std.debug.print("; kx path read at {?d}; scratch {d:.1} MiB; slots {d}\n", .{ obs, @as(f64, @floatFromInt(need)) / (1 << 20), st.dev.slots });
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(io).toNanoseconds();
    try syn.put(&driver, st.dev.proj, std.mem.sliceAsBytes(in.proj));
    for (0..layouts) |li| try st.putCols(li);
    for (o.chunks) |N| try st.pass(N, null);
    if (obs) |N| for (0..2) |v| {
        @memcpy(st.stage, in.q);
        real.plant(st.stage, T, v);
        try stream.synchronize();
        try syn.put(&driver, st.dev.qf.ptr, std.mem.sliceAsBytes(st.stage));
        try st.pass(N, v);
    };
    const t1 = clock.now(io).toNanoseconds();
    std.debug.print("GPU runs {d:.1} s\n", .{@as(f64, @floatFromInt(t1 - t0)) * 1e-9});
    for (st.runs.items) |r| std.debug.print("DIGEST\t{s}\t{d}\t{s}\t{s}\n", .{ label, L, r.name, hex(r.out) });
    if (o.dump) |path| {
        var wr = ink.st.Writer.init(gpa);
        defer wr.deinit();
        for (st.runs.items) |r| try wr.add(r.name, .bf16, &.{ T, HQ * D }, std.mem.sliceAsBytes(r.out));
        try wr.setMeta("tool", "tf-inkling2-attn --oracle --capture");
        try wr.setMeta("capture", std.fs.path.basename(o.capture));
        try wr.write(io, path);
        std.debug.print("wrote {s}\n", .{path});
    }
    const kx_differ = compare(&st);
    const ex = try k.attn.kxExceptions(&driver);
    if (o.check) {
        const nt = @min(o.threads, 128);
        st.preds = try gpa.alloc(real.Pred, T * HQ);
        defer gpa.free(st.preds);
        st.flags = try gpa.alloc(u8, T * HQ);
        defer gpa.free(st.flags);
        @memset(st.flags, 0);
        const work = try gpa.alloc(rb.Work, nt);
        defer gpa.free(work);
        for (work) |*wk| wk.* = try rb.Work.init(gpa, T + 1);
        defer for (work) |*wk| wk.deinit(gpa);
        const scores = try gpa.alloc([]f64, nt);
        defer gpa.free(scores);
        for (scores) |*s| s.* = try gpa.alloc(f64, T + 64);
        defer for (scores) |s| gpa.free(s);
        const accs = try gpa.alloc(host.Acc, nt * st.runs.items.len);
        defer gpa.free(accs);
        @memset(accs, .{});
        var oi: ?usize = null;
        if (obs) |N| for (st.runs.items, 0..) |r, ri| if (r.impl == .kx and !r.split and r.chunk == N) {
            oi = ri;
        };
        const c: Check = .{ .st = &st, .work = work, .scores = scores, .accs = accs, .obs = oi, .n = obs orelse T };
        try syn.parallel(nt, T * HQ, &c, Check.item);
        for (st.runs.items, 0..) |*r, ri| for (0..nt) |w| r.acc.add(accs[w * st.runs.items.len + ri]);
        std.debug.print("RUN\tprompt\tlayer\tkind\trun\toutputs\tinside_replay\tpct\tkx_either_pct\tmax_head_steps_out\tmax_head_steps_pt\teager_equal_pct\tranged_pct\teager_max_head_steps\toutside_by_head_steps(<=1/1-4/>4)\tclass_bad\tescape\trefused\tplans\n", .{});
        for (st.runs.items) |*r| if (r.checked()) {
            report(r, label, L, kind);
            const a = r.acc;
            if (a.escape > 0 or a.refused > 0) st.failed += 1;
            if (r.impl) |im| if (syn.isKx(im) and a.kx_in[0] != a.outputs) {
                st.failed += 1;
            };
        };
        if (obs) |N| paths(&st, N);
        if (st.want(.kx)) refLine(&st, L, kx_differ, ex[0] == 0);
        std.debug.print("host checks {d:.1} s\n", .{@as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t1)) * 1e-9});
    } else if (obs) |N| paths(&st, N);
    k.attn.printExceptions(&driver);
    std.debug.print("{s} oracle --capture {s} layer {d}: {d} failed checks\n", .{ if (st.failed == 0) "PASS" else "FAIL", label, L, st.failed });
    return if (st.failed == 0) 0 else 1;
}
