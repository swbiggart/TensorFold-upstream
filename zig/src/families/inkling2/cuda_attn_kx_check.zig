//! tf-inkling2-attn --op-check --forms kx, --bench --forms kx[,v01], --limb: the kx core's forms (cuda_attn.zig
//! `coreKx`, inkling2_kx_fp32.cuh) on pseudo-random inputs at a rank's head shape.
//!
//! --op-check --forms kx, for each S of --kx-s: every forward (kind x position x rows) through every variant, each
//! global one also on the pool's pages (NaN until written, page ids shuffled, the table's unused entries on a NaN scrub
//! page), and every row of every run against `kx-solo` (the row alone at its own position, R = 1, default dispatch):
//!   kx           the forward's default dispatch
//!   kx-tile      C 1 (sliding: kx-tile; global: kx-split C 1 or kx-split-walk)
//!   kx-dec2/4    C 2 / 4
//!   kx-split     global, kx-split wherever its scratch fits (G 128); -g7 (7 splits a grid column: many rounds a
//!                CTA), -c1, -c4
//!   kx-walk      global, kx-split-walk at every R
//!   kx-devpos    eager launches of the captured plan (the position read on the device, the grid over the slot's
//!                splits), up to 8 rows; -g7, -c1, -unpack (was `kx-graph`: no CUDA graph is involved)
//!   kx-graph     the captured plan as the engine runs it (Engine.graphStep): `core` captured from the stream into a
//!                CUDA graph and instantiated once a (variant, kind, keys, R), its inputs and output at fixed
//!                addresses, then replayed at every later forward's position (the inputs copied in and the position
//!                written on the stream first, as a step's header and projections land before its replay); -g7, -c1,
//!                -unpack
//!   kx-nosplit   global keys through the tile kernel (window 0, no split): only where pos0 + R <= S
//!   kx-unpack    the default dispatch with GQA packing off (the query heads of a kv head in separate CTAs);
//!                kx-pack: packing on at every R
//!   e2614fe      kortexa-ai e2614fe's tf_ik_attention (ring and contiguous keys): rows equal, information only
//! Every row byte-equal (NaN canonical), the pool's other runs still NaN afterwards, and `attn exceptions: rows 0`
//! (the NaN pool and the scrub page are never loaded). Inputs are per position (a row sees the same q', rel', sink and
//! gate in every forward), held only at the positions the forwards read.
//!
//! Then the graph walks (`graphWalk`, inputs of their own): each kx-graph variant at rows 1 and 8 captured once and
//! replayed at every position across tile, page, split and grid-wrap boundaries (global; sliding: the window's edge and
//! the ring's wraps), on the contiguous keys, the pool's pages, a ring view of a run of pages (one row, as
//! State.kvFor gives a one-row forward: recorded by Ctx.launchOn into a RunLog, its nodes found and repointed after
//! the run moves as cuda_engine.zig's Graph.follow does) and a page table that moves; every replay's rows against
//! kx-solo and against an eager launch of the same form. Planted controls that must fail: a replay with a stale
//! device position, with the K/V pointers left on the old run, and with the page table left on the old pages.
//!
//! --bench --forms kx[,v01]: `core` alone, the variants alternated in rounds; a time is a layer's (global layers as
//! a served step: `--layers` layers a call, each its own pages), median, p10 and p90 over the rounds. Global layers:
//! captured and eager pos-pointer (R <= 8) or eager, v01 against kx at each --kx-s; sliding layers: v01 eager, and v01
//! and kx at each --kx-c captured and eager pos-pointer (R <= 8) or eager. `captured`: launches of a CUDA graph holding
//! `graph_calls` of the cell's calls (captured once, so its launch is spread as in a step's graph); `eager pos-pointer`:
//! the same calls launched eagerly with the position read on the device (what this column measured before).
//!
//! --limb: planted inputs in a 16-row block (cases L1-L7) on kx's forms and on the e2614fe control.
const std = @import("std");
const cuda = @import("cuda");
const ic = @import("inkling2_cuda");
const ck = @import("cuda_attn_check.zig");

const attn = ic.attn;
const Mat = ic.ops.Mat;
const D = ic.weights.head_dim;
const dr = ic.weights.d_rel;
const Kind = ck.Kind;
const Opts = ck.Opts;

/// The positions the forwards read, merged into ranges and packed: row `at[i] + pos - lo[i]` holds position pos.
const Rows = struct {
    lo: []usize,
    hi: []usize,
    at: []usize,
    total: usize,

    fn init(a: std.mem.Allocator, positions: []const usize, most: usize, P: usize) !Rows {
        const sorted = try a.dupe(usize, positions);
        std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
        var lo: std.ArrayList(usize) = .empty;
        var hi: std.ArrayList(usize) = .empty;
        for (sorted) |p0| {
            const end = @min(P, p0 + most);
            if (hi.items.len > 0 and p0 <= hi.items[hi.items.len - 1]) {
                hi.items[hi.items.len - 1] = @max(hi.items[hi.items.len - 1], end);
            } else {
                try lo.append(a, p0);
                try hi.append(a, end);
            }
        }
        const at = try a.alloc(usize, lo.items.len);
        var total: usize = 0;
        for (lo.items, hi.items, at) |l, h, *x| {
            x.* = total;
            total += h - l;
        }
        return .{ .lo = lo.items, .hi = hi.items, .at = at, .total = total };
    }

    fn row(r: Rows, pos: usize) usize {
        for (r.lo, r.hi, r.at) |l, h, x| if (pos >= l and pos < h) return x + pos - l;
        unreachable;
    }
};

/// A forward's per-row inputs: q', rel', sink and gate.
const In = struct {
    q: Mat,
    rel: Mat,
    sink: Mat,
    gate: Mat,

    fn slice(i: In, r: usize, R: usize) In {
        return .{ .q = i.q.slice(r, R), .rel = i.rel.slice(r, R), .sink = i.sink.slice(r, R), .gate = i.gate.slice(r, R) };
    }
};

/// The device state of a check or bench: the op-check tool's buffers and pool (cuda_attn_check.zig), the packed
/// per-position inputs.
const Env = struct {
    x: ck.Ctx,
    rows: Rows,
    q: Mat,
    rel: Mat,
    sink: Mat,
    gate: Mat,

    fn kvOf(e: *const Env, kind: Kind, pg: bool) attn.Kv {
        if (kind == .global) return if (pg) e.x.kvp else .{ .k = e.x.b.k, .v = e.x.b.v, .slots = e.x.context, .heads = e.x.l.hk, .window = 0 };
        return .{ .k = e.x.b.k, .v = e.x.b.v, .slots = attn.window - 1 + ic.engine.prompt_rows, .heads = e.x.l.hk, .window = attn.window };
    }

    /// The packed inputs of R rows from pos0.
    fn inputs(e: *const Env, pos0: usize, R: usize) In {
        const all: In = .{ .q = e.q, .rel = e.rel, .sink = e.sink, .gate = e.gate };
        return all.slice(e.rows.row(pos0), R);
    }

    /// `core` for R rows at pos0 (form, kx options; `devpos`: the captured plan launched eagerly, the position read
    /// on the device: after setPos(pos0) or devPos(pos0)).
    fn core(e: *const Env, form: attn.Form, opt: attn.KxOpt, devpos: bool, kv: attn.Kv, global: bool, pos0: usize, R: usize, out: Mat) !void {
        try e.coreIn(form, opt, devpos, kv, global, pos0, e.inputs(pos0, R), out.slice(0, R));
    }

    /// `core` on the inputs `in` (R = in.q.rows) into `out`.
    fn coreIn(e: *const Env, form: attn.Form, opt: attn.KxOpt, devpos: bool, kv: attn.Kv, global: bool, pos0: usize, in: In, out: Mat) !void {
        const k = &e.x.k.attn;
        k.form = form;
        k.kxo = opt;
        k.rows = attn.max_rows;
        k.grid = attn.grid_splits;
        k.fold_min = attn.fold_splits;
        e.x.l.global = global;
        const dev: ?attn.Dev = if (devpos) .{ .posp = e.x.b.posp, .positions = e.x.context } else null;
        e.x.scratch.reset();
        try attn.core(&e.x.c, e.x.l, pos0, dev, in.q, in.rel, in.sink, in.gate, kv, out);
    }

    /// The device position pos0, written on the stream (it lands before the stream's next launch, as a step's header
    /// copy lands before its replay).
    fn devPos(e: *const Env, pos0: usize) !void {
        const view: cuda.DeviceBuffer = .{ .d = e.x.c.d, .ptr = e.x.b.posp, .len = 4, .borrowed = true };
        try view.fill32(@intCast(pos0), e.x.c.stream.handle);
    }

    /// e2614fe's tf_ik_attention on contiguous or ring keys (the bias table from tf_kx_relative).
    fn control(e: *const Env, kv: attn.Kv, global: bool, pos0: usize, R: usize, out: Mat) !void {
        e.x.l.global = global;
        e.x.scratch.reset();
        const r = e.rows.row(pos0);
        const bias = try e.x.c.tmp(attn.kxBiasBytes(R, e.x.l.hq));
        try attn.kxRelative(&e.x.c, e.x.l, e.rel.slice(r, R), bias);
        try attn.kxControl(&e.x.c, e.x.l, pos0, e.q.slice(r, R), bias, e.sink.slice(r, R), e.gate.slice(r, R), kv, out.slice(0, R));
    }

    fn sync(e: *const Env) !void {
        try e.x.c.stream.synchronize();
    }

    fn download(e: *const Env, m: Mat, R: usize, out: []u8) !void {
        try cuda.memory.download(e.x.c.d, m.ptr, out[0 .. R * m.cols * 2]);
    }
};

/// What a run of the tool holds alive (freed by deinit).
const Own = struct {
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: ic.kernels.Kernels,
    gemms: ic.ops.Gemms,
    scratch: ic.ops.Scratch,
    b: ck.Bufs = .{},
    l: ic.weights.Layer,
    pool: ?ck.Pool = null,

    fn deinit(o: *Own, gpa: std.mem.Allocator) void {
        if (o.pool) |pl| gpa.free(pl.table);
        o.b.deinit(gpa);
        o.scratch.deinit();
        o.gemms.deinit();
        o.k.unload();
        o.stream.deinit();
        o.ctx.deinit();
        o.driver.close();
    }
};

/// The op scratch the variants take at this shape: kx's need at each S (eager and captured), kx-split forced up to
/// its cap, and v0.1's need (bench).
fn scratchBytes(k: *ic.kernels.Kernels, o: Opts, most: usize) usize {
    var need: usize = attn.kx_split_cap + attn.kxBiasBytes(most, o.hq);
    for (o.kx_s) |s| need = @max(need, attn.kxNeed(.{ .s = s, .g = o.kx_g }, o.hq, o.context, @max(most, 1), attn.max_rows));
    if (o.v01 and o.bench) {
        const form = k.attn.form;
        k.attn.form = .v01;
        k.attn.fold_min = 0;
        need = @max(need, attn.scratchNeed(&k.attn, o.hq, o.hk, o.context, @max(most, 1), attn.max_rows));
        k.attn.fold_min = attn.fold_splits;
        k.attn.form = form;
    }
    return need;
}

/// The packed inputs at `positions` (each the `most` positions from it) into env: rows, q', rel', sink, gate.
fn makeInputs(gpa: std.mem.Allocator, a: std.mem.Allocator, b: *ck.Bufs, d: *const cuda.Driver, env: *Env, o: Opts, positions: []const usize, most: usize, rng: std.Random) !void {
    const rows = try Rows.init(a, positions, most, o.context);
    const nr = @max(rows.total, 1);
    env.rows = rows;
    env.q = Mat.of(try b.zeroed(gpa, d, nr * o.hq * D * 2), nr, o.hq * D, .bf16);
    env.rel = Mat.of(try b.zeroed(gpa, d, nr * o.hq * dr * 2), nr, o.hq * dr, .bf16);
    env.sink = Mat.of(try b.zeroed(gpa, d, nr * o.hq * 2), nr, o.hq, .bf16);
    env.gate = Mat.of(try b.zeroed(gpa, d, nr * o.hq * D * 2), nr, o.hq * D, .bf16);
    // the same generator use as cuda_attn_check.zig's fillInputs, range by range
    for (rows.lo, rows.hi, rows.at) |lo, hi, at| {
        const n = hi - lo;
        try ck.Bufs.randomInto(gpa, d, rng, env.q.row(at), n * o.hq * D, 1.5);
        try ck.Bufs.randomInto(gpa, d, rng, env.rel.row(at), n * o.hq * dr, 1.0);
        try ck.Bufs.randomInto(gpa, d, rng, env.sink.row(at), n * o.hq, 3.0);
        try ck.Bufs.randomInto(gpa, d, rng, env.gate.row(at), n * o.hq * D, 3.0);
    }
}

/// Loads the kernels, the caches (keys and values at every position), the pool (global layers) and the packed inputs.
fn setup(gpa: std.mem.Allocator, a: std.mem.Allocator, own: *Own, env: *Env, o: Opts, positions: []const usize, most: usize, rng: std.Random) !void {
    own.driver = try cuda.Driver.open();
    errdefer own.driver.close();
    own.ctx = try cuda.Context.init(&own.driver, 0);
    errdefer own.ctx.deinit();
    own.stream = try cuda.Stream.init(&own.driver, true);
    errdefer own.stream.deinit();
    own.k = try ic.kernels.Kernels.load(&own.driver);
    errdefer own.k.unload();
    if (own.k.attn.kx == null) return error.KxNotBuilt;
    own.gemms = try ic.ops.Gemms.init(gpa, &own.driver);
    errdefer own.gemms.deinit();
    own.scratch = try ic.ops.Scratch.init(&own.driver, scratchBytes(&own.k, o, most) + (16 << 20));
    errdefer own.scratch.deinit();
    own.b = .{};
    own.pool = null;
    const d = &own.driver;
    const b = &own.b;
    const P = o.context;
    try makeInputs(gpa, a, b, d, env, o, positions, most, rng);
    b.proj = try b.random(gpa, d, rng, dr * attn.rel_extent, 1.0);
    b.k = try b.random(gpa, d, rng, P * o.hk * D, 1.5);
    b.v = try b.random(gpa, d, rng, P * o.hk * D, 2.0);
    b.posp = try b.zeroed(gpa, d, 256);
    own.l = .{ .index = 0, .global = false, .boundary = false, .hq = @intCast(o.hq), .hk = @intCast(o.hk), .nq = 1, .experts = .{ .format = .nvfp4, .lo = 0, .hi = 0 } };
    own.l.rel_proj = b.proj;
    env.x = .{ .c = .{ .d = d, .stream = own.stream, .k = &own.k, .gemms = &own.gemms, .scratch = &own.scratch }, .k = &own.k, .scratch = &own.scratch, .b = b, .l = &own.l, .kv = undefined, .context = P, .grid = attn.grid_splits };
}

/// How a variant runs: eager launches; eager launches of the captured plan, the position read on the device
/// (`devpos`); or a CUDA graph of the captured plan, replayed (`graph`).
const Mode = enum { eager, devpos, graph };

const Var = struct {
    name: []const u8,
    opt: attn.KxOpt,
    mode: Mode = .eager,
    global_only: bool = false,
    /// compared only where the forward's keys lie in one split (kx-nosplit; global: pos0 + R <= S)
    nosplit: bool = false,
};

fn variants(s: usize, g: usize) [n_var]Var {
    const first = [_]Var{
        .{ .name = "kx", .opt = .{ .s = s, .g = g } },
        .{ .name = "kx-tile", .opt = .{ .s = s, .g = g, .c = 1 } },
        .{ .name = "kx-dec2", .opt = .{ .s = s, .g = g, .c = 2 } },
        .{ .name = "kx-dec4", .opt = .{ .s = s, .g = g, .c = 4 } },
        .{ .name = "kx-split", .opt = .{ .s = s, .g = g, .global = .split } },
        .{ .name = "kx-split-g7", .opt = .{ .s = s, .g = 7, .global = .split } },
        .{ .name = "kx-split-c1", .opt = .{ .s = s, .g = g, .c = 1, .global = .split } },
        .{ .name = "kx-split-c4", .opt = .{ .s = s, .g = g, .c = 4, .global = .split } },
        .{ .name = "kx-walk", .opt = .{ .s = s, .g = g, .global = .walk } },
        .{ .name = "kx-devpos", .opt = .{ .s = s, .g = g }, .mode = .devpos },
        .{ .name = "kx-devpos-g7", .opt = .{ .s = s, .g = 7 }, .mode = .devpos },
        .{ .name = "kx-devpos-c1", .opt = .{ .s = s, .g = g, .c = 1 }, .mode = .devpos },
        .{ .name = "kx-nosplit", .opt = .{ .s = s, .g = g, .global = .tile }, .nosplit = true },
        .{ .name = "kx-unpack", .opt = .{ .s = s, .g = g, .pack = false } },
        .{ .name = "kx-pack", .opt = .{ .s = s, .g = g, .pack = true } },
        .{ .name = "kx-devpos-unpack", .opt = .{ .s = s, .g = g, .pack = false }, .mode = .devpos },
    };
    return first ++ graphVariants(s, g);
}

/// The engine's captured forms, replayed from CUDA graphs: the default dispatch (Engine.graphStep's), a 7-split grid
/// (many rounds a CTA column), C 1, GQA packing off.
fn graphVariants(s: usize, g: usize) [n_graph]Var {
    return .{
        .{ .name = "kx-graph", .opt = .{ .s = s, .g = g }, .mode = .graph },
        .{ .name = "kx-graph-g7", .opt = .{ .s = s, .g = 7 }, .mode = .graph },
        .{ .name = "kx-graph-c1", .opt = .{ .s = s, .g = g, .c = 1 }, .mode = .graph },
        .{ .name = "kx-graph-unpack", .opt = .{ .s = s, .g = g, .pack = false }, .mode = .graph },
    };
}
const n_graph = 4;
const n_var = 16 + n_graph;

/// A plan as text: its kx form, C, S, span and grid.
fn planText(buf: []u8, p: attn.Plan) []const u8 {
    const pk: []const u8 = if (p.pack) " pk" else "";
    return switch (p.kx) {
        .tile => std.fmt.bufPrint(buf, "{s} C{d}{s}", .{ if (p.c == 1) "kx-tile" else "kx-dec", p.c, pk }) catch "kx-tile",
        .split => std.fmt.bufPrint(buf, "kx-split S={d} C{d} span {d} grid {d}{s}", .{ p.s, p.c, p.span, p.grid, pk }) catch "kx-split",
        .walk => std.fmt.bufPrint(buf, "kx-split-walk S={d} span {d}{s}", .{ p.s, p.span, pk }) catch "kx-split-walk",
        .none => @tagName(p.form),
    };
}

const Counts = struct { tile: usize = 0, dec: usize = 0, split: usize = 0, walk: usize = 0 };

/// tf-inkling2-attn --op-check --forms kx / --bench --forms kx.
pub fn run(gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const rl = o.rows orelse if (o.bench) &[_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 } else &[_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 2048 };
    var most: usize = 1;
    for (rl) |R| most = @max(most, R);
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rng = prng.random();
    var own: Own = undefined;
    var env: Env = undefined;
    try setup(gpa, a, &own, &env, o, o.positions, most, rng);
    defer own.deinit(gpa);
    const global = std.mem.indexOfScalar(Kind, o.kinds, .global) != null;
    var failed: usize = 0;
    if (global) {
        own.pool = try ck.makePool(gpa, &own.driver, rng, &env.x, o);
        if (!o.bench) failed += try ck.checkPool(gpa, &own.driver, &env.x, o, own.pool.?);
    }
    var cb: [160]u8 = undefined;
    own.k.attn.kxo = .{ .s = o.kx_s[0], .g = o.kx_g };
    own.k.attn.form = .kx;
    std.debug.print("{s}\n", .{own.k.attn.census(&cb)});
    if (o.bench) {
        try bench(gpa, io, &env, o, rl, own.pool);
        own.k.attn.printExceptions(&own.driver);
        return 0;
    }
    var rig = try Rig.init(gpa, a, &own, &env, o);
    defer rig.deinit(gpa);
    for (o.kx_s) |s| failed += try opCheck(gpa, &env, o, rl, most, s, &rig);
    if (global) failed += try ck.checkPool(gpa, &own.driver, &env.x, o, own.pool.?);
    own.k.attn.printExceptions(&own.driver);
    return if (failed == 0) 0 else 1;
}

/// The kx matrix at one S, then its graph walks: returns 1 when any row of any run differs from its kx-solo row, or
/// a walk fails (a replay differs from kx-solo or from the eager launch of its form, or a control does not fail).
fn opCheck(gpa: std.mem.Allocator, env: *Env, o: Opts, rl: []const usize, most: usize, s: usize, rig: *Rig) !usize {
    const vs = variants(s, o.kx_g);
    const row_bytes = o.hq * D * 2;
    const nout = 2 * vs.len + 2;
    var bufs: [2 * n_var + 2]Mat = undefined;
    var own_bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (own_bufs.items) |*b| b.free();
        own_bufs.deinit(gpa);
    }
    for (bufs[0..nout]) |*m| {
        var b = try cuda.DeviceBuffer.alloc(env.x.c.d, most * row_bytes);
        errdefer b.free();
        try own_bufs.append(gpa, b);
        m.* = Mat.of(b.ptr, most, o.hq * D, .bf16);
    }
    const solo_m = bufs[nout - 2];
    const ctl_m = bufs[nout - 1];
    const host = try gpa.alloc(u8, nout * most * row_bytes);
    defer gpa.free(host);
    var solo: std.AutoHashMapUnmanaged(usize, []u8) = .empty;
    defer {
        var it = solo.valueIterator();
        while (it.next()) |v| gpa.free(v.*);
        solo.deinit(gpa);
    }
    // the kx-graph variants' graphs: one a (variant, keys, R) of a kind, captured at its first forward
    var caps: std.AutoHashMapUnmanaged(usize, Cap) = .empty;
    defer {
        dropCaps(gpa, &caps);
        caps.deinit(gpa);
    }
    var g_caps: usize = 0;
    var g_nodes: usize = 0;
    var g_replays: usize = 0;
    var g_moved: usize = 0;
    var forwards: usize = 0;
    var runs: usize = 0;
    var paged_runs: usize = 0;
    var differ: usize = 0;
    var ctl_eq: usize = 0;
    var ctl_rows: usize = 0;
    var cnt: Counts = .{};
    const P = env.x.context;
    // a sliding row at p <= window - 1 (window start 0) has a global row's bytes at S = kx_sliding_split
    var low: std.AutoHashMapUnmanaged(usize, []u8) = .empty;
    defer {
        var it = low.valueIterator();
        while (it.next()) |v| gpa.free(v.*);
        low.deinit(gpa);
    }
    var cross_rows: usize = 0;
    var cross_eq: usize = 0;
    for (o.kinds) |kind| {
        {
            // the sliding rows kept for the cross-check (the solo rows of the kind before this one)
            var it = solo.iterator();
            while (it.next()) |x| {
                if (kind == .global and x.key_ptr.* <= attn.window - 1 and !low.contains(x.key_ptr.*)) try low.put(gpa, x.key_ptr.*, x.value_ptr.*) else gpa.free(x.value_ptr.*);
            }
            solo.clearRetainingCapacity();
        }
        dropCaps(gpa, &caps);
        const glob = kind == .global;
        for (o.positions) |pos0| for (rl) |R| {
            if (pos0 + R > P) continue;
            // the solo rows this forward needs that are not yet held: R = 1 forwards, default dispatch
            var t0: usize = 0;
            while (t0 < R) {
                var n: usize = 0;
                while (t0 + n < R and n < most and !solo.contains(pos0 + t0 + n)) : (n += 1) {
                    try env.core(.kx, .{ .s = s, .g = o.kx_g }, false, env.kvOf(kind, false), glob, pos0 + t0 + n, 1, solo_m.slice(n, 1));
                }
                if (n == 0) {
                    t0 += 1;
                    continue;
                }
                try env.sync();
                const tmp = try gpa.alloc(u8, n * row_bytes);
                defer gpa.free(tmp);
                try cuda.memory.download(env.x.c.d, solo_m.ptr, tmp);
                for (0..n) |i| try solo.put(gpa, pos0 + t0 + i, try gpa.dupe(u8, tmp[i * row_bytes ..][0..row_bytes]));
                t0 += n;
            }
            // the variants (each global one also paged)
            var list: [2 * n_var]struct { v: Var, vi: usize, pg: bool } = undefined;
            var nl: usize = 0;
            for (vs, 0..) |v, vi| {
                if (v.global_only and !glob) continue;
                if (v.mode != .eager and R > attn.max_rows) continue;
                if (v.nosplit and (if (glob) pos0 + R > s else env.kvOf(kind, false).lo(pos0) / attn.kx_sliding_split != (pos0 + R - 1) / attn.kx_sliding_split)) continue;
                for (0..@as(usize, if (glob) 2 else 1)) |pg| {
                    list[nl] = .{ .v = v, .vi = vi, .pg = pg == 1 };
                    nl += 1;
                }
            }
            try env.sync();
            try env.x.setPos(pos0);
            for (list[0..nl], 0..) |e, j| {
                const kv = env.kvOf(kind, e.pg);
                if (e.v.mode == .graph) {
                    const key = (2 * e.vi + @intFromBool(e.pg)) * (attn.max_rows + 1) + R;
                    const gop = try caps.getOrPut(gpa, key);
                    if (!gop.found_existing) {
                        gop.value_ptr.* = capture(gpa, env, rig.stage, e.v.opt, kv, glob, pos0, R) catch |err| {
                            _ = caps.remove(key);
                            return err;
                        };
                        g_caps += 1;
                        g_nodes += gop.value_ptr.nodes;
                    }
                    const cp = gop.value_ptr;
                    try replay(env, cp, rig.stage, env.inputs(pos0, R), pos0);
                    try env.x.c.copy(bufs[j].ptr, rig.stage.out.ptr, R * row_bytes);
                    g_replays += 1;
                    g_moved += @intFromBool(cp.pos != pos0);
                } else try env.core(.kx, e.v.opt, e.v.mode == .devpos, kv, glob, pos0, R, bufs[j]);
                const p = attn.plan(&env.x.k.attn, R, pos0, if (e.v.mode != .eager) attn.Dev{ .posp = 1, .positions = P } else null, kv, o.hq, o.hk);
                switch (p.kx) {
                    .tile => if (p.c == 1) {
                        cnt.tile += 1;
                    } else {
                        cnt.dec += 1;
                    },
                    .split => cnt.split += 1,
                    .walk => cnt.walk += 1,
                    .none => {},
                }
                if (e.pg) paged_runs += 1;
            }
            try env.control(env.kvOf(kind, false), glob, pos0, R, ctl_m);
            try env.sync();
            for (0..nl) |j| try env.download(bufs[j], R, host[j * most * row_bytes ..]);
            try env.download(ctl_m, R, host[(nout - 1) * most * row_bytes ..]);
            var bad: usize = 0;
            for (list[0..nl], 0..) |e, j| {
                runs += 1;
                for (0..R) |t| {
                    const want = solo.get(pos0 + t).?;
                    if (!std.mem.eql(u8, want, host[(j * most + t) * row_bytes ..][0..row_bytes])) {
                        if (bad < 4) std.debug.print("  DIFFER S={d} {t} pos0 {d} rows {d}: {s}{s} row {d} against kx-solo\n", .{ s, kind, pos0, R, e.v.name, if (e.pg) " paged" else "", t });
                        bad += 1;
                    }
                }
            }
            // e2614fe (information): kx's bits wherever its exception path and the merge do not enter
            for (0..R) |t| {
                ctl_rows += 1;
                ctl_eq += @intFromBool(std.mem.eql(u8, solo.get(pos0 + t).?, host[((nout - 1) * most + t) * row_bytes ..][0..row_bytes]));
            }
            var pb: [96]u8 = undefined;
            env.x.k.attn.form = .kx;
            env.x.k.attn.kxo = .{ .s = s, .g = o.kx_g };
            std.debug.print("FORM S={d} {t} pos0 {d} rows {d}: {s}; {d} runs, {d} rows differ\n", .{ s, kind, pos0, R, planText(&pb, attn.plan(&env.x.k.attn, R, pos0, null, env.kvOf(kind, false), o.hq, o.hk)), nl, bad });
            forwards += 1;
            if (bad > 0) differ += 1;
        };
        try printSoloDigest(gpa, s, kind, &solo);
        if (kind == .global and s == attn.kx_sliding_split) {
            var it = low.iterator();
            while (it.next()) |x| if (solo.get(x.key_ptr.*)) |g| {
                cross_rows += 1;
                if (std.mem.eql(u8, g, x.value_ptr.*)) cross_eq += 1 else if (cross_rows - cross_eq <= 4) std.debug.print("  DIFFER S={d} sliding p {d} against global (window start 0)\n", .{ s, x.key_ptr.* });
            };
        }
    }
    if (s == attn.kx_sliding_split and cross_rows > 0) {
        std.debug.print("{s} cross-check S={d}: sliding rows at p <= {d} against global rows: {d}/{d} equal\n", .{ if (cross_eq == cross_rows) "PASS" else "FAIL", s, attn.window - 1, cross_eq, cross_rows });
        if (cross_eq != cross_rows) differ += 1;
    }
    dropCaps(gpa, &caps);
    std.debug.print("kx vs e2614fe: {d}/{d} rows equal (info)\n", .{ ctl_eq, ctl_rows });
    std.debug.print("forms run: kx-tile {d}, kx-dec {d}, kx-split {d}, kx-split-walk {d}\n", .{ cnt.tile, cnt.dec, cnt.split, cnt.walk });
    std.debug.print("GRAPH S={d} matrix: {d} CUDA graphs captured and instantiated ({d} nodes; kx-graph, -g7, -c1, -unpack; contiguous and paged), {d} replays ({d} at a position other than the capture's), each among the runs above\n", .{ s, g_caps, g_nodes, g_replays, g_moved });
    const walk_bad = try graphWalk(gpa, &rig.walk, o, s, rig);
    const ok = differ == 0 and walk_bad == 0;
    std.debug.print("{s} op-check kx: heads {d}/{d}, S {d}, G {d}, context {d}: {d} forwards, {d} runs ({d} paged), every row against kx-solo, {d} differ{s}\n", .{ if (ok) "PASS" else "FAIL", o.hq, o.hk, s, o.kx_g, P, forwards, runs, paged_runs, differ, if (walk_bad > 0) "; the graph walk FAILED" else "" });
    return if (ok) 0 else 1;
}

fn dropCaps(gpa: std.mem.Allocator, caps: *std.AutoHashMapUnmanaged(usize, Cap)) void {
    var it = caps.valueIterator();
    while (it.next()) |c| c.deinit(gpa);
    caps.clearRetainingCapacity();
}

// ---------------------------------------------------------------------------------------------------------------
// CUDA graphs of `core` (the engine's captured form, Engine.graphOf / graphStep)

/// A captured forward's inputs and output at fixed addresses (up to attn.max_rows rows): a replay copies a position's
/// rows in first, as a step's projections write the activations its graph reads.
const Stage = struct {
    in: In,
    out: Mat,

    fn make(gpa: std.mem.Allocator, b: *ck.Bufs, d: *const cuda.Driver, hq: usize) !Stage {
        const n = attn.max_rows;
        return .{
            .in = .{
                .q = Mat.of(try b.zeroed(gpa, d, n * hq * D * 2), n, hq * D, .bf16),
                .rel = Mat.of(try b.zeroed(gpa, d, n * hq * dr * 2), n, hq * dr, .bf16),
                .sink = Mat.of(try b.zeroed(gpa, d, n * hq * 2), n, hq, .bf16),
                .gate = Mat.of(try b.zeroed(gpa, d, n * hq * D * 2), n, hq * D, .bf16),
            },
            .out = Mat.of(try b.zeroed(gpa, d, n * hq * D * 2), n, hq * D, .bf16),
        };
    }
};

/// A launch of a captured forward on a ring view of a run (Ctx.launchOn into a RunLog) and its node in the graph.
const RunNode = struct { node: cuda.graph.Node, at: ic.ops.RunLog.Launch };

/// `core` for R rows captured from the stream into a CUDA graph (the plan of a captured step: Dev{posp, positions})
/// and instantiated; `pos` the position it was captured at; `runs` its launches on a ring view of a run.
const Cap = struct {
    graph: cuda.graph.Graph,
    exec: cuda.graph.Exec,
    R: usize,
    pos: usize,
    /// the graph's nodes (kernels and copies)
    nodes: usize,
    runs: []RunNode,

    fn deinit(c: *Cap, gpa: std.mem.Allocator) void {
        c.exec.deinit();
        c.graph.deinit();
        gpa.free(c.runs);
    }
};

/// Captures `core` (kx form `opt`, keys `kv`) for R rows at pos0 on the stage, as Engine.graphOf does a step's: the
/// stream in relaxed capture, a RunLog while kv is a ring view of a run (Ctx.runs), the nodes it recorded found in
/// the graph (Engine.runNodes), instantiated and uploaded. The template graph is kept (its node handles repoint the
/// executable graph, as the engine keeps it while it holds run nodes).
fn capture(gpa: std.mem.Allocator, e: *Env, st: Stage, opt: attn.KxOpt, kv: attn.Kv, global: bool, pos0: usize, R: usize) !Cap {
    try e.sync();
    var log: ic.ops.RunLog = .{ .gpa = gpa };
    defer log.deinit();
    e.x.c.runs = if (kv.run != null) &log else null;
    defer e.x.c.runs = null;
    const stream = e.x.c.stream;
    try cuda.graph.beginCapture(stream, .relaxed);
    const body = e.coreIn(.kx, opt, true, kv, global, pos0, st.in.slice(0, R), st.out.slice(0, R));
    var g = cuda.graph.endCapture(stream) catch |err| {
        try body;
        return err;
    };
    errdefer g.deinit();
    try body;
    const runs = try runNodes(gpa, e.x.c.d, g, &log);
    errdefer gpa.free(runs);
    var exec = try g.instantiate();
    errdefer exec.deinit();
    try exec.upload(stream);
    return .{ .graph = g, .exec = exec, .R = R, .pos = pos0, .nodes = try g.nodeCount(), .runs = runs };
}

/// The graph's nodes of the launches `log` recorded, found by kernel and the first base argument (as
/// cuda_engine.zig's Engine.runNodes); error.RunNodes unless each is found once.
fn runNodes(gpa: std.mem.Allocator, d: *const cuda.Driver, graph: cuda.graph.Graph, log: *const ic.ops.RunLog) ![]RunNode {
    if (log.items.items.len == 0) return &.{};
    const out = try gpa.alloc(RunNode, log.items.items.len);
    errdefer gpa.free(out);
    for (out, log.items.items) |*o, x| o.* = .{ .node = null, .at = x };
    const all = try gpa.alloc(cuda.graph.Node, try graph.nodeCount());
    defer gpa.free(all);
    const nodes = try graph.nodes(all);
    var found: usize = 0;
    for (nodes) |node| {
        var p: cuda.abi.KernelNodeParams = std.mem.zeroes(cuda.abi.KernelNodeParams);
        if (d.api.cuGraphKernelNodeGetParams_v2(node, &p) != cuda.abi.success) continue; // not a kernel node
        const params = p.params orelse continue;
        for (out) |*o| {
            if (o.node != null or o.at.f.handle != p.func) continue;
            const i = o.at.at[0];
            const got = @as(*align(1) const u64, @ptrCast(params[i] orelse continue)).*;
            if (got != o.at.args.integer(i).?) continue;
            o.node = node;
            found += 1;
            break;
        }
    }
    if (found != out.len) return error.RunNodes;
    return out;
}

/// Before a replay after the stream's run moved to `now`: every recorded node pointed at it, its base arguments moved
/// by whole pages of `step` bytes (cuda_engine.zig's Graph.follow). Returns the nodes changed.
fn follow(c: *Cap, now: u32, step: usize) !usize {
    var n: usize = 0;
    for (c.runs) |*r| {
        if (now == r.at.run) continue;
        const delta = (@as(i64, now) - @as(i64, r.at.run)) * @as(i64, @intCast(step));
        for (r.at.at[0..r.at.n]) |i| {
            const was = r.at.args.integer(i).?;
            const to: u64 = @intCast(@as(i64, @intCast(was)) + delta);
            @memcpy(r.at.args.storage[r.at.args.offsets[i]..][0..8], std.mem.asBytes(&to));
        }
        try c.exec.setKernel(r.node, r.at.f, r.at.cfg, &r.at.args);
        r.at.run = now;
        n += 1;
    }
    return n;
}

/// A replay of `c` on the rows `src` (copied onto the stage) at device position pos0 (null: the device position left
/// as it is), all on the stream.
fn replay(e: *const Env, c: *const Cap, st: Stage, src: In, pos0: ?usize) !void {
    const to = st.in.slice(0, c.R);
    for ([_]Mat{ to.q, to.rel, to.sink, to.gate }, [_]Mat{ src.q, src.rel, src.sink, src.gate }) |dst, from| {
        if (from.rows != c.R) return error.BadAttnShape;
        try e.x.c.copy(dst.ptr, from.ptr, from.bytes());
    }
    if (pos0) |p| try e.devPos(p);
    try c.exec.launchOn(e.x.c.stream);
}

/// What the graph replays use besides the matrix: the stage, the walks' inputs (from a generator of their own, so the
/// matrix's inputs and its DIGEST lines stay as they were) and, with global layers, the arena of the run and
/// page-table moves.
const Rig = struct {
    stage: Stage,
    walk: Env,
    arena: ?Arena,
    prng: std.Random.DefaultPrng,

    fn init(gpa: std.mem.Allocator, a: std.mem.Allocator, own: *Own, env: *const Env, o: Opts) !Rig {
        const d = &own.driver;
        var r: Rig = .{ .stage = try Stage.make(gpa, &own.b, d, o.hq), .walk = env.*, .arena = null, .prng = std.Random.DefaultPrng.init(o.seed ^ 0x9e37_79b9_7f4a_7c15) };
        var starts: std.ArrayList(usize) = .empty;
        for (o.kx_s) |s| for (o.kinds) |kind| {
            var bb: [max_bounds]usize = undefined;
            for (walkBounds(&bb, kind, s, o.kx_g, o.context)) |b| try starts.append(a, b - walk_before);
        };
        try makeInputs(gpa, a, &own.b, d, &r.walk, o, starts.items, walk_span, r.prng.random());
        if (std.mem.indexOfScalar(Kind, o.kinds, .global) != null) r.arena = try Arena.init(gpa, &own.b, d, o);
        return r;
    }

    fn deinit(r: *Rig, gpa: std.mem.Allocator) void {
        if (r.arena) |*x| x.deinit(gpa);
    }
};

/// The run and page-table moves' own pool, laid out as the served pool (cuda_pages.zig: a region a layer's keys or
/// values, page id i at i x run_bytes in each): 2 entries + 2 pages, the last the scrub page, NaN until written.
/// `ids`: the stream's pages in position order (all the scrub page until placed); `table` the device page table
/// (entries + 8 i32, the spare entries on the scrub page) and `host` its host copy; `new` the next placement.
const Arena = struct {
    base: u64,
    pages: usize,
    run_bytes: usize,
    entries: usize,
    hk: usize,
    ids: []u32,
    new: []u32,
    host: []i32,
    table: u64,

    fn init(gpa: std.mem.Allocator, b: *ck.Bufs, d: *const cuda.Driver, o: Opts) !Arena {
        const entries = (o.context + attn.split_keys - 1) / attn.split_keys;
        const pages = 2 * entries + 2;
        const run_bytes = attn.split_keys * o.hk * D * 2;
        const base = try b.filled(gpa, d, 2 * pages * run_bytes, 0xff);
        const table = try b.zeroed(gpa, d, (entries + 8) * 4);
        const ids = try gpa.alloc(u32, entries);
        errdefer gpa.free(ids);
        const new = try gpa.alloc(u32, entries);
        errdefer gpa.free(new);
        const host = try gpa.alloc(i32, entries + 8);
        @memset(ids, @intCast(pages - 1));
        return .{ .base = base, .pages = pages, .run_bytes = run_bytes, .entries = entries, .hk = o.hk, .ids = ids, .new = new, .host = host, .table = table };
    }

    fn deinit(a: *Arena, gpa: std.mem.Allocator) void {
        gpa.free(a.ids);
        gpa.free(a.new);
        gpa.free(a.host);
    }

    fn scrub(a: *const Arena) u32 {
        return @intCast(a.pages - 1);
    }

    fn region(a: *const Arena, r: usize) u64 {
        return a.base + r * a.pages * a.run_bytes;
    }

    fn placed(a: *const Arena) bool {
        return a.ids[0] != a.scrub();
    }

    /// The stream's keys and values through the page table.
    fn paged(a: *const Arena) attn.Kv {
        return .{ .k = a.region(0), .v = a.region(1), .slots = a.entries * attn.split_keys, .heads = a.hk, .window = 0, .pages = a.table, .page = a.run_bytes / 2 };
    }

    fn isRun(a: *const Arena) bool {
        for (a.ids[1..], a.ids[0 .. a.ids.len - 1]) |x, prev| if (x != prev + 1) return false;
        return a.placed();
    }

    /// State.kvFor's view of a one-row forward on a run: the run's first page the base, a slot a position (P).
    fn ring(a: *const Arena, P: usize) attn.Kv {
        const start = a.ids[0];
        const off = @as(u64, start) * a.run_bytes;
        return .{ .k = a.region(0) + off, .v = a.region(1) + off, .slots = P, .heads = a.hk, .window = 0, .run = start };
    }

    /// The next placement: the run from page `start`.
    fn toRun(a: *Arena, start: usize) void {
        for (a.new, 0..) |*x, i| x.* = @intCast(start + i);
    }

    /// The next placement: `entries` pages of a permutation of the pool's (the scrub page aside), none an entry's
    /// present page.
    fn toShuffle(a: *Arena, gpa: std.mem.Allocator, rng: std.Random) !void {
        const all = try gpa.alloc(u32, a.pages - 1);
        defer gpa.free(all);
        for (all, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(u32, all);
        @memcpy(a.new, all[0..a.entries]);
        for (a.new, a.ids, 0..) |*x, old, i| if (x.* == old) {
            const j = (i + 1) % a.entries;
            std.mem.swap(u32, x, &a.new[j]);
        };
    }

    /// The stream's rows (from the contiguous cache) moved to the pages `new`, on the stream; the old pages it leaves
    /// zeroed (finite: a stale read gives wrong bytes and no exception); the device table follows when `commit`.
    fn place(a: *Arena, gpa: std.mem.Allocator, e: *const Env, commit: bool) !void {
        var keep = try std.DynamicBitSet.initEmpty(gpa, a.pages);
        defer keep.deinit();
        for (a.new) |id| keep.set(id);
        const c = &e.x.c;
        for (a.ids) |id| if (id != a.scrub() and !keep.isSet(id)) for (0..2) |r| try c.zero(a.region(r) + id * a.run_bytes, a.run_bytes);
        const row = a.hk * D * 2;
        const P = e.x.context;
        for (a.new, 0..) |id, i| {
            const n = @min(attn.split_keys, P - i * attn.split_keys) * row;
            for ([_]u64{ e.x.b.k, e.x.b.v }, 0..) |src, r| try c.copy(a.region(r) + id * a.run_bytes, src + i * attn.split_keys * row, n);
        }
        @memcpy(a.ids, a.new);
        if (commit) try a.commitTable(e);
    }

    /// The device page table onto `ids` (the stream idle first; the upload returns once landed).
    fn commitTable(a: *Arena, e: *const Env) !void {
        for (a.host, 0..) |*t, i| t.* = @intCast(if (i < a.entries) a.ids[i] else a.scrub());
        try e.sync();
        const view: cuda.DeviceBuffer = .{ .d = e.x.c.d, .ptr = a.table, .len = a.host.len * 4, .borrowed = true };
        try view.upload(0, std.mem.sliceAsBytes(a.host));
    }
};

const walk_before = 10;
const walk_after = 2;
/// The positions a walk's inputs cover from b - walk_before: its replays' rows and a control's next row.
const walk_span = walk_before + walk_after + attn.max_rows + 2;
const max_bounds = 8;

/// The boundaries a walk crosses: global layers a kx tile (64 keys), a page (256), the key split S and 2 S, the grid's
/// wrap at 7 S (-g7) and G S, 2 G S, and the slot's end; sliding layers a tile, the window's edge (1024: the first row
/// not seeing position 0), the ring's wraps (slots, 2 and 3 slots, the last before the context's half) and the
/// slot's end. A walk replays at every position from b - walk_before to b + walk_after.
fn walkBounds(buf: *[max_bounds]usize, kind: Kind, s: usize, g: usize, P: usize) []usize {
    const slots = attn.window - 1 + ic.engine.prompt_rows;
    const cand: [max_bounds]usize = if (kind == .global)
        .{ 64, attn.split_keys, s, 2 * s, 7 * s, g * s, 2 * g * s, P -| walk_before }
    else
        .{ 64, attn.window, slots, 2 * slots, 3 * slots, (P / 2) / slots * slots, P -| walk_before, attn.window + attn.kx_sliding_split };
    var n: usize = 0;
    for (cand) |b| {
        if (b < walk_before or b + walk_before > P or std.mem.indexOfScalar(usize, buf[0..n], b) != null) continue;
        buf[n] = b;
        n += 1;
    }
    std.mem.sort(usize, buf[0..n], {}, std.sort.asc(usize));
    return buf[0..n];
}

/// The keys a walk's graph reads: the contiguous cache (sliding: the ring), the pool's shuffled pages, a ring view of
/// a run of the arena's pages (one row: the served fallback), the arena's page table.
const KvForm = enum { contiguous, pool, run, table };

/// One S's graph walks: the kx-solo rows they compare with and their tallies.
const Walk = struct {
    gpa: std.mem.Allocator,
    w: *Env,
    rig: *Rig,
    s: usize,
    g: usize,
    kind: Kind = .global,
    rb: usize,
    twin: Mat,
    one: Mat,
    host: []u8,
    solo: std.AutoHashMapUnmanaged(usize, []u8) = .empty,
    caps: usize = 0,
    nodes: usize = 0,
    replays: usize = 0,
    moved: usize = 0,
    rows: usize = 0,
    bad_solo: usize = 0,
    bad_twin: usize = 0,
    bad_log: usize = 0,
    run_moves: usize = 0,
    repointed: usize = 0,
    table_moves: usize = 0,
    ctl: usize = 0,
    ctl_failed: usize = 0,
    printed: usize = 0,

    fn clearSolo(k: *Walk) void {
        var it = k.solo.valueIterator();
        while (it.next()) |v| k.gpa.free(v.*);
        k.solo.clearRetainingCapacity();
    }

    /// Row p alone (R = 1, default dispatch, eager, contiguous keys).
    fn soloRow(k: *Walk, p: usize) ![]const u8 {
        if (k.solo.get(p)) |r| return r;
        try k.w.core(.kx, .{ .s = k.s, .g = k.g }, false, k.w.kvOf(k.kind, false), k.kind == .global, p, 1, k.one);
        try k.w.sync();
        const r = try k.gpa.alloc(u8, k.rb);
        errdefer k.gpa.free(r);
        try cuda.memory.download(k.w.x.c.d, k.one.ptr, r);
        try k.solo.put(k.gpa, p, r);
        return r;
    }

    /// Replays `cp` at p, then the eager launch of the same form (the captured plan, the position read on the
    /// device) on the same keys `kv`; each row of the replay against kx-solo and against that launch.
    fn replayAt(k: *Walk, cp: *const Cap, v: Var, f: KvForm, kv: attn.Kv, p: usize) !void {
        const R = cp.R;
        for (0..R) |t| _ = try k.soloRow(p + t);
        try replay(k.w, cp, k.rig.stage, k.w.inputs(p, R), p);
        try k.w.core(.kx, v.opt, true, kv, k.kind == .global, p, R, k.twin);
        try k.w.sync();
        const rb = k.rb;
        try cuda.memory.download(k.w.x.c.d, k.rig.stage.out.ptr, k.host[0 .. R * rb]);
        try cuda.memory.download(k.w.x.c.d, k.twin.ptr, k.host[attn.max_rows * rb ..][0 .. R * rb]);
        var bad: usize = 0;
        var bad_t: usize = 0;
        for (0..R) |t| {
            const got = k.host[t * rb ..][0..rb];
            bad += @intFromBool(!std.mem.eql(u8, got, k.solo.get(p + t).?));
            bad_t += @intFromBool(!std.mem.eql(u8, got, k.host[(attn.max_rows + t) * rb ..][0..rb]));
        }
        k.replays += 1;
        k.moved += @intFromBool(p != cp.pos);
        k.rows += R;
        k.bad_solo += @intFromBool(bad > 0);
        k.bad_twin += @intFromBool(bad_t > 0);
        if ((bad > 0 or bad_t > 0) and k.printed < 8) {
            k.printed += 1;
            std.debug.print("  GRAPH DIFFER S={d} {t} {s} {t} rows {d} pos0 {d} (captured at {d}): {d} rows differ from kx-solo, {d} from the eager launch of the form\n", .{ k.s, k.kind, v.name, f, R, p, cp.pos, bad, bad_t });
        }
    }

    /// A planted control (it must differ from kx-solo): `cp` launched on p's rows, the device position written only
    /// when `write_pos`.
    fn control(k: *Walk, cp: *const Cap, v: Var, p: usize, write_pos: bool, what: []const u8) !void {
        const R = cp.R;
        for (0..R) |t| _ = try k.soloRow(p + t);
        try replay(k.w, cp, k.rig.stage, k.w.inputs(p, R), if (write_pos) p else null);
        try k.w.sync();
        try cuda.memory.download(k.w.x.c.d, k.rig.stage.out.ptr, k.host[0 .. R * k.rb]);
        var differs: usize = 0;
        for (0..R) |t| differs += @intFromBool(!std.mem.eql(u8, k.host[t * k.rb ..][0..k.rb], k.solo.get(p + t).?));
        k.ctl += 1;
        k.ctl_failed += @intFromBool(differs > 0);
        std.debug.print("GRAPH control S={d} {t} {s} rows {d} pos0 {d}, {s}: {d}/{d} rows differ from kx-solo, {s}\n", .{ k.s, k.kind, v.name, R, p, what, differs, R, if (differs > 0) "failed as required" else "DID NOT FAIL" });
    }

    /// One graph (variant v, keys f, R rows) captured at ps[0] and replayed at every position of ps, then once more
    /// at ps[0]; then f's moves and controls.
    fn walkOne(k: *Walk, v: Var, f: KvForm, R: usize, ps: []const usize) !void {
        const w = k.w;
        const P = w.x.context;
        const ar: ?*Arena = if (k.rig.arena) |*x| x else null;
        if ((f == .run and !ar.?.isRun()) or (f == .table and !ar.?.placed())) {
            ar.?.toRun(1);
            try ar.?.place(k.gpa, w, true);
        }
        const kv = switch (f) {
            .contiguous => w.kvOf(k.kind, false),
            .pool => w.kvOf(k.kind, true),
            .run => ar.?.ring(P),
            .table => ar.?.paged(),
        };
        var cp = try capture(k.gpa, w, k.rig.stage, v.opt, kv, k.kind == .global, ps[0], R);
        defer cp.deinit(k.gpa);
        k.caps += 1;
        k.nodes += cp.nodes;
        // a one-row forward on a run: RunLog holds kx-split's and the merge's launches, both found in the graph
        if (f == .run and cp.runs.len != 2) {
            k.bad_log += 1;
            std.debug.print("  GRAPH S={d} {s} on a run: {d} launches recorded on the run (want 2: kx-split, merge)\n", .{ k.s, v.name, cp.runs.len });
        }
        for (ps) |p| if (p + R <= P) try k.replayAt(&cp, v, f, kv, p);
        try k.replayAt(&cp, v, f, kv, ps[0]);
        const pm = ps[ps.len / 2];
        switch (f) {
            .contiguous => if (R == 1 and std.mem.eql(u8, v.name, "kx-graph")) {
                // the device position left at pm while pm + 1's rows are staged
                try k.replayAt(&cp, v, f, kv, pm);
                try k.control(&cp, v, pm + 1, false, "stale device position");
            },
            .run => {
                // the run moves to pages disjoint from it: first the replay with the K/V pointers still on the old
                // run, then the recorded nodes repointed (Graph.follow) and the walk again; then a move by one page
                const a = ar.?;
                const was = a.ids[0];
                a.toRun(if (was <= 1) a.entries + 1 else 0);
                try a.place(k.gpa, w, true);
                k.run_moves += 1;
                try k.control(&cp, v, pm, true, "K/V pointers left on the old run");
                for (0..2) |m| {
                    if (m == 1) {
                        a.toRun(if (a.ids[0] == 0) 1 else a.ids[0] - 1);
                        try a.place(k.gpa, w, true);
                        k.run_moves += 1;
                    }
                    const moved = try follow(&cp, a.ids[0], a.run_bytes);
                    k.repointed += moved;
                    if (moved != cp.runs.len) k.bad_log += 1;
                    const kv2 = a.ring(P);
                    for (ps) |p| if (p + R <= P) try k.replayAt(&cp, v, f, kv2, p);
                }
            },
            .table => {
                // the stream's pages move to other ids: first the replay with the device table still on the old
                // pages, then the table written and the walk again
                const a = ar.?;
                try a.toShuffle(k.gpa, k.rig.prng.random());
                try a.place(k.gpa, w, false);
                k.table_moves += 1;
                try k.control(&cp, v, pm, true, "page table left on the old pages");
                try a.commitTable(w);
                for (ps) |p| if (p + R <= P) try k.replayAt(&cp, v, f, kv, p);
            },
            .pool => {},
        }
    }
};

/// The graph walks at one S (see the header): returns 1 on any difference, a control that does not fail, or a run
/// launch not recorded or not repointed.
fn graphWalk(gpa: std.mem.Allocator, w: *Env, o: Opts, s: usize, rig: *Rig) !usize {
    const P = w.x.context;
    const rb = o.hq * D * 2;
    var twin_buf = try cuda.DeviceBuffer.alloc(w.x.c.d, attn.max_rows * rb);
    defer twin_buf.free();
    var one_buf = try cuda.DeviceBuffer.alloc(w.x.c.d, rb);
    defer one_buf.free();
    var k: Walk = .{ .gpa = gpa, .w = w, .rig = rig, .s = s, .g = o.kx_g, .rb = rb, .twin = Mat.of(twin_buf.ptr, attn.max_rows, o.hq * D, .bf16), .one = Mat.of(one_buf.ptr, 1, o.hq * D, .bf16), .host = try gpa.alloc(u8, 2 * attn.max_rows * rb) };
    defer gpa.free(k.host);
    defer {
        k.clearSolo();
        k.solo.deinit(gpa);
    }
    var ps: std.ArrayList(usize) = .empty;
    defer ps.deinit(gpa);
    var forms: [4]usize = @splat(0);
    for (o.kinds) |kind| {
        k.clearSolo();
        k.kind = kind;
        const glob = kind == .global;
        var bb: [max_bounds]usize = undefined;
        const bounds = walkBounds(&bb, kind, s, o.kx_g, P);
        if (bounds.len == 0) continue;
        ps.clearRetainingCapacity();
        for (bounds) |b| for (b - walk_before..b + walk_after + 1) |p| {
            if (ps.items.len == 0 or p > ps.items[ps.items.len - 1]) try ps.append(gpa, p);
        };
        std.debug.print("GRAPH S={d} {t} walk: boundaries {any}, {d} positions from {d} to {d}\n", .{ s, kind, bounds, ps.items.len, ps.items[0], ps.items[ps.items.len - 1] });
        for ([_]usize{ 1, attn.max_rows }) |R| for (graphVariants(s, o.kx_g)) |v| {
            if (v.global_only and !glob) continue;
            for (std.enums.values(KvForm), 0..) |f, fi| {
                if (!glob and f != .contiguous) continue;
                if ((f == .run or f == .table) and rig.arena == null) continue;
                if (f == .run and R != 1) continue;
                try k.walkOne(v, f, R, ps.items);
                forms[fi] += 1;
            }
        };
    }
    const ok = k.bad_solo == 0 and k.bad_twin == 0 and k.bad_log == 0 and k.ctl_failed == k.ctl and k.ctl > 0;
    std.debug.print("{s} graph replay kx: S {d}: {d} CUDA graphs captured and instantiated ({d} nodes; contiguous {d}, pool pages {d}, ring on a run {d}, moving page table {d}), {d} replays ({d} at a position other than the capture's, {d} rows), against kx-solo: {d} replays differ, against the eager launch of the same form: {d} differ; run moves {d} ({d} graph nodes repointed from the RunLog, {d} not recorded or not repointed), page-table moves {d}; controls {d}/{d} failed as required\n", .{ if (ok) "PASS" else "FAIL", s, k.caps, k.nodes, forms[0], forms[1], forms[2], forms[3], k.replays, k.moved, k.rows, k.bad_solo, k.bad_twin, k.run_moves, k.repointed, k.bad_log, k.table_moves, k.ctl_failed, k.ctl });
    return if (ok) 0 else 1;
}

/// `DIGEST op-check`: sha256 over a kind's kx-solo rows in position order (the bytes every variant matched), so two
/// builds' outputs can be compared without the rows.
fn printSoloDigest(gpa: std.mem.Allocator, s: usize, kind: Kind, solo: *const std.AutoHashMapUnmanaged(usize, []u8)) !void {
    const keys = try gpa.alloc(usize, solo.count());
    defer gpa.free(keys);
    var it = solo.keyIterator();
    var i: usize = 0;
    while (it.next()) |k| : (i += 1) keys[i] = k.*;
    std.mem.sort(usize, keys, {}, std.sort.asc(usize));
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (keys) |k| {
        h.update(std.mem.asBytes(&k));
        h.update(solo.get(k).?);
    }
    var dg: [32]u8 = undefined;
    h.final(&dg);
    std.debug.print("DIGEST op-check S={d} {t}: kx-solo {d} rows sha256 {s}\n", .{ s, kind, keys.len, std.fmt.bytesToHex(dg, .lower) });
}

// ---------------------------------------------------------------------------------------------------------------
// --bench

/// The bench's variants at a cell (a captured one twice: eager pos-pointer and graph).
const max_bv = 48;
/// A captured cell's CUDA graph holds this many of the cell's calls: its one launch is spread over them, as a step's
/// graph spreads its launch over its layers.
const graph_calls = 8;

/// Time quantiles of each of n closures (us a call), alternated in rounds of `per` calls: median, p10, p90 of the
/// rounds' means.
fn timeQ(io: std.Io, e: *const Env, n: usize, f: anytype, reps: usize) ![max_bv][3]f64 {
    const clock = std.Io.Clock.awake;
    const per = 10;
    const rounds = @max((reps + per - 1) / per, 3);
    if (rounds > 256) return error.TooManyReps;
    if (n > max_bv) return error.TooManyVariants;
    var t: [max_bv][256]f64 = undefined;
    for (0..n) |j| for (0..3) |_| try f.call(j);
    try e.sync();
    for (0..rounds) |r| for (0..n) |j| {
        const t0 = clock.now(io).toNanoseconds();
        for (0..per) |_| try f.call(j);
        try e.sync();
        t[j][r] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-3 / per;
    };
    var q: [max_bv][3]f64 = @splat(@splat(0));
    for (0..n) |j| {
        const x = t[j][0..rounds];
        std.mem.sort(f64, x, {}, std.sort.asc(f64));
        q[j] = .{ x[rounds / 2], x[(rounds - 1) / 10], x[rounds - 1 - (rounds - 1) / 10] };
    }
    return q;
}

fn packName(pk: ?bool) []const u8 {
    return if (pk) |x| (if (x) "-p1" else "-p0") else "";
}

const BVar = struct { name: []const u8, form: attn.Form, opt: attn.KxOpt = .{}, mode: Mode = .eager };

/// The BENCH line's label of a mode.
fn modeLabel(m: Mode) []const u8 {
    return switch (m) {
        .eager => "eager",
        .devpos => "eager pos-pointer",
        .graph => "captured",
    };
}

/// v at a cell: captured (R <= 8) as eager pos-pointer launches and as graph launches, else eager.
fn push(bv: *[max_bv]BVar, n: *usize, v: BVar, captured: bool) !void {
    const modes: []const Mode = if (captured) &.{ .devpos, .graph } else &.{.eager};
    for (modes) |m| {
        if (n.* == max_bv) return error.TooManyVariants;
        bv[n.*] = v;
        bv[n.*].mode = m;
        n.* += 1;
    }
}

/// A CUDA graph of `graph_calls` of the cell's call j launched as eager pos-pointer (f.eager), captured from the
/// stream (relaxed, as Engine.graphOf), instantiated and uploaded.
fn captureCalls(e: *const Env, f: anytype, j: usize) !cuda.graph.Exec {
    try e.sync();
    const stream = e.x.c.stream;
    try cuda.graph.beginCapture(stream, .relaxed);
    const body: anyerror!void = blk: {
        for (0..graph_calls) |_| f.eager(j) catch |err| break :blk err;
        break :blk {};
    };
    var g = cuda.graph.endCapture(stream) catch |err| {
        try body;
        return err;
    };
    defer g.deinit();
    try body;
    var x = try g.instantiate();
    errdefer x.deinit();
    try x.upload(stream);
    return x;
}

fn bench(gpa: std.mem.Allocator, io: std.Io, env: *Env, o: Opts, rl: []const usize, pool: ?ck.Pool) !void {
    var served: ?ck.Served = null;
    if (pool) |pl| {
        if (o.paged) served = try ck.Served.make(gpa, env.x.c.d, &env.x, o, pl);
    }
    const out = Mat.of(try env.x.b.zeroed(gpa, env.x.c.d, (std.mem.max(usize, rl)) * o.hq * D * 2), std.mem.max(usize, rl), o.hq * D, .bf16);
    var names: [max_bv][32]u8 = undefined;
    for (o.kinds) |kind| for (o.positions) |pos0| for (rl) |R| {
        if (pos0 + R > env.x.context) continue;
        const glob = kind == .global;
        const captured = R <= attn.max_rows;
        var bv: [max_bv]BVar = undefined;
        var n: usize = 0;
        if (glob) {
            if (o.v01) try push(&bv, &n, .{ .name = "v01", .form = .v01 }, captured);
            if (o.kx) for (o.kx_s) |s| for (o.kx_p) |pk| for (o.kx_gc) |gc| {
                var cb: [8]u8 = undefined;
                const cn = if (gc) |c| std.fmt.bufPrint(&cb, "-c{d}", .{c}) catch "" else "";
                if (n == max_bv) return error.TooManyVariants;
                try push(&bv, &n, .{ .name = try std.fmt.bufPrint(&names[n], "kx-s{d}{s}{s}", .{ s, packName(pk), cn }), .form = .kx, .opt = .{ .s = s, .g = o.kx_g, .pack = pk, .c = gc } }, captured);
            };
        } else {
            if (o.v01) {
                try push(&bv, &n, .{ .name = "v01", .form = .v01 }, false);
                if (captured) try push(&bv, &n, .{ .name = "v01-captured", .form = .v01 }, true);
            }
            if (o.kx) for (o.kx_c) |c| for (o.kx_p) |pk| for (o.kx_f) |gf| {
                if (n == max_bv) return error.TooManyVariants;
                const fname = if (gf == .auto) "" else @tagName(gf);
                try push(&bv, &n, .{ .name = try std.fmt.bufPrint(&names[n], "kx-c{d}{s}{s}{s}", .{ c, packName(pk), if (gf == .auto) "" else "-", fname }), .form = .kx, .opt = .{ .s = o.kx_s[0], .g = o.kx_g, .c = c, .pack = pk, .global = gf } }, captured);
            };
        }
        try env.sync();
        try env.x.setPos(pos0);
        const layers: usize = if (glob and served != null) served.?.n else 1;
        var execs: [max_bv]?cuda.graph.Exec = @splat(null);
        defer for (&execs) |*x| if (x.*) |*ex| ex.deinit();
        const F = struct {
            e: *const Env,
            sv: ?*const ck.Served,
            list: []const BVar,
            execs: *const [max_bv]?cuda.graph.Exec,
            kind: Kind,
            pos0: usize,
            R: usize,
            out: Mat,
            /// call j's eager launches (devpos: the captured plan, the position read on the device)
            fn eager(f: @This(), j: usize) !void {
                const v = f.list[j];
                const dev = v.mode != .eager;
                if (f.sv) |s| {
                    for (0..s.n) |l| try f.e.core(v.form, v.opt, dev, s.paged[l], true, f.pos0, f.R, f.out);
                } else try f.e.core(v.form, v.opt, dev, f.e.kvOf(f.kind, false), f.kind == .global, f.pos0, f.R, f.out);
            }
            fn call(f: @This(), j: usize) !void {
                if (f.list[j].mode == .graph) return f.execs[j].?.launchOn(f.e.x.c.stream);
                return f.eager(j);
            }
        };
        const f: F = .{ .e = env, .sv = if (glob and served != null) &served.? else null, .list = bv[0..n], .execs = &execs, .kind = kind, .pos0 = pos0, .R = R, .out = out };
        for (bv[0..n], 0..) |v, j| if (v.mode == .graph) {
            execs[j] = try captureCalls(env, f, j);
        };
        const q = try timeQ(io, env, n, f, o.reps);
        for (bv[0..n], 0..) |v, j| {
            var pb: [96]u8 = undefined;
            env.x.k.attn.form = v.form;
            env.x.k.attn.kxo = v.opt;
            const kv = if (glob and served != null) served.?.paged[0] else env.kvOf(kind, false);
            const p = attn.plan(&env.x.k.attn, R, pos0, if (v.mode != .eager) attn.Dev{ .posp = 1, .positions = env.x.context } else null, kv, o.hq, o.hk);
            const pt = if (v.form == .kx) planText(&pb, p) else (if (p.form == .split) (if (p.rows) "split+rows" else "split") else @tagName(p.form));
            const L: f64 = @floatFromInt(layers * @as(usize, if (v.mode == .graph) graph_calls else 1));
            std.debug.print("BENCH\t{t}\tpos0 {d}\trows {d}\t{s}\t{s}\tmedian {d:.2}\tp10 {d:.2}\tp90 {d:.2}\tus a layer\t{s}\n", .{ kind, pos0, R, modeLabel(v.mode), v.name, q[j][0] / L, q[j][1] / L, q[j][2] / L, pt });
        }
    };
}

// ---------------------------------------------------------------------------------------------------------------
// --limb

const Case = enum { L1, L2, L3, L4, L5, L6, L7 };
const case_names = [_][]const u8{ "L1 peaked", "L2 subnormal q", "L3 subnormal K, V", "L4 NaN q", "L5 NaN V", "L6 +Inf K", "L7 out of range" };
/// The cases the e2614fe control must fail (a row differs from its solo or its clean run).
const control_cases = [_]Case{ .L1, .L2, .L3, .L5, .L7 };

const lim_ctx = 4352; // 17 pages; the global block's rows 4088 .. 4103 leave the last page's rows 4104 .. 4351 unused
const lim_used = 4104;
const lim_global = 4088;
const lim_rows = 16;
const lim_sliding_cross = 1272;

/// The host's copies of the planted check's inputs (every position): q, rel (packed rows = positions), keys, values.
const Host = struct {
    q: []u16,
    rel: []u16,
    k: []u16,
    v: []u16,
};

/// The planted inputs of case c into h (from the pristine copy `base`); `clean`: the case's comparison run instead
/// (subnormals as +0, non-finite plants absent). pos0: the block's first position.
fn plant(h: *Host, base: *const Host, c: Case, clean: bool, pos0: usize, hq: usize, hk: usize, global: bool) void {
    @memcpy(h.q, base.q);
    @memcpy(h.rel, base.rel);
    @memcpy(h.k, base.k);
    @memcpy(h.v, base.v);
    const qa = struct {
        fn at(p: usize, head: usize, e: usize, n: usize) usize {
            return (p * n + head) * D + e;
        }
    }.at;
    switch (c) {
        .L1 => {
            // row 3, head 0: q'' = 22.625 e0, its rel' 0 (bias 0); its own key 22.625 e0, every other key of its
            // window -22.625 e0 (kv head 0): scores +45.25 and -45.25, the others' weights e^-90.5 (subnormal)
            const r3 = pos0 + 3;
            @memset(h.q[qa(r3, 0, 0, hq)..][0..D], 0);
            h.q[qa(r3, 0, 0, hq)] = 0x41b5;
            @memset(h.rel[(r3 * hq + 0) * dr ..][0..dr], 0);
            const lo = if (global) 0 else r3 -| (attn.window - 1);
            for (lo..r3 + 1) |p| {
                @memset(h.k[qa(p, 0, 0, hk)..][0..D], 0);
                h.k[qa(p, 0, 0, hk)] = if (p == r3) 0x41b5 else 0xc1b5;
            }
        },
        .L2 => {
            h.q[qa(pos0 + 5, 0, 0, hq)] = if (clean) 0 else 0x0001;
            h.q[qa(pos0 + 5, 0, 1, hq)] = if (clean) 0 else 0x8001;
        },
        .L3 => {
            h.k[qa(pos0 + 8, 0, 3, hk)] = if (clean) 0 else 0x0001;
            h.v[qa(pos0 + 8, 0, 3, hk)] = if (clean) 0 else 0x0001;
        },
        .L4 => if (!clean) {
            h.q[qa(pos0 + 7, 0, 0, hq)] = 0x7fc0;
        },
        .L5 => if (!clean) {
            h.v[qa(pos0 + 12, 0, 5, hk)] = 0x7fc0;
        },
        .L6 => {
            // rows 10-15 of kv group 0: q''[0] > 0 (so q[0] * +Inf is +Inf in every form), in both runs
            for (10..16) |t| for (0..hq / hk) |head| {
                h.q[qa(pos0 + t, head, 0, hq)] = 0x3f80;
            };
            if (!clean) h.k[qa(pos0 + 10, 0, 0, hk)] = 0x7f80;
        },
        .L7 => if (!clean) {
            // keys no row of the block reads: below the block's window start inside its first tile (sliding),
            // past its last row (every kind); every kv head
            const last = pos0 + lim_rows - 1;
            if (!global) {
                const lo = pos0 -| (attn.window - 1);
                for (lo / 64 * 64..lo) |p| for (0..hk) |g| {
                    h.k[qa(p, g, 0, hk)] = 0x7fc0;
                    h.v[qa(p, g, 1, hk)] = 0x0001;
                };
            }
            for (last + 1..@min(last + 40, lim_ctx)) |p| for (0..hk) |g| {
                h.k[qa(p, g, 2, hk)] = if (p % 2 == 0) 0x7fc0 else 0x0001;
                h.v[qa(p, g, 2, hk)] = if (p % 2 == 0) 0x0001 else 0x7fc0;
            };
        },
    }
}

const Block = struct { kind: Kind, pg: bool, pos0: usize };

fn limbBlocks(c: Case) [4]Block {
    const sl: usize = if (c == .L7) 1100 else 1000; // L7's plant below the window start needs lo(pos0) > 0
    // sliding at 1272: rows 1272 .. 1287 cross a window start split boundary (lo 255 -> 256 at row 1279) and a last
    // split boundary (p 1279 -> 1280) at once
    return .{ .{ .kind = .sliding, .pg = false, .pos0 = sl }, .{ .kind = .sliding, .pg = false, .pos0 = lim_sliding_cross }, .{ .kind = .global, .pg = false, .pos0 = lim_global }, .{ .kind = .global, .pg = true, .pos0 = lim_global } };
}

fn finite16(x: u16) bool {
    return x & 0x7f80 != 0x7f80;
}

const Limb = struct {
    gpa: std.mem.Allocator,
    env: *Env,
    own: *Own,
    o: Opts,
    h: Host,
    base: Host,
    out: [4]Mat,
    host: [4][]u16,

    /// The host's inputs onto the device: q, rel (packed rows = positions), keys and values (contiguous; the pool's
    /// rows 0 .. lim_used - 1 rewritten from them, its last page's rows past them NaN, or (L7 plant) 0x0001).
    fn upload(l: *Limb, c: Case, clean: bool) !void {
        const d = l.env.x.c.d;
        try put(d, l.env.q.ptr, l.h.q);
        try put(d, l.env.rel.ptr, l.h.rel);
        try put(d, l.env.x.b.k, l.h.k);
        try put(d, l.env.x.b.v, l.h.v);
        const pl = l.own.pool.?;
        try ck.writePool(&l.env.x, l.env.x.kvp, lim_used);
        // the last page's unused rows: NaN, or (L7) 0x0001
        const row = l.o.hk * D;
        const id: usize = @intCast(pl.table[lim_used / attn.split_keys]);
        const n = (attn.split_keys - lim_used % attn.split_keys) * row;
        const fill = try l.gpa.alloc(u16, n);
        defer l.gpa.free(fill);
        @memset(fill, if (c == .L7 and !clean) 0x0001 else 0xffff);
        for ([_]usize{ ck.run_k, ck.run_v }) |r| try put(d, pl.run(r, id) + (lim_used % attn.split_keys) * row * 2, fill);
    }

    /// kx's default dispatch (16 rows) and its other forms on block b, every row against its solo; the default run's
    /// outputs in host[0], its counter deltas returned.
    fn kxRun(l: *Limb, b: Block, s: usize, bad: *usize) ![3]u64 {
        const e = l.env;
        const kv = e.kvOf(b.kind, b.pg);
        const glob = b.kind == .global;
        const forms = if (glob) [_]attn.KxOpt{ .{ .s = s }, .{ .s = s, .global = .walk }, .{ .s = s, .c = 1 } } else [_]attn.KxOpt{ .{ .s = s }, .{ .s = s, .global = .walk }, .{ .s = s, .c = 4, .pack = false } };
        try e.sync();
        const c0 = try l.own.k.attn.kx.?.counts(e.x.c.d);
        try e.core(.kx, forms[0], false, kv, glob, b.pos0, lim_rows, l.out[0]);
        try e.sync();
        const c1 = try l.own.k.attn.kx.?.counts(e.x.c.d);
        for (forms[1..], 1..) |f, i| try e.core(.kx, f, false, kv, glob, b.pos0, lim_rows, l.out[i]);
        try e.sync();
        const w = l.o.hq * D;
        for (0..3) |i| try cuda.memory.download(e.x.c.d, l.out[i].ptr, std.mem.sliceAsBytes(l.host[i][0 .. lim_rows * w]));
        for (0..lim_rows) |t| {
            try e.core(.kx, forms[0], false, kv, glob, b.pos0 + t, 1, l.out[3]);
            try e.sync();
            try cuda.memory.download(e.x.c.d, l.out[3].ptr, std.mem.sliceAsBytes(l.host[3][0..w]));
            for (0..3) |i| if (!std.mem.eql(u16, l.host[3][0..w], l.host[i][t * w ..][0..w])) {
                bad.* += 1;
            };
        }
        var dl: [3]u64 = undefined;
        for (&dl, c1, c0) |*x, y, z| x.* = y -% z;
        return dl;
    }

    /// The e2614fe control on block b (contiguous or ring keys): whether a row differs from its solo, and its
    /// 16-row outputs in host[1].
    fn controlRun(l: *Limb, b: Block) !bool {
        const e = l.env;
        const kv = e.kvOf(b.kind, false);
        const glob = b.kind == .global;
        const w = l.o.hq * D;
        try e.control(kv, glob, b.pos0, lim_rows, l.out[1]);
        try e.sync();
        try cuda.memory.download(e.x.c.d, l.out[1].ptr, std.mem.sliceAsBytes(l.host[1][0 .. lim_rows * w]));
        var differs = false;
        for (0..lim_rows) |t| {
            try e.control(kv, glob, b.pos0 + t, 1, l.out[3]);
            try e.sync();
            try cuda.memory.download(e.x.c.d, l.out[3].ptr, std.mem.sliceAsBytes(l.host[3][0..w]));
            differs = differs or !std.mem.eql(u16, l.host[3][0..w], l.host[1][t * w ..][0..w]);
        }
        return differs;
    }
};

fn put(d: *const cuda.Driver, dst: u64, src: []const u16) !void {
    const view: cuda.DeviceBuffer = .{ .d = d, .ptr = dst, .len = src.len * 2, .borrowed = true };
    try view.upload(0, std.mem.sliceAsBytes(src));
}

/// tf-inkling2-attn --limb: cases L1-L7 on kx (sliding ring at 1000, L7 at 1100, and at 1272 across two split
/// boundaries; global at 4088 contiguous and paged) at each --kx-s, and the e2614fe control.
pub fn limb(gpa: std.mem.Allocator, io: std.Io, o0: Opts) !u8 {
    _ = io;
    var o = o0;
    o.context = lim_ctx;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rng = prng.random();
    var own: Own = undefined;
    var env: Env = undefined;
    try setup(gpa, a, &own, &env, o, &.{0}, lim_ctx, rng);
    defer own.deinit(gpa);
    own.pool = try ck.makePool(gpa, &own.driver, rng, &env.x, o);
    const d = &own.driver;
    const nq = lim_ctx * o.hq * D;
    const nk = lim_ctx * o.hk * D;
    var l: Limb = .{ .gpa = gpa, .env = &env, .own = &own, .o = o, .h = undefined, .base = undefined, .out = undefined, .host = undefined };
    for ([_]*Host{ &l.h, &l.base }) |h| h.* = .{ .q = try a.alloc(u16, nq), .rel = try a.alloc(u16, lim_ctx * o.hq * dr), .k = try a.alloc(u16, nk), .v = try a.alloc(u16, nk) };
    try cuda.memory.download(d, env.q.ptr, std.mem.sliceAsBytes(l.base.q));
    try cuda.memory.download(d, env.rel.ptr, std.mem.sliceAsBytes(l.base.rel));
    try cuda.memory.download(d, env.x.b.k, std.mem.sliceAsBytes(l.base.k));
    try cuda.memory.download(d, env.x.b.v, std.mem.sliceAsBytes(l.base.v));
    for (&l.out, &l.host) |*m, *h| {
        m.* = Mat.of(try env.x.b.zeroed(gpa, d, lim_rows * o.hq * D * 2), lim_rows, o.hq * D, .bf16);
        h.* = try a.alloc(u16, lim_rows * o.hq * D);
    }
    const clean_out = try a.alloc(u16, lim_rows * o.hq * D);
    const w = o.hq * D;
    const group = o.hq / o.hk;
    var cb: [160]u8 = undefined;
    own.k.attn.form = .kx;
    own.k.attn.kxo = .{ .s = o.kx_s[0] };
    std.debug.print("{s}\n", .{own.k.attn.census(&cb)});
    var passed: usize = 0;
    var ctl_failed: usize = 0;
    // DIGEST limb: sha256 over every case's clean run and kx default-dispatch run, in run order
    var dh = std.crypto.hash.sha2.Sha256.init(.{});
    for (std.enums.values(Case), 0..) |c, ci| {
        var ok = true;
        var why: [256]u8 = undefined;
        var ww: std.Io.Writer = .fixed(&why);
        for (o.kx_s) |s| for (limbBlocks(c)) |b| {
            const glob = b.kind == .global;
            // the clean run (cases with one)
            const has_clean = c != .L1;
            if (has_clean) {
                plant(&l.h, &l.base, c, true, b.pos0, o.hq, o.hk, glob);
                try l.upload(c, true);
                try env.core(.kx, .{ .s = s }, false, env.kvOf(b.kind, b.pg), glob, b.pos0, lim_rows, l.out[0]);
                try env.sync();
                try cuda.memory.download(d, l.out[0].ptr, std.mem.sliceAsBytes(clean_out));
                dh.update(std.mem.sliceAsBytes(clean_out));
            }
            plant(&l.h, &l.base, c, false, b.pos0, o.hq, o.hk, glob);
            try l.upload(c, false);
            var bad: usize = 0;
            const dl = try l.kxRun(b, s, &bad);
            const got = l.host[0][0 .. lim_rows * w];
            dh.update(std.mem.sliceAsBytes(got));
            var req = true;
            switch (c) {
                .L1 => req = dl[1] > 0,
                .L2, .L3 => req = dl[2] > 0 and std.mem.eql(u16, got, clean_out),
                .L4 => {
                    req = dl[0] == 1;
                    for (0..lim_rows) |t| for (0..o.hq) |h| {
                        const x = got[t * w + h * D ..][0..D];
                        const y = clean_out[t * w + h * D ..][0..D];
                        if (t == 7 and h == 0) {
                            for (x) |v| req = req and v == 0x7fc0;
                        } else req = req and std.mem.eql(u16, x, y);
                    };
                },
                .L5 => for (0..lim_rows) |t| for (0..o.hq) |h| {
                    const x = got[t * w + h * D ..][0..D];
                    if (t < 12) {
                        req = req and std.mem.eql(u16, x, clean_out[t * w + h * D ..][0..D]);
                    } else for (x, 0..) |v, col| {
                        req = req and if (h < group and col == 5) v == 0x7fc0 else finite16(v);
                    }
                },
                .L6 => for (0..lim_rows) |t| for (0..o.hq) |h| {
                    const x = got[t * w + h * D ..][0..D];
                    if (t < 10) {
                        req = req and std.mem.eql(u16, x, clean_out[t * w + h * D ..][0..D]);
                    } else for (x) |v| {
                        req = req and if (h < group) v == 0x7fc0 else finite16(v);
                    }
                },
                .L7 => req = dl[0] == 0 and dl[2] == 0 and std.mem.eql(u16, got, clean_out),
            }
            if (bad > 0 or !req) {
                ok = false;
                ww.print(" [S={d} {t}{s} pos0 {d}: {d} row/form pairs differ from solo, case rule {s}; counters {d}/{d}/{d}]", .{ s, b.kind, if (b.pg) " paged" else "", b.pos0, bad, if (req) "held" else "FAILED", dl[0], dl[1], dl[2] }) catch {};
            }
        };
        if (ok) passed += 1;
        std.debug.print("LIMB {s}: {s}{s}\n", .{ case_names[ci], if (ok) "PASS" else "FAIL", ww.buffered() });
        // the control: e2614fe's block-wide fallback must couple a row to its block-mates (or to the stale bytes)
        if (std.mem.indexOfScalar(Case, &control_cases, c) != null) {
            var failed_as_required = false;
            for (limbBlocks(c)) |b| {
                if (b.pg or (c == .L7 and b.kind == .global)) continue;
                const glob = b.kind == .global;
                var clean_ctl: ?[]u16 = null;
                if (c != .L1) {
                    plant(&l.h, &l.base, c, true, b.pos0, o.hq, o.hk, glob);
                    try l.upload(c, true);
                    try env.control(env.kvOf(b.kind, false), glob, b.pos0, lim_rows, l.out[2]);
                    try env.sync();
                    try cuda.memory.download(d, l.out[2].ptr, std.mem.sliceAsBytes(l.host[2][0 .. lim_rows * w]));
                    clean_ctl = l.host[2][0 .. lim_rows * w];
                }
                plant(&l.h, &l.base, c, false, b.pos0, o.hq, o.hk, glob);
                try l.upload(c, false);
                var differs = try l.controlRun(b);
                if (clean_ctl) |cc| {
                    // rows the plant cannot reach in the case's semantics must equal the clean run
                    const reach: usize = switch (c) {
                        .L2, .L3, .L7 => 0, // whole output = the clean run
                        .L5 => 12, // rows 0-11
                        else => lim_rows,
                    };
                    for (0..lim_rows) |t| if (t < reach or reach == 0) {
                        differs = differs or !std.mem.eql(u16, l.host[1][t * w ..][0..w], cc[t * w ..][0..w]);
                    };
                }
                failed_as_required = failed_as_required or differs;
            }
            if (failed_as_required) ctl_failed += 1;
            std.debug.print("LIMB control {s} (e2614fe): {s}\n", .{ case_names[ci], if (failed_as_required) "failed as required" else "DID NOT FAIL" });
        }
    }
    const n_cases = std.enums.values(Case).len;
    var dg: [32]u8 = undefined;
    dh.final(&dg);
    std.debug.print("DIGEST limb: kx sha256 {s}\n", .{std.fmt.bytesToHex(dg, .lower)});
    own.k.attn.printExceptions(d);
    std.debug.print("LIMB summary: {d}/{d} PASS, control {d}/{d} failed as required\n", .{ passed, n_cases, ctl_failed, control_cases.len });
    return if (passed == n_cases and ctl_failed == control_cases.len) 0 else 1;
}
