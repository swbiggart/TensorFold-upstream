//! tf-inkling2-attn --op-check / --bench: G-attn's `core` on pseudo-random inputs (no model, no capture) at a rank's
//! head shape, each forward of R rows at pos0 run through every path that must give the same bits:
//!   rows      the row passes (tf_ink2_attn_rstats / _rpv, R <= 8)
//!   prev      the previous kernels (TF_INK2_ATTN_ROWS=0: stats1 / pv1 for one row, the general passes for more)
//!   fused     the one-launch form
//!   graph     the row passes as a captured step runs them (the position read on the device; sliding layers over the
//!             derived 5-6 splits, global layers over at most grid_splits, each block walking its splits),
//!             `graph-prev` the previous kernels over every split of the slot, `graph-narrow` the row passes on a
//!             7-split global grid (many rounds a block), `graph-full` stage 1's captured step (every split of the
//!             slot launched, each block folding the row's pairs itself)
//!   fold      the row passes with the pairs folded in tf_ink2_attn_fold at any split count, `nofold` never (each
//!             second-pass block folds them), `prev-fold` the previous kernels with the fold kernel
//! and every row's output against the one-row forward at its own position (a row's bits never depend on the
//! forward's other rows). Inputs are per position, so a row at position p sees the same q', rel', sink and gate in
//! every forward.
//!
//! Paged (global layers, the shared KV page pool): each variant runs again on the `_pg` kernels against a pool whose
//! pages hold `--runs` runs of 256 rows (the layer's keys and values at runs 6 and 7, the drafter's at the last two),
//! page ids shuffled, every byte NaN until written, the table's unused entries on a NaN scrub page. The pool's rows are
//! written by tf_ink2_attn_kv_write_pg in chunks of 1 to 2,048 rows (some at a device position, as a captured step),
//! and every written page is compared with the contiguous cache (and the runs beside them must still be NaN). Each
//! paged variant must equal its contiguous one. The drafter's global-layer sites
//! (inkling2_dspark.cu rope, rows, block attention) run on a position-indexed table and on the pool: the tables'
//! bytes and the attention's partials must be equal.
//!
//! --bench times `core` alone (the variants alternated in rounds) at each position and row count: global layers
//! contiguous against paged (eager and, up to 8 rows, as a captured step; the drafter's block attention too), sliding
//! layers as before (eager with the in-block fold and with the fold kernel, captured over the full grid and as served).
//! Global layers are timed as a served step reads them (`--layers 8`, the default): one call is a step over 8 layers,
//! each with its own keys and values (8 contiguous caches; the pool's runs 2l and 2l + 1), so no call finds its own rows
//! in L2 from the call before; the time shown is a layer's (the step's / 8). With `--layout soa` a third form, the ring
//! kernels on the pool's own memory (each run an array of pages), is the control: paged against it compares the kernel
//! forms on the same memory. `--layers 1` is the earlier bench (one layer's calls back to back). With --evict-l2 every
//! layer's call first overwrites L2 (cold), the memsets' own time taken off.
const std = @import("std");
const cuda = @import("cuda");
const ic = @import("inkling2_cuda");

const attn = ic.attn;
const Mat = ic.ops.Mat;
const D = ic.weights.head_dim;
const dr = ic.weights.d_rel;

pub const usage =
    \\usage: tf-inkling2-attn --op-check|--bench [options]   (pseudo-random inputs, no model)
    \\  --heads HQ,HK      the rank's query and key heads (default 12,2: W = 4; 48,8 is W = 1)
    \\  --positions P,...  forwards' first positions (default 0,1,31,255,256,1000,1022,1023,1024,1500,3040,3119,8191,
    \\                     16383,16384,32700,65536,100000,131000)
    \\  --rows R,...       forward rows (default 1,2,3,4,5,6,7,8,16,2048; --bench 1..8)
    \\  --context N        the slot's positions: global layers' keys, a captured step's grid (default 131072 + 2048)
    \\  --kinds K,...      sliding, global (default both)
    \\  --reps N           --bench: calls a variant, a position and a row count (default 200)
    \\  --grid N           the `graph` variant's global grid in splits (default cuda_attn.zig grid_splits)
    \\  --runs N           a pool page's runs of 256 rows (default 18: 8 global layers' K and V, the drafter's 2)
    \\  --drafter 0|1      the drafter's sites (default 1 with global layers)
    \\  --paged 0|1        --bench: global layers contiguous against paged (default 1), or (0) the ring kernels alone as
    \\                     sliding layers are timed (the base build's global bench, for a build-against-build A/B)
    \\  --layout L         the pool's layout: aos (a page's runs side by side, `--runs` runs a page; default) or soa
    \\                     (each run its own array of pages: page n's rows of a layer n runs from the layer's first)
    \\  --ids I            page ids: shuffled (default) or sequential; alloc-fresh / alloc-churn: the served pool's
    \\                     allocator (cuda_pages.zig) over 4 contexts' pages, the stream opened fourth on a fresh pool,
    \\                     or after --churn N seeded opens and closes of 4 streams of 1 .. context pages
    \\  --churn N          alloc-churn's opens and closes (default 400)
    \\  --break-entry S    negative control: after the pool is written, the table's entry S points at the scrub page
    \\                     (the paged runs of every forward reaching split S must then differ)
    \\  --evict-l2 MIB     --bench, paged: a MIB memset before every layer's call (larger than L2: cold), its own time
    \\                     taken off
    \\  --layers N         --bench, paged: a call is a served step over N global layers, each its own keys and values
    \\                     (default 8; 1: one layer's calls back to back); --layout soa adds the ring kernels on the pool
    \\  --switch D,...    --op-check: instead of the forms, a stream's life on a served pool at each depth D: its one-row
    \\                     steps on the ring kernels while its pages are one run (the served fallback) and on the page
    \\                     table once a growth breaks the run, verify windows on the table, a new stream on a run again;
    \\                     every output against the contiguous cache (see checkSwitch)
    \\  --switch-off N     negative control for --switch: the ring view N pages off the run
    \\  --seed S
    \\
;

const Kind = enum { sliding, global };
/// The model's global layers a rank (the served bench's step).
const max_layers = 8;
const Layout = enum { aos, soa };

pub const Opts = struct {
    bench: bool = false,
    hq: usize = 12,
    hk: usize = 2,
    positions: []const usize = &.{ 0, 1, 31, 255, 256, 1000, 1022, 1023, 1024, 1500, 3040, 3119, 8191, 16383, 16384, 32700, 65536, 100000, 131000 },
    rows: ?[]const usize = null,
    context: usize = 131072 + 2048,
    kinds: []const Kind = &.{ .sliding, .global },
    reps: usize = 200,
    seed: u64 = 0x5eed_a77e,
    grid: usize = attn.grid_splits,
    runs: usize = 18,
    drafter: bool = true,
    paged: bool = true,
    break_entry: ?usize = null,
    layout: Layout = .aos,
    sequential: bool = false,
    /// --ids alloc-fresh / alloc-churn: the stream's pages from the served allocator (see the usage)
    alloc: ?enum { fresh, churn } = null,
    churn: usize = 400,
    /// --evict-l2 MIB (bench, global layers paged): a MIB memset before every timed call, so a call finds none of its
    /// own rows in L2 (a decode step walks every layer between two calls on one layer's rows); the memset's own time
    /// (timed alone in the same rounds) is taken off
    evict_mib: usize = 0,
    /// --layers N (bench, global layers paged): a timed call is a step over N layers' own caches (1: one layer)
    layers: usize = max_layers,
    /// --switch D,...: the run-switch check at these depths instead of the forms; --switch-off N its negative control
    switch_depths: ?[]const usize = null,
    switch_off: u32 = 0,

    fn rowList(o: *const Opts) []const usize {
        return o.rows orelse if (o.bench) &.{ 1, 2, 3, 4, 5, 6, 7, 8 } else &.{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 2048 };
    }
};

fn list(a: std.mem.Allocator, v: []const u8) ![]usize {
    var out: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, v, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseInt(usize, x, 10));
    return out.items;
}

pub fn parse(a: std.mem.Allocator, args: []const [:0]const u8) !?Opts {
    var o: Opts = .{ .bench = std.mem.eql(u8, args[1], "--bench") };
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        const eql = std.mem.eql;
        if (eql(u8, k, "--heads")) {
            const h = try list(a, v);
            if (h.len != 2) return null;
            o.hq = h[0];
            o.hk = h[1];
        } else if (eql(u8, k, "--positions")) o.positions = try list(a, v) else if (eql(u8, k, "--rows")) o.rows = try list(a, v) else if (eql(u8, k, "--context")) o.context = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--reps")) o.reps = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--grid")) o.grid = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--runs")) o.runs = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--drafter")) o.drafter = !eql(u8, v, "0") else if (eql(u8, k, "--paged")) o.paged = !eql(u8, v, "0") else if (eql(u8, k, "--layout")) o.layout = std.meta.stringToEnum(Layout, v) orelse return null else if (eql(u8, k, "--ids")) {
            o.sequential = eql(u8, v, "sequential");
            o.alloc = if (eql(u8, v, "alloc-fresh")) .fresh else if (eql(u8, v, "alloc-churn")) .churn else null;
        } else if (eql(u8, k, "--churn")) o.churn = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--break-entry")) o.break_entry = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--seed")) o.seed = try std.fmt.parseInt(u64, v, 0) else if (eql(u8, k, "--evict-l2")) o.evict_mib = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--layers")) o.layers = try std.fmt.parseInt(usize, v, 10) else if (eql(u8, k, "--switch")) o.switch_depths = try list(a, v) else if (eql(u8, k, "--switch-off")) o.switch_off = try std.fmt.parseInt(u32, v, 10) else if (eql(u8, k, "--kinds")) {
            var ks: std.ArrayList(Kind) = .empty;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| try ks.append(a, std.meta.stringToEnum(Kind, x) orelse return null);
            o.kinds = ks.items;
        } else return null;
    }
    if (i != args.len or o.hk == 0 or o.hq != attn.group * o.hk or o.reps == 0 or o.runs < 10) return null;
    if (o.bench and (o.layers == 0 or o.layers > max_layers or 2 * o.layers > o.runs)) return null;
    for (o.positions) |p| for (o.rowList()) |R| if (p + R > o.context) return null;
    if (o.switch_depths) |ds| for (ds) |dp| if (dp + switch_steps + 2 * attn.split_keys > o.context) return null;
    return o;
}

const Variant = enum { rows, prev, fused, graph, @"graph-prev", @"graph-narrow", @"graph-full", fold, nofold, @"prev-fold" };

/// A variant on the contiguous cache or on the pool's pages.
const Run = struct { v: Variant, pg: bool = false };

/// The pool's layout: a page `runs` runs of 256 rows; the target layer's keys and values at runs 6 and 7, the
/// drafter's at the last two.
const run_k = 6;
const run_v = 7;

/// The per-position inputs and the caches (filled once), and a forward's output buffers.
const Bufs = struct {
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,
    q: Mat = .{},
    rel: Mat = .{},
    sink: Mat = .{},
    gate: Mat = .{},
    proj: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    posp: u64 = 0,
    pool: u64 = 0,
    pages: u64 = 0,
    outs: [24]Mat = undefined,

    fn deinit(b: *Bufs, gpa: std.mem.Allocator) void {
        for (b.own.items) |*x| x.free();
        b.own.deinit(gpa);
    }

    /// n bf16 values, uniform in [-amp, amp], from the generator, into dst (or a new buffer).
    fn randomInto(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: std.Random, dst: u64, n: usize, amp: f32) !void {
        const host = try gpa.alloc(u16, n);
        defer gpa.free(host);
        for (host) |*x| {
            const f: f32 = (rng.float(f32) * 2 - 1) * amp;
            x.* = @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        const view: cuda.DeviceBuffer = .{ .d = d, .ptr = dst, .len = n * 2, .borrowed = true };
        try view.upload(0, std.mem.sliceAsBytes(host));
    }

    fn random(b: *Bufs, gpa: std.mem.Allocator, d: *const cuda.Driver, rng: std.Random, n: usize, amp: f32) !u64 {
        const p = try b.filled(gpa, d, n * 2, 0);
        try randomInto(gpa, d, rng, p, n, amp);
        return p;
    }

    fn filled(b: *Bufs, gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize, value: u8) !u64 {
        var buf = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer buf.free();
        try buf.fill8(value, null);
        try b.own.append(gpa, buf);
        return buf.ptr;
    }

    fn zeroed(b: *Bufs, gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) !u64 {
        return b.filled(gpa, d, bytes, 0);
    }
};

const Ctx = struct {
    c: ic.ops.Ctx,
    k: *ic.kernels.Kernels,
    scratch: *ic.ops.Scratch,
    b: *Bufs,
    l: *ic.weights.Layer,
    kv: attn.Kv,
    /// the same keys and values in the pool's pages (global layers)
    kvp: attn.Kv = undefined,
    context: usize,
    grid: usize,

    /// The device position the graph variants read (call with the stream idle).
    fn setPos(x: *const Ctx, pos0: usize) !void {
        const p: i32 = @intCast(pos0);
        const view: cuda.DeviceBuffer = .{ .d = x.c.d, .ptr = x.b.posp, .len = 4, .borrowed = true };
        try view.upload(0, std.mem.asBytes(&p));
    }

    /// `core` for the forward of R rows at pos0 through variant v into `out` (the graph variants: after setPos(pos0)).
    fn run(x: *const Ctx, r: Run, pos0: usize, R: usize, out: Mat) !void {
        try x.runOn(r.v, if (r.pg) x.kvp else x.kv, pos0, R, out);
    }

    /// `run` on the caches `kv` (a served step's layer).
    fn runOn(x: *const Ctx, v: Variant, kv: attn.Kv, pos0: usize, R: usize, out: Mat) !void {
        const k = &x.k.attn;
        k.rows = if (v == .prev or v == .@"graph-prev" or v == .@"prev-fold") 0 else attn.max_rows;
        k.form = if (v == .fused) .fused else .auto;
        k.grid = switch (v) {
            .@"graph-narrow" => 7,
            .@"graph-full" => std.math.maxInt(u32),
            else => x.grid,
        };
        k.fold_min = switch (v) {
            .fold, .@"prev-fold" => 0,
            .nofold, .@"graph-full" => std.math.maxInt(u32),
            else => attn.fold_splits,
        };
        const dev: ?attn.Dev = switch (v) {
            .graph, .@"graph-prev", .@"graph-narrow", .@"graph-full" => .{ .posp = x.b.posp, .positions = x.context },
            else => null,
        };
        x.scratch.reset();
        try attn.core(&x.c, x.l, pos0, dev, x.b.q.slice(pos0, R), x.b.rel.slice(pos0, R), x.b.sink.slice(pos0, R), x.b.gate.slice(pos0, R), kv, out.slice(0, R));
    }
};

const all_variants = [_]Variant{ .rows, .prev, .fused, .graph, .@"graph-prev", .@"graph-narrow", .@"graph-full", .fold, .nofold, .@"prev-fold" };
const wide_variants = [_]Variant{ .rows, .fused, .fold, .nofold };

/// The runs of a forward: the variants on the contiguous cache, then (global layers) each again on the pages.
fn runsOf(buf: *[2 * all_variants.len]Run, R: usize, paged: bool) []const Run {
    const vs: []const Variant = if (R <= attn.max_rows) &all_variants else &wide_variants;
    for (vs, 0..) |v, i| buf[i] = .{ .v = v };
    if (!paged) return buf[0..vs.len];
    for (vs, 0..) |v, i| buf[vs.len + i] = .{ .v = v, .pg = true };
    return buf[0 .. 2 * vs.len];
}

/// The pool, its page table and the contiguous cache's rows written into it. Run r of page id n (a layer's keys or
/// values for the page's 256 positions) is at base + r * region + n * stride: aos, a page's runs side by side
/// (stride the page, region a run); soa, each run an array of pages (stride a run, region the array).
const Pool = struct {
    entries: usize,
    pages: usize,
    base: u64,
    stride: usize,
    region: usize,
    run_bytes: usize,
    runs: usize,
    table: []i32,

    fn run(pl: Pool, r: usize, id: usize) u64 {
        return pl.base + r * pl.region + id * pl.stride;
    }

    /// Run r of the stream's page 0 (the kernels' kc / vc), and the kernels' page stride in elements.
    fn first(pl: Pool, r: usize) u64 {
        return pl.base + r * pl.region;
    }

    fn elems(pl: Pool) usize {
        return pl.stride / 2;
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
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
    const rows = o.rowList();
    var most: usize = 0;
    for (rows) |R| most = @max(most, R);
    // the split form's largest scratch at this context (the fold kernel at any split count, as the variants run it)
    k.attn.fold_min = 0;
    const need = attn.scratchNeed(&k.attn, o.hq, o.hk, o.context, ic.engine.prompt_rows, attn.max_rows);
    k.attn.fold_min = attn.fold_splits;
    var scratch = try ic.ops.Scratch.init(&driver, need + (16 << 20));
    defer scratch.deinit();
    var prng = std.Random.DefaultPrng.init(o.seed);
    const rng = prng.random();
    const P = o.context;
    var b: Bufs = .{};
    defer b.deinit(gpa);
    // q' within the clip's reach, the relative states clipped to 8, sink and gate: random at the positions a forward
    // reads (each position's own values, the same in every forward), zero elsewhere
    b.q = Mat.of(try b.zeroed(gpa, &driver, P * o.hq * D * 2), P, o.hq * D, .bf16);
    b.rel = Mat.of(try b.zeroed(gpa, &driver, P * o.hq * dr * 2), P, o.hq * dr, .bf16);
    b.sink = Mat.of(try b.zeroed(gpa, &driver, P * o.hq * 2), P, o.hq, .bf16);
    b.gate = Mat.of(try b.zeroed(gpa, &driver, P * o.hq * D * 2), P, o.hq * D, .bf16);
    {
        // the positions any forward reads, merged into ranges
        var lo: usize = 0;
        var hi: usize = 0;
        const sorted = try gpa.dupe(usize, o.positions);
        defer gpa.free(sorted);
        std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
        for (sorted, 0..) |p0, i| {
            const end = @min(P, p0 + most);
            if (i == 0) {
                lo = p0;
                hi = end;
            } else if (p0 <= hi) hi = @max(hi, end) else {
                try fillInputs(gpa, &driver, rng, &b, o, lo, hi);
                lo = p0;
                hi = end;
            }
        }
        if (sorted.len > 0) try fillInputs(gpa, &driver, rng, &b, o, lo, hi);
    }
    b.proj = try b.random(gpa, &driver, rng, dr * attn.rel_extent, 1.0);
    // keys within the clip's reach, values as the unweighted norm leaves them
    b.k = try b.random(gpa, &driver, rng, P * o.hk * D, 1.5);
    b.v = try b.random(gpa, &driver, rng, P * o.hk * D, 2.0);
    b.posp = try b.zeroed(gpa, &driver, 256);
    for (&b.outs) |*m| m.* = Mat.of(try b.zeroed(gpa, &driver, most * o.hq * D * 2), most, o.hq * D, .bf16);
    var l: ic.weights.Layer = .{ .index = 0, .global = false, .boundary = false, .hq = @intCast(o.hq), .hk = @intCast(o.hk), .nq = 1, .experts = .{ .format = .nvfp4, .lo = 0, .hi = 0 } };
    l.rel_proj = b.proj;
    var x: Ctx = .{ .c = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch }, .k = &k, .scratch = &scratch, .b = &b, .l = &l, .kv = undefined, .context = P, .grid = @max(o.grid, 1) };
    const global = std.mem.indexOfScalar(Kind, o.kinds, .global) != null;
    if (o.switch_depths) |ds| {
        l.global = true;
        x.kv = .{ .k = b.k, .v = b.v, .slots = P, .heads = o.hk, .window = 0 };
        return if (try checkSwitch(gpa, &driver, rng, &x, o, ds) == 0) 0 else 1;
    }
    var failed: usize = 0;
    var pool: ?Pool = null;
    defer if (pool) |pl| gpa.free(pl.table);
    if (global) {
        pool = try makePool(gpa, &driver, rng, &x, o);
        if (!o.bench) failed += try checkPool(gpa, &driver, &x, o, pool.?);
        if (o.break_entry) |e| {
            const pl = pool.?;
            if (e >= pl.entries) return error.BadBreakEntry;
            const scrub = pl.table[pl.entries];
            const view: cuda.DeviceBuffer = .{ .d = &driver, .ptr = b.pages + e * 4, .len = 4, .borrowed = true };
            try view.upload(0, std.mem.asBytes(&scrub));
            std.debug.print("negative control: table entry {d} (positions {d} .. {d}) now on the NaN scrub page\n", .{ e, e * attn.split_keys, e * attn.split_keys + attn.split_keys - 1 });
        }
    }
    var evict: ?cuda.DeviceBuffer = null;
    defer if (evict) |*e| e.free();
    if (o.bench and o.evict_mib > 0) evict = try cuda.DeviceBuffer.alloc(&driver, o.evict_mib << 20);
    var served: ?Served = null;
    if (o.bench and global and o.paged and o.layers > 1) served = try Served.make(gpa, &driver, &x, o, pool.?);
    const row_bytes = o.hq * D * 2;
    const host = try gpa.alloc(u8, b.outs.len * most * row_bytes);
    defer gpa.free(host);
    const one = try gpa.alloc(u8, row_bytes);
    defer gpa.free(one);
    var checked: usize = 0;
    var paged_runs: usize = 0;
    for (o.kinds) |kind| {
        l.global = kind == .global;
        // the engine's slots: global layers a slot a position, sliding layers the ring of 1023 + the widest forward
        x.kv = .{ .k = b.k, .v = b.v, .slots = if (l.global) P else attn.window - 1 + ic.engine.prompt_rows, .heads = o.hk, .window = if (l.global) 0 else attn.window };
        for (o.positions) |pos0| for (rows) |R| {
            if (pos0 + R > P) continue;
            if (o.bench) {
                if (l.global and served != null) try benchServed(io, &x, &served.?, evict, pos0, R, o.reps) else if (l.global and o.paged and evict != null) try benchPagedCold(io, &x, evict.?, pos0, R, o.reps) else if (l.global and o.paged) try benchPaged(io, &x, pos0, R, o.reps) else try bench(io, &x, kind, pos0, R, o.reps);
                continue;
            }
            var rb: [2 * all_variants.len]Run = undefined;
            const vs = runsOf(&rb, R, l.global);
            try stream.synchronize();
            try x.setPos(pos0);
            for (vs, 0..) |v, j| try x.run(v, pos0, R, b.outs[j]);
            try stream.synchronize();
            for (vs, 0..) |_, j| try cuda.memory.download(&driver, b.outs[j].ptr, host[j * most * row_bytes ..][0 .. R * row_bytes]);
            var bad: usize = 0;
            const half = if (l.global) vs.len / 2 else vs.len;
            for (vs[1..], 1..) |v, j| {
                // a contiguous variant against the first (every form gives the same bits), a paged one against its
                // contiguous twin
                const ref = if (v.pg) j - half else 0;
                if (v.pg) paged_runs += 1;
                if (!std.mem.eql(u8, host[ref * most * row_bytes ..][0 .. R * row_bytes], host[j * most * row_bytes ..][0 .. R * row_bytes])) {
                    bad += 1;
                    std.debug.print("  DIFFER {t} pos0 {d} rows {d}: {t}{s} against {t}{s}\n", .{ kind, pos0, R, v.v, if (v.pg) " paged" else "", vs[ref].v, if (vs[ref].pg) " paged" else "" });
                }
            }
            // each row against the one-row forward at its position (first and last 8 rows of a wide forward)
            for (0..R) |t| {
                if (R > 16 and t >= 8 and t + 8 < R) continue;
                if (R == 1) break;
                try x.run(.{ .v = .rows }, pos0 + t, 1, b.outs[b.outs.len - 1]);
                try stream.synchronize();
                try cuda.memory.download(&driver, b.outs[b.outs.len - 1].ptr, one);
                if (!std.mem.eql(u8, one, host[t * row_bytes ..][0..row_bytes])) {
                    bad += 1;
                    std.debug.print("  DIFFER {t} pos0 {d} rows {d}: row {d} against its one-row forward\n", .{ kind, pos0, R, t });
                }
            }
            checked += 1;
            if (bad > 0) failed += 1;
        };
    }
    if (o.bench) {
        if (global and o.drafter and o.paged) {
            if (served) |*sv| try benchDrafterServed(gpa, io, &driver, rng, &x, o, pool.?, sv) else try benchDrafter(gpa, io, &driver, rng, &x, o, pool.?);
        }
        return 0;
    }
    if (global and o.drafter) failed += try checkDrafter(gpa, &driver, rng, &x, o, pool.?);
    std.debug.print("{s} attn op check: heads {d}/{d}, context {d}: {d} forwards (kind x position x rows), {d} paged runs, {d} with a difference; variants rows, prev, fused, graph, graph-prev, graph-narrow, graph-full, fold, nofold, prev-fold at 1..{d} rows (global layers each also paged, page ids shuffled, {d} runs a page), every row against its one-row forward; scratch need {d:.1} MiB\n", .{ if (failed == 0) "PASS" else "FAIL", o.hq, o.hk, P, checked, paged_runs, failed, attn.max_rows, o.runs, @as(f64, @floatFromInt(need)) / (1 << 20) });
    return if (failed == 0) 0 else 1;
}

/// Random q', rel', sink and gate for positions lo .. hi - 1.
fn fillInputs(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: std.Random, b: *Bufs, o: Opts, lo: usize, hi: usize) !void {
    const n = hi - lo;
    try Bufs.randomInto(gpa, d, rng, b.q.row(lo), n * o.hq * D, 1.5);
    try Bufs.randomInto(gpa, d, rng, b.rel.row(lo), n * o.hq * dr, 1.0);
    try Bufs.randomInto(gpa, d, rng, b.sink.row(lo), n * o.hq, 3.0);
    try Bufs.randomInto(gpa, d, rng, b.gate.row(lo), n * o.hq * D, 3.0);
}

/// The pool (every byte NaN), its shuffled page table (the unused entries on a NaN scrub page), and the contiguous
/// cache's rows written into its pages by the paged cache write, in chunks of 1 to 2,048 rows (every fourth chunk at a
/// device position, as a captured step writes).
fn makePool(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: std.Random, x: *Ctx, o: Opts) !Pool {
    const P = o.context;
    const row = o.hk * D * 2;
    const run_bytes = attn.split_keys * row;
    const entries = (P + attn.split_keys - 1) / attn.split_keys;
    const spare = 8;
    // the served allocator's layouts: 4 contexts' pages (and the scrub page, last)
    const pages = if (o.alloc != null) 4 * entries + 1 else entries + spare + 1;
    x.b.pool = try x.b.filled(gpa, d, pages * o.runs * run_bytes, 0xff);
    const stride = if (o.layout == .aos) o.runs * run_bytes else run_bytes;
    const region = if (o.layout == .aos) run_bytes else pages * run_bytes;
    // ids: a shuffled permutation of the pool's pages; the first `entries` are the stream's, the next the scrub page
    const ids = try gpa.alloc(i32, pages);
    defer gpa.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i);
    if (o.alloc) |mode| try allocIds(gpa, rng, mode == .churn, o.churn, entries, ids) else if (!o.sequential) rng.shuffle(i32, ids);
    const table = try gpa.alloc(i32, entries + spare);
    for (table, 0..) |*v, i| v.* = if (i < entries) ids[i] else ids[entries];
    const tb = try x.b.zeroed(gpa, d, table.len * 4);
    {
        const view: cuda.DeviceBuffer = .{ .d = d, .ptr = tb, .len = table.len * 4, .borrowed = true };
        try view.upload(0, std.mem.sliceAsBytes(table));
    }
    x.b.pages = tb;
    const pl: Pool = .{ .entries = entries, .pages = pages, .base = x.b.pool, .stride = stride, .region = region, .run_bytes = run_bytes, .runs = o.runs, .table = table };
    x.kvp = .{ .k = pl.first(run_k), .v = pl.first(run_v), .slots = entries * attn.split_keys, .heads = o.hk, .window = 0, .pages = tb, .page = pl.elems() };
    try writePool(x, x.kvp, P);
    return pl;
}

/// The stream's `entries` page ids as the served allocator places them (ids[0 .. entries)), and the scrub page (the
/// pool's last) at ids[entries]: on a fresh pool of 4 x entries pages after three other streams' opens, or after
/// `churn` seeded opens and closes of 4 streams of 1 .. entries pages (then others close until the stream fits).
fn allocIds(gpa: std.mem.Allocator, rng: std.Random, churn: bool, steps: usize, entries: usize, ids: []i32) !void {
    const pg = ic.pages;
    const total = ids.len - 1;
    var a = try pg.Alloc.init(gpa, total);
    defer a.deinit();
    var ls: [4]pg.Lease = @splat(.{});
    defer for (&ls) |*l| a.finish(l);
    if (churn) {
        for (0..steps) |_| {
            const s = rng.uintLessThan(usize, ls.len);
            if (ls[s].ids.items.len > 0) {
                a.finish(&ls[s]);
                continue;
            }
            const n = 1 + rng.uintLessThan(usize, entries);
            a.promise(&ls[s], n) catch continue;
            try a.cover(&ls[s], n);
        }
        a.finish(&ls[0]);
        var k: usize = 1;
        while (!a.fits(entries)) : (k += 1) a.finish(&ls[k]);
    } else for (ls[1..]) |*l| {
        try a.promise(l, entries / 2 + 1);
        try a.cover(l, entries / 2 + 1);
    }
    const fb = a.stats.fallbacks;
    try a.promise(&ls[0], entries);
    try a.cover(&ls[0], entries);
    for (ls[0].ids.items, 0..) |v, i| ids[i] = @intCast(v);
    ids[entries] = @intCast(total);
    std.debug.print("page ids: the served allocator {s}: the stream's {d} pages in {d} run(s) from id {d}{s}; covers {d}, one run {d}, scattered {d}\n", .{ if (churn) "after churn" else "on a fresh pool", entries, pg.Alloc.runsOf(ls[0].ids.items), ls[0].ids.items[0], if (a.stats.fallbacks > fb) " (scattered: no free run held them)" else "", a.stats.covers, a.stats.one_run, a.stats.fallbacks });
}

/// The contiguous cache's rows written into the pool's pages at `kvp` by the paged cache write, in chunks of 1 to 2,048
/// rows (every fourth chunk at a device position, as a captured step writes).
fn writePool(x: *Ctx, kvp: attn.Kv, P: usize) !void {
    const km = Mat.of(x.b.k, P, kvp.heads * D, .bf16);
    const vm = Mat.of(x.b.v, P, kvp.heads * D, .bf16);
    const chunks = [_]usize{ 2048, 1, 7, 300, 255, 1024, 2048, 8, 513, 2 };
    var pos: usize = 0;
    var i: usize = 0;
    while (pos < P) : (i += 1) {
        const n = @min(chunks[i % chunks.len], P - pos);
        var dev: ?attn.Dev = null;
        if (i % 4 == 3) {
            try x.c.stream.synchronize();
            try x.setPos(pos);
            dev = .{ .posp = x.b.posp, .positions = P };
        }
        try ic.state.kvWrite(&x.c, kvp, pos, dev, km.slice(pos, n), vm.slice(pos, n));
        pos += n;
    }
    try x.c.stream.synchronize();
}

/// The paged cache write's result: every page's key and value runs equal to the contiguous rows (positions past the
/// context still NaN), the runs on either side and every page off the table still NaN.
fn checkPool(gpa: std.mem.Allocator, d: *const cuda.Driver, x: *const Ctx, o: Opts, pl: Pool) !usize {
    const P = o.context;
    const row = o.hk * D * 2;
    const got = try gpa.alloc(u8, pl.run_bytes);
    defer gpa.free(got);
    const want = try gpa.alloc(u8, pl.run_bytes);
    defer gpa.free(want);
    var used = try std.DynamicBitSet.initEmpty(gpa, pl.pages);
    defer used.deinit();
    var bad: usize = 0;
    for (0..pl.entries) |s| {
        const id: usize = @intCast(pl.table[s]);
        used.set(id);
        const n = @min(attn.split_keys, P - s * attn.split_keys);
        for ([_]struct { run: usize, src: u64 }{ .{ .run = run_k, .src = x.b.k }, .{ .run = run_v, .src = x.b.v } }) |r| {
            @memset(want, 0xff);
            try cuda.memory.download(d, r.src + s * attn.split_keys * row, want[0 .. n * row]);
            try cuda.memory.download(d, pl.run(r.run, id), got);
            if (!std.mem.eql(u8, got, want)) bad += 1;
        }
        for ([_]usize{ run_k - 1, run_v + 1 }) |r| {
            try cuda.memory.download(d, pl.run(r, id), got);
            if (std.mem.indexOfNone(u8, got, &.{0xff}) != null) bad += 1;
        }
    }
    var off_table: usize = 0;
    for (0..pl.pages) |id| {
        if (used.isSet(id)) continue;
        off_table += 1;
        for (0..pl.runs) |r| {
            try cuda.memory.download(d, pl.run(r, id), got);
            if (std.mem.indexOfNone(u8, got, &.{0xff}) != null) bad += 1;
        }
    }
    std.debug.print("{s} kv_write_pg: {d} positions into {d} {s} pages ({d} runs a page, {t}): each page's key and value runs equal the contiguous rows, the runs beside them and {d} pages off the table untouched; {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", P, pl.entries, if (o.sequential) "sequential" else "shuffled", o.runs, o.layout, off_table, bad });
    return if (bad == 0) 0 else 1;
}

/// The drafter's global-layer sites (inkling2_dspark.cu): rope's and rows' writes into a position-indexed table and
/// into the pool's last two runs, the tables compared page by page; then the block attention over contexts at each
/// position, its partials (ml, acc) compared.
fn checkDrafter(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: std.Random, x: *const Ctx, o: Opts, pl: Pool) !usize {
    var dk = try ic.dspark.Kernels.load(d);
    defer dk.mod.unload();
    const P = o.context;
    const hq = o.hq;
    const hk = o.hk;
    const row = hk * D * 2;
    const b = x.b;
    const run_dk = o.runs - 2;
    const run_dv = o.runs - 1;
    const pk = pl.first(run_dk);
    const pv = pl.first(run_dv);
    // the tables: the contiguous rows to start with, in both forms
    const tk = try b.zeroed(gpa, d, P * row);
    const tv = try b.zeroed(gpa, d, P * row);
    try x.c.copy(tk, b.k, P * row);
    try x.c.copy(tv, b.v, P * row);
    for (0..pl.entries) |s| {
        const n = @min(attn.split_keys, P - s * attn.split_keys);
        const id: usize = @intCast(pl.table[s]);
        try x.c.copy(pl.run(run_dk, id), b.k + s * attn.split_keys * row, n * row);
        try x.c.copy(pl.run(run_dv, id), b.v + s * attn.split_keys * row, n * row);
    }
    const most = 2048;
    const src = try b.random(gpa, d, rng, most * 2 * hk * D, 2.0); // [most][k | v] as absorb's kvo
    const gain = try b.zeroed(gpa, d, D * 4);
    {
        var g: [D]f32 = undefined;
        for (&g) |*v| v.* = 0.5 + rng.float(f32);
        const view: cuda.DeviceBuffer = .{ .d = d, .ptr = gain, .len = D * 4, .borrowed = true };
        try view.upload(0, std.mem.asBytes(&g));
    }
    const theta: f32 = 1e6;
    const eps: f32 = 1e-6;
    var cases: usize = 0;
    for (o.positions) |p0| for ([_]usize{ 1, 8, 300, 2048 }) |R| {
        if (p0 + R > P) continue;
        const captured = cases % 2 == 0;
        cases += 1;
        if (captured) {
            try x.c.stream.synchronize();
            try x.setPos(p0);
        }
        const posp: u64 = if (captured) b.posp else 0;
        // the table form: captured, rows at out + p * ld; eager, `out` the first row's (absorb's two calls)
        var a: cuda.Args = .{};
        a.add(src);
        a.add(@as(i64, @intCast(2 * hk * D)));
        a.add(gain);
        a.add(if (captured) tk else tk + p0 * row);
        a.add(@as(i64, @intCast(hk * D)));
        a.add(@as(i64, @intCast(p0)));
        a.add(theta);
        a.add(eps);
        a.add(posp);
        a.add(@as(i64, 0));
        a.add(@as(i64, 1024));
        try cuda.launch.launch(dk.rope, .{ .grid = .{ .x = @intCast(hk), .y = @intCast(R), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, x.c.stream, &a);
        var ap: cuda.Args = .{};
        ap.add(src);
        ap.add(@as(i64, @intCast(2 * hk * D)));
        ap.add(gain);
        ap.add(pk);
        ap.add(@as(i64, @intCast(hk * D)));
        ap.add(@as(i64, @intCast(p0)));
        ap.add(theta);
        ap.add(eps);
        ap.add(posp);
        ap.add(b.pages);
        ap.add(@as(i64, @intCast(pl.elems())));
        try cuda.launch.launch(dk.rope_pg, .{ .grid = .{ .x = @intCast(hk), .y = @intCast(R), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, x.c.stream, &ap);
        // the values: rows pos0 .. of the table, and of the pages
        var r: cuda.Args = .{};
        r.add(src + hk * D * 2);
        r.add(@as(i64, @intCast(2 * row)));
        r.add(tv);
        r.add(@as(i64, @intCast(row)));
        r.add(@as(i32, @intCast(row / 16)));
        r.add(@as(i64, @intCast(p0)));
        r.add(posp);
        r.add(@as(i64, 0));
        r.add(@as(i64, 1024));
        try cuda.launch.launch(dk.rows, .{ .grid = .{ .x = @intCast(R), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, x.c.stream, &r);
        var rp: cuda.Args = .{};
        rp.add(src + hk * D * 2);
        rp.add(@as(i64, @intCast(2 * row)));
        rp.add(pv);
        rp.add(@as(i64, @intCast(row)));
        rp.add(@as(i32, @intCast(row / 16)));
        rp.add(@as(i64, @intCast(p0)));
        rp.add(posp);
        rp.add(b.pages);
        rp.add(@as(i64, @intCast(pl.stride)));
        try cuda.launch.launch(dk.rows_pg, .{ .grid = .{ .x = @intCast(R), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, x.c.stream, &rp);
    };
    try x.c.stream.synchronize();
    // the tables, page by page
    var bad_tab: usize = 0;
    {
        const got = try gpa.alloc(u8, pl.run_bytes);
        defer gpa.free(got);
        const want = try gpa.alloc(u8, pl.run_bytes);
        defer gpa.free(want);
        for (0..pl.entries) |s| {
            const n = @min(attn.split_keys, P - s * attn.split_keys);
            const id: usize = @intCast(pl.table[s]);
            for ([_][2]u64{ .{ tk, run_dk }, .{ tv, run_dv } }) |t| {
                try cuda.memory.download(d, t[0] + s * attn.split_keys * row, want[0 .. n * row]);
                try cuda.memory.download(d, pl.run(t[1], id), got[0 .. n * row]);
                if (!std.mem.eql(u8, want[0 .. n * row], got[0 .. n * row])) bad_tab += 1;
            }
        }
    }
    // the block attention: 8 rows over the context [0, p) and the block, as the global layer runs it
    const T = 8;
    const nsplit_most = (P + T + attn.split_keys - 1) / attn.split_keys;
    const q = try b.random(gpa, d, rng, T * hq * D, 1.5);
    const bk = try b.random(gpa, d, rng, T * hk * D, 1.5);
    const bv = try b.random(gpa, d, rng, T * hk * D, 2.0);
    const ml_bytes = hq * nsplit_most * T * 2 * 4;
    const acc_bytes = hq * nsplit_most * T * D * 4;
    const ml = [2]u64{ try b.zeroed(gpa, d, ml_bytes), try b.zeroed(gpa, d, ml_bytes) };
    const acc = [2]u64{ try b.zeroed(gpa, d, acc_bytes), try b.zeroed(gpa, d, acc_bytes) };
    const h0 = try gpa.alloc(u8, acc_bytes);
    defer gpa.free(h0);
    const h1 = try gpa.alloc(u8, acc_bytes);
    defer gpa.free(h1);
    var attn_cases: usize = 0;
    var bad_attn: usize = 0;
    for (o.positions) |p| {
        if (p == 0 or p + T > P) continue;
        const nsplit = (p + T + attn.split_keys - 1) / attn.split_keys;
        for (0..2) |f| {
            var a: cuda.Args = .{};
            a.add(q);
            a.add(if (f == 0) tk else pk);
            a.add(if (f == 0) tv else pv);
            a.add(@as(i64, @intCast(hk * D)));
            if (f == 0) a.add(@as(i64, 0));
            a.add(@as(i32, @intCast(p)));
            a.add(bk);
            a.add(bv);
            a.add(@as(i64, @intCast(hk * D)));
            a.add(@as(i32, T));
            a.add(@as(i32, @intCast(hq)));
            a.add(@as(i32, @intCast(hq / hk)));
            a.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
            a.add(ml[f]);
            a.add(acc[f]);
            if (f == 1) {
                a.add(b.pages);
                a.add(@as(i64, @intCast(pl.elems())));
            }
            try cuda.launch.launch(if (f == 0) dk.attn else dk.attn_pg, .{ .grid = .{ .x = @intCast(nsplit), .y = @intCast(hq), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, x.c.stream, &a);
        }
        try x.c.stream.synchronize();
        const nm = hq * nsplit * T * 2 * 4;
        const na = hq * nsplit * T * D * 4;
        try cuda.memory.download(d, ml[0], h0[0..nm]);
        try cuda.memory.download(d, ml[1], h1[0..nm]);
        var same = std.mem.eql(u8, h0[0..nm], h1[0..nm]);
        try cuda.memory.download(d, acc[0], h0[0..na]);
        try cuda.memory.download(d, acc[1], h1[0..na]);
        same = same and std.mem.eql(u8, h0[0..na], h1[0..na]);
        attn_cases += 1;
        if (!same) {
            bad_attn += 1;
            std.debug.print("  DIFFER drafter attention at context {d}\n", .{p});
        }
    }
    const bad = bad_tab + bad_attn;
    std.debug.print("{s} drafter global layer on pages: rope and rows at {d} (position, rows) cases (half as a captured step) into the table and into shuffled pages, {d} page runs differ; block attention (8 rows) at {d} contexts, partials ml and acc: {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", cases, bad_tab, attn_cases, bad_attn });
    return if (bad == 0) 0 else 1;
}

/// The median of the rounds' means (us a call) of each of `n` closures, alternated in rounds of `per` calls.
fn timeAlternated(io: std.Io, x: *const Ctx, n: usize, f: anytype, reps: usize) ![max_alt]f64 {
    const clock = std.Io.Clock.awake;
    const per = 20;
    const rounds = (reps + per - 1) / per;
    if (rounds > 256) return error.TooManyReps;
    if (n > max_alt) return error.TooManyVariants;
    var t: [max_alt][256]f64 = undefined;
    for (0..n) |j| for (0..3) |_| try f.call(j); // warm-up
    try x.c.stream.synchronize();
    for (0..rounds) |r| for (0..n) |j| {
        const t0 = clock.now(io).toNanoseconds();
        for (0..per) |_| try f.call(j);
        try x.c.stream.synchronize();
        t[j][r] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-3 / per;
    };
    var med: [max_alt]f64 = @splat(0);
    for (0..n) |j| {
        std.mem.sort(f64, t[j][0..rounds], {}, std.sort.asc(f64));
        med[j] = t[j][rounds / 2];
    }
    return med;
}

/// A global layer's `core` as served, contiguous against paged: eager (`rows`: the row passes up to 8 rows, the fused
/// form above) and, up to 8 rows, as a captured step; alternated.
fn benchPaged(io: std.Io, x: *const Ctx, pos0: usize, R: usize, reps: usize) !void {
    try x.c.stream.synchronize();
    try x.setPos(pos0);
    const graph = R <= attn.max_rows;
    const F = struct {
        x: *const Ctx,
        pos0: usize,
        R: usize,
        fn call(f: @This(), j: usize) !void {
            const runs = [_]Run{ .{ .v = .rows }, .{ .v = .rows, .pg = true }, .{ .v = .graph }, .{ .v = .graph, .pg = true } };
            try f.x.run(runs[j], f.pos0, f.R, f.x.b.outs[0]);
        }
    };
    const f: F = .{ .x = x, .pos0 = pos0, .R = R };
    if (graph) {
        const m = try timeAlternated(io, x, 4, f, reps);
        std.debug.print("bench global pos0 {d:>7} rows {d:>4}: eager contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%); captured contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%)\n", .{ pos0, R, m[0], m[1], pct(m[1], m[0]), m[2], m[3], pct(m[3], m[2]) });
    } else {
        const m = try timeAlternated(io, x, 2, f, reps);
        std.debug.print("bench global pos0 {d:>7} rows {d:>4}: eager contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%)\n", .{ pos0, R, m[0], m[1], pct(m[1], m[0]) });
    }
}

/// benchPaged with `ev` (a buffer larger than L2) set before every call: variant 0 the memset alone, then each form
/// after its memset; each form's time is its round's less the memset's.
fn benchPagedCold(io: std.Io, x: *const Ctx, ev: cuda.DeviceBuffer, pos0: usize, R: usize, reps: usize) !void {
    try x.c.stream.synchronize();
    try x.setPos(pos0);
    const graph = R <= attn.max_rows;
    const F = struct {
        x: *const Ctx,
        ev: cuda.DeviceBuffer,
        pos0: usize,
        R: usize,
        fn call(f: @This(), j: usize) !void {
            try f.ev.fill8(@intCast(j), f.x.c.stream.handle);
            if (j == 0) return;
            const runs = [_]Run{ .{ .v = .rows }, .{ .v = .rows, .pg = true }, .{ .v = .graph }, .{ .v = .graph, .pg = true } };
            try f.x.run(runs[j - 1], f.pos0, f.R, f.x.b.outs[0]);
        }
    };
    const f: F = .{ .x = x, .ev = ev, .pos0 = pos0, .R = R };
    if (graph) {
        const m = try timeAlternated(io, x, 5, f, reps);
        const a = [_]f64{ m[1] - m[0], m[2] - m[0], m[3] - m[0], m[4] - m[0] };
        std.debug.print("bench global pos0 {d:>7} rows {d:>4} (L2 evicted, memset {d:.1} us): eager contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%); captured contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%)\n", .{ pos0, R, m[0], a[0], a[1], pct(a[1], a[0]), a[2], a[3], pct(a[3], a[2]) });
    } else {
        const m = try timeAlternated(io, x, 3, f, reps);
        const a = [_]f64{ m[1] - m[0], m[2] - m[0] };
        std.debug.print("bench global pos0 {d:>7} rows {d:>4} (L2 evicted, memset {d:.1} us): eager contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%)\n", .{ pos0, R, m[0], a[0], a[1], pct(a[1], a[0]) });
    }
}

const max_alt = 10;

/// The served bench's caches: each of `n` global layers' own keys and values contiguous (as v0.1 allocates a slot's
/// layers), in the pool's runs 2l and 2l + 1, and (soa layout) the ring kernels' view of those runs.
const Served = struct {
    n: usize,
    cont: [max_layers]attn.Kv,
    paged: [max_layers]attn.Kv,
    /// the ring kernels on the pool's memory (soa: run r is an array of pages; with shuffled ids a position reads
    /// another page's rows, the same bytes a call in the same pattern; the outputs are not compared)
    ring_pool: ?[max_layers]attn.Kv,
    /// the served fallback (soa, the stream's pages one run of consecutive ids): the ring kernels on the stream's own
    /// run, as State.kvFor gives a one-row forward (base the run's first page)
    run: ?[max_layers]attn.Kv = null,
    run_start: u32 = 0,

    fn make(gpa: std.mem.Allocator, d: *const cuda.Driver, x: *Ctx, o: Opts, pl: Pool) !Served {
        const P = o.context;
        const bytes = P * o.hk * D * 2;
        var s: Served = .{ .n = o.layers, .cont = undefined, .paged = undefined, .ring_pool = if (o.layout == .soa) @as([max_layers]attn.Kv, undefined) else null };
        for (0..s.n) |l| {
            s.cont[l] = .{ .k = x.b.k, .v = x.b.v, .slots = P, .heads = o.hk, .window = 0 };
            if (l > 0) {
                const k = try x.b.zeroed(gpa, d, bytes);
                const v = try x.b.zeroed(gpa, d, bytes);
                const kb: cuda.DeviceBuffer = .{ .d = d, .ptr = k, .len = bytes, .borrowed = true };
                const vb: cuda.DeviceBuffer = .{ .d = d, .ptr = v, .len = bytes, .borrowed = true };
                try kb.copyFrom(0, x.b.k, bytes, null);
                try vb.copyFrom(0, x.b.v, bytes, null);
                s.cont[l].k = k;
                s.cont[l].v = v;
            }
            s.paged[l] = x.kvp;
            s.paged[l].k = pl.first(2 * l);
            s.paged[l].v = pl.first(2 * l + 1);
            if (2 * l != run_k) try writePool(x, s.paged[l], P);
            if (s.ring_pool) |*r| r[l] = .{ .k = pl.first(2 * l), .v = pl.first(2 * l + 1), .slots = P, .heads = o.hk, .window = 0 };
        }
        if (o.layout == .soa) {
            const t = pl.table[0..pl.entries];
            var one = true;
            for (t[1..], t[0 .. t.len - 1]) |b, p| one = one and b == p + 1;
            if (one) {
                s.run_start = @intCast(t[0]);
                s.run = @as([max_layers]attn.Kv, undefined);
                for (0..s.n) |l| {
                    const off = @as(u64, s.run_start) * pl.stride;
                    s.run.?[l] = .{ .k = pl.first(2 * l) + off, .v = pl.first(2 * l + 1) + off, .slots = P, .heads = o.hk, .window = 0 };
                }
            }
        }
        try x.c.stream.synchronize();
        std.debug.print("served bench: a call is a step over {d} global layers (8 contiguous caches of {d} positions, the pool's runs 0 .. {d}, {t} layout, {s} ids){s}{s}\n", .{ s.n, P, 2 * s.n - 1, o.layout, if (o.alloc) |m| (if (m == .fresh) "alloc-fresh" else "alloc-churn") else if (o.sequential) "sequential" else "shuffled", if (s.ring_pool != null) "; control: the ring kernels on the pool's memory" else "", if (s.run != null) "; fallback: the ring kernels on the stream's run (one-row forwards as served)" else "" });
        return s;
    }

    fn kvOf(s: *const Served, form: usize, l: usize) attn.Kv {
        return switch (form) {
            0 => s.cont[l],
            1 => s.paged[l],
            2 => s.ring_pool.?[l],
            else => s.run.?[l],
        };
    }
};

/// A served step's global layers, contiguous against paged (and the ring kernels on the pool with --layout soa):
/// eager and, up to 8 rows, captured; alternated; us a layer. With `ev` every layer's call first overwrites L2 (the
/// memsets alone are variant 0, taken off the others).
fn benchServed(io: std.Io, x: *const Ctx, s: *const Served, ev: ?cuda.DeviceBuffer, pos0: usize, R: usize, reps: usize) !void {
    try x.c.stream.synchronize();
    try x.setPos(pos0);
    const graph = R <= attn.max_rows;
    const forms: usize = if (s.run != null) 4 else if (s.ring_pool != null) 3 else 2;
    const cold = ev != null;
    const Entry = struct { v: Variant, form: usize };
    var ents: [max_alt]Entry = undefined;
    var n: usize = 0;
    if (cold) {
        ents[n] = .{ .v = .rows, .form = max_alt };
        n += 1;
    }
    for ([_]Variant{ .rows, .graph }) |v| {
        if (v == .graph and !graph) break;
        for (0..forms) |fm| {
            ents[n] = .{ .v = v, .form = fm };
            n += 1;
        }
    }
    const F = struct {
        x: *const Ctx,
        s: *const Served,
        ev: ?cuda.DeviceBuffer,
        list: []const Entry,
        pos0: usize,
        R: usize,
        fn call(f: @This(), j: usize) !void {
            const e = f.list[j];
            for (0..f.s.n) |l| {
                if (f.ev) |b| try b.fill8(@intCast(j + l), f.x.c.stream.handle);
                if (e.form == max_alt) continue;
                try f.x.runOn(e.v, f.s.kvOf(e.form, l), f.pos0, f.R, f.x.b.outs[0]);
            }
        }
    };
    const m = try timeAlternated(io, x, n, F{ .x = x, .s = s, .ev = ev, .list = ents[0..n], .pos0 = pos0, .R = R }, reps);
    const L: f64 = @floatFromInt(s.n);
    const off: usize = @intFromBool(cold);
    var a: [max_alt]f64 = @splat(0);
    for (off..n) |j| a[j - off] = (m[j] - if (cold) m[0] else 0) / L;
    var line: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&line);
    for (0..@as(usize, if (graph) 2 else 1)) |g| {
        const t = a[g * forms ..];
        try w.print("{s} contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%)", .{ if (g == 0) "eager" else "; captured", t[0], t[1], pct(t[1], t[0]) });
        if (forms >= 3) try w.print(", ring-on-pool {d:>9.1} us (paged {d:>6.2}%)", .{ t[2], pct(t[1], t[2]) });
        if (forms == 4) try w.print(", fallback {d:>9.1} us ({d:>6.2}% of contiguous)", .{ t[3], pct(t[3], t[0]) });
    }
    var head: [96]u8 = undefined;
    const how = if (cold) try std.fmt.bufPrint(&head, ", L2 evicted before each layer, memset {d:.1} us", .{m[0] / L}) else "";
    std.debug.print("bench served global pos0 {d:>7} rows {d:>4} ({d} layers a step, us a layer{s}): {s}\n", .{ pos0, R, s.n, how, w.buffered() });
}

fn pct(a: f64, b: f64) f64 {
    return (a / b - 1.0) * 100.0;
}

/// The drafter's block attention (8 rows, its global layer) at each position, contiguous against paged; alternated.
fn benchDrafter(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, rng: std.Random, x: *const Ctx, o: Opts, pl: Pool) !void {
    var dk = try ic.dspark.Kernels.load(d);
    defer dk.mod.unload();
    const P = o.context;
    const hq = o.hq;
    const hk = o.hk;
    const b = x.b;
    const T = 8;
    const nsplit_most = (P + T + attn.split_keys - 1) / attn.split_keys;
    const q = try b.random(gpa, d, rng, T * hq * D, 1.5);
    const bk = try b.random(gpa, d, rng, T * hk * D, 1.5);
    const bv = try b.random(gpa, d, rng, T * hk * D, 2.0);
    const ml = try b.zeroed(gpa, d, hq * nsplit_most * T * 2 * 4);
    const acc = try b.zeroed(gpa, d, hq * nsplit_most * T * D * 4);
    const F = struct {
        x: *const Ctx,
        dk: *const ic.dspark.Kernels,
        p: usize,
        q: u64,
        bk: u64,
        bv: u64,
        ml: u64,
        acc: u64,
        hq: usize,
        hk: usize,
        pl: Pool,
        fn call(f: @This(), j: usize) !void {
            const nsplit = (f.p + T + attn.split_keys - 1) / attn.split_keys;
            var a: cuda.Args = .{};
            a.add(f.q);
            a.add(if (j == 0) f.x.b.k else f.pl.first(run_k));
            a.add(if (j == 0) f.x.b.v else f.pl.first(run_v));
            a.add(@as(i64, @intCast(f.hk * D)));
            if (j == 0) a.add(@as(i64, 0));
            a.add(@as(i32, @intCast(f.p)));
            a.add(f.bk);
            a.add(f.bv);
            a.add(@as(i64, @intCast(f.hk * D)));
            a.add(@as(i32, T));
            a.add(@as(i32, @intCast(f.hq)));
            a.add(@as(i32, @intCast(f.hq / f.hk)));
            a.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
            a.add(f.ml);
            a.add(f.acc);
            if (j == 1) {
                a.add(f.x.b.pages);
                a.add(@as(i64, @intCast(f.pl.elems())));
            }
            try cuda.launch.launch(if (j == 0) f.dk.attn else f.dk.attn_pg, .{ .grid = .{ .x = @intCast(nsplit), .y = @intCast(f.hq), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, f.x.c.stream, &a);
        }
    };
    for (o.positions) |p| {
        if (p == 0 or p + T > P) continue;
        const f: F = .{ .x = x, .dk = &dk, .p = p, .q = q, .bk = bk, .bv = bv, .ml = ml, .acc = acc, .hq = hq, .hk = hk, .pl = pl };
        const m = try timeAlternated(io, x, 2, f, o.reps);
        std.debug.print("bench drafter context {d:>7}: block attention contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%)\n", .{ p, m[0], m[1], pct(m[1], m[0]) });
    }
}

/// `core` alone, the variants alternated in rounds of 20 calls: us a call (median of the rounds' means).
fn bench(io: std.Io, x: *const Ctx, kind: Kind, pos0: usize, R: usize, reps: usize) !void {
    try x.c.stream.synchronize();
    try x.setPos(pos0);
    const F = struct {
        x: *const Ctx,
        pos0: usize,
        R: usize,
        fn call(f: @This(), j: usize) !void {
            const vs = [_]Variant{ .nofold, .rows, .@"graph-full", .graph };
            try f.x.run(.{ .v = vs[j] }, f.pos0, f.R, f.x.b.outs[0]);
        }
    };
    const med = try timeAlternated(io, x, 4, F{ .x = x, .pos0 = pos0, .R = R }, reps);
    std.debug.print("bench {t:<7} pos0 {d:>6} rows {d}: eager in-block fold {d:>7.1} us, fold kernel {d:>7.1} us ({d:>6.1}); graph-style (context {d}) stage 1 {d:>7.1} us, now (grid {d}) {d:>7.1} us ({d:>6.1})\n", .{ kind, pos0, R, med[0], med[1], med[1] - med[0], x.context, med[2], x.grid, med[3], med[3] - med[2] });
}

/// Decode steps a life of `--switch` takes at its depth.
const switch_steps = 640;

/// --switch: one-row launches on the ring kernels while a stream's pages are one run, as served (State.kvFor), and
/// the switch to the page table and back. A pool laid out as served (keys then values, each an array of pages; NaN
/// until written) with the served allocator; the stream's rows are the contiguous cache's. At each depth D, three
/// lives of a stream:
///   1. covered as one run after another stream's pages (so its run starts past 0); its prompt [0, D) written through
///      the table in chunks of 2,048 rows (paged kernels, as prompt forwards); then `switch_steps` decode steps: each
///      one row written and attended through the ring view of the run (eager and as a captured step), every eighth
///      step a verify window of 8 rows through the table instead (a mode switch each way); at each page edge the
///      stream grows: in place the first time (still one run), then, with the page after its last taken by
///      another stream, elsewhere (no longer one run: every later one-row step on the table, the fallback);
///   2. a new stream (the first's pages given back) covered as one run again: ring steps once more (a run restored),
///      with in-place growth;
///   3. the same on a pool churned by other streams (a run that does not start at a page a region's alignment).
/// Every output equals the contiguous cache's (the rows, prev and graph forms each) and the stream's pages hold the
/// contiguous rows. --switch-off N points the ring view N pages off the run: it must fail.
fn checkSwitch(gpa: std.mem.Allocator, d: *const cuda.Driver, rng: std.Random, x: *Ctx, o: Opts, depths: []const usize) !usize {
    const pg = ic.pages;
    const P = o.context;
    const row = o.hk * D * 2;
    const run_bytes = attn.split_keys * row;
    const entries = pg.pagesFor(P);
    const total = 3 * entries;
    const place: pg.Place = .{ .base = try x.b.filled(gpa, d, (total + 1) * 2 * run_bytes, 0xff), .pages = total + 1, .run = run_bytes };
    const scrub = place.scrub();
    const tb = try x.b.zeroed(gpa, d, entries * 4);
    const host = try gpa.alloc(u32, entries);
    defer gpa.free(host);
    const kvp: attn.Kv = .{ .k = place.region(0), .v = place.region(1), .slots = entries * attn.split_keys, .heads = o.hk, .window = 0, .pages = tb, .page = place.elems() };
    const km = Mat.of(x.b.k, P, o.hk * D, .bf16);
    const vm = Mat.of(x.b.v, P, o.hk * D, .bf16);
    const row_bytes = o.hq * D * 2;
    const want = try gpa.alloc(u8, attn.max_rows * row_bytes);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, attn.max_rows * row_bytes);
    defer gpa.free(got);
    var a = try pg.Alloc.init(gpa, total);
    defer a.deinit();
    var ls: [6]pg.Lease = @splat(.{});
    defer for (&ls) |*l| a.finish(l);
    const S = struct {
        ring1: usize = 0,
        table1: usize = 0,
        windows: usize = 0,
        switches: usize = 0,
        grown_in_place: usize = 0,
        grown_elsewhere: usize = 0,
        lives: usize = 0,
        bad: usize = 0,
        bad_pages: usize = 0,
        starts: [8]u32 = undefined,
    };
    var st: S = .{};
    const T = struct {
        /// the stream's table on the device (the rest of its entries on the scrub page); null unless one run
        fn upload(dd: *const cuda.Driver, xx: *Ctx, h: []u32, ids: []const u32, sc: u32, t: u64) !?u32 {
            @memcpy(h[0..ids.len], ids);
            @memset(h[ids.len..], sc);
            try xx.c.stream.synchronize();
            const view: cuda.DeviceBuffer = .{ .d = dd, .ptr = t, .len = h.len * 4, .borrowed = true };
            try view.upload(0, std.mem.sliceAsBytes(h));
            return if (pg.Alloc.runsOf(ids) == 1) ids[0] else null;
        }
    };
    for (depths) |D0| for (0..3) |life| {
        st.lives += 1;
        // the pool before the stream: life 0 another stream at the front; life 1 the same pool, the first stream's
        // pages given back; life 2 churned by other streams
        if (life == 0) {
            for (&ls) |*l| a.finish(l);
            try a.promise(&ls[1], entries / 4 + 3);
            try a.cover(&ls[1], entries / 4 + 3);
        } else if (life == 2) {
            for (&ls) |*l| a.finish(l);
            for (0..60) |_| {
                const k = 1 + rng.uintLessThan(usize, 4);
                if (ls[k].ids.items.len > 0) {
                    a.finish(&ls[k]);
                    continue;
                }
                const n = 1 + rng.uintLessThan(usize, entries / 2);
                a.promise(&ls[k], n) catch continue;
                try a.cover(&ls[k], n);
            }
        }
        const lease = &ls[0];
        a.finish(lease);
        const n0 = pg.pagesFor(D0 + 1);
        while (!a.fits(n0 + 4)) for (ls[1..]) |*l| if (l.ids.items.len > 0) {
            a.finish(l);
            break;
        };
        try a.promise(lease, n0);
        try a.cover(lease, n0);
        if (pg.Alloc.runsOf(lease.ids.items) != 1) return error.SwitchSetup; // a fresh cover is one run here
        try pg.fillIn(d, x.c.stream, place, 2, lease.ids.items, 0xff);
        var on_run = try T.upload(d, x, host, lease.ids.items, scrub, tb);
        st.starts[@min(st.lives - 1, st.starts.len - 1)] = on_run.?;
        // the prompt through the table
        var p: usize = 0;
        while (p < D0) {
            const n = @min(2048, D0 - p);
            try ic.state.kvWrite(&x.c, kvp, p, null, km.slice(p, n), vm.slice(p, n));
            p += n;
        }
        var growths: usize = 0;
        var last_mode: u8 = 0;
        var blocked: ?u32 = null;
        defer if (blocked) |id| {
            a.refs[id] = 0;
            a.nfree += 1;
        };
        var k: usize = 0;
        while (p < D0 + switch_steps) : (k += 1) {
            const window = k % 8 == 7;
            const R: usize = if (window) attn.max_rows else 1;
            // grow at a page edge (the stream's pages past p + R)
            while (p + R > lease.ids.items.len * attn.split_keys) {
                const last = lease.ids.items[lease.ids.items.len - 1];
                if (life == 0 and growths == 1 and last + 1 < total and a.refs[last + 1] == 0) {
                    // another holder takes the page after this one's last: the growth goes elsewhere
                    a.refs[last + 1] = 1;
                    a.nfree -= 1;
                    blocked = last + 1;
                }
                const in_place = last + 1 < total and a.refs[last + 1] == 0;
                try a.promise(lease, 1);
                try a.grow(lease, 1);
                if (in_place) st.grown_in_place += 1 else st.grown_elsewhere += 1;
                growths += 1;
                try pg.fillIn(d, x.c.stream, place, 2, lease.ids.items[lease.ids.items.len - 1 ..], 0xff);
                on_run = try T.upload(d, x, host, lease.ids.items, scrub, tb);
            }
            const ring = !window and on_run != null;
            var kv = kvp;
            if (ring) {
                const shift = if (on_run.? + o.switch_off + lease.ids.items.len < total) o.switch_off else 0;
                const off = @as(u64, on_run.? + shift) * place.run;
                kv = .{ .k = kvp.k + off, .v = kvp.v + off, .slots = P, .heads = o.hk, .window = 0, .run = on_run.? };
            }
            if (R == 1) {
                if (ring) st.ring1 += 1 else st.table1 += 1;
                const mode: u8 = if (ring) 1 else 2;
                if (last_mode != 0 and last_mode != mode) st.switches += 1;
                last_mode = mode;
            } else st.windows += 1;
            // the step's rows written in this step's form (one row as a captured step every other step)
            const graph_write = k % 2 == 1;
            try x.c.stream.synchronize();
            try x.setPos(p);
            const dev: ?attn.Dev = if (graph_write) .{ .posp = x.b.posp, .positions = P } else null;
            try ic.state.kvWrite(&x.c, kv, p, dev, km.slice(p, R), vm.slice(p, R));
            // the reference, then each form on the stream's pages
            try x.runOn(.rows, x.kv, p, R, x.b.outs[0]);
            try x.c.stream.synchronize();
            try cuda.memory.download(d, x.b.outs[0].ptr, want[0 .. R * row_bytes]);
            for ([_]Variant{ .rows, .prev, .graph }) |v| {
                try x.runOn(v, kv, p, R, x.b.outs[1]);
                try x.c.stream.synchronize();
                try cuda.memory.download(d, x.b.outs[1].ptr, got[0 .. R * row_bytes]);
                if (!std.mem.eql(u8, want[0 .. R * row_bytes], got[0 .. R * row_bytes])) {
                    st.bad += 1;
                    if (st.bad <= 8) std.debug.print("  DIFFER switch depth {d} life {d} pos {d} rows {d}: {t} on the {s}\n", .{ D0, life, p, R, v, if (ring) "ring view of the run" else "page table" });
                }
            }
            p += R;
        }
        // the stream's pages hold the contiguous rows (positions [0, p))
        {
            const pb = try gpa.alloc(u8, run_bytes);
            defer gpa.free(pb);
            const cb = try gpa.alloc(u8, run_bytes);
            defer gpa.free(cb);
            for (lease.ids.items, 0..) |id, j| {
                if (j * attn.split_keys >= p) break;
                const n = @min(attn.split_keys, p - j * attn.split_keys);
                for ([_][2]u64{ .{ x.b.k, place.region(0) }, .{ x.b.v, place.region(1) } }) |r| {
                    try cuda.memory.download(d, r[0] + j * run_bytes, cb[0 .. n * row]);
                    try cuda.memory.download(d, r[1] + @as(u64, id) * run_bytes, pb[0 .. n * row]);
                    if (!std.mem.eql(u8, cb[0 .. n * row], pb[0 .. n * row])) st.bad_pages += 1;
                }
            }
        }
        std.debug.print("  switch depth {d} life {d}: run from page {d}, {d} pages in {d} run(s) at the end, {d} growths\n", .{ D0, life, st.starts[@min(st.lives - 1, st.starts.len - 1)], lease.ids.items.len, pg.Alloc.runsOf(lease.ids.items), growths });
    };
    const bad = st.bad + st.bad_pages;
    std.debug.print("{s} run switch: heads {d}/{d}, context {d}, depths {any}, {d} lives: {d} one-row steps on the ring view of a run, {d} on the page table (pages not one run), {d} verify windows of {d} rows on the table, {d} mode switches; growths {d} in place, {d} elsewhere (a run broken); each step's rows, prev and graph forms against the contiguous cache: {d} differ; stream pages against the contiguous rows: {d} differ{s}\n", .{ if (bad == 0) "PASS" else "FAIL", o.hq, o.hk, P, depths, st.lives, st.ring1, st.table1, st.windows, attn.max_rows, st.switches, st.grown_in_place, st.grown_elsewhere, st.bad, st.bad_pages, if (o.switch_off > 0) " (negative control: the ring view off the run)" else "" });
    if (st.table1 == 0 or st.ring1 == 0 or st.switches == 0 or st.grown_elsewhere == 0) {
        std.debug.print("FAIL run switch: a case was not reached (one-row steps on the table {d}, on the ring {d}, switches {d}, growths elsewhere {d})\n", .{ st.table1, st.ring1, st.switches, st.grown_elsewhere });
        return 1;
    }
    return bad;
}

/// The drafter's block attention (8 rows over its global layer's context, then the block) timed as a served step
/// runs it: after the 8 global layers' verify window (8 rows each over its own keys and values on the pool's pages,
/// captured form), so its rows are not in L2 from the call before. Timed alone with events around its launch (the
/// layers' time not counted), alternated in rounds: contiguous (its own cache, as v0.1 and stage 2b keep it), paged
/// (the pool's last two runs through the table, as stage 2c would put it), and the ring kernel on those runs (the
/// stream's run when its pages are one run, else the runs from page 0: the control on the same memory).
fn benchDrafterServed(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, rng: std.Random, x: *const Ctx, o: Opts, pl: Pool, sv: *const Served) !void {
    _ = io;
    var dk = try ic.dspark.Kernels.load(d);
    defer dk.mod.unload();
    const P = o.context;
    const hq = o.hq;
    const hk = o.hk;
    const row = hk * D * 2;
    const b = x.b;
    const T = 8;
    const run_dk = o.runs - 2;
    const run_dv = o.runs - 1;
    // the drafter's own contiguous cache, and the same rows in the pool's last two runs
    const tk = try b.zeroed(gpa, d, P * row);
    const tv = try b.zeroed(gpa, d, P * row);
    try x.c.copy(tk, b.k, P * row);
    try x.c.copy(tv, b.v, P * row);
    for (0..pl.entries) |sp| {
        const n = @min(attn.split_keys, P - sp * attn.split_keys);
        const id: usize = @intCast(pl.table[sp]);
        try x.c.copy(pl.run(run_dk, id), b.k + sp * attn.split_keys * row, n * row);
        try x.c.copy(pl.run(run_dv, id), b.v + sp * attn.split_keys * row, n * row);
    }
    const ring_off: u64 = if (sv.run != null) @as(u64, sv.run_start) * pl.stride else 0;
    const nsplit_most = (P + T + attn.split_keys - 1) / attn.split_keys;
    const q = try b.random(gpa, d, rng, T * hq * D, 1.5);
    const bk = try b.random(gpa, d, rng, T * hk * D, 1.5);
    const bv = try b.random(gpa, d, rng, T * hk * D, 2.0);
    const ml = try b.zeroed(gpa, d, hq * nsplit_most * T * 2 * 4);
    const acc = try b.zeroed(gpa, d, hq * nsplit_most * T * D * 4);
    try x.c.stream.synchronize();
    const per = 10;
    var ev: [2 * per]cuda.Event = undefined;
    for (&ev) |*e| e.* = try cuda.Event.init(d, true);
    defer for (&ev) |*e| e.deinit();
    const forms = 3;
    const rounds = @max((o.reps + per - 1) / per, 3);
    if (rounds > 256) return error.TooManyReps;
    for (o.positions) |p| {
        if (p == 0 or p + T > P) continue;
        try x.setPos(p);
        const nsplit = (p + T + attn.split_keys - 1) / attn.split_keys;
        var t: [forms][256]f64 = undefined;
        for (0..rounds + 1) |r| for (0..forms) |j| {
            for (0..per) |c| {
                // the step's 8 global layers (verify window, captured form) on their own pages
                for (0..sv.n) |l| try x.runOn(.graph, sv.paged[l], p, T, x.b.outs[0]);
                try ev[2 * c].record(x.c.stream);
                var a: cuda.Args = .{};
                a.add(q);
                a.add(switch (j) {
                    0 => tk,
                    1 => pl.first(run_dk),
                    else => pl.first(run_dk) + ring_off,
                });
                a.add(switch (j) {
                    0 => tv,
                    1 => pl.first(run_dv),
                    else => pl.first(run_dv) + ring_off,
                });
                a.add(@as(i64, @intCast(hk * D)));
                if (j != 1) a.add(@as(i64, 0));
                a.add(@as(i32, @intCast(p)));
                a.add(bk);
                a.add(bv);
                a.add(@as(i64, @intCast(hk * D)));
                a.add(@as(i32, T));
                a.add(@as(i32, @intCast(hq)));
                a.add(@as(i32, @intCast(hq / hk)));
                a.add(@as(f32, 1.0 / @sqrt(@as(f32, D))));
                a.add(ml);
                a.add(acc);
                if (j == 1) {
                    a.add(b.pages);
                    a.add(@as(i64, @intCast(pl.elems())));
                }
                try cuda.launch.launch(if (j == 1) dk.attn_pg else dk.attn, .{ .grid = .{ .x = @intCast(nsplit), .y = @intCast(hq), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, x.c.stream, &a);
                try ev[2 * c + 1].record(x.c.stream);
            }
            try x.c.stream.synchronize();
            if (r == 0) continue; // warm-up round
            var sum: f64 = 0;
            for (0..per) |c| sum += @as(f64, try cuda.Event.elapsedMs(ev[2 * c], ev[2 * c + 1])) * 1e3;
            t[j][r - 1] = sum / per;
        };
        var m: [forms]f64 = undefined;
        for (0..forms) |j| {
            std.mem.sort(f64, t[j][0..rounds], {}, std.sort.asc(f64));
            m[j] = t[j][rounds / 2];
        }
        std.debug.print("bench drafter served context {d:>7} (after {d} layers' 8-row windows, us a call): block attention contiguous {d:>9.1} us, paged {d:>9.1} us ({d:>6.2}%), ring-on-pool{s} {d:>9.1} us (paged {d:>6.2}%)\n", .{ p, sv.n, m[0], m[1], pct(m[1], m[0]), if (sv.run != null) " (the stream's run)" else "", m[2], pct(m[1], m[2]) });
    }
}
