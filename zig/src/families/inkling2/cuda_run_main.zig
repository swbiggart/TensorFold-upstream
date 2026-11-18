//! tf-inkling2-cuda: Inkling-2-Small's layers [a, b) in `full` mode, one rank (W = 1) or W ranks (attention heads and
//! experts split, cuda_tp.zig: --world W --rank R, one process a GPU over NCCL; or --world W --virtual, every rank in
//! this process on one GPU, one after another), from an oracle
//! prompt file (tools/zig/inkling2_ref.py's <prompt>.safetensors): layer a's input rows and the block-residual
//! state at a come from the oracle, every layer's output rows and routes go to a safetensors dump that
//! tools/zig/inkling2_compare.py checks against the oracle. --dump-ops adds every op's inputs and outputs;
//! --feed takes torch's per-op outputs (tools/zig/inkling2_op_ref.py capture) so each op runs on torch's inputs.
//! At W > 1 rank 0 writes the dump (and runs the embed and head); every rank's settings fingerprint (executable,
//! checkpoint and its content-digest manifest, settings, the prompt file's bytes) must match before loading, each
//! rank's loaded bytes must hash as the manifest's lines for that rank (digests.zig), and every rank's layer outputs
//! and routes must hash the same at the end. Real ranks keep liveness lines (cuda/tp_watch.zig): a rank that fails
//! aborts its communicator and tells the others, and a rank that dies closes its line, so the others stop with an
//! error instead of waiting in a collective. A dump whose status is not "complete" is not a result.
//! Exit status: 0 done, 3 stopped at an op without a kernel (the dump holds what ran), 1 failed, 2 usage.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");

const fwd = ic.forward;
const Mat = ic.ops.Mat;
const H = ic.weights.hidden;

const usage =
    \\usage: tf-inkling2-cuda <model dir> <oracle prompt.safetensors> <out.safetensors> [options]
    \\  --layers A:B      run layers A .. B-1 (default 0:1); about 12 NVFP4 layers fit a 96 GB card
    \\  --embed           layer 0's input from the prompt's ids (head.embed), not the oracle's embed rows
    \\  --head            after layer 47: the final mix, norm, unembed and softcap at the oracle's logits.rows
    \\  --teacher         every layer's input from the oracle (h.<L-1> and the blocks below L), not the layer
    \\                    before's output: each layer is checked alone; with --head the final mix and norm run on
    \\                    the oracle's h.47 and blocks, and the unembed on the oracle's final rows
    \\  --chunk N         forwards of at most N rows (default: the whole prompt in one)
    \\  --dump-ops        every op's inputs and outputs: op.<layer|model>.<group>.<op>.<in|out>.<tensor>
    \\  --feed FILE       torch's op outputs (inkling2_op_ref.py capture): every op runs on torch's inputs and an
    \\                    op without a kernel passes with torch's outputs
    \\  --scratch-gib G   device scratch an op may take (default 2)
    \\  --world W         ranks (1, 2, 4 ...; default 1): heads and experts split as plan.zig cuts them
    \\  --rank R          this process's rank (W > 1, real ranks: NCCL_COMM_ID=<rank 0 host>:<port> on every rank)
    \\  --virtual         every rank of --world in this process on one GPU, in turn (no NCCL)
    \\  --mode M          the experts' arithmetic: full (default, or TF_INK2_MODE) or checkpoint (activations
    \\                    quantized as the checkpoint requires, TokenSpeed's numerics; cuda_moe.zig)
    \\  --experts L       whole (default, or TF_INK2_EXPERTS): whole experts a rank; split: a 1/W row slice of every
    \\                    expert on every rank (plan.zig), checkpoint mode only, bit-equal to W = 1
    \\  --prof          the stream time of every op (CUDA events between ops), a table per forward: ms a rank
    \\                  for each layer and op (virtual ranks: the mean over the ranks)
    \\  --synthetic     with --embed and A > 0: layer A's input and every block below it are the prompt's normed
    \\                  embedding (timing runs over any layers and any prompt; the values are not the model's)
    \\  --digests FILE    the content-digest manifest (tf-inkling2-plan --digests, from the source checkpoint) the
    \\                    loaded bytes must match; default <model dir>/inkling2-digests-w<W>.tsv (-split.tsv for
    \\                    --experts split) when there, which
    \\                    real ranks (W > 1) require; `none` loads unchecked
    \\The GPU is the first visible device (set CUDA_VISIBLE_DEVICES).
    \\
;

const Opts = struct {
    dir: []const u8,
    prompt: []const u8,
    out: []const u8,
    lo: u32 = 0,
    hi: u32 = 1,
    embed: bool = false,
    head: bool = false,
    teacher: bool = false,
    chunk: usize = 0,
    dump_ops: bool = false,
    feed: ?[]const u8 = null,
    scratch_gib: usize = 2,
    world: u32 = 1,
    rank: u32 = 0,
    virtual: bool = false,
    mode: ic.moe.Mode = .full,
    experts: ink.plan.Layout = .whole,
    digests: ?[]const u8 = null,
    prof: bool = false,
    synthetic: bool = false,
};

fn parse(a: std.mem.Allocator, args: []const [:0]const u8) !?Opts {
    if (args.len < 4) return null;
    var o: Opts = .{ .dir = args[1], .prompt = args[2], .out = args[3], .mode = ic.moe.envMode(), .experts = ink.plan.Layout.env() };
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--embed")) {
            o.embed = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--head")) {
            o.head = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--teacher")) {
            o.teacher = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--dump-ops")) {
            o.dump_ops = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--virtual")) {
            o.virtual = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--prof")) {
            o.prof = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--synthetic")) {
            o.synthetic = true;
            continue;
        }
        if (i + 1 >= args.len) return null;
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--layers")) {
            const colon = std.mem.indexOfScalar(u8, v, ':') orelse return null;
            o.lo = try std.fmt.parseInt(u32, v[0..colon], 10);
            o.hi = try std.fmt.parseInt(u32, v[colon + 1 ..], 10);
        } else if (std.mem.eql(u8, k, "--chunk")) {
            o.chunk = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--feed")) {
            o.feed = try a.dupe(u8, v);
        } else if (std.mem.eql(u8, k, "--scratch-gib")) {
            o.scratch_gib = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--world")) {
            o.world = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--rank")) {
            o.rank = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--mode")) {
            o.mode = std.meta.stringToEnum(ic.moe.Mode, v) orelse return null;
        } else if (std.mem.eql(u8, k, "--experts")) {
            o.experts = std.meta.stringToEnum(ink.plan.Layout, v) orelse return null;
        } else if (std.mem.eql(u8, k, "--digests")) {
            o.digests = try a.dupe(u8, v);
        } else return null;
    }
    return o;
}

/// Whole-prompt host copies of what the forward produced, written as one safetensors file at the end.
const Dump = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    rows: usize,
    tensors: std.StringArrayHashMapUnmanaged(Item) = .empty,
    feed: ?*ink.st.Reader = null,
    stubs: std.ArrayList([]u8) = .empty,
    fed: usize = 0,
    /// Only the head's selected rows: hooks number them from 0.
    head_rows: usize = 0,

    const Item = struct { dtype: ink.st.DType, rows: usize, cols: usize, data: []u8 };

    fn deinit(x: *Dump) void {
        var it = x.tensors.iterator();
        while (it.next()) |e| {
            x.gpa.free(e.key_ptr.*);
            x.gpa.free(e.value_ptr.data);
        }
        x.tensors.deinit(x.gpa);
        for (x.stubs.items) |s| x.gpa.free(s);
        x.stubs.deinit(x.gpa);
    }

    fn stDType(dt: ic.ops.Dt) ink.st.DType {
        return switch (dt) {
            .bf16 => .bf16,
            .f32 => .f32,
            .i32 => .i32,
            .u8 => .u8,
        };
    }

    /// Rows row0 .. row0 + m.rows of a [total, cols] tensor (created at first use).
    fn put(x: *Dump, name: []const u8, m: Mat, row0: usize, total: usize) !void {
        const es = m.dt.size();
        const it = x.tensors.getPtr(name) orelse blk: {
            // allocate before inserting, so a failed allocation leaves no half-built entry for deinit to free
            const key = try x.gpa.dupe(u8, name);
            errdefer x.gpa.free(key);
            const data = try x.gpa.alloc(u8, total * m.cols * es);
            errdefer x.gpa.free(data);
            @memset(data, 0);
            const gop = try x.tensors.getOrPut(x.gpa, key);
            gop.value_ptr.* = .{ .dtype = stDType(m.dt), .rows = total, .cols = m.cols, .data = data };
            break :blk gop.value_ptr;
        };
        if (it.cols != m.cols or row0 + m.rows > it.rows or it.dtype != stDType(m.dt)) return error.DumpShape;
        const dst = it.data[row0 * m.cols * es ..][0 .. m.rows * m.cols * es];
        if (m.dense()) return cuda.memory.download(x.d, m.ptr, dst);
        for (0..m.rows) |r| try cuda.memory.download(x.d, m.row(r), dst[r * m.cols * es ..][0 .. m.cols * es]);
    }

    fn opName(buf: []u8, scope: fwd.Scope, op: []const u8, dir: []const u8, t: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "op.{f}.{s}.{s}.{s}", .{ scope, op, dir, t });
    }

    fn record(ctx: *anyopaque, scope: fwd.Scope, op: []const u8, dir: fwd.Dir, t: fwd.Tensor, row0: usize) anyerror!void {
        const x: *Dump = @ptrCast(@alignCast(ctx));
        var buf: [160]u8 = undefined;
        var name = try opName(&buf, scope, op, @tagName(dir), t.name);
        if (t.whole) { // per forward: suffixed with its first row
            var b2: [176]u8 = undefined;
            name = try std.fmt.bufPrint(&b2, "{s}@{d}", .{ name, row0 });
            return x.put(name, t.m, 0, t.m.rows);
        }
        try x.put(name, t.m, row0, if (scope == .model and std.mem.startsWith(u8, op, "head.") and !std.mem.eql(u8, op, "head.embed") and !std.mem.eql(u8, op, "head.final_norm")) x.head_rows else x.rows);
    }

    fn feedFn(ctx: *anyopaque, scope: fwd.Scope, op: []const u8, t: fwd.Tensor, row0: usize) anyerror!bool {
        const x: *Dump = @ptrCast(@alignCast(ctx));
        const r = x.feed orelse return false;
        var buf: [160]u8 = undefined;
        const name = try opName(&buf, scope, op, "out", t.name);
        const e = r.find(name) orelse return false;
        if (e.dtype != stDType(t.m.dt) or e.rank != 2 or e.dim(1) != t.m.cols or row0 + t.m.rows > e.dim(0) or !t.m.dense()) {
            std.log.err("inkling2: capture's {s} is {s} {any}, the op's [{d}+{d}, {d}] {t}", .{ name, e.dtype.name(), e.shape[0..e.rank], row0, t.m.rows, t.m.cols, t.m.dt });
            return error.FeedShape;
        }
        const n = t.m.rows * t.m.cols * t.m.dt.size();
        const host = try x.gpa.alloc(u8, n);
        defer x.gpa.free(host);
        try r.readInto(e, row0 * t.m.cols * t.m.dt.size(), host);
        try x.stream.synchronize();
        const view: cuda.DeviceBuffer = .{ .d = x.d, .ptr = t.m.ptr, .len = n, .borrowed = true };
        try view.upload(0, host);
        x.fed += 1;
        return true;
    }

    fn stubFn(ctx: *anyopaque, scope: fwd.Scope, op: []const u8) anyerror!void {
        const x: *Dump = @ptrCast(@alignCast(ctx));
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{f}.{s}", .{ scope, op });
        for (x.stubs.items) |s| if (std.mem.eql(u8, s, name)) return;
        try x.stubs.append(x.gpa, try x.gpa.dupe(u8, name));
    }

    fn write(x: *Dump, io: std.Io, path: []const u8, meta: []const [2][]const u8) !void {
        var w = ink.st.Writer.init(x.gpa);
        defer w.deinit();
        var it = x.tensors.iterator();
        while (it.next()) |e| {
            const v = e.value_ptr.*;
            try w.addOwned(e.key_ptr.*, v.dtype, &.{ v.rows, v.cols }, v.data);
            e.value_ptr.data = &.{};
        }
        for (meta) |m| try w.setMeta(m[0], m[1]);
        try w.write(io, path);
    }
};

fn readI64(gpa: std.mem.Allocator, r: *const ink.st.Reader, name: []const u8) ![]i64 {
    const raw = try r.alloc(gpa, name, .i64);
    defer gpa.free(raw);
    const out = try gpa.alloc(i64, raw.len / 8);
    for (out, 0..) |*v, i| v.* = std.mem.readInt(i64, raw[8 * i ..][0..8], .little);
    return out;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const o = (parse(a, args) catch null) orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    return run(a, gpa, io, o) catch |e| {
        std.debug.print("FAIL tf-inkling2-cuda: {t}\n", .{e});
        cuda.tp_watch.fail(@errorName(e)); // real ranks: the others stop too (a no-op without lines)
        return 1;
    };
}

fn run(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    const clock = std.Io.Clock.awake;
    // the prompt
    var oracle = try ink.st.Reader.open(gpa, io, o.prompt);
    defer oracle.close();
    const ids = try readI64(gpa, &oracle, "ids");
    defer gpa.free(ids);
    const kept = try readI64(gpa, &oracle, "rows");
    defer gpa.free(kept);
    const T = ids.len;
    const every = kept.len == T;
    if (o.synthetic and (!o.embed or o.teacher or o.lo == 0)) {
        std.debug.print("--synthetic needs --embed, --layers A:B with A > 0, and no --teacher\n", .{});
        return 2;
    }
    if (!every and (o.teacher or !((o.lo == 0 or o.synthetic) and o.embed))) {
        std.debug.print("the oracle keeps {d} of the prompt's {d} rows: only --layers 0:N with --embed can run it, and --teacher cannot (inkling2_ref.py --all-rows keeps every row)\n", .{ kept.len, T });
        return 2;
    }
    if (o.world == 0 or o.rank >= o.world or (o.world == 1 and o.virtual)) {
        std.debug.print("--world {d} --rank {d}{s}: give a rank below the world size (--virtual needs --world 2 or more)\n", .{ o.world, o.rank, if (o.virtual) " --virtual" else "" });
        return 2;
    }
    if (o.world > 1 and (o.feed != null or o.dump_ops)) {
        std.debug.print("--feed and --dump-ops run at --world 1 only\n", .{});
        return 2;
    }
    if (o.experts == .split and o.mode != .checkpoint) {
        std.debug.print("--experts split runs in checkpoint mode only (--mode checkpoint or TF_INK2_MODE=checkpoint)\n", .{});
        return 2;
    }
    var d: ink.Diag = .{};
    var model = ink.describe.Model.fromDir(gpa, io, o.dir, .{ .world = o.world, .layout = o.experts }, &d) catch |e| {
        std.debug.print("FAIL tf-inkling2-cuda: {t}: {s}\n", .{ e, d.text() });
        return 1;
    };
    defer model.deinit();
    const layers = model.c.text.layers;
    if (o.lo >= o.hi or o.hi > layers) {
        std.debug.print("--layers {d}:{d}: give A < B <= {d}\n", .{ o.lo, o.hi, layers });
        return 2;
    }
    if (o.head and o.hi != layers) {
        std.debug.print("--head needs the last layer (--layers A:{d})\n", .{layers});
        return 2;
    }
    const chunk = if (o.chunk == 0) T else @min(o.chunk, T);
    const real = o.world > 1 and !o.virtual; // one process a rank, over NCCL
    // the content-digest manifest
    const manifest_path: ?[]const u8 = blk: {
        if (o.digests) |p| break :blk if (std.mem.eql(u8, p, "none")) null else p;
        const sx = if (o.experts == .split) "-split" else "";
        const p = try std.fmt.allocPrint(a, "{s}/inkling2-digests-w{d}{s}.tsv", .{ o.dir, o.world, sx });
        std.Io.Dir.cwd().access(io, p, .{}) catch {
            if (!real) break :blk null;
            std.debug.print("no content-digest manifest at {s}: give --digests FILE (tf-inkling2-plan --dir <source> --world {d}{s} --digests FILE) or --digests none\n", .{ p, o.world, if (o.experts == .split) " --experts split" else "" });
            return 2;
        };
        break :blk p;
    };
    const manifest: ?[]const u8 = if (manifest_path) |p| try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 28)) else null;
    if (manifest == null and real) std.debug.print("rank {d}: weight contents not checked (--digests none)\n", .{o.rank});
    // the ranks this process runs: rank o.rank, or every rank (virtual)
    const first_rank: u32 = if (o.virtual) 0 else o.rank;
    const nlocal: u32 = if (o.virtual) o.world else 1;
    const leader = first_rank == 0; // rank 0: embeds, runs the head, writes the dump
    // the device
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var group: ?cuda.tp_group.Group = null;
    defer if (group) |*g| g.close();
    if (real) {
        // liveness lines first (NCCL_COMM_ID's port + 1): from here a rank that ends early stops the others
        cuda.tp_watch.start(io, o.rank, o.world) catch |e| {
            std.debug.print("FAIL tf-inkling2-cuda: rank {d} of {d}: liveness lines: {t}{s}{s}\n", .{ o.rank, o.world, e, if (cuda.tp_watch.explain(e) != null) ": " else "", cuda.tp_watch.explain(e) orelse "" });
            return 1;
        };
        group = cuda.tp_group.Group.open(&driver, o.rank, o.world) catch |e| {
            std.debug.print("FAIL tf-inkling2-cuda: rank {d} of {d}: NCCL group: {t} (set NCCL_COMM_ID=<rank 0 host>:<port> on every rank)\n", .{ o.rank, o.world, e });
            cuda.tp_watch.fail(@errorName(e));
            return 1;
        };
    }
    // an error from here aborts the communicator (no kernel waits on a peer), and main tells the others by name
    errdefer if (group) |*g| g.abort();
    if (real) try agree(gpa, io, &driver, stream, &group.?, &model, o, ids, manifest);
    if (real) try ic.wire.openAll(gpa, io, &driver, &group.?); // the transport settings' vote, then the wires
    // a dump left by an earlier run under this name is not this run's result
    if (leader) std.Io.Dir.cwd().deleteFile(io, o.out) catch {};
    var k = try ic.kernels.Kernels.load(&driver);
    defer k.unload();
    k.moe.mode = o.mode;
    var gemms = try ic.ops.Gemms.init(gpa, &driver);
    defer gemms.deinit();
    var scratch = try ic.ops.Scratch.init(&driver, o.scratch_gib << 30);
    defer scratch.deinit();
    var feed: ?ink.st.Reader = if (o.feed) |p| try ink.st.Reader.open(gpa, io, p) else null;
    defer if (feed) |*f| f.close();
    var parts: ?ic.tp.Parts = if (o.world > 1) try ic.tp.Parts.initFor(&driver, o.world, chunk, o.experts) else null;
    defer if (parts) |*p| p.deinit();
    // the weights, state and activations of each rank this process runs
    const ranks = try gpa.alloc(Rank, nlocal);
    defer gpa.free(ranks);
    var made: usize = 0;
    defer for (ranks[0..made]) |*r| r.deinit();
    var t0 = clock.now(io).toNanoseconds();
    for (ranks, 0..) |*r, i| {
        const rank: u32 = first_rank + @as(u32, @intCast(i));
        const t1 = clock.now(io).toNanoseconds();
        var got: std.ArrayList(ink.digests.Digest) = .empty;
        defer got.deinit(gpa);
        r.w = try ic.weights.load(gpa, io, &driver, o.dir, &model, .{ .rank = rank, .lo = o.lo, .hi = o.hi, .embed = o.embed and rank == 0, .head = o.head, .digests = if (manifest != null) &got else null }, &fwd.prepare);
        errdefer r.w.deinit();
        const ls = secs(clock.now(io).toNanoseconds() - t1);
        const l0 = r.w.layers[0];
        std.debug.print("rank {d} of {d}: loaded layers {d}..{d}{s}{s}: q heads {d}, kv heads {d}, experts {d}..{d}{s} ({s} mode): {d:.2} GiB in {d:.1} s ({d:.2} GiB/s)\n", .{ rank, o.world, o.lo, o.hi - 1, if (r.w.embed != 0) " +embed" else "", if (r.w.unembed != 0) " +head" else "", l0.hq, l0.hk, l0.experts.lo, l0.experts.hi - 1, if (l0.experts.split > 1) try std.fmt.allocPrint(a, ", split: w13 rows {d}.. and w2 rows {d}.. of each", .{ l0.experts.part * l0.experts.n13, l0.experts.part * l0.experts.n2 }) else "", @tagName(o.mode), gib(r.w.bytes), ls, gib(r.w.bytes) / @max(ls, 1e-9) });
        if (manifest) |text| {
            const c = try ink.digests.check(gpa, text, &model, rank, got.items);
            if (c.differ + c.missing > 0) {
                std.debug.print("FAIL rank {d}: {d} of {d} loaded pieces differ from the digest manifest and {d} have no matching line (first: {s})\n", .{ rank, c.differ, c.pieces, c.missing, c.first.? });
                return error.WeightBytesDiffer;
            }
            std.debug.print("rank {d}: weight contents match the digest manifest ({d} pieces, loaded xxh64 {x:0>16})\n", .{ rank, c.pieces, c.loaded });
        }
        r.act = try ic.state.Act.init(gpa, &driver, chunk, l0.hq, l0.hk);
        errdefer r.act.deinit();
        r.st = try ic.state.State.init(gpa, &driver, &r.w, T, chunk, false);
        r.x = .{ .rank = rank, .world = o.world, .parts = if (parts) |*p| p else undefined, .group = if (group) |*g| g else null };
        made += 1;
    }
    const load_s = secs(clock.now(io).toNanoseconds() - t0);
    for (ranks[0].w.layers) |l| if (l.experts.format == .nvfp4) std.debug.print("  layer {d}: {s}, NVFP4 experts, outer_scale_denominator w13 {d} w2 {d}\n", .{ l.index, if (l.global) "global" else "sliding", l.experts.den13, l.experts.den2 }) else std.debug.print("  layer {d}: {s}, MXFP8 experts\n", .{ l.index, if (l.global) "global" else "sliding" });
    var dump: Dump = .{ .gpa = gpa, .d = &driver, .stream = stream, .rows = T, .feed = if (feed) |*f| f else null };
    defer dump.deinit();
    for (ranks, 0..) |*r, i| r.f = .{
        .c = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch },
        .w = &r.w,
        .st = &r.st,
        .a = &r.act,
        .x = if (o.world > 1) &r.x else null,
        .hooks = if (i == 0) .{
            .ctx = &dump,
            .record = if (o.dump_ops) Dump.record else null,
            .feed = if (feed != null) Dump.feedFn else null,
            .stub = Dump.stubFn,
        } else .{},
    };
    var prof: ?ic.forward.Prof = if (o.prof) .{ .d = &driver, .stream = stream } else null;
    defer if (prof) |*p| p.deinit();
    if (prof) |*p| for (ranks) |*r| {
        r.f.prof = p;
    };
    const f = &ranks[0].f;
    // the head's rows: the oracle's logits.rows. The head is split by vocabulary (every rank unembeds its share, the
    // shares gathered in rank order, rank 0 softcaps them), so every rank runs it.
    const run_head = o.head;
    const head_rows: []i64 = if (run_head) try readI64(gpa, &oracle, "logits.rows") else &.{};
    defer if (run_head) gpa.free(head_rows);
    dump.head_rows = head_rows.len;
    var head_bufs: ?HeadBufs = if (run_head) try HeadBufs.init(&driver, head_rows.len, o.world, ranks[0].w.vocab_rows.len(), ranks[0].w.unpadded_vocab) else null;
    defer if (head_bufs) |*hb| hb.deinit();
    {
        var held: u64 = 0;
        for (ranks) |*r| held += r.w.bytes + r.act.bytes + r.st.bytes;
        const mem = try ctx.memInfo();
        std.debug.print("prompt {s}: {d} rows, forwards of {d}; weights {d:.2} GiB, activations {d:.2} GiB, state {d:.2} GiB, partials {d:.2} GiB, op scratch {d} GiB; device {d:.2} of {d:.2} GiB free after load ({d:.2} GiB held by this process in all)\n", .{ std.fs.path.basename(o.prompt), T, chunk, gib(ranks[0].w.bytes), gib(ranks[0].act.bytes), gib(ranks[0].st.bytes), if (parts) |*p| gib(p.bytes()) else 0, o.scratch_gib, gib(mem.free), gib(mem.total), gib(held + (if (parts) |*p| p.bytes() else 0) + (o.scratch_gib << 30)) });
    }
    // from here an error aborts the communicator before anything frees or waits on the stream (a gather may be in
    // flight); main then tells the other ranks
    errdefer if (group) |*g| g.abort();
    const fault = Fault.read();
    t0 = clock.now(io).toNanoseconds();
    var status: []const u8 = "complete";
    var code: u8 = 0;
    var done_layers: u32 = o.hi; // the first layer that did not finish
    var rows_done: usize = 0; // rows of the forwards that finished
    var layer_ns: u64 = 0; // the layers' own time: each layer to its stream's end (no dumps, no oracle reads)
    var embed_rows: usize = 0;
    var start: usize = 0;
    outer: while (start < T) : (start += chunk) {
        const R = @min(chunk, T - start);
        for (ranks) |*r| inputs(gpa, &r.f, &oracle, ids, o, start, R) catch |e| {
            status = try stopped(a, r.f.stopped, e);
            code = if (e == error.NotImplemented) 3 else 1;
            break :outer;
        };
        if (o.embed and (o.lo == 0 or o.synthetic) and o.world > 1) try shareEmbed(ranks, o, R);
        if (o.synthetic) for (ranks) |*r| {
            const nb = fwd.blocksBefore(o.lo);
            for (0..nb) |i| try r.f.c.copyRows(r.f.a.block(i, R), r.f.a.h[0].slice(0, R));
            r.f.a.nblocks = nb;
        };
        if (o.embed and leader) {
            try stream.synchronize();
            try dump.put("embed", f.a.h[0].slice(0, R), start, T);
            embed_rows = start + R;
        }
        var li = o.lo;
        while (li < o.hi) : (li += 1) {
            if (o.teacher and li > o.lo) for (ranks) |*r| try oracleState(gpa, &r.f, &oracle, li, start, R);
            const lt = ic.tp.Timing.now();
            (if (fault.at(.fail, first_rank, li)) error.InducedFault else runLayer(ranks, o.virtual, li, R, start)) catch |e| {
                done_layers = li;
                status = try stopped(a, f.stopped, e);
                code = if (e == error.NotImplemented) 3 else 1;
                break :outer;
            };
            if (prof) |*p| try p.end();
            try stream.synchronize();
            layer_ns += ic.tp.Timing.now() - lt;
            if (group) |*g| try g.check();
            var nb: [32]u8 = undefined;
            const L = li;
            const act = f.a;
            try dump.put(try std.fmt.bufPrint(&nb, "h.{d}", .{L}), act.h[0].slice(0, R), start, T);
            try dump.put(try std.fmt.bufPrint(&nb, "route.ids.{d}", .{L}), act.ids.slice(0, R), start, T);
            try dump.put(try std.fmt.bufPrint(&nb, "route.w.{d}", .{L}), act.wts.slice(0, R), start, T);
            try dump.put(try std.fmt.bufPrint(&nb, "route.logits.{d}", .{L}), act.logits.slice(0, R), start, T);
        }
        if (run_head) {
            if (o.teacher) try oracleState(gpa, f, &oracle, layers, start, R);
            const y = f.final(R, start) catch |e| {
                status = try stopped(a, f.stopped, e);
                code = if (e == error.NotImplemented) 3 else 1;
                break :outer;
            };
            try stream.synchronize();
            try dump.put("final", y, start, T);
            // the oracle's logit rows inside this forward, gathered for the unembed
            const hb = &head_bufs.?;
            var n: usize = 0;
            var first: ?usize = null;
            for (head_rows, 0..) |r, j| if (r >= start and r < start + R) {
                if (first == null) first = j;
                const at: usize = @intCast(r);
                if (o.teacher) { // the oracle's final row (every row kept: its index is its position)
                    try rows(gpa, f, &oracle, "final", at, 1, hb.rows.slice(n, 1));
                } else try f.c.copyRows(hb.rows.slice(n, 1), y.slice(at - start, 1));
                n += 1;
            };
            if (n > 0) {
                const j0 = first.?;
                headShares(ranks, o, hb.rows.slice(0, n), hb.parts, hb.out.slice(0, n), j0) catch |e| {
                    status = try stopped(a, f.stopped, e);
                    code = if (e == error.NotImplemented) 3 else 1;
                    break :outer;
                };
                try stream.synchronize();
                if (leader) try dump.put("logits.full", hb.out.slice(0, n), j0, head_rows.len);
            }
        }
        rows_done = start + R;
        if (prof) |*p| {
            printProf(p, o.lo, o.hi, nlocal, start, R);
            p.reset();
        }
    }
    // a rank that stopped aborts its communicator and tells the others (they may wait in a gather for it)
    if (code != 0) if (group) |*g| bail(g, status);
    try stream.synchronize();
    const run_s = secs(clock.now(io).toNanoseconds() - t0);
    if (o.head and code == 0) {
        const rows_bytes = try gpa.alloc(u8, head_rows.len * 8);
        defer gpa.free(rows_bytes);
        for (head_rows, 0..) |r, i| std.mem.writeInt(i64, rows_bytes[8 * i ..][0..8], r, .little);
        try dump.tensors.put(gpa, try gpa.dupe(u8, "logits.rows"), .{ .dtype = .i64, .rows = head_rows.len, .cols = 1, .data = try gpa.dupe(u8, rows_bytes) });
    }
    // what ran, for the compare tool
    var stubs: std.ArrayList(u8) = .empty;
    for (dump.stubs.items, 0..) |s, i| {
        if (i > 0) try stubs.append(a, ',');
        try stubs.appendSlice(a, s);
    }
    if (group) |*g| if (code == 0) {
        // every rank's layer outputs and routes hash the same (the sums gave every rank the same bits)
        var mine = digest(&dump);
        if (fault.at(.digest, first_rank, null)) mine[0] ^= 1;
        var all: [ink.plan.max_world][32]u8 = undefined;
        try g.gather([32]u8, &driver, stream, mine, all[0..o.world]);
        // no rank waits on another after this barrier, so a line closing from here on is a rank ending as planned
        cuda.tp_watch.ended();
        var marks: [ink.plan.max_world]u32 = undefined;
        try g.gather(u32, &driver, stream, o.rank, marks[0..o.world]);
        var same = true;
        for (all[0..o.world]) |h| same = same and std.mem.eql(u8, &h, &mine);
        std.debug.print("rank {d}: layer outputs and routes {s} on all {d} ranks (sha256 {x})\n", .{ o.rank, if (same) "identical" else "DIFFER", o.world, mine[0..8] });
        if (!same) {
            code = 1;
            status = "failed: the ranks' layer outputs or routes differ";
        }
    };
    const meta = [_][2][]const u8{
        .{ "tool", "tf-inkling2-cuda" },
        .{ "prompt", std.fs.path.basename(o.prompt) },
        .{ "layers", try std.fmt.allocPrint(a, "{d}:{d}", .{ o.lo, o.hi }) },
        .{ "layers_done", try std.fmt.allocPrint(a, "{d}:{d}", .{ o.lo, done_layers }) },
        .{ "rows", try std.fmt.allocPrint(a, "{d}", .{T}) },
        .{ "rows_done", try std.fmt.allocPrint(a, "{d}", .{rows_done}) },
        .{ "embed_rows", try std.fmt.allocPrint(a, "{d}", .{embed_rows}) },
        .{ "chunk", try std.fmt.allocPrint(a, "{d}", .{chunk}) },
        .{ "mode", @tagName(o.mode) },
        .{ "experts", @tagName(o.experts) },
        .{ "world", try std.fmt.allocPrint(a, "{d}{s}", .{ o.world, if (o.virtual) " (virtual)" else "" }) },
        .{ "embed", if (o.embed) "computed" else "oracle" },
        .{ "inputs", if (o.teacher) "oracle at every layer (--teacher)" else "oracle at the first layer, then chained" },
        .{ "status", status },
        .{ "fed_stubs", stubs.items },
        .{ "feed", if (o.feed) |p| std.fs.path.basename(p) else "" },
    };
    if (leader) try dump.write(io, o.out, &meta);
    std.debug.print("rank {d}: {s} in {d:.2} s (load {d:.1} s); {d} tensors{s}{s}; op scratch peak {d} MiB; {d} op outputs fed from the capture\n", .{ first_rank, status, run_s, load_s, dump.tensors.count(), if (leader) " to " else " (not written: rank 0 writes)", if (leader) o.out else "", scratch.peak >> 20, dump.fed });
    {
        const nl: f64 = @floatFromInt(@max(1, done_layers - o.lo));
        const ms = @as(f64, @floatFromInt(layer_ns)) / 1e6;
        std.debug.print("rank {d}: layers {d:.1} ms ({d:.2} ms a layer over {d} rows)", .{ first_rank, ms, ms / nl, rows_done });
        if (ic.tp.Timing.enabled() and o.world > 1 and !o.virtual) {
            const X = ic.tp.Timing;
            std.debug.print("; exchanges attn {d:.1} ms in {d}, moe {d:.1} ms in {d} ({d:.3} ms a gather)", .{ @as(f64, @floatFromInt(X.ns[0])) / 1e6, X.calls[0], @as(f64, @floatFromInt(X.ns[1])) / 1e6, X.calls[1], @as(f64, @floatFromInt(X.ns[0] + X.ns[1])) / 1e6 / @as(f64, @floatFromInt(@max(1, X.calls[0] + X.calls[1]))) });
        }
        std.debug.print("\n", .{});
    }
    if (dump.stubs.items.len > 0) {
        // each op once, with the layers it passed in
        var seen: std.ArrayList([]const u8) = .empty;
        std.debug.print("ops without kernels, passed with torch's outputs:", .{});
        for (dump.stubs.items) |x| {
            const op = x[std.mem.indexOfScalar(u8, x, '.').? + 1 ..];
            var dup = false;
            for (seen.items) |y| dup = dup or std.mem.eql(u8, y, op);
            if (dup) continue;
            try seen.append(a, op);
            var n: usize = 0;
            for (dump.stubs.items) |z| n += @intFromBool(std.mem.eql(u8, z[std.mem.indexOfScalar(u8, z, '.').? + 1 ..], op));
            std.debug.print(" {s} (x{d})", .{ op, n });
        }
        std.debug.print("\n", .{});
    }
    return code;
}

/// One rank's weights, state and activations, its exchange and forward (addresses fixed: the forward points in).
const Rank = struct {
    w: ic.weights.Weights,
    act: ic.state.Act,
    st: ic.state.State,
    x: ic.tp.Exchange,
    f: fwd.Forward = undefined,

    fn deinit(r: *Rank) void {
        r.st.deinit();
        r.act.deinit();
        r.w.deinit();
    }
};

/// Layer li on every rank this process runs: real ranks gather inside the layer; virtual ranks run each stage on
/// every rank before the next stage, so the sums (and split experts' gathers) read every rank's part.
fn runLayer(ranks: []Rank, virtual: bool, li: u32, R: usize, pos0: usize) !void {
    if (!virtual) return ranks[0].f.layer(li, R, pos0);
    for (ranks) |*r| try r.f.attnStage(li, R, pos0);
    for (ranks) |*r| try r.f.mlpStage(li, R, pos0);
    if (ranks[0].f.split()) for (ranks) |*r| try r.f.downStage(li, R, pos0);
    for (ranks) |*r| try r.f.outStage(li, R, pos0);
}

/// W > 1 with --embed: rank 0's embedding rows (in its h[0]) to every rank's h[0].
fn shareEmbed(ranks: []Rank, o: Opts, R: usize) !void {
    const f0 = &ranks[0].f;
    const c = &f0.c;
    const h = f0.a.h[0].slice(0, R);
    const n = R * H * 2;
    if (o.virtual) {
        for (ranks[1..]) |*r| try c.copyRows(r.f.a.h[0].slice(0, R), h);
        return;
    }
    const x = &ranks[0].x;
    const buf = x.parts.ptr[0];
    try c.copy(buf + @as(u64, x.rank) * n, h.ptr, n);
    try x.group.?.allGather(c.stream, buf + @as(u64, x.rank) * n, buf, n);
    try c.copy(h.ptr, buf, n);
}

/// What every rank must agree on before loading: the executable, the checkpoint (config, quantization manifest,
/// every tensor's name, file, offset and size), and the run's settings and prompt.
const Fingerprint = extern struct {
    program: [32]u8,
    checkpoint: [32]u8,
    settings: [32]u8,
    rank: u32,
    world: u32,
};

fn hashFile(gpa: std.mem.Allocator, io: std.Io, h: *std.crypto.hash.sha2.Sha256, path: []const u8) !void {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => return h.update("(none)"),
        else => return e,
    };
    defer gpa.free(bytes);
    h.update(bytes);
}

fn agree(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, g: *const cuda.tp_group.Group, m: *const ink.describe.Model, o: Opts, ids: []const i64, manifest: ?[]const u8) !void {
    const Sha = std.crypto.hash.sha2.Sha256;
    var fp: Fingerprint = .{ .program = undefined, .checkpoint = undefined, .settings = undefined, .rank = o.rank, .world = o.world };
    var prompt_xxh64: u64 = 0;
    {
        var h = Sha.init(.{});
        try hashFile(gpa, io, &h, "/proc/self/exe");
        h.final(&fp.program);
    }
    {
        var h = Sha.init(.{});
        for ([_][]const u8{ "config.json", "hf_quant_config.json", "model.safetensors.index.json" }) |name| {
            const path = try std.fs.path.join(gpa, &.{ o.dir, name });
            defer gpa.free(path);
            try hashFile(gpa, io, &h, path);
        }
        for (m.inv.tensors) |t| {
            h.update(t.name);
            h.update(std.mem.asBytes(&[3]u64{ t.file, m.inv.offset(t), t.bytes() }));
        }
        // the expected contents: every rank's piece digests (each rank checks its own after loading)
        h.update(manifest orelse "(no digest manifest)");
        h.final(&fp.checkpoint);
    }
    {
        var h = Sha.init(.{});
        var buf: [512]u8 = undefined;
        const env = struct {
            fn get(name: [*:0]const u8) []const u8 {
                return if (std.c.getenv(name)) |v| std.mem.span(v) else "";
            }
        }.get;
        h.update(try std.fmt.bufPrint(&buf, "world {d} layers {d}:{d} embed {} head {} teacher {} chunk {d} scratch {d} mode {s} experts {s} TF_INK2_ATTN={s} XROWS={s} CK_SHAPE1={s} prompt {s}", .{ o.world, o.lo, o.hi, o.embed, o.head, o.teacher, o.chunk, o.scratch_gib, @tagName(o.mode), @tagName(o.experts), env("TF_INK2_ATTN"), env("TF_INK2_XROWS"), env("TF_INK2_CK_SHAPE1"), std.fs.path.basename(o.prompt) }));
        h.update(std.mem.sliceAsBytes(ids));
        // the prompt file's bytes: the oracle rows each rank reads (h.*, embed, final), not only its name and ids
        prompt_xxh64 = try fileXxh64(gpa, io, o.prompt);
        h.update(std.mem.asBytes(&prompt_xxh64));
        h.final(&fp.settings);
    }
    var all: [ink.plan.max_world]Fingerprint = undefined;
    try g.gather(Fingerprint, d, s, fp, all[0..o.world]);
    var ok = true;
    for (all[0..o.world], 0..) |x, r| {
        const why: ?[]const u8 = if (x.rank != r or x.world != o.world) "rank or world" else if (!std.mem.eql(u8, &x.program, &fp.program)) "the executable" else if (!std.mem.eql(u8, &x.checkpoint, &fp.checkpoint)) "the checkpoint" else if (!std.mem.eql(u8, &x.settings, &fp.settings)) "the settings or prompt" else null;
        if (why) |w| {
            std.debug.print("rank {d}: rank {d} differs from this rank in {s}\n", .{ o.rank, r, w });
            ok = false;
        }
    }
    if (!ok) return error.RankMismatch;
    std.debug.print("rank {d} of {d}: fingerprint agreed (executable {x}, checkpoint and digest manifest {x}, settings and prompt {x}; prompt file xxh64 {x:0>16})\n", .{ o.rank, o.world, fp.program[0..6], fp.checkpoint[0..6], fp.settings[0..6], prompt_xxh64 });
}

/// xxh64 of a file's bytes, read 16 MiB at a time.
fn fileXxh64(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !u64 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const buf = try gpa.alloc(u8, 16 << 20);
    defer gpa.free(buf);
    var h = std.hash.XxHash64.init(0);
    var at: u64 = 0;
    while (true) {
        const n = try f.readPositionalAll(io, buf, at);
        h.update(buf[0..n]);
        at += n;
        if (n < buf.len) break;
    }
    return h.final();
}

/// A real rank stops: the others hear it on their liveness lines and stop, and this rank's communicator is aborted, so
/// no kernel of its stream waits on a peer (cuda/tp_group.zig). Safe to call more than once.
fn bail(g: *cuda.tp_group.Group, why: []const u8) void {
    cuda.tp_watch.fail(why);
    g.abort();
}

/// TF_INK2_FAULT (tests of the failure paths): `fail:R:L` rank R fails before layer L; `digest:R` rank R's final hash
/// of its layer outputs is off by one bit (as if its outputs differed).
const Fault = struct {
    kind: enum { none, fail, digest } = .none,
    rank: u32 = 0,
    layer: u32 = 0,

    fn read() Fault {
        const v = std.mem.span(std.c.getenv("TF_INK2_FAULT") orelse return .{});
        var it = std.mem.splitScalar(u8, v, ':');
        const k = it.next() orelse return .{};
        const r = std.fmt.parseInt(u32, it.next() orelse return .{}, 10) catch return .{};
        if (std.mem.eql(u8, k, "digest")) return .{ .kind = .digest, .rank = r };
        const l = std.fmt.parseInt(u32, it.next() orelse return .{}, 10) catch return .{};
        if (std.mem.eql(u8, k, "fail")) return .{ .kind = .fail, .rank = r, .layer = l };
        return .{};
    }

    fn at(f: Fault, kind: @TypeOf(f.kind), rank: u32, layer: ?u32) bool {
        return f.kind == kind and f.rank == rank and (layer == null or layer.? == f.layer);
    }
};

/// sha256 over the dump's layer outputs and routes (the tensors every rank computes).
fn digest(x: *const Dump) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var it = x.tensors.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.startsWith(u8, k, "h.") and !std.mem.startsWith(u8, k, "route.")) continue;
        h.update(k);
        h.update(e.value_ptr.data);
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn stopped(a: std.mem.Allocator, s: ?fwd.Stop, e: anyerror) ![]const u8 {
    const x = s orelse return std.fmt.allocPrint(a, "failed: {t}", .{e});
    if (x.err == error.NotImplemented) return std.fmt.allocPrint(a, "stopped at layer {f} op {s}: NotImplemented (no kernel yet)", .{ x.scope, x.op });
    return std.fmt.allocPrint(a, "failed at layer {f} op {s}: {t}", .{ x.scope, x.op, x.err });
}

/// The forward's input rows start .. start + R: layer lo's input in h[0], the blocks below lo.
fn inputs(gpa: std.mem.Allocator, f: *fwd.Forward, oracle: *const ink.st.Reader, ids: []const i64, o: Opts, start: usize, R: usize) !void {
    const act = f.a;
    act.nblocks = 0;
    if ((o.lo == 0 or o.synthetic) and o.embed) {
        if (f.w.embed == 0) return; // rank > 0: rank 0's rows come by shareEmbed
        const host = try gpa.alloc(i32, R);
        defer gpa.free(host);
        for (host, ids[start..][0..R]) |*h, v| h.* = @intCast(v);
        try upload(f, act.tokens.ptr, std.mem.sliceAsBytes(host));
        try f.embed(R, start);
        return;
    }
    try oracleState(gpa, f, oracle, o.lo, start, R);
}

/// Layer L's input from the oracle, rows start .. start + R: h.<L-1> (or the embedding) in h[0], and the blocks
/// below L (L = 48: the final mix's inputs).
fn oracleState(gpa: std.mem.Allocator, f: *fwd.Forward, oracle: *const ink.st.Reader, L: u32, start: usize, R: usize) !void {
    const act = f.a;
    var buf: [16]u8 = undefined;
    const name = if (L == 0) "embed" else try std.fmt.bufPrint(&buf, "h.{d}", .{L - 1});
    try rows(gpa, f, oracle, name, start, R, act.h[0].slice(0, R));
    const nb = fwd.blocksBefore(L);
    for (0..nb) |i| {
        const b: u32 = @intCast(6 * i); // the input of layer b: the embedding, or layer b - 1's output
        const src = if (b == 0) "embed" else try std.fmt.bufPrint(&buf, "h.{d}", .{b - 1});
        try rows(gpa, f, oracle, src, start, R, act.block(i, R));
    }
    act.nblocks = nb;
}

fn rows(gpa: std.mem.Allocator, f: *fwd.Forward, r: *const ink.st.Reader, name: []const u8, lo: usize, n: usize, dst: Mat) !void {
    const e = r.find(name) orelse {
        std.log.err("inkling2: the oracle file has no {s}", .{name});
        return error.MissingTensor;
    };
    if (e.dtype != .bf16 or e.rank != 2 or e.dim(1) != H or lo + n > e.dim(0)) return error.UnexpectedOracleTensor;
    const host = try gpa.alloc(u8, n * H * 2);
    defer gpa.free(host);
    try r.readInto(e, lo * H * 2, host);
    try upload(f, dst.ptr, host);
}

fn upload(f: *fwd.Forward, ptr: u64, bytes: []const u8) !void {
    try f.c.stream.synchronize();
    const view: cuda.DeviceBuffer = .{ .d = f.c.d, .ptr = ptr, .len = bytes.len, .borrowed = true };
    try view.upload(0, bytes);
}

/// --prof: one forward's table, ms a rank (the sum over this process's ranks / nlocal) by op and layer.
fn printProf(p: *const ic.forward.Prof, lo: u32, hi: u32, nlocal: u32, start: usize, R: usize) void {
    const n: f64 = @floatFromInt(nlocal);
    std.debug.print("prof forward rows {d}..{d} ({d} rows), ms a rank\nprof op", .{ start, start + R - 1, R });
    for (lo..hi) |l| std.debug.print("\tL{d}", .{l});
    std.debug.print("\tmean\n", .{});
    var tot: [ic.forward.Prof.rows]f64 = @splat(0);
    for (p.labels[0..p.nlabels], 0..) |name, j| {
        std.debug.print("prof {s}", .{name});
        var sum: f64 = 0;
        for (lo..hi) |l| {
            const v = p.ms[l][j] / n;
            tot[l] += v;
            sum += v;
            std.debug.print("\t{d:.3}", .{v});
        }
        std.debug.print("\t{d:.3}\n", .{sum / @as(f64, @floatFromInt(hi - lo))});
    }
    std.debug.print("prof total", .{});
    var sum: f64 = 0;
    for (lo..hi) |l| {
        sum += tot[l];
        std.debug.print("\t{d:.3}", .{tot[l]});
    }
    std.debug.print("\t{d:.3}\n", .{sum / @as(f64, @floatFromInt(hi - lo))});
}

/// The head over `ys` (the final rows, the same on every rank): each rank's share of the logits into its block of
/// `parts`, gathered in rank order (real ranks) or written there one rank after another (virtual), then rank 0's
/// softcap over all of them into `out`.
fn headShares(ranks: []Rank, o: Opts, ys: Mat, parts: u64, out: Mat, row0: usize) !void {
    for (ranks) |*r| try r.f.unembedShare(ys, parts, r.x.rank, row0);
    const n = ys.rows;
    const share: usize = ranks[0].w.vocab_rows.len();
    if (!o.virtual and o.world > 1) {
        const x = &ranks[0].x;
        const bytes = n * share * 2;
        try x.group.?.allGather(ranks[0].f.c.stream, parts + @as(u64, x.rank) * bytes, parts, bytes);
    }
    if (ranks[0].x.rank != 0) return;
    try ranks[0].f.softcapShares(parts, o.world, out, row0);
}

/// The head's selected rows and their logits: the ranks' bf16 shares as gathered, and the softcapped logits.
const HeadBufs = struct {
    rows: Mat,
    parts: u64,
    out: Mat,
    own: [3]cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver, n: usize, world: u32, share: usize, unpadded: usize) !HeadBufs {
        var own: [3]cuda.DeviceBuffer = undefined;
        own[0] = try cuda.DeviceBuffer.alloc(d, @max(n, 1) * H * 2);
        errdefer own[0].free();
        own[1] = try cuda.DeviceBuffer.alloc(d, @as(usize, world) * @max(n, 1) * share * 2);
        errdefer own[1].free();
        own[2] = try cuda.DeviceBuffer.alloc(d, @max(n, 1) * unpadded * 4);
        return .{ .rows = Mat.of(own[0].ptr, n, H, .bf16), .parts = own[1].ptr, .out = Mat.of(own[2].ptr, n, unpadded, .f32), .own = own };
    }

    fn deinit(h: *HeadBufs) void {
        for (&h.own) |*b| b.free();
    }
};

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-9;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}
