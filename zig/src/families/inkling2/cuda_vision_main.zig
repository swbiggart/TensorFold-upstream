//! tf-inkling2-vision: Inkling-2's vision tower alone on one GPU (cuda_vision.zig), over the image oracle's files
//! (tools/zig/inkling2_vision_ref.py): an input file holds {rgb u8 [H, W, 3], hw i32 [2], grid i64 [3],
//! pixel_values f32 [T, 40, 40, 3]}.
//!
//! Modes:
//!   (rows)   --model DIR --input X [--out OUT] [--hash] [--repeat N] [--bench N]: the tower's rows [T, 4608]; OUT
//!            holds rows, grid and each layer's output at the oracle's sampled rows (layer.<i>, layer.rows).
//!   (ops)    --model DIR --input X --ops ORACLE.ops.safetensors --op-out OUT: each op of E2 .. M2 on torch's
//!            recorded inputs, ours written under the oracle's op names (and res1/res2 for the residual ends),
//!            with each op's bit-equal share and max |diff| against torch's output printed and in OUT.json.
//!   (e1)     --e1 --input X: E1 on the GPU against bf16(pixel_values) in patch order, bit for bit (no weights).
//!   (tables) --tables TABLES.safetensors: GELU on all 65,536 bf16 inputs, inv_freq, and the position taps of
//!            each listed grid against the oracle's tables (no weights).
//! Settings: --row-block R (2048), --q-block Q (2048), --slice-layers N (23), --core ref|tc, --weights FILE
//! (default: the checkpoint shard holding model.visual.* by DIR's index). The GPU is the first visible device.
//! --variant engine (rows mode, --no-layers): the tower in engine mode, as the serving engine's rank 0 runs it: only
//! its weights and `x` its own, every other buffer borrowed per unit from one region of `sliceNeed` bytes (filled with
//! a junk byte before each unit, so nothing can carry across), one layer a unit, the rows into a buffer of their own.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const vis = @import("cuda_vision.zig");

const st = ink.st;
const Stream = cuda.Stream;

const usage =
    \\usage: tf-inkling2-vision --model DIR --input X.input.safetensors [--out OUT.safetensors] [--hash]
    \\                          [--repeat N] [--bench N] [--before Y] [--no-layers] [--variant V] [settings]
    \\       tf-inkling2-vision --model DIR --input X --ops ORACLE.ops.safetensors --op-out OUT.safetensors [settings]
    \\       tf-inkling2-vision --e1 --input X.input.safetensors
    \\       tf-inkling2-vision --tables TABLES.safetensors
    \\settings: --row-block R (2048) --q-block Q (2048) --slice-layers N (23) --core ref|tc (ref)
    \\          --weights FILE (default: DIR's shard holding model.visual.*)
    \\          --variant engine: engine mode (rows, --no-layers; borrowed buffers a unit, one layer a unit)
    \\The GPU is the first visible device (set CUDA_VISIBLE_DEVICES).
    \\
;

const Opts = struct {
    model: ?[]const u8 = null,
    weights: ?[]const u8 = null,
    input: ?[]const u8 = null,
    before: ?[]const u8 = null,
    out: ?[]const u8 = null,
    hash: bool = false,
    no_layers: bool = false,
    variant: []const u8 = "",
    repeat: usize = 1,
    bench: usize = 0,
    ops: ?[]const u8 = null,
    op_out: ?[]const u8 = null,
    e1: bool = false,
    tables: ?[]const u8 = null,
    slice_layers: usize = vis.layers,
    vo: vis.Options = .{},

    fn engine(o: *const Opts) bool {
        return std.mem.eql(u8, o.variant, "engine");
    }
};

fn parse(args: []const [:0]const u8) !?Opts {
    var o: Opts = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--hash")) {
            o.hash = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--no-layers")) {
            o.no_layers = true;
            continue;
        }
        if (std.mem.eql(u8, k, "--e1")) {
            o.e1 = true;
            continue;
        }
        if (i + 1 >= args.len) return null;
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, k, "--model")) {
            o.model = v;
        } else if (std.mem.eql(u8, k, "--weights")) {
            o.weights = v;
        } else if (std.mem.eql(u8, k, "--input")) {
            o.input = v;
        } else if (std.mem.eql(u8, k, "--variant")) {
            o.variant = v;
        } else if (std.mem.eql(u8, k, "--before")) {
            o.before = v;
        } else if (std.mem.eql(u8, k, "--out")) {
            o.out = v;
        } else if (std.mem.eql(u8, k, "--repeat")) {
            o.repeat = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--bench")) {
            o.bench = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--ops")) {
            o.ops = v;
        } else if (std.mem.eql(u8, k, "--op-out")) {
            o.op_out = v;
        } else if (std.mem.eql(u8, k, "--tables")) {
            o.tables = v;
        } else if (std.mem.eql(u8, k, "--row-block")) {
            o.vo.row_block = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--q-block")) {
            o.vo.q_block = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--slice-layers")) {
            o.slice_layers = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--core")) {
            o.vo.core = std.meta.stringToEnum(vis.Core, v) orelse return null;
        } else return null;
    }
    if (o.vo.row_block == 0 or o.vo.q_block == 0 or o.slice_layers == 0 or o.repeat == 0) return null;
    if (o.tables != null) return o;
    if (o.input == null) return null;
    if (o.e1) return o;
    if (o.model == null and o.weights == null) return null;
    if ((o.ops == null) != (o.op_out == null)) return null;
    if (o.engine()) {
        if (o.ops != null or o.before != null or o.bench > 0 or (o.out != null and !o.no_layers)) return null;
        o.vo.own_work = false;
    }
    return o;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const o = (parse(args) catch null) orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    return run(a, init.gpa, init.io, o) catch |e| {
        std.debug.print("FAIL tf-inkling2-vision: {t}\n", .{e});
        return 1;
    };
}

/// The safetensors file of DIR's index that holds model.visual.proj_out.weight.
fn weightsFile(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    if (doc != .object) return error.BadIndex;
    const map = doc.object.get("weight_map") orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    const f = map.object.get("model.visual.proj_out.weight") orelse return error.NoVisionTower;
    if (f != .string) return error.BadIndex;
    return std.fs.path.join(a, &.{ dir, f.string });
}

const Input = struct {
    h: usize,
    w: usize,
    grid: vis.Grid,
    rgb: []u8,
    reader: st.Reader,
};

fn openInput(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, path: []const u8) !Input {
    var r = try st.Reader.open(gpa, io, path);
    errdefer r.close();
    const hw = try r.alloc(a, "hw", .i32);
    const h: usize = @intCast(std.mem.readInt(i32, hw[0..4], .little));
    const w: usize = @intCast(std.mem.readInt(i32, hw[4..8], .little));
    const gb = try r.alloc(a, "grid", .i64);
    const g = vis.Grid.of(h, w);
    const rows: usize = @intCast(std.mem.readInt(i64, gb[gb.len - 16 ..][0..8], .little));
    const cols: usize = @intCast(std.mem.readInt(i64, gb[gb.len - 8 ..][0..8], .little));
    if (rows != g.rows or cols != g.cols) {
        std.log.err("{s}: grid {d}x{d}, but {d}x{d} pixels give {d}x{d}", .{ path, rows, cols, h, w, g.rows, g.cols });
        return error.UnexpectedGrid;
    }
    const rgb = try r.alloc(a, "rgb", .u8);
    if (rgb.len != h * w * 3) return error.UnexpectedInput;
    return .{ .h = h, .w = w, .grid = g, .rgb = rgb, .reader = r };
}

fn run(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&driver, true);
    defer stream.deinit();
    if (o.tables) |p| return tables(a, gpa, io, &driver, stream, p);
    var in = try openInput(gpa, a, io, o.input.?);
    defer in.reader.close();
    if (o.e1) return e1Check(a, &driver, stream, &in);
    const clock = std.Io.Clock.awake;
    const wpath = o.weights orelse try weightsFile(a, io, o.model.?);
    const t0 = clock.now(io).toNanoseconds();
    var tower = try vis.Tower.load(gpa, io, &driver, wpath, o.vo, .{});
    defer tower.deinit();
    std.debug.print("tower loaded in {d:.1} s from {s}: {d} MiB on the device; row block {d}, q block {d}, core {t}\n", .{ secs(clock.now(io).toNanoseconds() - t0), std.fs.path.basename(wpath), tower.deviceBytes() >> 20, o.vo.row_block, o.vo.q_block, o.vo.core });
    if (o.ops) |p| return opCheck(a, gpa, io, &driver, stream, &tower, &in, p, o.op_out.?);
    if (o.before) |p| { // another image through the tower first: its state must not reach this image's rows
        var b = try openInput(gpa, a, io, p);
        defer b.reader.close();
        _ = try tower.run(stream, b.rgb, b.h, b.w);
        try stream.synchronize();
        std.debug.print("before: {s} ({d}x{d}) through the tower first\n", .{ std.fs.path.basename(p), b.w, b.h });
    }
    return towerRows(a, gpa, io, &driver, stream, &tower, &in, o);
}

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-9;
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}

// ---------------------------------------------------------------------------------------------------------- rows

/// The oracle's sampled tower rows (inkling2_vision_ref.py sample_tower_rows): 0 .. 32, the last 32, every 64th.
fn sampleRows(a: std.mem.Allocator, S: usize) ![]i32 {
    var list: std.ArrayList(i32) = .empty;
    for (0..S) |r| {
        if (r < 33 or r + 32 >= S or r % 64 == 0) try list.append(a, @intCast(r));
    }
    return list.items;
}

fn towerRows(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: Stream, tower: *vis.Tower, in: *const Input, o: Opts) !u8 {
    const clock = std.Io.Clock.awake;
    const g = in.grid;
    const S = g.towerRows();
    const T = g.tokens();
    std.debug.print("image {d}x{d}: grid {d}x{d}, {d} tokens, {d} tower rows\n", .{ in.w, in.h, g.rows, g.cols, T, S });
    const host = try a.alloc(u8, T * vis.out_width * 2);
    var first: ?[32]u8 = null;
    var same = true;
    // the sampled layer rows when OUT is wanted (one layer a step then, whatever --slice-layers says)
    var sample: []i32 = &.{};
    var sample_dev: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
    var gathered: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
    var layer_rows: [vis.layers][]u8 = undefined;
    const layers_out = o.out != null and !o.no_layers;
    if (layers_out) {
        sample = try sampleRows(a, S);
        sample_dev = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(sample));
        gathered = try cuda.DeviceBuffer.alloc(d, sample.len * vis.hidden * 2);
        for (&layer_rows) |*l| l.* = try a.alloc(u8, sample.len * vis.hidden * 2);
    }
    defer if (layers_out) {
        sample_dev.free();
        gathered.free();
    };
    // engine mode: one borrowed region a unit, and the rows' own buffer
    var region: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
    var rows_buf: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
    defer if (o.engine()) {
        region.free();
        rows_buf.free();
    };
    if (o.engine()) {
        region = try cuda.DeviceBuffer.alloc(d, tower.sliceNeed(g));
        rows_buf = try cuda.DeviceBuffer.alloc(d, T * vis.out_width * 2);
        std.debug.print("engine mode: {d} MiB resident, {d:.1} MiB borrowed a unit for this image ({d:.1} MiB at {d} a side)\n", .{ tower.deviceBytes() >> 20, @as(f64, @floatFromInt(region.len)) / (1 << 20), @as(f64, @floatFromInt(tower.sliceNeed(vis.Grid.of(o.vo.max_side, o.vo.max_side)))) / (1 << 20), o.vo.max_side });
    }
    for (0..o.repeat) |rep| {
        const t0 = clock.now(io).toNanoseconds();
        if (o.engine()) {
            try region.fill8(0x5a, s.handle);
            try tower.bindSlice(s, region.ptr, region.len, .begin, g);
            try tower.begin(s, in.rgb, in.h, in.w);
            for (0..vis.layers) |_| {
                try region.fill8(0x5a, s.handle);
                try tower.bindSlice(s, region.ptr, region.len, .layer, g);
                _ = try tower.step(s, 1);
            }
            try region.fill8(0x5a, s.handle);
            try tower.bindSlice(s, region.ptr, region.len, .merger, g);
            const out = try tower.finishInto(s, rows_buf.ptr);
            try s.synchronize();
            const t1 = clock.now(io).toNanoseconds();
            try cuda.memory.download(d, out.ptr, host);
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(host, &dg, .{});
            if (first) |f| {
                if (!std.mem.eql(u8, &f, &dg)) same = false;
            } else first = dg;
            if (o.hash) std.debug.print("HASH {s} run {d} rows [{d},{d}] sha256 {s} ({d:.1} ms; engine mode, one layer a unit, row block {d}, q block {d}, core {t})\n", .{ std.fs.path.basename(o.input.?), rep, T, vis.out_width, std.fmt.bytesToHex(dg, .lower), ms(t1 - t0), o.vo.row_block, o.vo.q_block, o.vo.core });
            continue;
        }
        try tower.begin(s, in.rgb, in.h, in.w);
        if (layers_out) {
            var i: usize = 0;
            while (i < vis.layers) : (i += 1) {
                _ = try tower.step(s, 1);
                try tower.k.gatherRows(s, tower.x, sample_dev.ptr, sample.len, gathered.ptr);
                try s.synchronize();
                try gathered.download(0, layer_rows[i]);
            }
        } else {
            while (try tower.step(s, o.slice_layers) > 0) {}
        }
        const out = try tower.finish(s);
        try s.synchronize();
        const t1 = clock.now(io).toNanoseconds();
        try cuda.memory.download(d, out.ptr, host);
        var dg: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &dg, .{});
        if (first) |f| {
            if (!std.mem.eql(u8, &f, &dg)) same = false;
        } else first = dg;
        if (o.hash) std.debug.print("HASH {s} run {d} rows [{d},{d}] sha256 {s} ({d:.1} ms; row block {d}, q block {d}, slice {d}, core {t})\n", .{ std.fs.path.basename(o.input.?), rep, T, vis.out_width, std.fmt.bytesToHex(dg, .lower), ms(t1 - t0), o.vo.row_block, o.vo.q_block, if (layers_out) 1 else o.slice_layers, o.vo.core });
    }
    if (o.repeat > 1) std.debug.print("REPEAT {d} runs: {s}\n", .{ o.repeat, if (same) "identical" else "DIFFER" });
    if (o.bench > 0) {
        const times = try a.alloc(f64, o.bench);
        _ = try tower.run(s, in.rgb, in.h, in.w); // warm-up
        try s.synchronize();
        for (times) |*x| {
            const t0 = clock.now(io).toNanoseconds();
            try tower.begin(s, in.rgb, in.h, in.w);
            while (try tower.step(s, o.slice_layers) > 0) {}
            _ = try tower.finish(s);
            try s.synchronize();
            x.* = ms(clock.now(io).toNanoseconds() - t0);
        }
        std.mem.sort(f64, times, {}, std.sort.asc(f64));
        std.debug.print("BENCH {s} {d}x{d} tokens {d} S {d}: median {d:.2} ms over {d} runs (min {d:.2}, max {d:.2}); core {t}, row block {d}, q block {d}\n", .{ std.fs.path.basename(o.input.?), in.w, in.h, T, S, times[times.len / 2], times.len, times[0], times[times.len - 1], o.vo.core, o.vo.row_block, o.vo.q_block });
    }
    if (o.out) |path| {
        var wr = st.Writer.init(gpa);
        defer wr.deinit();
        try wr.add("rows", .bf16, &.{ T, vis.out_width }, host);
        var gb: [3]i64 = .{ 1, @intCast(g.rows), @intCast(g.cols) };
        try wr.add("grid", .i64, &.{ 1, 3 }, std.mem.sliceAsBytes(&gb));
        if (layers_out) {
            const lr = try a.alloc(i64, sample.len);
            for (lr, sample) |*x, r| x.* = r;
            try wr.add("layer.rows", .i64, &.{lr.len}, std.mem.sliceAsBytes(lr));
            for (layer_rows, 0..) |l, i| {
                var nb: [32]u8 = undefined;
                try wr.add(try std.fmt.bufPrint(&nb, "layer.{d}", .{i}), .bf16, &.{ sample.len, vis.hidden }, l);
            }
        }
        if (o.variant.len > 0) try wr.setMeta("variant", o.variant);
        try wr.setMeta("fixture", std.fs.path.basename(o.input.?)[0 .. std.mem.indexOf(u8, std.fs.path.basename(o.input.?), ".input") orelse std.fs.path.basename(o.input.?).len]);
        try wr.setMeta("tool", "tf-inkling2-vision");
        try wr.setMeta("input", std.fs.path.basename(o.input.?));
        try wr.setMeta("core", @tagName(o.vo.core));
        try wr.setMeta("sha256", &std.fmt.bytesToHex(first.?, .lower));
        try wr.write(io, path);
        std.debug.print("wrote {s}\n", .{path});
    }
    return if (same) 0 else 1;
}

// ---------------------------------------------------------------------------------------------------------- E1

fn e1Check(a: std.mem.Allocator, d: *const cuda.Driver, s: Stream, in: *const Input) !u8 {
    const g = in.grid;
    const P = g.patches();
    const pv_bytes = try in.reader.alloc(a, "pixel_values", .f32);
    const pv = std.mem.bytesAsSlice(f32, pv_bytes);
    if (pv.len != g.tokens() * vis.block * vis.block * 3) return error.UnexpectedInput;
    var k = try vis.Kernels.load(d);
    defer k.unload();
    const table = vis.pixelTable();
    var tb = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&table));
    defer tb.free();
    var rgb = try cuda.DeviceBuffer.fromHost(d, in.rgb);
    defer rgb.free();
    var out = try cuda.DeviceBuffer.alloc(d, P * vis.features_pad * 2);
    defer out.free();
    try k.patchRows(s, rgb.ptr, tb.ptr, in.h, in.w, g.cols, 0, P, out.ptr);
    try s.synchronize();
    const got_b = try a.alloc(u8, P * vis.features_pad * 2);
    try out.download(0, got_b);
    const got = std.mem.bytesAsSlice(u16, got_b);
    var equal: usize = 0;
    var zero_tail = true;
    for (0..P) |p| {
        const t = p / 4;
        const mi = (p % 4) / 2;
        const mj = p % 2;
        for (0..vis.features) |f| {
            const py = f / 60;
            const px = (f / 3) % 20;
            const c = f % 3;
            const want = st.f32ToBf16(pv[((t * vis.block + mi * vis.patch + py) * vis.block + mj * vis.patch + px) * 3 + c]);
            if (got[p * vis.features_pad + f] == want) equal += 1;
        }
        for (vis.features..vis.features_pad) |f| if (got[p * vis.features_pad + f] != 0) {
            zero_tail = false;
        };
    }
    const total = P * vis.features;
    std.debug.print("E1 {d}x{d} grid {d}x{d}: {d} patches, bit-equal {d}/{d} to bf16(pixel_values), padding columns zero {}\n", .{ in.w, in.h, g.rows, g.cols, P, equal, total, zero_tail });
    return if (equal == total and zero_tail) 0 else 1;
}

// ------------------------------------------------------------------------------------------------------ tables

fn tables(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: Stream, path: []const u8) !u8 {
    var r = try st.Reader.open(gpa, io, path);
    defer r.close();
    var bad: usize = 0;
    // GELU on every bf16 input
    {
        const gin = try r.alloc(a, "gelu.in", .bf16);
        const gout = try r.alloc(a, "gelu.out", .bf16);
        var k = try vis.Kernels.load(d);
        defer k.unload();
        var b = try cuda.DeviceBuffer.fromHost(d, gin);
        defer b.free();
        const n = gin.len / 2;
        try k.gelu(s, b.ptr, b.ptr, n);
        try s.synchronize();
        const got = try a.alloc(u8, gin.len);
        try b.download(0, got);
        const c = compareBf16(std.mem.bytesAsSlice(u16, got), std.mem.bytesAsSlice(u16, gout));
        std.debug.print("TABLE gelu: {d} inputs, bit-equal {d}, max |diff| {e}\n", .{ n, c.equal, c.max_abs });
        const xs = std.mem.bytesAsSlice(u16, gin);
        const ys = std.mem.bytesAsSlice(u16, got);
        const zs = std.mem.bytesAsSlice(u16, gout);
        var shown: usize = 0;
        for (xs, ys, zs) |x, y, z| if (y != z and shown < 16) {
            shown += 1;
            std.debug.print("  gelu differs: in {x:0>4} ({e}) ours {x:0>4} ({e}) torch {x:0>4} ({e})\n", .{ x, st.bf16ToF32(x), y, st.bf16ToF32(y), z, st.bf16ToF32(z) });
        };
        if (c.equal != n) bad += 1;
    }
    {
        const inv = try r.alloc(a, "rope.inv_freq", .f32);
        const want = std.mem.bytesAsSlice(u32, inv);
        var eq: usize = 0;
        for (vis.invFreq(), 0..) |f, j| {
            if (@as(u32, @bitCast(f)) == want[j]) eq += 1;
        }
        std.debug.print("TABLE rope.inv_freq: bit-equal {d}/16\n", .{eq});
        if (eq != 16) bad += 1;
    }
    for (r.entries) |e| {
        if (!std.mem.startsWith(u8, e.name, "pos.idx.")) continue;
        const tag = e.name["pos.idx.".len..];
        const x = std.mem.indexOfScalar(u8, tag, 'x') orelse continue;
        const g: vis.Grid = .{ .rows = try std.fmt.parseInt(usize, tag[0..x], 10), .cols = try std.fmt.parseInt(usize, tag[x + 1 ..], 10) };
        const P = g.patches();
        const idx_b = try r.alloc(a, e.name, .i64);
        var nb: [64]u8 = undefined;
        const w_b = try r.alloc(a, try std.fmt.bufPrint(&nb, "pos.w.{s}", .{tag}), .f32);
        const want_idx = std.mem.bytesAsSlice(i64, idx_b);
        const want_w = std.mem.bytesAsSlice(u32, w_b);
        if (want_idx.len != 4 * P or want_w.len != 4 * P) return error.UnexpectedTable;
        const idx = try a.alloc(i32, 4 * P);
        const w = try a.alloc(f32, 4 * P);
        vis.positionTaps(g, idx, w);
        var ei: usize = 0;
        var ew: usize = 0;
        for (0..4 * P) |i| {
            if (idx[i] == want_idx[i]) ei += 1;
            if (@as(u32, @bitCast(w[i])) == want_w[i]) ew += 1;
        }
        std.debug.print("TABLE pos {d}x{d}: {d} taps, indices bit-equal {d}, weights bit-equal {d}\n", .{ g.rows, g.cols, 4 * P, ei, ew });
        if (ei != 4 * P or ew != 4 * P) bad += 1;
    }
    std.debug.print("TABLES: {d} tables differ\n", .{bad});
    return if (bad == 0) 0 else 1;
}

// --------------------------------------------------------------------------------------------------------- ops

const Cmp = struct { n: usize, equal: usize, max_abs: f64 };

fn compareBf16(got: []align(1) const u16, want: []align(1) const u16) Cmp {
    var c: Cmp = .{ .n = want.len, .equal = 0, .max_abs = 0 };
    for (got, want) |x, y| {
        if (x == y) {
            c.equal += 1;
            continue;
        }
        const dx: f64 = @abs(@as(f64, st.bf16ToF32(x)) - @as(f64, st.bf16ToF32(y)));
        if (!(dx <= c.max_abs)) c.max_abs = dx; // NaN shows as NaN
    }
    return c;
}

fn compareF32(got: []align(1) const u32, want: []align(1) const u32) Cmp {
    var c: Cmp = .{ .n = want.len, .equal = 0, .max_abs = 0 };
    for (got, want) |x, y| {
        if (x == y) {
            c.equal += 1;
            continue;
        }
        const dx: f64 = @abs(@as(f64, @as(f32, @bitCast(x))) - @as(f64, @as(f32, @bitCast(y))));
        if (!(dx <= c.max_abs)) c.max_abs = dx;
    }
    return c;
}

const OpCtx = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    s: Stream,
    r: *const st.Reader,
    wr: *st.Writer,
    json: *std.ArrayList(u8),
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,

    /// torch's tensor `name` (dtype dt) on the device, its bytes too.
    fn up(c: *OpCtx, name: []const u8, dt: st.DType) !struct { ptr: u64, host: []u8, e: st.Entry } {
        const e = c.r.find(name) orelse {
            std.log.err("the ops file has no {s}", .{name});
            return error.MissingTensor;
        };
        const host = try c.r.alloc(c.a, name, dt);
        const b = try cuda.DeviceBuffer.fromHost(c.d, host);
        try c.own.append(c.a, b);
        return .{ .ptr = b.ptr, .host = host, .e = e };
    }

    fn alloc(c: *OpCtx, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(c.d, bytes);
        try b.fill8(0, null);
        try c.own.append(c.a, b);
        return b.ptr;
    }

    fn down(c: *OpCtx, ptr: u64, bytes: usize) ![]u8 {
        try c.s.synchronize();
        const h = try c.a.alloc(u8, bytes);
        try cuda.memory.download(c.d, ptr, h);
        return h;
    }

    /// Ours (host bytes, torch's shape of `ref`) written as `key`, compared with torch's `ref`.
    fn report(c: *OpCtx, key: []const u8, ours: []const u8, ref: []const u8, dt: st.DType) !void {
        const e = c.r.find(ref) orelse return error.MissingTensor;
        const want = try c.r.alloc(c.a, ref, dt);
        if (want.len != ours.len) return error.UnexpectedShape;
        const cmp = if (dt == .f32)
            compareF32(std.mem.bytesAsSlice(u32, ours), std.mem.bytesAsSlice(u32, want))
        else
            compareBf16(std.mem.bytesAsSlice(u16, ours), std.mem.bytesAsSlice(u16, want));
        try c.wr.add(key, dt, e.shape[0..e.rank], ours);
        const share = @as(f64, @floatFromInt(cmp.equal)) / @as(f64, @floatFromInt(@max(cmp.n, 1)));
        std.debug.print("OP {s:<24} vs {s:<24} n {d:>10}  bit-equal {d:.6}  max|diff| {e}\n", .{ key, ref, cmp.n, share, cmp.max_abs });
        const sep: []const u8 = if (c.json.items.len > 1) ",\n" else "\n";
        try c.json.appendSlice(c.a, try std.fmt.allocPrint(c.a, "{s}  \"{s}\": {{\"ref\": \"{s}\", \"n\": {d}, \"equal\": {d}, \"share\": {d:.8}, \"max_abs\": {e}}}", .{ sep, key, ref, cmp.n, cmp.equal, share, cmp.max_abs }));
    }

    fn free(c: *OpCtx) void {
        for (c.own.items) |*b| b.free();
        c.own.clearRetainingCapacity();
    }
};

/// [1, 16, S, 64] to [S, 1024] (head-major to row-major), 2-byte elements.
fn headsToRows(a: std.mem.Allocator, x: []const u8, S: usize) ![]u8 {
    const out = try a.alloc(u8, x.len);
    const src = std.mem.bytesAsSlice(u16, x);
    const dst = std.mem.bytesAsSlice(u16, out);
    for (0..vis.heads) |h| for (0..S) |r| {
        @memcpy(dst[r * vis.hidden + h * vis.head_dim ..][0..vis.head_dim], src[(h * S + r) * vis.head_dim ..][0..vis.head_dim]);
    };
    return out;
}

fn rowsToHeads(a: std.mem.Allocator, x: []const u8, S: usize) ![]u8 {
    const out = try a.alloc(u8, x.len);
    const src = std.mem.bytesAsSlice(u16, x);
    const dst = std.mem.bytesAsSlice(u16, out);
    for (0..vis.heads) |h| for (0..S) |r| {
        @memcpy(dst[(h * S + r) * vis.head_dim ..][0..vis.head_dim], src[r * vis.hidden + h * vis.head_dim ..][0..vis.head_dim]);
    };
    return out;
}

fn opCheck(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: Stream, t: *vis.Tower, in: *const Input, ops_path: []const u8, out_path: []const u8) !u8 {
    var r = try st.Reader.open(gpa, io, ops_path);
    defer r.close();
    var wr = st.Writer.init(gpa);
    defer wr.deinit();
    var json: std.ArrayList(u8) = .empty;
    try json.append(a, '{');
    var c: OpCtx = .{ .a = a, .d = d, .s = s, .r = &r, .wr = &wr, .json = &json };
    defer c.free();
    const g = in.grid;
    const S = g.towerRows();
    const P = g.patches();
    const T = g.tokens();
    const H = vis.hidden;
    std.debug.print("ops {s}: grid {d}x{d}, S {d}\n", .{ std.fs.path.basename(ops_path), g.rows, g.cols, S });
    // the image's taps and rotary tables (no pixels)
    try t.prepareGrid(s, g);
    // E2 from torch's patches (padded to 1216)
    {
        const pin = try r.alloc(a, "E2.patch.in", .bf16);
        if (pin.len != P * vis.features * 2) return error.UnexpectedShape;
        const pad = try a.alloc(u8, P * vis.features_pad * 2);
        @memset(pad, 0);
        for (0..P) |p| @memcpy(pad[p * vis.features_pad * 2 ..][0 .. vis.features * 2], pin[p * vis.features * 2 ..][0 .. vis.features * 2]);
        var b = try cuda.DeviceBuffer.fromHost(d, pad);
        try c.own.append(a, b);
        _ = &b;
        const y = try c.alloc(P * H * 2);
        try t.linear(s, b.ptr, vis.features_pad, P, t.patch_w, y);
        try c.report("E2.patch.out", try c.down(y, P * H * 2), "E2.patch.out", .bf16);
    }
    // E3 + E4 from torch's E2 output
    {
        const e2 = try c.up("E2.patch.out", .bf16);
        const x = try c.alloc(S * H * 2);
        var p0: usize = 0;
        while (p0 < P) : (p0 += t.opt.row_block) {
            const n = @min(t.opt.row_block, P - p0);
            try t.k.embedRows(s, e2.ptr + p0 * H * 2, t.pos_table, t.idx, t.wts, p0, n, x);
        }
        try t.k.classRow(s, t.class, t.cls_pos, x);
        try c.report("E4.embeddings.out", try c.down(x, S * H * 2), "E4.embeddings.out", .bf16);
    }
    {
        const xin = try c.up("E5.ln_pre.in", .bf16);
        const y = try c.alloc(S * H * 2);
        try t.layerNormRows(s, xin.ptr, t.ln_pre_w, t.ln_pre_b, y, S);
        try c.report("E5.ln_pre.out", try c.down(y, S * H * 2), "E5.ln_pre.out", .bf16);
    }
    c.free();
    for (0..vis.layers) |li| {
        var nb: [64]u8 = undefined;
        if (r.find(try std.fmt.bufPrint(&nb, "L{d}.layer.in", .{li})) == null) continue;
        const l = &t.layer[li];
        const N = struct {
            fn of(al: std.mem.Allocator, i: usize, name: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "L{d}.{s}", .{ i, name });
            }
        };
        // L1
        {
            const xin = try c.up(try N.of(a, li, "L1.ln_1.in"), .bf16);
            const y = try c.alloc(S * H * 2);
            try t.layerNormRows(s, xin.ptr, l.ln1_w, l.ln1_b, y, S);
            try c.report(try N.of(a, li, "L1.ln_1.out"), try c.down(y, S * H * 2), try N.of(a, li, "L1.ln_1.out"), .bf16);
        }
        // L2: the fused q|k|v with its bias, from torch's normed rows
        {
            const xin = try c.up(try N.of(a, li, "L1.ln_1.out"), .bf16);
            const y = try c.alloc(S * 3 * H * 2);
            try t.linearBias(s, xin.ptr, H, S, l.qkv, l.b_qkv, y, 3 * H);
            const all = try c.down(y, S * 3 * H * 2);
            for ([_][]const u8{ "q", "k", "v" }, 0..) |p, j| {
                const part = try a.alloc(u8, S * H * 2);
                for (0..S) |row| @memcpy(part[row * H * 2 ..][0 .. H * 2], all[(row * 3 + j) * H * 2 ..][0 .. H * 2]);
                const key = try N.of(a, li, try std.fmt.allocPrint(a, "L2.{s}_proj.out", .{p}));
                try c.report(key, part, key, .bf16);
            }
        }
        // L3: rotary from torch's q, k; and our cos, sin against torch's
        {
            const qi = try c.r.alloc(a, try N.of(a, li, "rot.q_in"), .bf16);
            const ki = try c.r.alloc(a, try N.of(a, li, "rot.k_in"), .bf16);
            if (qi.len != S * H * 2 or ki.len != S * H * 2) return error.UnexpectedShape;
            const qkv = try a.alloc(u8, S * 3 * H * 2);
            @memset(qkv, 0);
            for (0..S) |row| {
                @memcpy(qkv[row * 3 * H * 2 ..][0 .. H * 2], qi[row * H * 2 ..][0 .. H * 2]);
                @memcpy(qkv[(row * 3 + 1) * H * 2 ..][0 .. H * 2], ki[row * H * 2 ..][0 .. H * 2]);
            }
            var b = try cuda.DeviceBuffer.fromHost(d, qkv);
            try c.own.append(a, b);
            _ = &b;
            var r0: usize = 0;
            while (r0 < S) : (r0 += t.opt.row_block) try t.k.rotary(s, b.ptr, 3 * H, t.cs, t.sn, r0, @min(t.opt.row_block, S - r0));
            const all = try c.down(b.ptr, S * 3 * H * 2);
            for ([_][]const u8{ "q", "k" }, 0..) |p, j| {
                const part = try a.alloc(u8, S * H * 2);
                for (0..S) |row| @memcpy(part[row * H * 2 ..][0 .. H * 2], all[(row * 3 + j) * H * 2 ..][0 .. H * 2]);
                const key = try N.of(a, li, try std.fmt.allocPrint(a, "rot.{s}_out", .{p}));
                try c.report(key, part, key, .bf16);
            }
            for ([_][]const u8{ "cos", "sin" }, [_]u64{ t.cs, t.sn }) |p, ptr| {
                const tab = try c.down(ptr, S * 32 * 4);
                const v32 = std.mem.bytesAsSlice(u32, tab);
                const wide = try a.alloc(u32, S * 64);
                for (0..S) |row| for (0..32) |i| {
                    wide[row * 64 + 2 * i] = v32[row * 32 + i];
                    wide[row * 64 + 2 * i + 1] = v32[row * 32 + i];
                };
                const key = try N.of(a, li, try std.fmt.allocPrint(a, "rot.{s}", .{p}));
                try c.report(key, std.mem.sliceAsBytes(wide), key, .f32);
            }
        }
        // L4: the core on torch's q', k, v
        {
            var bufs: [3]u64 = undefined;
            for ([_][]const u8{ "core.q", "core.k", "core.v" }, &bufs) |p, *ptr| {
                const raw = try c.r.alloc(a, try N.of(a, li, p), .bf16);
                if (raw.len != S * H * 2) return error.UnexpectedShape;
                var b = try cuda.DeviceBuffer.fromHost(d, try headsToRows(a, raw, S));
                try c.own.append(a, b);
                _ = &b;
                ptr.* = b.ptr;
            }
            const y = try c.alloc(S * H * 2);
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            try t.attention(s, t.opt.core, bufs[0], bufs[1], bufs[2], H, S, y);
            try s.synchronize();
            std.debug.print("core {t} on {d} rows: {d:.1} ms\n", .{ t.opt.core, S, ms(std.Io.Clock.awake.now(io).toNanoseconds() - t0) });
            const key = try N.of(a, li, "core.out");
            try c.report(key, try rowsToHeads(a, try c.down(y, S * H * 2), S), key, .bf16);
        }
        // L5: out_proj with its bias
        {
            const xin = try c.up(try N.of(a, li, "L2.out_proj.in"), .bf16);
            const y = try c.alloc(S * H * 2);
            try t.linearBias(s, xin.ptr, H, S, l.o, l.b_o, y, H);
            const key = try N.of(a, li, "L2.out_proj.out");
            try c.report(key, try c.down(y, S * H * 2), key, .bf16);
        }
        // L6: layer.in + ls_1 * torch's out_proj output (torch's value: ln_2's input)
        {
            const x = try c.up(try N.of(a, li, "layer.in"), .bf16);
            const y = try c.up(try N.of(a, li, "L2.out_proj.out"), .bf16);
            var r0: usize = 0;
            while (r0 < S) : (r0 += t.opt.row_block) try t.k.lsRes(s, y.ptr + r0 * H * 2, l.ls1, x.ptr + r0 * H * 2, @min(t.opt.row_block, S - r0));
            try c.report(try N.of(a, li, "res1.out"), try c.down(x.ptr, S * H * 2), try N.of(a, li, "L7.ln_2.in"), .bf16);
        }
        {
            const xin = try c.up(try N.of(a, li, "L7.ln_2.in"), .bf16);
            const y = try c.alloc(S * H * 2);
            try t.layerNormRows(s, xin.ptr, l.ln2_w, l.ln2_b, y, S);
            const key = try N.of(a, li, "L7.ln_2.out");
            try c.report(key, try c.down(y, S * H * 2), key, .bf16);
        }
        // L8: fc1 + bias + GELU from fc1's input; fc2 + bias from fc2's input
        {
            const xin = try c.up(try N.of(a, li, "L8.fc1.in"), .bf16);
            const y = try c.alloc(S * vis.inter * 2);
            try t.linearBiasGelu(s, xin.ptr, H, S, l.fc1, l.b_fc1, y);
            const key = try N.of(a, li, "L8.gelu.out");
            try c.report(key, try c.down(y, S * vis.inter * 2), key, .bf16);
        }
        {
            const xin = try c.up(try N.of(a, li, "L8.fc2.in"), .bf16);
            const y = try c.alloc(S * H * 2);
            try t.linearBias(s, xin.ptr, vis.inter, S, l.fc2, l.b_fc2, y, H);
            const key = try N.of(a, li, "L8.fc2.out");
            try c.report(key, try c.down(y, S * H * 2), key, .bf16);
        }
        // L9: ln_2's input + ls_2 * torch's fc2 output (torch's value: the layer's output)
        {
            const x = try c.up(try N.of(a, li, "L7.ln_2.in"), .bf16);
            const y = try c.up(try N.of(a, li, "L8.fc2.out"), .bf16);
            var r0: usize = 0;
            while (r0 < S) : (r0 += t.opt.row_block) try t.k.lsRes(s, y.ptr + r0 * H * 2, l.ls2, x.ptr + r0 * H * 2, @min(t.opt.row_block, S - r0));
            try c.report(try N.of(a, li, "res2.out"), try c.down(x.ptr, S * H * 2), try N.of(a, li, "layer.out"), .bf16);
        }
        c.free();
    }
    // M2
    {
        const xin = try c.up("M2.linear_1.in", .bf16);
        const y = try c.alloc(T * vis.proj_hidden * 2);
        try t.linear(s, xin.ptr, 4 * H, T, t.m1, y);
        try c.report("M2.linear_1.out", try c.down(y, T * vis.proj_hidden * 2), "M2.linear_1.out", .bf16);
    }
    {
        const xin = try c.up("M2.gelu.in", .bf16);
        const y = try c.alloc(T * vis.proj_hidden * 2);
        try t.k.gelu(s, xin.ptr, y, T * vis.proj_hidden);
        try c.report("M2.gelu.out", try c.down(y, T * vis.proj_hidden * 2), "M2.gelu.out", .bf16);
    }
    {
        const xin = try c.up("M2.linear_2.in", .bf16);
        const y = try c.alloc(T * vis.out_width * 2);
        try t.linear(s, xin.ptr, vis.proj_hidden, T, t.m2, y);
        try c.report("M2.linear_2.out", try c.down(y, T * vis.out_width * 2), "M2.linear_2.out", .bf16);
    }
    try json.appendSlice(a, "\n}\n");
    try wr.setMeta("tool", "tf-inkling2-vision --ops");
    try wr.setMeta("ops", std.fs.path.basename(ops_path));
    try wr.setMeta("core", @tagName(t.opt.core));
    try wr.write(io, out_path);
    const jpath = try std.fmt.allocPrint(a, "{s}.json", .{out_path});
    const f = try std.Io.Dir.cwd().createFile(io, jpath, .{ .truncate = true });
    defer f.close(io);
    try f.writePositionalAll(io, json.items, 0);
    std.debug.print("wrote {s} and {s}\n", .{ out_path, jpath });
    return 0;
}
