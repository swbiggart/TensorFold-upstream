//! tf-inkling2-draft: the CUDA drafter alone on a taps file (inkling2_dspark_ref.py gen: ids, n_prompt, taps
//! [T, 8, 4608]): positions 0 .. P - 1 absorbed from the file's taps, then the block at anchor P (the token ids[P]),
//! its intermediates written to OUT for `inkling2_dspark_ref.py check` against the torch reference's `dump`.
//! With --all, every anchor from n_prompt on: the proposals and the accepted lengths against the file's tokens;
//! --chances (with --all's lines) also each draft's pick probability (the lone cut's chances, uncalibrated).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");

const dspark = ic.dspark;
const ops = ic.ops;
const H = ic.weights.hidden;

const usage =
    \\usage: tf-inkling2-draft <model dir> <taps file> (--anchor P --out FILE | --all | --chances)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var anchor: ?usize = null;
    var out: ?[]const u8 = null;
    var all = false;
    var chances = false;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--anchor") and i + 1 < args.len) {
            anchor = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--out") and i + 1 < args.len) {
            out = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--all")) all = true else if (std.mem.eql(u8, args[i], "--chances")) {
            all = true;
            chances = true;
        } else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    return run(init.gpa, init.io, args[1], args[2], anchor, out, all, chances) catch |e| {
        std.debug.print("FAIL tf-inkling2-draft: {t}\n", .{e});
        return 1;
    };
}

fn run(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, taps_path: []const u8, anchor: ?usize, out: ?[]const u8, all: bool, chances: bool) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var dg: ink.Diag = .{};
    var model = ink.describe.Model.fromDir(gpa, io, dir, .{ .world = 1 }, &dg) catch |err| {
        std.log.err("{s}", .{dg.text()});
        return err;
    };
    defer model.deinit();
    var k = try ic.kernels.Kernels.load(&driver);
    defer k.unload();
    var gemms: ops.Gemms = .{ .gpa = gpa };
    var scratch = try ops.Scratch.init(&driver, 64 << 20);
    defer scratch.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const c: ops.Ctx = .{ .d = &driver, .stream = stream, .k = &k, .gemms = &gemms, .scratch = &scratch };
    // the taps file
    var r = try ink.st.Reader.open(gpa, io, taps_path);
    defer r.close();
    const ids_raw = try r.alloc(gpa, "ids", .i64);
    defer gpa.free(ids_raw);
    const T = ids_raw.len / 8;
    const ids = try gpa.alloc(u32, T);
    defer gpa.free(ids);
    for (ids, 0..) |*t, j| t.* = @intCast(std.mem.readInt(i64, ids_raw[8 * j ..][0..8], .little));
    const np_raw = try r.alloc(gpa, "n_prompt", .i64);
    defer gpa.free(np_raw);
    const n0: usize = @intCast(std.mem.readInt(i64, np_raw[0..8], .little));
    const te = r.find("taps") orelse return error.NoTaps;
    const dr = try dspark.load(gpa, io, &driver, dir, &model, .{ .rank = 0, .world = 1, .group = null, .context = T + dspark.block, .cap = ic.engine.prompt_rows, .lane = &k.attn.lane });
    defer dr.deinit();
    // the embedding table's rows and gain for anchors (host reads)
    const et = model.inv.find("model.llm.embed.weight") orelse return error.NoEmbed;
    const ng = model.inv.find("model.llm.embed_norm.weight") orelse return error.NoEmbed;
    const ep = try std.fs.path.join(gpa, &.{ dir, model.inv.files[et.file].name });
    defer gpa.free(ep);
    const ef = try std.Io.Dir.cwd().openFile(io, ep, .{});
    defer ef.close(io);
    var gain: [H]f32 = undefined;
    {
        const gp = try std.fs.path.join(gpa, &.{ dir, model.inv.files[ng.file].name });
        defer gpa.free(gp);
        const gf = try std.Io.Dir.cwd().openFile(io, gp, .{});
        defer gf.close(io);
        var raw: [H * 2]u8 = undefined;
        if (try gf.readPositionalAll(io, &raw, model.inv.offset(ng)) != raw.len) return error.ShortRead;
        for (&gain, 0..) |*g, j| g.* = ink.st.bf16ToF32(std.mem.readInt(u16, raw[2 * j ..][0..2], .little));
    }
    var gbuf = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(&gain));
    defer gbuf.free();
    var rowbuf = try cuda.DeviceBuffer.alloc(&driver, H * 2);
    defer rowbuf.free();
    // absorb every position from the file's taps, a chunk at a time ([T, 8, H] -> the drafter's [8][cap][H])
    const lo_pos: usize = 0;
    const rows_with_taps: usize = @intCast(te.dim(0)); // a reply ending in EOS has no taps for its last token
    // --all: positions up to the first anchor now, then each anchor's last position before its draft (a local layer's
    // ring holds only the last positions, so a context absorbed far ahead of an anchor would evict its own)
    const hi_pos: usize = if (all) @min(n0, rows_with_taps) else (anchor orelse return error.NoAnchor);
    const host = try gpa.alloc(u8, dspark.ntaps * H * 2);
    defer gpa.free(host);
    var p0 = lo_pos;
    while (p0 < hi_pos) {
        const n = @min(hi_pos - p0, ic.engine.prompt_rows);
        for (0..n) |j| {
            try r.readInto(te, (p0 + j) * dspark.ntaps * H * 2, host);
            for (0..dspark.ntaps) |t| {
                const view: cuda.DeviceBuffer = .{ .d = &driver, .ptr = dr.tap(t) + j * H * 2, .len = H * 2, .borrowed = true };
                try view.upload(0, host[t * H * 2 ..][0 .. H * 2]);
            }
        }
        try dr.absorb(&c, 0, p0, n, 0, 0);
        try stream.synchronize();
        p0 += n;
    }
    if (!all) {
        const p = hi_pos;
        // f_rnn reads the taps' row of p - 1: put it at row 0
        try r.readInto(te, (p - 1) * dspark.ntaps * H * 2, host);
        for (0..dspark.ntaps) |t| {
            const view: cuda.DeviceBuffer = .{ .d = &driver, .ptr = dr.tap(t), .len = H * 2, .borrowed = true };
            try view.upload(0, host[t * H * 2 ..][0 .. H * 2]);
        }
        try embedRow(&c, io, ef, model.inv.offset(et), ids[p], rowbuf, gbuf.ptr, dr.x);
        try dr.draft(&c, 0, p, ids[p], 0);
        var props: [dspark.drafts]u32 = undefined;
        try dr.proposals(&c, &props);
        std.debug.print("proposals", .{});
        for (props) |t| std.debug.print(" {d}", .{t});
        std.debug.print("\ntargets  ", .{});
        for (1..dspark.block) |j| std.debug.print(" {d}", .{if (p + j < T) ids[p + j] else 0});
        std.debug.print("\n", .{});
        var w = ink.st.Writer.init(gpa);
        defer w.deinit();
        try put(gpa, &w, &driver, "block.out", dr.hn, &.{ dspark.block, H }, .bf16);
        try put(gpa, &w, &driver, "base", dr.base, &.{ dspark.drafts, dr.vn }, .bf16);
        try put(gpa, &w, &driver, "frnn", dr.hb, &.{H}, .bf16);
        try put(gpa, &w, &driver, "inv", dr.inv, &.{ dspark.drafts, 3 * dspark.mrank }, .f32);
        try put(gpa, &w, &driver, "proposals", dr.walk + 4, &.{dspark.drafts}, .i32);
        const cs = try dr.bind(0);
        for (0..dspark.nlayers) |l| {
            var nb: [32]u8 = undefined;
            try putCtx(gpa, &w, &driver, try std.fmt.bufPrint(&nb, "ctx{d}.k", .{l}), cs, l, cs.keys(l), p);
            try putCtx(gpa, &w, &driver, try std.fmt.bufPrint(&nb, "ctx{d}.v", .{l}), cs, l, cs.values(l), p);
        }
        try w.write(io, out orelse return error.NoOut);
        return 0;
    }
    // every anchor: the accepted drafts (matching prefix) against the file's tokens
    var hist: [dspark.block]usize = @splat(0);
    var p = n0;
    var total: usize = 0;
    var absorbed = hi_pos;
    while (p + dspark.drafts < T) : (p += 1) {
        try r.readInto(te, (p - 1) * dspark.ntaps * H * 2, host);
        for (0..dspark.ntaps) |t| {
            const view: cuda.DeviceBuffer = .{ .d = &driver, .ptr = dr.tap(t), .len = H * 2, .borrowed = true };
            try view.upload(0, host[t * H * 2 ..][0 .. H * 2]);
        }
        if (p - 1 == absorbed and p - 1 < rows_with_taps) {
            try dr.absorb(&c, 0, p - 1, 1, 0, 0);
            absorbed = p;
        }
        try embedRow(&c, io, ef, model.inv.offset(et), ids[p], rowbuf, gbuf.ptr, dr.x);
        try dr.draft(&c, 0, p, ids[p], 0);
        var props: [dspark.drafts]u32 = undefined;
        try dr.proposals(&c, &props);
        var m: usize = 0;
        while (m < dspark.drafts and props[m] == ids[p + 1 + m]) m += 1;
        hist[m] += 1;
        total += 1;
        std.debug.print("anchor {d}:", .{p});
        for (props) |t| std.debug.print(" {d}", .{t});
        std.debug.print(" | {d}", .{m});
        if (chances) {
            var ch: [dspark.drafts]f32 = undefined;
            try dr.chancesMany(&c, &.{&ch});
            std.debug.print(" |", .{});
            for (ch) |x| std.debug.print(" {d:.5}", .{x});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("anchors {d}, accepted-prefix histogram at 7:", .{total});
    for (hist, 0..) |h, j| std.debug.print(" {d}:{d}", .{ j, h });
    std.debug.print("\n", .{});
    return 0;
}

/// The normed embedding of token `id` (read from the table's file row) into `dst` [1][H] bf16.
fn embedRow(c: *const ops.Ctx, io: std.Io, f: std.Io.File, table: u64, id: u32, row: cuda.DeviceBuffer, gain: u64, dst: u64) !void {
    var raw: [H * 2]u8 = undefined;
    if (try f.readPositionalAll(io, &raw, table + @as(u64, id) * H * 2) != raw.len) return error.ShortRead;
    try c.stream.synchronize();
    try row.upload(0, &raw);
    try c.rms(ops.Mat.of(row.ptr, 1, H, .bf16), gain, ops.Mat.of(dst, 1, H, .bf16), 1e-6);
}

/// Layer l's context keys or values at positions 0 .. p - 1 in position order (a ring's mirror checked equal to its
/// first rows); a local layer's ring holds only the last `ring` of them (the dump's first dimension says how many: `inkling2_dspark_ref.py check` compares the
/// reference's last rows).
fn putCtx(gpa: std.mem.Allocator, w: *ink.st.Writer, d: *const cuda.Driver, name: []const u8, cs: *const dspark.Caches, l: usize, table: u64, p: usize) !void {
    const ring = cs.wrap(l);
    if (ring == 0) return put(gpa, w, d, name, table, &.{ p, cs.row / 2 }, .bf16);
    const n = @min(p, ring);
    const all = try gpa.alloc(u8, cs.rows(l) * cs.row);
    defer gpa.free(all);
    try cuda.memory.download(d, table, all);
    // the mirror: rows ring .. ring + window - 1 repeat rows 0 .. window - 1
    const mirror = (cs.rows(l) - ring) * cs.row;
    if (!std.mem.eql(u8, all[0..mirror], all[ring * cs.row ..][0..mirror])) {
        std.debug.print("FAIL {s}: the ring's mirrored rows differ from their first rows\n", .{name});
        return error.DraftMirror;
    }
    const host = try gpa.alloc(u8, n * cs.row);
    errdefer gpa.free(host);
    for (0..n) |i| {
        const pos = p - n + i;
        @memcpy(host[i * cs.row ..][0..cs.row], all[(pos % ring) * cs.row ..][0..cs.row]);
    }
    try w.addOwned(name, .bf16, &.{ n, cs.row / 2 }, host);
}

fn put(gpa: std.mem.Allocator, w: *ink.st.Writer, d: *const cuda.Driver, name: []const u8, ptr: u64, shape: []const u64, dt: ink.st.DType) !void {
    var n: usize = dt.size();
    for (shape) |x| n *= x;
    const host = try gpa.alloc(u8, n);
    errdefer gpa.free(host);
    try cuda.memory.download(d, ptr, host);
    try w.addOwned(name, dt, shape, host);
}
