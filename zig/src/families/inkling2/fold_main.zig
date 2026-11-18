//! tf-inkling2-fold: the DSpark drafter's head folds (inkling2-dspark-fold.safetensors) from a checkpoint folder or a
//! rank folder, on the CPU (fold.zig: fp64 sums in a fixed order, bf16 stores as torch rounds them).
const std = @import("std");
const ink = @import("inkling2");
const fold = ink.fold;

const usage =
    \\usage: tf-inkling2-fold MODEL_DIR OUT [--rank R --world W] [--jobs N]
    \\       tf-inkling2-fold --record FILE LINE
    \\  Writes the DSpark drafter's head folds (fold_w = unembed . hidden_out.weight, fold_b = unembed .
    \\  hidden_out.bias; fp64 sums, bf16 stores) that tf-inkling2 loads from MODEL/inkling2-dspark-fold.safetensors.
    \\  MODEL_DIR holds model.safetensors.index.json, the shard with model.llm.unembed.weight and mtp.safetensors
    \\  (with markov_head.hidden_out.weight and .bias).
    \\  --rank R --world W  only rank R's vocabulary rows [R V/W, (R+1) V/W): a rank folder holds just those unembed
    \\                      rows. The file keeps the full [V, 256] shape (other rows zero; __metadata__ names the
    \\                      rows) and serves that rank only.
    \\  --jobs N            threads (default: every CPU)
    \\  OUT appears complete or not at all: the bytes go to a new temporary file beside it (a random name,
    \\  created exclusively, so an existing file or symlink is never opened), synced, then renamed to OUT.
    \\  --record FILE LINE  publishes LINE and a newline as FILE the same way (fold-rank.sh's digest record).
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len > 1 and std.mem.eql(u8, args[1], "--record")) {
        if (args.len != 4) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        fold.record(gpa, io, args[2], args[3]) catch |e| {
            std.debug.print("FAIL tf-inkling2-fold --record {s}: {t}\n", .{ args[2], e });
            return 1;
        };
        return 0;
    }
    var pos: std.ArrayList([]const u8) = .empty;
    var rank: ?u32 = null;
    var world: ?u32 = null;
    var jobs: usize = std.Thread.getCpuCount() catch 8;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--help") or std.mem.eql(u8, k, "-h")) {
            std.debug.print("{s}", .{usage});
            return 0;
        }
        if (!std.mem.startsWith(u8, k, "--")) {
            try pos.append(a, k);
            continue;
        }
        if (i + 1 >= args.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--rank")) rank = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--world")) {
            world = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--jobs")) {
            jobs = try std.fmt.parseInt(usize, v, 10);
        } else {
            std.debug.print("unknown option {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (pos.items.len != 2 or (rank == null) != (world == null)) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    if (world) |w| if (w == 0 or rank.? >= w) {
        std.debug.print("--rank {d} --world {d}: the rank must be below the world size\n", .{ rank.?, w });
        return 2;
    };
    build(gpa, a, io, pos.items[0], pos.items[1], rank, world, @max(1, jobs)) catch |e| {
        std.debug.print("FAIL tf-inkling2-fold: {t}\n", .{e});
        return 1;
    };
    return 0;
}

fn fail(comptime fmt: []const u8, args: anytype) error{FoldInputs} {
    std.debug.print("tf-inkling2-fold: " ++ fmt ++ "\n", args);
    return error.FoldInputs;
}

/// A bf16 or f32 tensor as fp64.
fn floats(gpa: std.mem.Allocator, a: std.mem.Allocator, r: *const ink.st.Reader, name: []const u8, n: usize) ![]f64 {
    const e = r.find(name) orelse return fail("mtp.safetensors has no {s}", .{name});
    if (e.count() != n) return fail("{s}: {d} elements, expected {d}", .{ name, e.count(), n });
    const raw = try gpa.alloc(u8, @intCast(e.bytes()));
    defer gpa.free(raw);
    try r.readInto(e, 0, raw);
    const out = try a.alloc(f64, n);
    switch (e.dtype) {
        .bf16 => for (out, 0..) |*o, j| {
            o.* = ink.st.bf16ToF32(std.mem.readInt(u16, raw[2 * j ..][0..2], .little));
        },
        .f32 => for (out, 0..) |*o, j| {
            o.* = @as(f32, @bitCast(std.mem.readInt(u32, raw[4 * j ..][0..4], .little)));
        },
        else => return fail("{s}: dtype {s}, expected BF16 or F32", .{ name, e.dtype.name() }),
    }
    var nonzero = false;
    for (out) |x| nonzero = nonzero or x != 0;
    if (!nonzero) return fail("{s} is all zero: this folder's mtp.safetensors does not hold its bytes (fetch the drafter's ranges with hidden_out)", .{name});
    return out;
}

fn build(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, dir: []const u8, out_path: []const u8, rank: ?u32, world: ?u32, jobs: usize) !void {
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(io).toNanoseconds();
    const cwd = std.Io.Dir.cwd();
    // the unembed's shard, from the index
    const index_text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), a, .limited(1 << 26));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, a, index_text, .{});
    const wm = if (index == .object) index.object.get("weight_map") else null;
    const shard_v = if (wm) |m| (if (m == .object) m.object.get(fold.unembed_name) else null) else null;
    const shard = if (shard_v) |s| (if (s == .string) s.string else null) else null;
    if (shard == null) return fail("{s}/model.safetensors.index.json names no shard for {s}", .{ dir, fold.unembed_name });
    var ur = try ink.st.Reader.open(gpa, io, try std.fs.path.join(a, &.{ dir, shard.? }));
    defer ur.close();
    const ue = ur.find(fold.unembed_name) orelse return fail("{s} has no {s}", .{ shard.?, fold.unembed_name });
    if (ue.dtype != .bf16 or ue.rank != 2) return fail("{s}: expected BF16 [V, H]", .{fold.unembed_name});
    const V: usize = @intCast(ue.dim(0));
    const H: usize = @intCast(ue.dim(1));
    var mr = try ink.st.Reader.open(gpa, io, try std.fs.path.join(a, &.{ dir, "mtp.safetensors" }));
    defer mr.close();
    const we = mr.find(fold.weight_name) orelse return fail("mtp.safetensors has no {s}", .{fold.weight_name});
    if (we.rank != 2 or we.dim(0) != H or we.dim(1) != fold.rank_dim) return fail("{s}: expected [{d}, {d}]", .{ fold.weight_name, H, fold.rank_dim });
    const w = try floats(gpa, a, &mr, fold.weight_name, H * fold.rank_dim);
    const b = try floats(gpa, a, &mr, fold.bias_name, H);
    var head = try fold.Head.init(gpa, H, w, b);
    defer head.deinit(gpa);
    var lo: usize = 0;
    var hi: usize = V;
    var meta: std.ArrayList([2][]const u8) = .empty;
    if (world) |W| {
        if (V % W != 0) return fail("vocabulary {d} does not split over {d} ranks", .{ V, W });
        lo = rank.? * (V / W);
        hi = lo + V / W;
        try meta.append(a, .{ "rows", try std.fmt.allocPrint(a, "{d}:{d}", .{ lo, hi }) });
        try meta.append(a, .{ "rank", try std.fmt.allocPrint(a, "{d}", .{rank.?}) });
        try meta.append(a, .{ "world", try std.fmt.allocPrint(a, "{d}", .{W}) });
    }
    const fw = try gpa.alloc(u16, V * fold.rank_dim);
    defer gpa.free(fw);
    const fb = try gpa.alloc(u16, V);
    defer gpa.free(fb);
    @memset(fw, 0);
    @memset(fb, 0);
    const t1 = clock.now(io).toNanoseconds();
    const zero = try fold.run(gpa, io, ur.file, ur.data_start + ue.begin, &head, lo, hi, fw[lo * fold.rank_dim .. hi * fold.rank_dim], fb[lo..hi], jobs);
    const t2 = clock.now(io).toNanoseconds();
    // the checkpoint's unembed has no all-zero row: one here is a hole of a rank folder (another rank's rows, or a copy
    // not finished)
    if (zero > 0) return fail("{d} of unembed rows {d}..{d} are all zero: this folder does not hold them ({s})", .{ zero, lo, hi, if (world == null) "a rank folder: give --rank R --world W" else "wrong --rank, or the copy is not complete" });
    const head_bytes = try fold.header(gpa, V, meta.items);
    defer gpa.free(head_bytes);
    // the one atomic publication: a fresh temporary file created exclusively beside OUT, synced, renamed over OUT
    try fold.publish(gpa, io, out_path, &.{ head_bytes, std.mem.sliceAsBytes(fb), std.mem.sliceAsBytes(fw) });
    const s = struct {
        fn of(x: i96, y: i96) f64 {
            return @as(f64, @floatFromInt(y - x)) * 1e-9;
        }
    }.of;
    std.debug.print("folds rows {d}..{d} of {d} (H {d}), {d} threads: read+fold {d:.2} s, total {d:.2} s -> {s}\n", .{ lo, hi, V, H, jobs, s(t1, t2), s(t0, clock.now(io).toNanoseconds()), out_path });
}
