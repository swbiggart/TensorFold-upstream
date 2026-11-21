//! tf-inkling2-decode --exp argmax (rank 0; B2 C4, greedy argmax on the device):
//!   op check: cuda_head.zig argmax (tf_ink2_head_argmax) against the host's greedy loop (lanes/gpu_rule.zig argmax:
//!     `best = 0; if (v > row[best]) best = i`) on 10,000 rows of the unpadded vocabulary in batches of 1 to 32 rows:
//!     random rows, ties of the largest value (the first must win), a NaN at column 0 (it wins), NaNs elsewhere (they
//!     never win, also where the largest value was), all -inf (0), +inf, -0 against +0, constant rows, the largest
//!     value last, few distinct values, all NaN.
//!   tail A/B: verify windows of 1 and 8 rows on slot 0 (forward + head as a captured graph, keep 1 after each, as
//!     the price table), alternating a head that downloads its fp32 logits and draws on the host (2c's tail) with
//!     one that downloads only the device's argmax rows (Engine.want_logits false), N times a width: medians of each,
//!     their paired difference, and the parts (the download, the host draw loop). Every full head's argmax rows must
//!     equal the host's argmax of its logits.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const head = @import("cuda_head.zig");
const ops = @import("cuda_ops.zig");

/// The host's greedy draw (lanes/gpu_rule.zig argmax, the lanes' `draw` for a greedy row), the op check's reference.
pub fn hostArgmax(values: []const f32) u32 {
    var best: usize = 0;
    for (values, 0..) |v, i| {
        if (v > values[best]) best = i;
    }
    return @intCast(best);
}

const rows_total = 10_000;
const kinds = 13;

/// Row `i` of the op check's rows (kind i % kinds) into `row`.
fn fill(rng: std.Random, row: []f32, i: usize) void {
    const U = row.len;
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    for (row) |*v| v.* = (rng.float(f32) - 0.5) * 60.0; // the softcap's range
    const k = i % kinds;
    switch (k) {
        0 => {}, // random: one largest value (almost surely)
        1 => { // the largest value again at 1-4 other columns, one of them sometimes column 0
            const m = hostArgmax(row);
            for (0..1 + rng.uintLessThan(usize, 4)) |_| row[rng.uintLessThan(usize, U)] = row[m];
            if (rng.boolean()) row[0] = row[m];
        },
        2 => row[0] = nan, // a NaN at column 0 wins
        3 => { // NaNs at ~1 % of the columns, one where the largest value was, never column 0
            const m = hostArgmax(row);
            for (0..U / 100) |_| row[1 + rng.uintLessThan(usize, U - 1)] = nan;
            if (m != 0) row[m] = nan;
        },
        4 => @memset(row, -inf), // all -inf: 0
        5 => { // all -inf with NaNs past column 0: 0
            @memset(row, -inf);
            for (0..16) |_| row[1 + rng.uintLessThan(usize, U - 1)] = nan;
        },
        6 => { // +inf at two columns (the first wins), a NaN between them sometimes
            const a = rng.uintLessThan(usize, U);
            const b = rng.uintLessThan(usize, U);
            row[a] = inf;
            row[b] = inf;
            if (rng.boolean()) row[(a + b) / 2] = nan;
        },
        7 => for (row) |*v| { // -1, -0 and +0 only: the first zero of either sign
            v.* = switch (rng.uintLessThan(u8, 3)) {
                0 => -1.0,
                1 => -0.0,
                else => 0.0,
            };
        },
        8 => @memset(row, 1.25), // constant: 0
        9 => row[U - 1] = 31.0, // the largest value in the last column
        10 => for (row) |*v| { // eight distinct values: many ties
            v.* = @floatFromInt(rng.uintLessThan(u8, 8));
        },
        11 => @memset(row, nan), // all NaN: 0
        12 => { // the largest value a subnormal against zeros, and NaN past it
            for (row) |*v| v.* = if (rng.boolean()) 0.0 else -std.math.floatTrueMin(f32);
            row[rng.uintLessThan(usize, U)] = std.math.floatTrueMin(f32);
            row[1 + rng.uintLessThan(usize, U - 1)] = nan;
        },
        else => unreachable,
    }
}

/// The op check on rank 0 (no other rank moves): every row's device argmax equals the host loop's.
pub fn opCheck(gpa: std.mem.Allocator, e: *engine.Engine) !bool {
    const U = e.vocab();
    const n_most = engine.head_rows;
    const host = try gpa.alloc(f32, n_most * U);
    defer gpa.free(host);
    var buf = try cuda.DeviceBuffer.alloc(e.d, n_most * U * 4 + n_most * 4);
    defer buf.free();
    const out_ptr = buf.ptr + n_most * U * 4;
    var prng = std.Random.DefaultPrng.init(0x1c4_a7a3);
    const rng = prng.random();
    const c = e.opCtx();
    const batches = [_]usize{ 32, 1, 8, 7, 3, 32, 2 };
    var differ: usize = 0;
    var by_kind: [kinds]usize = @splat(0);
    var done: usize = 0;
    var b: usize = 0;
    while (done < rows_total) : (b += 1) {
        const n = @min(batches[b % batches.len], rows_total - done);
        for (0..n) |r| fill(rng, host[r * U ..][0..U], done + r);
        try buf.upload(0, std.mem.sliceAsBytes(host[0 .. n * U]));
        try head.argmax(&c, ops.Mat.of(buf.ptr, n, U, .f32), out_ptr);
        try e.stream.synchronize();
        var got: [engine.head_rows]u32 = undefined;
        try cuda.memory.download(e.d, out_ptr, std.mem.sliceAsBytes(got[0..n]));
        for (0..n) |r| {
            const want = hostArgmax(host[r * U ..][0..U]);
            if (got[r] != want) {
                if (differ < 8) std.debug.print("  argmax row {d} (kind {d}): device {d}, host {d}\n", .{ done + r, (done + r) % kinds, got[r], want });
                differ += 1;
            }
            by_kind[(done + r) % kinds] += 1;
        }
        done += n;
    }
    // the kernel's time on the stream: 1 and 8 rows of the last batch's columns, 200 launches each
    var ev: [2]cuda.Event = .{ try cuda.Event.init(e.d, true), try cuda.Event.init(e.d, true) };
    defer for (&ev) |*x| x.deinit();
    var us: [2]f64 = undefined;
    for ([_]usize{ 1, 8 }, &us) |n, *u| {
        try ev[0].record(e.stream);
        for (0..200) |_| try head.argmax(&c, ops.Mat.of(buf.ptr, n, U, .f32), out_ptr);
        try ev[1].record(e.stream);
        try e.stream.synchronize();
        u.* = @as(f64, try cuda.Event.elapsedMs(ev[0], ev[1])) * 1000 / 200;
    }
    std.debug.print("argmax kernel: {d:.1} us at 1 row, {d:.1} us at 8 rows (200 launches each)\n", .{ us[0], us[1] });
    const pass = differ == 0;
    std.debug.print("{s} argmax op check: {d} rows of {d} columns in {d} launches of 1-32 rows, {d} differ from the host's greedy loop (kinds 0-{d}: random, ties, NaN at 0, NaNs elsewhere, all -inf, -inf with NaNs, +inf, -0/+0, constant, last, few values, all NaN, subnormal; {d} rows each at least)\n", .{ if (pass) "PASS" else "FAIL", done, U, b, differ, kinds - 1, std.mem.min(usize, &by_kind) });
    return pass;
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const m = xs.len / 2;
    return if (xs.len % 2 == 1) xs[m] else (xs[m - 1] + xs[m]) / 2;
}

/// Positions slot 0 takes past its decode for `tail` (a keep of 1 after each window).
pub fn tailPositions(reps: usize) usize {
    return 4 * (reps + 1) + 8;
}

/// The tail A/B (see the header) on slot 0, continuing past its decode (the price table's slot).
pub fn tail(gpa: std.mem.Allocator, x: follow.Leader, seq: []const u32, reps: usize) !bool {
    const e = x.e;
    if (!e.argmax_on) {
        std.debug.print("argmax tail: TF_INK2_GPU_ARGMAX=0, nothing to compare\n", .{});
        return true;
    }
    const U = e.vocab();
    const slot: u32 = 0;
    var pos = e.pos(slot);
    var rows: [8]u32 = undefined;
    for (&rows, 0..) |*r, i| r.* = seq[(pos + i) % seq.len];
    const t = try gpa.alloc(f64, 6 * reps);
    defer gpa.free(t);
    var mism: usize = 0;
    var heads: usize = 0;
    var ok = true;
    for ([_]usize{ 1, 8 }) |R| {
        // warm-up: each kind once (captures the window's graph)
        for (0..2) |_| {
            try x.forward(slot, pos, rows[0..R], .{ .lo = 0, .n = R });
            try x.keep(slot, 1);
            pos += 1;
        }
        const full = t[0..reps];
        const only = t[reps .. 2 * reps];
        const dl = t[2 * reps .. 3 * reps];
        const dl_max = t[3 * reps .. 4 * reps];
        const draw = t[4 * reps .. 5 * reps];
        const diff = t[5 * reps .. 6 * reps];
        for (0..reps) |i| {
            for (0..2) |j| {
                const want = (i + j) % 2 == 0; // alternate which comes first
                const c0 = e.counts;
                e.want_logits = want;
                defer e.want_logits = true;
                const t0 = std.Io.Clock.awake.now(e.io).toNanoseconds();
                try x.forward(slot, pos, rows[0..R], .{ .lo = 0, .n = R });
                var tok: [8]u32 = undefined;
                if (want) {
                    const lg = try e.logits();
                    const d0 = std.Io.Clock.awake.now(e.io).toNanoseconds();
                    for (0..R) |r| tok[r] = hostArgmax(lg[r * U ..][0..U]);
                    draw[i] = ms(@intCast(std.Io.Clock.awake.now(e.io).toNanoseconds() - d0));
                    const dev = try e.argmaxRows();
                    for (0..R) |r| mism += @intFromBool(dev[r] != tok[r]);
                    heads += R;
                } else @memcpy(tok[0..R], (try e.argmaxRows())[0..R]);
                const dt = ms(@intCast(std.Io.Clock.awake.now(e.io).toNanoseconds() - t0));
                const k: usize = @intFromBool(want);
                if (want) full[i] = dt else only[i] = dt;
                const dns = e.counts.head_ns[k] - c0.head_ns[k];
                if (want) dl[i] = ms(dns) else dl_max[i] = ms(dns);
                try x.keep(slot, 1);
                pos += 1;
            }
            diff[i] = full[i] - only[i];
        }
        std.debug.print("argmax tail, {d} row(s), {d} pairs (median ms): forward+head+draw with the logits {d:.3} (download {d:.3}, host draws {d:.3}), argmax rows only {d:.3} (download {d:.3}); saved {d:.3} (paired median)\n", .{ R, reps, median(full), median(dl), median(draw), median(only), median(dl_max), median(diff) });
    }
    ok = mism == 0;
    std.debug.print("{s} argmax tail: the device's argmax equals the host's on every full head's rows ({d} rows, {d} differ)\n", .{ if (ok) "PASS" else "FAIL", heads, mism });
    return ok;
}

test "inkling2 argmax check: the host's greedy loop on the rows the device must match" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    try std.testing.expectEqual(@as(u32, 0), hostArgmax(&.{ nan, 5, 7 })); // a NaN at column 0 wins
    try std.testing.expectEqual(@as(u32, 2), hostArgmax(&.{ 1, nan, 7, nan })); // NaNs elsewhere never do
    try std.testing.expectEqual(@as(u32, 0), hostArgmax(&.{ -inf, -inf, nan })); // all -inf: 0
    try std.testing.expectEqual(@as(u32, 1), hostArgmax(&.{ -1, -0.0, 0.0 })); // -0 equals +0: the first
    try std.testing.expectEqual(@as(u32, 1), hostArgmax(&.{ 3, 9, 9, 9 })); // ties: the first
    var prng = std.Random.DefaultPrng.init(1);
    var row: [64]f32 = undefined;
    for (0..kinds) |k| fill(prng.random(), &row, k); // every kind fills a row
}
