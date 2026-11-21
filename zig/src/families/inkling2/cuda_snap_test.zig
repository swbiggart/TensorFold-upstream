//! Host tests (no GPU) of the prompt cache's copies (cuda_snap.zig): the bytes a state takes, the ring's runs in
//! position order, the arena's best-fit extents and what each holds, the cut planner, and the step kinds.
const std = @import("std");
const snap = @import("cuda_snap.zig");
const follow = @import("cuda_follow.zig");
const engine = @import("cuda_engine.zig");

const t = std.testing;

/// Inkling-2-Small at W ranks: 48 layers, every sixth global, 8 / W key heads each.
fn shape(world: u32) [48]snap.LayerShape {
    var s: [48]snap.LayerShape = undefined;
    for (&s, 0..) |*l, i| l.* = .{ .hk = 8 / world, .global = (i + 1) % 6 == 0 };
    return s;
}

fn gibOf(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

test "inkling2 prompt cache on pages: a state's copy at W = 4, plain and with DSpark (the design's slots)" {
    const s = shape(4);
    // the global layers' whole pages are referenced: only the tail past them is copied (8,192 B a position: 8
    // global layers, K and V of 2 heads), beside 40 rings of 1,023 positions and 2.67 MiB of convolutions
    try t.expectEqual(@as(u64, 40 * 1024 * 1023 + 48 * 58368), snap.rawBytes(&s, 0, 2048));
    try t.expectEqual(@as(u64, 8192 * 100 + 40 * 1024 * 1023 + 48 * 58368), snap.rawBytes(&s, 0, 2048 + 100));
    try t.expectEqual(snap.rawBytes(&s, 0, 131072 + 255), snap.rawBytes(&s, 0, 1024 + 255));
    // one slot fits every state: 46 MiB plain, 50 MiB with DSpark (its rings and its global layer's tail)
    try t.expectEqual(@as(u64, 46 << 20), snap.slotBytes(&s, 0));
    try t.expectEqual(@as(u64, 50 << 20), snap.slotBytes(&s, 2));
    for ([_]usize{ 1, 255, 256, 1023, 1024, 2047, 2048, 3039, 131071, 1048575 }) |at| {
        try t.expect(snap.rawBytes(&s, 0, at) <= snap.slotBytes(&s, 0));
        try t.expect(snap.rawBytes(&s, 2, at) <= snap.slotBytes(&s, 2));
        try t.expectEqual(at % 256, snap.tailRows(at));
    }
    // below the windows every sliding layer holds every position, a global layer its tail
    try t.expectEqual(@as(u64, 40 * 1024 * 100 + 8 * 1024 * 100 + 48 * 58368), snap.rawBytes(&s, 0, 100));
    try t.expectEqual(@as(u64, 40 * 1024 * 100 + 8 * 1024 * 100 + 48 * 58368 + 5 * 1024 * 100), snap.rawBytes(&s, 2, 100));
}

test "inkling2 prompt cache: ring positions are copied in position order, wrapped runs and all" {
    var w: snap.Walk = .{ .c = null, .dir = .out, .base = 0 };
    // 3,071 slots; [2,477, 3,500) wraps at 3,071
    try w.rows(0, 3071, 1024, 3500 - 1023, 3500);
    try t.expectEqual(@as(u64, 1023 * 1024), w.off);
    const runs = snap.ringRuns(3071, 3500 - 1023, 3500);
    try t.expectEqual(@as(usize, 2), runs.n);
    try t.expectEqual([2]usize{ 2477, 594 }, .{ runs.r[0].slot, runs.r[0].len });
    try t.expectEqual([2]usize{ 0, 429 }, .{ runs.r[1].slot, runs.r[1].len });
    const one = snap.ringRuns(3071, 100, 1123);
    try t.expectEqual(@as(usize, 1), one.n);
    try t.expectEqual([2]usize{ 100, 1023 }, .{ one.r[0].slot, one.r[0].len });
    // a whole lap ending on the boundary is one run
    const lap = snap.ringRuns(3071, 2048, 3071);
    try t.expectEqual(@as(usize, 1), lap.n);
}

test "inkling2 prompt cache: the arena's extents, best fit with merging (slots of one size never fragment)" {
    const g = snap.granule;
    var x = try snap.Extents.init(t.allocator, 10 * g);
    defer x.deinit(t.allocator);
    const a = x.take(2 * g).?;
    const b = x.take(3 * g).?;
    const c = x.take(1 * g).?;
    try t.expectEqual([3]u64{ 0, 2 * g, 5 * g }, .{ a, b, c });
    try t.expectEqual(@as(u64, 4 * g), x.free());
    try t.expect(x.take(5 * g) == null); // refused: no hole holds it
    x.give(a, 2 * g); // holes: [0, 2) and [6, 10)
    try t.expectEqual(@as(u64, 0), x.take(2 * g).?); // best fit: the 2-granule hole, not the bigger one
    x.give(0, 2 * g);
    x.give(b, 3 * g); // [0, 5) merged
    try t.expectEqual(@as(usize, 2), x.holes.items.len);
    x.give(c, g); // [0, 10) whole again
    try t.expectEqual(@as(usize, 1), x.holes.items.len);
    try t.expectEqual(@as(u64, 10 * g), x.largest());
    // a full arena gives back in any order
    var parts: [10]u64 = undefined;
    for (&parts) |*p| p.* = x.take(g).?;
    try t.expect(x.take(g) == null);
    for ([_]usize{ 3, 7, 0, 9, 5, 1, 8, 2, 6, 4 }) |i| x.give(parts[i], g);
    try t.expectEqual(@as(usize, 1), x.holes.items.len);
    try t.expectEqual(@as(u64, 10 * g), x.free());
}

test "inkling2 prompt cache: a save overwrites what its extent overlapped; a restore checks offset, tag and length" {
    const g = snap.granule;
    var a: snap.Arena = .{ .size = 8 * g };
    defer a.held.deinit(t.allocator);
    try a.held.ensureTotalCapacity(t.allocator, 9);
    a.note(0, 2 * g, 1, 500);
    a.note(2 * g, 2 * g, 2, 700);
    try t.expect(a.holds(0, 1, 500) and a.holds(2 * g, 2, 700));
    try t.expect(!a.holds(0, 2, 500) and !a.holds(0, 1, 501) and !a.holds(g, 1, 500));
    a.note(g, 2 * g, 3, 900); // overlaps both
    try t.expect(!a.holds(0, 1, 500) and !a.holds(2 * g, 2, 700) and a.holds(g, 3, 900));
    try t.expectEqual(@as(usize, 1), a.held.items.len);
}

test "inkling2 prompt cache: the lane's chunks end at every mark and start at the resume point" {
    const marks = [_]u32{ 600, 2048, 2100, 5000 };
    try t.expectEqual(@as(usize, 600), snap.rowsTo(&marks, 0, 2048));
    try t.expectEqual(@as(usize, 1448), snap.rowsTo(&marks, 600, 2048));
    try t.expectEqual(@as(usize, 52), snap.rowsTo(&marks, 2048, 2048));
    try t.expectEqual(@as(usize, 2048), snap.rowsTo(&marks, 2100, 2048));
    try t.expectEqual(@as(usize, 852), snap.rowsTo(&marks, 4148, 2048));
    try t.expectEqual(@as(usize, 7), snap.rowsTo(&marks, 5000, 7));
    try t.expectEqual(@as(usize, 2048), snap.rowsTo(&.{}, 3, 2048));
    // a pass from 0 with these marks: every mark is a chunk start, no chunk is wider than 2,048 rows
    var at: usize = 0;
    var starts: std.ArrayList(usize) = .empty;
    defer starts.deinit(t.allocator);
    while (at < 6000) {
        try starts.append(t.allocator, at);
        at += @min(6000 - at, snap.rowsTo(&marks, at, 2048));
    }
    for (marks) |m| try t.expect(std.mem.indexOfScalar(usize, starts.items, m) != null);
    try t.expectEqualSlices(usize, &.{ 0, 600, 2048, 2100, 4148, 5000 }, starts.items);
}

test "inkling2 prompt cache: the save, restore and arena steps follow the draft step and fit the header" {
    try t.expectEqual(@as(u32, 9), @intFromEnum(follow.Kind.save));
    try t.expectEqual(@as(u32, 10), @intFromEnum(follow.Kind.restore));
    try t.expectEqual(@as(u32, 11), @intFromEnum(follow.Kind.arena));
    try t.expectEqual(@as(u32, 12), @intFromEnum(follow.Kind.move));
    try t.expectEqual(@as(u32, 13), @intFromEnum(follow.Kind.capture));
    const h: engine.Header = .{ .kind = @intFromEnum(follow.Kind.restore), .slot = 3, .rows = 131071, .pos0 = 40 << 30, .token = 0xffff_fffe };
    var back: engine.Header = undefined;
    @memcpy(std.mem.asBytes(&back), std.mem.asBytes(&h));
    try t.expectEqual(h, back);
}

test "a mirrored ring's restore writes every row below the mirror once more, and nothing else" {
    const slots = 3072;
    const mirror = 1024;
    for ([_]usize{ 1, 500, 1023, 1024, 1025, 2047, 3071, 3072, 3073, 4000, 6143, 6144, 9000, 131071 }) |at| {
        const lo = at -| 1024;
        var seen: [slots]u8 = @splat(0);
        var p = lo;
        while (snap.mirrorRun(slots, mirror, p, at)) |m| {
            try t.expect(m.p >= p and m.n > 0 and m.p + m.n <= at);
            for (m.p..m.p + m.n, 0..) |q, i| {
                try t.expectEqual(q % slots, m.row + i);
                seen[q % slots] += 1;
            }
            p = m.p + m.n;
        }
        for (lo..at) |q| try t.expectEqual(@as(u8, if (q % slots < mirror) 1 else 0), seen[q % slots]);
    }
}

test "marks inside a chunk: at most max_caps captured, the chunk ends at the next" {
    var out: [snap.max_caps]u32 = undefined;
    const marks = [_]u32{ 1, 2, 3, 8, 9, 1000, 1023, 1024, 2047, 2048, 2049, 3039 };
    const a = snap.inside(&marks, 0, 2048, &out);
    try t.expectEqual(@as(usize, 8), a.n);
    try t.expectEqual(@as(usize, 2047), a.rows); // the ninth ends the chunk, kept after it
    try t.expectEqualSlices(u32, marks[0..8], out[0..8]);
    const b = snap.inside(&marks, 2047, 993, &out);
    try t.expectEqual(@as(usize, 3), b.n);
    try t.expectEqual(@as(usize, 993), b.rows);
    try t.expectEqualSlices(u32, &.{ 2048, 2049, 3039 }, out[0..3]);
    // a mark on the chunk's end is not inside it; none at its start either
    const c = snap.inside(&.{ 2048, 4096 }, 2048, 2048, &out);
    try t.expectEqual(@as(usize, 0), c.n);
    try t.expectEqual(@as(usize, 2048), c.rows);
}

test "the capture area: every layer's four convolution states, as a state's" {
    const s = shape(4);
    try t.expectEqual(snap.convBytes(&s), snap.rawBytes(&s, 0, 0));
}
