//! tf-inkling2-decode --exp scratch (rank 0; every rank runs the forwards): the op scratch each kind of forward takes,
//! measured (cuda_ops.zig Scratch's high-water mark over the forward) against cuda_scratch.zig's `need` for it, which
//! must hold it. Forwards: on a fresh slot a prompt chunk of up to prompt_rows rows (two when the slot holds them), then
//! on that slot and on the deepest slot so far (after a --depths check: at the last depth) eager forwards of 1, 2, 4,
//! 8, 16 and 64 rows, captured steps of 1 and 8 rows (measured when the forward is captured; a replay takes nothing),
//! and one batched forward of 8 rows a slot over up to four slots. Each forward moves its slots on; run it last.
const std = @import("std");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const scratch = @import("cuda_scratch.zig");
const ops = @import("cuda_ops.zig");

const mib = ops.mib;

pub fn run(gpa: std.mem.Allocator, x0: follow.Leader, seq: []const u32) !bool {
    const e = x0.e;
    var x = x0;
    const cap = e.capacity();
    const ids = try gpa.alloc(u32, engine.prompt_rows);
    defer gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = seq[i % seq.len];
    // the deepest slot so far, and a fresh one
    var deep: ?u32 = null;
    for (e.slots.items, 0..) |s, i| if (s != null and e.pos(@intCast(i)) > 0) {
        if (deep == null or e.pos(@intCast(i)) > e.pos(deep.?)) deep = @intCast(i);
    };
    const fresh: u32 = @intCast(e.slots.items.len);
    const sn = scratch.startupNeed(&e.k, &e.w, cap, engine.prompt_rows, engine.graph_seg_rows);
    std.debug.print("scratch: {d} MiB held; start-up need at {d} positions a slot {d:.1} MiB (attention core {d:.1}, other ops {d:.1})\n", .{ e.scratch.buf.len >> 20, cap, mib(sn.total), mib(sn.core), mib(sn.other) });
    var ok = true;
    var n: usize = 0;
    // a prompt chunk (two when they fit, so the second's attention spans the first)
    var at: usize = 0;
    for (0..2) |_| {
        const R = @min(engine.prompt_rows, cap -| (at + 200));
        if (R < 64) break;
        ok = try one(&x, fresh, at, ids[0..R], false, &n) and ok;
        at += R;
    }
    var slots_buf: [2]u32 = undefined;
    var ns: usize = 0;
    if (deep) |d| if (d != fresh) {
        slots_buf[ns] = d;
        ns += 1;
    };
    if (e.pos(fresh) > 0) {
        slots_buf[ns] = fresh;
        ns += 1;
    }
    for (slots_buf[0..ns]) |s| {
        for ([_]usize{ 1, 2, 4, 8, 16, 64 }) |R| ok = try one(&x, s, e.pos(s), ids[0..R], false, &n) and ok;
        for ([_]usize{ 1, 8 }) |R| ok = try one(&x, s, e.pos(s), ids[0..R], true, &n) and ok;
    }
    // batched: 8 rows a slot over up to four slots past their prompts
    {
        var segs: [4]engine.Segment = undefined;
        var parts: [4]scratch.Part = undefined;
        var k: usize = 0;
        for (e.slots.items, 0..) |s, i| {
            if (k == segs.len) break;
            const p = e.pos(@intCast(i));
            if (s == null or p == 0 or p + 8 > cap) continue;
            segs[k] = .{ .slot = @intCast(i), .pos0 = p, .rows = 8 };
            parts[k] = .{ .pos0 = p, .rows = 8 };
            k += 1;
        }
        if (k >= 2) {
            const need = scratch.need(&e.k, &e.w, parts[0..k]);
            e.scratch.peak = 0;
            e.scratch.kind = .batched;
            try x.forwardMany(segs[0..k], ids[0 .. 8 * k], null);
            ok = report("batched", segs[0].slot, segs[0].pos0, 8 * k, false, e.scratch.peak, need, &n) and ok;
            e.scratch.kind = .other;
        }
    }
    e.scratch.report(0);
    std.debug.print("{s} scratch: {d} forwards, each measured peak within cuda_scratch.need\n", .{ if (ok) "PASS" else "FAIL", n });
    return ok;
}

fn one(x: *follow.Leader, slot: u32, pos0: usize, ids: []const u32, graph: bool, n: *usize) !bool {
    const e = x.e;
    if (pos0 + ids.len > e.capacity()) return true;
    const part: [1]scratch.Part = .{.{ .pos0 = pos0, .rows = ids.len, .positions = if (graph) e.capacity() else 0 }};
    const need = scratch.need(&e.k, &e.w, &part);
    const captured = e.counts.captured;
    e.scratch.peak = 0;
    e.scratch.kind = scratch.kindOf(&part);
    defer e.scratch.kind = .other;
    x.eager = !graph;
    defer x.eager = false;
    try x.forward(slot, pos0, ids, if (graph) .{ .lo = 0, .n = ids.len } else null);
    if (graph and e.counts.captured == captured) {
        std.debug.print("  graph slot {d} pos {d} rows {d}: replayed (no takes); need {d:.2} MiB\n", .{ slot, pos0, ids.len, mib(need) });
        return true;
    }
    return report(@tagName(scratch.kindOf(&part)), slot, pos0, ids.len, graph, e.scratch.peak, need, n);
}

fn report(kind: []const u8, slot: u32, pos0: usize, R: usize, graph: bool, peak: usize, need: usize, n: *usize) bool {
    n.* += 1;
    const ok = peak <= need;
    std.debug.print("  {s} {s} slot {d} pos {d} rows {d}: measured {d:.2} MiB, need {d:.2} MiB{s}\n", .{ if (graph) "graph" else "eager", kind, slot, pos0, R, mib(peak), mib(need), if (!ok) " SHORT" else if (peak == need) " (equal)" else "" });
    return ok;
}
