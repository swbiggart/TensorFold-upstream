//! tf-inkling2-decode's batched-decode gate (--streams S) and the state kernels' op check (--op-check).
//!
//! Streams gate: S slots on prompts of different lengths and contents (so different positions), each decoded alone
//! first (greedy, one-row steps: the slot's CUDA graph when it applies), its logits rows hashed. Then the same slots
//! again, stepped together in batched forwards (cuda_engine.zig forwardMany, a segment a slot), fed the solo run's
//! tokens: pass A one row a slot a step; pass B windows of 1 to 3 rows a slot (staggered, as verify windows of
//! different widths, so the set of slots changes as they finish); pass E pass A again with every forward eager (the
//! graphs' paired A/B: a forward of several streams runs as a captured graph from its key's second sighting, so pass
//! A is graphs after its first step). Each pass prints its forwards by kind. With the drafter loaded, pass C: every
//! round each slot keeps its last window's accepted rows and drafts (Leader.draft: the keep, then the block from the
//! taps of its own segment), then all slots' verify windows run in one forward: the anchor and 1 to 7 of the solo
//! run's tokens with one made wrong at a pseudo-random depth (or none), so every slot rolls back on its own after a
//! shared forward (a window at most head_rows / S rows, so 4 at 8 streams). Every slot's rows must equal its solo
//! rows bit for bit (the hashes equal), and in pass C every accepted row too.
//!
//! Op check: the verify window's history commit (tf_ink2_conv_commit_spec) against the three copies it replaced, and
//! the slot rollback (tf_ink2_conv_keep) against a copy a convolution, on random bytes: every byte equal. With the
//! drafter loaded, also its context writes as a captured forward makes them (cuda_dspark.zig absorb with a device
//! position): the keys' rope reading the position on the device and adding it to the table's base, against the host
//! position and address; the values' row kernel against the 2D copy it replaced; positions 0, 1023, 3071 and the
//! table's end - 8, rows 1..8, every byte of the table equal.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const follow = @import("cuda_follow.zig");
const ops = @import("cuda_ops.zig");
const dspark = @import("cuda_dspark.zig");

const Leader = follow.Leader;

/// The slots the gate takes (tf-inkling2-decode uses 0 .. 2).
pub const first_slot: u32 = 3;
/// Positions stream j's prompt is shorter than the next (so the streams sit at different positions).
const stagger = 61;

fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-9;
}

/// Stream j's prompt: `seq` from token j, `T - stagger * j` tokens.
fn promptOf(seq: []const u32, T: usize, j: usize) []const u32 {
    return seq[j..][0 .. T - stagger * j];
}

/// The prompt on `slot` (pos0 0 starts it over) in 2048-row chunks; the last row's logits into `row`.
fn prefill(x: Leader, slot: u32, prompt: []const u32, row: []f32) !void {
    var at: usize = 0;
    while (at < prompt.len) {
        const n = @min(prompt.len - at, engine.prompt_rows);
        const last = at + n == prompt.len;
        try x.forward(slot, at, prompt[at..][0..n], if (last) .{ .lo = n - 1, .n = 1 } else null);
        at += n;
        if (last) @memcpy(row, try x.e.logits());
    }
}

/// The streams gate (see the header): true when every stream's batched rows equal its solo rows in both passes.
pub fn streams(gpa: std.mem.Allocator, io: std.Io, x: Leader, seq: []const u32, T: usize, steps: usize, S: usize) !bool {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const U = e.vocab();
    if (S < 2 or S > engine.max_segs or T <= stagger * S or seq.len < T + S) return error.BadStreams;
    const N = steps; // logits rows a stream: the prompt's last, then N - 1 decode steps
    const solo = try gpa.alloc(f32, S * N * U);
    defer gpa.free(solo);
    const got = try gpa.alloc(f32, S * N * U);
    defer gpa.free(got);
    const toks = try gpa.alloc(u32, S * N);
    defer gpa.free(toks);
    // solo: each stream alone, greedy
    var solo_ns: i96 = 0;
    for (0..S) |j| {
        const slot = first_slot + @as(u32, @intCast(j));
        const p = promptOf(seq, T, j);
        const rows = solo[j * N * U ..][0 .. N * U];
        try prefill(x, slot, p, rows[0..U]);
        const t0 = clock.now(io).toNanoseconds();
        for (0..N) |k| {
            toks[j * N + k] = argmax(rows[k * U ..][0..U]);
            if (k + 1 == N) break;
            try x.forward(slot, p.len + k, toks[j * N + k ..][0..1], .{ .lo = 0, .n = 1 });
            @memcpy(rows[(k + 1) * U ..][0..U], try e.logits());
        }
        solo_ns += clock.now(io).toNanoseconds() - t0;
    }
    var ok = true;
    // passes A, B, then A again with every forward eager (pass E: the graphs' paired A/B in this process)
    var pass_ns: [3]i96 = @splat(0);
    var pass_fwd: [3]usize = @splat(0);
    var pass_med: [3]f64 = @splat(0);
    const fwd_ms = try gpa.alloc(f64, 3 * N + 8);
    defer gpa.free(fwd_ms);
    for (0..3) |pass| {
        var xp = x;
        xp.eager = pass == 2;
        @memset(got, std.math.nan(f32));
        for (0..S) |j| try prefill(x, first_slot + @as(u32, @intCast(j)), promptOf(seq, T, j), got[j * N * U ..][0..U]);
        // done[j]: stream j's rows so far (row 0 from its prompt); row k is fed toks[k - 1] at position len + k - 1
        var done: [engine.max_segs]usize = @splat(1);
        var round: usize = 0;
        const c0 = e.counts;
        const t0 = clock.now(io).toNanoseconds();
        while (true) : (round += 1) {
            var segs: [engine.max_segs]engine.Segment = undefined;
            var who: [engine.max_segs]usize = undefined;
            var ids: [engine.max_segs * 3]u32 = undefined;
            var n: usize = 0;
            var R: usize = 0;
            for (0..S) |j| {
                if (done[j] >= N) continue;
                const width = if (pass != 1) 1 else 1 + (round + j) % 3;
                const w = @min(width, N - done[j]);
                const plen = promptOf(seq, T, j).len;
                segs[n] = .{ .slot = first_slot + @as(u32, @intCast(j)), .pos0 = plen + done[j] - 1, .rows = w };
                @memcpy(ids[R..][0..w], toks[j * N + done[j] - 1 ..][0..w]);
                who[n] = j;
                n += 1;
                R += w;
            }
            if (n == 0) break;
            const tf = clock.now(io).toNanoseconds();
            try xp.forwardMany(segs[0..n], ids[0..R], .{ .lo = 0, .n = R });
            const lg = try e.logits();
            if (pass_fwd[pass] < fwd_ms.len) fwd_ms[pass_fwd[pass]] = secs(clock.now(io).toNanoseconds() - tf) * 1e3;
            var r: usize = 0;
            for (segs[0..n], who[0..n]) |g, j| {
                @memcpy(got[(j * N + done[j]) * U ..][0 .. g.rows * U], lg[r * U ..][0 .. g.rows * U]);
                done[j] += g.rows;
                r += g.rows;
            }
            pass_fwd[pass] += 1;
        }
        pass_ns[pass] = clock.now(io).toNanoseconds() - t0;
        const name: u8 = "ABE"[pass];
        {
            const v = fwd_ms[0..@min(pass_fwd[pass], fwd_ms.len)];
            std.mem.sort(f64, v, {}, std.sort.asc(f64));
            pass_med[pass] = if (v.len > 0) v[v.len / 2] else 0;
        }
        printPassCounts(e, name, c0, pass_fwd[pass]);
        for (0..S) |j| {
            const a = solo[j * N * U ..][0 .. N * U];
            const b = got[j * N * U ..][0 .. N * U];
            var equal: usize = 0;
            for (0..N) |k| equal += @intFromBool(std.mem.eql(u8, std.mem.sliceAsBytes(a[k * U ..][0..U]), std.mem.sliceAsBytes(b[k * U ..][0..U])));
            const ha = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(a));
            const hb = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(b));
            const same = ha == hb and equal == N;
            ok = ok and same;
            std.debug.print("  streams pass {c}, stream {d} (slot {d}, prompt {d}, positions {d} .. {d}): solo xxh64 {x:0>16}, batched {x:0>16}, rows bit-equal {d}/{d} {s}\n", .{ name, j, first_slot + j, promptOf(seq, T, j).len, promptOf(seq, T, j).len - 1, promptOf(seq, T, j).len + N - 2, ha, hb, equal, N, if (same) "EQUAL" else "DIFFER" });
        }
    }
    if (e.drafter != null) ok = try passC(io, x, seq, T, N, S, solo, toks) and ok;
    const solo_ms = secs(solo_ns) * 1e3 / @as(f64, @floatFromInt(S * (N - 1)));
    const a_ms = secs(pass_ns[0]) * 1e3 / @as(f64, @floatFromInt(@max(pass_fwd[0], 1)));
    std.debug.print("{s} streams: {d} streams stepped together equal their solo runs (pass A one row a stream, pass B windows of 1-3 rows, pass E pass A eager); solo {d:.1} ms a step ({d:.1} tok/s), batched {d:.1} ms a step of {d} rows ({d:.1} tok/s aggregate, {d:.1} a stream); pass B {d} forwards in {d:.2} s, {d:.1} ms a forward (median)\n", .{
        if (ok) "PASS" else "FAIL", S, solo_ms, 1e3 / solo_ms, a_ms, S, @as(f64, @floatFromInt(S)) * 1e3 / a_ms, 1e3 / a_ms, pass_fwd[1], secs(pass_ns[1]), pass_med[1],
    });
    std.debug.print("streams step of {d} rows (median, ms): pass A {d:.2} (graphs from the second sighting), pass E {d:.2} (eager): eager - graph {d:.2} ms\n", .{ S, pass_med[0], pass_med[2], pass_med[2] - pass_med[0] });
    return ok;
}

/// The forwards of a pass by kind: graph steps (replays and captures), captures, eager forwards.
fn printPassCounts(e: *const engine.Engine, pass: u8, c0: engine.Counts, n: usize) void {
    const c = e.counts;
    std.debug.print("  streams pass {c}: {d} forwards, {d} as graphs ({d} captured), {d} eager\n", .{ pass, n, c.graph - c0.graph, c.captured - c0.captured, c.eager - c0.eager });
}

/// Pass C (see the header): drafted verify windows of every slot in one forward, each slot's rollback its own.
fn passC(io: std.Io, x: Leader, seq: []const u32, T: usize, N: usize, S: usize, solo: []const f32, toks: []const u32) !bool {
    const e = x.e;
    const U = e.vocab();
    const clock = std.Io.Clock.awake;
    var together_ns: i96 = 0; // the drafts of a round's slots in one pass
    var alone_ns: i96 = 0; // and each slot's again in a pass of its own
    var passes: usize = 0;
    var same_drafts: usize = 0;
    var drafts_n: usize = 0;
    var done: [engine.max_segs]usize = @splat(1); // rows matched (row 0: the prompt's last)
    var keep_n: [engine.max_segs]usize = undefined; // rows of the slot's last forward its next draft keeps
    var row: [engine.max_segs]usize = @splat(0); // rows bit-equal to solo
    var rows_seen: [engine.max_segs]usize = @splat(0);
    var props_ok: usize = 0;
    var props_n: usize = 0;
    var hist: [dspark.block]usize = @splat(0);
    for (0..S) |j| {
        const p = promptOf(seq, T, j);
        const tmp = try std.heap.page_allocator.alloc(f32, U);
        defer std.heap.page_allocator.free(tmp);
        try prefill(x, first_slot + @as(u32, @intCast(j)), p, tmp);
        keep_n[j] = if (p.len % engine.prompt_rows == 0) engine.prompt_rows else p.len % engine.prompt_rows;
    }
    var rng = std.Random.DefaultPrng.init(0xba7c4ed);
    var rounds: usize = 0;
    var fwds: usize = 0;
    const c0 = e.counts;
    // a window's rows at most: the forward's head covers every row (8 at 4 streams; 4 at 8)
    const most = @min(dspark.block, engine.head_rows / S);
    while (true) : (rounds += 1) {
        var segs: [engine.max_segs]engine.Segment = undefined;
        var who: [engine.max_segs]usize = undefined;
        var wins: [engine.max_segs][dspark.block]u32 = undefined;
        var ids: [engine.max_segs * dspark.block]u32 = undefined;
        var n: usize = 0;
        var R: usize = 0;
        // every slot's keep and real block in one drafter pass (Leader.draftMany), then each slot's block again in a
        // pass of its own (Leader.draft; its keep already done): the same drafts, token for token
        var props_all: [engine.max_segs][dspark.drafts]u32 = undefined;
        {
            var items: [engine.max_segs]engine.DraftItem = undefined;
            var keeps: [engine.max_segs]usize = undefined;
            var outs: [engine.max_segs]*[dspark.drafts]u32 = undefined;
            var m: usize = 0;
            for (0..S) |j| {
                if (done[j] >= N) continue;
                items[m] = .{ .slot = first_slot + @as(u32, @intCast(j)), .anchor = toks[j * N + done[j] - 1] };
                keeps[m] = keep_n[j];
                outs[m] = &props_all[j];
                m += 1;
            }
            if (m == 0) break;
            var t0 = clock.now(io).toNanoseconds();
            try x.draftMany(items[0..m], keeps[0..m], outs[0..m]);
            together_ns += clock.now(io).toNanoseconds() - t0;
            t0 = clock.now(io).toNanoseconds();
            for (items[0..m], keeps[0..m], outs[0..m]) |it, k, o| {
                var alone: [dspark.drafts]u32 = undefined;
                try x.draft(it.slot, k, it.anchor, &alone);
                same_drafts += @intFromBool(std.mem.eql(u32, &alone, o));
                drafts_n += 1;
            }
            alone_ns += clock.now(io).toNanoseconds() - t0;
            passes += 1;
        }
        for (0..S) |j| {
            if (done[j] >= N) continue;
            const slot = first_slot + @as(u32, @intCast(j));
            const anchor = toks[j * N + done[j] - 1];
            const props = props_all[j];
            const D = @min(1 + (rounds + j) % dspark.drafts, N - done[j] - 1, most - 1);
            const wrong = rng.random().uintAtMost(usize, D); // D: none wrong
            for (0..D) |i| {
                const truth = toks[j * N + done[j] + i];
                props_ok += @intFromBool(props[i] == truth);
                props_n += 1;
                wins[n][1 + i] = if (i == wrong) (truth + 1) % @as(u32, @intCast(U)) else truth;
            }
            wins[n][0] = anchor;
            const plen = promptOf(seq, T, j).len;
            segs[n] = .{ .slot = slot, .pos0 = plen + done[j] - 1, .rows = D + 1 };
            @memcpy(ids[R..][0 .. D + 1], wins[n][0 .. D + 1]);
            who[n] = j;
            n += 1;
            R += D + 1;
        }
        if (n == 0) break;
        try x.forwardMany(segs[0..n], ids[0..R], .{ .lo = 0, .n = R });
        fwds += 1;
        const lg = try e.logits();
        var r: usize = 0;
        for (segs[0..n], who[0..n], wins[0..n]) |g, j, w| {
            const D = g.rows - 1;
            var k: usize = 0;
            while (k < D and argmax(lg[(r + k) * U ..][0..U]) == w[1 + k]) k += 1;
            // rows 0 .. k hold the solo run's positions done .. done + k
            for (0..k + 1) |i| {
                const a = solo[(j * N + done[j] + i) * U ..][0..U];
                const b = lg[(r + i) * U ..][0..U];
                row[j] += @intFromBool(std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b)));
                rows_seen[j] += 1;
            }
            hist[k] += 1;
            done[j] += k + 1;
            keep_n[j] = k + 1;
            r += g.rows;
        }
    }
    printPassCounts(e, 'C', c0, fwds);
    var ok = same_drafts == drafts_n;
    std.debug.print("{s} streams pass C drafts: every round's slots drafted in one pass equal their drafts alone {d}/{d}; {d:.2} ms a round's pass ({d:.2} streams on average) against {d:.2} ms a round drafting them one by one\n", .{ if (ok) "PASS" else "FAIL", same_drafts, drafts_n, secs(together_ns) * 1e3 / @as(f64, @floatFromInt(@max(passes, 1))), @as(f64, @floatFromInt(drafts_n)) / @as(f64, @floatFromInt(@max(passes, 1))), secs(alone_ns) * 1e3 / @as(f64, @floatFromInt(@max(passes, 1))) });
    for (0..S) |j| {
        try x.keep(first_slot + @as(u32, @intCast(j)), keep_n[j]);
        const same = row[j] == rows_seen[j] and rows_seen[j] == N - 1;
        ok = ok and same;
        std.debug.print("  streams pass C, stream {d}: accepted rows bit-equal to solo {d}/{d} {s}\n", .{ j, row[j], rows_seen[j], if (same) "EQUAL" else "DIFFER" });
    }
    std.debug.print("{s} streams pass C: {d} streams' drafted windows (mixed, 1-7 drafts) verified together and rolled back each on its own; {d} rounds; accepted drafts", .{ if (ok) "PASS" else "FAIL", S, rounds });
    for (hist, 0..) |h, i| std.debug.print(" {d}:{d}", .{ i, h });
    std.debug.print("; the real drafter's proposals equal the solo tokens {d}/{d}\n", .{ props_ok, props_n });
    return ok;
}

/// The state kernels' op check (see the header), on this rank alone: true when every byte is equal.
pub fn opCheck(gpa: std.mem.Allocator, e: *engine.Engine) !bool {
    const c = e.opCtx();
    var rng = std.Random.DefaultPrng.init(0x0b5e55ed);
    const cap_c: usize = 4608;
    const hist_bytes = (3 + state.spec_rows) * cap_c * 2;
    const host = try gpa.alloc(u8, 4 * hist_bytes);
    defer gpa.free(host);
    const back = try gpa.alloc(u8, 4 * hist_bytes);
    defer gpa.free(back);
    var bufs: [7]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (&bufs) |*b| {
        b.* = try cuda.DeviceBuffer.alloc(e.d, hist_bytes);
        made += 1;
    }
    // commit: state A/B (bufs 0, 1), hist A/B (2, 3), x (4)
    var cases: usize = 0;
    var bad: usize = 0;
    for ([_]usize{ 256, 512, 4608 }) |C| for (1..state.spec_rows + 1) |R| {
        rng.random().bytes(host);
        for (bufs[0..5], 0..) |b, i| try b.upload(0, host[i * (hist_bytes / 2) ..][0 .. hist_bytes / 2]);
        try bufs[1].upload(0, host[0 .. hist_bytes / 2]);
        try bufs[3].upload(0, host[2 * (hist_bytes / 2) ..][0 .. hist_bytes / 2]);
        const x = ops.Mat.of(bufs[4].ptr, R, C, .bf16);
        try state.convCommitSpecCopies(&c, bufs[0].ptr, bufs[2].ptr, x);
        try state.convCommitSpec(&c, bufs[1].ptr, bufs[3].ptr, x);
        try c.stream.synchronize();
        for ([2][2]usize{ .{ 0, 1 }, .{ 2, 3 } }) |pair| {
            try bufs[pair[0]].download(0, back[0 .. hist_bytes / 2]);
            try bufs[pair[1]].download(0, back[hist_bytes .. hist_bytes + hist_bytes / 2]);
            if (!std.mem.eql(u8, back[0 .. hist_bytes / 2], back[hist_bytes .. hist_bytes + hist_bytes / 2])) bad += 1;
        }
        cases += 1;
    };
    // keep: 4 convolutions (256, 256, 4608, 4608 columns) with their states A (bufs 0, 1) and B (2, 3), two a
    // buffer, and a history each (the narrow two in 4, the wide in 5 and 6), every keep 1 .. 7
    var keep_cases: usize = 0;
    for (1..state.spec_rows) |n| {
        rng.random().bytes(host);
        for (bufs, 0..) |b, i| try b.upload(0, host[(i % 4) * (hist_bytes / 2) ..][0..hist_bytes]);
        const cols = [4]u32{ 256, 256, 4608, 4608 };
        const half = hist_bytes / 2;
        const st_a = [4]u64{ bufs[0].ptr, bufs[0].ptr + half, bufs[1].ptr, bufs[1].ptr + half };
        const st_b = [4]u64{ bufs[2].ptr, bufs[2].ptr + half, bufs[3].ptr, bufs[3].ptr + half };
        const hs = [4]u64{ bufs[4].ptr, bufs[4].ptr + half, bufs[5].ptr, bufs[6].ptr };
        // B starts as A
        try c.copy(bufs[2].ptr, bufs[0].ptr, hist_bytes);
        try c.copy(bufs[3].ptr, bufs[1].ptr, hist_bytes);
        var table: [4]state.KeepEntry = undefined;
        for (&table, st_b, hs, cols) |*t, s, h, k| t.* = .{ .state = s, .hist = h, .cols = k };
        var tb = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(&table));
        defer tb.free();
        for (st_a, hs, cols) |s, h, k| try state.convKeep(&c, s, h, k, n);
        var st: state.State = .{ .gpa = gpa, .first = 0, .layers = &.{}, .context = 0, .keep_table = tb.ptr, .keep_entries = 4 };
        try state.keepAll(&c, &st, n);
        try c.stream.synchronize();
        for ([2][2]usize{ .{ 0, 2 }, .{ 1, 3 } }) |pair| {
            try bufs[pair[0]].download(0, back[0..hist_bytes]);
            try bufs[pair[1]].download(0, back[hist_bytes .. 2 * hist_bytes]);
            if (!std.mem.eql(u8, back[0..hist_bytes], back[hist_bytes .. 2 * hist_bytes])) bad += 1;
        }
        keep_cases += 1;
    }
    std.debug.print("{s} op check: window history commit (tf_ink2_conv_commit_spec) against its three copies, {d} cases (columns 256, 512, 4608; rows 1..8); slot rollback (tf_ink2_conv_keep) against a copy a convolution, {d} cases (keep 1..7, 4 convolutions); {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", cases, keep_cases, bad });
    var ok = bad == 0;
    if (e.drafter) |dr| ok = try drafterCheck(gpa, e, dr) and ok;
    return ok;
}

/// The drafter's context writes with a device position against the host forms (see the header).
fn drafterCheck(gpa: std.mem.Allocator, e: *engine.Engine, dr: *const dspark.Drafter) !bool {
    const c = e.opCtx();
    var rng = std.Random.DefaultPrng.init(0xd5a7c0de);
    const D = 128;
    const kvw = dr.hk * D; // a row of keys (or values), elements
    const row = kvw * 2;
    const cap = e.capacity();
    if (cap < 3071 + 8 + 8) return error.OpCheckContextShort;
    const R_max = state.spec_rows;
    // x: R_max rows of [k | v] (2 kvw bf16 a row, finite values); gain fp32 [128]
    const xh = try gpa.alloc(u16, R_max * 2 * kvw);
    defer gpa.free(xh);
    for (xh) |*v| v.* = @truncate(@as(u32, @bitCast(rng.random().float(f32) * 4 - 2)) >> 16);
    var gh: [D]f32 = undefined;
    for (&gh) |*g| g.* = 0.5 + rng.random().float(f32);
    var xb = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(xh));
    defer xb.free();
    var gb = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(&gh));
    defer gb.free();
    var pb = try cuda.DeviceBuffer.alloc(e.d, 256);
    defer pb.free();
    // two tables of `cap` rows: A by the host forms, B by the device forms
    var ta = try cuda.DeviceBuffer.alloc(e.d, cap * row);
    defer ta.free();
    var tb = try cuda.DeviceBuffer.alloc(e.d, cap * row);
    defer tb.free();
    const a_host = try gpa.alloc(u8, cap * row);
    defer gpa.free(a_host);
    const b_host = try gpa.alloc(u8, cap * row);
    defer gpa.free(b_host);
    var cases: usize = 0;
    var bad: usize = 0;
    for ([_]usize{ 0, 1023, 3071, cap - 8 }) |pos| for (1..R_max + 1) |R| for ([_]bool{ false, true }) |values| {
        try ta.fill8(0, null);
        try tb.fill8(0, null);
        const p32 = [1]i32{@intCast(pos)};
        try pb.upload(0, std.mem.sliceAsBytes(&p32));
        if (values) {
            try dspark.copy2d(&c, ta.ptr + pos * row, row, xb.ptr + row, 2 * row, row, R);
            try dr.rowsAt(&c, tb.ptr, row, xb.ptr + row, 2 * row, row, R, 0, pb.ptr, 0);
        } else {
            try dr.rope(&c, xb.ptr, 2 * kvw, dr.hk, R, gb.ptr, ta.ptr + pos * row, kvw, pos, 0, 0);
            try dr.rope(&c, xb.ptr, 2 * kvw, dr.hk, R, gb.ptr, tb.ptr, kvw, 0, pb.ptr, 0);
        }
        try c.stream.synchronize();
        try ta.download(0, a_host);
        try tb.download(0, b_host);
        if (!std.mem.eql(u8, a_host, b_host)) bad += 1;
        cases += 1;
    };
    // the local layers' mirrored rings: rows at (pos0 + r) % ring (and below the window again `ring` rows on) with
    // pos0 from the host and from the device, against each row written alone at its rows (host position, plain
    // address)
    const ring = dspark.window + dr.cap; // dspark.ringRows of a slot past it
    const rrows = ring + dspark.window; // dspark.localRows
    var ra = try cuda.DeviceBuffer.alloc(e.d, rrows * row);
    defer ra.free();
    var rb = try cuda.DeviceBuffer.alloc(e.d, rrows * row);
    defer rb.free();
    const ra_host = try gpa.alloc(u8, rrows * row);
    defer gpa.free(ra_host);
    const rb_host = try gpa.alloc(u8, rrows * row);
    defer gpa.free(rb_host);
    var ring_cases: usize = 0;
    for ([_]usize{ 0, dspark.window - 3, ring - 3, ring, 5 * ring + 1000, 131_000 }) |pos| for (1..R_max + 1) |R| for ([_]bool{ false, true }) |values| for ([_]bool{ false, true }) |on_dev| {
        try ra.fill8(0, null);
        try rb.fill8(0, null);
        const p32 = [1]i32{@intCast(pos)};
        try pb.upload(0, std.mem.sliceAsBytes(&p32));
        for (0..R) |r| {
            const rr = (pos + r) % ring;
            for ([_]usize{ rr, rr + ring }) |at_row| {
                if (at_row >= ring and rr >= dspark.window) continue;
                const at = at_row * row;
                if (values) {
                    try dspark.copy2d(&c, ra.ptr + at, row, xb.ptr + r * 2 * row + row, 2 * row, row, 1);
                } else try dr.rope(&c, xb.ptr + r * 2 * row, 2 * kvw, dr.hk, 1, gb.ptr, ra.ptr + at, kvw, pos + r, 0, 0);
            }
        }
        const p0: usize = if (on_dev) 0 else pos;
        const pp: u64 = if (on_dev) pb.ptr else 0;
        if (values) {
            try dr.rowsAt(&c, rb.ptr, row, xb.ptr + row, 2 * row, row, R, p0, pp, ring);
        } else try dr.rope(&c, xb.ptr, 2 * kvw, dr.hk, R, gb.ptr, rb.ptr, kvw, p0, pp, ring);
        try c.stream.synchronize();
        try ra.download(0, ra_host);
        try rb.download(0, rb_host);
        if (!std.mem.eql(u8, ra_host, rb_host)) bad += 1;
        ring_cases += 1;
    };
    std.debug.print("{s} op check: the drafter's context writes with a device position (rope adding the position to the table's base; the values' row kernel) against the host position and address (rope; the 2D copy), {d} cases (positions 0, 1023, 3071, {d}; rows 1..8); the local layers' mirrored rings of {d} positions in {d} rows (host and device positions; positions 0, {d}, {d}, {d}, {d}, 131000) against rows written one by one, {d} cases; {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", cases, cap - 8, ring, rrows, dspark.window - 3, ring - 3, ring, 5 * ring + 1000, ring_cases, bad });
    return bad == 0;
}

/// Total rows the shared price measures (`sharedPrice`): 1 .. 8, then up to the lane core's 32 a round.
pub const shared_sizes = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32 };

/// `--shared-grid`: forwards of S streams of r rows each (S x r rows in all), the shapes the lane core's shared rounds
/// take at 2, 4 and 8 streams (shared_sizes measures only the fewest streams of a row count).
pub const shared_grid = [_][2]usize{ .{ 2, 1 }, .{ 2, 4 }, .{ 2, 8 }, .{ 4, 1 }, .{ 4, 2 }, .{ 4, 4 }, .{ 4, 8 }, .{ 8, 1 }, .{ 8, 2 }, .{ 8, 3 }, .{ 8, 4 } };
/// The most streams a shared price forward takes: 4 (shared_sizes), 8 with the grid.
pub fn sharedStreams(grid: bool) usize {
    return if (grid) 8 else 4;
}

/// Positions a slot of the shared price takes past its prompt (a keep of one row after each forward).
pub fn sharedPositions(reps: usize, grid: bool) usize {
    return (shared_sizes.len + if (grid) shared_grid.len else 0) * (reps + 2);
}

/// The shared price (tf-inkling2-decode --exp shared-price, the drafter loaded): forwards of R rows in all as the
/// fewest streams of at most 8 rows each, at least two (one row: one stream), rows split evenly, on slots 3 ..
/// at the gate's staggered prompts, with every row's head logits; each segment keeps one row after, so the slots
/// stay at their positions. `reps` forwards a size after two not timed (a key of several streams runs eagerly at its
/// first sighting and is captured at its second; TF_INK2_GRAPH=0 or TF_INK2_GRAPH_SEGS=1: all eager). Prints each
/// size's median and mean ms from the call to the logits on the host, and the forwards of each kind: the lane core's
/// shared_costs (cuda_lanes.zig shared_ms). `grid` (--shared-grid 1): then `shared_grid`'s cells, S streams of r rows
/// each, on slots 3 .. 10 (8 prompts), printed as "grid S r median mean".
pub fn sharedPrice(io: std.Io, x: Leader, seq: []const u32, T: usize, reps: usize, grid: bool) !void {
    const e = x.e;
    if (e.drafter == null) return error.SharedPriceNeedsTheDrafter; // the rollback (keep) needs the window history
    const clock = std.Io.Clock.awake;
    const S_most = 8;
    const S_used = sharedStreams(grid);
    if (T <= stagger * S_used or reps == 0) return error.BadStreams;
    const tmp = try std.heap.page_allocator.alloc(f32, e.vocab());
    defer std.heap.page_allocator.free(tmp);
    for (0..S_used) |j| try prefill(x, first_slot + @as(u32, @intCast(j)), promptOf(seq, T, j), tmp);
    const t = try std.heap.page_allocator.alloc(f64, reps);
    defer std.heap.page_allocator.free(t);
    std.debug.print("shared price (ms, {d} forwards a size, slots {d}.., prompts {d} .. {d}): rows streams median mean\n", .{ reps, first_slot, T - stagger * (S_used - 1), T });
    const cells = shared_sizes.len + if (grid) shared_grid.len else 0;
    for (0..cells) |cell| {
        const in_grid = cell >= shared_sizes.len;
        const R: usize = if (in_grid) shared_grid[cell - shared_sizes.len][0] * shared_grid[cell - shared_sizes.len][1] else shared_sizes[cell];
        const S: usize = if (in_grid) shared_grid[cell - shared_sizes.len][0] else if (R == 1) 1 else @max(2, (R + 7) / 8);
        const c0 = e.counts;
        for (0..reps + 2) |i| {
            var segs: [S_most]engine.Segment = undefined;
            var ids: [32]u32 = undefined;
            var r: usize = 0;
            for (segs[0..S], 0..) |*g, j| {
                const slot = first_slot + @as(u32, @intCast(j));
                const rows = R / S + @intFromBool(j < R % S);
                g.* = .{ .slot = slot, .pos0 = e.pos(slot), .rows = rows };
                for (ids[r..][0..rows], 0..) |*v, k| v.* = seq[(g.pos0 + k) % T];
                r += rows;
            }
            const t0 = clock.now(io).toNanoseconds();
            try x.forwardMany(segs[0..S], ids[0..R], .{ .lo = 0, .n = R });
            _ = try e.logits();
            const ms = secs(clock.now(io).toNanoseconds() - t0) * 1e3;
            for (segs[0..S]) |g| try x.keep(g.slot, 1);
            if (i > 1) t[i - 2] = ms;
        }
        var sum: f64 = 0;
        for (t) |v| sum += v;
        std.mem.sort(f64, t, {}, std.sort.asc(f64));
        const med = if (reps % 2 == 1) t[reps / 2] else (t[reps / 2 - 1] + t[reps / 2]) / 2;
        const c = e.counts;
        if (in_grid and cell == shared_sizes.len) std.debug.print("shared price grid (ms, {d} forwards a cell, slots {d}..{d}): grid streams rows-a-stream median mean\n", .{ reps, first_slot, first_slot + S_most - 1 });
        if (in_grid) {
            std.debug.print("  grid {d} {d} {d:.2} {d:.2} (graphs {d}, captured {d}, eager {d})\n", .{ S, R / S, med, sum / @as(f64, @floatFromInt(reps)), c.graph - c0.graph, c.captured - c0.captured, c.eager - c0.eager });
        } else std.debug.print("  {d} {d} {d:.2} {d:.2} (graphs {d}, captured {d}, eager {d})\n", .{ R, S, med, sum / @as(f64, @floatFromInt(reps)), c.graph - c0.graph, c.captured - c0.captured, c.eager - c0.eager });
    }
}

/// Chunk invariance (the interleave design's G1, tf-inkling2-decode --exp chunks): the cut lists a prompt of T
/// tokens is prefilled with on slot `chunks_slot`, each followed by the check's decode (S - 1 greedy one-row steps);
/// a list's sizes are taken in order, the last one repeated, each clamped to the rows left, to `prompt_rows` and to
/// the forced cuts. T up to 4,096 (the check's 3,040): `short_lists` and three seeded random lists (16 .. 1,024 rows);
/// longer: 256-row chunks and one seeded random list with cuts forced at 20,000 and T - 1. Every list must give the
/// check's decode logits (its xxh64, the prompt's last row included): a prompt's state does not depend on its cuts.
const short_lists = [_][]const usize{ &.{2048}, &.{512}, &.{256}, &.{128}, &.{64}, &.{37}, &.{ 1, 2, 3, 8, 9, 1000 }, &.{ 1023, 1, 2047 }, &.{ 2047, 2, 2048 } };
const long_lists = [_][]const usize{&.{256}};
const random_lists_short: usize = 3;
const random_lists_long: usize = 1;
/// The slot `chunks` and `mixPrice` take (tf-inkling2-decode's draft gates start it over too).
pub const chunks_slot: u32 = 2;

/// The chunks experiment (see `short_lists`): true when every list's decode logits hash to `ref` (the check's) and
/// every row equals the first list's. Without `ref` (no check before it, or several depths) the hashes print only.
pub fn chunks(gpa: std.mem.Allocator, io: std.Io, x: Leader, seq: []const u32, T: usize, S: usize, ref: ?u64) !bool {
    const e = x.e;
    const U = e.vocab();
    const clock = std.Io.Clock.awake;
    const long = T > 4096;
    const fixed: []const []const usize = if (long) &long_lists else &short_lists;
    const lists = fixed.len + if (long) random_lists_long else random_lists_short;
    const first = try gpa.alloc(f32, S * U);
    defer gpa.free(first);
    const rows = try gpa.alloc(f32, S * U);
    defer gpa.free(rows);
    var forced_buf: [2]usize = undefined;
    var nf: usize = 0;
    if (long) {
        if (T > 20000 + 1) {
            forced_buf[nf] = 20000;
            nf += 1;
        }
        forced_buf[nf] = T - 1;
        nf += 1;
    }
    const forced = forced_buf[0..nf];
    var ok = true;
    std.debug.print("chunks: prompt {d} on slot {d} in {d} cut lists, each then {d} decode steps; reference {s}\n", .{ T, chunks_slot, lists, S - 1, if (ref != null) "the check's decode logits xxh64" else "none (hashes only)" });
    for (0..lists) |li| {
        var rng = std.Random.DefaultPrng.init(0xc0751de + li);
        const list: ?[]const usize = if (li < fixed.len) fixed[li] else null;
        var at: usize = 0;
        var nfwd: usize = 0;
        var lo: usize = std.math.maxInt(usize);
        var hi: usize = 0;
        const t0 = clock.now(io).toNanoseconds();
        while (at < T) : (nfwd += 1) {
            var n = if (list) |l| l[@min(nfwd, l.len - 1)] else rng.random().intRangeAtMost(usize, 16, 1024);
            n = @min(n, T - at, engine.prompt_rows);
            for (forced) |c| if (at < c and at + n > c) {
                n = c - at;
            };
            const last = at + n == T;
            try x.forward(chunks_slot, at, seq[at..][0..n], if (last) .{ .lo = n - 1, .n = 1 } else null);
            at += n;
            lo = @min(lo, n);
            hi = @max(hi, n);
            if (last) @memcpy(rows[0..U], try e.logits());
        }
        const pre = secs(clock.now(io).toNanoseconds() - t0);
        var toks_same: usize = 0;
        for (0..S) |k| {
            const t = argmax(rows[k * U ..][0..U]);
            toks_same += @intFromBool(seq.len >= T + S and t == seq[T + k]);
            if (k + 1 == S) break;
            try x.forward(chunks_slot, T + k, &.{t}, .{ .lo = 0, .n = 1 });
            @memcpy(rows[(k + 1) * U ..][0..U], try e.logits());
        }
        if (li == 0) @memcpy(first, rows);
        var equal: usize = 0;
        var first_diff: ?usize = null;
        for (0..S) |k| {
            const same_row = std.mem.eql(u8, std.mem.sliceAsBytes(first[k * U ..][0..U]), std.mem.sliceAsBytes(rows[k * U ..][0..U]));
            equal += @intFromBool(same_row);
            if (!same_row and first_diff == null) first_diff = k;
        }
        const h = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(rows));
        const same = (if (ref) |r| h == r else true) and equal == S;
        ok = ok and same;
        std.debug.print("  chunks list {d} ", .{li});
        if (list) |l| {
            std.debug.print("{{", .{});
            for (l, 0..) |v, i| std.debug.print("{s}{d}", .{ if (i > 0) ", " else "", v });
            std.debug.print("}}", .{});
        } else std.debug.print("random (seed {d}, 16..1024{s})", .{ li, if (forced.len > 0) ", cuts forced" else "" });
        std.debug.print(": {d} forwards of {d} .. {d} rows, prefill {d:.2} s; decode logits xxh64 {x:0>16}{s}; rows bit-equal to list 0's {d}/{d}", .{ nfwd, lo, hi, pre, h, if (ref) |r| (if (h == r) " EQUAL to the check's" else " DIFFERS from the check's") else "", equal, S });
        if (first_diff) |k| std.debug.print(" (first difference at row {d}, position {d})", .{ k, T - 1 + k });
        if (seq.len >= T + S and ref != null) std.debug.print("; tokens equal {d}/{d}", .{ toks_same, S });
        std.debug.print(" {s}\n", .{if (same) "EQUAL" else "DIFFER"});
    }
    std.debug.print("{s} chunks: {d} cut lists of a {d}-token prompt give the same decode logits{s}\n", .{ if (ok) "PASS" else "FAIL", lists, T, if (ref != null) " as the check" else "" });
    return ok;
}

/// The chunk sizes `mixPrice` measures.
pub const mix_ks = [_]usize{ 64, 128, 256, 512, 1024 };
/// Decode streams `mixPrice` takes at most (slots first_slot + 1 ..).
pub const mix_dec_most = 4;
/// A decode stream's prompt in `mixPrice`: 512 + 61 j tokens from token j.
fn mixDecPrompt(seq: []const u32, j: usize) []const u32 {
    return seq[j..][0 .. 512 + stagger * j];
}

/// Positions `mixPrice` takes on `chunks_slot` (each depth band from where the slot stands, at least its depth; a
/// band of depth 0 restarts the slot for every chunk).
pub fn mixPositions(reps: usize, depths: []const usize, decs: []const usize) usize {
    var sum_k: usize = 0;
    for (mix_ks) |k| sum_k += k;
    const band = (reps * sum_k * 2 + 4 * mix_ks[0]) * decs.len; // A and B a pair, after two warm-up pairs at the smallest k
    var pos: usize = 0;
    var top: usize = 0;
    for (depths) |d| {
        if (d == 0) {
            top = @max(top, mix_ks[mix_ks.len - 1]);
            continue;
        }
        pos = @max(pos, d) + band;
        top = @max(top, pos);
    }
    return top + 64;
}

/// The mixed step's price (the interleave design's S0 `mixprice`, tf-inkling2-decode --exp mixprice): on
/// `chunks_slot`, prompt chunks of k rows (`mix_ks`) at each depth band (`depths`: 0 restarts the slot for each
/// chunk; above, the slot is prefilled to the depth, then its chunks go on from there), beside R decode streams
/// (`decs`, one row each, slots first_slot + 1 ..), as pairs in one process, which goes first alternating:
///   A: the chunk alone (eager, no head; to its end on the device), then the decoders' forward with their head
///      (one stream: a captured step; several: a captured forward from its key's second sighting);
///   B: one forward of [the chunk, the decoders' rows] (eager), the head over the decode rows (the fold's step).
/// Each cell prints the medians over `reps` of the chunk (C), the decode (D), the mixed step (M), the fold's saving
/// (C + D - M, the median of the pairs' savings) and its share of C + D; the bits are not checked (a price).
pub fn mixPrice(io: std.Io, x: Leader, seq: []const u32, T: usize, reps: usize, depths: []const usize, decs: []const usize) !void {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    var most: usize = 0;
    for (decs) |r| {
        if (r == 0) return error.BadMixPrice; // the chunk alone is every pair's C
        most = @max(most, r);
    }
    if (reps == 0 or reps > 16 or most > mix_dec_most or T < 512 + stagger * mix_dec_most + mix_dec_most) return error.BadMixPrice;
    // the decode streams' prompts
    for (0..most) |j| {
        const p = mixDecPrompt(seq, j);
        var at: usize = 0;
        while (at < p.len) {
            const n = @min(p.len - at, engine.prompt_rows);
            try x.forward(first_slot + 1 + @as(u32, @intCast(j)), at, p[at..][0..n], null);
            at += n;
        }
    }
    std.debug.print("mixprice: chunks of {any} rows on slot {d} beside 1-row decode streams {any} (slots {d}..), {d} pairs a cell, which goes first alternating; ms from the call to the end on the device (the chunk) or to the logits on the host\n", .{ mix_ks, chunks_slot, decs, first_slot + 1, reps });
    var ids: [mix_ks[mix_ks.len - 1] + mix_dec_most]u32 = undefined;
    var flip = false;
    const Cell = struct { c: [16]f64, d: [16]f64, m: [16]f64, s: [16]f64, pos: usize };
    for (depths) |d| {
        // the slot to its depth (from where it stands)
        if (d > 0) {
            const t0 = clock.now(io).toNanoseconds();
            while (e.pos(chunks_slot) < d) {
                const p = e.pos(chunks_slot);
                const n = @min(d - p, engine.prompt_rows);
                var big: [engine.prompt_rows]u32 = undefined;
                for (big[0..n], 0..) |*v, i| v.* = seq[(p + i) % T];
                try x.forward(chunks_slot, p, big[0..n], null);
            }
            try e.stream.synchronize();
            std.debug.print("mixprice depth {d}: slot {d} at position {d} ({d:.2} s to get there)\n", .{ d, chunks_slot, e.pos(chunks_slot), secs(clock.now(io).toNanoseconds() - t0) });
        }
        var cells: [mix_ks.len][mix_dec_most + 1]Cell = undefined;
        for (decs) |R| {
            // warm-up (not timed): the decoders' graphs captured (a key of several streams at its second sighting)
            for (0..2) |_| {
                _ = try mixA(io, x, seq, T, d, mix_ks[0], R, &ids);
                _ = try mixB(io, x, seq, T, d, mix_ks[0], R, &ids);
            }
        }
        for (0..reps) |rep| for (mix_ks, 0..) |k, ki| for (decs) |R| {
            const cell = &cells[ki][R];
            if (rep == 0) cell.pos = 0;
            cell.pos += if (d == 0) 0 else e.pos(chunks_slot);
            flip = !flip;
            var a: [2]f64 = undefined;
            var m: f64 = undefined;
            if (flip) {
                a = try mixA(io, x, seq, T, d, k, R, &ids);
                m = try mixB(io, x, seq, T, d, k, R, &ids);
            } else {
                m = try mixB(io, x, seq, T, d, k, R, &ids);
                a = try mixA(io, x, seq, T, d, k, R, &ids);
            }
            cell.c[rep] = a[0];
            cell.d[rep] = a[1];
            cell.m[rep] = m;
            cell.s[rep] = a[0] + a[1] - m;
        };
        for (mix_ks, 0..) |k, ki| for (decs) |R| {
            const cell = &cells[ki][R];
            const c = medianOf(cell.c[0..reps]);
            const dd = medianOf(cell.d[0..reps]);
            const m = medianOf(cell.m[0..reps]);
            const sv = medianOf(cell.s[0..reps]);
            var lo: f64 = std.math.inf(f64);
            var hi: f64 = -std.math.inf(f64);
            for (cell.s[0..reps]) |v| {
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
            std.debug.print("  mixprice depth {d} (mean position {d}) k {d} decoders {d}: chunk {d:.2} ms ({d:.3} a row), decode {d:.2}, chunk + decode {d:.2}, mixed {d:.2}; the fold saves {d:.2} ms (pairs {d:.2} .. {d:.2}), {d:.1} % of chunk + decode\n", .{ d, cell.pos / reps, k, R, c, c / @as(f64, @floatFromInt(k)), dd, c + dd, m, sv, lo, hi, 100 * sv / (c + dd) });
        };
    }
}

/// The median of `xs` (sorted in place).
fn medianOf(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const m = xs.len / 2;
    return if (xs.len % 2 == 1) xs[m] else (xs[m - 1] + xs[m]) / 2;
}

/// The decode streams' segments and ids (one row each at its position; any token: a price).
fn mixDecs(e: *engine.Engine, seq: []const u32, T: usize, R: usize, segs: []engine.Segment, ids: []u32) void {
    for (0..R) |j| {
        const slot = first_slot + 1 + @as(u32, @intCast(j));
        const p = e.pos(slot);
        segs[j] = .{ .slot = slot, .pos0 = p, .rows = 1 };
        ids[j] = seq[p % T];
    }
}

/// The chunk's position at depth band `d` (0: the slot starts over).
fn mixAt(e: *engine.Engine, d: usize) usize {
    return if (d == 0) 0 else e.pos(chunks_slot);
}

/// Pair half A: the chunk alone, then the decoders' forward with its head; their ms.
fn mixA(io: std.Io, x: Leader, seq: []const u32, T: usize, d: usize, k: usize, R: usize, ids: []u32) ![2]f64 {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const p = mixAt(e, d);
    for (ids[0..k], 0..) |*v, i| v.* = seq[(p + i) % T];
    const t0 = clock.now(io).toNanoseconds();
    try x.forward(chunks_slot, p, ids[0..k], null);
    try e.stream.synchronize();
    const t1 = clock.now(io).toNanoseconds();
    var segs: [mix_dec_most]engine.Segment = undefined;
    mixDecs(e, seq, T, R, &segs, ids[0..R]);
    const t2 = clock.now(io).toNanoseconds();
    try x.forwardMany(segs[0..R], ids[0..R], .{ .lo = 0, .n = R });
    _ = try e.logits();
    const t3 = clock.now(io).toNanoseconds();
    return .{ secs(t1 - t0) * 1e3, secs(t3 - t2) * 1e3 };
}

/// Pair half B: one forward of [the chunk, the decoders' rows], the head over the decode rows; its ms.
fn mixB(io: std.Io, x: Leader, seq: []const u32, T: usize, d: usize, k: usize, R: usize, ids: []u32) !f64 {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const p = mixAt(e, d);
    for (ids[0..k], 0..) |*v, i| v.* = seq[(p + i) % T];
    var segs: [1 + mix_dec_most]engine.Segment = undefined;
    segs[0] = .{ .slot = chunks_slot, .pos0 = p, .rows = k };
    mixDecs(e, seq, T, R, segs[1..][0..R], ids[k..][0..R]);
    const t0 = clock.now(io).toNanoseconds();
    try x.forwardMany(segs[0 .. 1 + R], ids[0 .. k + R], .{ .lo = k, .n = R });
    _ = try e.logits();
    return secs(clock.now(io).toNanoseconds() - t0) * 1e3;
}
