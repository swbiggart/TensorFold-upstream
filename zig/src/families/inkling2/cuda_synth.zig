//! Tooling (tf-inkling2-decode --exp deep --fill synth): a slot's state for positions 0 .. D - 1 written directly
//! with deterministic synthetic values, no forward: every layer's keys and values (a global layer's positions
//! 0 .. D - 1, a sliding layer's whole ring), its four convolution states, and with the drafter the slot's draft
//! caches (every layer's rows; a local layer's mirrored rows kept equal to the rows they mirror). A decode step at D
//! then reads them as it reads a prefill's, so decode rows against one prompt forward over the same positions check
//! the indexing, grids, op scratch and log scaling of a 1M-position slot exactly in minutes, where a real prefill to
//! 1M takes hours.
//!
//! The values: one pattern of `elems` (a prime) bf16 values, about N(0, 1) (a sum of four uniforms), from a fixed
//! seed, laid over each buffer from its own offset (rank, layer, buffer): byte b of a buffer is the pattern's byte
//! (offset + b) mod `bytes`. A prime period keeps a row's bytes from repeating at any power-of-two distance, so a
//! read or write off by a wrapped 32-bit offset lands on other values. `check` reads the global layers' positions
//! 0 .. D - 1 back against the pattern after the experiment: no forward wrote below D.
const std = @import("std");
const cuda = @import("cuda");
const ops = @import("cuda_ops.zig");
const state = @import("cuda_state.zig");
const conv = @import("cuda_conv.zig");
const weights = @import("cuda_weights.zig");
const dspark = @import("cuda_dspark.zig");

/// bf16 values in the pattern (a prime), and its bytes.
pub const elems: usize = 8_388_593;
pub const bytes: usize = 2 * elems;
const seed: u64 = 0x1f2e3d4c5b6a7988;

/// The pattern on the host (`bytes` bytes; the caller frees it).
pub fn pattern(gpa: std.mem.Allocator) ![]u8 {
    const out = try gpa.alloc(u8, bytes);
    errdefer gpa.free(out);
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (0..elems) |i| {
        const u = r.int(u64);
        var s: f32 = 0;
        inline for (0..4) |j| s += @as(f32, @floatFromInt((u >> (16 * j)) & 0xffff)) / 65536.0 - 0.5;
        // the sum of four uniforms has variance 1/3: scaled to about N(0, 1), within +-3.47
        const x: f32 = s * 1.7320508;
        const b: u32 = @bitCast(x);
        const h: u16 = @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16); // round to nearest even
        std.mem.writeInt(u16, out[2 * i ..][0..2], h, .little);
    }
    return out;
}

/// Buffer `which` of layer `layer` (or the drafter's) on rank `rank`: its byte offset into the pattern (even, so
/// every bf16 stays whole).
pub fn offset(rank: u32, layer: usize, which: usize) usize {
    var h = std.hash.Wyhash.init(seed);
    h.update(std.mem.asBytes(&rank));
    h.update(std.mem.asBytes(&layer));
    h.update(std.mem.asBytes(&which));
    return 2 * @as(usize, @intCast(h.final() % elems));
}

/// `n` bytes at `dst` from the pattern on the device (`pat`) from byte `off`, wrapping.
fn lay(c: *const ops.Ctx, pat: u64, dst: u64, n: usize, off: usize) !void {
    var done: usize = 0;
    var at = off;
    while (done < n) {
        const k = @min(n - done, bytes - at);
        try c.copy(dst + done, pat + at, k);
        done += k;
        at = 0;
    }
}

/// Buffers a layer's fill takes, in order: keys, values, then the four convolution states.
const kv_k = 0;
const kv_v = 1;
const conv0 = 2;
/// The drafter's layers (`which` = drafter_which + 2 l for keys, + 1 for values), after every target layer's.
const drafter_which = 64;

/// Rows of layer `l`'s keys (and values) the fill writes: a global layer's positions 0 .. positions - 1, a sliding
/// layer's every ring slot.
fn kvRows(st: *const state.State, l: usize, positions: usize) usize {
    const kv = st.layers[l].kv;
    return if (kv.window == 0) positions else kv.slots;
}

/// Slot `st` (on rank `rank`) as if positions 0 .. positions - 1 had run, with the pattern's values (see the top);
/// `caches` the slot's draft caches when the drafter is loaded. Leaves the slot's position to the caller.
pub fn fill(gpa: std.mem.Allocator, c: *const ops.Ctx, st: *const state.State, caches: ?*const dspark.Caches, rank: u32, positions: usize) !void {
    const host = try pattern(gpa);
    defer gpa.free(host);
    var pat = try cuda.DeviceBuffer.fromHost(c.d, host);
    defer pat.free();
    for (st.layers, 0..) |l, i| {
        const layer = @as(usize, st.first) + i;
        const row = l.kv.heads * weights.head_dim * 2;
        if (l.kv.paged()) {
            // on pages: positions 0 .. positions - 1 a run of the slot's pages at a time, the pattern from where a
            // window's row would take it
            var runs = state.RowRuns.of(l.kv, st.ids, 0, positions);
            while (try runs.next()) |r| {
                try lay(c, pat.ptr, r.k, r.n * row, (offset(rank, layer, kv_k) + r.pos * row) % bytes);
                try lay(c, pat.ptr, r.v, r.n * row, (offset(rank, layer, kv_v) + r.pos * row) % bytes);
            }
        } else {
            const n = kvRows(st, i, positions) * row;
            try lay(c, pat.ptr, l.kv.k, n, offset(rank, layer, kv_k));
            try lay(c, pat.ptr, l.kv.v, n, offset(rank, layer, kv_v));
        }
        for ([4]u64{ l.conv_k, l.conv_v, l.conv_attn, l.conv_mlp }, [4]usize{ row, row, weights.hidden * 2, weights.hidden * 2 }, 0..) |p, w, j| {
            try lay(c, pat.ptr, p, conv.state_rows * w, offset(rank, layer, conv0 + j));
        }
    }
    if (caches) |dc| for (0..dspark.nlayers) |l| {
        if (!dspark.local[l]) if (dc.pagedKv(st.ids.len)) |kv| {
            // the global layer on pages (stage 2c): positions 0 .. positions - 1, the pattern from where its row in
            // the slot's own table would take it
            var runs = state.RowRuns.of(kv, st.ids, 0, @min(positions, st.limit()));
            while (try runs.next()) |r| {
                try lay(c, pat.ptr, r.k, r.n * dc.row, (offset(rank, 0, drafter_which + 2 * l) + r.pos * dc.row) % bytes);
                try lay(c, pat.ptr, r.v, r.n * dc.row, (offset(rank, 0, drafter_which + 2 * l + 1) + r.pos * dc.row) % bytes);
            }
            continue;
        };
        for ([2]u64{ dc.keys(l), dc.values(l) }, 0..) |p, j| {
            try lay(c, pat.ptr, p, dc.rows(l) * dc.row, offset(rank, 0, drafter_which + 2 * l + j));
            // a ring's first `window` rows are mirrored after it (cuda_dspark.zig localRows): keep them equal
            const ring = dc.wrap(l);
            if (ring > 0) try c.copy(p + ring * dc.row, p, dspark.window * dc.row);
        }
    };
    try c.stream.synchronize();
}

/// What `check` found: bytes compared and the first position whose row differs from the pattern (null: none).
/// `ranks_bad`: the ranks whose own check found a row unequal (every rank's, gathered by Engine.synthCheck).
pub const Checked = struct { bytes: u64 = 0, layers: u32 = 0, bad: ?usize = null, bad_layer: usize = 0, ranks_bad: u32 = 0 };

/// The global layers' keys and values of positions 0 .. positions - 1 read back against the pattern.
/// TF_INK2_SYNTH_FAULT=R (tests): rank R reads its first global layer's first key byte flipped, as a changed row.
pub fn check(gpa: std.mem.Allocator, c: *const ops.Ctx, st: *const state.State, rank: u32, positions: usize) !Checked {
    try c.stream.synchronize();
    const fault = if (std.c.getenv("TF_INK2_SYNTH_FAULT")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch std.math.maxInt(u32)) == rank else false;
    const host = try pattern(gpa);
    defer gpa.free(host);
    const chunk: usize = 64 << 20;
    const buf = try gpa.alloc(u8, chunk);
    defer gpa.free(buf);
    var out: Checked = .{};
    for (st.layers, 0..) |l, i| {
        if (l.kv.window != 0) continue;
        const layer = @as(usize, st.first) + i;
        const row = l.kv.heads * weights.head_dim * 2;
        out.layers += 1;
        for ([2]usize{ kv_k, kv_v }) |which| {
            const n = positions * row;
            const off = offset(rank, layer, which);
            var at: usize = 0;
            while (at < n) {
                // the next run of contiguous rows from logical byte `at` (a window: all of it; pages: a run of them)
                var runs = state.RowRuns.of(l.kv, st.ids, at / row, positions);
                const r = (try runs.next()).?;
                const p = if (which == kv_k) r.k else r.v;
                const k = @min(r.n * row, chunk - chunk % row);
                try cuda.memory.download(c.d, p, buf[0..k]);
                if (fault and out.layers == 1 and which == kv_k and at == 0) buf[0] ^= 1;
                // compare against the pattern from (off + at) mod bytes, a wrap at a time
                var done: usize = 0;
                while (done < k) {
                    const from = (off + at + done) % bytes;
                    const m = @min(k - done, bytes - from);
                    if (!std.mem.eql(u8, buf[done..][0..m], host[from..][0..m])) {
                        var j: usize = 0;
                        while (buf[done + j] == host[from + j]) j += 1;
                        const pos = (at + done + j) / row;
                        if (out.bad == null or pos < out.bad.?) {
                            out.bad = pos;
                            out.bad_layer = layer;
                        }
                        break;
                    }
                    done += m;
                }
                out.bytes += k;
                at += k;
            }
        }
    }
    return out;
}

test "inkling2 synth: the pattern is fixed, about N(0, 1), and offsets are even and inside it" {
    const gpa = std.testing.allocator;
    const p = try pattern(gpa);
    defer gpa.free(p);
    var sum: f64 = 0;
    var sq: f64 = 0;
    for (0..elems) |i| {
        const h = std.mem.readInt(u16, p[2 * i ..][0..2], .little);
        const x: f32 = @bitCast(@as(u32, h) << 16);
        try std.testing.expect(std.math.isFinite(x) and @abs(x) < 3.5);
        sum += x;
        sq += x * x;
    }
    const n: f64 = @floatFromInt(elems);
    try std.testing.expect(@abs(sum / n) < 0.01 and @abs(sq / n - 1) < 0.02);
    const q = try pattern(gpa);
    defer gpa.free(q);
    try std.testing.expectEqualSlices(u8, p, q);
    for (0..4) |r| for (0..48) |l| for (0..6) |w| {
        const o = offset(@intCast(r), l, w);
        try std.testing.expect(o % 2 == 0 and o < bytes);
    };
    try std.testing.expect(offset(0, 3, 0) != offset(1, 3, 0) and offset(0, 3, 0) != offset(0, 3, 1));
    var d: usize = 2;
    while (d * d <= elems) : (d += 1) try std.testing.expect(elems % d != 0);
}
