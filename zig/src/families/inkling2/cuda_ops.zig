//! What every Inkling-2 op group's entry points take: device matrices (`Mat`), the op context (`Ctx`: driver,
//! stream, every group's kernels, cuBLASLt plans, a per-op scratch arena), and the scaffold's own helpers (bf16
//! Linear via cuBLASLt, RMS norm rows via norms.cu, device copies). An entry point that has no kernel yet returns
//! `error.NotImplemented`; the forward (cuda_forward.zig) stops there by name, or under --feed takes torch's outputs.
//!
//! Layout rules for every `Mat` an entry point receives: row-major, `rows` rows of `cols` elements, `ld` elements
//! between rows (ld == cols unless a header says otherwise), 256-byte aligned base (the scaffold's buffers are), rows
//! are token rows in position order. Head-shaped tensors are flattened: [R, heads * dim], head h at columns
//! h*dim .. h*dim+dim-1 (Transformers' `.view(*shape, heads, dim)`).
const std = @import("std");
const cuda = @import("cuda");
const kernels_mod = @import("cuda_kernels.zig");

pub const Kernels = kernels_mod.Kernels;

pub const Dt = enum {
    bf16,
    f32,
    i32,
    u8,

    pub fn size(d: Dt) usize {
        return switch (d) {
            .bf16 => 2,
            .f32, .i32 => 4,
            .u8 => 1,
        };
    }
};

/// A row-major device matrix.
pub const Mat = struct {
    ptr: u64 = 0,
    rows: usize = 0,
    cols: usize = 0,
    ld: usize = 0,
    dt: Dt = .bf16,

    pub fn of(ptr: u64, rows: usize, cols: usize, dt: Dt) Mat {
        return .{ .ptr = ptr, .rows = rows, .cols = cols, .ld = cols, .dt = dt };
    }

    /// Bytes from the first row's start to the last row's end.
    pub fn bytes(m: Mat) usize {
        if (m.rows == 0) return 0;
        return ((m.rows - 1) * m.ld + m.cols) * m.dt.size();
    }

    pub fn row(m: Mat, i: usize) u64 {
        return m.ptr + i * m.ld * m.dt.size();
    }

    /// Rows `lo` .. `lo + n`.
    pub fn slice(m: Mat, lo: usize, n: usize) Mat {
        return .{ .ptr = m.row(lo), .rows = n, .cols = m.cols, .ld = m.ld, .dt = m.dt };
    }

    pub fn dense(m: Mat) bool {
        return m.ld == m.cols;
    }

    pub fn is(m: Mat, rows: usize, cols: usize, dt: Dt) bool {
        return m.rows == rows and m.cols == cols and m.dt == dt;
    }
};

/// The completed block-residual sources of a forward's rows: `count` matrices [R, hidden] bf16, block i at
/// `ptr + i * stride` bytes (Transformers' `block_states[..., i, :]`): the inputs of layers 0, 6, 12, ... below the
/// current layer (block 0 is the normed embedding).
pub const Blocks = struct {
    ptr: u64,
    stride: usize,
    count: usize,
    rows: usize,
    hidden: usize,

    pub fn at(b: Blocks, i: usize) Mat {
        return Mat.of(b.ptr + i * b.stride, b.rows, b.hidden, .bf16);
    }
};

/// cuBLASLt plans by shape (a torch Linear: bf16 X [m, k] . W [n, k]^T, fp32 accumulate, bf16 or fp32 out).
pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: ?cuda.cublaslt.Library = null,
    plans: std.AutoHashMapUnmanaged([4]u64, cuda.cublaslt.Linear) = .empty,
    workspace: ?cuda.DeviceBuffer = null,

    pub const workspace_bytes: usize = 32 << 20;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var g: Gemms = .{ .gpa = gpa };
        g.lt = try cuda.cublaslt.Library.open();
        errdefer g.lt.?.close();
        g.workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        return g;
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |p| p.deinit();
        g.plans.deinit(g.gpa);
        if (g.workspace) |*w| w.free();
        if (g.lt) |*l| l.close();
        g.* = undefined;
    }

    fn plan(g: *Gemms, m: usize, n: usize, k: usize, out: cuda.cublaslt.DataType) !*const cuda.cublaslt.Linear {
        const key = [4]u64{ m, n, k, @intCast(@intFromEnum(out)) };
        if (g.plans.getPtr(key)) |p| return p;
        var p = try cuda.cublaslt.Linear.init(&g.lt.?, m, n, k, out, workspace_bytes);
        errdefer p.deinit();
        try g.plans.put(g.gpa, key, p);
        return g.plans.getPtr(key).?;
    }
};

/// The op context every entry point takes.
pub const Ctx = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: *const Kernels,
    gemms: *Gemms,
    scratch: *Scratch,
    /// While a forward is captured: its launches on ring views of pool runs (`launchOn`).
    runs: ?*RunLog = null,

    /// `bytes` of device scratch (256-byte aligned) valid until the op returns; the forward resets it per op.
    pub fn tmp(c: *const Ctx, bytes: usize) !u64 {
        return c.scratch.take(bytes);
    }

    /// out [m, n] (bf16 or fp32) = x [m, k] bf16 . w [n, k]^T bf16: Transformers' bf16 `nn.Linear` (cuBLASLt, fp32
    /// accumulate, one rounding). x and out dense. Measured: bit-equal to torch at the same m, but cuBLASLt picks
    /// its algorithm by m, so a row's bits change with the forward's row count (8-row forwards against one of 25:
    /// rel 2e-3, a third of the bf16 values 1 ulp or more off). Not row-invariant: the G1 seam gate needs a
    /// projection whose reduction order does not depend on m (one algorithm a shape, or a tiled kernel of ours).
    pub fn linear(c: *const Ctx, x: Mat, w: u64, n: usize, out: Mat) !void {
        if (x.dt != .bf16 or !x.dense() or !out.dense() or out.rows != x.rows or out.cols != n) return error.BadLinearShape;
        if (x.rows == 0) return;
        const dt: cuda.cublaslt.DataType = switch (out.dt) {
            .bf16 => .bf16,
            .f32 => .f32,
            else => return error.BadLinearShape,
        };
        const p = try c.gemms.plan(x.rows, n, x.cols, dt);
        const ws = c.gemms.workspace.?;
        try p.run(x.ptr, w, out.ptr, ws.ptr, ws.len, c.stream.handle);
    }

    /// y [R, width] bf16 = bf16(x * rsqrt(mean(x^2) + eps) * gain), x bf16 or fp32, gain fp32 [width] (0: none):
    /// Transformers' `Inkling2RMSNorm` (fp32 math, gain after the normalisation) then `.to(bf16)`.
    pub fn rms(c: *const Ctx, x: Mat, gain: u64, y: Mat, eps: f32) !void {
        if (!x.dense() or !y.dense() or x.rows != y.rows or x.cols != y.cols or y.dt != .bf16) return error.BadNormShape;
        const kind: cuda.norms.Kind = switch (x.dt) {
            .bf16 => .bf16,
            .f32 => .f32,
            else => return error.BadNormShape,
        };
        try c.k.norms.rms(c.stream, x.ptr, kind, if (gain == 0) null else gain, y.ptr, .bf16, x.rows, x.cols, eps);
    }

    /// `n` bytes from `src` to `dst` on the op stream.
    pub fn copy(c: *const Ctx, dst: u64, src: u64, n: usize) !void {
        if (n == 0) return;
        const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = dst, .len = n, .borrowed = true };
        try view.copyFrom(0, src, n, c.stream.handle);
    }

    /// Rows of `src` into `dst` (same shape and dtype; any ld).
    pub fn copyRows(c: *const Ctx, dst: Mat, src: Mat) !void {
        if (dst.rows != src.rows or dst.cols != src.cols or dst.dt != src.dt) return error.BadCopyShape;
        if (dst.dense() and src.dense()) return c.copy(dst.ptr, src.ptr, src.bytes());
        for (0..src.rows) |i| try c.copy(dst.row(i), src.row(i), src.cols * src.dt.size());
    }

    pub fn zero(c: *const Ctx, ptr: u64, n: usize) !void {
        if (n == 0) return;
        const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = ptr, .len = n, .borrowed = true };
        try view.fill8(0, c.stream.handle);
    }

    /// A launch whose arguments may name a ring view of a stream's run of pool pages (`run`: the run's first page;
    /// `k`, `v` the view's bases): while a forward is captured (`runs` set) it is recorded, so a replay can point the
    /// node at the stream's run when that moves (cuda_engine.zig Graph).
    pub fn launchOn(c: *const Ctx, run: ?u32, k: u64, v: u64, f: cuda.Function, cfg: cuda.launch.Config, a: *cuda.Args) !void {
        try cuda.launch.launch(f, cfg, c.stream, a);
        if (run) |r| if (c.runs) |log| try log.add(f, cfg, a, k, v, r);
    }
};

/// The launches of a captured forward that read or write a ring view of a stream's run of pool pages
/// (Ctx.launchOn): each with its arguments and the argument slots holding the view's bases.
pub const RunLog = struct {
    pub const Launch = struct {
        f: cuda.Function,
        cfg: cuda.launch.Config,
        args: cuda.Args,
        /// the run's first page when captured
        run: u32,
        /// argument indices that hold a base (k or v), `n` of them
        at: [4]u8 = @splat(0),
        n: u8 = 0,
    };

    gpa: std.mem.Allocator,
    items: std.ArrayList(Launch) = .empty,

    pub fn deinit(l: *RunLog) void {
        l.items.deinit(l.gpa);
    }

    fn add(l: *RunLog, f: cuda.Function, cfg: cuda.launch.Config, a: *const cuda.Args, k: u64, v: u64, run: u32) !void {
        var x: Launch = .{ .f = f, .cfg = cfg, .args = a.*, .run = run };
        for (0..a.count) |i| {
            const val = a.integer(i) orelse continue;
            if (a.sizes[i] != 8 or (val != k and val != v)) continue;
            if (x.n == x.at.len) return error.RunLaunchArgs;
            x.at[x.n] = @intCast(i);
            x.n += 1;
        }
        if (x.n == 0) return error.RunLaunchArgs;
        try l.items.append(l.gpa, x);
    }
};

/// A bump arena of device memory, reset by the forward before every op. It keeps the most any op took (`peak`), and
/// by forward kind (`kind`, set by whoever starts the forward; `peaks`); a new high is logged once it passes the last
/// logged one by an eighth. What a forward will take is cuda_scratch.zig's `need` (checked before a step's header).
pub const Scratch = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,
    peak: usize = 0,
    kind: Kind = .other,
    peaks: [kinds]usize = @splat(0),
    logged: usize = 0,

    /// The forward a take belongs to: a prompt chunk (more than a verify window's rows, one stream), a decode row,
    /// a verify window (2-8 rows), several streams' rows (batched), or anything else (op checks, tools).
    pub const Kind = enum { prompt, decode, window, batched, other };
    pub const kinds = @typeInfo(Kind).@"enum".field_names.len;

    pub fn init(d: *const cuda.Driver, bytes: usize) !Scratch {
        return .{ .buf = try cuda.DeviceBuffer.alloc(d, bytes) };
    }

    pub fn deinit(s: *Scratch) void {
        s.buf.free();
    }

    pub fn reset(s: *Scratch) void {
        s.used = 0;
    }

    pub fn take(s: *Scratch, bytes: usize) !u64 {
        const at = std.mem.alignForward(usize, s.used, 256);
        if (at + bytes > s.buf.len) {
            std.log.err("inkling2: op scratch of {d} MiB is short of {d} MiB ({t} forward; raise TF_INK2_SCRATCH_GIB)", .{ s.buf.len >> 20, (at + bytes) >> 20, s.kind });
            return error.ScratchFull;
        }
        s.used = at + bytes;
        if (s.used > s.peak) s.rise();
        const k = @intFromEnum(s.kind);
        s.peaks[k] = @max(s.peaks[k], s.used);
        return s.buf.ptr + at;
    }

    fn rise(s: *Scratch) void {
        s.peak = s.used;
        if (s.peak <= s.logged + s.logged / 8) return;
        s.logged = s.peak;
        std.log.info("inkling2: op scratch peak {d:.1} MiB of {d} MiB ({t} forward)", .{ mib(s.peak), s.buf.len >> 20, s.kind });
    }

    /// The peaks by forward kind, one log line.
    pub fn report(s: *const Scratch, rank: u32) void {
        std.log.info("inkling2 rank {d}: op scratch peaks (MiB of {d}): prompt {d:.1}, decode {d:.1}, window {d:.1}, batched {d:.1}, other {d:.1}", .{ rank, s.buf.len >> 20, mib(s.peaks[0]), mib(s.peaks[1]), mib(s.peaks[2]), mib(s.peaks[3]), mib(s.peaks[4]) });
    }
};

pub fn mib(b: usize) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 20);
}

test "inkling2 cuda ops: Mat rows, slices and byte spans" {
    const m: Mat = .{ .ptr = 4096, .rows = 3, .cols = 10, .ld = 16, .dt = .bf16 };
    try std.testing.expectEqual(@as(usize, (2 * 16 + 10) * 2), m.bytes());
    try std.testing.expectEqual(@as(u64, 4096 + 2 * 16 * 2), m.row(2));
    const s = m.slice(1, 2);
    try std.testing.expectEqual(@as(u64, 4096 + 32), s.ptr);
    try std.testing.expect(!s.dense());
    try std.testing.expect(Mat.of(0, 4, 384, .f32).is(4, 384, .f32));
    const b: Blocks = .{ .ptr = 1 << 20, .stride = 4 * 4608 * 2, .count = 2, .rows = 4, .hidden = 4608 };
    try std.testing.expectEqual(@as(u64, (1 << 20) + 4 * 4608 * 2), b.at(1).ptr);
}

test "inkling2 cuda ops: the scratch's peaks by forward kind" {
    var s: Scratch = .{ .buf = .{ .d = undefined, .ptr = 1 << 20, .len = 64 << 20, .borrowed = true } };
    s.kind = .prompt;
    _ = try s.take(1000);
    _ = try s.take(10);
    try std.testing.expectEqual(@as(usize, 1024 + 10), s.peaks[0]);
    s.reset();
    s.kind = .decode;
    try std.testing.expectEqual(@as(u64, (1 << 20)), try s.take(300));
    try std.testing.expectEqual(@as(usize, 300), s.peaks[1]);
    try std.testing.expectEqual(@as(usize, 1034), s.peak);
}
