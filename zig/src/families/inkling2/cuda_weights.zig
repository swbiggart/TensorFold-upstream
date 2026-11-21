//! One rank's Inkling-2 weights on the device for a layer range [a, b), from the rank plan's byte ranges (plan.zig):
//! each tensor the rank holds is read from its safetensors run(s) and uploaded as one dense array. Reader threads
//! (ReadOpts: 8 by default, direct reads into pinned buffers, no page cache and no second host copy) each take whole
//! pieces, hash every byte as read (the content digests, digests.zig) and copy each chunk to the device while reading
//! the next. Quantized expert
//! bytes stay packed as stored (NVFP4 / MXFP8, with their scales, amax and denominator); BF16 and F32 tensors keep
//! their dtype, except the RMS norm gains and the router weight, converted exactly to fp32 (Transformers keeps the
//! router in fp32 and multiplies gains in fp32). Group kernels that want another layout repack in their
//! `prepare(layer)` hook (cuda_<group>.zig), which runs once per layer after the upload; what they allocate there
//! goes in `Weights.own` so it is freed with the weights.
//!
//! Device layouts (rank r of W; hq = its query heads, hk = its key heads, E = its experts, H = 4608, D = 128):
//!   wq, wg      bf16 [hq*D, H]   rows by query head (Transformers q_proj, gate_proj)
//!   wr          bf16 [hq*16, H]  relative-state rows (r_proj), 16 a head
//!   wsink       bf16 [hq, H]
//!   wk, wv      bf16 [hk*D, H]
//!   wo          bf16 [H, hq*D]   the rank's column cut of o_proj, gathered dense
//!   q_scale     bf16 [hq, D], rel_scale bf16 [hq, 16]   (Inkling2ScaledL2Clip.scale)
//!   rel_proj    bf16 [16, 1024]  (Inkling2RelativeLogits.proj; extent 1024 in both layer kinds)
//!   local:  k_conv, v_conv bf16 [hk*D, 4] (conv1d weight [C, 1, 4]; tap 3 multiplies the current row)
//!   global: k_conv, v_conv bf16 [4, hk*D] (static_weight [lag, C]; lag 0 = current row)
//!           k_wproj, v_wproj bf16 [hk*4, H] (weight_proj; row = head*4 + lag)
//!   attn_conv, mlp_conv bf16 [H, 4] (post-attention / post-MLP conv1d weight, tap 3 = current row)
//!   in_norm[0..3] fp32 [H]: local attn_norm in [0]; global q, k, v input norms
//!   mlp_norm fp32 [H]; attn_query fp32 [nq, H] (0 at layer 0); mlp_query fp32 [H]
//!   router bf16 [384, H] as stored (fp32 with TF_INK2_ROUTER=fp32, cuda_moe.zig routerBf16); router_bias fp32 [384]
//!   experts: see `Experts`.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

pub const hidden = 4608;
pub const head_dim = 128;
pub const d_rel = 16;
pub const taps = 4;

pub const Format = enum { nvfp4, mxfp8 };

/// A layer's experts as stored, the rank's span [lo, hi) of the 384 (row `e - lo` holds expert e).
///   NVFP4 (layers 0-39): w13 u8 [E, 2I, H/2] (e2m1 pairs, low nibble = even column), w13_scale e4m3 [E, 2I, H/16],
///     w13_amax f32 [E]; w2 u8 [E, H, I/2], w2_scale e4m3 [E, H, I/16], w2_amax f32 [E]; den13 / den2 the
///     outer_scale_denominator scalars (read from the file: 2688 in the release). Weight value =
///     e2m1 * e4m3 * (amax[e] / den), as the oracle decodes it.
///   MXFP8 (layers 40-47): w13 e4m3 [E, 2I, H], w13_scale u8 [E, 2I, H/32] (ue8m0, one a row a 32 columns);
///     w2 e4m3 [E, H, I], w2_scale u8 [E, H, I/32]. Weight value = e4m3 * 2^(scale - 127).
///   w13 rows are interleaved: row 2i = gate_i, row 2i + 1 = up_i (I = 2304).
/// Split experts (plan.Layout.split, W ranks): every expert (lo = 0, hi = 384), and of each only its rows
/// [n13 * part, + n13) of w13 (gate/up pairs n13 / 2 * part ..) and [n2 * part, + n2) of w2 (all of K), with their
/// scales' rows, dense: w13 [E, n13, ...], w2 [E, n2, ...]; amax whole [384]. Whole experts: n13 = 2I, n2 = H.
pub const Experts = struct {
    format: Format,
    lo: u32,
    hi: u32,
    /// The split layout's world size (1: whole experts) and this rank's place in it.
    split: u32 = 1,
    part: u32 = 0,
    /// Rows of each expert held: w13 (gate/up interleaved) and w2 (output rows).
    n13: u32 = 2 * 2304,
    n2: u32 = hidden,
    w13: u64 = 0,
    w13_scale: u64 = 0,
    w2: u64 = 0,
    w2_scale: u64 = 0,
    w13_amax: u64 = 0,
    w2_amax: u64 = 0,
    den13: f32 = 0,
    den2: f32 = 0,
    /// Free for the MoE group's repacked forms (moe.prepare).
    extra: [4]u64 = @splat(0),

    pub fn count(e: Experts) u32 {
        return e.hi - e.lo;
    }
};

pub const Layer = struct {
    index: u32,
    global: bool,
    boundary: bool,
    /// The rank's query and key heads.
    hq: u32,
    hk: u32,
    /// Attention-input queries: 0 (layer 0), 1 (sliding) or 3 (global).
    nq: u32,
    in_norm: [3]u64 = @splat(0),
    mlp_norm: u64 = 0,
    attn_query: u64 = 0,
    mlp_query: u64 = 0,
    wq: u64 = 0,
    wk: u64 = 0,
    wv: u64 = 0,
    wr: u64 = 0,
    wg: u64 = 0,
    wsink: u64 = 0,
    wo: u64 = 0,
    q_scale: u64 = 0,
    rel_scale: u64 = 0,
    rel_proj: u64 = 0,
    k_conv: u64 = 0,
    v_conv: u64 = 0,
    k_wproj: u64 = 0,
    v_wproj: u64 = 0,
    attn_conv: u64 = 0,
    mlp_conv: u64 = 0,
    router: u64 = 0,
    router_bias: u64 = 0,
    experts: Experts,
    /// Free for the attention and convolution groups' repacked forms (their prepare hooks).
    extra: [4]u64 = @splat(0),
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    /// Layers a .. b, `layers[i]` is layer a + i.
    first: u32,
    layers: []Layer,
    embed: u64 = 0, // bf16 [embed_rows.len(), H]: the table rows this rank holds (cuda_embed.zig)
    embed_norm: u64 = 0, // fp32 [H]
    /// The embedding rows `embed` holds: vocabulary entries embed_rows.lo .. embed_rows.hi (the whole table on rank 0
    /// as loaded; cuda_embed.zig `spread` leaves each rank its share).
    embed_rows: ink.plan.Span = .{ .lo = 0, .hi = 0 },
    unembed: u64 = 0, // bf16 [vocab_rows.len(), H]: the rank's share of the vocabulary (plan.zig)
    norm: u64 = 0, // fp32 [H]
    output_query: u64 = 0, // fp32 [H]
    /// The unembed rows this rank holds: vocabulary entries vocab_rows.lo .. vocab_rows.hi (all of them at W = 1).
    vocab_rows: ink.plan.Span = .{ .lo = 0, .hi = 0 },
    vocab: u32 = 0,
    unpadded_vocab: u32 = 0,
    own: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,
    /// Bytes freed after upload (repacked sources: the bf16 attention copies).
    released: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.own.items) |*b| b.free();
        w.own.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    /// Frees the owned buffer at `ptr` (a prepare hook dropping a source it has repacked); false if none is owned there.
    pub fn release(w: *Weights, ptr: u64) bool {
        for (w.own.items, 0..) |*b, i| if (b.ptr == ptr) {
            w.bytes -= b.len;
            w.released += b.len;
            b.free();
            _ = w.own.swapRemove(i);
            return true;
        };
        return false;
    }

    /// Device memory owned by the weights (for a prepare hook's repacked forms).
    pub fn alloc(w: *Weights, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(w.d, bytes);
        errdefer b.free();
        try w.own.append(w.gpa, b);
        w.bytes += bytes;
        return b.ptr;
    }

    /// Frees the buffer at `ptr` (one the loader uploaded), if the weights own one there.
    pub fn drop(w: *Weights, ptr: u64) void {
        if (ptr != 0) _ = w.release(ptr);
    }

    pub fn layer(w: *const Weights, l: u32) *const Layer {
        return &w.layers[l - w.first];
    }

    pub fn holds(w: *const Weights, l: u32) bool {
        return l >= w.first and l < w.first + w.layers.len;
    }
};

pub const Want = struct {
    rank: u32 = 0,
    lo: u32,
    hi: u32,
    /// The embedding and its norm (rank 0 loads them at any W; at W > 1 cuda_embed.zig `spread` moves them off rank 0).
    embed: bool = false,
    /// The final mix's query, the final norm and the rank's share of the unembed (vocab_rows).
    head: bool = false,
    /// The final mix's query and the final norm alone (the drafter's last tap on every rank).
    final: bool = false,
    /// Each loaded piece's content digest (xxh64 of its bytes as read, plan order): digests.zig checks them.
    digests: ?*std.ArrayList(ink.digests.Digest) = null,
};

/// Called once per loaded layer, after its tensors are on the device (each group's `prepare`).
pub const Prepare = *const fn (w: *Weights, l: *Layer) anyerror!void;

/// How the loader reads (environment, read once per load):
///   TF_INK2_LOAD_THREADS=N   reader threads (default 8); each reads whole pieces, hashes them and uploads them
///   TF_INK2_LOAD_DIRECT=0    page-cache reads instead of direct (O_DIRECT) ones; a file direct reads refuse falls
///                            back on its own
///   TF_INK2_LOAD_CHUNK_MIB=M a thread's two pinned read buffers, M MiB each (default 32)
///   TF_INK2_LOAD_RECHECK=1   after the load, every uploaded piece is read back from the device and hashed again
///                            (must equal the hash of the bytes read: proves the upload)
pub const ReadOpts = struct {
    threads: u32 = 8,
    direct: bool = true,
    chunk: usize = 32 << 20,
    recheck: bool = false,

    pub fn env() ReadOpts {
        var o: ReadOpts = .{};
        if (std.c.getenv("TF_INK2_LOAD_THREADS")) |v| o.threads = std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch o.threads, 1, 64);
        if (std.c.getenv("TF_INK2_LOAD_DIRECT")) |v| o.direct = v[0] != '0';
        if (std.c.getenv("TF_INK2_LOAD_CHUNK_MIB")) |v| o.chunk = @as(usize, std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 32, 1, 1024)) << 20;
        if (std.c.getenv("TF_INK2_LOAD_RECHECK")) |v| o.recheck = v[0] == '1';
        return o;
    }
};

/// Direct reads' alignment of file offsets, lengths and buffer addresses (a multiple of any logical block size).
const direct_align: u64 = 4096;
/// Rows of a strided run at most this far apart are read in one read (the gap's bytes read and skipped).
const merge_gap: u64 = 128 << 10;

/// One step of a run's load: `reads` reads of `read_len` bytes, read i from file offset file_at + i * read_step into
/// the buffer at i * read_len (all aligned to the reader's alignment), then `rows` rows of `width` bytes, the first at
/// buffer offset data_at and each `pitch` bytes after the one before, hashed in order and copied to the run's dense
/// device array at byte `dst` (rows back to back).
pub const Chunk = struct {
    file_at: u64,
    read_len: usize,
    reads: usize,
    read_step: u64,
    data_at: usize,
    pitch: usize,
    width: usize,
    rows: usize,
    dst: u64,

    /// Bytes of the buffer the reads fill.
    pub fn span(c: Chunk) usize {
        return c.reads * c.read_len;
    }

    /// Bytes read i must return (less at the end of the file is a short read).
    pub fn need(c: Chunk) usize {
        return if (c.reads == 1) c.data_at + (c.rows - 1) * c.pitch + c.width else c.data_at + c.width;
    }
};

/// A run's chunks for buffers of `cap` bytes and reads aligned to `a` (1: any): rows close together share a read;
/// rows far apart are read one by one, several a chunk; a row longer than a chunk is read in parts.
pub const Chunks = struct {
    r: ink.plan.Run,
    a: u64,
    cap: usize,
    row: u64 = 0,
    part: u64 = 0,

    fn up(x: u64, a: u64) u64 {
        return (x + a - 1) / a * a;
    }

    pub fn next(it: *Chunks) ?Chunk {
        const r = it.r;
        if (it.row >= r.count or r.len == 0) return null;
        const a = it.a;
        const cap: u64 = it.cap;
        const len = r.len;
        const off0 = r.offset + it.row * r.stride;
        if (len + 3 * a > cap or it.part > 0) {
            // a part of a long row: at most cap - 2a bytes, so its aligned read fits the buffer
            const off = off0 + it.part;
            const n = @min(len - it.part, cap - 2 * a);
            const at = off / a * a;
            const c: Chunk = .{ .file_at = at, .read_len = @intCast(up(off + n, a) - at), .reads = 1, .read_step = 0, .data_at = @intCast(off - at), .pitch = @intCast(n), .width = @intCast(n), .rows = 1, .dst = it.row * len + it.part };
            it.part += n;
            if (it.part == len) {
                it.part = 0;
                it.row += 1;
            }
            return c;
        }
        const left = r.count - it.row;
        if (left > 1 and r.stride >= len and r.stride - len <= merge_gap) {
            // rows close together: one read over as many as fit
            const at = off0 / a * a;
            const first = off0 - at + len;
            const n = @min(left, 1 + (cap - up(first, a) - a) / r.stride);
            const c: Chunk = .{ .file_at = at, .read_len = @intCast(up(off0 + (n - 1) * r.stride + len, a) - at), .reads = 1, .read_step = 0, .data_at = @intCast(off0 - at), .pitch = @intCast(r.stride), .width = @intCast(len), .rows = @intCast(n), .dst = it.row * len };
            it.row += n;
            return c;
        }
        // rows read one by one: several a chunk when every row sits at the same offset from the alignment
        const mis = off0 % a;
        const read_len = up(mis + len, a);
        const n = if (r.stride % a == 0) @min(left, cap / read_len) else 1;
        const c: Chunk = .{ .file_at = off0 - mis, .read_len = @intCast(read_len), .reads = @intCast(n), .read_step = r.stride, .data_at = @intCast(mis), .pitch = @intCast(read_len), .width = @intCast(len), .rows = @intCast(n), .dst = it.row * len };
        it.row += n;
        return c;
    }
};

/// What the loader does with a piece: upload its bytes as they are, or convert them first (bf16 gains to fp32, the
/// bf16 router to fp32, a scalar into the experts' facts). Every job hashes the bytes it reads (xxh64, rows in order).
const Job = struct {
    piece: u32,
    run: ink.plan.Run,
    kind: enum { raw, gain, router, scalar },
    dtype: ink.inventory.DType = .bf16,
    /// The device array (raw, gain, router), allocated before any read.
    dst: u64 = 0,
    /// A scalar's place.
    out: ?*f32 = null,
    xxh64: u64 = 0,
};

/// The reader threads: each takes the next job (largest first), reads it into its two pinned buffers in turn, hashes
/// each chunk and copies it to the device on its own stream while it reads the next one.
const Pool = struct {
    d: *const cuda.Driver,
    ctx: cuda.abi.Context,
    dir: []const u8,
    m: *const ink.describe.Model,
    jobs: []Job,
    order: []const u32,
    opts: ReadOpts,
    rank: u32,
    recheck: bool = false,
    next: std.atomic.Value(usize) = .init(0),
    err: std.atomic.Value(u16) = .init(0),
    /// Files that refused direct reads (read through the page cache instead).
    buffered: std.atomic.Value(u32) = .init(0),

    fn fail(p: *Pool, e: anyerror) void {
        _ = p.err.cmpxchgStrong(0, @intFromError(e), .acq_rel, .acquire);
        p.next.store(p.order.len, .release);
    }

    fn failed(p: *const Pool) ?anyerror {
        const e = p.err.load(.acquire);
        return if (e == 0) null else @errorFromInt(e);
    }

    fn work(p: *Pool) void {
        const run = if (p.recheck) p.recheckLoop() else p.loop();
        run catch |e| p.fail(e);
    }

    fn current(p: *Pool) !void {
        try p.d.check(p.d.api.cuCtxSetCurrent(p.ctx), "cuCtxSetCurrent");
        cuda.copy_source.setCurrent(p.ctx);
    }

    fn loop(p: *Pool) !void {
        try p.current();
        var t: Thread = .{ .p = p };
        defer t.deinit();
        try t.init();
        while (true) {
            const k = p.next.fetchAdd(1, .acq_rel);
            if (k >= p.order.len) break;
            try t.job(&p.jobs[p.order[k]]);
        }
        try t.s.?.synchronize();
    }

    /// Every raw piece read back from the device and hashed again.
    fn recheckLoop(p: *Pool) !void {
        try p.current();
        var hb = try cuda.HostBuffer.alloc(p.d, p.opts.chunk);
        defer hb.free();
        const buf = hb.bytes;
        while (true) {
            const k = p.next.fetchAdd(1, .acq_rel);
            if (k >= p.order.len) return;
            const j = &p.jobs[p.order[k]];
            if (j.kind != .raw) continue;
            const total: usize = @intCast(j.run.bytes());
            var h = std.hash.XxHash64.init(0);
            var done: usize = 0;
            while (done < total) {
                const n = @min(total - done, buf.len);
                try p.d.check(p.d.api.cuMemcpyDtoH_v2(buf.ptr, j.dst + done, n), "cuMemcpyDtoH");
                h.update(buf[0..n]);
                done += n;
            }
            if (h.final() != j.xxh64) {
                std.log.err("inkling2 rank {d}: {s}: the device's bytes hash {x:0>16}, the bytes read {x:0>16}", .{ p.rank, p.m.inv.tensors[p.m.p.ranks[p.rank].pieces[j.piece].tensor].name, h.final(), j.xxh64 });
                return error.UploadDiffers;
            }
        }
    }
};

/// A file as one reader thread holds it.
const Src = struct {
    fd: i32,
    /// The alignment its reads need (1: page-cache reads).
    a: u64,
};

const is_linux = @import("builtin").os.tag == .linux;

const Thread = struct {
    p: *Pool,
    s: ?cuda.Stream = null,
    bufs: [2]?cuda.HostBuffer = .{ null, null },
    evs: [2]?cuda.Event = .{ null, null },
    busy: [2]bool = .{ false, false },
    which: u1 = 0,
    files: []?Src = &.{},

    fn init(t: *Thread) !void {
        const p = t.p;
        t.files = try std.heap.page_allocator.alloc(?Src, p.m.inv.files.len);
        @memset(t.files, null);
        for (&t.bufs) |*b| b.* = try cuda.HostBuffer.alloc(p.d, p.opts.chunk);
        for (&t.evs) |*e| e.* = try cuda.Event.init(p.d, false);
        t.s = try cuda.Stream.init(p.d, true);
    }

    /// The stream first (it waits for the copies), then the buffers the copies read.
    fn deinit(t: *Thread) void {
        if (t.s) |*s| s.deinit();
        for (&t.evs) |*e| if (e.*) |*x| x.deinit();
        for (&t.bufs) |*b| if (b.*) |*x| x.free();
        for (t.files) |f| if (f) |x| close(x);
        if (t.files.len > 0) std.heap.page_allocator.free(t.files);
    }

    fn close(f: Src) void {
        if (comptime is_linux) _ = std.os.linux.close(f.fd);
    }

    fn file(t: *Thread, i: u16) !Src {
        if (t.files[i]) |f| return f;
        const p = t.p;
        const gpa = std.heap.page_allocator;
        const path = try std.fs.path.joinZ(gpa, &.{ p.dir, p.m.inv.files[i].name });
        defer gpa.free(path);
        if (comptime !is_linux) return error.Unsupported;
        const linux = std.os.linux;
        var direct = p.opts.direct;
        while (true) {
            const rc = linux.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = direct }, 0);
            switch (linux.errno(rc)) {
                .SUCCESS => {
                    t.files[i] = .{ .fd = @intCast(rc), .a = if (direct) direct_align else 1 };
                    return t.files[i].?;
                },
                .INTR => continue,
                .INVAL => if (direct) {
                    direct = false;
                    _ = p.buffered.fetchAdd(1, .acq_rel);
                    continue;
                } else return error.OpenFailed,
                .NOENT => return error.FileNotFound,
                .ACCES, .PERM => return error.AccessDenied,
                else => return error.OpenFailed,
            }
        }
    }

    /// `buf.len` bytes at `off`; at least `need` of them must be there.
    fn read(f: Src, buf: []u8, off: u64, need: usize) !void {
        if (comptime !is_linux) return error.Unsupported;
        const linux = std.os.linux;
        var got: usize = 0;
        while (got < buf.len) {
            const rc = linux.pread(f.fd, buf[got..].ptr, buf.len - got, @intCast(off + got));
            switch (linux.errno(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                else => return error.ReadFailed,
            }
            if (rc == 0) break;
            got += rc;
            if (got % f.a != 0) break; // a direct read's short tail: the end of the file
        }
        if (got < need) return error.ShortRead;
    }

    /// The other buffer, once the copy that last read it is done.
    fn buffer(t: *Thread) !struct { []u8, u1 } {
        const i = t.which;
        t.which ^= 1;
        if (t.busy[i]) try t.evs[i].?.synchronize();
        t.busy[i] = false;
        return .{ t.bufs[i].?.bytes, i };
    }

    fn job(t: *Thread, j: *Job) !void {
        const p = t.p;
        const f = try t.file(j.run.file);
        var h = std.hash.XxHash64.init(0);
        var it: Chunks = .{ .r = j.run, .a = f.a, .cap = p.opts.chunk };
        if (j.kind != .raw) {
            // a small piece: one chunk, converted on the host
            const c = it.next() orelse return error.UnexpectedInkling2Layout;
            if (c.rows != 1 or it.next() != null) return error.UnexpectedInkling2Layout;
            const buf = (try t.buffer())[0];
            try read(f, buf[0..c.span()], c.file_at, c.need());
            const raw = buf[c.data_at..][0..c.width];
            h.update(raw);
            j.xxh64 = h.final();
            try convert(p.d, j, raw);
            return;
        }
        const s = t.s.?;
        while (it.next()) |c| {
            const buf, const i = try t.buffer();
            for (0..c.reads) |k| try read(f, buf[k * c.read_len ..][0..c.read_len], c.file_at + k * c.read_step, c.need());
            for (0..c.rows) |k| h.update(buf[c.data_at + k * c.pitch ..][0..c.width]);
            const src = buf[c.data_at..].ptr;
            const dst = j.dst + c.dst;
            if (c.rows == 1 or c.pitch == c.width) {
                try p.d.check(p.d.api.cuMemcpyHtoDAsync_v2(dst, src, c.rows * c.width, s.handle), "cuMemcpyHtoDAsync");
            } else {
                const cp: cuda.abi.Memcpy2D = .{ .src_type = @intCast(cuda.abi.memory_type_host), .src_host = src, .src = 0, .src_pitch = c.pitch, .dst = dst, .dst_pitch = c.width, .width = c.width, .height = c.rows };
                try p.d.check(p.d.api.cuMemcpy2DAsync_v2(&cp, s.handle), "cuMemcpy2DAsync");
            }
            try t.evs[i].?.record(s);
            t.busy[i] = true;
        }
        j.xxh64 = h.final();
        if (f.a == 1) forget(f, j.run);
    }

    /// Drops a page-cache read run's pages once read, so a unified-memory device (GB10) does not hold the weights
    /// twice while loading. Advice only: errors are ignored. TF_INK2_KEEP_CACHE=1 keeps them.
    fn forget(f: Src, r: ink.plan.Run) void {
        if (comptime !is_linux) return;
        if (std.c.getenv("TF_INK2_KEEP_CACHE")) |v| if (v[0] == '1') return;
        const span = (r.count - 1) * r.stride + r.len;
        _ = std.os.linux.fadvise(f.fd, @intCast(r.offset), @intCast(span), std.os.linux.POSIX_FADV.DONTNEED);
    }
};

/// A small piece's bytes, converted, to its device array (or its scalar's place).
fn convert(d: *const cuda.Driver, j: *Job, raw: []const u8) !void {
    const gpa = std.heap.page_allocator;
    switch (j.kind) {
        .raw => unreachable,
        .scalar => j.out.?.* = @bitCast(std.mem.readInt(u32, raw[0..4], .little)),
        .gain, .router => {
            const view: cuda.DeviceBuffer = .{ .d = d, .ptr = j.dst, .len = if (j.dtype == .f32) raw.len else raw.len * 2, .borrowed = true };
            if (j.dtype == .f32) return view.upload(0, raw);
            const out = try gpa.alloc(f32, raw.len / 2);
            defer gpa.free(out);
            for (out, 0..) |*o, i| o.* = ink.st.bf16ToF32(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
            try view.upload(0, std.mem.sliceAsBytes(out));
        },
    }
}

/// The load's jobs as the plan's pieces are walked (in plan order): each device array allocated here, in the order
/// and sizes the pieces come in.
const Jobs = struct {
    gpa: std.mem.Allocator,
    w: *Weights,
    list: std.ArrayList(Job) = .empty,
    piece: u32 = 0,

    fn alloc(J: *Jobs, bytes: u64) !u64 {
        var buf = try cuda.DeviceBuffer.alloc(J.w.d, @intCast(bytes));
        errdefer buf.free();
        try J.w.own.append(J.gpa, buf);
        J.w.bytes += bytes;
        return buf.ptr;
    }

    /// The run's bytes as one dense device array.
    fn raw(J: *Jobs, r: ink.plan.Run) !u64 {
        const ptr = try J.alloc(r.bytes());
        try J.list.append(J.gpa, .{ .piece = J.piece, .run = r, .kind = .raw, .dst = ptr });
        return ptr;
    }

    /// A bf16 (or fp32) vector as fp32 on the device.
    fn gain(J: *Jobs, r: ink.plan.Run, dtype: ink.inventory.DType) !u64 {
        if (r.count != 1) return error.UnexpectedInkling2Layout;
        if (dtype != .f32 and dtype != .bf16) return error.UnexpectedInkling2Layout;
        const ptr = try J.alloc(if (dtype == .f32) r.len else r.len / 2 * 4);
        try J.list.append(J.gpa, .{ .piece = J.piece, .run = r, .kind = .gain, .dtype = dtype, .dst = ptr });
        return ptr;
    }

    /// The router [384, H] bf16 as fp32 (exact).
    fn router(J: *Jobs, r: ink.plan.Run) !u64 {
        if (r.count != 1 or r.len % 2 != 0) return error.UnexpectedInkling2Layout;
        const ptr = try J.alloc(r.len / 2 * 4);
        try J.list.append(J.gpa, .{ .piece = J.piece, .run = r, .kind = .router, .dst = ptr });
        return ptr;
    }

    fn scalar(J: *Jobs, r: ink.plan.Run, out: *f32) !void {
        if (r.len != 4 or r.count != 1) return error.UnexpectedInkling2Layout;
        try J.list.append(J.gpa, .{ .piece = J.piece, .run = r, .kind = .scalar, .out = out });
    }
};

/// Runs the jobs on the reader threads (and, asked, the recheck after), largest first.
fn readAll(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, m: *const ink.describe.Model, jobs: []Job, rank: u32) !void {
    if (jobs.len == 0) return;
    const opts = ReadOpts.env();
    var ctx: cuda.abi.Context = null;
    try d.check(d.api.cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
    const order = try gpa.alloc(u32, jobs.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, @as([]const Job, jobs), struct {
        fn lt(js: []const Job, x: u32, y: u32) bool {
            return js[x].run.bytes() > js[y].run.bytes();
        }
    }.lt);
    var total: u64 = 0;
    for (jobs) |j| total += j.run.bytes();
    const n = @min(opts.threads, jobs.len);
    const threads = try gpa.alloc(std.Thread, n);
    defer gpa.free(threads);
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(io).toNanoseconds();
    for (0..2) |pass| {
        if (pass == 1 and !opts.recheck) break;
        var p: Pool = .{ .d = d, .ctx = ctx, .dir = dir, .m = m, .jobs = jobs, .order = order, .opts = opts, .rank = rank, .recheck = pass == 1 };
        var started: usize = 0;
        for (threads) |*t| {
            t.* = std.Thread.spawn(.{}, Pool.work, .{&p}) catch |e| {
                p.fail(e);
                break;
            };
            started += 1;
        }
        for (threads[0..started]) |t| t.join();
        if (p.failed()) |e| return e;
        const s = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-9;
        const gib = @as(f64, @floatFromInt(total)) / (1 << 30);
        if (pass == 0) {
            std.log.info("inkling2 rank {d}: read, hashed and uploaded {d} pieces, {d:.2} GiB in {d:.1} s ({d:.2} GiB/s; {d} threads, {s} reads, {d} MiB buffers{s})", .{ rank, jobs.len, gib, s, gib / @max(s, 1e-9), n, if (opts.direct) "direct" else "page-cache", opts.chunk >> 20, if (p.buffered.load(.acquire) > 0) "; some files through the page cache" else "" });
        } else std.log.info("inkling2 rank {d}: recheck: every uploaded piece's device bytes hash as read ({d:.1} s since the load began)", .{ rank, s });
    }
}

fn layerOf(c: *const ink.config.Config, l: u32, split: ink.plan.Split, rank: u32, format: Format) Layer {
    const t = &c.text;
    return .{
        .index = l,
        .global = t.global[l],
        .boundary = t.blockBoundary(l),
        .hq = split.qSpan(c, rank).len(),
        .hk = split.kvSpan(c, rank).len(),
        .nq = if (l == 0) 0 else if (t.global[l]) 3 else 1,
        .experts = .{
            .format = format,
            .lo = split.expertSpan(rank).lo,
            .hi = split.expertSpan(rank).hi,
            .split = if (split.layout == .split) split.world else 1,
            .part = if (split.layout == .split) rank else 0,
            .n13 = split.rowSpan(2 * t.moe_intermediate, rank).len(),
            .n2 = split.rowSpan(t.hidden, rank).len(),
        },
    };
}

/// Rank `want.rank`'s weights of layers [want.lo, want.hi) of the checked model `m` (read from `dir`), with the
/// embedding and the head when asked; `prepare` hooks run per layer after its upload. The plan's pieces are walked in
/// order first (each device array allocated, nothing read), then reader threads read, hash and upload them (ReadOpts);
/// the digests come out in plan order, as the pieces are.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, m: *const ink.describe.Model, want: Want, prepare: []const Prepare) !Weights {
    const c = &m.c;
    if (want.lo >= want.hi or want.hi > c.text.layers or want.rank >= m.p.split.world) return error.BadLayerRange;
    var w: Weights = .{ .gpa = gpa, .d = d, .first = want.lo, .layers = try gpa.alloc(Layer, want.hi - want.lo), .vocab = c.text.vocab, .unpadded_vocab = c.text.unpadded_vocab, .vocab_rows = ink.plan.canon(c.text.vocab, want.rank, m.p.split.world) };
    errdefer w.deinit();
    for (w.layers, want.lo..) |*l, i| {
        const li: u32 = @intCast(i);
        var buf: [96]u8 = undefined;
        const fmt: Format = if (m.inv.find(try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.w13_weight_amax", .{li})) != null) .nvfp4 else .mxfp8;
        l.* = layerOf(c, li, m.p.split, want.rank, fmt);
    }
    var J: Jobs = .{ .gpa = gpa, .w = &w };
    defer J.list.deinit(gpa);
    const rank = &m.p.ranks[want.rank];
    const llm = "model.llm.";
    for (rank.pieces, 0..) |piece, pi| {
        J.piece = @intCast(pi);
        const t = m.inv.tensors[piece.tensor];
        const role = m.roles[piece.tensor];
        if (!std.mem.startsWith(u8, t.name, llm)) continue; // vision tower, drafter
        const n = t.name[llm.len..];
        if (role.layer) |li| {
            if (li < want.lo or li >= want.hi) continue;
            const l = &w.layers[li - want.lo];
            if (std.mem.startsWith(u8, n, "attn_res.")) {
                if (std.mem.startsWith(u8, n, "attn_res.attn_queries.")) l.attn_query = try J.raw(piece.run) else l.mlp_query = try J.raw(piece.run);
                continue;
            }
            const leaf = n[std.mem.indexOfScalarPos(u8, n, "layers.".len, '.').? + 1 ..];
            try assign(&J, l, leaf, piece.run, t.dtype);
            continue;
        }
        if (std.mem.eql(u8, n, "embed.weight")) {
            if (want.embed) {
                w.embed = try J.raw(piece.run);
                w.embed_rows = .{ .lo = 0, .hi = w.vocab };
            }
        } else if (std.mem.eql(u8, n, "embed_norm.weight")) {
            if (want.embed) w.embed_norm = try J.gain(piece.run, t.dtype);
        } else if (std.mem.eql(u8, n, "unembed.weight")) {
            if (piece.run.count != 1 or piece.run.len != @as(u64, w.vocab_rows.len()) * hidden * 2) return error.UnexpectedInkling2Layout;
            if (want.head) w.unembed = try J.raw(piece.run);
        } else if (std.mem.eql(u8, n, "norm.weight")) {
            if (want.head or want.final) w.norm = try J.gain(piece.run, t.dtype);
        } else if (std.mem.eql(u8, n, "attn_res.output_query")) {
            if (want.head or want.final) w.output_query = try J.raw(piece.run);
        } else return error.UnknownInkling2Tensor;
    }
    try readAll(gpa, io, d, dir, m, J.list.items, want.rank);
    if (want.digests) |out| for (J.list.items) |j| if (j.run.bytes() > 0) try out.append(gpa, .{ .piece = j.piece, .xxh64 = j.xxh64 });
    for (w.layers) |*l| try complete(l);
    if (want.embed and (w.embed == 0 or w.embed_norm == 0)) return error.MissingInkling2Tensor;
    if (want.head and (w.unembed == 0 or w.norm == 0 or w.output_query == 0)) return error.MissingInkling2Tensor;
    for (w.layers) |*l| for (prepare) |p| try p(&w, l);
    return w;
}

const How = enum { raw, gain, router };

fn assign(J: *Jobs, l: *Layer, leaf: []const u8, r: ink.plan.Run, dtype: ink.inventory.DType) !void {
    const E = &l.experts;
    const table = [_]struct { []const u8, *u64, How }{
        .{ "attn.wq_du.weight", &l.wq, .raw },                     .{ "attn.wk_dv.weight", &l.wk, .raw },
        .{ "attn.wv_dv.weight", &l.wv, .raw },                     .{ "attn.wr_du.weight", &l.wr, .raw },
        .{ "attn.gate_attn.weight", &l.wg, .raw },                 .{ "attn.attn_sink_proj.weight", &l.wsink, .raw },
        .{ "attn.wo_ud.weight", &l.wo, .raw },                     .{ "attn.q_l2_clip.scale", &l.q_scale, .raw },
        .{ "attn.rel_l2_clip.scale", &l.rel_scale, .raw },         .{ "attn.rel_logits_proj.proj", &l.rel_proj, .raw },
        .{ "attn.k_sconv.weight", &l.k_conv, .raw },               .{ "attn.k_sconv.static_w", &l.k_conv, .raw },
        .{ "attn.v_sconv.weight", &l.v_conv, .raw },               .{ "attn.v_sconv.static_w", &l.v_conv, .raw },
        .{ "attn.k_sconv.weight_proj", &l.k_wproj, .raw },         .{ "attn.v_sconv.weight_proj", &l.v_wproj, .raw },
        .{ "attn_norm.weight", &l.in_norm[0], .gain },             .{ "q_input_norm.weight", &l.in_norm[0], .gain },
        .{ "k_input_norm.weight", &l.in_norm[1], .gain },          .{ "v_input_norm.weight", &l.in_norm[2], .gain },
        .{ "mlp_norm.weight", &l.mlp_norm, .gain },                .{ "attn_sconv.weight", &l.attn_conv, .raw },
        .{ "mlp_sconv.weight", &l.mlp_conv, .raw },                .{ "mlp.gate.weight", &l.router, .router },
        .{ "mlp.gate.bias", &l.router_bias, .raw },                .{ "mlp.experts.w13_weight", &E.w13, .raw },
        .{ "mlp.experts.w13_weight_scale", &E.w13_scale, .raw },   .{ "mlp.experts.w2_weight", &E.w2, .raw },
        .{ "mlp.experts.w2_weight_scale", &E.w2_scale, .raw },     .{ "mlp.experts.w13_weight_amax", &E.w13_amax, .raw },
        .{ "mlp.experts.w2_weight_amax", &E.w2_amax, .raw },
    };
    for (table) |e| if (std.mem.eql(u8, leaf, e[0])) {
        if (E.split > 1 and std.mem.startsWith(u8, leaf, "mlp.experts.")) try splitRun(E, leaf, r);
        e[1].* = switch (e[2]) {
            .raw => try J.raw(r),
            .gain => try J.gain(r, dtype),
            .router => if (@import("cuda_moe.zig").routerBf16()) try J.raw(r) else try J.router(r),
        };
        return;
    };
    if (std.mem.eql(u8, leaf, "mlp.experts.w13_weight_outer_scale_denominator")) {
        try J.scalar(r, &E.den13);
    } else if (std.mem.eql(u8, leaf, "mlp.experts.w2_weight_outer_scale_denominator")) {
        try J.scalar(r, &E.den2);
    } else return error.UnknownInkling2Tensor;
}

/// A split rank's expert piece is what the kernels index: the amax whole (an fp32 a expert), every other tensor one
/// strided run an expert of its n13 (w13) or n2 (w2) rows, `row` bytes a row (codes or values, then scales).
fn splitRun(E: *const Experts, leaf: []const u8, r: ink.plan.Run) !void {
    const n = E.hi - E.lo;
    const nv = E.format == .nvfp4;
    const amax = std.mem.endsWith(u8, leaf, "_amax");
    const w13 = std.mem.indexOf(u8, leaf, "w13") != null;
    const K: u64 = if (w13) hidden else 2304;
    const rows: u64 = if (w13) E.n13 else E.n2;
    const row: u64 = if (std.mem.endsWith(u8, leaf, "_scale")) (if (nv) K / 16 else K / 32) else (if (nv) K / 2 else K);
    const ok = if (amax) r.bytes() == @as(u64, n) * 4 else r.count == n and r.len == rows * row;
    if (!ok) {
        std.log.err("inkling2: split experts: {s} is held as {d} runs of {d} bytes, not this rank's slice", .{ leaf, r.count, r.len });
        return error.UnexpectedInkling2Layout;
    }
}

/// Every tensor the layer needs is there.
fn complete(l: *const Layer) !void {
    const E = &l.experts;
    const need = [_]u64{ l.mlp_norm, l.mlp_query, l.wq, l.wk, l.wv, l.wr, l.wg, l.wsink, l.wo, l.q_scale, l.rel_scale, l.rel_proj, l.k_conv, l.v_conv, l.attn_conv, l.mlp_conv, l.router, l.router_bias, E.w13, E.w13_scale, E.w2, E.w2_scale, l.in_norm[0] };
    for (need, 0..) |p, i| if (p == 0) {
        std.log.err("inkling2: layer {d} is missing tensor #{d} of its set", .{ l.index, i });
        return error.MissingInkling2Tensor;
    };
    if (l.nq > 0 and l.attn_query == 0) return error.MissingInkling2Tensor;
    if (l.global and (l.in_norm[1] == 0 or l.in_norm[2] == 0 or l.k_wproj == 0 or l.v_wproj == 0)) return error.MissingInkling2Tensor;
    if (E.format == .nvfp4 and (E.w13_amax == 0 or E.w2_amax == 0 or E.den13 == 0 or E.den2 == 0)) return error.MissingInkling2Tensor;
}

test "inkling2 cuda weights: a layer's facts at W = 1 and W = 4" {
    const c = try ink.config.parse(std.testing.allocator, ink.config.fixture, null);
    const one = try ink.plan.Split.even(&c, 1, null);
    const l0 = layerOf(&c, 0, one, 0, .nvfp4);
    try std.testing.expect(l0.boundary and !l0.global and l0.nq == 0 and l0.hq == 48 and l0.hk == 8);
    try std.testing.expectEqual(@as(u32, 384), l0.experts.count());
    const l5 = layerOf(&c, 5, one, 0, .nvfp4);
    try std.testing.expect(l5.global and !l5.boundary and l5.nq == 3);
    const four = try ink.plan.Split.even(&c, 4, null);
    const l47 = layerOf(&c, 47, four, 2, .mxfp8);
    try std.testing.expect(l47.global and l47.hq == 12 and l47.hk == 2 and l47.experts.lo == 192 and l47.experts.hi == 288);
    try std.testing.expect(l47.experts.n13 == 4608 and l47.experts.n2 == 4608 and l47.experts.split == 1);
    const sp = try ink.plan.Split.sliced(&c, 4, null);
    const s3 = layerOf(&c, 3, sp, 3, .nvfp4);
    try std.testing.expect(s3.experts.lo == 0 and s3.experts.hi == 384 and s3.experts.split == 4 and s3.experts.part == 3);
    try std.testing.expect(s3.experts.n13 == 1152 and s3.experts.n2 == 1152);
}

test "inkling2 cuda weights: a run's chunks read, hash and place exactly its rows (direct and page-cache reads)" {
    const gpa = std.testing.allocator;
    const file = try gpa.alloc(u8, 2 << 20);
    defer gpa.free(file);
    for (file, 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 7);
    const R = ink.plan.Run;
    const runs = [_]R{
        .{ .file = 0, .offset = 0, .len = 1000, .stride = 1000, .count = 1 },
        .{ .file = 0, .offset = 12345, .len = 300000, .stride = 0, .count = 1 }, // longer than a small chunk
        .{ .file = 0, .offset = 777, .len = 3072, .stride = 12288, .count = 60 }, // rows close together
        .{ .file = 0, .offset = 4097, .len = 5000, .stride = 200000, .count = 4 }, // far apart, unaligned stride
        .{ .file = 0, .offset = 100, .len = 20000, .stride = 4096 * 40, .count = 5 }, // far apart, aligned stride
        .{ .file = 0, .offset = 4096, .len = 70000, .stride = 4096 * 50, .count = 3 }, // long rows
        .{ .file = 0, .offset = 3, .len = 1, .stride = 4096 * 33, .count = 9 },
    };
    for (runs) |r| for ([_]u64{ 1, direct_align }) |a| for ([_]usize{ 64 << 10, 1 << 20 }) |cap| {
        const out = try gpa.alloc(u8, r.bytes());
        defer gpa.free(out);
        const buf = try gpa.alloc(u8, cap);
        defer gpa.free(buf);
        var h = std.hash.XxHash64.init(0);
        var at: u64 = 0;
        var it: Chunks = .{ .r = r, .a = a, .cap = cap };
        while (it.next()) |c| {
            try std.testing.expect(c.span() <= cap and c.file_at % a == 0 and c.read_len % a == 0 and c.rows > 0);
            try std.testing.expect(c.reads == 1 or (c.reads == c.rows and c.read_step % a == 0 and c.pitch == c.read_len));
            try std.testing.expect(c.need() <= c.read_len and c.dst == at);
            for (0..c.reads) |k| @memcpy(buf[k * c.read_len ..][0..c.read_len], file[c.file_at + k * c.read_step ..][0..c.read_len]);
            for (0..c.rows) |k| {
                const row = buf[c.data_at + k * c.pitch ..][0..c.width];
                @memcpy(out[c.dst + k * c.width ..][0..c.width], row);
                h.update(row);
            }
            at += c.rows * c.width;
        }
        try std.testing.expectEqual(r.bytes(), at);
        var want = std.hash.XxHash64.init(0);
        for (0..r.count) |j| {
            const row = file[r.offset + j * r.stride ..][0..r.len];
            try std.testing.expectEqualSlices(u8, row, out[j * r.len ..][0..r.len]);
            want.update(row);
        }
        try std.testing.expectEqual(want.final(), h.final());
    };
}
