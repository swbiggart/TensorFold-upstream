// Inkling-2's attention group (G-attn): the clips, value norm and log scaling of the heads, and the sliding-window
// and global attention with the relative-position bias, the sink and the output gate. The entry points, shapes and
// math are in zig/src/families/inkling2/cuda_attn.zig; their kernels go here.
//
// The math follows Transformers' modeling_inkling2.py (Apache-2.0): Inkling2L2Clip, Inkling2ScaledL2Clip,
// Inkling2UnweightedRMSNorm, Inkling2RelativeLogits, the log scaling of Inkling2FullAttention and
// eager_attention_forward, with the eager form's bf16 roundings kept: s = bf16(bf16(bf16(q.k) * scale) + bias),
// p = bf16(softmax over the keys and the sink), o = bf16(sum p v), out = bf16(o * sigmoid(gate)). Products and sums
// are fp32 in a fixed order (sequential fmaf chains), so a row's bits never depend on the forward's other rows.
//
// Row invariance of the attention (`core`): keys are visited in tiles of KT positions and splits of SPLIT positions,
// both aligned to absolute positions, and every sum is two-level (tiles in a split, then splits in order) in all
// three launch forms: one block per row block and key head over all splits (tf_ink2_attn_core), or one block per
// split (tf_ink2_attn_stats, tf_ink2_attn_pv) and an ordered sum over a row's splits (tf_ink2_attn_merge). A tile or
// split with no key of a row's window leaves that row's sums unchanged exactly, so the forms give the same bits.
//
// Captured decode steps (a CUDA graph replayed at every position): `posp`, when not null, holds the forward's first
// position on the device and replaces `pos0`; the split form then runs with s_first = 0 over a grid of every split
// the slot can hold, and blocks of splits outside the rows' windows return at once (the arithmetic is unchanged).
// Sliding layers on the row kernels (tf_ink2_attn_rstats / _rpv, 1 .. 8 rows) instead take s_first < 0: the first
// split is derived from the device position and the grid holds only the splits a window can touch. Global layers on
// the row kernels keep s_first = 0 with a grid of at most GRID splits (cuda_attn.zig grid_splits): a block takes
// split blockIdx.z, then blockIdx.z + gridDim.z, ... up to the row's last, so the work follows the device position
// and not the slot's capacity (the rounds are the bucket, chosen on the device; every split's arithmetic unchanged).
//
// The second pass needs each row's softmax pair (m, z): (sink, 1) and the row's splits' pairs folded in order. A
// forward over many splits takes it from tf_ink2_attn_fold (once a row and query head, `mz`) instead of folding it in
// every split's block, which made the pass quadratic in the depth; the same folds in the same order, so the same bits.
//
// Paged global layers (the shared KV page pool): every kernel that turns a position into a key or value address is
// built twice from one source, templated on where the rows live (`Ring` or `Paged` below), and the host picks the form
// by the layer's storage (cuda_attn.zig Kv.pages): the ring form is the kernel by its old name, the paged form the same
// name with `_pg` and two more arguments (the slot's page table, the pool's page stride). A page holds SPLIT positions,
// so a split is a page: a block reads its split's page id once, before the split's keys (in the fused core, the next
// split's while it works on this one; the row passes read all of a block's ids at its start), and a key's address is
// the page's base plus (p % SPLIT) rows, with no table read and no branch a key. Only addresses change: every load reads the bytes the cache write put at that position, and every
// expression and its order stay as they are, so the two forms give the same bits.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

// The module's presence check (cuda_kernels.zig Group): writes the group's number.
extern "C" __global__ void tf_ink2_attn_probe(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 1;
}

namespace ink2_attn {

constexpr int D = 128;                  // head dim
constexpr int DREL = 16;                // relative-state width
constexpr int EXTENT = 1024;            // relative-bias distances 0 .. EXTENT - 1 (both layer kinds)
constexpr int G = 6;                    // query heads a key head
constexpr int RB = 8;                   // rows a block
constexpr int NV = RB * G;              // query vectors a block
constexpr int KT = 32;                  // keys a tile (absolute positions 32j .. 32j + 31)
constexpr int SPLIT = 256;              // keys a split (absolute), the outer level of every fold
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int SPT = NV * KT / THREADS;  // scores a thread
constexpr int OPT = NV * D / THREADS;   // outputs a thread
constexpr int KW = D / 2 + 1;           // 32-bit words a key row in shared memory (65: a lane a bank)
static_assert(SPLIT % KT == 0 && SPT * THREADS == NV * KT && OPT * THREADS == NV * D && THREADS == 2 * D, "shape");

// Python floats as torch's fp32 opmath scalars.
__device__ __forceinline__ float scale() { return (float)0.08838834764831845; }  // 128 ** -0.5
__device__ __forceinline__ float qk_thr() { return (float)22.627416997969522; }
__device__ __forceinline__ float rel_thr() { return (float)8.0; }
__device__ __forceinline__ float eps() { return (float)1e-6; }

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }

// torch.sigmoid on fp32
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Where a layer's key and value rows live (`kc`, `vc` and a row's `ld` elements as each kernel has them):
//   Ring: position p at row p % slots of kc / vc (sliding layers' rings; a contiguous global layer, slots >= the
//     context). `at` is 0, so the ring form's addresses are the expressions they always were.
//   Paged: position p at row p % SPLIT of page pages[p / SPLIT]; page n's rows of this layer start `page` elements
//     after page n - 1's (kc / vc point at page 0's: the pool's base plus the layer's run in a page).
// `at(s)` is split s's page offset in elements (one table read), `row(p)` a key's row in its split's page or ring.
struct Ring {
    static constexpr bool table = false;  // a split's base needs no load
    int slots;
    __device__ __forceinline__ size_t at(int) const { return 0; }
    __device__ __forceinline__ int row(int p) const { return p % slots; }
};

struct Paged {
    static constexpr bool table = true;  // a split's base waits on its page id
    const int* pages;
    long long page;
    __device__ __forceinline__ size_t at(int s) const { return (size_t)__ldg(pages + s) * (size_t)page; }
    __device__ __forceinline__ int row(int p) const { return p & (SPLIT - 1); }
};

// Folds a split's (max, sum of exp(s - max)) into a row's running pair; an empty split (zs == 0) changes nothing.
__device__ __forceinline__ void fold(float& m, float& z, float ms, float zs) {
    if (!(zs > 0.0f)) return;
    const float mn = fmaxf(m, ms);
    z = __fadd_rn(__fmul_rn(z, expf(m - mn)), __fmul_rn(zs, expf(ms - mn)));
    m = mn;
}

struct Smem {
    bf162 q[NV][D / 2];
    uint32_t k[KT][KW];
    bf162 v[KT][D / 2];
    float rel[NV][DREL];
    float s[NV][KT + 1];  // scores, then probabilities
    float m[NV], z[NV];   // the row's softmax max and sum (pass 2), or its fold over splits (pass 1)
    float ms[NV], zs[NV]; // the current split's
    int pos[RB], lo[RB];
};

struct Geo {
    int R, pos0, window, hq, hk, slots;
    int row0, kvh, klo, khi;  // the block's rows and key head; keys of its rows' windows

    __device__ int lo(int p) const { return window == 0 ? 0 : max(0, p - (window - 1)); }

    __device__ void init(int R_, int pos0_, int window_, int hq_, int hk_, int slots_) {
        R = R_, pos0 = pos0_, window = window_, hq = hq_, hk = hk_, slots = slots_;
        row0 = blockIdx.x * RB;
        kvh = blockIdx.y;
        klo = lo(pos0 + row0);
        khi = pos0 + min(row0 + RB, R) - 1;
    }

    __device__ int vrow(int vec) const { return row0 + vec / G; }
    __device__ int vhead(int vec) const { return kvh * G + vec % G; }
};

// The block's query vectors, relative states and row windows; m, z start at (sink, 1) when `sink` is given.
__device__ void setup(Smem& sm, const Geo& g, const bf16* q, const bf16* rel, const bf16* sink) {
    const int tid = threadIdx.x;
    for (int i = tid; i < NV * (D / 2); i += THREADS) {
        const int vec = i / (D / 2), w = i % (D / 2), t = g.vrow(vec);
        bf162 x = __floats2bfloat162_rn(0.0f, 0.0f);
        if (t < g.R) x = reinterpret_cast<const bf162*>(q + ((size_t)t * g.hq + g.vhead(vec)) * D)[w];
        sm.q[vec][w] = x;
    }
    for (int i = tid; i < NV * DREL; i += THREADS) {
        const int vec = i / DREL, j = i % DREL, t = g.vrow(vec);
        sm.rel[vec][j] = t < g.R ? f32(rel[((size_t)t * g.hq + g.vhead(vec)) * DREL + j]) : 0.0f;
    }
    if (tid < RB) {
        const int p = g.pos0 + g.row0 + tid;
        sm.pos[tid] = p;
        sm.lo[tid] = g.lo(p);
    }
    if (tid < NV) {
        const int t = g.vrow(tid);
        sm.m[tid] = (sink != nullptr && t < g.R) ? f32(sink[(size_t)t * g.hq + g.vhead(tid)]) : -INFINITY;
        sm.z[tid] = sink != nullptr ? 1.0f : 0.0f;
        sm.ms[tid] = -INFINITY;
        sm.zs[tid] = 0.0f;
    }
}

// Key (and value) rows of positions k0 .. k0 + KT - 1 from the stream's slots; positions outside the block's
// windows read as zeros (never scored, and a zero value row keeps p * v finite). kc, vc: the tile's split's base
// (its page's, paged).
template <class A>
__device__ void load_kv(Smem& sm, const Geo& g, const A& a, const bf16* kc, const bf16* vc, int k0) {
    const size_t ld = (size_t)g.hk * D;
    for (int i = threadIdx.x; i < KT * (D / 8); i += THREADS) {
        const int r = i / (D / 8), c = i % (D / 8), p = k0 + r;
        const bool in = p >= g.klo && p <= g.khi;
        const size_t at = (size_t)(in ? a.row(p) : 0) * ld + (size_t)g.kvh * D;
        uint4 x = make_uint4(0, 0, 0, 0);
        if (in) x = reinterpret_cast<const uint4*>(kc + at)[c];
        uint32_t* dk = &sm.k[r][c * 4];
        dk[0] = x.x, dk[1] = x.y, dk[2] = x.z, dk[3] = x.w;
        if (vc != nullptr) {
            uint4 y = make_uint4(0, 0, 0, 0);
            if (in) y = reinterpret_cast<const uint4*>(vc + at)[c];
            reinterpret_cast<uint4*>(&sm.v[r][0])[c] = y;
        }
    }
}

// Scores of the tile at k0 for every query vector: bf16(bf16(bf16(q.k) * scale) + bias), -inf outside a row's window;
// with `prob`, the probabilities bf16(exp(s - m) / z) instead (0 outside the window).
__device__ void scores(Smem& sm, const Geo& g, const bf16* proj, int k0, bool prob) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[SPT];
#pragma unroll
    for (int j = 0; j < SPT; ++j) acc[j] = 0.0f;
    const uint32_t* kr = sm.k[lane];
#pragma unroll 4
    for (int w = 0; w < D / 2; ++w) {
        uint32_t kw = kr[w];
        const float2 kf = __bfloat1622float2(*reinterpret_cast<const bf162*>(&kw));
#pragma unroll
        for (int j = 0; j < SPT; ++j) {
            const float2 qf = __bfloat1622float2(sm.q[warp + WARPS * j][w]);
            acc[j] = fmaf(qf.x, kf.x, acc[j]);
            acc[j] = fmaf(qf.y, kf.y, acc[j]);
        }
    }
    const int kp = k0 + lane;
#pragma unroll
    for (int j = 0; j < SPT; ++j) {
        const int vec = warp + WARPS * j, r = vec / G;
        float x = prob ? 0.0f : -INFINITY;
        if (g.row0 + r < g.R && kp >= sm.lo[r] && kp <= sm.pos[r]) {
            float s = bf(__fmul_rn(bf(acc[j]), scale()));
            const int dist = sm.pos[r] - kp;
            if (dist < EXTENT) {
                float b = 0.0f;
#pragma unroll
                for (int i = 0; i < DREL; ++i) b = fmaf(sm.rel[vec][i], f32(proj[i * EXTENT + dist]), b);
                s = bf(__fadd_rn(s, bf(b)));
            }
            x = prob ? bf(__fdiv_rn(expf(s - sm.m[vec]), sm.z[vec])) : s;
        }
        sm.s[vec][lane] = x;
    }
}

// The tile's scores into each vector's split pair (ms, zs): max, then the sum of exp(s - max) in key order.
__device__ void tile_stats(Smem& sm) {
    const int v = threadIdx.x;
    if (v >= NV) return;
    float tm = -INFINITY;
#pragma unroll 8
    for (int kk = 0; kk < KT; ++kk) tm = fmaxf(tm, sm.s[v][kk]);
    if (tm == -INFINITY) return;
    const float mn = fmaxf(sm.ms[v], tm);
    float z = __fmul_rn(sm.zs[v], expf(sm.ms[v] - mn));
    for (int kk = 0; kk < KT; ++kk) {
        const float x = sm.s[v][kk];
        if (x != -INFINITY) z = __fadd_rn(z, expf(x - mn));
    }
    sm.ms[v] = mn;
    sm.zs[v] = z;
}

// o[j] += p v over the tile, for this thread's column d and vectors vg * OPT .. vg * OPT + OPT - 1.
__device__ void pv(const Smem& sm, float (&o)[OPT]) {
    const int d = threadIdx.x & (D - 1), vg = threadIdx.x / D;
    const bf16* vcol = reinterpret_cast<const bf16*>(&sm.v[0][0]) + d;
    for (int kk = 0; kk < KT; ++kk) {
        const float vf = f32(vcol[kk * D]);
#pragma unroll
        for (int j = 0; j < OPT; ++j) o[j] = fmaf(sm.s[vg * OPT + j][kk], vf, o[j]);
    }
}

// The first tile start and last key of split s within the block's keys.
__device__ __forceinline__ void split_keys(const Geo& g, int s, int& k0, int& k1) {
    k0 = max(s * SPLIT, (g.klo / KT) * KT);
    k1 = min(s * SPLIT + SPLIT - 1, g.khi);
}

// Pass 1 over split s: the split pair (ms, zs) of every vector. `off`: the split's page offset (a.at(s)).
template <class A>
__device__ void split_stats(Smem& sm, const Geo& g, const A& a, const bf16* kc, const bf16* proj, int s, size_t off) {
    int k0, k1;
    split_keys(g, s, k0, k1);
    for (int kt = k0; kt <= k1; kt += KT) {
        load_kv(sm, g, a, kc + off, nullptr, kt);
        __syncthreads();
        scores(sm, g, proj, kt, false);
        __syncthreads();
        tile_stats(sm);
        __syncthreads();
    }
}

// Pass 2 over split s: os += p v.
template <class A>
__device__ void split_pv(Smem& sm, const Geo& g, const A& a, const bf16* kc, const bf16* vc, const bf16* proj, int s,
                         size_t off, float (&os)[OPT]) {
    int k0, k1;
    split_keys(g, s, k0, k1);
    for (int kt = k0; kt <= k1; kt += KT) {
        load_kv(sm, g, a, kc + off, vc + off, kt);
        __syncthreads();
        scores(sm, g, proj, kt, true);
        __syncthreads();
        pv(sm, os);
        __syncthreads();
    }
}

// out = bf16(f32(bf16(o)) * sigmoid(f32(gate))): eager's PV rounding, then Inkling2's output gate.
__device__ __forceinline__ bf16 gated(float o, bf16 gate) {
    return __float2bfloat16_rn(__fmul_rn(bf(o), sigmoid(f32(gate))));
}

}  // namespace ink2_attn

using namespace ink2_attn;

// L5-L9 for every row: grid (R, ceil((2 hq + 2 hk) / 8)), 256 threads, a warp an item (query head, key head, value
// head, relative head). q' = clip(q * q_scale, 22.63), k' = clip(kc, 22.63), v' = vc * rsqrt(mean(vc^2) + 1e-6),
// rel' = clip(rel * rel_scale, 8); global layers then q' = bf16(q' * tau), rel' = bf16(rel' * tau).
// clip(x, thr) = bf16(x * min(1, (1 / (||x|| + 1e-6)) * thr)) in fp32 (torch's `thr / t` is t.reciprocal() * thr).
extern "C" __global__ void __launch_bounds__(256) tf_ink2_attn_heads(
    const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc, const bf16* __restrict__ rel,
    const bf16* __restrict__ q_scale, const bf16* __restrict__ rel_scale, bf16* __restrict__ qo, bf16* __restrict__ ko,
    bf16* __restrict__ vo, bf16* __restrict__ relo, int hq, int hk, int pos0, int global, const int* __restrict__ posp) {
    if (posp != nullptr) pos0 = *posp;
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const int item = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (item >= 2 * hq + 2 * hk) return;
    float tau = 1.0f;
    if (global) {
        // 1 + 0.1 * log(clamp((p + 1) / 128000, min=1)): torch divides by a scalar as a multiply by its reciprocal
        float r = __fmul_rn((float)(pos0 + t + 1), 1.0f / 128000.0f);
        r = r < 1.0f ? 1.0f : r;
        tau = __fadd_rn(1.0f, __fmul_rn(logf(r), (float)0.1));
    }
    if (item < hq + hk || item >= hq + 2 * hk) {
        const bool is_rel = item >= hq + 2 * hk;
        const bool is_q = item < hq;
        const int h = is_rel ? item - hq - 2 * hk : is_q ? item : item - hq;
        const int n = is_rel ? DREL : D;
        const int per = n / 32 > 0 ? n / 32 : 1;  // elements a lane: 4 (D) or 1 (lanes 0..15 of DREL)
        const bf16* src = is_rel ? rel + ((size_t)t * hq + h) * DREL : is_q ? q + ((size_t)t * hq + h) * D : kc + ((size_t)t * hk + h) * D;
        const bf16* sc = is_rel ? rel_scale + (size_t)h * DREL : is_q ? q_scale + (size_t)h * D : nullptr;
        bf16* dst = is_rel ? relo + ((size_t)t * hq + h) * DREL : is_q ? qo + ((size_t)t * hq + h) * D : ko + ((size_t)t * hk + h) * D;
        float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ss = 0.0f;
        const int c0 = lane * per;
        if (c0 < n) {
            for (int i = 0; i < per; ++i) {
                x[i] = f32(src[c0 + i]);
                if (sc != nullptr) x[i] = __fmul_rn(x[i], f32(sc[c0 + i]));
                ss = __fadd_rn(ss, __fmul_rn(x[i], x[i]));
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, off));
        const float thr = is_rel ? rel_thr() : qk_thr();
        float f = __fmul_rn(__fdiv_rn(1.0f, __fadd_rn(sqrtf(ss), eps())), thr);
        f = f > 1.0f ? 1.0f : f;  // clamp(max=1.0), a NaN passes
        if (c0 < n) {
            for (int i = 0; i < per; ++i) {
                float y = bf(__fmul_rn(x[i], f));
                if (global && !(item >= hq && item < hq + hk)) y = bf(__fmul_rn(y, tau));
                dst[c0 + i] = __float2bfloat16_rn(y);
            }
        }
        return;
    }
    // value head: Inkling2UnweightedRMSNorm on fp32 values (mean as sum * 1/128, rsqrt as torch.rsqrt)
    const int h = item - hq - hk;
    const bf16* src = vc + ((size_t)t * hk + h) * D;
    bf16* dst = vo + ((size_t)t * hk + h) * D;
    float x[4], ss = 0.0f;
    for (int i = 0; i < 4; ++i) {
        x[i] = f32(src[lane * 4 + i]);
        ss = __fadd_rn(ss, __fmul_rn(x[i], x[i]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, off));
    const float r = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / (float)D), eps()));
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __float2bfloat16_rn(__fmul_rn(x[i], r));
}

// L10, L12-L14 in one launch: grid (ceil(R / RB), hk), THREADS threads; a block holds RB rows' G query heads of one
// key head and runs both passes over every split of its rows' windows. Keys: position p at slot p % slots of
// kc, vc [slots, hk * D] (tf_ink2_attn_core_pg: at row p % SPLIT of page pages[p / SPLIT]). out bf16 [R, hq * D].
// Paged, each split's page id is read while the split before it runs.
template <class A>
__device__ __forceinline__ void core_body(const bf16* __restrict__ q, const bf16* __restrict__ rel,
                                          const bf16* __restrict__ sink, const bf16* __restrict__ gate,
                                          const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                          const bf16* __restrict__ proj, bf16* __restrict__ out, int R, int pos0,
                                          int window, int hq, int hk, int slots, const A& a) {
    __shared__ __align__(16) Smem sm;
    Geo g;
    g.init(R, pos0, window, hq, hk, slots);
    setup(sm, g, q, rel, sink);
    __syncthreads();
    const int s0 = g.klo / SPLIT, s1 = g.khi / SPLIT;
    size_t off = a.at(s0);
    for (int s = s0; s <= s1; ++s) {
        const size_t next = s < s1 ? a.at(s + 1) : 0;
        split_stats(sm, g, a, kc, proj, s, off);
        if (threadIdx.x < NV) {
            fold(sm.m[threadIdx.x], sm.z[threadIdx.x], sm.ms[threadIdx.x], sm.zs[threadIdx.x]);
            sm.ms[threadIdx.x] = -INFINITY;
            sm.zs[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        off = next;
    }
    float o[OPT], os[OPT];
#pragma unroll
    for (int j = 0; j < OPT; ++j) o[j] = 0.0f, os[j] = 0.0f;
    off = a.at(s0);
    for (int s = s0; s <= s1; ++s) {
        const size_t next = s < s1 ? a.at(s + 1) : 0;
        split_pv(sm, g, a, kc, vc, proj, s, off, os);
#pragma unroll
        for (int j = 0; j < OPT; ++j) o[j] = __fadd_rn(o[j], os[j]), os[j] = 0.0f;
        off = next;
    }
    const int d = threadIdx.x & (D - 1), vg = threadIdx.x / D;
#pragma unroll
    for (int j = 0; j < OPT; ++j) {
        const int vec = vg * OPT + j, t = g.vrow(vec);
        if (t >= R) continue;
        const size_t at = ((size_t)t * hq + g.vhead(vec)) * D + d;
        out[at] = gated(o[j], gate[at]);
    }
}

// The core's two forms: keys in the ring (or a slot a position), and in the pool's pages (`pages`: the slot's page
// table, `page`: the pool's page stride in elements; kc, vc this layer's rows in page 0).
extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_core(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ sink,
    const bf16* __restrict__ gate, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
    const bf16* __restrict__ proj, bf16* __restrict__ out, int R, int pos0, int window, int hq, int hk, int slots) {
    core_body(q, rel, sink, gate, kc, vc, proj, out, R, pos0, window, hq, hk, slots, Ring{slots});
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_core_pg(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ sink,
    const bf16* __restrict__ gate, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
    const bf16* __restrict__ proj, bf16* __restrict__ out, int R, int pos0, int window, int hq, int hk, int slots,
    const int* __restrict__ pages, long long page) {
    core_body(q, rel, sink, gate, kc, vc, proj, out, R, pos0, window, hq, hk, slots, Paged{pages, page});
}

// Split form, pass 1: grid (ceil(R / RB), hk, splits), split s = s_first + blockIdx.z; st float2 [splits, R, hq]
// gets each row's (max, sum) over its keys in split s (rows without keys there get (-inf, 0)).
template <class A>
__device__ __forceinline__ void stats_body(const bf16* __restrict__ q, const bf16* __restrict__ rel,
                                           const bf16* __restrict__ kc, const bf16* __restrict__ proj,
                                           float2* __restrict__ st, int R, int pos0, int window, int hq, int hk,
                                           int slots, int s_first, const int* __restrict__ posp, const A& a) {
    __shared__ __align__(16) Smem sm;
    if (posp != nullptr) pos0 = *posp;
    Geo g;
    g.init(R, pos0, window, hq, hk, slots);
    const int s = s_first + blockIdx.z;
    if (s < g.klo / SPLIT || s > g.khi / SPLIT) return;
    const size_t off = a.at(s);
    setup(sm, g, q, rel, nullptr);
    __syncthreads();
    split_stats(sm, g, a, kc, proj, s, off);
    const int v = threadIdx.x;
    if (v < NV && g.vrow(v) < R) st[((size_t)blockIdx.z * R + g.vrow(v)) * hq + g.vhead(v)] = make_float2(sm.ms[v], sm.zs[v]);
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_stats(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ kc,
    const bf16* __restrict__ proj, float2* __restrict__ st, int R, int pos0, int window, int hq, int hk, int slots,
    int s_first, const int* __restrict__ posp) {
    stats_body(q, rel, kc, proj, st, R, pos0, window, hq, hk, slots, s_first, posp, Ring{slots});
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_stats_pg(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ kc,
    const bf16* __restrict__ proj, float2* __restrict__ st, int R, int pos0, int window, int hq, int hk, int slots,
    int s_first, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    stats_body(q, rel, kc, proj, st, R, pos0, window, hq, hk, slots, s_first, posp, Paged{pages, page});
}

// Split form, pass 2: the same grid; each row's (m, z) folded from (sink, 1) and its own splits' pairs in order (or,
// `mz` given, tf_ink2_attn_fold's), then op fp32 [splits, R, hq * D] gets the split's sum of p v.
template <class A>
__device__ __forceinline__ void pv_body(const bf16* __restrict__ q, const bf16* __restrict__ rel,
                                        const bf16* __restrict__ sink, const bf16* __restrict__ kc,
                                        const bf16* __restrict__ vc, const bf16* __restrict__ proj,
                                        const float2* __restrict__ st, const float2* __restrict__ mz,
                                        float* __restrict__ op, int R, int pos0, int window, int hq, int hk, int slots,
                                        int s_first, const int* __restrict__ posp, const A& a) {
    __shared__ __align__(16) Smem sm;
    if (posp != nullptr) pos0 = *posp;
    Geo g;
    g.init(R, pos0, window, hq, hk, slots);
    const int s = s_first + blockIdx.z;
    if (s < g.klo / SPLIT || s > g.khi / SPLIT) return;
    const size_t off = a.at(s);
    setup(sm, g, q, rel, sink);
    const int v = threadIdx.x;
    if (v < NV) {
        const int t = g.vrow(v);
        if (t < R) {
            if (mz != nullptr) {
                const float2 x = mz[(size_t)t * hq + g.vhead(v)];
                sm.m[v] = x.x;
                sm.z[v] = x.y;
            } else {
                const int p = pos0 + t;
                for (int u = g.lo(p) / SPLIT; u <= p / SPLIT; ++u) {
                    const float2 x = st[((size_t)(u - s_first) * R + t) * hq + g.vhead(v)];
                    fold(sm.m[v], sm.z[v], x.x, x.y);
                }
            }
        }
    }
    __syncthreads();
    float os[OPT];
#pragma unroll
    for (int j = 0; j < OPT; ++j) os[j] = 0.0f;
    split_pv(sm, g, a, kc, vc, proj, s, off, os);
    const int d = threadIdx.x & (D - 1), vg = threadIdx.x / D;
#pragma unroll
    for (int j = 0; j < OPT; ++j) {
        const int vec = vg * OPT + j, t = g.vrow(vec);
        if (t < R) op[(((size_t)blockIdx.z * R + t) * hq + g.vhead(vec)) * D + d] = os[j];
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_pv(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ sink,
    const bf16* __restrict__ kc, const bf16* __restrict__ vc, const bf16* __restrict__ proj,
    const float2* __restrict__ st, const float2* __restrict__ mz, float* __restrict__ op, int R, int pos0, int window,
    int hq, int hk, int slots, int s_first, const int* __restrict__ posp) {
    pv_body(q, rel, sink, kc, vc, proj, st, mz, op, R, pos0, window, hq, hk, slots, s_first, posp, Ring{slots});
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_pv_pg(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ sink,
    const bf16* __restrict__ kc, const bf16* __restrict__ vc, const bf16* __restrict__ proj,
    const float2* __restrict__ st, const float2* __restrict__ mz, float* __restrict__ op, int R, int pos0, int window,
    int hq, int hk, int slots, int s_first, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    pv_body(q, rel, sink, kc, vc, proj, st, mz, op, R, pos0, window, hq, hk, slots, s_first, posp, Paged{pages, page});
}

// Split form, the sum: grid (R, hq), D threads; o = sum of the row's own splits in order from 0, then the gate. With
// posp given and s_first < 0, the first split is lo(pos0) / SPLIT (tf_ink2_attn_rstats / _rpv's derived grid).
extern "C" __global__ void __launch_bounds__(D) tf_ink2_attn_merge(
    const float* __restrict__ op, const bf16* __restrict__ gate, bf16* __restrict__ out, int R, int pos0, int window,
    int hq, int s_first, const int* __restrict__ posp) {
    if (posp != nullptr) {
        pos0 = *posp;
        if (s_first < 0) s_first = (window == 0 ? 0 : max(0, pos0 - (window - 1))) / SPLIT;
    }
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int p = pos0 + t;
    const int lo = window == 0 ? 0 : max(0, p - (window - 1));
    float o = 0.0f;
    for (int u = lo / SPLIT; u <= p / SPLIT; ++u) o = __fadd_rn(o, op[(((size_t)(u - s_first) * R + t) * hq + h) * D + d]);
    const size_t at = ((size_t)t * hq + h) * D + d;
    out[at] = gated(o, gate[at]);
}

// Split form, between the passes (forwards over many splits): each row's (m, z) for each query head, folded from
// (sink, 1) and the row's splits' pairs in order (the second pass's own fold, done once instead of in every split's
// block); mz float2 [R, hq]. Grid R, 256 threads (hq <= 256); s_first and posp as tf_ink2_attn_merge's. The pairs
// are staged in shared memory a chunk at a time (all of a chunk's loads in flight), then thread h folds head h's in
// order: fold()'s expressions as selects, so the loop has no branch and its exponentials can run ahead of the sums.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_attn_fold(
    const bf16* __restrict__ sink, const float2* __restrict__ st, float2* __restrict__ mz, int R, int pos0, int window,
    int hq, int s_first, const int* __restrict__ posp) {
    constexpr int BUF = 3072;  // pairs a chunk (24 KiB)
    __shared__ float2 buf[BUF];
    if (posp != nullptr) {
        pos0 = *posp;
        if (s_first < 0) s_first = (window == 0 ? 0 : max(0, pos0 - (window - 1))) / SPLIT;
    }
    const int t = blockIdx.x, h = threadIdx.x;
    if (t >= R) return;
    const int p = pos0 + t;
    const int lo = window == 0 ? 0 : max(0, p - (window - 1));
    const int u0 = lo / SPLIT, u1 = p / SPLIT, per = BUF / hq;
    float m = h < hq ? f32(sink[(size_t)t * hq + h]) : 0.0f, z = 1.0f;
    for (int c = u0; c <= u1; c += per) {
        const int n = min(per, u1 - c + 1);
        __syncthreads();  // the previous chunk's reads are done
        for (int e = threadIdx.x; e < n * hq; e += blockDim.x) buf[e] = st[((size_t)(c + e / hq - s_first) * R + t) * hq + e % hq];
        __syncthreads();
        if (h < hq) {
#pragma unroll 8
            for (int i = 0; i < n; ++i) {
                // fold(m, z, ms, zs): an empty split (!(zs > 0)) leaves the pair as it is
                const float2 x = buf[i * hq + h];
                const float mn = fmaxf(m, x.x);
                const float zn = __fadd_rn(__fmul_rn(z, expf(m - mn)), __fmul_rn(x.y, expf(x.x - mn)));
                const bool use = x.y > 0.0f;
                z = use ? zn : z;
                m = use ? mn : m;
            }
        }
    }
    if (h < hq) mz[(size_t)t * hq + h] = make_float2(m, z);
}

// L11, the cache write: rows t < R of k, v (bf16 [R, n]) to slot (pos0 + t) % slots of kc, vc [slots, n]
// (tf_ink2_attn_kv_write_pg: to row (pos0 + t) % SPLIT of page pages[(pos0 + t) / SPLIT], `page` elements a page);
// grid R, 128 threads, 16 bytes a thread. `posp` as above.
template <class A>
__device__ __forceinline__ void kv_write_body(const uint4* __restrict__ k, const uint4* __restrict__ v,
                                              uint4* __restrict__ kc, uint4* __restrict__ vc, int n16, int pos0,
                                              const int* __restrict__ posp, const A& a) {
    if (posp != nullptr) pos0 = *posp;
    const int t = blockIdx.x, p = pos0 + t;
    const size_t dst = a.at(p / SPLIT) / 8 + (size_t)a.row(p) * n16, src = (size_t)t * n16;
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
        kc[dst + i] = k[src + i];
        vc[dst + i] = v[src + i];
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_ink2_attn_kv_write(
    const uint4* __restrict__ k, const uint4* __restrict__ v, uint4* __restrict__ kc, uint4* __restrict__ vc, int n16,
    int slots, int pos0, const int* __restrict__ posp) {
    kv_write_body(k, v, kc, vc, n16, pos0, posp, Ring{slots});
}

extern "C" __global__ void __launch_bounds__(128) tf_ink2_attn_kv_write_pg(
    const uint4* __restrict__ k, const uint4* __restrict__ v, uint4* __restrict__ kc, uint4* __restrict__ vc, int n16,
    int slots, int pos0, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    kv_write_body(k, v, kc, vc, n16, pos0, posp, Paged{pages, page});
}

// One decode row (R = 1) in the split form, the same arithmetic as tf_ink2_attn_stats / tf_ink2_attn_pv with the
// work laid out for one row: every key of the split is scored at once (a thread a key, the row's G vectors), then the
// tile-by-tile folds run as before, but with the exponentials taken in parallel and only the ordered sums left serial.
// Every value is the same expression on the same inputs in the same order, so the bits equal the general kernels'.
// Same arguments and grid (1, hk, splits); THREADS == SPLIT threads.
namespace ink2_attn1 {
using namespace ink2_attn;
static_assert(THREADS == SPLIT && SPLIT % KT == 0, "a thread a key of the split");
constexpr int TILES = SPLIT / KT;
constexpr int PV1_KC = 128;  // keys whose value rows tf_ink2_attn_pv1 stages at once (32 KiB of shared memory)
constexpr int RS_IDS = 64;   // page ids tf_ink2_attn_rstats_pg / _rpv_pg stage a block (later splits read the table)

struct Smem1 {
    bf162 q[G][D / 2];
    float rel[G][DREL];
    float s[G][SPLIT];    // scores (stats), then exp(s - mn) or -1 for -inf; probabilities (pv)
    float tm[G][TILES];   // a tile's max
    float mn[G][TILES];   // the running max a tile's sums use
    float m[G], z[G];
};

// The block's row's query vectors and relative states (vector v = query head kvh * G + v).
__device__ void setup1(Smem1& sm, const bf16* q, const bf16* rel, int hq, int kvh) {
    for (int i = threadIdx.x; i < G * (D / 2); i += THREADS) {
        const int v = i / (D / 2), w = i % (D / 2);
        sm.q[v][w] = reinterpret_cast<const bf162*>(q + ((size_t)kvh * G + v) * D)[w];
    }
    for (int i = threadIdx.x; i < G * DREL; i += THREADS) {
        const int v = i / DREL, j = i % DREL;
        sm.rel[v][j] = f32(rel[((size_t)kvh * G + v) * DREL + j]);
    }
}

// Key kp's score for each vector (scores(): bf16(bf16(bf16(q.k) * scale) + bias), -inf outside [lo, pos]); with
// `prob`, bf16(exp(s - m) / z) (0 outside). kc: the key's split's base (its page's, paged).
template <class A>
__device__ void score1(const Smem1& sm, const A& a, const bf16* kc, const bf16* proj, int kp, int lo, int pos, int hk,
                       int kvh, bool prob, float (&x)[G]) {
    const bool in = kp >= lo && kp <= pos;
    float acc[G];
#pragma unroll
    for (int v = 0; v < G; ++v) acc[v] = 0.0f;
    if (in) {
        const uint4* kr = reinterpret_cast<const uint4*>(kc + (size_t)a.row(kp) * hk * D + (size_t)kvh * D);
#pragma unroll 2
        for (int c = 0; c < D / 8; ++c) {
            const uint4 u = kr[c];
            const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                uint32_t kw = words[e];
                const float2 kf = __bfloat1622float2(*reinterpret_cast<const bf162*>(&kw));
                const int w = c * 4 + e;
#pragma unroll
                for (int v = 0; v < G; ++v) {
                    const float2 qf = __bfloat1622float2(sm.q[v][w]);
                    acc[v] = fmaf(qf.x, kf.x, acc[v]);
                    acc[v] = fmaf(qf.y, kf.y, acc[v]);
                }
            }
        }
    }
#pragma unroll
    for (int v = 0; v < G; ++v) {
        float y = prob ? 0.0f : -INFINITY;
        if (in) {
            float s = bf(__fmul_rn(bf(acc[v]), scale()));
            const int dist = pos - kp;
            if (dist < EXTENT) {
                float b = 0.0f;
#pragma unroll
                for (int i = 0; i < DREL; ++i) b = fmaf(sm.rel[v][i], f32(proj[i * EXTENT + dist]), b);
                s = bf(__fadd_rn(s, bf(b)));
            }
            y = prob ? bf(__fdiv_rn(expf(s - sm.m[v]), sm.z[v])) : s;
        }
        x[v] = y;
    }
}
}  // namespace ink2_attn1

template <class A>
__device__ __forceinline__ void stats1_body(const bf16* __restrict__ q, const bf16* __restrict__ rel,
                                            const bf16* __restrict__ kc, const bf16* __restrict__ proj,
                                            float2* __restrict__ st, int R, int pos0, int window, int hq, int hk,
                                            int s_first, const int* __restrict__ posp, const A& a) {
    using namespace ink2_attn1;
    __shared__ __align__(16) Smem1 sm;
    if (posp != nullptr) pos0 = *posp;
    const int kvh = blockIdx.y, s = s_first + blockIdx.z;
    const int lo = window == 0 ? 0 : max(0, pos0 - (window - 1)), pos = pos0;
    if (R != 1 || s < lo / SPLIT || s > pos / SPLIT) return;
    kc += a.at(s);
    setup1(sm, q, rel, hq, kvh);
    __syncthreads();
    // split_keys(): the tiles from k0 (tile-aligned) through the one holding k1
    const int k0 = max(s * SPLIT, (lo / KT) * KT), k1 = min(s * SPLIT + SPLIT - 1, pos);
    const int ntiles = (k1 - k0) / KT + 1;
    const int i = threadIdx.x;
    if (i < ntiles * KT) {
        float x[G];
        score1(sm, a, kc, proj, k0 + i, lo, pos, hk, kvh, false, x);
#pragma unroll
        for (int v = 0; v < G; ++v) sm.s[v][i] = x[v];
    }
    __syncthreads();
    // tile_stats()'s max of each tile, in key order
    if (i < G * ntiles) {
        const int v = i / ntiles, t = i % ntiles;
        float tm = -INFINITY;
        for (int kk = 0; kk < KT; ++kk) tm = fmaxf(tm, sm.s[v][t * KT + kk]);
        sm.tm[v][t] = tm;
    }
    __syncthreads();
    // the running max each tile's sum is taken against (tiles whose max is -inf leave the pair unchanged)
    if (i < G) {
        float ms = -INFINITY;
        for (int t = 0; t < ntiles; ++t) {
            if (sm.tm[i][t] == -INFINITY) continue;
            ms = fmaxf(ms, sm.tm[i][t]);
            sm.mn[i][t] = ms;
        }
    }
    __syncthreads();
    if (i < ntiles * KT) {
        const int t = i / KT;
#pragma unroll
        for (int v = 0; v < G; ++v) {
            const float x = sm.s[v][i];
            sm.s[v][i] = (sm.tm[v][t] == -INFINITY || x == -INFINITY) ? -1.0f : expf(x - sm.mn[v][t]);  // -1: not summed
        }
    }
    __syncthreads();
    // tile_stats()'s sums, in key order
    if (i < G) {
        float ms = -INFINITY, zs = 0.0f;
        for (int t = 0; t < ntiles; ++t) {
            if (sm.tm[i][t] == -INFINITY) continue;
            const float mn = sm.mn[i][t];
            float z = __fmul_rn(zs, expf(ms - mn));
            for (int kk = 0; kk < KT; ++kk) {
                const float e = sm.s[i][t * KT + kk];
                if (e != -1.0f) z = __fadd_rn(z, e);
            }
            ms = mn;
            zs = z;
        }
        st[(size_t)blockIdx.z * hq + (size_t)kvh * G + i] = make_float2(ms, zs);
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_stats1(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ kc,
    const bf16* __restrict__ proj, float2* __restrict__ st, int R, int pos0, int window, int hq, int hk, int slots,
    int s_first, const int* __restrict__ posp) {
    stats1_body(q, rel, kc, proj, st, R, pos0, window, hq, hk, s_first, posp, Ring{slots});
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_stats1_pg(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ kc,
    const bf16* __restrict__ proj, float2* __restrict__ st, int R, int pos0, int window, int hq, int hk, int slots,
    int s_first, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    stats1_body(q, rel, kc, proj, st, R, pos0, window, hq, hk, s_first, posp, Paged{pages, page});
}

template <class A>
__device__ __forceinline__ void pv1_body(const bf16* __restrict__ q, const bf16* __restrict__ rel,
                                         const bf16* __restrict__ sink, const bf16* __restrict__ kc,
                                         const bf16* __restrict__ vc, const bf16* __restrict__ proj,
                                         const float2* __restrict__ st, const float2* __restrict__ mz,
                                         float* __restrict__ op, int R, int pos0, int window, int hq, int hk,
                                         int s_first, const int* __restrict__ posp, const A& a) {
    using namespace ink2_attn1;
    __shared__ __align__(16) Smem1 sm;
    if (posp != nullptr) pos0 = *posp;
    const int kvh = blockIdx.y, s = s_first + blockIdx.z;
    const int lo = window == 0 ? 0 : max(0, pos0 - (window - 1)), pos = pos0;
    if (R != 1 || s < lo / SPLIT || s > pos / SPLIT) return;
    const size_t off = a.at(s);
    kc += off;
    vc += off;
    setup1(sm, q, rel, hq, kvh);
    const int i = threadIdx.x;
    if (i < G) {
        // the row's (m, z): from (sink, 1), its splits' pairs folded in order (or tf_ink2_attn_fold's)
        float m = f32(sink[(size_t)kvh * G + i]), z = 1.0f;
        if (mz != nullptr) {
            const float2 x = mz[(size_t)kvh * G + i];
            m = x.x;
            z = x.y;
        } else
            for (int u = lo / SPLIT; u <= pos / SPLIT; ++u) {
                const float2 x = st[(size_t)(u - s_first) * hq + (size_t)kvh * G + i];
                fold(m, z, x.x, x.y);
            }
        sm.m[i] = m;
        sm.z[i] = z;
    }
    __syncthreads();
    const int k0 = max(s * SPLIT, (lo / KT) * KT), k1 = min(s * SPLIT + SPLIT - 1, pos);
    const int nk = ((k1 - k0) / KT + 1) * KT;
    if (i < nk) {
        float x[G];
        score1(sm, a, kc, proj, k0 + i, lo, pos, hk, kvh, true, x);
#pragma unroll
        for (int v = 0; v < G; ++v) sm.s[v][i] = x[v];
    }
    __syncthreads();
    // pv(): os = sum of p v over the split's keys in order, a column d of G / 2 vectors a thread. The keys' value rows
    // are staged in shared memory PV1_KC keys at a time by the whole block (16-byte loads, all in flight at once; a
    // key outside the window stages zeros, the 0.0f the chain added before), so the chain no longer waits on one
    // global load a key; the fmaf chain itself is unchanged (same operands, same order).
    __shared__ __align__(16) bf16 sv[PV1_KC][D];
    const int d = i & (D - 1), v0 = (i / D) * (G / 2);
    const size_t ld = (size_t)hk * D;
    float o[G / 2];
#pragma unroll
    for (int j = 0; j < G / 2; ++j) o[j] = 0.0f;
    for (int c0 = 0; c0 < nk; c0 += PV1_KC) {
        const int nc = min(PV1_KC, nk - c0);
        __syncthreads();  // the previous chunk's reads are done
        for (int e = i; e < nc * (D / 8); e += THREADS) {
            const int kk = e / (D / 8), w = e % (D / 8), kp = k0 + c0 + kk;
            uint4 u = make_uint4(0u, 0u, 0u, 0u);
            if (kp >= lo && kp <= pos) u = reinterpret_cast<const uint4*>(vc + (size_t)a.row(kp) * ld + (size_t)kvh * D)[w];
            reinterpret_cast<uint4*>(&sv[kk][0])[w] = u;
        }
        __syncthreads();
#pragma unroll 4
        for (int kk = 0; kk < nc; ++kk) {
            const float vf = f32(sv[kk][d]);
#pragma unroll
            for (int j = 0; j < G / 2; ++j) o[j] = fmaf(sm.s[v0 + j][c0 + kk], vf, o[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < G / 2; ++j) op[((size_t)blockIdx.z * hq + (size_t)kvh * G + v0 + j) * D + d] = o[j];
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_pv1(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ sink,
    const bf16* __restrict__ kc, const bf16* __restrict__ vc, const bf16* __restrict__ proj,
    const float2* __restrict__ st, const float2* __restrict__ mz, float* __restrict__ op, int R, int pos0, int window,
    int hq, int hk, int slots, int s_first, const int* __restrict__ posp) {
    pv1_body(q, rel, sink, kc, vc, proj, st, mz, op, R, pos0, window, hq, hk, s_first, posp, Ring{slots});
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_pv1_pg(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ sink,
    const bf16* __restrict__ kc, const bf16* __restrict__ vc, const bf16* __restrict__ proj,
    const float2* __restrict__ st, const float2* __restrict__ mz, float* __restrict__ op, int R, int pos0, int window,
    int hq, int hk, int slots, int s_first, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    pv1_body(q, rel, sink, kc, vc, proj, st, mz, op, R, pos0, window, hq, hk, s_first, posp, Paged{pages, page});
}

// The split form for forwards of 1 .. 8 rows (decode steps, DSpark verify windows, a batched step's segments): the
// one-row passes above with a block a (row, key head, split), grid (R, hk, splits), so a window's rows run side by
// side instead of padded into one 8-row block, and the q.k of a split is taken once:
//   tf_ink2_attn_rstats: stats1's scores and folds for row blockIdx.x (each key's score also stored, as the bf16 it
//     already is, in sc bf16 [splits, R, hq, SPLIT] at the key's index from the row's first tile), and its pair;
//   tf_ink2_attn_rpv: the row's (m, z) folded as pv1 does, the probabilities bf16(exp(s - m) / z) from the stored
//     scores (0 where the score is -inf: outside the row's window, where score1 gives 0), the value rows staged in
//     shared memory (a chunk's 16-byte loads all in flight, the next chunk's issued before this one's sums), and
//     pv1's fmaf chain (same operands, same order).
// Every value is the same expression on the same inputs in the same order as stats1 / pv1 (and the general stats /
// pv), so the bits are the same. A row's window, tiles and keys are its own: the other rows of the forward do not
// enter its block.
//
// Captured steps: with posp given and s_first < 0, the forward's first split is derived on the device from the
// position, lo(pos0) / SPLIT (sliding layers: the grid then holds only the 5 or 6 splits a window of R rows can
// touch, not every split of the slot); tf_ink2_attn_merge derives it the same way, so all three index the scratch
// alike. A grid narrower than the row's splits (a captured global layer: s_first 0, at most GRID splits) is walked
// by blockIdx.z, blockIdx.z + gridDim.z, ...: a block runs each of its splits in turn, each as above (z indexes the
// scratch). In a single-split block (every eager forward) that is one pass, as before.
namespace ink2_attn1 {

__device__ __forceinline__ int first_split(int pos0, int window) {
    return (window == 0 ? 0 : max(0, pos0 - (window - 1))) / SPLIT;
}

// The block's first grid index z (blockIdx.z + k * gridDim.z) whose split s_first + z is at or past split `from`.
__device__ __forceinline__ int first_z(int s_first, int from) {
    int z = blockIdx.z;
    if (s_first + z < from) z += (from - s_first - z + gridDim.z - 1) / gridDim.z * gridDim.z;
    return z;
}

// score1 with the key row's sixteen 16-byte loads issued before the first product (the fmaf chain unchanged).
template <class A>
__device__ void score_r(const Smem1& sm, const A& a, const bf16* kc, const bf16* proj, int kp, int lo, int pos, int hk,
                        int kvh, float (&x)[G]) {
    const bool in = kp >= lo && kp <= pos;
    float acc[G];
#pragma unroll
    for (int v = 0; v < G; ++v) acc[v] = 0.0f;
    if (in) {
        const uint4* kr = reinterpret_cast<const uint4*>(kc + (size_t)a.row(kp) * hk * D + (size_t)kvh * D);
        uint4 u[D / 8];
#pragma unroll
        for (int c = 0; c < D / 8; ++c) u[c] = kr[c];
#pragma unroll
        for (int c = 0; c < D / 8; ++c) {
            const uint32_t words[4] = {u[c].x, u[c].y, u[c].z, u[c].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                uint32_t kw = words[e];
                const float2 kf = __bfloat1622float2(*reinterpret_cast<const bf162*>(&kw));
                const int w = c * 4 + e;
#pragma unroll
                for (int v = 0; v < G; ++v) {
                    const float2 qf = __bfloat1622float2(sm.q[v][w]);
                    acc[v] = fmaf(qf.x, kf.x, acc[v]);
                    acc[v] = fmaf(qf.y, kf.y, acc[v]);
                }
            }
        }
    }
#pragma unroll
    for (int v = 0; v < G; ++v) {
        float y = -INFINITY;
        if (in) {
            float s = bf(__fmul_rn(bf(acc[v]), scale()));
            const int dist = pos - kp;
            if (dist < EXTENT) {
                float b = 0.0f;
#pragma unroll
                for (int i = 0; i < DREL; ++i) b = fmaf(sm.rel[v][i], f32(proj[i * EXTENT + dist]), b);
                s = bf(__fadd_rn(s, bf(b)));
            }
            y = s;
        }
        x[v] = y;
    }
}
}  // namespace ink2_attn1

template <class A>
__device__ __forceinline__ void rstats_body(const bf16* __restrict__ q, const bf16* __restrict__ rel,
                                            const bf16* __restrict__ kc, const bf16* __restrict__ proj,
                                            float2* __restrict__ st, bf16* __restrict__ sc, int R, int pos0,
                                            int window, int hq, int hk, int slots, int s_first,
                                            const int* __restrict__ posp, const A& a) {
    using namespace ink2_attn1;
    __shared__ __align__(16) Smem1 sm;
    // paged: the page ids of the block's splits (s_first + blockIdx.z + k gridDim.z, k < RS_IDS), read first of all so
    // their latency runs under setup1's and the device position's loads and no split's key loads wait on a table read.
    // A block's first split is s_first + blockIdx.z whenever s_first >= 0 at entry (global layers: lo = 0); entries
    // past the table are not read and ids past the row's last split are never used.
    __shared__ int ids[A::table ? RS_IDS : 1];
    const bool staged = A::table && s_first >= 0;
    if constexpr (A::table) {
        if (staged && threadIdx.x < RS_IDS) {
            const int s = s_first + (int)blockIdx.z + (int)threadIdx.x * (int)gridDim.z;
            if (s < slots / SPLIT) ids[threadIdx.x] = __ldg(a.pages + s);
        }
    }
    const int z0 = blockIdx.z;
    if (posp != nullptr) {
        pos0 = *posp;
        if (s_first < 0) s_first = first_split(pos0, window);
    }
    const int t = blockIdx.x, kvh = blockIdx.y;
    const int pos = pos0 + t, lo = window == 0 ? 0 : max(0, pos - (window - 1));
    if (t >= R) return;
    int zi = first_z(s_first, lo / SPLIT);
    if (s_first + zi > pos / SPLIT) return;
    setup1(sm, q + (size_t)t * hq * D, rel + (size_t)t * hq * DREL, hq, kvh);
    for (; s_first + zi <= pos / SPLIT; zi += gridDim.z) {
        __syncthreads();  // setup1's writes (and ids'); a later split: the previous one's reads of sm.s, sm.tm and sm.mn
        const int s = s_first + zi;
        size_t at = 0;  // the split's page offset (ring: none)
        if constexpr (A::table) {
            const int kz = (zi - z0) / (int)gridDim.z;  // the block's kz-th split
            at = staged && kz < RS_IDS ? (size_t)ids[kz] * (size_t)a.page : a.at(s);
        }
        const bf16* kb = kc + at;
        const int k0 = max(s * SPLIT, (lo / KT) * KT), k1 = min(s * SPLIT + SPLIT - 1, pos);
        const int ntiles = (k1 - k0) / KT + 1;
        const int i = threadIdx.x;
        const size_t row = ((size_t)zi * R + t) * hq + (size_t)kvh * G;  // the row's first vector in st / sc
        if (i < ntiles * KT) {
            float x[G];
            score_r(sm, a, kb, proj, k0 + i, lo, pos, hk, kvh, x);
#pragma unroll
            for (int v = 0; v < G; ++v) {
                sm.s[v][i] = x[v];
                sc[(row + v) * SPLIT + i] = __float2bfloat16_rn(x[v]);  // x is -inf or a bf16 value: stored exactly
            }
        }
        __syncthreads();
        if (i < G * ntiles) {
            const int v = i / ntiles, tt = i % ntiles;
            float tm = -INFINITY;
            for (int kk = 0; kk < KT; ++kk) tm = fmaxf(tm, sm.s[v][tt * KT + kk]);
            sm.tm[v][tt] = tm;
        }
        __syncthreads();
        if (i < G) {
            float ms = -INFINITY;
            for (int tt = 0; tt < ntiles; ++tt) {
                if (sm.tm[i][tt] == -INFINITY) continue;
                ms = fmaxf(ms, sm.tm[i][tt]);
                sm.mn[i][tt] = ms;
            }
        }
        __syncthreads();
        if (i < ntiles * KT) {
            const int tt = i / KT;
#pragma unroll
            for (int v = 0; v < G; ++v) {
                const float x = sm.s[v][i];
                sm.s[v][i] = (sm.tm[v][tt] == -INFINITY || x == -INFINITY) ? -1.0f : expf(x - sm.mn[v][tt]);
            }
        }
        __syncthreads();
        if (i < G) {
            float ms = -INFINITY, zs = 0.0f;
            for (int tt = 0; tt < ntiles; ++tt) {
                if (sm.tm[i][tt] == -INFINITY) continue;
                const float mn = sm.mn[i][tt];
                float z = __fmul_rn(zs, expf(ms - mn));
                for (int kk = 0; kk < KT; ++kk) {
                    const float e = sm.s[i][tt * KT + kk];
                    if (e != -1.0f) z = __fadd_rn(z, e);
                }
                ms = mn;
                zs = z;
            }
            st[row + i] = make_float2(ms, zs);
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_rstats(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ kc,
    const bf16* __restrict__ proj, float2* __restrict__ st, bf16* __restrict__ sc, int R, int pos0, int window, int hq,
    int hk, int slots, int s_first, const int* __restrict__ posp) {
    rstats_body(q, rel, kc, proj, st, sc, R, pos0, window, hq, hk, slots, s_first, posp, ink2_attn::Ring{slots});
}

// (THREADS, 1): under (THREADS) alone ptxas holds the paged form at 64 registers and spills 24 bytes of its loop state
// (the ring form fits 64 with none); at 80 it spills nothing and keeps more key loads in flight (pool lane G2: a one-row
// step at 131k 5 % faster captured; 8 rows 10-12 % faster than the ring form).
extern "C" __global__ void __launch_bounds__(THREADS, 1) tf_ink2_attn_rstats_pg(
    const bf16* __restrict__ q, const bf16* __restrict__ rel, const bf16* __restrict__ kc,
    const bf16* __restrict__ proj, float2* __restrict__ st, bf16* __restrict__ sc, int R, int pos0, int window, int hq,
    int hk, int slots, int s_first, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    rstats_body(q, rel, kc, proj, st, sc, R, pos0, window, hq, hk, slots, s_first, posp, ink2_attn::Paged{pages, page});
}

template <class A>
__device__ __forceinline__ void rpv_body(const bf16* __restrict__ sink, const bf16* __restrict__ vc,
                                         const float2* __restrict__ st, const float2* __restrict__ mz,
                                         const bf16* __restrict__ sc, float* __restrict__ op, int R, int pos0,
                                         int window, int hq, int hk, int slots, int s_first,
                                         const int* __restrict__ posp, const A& a) {
    using namespace ink2_attn1;
    __builtin_assume(threadIdx.x < THREADS);  // the kernel's launch bounds, as the ring kernel's own body had them
    __shared__ float pm[G], pz[G];
    __shared__ float p[G][SPLIT];
    __shared__ __align__(16) bf16 sv[PV1_KC][D];
    // paged: the block's page ids read first of all, as rstats does (the first split's into every thread's register,
    // the block's splits' into ids, read after the first barrier), so no split's value loads wait on a table read
    __shared__ int ids[A::table ? RS_IDS : 1];
    const bool staged = A::table && s_first >= 0;
    int id0 = 0;
    if constexpr (A::table) {
        if (staged) {
            const int n = slots / SPLIT, s0 = s_first + (int)blockIdx.z;
            if (s0 < n) id0 = __ldg(a.pages + s0);
            const int s = s0 + (int)threadIdx.x * (int)gridDim.z;
            if (threadIdx.x < RS_IDS && s < n) ids[threadIdx.x] = __ldg(a.pages + s);
        }
    }
    const int z0 = blockIdx.z;
    if (posp != nullptr) {
        pos0 = *posp;
        if (s_first < 0) s_first = first_split(pos0, window);
    }
    const int t = blockIdx.x, kvh = blockIdx.y;
    const int pos = pos0 + t, lo = window == 0 ? 0 : max(0, pos - (window - 1));
    if (t >= R) return;
    int z = first_z(s_first, lo / SPLIT);
    if (s_first + z > pos / SPLIT) return;
    const int i = threadIdx.x;
    const size_t ld = (size_t)hk * D;
    // the split's value rows, PV1_KC keys a chunk, 16-byte loads into registers
    constexpr int PER = PV1_KC * (D / 8) / THREADS;  // 16-byte loads a thread a chunk
    static_assert(PER * THREADS == PV1_KC * (D / 8), "staging");
    uint4 ld_[PER];
    // vb: the split's base (its page's, paged)
    auto fetch = [&](const bf16* vb, int k0, int nk, int c0) {
#pragma unroll
        for (int e0 = 0; e0 < PER; ++e0) {
            const int e = i + e0 * THREADS, kk = e / (D / 8), w = e % (D / 8), kp = k0 + c0 + kk;
            uint4 u = make_uint4(0u, 0u, 0u, 0u);
            if (c0 + kk < nk && kp >= lo && kp <= pos) u = reinterpret_cast<const uint4*>(vb + (size_t)a.row(kp) * ld + (size_t)kvh * D)[w];
            ld_[e0] = u;
        }
    };
    // the keys of split s: k0 .. k0 + nk - 1 (whole tiles)
    auto keys = [&](int s, int& k0, int& nk) {
        k0 = max(s * SPLIT, (lo / KT) * KT);
        const int k1 = min(s * SPLIT + SPLIT - 1, pos);
        nk = ((k1 - k0) / KT + 1) * KT;
    };
    int k0, nk;
    keys(s_first + z, k0, nk);
    // paged: the row's sink and folded pair are loaded before the page id the value fetch waits on, so the two
    // latencies overlap (the ring form loads them where it always did)
    float early_m = 0.0f;
    float2 early_mz = make_float2(0.0f, 0.0f);
    if constexpr (A::table) {
        if (i < G) {
            early_m = f32(sink[(size_t)t * hq + (size_t)kvh * G + i]);
            if (mz != nullptr) early_mz = mz[(size_t)t * hq + (size_t)kvh * G + i];
        }
    }
    // the first split's first chunk, in flight while the row's (m, z) is made
    size_t at0 = 0;  // its page offset (ring: none)
    if constexpr (A::table) at0 = staged && z == z0 ? (size_t)id0 * (size_t)a.page : a.at(s_first + z);
    const bf16* vb = vc + at0;
    fetch(vb, k0, nk, 0);
    if (i < G) {
        // the row's (m, z): from (sink, 1), its splits' pairs folded in order (or tf_ink2_attn_fold's)
        float m = A::table ? early_m : f32(sink[(size_t)t * hq + (size_t)kvh * G + i]), zz = 1.0f;
        if (mz != nullptr) {
            const float2 x = A::table ? early_mz : mz[(size_t)t * hq + (size_t)kvh * G + i];
            m = x.x;
            zz = x.y;
        } else
            for (int u = lo / SPLIT; u <= pos / SPLIT; ++u) {
                const float2 x = st[((size_t)(u - s_first) * R + t) * hq + (size_t)kvh * G + i];
                fold(m, zz, x.x, x.y);
            }
        pm[i] = m;
        pz[i] = zz;
    }
    __syncthreads();
    const int d = i & (D - 1), v0 = (i / D) * (G / 2);
    for (bool first = true; s_first + z <= pos / SPLIT; z += gridDim.z, first = false) {
        if (!first) {
            keys(s_first + z, k0, nk);
            size_t at = 0;  // the split's page offset (ring: none)
            if constexpr (A::table) {
                const int kz = (z - z0) / (int)gridDim.z;  // the block's kz-th split (ids written before the barrier above)
                at = staged && kz < RS_IDS ? (size_t)ids[kz] * (size_t)a.page : a.at(s_first + z);
            }
            vb = vc + at;
            fetch(vb, k0, nk, 0);
            __syncthreads();  // the previous split's reads of p and sv are done
        }
        const size_t row = ((size_t)z * R + t) * hq + (size_t)kvh * G;
        if (i < nk) {
#pragma unroll
            for (int v = 0; v < G; ++v) {
                const float x = f32(sc[(row + v) * SPLIT + i]);
                p[v][i] = x == -INFINITY ? 0.0f : bf(__fdiv_rn(expf(x - pm[v]), pz[v]));
            }
        }
        float o[G / 2];
#pragma unroll
        for (int j = 0; j < G / 2; ++j) o[j] = 0.0f;
        for (int c0 = 0; c0 < nk; c0 += PV1_KC) {
            const int nc = min(PV1_KC, nk - c0);
            __syncthreads();  // the previous chunk's reads (and the probabilities' writes) are done
#pragma unroll
            for (int e0 = 0; e0 < PER; ++e0) {
                const int e = i + e0 * THREADS;
                reinterpret_cast<uint4*>(&sv[e / (D / 8)][0])[e % (D / 8)] = ld_[e0];
            }
            __syncthreads();
            if (c0 + PV1_KC < nk) fetch(vb, k0, nk, c0 + PV1_KC);
#pragma unroll 4
            for (int kk = 0; kk < nc; ++kk) {
                const float vf = f32(sv[kk][d]);
#pragma unroll
                for (int j = 0; j < G / 2; ++j) o[j] = fmaf(p[v0 + j][c0 + kk], vf, o[j]);
            }
        }
#pragma unroll
        for (int j = 0; j < G / 2; ++j) op[(row + v0 + j) * D + d] = o[j];
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_rpv(
    const bf16* __restrict__ sink, const bf16* __restrict__ vc, const float2* __restrict__ st,
    const float2* __restrict__ mz, const bf16* __restrict__ sc, float* __restrict__ op, int R, int pos0, int window,
    int hq, int hk, int slots, int s_first, const int* __restrict__ posp) {
    rpv_body(sink, vc, st, mz, sc, op, R, pos0, window, hq, hk, slots, s_first, posp, ink2_attn::Ring{slots});
}

extern "C" __global__ void __launch_bounds__(THREADS) tf_ink2_attn_rpv_pg(
    const bf16* __restrict__ sink, const bf16* __restrict__ vc, const float2* __restrict__ st,
    const float2* __restrict__ mz, const bf16* __restrict__ sc, float* __restrict__ op, int R, int pos0, int window,
    int hq, int hk, int slots, int s_first, const int* __restrict__ posp, const int* __restrict__ pages, long long page) {
    rpv_body(sink, vc, st, mz, sc, op, R, pos0, window, hq, hk, slots, s_first, posp, ink2_attn::Paged{pages, page});
}
