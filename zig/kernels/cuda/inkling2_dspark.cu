// Inkling-2's DSpark drafter (mtp.safetensors): the context mixtures of the target's taps, the draft layers' norms,
// head norms with rotary positions, the block attention (non-causal in the block, over the context before it), the
// SwiGLU, and the Markov head's recurrence, corrected logits and greedy pick. The entry points, shapes and math are
// in zig/src/families/inkling2/cuda_dspark.zig; the torch reference is tools/zig/inkling2_dspark_ref.py.
//
// The drafter math follows TokenSpeed's native Inkling-2 draft (models/inkling2_mtp.py, MIT, Copyright (c) 2026
// LightSeek Foundation; see the notice in the reference tool).

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_ink2_dk {

constexpr int H = 4608;
constexpr int VEC = 8;
constexpr int NT = H / VEC;  // 576 threads a row
constexpr int WARPS = NT / 32;
constexpr int D = 128;
constexpr int RANK = 256;
constexpr int SPLIT = 256;  // keys a CTA of the attention

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* f) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ void load8(const float* p, float* f) {
    const float4 a = reinterpret_cast<const float4*>(p)[0];
    const float4 b = reinterpret_cast<const float4*>(p)[1];
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* f) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    return v;
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// A row's sum of squares over H in a fixed order (8 columns a thread, the warp's xor tree, warps in order).
__device__ __forceinline__ float row_ss(const float* f, float* red) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) s = __fadd_rn(s, __fmul_rn(f[i], f[i]));
    s = warp_sum(s);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) red[w] = s;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < WARPS; ++i) t = __fadd_rn(t, red[i]);
    __syncthreads();
    return t;
}

// y = bf16(x * rsqrt(ss / H + eps) * g), the mean as a product with fp32(1 / H) (PrecisionRMSNorm's reciprocal).
__device__ __forceinline__ void norm_store(const float* f, const float* g, float ss, float eps, __nv_bfloat16* out) {
    const float r = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / H), eps));
    float y[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) y[i] = __fmul_rn(__fmul_rn(f[i], r), g[i]);
    store8(out, y);
}

}  // namespace tf_ink2_dk

using namespace tf_ink2_dk;

extern "C" __global__ void tf_ink2_dspark_probe(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 7;
}

// The context mixture of stage s: out[r] = norm_s(sum_t taps[t][r] * alpha[t]), taps bf16 [ntaps][rows][H] (tap t at
// taps + t * tap_stride elements), alpha fp32 [ntaps] (the stage's bf16 softmax as fp32), gain fp32 [H].
// Grid rows, NT threads.
extern "C" __global__ void __launch_bounds__(NT) tf_ink2_dspark_mix(const __nv_bfloat16* __restrict__ taps,
                                                                    long long tap_stride, int ntaps,
                                                                    const float* __restrict__ alpha,
                                                                    const float* __restrict__ gain,
                                                                    __nv_bfloat16* __restrict__ out, float eps) {
    __shared__ float red[WARPS];
    const int64_t r = blockIdx.x;
    const int c0 = threadIdx.x * VEC;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.f;
    for (int t = 0; t < ntaps; ++t) {
        float f[VEC];
        load8(taps + t * tap_stride + r * H + c0, f);
        const float a = alpha[t];
#pragma unroll
        for (int i = 0; i < VEC; ++i) acc[i] = __fadd_rn(acc[i], __fmul_rn(f[i], a));
    }
    const float ss = row_ss(acc, red);
    float g[VEC];
    load8(gain + c0, g);
    norm_store(acc, g, ss, eps, out + r * H + c0);
}

// The pre-norm residual: total = h + res (fp32; h may be null: the first layer's input alone), res = bf16(total),
// out = norm(total) with gain fp32 [H]. Grid rows, NT threads.
extern "C" __global__ void __launch_bounds__(NT) tf_ink2_dspark_addnorm(const __nv_bfloat16* __restrict__ h,
                                                                        __nv_bfloat16* __restrict__ res,
                                                                        const float* __restrict__ gain,
                                                                        __nv_bfloat16* __restrict__ out, float eps) {
    __shared__ float red[WARPS];
    const int64_t r = blockIdx.x;
    const int c0 = threadIdx.x * VEC;
    float f[VEC];
    load8(res + r * H + c0, f);
    if (h != nullptr) {
        float a[VEC];
        load8(h + r * H + c0, a);
#pragma unroll
        for (int i = 0; i < VEC; ++i) f[i] = __fadd_rn(a[i], f[i]);
        store8(res + r * H + c0, f);
    }
    const float ss = row_ss(f, red);
    float g[VEC];
    load8(gain + c0, g);
    norm_store(f, g, ss, eps, out + r * H + c0);
}

// Per-head RMS norm (gain fp32 [128], mean by reciprocal), the bf16 rounding, then the neox rotation at position
// pos0 + row (theta; a bf16 cos / sin table as TokenSpeed's RotaryEmbedding keeps). x bf16 rows of x_ld elements, head
// h at columns h * 128; out rows of out_ld elements. A warp a (row, head): grid (heads, rows), 32 threads.
// `posp` not null (a captured forward, replayed at every position): pos0 is read there on the device, and `out` is the
// base of a position-indexed table (row p at out + p * out_ld), so the rows land at pos0 .. as `out + pos0 * out_ld`
// would place them. `ring` > 0: `out` is a mirrored ring's base (the drafter's local layers): position p at row
// p % ring, and rows below `mirror` also at row + ring, so any `mirror` consecutive positions sit in consecutive rows
// (pos0 from the host or the device). The arithmetic is unchanged.
//
// Paged (tf_ink2_dspark_rope_pg, the drafter's global layer in the KV page pool): row p of the table is row p % SPLIT
// of page pages[p / SPLIT], pages `page` elements apart (`out` this layer's rows in page 0); no ring, no mirror. The
// two forms are one source (`PAGED`), chosen on the host by the layer's storage; only the row's address differs.
template <bool PAGED>
__device__ __forceinline__ void rope_body(const __nv_bfloat16* __restrict__ x, long long x_ld,
                                          const float* __restrict__ gain, __nv_bfloat16* __restrict__ out,
                                          long long out_ld, long long pos0, float theta, float eps,
                                          const int* __restrict__ posp, long long ring, long long mirror,
                                          const int* __restrict__ pages, long long page) {
    if (posp != nullptr) pos0 = *posp;
    const int head = blockIdx.x;
    const int64_t r = blockIdx.y;
    const long long orow = ring > 0 ? (pos0 + r) % ring : (posp != nullptr ? pos0 + r : r);
    const int lane = threadIdx.x;
    const __nv_bfloat16* src = x + r * x_ld + head * D + lane * 4;
    float f[4];
    {
        const uint2 u = *reinterpret_cast<const uint2*>(src);
        const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&u);
        const float2 a = __bfloat1622float2(hh[0]), b = __bfloat1622float2(hh[1]);
        f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = __fadd_rn(ss, __fmul_rn(f[i], f[i]));
    ss = warp_sum(ss);
    const float rs = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / D), eps));
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) n[i] = bfr(__fmul_rn(__fmul_rn(f[i], rs), gain[lane * 4 + i]));
    // the partner half: element i pairs with i + 64 (lanes 0-15 with 16-31)
    float p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = __shfl_xor_sync(0xFFFFFFFFu, n[i], 16);
    const float pos = static_cast<float>(pos0 + r);
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = lane * 4 + i;
        const int j = e & 63;
        const float inv = __fdiv_rn(1.0f, powf(theta, __fdiv_rn(static_cast<float>(2 * j), static_cast<float>(D))));
        const float fr = __fmul_rn(pos, inv);
        const float c = bfr(cosf(fr)), s = bfr(sinf(fr));
        // first half: l * cos - r * sin; second half: r * cos + l * sin (l the first-half element)
        o[i] = e < 64 ? __fsub_rn(__fmul_rn(n[i], c), __fmul_rn(p[i], s)) : __fadd_rn(__fmul_rn(n[i], c), __fmul_rn(p[i], s));
    }
    __nv_bfloat16* dst;
    if constexpr (PAGED) {
        const long long p = pos0 + r;
        dst = out + (long long)pages[p / SPLIT] * page + (p % SPLIT) * out_ld + head * D + lane * 4;
    } else
        dst = out + orow * out_ld + head * D + lane * 4;
    uint2 u;
    __nv_bfloat162* hh = reinterpret_cast<__nv_bfloat162*>(&u);
    hh[0] = __floats2bfloat162_rn(o[0], o[1]);
    hh[1] = __floats2bfloat162_rn(o[2], o[3]);
    *reinterpret_cast<uint2*>(dst) = u;
    if (!PAGED && ring > 0 && orow < mirror) *reinterpret_cast<uint2*>(dst + ring * out_ld) = u;
}

extern "C" __global__ void tf_ink2_dspark_rope(const __nv_bfloat16* __restrict__ x, long long x_ld,
                                               const float* __restrict__ gain, __nv_bfloat16* __restrict__ out,
                                               long long out_ld, long long pos0, float theta, float eps,
                                               const int* __restrict__ posp, long long ring, long long mirror) {
    rope_body<false>(x, x_ld, gain, out, out_ld, pos0, theta, eps, posp, ring, mirror, nullptr, 0);
}

extern "C" __global__ void tf_ink2_dspark_rope_pg(const __nv_bfloat16* __restrict__ x, long long x_ld,
                                                  const float* __restrict__ gain, __nv_bfloat16* __restrict__ out,
                                                  long long out_ld, long long pos0, float theta, float eps,
                                                  const int* __restrict__ posp, const int* __restrict__ pages,
                                                  long long page) {
    rope_body<true>(x, x_ld, gain, out, out_ld, pos0, theta, eps, posp, 0, 0, pages, page);
}

// Rows r < gridDim.x of src (src_ld bytes apart) to row pos0 + r of a position-indexed table dst (dst_ld bytes a
// row), 16 bytes a thread, width16 of them a row; `posp` not null: pos0 is read there on the device (a captured
// forward); `ring` > 0: dst is a mirrored ring (as tf_ink2_dspark_rope's). The drafter's context values
// (cuda_dspark.zig absorb), in place of a 2D copy to dst + pos0 * dst_ld.
// Paged (tf_ink2_dspark_rows_pg): row p at row p % SPLIT of page pages[p / SPLIT], `page` BYTES apart (as
// tf_ink2_dspark_rope_pg's rows); one source with the ring form.
template <bool PAGED>
__device__ __forceinline__ void rows_body(const unsigned char* __restrict__ src, long long src_ld,
                                          unsigned char* __restrict__ dst, long long dst_ld, int width16,
                                          long long pos0, const int* __restrict__ posp, long long ring,
                                          long long mirror, const int* __restrict__ pages, long long page) {
    if (posp != nullptr) pos0 = *posp;
    const long long r = blockIdx.x;
    const uint4* s = reinterpret_cast<const uint4*>(src + r * src_ld);
    const long long row = ring > 0 ? (pos0 + r) % ring : pos0 + r;
    uint4* d;
    if constexpr (PAGED)
        d = reinterpret_cast<uint4*>(dst + (long long)pages[row / SPLIT] * page + (row % SPLIT) * dst_ld);
    else
        d = reinterpret_cast<uint4*>(dst + row * dst_ld);
    for (int i = threadIdx.x; i < width16; i += blockDim.x) d[i] = s[i];
    if (!PAGED && ring > 0 && row < mirror) {
        uint4* m = reinterpret_cast<uint4*>(dst + (row + ring) * dst_ld);
        for (int i = threadIdx.x; i < width16; i += blockDim.x) m[i] = s[i];
    }
}

extern "C" __global__ void tf_ink2_dspark_rows(const unsigned char* __restrict__ src, long long src_ld,
                                               unsigned char* __restrict__ dst, long long dst_ld, int width16,
                                               long long pos0, const int* __restrict__ posp, long long ring,
                                               long long mirror) {
    rows_body<false>(src, src_ld, dst, dst_ld, width16, pos0, posp, ring, mirror, nullptr, 0);
}

extern "C" __global__ void tf_ink2_dspark_rows_pg(const unsigned char* __restrict__ src, long long src_ld,
                                                  unsigned char* __restrict__ dst, long long dst_ld, int width16,
                                                  long long pos0, const int* __restrict__ posp,
                                                  const int* __restrict__ pages, long long page) {
    rows_body<true>(src, src_ld, dst, dst_ld, width16, pos0, posp, 0, 0, pages, page);
}

// The block attention, part 1. Query rows q bf16 [R, hq * 128] (R <= 8); keys: context positions lo .. lo + nctx - 1
// at kc / vc + pos * kv_ld (position-indexed rows, this rank's hk heads), then the block's R rows bk [R, hk * 128]
// and bv (rows bv_ld apart); every query row sees every key (non-causal in the block). Query head h reads key head h / group.
// A CTA a (split of SPLIT keys, query head), 256 threads: thread t scores key t of the split for all rows; then each
// row's split max, exp sum and p . v. Partials: ml fp32 [hq][nsplit][R][2], acc fp32 [hq][nsplit][R][128].
//
// Paged (tf_ink2_dspark_attn_pg, the global layer: lo = 0, so split s is page s): context key k at row k % SPLIT of
// page pages[k / SPLIT], `page` elements apart (kc / vc this layer's rows in page 0); the CTA reads its split's page id
// once, and a key's address is that base plus its row. One source with the table form (`PAGED`).
template <bool PAGED>
__device__ __forceinline__ void attn_body(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    long long kv_ld, long long lo, int nctx, const __nv_bfloat16* __restrict__ bk, const __nv_bfloat16* __restrict__ bv,
    long long bv_ld, int R, int hq, int group, float scale, float* __restrict__ ml, float* __restrict__ acc,
    const int* __restrict__ pages, long long page) {
    if constexpr (PAGED) {
        // the split's page (a split past the context holds only the block's keys and reads no page)
        if ((int)blockIdx.x * SPLIT < nctx) {
            const long long off = (long long)pages[blockIdx.x] * page - (long long)blockIdx.x * SPLIT * kv_ld;
            kc += off;
            vc += off;
        }
    }
    __shared__ float qs[8][D];
    __shared__ float ps[8][SPLIT];
    __shared__ float rm[8], rl[8];
    __shared__ float red[8][8];
    const int split = blockIdx.x, nsplit = gridDim.x, h = blockIdx.y, kvh = h / group;
    const int t = threadIdx.x;
    const int hkD = (hq / group) * D;
    for (int i = t; i < R * D; i += 256) qs[i / D][i % D] = __bfloat162float(q[(i / D) * (hq * D) + h * D + i % D]);
    __syncthreads();
    const int nkeys = nctx + R;
    const int key = split * SPLIT + t;
    float s[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) s[r] = -INFINITY;
    if (key < nkeys) {
        const __nv_bfloat16* kr = key < nctx ? kc + (lo + key) * kv_ld + kvh * D : bk + (key - nctx) * hkD + kvh * D;
        float a[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) a[r] = 0.f;
        for (int d0 = 0; d0 < D; d0 += 8) {
            float kf[8];
            load8(kr + d0, kf);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r < R) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) a[r] = __fmaf_rn(qs[r][d0 + i], kf[i], a[r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) if (r < R) s[r] = __fmul_rn(a[r], scale);
    }
    // each row's max over the split
    const int w = t >> 5, ln = t & 31;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float m = s[r];
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        if (ln == 0) red[r][w] = m;
    }
    __syncthreads();
    if (t < 8) {
        float m = -INFINITY;
        for (int i = 0; i < 8; ++i) m = fmaxf(m, red[t][i]);
        rm[t] = m;
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float p = (r < R && key < nkeys) ? expf(__fsub_rn(s[r], rm[r])) : 0.f;
        ps[r][t] = p;
        const float e = warp_sum(p);
        if (ln == 0) red[r][w] = e;
    }
    __syncthreads();
    if (t < 8) {
        float l = 0.f;
        for (int i = 0; i < 8; ++i) l = __fadd_rn(l, red[t][i]);
        rl[t] = l;
    }
    __syncthreads();
    // p . v: thread t owns dim t % 128 for rows t / 128, t / 128 + 2, ...
    const int d = t & (D - 1);
    const int kn = min(SPLIT, nkeys - split * SPLIT);
    for (int r = t / D; r < R; r += 2) {
        float o = 0.f;
        for (int j = 0; j < kn; ++j) {
            const int k = split * SPLIT + j;
            const __nv_bfloat16* vr = k < nctx ? vc + (lo + k) * kv_ld + kvh * D : bv + (k - nctx) * bv_ld + kvh * D;
            o = __fmaf_rn(ps[r][j], __bfloat162float(vr[d]), o);
        }
        const int64_t at = (static_cast<int64_t>(h) * nsplit + split) * R + r;
        acc[at * D + d] = o;
        if (d == 0) {
            ml[at * 2] = rm[r];
            ml[at * 2 + 1] = rl[r];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_dspark_attn(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    long long kv_ld, long long lo, int nctx, const __nv_bfloat16* __restrict__ bk, const __nv_bfloat16* __restrict__ bv,
    long long bv_ld, int R, int hq, int group, float scale, float* __restrict__ ml, float* __restrict__ acc) {
    attn_body<false>(q, kc, vc, kv_ld, lo, nctx, bk, bv, bv_ld, R, hq, group, scale, ml, acc, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_dspark_attn_pg(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    long long kv_ld, int nctx, const __nv_bfloat16* __restrict__ bk, const __nv_bfloat16* __restrict__ bv,
    long long bv_ld, int R, int hq, int group, float scale, float* __restrict__ ml, float* __restrict__ acc,
    const int* __restrict__ pages, long long page) {
    attn_body<true>(q, kc, vc, kv_ld, 0, nctx, bk, bv, bv_ld, R, hq, group, scale, ml, acc, pages, page);
}

// Part 2: each (head, row)'s splits merged in order: out bf16 [R, hq * 128]. Grid (hq, R), 128 threads.
extern "C" __global__ void tf_ink2_dspark_attn_merge(const float* __restrict__ ml, const float* __restrict__ acc,
                                                     int nsplit, int R, int hq, __nv_bfloat16* __restrict__ out) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, ml[((static_cast<int64_t>(h) * nsplit + s) * R + r) * 2]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const int64_t at = (static_cast<int64_t>(h) * nsplit + s) * R + r;
        const float m = ml[at * 2];
        if (m == -INFINITY) continue;
        const float f = expf(__fsub_rn(m, M));
        L = __fadd_rn(L, __fmul_rn(ml[at * 2 + 1], f));
        o = __fadd_rn(o, __fmul_rn(acc[at * D + d], f));
    }
    out[static_cast<int64_t>(r) * hq * D + h * D + d] = __float2bfloat16_rn(__fdiv_rn(o, L));
}

// act bf16 [R, I] = bf16(silu(f32(gate)) * f32(up)); gu bf16 [R, 2I] (the rank's gate columns, then its up columns).
// Grid (ceil(I / 256), R), 256 threads.
extern "C" __global__ void tf_ink2_dspark_swiglu(const __nv_bfloat16* __restrict__ gu, int I,
                                                 __nv_bfloat16* __restrict__ act) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= I) return;
    const int64_t r = blockIdx.y;
    const float g = __bfloat162float(gu[r * 2 * I + c]);
    const float u = __bfloat162float(gu[r * 2 * I + I + c]);
    const float sg = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    act[r * I + c] = __float2bfloat16_rn(__fmul_rn(sg, u));
}

// The Markov head's recurrent step s (one CTA, 768 threads): projected = bf16(inv[o] + bias[o] + state . W[o, 0:256]
// + w1[prev] . W[o, 256:512]) with W the rnn_proj weight bf16 [768, ld] (inv: the step-invariant fp32 product of the
// row and the features); gate, candidate, output thirds; state = bf16(g * state + (1 - g) * tanh(c)) with
// g = sigmoid(gate); latent = bf16(tanh(out)). prev: walk[s] (the token before this step's).
// Several streams' drafts at once (cuda_dspark.zig draftMany): grid (1, streams), stream y's inv at inv + y * inv_ys,
// its walk at walk + 8 y, its state and latent at + RANK y; each CTA's arithmetic is a lone stream's.
extern "C" __global__ void __launch_bounds__(768) tf_ink2_dspark_mstep(
    const float* __restrict__ inv, const __nv_bfloat16* __restrict__ bias, const __nv_bfloat16* __restrict__ W,
    long long ld, const __nv_bfloat16* __restrict__ w1, const int* __restrict__ walk, int s,
    __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ latent, long long inv_ys) {
    __shared__ float st[RANK], pv[RANK], pj[3 * RANK];
    const int64_t y = blockIdx.y;
    inv += y * inv_ys;
    walk += y * 8;
    state += y * RANK;
    latent += y * RANK;
    const int o = threadIdx.x;
    const int64_t prev = walk[s];
    if (o < RANK) {
        st[o] = __bfloat162float(state[o]);
        pv[o] = __bfloat162float(w1[prev * RANK + o]);
    }
    __syncthreads();
    const __nv_bfloat16* wr = W + static_cast<int64_t>(o) * ld;
    float a = 0.f;
    for (int k = 0; k < RANK; k += 8) {
        float wf[8];
        load8(wr + k, wf);
#pragma unroll
        for (int i = 0; i < 8; ++i) a = __fmaf_rn(st[k + i], wf[i], a);
    }
    for (int k = 0; k < RANK; k += 8) {
        float wf[8];
        load8(wr + RANK + k, wf);
#pragma unroll
        for (int i = 0; i < 8; ++i) a = __fmaf_rn(pv[k + i], wf[i], a);
    }
    pj[o] = bfr(__fadd_rn(__fadd_rn(inv[o], __bfloat162float(bias[o])), a));
    __syncthreads();
    if (o < RANK) {
        const float g = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-pj[o])));
        const float c = tanhf(pj[RANK + o]);
        const float ns = __fadd_rn(__fmul_rn(g, st[o]), __fmul_rn(__fsub_rn(1.0f, g), c));
        state[o] = __float2bfloat16_rn(ns);
        latent[o] = __float2bfloat16_rn(tanhf(pj[2 * RANK + o]));
    }
}

// The corrected, capped logits of this rank's vocabulary slice for one step, and each CTA's best: a warp a column v:
// corr = bf16(fold_b[v] + latent . fold_w[v]); logit = bf16(base[v] + corr); capped = bf16(tanh(logit * fp32(1/cap))
// * cap); columns at or past `valid` (local) are skipped. best: (value, local column) of the CTA's columns, ties to the
// lower column. Grid nblk, 256 threads (8 warps); best fp32 / i32 [nblk]. Several streams: grid (nblk, streams),
// stream y's base row at base + y * base_ys, its latent at + RANK y, its bests at + nblk y.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_dspark_mlogits(
    const __nv_bfloat16* __restrict__ base, const __nv_bfloat16* __restrict__ latent,
    const __nv_bfloat16* __restrict__ fold_w, const __nv_bfloat16* __restrict__ fold_b, int n, int valid, float cap,
    float* __restrict__ best_v, int* __restrict__ best_i, long long base_ys) {
    __shared__ float lt[RANK];
    {
        const int64_t y = blockIdx.y;
        base += y * base_ys;
        latent += y * RANK;
        best_v += y * gridDim.x;
        best_i += y * gridDim.x;
    }
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int t = threadIdx.x, w = t >> 5, ln = t & 31;
    lt[t] = __bfloat162float(latent[t]);
    __syncthreads();
    float mv = -INFINITY;
    int mi = 0x7fffffff;
    const float inv = __fdiv_rn(1.0f, cap);
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int c0 = blockIdx.x * per, c1 = min(n, c0 + per);
    for (int v = c0 + w; v < c1; v += 8) {
        if (v >= valid) break;
        float wf[8];
        load8(fold_w + static_cast<int64_t>(v) * RANK + ln * 8, wf);
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) a = __fmaf_rn(lt[ln * 8 + i], wf[i], a);
        a = warp_sum(a);
        const float corr = bfr(__fadd_rn(a, __bfloat162float(fold_b[v])));
        const float lg = bfr(__fadd_rn(__bfloat162float(base[v]), corr));
        const float cp = bfr(__fmul_rn(tanhf(__fmul_rn(lg, inv)), cap));
        if (cp > mv) {  // columns rise within the warp: the first of equals stays
            mv = cp;
            mi = v;
        }
    }
    if (ln == 0) {
        bv[w] = mv;
        bi[w] = mi;
    }
    __syncthreads();
    if (t == 0) {
        float v = -INFINITY;
        int i = 0x7fffffff;
        for (int k = 0; k < 8; ++k)
            if (bv[k] > v || (bv[k] == v && bi[k] < i)) {
                v = bv[k];
                i = bi[k];
            }
        best_v[blockIdx.x] = v;
        best_i[blockIdx.x] = i;
    }
}

// The rank's pick over the CTAs' bests (ties to the lower column), as (value, global id) at pair (a float and an
// int, 8 bytes); with walk != null (one rank) also walk[s + 1] = the id. One CTA, 32 threads. Several streams: grid
// (1, streams), stream y's bests at + nblk y, its pair at pair + pair_ys bytes y, its walk at + 8 y.
extern "C" __global__ void tf_ink2_dspark_mpick(const float* __restrict__ best_v, const int* __restrict__ best_i,
                                                int nblk, int vocab_lo, float* __restrict__ pair, int* walk, int s,
                                                int pair_ys) {
    if (threadIdx.x != 0) return;
    {
        const int64_t y = blockIdx.y;
        best_v += y * nblk;
        best_i += y * nblk;
        pair = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(pair) + y * pair_ys);
        if (walk != nullptr) walk += y * 8;
    }
    float v = -INFINITY;
    int i = 0x7fffffff;
    for (int k = 0; k < nblk; ++k)
        if (best_v[k] > v || (best_v[k] == v && best_i[k] < i)) {
            v = best_v[k];
            i = best_i[k];
        }
    const int id = (i == 0x7fffffff) ? 0 : vocab_lo + i;
    pair[0] = v;
    reinterpret_cast<int*>(pair)[1] = id;
    if (walk != nullptr) walk[s + 1] = id;
}

// Every rank's pair (rank order, `stride` bytes apart) merged: the best value, ties to the lower rank (its ids are
// lower); walk[s + 1] = its id. One thread. Several streams: grid (1, streams), stream y's pairs at + pair_ys bytes y
// within each rank's, its walk at + 8 y.
extern "C" __global__ void tf_ink2_dspark_mmerge(const unsigned char* __restrict__ pairs, int world, int stride,
                                                 int* walk, int s, int pair_ys) {
    if (threadIdx.x != 0) return;
    {
        const int64_t y = blockIdx.y;
        pairs += y * pair_ys;
        walk += y * 8;
    }
    float v = -INFINITY;
    int id = 0;
    for (int r = 0; r < world; ++r) {
        const float pv = *reinterpret_cast<const float*>(pairs + r * stride);
        const int pi = *reinterpret_cast<const int*>(pairs + r * stride + 4);
        if (r == 0 || pv > v) {
            v = pv;
            id = pi;
        }
    }
    walk[s + 1] = id;
}

// The rows the Markov head's step-invariant product reads: x bf16 [n, 2H] = [row j | features]. Grid n, NT threads.
extern "C" __global__ void tf_ink2_dspark_mcat(const __nv_bfloat16* __restrict__ rows, long long rows_ld,
                                               const __nv_bfloat16* __restrict__ feat, __nv_bfloat16* __restrict__ x) {
    const int64_t r = blockIdx.x;
    const int c0 = threadIdx.x * VEC;
    *reinterpret_cast<uint4*>(x + r * 2 * H + c0) = *reinterpret_cast<const uint4*>(rows + r * rows_ld + c0);
    *reinterpret_cast<uint4*>(x + r * 2 * H + H + c0) = *reinterpret_cast<const uint4*>(feat + c0);
}
