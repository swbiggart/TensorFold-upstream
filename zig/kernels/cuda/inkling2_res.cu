// Inkling-2's residual group (G-res): the block attention residual's mix over the stored block states and the
// current partial, the partial update and the layer output. The entry points, shapes and math are in
// zig/src/families/inkling2/cuda_res.zig.
//
// The math follows Transformers' modeling_inkling2.py (Apache-2.0, Copyright 2026 the HuggingFace Team):
// Inkling2BlockAttentionResidual (Inkling2UnweightedRMSNorm on the fp32 sources, then scaled dot-product attention
// of the fp32 queries over them) and the decoder layers' residual adds. Rewritten here as row kernels.
//
// Row invariance: a row is one CTA and every sum runs in an order fixed by the row's width alone (each thread's
// 8 columns in order, a warp's xor tree, then the warps in order), so a row's bits do not depend on the row count.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_ink2_res {

constexpr int H = 4608;          // hidden
constexpr int VEC = 8;           // columns a thread
constexpr int NT = H / VEC;      // 576 threads a row
constexpr int WARPS = NT / 32;   // 18
constexpr int MAX_SRC = 9;       // 8 blocks and the candidate
constexpr int MAX_Q = 3;         // a global layer's q, k, v sources

static_assert(H % (VEC * 32) == 0, "a row is whole warps of 8 columns");

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

__device__ __forceinline__ void store8(float* p, const float* f) {
    reinterpret_cast<float4*>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
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

// The warps' sums in warp order (every thread reads the same values, so every thread gets the same total).
__device__ __forceinline__ float warps_sum(const float* sh) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) t = __fadd_rn(t, sh[w]);
    return t;
}

// One row of Mix: sources s_0 .. s_{nb-1} = the blocks (bf16), s_nb = cand (bf16 or fp32), all fp32;
//   n_i = s_i * rsqrt(mean(s_i^2) + eps); score_ji = (q_j . n_i) * scale; p_j = softmax_i(score_j);
//   out_j = sum_i p_ji n_i (fp32), for j < nq.
// Torch's order where it is cheap to keep: square then sum, the mean as a product with fp32(1/H), rsqrtf,
// n_i elementwise before the dot, exp(s - max) / sum, and p_ji n_i accumulated in source order.
template <typename C>
__device__ __forceinline__ void mix_row(const __nv_bfloat16* __restrict__ blocks, long long block_stride, int nblocks,
                                        const C* __restrict__ cand, const float* __restrict__ queries, int nq,
                                        float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2,
                                        float eps, float scale) {
    __shared__ float sh_ss[MAX_SRC][WARPS];
    __shared__ float sh_dot[MAX_SRC][MAX_Q][WARPS];
    __shared__ float sh_inv[MAX_SRC];
    __shared__ float sh_p[MAX_Q][MAX_SRC];
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = t * VEC;
    const int ns = nblocks + 1;
    float q[MAX_Q][VEC];
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
        if (j < nq) load8(queries + j * H + c0, q[j]);
    // pass 1: each source's inverse RMS and its scores
#pragma unroll 1
    for (int i = 0; i < ns; ++i) {
        float v[VEC];
        if (i < nblocks) load8(blocks + i * block_stride + row * H + c0, v);
        else load8(cand + row * H + c0, v);
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < VEC; ++k) ss = __fadd_rn(ss, __fmul_rn(v[k], v[k]));
        ss = warp_sum(ss);
        if (lane == 0) sh_ss[i][warp] = ss;
        __syncthreads();
        const float mean = __fmul_rn(warps_sum(sh_ss[i]), 1.0f / H);
        const float inv = rsqrtf(__fadd_rn(mean, eps));
        if (t == 0) sh_inv[i] = inv;
        float d[MAX_Q] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            const float n = __fmul_rn(v[k], inv);
#pragma unroll
            for (int j = 0; j < MAX_Q; ++j)
                if (j < nq) d[j] = __fmaf_rn(q[j][k], n, d[j]);
        }
#pragma unroll
        for (int j = 0; j < MAX_Q; ++j)
            if (j < nq) {
                const float s = warp_sum(d[j]);
                if (lane == 0) sh_dot[i][j][warp] = s;
            }
    }
    __syncthreads();
    // the softmax over the sources, one thread a query
    if (t < nq) {
        float s[MAX_SRC];
        float m = -__int_as_float(0x7f800000);
#pragma unroll
        for (int i = 0; i < MAX_SRC; ++i)
            if (i < ns) {
                s[i] = __fmul_rn(warps_sum(sh_dot[i][t]), scale);
                m = fmaxf(m, s[i]);
            }
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < MAX_SRC; ++i)
            if (i < ns) {
                s[i] = expf(__fsub_rn(s[i], m));
                sum = __fadd_rn(sum, s[i]);
            }
#pragma unroll
        for (int i = 0; i < MAX_SRC; ++i)
            if (i < ns) sh_p[t][i] = __fdiv_rn(s[i], sum);
    }
    __syncthreads();
    // pass 2: the weighted sum of the normalized sources
    float acc[MAX_Q][VEC];
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
#pragma unroll
        for (int k = 0; k < VEC; ++k) acc[j][k] = 0.f;
#pragma unroll 1
    for (int i = 0; i < ns; ++i) {
        float v[VEC];
        if (i < nblocks) load8(blocks + i * block_stride + row * H + c0, v);
        else load8(cand + row * H + c0, v);
        const float inv = sh_inv[i];
        float p[MAX_Q];
#pragma unroll
        for (int j = 0; j < MAX_Q; ++j) p[j] = j < nq ? sh_p[j][i] : 0.f;
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            const float n = __fmul_rn(v[k], inv);
#pragma unroll
            for (int j = 0; j < MAX_Q; ++j) acc[j][k] = __fmaf_rn(p[j], n, acc[j][k]);
        }
    }
    float* outs[MAX_Q] = {out0, out1, out2};
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
        if (j < nq) store8(outs[j] + row * H + c0, acc[j]);
}

}  // namespace tf_ink2_res

using namespace tf_ink2_res;

// The module's presence check (cuda_kernels.zig Group): writes the group's number.
extern "C" __global__ void tf_ink2_res_probe(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 3;
}

// Mix with a bf16 candidate (attn_input: h_in; final_input: h_out(47)). Grid: rows; block: NT threads.
// blocks: block i of row r at blocks + i * block_stride + r * H (elements); out_j dense [rows, H] fp32.
extern "C" __global__ void __launch_bounds__(NT) tf_ink2_res_mix_bf16(const __nv_bfloat16* blocks, long long block_stride,
                                                                      int nblocks, const __nv_bfloat16* cand,
                                                                      const float* queries, int nq, float* out0,
                                                                      float* out1, float* out2, float eps, float scale) {
    mix_row(blocks, block_stride, nblocks, cand, queries, nq, out0, out1, out2, eps, scale);
}

// Mix with an fp32 candidate (mlp_input: the partial).
extern "C" __global__ void __launch_bounds__(NT) tf_ink2_res_mix_f32(const __nv_bfloat16* blocks, long long block_stride,
                                                                     int nblocks, const float* cand,
                                                                     const float* queries, int nq, float* out0,
                                                                     float* out1, float* out2, float eps, float scale) {
    mix_row(blocks, block_stride, nblocks, cand, queries, nq, out0, out1, out2, eps, scale);
}

// partial fp32 = boundary ? f32(a) : f32(h_in) + f32(a); n elements (a multiple of 8), 8 a thread.
extern "C" __global__ void tf_ink2_res_partial(const __nv_bfloat16* __restrict__ h_in, const __nv_bfloat16* __restrict__ a,
                                               float* __restrict__ out, int boundary, long long n) {
    const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;
    if (at >= n) return;
    float x[VEC], y[VEC];
    load8(a + at, y);
    if (!boundary) {
        load8(h_in + at, x);
#pragma unroll
        for (int k = 0; k < VEC; ++k) y[k] = __fadd_rn(x[k], y[k]);
    }
    store8(out + at, y);
}

// h bf16 = bf16(partial + f32(mconv)); n elements (a multiple of 8), 8 a thread.
extern "C" __global__ void tf_ink2_res_output(const float* __restrict__ partial, const __nv_bfloat16* __restrict__ mconv,
                                              __nv_bfloat16* __restrict__ out, long long n) {
    const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;
    if (at >= n) return;
    float x[VEC], y[VEC];
    load8(partial + at, x);
    load8(mconv + at, y);
#pragma unroll
    for (int k = 0; k < VEC; ++k) y[k] = __fadd_rn(x[k], y[k]);
    store8(out + at, y);
}

// tf_ink2_res_rank_sum over rows of the partials: rank r's n elements at parts + r * stride (the rows a rank sums of
// a prompt chunk's [world][R][H] partials, cuda_xrows.zig); the same adds in the same order, so the same bits.
extern "C" __global__ void tf_ink2_res_rank_sum_rows(const float* __restrict__ parts, int world, long long stride,
                                                     __nv_bfloat16* __restrict__ out, long long n) {
    const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;
    if (at >= n) return;
    float x[VEC], y[VEC];
    load8(parts + at, x);
    for (int r = 1; r < world; ++r) {
        load8(parts + static_cast<int64_t>(r) * stride + at, y);
#pragma unroll
        for (int k = 0; k < VEC; ++k) x[k] = __fadd_rn(x[k], y[k]);
    }
    store8(out + at, x);
}

// W > 1: out bf16 = bf16(parts[0] + parts[1] + ... + parts[world - 1]), each rank's fp32 partial (n elements, rank r's
// at parts + r * n) added in rank order, one rounding at the end; every rank runs it on the same gathered bytes, so
// every rank's output is the same bits. n a multiple of 8, 8 a thread.
extern "C" __global__ void tf_ink2_res_rank_sum(const float* __restrict__ parts, int world, __nv_bfloat16* __restrict__ out,
                                                long long n) {
    const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;
    if (at >= n) return;
    float x[VEC], y[VEC];
    load8(parts + at, x);
    for (int r = 1; r < world; ++r) {
        load8(parts + static_cast<int64_t>(r) * n + at, y);
#pragma unroll
        for (int k = 0; k < VEC; ++k) x[k] = __fadd_rn(x[k], y[k]);
    }
    store8(out + at, x);
}
