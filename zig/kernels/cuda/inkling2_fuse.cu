// Inkling-2's fused stateless neighbours (G-fuse): one launch where the forward ran two to four, each the same
// expressions in the same order as the kernels it replaces, so the same bits. The entry points, shapes and the
// kernels each one replaces are in zig/src/families/inkling2/cuda_fuse.zig.
//
//   tf_ink2_fuse_attn_in   res.attn_input (inkling2_res.cu tf_ink2_res_mix_bf16) + the attention norms
//                          (norms.cu tf_rms_rows on the fp32 mix outputs: res.attn_norm, or res.q/k/v_norm)
//   tf_ink2_fuse_mlp_in    res.partial (tf_ink2_res_partial) + the boundary push (a row copy) + res.mlp_input
//                          (tf_ink2_res_mix_f32) + res.mlp_norm (tf_rms_rows)
//   tf_ink2_fuse_route     moe.route's select (inkling2_moe.cu tf_ink2_moe_select) + checkpoint mode's slot order
//                          (inkling2_ck.cu tf_ink2_ck_slot_order)
//
// The row kernels run a CTA a row of norms.cu's 1,024 threads: the mix on the first 576 (inkling2_res.cu's 8 columns
// a thread, its warps and its sums in its order), each norm on all 1,024 (tf_rms_rows' columns t + 1024 j, its fma
// chain, its warp trees and rms_scale) over the mix output staged in shared memory. A row's bits do not depend on R.
//
// The same ops' latency forms (tf_ink2_lat_*: the decode default) follow at the end.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/reduce.cuh"

namespace tf_ink2_fuse {

constexpr int H = 4608;               // hidden
constexpr int VEC = 8;                // columns a mix thread
constexpr int MT = H / VEC;           // 576 mix threads a row (inkling2_res.cu's NT)
constexpr int MWARPS = MT / 32;       // 18
constexpr int NT = 1024;              // threads a row: norms.cu's tf_rms_rows block
constexpr int NWARPS = NT / 32;       // 32
constexpr int NPER = (H + NT - 1) / NT;  // 5 columns a norm thread (t + 1024 j)
constexpr int MAX_SRC = 9;            // 8 blocks and the candidate
constexpr int MAX_Q = 3;              // a global layer's q, k, v sources

static_assert(MT % 32 == 0 && MT <= NT, "the mixers are whole warps of the block");

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

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    return v;
}

// The mix's warps' sums in warp order (inkling2_res.cu warps_sum).
__device__ __forceinline__ float mix_warps_sum(const float* sh) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < MWARPS; ++w) t = __fadd_rn(t, sh[w]);
    return t;
}

struct MixShared {
    float ss[MAX_SRC][MWARPS];
    float dot[MAX_SRC][MAX_Q][MWARPS];
    float inv[MAX_SRC];
    float p[MAX_Q][MAX_SRC];
};

// inkling2_res.cu mix_row on threads 0 .. 575 (the rest only meet the barriers): source i's 8 columns of this thread
// come from `src(i, v)`; acc[j] is out_j's 8 columns c0 .. c0 + 7 (mixers only).
template <typename Src>
__device__ __forceinline__ void mix(MixShared& sh, const Src& src, int ns, const float* __restrict__ queries, int nq,
                                    float eps, float scale, float (&acc)[MAX_Q][VEC]) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const bool mixer = t < MT;
    const int c0 = t * VEC;
    float q[MAX_Q][VEC];
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
        if (mixer && j < nq) load8(queries + j * H + c0, q[j]);
    // pass 1: each source's inverse RMS and its scores
#pragma unroll 1
    for (int i = 0; i < ns; ++i) {
        float v[VEC];
        if (mixer) {
            src(i, v);
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < VEC; ++k) ss = __fadd_rn(ss, __fmul_rn(v[k], v[k]));
            ss = warp_sum(ss);
            if (lane == 0) sh.ss[i][warp] = ss;
        }
        __syncthreads();
        if (mixer) {
            const float mean = __fmul_rn(mix_warps_sum(sh.ss[i]), 1.0f / H);
            const float inv = rsqrtf(__fadd_rn(mean, eps));
            if (t == 0) sh.inv[i] = inv;
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
                    if (lane == 0) sh.dot[i][j][warp] = s;
                }
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
                s[i] = __fmul_rn(mix_warps_sum(sh.dot[i][t]), scale);
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
            if (i < ns) sh.p[t][i] = __fdiv_rn(s[i], sum);
    }
    __syncthreads();
    // pass 2: the weighted sum of the normalized sources
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
#pragma unroll
        for (int k = 0; k < VEC; ++k) acc[j][k] = 0.f;
    if (!mixer) return;
#pragma unroll 1
    for (int i = 0; i < ns; ++i) {
        float v[VEC];
        src(i, v);
        const float inv = sh.inv[i];
        float p[MAX_Q];
#pragma unroll
        for (int j = 0; j < MAX_Q; ++j) p[j] = j < nq ? sh.p[j][i] : 0.f;
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            const float n = __fmul_rn(v[k], inv);
#pragma unroll
            for (int j = 0; j < MAX_Q; ++j) acc[j][k] = __fmaf_rn(p[j], n, acc[j][k]);
        }
    }
}

struct NormShared {
    float x[H];          // the row being normed (fp32, as the mix stored it)
    float warps[NWARPS];
    float inv;
};

// One row of norms.cu tf_rms_rows (fp32 x, bf16 y, the cached path of 8 columns a thread at t + 1024 j) over the
// mix's out_j (a mixer's acc[j]): y = bf16(x * rms_scale(sum x^2) * w), w null: no gain. Every thread calls.
__device__ __forceinline__ void norm(NormShared& sh, const float (&acc)[MAX_Q][VEC], int j, const float* __restrict__ w,
                                     __nv_bfloat16* __restrict__ y, float eps) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    __syncthreads();  // an earlier norm's readers of sh are done
    if (t < MT) {
#pragma unroll
        for (int jj = 0; jj < MAX_Q; ++jj)
            if (jj == j) {
#pragma unroll
                for (int k = 0; k < VEC; ++k) sh.x[t * VEC + k] = acc[jj][k];
            }
    }
    __syncthreads();
    float v[NPER];
    float part = 0.f;
#pragma unroll
    for (int k = 0; k < NPER; ++k) {  // tf_rms_rows' zeros past the row add nothing to a non-negative sum
        const int i = t + k * NT;
        v[k] = i < H ? sh.x[i] : 0.f;
        if (i < H) part = __fmaf_rn(v[k], v[k], part);
    }
    // norms.cu inv_rms: the warp's xor tree, then the 32 warps' sums in one more xor tree, then rms_scale
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if (lane == 0) sh.warps[warp] = part;
    __syncthreads();
    if (t < 32) {
        float s = sh.warps[t];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, m));
        if (t == 0) sh.inv = tf_common::rms_scale(s, H, eps);
    }
    __syncthreads();
    const float inv = sh.inv;
#pragma unroll
    for (int k = 0; k < NPER; ++k) {
        const int i = t + k * NT;
        if (i < H) y[i] = __float2bfloat16_rn(w != nullptr ? __fmul_rn(__fmul_rn(v[k], inv), w[i]) : __fmul_rn(v[k], inv));
    }
}

// attn_in's sources: the blocks, then the bf16 candidate (h_in).
struct AttnSrc {
    const __nv_bfloat16* blocks;
    long long stride;
    int nblocks;
    const __nv_bfloat16* cand;
    int64_t row;
    int c0;
    __device__ __forceinline__ void operator()(int i, float* v) const {
        if (i < nblocks) load8(blocks + i * stride + row * H + c0, v);
        else load8(cand + row * H + c0, v);
    }
};

// mlp_in's sources: the blocks below the layer, h_in when the layer pushes it (a boundary), then the partial (fp32,
// this thread's own 8 columns, as tf_ink2_res_partial stored them).
struct MlpSrc {
    const __nv_bfloat16* blocks;
    long long stride;
    int nblocks;
    const __nv_bfloat16* pushed;  // null: no push
    int64_t row;
    int c0;
    float part[VEC];
    __device__ __forceinline__ void operator()(int i, float* v) const {
        if (i < nblocks) load8(blocks + i * stride + row * H + c0, v);
        else if (pushed != nullptr && i == nblocks) load8(pushed + row * H + c0, v);
        else {
#pragma unroll
            for (int k = 0; k < VEC; ++k) v[k] = part[k];
        }
    }
};

}  // namespace tf_ink2_fuse

using namespace tf_ink2_fuse;

// The module's presence check (cuda_kernels.zig Group): writes the group's number.
extern "C" __global__ void tf_ink2_fuse_probe(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 8;
}

// res.attn_input then its norms. Grid: rows; block: NT. The mix as tf_ink2_res_mix_bf16 (out_j fp32, stored), then
// nn norms: norm j of out_{nq == 1 ? 0 : j} with gain g_j (null: none) into y_j bf16 [rows, H].
extern "C" __global__ void __launch_bounds__(NT)
    tf_ink2_fuse_attn_in(const __nv_bfloat16* blocks, long long block_stride, int nblocks, const __nv_bfloat16* cand,
                         const float* queries, int nq, float* out0, float* out1, float* out2, float eps, float scale,
                         int nn, const float* g0, const float* g1, const float* g2, __nv_bfloat16* y0,
                         __nv_bfloat16* y1, __nv_bfloat16* y2, float norm_eps) {
    __shared__ MixShared msh;
    __shared__ NormShared nsh;
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const AttnSrc src{blocks, block_stride, nblocks, cand, row, t * VEC};
    float acc[MAX_Q][VEC];
    mix(msh, src, nblocks + 1, queries, nq, eps, scale, acc);
    if (t < MT) {
        float* outs[MAX_Q] = {out0, out1, out2};
#pragma unroll
        for (int j = 0; j < MAX_Q; ++j)
            if (j < nq) store8(outs[j] + row * H + t * VEC, acc[j]);
    }
    const float* gs[MAX_Q] = {g0, g1, g2};
    __nv_bfloat16* ys[MAX_Q] = {y0, y1, y2};
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
        if (j < nn) norm(nsh, acc, nq == 1 ? 0 : j, gs[j], ys[j] + row * H, norm_eps);
}

// res.partial, the boundary push, res.mlp_input and res.mlp_norm. Grid: rows; block: NT.
//   partial fp32 = boundary ? f32(a) : f32(h_in) + f32(a)                       (tf_ink2_res_partial)
//   boundary: push (block `nblocks`) = h_in, bit for bit                         (state.pushBlock's row copy)
//   mx fp32 = Mix(query; blocks 0 .. nblocks - 1 [, h_in], partial)             (tf_ink2_res_mix_f32)
//   m bf16 = bf16(mx * rms_scale * gain)                                         (tf_rms_rows)
extern "C" __global__ void __launch_bounds__(NT)
    tf_ink2_fuse_mlp_in(const __nv_bfloat16* h_in, const __nv_bfloat16* a, int boundary, float* partial,
                        const __nv_bfloat16* blocks, long long block_stride, int nblocks, __nv_bfloat16* push,
                        const float* query, float* mx, float eps, float scale, const float* gain, __nv_bfloat16* m,
                        float norm_eps) {
    __shared__ MixShared msh;
    __shared__ NormShared nsh;
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int c0 = t * VEC;
    const int64_t at = row * H + c0;
    float pv[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (t < MT) {
        float x[VEC];
        load8(a + at, pv);
        if (!boundary) {
            load8(h_in + at, x);
#pragma unroll
            for (int k = 0; k < VEC; ++k) pv[k] = __fadd_rn(x[k], pv[k]);
        } else {
            *reinterpret_cast<uint4*>(push + at) = *reinterpret_cast<const uint4*>(h_in + at);
        }
        store8(partial + at, pv);
    }
    MlpSrc src{blocks, block_stride, nblocks, boundary ? h_in : nullptr, row, c0};
#pragma unroll
    for (int k = 0; k < VEC; ++k) src.part[k] = pv[k];
    float acc[MAX_Q][VEC];
    mix(msh, src, nblocks + (boundary ? 1 : 0) + 1, query, 1, eps, scale, acc);
    if (t < MT) store8(mx + at, acc[0]);
    norm(nsh, acc, 0, gain, m + row * H, norm_eps);
}

// ------------------------------------------------------------------------------------------------- router

constexpr int NEXP = 384;
constexpr int TOPK = 8;
constexpr float ROUTE_SCALE = 8.0f;

// tf_ink2_moe_select (a row a warp: the 8 largest sigmoid(logit) + bias, ties to the lower id, weights
// s / (sum s + 1e-20) * 8 with the sum in ascending id), then tf_ink2_ck_slot_order on its picks (descending
// sigmoid(logit) + bias, a stable sort of the ascending ids; a NaN pick, id -1, scores -inf): ids and wts in slot
// order. Every lane holds the picks, so every lane sorts them the same way; lane k writes slot k.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_fuse_route(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
                       float* __restrict__ wts, int R) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= R) return;
    constexpr int PER = NEXP / 32;
    float s[PER], sc[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        const float l = logits[(size_t)row * NEXP + e];
        s[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l)));
        sc[i] = __fadd_rn(s[i], bias[e]);
    }
    uint32_t taken = 0;
    int pick[TOPK];
    float ps[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float best = -INFINITY;
        int bid = 0x7FFFFFFF;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (!(taken >> i & 1u) && sc[i] > best) best = sc[i], bid = lane + 32 * i;  // ascending id: > keeps the lower
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_xor_sync(0xFFFFFFFFu, best, off);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bid, off);
            if (ob > best || (ob == best && oi < bid)) best = ob, bid = oi;
        }
        float v = 0.f;
        if (bid != 0x7FFFFFFF && (bid & 31) == lane) {
#pragma unroll
            for (int i = 0; i < PER; ++i)
                if (i == (bid >> 5)) v = s[i], taken |= 1u << i;
        }
        v = __shfl_sync(0xFFFFFFFFu, v, bid & 31);
        pick[k] = bid == 0x7FFFFFFF ? -1 : bid;
        ps[k] = v;
    }
    // ascending id (insertion sort, the same on every lane; -1 sorts first)
#pragma unroll
    for (int i = 1; i < TOPK; ++i) {
#pragma unroll
        for (int j = i; j > 0; --j) {
            if (pick[j] < pick[j - 1]) {
                const int ti = pick[j];
                pick[j] = pick[j - 1];
                pick[j - 1] = ti;
                const float tv = ps[j];
                ps[j] = ps[j - 1];
                ps[j - 1] = tv;
            }
        }
    }
    float sum = ps[0];
#pragma unroll
    for (int k = 1; k < TOPK; ++k) sum = __fadd_rn(sum, ps[k]);
    const float den = __fadd_rn(sum, 1e-20f);
    // select's outputs in ascending id, then the slot order's sort keys from the logits, as tf_ink2_ck_slot_order
    int id[TOPK];
    float w[TOPK], key[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        id[k] = pick[k];
        w[k] = __fmul_rn(__fdiv_rn(ps[k], den), ROUTE_SCALE);
        if (id[k] < 0) {
            key[k] = -INFINITY;
            continue;
        }
        const float l = __ldg(logits + (size_t)row * NEXP + id[k]);
        key[k] = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l))), __ldg(bias + id[k]));
    }
#pragma unroll
    for (int i = 1; i < TOPK; ++i)
#pragma unroll
        for (int j = i; j > 0; --j)
            if (key[j] > key[j - 1]) {
                const float ts = key[j], tw = w[j];
                const int ti = id[j];
                key[j] = key[j - 1], w[j] = w[j - 1], id[j] = id[j - 1];
                key[j - 1] = ts, w[j - 1] = tw, id[j - 1] = ti;
            }
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        if (lane == k) {
            ids[(size_t)row * TOPK + k] = id[k];
            wts[(size_t)row * TOPK + k] = w[k];
        }
    }
}

// ------------------------------------------------------------------------------------------------- latency forms
//
// The same three ops rewritten for one-row latency (cuda_fuse.zig: the default for decode-sized calls), the same
// expressions in the same order and the same reduction trees as the kernels they replace, so the same bits:
//
//   tf_ink2_lat_attn_in   res.attn_input + its norms      every source loaded once, up front, and kept in registers
//   tf_ink2_lat_mlp_in    res.partial + push + mlp_input  (bf16 packed) for both passes; a barrier a phase (5)
//                         + mlp_norm                      instead of one a source and four a norm; the softmax on
//                                                         warp 0's lanes, a lane a (query, source)
//   tf_ink2_lat_route     select + slot order             the 8 rounds on the lanes' best scores with two warp
//                                                         reductions a round instead of ten dependent shuffles
//
// The row kernels run 576 threads (the mix's 18 warps of 8 columns a thread). The norm's sums are norms.cu's
// tf_rms_rows on 1,024 threads (columns v + 1024 k, its fma chain, its warp xor tree, the 32 warps' xor tree,
// rms_scale): thread t is the norm's threads t and t + 576 (t < 448), so warp w is the norm's warps w and w + 18, the
// same lanes. Every warp then runs the last xor tree itself (the same bits on every lane), so no thread waits for one
// warp's result. The normed rows are written from the mix's registers (each element y = bf16(x * inv * g) on its own).
// Measured at one row (tf-inkling2-fuse, 48 layers' worth, GB10): select + slot order 13.0 to 2.5 us, attention input
// with its norm 5.1-12.7 to 4.6-8.1 us (0 to 7 blocks), MLP input 6.6-14.0 to 5.0-9.3 us; in the W = 4 decode step
// (tf-inkling2-decode --exp check,fuse) 1.23 +- 0.05 ms a step less than the release's ops.

namespace tf_ink2_lat {

constexpr int H = 4608;
constexpr int VEC = 8;
constexpr int NT = H / VEC;             // 576 threads a row
constexpr int NW = NT / 32;             // 18 warps
constexpr int VT = 1024;                // the norm's threads (norms.cu tf_rms_rows)
constexpr int VPER = (H + VT - 1) / VT; // 5 columns a norm thread
constexpr int MAX_SRC = 9;              // 8 blocks and the candidate
constexpr int MAX_Q = 3;

static_assert(NT % 32 == 0 && NT < VT && 2 * NT >= VT, "a thread holds one or two of the norm's threads");

__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }

__device__ __forceinline__ void unpack(const uint4 u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ void load8f(const float* p, float* f) {
    const float4 a = reinterpret_cast<const float4*>(p)[0];
    const float4 b = reinterpret_cast<const float4*>(p)[1];
    f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

__device__ __forceinline__ void store8f(float* p, const float* f) {
    reinterpret_cast<float4*>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}

__device__ __forceinline__ float xor_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
    return v;
}

// inkling2_res.cu warps_sum: the 18 warps' sums in warp order.
__device__ __forceinline__ float warps_sum(const float* sh) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) t = __fadd_rn(t, sh[w]);
    return t;
}

struct Shared {
    float ss[MAX_SRC][NW];
    float dot[MAX_SRC][MAX_Q][NW];
    float p[MAX_Q][MAX_SRC];
    float nw[MAX_Q][32];
};

// A row's sources in registers: bf16 sources 0 .. nbf - 1 packed (8 columns a uint4), then, when F32, one fp32 source
// (the last, index nbf). Every index is a constant after unrolling (no runtime index into pk, which would put it in
// local memory): a source is pk[i] unpacked, replaced by f when i is nbf.
template <bool F32>
struct Sources {
    uint4 pk[MAX_SRC];
    float f[VEC];
    int nbf;
    __device__ __forceinline__ int count() const { return nbf + (F32 ? 1 : 0); }
    __device__ __forceinline__ void get(int i, float* v) const {
        unpack(pk[i], v);
        if (F32) {
            const bool last = i == nbf;
#pragma unroll
            for (int k = 0; k < VEC; ++k) v[k] = last ? f[k] : v[k];
        }
    }
};

// One row of inkling2_res.cu mix_row on registers (sources and queries loaded by the caller): acc[j] = out_j's 8
// columns of this thread. Two barriers and the softmax's one: scores need every warp's sums.
template <int NQ, bool F32>
__device__ __forceinline__ void mix(Shared& sh, const Sources<F32>& src, const float (&q)[NQ][VEC], float eps,
                                    float scale, float (&acc)[NQ][VEC]) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int ns = src.count();
    // each source's sum of squares: this thread's 8 columns in order, then the warp's xor tree
#pragma unroll
    for (int i = 0; i < MAX_SRC; ++i)
        if (i < ns) {
            float v[VEC];
            src.get(i, v);
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < VEC; ++k) ss = __fadd_rn(ss, __fmul_rn(v[k], v[k]));
            ss = xor_sum(ss);
            if (lane == 0) sh.ss[i][warp] = ss;
        }
    __syncthreads();
    // the inverse RMS of source i on lane i of every warp (the 18 warps' sums in order), then on every lane
    float inv_l = 0.f;
    if (lane < ns) {
        const float mean = __fmul_rn(warps_sum(sh.ss[lane]), 1.0f / H);
        inv_l = rsqrtf(__fadd_rn(mean, eps));
    }
    float inv[MAX_SRC];
#pragma unroll
    for (int i = 0; i < MAX_SRC; ++i) inv[i] = __shfl_sync(0xFFFFFFFFu, inv_l, i);
    // the scores' sums: q_j . n_i over this thread's columns, then the warp's xor tree
#pragma unroll
    for (int i = 0; i < MAX_SRC; ++i)
        if (i < ns) {
            float v[VEC];
            src.get(i, v);
            float d[NQ];
#pragma unroll
            for (int j = 0; j < NQ; ++j) d[j] = 0.f;
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                const float n = __fmul_rn(v[k], inv[i]);
#pragma unroll
                for (int j = 0; j < NQ; ++j) d[j] = __fmaf_rn(q[j][k], n, d[j]);
            }
#pragma unroll
            for (int j = 0; j < NQ; ++j) {
                const float s = xor_sum(d[j]);
                if (lane == 0) sh.dot[i][j][warp] = s;
            }
        }
    __syncthreads();
    // the softmax over the sources on warp 0, lane 9 j + i holding query j's score of source i: its max and its sum
    // run through the same scores in the same order as mix_row's one thread a query (every lane of the group runs
    // them on the gathered scores), the exponentials and the divisions one a lane
    if (warp == 0) {
        const int j = lane / MAX_SRC, i = lane - j * MAX_SRC;
        const bool on = j < NQ && i < ns;
        const float sv = on ? __fmul_rn(warps_sum(sh.dot[i][j]), scale) : 0.f;
        const int g = j < NQ ? j * MAX_SRC : 0;
        float m = -__int_as_float(0x7f800000);
#pragma unroll
        for (int k = 0; k < MAX_SRC; ++k) {
            const float o = __shfl_sync(0xFFFFFFFFu, sv, g + k);
            if (k < ns) m = fmaxf(m, o);
        }
        const float e = expf(__fsub_rn(sv, m));
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < MAX_SRC; ++k) {
            const float o = __shfl_sync(0xFFFFFFFFu, e, g + k);
            if (k < ns) sum = __fadd_rn(sum, o);
        }
        if (on) sh.p[j][i] = __fdiv_rn(e, sum);
    }
    __syncthreads();
    // the weighted sum of the normalized sources, in source order
#pragma unroll
    for (int j = 0; j < NQ; ++j)
#pragma unroll
        for (int k = 0; k < VEC; ++k) acc[j][k] = 0.f;
#pragma unroll
    for (int i = 0; i < MAX_SRC; ++i)
        if (i < ns) {
            float v[VEC];
            src.get(i, v);
            float p[NQ];
#pragma unroll
            for (int j = 0; j < NQ; ++j) p[j] = sh.p[j][i];
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                const float n = __fmul_rn(v[k], inv[i]);
#pragma unroll
                for (int j = 0; j < NQ; ++j) acc[j][k] = __fmaf_rn(p[j], n, acc[j][k]);
            }
        }
}

// The norms' scales of nx staged rows xs[j] (fp32 [H] each, in shared memory, written before the caller's barrier):
// inv[j] = tf_rms_rows' rms_scale of row j, the same bits on every thread. One barrier.
template <int NX>
__device__ __forceinline__ void norm_scales(Shared& sh, const float* xs, float eps, float (&inv)[NX]) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
#pragma unroll
    for (int j = 0; j < NX; ++j) {
        const float* x = xs + j * H;
#pragma unroll
        for (int h = 0; h < 2; ++h) {  // the norm's threads t and t + NT
            const int v = t + h * NT;
            if (v < VT) {
                float part = 0.f;
#pragma unroll
                for (int k = 0; k < VPER; ++k) {  // tf_rms_rows' zeros past the row add nothing to a non-negative sum
                    const int i = v + k * VT;
                    if (i < H) part = __fmaf_rn(x[i], x[i], part);
                }
                part = xor_sum(part);
                if (lane == 0) sh.nw[j][v >> 5] = part;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NX; ++j) inv[j] = tf_common::rms_scale(xor_sum(sh.nw[j][lane]), H, eps);
}

// y = bf16(x * inv * g) on this thread's 8 columns (no gain: x * inv), one 16-byte store.
__device__ __forceinline__ void norm_store(const float* x, float inv, bool g, const float* gv, __nv_bfloat16* y) {
    float o[VEC];
    if (g) {
#pragma unroll
        for (int k = 0; k < VEC; ++k) o[k] = __fmul_rn(__fmul_rn(x[k], inv), gv[k]);
    } else {
#pragma unroll
        for (int k = 0; k < VEC; ++k) o[k] = __fmul_rn(x[k], inv);
    }
    uint4 u;
    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&u);
#pragma unroll
    for (int k = 0; k < VEC; ++k) h[k] = __float2bfloat16_rn(o[k]);
    *reinterpret_cast<uint4*>(y) = u;
}

// The attention input of one row: NQ mixes (stored), then nn norms of out_{NQ == 1 ? 0 : j}.
template <int NQ>
__device__ __forceinline__ void attn_in_row(const __nv_bfloat16* blocks, long long block_stride, int nblocks,
                                            const __nv_bfloat16* cand, const float* queries, float* out0, float* out1,
                                            float* out2, float eps, float scale, int nn, const float* g0,
                                            const float* g1, const float* g2, __nv_bfloat16* y0, __nv_bfloat16* y1,
                                            __nv_bfloat16* y2, float norm_eps) {
    __shared__ Shared sh;
    extern __shared__ float4 xs4[];  // the normed rows, fp32 [NQ][H]
    float* xs = reinterpret_cast<float*>(xs4);
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int c0 = t * VEC;
    // every load first: the sources, then the queries
    Sources<false> src;
    src.nbf = nblocks + 1;
    const uint4 uc = ld16(cand + row * H + c0);
#pragma unroll
    for (int i = 0; i < MAX_SRC; ++i) src.pk[i] = i < nblocks ? ld16(blocks + i * block_stride + row * H + c0) : uc;
    float q[NQ][VEC];
#pragma unroll
    for (int j = 0; j < NQ; ++j) load8f(queries + j * H + c0, q[j]);
    const float* gs[MAX_Q] = {g0, g1, g2};
    float gv[MAX_Q][VEC];
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
        if (j < nn && gs[j] != nullptr) load8f(gs[j] + c0, gv[j]);
    float acc[NQ][VEC];
    mix<NQ>(sh, src, q, eps, scale, acc);
    float* outs[MAX_Q] = {out0, out1, out2};
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        store8f(outs[j] + row * H + c0, acc[j]);
        store8f(xs + j * H + c0, acc[j]);
    }
    __syncthreads();
    float inv[NQ];
    norm_scales<NQ>(sh, xs, norm_eps, inv);
    __nv_bfloat16* ys[MAX_Q] = {y0, y1, y2};
#pragma unroll
    for (int j = 0; j < MAX_Q; ++j)
        if (j < nn) {
            const int x = NQ == 1 ? 0 : j;
            norm_store(acc[x], inv[x], gs[j] != nullptr, gv[j], ys[j] + row * H + c0);
        }
}

// res.partial, the boundary push, res.mlp_input and res.mlp_norm of one row (tf_ink2_fuse_mlp_in's math).
__device__ __forceinline__ void mlp_in_row(const __nv_bfloat16* h_in, const __nv_bfloat16* a, int boundary,
                                           float* partial, const __nv_bfloat16* blocks, long long block_stride,
                                           int nblocks, __nv_bfloat16* push, const float* query, float* mx, float eps,
                                           float scale, const float* gain, __nv_bfloat16* m, float norm_eps) {
    __shared__ Shared sh;
    extern __shared__ float4 xs4[];
    float* xs = reinterpret_cast<float*>(xs4);
    const int64_t row = blockIdx.x;
    const int t = threadIdx.x;
    const int c0 = t * VEC;
    const int64_t at = row * H + c0;
    // every load first: a, h_in, the blocks, the query, the gain
    const uint4 ua = ld16(a + at);
    const uint4 uh = ld16(h_in + at);
    Sources<true> src;
    src.nbf = nblocks + (boundary ? 1 : 0);
#pragma unroll
    for (int i = 0; i < MAX_SRC; ++i) src.pk[i] = i < nblocks ? ld16(blocks + i * block_stride + row * H + c0) : uh;
    float q[1][VEC];
    load8f(query + c0, q[0]);
    float gv[VEC];
    if (gain != nullptr) load8f(gain + c0, gv);
    // tf_ink2_res_partial, and the push (state.pushBlock's row copy)
    unpack(ua, src.f);
    if (!boundary) {
        float x[VEC];
        unpack(uh, x);
#pragma unroll
        for (int k = 0; k < VEC; ++k) src.f[k] = __fadd_rn(x[k], src.f[k]);
    } else {
        *reinterpret_cast<uint4*>(push + at) = uh;
    }
    store8f(partial + at, src.f);
    float acc[1][VEC];
    mix<1>(sh, src, q, eps, scale, acc);
    store8f(mx + at, acc[0]);
    store8f(xs + c0, acc[0]);
    __syncthreads();
    float inv[1];
    norm_scales<1>(sh, xs, norm_eps, inv);
    norm_store(acc[0], inv[0], gain != nullptr, gv, m + at);
}

}  // namespace tf_ink2_lat

// tf_ink2_fuse_attn_in's latency form: the same arguments; grid rows, block 576, dynamic shared nq * H * 4 bytes.
extern "C" __global__ void __launch_bounds__(tf_ink2_lat::NT)
    tf_ink2_lat_attn_in(const __nv_bfloat16* blocks, long long block_stride, int nblocks, const __nv_bfloat16* cand,
                        const float* queries, int nq, float* out0, float* out1, float* out2, float eps, float scale,
                        int nn, const float* g0, const float* g1, const float* g2, __nv_bfloat16* y0,
                        __nv_bfloat16* y1, __nv_bfloat16* y2, float norm_eps) {
    if (nq == 1)
        tf_ink2_lat::attn_in_row<1>(blocks, block_stride, nblocks, cand, queries, out0, out1, out2, eps, scale, nn, g0,
                                    g1, g2, y0, y1, y2, norm_eps);
    else
        tf_ink2_lat::attn_in_row<3>(blocks, block_stride, nblocks, cand, queries, out0, out1, out2, eps, scale, nn, g0,
                                    g1, g2, y0, y1, y2, norm_eps);
}

// tf_ink2_fuse_mlp_in's latency form: the same arguments; grid rows, block 576, dynamic shared H * 4 bytes.
extern "C" __global__ void __launch_bounds__(tf_ink2_lat::NT)
    tf_ink2_lat_mlp_in(const __nv_bfloat16* h_in, const __nv_bfloat16* a, int boundary, float* partial,
                       const __nv_bfloat16* blocks, long long block_stride, int nblocks, __nv_bfloat16* push,
                       const float* query, float* mx, float eps, float scale, const float* gain, __nv_bfloat16* m,
                       float norm_eps) {
    tf_ink2_lat::mlp_in_row(h_in, a, boundary, partial, blocks, block_stride, nblocks, push, query, mx, eps, scale, gain,
                            m, norm_eps);
}

// tf_ink2_fuse_route's latency form (select + slot order; the same arguments, a warp a row, 256 threads a block).
// The picks: the 8 rounds of select, each the largest valid score (not NaN, above -inf), ties to the lower id. A score
// is a 32-bit key that orders as the float does (-0 read as +0, as > reads them; invalid scores 0, below every valid
// key); each lane takes its own best untaken (key, id) of its 12 (a tree), the warp the largest key (one reduction)
// and the lowest id holding it (a second); the winner's lane marks it taken. That is select's pick in every round. The slot order
// sorts select's picks by the same score descending, ties to the lower id, the empty picks (-1) last: the order the
// rounds draw them in. The weights are select's: s / (sum s in ascending id + 1e-20) * 8.
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_lat_route(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
                      float* __restrict__ wts, int R) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= R) return;
    constexpr int PER = NEXP / 32;
    float l[PER], b[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        l[i] = logits[(size_t)row * NEXP + lane + 32 * i];
        b[i] = bias[lane + 32 * i];
    }
    float s[PER];
    uint32_t key[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        s[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l[i])));
        const float sc = __fadd_rn(s[i], b[i]);
        const bool valid = sc > -INFINITY;  // false for NaN
        const uint32_t u = sc == 0.f ? 0u : __float_as_uint(sc);
        key[i] = !valid ? 0u : (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    int pick[TOPK];
    uint32_t taken = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        // this lane's best untaken: the largest key, ties to the lower i (the lower id); a tree over (key, 15 - i)
        unsigned long long c[16];
#pragma unroll
        for (int i = 0; i < 16; ++i)
            c[i] = (i < PER && !(taken >> i & 1u)) ? ((unsigned long long)key[i] << 4 | (unsigned long long)(15 - i)) : 0ull;
#pragma unroll
        for (int w = 8; w > 0; w >>= 1)
#pragma unroll
            for (int i = 0; i < w; ++i) c[i] = c[i] > c[i + w] ? c[i] : c[i + w];
        const uint32_t bk = (uint32_t)(c[0] >> 4);
        const int bi = 15 - (int)(c[0] & 15ull);
        const uint32_t top = __reduce_max_sync(0xFFFFFFFFu, bk);
        const uint32_t id = (uint32_t)__reduce_min_sync(0xFFFFFFFFu, bk == top ? (uint32_t)(lane + 32 * bi) : 0xFFFFFFFFu);
        const bool none = top == 0u;
        pick[k] = none ? -1 : (int)id;
        if (!none && (int)id == lane + 32 * bi) taken |= 1u << bi;
    }
    // the picks' s: lane id & 31's s[id >> 5] (0 for none, as select's)
    float ps[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < PER; ++i) v = (pick[k] >= 0 && (pick[k] >> 5) == i) ? s[i] : v;
        ps[k] = __shfl_sync(0xFFFFFFFFu, v, pick[k] < 0 ? 31 : pick[k] & 31);
    }
    // select's sum in ascending id (its insertion sort; -1 sorts first)
    int sid[TOPK];
    float sps[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) sid[k] = pick[k], sps[k] = ps[k];
#pragma unroll
    for (int i = 1; i < TOPK; ++i) {
#pragma unroll
        for (int j = i; j > 0; --j) {
            if (sid[j] < sid[j - 1]) {
                const int ti = sid[j];
                sid[j] = sid[j - 1];
                sid[j - 1] = ti;
                const float tv = sps[j];
                sps[j] = sps[j - 1];
                sps[j - 1] = tv;
            }
        }
    }
    float sum = sps[0];
#pragma unroll
    for (int k = 1; k < TOPK; ++k) sum = __fadd_rn(sum, sps[k]);
    const float den = __fadd_rn(sum, 1e-20f);
    // lane k < 8 writes slot k: its pick and weight
    int my_id = -1;
    float my_ps = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
        if (lane == k) my_id = pick[k], my_ps = ps[k];
    if (lane < TOPK) {
        ids[(size_t)row * TOPK + lane] = my_id;
        wts[(size_t)row * TOPK + lane] = __fmul_rn(__fdiv_rn(my_ps, den), ROUTE_SCALE);
    }
}
