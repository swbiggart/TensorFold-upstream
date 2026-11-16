// Inkling-2's head group (G-head): the unembed as row-invariant bf16 rows (dense_rows.cu's sums, a bf16 store) and
// the final logit softcap over the unpadded vocabulary. The embedding and the norms run on upstream kernels. The
// entry points, shapes and math are in zig/src/families/inkling2/cuda_head.zig.
//
// Math: Transformers' modeling_inkling2.py (Apache-2.0, Copyright 2026 the HuggingFace Team),
// Inkling2ForCausalLM.forward: logits = lm_head(h) (bf16), .float(), Inkling2Softcap (tanh(x / 30) * 30, where
// torch divides by a Python scalar as a product with fp32(1 / 30)), then the first unpadded_vocab_size columns.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_ink2_head {

constexpr int WARPS = 8;
constexpr int VEC = 8; // bf16 a 16-byte load

__device__ __forceinline__ void unpack(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

// out[r, n] = bf16(x[r] . w[n]) for a tile of RT rows: a warp an output column; lane l sums k = 8 l + 256 j in
// order, then the fixed xor tree (dense_rows.cu's order), so a row's bits do not depend on RT or the row count.
template <int RT>
__device__ __forceinline__ void tile(const __nv_bfloat16* __restrict__ x, int64_t x_stride,
                                     const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                     int64_t out_stride, int rows, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int r0 = blockIdx.y * RT;
    if (col >= n) return;
    float acc[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) acc[r] = 0.f;
    const __nv_bfloat16* wr = w + static_cast<int64_t>(col) * k;
#pragma unroll 4
    for (int k0 = lane * VEC; k0 < k; k0 += 32 * VEC) {
        float wf[VEC];
        unpack(*reinterpret_cast<const uint4*>(wr + k0), wf);
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            if (r0 + r >= rows) break;
            float xf[VEC];
            unpack(*reinterpret_cast<const uint4*>(x + (r0 + r) * x_stride + k0), xf);
#pragma unroll
            for (int i = 0; i < VEC; ++i) acc[r] = __fmaf_rn(xf[i], wf[i], acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        float v = acc[r];
        for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
        if (lane == 0 && r0 + r < rows) out[(r0 + r) * out_stride + col] = __float2bfloat16_rn(v);
    }
}

}  // namespace tf_ink2_head

using namespace tf_ink2_head;

// The module's presence check (cuda_kernels.zig Group): writes the group's number.
extern "C" __global__ void tf_ink2_head_probe(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 5;
}

// The unembed: out [rows, n] bf16 = x [rows, k] bf16 . w [n, k]^T. Grid (ceil(n / WARPS), ceil(rows / RT)),
// WARPS * 32 threads; k % 8 == 0, x and w 16-byte aligned, strides in elements.
#define TF_INK2_UNEMBED(RT)                                                                                            \
    extern "C" __global__ void __launch_bounds__(WARPS * 32) tf_ink2_head_unembed_##RT(                               \
        const __nv_bfloat16* x, long long x_stride, const __nv_bfloat16* w, __nv_bfloat16* out, long long out_stride, \
        int rows, int n, int k) {                                                                                      \
        tile<RT>(x, x_stride, w, out, out_stride, rows, n, k);                                                         \
    }

TF_INK2_UNEMBED(1)
TF_INK2_UNEMBED(4)
TF_INK2_UNEMBED(16)
TF_INK2_UNEMBED(32)

// out[r, c] fp32 = tanh(f32(l[r, c]) * fp32(1 / cap)) * cap for c < cols. Grid (ceil(cols / 256), rows), 256 threads.
extern "C" __global__ void tf_ink2_head_softcap(const __nv_bfloat16* __restrict__ l, long long l_stride,
                                                float* __restrict__ out, long long out_stride, int cols, float cap) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const int64_t r = blockIdx.y;
    const float inv = __fdiv_rn(1.0f, cap);
    const float x = __bfloat162float(l[r * l_stride + c]);
    out[r * out_stride + c] = __fmul_rn(tanhf(__fmul_rn(x, inv)), cap);
}

// Greedy rows on the device (B2 C4): out[r] = lanes/gpu_rule.zig argmax of row r of x [rows, cols] fp32, the first
// index of the largest value as the host's loop finds it (`best = 0; if (v > row[best]) best = i`): a NaN at column 0
// wins (nothing compares greater), a NaN elsewhere never does, -0 equals +0 (the first wins), all -inf gives 0. Each
// thread keeps the first index of its columns' largest non-NaN value; the block reduces (value, index) pairs by
// "greater value, else lower index", which is that loop's answer whatever the order. Grid rows, 1024 threads.
extern "C" __global__ void __launch_bounds__(1024) tf_ink2_head_argmax(const float* __restrict__ x, long long stride,
                                                                       int cols, unsigned* __restrict__ out) {
    const float* row = x + (int64_t)blockIdx.x * stride;
    float bv = -__int_as_float(0x7f800000); // -inf
    int bi = 0x7fffffff;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = row[c];
        if (v > bv || (v == bv && c < bi)) { // a NaN takes neither branch
            bv = v;
            bi = c;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : -__int_as_float(0x7f800000);
    bi = lane < nw ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) out[blockIdx.x] = (cols == 0 || row[0] != row[0] || bi == 0x7fffffff) ? 0u : (unsigned)bi;
}

// The embedding held by vocabulary rows over ranks (cuda_embed.zig; v0.1 kept it whole on rank 0). Both kernels copy
// bits; the norm after them is the same kernel on the same rows. Grid (rows, ceil(d / 8 / 256)), 256 threads, 16 bytes
// a thread (d % 8 == 0, every pointer 16-byte aligned).
// share: out[r] = w[ids[r] - lo] when lo <= ids[r] < hi (the rank's rows of the table), else zeros.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_head_embed_share(const int* __restrict__ ids,
                                                                           const __nv_bfloat16* __restrict__ w,
                                                                           __nv_bfloat16* __restrict__ out, int d,
                                                                           int lo, int hi) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (8 * c >= d) return;
    const int id = ids[r];
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (id >= lo && id < hi) v = *reinterpret_cast<const uint4*>(w + static_cast<int64_t>(id - lo) * d + 8 * c);
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = v;
}

// pick: out[r] = parts[owner(ids[r])][r] from the gathered slots parts [world][rows][d]: the owners are ranks first ..
// first + owners - 1, each holding canon's share of the vocabulary (plan.zig canon: the first vocab % owners one row
// more), so owner(id) = first + the share holding id.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_head_embed_pick(const int* __restrict__ ids,
                                                                          const __nv_bfloat16* __restrict__ parts,
                                                                          __nv_bfloat16* __restrict__ out, int d,
                                                                          int rows, int first, int owners, int vocab) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    if (8 * c >= d) return;
    const int id = ids[r];
    const int base = vocab / owners, extra = vocab % owners;
    const int edge = extra * (base + 1);
    int q = id < edge ? id / (base + 1) : extra + (id - edge) / base;
    q = q < 0 ? 0 : (q >= owners ? owners - 1 : q);
    const int64_t slot = static_cast<int64_t>(first + q) * rows + r;
    *reinterpret_cast<uint4*>(out + r * d + 8 * c) = *reinterpret_cast<const uint4*>(parts + slot * d + 8 * c);
}
