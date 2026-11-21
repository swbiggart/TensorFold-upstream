// Inkling-2's experts in `checkpoint` mode (cuda_moe.zig, Mode.checkpoint): activations quantized the way the
// checkpoint was trained and is served, on the block-scaled tensor cores (sm_120a / sm_121a).
//   NVFP4 layers (0-39): rows to NVFP4 per token (outer scale amax / 1536, an e4m3 scale a 16 chosen between the
//     6- and 4-target candidates by reconstruction error), FP4 x FP4 mma (kind::mxf4nvf4, scale_vec::4X, ue4m3);
//     each sum times (row amax * expert amax) / (1536 * outer_scale_denominator).
//   MXFP8 layers (40-47): rows to MXFP8 (a ue8m0 scale a 32, rounded up), FP8 x FP8 mma (kind::mxf8f6f4,
//     scale_vec::1X, ue8m0; checked on sm_120 against a double reference).
// Then gate/up rounded to bf16, the capped SwiGLU in fp32 with the routing weight applied before its bf16 store, that
// row re-quantized (the same format) for the down projection, each route's down row rounded to bf16, and the fp32 sum
// of a token's route rows in slot order (descending sigmoid + bias, the order TokenSpeed folds), one bf16 rounding.
//
// The numerics are TokenSpeed's Torch reference (github.com/lightseekorg/tokenspeed, tokenspeed-kernel ops/moe/
// block_scaled.py: reference_block_quantize, reference_capped_swiglu, torch_block_scaled_moe_routes; MIT licence,
// Copyright (c) 2026 LightSeek Foundation), each torch op one rounding here (built with --fmad=false, IEEE division).
// The GEMMs differ from that reference only in fp32 summation order: each mma's block (64 k for FP4, 32 k for FP8)
// is summed by the tensor core into a fresh fragment, which one IEEE add puts into its quarter's fp32 sum (K in four
// quarters, a warp each, the blocks in order), and the quarters are added in order. Every output element is one
// fixed sequence whatever the call's row count, tile shape or block, so a row's bits do not depend on its window or
// chunk.
//
// Split experts (W ranks, each a 1/W row slice of every expert): the GEMMs take N, the weight rows an expert holds
// (w13: N/2 gate/up pairs; w2: N output rows, all of K), and write rows ostride apart; the up pass leaves the rank's
// h slice [routes, N/2], the ranks all-gather those in rank order ([W][routes][I/W]), the quantizer reads a row of h
// from its W segments (the per-token amax and scales over the whole row, as at W = 1), the down pass and the fold
// run on the rank's output columns, and tf_ink2_ck_columns lays the gathered column slices out as rows. Every output
// element is computed whole on one rank by the same sequence as at W = 1, so the result is bit-equal to W = 1.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr int TOPK = 8;
constexpr float GATE_CAP = 4.0f;
constexpr float UP_CAP = 25.0f;
constexpr float FMAX = 3.4028234663852886e38f;

// torch's `x * (1.0 / c)`: a Python float scalar, applied in fp32
constexpr float INV_1536 = (float)(1.0 / 1536.0);
constexpr float INV_6 = (float)(1.0 / 6.0);
constexpr float INV_448 = (float)(1.0 / 448.0);

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float lo_f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_f(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

// torch's clamp(max=c): NaN stays NaN
__device__ __forceinline__ float clamp_max(float v, float c) { return v > c ? c : v; }

// The larger, or a NaN when either is (torch's amax propagates it)
__device__ __forceinline__ float max_nan(float a, float b) { return (a > b || a != a) ? a : b; }

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

__device__ __forceinline__ uint32_t f2e4m3(float v) {
  return static_cast<uint32_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// TokenSpeed's _fp4_round: |v| to the e2m1 magnitude code (ties to the even code), the sign bit from signbit(v)
// (so -0 and small negatives give code 8), and the rounded value with that sign.
__device__ __forceinline__ uint32_t fp4_round(float v, float& rounded) {
  const float a = fabsf(v);
  const uint32_t c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
  const float tab = c < 2u ? (c ? 0.5f : 0.0f) : __uint_as_float((((c >> 1) + 126u) << 23) | ((c & 1u) << 22));
  const bool neg = signbit(v);
  rounded = neg ? -tab : tab;
  return c | (neg ? 8u : 0u);
}

// Inkling2SoftcappedSwiGLU as TokenSpeed's reference_capped_swiglu orders it (sigmoid of the raw gate; `x / c` on a
// Python scalar is x * f32(1 / c) on the GPU), the same ops as inkling2_moe.cu's full mode.
__device__ __forceinline__ float capped_swiglu(float g, float u) {
  const float gc = __fmul_rn(tanhf(__fmul_rn(g, 1.0f / GATE_CAP)), GATE_CAP);
  const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
  const float uc = __fmul_rn(tanhf(__fmul_rn(u, 1.0f / UP_CAP)), UP_CAP);
  return __fmul_rn(__fmul_rn(gc, sg), uc);
}

__device__ __forceinline__ void unpack16(const __nv_bfloat16* p, float (&f)[16]) {
  const uint4 a = *reinterpret_cast<const uint4*>(p);
  const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) f[2 * i] = lo_f(w[i]), f[2 * i + 1] = hi_f(w[i]);
}

// A block's maximum of |x| (any order: max is exact), every thread gets it.
__device__ __forceinline__ float block_max(float v, float* red) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = max_nan(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float m = red[0];
  for (int i = 1; i < (int)(blockDim.x >> 5); ++i) m = max_nan(m, red[i]);
  __syncthreads();
  return m;
}

// Element col of row `row` of x: a row in segments of `seg` values, segment s at x + s * seg_stride (seg_stride 0 and
// seg = K: the plain row-major [rows, K]; the split layout's gathered h: [W][rows][K / W]). Blocks never straddle.
__device__ __forceinline__ const __nv_bfloat16* seg_at(const __nv_bfloat16* x, int row, int col, int seg,
                                                       long long seg_stride) {
  const int s = col / seg;
  return x + (size_t)s * seg_stride + (size_t)row * seg + (col - s * seg);
}

// Whether route i (a token row * 8 + slot) runs here: rows = -1 quantizes every row (the tokens' rows); otherwise
// row i is a route's activation row, quantized only when its expert is in [lo, hi).
__device__ __forceinline__ bool row_here(const int* ids, int row, int lo, int hi) {
  if (ids == nullptr) return true;
  const int e = __ldg(ids + row);
  return e >= lo && e < hi;
}

}  // namespace

// ------------------------------------------------------------------------------------------------- quantizers

// NVFP4 rows (reference_block_quantize, "nvfp4"): a block a row (up to 512 threads: cuda_moe.zig gives a thread a 16-value
// block, so a row is one pass of each loop; the amax is a max, any order). x bf16 [rows, K] (in segments: seg_at);
// codes u8 [rows, K/2] (low nibble the earlier column), scales e4m3 [rows, K/16], amax fp32 [rows]. With ids, row i is
// quantized only when ids[i] is in [lo, hi) (a route's activation row whose expert this rank holds).
namespace {
// One row's NVFP4 quantization by the whole block (red: a float a warp of the block, in shared memory).
__device__ __forceinline__ void quant_nvfp4_row(const __nv_bfloat16* __restrict__ x, int K, int seg,
                                                long long seg_stride, int row, uint8_t* __restrict__ codes,
                                                uint8_t* __restrict__ scales, float* __restrict__ amax_out,
                                                float* red) {
  const int nb = K / 16;
  float m = 0.0f;
  for (int b = threadIdx.x; b < nb; b += blockDim.x) {
    float f[16];
    unpack16(seg_at(x, row, 16 * b, seg, seg_stride), f);
#pragma unroll
    for (int i = 0; i < 16; ++i) m = max_nan(m, fabsf(f[i]));
  }
  const float amax = block_max(m, red);
  if (threadIdx.x == 0) amax_out[row] = amax;
  const float encoder = clamp_max(__fdiv_rn(1536.0f, amax), FMAX);
  const float decoder = __fdiv_rn(1.0f, encoder);
  const float outer = __fmul_rn(amax, INV_1536);
  const float enc6 = __fmul_rn(encoder, INV_6), enc4 = __fmul_rn(encoder, 0.25f);
  for (int b = threadIdx.x; b < nb; b += blockDim.x) {
    float f[16];
    unpack16(seg_at(x, row, 16 * b, seg, seg_stride), f);
    float maxima = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) maxima = max_nan(maxima, fabsf(f[i]));
    uint32_t code[2][16];
    uint32_t s8[2];
    float err[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float sv = clamp_max(__fmul_rn(maxima, c == 0 ? enc6 : enc4), 448.0f);
      s8[c] = f2e4m3(sv);
      const float sf = e4m3f(s8[c]);
      const float inv = clamp_max(__fdiv_rn(1.0f, __fmul_rn(sf, decoder)), FMAX);
      float e[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float r;
        code[c][i] = fp4_round(__fmul_rn(f[i], inv), r);
        e[i] = fabsf(__fsub_rn(f[i], __fmul_rn(__fmul_rn(r, sf), outer)));
      }
      // pairwise, adjacent pairs first (the reference's reshape(..., 2) sums)
#pragma unroll
      for (int w = 8; w >= 1; w >>= 1)
#pragma unroll
        for (int i = 0; i < w; ++i) e[i] = __fadd_rn(e[2 * i], e[2 * i + 1]);
      err[c] = e[0];
    }
    const int pick = err[1] < err[0] ? 1 : 0;
    uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) w0 |= code[pick][i] << (4 * i), w1 |= code[pick][8 + i] << (4 * i);
    *reinterpret_cast<uint2*>(codes + (size_t)row * (K / 2) + 8 * b) = make_uint2(w0, w1);
    scales[(size_t)row * nb + b] = static_cast<uint8_t>(s8[pick]);
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_ck_quant_nvfp4(const __nv_bfloat16* __restrict__ x, int K, int seg, long long seg_stride,
                           const int* __restrict__ ids, int lo, int hi, uint8_t* __restrict__ codes,
                           uint8_t* __restrict__ scales, float* __restrict__ amax_out) {
  __shared__ float red[16];
  const int row = blockIdx.x;
  if (!row_here(ids, row, lo, hi)) return;
  quant_nvfp4_row(x, K, seg, seg_stride, row, codes, scales, amax_out, red);
}

// MXFP8 rows (reference_block_quantize, "mxfp8"): a block a row (up to 512 threads, a thread a 32-value block). x bf16
// [rows, K] (in segments: seg_at); values e4m3 [rows, K], exponents ue8m0 [rows, K/32] (2^ceil(log2(amax / 448)), 0
// below 2^-127, 255 for NaN). ids as for NVFP4.
namespace {
// One row's MXFP8 quantization by the whole block.
__device__ __forceinline__ void quant_mxfp8_row(const __nv_bfloat16* __restrict__ x, int K, int seg,
                                                long long seg_stride, int row, uint8_t* __restrict__ vals,
                                                uint8_t* __restrict__ exps) {
  for (int b = threadIdx.x; b < K / 32; b += blockDim.x) {
    float f[32];
    const __nv_bfloat16* xb = seg_at(x, row, 32 * b, seg, seg_stride);
    unpack16(xb, *reinterpret_cast<float(*)[16]>(f));
    unpack16(xb + 16, *reinterpret_cast<float(*)[16]>(f + 16));
    float maxima = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) maxima = max_nan(maxima, fabsf(f[i]));
    const float raw = __fmul_rn(maxima, INV_448);
    int e = (int)(((uint32_t)__float_as_int(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
    e = raw <= __int_as_float(0x00400000) ? 0 : min(e, 254);  // 2^-127
    if (raw != raw) e = 255;
    // 1 / 2^(e - 127) (exact: a power of two, 2^-127 a denormal); 2^127 for e = 0; 0 for NaN
    const float inv = e == 0 ? __int_as_float(0x7F000000) : e == 255 ? 0.0f : __fdiv_rn(1.0f, __int_as_float(e << 23));
    uint32_t w[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      uint32_t v = 0u;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float y = __fmul_rn(f[4 * q + i], inv);
        y = y < -448.0f ? -448.0f : (y > 448.0f ? 448.0f : y);
        v |= f2e4m3(y) << (8 * i);
      }
      w[q] = v;
    }
    uint4* dst = reinterpret_cast<uint4*>(vals + (size_t)row * K + 32 * b);
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    exps[(size_t)row * (K / 32) + b] = static_cast<uint8_t>(e);
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_ck_quant_mxfp8(const __nv_bfloat16* __restrict__ x, int K, int seg, long long seg_stride,
                           const int* __restrict__ ids, int lo, int hi, uint8_t* __restrict__ vals,
                           uint8_t* __restrict__ exps) {
  const int row = blockIdx.x;
  if (!row_here(ids, row, lo, hi)) return;
  quant_mxfp8_row(x, K, seg, seg_stride, row, vals, exps);
}

// ------------------------------------------------------------------------------------------------- the mmas

namespace {

// mma_fp4 is upstream TensorFold's zig/kernels/cuda/mma4.cuh mma_fp4; mma_mx8 is its MXFP8 counterpart, the
// instruction form and scale layout from NVIDIA's PTX ISA ("Block Scaling for mma.sync").

// NVFP4 x NVFP4, 64 k, an e4m3 scale a 16 on both sides (A: lanes t = 0, 1 give rows g, g + 8's four; B: column g's)
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
        "h"(z));
#else
  __trap();
#endif
}

// MXFP8 x MXFP8, 32 k, a ue8m0 scale a 32 on both sides (the scale in byte 0; A: lanes t = 0, 1 give rows g, g + 8)
__device__ __forceinline__ void mma_mx8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
        "h"(z));
#else
  __trap();
#endif
}

enum { NVFP4 = 0, MXFP8 = 1 };
enum { UP = 0, DOWN = 1 };
constexpr int KSTEP = 64;
constexpr int CK_WARPS = 4;

// One 64-k step for this lane: n8 tile j's weights (column n = 8j + g of the warp's) and the RT row tiles' inputs.
//   NVFP4: weight bytes 4t and 16 + 4t of the step's 32 (k 8t.., 32 + 8t..) and the column's four e4m3 scales; input
//          rows g, g + 8 the same bytes (a[rt][0..3] = row g low, row g + 8 low, row g high, row g + 8 high) and the
//          scales of row g (t even) or g + 8 (t odd).
//   MXFP8: bytes 8t .. 8t + 7 of each of the step's two 32-blocks (a permutation of k inside a block, the same for
//          both operands) and the two blocks' ue8m0 scales (low byte the first); inputs likewise.
template <int FMT, int RT, int NT>
struct Step {
  uint32_t w[NT][FMT == NVFP4 ? 2 : 4];
  uint32_t s[NT];
  uint32_t a[RT][FMT == NVFP4 ? 4 : 8];
  uint32_t sa[RT];
};

template <int FMT, int RT, int NT>
__device__ __forceinline__ void load_step(Step<FMT, RT, NT>& st, const uint8_t* const (&wrow)[NT],
                                          const uint8_t* const (&srow)[NT], const uint8_t* const (&ar)[RT][2],
                                          const uint8_t* const (&sr)[RT], int ks, int t) {
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    if constexpr (FMT == NVFP4) {
      const uint8_t* p = wrow[j] + ks * 32;
      st.w[j][0] = __ldg(reinterpret_cast<const uint32_t*>(p + 4 * t));
      st.w[j][1] = __ldg(reinterpret_cast<const uint32_t*>(p + 16 + 4 * t));
      st.s[j] = __ldg(reinterpret_cast<const uint32_t*>(srow[j] + ks * 4));
    } else {
      const uint8_t* p = wrow[j] + ks * 64;
      const uint2 q0 = __ldg(reinterpret_cast<const uint2*>(p + 8 * t));
      const uint2 q1 = __ldg(reinterpret_cast<const uint2*>(p + 32 + 8 * t));
      st.w[j][0] = q0.x, st.w[j][1] = q0.y, st.w[j][2] = q1.x, st.w[j][3] = q1.y;
      st.s[j] = __ldg(reinterpret_cast<const uint16_t*>(srow[j] + ks * 2));
    }
  }
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    const uint8_t* p0 = ar[rt][0];
    const uint8_t* p1 = ar[rt][1];
    if constexpr (FMT == NVFP4) {
      st.a[rt][0] = p0 ? __ldg(reinterpret_cast<const uint32_t*>(p0 + ks * 32 + 4 * t)) : 0u;
      st.a[rt][1] = p1 ? __ldg(reinterpret_cast<const uint32_t*>(p1 + ks * 32 + 4 * t)) : 0u;
      st.a[rt][2] = p0 ? __ldg(reinterpret_cast<const uint32_t*>(p0 + ks * 32 + 16 + 4 * t)) : 0u;
      st.a[rt][3] = p1 ? __ldg(reinterpret_cast<const uint32_t*>(p1 + ks * 32 + 16 + 4 * t)) : 0u;
      st.sa[rt] = sr[rt] ? __ldg(reinterpret_cast<const uint32_t*>(sr[rt] + ks * 4)) : 0u;
    } else {
      const uint2 z = make_uint2(0u, 0u);
      const uint2 x00 = p0 ? __ldg(reinterpret_cast<const uint2*>(p0 + ks * 64 + 8 * t)) : z;
      const uint2 x10 = p1 ? __ldg(reinterpret_cast<const uint2*>(p1 + ks * 64 + 8 * t)) : z;
      const uint2 x01 = p0 ? __ldg(reinterpret_cast<const uint2*>(p0 + ks * 64 + 32 + 8 * t)) : z;
      const uint2 x11 = p1 ? __ldg(reinterpret_cast<const uint2*>(p1 + ks * 64 + 32 + 8 * t)) : z;
      st.a[rt][0] = x00.x, st.a[rt][1] = x10.x, st.a[rt][2] = x00.y, st.a[rt][3] = x10.y;
      st.a[rt][4] = x01.x, st.a[rt][5] = x11.x, st.a[rt][6] = x01.y, st.a[rt][7] = x11.y;
      st.sa[rt] = sr[rt] ? __ldg(reinterpret_cast<const uint16_t*>(sr[rt] + ks * 2)) : 0u;
    }
  }
}

// The step's mmas: each into a fresh fragment, added to the fp32 sums with one IEEE add (NVFP4 one 64-k mma, MXFP8
// two 32-k ones in k order).
template <int FMT, int RT, int NT>
__device__ __forceinline__ void mma_step(float (&acc)[RT][NT][4], const Step<FMT, RT, NT>& st, int cnt) {
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) {
    if (rt * 16 >= cnt) break;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      if constexpr (FMT == NVFP4) {
        const uint32_t a[4] = {st.a[rt][0], st.a[rt][1], st.a[rt][2], st.a[rt][3]};
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma_fp4(p, a, st.w[j][0], st.w[j][1], st.sa[rt], st.s[j]);
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[q]);
      } else {
        const uint32_t a0[4] = {st.a[rt][0], st.a[rt][1], st.a[rt][2], st.a[rt][3]};
        const uint32_t a1[4] = {st.a[rt][4], st.a[rt][5], st.a[rt][6], st.a[rt][7]};
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma_mx8(p, a0, st.w[j][0], st.w[j][1], st.sa[rt] & 0xFFu, st.s[j] & 0xFFu);
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[q]);
        float p2[4] = {0.f, 0.f, 0.f, 0.f};
        mma_mx8(p2, a1, st.w[j][2], st.w[j][3], st.sa[rt] >> 8, st.s[j] >> 8);
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p2[q]);
      }
    }
  }
}

// Grouped experts on quantized rows: for each item (expert e, members first .. first + cnt <= 16 * RT) and each block
// of 8 * NT output columns, sums of A row (member / x_div) times W_e's rows, then the epilogue.
//   NVFP4: A codes u8 [*, K/2], scales e4m3 [*, K/16], amax fp32 [*]; W u8 [E, N, K/2], S e4m3 [E, N, K/16],
//          wamax fp32 [E]; a sum times (amax[a row] * wamax[e]) * rcp, rcp = 1 / (1536 * outer_scale_denominator).
//   MXFP8: A e4m3 [*, K], exponents [*, K/32]; W e4m3 [E, N, K], S ue8m0 [E, N, K/32].
// UP: N weight rows of interleaved (gate, up) pairs (4608 whole, 4608 / W split): act[member * ostride + i] =
//     bf16(capped_swiglu(bf16 gate_i, bf16 up_i) * gamma), i < N / 2.
// DOWN: N output rows (4608, or the rank's 4608 / W): out[member * ostride + n] = bf16(sum).
// A block takes a unit (item, column block), grid-stride; its 4 warps take K's four quarters (K / 64 steps each, a
// step at a time in order, the next one's loads in flight), and warp 0 adds the quarters' sums in order,
// ((q0 + q1) + q2) + q3, then runs the epilogue: one sum order for every shape, so a row's bits do not depend on it.
template <int FMT, int EPI, int RT, int NT>
__device__ __forceinline__ void ck_body(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As,
                                        const float* __restrict__ Aamax, int x_div, int K,
                                        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                        const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,
                                        const int* __restrict__ items, const int* __restrict__ nitems,
                                        const int* __restrict__ members, int N, int ostride,
                                        __nv_bfloat16* __restrict__ out) {
  const int NB = N / (8 * NT);
  constexpr int NV = RT * NT * 4;
  __shared__ float part[CK_WARPS - 1][NV][32];
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, warp = threadIdx.x >> 5;
  const size_t arow_bytes = FMT == NVFP4 ? (size_t)K / 2 : (size_t)K;
  const size_t srow_bytes = FMT == NVFP4 ? (size_t)K / 16 : (size_t)K / 32;
  const int KS = K / KSTEP;
  const int ks0 = warp * KS / CK_WARPS, ks1 = (warp + 1) * KS / CK_WARPS;
  const int units = __ldg(nitems) * NB;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const uint8_t* wrow[NT];
    const uint8_t* srow[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const size_t n = (size_t)e * N + (size_t)cb * (8 * NT) + 8 * j + g;
      wrow[j] = W + n * arow_bytes;
      srow[j] = S + n * srow_bytes;
    }
    // rows g and g + 8 of each 16-row tile; the scale lanes: t even gives row g's, t odd row g + 8's
    const uint8_t* ar[RT][2];
    const uint8_t* sr[RT];
    int mem[RT][2];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = rt * 16 + g + 8 * h;
        mem[rt][h] = r < cnt ? __ldg(members + first + r) : -1;
        ar[rt][h] = r < cnt ? A + (size_t)(mem[rt][h] / x_div) * arow_bytes : nullptr;
      }
      const int hs = t & 1;
      sr[rt] = mem[rt][hs] >= 0 ? As + (size_t)(mem[rt][hs] / x_div) * srow_bytes : nullptr;
    }
    float acc[RT][NT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[rt][j][0] = acc[rt][j][1] = acc[rt][j][2] = acc[rt][j][3] = 0.f;
    Step<FMT, RT, NT> cur, nxt;
    load_step<FMT, RT, NT>(cur, wrow, srow, ar, sr, ks0, t);
    for (int ks = ks0; ks < ks1; ++ks) {
      if (ks + 1 < ks1) load_step<FMT, RT, NT>(nxt, wrow, srow, ar, sr, ks + 1, t);
      mma_step<FMT, RT, NT>(acc, cur, cnt);
      if (ks + 1 < ks1) cur = nxt;
    }
    // the quarters' sums, in order, into warp 0
    if (warp > 0) {
      float* dst = &part[warp - 1][0][lane];
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) dst[((rt * NT + j) * 4 + q) * 32] = acc[rt][j][q];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int w = 0; w < CK_WARPS - 1; ++w)
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
          for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q)
              acc[rt][j][q] = __fadd_rn(acc[rt][j][q], part[w][(rt * NT + j) * 4 + q][lane]);
      // epilogue: c0, c1 = row g, columns 2t, 2t + 1 of tile j; c2, c3 = row g + 8
      const float wa = FMT == NVFP4 ? __ldg(wamax + e) : 0.0f;
#pragma unroll
      for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mi = mem[rt][h];
          if (mi < 0) continue;
          float rs = 1.0f;
          if constexpr (FMT == NVFP4) rs = __fmul_rn(__fmul_rn(__ldg(Aamax + mi / x_div), wa), rcp);
          const float gm = EPI == UP ? __ldg(gamma + mi) : 0.0f;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = cb * (8 * NT) + 8 * j + 2 * t;
            float c0 = acc[rt][j][2 * h], c1 = acc[rt][j][2 * h + 1];
            if constexpr (FMT == NVFP4) c0 = __fmul_rn(c0, rs), c1 = __fmul_rn(c1, rs);
            if constexpr (EPI == UP) {
              const float a = __fmul_rn(capped_swiglu(bf16r(c0), bf16r(c1)), gm);
              out[(size_t)mi * ostride + (col >> 1)] = __float2bfloat16_rn(a);
            } else {
              *reinterpret_cast<uint32_t*>(out + (size_t)mi * ostride + col) = pack_bf16(c0, c1);
            }
          }
        }
    }
    __syncthreads();
  }
}

}  // namespace

#define TF_INK2_CK(NAME, FMT, EPI, RT, NT)                                                                          \
  extern "C" __global__ void __launch_bounds__(CK_WARPS * 32)                                                      \
      NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As, const float* __restrict__ Aamax,         \
           int x_div, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,                         \
           const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,                            \
           const int* __restrict__ items, const int* __restrict__ nitems, const int* __restrict__ members, int N,  \
           int ostride, __nv_bfloat16* __restrict__ out) {                                                          \
    ck_body<FMT, EPI, RT, NT>(A, As, Aamax, x_div, K, W, S, wamax, rcp, gamma, items, nitems, members, N, ostride, \
                              out);                                                                                 \
  }

// [format][up, down][shape]: shape 0 = 16 rows by 8 columns a warp (decode), 1 = 16 by 32, 2 = 64 by 32 (prompts)
TF_INK2_CK(tf_ink2_ck_up_nvfp4_s0, NVFP4, UP, 1, 1)
TF_INK2_CK(tf_ink2_ck_up_nvfp4_s1, NVFP4, UP, 1, 4)
TF_INK2_CK(tf_ink2_ck_up_nvfp4_s2, NVFP4, UP, 4, 4)
TF_INK2_CK(tf_ink2_ck_down_nvfp4_s0, NVFP4, DOWN, 1, 1)
TF_INK2_CK(tf_ink2_ck_down_nvfp4_s1, NVFP4, DOWN, 1, 4)
TF_INK2_CK(tf_ink2_ck_down_nvfp4_s2, NVFP4, DOWN, 4, 4)
TF_INK2_CK(tf_ink2_ck_up_mxfp8_s0, MXFP8, UP, 1, 1)
TF_INK2_CK(tf_ink2_ck_up_mxfp8_s1, MXFP8, UP, 1, 4)
TF_INK2_CK(tf_ink2_ck_up_mxfp8_s2, MXFP8, UP, 4, 4)
TF_INK2_CK(tf_ink2_ck_down_mxfp8_s0, MXFP8, DOWN, 1, 1)
TF_INK2_CK(tf_ink2_ck_down_mxfp8_s1, MXFP8, DOWN, 1, 4)
TF_INK2_CK(tf_ink2_ck_down_mxfp8_s2, MXFP8, DOWN, 4, 4)

// ------------------------------------------------------------------------------------------------- prompt tiles

namespace {

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

constexpr int TM = 64, TN = 128;  // a block's rows (one plan item) and columns; 8 warps 2 x 4, 32 x 32 a warp
constexpr int TRT = 2, TNT = 4;   // a warp's 16-row tiles and n8 tiles
constexpr int TWARPS = 8;
constexpr int RB = 128;           // bytes of a row a stage: 4 NVFP4 steps or 2 MXFP8 steps

}  // namespace

// The 64-row shape (prompts) with A and B staged in shared memory: a block a unit (item, 128 columns), warp (wm, wn)
// the rows 32 wm .. and columns 32 wn ..; a stage holds 128 contiguous bytes of each row (4 NVFP4 or 2 MXFP8 steps;
// 16-byte chunks XOR-swizzled by row), two stages, the next one loading (cp.async) while this one is read. Each
// output's sum is the decode form's: a fresh fragment a mma, one IEEE add into its K quarter's sum, the quarters added
// in order (the running total starts as quarter 0's sum and takes quarters 1, 2, 3), so a row's bits equal
// ck_body's at any shape.
template <int FMT, int EPI>
__device__ __forceinline__ void ck_tile(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As,
                                        const float* __restrict__ Aamax, int x_div, int K,
                                        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                        const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,
                                        const int* __restrict__ items, const int* __restrict__ nitems,
                                        const int* __restrict__ members, int N, int ostride,
                                        __nv_bfloat16* __restrict__ out) {
  const int NB = N / TN;
  constexpr int SB = FMT == NVFP4 ? 32 : 64;  // bytes of a row a step
  constexpr int SPS = RB / SB;                // steps a stage
  constexpr int CH = RB / 16;                 // 16-byte chunks of a row a stage
  __shared__ __align__(128) uint8_t sa_[2][TM][RB];
  __shared__ __align__(128) uint8_t sb_[2][TN][RB];
  const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, t = lane & 3, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const size_t arow_bytes = FMT == NVFP4 ? (size_t)K / 2 : (size_t)K;
  const size_t srow_bytes = FMT == NVFP4 ? (size_t)K / 16 : (size_t)K / 32;
  const int KS = K / KSTEP, QS = KS / 4, NS = KS / SPS;
  const int units = __ldg(nitems) * NB;
  auto chunk = [](int r, int c) { return 16 * (c ^ (r & 7)); };
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const uint8_t* wbase = W + ((size_t)e * N + (size_t)cb * TN) * arow_bytes;
    // stage ss into buffer bf: bytes ss * 128 .. of A's rows (zero past cnt) and of B's 128 columns
    auto stage = [&](int ss, int bf) {
#pragma unroll
      for (int k = 0; k < TM * CH / (TWARPS * 32); ++k) {
        const int i = tid + k * TWARPS * 32, r = i / CH, c = i - r * CH;
        const bool v = r < cnt;
        const int mi = v ? __ldg(members + first + r) : 0;
        cp_async16(&sa_[bf][r][chunk(r, c)], A + (size_t)(mi / x_div) * arow_bytes + ss * RB + 16 * c, v);
      }
#pragma unroll
      for (int k = 0; k < TN * CH / (TWARPS * 32); ++k) {
        const int i = tid + k * TWARPS * 32, n = i / CH, c = i - n * CH;
        cp_async16(&sb_[bf][n][chunk(n, c)], wbase + (size_t)n * arow_bytes + ss * RB + 16 * c, true);
      }
    };
    stage(0, 0);
    cp_commit();
    // the lane's scale rows: A row (t even: g, odd: g + 8) of each 16-row tile, B column g of each n8 tile
    const uint8_t* sr[TRT];
    const uint8_t* sc[TNT];
    bool live[TRT];
#pragma unroll
    for (int rt = 0; rt < TRT; ++rt) {
      const int r0 = wm * 32 + rt * 16, rs = r0 + g + 8 * (t & 1);
      live[rt] = r0 < cnt;
      sr[rt] = rs < cnt ? As + (size_t)(__ldg(members + first + rs) / x_div) * srow_bytes : nullptr;
    }
#pragma unroll
    for (int j = 0; j < TNT; ++j) sc[j] = S + ((size_t)e * N + (size_t)cb * TN + wn * 32 + 8 * j + g) * srow_bytes;
    auto scales = [&](int ks, uint32_t (&ra)[TRT], uint32_t (&rb)[TNT]) {
#pragma unroll
      for (int rt = 0; rt < TRT; ++rt) {
        if constexpr (FMT == NVFP4)
          ra[rt] = sr[rt] ? __ldg(reinterpret_cast<const uint32_t*>(sr[rt] + ks * 4)) : 0u;
        else
          ra[rt] = sr[rt] ? __ldg(reinterpret_cast<const uint16_t*>(sr[rt] + ks * 2)) : 0u;
      }
#pragma unroll
      for (int j = 0; j < TNT; ++j) {
        if constexpr (FMT == NVFP4)
          rb[j] = __ldg(reinterpret_cast<const uint32_t*>(sc[j] + ks * 4));
        else
          rb[j] = __ldg(reinterpret_cast<const uint16_t*>(sc[j] + ks * 2));
      }
    };
    float tot[TRT][TNT][4], acc[TRT][TNT][4];
#pragma unroll
    for (int rt = 0; rt < TRT; ++rt)
#pragma unroll
      for (int j = 0; j < TNT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) tot[rt][j][q] = 0.f, acc[rt][j][q] = 0.f;
    uint32_t ra[TRT], rb[TNT], na[TRT], nb[TNT];
    scales(0, ra, rb);
    for (int ss = 0; ss < NS; ++ss) {
      const int bf = ss & 1;
      if (ss + 1 < NS) {
        stage(ss + 1, bf ^ 1);  // that buffer was last read at stage ss - 1, closed by its barrier
        cp_commit();
        cp_wait<1>();
      } else {
        cp_wait<0>();
      }
      __syncthreads();  // stage ss landed for every thread
#pragma unroll
      for (int sub = 0; sub < SPS; ++sub) {
        const int ks = ss * SPS + sub;
        if (ks + 1 < KS) scales(ks + 1, na, nb);
#pragma unroll
        for (int rt = 0; rt < TRT; ++rt) {
          if (!live[rt]) continue;
          const int r0 = wm * 32 + rt * 16 + g, r1 = r0 + 8;
          if constexpr (FMT == NVFP4) {
            const int c0 = 2 * sub;
            const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(&sa_[bf][r0][chunk(r0, c0) + 4 * t]),
                                   *reinterpret_cast<const uint32_t*>(&sa_[bf][r1][chunk(r1, c0) + 4 * t]),
                                   *reinterpret_cast<const uint32_t*>(&sa_[bf][r0][chunk(r0, c0 + 1) + 4 * t]),
                                   *reinterpret_cast<const uint32_t*>(&sa_[bf][r1][chunk(r1, c0 + 1) + 4 * t])};
#pragma unroll
            for (int j = 0; j < TNT; ++j) {
              const int n = wn * 32 + 8 * j + g;
              const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sb_[bf][n][chunk(n, c0) + 4 * t]);
              const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sb_[bf][n][chunk(n, c0 + 1) + 4 * t]);
              float p[4] = {0.f, 0.f, 0.f, 0.f};
              mma_fp4(p, a, b0, b1, ra[rt], rb[j]);
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[q]);
            }
          } else {
            // bytes 8t .. 8t + 7 of each 32-block: chunk (t >> 1) of the block's two, its half t & 1
            const int c0 = 4 * sub + (t >> 1), hb = 8 * (t & 1);
            const uint2 x00 = *reinterpret_cast<const uint2*>(&sa_[bf][r0][chunk(r0, c0) + hb]);
            const uint2 x10 = *reinterpret_cast<const uint2*>(&sa_[bf][r1][chunk(r1, c0) + hb]);
            const uint2 x01 = *reinterpret_cast<const uint2*>(&sa_[bf][r0][chunk(r0, c0 + 2) + hb]);
            const uint2 x11 = *reinterpret_cast<const uint2*>(&sa_[bf][r1][chunk(r1, c0 + 2) + hb]);
            const uint32_t a0[4] = {x00.x, x10.x, x00.y, x10.y};
            const uint32_t a1[4] = {x01.x, x11.x, x01.y, x11.y};
#pragma unroll
            for (int j = 0; j < TNT; ++j) {
              const int n = wn * 32 + 8 * j + g;
              const uint2 q0 = *reinterpret_cast<const uint2*>(&sb_[bf][n][chunk(n, c0) + hb]);
              const uint2 q1 = *reinterpret_cast<const uint2*>(&sb_[bf][n][chunk(n, c0 + 2) + hb]);
              float p[4] = {0.f, 0.f, 0.f, 0.f};
              mma_mx8(p, a0, q0.x, q0.y, ra[rt] & 0xFFu, rb[j] & 0xFFu);
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[q]);
              float p2[4] = {0.f, 0.f, 0.f, 0.f};
              mma_mx8(p2, a1, q1.x, q1.y, ra[rt] >> 8, rb[j] >> 8);
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p2[q]);
            }
          }
        }
        if ((ks + 1) % QS == 0) {  // a quarter ends: the total takes its sum (quarter 0's sum is the total)
          const bool q0 = ks + 1 == QS;
#pragma unroll
          for (int rt = 0; rt < TRT; ++rt)
#pragma unroll
            for (int j = 0; j < TNT; ++j)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                tot[rt][j][q] = q0 ? acc[rt][j][q] : __fadd_rn(tot[rt][j][q], acc[rt][j][q]);
                acc[rt][j][q] = 0.f;
              }
        }
        if (ks + 1 < KS) {
#pragma unroll
          for (int rt = 0; rt < TRT; ++rt) ra[rt] = na[rt];
#pragma unroll
          for (int j = 0; j < TNT; ++j) rb[j] = nb[j];
        }
      }
      __syncthreads();  // every read of buffer bf is done before stage ss + 2 refills it
    }
    // epilogue (ck_body's): c0, c1 = row g, columns 2t, 2t + 1 of tile j; c2, c3 = row g + 8
    const float wa = FMT == NVFP4 ? __ldg(wamax + e) : 0.0f;
#pragma unroll
    for (int rt = 0; rt < TRT; ++rt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm * 32 + rt * 16 + g + 8 * h;
        if (r >= cnt) continue;
        const int mi = __ldg(members + first + r);
        float rs = 1.0f;
        if constexpr (FMT == NVFP4) rs = __fmul_rn(__fmul_rn(__ldg(Aamax + mi / x_div), wa), rcp);
        const float gm = EPI == UP ? __ldg(gamma + mi) : 0.0f;
#pragma unroll
        for (int j = 0; j < TNT; ++j) {
          const int col = cb * TN + wn * 32 + 8 * j + 2 * t;
          float c0 = tot[rt][j][2 * h], c1 = tot[rt][j][2 * h + 1];
          if constexpr (FMT == NVFP4) c0 = __fmul_rn(c0, rs), c1 = __fmul_rn(c1, rs);
          if constexpr (EPI == UP) {
            const float a = __fmul_rn(capped_swiglu(bf16r(c0), bf16r(c1)), gm);
            out[(size_t)mi * ostride + (col >> 1)] = __float2bfloat16_rn(a);
          } else {
            *reinterpret_cast<uint32_t*>(out + (size_t)mi * ostride + col) = pack_bf16(c0, c1);
          }
        }
      }
  }
}

#define TF_INK2_CK_TILE(NAME, FMT, EPI)                                                                             \
  extern "C" __global__ void __launch_bounds__(TWARPS * 32)                                                        \
      NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As, const float* __restrict__ Aamax,         \
           int x_div, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,                         \
           const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,                            \
           const int* __restrict__ items, const int* __restrict__ nitems, const int* __restrict__ members, int N,  \
           int ostride, __nv_bfloat16* __restrict__ out) {                                                          \
    ck_tile<FMT, EPI>(A, As, Aamax, x_div, K, W, S, wamax, rcp, gamma, items, nitems, members, N, ostride, out);   \
  }

// shape 3: 64 rows by 128 columns a block, staged (prompt chunks)
TF_INK2_CK_TILE(tf_ink2_ck_up_nvfp4_s3, NVFP4, UP)
TF_INK2_CK_TILE(tf_ink2_ck_down_nvfp4_s3, NVFP4, DOWN)
TF_INK2_CK_TILE(tf_ink2_ck_up_mxfp8_s3, MXFP8, UP)
TF_INK2_CK_TILE(tf_ink2_ck_down_mxfp8_s3, MXFP8, DOWN)

// ------------------------------------------------------------------------------------------------- window tiles

namespace {

template <int BYTES>
__device__ __forceinline__ void cp_async_small(void* smem, const void* gmem, bool valid) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  if constexpr (BYTES == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem), "r"(valid ? 16 : 0));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(d), "l"(gmem), "n"(BYTES),
                 "r"(valid ? BYTES : 0));
}

// A window tile's geometry (cuda_moe.zig's ck_geo holds the same numbers): TRT 16-row tiles a warp, WM x WN warps
// (TM = 16 TRT WM rows, TN = 32 WN columns a block), NSTG stages of RB bytes a row, SS: the block scales staged with
// the values (SSB bytes a row a stage) instead of loaded a step ahead.
template <int FMT, int TRT, int WM, int WN, int NSTG, int RB, bool SS>
struct Geo {
  static constexpr int TM = 16 * TRT * WM, TN = 32 * WN, THREADS = 32 * WM * WN;
  static constexpr int SB = FMT == NVFP4 ? 32 : 64;  // bytes of a row a step
  static constexpr int SPS = RB / SB;                 // steps a stage
  static constexpr int CH = RB / 16;                  // 16-byte chunks of a row a stage
  static constexpr int SSB = SS ? SPS * (FMT == NVFP4 ? 4 : 2) : 0;
  static constexpr int SMEM = NSTG * (TM + TN) * (RB + SSB);
};

}  // namespace

// Shapes 4, 5: the staged tile of shape 3 for windows and batched steps (calls of 2-8 rows, whose plan items hold a
// few routes each), its geometry and pipeline depth as parameters (Geo), shared memory dynamic: 16 rows (one mma row
// tile) by 64 columns, longer runs of each weight row a stage (GB10's LPDDR5x serves 384- or 256-byte runs a row
// faster than 128), and the block scales staged with the values (SS) rather than loaded a step ahead. The arithmetic is
// ck_tile's, step for step: a fresh fragment a mma, one IEEE add into its K quarter's sum, the quarters added in order,
// the same epilogue; only where the bytes wait (shared memory, how many stages ahead, the scales' path) differs, so a
// row's bits equal every other shape's.
template <int FMT, int EPI, int TRT, int WM, int WN, int NSTG, int RB, bool SS, bool PF = false, bool FOLD = false>
__device__ __forceinline__ void ck_stage(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As,
                                         const float* __restrict__ Aamax, int x_div, int K,
                                         const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                         const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,
                                         const int* __restrict__ items, const int* __restrict__ nitems,
                                         const int* __restrict__ members, int N, int ostride,
                                         __nv_bfloat16* __restrict__ out, int* __restrict__ cntr = nullptr,
                                         __nv_bfloat16* __restrict__ y = nullptr) {
  using G = Geo<FMT, TRT, WM, WN, NSTG, RB, SS>;
  constexpr int TM = G::TM, TN = G::TN, NTH = G::THREADS, SPS = G::SPS, CH = G::CH;
  constexpr int SSB = SS ? G::SSB : 4;  // (unused without SS)
  constexpr int SC = SSB % 16 == 0 ? 16 : SSB % 8 == 0 ? 8 : 4, SCR = SSB / SC;  // a scale copy, copies a row a stage
  static_assert(NSTG >= 2 && SPS >= 1 && CH % 8 == 0 && (!SS || SSB % 4 == 0), "window tile geometry");
  static_assert(TN * CH % NTH == 0, "B stage chunks a thread");
  extern __shared__ __align__(128) uint8_t smem[];
  {
    uint32_t have;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
    if (have < (uint32_t)G::SMEM) __trap();
  }
  auto sa_ = reinterpret_cast<uint8_t(*)[TM][RB]>(smem);
  auto sb_ = reinterpret_cast<uint8_t(*)[TN][RB]>(smem + NSTG * TM * RB);
  auto qa_ = reinterpret_cast<uint8_t(*)[TM][SSB]>(smem + NSTG * (TM + TN) * RB);
  auto qb_ = reinterpret_cast<uint8_t(*)[TN][SSB]>(smem + NSTG * (TM + TN) * RB + NSTG * TM * SSB);
  const int NB = N / TN;
  const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, t = lane & 3, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const size_t arow_bytes = FMT == NVFP4 ? (size_t)K / 2 : (size_t)K;
  const size_t srow_bytes = FMT == NVFP4 ? (size_t)K / 16 : (size_t)K / 32;
  const int KS = K / KSTEP, QS = KS / 4, NS = KS / SPS;
  const int units = __ldg(nitems) * NB;
  // 16-byte chunks XOR-swizzled by row within each group of 8 (the 8 rows a mma reads fall in different banks)
  auto chunk = [](int r, int c) { return 16 * ((c & ~7) | ((c ^ r) & 7)); };
  // a unit's place: its item's expert, first member and count, and its column block's weight and scale rows
  struct Unit {
    const uint8_t* wb;
    const uint8_t* sb;
    int e, cb, first, cnt;
  };
  auto unit_at = [&](int unit) {
    const int it = unit / NB;
    Unit u;
    u.cb = unit - it * NB;
    u.e = __ldg(items + 3 * it), u.first = __ldg(items + 3 * it + 1), u.cnt = __ldg(items + 3 * it + 2);
    u.wb = W + ((size_t)u.e * N + (size_t)u.cb * TN) * arow_bytes;
    u.sb = S + ((size_t)u.e * N + (size_t)u.cb * TN) * srow_bytes;
    return u;
  };
  // stage ss of unit u into buffer bf: bytes ss * RB .. of A's rows (zero past cnt) and of B's TN columns (with SS
  // their scales' bytes ss * SSB .. too)
  auto stage = [&](const Unit& u, int ss, int bf) {
#pragma unroll
    for (int k = 0; k < (TM * CH + NTH - 1) / NTH; ++k) {
      const int i = tid + k * NTH, r = i / CH, c = i - r * CH;
      if (TM * CH % NTH != 0 && i >= TM * CH) break;
      const bool v = r < u.cnt;
      const int mi = v ? __ldg(members + u.first + r) : 0;
      cp_async16(&sa_[bf][r][chunk(r, c)], A + (size_t)(mi / x_div) * arow_bytes + ss * RB + 16 * c, v);
    }
#pragma unroll
    for (int k = 0; k < TN * CH / NTH; ++k) {
      const int i = tid + k * NTH, n = i / CH, c = i - n * CH;
      cp_async16(&sb_[bf][n][chunk(n, c)], u.wb + (size_t)n * arow_bytes + ss * RB + 16 * c, true);
    }
    if constexpr (SS) {
#pragma unroll
      for (int k = 0; k < ((TM + TN) * SCR + NTH - 1) / NTH; ++k) {
        const int i = tid + k * NTH, r = i / SCR, c = i - r * SCR;
        if ((TM + TN) * SCR % NTH != 0 && i >= (TM + TN) * SCR) break;
        if (r < TM) {
          const bool v = r < u.cnt;
          const int mi = v ? __ldg(members + u.first + r) : 0;
          cp_async_small<SC>(&qa_[bf][r][SC * c], As + (size_t)(mi / x_div) * srow_bytes + ss * SSB + SC * c, v);
        } else {
          const int n = r - TM;
          cp_async_small<SC>(&qb_[bf][n][SC * c], u.sb + (size_t)n * srow_bytes + ss * SSB + SC * c, true);
        }
      }
    }
  };
  // PF (the persistent forms): a block walks its units with one ring of stages, the next unit's first NSTG - 1
  // stages loading while this unit's last ones are read (base: the ring place of the unit's stage 0; primed: this
  // unit's first stages were issued by the previous one). Without PF every unit fills and drains its own pipeline.
  int base = 0;
  bool primed = false;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const Unit U = unit_at(unit);
    const int e = U.e, cb = U.cb, first = U.first, cnt = U.cnt;
    const uint8_t* sbase = U.sb;
    const bool hasn = PF && NS >= NSTG - 1 && unit + (int)gridDim.x < units;
    Unit Nx = U;
    if (hasn) Nx = unit_at(unit + gridDim.x);
    if (!primed) {
      stage(U, 0, base % NSTG);
      cp_commit();
#pragma unroll
      for (int i = 1; i < NSTG - 1; ++i) {
        if (i < NS) stage(U, i, (base + i) % NSTG);
        cp_commit();
      }
    }
    // without SS: the lane's scale rows, A row (t even: g, odd: g + 8) of each 16-row tile, B column g of each n8
    // tile, loaded a step ahead as in ck_tile
    const uint8_t* sr[TRT];
    const uint8_t* sc[TNT];
    bool live[TRT];
#pragma unroll
    for (int rt = 0; rt < TRT; ++rt) {
      const int r0 = wm * 16 * TRT + rt * 16, rs = r0 + g + 8 * (t & 1);
      live[rt] = r0 < cnt;
      sr[rt] = !SS && rs < cnt ? As + (size_t)(__ldg(members + first + rs) / x_div) * srow_bytes : nullptr;
    }
#pragma unroll
    for (int j = 0; j < TNT; ++j) sc[j] = sbase + (size_t)(wn * 32 + 8 * j + g) * srow_bytes;
    auto scales = [&](int ks, uint32_t(&ra)[TRT], uint32_t(&rb)[TNT]) {
#pragma unroll
      for (int rt = 0; rt < TRT; ++rt) {
        if constexpr (FMT == NVFP4)
          ra[rt] = sr[rt] ? __ldg(reinterpret_cast<const uint32_t*>(sr[rt] + ks * 4)) : 0u;
        else
          ra[rt] = sr[rt] ? __ldg(reinterpret_cast<const uint16_t*>(sr[rt] + ks * 2)) : 0u;
      }
#pragma unroll
      for (int j = 0; j < TNT; ++j) {
        if constexpr (FMT == NVFP4)
          rb[j] = __ldg(reinterpret_cast<const uint32_t*>(sc[j] + ks * 4));
        else
          rb[j] = __ldg(reinterpret_cast<const uint16_t*>(sc[j] + ks * 2));
      }
    };
    float tot[TRT][TNT][4], acc[TRT][TNT][4];
#pragma unroll
    for (int rt = 0; rt < TRT; ++rt)
#pragma unroll
      for (int j = 0; j < TNT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) tot[rt][j][q] = 0.f, acc[rt][j][q] = 0.f;
    uint32_t ra[TRT], rb[TNT], na[TRT], nb[TNT];
    if constexpr (!SS) scales(0, ra, rb);
    for (int ss = 0; ss < NS; ++ss) {
      const int bf = (base + ss) % NSTG;
      // stage ss + NSTG - 1 into the buffer read at stage ss - 1 (closed by its barrier), or past this unit's last
      // stage the next unit's (PF); then at most NSTG - 1 groups younger than stage ss may still be in flight
      const int nx = ss + NSTG - 1;
      if (nx < NS)
        stage(U, nx, (base + nx) % NSTG);
      else if (hasn)
        stage(Nx, nx - NS, (base + nx) % NSTG);
      cp_commit();
      cp_wait<NSTG - 1>();
      __syncthreads();  // stage ss landed for every thread
#pragma unroll
      for (int sub = 0; sub < SPS; ++sub) {
        const int ks = ss * SPS + sub;
        if constexpr (SS) {
#pragma unroll
          for (int rt = 0; rt < TRT; ++rt) {
            const int rs = wm * 16 * TRT + rt * 16 + g + 8 * (t & 1);
            if constexpr (FMT == NVFP4)
              ra[rt] = *reinterpret_cast<const uint32_t*>(&qa_[bf][rs][4 * sub]);
            else
              ra[rt] = *reinterpret_cast<const uint16_t*>(&qa_[bf][rs][2 * sub]);
          }
#pragma unroll
          for (int j = 0; j < TNT; ++j) {
            const int n = wn * 32 + 8 * j + g;
            if constexpr (FMT == NVFP4)
              rb[j] = *reinterpret_cast<const uint32_t*>(&qb_[bf][n][4 * sub]);
            else
              rb[j] = *reinterpret_cast<const uint16_t*>(&qb_[bf][n][2 * sub]);
          }
        } else {
          if (ks + 1 < KS) scales(ks + 1, na, nb);
        }
#pragma unroll
        for (int rt = 0; rt < TRT; ++rt) {
          if (!live[rt]) continue;
          const int r0 = wm * 16 * TRT + rt * 16 + g, r1 = r0 + 8;
          if constexpr (FMT == NVFP4) {
            const int c0 = 2 * sub;
            const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(&sa_[bf][r0][chunk(r0, c0) + 4 * t]),
                                   *reinterpret_cast<const uint32_t*>(&sa_[bf][r1][chunk(r1, c0) + 4 * t]),
                                   *reinterpret_cast<const uint32_t*>(&sa_[bf][r0][chunk(r0, c0 + 1) + 4 * t]),
                                   *reinterpret_cast<const uint32_t*>(&sa_[bf][r1][chunk(r1, c0 + 1) + 4 * t])};
#pragma unroll
            for (int j = 0; j < TNT; ++j) {
              const int n = wn * 32 + 8 * j + g;
              const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sb_[bf][n][chunk(n, c0) + 4 * t]);
              const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sb_[bf][n][chunk(n, c0 + 1) + 4 * t]);
              float p[4] = {0.f, 0.f, 0.f, 0.f};
              mma_fp4(p, a, b0, b1, ra[rt], rb[j]);
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[q]);
            }
          } else {
            // bytes 8t .. 8t + 7 of each 32-block: chunk (t >> 1) of the block's two, its half t & 1
            const int c0 = 4 * sub + (t >> 1), hb = 8 * (t & 1);
            const uint2 x00 = *reinterpret_cast<const uint2*>(&sa_[bf][r0][chunk(r0, c0) + hb]);
            const uint2 x10 = *reinterpret_cast<const uint2*>(&sa_[bf][r1][chunk(r1, c0) + hb]);
            const uint2 x01 = *reinterpret_cast<const uint2*>(&sa_[bf][r0][chunk(r0, c0 + 2) + hb]);
            const uint2 x11 = *reinterpret_cast<const uint2*>(&sa_[bf][r1][chunk(r1, c0 + 2) + hb]);
            const uint32_t a0[4] = {x00.x, x10.x, x00.y, x10.y};
            const uint32_t a1[4] = {x01.x, x11.x, x01.y, x11.y};
#pragma unroll
            for (int j = 0; j < TNT; ++j) {
              const int n = wn * 32 + 8 * j + g;
              const uint2 q0 = *reinterpret_cast<const uint2*>(&sb_[bf][n][chunk(n, c0) + hb]);
              const uint2 q1 = *reinterpret_cast<const uint2*>(&sb_[bf][n][chunk(n, c0 + 2) + hb]);
              float p[4] = {0.f, 0.f, 0.f, 0.f};
              mma_mx8(p, a0, q0.x, q0.y, ra[rt] & 0xFFu, rb[j] & 0xFFu);
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[q]);
              float p2[4] = {0.f, 0.f, 0.f, 0.f};
              mma_mx8(p2, a1, q1.x, q1.y, ra[rt] >> 8, rb[j] >> 8);
#pragma unroll
              for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p2[q]);
            }
          }
        }
        if ((ks + 1) % QS == 0) {  // a quarter ends: the total takes its sum (quarter 0's sum is the total)
          const bool q0 = ks + 1 == QS;
#pragma unroll
          for (int rt = 0; rt < TRT; ++rt)
#pragma unroll
            for (int j = 0; j < TNT; ++j)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                tot[rt][j][q] = q0 ? acc[rt][j][q] : __fadd_rn(tot[rt][j][q], acc[rt][j][q]);
                acc[rt][j][q] = 0.f;
              }
        }
        if constexpr (!SS) {
          if (ks + 1 < KS) {
#pragma unroll
            for (int rt = 0; rt < TRT; ++rt) ra[rt] = na[rt];
#pragma unroll
            for (int j = 0; j < TNT; ++j) rb[j] = nb[j];
          }
        }
      }
      __syncthreads();  // every read of buffer bf is done before a later stage refills it
    }
    if constexpr (!PF) cp_wait<0>();  // the trailing (empty) groups (PF: the next unit's stages, left in flight)
    // epilogue (ck_body's): c0, c1 = row g, columns 2t, 2t + 1 of tile j; c2, c3 = row g + 8
    const float wa = FMT == NVFP4 ? __ldg(wamax + e) : 0.0f;
#pragma unroll
    for (int rt = 0; rt < TRT; ++rt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm * 16 * TRT + rt * 16 + g + 8 * h;
        if (r >= cnt) continue;
        const int mi = __ldg(members + first + r);
        float rs = 1.0f;
        if constexpr (FMT == NVFP4) rs = __fmul_rn(__fmul_rn(__ldg(Aamax + mi / x_div), wa), rcp);
        const float gm = EPI == UP ? __ldg(gamma + mi) : 0.0f;
#pragma unroll
        for (int j = 0; j < TNT; ++j) {
          const int col = cb * TN + wn * 32 + 8 * j + 2 * t;
          float c0 = tot[rt][j][2 * h], c1 = tot[rt][j][2 * h + 1];
          if constexpr (FMT == NVFP4) c0 = __fmul_rn(c0, rs), c1 = __fmul_rn(c1, rs);
          if constexpr (EPI == UP) {
            const float a = __fmul_rn(capped_swiglu(bf16r(c0), bf16r(c1)), gm);
            out[(size_t)mi * ostride + (col >> 1)] = __float2bfloat16_rn(a);
          } else {
            *reinterpret_cast<uint32_t*>(out + (size_t)mi * ostride + col) = pack_bf16(c0, c1);
          }
        }
      }
    if constexpr (FOLD) {
      // the fold (tf_ink2_ck_fold's, split experts): each of the unit's rows counts this column block's route rows
      // stored (cntr [R][NB], zeroed by the plan's launch); the block that stores a row's eighth one sums that row's
      // TOPK d rows in slot order in fp32 from 0, one bf16 rounding, into y [R][N] (columns cb * TN ..). d is read back
      // through L2 (other blocks wrote it in this launch), after their fences and the counter.
      __shared__ int fold_row[TM];
      __threadfence();
      __syncthreads();
      if (tid < cnt) {
        const int row = __ldg(members + first + tid) / TOPK;
        fold_row[tid] = atomicAdd(cntr + (size_t)row * NB + cb, 1) == TOPK - 1 ? row : -1;
      }
      __syncthreads();
      for (int r = 0; r < cnt; ++r) {
        const int row = fold_row[r];
        if (row < 0) continue;
        __threadfence();
        for (int c = tid; c < TN / 2; c += NTH) {
          const int c2 = cb * (TN / 2) + c;
          float a0 = 0.f, a1 = 0.f;
#pragma unroll
          for (int k = 0; k < TOPK; ++k) {
            const uint32_t v = __ldcg(reinterpret_cast<const unsigned int*>(out + ((size_t)row * TOPK + k) * ostride) + c2);
            a0 = __fadd_rn(a0, lo_f(v));
            a1 = __fadd_rn(a1, hi_f(v));
          }
          reinterpret_cast<uint32_t*>(y + (size_t)row * N)[c2] = pack_bf16(a0, a1);
        }
      }
      __syncthreads();  // fold_row is the next unit's
    }
    if constexpr (PF) {
      base += NS;
      primed = hasn;
    }
  }
  if constexpr (PF) cp_wait<0>();
}

#define TF_INK2_CK_STAGE(SH, TRT, WM, WN, NSTG, RB, SS) TF_INK2_CK_STAGEP(SH, TRT, WM, WN, NSTG, RB, SS, false)
#define TF_INK2_CK_STAGEP(SH, TRT, WM, WN, NSTG, RB, SS, PF)                                                        \
  TF_INK2_CK_STAGE1(tf_ink2_ck_up_nvfp4_s##SH, NVFP4, UP, TRT, WM, WN, NSTG, RB, SS, PF)                            \
  TF_INK2_CK_STAGE1(tf_ink2_ck_down_nvfp4_s##SH, NVFP4, DOWN, TRT, WM, WN, NSTG, RB, SS, PF)                        \
  TF_INK2_CK_STAGE1(tf_ink2_ck_up_mxfp8_s##SH, MXFP8, UP, TRT, WM, WN, NSTG, RB, SS, PF)                            \
  TF_INK2_CK_STAGE1(tf_ink2_ck_down_mxfp8_s##SH, MXFP8, DOWN, TRT, WM, WN, NSTG, RB, SS, PF)
#define TF_INK2_CK_STAGE1(NAME, FMT, EPI, TRT, WM, WN, NSTG, RB, SS, PF)                                            \
  extern "C" __global__ void __launch_bounds__(WM * WN * 32)                                                       \
      NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As, const float* __restrict__ Aamax,         \
           int x_div, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,                         \
           const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,                            \
           const int* __restrict__ items, const int* __restrict__ nitems, const int* __restrict__ members, int N,  \
           int ostride, __nv_bfloat16* __restrict__ out) {                                                          \
    ck_stage<FMT, EPI, TRT, WM, WN, NSTG, RB, SS, PF>(A, As, Aamax, x_div, K, W, S, wamax, rcp, gamma, items, nitems,  \
                                                  members, N, ostride, out);                                       \
  }

// The down GEMM with the fold (K1, split experts, calls of up to 8 rows): ck_stage's down pass, then each row's fold
// by the block that stores its last route row (see FOLD there); y [R][N] bf16, cntr [R][N / TN] ints, zero at entry.
#define TF_INK2_CK_STAGEF(SH, TRT, WM, WN, NSTG, RB, SS, PF)                                                        \
  TF_INK2_CK_STAGEF1(tf_ink2_ck_downf_nvfp4_s##SH, NVFP4, TRT, WM, WN, NSTG, RB, SS, PF)                            \
  TF_INK2_CK_STAGEF1(tf_ink2_ck_downf_mxfp8_s##SH, MXFP8, TRT, WM, WN, NSTG, RB, SS, PF)
#define TF_INK2_CK_STAGEF1(NAME, FMT, TRT, WM, WN, NSTG, RB, SS, PF)                                                \
  extern "C" __global__ void __launch_bounds__(WM * WN * 32)                                                       \
      NAME(const uint8_t* __restrict__ A, const uint8_t* __restrict__ As, const float* __restrict__ Aamax,         \
           int x_div, int K, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,                         \
           const float* __restrict__ wamax, float rcp, const float* __restrict__ gamma,                            \
           const int* __restrict__ items, const int* __restrict__ nitems, const int* __restrict__ members, int N,  \
           int ostride, __nv_bfloat16* __restrict__ out, int* __restrict__ cntr, __nv_bfloat16* __restrict__ y) {  \
    ck_stage<FMT, DOWN, TRT, WM, WN, NSTG, RB, SS, PF, true>(A, As, Aamax, x_div, K, W, S, wamax, rcp, gamma,      \
                                                             items, nitems, members, N, ostride, out, cntr, y);   \
  }

// shape: TRT, WM, WN, stages, bytes a row a stage, scales staged (cuda_moe.zig ck_stages: the same table)
TF_INK2_CK_STAGE(4, 1, 1, 2, 2, 384, true)  // 16 x 64, 2 stages of 384 bytes (NVFP4 windows)
TF_INK2_CK_STAGE(5, 1, 1, 2, 3, 256, true)  // 16 x 64, 3 stages of 256 bytes (MXFP8 windows, NVFP4 up from 6 rows)
// candidates for one-row calls on a GB10 (two blocks an SM, or a deeper pipeline); the same arithmetic
TF_INK2_CK_STAGE(6, 1, 1, 1, 2, 384, true)  // 16 x 32, 2 stages of 384 bytes (41 KiB: two blocks an SM)
TF_INK2_CK_STAGE(7, 1, 1, 2, 4, 128, true)  // 16 x 64, 4 stages of 128 bytes (46 KiB: two blocks an SM)
TF_INK2_CK_STAGE(8, 1, 1, 1, 3, 384, true)  // 16 x 32, 3 stages of 384 bytes
TF_INK2_CK_STAGE(9, 1, 1, 2, 2, 256, true)  // 16 x 64, 2 stages of 256 bytes (46 KiB; NVFP4 up and MXFP8 only)
// the persistent forms of 4, 5, 6, 7 (PF): launched with fewer blocks than units, each block walking several
TF_INK2_CK_STAGEP(10, 1, 1, 2, 2, 384, true, true)
TF_INK2_CK_STAGEP(11, 1, 1, 2, 3, 256, true, true)
TF_INK2_CK_STAGEP(12, 1, 1, 1, 2, 384, true, true)
TF_INK2_CK_STAGEP(13, 1, 1, 2, 4, 128, true, true)
// the folding down GEMMs of every window tile (the same numbers)
TF_INK2_CK_STAGEF(4, 1, 1, 2, 2, 384, true, false)
TF_INK2_CK_STAGEF(5, 1, 1, 2, 3, 256, true, false)
TF_INK2_CK_STAGEF(6, 1, 1, 1, 2, 384, true, false)
TF_INK2_CK_STAGEF(7, 1, 1, 2, 4, 128, true, false)
TF_INK2_CK_STAGEF(8, 1, 1, 1, 3, 384, true, false)
TF_INK2_CK_STAGEF(9, 1, 1, 2, 2, 256, true, false)
TF_INK2_CK_STAGEF(10, 1, 1, 2, 2, 384, true, true)
TF_INK2_CK_STAGEF(11, 1, 1, 2, 3, 256, true, true)
TF_INK2_CK_STAGEF(12, 1, 1, 1, 2, 384, true, true)
TF_INK2_CK_STAGEF(13, 1, 1, 2, 4, 128, true, true)

// ------------------------------------------------------------------------------------------------- routes

// The route plan (tf_ink2_moe_plan's output, the same bits) without its per-expert scans of every route: one block of
// 1024 threads. Routes i = row * 8 + slot whose id is in [lo, lo + E) grouped by expert, each expert's members in route
// order, items (expert, first member, count <= tile), *nitems their number. A row picks an expert at most once, so an
// expert's route order is its row order: a route's place is the count of its expert's earlier rows, read from a row
// bitmap an expert (bm: scratch u32 [2][E][ceil(R / 32)], the bitmaps then their words' prefix counts).
namespace {
// The plan by one block of 1024 threads (cnt, off: 1024 ints each, wsum: [2][32], in shared memory).
__device__ __forceinline__ void plan_body(const int* __restrict__ ids, int routes, int lo, int E, int tile,
                                          int* __restrict__ members, int* __restrict__ items,
                                          int* __restrict__ nitems, uint32_t* __restrict__ bm, int* cnt, int* off,
                                          int (*wsum)[32]) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int R = routes / TOPK, NW = (R + 31) >> 5;
  uint32_t* pre = bm + (size_t)E * NW;
  for (int i = t; i < E * NW; i += blockDim.x) bm[i] = 0u;
  cnt[t] = 0;
  __syncthreads();
  for (int j = t; j < routes; j += blockDim.x) {
    const int e = __ldg(ids + j) - lo;
    if (e < 0 || e >= E) continue;
    const int row = j / TOPK;
    atomicOr(bm + (size_t)e * NW + (row >> 5), 1u << (row & 31));
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int c = t < E ? cnt[t] : 0;
  const int ni = (c + tile - 1) / tile;
  if (t < E) {
    const uint32_t* b = bm + (size_t)t * NW;
    uint32_t* p = pre + (size_t)t * NW;
    uint32_t sum = 0u;
#pragma unroll 8
    for (int w = 0; w < NW; ++w) {
      p[w] = sum;
      sum += __popc(b[w]);
    }
  }
  // exclusive prefix sums of the counts (member offsets) and of the items a expert, over the block
  int a = c, b = ni;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int x = __shfl_up_sync(0xFFFFFFFFu, a, o), y = __shfl_up_sync(0xFFFFFFFFu, b, o);
    if (lane >= o) a += x, b += y;
  }
  if (lane == 31) wsum[0][warp] = a, wsum[1][warp] = b;
  __syncthreads();
  if (warp == 0) {
    int x = wsum[0][lane], y = wsum[1][lane];
    const int x0 = x, y0 = y;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xFFFFFFFFu, x, o), v = __shfl_up_sync(0xFFFFFFFFu, y, o);
      if (lane >= o) x += u, y += v;
    }
    wsum[0][lane] = x - x0;
    wsum[1][lane] = y - y0;
  }
  __syncthreads();
  const int first = a - c + wsum[0][warp], ifirst = b - ni + wsum[1][warp];
  off[t] = first;
  if (t == (int)blockDim.x - 1) *nitems = ifirst + ni;
  for (int x = 0; x < ni; ++x) {
    int* it = items + 3 * (ifirst + x);
    it[0] = t;
    it[1] = first + x * tile;
    it[2] = min(tile, c - x * tile);
  }
  __syncthreads();
  for (int j = t; j < routes; j += blockDim.x) {
    const int e = __ldg(ids + j) - lo;
    if (e < 0 || e >= E) continue;
    const int row = j / TOPK, w = row >> 5;
    const size_t at = (size_t)e * NW + w;
    members[off[e] + (int)pre[at] + __popc(bm[at] & ((1u << (row & 31)) - 1u))] = j;
  }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(1024)
    tf_ink2_ck_plan(const int* __restrict__ ids, int routes, int lo, int E, int tile, int* __restrict__ members,
                    int* __restrict__ items, int* __restrict__ nitems, uint32_t* __restrict__ bm) {
  __shared__ int cnt[1024];
  __shared__ int off[1024];
  __shared__ int wsum[2][32];
  plan_body(ids, routes, lo, E, tile, members, items, nitems, bm, cnt, off, wsum);
}

// K1 (cuda_moe.zig Form.k1, calls of up to 8 rows): the plan and the tokens' rows quantized in one launch of 1024-thread
// blocks: block 0 zeroes the fold counters (ncntr ints, read by the down GEMM's folding form) and runs the plan; block
// 1 + i quantizes row i (dense x [R, K]). A quantizer's threads past its row's scale blocks take no block and add 0 to
// the amax (a max: the same value), so every output equals tf_ink2_ck_plan's and tf_ink2_ck_quant_*'s.
template <bool NV>
__device__ __forceinline__ void plan_quant(const int* __restrict__ ids, int routes, int lo, int E, int tile,
                                           int* __restrict__ members, int* __restrict__ items,
                                           int* __restrict__ nitems, uint32_t* __restrict__ bm,
                                           int* __restrict__ cntr, int ncntr, const __nv_bfloat16* __restrict__ x,
                                           int K, uint8_t* __restrict__ q, uint8_t* __restrict__ sc,
                                           float* __restrict__ amax) {
  __shared__ int cnt[1024];
  __shared__ int off[1024];
  __shared__ int wsum[2][32];
  __shared__ float red[32];
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < ncntr; i += blockDim.x) cntr[i] = 0;
    plan_body(ids, routes, lo, E, tile, members, items, nitems, bm, cnt, off, wsum);
    return;
  }
  if constexpr (NV)
    quant_nvfp4_row(x, K, K, 0, blockIdx.x - 1, q, sc, amax, red);
  else
    quant_mxfp8_row(x, K, K, 0, blockIdx.x - 1, q, sc);
}

extern "C" __global__ void __launch_bounds__(1024)
    tf_ink2_ck_planq_nvfp4(const int* __restrict__ ids, int routes, int lo, int E, int tile,
                           int* __restrict__ members, int* __restrict__ items, int* __restrict__ nitems,
                           uint32_t* __restrict__ bm, int* __restrict__ cntr, int ncntr,
                           const __nv_bfloat16* __restrict__ x, int K, uint8_t* __restrict__ q,
                           uint8_t* __restrict__ sc, float* __restrict__ amax) {
  plan_quant<true>(ids, routes, lo, E, tile, members, items, nitems, bm, cntr, ncntr, x, K, q, sc, amax);
}

extern "C" __global__ void __launch_bounds__(1024)
    tf_ink2_ck_planq_mxfp8(const int* __restrict__ ids, int routes, int lo, int E, int tile,
                           int* __restrict__ members, int* __restrict__ items, int* __restrict__ nitems,
                           uint32_t* __restrict__ bm, int* __restrict__ cntr, int ncntr,
                           const __nv_bfloat16* __restrict__ x, int K, uint8_t* __restrict__ q,
                           uint8_t* __restrict__ sc, float* __restrict__ amax) {
  plan_quant<false>(ids, routes, lo, E, tile, members, items, nitems, bm, cntr, ncntr, x, K, q, sc, amax);
}

// The router's picks (ascending id, from inkling2_moe.cu tf_ink2_moe_select) put in TokenSpeed's slot order:
// descending sigmoid(logit) + bias, ties to the lower id (a stable sort of the ascending input). The weights move with
// their ids (the same values: Transformers' router in both modes). A thread a row.
extern "C" __global__ void __launch_bounds__(128)
    tf_ink2_ck_slot_order(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
                          float* __restrict__ wts, int R) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  int id[TOPK];
  float w[TOPK], sc[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    id[k] = ids[(size_t)row * TOPK + k];
    w[k] = wts[(size_t)row * TOPK + k];
    if (id[k] < 0) {
      sc[k] = -INFINITY;
      continue;
    }
    const float l = __ldg(logits + (size_t)row * 384 + id[k]);
    sc[k] = __fadd_rn(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l))), __ldg(bias + id[k]));
  }
#pragma unroll
  for (int i = 1; i < TOPK; ++i)
#pragma unroll
    for (int j = i; j > 0; --j)
      if (sc[j] > sc[j - 1]) {
        const float ts = sc[j], tw = w[j];
        const int ti = id[j];
        sc[j] = sc[j - 1], w[j] = w[j - 1], id[j] = id[j - 1];
        sc[j - 1] = ts, w[j - 1] = tw, id[j - 1] = ti;
      }
#pragma unroll
  for (int k = 0; k < TOPK; ++k) ids[(size_t)row * TOPK + k] = id[k], wts[(size_t)row * TOPK + k] = w[k];
}

// out [R, N] = the fp32 sum from 0 of the row's route rows d[row * 8 + slot] (bf16 [R * 8, N], the routing weight
// already inside) in slot order, routes of experts outside [lo, hi) skipped (ids null: none skipped); bf16 once (W = 1, or a split rank's
// columns), or the fp32 rank partial (whole experts at W > 1). N = 4608, or a split rank's 4608 / W.
namespace {
template <bool F32>
__device__ __forceinline__ void fold_body(const __nv_bfloat16* __restrict__ d, const int* __restrict__ ids, int lo,
                                          int hi, int N, void* __restrict__ out) {
  const int row = blockIdx.x;
  bool here[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    if (ids == nullptr) {  // split experts: every route is this rank's
      here[k] = true;
      continue;
    }
    const int e = __ldg(ids + (size_t)row * TOPK + k);
    here[k] = e >= lo && e < hi;
  }
  for (int c = threadIdx.x; c < N / 2; c += blockDim.x) {
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if (!here[k]) continue;
      const uint32_t v = __ldg(reinterpret_cast<const unsigned int*>(d + ((size_t)row * TOPK + k) * N) + c);
      a0 = __fadd_rn(a0, lo_f(v));
      a1 = __fadd_rn(a1, hi_f(v));
    }
    if constexpr (F32) {
      reinterpret_cast<float2*>(static_cast<float*>(out) + (size_t)row * N)[c] = make_float2(a0, a1);
    } else {
      reinterpret_cast<uint32_t*>(static_cast<__nv_bfloat16*>(out) + (size_t)row * N)[c] = pack_bf16(a0, a1);
    }
  }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_ck_fold(const __nv_bfloat16* __restrict__ d, const int* __restrict__ ids, int lo, int hi, int N,
                    __nv_bfloat16* __restrict__ out) {
  fold_body<false>(d, ids, lo, hi, N, out);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_ck_fold_f32(const __nv_bfloat16* __restrict__ d, const int* __restrict__ ids, int lo, int hi, int N,
                        float* __restrict__ out) {
  fold_body<true>(d, ids, lo, hi, N, out);
}

// Split experts, after the column all-gather: out bf16 [R, W * n] from parts bf16 [W][R][n] (rank w's columns
// w * n ..). A block a row, 4 bytes a thread a step (a plain copy: no arithmetic).
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_ck_columns(const __nv_bfloat16* __restrict__ parts, int W, int R, int n, __nv_bfloat16* __restrict__ out) {
  const int row = blockIdx.x;
  const int half = n / 2;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(parts);
  uint32_t* dst = reinterpret_cast<uint32_t*>(out + (size_t)row * W * n);
  for (int c = threadIdx.x; c < W * half; c += blockDim.x) {
    const int w = c / half, j = c - w * half;
    dst[c] = __ldg(src + ((size_t)w * R + row) * half + j);
  }
}

// ------------------------------------------------------------------------------------------------- MoE-A as codes
// Split experts, prompt chunks (cuda_moe.zig splitAmax / splitQuant / splitDownCodes; TF_INK2_XCODES): each rank
// quantizes its own I/W columns of every route's act row before MoE-A and the ranks gather the codes and scales in
// place of the bf16 columns. The down pass then reads the same codes, scales and amax the whole-row quantizer above
// leaves. MXFP8 is block-local (32 values, none straddles ranks: I/W = 576). NVFP4's only cross-block input is the
// row's amax, a max (exact in any order): each rank gathers every rank's segment amax first, and each 16-value block
// then takes the same steps as tf_ink2_ck_quant_nvfp4's second loop with that amax.

namespace {
// tf_ink2_ck_quant_nvfp4's second loop for one row: x the row's K values (dense), the row's amax.
__device__ __forceinline__ void quant_nvfp4_row(const __nv_bfloat16* x, int K, float amax, uint8_t* codes,
                                                uint8_t* scales) {
  const int nb = K / 16;
  const float encoder = clamp_max(__fdiv_rn(1536.0f, amax), FMAX);
  const float decoder = __fdiv_rn(1.0f, encoder);
  const float outer = __fmul_rn(amax, INV_1536);
  const float enc6 = __fmul_rn(encoder, INV_6), enc4 = __fmul_rn(encoder, 0.25f);
  for (int b = threadIdx.x; b < nb; b += blockDim.x) {
    float f[16];
    unpack16(x + 16 * b, f);
    float maxima = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) maxima = max_nan(maxima, fabsf(f[i]));
    uint32_t code[2][16];
    uint32_t s8[2];
    float err[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const float sv = clamp_max(__fmul_rn(maxima, c == 0 ? enc6 : enc4), 448.0f);
      s8[c] = f2e4m3(sv);
      const float sf = e4m3f(s8[c]);
      const float inv = clamp_max(__fdiv_rn(1.0f, __fmul_rn(sf, decoder)), FMAX);
      float e[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float r;
        code[c][i] = fp4_round(__fmul_rn(f[i], inv), r);
        e[i] = fabsf(__fsub_rn(f[i], __fmul_rn(__fmul_rn(r, sf), outer)));
      }
#pragma unroll
      for (int w = 8; w >= 1; w >>= 1)
#pragma unroll
        for (int i = 0; i < w; ++i) e[i] = __fadd_rn(e[2 * i], e[2 * i + 1]);
      err[c] = e[0];
    }
    const int pick = err[1] < err[0] ? 1 : 0;
    uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) w0 |= code[pick][i] << (4 * i), w1 |= code[pick][8 + i] << (4 * i);
    *reinterpret_cast<uint2*>(codes + 8 * b) = make_uint2(w0, w1);
    scales[b] = static_cast<uint8_t>(s8[pick]);
  }
}

// The row amax over the ranks' segment amaxes [W][rows] (as block_max's: max_nan from 0, any order).
__device__ __forceinline__ float parts_amax(const float* amaxes, int W, int rows, int row) {
  float m = 0.0f;
  for (int w = 0; w < W; ++w) m = max_nan(m, __ldg(amaxes + (size_t)w * rows + row));
  return m;
}
}  // namespace

// A block a row: max_nan(0, |x|) over the row's K values (x dense [rows, K]) into amax[row].
extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_ck_seg_amax(const __nv_bfloat16* __restrict__ x, int K, float* __restrict__ amax) {
  __shared__ float red[16];
  const int row = blockIdx.x;
  float m = 0.0f;
  for (int b = threadIdx.x; b < K / 16; b += blockDim.x) {
    float f[16];
    unpack16(x + (size_t)row * K + 16 * b, f);
#pragma unroll
    for (int i = 0; i < 16; ++i) m = max_nan(m, fabsf(f[i]));
  }
  const float a = block_max(m, red);
  if (threadIdx.x == 0) amax[row] = a;
}

// A block a row: the rank's K values of the row (x dense [rows, K]) as NVFP4 with the row's amax over every rank's
// segment amax (amaxes [W][rows]): codes u8 [rows, K/2], scales e4m3 [rows, K/16].
extern "C" __global__ void __launch_bounds__(512)
    tf_ink2_ck_quant_nvfp4_parts(const __nv_bfloat16* __restrict__ x, int K, const float* __restrict__ amaxes, int W,
                                 int rows, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
  const int row = blockIdx.x;
  quant_nvfp4_row(x + (size_t)row * K, K, parts_amax(amaxes, W, rows, row), codes + (size_t)row * (K / 2),
                  scales + (size_t)row * (K / 16));
}

// A block a row: every rank's gathered slot (slot w at parts + w * slot bytes: codes [rows][cb] then scales
// [rows][sb]) laid out as the whole row's codes aq [rows][W * cb] and scales asc [rows][W * sb], rank order; with
// amaxes (NVFP4), the row's amax into aa[row]. A plain copy (and a max).
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_ck_codes_rows(const uint8_t* __restrict__ parts, long long slot, int W, int rows, int cb, int sb,
                          const float* __restrict__ amaxes, uint8_t* __restrict__ aq, uint8_t* __restrict__ asc,
                          float* __restrict__ aa) {
  const int row = blockIdx.x;
  const int cw = cb / 4;
  uint32_t* dq = reinterpret_cast<uint32_t*>(aq + (size_t)row * W * cb);
  for (int c = threadIdx.x; c < W * cw; c += blockDim.x) {
    const int w = c / cw, j = c - w * cw;
    dq[c] = __ldg(reinterpret_cast<const uint32_t*>(parts + (size_t)w * slot + (size_t)row * cb) + j);
  }
  uint8_t* ds = asc + (size_t)row * W * sb;
  for (int c = threadIdx.x; c < W * sb; c += blockDim.x) {
    const int w = c / sb, j = c - w * sb;
    ds[c] = __ldg(parts + (size_t)w * slot + (size_t)rows * cb + (size_t)row * sb + j);
  }
  if (amaxes != nullptr && threadIdx.x == 0) aa[row] = parts_amax(amaxes, W, rows, row);
}

// The module's presence check.
extern "C" __global__ void tf_ink2_ck_probe(int* out) {
  if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 6;
}
