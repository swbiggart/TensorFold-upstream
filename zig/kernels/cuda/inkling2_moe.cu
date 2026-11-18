// Inkling-2's MoE group (G-moe): the sigmoid router with its selection bias, and the grouped experts in full mode
// (NVFP4 W4A16 and MXFP8 W8A16 decoded in the kernel, softcapped SwiGLU, the fp32 sum in expert-id order). The
// entry points, shapes and math are in zig/src/families/inkling2/cuda_moe.zig; their kernels go here.
//
// The math follows Transformers' modeling_inkling2.py (Apache-2.0: Inkling2TopKRouter, Inkling2Experts,
// Inkling2SoftcappedSwiGLU) step for step, each torch op one rounding here: built with --fmad=false, IEEE division.
// Every output element is computed by one fixed sequence that does not depend on how many rows a call holds (no
// split K, the same k order for every tile shape), so a row's bits do not depend on its window or chunk.
//
// Expert GEMMs: m16n8k16 bf16 MMAs, each 64-k step's into a fresh fragment that one IEEE add puts into the fp32
// sum (the tensor core aligns and truncates inside its accumulation; a short chain keeps that error at the step's
// scale, not the whole sum's). A warp owns 16 * RT rows of one expert by 32 output
// columns (four n8 tiles) and walks K in steps of 64. Within a step, lane t of a quad owns 16 contiguous k
// (16t .. 16t + 15) of both operands, and MMA s of the step takes k 16t + 4s + {0, 1} as the fragment's slots
// (2t, 2t + 1) and 16t + 4s + {2, 3} as (2t + 8, 2t + 9): a permutation of k inside the step that A and B share, so
// each lane reads contiguous bytes of a weight row (8 for NVFP4, 16 for MXFP8) and of an input row (32).
// Weights are decoded to the oracle's bf16 values before the MMA (tools/zig/inkling2_ref.py dequant_rows):
//   NVFP4: bf16(f32(f32(e2m1 * e4m3) * f32(amax[e] / den)))     MXFP8: bf16(e4m3 * 2^(ue8m0 - 127)) (exact)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr int HID = 4608;
constexpr int NEXP = 384;
constexpr int TOPK = 8;
constexpr int INTER = 2304;
constexpr float ROUTE_SCALE = 8.0f;
constexpr float GATE_CAP = 4.0f;
constexpr float UP_CAP = 25.0f;

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float lo_f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_f(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ld_nc4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_nc2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_nc_u8(const void* p) {
  uint16_t r;
  asm volatile("ld.global.nc.u8 %0, [%1];\n" : "=h"(r) : "l"(p));
  return r;
}

// One E2M1 code (low 4 bits) as fp32: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (code 8 is -0).
__device__ __forceinline__ float e2m1f(uint32_t c) {
  const uint32_t mag = c & 7u;
  const uint32_t bits = mag < 2u ? (mag ? 0x3F000000u : 0u) : ((((mag >> 1) + 126u) << 23) | ((mag & 1u) << 22));
  return __uint_as_float(bits | ((c & 8u) << 28));
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

// A UE8M0 byte as 2^(b - 127); 0 is 2^-127 (a denormal), 255 NaN.
__device__ __forceinline__ float ue8m0f(uint32_t b) {
  return b == 0u ? __uint_as_float(0x00400000u) : b == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float(b << 23);
}

// One byte's two E2M1 codes (low nibble = the earlier column) as the oracle's bf16 pair.
__device__ __forceinline__ uint32_t nv_pair(uint32_t byte, float sv, float outer) {
  const float a = __fmul_rn(__fmul_rn(e2m1f(byte & 15u), sv), outer);
  const float b = __fmul_rn(__fmul_rn(e2m1f(byte >> 4), sv), outer);
  return pack_bf16(a, b);
}

// Two E4M3 bytes (the low 16 bits, low byte first) times their block's power of two, as bf16 (exact).
__device__ __forceinline__ uint32_t mx_pair(uint32_t two, float sf) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  return pack_bf16(__fmul_rn(f.x, sf), __fmul_rn(f.y, sf));
}

// Inkling2SoftcappedSwiGLU on the bf16 gate and up (torch's ops in order: `x / c` on a CPU scalar is x * f32(1 / c)).
__device__ __forceinline__ float capped_swiglu(float g, float u) {
  const float gc = __fmul_rn(tanhf(__fmul_rn(g, 1.0f / GATE_CAP)), GATE_CAP);
  const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
  const float uc = __fmul_rn(tanhf(__fmul_rn(u, 1.0f / UP_CAP)), UP_CAP);
  return __fmul_rn(__fmul_rn(gc, sg), uc);
}

// ------------------------------------------------------------------------------------------------- router

constexpr int LG_ROWS = 4;   // rows a block
constexpr int LG_WARPS = 8;  // a warp an expert at a time
constexpr int LG_SPLIT = 24; // blocks over the experts (16 a block)

// logits [R, 384] fp32 = f32(m) . W^T, W fp32 [384, 4608]. A lane sums k = 4 * (32 i + lane) .. + 3 for i = 0 .. 35
// in order (fma), then the warp's butterfly: one order for every row, whatever the block holds.
extern "C" __global__ void __launch_bounds__(LG_WARPS * 32)
    tf_ink2_moe_logits(const __nv_bfloat16* __restrict__ m, const float* __restrict__ w, float* __restrict__ logits,
                       int R) {
  __shared__ __align__(16) __nv_bfloat16 xs[LG_ROWS][HID];
  const int r0 = blockIdx.x * LG_ROWS;
  const int nr = min(LG_ROWS, R - r0);
  for (int i = threadIdx.x; i < nr * (HID / 8); i += blockDim.x) {
    const int r = i / (HID / 8), c = i - r * (HID / 8);
    reinterpret_cast<uint4*>(xs[r])[c] = reinterpret_cast<const uint4*>(m + (size_t)(r0 + r) * HID)[c];
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int e = blockIdx.y * LG_WARPS + warp; e < NEXP; e += LG_SPLIT * LG_WARPS) {
    const float4* we = reinterpret_cast<const float4*>(w + (size_t)e * HID);
    float acc[LG_ROWS];
#pragma unroll
    for (int r = 0; r < LG_ROWS; ++r) acc[r] = 0.f;
#pragma unroll 4
    for (int i = 0; i < HID / 128; ++i) {
      const int c = i * 32 + lane;
      const float4 v = __ldg(we + c);
#pragma unroll
      for (int r = 0; r < LG_ROWS; ++r) {
        if (r < nr) {
          const uint2 xv = reinterpret_cast<const uint2*>(xs[r])[c];
          acc[r] = __fmaf_rn(lo_f(xv.x), v.x, acc[r]);
          acc[r] = __fmaf_rn(hi_f(xv.x), v.y, acc[r]);
          acc[r] = __fmaf_rn(lo_f(xv.y), v.z, acc[r]);
          acc[r] = __fmaf_rn(hi_f(xv.y), v.w, acc[r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < LG_ROWS; ++r) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[r] = __fadd_rn(acc[r], __shfl_xor_sync(0xFFFFFFFFu, acc[r], off));
      if (lane == 0 && r < nr) logits[(size_t)(r0 + r) * NEXP + e] = acc[r];
    }
  }
}

// tf_ink2_moe_logits with the router as the checkpoint's bf16 [384, 4608]: each value widened to fp32 (exact, the
// fp32 copy's value) and summed in the same order, so the same logits bits; gridDim.y blocks over the experts.
extern "C" __global__ void __launch_bounds__(LG_WARPS * 32)
    tf_ink2_moe_logits_bf16(const __nv_bfloat16* __restrict__ m, const __nv_bfloat16* __restrict__ w,
                            float* __restrict__ logits, int R) {
  __shared__ __align__(16) __nv_bfloat16 xs[LG_ROWS][HID];
  const int r0 = blockIdx.x * LG_ROWS;
  const int nr = min(LG_ROWS, R - r0);
  for (int i = threadIdx.x; i < nr * (HID / 8); i += blockDim.x) {
    const int r = i / (HID / 8), c = i - r * (HID / 8);
    reinterpret_cast<uint4*>(xs[r])[c] = reinterpret_cast<const uint4*>(m + (size_t)(r0 + r) * HID)[c];
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int e = blockIdx.y * LG_WARPS + warp; e < NEXP; e += gridDim.y * LG_WARPS) {
    const uint2* we = reinterpret_cast<const uint2*>(w + (size_t)e * HID);
    float acc[LG_ROWS];
#pragma unroll
    for (int r = 0; r < LG_ROWS; ++r) acc[r] = 0.f;
#pragma unroll 4
    for (int i = 0; i < HID / 128; ++i) {
      const int c = i * 32 + lane;
      const uint2 wv = __ldg(we + c);
      const float4 v = make_float4(lo_f(wv.x), hi_f(wv.x), lo_f(wv.y), hi_f(wv.y));
#pragma unroll
      for (int r = 0; r < LG_ROWS; ++r) {
        if (r < nr) {
          const uint2 xv = reinterpret_cast<const uint2*>(xs[r])[c];
          acc[r] = __fmaf_rn(lo_f(xv.x), v.x, acc[r]);
          acc[r] = __fmaf_rn(hi_f(xv.x), v.y, acc[r]);
          acc[r] = __fmaf_rn(lo_f(xv.y), v.z, acc[r]);
          acc[r] = __fmaf_rn(hi_f(xv.y), v.w, acc[r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < LG_ROWS; ++r) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[r] = __fadd_rn(acc[r], __shfl_xor_sync(0xFFFFFFFFu, acc[r], off));
      if (lane == 0 && r < nr) logits[(size_t)(r0 + r) * NEXP + e] = acc[r];
    }
  }
}

// A row a warp: s = sigmoid(logit); the 8 largest s + bias (ties to the lower id), in ascending id; weights
// s / (sum s + 1e-20) * 8 with the sum taken in ascending id. A NaN score is never picked (its id is -1 then).
extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_moe_select(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
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
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    if (lane == k) {
      ids[(size_t)row * TOPK + k] = pick[k];
      wts[(size_t)row * TOPK + k] = __fmul_rn(__fdiv_rn(ps[k], den), ROUTE_SCALE);
    }
  }
}

// ------------------------------------------------------------------------------------------------- grouping

constexpr int PLAN_STAGE = 4096;

// One block, a thread a local expert (E <= blockDim.x <= 1024): routes i = row * 8 + slot whose id is in
// [lo, lo + E), grouped by expert in route order (deterministic, no atomics): members[] the routes, items[] one
// (expert, first member, count <= tile) a tile, *nitems their number.
extern "C" __global__ void __launch_bounds__(1024)
    tf_ink2_moe_plan(const int* __restrict__ ids, int routes, int lo, int E, int tile, int* __restrict__ members,
                     int* __restrict__ items, int* __restrict__ nitems) {
  __shared__ int sid[PLAN_STAGE];
  __shared__ int cnt[1024];
  __shared__ int off[1024];
  __shared__ int ioff[1024];
  const int e = threadIdx.x;
  int c = 0;
  for (int base = 0; base < routes; base += PLAN_STAGE) {
    const int n = min(PLAN_STAGE, routes - base);
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) sid[j] = ids[base + j] - lo;
    __syncthreads();
    if (e < E)
      for (int j = 0; j < n; ++j) c += sid[j] == e;
  }
  cnt[e] = e < E ? c : 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    int o = 0, io = 0;
    for (int x = 0; x < E; ++x) {
      off[x] = o;
      ioff[x] = io;
      o += cnt[x];
      io += (cnt[x] + tile - 1) / tile;
    }
    *nitems = io;
  }
  __syncthreads();
  int at = e < E ? off[e] : 0;
  for (int base = 0; base < routes; base += PLAN_STAGE) {
    const int n = min(PLAN_STAGE, routes - base);
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) sid[j] = ids[base + j] - lo;
    __syncthreads();
    if (e < E)
      for (int j = 0; j < n; ++j)
        if (sid[j] == e) members[at++] = base + j;
  }
  if (e < E) {
    const int ni = (c + tile - 1) / tile;
    for (int x = 0; x < ni; ++x) {
      int* it = items + 3 * (ioff[e] + x);
      it[0] = e;
      it[1] = off[e] + x * tile;
      it[2] = min(tile, c - x * tile);
    }
  }
}

// ------------------------------------------------------------------------------------------------- experts

constexpr int NT = 4;     // n8 tiles a warp: 32 output columns
constexpr int KSTEP = 64; // k a step
constexpr int EX_WARPS = 4;

enum { NVFP4 = 0, MXFP8 = 1 };
enum { UP = 0, DOWN = 1 };

template <int FMT>
struct WStep;

template <>
struct WStep<NVFP4> {
  uint2 w[NT];
  uint32_t s[NT];
};

template <>
struct WStep<MXFP8> {
  uint4 w[NT];
  uint32_t s[NT];
};

// The lane's weight bytes and scale byte of n8 tile j (row n = 8j + gq of the warp's 32) at step ks.
template <int FMT>
__device__ __forceinline__ void load_w(WStep<FMT>& st, const uint8_t* wrow, const uint8_t* srow, size_t wtile,
                                       size_t stile, int ks, int t) {
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    if constexpr (FMT == NVFP4) {
      st.w[j] = ld_nc2(wrow + j * wtile + ks * (KSTEP / 2) + 8 * t);
      st.s[j] = ld_nc_u8(srow + j * stile + ks * (KSTEP / 16) + t);
    } else {
      st.w[j] = ld_nc4(wrow + j * wtile + ks * KSTEP + 16 * t);
      st.s[j] = ld_nc_u8(srow + j * stile + ks * (KSTEP / 32) + (t >> 1));
    }
  }
}

// The step's B fragments: b[j][s][0..1] for n8 tile j, MMA s.
template <int FMT>
__device__ __forceinline__ void decode_w(uint32_t (&b)[NT][4][2], const WStep<FMT>& st, float outer) {
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    if constexpr (FMT == NVFP4) {
      const float sv = e4m3f(st.s[j]);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint32_t word = s < 2 ? st.w[j].x : st.w[j].y;
        const int sh = (s & 1) * 16;
        b[j][s][0] = nv_pair((word >> sh) & 0xFFu, sv, outer);
        b[j][s][1] = nv_pair((word >> (sh + 8)) & 0xFFu, sv, outer);
      }
    } else {
      const float sf = ue8m0f(st.s[j]);
      const uint32_t words[4] = {st.w[j].x, st.w[j].y, st.w[j].z, st.w[j].w};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        b[j][s][0] = mx_pair(words[s], sf);
        b[j][s][1] = mx_pair(words[s] >> 16, sf);
      }
    }
  }
}

// Grouped experts: for each item (expert e, members first .. first + cnt) and each 32-column block, out rows =
// epilogue(X rows . W_e^T). X row of member i = X + (i / x_div) * K. UP: W = w13 [E, 4608, K = 4608] interleaved
// (gate, up) rows; the epilogue writes bf16(capped_swiglu(bf16 gate, bf16 up)) to act[i, 2304]. DOWN: W = w2
// [E, 4608, K = 2304]; writes bf16 to out[i, 4608]. Units (item, column block) dealt to warps grid-stride.
template <int FMT, int EPI, int RT>
__device__ __forceinline__ void experts_body(const __nv_bfloat16* __restrict__ X, int x_div, int K,
                                             const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                             const float* __restrict__ amax, float den, const int* __restrict__ items,
                                             const int* __restrict__ nitems, const int* __restrict__ members,
                                             __nv_bfloat16* __restrict__ out) {
  constexpr int N = HID;  // w13: 2 * 2304 interleaved rows; w2: 4608 rows
  constexpr int NB = N / (8 * NT);
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const size_t wrow_bytes = FMT == NVFP4 ? (size_t)K / 2 : (size_t)K;
  const size_t srow_bytes = FMT == NVFP4 ? (size_t)K / 16 : (size_t)K / 32;
  const int KS = K / KSTEP;
  const int units = __ldg(nitems) * NB;
  const int warps = gridDim.x * EX_WARPS;
  for (int unit = blockIdx.x * EX_WARPS + (threadIdx.x >> 5); unit < units; unit += warps) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    const size_t n0 = (size_t)e * N + (size_t)cb * (8 * NT) + gq;
    const uint8_t* wrow = W + n0 * wrow_bytes;
    const uint8_t* srow = S + n0 * srow_bytes;
    const float outer = FMT == NVFP4 ? __fdiv_rn(__ldg(amax + e), den) : 1.0f;
    const __nv_bfloat16* xr[RT][2];
    int mem[RT][2];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = rt * 16 + gq + 8 * h;
        mem[rt][h] = r < cnt ? __ldg(members + first + r) : -1;
        xr[rt][h] = r < cnt ? X + (size_t)(mem[rt][h] / x_div) * K + 16 * t : nullptr;
      }
    float acc[RT][NT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
      for (int j = 0; j < NT; ++j) acc[rt][j][0] = acc[rt][j][1] = acc[rt][j][2] = acc[rt][j][3] = 0.f;
    WStep<FMT> cur, nxt;
    load_w<FMT>(cur, wrow, srow, 8 * wrow_bytes, 8 * srow_bytes, 0, t);
    for (int ks = 0; ks < KS; ++ks) {
      if (ks + 1 < KS) load_w<FMT>(nxt, wrow, srow, 8 * wrow_bytes, 8 * srow_bytes, ks + 1, t);
      uint32_t b[NT][4][2];
      decode_w<FMT>(b, cur, outer);
#pragma unroll
      for (int rt = 0; rt < RT; ++rt) {
        if (rt * 16 >= cnt) break;
        uint4 x0[2], x1[2];
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          x0[q] = xr[rt][0] ? __ldg(reinterpret_cast<const uint4*>(xr[rt][0] + ks * KSTEP) + q) : z;
          x1[q] = xr[rt][1] ? __ldg(reinterpret_cast<const uint4*>(xr[rt][1] + ks * KSTEP) + q) : z;
        }
        // the step's 64 k into a fresh fragment, then one IEEE add: the tensor core's own accumulation (aligned
        // to the largest term and truncated) then spans 4 MMAs instead of the whole K
        float p[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) p[j][0] = p[j][1] = p[j][2] = p[j][3] = 0.f;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint4& p0 = x0[s >> 1];
          const uint4& p1 = x1[s >> 1];
          const uint32_t a0 = (s & 1) ? p0.z : p0.x, a2 = (s & 1) ? p0.w : p0.y;
          const uint32_t a1 = (s & 1) ? p1.z : p1.x, a3 = (s & 1) ? p1.w : p1.y;
#pragma unroll
          for (int j = 0; j < NT; ++j) mma_bf16(p[j], a0, a1, a2, a3, b[j][s][0], b[j][s][1]);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[rt][j][q] = __fadd_rn(acc[rt][j][q], p[j][q]);
      }
      if (ks + 1 < KS) cur = nxt;
    }
    // epilogue: c0, c1 = row gq, columns 2t, 2t + 1 of tile j; c2, c3 = row gq + 8
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mi = mem[rt][h];
        if (mi < 0) continue;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int col = cb * (8 * NT) + 8 * j + 2 * t;
          const float c0 = acc[rt][j][2 * h], c1 = acc[rt][j][2 * h + 1];
          if constexpr (EPI == UP) {
            const float a = capped_swiglu(bf16r(c0), bf16r(c1));
            out[(size_t)mi * INTER + (col >> 1)] = __float2bfloat16_rn(a);
          } else {
            *reinterpret_cast<uint32_t*>(out + (size_t)mi * HID + col) = pack_bf16(c0, c1);
          }
        }
      }
  }
}

}  // namespace

#define TF_INK2_EXPERTS(NAME, FMT, EPI, RT)                                                                         \
  extern "C" __global__ void __launch_bounds__(EX_WARPS * 32)                                                       \
      NAME(const __nv_bfloat16* __restrict__ X, int x_div, int K, const uint8_t* __restrict__ W,                    \
           const uint8_t* __restrict__ S, const float* __restrict__ amax, float den, const int* __restrict__ items, \
           const int* __restrict__ nitems, const int* __restrict__ members, __nv_bfloat16* __restrict__ out) {      \
    experts_body<FMT, EPI, RT>(X, x_div, K, W, S, amax, den, items, nitems, members, out);                          \
  }

TF_INK2_EXPERTS(tf_ink2_moe_up_nvfp4_r1, NVFP4, UP, 1)
TF_INK2_EXPERTS(tf_ink2_moe_up_nvfp4_r4, NVFP4, UP, 4)
TF_INK2_EXPERTS(tf_ink2_moe_down_nvfp4_r1, NVFP4, DOWN, 1)
TF_INK2_EXPERTS(tf_ink2_moe_down_nvfp4_r4, NVFP4, DOWN, 4)
TF_INK2_EXPERTS(tf_ink2_moe_up_mxfp8_r1, MXFP8, UP, 1)
TF_INK2_EXPERTS(tf_ink2_moe_up_mxfp8_r4, MXFP8, UP, 4)
TF_INK2_EXPERTS(tf_ink2_moe_down_mxfp8_r1, MXFP8, DOWN, 1)
TF_INK2_EXPERTS(tf_ink2_moe_down_mxfp8_r4, MXFP8, DOWN, 4)

// out [R, 4608] bf16 = bf16(sum over the row's routes in ascending expert id of f32(d[row * 8 + slot]) * w[slot]),
// fp32 from 0, each product and sum rounded once (Transformers' index_add_ loop over experts). Routes whose id is
// outside [lo, hi) are not this rank's and are skipped. The _f32 form keeps the fp32 sum (a rank's partial at
// W > 1, summed over the ranks in rank order and rounded to bf16 once after the exchange).
namespace {
template <bool F32>
__device__ __forceinline__ void combine_body(const __nv_bfloat16* __restrict__ d, const int* __restrict__ ids,
                                             const float* __restrict__ wts, int lo, int hi, void* __restrict__ out) {
  const int row = blockIdx.x;
  int id[TOPK], slot[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) id[k] = __ldg(ids + (size_t)row * TOPK + k), slot[k] = k;
#pragma unroll
  for (int i = 1; i < TOPK; ++i)
#pragma unroll
    for (int j = i; j > 0; --j)
      if (id[j] < id[j - 1]) {
        const int a = id[j], b = slot[j];
        id[j] = id[j - 1], slot[j] = slot[j - 1];
        id[j - 1] = a, slot[j - 1] = b;
      }
  float w[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) w[k] = __ldg(wts + (size_t)row * TOPK + slot[k]);
  for (int c = threadIdx.x; c < HID / 2; c += blockDim.x) {
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if (id[k] < lo || id[k] >= hi) continue;
      const uint32_t v = __ldg(reinterpret_cast<const unsigned int*>(d + ((size_t)row * TOPK + slot[k]) * HID) + c);
      a0 = __fadd_rn(a0, __fmul_rn(lo_f(v), w[k]));
      a1 = __fadd_rn(a1, __fmul_rn(hi_f(v), w[k]));
    }
    if constexpr (F32) {
      reinterpret_cast<float2*>(static_cast<float*>(out) + (size_t)row * HID)[c] = make_float2(a0, a1);
    } else {
      reinterpret_cast<uint32_t*>(static_cast<__nv_bfloat16*>(out) + (size_t)row * HID)[c] = pack_bf16(a0, a1);
    }
  }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_moe_combine(const __nv_bfloat16* __restrict__ d, const int* __restrict__ ids,
                        const float* __restrict__ wts, int lo, int hi, __nv_bfloat16* __restrict__ out) {
  combine_body<false>(d, ids, wts, lo, hi, out);
}

extern "C" __global__ void __launch_bounds__(256)
    tf_ink2_moe_combine_f32(const __nv_bfloat16* __restrict__ d, const int* __restrict__ ids,
                            const float* __restrict__ wts, int lo, int hi, float* __restrict__ out) {
  combine_body<true>(d, ids, wts, lo, hi, out);
}

// The module's presence check (cuda_kernels.zig Group): writes the group's number.
extern "C" __global__ void tf_ink2_moe_probe(int* out) {
  if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 4;
}
