// Inkling-2's vision tower (families/inkling2/cuda_vision.zig): every op of FABLE-IMAGE-INPUT-DESIGN 1.3 but the
// projections (the lane matmul, cuda/qmmf.zig) and the tensor-core attention core (inkling2_vision_attn.cuh).
// Built with the glue flags: no contraction, IEEE division and square root, so each op keeps torch's rounding
// points with plain __f*_rn arithmetic. Rows are the image's tower rows: row 0 the class token, then its P = 4T
// patches in block order (tokens in raster order, inside a token (0,0), (0,1), (1,0), (1,1)).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "common/glue_math.cuh"
#include "vit_norm.cuh"

#include "inkling2_vision_attn.cuh"  // tf_iv_attn, the tensor-core core (--core tc, the default)

using tf_glue::bf;
using tf_glue::rbf;

namespace {

constexpr int NT = 256;
constexpr int W = 1024;     // hidden width
constexpr int HEADS = 16;
constexpr int HD = 64;      // head dim
constexpr int BLOCK = 40;   // pixels a token side
constexpr int PATCH = 20;   // pixels a patch side
constexpr int FEAT = 1200;  // a patch's features (py, px, c)
constexpr int FEAT_PAD = 1216;

// A patch's (row, col) on the 2*rows x 2*cols patch grid from its block-order index.
__device__ __forceinline__ void patch_at(int p, int cols, int& row, int& col) {
    const int t = p >> 2, in = p & 3;
    row = 2 * (t / cols) + (in >> 1);
    col = 2 * (t % cols) + (in & 1);
}

// erff as torch 2.14.1+cu130's GELU kernel runs it (CUDA 13.0's erff, read from its sm_120 SASS: it differs from
// CUDA 13.3's in 7 of the 65,536 bf16 GELU outputs). |a| >= 0x3f8060fe: 1 - 2^(-|a| p(|a|) - |a|) with a's sign;
// else a p(a^2) + a. p is a degree-6 polynomial by fma, its coefficients per branch; 2^x is MUFU.EX2 (ex2.approx.ftz).
__device__ __forceinline__ float torch_erff(float a) {
    const float t = fabsf(a);
    const bool big = t >= __int_as_float(0x3f8060fe);
    const float s = big ? t : __fmul_rn(a, a);
    const unsigned c[2][7] = {{0x38b1e96au, 0xba574d20u, 0x3baad5eau, 0xbcdc1be7u, 0x3de718afu, 0xbec093acu, 0x3e0375d3u},
                              {0x38eb4c3au, 0xbaae005bu, 0x3c09919fu, 0xbd24d99au, 0x3e235519u, 0x3f69b4f9u, 0x3f210a14u}};
    const unsigned* k = c[big ? 1 : 0];
    float p = __fmaf_rn(s, __uint_as_float(k[0]), __uint_as_float(k[1]));
#pragma unroll
    for (int i = 2; i < 7; ++i) p = __fmaf_rn(s, p, __uint_as_float(k[i]));
    const float m = big ? -t : a;
    const float r = __fmaf_rn(p, m, m);
    if (!big) return r;
    float e;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(r));
    return copysignf(__fsub_rn(1.0f, e), a);
}

// torch's F.gelu (approximate='none') on CUDA in fp32: x * 0.5 * (1 + erf(x * M_SQRT1_2)).
__device__ __forceinline__ float gelu_erf(float x) {
    constexpr float alpha = M_SQRT1_2;
    return __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.0f, torch_erff(__fmul_rn(x, alpha))));
}

// x = bf16(x + bf16(ls * y)), y already bf16: the layer scale's product and the residual add, two roundings.
__device__ __forceinline__ __nv_bfloat16 ls_res(float y, __nv_bfloat16 ls, __nv_bfloat16 x) {
    const float t = rbf(__fmul_rn(bf(ls), y));
    return __float2bfloat16_rn(__fadd_rn(bf(x), t));
}

}  // namespace

// E1. Block a patch row of [n][1216]: patch p0 + blockIdx.x's 1200 features (py, px, c) = bf16(table[byte]) or
// bf16(table[256]) past the image (padding right and bottom), then 16 zeros. rgb [H][W][3].
extern "C" __global__ void __launch_bounds__(NT) tf_iv_patchify(const uint8_t* __restrict__ rgb, const float* __restrict__ table,
                                                                int H, int Wd, int cols, int p0, __nv_bfloat16* __restrict__ out) {
    const int p = p0 + blockIdx.x;
    int prow, pcol;
    patch_at(p, cols, prow, pcol);
    const int y0 = prow * PATCH, x0 = pcol * PATCH;
    __nv_bfloat16* o = out + static_cast<int64_t>(blockIdx.x) * FEAT_PAD;
    for (int f = threadIdx.x; f < FEAT_PAD; f += NT) {
        float v = 0.0f;
        if (f < FEAT) {
            const int py = f / (3 * PATCH), px = (f / 3) % PATCH, c = f % 3;
            const int y = y0 + py, x = x0 + px;
            v = (y < H && x < Wd) ? table[rgb[(static_cast<int64_t>(y) * Wd + x) * 3 + c]] : table[256];
        }
        o[f] = __float2bfloat16_rn(v);
    }
}

// E3 + E4. Block a patch p0 + blockIdx.x: pos = bf16((((0 + t0) + t1) + t2) + t3), t_i = float(table[idx_i]) * w_i in
// fp32; x[1 + p] = bf16(float(e2[p - p0]) + float(pos)). idx [P][4], w [P][4] from the host (vision_utils' order).
extern "C" __global__ void __launch_bounds__(NT) tf_iv_embed(const __nv_bfloat16* __restrict__ e2, const __nv_bfloat16* __restrict__ table,
                                                             const int* __restrict__ idx, const float* __restrict__ wts, int p0,
                                                             __nv_bfloat16* __restrict__ x) {
    const int p = p0 + blockIdx.x;
    const int* id = idx + static_cast<int64_t>(p) * 4;
    const float* w = wts + static_cast<int64_t>(p) * 4;
    const __nv_bfloat16* e = e2 + static_cast<int64_t>(blockIdx.x) * W;
    __nv_bfloat16* o = x + static_cast<int64_t>(1 + p) * W;
    for (int c = threadIdx.x; c < W; c += NT) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) s = __fadd_rn(s, __fmul_rn(bf(table[static_cast<int64_t>(id[i]) * W + c]), w[i]));
        o[c] = __float2bfloat16_rn(__fadd_rn(bf(e[c]), rbf(s)));
    }
}

// E4's class row: x[0] = bf16(class_embedding + cls_position_embedding).
extern "C" __global__ void __launch_bounds__(NT) tf_iv_class(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                                                             __nv_bfloat16* __restrict__ x) {
    for (int c = threadIdx.x; c < W; c += NT) x[c] = __float2bfloat16_rn(__fadd_rn(bf(a[c]), bf(b[c])));
}

// E5, L1, L7: LayerNorm(1024) rows (vit_norm.cuh, fixed fp32 order), block a row. X and Y must not overlap.
extern "C" __global__ void __launch_bounds__(NT) tf_iv_layer_norm(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ Wt,
                                                                  const __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ Y, float eps) {
    tf_vit_norm::layer_norm<W / NT>(X, Wt, B, Y, W, eps, tf_vit_norm::ALL);
}

// R: the rotary tables of rows r0 + blockIdx.x: pair j < 16 turns by the column angle at frequency j, j >= 16 by the
// row angle at frequency j - 16; angle = float(pos) * inv[j % 16]; cos, sin [S][32] fp32. Position (row + 1, col + 1)
// for a patch, (0, 0) for the class row.
extern "C" __global__ void __launch_bounds__(32) tf_iv_rope_table(const float* __restrict__ inv, int cols, int r0,
                                                                  float* __restrict__ cs, float* __restrict__ sn) {
    const int r = r0 + blockIdx.x, j = threadIdx.x;
    int prow = -1, pcol = -1;
    if (r > 0) patch_at(r - 1, cols, prow, pcol);
    const float pos = static_cast<float>(j < 16 ? pcol + 1 : prow + 1);
    const float a = __fmul_rn(pos, inv[j & 15]);
    cs[static_cast<int64_t>(r) * 32 + j] = cosf(a);
    sn[static_cast<int64_t>(r) * 32 + j] = sinf(a);
}

// L3: q and k of rows r0 + blockIdx.x in place (row r at qkv + r * ld, q at +0, k at +1024): per head, pair (a, b) at
// (2i, 2i + 1) -> (bf16(a cos - b sin), bf16(b cos + a sin)), each product and the sum rounded in fp32 (torch's
// q * cos + rotate_pairwise(q) * sin, rotate (a, b) = (-b, a)).
extern "C" __global__ void __launch_bounds__(NT) tf_iv_rotary(__nv_bfloat16* qkv, int ld, const float* __restrict__ cs,
                                                              const float* __restrict__ sn, int r0) {
    const int r = r0 + blockIdx.x;
    __nv_bfloat16* row = qkv + static_cast<int64_t>(r) * ld;
    for (int e = threadIdx.x; e < 2 * HEADS * 32; e += NT) {
        const int part = e / (HEADS * 32), h = (e / 32) % HEADS, i = e % 32;
        __nv_bfloat16* at = row + part * W + h * HD + 2 * i;
        const float c = cs[static_cast<int64_t>(r) * 32 + i], s = sn[static_cast<int64_t>(r) * 32 + i];
        const float a = bf(at[0]), b = bf(at[1]);
        at[0] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(a, c), __fmul_rn(-b, s)));
        at[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s)));
    }
}

// Block (row, column block): O[row * ldo + c] = bf16(ACC[row * N + c] + bias[c]) (a Linear's bias, torch's one rounding).
extern "C" __global__ void __launch_bounds__(NT) tf_iv_bias(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                            __nv_bfloat16* __restrict__ O, int N, int ldo) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[static_cast<int64_t>(blockIdx.x) * ldo + c] = __float2bfloat16_rn(__fadd_rn(ACC[at], bf(BIAS[c])));
}

// L8 fc1's end: O = bf16(gelu(bf16(ACC + bias))), [rows][N].
extern "C" __global__ void __launch_bounds__(NT) tf_iv_bias_gelu(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                 __nv_bfloat16* __restrict__ O, int N) {
    const int c = blockIdx.y * NT + threadIdx.x;
    if (c >= N) return;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * N + c;
    O[at] = __float2bfloat16_rn(gelu_erf(rbf(__fadd_rn(ACC[at], bf(BIAS[c])))));
}

// L5 + L6, L8 fc2 + L9: y = bf16(ACC + bias); X = bf16(X + bf16(ls * y)) in place, [rows][1024].
extern "C" __global__ void __launch_bounds__(NT) tf_iv_bias_ls_res(const float* __restrict__ ACC, const __nv_bfloat16* __restrict__ BIAS,
                                                                   const __nv_bfloat16* __restrict__ LS, __nv_bfloat16* __restrict__ X) {
    const int c = blockIdx.y * NT + threadIdx.x;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * W + c;
    X[at] = ls_res(rbf(__fadd_rn(ACC[at], bf(BIAS[c]))), LS[c], X[at]);
}

// The same residual end from a bf16 y (the op check: torch's projection output as y).
extern "C" __global__ void __launch_bounds__(NT) tf_iv_ls_res(const __nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ LS,
                                                              __nv_bfloat16* __restrict__ X) {
    const int c = blockIdx.y * NT + threadIdx.x;
    const int64_t at = static_cast<int64_t>(blockIdx.x) * W + c;
    X[at] = ls_res(bf(Y[at]), LS[c], X[at]);
}

// M2's GELU (and the op check's GELU table): O = bf16(gelu(float(X))), count elements; X may be O.
extern "C" __global__ void __launch_bounds__(NT) tf_iv_gelu(const __nv_bfloat16* X, __nv_bfloat16* O, int64_t count) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * NT + threadIdx.x;
    if (at < count) O[at] = __float2bfloat16_rn(gelu_erf(bf(X[at])));
}

// Block a listed row: out[i] = x[rows[i]] (1024 wide): the sampled layer rows.
extern "C" __global__ void __launch_bounds__(NT) tf_iv_gather(const __nv_bfloat16* __restrict__ x, const int* __restrict__ rows,
                                                              __nv_bfloat16* __restrict__ out) {
    const __nv_bfloat16* src = x + static_cast<int64_t>(rows[blockIdx.x]) * W;
    for (int c = threadIdx.x; c < W; c += NT) out[static_cast<int64_t>(blockIdx.x) * W + c] = src[c];
}

// L4, the reference core: one thread a (query row, head); grid (ceil(qn / 128), 16), 128 threads. Scores
// s = fp32 dot(q, k) (fma chain over d = 0 .. 63) * 0.125, keys 0 .. S - 1 in order; two passes: m = max s, then
// p = expf(s - m), Z = sum p, O = sum p v (fma, keys in order); out = bf16(O / Z). q, k, v (row r, head h, d) at
// base + r * ld + h * 64 + d; out row r at out + (r - q0) * 1024 + h * 64 + d (tf_iv_attn's arguments).
namespace {
constexpr int REF_ROWS = 128;
constexpr int REF_KEYS = 32;
}  // namespace

extern "C" __global__ void __launch_bounds__(REF_ROWS) tf_iv_attn_ref(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                                      const __nv_bfloat16* __restrict__ v, int ld, __nv_bfloat16* __restrict__ out,
                                                                      int S, int q0, int qn) {
    __shared__ float4 ks[REF_KEYS][HD / 4];
    __shared__ float4 vs[REF_KEYS][HD / 4];
    const int h = blockIdx.y;
    const int i = blockIdx.x * REF_ROWS + threadIdx.x;
    const bool live = i < qn;
    const int r = q0 + (live ? i : 0);
    float qr[HD];
#pragma unroll
    for (int d = 0; d < HD; ++d) qr[d] = bf(q[static_cast<int64_t>(r) * ld + h * HD + d]);
    auto load = [&](const __nv_bfloat16* src, float4 (*dst)[HD / 4], int k0) {
        for (int e = threadIdx.x; e < REF_KEYS * HD; e += REF_ROWS) {
            const int kk = e / HD, d = e % HD;
            const int key = k0 + kk;
            reinterpret_cast<float*>(dst[kk])[d] = key < S ? bf(src[static_cast<int64_t>(key) * ld + h * HD + d]) : 0.0f;
        }
    };
    auto score = [&](int kk) {
        float s = 0.0f;
#pragma unroll
        for (int d4 = 0; d4 < HD / 4; ++d4) {
            const float4 kv = ks[kk][d4];
            s = __fmaf_rn(qr[4 * d4 + 0], kv.x, s);
            s = __fmaf_rn(qr[4 * d4 + 1], kv.y, s);
            s = __fmaf_rn(qr[4 * d4 + 2], kv.z, s);
            s = __fmaf_rn(qr[4 * d4 + 3], kv.w, s);
        }
        return __fmul_rn(s, 0.125f);
    };
    float m = -INFINITY;
    for (int k0 = 0; k0 < S; k0 += REF_KEYS) {
        __syncthreads();
        load(k, ks, k0);
        __syncthreads();
        const int nk = min(REF_KEYS, S - k0);
        for (int kk = 0; kk < nk; ++kk) m = fmaxf(m, score(kk));
    }
    float acc[HD];
#pragma unroll
    for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
    float z = 0.0f;
    for (int k0 = 0; k0 < S; k0 += REF_KEYS) {
        __syncthreads();
        load(k, ks, k0);
        load(v, vs, k0);
        __syncthreads();
        const int nk = min(REF_KEYS, S - k0);
        for (int kk = 0; kk < nk; ++kk) {
            const float p = expf(__fsub_rn(score(kk), m));
            z = __fadd_rn(z, p);
#pragma unroll
            for (int d4 = 0; d4 < HD / 4; ++d4) {
                const float4 vv = vs[kk][d4];
                acc[4 * d4 + 0] = __fmaf_rn(p, vv.x, acc[4 * d4 + 0]);
                acc[4 * d4 + 1] = __fmaf_rn(p, vv.y, acc[4 * d4 + 1]);
                acc[4 * d4 + 2] = __fmaf_rn(p, vv.z, acc[4 * d4 + 2]);
                acc[4 * d4 + 3] = __fmaf_rn(p, vv.w, acc[4 * d4 + 3]);
            }
        }
    }
    if (!live) return;
    __nv_bfloat16* o = out + static_cast<int64_t>(i) * W + h * HD;
#pragma unroll
    for (int d = 0; d < HD; ++d) o[d] = __float2bfloat16_rn(__fdiv_rn(acc[d], z));
}
