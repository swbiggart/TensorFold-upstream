// Inkling-2's vision tower attention core (L4 of the tower; FABLE-IMAGE-INPUT-DESIGN 1.3): one image, 16 heads of
// 64, no mask, no bias, sink, gate or window; keys = all S rows of the image. Adapted from inkling2_kx_fp32.cuh's tile
// arithmetic (mma.sync m16n8k16 bf16 -> fp32, P in bf16 limbs) for D = 64, with the scores kept in registers.
//
// The arithmetic of a (row, head), the same for every launch:
//   scores s = fp32(q . k) on mma.sync m16n8k16 (bf16 products exact, fp32 accumulate, dims 0-15, 16-31, 32-47,
//   48-63 in that order) times 0.125 (exact); 64-key tiles at absolute keys 64t .. 64t + 63, t = 0 .. ceil(S / 64) - 1
//   in that order for every row (keys >= S: score -inf, value +0); per tile: m' = max(m, the tile's max) (a thread's
//   16 columns in order, then xor 1, 2 over the quad), a = m == m' ? 1 : expf(m - m'), p = expf(s - m') in fp32,
//   Z = Z * a + (the thread's 16 p in order, then xor 1, 2), O = O * a + T (product, then sum, each rounded) where T
//   is the tile's P.V accumulated from +0 on the same mma (keys 0-15, 16-31, 32-47, 48-63 of the tile, limbs 0, 1, 2
//   within each) with P as LIMBS bf16 limbs of p (limb i = bf16_rn(p - limbs 0 .. i-1)); out = bf16_rn(O / Z),
//   rounded once. (T per tile, not one mma accumulator over all keys: the tensor cores' fp32 accumulation drifts over
//   long key ranges; per tile the output equals bf16(fp64) to 1e-6 relative even at S = 42,437, for ~1% time.)
// A row's bits depend on its own q and on k, v only: never on q0, qn, the grid, the CTA's other rows or the slot it
// takes in an MMA. Deterministic: no atomics, no data-dependent order. Non-finite inputs are not screened (the tower's
// q, k, v come from LayerNorm, a projection and the rotation); a NaN or inf reaches the row's output. Subnormal
// inputs are used as the tensor cores treat them (the same bits every run).
//
// Build with the glue flags (-O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true -std=c++20
// --expt-relaxed-constexpr); needs sm_80 or later (mma.sync bf16, ldmatrix, cp.async).
//
// Launch (host side; the constants below are host-callable):
//   tf_iv_attn<<<dim3(iv::blocks(qn), iv::kHeads, 1), iv::kThreads, 0 /* static shared, iv::kSmemBytes */>>>(
//       q, k, v, ld, out, S, q0, qn);
//   q, k, v: element (row r, head h, dim d) at base + r * ld + h * 64 + d (ld = 3072 and bases +0 / +1024 / +2048 for
//   one fused [S][3072] qkv buffer); ld % 8 == 0 and 16-byte aligned bases. out: row r at out + (r - q0) * 1024 +
//   h * 64 + d (dense [qn][1024]). 0 <= q0, 1 <= qn, q0 + qn <= S. No dynamic shared memory, no scratch.
// Include this header in one .cu only (it defines the extern "C" kernel). iv::attn<2> is the 2-limb variant (same
// accuracy on every measured case, ~16% faster at S = 42,437, ~0.1% of output bits differ); not instantiated here.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace iv {

constexpr int kHeads = 16;      // heads (query and key/value)
constexpr int kDim = 64;        // head dim
constexpr int kKeys = 64;       // keys a tile (absolute positions 64t .. 64t + 63)
constexpr int kWarps = 4;       // warps a CTA, 16 query rows each
constexpr int kRows = 16 * kWarps;     // query rows a CTA
constexpr int kThreads = 32 * kWarps;  // threads a CTA
constexpr int kPad = kDim + 8;         // shared row (144 bytes: ldmatrix rows on distinct banks)
constexpr int kStages = 2;             // key/value tiles in flight
constexpr int kSmemBytes = kStages * 2 * kKeys * kPad * 2;  // static shared memory a CTA (36,864 bytes; sm_12x
                                                             // reports 37,888 with the system's 1 KiB)
constexpr float kScale = 0.125f;       // 64^-0.5, exact
constexpr int kLimbs = 3;              // P's bf16 limbs in tf_iv_attn

// Grid x for qn query rows (grid y = kHeads, z = 1).
__host__ __device__ constexpr unsigned blocks(int qn) { return (unsigned)((qn + kRows - 1) / kRows); }

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const __nv_bfloat16* p) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const __nv_bfloat16* p) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
// 16 bytes global -> shared; bytes = 0 fills +0 (a key past S).
__device__ __forceinline__ void cp16(void* dst, const void* src, int bytes) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(at), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// Two fp32 as bf16 (round to nearest even): the low half holds a.
__device__ __forceinline__ uint32_t pack(float a, float c) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(a, c);
    return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ __forceinline__ float lo(uint32_t x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float hi(uint32_t x) { return __uint_as_float(x & 0xffff0000u); }

struct Tile {
    __align__(16) __nv_bfloat16 k[kKeys][kPad];
    __align__(16) __nv_bfloat16 v[kKeys][kPad];
};

// Keys base .. base + 63 of head h into t (cp.async; keys >= S as +0).
__device__ __forceinline__ void load_tile(Tile& t, const __nv_bfloat16* k, const __nv_bfloat16* v, int ld, int h,
                                          int S, int base) {
    // 64 keys x 8 chunks of 16 bytes, for k and for v
#pragma unroll
    for (int i = threadIdx.x; i < kKeys * 8; i += kThreads) {
        const int j = i >> 3, c = (i & 7) * 8, key = base + j;
        const bool live = key < S;
        const size_t at = (size_t)(live ? key : 0) * ld + h * kDim + c;
        cp16(&t.k[j][c], k + at, live ? 16 : 0);
        cp16(&t.v[j][c], v + at, live ? 16 : 0);
    }
}

// One CTA: query rows q0 + 64 * blockIdx.x + 16 * warp + 0 .. 15 of head blockIdx.y against keys 0 .. S - 1.
template <int LIMBS>
__device__ __forceinline__ void attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v, int ld, __nv_bfloat16* __restrict__ out,
                                     int S, int q0, int qn) {
    static_assert(LIMBS >= 1 && LIMBS <= 3, "P limbs");
    __shared__ Tile tiles[kStages];
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, sub = lane & 3;
    const int h = blockIdx.y;
    const int r0 = blockIdx.x * kRows + w * 16;  // this warp's first row, from q0
    const int ntiles = (S + kKeys - 1) / kKeys;
    load_tile(tiles[0], k, v, ld, h, S, 0);
    cp_commit();

    // q's A fragments: rows g and g + 8 of the warp (a row past the block repeats the block's last row; not stored)
    uint32_t qa[4][4];
    {
        const int ra = q0 + min(r0 + g, qn - 1), rb = q0 + min(r0 + g + 8, qn - 1);
        const __nv_bfloat16* pa = q + (size_t)ra * ld + h * kDim + 2 * sub;
        const __nv_bfloat16* pb = q + (size_t)rb * ld + h * kDim + 2 * sub;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            qa[s][0] = *reinterpret_cast<const uint32_t*>(pa + s * 16);
            qa[s][1] = *reinterpret_cast<const uint32_t*>(pb + s * 16);
            qa[s][2] = *reinterpret_cast<const uint32_t*>(pa + s * 16 + 8);
            qa[s][3] = *reinterpret_cast<const uint32_t*>(pb + s * 16 + 8);
        }
    }
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[n][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, z[2] = {0.f, 0.f};  // rows g, g + 8
    float ot[8][4];  // this tile's P.V

    for (int ti = 0; ti < ntiles; ++ti) {
        if (ti + 1 < ntiles) {
            load_tile(tiles[(ti + 1) & 1], k, v, ld, h, S, (ti + 1) * kKeys);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const Tile& tl = tiles[ti & 1];
        const int base = ti * kKeys;

        // scores: keys 8n .. 8n + 7 of the tile (thread: columns 2 sub, 2 sub + 1), dims in steps of 16 ascending
        float sc[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) sc[n][e] = 0.f;
            uint32_t kb[2][4];
            ldsm_x4(kb[0], &tl.k[n * 8 + (lane & 7)][(lane >> 3) * 8]);
            ldsm_x4(kb[1], &tl.k[n * 8 + (lane & 7)][32 + (lane >> 3) * 8]);
            mma(sc[n], qa[0], kb[0][0], kb[0][1]);
            mma(sc[n], qa[1], kb[0][2], kb[0][3]);
            mma(sc[n], qa[2], kb[1][0], kb[1][1]);
            mma(sc[n], qa[3], kb[1][2], kb[1][3]);
        }
        const bool tail = base + kKeys > S;
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float s = sc[n][e] * kScale;
                if (tail && base + n * 8 + 2 * sub + (e & 1) >= S) s = -INFINITY;
                sc[n][e] = s;
            }
        // the tile's max and the rescale (row g: e = 0, 1; row g + 8: e = 2, 3)
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = m[r];
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                mx = fmaxf(mx, sc[n][2 * r]);
                mx = fmaxf(mx, sc[n][2 * r + 1]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            alpha[r] = m[r] == mx ? 1.f : expf(m[r] - mx);
            m[r] = mx;
        }
        // the weights and their sums
        float zt[2] = {0.f, 0.f};
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float p = expf(sc[n][e] - m[e >> 1]);
                sc[n][e] = p;
            }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                zt[r] = __fadd_rn(zt[r], sc[n][2 * r]);
                zt[r] = __fadd_rn(zt[r], sc[n][2 * r + 1]);
            }
            zt[r] = __fadd_rn(zt[r], __shfl_xor_sync(0xffffffffu, zt[r], 1));
            zt[r] = __fadd_rn(zt[r], __shfl_xor_sync(0xffffffffu, zt[r], 2));
            z[r] = __fadd_rn(__fmul_rn(z[r], alpha[r]), zt[r]);
        }
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) ot[n][e] = 0.f;

        // P.V: keys 16s .. 16s + 15 ascending, limbs 0 .. LIMBS - 1 within each
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            // P's A fragment (rows g, g + 8; keys 16s + 2 sub, +1 and 16s + 8 + 2 sub, +1) as limbs
            float pr[4][2] = {{sc[2 * s][0], sc[2 * s][1]},
                              {sc[2 * s][2], sc[2 * s][3]},
                              {sc[2 * s + 1][0], sc[2 * s + 1][1]},
                              {sc[2 * s + 1][2], sc[2 * s + 1][3]}};
            uint32_t pa[LIMBS][4];
#pragma unroll
            for (int l = 0; l < LIMBS; ++l)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t x = pack(pr[i][0], pr[i][1]);
                    pa[l][i] = x;
                    if (l + 1 < LIMBS) {
                        pr[i][0] = __fsub_rn(pr[i][0], lo(x));
                        pr[i][1] = __fsub_rn(pr[i][1], hi(x));
                    }
                }
            // V's B fragments: dims 16j .. 16j + 15 (two n-tiles) by ldmatrix .trans of the key-major tile
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t vb[4];
                ldsm_x4_t(vb, &tl.v[s * 16 + (lane & 15)][j * 16 + (lane >> 4) * 8]);
#pragma unroll
                for (int l = 0; l < LIMBS; ++l) {
                    mma(ot[2 * j], pa[l], vb[0], vb[1]);
                    mma(ot[2 * j + 1], pa[l], vb[2], vb[3]);
                }
            }
        }
        // O = O * a + T
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[n][e] = __fadd_rn(__fmul_rn(o[n][e], alpha[e >> 1]), ot[n][e]);
        __syncthreads();  // the next iteration's load overwrites this tile's stage
    }

    // out = bf16(O / Z), rows g and g + 8, columns 8n + 2 sub, +1
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = r0 + g + 8 * r;
        if (row >= qn) continue;
        __nv_bfloat16* po = out + (size_t)row * (kHeads * kDim) + h * kDim + 2 * sub;
#pragma unroll
        for (int n = 0; n < 8; ++n)
            *reinterpret_cast<uint32_t*>(po + n * 8) = pack(__fdiv_rn(o[n][2 * r], z[r]), __fdiv_rn(o[n][2 * r + 1], z[r]));
    }
}

}  // namespace iv

// The tower's core: 3-limb P (iv::kLimbs). Grid (iv::blocks(qn), iv::kHeads, 1), iv::kThreads threads, static shared.
extern "C" __global__ void __launch_bounds__(iv::kThreads) tf_iv_attn(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                                                      const __nv_bfloat16* v, int ld, __nv_bfloat16* out,
                                                                      int S, int q0, int qn) {
    iv::attn<iv::kLimbs>(q, k, v, ld, out, S, q0, qn);
}
