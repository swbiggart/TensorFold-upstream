// Copyright kortexa-ai. Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
// From kortexa-ai/TensorFold e2614fe (zig/kernels/cuda/inkling2_attention.cuh, inkling2_attention_decode.cuh,
// inkling2_heads.cuh); modified: one tile body for every form (ring and paged keys, the position read on the device,
// scalar arguments), out-of-range keys loaded as +0, subnormal q/K/V and P limbs flushed per element, non-finite
// keys and values recomputed per row (never per block), NaN outputs stored as 0x7FC0, exception counters, column
// splits, absolute S-key splits with an ordered merge, and the relative bias in one launch.
//
// Inkling-2's fp32 attention core (cuda_attn.zig `kx`, the default form; TF_INK2_ATTN=v01 keeps inkling2_attn.cu's).
// The arithmetic of a (row, query head), the same in every form:
//   q'' bf16 fragment (subnormal -> +0, non-finite -> +0 and the row recomputed); 64-key tiles at absolute
//   positions, keys outside the block's range [k_first, k_last] loaded as +0 (never read); Q.K on mma.sync m16n8k16
//   (fp32 accumulate) times 1/sqrt(128) plus the bf16 bias; online softmax per tile (NaN-propagating max, rescale
//   expf(old - new)); P as three bf16 limbs of fp32 p (a subnormal limb flushed to +0, the rest dropped); P.V on the
//   same mma; out = bf16(((O / Z) * sigmoid(gate)) * sigmoid((m + logf(Z)) - sink)), NaN stored as 0x7FC0.
//   Keys are split into absolute S-key splits (global layers S; sliding layers 256, from the split of the window's
//   start): each split from (-inf, 0, +0), then the row's splits floor(lo(p) / S) .. floor(p / S) merged in order (a
//   copy, then M' = nanmax(M, m), a = M == M' ? 1 : expf(M - M'), b likewise, Z = Z * a + z * b, O = O * a + o * b,
//   each product rounded, then the add).
// A row whose own window holds a non-finite key or value, or whose q'' holds a non-finite element, is recomputed
// alone by kx_row_scalar (e2614fe's per-key path), the same function in every form, so every form gives the same bits.
#pragma once
#include "inkling2_kx_attention.cuh"

namespace kx {
using namespace inkling2;

constexpr int D = 128;   // head dim
constexpr int NK = 64;   // keys a tile (absolute positions 64t .. 64t + 63)
constexpr int RT = 16;   // rows a CTA
constexpr int EXT = 1024; // relative-bias distances
constexpr int PAGE = 256; // the pool's page (cuda_attn.zig split_keys)
constexpr float kScale = 0.08838834764831845f;

// The shared exception counters (cuModuleGetGlobal): [0] (row, head) recomputes, [1] (CTA, tile) pairs with a
// flushed P limb, [2] (CTA, tile) pairs (and CTAs' q) with a flushed subnormal input.
}  // namespace kx
extern "C" {
__device__ unsigned long long tf_kx_counts[3];
}
namespace kx {

struct Args {
    const bf16 *q, *bias, *k, *v, *gate, *sink;
    bf16* out;
    int R, pos0, hq, hk, slots, window;
};

// Where key p's row starts (elements from k / v): a ring (a slot a position modulo `slots`) or the pool's pages.
struct Ring {
    int slots, hk;
    __device__ __forceinline__ size_t off(int p) const { return (size_t)(p % slots) * hk * D; }
    // keys p, p + 8, ..., p + 56 (a thread's keys of a tile): one modulo, then a wrap (slots >= 64)
    __device__ __forceinline__ void offs(int p, size_t (&o)[8]) const {
        int x = p % slots;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[i] = (size_t)x * hk * D;
            x += 8;
            if (x >= slots) x -= slots;
        }
    }
};
struct Paged {
    const int* pages;
    long long page;
    int hk;
    __device__ __forceinline__ size_t off(int p) const {
        return (size_t)__ldg(pages + p / PAGE) * (size_t)page + (size_t)(p % PAGE) * hk * D;
    }
    // keys p, p + 8, ..., p + 56 of one 64-key tile: one page (a tile lies inside a page)
    __device__ __forceinline__ void offs(int p, size_t (&o)[8]) const {
        const size_t at = (size_t)__ldg(pages + p / PAGE) * (size_t)page;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = at + (size_t)((p + 8 * i) % PAGE) * hk * D;
    }
};

struct Smem {
    __align__(16) bf16 keys[NK][136];
    __align__(16) bf16 values[NK][136];  // key-major like keys; P.V's B fragments by ldmatrix .trans
    __align__(16) bf16 prob[3][RT][NK + 8];  // padded rows: the A fragments' loads and the limbs' stores spread over banks
    float scores[RT][NK + 4];
    float maxima[RT], totals[RT], alpha[RT];
    int flag[RT], qflag[RT];
    unsigned long long nfmask;
    int subn;
    // kx-split-walk: the rows' running (M, Z) and this split's merge factors
    float M[RT], Z[RT], wa[RT], wb[RT];
    int take[RT];
};

// The (row, query head) of each of a CTA's 16 A rows (slots). Unpacked (P false): rows i0 .. i0 + 15 of query head
// h0 = blockIdx.y (grid y = hq). Packed (P, GQA packing): the gp = hq / hk query heads that share kv head
// blockIdx.y in one tile's rows, pair i0 + s -> row (i0 + s) / gp, head h0 + (i0 + s) % gp (grid y = hk), so a key and
// value tile serves gp heads. A slot's bytes depend only on its own (row, head): the slot is an A row of the MMA, and
// every other step is per row (5.1).
template <bool P>
struct Map {
    int i0, gp, h0, npair;
    __device__ __forceinline__ int row(int s) const { return P ? (i0 + s) / gp : i0 + s; }
    __device__ __forceinline__ int head(int s) const { return P ? h0 + (i0 + s) % gp : h0; }
    __device__ __forceinline__ bool live(int s) const { return i0 + s < npair; }
    // the pair whose q a slot's A row holds (a dead slot repeats the last pair; its scores are masked)
    __device__ __forceinline__ int qrow(int s) const { return P ? min(i0 + s, npair - 1) / gp : min(i0 + s, npair - 1); }
    __device__ __forceinline__ int qhead(int s) const { return P ? h0 + min(i0 + s, npair - 1) % gp : h0; }
    // the CTA's first and last rows
    __device__ __forceinline__ int first() const { return row(0); }
    __device__ __forceinline__ int last() const { return row(min(RT - 1, npair - 1 - i0)); }
};
template <bool P>
__device__ __forceinline__ Map<P> map_of(const Args& a) {
    const int grp = a.hq / a.hk;
    Map<P> m;
    m.i0 = blockIdx.x * RT;
    m.gp = P ? grp : 1;
    m.h0 = P ? blockIdx.y * grp : blockIdx.y;
    m.npair = P ? a.R * grp : a.R;
    return m;
}

__device__ __forceinline__ bool nonfinite16(uint32_t h) { return (h & 0x7f80u) == 0x7f80u; }
__device__ __forceinline__ bool subnormal16(uint32_t h) { return (h & 0x7f80u) == 0 && (h & 0x7fu) != 0; }

// Two bf16 in a word, magnitudes a = x & 0x7fff7fff: bit 15 (low element) / bit 31 (high) set where the element is
// non-finite (a >= 0x7f80: + 0x80 carries into bit 15), resp. subnormal (exponent 0: (a | 0x8000) - 0x80 clears bit 15;
// a != 0: a + 0x7fff sets it). No carry or borrow crosses the halves.
__device__ __forceinline__ uint32_t nf_bits(uint32_t x) { return ((x & 0x7fff7fffu) + 0x00800080u) & 0x80008000u; }
__device__ __forceinline__ uint32_t sn_bits(uint32_t x) {
    const uint32_t a = x & 0x7fff7fffu;
    return ~((a | 0x80008000u) - 0x00800080u) & (a + 0x7fff7fffu) & 0x80008000u;
}
// Two bf16 in a word: a subnormal or non-finite element to +0.
__device__ __forceinline__ uint32_t clean2(uint32_t x, uint32_t& bad, uint32_t& subn) {
    const uint32_t nf = nf_bits(x), sn = sn_bits(x);
    bad |= nf;
    subn |= sn;
    return x & ~(((nf | sn) >> 15) * 0xffffu);
}

// Eight bf16: the same, the masks applied only when an element needs it (rare).
__device__ __forceinline__ uint4 clean8(uint4 x, uint32_t& bad, uint32_t& subn) {
    const uint32_t nf = nf_bits(x.x) | nf_bits(x.y) | nf_bits(x.z) | nf_bits(x.w);
    const uint32_t sn = sn_bits(x.x) | sn_bits(x.y) | sn_bits(x.z) | sn_bits(x.w);
    bad |= nf;
    subn |= sn;
    if ((nf | sn) == 0) return x;
    return make_uint4(clean2(x.x, bad, subn), clean2(x.y, bad, subn), clean2(x.z, bad, subn), clean2(x.w, bad, subn));
}

// A bf16 for the scalar path: a subnormal to +0 (non-finite values keep their class).
__device__ __forceinline__ float fs(bf16 x) {
    const uint32_t h = __bfloat16_as_ushort(x);
    return subnormal16(h) ? 0.f : f(x);
}

__device__ __forceinline__ float nanmax(float a, float b) { return isnan(a) || isnan(b) ? nanf("") : fmaxf(a, b); }

// Two 8x8 bf16 matrices from shared memory, transposed (lanes 0-15 give the rows' addresses): the m16n8k16 B fragment
// {B[2s][n], B[2s+1][n]}, {B[2s+8][n], B[2s+9][n]} (s = lane % 4, n = lane / 4) of a key-major tile. Moves bytes only.
__device__ __forceinline__ void ldsm_x2_t(uint32_t& r0, uint32_t& r1, const bf16* p) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(at) : "memory");
}

// Four 8x8 bf16 matrices from shared memory (lanes 8i .. 8i + 7 give matrix i's rows): thread T gets row T / 4,
// elements 2 (T % 4), +1 of each, i.e. the m16n8k16 B fragment of key-major rows. Moves bytes only.
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const bf16* p) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(at)
                 : "memory");
}

// A tile's keys and values as this thread loads them (8 keys x 16 bytes of K and of V): issued together, before any
// is stored, so the loads overlap (the stores to shared memory would otherwise order each load after the last store).
struct Raw {
    uint4 k[8], v[8];
    unsigned live;  // bit i: key i of this thread was loaded (inside [k_first, k_last])
};

// Two bf16 lanes: the larger, resp. smaller, magnitude of a and b lane by lane (NaN wins the larger). A screen only.
__device__ __forceinline__ uint32_t maxabs2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t minabs2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("min.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t maxabs8(uint4 x) { return maxabs2(maxabs2(x.x, x.y), maxabs2(x.z, x.w)); }
__device__ __forceinline__ uint32_t minabs8(uint4 x) { return minabs2(minabs2(x.x, x.y), minabs2(x.z, x.w)); }
template <class A>
__device__ __forceinline__ void fetch(const Args& a, const A& L, int kh, int base, int k_first, int k_last, Raw& r) {
    const int t = threadIdx.x, j0 = t >> 4, c0 = (t & 15) * 8;
    size_t o[8];
    L.offs(base + j0, o);
    r.live = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key = base + j0 + 8 * i;
        r.k[i] = make_uint4(0, 0, 0, 0);
        r.v[i] = make_uint4(0, 0, 0, 0);
        if (key >= k_first && key <= k_last) {
            r.live |= 1u << i;
            r.k[i] = __ldg(reinterpret_cast<const uint4*>(a.k + o[i] + (size_t)kh * D + c0));
            r.v[i] = __ldg(reinterpret_cast<const uint4*>(a.v + o[i] + (size_t)kh * D + c0));
        }
    }
}

// The output of a (row, head, column): ((O / Z) * sigmoid(gate)) * sigmoid((m + logf(Z)) - sink); NaN as 0x7FC0.
__device__ __forceinline__ bf16 final_out(float o, float z, float m, bf16 gate, bf16 sink) {
    const float lse = m + logf(z);
    const float value = o / z;
    const float y = (value * sigmoid(f(gate))) * sigmoid(lse - f(sink));
    return isnan(y) ? __ushort_as_bfloat16(0x7fc0) : b(y);
}

// e2614fe's ik_attention_exception for one row (keys lo(p) .. p, one thread a column, the block's tree sum), with
// subnormal inputs read as +0. Writes the columns this CTA owns (column group z of C: (c % 32) / (8 * NC) == z).
template <class A>
__device__ __noinline__ void row_scalar(const Args& a, const A& L, int row, int head, int kh, int zc, int NC) {
    const int c = threadIdx.x;
    const int last = a.pos0 + row, first = a.window ? max(0, last - a.window) : 0;
    const size_t qi = ((size_t)row * a.hq + head) * D + c;
    const float q = fs(a.q[qi]);
    float m = -INFINITY, z = 0.f, out = 0.f;
    bool nan_value = false, pos_inf = false, neg_inf = false;
    for (int key = first; key <= last; ++key) {
        const size_t ki = L.off(key) + (size_t)kh * D + c;
        float score = sum(q * fs(a.k[ki])) * kScale;
        const int distance = last - key;
        if (distance < EXT) score += f(a.bias[((size_t)row * a.hq + head) * EXT + distance]);
        const float nm = nanmax(m, score);
        const float alpha = m == -INFINITY ? 0.f : expf(m - nm), p = expf(score - nm);
        const float v = fs(a.v[ki]);
        nan_value |= isnan(v);
        pos_inf |= isinf(v) && v > 0;
        neg_inf |= isinf(v) && v < 0;
        out = out * alpha + p * (isfinite(v) ? v : 0.f);
        z = z * alpha + p;
        m = nm;
    }
    out = nan_value || (pos_inf && neg_inf) ? nanf("") : pos_inf ? INFINITY : neg_inf ? -INFINITY : out / z;
    if (!isfinite(m) || !isfinite(z)) out = nanf("");
    const float gate = sigmoid(f(a.gate[qi]));
    const float sink = sigmoid((m + logf(z)) - f(a.sink[(size_t)row * a.hq + head]));
    const float y = (out * gate) * sink;
    if ((c % 32) / (8 * NC) == zc) a.out[qi] = isnan(y) ? __ushort_as_bfloat16(0x7fc0) : b(y);
}

// A CTA's 16 slots: the q check (non-finite rows flagged, subnormals counted) and the A fragment.
template <bool P>
__device__ __forceinline__ void load_q(const Args& a, Smem& sm, const Map<P>& mp, uint32_t (&qa)[8][4], bool count) {
    const int t = threadIdx.x, lane = t & 31, g = lane >> 2, sub = lane & 3;
    bool subn = false;
    for (int i = t; i < RT * D; i += 128) {
        const int sl = i / D;
        if (!mp.live(sl)) continue;
        const uint32_t h = __bfloat16_as_ushort(a.q[((size_t)mp.row(sl) * a.hq + mp.head(sl)) * D + i % D]);
        if (nonfinite16(h)) sm.qflag[sl] = 1;
        subn |= subnormal16(h);
    }
    if (__syncthreads_or(subn) && count && t == 0) atomicAdd(&tf_kx_counts[2], 1ull);
    uint32_t bad = 0, sn = 0;
    const bf16* q0b = a.q + ((size_t)mp.qrow(g) * a.hq + mp.qhead(g)) * D;
    const bf16* q1b = a.q + ((size_t)mp.qrow(g + 8) * a.hq + mp.qhead(g + 8)) * D;
    for (int s = 0; s < 8; ++s) {
        const bf16* q0 = q0b + s * 16;
        const bf16* q1 = q1b + s * 16;
        qa[s][0] = clean2(*reinterpret_cast<const uint32_t*>(q0 + 2 * sub), bad, sn);
        qa[s][1] = clean2(*reinterpret_cast<const uint32_t*>(q1 + 2 * sub), bad, sn);
        qa[s][2] = clean2(*reinterpret_cast<const uint32_t*>(q0 + 8 + 2 * sub), bad, sn);
        qa[s][3] = clean2(*reinterpret_cast<const uint32_t*>(q1 + 8 + 2 * sub), bad, sn);
    }
}

// The state of a range: maxima -inf, totals 0, out +0 (and the rows' flags from their q).
template <int NC>
__device__ __forceinline__ void reset(Smem& sm, float (&out)[NC][4]) {
    const int t = threadIdx.x;
#pragma unroll
    for (int n = 0; n < NC; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) out[n][e] = 0.f;
    if (t < RT) {
        sm.maxima[t] = -INFINITY;
        sm.totals[t] = 0.f;
        sm.flag[t] = sm.qflag[t];
    }
    __syncthreads();
}

// A6-A10 of one tile from its scores sc, G lanes a row (G = 8, 16 or 32; rows 0 .. 128 / G - 1, the CTA's live slots
// among them): the max, the rescale, the weights and their limbs, the sums. Lane l of a row computes the weights of its
// columns l, l + G, ...; the max and the sum are 5.1's chains: for col0 = 0 .. 7, over columns col0, col0 + 8, ...,
// col0 + 56 in order (the sum's terms gathered from the lanes that hold them), then xor 1, 2, 4 over col0. Every
// value is computed from the same operands in the same order whatever G is. Returns whether a limb was flushed.
// `fresh`: the range's first tile, its running (max, total) the reset's (-inf, 0) without reading them.
template <int G>
__device__ __forceinline__ bool softmax_g(Smem& sm, const float (*sc)[NK + 4], bool fresh) {
    constexpr int H = G / 8, NJ = NK / G;  // lanes holding one chain's terms; columns a lane
    const int t = threadIdx.x, rr = t / G, l = t % G, col0 = l % 8;
    const float old = fresh ? -INFINITY : sm.maxima[rr];
    float m = old;
    float sv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) sv[i] = sc[rr][col0 + 8 * i];
#pragma unroll
    for (int i = 0; i < 8; ++i) m = nanmax(m, sv[i]);
    for (int off = 1; off < 8; off <<= 1) m = nanmax(m, __shfl_xor_sync(0xffffffffu, m, off, 8));
    const float aold = old == m ? 1.f : expf(old - m);
    bool flushed = false;
    float pz[NJ];  // this lane's weights (full fp32, for the sum)
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int i = l / 8 + H * j, col = col0 + 8 * i;  // = l + G * j
        float p = sv[i] == -INFINITY ? 0.f : expf(sv[i] - m);
        pz[j] = p;
#pragma unroll
        for (int limb = 0; limb < 3; ++limb) {
            bf16 piece = b(p);
            p -= f(piece);
            if (subnormal16(__bfloat16_as_ushort(piece))) {
                piece = b(0.f);
                flushed = true;
            }
            sm.prob[limb][rr][col] = piece;
        }
    }
    float z = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (H == 1) z += pz[i];
        else z += __shfl_sync(0xffffffffu, pz[i / H], col0 + 8 * (i % H), G);
    }
    for (int off = 1; off < 8; off <<= 1) z += __shfl_xor_sync(0xffffffffu, z, off, 8);
    __syncwarp();
    if (!l) {
        sm.totals[rr] = (fresh ? 0.f : sm.totals[rr]) * aold + z;
        sm.maxima[rr] = m;
        sm.alpha[rr] = aold;
    }
    return flushed;
}

// The tile's softmax with as many lanes a row as the CTA's live slots allow (n live slots: 32 lanes up to 4, 16 up to
// 8, else 8). A dead slot's state is never read for an output.
__device__ __forceinline__ bool softmax(Smem& sm, const float (*sc)[NK + 4], int live, bool fresh = false) {
    if (live <= 4) return softmax_g<32>(sm, sc, fresh);
    if (live <= 8) return softmax_g<16>(sm, sc, fresh);
    return softmax_g<8>(sm, sc, fresh);
}

// Keys k_first .. k_last of the CTA's slots into (maxima, totals, out): steps A2-A11, tile by tile.
// Column group zc of C = 4 / NC: this CTA's P.V columns. `count`: this CTA adds to the counters. The next tile's keys
// and values are loaded while this one is computed (registers), its bias before its scores; the arithmetic and its
// order are 5.1's.
// XS (kx-split-walk): `raw` already holds the range's first tile, and the last tile loads the next range's first
// (keys n_first .. n_last; none when n_first > n_last) while it is computed, so a walk's split boundary does not wait
// on its loads. Moves bytes only: the loaded tiles and the arithmetic are the same.
template <class A, int NC, bool P, bool XS = false>
__device__ __forceinline__ void range(const Args& a, const A& L, Smem& sm, const Map<P>& mp, int kh, int k_first,
                                      int k_last, const uint32_t (&qa)[8][4], float (&out)[NC][4], int zc, bool count,
                                      Raw* xraw = nullptr, int n_first = 1, int n_last = 0) {
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, sub = lane & 3;
    // this thread's score rows: slots g and g + 8
    const int srow[2] = {mp.row(g), mp.row(g + 8)};
    const bool slive[2] = {mp.live(g), mp.live(g + 8)};
    const int nlive = min(RT, mp.npair - mp.i0);
    const size_t sbias[2] = {((size_t)srow[0] * a.hq + mp.head(g)) * EXT, ((size_t)srow[1] * a.hq + mp.head(g + 8)) * EXT};
    Raw own;
    Raw& raw = XS ? *xraw : own;
    if (!XS) fetch(a, L, kh, k_first / NK * NK, k_first, k_last, raw);
    for (int base = k_first / NK * NK; base <= k_last; base += NK) {
        // A3: the tile's keys and values; outside [k_first, k_last] +0 (never read); subnormal +0, non-finite +0
        // with the key's bit set
        unsigned long long nf = 0;
        uint32_t sn = 0;
        // screen: the loaded elements' largest magnitude (non-finite: >= 0x7f80 or NaN) and smallest (subnormal or
        // zero: < 0x0080); only a thread whose screen fires runs the per-element rule (the same bytes either way)
        uint32_t hi = 0, lo = 0x7f7f7f7fu;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (raw.live >> i & 1) {
                hi = maxabs2(hi, maxabs2(maxabs8(raw.k[i]), maxabs8(raw.v[i])));
                lo = minabs2(lo, minabs2(minabs8(raw.k[i]), minabs8(raw.v[i])));
            }
        const bool screen = (hi & 0x7fffu) >= 0x7f80u || (hi >> 16 & 0x7fffu) >= 0x7f80u || (lo & 0x7fffu) < 0x80u ||
                            (lo >> 16 & 0x7fffu) < 0x80u;
        if (screen) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t bad = 0;
                raw.k[i] = clean8(raw.k[i], bad, sn);
                raw.v[i] = clean8(raw.v[i], bad, sn);
                if (bad) nf |= 1ull << ((t >> 4) + 8 * i);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int j = (t >> 4) + 8 * i, c0 = (t & 15) * 8;
            *reinterpret_cast<uint4*>(&sm.keys[j][c0]) = raw.k[i];
            *reinterpret_cast<uint4*>(&sm.values[j][c0]) = raw.v[i];
        }
        // A5's bias of this thread's scores (bf16 as stored), loaded before the scores need it
        uint16_t bz[2][2][2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = w * 16 + n * 8 + 2 * sub + e;
                    const int distance = a.pos0 + srow[half] - (base + col);
                    const bool valid = slive[half] && distance >= 0 && (!a.window || distance <= a.window);
                    bz[n][half][e] = valid && distance < EXT
                                         ? __ldg(reinterpret_cast<const unsigned short*>(a.bias) + sbias[half] + distance)
                                         : 0;
                }
        if (base + NK <= k_last) fetch(a, L, kh, base + NK, k_first, k_last, raw);
        else if (XS && n_first <= n_last) fetch(a, L, kh, n_first / NK * NK, n_first, n_last, raw);
        if (__syncthreads_or(nf != 0 || sn != 0)) {
            if (nf) atomicOr(&sm.nfmask, nf);
            if (sn) sm.subn = 1;
            __syncthreads();
            if (t < RT && mp.live(t)) {
                const int p = a.pos0 + mp.row(t), lo = a.window ? max(0, p - a.window) : 0;
                const int x0 = max(lo, base) - base, x1 = min(p, base + NK - 1) - base;
                if (x0 <= x1) {
                    const unsigned long long hi = x1 == 63 ? ~0ull : (1ull << (x1 + 1)) - 1;
                    if (sm.nfmask & hi & ~((1ull << x0) - 1)) sm.flag[t] = 1;
                }
            }
            if (t == 0 && count && sm.subn) atomicAdd(&tf_kx_counts[2], 1ull);
            __syncthreads();
            if (t == 0) {
                sm.nfmask = 0;
                sm.subn = 0;
            }
        }
        // A4-A5: scores
        float sc[2][4] = {};
        // the B fragments of keys w * 16 + n * 8 .. + 7 by ldmatrix (four 8x8 matrices an instruction: dims 16s .. 16s + 31)
        uint32_t kb[2][8][2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int s = 0; s < 8; s += 2)
                ldsm_x4(kb[n][s][0], kb[n][s][1], kb[n][s + 1][0], kb[n][s + 1][1],
                        &sm.keys[w * 16 + n * 8 + (lane & 7)][s * 16 + (lane >> 3) * 8]);
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int s = 0; s < 8; ++s) ik_mma(sc[n], qa[s], kb[n][s][0], kb[n][s][1]);
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = w * 16 + n * 8 + 2 * sub + e;
                    const int distance = a.pos0 + srow[half] - (base + col);
                    float score = sc[n][2 * half + e] * kScale;
                    const bool valid = slive[half] && distance >= 0 && (!a.window || distance <= a.window);
                    if (valid && distance < EXT) score += f(__ushort_as_bfloat16(bz[n][half][e]));
                    sm.scores[g + half * 8][col] = valid ? score : -INFINITY;
                }
        __syncthreads();
        // A6-A10: the tile's max, the rescale, the weights and their limbs, the sums
        // XS: each range starts from (-inf, 0) without a reset of the shared (maxima, totals)
        const bool flushed = softmax(sm, sm.scores, nlive, XS && base == k_first / NK * NK);
        if (__syncthreads_or(flushed) && count && t == 0) atomicAdd(&tf_kx_counts[1], 1ull);
        // A11: P.V on this CTA's columns (per accumulator: s ascending, limbs 0, 1, 2)
#pragma unroll
        for (int n = 0; n < NC; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) out[n][e] *= sm.alpha[g + (e / 2) * 8];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint32_t vb[NC][2];
#pragma unroll
            for (int n = 0; n < NC; ++n) ldsm_x2_t(vb[n][0], vb[n][1], &sm.values[s * 16 + (lane & 15)][w * 32 + (n + zc * NC) * 8]);
            uint32_t p[3][4];
#pragma unroll
            for (int limb = 0; limb < 3; ++limb) {
                p[limb][0] = *reinterpret_cast<const uint32_t*>(&sm.prob[limb][g][s * 16 + 2 * sub]);
                p[limb][1] = *reinterpret_cast<const uint32_t*>(&sm.prob[limb][g + 8][s * 16 + 2 * sub]);
                p[limb][2] = *reinterpret_cast<const uint32_t*>(&sm.prob[limb][g][s * 16 + 8 + 2 * sub]);
                p[limb][3] = *reinterpret_cast<const uint32_t*>(&sm.prob[limb][g + 8][s * 16 + 8 + 2 * sub]);
            }
#pragma unroll
            for (int limb = 0; limb < 3; ++limb)
#pragma unroll
                for (int n = 0; n < NC; ++n) ik_mma(out[n], p[limb], vb[n][0], vb[n][1]);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void init(Smem& sm) {
    const int t = threadIdx.x;
    if (t < RT) sm.qflag[t] = 0;
    if (t == 0) {
        sm.nfmask = 0;
        sm.subn = 0;
    }
    __syncthreads();
}

// The flagged rows of a CTA, one by one (the block's tree sum: every thread).
template <class A, bool P>
__device__ __forceinline__ void recompute(const Args& a, const A& L, Smem& sm, const Map<P>& mp, int kh, int zc, int NC,
                                          bool count) {
    for (int rr = 0; rr < RT && mp.live(rr); ++rr)
        if (sm.flag[rr]) {
            row_scalar(a, L, mp.row(rr), mp.head(rr), kh, zc, NC);
            if (count && threadIdx.x == 0) atomicAdd(&tf_kx_counts[0], 1ull);
        }
}

// kx-tile / kx-dec: grid (ceil(R / 16), hq, C), 128 threads; packed (P): grid (ceil(R * hq / hk / 16), hk, C). A CTA's
// slots over keys lo(its first row) .. its last row (sliding: window 1023; window 0: every key from 0).
template <class A, int C, bool P>
__device__ __forceinline__ void tile(Args a, const A& L, const int* posp) {
    constexpr int NC = 4 / C;
    __shared__ Smem sm;
    if (posp != nullptr) a.pos0 = *posp;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, sub = lane & 3;
    const Map<P> mp = map_of<P>(a);
    const int kh = P ? blockIdx.y : blockIdx.y / (a.hq / a.hk), zc = blockIdx.z;
    const bool count = zc == 0;
    const int k_last = a.pos0 + mp.last();
    const int k_first = a.window ? max(0, a.pos0 + mp.first() - a.window) : 0;
    init(sm);
    uint32_t qa[8][4];
    load_q<P>(a, sm, mp, qa, count);
    float out[NC][4];
    reset<NC>(sm, out);
    range<A, NC, P>(a, L, sm, mp, kh, k_first, k_last, qa, out, zc, count);
#pragma unroll
    for (int n = 0; n < NC; ++n)
        for (int half = 0; half < 2; ++half)
            for (int e = 0; e < 2; ++e) {
                const int rr = g + half * 8, c = w * 32 + (n + zc * NC) * 8 + 2 * sub + e;
                if (!mp.live(rr) || sm.flag[rr]) continue;
                const int row = mp.row(rr), head = mp.head(rr);
                const size_t at = ((size_t)row * a.hq + head) * D + c;
                a.out[at] = final_out(out[n][half * 2 + e], sm.totals[rr], sm.maxima[rr], a.gate[at],
                                      a.sink[(size_t)row * a.hq + head]);
            }
    recompute<A, P>(a, L, sm, mp, kh, zc, NC, count);
}

// The first split a forward's partials are indexed from: floor(lo(pos0) / S) (sliding; 0 global). tf_kx_split and
// tf_kx_merge index the scratch by s - first.
template <int S>
__device__ __forceinline__ int split_base(const Args& a) {
    return a.window ? max(0, a.pos0 - a.window) / S : 0;
}

// kx-split: grid (ceil(R / 16), hq, C * G) (packed: (ceil(R * hq / hk / 16), hk, C * G)); CTA z takes column group z % C
// and splits f + z / C, f + z / C + G, ... (f = split_base) while a split starts at or before the CTA's last row. Each
// split from (-inf, 0, +0) over its keys (from the CTA's first row's window start); writes per (split, row, head):
// (m, z) in st, the row's flag in fl, o (this CTA's columns) in op. Splits are absolute: split s holds keys sS .. sS +
// S - 1.
template <class A, int S, int C, bool P>
__device__ __forceinline__ void split(Args a, const A& L, float2* st, int* fl, float* op, const int* posp) {
    constexpr int NC = 4 / C;
    __shared__ Smem sm;
    if (posp != nullptr) a.pos0 = *posp;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, sub = lane & 3;
    const Map<P> mp = map_of<P>(a);
    const int kh = P ? blockIdx.y : blockIdx.y / (a.hq / a.hk);
    const int zc = blockIdx.z % C, f0 = split_base<S>(a), z0 = f0 + blockIdx.z / C, G = gridDim.z / C;
    const bool count = zc == 0;
    const int last = a.pos0 + mp.last();
    if (z0 * S > last) return;
    init(sm);
    uint32_t qa[8][4];
    load_q<P>(a, sm, mp, qa, count);
    float out[NC][4];
    for (int s = z0; s * S <= last; s += G) {
        reset<NC>(sm, out);
        const int k_first = max(s * S, a.window ? max(0, a.pos0 + mp.first() - a.window) : 0);
        range<A, NC, P>(a, L, sm, mp, kh, k_first, min(s * S + S - 1, last), qa, out, zc, count);
        const size_t base = (size_t)(s - f0) * a.R;
#pragma unroll
        for (int n = 0; n < NC; ++n)
            for (int half = 0; half < 2; ++half)
                for (int e = 0; e < 2; ++e) {
                    const int rr = g + half * 8, c = w * 32 + (n + zc * NC) * 8 + 2 * sub + e;
                    if (mp.live(rr)) op[((base + mp.row(rr)) * a.hq + mp.head(rr)) * D + c] = out[n][half * 2 + e];
                }
        if (count && t < RT && mp.live(t)) {
            const size_t at = (base + mp.row(t)) * a.hq + mp.head(t);
            st[at] = make_float2(sm.maxima[t], sm.totals[t]);
            fl[at] = sm.flag[t];
        }
        __syncthreads();
    }
}

// One row's merge step: (M, Z) and the factors a (on the running) and b (on split s).
__device__ __forceinline__ void merge_factors(float& M, float& Z, float ms, float zs, float& fa, float& fb) {
    const float mn = nanmax(M, ms);
    fa = M == mn ? 1.f : expf(M - mn);
    fb = ms == mn ? 1.f : expf(ms - mn);
    Z = __fadd_rn(__fmul_rn(Z, fa), __fmul_rn(zs, fb));
    M = mn;
}

// tf_kx_merge: grid (R, hq), 128 threads (a column each). Row r's splits floor(lo(p) / S) .. floor(p / S) in order (a
// copy of the first, then the merge steps), then the output (or, when any of its splits flagged it, kx_row_scalar over
// its whole window).
template <class A, int S>
__device__ __forceinline__ void merge(Args a, const A& L, const float2* st, const int* fl, const float* op,
                                      const int* posp) {
    if (posp != nullptr) a.pos0 = *posp;
    const int row = blockIdx.x, head = blockIdx.y, c = threadIdx.x, kh = head / (a.hq / a.hk);
    const int p = a.pos0 + row, last = p / S, f0 = split_base<S>(a);
    const int first = a.window ? max(0, p - a.window) / S : 0;
    size_t at = ((size_t)(first - f0) * a.R + row) * a.hq + head;
    float2 mz = st[at];
    float M = mz.x, Z = mz.y, O = op[at * D + c];
    bool flagged = fl[at] != 0;
    // unrolled so a few splits' loads are in flight at once (the merge's order is the loop's)
#pragma unroll 8
    for (int s = first + 1; s <= last; ++s) {
        at = ((size_t)(s - f0) * a.R + row) * a.hq + head;
        mz = st[at];
        float fa, fb;
        merge_factors(M, Z, mz.x, mz.y, fa, fb);
        O = __fadd_rn(__fmul_rn(O, fa), __fmul_rn(op[at * D + c], fb));
        flagged |= fl[at] != 0;
    }
    if (flagged) {
        row_scalar(a, L, row, head, kh, 0, 4);
        if (c == 0) atomicAdd(&tf_kx_counts[0], 1ull);
        return;
    }
    const size_t qi = ((size_t)row * a.hq + head) * D + c;
    a.out[qi] = final_out(O, Z, M, a.gate[qi], a.sink[(size_t)row * a.hq + head]);
}

// kx-split-walk: grid (ceil(R / 16), hq) (packed: (ceil(R * hq / hk / 16), hk)), 128 threads. Splits floor(lo(first
// row) / S) .. floor(last row / S) in order, each from (-inf, 0, +0), merged into the slots' running (M, Z, O) in
// registers as tf_kx_merge merges them: a row takes its splits floor(lo(p) / S) .. floor(p / S), the first as a copy.
template <class A, int S, bool P>
__device__ __forceinline__ void walk(Args a, const A& L, const int* posp) {
    __shared__ Smem sm;
    if (posp != nullptr) a.pos0 = *posp;
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, sub = lane & 3;
    const Map<P> mp = map_of<P>(a);
    const int kh = P ? blockIdx.y : blockIdx.y / (a.hq / a.hk);
    const int last = a.pos0 + mp.last();
    const int k_lo = a.window ? max(0, a.pos0 + mp.first() - a.window) : 0;
    init(sm);
    uint32_t qa[8][4];
    load_q<P>(a, sm, mp, qa, true);
    float out[4][4], O[4][4] = {};
    // this thread's rows g and g + 8: their splits s0 .. sl, running (M, Z) in registers (the same steps as
    // tf_kx_merge's, computed by every thread that holds a column of the row)
    int s0r[2], slr[2];
    float Mr[2] = {-INFINITY, -INFINITY}, Zr[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int p = a.pos0 + mp.row(g + 8 * h);
        s0r[h] = mp.live(g + 8 * h) ? (a.window ? max(0, p - a.window) / S : 0) : 1 << 30;
        slr[h] = mp.live(g + 8 * h) ? p / S : -1;
    }
    // the flags start from q's and are never reset: a split a row does not take holds none of its window's keys, so
    // it never flags the row
    reset<4>(sm, out);
    // the first split's first tile; each split's last tile loads the next split's first (range XS)
    Raw raw;
    fetch(a, L, kh, k_lo / NK * NK, k_lo, min(k_lo / S * S + S - 1, last), raw);
    for (int s = k_lo / S; s * S <= last; ++s) {
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) out[n][e] = 0.f;
        range<A, 4, P, true>(a, L, sm, mp, kh, max(s * S, k_lo), min(s * S + S - 1, last), qa, out, 0, true, &raw,
                             (s + 1) * S, min((s + 1) * S + S - 1, last));
        // the split's (m, z) are final (range ends on a barrier) and are read before the next range's first barrier
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (s < s0r[h] || s > slr[h]) continue;
            const float ms = sm.maxima[g + 8 * h], zs = sm.totals[g + 8 * h];
            if (s == s0r[h]) {
                Mr[h] = ms;
                Zr[h] = zs;
#pragma unroll
                for (int n = 0; n < 4; ++n)
#pragma unroll
                    for (int e = 2 * h; e < 2 * h + 2; ++e) O[n][e] = out[n][e];
            } else {
                float fa, fb;
                merge_factors(Mr[h], Zr[h], ms, zs, fa, fb);
#pragma unroll
                for (int n = 0; n < 4; ++n)
#pragma unroll
                    for (int e = 2 * h; e < 2 * h + 2; ++e) O[n][e] = __fadd_rn(__fmul_rn(O[n][e], fa), __fmul_rn(out[n][e], fb));
            }
        }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n)
        for (int half = 0; half < 2; ++half)
            for (int e = 0; e < 2; ++e) {
                const int rr = g + half * 8, c = w * 32 + n * 8 + 2 * sub + e;
                if (!mp.live(rr) || sm.flag[rr]) continue;
                const int row = mp.row(rr), head = mp.head(rr);
                const size_t at = ((size_t)row * a.hq + head) * D + c;
                a.out[at] = final_out(O[n][half * 2 + e], Zr[half], Mr[half], a.gate[at], a.sink[(size_t)row * a.hq + head]);
            }
    recompute<A, P>(a, L, sm, mp, kh, 0, 4, true);
}

}  // namespace kx

// The relative bias table: out bf16 [R * hq, 1024] = bf16(r[pair, :16] . proj[:, key]), fp32 sums in d order
// (e2614fe's tf_ik_relative expression); grid (R * hq, 1024 / 128), 128 threads, one key a thread.
extern "C" __global__ void __launch_bounds__(128) tf_kx_relative(const bf16* r, const bf16* proj, bf16* out) {
    const size_t pair = blockIdx.x;
    const int key = blockIdx.y * 128 + threadIdx.x;
    float value = 0.f;
#pragma unroll
    for (int d = 0; d < 16; ++d) value += f(r[pair * 16 + d]) * f(proj[d * kx::EXT + key]);
    out[pair * kx::EXT + key] = b(value);
}

// Scalar arguments (cuda_attn.zig `coreKx`): q, bias, k, v, gate, sink, out, [st, fl, op,] R, pos0, hq, hk, slots,
// window, posp[, pages, page].
#define KX_ARGS                                                                                                    \
    const bf16 *q, const bf16 *bias, const bf16 *k, const bf16 *v, const bf16 *gate, const bf16 *sink, bf16 *out
#define KX_GEO int R, int pos0, int hq, int hk, int slots, int window, const int *posp
#define KX_A kx::Args{q, bias, k, v, gate, sink, out, R, pos0, hq, hk, slots, window}

#define KX_TILE1(C, P, SFX)                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) tf_kx_tile_c##C##SFX(KX_ARGS, KX_GEO) {                     \
        kx::tile<kx::Ring, C, P>(KX_A, kx::Ring{slots, hk}, posp);                                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128)                                                              \
        tf_kx_tile_c##C##SFX##_pg(KX_ARGS, KX_GEO, const int* pages, long long page) {                             \
        kx::tile<kx::Paged, C, P>(KX_A, kx::Paged{pages, page, hk}, posp);                                         \
    }
#define KX_TILE(C) KX_TILE1(C, false, ) KX_TILE1(C, true, _pk)
KX_TILE(1)
KX_TILE(2)
KX_TILE(4)

#define KX_SPLIT1(S, C, P, SFX)                                                                                    \
    extern "C" __global__ void __launch_bounds__(128)                                                              \
        tf_kx_split_s##S##_c##C##SFX(KX_ARGS, float2* st, int* fl, float* op, KX_GEO) {                            \
        kx::split<kx::Ring, S, C, P>(KX_A, kx::Ring{slots, hk}, st, fl, op, posp);                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) tf_kx_split_s##S##_c##C##SFX##_pg(                           \
        KX_ARGS, float2* st, int* fl, float* op, KX_GEO, const int* pages, long long page) {                       \
        kx::split<kx::Paged, S, C, P>(KX_A, kx::Paged{pages, page, hk}, st, fl, op, posp);                         \
    }
#define KX_SPLIT(S, C) KX_SPLIT1(S, C, false, ) KX_SPLIT1(S, C, true, _pk)

#define KX_WALK1(S, P, SFX)                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) tf_kx_walk_s##S##SFX(KX_ARGS, KX_GEO) {                     \
        kx::walk<kx::Ring, S, P>(KX_A, kx::Ring{slots, hk}, posp);                                                 \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128)                                                              \
        tf_kx_walk_s##S##SFX##_pg(KX_ARGS, KX_GEO, const int* pages, long long page) {                             \
        kx::walk<kx::Paged, S, P>(KX_A, kx::Paged{pages, page, hk}, posp);                                         \
    }

#define KX_S(S)                                                                                                    \
    KX_SPLIT(S, 1)                                                                                                 \
    KX_SPLIT(S, 4)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(128)                                                              \
        tf_kx_merge_s##S(KX_ARGS, float2* st, int* fl, float* op, KX_GEO) {                                        \
        kx::merge<kx::Ring, S>(KX_A, kx::Ring{slots, hk}, st, fl, op, posp);                                       \
    }                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128)                                                              \
        tf_kx_merge_s##S##_pg(KX_ARGS, float2* st, int* fl, float* op, KX_GEO, const int* pages, long long page) { \
        kx::merge<kx::Paged, S>(KX_A, kx::Paged{pages, page, hk}, st, fl, op, posp);                               \
    }                                                                                                              \
    KX_WALK1(S, false, )                                                                                           \
    KX_WALK1(S, true, _pk)
KX_S(256)
KX_S(512)
KX_S(1024)
KX_S(2048)
KX_S(4096)
