// Copyright kortexa-ai. Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
// From kortexa-ai/TensorFold e2614fe (zig/kernels/cuda/inkling2_attention.cuh); modified: the include renamed to
// inkling2_kx_common.cuh and the kernel renamed tf_ik_attention_e2614fe (a control for tf-inkling2-attn --limb only;
// the engine runs inkling2_kx_fp32.cuh's forms). Otherwise unchanged.
// Relative-bias GQA on BF16 keys/values. Q.K and all three limbs of FP32 P.V sum in FP32.
#pragma once
#include "inkling2_kx_common.cuh"
using namespace inkling2;

struct InklingAttention {
    const bf16 *q, *k, *v, *relative, *gate, *sink;
    bf16* out;
    int rows, pos, hq, hk, capacity, window;
};

__device__ inline uint32_t ik_pack(float a, float c) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(a, c);
    return *reinterpret_cast<const uint32_t*>(&p);
}
__device__ inline void ik_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Exceptional values take a masked SIMT path; hidden poisoned V must never enter a zero-probability MMA.
__device__ inline void ik_attention_exception(InklingAttention a, int r0, int head) {
    const int c = threadIdx.x, kh = head / (a.hq / a.hk);
    for (int row = r0; row < min(a.rows, r0 + 16); ++row) {
        const int last = a.pos + row, first = a.window ? max(0, last - a.window) : 0;
        const size_t qi = ((size_t)row * a.hq + head) * 128 + c;
        const float q = f(a.q[qi]);
        float m = -INFINITY, z = 0.f, out = 0.f;
        bool nan_value = false, pos_inf = false, neg_inf = false;
        for (int key = first; key <= last; ++key) {
            const size_t ki = ((size_t)(key % a.capacity) * a.hk + kh) * 128 + c;
            float score = sum(q * f(a.k[ki])) * 0.08838834764831845f;
            const int distance = last - key;
            if (distance < 1024) score += f(a.relative[((size_t)row * a.hq + head) * 1024 + distance]);
            const float nm = isnan(m) || isnan(score) ? nanf("") : fmaxf(m, score);
            const float alpha = m == -INFINITY ? 0.f : expf(m - nm), p = expf(score - nm);
            const float v = f(a.v[ki]);
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
        const float sink = sigmoid((m + logf(z)) - f(a.sink[row * a.hq + head]));
        a.out[qi] = b((out * gate) * sink);
    }
}

extern "C" __global__ __launch_bounds__(128) void tf_ik_attention_e2614fe(InklingAttention a) {
    constexpr int D = 128, NK = 64;
    __shared__ __align__(16) bf16 keys[NK][136], values[D][72], probabilities[3][16][NK];
    __shared__ float scores[16][NK], maxima[16], totals[16], alpha[16];
    const int t = threadIdx.x, w = t >> 5, lane = t & 31, g = lane >> 2, sub = lane & 3;
    const int r0 = blockIdx.x * 16, head = blockIdx.y, kh = head / (a.hq / a.hk);
    const int row0 = r0 + g, row1 = row0 + 8;
    const int last = a.pos + min(a.rows, r0 + 16) - 1;
    const int first = a.window ? max(0, a.pos + r0 - a.window) : 0;
    bool unusual_q = false;
    for (int i = t; i < min(16, a.rows - r0) * D; i += 128) {
        const float q = f(a.q[((size_t)(r0 + i / D) * a.hq + head) * D + i % D]);
        unusual_q |= !isfinite(q) || (q != 0.f && fabsf(q) < 0x1p-126f);
    }
    if (__syncthreads_or(unusual_q)) { ik_attention_exception(a, r0, head); return; }
    uint32_t qa[8][4];
    for (int s = 0; s < 8; ++s) {
        const bf16* q0 = a.q + ((size_t)min(row0, a.rows - 1) * a.hq + head) * D + s * 16;
        const bf16* q1 = a.q + ((size_t)min(row1, a.rows - 1) * a.hq + head) * D + s * 16;
        qa[s][0] = *reinterpret_cast<const uint32_t*>(q0 + 2 * sub);
        qa[s][1] = *reinterpret_cast<const uint32_t*>(q1 + 2 * sub);
        qa[s][2] = *reinterpret_cast<const uint32_t*>(q0 + 8 + 2 * sub);
        qa[s][3] = *reinterpret_cast<const uint32_t*>(q1 + 8 + 2 * sub);
    }
    float out[4][4] = {};
    if (t < 16) { maxima[t] = -INFINITY; totals[t] = 0.f; }
    __syncthreads();
    for (int base = first / NK * NK; base <= last; base += NK) {
        bool exceptional = false;
        for (int i = t; i < NK * D; i += 128) {
            const int key = base + i / D, c = i % D;
            const size_t offset = ((size_t)(key % a.capacity) * a.hk + kh) * D + c;
            const bf16 k = key <= last ? a.k[offset] : b(0.f);
            const bf16 v = key <= last ? a.v[offset] : b(0.f);
            keys[i / D][c] = k;
            values[c][i / D] = v;
            exceptional |= !isfinite(f(k)) || !isfinite(f(v)) ||
                (f(k) != 0.f && fabsf(f(k)) < 0x1p-126f) || (f(v) != 0.f && fabsf(f(v)) < 0x1p-126f);
        }
        if (__syncthreads_or(exceptional)) { ik_attention_exception(a, r0, head); return; }
        float sc[2][4] = {};
        for (int n = 0; n < 2; ++n) for (int s = 0; s < 8; ++s) {
            const int key = w * 16 + n * 8 + g;
            ik_mma(sc[n], qa[s], *reinterpret_cast<const uint32_t*>(&keys[key][s * 16 + 2 * sub]),
                   *reinterpret_cast<const uint32_t*>(&keys[key][s * 16 + 8 + 2 * sub]));
        }
        for (int n = 0; n < 2; ++n) for (int half = 0; half < 2; ++half) for (int e = 0; e < 2; ++e) {
            const int row = r0 + g + half * 8, col = w * 16 + n * 8 + 2 * sub + e;
            const int distance = a.pos + row - (base + col);
            float score = sc[n][2 * half + e] * 0.08838834764831845f;
            const bool valid = row < a.rows && distance >= 0 && (!a.window || distance <= a.window);
            if (valid && distance < 1024) score += f(a.relative[((size_t)row * a.hq + head) * 1024 + distance]);
            scores[g + half * 8][col] = valid ? score : -INFINITY;
        }
        __syncthreads();
        const int rr = t / 8, col0 = t % 8;
        float m = maxima[rr];
        for (int col = col0; col < NK; col += 8) {
            const float v = scores[rr][col];
            m = isnan(m) || isnan(v) ? nanf("") : fmaxf(m, v);
        }
        for (int off = 1; off < 8; off <<= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, m, off, 8);
            m = isnan(m) || isnan(other) ? nanf("") : fmaxf(m, other);
        }
        const float aold = maxima[rr] == m ? 1.f : expf(maxima[rr] - m);
        float z = 0.f;
        for (int col = col0; col < NK; col += 8) {
            float p = scores[rr][col] == -INFINITY ? 0.f : expf(scores[rr][col] - m);
            z += p;
            for (int limb = 0; limb < 3; ++limb) {
                const bf16 piece = b(p);
                probabilities[limb][rr][col] = piece;
                exceptional |= f(piece) != 0.f && fabsf(f(piece)) < 0x1p-126f;
                p -= f(piece);
            }
            exceptional |= p != 0.f;
        }
        if (__syncthreads_or(exceptional)) { ik_attention_exception(a, r0, head); return; }
        for (int off = 1; off < 8; off <<= 1) z += __shfl_xor_sync(0xffffffffu, z, off, 8);
        if (!col0) { totals[rr] = totals[rr] * aold + z; maxima[rr] = m; alpha[rr] = aold; }
        __syncthreads();
        for (int n = 0; n < 4; ++n) for (int e = 0; e < 4; ++e) out[n][e] *= alpha[g + (e / 2) * 8];
        for (int s = 0; s < 4; ++s) for (int limb = 0; limb < 3; ++limb) {
            uint32_t p[4];
            p[0] = *reinterpret_cast<const uint32_t*>(&probabilities[limb][g][s * 16 + 2 * sub]);
            p[1] = *reinterpret_cast<const uint32_t*>(&probabilities[limb][g + 8][s * 16 + 2 * sub]);
            p[2] = *reinterpret_cast<const uint32_t*>(&probabilities[limb][g][s * 16 + 8 + 2 * sub]);
            p[3] = *reinterpret_cast<const uint32_t*>(&probabilities[limb][g + 8][s * 16 + 8 + 2 * sub]);
            for (int n = 0; n < 4; ++n) {
                const int c = w * 32 + n * 8 + g;
                ik_mma(out[n], p, *reinterpret_cast<const uint32_t*>(&values[c][s * 16 + 2 * sub]),
                       *reinterpret_cast<const uint32_t*>(&values[c][s * 16 + 8 + 2 * sub]));
            }
        }
        __syncthreads();
    }
    for (int n = 0; n < 4; ++n) for (int half = 0; half < 2; ++half) for (int e = 0; e < 2; ++e) {
        const int row = r0 + g + half * 8, c = w * 32 + n * 8 + 2 * sub + e;
        if (row >= a.rows) continue;
        const size_t at = ((size_t)row * a.hq + head) * D + c;
        const float z = totals[g + half * 8], lse = maxima[g + half * 8] + logf(z);
        const float value = out[n][half * 2 + e] / z;
        a.out[at] = b((value * sigmoid(f(a.gate[at]))) * sigmoid(lse - f(a.sink[row * a.hq + head])));
    }
}
