// Inkling-2's convolution group (G-conv): the four short depthwise 4-tap causal convolutions of a layer (static,
// and static plus per-token taps in global layers' K and V) over a stream's 3 rows of state. The entry points,
// shapes and math are in zig/src/families/inkling2/cuda_conv.zig; their kernels go here.
//
// Math: Transformers modeling_inkling2.py (Apache-2.0), Inkling2ShortConvolution and
// Inkling2DynamicShortConvolution, in the fp32 steps torch runs them on CUDA (checked bit for bit against
// the op captures): the static taps as one fused multiply-add chain, oldest row first (cuDNN's depthwise conv1d
// with TF32 off; every order tried rounds to the same bf16 on the captures); the dynamic taps as fp32(static) +
// fp32(dynamic), each product rounded, summed as torch's sum(-1) over the 4 lags sums them, (p0 + p2) + (p1 + p3)
// with lag 0 the current row (fp32-equal to torch's sum on every global layer's K and V checked, 5, 11, 41 and 47,
// where the 26 other orders are not); then (sum + x_t) * inv, where inv = fp32(1) / fp32(sqrt(5 or 9)) (torch's CUDA division
// by a Python scalar multiplies by its fp32 reciprocal); one bf16 rounding.
//
// A thread owns two adjacent channels and walks RT rows of the forward, carrying the three rows before each one in
// registers: a row's bits depend only on its own four input rows and taps, never on the row count or the tiling.
// Rows before the forward's first come from the stream's state S [3, C] (row 2 = the input one position back).

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int NT = 128;  // threads a block
constexpr int RT = 16;   // rows a thread walks

__device__ __forceinline__ float2 pair(const __nv_bfloat16* p) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

// Input row r of the forward (r >= 0) or, before it, state row 3 + r; channels c, c + 1.
__device__ __forceinline__ float2 input(const __nv_bfloat16* __restrict__ X, long long ldx,
                                        const __nv_bfloat16* __restrict__ S, int C, int r, int c) {
    return r >= 0 ? pair(X + static_cast<int64_t>(r) * ldx + c) : pair(S + static_cast<int64_t>(3 + r) * C + c);
}

__device__ __forceinline__ void store(__nv_bfloat16* p, float a, float b) {
    *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

}  // namespace

// The module's presence check (cuda_kernels.zig Group): writes the group's number.
extern "C" __global__ void tf_ink2_conv_probe(int* out) {
    if (out != nullptr && threadIdx.x == 0 && blockIdx.x == 0) *out = 2;
}

// Static form: Y[t, c] = bf16(((((W[c,0] x_{t-3} + W[c,1] x_{t-2}) + W[c,2] x_{t-1}) + W[c,3] x_t) + x_t) * inv),
// W bf16 [C, 4] (conv1d weight, tap 3 = the current row). X [R, C] ld ldx, Y [R, C] ld ldy, S [3, C].
// Grid (ceil(C / 2NT), ceil(R / RT)), NT threads.
extern "C" __global__ void __launch_bounds__(NT) tf_ink2_conv_static(
    const __nv_bfloat16* __restrict__ X, long long ldx, const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ Y, long long ldy, int R, int C, float inv) {
    const int c = 2 * (blockIdx.x * NT + threadIdx.x);
    const int r0 = blockIdx.y * RT;
    if (c >= C || r0 >= R) return;
    const int r1 = min(R, r0 + RT);
    float wa[4], wb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        wa[j] = __bfloat162float(W[static_cast<int64_t>(c) * 4 + j]);
        wb[j] = __bfloat162float(W[static_cast<int64_t>(c + 1) * 4 + j]);
    }
    float2 h0 = input(X, ldx, S, C, r0 - 3, c), h1 = input(X, ldx, S, C, r0 - 2, c), h2 = input(X, ldx, S, C, r0 - 1, c);
    for (int t = r0; t < r1; ++t) {
        const float2 x = input(X, ldx, S, C, t, c);
        float a = __fmul_rn(wa[0], h0.x), b = __fmul_rn(wb[0], h0.y);
        a = __fmaf_rn(wa[1], h1.x, a);
        b = __fmaf_rn(wb[1], h1.y, b);
        a = __fmaf_rn(wa[2], h2.x, a);
        b = __fmaf_rn(wb[2], h2.y, b);
        a = __fmaf_rn(wa[3], x.x, a);
        b = __fmaf_rn(wb[3], x.y, b);
        store(Y + static_cast<int64_t>(t) * ldy + c, __fmul_rn(__fadd_rn(a, x.x), inv), __fmul_rn(__fadd_rn(b, x.y), inv));
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
}

// Dynamic form (global layers' K and V): per lag w = fp32(SW[lag, c]) + fp32(D[t, (c / 128) * 4 + lag]);
// p_lag = x_{t-lag} w_lag; Y[t, c] = bf16((((p0 + p2) + (p1 + p3)) + x_t) * inv), every product and sum rounded.
// SW bf16 [4, C] (lag 0 = the current row), D bf16 [R, (C / 128) * 4] ld ldd (weight_proj's output). Grid as above.
extern "C" __global__ void __launch_bounds__(NT) tf_ink2_conv_dynamic(
    const __nv_bfloat16* __restrict__ X, long long ldx, const __nv_bfloat16* __restrict__ S,
    const __nv_bfloat16* __restrict__ SW, const __nv_bfloat16* __restrict__ D, long long ldd,
    __nv_bfloat16* __restrict__ Y, long long ldy, int R, int C, float inv) {
    const int c = 2 * (blockIdx.x * NT + threadIdx.x);
    const int r0 = blockIdx.y * RT;
    if (c >= C || r0 >= R) return;
    const int r1 = min(R, r0 + RT);
    const int head = c >> 7;  // 128 channels a head; c and c + 1 share it
    float sa[4], sb[4];
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const float2 s = pair(SW + static_cast<int64_t>(lag) * C + c);
        sa[lag] = s.x;
        sb[lag] = s.y;
    }
    float2 h0 = input(X, ldx, S, C, r0 - 3, c), h1 = input(X, ldx, S, C, r0 - 2, c), h2 = input(X, ldx, S, C, r0 - 1, c);
    for (int t = r0; t < r1; ++t) {
        const float2 x = input(X, ldx, S, C, t, c);
        const __nv_bfloat16* d = D + static_cast<int64_t>(t) * ldd + head * 4;
        float dl[4];
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) dl[lag] = __bfloat162float(d[lag]);
        float pa[4], pb[4];  // the lag products, lag 0 = the current row
        pa[0] = __fmul_rn(x.x, __fadd_rn(sa[0], dl[0]));
        pb[0] = __fmul_rn(x.y, __fadd_rn(sb[0], dl[0]));
        pa[1] = __fmul_rn(h2.x, __fadd_rn(sa[1], dl[1]));
        pb[1] = __fmul_rn(h2.y, __fadd_rn(sb[1], dl[1]));
        pa[2] = __fmul_rn(h1.x, __fadd_rn(sa[2], dl[2]));
        pb[2] = __fmul_rn(h1.y, __fadd_rn(sb[2], dl[2]));
        pa[3] = __fmul_rn(h0.x, __fadd_rn(sa[3], dl[3]));
        pb[3] = __fmul_rn(h0.y, __fadd_rn(sb[3], dl[3]));
        const float a = __fadd_rn(__fadd_rn(pa[0], pa[2]), __fadd_rn(pa[1], pa[3]));
        const float b = __fadd_rn(__fadd_rn(pb[0], pb[2]), __fadd_rn(pb[1], pb[3]));
        store(Y + static_cast<int64_t>(t) * ldy + c, __fmul_rn(__fadd_rn(a, x.x), inv), __fmul_rn(__fadd_rn(b, x.y), inv));
        h0 = h1;
        h1 = h2;
        h2 = x;
    }
}

// The state commit after a forward (cuda_state.zig convCommit): state bf16 [3, C] (dense) becomes the last 3 rows of
// (state rows, then x rows 0 .. R - 1), x bf16 [R, C] with row stride ldx elements. A thread a column, so the shift is
// in place. One launch instead of a copy a row.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_conv_commit(
    unsigned short* __restrict__ state, const unsigned short* __restrict__ x, long long ldx, int R, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const unsigned short s[3] = {state[c], state[C + c], state[2 * C + c]};
    unsigned short o[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int j = R + i;  // index in (state, x)
        o[i] = j < 3 ? s[j] : x[(size_t)(j - 3) * ldx + c];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) state[(size_t)i * C + c] = o[i];
}

// A verify window's commit on a slot with drafts (cuda_state.zig convCommitSpec): hist bf16 [3 + R, C] (dense)
// becomes (state rows, then x rows 0 .. R - 1) and state bf16 [3, C] the last 3 rows of it; x bf16 [R, C] (R <= 8)
// with row stride ldx elements. A thread a column reads all it needs before it writes, so the bytes are those of the
// three copies this replaces (state to hist, x after it, hist's last 3 rows back to state), in one launch.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_conv_commit_spec(
    unsigned short* __restrict__ state, unsigned short* __restrict__ hist, const unsigned short* __restrict__ x,
    long long ldx, int R, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || R < 1 || R > 8) return;
    unsigned short h[11];
#pragma unroll
    for (int i = 0; i < 3; ++i) h[i] = state[(size_t)i * C + c];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (i < R) h[3 + i] = x[(size_t)i * ldx + c];
#pragma unroll
    for (int i = 0; i < 11; ++i)
        if (i < 3 + R) hist[(size_t)i * C + c] = h[i];
#pragma unroll
    for (int i = 1; i <= 8; ++i)
        if (i == R) {
#pragma unroll
            for (int j = 0; j < 3; ++j) state[(size_t)j * C + c] = h[i + j];
        }
}

// A convolution and its state commit as one launch, for conv.kv's K and V (cuda_conv.zig kvCommit: blockIdx.z 0 is
// K, 1 is V) or conv.attn / conv.mlp (postCommit: one, static); a forward of R <= RT rows, so one thread walks every
// row of its two channels. Each thread runs tf_ink2_conv_static's or
// tf_ink2_conv_dynamic's expressions for its channels (the same bits), having read the state rows it needs, then
// writes the commit's bytes for its channels: the state becomes the last 3 rows of (state, x rows 0 .. R - 1)
// (tf_ink2_conv_commit), or, with a history (H not null, R <= 8), the history becomes (state, x) and the state its
// rows R .. R + 2 (tf_ink2_conv_commit_spec). A thread reads all it needs before it writes; no other thread reads
// its channels.
struct KvCommit {
    const __nv_bfloat16* X[2];
    long long ldx[2];
    __nv_bfloat16* S[2];
    const __nv_bfloat16* W[2];
    const __nv_bfloat16* D[2];  // the dynamic form's per-token taps (global layers), else null
    long long ldd[2];
    __nv_bfloat16* Y[2];
    long long ldy[2];
    __nv_bfloat16* H[2];  // the history of a slot with drafts, else null
};

// One convolution and its commit for channels c, c + 1 (tf_ink2_conv_kv_commit's body): X [R, C] ld ldx, the state
// S [3, C], taps W (static [C, 4], or with D not null the dynamic form's SW [4, C] and D [R, C / 128 * 4] ld ldd),
// Y [R, C] ld ldy, the history Hs or null. Both kernels below run it, so their arithmetic is the same code.
__device__ __forceinline__ void conv_commit(const __nv_bfloat16* __restrict__ X, long long ldx, __nv_bfloat16* __restrict__ S,
                                            const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ D,
                                            long long ldd, __nv_bfloat16* __restrict__ Y, long long ldy,
                                            __nv_bfloat16* Hs, int R, int C, float inv, int c) {
    float2 h0 = input(X, ldx, S, C, -3, c), h1 = input(X, ldx, S, C, -2, c), h2 = input(X, ldx, S, C, -1, c);
    if (D == nullptr) {
        float wa[4], wb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wa[j] = __bfloat162float(W[static_cast<int64_t>(c) * 4 + j]);
            wb[j] = __bfloat162float(W[static_cast<int64_t>(c + 1) * 4 + j]);
        }
        for (int t = 0; t < R; ++t) {
            const float2 x = input(X, ldx, S, C, t, c);
            float a = __fmul_rn(wa[0], h0.x), b = __fmul_rn(wb[0], h0.y);
            a = __fmaf_rn(wa[1], h1.x, a);
            b = __fmaf_rn(wb[1], h1.y, b);
            a = __fmaf_rn(wa[2], h2.x, a);
            b = __fmaf_rn(wb[2], h2.y, b);
            a = __fmaf_rn(wa[3], x.x, a);
            b = __fmaf_rn(wb[3], x.y, b);
            store(Y + static_cast<int64_t>(t) * ldy + c, __fmul_rn(__fadd_rn(a, x.x), inv), __fmul_rn(__fadd_rn(b, x.y), inv));
            h0 = h1;
            h1 = h2;
            h2 = x;
        }
    } else {
        const __nv_bfloat16* __restrict__ SW = W;
        const int head = c >> 7;
        float sa[4], sb[4];
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) {
            const float2 sw = pair(SW + static_cast<int64_t>(lag) * C + c);
            sa[lag] = sw.x;
            sb[lag] = sw.y;
        }
        for (int t = 0; t < R; ++t) {
            const float2 x = input(X, ldx, S, C, t, c);
            const __nv_bfloat16* d = D + static_cast<int64_t>(t) * ldd + head * 4;
            float dl[4];
#pragma unroll
            for (int lag = 0; lag < 4; ++lag) dl[lag] = __bfloat162float(d[lag]);
            float pa[4], pb[4];
            pa[0] = __fmul_rn(x.x, __fadd_rn(sa[0], dl[0]));
            pb[0] = __fmul_rn(x.y, __fadd_rn(sb[0], dl[0]));
            pa[1] = __fmul_rn(h2.x, __fadd_rn(sa[1], dl[1]));
            pb[1] = __fmul_rn(h2.y, __fadd_rn(sb[1], dl[1]));
            pa[2] = __fmul_rn(h1.x, __fadd_rn(sa[2], dl[2]));
            pb[2] = __fmul_rn(h1.y, __fadd_rn(sb[2], dl[2]));
            pa[3] = __fmul_rn(h0.x, __fadd_rn(sa[3], dl[3]));
            pb[3] = __fmul_rn(h0.y, __fadd_rn(sb[3], dl[3]));
            const float a = __fadd_rn(__fadd_rn(pa[0], pa[2]), __fadd_rn(pa[1], pa[3]));
            const float b = __fadd_rn(__fadd_rn(pb[0], pb[2]), __fadd_rn(pb[1], pb[3]));
            store(Y + static_cast<int64_t>(t) * ldy + c, __fmul_rn(__fadd_rn(a, x.x), inv), __fmul_rn(__fadd_rn(b, x.y), inv));
            h0 = h1;
            h1 = h2;
            h2 = x;
        }
    }
    // the commit: copies of the raw bf16 pairs (state rows, then input rows)
    const uint32_t* S32 = reinterpret_cast<const uint32_t*>(S);
    uint32_t* S32w = reinterpret_cast<uint32_t*>(S);
    const int c2 = c / 2, C2 = C / 2;
    uint32_t st[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) st[i] = S32[(size_t)i * C2 + c2];  // nothing has written them yet
    const uint32_t* X32 = reinterpret_cast<const uint32_t*>(X);
    auto row = [&](int j) -> uint32_t {  // row j of (state, x)
        return j < 3 ? st[j] : X32[((size_t)(j - 3) * ldx + c) / 2];
    };
    if (Hs == nullptr) {
        uint32_t o[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) o[i] = row(R + i);
#pragma unroll
        for (int i = 0; i < 3; ++i) S32w[(size_t)i * C2 + c2] = o[i];
    } else {
        if (R > 8) return;
        uint32_t* H32 = reinterpret_cast<uint32_t*>(Hs);
        uint32_t h[11];
#pragma unroll
        for (int i = 0; i < 11; ++i)
            if (i < 3 + R) h[i] = row(i);
#pragma unroll
        for (int i = 0; i < 11; ++i)
            if (i < 3 + R) H32[(size_t)i * C2 + c2] = h[i];
#pragma unroll
        for (int i = 1; i <= 8; ++i)
            if (i == R) {
#pragma unroll
                for (int j = 0; j < 3; ++j) S32w[(size_t)j * C2 + c2] = h[i + j];
            }
    }
}

extern "C" __global__ void __launch_bounds__(NT) tf_ink2_conv_kv_commit(KvCommit p, int R, int C, float inv) {
    const int z = blockIdx.z;
    const int c = 2 * (blockIdx.x * NT + threadIdx.x);
    if (c >= C || R < 1 || R > RT) return;
    conv_commit(p.X[z], p.ldx[z], p.S[z], p.W[z], p.D[z], p.ldd[z], p.Y[z], p.ldy[z], p.H[z], R, C, inv, c);
}

// tf_ink2_conv_kv_commit for every segment of a batched forward (cuda_conv.zig kvCommitSegs, postCommitSegs):
// blockIdx.y is segment g, rows row0[g] .. row0[g] + rows[g] - 1 of the shared X, D and Y, against its own state
// S[g] and history H[g]. A segment of no rows, or of more than the fused form takes, does nothing (the host sends
// none). Segments never share a state and their rows are disjoint, so this is the launches a segment in one.
constexpr int MS = 8;  // segments a launch
struct KvCommitSegs {
    const __nv_bfloat16* X[2];
    long long ldx[2];
    const __nv_bfloat16* W[2];
    const __nv_bfloat16* D[2];
    long long ldd[2];
    __nv_bfloat16* Y[2];
    long long ldy[2];
    __nv_bfloat16* S[MS][2];
    __nv_bfloat16* H[MS][2];
    int row0[MS];
    int rows[MS];
};

extern "C" __global__ void __launch_bounds__(NT) tf_ink2_conv_kv_commit_segs(KvCommitSegs p, int C, float inv) {
    const int z = blockIdx.z;
    const int g = blockIdx.y;
    const int c = 2 * (blockIdx.x * NT + threadIdx.x);
    const int R = p.rows[g];
    if (c >= C || R < 1 || R > RT || (p.H[g][z] != nullptr && R > 8)) return;
    const long long r0 = p.row0[g];
    const __nv_bfloat16* D = p.D[z] == nullptr ? nullptr : p.D[z] + r0 * p.ldd[z];
    conv_commit(p.X[z] + r0 * p.ldx[z], p.ldx[z], p.S[g][z], p.W[z], D, p.ldd[z], p.Y[z] + r0 * p.ldy[z], p.ldy[z],
                p.H[g][z], R, C, inv, c);
}

// One convolution of a slot's `keep` table (cuda_state.zig KeepEntry): its state bf16 [3, cols] and its history
// bf16 [3 + 8, cols] (the state before the window, then the window's input rows).
struct KeepEntry {
    unsigned long long state;
    unsigned long long hist;
    int cols;
    int pad;
};

// A slot's rollback (cuda_engine.zig keep, cuda_state.zig keepAll): every convolution of every layer back to after
// the window's first `keep` rows, state = hist rows keep .. keep + 2, blockIdx.y one entry of `table`. One launch for
// the slot instead of a copy a convolution; the same bytes.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_conv_keep(const KeepEntry* __restrict__ table, int keep) {
    const KeepEntry e = table[blockIdx.y];
    unsigned short* s = reinterpret_cast<unsigned short*>(e.state);
    const unsigned short* h = reinterpret_cast<const unsigned short*>(e.hist) + (size_t)keep * e.cols;
    const int n = 3 * e.cols;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s[i] = h[i];
}
