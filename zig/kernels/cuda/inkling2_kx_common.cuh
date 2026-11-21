// Copyright kortexa-ai. Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
// From kortexa-ai/TensorFold e2614fe (zig/kernels/cuda/inkling2_common.cuh); modified: this header only.
// Inkling 2's FP32 reductions and BF16 storage boundaries. No contracted operations.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <math.h>

namespace inkling2 {
using bf16 = __nv_bfloat16;
__device__ inline float f(bf16 x) { return __bfloat162float(x); }
__device__ inline bf16 b(float x) { return __float2bfloat16_rn(x); }
__device__ inline float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return __shfl_sync(0xffffffffu, x, 0);
}
__device__ inline float sum(float x) {
    __shared__ float partial[32], result;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (!lane) partial[warp] = x;
    __syncthreads();
    if (!warp) {
        x = lane < blockDim.x / 32 ? partial[lane] : 0.f;
        x = warp_sum(x);
        if (!lane) result = x;
    }
    __syncthreads();
    return result;
}
__device__ inline float maximum(float x) {
    __shared__ float partial[32], result;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int d = 16; d; d >>= 1) {
        const float y = __shfl_down_sync(0xffffffffu, x, d);
        x = isnan(x) || isnan(y) ? nanf("") : fmaxf(x, y);
    }
    if (!lane) partial[warp] = x;
    __syncthreads();
    if (!warp) {
        x = lane < blockDim.x / 32 ? partial[lane] : 0.f;
        for (int d = 16; d; d >>= 1) {
            const float y = __shfl_down_sync(0xffffffffu, x, d);
            x = isnan(x) || isnan(y) ? nanf("") : fmaxf(x, y);
        }
        if (!lane) result = x;
    }
    __syncthreads();
    return result;
}
__device__ inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
}
