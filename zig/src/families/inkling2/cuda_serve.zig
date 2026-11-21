//! Inkling-2-Small served on CUDA: the native registry's entry (cuda_native.zig) over the lane backend
//! (cuda_lanes.zig, its sampler sample.zig) and the rank engine (inkling2_cuda: cuda_engine.zig, cuda_follow.zig).
pub const native = @import("cuda_native.zig");
pub const lanes = @import("cuda_lanes.zig");
pub const sample = @import("sample.zig");
pub const image_offer = @import("image_offer.zig");

test {
    _ = native;
    _ = lanes;
    _ = sample;
    _ = image_offer;
}
