//! Inkling-2-Small on CUDA (`full` mode): one rank's weights of a layer range on the device, the forward as op-group
//! entry points, and the state it carries. At W > 1 a rank holds its heads and experts (plan.zig) and the forward's
//! two sums a layer go through cuda_tp.zig (real ranks over NCCL, or virtual ranks in one process).
//!
//! Op groups, one file and one kernel image each (the lanes that fill them in):
//!   G-attn  cuda_attn.zig  inkling2_attn.cu   project (done: cuBLASLt), heads, core, output (done: cuBLASLt)
//!   G-conv  cuda_conv.zig  inkling2_conv.cu   kv, post
//!   G-res   cuda_res.zig   inkling2_res.cu    mix, norm (done: norms.cu), partial, output
//!   G-moe   cuda_moe.zig   inkling2_moe.cu    route, experts
//!   G-head  cuda_head.zig  inkling2_head.cu   embed (done), finalNorm (done), unembed (done: cuBLASLt), softcap;
//!                                             the table by vocabulary rows over the ranks (cuda_embed.zig)
//!   G-fuse  cuda_fuse.zig  inkling2_fuse.cu   stateless neighbours as one launch (attnIn, mlpIn, route) and their
//!                                             one-row latency forms (the decode default), same bits
//! The scaffold: cuda_ops.zig (Mat, Ctx), cuda_weights.zig, cuda_state.zig, cuda_forward.zig, and the runner
//! tf-inkling2-cuda (cuda_run_main.zig) with tools/zig/inkling2_compare.py and inkling2_op_ref.py.
pub const ops = @import("cuda_ops.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const weights = @import("cuda_weights.zig");
pub const state = @import("cuda_state.zig");
pub const forward = @import("cuda_forward.zig");
pub const attn = @import("cuda_attn.zig");
pub const conv = @import("cuda_conv.zig");
pub const res = @import("cuda_res.zig");
pub const moe = @import("cuda_moe.zig");
pub const head = @import("cuda_head.zig");
pub const tp = @import("cuda_tp.zig");
pub const wire = @import("cuda_wire.zig");
pub const engine = @import("cuda_engine.zig");
pub const follow = @import("cuda_follow.zig");
pub const dspark = @import("cuda_dspark.zig");
pub const batch_check = @import("cuda_batch_check.zig");
pub const fuse = @import("cuda_fuse.zig");
pub const snap = @import("cuda_snap.zig");
pub const nvme = @import("cuda_nvme.zig");
pub const scratch = @import("cuda_scratch.zig");
pub const mem_check = @import("cuda_mem_check.zig");
pub const argmax_check = @import("cuda_argmax_check.zig");
pub const synth = @import("cuda_synth.zig");
pub const pages = @import("cuda_pages.zig");
pub const embed = @import("cuda_embed.zig");
pub const xover = @import("cuda_xover.zig");
pub const vision = @import("cuda_vision.zig");
pub const image_check = @import("cuda_image_check.zig");

test {
    _ = ops;
    _ = kernels;
    _ = weights;
    _ = state;
    _ = forward;
    _ = attn;
    _ = conv;
    _ = res;
    _ = moe;
    _ = head;
    _ = tp;
    _ = engine;
    _ = follow;
    _ = dspark;
    _ = argmax_check;
    _ = fuse;
    _ = snap;
    _ = nvme;
    _ = scratch;
    _ = wire;
    _ = synth;
    _ = pages;
    _ = embed;
    _ = vision;
    _ = image_check;
}
