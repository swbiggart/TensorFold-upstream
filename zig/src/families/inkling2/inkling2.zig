//! Inkling-2-Small (thinkingmachines, model_type inkling2, Inkling2ForConditionalGeneration): the family skeleton.
//!
//! What is here: the config (text, vision, drafter), the quant manifest (quant_method inkling2), every checkpoint
//! tensor's role and format, the TP attention / EP experts rank plan as byte ranges inside the safetensors, the
//! per-rank memory table, safetensors files for the CUDA runner (st.zig), and the native host's refusal until the
//! served forward exists. The CUDA forward (Phase 1) is the separate module cuda.zig.
pub const config = @import("config.zig");
pub const quant = @import("quant.zig");
pub const inventory = @import("inventory.zig");
pub const roles = @import("roles.zig");
pub const plan = @import("plan.zig");
pub const describe = @import("describe.zig");
pub const digests = @import("digests.zig");
pub const native = @import("native.zig");
pub const st = @import("st.zig");
pub const fold = @import("fold.zig");
pub const Diag = @import("diag.zig").Diag;

test {
    _ = config;
    _ = quant;
    _ = inventory;
    _ = roles;
    _ = plan;
    _ = describe;
    _ = digests;
    _ = native;
    _ = st;
    _ = fold;
}
