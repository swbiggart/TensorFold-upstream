//! Inkling-2-Small in the build: its host module (no GPU, no core), its host tests, tf-inkling2-plan; its CUDA
//! module (the forward's op groups over the runtime) and tf-inkling2-cuda, the layer-range runner.
const std = @import("std");

/// The family module: config, quant manifest, tensor roles, the rank plan, safetensors files and the refusal.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/inkling2.zig"), .target = target, .optimize = optimize });
}

/// test-inkling2-host (in `test`) and tf-inkling2-plan, the per-rank memory plan and byte ranges.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug) }));
    run.has_side_effects = true; // it writes a temporary folder
    b.step("test-inkling2-host", "Inkling-2-Small: config, quant manifest, tensor roles and the rank plan against the release's files (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
    const main = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/plan_main.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "inkling2", .module = module(b, target, optimize) }} });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-plan", .root_module = main });
    b.installArtifact(exe);
    b.step("tf-inkling2-plan", "Inkling-2-Small's per-rank memory plan and byte ranges (host tool)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const fold_main = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/fold_main.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "inkling2", .module = module(b, target, optimize) }} });
    const fold = b.addExecutable(.{ .name = "tf-inkling2-fold", .root_module = fold_main });
    b.installArtifact(fold);
    b.step("tf-inkling2-fold", "Inkling-2-Small's DSpark head folds from a checkpoint or rank folder (host tool, no GPU)").dependOn(&b.addInstallArtifact(fold, .{}).step);
}

/// The CUDA module over the runtime and the host module `host`: weights, state, the forward and its op groups.
pub fn cudaModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, host: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2", .module = host },
        .{ .name = "tp", .module = @import("tp.zig").module(b, target, optimize, cuda) }, // TF_INK2_ROCE_MULTI's wire
    } });
}

/// The served family (native registry entry and lane backend) over the CUDA module, the lane core and `host`.
pub fn serveModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, host: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_serve.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "lanes", .module = lanes },
        .{ .name = "inkling2", .module = host },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
    } });
}

/// tf-inkling2-decode (Linux): the serving engine's decode steps against whole-prompt forwards over the same tokens.
fn decodeTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, host: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_decode_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2", .module = host },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-decode", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-decode", "Inkling-2-Small's decode steps (per-stream state) against whole-prompt forwards").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// tf-inkling2-draft (Linux): the DSpark drafter alone on a taps file (the op check against the torch reference).
fn draftTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, host: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_dspark_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2", .module = host },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-draft", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-draft", "Inkling-2-Small's DSpark drafter alone on a taps file (op check)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// tf-inkling2-cuda (Linux): layers [a, b) on one GPU from an oracle prompt file, every layer's rows dumped.
pub fn cudaTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const host = module(b, target, optimize);
    decodeTool(b, target, optimize, strip, cuda, host);
    draftTool(b, target, optimize, strip, cuda, host);
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_run_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2", .module = host },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-cuda", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-cuda", "Inkling-2-Small's layers [a, b) on one GPU against the oracle (full mode)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    attnTool(b, target, optimize, strip, cuda, host, core);
    moeTool(b, target, optimize, strip, cuda, host);
    fuseTool(b, target, optimize, strip, cuda, host);
}

/// tf-inkling2-attn (Linux): G-attn's ops alone over a captured prompt at several forward sizes; --oracle takes
/// core's relbias_attention_ref.zig.
fn attnTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, host: *std.Build.Module, core: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_attn_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2", .module = host },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
        .{ .name = "core", .module = core },
    } });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-attn", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-attn", "Inkling-2-Small's attention ops alone over a captured prompt at several forward sizes").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// tf-inkling2-moe (Linux): G-moe's route and experts alone on one layer, any rows, several forward sizes.
fn moeTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, host: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_moe_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2", .module = host },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-moe", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-moe", "Inkling-2-Small's MoE ops alone on one layer: row invariance across forward sizes").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// tf-inkling2-fuse (Linux): G-fuse's fused ops against the unfused sequences they replace (op check, no weights).
fn fuseTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, host: *std.Build.Module) void {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/cuda_fuse_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "inkling2_cuda", .module = cudaModule(b, target, optimize, cuda, host) },
    } });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-fuse", .root_module = m });
    b.installArtifact(exe);
    b.step("tf-inkling2-fuse", "Inkling-2-Small's fused ops against the unfused sequences (op check, no weights)").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// The CUDA module's host tests (no GPU) in test-inkling2-cuda-host.
pub fn cudaHostTests(b: *std.Build, cuda: *std.Build.Module, lanes: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cudaModule(b, host, .debug, cuda, module(b, host, .debug)) })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = serveModule(b, host, .debug, cuda, lanes, module(b, host, .debug)) })).step);
}
