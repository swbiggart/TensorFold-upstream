//! Every kernel the Inkling-2 forward launches: the upstream ones the scaffold uses (norms.cu's RMS rows,
//! embed_rows.cu's bf16 table rows) and one module a group (inkling2_<group>.cu, each group's `Kernels`).
const std = @import("std");
const cuda = @import("cuda");
const attn = @import("cuda_attn.zig");
const conv = @import("cuda_conv.zig");
const res = @import("cuda_res.zig");
const moe = @import("cuda_moe.zig");
const head = @import("cuda_head.zig");
const fuse = @import("cuda_fuse.zig");

pub const Kernels = struct {
    norms: cuda.norms.Norms,
    embed: cuda.embed_rows.EmbedRows,
    attn: attn.Kernels,
    conv: conv.Kernels,
    res: res.Kernels,
    moe: moe.Kernels,
    head: head.Kernels,
    fuse: fuse.Kernels,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var k: Kernels = undefined;
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.attn = try attn.Kernels.load(d);
        errdefer k.attn.unload();
        k.conv = try conv.Kernels.load(d);
        errdefer k.conv.unload();
        k.res = try res.Kernels.load(d);
        errdefer k.res.unload();
        k.moe = try moe.Kernels.load(d);
        errdefer k.moe.unload();
        k.head = try head.Kernels.load(d);
        errdefer k.head.unload();
        k.fuse = try fuse.Kernels.load(d);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.fuse.unload();
        k.head.unload();
        k.moe.unload();
        k.res.unload();
        k.conv.unload();
        k.attn.unload();
        k.embed.unload();
        k.norms.unload();
        k.* = undefined;
    }
};

/// One group's module and its probe (each inkling2_<group>.cu starts with `tf_ink2_<group>_probe`).
pub fn Group(comptime name: []const u8) type {
    return struct {
        mod: cuda.Module,
        probe: cuda.Function,

        pub fn load(d: *const cuda.Driver) !@This() {
            if (!cuda.kernels.available) return error.BuiltWithoutKernels;
            var mod = try cuda.Module.load(d, @field(cuda.kernels, "inkling2_" ++ name));
            errdefer mod.unload();
            return .{ .mod = mod, .probe = try mod.function("tf_ink2_" ++ name ++ "_probe") };
        }

        pub fn unload(g: *@This()) void {
            g.mod.unload();
        }
    };
}
