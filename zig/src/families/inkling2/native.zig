//! What the native host says for an Inkling-2-Small checkpoint while the family has no forward: the checkpoint is
//! recognized and checked (config, quant manifest, every tensor's role, the rank plan at the requested world size),
//! then refused by name rather than as an unknown model.
const std = @import("std");
const config = @import("config.zig");
const describe = @import("describe.zig");
const Diag = @import("diag.zig").Diag;

pub const model_type = config.model_type;
pub const architecture = config.architecture;
pub const title = "Inkling-2-Small";

pub fn recognizes(mt: []const u8) bool {
    return std.mem.eql(u8, mt, model_type);
}

/// The open's problem for `dir` at `tp` ranks: what was read and checked, then that the forward is not built yet.
pub fn refusal(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, tp: u32) ![]const u8 {
    const not_yet = "the inkling2 family's forward is not implemented yet (family skeleton: tf-inkling2-plan prints the rank plan)";
    var d: Diag = .{};
    var x = describe.Model.fromDir(gpa, io, dir, .{ .world = @max(tp, 1) }, &d) catch |e| {
        if (e == error.OutOfMemory) return e;
        return std.fmt.allocPrint(a, "{s} ({s}, {s}): {s}: {s}", .{ title, model_type, architecture, if (d.len > 0) d.text() else @errorName(e), not_yet });
    };
    defer x.deinit();
    var most: u64 = 0;
    for (x.p.ranks) |r| most = @max(most, r.total());
    return std.fmt.allocPrint(a, "{s} ({s}, {s}): config, quant manifest and {d} tensors checked; at --tp {d} the largest rank holds {d:.1} GiB of weights; {s}", .{
        title, model_type, architecture, x.inv.tensors.len, x.p.split.world, @as(f64, @floatFromInt(most)) / describe.gib, not_yet,
    });
}

test "inkling2 native: the refusal names the family and the missing forward, even for an unreadable folder" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const why = try refusal(arena.allocator(), std.testing.allocator, std.testing.io, "/nonexistent-inkling2-folder", 4);
    try std.testing.expect(std.mem.indexOf(u8, why, "Inkling-2-Small (inkling2, Inkling2ForConditionalGeneration)") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "config.json") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "not implemented yet") != null);
    try std.testing.expect(recognizes("inkling2") and !recognizes("inkling2_text"));
}
