//! Why a check refused: one line the caller prints or hands back as the open's problem.
const std = @import("std");

pub const Diag = struct {
    buf: [400]u8 = undefined,
    len: usize = 0,

    pub fn set(d: *Diag, comptime fmt: []const u8, args: anytype) void {
        const out = std.fmt.bufPrint(&d.buf, fmt, args) catch d.buf[0..];
        d.len = out.len;
    }

    pub fn text(d: *const Diag) []const u8 {
        return d.buf[0..d.len];
    }
};

/// `d.set(fmt, args)` when `d` is given; always returns `err` so callers write `return fail(d, error.X, ...)`.
pub fn fail(d: ?*Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    if (d) |x| x.set(fmt, args);
    return err;
}
