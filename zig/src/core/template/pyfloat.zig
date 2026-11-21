//! Python's float() on a str, the conversion Jinja2's ``float`` filter makes before it falls back to its default.
const std = @import("std");
const uni = @import("unicode.zig");
const Allocator = std.mem.Allocator;

/// The zero of every run of ten Unicode decimal digits (Nd) Python 3.12 reads in float(), U+0030 first.
const zeros = [_]u21{
    0x30,    0x660,   0x6f0,   0x7c0,   0x966,   0x9e6,   0xa66,   0xae6,   0xb66,   0xbe6,   0xc66,   0xce6,
    0xd66,   0xde6,   0xe50,   0xed0,   0xf20,   0x1040,  0x1090,  0x17e0,  0x1810,  0x1946,  0x19d0,  0x1a80,
    0x1a90,  0x1b50,  0x1bb0,  0x1c40,  0x1c50,  0xa620,  0xa8d0,  0xa900,  0xa9d0,  0xa9f0,  0xaa50,  0xabf0,
    0xff10,  0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0,
    0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8,
    0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0,
};

fn decimal(cp: u21) ?u8 {
    for (zeros) |z| if (cp >= z and cp < z + 10) return @intCast(cp - z);
    return null;
}

fn asciiSpace(c: u8) bool {
    return c == ' ' or (c >= 0x09 and c <= 0x0d);
}

/// Digits with single underscores between them, from `i`; false when the run is empty or an underscore is misplaced.
fn digits(s: []const u8, i: *usize) bool {
    const start = i.*;
    while (i.* < s.len) : (i.* += 1) {
        if (std.ascii.isDigit(s[i.*])) continue;
        if (s[i.*] == '_' and i.* > start and i.* + 1 < s.len and std.ascii.isDigit(s[i.* + 1]) and std.ascii.isDigit(s[i.* - 1])) continue;
        break;
    }
    return i.* > start;
}

/// Python's float literal grammar after the sign: digits, a point, an exponent; or inf, infinity, nan.
fn valid(s: []const u8) bool {
    var i: usize = 0;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
    for ([_][]const u8{ "inf", "infinity", "nan" }) |word| if (std.ascii.eqlIgnoreCase(s[i..], word)) return true;
    var any = digits(s, &i);
    if (i < s.len and s[i] == '.') {
        i += 1;
        if (i < s.len and std.ascii.isDigit(s[i])) any = digits(s, &i) or any;
    }
    if (!any) return false;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        if (!digits(s, &i)) return false;
    }
    return i == s.len;
}

/// ``float(s)``, or null where Python raises ValueError: Unicode spaces and decimal digits read as ASCII ones.
pub fn parse(a: Allocator, s: []const u8) Allocator.Error!?f64 {
    var ascii: std.ArrayList(u8) = .empty;
    defer ascii.deinit(a);
    var i: usize = 0;
    while (i < s.len) {
        const cp = uni.next(s, &i);
        if (cp < 0x7f) {
            try ascii.append(a, @intCast(cp));
        } else if (uni.isSpace(cp)) {
            try ascii.append(a, ' ');
        } else if (decimal(cp)) |d| {
            try ascii.append(a, '0' + d);
        } else return null;
    }
    var text: []const u8 = ascii.items;
    while (text.len > 0 and asciiSpace(text[0])) text = text[1..];
    while (text.len > 0 and asciiSpace(text[text.len - 1])) text = text[0 .. text.len - 1];
    if (!valid(text)) return null;
    var clean: std.ArrayList(u8) = .empty;
    defer clean.deinit(a);
    for (text) |c| if (c != '_') try clean.append(a, c);
    return std.fmt.parseFloat(f64, clean.items) catch null;
}

test "python float of a str" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, ?f64 }{
        .{ "0.5", 0.5 },         .{ " 1_0.5 ", 10.5 }, .{ "1__0", null },     .{ "abc", null },     .{ ".5", 0.5 },
        .{ "5.", 5.0 },          .{ "1e5_0", 1e50 },   .{ "0x10", null },     .{ "\u{661}\u{662}", 12 }, .{ "+.5e-1", 0.05 },
        .{ ".", null },          .{ "e5", null },      .{ "1e", null },       .{ "_1", null },      .{ "1_", null },
        .{ "\u{3000}7\u{a0}", 7 }, .{ "\x1c1", null }, .{ "", null },         .{ "1 2", null },     .{ "1.5_5", 1.55 },
    };
    for (cases) |c| try std.testing.expectEqual(c[1], try parse(a, c[0]));
    try std.testing.expect(std.math.isInf((try parse(a, "-InFinity")).?));
    try std.testing.expect(std.math.isInf((try parse(a, "1e400")).?));
    try std.testing.expect(std.math.isNan((try parse(a, "nan")).?));
}
