//! Inkling 2's reply and call grammars through a real libtfgrammar (zig build test-grammar-tools): a character
//! vocabulary with its marker tokens as added tokens, every reply walked a token at a time against the masks.
const std = @import("std");
const sys = @import("sys");
const lanes = @import("lanes");
const json = @import("json");
const built = @import("grammar_library");
const inkling2 = @import("inkling2_grammar.zig");

const ascii = blk: {
    var s: [95]u8 = undefined;
    for (&s, 0..) |*c, i| c.* = @intCast(0x20 + i);
    break :blk s;
};

/// Ids 0..94 printable ASCII, 95 a newline, then the added tokens: the markers, two stops and an unused one.
const text_tokens = ascii.len + 1;
const markers = [_][]const u8{
    "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>",
    "<|content_model_end_sampling|>", "<|message_user|>", "<|unused|>",
};
const mm: u32 = text_tokens;
const thinking: u32 = text_tokens + 1;
const content: u32 = text_tokens + 2;
const invoke: u32 = text_tokens + 3;
const end: u32 = text_tokens + 4;
const stop: u32 = text_tokens + 5;
const user: u32 = text_tokens + 6;
const vocab = text_tokens + markers.len;
const words = (vocab + 31) / 32;

const added = blk: {
    var ids: [markers.len]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = text_tokens + i;
    break :blk ids;
};
const grammar_ids: inkling2.Ids = .{ .message_model = mm, .content_thinking = thinking, .content_text = content, .invoke_tool_json = invoke, .end_message = end, .added = &added };

const Lib = struct {
    f: lanes.grammar.Library,
    h: *anyopaque,

    fn open(a: std.mem.Allocator) !Lib {
        var dl = try sys.DynLib.open(built.library);
        var f: lanes.grammar.Library = undefined;
        const info = @typeInfo(lanes.grammar.Library).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(f, name) = if (@typeInfo(T) == .optional) dl.lookup(@typeInfo(T).optional.child, "tfg_" ++ name) else dl.lookup(T, "tfg_" ++ name) orelse return error.MissingSymbol;
        }
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(a, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (0..vocab) |i| {
            if (i > 0) try text.append(a, ',');
            const piece: []const u8 = if (i < ascii.len) ascii[i..][0..1] else if (i == ascii.len) "\n" else markers[i - text_tokens];
            try text.print(a, "{f}:{d}", .{ std.json.fmt(piece, .{}), i });
        }
        try text.appendSlice(a, "}},\"decoder\":{\"type\":\"Fuse\"},\"added_tokens\":[");
        for (markers, 0..) |m, i| try text.print(a, "{s}{{\"id\":{d},\"content\":\"{s}\",\"special\":true}}", .{ if (i > 0) "," else "", text_tokens + i, m });
        try text.appendSlice(a, "]}");
        var err: [512]u8 = undefined;
        const stops = [_]i32{ stop, user };
        const h = f.open(text.items.ptr, text.items.len, vocab, &stops, stops.len, &err, err.len) orelse return error.GrammarSetup;
        return .{ .f = f, .h = h };
    }

    fn compile(l: Lib, text: []const u8) !*anyopaque {
        var err: [512]u8 = undefined;
        err[0] = 0;
        return l.f.compile(l.h, @backingInt(lanes.grammar.Kind.structural_tag), text.ptr, text.len, &err, err.len) orelse {
            std.debug.print("compile: {s}\n{s}\n", .{ std.mem.sliceTo(&err, 0), text });
            return error.Grammar;
        };
    }
};

/// A reply's ids: a marker's spelling is its token, other text a token a character.
fn idsOf(a: std.mem.Allocator, pieces: []const []const u8) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    for (pieces) |p| {
        if (for (markers, 0..) |m, i| {
            if (std.mem.eql(u8, m, p)) break i;
        } else null) |i| {
            try out.append(a, @intCast(text_tokens + i));
            continue;
        }
        for (p) |c| try out.append(a, if (c == '\n') ascii.len else c - 0x20);
    }
    return out.items;
}

const Walk = struct {
    l: Lib,
    m: *anyopaque,

    fn init(l: Lib, g: *anyopaque) !Walk {
        return .{ .l = l, .m = l.f.matcher(g) orelse return error.Matcher };
    }
    fn deinit(w: Walk) void {
        w.l.f.matcher_free(w.m);
    }
    fn allowed(w: Walk, a: std.mem.Allocator) ![]const u32 {
        var bits: [words]u32 = @splat(0);
        if (w.l.f.fill(w.m, @ptrCast(&bits), words) != 0) return error.Fill;
        var out: std.ArrayList(u32) = .empty;
        for (0..vocab) |i| if (bits[i / 32] >> @intCast(i % 32) & 1 != 0) try out.append(a, @intCast(i));
        return out.items;
    }
    /// Each token must be in the mask and then accepted: the index of the first that is not, else null.
    fn walk(w: Walk, tokens: []const u32) !?usize {
        for (tokens, 0..) |tok, n| {
            var bits: [words]u32 = @splat(0);
            if (w.l.f.fill(w.m, @ptrCast(&bits), words) != 0) return error.Fill;
            if (bits[tok / 32] >> @intCast(tok % 32) & 1 == 0) return n;
            switch (w.l.f.accept(w.m, @intCast(tok))) {
                1 => {},
                0 => return n,
                else => return error.Accept,
            }
        }
        return null;
    }
};

/// Whether the grammar takes the whole reply and then the stop token, ending there.
fn admits(a: std.mem.Allocator, l: Lib, g: *anyopaque, pieces: []const []const u8) !bool {
    const w = try Walk.init(l, g);
    defer w.deinit();
    if (try w.walk(try idsOf(a, pieces)) != null) return false;
    if (try w.walk(&.{stop}) != null) return false;
    return w.l.f.terminated(w.m) == 1;
}

fn after(a: std.mem.Allocator, l: Lib, g: *anyopaque, pieces: []const []const u8) ![]const u32 {
    const w = try Walk.init(l, g);
    defer w.deinit();
    if (try w.walk(try idsOf(a, pieces)) != null) return error.TestRefused;
    return w.allowed(a);
}

const think = [_][]const u8{ "<|message_model|>", "<|content_thinking|>", "Use < and |> and </think> freely: a<|b.", "<|end_message|>" };

test "Inkling 2 replies through xgrammar: an optional free thinking part, then the JSON body in its typed part" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const l = try Lib.open(a);
    defer l.f.close(l.h);
    const schema = (try json.parse(a, "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"c\":{\"type\":\"number\"}},\"required\":[\"city\",\"c\"],\"additionalProperties\":false}")).ok;
    const object = (try json.parse(a, "{\"type\":\"object\"}")).ok;
    for ([_]json.Value{ object, schema }) |s| {
        const g = try l.compile(try inkling2.reply(a, grammar_ids, s));
        defer l.f.free(g);
        const t = std.testing;
        try t.expectEqualSlices(u32, &.{mm}, try after(a, l, g, &.{}));
        try t.expectEqualSlices(u32, &.{ thinking, content }, try after(a, l, g, &.{"<|message_model|>"}));
        // the thinking part: every text token and its end, no other marker
        const free = try after(a, l, g, &.{ "<|message_model|>", "<|content_thinking|>", "hm" });
        try t.expectEqual(@as(usize, text_tokens + 1), free.len);
        try t.expectEqual(end, free[free.len - 1]);
        const body = "{\"city\": \"Paris\", \"c\": 12.5}";
        const reply = [_][]const u8{ "<|message_model|>", "<|content_text|>", body, "<|end_message|>" };
        try t.expect(try admits(a, l, g, &reply));
        try t.expect(try admits(a, l, g, &(think ++ reply)));
        try t.expectEqualSlices(u32, &.{ stop, user }, try after(a, l, g, &(think ++ reply)));
        // no marker in a JSON string, as one token or spelled; other <| text is fine
        const inside = try after(a, l, g, &.{ "<|message_model|>", "<|content_text|>", "{\"city\": \"Pa" });
        try t.expect(inside[inside.len - 1] < text_tokens);
        try t.expect(!try admits(a, l, g, &.{ "<|message_model|>", "<|content_text|>", "{\"city\": \"a<|end_message|>\", \"c\": 1}", "<|end_message|>" }));
        try t.expect(try admits(a, l, g, &.{ "<|message_model|>", "<|content_text|>", "{\"city\": \"a <| b |> c\", \"c\": 1}", "<|end_message|>" }));
        // the opener is the token, never spelled; no reply without its typed part, no second answer, no bare JSON
        try t.expect(!try admits(a, l, g, &.{ "<|message_model|>", "<|content_text|>", body }));
        try t.expect(!try admits(a, l, g, &.{body}));
        try t.expect(!try admits(a, l, g, &(reply ++ reply)));
        try t.expect(!try admits(a, l, g, &.{ "<|message_model|>", "<|content_thinking|>", "hm", "<|end_message|>" }));
        try t.expect(!try admits(a, l, g, &.{ "<|message_model|>", "<|content_thinking|>", "a", "<|unused|>", "<|end_message|>" }));
    }
    // the schema holds: a missing required key or a wrong type is refused
    const g = try l.compile(try inkling2.reply(a, grammar_ids, schema));
    defer l.f.free(g);
    try std.testing.expect(!try admits(a, l, g, &.{ "<|message_model|>", "<|content_text|>", "{\"city\": \"Paris\"}", "<|end_message|>" }));
    try std.testing.expect(!try admits(a, l, g, &.{ "<|message_model|>", "<|content_text|>", "{\"city\": 3, \"c\": 1}", "<|end_message|>" }));
    // grammar_budget does not cover token-level rules: its existing 400
    try std.testing.expect(l.f.budget_ready == null or l.f.budget_ready.?(g) != 1);
}

test "Inkling 2 tool calls through xgrammar: named, required, parallel, and the template's key order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const l = try Lib.open(a);
    defer l.f.close(l.h);
    const t = std.testing;
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"days\":{\"type\":\"integer\"},\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"zone\":{\"type\":\"string\"}},\"required\":[\"zone\"]}}}]")).ok.array;
    const weather = [_][]const u8{ "<|message_model|>", "weather", "<|content_invoke_tool_json|>", "{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\",\"days\":2}}", "<|end_message|>" };
    const time = [_][]const u8{ "<|message_model|>", "time", "<|content_invoke_tool_json|>", "{\"name\":\"time\",\"args\":{\"zone\":\"Asia/Tokyo\"}}", "<|end_message|>" };
    const unsorted = [_][]const u8{ "<|message_model|>", "weather", "<|content_invoke_tool_json|>", "{\"name\":\"weather\",\"args\":{\"days\":2,\"city\":\"Oslo\"}}", "<|end_message|>" };
    const no_city = [_][]const u8{ "<|message_model|>", "weather", "<|content_invoke_tool_json|>", "{\"name\":\"weather\",\"args\":{\"days\":2}}", "<|end_message|>" };
    const other_name = [_][]const u8{ "<|message_model|>", "weather", "<|content_invoke_tool_json|>", "{\"name\":\"time\",\"args\":{\"zone\":\"UTC\"}}", "<|end_message|>" };
    const text_part = [_][]const u8{ "<|message_model|>", "<|content_text|>", "Sure.", "<|end_message|>" };
    const Case = struct { what: []const u8, tools: []const json.Value, parallel: bool, reply: []const []const u8, ok: bool };
    const cases = [_]Case{
        .{ .what = "named", .tools = tools[0..1], .parallel = false, .reply = &weather, .ok = true },
        .{ .what = "named after thinking", .tools = tools[0..1], .parallel = false, .reply = &(think ++ weather), .ok = true },
        .{ .what = "named, another tool", .tools = tools[0..1], .parallel = false, .reply = &time, .ok = false },
        .{ .what = "named, twice", .tools = tools[0..1], .parallel = false, .reply = &(weather ++ weather), .ok = false },
        .{ .what = "named, unsorted keys", .tools = tools[0..1], .parallel = false, .reply = &unsorted, .ok = false },
        .{ .what = "named, no required key", .tools = tools[0..1], .parallel = false, .reply = &no_city, .ok = false },
        .{ .what = "named, its body names another", .tools = tools[0..1], .parallel = false, .reply = &other_name, .ok = false },
        .{ .what = "named, text instead", .tools = tools[0..1], .parallel = false, .reply = &text_part, .ok = false },
        .{ .what = "named, no call", .tools = tools[0..1], .parallel = false, .reply = &think, .ok = false },
        .{ .what = "required, either", .tools = tools, .parallel = false, .reply = &time, .ok = true },
        .{ .what = "required, two without parallel", .tools = tools, .parallel = false, .reply = &(weather ++ time), .ok = false },
        .{ .what = "parallel, one", .tools = tools, .parallel = true, .reply = &(think ++ weather), .ok = true },
        .{ .what = "parallel, three", .tools = tools, .parallel = true, .reply = &(weather ++ time ++ weather), .ok = true },
        .{ .what = "parallel, text between", .tools = tools, .parallel = true, .reply = &(weather ++ text_part ++ time), .ok = false },
    };
    var wrong: usize = 0;
    for (cases) |c| {
        const g = try l.compile(try inkling2.calls(a, grammar_ids, c.tools, c.parallel));
        defer l.f.free(g);
        const got = try admits(a, l, g, c.reply);
        if (got == c.ok) continue;
        std.debug.print("{s}: want {}, got {}\n", .{ c.what, c.ok, got });
        wrong += 1;
    }
    try t.expectEqual(@as(usize, 0), wrong);
    // after <|message_model|>: a thinking part or an offered tool's name
    const g = try l.compile(try inkling2.calls(a, grammar_ids, tools, true));
    defer l.f.free(g);
    try t.expectEqualSlices(u32, &.{ 't' - 0x20, 'w' - 0x20, thinking }, try after(a, l, g, &.{"<|message_model|>"}));
}
