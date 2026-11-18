//! Inkling 2's typed reply parts as reasoning_content, content and tool_calls, whole or streamed in any chunks.
//!
//! The grammar is the checkpoint's own (tokenizer_config.json ``response_template`` and chat_template.jinja, released
//! by Thinking Machines under the Apache License 2.0); no code is copied from them:
//!
//!   reply  := part* (``<|content_model_end_sampling|>`` | end of text)
//!   part   := [``<|message_model|>`` NAME] OPEN BODY ``<|end_message|>``
//!   OPEN   := ``<|content_thinking|>``                                  -> reasoning_content (joined, not stripped)
//!           | ``<|content_text|>`` | ``<|content_tool_error|>`` | ``<|content_invoke_tool_text|>`` -> content
//!           | ``<|content_invoke_tool_json|>`` BODY = {"name": NAME, "args": {...}}       -> one tool call
//!
//! Text outside a part is not part of any field, as in Transformers' response-template parser
//! (utils/chat_parsing/response_parser.py, Apache License 2.0). Beyond it, this reader also: closes a
//! part a new header or opener interrupts, ends the reply at a user, system or tool header (another speaker's turn),
//! keeps a part max_tokens cut (its text so far; a cut call's JSON stays text), and keeps a call it cannot use (bad
//! JSON, no name, arguments that are not an object, a tool the request did not offer) as content text.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;

pub const Marker = enum {
    message_model,
    message_user,
    message_system,
    message_tool,
    content_text,
    content_thinking,
    content_tool_error,
    content_invoke_tool_text,
    content_invoke_tool_json,
    content_xml,
    content_image,
    end_message,
    content_model_end_sampling,
    endoftext,
    begin_of_text,

    /// The marker's spelling, which is also its added token's text.
    pub fn spelling(m: Marker) []const u8 {
        return switch (m) {
            inline else => |tag| "<|" ++ @tagName(tag) ++ "|>",
        };
    }

    pub fn of(text: []const u8) ?Marker {
        if (text.len < 4 or !std.mem.startsWith(u8, text, "<|") or !std.mem.endsWith(u8, text, "|>")) return null;
        return std.meta.stringToEnum(Marker, text[2 .. text.len - 2]);
    }
};

/// What a part's body feeds.
pub const Kind = enum { text, thinking, tool_json, ignored };

pub fn kindOf(m: Marker) ?Kind {
    return switch (m) {
        .content_text, .content_tool_error, .content_invoke_tool_text => .text,
        .content_thinking => .thinking,
        .content_invoke_tool_json => .tool_json,
        .content_xml, .content_image => .ignored,
        else => null,
    };
}

pub const Call = struct { name: []const u8, arguments: []const u8 };

pub const State = union(enum) { outside, header, part: Kind, ended };

pub const Parser = struct {
    a: Allocator,
    /// The tool names the request offered; a call to any other name stays text.
    offered: []const []const u8 = &.{},
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    calls: std.ArrayList(Call) = .empty,
    /// Tool parts kept as text: bad JSON, no name, non-object arguments, or a tool the request did not offer.
    rejected: usize = 0,
    /// Non-space bytes outside every part, which no field takes.
    stray: usize = 0,
    state: State = .outside,
    header: std.ArrayList(u8) = .empty,
    body: std.ArrayList(u8) = .empty,
    held: std.ArrayList(u8) = .empty,

    pub fn deinit(p: *Parser) void {
        inline for (.{ &p.reasoning, &p.content, &p.header, &p.body, &p.held }) |list| list.deinit(p.a);
        for (p.calls.items) |c| {
            p.a.free(c.name);
            p.a.free(c.arguments);
        }
        p.calls.deinit(p.a);
    }

    pub fn ended(p: *const Parser) bool {
        return p.state == .ended;
    }

    pub fn thinking(p: *const Parser) bool {
        return p.state == .part and p.state.part == .thinking;
    }

    /// Decoded reply text: marker spellings are markers, and one split across chunks is held until it completes.
    pub fn feedText(p: *Parser, bytes: []const u8) Allocator.Error!void {
        try p.held.appendSlice(p.a, bytes);
        const s = p.held.items;
        var done: usize = 0;
        var i: usize = 0;
        var hold: ?usize = null;
        while (i < s.len) {
            const lt = std.mem.indexOfScalarPos(u8, s, i, '<') orelse break;
            const rest = s[lt..];
            var hit: ?Marker = null;
            var maybe = false;
            for (std.enums.values(Marker)) |m| {
                const w = m.spelling();
                if (std.mem.startsWith(u8, rest, w)) {
                    hit = m;
                    break;
                }
                if (rest.len < w.len and std.mem.startsWith(u8, w, rest)) maybe = true;
            }
            if (hit) |m| {
                try p.feedLiteral(s[done..lt]);
                try p.feedMarker(m);
                done = lt + m.spelling().len;
                i = done;
            } else if (maybe) {
                hold = lt; // a marker may complete with the next chunk
                break;
            } else i = lt + 1;
        }
        const keep = hold orelse s.len;
        try p.feedLiteral(s[done..keep]);
        const left = s.len - keep;
        std.mem.copyForwards(u8, p.held.items[0..left], s[keep..]);
        p.held.items.len = left;
    }

    /// Bytes that are text whatever they spell (an ordinary token's bytes).
    pub fn feedLiteral(p: *Parser, bytes: []const u8) Allocator.Error!void {
        if (bytes.len == 0) return;
        switch (p.state) {
            .ended => {},
            .outside => for (bytes) |b| {
                p.stray += @intFromBool(std.mem.indexOfScalar(u8, " \t\r\n", b) == null);
            },
            .header => try p.header.appendSlice(p.a, bytes),
            .part => |k| switch (k) {
                .text => try p.content.appendSlice(p.a, bytes),
                .thinking => try p.reasoning.appendSlice(p.a, bytes),
                .tool_json => try p.body.appendSlice(p.a, bytes),
                .ignored => {},
            },
        }
    }

    /// One marker (an added token, or its spelling in text).
    pub fn feedMarker(p: *Parser, m: Marker) Allocator.Error!void {
        if (p.state == .ended) return;
        if (kindOf(m)) |k| {
            try p.close();
            p.state = .{ .part = k };
            return;
        }
        switch (m) {
            .message_model => {
                try p.close();
                p.header.clearRetainingCapacity();
                p.state = .header;
            },
            .end_message => {
                try p.close();
                p.state = .outside;
            },
            .message_user, .message_system, .message_tool, .content_model_end_sampling, .endoftext => {
                try p.close();
                p.state = .ended;
            },
            .begin_of_text => {},
            else => unreachable,
        }
    }

    /// The reply's end: a held partial marker is text, and an open tool part stays text.
    pub fn finish(p: *Parser) Allocator.Error!void {
        const held = try p.a.dupe(u8, p.held.items);
        defer p.a.free(held);
        p.held.clearRetainingCapacity();
        try p.feedLiteral(held);
        if (p.state == .part and p.state.part == .tool_json) {
            try p.content.appendSlice(p.a, p.body.items);
            p.body.clearRetainingCapacity();
            p.rejected += 1;
        }
        if (p.state != .ended) p.state = .outside;
    }

    fn close(p: *Parser) Allocator.Error!void {
        if (p.state == .part and p.state.part == .tool_json) {
            defer p.body.clearRetainingCapacity();
            try p.calls.ensureUnusedCapacity(p.a, 1); // a call's owned name and arguments land with no step left to fail
            if (try p.call(p.body.items)) |c| {
                p.calls.appendAssumeCapacity(c);
            } else {
                try p.content.appendSlice(p.a, p.body.items);
                p.rejected += 1;
            }
        }
        p.state = .outside;
    }

    /// ``{"name": NAME, "args": {...}}`` as a call to an offered tool, or null.
    fn call(p: *Parser, body: []const u8) Allocator.Error!?Call {
        var arena = std.heap.ArenaAllocator.init(p.a);
        defer arena.deinit();
        const a = arena.allocator();
        var fixed: std.ArrayList(u8) = .empty;
        try @import("tokenizer").unicode.appendLossy(&fixed, a, body);
        const value = switch (try json.parseText(a, fixed.items)) {
            .ok => |v| v,
            .err => return null,
        };
        if (value != .object) return null;
        const name = value.strField("name") orelse return null;
        const known = for (p.offered) |o| {
            if (std.mem.eql(u8, o, name)) break true;
        } else false;
        if (!known) return null;
        const args: Value = value.get("args") orelse .null;
        const object: Value = switch (args) {
            .null => .{ .object = try json.newObject(a) },
            .object => args,
            else => return null,
        };
        const text = try json.stringify(a, object, .{ .ascii = false, .compact = true });
        const owned_name = try p.a.dupe(u8, name);
        errdefer p.a.free(owned_name);
        return .{ .name = owned_name, .arguments = try p.a.dupe(u8, text) };
    }
};

const Value = json.Value;

/// A whole reply's text, parsed at once.
pub fn parseText(a: Allocator, text: []const u8, offered: []const []const u8) Allocator.Error!Parser {
    var p: Parser = .{ .a = a, .offered = offered };
    errdefer p.deinit();
    try p.feedText(text);
    try p.finish();
    return p;
}

/// The thinking-effort the template hears: thinking off is ``none`` (its ``0`` line); on, the request's effort or the
/// template's own default (0.9) when the request names none.
pub fn templateEffort(thinking: bool, effort: ?[]const u8) ?[]const u8 {
    if (!thinking) return "none";
    return effort;
}

/// The refusal for image, audio and video parts until Inkling 2's vision tower runs in this engine.
pub const image_refusal = "Inkling 2 image input needs its vision tower, which this engine does not run yet; send text only";

/// The token ids that end a reply besides config.json's EOS (200006): another speaker's header or the text end.
pub const extra_stops = [_]Marker{ .endoftext, .message_user, .message_system, .message_tool };

/// Whether a template is Inkling 2's: its typed openers and the numeric thinking-effort line.
pub fn isTemplate(source: []const u8) bool {
    return std.mem.indexOf(u8, source, "<|content_invoke_tool_json|>") != null and
        std.mem.indexOf(u8, source, "Thinking effort level: ") != null;
}

// ---------------------------------------------------------------------------------------------------------------
// tests

const t = std.testing;

fn expectParsed(p: *const Parser, reasoning: []const u8, content: []const u8, calls: []const [2][]const u8) !void {
    try t.expectEqualStrings(reasoning, p.reasoning.items);
    try t.expectEqualStrings(content, p.content.items);
    try t.expectEqual(calls.len, p.calls.items.len);
    for (calls, p.calls.items) |want, got| {
        try t.expectEqualStrings(want[0], got.name);
        try t.expectEqualStrings(want[1], got.arguments);
    }
}

const tools = [_][]const u8{ "weather", "search" };

const sample = "<|message_model|><|content_thinking|>Need the weather <tool> in two cities.\n<|end_message|>" ++
    "<|message_model|><|content_text|>Checking both.<|end_message|>" ++
    "<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"K\u{f8}benhavn\"}}<|end_message|>" ++
    "<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Tokyo\",\"days\":[1,2]}}<|end_message|>" ++
    "<|content_model_end_sampling|>ignored after the end";

test "Inkling 2 reply: thinking, text and two tool calls from one whole reply" {
    var p = try parseText(t.allocator, sample, &tools);
    defer p.deinit();
    try expectParsed(&p, "Need the weather <tool> in two cities.\n", "Checking both.", &.{
        .{ "weather", "{\"city\":\"K\u{f8}benhavn\"}" },
        .{ "weather", "{\"city\":\"Tokyo\",\"days\":[1,2]}" },
    });
    try t.expect(p.ended());
    try t.expectEqual(@as(usize, 0), p.rejected);
}

fn chunked(a: Allocator, text: []const u8, seed: u64, max: usize, strict: bool) !Parser {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    var p: Parser = .{ .a = a, .offered = &tools };
    errdefer p.deinit();
    var at: usize = 0;
    var last_reasoning: usize = 0;
    var last_content: usize = 0;
    while (at < text.len) {
        const n = @min(text.len - at, 1 + r.uintLessThan(usize, max));
        try p.feedText(text[at .. at + n]);
        at += n;
        // append-only: what a stream has shown never changes
        try t.expect(p.reasoning.items.len >= last_reasoning and p.content.items.len >= last_content);
        last_reasoning = p.reasoning.items.len;
        last_content = p.content.items.len;
        // no held marker byte ever reaches a field
        if (strict) try t.expect(std.mem.indexOf(u8, p.content.items, "<|") == null);
    }
    try p.finish();
    return p;
}

test "Inkling 2 reply: random chunks, single bytes included, equal the whole reply" {
    var whole = try parseText(t.allocator, sample, &tools);
    defer whole.deinit();
    for (0..400) |seed| {
        var p = try chunked(t.allocator, sample, seed, if (seed % 4 == 0) 1 else 1 + seed % 23, true);
        defer p.deinit();
        try expectParsed(&p, whole.reasoning.items, whole.content.items, &.{
            .{ whole.calls.items[0].name, whole.calls.items[0].arguments },
            .{ whole.calls.items[1].name, whole.calls.items[1].arguments },
        });
    }
}

test "Inkling 2 reply: random replies of markers, near-markers and JSON parse the same in any chunks" {
    const pieces = [_][]const u8{
        "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_invoke_tool_json|>",
        "<|content_tool_error|>", "<|content_xml|>", "<|message_tool|>", "<|content_model_end_sampling|>", "weather",
        "{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\"}}", "{\"name\":\"search\"}", "{\"name\":", "<", "<|", "<|end_",
        "<|content_", "|>", "a b", " ", "\n", "\u{e9}\u{4e16}\u{1f600}", "<b>", "<|x|>", "1 < 2", "|",
    };
    var rng = std.Random.DefaultPrng.init(20261112);
    const r = rng.random();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(t.allocator);
    for (0..300) |round| {
        text.clearRetainingCapacity();
        for (0..r.uintLessThan(usize, 40)) |_| try text.appendSlice(t.allocator, pieces[r.uintLessThan(usize, pieces.len)]);
        var whole = try parseText(t.allocator, text.items, &tools);
        defer whole.deinit();
        for (0..4) |k| {
            var p = try chunked(t.allocator, text.items, round * 4 + k, if (k == 0) 1 else 1 + r.uintLessThan(usize, 30), false);
            defer p.deinit();
            try t.expectEqualStrings(whole.reasoning.items, p.reasoning.items);
            try t.expectEqualStrings(whole.content.items, p.content.items);
            try t.expectEqual(whole.calls.items.len, p.calls.items.len);
            for (whole.calls.items, p.calls.items) |x, y| {
                try t.expectEqualStrings(x.name, y.name);
                try t.expectEqualStrings(x.arguments, y.arguments);
            }
            try t.expectEqual(whole.rejected, p.rejected);
            try t.expectEqual(whole.stray, p.stray);
        }
    }
}

test "Inkling 2 reply: markers as tokens equal markers as text" {
    var p: Parser = .{ .a = t.allocator, .offered = &tools };
    defer p.deinit();
    // the same reply as a token stream would deliver it: added tokens as markers, ordinary tokens' bytes split anywhere
    var rest: []const u8 = sample;
    while (rest.len > 0) {
        if (std.mem.indexOf(u8, rest, "<|")) |at| {
            for (rest[0..at]) |b| try p.feedLiteral(&.{b});
            const end = std.mem.indexOfPos(u8, rest, at, "|>").? + 2;
            try p.feedMarker(Marker.of(rest[at..end]).?);
            rest = rest[end..];
        } else {
            try p.feedLiteral(rest);
            rest = "";
        }
    }
    try p.finish();
    var whole = try parseText(t.allocator, sample, &tools);
    defer whole.deinit();
    try t.expectEqualStrings(whole.reasoning.items, p.reasoning.items);
    try t.expectEqualStrings(whole.content.items, p.content.items);
    try t.expectEqual(whole.calls.items.len, p.calls.items.len);
}

test "Inkling 2 reply: a marker spelled by ordinary tokens stays text" {
    var p: Parser = .{ .a = t.allocator };
    defer p.deinit();
    try p.feedMarker(.message_model);
    try p.feedMarker(.content_text);
    try p.feedLiteral("quote: <|end_message|> and <|content_thinking|>");
    try p.feedMarker(.end_message);
    try p.finish();
    try expectParsed(&p, "", "quote: <|end_message|> and <|content_thinking|>", &.{});
}

test "Inkling 2 reply: unusable calls stay text, partial JSON waits, a cut call is text" {
    const bad = "<|content_invoke_tool_json|>{\"name\":\"nope\",\"args\":{}}<|end_message|>" ++
        "<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":[1]}<|end_message|>" ++
        "<|content_invoke_tool_json|>{not json<|end_message|>" ++
        "<|content_invoke_tool_json|>{\"name\":\"search\"}<|end_message|>";
    var p = try parseText(t.allocator, bad, &tools);
    defer p.deinit();
    try expectParsed(&p, "", "{\"name\":\"nope\",\"args\":{}}{\"name\":\"weather\",\"args\":[1]}{not json", &.{.{ "search", "{}" }});
    try t.expectEqual(@as(usize, 3), p.rejected);
    // a call's JSON arrives in pieces and is only a call once its part closes
    var s: Parser = .{ .a = t.allocator, .offered = &tools };
    defer s.deinit();
    try s.feedText("<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"sea");
    try t.expectEqual(@as(usize, 0), s.calls.items.len);
    try s.feedText("rch\",\"args\":{\"q\":\"zig\"}}<|end_mes");
    try t.expectEqual(@as(usize, 0), s.calls.items.len);
    try s.feedText("sage|>");
    try expectParsed(&s, "", "", &.{.{ "search", "{\"q\":\"zig\"}" }});
    // max_tokens cut a call: its JSON so far is text, as the shared parser keeps an unclosed envelope
    var cut = try parseText(t.allocator, "<|content_text|>Hi<|end_message|><|content_invoke_tool_json|>{\"name\":\"we", &tools);
    defer cut.deinit();
    try expectParsed(&cut, "", "Hi{\"name\":\"we", &.{});
}

test "Inkling 2 reply: interrupted parts, other speakers, stray text and a trailing partial marker" {
    var p = try parseText(t.allocator, "stray<|content_thinking|>a<|content_text|>b<|message_model|>x<|content_thinking|>c<|end_message|>" ++
        "<|message_user|><|content_text|>not ours<|end_message|>", &tools);
    defer p.deinit();
    try expectParsed(&p, "ac", "b", &.{});
    try t.expectEqual(@as(usize, 5), p.stray);
    try t.expect(p.ended());
    // a reply max_tokens cut inside a marker's spelling keeps those bytes as text
    var q: Parser = .{ .a = t.allocator };
    defer q.deinit();
    try q.feedText("<|content_text|>a <|end_");
    try t.expectEqualStrings("a ", q.content.items);
    try q.finish();
    try t.expectEqualStrings("a <|end_", q.content.items);
    // a '<' that cannot start a marker is text at once
    var r: Parser = .{ .a = t.allocator };
    defer r.deinit();
    try r.feedText("<|content_text|>1 < 2 <b> <|x|> <");
    try t.expectEqualStrings("1 < 2 <b> <|x|> ", r.content.items);
}

fn parseSample(a: Allocator) !void {
    var p = try parseText(a, sample, &tools);
    defer p.deinit();
    try t.expectEqual(@as(usize, 2), p.calls.items.len);
}

test "Inkling 2 reply: every allocation failure frees what the parse held, calls included" {
    try t.checkAllAllocationFailures(t.allocator, parseSample, .{});
}

test "Inkling 2 reply: the effort the template hears and the template probe" {
    try t.expectEqualStrings("none", templateEffort(false, "high").?);
    try t.expectEqualStrings("none", templateEffort(false, null).?);
    try t.expectEqualStrings("low", templateEffort(true, "low").?);
    try t.expectEqual(@as(?[]const u8, null), templateEffort(true, null));
    try t.expect(isTemplate("<|content_invoke_tool_json|> ... Thinking effort level: "));
    try t.expect(!isTemplate("<|im_start|>"));
    try t.expectEqual(Marker.end_message, Marker.of("<|end_message|>").?);
    try t.expectEqual(@as(?Marker, null), Marker.of("<|unused_200054|>"));
}
