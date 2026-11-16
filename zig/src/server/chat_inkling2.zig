//! Inkling 2's typed reply parts behind a chat reply: added tokens are the markers, ordinary tokens are text.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const mt = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const specs = @import("tool_specs.zig");
const parts = @import("inkling2_reply.zig");
const Result = @import("kimi_reply.zig").Result;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The parser advanced through a reply's tokens; a stream keeps one and feeds it only new tokens.
pub const Stream = struct {
    parser: parts.Parser,
    text: mt.Text,
    markers: [std.enums.values(parts.Marker).len]?u32,
    /// The reply as the generation's stop check reads it (ordinary tokens' bytes, markers spelled), and each token's end.
    raw: std.ArrayList(u8) = .empty,
    ends: std.ArrayList(usize) = .empty,
    marked: std.ArrayList(?parts.Marker) = .empty,
    fed: usize = 0, // bytes of ``raw`` the parser has had
    next: usize = 0, // the token holding byte ``fed``
    reasoning_tokens: usize = 0,
    content_at: std.ArrayList(usize) = .empty, // the tokens whose bytes are content, in order: its logprob rows
    body_at: std.ArrayList(usize) = .empty, // the open tool part's tokens, content too if its call stays text

    pub fn init(a: Allocator, text: mt.Text, tools: []const Value) Allocator.Error!Stream {
        const names = try a.alloc([]const u8, tools.len);
        for (tools, names) |tool, *n| n.* = try specs.toolName(a, tool);
        var s: Stream = .{ .parser = .{ .a = a, .offered = names }, .text = text, .markers = undefined };
        for (std.enums.values(parts.Marker), &s.markers) |m, *id| id.* = text.tokenId(m.spelling());
        return s;
    }

    fn markerOf(s: *const Stream, id: u32) ?parts.Marker {
        for (s.markers, std.enums.values(parts.Marker)) |m, marker| if (m == id) return marker;
        return null;
    }

    /// Reads ``ids``' new tokens and feeds the parser the reply up to its first stop string, found where the
    /// generation's stop check finds it (markers spelled); while ``partial``, a tail that may begin a stop waits.
    pub fn extend(s: *Stream, ids: []const u32, stops: reply_text.Stops, partial: bool) Allocator.Error!void {
        const a = s.parser.a;
        while (s.ends.items.len < ids.len) {
            const id = ids[s.ends.items.len];
            const marker = s.markerOf(id);
            if (marker) |m| try s.raw.appendSlice(a, m.spelling()) else {
                const bytes = (try s.text.tokenBytes(a, id)) orelse try s.text.decode(a, &.{id});
                defer a.free(bytes);
                try s.raw.appendSlice(a, bytes);
            }
            try s.marked.append(a, marker);
            try s.ends.append(a, s.raw.items.len);
        }
        const limit = if (stops.strings.len > 0) stops.visible(s.raw.items, partial).len else s.raw.items.len;
        while (s.next < s.ends.items.len and !s.parser.ended()) : (s.next += 1) {
            const start = if (s.next == 0) 0 else s.ends.items[s.next - 1];
            const end = s.ends.items[s.next];
            if (s.marked.items[s.next]) |m| {
                if (end > limit) break; // a marker the stop cuts is not read
                const before = s.parser.content.items.len;
                try s.parser.feedMarker(m);
                try s.settle(before);
            } else {
                const to = @min(end, limit);
                if (to == s.fed and end > s.fed) break;
                if (s.fed == start) try s.note(s.next);
                try s.parser.feedLiteral(s.raw.items[s.fed..to]);
                if (to < end) {
                    s.fed = to;
                    break;
                }
            }
            s.fed = end;
        }
    }

    /// Where an ordinary token's bytes go: the reasoning count, the content's rows, or the open call's.
    fn note(s: *Stream, at: usize) Allocator.Error!void {
        if (s.parser.state != .part) return;
        switch (s.parser.state.part) {
            .thinking => s.reasoning_tokens += 1,
            .text => try s.content_at.append(s.parser.a, at),
            .tool_json => try s.body_at.append(s.parser.a, at),
            .ignored => {},
        }
    }

    /// A tool part closed: a call kept as text makes its tokens content.
    fn settle(s: *Stream, before: usize) Allocator.Error!void {
        if (s.parser.body.items.len > 0) return;
        if (s.parser.content.items.len > before) try s.content_at.appendSlice(s.parser.a, s.body_at.items);
        s.body_at.clearRetainingCapacity();
    }

    /// The parsed fields so far; ``finished`` also settles an open tool part and checks the calls.
    pub fn result(s: *Stream, finished: bool) Allocator.Error!Result {
        const a = s.parser.a;
        if (finished) {
            const before = s.parser.content.items.len;
            try s.parser.finish();
            try s.settle(before);
        }
        var calls: std.ArrayList(Value) = .empty;
        if (finished) for (s.parser.calls.items) |c| {
            const function = try json.newObject(a);
            try function.put(a, "name", .{ .string = c.name });
            try function.put(a, "arguments", .{ .string = c.arguments });
            const call = try json.newObject(a);
            try call.put(a, "type", .{ .string = "function" });
            try call.put(a, "function", .{ .object = function });
            try calls.append(a, .{ .object = call });
        };
        return .{
            .content = try whole(a, s.parser.content.items, finished),
            .reasoning = try whole(a, s.parser.reasoning.items, finished),
            .calls = if (calls.items.len > 0) calls.items else null,
            .reasoning_tokens = s.reasoning_tokens,
            .rejected = s.parser.rejected > 0,
            .content_at = s.content_at.items,
        };
    }
};

/// UTF-8 text of raw token bytes: invalid bytes as U+FFFD; while streaming, a character still split is held back.
fn whole(a: Allocator, bytes: []const u8, finished: bool) Allocator.Error![]const u8 {
    var end = bytes.len;
    if (!finished) for (bytes.len -| 3..bytes.len) |i| {
        const n = std.unicode.utf8ByteSequenceLength(bytes[i]) catch continue;
        if (n > bytes.len - i) {
            end = i;
            break;
        }
    };
    if (std.unicode.utf8ValidateSlice(bytes[0..end])) return a.dupe(u8, bytes[0..end]);
    var out: std.ArrayList(u8) = .empty;
    try @import("tokenizer").unicode.appendLossy(&out, a, bytes[0..end]);
    return out.items;
}

/// A whole reply (``ids`` without its EOS), as the finished response reads it; stop strings cut it where the
/// generation's stop check matched them, so a stop the part markers break up cuts nothing.
pub fn read(g: anytype, ids: []const u32, finished: bool) Allocator.Error!Result {
    if (!g.chat) return .{ .content = try g.srv.text.decode(g.a, ids), .reasoning = "", .calls = null, .reasoning_tokens = 0 };
    var s = try Stream.init(g.a, g.srv.text, g.tools);
    try s.extend(ids, g.stops, !finished);
    var out = try s.result(finished);
    if (g.stops.strings.len > 0) {
        const raw = try g.srv.text.decode(g.a, ids);
        if (g.stops.matched(raw) != null) out.calls = null;
    }
    return out;
}

/// Streams the reasoning and content a chat reply's new tokens add; calls arrive with the finished reply. Neither
/// field is stripped: the checkpoint's response template reads both with ``strip: false``.
pub fn stream(g: anytype, sink: chat.Sink) chat.Failure!void {
    const ids = reply_text.stripTrailing(g.collected.items, g.srv.eos);
    if (g.inkling2 == null) g.inkling2 = try Stream.init(g.a, g.srv.text, g.tools);
    const s = &g.inkling2.?;
    try s.extend(ids, g.stops, true);
    const now = try s.result(false);
    const reason = g.streamed_reasoning.after(now.reasoning, false);
    if (reason.len > 0) {
        try g.streamed_reasoning.set(g.a, now.reasoning, reason);
        try g.emit(sink, try reply_text.deltaOf(g.a, "reasoning_content", reason));
    }
    const delta = g.streamed.after(now.content, false);
    if (delta.len > 0) {
        try g.streamed.set(g.a, now.content, delta);
        // the rows of the content tokens the streamed text now spells, never a marker's or the reasoning's
        try g.emit(sink, if (g.lp) |*l| (try l.deltaAt(g.srv, g.a, g.collected.items, s.content_at.items, delta, g.streamed.text.items.len, false)).? else .{ .string = delta });
    }
}
