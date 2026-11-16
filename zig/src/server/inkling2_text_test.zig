//! Inkling 2's text layer through the real OpenAI handler: typed reply parts, its stops, its effort line, image refusal.
//! The hermetic cases use a small character tokenizer with Inkling 2's marker ids; with TF_INKLING2_DIR naming a
//! folder holding the checkpoint's tokenizer.json, tokenizer_config.json, config.json and chat_template.jinja, the
//! same checks also run on the release files.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const http = @import("http_conn.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const parts = @import("inkling2_reply.zig");
const V = json.Value;
const t = std.testing;

const Script = struct {
    a: std.mem.Allocator,
    ids: []const u32,
    prompt: []const u32 = &.{},
    eos: []const u32 = &.{},
    logprobs: bool = false, // the backend advertises logprobs: token i's row is -(i + 1) / 4
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ptr: *anyopaque) api.Info {
        const s: *@This() = @ptrCast(@alignCast(ptr));
        return .{ .context_window = 32768, .reasoning = true, .logprobs = s.logprobs };
    }
    fn submit(ptr: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ptr));
        s.prompt = s.a.dupe(u32, request.prompt) catch return error.Busy;
        s.eos = s.a.dupe(u32, request.eos) catch return error.Busy;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (s.ids, 0..) |token, i| {
            if (request.logprobs) |count| {
                var row: api.LogprobRow = .{ .token = token, .logprob = -@as(f32, @floatFromInt(i + 1)) / 4, .count = count };
                for (0..count) |k| {
                    row.ids[k] = token;
                    row.logprobs[k] = row.logprob;
                }
                sink.event(sink.ctx, id, &.{ .logprobs = &.{row} });
            }
            sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Receipt = struct {
    a: std.mem.Allocator,
    code: u16 = 0,
    body: V = .null,
    events: std.ArrayList(V) = .empty,
    fn out(r: *@This()) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = begin, .event = event, .reply = whole } };
    }
    fn begin(ptr: *anyopaque, _: usize) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        r.code = 200;
    }
    fn event(ptr: *anyopaque, value: ?V) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        if (value) |v| r.events.append(r.a, v) catch return error.Closed;
    }
    fn whole(ptr: *anyopaque, code: u16, value: V) void {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        r.code = code;
        r.body = value;
    }
};

/// Inkling 2's added tokens at their release ids.
const markers = [_]struct { []const u8, u32 }{
    .{ "<|endoftext|>", 199999 },              .{ "<|message_user|>", 200000 },        .{ "<|message_model|>", 200001 },
    .{ "<|message_system|>", 200002 },         .{ "<|message_tool|>", 200003 },        .{ "<|content_text|>", 200004 },
    .{ "<|content_image|>", 200005 },          .{ "<|content_model_end_sampling|>", 200006 }, .{ "<|content_thinking|>", 200008 },
    .{ "<|end_message|>", 200010 },            .{ "<|content_tool_error|>", 200022 },  .{ "<|content_xml|>", 200024 },
    .{ "<|begin_of_text|>", 200028 },          .{ "<|content_invoke_tool_json|>", 200049 }, .{ "<|content_invoke_tool_text|>", 200057 },
};

/// A template that writes the effort it hears where Inkling 2's writes its effort line, and names its typed parts.
const template =
    \\{#- <|content_invoke_tool_json|> Thinking effort level: "none" "minimal" "low" "medium" "high" "xhigh" "max" -#}
    \\{%- for m in messages -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>{%- endfor -%}
    \\<|message_system|><|content_text|>effort={{ reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'default' }}<|end_message|>
;

const Fixture = struct {
    tmp: t.TmpDir,
    path: []u8,

    fn init(a: std.mem.Allocator) !Fixture {
        var tok: std.ArrayList(u8) = .empty;
        try tok.appendSlice(a, "{\"model\":{\"type\":\"BPE\",\"merges\":[],\"vocab\":{\"\\n\":95");
        for (0x20..0x7f) |c| {
            try tok.appendSlice(a, ",\"");
            if (c == '"' or c == '\\') try tok.append(a, '\\');
            try tok.append(a, @intCast(c));
            try tok.print(a, "\":{d}", .{c - 0x20});
        }
        for (markers) |m| try tok.print(a, ",\"{s}\":{d}", .{ m[0], m[1] });
        try tok.appendSlice(a, "}},\"decoder\":{\"type\":\"Fuse\"},\"added_tokens\":[");
        for (markers, 0..) |m, i| try tok.print(a, "{s}{{\"id\":{d},\"content\":\"{s}\",\"special\":true}}", .{ if (i > 0) "," else "", m[1], m[0] });
        try tok.appendSlice(a, "]}");
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        const io = t.io;
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tok.items });
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"inkling2\",\"eos_token_id\":200006}" });
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
        try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = template });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(io, ".", a) };
    }
};

const reply_text = "<|message_model|><|content_thinking|>Plan: look up both.<|end_message|>" ++
    "<|message_model|><|content_text|>Checking.<|end_message|>" ++
    "<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\"}}<|end_message|>" ++
    "<|message_model|>weather<|content_invoke_tool_json|>{\"name\":\"weather\",\"args\":{\"city\":\"Rome\",\"days\":2}}<|end_message|>" ++
    "<|content_model_end_sampling|>";

const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"parameters\":{\"type\":\"object\"}}}]";

const Seen = struct { reasoning: []const u8, content: []const u8, calls: usize, finish: []const u8 };

fn ask(a: std.mem.Allocator, srv: *Server, body_text: []const u8, stream: bool) !struct { Receipt, Seen } {
    var body = (try json.parse(a, body_text)).ok;
    if (stream) try body.object.put(a, "stream", .{ .bool = true });
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var r: Receipt = .{ .a = a };
    openai.run(srv, a, r.out(), .{ .conn = &conn }, true, body);
    if (r.code != 200) return .{ r, .{ .reasoning = "", .content = "", .calls = 0, .finish = "" } };
    if (!stream) {
        const choice = r.body.get("choices").?.array[0];
        const message = choice.get("message").?;
        const calls = if (message.get("tool_calls")) |list| list.array.len else 0;
        if (message.get("tool_calls")) |list| for (list.array) |c| try t.expect(c.strField("id") != null);
        return .{ r, .{ .reasoning = message.strField("reasoning_content") orelse "", .content = message.strField("content") orelse "", .calls = calls, .finish = choice.strField("finish_reason") orelse "" } };
    }
    var content: std.ArrayList(u8) = .empty;
    var reason: std.ArrayList(u8) = .empty;
    var calls: usize = 0;
    var finish: []const u8 = "";
    for (r.events.items) |event| {
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        if (choices.array[0].strField("finish_reason")) |f| finish = f;
        const delta = choices.array[0].get("delta") orelse continue;
        if (delta.strField("content")) |s| try content.appendSlice(a, s);
        if (delta.strField("reasoning_content")) |s| try reason.appendSlice(a, s);
        if (delta.get("tool_calls")) |list| for (list.array) |c| {
            if (c.get("function").?.has("name")) calls += 1;
        };
    }
    return .{ r, .{ .reasoning = reason.items, .content = content.items, .calls = calls, .finish = finish } };
}

fn serve(a: std.mem.Allocator, dir: []const u8, script: *Script) !struct { *HfText, *Server } {
    var problem: []const u8 = "";
    const text = HfText.load(t.allocator, t.io, dir, null, a, &problem) catch |e| {
        std.debug.print("load: {s}\n", .{problem});
        return e;
    };
    errdefer text.deinit();
    const srv = try Server.init(a, t.io, script.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
    return .{ text, srv };
}

/// The typed reply, whole and streamed one token at a time, and the stops the engine is given.
fn replyChecks(a: std.mem.Allocator, dir: []const u8) !void {
    var script: Script = .{ .a = a, .ids = &.{} };
    const text, const srv = try serve(a, dir, &script);
    defer text.deinit();
    defer srv.deinit();
    try t.expect(text.text().vtable.format == .inkling2);
    try t.expectEqualSlices(u32, &.{ 200006, 199999, 200000, 200002, 200003 }, srv.eos);
    try t.expect(srv.info.capability(.thinking_switch) and srv.info.capability(.thinking_applicable));
    script.ids = try text.text().encode(a, reply_text, false);
    const body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}],\"max_tokens\":512,\"tools\":" ++ tools_json ++ "}";
    for ([_]bool{ false, true }) |stream| {
        _, const seen = try ask(a, srv, body, stream);
        try t.expectEqualStrings("Plan: look up both.", seen.reasoning);
        try t.expectEqualStrings("Checking.", seen.content);
        try t.expectEqual(@as(usize, 2), seen.calls);
        try t.expectEqualStrings("tool_calls", seen.finish);
        try t.expectEqualSlices(u32, srv.eos, script.eos);
    }
    const whole, _ = try ask(a, srv, body, false);
    const calls = whole.body.get("choices").?.array[0].get("message").?.get("tool_calls").?.array;
    try t.expectEqualStrings("weather", calls[1].get("function").?.strField("name").?);
    try t.expectEqualStrings("{\"city\":\"Rome\",\"days\":2}", calls[1].get("function").?.strField("arguments").?);
    // without tools the same calls are the request's text, not calls
    const plain = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}],\"max_tokens\":512}";
    for ([_]bool{ false, true }) |stream| {
        _, const seen = try ask(a, srv, plain, stream);
        try t.expectEqual(@as(usize, 0), seen.calls);
        try t.expect(std.mem.startsWith(u8, seen.content, "Checking.{\"name\":\"weather\""));
    }
    // a marker the model spells with ordinary tokens is text
    script.ids = try text.text().encodePlain(a, "<|content_text|>quoted<|end_message|>");
    for ([_]bool{ false, true }) |stream| {
        _, const seen = try ask(a, srv, plain, stream);
        try t.expectEqualStrings("", seen.content);
    }
    script.ids = try std.mem.concat(a, u32, &.{ &.{ 200001, 200004 }, try text.text().encodePlain(a, "a <|end_message|> b"), &.{ 200010, 200006 } });
    for ([_]bool{ false, true }) |stream| {
        _, const seen = try ask(a, srv, plain, stream);
        try t.expectEqualStrings("a <|end_message|> b", seen.content);
    }
}

/// A reply's ids: Inkling 2's marker tokens and ordinary text, each piece encoded on its own.
fn idsOf(a: std.mem.Allocator, text: *HfText, pieces: []const []const u8) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    for (pieces) |piece| {
        if (parts.Marker.of(piece)) |m| {
            try out.append(a, text.text().tokenId(m.spelling()).?);
        } else try out.appendSlice(a, try text.text().encodePlain(a, piece));
    }
    return out.items;
}

const plain_body = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":512";

/// A stop the generation never saw (it spans two parts' markers) truncates nothing; one inside a part does. (A stop
/// letter the markers spell, as d in <|end_message|>, ends the reply there, as the generation's stop check does.)
fn stopChecks(a: std.mem.Allocator, dir: []const u8) !void {
    var script: Script = .{ .a = a, .ids = &.{} };
    const text, const srv = try serve(a, dir, &script);
    defer text.deinit();
    defer srv.deinit();
    script.ids = try idsOf(a, text, &.{ "<|message_model|>", "<|content_text|>", "ab", "<|end_message|>", "<|message_model|>", "<|content_text|>", "yz", "<|end_message|>", "<|content_model_end_sampling|>" });
    const cases = [_][3][]const u8{
        .{ "[\"by\"]", "abyz", "stop" }, // the reply's text holds ab<|end_message|>...yz: no stop
        .{ "[\"z\"]", "aby", "stop" },
        .{ "[\"b<|end_message|>\"]", "a", "stop" },
    };
    for (cases) |case| for ([_]bool{ false, true }) |stream| {
        const body = try std.fmt.allocPrint(a, "{s},\"stop\":{s}}}", .{ plain_body, case[0] });
        const r, const seen = try ask(a, srv, body, stream);
        try t.expectEqual(@as(u16, 200), r.code);
        if (!std.mem.eql(u8, case[1], seen.content)) {
            std.debug.print("stop {s} stream={}: want {s}, got {s}\n", .{ case[0], stream, case[1], seen.content });
            return error.TestUnexpectedResult;
        }
        try t.expectEqualStrings(case[2], seen.finish);
    };
}

/// The rows a reply's logprobs carry, whole or streamed.
fn rowsOf(a: std.mem.Allocator, r: *const Receipt, stream: bool) ![]const V {
    if (!stream) return r.body.get("choices").?.array[0].get("logprobs").?.get("content").?.array;
    var rows: std.ArrayList(V) = .empty;
    for (r.events.items) |event| {
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const lp = choices.array[0].get("logprobs") orelse continue;
        if (lp == .object) if (lp.get("content")) |c| try rows.appendSlice(a, c.array);
    }
    return rows.items;
}

/// Logprob rows for the content's tokens only: never a marker's, a header name's or the reasoning's.
fn logprobChecks(a: std.mem.Allocator, dir: []const u8) !void {
    var script: Script = .{ .a = a, .ids = &.{}, .logprobs = true };
    const text, const srv = try serve(a, dir, &script);
    defer text.deinit();
    defer srv.deinit();
    const head = try idsOf(a, text, &.{ "<|message_model|>", "<|content_thinking|>", "hm", "<|end_message|>", "<|message_model|>", "<|content_text|>" });
    const said = try idsOf(a, text, &.{"Hi there"});
    script.ids = try std.mem.concat(a, u32, &.{ head, said, try idsOf(a, text, &.{ "<|end_message|>", "<|content_model_end_sampling|>" }) });
    for ([_][]const u8{ ",\"chat_template_kwargs\":{\"enable_thinking\":false}", "" }) |switch_| for ([_]bool{ false, true }) |stream| {
        const body = try std.fmt.allocPrint(a, "{s},\"logprobs\":true,\"top_logprobs\":1{s}}}", .{ plain_body, switch_ });
        const r, const seen = try ask(a, srv, body, stream);
        if (r.code != 200) {
            std.debug.print("logprobs{s} stream={}: {d} {s}\n", .{ switch_, stream, r.code, try json.stringify(a, r.body, .{ .ascii = false, .compact = true }) });
            return error.TestUnexpectedResult;
        }
        try t.expectEqualStrings("Hi there", seen.content);
        const rows = try rowsOf(a, &r, stream);
        var spelled: std.ArrayList(u8) = .empty;
        for (rows) |row| try spelled.appendSlice(a, row.strField("token").?);
        if (rows.len != said.len or !std.mem.eql(u8, "Hi there", spelled.items)) {
            std.debug.print("logprobs{s} stream={}: {d} rows spell {s}\n", .{ switch_, stream, rows.len, spelled.items });
            return error.TestUnexpectedResult;
        }
        for (rows, head.len..) |row, i| try t.expectEqual(-@as(f64, @floatFromInt(i + 1)) / 4, row.get("logprob").?.float);
    };
    // a call to a tool the request did not offer stays content, and so do its tokens' rows; its header name is neither
    script.ids = try idsOf(a, text, &.{ "<|message_model|>", "<|content_text|>", "Hi", "<|end_message|>", "<|message_model|>", "w", "<|content_invoke_tool_json|>", "{}", "<|end_message|>", "<|content_model_end_sampling|>" });
    for ([_]bool{ false, true }) |stream| {
        const r, const seen = try ask(a, srv, plain_body ++ ",\"logprobs\":true}", stream);
        try t.expectEqualStrings("Hi{}", seen.content);
        var spelled: std.ArrayList(u8) = .empty;
        for (try rowsOf(a, &r, stream)) |row| try spelled.appendSlice(a, row.strField("token").?);
        try t.expectEqualStrings("Hi{}", spelled.items);
    }
}

/// The response template's strip: false, for reasoning_content and content alike.
fn whitespaceChecks(a: std.mem.Allocator, dir: []const u8) !void {
    var script: Script = .{ .a = a, .ids = &.{} };
    const text, const srv = try serve(a, dir, &script);
    defer text.deinit();
    defer srv.deinit();
    const cases = [_][2][]const u8{ .{ "  Plan it.\n", " Hi \n" }, .{ " \n ", "\nok" } };
    for (cases) |case| {
        script.ids = try idsOf(a, text, &.{ "<|message_model|>", "<|content_thinking|>", case[0], "<|end_message|>", "<|message_model|>", "<|content_text|>", case[1], "<|end_message|>", "<|content_model_end_sampling|>" });
        for ([_]bool{ false, true }) |stream| {
            _, const seen = try ask(a, srv, plain_body ++ "}", stream);
            try t.expectEqualStrings(case[0], seen.reasoning);
            try t.expectEqualStrings(case[1], seen.content);
        }
    }
}

/// The effort line each request's thinking switch and effort give, and the image refusal.
fn promptChecks(a: std.mem.Allocator, dir: []const u8, line: []const u8, efforts: []const [2][]const u8) !void {
    var script: Script = .{ .a = a, .ids = &.{ 200001, 200004, 200010, 200006 } };
    const text, const srv = try serve(a, dir, &script);
    defer text.deinit();
    defer srv.deinit();
    for (efforts) |case| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"max_tokens\":8{s}}}", .{case[0]});
        const r, _ = try ask(a, srv, body, false);
        try t.expectEqual(@as(u16, 200), r.code);
        const prompt = try text.text().decode(a, script.prompt);
        const want = try std.mem.concat(a, u8, &.{ line, case[1], "<|end_message|>" });
        if (std.mem.indexOf(u8, prompt, want) == null) {
            std.debug.print("{s}: want {s} in {s}\n", .{ case[0], want, prompt });
            return error.TestUnexpectedResult;
        }
    }
    for ([_][]const u8{
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}",
        "{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\",\"format\":\"wav\"}}",
    }) |part| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"What?\"}},{s}]}}],\"max_tokens\":8}}", .{part});
        const r, _ = try ask(a, srv, body, false);
        try t.expectEqual(@as(u16, 400), r.code);
        const message = r.body.get("error").?.strField("message").?;
        if (std.mem.indexOf(u8, message, parts.image_refusal) == null) {
            std.debug.print("refusal: {s}\n", .{message});
            return error.TestUnexpectedResult;
        }
    }
}

const switches = [_][2][]const u8{
    .{ "", "default" },
    .{ ",\"reasoning_effort\":\"low\"", "low" },
    .{ ",\"reasoning_effort\":\"max\"", "max" },
    .{ ",\"reasoning_effort\":\"none\"", "none" },
    .{ ",\"chat_template_kwargs\":{\"enable_thinking\":false}", "none" },
    .{ ",\"chat_template_kwargs\":{\"enable_thinking\":true},\"reasoning_effort\":\"minimal\"", "minimal" },
};

test "Inkling 2 text: typed reply parts, whole and streamed, through the OpenAI handler" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try replyChecks(a, f.path);
}

test "Inkling 2 text: a stop spanning two parts truncates nothing, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try stopChecks(a, f.path);
}

test "Inkling 2 text: logprobs carry the content tokens only, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try logprobChecks(a, f.path);
}

test "Inkling 2 text: reasoning and content keep their whitespace, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try whitespaceChecks(a, f.path);
}

test "Inkling 2 text: thinking switches and efforts reach the effort line; images are refused" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a);
    defer f.tmp.cleanup();
    try promptChecks(a, f.path, "<|message_system|><|content_text|>effort=", &switches);
}

test "Inkling 2 text: the release tokenizer and template (TF_INKLING2_DIR)" {
    const dir = std.c.getenv("TF_INKLING2_DIR") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try replyChecks(a, std.mem.span(dir));
    try stopChecks(a, std.mem.span(dir));
    try logprobChecks(a, std.mem.span(dir));
    try whitespaceChecks(a, std.mem.span(dir));
    // the release template's numbers: its own default 0.9, effort names through its map, thinking off as 0
    const numbers = [_][2][]const u8{
        .{ "", "0.9" },
        .{ ",\"reasoning_effort\":\"low\"", "0.2" },
        .{ ",\"reasoning_effort\":\"medium\"", "0.7" },
        .{ ",\"reasoning_effort\":\"max\"", "0.99" },
        .{ ",\"reasoning_effort\":\"none\"", "0" },
        .{ ",\"chat_template_kwargs\":{\"enable_thinking\":false}", "0" },
        .{ ",\"chat_template_kwargs\":{\"enable_thinking\":true},\"reasoning_effort\":\"minimal\"", "0.1" },
    };
    try promptChecks(a, std.mem.span(dir), "<|message_system|><|content_text|>Thinking effort level: ", &numbers);
}
