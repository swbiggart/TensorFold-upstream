//! Inkling 2's reply grammar through the real OpenAI handler, with a character tokenizer holding all 60 of its added
//! tokens at their release ids: the grammar text each request compiles (fixtures/inkling2_grammar_vectors.json, the
//! masked oracle's own grammars), the requests that keep no grammar exactly as before, the 400s, and replies the
//! grammar admits read back into their fields. No grammar library is loaded: a stand-in records each compile.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const openai = @import("openai.zig");
const http = @import("http_conn.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const inkling2 = @import("inkling2_grammar.zig");
const parts = @import("inkling2_reply.zig");
const V = json.Value;
const t = std.testing;

/// Each case's OpenAI request and the grammar text the masked oracle (tools/zig/inkling2_ref.py) compiles for it.
const vectors = @embedFile("fixtures/inkling2_grammar_vectors.json");

/// Inkling 2's named added tokens at their release ids; every other id in 199998..200057 is an unused one.
const named = [_]struct { u32, []const u8 }{
    .{ 199999, "<|endoftext|>" },          .{ 200000, "<|message_user|>" },       .{ 200001, "<|message_model|>" },
    .{ 200002, "<|message_system|>" },     .{ 200003, "<|message_tool|>" },       .{ 200004, "<|content_text|>" },
    .{ 200005, "<|content_image|>" },      .{ 200006, "<|content_model_end_sampling|>" }, .{ 200008, "<|content_thinking|>" },
    .{ 200010, "<|end_message|>" },        .{ 200022, "<|content_tool_error|>" }, .{ 200024, "<|content_xml|>" },
    .{ 200028, "<|begin_of_text|>" },      .{ 200049, "<|content_invoke_tool_json|>" }, .{ 200057, "<|content_invoke_tool_text|>" },
};

fn addedText(a: std.mem.Allocator, id: u32) ![]const u8 {
    for (named) |n| if (n[0] == id) return n[1];
    return if (id == 199998) "<|unused|>" else std.fmt.allocPrint(a, "<|unused_{d}|>", .{id});
}

const template =
    \\{#- <|content_invoke_tool_json|> Thinking effort level: "none" "minimal" "low" "medium" "high" "xhigh" "max" -#}
    \\{%- for m in messages -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>{%- endfor -%}
    \\<|message_system|><|content_text|>effort={{ reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'default' }}<|end_message|>
;

/// A checkpoint folder: printable ASCII and newline as ids 0..95, the 60 added tokens, and `extra` added tokens.
const Fixture = struct {
    tmp: t.TmpDir,
    path: []u8,

    fn init(a: std.mem.Allocator, extra: []const struct { u32, []const u8 }) !Fixture {
        var tok: std.ArrayList(u8) = .empty;
        try tok.appendSlice(a, "{\"model\":{\"type\":\"BPE\",\"merges\":[],\"vocab\":{\"\\n\":95");
        for (0x20..0x7f) |c| {
            try tok.appendSlice(a, ",\"");
            if (c == '"' or c == '\\') try tok.append(a, '\\');
            try tok.append(a, @intCast(c));
            try tok.print(a, "\":{d}", .{c - 0x20});
        }
        var added: std.ArrayList(struct { u32, []const u8 }) = .empty;
        for (199998..200058) |id| try added.append(a, .{ @intCast(id), try addedText(a, @intCast(id)) });
        try added.appendSlice(a, extra);
        for (added.items) |m| try tok.print(a, ",\"{s}\":{d}", .{ m[1], m[0] });
        try tok.appendSlice(a, "}},\"decoder\":{\"type\":\"Fuse\"},\"added_tokens\":[");
        for (added.items, 0..) |m, i| try tok.print(a, "{s}{{\"id\":{d},\"content\":\"{s}\",\"special\":true}}", .{ if (i > 0) "," else "", m[0], m[1] });
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

/// The engine: records what each request carries, then answers `ids`.
const Engine = struct {
    a: std.mem.Allocator,
    ids: []const u32 = &.{},
    submits: usize = 0,
    prompt: []const u32 = &.{},
    eos: []const u32 = &.{},
    max_tokens: u32 = 0,
    structured: bool = false,
    kind: []const u8 = "",
    field: []const u8 = "",
    text: []const u8 = "",
    after: ?u32 = null,

    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768, .reasoning = true };
    }
    fn submit(ptr: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        e.submits += 1;
        // copied now: a request's slices only last until `finished`
        e.prompt = e.a.dupe(u32, request.prompt) catch return error.Busy;
        e.eos = e.a.dupe(u32, request.eos) catch return error.Busy;
        e.max_tokens = request.max_tokens;
        e.structured = request.structure != null;
        if (request.structure) |st| {
            e.kind = @tagName(st.kind);
            e.field = e.a.dupe(u8, st.field) catch return error.Busy;
            e.text = e.a.dupe(u8, st.text) catch return error.Busy;
            e.after = st.after;
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.ids) |token| sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
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

/// The grammar library's stand-in: counts compiles and keeps the last text; grammar_budget is not built in.
const Compile = struct {
    count: usize = 0,
    kind: c_int = -1,

    fn compiler(c: *Compile) api.grammar.Compiler {
        var out: api.grammar.Compiler = undefined;
        out.h = c;
        out.words = 6282;
        out.mutex = .init;
        out.f.compile = compile;
        out.f.free = free;
        out.f.close = free;
        out.f.budget = null;
        out.f.budget_ready = null;
        return out;
    }
    fn compile(ptr: *anyopaque, kind: c_int, _: [*]const u8, _: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque {
        const c: *Compile = @ptrCast(@alignCast(ptr));
        c.count += 1;
        c.kind = kind;
        return ptr;
    }
    fn free(_: *anyopaque) callconv(.c) void {}
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

const Seen = struct { reasoning: []const u8 = "", content: []const u8 = "", calls: []const [2][]const u8 = &.{}, finish: []const u8 = "" };

/// One request through the OpenAI handler, whole or streamed: its status and the reply's fields.
fn ask(a: std.mem.Allocator, srv: *Server, body: V, stream: bool) !struct { Receipt, Seen } {
    const sent = try json.deepCopy(a, body);
    if (stream) try sent.object.put(a, "stream", .{ .bool = true });
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var r: Receipt = .{ .a = a };
    openai.run(srv, a, r.out(), .{ .conn = &conn }, true, sent);
    if (r.code != 200) return .{ r, .{} };
    var calls: std.ArrayList([2][]const u8) = .empty;
    if (!stream) {
        const choice = r.body.get("choices").?.array[0];
        const message = choice.get("message").?;
        if (message.get("tool_calls")) |list| for (list.array) |c| try calls.append(a, .{ c.get("function").?.strField("name").?, c.get("function").?.strField("arguments").? });
        return .{ r, .{ .reasoning = message.strField("reasoning_content") orelse "", .content = message.strField("content") orelse "", .calls = calls.items, .finish = choice.strField("finish_reason") orelse "" } };
    }
    var content: std.ArrayList(u8) = .empty;
    var reason: std.ArrayList(u8) = .empty;
    var args: std.ArrayList(std.ArrayList(u8)) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    var finish: []const u8 = "";
    for (r.events.items) |event| {
        const choices = event.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        if (choices.array[0].strField("finish_reason")) |f| finish = f;
        const delta = choices.array[0].get("delta") orelse continue;
        if (delta.strField("content")) |s| try content.appendSlice(a, s);
        if (delta.strField("reasoning_content")) |s| try reason.appendSlice(a, s);
        if (delta.get("tool_calls")) |list| for (list.array) |c| {
            const f = c.get("function").?;
            if (f.strField("name")) |n| {
                try names.append(a, n);
                try args.append(a, .empty);
            }
            if (f.strField("arguments")) |s| try args.items[args.items.len - 1].appendSlice(a, s);
        };
    }
    for (names.items, args.items) |n, x| try calls.append(a, .{ n, x.items });
    return .{ r, .{ .reasoning = reason.items, .content = content.items, .calls = calls.items, .finish = finish } };
}

const Served = struct { text: *HfText, srv: *Server };

fn serve(a: std.mem.Allocator, dir: []const u8, engine: *Engine, compile: *Compile, compiler: *api.grammar.Compiler) !Served {
    var problem: []const u8 = "";
    const text = HfText.load(t.allocator, t.io, dir, null, a, &problem) catch |e| {
        std.debug.print("load: {s}\n", .{problem});
        return e;
    };
    errdefer text.deinit();
    const srv = try Server.init(a, t.io, engine.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
    srv.info.structures = true; // the backend's first_masked (S1)
    compiler.* = compile.compiler();
    srv.grammars = compiler;
    return .{ .text = text, .srv = srv };
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

fn parse(a: std.mem.Allocator, s: []const u8) !V {
    return (try json.parse(a, s)).ok;
}

/// The ids the grammar names and all 60 added tokens, from the tokenizer in `dir`.
fn idChecks(a: std.mem.Allocator, dir: []const u8) !void {
    var engine: Engine = .{ .a = a };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, dir, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    const ids = s.srv.inkling2_ids orelse return error.TestExpectedIds;
    try t.expectEqual(@as(u32, 200001), ids.message_model);
    try t.expectEqual(@as(u32, 200008), ids.content_thinking);
    try t.expectEqual(@as(u32, 200004), ids.content_text);
    try t.expectEqual(@as(u32, 200049), ids.invoke_tool_json);
    try t.expectEqual(@as(u32, 200010), ids.end_message);
    try t.expectEqual(@as(usize, 60), ids.added.len);
    for (ids.added, 199998..) |id, want| try t.expectEqual(@as(u32, @intCast(want)), id);
    try t.expectEqualSlices(u32, &.{ 200006, 199999, 200000, 200002, 200003 }, s.srv.eos);
}

/// Each vector's request through the handler compiles the masked oracle's grammar text, byte for byte.
fn vectorChecks(a: std.mem.Allocator, dir: []const u8) !void {
    var engine: Engine = .{ .a = a };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, dir, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    const cases = try parse(a, vectors);
    try t.expectEqual(@as(usize, 7), cases.array.len);
    var thinking_on: usize = 0;
    for (cases.array) |case| {
        const name = case.strField("name").?;
        const body = case.get("request").?;
        try body.object.put(a, "model", .{ .string = "m" });
        if (body.strField("reasoning_effort") == null or !std.mem.eql(u8, body.strField("reasoning_effort").?, "none")) thinking_on += 1;
        const tools = body.get("tools");
        // a reply the grammar admits: the first offered (or the named) tool's call, else a JSON body
        engine.ids = if (tools) |list| blk: {
            const choice = body.get("tool_choice").?;
            const tool = if (choice == .object) choice.get("function").?.strField("name").? else list.array[0].get("function").?.strField("name").?;
            const call = try std.fmt.allocPrint(a, "{{\"name\":\"{s}\",\"args\":{{}}}}", .{tool});
            break :blk try idsOf(a, s.text, &.{ "<|message_model|>", tool, "<|content_invoke_tool_json|>", call, "<|end_message|>", "<|content_model_end_sampling|>" });
        } else try idsOf(a, s.text, &.{ "<|message_model|>", "<|content_text|>", "{\"a\":1}", "<|end_message|>", "<|content_model_end_sampling|>" });
        const before = compile.count;
        const r, const seen = try ask(a, s.srv, body, false);
        if (r.code != 200) {
            std.debug.print("{s}: {d} {s}\n", .{ name, r.code, try json.stringify(a, r.body, .{ .ascii = false, .compact = true }) });
            return error.TestUnexpectedResult;
        }
        try t.expect(engine.structured);
        try t.expectEqualStrings("structural_tag", engine.kind);
        try t.expectEqual(@as(c_int, 5), compile.kind);
        try t.expectEqual(before + 1, compile.count);
        try t.expectEqual(@as(?u32, null), engine.after);
        try t.expectEqualStrings(if (tools != null) "tool_choice" else "response_format", engine.field);
        if (!std.mem.eql(u8, case.strField("grammar").?, engine.text)) {
            std.debug.print("{s}:\nwant {s}\n got {s}\n", .{ name, case.strField("grammar").?, engine.text });
            return error.TestUnexpectedResult;
        }
        if (tools != null) {
            try t.expectEqual(@as(usize, 1), seen.calls.len);
            try t.expectEqualStrings("tool_calls", seen.finish);
        } else try t.expectEqualStrings("{\"a\":1}", seen.content);
    }
    try t.expect(thinking_on >= 2 and thinking_on < cases.array.len); // both thinking switches are among the cases
}

test "Inkling 2 grammar: the ids the grammar names, and all 60 added tokens" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{});
    defer f.tmp.cleanup();
    try idChecks(a, f.path);
}

test "Inkling 2 grammar: each request compiles the masked oracle's grammar, byte for byte" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{});
    defer f.tmp.cleanup();
    try vectorChecks(a, f.path);
}

test "Inkling 2 grammar: the release tokenizer and template give the same ids and grammars (TF_INKLING2_DIR)" {
    const dir = std.c.getenv("TF_INKLING2_DIR") orelse return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try idChecks(a, std.mem.span(dir));
    try vectorChecks(a, std.mem.span(dir));
}

test "Inkling 2 grammar: thinking on or off, the same grammar; parallel calls are one or more, a single call one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{});
    defer f.tmp.cleanup();
    var engine: Engine = .{ .a = a, .ids = &.{ 200001, 200004, 91, 93, 200010, 200006 } };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, f.path, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    const base = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":512,\"response_format\":{\"type\":\"json_object\"}";
    var texts: [3][]const u8 = undefined;
    for ([_][]const u8{ "", ",\"reasoning_effort\":\"none\"", ",\"chat_template_kwargs\":{\"enable_thinking\":false}" }, &texts) |switch_, *out| {
        const r, _ = try ask(a, s.srv, try parse(a, try std.fmt.allocPrint(a, "{s}{s}}}", .{ base, switch_ })), false);
        try t.expectEqual(@as(u16, 200), r.code);
        try t.expectEqual(@as(?u32, null), engine.after);
        out.* = engine.text;
    }
    try t.expectEqualStrings(texts[0], texts[1]);
    try t.expectEqualStrings(texts[0], texts[2]);
    // the optional thinking part, then <|message_model|><|content_text|>, the body and <|end_message|>
    try t.expect(std.mem.startsWith(u8, texts[0], "{\"type\":\"structural_tag\",\"format\":{\"type\":\"sequence\",\"elements\":[{\"type\":\"optional\",\"content\":{\"type\":\"sequence\",\"elements\":[{\"type\":\"token\",\"token\":200001},{\"type\":\"token\",\"token\":200008},{\"type\":\"any_tokens\",\"exclude_tokens\":[199998,199999,"));
    try t.expect(std.mem.endsWith(u8, texts[0], "{\"type\":\"token\",\"token\":200001},{\"type\":\"token\",\"token\":200004},{\"type\":\"json_schema\",\"json_schema\":{\"type\":\"object\"},\"max_whitespace_cnt\":32,\"excludes\":[\"<|message_\",\"<|content_\",\"<|end\",\"<|begin_of_text|>\",\"<|unused\"]},{\"type\":\"token\",\"token\":200010}]}}"));
    // tools: one call to the named function; "required" allows one or more unless parallel_tool_calls is false
    const tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}},{\"type\":\"function\",\"function\":{\"name\":\"b\"}}]";
    engine.ids = try idsOf(a, s.text, &.{ "<|message_model|>", "a", "<|content_invoke_tool_json|>", "{\"name\":\"a\",\"args\":{}}", "<|end_message|>", "<|content_model_end_sampling|>" });
    const Want = struct { []const u8, bool, bool }; // the request's tool fields; plus; or
    for ([_]Want{
        .{ ",\"tool_choice\":\"required\"", true, true },
        .{ ",\"tool_choice\":\"required\",\"parallel_tool_calls\":false", false, true },
        .{ ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"a\"}}", true, false },
        .{ ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"a\"}},\"parallel_tool_calls\":false", false, false },
    }) |w| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"max_tokens\":512,{s}{s}}}", .{ tools, w[0] });
        const r, const seen = try ask(a, s.srv, try parse(a, body), false);
        try t.expectEqual(@as(u16, 200), r.code);
        try t.expectEqual(w[1], std.mem.indexOf(u8, engine.text, "{\"type\":\"plus\"") != null);
        try t.expectEqual(w[2], std.mem.indexOf(u8, engine.text, "{\"type\":\"or\"") != null);
        try t.expectEqual(w[2], std.mem.indexOf(u8, engine.text, "{\"type\":\"const_string\",\"value\":\"b\"}") != null);
        try t.expectEqual(@as(usize, 1), seen.calls.len);
    }
}

test "Inkling 2 grammar: a plain request and an auto-tools request keep today's grammar-free path" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{});
    defer f.tmp.cleanup();
    var engine: Engine = .{ .a = a };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, f.path, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    const strict = "[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"strict\":true,\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"additionalProperties\":false}}}]";
    const loose = "[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"parameters\":{\"type\":\"object\"}}}]";
    const head = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather in Oslo?\"}],\"max_tokens\":512";
    const bodies = [_][]const u8{
        head ++ "}",
        head ++ ",\"reasoning_effort\":\"none\"}",
        head ++ ",\"tools\":" ++ strict ++ "}",
        head ++ ",\"tools\":" ++ strict ++ ",\"tool_choice\":\"auto\"}",
        head ++ ",\"tools\":" ++ strict ++ ",\"tool_choice\":\"auto\",\"parallel_tool_calls\":false}",
        head ++ ",\"tools\":" ++ loose ++ "}",
        head ++ ",\"tools\":" ++ strict ++ ",\"tool_choice\":\"none\"}",
        head ++ ",\"response_format\":{\"type\":\"text\"}}",
    };
    const replies = [_][]const []const u8{
        &.{ "<|message_model|>", "<|content_thinking|>", "Look it up.", "<|end_message|>", "<|message_model|>", "weather", "<|content_invoke_tool_json|>", "{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\"}}", "<|end_message|>", "<|content_model_end_sampling|>" },
        &.{ "<|message_model|>", "<|content_text|>", "Rain.", "<|end_message|>", "<|content_model_end_sampling|>" },
    };
    for (bodies) |body| for (replies) |reply| for ([_]bool{ false, true }) |stream| {
        engine.ids = try idsOf(a, s.text, reply);
        // today: the engine offers no grammar (Inkling 2's backend had no first_masked)
        s.srv.info.structures = false;
        const r0, const seen0 = try ask(a, s.srv, try parse(a, body), stream);
        const today = engine;
        // with structures on: the same request, prompt, stops and reply, no grammar, and no compiler is consulted
        s.srv.info.structures = true;
        s.srv.grammars = null;
        const r1, const seen1 = try ask(a, s.srv, try parse(a, body), stream);
        try t.expectEqual(@as(u16, 200), r0.code);
        try t.expectEqual(r0.code, r1.code);
        try t.expect(!today.structured and !engine.structured);
        try t.expectEqualSlices(u32, today.prompt, engine.prompt);
        try t.expectEqualSlices(u32, today.eos, engine.eos);
        try t.expectEqual(today.max_tokens, engine.max_tokens);
        try t.expectEqualStrings(seen0.reasoning, seen1.reasoning);
        try t.expectEqualStrings(seen0.content, seen1.content);
        try t.expectEqualStrings(seen0.finish, seen1.finish);
        try t.expectEqual(seen0.calls.len, seen1.calls.len);
        for (seen0.calls, seen1.calls) |x, y| {
            try t.expectEqualStrings(x[0], y[0]);
            try t.expectEqualStrings(x[1], y[1]);
        }
        try t.expect(s.srv.grammars == null and s.srv.grammar_problem == null); // the library was never opened
        try t.expectEqual(@as(usize, 0), compile.count);
    };
}

test "Inkling 2 grammar: regex, choice, EBNF and grammar_budget are refused with a 400, before the engine" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{});
    defer f.tmp.cleanup();
    var engine: Engine = .{ .a = a };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, f.path, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    const head = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":512";
    const cases = [_][2][]const u8{
        .{ head ++ ",\"guided_regex\":\"a+\"}", "guided_regex (regex) is not supported for Inkling 2 yet" },
        .{ head ++ ",\"guided_choice\":[\"a\",\"b\"]}", "guided_choice (choice) is not supported for Inkling 2 yet" },
        .{ head ++ ",\"guided_grammar\":\"root ::= \\\"a\\\"\"}", "guided_grammar (grammar) is not supported for Inkling 2 yet" },
        .{ head ++ ",\"structured_outputs\":{\"regex\":\"a+\"}}", "structured_outputs (regex) is not supported for Inkling 2 yet" },
        .{ head ++ ",\"structured_outputs\":{\"choice\":[\"a\"]}}", "structured_outputs (choice) is not supported for Inkling 2 yet" },
        .{ head ++ ",\"response_format\":{\"type\":\"json_object\"},\"grammar_budget\":true}", "grammar_budget" },
        .{ head ++ ",\"response_format\":{\"type\":\"structural_tag\"}}", "response_format type must be text, json_object or json_schema" },
    };
    for (cases) |c| for ([_]bool{ false, true }) |stream| {
        const r, _ = try ask(a, s.srv, try parse(a, c[0]), stream);
        try t.expectEqual(@as(u16, 400), r.code);
        const message = r.body.get("error").?.strField("message").?;
        if (std.mem.indexOf(u8, message, c[1]) == null) {
            std.debug.print("{s}: {s}\n", .{ c[0], message });
            return error.TestUnexpectedResult;
        }
    };
    try t.expectEqual(@as(usize, 0), engine.submits);
}

test "Inkling 2 grammar: an added token outside the excluded prefixes leaves the grammar off, with a 400" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{.{ 200058, "<|tool_new|>" }});
    defer f.tmp.cleanup();
    var engine: Engine = .{ .a = a };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, f.path, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    try t.expect(s.srv.inkling2_ids == null);
    const r, _ = try ask(a, s.srv, try parse(a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"response_format\":{\"type\":\"json_object\"}}"), false);
    try t.expectEqual(@as(u16, 400), r.code);
    try t.expect(std.mem.indexOf(u8, r.body.get("error").?.strField("message").?, "marker tokens do not match Inkling 2's") != null);
    try t.expectEqual(@as(usize, 0), engine.submits + compile.count);
}

test "Inkling 2 grammar: replies the grammar admits read back into reasoning_content, content and tool_calls" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.init(a, &.{});
    defer f.tmp.cleanup();
    var engine: Engine = .{ .a = a };
    var compile: Compile = .{};
    var compiler: api.grammar.Compiler = undefined;
    const s = try serve(a, f.path, &engine, &compile, &compiler);
    defer s.text.deinit();
    defer s.srv.deinit();
    const head = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":512";
    const schema = head ++ ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"w\",\"schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"c\":{\"type\":\"number\"}},\"required\":[\"city\",\"c\"]}}}}";
    const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"time\",\"parameters\":{\"type\":\"object\",\"properties\":{\"zone\":{\"type\":\"string\"}}}}}]";
    const think = [_][]const u8{ "<|message_model|>", "<|content_thinking|>", "Paris, mild: say so as JSON.", "<|end_message|>" };
    const body = [_][]const u8{ "<|message_model|>", "<|content_text|>", "{\"city\": \"Paris\", \"c\": 12.5}", "<|end_message|>", "<|content_model_end_sampling|>" };
    const weather = [_][]const u8{ "<|message_model|>", "weather", "<|content_invoke_tool_json|>", "{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\",\"days\":2}}", "<|end_message|>" };
    const time = [_][]const u8{ "<|message_model|>", "time", "<|content_invoke_tool_json|>", "{\"name\":\"time\",\"args\":{\"zone\":\"Asia/Tokyo\"}}", "<|end_message|>" };
    const end = [_][]const u8{"<|content_model_end_sampling|>"};
    const Case = struct { body: []const u8, reply: []const []const u8, seen: Seen };
    const cases = [_]Case{
        .{ .body = head ++ ",\"response_format\":{\"type\":\"json_object\"}}", .reply = &body, .seen = .{ .content = "{\"city\": \"Paris\", \"c\": 12.5}", .finish = "stop" } },
        .{ .body = schema, .reply = &(think ++ body), .seen = .{ .reasoning = "Paris, mild: say so as JSON.", .content = "{\"city\": \"Paris\", \"c\": 12.5}", .finish = "stop" } },
        .{ .body = head ++ ",\"reasoning_effort\":\"none\"" ++ schema[head.len..], .reply = &body, .seen = .{ .content = "{\"city\": \"Paris\", \"c\": 12.5}", .finish = "stop" } },
        .{ .body = head ++ tools ++ ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"weather\"}}}", .reply = &(weather ++ end), .seen = .{ .calls = &.{.{ "weather", "{\"city\":\"Oslo\",\"days\":2}" }}, .finish = "tool_calls" } },
        .{ .body = head ++ tools ++ ",\"tool_choice\":\"required\"}", .reply = &(think ++ weather ++ time ++ end), .seen = .{ .reasoning = "Paris, mild: say so as JSON.", .calls = &.{ .{ "weather", "{\"city\":\"Oslo\",\"days\":2}" }, .{ "time", "{\"zone\":\"Asia/Tokyo\"}" } }, .finish = "tool_calls" } },
        .{ .body = head ++ tools ++ ",\"tool_choice\":\"required\",\"parallel_tool_calls\":false,\"reasoning_effort\":\"none\"}", .reply = &(time ++ end), .seen = .{ .calls = &.{.{ "time", "{\"zone\":\"Asia/Tokyo\"}" }}, .finish = "tool_calls" } },
    };
    for (cases) |c| for ([_]bool{ false, true }) |stream| {
        engine.ids = try idsOf(a, s.text, c.reply);
        const r, const seen = try ask(a, s.srv, try parse(a, c.body), stream);
        try t.expectEqual(@as(u16, 200), r.code);
        try t.expect(engine.structured);
        if (c.seen.content.len > 0) {
            const parsed = try json.parse(a, seen.content);
            try t.expect(parsed == .ok and parsed.ok == .object); // content is the JSON body alone
        }
        try t.expectEqualStrings(c.seen.reasoning, seen.reasoning);
        try t.expectEqualStrings(c.seen.content, seen.content);
        try t.expectEqualStrings(c.seen.finish, seen.finish);
        try t.expectEqual(c.seen.calls.len, seen.calls.len);
        for (c.seen.calls, seen.calls) |want, got| {
            try t.expectEqualStrings(want[0], got[0]);
            try t.expectEqualStrings(want[1], got[1]);
        }
    };
}

test "Inkling 2 grammar: a call's arguments' keys are sorted as the template writes them" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ids: inkling2.Ids = .{ .message_model = 1, .content_thinking = 2, .content_text = 3, .invoke_tool_json = 4, .end_message = 5, .added = &.{ 1, 2, 3, 4, 5 } };
    const tools = try parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"zeta\":{\"type\":\"object\",\"properties\":{\"y\":{},\"x\":{}}},\"Beta\":{},\"alpha\":{}},\"required\":[\"zeta\"]}}}]");
    const text = try inkling2.calls(a, ids, tools.array, false);
    // top-level keys sorted (uppercase first, as Python sorts), nested ones as given; the request's tool unchanged
    try t.expect(std.mem.indexOf(u8, text, "\"args\":{\"type\":\"object\",\"properties\":{\"Beta\":{},\"alpha\":{},\"zeta\":{\"type\":\"object\",\"properties\":{\"y\":{},\"x\":{}}}},\"required\":[\"zeta\"]}") != null);
    try t.expectEqualStrings("zeta", tools.array[0].get("function").?.get("parameters").?.get("properties").?.object.keys()[0]);
    // a tool without parameters takes no arguments: {}
    const bare = try inkling2.calls(a, ids, (try parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"now\"}}]")).array, false);
    try t.expect(std.mem.indexOf(u8, bare, "\"args\":{\"type\":\"object\",\"properties\":{}}") != null);
}
