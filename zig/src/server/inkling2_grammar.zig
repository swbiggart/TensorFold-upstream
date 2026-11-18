//! Inkling 2's structured output: its typed reply parts around the constrained body, as one structural tag.
//!
//! Inkling 2 has no end-of-thinking token: a reply is typed parts (inkling2_reply.zig), and the model opens its first
//! part itself. So the grammar covers the whole reply from its first token (`after` stays null):
//!
//!   REPLY := [THINK] <|message_model|> <|content_text|> BODY(schema) <|end_message|>
//!   TOOLS := [THINK] CALL                      (a named function, or "required" without parallel calls)
//!          | [THINK] CALL+                     ("required" with parallel calls)
//!   CALL  := <|message_model|> NAME <|content_invoke_tool_json|> BODY({"name": NAME, "args": parameters}) <|end_message|>
//!   THINK := <|message_model|> <|content_thinking|> (any token but an added one)* <|end_message|>
//!
//! THINK is optional whatever the request's thinking switch: the grammar leaves the choice to think to the model, as
//! an unconstrained reply does, and constrains only the answer. The stop token follows the grammar. BODY is xgrammar's
//! JSON schema with `excludes`, so no string or key holds an added token, as one token or spelled. A call's arguments
//! keep the template's key order: it writes their top-level keys sorted (dictsort), so the schema's properties are
//! sorted the same way.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const model_text = @import("model_text.zig");
const parts = @import("inkling2_reply.zig");
const tool_specs = @import("tool_specs.zig");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// Prefixes of every added token's text: a JSON body's strings and keys never start one of them.
pub const excludes = [_][]const u8{ "<|message_", "<|content_", "<|end", "<|begin_of_text|>", "<|unused" };

/// The most blank characters between a body's JSON tokens (grammar.py BLANKS, as tool_grammar's arguments).
const value_blanks = 32;

/// The marker tokens the grammar names, and every added token (none may appear inside a free part).
pub const Ids = struct {
    message_model: u32,
    content_thinking: u32,
    content_text: u32,
    invoke_tool_json: u32,
    end_message: u32,
    added: []const u32,

    /// The ids from the tokenizer; null when a marker is not one token or an added token's text escapes `excludes`.
    pub fn of(a: Allocator, text: model_text.Text) Allocator.Error!?Ids {
        var added: std.ArrayList(u32) = .empty;
        for (0..text.vocabSize()) |i| if (text.isAdded(@intCast(i))) {
            const spelled = try text.tokenString(a, @intCast(i));
            const covered = for (excludes) |e| {
                if (std.mem.startsWith(u8, spelled, e)) break true;
            } else false;
            if (!covered) return null;
            try added.append(a, @intCast(i));
        };
        return .{
            .message_model = marker(text, .message_model) orelse return null,
            .content_thinking = marker(text, .content_thinking) orelse return null,
            .content_text = marker(text, .content_text) orelse return null,
            .invoke_tool_json = marker(text, .content_invoke_tool_json) orelse return null,
            .end_message = marker(text, .end_message) orelse return null,
            .added = added.items,
        };
    }

    fn marker(text: model_text.Text, m: parts.Marker) ?u32 {
        const id = text.tokenId(m.spelling()) orelse return null;
        return if (text.isAdded(id)) id else null;
    }
};

/// The reply grammar for response_format json_object / json_schema (and guided_json, structured_outputs json).
pub fn structure(srv: *Server, cx: *Cx, s: grammar.Spec, thinking: bool) errors.Refused!api.Structure {
    const schema: Value = switch (s.kind) {
        .json => try anyObject(cx.a),
        .json_schema => switch (try json.parseText(cx.a, s.text)) {
            .ok => |v| v,
            .err => return cx.fail(.request, "{s} is not valid JSON", .{s.field}),
        },
        else => return cx.fail(.request, "{s} ({s}) is not supported for Inkling 2 yet: send response_format json_object or json_schema", .{ s.field, @tagName(s.kind) }),
    };
    const ids = try idsOrRefuse(srv, cx, s.field);
    return srv.compiledStructure(cx, .structural_tag, try reply(cx.a, ids, schema), s.field, thinking);
}

/// The grammar for tool_choice "required" or a named function: one call to an offered tool, or (`parallel`) several.
pub fn toolStructure(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, thinking: bool) errors.Refused!api.Structure {
    const ids = try idsOrRefuse(srv, cx, "tool_choice");
    const reason = try std.fmt.allocPrint(cx.a, "tool_choice \"required\" or a named function on backend '{s}'", .{srv.backendName()});
    var st = try srv.compiledStructure(cx, .structural_tag, try calls(cx.a, ids, tools, parallel), reason, thinking);
    st.field = "tool_choice";
    return st;
}

fn idsOrRefuse(srv: *Server, cx: *Cx, field: []const u8) errors.Refused!Ids {
    return srv.inkling2_ids orelse cx.fail(.request, "{s}: this checkpoint's marker tokens do not match Inkling 2's, so its reply cannot be constrained", .{field});
}

/// REPLY(schema) as xgrammar's structural tag JSON text.
pub fn reply(a: Allocator, ids: Ids, schema: Value) Allocator.Error![]const u8 {
    return tag(a, try list(a, &.{
        try node(a, "optional", &.{.{ "content", try think(a, ids) }}),
        try token(a, ids.message_model),
        try token(a, ids.content_text),
        try body(a, schema),
        try token(a, ids.end_message),
    }));
}

/// TOOLS(tools) as xgrammar's structural tag JSON text.
pub fn calls(a: Allocator, ids: Ids, tools: []const Value, parallel: bool) Allocator.Error![]const u8 {
    const each = try a.alloc(Value, tools.len);
    for (tools, each) |t, *c| c.* = try call(a, ids, t);
    const one = if (each.len == 1) each[0] else try node(a, "or", &.{.{ "elements", .{ .array = each } }});
    const x = if (parallel) try node(a, "plus", &.{.{ "content", one }}) else one;
    return tag(a, try list(a, &.{ try node(a, "optional", &.{.{ "content", try think(a, ids) }}), x }));
}

/// <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>.
fn call(a: Allocator, ids: Ids, tool: Value) Allocator.Error!Value {
    const name = try tool_specs.toolName(a, tool);
    const fixed = try json.newObject(a);
    try fixed.put(a, "const", .{ .string = name });
    const props = try json.newObject(a);
    try props.put(a, "name", .{ .object = fixed });
    try props.put(a, "args", try sortedProperties(a, try @import("tool_grammar.zig").parameters(a, tool)));
    const whole = try json.newObject(a);
    try whole.put(a, "type", .{ .string = "object" });
    try whole.put(a, "properties", .{ .object = props });
    try whole.put(a, "required", try list(a, &.{ .{ .string = "name" }, .{ .string = "args" } }));
    try whole.put(a, "additionalProperties", .{ .bool = false });
    return node(a, "sequence", &.{.{ "elements", try list(a, &.{
        try token(a, ids.message_model),
        try constant(a, name),
        try token(a, ids.invoke_tool_json),
        try body(a, .{ .object = whole }),
        try token(a, ids.end_message),
    }) }});
}

/// The schema with its top-level properties in sorted key order (the template's dictsort of a call's arguments).
fn sortedProperties(a: Allocator, schema: Value) Allocator.Error!Value {
    const props = schema.get("properties") orelse return schema;
    if (props != .object) return schema;
    const keys = try a.dupe([]const u8, props.object.keys());
    std.mem.sort([]const u8, keys, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    const sorted = try json.newObject(a);
    for (keys) |k| try sorted.put(a, k, props.object.get(k).?);
    const out = try json.copyObject(a, schema.object);
    try out.put(a, "properties", .{ .object = sorted });
    return .{ .object = out };
}

/// A free thinking part: any tokens but added ones, up to its end marker.
fn think(a: Allocator, ids: Ids) Allocator.Error!Value {
    const exclude = try a.alloc(Value, ids.added.len);
    for (ids.added, exclude) |id, *e| e.* = try json.intValue(a, id);
    return node(a, "sequence", &.{.{ "elements", try list(a, &.{
        try token(a, ids.message_model),
        try token(a, ids.content_thinking),
        try node(a, "any_tokens", &.{.{ "exclude_tokens", .{ .array = exclude } }}),
        try token(a, ids.end_message),
    }) }});
}

fn body(a: Allocator, schema: Value) Allocator.Error!Value {
    const ex = try a.alloc(Value, excludes.len);
    for (excludes, ex) |e, *v| v.* = .{ .string = e };
    return node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "max_whitespace_cnt", try json.intValue(a, value_blanks) }, .{ "excludes", .{ .array = ex } } });
}

/// json_object's schema, {"type":"object"}: any object (strict mode closes an object only beside its properties).
fn anyObject(a: Allocator) Allocator.Error!Value {
    return node(a, "object", &.{});
}

/// {"type":"structural_tag","format":{"type":"sequence","elements":...}} as compact JSON text.
fn tag(a: Allocator, elements: Value) Allocator.Error![]const u8 {
    const format = try node(a, "sequence", &.{.{ "elements", elements }});
    return json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

fn token(a: Allocator, id: u32) Allocator.Error!Value {
    return node(a, "token", &.{.{ "token", try json.intValue(a, id) }});
}

fn constant(a: Allocator, text: []const u8) Allocator.Error!Value {
    return node(a, "const_string", &.{.{ "value", .{ .string = text } }});
}

fn list(a: Allocator, items: []const Value) Allocator.Error!Value {
    return .{ .array = try a.dupe(Value, items) };
}

/// {"type": kind, fields...}.
fn node(a: Allocator, kind: []const u8, entries: []const struct { []const u8, Value }) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = kind });
    for (entries) |e| try o.put(a, e[0], e[1]);
    return .{ .object = o };
}
