//! tf-inkling2-plan: Inkling-2-Small's rank plan, checked against a checkpoint folder or the release's inventory.
const std = @import("std");
const ink = @import("inkling2");

const usage =
    \\usage: tf-inkling2-plan [options]
    \\  --dir DIR          read DIR's config.json, hf_quant_config.json and safetensors headers
    \\                     (default: the release's files as embedded at build time)
    \\  --inventory FILE   the inventory's text form (tools/zig/inkling2_inventory.py) instead of DIR's headers
    \\  --world N          ranks, experts split evenly (default 4)
    \\  --experts A,B,..   experts a rank, in rank order (sets the world size), e.g. 93,97,97,97
    \\  --experts split    every expert on every rank, a 1/W row slice of each (w13 gate/up pairs, w2 output rows,
    \\                     their scales; amax whole): the checkpoint-mode split layout (--experts whole: the default)
    \\  --drafter R        the rank holding the DSpark drafter (default: the last)
    \\  --positions N      KV pool positions (default 524288)
    \\  --streams N        streams for the per-stream state line (default 8)
    \\  --ranges R         print rank R's byte ranges instead of the table
    \\  --digests OUT      hash every rank's pieces from --dir (the source checkpoint) into OUT, the content-digest
    \\                     manifest tf-inkling2-cuda checks its loaded bytes against (one file for all ranks)
    \\  --layers A:B       with --digests: only the pieces of layers A .. B-1
    \\  --jobs N           with --digests: reading threads (default 8)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var dir: ?[]const u8 = null;
    var inventory_path: ?[]const u8 = null;
    var want: ink.describe.Want = .{};
    var positions: u64 = 524288;
    var streams: u64 = 8;
    var ranges: ?u32 = null;
    var digests: ?[]const u8 = null;
    var layers: ?[2]u32 = null;
    var jobs: usize = 8;
    var counts: std.ArrayList(u32) = .empty;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--help") or std.mem.eql(u8, k, "-h")) {
            std.debug.print("{s}", .{usage});
            return 0;
        }
        if (i + 1 >= args.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--dir")) dir = v else if (std.mem.eql(u8, k, "--inventory")) inventory_path = v else if (std.mem.eql(u8, k, "--world")) {
            want.world = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--experts")) {
            if (std.meta.stringToEnum(ink.plan.Layout, v)) |l| {
                want.layout = l;
                continue;
            }
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |n| try counts.append(a, try std.fmt.parseInt(u32, n, 10));
        } else if (std.mem.eql(u8, k, "--drafter")) {
            want.drafter = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--positions")) {
            positions = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, k, "--streams")) {
            streams = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, k, "--ranges")) {
            ranges = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--digests")) {
            digests = v;
        } else if (std.mem.eql(u8, k, "--jobs")) {
            jobs = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--layers")) {
            const colon = std.mem.indexOfScalar(u8, v, ':') orelse return 2;
            layers = .{ try std.fmt.parseInt(u32, v[0..colon], 10), try std.fmt.parseInt(u32, v[colon + 1 ..], 10) };
        } else {
            std.debug.print("unknown option {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    want.experts = counts.items;
    var d: ink.Diag = .{};
    var model = open(a, gpa, io, dir, inventory_path, want, &d) catch |e| {
        std.debug.print("FAIL tf-inkling2-plan: {t}: {s}\n", .{ e, d.text() });
        return 1;
    };
    defer model.deinit();
    var buf: [8192]u8 = undefined;
    if (digests) |path| {
        if (dir == null or inventory_path != null) {
            std.debug.print("--digests reads the source checkpoint: give --dir (and no --inventory)\n", .{});
            return 2;
        }
        const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{path});
        const file = try std.Io.Dir.cwd().createFile(io, tmp, .{});
        var fw = file.writer(io, &buf);
        digestsTo(gpa, io, &fw.interface, dir.?, &model, layers, jobs) catch |e| {
            file.close(io);
            std.debug.print("FAIL tf-inkling2-plan --digests: {t}\n", .{e});
            return 1;
        };
        file.close(io);
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), path, io);
        std.debug.print("{s}: world {d}\n", .{ path, model.p.split.world });
        return 0;
    }
    var out = std.Io.File.stdout().writer(io, &buf);
    if (ranges) |r| {
        if (r >= model.p.split.world) {
            std.debug.print("--ranges {d}: the world size is {d}\n", .{ r, model.p.split.world });
            return 2;
        }
        try ink.describe.ranges(&out.interface, &model, r);
    } else try ink.describe.report(&out.interface, &model, positions, streams);
    try out.interface.flush();
    return 0;
}

fn digestsTo(gpa: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, dir: []const u8, model: *const ink.describe.Model, layers: ?[2]u32, jobs: usize) !void {
    try ink.digests.write(gpa, io, w, dir, model, layers, jobs);
    try w.flush();
}

fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dir: ?[]const u8, inventory_path: ?[]const u8, want: ink.describe.Want, d: *ink.Diag) !ink.describe.Model {
    const inv_text: ?[]const u8 = if (inventory_path) |p| try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 26)) else null;
    if (dir) |root| {
        if (inv_text == null) return ink.describe.Model.fromDir(gpa, io, root, want, d);
        const cfg = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "config.json" }), a, .limited(1 << 22));
        const q = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ root, "hf_quant_config.json" }), a, .limited(1 << 24));
        return ink.describe.Model.fromTexts(gpa, cfg, q, inv_text.?, want, d);
    }
    return ink.describe.Model.fromTexts(gpa, ink.config.fixture, ink.quant.fixture, inv_text orelse ink.inventory.fixture, want, d);
}
