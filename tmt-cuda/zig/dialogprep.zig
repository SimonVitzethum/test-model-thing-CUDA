//! dialogprep: byte-level dialog training data for `train dialog=1`
//! (port of tools/dialogprep.cpp; output is byte-identical).
//!
//!   zcat oasst_ready.trees.jsonl.gz | dialogprep oasst OUT_PREFIX [lang=en] [test_frac=0.05]
//!   dialogprep tsv PAIRS.tsv OUT_PREFIX [test_frac=0.05]     (user <tab> assistant per line)
//!
//! Output OUT_PREFIX_train.bin / OUT_PREFIX_test.bin, a plain byte stream:
//!   0x1E            start of a conversation (docsep=30 resets the state)
//!   0x02 text 0x04  user turn
//!   0x03 text 0x04  assistant turn
//! Control bytes inside messages become spaces (newlines and tabs are kept).
//! oasst: every root-to-leaf path whose messages are all in `lang` is one
//! conversation; the split is by tree.
const std = @import("std");
const json = @import("json.zig");
const cli = @import("cli.zig");

const CONV = 0x1E;
const USER = 0x02;
const ASSISTANT = 0x03;
const END = 0x04;

const Turn = struct { assistant: bool, text: []const u8 };

fn sanitize(gpa: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try gpa.alloc(u8, s.len);
    for (s, 0..) |c, i| out[i] = if (c < 0x20 and c != '\n' and c != '\t') ' ' else c;
    const trimmed = std.mem.trim(u8, out, " \n\t");
    const res = try gpa.dupe(u8, trimmed);
    gpa.free(out);
    return res;
}

const Writer = struct {
    files: [2]std.Io.File,
    bufs: [2][1 << 16]u8 = undefined,
    writers: [2]std.Io.File.Writer = undefined,
    conversations: [2]usize = .{ 0, 0 },
    bytes: [2]usize = .{ 0, 0 },

    fn open(self: *Writer, io: std.Io, gpa: std.mem.Allocator, prefix: []const u8) !void {
        const names = [2][]const u8{ "_train.bin", "_test.bin" };
        for (0..2) |k| {
            const path = try std.mem.concat(gpa, u8, &.{ prefix, names[k] });
            defer gpa.free(path);
            self.files[k] = try std.Io.Dir.cwd().createFile(io, path, .{});
            self.writers[k] = self.files[k].writer(io, &self.bufs[k]);
        }
    }
    fn conversation(self: *Writer, split: usize, turns: []const Turn) !void {
        const w = &self.writers[split].interface;
        try w.writeByte(CONV);
        var n: usize = 1;
        for (turns) |t| {
            try w.writeByte(if (t.assistant) ASSISTANT else USER);
            try w.writeAll(t.text);
            try w.writeByte(END);
            n += t.text.len + 2;
        }
        self.conversations[split] += 1;
        self.bytes[split] += n;
    }
    fn finish(self: *Writer, io: std.Io, prefix: []const u8) !void {
        const names = [2][]const u8{ "train", "test" };
        var ebuf: [512]u8 = undefined;
        var e = std.Io.File.stderr().writer(io, &ebuf);
        for (0..2) |k| {
            try self.writers[k].interface.flush();
            self.files[k].close(io);
            try e.interface.print("wrote {s}_{s}.bin: {d} conversations, {d} bytes\n", .{ prefix, names[k], self.conversations[k], self.bytes[k] });
        }
        try e.interface.flush();
    }
};

/// Depth-first over an oasst message tree; every leaf closes one conversation.
fn walk(gpa: std.mem.Allocator, j: json.Json, msg: usize, lang: []const u8, path: *std.ArrayList(Turn), w: *Writer, split: usize) !void {
    const l = try j.find(msg, "lang") orelse return;
    const role = try j.find(msg, "role") orelse return;
    const text = try j.find(msg, "text") orelse return;
    if (!std.mem.eql(u8, try j.rawString(l), lang)) return;
    const raw = try json.unescape(gpa, try j.rawString(text), true);
    defer gpa.free(raw);
    const t = try sanitize(gpa, raw);
    defer gpa.free(t);
    if (t.len == 0) return;
    const assistant = std.mem.eql(u8, try j.rawString(role), "assistant");
    if (path.items.len == 0 and assistant) return; // conversations start with the user
    try path.append(gpa, .{ .assistant = assistant, .text = t });
    defer _ = path.pop();
    var leaf = true;
    var it = try j.items(try j.find(msg, "replies"));
    while (try it.next()) |reply| {
        leaf = false;
        try walk(gpa, j, reply, lang, path, w, split);
    }
    if (leaf and assistant) try w.conversation(split, path.items); // end on an assistant turn
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const args = try cli.Args.parse(init.arena.allocator(), init.minimal.args);
    const frac = try args.float("test_frac", 0.05);
    const pos = args.positional.items;
    if (pos.len == 2 and std.mem.eql(u8, pos[0], "oasst")) {
        var w = Writer{ .files = undefined };
        try w.open(io, gpa, pos[1]);
        const lang = args.get("lang", "en");
        var in_buf: [1 << 16]u8 = undefined;
        var in = std.Io.File.stdin().reader(io, &in_buf);
        var line_buf = std.Io.Writer.Allocating.init(gpa);
        defer line_buf.deinit();
        var trees: usize = 0;
        var errors: usize = 0;
        while (try cli.readLine(&in.interface, &line_buf)) |line| {
            if (line.len == 0) continue;
            const j = json.Json{ .s = line };
            const ok = blk: {
                const id = (j.find(0, "message_tree_id") catch break :blk false) orelse break :blk true;
                const prompt = (j.find(0, "prompt") catch break :blk false) orelse break :blk true;
                var path: std.ArrayList(Turn) = .empty;
                defer path.deinit(gpa);
                const key = j.rawString(id) catch break :blk false;
                walk(gpa, j, prompt, lang, &path, &w, @intFromBool(json.inTest(key, frac))) catch |e| switch (e) {
                    error.Malformed => break :blk false,
                    else => return e,
                };
                trees += 1;
                break :blk true;
            };
            if (!ok) errors += 1;
        }
        var ebuf: [256]u8 = undefined;
        var e = std.Io.File.stderr().writer(io, &ebuf);
        try e.interface.print("{d} trees, {d} parse errors\n", .{ trees, errors });
        try e.interface.flush();
        try w.finish(io, pos[1]);
        return;
    }
    if (pos.len == 3 and std.mem.eql(u8, pos[0], "tsv")) {
        const data = try std.Io.Dir.cwd().readFileAlloc(io, pos[1], gpa, .unlimited);
        defer gpa.free(data);
        var w = Writer{ .files = undefined };
        try w.open(io, gpa, pos[2]);
        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |line| {
            if (lines.peek() == null and line.len == 0) break;
            const tab = std.mem.indexOfScalar(u8, line, '\t') orelse continue;
            const u = try sanitize(gpa, line[0..tab]);
            defer gpa.free(u);
            const a = try sanitize(gpa, line[tab + 1 ..]);
            defer gpa.free(a);
            if (u.len != 0 and a.len != 0)
                try w.conversation(@intFromBool(json.inTest(line, frac)), &.{ .{ .assistant = false, .text = u }, .{ .assistant = true, .text = a } });
        }
        try w.finish(io, pos[2]);
        return;
    }
    return cli.fail(io, "usage: dialogprep oasst OUT_PREFIX [lang=en] [test_frac=0.05]   < trees.jsonl\n" ++
        "       dialogprep tsv PAIRS.tsv OUT_PREFIX [test_frac=0.05]\n", .{});
}
