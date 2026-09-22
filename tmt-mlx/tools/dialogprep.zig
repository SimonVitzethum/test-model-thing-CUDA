// dialogprep: build byte-level dialog training data for `train dialog=1`.
//
//   zcat oasst_ready.trees.jsonl.gz | dialogprep oasst OUT_PREFIX [lang=en] [test_frac=0.05]
//   dialogprep tsv PAIRS.tsv OUT_PREFIX [test_frac=0.05]     (user <tab> assistant per line)
//
// Output: OUT_PREFIX_train.bin and OUT_PREFIX_test.bin, a plain byte stream:
//   0x1E                      start of a conversation (use docsep=30 to reset state)
//   0x02 text 0x04            user turn
//   0x03 text 0x04            assistant turn
// With dialog=1 the loss covers only assistant text and its closing 0x04, so the
// model learns to answer and to end its turn. Control bytes inside messages are
// replaced by spaces (newlines and tabs are kept).
//
// oasst: OpenAssistant oasst1 message trees (Apache-2.0). Every root-to-leaf path
// whose messages are all in `lang` becomes one conversation. The split is by
// tree, so test conversations never share a prompt with training.
const std = @import("std");
const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
});

const gpa = std.heap.c_allocator;
const CONV = 0x1E;
const USER = 0x02;
const ASSISTANT = 0x03;
const END = 0x04;

fn die(comptime fmt: []const u8, args: anytype) noreturn {
    const s = std.fmt.allocPrint(gpa, "error: " ++ fmt ++ "\n", args) catch "error\n";
    _ = c.fwrite(s.ptr, 1, s.len, c.stderr());
    std.process.exit(1);
}
fn note(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.allocPrint(gpa, fmt, args) catch return;
    defer gpa.free(s);
    _ = c.fwrite(s.ptr, 1, s.len, c.stderr());
}

fn fnv(s: []const u8) u64 {
    var h: u64 = 14695981039346656037;
    for (s) |b| h = (h ^ b) *% 1099511628211;
    return h;
}
fn inTest(key: []const u8, frac: f64) bool {
    return @as(f64, @floatFromInt(fnv(key) >> 11)) / @as(f64, @floatFromInt(@as(u64, 1) << 53)) < frac;
}

/// Control bytes become spaces (newlines and tabs stay), then trim.
fn sanitize(alloc: std.mem.Allocator, s: []const u8) []u8 {
    const out = alloc.alloc(u8, s.len) catch unreachable;
    for (s, out) |x, *o| o.* = if (x < 0x20 and x != '\n' and x != '\t') ' ' else x;
    return @constCast(std.mem.trim(u8, out, " \n\t"));
}

// ---- minimal JSON navigation over one line, no DOM ----
const Json = struct {
    s: []const u8,

    fn ws(self: Json, p: usize) usize {
        var i = p;
        while (i < self.s.len and (self.s[i] == ' ' or self.s[i] == '\t' or self.s[i] == '\r' or self.s[i] == '\n')) i += 1;
        return i;
    }
    fn skipString(self: Json, p: usize) !usize { // p at opening quote
        var i = p + 1;
        while (i < self.s.len) : (i += 1) {
            if (self.s[i] == '\\') i += 1 else if (self.s[i] == '"') return i + 1;
        }
        return error.Unterminated;
    }
    fn skip(self: Json, p0: usize) !usize {
        var p = self.ws(p0);
        if (p >= self.s.len) return error.UnexpectedEnd;
        const ch = self.s[p];
        if (ch == '"') return self.skipString(p);
        if (ch == '{' or ch == '[') {
            var depth: i32 = 0;
            while (p < self.s.len) : (p += 1) {
                if (self.s[p] == '"') {
                    p = (try self.skipString(p)) - 1;
                    continue;
                }
                if (self.s[p] == '{' or self.s[p] == '[') depth += 1 else if (self.s[p] == '}' or self.s[p] == ']') {
                    depth -= 1;
                    if (depth == 0) return p + 1;
                }
            }
            return error.Unterminated;
        }
        while (p < self.s.len and self.s[p] != ',' and self.s[p] != '}' and self.s[p] != ']') p += 1;
        return p;
    }
    fn rawString(self: Json, p: usize) ![]const u8 { // content between quotes (escapes kept)
        const e = try self.skipString(p);
        return self.s[p + 1 .. e - 1];
    }
    /// Value position of `key` in the object starting at p, or null.
    fn find(self: Json, p0: ?usize, key: []const u8) !?usize {
        var p = self.ws(p0 orelse return null);
        if (p >= self.s.len or self.s[p] != '{') return null;
        p = self.ws(p + 1);
        while (p < self.s.len and self.s[p] != '}') {
            const k = try self.rawString(p);
            p = self.ws(try self.skipString(p));
            if (self.s[p] != ':') return error.ExpectedColon;
            const v = self.ws(p + 1);
            if (std.mem.eql(u8, k, key)) return v;
            p = self.ws(try self.skip(v));
            if (self.s[p] == ',') p = self.ws(p + 1);
        }
        return null;
    }
    /// Positions of array elements (or object values).
    fn each(self: Json, p0: ?usize, out: *std.ArrayList(usize)) !void {
        var p = self.ws(p0 orelse return);
        if (p >= self.s.len) return;
        const open = self.s[p];
        if (open != '[' and open != '{') return;
        const close: u8 = if (open == '[') ']' else '}';
        p = self.ws(p + 1);
        while (p < self.s.len and self.s[p] != close) {
            if (open == '{') {
                p = self.ws(try self.skipString(p));
                p = self.ws(p + 1);
            }
            try out.append(gpa, p);
            p = self.ws(try self.skip(p));
            if (self.s[p] == ',') p = self.ws(p + 1);
        }
    }
};

fn appendUtf8(out: *std.ArrayList(u8), cp: u32) void {
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(@truncate(cp), &buf) catch {
        out.append(gpa, '?') catch unreachable;
        return;
    };
    out.appendSlice(gpa, buf[0..n]) catch unreachable;
}
fn unescape(alloc: std.mem.Allocator, r: []const u8) []u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < r.len) : (i += 1) {
        if (r[i] != '\\') {
            out.append(alloc, r[i]) catch unreachable;
            continue;
        }
        i += 1;
        if (i >= r.len) break;
        switch (r[i]) {
            'n' => out.append(alloc, '\n') catch unreachable,
            't', 'r', 'b', 'f' => out.append(alloc, ' ') catch unreachable,
            'u' => {
                if (i + 4 >= r.len) break;
                var cp: u32 = std.fmt.parseInt(u32, r[i + 1 ..][0..4], 16) catch 0xFFFD;
                i += 4;
                if (cp >= 0xD800 and cp < 0xDC00 and i + 6 < r.len and r[i + 1] == '\\' and r[i + 2] == 'u') {
                    const lo = std.fmt.parseInt(u32, r[i + 3 ..][0..4], 16) catch 0xDC00;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
                appendUtf8(&out, cp);
            },
            else => out.append(alloc, r[i]) catch unreachable,
        }
    }
    return out.toOwnedSlice(alloc) catch unreachable;
}

const Turn = struct { assistant: bool, text: []const u8 };

const Writer = struct {
    out: [2]*c.FILE,
    conversations: [2]usize = .{ 0, 0 },
    bytes: [2]usize = .{ 0, 0 },

    fn init(prefix: []const u8) Writer {
        var w: Writer = .{ .out = undefined };
        for (0..2) |k| {
            var buf: [4096]u8 = undefined;
            const name = std.fmt.bufPrintZ(&buf, "{s}_{s}.bin", .{ prefix, if (k == 0) "train" else "test" }) catch die("path too long", .{});
            w.out[k] = c.fopen(name.ptr, "wb") orelse die("cannot write {s}", .{name});
        }
        return w;
    }
    fn conversation(self: *Writer, split: usize, turns: []const Turn) void {
        var s: std.ArrayList(u8) = .empty;
        defer s.deinit(gpa);
        s.append(gpa, CONV) catch unreachable;
        for (turns) |t| {
            s.append(gpa, if (t.assistant) ASSISTANT else USER) catch unreachable;
            s.appendSlice(gpa, t.text) catch unreachable;
            s.append(gpa, END) catch unreachable;
        }
        _ = c.fwrite(s.items.ptr, 1, s.items.len, self.out[split]);
        self.conversations[split] += 1;
        self.bytes[split] += s.items.len;
    }
    fn report(self: *Writer, prefix: []const u8) void {
        for (0..2) |k| {
            _ = c.fclose(self.out[k]);
            note("wrote {s}_{s}.bin: {d} conversations, {d} bytes\n", .{ prefix, if (k == 0) "train" else "test", self.conversations[k], self.bytes[k] });
        }
    }
};

/// Depth-first over an oasst message tree; every leaf closes one conversation.
fn walk(j: Json, msg: usize, lang: []const u8, path: *std.ArrayList(Turn), w: *Writer, split: usize) !void {
    const l = (try j.find(msg, "lang")) orelse return;
    const role = (try j.find(msg, "role")) orelse return;
    const text = (try j.find(msg, "text")) orelse return;
    if (!std.mem.eql(u8, try j.rawString(l), lang)) return;
    const raw = unescape(gpa, try j.rawString(text));
    const t = sanitize(gpa, raw);
    if (t.len == 0) return;
    const assistant = std.mem.eql(u8, try j.rawString(role), "assistant");
    if (path.items.len == 0 and assistant) return; // conversations start with the user
    try path.append(gpa, .{ .assistant = assistant, .text = t });
    defer _ = path.pop();
    var replies: std.ArrayList(usize) = .empty;
    defer replies.deinit(gpa);
    try j.each(try j.find(msg, "replies"), &replies);
    for (replies.items) |r| try walk(j, r, lang, path, w, split);
    if (replies.items.len == 0 and assistant) w.conversation(split, path.items); // end on an assistant turn
}

fn readLine(buf: *std.ArrayList(u8), file: *c.FILE) bool {
    buf.clearRetainingCapacity();
    while (true) {
        const ch = c.fgetc(file);
        if (ch == c.EOF) return buf.items.len > 0;
        if (ch == '\n') return true;
        buf.append(gpa, @intCast(ch)) catch unreachable;
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var pos: std.ArrayList([]const u8) = .empty;
    var lang: []const u8 = "en";
    var frac: f64 = 0.05;
    for (args[1..]) |a| {
        if (std.mem.indexOfScalar(u8, a, '=')) |eq| {
            const key = a[0..eq];
            const value = a[eq + 1 ..];
            if (std.mem.eql(u8, key, "lang")) lang = value else if (std.mem.eql(u8, key, "test_frac"))
                frac = std.fmt.parseFloat(f64, value) catch die("invalid test_frac", .{})
            else die("unknown option: {s}", .{key});
        } else try pos.append(gpa, a);
    }
    if (pos.items.len == 2 and std.mem.eql(u8, pos.items[0], "oasst")) {
        var w = Writer.init(pos.items[1]);
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(gpa);
        var trees: usize = 0;
        var errors: usize = 0;
        while (readLine(&line, c.stdin())) {
            if (line.items.len == 0) continue;
            const j = Json{ .s = line.items };
            const walked = blk: {
                const id = (j.find(0, "message_tree_id") catch break :blk false) orelse break :blk true;
                const prompt = (j.find(0, "prompt") catch break :blk false) orelse break :blk true;
                var path: std.ArrayList(Turn) = .empty;
                defer path.deinit(gpa);
                const key = j.rawString(id) catch break :blk false;
                walk(j, prompt, lang, &path, &w, @intFromBool(inTest(key, frac))) catch break :blk false;
                break :blk true;
            };
            if (walked) trees += 1 else errors += 1;
        }
        note("{d} trees, {d} parse errors\n", .{ trees, errors });
        w.report(pos.items[1]);
        return;
    }
    if (pos.items.len == 3 and std.mem.eql(u8, pos.items[0], "tsv")) {
        var buf: [4096]u8 = undefined;
        const name = std.fmt.bufPrintZ(&buf, "{s}", .{pos.items[1]}) catch die("path too long", .{});
        const in = c.fopen(name.ptr, "rb") orelse die("cannot read {s}", .{pos.items[1]});
        var w = Writer.init(pos.items[2]);
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(gpa);
        while (readLine(&line, in)) {
            const tab = std.mem.indexOfScalar(u8, line.items, '\t') orelse continue;
            const user = sanitize(gpa, line.items[0..tab]);
            const assistant = sanitize(gpa, line.items[tab + 1 ..]);
            if (user.len > 0 and assistant.len > 0)
                w.conversation(@intFromBool(inTest(line.items, frac)), &.{ .{ .assistant = false, .text = user }, .{ .assistant = true, .text = assistant } });
        }
        _ = c.fclose(in);
        w.report(pos.items[2]);
        return;
    }
    note("usage: dialogprep oasst OUT_PREFIX [lang=en] [test_frac=0.05]   < trees.jsonl\n" ++
        "       dialogprep tsv PAIRS.tsv OUT_PREFIX [test_frac=0.05]\n", .{});
    std.process.exit(1);
}
