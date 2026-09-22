// chat: interactive CLI with a persistent recurrent state.
//   chat CHECKPOINT [temp=0.7] [maxlen=512] [seed=1] [mode=auto|dialog|raw]
//
// The state is never truncated: everything said so far stays in the recurrent
// memory until /reset (no context window).
//
// dialog  (checkpoints trained with dialog=1, see tools/dialogprep.zig):
//         each input is fed as 0x02 text 0x04 0x03 and the model answers until it
//         ends its turn with 0x04 (or maxlen). A conversation starts with 0x1E.
// raw     (plain text checkpoints): the input is fed as text and the model
//         continues it until a newline (or maxlen).
// Commands: /reset  /temp X  /maxlen N  /mode dialog|raw  /help  /quit
const std = @import("std");
const model = @import("model.zig");
const gen = @import("generate.zig");
const data = @import("data.zig");
const u = @import("util.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        u.print("usage: chat CHECKPOINT [temp=0.7] [maxlen=512] [seed=1] [mode=auto|dialog|raw]\n", .{});
        std.process.exit(1);
    }
    model.initCompute();
    var temp: f32 = 0.7;
    var maxlen: i64 = 512;
    var seed: u64 = 1;
    var mode: []const u8 = "auto";
    for (args[2..]) |a| {
        const kv = u.splitKV(a) orelse u.die("expected key=value", .{});
        if (std.mem.eql(u8, kv.key, "temp")) temp = u.parseF32(kv.value) else if (std.mem.eql(u8, kv.key, "maxlen")) maxlen = u.parseInt(i64, kv.value) else if (std.mem.eql(u8, kv.key, "seed")) seed = u.parseCount(kv.value) else if (std.mem.eql(u8, kv.key, "mode")) mode = kv.value else u.die("unknown option: {s}", .{kv.key});
    }
    var g = gen.Generator.init(args[1], seed);
    if (std.mem.eql(u8, mode, "auto")) mode = if (g.file_cfg.dialog == 1) "dialog" else "raw";
    var dialog = std.mem.eql(u8, mode, "dialog");
    if (!dialog and !std.mem.eql(u8, mode, "raw")) u.die("mode must be auto, dialog or raw", .{});
    const tty = u.c.isatty(0) == 1;
    u.eprint("tmt chat ({s} mode, temp {d:.2}). /help for commands.\n", .{ mode, temp });
    const begin = struct {
        fn f(gg: *gen.Generator, d: bool) void {
            gg.reset();
            if (d) gg.feed(data.CONV);
        }
    }.f;
    begin(&g, dialog);
    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(u.gpa);
    while (true) {
        if (tty) u.print("you> ", .{});
        line.clearRetainingCapacity();
        while (true) {
            const ch = u.c.fgetc(u.c.stdin());
            if (ch == u.c.EOF) {
                if (line.items.len == 0) return;
                break;
            }
            if (ch == '\n') break;
            try line.append(u.gpa, @intCast(ch));
        }
        const text = line.items;
        if (text.len > 0 and text[0] == '/') {
            const sp = std.mem.indexOfScalar(u8, text, ' ');
            const cmd = if (sp) |i| text[0..i] else text;
            const arg = if (sp) |i| text[i + 1 ..] else "";
            if (std.mem.eql(u8, cmd, "/quit") or std.mem.eql(u8, cmd, "/exit")) break;
            if (std.mem.eql(u8, cmd, "/reset")) {
                begin(&g, dialog);
                u.print("(state reset)\n", .{});
            } else if (std.mem.eql(u8, cmd, "/temp") and arg.len > 0) {
                temp = u.parseF32(arg);
                u.print("(temp {d:.2})\n", .{temp});
            } else if (std.mem.eql(u8, cmd, "/maxlen") and arg.len > 0) {
                maxlen = u.parseInt(i64, arg);
                u.print("(maxlen {d})\n", .{maxlen});
            } else if (std.mem.eql(u8, cmd, "/mode") and (std.mem.eql(u8, arg, "dialog") or std.mem.eql(u8, arg, "raw"))) {
                dialog = std.mem.eql(u8, arg, "dialog");
                begin(&g, dialog);
                u.print("({s} mode, state reset)\n", .{arg});
            } else u.print("commands: /reset /temp X /maxlen N /mode dialog|raw /quit  ({d} bytes in state)\n", .{g.fed});
            continue;
        }
        if (text.len == 0) continue;
        if (dialog) {
            g.feed(data.USER);
            g.feedText(text);
            g.feed(data.END);
            g.feed(data.ASSISTANT);
        } else g.feedText(text);
        u.print("tmt> ", .{});
        if (!dialog) u.print("{s}", .{text});
        var ended = false;
        var i: i64 = 0;
        while (i < maxlen) : (i += 1) {
            const b = g.sample(temp);
            if (dialog and data.isMarker(b)) { // model ended its turn
                g.feed(data.END);
                ended = true;
                break;
            }
            if (!dialog and b == '\n') {
                g.feed(b);
                ended = true;
                break;
            }
            u.print("{c}", .{b});
            g.feed(b);
        }
        if (!ended) g.feed(if (dialog) data.END else '\n'); // close the turn at maxlen
        u.print("\n", .{});
    }
}
