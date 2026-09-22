//! chat: interactive CLI with a persistent recurrent state (port of src/chat.cu).
//!   chat CHECKPOINT [temp=0.7] [maxlen=512] [seed=1] [mode=auto|dialog|raw]
//!
//! The state is never truncated: everything said stays in the recurrent memory
//! until /reset (no context window).
//! dialog: each input is fed as 0x02 text 0x04 0x03; the model answers until it
//!         ends its turn with 0x04 (or maxlen). A conversation starts with 0x1E.
//! raw:    the input is fed as text and the model continues it until a newline.
//! Commands: /reset  /temp X  /maxlen N  /mode dialog|raw  /help  /quit
const std = @import("std");
const cli = @import("cli.zig");
const generate = @import("model/generate.zig");
const gpu = @import("model/gpu.zig");

/// The compiled kernels, loaded with the model.
const kernels_ptx = @embedFile("kernels.ptx");

const CONV = 0x1E;
const USER = 0x02;
const ASSISTANT = 0x03;
const END = 0x04;

fn isMarker(b: u8) bool {
    return b == CONV or b == USER or b == ASSISTANT or b == END;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try cli.Args.parse(init.arena.allocator(), init.minimal.args);
    const pos = args.positional.items;
    if (pos.len != 1)
        return cli.fail(io, "usage: chat CHECKPOINT [temp=0.7] [maxlen=512] [seed=1] [mode=auto|dialog|raw]\n", .{});
    var temp: f32 = @floatCast(try args.float("temp", 0.7));
    var maxlen = try args.int(usize, "maxlen", 512);
    var mode = args.get("mode", "auto");
    const path = try init.arena.allocator().dupeZ(u8, pos[0]);
    var g = generate.Generator.open(init.arena.allocator(), io, path, try args.int(u32, "seed", 1), kernels_ptx) catch
        return cli.fail(io, "error: {s}\n", .{gpu.lastError()});
    defer g.close();
    if (std.mem.eql(u8, mode, "auto")) mode = if (g.file_cfg.dialog != 0) "dialog" else "raw";
    if (!std.mem.eql(u8, mode, "dialog") and !std.mem.eql(u8, mode, "raw"))
        return cli.fail(io, "error: mode must be auto, dialog or raw\n", .{});
    const tty = std.Io.File.stdin().isTty(io) catch false;

    var obuf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &obuf);
    const w = &out.interface;
    var ebuf: [256]u8 = undefined;
    var err = std.Io.File.stderr().writerStreaming(io, &ebuf);
    try err.interface.print("tmt chat ({s} mode, temp {d:.2}). /help for commands.\n", .{ mode, temp });
    try err.interface.flush();

    const Begin = struct {
        fn run(gen: *generate.Generator, m: []const u8) !void {
            try gen.reset();
            if (std.mem.eql(u8, m, "dialog")) try gen.feed(CONV);
        }
    };
    try Begin.run(&g, mode);
    var ibuf: [1 << 16]u8 = undefined;
    var in = std.Io.File.stdin().reader(io, &ibuf);
    var line_buf = std.Io.Writer.Allocating.init(init.gpa);
    defer line_buf.deinit();
    while (true) {
        if (tty) {
            try w.writeAll("you> ");
            try w.flush();
        }
        const line = try cli.readLine(&in.interface, &line_buf) orelse break;
        if (line.len > 0 and line[0] == '/') {
            const sp = std.mem.indexOfScalar(u8, line, ' ');
            const cmd = line[0 .. sp orelse line.len];
            const arg = if (sp) |s| line[s + 1 ..] else "";
            if (std.mem.eql(u8, cmd, "/quit") or std.mem.eql(u8, cmd, "/exit")) break;
            if (std.mem.eql(u8, cmd, "/reset")) {
                try Begin.run(&g, mode);
                try w.writeAll("(state reset)\n");
            } else if (std.mem.eql(u8, cmd, "/temp") and arg.len > 0) {
                temp = std.fmt.parseFloat(f32, arg) catch temp;
                try w.print("(temp {d:.2})\n", .{temp});
            } else if (std.mem.eql(u8, cmd, "/maxlen") and arg.len > 0) {
                maxlen = std.fmt.parseInt(usize, arg, 10) catch maxlen;
                try w.print("(maxlen {d})\n", .{maxlen});
            } else if (std.mem.eql(u8, cmd, "/mode") and (std.mem.eql(u8, arg, "dialog") or std.mem.eql(u8, arg, "raw"))) {
                mode = if (std.mem.eql(u8, arg, "dialog")) "dialog" else "raw";
                try Begin.run(&g, mode);
                try w.print("({s} mode, state reset)\n", .{mode});
            } else try w.print("commands: /reset /temp X /maxlen N /mode dialog|raw /quit  ({d} bytes in state)\n", .{g.fed});
            try w.flush();
            continue;
        }
        if (line.len == 0) continue;
        const dialog = std.mem.eql(u8, mode, "dialog");
        if (dialog) {
            try g.feed(USER);
            try g.feedText(line);
            try g.feed(END);
            try g.feed(ASSISTANT);
        } else try g.feedText(line);
        try w.writeAll("tmt> ");
        if (!dialog) try w.writeAll(line);
        try w.flush();
        var ended = false;
        for (0..maxlen) |_| {
            const b = g.sample(temp);
            if (dialog and isMarker(b)) { // the model ended its turn
                try g.feed(END);
                ended = true;
                break;
            }
            if (!dialog and b == '\n') {
                try g.feed(b);
                ended = true;
                break;
            }
            try w.writeByte(b);
            try w.flush();
            try g.feed(b);
        }
        if (!ended) try g.feed(if (dialog) END else '\n'); // close the turn at maxlen
        try w.writeByte('\n');
        try w.flush();
    }
}
