//! sample: CHECKPOINT "prompt..." [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]
//! Feeds the prompt byte by byte, then continues it (port of src/sample.cu).
const std = @import("std");
const cli = @import("cli.zig");
const tmt = @import("tmt.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try cli.Args.parse(init.arena.allocator(), init.minimal.args);
    const pos = args.positional.items;
    if (pos.len != 2 or pos[1].len == 0)
        return cli.fail(io, "usage: sample CHECKPOINT \"prompt\" [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]\n", .{});
    var it = args.options.keyIterator();
    while (it.next()) |k| {
        if (!std.mem.eql(u8, k.*, "temp") and !std.mem.eql(u8, k.*, "maxlen") and
            !std.mem.eql(u8, k.*, "stop") and !std.mem.eql(u8, k.*, "seed"))
            return cli.fail(io, "error: unknown option: {s}\n", .{k.*});
    }
    const temp: f32 = @floatCast(try args.float("temp", 0.7));
    const stop_thr: f32 = @floatCast(try args.float("stop", 0.5));
    const maxlen = try args.int(usize, "maxlen", 256);
    const path = try init.arena.allocator().dupeZ(u8, pos[0]);
    var g = tmt.Generator.open(path, try args.int(u32, "seed", 1)) catch
        return cli.fail(io, "error: {s}\n", .{tmt.lastError()});
    defer g.close();
    var buf: [256]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &out.interface;
    // Every byte is fed exactly once: the prediction after the last prompt
    // byte gives the first output byte.
    try g.feedText(pos[1]);
    for (0..maxlen) |i| {
        if (g.stopTrained() and g.stop_prob > stop_thr) break;
        const b = g.sample(temp);
        try w.writeByte(b);
        try w.flush();
        if (i + 1 < maxlen) try g.feed(b);
    }
    try w.writeByte('\n');
    try w.flush();
}
