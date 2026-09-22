// sample CHECKPOINT "prompt..." [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]
// Feeds the prompt byte by byte, then continues it (src/generate.zig).
const std = @import("std");
const model = @import("model.zig");
const gen = @import("generate.zig");
const u = @import("util.zig");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        u.print("usage: sample CHECKPOINT \"prompt\" [temp=0.7] [maxlen=256] [stop=0.5] [seed=1]\n", .{});
        std.process.exit(1);
    }
    model.initCompute();
    var temp: f32 = 0.7;
    var stop_thr: f32 = 0.5;
    var maxlen: i64 = 256;
    var seed: u64 = 1;
    for (args[3..]) |a| {
        const kv = u.splitKV(a) orelse u.die("expected key=value", .{});
        if (std.mem.eql(u8, kv.key, "temp")) temp = u.parseF32(kv.value) else if (std.mem.eql(u8, kv.key, "maxlen")) maxlen = u.parseInt(i64, kv.value) else if (std.mem.eql(u8, kv.key, "stop")) stop_thr = u.parseF32(kv.value) else if (std.mem.eql(u8, kv.key, "seed")) seed = u.parseCount(kv.value) else u.die("unknown option: {s}", .{kv.key});
    }
    const prompt = args[2];
    if (prompt.len == 0) u.die("empty prompt", .{});
    var g = gen.Generator.init(args[1], seed);
    // Every byte is fed exactly once: the prediction after the last prompt
    // byte gives the first output byte.
    g.feedText(prompt);
    var i: i64 = 0;
    while (i < maxlen) : (i += 1) {
        if (g.stopTrained() and g.stop_prob > stop_thr) break;
        const b = g.sample(temp);
        u.print("{c}", .{b});
        if (i + 1 < maxlen) g.feed(b);
    }
    u.print("\n", .{});
}
