//! needle: does having read a passage make it cheaper to read again?
//!
//! Held-out bits per byte is an aggregate over natural text, and that
//! aggregate is dominated by local structure. Two models can sit a hundredth
//! of a bit apart and differ completely in whether one specific distant
//! passage is still reachable. That is what this measures and BPB cannot.
//!
//! An earlier version of this tool inserted a synthetic key - `The access
//! code is QZ7412.` - and scored the code after a matching cue. It measured
//! nothing, and the positive control at 32 bytes distance came back inside
//! its error bar along with everything else. The reason was in the absolute
//! numbers: 12-14 bits per answer byte, where a uniform guess over all 256
//! values is 8. A random code is so far outside a Wikipedia byte model's
//! distribution that it gets probability 2^-13 whether or not it was read.
//! The measurement sat in a tail where no signal survives.
//!
//! So the needle is now *real text*, which the model can express, and the
//! test is induction rather than question answering:
//!
//!     ... context ...  SPAN(48 bytes)  ... dist bytes ...  SPAN[0..16]|SPAN[16..48]
//!
//! Only the bytes after the bar are scored. The control is the same document
//! with the early occurrence of SPAN replaced by different text, so the model
//! meets the continuation cold. The difference is what having read it was
//! worth, in bits per byte.
//!
//! Two cue lengths: 16 bytes, which is enough to identify the passage
//! uniquely, and 6, which is not - the gap between them says whether any
//! retrieval needs a long literal match or tolerates a partial one.
//!
//! usage: needle CHECKPOINT DATA [dist=2048] [trials=48] [seed=1] [key=value ...]
const std = @import("std");
const config = @import("model/config.zig");
const checkpoint = @import("model/checkpoint.zig");
const session = @import("model/session.zig");
const gpu = @import("model/gpu.zig");
const stdrand = @import("stdrand.zig");
const cli = @import("cli.zig");

const kernels_ptx = @embedFile("kernels.ptx");
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

const SPAN = 48; // bytes of the passage that is planted
const ANSWER = 32; // the tail of it that gets scored

/// Mean loss in bits over the answer bytes of one document.
fn score(s: *session.Session, doc: []const u8, answer_at: usize, gpa: std.mem.Allocator) !f64 {
    const T: usize = @intCast(s.m.c.seqlen);
    try s.resetState();
    const ids = try gpa.alloc(i32, T);
    const nxt = try gpa.alloc(i32, T);
    const ends = try gpa.alloc(i32, T);
    const per = try gpa.alloc(f32, T);
    defer inline for (.{ ids, nxt, ends }) |x| gpa.free(x);
    defer gpa.free(per);
    @memset(ends, 0);

    var total: f64 = 0;
    var n: usize = 0;
    var at: usize = 0;
    while (at + T < doc.len) : (at += T) {
        for (0..T) |t| {
            ids[t] = doc[at + t];
            // Only the answer bytes carry a target; everywhere else the loss
            // kernel is told to skip, so the window average cannot dilute it.
            const pos = at + t;
            nxt[t] = if (pos >= answer_at and pos < answer_at + ANSWER)
                @as(i32, doc[pos + 1])
            else
                -1;
        }
        _ = try s.forward(ids, nxt, ends);
        try s.losses(per);
        for (0..T) |t| {
            const pos = at + t;
            if (pos >= answer_at and pos < answer_at + ANSWER) {
                total += per[t];
                n += 1;
            }
        }
    }
    return if (n == 0) 0 else total / @as(f64, @floatFromInt(n)) / 0.6931471805599453;
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const gpa = init.arena.allocator();
    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &stdout.interface;
    const argv = try init.minimal.args.toSlice(gpa);
    if (argv.len < 3) return cli.fail(io, "usage: needle CHECKPOINT DATA [dist=2048] [trials=24]\n", .{});
    const args = try cli.Args.parse(gpa, init.minimal.args);
    const dist: usize = try args.int(usize, "dist", 2048);
    const trials: usize = try args.int(usize, "trials", 24);
    const seed: u32 = try args.int(u32, "seed", 1);

    const file_cfg = checkpoint.readConfig(gpa, io, argv[1]) catch
        return cli.fail(io, "{s}\n", .{checkpoint.lastError()});
    var cfg = file_cfg;
    cfg.batch = 1;
    cfg.seqlen = 256;
    cfg.maxcarry = 0; // carry the whole document
    try config.validate(cfg);

    const data = try std.Io.Dir.cwd().readFileAlloc(io, argv[2], gpa, .unlimited);
    var sess = session.Session.init(gpa, io, cfg, kernels_ptx) catch
        return cli.fail(io, "{s}\n", .{gpu.lastError()});
    defer sess.deinit();
    checkpoint.loadWeights(gpa, io, argv[1], &sess.m, file_cfg) catch
        return cli.fail(io, "{s}\n", .{checkpoint.lastError()});

    var rng = stdrand.Mt19937.init(seed);
    const cue_lens = [_]usize{ 16, 6 };
    const names = [_][:0]const u8{ "16-byte cue  ", "6-byte cue   " };

    var b: [512]u8 = undefined;
    const p = struct {
        fn f(out: *std.Io.Writer, bb: []u8, comptime fmt: [:0]const u8, a2: anytype) !void {
            const c = @call(.auto, snprintf, .{ bb.ptr, bb.len, fmt.ptr } ++ a2);
            try out.writeAll(bb[0..@intCast(@min(c, bb.len - 1))]);
            try out.flush();
        }
    }.f;
    try p(w, &b, "needle: %s, distance %zu bytes, %zu trials\n", .{ argv[1].ptr, dist, trials });
    try p(w, &b, "loss on %d bytes of real text, in bits. seen: the passage appeared\n", .{@as(c_int, ANSWER)});
    try p(w, &b, "earlier; cold: it did not. benefit is what reading it was worth.\n\n", .{});
    try p(w, &b, "%-14s %10s %10s %16s\n", .{ "cue", "seen", "cold", "benefit" });

    for (cue_lens, names) |cue, name| {
        var seen_sum: f64 = 0;
        var cold_sum: f64 = 0;
        var dsum: f64 = 0;
        var dsq: f64 = 0;
        for (0..trials) |_| {
            const span_at = 4096 + @as(usize, @intFromFloat(rng.canonical() *
                @as(f64, @floatFromInt(data.len - dist - 12288))));
            const span = data[span_at .. span_at + SPAN];
            // A different passage of the same length, for the control. Taken
            // from elsewhere so the two documents differ only in whether the
            // scored text was present.
            const other_at = @as(usize, @intFromFloat(rng.canonical() *
                @as(f64, @floatFromInt(data.len - SPAN - 1))));
            const other = data[other_at .. other_at + SPAN];

            var docA: std.ArrayList(u8) = .empty;
            var docB: std.ArrayList(u8) = .empty;
            defer docA.deinit(gpa);
            defer docB.deinit(gpa);
            const run_up = span_at - 1024;
            try docA.appendSlice(gpa, data[run_up .. run_up + 1024]);
            try docB.appendSlice(gpa, data[run_up .. run_up + 1024]);
            try docA.appendSlice(gpa, span);   // planted
            try docB.appendSlice(gpa, other);  // not planted
            const filler = span_at + SPAN;
            try docA.appendSlice(gpa, data[filler .. filler + dist]);
            try docB.appendSlice(gpa, data[filler .. filler + dist]);
            // The cue, then the bytes that are scored.
            try docA.appendSlice(gpa, span[0..cue]);
            try docB.appendSlice(gpa, span[0..cue]);
            const answer_at = docA.items.len - 1;
            try docA.appendSlice(gpa, span[cue .. cue + ANSWER]);
            try docB.appendSlice(gpa, span[cue .. cue + ANSWER]);
            const tail = filler + dist;
            try docA.appendSlice(gpa, data[tail .. tail + 1024]);
            try docB.appendSlice(gpa, data[tail .. tail + 1024]);

            const a_one = try score(&sess, docA.items, answer_at, gpa);
            const b_one = try score(&sess, docB.items, answer_at, gpa);
            seen_sum += a_one;
            cold_sum += b_one;
            dsum += b_one - a_one;
            dsq += (b_one - a_one) * (b_one - a_one);
        }
        const ft: f64 = @floatFromInt(trials);
        const md = dsum / ft;
        const sd = @sqrt(@max(dsq / ft - md * md, 0) * ft / @max(ft - 1, 1));
        try p(w, &b, "%-14s %10.3f %10.3f %+10.3f +-%.3f\n",
            .{ name.ptr, seen_sum / ft, cold_sum / ft, md, sd / @sqrt(ft) });
    }
    return 0;
}
