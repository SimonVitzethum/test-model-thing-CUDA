//! memprobe: CHECKPOINT DATA [samples=64] [seed=1] [key=value ...]
//! Phase 0 of the two-memory idea (MOONSHOTS.md section C).
//!
//! The question: does a one-shot, gradient-free store of recently seen bytes
//! say anything about the *continuation* that the weights do not already
//! know? Consolidation is only worth building if the answer is yes, because
//! a store that only helps on the bytes it recorded is a lookup table, and
//! distilling from it copies the corpus into the weights.
//!
//! So the store is filled with the bytes *preceding* a window and the loss
//! is measured on the window itself, which the store never saw. Three
//! conditions, and the third is what makes the first two mean anything:
//!
//!   none   memory padded out, the model on its own
//!   prev   memory holds the bytes immediately before the window
//!   rand   memory holds bytes from an unrelated place in the corpus
//!
//! `prev` beating `none` is not yet evidence: a memory that is merely
//! non-empty may help or hurt whatever is in it. The signal is `prev`
//! beating `rand`, which is the part that depends on the content being the
//! actual context.
//!
//! Nothing is trained or written. `forwardMem` resets the recurrent state,
//! so the store competes with no carry at all - the test is generous by
//! construction, which is what makes a null result decisive.
const std = @import("std");
const config = @import("model/config.zig");
const checkpoint = @import("model/checkpoint.zig");
const session = @import("model/session.zig");
const gpu = @import("model/gpu.zig");
const stdrand = @import("stdrand.zig");

const kernels_ptx = @embedFile("kernels.ptx");
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

const Fatal = error{Fatal};

const Out = struct {
    w: *std.Io.Writer,
    fn print(o: Out, comptime fmt: [:0]const u8, args: anytype) !void {
        var buf: [1024]u8 = undefined;
        const n = @call(.auto, snprintf, .{ &buf, buf.len, fmt.ptr } ++ args);
        try o.w.writeAll(buf[0..@intCast(@min(n, buf.len - 1))]);
        if (fmt[fmt.len - 1] == '\n') try o.w.flush();
    }
};

var err_msg: [512]u8 = undefined;
var err_len: usize = 0;
fn fail(comptime fmt: []const u8, args: anytype) Fatal {
    const s: []const u8 = std.fmt.bufPrint(&err_msg, fmt, args) catch err_msg[0..];
    err_len = s.len;
    return error.Fatal;
}
fn cfgFail() Fatal {
    return fail("{s}", .{config.lastError()});
}
fn ckptFail() Fatal {
    return fail("{s}", .{checkpoint.lastError()});
}
fn gpuFail() Fatal {
    return fail("{s}", .{gpu.lastError()});
}

/// Mean and standard error over the sampled batches, so that a small
/// difference can be told apart from none.
const Stat = struct {
    sum: f64 = 0,
    sumsq: f64 = 0,
    n: usize = 0,
    fn add(s: *Stat, x: f64) void {
        s.sum += x;
        s.sumsq += x * x;
        s.n += 1;
    }
    fn mean(s: Stat) f64 {
        return s.sum / @as(f64, @floatFromInt(@max(s.n, 1)));
    }
    fn stderror(s: Stat) f64 {
        if (s.n < 2) return 0;
        const m = s.mean();
        const fn_: f64 = @floatFromInt(s.n);
        const varr = @max(s.sumsq / fn_ - m * m, 0) * fn_ / (fn_ - 1);
        return @sqrt(varr / fn_);
    }
};

fn main_run(init: std.process.Init, out: Out) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 3) {
        try out.print("usage: memprobe CHECKPOINT DATA [samples=64] [seed=1] [key=value ...]\n", .{});
        return error.Usage;
    }
    const ckpt = argv[1];
    var samples: i64 = 64;
    var seed: u32 = 1;

    const file_cfg = checkpoint.readConfig(arena, io, ckpt) catch return ckptFail();
    var cfg = file_cfg;
    for (argv[3..]) |a| {
        const eq = std.mem.indexOfScalar(u8, a, '=') orelse return fail("expected key=value: {s}", .{a});
        const k = a[0..eq];
        const v = a[eq + 1 ..];
        if (std.mem.eql(u8, k, "samples")) {
            samples = std.fmt.parseInt(i64, v, 10) catch return fail("invalid samples", .{});
        } else if (std.mem.eql(u8, k, "seed")) {
            seed = std.fmt.parseInt(u32, v, 10) catch return fail("invalid seed", .{});
        } else config.set(&cfg, k, v) catch return cfgFail();
    }
    if (cfg.mem != 1) return fail("this checkpoint has mem=0; there is no store to probe", .{});
    if (samples <= 0) return fail("samples must be positive", .{});
    config.validate(cfg) catch return cfgFail();

    const data = std.Io.Dir.cwd().readFileAlloc(io, argv[2], arena, .unlimited) catch &.{};
    const B: usize = @intCast(cfg.batch);
    const T: usize = @intCast(cfg.seqlen);
    const M: usize = @intCast(cfg.mem_len);
    const N = B * T;
    if (data.len < 4 * (M + T + 2)) return fail("dataset too small for mem_len + seqlen", .{});

    var sess = session.Session.init(arena, io, cfg, kernels_ptx) catch return gpuFail();
    defer sess.deinit();
    checkpoint.loadWeights(arena, io, ckpt, &sess.m, file_cfg) catch return ckptFail();

    const ids = try arena.alloc(i32, N);
    const nxt = try arena.alloc(i32, N);
    const m_none = try arena.alloc(i32, B * M);
    const m_prev = try arena.alloc(i32, B * M);
    const m_rand = try arena.alloc(i32, B * M);
    @memset(m_none, -1); // the padding value the encoder skips

    var rng = stdrand.Mt19937.init(seed);
    var st_none: Stat = .{};
    var st_prev: Stat = .{};
    var st_rand: Stat = .{};

    const span = data.len - (M + T + 2);
    var s: i64 = 0;
    while (s < samples) : (s += 1) {
        for (0..B) |b| {
            const p = M + @as(usize, @intFromFloat(rng.canonical() * @as(f64, @floatFromInt(span))));
            for (0..T) |t| {
                ids[b * T + t] = data[p + t];
                nxt[b * T + t] = data[p + t + 1];
            }
            // The bytes immediately before the window, which the window
            // continues, against bytes from somewhere else entirely.
            for (0..M) |i| m_prev[b * M + i] = data[p - M + i];
            const q = @as(usize, @intFromFloat(rng.canonical() * @as(f64, @floatFromInt(span))));
            for (0..M) |i| m_rand[b * M + i] = data[q + i];
        }
        st_none.add((sess.forwardMem(ids, nxt, m_none) catch return gpuFail()).ce);
        st_prev.add((sess.forwardMem(ids, nxt, m_prev) catch return gpuFail()).ce);
        st_rand.add((sess.forwardMem(ids, nxt, m_rand) catch return gpuFail()).ce);
    }

    const ln2 = 0.6931471805599453;
    try out.print("memprobe: %s  dim=%d layers=%d mem_len=%d mem_heads=%d mem_rdim=%d\n", .{
        ckpt.ptr, @as(c_int, cfg.dim), @as(c_int, cfg.layers), @as(c_int, cfg.mem_len),
        @as(c_int, cfg.mem_heads), @as(c_int, cfg.mem_rdim),
    });
    try out.print("%lld batches of %d x %d bytes, store filled with the %d bytes before each window\n\n", .{
        @as(c_longlong, samples), @as(c_int, cfg.batch), @as(c_int, cfg.seqlen), @as(c_int, cfg.mem_len),
    });
    try out.print("%-8s %12s %12s %12s\n", .{ "store", "CE", "+- stderr", "BPB" });
    const rows = [_]struct { name: [:0]const u8, st: Stat }{
        .{ .name = "none", .st = st_none },
        .{ .name = "prev", .st = st_prev },
        .{ .name = "rand", .st = st_rand },
    };
    for (rows) |r|
        try out.print("%-8s %12.5f %12.5f %12.5f\n", .{ r.name.ptr, r.st.mean(), r.st.stderror(), r.st.mean() / ln2 });

    // What the numbers have to show for consolidation to have anything to
    // consolidate: the store must beat an unrelated store, not just an
    // empty one.
    const a_prev = st_none.mean() - st_prev.mean();
    const a_rand = st_none.mean() - st_rand.mean();
    // How much lower the loss is when the store holds the real preceding
    // context rather than unrelated text. Positive means content-specific.
    const signal = st_rand.mean() - st_prev.mean();
    const noise = @sqrt(st_prev.stderror() * st_prev.stderror() + st_rand.stderror() * st_rand.stderror());
    try out.print("\nA(prev) = CE(none) - CE(prev) = %+.5f\n", .{a_prev});
    try out.print("A(rand) = CE(none) - CE(rand) = %+.5f\n", .{a_rand});
    try out.print("content signal = CE(rand) - CE(prev) = %+.5f  (%.1f stderr)\n", .{
        signal, if (noise > 0) signal / noise else 0,
    });
    if (signal > 3 * noise) {
        try out.print("\nThe store carries something about the continuation that is specific to\nits content: the real preceding context beats unrelated text by %.3f nats.\nConsolidation has something to consolidate.\n", .{signal});
        if (a_prev < 0)
            try out.print("\nNote that filling the store still costs against leaving it empty. For a\nstore trained on facts rather than prose that is expected, and it does not\nweaken the result above - the comparison that matters is prev against rand,\nwhich holds the amount of content fixed and varies only what it is.\n", .{});
    } else {
        try out.print("\nNo content-specific advantage on bytes the store never saw. Whatever the\nstore helps with, it is not this, and section C stops here.\n", .{});
    }
}

pub fn main(init: std.process.Init) u8 {
    const io = init.io;
    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    main_run(init, .{ .w = &stdout.interface }) catch |e| {
        stdout.interface.flush() catch {};
        if (e == error.Usage) return 1;
        var ebuf: [600]u8 = undefined;
        var stderr = std.Io.File.stderr().writerStreaming(io, &ebuf);
        stderr.interface.print("error: {s}\n", .{err_msg[0..err_len]}) catch {};
        stderr.interface.flush() catch {};
        return 1;
    };
    stdout.interface.flush() catch {};
    return 0;
}
