//! fp4test: how much does NVFP4 cost in accuracy, and what does it buy?
//!
//! Two questions, in order. First the quantiser on its own: take real bf16
//! weights, encode to e2m1 with e4m3 block scales, decode, and compare. That
//! bounds everything downstream - a matmul cannot be more accurate than its
//! operands. Then the matmul itself, against the bf16 one it would replace.
//!
//! usage: fp4test [n=4194304] [m=512] [k=512]
const std = @import("std");
const gpu = @import("model/gpu.zig");
const linalg = @import("model/linalg.zig");
const cli = @import("cli.zig");
const fp4 = @import("model/fp4.zig");

const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(id: c_int, ts: *Timespec) c_int;
fn now() f64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e9 + @as(f64, @floatFromInt(ts.nsec));
}

const kernels_ptx = @embedFile("kernels.ptx");
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    const r = (b +% 0x7FFF +% ((b >> 16) & 1)) >> 16;
    return @truncate(r);
}
fn unbf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const gpa = init.arena.allocator();
    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &stdout.interface;
    const args = try cli.Args.parse(gpa, init.minimal.args);
    const n: usize = try args.int(usize, "n", 4 << 20);
    const blocks = n / 16;
    // bit 0 stochastic rounding, bit 1 Hadamard rotation inside the block
    const qmode: i32 = @intCast(try args.int(u32, "mode", 3));

    var mod = try gpu.Kernels.load(gpa, kernels_ptx);
    var mem = gpu.Memory.init(gpa);
    defer mem.deinit();
    const src = try mem.allocT(u16, n);
    const packed_ = try mem.allocT(u8, n / 2);
    const scale = try mem.allocT(u8, blocks);
    const back = try mem.allocT(u16, n);
    const gscale = try mem.allocT(f32, 1);

    var b: [512]u8 = undefined;
    const p = struct {
        fn f(out: *std.Io.Writer, bb: []u8, comptime fmt: [:0]const u8, a2: anytype) !void {
            const c = @call(.auto, snprintf, .{ bb.ptr, bb.len, fmt.ptr } ++ a2);
            try out.writeAll(bb[0..@intCast(@min(c, bb.len - 1))]);
            try out.flush();
        }
    }.f;
    try p(w, &b, "fp4test: %zu values, %zu blocks of 16\n\n", .{ n, blocks });
    try p(w, &b, "%-26s %12s %12s %10s\n", .{ "distribution", "rel L2", "max rel", "bits/val" });

    // Three shapes: what weights look like, what activations look like after
    // a norm, and one with an outlier, which is the case block scaling is for.
    const kinds = [_][:0]const u8{ "gaussian (weights)  ", "gaussian + outliers ", "lognormal (acts)    " };
    for (kinds, 0..) |kind, ki| {
        const host = try gpa.alloc(u16, n);
        defer gpa.free(host);
        var st: u64 = 12345 + ki;
        for (host, 0..) |*q, i| {
            st = st *% 6364136223846793005 +% 1442695040888963407;
            const u1v = @as(f32, @floatFromInt((st >> 33) & 0xFFFFFF)) / 16777216.0 + 1e-7;
            st = st *% 6364136223846793005 +% 1442695040888963407;
            const u2v = @as(f32, @floatFromInt((st >> 33) & 0xFFFFFF)) / 16777216.0;
            var g = @sqrt(-2.0 * @log(u1v)) * @cos(6.2831853 * u2v) * 0.02;
            if (ki == 1 and i % 997 == 0) g *= 60.0; // one outlier per block-ish
            if (ki == 2) g = @exp(g * 8.0) * 0.01;
            q.* = bf(g);
        }
        try gpu.upload(src, std.mem.sliceAsBytes(host));
        try gpu.zero(gscale, 4);
        try (try mod.get("fp4_absmax")).launch(256, 256, .{ src, gscale, @as(i64, @intCast(n)) });
        // The per-tensor scale divides the block scales into e4m3's range:
        // largest magnitude / 6 for e2m1, then / 448 for e4m3's own maximum.
        var hg: f32 = 0;
        try gpu.download(std.mem.asBytes(&hg), gscale);
        hg = if (hg > 0) hg / 6.0 / 448.0 else 1.0;
        try gpu.upload(gscale, std.mem.asBytes(&hg));
        try (try mod.get("fp4_quant")).launch(@intCast((blocks + 255) / 256), 256,
            .{ src, packed_, scale, gscale, qmode, @as(u32, 1), @as(i64, @intCast(blocks)) });
        try (try mod.get("fp4_dequant")).launch(@intCast((blocks + 255) / 256), 256,
            .{ packed_, scale, gscale, qmode, back, @as(i64, @intCast(blocks)) });
        const got = try gpa.alloc(u16, n);
        defer gpa.free(got);
        try gpu.download(std.mem.sliceAsBytes(got), back);

        var num: f64 = 0;
        var den: f64 = 0;
        var mxrel: f64 = 0;
        for (host, got) |a, c| {
            const x = unbf(a);
            const y = unbf(c);
            num += @as(f64, (x - y)) * (x - y);
            den += @as(f64, x) * x;
            if (@abs(x) > 1e-6) mxrel = @max(mxrel, @abs((x - y) / x));
        }
        // 4 bits per value plus one 8-bit scale per sixteen.
        try p(w, &b, "%-26s %12.5f %12.3f %10.2f\n",
            .{ kind.ptr, @sqrt(num / @max(den, 1e-30)), mxrel, @as(f64, 4.5) });
    }
    try p(w, &b, "\nbf16 is 16 bits per value, so NVFP4 moves 3.6x fewer bytes.\n", .{});

    // The matmul itself, against the bf16 one it would replace. The shapes
    // are the ones the model runs: an expert is D x D, and the rows are a
    // window's worth of tokens routed to it.
    try p(w, &b, "\n%-28s %10s %10s %9s %10s\n", .{ "shape (M,N,K)", "bf16 us", "nvfp4 us", "speedup", "rel err" });
    const shapes = [_][3]i32{
        .{ 1024, 512, 512 }, .{ 2048, 512, 512 }, .{ 4096, 512, 512 },
        .{ 512, 512, 2048 }, .{ 2048, 2048, 2048 },
    };
    for (shapes) |sh| {
        const M = sh[0];
        const NN = sh[1];
        const K = sh[2];
        const um: usize = @intCast(M);
        const un: usize = @intCast(NN);
        const uk: usize = @intCast(K);
        // bf16 operands, and the same values quantised.
        const Ab = try mem.allocT(u16, um * uk);
        const Bb = try mem.allocT(u16, un * uk);
        const Db = try mem.allocT(u16, um * un);
        const Aq = try mem.allocT(u8, um * uk / 2);
        const Bq = try mem.allocT(u8, un * uk / 2);
        const As = try mem.allocT(u8, um * uk / 16);
        const Bs = try mem.allocT(u8, un * uk / 16);
        const Dq = try mem.allocT(u16, um * un);
        const ga = try mem.allocT(f32, 1);
        const dal = try mem.allocT(f32, 1);
        const dze = try mem.callocT(f32, 1);

        const ha = try gpa.alloc(u16, um * uk);
        const hb2 = try gpa.alloc(u16, un * uk);
        defer gpa.free(ha);
        defer gpa.free(hb2);
        // Gaussian with the occasional outlier, because that is what weights
        // and activations look like - and because uniform data is the one
        // distribution on which neither stochastic rounding nor a rotation
        // can help, so testing on it answers the wrong question.
        var st: u64 = 7;
        const fill = struct {
            fn f(dst2: []u16, seed: *u64, outliers: bool) void {
                for (dst2, 0..) |*q, i| {
                    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
                    const u1v = @as(f32, @floatFromInt((seed.* >> 33) & 0xFFFFFF)) / 16777216.0 + 1e-7;
                    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
                    const u2v = @as(f32, @floatFromInt((seed.* >> 33) & 0xFFFFFF)) / 16777216.0;
                    var g2 = @sqrt(-2.0 * @log(u1v)) * @cos(6.2831853 * u2v) * 0.02;
                    if (outliers and i % 211 == 0) g2 *= 25.0;
                    q.* = bf(g2);
                }
            }
        }.f;
        fill(ha, &st, true);
        fill(hb2, &st, false);
        try gpu.upload(Ab, std.mem.sliceAsBytes(ha));
        try gpu.upload(Bb, std.mem.sliceAsBytes(hb2));

        var amax: f32 = 0;
        for (ha) |q| amax = @max(amax, @abs(unbf(q)));
        for (hb2) |q| amax = @max(amax, @abs(unbf(q)));
        var hg2: f32 = amax / 6.0 / 448.0;
        try gpu.upload(ga, std.mem.asBytes(&hg2));
        const nbA = um * uk / 16;
        const nbB = un * uk / 16;
        try (try mod.get("fp4_quant")).launch(@intCast((nbA + 255) / 256), 256, .{ Ab, Aq, As, ga, qmode, @as(u32, 1), @as(i64, @intCast(nbA)) });
        try (try mod.get("fp4_quant")).launch(@intCast((nbB + 255) / 256), 256, .{ Bb, Bq, Bs, ga, qmode, @as(u32, 2), @as(i64, @intCast(nbB)) });

        var g = fp4.Gemm.init(M, NN, K, @ptrCast(Bs), @ptrCast(As)) catch {
            try p(w, &b, "%5d,%5d,%-14d %s\n", .{ M, NN, K, fp4.lastError().ptr });
            continue;
        };
        defer g.deinit();

        const reps: usize = 50;
        try linalg.linearFwd(M, NN, K, Ab, Bb, Db);
        {
            const a2 = hg2 * hg2;
            try gpu.upload(dal, std.mem.asBytes(&a2));
        }
        try g.run(@ptrCast(Bq), @ptrCast(Aq), @ptrCast(Dq), @ptrCast(dal), @ptrCast(dze));
        try gpu.synchronize();
        var t0 = now();
        for (0..reps) |_| try linalg.linearFwd(M, NN, K, Ab, Bb, Db);
        try gpu.synchronize();
        const t_bf = (now() - t0) / 1e3 / @as(f64, @floatFromInt(reps));
        t0 = now();
        for (0..reps) |_| {
            const a2 = hg2 * hg2;
            try gpu.upload(dal, std.mem.asBytes(&a2));
        }
        try g.run(@ptrCast(Bq), @ptrCast(Aq), @ptrCast(Dq), @ptrCast(dal), @ptrCast(dze));
        try gpu.synchronize();
        const t_q = (now() - t0) / 1e3 / @as(f64, @floatFromInt(reps));

        const hd1 = try gpa.alloc(u16, um * un);
        const hd2 = try gpa.alloc(u16, um * un);
        defer gpa.free(hd1);
        defer gpa.free(hd2);
        try gpu.download(std.mem.sliceAsBytes(hd1), Db);
        try gpu.download(std.mem.sliceAsBytes(hd2), Dq);
        var num: f64 = 0;
        var den: f64 = 0;
        for (hd1, hd2) |x, y| {
            const a = unbf(x);
            const c = unbf(y);
            num += @as(f64, a - c) * (a - c);
            den += @as(f64, a) * a;
        }
        try p(w, &b, "%5d,%5d,%-14d %10.1f %10.1f %8.2fx %10.4f\n",
            .{ M, NN, K, t_bf, t_q, t_bf / t_q, @sqrt(num / @max(den, 1e-30)) });
    }
    // Does the batched form exist? The experts run batched, so an FP4 path
    // that cannot batch cannot replace them.
    {
        const M: i32 = 2048;
        const NN: i32 = 512;
        const K: i32 = 512;
        const um: usize = @intCast(M);
        const un: usize = @intCast(NN);
        const uk: usize = @intCast(K);
        for ([_]i32{ 2, 8, 16 }) |bc| {
            const ub: usize = @intCast(bc);
            const As = try mem.allocT(u8, ub * um * uk / 16);
            const Bs = try mem.allocT(u8, ub * un * uk / 16);
            var gg = fp4.Gemm.initBatched(M, NN, K, @ptrCast(Bs), @ptrCast(As), bc,
                @intCast(un * uk), @intCast(um * uk), @intCast(um * un)) catch {
                try p(w, &b, "batched x%-3d %s\n", .{ bc, fp4.lastError().ptr });
                continue;
            };
            gg.deinit();
            try p(w, &b, "batched x%-3d algorithm found\n", .{bc});
        }
    }
    return 0;
}
