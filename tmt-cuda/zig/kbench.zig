//! kbench: how close is each kernel to what this GPU can do?
//!
//! Every kernel runs on its own, at the shapes the model really uses, many
//! times between one pair of events, so the number is the kernel's own cost
//! and not the cost of measuring it. For each one the table states how many
//! bytes it must move, or how many multiply-adds it must do, and what share
//! of the machine's measured ceiling that works out to. `us/step` multiplies
//! by how often a training step runs it, which turns the table into a budget.
//!
//! usage: kbench [dim=512] [layers=16] [batch=16] [seqlen=128] [experts=8]
//!               [topk=2] [repeat=200]
const std = @import("std");
const gpu = @import("model/gpu.zig");
const linalg = @import("model/linalg.zig");
const ops = @import("model/ops.zig");
const ptx = @import("ptx.zig");
const cli = @import("cli.zig");

const kernels_ptx = @embedFile("kernels.ptx");
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

extern fn cuEventCreate(event: *?*anyopaque, flags: c_uint) c_int;
extern fn cuEventRecord(event: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn cuEventSynchronize(event: ?*anyopaque) c_int;
extern fn cuEventElapsedTime(ms: *f32, start: ?*anyopaque, end: ?*anyopaque) c_int;

var ev_start: ?*anyopaque = null;
var ev_stop: ?*anyopaque = null;

/// Warms up once, then times `repeat` calls between two events.
fn timeIt(repeat: usize, ctx: anytype, comptime body: anytype) !f64 {
    try body(ctx);
    try gpu.synchronize();
    if (ev_start == null) {
        _ = cuEventCreate(&ev_start, 0);
        _ = cuEventCreate(&ev_stop, 0);
    }
    const rc1 = cuEventRecord(ev_start, null);
    for (0..repeat) |_| try body(ctx);
    const rc2 = cuEventRecord(ev_stop, null);
    const rc3 = cuEventSynchronize(ev_stop);
    var ms: f32 = 0;
    const rc4 = cuEventElapsedTime(&ms, ev_start, ev_stop);
    if (rc1 != 0 or rc2 != 0 or rc3 != 0 or rc4 != 0) {
        var b2: [128]u8 = undefined;
        const n = snprintf(&b2, b2.len, "event error: %d %d %d %d\n", rc1, rc2, rc3, rc4);
        _ = std.Io.File.stderr().writeStreamingAll(undefined, b2[0..@intCast(n)]) catch {};
        return error.Event;
    }
    return @as(f64, ms) * 1000.0 / @as(f64, @floatFromInt(repeat));
}

const Row = struct {
    name: []const u8,
    us: f64,
    /// Bytes the kernel must read and write at this shape.
    bytes: f64 = 0,
    /// Multiply-adds it must perform, counted as two operations each.
    flops: f64 = 0,
    /// How often one training step runs it.
    per_step: f64 = 0,
};

var rows: std.ArrayList(Row) = .empty;

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const gpa = init.arena.allocator();
    var buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &stdout.interface;
    const args = try cli.Args.parse(gpa, init.minimal.args);

    const D: usize = try args.int(usize, "dim", 512);
    const B: usize = try args.int(usize, "batch", 16);
    const T: usize = try args.int(usize, "seqlen", 128);
    const E: usize = try args.int(usize, "experts", 8);
    const K: usize = try args.int(usize, "topk", 2);
    const layers: f64 = @floatFromInt(try args.int(usize, "layers", 16));
    const repeat: usize = try args.int(usize, "repeat", 200);
    const N = B * T;
    const ND: f64 = @floatFromInt(N * D);
    const fN: f64 = @floatFromInt(N);
    const fD: f64 = @floatFromInt(D);
    const NKD: f64 = @floatFromInt(N * K * D);

    var mod = try gpu.Kernels.load(gpa, kernels_ptx);
    var mem = gpu.Memory.init(gpa);
    defer mem.deinit();

    const stream_n: usize = 64 << 20; // far past any cache
    const span = @max(@max(N * D, N * 256) * 2, stream_n);
    const fa = try mem.callocT(f32, span);
    const fb = try mem.callocT(f32, span);
    const fc = try mem.callocT(f32, span);
    const ba = try mem.callocT(u16, span);
    const bb = try mem.callocT(u16, span);
    const bc = try mem.callocT(u16, span);
    const ia = try mem.callocT(i32, span);
    const islots = try mem.callocT(i32, span);
    // Index buffers decide how much these kernels collide, so they hold
    // realistic values: byte ids spread over 256 rows, and slots a
    // permutation, as the dispatch produces.
    {
        const host = try gpa.alloc(i32, @max(N * K, N));
        defer gpa.free(host);
        var r: u32 = 12345;
        for (host, 0..) |*v, i| {
            r = r *% 1664525 +% 1013904223;
            v.* = @intCast((r >> 8) % 256);
            _ = i;
        }
        try gpu.upload(ia, std.mem.sliceAsBytes(host[0..N]));
        for (host, 0..) |*v, i| v.* = @intCast(i % (N * K));
        try gpu.upload(islots, std.mem.sliceAsBytes(host[0 .. N * K]));
    }

    // ---- the ceilings, measured with the same clock ----
    const stream_us = try timeIt(20, .{ .k = try mod.get("copy_bf16"), .a = fa, .b = ba, .n = stream_n }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(@intCast((c.n + 255) / 256), 256, .{ c.a, c.b, @as(i64, @intCast(c.n)) });
        }
    }.f);
    const peak_gbs = 6.0 * @as(f64, @floatFromInt(stream_n)) / (stream_us * 1000.0);
    const gemm_us = try timeIt(50, .{ .a = ba, .b = bb, .c = bc }, struct {
        fn f(c: anytype) !void {
            try linalg.linearFwd(1024, 1024, 1024, c.a, c.b, c.c);
        }
    }.f);
    const peak_tflops = 2.0 * 1024.0 * 1024.0 * 1024.0 / (gemm_us * 1e6);

    // ---- the kernels, in the shapes of one layer ----
    const blocks: u32 = @intCast((N * D + 255) / 256);
    const nblocks: u32 = @intCast((N + 255) / 256);
    const gy: u32 = @intCast((D + 255) / 256);
    const opt = ops.CellOpt{};

    try rows.append(gpa, .{ .name = "copy_bf16", .bytes = 6 * ND, .per_step = 3 * layers, .us = try timeIt(repeat, .{ .k = try mod.get("copy_bf16"), .a = fa, .b = ba, .n = N * D, .g = blocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.a, c.b, @as(i64, @intCast(c.n)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "cast_add", .bytes = 10 * ND, .per_step = 1.5 * layers, .us = try timeIt(repeat, .{ .k = try mod.get("cast_add"), .a = ba, .b = fa, .n = N * D, .g = blocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.a, c.b, @as(i64, @intCast(c.n)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "state_pass", .bytes = 6 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("state_pass"), .x = ba, .s = fa, .d = fb, .B = B, .T = T, .D = D, .g = gy, .o = opt }, struct {
        fn f(c: anytype) !void {
            try c.k.launchGrid(@intCast(c.B), c.g, 256, .{ c.x, c.s, c.d, @as(u64, 0), @as(i32, @intCast(c.B)), @as(i32, @intCast(c.T)), @as(i32, @intCast(c.D)), @as(u64, 0), c.o });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "state_bwd", .bytes = 12 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("state_bwd"), .ds = ba, .s = fa, .d = fb, .dx = fc, .dd = fb, .x = bb, .B = B, .T = T, .D = D, .g = gy, .o = opt }, struct {
        fn f(c: anytype) !void {
            try c.k.launchGrid(@intCast(c.B), c.g, 256, .{ c.ds, c.s, c.d, c.dx, c.dd, @as(i32, @intCast(c.B)), @as(i32, @intCast(c.T)), @as(i32, @intCast(c.D)), c.x, @as(u64, 0), @as(u64, 0), @as(u64, 0), c.o });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "ln_fwd", .bytes = 4 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("ln_fwd"), .x = ba, .g1 = fa, .g2 = fb, .y = bb, .m = fc, .r = fc, .N = N, .D = D }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(@intCast(c.N), 256, .{ c.x, c.g1, c.g2, c.y, c.m, c.r, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.D)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "ln_bwd", .bytes = 6 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("ln_bwd"), .x = ba, .dy = bb, .g1 = fa, .m = fb, .r = fc, .dx = bc, .dg = fa, .db = fb, .N = N, .D = D }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(@intCast(c.N), 256, .{ c.x, c.dy, c.g1, c.m, c.r, c.dx, c.dg, c.db, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.D)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "emb_gather", .bytes = 4 * ND, .per_step = 1, .us = try timeIt(repeat, .{ .k = try mod.get("emb_gather"), .W = ba, .ids = ia, .o = bb, .N = N, .D = D, .g = nblocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.W, c.ids, c.o, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.D)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "emb_scatter", .bytes = 12 * ND, .per_step = 1, .us = try timeIt(repeat, .{ .k = try mod.get("emb_scatter"), .d = fa, .ids = ia, .W = fb, .N = N, .D = D, .g = nblocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.d, c.ids, c.W, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.D)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "ce_fwd", .bytes = 1536 * fN, .per_step = 1, .us = try timeIt(repeat, .{ .k = try mod.get("ce_fwd"), .l = ba, .t = ia, .p = fa, .o = fb, .N = N, .g = nblocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.l, c.t, c.p, c.o, @as(i32, @intCast(c.N)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "ce_bwd", .bytes = 1536 * fN, .per_step = 1, .us = try timeIt(repeat, .{ .k = try mod.get("ce_bwd"), .p = fa, .t = ia, .d = ba, .N = N, .g = nblocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.p, c.t, c.d, @as(f32, 1.0), @as(i32, @intCast(c.N)) });
        }
    }.f) });
    try rows.append(gpa, .{ .name = "dense_activation", .bytes = 6 * ND, .per_step = 2 * layers, .us = try timeIt(repeat, .{ .k = try mod.get("dense_activation"), .p = ba, .g1 = bb, .o = bc, .n = N * D, .g = blocks }, struct {
        fn f(c: anytype) !void {
            try c.k.launch(c.g, 256, .{ c.p, c.g1, c.o, @as(i64, @intCast(c.n)), @as(f32, 0) });
        }
    }.f) });
    if (E > 1) {
        try rows.append(gpa, .{ .name = "router_topk", .bytes = 2 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("router_topk"), .x = ba, .W = bb, .l = fa, .p = fb, .i = ia, .w = fc, .N = N, .E = E, .K = K, .D = D }, struct {
            fn f(c: anytype) !void {
                try c.k.launch(@intCast(c.N), 64, .{ c.x, c.W, c.l, c.p, c.i, c.w, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.E)), @as(i32, @intCast(c.K)), @as(i32, @intCast(c.D)) });
            }
        }.f) });
        try rows.append(gpa, .{ .name = "gather_slot", .bytes = 4 * NKD, .per_step = 2 * layers, .us = try timeIt(repeat, .{ .k = try mod.get("gather_slot"), .x = ba, .s = islots, .o = bb, .N = N, .K = K, .D = D, .g = @as(u32, @intCast((N * K * D + 255) / 256)) }, struct {
            fn f(c: anytype) !void {
                try c.k.launch(c.g, 256, .{ c.x, c.s, c.o, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.K)), @as(i32, @intCast(c.D)) });
            }
        }.f) });
        try rows.append(gpa, .{ .name = "combine", .bytes = 2 * NKD + 4 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("combine"), .y = ba, .w = fa, .s = islots, .o = bb, .N = N, .K = K, .D = D, .g = blocks }, struct {
            fn f(c: anytype) !void {
                try c.k.launch(c.g, 256, .{ c.y, c.w, c.s, c.o, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.K)), @as(i32, @intCast(c.D)), @as(f32, 1) });
            }
        }.f) });
        try rows.append(gpa, .{ .name = "scatter_add", .bytes = 2 * NKD + 4 * ND, .per_step = layers, .us = try timeIt(repeat, .{ .k = try mod.get("scatter_add"), .d = ba, .s = islots, .o = bb, .N = N, .K = K, .D = D, .g = blocks }, struct {
            fn f(c: anytype) !void {
                try c.k.launch(c.g, 256, .{ c.d, c.s, c.o, @as(i32, @intCast(c.N)), @as(i32, @intCast(c.K)), @as(i32, @intCast(c.D)) });
            }
        }.f) });
    }
    // The matmuls of one layer: the expert (or dense) projection and its two
    // gradients, all (N x D) by (D x D).
    const rowsK = if (E > 1) N * K else N;
    try rows.append(gpa, .{ .name = "gemm all rows", .flops = 2 * @as(f64, @floatFromInt(rowsK)) * fD * fD, .per_step = if (E > 1) 0 else layers, .us = try timeIt(repeat, .{ .a = ba, .b = bb, .c = bc, .M = rowsK, .D = D }, struct {
        fn f(c: anytype) !void {
            try linalg.linearFwd(@intCast(c.M), @intCast(c.D), @intCast(c.D), c.a, c.b, c.c);
        }
    }.f) });
    try rows.append(gpa, .{ .name = "gemm dX", .flops = 2 * @as(f64, @floatFromInt(rowsK)) * fD * fD, .per_step = if (E > 1) 0 else layers, .us = try timeIt(repeat, .{ .a = ba, .b = bb, .c = bc, .M = rowsK, .D = D }, struct {
        fn f(c: anytype) !void {
            try linalg.linearDX(@intCast(c.M), @intCast(c.D), @intCast(c.D), c.a, c.b, c.c, 0);
        }
    }.f) });
    try rows.append(gpa, .{ .name = "gemm dW", .flops = 2 * @as(f64, @floatFromInt(rowsK)) * fD * fD, .per_step = if (E > 1) 0 else layers, .us = try timeIt(repeat, .{ .a = ba, .b = bb, .c = fa, .M = rowsK, .D = D }, struct {
        fn f(c: anytype) !void {
            try linalg.linearDW(@intCast(c.M), @intCast(c.D), @intCast(c.D), c.a, c.b, c.c, 0);
        }
    }.f) });
    if (E > 1) {
        // What the dispatch really runs: one matmul per expert, each over its
        // own rows, padded to a multiple of 128.
        const per_expert = ((N * K / E + 127) / 128) * 128;
        try rows.append(gpa, .{ .name = "gemm one expert", .flops = 2 * @as(f64, @floatFromInt(per_expert)) * fD * fD, .per_step = layers * @as(f64, @floatFromInt(E)) * 3, .us = try timeIt(repeat, .{ .a = ba, .b = bb, .c = bc, .M = per_expert, .D = D }, struct {
            fn f(c: anytype) !void {
                try linalg.linearFwd(@intCast(c.M), @intCast(c.D), @intCast(c.D), c.a, c.b, c.c);
            }
        }.f) });
    }
    try rows.append(gpa, .{ .name = "gemm decoder", .flops = 2 * fN * fD * 256, .per_step = 3, .us = try timeIt(repeat, .{ .a = ba, .b = bb, .c = bc, .N = N, .D = D }, struct {
        fn f(c: anytype) !void {
            try linalg.linearFwd(@intCast(c.N), 256, @intCast(c.D), c.a, c.b, c.c);
        }
    }.f) });

    if (E > 1) {
        // Eight expert matmuls as one batched call, against the same eight
        // issued one after the other.
        const per_expert = ((N * K / E + 127) / 128) * 128;
        const table = try gpa.alloc(u64, E * 3);
        defer gpa.free(table);
        const pa = try mem.allocT(u64, E);
        const pb = try mem.allocT(u64, E);
        const pc = try mem.allocT(u64, E);
        for (0..E) |e| {
            table[e] = @intFromPtr(ba + e * per_expert * D);
            table[E + e] = @intFromPtr(bb + e * D * D);
            table[2 * E + e] = @intFromPtr(bc + e * per_expert * D);
        }
        try gpu.upload(pa, std.mem.sliceAsBytes(table[0..E]));
        try gpu.upload(pb, std.mem.sliceAsBytes(table[E .. 2 * E]));
        try gpu.upload(pc, std.mem.sliceAsBytes(table[2 * E ..]));
        const flops = 2 * @as(f64, @floatFromInt(per_expert * E)) * fD * fD;
        try rows.append(gpa, .{ .name = "gemm experts batched", .flops = flops, .per_step = layers * 3, .us = try timeIt(repeat, .{ .a = @intFromPtr(pa), .b = @intFromPtr(pb), .c = @intFromPtr(pc), .M = per_expert, .D = D, .E = E }, struct {
            fn f(c: anytype) !void {
                try linalg.linearFwdBatched(@intCast(c.M), @intCast(c.D), @intCast(c.D), c.a, c.b, c.c, @intCast(c.E));
            }
        }.f) });
        try rows.append(gpa, .{ .name = "gemm experts in turn", .flops = flops, .per_step = layers * 3, .us = try timeIt(repeat, .{ .a = ba, .b = bb, .c = bc, .M = per_expert, .D = D, .E = E }, struct {
            fn f(c: anytype) !void {
                for (0..c.E) |e| try linalg.linearFwd(@intCast(c.M), @intCast(c.D), @intCast(c.D), c.a + e * c.M * c.D, c.b + e * c.D * c.D, c.c + e * c.M * c.D);
            }
        }.f) });
    }

    // Host cost of issuing work: the wall-clock time to enqueue many calls
    // without waiting for any of them. Device time says what the GPU does;
    // this says whether the host can keep it fed.
    {
        const issue = 2000;
        const fissue: f64 = @floatFromInt(issue);
        var at = std.Io.Clock.awake.now(io);
        for (0..issue) |_| try linalg.linearFwd(512, @intCast(D), @intCast(D), ba, bb, bc);
        var ns = at.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        try gpu.synchronize();
        try rows.append(gpa, .{ .name = "cublas call (host)", .us = @as(f64, @floatFromInt(ns)) / 1000.0 / fissue, .per_step = 0 });
        const k = try mod.get("copy_bf16");
        at = std.Io.Clock.awake.now(io);
        for (0..issue) |_| try k.launch(1, 32, .{ fa, ba, @as(i64, 1) });
        ns = at.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        try gpu.synchronize();
        try rows.append(gpa, .{ .name = "kernel launch (host)", .us = @as(f64, @floatFromInt(ns)) / 1000.0 / fissue, .per_step = 0 });
    }

    // How much does issuing work cost, independent of the work itself? A
    // kernel with nothing to do, launched many times without waiting.
    {
        const k = try mod.get("zero_f32");
        const us = try timeIt(2000, .{ .k = k, .a = fa }, struct {
            fn f(c: anytype) !void {
                try c.k.launch(1, 32, .{ c.a, @as(i64, 1) });
            }
        }.f);
        try rows.append(gpa, .{ .name = "empty launch", .us = us, .per_step = 0 });
    }

    // ---- the table ----
    // Printed separately: one snprintf with many mixed integer and float
    // arguments comes out shifted.
    var line: [512]u8 = undefined;
    var n = snprintf(&line, line.len, "\nshapes: N=%d D=%d E=%d K=%d\n", @as(c_int, @intCast(N)), @as(c_int, @intCast(D)), @as(c_int, @intCast(E)), @as(c_int, @intCast(K)));
    try w.writeAll(line[0..@intCast(n)]);
    n = snprintf(&line, line.len, "ceilings: %.0f GB/s streaming\n", peak_gbs);
    try w.writeAll(line[0..@intCast(n)]);
    n = snprintf(&line, line.len, "          %.1f TFLOPS bf16 at 1024^3\n", peak_tflops);
    try w.writeAll(line[0..@intCast(n)]);
    n = snprintf(&line, line.len, "%-20s%10s%10s%10s%9s%10s\n", "kernel".ptr, "us/call".ptr, "GB/s".ptr, "TFLOPS".ptr, "ceiling".ptr, "us/step".ptr);
    try w.writeAll(line[0..@intCast(n)]);
    var budget: f64 = 0;
    for (rows.items) |r| {
        const gbs = if (r.bytes > 0) r.bytes / (r.us * 1000.0) else 0;
        const tf = if (r.flops > 0) r.flops / (r.us * 1e6) else 0;
        const frac = if (r.flops > 0) tf / peak_tflops else gbs / peak_gbs;
        budget += r.us * r.per_step;
        const m = snprintf(&line, line.len, "%-20s%10.1f%10.0f%10.1f%8.0f%%%10.1f\n", r.name.ptr, r.us, gbs, tf, 100 * frac, r.us * r.per_step);
        try w.writeAll(line[0..@intCast(m)]);
    }
    n = snprintf(&line, line.len, "%-20s%48.1f us of device work per training step\n", "sum".ptr, budget);
    try w.writeAll(line[0..@intCast(n)]);
    try w.flush();
    return 0;
}
