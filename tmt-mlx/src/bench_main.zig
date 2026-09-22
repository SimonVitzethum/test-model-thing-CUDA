// bench [B T D]: recurrence kernel forward and forward+backward against a CPU
// reference, plus the device's memory bandwidth and bf16 matmul roof.
const std = @import("std");
const mx = @import("mlx.zig");
const cell = @import("cell.zig");
const u = @import("util.zig");

fn timeIt(iters: usize, f: anytype, args: anytype) f64 {
    _ = @call(.auto, f, args);
    mx.synchronize();
    const t0 = u.now();
    for (0..iters) |_| _ = @call(.auto, f, args);
    mx.synchronize();
    return (u.now() - t0) / @as(f64, @floatFromInt(iters));
}

const Inputs = struct { x: mx.Array, dec: mx.Array, gate: mx.Array, carry: mx.Array, ids: mx.Array };
const opt = cell.Opt{ .gated = true, .docsep = -1 };

fn forward(in: Inputs) void {
    const m = mx.mark();
    defer mx.release(m);
    mx.eval1(cell.forwardRaw(in.x, in.dec, in.gate, in.carry, in.ids, opt));
}
fn forwardBackward(in: Inputs) void {
    const m = mx.mark();
    defer mx.release(m);
    const S = cell.forwardRaw(in.x, in.dec, in.gate, in.carry, in.ids, opt);
    const g = cell.backwardRaw(S, S, in.x, in.dec, in.gate, in.carry, in.ids, opt);
    mx.eval(&.{ g.dx, g.gdec, g.ggate, g.dcarry });
}
fn copyOnce(a: mx.Array) void {
    const m = mx.mark();
    defer mx.release(m);
    mx.eval1(mx.addS(a, 1));
}
fn matmulOnce(a: mx.Array, b: mx.Array) void {
    const m = mx.mark();
    defer mx.release(m);
    mx.eval1(mx.matmul(a, b));
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    mx.init();
    const B: i32 = if (args.len > 1) u.parseInt(i32, args[1]) else 16;
    const T: i32 = if (args.len > 2) u.parseInt(i32, args[2]) else 128;
    const D: i32 = if (args.len > 3) u.parseInt(i32, args[3]) else 1280;
    if (B <= 0 or T <= 0 or D <= 0) u.die("require B T D > 0", .{});
    const N: usize = @intCast(B * T * D);
    u.print("cell: B={d} T={d} D={d} ({d:.1}M elem)\n", .{ B, T, D, @as(f64, @floatFromInt(N)) / 1e6 });

    var rng = std.Random.DefaultPrng.init(0);
    const r = rng.random();
    const hx = try u.gpa.alloc(f32, N);
    for (hx) |*v| v.* = r.float(f32) * 2 - 1;
    const hdec = try u.gpa.alloc(f32, @intCast(D));
    const hgate = try u.gpa.alloc(f32, @intCast(D));
    for (hdec, hgate, 0..) |*d, *g, i| {
        d.* = 2;
        g.* = 0.3 * @sin(@as(f32, @floatFromInt(i)));
    }
    const x = mx.keep(mx.astype(mx.fromSlice(f32, hx, &.{ B, T, D }), mx.bf16));
    const in = Inputs{
        .x = x,
        .dec = mx.keep(mx.fromSlice(f32, hdec, &.{D})),
        .gate = mx.keep(mx.fromSlice(f32, hgate, &.{D})),
        .carry = mx.keep(mx.zeros(&.{ B, D }, mx.f32_)),
        .ids = mx.keep(mx.zeros(&.{ B, T }, mx.i32_)),
    };
    // CPU reference on the bf16-rounded inputs.
    const xb = try u.gpa.alloc(f32, N);
    mx.toF32(x, xb);
    const S = mx.toVecF32(cell.forwardRaw(in.x, in.dec, in.gate, in.carry, in.ids, opt));
    var max_err: f64 = 0;
    const Bu: usize = @intCast(B);
    const Tu: usize = @intCast(T);
    const Du: usize = @intCast(D);
    for (0..Bu) |b| for (0..Du) |d| {
        var st: f64 = 0;
        for (0..Tu) |t| {
            const i = (b * Tu + t) * Du + d;
            const a = 1 / (1 + @exp(-(@as(f64, hdec[d]) + @as(f64, hgate[d]) * xb[i])));
            st = a * st + (1 - a) * xb[i];
            max_err = @max(max_err, @abs(st - S[i]));
        }
    };

    const fwd = timeIt(50, forward, .{in});
    const both = timeIt(50, forwardBackward, .{in});
    const nf: f64 = @floatFromInt(N);
    // Estimated traffic: forward reads x (2 B) and writes S (4 B); backward reads dS, S, x and writes dx.
    u.print("max error vs CPU: {e:.2} {s}\n", .{ max_err, if (max_err < 1e-4) "OK" else "FAIL" });
    u.print("forward: {d:.3} ms, estimated {d:.0} GB/s\n", .{ fwd * 1e3, nf * 6 / fwd / 1e9 });
    u.print("forward+backward: {d:.3} ms, estimated {d:.0} GB/s\n", .{ both * 1e3, nf * 20 / both / 1e9 });

    // Roofline probes.
    const big = mx.keep(mx.zeros(&.{1 << 26}, mx.f32_));
    const copy = timeIt(10, copyOnce, .{big});
    u.print("memory: {d:.0} GB/s (read+write, 256 MB)\n", .{2.0 * 4 * (1 << 26) / copy / 1e9});
    const n: i32 = 4096;
    const a = mx.keep(mx.astype(mx.full(&.{ n, n }, 0.01, mx.f32_), mx.bf16));
    const mm = timeIt(10, matmulOnce, .{ a, a });
    const nn: f64 = @floatFromInt(n);
    u.print("bf16 matmul {d}^3: {d:.2} TFLOPS\n", .{ n, 2 * nn * nn * nn / mm / 1e12 });
    if (max_err >= 1e-4) std.process.exit(2);
}
