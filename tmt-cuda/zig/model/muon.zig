//! Muon: the update direction for a two-dimensional weight is its momentum
//! passed through a Newton-Schulz iteration, which drives the singular values
//! of the update towards one. Vectors, the embedding and the output head keep
//! AdamW, as in the published recipe.
//!
//! The iteration runs in bf16 because its three matrix products per step go
//! over the tensor cores; the momentum, the weights and the norm stay fp32.
const std = @import("std");
const gpu = @import("gpu.zig");
const linalg = @import("linalg.zig");
const params = @import("params.zig");

pub const bf16 = u16;

/// The quintic coefficients of the Newton-Schulz iteration (Jordan et al.).
const NS_A: f32 = 3.4445;
const NS_B: f32 = -4.7750;
const NS_C: f32 = 2.0315;
const NS_STEPS = 5;
const MOMENTUM: f32 = 0.95;

/// Scratch shared by all matrices; sized for the largest one.
pub const Ws = struct {
    xb: [*]bf16 = undefined, // the iterate, rows x cols
    A: [*]bf16 = undefined, // X X^T, rows x rows
    AA: [*]bf16 = undefined, // its square
    B: [*]bf16 = undefined, // the polynomial b*A + c*A@A
    T: [*]bf16 = undefined, // B X
    sumsq: [*]f32 = undefined,
    rows: usize = 0,
    cols: usize = 0,

    pub fn alloc(w: *Ws, mem: *gpu.Memory, rows: usize, cols: usize) !void {
        w.rows = rows;
        w.cols = cols;
        w.xb = try mem.allocT(bf16, rows * cols);
        w.A = try mem.allocT(bf16, rows * rows);
        w.AA = try mem.allocT(bf16, rows * rows);
        w.B = try mem.allocT(bf16, rows * rows);
        w.T = try mem.allocT(bf16, rows * cols);
        w.sumsq = try mem.allocT(f32, 1);
    }
};

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

/// One weight matrix: momentum, orthogonalization, update.
pub fn step(k: *gpu.Kernels, w: *Ws, p: params.Par, x: [*]f32, lr: f32, wd: f32) !void {
    const rows: usize = @intCast(p.rows);
    const cols: usize = @intCast(p.cols);
    const n: usize = @intCast(p.n);
    try (try k.get("muon_momentum")).launch(blocks(n), 256, .{ p.m, p.grad, x, MOMENTUM, p.n });
    try gpu.zero(w.sumsq, 4);
    try (try k.get("muon_sumsq")).launch(@min(blocks(n), 1024), 256, .{ x, w.sumsq, p.n });
    try (try k.get("muon_start")).launch(blocks(n), 256, .{ x, w.sumsq, w.xb, p.n });
    for (0..NS_STEPS) |_| {
        // A = X X^T, then its square; both are symmetric, so one GEMM shape serves.
        try linalg.linearFwd(@intCast(rows), @intCast(rows), @intCast(cols), w.xb, w.xb, w.A);
        try linalg.linearFwd(@intCast(rows), @intCast(rows), @intCast(rows), w.A, w.A, w.AA);
        try (try k.get("muon_poly")).launch(blocks(rows * rows), 256,
            .{ w.A, w.AA, w.B, NS_B, NS_C, @as(i64, @intCast(rows * rows)) });
        try linalg.linearDX(@intCast(rows), @intCast(rows), @intCast(cols), w.B, w.xb, w.T, 0);
        try (try k.get("muon_blend")).launch(blocks(n), 256, .{ w.xb, w.T, NS_A, p.n });
    }
    // The published scaling keeps the update's size independent of the shape.
    const scale = params.sqrtf(@floatFromInt(@max(rows, cols)));
    try (try k.get("muon_update")).launch(blocks(n), 256, .{ p.master, w.xb, p.work, lr, scale, wd, p.n });
    // AdamW clears the gradients it consumes; these belong to Muon.
    try gpu.zeroAsync(p.grad, n * 4);
}
