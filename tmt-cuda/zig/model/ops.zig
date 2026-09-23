//! The thin host wrappers around the kernels: the recurrence, LayerNorm and
//! the embedding table (ports of the inline functions in src/cell.cu,
//! src/norm.cu and src/emb.cu).
const std = @import("std");
const gpu = @import("gpu.zig");

pub const bf16 = u16;

/// Options of the recurrence kernels, same layout as the kernel's CellOpt.
pub const CellOpt = extern struct {
    ids: u64 = 0,
    docsep: i32 = -1,
    gamma: f32 = 1,
    trDec: u64 = 0,
    trGate: u64 = 0,
    lam: u64 = 0,
    prod: u64 = 0,
    logDec: u64 = 0,
    logGate: u64 = 0,
    logEmb: u64 = 0,
    /// (B,T) 0 marks a padded step, which leaves the state untouched.
    mask: u64 = 0,
    /// (B,T) bytes per step, so a half-life stays one in bytes at patch rate.
    plen: u64 = 0,
};

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

pub fn embForward(k: *gpu.Kernels, W: [*]const bf16, ids: [*]const i32, out: [*]bf16, N: i32, D: i32) !void {
    try (try k.get("emb_gather")).launch(blocks(@intCast(N)), 256, .{ W, ids, out, N, D });
}
/// The deterministic form: one thread per (byte, channel), summing the
/// segment the host sorted for it.
pub fn embBackwardSorted(k: *gpu.Kernels, dOut: [*]const f32, perm: [*]const i32,
                         tile_from: [*]const i32, tile_to: [*]const i32, byte_tiles: [*]const i32,
                         partial: [*]f32, tiles: usize, dW: [*]f32, D: i32) !void {
    const gy: u32 = @intCast(@divTrunc(D + 255, 256));
    try (try k.get("emb_tiles")).launchGrid(@intCast(tiles), gy, 256, .{ dOut, perm, tile_from, tile_to, partial, D });
    try (try k.get("emb_reduce")).launchGrid(256, gy, 256, .{ partial, byte_tiles, dW, D });
}
pub fn embBackward(k: *gpu.Kernels, dOut: [*]const f32, ids: [*]const i32, dW: [*]f32, N: i32, D: i32) !void {
    try (try k.get("emb_scatter")).launch(blocks(@intCast(N)), 256, .{ dOut, ids, dW, N, D });
}

/// Only the state loop; the caller normalizes afterwards.
pub fn stateForward(k: *gpu.Kernels, X: [*]const bf16, S: [*]f32, decay: [*]const f32, carry: u64,
                    B: i32, T: i32, D: i32, gate: u64, o: CellOpt) !void {
    const gy: u32 = @intCast(@divTrunc(D + 255, 256));
    try (try k.get("state_pass")).launchGrid(@intCast(B), gy, 256, .{ X, S, decay, carry, B, T, D, gate, o });
}

/// The three passes of the recurrent cell: state, statistics, output.
pub fn cellForward(k: *gpu.Kernels, X: [*]const bf16, S: [*]f32, decay: [*]const f32,
                   mean: [*]f32, rstd: [*]f32, Y: [*]bf16, B: i32, T: i32, D: i32) !void {
    const gy: u32 = @intCast(@divTrunc(D + 255, 256));
    try (try k.get("state_pass")).launchGrid(@intCast(B), gy, 256,
        .{ X, S, decay, @as(u64, 0), B, T, D, @as(u64, 0), CellOpt{} });
    try (try k.get("stats_pass")).launch(@intCast(B * T), 256, .{ S, mean, rstd, B, T, D });
    try (try k.get("out_pass")).launchGrid(@intCast(B), gy, 256, .{ X, S, mean, rstd, Y, B, T, D });
}

/// Exact within-window derivative; zeroes the two parameter gradients first.
pub fn cellBackward(k: *gpu.Kernels, dS: [*]const bf16, S: [*]const f32, decay: [*]const f32,
                    dX: [*]f32, dDec: [*]f32, B: i32, T: i32, D: i32, X: [*]const bf16,
                    initial: u64, gate: u64, dGate: u64, o: CellOpt) !void {
    const gy: u32 = @intCast(@divTrunc(D + 255, 256));
    try gpu.zero(dDec, @as(usize, @intCast(D)) * 4);
    if (dGate != 0) try gpu.zero(@as(*anyopaque, @ptrFromInt(dGate)), @as(usize, @intCast(D)) * 4);
    try (try k.get("state_bwd")).launchGrid(@intCast(B), gy, 256,
        .{ dS, S, decay, dX, dDec, B, T, D, X, initial, gate, dGate, o });
}

pub fn embTrace(k: *gpu.Kernels, X: [*]const bf16, S: [*]const f32, initial: [*]const f32,
                decay: [*]const f32, gate: u64, trEmb: [*]f32, dEmb: [*]f32,
                B: i32, T: i32, D: i32, o: CellOpt) !void {
    const gy: u32 = @intCast(@divTrunc(D + 255, 256));
    try (try k.get("emb_trace")).launchGrid(@intCast(B), gy, 256,
        .{ X, S, initial, decay, gate, trEmb, dEmb, B, T, D, o });
}

pub fn layernormFwd(k: *gpu.Kernels, X: [*]const bf16, gamma: [*]const f32, beta: [*]const f32,
                    Y: [*]bf16, mean: [*]f32, rstd: [*]f32, N: i32, D: i32) !void {
    try (try k.get("ln_fwd")).launch(@intCast(N), 256, .{ X, gamma, beta, Y, mean, rstd, N, D });
}

/// dGamma and dBeta are accumulated, so they start at zero.
pub fn layernormBwd(k: *gpu.Kernels, X: [*]const bf16, dY: [*]const bf16, gamma: [*]const f32,
                    mean: [*]const f32, rstd: [*]const f32, dX: [*]bf16, dGamma: [*]f32,
                    dBeta: [*]f32, N: i32, D: i32) !void {
    try gpu.zero(dGamma, @as(usize, @intCast(D)) * 4);
    try gpu.zero(dBeta, @as(usize, @intCast(D)) * 4);
    try (try k.get("ln_bwd")).launch(@intCast(N), 256, .{ X, dY, gamma, mean, rstd, dX, dGamma, dBeta, N, D });
}
