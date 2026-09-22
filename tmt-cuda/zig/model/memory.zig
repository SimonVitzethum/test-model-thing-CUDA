//! Knowledge-graph fact memory: per-layer buffers and the shared encoding
//! (port of the structures in src/memory.cu).
const gpu = @import("gpu.zig");

pub const bf16 = u16;

pub const Layer = struct {
    use: bool = false,
    gamma: usize = 0,
    beta: usize = 0,
    wq: usize = 0,
    wk: usize = 0,
    wv: usize = 0,
    wo: usize = 0,
    Xsnap: [*]bf16 = undefined,
    Hn: [*]bf16 = undefined,
    Q: [*]bf16 = undefined,
    K: [*]bf16 = undefined,
    V: [*]bf16 = undefined,
    O: [*]bf16 = undefined,
    mean: [*]f32 = undefined,
    rstd: [*]f32 = undefined,
    P: [*]f32 = undefined, // (B,H,T,M) attention probabilities
};

pub const Shared = struct {
    eprev: usize = 0, // parameter indices
    pos: usize = 0,
    decay: usize = 0,
    gate: usize = 0,
    rq: usize = 0, // stage-2 retrieval heads
    rk: usize = 0,
    ids: [*]i32 = undefined, // (B,M) fact bytes, -1 is padding
    enc0: [*]bf16 = undefined, // byte/previous-byte/position encoding
    state: [*]f32 = undefined, // recurrence over the fact bytes
    dEnc0: [*]f32 = undefined,
    enc: [*]bf16 = undefined, // e + s
    dO: [*]bf16 = undefined,
    dQ: [*]bf16 = undefined,
    dK: [*]bf16 = undefined,
    dV: [*]bf16 = undefined,
    dHn: [*]bf16 = undefined,
    dXln: [*]bf16 = undefined,
    dEncB: [*]bf16 = undefined,
    dS: [*]f32 = undefined,
    dEnc: [*]f32 = undefined,
};

const std = @import("std");
const linalg = @import("linalg.zig");
const ops = @import("ops.zig");

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

/// Fact bytes to their encoding: the byte/previous-byte/position sum plus the
/// causal recurrence over the memory.
pub fn encode(k: *gpu.Kernels, E: [*]const bf16, Eprev: [*]const bf16, P: [*]const bf16,
              decay: [*]const f32, gate: [*]const f32, s: *Shared, B: i32, M: i32, D: i32) !void {
    const n: usize = @intCast(@as(i64, B) * M * D);
    try (try k.get("mem_encode")).launch(blocks(n), 256, .{ E, Eprev, P, s.ids, s.enc0, B, M, D });
    try ops.stateForward(k, s.enc0, s.state, decay, 0, B, M, D, @intFromPtr(gate), .{});
    try (try k.get("mem_sum")).launch(blocks(n), 256, .{ s.enc0, s.state, s.enc, @as(i64, @intCast(n)) });
}

/// x is updated in place: x += o Wo^T.
pub fn forward(k: *gpu.Kernels, X: [*]bf16, L: *Layer, s: *Shared, gamma: [*]const f32, beta: [*]const f32,
               Wq: [*]const bf16, Wk: [*]const bf16, Wv: [*]const bf16, Wo: [*]const bf16, Y: [*]bf16,
               B: i32, T: i32, M: i32, H: i32, dh: i32, D: i32) !void {
    const N = B * T;
    const HD = H * dh;
    const ND: usize = @intCast(@as(i64, N) * D);
    try gpu.copyDevice(L.Xsnap, X, ND * 2);
    try ops.layernormFwd(k, X, gamma, beta, L.Hn, L.mean, L.rstd, N, D);
    try linalg.linearFwd(N, HD, D, L.Hn, Wq, L.Q);
    try linalg.linearFwd(B * M, HD, D, s.enc, Wk, L.K);
    try linalg.linearFwd(B * M, HD, D, s.enc, Wv, L.V);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));
    try (try k.get("mem_attn_fwd")).launch(blocks(@intCast(@as(i64, B) * H * T)), 256,
        .{ L.Q, L.K, L.V, s.ids, L.P, L.O, B, T, M, H, dh, scale });
    try linalg.linearFwd(N, D, HD, L.O, Wo, Y);
    try (try k.get("mem_add_bf16")).launch(blocks(ND), 256, .{ X, Y, @as(i64, @intCast(ND)) });
}

/// dX holds dL/dx_out and becomes dL/dx_in; the residual stays in it.
pub fn backward(k: *gpu.Kernels, dX: [*]bf16, L: *Layer, s: *Shared, gamma: [*]const f32,
                Wq: [*]const bf16, Wk: [*]const bf16, Wv: [*]const bf16, Wo: [*]const bf16,
                gGamma: [*]f32, gBeta: [*]f32, gWq: [*]f32, gWk: [*]f32, gWv: [*]f32, gWo: [*]f32,
                B: i32, T: i32, M: i32, H: i32, dh: i32, D: i32) !void {
    const N = B * T;
    const HD = H * dh;
    const ND: usize = @intCast(@as(i64, N) * D);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));
    try linalg.linearDW(N, D, HD, dX, L.O, gWo, 0);
    try linalg.linearDX(N, D, HD, dX, Wo, s.dO, 0);
    try (try k.get("mem_attn_bwd_q")).launch(blocks(@intCast(@as(i64, B) * H * T)), 256,
        .{ s.dO, L.K, L.V, L.P, s.dS, s.dQ, B, T, M, H, dh, scale });
    try (try k.get("mem_attn_bwd_kv")).launch(blocks(@intCast(@as(i64, B) * H * M)), 256,
        .{ L.Q, s.dO, L.P, s.dS, s.dK, s.dV, B, T, M, H, dh, scale });
    try linalg.linearDW(N, HD, D, s.dQ, L.Hn, gWq, 0);
    try linalg.linearDX(N, HD, D, s.dQ, Wq, s.dHn, 0);
    try linalg.linearDW(B * M, HD, D, s.dK, s.enc, gWk, 0);
    try linalg.linearDW(B * M, HD, D, s.dV, s.enc, gWv, 0);
    try linalg.linearDX(B * M, HD, D, s.dK, Wk, s.dEncB, 0);
    try linalg.linearDX(B * M, HD, D, s.dV, Wv, s.dEncB, 1);
    const bmd: usize = @intCast(@as(i64, B) * M * D);
    try (try k.get("cast_add")).launch(blocks(bmd), 256, .{ s.dEncB, s.dEnc, @as(i64, @intCast(bmd)) });
    try ops.layernormBwd(k, L.Xsnap, s.dHn, gamma, L.mean, L.rstd, s.dXln, gGamma, gBeta, N, D);
    try (try k.get("mem_add_bf16")).launch(blocks(ND), 256, .{ dX, s.dXln, @as(i64, @intCast(ND)) });
}

/// Back through the recurrence of the encoding and into E, Eprev and P.
pub fn encodeBackward(k: *gpu.Kernels, s: *Shared, decay: [*]const f32, gate: [*]const f32,
                      dDecay: [*]f32, dGate: [*]f32, dE: [*]f32, dEprev: [*]f32, dP: [*]f32,
                      B: i32, M: i32, D: i32) !void {
    const n: usize = @intCast(@as(i64, B) * M * D);
    try (try k.get("copy_bf16")).launch(blocks(n), 256, .{ s.dEnc, s.dEncB, @as(i64, @intCast(n)) });
    try ops.cellBackward(k, s.dEncB, s.state, decay, s.dEnc0, dDecay, B, M, D, s.enc0, 0,
        @intFromPtr(gate), @intFromPtr(dGate), .{});
    try (try k.get("add_f32")).launch(blocks(n), 256, .{ s.dEnc, s.dEnc0, @as(i64, @intCast(n)) });
    try (try k.get("mem_encode_bwd")).launch(blocks(n), 256, .{ s.dEnc, s.ids, dE, dEprev, dP, B, M, D });
}
