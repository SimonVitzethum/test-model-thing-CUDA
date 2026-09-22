//! Mixture of experts: the per-layer state kept from forward to backward and
//! the shared workspace (port of the structures in src/moe.cu).
const std = @import("std");
const gpu = @import("gpu.zig");

pub const bf16 = u16;

/// Kept per layer between forward and backward.
pub const Keep = struct {
    zloss: f32 = 0,
    logits: [*]f32 = undefined, // (N,E)
    probs: [*]f32 = undefined, // (N,E)
    idx: [*]i32 = undefined, // (N,K)
    slotw: [*]f32 = undefined, // (Tk)
    slot_of: [*]i32 = undefined, // (N,K)
    counts: [*]i32 = undefined, // (E) device copy for the router gradient
    perm: [*]i32 = undefined, // (Tk)
    H: [*]bf16 = undefined, // (N,D) the MoE input
    Yg: [*]bf16 = undefined, // (Tk,D) expert pre-activations
    hcnt: [16]i32 = @splat(0), // host: tokens per expert
    hoff: [16]i32 = @splat(0), // host: offsets
    phoff: [16]i32 = @splat(0), // host: offsets including padding
    N: i32 = 0,
    E: i32 = 0,
    K: i32 = 0,
    D: i32 = 0,
    Tk: i32 = 0,
    Ptk: i32 = 0,
    /// Router statistics slot owned by the model, read for all layers at once.
    stat: ?[*]f32 = null,

    pub fn alloc(k: *Keep, mem: *gpu.Memory, N: usize, E: usize, K: usize, D: usize) !void {
        const Tk = N * K + E * 128; // padding slack, 128 slots per expert
        k.logits = try mem.allocT(f32, N * E);
        k.probs = try mem.allocT(f32, N * E);
        k.idx = try mem.allocT(i32, N * K);
        k.slotw = try mem.allocT(f32, Tk);
        k.slot_of = try mem.allocT(i32, N * K);
        k.counts = try mem.allocT(i32, E);
        k.perm = try mem.allocT(i32, Tk);
        k.H = try mem.allocT(bf16, N * D);
        k.Yg = try mem.allocT(bf16, Tk * D);
    }
};

/// Shared transient buffers; the layers run one after the other.
pub const Ws = struct {
    w: [*]f32 = undefined, // (N,K) routing weights
    offsets: [*]i32 = undefined, // (E)
    cursor: [*]i32 = undefined, // (E)
    sum_p: [*]f32 = undefined, // (E+1)
    Xg: [*]bf16 = undefined, // (Tk,D) gathered rows
    dYg: [*]bf16 = undefined,
    s_j: [*]f32 = undefined, // (N,K)
    dlogits: [*]f32 = undefined, // (N,E)
    dlog_b: [*]bf16 = undefined,
    dXg: [*]bf16 = undefined,
    Xf: [*]f32 = undefined, // (N,D) fp32 copy of H for the router weight gradient

    pub fn alloc(w: *Ws, mem: *gpu.Memory, N: usize, E: usize, K: usize, D: usize) !void {
        const Tk = N * K + E * 128;
        w.w = try mem.allocT(f32, N * K);
        w.offsets = try mem.allocT(i32, E);
        w.cursor = try mem.allocT(i32, E);
        w.sum_p = try mem.allocT(f32, E + 1);
        w.Xg = try mem.allocT(bf16, Tk * D);
        w.dYg = try mem.allocT(bf16, Tk * D);
        w.s_j = try mem.allocT(f32, N * K);
        w.dlogits = try mem.allocT(f32, N * E);
        w.dlog_b = try mem.allocT(bf16, N * E);
        w.dXg = try mem.allocT(bf16, Tk * D);
        w.Xf = try mem.allocT(f32, N * D);
    }
};
