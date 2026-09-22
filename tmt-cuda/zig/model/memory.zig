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
