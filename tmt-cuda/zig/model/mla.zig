//! Long-range latent cache (MLA): parameter indices, the ring cache, the
//! workspace and what the forward pass keeps for the backward one (port of the
//! structures in src/mla.cu).
const gpu = @import("gpu.zig");

pub const bf16 = u16;

/// Parameter indices of one MLA layer.
pub const P = struct { q: usize = 0, dkv: usize = 0, kr: usize = 0, uk: usize = 0,
                       uv: usize = 0, o: usize = 0, gamma: usize = 0, beta: usize = 0 };

/// The ring cache of one layer, shared by all streams of the batch.
pub const Cache = struct {
    lat: [*]bf16 = undefined, // (B,Cmax,L)
    kr: [*]bf16 = undefined, // (B,Cmax,R)
    head: i64 = 0,
    base0: i64 = 0,
    B: i32 = 0,
    Cmax: i32 = 0,
    L: i32 = 0,
    R: i32 = 0,
};

/// Kept per layer from forward to backward.
pub const Keep = struct {
    qc: [*]bf16 = undefined, // (B,H,T,dh)
    qr: [*]bf16 = undefined, // (B,H,T,R)
    Oflat: [*]bf16 = undefined, // (N,H*dh)
    Xq: [*]bf16 = undefined, // (N,D) layer input
    LSE: [*]f32 = undefined, // (B,H,T)
};

/// One workspace, allocated at the maximum size and reused by every layer.
pub const Ws = struct {
    qf: [*]bf16 = undefined,
    qcnh: [*]bf16 = undefined,
    qrnh: [*]bf16 = undefined,
    qrb: [*]bf16 = undefined,
    latw: [*]bf16 = undefined,
    krw: [*]bf16 = undefined,
    krb: [*]bf16 = undefined,
    qcb: [*]bf16 = undefined,
    qrb2: [*]bf16 = undefined,
    Kc: [*]bf16 = undefined,
    Vc: [*]bf16 = undefined,
    Kcf: [*]bf16 = undefined,
    Vcf: [*]bf16 = undefined,
    krck: [*]bf16 = undefined,
    clat: [*]bf16 = undefined,
    ckr: [*]bf16 = undefined,
    S: [*]f32 = undefined,
    dP: [*]f32 = undefined,
    Pp: [*]bf16 = undefined,
    dS: [*]bf16 = undefined,
    O: [*]f32 = undefined,
    m: [*]f32 = undefined,
    l: [*]f32 = undefined,
    al: [*]f32 = undefined,
    dKc: [*]f32 = undefined,
    dVc: [*]f32 = undefined,
    dKrH: [*]f32 = undefined,
    dQc: [*]f32 = undefined,
    dQr: [*]f32 = undefined,
    dOf: [*]f32 = undefined,
    dOb: [*]bf16 = undefined,
    dLatc: [*]f32 = undefined,
    dKrc: [*]f32 = undefined,
    dLat: [*]f32 = undefined,
    dKr: [*]f32 = undefined,
    dLatb: [*]bf16 = undefined,
    dKrb: [*]bf16 = undefined,
    dOflat: [*]bf16 = undefined,
    dQflat: [*]bf16 = undefined,
    dKVb: [*]bf16 = undefined,

    /// Allocated once at the size the configuration allows.
    pub fn alloc(w: *Ws, mem: *gpu.Memory, B: usize, T: usize, H: usize, dh: usize,
                 L: usize, R: usize, Cc: usize) !void {
        const N = B * T;
        const BH = B * H;
        w.qf = try mem.allocT(bf16, N * H * (dh + R));
        w.qcnh = try mem.allocT(bf16, N * H * dh);
        w.qrnh = try mem.allocT(bf16, N * H * R);
        w.qrb = try mem.allocT(bf16, N * H * R);
        w.latw = try mem.allocT(bf16, N * L);
        w.krw = try mem.allocT(bf16, N * R);
        w.krb = try mem.allocT(bf16, N * R);
        w.qcb = try mem.allocT(bf16, BH * T * dh);
        w.qrb2 = try mem.allocT(bf16, BH * T * R);
        w.Kc = try mem.allocT(bf16, BH * Cc * dh);
        w.Vc = try mem.allocT(bf16, BH * Cc * dh);
        w.Kcf = try mem.allocT(bf16, B * Cc * H * dh);
        w.Vcf = try mem.allocT(bf16, B * Cc * H * dh);
        w.krck = try mem.allocT(bf16, BH * Cc * R);
        w.clat = try mem.allocT(bf16, B * Cc * L);
        w.ckr = try mem.allocT(bf16, B * Cc * R);
        w.S = try mem.allocT(f32, BH * T * Cc);
        w.Pp = try mem.allocT(bf16, BH * T * Cc);
        w.dS = try mem.allocT(bf16, BH * T * Cc);
        w.dP = try mem.allocT(f32, BH * T * Cc);
        w.O = try mem.allocT(f32, BH * T * dh);
        w.m = try mem.allocT(f32, BH * T);
        w.l = try mem.allocT(f32, BH * T);
        w.al = try mem.allocT(f32, BH * T);
        w.dKc = try mem.allocT(f32, BH * Cc * dh);
        w.dVc = try mem.allocT(f32, BH * Cc * dh);
        w.dKrH = try mem.allocT(f32, BH * Cc * R);
        w.dQc = try mem.allocT(f32, BH * T * dh);
        w.dQr = try mem.allocT(f32, BH * T * R);
        w.dOf = try mem.allocT(f32, BH * T * dh);
        w.dOb = try mem.allocT(bf16, BH * T * dh);
        w.dLatc = try mem.allocT(f32, B * Cc * @max(L, R));
        w.dKrc = try mem.allocT(f32, B * Cc * R);
        w.dLat = try mem.allocT(f32, N * L);
        w.dKr = try mem.allocT(f32, N * R);
        w.dLatb = try mem.allocT(bf16, N * L);
        w.dKrb = try mem.allocT(bf16, N * R);
        w.dOflat = try mem.allocT(bf16, N * H * dh);
        w.dQflat = try mem.allocT(bf16, N * H * (dh + R));
        w.dKVb = try mem.allocT(bf16, B * Cc * H * dh);
    }
};
