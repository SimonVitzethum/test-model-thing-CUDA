//! Long-range latent cache (MLA): parameter indices, the ring cache, the
//! workspace and what the forward pass keeps for the backward one (port of the
//! structures in src/mla.cu).
const gpu = @import("gpu.zig");
const linalg = @import("linalg.zig");

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

fn blocks(n: i64) u32 {
    return @intCast(@divTrunc(n + 255, 256));
}

// The batched GEMMs below read the row-major matrices as transposed
// column-major ones, which is why the operations and leading dimensions look
// inverted; each comment gives the mathematical form that is computed.

/// S[b,h] (T,Cc) = scale * K[b,h](Cc,k) @ Q[b,h](T,k)^T, one GEMM per (stream, head).
fn scoresGemm(Kc: [*]const bf16, Qc: [*]const bf16, S: [*]f32, B: i32, H: i32, T: i32,
              Cc: i32, dk: i32, beta: f32, scale: f32) !void {
    try linalg.batched(.{ .transa = true, .transb = false, .m = Cc, .n = T, .k = dk,
        .A = Kc, .lda = dk, .strideA = @as(i64, Cc) * dk,
        .B = Qc, .ldb = dk, .strideB = @as(i64, T) * dk,
        .C = S, .ldc = Cc, .strideC = @as(i64, T) * Cc,
        .batch = B * H, .alpha = scale, .beta = beta });
}

/// O[b,h] (T,dh) = P[b,h](T,Cc) @ Vc[b,h](Cc,dh).
fn pvGemm(Pp: [*]const bf16, Vc: [*]const bf16, O: [*]f32, B: i32, H: i32, T: i32,
          Cc: i32, dh: i32, beta: f32) !void {
    try linalg.batched(.{ .transa = false, .transb = false, .m = dh, .n = T, .k = Cc,
        .A = Vc, .lda = dh, .strideA = @as(i64, Cc) * dh,
        .B = Pp, .ldb = Cc, .strideB = @as(i64, T) * Cc,
        .C = O, .ldc = dh, .strideC = @as(i64, T) * dh,
        .batch = B * H, .beta = beta });
}

/// dP[b,h] (T,Cc) = dO[b,h](T,dh) @ Vc[b,h](Cc,dh)^T.
fn dpGemm(dO: [*]const bf16, Vc: [*]const bf16, dP: [*]f32, B: i32, H: i32, T: i32,
          Cc: i32, dh: i32) !void {
    try linalg.batched(.{ .transa = true, .transb = false, .m = Cc, .n = T, .k = dh,
        .A = Vc, .lda = dh, .strideA = @as(i64, Cc) * dh,
        .B = dO, .ldb = dh, .strideB = @as(i64, T) * dh,
        .C = dP, .ldc = Cc, .strideC = @as(i64, T) * Cc,
        .batch = B * H });
}

/// dQ[b,h] (T,k) += dS[b,h](T,Cc) @ K[b,h](Cc,k); the two query halves
/// accumulate over all chunks, hence beta = 1.
fn dqGemm(dS: [*]const bf16, Kc: [*]const bf16, dQ: [*]f32, B: i32, H: i32, T: i32,
          Cc: i32, dk: i32) !void {
    try linalg.batched(.{ .transa = false, .transb = false, .m = dk, .n = T, .k = Cc,
        .A = Kc, .lda = dk, .strideA = @as(i64, Cc) * dk,
        .B = dS, .ldb = Cc, .strideB = @as(i64, T) * Cc,
        .C = dQ, .ldc = dk, .strideC = @as(i64, T) * dk,
        .batch = B * H, .beta = 1 });
}

/// dKc[b,h] (Cc,k) = dS[b,h](T,Cc)^T @ Q[b,h](T,k).
fn dkGemm(dS: [*]const bf16, Qc: [*]const bf16, dKc: [*]f32, B: i32, H: i32, T: i32,
          Cc: i32, dk: i32) !void {
    try linalg.batched(.{ .transa = false, .transb = true, .m = dk, .n = Cc, .k = T,
        .A = Qc, .lda = dk, .strideA = @as(i64, T) * dk,
        .B = dS, .ldb = Cc, .strideB = @as(i64, T) * Cc,
        .C = dKc, .ldc = dk, .strideC = @as(i64, Cc) * dk,
        .batch = B * H });
}

/// dVc[b,h] (Cc,dh) = P[b,h](T,Cc)^T @ dO[b,h](T,dh).
fn dvGemm(Pp: [*]const bf16, dO: [*]const bf16, dVc: [*]f32, B: i32, H: i32, T: i32,
          Cc: i32, dh: i32) !void {
    try linalg.batched(.{ .transa = false, .transb = true, .m = dh, .n = Cc, .k = T,
        .A = dO, .lda = dh, .strideA = @as(i64, T) * dh,
        .B = Pp, .ldb = Cc, .strideB = @as(i64, T) * Cc,
        .C = dVc, .ldc = dh, .strideC = @as(i64, Cc) * dh,
        .batch = B * H });
}

/// dLat(BCc,L) = dK(BCc,H*dh) @ Wu(H*dh,L), accumulating over the two up
/// projections. A single batch takes the place of the plain GEMM of the C++
/// original, because only the batched call writes an fp32 result matrix.
fn dlatGemm(dK: [*]const bf16, Wu: [*]const bf16, dLat: [*]f32, BCc: i32, Hdh: i32,
            L: i32, beta: f32) !void {
    try linalg.batched(.{ .transa = false, .transb = false, .m = L, .n = BCc, .k = Hdh,
        .A = Wu, .lda = L, .strideA = @as(i64, L) * Hdh,
        .B = dK, .ldb = Hdh, .strideB = @as(i64, BCc) * Hdh,
        .C = dLat, .ldc = L, .strideC = @as(i64, L) * BCc,
        .batch = 1, .beta = beta });
}

/// Forward: Xq(N,D) -> Y(N,D), residual-ready, the caller adds it. Writes the
/// window into the ring cache and reads the whole cache in chunks; hpos holds
/// the absolute positions on the device.
pub fn forward(k: *gpu.Kernels, Xq: [*]const bf16, N: i32,
               Wq: [*]const bf16, Wdkv: [*]const bf16, Wkr: [*]const bf16,
               Wuk: [*]const bf16, Wuv: [*]const bf16, Wo: [*]const bf16,
               C: *Cache, Kp: *Keep, W: *Ws, Y: [*]bf16, hpos: [*]const i64,
               B: i32, T: i32, H: i32, dh: i32, L: i32, R: i32, Cc: i32, Cmax: i32,
               theta: f32, scale: f32, D: i32) !void {
    // Element counts are computed in 64 bit, as in the C++ original.
    const Ni: i64 = N;
    const Bi: i64 = B;
    const Ti: i64 = T;
    const Li: i64 = L;
    const Ri: i64 = R;
    const dhi: i64 = dh;
    const Cci: i64 = Cc;
    const BH: i64 = Bi * @as(i64, H);
    const BHT: i64 = BH * Ti;
    if (C.head + Ti > @as(i64, Cmax)) {
        // Only the oldest prefix is evicted, which keeps the retained cache
        // contiguous, so base0 alone tracks the absolute positions.
        const drop: i32 = @intCast(C.head + Ti - @as(i64, Cmax));
        const keep: i32 = @intCast(C.head - drop);
        try (try k.get("compact_cache")).launch(blocks(Bi * Li), 256, .{ C.lat, B, Cmax, L, drop, keep });
        try (try k.get("compact_cache")).launch(blocks(Bi * Ri), 256, .{ C.kr, B, Cmax, R, drop, keep });
        C.base0 += drop;
        C.head = keep;
    }
    const H0win = C.head; // window start in the cache, linear and without wrap
    // The layer input is kept because the weight gradients need it.
    try gpu.copyDevice(Kp.Xq, Xq, @intCast(Ni * @as(i64, D) * 2));
    try linalg.linearFwd(N, H * (dh + R), D, Xq, Wq, W.qf);
    try linalg.linearFwd(N, L, D, Xq, Wdkv, W.latw);
    try linalg.linearFwd(N, R, D, Xq, Wkr, W.krw);
    try (try k.get("qsplit")).launch(blocks(Ni * @as(i64, H) * (dhi + Ri)), 256,
        .{ W.qf, W.qcnh, W.qrnh, N, H, dh, R });
    try (try k.get("rope")).launch(blocks(Ni * @as(i64, H) * Ri), 256,
        .{ W.qrnh, W.qrb, hpos, N, H, R, R, theta, false });
    try (try k.get("rope")).launch(blocks(Ni * Ri), 256,
        .{ W.krw, W.krb, hpos, N, @as(i32, 1), R, R, theta, false });
    // The cache write runs per stream, so B small launches.
    for (0..@intCast(B)) |bi| {
        const b: usize = bi;
        try (try k.get("cache_write")).launch(blocks(Ti * (Li + Ri)), 256, .{
            W.latw + b * @as(usize, @intCast(Ti * Li)), W.krb + b * @as(usize, @intCast(Ti * Ri)),
            C.lat + b * @as(usize, @intCast(@as(i64, Cmax) * Li)), C.kr + b * @as(usize, @intCast(@as(i64, Cmax) * Ri)),
            C.head, T, Cmax, L, R });
    }
    const Clen = C.head + Ti; // cache length after the write
    // Layouts for the batched GEMMs: heads-major (B,H,T,*).
    try (try k.get("to_bhnt")).launch(blocks(BHT * dhi), 256, .{ W.qcnh, W.qcb, B, H, T, dh });
    try (try k.get("to_bhnt")).launch(blocks(BHT * Ri), 256, .{ W.qrb, W.qrb2, B, H, T, R });
    // The accumulators of the online softmax.
    try (try k.get("fill_f32")).launch(blocks(BHT), 256, .{ W.m, @as(f32, -1e30), BHT });
    try gpu.zero(W.l, @intCast(BHT * 4));
    try gpu.zero(W.O, @intCast(BHT * dhi * 4));
    var c0: i64 = 0;
    while (c0 < Clen) : (c0 += Cci) {
        const cce: i32 = @intCast(@min(Clen - c0, Cci));
        const ccei: i64 = cce;
        // cache_gather covers one stream, so again one launch per stream.
        for (0..@intCast(B)) |bi| {
            const b: usize = bi;
            try (try k.get("cache_gather")).launch(blocks(ccei * (Li + Ri)), 256, .{
                C.lat + b * @as(usize, @intCast(@as(i64, Cmax) * Li)), C.kr + b * @as(usize, @intCast(@as(i64, Cmax) * Ri)),
                W.clat + b * @as(usize, @intCast(ccei * Li)), W.ckr + b * @as(usize, @intCast(ccei * Ri)),
                @as(i64, 0), @as(i32, @intCast(c0)), cce, Cmax, L, R });
        }
        // The up projections run flat over all streams: (B*cce,L) -> (B*cce,H*dh).
        try linalg.linearFwd(B * cce, H * dh, L, W.clat, Wuk, W.Kcf);
        try linalg.linearFwd(B * cce, H * dh, L, W.clat, Wuv, W.Vcf);
        try (try k.get("to_bhnt")).launch(blocks(BH * ccei * dhi), 256, .{ W.Kcf, W.Kc, B, H, cce, dh });
        try (try k.get("to_bhnt")).launch(blocks(BH * ccei * dhi), 256, .{ W.Vcf, W.Vc, B, H, cce, dh });
        try (try k.get("rep_hr")).launch(blocks(BH * ccei * Ri), 256, .{ W.ckr, W.krck, B, H, cce, R });
        // Content scores first, then the rotary part added on top; the scale
        // rides along as the GEMM's alpha.
        try scoresGemm(W.Kc, W.qcb, W.S, B, H, T, cce, dh, 0, scale);
        try scoresGemm(W.krck, W.qrb2, W.S, B, H, T, cce, R, 1, scale);
        try (try k.get("causal_mask")).launch(blocks(BHT * ccei), 256,
            .{ W.S, C.base0, @as(i32, @intCast(c0)), C.base0 + H0win, B, H, T, cce });
        // Every matrix of this chunk uses the actual cce as its stride.
        try gpu.zero(W.Pp, @intCast(BHT * ccei * 2));
        const last = c0 + Cci >= Clen;
        try (try k.get("softmax_scale")).launch(@intCast(BHT), 256,
            .{ W.S, W.Pp, W.m, W.l, W.al, Kp.LSE, BHT, cce, last });
        try (try k.get("scale_rows")).launch(blocks(BHT * dhi), 256, .{ W.O, W.al, BHT, dh });
        try pvGemm(W.Pp, W.Vc, W.O, B, H, T, cce, dh, 1);
    }
    try (try k.get("norm_out")).launch(blocks(BHT * dhi), 256, .{ W.O, W.l, BHT, dh });
    try gpu.copyDevice(Kp.qc, W.qcb, @intCast(BHT * dhi * 2));
    try gpu.copyDevice(Kp.qr, W.qrb2, @intCast(BHT * Ri * 2));
    try (try k.get("o_to_flat")).launch(blocks(BHT * dhi), 256, .{ W.O, Kp.Oflat, B, H, T, dh });
    try linalg.linearFwd(N, D, H * dh, Kp.Oflat, Wo, Y);
    C.head = H0win + Ti;
}

/// Backward: dY(N,D) -> dX(N,D); the weight gradients are fp32 and the caller
/// zeroes them. Uses what the forward pass kept (qc, qr, Oflat, LSE). The cache
/// itself stays untouched, since the past carries no gradient; the gradient of
/// the current window flows through the lat/kr linear layers.
pub fn backward(k: *gpu.Kernels, dY: [*]const bf16, N: i32,
                Wq: [*]const bf16, Wdkv: [*]const bf16, Wkr: [*]const bf16,
                Wuk: [*]const bf16, Wuv: [*]const bf16, Wo: [*]const bf16,
                gq: [*]f32, gdkv: [*]f32, gkr: [*]f32, guk: [*]f32, guv: [*]f32, go: [*]f32,
                C: *Cache, Kp: *Keep, W: *Ws, dX: [*]bf16, Xq: [*]const bf16,
                hpos: [*]const i64, B: i32, T: i32, H: i32, dh: i32, L: i32, R: i32,
                Cc: i32, theta: f32, scale: f32, D: i32) !void {
    const Ni: i64 = N;
    const Bi: i64 = B;
    const Ti: i64 = T;
    const Hi: i64 = H;
    const Li: i64 = L;
    const Ri: i64 = R;
    const dhi: i64 = dh;
    const Cci: i64 = Cc;
    const Cmaxi: i64 = C.Cmax;
    const BH: i64 = Bi * Hi;
    const BHT: i64 = BH * Ti;
    try linalg.linearDW(N, D, H * dh, dY, Kp.Oflat, go, 0);
    try linalg.linearDX(N, D, H * dh, dY, Wo, W.dOflat, 0);
    // dO goes to fp32 heads-major and from there to a bf16 copy, because a
    // cuBLAS GEMM may not mix the two input types.
    try (try k.get("do_to_bhnt")).launch(blocks(BHT * dhi), 256, .{ W.dOflat, W.dOf, B, H, T, dh });
    try (try k.get("copy_bf16")).launch(blocks(BHT * dhi), 256, .{ W.dOf, W.dOb, BHT * dhi });
    try gpu.zero(W.dQc, @intCast(BHT * dhi * 4));
    try gpu.zero(W.dQr, @intCast(BHT * Ri * 4));
    try gpu.zero(W.dLat, @intCast(Ni * Li * 4));
    try gpu.zero(W.dKr, @intCast(Ni * Ri * 4));
    const H0win = C.head - Ti; // window start; head already points behind it
    const Clen = C.head;
    var c0: i64 = 0;
    while (c0 < Clen) : (c0 += Cci) {
        const cce: i32 = @intCast(@min(Clen - c0, Cci));
        const ccei: i64 = cce;
        // The chunk data are rebuilt exactly as in the forward pass.
        for (0..@intCast(B)) |bi| {
            const b: usize = bi;
            try (try k.get("cache_gather")).launch(blocks(ccei * (Li + Ri)), 256, .{
                C.lat + b * @as(usize, @intCast(Cmaxi * Li)), C.kr + b * @as(usize, @intCast(Cmaxi * Ri)),
                W.clat + b * @as(usize, @intCast(ccei * Li)), W.ckr + b * @as(usize, @intCast(ccei * Ri)),
                @as(i64, 0), @as(i32, @intCast(c0)), cce, C.Cmax, L, R });
        }
        try linalg.linearFwd(B * cce, H * dh, L, W.clat, Wuk, W.Kcf);
        try linalg.linearFwd(B * cce, H * dh, L, W.clat, Wuv, W.Vcf);
        try (try k.get("to_bhnt")).launch(blocks(BH * ccei * dhi), 256, .{ W.Kcf, W.Kc, B, H, cce, dh });
        try (try k.get("to_bhnt")).launch(blocks(BH * ccei * dhi), 256, .{ W.Vcf, W.Vc, B, H, cce, dh });
        try (try k.get("rep_hr")).launch(blocks(BH * ccei * Ri), 256, .{ W.ckr, W.krck, B, H, cce, R });
        // The scores are recomputed rather than stored.
        try scoresGemm(W.Kc, Kp.qc, W.S, B, H, T, cce, dh, 0, scale);
        try scoresGemm(W.krck, Kp.qr, W.S, B, H, T, cce, R, 1, scale);
        try gpu.zero(W.Pp, @intCast(BHT * ccei * 2));
        try (try k.get("causal_mask")).launch(blocks(BHT * ccei), 256,
            .{ W.S, C.base0, @as(i32, @intCast(c0)), C.base0 + H0win, B, H, T, cce });
        try dpGemm(W.dOb, W.Vc, W.dP, B, H, T, cce, dh);
        // The softmax Jacobian needs the dot product of dO and O over all key
        // chunks, so it is taken from the final output instead of this chunk's
        // probabilities; the kernel rebuilds P from S and the kept LSE.
        try gpu.zero(W.dS, @intCast(BHT * ccei * 2));
        try gpu.zero(W.Pp, @intCast(BHT * ccei * 2));
        try (try k.get("softmax_bwd")).launch(@intCast(BHT), 256,
            .{ W.S, W.dP, Kp.LSE, W.dS, W.Pp, @as(i32, @intCast(BHT)), cce, W.dOf, Kp.Oflat, H, T, dh, scale });
        // The query gradient accumulates over the chunks, content and rotary.
        try dqGemm(W.dS, W.Kc, W.dQc, B, H, T, cce, dh);
        try dqGemm(W.dS, W.krck, W.dQr, B, H, T, cce, R);
        try dkGemm(W.dS, Kp.qc, W.dKc, B, H, T, cce, dh);
        try dvGemm(W.Pp, W.dOb, W.dVc, B, H, T, cce, dh);
        // dKc/dVc (B,H,cce,dh) flattened to bf16 (B*cce,H*dh) feeds both the
        // latent gradient and the chunk-accumulated up-weight gradients.
        const nflat = Bi * ccei * Hi * dhi;
        const upbeta: f32 = if (c0 == 0) 0 else 1;
        try (try k.get("o_to_flat")).launch(blocks(nflat), 256, .{ W.dKc, W.dKVb, B, H, cce, dh });
        try dlatGemm(W.dKVb, Wuk, W.dLatc, B * cce, H * dh, L, 0);
        try linalg.linearDW(B * cce, H * dh, L, W.dKVb, W.clat, guk, upbeta);
        try (try k.get("o_to_flat")).launch(blocks(nflat), 256, .{ W.dVc, W.dKVb, B, H, cce, dh });
        try dlatGemm(W.dKVb, Wuv, W.dLatc, B * cce, H * dh, L, 1);
        try linalg.linearDW(B * cce, H * dh, L, W.dKVb, W.clat, guv, upbeta);
        try (try k.get("masked_scatter")).launch(blocks(Bi * ccei * Li), 256,
            .{ W.dLatc, W.dLat, C.base0, H0win, B, T, cce, L, @as(i32, @intCast(c0)) });
        // The rotary key gradient is summed over the heads and rotated back.
        try dkGemm(W.dS, Kp.qr, W.dKrH, B, H, T, cce, R);
        const n = Bi * ccei * Ri;
        try (try k.get("sum_heads")).launch(blocks(n), 256, .{ W.dKrH, W.dKrc, B, H, cce, R });
        // dLatc serves as the scratch buffer here: it has already been
        // scattered above and is dead.
        try (try k.get("rope_bwd")).launch(blocks(n), 256,
            .{ W.dKrc, W.dLatc, C.base0, @as(i32, @intCast(c0)), B, cce, R, theta });
        try (try k.get("masked_scatter")).launch(blocks(n), 256,
            .{ W.dLatc, W.dKr, C.base0, H0win, B, T, cce, R, @as(i32, @intCast(c0)) });
    }
    // The window gradients flow back through the three input linear layers.
    try (try k.get("dq_join")).launch(blocks(Ni * Hi * (dhi + Ri)), 256,
        .{ W.dQc, W.dQr, W.dQflat, B, H, T, dh, R, hpos, theta });
    try (try k.get("copy_bf16")).launch(blocks(Ni * Li), 256, .{ W.dLat, W.dLatb, Ni * Li });
    try (try k.get("copy_bf16")).launch(blocks(Ni * Ri), 256, .{ W.dKr, W.dKrb, Ni * Ri });
    // dX is the sum of the three paths, accumulated through beta.
    try linalg.linearDX(N, H * (dh + R), D, W.dQflat, Wq, dX, 0);
    try linalg.linearDX(N, L, D, W.dLatb, Wdkv, dX, 1);
    try linalg.linearDX(N, R, D, W.dKrb, Wkr, dX, 1);
    try linalg.linearDW(N, H * (dh + R), D, W.dQflat, Xq, gq, 0);
    try linalg.linearDW(N, L, D, W.dLatb, Xq, gdkv, 0);
    try linalg.linearDW(N, R, D, W.dKrb, Xq, gkr, 0);
}
