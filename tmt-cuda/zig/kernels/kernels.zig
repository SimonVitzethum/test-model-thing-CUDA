//! The CUDA kernels in Zig. Compiled to PTX (see build.zig) and launched from
//! the host through the driver API; each kernel is a line-by-line port of its
//! counterpart in src/*.cu and must produce bit-identical results.
const cuda = @import("cuda.zig");
const bf16 = cuda.bf16;

// ---- src/emb.cu ----
/// Embedding gather: one row of the table per position, zero for ignored
/// targets. One thread per element rather than per row, so that neighbouring
/// threads read and write neighbouring addresses.
export fn emb_gather(W: cuda.ConstGlobal(bf16), ids: cuda.ConstGlobal(i32), out: cuda.Global(bf16),
                     N: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, N) * D) return;
    const d = @rem(i, D);
    const row = @divTrunc(i, D);
    const id = ids[@intCast(row)];
    out[@intCast(i)] = if (id < 0) cuda.f2bf(0) else W[@intCast(@as(i64, id) * D + d)];
}

/// Embedding scatter: accumulate the fp32 gradients into the table.
export fn emb_scatter(dOut: cuda.ConstGlobal(f32), ids: cuda.ConstGlobal(i32), dW: cuda.Global(f32),
                      N: i32, D: i32) callconv(.nvptx_kernel) void {
    const i = cuda.globalIdX();
    if (i >= @as(u32, @bitCast(N))) return;
    const d = dW + @as(usize, @intCast(ids[i])) * @as(usize, @intCast(D));
    const s = dOut + @as(usize, i) * @as(usize, @intCast(D));
    for (0..@intCast(D)) |dd| cuda.atomicAddF32(&d[dd], s[dd]);
}

// ---- src/model.cu helpers ----
/// acc += x, elementwise (add_f32_kernel).
export fn add_f32(acc: cuda.Global(f32), x: cuda.ConstGlobal(f32), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) acc[@intCast(i)] += x[@intCast(i)];
}

/// master -> bf16 working copy (copy_bf16_kernel in model.cu, to_bf16_kernel
/// in moe.cu: the same function, so one kernel serves both).
export fn copy_bf16(src: cuda.ConstGlobal(f32), dst: cuda.Global(bf16), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) dst[@intCast(i)] = cuda.f2bf(src[@intCast(i)]);
}

// ---- src/norm.cu ----
// One block per row; the block reduces twice (sum, then sum of squares).
var ln_buf: [256]f32 addrspace(.shared) = undefined;
var ln_m: f32 addrspace(.shared) = undefined;
var ln_rs: f32 addrspace(.shared) = undefined;

/// Sum of `v` over the block, left in ln_buf[0] (every thread must call it).
fn blockSum(v: f32) f32 {
    const t = cuda.threadIdxX();
    ln_buf[t] = v;
    cuda.syncThreads();
    var s = cuda.blockDimX() / 2;
    while (s > 0) : (s >>= 1) {
        if (t < s) ln_buf[t] += ln_buf[t + s];
        cuda.syncThreads();
    }
    return ln_buf[0];
}

export fn ln_fwd(X: cuda.ConstGlobal(bf16), gamma: cuda.ConstGlobal(f32), beta: cuda.ConstGlobal(f32),
                 Y: cuda.Global(bf16), mean: cuda.Global(f32), rstd: cuda.Global(f32),
                 N: i32, D: i32) callconv(.nvptx_kernel) void {
    const r = cuda.blockIdxX();
    if (r >= @as(u32, @bitCast(N))) return;
    const row = @as(usize, r) * @as(usize, @intCast(D));
    const t = cuda.threadIdxX();
    const step = cuda.blockDimX();
    var sum: f32 = 0;
    var sq: f32 = 0;
    var d = t;
    while (d < @as(u32, @bitCast(D))) : (d += step) {
        const v = cuda.bf2f(X[row + d]);
        sum += v;
        sq += v * v;
    }
    const total = blockSum(sum);
    if (t == 0) ln_m = cuda.fdiv(total, @floatFromInt(D));
    cuda.syncThreads();
    const total_sq = blockSum(sq);
    if (t == 0) {
        const v = cuda.fdiv(total_sq, @floatFromInt(D)) - cuda.mulNoFma(ln_m, ln_m);
        mean[r] = ln_m;
        ln_rs = cuda.frsqrt(v + 1e-5);
        rstd[r] = ln_rs;
    }
    cuda.syncThreads();
    const mm = ln_m;
    const rr = ln_rs;
    d = t;
    while (d < @as(u32, @bitCast(D))) : (d += step) {
        const xh = (cuda.bf2f(X[row + d]) - mm) * rr;
        Y[row + d] = cuda.f2bf(xh * gamma[d] + beta[d]);
    }
}

var ln_t1: f32 addrspace(.shared) = undefined;

export fn ln_bwd(X: cuda.ConstGlobal(bf16), dY: cuda.ConstGlobal(bf16), gamma: cuda.ConstGlobal(f32),
                 mean: cuda.ConstGlobal(f32), rstd: cuda.ConstGlobal(f32), dX: cuda.Global(bf16),
                 dGamma: cuda.Global(f32), dBeta: cuda.Global(f32), N: i32, D: i32) callconv(.nvptx_kernel) void {
    const r = cuda.blockIdxX();
    if (r >= @as(u32, @bitCast(N))) return;
    const row = @as(usize, r) * @as(usize, @intCast(D));
    const t = cuda.threadIdxX();
    const step = cuda.blockDimX();
    const m = mean[r];
    const rr = rstd[r];
    var s1: f32 = 0;
    var s2: f32 = 0;
    var d = t;
    while (d < @as(u32, @bitCast(D))) : (d += step) {
        const dy = cuda.bf2f(dY[row + d]) * gamma[d];
        const xh = (cuda.bf2f(X[row + d]) - m) * rr;
        s1 += dy;
        s2 += dy * xh;
        cuda.atomicAddF32(&dBeta[d], cuda.bf2f(dY[row + d]));
        cuda.atomicAddF32(&dGamma[d], cuda.bf2f(dY[row + d]) * xh);
    }
    const r1 = blockSum(s1);
    if (t == 0) ln_t1 = r1;
    cuda.syncThreads();
    const r2 = blockSum(s2);
    const a = cuda.fdiv(ln_t1, @floatFromInt(D));
    const b = cuda.fdiv(r2, @floatFromInt(D));
    d = t;
    while (d < @as(u32, @bitCast(D))) : (d += step) {
        const dy = cuda.bf2f(dY[row + d]) * gamma[d];
        const xh = (cuda.bf2f(X[row + d]) - m) * rr;
        dX[row + d] = cuda.f2bf(rr * (dy - a - xh * b));
    }
}

// ---- src/loss.cu ----
/// Cross entropy over 256 bytes: max-subtract, exp-sum, -logit_t + logsum.
/// Keeps the fp32 probabilities for the backward pass. tgt < 0 is ignored.
export fn ce_fwd(logits: cuda.ConstGlobal(bf16), tgt: cuda.ConstGlobal(i32), probs: cuda.Global(f32),
                 loss_out: cuda.Global(f32), N: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const L = logits + @as(usize, r) * 256;
    var mx = cuda.bf2f(L[0]);
    for (1..256) |i| mx = cuda.__nv_fmaxf(mx, cuda.bf2f(L[i]));
    var se: f32 = 0;
    for (0..256) |i| se += cuda.__nv_fast_expf(cuda.bf2f(L[i]) - mx);
    const lse = @mulAdd(f32, cuda.lg2(se), cuda.ln2, mx);
    for (0..256) |i| probs[@as(usize, r) * 256 + i] = cuda.fdiv(cuda.__nv_fast_expf(cuda.bf2f(L[i]) - mx), se);
    loss_out[r] = if (tgt[r] < 0) 0 else lse - cuda.bf2f(L[@intCast(tgt[r])]);
}

/// dLogits = w*(p - onehot)/N
export fn ce_bwd(probs: cuda.ConstGlobal(f32), tgt: cuda.ConstGlobal(i32), dLogits: cuda.Global(bf16),
                 w: f32, N: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    for (0..256) |i| {
        const p = probs[@as(usize, r) * 256 + i] - @as(f32, if (@as(i32, @intCast(i)) == tgt[r]) 1 else 0);
        dLogits[@as(usize, r) * 256 + i] = cuda.f2bf(if (tgt[r] < 0) 0 else cuda.fdiv(w * p, @floatFromInt(N)));
    }
}

/// Stop head: BCE with pos_weight, in the numerically stable form.
export fn stop_fwd(s: cuda.ConstGlobal(bf16), end: cuda.ConstGlobal(i32), loss_out: cuda.Global(f32),
                   pos_w: f32, N: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const z = cuda.bf2f(s[r]);
    const t: f32 = if (end[r] != 0) 1 else 0;
    const softplus = cuda.__nv_fmaxf(-z, 0) + cuda.__nv_log1pf(cuda.__nv_fast_expf(-cuda.__nv_fabsf(z)));
    loss_out[r] = if (t > 0.5) pos_w * softplus else z + softplus;
}

export fn stop_bwd(s: cuda.ConstGlobal(bf16), end: cuda.ConstGlobal(i32), ds: cuda.Global(bf16),
                   pos_w: f32, w: f32, N: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const z = cuda.bf2f(s[r]);
    const t: f32 = if (end[r] != 0) 1 else 0;
    var g = cuda.sigmoid(z) - t;
    if (t > 0.5) g *= pos_w;
    ds[r] = cuda.f2bf(cuda.fdiv(w * g, @floatFromInt(N)));
}

/// Latent MSE share of dX: dX += 2*(x-tgt)/M * w
export fn latent_bwd(x: cuda.ConstGlobal(f32), tgt: cuda.ConstGlobal(f32), dX: cuda.Global(f32),
                     w: f32, M: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= M) return;
    const u: usize = @intCast(i);
    dX[u] += cuda.fdiv(2.0 * (x[u] - tgt[u]), @floatFromInt(M)) * w;
}

/// Variance hinge: dX += scl * 2*(x-mean)/M
export fn var_bwd(x: cuda.ConstGlobal(f32), dX: cuda.Global(f32), mean: f32, scl: f32,
                  M: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= M) return;
    const u: usize = @intCast(i);
    dX[u] += cuda.fdiv(scl * 2.0 * (x[u] - mean), @floatFromInt(M));
}

// ---- src/adam.cu ----
/// bf16 gradient -> fp32 accumulation (cast_add_kernel).
export fn cast_add(s: cuda.ConstGlobal(bf16), d: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) d[@intCast(i)] += cuda.bf2f(s[@intCast(i)]);
}

// ---- src/cell.cu ----
/// Options of the recurrence kernels, same layout as the C++ CellOpt.
pub const CellOpt = extern struct {
    ids: ?cuda.ConstGlobal(i32) = null, // (B,T) input bytes, for the document reset
    docsep: i32 = -1,
    gamma: f32 = 1, // trace decay per byte (1: exact traces)
    trDec: ?cuda.Global(f32) = null,
    trGate: ?cuda.Global(f32) = null,
    lam: ?cuda.Global(f32) = null,
    prod: ?cuda.Global(f32) = null,
    logDec: ?cuda.Global(f32) = null,
    logGate: ?cuda.Global(f32) = null,
    logEmb: ?cuda.Global(f32) = null,
    /// (B,T) 0 marks a padded step, which leaves the state untouched.
    mask: ?cuda.ConstGlobal(i32) = null,
    /// (B,T) bytes covered by this step; the decay per step becomes a^len, so
    /// a half-life stays a half-life in bytes when a layer runs per patch.
    plen: ?cuda.ConstGlobal(i32) = null,
};

/// One step of the recurrence: the decay that multiplies the state, the
/// sigmoid it came from, and how many bytes the step covers. The backward
/// derives the local term from these, in the association the C++ kernel uses,
/// so a model without patching computes the same bits as before.
const Step = struct { eff: f32, a: f32, len: f32 = 1, scaled: bool = false };

inline fn cellStep(o: CellOpt, decay: f32, gate: ?cuda.ConstGlobal(f32), d: u32, x: f32,
                   b: u32, T: i32, t: i32) Step {
    const at: usize = @intCast(@as(i32, @intCast(b)) * T + t);
    if (o.mask) |m| if (m[at] == 0) return .{ .eff = 1, .a = 1 };
    if (o.docsep >= 0 and o.ids.?[at] == o.docsep) return .{ .eff = 0, .a = 0 };
    const a = cuda.sigmoid(decay + if (gate) |g| g[d] * x else 0);
    if (o.plen) |lens| {
        const len: f32 = @floatFromInt(lens[at]);
        return .{ .eff = cuda.__nv_fast_powf(a, len), .a = a, .len = len, .scaled = true };
    }
    return .{ .eff = a, .a = a };
}

/// d(decay of the step) / d(pre-activation), times (prev - x).
inline fn cellLocal(st: Step, prev: f32, x: f32, masked: bool) f32 {
    if (masked) return 0;
    if (st.scaled) return (prev - x) * (st.len * st.eff * (1.0 - st.a));
    return (prev - x) * st.a * (1.0 - st.a);
}

/// Pass 1: the time loop, one thread per (stream, channel).
export fn state_pass(X: cuda.ConstGlobal(bf16), S: cuda.Global(f32), decay: cuda.ConstGlobal(f32),
                     carry: ?cuda.ConstGlobal(f32), B: i32, T: i32, D: i32,
                     gate: ?cuda.ConstGlobal(f32), o: CellOpt) callconv(.nvptx_kernel) void {
    const b = cuda.blockIdxX();
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (b >= @as(u32, @bitCast(B)) or d >= @as(u32, @bitCast(D))) return;
    var state: f32 = if (carry) |c| c[b * @as(u32, @bitCast(D)) + d] else 0;
    var t: i32 = 0;
    while (t < T) : (t += 1) {
        const idx: u32 = @bitCast((@as(i32, @intCast(b)) * T + t) * D + @as(i32, @intCast(d)));
        const x = cuda.bf2f(X[idx]);
        const dec = cellStep(o, decay[d], gate, d, x, b, T, t).eff;
        state = dec * state + (1.0 - dec) * x;
        S[idx] = state;
    }
}

/// Pass 2: one block per (stream, position) reduces mean and rstd over D.
export fn stats_pass(S: cuda.ConstGlobal(f32), mean: cuda.Global(f32), rstd: cuda.Global(f32),
                     B: i32, T: i32, D: i32) callconv(.nvptx_kernel) void {
    const row = cuda.blockIdxX();
    if (row >= @as(u32, @bitCast(B * T))) return;
    const base = row * @as(u32, @bitCast(D));
    const t = cuda.threadIdxX();
    const step = cuda.blockDimX();
    var sum: f32 = 0;
    var sq: f32 = 0;
    var d = t;
    while (d < @as(u32, @bitCast(D))) : (d += step) {
        const v = S[base + d];
        sum += v;
        sq += v * v;
    }
    const total = blockSum(sum);
    if (t == 0) ln_m = total;
    cuda.syncThreads();
    const total_sq = blockSum(sq);
    if (t == 0) {
        const m = cuda.fdiv(ln_m, @floatFromInt(D));
        const v = cuda.fdiv(total_sq, @floatFromInt(D)) - cuda.mulNoFma(m, m);
        mean[row] = m;
        rstd[row] = cuda.frsqrt(v + 1e-5);
    }
}

/// Pass 3: normalize, SiLU, residual.
export fn out_pass(X: cuda.ConstGlobal(bf16), S: cuda.ConstGlobal(f32), mean: cuda.ConstGlobal(f32),
                   rstd: cuda.ConstGlobal(f32), Y: cuda.Global(bf16), B: i32, T: i32, D: i32) callconv(.nvptx_kernel) void {
    const b = cuda.blockIdxX();
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (b >= @as(u32, @bitCast(B)) or d >= @as(u32, @bitCast(D))) return;
    var t: i32 = 0;
    while (t < T) : (t += 1) {
        const row: u32 = @bitCast(@as(i32, @intCast(b)) * T + t);
        const idx: u32 = @bitCast(@as(i32, @bitCast(row)) * D + @as(i32, @intCast(d)));
        const h = (S[idx] - mean[row]) * rstd[row];
        Y[idx] = cuda.f2bf(cuda.silu(h) + cuda.bf2f(X[idx]));
    }
}

/// Exact within-window derivative, plus the hybrid traces across the window
/// boundary (see the comment in src/cell.cu).
export fn state_bwd(dS: cuda.ConstGlobal(bf16), S: cuda.ConstGlobal(f32), decay: cuda.ConstGlobal(f32),
                    dX: cuda.Global(f32), dDec: cuda.Global(f32), B: i32, T: i32, D: i32,
                    X: cuda.ConstGlobal(bf16), initial: ?cuda.ConstGlobal(f32),
                    gate: ?cuda.ConstGlobal(f32), dGate: ?cuda.Global(f32), o: CellOpt) callconv(.nvptx_kernel) void {
    const b = cuda.blockIdxX();
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (b >= @as(u32, @bitCast(B)) or d >= @as(u32, @bitCast(D))) return;
    var future: f32 = 0;
    var gd: f32 = 0;
    var gg: f32 = 0;
    var suffix: f32 = 1;
    var ld: f32 = 0;
    var lg: f32 = 0;
    var t: i32 = T - 1;
    while (t >= 0) : (t -= 1) {
        const idx: usize = @intCast((@as(i64, b) * T + t) * D + @as(i64, d));
        const x = cuda.bf2f(X[idx]);
        const st = cellStep(o, decay[d], gate, d, x, b, T, t);
        const a = st.eff;
        const total = cuda.bf2f(dS[idx]) + future;
        const prev = if (t != 0) S[idx - @as(usize, @intCast(D))] else if (initial) |i| i[@as(usize, b) * @as(usize, @intCast(D)) + d] else 0;
        const local = cellLocal(st, prev, x, a == 1 and o.mask != null);
        const gz = total * local;
        dX[idx] = total * (1.0 - a) + if (gate) |g| gz * g[d] else 0;
        gd += gz;
        gg += gz * x;
        future = total * a;
        ld += cuda.mulNoFma(suffix, local);
        lg += suffix * local * x;
        suffix *= o.gamma * a;
    }
    const bd: usize = @as(usize, b) * @as(usize, @intCast(D)) + d;
    if (o.trDec) |tr| {
        const e = tr[bd];
        const part = future * e;
        gd += part;
        if (o.logDec) |l| cuda.atomicAddF32(&l[d], part);
        tr[bd] = suffix * e + ld;
    }
    if (o.trGate) |tr| if (gate != null) {
        const e = tr[bd];
        const part = future * e;
        gg += part;
        if (o.logGate) |l| cuda.atomicAddF32(&l[d], part);
        tr[bd] = suffix * e + lg;
    };
    if (o.lam) |lam| {
        lam[bd] = future;
        o.prod.?[bd] = suffix;
    }
    cuda.atomicAddF32(&dDec[d], gd);
    if (dGate) |dg| if (gate != null) cuda.atomicAddF32(&dg[d], gg);
}

/// Embedding trace e[b][r][d] = ds_d/dEmb[r][d], using lambda and P from state_bwd.
export fn emb_trace(X: cuda.ConstGlobal(bf16), S: cuda.ConstGlobal(f32), initial: cuda.ConstGlobal(f32),
                    decay: cuda.ConstGlobal(f32), gate: ?cuda.ConstGlobal(f32), trEmb: cuda.Global(f32),
                    dEmb: cuda.Global(f32), B: i32, T: i32, D: i32, o: CellOpt) callconv(.nvptx_kernel) void {
    const b = cuda.blockIdxX();
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (b >= @as(u32, @bitCast(B)) or d >= @as(u32, @bitCast(D))) return;
    const dim: usize = @intCast(D);
    const bd: usize = @as(usize, b) * dim + d;
    const l = o.lam.?[bd];
    const P = o.prod.?[bd];
    const e = trEmb + @as(usize, b) * 256 * dim + d;
    for (0..256) |r| {
        const v = e[r * dim];
        if (v != 0) {
            cuda.atomicAddF32(&dEmb[r * dim + d], l * v);
            if (o.logEmb) |le| cuda.atomicAddF32(&le[r * dim + d], l * v);
        }
        e[r * dim] = P * v;
    }
    var suffix: f32 = 1;
    var t: i32 = T - 1;
    while (t >= 0) : (t -= 1) {
        const idx: usize = @intCast((@as(i64, b) * T + t) * D + @as(i64, d));
        const x = cuda.bf2f(X[idx]);
        const st = cellStep(o, decay[d], gate, d, x, b, T, t);
        const a = st.eff;
        const prev = if (t != 0) S[idx - dim] else initial[bd];
        const k = if (gate) |g| @mulAdd(f32, cellLocal(st, prev, x, false), g[d], 1.0 - a) else 1.0 - a;
        e[@as(usize, @intCast(o.ids.?[@intCast(@as(i32, @intCast(b)) * T + t)])) * dim] += suffix * k;
        suffix *= o.gamma * a;
    }
}

// ---- src/adam.cu (multi-tensor optimizer) ----
/// One chunk of at most MT_CHUNK elements of one parameter.
pub const MTChunk = extern struct { param: i32, len: i32, start: i64 };
/// Device arrays, one entry per parameter; same layout as the C++ MTParams.
pub const MTParams = extern struct {
    master: cuda.Global(cuda.Global(f32)),
    m: cuda.Global(cuda.Global(f32)),
    v: cuda.Global(cuda.Global(f32)),
    grad: cuda.Global(cuda.Global(f32)),
    work: cuda.Global(cuda.Global(bf16)),
    /// Bit 0: in the gradient norm. Bit 1: updated by AdamW. Bit 2: Muon owns
    /// it. Bit 3: its moments are bf16. Bit 4: its master copy is bf16.
    flags: cuda.Global(u8),
    /// Per-parameter factor on the learning rate; 1 unless mup scales it.
    lrmul: cuda.Global(f32),
    /// Running average of the weights over training; null unless wavg is on.
    avg: cuda.Global(cuda.Global(f32)),
    chunks: cuda.Global(MTChunk),
    nchunks: i32,
    sumsq: cuda.Global(f64), // global squared gradient norm
    /// Counter for the stochastic rounding; the step number, so a rerun of
    /// the same configuration rounds the same way. Which arrays are bf16 is
    /// per parameter and lives in `flags`.
    seed: u32 = 0,
};

/// The optimizer state arrays are addressed through the same pointers whether
/// they hold fp32 or bf16; only the element width changes. The flag is
/// uniform over the whole grid, so the branch is free.
inline fn optLoad(p: cuda.Global(f32), i: usize, half: bool) f32 {
    if (!half) return p[i];
    const q: cuda.Global(bf16) = @ptrCast(p);
    return cuda.bf2f(q[i]);
}
inline fn optStore(p: cuda.Global(f32), i: usize, half: bool, x: f32, r: u32) void {
    if (!half) {
        p[i] = x;
        return;
    }
    const q: cuda.Global(bf16) = @ptrCast(p);
    q[i] = cuda.f2bfSr(x, r);
}

var mt_buf: [256]f64 addrspace(.shared) = undefined;

export fn mt_sumsq(P: MTParams) callconv(.nvptx_kernel) void {
    const c = P.chunks[cuda.blockIdxX()];
    const t = cuda.threadIdxX();
    var s: f64 = 0;
    if (P.flags[@intCast(c.param)] & 1 != 0) {
        const g = P.grad[@intCast(c.param)] + @as(usize, @intCast(c.start));
        var i = t;
        while (i < @as(u32, @bitCast(c.len))) : (i += cuda.blockDimX()) s += @as(f64, g[i]) * @as(f64, g[i]);
    }
    mt_buf[t] = s;
    cuda.syncThreads();
    var half = cuda.blockDimX() / 2;
    while (half > 0) : (half >>= 1) {
        if (t < half) mt_buf[t] += mt_buf[t + half];
        cuda.syncThreads();
    }
    if (t == 0 and mt_buf[0] != 0) _ = @atomicRmw(f64, &P.sumsq[0], .Add, mt_buf[0], .monotonic);
}

/// Running average of the weights over training (LAWA), kept in fp32 beside
/// the master copy. It is only ever read for evaluation, so it never feeds
/// back into the step, and it runs every `wavg_every` steps rather than every
/// step because it moves 12 bytes per parameter in an optimizer that is
/// already at the bandwidth ceiling.
export fn mt_wavg(P: MTParams, tau: f32) callconv(.nvptx_kernel) void {
    const c = P.chunks[cuda.blockIdxX()];
    if (P.flags[@intCast(c.param)] & 6 == 0) return; // neither AdamW nor Muon
    // The bf16 arrays are addressed through an fp32 pointer, so the element
    // offset has to be applied after the cast, not before it: the index goes
    // into the accessor whole.
    const at: usize = @intCast(c.start);
    const master = P.master[@intCast(c.param)];
    const avg = P.avg[@intCast(c.param)];
    const hw = P.flags[@intCast(c.param)] & 16 != 0; // the average follows the master
    var i = cuda.threadIdxX();
    while (i < @as(u32, @bitCast(c.len))) : (i += cuda.blockDimX()) {
        const j = at + i;
        const a = tau * optLoad(avg, j, hw) + (1 - tau) * optLoad(master, j, hw);
        optStore(avg, j, hw, a, cuda.mix32(@intCast(j), P.seed ^ @as(u32, @bitCast(c.param)) *% 0x2545f491));
    }
}

/// AdamW with global-norm clipping computed on the device. A non-finite norm
/// skips the update entirely (the host then refuses the step).
export fn mt_adam(P: MTParams, clip: f32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32,
                  bc1: f32, bc2: f32) callconv(.nvptx_kernel) void {
    const c = P.chunks[cuda.blockIdxX()];
    // Chunks this kernel does not update (the frozen target, an unused stop
    // head, the weights Muon owns) still need their gradient cleared, which
    // used to be a separate pass over every parameter.
    if (P.flags[@intCast(c.param)] & 2 == 0) {
        if (P.flags[@intCast(c.param)] & 4 == 0) {
            const g = P.grad[@intCast(c.param)] + @as(usize, @intCast(c.start));
            var j = cuda.threadIdxX();
            while (j < @as(u32, @bitCast(c.len))) : (j += cuda.blockDimX()) g[j] = 0;
        }
        return;
    }
    const s = P.sumsq[0];
    if (!(s - s == 0)) return; // non-finite
    const scale: f32 = if (clip > 0) cuda.__nv_fminf(1, cuda.fdiv(clip, cuda.__nv_fmaxf(@floatCast(@sqrt(s)), 1e-12))) else 1;
    // The bf16 state is addressed through the same fp32 pointer, so the
    // element offset belongs inside the accessor, after the cast.
    const at: usize = @intCast(c.start);
    const master = P.master[@intCast(c.param)];
    const m = P.m[@intCast(c.param)];
    const v = P.v[@intCast(c.param)];
    const grad: cuda.Global(f32) = P.grad[@intCast(c.param)] + at;
    const work = P.work[@intCast(c.param)] + at;
    const step_lr = lr * P.lrmul[@intCast(c.param)];
    const hm = P.flags[@intCast(c.param)] & 8 != 0;
    const hw = P.flags[@intCast(c.param)] & 16 != 0;
    // One counter stream per parameter, indexed by the element.
    const key = P.seed ^ @as(u32, @bitCast(c.param)) *% 0x2545f491;
    var i = cuda.threadIdxX();
    while (i < @as(u32, @bitCast(c.len))) : (i += cuda.blockDimX()) {
        const g = grad[i] * scale;
        const j = at + i;
        const nm = b1 * optLoad(m, j, hm) + (1.0 - b1) * g;
        const nv = b2 * optLoad(v, j, hm) + (1.0 - b2) * g * g;
        // One 32-bit word carries the two 16-bit draws the moments need; the
        // master copy takes a second word, so its rounding is independent.
        const r = cuda.mix32(@intCast(j), key);
        optStore(m, j, hm, nm, r);
        optStore(v, j, hm, nv, r >> 16);
        grad[i] = 0; // consumed; the next window accumulates into it again
        const x = optLoad(master, j, hw);
        const w = x - step_lr * (cuda.fdiv(cuda.fdiv(nm, bc1), cuda.fsqrt(cuda.fdiv(nv, bc2)) + eps) + wd * x);
        optStore(master, j, hw, w, cuda.mix32(@intCast(j), key +% 0x9e3779b9));
        // With a bf16 master the working copy is the master, so the store
        // above already wrote it.
        if (!hw) work[i] = cuda.f2bf(w);
    }
}

export fn mt_zero(P: MTParams) callconv(.nvptx_kernel) void {
    const c = P.chunks[cuda.blockIdxX()];
    const g = P.grad[@intCast(c.param)] + @as(usize, @intCast(c.start));
    var i = cuda.threadIdxX();
    while (i < @as(u32, @bitCast(c.len))) : (i += cuda.blockDimX()) g[i] = 0;
}

// ---- src/moe.cu ----
var moe_lp: [16]f32 addrspace(.shared) = undefined;

/// Fused router and top-k: one block of 64 threads per row, eight experts in
/// parallel (eight threads each), then softmax, top-k and renormalized weights.
/// Zero-mean noise on a routing logit, reproducible from its coordinates so
/// the same configuration routes the same way twice.
inline fn routerNoise(row: u32, e: u32, seed: u32) f32 {
    var h = (row *% 0x9e3779b9) ^ (e *% 0x85ebca6b) ^ (seed *% 0xc2b2ae35);
    h ^= h >> 15;
    h *%= 0x2545f491;
    h ^= h >> 13;
    const u = @as(f32, @floatFromInt(h >> 9)) * (1.0 / 8388608.0);
    const v = @as(f32, @floatFromInt((h *% 2654435761) >> 9)) * (1.0 / 8388608.0);
    return u + v - 1.0; // triangular on [-1,1], mean zero
}

/// Launched with eight lanes per expert, so the caller must provide `E*8`
/// threads. An earlier version launched sixty-four regardless, which covers
/// eight experts; beyond that the logits of experts 8 and up were never
/// written and the row below read uninitialised shared memory. It did not
/// crash - it silently routed on garbage, which looked exactly like a
/// training instability.
///
/// `noise` is the current amplitude of the routing noise. Expert selection is
/// a positive feedback loop - a slightly preferred expert gets more tokens,
/// so more gradient, so gets preferred more - and it starts before any expert
/// has learned anything to be preferred for. Noise early in training breaks
/// that loop by making selection independent of the tiny initial differences,
/// and it decays to nothing, so the converged model routes cleanly.
export fn router_topk(X: cuda.ConstGlobal(bf16), Wr: cuda.ConstGlobal(bf16), logits: cuda.Global(f32),
                      probs: cuda.Global(f32), idx: cuda.Global(i32), w: cuda.Global(f32),
                      N: i32, E: i32, K: i32, D: i32, noise: f32, seed: u32) callconv(.nvptx_kernel) void {
    const r = cuda.blockIdxX();
    if (r >= @as(u32, @bitCast(N))) return;
    const te = cuda.threadIdxX();
    const e = te / 8;
    const c = te % 8;
    const dim: usize = @intCast(D);
    const x: cuda.ConstGlobal(u32) = @ptrCast(@alignCast(X + @as(usize, r) * dim));
    var acc: f32 = 0;
    if (e < @as(u32, @bitCast(E))) {
        const we: cuda.ConstGlobal(u32) = @ptrCast(@alignCast(Wr + @as(usize, e) * dim));
        const D2 = @as(u32, @bitCast(D)) / 2;
        var d = c;
        while (d < D2) : (d += 8) {
            const a = cuda.bf2x2(x[d]);
            const b = cuda.bf2x2(we[d]);
            acc += a[0] * b[0] + a[1] * b[1];
        }
        if (D & 1 != 0 and c == 0)
            acc += cuda.bf2f(X[@as(usize, r) * dim + dim - 1]) * cuda.bf2f(Wr[@as(usize, e) * dim + dim - 1]);
        // The shuffle is warp-local, so the mask names this expert's eight
        // lanes *within its warp*. Shifting by e*8 works only while all
        // experts fit in one warp - at more than four it shifts past 32 and
        // the mask comes out empty.
        const mask: u32 = @as(u32, 0xFF) << @intCast((te % 32) / 8 * 8);
        var o: u32 = 4;
        while (o > 0) : (o >>= 1) acc += cuda.shflDown(mask, acc, o);
        if (c == 0) {
            // The noise enters the logit, so it shifts both the selection and
            // the weight that selection carries. The stored logit stays clean,
            // because the z-loss penalises the router's own magnitude and
            // should not be charged for the exploration.
            moe_lp[e] = if (noise != 0) acc + noise * routerNoise(r, e, seed) else acc;
            logits[r * @as(u32, @bitCast(E)) + e] = acc;
        }
    }
    cuda.syncThreads();
    if (te < @as(u32, @bitCast(E))) {
        var mx = moe_lp[0];
        for (1..@intCast(E)) |i| mx = cuda.__nv_fmaxf(mx, moe_lp[i]);
        var se: f32 = 0;
        for (0..@intCast(E)) |i| se += cuda.__nv_fast_expf(moe_lp[i] - mx);
        probs[r * @as(u32, @bitCast(E)) + te] = cuda.fdiv(cuda.__nv_fast_expf(moe_lp[te] - mx), se);
    }
    cuda.syncThreads();
    if (te == 0) topk_row(probs, idx, w, r, E, K);
}

/// Top-k without ties: the largest probabilities, then renormalized weights.
inline fn topk_row(probs: cuda.ConstGlobal(f32), idx: cuda.Global(i32), w: cuda.Global(f32),
                   r: u32, E: i32, K: i32) void {
    const rE = r * @as(u32, @bitCast(E));
    const rK = r * @as(u32, @bitCast(K));
    for (0..@intCast(K)) |j| {
        var best: i32 = -1;
        var bv: f32 = -1e30;
        for (0..@intCast(E)) |e| {
            var used = false;
            for (0..j) |q| if (idx[rK + q] == @as(i32, @intCast(e))) {
                used = true;
            };
            if (!used and probs[rE + e] > bv) {
                bv = probs[rE + e];
                best = @intCast(e);
            }
        }
        idx[rK + j] = best;
    }
    var z: f32 = 0;
    for (0..@intCast(K)) |j| z += probs[rE + @as(u32, @intCast(idx[rK + j]))];
    for (0..@intCast(K)) |j| w[rK + j] = cuda.fdiv(probs[rE + @as(u32, @intCast(idx[rK + j]))], z);
}

/// Top-k from fp32 logits (reference path).
export fn topk(logits: cuda.ConstGlobal(f32), idx: cuda.Global(i32), w: cuda.Global(f32),
               probs: cuda.Global(f32), N: i32, E: i32, K: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const L = logits + @as(usize, r) * @as(usize, @intCast(E));
    var mx = L[0];
    for (1..@intCast(E)) |e| mx = cuda.__nv_fmaxf(mx, L[e]);
    var se: f32 = 0;
    for (0..@intCast(E)) |e| se += cuda.__nv_fast_expf(L[e] - mx);
    for (0..@intCast(E)) |e| probs[r * @as(u32, @bitCast(E)) + e] = cuda.fdiv(cuda.__nv_fast_expf(L[e] - mx), se);
    topk_row(probs, idx, w, r, E, K);
}

export fn count_experts(idx: cuda.ConstGlobal(i32), counts: cuda.Global(i32), N: i32, K: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    for (0..@intCast(K)) |j| _ = cuda.atomicAddI32(&counts[@intCast(idx[r * @as(u32, @bitCast(K)) + j])], 1);
}

export fn fill_slots(idx: cuda.ConstGlobal(i32), w: cuda.ConstGlobal(f32), cursor: cuda.Global(i32),
                     perm: cuda.Global(i32), slotw: cuda.Global(f32), slot_of: cuda.Global(i32),
                     N: i32, K: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const rK = r * @as(u32, @bitCast(K));
    for (0..@intCast(K)) |j| {
        const e = idx[rK + j];
        const pos = cuda.atomicAddI32(&cursor[@intCast(e)], 1);
        perm[@intCast(pos)] = @intCast(r);
        slotw[@intCast(pos)] = w[rK + j];
        slot_of[rK + j] = pos;
    }
}

export fn gather_rows(X: cuda.ConstGlobal(bf16), perm: cuda.ConstGlobal(i32), Xg: cuda.Global(bf16),
                      TK: i32, D: i32) callconv(.nvptx_kernel) void {
    const pos = cuda.globalIdX();
    if (pos >= @as(u32, @bitCast(TK))) return;
    const dim: usize = @intCast(D);
    const s = X + @as(usize, @intCast(perm[pos])) * dim;
    const d = Xg + @as(usize, pos) * dim;
    for (0..dim) |dd| d[dd] = s[dd];
}

/// Xg[slot_of[r][j]] = X[r], one thread per element.
export fn gather_slot(X: cuda.ConstGlobal(bf16), slot_of: cuda.ConstGlobal(i32), Xg: cuda.Global(bf16),
                      N: i32, K: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const n = @as(i64, N) * K * D;
    if (i >= n) return;
    const dd = @mod(i, D);
    const tmp = @divTrunc(i, D);
    const j = @mod(tmp, K);
    const r = @divTrunc(tmp, K);
    const pos: i64 = slot_of[@intCast(r * K + j)];
    Xg[@intCast(pos * D + dd)] = X[@intCast(r * D + dd)];
}

/// y[r] = beta*y[r] + sum_j slotw*silu(Yg[slot]).
export fn combine(Yg: cuda.ConstGlobal(bf16), slotw: cuda.ConstGlobal(f32), slot_of: cuda.ConstGlobal(i32),
                  Y: cuda.Global(bf16), N: i32, K: i32, D: i32, beta: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const n = @as(i64, N) * D;
    if (i >= n) return;
    const dd = @mod(i, D);
    const r = @divTrunc(i, D);
    var acc: f32 = if (beta != 0) beta * cuda.bf2f(Y[@intCast(i)]) else 0;
    for (0..@intCast(K)) |j| {
        const pos: i64 = slot_of[@intCast(r * K + @as(i64, @intCast(j)))];
        acc += slotw[@intCast(pos)] * cuda.silu(cuda.bf2f(Yg[@intCast(pos * D + dd)]));
    }
    Y[@intCast(i)] = cuda.f2bf(acc);
}

/// dYg[slot] = silu'(pre) * w * dy, one thread per element.
export fn combine_bwd(dY: cuda.ConstGlobal(bf16), Yg: cuda.ConstGlobal(bf16), slotw: cuda.ConstGlobal(f32),
                      slot_of: cuda.ConstGlobal(i32), dYg: cuda.Global(bf16), s_j: cuda.Global(f32),
                      N: i32, K: i32, D: i32) callconv(.nvptx_kernel) void {
    _ = s_j;
    const i: i64 = @intCast(cuda.globalIdX());
    const n = @as(i64, N) * K * D;
    if (i >= n) return;
    const dd = @mod(i, D);
    const tmp = @divTrunc(i, D);
    const j = @mod(tmp, K);
    const r = @divTrunc(tmp, K);
    const pos: i64 = slot_of[@intCast(r * K + j)];
    const dy = cuda.bf2f(dY[@intCast(r * D + dd)]);
    const pre = cuda.bf2f(Yg[@intCast(pos * D + dd)]);
    dYg[@intCast(pos * D + dd)] = cuda.f2bf(cuda.siluBwd(pre, slotw[@intCast(pos)] * dy));
}

/// s_j[r][j] = dot(dY[r], silu(Yg[slot])): one block per (r,j).
export fn sdot(dY: cuda.ConstGlobal(bf16), Yg: cuda.ConstGlobal(bf16), slot_of: cuda.ConstGlobal(i32),
               s_j: cuda.Global(f32), N: i32, K: i32, D: i32) callconv(.nvptx_kernel) void {
    const rj = cuda.blockIdxX();
    if (rj >= @as(u32, @bitCast(N * K))) return;
    const r = rj / @as(u32, @bitCast(K));
    const j = rj % @as(u32, @bitCast(K));
    const pos: usize = @intCast(slot_of[r * @as(u32, @bitCast(K)) + j]);
    const dim: usize = @intCast(D);
    const t = cuda.threadIdxX();
    var acc: f32 = 0;
    var dd = t;
    while (dd < @as(u32, @bitCast(D))) : (dd += cuda.blockDimX())
        acc += cuda.bf2f(dY[@as(usize, r) * dim + dd]) * cuda.silu(cuda.bf2f(Yg[pos * dim + dd]));
    ln_buf[t] = acc;
    cuda.syncThreads();
    var half: u32 = 128;
    while (half > 0) : (half >>= 1) {
        if (t < half) ln_buf[t] += ln_buf[t + half];
        cuda.syncThreads();
    }
    if (t == 0) s_j[rj] = ln_buf[0];
}

/// Router backward through softmax, top-k and renormalization, plus the
/// analytic gradients of the load-balancing and z-loss terms.
export fn router_bwd(probs: cuda.ConstGlobal(f32), idx: cuda.ConstGlobal(i32), s_j: cuda.ConstGlobal(f32),
                     dlogits: cuda.Global(f32), N: i32, E: i32, K: i32, logits: cuda.ConstGlobal(f32),
                     counts: cuda.ConstGlobal(i32), aux: f32, zcoef: f32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const rE = r * @as(u32, @bitCast(E));
    const rK = r * @as(u32, @bitCast(K));
    var Z: f32 = 0;
    var sp: f32 = 0;
    for (0..@intCast(K)) |j| {
        const e: u32 = @intCast(idx[rK + j]);
        const p = probs[rE + e];
        Z += p;
        sp += s_j[rK + j] * p;
    }
    const Z2 = Z * Z + 1e-12;
    for (0..@intCast(E)) |e| dlogits[rE + e] = 0;
    for (0..@intCast(K)) |j| {
        const e: u32 = @intCast(idx[rK + j]);
        dlogits[rE + e] = cuda.fdiv(cuda.mulNoFma(s_j[rK + j], Z) - sp, Z2);
    }
    var acc: f32 = 0;
    for (0..@intCast(E)) |e| acc += dlogits[rE + e] * probs[rE + e];
    for (0..@intCast(E)) |e| dlogits[rE + e] = probs[rE + e] * (dlogits[rE + e] - acc);
    var expected: f32 = 0;
    var mx = logits[rE];
    var sum: f32 = 0;
    const nk: f32 = @floatFromInt(N * K);
    for (0..@intCast(E)) |e| {
        expected += cuda.fdiv(probs[rE + e] * @as(f32, @floatFromInt(counts[e])), nk);
        mx = cuda.__nv_fmaxf(mx, logits[rE + e]);
    }
    for (0..@intCast(E)) |e| sum += cuda.__nv_fast_expf(logits[rE + e] - mx);
    const lse = mx + cuda.__nv_fast_logf(sum);
    const fn_: f32 = @floatFromInt(N);
    for (0..@intCast(E)) |e| {
        const p = probs[rE + e];
        dlogits[rE + e] += cuda.fdiv(aux * @as(f32, @floatFromInt(E)), fn_) * p *
            (cuda.fdiv(@as(f32, @floatFromInt(counts[e])), nk) - expected) +
            cuda.fdiv(zcoef * 2.0, fn_) * lse * p;
    }
}

/// dX[r] += sum_j dXg[slot], read-modify-write per element (no atomics).
export fn scatter_add(dXg: cuda.ConstGlobal(bf16), slot_of: cuda.ConstGlobal(i32), dX: cuda.Global(bf16),
                      N: i32, K: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const n = @as(i64, N) * D;
    if (i >= n) return;
    const dd = @mod(i, D);
    const r = @divTrunc(i, D);
    var acc = cuda.bf2f(dX[@intCast(i)]);
    for (0..@intCast(K)) |j| acc += cuda.bf2f(dXg[@intCast(@as(i64, slot_of[@intCast(r * K + @as(i64, @intCast(j)))]) * D + dd)]);
    dX[@intCast(i)] = cuda.f2bf(acc);
}

export fn to_f32(s: cuda.ConstGlobal(bf16), d: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) d[@intCast(i)] = cuda.bf2f(s[@intCast(i)]);
}

/// Switch auxiliary loss: the summed probabilities per expert. One block per
/// expert, because a single block of E threads left the machine idle while it
/// walked the whole window.
export fn aux_sum(probs: cuda.ConstGlobal(f32), sum_p: cuda.Global(f32), N: i32, E: i32) callconv(.nvptx_kernel) void {
    const e = cuda.blockIdxX();
    if (e >= @as(u32, @bitCast(E))) return;
    const t = cuda.threadIdxX();
    const stride = cuda.blockDimX();
    var acc: f32 = 0;
    var r = t;
    while (r < @as(u32, @bitCast(N))) : (r += stride) acc += probs[@as(usize, r) * @as(usize, @intCast(E)) + e];
    ln_buf[t] = acc;
    cuda.syncThreads();
    var half = stride / 2;
    while (half > 0) : (half >>= 1) {
        if (t < half) ln_buf[t] += ln_buf[t + half];
        cuda.syncThreads();
    }
    if (t == 0) sum_p[e] = ln_buf[0];
}

/// z-loss: mean(lse^2) over the rows.
export fn zloss(logits: cuda.ConstGlobal(f32), out: cuda.Global(f32), N: i32, E: i32) callconv(.nvptx_kernel) void {
    const r = cuda.globalIdX();
    if (r >= @as(u32, @bitCast(N))) return;
    const rE = r * @as(u32, @bitCast(E));
    var mx = logits[rE];
    for (1..@intCast(E)) |j| mx = cuda.__nv_fmaxf(mx, logits[rE + j]);
    var sum: f32 = 0;
    for (0..@intCast(E)) |j| sum += cuda.__nv_fast_expf(logits[rE + j] - mx);
    const lse = mx + cuda.__nv_fast_logf(sum);
    cuda.atomicAddF32(&out[0], cuda.fdiv(lse * lse, @floatFromInt(N)));
}

/// Dense path: silu forward, or silu backward when a gradient is given.
export fn dense_activation(pre: cuda.ConstGlobal(bf16), grad: ?cuda.ConstGlobal(bf16), out: cuda.Global(bf16),
                           n: i64, beta: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= n) return;
    const u: usize = @intCast(i);
    const base: f32 = if (beta != 0) beta * cuda.bf2f(out[u]) else 0;
    out[u] = cuda.f2bf(base + if (grad) |g| cuda.siluBwd(cuda.bf2f(pre[u]), cuda.bf2f(g[u])) else cuda.silu(cuda.bf2f(pre[u])));
}

// ---- src/memory.cu (knowledge-graph fact memory) ----
// mem_cast_add, mem_f32_to_bf16 and mem_add_f32 are the same functions as
// cast_add, copy_bf16 and add_f32 above and reuse those kernels.

/// Fact bytes to their encoding: E[byte] + E_prev[previous byte] + P[position].
export fn mem_encode(E: cuda.ConstGlobal(bf16), Eprev: cuda.ConstGlobal(bf16), P: cuda.ConstGlobal(bf16),
                     ids: cuda.ConstGlobal(i32), enc: cuda.Global(bf16), B: i32, M: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * M * D) return;
    const d = @rem(i, D);
    const bj = @divTrunc(i, D);
    const j = @rem(bj, M);
    const b = @divTrunc(bj, M);
    const byte = ids[@intCast(b * M + j)];
    if (byte < 0) {
        enc[@intCast(i)] = cuda.f2bf(0);
        return;
    }
    const prev: i32 = if (j != 0) ids[@intCast(b * M + j - 1)] else -1;
    var v = cuda.bf2f(E[@intCast(@as(i64, byte) * D + d)]) + cuda.bf2f(P[@intCast(j * D + d)]);
    if (prev >= 0) v += cuda.bf2f(Eprev[@intCast(@as(i64, prev) * D + d)]);
    enc[@intCast(i)] = cuda.f2bf(v);
}

/// Scatter the encoding gradient into E, Eprev and P (padding skipped).
export fn mem_encode_bwd(dEnc: cuda.ConstGlobal(f32), ids: cuda.ConstGlobal(i32), dE: cuda.Global(f32),
                         dEprev: cuda.Global(f32), dP: cuda.Global(f32), B: i32, M: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * M * D) return;
    const d = @rem(i, D);
    const bj = @divTrunc(i, D);
    const j = @rem(bj, M);
    const b = @divTrunc(bj, M);
    const byte = ids[@intCast(b * M + j)];
    if (byte < 0) return;
    const g = dEnc[@intCast(i)];
    cuda.atomicAddF32(&dE[@intCast(@as(i64, byte) * D + d)], g);
    cuda.atomicAddF32(&dP[@intCast(j * D + d)], g);
    const prev: i32 = if (j != 0) ids[@intCast(b * M + j - 1)] else -1;
    if (prev >= 0) cuda.atomicAddF32(&dEprev[@intCast(@as(i64, prev) * D + d)], g);
}

/// One thread per (stream, head, position): masked softmax over the slots, o = P V.
export fn mem_attn_fwd(Q: cuda.ConstGlobal(bf16), K: cuda.ConstGlobal(bf16), V: cuda.ConstGlobal(bf16),
                       ids: cuda.ConstGlobal(i32), P: cuda.Global(f32), O: cuda.Global(bf16),
                       B: i32, T: i32, M: i32, H: i32, dh: i32, scale: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T) return;
    const t = @rem(i, T);
    const bh = @divTrunc(i, T);
    const h = @rem(bh, H);
    const b = @divTrunc(bh, H);
    const HD = H * dh;
    const q = Q + @as(usize, @intCast((b * T + t) * HD + h * dh));
    const p = P + @as(usize, @intCast(i * M));
    var mx: f32 = -1e30;
    var j: i64 = 0;
    while (j < M) : (j += 1) {
        if (ids[@intCast(b * M + j)] < 0) {
            p[@intCast(j)] = -1e30;
            continue;
        }
        const k = K + @as(usize, @intCast((b * M + j) * HD + h * dh));
        var s: f32 = 0;
        for (0..@intCast(dh)) |d| s += cuda.bf2f(q[d]) * cuda.bf2f(k[d]);
        p[@intCast(j)] = s * scale;
        mx = cuda.__nv_fmaxf(mx, p[@intCast(j)]);
    }
    var sum: f32 = 0;
    j = 0;
    while (j < M) : (j += 1) {
        p[@intCast(j)] = if (p[@intCast(j)] > -1e29) cuda.__nv_fast_expf(p[@intCast(j)] - mx) else 0;
        sum += p[@intCast(j)];
    }
    const inv: f32 = if (sum > 0) cuda.frcp(sum) else 0;
    j = 0;
    while (j < M) : (j += 1) p[@intCast(j)] *= inv;
    const o = O + @as(usize, @intCast((b * T + t) * HD + h * dh));
    for (0..@intCast(dh)) |d| {
        var acc: f32 = 0;
        j = 0;
        while (j < M) : (j += 1) {
            if (p[@intCast(j)] != 0)
                acc += p[@intCast(j)] * cuda.bf2f(V[@intCast((b * M + j) * HD + h * dh + @as(i64, @intCast(d)))]);
        }
        o[d] = cuda.f2bf(acc);
    }
}

/// Backward part 1: dS = P (dP - <P, dP>) and dQ.
export fn mem_attn_bwd_q(dO: cuda.ConstGlobal(bf16), K: cuda.ConstGlobal(bf16), V: cuda.ConstGlobal(bf16),
                         P: cuda.ConstGlobal(f32), dS: cuda.Global(f32), dQ: cuda.Global(bf16),
                         B: i32, T: i32, M: i32, H: i32, dh: i32, scale: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T) return;
    const t = @rem(i, T);
    const bh = @divTrunc(i, T);
    const h = @rem(bh, H);
    const b = @divTrunc(bh, H);
    const HD = H * dh;
    const g = dO + @as(usize, @intCast((b * T + t) * HD + h * dh));
    const p = P + @as(usize, @intCast(i * M));
    const s = dS + @as(usize, @intCast(i * M));
    var dot: f32 = 0;
    var j: i64 = 0;
    while (j < M) : (j += 1) {
        var dp: f32 = 0;
        if (p[@intCast(j)] != 0) {
            const v = V + @as(usize, @intCast((b * M + j) * HD + h * dh));
            for (0..@intCast(dh)) |d| dp += cuda.bf2f(g[d]) * cuda.bf2f(v[d]);
        }
        s[@intCast(j)] = dp;
        dot += p[@intCast(j)] * dp;
    }
    j = 0;
    while (j < M) : (j += 1) s[@intCast(j)] = p[@intCast(j)] * (s[@intCast(j)] - dot);
    const dq = dQ + @as(usize, @intCast((b * T + t) * HD + h * dh));
    for (0..@intCast(dh)) |d| {
        var acc: f32 = 0;
        j = 0;
        while (j < M) : (j += 1) {
            if (s[@intCast(j)] != 0)
                acc += s[@intCast(j)] * cuda.bf2f(K[@intCast((b * M + j) * HD + h * dh + @as(i64, @intCast(d)))]);
        }
        dq[d] = cuda.f2bf(acc * scale);
    }
}

/// Backward part 2: dK_j = scale * sum_t dS q_t, dV_j = sum_t P dO_t.
export fn mem_attn_bwd_kv(Q: cuda.ConstGlobal(bf16), dO: cuda.ConstGlobal(bf16), P: cuda.ConstGlobal(f32),
                          dS: cuda.ConstGlobal(f32), dK: cuda.Global(bf16), dV: cuda.Global(bf16),
                          B: i32, T: i32, M: i32, H: i32, dh: i32, scale: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * M) return;
    const j = @rem(i, M);
    const bh = @divTrunc(i, M);
    const h = @rem(bh, H);
    const b = @divTrunc(bh, H);
    const HD = H * dh;
    const dk = dK + @as(usize, @intCast((b * M + j) * HD + h * dh));
    const dv = dV + @as(usize, @intCast((b * M + j) * HD + h * dh));
    for (0..@intCast(dh)) |d| {
        var ak: f32 = 0;
        var av: f32 = 0;
        var t: i64 = 0;
        while (t < T) : (t += 1) {
            const pt: usize = @intCast((bh * T + t) * M + j);
            const qt: usize = @intCast((b * T + t) * HD + h * dh + @as(i64, @intCast(d)));
            ak += dS[pt] * cuda.bf2f(Q[qt]);
            av += P[pt] * cuda.bf2f(dO[qt]);
        }
        dk[d] = cuda.f2bf(ak * scale);
        dv[d] = cuda.f2bf(av);
    }
}

export fn mem_add_bf16(a: cuda.Global(bf16), b: cuda.ConstGlobal(bf16), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) a[@intCast(i)] = cuda.f2bf(cuda.bf2f(a[@intCast(i)]) + cuda.bf2f(b[@intCast(i)]));
}

/// enc = enc0 + state (the causal recurrence over the fact bytes).
export fn mem_sum(e: cuda.ConstGlobal(bf16), s: cuda.ConstGlobal(f32), out: cuda.Global(bf16), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) out[@intCast(i)] = cuda.f2bf(cuda.bf2f(e[@intCast(i)]) + s[@intCast(i)]);
}

// ---- src/mla.cu (long-range latent cache) ----
// mla_f2b, f32_to_bf16 and add_bf16_kernel_mla are the same functions as
// copy_bf16 and mem_add_bf16 above and reuse those kernels; so does
// nh_to_bhnt_bf16, which is to_bhnt.

/// RoPE over the first R features, in place of the pair (even, odd).
export fn rope(x: cuda.ConstGlobal(bf16), y: cuda.Global(bf16), pos: cuda.ConstGlobal(i64),
               N: i32, H: i32, F: i32, R: i32, theta: f32, neg: bool) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, N) * H * F) return;
    const f = @rem(i, F);
    const tmp = @divTrunc(i, F);
    const r = @divTrunc(tmp, H);
    const v = cuda.bf2f(x[@intCast(i)]);
    if (f < R) {
        const p2 = @divTrunc(f, 2);
        var ang = @as(f32, @floatFromInt(pos[@intCast(r)])) /
            cuda.__nv_fast_powf(theta, @as(f32, @floatFromInt(2 * p2)) / @as(f32, @floatFromInt(R)));
        if (neg) ang = -ang;
        const co = cuda.__nv_fast_cosf(ang);
        const si = cuda.__nv_fast_sinf(ang);
        const u = cuda.bf2f(x[@intCast(i ^ 1)]);
        const sgn: f32 = if (@rem(f, 2) == 0) -1.0 else 1.0;
        y[@intCast(i)] = cuda.f2bf(v * co + sgn * u * si);
    } else y[@intCast(i)] = x[@intCast(i)];
}

/// Write one window of latents and rotated keys into the ring cache.
export fn cache_write(latw: cuda.ConstGlobal(bf16), krw: cuda.ConstGlobal(bf16), lat: cuda.Global(bf16),
                      kr: cuda.Global(bf16), head: i64, N: i32, Cmax: i32, L: i32, R: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const LR = L + R;
    if (i >= @as(i64, N) * LR) return;
    const row = @divTrunc(i, LR);
    const f = @rem(i, LR);
    const dst = @rem(head + row, Cmax);
    if (f < L) lat[@intCast(dst * L + f)] = latw[@intCast(row * L + f)]
    else kr[@intCast(dst * R + (f - L))] = krw[@intCast(row * R + (f - L))];
}

/// Split the query into its compressed and its rotary half.
export fn qsplit(q: cuda.ConstGlobal(bf16), qc: cuda.Global(bf16), qr: cuda.Global(bf16),
                 N: i32, H: i32, dh: i32, R: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const W = dh + R;
    if (i >= @as(i64, N) * H * W) return;
    const fr = @rem(i, W);
    const tmp = @divTrunc(i, W);
    const h = @rem(tmp, H);
    const r = @divTrunc(tmp, H);
    if (fr < dh) qc[@intCast((r * H + h) * dh + fr)] = q[@intCast(i)]
    else qr[@intCast((r * H + h) * R + (fr - dh))] = q[@intCast(i)];
}

export fn qjoin(dqc: cuda.ConstGlobal(bf16), dqr: cuda.ConstGlobal(bf16), dq: cuda.Global(bf16),
                N: i32, H: i32, dh: i32, R: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const W = dh + R;
    if (i >= @as(i64, N) * H * W) return;
    const fr = @rem(i, W);
    const tmp = @divTrunc(i, W);
    const h = @rem(tmp, H);
    const r = @divTrunc(tmp, H);
    dq[@intCast(i)] = if (fr < dh) dqc[@intCast((r * H + h) * dh + fr)] else dqr[@intCast((r * H + h) * R + (fr - dh))];
}

/// Online softmax update for one cache chunk (one query block).
export fn online_update(S: cuda.ConstGlobal(f32), Vc: cuda.ConstGlobal(bf16), O: cuda.Global(f32),
                        m: cuda.Global(f32), l: cuda.Global(f32), Qc: i32, Cc: i32, dh: i32,
                        scale: f32) callconv(.nvptx_kernel) void {
    _ = scale;
    const q = cuda.blockIdxX();
    if (q >= @as(u32, @bitCast(Qc))) return;
    const s = S + @as(usize, q) * @as(usize, @intCast(Cc));
    var mx = s[0];
    for (1..@intCast(Cc)) |c| mx = cuda.__nv_fmaxf(mx, s[c]);
    const m_old = m[q];
    const m_new = cuda.__nv_fmaxf(m_old, mx);
    const alpha = cuda.__nv_fast_expf(m_old - m_new);
    const l_new = l[q] * alpha;
    const o = O + @as(usize, q) * @as(usize, @intCast(dh));
    var d = cuda.threadIdxX();
    while (d < @as(u32, @bitCast(dh))) : (d += cuda.blockDimX()) {
        var acc = o[d] * alpha;
        for (0..@intCast(Cc)) |c| acc += cuda.__nv_fast_expf(s[c] - m_new) * cuda.bf2f(Vc[c * @as(usize, @intCast(dh)) + d]);
        o[d] = acc;
    }
    cuda.syncThreads();
    if (cuda.threadIdxX() == 0) {
        var ps: f32 = 0;
        for (0..@intCast(Cc)) |c| ps += cuda.__nv_fast_expf(s[c] - m_new);
        l[q] = l_new + ps;
        m[q] = m_new;
    }
}

/// The same update over all (stream, head, position) rows.
export fn online_update_batched(S: cuda.ConstGlobal(f32), Vc: cuda.ConstGlobal(bf16), O: cuda.Global(f32),
                                m: cuda.Global(f32), l: cuda.Global(f32), T: i32, Cc: i32, dh: i32,
                                BH: i64) callconv(.nvptx_kernel) void {
    const qb: i64 = @intCast(cuda.blockIdxX());
    if (qb >= BH * T) return;
    const bh = @divTrunc(qb, T);
    const q = @rem(qb, T);
    const s = S + @as(usize, @intCast((bh * T + q) * Cc));
    var mx = s[0];
    for (1..@intCast(Cc)) |c| mx = cuda.__nv_fmaxf(mx, s[c]);
    const m_old = m[@intCast(qb)];
    const m_new = cuda.__nv_fmaxf(m_old, mx);
    const alpha = cuda.__nv_fast_expf(m_old - m_new);
    const l_new = l[@intCast(qb)] * alpha;
    const vc = Vc + @as(usize, @intCast(bh * Cc * dh));
    const o = O + @as(usize, @intCast(qb * dh));
    var d = cuda.threadIdxX();
    while (d < @as(u32, @bitCast(dh))) : (d += cuda.blockDimX()) {
        var acc = o[d] * alpha;
        for (0..@intCast(Cc)) |c| acc += cuda.__nv_fast_expf(s[c] - m_new) * cuda.bf2f(vc[c * @as(usize, @intCast(dh)) + d]);
        o[d] = acc;
    }
    cuda.syncThreads();
    if (cuda.threadIdxX() == 0) {
        var ps: f32 = 0;
        for (0..@intCast(Cc)) |c| ps += cuda.__nv_fast_expf(s[c] - m_new);
        l[@intCast(qb)] = l_new + ps;
        m[@intCast(qb)] = m_new;
    }
}

export fn norm_out(O: cuda.Global(f32), l: cuda.ConstGlobal(f32), rows: i64, dh: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= rows * dh) return;
    O[@intCast(i)] = cuda.fdiv(O[@intCast(i)], l[@intCast(@divTrunc(i, dh))]);
}

/// Softmax over one chunk plus the blend factors of the online pass.
export fn softmax_scale(S: cuda.ConstGlobal(f32), P: cuda.Global(bf16), m: cuda.Global(f32),
                        l: cuda.Global(f32), alpha: cuda.Global(f32), LSE: cuda.Global(f32),
                        rows: i64, Cc: i32, last: bool) callconv(.nvptx_kernel) void {
    const r: i64 = @intCast(cuda.blockIdxX());
    if (r >= rows) return;
    const s = S + @as(usize, @intCast(r * Cc));
    var mx = s[0];
    for (1..@intCast(Cc)) |c| mx = cuda.__nv_fmaxf(mx, s[c]);
    const m_old = m[@intCast(r)];
    const m_new = cuda.__nv_fmaxf(m_old, mx);
    const al = cuda.__nv_fast_expf(m_old - m_new);
    var ps: f32 = 0;
    var c = cuda.threadIdxX();
    while (c < @as(u32, @bitCast(Cc))) : (c += cuda.blockDimX()) {
        const p = cuda.__nv_fast_expf(s[c] - m_new);
        P[@intCast(r * Cc + c)] = cuda.f2bf(p);
        ps += p;
    }
    const t = cuda.threadIdxX();
    ln_buf[t] = ps;
    cuda.syncThreads();
    var half: u32 = 128;
    while (half > 0) : (half >>= 1) {
        if (t < half) ln_buf[t] += ln_buf[t + half];
        cuda.syncThreads();
    }
    if (t == 0) {
        l[@intCast(r)] = l[@intCast(r)] * al + ln_buf[0];
        m[@intCast(r)] = m_new;
        alpha[@intCast(r)] = al;
        // nvcc fuses log(l)*ln2 + m into one instruction here.
        if (last) LSE[@intCast(r)] = @mulAdd(f32, cuda.lg2(l[@intCast(r)]), cuda.ln2, m_new);
    }
}

export fn scale_rows(O: cuda.Global(f32), alpha: cuda.ConstGlobal(f32), rows: i64, dh: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= rows * dh) return;
    O[@intCast(i)] *= alpha[@intCast(@divTrunc(i, dh))];
}

/// Softmax Jacobian; the dot product runs over all key chunks.
export fn softmax_bwd(S: cuda.ConstGlobal(f32), dP: cuda.ConstGlobal(f32), LSE: cuda.ConstGlobal(f32),
                      dS: cuda.Global(bf16), P: cuda.Global(bf16), rows: i32, Cc: i32,
                      dO: cuda.ConstGlobal(f32), O: cuda.ConstGlobal(bf16), H: i32, T: i32,
                      dh: i32, scale: f32) callconv(.nvptx_kernel) void {
    const row = cuda.blockIdxX();
    if (row >= @as(u32, @bitCast(rows))) return;
    const t = @rem(@as(i32, @intCast(row)), T);
    const h = @rem(@divTrunc(@as(i32, @intCast(row)), T), H);
    const b = @divTrunc(@as(i32, @intCast(row)), T * H);
    const out: usize = @intCast((@as(i64, b) * T + t) * H * dh + h * dh);
    var dot: f32 = 0;
    for (0..@intCast(dh)) |d| dot += dO[@as(usize, row) * @as(usize, @intCast(dh)) + d] * cuda.bf2f(O[out + d]);
    var c = cuda.threadIdxX();
    while (c < @as(u32, @bitCast(Cc))) : (c += cuda.blockDimX()) {
        const i: usize = @as(usize, row) * @as(usize, @intCast(Cc)) + c;
        const p = cuda.__nv_fast_expf(S[i] - LSE[row]);
        P[i] = cuda.f2bf(p);
        dS[i] = cuda.f2bf(scale * p * (dP[i] - dot));
    }
}

/// (N,H,F) -> (B,H,T,F); also serves nh_to_bhnt_bf16.
export fn to_bhnt(s: cuda.ConstGlobal(bf16), d: cuda.Global(bf16), B: i32, H: i32, T: i32, F: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T * F) return;
    const f = @rem(i, F);
    const tmp = @divTrunc(i, F);
    const t = @rem(tmp, T);
    const h = @rem(@divTrunc(tmp, T), H);
    const b = @divTrunc(tmp, @as(i64, T) * H);
    d[@intCast(i)] = s[@intCast(((b * T + t) * H + h) * F + f)];
}

export fn to_nh(s: cuda.ConstGlobal(bf16), d: cuda.Global(bf16), B: i32, H: i32, T: i32, F: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T * F) return;
    const f = @rem(i, F);
    const tmp = @divTrunc(i, F);
    const h = @rem(tmp, H);
    const r = @divTrunc(tmp, H);
    d[@intCast((r * H + h) * F + f)] = s[@intCast(i)];
}

/// Gather one cache chunk, wrapping around the ring.
export fn cache_gather(lat: cuda.ConstGlobal(bf16), kr: cuda.ConstGlobal(bf16), clat: cuda.Global(bf16),
                       ckr: cuda.Global(bf16), head: i64, c0: i32, Cc: i32, Cmax: i32,
                       L: i32, R: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const LR = L + R;
    if (i >= @as(i64, Cc) * LR) return;
    const f = @rem(i, LR);
    const c = @divTrunc(i, LR);
    const slot = @rem(head + c0 + c, Cmax);
    if (f < L) clat[@intCast(c * L + f)] = lat[@intCast(slot * L + f)]
    else ckr[@intCast(c * R + (f - L))] = kr[@intCast(slot * R + (f - L))];
}

/// Scatter a chunk gradient back into the window (only slots inside it).
export fn masked_scatter(dC: cuda.ConstGlobal(f32), dW: cuda.Global(f32), base0: i64, H0: i64,
                         B: i32, T: i32, Cc: i32, F: i32, c0: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Cc * F) return;
    const f = @rem(i, F);
    const tmp = @divTrunc(i, F);
    const c = @rem(tmp, Cc);
    const b = @divTrunc(tmp, Cc);
    const pos = base0 + c0 + c;
    const w0 = base0 + H0;
    const w1 = w0 + T;
    if (pos < w0 or pos >= w1) return;
    const t = pos - w0;
    dW[@intCast((b * T + t) * F + f)] += dC[@intCast((b * Cc + c) * F + f)];
}

/// fp32 (B,H,T,F) -> bf16 (N,H*F), heads concatenated; the same function as
/// bhnt_to_flat_bf16_kernel, which reuses this kernel.
export fn o_to_flat(s: cuda.ConstGlobal(f32), d: cuda.Global(bf16), B: i32, H: i32, T: i32, dh: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T * dh) return;
    const f = @rem(i, dh);
    const tmp = @divTrunc(i, dh);
    const t = @rem(tmp, T);
    const h = @rem(@divTrunc(tmp, T), H);
    const b = @divTrunc(tmp, @as(i64, T) * H);
    d[@intCast(((b * T + t) * H + h) * dh + f)] = cuda.f2bf(s[@intCast(i)]);
}

export fn causal_mask(S: cuda.Global(f32), key_base: i64, c0: i32, q_base: i64, B: i32, H: i32,
                      T: i32, Cc: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T * Cc) return;
    const c = @rem(i, Cc);
    const tmp = @divTrunc(i, Cc);
    const t = @rem(tmp, T);
    if (key_base + c0 + c > q_base + t) S[@intCast(i)] = -1e30;
}

export fn fill_f32(a: cuda.Global(f32), v: f32, n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) a[@intCast(i)] = v;
}

/// Replicate the rotary keys of a chunk for every head.
export fn rep_hr(s: cuda.ConstGlobal(bf16), d: cuda.Global(bf16), B: i32, H: i32, Cc: i32, R: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * Cc * R) return;
    const f = @rem(i, R);
    const tmp = @divTrunc(i, R);
    const c = @rem(tmp, Cc);
    const b = @divTrunc(tmp, @as(i64, Cc) * H);
    d[@intCast(i)] = s[@intCast((b * Cc + c) * R + f)];
}

/// (B,H,Cc,F) -> (B*Cc,H*F), heads concatenated.
export fn bhn_to_flat(s: cuda.ConstGlobal(f32), d: cuda.Global(f32), B: i32, H: i32, Cc: i32, F: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Cc * H * F) return;
    const f = @rem(i, F);
    const tmp = @divTrunc(i, F);
    const h = @rem(tmp, H);
    const bc = @divTrunc(tmp, H);
    d[@intCast(i)] = s[@intCast((bc * H + h) * F + f)];
}

/// RoPE backward: rotation by -angle(pos).
export fn rope_bwd(s: cuda.ConstGlobal(f32), d: cuda.Global(f32), base: i64, c0: i32, B: i32,
                   Cc: i32, R: i32, theta: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Cc * R) return;
    const f = @rem(i, R);
    const tmp = @divTrunc(i, R);
    const c = @rem(tmp, Cc);
    const ang = -@as(f32, @floatFromInt(base + c0 + c)) /
        cuda.__nv_fast_powf(theta, @as(f32, @floatFromInt(2 * @divTrunc(f, 2))) / @as(f32, @floatFromInt(R)));
    const co = cuda.__nv_fast_cosf(ang);
    const si = cuda.__nv_fast_sinf(ang);
    const u = s[@intCast(i ^ 1)];
    const sgn: f32 = if (@rem(f, 2) == 0) -1.0 else 1.0;
    d[@intCast(i)] = s[@intCast(i)] * co + sgn * u * si;
}

/// fp32 (B,H,T,F) -> bf16 (N,H*F) with the inverse RoPE on the rotary half.
export fn dq_join(dQc: cuda.ConstGlobal(f32), dQr: cuda.ConstGlobal(f32), dq: cuda.Global(bf16),
                  B: i32, H: i32, T: i32, dh: i32, R: i32, pos: cuda.ConstGlobal(i64),
                  theta: f32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const W = dh + R;
    if (i >= @as(i64, B) * T * H * W) return;
    const f = @rem(i, W);
    const tmp = @divTrunc(i, W);
    const h = @rem(tmp, H);
    const r = @divTrunc(tmp, H);
    const b = @divTrunc(r, T);
    const t = @rem(r, T);
    if (f < dh) {
        dq[@intCast(i)] = cuda.f2bf(dQc[@intCast((b * H + h) * T * dh + t * dh + f)]);
    } else {
        const j = f - dh;
        const base = (b * H + h) * @as(i64, T) * R + t * R;
        const angle = @as(f32, @floatFromInt(pos[@intCast(r)])) /
            cuda.__nv_fast_powf(theta, @as(f32, @floatFromInt(2 * @divTrunc(j, 2))) / @as(f32, @floatFromInt(R)));
        const sign: f32 = if (@rem(j, 2) == 0) 1.0 else -1.0;
        dq[@intCast(i)] = cuda.f2bf(dQr[@intCast(base + j)] * cuda.__nv_fast_cosf(angle) +
            sign * dQr[@intCast(base + (j ^ 1))] * cuda.__nv_fast_sinf(angle));
    }
}

/// Evict the oldest prefix when a new window would exceed the capacity.
export fn compact_cache(data: cuda.Global(bf16), B: i32, capacity: i32, F: i32, drop: i32, keep: i32) callconv(.nvptx_kernel) void {
    const i = cuda.globalIdX();
    if (i >= @as(u32, @bitCast(B * F))) return;
    const b: i64 = @intCast(i / @as(u32, @bitCast(F)));
    const f: i64 = @intCast(i % @as(u32, @bitCast(F)));
    var c: i64 = 0;
    while (c < keep) : (c += 1)
        data[@intCast((b * capacity + c) * F + f)] = data[@intCast((b * capacity + c + drop) * F + f)];
}

export fn do_to_bhnt(s: cuda.ConstGlobal(bf16), d: cuda.Global(f32), B: i32, H: i32, T: i32, dh: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * H * T * dh) return;
    const f = @rem(i, dh);
    const tmp = @divTrunc(i, dh);
    const t = @rem(tmp, T);
    const h = @rem(@divTrunc(tmp, T), H);
    const b = @divTrunc(tmp, @as(i64, T) * H);
    d[@intCast(i)] = cuda.bf2f(s[@intCast(((b * T + t) * H + h) * dh + f)]);
}

/// Zero the columns [cce, Cc) of a (rows, Cc) matrix.
export fn zero_tail(A: cuda.Global(f32), rows: i64, cce: i32, Cc: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    const w = Cc - cce;
    if (i >= rows * w or cce >= Cc) return;
    const c = @rem(i, w);
    const r = @divTrunc(i, w);
    A[@intCast(r * Cc + cce + c)] = 0;
}

/// Sum over the heads: (B,H,Cc,F) -> (B*Cc,F).
export fn sum_heads(s: cuda.ConstGlobal(f32), d: cuda.Global(f32), B: i32, H: i32, Cc: i32, F: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Cc * F) return;
    const f = @rem(i, F);
    const tmp = @divTrunc(i, F);
    const c = @rem(tmp, Cc);
    const b = @divTrunc(tmp, Cc);
    var acc: f32 = 0;
    var h: i64 = 0;
    while (h < H) : (h += 1) acc += s[@intCast((b * H + h) * @as(i64, Cc) * F + c * F + f)];
    d[@intCast((b * Cc + c) * F + f)] = acc;
}

// ---- src/model.cu (elementary kernels) ----
// add_bf16_kernel and f32_of_bf16_kernel are mem_add_bf16 and to_f32 above.

export fn zero_f32(a: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) a[@intCast(i)] = 0;
}

/// The state of the last position of the window becomes the next carry.
export fn extract_carry(S: cuda.ConstGlobal(f32), carry: cuda.Global(f32), B: i32, T: i32, D: i32) callconv(.nvptx_kernel) void {
    const b = cuda.blockIdxX();
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (b >= @as(u32, @bitCast(B)) or d >= @as(u32, @bitCast(D))) return;
    const dim: usize = @intCast(D);
    carry[@as(usize, b) * dim + d] = S[(@as(usize, b) * @as(usize, @intCast(T)) + @as(usize, @intCast(T - 1))) * dim + d];
}

var mv_b1: [256]f32 addrspace(.shared) = undefined;
var mv_b2: [256]f32 addrspace(.shared) = undefined;

/// Mean and variance of a bf16 tensor, for the variance hinge.
export fn meanvar(X: cuda.ConstGlobal(bf16), out2: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const t = cuda.threadIdxX();
    var s: f32 = 0;
    var q: f32 = 0;
    var i: i64 = @intCast(t);
    while (i < n) : (i += @intCast(cuda.blockDimX())) {
        const v = cuda.bf2f(X[@intCast(i)]);
        s += v;
        q += v * v;
    }
    mv_b1[t] = s;
    mv_b2[t] = q;
    cuda.syncThreads();
    var half: u32 = 128;
    while (half > 0) : (half >>= 1) {
        if (t < half) {
            mv_b1[t] += mv_b1[t + half];
            mv_b2[t] += mv_b2[t + half];
        }
        cuda.syncThreads();
    }
    if (t == 0) {
        const m = cuda.fdiv(mv_b1[0], @floatFromInt(n));
        out2[0] = m;
        out2[1] = cuda.fdiv(mv_b2[0], @floatFromInt(n)) - cuda.mulNoFma(m, m);
    }
}

/// EMA target encoder: tgt = tau*tgt + (1-tau)*src.
export fn ema(tgt: cuda.Global(f32), src: cuda.ConstGlobal(f32), tau: f32, n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i < n) tgt[@intCast(i)] = tau * tgt[@intCast(i)] + (1.0 - tau) * src[@intCast(i)];
}

/// Mean squared error of two fp32 tensors (latent loss).
export fn mse_mean(a: cuda.ConstGlobal(f32), b: cuda.ConstGlobal(f32), out: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const t = cuda.threadIdxX();
    var s: f32 = 0;
    var i: i64 = @intCast(t);
    while (i < n) : (i += @intCast(cuda.blockDimX())) {
        const d = a[@intCast(i)] - b[@intCast(i)];
        s += d * d;
    }
    ln_buf[t] = s;
    cuda.syncThreads();
    var half: u32 = 128;
    while (half > 0) : (half >>= 1) {
        if (t < half) ln_buf[t] += ln_buf[t + half];
        cuda.syncThreads();
    }
    if (t == 0) out[0] = cuda.fdiv(ln_buf[0], @floatFromInt(n));
}

// ---- Muon: orthogonalized momentum for the two-dimensional weights ----
// The update direction is the momentum passed through a Newton-Schulz
// iteration, which pushes its singular values towards one. Everything runs in
// fp32 on the master weights; only the matrix products go through cuBLAS.

/// m = beta*m + g, and the Newton-Schulz input g + beta*m (Nesterov).
/// `half` keeps the momentum in bf16 with stochastic rounding (`mom_bf16`).
export fn muon_momentum(m: cuda.Global(f32), g: cuda.ConstGlobal(f32), x: cuda.Global(f32),
                        beta: f32, n: i64, half: i32, seed: u32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= n) return;
    const u: usize = @intCast(i);
    const h = half != 0;
    const nm = beta * optLoad(m, u, h) + g[u];
    optStore(m, u, h, nm, cuda.mix32(@intCast(u), seed));
    x[u] = g[u] + beta * nm;
}

/// Sum of squares of a matrix, accumulated per block.
export fn muon_sumsq(x: cuda.ConstGlobal(f32), out: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const t = cuda.threadIdxX();
    var s: f32 = 0;
    var i: i64 = @as(i64, cuda.blockIdxX()) * 256 + @as(i64, t);
    const stride: i64 = @as(i64, cuda.gridDimX()) * 256;
    while (i < n) : (i += stride) s += x[@intCast(i)] * x[@intCast(i)];
    ln_buf[t] = s;
    cuda.syncThreads();
    var half: u32 = 128;
    while (half > 0) : (half >>= 1) {
        if (t < half) ln_buf[t] += ln_buf[t + half];
        cuda.syncThreads();
    }
    if (t == 0 and ln_buf[0] != 0) cuda.atomicAddF32(&out[0], ln_buf[0]);
}

/// The Newton-Schulz iteration starts from a matrix of unit Frobenius norm,
/// and runs in bf16 because its matrix products go over the tensor cores.
export fn muon_start(x: cuda.ConstGlobal(f32), sumsq: cuda.ConstGlobal(f32), xb: cuda.Global(bf16),
                     n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= n) return;
    xb[@intCast(i)] = cuda.f2bf(x[@intCast(i)] * cuda.frsqrt(sumsq[0] + 1e-12));
}

/// out = b*A + c*(A@A), the polynomial of one Newton-Schulz step.
export fn muon_poly(A: cuda.ConstGlobal(bf16), AA: cuda.ConstGlobal(bf16), out: cuda.Global(bf16),
                    b: f32, c: f32, n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= n) return;
    out[@intCast(i)] = cuda.f2bf(b * cuda.bf2f(A[@intCast(i)]) + c * cuda.bf2f(AA[@intCast(i)]));
}

/// x = a*x + t, the other half of a Newton-Schulz step.
export fn muon_blend(x: cuda.Global(bf16), t: cuda.ConstGlobal(bf16), a: f32, n: i64) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= n) return;
    x[@intCast(i)] = cuda.f2bf(a * cuda.bf2f(x[@intCast(i)]) + cuda.bf2f(t[@intCast(i)]));
}

/// The weight step, with decoupled weight decay, plus the bf16 working copy.
export fn muon_update(master: cuda.Global(f32), x: cuda.ConstGlobal(bf16), work: cuda.Global(bf16),
                      lr: f32, scale: f32, wd: f32, n: i64, half: i32, seed: u32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= n) return;
    const u: usize = @intCast(i);
    const h = half != 0;
    const p = optLoad(master, u, h);
    const w = p - lr * (scale * cuda.bf2f(x[u]) + wd * p);
    optStore(master, u, h, w, cuda.mix32(@intCast(u), seed));
    // A bf16 master *is* the working copy; the store above wrote it.
    if (!h) work[u] = cuda.f2bf(w);
}

// The fp32 -> bf16 conversion the Newton-Schulz products need is copy_bf16.

// ---- dynamic byte patching ----
// The lower layers run per byte, a middle group runs once per patch, and the
// upper layers run per byte again as a local decoder. Pooling reads the
// representation at the last byte of each patch; the broadcast gives byte i
// the output of the last patch that ends at or before i, so the boundary byte
// itself already sees its own patch (its output depends only on bytes <= i).

/// Gather the representation at each patch boundary: Xp[b,j] = X[b, ends[b,j]].
/// A negative end marks a padded patch slot, whose row becomes zero.
export fn patch_pool(X: cuda.ConstGlobal(bf16), ends: cuda.ConstGlobal(i32), Xp: cuda.Global(bf16),
                     B: i32, T: i32, Tp: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Tp * D) return;
    const d = @rem(i, D);
    const bj = @divTrunc(i, D);
    const j = @rem(bj, Tp);
    const b = @divTrunc(bj, Tp);
    const at = ends[@intCast(b * Tp + j)];
    Xp[@intCast(i)] = if (at < 0) cuda.f2bf(0) else X[@intCast((b * T + at) * D + d)];
}

/// X[b,i] += Xp[b, patch_of[b,i]], with patch_of < 0 where no patch has ended yet.
export fn patch_broadcast(Xp: cuda.ConstGlobal(bf16), patch_of: cuda.ConstGlobal(i32), X: cuda.Global(bf16),
                          B: i32, T: i32, Tp: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * T * D) return;
    const d = @rem(i, D);
    const bt = @divTrunc(i, D);
    const t = @rem(bt, T);
    const b = @divTrunc(bt, T);
    const j = patch_of[@intCast(b * T + t)];
    if (j < 0) return;
    X[@intCast(i)] = cuda.f2bf(cuda.bf2f(X[@intCast(i)]) + cuda.bf2f(Xp[@intCast((b * Tp + j) * D + d)]));
}

/// Backward of the pooling: the boundary byte receives the patch gradient.
export fn patch_pool_bwd(dXp: cuda.ConstGlobal(bf16), ends: cuda.ConstGlobal(i32), dX: cuda.Global(bf16),
                         B: i32, T: i32, Tp: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Tp * D) return;
    const d = @rem(i, D);
    const bj = @divTrunc(i, D);
    const j = @rem(bj, Tp);
    const b = @divTrunc(bj, Tp);
    const at = ends[@intCast(b * Tp + j)];
    if (at < 0) return;
    const to: usize = @intCast((b * T + at) * D + d);
    dX[to] = cuda.f2bf(cuda.bf2f(dX[to]) + cuda.bf2f(dXp[@intCast(i)]));
}

/// Backward of the broadcast: each patch sums the gradient of the bytes it
/// was broadcast to. The bytes of one patch are contiguous, so one thread per
/// (patch, channel) adds them in order, which needs no atomics.
export fn patch_broadcast_bwd(dX: cuda.ConstGlobal(bf16), first: cuda.ConstGlobal(i32), last: cuda.ConstGlobal(i32),
                              dXp: cuda.Global(bf16), B: i32, T: i32, Tp: i32, D: i32) callconv(.nvptx_kernel) void {
    const i: i64 = @intCast(cuda.globalIdX());
    if (i >= @as(i64, B) * Tp * D) return;
    const d = @rem(i, D);
    const bj = @divTrunc(i, D);
    const j = @rem(bj, Tp);
    const b = @divTrunc(bj, Tp);
    const lo = first[@intCast(b * Tp + j)];
    const hi = last[@intCast(b * Tp + j)];
    var acc: f32 = 0;
    var t: i64 = lo;
    while (t <= hi and lo >= 0) : (t += 1) acc += cuda.bf2f(dX[@intCast((b * T + t) * D + d)]);
    dXp[@intCast(i)] = cuda.f2bf(acc);
}

/// The embedding gradient without atomics, in two passes.
///
/// The host sorts the window's positions by byte value and cuts each byte's
/// run into tiles of equal size. One block per (tile, channel block) sums its
/// tile, and a second pass adds a byte's tiles together in a fixed order.
/// Summing a whole byte in one block would be reproducible too, but the
/// frequent bytes are hundreds of positions long and the rare ones two, so
/// one block would run while the rest of the machine idles.
export fn emb_tiles(dOut: cuda.ConstGlobal(f32), perm: cuda.ConstGlobal(i32),
                    tile_from: cuda.ConstGlobal(i32), tile_to: cuda.ConstGlobal(i32),
                    partial: cuda.Global(f32), D: i32) callconv(.nvptx_kernel) void {
    const tile = cuda.blockIdxX();
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (d >= @as(u32, @bitCast(D))) return;
    const dim: usize = @intCast(D);
    var acc: f32 = 0;
    var i = tile_from[tile];
    const to = tile_to[tile];
    while (i < to) : (i += 1) acc += dOut[@as(usize, @intCast(perm[@intCast(i)])) * dim + d];
    partial[@as(usize, tile) * dim + d] = acc;
}

/// Second pass: a byte's tiles, added in the order the host laid them out.
export fn emb_reduce(partial: cuda.ConstGlobal(f32), byte_tiles: cuda.ConstGlobal(i32),
                     dW: cuda.Global(f32), D: i32) callconv(.nvptx_kernel) void {
    const r = cuda.blockIdxX(); // byte value
    const d = cuda.blockIdxY() * cuda.blockDimX() + cuda.threadIdxX();
    if (d >= @as(u32, @bitCast(D))) return;
    const from = byte_tiles[r];
    const to = byte_tiles[r + 1];
    if (from == to) return;
    const dim: usize = @intCast(D);
    var acc: f32 = 0;
    var t = from;
    while (t < to) : (t += 1) acc += partial[@as(usize, @intCast(t)) * dim + d];
    dW[@as(usize, r) * dim + d] += acc;
}

// ------------------------------------------------------------------ NVFP4
//
// NVFP4 is e2m1 - one sign bit, two exponent bits, one mantissa bit, so the
// representable magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 - with one e4m3
// scale per sixteen consecutive values. Two values share a byte.
//
// The format carries about two decimal digits, so everything depends on the
// scale being chosen per block rather than per tensor: within sixteen
// neighbouring weights the dynamic range is small, across a whole matrix it
// is not.

/// The eight magnitudes of e2m1, in order of their encoding.
const FP4_MAG = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };

/// Nearest e2m1 code for a value already divided by its block scale.
/// Ties go to even, as they do in every other rounding in this file.
inline fn toFp4(v: f32) u8 {
    const a = @abs(v);
    var code: u8 = 0;
    // Eight magnitudes, so a linear scan beats any cleverness and stays
    // branch-predictable.
    var best: f32 = 1e30;
    for (FP4_MAG, 0..) |m, i| {
        const d = @abs(a - m);
        if (d < best) {
            best = d;
            code = @intCast(i);
        }
    }
    return code | (if (v < 0) @as(u8, 8) else 0);
}

inline fn fromFp4(c: u8) f32 {
    const m = FP4_MAG[c & 7];
    return if (c & 8 != 0) -m else m;
}

/// e4m3 for the block scales, without the fp8 intrinsics: the scale is
/// always positive here, so only the exponent and three mantissa bits matter.
inline fn toE4M3(v: f32) u8 {
    if (!(v > 0)) return 0;
    const b: u32 = @bitCast(v);
    var e: i32 = @intCast((b >> 23) & 0xFF);
    e -= 127;
    if (e < -6) return 0; // underflows the format
    if (e > 8) return 0x7E; // the largest finite value
    const mant = (b >> 20) & 7;
    // Round to nearest on the discarded mantissa bits.
    const rest = b & 0xFFFFF;
    var m = mant;
    var ee = e;
    if (rest > 0x80000 or (rest == 0x80000 and (mant & 1) != 0)) {
        m += 1;
        if (m == 8) {
            m = 0;
            ee += 1;
            if (ee > 8) return 0x7E;
        }
    }
    return @intCast(((@as(u32, @intCast(ee + 7)) & 0xF) << 3) | m);
}

inline fn fromE4M3(c: u8) f32 {
    if (c == 0) return 0;
    const e: i32 = @intCast((c >> 3) & 0xF);
    const m: u32 = c & 7;
    const bits: u32 = (@as(u32, @intCast(e - 7 + 127)) << 23) | (m << 20);
    return @bitCast(bits);
}

/// The largest magnitude in the tensor, for the per-tensor scale. One block
/// per launch slice, atomics into a single float.
export fn fp4_absmax(src: cuda.ConstGlobal(bf16), out: cuda.Global(f32), n: i64) callconv(.nvptx_kernel) void {
    const t = cuda.threadIdxX();
    var m: f32 = 0;
    var i: i64 = @intCast(cuda.globalIdX());
    const stride: i64 = @intCast(cuda.blockDimX() * cuda.gridDimX());
    while (i < n) : (i += stride) m = cuda.__nv_fmaxf(m, @abs(cuda.bf2f(src[@intCast(i)])));
    mt_buf[t] = m;
    cuda.syncThreads();
    var half = cuda.blockDimX() / 2;
    while (half > 0) : (half >>= 1) {
        if (t < half) mt_buf[t] = cuda.__nv_fmaxf(@floatCast(mt_buf[t]), @floatCast(mt_buf[t + half]));
        cuda.syncThreads();
    }
    if (t == 0) _ = cuda.atomicMaxF32(&out[0], @floatCast(mt_buf[0]));
}

/// Quantises a bf16 matrix to NVFP4: two e2m1 values per output byte, one
/// e4m3 scale per sixteen values along the contiguous axis, and one fp32
/// scale for the whole tensor.
///
/// The per-tensor scale is not optional. e4m3 spans 2^-6 to 2^8, so a block
/// scale below about 0.016 underflows to zero - and for weights of magnitude
/// 0.02 the natural block scale is 0.008. Without the outer scale every
/// block quantises to zero, which is exactly what a first version of this
/// did. `gscale` divides the block scales into e4m3's range; the decode
/// multiplies it back.
///
/// Within a block the scale is the largest magnitude over 6, the largest
/// e2m1 magnitude, so the extreme value lands on a representable point and
/// nothing clips.
export fn fp4_quant(src: cuda.ConstGlobal(bf16), dst: cuda.Global(u8), scale: cuda.Global(u8),
                    gscale: cuda.ConstGlobal(f32), nblocks: i64) callconv(.nvptx_kernel) void {
    const b: i64 = @intCast(cuda.globalIdX());
    if (b >= nblocks) return;
    const at: usize = @intCast(b * 16);
    const g = gscale[0];
    var mx: f32 = 0;
    for (0..16) |i| mx = cuda.__nv_fmaxf(mx, @abs(cuda.bf2f(src[at + i])));
    const s = if (mx > 0 and g > 0) cuda.fdiv(mx * (1.0 / 6.0), g) else @as(f32, 0);
    const sc = toE4M3(s);
    scale[@intCast(b)] = sc;
    const eff = fromE4M3(sc) * g;
    const inv = if (eff > 0) cuda.frcp(eff) else @as(f32, 0);
    for (0..8) |i| {
        const lo = toFp4(cuda.bf2f(src[at + 2 * i]) * inv);
        const hi = toFp4(cuda.bf2f(src[at + 2 * i + 1]) * inv);
        dst[@intCast(b * 8 + @as(i64, @intCast(i)))] = lo | (hi << 4);
    }
}

/// The inverse, for checking the quantiser against the values it came from.
export fn fp4_dequant(src: cuda.ConstGlobal(u8), scale: cuda.ConstGlobal(u8),
                      gscale: cuda.ConstGlobal(f32), dst: cuda.Global(bf16),
                      nblocks: i64) callconv(.nvptx_kernel) void {
    const b: i64 = @intCast(cuda.globalIdX());
    if (b >= nblocks) return;
    const s = fromE4M3(scale[@intCast(b)]) * gscale[0];
    for (0..8) |i| {
        const byte = src[@intCast(b * 8 + @as(i64, @intCast(i)))];
        dst[@intCast(b * 16 + @as(i64, @intCast(i)) * 2)] = cuda.f2bf(fromFp4(byte & 0xF) * s);
        dst[@intCast(b * 16 + @as(i64, @intCast(i)) * 2 + 1)] = cuda.f2bf(fromFp4(byte >> 4) * s);
    }
}
