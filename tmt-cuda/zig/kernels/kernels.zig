//! The CUDA kernels in Zig. Compiled to PTX (see build.zig) and launched from
//! the host through the driver API; each kernel is a line-by-line port of its
//! counterpart in src/*.cu and must produce bit-identical results.
const cuda = @import("cuda.zig");
const bf16 = cuda.bf16;

// ---- src/emb.cu ----
/// Embedding gather: one row of the table per position, zero for ignored targets.
export fn emb_gather(W: cuda.ConstGlobal(bf16), ids: cuda.ConstGlobal(i32), out: cuda.Global(bf16),
                     N: i32, D: i32) callconv(.nvptx_kernel) void {
    const i = cuda.globalIdX();
    if (i >= @as(u32, @bitCast(N))) return;
    const d = out + @as(usize, i) * @as(usize, @intCast(D));
    if (ids[i] < 0) {
        for (0..@intCast(D)) |dd| d[dd] = cuda.f2bf(0);
        return;
    }
    const s = W + @as(usize, @intCast(ids[i])) * @as(usize, @intCast(D));
    for (0..@intCast(D)) |dd| d[dd] = s[dd];
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
    const lse = cuda.__nv_fast_logf(se) + mx;
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
};

/// a = sigmoid(decay + gate*x), or 0 at a document separator (s_t = x_t).
inline fn cellA(o: CellOpt, decay: f32, gate: ?cuda.ConstGlobal(f32), d: u32, x: f32, b: u32, T: i32, t: i32) f32 {
    if (o.docsep >= 0 and o.ids.?[@intCast(@as(i32, @intCast(b)) * T + t)] == o.docsep) return 0;
    return cuda.sigmoid(decay + if (gate) |g| g[d] * x else 0);
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
        const dec = cellA(o, decay[d], gate, d, x, b, T, t);
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
        const a = cellA(o, decay[d], gate, d, x, b, T, t);
        const total = cuda.bf2f(dS[idx]) + future;
        const prev = if (t != 0) S[idx - @as(usize, @intCast(D))] else if (initial) |i| i[@as(usize, b) * @as(usize, @intCast(D)) + d] else 0;
        const local = (prev - x) * a * (1.0 - a);
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
        const a = cellA(o, decay[d], gate, d, x, b, T, t);
        const prev = if (t != 0) S[idx - dim] else initial[bd];
        const k = (1.0 - a) + if (gate) |g| (prev - x) * a * (1.0 - a) * g[d] else 0;
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
    flags: cuda.Global(u8), // bit 0: in the gradient norm, bit 1: updated by AdamW
    chunks: cuda.Global(MTChunk),
    nchunks: i32,
    sumsq: cuda.Global(f64), // global squared gradient norm
};

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

/// AdamW with global-norm clipping computed on the device. A non-finite norm
/// skips the update entirely (the host then refuses the step).
export fn mt_adam(P: MTParams, clip: f32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32,
                  bc1: f32, bc2: f32) callconv(.nvptx_kernel) void {
    const c = P.chunks[cuda.blockIdxX()];
    if (P.flags[@intCast(c.param)] & 2 == 0) return;
    const s = P.sumsq[0];
    if (!(s - s == 0)) return; // non-finite
    const scale: f32 = if (clip > 0) cuda.__nv_fminf(1, cuda.fdiv(clip, cuda.__nv_fmaxf(@floatCast(@sqrt(s)), 1e-12))) else 1;
    const at: usize = @intCast(c.start);
    const master = P.master[@intCast(c.param)] + at;
    const m = P.m[@intCast(c.param)] + at;
    const v = P.v[@intCast(c.param)] + at;
    const grad = P.grad[@intCast(c.param)] + at;
    const work = P.work[@intCast(c.param)] + at;
    var i = cuda.threadIdxX();
    while (i < @as(u32, @bitCast(c.len))) : (i += cuda.blockDimX()) {
        const g = grad[i] * scale;
        const nm = b1 * m[i] + (1.0 - b1) * g;
        const nv = b2 * v[i] + (1.0 - b2) * g * g;
        m[i] = nm;
        v[i] = nv;
        const w = master[i] - lr * (cuda.fdiv(cuda.fdiv(nm, bc1), cuda.fsqrt(cuda.fdiv(nv, bc2)) + eps) + wd * master[i]);
        master[i] = w;
        work[i] = cuda.f2bf(w);
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
export fn router_topk(X: cuda.ConstGlobal(bf16), Wr: cuda.ConstGlobal(bf16), logits: cuda.Global(f32),
                      probs: cuda.Global(f32), idx: cuda.Global(i32), w: cuda.Global(f32),
                      N: i32, E: i32, K: i32, D: i32) callconv(.nvptx_kernel) void {
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
        const mask: u32 = @as(u32, 0xFF) << @intCast(e * 8);
        var o: u32 = 4;
        while (o > 0) : (o >>= 1) acc += cuda.shflDown(mask, acc, o);
        if (c == 0) {
            moe_lp[e] = acc;
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

/// Switch auxiliary loss: the summed probabilities per expert.
export fn aux_sum(probs: cuda.ConstGlobal(f32), sum_p: cuda.Global(f32), N: i32, E: i32) callconv(.nvptx_kernel) void {
    const e = cuda.threadIdxX();
    if (e >= @as(u32, @bitCast(E))) return;
    var acc: f32 = 0;
    for (0..@intCast(N)) |r| acc += probs[r * @as(usize, @intCast(E)) + e];
    sum_p[e] = acc;
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
