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

/// master -> bf16 working copy (copy_bf16_kernel).
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
        const v = cuda.fdiv(total_sq, @floatFromInt(D)) - ln_m * ln_m;
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
