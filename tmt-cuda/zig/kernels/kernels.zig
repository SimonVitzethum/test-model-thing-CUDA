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
