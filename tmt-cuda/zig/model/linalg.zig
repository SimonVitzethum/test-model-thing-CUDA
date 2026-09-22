//! Linear layers in bf16 through cuBLAS (port of src/linalg.cu and the GEMM
//! helpers of src/util.h and src/mla.cu). Weights use the nn.Linear layout
//! W(N,K) row-major: Y(M,N) = X(M,K) @ W^T, dX = dY @ W, dW = dY^T @ X.
const std = @import("std");
const gpu = @import("gpu.zig");

pub const bf16 = u16;

const OP_N: c_int = 0;
const OP_T: c_int = 1;
const R_32F: c_int = 0;
const R_16BF: c_int = 14;
const COMPUTE_32F: c_int = 68;
const GEMM_DEFAULT_TENSOR_OP: c_int = 99;

extern fn cublasCreate_v2(handle: *?*anyopaque) c_int;
extern fn cublasGemmEx(handle: ?*anyopaque, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int,
                       alpha: *const f32, A: ?*const anyopaque, Atype: c_int, lda: c_int,
                       B: ?*const anyopaque, Btype: c_int, ldb: c_int, beta: *const f32,
                       C: ?*anyopaque, Ctype: c_int, ldc: c_int, compute: c_int, algo: c_int) c_int;
extern fn cublasGemmStridedBatchedEx(handle: ?*anyopaque, transa: c_int, transb: c_int, m: c_int, n: c_int, k: c_int,
                                     alpha: *const f32, A: ?*const anyopaque, Atype: c_int, lda: c_int, strideA: c_longlong,
                                     B: ?*const anyopaque, Btype: c_int, ldb: c_int, strideB: c_longlong, beta: *const f32,
                                     C: ?*anyopaque, Ctype: c_int, ldc: c_int, strideC: c_longlong,
                                     batch: c_int, compute: c_int, algo: c_int) c_int;

var handle: ?*anyopaque = null;
fn h() gpu.Error!?*anyopaque {
    if (handle == null and cublasCreate_v2(&handle) != 0) return fail("cublasCreate");
    return handle;
}
var msg: [128]u8 = undefined;
fn fail(what: []const u8) gpu.Error {
    _ = std.fmt.bufPrint(&msg, "{s} failed", .{what}) catch {};
    return error.Cuda;
}
fn check(rc: c_int, what: []const u8) gpu.Error!void {
    if (rc != 0) return fail(what);
}

/// Y(M,N) = X(M,K) @ W(N,K)^T
pub fn linearFwd(M: i32, N: i32, K: i32, X: [*]const bf16, W: [*]const bf16, Y: [*]bf16) !void {
    const al: f32 = 1;
    const be: f32 = 0;
    try check(cublasGemmEx(try h(), OP_T, OP_N, N, M, K, &al, W, R_16BF, K, X, R_16BF, K,
        &be, Y, R_16BF, N, COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP), "linear_fwd");
}

/// dX(M,K) = dY(M,N) @ W(N,K), accumulating when beta is nonzero.
pub fn linearDX(M: i32, N: i32, K: i32, dY: [*]const bf16, W: [*]const bf16, dX: [*]bf16, beta: f32) !void {
    const al: f32 = 1;
    try check(cublasGemmEx(try h(), OP_N, OP_N, K, M, N, &al, W, R_16BF, K, dY, R_16BF, N,
        &beta, dX, R_16BF, K, COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP), "linear_dX");
}

/// dW(N,K) fp32 = dY(M,N)^T @ X(M,K), accumulating when beta is nonzero.
pub fn linearDW(M: i32, N: i32, K: i32, dY: [*]const bf16, X: [*]const bf16, dW: [*]f32, beta: f32) !void {
    const al: f32 = 1;
    try check(cublasGemmEx(try h(), OP_N, OP_T, K, N, M, &al, X, R_16BF, K, dY, R_16BF, N,
        &beta, dW, R_32F, K, COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP), "linear_dW");
}

/// Row-major C(M,N) = A(M,K) @ B(K,N), bf16 in and out.
pub fn gemmNN(M: i32, N: i32, K: i32, A: [*]const bf16, lda: i32, B: [*]const bf16, ldb: i32,
              C: [*]bf16, ldc: i32) !void {
    _ = lda;
    _ = ldc;
    const al: f32 = 1;
    const be: f32 = 0;
    try check(cublasGemmEx(try h(), OP_N, OP_N, N, M, K, &al, B, R_16BF, ldb, A, R_16BF, K,
        &be, C, R_16BF, N, COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP), "gemm_nn");
}

/// The batched form the MLA attention uses: one GEMM per (stream, head).
pub const Batched = struct {
    transa: bool,
    transb: bool,
    m: i32,
    n: i32,
    k: i32,
    A: [*]const bf16,
    lda: i32,
    strideA: i64,
    B: [*]const bf16,
    ldb: i32,
    strideB: i64,
    C: [*]f32,
    ldc: i32,
    strideC: i64,
    batch: i32,
    beta: f32 = 0,
};
pub fn batched(g: Batched) !void {
    const al: f32 = 1;
    try check(cublasGemmStridedBatchedEx(try h(), if (g.transa) OP_T else OP_N, if (g.transb) OP_T else OP_N,
        g.m, g.n, g.k, &al, g.A, R_16BF, g.lda, g.strideA, g.B, R_16BF, g.ldb, g.strideB,
        &g.beta, g.C, R_32F, g.ldc, g.strideC, g.batch, COMPUTE_32F, GEMM_DEFAULT_TENSOR_OP), "batched GEMM");
}
