//! NVFP4 matrix multiply through cuBLASLt.
//!
//! e2m1 values - two to a byte - with one e4m3 scale per sixteen along the
//! contiguous axis, and one fp32 scale for the whole tensor. The library has
//! kernels for this on sm_120, so the matmul itself needs no hand-written
//! MMA; what this file provides is the descriptor plumbing, which is where
//! the format's rules actually live.
//!
//! Two of those rules are not optional and cost a day each if missed. The
//! heuristic reports NOT_SUPPORTED unless the scale *pointers* are set, not
//! merely the scale modes. And the layouts must be K-major on both sides -
//! the A^T B form - because that is the only orientation the FP4 kernels
//! come in.
const std = @import("std");

pub const bf16 = u16;

const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const CUDA_R_4F_E2M1: c_int = 33;
const CUBLAS_COMPUTE_32F: c_int = 68;

const DESC_TRANSA: c_int = 3;
const DESC_TRANSB: c_int = 4;
const DESC_A_SCALE_POINTER: c_int = 17;
const DESC_B_SCALE_POINTER: c_int = 18;
const DESC_A_SCALE_MODE: c_int = 31;
const DESC_B_SCALE_MODE: c_int = 32;
const PREF_MAX_WORKSPACE: c_int = 1;
/// One e4m3 scale per sixteen values: NVFP4 rather than the MXFP4 variant,
/// which this GPU's cuBLASLt does not implement.
const SCALE_VEC16_UE4M3: c_int = 1;

const OP_N: c_int = 0;
const OP_T: c_int = 1;

extern fn cudaMalloc(p: *?*anyopaque, n: usize) c_int;
extern fn cublasLtCreate(h: *?*anyopaque) c_int;
extern fn cublasLtMatmulDescCreate(d: *?*anyopaque, compute: c_int, scale: c_int) c_int;
extern fn cublasLtMatmulDescSetAttribute(d: ?*anyopaque, attr: c_int, buf: *const anyopaque, n: usize) c_int;
extern fn cublasLtMatmulDescDestroy(d: ?*anyopaque) c_int;
extern fn cublasLtMatrixLayoutCreate(l: *?*anyopaque, dtype: c_int, rows: u64, cols: u64, ld: i64) c_int;
extern fn cublasLtMatrixLayoutDestroy(l: ?*anyopaque) c_int;
extern fn cublasLtMatmulPreferenceCreate(p: *?*anyopaque) c_int;
extern fn cublasLtMatmulPreferenceSetAttribute(p: ?*anyopaque, attr: c_int, buf: *const anyopaque, n: usize) c_int;
extern fn cublasLtMatmulPreferenceDestroy(p: ?*anyopaque) c_int;
extern fn cublasLtMatmulAlgoGetHeuristic(h: ?*anyopaque, d: ?*anyopaque, a: ?*anyopaque, b: ?*anyopaque,
    c: ?*anyopaque, dd: ?*anyopaque, pref: ?*anyopaque, want: c_int, res: *anyopaque, found: *c_int) c_int;
extern fn cublasLtMatmul(h: ?*anyopaque, d: ?*anyopaque, alpha: *const f32, A: ?*const anyopaque,
    la: ?*anyopaque, B: ?*const anyopaque, lb: ?*anyopaque, beta: *const f32, C: ?*const anyopaque,
    lc: ?*anyopaque, D: ?*anyopaque, ld: ?*anyopaque, algo: *const anyopaque, ws: ?*anyopaque,
    wsb: usize, stream: ?*anyopaque) c_int;

/// cublasLtMatmulHeuristicResult_t is opaque to us; only its size matters.
const HeuristicResult = extern struct { raw: [88]u8 align(8) };

var lt: ?*anyopaque = null;
var workspace: ?*anyopaque = null;
const WS_BYTES: usize = 32 << 20;

var last: [160]u8 = undefined;
var last_len: usize = 0;
pub fn lastError() []const u8 {
    return last[0..last_len];
}
fn fail(comptime what: []const u8, rc: c_int) error{Fp4} {
    const s = std.fmt.bufPrint(&last, "{s}: cuBLASLt status {d}", .{ what, rc }) catch last[0..];
    last_len = s.len;
    return error.Fp4;
}

/// One matmul shape, with its descriptors and chosen algorithm kept so the
/// heuristic is queried once rather than per call.
pub const Gemm = struct {
    desc: ?*anyopaque = null,
    la: ?*anyopaque = null,
    lb: ?*anyopaque = null,
    ld: ?*anyopaque = null,
    algo: HeuristicResult = undefined,
    M: i32 = 0,
    N: i32 = 0,
    K: i32 = 0,

    /// Same shape and same output layout as `linalg.linearFwd`:
    /// Y(M,N) = X(M,K) @ W(N,K)^T, with X and W K-major and Y written with
    /// leading dimension N - which is what the rest of the model expects.
    ///
    /// In cuBLASLt's column-major terms that is C(N,M) = W^T(N,K) X(K,M),
    /// so W is the transposed operand and the output is N x M with ld = N.
    /// Producing M x N instead - the arrangement that reads more naturally -
    /// yields the transpose, and comparing that against the reference gives
    /// a relative error of sqrt(2), the signature of two uncorrelated
    /// tensors rather than of a scale being off.
    pub fn init(M: i32, N: i32, K: i32, w_scale: *const anyopaque, x_scale: *const anyopaque) !Gemm {
        if (lt == null) {
            const rc = cublasLtCreate(&lt);
            if (rc != 0) return fail("cublasLtCreate", rc);
            // The workspace outlives every Gemm and is never freed; there is
            // one of it for the process.
            if (cudaMalloc(&workspace, WS_BYTES) != 0) return fail("workspace", -1);
        }
        var g = Gemm{ .M = M, .N = N, .K = K };
        var rc = cublasLtMatmulDescCreate(&g.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (rc != 0) return fail("MatmulDescCreate", rc);
        const ta: c_int = OP_T;
        const tb: c_int = OP_N;
        const mode: c_int = SCALE_VEC16_UE4M3;
        _ = cublasLtMatmulDescSetAttribute(g.desc, DESC_TRANSA, &ta, @sizeOf(c_int));
        _ = cublasLtMatmulDescSetAttribute(g.desc, DESC_TRANSB, &tb, @sizeOf(c_int));
        _ = cublasLtMatmulDescSetAttribute(g.desc, DESC_A_SCALE_MODE, &mode, @sizeOf(c_int));
        _ = cublasLtMatmulDescSetAttribute(g.desc, DESC_B_SCALE_MODE, &mode, @sizeOf(c_int));
        // Without these the heuristic answers NOT_SUPPORTED, whatever the
        // hardware can do.
        // The attribute is the pointer value itself, so what is handed over
        // is the address of a variable holding it.
        const ap: usize = @intFromPtr(w_scale);
        const bp: usize = @intFromPtr(x_scale);
        _ = cublasLtMatmulDescSetAttribute(g.desc, DESC_A_SCALE_POINTER, &ap, @sizeOf(usize));
        _ = cublasLtMatmulDescSetAttribute(g.desc, DESC_B_SCALE_POINTER, &bp, @sizeOf(usize));

        rc = cublasLtMatrixLayoutCreate(&g.la, CUDA_R_4F_E2M1, @intCast(K), @intCast(N), @intCast(K));
        if (rc != 0) return fail("layout W", rc);
        rc = cublasLtMatrixLayoutCreate(&g.lb, CUDA_R_4F_E2M1, @intCast(K), @intCast(M), @intCast(K));
        if (rc != 0) return fail("layout X", rc);
        rc = cublasLtMatrixLayoutCreate(&g.ld, CUDA_R_16BF, @intCast(N), @intCast(M), @intCast(N));
        if (rc != 0) return fail("layout Y", rc);

        var pref: ?*anyopaque = null;
        _ = cublasLtMatmulPreferenceCreate(&pref);
        defer _ = cublasLtMatmulPreferenceDestroy(pref);
        const ws: usize = WS_BYTES;
        _ = cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE, &ws, @sizeOf(usize));
        var found: c_int = 0;
        rc = cublasLtMatmulAlgoGetHeuristic(lt, g.desc, g.la, g.lb, g.ld, g.ld, pref, 1, &g.algo, &found);
        if (rc != 0 or found == 0) return fail("no NVFP4 algorithm for this shape", rc);
        return g;
    }

    /// `alpha` carries the two per-tensor scales. cuBLASLt applies the e4m3
    /// block scales and nothing else, so the outer fp32 scale that keeps
    /// those blocks inside e4m3's range has to be multiplied back in here -
    /// leaving it out makes the result too large by 1/(gA*gB), which for
    /// weights of this size is a factor of about 8e9.
    pub fn run(g: *const Gemm, W: *const anyopaque, X: *const anyopaque, Y: *anyopaque, alpha: f32) !void {
        const al: f32 = alpha;
        const be: f32 = 0;
        const rc = cublasLtMatmul(lt, g.desc, &al, W, g.la, X, g.lb, &be, Y, g.ld, Y, g.ld,
            &g.algo, workspace, WS_BYTES, null);
        if (rc != 0) return fail("cublasLtMatmul", rc);
    }

    pub fn deinit(g: *Gemm) void {
        _ = cublasLtMatrixLayoutDestroy(g.ld);
        _ = cublasLtMatrixLayoutDestroy(g.lb);
        _ = cublasLtMatrixLayoutDestroy(g.la);
        _ = cublasLtMatmulDescDestroy(g.desc);
    }
};
