//! What a CUDA kernel needs from the language: thread indices, the bf16
//! working type and the libdevice math functions. The names match what nvcc
//! emits for the same source, so a Zig kernel and its C++ original produce
//! identical PTX instructions.
const std = @import("std");

// ---- thread and block indices (NVVM special registers) ----
extern fn @"llvm.nvvm.read.ptx.sreg.tid.x"() u32;
extern fn @"llvm.nvvm.read.ptx.sreg.tid.y"() u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.x"() u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ctaid.y"() u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() u32;
extern fn @"llvm.nvvm.read.ptx.sreg.ntid.y"() u32;
extern fn @"llvm.nvvm.read.ptx.sreg.nctaid.x"() u32;
extern fn @"llvm.nvvm.barrier0"() void;

pub inline fn threadIdxX() u32 {
    return @"llvm.nvvm.read.ptx.sreg.tid.x"();
}
pub inline fn threadIdxY() u32 {
    return @"llvm.nvvm.read.ptx.sreg.tid.y"();
}
pub inline fn blockIdxX() u32 {
    return @"llvm.nvvm.read.ptx.sreg.ctaid.x"();
}
pub inline fn blockIdxY() u32 {
    return @"llvm.nvvm.read.ptx.sreg.ctaid.y"();
}
pub inline fn blockDimX() u32 {
    return @"llvm.nvvm.read.ptx.sreg.ntid.x"();
}
pub inline fn blockDimY() u32 {
    return @"llvm.nvvm.read.ptx.sreg.ntid.y"();
}
pub inline fn gridDimX() u32 {
    return @"llvm.nvvm.read.ptx.sreg.nctaid.x"();
}
pub inline fn syncThreads() void {
    @"llvm.nvvm.barrier0"();
}
/// Global thread index, the `blockIdx.x * blockDim.x + threadIdx.x` of every kernel.
pub inline fn globalIdX() u32 {
    return blockIdxX() * blockDimX() + threadIdxX();
}

// ---- working precision ----
/// bf16 as it is stored on the device: the upper 16 bits of an f32.
pub const bf16 = u16;

pub inline fn bf2f(x: bf16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
/// __float2bfloat16_rn: round to nearest even, quiet NaN preserved.
pub inline fn f2bf(x: f32) bf16 {
    const u: u32 = @bitCast(x);
    if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0; // NaN
    const bias: u32 = ((u >> 16) & 1) +% 0x7fff;
    return @truncate((u +% bias) >> 16);
}

// ---- math (libdevice, the same functions nvcc calls for expf and friends) ----
pub extern fn __nv_expf(x: f32) f32;
pub extern fn __nv_logf(x: f32) f32;
pub extern fn __nv_powf(x: f32, y: f32) f32;
pub extern fn __nv_rsqrtf(x: f32) f32;
pub extern fn __nv_fmaxf(x: f32, y: f32) f32;
pub extern fn __nv_fminf(x: f32, y: f32) f32;
pub extern fn __nv_fabsf(x: f32) f32;
pub extern fn __nv_tanhf(x: f32) f32;

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + __nv_expf(-x));
}
pub inline fn silu(x: f32) f32 {
    return x * sigmoid(x);
}
pub inline fn siluBwd(x: f32, dy: f32) f32 {
    const s = sigmoid(x);
    return dy * s * (1.0 + x * (1.0 - s));
}

// ---- memory ----
/// A pointer into device global memory.
pub fn Global(comptime T: type) type {
    return [*]addrspace(.global) T;
}
pub fn ConstGlobal(comptime T: type) type {
    return [*]addrspace(.global) const T;
}
/// atomicAdd on global memory (device scope, relaxed, like CUDA's).
pub inline fn atomicAddF32(p: *addrspace(.global) f32, v: f32) void {
    _ = @atomicRmw(f32, p, .Add, v, .monotonic);
}
pub inline fn atomicAddI32(p: *addrspace(.global) i32, v: i32) i32 {
    return @atomicRmw(i32, p, .Add, v, .monotonic);
}
