//! The CUDA runtime as the host side needs it: device memory, copies and the
//! kernel module. Kernels themselves live in zig/kernels/ and are launched
//! through the driver API (zig/ptx.zig), which shares this runtime's primary
//! context, so the pointers here work for both.
const std = @import("std");
const ptx = @import("../ptx.zig");

pub const Error = error{Cuda};

extern fn cudaMalloc(p: *?*anyopaque, n: usize) c_int;
extern fn cudaFree(p: ?*anyopaque) c_int;
extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, n: usize, kind: c_int) c_int;
extern fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, n: usize, kind: c_int, stream: ?*anyopaque) c_int;
extern fn cudaMemset(p: ?*anyopaque, value: c_int, n: usize) c_int;
extern fn cudaMemsetAsync(p: ?*anyopaque, value: c_int, n: usize, stream: ?*anyopaque) c_int;
extern fn cudaDeviceSynchronize() c_int;
extern fn cudaGetLastError() c_int;
extern fn cudaGetErrorString(code: c_int) [*:0]const u8;
extern fn cudaSetDeviceFlags(flags: c_uint) c_int;

const HostToDevice: c_int = 1;
const DeviceToHost: c_int = 2;
const DeviceToDevice: c_int = 3;

var last: [256]u8 = undefined;
var last_len: usize = 0;
/// Message of the last failed call, for the CLI's error line.
pub fn lastError() []const u8 {
    if (last_len == 0) return ptx.lastError();
    return last[0..last_len];
}
fn check(rc: c_int, what: []const u8) Error!void {
    if (rc == 0) return;
    const s = std.fmt.bufPrint(&last, "{s}: {s}", .{ what, std.mem.span(cudaGetErrorString(rc)) }) catch last[0..];
    last_len = s.len;
    return error.Cuda;
}

/// How the host thread waits for the GPU; set before the first CUDA call.
/// TMT_CUDA_WAIT=yield (default) polls but yields the core, block sleeps
/// (about a third of a core, ~5% slower), spin is CUDA's busy wait.
pub fn setWaitMode(env: ?[]const u8) void {
    const ScheduleSpin: c_uint = 1;
    const ScheduleYield: c_uint = 2;
    const ScheduleBlockingSync: c_uint = 4;
    var flag = ScheduleYield;
    if (env) |v| {
        if (std.mem.eql(u8, v, "spin")) flag = ScheduleSpin;
        if (std.mem.eql(u8, v, "block")) flag = ScheduleBlockingSync;
    }
    _ = cudaSetDeviceFlags(flag);
}

pub fn synchronize() Error!void {
    try check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
/// Fails if any kernel launch since the last check was rejected.
pub fn checkLaunch() Error!void {
    try check(cudaGetLastError(), "kernel launch");
}

pub fn upload(dst: anytype, src: []const u8) Error!void {
    try check(cudaMemcpy(@ptrCast(dst), src.ptr, src.len, HostToDevice), "cudaMemcpy to device");
}
pub fn download(dst: []u8, src: anytype) Error!void {
    try check(cudaMemcpy(dst.ptr, @ptrCast(src), dst.len, DeviceToHost), "cudaMemcpy from device");
}
pub fn copyDevice(dst: anytype, src: anytype, bytes: usize) Error!void {
    try check(cudaMemcpy(@ptrCast(dst), @ptrCast(src), bytes, DeviceToDevice), "cudaMemcpy device to device");
}
pub fn zero(p: anytype, bytes: usize) Error!void {
    try check(cudaMemset(@ptrCast(p), 0, bytes), "cudaMemset");
}
pub fn zeroAsync(p: anytype, bytes: usize) Error!void {
    try check(cudaMemsetAsync(@ptrCast(p), 0, bytes, null), "cudaMemsetAsync");
}

/// Device memory owned by one model; freed together.
pub const Memory = struct {
    blocks: std.ArrayList(*anyopaque) = .empty,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator) Memory {
        return .{ .gpa = gpa };
    }
    /// `bytes` of device memory, remembered for the final free.
    pub fn alloc(m: *Memory, bytes: usize) !*anyopaque {
        var p: ?*anyopaque = null;
        try check(cudaMalloc(&p, @max(bytes, 1)), "cudaMalloc");
        try m.blocks.append(m.gpa, p.?);
        return p.?;
    }
    pub fn allocT(m: *Memory, comptime T: type, n: usize) ![*]T {
        return @ptrCast(@alignCast(try m.alloc(n * @sizeOf(T))));
    }
    /// Zero-initialized device memory.
    pub fn callocT(m: *Memory, comptime T: type, n: usize) ![*]T {
        const p = try m.allocT(T, n);
        try zero(p, n * @sizeOf(T));
        return p;
    }
    pub fn deinit(m: *Memory) void {
        for (m.blocks.items) |p| _ = cudaFree(p);
        m.blocks.deinit(m.gpa);
    }
};

/// The compiled kernels, looked up by name once and then reused.
pub const Kernels = struct {
    module: ptx.Module,
    names: std.StringHashMap(ptx.Kernel),

    pub fn load(gpa: std.mem.Allocator, image: [:0]const u8) !Kernels {
        return .{ .module = try ptx.Module.load(image), .names = std.StringHashMap(ptx.Kernel).init(gpa) };
    }
    pub fn get(k: *Kernels, comptime name: [:0]const u8) !ptx.Kernel {
        if (k.names.get(name)) |f| return f;
        const f = try k.module.get(name.ptr);
        try k.names.put(name, f);
        return f;
    }
};
