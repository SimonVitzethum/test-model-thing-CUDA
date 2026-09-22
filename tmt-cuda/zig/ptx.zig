//! Loading and launching the Zig kernels (zig/kernels/*.zig, compiled to PTX
//! by build.zig) through the CUDA driver API. The driver shares the runtime's
//! primary context, so device pointers from cudaMalloc work unchanged.
const std = @import("std");

extern fn cuInit(flags: c_uint) c_int;
extern fn cuCtxGetCurrent(ctx: *?*anyopaque) c_int;
extern fn cuDevicePrimaryCtxRetain(ctx: *?*anyopaque, dev: c_int) c_int;
extern fn cuCtxSetCurrent(ctx: ?*anyopaque) c_int;
extern fn cuDeviceGet(dev: *c_int, ordinal: c_int) c_int;
extern fn cuModuleLoadData(module: *?*anyopaque, image: [*]const u8) c_int;
extern fn cuModuleGetFunction(f: *?*anyopaque, module: ?*anyopaque, name: [*:0]const u8) c_int;
extern fn cuLaunchKernel(f: ?*anyopaque, gx: c_uint, gy: c_uint, gz: c_uint,
                         bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint,
                         stream: ?*anyopaque, params: [*]?*anyopaque, extra: ?*anyopaque) c_int;
extern fn cuCtxSynchronize() c_int;
extern fn cuGetErrorString(code: c_int, str: *[*:0]const u8) c_int;

pub const Error = error{Cuda};

var last: [256]u8 = undefined;
var last_len: usize = 0;
pub fn lastError() []const u8 {
    return last[0..last_len];
}
fn check(rc: c_int, what: []const u8) Error!void {
    if (rc == 0) return;
    var msg: [*:0]const u8 = "unknown";
    _ = cuGetErrorString(rc, &msg);
    const s = std.fmt.bufPrint(&last, "{s}: {s}", .{ what, std.mem.span(msg) }) catch last[0..];
    last_len = s.len;
    return error.Cuda;
}

/// The compiled kernel module; the PTX is embedded in the binary.
pub const Module = struct {
    handle: ?*anyopaque,

    pub fn load(ptx: [:0]const u8) Error!Module {
        try check(cuInit(0), "cuInit");
        var ctx: ?*anyopaque = null;
        try check(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
        if (ctx == null) { // no runtime call has created the primary context yet
            var dev: c_int = 0;
            try check(cuDeviceGet(&dev, 0), "cuDeviceGet");
            try check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
            try check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
        }
        var m: ?*anyopaque = null;
        try check(cuModuleLoadData(&m, ptx.ptr), "cuModuleLoadData");
        return .{ .handle = m };
    }
    pub fn get(m: Module, name: [*:0]const u8) Error!Kernel {
        var f: ?*anyopaque = null;
        try check(cuModuleGetFunction(&f, m.handle, name), "cuModuleGetFunction");
        return .{ .f = f };
    }
};

pub const Kernel = struct {
    f: ?*anyopaque,

    /// Launch with `grid` blocks of `block` threads; `args` is a tuple of the
    /// kernel's arguments, in order.
    pub fn launch(k: Kernel, grid: u32, block: u32, args: anytype) Error!void {
        // The driver copies each argument by the size in the kernel's
        // signature, so it gets a pointer to the value itself.
        const fields = @typeInfo(@TypeOf(args)).@"struct".fields;
        comptime var types: [fields.len]type = undefined;
        inline for (fields, 0..) |f, i| types[i] = f.type;
        var copy: std.meta.Tuple(&types) = undefined; // runtime storage, no comptime fields
        var storage: [fields.len]?*anyopaque = undefined;
        inline for (fields, 0..) |f, i| {
            copy[i] = @field(args, f.name);
            storage[i] = @ptrCast(&copy[i]);
        }
        try check(cuLaunchKernel(k.f, grid, 1, 1, block, 1, 1, 0, null, &storage, null), "cuLaunchKernel");
        try check(cuCtxSynchronize(), "cuCtxSynchronize");
    }
};
