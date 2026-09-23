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
extern fn cuEventCreate(event: *?*anyopaque, flags: c_uint) c_int;
extern fn cuEventRecord(event: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn cuEventSynchronize(event: ?*anyopaque) c_int;
extern fn cuEventElapsedTime(ms: *f32, start: ?*anyopaque, end: ?*anyopaque) c_int;
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
        return .{ .f = f, .name = std.mem.span(name) };
    }
};

/// Per-kernel timing, switched on with TMT_PROFILE=1. Events are recorded
/// around every launch and only read once the window has run, so the launches
/// stay pipelined and the numbers are the time the GPU really spent, not the
/// time the host spent issuing work.
pub const Profile = struct {
    pub const Entry = struct { name: []const u8, ms: f64 = 0, calls: u64 = 0 };
    const pool_size = 8192;
    var entries: [256]Entry = undefined;
    var count: usize = 0;
    var starts: [pool_size]?*anyopaque = @splat(null);
    var stops: [pool_size]?*anyopaque = @splat(null);
    var names: [pool_size][]const u8 = undefined;
    var used: usize = 0;
    pub var on: bool = false;

    pub fn enable() void {
        on = true;
    }
    fn begin(name: []const u8) void {
        if (!on or used == pool_size) return;
        if (starts[used] == null) {
            _ = cuEventCreate(&starts[used], 0);
            _ = cuEventCreate(&stops[used], 0);
        }
        names[used] = name;
        _ = cuEventRecord(starts[used], null);
    }
    fn end() void {
        if (!on or used == pool_size) return;
        _ = cuEventRecord(stops[used], null);
        used += 1;
    }
    /// Reads the pool; called once per window, when the host waits anyway.
    pub fn collect() void {
        if (!on or used == 0) return;
        _ = cuEventSynchronize(stops[used - 1]);
        for (0..used) |i| {
            var ms: f32 = 0;
            _ = cuEventElapsedTime(&ms, starts[i], stops[i]);
            add(names[i], ms);
        }
        used = 0;
    }
    /// Used by the cuBLAS wrappers, which record around their own call.
    pub fn beginNamed(name: []const u8) void {
        begin(name);
    }
    pub fn endNamed() void {
        end();
    }
    pub fn add(name: []const u8, ms: f32) void {
        for (entries[0..count]) |*e| if (std.mem.eql(u8, e.name, name)) {
            e.ms += ms;
            e.calls += 1;
            return;
        };
        if (count == entries.len) return;
        entries[count] = .{ .name = name, .ms = ms, .calls = 1 };
        count += 1;
    }
    /// The table, longest first.
    pub fn report(out: *std.Io.Writer) !void {
        if (count == 0) return;
        var total: f64 = 0;
        for (entries[0..count]) |e| total += e.ms;
        std.mem.sort(Entry, entries[0..count], {}, struct {
            fn less(_: void, a: Entry, b: Entry) bool {
                return a.ms > b.ms;
            }
        }.less);
        try out.print("\n{s:<24}{s:>10}{s:>9}{s:>10}{s:>12}\n", .{ "kernel", "ms", "share", "calls", "us/call" });
        for (entries[0..count]) |e| try out.print("{s:<24}{d:>10.1}{d:>8.1}%{d:>10}{d:>12.1}\n", .{
            e.name, e.ms, 100 * e.ms / total, e.calls, 1000 * e.ms / @as(f64, @floatFromInt(e.calls)),
        });
        try out.print("{s:<24}{d:>10.1}\n", .{ "total on the device", total });
    }
};

pub const Kernel = struct {
    f: ?*anyopaque,
    name: []const u8 = "",

    /// Launch with `grid` blocks of `block` threads; `args` is a tuple of the
    /// kernel's arguments, in order.
    pub fn launch(k: Kernel, grid: u32, block: u32, args: anytype) Error!void {
        return k.launchGrid(grid, 1, block, args);
    }
    /// Two-dimensional grid, as the recurrence kernels use it (stream, channel).
    pub fn launchGrid(k: Kernel, gx: u32, gy: u32, block: u32, args: anytype) Error!void {
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
        // Launches are asynchronous; a failure surfaces at the next
        // synchronization, which every window does through its loss readback.
        Profile.begin(k.name);
        try check(cuLaunchKernel(k.f, gx, gy, 1, block, 1, 1, 0, null, &storage, null), "cuLaunchKernel");
        Profile.end();
    }
};
