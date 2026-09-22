// Small host helpers: formatted stdout/stderr through libc, key=value
// arguments, a monotonic clock and the deterministic initialization RNG.
const std = @import("std");
pub const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
    @cInclude("time.h");
    @cInclude("unistd.h");
});

pub const gpa = std.heap.c_allocator;

pub fn print(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.allocPrint(gpa, fmt, args) catch return;
    defer gpa.free(s);
    _ = c.fwrite(s.ptr, 1, s.len, c.stdout());
    _ = c.fflush(c.stdout());
}
pub fn eprint(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.allocPrint(gpa, fmt, args) catch return;
    defer gpa.free(s);
    _ = c.fwrite(s.ptr, 1, s.len, c.stderr());
}
/// Prints "error: msg" to stderr and exits with status 1.
pub fn die(comptime fmt: []const u8, args: anytype) noreturn {
    eprint("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

/// printf-style number formatting (%g, %.9g, ...) for output compatible with the CUDA tools.
pub fn cfmt(buf: []u8, comptime spec: [*:0]const u8, v: f64) []const u8 {
    const n = c.snprintf(buf.ptr, buf.len, spec, v);
    return buf[0..@intCast(n)];
}

pub fn now() f64 {
    var ts: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.tv_sec)) + @as(f64, @floatFromInt(ts.tv_nsec)) * 1e-9;
}

pub const KV = struct { key: []const u8, value: []const u8 };
pub fn splitKV(arg: []const u8) ?KV {
    const eq = std.mem.indexOfScalar(u8, arg, '=') orelse return null;
    return .{ .key = arg[0..eq], .value = arg[eq + 1 ..] };
}
pub fn parseF32(s: []const u8) f32 {
    return std.fmt.parseFloat(f32, s) catch die("invalid number: {s}", .{s});
}
pub fn parseInt(comptime T: type, s: []const u8) T {
    return std.fmt.parseInt(T, s, 10) catch die("invalid integer: {s}", .{s});
}
/// Strict nonnegative integer option (steps=, saveevery=).
pub fn parseCount(s: []const u8) u64 {
    const v = std.fmt.parseInt(i64, s, 10) catch die("invalid nonnegative integer", .{});
    if (v < 0) die("invalid nonnegative integer", .{});
    return @intCast(v);
}

// Same LCG as the CUDA host initializer, so both builds start from identical weights.
pub fn lcg(s: *u32) u32 {
    s.* = s.* *% 1664525 +% 1013904223;
    return s.*;
}
pub fn hostUniform(h: []f32, a: f32, b: f32, s: *u32) void {
    // Fused multiply-add: the CUDA host initializer is compiled with FP
    // contraction, so the same seed must produce bit-identical weights.
    for (h) |*v| v.* = @mulAdd(f32, b - a, @as(f32, @floatFromInt(lcg(s) >> 9)) * (1.0 / 8388608.0), a);
}
pub fn hostNormal(h: []f32, stdev: f32, s: *u32) void {
    var i: usize = 0;
    while (i < h.len) {
        const u1_ = @as(f32, @floatFromInt((lcg(s) >> 9) + 1)) * (1.0 / 8388609.0);
        const u2_ = @as(f32, @floatFromInt((lcg(s) >> 9) + 1)) * (1.0 / 8388609.0);
        const r = @sqrt(-2 * @log(u1_));
        const t = 6.2831853 * u2_;
        h[i] = r * @cos(t) * stdev;
        i += 1;
        if (i < h.len) {
            h[i] = r * @sin(t) * stdev;
            i += 1;
        }
    }
}
