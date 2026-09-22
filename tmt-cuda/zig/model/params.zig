//! Owned parameter storage and weight initialization (port of the
//! ParameterStore and the host_init/host_normal helpers of src/train.cu).
//! The random numbers are the same linear congruential sequence, in the same
//! order, so a model built from a given seed is bit-identical to the C++ one.
const std = @import("std");
const gpu = @import("gpu.zig");

pub const bf16 = u16;

/// One parameter: fp32 master, Adam moments, gradient and the bf16 working copy.
pub const Par = struct {
    master: [*]f32,
    m: [*]f32,
    v: [*]f32,
    grad: [*]f32,
    work: [*]bf16,
    n: i64,
    /// Shape as the weight is stored, (out, in); 1 x n for the vectors.
    rows: i64 = 1,
    cols: i64 = 0,
};

pub const Store = struct {
    memory: gpu.Memory,
    values: std.ArrayList(Par) = .empty,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator) Store {
        return .{ .memory = gpu.Memory.init(gpa), .gpa = gpa };
    }
    pub fn deinit(s: *Store) void {
        s.values.deinit(s.gpa);
        s.memory.deinit();
    }
    pub fn at(s: *const Store, i: usize) Par {
        return s.values.items[i];
    }
    /// A two-dimensional weight, which the Muon path can orthogonalize.
    pub fn add2d(s: *Store, rows: i64, cols: i64) !usize {
        const i = try s.add(rows * cols);
        s.values.items[i].rows = rows;
        s.values.items[i].cols = cols;
        return i;
    }
    /// A new parameter of n elements; moments and gradient start at zero.
    pub fn add(s: *Store, n: i64) !usize {
        if (n <= 0 or n > 2147483647) return error.ParameterTooLarge;
        const un: usize = @intCast(n);
        const p = Par{
            .master = try s.memory.allocT(f32, un),
            .m = try s.memory.callocT(f32, un),
            .v = try s.memory.callocT(f32, un),
            .grad = try s.memory.callocT(f32, un),
            .work = try s.memory.allocT(bf16, un),
            .n = n,
            .cols = n,
        };
        try s.values.append(s.gpa, p);
        return s.values.items.len - 1;
    }
};

/// Uniform in [a, b), from the same generator the C++ build uses.
pub fn hostInit(h: []f32, a: f32, b: f32, seed: *u32) void {
    for (h) |*v| {
        seed.* = seed.* *% 1664525 +% 1013904223;
        v.* = a + (b - a) * (@as(f32, @floatFromInt(seed.* >> 9)) * (1.0 / 8388608.0));
    }
}

// The host math goes through libm, like the C++ build, so the initial weights
// come out bit-identical.
pub extern "c" fn sqrtf(x: f32) f32;
pub extern "c" fn logf(x: f32) f32;
pub extern "c" fn cosf(x: f32) f32;
pub extern "c" fn sinf(x: f32) f32;
pub extern "c" fn expf(x: f32) f32;
pub extern "c" fn powf(x: f32, y: f32) f32;

/// Normal deviates by Box-Muller, in pairs, as in the C++ build.
pub fn hostNormal(h: []f32, std_dev: f32, seed: *u32) void {
    var i: usize = 0;
    while (i < h.len) {
        seed.* = seed.* *% 1664525 +% 1013904223;
        const ua: f32 = @as(f32, @floatFromInt((seed.* >> 9) + 1)) * (1.0 / 8388609.0);
        seed.* = seed.* *% 1664525 +% 1013904223;
        const ub: f32 = @as(f32, @floatFromInt((seed.* >> 9) + 1)) * (1.0 / 8388609.0);
        const r = sqrtf(-2.0 * logf(ua));
        const t = 6.2831853 * ub;
        h[i] = r * cosf(t) * std_dev;
        i += 1;
        if (i < h.len) {
            h[i] = r * sinf(t) * std_dev;
            i += 1;
        }
    }
}
