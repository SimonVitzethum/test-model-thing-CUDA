//! Owned parameter storage and weight initialization (port of the
//! ParameterStore and the host_init/host_normal helpers of src/train.cu).
//! The random numbers are the same linear congruential sequence, in the same
//! order, so a model built from a given seed is bit-identical to the C++ one.
const std = @import("std");
const gpu = @import("gpu.zig");

pub const bf16 = u16;

/// One parameter: fp32 master, Adam moments, gradient and the bf16 working copy.
///
/// The moments and the master copy can be kept in bf16 instead (`mom_bf16`,
/// `master_bf16`), which halves the bytes the optimizer moves for them. The
/// pointers keep their fp32 type because they are device addresses that the
/// host never dereferences; `half_moments` and `half_master` say what the
/// memory behind them actually holds, and every reader has to ask.
///
/// A bf16 master is only ever given to a two-dimensional weight. The forward
/// pass reads those through the bf16 working copy alone, whereas it reads the
/// per-channel vectors - the decay, the norm scale and shift, the gate - out
/// of the master itself as fp32. Those vectors, and the three embedding
/// tables, are about one percent of the parameters, so keeping them fp32
/// costs nothing measurable and keeps every other kernel untouched.
pub const Par = struct {
    master: [*]f32,
    m: [*]f32,
    v: [*]f32,
    grad: [*]f32,
    work: [*]bf16,
    half_moments: bool = false,
    half_master: bool = false,
    /// Running average of the weights over training; only allocated when
    /// `wavg` is on, and never read by the training step itself.
    avg: ?[*]f32 = null,
    n: i64,
    /// Shape as the weight is stored, (out, in); 1 x n for the vectors.
    rows: i64 = 1,
    cols: i64 = 0,
};

pub const Store = struct {
    memory: gpu.Memory,
    values: std.ArrayList(Par) = .empty,
    gpa: std.mem.Allocator,
    /// Set before the parameters are built when `wavg` is on; every parameter
    /// then carries an averaged copy as well.
    want_avg: bool = false,
    /// Set before the parameters are built; the widths of the optimizer state.
    half_moments: bool = false,
    half_master: bool = false,

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
    /// A two-dimensional weight, which the Muon path can orthogonalize and
    /// which the forward pass only ever reads through its bf16 copy.
    pub fn add2d(s: *Store, rows: i64, cols: i64) !usize {
        const i = try s.addSized(rows * cols, s.half_master);
        s.values.items[i].rows = rows;
        s.values.items[i].cols = cols;
        return i;
    }
    /// A new parameter of n elements; moments and gradient start at zero.
    pub fn add(s: *Store, n: i64) !usize {
        return s.addSized(n, false);
    }
    fn addSized(s: *Store, n: i64, half_master: bool) !usize {
        if (n <= 0 or n > 2147483647) return error.ParameterTooLarge;
        const un: usize = @intCast(n);
        // A bf16 master is bit for bit the working copy the forward pass
        // reads, so the two share one buffer and the update writes it once.
        const master: [*]f32 = if (half_master)
            @ptrCast(@alignCast(try s.memory.allocT(bf16, un)))
        else
            try s.memory.allocT(f32, un);
        const p = Par{
            .master = master,
            .m = if (s.half_moments) @ptrCast(@alignCast(try s.memory.callocT(bf16, un))) else try s.memory.callocT(f32, un),
            .v = if (s.half_moments) @ptrCast(@alignCast(try s.memory.callocT(bf16, un))) else try s.memory.callocT(f32, un),
            .grad = try s.memory.callocT(f32, un),
            .work = if (half_master) @ptrCast(master) else try s.memory.allocT(bf16, un),
            .half_moments = s.half_moments,
            .half_master = half_master,
            .avg = if (s.want_avg)
                (if (half_master) @as([*]f32, @ptrCast(@alignCast(try s.memory.callocT(bf16, un)))) else try s.memory.callocT(f32, un))
            else
                null,
            .n = n,
            .cols = n,
        };
        try s.values.append(s.gpa, p);
        return s.values.items.len - 1;
    }
};

// ---- the host side of a bf16 master copy ----
// Initialization, checkpoints and the diagnostics all speak fp32, so the
// conversion happens in the staging buffer on the way past. It is
// round-to-nearest, not stochastic: these are one-off conversions of a value
// that is not being accumulated into, and stochastic rounding buys nothing
// there - it only matters where a small update would otherwise be lost.

/// Round to nearest even, the same arithmetic as the device's `f2bf`.
pub fn f2bfHost(x: f32) bf16 {
    const u: u32 = @bitCast(x);
    if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0;
    const bias: u32 = ((u >> 16) & 1) +% 0x7fff;
    return @truncate((u +% bias) >> 16);
}
pub fn bf2fHost(x: bf16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// Bytes one master (or averaged) copy of `p` occupies on the device.
pub fn masterBytes(p: Par) usize {
    return @as(usize, @intCast(p.n)) * @as(usize, if (p.half_master) 2 else 4);
}

/// Host fp32 -> the master copy, whatever width it has. Because a bf16
/// master is also the working copy, this leaves nothing more to do.
pub fn uploadMaster(gpa: std.mem.Allocator, p: Par, h: []const f32) !void {
    if (!p.half_master) return gpu.upload(p.master, std.mem.sliceAsBytes(h));
    const buf = try gpa.alloc(bf16, h.len);
    defer gpa.free(buf);
    for (h, buf) |x, *b| b.* = f2bfHost(x);
    try gpu.upload(p.master, std.mem.sliceAsBytes(buf));
}

/// The master copy -> host fp32, for the diagnostics and for sampling.
pub fn downloadMaster(gpa: std.mem.Allocator, p: Par, h: []f32) !void {
    if (!p.half_master) return gpu.download(std.mem.sliceAsBytes(h), p.master);
    const buf = try gpa.alloc(bf16, h.len);
    defer gpa.free(buf);
    try gpu.download(std.mem.sliceAsBytes(buf), p.master);
    for (buf, h) |b, *x| x.* = bf2fHost(b);
}

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
