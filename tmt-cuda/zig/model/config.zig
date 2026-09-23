//! The model configuration (port of src/config.h). One struct drives parsing,
//! the checkpoint header and validation; the field order is the order of the
//! stored text, so checkpoints stay interchangeable with the C++ build.
const std = @import("std");

pub const Error = error{Config};

var msg_buf: [256]u8 = undefined;
var msg_len: usize = 0;
/// Message of the last rejected key, value or constraint.
pub fn lastError() []const u8 {
    return msg_buf[0..msg_len];
}
fn fail(comptime fmt: []const u8, args: anytype) Error {
    const s = std.fmt.bufPrint(&msg_buf, fmt, args) catch msg_buf[0..];
    msg_len = s.len;
    return error.Config;
}

pub const Cfg = extern struct {
    dim: i32 = 256,
    layers: i32 = 4,
    experts: i32 = 1,
    topk: i32 = 1,
    batch: i32 = 8,
    seqlen: i32 = 128,
    gated: i32 = 1,
    half_min: f32 = 2,
    half_max: f32 = 512,
    lr: f32 = 5e-4,
    warmup: i32 = 200,
    decaysteps: i32 = 8000,
    minlr: f32 = 0.1,
    aux: f32 = 0.01,
    zloss: f32 = 0.001,
    latent: f32 = 0,
    ce: f32 = 1,
    @"var": f32 = 0,
    stop: f32 = 0,
    stopposw: f32 = 20,
    ematau: f32 = 0.99,
    gradclip: f32 = 1,
    maxcarry: i32 = 0,
    seed: i32 = 1234,
    traces: i32 = 0,
    trace_decay: f32 = 1,
    docsep: i32 = -1,
    dialog: i32 = 0,
    patch: i32 = 0,
    patch_lo: i32 = 2,
    patch_hi: i32 = 0,
    patch_max: i32 = 32,
    patch_decay: i32 = 0,
    accum: i32 = 1,
    mup: i32 = 0,
    mup_base: i32 = 256,
    muon: i32 = 0,
    muon_lr: f32 = 0.02,
    mtp: i32 = 0,
    mtp_weight: f32 = 0.3,
    wavg: f32 = 0,
    wavg_every: i32 = 8,
    mom_bf16: i32 = 0,
    master_bf16: i32 = 0,
    mem: i32 = 0,
    mem_len: i32 = 256,
    mem_heads: i32 = 4,
    mem_dh: i32 = 32,
    mem_every: i32 = 2,
    mem_rdim: i32 = 0,
    mla: i32 = 0,
    mla_heads: i32 = 4,
    mla_dh: i32 = 32,
    mla_L: i32 = 32,
    mla_R: i32 = 16,
    mla_cache: i32 = 4096,
    mla_every: i32 = 2,
    mla_cc: i32 = 256,
    mla_theta: f32 = 10000,
};

const fields = @typeInfo(Cfg).@"struct".fields;

extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

/// key=value, as the C++ build parses it: the whole value must be consumed
/// and the result must be finite.
pub fn set(c: *Cfg, key: []const u8, raw: []const u8) Error!void {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    inline for (fields) |f| {
        if (std.mem.eql(u8, key, f.name)) {
            if (f.type == i32) {
                const v = std.fmt.parseInt(i32, trimmed, 10) catch return fail("invalid value for {s}", .{key});
                @field(c, f.name) = v;
            } else {
                const v = std.fmt.parseFloat(f32, trimmed) catch return fail("invalid value for {s}", .{key});
                if (!std.math.isFinite(v)) return fail("invalid value for {s}", .{key});
                @field(c, f.name) = v;
            }
            return;
        }
    }
    return fail("unknown option: {s}", .{key});
}

/// The stored text: one key=value line per field, floats with nine significant
/// digits, exactly like the C++ stream with setprecision(9).
pub fn text(gpa: std.mem.Allocator, c: Cfg) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var buf: [64]u8 = undefined;
    inline for (fields) |f| {
        try out.appendSlice(gpa, f.name);
        try out.append(gpa, '=');
        const n = if (f.type == i32)
            snprintf(&buf, buf.len, "%d", @field(c, f.name))
        else
            snprintf(&buf, buf.len, "%.9g", @as(f64, @field(c, f.name)));
        try out.appendSlice(gpa, buf[0..@intCast(n)]);
        try out.append(gpa, '\n');
    }
    return out.toOwnedSlice(gpa);
}

/// Value of `key` in a config text.
pub fn value(t: []const u8, key: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, t, '\n');
    while (lines.next()) |line| {
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        if (std.mem.eql(u8, line[0..eq], key)) return line[eq + 1 ..];
    }
    return null;
}

/// Parse a whole config text (a checkpoint header); unknown keys are refused
/// by `set`, which keeps old checkpoints readable only through the relaxed
/// schema check in checkpoint.zig.
pub fn parse(t: []const u8) Error!Cfg {
    var c = Cfg{};
    var lines = std.mem.splitScalar(u8, t, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse return fail("invalid configuration line", .{});
        try set(&c, line[0..eq], line[eq + 1 ..]);
    }
    return c;
}

const INT_MAX: i64 = 2147483647;

fn require(ok: bool, comptime message: []const u8) Error!void {
    if (!ok) return fail(message, .{});
}

pub fn validate(c: Cfg) Error!void {
    try require(c.dim > 0 and c.layers > 0 and c.layers <= 256, "invalid model dimensions");
    try require(c.experts >= 1 and c.experts <= 16 and c.topk >= 1 and c.topk <= c.experts,
        "require 1 <= topk <= experts <= 16");
    try require(c.batch > 0 and c.seqlen > 0 and @as(i64, c.batch) * c.seqlen <= @divTrunc(INT_MAX, 256),
        "invalid batch/seqlen");
    try require(@as(i64, c.batch) * c.seqlen * c.dim <= INT_MAX and @as(i64, c.dim) * c.dim <= INT_MAX,
        "tensor exceeds supported index range");
    try require(c.gated == 0 or c.gated == 1, "gated must be 0 or 1");
    try require(c.half_min >= 1 and c.half_max >= c.half_min and c.half_max <= 100000,
        "require 1 <= half_min <= half_max <= 100000");
    try require(c.lr > 0 and c.warmup >= 0 and c.decaysteps > 0 and c.minlr >= 0 and c.minlr <= 1,
        "invalid learning-rate schedule");
    try require(c.ce > 0 and c.aux >= 0 and c.zloss >= 0 and c.latent >= 0 and c.@"var" >= 0 and c.stop >= 0,
        "loss weights must be nonnegative, ce must be positive");
    try require(c.stopposw > 0 and c.ematau >= 0 and c.ematau < 1 and c.gradclip >= 0 and c.maxcarry >= 0,
        "invalid optimizer/state configuration");
    try require(c.traces == 0 or c.traces == 1, "traces must be 0 or 1");
    try require(c.trace_decay > 0 and c.trace_decay <= 1, "require 0 < trace_decay <= 1");
    try require(c.docsep >= -1 and c.docsep <= 255, "docsep must be -1 (off) or a byte value");
    try require(c.dialog == 0 or c.dialog == 1, "dialog must be 0 or 1");
    try require(c.accum >= 1 and c.accum <= 1024, "require 1 <= accum <= 1024");
    try require((c.mup == 0 or c.mup == 1) and c.mup_base > 0, "mup must be 0 or 1 with mup_base > 0");
    try require(c.patch >= -1 and c.patch_lo >= 0 and c.patch_hi >= 0 and
        c.patch_lo + c.patch_hi <= c.layers and c.patch_max > 1 and
        (c.patch_decay == 0 or c.patch_decay == 1),
        "invalid patching configuration");
    if (c.patch_hi > 0) {
        try require(c.patch != 0, "patch_hi needs a patch rule (patch=N or patch=-1)");
        try require(c.patch < 0 or @rem(c.seqlen, c.patch) == 0,
            "a fixed patch stride must divide seqlen");
        // The latent cache counts its positions in bytes, which a layer running
        // per patch would break; keep the two apart until that is worked out.
        if (c.mla != 0) {
            var l = c.patch_lo;
            while (l < c.patch_lo + c.patch_hi) : (l += 1)
                try require(@rem(l, c.mla_every) != 0, "MLA layers cannot run at patch rate yet");
        }
    } else try require(c.patch == 0, "patch=N needs patch_hi > 0");
    try require((c.muon == 0 or c.muon == 1) and c.muon_lr > 0, "muon must be 0 or 1 with muon_lr > 0");
    try require(c.mtp >= 0 and c.mtp <= 7 and c.mtp_weight >= 0,
        "require 0 <= mtp <= 7 and mtp_weight >= 0");
    try require(c.wavg >= 0 and c.wavg < 1 and c.wavg_every >= 1,
        "require 0 <= wavg < 1 (0 is off) and wavg_every >= 1");
    try require((c.mom_bf16 == 0 or c.mom_bf16 == 1) and (c.master_bf16 == 0 or c.master_bf16 == 1),
        "mom_bf16 and master_bf16 must be 0 or 1");
    try require(c.mem == 0 or c.mem == 1, "mem must be 0 or 1");
    if (c.mem != 0) {
        try require(c.mem_len > 0 and c.mem_len <= 4096 and c.mem_heads > 0 and c.mem_dh > 0 and
            c.mem_every >= 1 and c.mem_every <= c.layers and c.mem_rdim >= 0 and c.mem_rdim <= 1024,
            "invalid memory configuration");
        try require(@as(i64, c.batch) * c.mem_len * c.dim <= INT_MAX and
            @as(i64, c.batch) * c.mem_heads * c.seqlen * c.mem_len <= INT_MAX and
            @as(i64, c.mem_heads) * c.mem_dh * c.dim <= INT_MAX, "memory exceeds supported index range");
    }
    try require(c.mla == 0 or c.mla == 1, "mla must be 0 or 1");
    if (c.mla != 0) {
        try require(c.mla_heads > 0 and c.mla_dh > 0 and c.mla_L > 0 and c.mla_R > 0 and @rem(c.mla_R, 2) == 0,
            "invalid MLA dimensions (RoPE dimension must be even)");
        try require(c.mla_cache >= c.seqlen and c.mla_cc > 0 and c.mla_every > 0 and c.mla_theta > 1,
            "invalid MLA cache/chunk configuration");
        try require(@as(i64, c.mla_heads) * (@as(i64, c.mla_dh) + c.mla_R) <= @divTrunc(INT_MAX, c.dim) and
            @as(i64, c.dim) * c.mla_L <= INT_MAX and @as(i64, c.dim) * c.mla_R <= INT_MAX and
            @as(i64, c.batch) * c.seqlen * c.mla_heads <= @divTrunc(INT_MAX, @max(c.mla_dh, c.mla_R)),
            "MLA projections exceed supported index range");
        const bh = @as(i64, c.batch) * c.mla_heads;
        try require(bh <= @divTrunc(INT_MAX, c.seqlen) and bh * c.seqlen <= @divTrunc(INT_MAX, c.mla_cc) and
            @as(i64, c.batch) * c.mla_cache <= @divTrunc(INT_MAX, @max(c.mla_L, c.mla_R)),
            "MLA cache/workspace exceeds supported index range");
    }
}

test "defaults survive a round trip through the text form" {
    const gpa = std.testing.allocator;
    const t = try text(gpa, .{});
    defer gpa.free(t);
    const back = try parse(t);
    try std.testing.expectEqual(@as(i32, 256), back.dim);
    try std.testing.expectEqual(@as(f32, 5e-4), back.lr);
    // Nine significant digits of the float widened to double, like the C++ stream.
    try std.testing.expect(std.mem.indexOf(u8, t, "lr=0.000500000024\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, t, "ematau=0.99000001\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, t, "mla_theta=10000\n") != null);
}
