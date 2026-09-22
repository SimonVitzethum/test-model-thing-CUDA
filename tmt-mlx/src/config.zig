// One schema drives parsing, validation and checkpoint metadata. The field
// order and the text format (`key=value\n`, floats as %.9g) match the CUDA
// version byte for byte, so V3 checkpoints are interchangeable.
const std = @import("std");
const cstd = @cImport(@cInclude("stdio.h"));

pub const Cfg = struct {
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

pub const Error = error{ InvalidValue, UnknownOption, InvalidConfig };
pub var last_error: []const u8 = "";

fn fail(comptime msg: []const u8) Error {
    last_error = msg;
    return error.InvalidConfig;
}

pub fn set(c: *Cfg, key: []const u8, value: []const u8) Error!void {
    inline for (@typeInfo(Cfg).@"struct".fields) |f| {
        if (std.mem.eql(u8, key, f.name)) {
            if (f.type == i32) {
                @field(c, f.name) = std.fmt.parseInt(i32, value, 10) catch {
                    last_error = "invalid value";
                    return error.InvalidValue;
                };
            } else {
                const v = std.fmt.parseFloat(f32, value) catch {
                    last_error = "invalid value";
                    return error.InvalidValue;
                };
                if (!std.math.isFinite(v)) return error.InvalidValue;
                @field(c, f.name) = v;
            }
            return;
        }
    }
    last_error = "unknown option";
    return error.UnknownOption;
}

pub fn isKey(key: []const u8) bool {
    inline for (@typeInfo(Cfg).@"struct".fields) |f| if (std.mem.eql(u8, key, f.name)) return true;
    return false;
}

/// Canonical text form; caller frees.
pub fn text(alloc: std.mem.Allocator, c: Cfg) []u8 {
    var out: std.ArrayList(u8) = .empty;
    inline for (@typeInfo(Cfg).@"struct".fields) |f| {
        out.appendSlice(alloc, f.name) catch unreachable;
        out.append(alloc, '=') catch unreachable;
        var buf: [64]u8 = undefined;
        const n: usize = if (f.type == i32)
            @intCast(cstd.snprintf(&buf, buf.len, "%d", @as(c_int, @field(c, f.name))))
        else
            @intCast(cstd.snprintf(&buf, buf.len, "%.9g", @as(f64, @field(c, f.name))));
        out.appendSlice(alloc, buf[0..n]) catch unreachable;
        out.append(alloc, '\n') catch unreachable;
    }
    return out.toOwnedSlice(alloc) catch unreachable;
}

pub fn equal(a: Cfg, b: Cfg) bool {
    const ta = text(std.heap.c_allocator, a);
    defer std.heap.c_allocator.free(ta);
    const tb = text(std.heap.c_allocator, b);
    defer std.heap.c_allocator.free(tb);
    return std.mem.eql(u8, ta, tb);
}

pub fn validate(c: Cfg) Error!void {
    const max_i: i64 = std.math.maxInt(i32);
    if (!(c.dim > 0 and c.layers > 0 and c.layers <= 256)) return fail("invalid model dimensions");
    if (!(c.experts >= 1 and c.experts <= 16 and c.topk >= 1 and c.topk <= c.experts))
        return fail("require 1 <= topk <= experts <= 16");
    if (!(c.batch > 0 and c.seqlen > 0 and @as(i64, c.batch) * c.seqlen <= @divTrunc(max_i, 256)))
        return fail("invalid batch/seqlen");
    if (!(@as(i64, c.batch) * c.seqlen * c.dim <= max_i and @as(i64, c.dim) * c.dim <= max_i))
        return fail("tensor exceeds supported index range");
    if (!(c.gated == 0 or c.gated == 1)) return fail("gated must be 0 or 1");
    if (!(c.half_min >= 1 and c.half_max >= c.half_min and c.half_max <= 100000))
        return fail("require 1 <= half_min <= half_max <= 100000");
    if (!(c.lr > 0 and c.warmup >= 0 and c.decaysteps > 0 and c.minlr >= 0 and c.minlr <= 1))
        return fail("invalid learning-rate schedule");
    if (!(c.ce > 0 and c.aux >= 0 and c.zloss >= 0 and c.latent >= 0 and c.@"var" >= 0 and c.stop >= 0))
        return fail("loss weights must be nonnegative, ce must be positive");
    if (!(c.stopposw > 0 and c.ematau >= 0 and c.ematau < 1 and c.gradclip >= 0 and c.maxcarry >= 0))
        return fail("invalid optimizer/state configuration");
    if (!(c.traces == 0 or c.traces == 1)) return fail("traces must be 0 or 1");
    if (!(c.trace_decay > 0 and c.trace_decay <= 1)) return fail("require 0 < trace_decay <= 1");
    if (!(c.docsep >= -1 and c.docsep <= 255)) return fail("docsep must be -1 (off) or a byte value");
    if (!(c.dialog == 0 or c.dialog == 1)) return fail("dialog must be 0 or 1");
    if (!(c.mla == 0 or c.mla == 1)) return fail("mla must be 0 or 1");
    if (c.mla == 1) {
        if (!(c.mla_heads > 0 and c.mla_dh > 0 and c.mla_L > 0 and c.mla_R > 0 and @mod(c.mla_R, 2) == 0))
            return fail("invalid MLA dimensions (RoPE dimension must be even)");
        if (!(c.mla_cache >= c.seqlen and c.mla_cc > 0 and c.mla_every > 0 and c.mla_theta > 1))
            return fail("invalid MLA cache/chunk configuration");
        if (!(@as(i64, c.mla_heads) * (@as(i64, c.mla_dh) + c.mla_R) <= @divTrunc(max_i, c.dim) and
            @as(i64, c.batch) * c.mla_cache <= @divTrunc(max_i, @max(c.mla_L, c.mla_R))))
            return fail("MLA cache/workspace exceeds supported index range");
    }
}

/// Architecture keys: everything that fixes weight shapes or the state layout.
pub fn sameModel(a: Cfg, b: Cfg) bool {
    return a.dim == b.dim and a.layers == b.layers and a.experts == b.experts and a.topk == b.topk and
        a.gated == b.gated and a.half_min == b.half_min and a.half_max == b.half_max and a.mla == b.mla and
        a.mla_heads == b.mla_heads and a.mla_dh == b.mla_dh and a.mla_L == b.mla_L and a.mla_R == b.mla_R and
        a.mla_cache == b.mla_cache and a.mla_every == b.mla_every and a.mla_cc == b.mla_cc and
        a.mla_theta == b.mla_theta;
}
