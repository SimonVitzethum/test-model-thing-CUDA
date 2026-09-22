//! Zig bindings to the CUDA model (src/capi.h) and helpers shared by the
//! host programs.
const std = @import("std");
const stdrand = @import("stdrand.zig");

pub const Gen = opaque {};
extern fn tmt_last_error() [*:0]const u8;
extern fn tmt_gen_open(checkpoint: [*:0]const u8) ?*Gen;
extern fn tmt_gen_close(h: *Gen) void;
extern fn tmt_gen_config(h: *const Gen) [*:0]const u8;
extern fn tmt_gen_feed(h: *Gen, byte: c_int, logits: *[256]f32, stop_prob: ?*f32) c_int;
extern fn tmt_gen_reset(h: *Gen) c_int;
extern fn tmt_gen_fed(h: *const Gen) c_long;
extern "c" fn exp(x: f64) f64; // glibc, as used by the C++ sampler (bit-identical probabilities)

pub fn lastError() []const u8 {
    return std.mem.span(tmt_last_error());
}

/// Value of `key` in a config text of key=value lines.
pub fn configValue(text: []const u8, key: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        if (std.mem.eql(u8, line[0..eq], key)) return line[eq + 1 ..];
    }
    return null;
}

/// Byte-by-byte generator with a persistent recurrent state (port of
/// src/generate.h; same sampling as the C++ version for the same seed).
pub const Generator = struct {
    h: *Gen,
    rng: stdrand.Mt19937,
    logits: [256]f32 = undefined,
    stop_prob: f32 = 0,
    ready: bool = false,

    pub fn open(checkpoint: [:0]const u8, seed: u32) !Generator {
        const h = tmt_gen_open(checkpoint.ptr) orelse return error.OpenFailed;
        return .{ .h = h, .rng = stdrand.Mt19937.init(seed) };
    }
    pub fn close(g: *Generator) void {
        tmt_gen_close(g.h);
    }
    pub fn config(g: *const Generator) []const u8 {
        return std.mem.span(tmt_gen_config(g.h));
    }
    pub fn configInt(g: *const Generator, key: []const u8) i64 {
        const v = configValue(g.config(), key) orelse return 0;
        return std.fmt.parseInt(i64, v, 10) catch 0;
    }
    /// An untrained stop head (stop=0) gives random logits and is ignored.
    pub fn stopTrained(g: *const Generator) bool {
        const v = configValue(g.config(), "stop") orelse return false;
        return (std.fmt.parseFloat(f64, v) catch 0) > 0;
    }
    pub fn fed(g: *const Generator) i64 {
        return tmt_gen_fed(g.h);
    }
    pub fn feed(g: *Generator, byte: u8) !void {
        if (tmt_gen_feed(g.h, byte, &g.logits, &g.stop_prob) != 0) return error.FeedFailed;
        g.ready = true;
    }
    pub fn feedText(g: *Generator, text: []const u8) !void {
        for (text) |c| try g.feed(c);
    }
    pub fn reset(g: *Generator) !void {
        if (tmt_gen_reset(g.h) != 0) return error.ResetFailed;
        g.ready = false;
    }
    /// Next byte from the prediction after the last fed byte (argmax if temp <= 0).
    pub fn sample(g: *Generator, temp: f32) u8 {
        std.debug.assert(g.ready);
        var best: usize = 0;
        for (g.logits, 0..) |v, i| if (v > g.logits[best]) {
            best = i;
        };
        if (temp <= 0) return @intCast(best);
        const mx = g.logits[best];
        var p: [256]f64 = undefined;
        var sum: f64 = 0;
        for (g.logits, 0..) |v, i| {
            const d: f32 = (v - mx) / temp; // float arithmetic, then double exp, as in C++
            p[i] = exp(d);
            sum += p[i];
        }
        var r = g.rng.canonical() * sum;
        for (p, 0..) |pi, i| {
            r -= pi;
            if (r <= 0) return @intCast(i);
        }
        return 255;
    }
};
