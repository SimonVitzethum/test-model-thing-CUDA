//! gradcheck: DATA CHECKPOINT [len=4096] [window=128] [seqs=4] [offset=0] [key=value ...]
//! Compares windowed gradients against the exact BPTT gradient of whole
//! sequences (port of src/gradcheck.cu). Nothing is trained or written.
const std = @import("std");
const tmt = @import("tmt.zig");

const Fatal = error{Fatal};

const Out = struct {
    w: *std.Io.Writer,
    fn print(o: Out, comptime fmt: [:0]const u8, args: anytype) !void {
        var buf: [1024]u8 = undefined;
        const n = @call(.auto, tmt.snprintf, .{ &buf, buf.len, fmt.ptr } ++ args);
        try o.w.writeAll(buf[0..@intCast(@min(n, buf.len - 1))]);
        if (fmt[fmt.len - 1] == '\n') try o.w.flush();
    }
};

var err_msg: [512]u8 = undefined;
var err_len: usize = 0;
fn fail(comptime fmt: []const u8, args: anytype) Fatal {
    const s: []const u8 = std.fmt.bufPrint(&err_msg, fmt, args) catch err_msg[0..];
    err_len = s.len;
    return error.Fatal;
}
fn api(rc: c_int) Fatal!void {
    if (rc != 0) return fail("{s}", .{tmt.lastError()});
}

/// Gradients of one configuration, accumulated over all windows, plus the
/// group each parameter belongs to.
const Run = struct {
    grads: [][]f32,
    groups: [][]const u8,
    ce_mean: f64,
};

fn run_cfg(gpa: std.mem.Allocator, cfg: *const tmt.Cfg, ckpt: [:0]const u8, data: []const u8,
           starts: []const usize, len: usize, keep_groups: bool) !Run {
    const model = tmt.tmt_model_new(cfg) orelse return fail("{s}", .{tmt.lastError()});
    defer tmt.tmt_model_free(model);
    try api(tmt.tmt_model_load_weights(model, ckpt.ptr));
    const text = std.mem.span(tmt.tmt_cfg_text(@constCast(cfg)));
    const B: usize = @intCast(std.fmt.parseInt(i64, tmt.configValue(text, "batch").?, 10) catch 0);
    const T: usize = @intCast(std.fmt.parseInt(i64, tmt.configValue(text, "seqlen").?, 10) catch 0);
    const N = B * T;

    const np: usize = @intCast(tmt.tmt_model_param_count(model));
    const acc = try gpa.alloc([]f32, np);
    const groups = try gpa.alloc([]const u8, np);
    for (acc, groups, 0..) |*a, *g, j| {
        const n: usize = @intCast(tmt.tmt_model_param_size(model, @intCast(j)));
        a.* = try gpa.alloc(f32, n);
        @memset(a.*, 0);
        g.* = if (keep_groups) try gpa.dupe(u8, std.mem.span(tmt.tmt_model_param_group(model, @intCast(j)))) else "";
    }
    const buf = try gpa.alloc(f32, blk: {
        var mx: usize = 0;
        for (acc) |a| mx = @max(mx, a.len);
        break :blk mx;
    });
    defer gpa.free(buf);

    const ids = try gpa.alloc(c_int, N);
    const nxt = try gpa.alloc(c_int, N);
    const end = try gpa.alloc(c_int, N);
    defer gpa.free(ids);
    defer gpa.free(nxt);
    defer gpa.free(end);
    var ce_sum: f64 = 0;
    const windows = len / T;
    for (0..windows) |w| {
        for (0..B) |b| for (0..T) |t| {
            const o = starts[b] + w * T + t;
            ids[b * T + t] = data[o];
            nxt[b * T + t] = data[o + 1];
            end[b * T + t] = @intFromBool(data[o + 1] == 10);
        };
        var loss: f32 = 0;
        var ce: f32 = 0;
        try api(tmt.tmt_model_forward(model, ids.ptr, nxt.ptr, end.ptr, &loss, &ce));
        try api(tmt.tmt_model_backward(model, 0));
        ce_sum += ce;
        for (acc, 0..) |a, j| {
            try api(tmt.tmt_model_param_grad(model, @intCast(j), buf.ptr));
            for (a, buf[0..a.len]) |*x, g| x.* += g;
        }
    }
    return .{ .grads = acc, .groups = groups, .ce_mean = ce_sum / @as(f64, @floatFromInt(windows)) };
}

const group_order = [_][]const u8{ "decay", "gate", "embedding", "norm", "router", "experts", "decoder", "mla" };

fn main_run(init: std.process.Init, out: Out) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 3) {
        try out.print("usage: gradcheck DATA CHECKPOINT [len=4096] [window=128] [seqs=4] [offset=0] [key=value ...]\n", .{});
        return error.Usage;
    }
    const ckpt = try arena.dupeZ(u8, argv[2]);
    const cfg = tmt.tmt_cfg_from_checkpoint(ckpt.ptr) orelse return fail("{s}", .{tmt.lastError()});
    defer tmt.tmt_cfg_free(cfg);
    var len: i64 = 4096;
    var window: i64 = 128;
    var seqs: i64 = 4;
    var offset: usize = 0;
    for (argv[3..]) |a| {
        const eq = std.mem.indexOfScalar(u8, a, '=') orelse return fail("expected key=value", .{});
        const k = a[0..eq];
        const v = a[eq + 1 ..];
        if (std.mem.eql(u8, k, "len")) {
            len = std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{});
        } else if (std.mem.eql(u8, k, "window")) {
            window = std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{});
        } else if (std.mem.eql(u8, k, "seqs")) {
            seqs = std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{});
        } else if (std.mem.eql(u8, k, "offset")) {
            offset = std.fmt.parseInt(usize, v, 10) catch return fail("stoul", .{});
        } else try api(tmt.tmt_cfg_set(cfg, try arena.dupeZ(u8, k), try arena.dupeZ(u8, v)));
    }
    if (len <= 0 or window <= 0 or seqs <= 0 or @rem(len, window) != 0)
        return fail("require len, window, seqs > 0 and len divisible by window", .{});
    const data = std.Io.Dir.cwd().readFileAlloc(io, argv[1], arena, .unlimited) catch &.{};
    const ulen: usize = @intCast(len);
    const useqs: usize = @intCast(seqs);
    if (data.len < offset + useqs * (ulen + 1)) return fail("dataset too small", .{});
    const stride = (data.len - offset - ulen - 1) / useqs;
    const starts = try arena.alloc(usize, useqs);
    for (starts, 0..) |*s, i| s.* = offset + i * stride;

    // with mla=1, len must not exceed mla_cache
    try api(tmt.tmt_cfg_set(cfg, "batch", (try std.fmt.allocPrintSentinel(arena, "{d}", .{seqs}, 0)).ptr));
    const len_z = try std.fmt.allocPrintSentinel(arena, "{d}", .{len}, 0);
    const win_z = try std.fmt.allocPrintSentinel(arena, "{d}", .{window}, 0);
    const text = try arena.dupe(u8, std.mem.span(tmt.tmt_cfg_text(cfg)));
    const trace_decay = std.fmt.parseFloat(f64, tmt.configValue(text, "trace_decay") orelse "0") catch 0;
    const docsep = std.fmt.parseInt(i64, tmt.configValue(text, "docsep") orelse "0", 10) catch 0;

    var cfgs: [3]*tmt.Cfg = undefined;
    const spec = [3][2][:0]const u8{ .{ len_z, "0" }, .{ win_z, "0" }, .{ win_z, "1" } };
    for (&cfgs, spec) |*c, s| {
        c.* = tmt.tmt_cfg_copy(cfg).?;
        try api(tmt.tmt_cfg_set(c.*, "seqlen", s[0].ptr));
        try api(tmt.tmt_cfg_set(c.*, "traces", s[1].ptr));
        try api(tmt.tmt_cfg_validate(c.*));
    }
    defer for (cfgs) |c| tmt.tmt_cfg_free(c);
    const gf = try run_cfg(arena, cfgs[0], ckpt, data, starts, ulen, true);
    const gt = try run_cfg(arena, cfgs[1], ckpt, data, starts, ulen, false);
    const gh = try run_cfg(arena, cfgs[2], ckpt, data, starts, ulen, false);

    try out.print("gradcheck: len=%d window=%d seqs=%d trace_decay=%g docsep=%d  CE full=%.4f windowed=%.4f\n", .{
        @as(c_int, @intCast(len)), @as(c_int, @intCast(window)), @as(c_int, @intCast(seqs)),
        trace_decay, @as(c_int, @intCast(docsep)), gf.ce_mean, gt.ce_mean,
    });
    try out.print("%-10s %12s %12s %14s\n", .{ "params", "cos TBPTT", "cos hybrid", "|hybrid|/|full|" });
    // Windowed sums of per-window means equal (len/window) x the full-window mean.
    const scale = @as(f64, @floatFromInt(len)) / @as(f64, @floatFromInt(window));
    for (group_order) |name| {
        var ff: f64 = 0;
        var tt: f64 = 0;
        var hh: f64 = 0;
        var ft: f64 = 0;
        var fh: f64 = 0;
        var any = false;
        for (gf.groups, 0..) |g, j| {
            if (!std.mem.eql(u8, g, name)) continue;
            any = true;
            for (gf.grads[j], gt.grads[j], gh.grads[j]) |fv, tv, hv| {
                const f = @as(f64, fv) * scale;
                const t: f64 = tv;
                const h: f64 = hv;
                ff += f * f;
                tt += t * t;
                hh += h * h;
                ft += f * t;
                fh += f * h;
            }
        }
        if (!any or ff == 0) continue;
        try out.print("%-10s %12.4f %12.4f %14.4f\n", .{
            name.ptr, ft / @max(@sqrt(ff * tt), 1e-30), fh / @max(@sqrt(ff * hh), 1e-30), @sqrt(hh / ff),
        });
    }
}

pub fn main(init: std.process.Init) u8 {
    const io = init.io;
    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    main_run(init, .{ .w = &stdout.interface }) catch |e| {
        stdout.interface.flush() catch {};
        if (e == error.Usage) return 1;
        var ebuf: [600]u8 = undefined;
        var ew = std.Io.File.stderr().writerStreaming(io, &ebuf);
        if (e == error.Fatal) ew.interface.print("error: {s}\n", .{err_msg[0..err_len]}) catch {} else ew.interface.print("error: {s}\n", .{@errorName(e)}) catch {};
        ew.interface.flush() catch {};
        return 1;
    };
    stdout.interface.flush() catch {};
    return 0;
}
