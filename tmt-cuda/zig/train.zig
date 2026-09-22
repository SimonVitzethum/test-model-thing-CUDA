//! train: DATA CHECKPOINT [mode=train|eval] [steps=N] [saveevery=N] [init=PATH] [key=value ...]
//! Training and evaluation CLI (port of src/train_main.cu; same checkpoints,
//! same log lines). The model and its kernels run through the C API.
const std = @import("std");
const tmt = @import("tmt.zig");

const Fatal = error{Fatal};

/// Line output through libc snprintf, flushed per line like a log.
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

fn cfgText(gpa: std.mem.Allocator, c: *tmt.Cfg) ![]u8 {
    return gpa.dupe(u8, std.mem.span(tmt.tmt_cfg_text(c)));
}
fn cfgInt(text: []const u8, key: []const u8) i64 {
    return std.fmt.parseInt(i64, tmt.configValue(text, key) orelse "0", 10) catch 0;
}
fn cfgFloat(text: []const u8, key: []const u8) f64 {
    return std.fmt.parseFloat(f64, tmt.configValue(text, key) orelse "0") catch 0;
}
/// Like the C++ integer_option: a nonnegative int, nothing else.
fn integerOption(v: []const u8) Fatal!u64 {
    const n = std.fmt.parseInt(i32, v, 10) catch return fail("invalid nonnegative integer", .{});
    if (n < 0) return fail("invalid nonnegative integer", .{});
    return @intCast(n);
}

/// Read-only memory map of the dataset plus its FNV-1a hash.
const Dataset = struct {
    bytes: []align(std.heap.page_size_min) const u8,
    hash: u64,
    fn open(io: std.Io, path: []const u8) !Dataset {
        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return fail("dataset missing or shorter than two bytes", .{});
        defer file.close(io);
        const size = (file.stat(io) catch return fail("dataset missing or shorter than two bytes", .{})).size;
        if (size < 2) return fail("dataset missing or shorter than two bytes", .{});
        const bytes = std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0) catch
            return fail("dataset mmap failed", .{});
        var h: u64 = 14695981039346656037;
        for (bytes) |c| h = (h ^ c) *% 1099511628211;
        return .{ .bytes = bytes, .hash = h };
    }
    fn close(d: Dataset) void {
        std.posix.munmap(d.bytes);
    }
};

fn exists(io: std.Io, path: []const u8) Fatal!bool {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |e| switch (e) {
        error.FileNotFound => return false,
        else => return fail("cannot stat checkpoint", .{}),
    };
    f.close(io);
    return true;
}

fn run(init: std.process.Init, out: Out) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 3) {
        try out.print("usage: train DATA CHECKPOINT [mode=train|eval] [steps=N] [saveevery=N] [key=value ...]\n", .{});
        return error.Usage;
    }
    const path = argv[2];
    const have = try exists(io, path);
    // init=PATH starts a NEW run from another checkpoint's weights (fresh
    // optimizer and progress); its configuration is the default for this run.
    var init_path: ?[:0]const u8 = null;
    for (argv[3..]) |a| if (std.mem.startsWith(u8, a, "init=")) {
        init_path = try arena.dupeZ(u8, a[5..]);
    };
    if (init_path != null and have) return fail("init= only applies to a new checkpoint path", .{});
    const cfg = (if (have) tmt.tmt_cfg_from_checkpoint(path) else if (init_path) |p| tmt.tmt_cfg_from_checkpoint(p) else tmt.tmt_cfg_new()) orelse
        return fail("{s}", .{tmt.lastError()});
    defer tmt.tmt_cfg_free(cfg);
    const saved_config = try cfgText(arena, cfg);
    var mode: []const u8 = "train";
    var steps: u64 = 0;
    var saveevery: u64 = 500;
    for (argv[3..]) |a| {
        const eq = std.mem.indexOfScalar(u8, a, '=') orelse return fail("expected key=value", .{});
        const key = a[0..eq];
        const value = a[eq + 1 ..];
        if (std.mem.eql(u8, key, "init")) continue;
        if (std.mem.eql(u8, key, "mode")) {
            mode = value;
        } else if (std.mem.eql(u8, key, "steps")) {
            steps = try integerOption(value);
        } else if (std.mem.eql(u8, key, "saveevery")) {
            saveevery = try integerOption(value);
        } else try api(tmt.tmt_cfg_set(cfg, try arena.dupeZ(u8, key), try arena.dupeZ(u8, value)));
    }
    if (!std.mem.eql(u8, mode, "train") and !std.mem.eql(u8, mode, "eval")) return fail("mode must be train or eval", .{});
    const evaluation = std.mem.eql(u8, mode, "eval");
    if (evaluation and !have) return fail("evaluation requires an existing checkpoint", .{});
    try api(tmt.tmt_cfg_validate(cfg));
    // Evaluation may change seqlen, maxcarry, docsep and dialog: none affects
    // weights or the stored state layout, and eval starts from a fresh state.
    const compared = tmt.tmt_cfg_copy(cfg).?;
    defer tmt.tmt_cfg_free(compared);
    if (evaluation and have) {
        const stored = tmt.tmt_cfg_from_checkpoint(path) orelse return fail("{s}", .{tmt.lastError()});
        defer tmt.tmt_cfg_free(stored);
        const st = try cfgText(arena, stored);
        for ([_][:0]const u8{ "seqlen", "maxcarry", "docsep", "dialog" }) |k|
            try api(tmt.tmt_cfg_set(compared, k, try arena.dupeZ(u8, tmt.configValue(st, k).?)));
    }
    if (have and !std.mem.eql(u8, try cfgText(arena, compared), saved_config))
        return fail("configuration differs from checkpoint; choose a new checkpoint path", .{});

    const text = try cfgText(arena, cfg);
    const B: u64 = @intCast(cfgInt(text, "batch"));
    const T: u64 = @intCast(cfgInt(text, "seqlen"));
    const N = B * T;
    const dialog = cfgInt(text, "dialog") != 0;
    const traces = cfgInt(text, "traces") != 0;
    const experts: usize = @intCast(cfgInt(text, "experts"));
    const layers: usize = @intCast(cfgInt(text, "layers"));
    const maxcarry: u64 = @intCast(cfgInt(text, "maxcarry"));
    const data = try Dataset.open(io, argv[1]);
    defer data.close();
    if (data.bytes.len < B * (if (evaluation) 2 else T + 1)) return fail("dataset too small for batch/seqlen", .{});
    const model = tmt.tmt_model_new(cfg) orelse return fail("{s}", .{tmt.lastError()});
    defer tmt.tmt_model_free(model);
    var progress = tmt.Progress{};
    const path_z = try arena.dupeZ(u8, path);
    if (have) {
        try api(tmt.tmt_model_load(model, path_z, &progress));
    } else if (init_path) |p| try api(tmt.tmt_model_load_weights(model, p)); // architecture must match
    if (!evaluation and have and (progress.data_size != data.bytes.len or progress.data_hash != data.hash))
        return fail("resume dataset differs from checkpoint", .{});
    if (evaluation) {
        try api(tmt.tmt_model_reset_state(model));
        progress = .{};
    }
    progress.data_size = data.bytes.len;
    progress.data_hash = data.hash;
    const size: u64 = data.bytes.len;
    const per = size / B;
    if (!evaluation and (progress.cursor % T != 0 or progress.cursor > ((per - 1) / T) * T))
        return fail("invalid checkpoint dataset cursor", .{});
    const ids = try arena.alloc(c_int, N);
    const targets = try arena.alloc(c_int, N);
    const ends = try arena.alloc(c_int, N);
    const valid = try arena.alloc(bool, N);
    const losses = try arena.alloc(f32, N);
    const use_acc = try arena.alloc(i64, @max(1, layers * experts)); // router balance (MoE collapse)
    @memset(use_acc, 0);
    const counts = try arena.alloc(i64, @max(1, layers * experts));
    var use_win: u64 = 0;
    const begin_step = progress.step;
    var measured: u64 = 0;
    var ce_sum: f64 = 0;
    const ln2 = tmt.log(2.0);
    const begin = std.Io.Clock.awake.now(io);
    tmt.tmt_install_stop_handler();
    const mode_z = try arena.dupeZ(u8, mode);
    try out.print("CUDA hierarchical byte model: D=%d L=%d E=%d k=%d B=%d T=%d mode=%s step=%llu\n", .{
        @as(c_int, @intCast(cfgInt(text, "dim"))),  @as(c_int, @intCast(layers)), @as(c_int, @intCast(experts)),
        @as(c_int, @intCast(cfgInt(text, "topk"))), @as(c_int, @intCast(B)),      @as(c_int, @intCast(T)),
        mode_z.ptr,                                 @as(c_ulonglong, progress.step),
    });
    while (tmt.tmt_stop_requested() == 0 and (steps == 0 or progress.step - begin_step < steps)) {
        if (progress.step >= std.math.maxInt(i32) - 1) return fail("optimizer step limit reached", .{});
        if (!evaluation and progress.cursor + T >= per) {
            progress.cursor = 0;
            progress.epoch += 1;
            progress.carried = 0;
            try api(tmt.tmt_model_reset_state(model));
        }
        if (evaluation and progress.cursor >= (size + B - 1) / B - 1) break;
        if (maxcarry != 0 and progress.carried >= maxcarry) {
            progress.carried = 0;
            try api(tmt.tmt_model_reset_state(model));
        }
        var count: u64 = 0;
        for (0..B) |b| {
            const start = if (evaluation) size * b / B else per * b;
            const end = if (evaluation) size * (b + 1) / B else start + per;
            // dialog=1: only bytes inside an assistant turn (after 0x03, up to and
            // including its closing 0x04) are scored; the rest is context.
            var assistant = false;
            if (dialog) {
                var k = start + progress.cursor;
                while (k > start) {
                    k -= 1;
                    const c = data.bytes[k];
                    if (c == 0x02 or c == 0x03 or c == 0x04 or c == 0x1E) {
                        assistant = c == 0x03;
                        break;
                    }
                }
            }
            for (0..T) |t| {
                const i = b * T + t;
                const offset = start + progress.cursor + t;
                valid[i] = offset + 1 < end;
                ids[i] = if (valid[i]) data.bytes[offset] else 0;
                targets[i] = if (valid[i]) data.bytes[offset + 1] else 0;
                if (dialog and valid[i]) {
                    const c = data.bytes[offset];
                    if (c == 0x02 or c == 0x03 or c == 0x04 or c == 0x1E) assistant = c == 0x03;
                    if (!assistant) { // context only, not scored
                        valid[i] = false;
                        targets[i] = -1;
                    }
                }
                ends[i] = @intFromBool(targets[i] == 10);
                count += @intFromBool(valid[i]);
            }
        }
        var loss: f32 = 0;
        var ce: f32 = 0;
        try api(tmt.tmt_model_forward(model, ids.ptr, targets.ptr, ends.ptr, &loss, &ce));
        if (!evaluation and experts > 1) {
            try api(tmt.tmt_model_expert_counts(model, counts.ptr));
            for (use_acc, counts) |*a, c| a.* += c;
            use_win += 1;
        }
        if (evaluation) {
            // Score only real next-byte pairs in the final padded window.
            try api(tmt.tmt_model_losses(model, losses.ptr));
            for (losses, valid) |l, v| if (v) {
                ce_sum += l;
            };
        } else {
            if (!std.math.isFinite(loss)) return fail("non-finite loss; update refused", .{});
            const diagnostics = (progress.step + 1) % 100 == 0;
            const log_traces = traces and diagnostics;
            try api(tmt.tmt_model_backward(model, @intFromBool(log_traces)));
            if (log_traces) {
                var s: [9]f64 = undefined;
                try api(tmt.tmt_model_trace_stats(model, &s));
                try out.print("traces:", .{});
                const names = [3][:0]const u8{ "decay", "gate", "emb" };
                for (0..3) |g| {
                    if (g == 1 and cfgInt(text, "gated") == 0) continue;
                    const pp = s[3 * g];
                    const ww = s[3 * g + 1];
                    const pw = s[3 * g + 2];
                    try out.print(" %s |tr|/|win|=%.3g cos=%.3f", .{ names[g].ptr, @sqrt(pp / @max(ww, 1e-30)), pw / @max(@sqrt(pp * ww), 1e-30) });
                }
                try out.print("\n", .{});
            }
            if (diagnostics) {
                var sum: [5]f64 = undefined;
                var cnt: [5]i64 = undefined;
                try api(tmt.tmt_model_state_buckets(model, &sum, &cnt));
                const names = [5][:0]const u8{ "<16", "<128", "<1k", "<8k", ">=8k" };
                try out.print("state:", .{});
                for (0..5) |k| if (cnt[k] != 0)
                    try out.print(" h%s n=%ld |s|=%.3g", .{ names[k].ptr, @as(c_long, @intCast(@divTrunc(cnt[k], @as(i64, @intCast(B))))), sum[k] / @as(f64, @floatFromInt(cnt[k])) });
                try out.print("\n", .{});
            }
            try api(tmt.tmt_model_optimizer_step(model, @intCast(progress.step)));
            ce_sum += ce * @as(f32, @floatFromInt(count)); // float product, as in C++ (ce * count)
        }
        if (!std.math.isFinite(ce_sum)) return fail("non-finite evaluation score", .{});
        measured += count;
        progress.step += 1;
        progress.cursor += T;
        progress.carried += T;
        if (!evaluation and saveevery != 0 and progress.step % saveevery == 0) try api(tmt.tmt_model_save(model, path_z, &progress));
        if (progress.step % 20 == 0)
            try out.print("step=%llu loss=%.6f ce=%.6f bpb=%.6f\n", .{ @as(c_ulonglong, progress.step), @as(f64, loss), @as(f64, ce), @as(f64, ce) / ln2 });
        if (!evaluation and experts > 1 and use_win > 0 and progress.step % 100 == 0) {
            var tot: i64 = 0;
            for (use_acc) |v| tot += v;
            var mn: f64 = 1;
            var mx: f64 = 0;
            var dead: c_long = 0;
            for (use_acc) |v| {
                const sh: f64 = if (tot != 0) @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(tot)) * @as(f64, @floatFromInt(layers)) * @as(f64, @floatFromInt(experts)) else 0;
                mn = @min(mn, sh);
                mx = @max(mx, sh);
                if (sh < 0.01) dead += 1;
            }
            try out.print("router: min=%.3f max=%.3f dead=%ld/%d (share 1.0=uniform)\n", .{ mn, mx, dead, @as(c_int, @intCast(layers * experts)) });
            @memset(use_acc, 0);
            use_win = 0;
        }
    }
    try api(tmt.tmt_synchronize());
    const ns = begin.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const seconds = @as(f64, @floatFromInt(ns)) / 1e9;
    if (!evaluation) try api(tmt.tmt_model_save(model, path_z, &progress));
    if (measured == 0) return fail("no byte pairs evaluated", .{});
    // Stable machine-readable record for experiment runners.
    const m: f64 = @floatFromInt(measured);
    try out.print("{\"mode\":\"%s\",\"steps\":%llu,\"bytes\":%llu,\"ce\":%.9g,\"bpb\":%.9g,\"seconds\":%.6f,\"bytes_per_second\":%.3f}\n", .{
        mode_z.ptr, @as(c_ulonglong, progress.step - begin_step), @as(c_ulonglong, measured), ce_sum / m, ce_sum / m / ln2, seconds, m / @max(seconds, 1e-9),
    });
}

pub fn main(init: std.process.Init) u8 {
    const io = init.io;
    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    run(init, .{ .w = &stdout.interface }) catch |e| {
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
