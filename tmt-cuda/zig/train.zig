//! train: DATA CHECKPOINT [mode=train|eval] [steps=N] [saveevery=N] [init=PATH] [key=value ...]
//! Training and evaluation CLI (port of src/train_main.cu; same checkpoints,
//! same log lines). The model and its kernels run through the C API.
const std = @import("std");
const config = @import("model/config.zig");
const checkpoint = @import("model/checkpoint.zig");
const session = @import("model/session.zig");
const model = @import("model/model.zig");
const gpu = @import("model/gpu.zig");
const ptx = @import("ptx.zig");

/// The compiled kernels, loaded by the session at startup.
const kernels_ptx = @embedFile("kernels.ptx");
/// Numbers are formatted through libc, so the log matches to the last digit.
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

const Fatal = error{Fatal};

/// Line output through libc snprintf, flushed per line like a log.
const Out = struct {
    w: *std.Io.Writer,
    fn print(o: Out, comptime fmt: [:0]const u8, args: anytype) !void {
        var buf: [1024]u8 = undefined;
        const n = @call(.auto, snprintf, .{ &buf, buf.len, fmt.ptr } ++ args);
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
/// The three modules report their own reason for refusing.
/// Writes the averaged weights beside the checkpoint as `<path>.avg`, in the
/// same format, so that evaluation and sampling can read it with no new
/// loading code. It is a separate file rather than a section in the
/// checkpoint because the moments it carries belong to the training weights
/// and are meaningless for resuming - and because the format is written byte
/// for byte as the C++ build writes it.
fn saveAveraged(gpa: std.mem.Allocator, io: std.Io, path: []const u8,
                sess: *session.Session, progress: *checkpoint.Progress) !void {
    if (sess.m.c.wavg <= 0) return;
    var buf: [4096]u8 = undefined;
    if (path.len + 4 > buf.len) return;
    const p = try std.fmt.bufPrint(&buf, "{s}.avg", .{path});
    model.swapAverage(&sess.m);
    defer model.swapAverage(&sess.m);
    try checkpoint.save(gpa, io, p, &sess.m, &sess.state, progress);
}

fn cfgFail() Fatal {
    return fail("{s}", .{config.lastError()});
}
fn ckptFail() Fatal {
    return fail("{s}", .{checkpoint.lastError()});
}
fn gpuFail() Fatal {
    return fail("{s}", .{gpu.lastError()});
}
fn cfgInt(text: []const u8, key: []const u8) i64 {
    return std.fmt.parseInt(i64, config.value(text, key) orelse "0", 10) catch 0;
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
    var cfg = if (have)
        checkpoint.readConfig(arena, io, path) catch return ckptFail()
    else if (init_path) |p|
        checkpoint.readConfig(arena, io, p) catch return ckptFail()
    else
        config.Cfg{};
    const saved_config = try config.text(arena, cfg);
    var mode: []const u8 = "train";
    var steps: u64 = 0;
    var saveevery: u64 = 500;
    var lossout: []const u8 = "";
    var entfile: []const u8 = "";
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
        } else if (std.mem.eql(u8, key, "lossout")) {
            lossout = value;
        } else if (std.mem.eql(u8, key, "entfile")) {
            entfile = value;
        } else config.set(&cfg, key, value) catch return cfgFail();
    }
    if (!std.mem.eql(u8, mode, "train") and !std.mem.eql(u8, mode, "eval")) return fail("mode must be train or eval", .{});
    const evaluation = std.mem.eql(u8, mode, "eval");
    if (evaluation and !have) return fail("evaluation requires an existing checkpoint", .{});
    config.validate(cfg) catch return cfgFail();
    // Evaluation may change seqlen, maxcarry, docsep and dialog: none affects
    // weights or the stored state layout, and eval starts from a fresh state.
    var compared = cfg;
    if (evaluation and have) {
        const stored = checkpoint.readConfig(arena, io, path) catch return ckptFail();
        compared.seqlen = stored.seqlen;
        compared.maxcarry = stored.maxcarry;
        compared.docsep = stored.docsep;
        compared.dialog = stored.dialog;
    }
    if (have and !std.mem.eql(u8, try config.text(arena, compared), saved_config))
        return fail("configuration differs from checkpoint; choose a new checkpoint path", .{});

    const text = try config.text(arena, cfg);
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
    var sess = session.Session.init(arena, io, cfg, kernels_ptx) catch return gpuFail();
    if (entfile.len > 0) {
        const raw = std.Io.Dir.cwd().readFileAlloc(io, entfile, arena, .unlimited) catch
            return fail("cannot read entfile", .{});
        if (raw.len != 65536 * 4) return fail("entfile must hold 65536 f32 values", .{});
        const tbl = try arena.alloc(f32, 65536);
        for (0..65536) |i| tbl[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        sess.m.ent = tbl;
    }
    defer sess.deinit();
    var progress = checkpoint.Progress{};
    if (have) {
        checkpoint.load(arena, io, path, &sess.m, &sess.state, &progress) catch return ckptFail();
    } else if (init_path) |p| { // the architecture must match
        const file_cfg = checkpoint.readConfig(arena, io, p) catch return ckptFail();
        checkpoint.loadWeights(arena, io, p, &sess.m, file_cfg) catch return ckptFail();
    }
    model.seedAverage(&sess.m) catch return gpuFail();
    if (!evaluation and have and (progress.data_size != data.bytes.len or progress.data_hash != data.hash))
        return fail("resume dataset differs from checkpoint", .{});
    if (evaluation) {
        sess.resetState() catch return gpuFail();
        progress = .{};
    }
    progress.data_size = data.bytes.len;
    progress.data_hash = data.hash;
    const size: u64 = data.bytes.len;
    const per = size / B;
    if (!evaluation and (progress.cursor % T != 0 or progress.cursor > ((per - 1) / T) * T))
        return fail("invalid checkpoint dataset cursor", .{});
    const mtp: usize = @intCast(cfgInt(text, "mtp"));
    // Windows added up before one optimizer update. The update is the
    // expensive part at this model size, so doing it less often pays.
    const accum: u64 = @intCast(@max(1, cfgInt(text, "accum")));
    const ids = try arena.alloc(c_int, N);
    const targets = try arena.alloc(c_int, N);
    const ends = try arena.alloc(c_int, N);
    const valid = try arena.alloc(bool, N);
    // One target row per extra prediction head, k bytes further ahead.
    const mtp_targets = try arena.alloc(c_int, mtp * N);
    const losses = try arena.alloc(f32, N);
    const use_acc = try arena.alloc(i64, @max(1, layers * experts)); // router balance (MoE collapse)
    @memset(use_acc, 0);
    const counts = try arena.alloc(i64, @max(1, layers * experts));
    var use_win: u64 = 0;
    const begin_step = progress.step;
    var measured: u64 = 0;
    var ce_sum: f64 = 0;
    // `lossout` writes the per-byte cross entropy back in corpus order, one
    // f32 per byte of the evaluated range, NaN where nothing was scored. It
    // is what lets an external predictor be interpolated against this model
    // without the model having to know about it: at the true byte, mixing
    // two distributions needs only both of their values there.
    const perbyte: []f32 = if (evaluation and lossout.len > 0)
        try arena.alloc(f32, data.bytes.len)
    else
        &.{};
    if (perbyte.len > 0) @memset(perbyte, std.math.nan(f32));
    const ln2 = @log(2.0);
    const begin = std.Io.Clock.awake.now(io);
    // TMT_PROFILE=1 times every kernel and matmul; it costs throughput,
    // because each one is waited for, but shows where the step goes.
    if (getenv("TMT_PROFILE") != null) ptx.Profile.enable();
    session.installStopHandler();
    const mode_z = try arena.dupeZ(u8, mode);
    try out.print("CUDA hierarchical byte model: D=%d L=%d E=%d k=%d B=%d T=%d mode=%s step=%llu\n", .{
        @as(c_int, @intCast(cfgInt(text, "dim"))),  @as(c_int, @intCast(layers)), @as(c_int, @intCast(experts)),
        @as(c_int, @intCast(cfgInt(text, "topk"))), @as(c_int, @intCast(B)),      @as(c_int, @intCast(T)),
        mode_z.ptr,                                 @as(c_ulonglong, progress.step),
    });
    while (!session.stopRequested() and (steps == 0 or progress.step - begin_step < steps)) {
        if (progress.step >= std.math.maxInt(i32) - 1) return fail("optimizer step limit reached", .{});
        if (!evaluation and progress.cursor + T >= per) {
            progress.cursor = 0;
            progress.epoch += 1;
            progress.carried = 0;
            sess.resetState() catch return gpuFail();
        }
        if (evaluation and progress.cursor >= (size + B - 1) / B - 1) break;
        if (maxcarry != 0 and progress.carried >= maxcarry) {
            progress.carried = 0;
            sess.resetState() catch return gpuFail();
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
                for (0..mtp) |h| {
                    const ahead = offset + 2 + h;
                    const scored = valid[i] and ahead < end and targets[i] >= 0;
                    mtp_targets[h * N + i] = if (scored) data.bytes[ahead] else -1;
                }
                count += @intFromBool(valid[i]);
            }
        }
        sess.setMtpTargets(mtp_targets) catch return gpuFail();
        // Routing noise, decaying over `rwarm` steps and never applied while
        // evaluating. Expert selection reinforces itself from the first step,
        // before any expert has earned a preference; noise early on removes
        // the tiny initial differences the loop would otherwise amplify.
        sess.m.rnoise_now = if (evaluation or cfg.rnoise <= 0 or cfg.rwarm <= 0) 0 else blk: {
            const f = 1.0 - @as(f32, @floatFromInt(progress.step)) / @as(f32, @floatFromInt(cfg.rwarm));
            break :blk if (f <= 0) 0 else cfg.rnoise * f;
        };
        sess.m.rseed = @truncate(@as(u64, @intCast(progress.step)) *% 0x9e3779b97f4a7c15);
        const window = sess.forward(ids, targets, ends) catch return gpuFail();
        const loss = window.total;
        const ce = window.ce;
        if (!evaluation and experts > 1) {
            sess.expertCounts(counts);
            for (use_acc, counts) |*a, c| a.* += c;
            use_win += 1;
        }
        if (evaluation) {
            // Score only real next-byte pairs in the final padded window.
            sess.losses(losses) catch return gpuFail();
            for (losses, valid) |l, v| if (v) {
                ce_sum += l;
            };
            if (perbyte.len > 0) {
                for (0..B) |b| {
                    const st = size * b / B;
                    for (0..T) |t| {
                        const i = b * T + t;
                        if (!valid[i]) continue;
                        const off = st + progress.cursor + t;
                        if (off < perbyte.len) perbyte[@intCast(off)] = losses[i];
                    }
                }
            }
        } else {
            if (!std.math.isFinite(loss)) return fail("non-finite loss; update refused", .{});
            const diagnostics = (progress.step + 1) % 100 == 0;
            const log_traces = traces and diagnostics;
            const first = progress.step % accum == 0;
            sess.backwardAccumulating(log_traces, first, @intCast(accum)) catch return gpuFail();
            if (log_traces) {
                var s: [9]f64 = undefined;
                sess.traceStats(&s) catch return gpuFail();
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
                sess.stateBuckets(&sum, &cnt) catch return gpuFail();
                const names = [5][:0]const u8{ "<16", "<128", "<1k", "<8k", ">=8k" };
                try out.print("state:", .{});
                for (0..5) |k| if (cnt[k] != 0)
                    try out.print(" h%s n=%ld |s|=%.3g", .{ names[k].ptr, @as(c_long, @intCast(@divTrunc(cnt[k], @as(i64, @intCast(B))))), sum[k] / @as(f64, @floatFromInt(cnt[k])) });
                try out.print("\n", .{});
            }
            if ((progress.step + 1) % accum == 0)
                sess.optimizerStep(@intCast(progress.step / accum)) catch |e| return if (e == error.NonFiniteGradient)
                    fail("non-finite gradient; update refused", .{})
                else
                    gpuFail();
            ce_sum += ce * @as(f32, @floatFromInt(count)); // float product, as in C++ (ce * count)
        }
        if (!std.math.isFinite(ce_sum)) return fail("non-finite evaluation score", .{});
        measured += count;
        progress.step += 1;
        progress.cursor += T;
        progress.carried += T;
        if (!evaluation and saveevery != 0 and progress.step % saveevery == 0) {
            checkpoint.save(arena, io, path, &sess.m, &sess.state, &progress) catch return ckptFail();
            saveAveraged(arena, io, path, &sess, &progress) catch return ckptFail();
        }
        ptx.Profile.collect(); // the events are read where the host waits anyway
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
    gpu.synchronize() catch return gpuFail();
    if (perbyte.len > 0) {
        const f = std.Io.Dir.cwd().createFile(io, lossout, .{}) catch
            return fail("cannot write lossout", .{});
        defer f.close(io);
        var wbuf: [1 << 16]u8 = undefined;
        var fw = f.writerStreaming(io, &wbuf);
        fw.interface.writeAll(std.mem.sliceAsBytes(perbyte)) catch
            return fail("cannot write lossout", .{});
        fw.interface.flush() catch return fail("cannot write lossout", .{});
    }
    const ns = begin.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const seconds = @as(f64, @floatFromInt(ns)) / 1e9;
    if (!evaluation) {
        checkpoint.save(arena, io, path, &sess.m, &sess.state, &progress) catch return ckptFail();
        saveAveraged(arena, io, path, &sess, &progress) catch return ckptFail();
    }
    if (measured == 0) return fail("no byte pairs evaluated", .{});
    try ptx.Profile.report(out.w);
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
