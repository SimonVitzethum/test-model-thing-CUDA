// train DATA CHECKPOINT [mode=train|eval] [steps=N] [saveevery=N] [init=PATH] [key=value ...]
// Training and evaluation share the model forward path.
const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const config = @import("config.zig");
const ckpt = @import("checkpoint.zig");
const data = @import("data.zig");
const u = @import("util.zig");
const cs = @cImport({
    @cInclude("signal.h");
    @cInclude("sys/stat.h");
    @cInclude("errno.h");
});
const Cfg = config.Cfg;

var interrupted: std.atomic.Value(bool) = .init(false);
fn onSignal(_: c_int) callconv(.c) void {
    interrupted.store(true, .monotonic);
}

pub fn orDie(comptime T: type, r: anytype, msg: *const []const u8) T {
    return r catch u.die("{s}", .{msg.*});
}

fn exists(path: [:0]const u8) bool {
    var st: cs.struct_stat = undefined;
    if (cs.stat(path.ptr, &st) == 0) return true;
    if (std.c._errno().* != cs.ENOENT) u.die("cannot stat checkpoint", .{});
    return false;
}

// Trace part vs in-window part of the gradient, per parameter type. A cosine
// near -1 or strongly fluctuating means traces fight the window gradient.
fn printTraceStats(m: *model.Model) void {
    const t = m.trlog orelse return;
    u.print("traces:", .{});
    const Acc = struct { pp: f64 = 0, ww: f64 = 0, pw: f64 = 0 };
    const report = struct {
        fn add(acc: *Acc, part: []const f32, grad: []const f32) void {
            for (part, grad) |p, g| {
                const pf: f64 = p;
                const w: f64 = @as(f64, g) - pf;
                acc.pp += pf * pf;
                acc.ww += w * w;
                acc.pw += pf * w;
            }
        }
        fn show(name: []const u8, acc: Acc) void {
            var b1: [32]u8 = undefined;
            u.print(" {s} |tr|/|win|={s} cos={d:.3}", .{ name, u.cfmt(&b1, "%.3g", @sqrt(acc.pp / @max(acc.ww, 1e-30))), acc.pw / @max(@sqrt(acc.pp * acc.ww), 1e-30) });
        }
    };
    var dec = Acc{};
    var gate = Acc{};
    for (m.layers, 0..) |ly, l| {
        const gd = model.getGrad(m, ly.decay);
        defer u.gpa.free(gd);
        report.add(&dec, t.dec[l], gd);
        const gg = model.getGrad(m, ly.gate);
        defer u.gpa.free(gg);
        report.add(&gate, t.gate[l], gg);
    }
    report.show("decay", dec);
    if (m.c.gated == 1) report.show("gate", gate);
    var emb = Acc{};
    const ge = model.getGrad(m, m.emb);
    defer u.gpa.free(ge);
    report.add(&emb, t.emb, ge);
    report.show("emb", emb);
    u.print("\n", .{});
}

// Mean |state| per half-life bucket (from decay alone; the gate makes the
// effective half-life input-dependent). Shows whether long channels store anything.
fn printStateBuckets(m: *model.Model, st: *model.State) void {
    const D: usize = @intCast(m.c.dim);
    const B: usize = @intCast(m.c.batch);
    const edges = [_]f64{ 16, 128, 1024, 8192, 1e300 };
    const names = [_][]const u8{ "<16", "<128", "<1k", "<8k", ">=8k" };
    var sum = [_]f64{0} ** 5;
    var count = [_]u64{0} ** 5;
    for (m.layers, 0..) |ly, l| {
        const decay = model.getMaster(m, ly.decay);
        defer u.gpa.free(decay);
        const carry = mx.toVecF32(st.carry[l]);
        defer u.gpa.free(carry);
        for (0..D) |d| {
            const a = 1 / (1 + @exp(-@as(f64, decay[d])));
            const half = if (a >= 1) 1e300 else @log(0.5) / @log(a);
            var k: usize = 0;
            while (half >= edges[k]) k += 1;
            for (0..B) |b| sum[k] += @abs(carry[b * D + d]);
            count[k] += B;
        }
    }
    u.print("state:", .{});
    for (0..5) |k| if (count[k] > 0) {
        var b1: [32]u8 = undefined;
        u.print(" h{s} n={d} |s|={s}", .{ names[k], count[k] / B, u.cfmt(&b1, "%.3g", sum[k] / @as(f64, @floatFromInt(count[k]))) });
    };
    u.print("\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        u.print("usage: train DATA CHECKPOINT [mode=train|eval] [steps=N] [saveevery=N] [key=value ...]\n", .{});
        std.process.exit(1);
    }
    model.initCompute();
    const path = args[2];
    const have = exists(path);
    // init=PATH starts a NEW run from another checkpoint's weights (fresh
    // optimizer and progress); its configuration is the default for this run.
    var init_path: ?[:0]const u8 = null;
    for (args[3..]) |a| if (std.mem.startsWith(u8, a, "init=")) {
        init_path = a[5..];
    };
    if (init_path != null and have) u.die("init= only applies to a new checkpoint path", .{});
    const init_cfg: Cfg = if (init_path) |p| orDie(Cfg, ckpt.readConfig(p), &ckpt.last_error) else .{};
    var cfg: Cfg = if (have) orDie(Cfg, ckpt.readConfig(path), &ckpt.last_error) else init_cfg;
    const saved_cfg = cfg;
    var mode: []const u8 = "train";
    var steps: u64 = 0;
    var saveevery: u64 = 500;
    for (args[3..]) |a| {
        const kv = u.splitKV(a) orelse u.die("expected key=value", .{});
        if (std.mem.eql(u8, kv.key, "init")) continue;
        if (std.mem.eql(u8, kv.key, "mode")) mode = kv.value else if (std.mem.eql(u8, kv.key, "steps")) steps = u.parseCount(kv.value) else if (std.mem.eql(u8, kv.key, "saveevery")) saveevery = u.parseCount(kv.value) else config.set(&cfg, kv.key, kv.value) catch |e| switch (e) {
            error.UnknownOption => u.die("unknown option: {s}", .{kv.key}),
            else => u.die("invalid value for {s}", .{kv.key}),
        };
    }
    const evaluation = std.mem.eql(u8, mode, "eval");
    if (!evaluation and !std.mem.eql(u8, mode, "train")) u.die("mode must be train or eval", .{});
    if (evaluation and !have) u.die("evaluation requires an existing checkpoint", .{});
    config.validate(cfg) catch u.die("{s}", .{config.last_error});
    // Evaluation may change seqlen, maxcarry, docsep and dialog: none affects
    // weights or the stored state layout, and eval starts from a fresh state.
    var compared = cfg;
    if (evaluation) {
        compared.seqlen = saved_cfg.seqlen;
        compared.maxcarry = saved_cfg.maxcarry;
        compared.docsep = saved_cfg.docsep;
        compared.dialog = saved_cfg.dialog;
    }
    if (have and !config.equal(compared, saved_cfg))
        u.die("configuration differs from checkpoint; choose a new checkpoint path", .{});
    var ds = data.Dataset.open(args[1]);
    defer ds.close();
    const B: usize = @intCast(cfg.batch);
    const T: usize = @intCast(cfg.seqlen);
    const N = B * T;
    if (ds.bytes.len < B * (if (evaluation) 2 else T + 1)) u.die("dataset too small for batch/seqlen", .{});

    var m = model.build(cfg);
    var st = model.State.init(&m);
    var progress = ckpt.Progress{};
    if (have) {
        ckpt.load(path, &m, &st, &progress) catch u.die("{s}", .{ckpt.last_error});
    } else if (init_path) |p| ckpt.loadWeights(p, &m, init_cfg) catch u.die("{s}", .{ckpt.last_error});
    if (!evaluation and have and (progress.data_size != ds.bytes.len or progress.data_hash != ds.hash))
        u.die("resume dataset differs from checkpoint", .{});
    if (evaluation) {
        st.reset();
        progress = .{};
    }
    progress.data_size = ds.bytes.len;
    progress.data_hash = ds.hash;
    const per = ds.bytes.len / B;
    if (!evaluation and (progress.cursor % T != 0 or progress.cursor > ((per - 1) / T) * T))
        u.die("invalid checkpoint dataset cursor", .{});

    const ids = try u.gpa.alloc(i32, N);
    const targets = try u.gpa.alloc(i32, N);
    const ends = try u.gpa.alloc(i32, N);
    const valid = try u.gpa.alloc(bool, N);
    const losses = try u.gpa.alloc(f32, N);
    const begin_step = progress.step;
    var measured: u64 = 0;
    var ce_sum: f64 = 0;
    // Router balance accumulator (MoE collapse early warning).
    const use_acc = try u.gpa.alloc([16]i64, @intCast(cfg.layers));
    for (use_acc) |*r| r.* = @splat(0);
    var use_win: u64 = 0;
    const begin = u.now();
    _ = cs.signal(cs.SIGINT, onSignal);
    _ = cs.signal(cs.SIGTERM, onSignal);
    u.print("MLX hierarchical byte model: D={d} L={d} E={d} k={d} B={d} T={d} mode={s} step={d}\n", .{ cfg.dim, cfg.layers, cfg.experts, cfg.topk, B, T, mode, progress.step });
    const bytes = ds.bytes;
    while (!interrupted.load(.monotonic) and (steps == 0 or progress.step - begin_step < steps)) {
        if (progress.step >= std.math.maxInt(i32) - 1) u.die("optimizer step limit reached", .{});
        if (!evaluation and progress.cursor + T >= per) {
            progress.cursor = 0;
            progress.epoch += 1;
            progress.carried = 0;
            st.reset();
        }
        if (evaluation and progress.cursor >= (bytes.len + B - 1) / B - 1) break;
        if (cfg.maxcarry > 0 and progress.carried >= @as(u64, @intCast(cfg.maxcarry))) {
            progress.carried = 0;
            st.reset();
        }
        var count: u64 = 0;
        for (0..B) |b| {
            const start = if (evaluation) bytes.len * b / B else per * b;
            const end = if (evaluation) bytes.len * (b + 1) / B else start + per;
            // dialog=1: only bytes inside an assistant turn (after 0x03, up to and
            // including its closing 0x04) are scored; the rest is context.
            var assistant = false;
            if (cfg.dialog == 1) {
                var k = start + progress.cursor;
                while (k > start) {
                    k -= 1;
                    if (data.isMarker(bytes[k])) {
                        assistant = bytes[k] == data.ASSISTANT;
                        break;
                    }
                }
            }
            for (0..T) |t| {
                const i = b * T + t;
                const offset = start + progress.cursor + t;
                valid[i] = offset + 1 < end;
                ids[i] = if (valid[i]) bytes[offset] else 0;
                targets[i] = if (valid[i]) bytes[offset + 1] else 0;
                if (cfg.dialog == 1 and valid[i]) {
                    if (data.isMarker(bytes[offset])) assistant = bytes[offset] == data.ASSISTANT;
                    if (!assistant) { // context only, not scored
                        valid[i] = false;
                        targets[i] = -1;
                    }
                }
                ends[i] = @intFromBool(targets[i] == 10);
                count += @intFromBool(valid[i]);
            }
        }
        const m0 = mx.mark();
        const batch = model.makeBatch(cfg, ids, targets, ends);
        const diagnostics = (progress.step + 1) % 100 == 0;
        m.log_traces = cfg.traces == 1 and diagnostics and !evaluation;
        const r = model.forwardWindow(&m, &st, batch, .{ .grad = !evaluation });
        mx.release(m0);
        if (!evaluation and cfg.experts > 1) {
            for (use_acc, m.counts) |*acc, cnt| for (0..@intCast(cfg.experts)) |e| {
                acc[e] += cnt[e];
            };
            use_win += 1;
        }
        if (evaluation) {
            // Score only real next-byte pairs in the final padded window.
            mx.toF32(m.losses, losses);
            for (losses, valid) |l, ok| if (ok) {
                ce_sum += l;
            };
        } else {
            if (!std.math.isFinite(r.loss)) u.die("non-finite loss; update refused", .{});
            if (m.log_traces) printTraceStats(&m);
            if (diagnostics) printStateBuckets(&m, &st);
            model.optimizerStep(&m, progress.step) catch u.die("non-finite gradient; update refused", .{});
            ce_sum += r.ce * @as(f64, @floatFromInt(count));
        }
        if (!std.math.isFinite(ce_sum)) u.die("non-finite evaluation score", .{});
        measured += count;
        progress.step += 1;
        progress.cursor += T;
        progress.carried += T;
        if (!evaluation and saveevery > 0 and progress.step % saveevery == 0)
            ckpt.save(path, &m, &st, &progress) catch u.die("{s}", .{ckpt.last_error});
        if (progress.step % 20 == 0)
            u.print("step={d} loss={d:.6} ce={d:.6} bpb={d:.6}\n", .{ progress.step, r.loss, r.ce, r.ce / @log(2.0) });
        if (!evaluation and cfg.experts > 1 and use_win > 0 and progress.step % 100 == 0) {
            var tot: i64 = 0;
            for (use_acc) |row| for (row[0..@intCast(cfg.experts)]) |v| {
                tot += v;
            };
            var mn: f64 = 1;
            var mxv: f64 = 0;
            var dead: usize = 0;
            for (use_acc) |row| for (row[0..@intCast(cfg.experts)]) |v| {
                const s = if (tot > 0) @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(tot)) * @as(f64, @floatFromInt(cfg.layers * cfg.experts)) else 0;
                mn = @min(mn, s);
                mxv = @max(mxv, s);
                if (s < 0.01) dead += 1;
            };
            u.print("router: min={d:.3} max={d:.3} dead={d}/{d} (share 1.0=uniform)\n", .{ mn, mxv, dead, cfg.layers * cfg.experts });
            for (use_acc) |*row| row.* = @splat(0);
            use_win = 0;
        }
    }
    mx.synchronize();
    const seconds = u.now() - begin;
    if (!evaluation) ckpt.save(path, &m, &st, &progress) catch u.die("{s}", .{ckpt.last_error});
    if (measured == 0) u.die("no byte pairs evaluated", .{});
    // Stable machine-readable record for experiment runners.
    const mf: f64 = @floatFromInt(measured);
    var b1: [48]u8 = undefined;
    var b2: [48]u8 = undefined;
    u.print("{{\"mode\":\"{s}\",\"steps\":{d},\"bytes\":{d},\"ce\":{s},\"bpb\":{s},\"seconds\":{d:.6},\"bytes_per_second\":{d:.3}}}\n", .{
        mode, progress.step - begin_step, measured, u.cfmt(&b1, "%.9g", ce_sum / mf), u.cfmt(&b2, "%.9g", ce_sum / mf / @log(2.0)), seconds, mf / @max(seconds, 1e-9),
    });
}
