// gradcheck: DATA CHECKPOINT [len=4096] [window=128] [seqs=4] [offset=0] [key=value ...]
// Compares windowed gradients against the exact BPTT gradient of whole
// sequences, using trained weights. For each parameter type it reports the
// cosine similarity of truncated BPTT (traces=0) and the hybrid (traces=1)
// to full BPTT over `len` bytes. Nothing is trained or written.
const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const config = @import("config.zig");
const ckpt = @import("checkpoint.zig");
const data = @import("data.zig");
const u = @import("util.zig");
const Cfg = config.Cfg;

const Group = struct { name: []const u8, params: std.ArrayList(usize) = .empty };

fn groups(m: *const model.Model) [8]Group {
    var g = [_]Group{
        .{ .name = "decay" },    .{ .name = "gate" },    .{ .name = "embedding" }, .{ .name = "norm" },
        .{ .name = "router" },   .{ .name = "experts" }, .{ .name = "decoder" },   .{ .name = "mla" },
    };
    g[2].params.append(u.gpa, m.emb) catch unreachable;
    g[6].params.append(u.gpa, m.dec) catch unreachable;
    for (m.layers) |ly| {
        g[0].params.append(u.gpa, ly.decay) catch unreachable;
        if (m.c.gated == 1) g[1].params.append(u.gpa, ly.gate) catch unreachable;
        g[3].params.append(u.gpa, ly.gamma) catch unreachable;
        g[3].params.append(u.gpa, ly.beta) catch unreachable;
        if (m.c.experts > 1) g[4].params.append(u.gpa, ly.router) catch unreachable;
        for (0..@intCast(m.c.experts)) |e| g[5].params.append(u.gpa, ly.exp[e]) catch unreachable;
        if (ly.mla) |p| inline for (.{ "q", "dkv", "kr", "uk", "uv", "o", "gamma", "beta" }) |fld|
            g[7].params.append(u.gpa, @field(p, fld)) catch unreachable;
    }
    return g;
}

/// Accumulated gradients of one configuration over all windows of the sequences.
fn run(cfg: Cfg, file_cfg: Cfg, path: [:0]const u8, bytes: []const u8, starts: []const usize, len: usize, ce_mean: *f64) [][]f32 {
    var m = model.build(cfg);
    ckpt.loadWeights(path, &m, file_cfg) catch u.die("{s}", .{ckpt.last_error});
    var st = model.State.init(&m);
    const B: usize = @intCast(cfg.batch);
    const T: usize = @intCast(cfg.seqlen);
    const N = B * T;
    const acc = u.gpa.alloc([]f32, m.nparams()) catch unreachable;
    for (acc, 0..) |*a, j| {
        a.* = u.gpa.alloc(f32, m.at(j).n) catch unreachable;
        @memset(a.*, 0);
    }
    const ids = u.gpa.alloc(i32, N) catch unreachable;
    const nxt = u.gpa.alloc(i32, N) catch unreachable;
    const end = u.gpa.alloc(i32, N) catch unreachable;
    var ce_sum: f64 = 0;
    for (0..len / T) |w| {
        for (0..B) |b| for (0..T) |t| {
            const o = starts[b] + w * T + t;
            ids[b * T + t] = bytes[o];
            nxt[b * T + t] = bytes[o + 1];
            end[b * T + t] = @intFromBool(bytes[o + 1] == 10);
        };
        const m0 = mx.mark();
        const r = model.forwardWindow(&m, &st, model.makeBatch(cfg, ids, nxt, end), .{ .grad = true });
        mx.release(m0);
        ce_sum += r.ce;
        for (acc, 0..) |a, j| {
            const g = model.getGrad(&m, j);
            defer u.gpa.free(g);
            for (a, g) |*x, y| x.* += y;
        }
    }
    ce_mean.* = ce_sum / @as(f64, @floatFromInt(len / T));
    return acc;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        u.print("usage: gradcheck DATA CHECKPOINT [len=4096] [window=128] [seqs=4] [offset=0] [key=value ...]\n", .{});
        std.process.exit(1);
    }
    model.initCompute();
    const path = args[2];
    const file_cfg = ckpt.readConfig(path) catch u.die("{s}", .{ckpt.last_error});
    var cfg = file_cfg;
    var len: usize = 4096;
    var window: usize = 128;
    var seqs: usize = 4;
    var offset: usize = 0;
    for (args[3..]) |a| {
        const kv = u.splitKV(a) orelse u.die("expected key=value", .{});
        if (std.mem.eql(u8, kv.key, "len")) len = @intCast(u.parseCount(kv.value)) else if (std.mem.eql(u8, kv.key, "window")) window = @intCast(u.parseCount(kv.value)) else if (std.mem.eql(u8, kv.key, "seqs")) seqs = @intCast(u.parseCount(kv.value)) else if (std.mem.eql(u8, kv.key, "offset")) offset = @intCast(u.parseCount(kv.value)) else config.set(&cfg, kv.key, kv.value) catch u.die("invalid option {s}", .{kv.key});
    }
    if (len == 0 or window == 0 or seqs == 0 or len % window != 0)
        u.die("require len, window, seqs > 0 and len divisible by window", .{});
    var ds = data.Dataset.open(args[1]);
    defer ds.close();
    if (ds.bytes.len < offset + seqs * (len + 1)) u.die("dataset too small", .{});
    const stride = (ds.bytes.len - offset - len - 1) / seqs;
    const starts = try u.gpa.alloc(usize, seqs);
    for (starts, 0..) |*s, i| s.* = offset + i * stride;

    cfg.batch = @intCast(seqs); // with mla=1, len must not exceed mla_cache
    var full = cfg;
    var tbptt = cfg;
    var hybrid = cfg;
    full.seqlen = @intCast(len);
    full.traces = 0;
    tbptt.seqlen = @intCast(window);
    tbptt.traces = 0;
    hybrid.seqlen = @intCast(window);
    hybrid.traces = 1;
    for ([_]Cfg{ full, tbptt, hybrid }) |cc| config.validate(cc) catch u.die("{s}", .{config.last_error});
    var ce_full: f64 = 0;
    var ce_t: f64 = 0;
    var ce_h: f64 = 0;
    const gf = run(full, file_cfg, path, ds.bytes, starts, len, &ce_full);
    const gt = run(tbptt, file_cfg, path, ds.bytes, starts, len, &ce_t);
    const gh = run(hybrid, file_cfg, path, ds.bytes, starts, len, &ce_h);

    var shape = model.build(full);
    var b1: [32]u8 = undefined;
    u.print("gradcheck: len={d} window={d} seqs={d} trace_decay={s} docsep={d}  CE full={d:.4} windowed={d:.4}\n", .{ len, window, seqs, u.cfmt(&b1, "%g", cfg.trace_decay), cfg.docsep, ce_full, ce_t });
    u.print("{s:<10} {s:>12} {s:>12} {s:>14}\n", .{ "params", "cos TBPTT", "cos hybrid", "|hybrid|/|full|" });
    // Windowed sums of per-window means equal (len/window) x the full-window mean.
    const scale: f64 = @as(f64, @floatFromInt(len)) / @as(f64, @floatFromInt(window));
    for (groups(&shape)) |g| {
        if (g.params.items.len == 0) continue;
        var ff: f64 = 0;
        var tt: f64 = 0;
        var hh: f64 = 0;
        var ft: f64 = 0;
        var fh: f64 = 0;
        for (g.params.items) |j| for (gf[j], gt[j], gh[j]) |fv, tv, hv| {
            const fs = fv * scale;
            ff += fs * fs;
            tt += @as(f64, tv) * tv;
            hh += @as(f64, hv) * hv;
            ft += fs * tv;
            fh += fs * hv;
        };
        if (ff == 0) continue;
        u.print("{s:<10} {d:>12.4} {d:>12.4} {d:>14.4}\n", .{ g.name, ft / @max(@sqrt(ff * tt), 1e-30), fh / @max(@sqrt(ff * hh), 1e-30), @sqrt(hh / ff) });
    }
}
