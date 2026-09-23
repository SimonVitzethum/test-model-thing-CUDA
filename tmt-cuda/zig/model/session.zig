//! What the command-line programs need from a model: a window of bytes in,
//! losses and gradients out, plus the diagnostics the training log prints.
//! Everything below this file is the model itself (model.zig and friends).
const std = @import("std");
const gpu = @import("gpu.zig");
const cfgmod = @import("config.zig");
const model = @import("model.zig");
const ops = @import("ops.zig");

pub const Cfg = cfgmod.Cfg;
pub const Progress = @import("checkpoint.zig").Progress;
pub const checkpoint = @import("checkpoint.zig");
pub const config = cfgmod;

pub const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// Heap-allocated, because the model keeps a pointer to it and callers
    /// move the session around (into a struct, out of a constructor).
    kernels: *gpu.Kernels,
    m: model.Model,
    state: model.StreamState,
    /// Lazily created by the knowledge-graph trainer.
    acc: [][*]f32 = &.{},
    ext: ?[*]f32 = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, c: Cfg, ptx: [:0]const u8) !Session {
        const kernels = try gpa.create(gpu.Kernels);
        errdefer gpa.destroy(kernels);
        kernels.* = try gpu.Kernels.load(gpa, ptx);
        var s = Session{ .gpa = gpa, .io = io, .kernels = kernels, .m = undefined, .state = undefined };
        s.m = try model.build(gpa, c, kernels);
        s.state = try model.buildState(gpa, &s.m);
        return s;
    }
    pub fn deinit(s: *Session) void {
        if (s.acc.len != 0) s.gpa.free(s.acc);
        s.state.deinit();
        s.m.deinit();
        s.gpa.destroy(s.kernels);
    }

    pub fn resetState(s: *Session) !void {
        try model.reset(&s.state, s.m.c);
    }

    /// Targets for the multi-token heads: `mtp * N` bytes, head k holding the
    /// byte k+1 positions further ahead (negative where there is none).
    pub fn setMtpTargets(s: *Session, targets: []const i32) !void {
        if (s.m.c.mtp == 0) return;
        try gpu.upload(s.m.nxt_mtp, std.mem.sliceAsBytes(targets));
    }

    /// One window: ids, targets (negative ones are not scored) and the
    /// end-of-line flags for the stop head.
    pub fn forward(s: *Session, ids: []const i32, targets: []const i32, ends: []const i32) !model.Losses {
        const n = ids.len * 4;
        // The patch boundaries follow from the bytes, so they are worked out
        // here, where the window still exists on the host.
        try model.layoutPatches(&s.m, ids);
        try model.sortByByte(&s.m, ids);
        try gpu.upload(s.m.ids, std.mem.sliceAsBytes(ids));
        try gpu.upload(s.m.nxt, std.mem.sliceAsBytes(targets));
        try gpu.upload(s.m.end, std.mem.sliceAsBytes(ends));
        std.debug.assert(n == targets.len * 4 and n == ends.len * 4);
        return model.forwardWindow(&s.m, &s.state);
    }
    /// Per-position cross entropy of the last forward.
    pub fn losses(s: *Session, out: []f32) !void {
        const N: u32 = @intCast(out.len);
        try (try s.kernels.get("ce_fwd")).launch((N + 255) / 256, 256,
            .{ s.m.logits, s.m.nxt, s.m.probs, s.m.losstmp, @as(i32, @intCast(N)) });
        try gpu.download(std.mem.sliceAsBytes(out), s.m.losstmp);
    }
    pub fn backward(s: *Session, log_traces: bool) !void {
        s.m.log_traces = log_traces;
        try model.backwardWindow(&s.m, &s.state);
    }
    pub fn optimizerStep(s: *Session, step: i32) !void {
        try model.optimizerStep(&s.m, step);
    }

    /// Forward with the fact bytes of this batch and a fresh state, as the
    /// knowledge-graph trainer runs it (every example is its own window).
    pub fn forwardMem(s: *Session, ids: []const i32, targets: []const i32, mem: []const i32) !model.Losses {
        const N = ids.len;
        const ends = try s.gpa.alloc(i32, N);
        defer s.gpa.free(ends);
        @memset(ends, 0);
        try gpu.upload(s.m.MS.ids, std.mem.sliceAsBytes(mem));
        try s.resetState();
        return s.forward(ids, targets, ends);
    }
    /// Backward with an extra gradient on the final representation, which the
    /// stage-2 retrieval loss adds to the answer loss.
    pub fn backwardExt(s: *Session, dX: ?[]const f32) !void {
        if (dX) |g| {
            if (s.ext == null) s.ext = try s.m.mem.allocT(f32, g.len);
            try gpu.upload(s.ext.?, std.mem.sliceAsBytes(g));
            s.m.dXext = s.ext;
        }
        defer s.m.dXext = null;
        try model.backwardWindow(&s.m, &s.state);
    }
    /// The representation rows at the given positions of the last forward;
    /// a negative position leaves its row untouched.
    pub fn readRows(s: *Session, last: []const i32, out: []f32) !void {
        const T: usize = @intCast(s.m.c.seqlen);
        const D: usize = @intCast(s.m.c.dim);
        const row = try s.gpa.alloc(u16, D);
        defer s.gpa.free(row);
        for (last, 0..) |at, b| {
            if (at < 0) continue;
            const src = s.m.X + (b * T + @as(usize, @intCast(at))) * D;
            try gpu.download(std.mem.sliceAsBytes(row), src);
            for (row, 0..) |bits, d| out[b * D + d] = @bitCast(@as(u32, bits) << 16);
        }
    }
    /// The logits of the last forward, converted to float.
    pub fn logits(s: *Session, out: []f32) !void {
        const raw = try s.gpa.alloc(u16, out.len);
        defer s.gpa.free(raw);
        try gpu.download(std.mem.sliceAsBytes(raw), s.m.logits);
        for (raw, 0..) |bits, i| out[i] = @bitCast(@as(u32, bits) << 16);
    }
    /// Index of a stage-2 retrieval head (0 = query, 1 = key), null if absent.
    pub fn retrievalParam(s: *const Session, which: usize) ?usize {
        if (s.m.c.mem == 0 or s.m.c.mem_rdim <= 0) return null;
        return if (which == 0) s.m.MS.rq else s.m.MS.rk;
    }
    pub fn paramMaster(s: *const Session, j: usize, out: []f32) !void {
        try gpu.download(std.mem.sliceAsBytes(out), s.m.store.at(j).master);
    }
    pub fn setParamGrad(s: *const Session, j: usize, in: []const f32) !void {
        try gpu.upload(s.m.store.at(j).grad, std.mem.sliceAsBytes(in));
    }

    /// A gradient accumulator over the several passes of one step: the
    /// knowledge-graph trainer runs an answer pass and, with retrieval, one
    /// reading pass per candidate batch before it updates.
    fn ensureAcc(s: *Session) !void {
        if (s.acc.len != 0) return;
        s.acc = try s.gpa.alloc([*]f32, s.m.store.values.items.len);
        for (s.acc, 0..) |*a, j| a.* = try s.m.mem.allocT(f32, @intCast(s.m.store.at(j).n));
    }
    pub fn accZero(s: *Session) !void {
        try ensureAcc(s);
        for (s.acc, 0..) |a, j| try gpu.zero(a, @as(usize, @intCast(s.m.store.at(j).n)) * 4);
    }
    /// Adds the current gradients of one parameter, or of all of them.
    pub fn accAdd(s: *Session, only: ?usize) !void {
        try ensureAcc(s);
        const first = only orelse 0;
        const last = if (only) |j| j + 1 else s.acc.len;
        for (first..last) |j| {
            const n: usize = @intCast(s.m.store.at(j).n);
            try (try s.kernels.get("add_f32")).launch(@intCast((n + 255) / 256), 256,
                .{ s.acc[j], s.m.store.at(j).grad, s.m.store.at(j).n });
        }
    }
    pub fn accStore(s: *Session) !void {
        try ensureAcc(s);
        for (s.acc, 0..) |a, j| {
            const n: usize = @intCast(s.m.store.at(j).n);
            try gpu.copyDevice(s.m.store.at(j).grad, a, n * 4);
        }
    }

    pub fn paramCount(s: *const Session) usize {
        return s.m.store.values.items.len;
    }
    pub fn paramSize(s: *const Session, j: usize) usize {
        return @intCast(s.m.store.at(j).n);
    }
    /// Which group a parameter belongs to, for the gradient comparison.
    pub fn paramGroup(s: *const Session, j: usize) []const u8 {
        const m = &s.m;
        if (j == m.emb) return "embedding";
        if (j == m.dec) return "decoder";
        for (m.L, 0..) |ly, l| {
            if (j == ly.decay) return "decay";
            if (j == ly.gate and m.c.gated != 0) return "gate";
            if (j == ly.gamma or j == ly.beta) return "norm";
            if (j == ly.router and m.c.experts > 1) return "router";
            for (0..@intCast(m.c.experts)) |e| if (j == ly.exp[e]) return "experts";
            if (m.ML[l].use) {
                const q = m.ML[l].p;
                for ([_]usize{ q.q, q.dkv, q.kr, q.uk, q.uv, q.o, q.gamma, q.beta }) |id|
                    if (j == id) return "mla";
            }
        }
        return "";
    }
    pub fn paramGrad(s: *const Session, j: usize, out: []f32) !void {
        try gpu.download(std.mem.sliceAsBytes(out), s.m.store.at(j).grad);
    }

    /// Tokens per expert of the last window, one entry per layer and expert.
    pub fn expertCounts(s: *const Session, out: []i64) void {
        const E: usize = @intCast(s.m.c.experts);
        for (s.m.L, 0..) |ly, l|
            for (0..E) |e| {
                out[l * E + e] = ly.mc.hcnt[e];
            };
    }

    /// How much of the decay, gate and embedding gradient comes from the
    /// traces rather than from inside the window: {pp, ww, pw} for each.
    pub fn traceStats(s: *Session, out: *[9]f64) !void {
        const c = s.m.c;
        const D: usize = @intCast(c.dim);
        const L: usize = @intCast(c.layers);
        const trlog = s.m.trlog orelse return error.NoTraces;
        const log = try s.gpa.alloc(f32, (2 * L + 256) * D);
        defer s.gpa.free(log);
        try gpu.download(std.mem.sliceAsBytes(log), trlog);
        const group = struct {
            fn f(sess: *Session, logs: []const f32, ids: []const usize, offs: []const usize, r: *[3]f64) !void {
                var pp: f64 = 0;
                var ww: f64 = 0;
                var pw: f64 = 0;
                for (ids, offs) |param, off| {
                    const n: usize = @intCast(sess.m.store.at(param).n);
                    const g = try sess.gpa.alloc(f32, n);
                    defer sess.gpa.free(g);
                    try gpu.download(std.mem.sliceAsBytes(g), sess.m.store.at(param).grad);
                    for (g, 0..) |gv, i| {
                        const p: f64 = logs[off + i];
                        const w: f64 = @as(f64, gv) - p;
                        pp += p * p;
                        ww += w * w;
                        pw += p * w;
                    }
                }
                r.* = .{ pp, ww, pw };
            }
        }.f;
        const dec = try s.gpa.alloc(usize, L);
        defer s.gpa.free(dec);
        const gate = try s.gpa.alloc(usize, L);
        defer s.gpa.free(gate);
        const dec_off = try s.gpa.alloc(usize, L);
        defer s.gpa.free(dec_off);
        const gate_off = try s.gpa.alloc(usize, L);
        defer s.gpa.free(gate_off);
        for (s.m.L, 0..) |ly, l| {
            dec[l] = ly.decay;
            gate[l] = ly.gate;
            dec_off[l] = l * D;
            gate_off[l] = (L + l) * D;
        }
        try group(s, log, dec, dec_off, out[0..3]);
        try group(s, log, gate, gate_off, out[3..6]);
        try group(s, log, &.{s.m.emb}, &.{2 * L * D}, out[6..9]);
    }

    /// Mean |state| per half-life bucket (<16, <128, <1k, <8k, >=8k).
    pub fn stateBuckets(s: *Session, sum: *[5]f64, count: *[5]i64) !void {
        const c = s.m.c;
        const D: usize = @intCast(c.dim);
        const B: usize = @intCast(c.batch);
        const edges = [_]f64{ 16, 128, 1024, 8192, 1e300 };
        sum.* = @splat(0);
        count.* = @splat(0);
        const decay = try s.gpa.alloc(f32, D);
        defer s.gpa.free(decay);
        const carry = try s.gpa.alloc(f32, B * D);
        defer s.gpa.free(carry);
        for (s.m.L, 0..) |ly, l| {
            try gpu.download(std.mem.sliceAsBytes(decay), s.m.store.at(ly.decay).master);
            try gpu.download(std.mem.sliceAsBytes(carry), s.state.carry[l]);
            for (decay, 0..) |dv, d| {
                const a = 1 / (1 + @exp(-@as(f64, dv)));
                const half = if (a >= 1) 1e300 else @log(0.5) / @log(a);
                var k: usize = 0;
                while (half >= edges[k]) k += 1;
                for (0..B) |b| sum[k] += @abs(@as(f64, carry[b * D + d]));
                count[k] += @intCast(B);
            }
        }
    }
};

// SIGINT and SIGTERM set a flag instead of ending the run, so training stops
// after the current window and still writes its checkpoint.
var stop_flag = std.atomic.Value(bool).init(false);
fn onStop(_: std.os.linux.SIG) callconv(.c) void {
    stop_flag.store(true, .monotonic);
}
pub fn installStopHandler() void {
    const act = std.posix.Sigaction{
        .handler = .{ .handler = onStop },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.INT, &act, null);
    std.posix.sigaction(std.posix.SIG.TERM, &act, null);
}
pub fn stopRequested() bool {
    return stop_flag.load(.monotonic);
}
