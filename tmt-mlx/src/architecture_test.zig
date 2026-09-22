// Native regression tests; no Python dependency.
//   architecture_test                 run all tests
//   architecture_test --compare A B   compare two checkpoints (CLI resume tests)
const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const cell = @import("cell.zig");
const config = @import("config.zig");
const ckpt = @import("checkpoint.zig");
const u = @import("util.zig");
const Array = mx.Array;
const Cfg = config.Cfg;
const gpa = u.gpa;

fn require(ok: bool, what: []const u8) void {
    if (!ok) u.die("FAIL: {s}", .{what});
}
fn near(got: f64, ref: f64, atol: f64, rtol: f64, what: []const u8) void {
    if (!std.math.isFinite(got) or !std.math.isFinite(ref) or @abs(got - ref) > atol + rtol * @abs(ref))
        u.die("FAIL: {s}: got {d:.9} expected {d:.9}", .{ what, got, ref });
}
/// Round to the nearest bf16 value (ties to even), as the device conversion does.
fn bfr(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return @bitCast((b +% 0x7FFF +% ((b >> 16) & 1)) & 0xFFFF0000);
}
/// Round to the working precision of the forward pass (TMT_COMPUTE).
fn cround(x: f32) f32 {
    return switch (model.cdt) {
        mx.bf16 => bfr(x),
        mx.f16_ => @floatCast(@as(f16, @floatCast(x))),
        else => x,
    };
}
fn f(i: usize) f32 {
    return @floatFromInt(i);
}

fn smallConfig(experts: i32) Cfg {
    var c = Cfg{};
    c.dim = 16;
    c.layers = 2;
    c.batch = 2;
    c.seqlen = 7;
    c.experts = experts;
    c.topk = if (experts == 1) 1 else 2;
    c.warmup = 0;
    c.lr = 0.005;
    c.half_max = 16;
    return c;
}

/// Repeating "ABC" pattern: ids 65+i%3, targets the next byte.
fn inputs(c: Cfg) model.Batch {
    const n: usize = @intCast(c.batch * c.seqlen);
    const x = gpa.alloc(i32, n) catch unreachable;
    const y = gpa.alloc(i32, n) catch unreachable;
    const e = gpa.alloc(i32, n) catch unreachable;
    for (0..n) |i| {
        x[i] = 65 + @as(i32, @intCast(i % 3));
        y[i] = 65 + @as(i32, @intCast((i + 1) % 3));
        e[i] = 0;
    }
    return keepBatch(model.makeBatch(c, x, y, e));
}
fn keepBatch(b: model.Batch) model.Batch {
    return .{ .ids = mx.keep(b.ids), .nxt = mx.keep(b.nxt), .end = mx.keep(b.end) };
}
fn step(m: *model.Model, st: *model.State, b: model.Batch, grad: bool) model.Result {
    const m0 = mx.mark();
    defer mx.release(m0);
    return model.forwardWindow(m, st, b, .{ .grad = grad });
}
fn setArray(a: *Array, values: []const f32, shp: []const i32) void {
    const m0 = mx.mark();
    defer mx.release(m0);
    mx.assign(a, mx.fromSlice(f32, values, shp));
}
fn copyParam(from: *model.Model, i: usize, to: *model.Model, j: usize) void {
    const v = model.getMaster(from, i);
    defer gpa.free(v);
    model.setMaster(to, j, v);
}
fn relError(x: []const f32, y: []const f32) [2]f64 {
    var e: f64 = 0;
    var n: f64 = 0;
    for (x, y) |a, b| {
        e += std.math.pow(f64, a - b, 2);
        n += @as(f64, b) * b;
    }
    return .{ @sqrt(e), @sqrt(n) };
}

// ---------------------------------------------------------------------------

fn testCell() void {
    const B = 2;
    const T = 5;
    const D = 3;
    const N = B * T * D;
    var x: [N]f32 = undefined;
    var g: [N]f32 = undefined;
    for (0..N) |i| {
        x[i] = bfr(@sin(f(i) * 0.71));
        g[i] = bfr(@cos(f(i) * 0.37));
    }
    const initial = [_]f32{ 0.3, -0.5, 0.8, -0.4, 0.7, 0.2 };
    const decay = [_]f32{ -0.4, 0.5, 2 };
    const gate = [_]f32{ 0.7, -0.3, 0.4 };
    const m0 = mx.mark();
    defer mx.release(m0);
    const X = mx.astype(mx.fromSlice(f32, &x, &.{ B, T, D }), mx.bf16);
    const G = mx.fromSlice(f32, &g, &.{ B, T, D });
    const C = mx.fromSlice(f32, &initial, &.{ B, D });
    const A = mx.fromSlice(f32, &decay, &.{D});
    const Q = mx.fromSlice(f32, &gate, &.{D});
    const ids = mx.zeros(&.{ B, T }, mx.i32_);
    for ([_]bool{ false, true }) |gated| {
        const o = cell.Opt{ .gated = gated, .docsep = -1 };
        const S = cell.forwardRaw(X, A, Q, C, ids, o);
        const r = cell.backwardRaw(G, S, X, A, Q, C, ids, o);
        var state: [N]f32 = undefined;
        var dx: [N]f32 = undefined;
        var da: [D]f32 = undefined;
        var dq: [D]f32 = undefined;
        var dc: [B * D]f32 = undefined;
        mx.toF32(S, &state);
        mx.toF32(r.dx, &dx);
        mx.toF32(mx.sum(r.gdec, 0, false), &da);
        mx.toF32(mx.sum(r.ggate, 0, false), &dq);
        mx.toF32(r.dcarry, &dc);
        var xx: [N]f64 = undefined;
        var aa: [D]f64 = undefined;
        var qq: [D]f64 = undefined;
        var cc: [B * D]f64 = undefined;
        for (0..N) |i| xx[i] = x[i];
        for (0..D) |i| {
            aa[i] = decay[i];
            qq[i] = gate[i];
        }
        for (0..B * D) |i| cc[i] = initial[i];
        const Obj = struct {
            fn eval(xv: []const f64, av: []const f64, qv: []const f64, cv: []const f64, gr: []const f32, gt: bool, out: ?[]const f32) f64 {
                var result: f64 = 0;
                for (0..B) |b| for (0..D) |d| {
                    var st = cv[b * D + d];
                    for (0..T) |t| {
                        const i = (b * T + t) * D + d;
                        const a = 1 / (1 + @exp(-av[d] - (if (gt) qv[d] * xv[i] else 0)));
                        st = a * st + (1 - a) * xv[i];
                        if (out) |o_| near(o_[i], st, 2e-6, 2e-6, "cell forward");
                        result += st * gr[i];
                    }
                };
                return result;
            }
        };
        _ = Obj.eval(&xx, &aa, &qq, &cc, &g, gated, &state);
        const eps = 1e-4;
        const diff = struct {
            fn d(v: []f64, i: usize, xv: []const f64, av: []const f64, qv: []const f64, cv: []const f64, gr: []const f32, gt: bool) f64 {
                const old = v[i];
                v[i] = old + eps;
                const plus = Obj.eval(xv, av, qv, cv, gr, gt, null);
                v[i] = old - eps;
                const minus = Obj.eval(xv, av, qv, cv, gr, gt, null);
                v[i] = old;
                return (plus - minus) / (2 * eps);
            }
        }.d;
        for (0..N) |i| near(dx[i], diff(&xx, i, &xx, &aa, &qq, &cc, &g, gated), 2e-5, 2e-4, "cell input gradient");
        for (0..D) |i| {
            near(da[i], diff(&aa, i, &xx, &aa, &qq, &cc, &g, gated), 2e-5, 2e-4, "cell decay gradient with carry");
            if (gated) near(dq[i], diff(&qq, i, &xx, &aa, &qq, &cc, &g, gated), 2e-5, 2e-4, "cell gate gradient");
        }
        for (0..B * D) |i| near(dc[i], diff(&cc, i, &xx, &aa, &qq, &cc, &g, gated), 2e-5, 2e-4, "cell carry gradient");
        // The differentiable wrapper returns the same gradients through MLX autodiff.
        const rec = cell.Recurrence.init(o);
        const Ctx = struct { rec: cell.Recurrence, G: Array, ids: Array };
        var ctx = Ctx{ .rec = rec, .G = G, .ids = ids };
        const loss = struct {
            fn l(p: *anyopaque, in: []const Array) []Array {
                const k: *Ctx = @ptrCast(@alignCast(p));
                const out = gpa.alloc(Array, 1) catch unreachable;
                out[0] = mx.sumAll(mx.mul(k.rec.apply(in[0], in[1], in[2], in[3], k.ids), k.G));
                return out;
            }
        }.l;
        const vg = mx.valueAndGrad(loss, &ctx, &.{ X, A, Q, C }, &.{ 0, 1, 2, 3 });
        var gx: [N]f32 = undefined;
        mx.toF32(vg.grads[0], &gx);
        for (0..N) |i| near(gx[i], dx[i], 2e-2, 1e-2, "autodiff input gradient (bf16)");
        var ga: [D]f32 = undefined;
        mx.toF32(vg.grads[1], &ga);
        for (0..D) |i| near(ga[i], da[i], 1e-6, 1e-5, "autodiff decay gradient");
        var gc: [B * D]f32 = undefined;
        mx.toF32(vg.grads[3], &gc);
        for (0..B * D) |i| near(gc[i], dc[i], 1e-6, 1e-5, "autodiff carry gradient");
    }
    u.print("PASS normalized/gated recurrence: forward and finite-difference gradients with nonzero carry\n", .{});
}

fn testRouter() void {
    const N = 5;
    const E = 4;
    const K = 2;
    var logits: [N * E]f32 = undefined;
    for (0..N * E) |i| logits[i] = @sin(f(i) * 0.43) * 2;
    // Host top-k by softmax probability, as the model selects experts.
    var cnt = [_]f32{0} ** E;
    for (0..N) |r| {
        var used = [_]bool{false} ** E;
        for (0..K) |_| {
            var best: usize = 0;
            var bv: f32 = -1e30;
            for (0..E) |e| if (!used[e] and logits[r * E + e] > bv) {
                bv = logits[r * E + e];
                best = e;
            };
            used[best] = true;
            cnt[best] += 1;
        }
    }
    const m0 = mx.mark();
    defer mx.release(m0);
    const L = mx.fromSlice(f32, &logits, &.{ N, E });
    const counts = mx.fromSlice(f32, &cnt, &.{E});
    for ([_][2]f32{ .{ 0.7, 0 }, .{ 0, 0.3 } }) |coef| {
        const Ctx = struct { counts: Array, coef: [2]f32 };
        var ctx = Ctx{ .counts = counts, .coef = coef };
        const loss = struct {
            fn l(p: *anyopaque, in: []const Array) []Array {
                const k: *Ctx = @ptrCast(@alignCast(p));
                const reg = model.routerRegularizers(in[0], k.counts, K);
                const out = gpa.alloc(Array, 1) catch unreachable;
                out[0] = mx.add(mx.mulS(reg[0], k.coef[0]), mx.mulS(reg[1], k.coef[1]));
                return out;
            }
        }.l;
        const vg = mx.valueAndGrad(loss, &ctx, &.{L}, &.{0});
        var got: [N * E]f32 = undefined;
        mx.toF32(vg.grads[0], &got);
        const obj = struct {
            fn o(z: []const f64, cf: [2]f32, cn: []const f32) f64 {
                var total: f64 = 0;
                for (0..N) |r| {
                    var s: f64 = 0;
                    for (0..E) |e| s += @exp(z[r * E + e]);
                    const lse = @log(s);
                    total += cf[1] * lse * lse / N;
                    for (0..E) |e| total += cf[0] * E / @as(f64, N) * @exp(z[r * E + e]) / s * cn[e] / (N * K);
                }
                return total;
            }
        }.o;
        var z: [N * E]f64 = undefined;
        for (0..N * E) |i| z[i] = logits[i];
        for (0..N * E) |i| {
            const old = z[i];
            z[i] = old + 1e-4;
            const hi = obj(&z, coef, &cnt);
            z[i] = old - 1e-4;
            const lo = obj(&z, coef, &cnt);
            z[i] = old;
            near(got[i], (hi - lo) / 2e-4, 2e-6, 2e-4, "router regularizer gradient");
        }
    }
    u.print("PASS MoE auxiliary and z-loss finite-difference gradients\n", .{});
}

// MoE backward against the dense path: two experts with identical weights,
// both selected (normalized top-k weights sum to 1), compute exactly the dense
// layer. The summed expert gradients and all other gradients must match.
fn testMoeBackward() void {
    var dense = smallConfig(1);
    var moe = smallConfig(2);
    moe.topk = 2;
    dense.aux = 0;
    moe.aux = 0;
    dense.zloss = 0;
    moe.zloss = 0;
    var a = model.build(dense);
    var b = model.build(moe);
    for (a.layers, b.layers) |la, lb| {
        copyParam(&a, la.exp[0], &b, lb.exp[0]);
        copyParam(&a, la.exp[0], &b, lb.exp[1]);
        copyParam(&a, la.decay, &b, lb.decay);
        copyParam(&a, la.gate, &b, lb.gate);
        copyParam(&a, la.gamma, &b, lb.gamma);
        copyParam(&a, la.beta, &b, lb.beta);
    }
    copyParam(&a, a.emb, &b, b.emb);
    copyParam(&a, a.dec, &b, b.dec);
    copyParam(&a, a.stop, &b, b.stop);
    copyParam(&a, a.tgt, &b, b.tgt);
    var sa = model.State.init(&a);
    var sb = model.State.init(&b);
    const ra = step(&a, &sa, inputs(dense), true);
    const rb = step(&b, &sb, inputs(moe), true);
    near(rb.ce, ra.ce, 1e-3, 1e-3, "MoE with identical experts differs from dense forward");
    var l: usize = a.layers.len;
    while (l > 0) {
        l -= 1;
        const ref = model.getGrad(&a, a.layers[l].exp[0]);
        const g0 = model.getGrad(&b, b.layers[l].exp[0]);
        const g1 = model.getGrad(&b, b.layers[l].exp[1]);
        for (g0, g1) |*x, y| x.* += y;
        var e = relError(g0, ref);
        require(e[1] > 0, "reference gradient is zero");
        near(e[0], 0, 1e-6 + 0.03 * e[1], 0, "summed MoE expert gradients differ from dense");
        e = relError(model.getGrad(&b, b.layers[l].decay), model.getGrad(&a, a.layers[l].decay));
        near(e[0], 0, 1e-6 + 0.03 * e[1], 0, "gradient below MoE layer differs from dense");
    }
    const e = relError(model.getGrad(&b, b.emb), model.getGrad(&a, a.emb));
    near(e[0], 0, 1e-6 + 0.03 * e[1], 0, "embedding gradient through MoE differs from dense");
    u.print("PASS MoE backward: identical experts reproduce dense expert and input gradients\n", .{});
}

// Optimizer against an independent host reference: global-norm clipping, AdamW
// on every parameter except the EMA target and an unused stop head, EMA target
// update, and a non-finite gradient refusing the whole step.
fn testOptimizer() void {
    var c = smallConfig(3);
    c.gradclip = 0.05;
    var m = model.build(c);
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    const P = m.nparams();
    const w = gpa.alloc([]f32, P) catch unreachable;
    const mm = gpa.alloc([]f32, P) catch unreachable;
    const vv = gpa.alloc([]f32, P) catch unreachable;
    const g = gpa.alloc([]f32, P) catch unreachable;
    for (0..P) |j| {
        const p = m.at(j);
        w[j] = model.getMaster(&m, j);
        mm[j] = gpa.alloc(f32, p.n) catch unreachable;
        vv[j] = gpa.alloc(f32, p.n) catch unreachable;
        g[j] = gpa.alloc(f32, p.n) catch unreachable;
        for (0..p.n) |i| {
            mm[j][i] = (r.float(f32) * 2 - 1) * 0.01;
            vv[j][i] = r.float(f32) * 1e-4;
            g[j][i] = r.float(f32) * 2 - 1;
        }
        setArray(&p.m, mm[j], p.shp());
        setArray(&p.v, vv[j], p.shp());
        setArray(&m.grads[j], g[j], p.shp());
    }
    const s = 5;
    model.optimizerStep(&m, s) catch u.die("FAIL: optimizer refused a finite step", .{});
    var sumsq: f64 = 0;
    for (0..P) |j| if (j != m.tgt) for (g[j]) |e| {
        sumsq += @as(f64, e) * e;
    };
    const scale = @min(1.0, c.gradclip / @max(@sqrt(sumsq), 1e-12));
    require(scale < 1, "test does not exercise clipping");
    const B1: f64 = @as(f32, 0.9);
    const B2: f64 = @as(f32, 0.999);
    const lr: f64 = model.lrAt(c, s);
    const bc1: f64 = 1 - std.math.pow(f32, 0.9, s + 1);
    const bc2: f64 = 1 - std.math.pow(f32, 0.999, s + 1);
    for (0..P) |j| {
        if (j == m.tgt) continue;
        const p = m.at(j);
        const gw = mx.toVecF32(p.master);
        const gm = mx.toVecF32(p.m);
        const gv = mx.toVecF32(p.v);
        for (0..p.n) |i| {
            if (j == m.stop) {
                near(gw[i], w[j][i], 0, 0, "unused stop head was updated");
                continue;
            }
            const gg = g[j][i] * scale;
            const nm = B1 * mm[j][i] + (1 - B1) * gg;
            const nv = B2 * vv[j][i] + (1 - B2) * gg * gg;
            const nw = w[j][i] - lr * ((nm / bc1) / (@sqrt(nv / bc2) + 1e-8) + 0.01 * w[j][i]);
            near(gm[i], nm, 1e-8, 1e-4, "optimizer first moment");
            near(gv[i], nv, 1e-10, 1e-4, "optimizer second moment");
            near(gw[i], nw, 1e-7, 1e-5, "optimizer weight");
        }
    }
    const tgt = model.getMaster(&m, m.tgt);
    const emb = model.getMaster(&m, m.emb);
    for (tgt, emb, 0..) |t, e, i| near(t, c.ematau * w[m.tgt][i] + (1 - c.ematau) * e, 1e-6, 1e-5, "EMA target");
    // Non-finite gradient: the step is refused and nothing changes.
    const before = model.getMaster(&m, m.emb);
    g[m.dec][0] = std.math.nan(f32);
    setArray(&m.grads[m.dec], g[m.dec], m.at(m.dec).shp());
    const refused = if (model.optimizerStep(&m, s + 1)) false else |_| true;
    require(refused, "non-finite gradient accepted");
    const after = model.getMaster(&m, m.emb);
    for (after, before) |x, y| near(x, y, 0, 0, "weights changed by a refused step");
    u.print("PASS optimizer: clipping, AdamW, EMA and refused steps match the reference ({d} parameters)\n", .{P});
}

fn testModel(experts: i32) void {
    var m = model.build(smallConfig(experts));
    var st = model.State.init(&m);
    const b = inputs(m.c);
    _ = step(&m, &st, b, false);
    const before = mx.toVecF32(m.logits);
    st.reset();
    const wi = m.layers[0].exp[0];
    const saved = model.getMaster(&m, wi);
    const changed = gpa.dupe(f32, saved) catch unreachable;
    for (changed, 0..) |*v, i| v.* += (@as(f32, @floatFromInt(i % 5)) - 2) * 0.1;
    model.setMaster(&m, wi, changed);
    _ = step(&m, &st, b, false);
    const after = mx.toVecF32(m.logits);
    var delta: f64 = 0;
    for (before, after) |x, y| delta += @abs(x - y);
    require(delta > 0.01, "higher layers ignore lower expert output");
    model.setMaster(&m, wi, saved);
    var first: f32 = 0;
    var last: f32 = 0;
    for (0..30) |s| {
        st.reset();
        const r = step(&m, &st, b, true);
        if (s == 0) first = r.ce;
        last = r.ce;
        model.optimizerStep(&m, s) catch u.die("FAIL: step refused", .{});
    }
    require(last < first * 0.8, "training does not reduce CE on repeated byte pattern");
    u.print("PASS hierarchical {s} model learns: CE {d:.4} -> {d:.4}\n", .{ if (experts == 1) "dense" else "MoE", first, last });
    // Two separately owned states produce the same outputs from reset.
    var other = model.State.init(&m);
    st.reset();
    _ = step(&m, &st, b, false);
    const out1 = mx.toVecF32(m.logits);
    _ = step(&m, &other, b, false);
    const out2 = mx.toVecF32(m.logits);
    for (out1, out2) |x, y| near(x, y, 0, 0, "independent stream state");
    // Exact resume: weights + Adam + progress + carry, followed by another update.
    var pbuf: [128]u8 = undefined;
    const path = std.fmt.bufPrintZ(&pbuf, "/tmp/tmt-mlx-test-{d}-{d}.ckpt", .{ u.c.getpid(), experts }) catch unreachable;
    var progress = ckpt.Progress{ .step = 30, .cursor = 7, .carried = 7, .data_size = 1000, .data_hash = 42 };
    ckpt.save(path, &m, &st, &progress) catch u.die("FAIL: save: {s}", .{ckpt.last_error});
    var restored = model.build(ckpt.readConfig(path) catch u.die("FAIL: {s}", .{ckpt.last_error}));
    var rs = model.State.init(&restored);
    var rp = ckpt.Progress{};
    ckpt.load(path, &restored, &rs, &rp) catch u.die("FAIL: load: {s}", .{ckpt.last_error});
    require(rp.step == 30 and rp.cursor == 7 and rs.position == 7, "checkpoint progress mismatch");
    _ = step(&m, &st, b, true);
    model.optimizerStep(&m, 30) catch unreachable;
    _ = step(&restored, &rs, b, true);
    model.optimizerStep(&restored, 30) catch unreachable;
    for (0..m.nparams()) |j| {
        const x = model.getMaster(&m, j);
        const y = model.getMaster(&restored, j);
        for (x, y) |p, q| near(p, q, 1e-6, 1e-5, "checkpoint resumed update");
    }
    {
        const fp = u.c.fopen(path.ptr, "r+b");
        _ = u.c.fseek(fp, 20, u.c.SEEK_SET);
        const byte = u.c.fgetc(fp);
        _ = u.c.fseek(fp, 20, u.c.SEEK_SET);
        _ = u.c.fputc(byte ^ 1, fp);
        _ = u.c.fclose(fp);
    }
    const rejected = if (ckpt.verify(path)) false else |_| true;
    _ = u.c.unlink(path.ptr);
    require(rejected, "corrupted checkpoint accepted");
    u.print("PASS separate streams, checkpoint resume and corruption rejection\n", .{});
}

// Hybrid traces: with one layer only layer 0's carry crosses the window
// boundary, and it depends only on decay, gate and embedding. Two T-windows with
// traces must therefore reproduce the full BPTT gradient of one 2T-window for
// ALL parameters; without traces they must not.
const TT = 6;
fn traceConfig(seqlen: i32, traces: i32, docsep: i32) Cfg {
    var c = smallConfig(1);
    c.layers = 1;
    c.seqlen = seqlen;
    c.traces = traces;
    c.aux = 0;
    c.zloss = 0;
    c.docsep = docsep;
    return c;
}
fn traceSeq() [2 * (2 * TT + 1)]i32 {
    var seq: [2 * (2 * TT + 1)]i32 = undefined;
    for (0..seq.len) |i| seq[i] = 65 + @as(i32, @intCast((i * 7 + i / 5) % 11));
    return seq;
}
fn traceFeed(m: *model.Model, st: *model.State, offset: usize, seq: []const i32) void {
    const B: usize = @intCast(m.c.batch);
    const L: usize = @intCast(m.c.seqlen);
    var x: [64]i32 = undefined;
    var y: [64]i32 = undefined;
    var e: [64]i32 = @splat(0);
    for (0..B) |b| for (0..L) |t| {
        x[b * L + t] = seq[b * (2 * TT + 1) + offset + t];
        y[b * L + t] = seq[b * (2 * TT + 1) + offset + t + 1];
    };
    const m0 = mx.mark();
    defer mx.release(m0);
    _ = model.forwardWindow(m, st, model.makeBatch(m.c, x[0 .. B * L], y[0 .. B * L], e[0 .. B * L]), .{ .grad = true });
}
fn nonzeroGate(m: *model.Model) void {
    const i = m.layers[0].gate;
    const h = gpa.alloc(f32, m.at(i).n) catch unreachable;
    for (h, 0..) |*v, d| v.* = 0.8 * @sin(f(d) * 1.3 + 0.4);
    model.setMaster(m, i, h);
}
fn allGrads(m: *model.Model) [][]f32 {
    const g = gpa.alloc([]f32, m.nparams()) catch unreachable;
    for (g, 0..) |*x, j| x.* = model.getGrad(m, j);
    return g;
}
fn twoWindows(traces: i32, docsep: i32, seq: []const i32) [][]f32 {
    var m = model.build(traceConfig(TT, traces, docsep));
    nonzeroGate(&m);
    var st = model.State.init(&m);
    traceFeed(&m, &st, 0, seq);
    const g = allGrads(&m);
    traceFeed(&m, &st, TT, seq);
    const h = allGrads(&m);
    for (g, h) |x, y| for (x, y) |*p, q| {
        p.* += q;
    };
    return g;
}
fn testTraces(docsep: i32) void {
    const seq = traceSeq();
    var full = model.build(traceConfig(2 * TT, 0, docsep));
    nonzeroGate(&full);
    var fs = model.State.init(&full);
    traceFeed(&full, &fs, 0, &seq);
    const ref = allGrads(&full);
    for (ref) |r| for (r) |*v| {
        v.* *= 2; // sum of two window means = 2 x the mean over the double window
    };
    const hybrid = twoWindows(1, docsep, &seq);
    const plain = twoWindows(0, docsep, &seq);
    for (0..ref.len) |j| {
        if (j == full.tgt or j == full.stop) continue;
        const e = relError(hybrid[j], ref[j]);
        near(e[0], 0, 1e-5 + 0.02 * e[1], 0, "hybrid trace gradient vs full BPTT");
    }
    if (docsep >= 0) {
        u.print("PASS hybrid traces stay exact with document resets (docsep={d})\n", .{docsep});
        return;
    }
    var worst_hybrid: f64 = 0;
    var best_plain: f64 = 1e30;
    for ([_]usize{ full.layers[0].decay, full.layers[0].gate, full.emb }) |j| {
        const eh = relError(hybrid[j], ref[j]);
        const ep = relError(plain[j], ref[j]);
        require(ep[0] > 0.05 * ep[1] and ep[0] > 10 * eh[0], "traces do not change the cross-window gradient");
        worst_hybrid = @max(worst_hybrid, eh[0] / eh[1]);
        best_plain = @min(best_plain, ep[0] / ep[1]);
    }
    u.print("PASS hybrid traces reproduce full BPTT across a window boundary (1 layer, all parameters; decay/gate/emb rel. error {e:.1} with traces vs >= {d:.2} without)\n", .{ worst_hybrid, best_plain });
}

// Document reset: after a separator byte, outputs no longer depend on the
// previous document (a_t = 0 cuts carry and traces through the recurrence).
fn testDocsep() void {
    var c = smallConfig(1);
    c.traces = 1;
    c.docsep = 88;
    c.seqlen = 9;
    var m = model.build(c);
    var st = model.State.init(&m);
    const B: usize = @intCast(c.batch);
    const T: usize = @intCast(c.seqlen);
    const sep = 4;
    var outs: [2][]f32 = undefined;
    for (0..2) |variant| {
        var x: [18]i32 = undefined;
        var y: [18]i32 = undefined;
        var e: [18]i32 = @splat(0);
        for (0..B) |b| for (0..T) |t| {
            x[b * T + t] = @intCast(if (t < sep) 65 + (t * 3 + variant * 5 + b) % 7 else if (t == sep) 88 else 66 + (t + b) % 5);
            y[b * T + t] = @intCast(66 + (t + 1) % 5);
        };
        st.reset();
        // A nonzero incoming carry from an earlier "document" must not leak either.
        const carry = gpa.alloc(f32, B * @as(usize, @intCast(c.dim))) catch unreachable;
        for (carry, 0..) |*v, i| v.* = @sin(f(i) + f(variant) * 3);
        for (st.carry) |*a| setArray(a, carry, &.{ c.batch, c.dim });
        const m0 = mx.mark();
        _ = model.forwardWindow(&m, &st, model.makeBatch(c, &x, &y, &e), .{});
        mx.release(m0);
        outs[variant] = mx.toVecF32(m.logits);
    }
    var before: f64 = 0;
    for (0..B) |b| for (0..T) |t| for (0..256) |k| {
        const i = (b * T + t) * 256 + k;
        if (t >= sep) near(outs[0][i], outs[1][i], 0, 0, "output after separator depends on previous document") else before += @abs(outs[0][i] - outs[1][i]);
    };
    require(before > 0, "test inputs do not differ before the separator");
    u.print("PASS document reset: outputs after a separator are independent of the previous document\n", .{});
}

// trace_decay is defined per byte: after 2T bytes the traces must be the same
// whether they were advanced in windows of T bytes or of one byte.
fn traceState(seqlen: i32) []f32 {
    var c = smallConfig(1);
    c.traces = 1;
    c.trace_decay = 0.9;
    c.seqlen = seqlen;
    c.aux = 0;
    c.zloss = 0;
    var m = model.build(c);
    var st = model.State.init(&m);
    const B: usize = @intCast(c.batch);
    const S: usize = @intCast(seqlen);
    for (0..2 * TT / S) |w| {
        var x: [32]i32 = undefined;
        var y: [32]i32 = undefined;
        var e: [32]i32 = @splat(0);
        for (0..B) |b| for (0..S) |t| {
            const p = w * S + t;
            x[b * S + t] = @intCast(65 + (p * 7 + b) % 9);
            y[b * S + t] = @intCast(65 + (p * 7 + b + 7) % 9);
        };
        const m0 = mx.mark();
        _ = model.forwardWindow(&m, &st, model.makeBatch(c, x[0 .. B * S], y[0 .. B * S], e[0 .. B * S]), .{ .grad = true });
        mx.release(m0);
    }
    var all: std.ArrayList(f32) = .empty;
    all.appendSlice(gpa, mx.toVecF32(st.temb)) catch unreachable;
    for (st.tdec, st.tgate) |d, g| {
        all.appendSlice(gpa, mx.toVecF32(d)) catch unreachable;
        all.appendSlice(gpa, mx.toVecF32(g)) catch unreachable;
    }
    return all.items;
}
fn testTraceDecay() void {
    const a = traceState(TT);
    const b = traceState(1);
    const e = relError(a, b);
    near(e[0], 0, 1e-6 + 2e-3 * e[1], 0, "trace_decay depends on the window length");
    u.print("PASS trace_decay is per byte: traces after 2T bytes match for windows of T and of 1 byte\n", .{});
}

fn mlaConfig() Cfg {
    var c = smallConfig(1);
    c.layers = 1;
    c.mla = 1;
    c.mla_heads = 2;
    c.mla_dh = 4;
    c.mla_L = 4;
    c.mla_R = 6;
    c.mla_cache = 17;
    c.mla_cc = 3;
    return c;
}
fn testMlaChunks() void {
    var ca = mlaConfig();
    var cb = ca;
    cb.mla_cc = 64;
    var a = model.build(ca);
    var b = model.build(cb);
    var sa = model.State.init(&a);
    var sb = model.State.init(&b);
    const in = inputs(ca);
    ca = a.c;
    for (0..4) |round| {
        _ = step(&a, &sa, in, true);
        _ = step(&b, &sb, in, true);
        const ya = mx.toVecF32(a.logits);
        const yb = mx.toVecF32(b.logits);
        var max_error: f64 = 0;
        for (ya, yb) |x, y| max_error = @max(max_error, @abs(x - y));
        require(max_error < 0.035, "MLA output depends on chunk size");
        var max_grad_error: f64 = 0;
        var max_grad: f64 = 0;
        for (0..a.nparams()) |j| {
            const x = model.getGrad(&a, j);
            const y = model.getGrad(&b, j);
            for (x, y) |p, q| {
                max_grad_error = @max(max_grad_error, @abs(p - q));
                max_grad = @max(max_grad, @abs(q));
            }
        }
        require(max_grad_error < 0.003 + 0.04 * max_grad, "MLA gradient depends on chunk size");
        const k = sa.cache[0];
        require(k.head <= 17 and k.base0 + k.head == sa.position, "MLA cache lost absolute positions");
        if (round >= 2) require(k.head == 17 and k.base0 > 0, "MLA discards entire cache on overflow");
    }
    u.print("PASS MLA partial chunks, multi-chunk gradients, RoPE positions and prefix eviction\n", .{});
}

// Independent FP64 attention reference, including analytic backward. This is
// intentionally scalar code, separate from the MLX graph.
fn testMlaReference() void {
    var c = smallConfig(1);
    c.layers = 1;
    c.mla = 1;
    c.mla_heads = 2;
    c.mla_dh = 3;
    c.mla_L = 4;
    c.mla_R = 2;
    c.mla_cc = 3;
    var m = model.build(c);
    const B: usize = @intCast(c.batch);
    const T: usize = @intCast(c.seqlen);
    const D: usize = @intCast(c.dim);
    const H: usize = @intCast(c.mla_heads);
    const F: usize = @intCast(c.mla_dh);
    const R: usize = @intCast(c.mla_R);
    const L: usize = @intCast(c.mla_L);
    const N = B * T;
    const p = m.layers[0].mla.?;
    const x = gpa.alloc(f64, N * D) catch unreachable;
    const dy = gpa.alloc(f64, N * D) catch unreachable;
    const xf = gpa.alloc(f32, N * D) catch unreachable;
    const dyf = gpa.alloc(f32, N * D) catch unreachable;
    for (0..N * D) |i| {
        xf[i] = cround(@sin(f(i) * 0.37));
        x[i] = xf[i];
        dyf[i] = cround(@cos(f(i) * 0.19) * 0.1);
        dy[i] = dyf[i];
    }
    const weight = struct {
        fn w(mm: *model.Model, id: usize) []f64 {
            const v = model.getMaster(mm, id);
            const out = gpa.alloc(f64, v.len) catch unreachable;
            for (v, out) |a, *o| o.* = cround(a); // working-precision copy
            return out;
        }
    }.w;
    const wq = weight(&m, p.q);
    const wk = weight(&m, p.dkv);
    const wr = weight(&m, p.kr);
    const wu = weight(&m, p.uk);
    const wv = weight(&m, p.uv);
    const wo = weight(&m, p.o);
    const Lin = struct {
        fn fwd(a: []const f64, w: []const f64, rows: usize, out: usize, in: usize) []f64 {
            const y = gpa.alloc(f64, rows * out) catch unreachable;
            @memset(y, 0);
            for (0..rows) |r| for (0..out) |o| for (0..in) |i| {
                y[r * out + o] += a[r * in + i] * w[o * in + i];
            };
            return y;
        }
        fn bwd(a: []const f64, w: []const f64, grad: []const f64, rows: usize, out: usize, in: usize, da: []f64) []f64 {
            const dw = gpa.alloc(f64, out * in) catch unreachable;
            @memset(dw, 0);
            for (0..rows) |r| for (0..out) |o| for (0..in) |i| {
                dw[o * in + i] += grad[r * out + o] * a[r * in + i];
                da[r * in + i] += grad[r * out + o] * w[o * in + i];
            };
            return dw;
        }
    };
    const q = Lin.fwd(x, wq, N, H * (F + R), D);
    const lat = Lin.fwd(x, wk, N, L, D);
    const kr = Lin.fwd(x, wr, N, R, D);
    const key = Lin.fwd(lat, wu, N, H * F, L);
    const value = Lin.fwd(lat, wv, N, H * F, L);
    // Nonzero positions exercise inverse query and key rotations.
    const base = 11;
    const Rot = struct {
        fn apply(a: []f64, heads: usize, width: usize, offset: usize, inverse: bool, n_: usize, t_: usize, r_: usize, theta: f64) void {
            for (0..n_) |n| for (0..heads) |h| {
                var j: usize = 0;
                while (j < r_) : (j += 2) {
                    const angle = @as(f64, @floatFromInt(base + n % t_)) / std.math.pow(f64, theta, f(j) / f(r_)) * (if (inverse) @as(f64, -1) else 1);
                    const cs = @cos(angle);
                    const sn = @sin(angle);
                    const i = (n * heads + h) * width + offset + j;
                    const v = a[i];
                    const w = a[i + 1];
                    a[i] = cs * v - sn * w;
                    a[i + 1] = sn * v + cs * w;
                }
            };
        }
    };
    const theta: f64 = c.mla_theta;
    Rot.apply(q, H, F + R, F, false, N, T, R, theta);
    Rot.apply(kr, 1, R, 0, false, N, T, R, theta);
    const prob = gpa.alloc(f64, B * H * T * T) catch unreachable;
    @memset(prob, 0);
    const out = gpa.alloc(f64, N * H * F) catch unreachable;
    @memset(out, 0);
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(F)));
    for (0..B) |b| for (0..H) |h| for (0..T) |t| {
        const n = b * T + t;
        var maxs: f64 = -1e30;
        var sum: f64 = 0;
        for (0..t + 1) |k| {
            const nk = b * T + k;
            var s: f64 = 0;
            for (0..F) |d| s += q[(n * H + h) * (F + R) + d] * key[(nk * H + h) * F + d];
            for (0..R) |d| s += q[(n * H + h) * (F + R) + F + d] * kr[nk * R + d];
            const j = ((b * H + h) * T + t) * T + k;
            prob[j] = s * scale;
            maxs = @max(maxs, prob[j]);
        }
        for (0..t + 1) |k| {
            const j = ((b * H + h) * T + t) * T + k;
            prob[j] = @exp(prob[j] - maxs);
            sum += prob[j];
        }
        for (0..t + 1) |k| {
            const j = ((b * H + h) * T + t) * T + k;
            prob[j] /= sum;
            for (0..F) |d| out[(n * H + h) * F + d] += prob[j] * value[((b * T + k) * H + h) * F + d];
        }
    };
    const y = Lin.fwd(out, wo, N, D, H * F);
    const zeroed = struct {
        fn z(n: usize) []f64 {
            const v = gpa.alloc(f64, n) catch unreachable;
            @memset(v, 0);
            return v;
        }
    }.z;
    const dout = zeroed(N * H * F);
    const dwo = Lin.bwd(out, wo, dy, N, D, H * F, dout);
    const dq = zeroed(q.len);
    const dkr = zeroed(kr.len);
    const dk = zeroed(key.len);
    const dv = zeroed(value.len);
    for (0..B) |b| for (0..H) |h| for (0..T) |t| {
        const n = b * T + t;
        var dot: f64 = 0;
        for (0..F) |d| dot += dout[(n * H + h) * F + d] * out[(n * H + h) * F + d];
        for (0..t + 1) |k| {
            const nk = b * T + k;
            const j = ((b * H + h) * T + t) * T + k;
            var dp: f64 = 0;
            for (0..F) |d| {
                dp += dout[(n * H + h) * F + d] * value[(nk * H + h) * F + d];
                dv[(nk * H + h) * F + d] += prob[j] * dout[(n * H + h) * F + d];
            }
            const ds = scale * prob[j] * (dp - dot);
            for (0..F) |d| {
                dq[(n * H + h) * (F + R) + d] += ds * key[(nk * H + h) * F + d];
                dk[(nk * H + h) * F + d] += ds * q[(n * H + h) * (F + R) + d];
            }
            for (0..R) |d| {
                dq[(n * H + h) * (F + R) + F + d] += ds * kr[nk * R + d];
                dkr[nk * R + d] += ds * q[(n * H + h) * (F + R) + F + d];
            }
        }
    };
    Rot.apply(dq, H, F + R, F, true, N, T, R, theta);
    Rot.apply(dkr, 1, R, 0, true, N, T, R, theta);
    const dlat = zeroed(N * L);
    const dwu = Lin.bwd(lat, wu, dk, N, H * F, L, dlat);
    const dwv = Lin.bwd(lat, wv, dv, N, H * F, L, dlat);
    const dx = zeroed(N * D);
    const dwq = Lin.bwd(x, wq, dq, N, H * (F + R), D, dx);
    const dwk = Lin.bwd(x, wk, dlat, N, L, D, dx);
    const dwr = Lin.bwd(x, wr, dkr, N, R, D, dx);

    // MLX: y = MLA(X) with an empty cache at absolute position 11; loss = sum(y·dY).
    const m0 = mx.mark();
    defer mx.release(m0);
    const cache = model.Cache{ .base0 = base };
    const Ctx = struct { m: *model.Model, p: model.MlaP, cache: *const model.Cache, dy: Array, np: usize };
    var ctx = Ctx{ .m = &m, .p = p, .cache = &cache, .dy = mx.fromSlice(f32, dyf, &.{ c.batch, c.seqlen, c.dim }), .np = m.nparams() };
    const loss = struct {
        fn l(pp: *anyopaque, in: []const Array) []Array {
            const k: *Ctx = @ptrCast(@alignCast(pp));
            const yy = model.mlaApply(k.m, k.p, in[0..k.np], in[k.np], k.cache, base);
            const o = gpa.alloc(Array, 2) catch unreachable;
            o[0] = mx.sumAll(mx.mul(mx.astype(yy, mx.f32_), k.dy));
            o[1] = yy;
            return o;
        }
    }.l;
    var ins: std.ArrayList(Array) = .empty;
    for (m.params.items) |par| ins.append(gpa, par.master) catch unreachable;
    ins.append(gpa, mx.astype(mx.fromSlice(f32, xf, &.{ c.batch, c.seqlen, c.dim }), model.cdt)) catch unreachable;
    const args = [_]i32{ @intCast(p.q), @intCast(p.dkv), @intCast(p.kr), @intCast(p.uk), @intCast(p.uv), @intCast(p.o), @intCast(m.nparams()) };
    const vg = mx.valueAndGrad(loss, &ctx, ins.items, &args);
    const check = struct {
        fn ch(got_a: Array, ref: []const f64, tol: f64, name: []const u8) void {
            const got = mx.toVecF32(got_a);
            var norm: f64 = 0;
            var err: f64 = 0;
            for (got, ref) |g, r| {
                norm += r * r;
                err += (g - r) * (g - r);
            }
            near(@sqrt(err), 0, 1e-4 + tol * @sqrt(norm), 0, name);
        }
    }.ch;
    check(vg.values[1], y, 0.025, "MLA CPU forward");
    check(vg.grads[6], dx, 0.025, "MLA CPU input gradient");
    check(vg.grads[0], dwq, 0.03, "MLA Q gradient");
    check(vg.grads[1], dwk, 0.03, "MLA down KV gradient");
    check(vg.grads[2], dwr, 0.03, "MLA RoPE gradient");
    check(vg.grads[3], dwu, 0.03, "MLA up K gradient");
    check(vg.grads[4], dwv, 0.03, "MLA up V gradient");
    check(vg.grads[5], dwo, 0.03, "MLA output gradient");
    u.print("PASS MLA forward and ALL gradients against independent FP64 CPU reference\n", .{});
}

fn compareCheckpoints(first: [:0]const u8, second: [:0]const u8) void {
    const ca = ckpt.readConfig(first) catch u.die("FAIL: {s}", .{ckpt.last_error});
    const cb = ckpt.readConfig(second) catch u.die("FAIL: {s}", .{ckpt.last_error});
    require(config.equal(ca, cb), "checkpoint configs differ");
    var a = model.build(ca);
    var b = model.build(cb);
    var sa = model.State.init(&a);
    var sb = model.State.init(&b);
    var pa = ckpt.Progress{};
    var pb = ckpt.Progress{};
    ckpt.load(first, &a, &sa, &pa) catch u.die("FAIL: {s}", .{ckpt.last_error});
    ckpt.load(second, &b, &sb, &pb) catch u.die("FAIL: {s}", .{ckpt.last_error});
    require(std.meta.eql(pa, pb) and sa.position == sb.position, "checkpoint progress differs");
    const cmp = struct {
        fn c(x: Array, y: Array) void {
            const av = mx.toVecF32(x);
            const bv = mx.toVecF32(y);
            require(av.len == bv.len, "resumed state shapes differ");
            for (av, bv) |p, q| near(p, q, 1e-7, 1e-5, "resumed checkpoint values");
        }
    }.c;
    for (a.params.items, b.params.items) |x, y| {
        cmp(x.master, y.master);
        cmp(x.m, y.m);
        cmp(x.v, y.v);
    }
    if (ca.traces == 1) {
        cmp(sa.temb, sb.temb);
        for (sa.tdec, sb.tdec, sa.tgate, sb.tgate) |x, y, z, w| {
            cmp(x, y);
            cmp(z, w);
        }
    }
    for (a.layers, 0..) |ly, l| {
        cmp(sa.carry[l], sb.carry[l]);
        if (ly.mla == null) continue;
        const x = sa.cache[l];
        const y = sb.cache[l];
        require(x.head == y.head and x.base0 == y.base0, "cache metadata differs");
        if (x.head > 0) {
            const cmpc = struct {
                fn c(p: Array, q: Array) void {
                    const av = mx.toVecF32(p);
                    const bv = mx.toVecF32(q);
                    for (av, bv) |r, s| near(r, s, 1e-6, 1e-5, "resumed cache values");
                }
            }.c;
            cmpc(x.lat, y.lat);
            cmpc(x.kr, y.kr);
        }
    }
    u.print("PASS checkpoint parameters, Adam moments, data progress and stream history match within FP32 tolerance\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    model.initCompute();
    if (args.len == 4 and std.mem.eql(u8, args[1], "--compare")) {
        compareCheckpoints(args[2], args[3]);
        return;
    }
    testCell();
    testRouter();
    testMoeBackward();
    testOptimizer();
    testModel(1);
    testModel(3);
    testTraces(-1);
    testTraces(70);
    testDocsep();
    testTraceDecay();
    testMlaChunks();
    testMlaReference();
    mx.synchronize();
    u.print("ALL ARCHITECTURE TESTS PASSED\n", .{});
}
