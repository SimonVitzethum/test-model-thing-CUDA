// MLX model, explicit streaming state and the shared forward path.
//
// The forward pass is one pure function of arrays. Training differentiates it
// with MLX (value_and_grad); the recurrence has a hand-written Metal backward
// (cell.zig). Train, eval and sampling all run exactly this forward.
//
//   build(c) -> Model;  State.init(&m)
//   forwardWindow(&m, &state, batch, .{ .grad = true })   // grads in m.grads
//   optimizerStep(&m, step)
const std = @import("std");
const mx = @import("mlx.zig");
const cell = @import("cell.zig");
const config = @import("config.zig");
const u = @import("util.zig");
const Array = mx.Array;
const Cfg = config.Cfg;

/// Working precision of weights/activations in the forward pass. Masters,
/// optimizer state, recurrent state and reductions are always f32. The MLA
/// cache is stored in bf16 (checkpoint format). TMT_COMPUTE picks the working
/// copies. Apple GPUs have no bf16 matmul units, so the CUDA working precision
/// is not the fast one here: f32 measured ~17% faster than bf16 end to end (and
/// f16 sits in between). The default is therefore f32; TMT_COMPUTE=bf16 mirrors
/// the CUDA numerics and halves activation memory, TMT_COMPUTE=f16 trades range
/// for a slightly smaller footprint at f32-like speed.
pub var cdt: mx.Dtype = mx.f32_;
/// TMT_COMPILE=0 disables graph compilation (debugging).
pub var compile_enabled: bool = true;
pub fn initCompute() void {
    mx.init();
    if (std.c.getenv("TMT_COMPILE")) |v| compile_enabled = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("TMT_COMPUTE")) |v| {
        const name = std.mem.span(v);
        cdt = if (std.mem.eql(u8, name, "f32")) mx.f32_ else if (std.mem.eql(u8, name, "f16")) mx.f16_ else if (std.mem.eql(u8, name, "bf16")) mx.bf16 else u.die("TMT_COMPUTE must be bf16, f16 or f32", .{});
    }
}

pub const Par = struct {
    master: Array,
    m: Array,
    v: Array,
    shape: [2]i32,
    ndim: usize,
    n: usize,

    pub fn shp(self: *const Par) []const i32 {
        return self.shape[0..self.ndim];
    }
};
pub const MlaP = struct { q: usize, dkv: usize, kr: usize, uk: usize, uv: usize, o: usize, gamma: usize, beta: usize };
pub const Layer = struct {
    decay: usize,
    gate: usize,
    gamma: usize,
    beta: usize,
    router: usize,
    exp: [16]usize = @splat(0),
    mla: ?MlaP = null,
};

pub const Model = struct {
    c: Cfg,
    params: std.ArrayList(Par) = .empty,
    emb: usize = 0,
    tgt: usize = 0,
    dec: usize = 0,
    stop: usize = 0,
    layers: []Layer = &.{},
    recur: cell.Recurrence = undefined,
    /// Gradients of the last forwardWindow(.grad = true), one per parameter (f32).
    grads: []Array = &.{},
    /// Outputs of the last window: logits (N,256), per-position CE (N), stop logits (N).
    logits: Array = mx.none,
    losses: Array = mx.none,
    stoplog: Array = mx.none,
    /// Tokens routed to each expert in the last window, per layer (E > 1).
    counts: [][16]i64 = &.{},
    /// Trace diagnostics: when log_traces is set, the trace part of the decay,
    /// gate and embedding gradients of the last window (host copies).
    log_traces: bool = false,
    trlog: ?struct { dec: [][]f32, gate: [][]f32, emb: []f32 } = null,
    /// Compiled AdamW step over all updated parameters (built on first use).
    adam: ?struct { f: mx.Compiled, updated: []usize } = null,
    /// Compiled window function. Only used when the forward reads no state
    /// outside its inputs (no MLA cache, no traces), because compilation would
    /// otherwise freeze those captured arrays into the plan.
    compiled: ?mx.Compiled = null,
    compiled_grad: ?mx.CompiledGrad = null,

    pub fn nparams(self: *const Model) usize {
        return self.params.items.len;
    }
    pub fn at(self: *Model, i: usize) *Par {
        return &self.params.items[i];
    }
    pub fn deinit(self: *Model) void {
        for (self.params.items) |p| {
            mx.free(p.master);
            mx.free(p.m);
            mx.free(p.v);
        }
        for (self.grads) |g| mx.free(g);
        mx.free(self.logits);
        mx.free(self.losses);
        mx.free(self.stoplog);
        self.params.deinit(mx.gpa);
    }
};

pub const Cache = struct { lat: Array = mx.none, kr: Array = mx.none, head: i64 = 0, base0: i64 = 0 };

pub const State = struct {
    carry: []Array,
    cache: []Cache,
    position: i64 = 0,
    // Hybrid traces (traces=1): ds_(t0-1)/dθ carried across windows.
    tdec: []Array = &.{},
    tgate: []Array = &.{},
    temb: Array = mx.none,
    c: Cfg,

    pub fn init(m: *const Model) State {
        const c = m.c;
        const L: usize = @intCast(c.layers);
        var s = State{
            .carry = mx.gpa.alloc(Array, L) catch mx.oom(),
            .cache = mx.gpa.alloc(Cache, L) catch mx.oom(),
            .c = c,
        };
        for (s.carry) |*a| a.* = mx.none;
        for (s.cache) |*a| a.* = .{};
        if (c.traces == 1) {
            s.tdec = mx.gpa.alloc(Array, L) catch mx.oom();
            s.tgate = mx.gpa.alloc(Array, L) catch mx.oom();
            for (s.tdec, s.tgate) |*a, *b| {
                a.* = mx.none;
                b.* = mx.none;
            }
        }
        s.reset();
        return s;
    }
    /// Zero carries, caches, traces and position; weights stay unchanged.
    pub fn reset(s: *State) void {
        const m0 = mx.mark();
        defer mx.release(m0);
        const bd = [_]i32{ s.c.batch, s.c.dim };
        for (s.carry) |*a| mx.assign(a, mx.zeros(&bd, mx.f32_));
        for (s.cache) |*k| {
            mx.free(k.lat);
            mx.free(k.kr);
            k.* = .{};
        }
        for (s.tdec) |*a| mx.assign(a, mx.zeros(&bd, mx.f32_));
        for (s.tgate) |*a| mx.assign(a, mx.zeros(&bd, mx.f32_));
        if (s.c.traces == 1) mx.assign(&s.temb, mx.zeros(&.{ s.c.batch, 256, s.c.dim }, mx.f32_));
        s.position = 0;
    }
    pub fn deinit(s: *State) void {
        for (s.carry) |a| mx.free(a);
        for (s.cache) |k| {
            mx.free(k.lat);
            mx.free(k.kr);
        }
        for (s.tdec) |a| mx.free(a);
        for (s.tgate) |a| mx.free(a);
        mx.free(s.temb);
    }
};

pub fn lrAt(c: Cfg, step: u64) f32 {
    const s: f32 = @floatFromInt(step);
    if (step < c.warmup) return c.lr * (s + 1) / @as(f32, @floatFromInt(c.warmup));
    const p = @min(1.0, (s - @as(f32, @floatFromInt(c.warmup))) / @as(f32, @floatFromInt(c.decaysteps)));
    return c.lr * (c.minlr + 0.5 * (1 - c.minlr) * (1 + @cos(3.14159265 * p)));
}

// ---------- construction ----------

fn addParam(m: *Model, shape: []const i32, init_values: []const f32) usize {
    var n: usize = 1;
    for (shape) |s| n *= @intCast(s);
    std.debug.assert(n == init_values.len);
    const m0 = mx.mark();
    defer mx.release(m0);
    var p = Par{ .master = mx.keep(mx.fromSlice(f32, init_values, shape)), .m = undefined, .v = undefined, .shape = .{ 0, 0 }, .ndim = shape.len, .n = n };
    for (shape, 0..) |s, i| p.shape[i] = s;
    p.m = mx.keep(mx.zeros(shape, mx.f32_));
    p.v = mx.keep(mx.zeros(shape, mx.f32_));
    m.params.append(mx.gpa, p) catch mx.oom();
    return m.params.items.len - 1;
}

/// Builds and initializes a model. The initialization RNG, its consumption
/// order and the parameter order match the CUDA build (checkpoint layout).
pub fn build(c: Cfg) Model {
    var m = Model{ .c = c };
    const D = c.dim;
    const Du: usize = @intCast(D);
    const L: usize = @intCast(c.layers);
    const E: usize = @intCast(c.experts);
    var seed: u32 = if (c.seed != 0) @bitCast(c.seed) else 1234;
    var h: std.ArrayList(f32) = .empty;
    defer h.deinit(mx.gpa);
    const buf = struct {
        fn get(list: *std.ArrayList(f32), n: usize) []f32 {
            list.resize(mx.gpa, n) catch mx.oom();
            return list.items;
        }
    }.get;
    const uni = struct {
        fn f(list: *std.ArrayList(f32), n: usize, a: f32, b: f32, s: *u32) []f32 {
            list.resize(mx.gpa, n) catch mx.oom();
            u.hostUniform(list.items, a, b, s);
            return list.items;
        }
    }.f;
    const constant = struct {
        fn f(list: *std.ArrayList(f32), n: usize, v: f32) []f32 {
            list.resize(mx.gpa, n) catch mx.oom();
            @memset(list.items, v);
            return list.items;
        }
    }.f;

    m.emb = addParam(&m, &.{ 256, D }, uni(&h, 256 * Du, -0.05, 0.05, &seed));
    m.tgt = addParam(&m, &.{ 256, D }, h.items);
    const a = @sqrt(1.0 / @as(f32, @floatFromInt(D)));
    m.dec = addParam(&m, &.{ 256, D }, uni(&h, 256 * Du, -a, a, &seed));
    m.stop = addParam(&m, &.{ 1, D }, uni(&h, Du, -0.01, 0.01, &seed));
    m.layers = mx.gpa.alloc(Layer, L) catch mx.oom();
    for (m.layers) |*ly| {
        const dec = buf(&h, Du);
        for (dec, 0..) |*v, d| {
            const half = c.half_min * std.math.pow(f32, c.half_max / c.half_min, @as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(@max(1, D - 1))));
            const ad = @exp(-@log(@as(f32, 2)) / half);
            v.* = @log(ad / (1 - ad));
        }
        ly.* = .{ .decay = addParam(&m, &.{D}, dec), .gate = undefined, .gamma = undefined, .beta = undefined, .router = undefined };
        ly.gate = addParam(&m, &.{D}, uni(&h, Du, 0, 0, &seed));
        ly.gamma = addParam(&m, &.{D}, constant(&h, Du, 1));
        ly.beta = addParam(&m, &.{D}, constant(&h, Du, 0));
        const r = buf(&h, E * Du);
        u.hostNormal(r, 0.02, &seed);
        ly.router = addParam(&m, &.{ c.experts, D }, r);
        for (0..E) |e| ly.exp[e] = addParam(&m, &.{ D, D }, uni(&h, Du * Du, -a, a, &seed));
    }
    if (c.mla == 1) {
        const H = c.mla_heads;
        const dh = c.mla_dh;
        const Lr = c.mla_L;
        const R = c.mla_R;
        const n = struct {
            fn f(x: i32, y: i32) usize {
                return @intCast(x * y);
            }
        }.f;
        for (m.layers, 0..) |*ly, l| {
            if (@mod(@as(i32, @intCast(l)), c.mla_every) != 0) continue;
            const b = @sqrt(1.0 / @as(f32, @floatFromInt(Lr)));
            const b2 = @sqrt(1.0 / @as(f32, @floatFromInt(H * dh)));
            var p: MlaP = undefined;
            p.q = addParam(&m, &.{ H * (dh + R), D }, uni(&h, n(H * (dh + R), D), -a, a, &seed));
            p.dkv = addParam(&m, &.{ Lr, D }, uni(&h, n(Lr, D), -a, a, &seed));
            p.kr = addParam(&m, &.{ R, D }, uni(&h, n(R, D), -a, a, &seed));
            p.uk = addParam(&m, &.{ H * dh, Lr }, uni(&h, n(H * dh, Lr), -b, b, &seed));
            p.uv = addParam(&m, &.{ H * dh, Lr }, uni(&h, n(H * dh, Lr), -b, b, &seed));
            p.o = addParam(&m, &.{ D, H * dh }, uni(&h, n(D, H * dh), -b2, b2, &seed));
            p.gamma = addParam(&m, &.{D}, constant(&h, Du, 1));
            p.beta = addParam(&m, &.{D}, constant(&h, Du, 0));
            ly.mla = p;
        }
    }
    m.recur = cell.Recurrence.init(.{ .gated = c.gated == 1, .docsep = c.docsep });
    m.grads = mx.gpa.alloc(Array, m.nparams()) catch mx.oom();
    for (m.grads) |*g| g.* = mx.none;
    m.counts = mx.gpa.alloc([16]i64, L) catch mx.oom();
    for (m.counts) |*k| k.* = @splat(0);
    return m;
}

// ---------- forward ----------

const Out = struct {
    loss: Array,
    ce: Array,
    losses: Array,
    logits: Array,
    stoplog: Array,
    carry: []Array,
    lat: []Array,
    kr: []Array,
    counts: []Array,
    tdec: []Array,
    tgate: []Array,
    temb: Array,
};

fn wt(w: []const Array, i: usize) Array {
    return mx.astype(w[i], cdt);
}
fn linear(x: Array, w: Array) Array { // x (...,K) @ w(N,K)^T
    return mx.matmul(x, mx.transpose(w, &.{ 1, 0 }));
}
fn toF(a: Array) Array {
    return mx.astype(a, mx.f32_);
}

/// RoPE on adjacent pairs (2i, 2i+1) of the last axis R; cs/sn: (1,T,1,R/2,1).
fn rope(x: Array, cs: Array, sn: Array) Array {
    const s = mx.shape(x);
    const shp = [_]i32{ s[0], s[1], s[2], @divExact(s[3], 2), 2 };
    const v = mx.reshape(toF(x), &shp);
    const xe = mx.sliceAxis(v, -1, 0, 1);
    const xo = mx.sliceAxis(v, -1, 1, 2);
    const ye = mx.sub(mx.mul(xe, cs), mx.mul(xo, sn));
    const yo = mx.add(mx.mul(xo, cs), mx.mul(xe, sn));
    return mx.reshape(mx.concat(&.{ ye, yo }, -1), &.{ s[0], s[1], s[2], s[3] });
}

fn ropeTables(c: Cfg, position: i64, T: i32) [2]Array {
    const half: usize = @intCast(@divExact(c.mla_R, 2));
    const Tu: usize = @intCast(T);
    const cs = mx.gpa.alloc(f32, Tu * half) catch mx.oom();
    defer mx.gpa.free(cs);
    const sn = mx.gpa.alloc(f32, Tu * half) catch mx.oom();
    defer mx.gpa.free(sn);
    for (0..Tu) |t| for (0..half) |p| {
        const ang = @as(f32, @floatFromInt(position + @as(i64, @intCast(t)))) /
            std.math.pow(f32, c.mla_theta, @as(f32, @floatFromInt(2 * p)) / @as(f32, @floatFromInt(c.mla_R)));
        cs[t * half + p] = @cos(ang);
        sn[t * half + p] = @sin(ang);
    };
    const shp = [_]i32{ 1, T, 1, @intCast(half), 1 };
    return .{ mx.fromSlice(f32, cs, &shp), mx.fromSlice(f32, sn, &shp) };
}

// Embedding lookup with a deterministic backward: the gradient is a single
// one-hot matmul instead of a scatter-add, whose atomic order would otherwise
// make gradients differ between two identical runs.
fn embFwd(_: *anyopaque, in: []const Array) []Array {
    const out = mx.gpa.alloc(Array, 1) catch mx.oom();
    out[0] = mx.take(mx.astype(in[0], cdt), in[1], 0);
    return out;
}
fn embVjp(_: *anyopaque, p: []const Array, cot: []const Array, _: []const Array) []Array {
    const rows = mx.dim(p[0], 0);
    const onehot = mx.astype(mx.equal(mx.expandDims(p[1], -1), mx.arange(0, @floatFromInt(rows), mx.i32_)), mx.f32_);
    const r = mx.gpa.alloc(Array, 2) catch mx.oom();
    r[0] = mx.matmul(mx.swapaxes(onehot, 0, 1), mx.astype(cot[0], mx.f32_));
    r[1] = mx.zeros(mx.shape(p[1]), mx.dtype(p[1]));
    return r;
}
var emb_lookup: ?mx.CustomFn = null;
fn embedding(table: Array, ids: Array) Array {
    if (emb_lookup == null) emb_lookup = mx.CustomFn.init(embFwd, embVjp, @constCast(@ptrCast(&emb_lookup)));
    const out = emb_lookup.?.apply(&.{ table, ids });
    defer mx.gpa.free(out);
    return out[0];
}

/// Switch load-balancing loss E·Σ_e mean_p_e·frac_e (frac = routed share, constant)
/// and z-loss mean(logsumexp²) of the router logits (N,E).
pub fn routerRegularizers(logits: Array, counts: Array, K: i32) [2]Array {
    const N = mx.dim(logits, 0);
    const E = mx.dim(logits, 1);
    const probs = mx.softmax(logits, -1);
    const frac = mx.mulS(mx.stopGradient(counts), 1.0 / @as(f32, @floatFromInt(N * K)));
    const aux = mx.mulS(mx.sumAll(mx.mul(mx.mulS(mx.sum(probs, 0, false), 1.0 / @as(f32, @floatFromInt(N))), frac)), @floatFromInt(E));
    const z = mx.meanAll(mx.square(mx.logsumexp(logits, -1, false)));
    return .{ aux, z };
}

const MoeOut = struct { y: Array, aux: Array, z: Array, counts: Array };

/// Dense (E=1) or top-k MoE feedforward with SiLU; returns X + FFN(H).
fn moe(c: Cfg, ly: *const Layer, w: []const Array, H: Array, X: Array) MoeOut {
    const N = mx.dim(H, 0);
    const D = c.dim;
    const E = c.experts;
    const K = c.topk;
    if (E == 1) {
        const pre = linear(H, wt(w, ly.exp[0]));
        return .{ .y = mx.astype(mx.add(toF(X), mx.silu(toF(pre))), cdt), .aux = mx.none, .z = mx.none, .counts = mx.none };
    }
    // Router logits in f32 from the working-precision weights.
    const logits = linear(toF(H), toF(wt(w, ly.router)));
    const probs = mx.softmax(logits, -1);
    const idx = mx.sliceAxis(mx.argsort(mx.negative(mx.stopGradient(probs)), -1), 1, 0, K); // (N,K)
    const pk = mx.takeAlong(probs, idx, 1);
    const wts = mx.div(pk, mx.sum(pk, 1, true)); // renormalized top-k weights
    const onehot = mx.astype(mx.equal(mx.expandDims(idx, -1), mx.astype(mx.arange(0, @floatFromInt(E), mx.i32_), mx.u32_)), mx.f32_);
    const counts = mx.sum(mx.reshape(onehot, &.{ N * K, E }), 0, false);
    const reg = routerRegularizers(logits, counts, K);
    const aux = reg[0];
    const z = reg[1];
    // Token dispatch: every (token, slot) pair becomes one row, the rows are
    // permuted so that each expert owns a contiguous segment, and one gathered
    // matmul runs them. Sorted indices also make both backward paths
    // segment-based instead of atomic scatter-adds, so gradients (and resumed
    // runs) are reproducible.
    const flat = mx.reshape(idx, &.{N * K});
    const order = mx.argsort(flat, 0);
    const rows = mx.reshape(mx.broadcastTo(mx.expandDims(H, 1), &.{ N, K, D }), &.{ N * K, D });
    const xs = mx.reshape(mx.take(rows, order, 0), &.{ N * K, 1, D });
    var ws: [16]Array = undefined;
    for (0..@intCast(E)) |e| ws[e] = wt(w, ly.exp[e]);
    const wstack = mx.swapaxes(mx.stack(ws[0..@intCast(E)], 0), -1, -2); // (E, in, out)
    const pre = mx.gatherMm(xs, wstack, mx.take(flat, order, 0), true); // (N*K,1,D)
    const unsorted = mx.take(mx.reshape(pre, &.{ N * K, D }), mx.argsort(order, 0), 0);
    const act = mx.mul(mx.silu(toF(mx.reshape(unsorted, &.{ N, K, D }))), mx.expandDims(wts, -1));
    const y = mx.astype(mx.add(toF(X), mx.sum(act, 1, false)), cdt);
    return .{ .y = y, .aux = aux, .z = z, .counts = counts };
}

const MlaOut = struct { y: Array, lat: Array, kr: Array };

/// The MLA block alone on a (B,T,D) input, for tests; w holds all masters.
pub fn mlaApply(m: *const Model, p: MlaP, w: []const Array, Xq: Array, cache: *const Cache, position: i64) Array {
    const T = mx.dim(Xq, 1);
    const rt = ropeTables(m.c, position, T);
    const head: i32 = @intCast(cache.head);
    const lat = if (head > 0) cache.lat else mx.zeros(&.{ m.c.batch, 0, m.c.mla_L }, mx.bf16);
    const kr = if (head > 0) cache.kr else mx.zeros(&.{ m.c.batch, 0, m.c.mla_R }, mx.bf16);
    return mlaBlock(m.c, p, w, Xq, lat, kr, qposArray(head, T), rt).y;
}

/// Absolute positions of the window's queries inside the cache, (1,1,T,1).
fn qposArray(head: i32, T: i32) Array {
    return mx.reshape(mx.arange(@floatFromInt(head), @floatFromInt(head + T), mx.i32_), &.{ 1, 1, T, 1 });
}

fn ln(x: Array, gamma: Array, beta: Array) Array {
    return mx.astype(mx.layerNorm(toF(x), gamma, beta, 1e-5), cdt);
}

pub const Batch = struct { ids: Array, nxt: Array, end: Array };

/// Uploads one window of host data: ids/targets (B*T), targets < 0 are ignored.
pub fn makeBatch(c: Cfg, ids: []const i32, nxt: []const i32, end: []const i32) Batch {
    const shp = [_]i32{ c.batch, @divExact(@as(i32, @intCast(ids.len)), c.batch) };
    return .{
        .ids = mx.fromSlice(i32, ids, &shp),
        .nxt = mx.fromSlice(i32, nxt, &.{@intCast(nxt.len)}),
        .end = mx.fromSlice(i32, end, &.{@intCast(end.len)}),
    };
}

/// Everything the window reads besides weights, carries and the batch. These
/// are passed in (not captured) so that the whole function can be compiled:
/// a captured array would be frozen into the compiled plan.
const Extra = struct {
    lat: []const Array = &.{}, // per MLA layer: cache latents (B,head,L)
    kr: []const Array = &.{}, //                cache RoPE keys (B,head,R)
    qpos: Array = mx.none, //    (1,1,T,1) absolute query positions in the cache
    cs: Array = mx.none, //      RoPE cos/sin tables for this window
    sn: Array = mx.none,
    tdec: []const Array = &.{}, // per layer decay/gate traces (B,D)
    tgate: []const Array = &.{},
    temb: Array = mx.none, //    (B,256,D) embedding trace
};

fn forwardCore(m: *const Model, w: []const Array, carries: []const Array, batch: Batch, x: Extra, traces: bool) Out {
    const c = m.c;
    const B = c.batch;
    const T = mx.dim(batch.ids, 1);
    const D = c.dim;
    const N = B * T;
    const L: usize = @intCast(c.layers);
    const alloc = struct {
        fn f(n: usize) []Array {
            const s = mx.gpa.alloc(Array, n) catch mx.oom();
            for (s) |*a| a.* = mx.none;
            return s;
        }
    }.f;
    var out = Out{
        .loss = undefined, .ce = undefined, .losses = undefined, .logits = undefined, .stoplog = undefined,
        .carry = alloc(L), .lat = alloc(L), .kr = alloc(L), .counts = alloc(L),
        .tdec = alloc(if (traces) L else 0), .tgate = alloc(if (traces) L else 0), .temb = mx.none,
    };
    const ids_flat = mx.reshape(batch.ids, &.{N});
    var X = mx.reshape(embedding(w[m.emb], ids_flat), &.{ B, T, D });
    var aux_acc: ?Array = null;
    var z_acc: ?Array = null;
    for (m.layers, 0..) |*ly, l| {
        const x_in = X;
        const S = m.recur.apply(x_in, w[ly.decay], w[ly.gate], carries[l], batch.ids);
        out.carry[l] = mx.stopGradient(mx.reshape(mx.sliceAxis(S, 1, T - 1, T), &.{ B, D }));
        if (traces) {
            const sg = mx.stopGradient;
            const o = cell.Opt{ .gated = c.gated == 1, .docsep = c.docsep };
            const tr = cell.advanceTraces(sg(S), sg(x_in), sg(w[ly.decay]), sg(w[ly.gate]), sg(carries[l]), batch.ids, x.tdec[l], x.tgate[l], c.trace_decay, o);
            out.tdec[l] = tr.tdec;
            out.tgate[l] = tr.tgate;
            if (l == 0) out.temb = cell.advanceEmbTrace(sg(x_in), sg(S), sg(carries[l]), sg(w[ly.decay]), sg(w[ly.gate]), batch.ids, x.temb, tr.prod, c.trace_decay, o);
        }
        // Norm input is the working-precision copy of the state, as in the CUDA build.
        const H = ln(mx.astype(S, cdt), w[ly.gamma], w[ly.beta]);
        const r = moe(c, ly, w, mx.reshape(H, &.{ N, D }), mx.reshape(X, &.{ N, D }));
        X = mx.reshape(r.y, &.{ B, T, D });
        if (c.experts > 1) {
            aux_acc = if (aux_acc) |acc| mx.add(acc, r.aux) else r.aux;
            z_acc = if (z_acc) |acc| mx.add(acc, r.z) else r.z;
            out.counts[l] = r.counts;
        }
        if (ly.mla) |p| {
            const a = mlaBlock(c, p, w, ln(X, w[p.gamma], w[p.beta]), x.lat[l], x.kr[l], x.qpos, .{ x.cs, x.sn });
            X = mx.add(X, a.y);
            out.lat[l] = a.lat;
            out.kr[l] = a.kr;
        }
    }
    const Xf = mx.reshape(X, &.{ N, D });
    out.logits = linear(Xf, wt(w, m.dec));
    const lf = toF(out.logits);
    const valid = mx.greaterEqual(batch.nxt, mx.scalarInt(0));
    const tidx = mx.where(valid, batch.nxt, mx.scalarInt(0));
    const picked = mx.reshape(mx.takeAlong(lf, mx.expandDims(tidx, -1), 1), &.{N});
    out.losses = mx.where(valid, mx.sub(mx.logsumexp(lf, -1, false), picked), mx.scalar(0));
    out.ce = mx.mulS(mx.sumAll(out.losses), 1.0 / @as(f32, @floatFromInt(N)));
    var tot = mx.mulS(out.ce, c.ce);
    out.stoplog = mx.reshape(linear(Xf, wt(w, m.stop)), &.{N});
    if (c.stop > 0) { // BCE with pos_weight on raw logits, target: next byte is '\n'
        const z = toF(out.stoplog);
        const tpos = mx.greater(batch.end, mx.scalarInt(0));
        const sp = mx.add(mx.maximum(mx.negative(z), mx.scalar(0)), mx.log1p(mx.exp(mx.negative(mx.abs(z)))));
        const l = mx.where(tpos, mx.mulS(sp, c.stopposw), mx.add(z, sp));
        tot = mx.add(tot, mx.mulS(mx.meanAll(l), c.stop));
    }
    if (c.latent > 0) { // MSE against the EMA target embedding of the next byte
        const target = mx.where(mx.expandDims(valid, -1), mx.take(mx.stopGradient(wt(w, m.tgt)), tidx, 0), mx.scalar(0));
        const mse = mx.meanAll(mx.square(mx.sub(toF(Xf), toF(target))));
        tot = mx.add(tot, mx.mulS(mse, c.latent));
    }
    if (c.@"var" > 0) { // variance hinge against collapse
        const xf = toF(Xf);
        const mean = mx.meanAll(xf);
        const variance = mx.meanAll(mx.square(mx.sub(xf, mean)));
        const hinge = mx.maximum(mx.scalar(0), mx.sub(mx.scalar(1), mx.sqrt(mx.addS(variance, 1e-4))));
        tot = mx.add(tot, mx.mulS(hinge, c.@"var"));
    }
    if (aux_acc) |acc| {
        const reg = mx.add(mx.mulS(acc, c.aux), mx.mulS(z_acc.?, c.zloss));
        tot = mx.add(tot, mx.mulS(reg, 1.0 / @as(f32, @floatFromInt(c.layers))));
    }
    out.loss = tot;
    return out;
}

fn mlaBlock(c: Cfg, p: MlaP, w: []const Array, Xq: Array, cache_lat: Array, cache_kr: Array, qpos_in: Array, rt: [2]Array) MlaOut {
    const B = c.batch;
    const T = mx.dim(Xq, 1);
    const D = c.dim;
    const H = c.mla_heads;
    const dh = c.mla_dh;
    const R = c.mla_R;
    const Lr = c.mla_L;
    const N = B * T;
    const xq = mx.reshape(Xq, &.{ N, D });
    const q = mx.reshape(linear(xq, wt(w, p.q)), &.{ B, T, H, dh + R });
    const qc = toF(mx.transpose(mx.sliceAxis(q, -1, 0, dh), &.{ 0, 2, 1, 3 })); // (B,H,T,dh)
    const qr = mx.transpose(rope(mx.sliceAxis(q, -1, dh, dh + R), rt[0], rt[1]), &.{ 0, 2, 1, 3 }); // (B,H,T,R)
    // Window latents and rotated keys, rounded to the cache precision.
    const lat = mx.astype(mx.reshape(linear(xq, wt(w, p.dkv)), &.{ B, T, Lr }), mx.bf16);
    const kr = mx.astype(mx.reshape(rope(mx.reshape(linear(xq, wt(w, p.kr)), &.{ B, T, 1, R }), rt[0], rt[1]), &.{ B, T, R }), mx.bf16);
    const head: i32 = mx.dim(cache_lat, 1);
    const all_lat = if (head > 0) mx.concat(&.{ mx.stopGradient(cache_lat), lat }, 1) else lat;
    const all_kr = if (head > 0) mx.concat(&.{ mx.stopGradient(cache_kr), kr }, 1) else kr;
    const clen = head + T;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));
    // Chunked online softmax over the cache (chunk = mla_cc), accumulators in f32.
    var mrun = mx.full(&.{ B, H, T, 1 }, -1e30, mx.f32_);
    var lrun = mx.zeros(&.{ B, H, T, 1 }, mx.f32_);
    var o = mx.zeros(&.{ B, H, T, dh }, mx.f32_);
    const qpos = qpos_in; // (1,1,T,1): head + t, passed in so it is not baked into a compiled plan
    const uk = wt(w, p.uk);
    const uv = wt(w, p.uv);
    var c0: i32 = 0;
    while (c0 < clen) : (c0 += c.mla_cc) {
        const cce = @min(c.mla_cc, clen - c0);
        const clat = mx.astype(mx.sliceAxis(all_lat, 1, c0, c0 + cce), cdt);
        const kc = toF(mx.transpose(mx.reshape(linear(clat, uk), &.{ B, cce, H, dh }), &.{ 0, 2, 1, 3 }));
        const vc = toF(mx.transpose(mx.reshape(linear(clat, uv), &.{ B, cce, H, dh }), &.{ 0, 2, 1, 3 }));
        const ckr = toF(mx.expandDims(mx.sliceAxis(all_kr, 1, c0, c0 + cce), 1)); // (B,1,cce,R)
        var s = mx.add(mx.matmul(qc, mx.swapaxes(kc, -1, -2)), mx.matmul(qr, mx.swapaxes(ckr, -1, -2)));
        s = mx.mulS(s, scale);
        const kpos = mx.reshape(mx.arange(@floatFromInt(c0), @floatFromInt(c0 + cce), mx.i32_), &.{ 1, 1, 1, cce });
        s = mx.where(mx.greater(kpos, qpos), mx.scalar(-1e30), s);
        const mnew = mx.stopGradient(mx.maximum(mrun, mx.max(s, -1, true)));
        const alpha = mx.exp(mx.sub(mrun, mnew));
        const pr = mx.exp(mx.sub(s, mnew));
        lrun = mx.add(mx.mul(lrun, alpha), mx.sum(pr, -1, true));
        o = mx.add(mx.mul(o, alpha), mx.matmul(pr, vc));
        mrun = mnew;
    }
    o = mx.div(o, lrun);
    const oflat = mx.astype(mx.reshape(mx.transpose(o, &.{ 0, 2, 1, 3 }), &.{ N, H * dh }), cdt);
    const y = mx.reshape(linear(oflat, wt(w, p.o)), &.{ B, T, D });
    return .{ .y = y, .lat = mx.stopGradient(lat), .kr = mx.stopGradient(kr) };
}

// ---------- packing for value_and_grad ----------

fn freeOut(o: Out) void {
    for ([_][]Array{ o.carry, o.lat, o.kr, o.counts, o.tdec, o.tgate }) |s| mx.gpa.free(s);
}

fn pack(m: *const Model, o: Out, traces: bool) []Array {
    var list: std.ArrayList(Array) = .empty;
    list.appendSlice(mx.gpa, &.{ o.loss, o.ce, o.losses, o.logits, o.stoplog }) catch mx.oom();
    list.appendSlice(mx.gpa, o.carry) catch mx.oom();
    for (m.layers, 0..) |ly, l| if (ly.mla != null) list.appendSlice(mx.gpa, &.{ o.lat[l], o.kr[l] }) catch mx.oom();
    if (m.c.experts > 1) list.appendSlice(mx.gpa, o.counts) catch mx.oom();
    if (traces) {
        list.appendSlice(mx.gpa, o.tdec) catch mx.oom();
        list.appendSlice(mx.gpa, o.tgate) catch mx.oom();
        list.append(mx.gpa, o.temb) catch mx.oom();
    }
    return list.toOwnedSlice(mx.gpa) catch mx.oom();
}
fn unpack(m: *const Model, v: []Array, traces: bool) Out {
    const L = m.layers.len;
    const alloc = struct {
        fn f(n: usize) []Array {
            const s = mx.gpa.alloc(Array, n) catch mx.oom();
            for (s) |*a| a.* = mx.none;
            return s;
        }
    }.f;
    var o = Out{ .loss = v[0], .ce = v[1], .losses = v[2], .logits = v[3], .stoplog = v[4], .carry = alloc(L), .lat = alloc(L), .kr = alloc(L), .counts = alloc(L), .tdec = alloc(0), .tgate = alloc(0), .temb = mx.none };
    var i: usize = 5;
    for (0..L) |l| {
        o.carry[l] = v[i];
        i += 1;
    }
    for (m.layers, 0..) |ly, l| if (ly.mla != null) {
        o.lat[l] = v[i];
        o.kr[l] = v[i + 1];
        i += 2;
    };
    if (m.c.experts > 1) for (0..L) |l| {
        o.counts[l] = v[i];
        i += 1;
    };
    if (traces) {
        o.tdec = v[i .. i + L];
        o.tgate = v[i + L .. i + 2 * L];
        o.temb = v[i + 2 * L];
    }
    return o;
}

const Ctx = struct { m: *const Model, np: usize, traces: bool, mla_layers: usize };

fn lossFn(ctx: *anyopaque, in: []const Array) []Array {
    const k: *Ctx = @ptrCast(@alignCast(ctx));
    const m = k.m;
    const L = m.layers.len;
    const b = Batch{ .ids = in[k.np + L], .nxt = in[k.np + L + 1], .end = in[k.np + L + 2] };
    var i = k.np + L + 3;
    var x = Extra{};
    if (k.mla_layers > 0) {
        const lat = mx.gpa.alloc(Array, L) catch mx.oom();
        const kr = mx.gpa.alloc(Array, L) catch mx.oom();
        for (m.layers, 0..) |ly, l| if (ly.mla != null) {
            lat[l] = in[i];
            kr[l] = in[i + 1];
            i += 2;
        };
        x.lat = lat;
        x.kr = kr;
        x.qpos = in[i];
        x.cs = in[i + 1];
        x.sn = in[i + 2];
        i += 3;
    }
    if (k.traces) {
        x.tdec = in[i .. i + L];
        x.tgate = in[i + L .. i + 2 * L];
        x.temb = in[i + 2 * L];
    }
    defer if (k.mla_layers > 0) {
        mx.gpa.free(x.lat);
        mx.gpa.free(x.kr);
    };
    const o = forwardCore(m, in[0..k.np], in[k.np .. k.np + L], b, x, k.traces);
    defer freeOut(o);
    return pack(m, o, k.traces);
}

// ---------- window step ----------

pub const Opts = struct { grad: bool = false };
pub const Result = struct { loss: f32, ce: f32 };

/// Runs one window on `st` (carry, caches and position advance). With
/// .grad = true, the parameter gradients land in m.grads and, with traces=1,
/// the trace contributions are added and the traces advance.
pub fn forwardWindow(m: *Model, st: *State, batch: Batch, opts: Opts) Result {
    const c = m.c;
    const T = mx.dim(batch.ids, 1);
    const L = m.layers.len;
    const traces = opts.grad and c.traces == 1;
    const m0 = mx.mark();
    defer mx.release(m0);
    // Evict only the oldest prefix when the window would exceed the cache.
    for (m.layers, st.cache) |ly, *k| if (ly.mla != null and k.head + T > c.mla_cache) {
        const drop: i32 = @intCast(k.head + T - c.mla_cache);
        const keep: i32 = @intCast(k.head - drop);
        if (keep > 0) {
            mx.assign(&k.lat, mx.sliceAxis(k.lat, 1, drop, drop + keep));
            mx.assign(&k.kr, mx.sliceAxis(k.kr, 1, drop, drop + keep));
        } else {
            mx.free(k.lat);
            mx.free(k.kr);
            k.lat = mx.none;
            k.kr = mx.none;
        }
        k.base0 += drop;
        k.head = keep;
    };
    const np = m.nparams();
    var inputs: std.ArrayList(Array) = .empty;
    defer inputs.deinit(mx.gpa);
    for (m.params.items) |p| inputs.append(mx.gpa, p.master) catch mx.oom();
    inputs.appendSlice(mx.gpa, st.carry) catch mx.oom();
    inputs.appendSlice(mx.gpa, &.{ batch.ids, batch.nxt, batch.end }) catch mx.oom();
    var mla_layers: usize = 0;
    for (m.layers) |ly| mla_layers += @intFromBool(ly.mla != null);
    if (mla_layers > 0) {
        var head: i32 = 0;
        for (m.layers, st.cache) |ly, k| if (ly.mla != null) {
            head = @intCast(k.head);
            const lat = if (k.head > 0) k.lat else mx.zeros(&.{ c.batch, 0, c.mla_L }, mx.bf16);
            const kr = if (k.head > 0) k.kr else mx.zeros(&.{ c.batch, 0, c.mla_R }, mx.bf16);
            inputs.appendSlice(mx.gpa, &.{ lat, kr }) catch mx.oom();
        };
        const rt = ropeTables(c, st.position, T);
        inputs.appendSlice(mx.gpa, &.{ qposArray(head, T), rt[0], rt[1] }) catch mx.oom();
    }
    if (traces) {
        inputs.appendSlice(mx.gpa, st.tdec) catch mx.oom();
        inputs.appendSlice(mx.gpa, st.tgate) catch mx.oom();
        inputs.append(mx.gpa, st.temb) catch mx.oom();
    }
    var ctx = Ctx{ .m = m, .np = np, .traces = traces, .mla_layers = mla_layers };
    // Everything the graph reads is an input now, so any configuration can be
    // compiled. MLX keys its plan on the input shapes, so an MLA cache that is
    // still filling would compile once per size; compile only once it is full.
    var stable = true;
    for (m.layers, st.cache) |ly, k| if (ly.mla != null and k.head + T != c.mla_cache) {
        stable = false;
    };
    const compilable = compile_enabled and stable;
    var argnums: std.ArrayList(i32) = .empty;
    defer argnums.deinit(mx.gpa);
    for (0..np) |i| if (i != m.tgt) argnums.append(mx.gpa, @intCast(i)) catch mx.oom();
    if (traces) for (0..L) |l| argnums.append(mx.gpa, @intCast(np + l)) catch mx.oom();
    // The two plans differ (the gradient variant also advances the traces and
    // returns their new values), so each is compiled from its own call.
    if (compilable) {
        const kept = mx.gpa.create(Ctx) catch mx.oom();
        kept.* = ctx;
        if (opts.grad and m.compiled_grad == null)
            m.compiled_grad = mx.CompiledGrad.init(lossFn, kept, argnums.items)
        else if (!opts.grad and m.compiled == null)
            m.compiled = mx.Compiled.init(lossFn, kept)
        else
            mx.gpa.destroy(kept);
    }
    const use_compiled = compilable and (if (opts.grad) m.compiled_grad != null else m.compiled != null);
    var values: []Array = undefined;
    var grads: []Array = &.{};
    var owned: []Array = &.{};
    if (opts.grad) {
        const r = if (use_compiled) m.compiled_grad.?.apply(inputs.items) else mx.valueAndGrad(lossFn, &ctx, inputs.items, argnums.items);
        values = r.values;
        grads = r.grads;
        if (use_compiled) owned = values.ptr[0 .. values.len + grads.len]; // one allocation
    } else {
        values = if (use_compiled) m.compiled.?.apply(inputs.items) else lossFn(&ctx, inputs.items);
        owned = values;
    }
    defer if (owned.len > 0) mx.gpa.free(owned) else {
        mx.gpa.free(values);
        mx.gpa.free(grads);
    };
    const o = unpack(m, values, traces);
    defer for ([_][]Array{ o.carry, o.lat, o.kr, o.counts }) |s| mx.gpa.free(s);

    if (opts.grad) {
        var gi: usize = 0;
        for (0..np) |i| {
            if (i == m.tgt) {
                mx.assign(&m.grads[i], mx.zeros(m.at(i).shp(), mx.f32_));
                continue;
            }
            mx.assign(&m.grads[i], grads[gi]);
            gi += 1;
        }
        if (traces) addTraceGradients(m, st, grads[gi .. gi + L]);
    }
    // Stream state: carry-out, cache append, position.
    for (st.carry, o.carry) |*a, b| mx.assign(a, b);
    for (m.layers, st.cache, 0..) |ly, *k, l| if (ly.mla != null) {
        if (k.head > 0) {
            mx.assign(&k.lat, mx.concat(&.{ k.lat, o.lat[l] }, 1));
            mx.assign(&k.kr, mx.concat(&.{ k.kr, o.kr[l] }, 1));
        } else {
            mx.assign(&k.lat, o.lat[l]);
            mx.assign(&k.kr, o.kr[l]);
        }
        k.head += T;
    };
    if (traces) {
        for (st.tdec, o.tdec) |*a, b| mx.assign(a, b);
        for (st.tgate, o.tgate) |*a, b| mx.assign(a, b);
        mx.assign(&st.temb, o.temb);
    }
    st.position += T;
    mx.assign(&m.logits, o.logits);
    mx.assign(&m.losses, o.losses);
    mx.assign(&m.stoplog, o.stoplog);
    // One evaluation for the whole window.
    var ev: std.ArrayList(Array) = .empty;
    defer ev.deinit(mx.gpa);
    ev.appendSlice(mx.gpa, &.{ o.loss, o.ce, m.logits, m.losses, m.stoplog }) catch mx.oom();
    ev.appendSlice(mx.gpa, st.carry) catch mx.oom();
    for (st.cache) |k| if (k.head > 0) ev.appendSlice(mx.gpa, &.{ k.lat, k.kr }) catch mx.oom();
    if (opts.grad) ev.appendSlice(mx.gpa, m.grads) catch mx.oom();
    if (traces) {
        ev.appendSlice(mx.gpa, st.tdec) catch mx.oom();
        ev.appendSlice(mx.gpa, st.tgate) catch mx.oom();
        ev.append(mx.gpa, st.temb) catch mx.oom();
    }
    for (o.counts) |a| if (a.ctx != null) ev.append(mx.gpa, a) catch mx.oom();
    mx.eval(ev.items);
    if (c.experts > 1) for (o.counts, 0..) |a, l| {
        var cnt: [16]f32 = undefined;
        mx.toF32(a, cnt[0..@intCast(c.experts)]);
        for (0..@intCast(c.experts)) |e| m.counts[l][e] = @intFromFloat(cnt[e]);
    };
    return .{ .loss = mx.item(o.loss), .ce = mx.item(o.ce) };
}

/// λ·e: the credit of all bytes before the window for decay, gate and embedding.
fn addTraceGradients(m: *Model, st: *const State, lam: []const Array) void {
    const c = m.c;
    const log = m.log_traces;
    if (log) {
        const L = m.layers.len;
        var t = @TypeOf(m.trlog.?){ .dec = mx.gpa.alloc([]f32, L) catch mx.oom(), .gate = mx.gpa.alloc([]f32, L) catch mx.oom(), .emb = undefined };
        for (m.layers, 0..) |ly, l| {
            const pd = mx.sum(mx.mul(lam[l], st.tdec[l]), 0, false);
            t.dec[l] = mx.toVecF32(pd);
            mx.assign(&m.grads[ly.decay], mx.add(m.grads[ly.decay], pd));
            const pg = if (c.gated == 1) mx.sum(mx.mul(lam[l], st.tgate[l]), 0, false) else mx.zeros(&.{c.dim}, mx.f32_);
            t.gate[l] = mx.toVecF32(pg);
            mx.assign(&m.grads[ly.gate], mx.add(m.grads[ly.gate], pg));
        }
        const pe = mx.sum(mx.mul(mx.expandDims(lam[0], 1), st.temb), 0, false);
        t.emb = mx.toVecF32(pe);
        mx.assign(&m.grads[m.emb], mx.add(m.grads[m.emb], pe));
        m.trlog = t;
        return;
    }
    for (m.layers, 0..) |ly, l| {
        mx.assign(&m.grads[ly.decay], mx.add(m.grads[ly.decay], mx.sum(mx.mul(lam[l], st.tdec[l]), 0, false)));
        if (c.gated == 1) mx.assign(&m.grads[ly.gate], mx.add(m.grads[ly.gate], mx.sum(mx.mul(lam[l], st.tgate[l]), 0, false)));
    }
    mx.assign(&m.grads[m.emb], mx.add(m.grads[m.emb], mx.sum(mx.mul(mx.expandDims(lam[0], 1), st.temb), 0, false)));
}

// ---------- optimizer ----------

pub const StepError = error{NonFiniteGradient};

/// Global-norm clipping, AdamW on every parameter except the EMA target (and an
/// unused stop head), then the EMA target update. A non-finite gradient norm
/// refuses the whole step and leaves every weight unchanged.
pub fn optimizerStep(m: *Model, step: u64) StepError!void {
    const c = m.c;
    const m0 = mx.mark();
    defer mx.release(m0);
    var sq: ?Array = null;
    for (m.grads, 0..) |g, i| {
        if (i == m.tgt) continue;
        const s = mx.sumAll(mx.square(g));
        sq = if (sq) |a| mx.add(a, s) else s;
    }
    const sumsq: f64 = mx.item(sq.?);
    if (!std.math.isFinite(sumsq)) return error.NonFiniteGradient;
    const scale: f32 = if (c.gradclip > 0) @floatCast(@min(1.0, c.gradclip / @max(@sqrt(sumsq), 1e-12))) else 1;
    const lr = lrAt(c, step);
    const bc1 = 1 - std.math.pow(f32, 0.9, @floatFromInt(step + 1));
    const bc2 = 1 - std.math.pow(f32, 0.999, @floatFromInt(step + 1));
    if (m.adam == null) buildAdam(m);
    // One compiled call for every parameter: MLX fuses the elementwise chains,
    // so the step costs a few kernels per tensor instead of a dozen launches.
    var in: std.ArrayList(Array) = .empty;
    defer in.deinit(mx.gpa);
    for ([_]f32{ scale, lr, bc1, bc2, c.ematau }) |v| in.append(mx.gpa, mx.fromSlice(f32, &.{v}, &.{1})) catch mx.oom();
    for (m.adam.?.updated) |i| {
        const p = m.at(i);
        in.appendSlice(mx.gpa, &.{ p.master, p.m, p.v, m.grads[i] }) catch mx.oom();
    }
    in.append(mx.gpa, m.at(m.tgt).master) catch mx.oom();
    const out = m.adam.?.f.apply(in.items);
    defer mx.gpa.free(out);
    for (m.adam.?.updated, 0..) |i, t| {
        const p = m.at(i);
        mx.assign(&p.master, out[3 * t]);
        mx.assign(&p.m, out[3 * t + 1]);
        mx.assign(&p.v, out[3 * t + 2]);
    }
    mx.assign(&m.at(m.tgt).master, out[out.len - 1]);
    var ev: std.ArrayList(Array) = .empty;
    defer ev.deinit(mx.gpa);
    for (m.adam.?.updated) |i| {
        const p = m.at(i);
        ev.appendSlice(mx.gpa, &.{ p.master, p.m, p.v }) catch mx.oom();
    }
    ev.append(mx.gpa, m.at(m.tgt).master) catch mx.oom();
    mx.eval(ev.items);
}

/// AdamW over all updated parameters plus the EMA target, as one compiled
/// function. Inputs: scale, lr, bc1, bc2, ematau, then (master, m, v, grad)
/// per parameter, then the EMA target. Outputs: (master, m, v) per parameter
/// and the new EMA target.
fn adamFn(ctx: *anyopaque, in: []const Array) []Array {
    const k: *AdamCtx = @ptrCast(@alignCast(ctx));
    const scale = in[0];
    const lr = in[1];
    const bc1 = in[2];
    const bc2 = in[3];
    const tau = in[4];
    const n = k.count;
    const out = mx.gpa.alloc(Array, 3 * n + 1) catch mx.oom();
    var emb_new: Array = mx.none;
    for (0..n) |t| {
        const master = in[5 + 4 * t];
        const mm = in[6 + 4 * t];
        const vv = in[7 + 4 * t];
        const g = mx.mul(in[8 + 4 * t], scale);
        const nm = mx.add(mx.mulS(mm, 0.9), mx.mulS(g, 0.1));
        const nv = mx.add(mx.mulS(vv, 0.999), mx.mulS(mx.square(g), 0.001));
        const upd = mx.add(mx.div(mx.div(nm, bc1), mx.addS(mx.sqrt(mx.div(nv, bc2)), 1e-8)), mx.mulS(master, 0.01));
        out[3 * t] = mx.sub(master, mx.mul(upd, lr));
        out[3 * t + 1] = nm;
        out[3 * t + 2] = nv;
        if (t == k.emb_slot) emb_new = out[3 * t];
    }
    // EMA target encoder: updated from the new embedding, never by AdamW.
    const tgt = in[5 + 4 * n];
    out[3 * n] = mx.add(mx.mul(tgt, tau), mx.mul(emb_new, mx.sub(mx.scalar(1), tau)));
    return out;
}

const AdamCtx = struct { count: usize, emb_slot: usize };

fn buildAdam(m: *Model) void {
    var updated: std.ArrayList(usize) = .empty;
    var emb_slot: usize = 0;
    for (0..m.nparams()) |i| {
        if (i == m.tgt or (i == m.stop and m.c.stop == 0)) continue;
        if (i == m.emb) emb_slot = updated.items.len;
        updated.append(mx.gpa, i) catch mx.oom();
    }
    const ctx = mx.gpa.create(AdamCtx) catch mx.oom();
    ctx.* = .{ .count = updated.items.len, .emb_slot = emb_slot };
    m.adam = .{ .f = mx.Compiled.init(adamFn, ctx), .updated = updated.toOwnedSlice(mx.gpa) catch mx.oom() };
}

// ---------- host helpers ----------

pub fn setMaster(m: *Model, i: usize, values: []const f32) void {
    const m0 = mx.mark();
    defer mx.release(m0);
    const p = m.at(i);
    std.debug.assert(values.len == p.n);
    mx.assign(&p.master, mx.fromSlice(f32, values, p.shp()));
}
pub fn getMaster(m: *Model, i: usize) []f32 {
    return mx.toVecF32(m.at(i).master);
}
pub fn getGrad(m: *Model, i: usize) []f32 {
    return mx.toVecF32(m.grads[i]);
}
