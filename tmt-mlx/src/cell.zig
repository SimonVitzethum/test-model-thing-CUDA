// Gated hierarchical recurrence as Metal kernels with an exact custom VJP.
//
//   a_t = sigmoid(decay + gate * x_t)       (a_t = 0 where ids == docsep)
//   s_t = a_t * s_(t-1) + (1 - a_t) * x_t   (s_(-1) = incoming carry)
//
// The CUDA build ran one thread per (stream, channel) across the whole window.
// On Apple Silicon that starves the GPU: a window has only B·D chains, and the
// kernel reaches 10 GB/s at 4k threads versus 75 GB/s at 256k (measured with
// `bench`). Since a_t depends only on x_t, the scan is linear with known
// coefficients and can be split over time:
//
//   phase A  each thread scans one time chunk from a zero state and keeps
//            (P = Π a_t, L = local end state),
//   phase B  one thread per channel walks the chunks and turns those into the
//            true chunk-entry states (s_c = L + P · s_(c-1)),
//   phase C  each thread re-scans its chunk from that state and writes out.
//
// All chunks of one channel live in the same threadgroup, so phase B is a pass
// over threadgroup memory between two barriers instead of a second launch.
// CHUNKS = 1 degenerates to the original one-thread-per-channel kernel.
const std = @import("std");
const mx = @import("mlx.zig");
const Array = mx.Array;

const header =
    \\inline float cell_a(float decay, float g, float x, int id, int docsep1) {
    \\  if (docsep1 > 0 && id == docsep1 - 1) return 0.0f;
    \\  return 1.0f / (1.0f + metal::precise::exp(-(decay + g * x)));
    \\}
    \\// Per-thread indices, chunk bounds and the shared scratch, for every kernel.
    \\#define TMT_SETUP(TPOS, GPOS) \
    \\  const int lanes = LANES, chunks = CHUNKS; \
    \\  const int tid = int(TPOS); \
    \\  const int c = tid / lanes, lane = tid % lanes; \
    \\  const int gid = int(GPOS); \
    \\  const int tiles = (D + lanes - 1) / lanes; \
    \\  const int b = gid / tiles, d = (gid % tiles) * lanes + lane; \
    \\  const int cl = (T + chunks - 1) / chunks; \
    \\  const int t0 = c * cl, t1 = min(T, t0 + cl); \
    \\  const bool active = (d < D) && (t0 < T); \
    \\  threadgroup float sP[LANES * CHUNKS]; \
    \\  threadgroup float sL[LANES * CHUNKS]; \
    \\  threadgroup float sPre[LANES * CHUNKS];
    \\
;

const fwd_src =
    \\TMT_SETUP(thread_position_in_threadgroup.x, threadgroup_position_in_grid.x);
    \\float dec = 0.0f, g = 0.0f;
    \\if (d < D) { dec = decay[d]; g = GATED ? gate[d] : 0.0f; }
    \\// A + B are only needed when the window is split (CHUNKS is a compile-time
    \\// constant, so a single chunk compiles down to the plain scan below).
    \\if (chunks > 1) {
    \\  float P = 1.0f, L = 0.0f;
    \\  if (active) {
    \\    for (int t = t0; t < t1; ++t) {
    \\      float xv = float(x[(b * T + t) * D + d]);
    \\      float a = cell_a(dec, g, xv, ids[b * T + t], DOCSEP1);
    \\      L = a * L + (1.0f - a) * xv;
    \\      P *= a;
    \\    }
    \\  }
    \\  sP[tid] = P; sL[tid] = L;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (c == 0 && d < D) {
    \\    float s = carry[b * D + d];
    \\    for (int k = 0; k < chunks; ++k) {
    \\      int j = k * lanes + lane;
    \\      sPre[j] = s;
    \\      s = sL[j] + sP[j] * s;
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\// C: scan from the true entry state and write the states.
    \\if (active) {
    \\  float st = (chunks > 1) ? sPre[tid] : carry[b * D + d];
    \\  for (int t = t0; t < t1; ++t) {
    \\    int idx = (b * T + t) * D + d;
    \\    float xv = float(x[idx]);
    \\    float a = cell_a(dec, g, xv, ids[b * T + t], DOCSEP1);
    \\    st = a * st + (1.0f - a) * xv;
    \\    S[idx] = st;
    \\  }
    \\}
;

// Backward of the same scan, in reverse: future_(t-1) = a_t · (dS_t + future_t).
// Phase C also produces the per-chunk decay/gate sums (reduced by the caller)
// and, from chunk 0, dL/d(carry) = λ, which the hybrid traces need.
const bwd_src =
    \\TMT_SETUP(thread_position_in_threadgroup.x, threadgroup_position_in_grid.x);
    \\float dec = 0.0f, g = 0.0f;
    \\if (d < D) { dec = decay[d]; g = GATED ? gate[d] : 0.0f; }
    \\if (chunks > 1) {
    \\  float P = 1.0f, L = 0.0f;
    \\  if (active) {
    \\    for (int t = t1 - 1; t >= t0; --t) {
    \\      int idx = (b * T + t) * D + d;
    \\      float xv = float(x[idx]);
    \\      float a = cell_a(dec, g, xv, ids[b * T + t], DOCSEP1);
    \\      L = a * (dS[idx] + L);
    \\      P *= a;
    \\    }
    \\  }
    \\  sP[tid] = P; sL[tid] = L;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  // Incoming future of each chunk, accumulated from the last chunk backwards.
    \\  if (c == 0 && d < D) {
    \\    float f = 0.0f;
    \\    for (int k = chunks - 1; k >= 0; --k) {
    \\      int j = k * lanes + lane;
    \\      sPre[j] = f;
    \\      f = sL[j] + sP[j] * f;
    \\    }
    \\    dcarry[b * D + d] = f;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\float gd = 0.0f, gg = 0.0f;
    \\if (active) {
    \\  float future = (chunks > 1) ? sPre[tid] : 0.0f;
    \\  for (int t = t1 - 1; t >= t0; --t) {
    \\    int idx = (b * T + t) * D + d;
    \\    float xv = float(x[idx]);
    \\    float a = cell_a(dec, g, xv, ids[b * T + t], DOCSEP1);
    \\    float total = dS[idx] + future;
    \\    float prev = t ? S[idx - D] : carry[b * D + d];
    \\    float local = (prev - xv) * a * (1.0f - a);
    \\    float gz = total * local;
    \\    dx[idx] = total * (1.0f - a) + gz * g;
    \\    gd += gz; gg += gz * xv;
    \\    future = total * a;
    \\  }
    \\  if (chunks == 1) dcarry[b * D + d] = future;
    \\}
    \\if (d < D) {
    \\  gdec[(b * chunks + c) * D + d] = gd;
    \\  ggate[(b * chunks + c) * D + d] = gg;
    \\}
;

// Closed-form trace advance, e_t = γ·a_t·e_(t-1) + ds_t/dθ (see README).
// Within a chunk this is a suffix sum; phase B weights each chunk with the
// product of all later chunks, so no third pass is needed.
const trace_src =
    \\TMT_SETUP(thread_position_in_threadgroup.x, threadgroup_position_in_grid.x);
    \\float gm = gamma[0];
    \\float dec = 0.0f, g = 0.0f;
    \\if (d < D) { dec = decay[d]; g = GATED ? gate[d] : 0.0f; }
    \\float P = 1.0f, ld = 0.0f, lg = 0.0f;
    \\if (active) {
    \\  for (int t = t1 - 1; t >= t0; --t) {
    \\    int idx = (b * T + t) * D + d;
    \\    float xv = float(x[idx]);
    \\    float a = cell_a(dec, g, xv, ids[b * T + t], DOCSEP1);
    \\    float prev = t ? S[idx - D] : carry[b * D + d];
    \\    float local = (prev - xv) * a * (1.0f - a);
    \\    ld += P * local; lg += P * local * xv;
    \\    P *= gm * a;
    \\  }
    \\}
    \\sP[tid] = P; sL[tid] = ld; sPre[tid] = lg;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (c == 0 && d < D) {
    \\  float after = 1.0f, sum_d = 0.0f, sum_g = 0.0f;
    \\  for (int k = chunks - 1; k >= 0; --k) {
    \\    int j = k * lanes + lane;
    \\    sum_d += after * sL[j];
    \\    sum_g += after * sPre[j];
    \\    after *= sP[j];
    \\  }
    \\  int bd = b * D + d;
    \\  tdec_out[bd] = after * tdec[bd] + sum_d;
    \\  tgate_out[bd] = GATED ? after * tgate[bd] + sum_g : tgate[bd];
    \\  prod[bd] = after;
    \\}
;

// Embedding trace e[b][r][d] = ds_d/dEmb[r][d] for layer 0 (input = embedding).
// One thread per channel: it walks all 256 rows, and splitting the time loop
// would make several chunks scatter into the same row.
const emb_src =
    \\uint i = thread_position_in_grid.x;
    \\if (i >= uint(B * D)) return;
    \\int b = i / D, d = i % D;
    \\float dec = decay[d], g = GATED ? gate[d] : 0.0f, gm = gamma[0];
    \\float P = prod[b * D + d];
    \\for (int r = 0; r < 256; ++r) {
    \\  int e = (b * 256 + r) * D + d;
    \\  temb_out[e] = P * temb[e];
    \\}
    \\float suffix = 1.0f;
    \\for (int t = T - 1; t >= 0; --t) {
    \\  int idx = (b * T + t) * D + d;
    \\  float xv = float(x[idx]);
    \\  int id = ids[b * T + t];
    \\  float a = cell_a(dec, g, xv, id, DOCSEP1);
    \\  float prev = t ? S[idx - D] : carry[b * D + d];
    \\  float k = (1.0f - a) + (prev - xv) * a * (1.0f - a) * g;
    \\  temb_out[(b * 256 + id) * D + d] += suffix * k;
    \\  suffix *= gm * a;
    \\}
;

var kernels: ?struct { fwd: mx.Kernel, bwd: mx.Kernel, trace: mx.Kernel, emb: mx.Kernel } = null;

fn get() @TypeOf(kernels.?) {
    if (kernels == null) kernels = .{
        .fwd = mx.Kernel.init("tmt_cell_fwd", &.{ "x", "decay", "gate", "carry", "ids" }, &.{"S"}, fwd_src, header),
        .bwd = mx.Kernel.init("tmt_cell_bwd", &.{ "dS", "S", "x", "decay", "gate", "carry", "ids" }, &.{ "dx", "gdec", "ggate", "dcarry" }, bwd_src, header),
        .trace = mx.Kernel.init("tmt_cell_trace", &.{ "S", "x", "decay", "gate", "carry", "ids", "tdec", "tgate", "gamma" }, &.{ "tdec_out", "tgate_out", "prod" }, trace_src, header),
        .emb = mx.Kernel.init("tmt_emb_trace", &.{ "x", "S", "carry", "decay", "gate", "ids", "temb", "prod", "gamma" }, &.{"temb_out"}, emb_src, header),
    };
    return kernels.?;
}

pub const Opt = struct { gated: bool, docsep: i32 };

const LANES = 32; // channels per threadgroup: one SIMD width, coalesced loads

/// Time chunks per channel. Splitting the window costs a second scan, so it
/// only pays while the GPU is not filled by B·D chains alone; measured on an
/// M2, the break-even sits around 16k chains. TMT_CHUNKS overrides it.
var forced_chunks: ?i32 = null;
fn chunkCount(B: i32, T: i32, D: i32) i32 {
    if (forced_chunks == null) forced_chunks = if (std.c.getenv("TMT_CHUNKS")) |v|
        std.fmt.parseInt(i32, std.mem.span(v), 10) catch 0
    else
        0;
    if (forced_chunks.? > 0) return @max(1, @min(forced_chunks.?, T));
    const want = @divTrunc(@as(i32, 16384), @max(1, B * D));
    return @max(1, @min(@min(want, @divTrunc(T, 16)), 8));
}

const Launch = struct { threads: usize, group: usize, chunks: i32 };

fn launch(x: Array) Launch {
    const B = mx.dim(x, 0);
    const T = mx.dim(x, 1);
    const D = mx.dim(x, 2);
    const ch = chunkCount(B, T, D);
    const tiles = @divTrunc(D + LANES - 1, LANES);
    return .{ .threads = @intCast(B * tiles * LANES * ch), .group = @intCast(LANES * ch), .chunks = ch };
}

fn templates(buf: *[7]mx.Kernel.Tmpl, x: Array, o: Opt, l: Launch) []const mx.Kernel.Tmpl {
    buf.* = .{
        .{ .name = "B", .value = mx.dim(x, 0) },
        .{ .name = "T", .value = mx.dim(x, 1) },
        .{ .name = "D", .value = mx.dim(x, 2) },
        .{ .name = "GATED", .value = @intFromBool(o.gated) },
        .{ .name = "DOCSEP1", .value = o.docsep + 1 },
        .{ .name = "LANES", .value = LANES },
        .{ .name = "CHUNKS", .value = l.chunks },
    };
    return buf;
}

/// Forward only: S (B,T,D) f32 from x (B,T,D), decay/gate (D) f32, carry (B,D) f32, ids (B,T) i32.
pub fn forwardRaw(x: Array, decay: Array, gate: Array, carry: Array, ids: Array, o: Opt) Array {
    var t: [7]mx.Kernel.Tmpl = undefined;
    var out: [1]Array = undefined;
    const l = launch(x);
    get().fwd.call(&.{ x, decay, gate, carry, ids }, &.{.{ .shape = mx.shape(x), .dtype = mx.f32_ }}, templates(&t, x, o, l), l.threads, l.group, &out);
    return out[0];
}

pub const Grads = struct { dx: Array, gdec: Array, ggate: Array, dcarry: Array };

/// Exact window backward from dL/dS. The decay/gate sums come back per stream
/// and time chunk, (B, chunks, D); the caller reduces them.
pub fn backwardRaw(dS: Array, S: Array, x: Array, decay: Array, gate: Array, carry: Array, ids: Array, o: Opt) Grads {
    var t: [7]mx.Kernel.Tmpl = undefined;
    const l = launch(x);
    const bcd = [_]i32{ mx.dim(x, 0), l.chunks, mx.dim(x, 2) };
    const bd = [_]i32{ mx.dim(x, 0), mx.dim(x, 2) };
    var out: [4]Array = undefined;
    get().bwd.call(&.{ mx.astype(dS, mx.f32_), S, x, decay, gate, carry, ids }, &.{
        .{ .shape = mx.shape(x), .dtype = mx.f32_ },
        .{ .shape = &bcd, .dtype = mx.f32_ },
        .{ .shape = &bcd, .dtype = mx.f32_ },
        .{ .shape = &bd, .dtype = mx.f32_ },
    }, templates(&t, x, o, l), l.threads, l.group, &out);
    return .{ .dx = out[0], .gdec = out[1], .ggate = out[2], .dcarry = out[3] };
}

pub const Traces = struct { tdec: Array, tgate: Array, prod: Array };

pub fn advanceTraces(S: Array, x: Array, decay: Array, gate: Array, carry: Array, ids: Array, tdec: Array, tgate: Array, gamma: f32, o: Opt) Traces {
    var t: [7]mx.Kernel.Tmpl = undefined;
    const l = launch(x);
    const bd = [_]i32{ mx.dim(x, 0), mx.dim(x, 2) };
    var out: [3]Array = undefined;
    get().trace.call(&.{ S, x, decay, gate, carry, ids, tdec, tgate, mx.fromSlice(f32, &.{gamma}, &.{1}) }, &.{
        .{ .shape = &bd, .dtype = mx.f32_ },
        .{ .shape = &bd, .dtype = mx.f32_ },
        .{ .shape = &bd, .dtype = mx.f32_ },
    }, templates(&t, x, o, l), l.threads, l.group, &out);
    return .{ .tdec = out[0], .tgate = out[1], .prod = out[2] };
}

pub fn advanceEmbTrace(x: Array, S: Array, carry: Array, decay: Array, gate: Array, ids: Array, temb: Array, prod: Array, gamma: f32, o: Opt) Array {
    var t: [7]mx.Kernel.Tmpl = undefined;
    var out: [1]Array = undefined;
    const l = launch(x);
    const n: usize = @intCast(mx.dim(x, 0) * mx.dim(x, 2));
    get().emb.call(&.{ x, S, carry, decay, gate, ids, temb, prod, mx.fromSlice(f32, &.{gamma}, &.{1}) }, &.{.{ .shape = mx.shape(temb), .dtype = mx.f32_ }}, templates(&t, x, o, l), n, 0, &out);
    return out[0];
}

// ---- differentiable wrapper ----
const Ctx = struct { o: Opt };

fn fwdFn(ctx: *anyopaque, in: []const Array) []Array {
    const k: *Ctx = @ptrCast(@alignCast(ctx));
    const out = mx.gpa.alloc(Array, 1) catch mx.oom();
    out[0] = forwardRaw(in[0], in[1], in[2], in[3], in[4], k.o);
    return out;
}
fn vjpFn(ctx: *anyopaque, p: []const Array, cot: []const Array, outs: []const Array) []Array {
    const k: *Ctx = @ptrCast(@alignCast(ctx));
    const g = backwardRaw(cot[0], outs[0], p[0], p[1], p[2], p[3], p[4], k.o);
    const r = mx.gpa.alloc(Array, 5) catch mx.oom();
    r[0] = mx.astype(g.dx, mx.dtype(p[0]));
    r[1] = sumPartials(g.gdec);
    r[2] = if (k.o.gated) sumPartials(g.ggate) else mx.zeros(&.{mx.dim(p[2], 0)}, mx.f32_);
    r[3] = g.dcarry;
    r[4] = mx.zeros(mx.shape(p[4]), mx.i32_);
    return r;
}
/// (B, chunks, D) partial sums -> (D).
fn sumPartials(a: Array) Array {
    return mx.sum(mx.sum(a, 0, false), 0, false);
}

/// Differentiable recurrence: apply(&.{x, decay, gate, carry, ids}) -> S.
pub const Recurrence = struct {
    f: mx.CustomFn,

    pub fn init(o: Opt) Recurrence {
        const ctx = mx.gpa.create(Ctx) catch mx.oom();
        ctx.* = .{ .o = o };
        return .{ .f = mx.CustomFn.init(fwdFn, vjpFn, ctx) };
    }
    pub fn apply(self: Recurrence, x: Array, decay: Array, gate: Array, carry: Array, ids: Array) Array {
        const out = self.f.apply(&.{ x, decay, gate, carry, ids });
        defer mx.gpa.free(out);
        return out[0];
    }
};
