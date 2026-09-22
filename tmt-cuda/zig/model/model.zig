//! The model itself: parameters, buffers and the streaming state (port of
//! src/model.cu). The kernels come from the PTX module, the GEMMs from cuBLAS.
const std = @import("std");
const gpu = @import("gpu.zig");
const cfgmod = @import("config.zig");
const params = @import("params.zig");
const moe = @import("moe.zig");
const mla = @import("mla.zig");
const memory = @import("memory.zig");

pub const bf16 = u16;
pub const Cfg = cfgmod.Cfg;
const MT_CHUNK: i64 = 4096;

pub const Layer = struct {
    decay: usize = 0,
    gate: usize = 0,
    gamma: usize = 0,
    beta: usize = 0,
    router: usize = 0,
    exp: [16]usize = @splat(0),
    input: [*]bf16 = undefined,
    S: [*]f32 = undefined, // (N,D) states
    mean: [*]f32 = undefined,
    rstd: [*]f32 = undefined,
    initial: [*]f32 = undefined, // (B,D) incoming carry
    mc: moe.Keep = .{},
};

pub const MlaLayer = struct {
    use: bool = false,
    p: mla.P = .{},
    keep: mla.Keep = .{},
    mmean: [*]f32 = undefined,
    mrstd: [*]f32 = undefined,
    Xsnap: [*]bf16 = undefined,
};

/// Multi-tensor view of all parameters, as the optimizer kernels take it.
pub const MTChunk = extern struct { param: i32, len: i32, start: i64 };
pub const MTParams = extern struct {
    master: u64 = 0,
    m: u64 = 0,
    v: u64 = 0,
    grad: u64 = 0,
    work: u64 = 0,
    flags: u64 = 0,
    chunks: u64 = 0,
    nchunks: i32 = 0,
    sumsq: u64 = 0,
};

pub const Model = struct {
    c: Cfg,
    gpa: std.mem.Allocator,
    store: params.Store,
    mem: gpu.Memory,
    kernels: *gpu.Kernels,

    emb: usize = 0,
    tgt: usize = 0,
    dec: usize = 0,
    stop: usize = 0,
    L: []Layer = &.{},
    ML: []MlaLayer = &.{},
    MW: mla.Ws = .{},
    MEM: []memory.Layer = &.{},
    MS: memory.Shared = .{},
    moeW: moe.Ws = .{},

    enc: [*]bf16 = undefined,
    X: [*]bf16 = undefined,
    Sb: [*]bf16 = undefined,
    H: [*]bf16 = undefined,
    M: [*]bf16 = undefined,
    dXs: [*]bf16 = undefined,
    dM: [*]bf16 = undefined,
    dSnorm: [*]bf16 = undefined,
    dH: [*]bf16 = undefined,
    logits: [*]bf16 = undefined,
    dlogits: [*]bf16 = undefined,
    stoplog: [*]bf16 = undefined,
    dstop: [*]bf16 = undefined,
    tgtX: [*]bf16 = undefined,
    dXres: [*]f32 = undefined,
    dEnc: [*]f32 = undefined,
    dEncTmp: [*]f32 = undefined,
    probs: [*]f32 = undefined,
    losstmp: [*]f32 = undefined,
    mv: [*]f32 = undefined,
    dDec: [*]f32 = undefined,
    lam: [*]f32 = undefined,
    prod: [*]f32 = undefined,
    trlog: ?[*]f32 = null,
    ids: [*]i32 = undefined,
    nxt: [*]i32 = undefined,
    end: [*]i32 = undefined,
    pos: [*]i64 = undefined,
    moe_stats: ?[*]f32 = null,
    /// Diagnostics: accumulate the trace part of the gradients in `trlog`.
    log_traces: bool = false,
    /// Optional external gradient on the final representation (retrieval loss).
    dXext: ?[*]f32 = null,
    opt: MTParams = .{},

    pub fn deinit(m: *Model) void {
        m.store.deinit();
        m.mem.deinit();
        if (m.L.len != 0) m.gpa.free(m.L);
        if (m.ML.len != 0) m.gpa.free(m.ML);
        if (m.MEM.len != 0) m.gpa.free(m.MEM);
    }
};

/// Learning rate: linear warmup, then a cosine decay to `minlr`.
pub fn lrAt(c: Cfg, step: i32) f32 {
    if (step < c.warmup)
        return c.lr * @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(c.warmup));
    const p = @min(@as(f32, 1.0), @as(f32, @floatFromInt(step - c.warmup)) / @as(f32, @floatFromInt(c.decaysteps)));
    return c.lr * (c.minlr + 0.5 * (1 - c.minlr) * (1 + params.cosf(3.14159265 * p)));
}

/// Build every parameter and buffer. The order of the random draws is the
/// order of the C++ build, so the same seed gives the same weights.
pub fn build(gpa: std.mem.Allocator, c: Cfg, kernels: *gpu.Kernels) !Model {
    var m = Model{ .c = c, .gpa = gpa, .store = params.Store.init(gpa), .mem = gpu.Memory.init(gpa), .kernels = kernels };
    errdefer m.deinit();
    const D: usize = @intCast(c.dim);
    const nl: usize = @intCast(c.layers);
    const E: usize = @intCast(c.experts);
    const K: usize = @intCast(c.topk);
    const B: usize = @intCast(c.batch);
    const T: usize = @intCast(c.seqlen);
    const N = B * T;
    const ND = N * D;
    var seed: u32 = if (c.seed != 0) @bitCast(c.seed) else 1234;

    var host: std.ArrayList(f32) = .empty;
    defer host.deinit(gpa);
    // Uniform initialization plus the bf16 working copy, as init_u does.
    const initU = struct {
        fn f(mm: *Model, h: *std.ArrayList(f32), g: std.mem.Allocator, p: usize, a: f32, b: f32, s: *u32) !void {
            const n: usize = @intCast(mm.store.at(p).n);
            try h.resize(g, n);
            params.hostInit(h.items, a, b, s);
            try gpu.upload(mm.store.at(p).master, std.mem.sliceAsBytes(h.items));
            try toWork(mm, p);
        }
    }.f;

    m.emb = try m.store.add(256 * @as(i64, c.dim));
    try initU(&m, &host, gpa, m.emb, -0.05, 0.05, &seed);
    m.tgt = try m.store.add(256 * @as(i64, c.dim));
    try gpu.copyDevice(m.store.at(m.tgt).master, m.store.at(m.emb).master, 256 * D * 4);
    try gpu.copyDevice(m.store.at(m.tgt).work, m.store.at(m.emb).work, 256 * D * 2);
    m.dec = try m.store.add(256 * @as(i64, c.dim));
    {
        const a = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
        try initU(&m, &host, gpa, m.dec, -a, a, &seed);
    }
    m.stop = try m.store.add(@intCast(D));
    try initU(&m, &host, gpa, m.stop, -0.01, 0.01, &seed);

    m.L = try gpa.alloc(Layer, nl);
    for (m.L) |*ly| ly.* = .{};
    for (m.L) |*ly| {
        ly.decay = try m.store.add(@intCast(D));
        try host.resize(gpa, D);
        for (host.items, 0..) |*v, d| {
            const half = c.half_min * params.powf(c.half_max / c.half_min,
                @as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(@max(1, D - 1))));
            const a = params.expf(-params.logf(2.0) / half);
            v.* = params.logf(a / (1.0 - a));
        }
        try gpu.upload(m.store.at(ly.decay).master, std.mem.sliceAsBytes(host.items));
        try toWork(&m, ly.decay);
        ly.gate = try m.store.add(@intCast(D)); // draws D values and keeps zero
        try initU(&m, &host, gpa, ly.gate, 0, 0, &seed);
        ly.gamma = try m.store.add(@intCast(D));
        @memset(host.items, 1.0);
        try gpu.upload(m.store.at(ly.gamma).master, std.mem.sliceAsBytes(host.items));
        try toWork(&m, ly.gamma);
        ly.beta = try m.store.add(@intCast(D));
        try gpu.zero(m.store.at(ly.beta).master, D * 4);
        try gpu.zero(m.store.at(ly.beta).work, D * 2);
        ly.router = try m.store.add(@intCast(E * D));
        try host.resize(gpa, E * D);
        params.hostNormal(host.items, 0.02, &seed);
        try gpu.upload(m.store.at(ly.router).master, std.mem.sliceAsBytes(host.items));
        try toWork(&m, ly.router);
        const a = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
        for (0..E) |e| {
            ly.exp[e] = try m.store.add(@intCast(D * D));
            try initU(&m, &host, gpa, ly.exp[e], -a, a, &seed);
        }
        ly.S = try m.mem.allocT(f32, ND);
        ly.mean = try m.mem.allocT(f32, N);
        ly.rstd = try m.mem.allocT(f32, N);
        ly.initial = try m.mem.allocT(f32, B * D);
        ly.input = try m.mem.allocT(bf16, ND);
        try ly.mc.alloc(&m.mem, N, E, K, D);
        ly.mc.N = @intCast(N);
        ly.mc.E = @intCast(E);
        ly.mc.K = @intCast(K);
        ly.mc.D = @intCast(D);
    }
    try m.moeW.alloc(&m.mem, N, E, K, D);
    if (E > 1) {
        m.moe_stats = try m.mem.allocT(f32, nl * 17);
        for (m.L, 0..) |*ly, l| ly.mc.stat = m.moe_stats.? + l * 17;
    }
    m.enc = try m.mem.allocT(bf16, ND);
    m.X = try m.mem.allocT(bf16, ND);
    m.Sb = try m.mem.allocT(bf16, ND);
    m.H = try m.mem.allocT(bf16, ND);
    m.M = try m.mem.allocT(bf16, ND);
    m.dXs = try m.mem.allocT(bf16, ND);
    m.dM = try m.mem.allocT(bf16, ND);
    m.dSnorm = try m.mem.allocT(bf16, ND);
    m.dH = try m.mem.allocT(bf16, ND);
    m.dXres = try m.mem.allocT(f32, ND);
    m.dEnc = try m.mem.allocT(f32, ND);
    m.dEncTmp = try m.mem.allocT(f32, ND);
    m.dDec = try m.mem.allocT(f32, D);
    m.lam = try m.mem.allocT(f32, B * D);
    m.prod = try m.mem.allocT(f32, B * D);
    if (c.traces != 0) m.trlog = try m.mem.allocT(f32, (2 * nl + 256) * D);
    m.ids = try m.mem.allocT(i32, N);
    m.nxt = try m.mem.allocT(i32, N);
    m.end = try m.mem.allocT(i32, N);
    m.logits = try m.mem.allocT(bf16, N * 256);
    m.dlogits = try m.mem.allocT(bf16, N * 256);
    m.stoplog = try m.mem.allocT(bf16, N);
    m.dstop = try m.mem.allocT(bf16, N);
    m.probs = try m.mem.allocT(f32, N * 256);
    m.losstmp = try m.mem.allocT(f32, N);
    m.tgtX = try m.mem.allocT(bf16, ND);
    m.mv = try m.mem.allocT(f32, 2);
    m.pos = try m.mem.allocT(i64, N);

    m.ML = try gpa.alloc(MlaLayer, nl);
    for (m.ML) |*ml| ml.* = .{};
    if (c.mla != 0) {
        const H: usize = @intCast(c.mla_heads);
        const dh: usize = @intCast(c.mla_dh);
        const Lr: usize = @intCast(c.mla_L);
        const R: usize = @intCast(c.mla_R);
        const Cc: usize = @intCast(c.mla_cc);
        const a = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
        for (m.ML, 0..) |*ml, l| {
            ml.use = @rem(@as(i32, @intCast(l)), c.mla_every) == 0;
            if (!ml.use) continue;
            ml.p.q = try m.store.add(@intCast(D * H * (dh + R)));
            try initU(&m, &host, gpa, ml.p.q, -a, a, &seed);
            ml.p.dkv = try m.store.add(@intCast(D * Lr));
            try initU(&m, &host, gpa, ml.p.dkv, -a, a, &seed);
            ml.p.kr = try m.store.add(@intCast(D * R));
            try initU(&m, &host, gpa, ml.p.kr, -a, a, &seed);
            const b = params.sqrtf(1.0 / @as(f32, @floatFromInt(Lr)));
            ml.p.uk = try m.store.add(@intCast(Lr * H * dh));
            try initU(&m, &host, gpa, ml.p.uk, -b, b, &seed);
            ml.p.uv = try m.store.add(@intCast(Lr * H * dh));
            try initU(&m, &host, gpa, ml.p.uv, -b, b, &seed);
            const b2 = params.sqrtf(1.0 / @as(f32, @floatFromInt(H * dh)));
            ml.p.o = try m.store.add(@intCast(H * dh * D));
            try initU(&m, &host, gpa, ml.p.o, -b2, b2, &seed);
            ml.p.gamma = try m.store.add(@intCast(D));
            try host.resize(gpa, D);
            @memset(host.items, 1.0);
            try gpu.upload(m.store.at(ml.p.gamma).master, std.mem.sliceAsBytes(host.items));
            try toWork(&m, ml.p.gamma);
            ml.p.beta = try m.store.add(@intCast(D));
            try gpu.zero(m.store.at(ml.p.beta).master, D * 4);
            try gpu.zero(m.store.at(ml.p.beta).work, D * 2);
            ml.keep.qc = try m.mem.allocT(bf16, B * H * T * dh);
            ml.keep.qr = try m.mem.allocT(bf16, B * H * T * R);
            ml.keep.Oflat = try m.mem.allocT(bf16, N * H * dh);
            ml.keep.Xq = try m.mem.allocT(bf16, ND);
            ml.keep.LSE = try m.mem.allocT(f32, B * H * T);
            ml.mmean = try m.mem.allocT(f32, N);
            ml.mrstd = try m.mem.allocT(f32, N);
            ml.Xsnap = try m.mem.allocT(bf16, ND);
        }
        try m.MW.alloc(&m.mem, B, T, H, dh, Lr, R, Cc);
    }

    // The fact memory comes last, so every other parameter keeps its
    // initialization; Wo = 0 makes the untrained memory an exact no-op.
    m.MEM = try gpa.alloc(memory.Layer, nl);
    for (m.MEM) |*me| me.* = .{};
    if (c.mem != 0) {
        const Mlen: usize = @intCast(c.mem_len);
        const HD: usize = @intCast(c.mem_heads * c.mem_dh);
        m.MS.eprev = try m.store.add(256 * @as(i64, c.dim));
        try initU(&m, &host, gpa, m.MS.eprev, -0.05, 0.05, &seed);
        m.MS.pos = try m.store.add(@intCast(Mlen * D));
        try initU(&m, &host, gpa, m.MS.pos, -0.05, 0.05, &seed);
        m.MS.decay = try m.store.add(@intCast(D));
        try host.resize(gpa, D);
        for (host.items, 0..) |*v, d| {
            const half = params.powf(64.0, @as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(@max(1, D - 1))));
            const a = params.expf(-params.logf(2.0) / half);
            v.* = params.logf(a / (1.0 - a));
        }
        try gpu.upload(m.store.at(m.MS.decay).master, std.mem.sliceAsBytes(host.items));
        try toWork(&m, m.MS.decay);
        m.MS.gate = try m.store.add(@intCast(D));
        try initU(&m, &host, gpa, m.MS.gate, 0, 0, &seed);
        const a = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
        for (m.MEM, 0..) |*me, l| {
            me.use = @rem(@as(i32, @intCast(l)) + 1, c.mem_every) == 0;
            if (!me.use) continue;
            me.gamma = try m.store.add(@intCast(D));
            try host.resize(gpa, D);
            @memset(host.items, 1.0);
            try gpu.upload(m.store.at(me.gamma).master, std.mem.sliceAsBytes(host.items));
            try toWork(&m, me.gamma);
            me.beta = try m.store.add(@intCast(D));
            me.wq = try m.store.add(@intCast(HD * D));
            try initU(&m, &host, gpa, me.wq, -a, a, &seed);
            me.wk = try m.store.add(@intCast(HD * D));
            try initU(&m, &host, gpa, me.wk, -a, a, &seed);
            me.wv = try m.store.add(@intCast(HD * D));
            try initU(&m, &host, gpa, me.wv, -a, a, &seed);
            me.wo = try m.store.add(@intCast(D * HD));
            for ([_]usize{ me.beta, me.wo }) |p| {
                const n: usize = @intCast(m.store.at(p).n);
                try gpu.zero(m.store.at(p).master, n * 4);
                try gpu.zero(m.store.at(p).work, n * 2);
            }
            me.Xsnap = try m.mem.allocT(bf16, ND);
            me.Hn = try m.mem.allocT(bf16, ND);
            me.Q = try m.mem.allocT(bf16, N * HD);
            me.O = try m.mem.allocT(bf16, N * HD);
            me.K = try m.mem.allocT(bf16, B * Mlen * HD);
            me.V = try m.mem.allocT(bf16, B * Mlen * HD);
            me.mean = try m.mem.allocT(f32, N);
            me.rstd = try m.mem.allocT(f32, N);
            me.P = try m.mem.allocT(f32, B * @as(usize, @intCast(c.mem_heads)) * T * Mlen);
        }
        m.MS.ids = try m.mem.allocT(i32, B * Mlen);
        try fill(&m, @ptrCast(m.MS.ids), 0xff, B * Mlen * 4); // -1: no memory
        m.MS.enc = try m.mem.allocT(bf16, B * Mlen * D);
        m.MS.enc0 = try m.mem.allocT(bf16, B * Mlen * D);
        m.MS.state = try m.mem.allocT(f32, B * Mlen * D);
        m.MS.dEnc0 = try m.mem.allocT(f32, B * Mlen * D);
        m.MS.dO = try m.mem.allocT(bf16, N * HD);
        m.MS.dQ = try m.mem.allocT(bf16, N * HD);
        m.MS.dK = try m.mem.allocT(bf16, B * Mlen * HD);
        m.MS.dV = try m.mem.allocT(bf16, B * Mlen * HD);
        m.MS.dHn = try m.mem.allocT(bf16, ND);
        m.MS.dXln = try m.mem.allocT(bf16, ND);
        m.MS.dEncB = try m.mem.allocT(bf16, B * Mlen * D);
        m.MS.dS = try m.mem.allocT(f32, B * @as(usize, @intCast(c.mem_heads)) * T * Mlen);
        m.MS.dEnc = try m.mem.allocT(f32, B * Mlen * D);
        if (c.mem_rdim > 0) { // retrieval heads, after every other parameter
            const r = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
            m.MS.rq = try m.store.add(@intCast(@as(usize, @intCast(c.mem_rdim)) * D));
            try initU(&m, &host, gpa, m.MS.rq, -r, r, &seed);
            m.MS.rk = try m.store.add(@intCast(@as(usize, @intCast(c.mem_rdim)) * D));
            try initU(&m, &host, gpa, m.MS.rk, -r, r, &seed);
        }
    }
    try buildOptTable(&m);
    try gpu.checkLaunch();
    return m;
}

/// master -> bf16 working copy.
fn toWork(m: *Model, p: usize) !void {
    const par = m.store.at(p);
    const n: u32 = @intCast(@divTrunc(par.n + 255, 256));
    try (try m.kernels.get("copy_bf16")).launch(n, 256, .{ par.master, par.work, par.n });
}

extern fn cudaMemset(p: ?*anyopaque, value: c_int, n: usize) c_int;
fn fill(m: *Model, p: *anyopaque, value: u8, bytes: usize) !void {
    _ = m;
    if (cudaMemset(p, value, bytes) != 0) return error.Cuda;
}

/// One chunk table over all parameters for the multi-tensor kernels. The norm
/// covers everything but the EMA target; AdamW also skips an unused stop head.
fn buildOptTable(m: *Model) !void {
    const P = m.store.values.items.len;
    const gpa = m.gpa;
    const master = try gpa.alloc(u64, P);
    const mm = try gpa.alloc(u64, P);
    const vv = try gpa.alloc(u64, P);
    const grad = try gpa.alloc(u64, P);
    const work = try gpa.alloc(u64, P);
    const flags = try gpa.alloc(u8, P);
    defer inline for (.{ master, mm, vv, grad, work }) |x| gpa.free(x);
    defer gpa.free(flags);
    var chunks: std.ArrayList(MTChunk) = .empty;
    defer chunks.deinit(gpa);
    for (0..P) |i| {
        const p = m.store.at(i);
        master[i] = @intFromPtr(p.master);
        mm[i] = @intFromPtr(p.m);
        vv[i] = @intFromPtr(p.v);
        grad[i] = @intFromPtr(p.grad);
        work[i] = @intFromPtr(p.work);
        const frozen = i == m.tgt;
        const unused_stop = i == m.stop and m.c.stop == 0;
        flags[i] = (if (frozen) @as(u8, 0) else 1) | (if (frozen or unused_stop) @as(u8, 0) else 2);
        var s: i64 = 0;
        while (s < p.n) : (s += MT_CHUNK) {
            try chunks.append(gpa, .{ .param = @intCast(i), .len = @intCast(@min(MT_CHUNK, p.n - s)), .start = s });
        }
    }
    const up = struct {
        fn f(mo: *Model, bytes: []const u8) !u64 {
            const d = try mo.mem.alloc(bytes.len);
            try gpu.upload(d, bytes);
            return @intFromPtr(d);
        }
    }.f;
    m.opt = .{
        .master = try up(m, std.mem.sliceAsBytes(master)),
        .m = try up(m, std.mem.sliceAsBytes(mm)),
        .v = try up(m, std.mem.sliceAsBytes(vv)),
        .grad = try up(m, std.mem.sliceAsBytes(grad)),
        .work = try up(m, std.mem.sliceAsBytes(work)),
        .flags = try up(m, flags),
        .chunks = try up(m, std.mem.sliceAsBytes(chunks.items)),
        .nchunks = @intCast(chunks.items.len),
        .sumsq = @intFromPtr(try m.mem.alloc(8)),
    };
}

/// The streaming state of one data stream per batch row.
pub const StreamState = struct {
    mem: gpu.Memory,
    gpa: std.mem.Allocator,
    carry: [][*]f32 = &.{},
    cache: []mla.Cache = &.{},
    position: i64 = 0,
    tdec: [][*]f32 = &.{}, // hybrid traces across windows
    tgate: [][*]f32 = &.{},
    temb: ?[*]f32 = null,

    pub fn deinit(s: *StreamState) void {
        if (s.carry.len != 0) s.gpa.free(s.carry);
        if (s.cache.len != 0) s.gpa.free(s.cache);
        if (s.tdec.len != 0) s.gpa.free(s.tdec);
        if (s.tgate.len != 0) s.gpa.free(s.tgate);
        s.mem.deinit();
    }
};

pub fn buildState(gpa: std.mem.Allocator, m: *const Model) !StreamState {
    const c = m.c;
    const nl: usize = @intCast(c.layers);
    const B: usize = @intCast(c.batch);
    const D: usize = @intCast(c.dim);
    var s = StreamState{ .mem = gpu.Memory.init(gpa), .gpa = gpa };
    errdefer s.deinit();
    s.carry = try gpa.alloc([*]f32, nl);
    s.cache = try gpa.alloc(mla.Cache, nl);
    for (s.cache) |*x| x.* = .{};
    if (c.traces != 0) {
        s.tdec = try gpa.alloc([*]f32, nl);
        s.tgate = try gpa.alloc([*]f32, nl);
        for (0..nl) |l| {
            s.tdec[l] = try s.mem.allocT(f32, B * D);
            s.tgate[l] = try s.mem.allocT(f32, B * D);
        }
        s.temb = try s.mem.allocT(f32, 256 * B * D);
    }
    for (0..nl) |l| {
        s.carry[l] = try s.mem.allocT(f32, B * D);
        if (m.ML[l].use) {
            const cache = &s.cache[l];
            cache.B = c.batch;
            cache.Cmax = c.mla_cache;
            cache.L = c.mla_L;
            cache.R = c.mla_R;
            const cells = B * @as(usize, @intCast(c.mla_cache));
            cache.lat = try s.mem.callocT(bf16, cells * @as(usize, @intCast(c.mla_L)));
            cache.kr = try s.mem.callocT(bf16, cells * @as(usize, @intCast(c.mla_R)));
        }
    }
    try reset(&s, c);
    return s;
}

pub fn reset(s: *StreamState, c: Cfg) !void {
    const bd: usize = @as(usize, @intCast(c.batch)) * @as(usize, @intCast(c.dim)) * 4;
    for (s.carry) |carry| try gpu.zero(carry, bd);
    for (s.cache) |*cache| {
        cache.head = 0;
        cache.base0 = 0;
    }
    for (s.tdec) |t| try gpu.zero(t, bd);
    for (s.tgate) |t| try gpu.zero(t, bd);
    if (s.temb) |t| try gpu.zero(t, 256 * bd);
    s.position = 0;
}
