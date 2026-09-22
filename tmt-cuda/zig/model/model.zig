//! The model itself: parameters, buffers and the streaming state (port of
//! src/model.cu). The kernels come from the PTX module, the GEMMs from cuBLAS.
const std = @import("std");
const gpu = @import("gpu.zig");
const cfgmod = @import("config.zig");
const params = @import("params.zig");
const moe = @import("moe.zig");
const mla = @import("mla.zig");
const memory = @import("memory.zig");
const muon = @import("muon.zig");

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
    /// Multi-token prediction: one more decoder per extra byte of lookahead.
    dec_mtp: []usize = &.{},
    logits_mtp: [*]bf16 = undefined,
    probs_mtp: [*]f32 = undefined,
    nxt_mtp: [*]i32 = undefined,
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
    /// The weights Muon updates instead of AdamW, with its scratch.
    muon_params: []usize = &.{},
    muon_ws: muon.Ws = .{},
    muon_x: [*]f32 = undefined,

    pub fn deinit(m: *Model) void {
        m.store.deinit();
        m.mem.deinit();
        if (m.L.len != 0) m.gpa.free(m.L);
        if (m.ML.len != 0) m.gpa.free(m.ML);
        if (m.MEM.len != 0) m.gpa.free(m.MEM);
        if (m.dec_mtp.len != 0) m.gpa.free(m.dec_mtp);
        if (m.muon_params.len != 0) m.gpa.free(m.muon_params);
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
    if (c.mtp > 0) { // the extra decoders start like the main one
        m.dec_mtp = try gpa.alloc(usize, @intCast(c.mtp));
        const a = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
        for (m.dec_mtp) |*p| {
            p.* = try m.store.add(256 * @as(i64, c.dim));
            try initU(&m, &host, gpa, p.*, -a, a, &seed);
        }
    }

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
        ly.router = try m.store.add2d(@intCast(E), @intCast(D));
        try host.resize(gpa, E * D);
        params.hostNormal(host.items, 0.02, &seed);
        try gpu.upload(m.store.at(ly.router).master, std.mem.sliceAsBytes(host.items));
        try toWork(&m, ly.router);
        const a = params.sqrtf(1.0 / @as(f32, @floatFromInt(D)));
        for (0..E) |e| {
            ly.exp[e] = try m.store.add2d(@intCast(D), @intCast(D));
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
    if (c.mtp > 0) {
        const heads: usize = @intCast(c.mtp);
        m.logits_mtp = try m.mem.allocT(bf16, heads * N * 256);
        m.probs_mtp = try m.mem.allocT(f32, heads * N * 256);
        m.nxt_mtp = try m.mem.allocT(i32, heads * N);
    }
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
            ml.p.q = try m.store.add2d(@intCast(H * (dh + R)), @intCast(D));
            try initU(&m, &host, gpa, ml.p.q, -a, a, &seed);
            ml.p.dkv = try m.store.add2d(@intCast(Lr), @intCast(D));
            try initU(&m, &host, gpa, ml.p.dkv, -a, a, &seed);
            ml.p.kr = try m.store.add2d(@intCast(R), @intCast(D));
            try initU(&m, &host, gpa, ml.p.kr, -a, a, &seed);
            const b = params.sqrtf(1.0 / @as(f32, @floatFromInt(Lr)));
            ml.p.uk = try m.store.add2d(@intCast(H * dh), @intCast(Lr));
            try initU(&m, &host, gpa, ml.p.uk, -b, b, &seed);
            ml.p.uv = try m.store.add2d(@intCast(H * dh), @intCast(Lr));
            try initU(&m, &host, gpa, ml.p.uv, -b, b, &seed);
            const b2 = params.sqrtf(1.0 / @as(f32, @floatFromInt(H * dh)));
            ml.p.o = try m.store.add2d(@intCast(D), @intCast(H * dh));
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
            me.wq = try m.store.add2d(@intCast(HD), @intCast(D));
            try initU(&m, &host, gpa, me.wq, -a, a, &seed);
            me.wk = try m.store.add2d(@intCast(HD), @intCast(D));
            try initU(&m, &host, gpa, me.wk, -a, a, &seed);
            me.wv = try m.store.add2d(@intCast(HD), @intCast(D));
            try initU(&m, &host, gpa, me.wv, -a, a, &seed);
            me.wo = try m.store.add2d(@intCast(D), @intCast(HD));
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
    if (c.muon != 0) try collectMuon(&m);
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

/// Muon orthogonalizes the hidden weight matrices; the embedding, the output
/// head and every vector keep AdamW, as the published recipe prescribes.
fn isMuonParam(m: *const Model, j: usize) bool {
    if (m.c.muon == 0) return false;
    if (j == m.emb or j == m.tgt or j == m.dec or j == m.stop) return false;
    for (m.dec_mtp) |p| if (j == p) return false;
    const p = m.store.at(j);
    return p.rows > 1 and p.cols > 1;
}

fn collectMuon(m: *Model) !void {
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(m.gpa);
    var rows: usize = 0;
    var cols: usize = 0;
    var widest: usize = 0;
    for (0..m.store.values.items.len) |j| {
        if (!isMuonParam(m, j)) continue;
        try list.append(m.gpa, j);
        const p = m.store.at(j);
        rows = @max(rows, @as(usize, @intCast(p.rows)));
        cols = @max(cols, @as(usize, @intCast(p.cols)));
        widest = @max(widest, @as(usize, @intCast(p.n)));
    }
    m.muon_params = try list.toOwnedSlice(m.gpa);
    if (m.muon_params.len == 0) return;
    try m.muon_ws.alloc(&m.mem, rows, cols);
    m.muon_x = try m.mem.allocT(f32, widest);
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
        // Muon weights still count towards the gradient norm, but AdamW skips them.
        const adam = !frozen and !unused_stop and !isMuonParam(m, i);
        flags[i] = (if (frozen) @as(u8, 0) else 1) | (if (adam) @as(u8, 2) else 0);
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

const linalg = @import("linalg.zig");
const ops = @import("ops.zig");

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

/// Sum of a device vector, accumulated in double on the host.
fn hostSum(gpa: std.mem.Allocator, d: [*]const f32, n: usize) !f32 {
    const h = try gpa.alloc(f32, n);
    defer gpa.free(h);
    try gpu.download(std.mem.sliceAsBytes(h), d);
    var s: f64 = 0;
    for (h) |v| s += v;
    return @floatCast(s);
}

pub const Losses = struct { total: f32, ce: f32 };

/// One window: embedding, the layer stack, the decoder and the losses.
pub fn forwardWindow(m: *Model, s: *StreamState) !Losses {
    const c = m.c;
    const B: usize = @intCast(c.batch);
    const T: usize = @intCast(c.seqlen);
    const D: usize = @intCast(c.dim);
    const E: usize = @intCast(c.experts);
    const N = B * T;
    const ND = N * D;
    const k = m.kernels;

    try ops.embForward(k, m.store.at(m.emb).work, m.ids, m.enc, @intCast(N), c.dim);
    try gpu.copyDevice(m.X, m.enc, ND * 2);
    if (c.mem != 0)
        try memory.encode(k, m.store.at(m.emb).work, m.store.at(m.MS.eprev).work,
            m.store.at(m.MS.pos).work, m.store.at(m.MS.decay).master, m.store.at(m.MS.gate).master,
            &m.MS, c.batch, c.mem_len, c.dim);
    if (c.mla != 0) { // positions advance in lockstep across the streams
        const hp = try m.gpa.alloc(i64, N);
        defer m.gpa.free(hp);
        for (0..B) |b| for (0..T) |t| {
            hp[b * T + t] = s.position + @as(i64, @intCast(t));
        };
        try gpu.upload(m.pos, std.mem.sliceAsBytes(hp));
    }
    var aux_acc: f32 = 0;
    var z_acc: f32 = 0;
    var Wx: [16][*]bf16 = undefined;
    for (m.L, 0..) |*ly, l| {
        try gpu.copyDevice(ly.input, m.X, ND * 2);
        try gpu.copyDevice(ly.initial, s.carry[l], B * D * 4);
        const opt = ops.CellOpt{ .ids = @intFromPtr(m.ids), .docsep = c.docsep };
        const gate: u64 = if (c.gated != 0) @intFromPtr(m.store.at(ly.gate).master) else 0;
        try ops.stateForward(k, ly.input, ly.S, m.store.at(ly.decay).master,
            @intFromPtr(ly.initial), c.batch, c.seqlen, c.dim, gate, opt);
        try (try k.get("copy_bf16")).launch(blocks(ND), 256, .{ ly.S, m.Sb, @as(i64, @intCast(ND)) });
        try ops.layernormFwd(k, m.Sb, m.store.at(ly.gamma).master, m.store.at(ly.beta).master,
            m.H, ly.mean, ly.rstd, @intCast(N), c.dim);
        for (0..E) |e| Wx[e] = m.store.at(ly.exp[e]).work;
        // The residual add is folded into the combine (beta = 1).
        aux_acc += try moe.forward(k, m.H, m.store.at(ly.router).work, Wx[0..E], m.X,
            &ly.mc, &m.moeW, N, E, @intCast(c.topk), D, 1.0);
        if (c.mla != 0 and m.ML[l].use) {
            const ml = &m.ML[l];
            try gpu.copyDevice(ml.Xsnap, m.X, ND * 2);
            try ops.layernormFwd(k, m.X, m.store.at(ml.p.gamma).master, m.store.at(ml.p.beta).master,
                m.H, ml.mmean, ml.mrstd, @intCast(N), c.dim);
            const sc = 1.0 / @sqrt(@as(f32, @floatFromInt(c.mla_dh)));
            try mla.forward(k, m.H, @intCast(N), m.store.at(ml.p.q).work, m.store.at(ml.p.dkv).work,
                m.store.at(ml.p.kr).work, m.store.at(ml.p.uk).work, m.store.at(ml.p.uv).work,
                m.store.at(ml.p.o).work, &s.cache[l], &ml.keep, &m.MW, m.M, m.pos,
                c.batch, c.seqlen, c.mla_heads, c.mla_dh, c.mla_L, c.mla_R, c.mla_cc,
                c.mla_cache, c.mla_theta, sc, c.dim);
            try (try k.get("mem_add_bf16")).launch(blocks(ND), 256, .{ m.X, m.M, @as(i64, @intCast(ND)) });
        }
        if (c.mem != 0 and m.MEM[l].use) {
            const me = &m.MEM[l];
            try memory.forward(k, m.X, me, &m.MS, m.store.at(me.gamma).master, m.store.at(me.beta).master,
                m.store.at(me.wq).work, m.store.at(me.wk).work, m.store.at(me.wv).work,
                m.store.at(me.wo).work, m.M, c.batch, c.seqlen, c.mem_len, c.mem_heads, c.mem_dh, c.dim);
        }
        const gy: u32 = @intCast(@divTrunc(c.dim + 255, 256));
        try (try k.get("extract_carry")).launchGrid(@intCast(B), gy, 256,
            .{ ly.S, s.carry[l], c.batch, c.seqlen, c.dim });
    }
    s.position += @intCast(T);
    if (E > 1) { // one host read for the router statistics of every layer
        const hs = try m.gpa.alloc(f32, m.L.len * 17);
        defer m.gpa.free(hs);
        try gpu.download(std.mem.sliceAsBytes(hs), m.moe_stats.?);
        for (m.L, 0..) |*ly, l| {
            aux_acc += moe.auxFrom(&ly.mc, hs[l * 17 ..][0 .. E + 1]);
            z_acc += ly.mc.zloss;
        }
    }
    try linalg.linearFwd(@intCast(N), 256, c.dim, m.X, m.store.at(m.dec).work, m.logits);
    try (try k.get("ce_fwd")).launch(blocks(N), 256, .{ m.logits, m.nxt, m.probs, m.losstmp, @as(i32, @intCast(N)) });
    const ce_out = try hostSum(m.gpa, m.losstmp, N) / @as(f32, @floatFromInt(N));

    // Multi-token prediction: the same representation predicts the bytes
    // further ahead, which is extra signal per byte for a data-limited model.
    var mtp_ce: f32 = 0;
    for (m.dec_mtp, 0..) |p, head| {
        const logits = m.logits_mtp + head * N * 256;
        const probs = m.probs_mtp + head * N * 256;
        const targets = m.nxt_mtp + head * N;
        try linalg.linearFwd(@intCast(N), 256, c.dim, m.X, m.store.at(p).work, logits);
        try (try k.get("ce_fwd")).launch(blocks(N), 256, .{ logits, targets, probs, m.losstmp, @as(i32, @intCast(N)) });
        mtp_ce += try hostSum(m.gpa, m.losstmp, N) / @as(f32, @floatFromInt(N));
    }
    if (m.dec_mtp.len > 0) mtp_ce /= @floatFromInt(m.dec_mtp.len);

    // The optional terms each cost one host synchronization and only run when enabled.
    var stop_mean: f32 = 0;
    var var_loss: f32 = 0;
    var mse: f32 = 0;
    if (c.stop > 0) {
        try linalg.linearFwd(@intCast(N), 1, c.dim, m.X, m.store.at(m.stop).work, m.stoplog);
        try (try k.get("stop_fwd")).launch(blocks(N), 256, .{ m.stoplog, m.end, m.losstmp, c.stopposw, @as(i32, @intCast(N)) });
        stop_mean = try hostSum(m.gpa, m.losstmp, N) / @as(f32, @floatFromInt(N));
    }
    if (c.latent > 0) { // against the EMA target embedding of the next byte
        try ops.embForward(k, m.store.at(m.tgt).work, m.nxt, m.tgtX, @intCast(N), c.dim);
        try (try k.get("to_f32")).launch(blocks(ND), 256, .{ m.X, m.dXres, @as(i64, @intCast(ND)) });
        try (try k.get("to_f32")).launch(blocks(ND), 256, .{ m.tgtX, m.dEnc, @as(i64, @intCast(ND)) });
        try (try k.get("mse_mean")).launch(1, 256, .{ m.dXres, m.dEnc, m.mv, @as(i64, @intCast(ND)) });
        try gpu.download(std.mem.asBytes(&mse), m.mv);
    }
    if (c.@"var" > 0) {
        var hmv: [2]f32 = undefined;
        try (try k.get("meanvar")).launch(1, 256, .{ m.X, m.mv, @as(i64, @intCast(ND)) });
        try gpu.download(std.mem.sliceAsBytes(hmv[0..2]), m.mv);
        var_loss = @max(0.0, 1.0 - params.sqrtf(hmv[1] + 1e-4));
    }
    const total = c.@"var" * var_loss + c.latent * mse + c.ce * ce_out + c.stop * stop_mean +
        c.mtp_weight * mtp_ce + (c.aux * aux_acc + c.zloss * z_acc) / @as(f32, @floatFromInt(m.L.len));
    try gpu.checkLaunch();
    return .{ .total = total, .ce = ce_out };
}

/// The backward pass of the window the forward just computed. With traces=1
/// it also advances the stream's traces, which only training does.
pub fn backwardWindow(m: *Model, s: *StreamState) !void {
    const c = m.c;
    const B: usize = @intCast(c.batch);
    const T: usize = @intCast(c.seqlen);
    const D: usize = @intCast(c.dim);
    const N = B * T;
    const ND = N * D;
    const k = m.kernels;
    const nd_i64: i64 = @intCast(ND);

    try (try k.get("mt_zero")).launch(@intCast(m.opt.nchunks), 256, .{m.opt}); // all gradients at once
    const logging = m.log_traces and m.trlog != null;
    if (logging) try gpu.zero(m.trlog.?, (2 * @as(usize, @intCast(c.layers)) + 256) * D * 4);

    // Rebuild the loss auxiliaries from the activations the forward left.
    var hmv = [2]f32{ 0, 1 };
    if (c.@"var" > 0) {
        try (try k.get("meanvar")).launch(1, 256, .{ m.X, m.mv, nd_i64 });
        try gpu.download(std.mem.sliceAsBytes(hmv[0..2]), m.mv);
    }
    if (c.latent > 0) try (try k.get("to_f32")).launch(blocks(ND), 256, .{ m.tgtX, m.dEnc, nd_i64 });

    try gpu.zero(m.dXres, ND * 4);
    try (try k.get("ce_bwd")).launch(blocks(N), 256, .{ m.probs, m.nxt, m.dlogits, c.ce, @as(i32, @intCast(N)) });
    try linalg.linearDW(@intCast(N), 256, c.dim, m.dlogits, m.X, m.store.at(m.dec).grad, 0);
    try linalg.linearDX(@intCast(N), 256, c.dim, m.dlogits, m.store.at(m.dec).work, m.dXs, 0);
    try (try k.get("cast_add")).launch(blocks(ND), 256, .{ m.dXs, m.dXres, nd_i64 });
    for (m.dec_mtp, 0..) |p, head| {
        const probs = m.probs_mtp + head * N * 256;
        const targets = m.nxt_mtp + head * N;
        const dlogits = m.logits_mtp + head * N * 256; // reused as the head's gradient
        try (try k.get("ce_bwd")).launch(blocks(N), 256,
            .{ probs, targets, dlogits, c.ce * c.mtp_weight, @as(i32, @intCast(N)) });
        try linalg.linearDW(@intCast(N), 256, c.dim, dlogits, m.X, m.store.at(p).grad, 0);
        try linalg.linearDX(@intCast(N), 256, c.dim, dlogits, m.store.at(p).work, m.dXs, 0);
        try (try k.get("cast_add")).launch(blocks(ND), 256, .{ m.dXs, m.dXres, nd_i64 });
    }
    if (m.dXext) |ext| try (try k.get("add_f32")).launch(blocks(ND), 256, .{ m.dXres, ext, nd_i64 });
    if (c.stop > 0) {
        try (try k.get("stop_bwd")).launch(blocks(N), 256,
            .{ m.stoplog, m.end, m.dstop, c.stopposw, c.stop, @as(i32, @intCast(N)) });
        try linalg.linearDW(@intCast(N), 1, c.dim, m.dstop, m.X, m.store.at(m.stop).grad, 0);
        try linalg.linearDX(@intCast(N), 1, c.dim, m.dstop, m.store.at(m.stop).work, m.dXs, 0);
        try (try k.get("cast_add")).launch(blocks(ND), 256, .{ m.dXs, m.dXres, nd_i64 });
    }
    if (c.latent > 0 or c.@"var" > 0)
        try (try k.get("to_f32")).launch(blocks(ND), 256, .{ m.X, m.dEncTmp, nd_i64 });
    if (c.latent > 0)
        try (try k.get("latent_bwd")).launch(blocks(ND), 256, .{ m.dEncTmp, m.dEnc, m.dXres, c.latent, nd_i64 });
    if (c.@"var" > 0) { // the hinge is active only below unit variance
        const scl: f32 = if (hmv[1] + 1e-4 >= 1.0) 0.0 else -0.5 / params.sqrtf(hmv[1] + 1e-4) * c.@"var";
        try (try k.get("var_bwd")).launch(blocks(ND), 256, .{ m.dEncTmp, m.dXres, hmv[0], scl, nd_i64 });
    }
    try (try k.get("copy_bf16")).launch(blocks(ND), 256, .{ m.dXres, m.dXs, nd_i64 });
    try gpu.zero(m.dEnc, ND * 4);
    if (c.mem != 0) try gpu.zero(m.MS.dEnc, B * @as(usize, @intCast(c.mem_len)) * D * 4);

    var Wx: [16][*]bf16 = undefined;
    var dWx: [16][*]f32 = undefined;
    var l: usize = m.L.len;
    while (l > 0) {
        l -= 1;
        const ly = &m.L[l];
        // The memory came last in the forward, so it goes first here.
        if (c.mem != 0 and m.MEM[l].use) {
            const me = &m.MEM[l];
            try memory.backward(k, m.dXs, me, &m.MS, m.store.at(me.gamma).master,
                m.store.at(me.wq).work, m.store.at(me.wk).work, m.store.at(me.wv).work, m.store.at(me.wo).work,
                m.store.at(me.gamma).grad, m.store.at(me.beta).grad, m.store.at(me.wq).grad,
                m.store.at(me.wk).grad, m.store.at(me.wv).grad, m.store.at(me.wo).grad,
                c.batch, c.seqlen, c.mem_len, c.mem_heads, c.mem_dh, c.dim);
        }
        if (c.mla != 0 and m.ML[l].use) {
            const ml = &m.ML[l];
            try gpu.copyDevice(m.dM, m.dXs, ND * 2);
            const sc = 1.0 / @sqrt(@as(f32, @floatFromInt(c.mla_dh)));
            try mla.backward(k, m.dM, @intCast(N), m.store.at(ml.p.q).work, m.store.at(ml.p.dkv).work,
                m.store.at(ml.p.kr).work, m.store.at(ml.p.uk).work, m.store.at(ml.p.uv).work,
                m.store.at(ml.p.o).work, m.store.at(ml.p.q).grad, m.store.at(ml.p.dkv).grad,
                m.store.at(ml.p.kr).grad, m.store.at(ml.p.uk).grad, m.store.at(ml.p.uv).grad,
                m.store.at(ml.p.o).grad, &s.cache[l], &ml.keep, &m.MW, m.dH, ml.keep.Xq, m.pos,
                c.batch, c.seqlen, c.mla_heads, c.mla_dh, c.mla_L, c.mla_R, c.mla_cc,
                c.mla_theta, sc, c.dim);
            try ops.layernormBwd(k, ml.Xsnap, m.dH, m.store.at(ml.p.gamma).master, ml.mmean, ml.mrstd,
                m.dSnorm, m.store.at(ml.p.gamma).grad, m.store.at(ml.p.beta).grad, @intCast(N), c.dim);
            try (try k.get("mem_add_bf16")).launch(blocks(ND), 256, .{ m.dXs, m.dSnorm, nd_i64 });
        }
        // The residual passthrough stays in dXs while the block works on a copy.
        try gpu.copyDevice(m.dM, m.dXs, ND * 2);
        const E: usize = @intCast(c.experts);
        for (0..E) |e| {
            Wx[e] = m.store.at(ly.exp[e]).work;
            dWx[e] = m.store.at(ly.exp[e]).grad;
        }
        const layers_f: f32 = @floatFromInt(c.layers);
        try moe.backward(k, m.dM, Wx[0..E], m.store.at(ly.router).work, m.store.at(ly.router).grad,
            dWx[0..E], m.dH, &ly.mc, &m.moeW, c.aux / layers_f, c.zloss / layers_f);
        try (try k.get("copy_bf16")).launch(blocks(ND), 256, .{ ly.S, m.Sb, nd_i64 });
        try ops.layernormBwd(k, m.Sb, m.dH, m.store.at(ly.gamma).master, ly.mean, ly.rstd, m.dSnorm,
            m.store.at(ly.gamma).grad, m.store.at(ly.beta).grad, @intCast(N), c.dim);
        const gate: u64 = if (c.gated != 0) @intFromPtr(m.store.at(ly.gate).master) else 0;
        var opt = ops.CellOpt{ .ids = @intFromPtr(m.ids), .docsep = c.docsep, .gamma = c.trace_decay };
        if (c.traces != 0) {
            opt.trDec = @intFromPtr(s.tdec[l]);
            opt.trGate = @intFromPtr(s.tgate[l]);
            if (l == 0) {
                opt.lam = @intFromPtr(m.lam);
                opt.prod = @intFromPtr(m.prod);
            }
            if (logging) {
                opt.logDec = @intFromPtr(m.trlog.? + l * D);
                opt.logGate = @intFromPtr(m.trlog.? + (@as(usize, @intCast(c.layers)) + l) * D);
                opt.logEmb = @intFromPtr(m.trlog.? + 2 * @as(usize, @intCast(c.layers)) * D);
            }
        }
        try ops.cellBackward(k, m.dSnorm, ly.S, m.store.at(ly.decay).master, m.dEncTmp,
            m.store.at(ly.decay).grad, c.batch, c.seqlen, c.dim, ly.input, @intFromPtr(ly.initial),
            gate, @intFromPtr(m.store.at(ly.gate).grad), opt);
        if (c.traces != 0 and l == 0)
            try ops.embTrace(k, ly.input, ly.S, ly.initial, m.store.at(ly.decay).master, gate,
                s.temb.?, m.store.at(m.emb).grad, c.batch, c.seqlen, c.dim, opt);
        // Chain into the previous layer, residual passthrough included.
        try (try k.get("cast_add")).launch(blocks(ND), 256, .{ m.dXs, m.dEncTmp, nd_i64 });
        try (try k.get("copy_bf16")).launch(blocks(ND), 256, .{ m.dEncTmp, m.dXs, nd_i64 });
    }
    if (c.mem != 0)
        try memory.encodeBackward(k, &m.MS, m.store.at(m.MS.decay).master, m.store.at(m.MS.gate).master,
            m.store.at(m.MS.decay).grad, m.store.at(m.MS.gate).grad, m.store.at(m.emb).grad,
            m.store.at(m.MS.eprev).grad, m.store.at(m.MS.pos).grad, c.batch, c.mem_len, c.dim);
    try (try k.get("cast_add")).launch(blocks(ND), 256, .{ m.dXs, m.dEnc, nd_i64 });
    try ops.embBackward(k, m.dEnc, m.ids, m.store.at(m.emb).grad, @intCast(N), c.dim);
    try gpu.checkLaunch();
}

/// AdamW over every parameter, with the gradient norm and the clipping
/// computed on the device; the only host synchronization is the norm itself.
pub fn optimizerStep(m: *Model, step: i32) !void {
    const c = m.c;
    const b1: f32 = 0.9;
    const b2: f32 = 0.999;
    const k = m.kernels;
    try gpu.zeroAsync(@as(*anyopaque, @ptrFromInt(m.opt.sumsq)), 8);
    try (try k.get("mt_sumsq")).launch(@intCast(m.opt.nchunks), 256, .{m.opt});
    try (try k.get("mt_adam")).launch(@intCast(m.opt.nchunks), 256, .{
        m.opt, c.gradclip, lrAt(c, step), b1, b2, @as(f32, 1e-8), @as(f32, 0.01),
        1.0 - params.powf(b1, @floatFromInt(step + 1)), 1.0 - params.powf(b2, @floatFromInt(step + 1)),
    });
    for (m.muon_params) |j| { // the same schedule shape, Muon's own size
        const lr = c.muon_lr * lrAt(c, step) / c.lr;
        try muon.step(k, &m.muon_ws, m.store.at(j), m.muon_x, lr, 0.01);
    }
    const tgt = m.store.at(m.tgt);
    const n = blocks(@intCast(tgt.n));
    try (try k.get("ema")).launch(n, 256, .{ tgt.master, m.store.at(m.emb).master, c.ematau, tgt.n });
    try (try k.get("copy_bf16")).launch(n, 256, .{ tgt.master, tgt.work, tgt.n });
    var sumsq: f64 = 0;
    try gpu.download(std.mem.asBytes(&sumsq), @as(*anyopaque, @ptrFromInt(m.opt.sumsq)));
    if (!std.math.isFinite(sumsq)) return error.NonFiniteGradient;
}
