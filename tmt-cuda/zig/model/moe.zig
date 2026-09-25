//! Mixture of experts: the per-layer state kept from forward to backward and
//! the shared workspace (port of the structures in src/moe.cu).
const std = @import("std");
const gpu = @import("gpu.zig");

pub const bf16 = u16;

/// How many rows one expert may be padded to for the batched matmul: twice
/// the balanced share is as much padding as the batched form can ever repay.
pub fn uniformRows(N: usize, E: usize, K: usize) usize {
    return ((2 * N * K / E + 127) / 128) * 128;
}

/// One batched matmul over all experts is about 1.7 times faster than the same
/// work issued per expert, but it has to pad every expert to the longest one.
/// It therefore only pays while the router is reasonably balanced, which is
/// what this decides, per window.
fn batchWorthIt(k: *const Keep, E: usize, pm: i32, capacity: usize) bool {
    if (pm <= 0 or @as(usize, @intCast(pm)) > capacity) return false;
    var own: i64 = 0;
    for (0..E) |e| own += @divTrunc(k.hcnt[e] + 127, 128) * 128;
    const padded: i64 = @as(i64, pm) * @as(i64, @intCast(E));
    return padded * 4 <= own * 5; // at most a quarter more work than the loop
}

/// Slots the dispatch buffers must hold: enough for every expert to be padded
/// alike, which is what lets one batched matmul serve them all.
pub fn slotCapacity(N: usize, E: usize, K: usize) usize {
    return @max(N * K + E * 128, E * uniformRows(N, E, K));
}

/// Kept per layer between forward and backward.
pub const Keep = struct {
    zloss: f32 = 0,
    logits: [*]f32 = undefined, // (N,E)
    probs: [*]f32 = undefined, // (N,E)
    idx: [*]i32 = undefined, // (N,K)
    slotw: [*]f32 = undefined, // (Tk)
    slot_of: [*]i32 = undefined, // (N,K)
    counts: [*]i32 = undefined, // (E) device copy for the router gradient
    perm: [*]i32 = undefined, // (Tk)
    H: [*]bf16 = undefined, // (N,D) the MoE input
    Yg: [*]bf16 = undefined, // (Tk,D) expert pre-activations
    hcnt: [16]i32 = @splat(0), // host: tokens per expert
    hoff: [16]i32 = @splat(0), // host: offsets
    phoff: [16]i32 = @splat(0), // host: offsets including padding
    N: i32 = 0,
    E: i32 = 0,
    K: i32 = 0,
    D: i32 = 0,
    Tk: i32 = 0,
    Ptk: i32 = 0,
    /// Router statistics slot owned by the model, read for all layers at once.
    stat: ?[*]f32 = null,

    pub fn alloc(k: *Keep, mem: *gpu.Memory, N: usize, E: usize, K: usize, D: usize) !void {
        const Tk = slotCapacity(N, E, K);
        k.logits = try mem.allocT(f32, N * E);
        k.probs = try mem.allocT(f32, N * E);
        k.idx = try mem.allocT(i32, N * K);
        k.slotw = try mem.allocT(f32, Tk);
        k.slot_of = try mem.allocT(i32, N * K);
        k.counts = try mem.allocT(i32, E);
        k.perm = try mem.allocT(i32, Tk);
        k.H = try mem.allocT(bf16, N * D);
        k.Yg = try mem.allocT(bf16, Tk * D);
    }
};

/// Shared transient buffers; the layers run one after the other.
pub const Ws = struct {
    /// Device arrays of pointers for the batched expert matmuls: the rows of
    /// each expert, its output, and the gradients of both.
    pxg: u64 = 0,
    pyg: u64 = 0,
    pdyg: u64 = 0,
    pdxg: u64 = 0,
    /// The row count each expert is padded to; the pointer arrays describe
    /// this padding, so they are rebuilt when it changes.
    padded: usize = 0,
    /// How many rows per expert the buffers can hold.
    capacity: usize = 0,
    w: [*]f32 = undefined, // (N,K) routing weights
    offsets: [*]i32 = undefined, // (E)
    cursor: [*]i32 = undefined, // (E)
    sum_p: [*]f32 = undefined, // (E+1)
    Xg: [*]bf16 = undefined, // (Tk,D) gathered rows
    dYg: [*]bf16 = undefined,
    s_j: [*]f32 = undefined, // (N,K)
    dlogits: [*]f32 = undefined, // (N,E)
    dlog_b: [*]bf16 = undefined,
    dXg: [*]bf16 = undefined,
    Xf: [*]f32 = undefined, // (N,D) fp32 copy of H for the router weight gradient

    pub fn alloc(w: *Ws, mem: *gpu.Memory, N: usize, E: usize, K: usize, D: usize) !void {
        // Room for every expert to be padded to the same length, up to twice
        // the balanced share. Beyond that the dispatch falls back to one
        // matmul per expert rather than padding the whole window away.
        w.capacity = uniformRows(N, E, K);
        const Tk = slotCapacity(N, E, K);
        w.w = try mem.allocT(f32, N * K);
        w.offsets = try mem.allocT(i32, E);
        w.cursor = try mem.allocT(i32, E);
        w.sum_p = try mem.allocT(f32, E + 1);
        w.Xg = try mem.allocT(bf16, Tk * D);
        w.dYg = try mem.allocT(bf16, Tk * D);
        w.s_j = try mem.allocT(f32, N * K);
        w.dlogits = try mem.allocT(f32, N * E);
        w.dlog_b = try mem.allocT(bf16, N * E);
        w.dXg = try mem.allocT(bf16, Tk * D);
        w.Xf = try mem.allocT(f32, N * D);
        w.pxg = @intFromPtr(try mem.allocT(u64, E));
        w.pyg = @intFromPtr(try mem.allocT(u64, E));
        w.pdyg = @intFromPtr(try mem.allocT(u64, E));
        w.pdxg = @intFromPtr(try mem.allocT(u64, E));
    }

    /// Points the arrays at the slices of this window's padding.
    fn setPointers(w: *Ws, gpa: std.mem.Allocator, E: usize, D: usize, pm: usize, k: *const Keep) !void {
        if (w.padded == pm) return;
        const host = try gpa.alloc(u64, E * 4);
        defer gpa.free(host);
        for (0..E) |e| {
            const at = e * pm * D;
            host[e] = @intFromPtr(w.Xg + at);
            host[E + e] = @intFromPtr(k.Yg + at);
            host[2 * E + e] = @intFromPtr(w.dYg + at);
            host[3 * E + e] = @intFromPtr(w.dXg + at);
        }
        try gpu.upload(@as(*anyopaque, @ptrFromInt(w.pxg)), std.mem.sliceAsBytes(host[0..E]));
        try gpu.upload(@as(*anyopaque, @ptrFromInt(w.pyg)), std.mem.sliceAsBytes(host[E .. 2 * E]));
        try gpu.upload(@as(*anyopaque, @ptrFromInt(w.pdyg)), std.mem.sliceAsBytes(host[2 * E .. 3 * E]));
        try gpu.upload(@as(*anyopaque, @ptrFromInt(w.pdxg)), std.mem.sliceAsBytes(host[3 * E ..]));
        w.padded = pm;
    }
};

const linalg = @import("linalg.zig");

/// Auxiliary load-balancing loss from the statistics the forward deferred.
pub fn auxFrom(k: *Keep, hsum: []const f32) f32 {
    k.zloss = hsum[@intCast(k.E)];
    var aux: f32 = 0;
    for (0..@intCast(k.E)) |e|
        aux += (hsum[e] / @as(f32, @floatFromInt(k.N))) *
            (@as(f32, @floatFromInt(k.hcnt[e])) / @as(f32, @floatFromInt(k.N * k.K)));
    return @as(f32, @floatFromInt(k.E)) * aux;
}

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

/// Top-k dispatch, the expert matmuls and the weighted combine. Returns the
/// auxiliary loss unless the caller collects the statistics itself.
pub fn forward(gpa: std.mem.Allocator, kern: *gpu.Kernels, X: [*]const bf16, Wrouter: [*]const bf16,
               Wexp: []const [*]bf16, wptr: u64, Y: [*]bf16, k: *Keep, w: *Ws,
               N: usize, E: usize, K: usize, D: usize, beta: f32,
               noise: f32, seed: u32) !f32 {
    k.N = @intCast(N);
    k.E = @intCast(E);
    k.K = @intCast(K);
    k.D = @intCast(D);
    const n = N * D;
    try gpu.copyDevice(k.H, X, n * 2);
    if (E == 1) {
        try linalg.linearFwd(@intCast(N), @intCast(D), @intCast(D), X, Wexp[0], k.Yg);
        try (try kern.get("dense_activation")).launch(blocks(n), 256,
            .{ k.Yg, @as(u64, 0), Y, @as(i64, @intCast(n)), beta });
        k.Tk = @intCast(N);
        return 0;
    }
    // Eight lanes per expert; sixty-four threads only ever covered eight.
    try (try kern.get("router_topk")).launch(@intCast(N), @intCast(E * 8),
        .{ X, Wrouter, k.logits, k.probs, k.idx, w.w, @as(i32, @intCast(N)), @as(i32, @intCast(E)),
           @as(i32, @intCast(K)), @as(i32, @intCast(D)), noise, seed });
    try gpu.zero(k.counts, E * 4);
    try (try kern.get("count_experts")).launch(blocks(N), 256,
        .{ k.idx, k.counts, @as(i32, @intCast(N)), @as(i32, @intCast(K)) });
    // The offsets are computed on the host; E is small, one synchronization.
    try gpu.download(std.mem.sliceAsBytes(k.hcnt[0..E]), k.counts);
    var tk: i32 = 0;
    var ptk: i32 = 0;
    for (0..E) |e| {
        k.hoff[e] = tk;
        tk += k.hcnt[e];
    }
    // Padded to blocks of 128, which keeps the GEMM shapes stable. When every
    // expert fits the same padded length, they are laid out alike and one
    // batched matmul serves all of them.
    var pm: i32 = 0;
    for (0..E) |e| pm = @max(pm, @divTrunc(k.hcnt[e] + 127, 128) * 128);
    const uniform = batchWorthIt(k, E, pm, w.capacity);
    for (0..E) |e| {
        k.phoff[e] = if (uniform) @as(i32, @intCast(e)) * pm else ptk;
        ptk += @divTrunc(k.hcnt[e] + 127, 128) * 128;
    }
    if (uniform) ptk = @as(i32, @intCast(E)) * pm;
    k.Tk = tk;
    k.Ptk = ptk;
    try gpu.upload(w.cursor, std.mem.sliceAsBytes(k.phoff[0..E]));
    try (try kern.get("fill_slots")).launch(blocks(N), 256,
        .{ k.idx, w.w, w.cursor, k.perm, k.slotw, k.slot_of, @as(i32, @intCast(N)), @as(i32, @intCast(K)) });
    try gpu.zero(w.Xg, @as(usize, @intCast(@max(ptk, 1))) * D * 2);
    if (tk > 0) {
        try (try kern.get("gather_slot")).launch(blocks(N * K * D), 256,
            .{ X, k.slot_of, w.Xg, @as(i32, @intCast(N)), @as(i32, @intCast(K)), @as(i32, @intCast(D)) });
        if (uniform) {
            try w.setPointers(gpa, E, D, @intCast(pm), k);
            try linalg.linearFwdBatched(pm, @intCast(D), @intCast(D), w.pxg, wptr, w.pyg, @intCast(E));
        } else for (0..E) |e| {
            const rows = @divTrunc(k.hcnt[e] + 127, 128) * 128;
            if (rows == 0) continue;
            const off: usize = @as(usize, @intCast(k.phoff[e])) * D;
            try linalg.linearFwd(rows, @intCast(D), @intCast(D), w.Xg + off, Wexp[e], k.Yg + off);
        }
    }
    try (try kern.get("combine")).launch(blocks(N * D), 256,
        .{ k.Yg, k.slotw, k.slot_of, Y, @as(i32, @intCast(N)), @as(i32, @intCast(K)),
           @as(i32, @intCast(D)), beta });
    try (try kern.get("aux_sum")).launch(@intCast(E), 256, .{ k.probs, w.sum_p, @as(i32, @intCast(N)), @as(i32, @intCast(E)) });
    try gpu.zero(w.sum_p + E, 4);
    try (try kern.get("zloss")).launch(blocks(N), 256,
        .{ k.logits, w.sum_p + E, @as(i32, @intCast(N)), @as(i32, @intCast(E)) });
    if (k.stat) |stat| { // deferred: the owner reads every layer at once
        try gpu.copyDevice(stat, w.sum_p, (E + 1) * 4);
        k.zloss = 0;
        return 0;
    }
    var hsum: [17]f32 = undefined;
    try gpu.download(std.mem.sliceAsBytes(hsum[0 .. E + 1]), w.sum_p);
    return auxFrom(k, hsum[0 .. E + 1]);
}

/// Backward through combine, the experts and the router, including the
/// analytic gradients of the balancing terms.
pub fn backward(gpa: std.mem.Allocator, kern: *gpu.Kernels, dY: [*]const bf16, Wexp: []const [*]bf16,
                wptr: u64, Wrouter: [*]const bf16, dWrouter: [*]f32, dWexp: []const [*]f32,
                dwptr: u64, dX: [*]bf16, k: *Keep, w: *Ws, aux: f32, zcoef: f32) !void {
    const N: usize = @intCast(k.N);
    const E: usize = @intCast(k.E);
    const K: usize = @intCast(k.K);
    const D: usize = @intCast(k.D);
    const n = N * D;
    try gpu.zero(dX, n * 2);
    try gpu.zero(dWrouter, E * D * 4);
    if (E == 1) {
        try (try kern.get("dense_activation")).launch(blocks(n), 256,
            .{ k.Yg, dY, w.dXg, @as(i64, @intCast(n)), @as(f32, 0) });
        try linalg.linearDW(@intCast(N), @intCast(D), @intCast(D), w.dXg, k.H, dWexp[0], 0);
        try linalg.linearDX(@intCast(N), @intCast(D), @intCast(D), w.dXg, Wexp[0], dX, 0);
        return;
    }
    if (k.Tk == 0) return;
    // Recompute the gathered inputs from the kept layer input.
    const padded: usize = @intCast(@max(k.Ptk, 1));
    try gpu.zero(w.Xg, padded * D * 2);
    try (try kern.get("gather_slot")).launch(blocks(N * K * D), 256,
        .{ k.H, k.slot_of, w.Xg, @as(i32, @intCast(N)), @as(i32, @intCast(K)), @as(i32, @intCast(D)) });
    // The padded expert-output gradient is zeroed before combine fills it.
    try gpu.zero(w.dYg, padded * D * 2);
    try (try kern.get("combine_bwd")).launch(blocks(N * K * D), 256,
        .{ dY, k.Yg, k.slotw, k.slot_of, w.dYg, @as(u64, 0), @as(i32, @intCast(N)),
           @as(i32, @intCast(K)), @as(i32, @intCast(D)) });
    try (try kern.get("sdot")).launch(@intCast(N * K), 256,
        .{ dY, k.Yg, k.slot_of, w.s_j, @as(i32, @intCast(N)), @as(i32, @intCast(K)), @as(i32, @intCast(D)) });
    try (try kern.get("router_bwd")).launch(blocks(N), 256,
        .{ k.probs, k.idx, w.s_j, w.dlogits, @as(i32, @intCast(N)), @as(i32, @intCast(E)),
           @as(i32, @intCast(K)), k.logits, k.counts, aux, zcoef });
    try (try kern.get("copy_bf16")).launch(blocks(N * E), 256, .{ w.dlogits, w.dlog_b, @as(i64, @intCast(N * E)) });
    try linalg.linearDW(@intCast(N), @intCast(E), @intCast(D), w.dlog_b, k.H, dWrouter, 0);
    // The router's input gradient accumulates into dX.
    try linalg.linearDX(@intCast(N), @intCast(E), @intCast(D), w.dlog_b, Wrouter, dX, 1);
    // The same padding as in the forward, so the same batched shapes apply.
    var pm: i32 = 0;
    for (0..E) |e| pm = @max(pm, @divTrunc(k.hcnt[e] + 127, 128) * 128);
    const uniform = batchWorthIt(k, E, pm, w.capacity) and w.padded == @as(usize, @intCast(pm));
    if (uniform) {
        try linalg.linearDWBatched(pm, @intCast(D), @intCast(D), w.pdyg, w.pxg, dwptr, @intCast(E));
        try linalg.linearDXBatched(pm, @intCast(D), @intCast(D), w.pdyg, wptr, w.pdxg, @intCast(E));
    } else for (0..E) |e| {
        const rows = @divTrunc(k.hcnt[e] + 127, 128) * 128;
        if (rows == 0) continue;
        const off: usize = @as(usize, @intCast(k.phoff[e])) * D;
        try linalg.linearDW(rows, @intCast(D), @intCast(D), w.dYg + off, w.Xg + off, dWexp[e], 0);
        try linalg.linearDX(rows, @intCast(D), @intCast(D), w.dYg + off, Wexp[e], w.dXg + off, 0);
    }
    _ = gpa;
    try (try kern.get("scatter_add")).launch(blocks(N * D), 256,
        .{ w.dXg, k.slot_of, dX, @as(i32, @intCast(N)), @as(i32, @intCast(K)), @as(i32, @intCast(D)) });
}
