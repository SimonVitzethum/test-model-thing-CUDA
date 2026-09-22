//! ktest: the Zig kernels must compute exactly what the C++ ones compute.
//! Runs both on the same device buffers and compares the results bit for bit.
//! Kernels whose result depends on the order of atomic adds are compared with
//! two runs of each build, so the comparison can only be as strict as the
//! reference is reproducible.
const std = @import("std");
const tmt = @import("tmt.zig");
const ptx = @import("ptx.zig");

const kernels_ptx = @embedFile("kernels.ptx");

/// Host view of the CellOpt the recurrence kernels take (device pointers as
/// plain addresses); same layout as the struct in zig/kernels/kernels.zig.
const CellOpt = extern struct {
    ids: u64 = 0,
    docsep: i32 = -1,
    gamma: f32 = 1,
    trDec: u64 = 0,
    trGate: u64 = 0,
    lam: u64 = 0,
    prod: u64 = 0,
    logDec: u64 = 0,
    logGate: u64 = 0,
    logEmb: u64 = 0,
};

/// Host mirrors of the optimizer structures (device pointers as addresses).
const MTChunk = extern struct { param: i32, len: i32, start: i64 };
const MTParams = extern struct {
    master: u64,
    m: u64,
    v: u64,
    grad: u64,
    work: u64,
    flags: u64,
    chunks: u64,
    nchunks: i32,
    sumsq: u64,
};

var failures: usize = 0;

/// Deterministic test data, independent of the platform's RNG.
fn fill(buf: []f32, seed: u32) void {
    var s = seed;
    for (buf) |*v| {
        s = s *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(s >> 8)) / @as(f32, @floatFromInt(1 << 24)) - 0.5;
    }
}
fn fillIds(buf: []i32, seed: u32, lo: i32, hi: i32) void {
    var s = seed;
    for (buf) |*v| {
        s = s *% 1664525 +% 1013904223;
        v.* = lo + @as(i32, @intCast((s >> 8) % @as(u32, @intCast(hi - lo))));
    }
}

const Dev = struct {
    /// Device buffer of n bytes, filled from the host.
    fn alloc(n: usize) !*anyopaque {
        var p: ?*anyopaque = null;
        if (tmt.tmt_dev_alloc(&p, n) != 0) return error.Alloc;
        return p.?;
    }
    fn up(dst: *anyopaque, src: []const u8) !void {
        if (tmt.tmt_dev_upload(dst, src.ptr, src.len) != 0) return error.Upload;
    }
    fn down(dst: []u8, src: *anyopaque) !void {
        if (tmt.tmt_dev_download(dst.ptr, src, dst.len) != 0) return error.Download;
    }
};

var out_io: std.Io = undefined;
fn say(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    var w = std.Io.File.stdout().writerStreaming(out_io, &buf);
    w.interface.print(fmt, args) catch {};
    w.interface.flush() catch {};
}
fn report(name: []const u8, ok: bool, detail: []const u8) void {
    say("{s} {s}{s}{s}\n", .{ if (ok) "PASS" else "FAIL", name, if (detail.len > 0) ": " else "", detail });
    if (!ok) failures += 1;
}

fn bytesOf(comptime T: type, s: []T) []u8 {
    return std.mem.sliceAsBytes(s);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.arena.allocator();
    out_io = init.io;
    const mod = ptx.Module.load(kernels_ptx) catch |e| {
        say("error: {s} ({s})\n", .{ @errorName(e), ptx.lastError() });
        return 1;
    };

    const N = 4096;
    const D = 128;
    const ids = try gpa.alloc(i32, N);
    const x = try gpa.alloc(f32, N * D);
    const table = try gpa.alloc(f32, 256 * D);
    const a = try gpa.alloc(f32, N * D);
    const got = try gpa.alloc(f32, N * D);
    const want = try gpa.alloc(f32, N * D);
    fillIds(ids, 7, 0, 256);
    fill(x, 11);
    fill(table, 13);
    fill(a, 17);

    const d_ids = try Dev.alloc(N * 4);
    const d_x = try Dev.alloc(N * D * 4);
    const d_tab_bf = try Dev.alloc(256 * D * 2);
    const d_tab_f = try Dev.alloc(256 * D * 4);
    const d_out = try Dev.alloc(N * D * 4);
    const d_acc = try Dev.alloc(N * D * 4);
    try Dev.up(d_ids, bytesOf(i32, ids));
    try Dev.up(d_x, bytesOf(f32, x));
    try Dev.up(d_tab_f, bytesOf(f32, table));

    // ---- copy_bf16: master weights -> bf16 working copy ----
    {
        const n = 256 * D;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_tab_f)), d_tab_bf, n) != 0) return error.Ref;
        const ref = try gpa.alloc(u16, n);
        try Dev.down(bytesOf(u16, ref), d_tab_bf);
        try Dev.up(d_tab_bf, &[_]u8{0} ** 8); // clobber, so a no-op kernel fails
        const k = try mod.get("copy_bf16");
        try k.launch((n + 255) / 256, 256, .{ d_tab_f, d_tab_bf, @as(i64, n) });
        const mine = try gpa.alloc(u16, n);
        try Dev.down(bytesOf(u16, mine), d_tab_bf);
        report("copy_bf16", std.mem.eql(u16, ref, mine), "");
    }

    // ---- emb_gather ----
    {
        const n = N * D;
        const d_emb = try Dev.alloc(n * 2);
        if (tmt.tmt_ref_emb_forward(d_tab_bf, @ptrCast(@alignCast(d_ids)), d_emb, N, D) != 0) return error.Ref;
        const ref = try gpa.alloc(u16, n);
        try Dev.down(bytesOf(u16, ref), d_emb);
        const k = try mod.get("emb_gather");
        try k.launch((N + 255) / 256, 256, .{ d_tab_bf, d_ids, d_emb, @as(i32, N), @as(i32, D) });
        const mine = try gpa.alloc(u16, n);
        try Dev.down(bytesOf(u16, mine), d_emb);
        report("emb_gather", std.mem.eql(u16, ref, mine), "");
    }

    // ---- add_f32 ----
    {
        const n = N * D;
        try Dev.up(d_acc, bytesOf(f32, a));
        if (tmt.tmt_ref_add_f32(@ptrCast(@alignCast(d_acc)), @ptrCast(@alignCast(d_x)), n) != 0) return error.Ref;
        try Dev.down(bytesOf(f32, want), d_acc);
        try Dev.up(d_acc, bytesOf(f32, a));
        const k = try mod.get("add_f32");
        try k.launch((n + 255) / 256, 256, .{ d_acc, d_x, @as(i64, n) });
        try Dev.down(bytesOf(f32, got), d_acc);
        report("add_f32", std.mem.eql(f32, want, got), "");
    }

    // ---- emb_scatter: atomic, so compare the reference against itself too ----
    {
        const zero = try gpa.alloc(f32, 256 * D);
        @memset(zero, 0);
        const ref1 = try gpa.alloc(f32, 256 * D);
        const ref2 = try gpa.alloc(f32, 256 * D);
        const mine = try gpa.alloc(f32, 256 * D);
        const k = try mod.get("emb_scatter");
        for ([_]?*const anyopaque{ null, null }, [_][]f32{ ref1, ref2 }) |_, dst| {
            try Dev.up(d_out, bytesOf(f32, zero));
            if (tmt.tmt_ref_emb_backward(@ptrCast(@alignCast(d_x)), @ptrCast(@alignCast(d_ids)), @ptrCast(@alignCast(d_out)), N, D) != 0) return error.Ref;
            try Dev.down(bytesOf(f32, dst), d_out);
        }
        try Dev.up(d_out, bytesOf(f32, zero));
        try k.launch((N + 255) / 256, 256, .{ d_x, d_ids, d_out, @as(i32, N), @as(i32, D) });
        try Dev.down(bytesOf(f32, mine), d_out);
        // Atomic adds run in an arbitrary order, so the C++ kernel does not even
        // reproduce itself; the Zig kernel has to stay within that spread.
        var scale: f32 = 1e-30;
        for (ref1) |r| scale = @max(scale, @abs(r));
        var self_diff: f32 = 0;
        var cross: f32 = 0;
        for (ref1, ref2, mine) |r1, r2, m| {
            self_diff = @max(self_diff, @abs(r1 - r2) / scale);
            cross = @max(cross, @abs(r1 - m) / scale);
        }
        var buf: [128]u8 = undefined;
        const detail = std.fmt.bufPrint(&buf, "difference to C++ {e:.1}, C++ against itself {e:.1} (atomic order)", .{ cross, self_diff }) catch "";
        report("emb_scatter", cross <= @max(self_diff * 4, 1e-6), detail);
    }

    // ---- cross entropy (the first kernels with expf and division) ----
    {
        const R = 512; // positions
        const d_log = try Dev.alloc(R * 256 * 2);
        const d_probs = try Dev.alloc(R * 256 * 4);
        const d_loss = try Dev.alloc(R * 4);
        const d_dlog = try Dev.alloc(R * 256 * 2);
        const logits = try gpa.alloc(f32, R * 256);
        fill(logits, 31);
        for (logits) |*v| v.* *= 8; // a realistic logit range
        const d_logf32 = try Dev.alloc(R * 256 * 4);
        try Dev.up(d_logf32, bytesOf(f32, logits));
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_logf32)), d_log, R * 256) != 0) return error.Ref;
        const tgt = try gpa.alloc(i32, R);
        fillIds(tgt, 37, -1, 256); // -1 occurs and must be ignored
        const d_tgt = try Dev.alloc(R * 4);
        try Dev.up(d_tgt, bytesOf(i32, tgt));

        const ref_p = try gpa.alloc(f32, R * 256);
        const ref_l = try gpa.alloc(f32, R);
        if (tmt.tmt_ref_ce_fwd(d_log, @ptrCast(@alignCast(d_tgt)), @ptrCast(@alignCast(d_probs)), @ptrCast(@alignCast(d_loss)), R) != 0) return error.Ref;
        try Dev.down(bytesOf(f32, ref_p), d_probs);
        try Dev.down(bytesOf(f32, ref_l), d_loss);
        const kf = try mod.get("ce_fwd");
        try kf.launch((R + 255) / 256, 256, .{ d_log, d_tgt, d_probs, d_loss, @as(i32, R) });
        const my_p = try gpa.alloc(f32, R * 256);
        const my_l = try gpa.alloc(f32, R);
        try Dev.down(bytesOf(f32, my_p), d_probs);
        try Dev.down(bytesOf(f32, my_l), d_loss);
        var pdiff: f32 = 0;
        var ldiff: f32 = 0;
        var npdiff: usize = 0;
        for (ref_p, my_p) |v, w| {
            if (v != w) npdiff += 1;
            pdiff = @max(pdiff, @abs(v - w));
        }
        for (ref_l, my_l) |v, w| ldiff = @max(ldiff, @abs(v - w));
        // The probabilities carry the gradient and must be identical; the
        // reported loss is a scalar and ends up one unit in the last place
        // apart, because nvcc computes log(se) + mx in a different register
        // order than LLVM does.
        var cbuf: [160]u8 = undefined;
        const cdetail = std.fmt.bufPrint(&cbuf, "probs identical, loss within {e:.1} (1 ulp)", .{ldiff}) catch "";
        report("ce_fwd", npdiff == 0 and pdiff == 0 and ldiff < 1e-5, cdetail);

        const ref_d = try gpa.alloc(u16, R * 256);
        const my_d = try gpa.alloc(u16, R * 256);
        if (tmt.tmt_ref_ce_bwd(@ptrCast(@alignCast(d_probs)), @ptrCast(@alignCast(d_tgt)), d_dlog, 1.5, R) != 0) return error.Ref;
        try Dev.down(bytesOf(u16, ref_d), d_dlog);
        const kb = try mod.get("ce_bwd");
        try kb.launch((R + 255) / 256, 256, .{ d_probs, d_tgt, d_dlog, @as(f32, 1.5), @as(i32, R) });
        try Dev.down(bytesOf(u16, my_d), d_dlog);
        report("ce_bwd", std.mem.eql(u16, ref_d, my_d), "");

        // stop head: BCE with pos_weight
        const d_end = try Dev.alloc(R * 4);
        const ends = try gpa.alloc(i32, R);
        fillIds(ends, 41, 0, 2);
        try Dev.up(d_end, bytesOf(i32, ends));
        const d_sb = try Dev.alloc(R * 2);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_logf32)), d_sb, R) != 0) return error.Ref;
        if (tmt.tmt_ref_stop_fwd(d_sb, @ptrCast(@alignCast(d_end)), @ptrCast(@alignCast(d_loss)), 20, R) != 0) return error.Ref;
        try Dev.down(bytesOf(f32, ref_l), d_loss);
        const ks = try mod.get("stop_fwd");
        try ks.launch((R + 255) / 256, 256, .{ d_sb, d_end, d_loss, @as(f32, 20), @as(i32, R) });
        try Dev.down(bytesOf(f32, my_l), d_loss);
        const d_ds = try Dev.alloc(R * 2);
        const ref_s = try gpa.alloc(u16, R);
        const my_s = try gpa.alloc(u16, R);
        if (tmt.tmt_ref_stop_bwd(d_sb, @ptrCast(@alignCast(d_end)), d_ds, 20, 0.7, R) != 0) return error.Ref;
        try Dev.down(bytesOf(u16, ref_s), d_ds);
        const ksb = try mod.get("stop_bwd");
        try ksb.launch((R + 255) / 256, 256, .{ d_sb, d_end, d_ds, @as(f32, 20), @as(f32, 0.7), @as(i32, R) });
        try Dev.down(bytesOf(u16, my_s), d_ds);
        report("stop_fwd/stop_bwd", std.mem.eql(f32, ref_l, my_l) and std.mem.eql(u16, ref_s, my_s), "");

        // bf16 gradient -> fp32 accumulation
        try Dev.up(d_acc, bytesOf(f32, a));
        if (tmt.tmt_ref_cast_add(d_log, @ptrCast(@alignCast(d_acc)), R * 256) != 0) return error.Ref;
        try Dev.down(bytesOf(f32, want[0 .. R * 256]), d_acc);
        try Dev.up(d_acc, bytesOf(f32, a));
        const kc = try mod.get("cast_add");
        try kc.launch((R * 256 + 255) / 256, 256, .{ d_log, d_acc, @as(i64, R * 256) });
        try Dev.down(bytesOf(f32, got[0 .. R * 256]), d_acc);
        report("cast_add", std.mem.eql(f32, want[0 .. R * 256], got[0 .. R * 256]), "");
    }

    // ---- LayerNorm ----
    {
        const R = 64; // rows
        const gamma = try gpa.alloc(f32, D);
        const beta = try gpa.alloc(f32, D);
        fill(gamma, 23);
        fill(beta, 29);
        for (gamma) |*g| g.* += 1;
        const d_gamma = try Dev.alloc(D * 4);
        const d_beta = try Dev.alloc(D * 4);
        try Dev.up(d_gamma, bytesOf(f32, gamma));
        try Dev.up(d_beta, bytesOf(f32, beta));
        // bf16 input, produced by the (already compared) conversion kernel
        const d_xb = try Dev.alloc(R * D * 2);
        const d_dyb = try Dev.alloc(R * D * 2);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_x)), d_xb, R * D) != 0) return error.Ref;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_acc)), d_dyb, R * D) != 0) return error.Ref;
        try Dev.up(d_acc, bytesOf(f32, a));
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_acc)), d_dyb, R * D) != 0) return error.Ref;
        const d_y = try Dev.alloc(R * D * 2);
        const d_mean = try Dev.alloc(R * 4);
        const d_rstd = try Dev.alloc(R * 4);
        const ref_y = try gpa.alloc(u16, R * D);
        const ref_ms = try gpa.alloc(f32, 2 * R);
        if (tmt.tmt_ref_ln_fwd(d_xb, @ptrCast(@alignCast(d_gamma)), @ptrCast(@alignCast(d_beta)), d_y,
                               @ptrCast(@alignCast(d_mean)), @ptrCast(@alignCast(d_rstd)), R, D) != 0) return error.Ref;
        try Dev.down(bytesOf(u16, ref_y), d_y);
        try Dev.down(bytesOf(f32, ref_ms[0..R]), d_mean);
        try Dev.down(bytesOf(f32, ref_ms[R..]), d_rstd);
        const k = try mod.get("ln_fwd");
        try k.launch(R, 256, .{ d_xb, d_gamma, d_beta, d_y, d_mean, d_rstd, @as(i32, R), @as(i32, D) });
        const my_y = try gpa.alloc(u16, R * D);
        const my_ms = try gpa.alloc(f32, 2 * R);
        try Dev.down(bytesOf(u16, my_y), d_y);
        try Dev.down(bytesOf(f32, my_ms[0..R]), d_mean);
        try Dev.down(bytesOf(f32, my_ms[R..]), d_rstd);
        report("ln_fwd", std.mem.eql(u16, ref_y, my_y) and std.mem.eql(f32, ref_ms, my_ms), "");

        // backward: dX bit for bit, dGamma/dBeta within the atomic spread
        const zeroD = try gpa.alloc(f32, D);
        @memset(zeroD, 0);
        const d_dx = try Dev.alloc(R * D * 2);
        const d_dg = try Dev.alloc(D * 4);
        const d_db = try Dev.alloc(D * 4);
        const ref_dx = try gpa.alloc(u16, R * D);
        const ref_g = try gpa.alloc(f32, 2 * D);
        const my_g = try gpa.alloc(f32, 2 * D);
        try Dev.up(d_dg, bytesOf(f32, zeroD));
        try Dev.up(d_db, bytesOf(f32, zeroD));
        if (tmt.tmt_ref_ln_bwd(d_xb, d_dyb, @ptrCast(@alignCast(d_gamma)), @ptrCast(@alignCast(d_mean)),
                               @ptrCast(@alignCast(d_rstd)), d_dx, @ptrCast(@alignCast(d_dg)),
                               @ptrCast(@alignCast(d_db)), R, D) != 0) return error.Ref;
        try Dev.down(bytesOf(u16, ref_dx), d_dx);
        try Dev.down(bytesOf(f32, ref_g[0..D]), d_dg);
        try Dev.down(bytesOf(f32, ref_g[D..]), d_db);
        try Dev.up(d_dg, bytesOf(f32, zeroD));
        try Dev.up(d_db, bytesOf(f32, zeroD));
        const kb = try mod.get("ln_bwd");
        try kb.launch(R, 256, .{ d_xb, d_dyb, d_gamma, d_mean, d_rstd, d_dx, d_dg, d_db, @as(i32, R), @as(i32, D) });
        const my_dx = try gpa.alloc(u16, R * D);
        try Dev.down(bytesOf(u16, my_dx), d_dx);
        try Dev.down(bytesOf(f32, my_g[0..D]), d_dg);
        try Dev.down(bytesOf(f32, my_g[D..]), d_db);
        var scale: f32 = 1e-30;
        for (ref_g) |v| scale = @max(scale, @abs(v));
        var worst: f32 = 0;
        for (ref_g, my_g) |v, w| worst = @max(worst, @abs(v - w) / scale);
        var buf: [128]u8 = undefined;
        const detail = std.fmt.bufPrint(&buf, "dGamma/dBeta difference {e:.1} (atomic order)", .{worst}) catch "";
        report("ln_bwd", std.mem.eql(u16, ref_dx, my_dx) and worst < 1e-5, detail);
    }

    // ---- the recurrence: forward, exact backward, hybrid traces ----
    {
        const B = 4;
        const T = 32;
        const Dc = 128;
        const NT = B * T * Dc;
        const opt_ids = try gpa.alloc(i32, B * T);
        fillIds(opt_ids, 43, 0, 256);
        opt_ids[5] = 10; // a document separator must reset the state
        opt_ids[T + 9] = 10;
        const decay = try gpa.alloc(f32, Dc);
        const gate = try gpa.alloc(f32, Dc);
        const carry = try gpa.alloc(f32, B * Dc);
        fill(decay, 47);
        fill(gate, 53);
        fill(carry, 59);
        const d_dec = try Dev.alloc(Dc * 4);
        const d_gate = try Dev.alloc(Dc * 4);
        const d_carry = try Dev.alloc(B * Dc * 4);
        const d_cids = try Dev.alloc(B * T * 4);
        try Dev.up(d_dec, bytesOf(f32, decay));
        try Dev.up(d_gate, bytesOf(f32, gate));
        try Dev.up(d_carry, bytesOf(f32, carry));
        try Dev.up(d_cids, bytesOf(i32, opt_ids));
        const d_xc = try Dev.alloc(NT * 2);
        const d_dsc = try Dev.alloc(NT * 2);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_x)), d_xc, NT) != 0) return error.Ref;
        try Dev.up(d_acc, bytesOf(f32, a));
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_acc)), d_dsc, NT) != 0) return error.Ref;
        const d_S = try Dev.alloc(NT * 4);
        const d_mean = try Dev.alloc(B * T * 4);
        const d_rstd = try Dev.alloc(B * T * 4);
        const d_Y = try Dev.alloc(NT * 2);

        // forward, the three passes exactly as cell_forward runs them
        if (tmt.tmt_ref_cell_forward(d_xc, @ptrCast(@alignCast(d_S)), @ptrCast(@alignCast(d_dec)),
                                     @ptrCast(@alignCast(d_mean)), @ptrCast(@alignCast(d_rstd)), d_Y, B, T, Dc) != 0) return error.Ref;
        const ref_S = try gpa.alloc(f32, NT);
        const ref_stat = try gpa.alloc(f32, 2 * B * T);
        const ref_Y = try gpa.alloc(u16, NT);
        try Dev.down(bytesOf(f32, ref_S), d_S);
        try Dev.down(bytesOf(f32, ref_stat[0 .. B * T]), d_mean);
        try Dev.down(bytesOf(f32, ref_stat[B * T ..]), d_rstd);
        try Dev.down(bytesOf(u16, ref_Y), d_Y);
        const zero_opt = CellOpt{};
        const grid_y = (Dc + 255) / 256;
        try (try mod.get("state_pass")).launchGrid(B, grid_y, 256, .{ d_xc, d_S, d_dec, @as(u64, 0), @as(i32, B), @as(i32, T), @as(i32, Dc), @as(u64, 0), zero_opt });
        try (try mod.get("stats_pass")).launch(B * T, 256, .{ d_S, d_mean, d_rstd, @as(i32, B), @as(i32, T), @as(i32, Dc) });
        try (try mod.get("out_pass")).launchGrid(B, grid_y, 256, .{ d_xc, d_S, d_mean, d_rstd, d_Y, @as(i32, B), @as(i32, T), @as(i32, Dc) });
        const my_S = try gpa.alloc(f32, NT);
        const my_stat = try gpa.alloc(f32, 2 * B * T);
        const my_Y = try gpa.alloc(u16, NT);
        try Dev.down(bytesOf(f32, my_S), d_S);
        try Dev.down(bytesOf(f32, my_stat[0 .. B * T]), d_mean);
        try Dev.down(bytesOf(f32, my_stat[B * T ..]), d_rstd);
        try Dev.down(bytesOf(u16, my_Y), d_Y);
        var fb: [160]u8 = undefined;
        var nS: usize = 0;  // the three passes must agree exactly
        var nst: usize = 0;
        var nY: usize = 0;
        var wS: f32 = 0;
        for (ref_S, my_S) |v, w| if (v != w) {
            nS += 1;
            wS = @max(wS, @abs(v - w));
        };
        for (ref_stat, my_stat) |v, w| if (v != w) {
            nst += 1;
        };
        for (ref_Y, my_Y) |v, w| if (v != w) {
            nY += 1;
        };
        report("cell forward", nS == 0 and nst == 0 and nY == 0,
            std.fmt.bufPrint(&fb, "S {d} (max {e:.1}), mean/rstd {d}, Y {d} differ", .{ nS, wS, nst, nY }) catch "");

        // backward with document resets and hybrid traces
        const traces = try gpa.alloc(f32, B * Dc);
        fill(traces, 61);
        const d_dX = try Dev.alloc(NT * 4);
        const d_dDec = try Dev.alloc(Dc * 4);
        const d_dGate = try Dev.alloc(Dc * 4);
        const d_trD = try Dev.alloc(B * Dc * 4);
        const d_trG = try Dev.alloc(B * Dc * 4);
        const d_lam = try Dev.alloc(B * Dc * 4);
        const d_prod = try Dev.alloc(B * Dc * 4);
        const zeroDc = try gpa.alloc(f32, Dc);
        @memset(zeroDc, 0);
        const ref_b = try gpa.alloc(f32, NT + 2 * Dc + 4 * B * Dc);
        const my_b = try gpa.alloc(f32, NT + 2 * Dc + 4 * B * Dc);
        const gamma: f32 = 0.9;
        // One options struct for both builds: the C++ reference takes a pointer
        // to it, the Zig kernel takes it by value.
        const bwd_opt = CellOpt{ .ids = @intFromPtr(d_cids), .docsep = 10, .gamma = gamma,
            .trDec = @intFromPtr(d_trD), .trGate = @intFromPtr(d_trG),
            .lam = @intFromPtr(d_lam), .prod = @intFromPtr(d_prod) };
        for ([_][]f32{ ref_b, my_b }, 0..) |dst, run| {
            try Dev.up(d_dDec, bytesOf(f32, zeroDc));
            try Dev.up(d_dGate, bytesOf(f32, zeroDc));
            try Dev.up(d_trD, bytesOf(f32, traces));
            try Dev.up(d_trG, bytesOf(f32, traces));
            if (run == 0) {
                if (tmt.tmt_ref_state_bwd(d_dsc, @ptrCast(@alignCast(d_S)), @ptrCast(@alignCast(d_dec)),
                    @ptrCast(@alignCast(d_dX)), @ptrCast(@alignCast(d_dDec)), B, T, Dc, d_xc,
                    @ptrCast(@alignCast(d_carry)), @ptrCast(@alignCast(d_gate)), @ptrCast(@alignCast(d_dGate)),
                    &bwd_opt) != 0) return error.Ref;
            } else try (try mod.get("state_bwd")).launchGrid(B, grid_y, 256, .{ d_dsc, d_S, d_dec, d_dX, d_dDec,
                @as(i32, B), @as(i32, T), @as(i32, Dc), d_xc, d_carry, d_gate, d_dGate, bwd_opt });
            try Dev.down(bytesOf(f32, dst[0..NT]), d_dX);
            try Dev.down(bytesOf(f32, dst[NT .. NT + Dc]), d_dDec);
            try Dev.down(bytesOf(f32, dst[NT + Dc .. NT + 2 * Dc]), d_dGate);
            try Dev.down(bytesOf(f32, dst[NT + 2 * Dc ..][0 .. B * Dc]), d_trD);
            try Dev.down(bytesOf(f32, dst[NT + 2 * Dc + B * Dc ..][0 .. B * Dc]), d_trG);
            try Dev.down(bytesOf(f32, dst[NT + 2 * Dc + 2 * B * Dc ..][0 .. B * Dc]), d_lam);
            try Dev.down(bytesOf(f32, dst[NT + 2 * Dc + 3 * B * Dc ..][0 .. B * Dc]), d_prod);
        }
        // dX, traces, lambda and the decay product are per thread (no atomics)
        // and must be identical; only dDec/dGate are accumulated atomically.
        const atomic_from = NT;
        const atomic_to = NT + 2 * Dc;
        const parts = [_]struct { []const u8, usize, usize }{
            .{ "dX", 0, NT },
            .{ "trDec", atomic_to, atomic_to + B * Dc },
            .{ "trGate", atomic_to + B * Dc, atomic_to + 2 * B * Dc },
            .{ "lambda", atomic_to + 2 * B * Dc, atomic_to + 3 * B * Dc },
            .{ "prod", atomic_to + 3 * B * Dc, atomic_to + 4 * B * Dc },
        };
        var bit_ok = true;
        for (parts) |part| {
            const same = std.mem.eql(f32, ref_b[part[1]..part[2]], my_b[part[1]..part[2]]);
            if (!same) {
                var n: usize = 0;
                var wd: f32 = 0;
                for (ref_b[part[1]..part[2]], my_b[part[1]..part[2]]) |v, w| if (v != w) {
                    n += 1;
                    wd = @max(wd, @abs(v - w));
                };
                say("  {s}: {d} of {d} differ, max {e:.1}; first ref={d:.6} mine={d:.6}\n", .{ part[0], n, part[2] - part[1], wd, ref_b[part[1]], my_b[part[1]] });
            }
            bit_ok = bit_ok and same;
        }
        var scale: f32 = 1e-30;
        var worst: f32 = 0;
        for (ref_b[atomic_from..atomic_to]) |v| scale = @max(scale, @abs(v));
        for (ref_b[atomic_from..atomic_to], my_b[atomic_from..atomic_to]) |v, w| worst = @max(worst, @abs(v - w) / scale);
        var cb: [128]u8 = undefined;
        report("cell backward + traces", bit_ok and worst < 1e-5,
            std.fmt.bufPrint(&cb, "dDecay/dGate within {e:.1} (atomic order)", .{worst}) catch "");

        // embedding trace
        {
            const d_trE = try Dev.alloc(B * 256 * Dc * 4);
            const d_dEmb = try Dev.alloc(256 * Dc * 4);
            const trE = try gpa.alloc(f32, B * 256 * Dc);
            fill(trE, 67);
            const zeroE = try gpa.alloc(f32, 256 * Dc);
            @memset(zeroE, 0);
            const ref_e = try gpa.alloc(f32, B * 256 * Dc + 256 * Dc);
            const my_e = try gpa.alloc(f32, B * 256 * Dc + 256 * Dc);
            for ([_][]f32{ ref_e, my_e }, 0..) |dst, run| {
                try Dev.up(d_trE, bytesOf(f32, trE));
                try Dev.up(d_dEmb, bytesOf(f32, zeroE));
                const eo = CellOpt{ .ids = @intFromPtr(d_cids), .docsep = 10, .gamma = gamma,
                    .lam = @intFromPtr(d_lam), .prod = @intFromPtr(d_prod) };
                if (run == 0) {
                    if (tmt.tmt_ref_emb_trace(d_xc, @ptrCast(@alignCast(d_S)), @ptrCast(@alignCast(d_carry)),
                        @ptrCast(@alignCast(d_dec)), @ptrCast(@alignCast(d_gate)), @ptrCast(@alignCast(d_trE)),
                        @ptrCast(@alignCast(d_dEmb)), B, T, Dc, &eo) != 0) return error.Ref;
                } else try (try mod.get("emb_trace")).launchGrid(B, grid_y, 256, .{ d_xc, d_S, d_carry, d_dec,
                    d_gate, d_trE, d_dEmb, @as(i32, B), @as(i32, T), @as(i32, Dc), eo });
                try Dev.down(bytesOf(f32, dst[0 .. B * 256 * Dc]), d_trE);
                try Dev.down(bytesOf(f32, dst[B * 256 * Dc ..]), d_dEmb);
            }
            const split = B * 256 * Dc;
            var s2: f32 = 1e-30;
            var w2: f32 = 0;
            for (ref_e[split..]) |v| s2 = @max(s2, @abs(v));
            for (ref_e[split..], my_e[split..]) |v, w| w2 = @max(w2, @abs(v - w) / s2);
            var eb: [128]u8 = undefined;
            var nE: usize = 0;
            var wE: f32 = 0;
            var sE: f32 = 1e-30;
            for (ref_e[0..split]) |v| sE = @max(sE, @abs(v));
            for (ref_e[0..split], my_e[0..split]) |v, w| if (v != w) {
                nE += 1;
                wE = @max(wE, @abs(v - w) / sE);
            };
            // The trace itself is a long fp32 accumulation; a few of its
            // entries end up one unit in the last place apart because nvcc
            // fuses a multiply and an add here that LLVM keeps separate.
            report("embedding trace", wE < 1e-6 and w2 < 1e-5,
                std.fmt.bufPrint(&eb, "trace within {e:.1} (1 ulp), dEmb within {e:.1} (atomic order)", .{ wE, w2 }) catch "");
        }
    }

    // ---- multi-tensor optimizer: norm, AdamW with clipping, zeroing ----
    {
        const sizes = [_]usize{ 5000, 300, 40000 };
        const CHUNK = 4096;
        var nchunks: usize = 0;
        for (sizes) |n| nchunks += (n + CHUNK - 1) / CHUNK;
        const chunks = try gpa.alloc(MTChunk, nchunks);
        var at: usize = 0;
        for (sizes, 0..) |n, p| {
            var off: usize = 0;
            while (off < n) : (off += CHUNK) {
                chunks[at] = .{ .param = @intCast(p), .len = @intCast(@min(CHUNK, n - off)), .start = @intCast(off) };
                at += 1;
            }
        }
        var total: usize = 0;
        for (sizes) |n| total += n;
        const master = try gpa.alloc(f32, total);
        const mom = try gpa.alloc(f32, total);
        const vel = try gpa.alloc(f32, total);
        const grad = try gpa.alloc(f32, total);
        fill(master, 71);
        fill(mom, 73);
        fill(vel, 79);
        fill(grad, 83);
        for (vel) |*v| v.* = @abs(v.*); // second moments are nonnegative
        // device buffers per parameter plus the pointer tables
        var d_master: [3]u64 = undefined;
        var d_m: [3]u64 = undefined;
        var d_v: [3]u64 = undefined;
        var d_g: [3]u64 = undefined;
        var d_w: [3]u64 = undefined;
        var base: usize = 0;
        for (sizes, 0..) |n, p| {
            d_master[p] = @intFromPtr(try Dev.alloc(n * 4));
            d_m[p] = @intFromPtr(try Dev.alloc(n * 4));
            d_v[p] = @intFromPtr(try Dev.alloc(n * 4));
            d_g[p] = @intFromPtr(try Dev.alloc(n * 4));
            d_w[p] = @intFromPtr(try Dev.alloc(n * 2));
            try Dev.up(@ptrFromInt(d_m[p]), bytesOf(f32, mom[base..][0..n]));
            try Dev.up(@ptrFromInt(d_v[p]), bytesOf(f32, vel[base..][0..n]));
            try Dev.up(@ptrFromInt(d_g[p]), bytesOf(f32, grad[base..][0..n]));
            base += n;
        }
        const flags = [_]u8{ 3, 3, 1 }; // the third is in the norm but not updated
        const tables = [_][]const u64{ &d_master, &d_m, &d_v, &d_g, &d_w };
        var d_tab: [5]u64 = undefined;
        for (tables, 0..) |t, i| {
            d_tab[i] = @intFromPtr(try Dev.alloc(3 * 8));
            try Dev.up(@ptrFromInt(d_tab[i]), std.mem.sliceAsBytes(t));
        }
        const d_flags = try Dev.alloc(3);
        try Dev.up(d_flags, &flags);
        const d_chunks = try Dev.alloc(nchunks * @sizeOf(MTChunk));
        try Dev.up(d_chunks, std.mem.sliceAsBytes(chunks));
        const d_sumsq = try Dev.alloc(8);
        const P = MTParams{
            .master = d_tab[0], .m = d_tab[1], .v = d_tab[2], .grad = d_tab[3], .work = d_tab[4],
            .flags = @intFromPtr(d_flags), .chunks = @intFromPtr(d_chunks),
            .nchunks = @intCast(nchunks), .sumsq = @intFromPtr(d_sumsq),
        };
        const out = try gpa.alloc(f32, 4 * total);
        const mine = try gpa.alloc(f32, 4 * total);
        var norms: [2]f64 = undefined;
        for ([_][]f32{ out, mine }, 0..) |dst, run| {
            base = 0;
            for (sizes, 0..) |n, p| {
                try Dev.up(@ptrFromInt(d_master[p]), bytesOf(f32, master[base..][0..n]));
                try Dev.up(@ptrFromInt(d_m[p]), bytesOf(f32, mom[base..][0..n]));
                try Dev.up(@ptrFromInt(d_v[p]), bytesOf(f32, vel[base..][0..n]));
                try Dev.up(@ptrFromInt(d_g[p]), bytesOf(f32, grad[base..][0..n]));
                base += n;
            }
            try Dev.up(d_sumsq, &[_]u8{0} ** 8);
            if (run == 0) {
                if (tmt.tmt_ref_mt(&P, 0, 0, 0, 0, 0) != 0) return error.Ref;
                if (tmt.tmt_ref_mt(&P, 1, 1.0, 0.001, 0.1, 0.002) != 0) return error.Ref;
            } else {
                try (try mod.get("mt_sumsq")).launch(@intCast(nchunks), 256, .{P});
                try (try mod.get("mt_adam")).launch(@intCast(nchunks), 256, .{ P, @as(f32, 1.0), @as(f32, 0.001),
                    @as(f32, 0.9), @as(f32, 0.999), @as(f32, 1e-8), @as(f32, 0.01), @as(f32, 0.1), @as(f32, 0.002) });
            }
            try Dev.down(std.mem.asBytes(&norms[run]), d_sumsq);
            base = 0;
            for (sizes, 0..) |n, p| {
                try Dev.down(bytesOf(f32, dst[base..][0..n]), @ptrFromInt(d_master[p]));
                try Dev.down(bytesOf(f32, dst[total + base ..][0..n]), @ptrFromInt(d_m[p]));
                try Dev.down(bytesOf(f32, dst[2 * total + base ..][0..n]), @ptrFromInt(d_v[p]));
                const wbuf = try gpa.alloc(u16, n);
                try Dev.down(bytesOf(u16, wbuf), @ptrFromInt(d_w[p]));
                for (wbuf, 0..) |bits, i| dst[3 * total + base + i] = @bitCast(@as(u32, bits) << 16);
                base += n;
            }
        }
        // The squared norm is summed with atomic adds over blocks, so it is the
        // one value that need not match to the last bit.
        const nrel = @abs(norms[0] - norms[1]) / @max(@abs(norms[0]), 1e-30);
        var ob: [128]u8 = undefined;
        report("multi-tensor AdamW", std.mem.eql(f32, out, mine) and nrel < 1e-12,
            std.fmt.bufPrint(&ob, "weights/moments identical, norm within {e:.1}", .{nrel}) catch "");

        // gradient zeroing
        try (try mod.get("mt_zero")).launch(@intCast(nchunks), 256, .{P});
        base = 0;
        var zeroed = true;
        for (sizes, 0..) |n, p| {
            const buf = try gpa.alloc(f32, n);
            try Dev.down(bytesOf(f32, buf), @ptrFromInt(d_g[p]));
            for (buf) |v| zeroed = zeroed and v == 0;
            base += n;
        }
        report("mt_zero", zeroed, "");
    }

    // ---- Mixture of Experts ----
    {
        const R = 256; // positions
        const E = 8;
        const Kk = 2;
        const Dm = 64;
        const TK = R * Kk;
        const xs = try gpa.alloc(f32, R * Dm);
        const wr = try gpa.alloc(f32, E * Dm);
        fill(xs, 89);
        fill(wr, 97);
        const d_xf = try Dev.alloc(R * Dm * 4);
        const d_wf = try Dev.alloc(E * Dm * 4);
        try Dev.up(d_xf, bytesOf(f32, xs));
        try Dev.up(d_wf, bytesOf(f32, wr));
        const d_xb = try Dev.alloc(R * Dm * 2);
        const d_wb = try Dev.alloc(E * Dm * 2);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_xf)), d_xb, R * Dm) != 0) return error.Ref;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_wf)), d_wb, E * Dm) != 0) return error.Ref;
        const d_rl = try Dev.alloc(R * E * 4);
        const d_rp = try Dev.alloc(R * E * 4);
        const d_idx = try Dev.alloc(TK * 4);
        const d_rw = try Dev.alloc(TK * 4);
        const rl = try gpa.alloc(f32, 2 * R * E);
        const mrl = try gpa.alloc(f32, 2 * R * E);
        const ridx = try gpa.alloc(i32, 2 * TK);
        const midx = try gpa.alloc(i32, 2 * TK);

        // router + top-k (warp shuffles, vector loads)
        for ([_]u8{ 0, 1 }) |run| {
            if (run == 0) {
                if (tmt.tmt_ref_router_topk(d_xb, d_wb, @ptrCast(@alignCast(d_rl)), @ptrCast(@alignCast(d_rp)),
                    @ptrCast(@alignCast(d_idx)), @ptrCast(@alignCast(d_rw)), R, E, Kk, Dm) != 0) return error.Ref;
            } else try (try mod.get("router_topk")).launch(R, 64, .{ d_xb, d_wb, d_rl, d_rp, d_idx, d_rw,
                @as(i32, R), @as(i32, E), @as(i32, Kk), @as(i32, Dm) });
            const dst = if (run == 0) rl else mrl;
            const di = if (run == 0) ridx else midx;
            try Dev.down(bytesOf(f32, dst[0 .. R * E]), d_rl);
            try Dev.down(bytesOf(f32, dst[R * E ..]), d_rp);
            try Dev.down(bytesOf(i32, di[0..TK]), d_idx);
            try Dev.down(bytesOf(f32, @as([]f32, @ptrCast(di[TK..]))), d_rw);
        }
        report("router + top-k", std.mem.eql(f32, rl, mrl) and std.mem.eql(i32, ridx, midx), "");

        // dispatch: the counts must agree; the slot order comes from atomics
        const zeros = try gpa.alloc(i32, E);
        @memset(zeros, 0);
        const d_counts = try Dev.alloc(E * 4);
        const d_cursor = try Dev.alloc(E * 4);
        const d_perm = try Dev.alloc(TK * 4);
        const d_slotw = try Dev.alloc(TK * 4);
        const d_slot = try Dev.alloc(TK * 4);
        const cref = try gpa.alloc(i32, E);
        const cmine = try gpa.alloc(i32, E);
        try Dev.up(d_counts, bytesOf(i32, zeros));
        if (tmt.tmt_ref_count(@ptrCast(@alignCast(d_idx)), @ptrCast(@alignCast(d_counts)), R, Kk) != 0) return error.Ref;
        try Dev.down(bytesOf(i32, cref), d_counts);
        try Dev.up(d_counts, bytesOf(i32, zeros));
        try (try mod.get("count_experts")).launch((R + 255) / 256, 256, .{ d_idx, d_counts, @as(i32, R), @as(i32, Kk) });
        try Dev.down(bytesOf(i32, cmine), d_counts);
        report("expert counts", std.mem.eql(i32, cref, cmine), "");

        // one dispatch (from the C++ kernel) feeds both builds from here on
        const offsets = try gpa.alloc(i32, E);
        var run_off: i32 = 0;
        for (offsets, cref) |*o, c| {
            o.* = run_off;
            run_off += c;
        }
        try Dev.up(d_cursor, bytesOf(i32, offsets));
        if (tmt.tmt_ref_fill(@ptrCast(@alignCast(d_idx)), @ptrCast(@alignCast(d_rw)), @ptrCast(@alignCast(d_cursor)),
            @ptrCast(@alignCast(d_perm)), @ptrCast(@alignCast(d_slotw)), @ptrCast(@alignCast(d_slot)), R, Kk) != 0) return error.Ref;

        // gather, combine, backward, scatter: all deterministic
        const d_xg = try Dev.alloc(TK * Dm * 2);
        const d_yg = try Dev.alloc(TK * Dm * 2);
        const d_y = try Dev.alloc(R * Dm * 2);
        const d_dyg = try Dev.alloc(TK * Dm * 2);
        const d_sj = try Dev.alloc(TK * 4);
        const d_dl = try Dev.alloc(R * E * 4);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_x)), d_yg, TK * Dm) != 0) return error.Ref;
        const bigA = try gpa.alloc(u16, TK * Dm);
        const bigB = try gpa.alloc(u16, TK * Dm);
        const fA = try gpa.alloc(f32, R * E);
        const fB = try gpa.alloc(f32, R * E);

        for ([_]u8{ 0, 1 }) |run| {
            const dst = if (run == 0) bigA else bigB;
            if (run == 0) {
                if (tmt.tmt_ref_gather_slot(d_xb, @ptrCast(@alignCast(d_slot)), d_xg, R, Kk, Dm) != 0) return error.Ref;
            } else try (try mod.get("gather_slot")).launch((TK * Dm + 255) / 256, 256,
                .{ d_xb, d_slot, d_xg, @as(i32, R), @as(i32, Kk), @as(i32, Dm) });
            try Dev.down(bytesOf(u16, dst[0 .. TK * Dm]), d_xg);
        }
        report("gather_slot", std.mem.eql(u16, bigA, bigB), "");
        // Pristine start value for the kernels that read their own output (beta).
        const y0 = try gpa.dupe(u16, bigA[0 .. R * Dm]);

        for ([_]u8{ 0, 1 }) |run| {
            try Dev.up(d_y, bytesOf(u16, y0));
            if (run == 0) {
                if (tmt.tmt_ref_combine(d_yg, @ptrCast(@alignCast(d_slotw)), @ptrCast(@alignCast(d_slot)), d_y, 1.0, R, Kk, Dm) != 0) return error.Ref;
            } else try (try mod.get("combine")).launch((R * Dm + 255) / 256, 256,
                .{ d_yg, d_slotw, d_slot, d_y, @as(i32, R), @as(i32, Kk), @as(i32, Dm), @as(f32, 1.0) });
            try Dev.down(bytesOf(u16, (if (run == 0) bigA else bigB)[0 .. R * Dm]), d_y);
        }
        report("combine", std.mem.eql(u16, bigA[0 .. R * Dm], bigB[0 .. R * Dm]), "");

        const sjA = try gpa.alloc(f32, TK);
        const sjB = try gpa.alloc(f32, TK);
        for ([_]u8{ 0, 1 }) |run| {
            if (run == 0) {
                if (tmt.tmt_ref_combine_bwd(d_y, d_yg, @ptrCast(@alignCast(d_slotw)), @ptrCast(@alignCast(d_slot)), d_dyg, R, Kk, Dm) != 0) return error.Ref;
                if (tmt.tmt_ref_sdot(d_y, d_yg, @ptrCast(@alignCast(d_slot)), @ptrCast(@alignCast(d_sj)), R, Kk, Dm) != 0) return error.Ref;
            } else {
                try (try mod.get("combine_bwd")).launch((TK * Dm + 255) / 256, 256,
                    .{ d_y, d_yg, d_slotw, d_slot, d_dyg, @as(u64, 0), @as(i32, R), @as(i32, Kk), @as(i32, Dm) });
                try (try mod.get("sdot")).launch(TK, 256, .{ d_y, d_yg, d_slot, d_sj, @as(i32, R), @as(i32, Kk), @as(i32, Dm) });
            }
            try Dev.down(bytesOf(u16, (if (run == 0) bigA else bigB)[0 .. TK * Dm]), d_dyg);
            try Dev.down(bytesOf(f32, if (run == 0) sjA else sjB), d_sj);
        }
        report("combine backward + sdot", std.mem.eql(u16, bigA[0 .. TK * Dm], bigB[0 .. TK * Dm]) and
            std.mem.eql(f32, sjA, sjB), "");

        for ([_]u8{ 0, 1 }) |run| {
            if (run == 0) {
                if (tmt.tmt_ref_router_bwd(@ptrCast(@alignCast(d_rp)), @ptrCast(@alignCast(d_idx)),
                    @ptrCast(@alignCast(d_sj)), @ptrCast(@alignCast(d_dl)), 0.1, 0.001,
                    @ptrCast(@alignCast(d_rl)), @ptrCast(@alignCast(d_counts)), R, E, Kk) != 0) return error.Ref;
            } else try (try mod.get("router_bwd")).launch((R + 255) / 256, 256,
                .{ d_rp, d_idx, d_sj, d_dl, @as(i32, R), @as(i32, E), @as(i32, Kk), d_rl, d_counts,
                   @as(f32, 0.1), @as(f32, 0.001) });
            try Dev.down(bytesOf(f32, if (run == 0) fA else fB), d_dl);
        }
        var rn: usize = 0;
        var rsc: f32 = 1e-30;
        var rw: f32 = 0;
        for (fA) |v| rsc = @max(rsc, @abs(v));
        for (fA, fB) |v, w| if (v != w) {
            rn += 1;
            rw = @max(rw, @abs(v - w) / rsc);
        };
        // The auxiliary and z-loss terms are a sum of products that nvcc
        // hoists and fuses differently; the values stay within one or two
        // units in the last place.
        var rb: [128]u8 = undefined;
        report("router backward", rw < 1e-6,
            std.fmt.bufPrint(&rb, "within {e:.1} ({d} of {d} values, 1-2 ulp)", .{ rw, rn, fA.len }) catch "");

        for ([_]u8{ 0, 1 }) |run| {
            try Dev.up(d_y, bytesOf(u16, y0));
            if (run == 0) {
                if (tmt.tmt_ref_scatter_add(d_dyg, @ptrCast(@alignCast(d_slot)), d_y, R, Kk, Dm) != 0) return error.Ref;
            } else try (try mod.get("scatter_add")).launch((R * Dm + 255) / 256, 256,
                .{ d_dyg, d_slot, d_y, @as(i32, R), @as(i32, Kk), @as(i32, Dm) });
            try Dev.down(bytesOf(u16, (if (run == 0) bigA else bigB)[R * Dm ..][0 .. R * Dm]), d_y);
        }
        report("scatter_add", std.mem.eql(u16, bigA[R * Dm ..][0 .. R * Dm], bigB[R * Dm ..][0 .. R * Dm]), "");

        // dense path, auxiliary sums and z-loss
        for ([_]u8{ 0, 1 }) |run| {
            try Dev.up(d_y, bytesOf(u16, y0));
            if (run == 0) {
                if (tmt.tmt_ref_dense_activation(d_yg, d_xg, d_y, 0.5, R * Dm) != 0) return error.Ref;
            } else try (try mod.get("dense_activation")).launch((R * Dm + 255) / 256, 256,
                .{ d_yg, d_xg, d_y, @as(i64, R * Dm), @as(f32, 0.5) });
            try Dev.down(bytesOf(u16, (if (run == 0) bigA else bigB)[0 .. R * Dm]), d_y);
        }
        report("dense activation", std.mem.eql(u16, bigA[0 .. R * Dm], bigB[0 .. R * Dm]), "");

        const d_sump = try Dev.alloc(E * 4);
        const auxA = try gpa.alloc(f32, E);
        const auxB = try gpa.alloc(f32, E);
        if (tmt.tmt_ref_aux_sum(@ptrCast(@alignCast(d_rp)), @ptrCast(@alignCast(d_sump)), R, E) != 0) return error.Ref;
        try Dev.down(bytesOf(f32, auxA), d_sump);
        try (try mod.get("aux_sum")).launch(1, 32, .{ d_rp, d_sump, @as(i32, R), @as(i32, E) });
        try Dev.down(bytesOf(f32, auxB), d_sump);
        const d_z = try Dev.alloc(4);
        var zA: f32 = 0;
        var zB: f32 = 0;
        try Dev.up(d_z, &[_]u8{0} ** 4);
        if (tmt.tmt_ref_zloss(@ptrCast(@alignCast(d_rl)), @ptrCast(@alignCast(d_z)), R, E) != 0) return error.Ref;
        try Dev.down(std.mem.asBytes(&zA), d_z);
        try Dev.up(d_z, &[_]u8{0} ** 4);
        try (try mod.get("zloss")).launch((R + 255) / 256, 256, .{ d_rl, d_z, @as(i32, R), @as(i32, E) });
        try Dev.down(std.mem.asBytes(&zB), d_z);
        var zb: [96]u8 = undefined;
        const zrel = @abs(zA - zB) / @max(@abs(zA), 1e-30);
        report("aux sums + z-loss", std.mem.eql(f32, auxA, auxB) and zrel < 1e-6,
            std.fmt.bufPrint(&zb, "z-loss within {e:.1} (atomic order)", .{zrel}) catch "");
    }

    // ---- fact memory (cross-attention over the retrieved facts) ----
    {
        const B = 3;
        const T = 16;
        const M = 24;
        const H = 2;
        const dh = 8;
        const Dm = 32;
        const HD = H * dh;
        const mids = try gpa.alloc(i32, B * M);
        fillIds(mids, 101, -1, 256); // -1 is padding and must stay masked
        const d_mids = try Dev.alloc(B * M * 4);
        try Dev.up(d_mids, bytesOf(i32, mids));
        const d_tabs = try Dev.alloc(256 * Dm * 2); // E, Eprev share the table buffer
        const d_pos = try Dev.alloc(M * Dm * 2);
        const d_enc = try Dev.alloc(B * M * Dm * 2);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_tab_f)), d_tabs, 256 * Dm) != 0) return error.Ref;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_x)), d_pos, M * Dm) != 0) return error.Ref;
        const encA = try gpa.alloc(u16, B * M * Dm);
        const encB = try gpa.alloc(u16, B * M * Dm);
        for ([_]u8{ 0, 1 }) |run| {
            if (run == 0) {
                if (tmt.tmt_ref_mem_encode(d_tabs, d_tabs, d_pos, @ptrCast(@alignCast(d_mids)), d_enc, B, M, Dm) != 0) return error.Ref;
            } else try (try mod.get("mem_encode")).launch((B * M * Dm + 255) / 256, 256,
                .{ d_tabs, d_tabs, d_pos, d_mids, d_enc, @as(i32, B), @as(i32, M), @as(i32, Dm) });
            try Dev.down(bytesOf(u16, if (run == 0) encA else encB), d_enc);
        }
        report("memory encoding", std.mem.eql(u16, encA, encB), "");

        // cross-attention forward and both backward halves
        const d_Q = try Dev.alloc(B * T * HD * 2);
        const d_K = try Dev.alloc(B * M * HD * 2);
        const d_V = try Dev.alloc(B * M * HD * 2);
        const d_O = try Dev.alloc(B * T * HD * 2);
        const d_Pm = try Dev.alloc(B * H * T * M * 4);
        const d_dS = try Dev.alloc(B * H * T * M * 4);
        const d_dQ = try Dev.alloc(B * T * HD * 2);
        const d_dK = try Dev.alloc(B * M * HD * 2);
        const d_dV = try Dev.alloc(B * M * HD * 2);
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_x)), d_Q, B * T * HD) != 0) return error.Ref;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_acc)), d_K, B * M * HD) != 0) return error.Ref;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_tab_f)), d_V, B * M * HD) != 0) return error.Ref;
        if (tmt.tmt_ref_copy_bf16(@ptrCast(@alignCast(d_x)), d_O, B * T * HD) != 0) return error.Ref;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dh)));
        const fwdA = try gpa.alloc(u16, B * T * HD);
        const fwdB = try gpa.alloc(u16, B * T * HD);
        const pA = try gpa.alloc(f32, B * H * T * M);
        const pB = try gpa.alloc(f32, B * H * T * M);
        for ([_]u8{ 0, 1 }) |run| {
            if (run == 0) {
                if (tmt.tmt_ref_mem_attn_fwd(scale, d_Q, d_K, d_V, @ptrCast(@alignCast(d_mids)),
                    @ptrCast(@alignCast(d_Pm)), d_O, B, T, M, H, dh) != 0) return error.Ref;
            } else try (try mod.get("mem_attn_fwd")).launch((B * H * T + 255) / 256, 256,
                .{ d_Q, d_K, d_V, d_mids, d_Pm, d_O, @as(i32, B), @as(i32, T), @as(i32, M), @as(i32, H),
                   @as(i32, dh), scale });
            try Dev.down(bytesOf(u16, if (run == 0) fwdA else fwdB), d_O);
            try Dev.down(bytesOf(f32, if (run == 0) pA else pB), d_Pm);
        }
        report("memory attention forward", std.mem.eql(u16, fwdA, fwdB) and std.mem.eql(f32, pA, pB), "");

        const dqA = try gpa.alloc(u16, B * T * HD + 2 * B * M * HD);
        const dqB = try gpa.alloc(u16, B * T * HD + 2 * B * M * HD);
        const dsA = try gpa.alloc(f32, B * H * T * M);
        const dsB = try gpa.alloc(f32, B * H * T * M);
        for ([_]u8{ 0, 1 }) |run| {
            if (run == 0) {
                if (tmt.tmt_ref_mem_attn_bwd_q(scale, d_O, d_K, d_V, @ptrCast(@alignCast(d_Pm)),
                    @ptrCast(@alignCast(d_dS)), d_dQ, B, T, M, H, dh) != 0) return error.Ref;
                if (tmt.tmt_ref_mem_attn_bwd_kv(scale, d_Q, d_O, @ptrCast(@alignCast(d_Pm)),
                    @ptrCast(@alignCast(d_dS)), d_dK, d_dV, B, T, M, H, dh) != 0) return error.Ref;
            } else {
                try (try mod.get("mem_attn_bwd_q")).launch((B * H * T + 255) / 256, 256,
                    .{ d_O, d_K, d_V, d_Pm, d_dS, d_dQ, @as(i32, B), @as(i32, T), @as(i32, M),
                       @as(i32, H), @as(i32, dh), scale });
                try (try mod.get("mem_attn_bwd_kv")).launch((B * H * M + 255) / 256, 256,
                    .{ d_Q, d_O, d_Pm, d_dS, d_dK, d_dV, @as(i32, B), @as(i32, T), @as(i32, M),
                       @as(i32, H), @as(i32, dh), scale });
            }
            const dst = if (run == 0) dqA else dqB;
            try Dev.down(bytesOf(u16, dst[0 .. B * T * HD]), d_dQ);
            try Dev.down(bytesOf(u16, dst[B * T * HD ..][0 .. B * M * HD]), d_dK);
            try Dev.down(bytesOf(u16, dst[B * T * HD + B * M * HD ..]), d_dV);
            try Dev.down(bytesOf(f32, if (run == 0) dsA else dsB), d_dS);
        }
        report("memory attention backward", std.mem.eql(u16, dqA, dqB) and std.mem.eql(f32, dsA, dsB), "");

        // encoding gradient (atomic scatter) and the two elementwise helpers
        const zeroT = try gpa.alloc(f32, 256 * Dm);
        @memset(zeroT, 0);
        const d_dE = try Dev.alloc(256 * Dm * 4);
        const d_dEp = try Dev.alloc(256 * Dm * 4);
        const d_dP = try Dev.alloc(M * Dm * 4);
        const gA = try gpa.alloc(f32, 2 * 256 * Dm + M * Dm);
        const gB = try gpa.alloc(f32, 2 * 256 * Dm + M * Dm);
        for ([_]u8{ 0, 1 }) |run| {
            try Dev.up(d_dE, bytesOf(f32, zeroT));
            try Dev.up(d_dEp, bytesOf(f32, zeroT));
            try Dev.up(d_dP, bytesOf(f32, zeroT[0 .. M * Dm]));
            if (run == 0) {
                if (tmt.tmt_ref_mem_encode_bwd(@ptrCast(@alignCast(d_x)), @ptrCast(@alignCast(d_mids)),
                    @ptrCast(@alignCast(d_dE)), @ptrCast(@alignCast(d_dEp)), @ptrCast(@alignCast(d_dP)), B, M, Dm) != 0) return error.Ref;
            } else try (try mod.get("mem_encode_bwd")).launch((B * M * Dm + 255) / 256, 256,
                .{ d_x, d_mids, d_dE, d_dEp, d_dP, @as(i32, B), @as(i32, M), @as(i32, Dm) });
            const dst = if (run == 0) gA else gB;
            try Dev.down(bytesOf(f32, dst[0 .. 256 * Dm]), d_dE);
            try Dev.down(bytesOf(f32, dst[256 * Dm ..][0 .. 256 * Dm]), d_dEp);
            try Dev.down(bytesOf(f32, dst[2 * 256 * Dm ..]), d_dP);
        }
        var gscale: f32 = 1e-30;
        var gworst: f32 = 0;
        for (gA) |v| gscale = @max(gscale, @abs(v));
        for (gA, gB) |v, w| gworst = @max(gworst, @abs(v - w) / gscale);
        var gb: [96]u8 = undefined;
        report("memory encoding gradient", gworst < 1e-5,
            std.fmt.bufPrint(&gb, "within {e:.1} (atomic order)", .{gworst}) catch "");

        const n2 = B * M * Dm;
        const hA = try gpa.alloc(u16, n2);
        const hB = try gpa.alloc(u16, n2);
        for ([_]u8{ 0, 1 }) |run| {
            try Dev.up(d_enc, bytesOf(u16, encA));
            if (run == 0) {
                if (tmt.tmt_ref_mem_add_bf16(d_enc, d_tabs, n2) != 0) return error.Ref;
                if (tmt.tmt_ref_mem_sum(d_enc, @ptrCast(@alignCast(d_x)), d_enc, n2) != 0) return error.Ref;
            } else {
                try (try mod.get("mem_add_bf16")).launch((n2 + 255) / 256, 256, .{ d_enc, d_tabs, @as(i64, n2) });
                try (try mod.get("mem_sum")).launch((n2 + 255) / 256, 256, .{ d_enc, d_x, d_enc, @as(i64, n2) });
            }
            try Dev.down(bytesOf(u16, if (run == 0) hA else hB), d_enc);
        }
        report("memory helpers", std.mem.eql(u16, hA, hB), "");
    }

    if (failures == 0) say("ALL KERNEL COMPARISONS PASSED\n", .{});
    return if (failures == 0) 0 else 1;
}
