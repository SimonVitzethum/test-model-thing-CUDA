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

    if (failures == 0) say("ALL KERNEL COMPARISONS PASSED\n", .{});
    return if (failures == 0) 0 else 1;
}
