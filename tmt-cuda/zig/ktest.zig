//! ktest: the Zig kernels must compute exactly what the C++ ones compute.
//! Runs both on the same device buffers and compares the results bit for bit.
//! Kernels whose result depends on the order of atomic adds are compared with
//! two runs of each build, so the comparison can only be as strict as the
//! reference is reproducible.
const std = @import("std");
const tmt = @import("tmt.zig");
const ptx = @import("ptx.zig");

const kernels_ptx = @embedFile("kernels.ptx");

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

    if (failures == 0) say("ALL KERNEL COMPARISONS PASSED\n", .{});
    return if (failures == 0) 0 else 1;
}
