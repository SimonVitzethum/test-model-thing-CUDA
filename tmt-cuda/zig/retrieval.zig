//! Stage-2 retrieval head, host math (port of src/retrieval.h).
//!
//! Query  q_b = Wq x_b  from the model state at the last question byte,
//! key    k_c = Wk y_c  from the model state at the last byte of a node label.
//! Score  s_bc = cos(q_b, k_c) / tau; InfoNCE loss with positive pos[b]:
//!   L = mean_b ( logsumexp_c s_bc - s_b,pos[b] )
//! exp/log go through libc, so the numbers match the C++ build bit for bit.
const std = @import("std");
const tmt = @import("tmt.zig");

extern "c" fn exp(x: f64) f64;

pub const Batch = struct {
    B: usize,
    C: usize,
    D: usize,
    R: usize,
    tau: f32 = 0.05,
    loss: f64 = 0,
    accuracy: f64 = 0,
};

/// out[r] = sum_d W[r*D + d] v[d]
fn headApply(W: []const f32, v: []const f32, R: usize, D: usize, out: []f64) void {
    for (0..R) |r| {
        var a: f64 = 0;
        for (0..D) |d| a += @as(f64, W[r * D + d]) * @as(f64, v[d]);
        out[r] = a;
    }
}
fn normalize(v: []f64) f64 {
    var n: f64 = 0;
    for (v) |e| n += e * e;
    n = @sqrt(@max(n, 1e-24));
    for (v) |*e| e.* /= n;
    return n;
}

/// Unit vector of W v (the normalized query or key), for scoring the index.
pub fn headUnit(gpa: std.mem.Allocator, W: []const f32, v: []const f32, R: usize, D: usize) ![]f32 {
    const h = try gpa.alloc(f64, R);
    defer gpa.free(h);
    headApply(W, v, R, D, h);
    _ = normalize(h);
    const out = try gpa.alloc(f32, R);
    for (out, h) |*o, e| o.* = @floatCast(e);
    return out;
}

/// Fills dx, dy (overwritten) and dWq, dWk (accumulated) and returns loss and
/// accuracy in `rb`.
pub fn loss(gpa: std.mem.Allocator, rb: *Batch, x: []const f32, y: []const f32, pos: []const usize,
            Wq: []const f32, Wk: []const f32, dx: []f32, dy: []f32, dWq: []f64, dWk: []f64) !void {
    const B = rb.B;
    const C = rb.C;
    const D = rb.D;
    const R = rb.R;
    const q = try gpa.alloc(f64, B * R);
    const k = try gpa.alloc(f64, C * R);
    const qn = try gpa.alloc(f64, B);
    const kn = try gpa.alloc(f64, C);
    const dq = try gpa.alloc(f64, B * R);
    const dk = try gpa.alloc(f64, C * R);
    const s = try gpa.alloc(f64, C);
    defer inline for (.{ q, k, qn, kn, dq, dk, s }) |p| gpa.free(p);
    for (0..B) |b| {
        headApply(Wq, x[b * D ..][0..D], R, D, q[b * R ..][0..R]);
        qn[b] = normalize(q[b * R ..][0..R]);
    }
    for (0..C) |c| {
        headApply(Wk, y[c * D ..][0..D], R, D, k[c * R ..][0..R]);
        kn[c] = normalize(k[c * R ..][0..R]);
    }
    @memset(dq, 0);
    @memset(dk, 0);
    rb.loss = 0;
    rb.accuracy = 0;
    for (0..B) |b| {
        var mx: f64 = -1e300;
        var best: usize = 0;
        for (0..C) |c| {
            var dot: f64 = 0;
            for (0..R) |r| dot += q[b * R + r] * k[c * R + r];
            s[c] = dot / rb.tau;
            if (s[c] > mx) {
                mx = s[c];
                best = c;
            }
        }
        var sum: f64 = 0;
        for (s) |v| sum += exp(v - mx);
        rb.loss += mx + tmt.log(sum) - s[pos[b]];
        rb.accuracy += @floatFromInt(@intFromBool(best == pos[b]));
        for (0..C) |c| {
            const g = (exp(s[c] - mx) / sum - @as(f64, if (c == pos[b]) 1 else 0)) / (@as(f64, @floatFromInt(B)) * rb.tau);
            for (0..R) |r| {
                dq[b * R + r] += g * k[c * R + r];
                dk[c * R + r] += g * q[b * R + r];
            }
        }
    }
    rb.loss /= @floatFromInt(B);
    rb.accuracy /= @floatFromInt(B);
    // Through the normalization: d(v/|v|) = (g - u (u . g)) / |v|.
    const throughNorm = struct {
        fn f(g: []f64, u: []const f64, n: f64) void {
            var dot: f64 = 0;
            for (u, g) |ui, gi| dot += ui * gi;
            for (g, u) |*gi, ui| gi.* = (gi.* - ui * dot) / n;
        }
    }.f;
    @memset(dx, 0);
    @memset(dy, 0);
    for (0..B) |b| {
        throughNorm(dq[b * R ..][0..R], q[b * R ..][0..R], qn[b]);
        for (0..R) |r| for (0..D) |d| {
            dWq[r * D + d] += dq[b * R + r] * @as(f64, x[b * D + d]);
            dx[b * D + d] += @floatCast(dq[b * R + r] * @as(f64, Wq[r * D + d]));
        };
    }
    for (0..C) |c| {
        throughNorm(dk[c * R ..][0..R], k[c * R ..][0..R], kn[c]);
        for (0..R) |r| for (0..D) |d| {
            dWk[r * D + d] += dk[c * R + r] * @as(f64, y[c * D + d]);
            dy[c * D + d] += @floatCast(dk[c * R + r] * @as(f64, Wk[r * D + d]));
        };
    }
}
