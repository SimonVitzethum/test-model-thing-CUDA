// Thin Zig layer over the MLX C API (mlx-c).
//
// Every array produced by an op is registered in a temporary pool and freed by
// `release(mark)`; long-lived arrays (weights, optimizer moments, stream state)
// are held through `keep`/`Owned`. MLX errors are fatal like CUDA_CHECK was.
const std = @import("std");
pub const c = @cImport({
    @cInclude("mlx/c/mlx.h");
});

pub const Array = c.mlx_array;
pub const Dtype = c.mlx_dtype;
pub const f32_ = c.MLX_FLOAT32;
pub const bf16 = c.MLX_BFLOAT16;
pub const f16_ = c.MLX_FLOAT16;
pub const i32_ = c.MLX_INT32;
pub const u32_ = c.MLX_UINT32;
pub const u16_ = c.MLX_UINT16;
pub const bool_ = c.MLX_BOOL;
pub const none: Array = .{ .ctx = null };

pub const gpa = std.heap.c_allocator;
var stream: c.mlx_stream = undefined;
var temps: std.ArrayList(Array) = .empty;

fn onError(msg: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    std.debug.print("MLX error: {s}\n", .{msg});
    std.process.exit(1);
}

pub fn init() void {
    stream = c.mlx_default_gpu_stream_new();
    c.mlx_set_error_handler(onError, null, null);
}

pub fn oom() noreturn {
    std.debug.print("out of host memory\n", .{});
    std.process.exit(1);
}
fn check(rc: c_int) void {
    if (rc != 0) {
        std.debug.print("MLX call failed\n", .{});
        std.process.exit(1);
    }
}
fn track(a: Array) Array {
    temps.append(gpa, a) catch oom();
    return a;
}

// ---- lifetime ----
pub fn mark() usize {
    return temps.items.len;
}
pub fn release(m: usize) void {
    for (temps.items[m..]) |a| _ = c.mlx_array_free(a);
    temps.shrinkRetainingCapacity(m);
}
/// New untracked handle to the same array (caller frees with `free`).
pub fn keep(a: Array) Array {
    var r = c.mlx_array_new();
    check(c.mlx_array_set(&r, a));
    return r;
}
pub fn free(a: Array) void {
    if (a.ctx != null) _ = c.mlx_array_free(a);
}
/// Replace a long-lived handle: frees the old one, keeps the new array.
pub fn assign(dst: *Array, a: Array) void {
    const n = keep(a);
    free(dst.*);
    dst.* = n;
}

// ---- construction ----
pub fn scalar(v: f32) Array {
    return track(c.mlx_array_new_float32(v));
}
pub fn scalarInt(v: i32) Array {
    return track(c.mlx_array_new_int(v));
}
pub fn fromSlice(comptime T: type, data: []const T, shp: []const i32) Array {
    const dt: Dtype = switch (T) {
        f32 => f32_,
        i32 => i32_,
        u32 => u32_,
        u16 => u16_,
        else => @compileError("unsupported element type"),
    };
    var n: usize = 1;
    for (shp) |s| n *= @intCast(s);
    std.debug.assert(n == data.len);
    return track(c.mlx_array_new_data(data.ptr, shp.ptr, @intCast(shp.len), dt));
}
/// bf16 array from raw bf16 bit patterns.
pub fn fromBf16Bits(bits: []const u16, shp: []const i32) Array {
    return view(fromSlice(u16, bits, shp), bf16);
}
pub fn zeros(shp: []const i32, dt: Dtype) Array {
    var r = c.mlx_array_new();
    check(c.mlx_zeros(&r, shp.ptr, shp.len, dt, stream));
    return track(r);
}
pub fn full(shp: []const i32, v: f32, dt: Dtype) Array {
    var r = c.mlx_array_new();
    check(c.mlx_full(&r, shp.ptr, shp.len, scalar(v), dt, stream));
    return track(r);
}
pub fn arange(start: f64, stop: f64, dt: Dtype) Array {
    var r = c.mlx_array_new();
    check(c.mlx_arange(&r, start, stop, 1, dt, stream));
    return track(r);
}

// ---- introspection / host transfer ----
pub fn shape(a: Array) []const i32 {
    const n = c.mlx_array_ndim(a);
    return c.mlx_array_shape(a)[0..n];
}
pub fn dim(a: Array, axis: i32) i32 {
    const s = shape(a);
    const i: usize = @intCast(if (axis < 0) @as(i32, @intCast(s.len)) + axis else axis);
    return s[i];
}
pub fn size(a: Array) usize {
    return c.mlx_array_size(a);
}
pub fn dtype(a: Array) Dtype {
    return c.mlx_array_dtype(a);
}
pub fn eval(arrays: []const Array) void {
    const v = c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = c.mlx_vector_array_free(v);
    check(c.mlx_eval(v));
}
pub fn eval1(a: Array) void {
    check(c.mlx_array_eval(a));
}
/// Copies a (any float dtype) as f32 in row-major order into dst.
pub fn toF32(a: Array, dst: []f32) void {
    const m = mark();
    defer release(m);
    const x = contiguous(astype(a, f32_));
    eval1(x);
    std.debug.assert(size(x) == dst.len);
    @memcpy(dst, c.mlx_array_data_float32(x)[0..dst.len]);
}
pub fn toI32(a: Array, dst: []i32) void {
    const m = mark();
    defer release(m);
    const x = contiguous(astype(a, i32_));
    eval1(x);
    std.debug.assert(size(x) == dst.len);
    @memcpy(dst, c.mlx_array_data_int32(x)[0..dst.len]);
}
/// Raw bf16 bit patterns of a bf16 array.
pub fn toBf16Bits(a: Array, dst: []u16) void {
    const m = mark();
    defer release(m);
    const x = contiguous(view(astype(a, bf16), u16_));
    eval1(x);
    std.debug.assert(size(x) == dst.len);
    @memcpy(dst, c.mlx_array_data_uint16(x)[0..dst.len]);
}
pub fn item(a: Array) f32 {
    var v: [1]f32 = undefined;
    toF32(a, &v);
    return v[0];
}
pub fn toVecF32(a: Array) []f32 {
    const v = gpa.alloc(f32, size(a)) catch oom();
    toF32(a, v);
    return v;
}

// ---- elementwise ----
const Bin = fn ([*c]Array, Array, Array, c.mlx_stream) callconv(.c) c_int;
const Un = fn ([*c]Array, Array, c.mlx_stream) callconv(.c) c_int;
inline fn bin(comptime f: Bin, a: Array, b: Array) Array {
    var r = c.mlx_array_new();
    check(f(&r, a, b, stream));
    return track(r);
}
inline fn un(comptime f: Un, a: Array) Array {
    var r = c.mlx_array_new();
    check(f(&r, a, stream));
    return track(r);
}
pub fn add(a: Array, b: Array) Array {
    return bin(c.mlx_add, a, b);
}
pub fn sub(a: Array, b: Array) Array {
    return bin(c.mlx_subtract, a, b);
}
pub fn mul(a: Array, b: Array) Array {
    return bin(c.mlx_multiply, a, b);
}
pub fn div(a: Array, b: Array) Array {
    return bin(c.mlx_divide, a, b);
}
pub fn maximum(a: Array, b: Array) Array {
    return bin(c.mlx_maximum, a, b);
}
pub fn equal(a: Array, b: Array) Array {
    return bin(c.mlx_equal, a, b);
}
pub fn greater(a: Array, b: Array) Array {
    return bin(c.mlx_greater, a, b);
}
pub fn greaterEqual(a: Array, b: Array) Array {
    return bin(c.mlx_greater_equal, a, b);
}
pub fn matmul(a: Array, b: Array) Array {
    return bin(c.mlx_matmul, a, b);
}
pub fn floorDivide(a: Array, b: Array) Array {
    return bin(c.mlx_floor_divide, a, b);
}
pub fn exp(a: Array) Array {
    return un(c.mlx_exp, a);
}
pub fn log(a: Array) Array {
    return un(c.mlx_log, a);
}
pub fn log1p(a: Array) Array {
    return un(c.mlx_log1p, a);
}
pub fn sqrt(a: Array) Array {
    return un(c.mlx_sqrt, a);
}
pub fn square(a: Array) Array {
    return un(c.mlx_square, a);
}
pub fn sigmoid(a: Array) Array {
    return un(c.mlx_sigmoid, a);
}
pub fn negative(a: Array) Array {
    return un(c.mlx_negative, a);
}
pub fn abs(a: Array) Array {
    return un(c.mlx_abs, a);
}
pub fn cos(a: Array) Array {
    return un(c.mlx_cos, a);
}
pub fn sin(a: Array) Array {
    return un(c.mlx_sin, a);
}
pub fn isfinite(a: Array) Array {
    return un(c.mlx_isfinite, a);
}
pub fn stopGradient(a: Array) Array {
    return un(c.mlx_stop_gradient, a);
}
pub fn silu(a: Array) Array {
    return mul(a, sigmoid(a));
}
pub fn addS(a: Array, v: f32) Array {
    return add(a, scalar(v));
}
pub fn mulS(a: Array, v: f32) Array {
    return mul(a, scalar(v));
}
pub fn where(cond: Array, x: Array, y: Array) Array {
    var r = c.mlx_array_new();
    check(c.mlx_where(&r, cond, x, y, stream));
    return track(r);
}
pub fn astype(a: Array, dt: Dtype) Array {
    var r = c.mlx_array_new();
    check(c.mlx_astype(&r, a, dt, stream));
    return track(r);
}
pub fn view(a: Array, dt: Dtype) Array {
    var r = c.mlx_array_new();
    check(c.mlx_view(&r, a, dt, stream));
    return track(r);
}
pub fn contiguous(a: Array) Array {
    var r = c.mlx_array_new();
    check(c.mlx_contiguous(&r, a, false, stream));
    return track(r);
}

// ---- reductions ----
pub fn sumAll(a: Array) Array {
    var r = c.mlx_array_new();
    check(c.mlx_sum(&r, a, false, stream));
    return track(r);
}
pub fn sum(a: Array, axis: i32, keepdims: bool) Array {
    var r = c.mlx_array_new();
    check(c.mlx_sum_axis(&r, a, axis, keepdims, stream));
    return track(r);
}
pub fn meanAll(a: Array) Array {
    var r = c.mlx_array_new();
    check(c.mlx_mean(&r, a, false, stream));
    return track(r);
}
pub fn max(a: Array, axis: i32, keepdims: bool) Array {
    var r = c.mlx_array_new();
    check(c.mlx_max_axis(&r, a, axis, keepdims, stream));
    return track(r);
}
pub fn logsumexp(a: Array, axis: i32, keepdims: bool) Array {
    var r = c.mlx_array_new();
    check(c.mlx_logsumexp_axis(&r, a, axis, keepdims, stream));
    return track(r);
}
pub fn softmax(a: Array, axis: i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_softmax_axis(&r, a, axis, true, stream));
    return track(r);
}
pub fn argsort(a: Array, axis: i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_argsort_axis(&r, a, axis, stream));
    return track(r);
}

// ---- shape ----
pub fn reshape(a: Array, shp: []const i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_reshape(&r, a, shp.ptr, shp.len, stream));
    return track(r);
}
pub fn transpose(a: Array, axes: []const i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_transpose_axes(&r, a, axes.ptr, axes.len, stream));
    return track(r);
}
pub fn swapaxes(a: Array, x: i32, y: i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_swapaxes(&r, a, x, y, stream));
    return track(r);
}
pub fn expandDims(a: Array, axis: i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_expand_dims(&r, a, axis, stream));
    return track(r);
}
pub fn broadcastTo(a: Array, shp: []const i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_broadcast_to(&r, a, shp.ptr, shp.len, stream));
    return track(r);
}
/// a[..., start:stop, ...] along one axis.
pub fn sliceAxis(a: Array, axis: i32, start: i32, stop: i32) Array {
    const s = shape(a);
    var lo: [8]i32 = @splat(0);
    var hi: [8]i32 = undefined;
    var st: [8]i32 = @splat(1);
    for (s, 0..) |n, i| hi[i] = n;
    const ax: usize = @intCast(if (axis < 0) @as(i32, @intCast(s.len)) + axis else axis);
    lo[ax] = start;
    hi[ax] = stop;
    var r = c.mlx_array_new();
    check(c.mlx_slice(&r, a, &lo, s.len, &hi, s.len, &st, s.len, stream));
    return track(r);
}
pub fn concat(arrays: []const Array, axis: i32) Array {
    const v = c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = c.mlx_vector_array_free(v);
    var r = c.mlx_array_new();
    check(c.mlx_concatenate_axis(&r, v, axis, stream));
    return track(r);
}
pub fn stack(arrays: []const Array, axis: i32) Array {
    const v = c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = c.mlx_vector_array_free(v);
    var r = c.mlx_array_new();
    check(c.mlx_stack_axis(&r, v, axis, stream));
    return track(r);
}

// ---- indexing ----
pub fn take(a: Array, idx: Array, axis: i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_take_axis(&r, a, idx, axis, stream));
    return track(r);
}
pub fn takeAlong(a: Array, idx: Array, axis: i32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_take_along_axis(&r, a, idx, axis, stream));
    return track(r);
}
pub fn gatherMm(a: Array, b: Array, rhs: Array, sorted: bool) Array {
    var r = c.mlx_array_new();
    check(c.mlx_gather_mm(&r, a, b, none, rhs, sorted, stream));
    return track(r);
}

// ---- fused ----
pub fn layerNorm(x: Array, w: Array, b: Array, eps: f32) Array {
    var r = c.mlx_array_new();
    check(c.mlx_fast_layer_norm(&r, x, w, b, eps, stream));
    return track(r);
}

// ---- vectors ----
pub fn vecGet(v: c.mlx_vector_array, i: usize) Array {
    var r = c.mlx_array_new();
    check(c.mlx_vector_array_get(&r, v, i));
    return track(r);
}
pub fn vecToSlice(v: c.mlx_vector_array) []Array {
    const n = c.mlx_vector_array_size(v);
    const out = gpa.alloc(Array, n) catch oom();
    for (out, 0..) |*o, i| o.* = vecGet(v, i);
    return out;
}

// ---- Metal kernels ----
pub const Kernel = struct {
    k: c.mlx_fast_metal_kernel,

    pub fn init(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) Kernel {
        const in = c.mlx_vector_string_new();
        defer _ = c.mlx_vector_string_free(in);
        for (inputs) |s| check(c.mlx_vector_string_append_value(in, s));
        const out = c.mlx_vector_string_new();
        defer _ = c.mlx_vector_string_free(out);
        for (outputs) |s| check(c.mlx_vector_string_append_value(out, s));
        return .{ .k = c.mlx_fast_metal_kernel_new(name, in, out, source, header, true, false) };
    }

    pub const Out = struct { shape: []const i32, dtype: Dtype };
    pub const Tmpl = struct { name: [*:0]const u8, value: i32 };

    /// Launches `threads` threads in a 1-D grid; `group` fixes the threadgroup
    /// size (0 = pick one). Results are tracked arrays.
    pub fn call(self: Kernel, inputs: []const Array, outputs: []const Out, templates: []const Tmpl, threads: usize, group: usize, results: []Array) void {
        const cfg = c.mlx_fast_metal_kernel_config_new();
        defer c.mlx_fast_metal_kernel_config_free(cfg);
        for (outputs) |o| check(c.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
        for (templates) |t| check(c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.value));
        const tg: usize = if (group > 0) group else @min(256, @max(threads, 1));
        const grid = (@max(threads, 1) + tg - 1) / tg * tg;
        check(c.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(grid), 1, 1));
        check(c.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(tg), 1, 1));
        const in = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = c.mlx_vector_array_free(in);
        var res = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(res);
        check(c.mlx_fast_metal_kernel_apply(&res, self.k, in, cfg, stream));
        std.debug.assert(c.mlx_vector_array_size(res) == results.len);
        for (results, 0..) |*r, i| r.* = vecGet(res, i);
    }
};

// ---- transforms ----
/// Values of a function and gradients w.r.t. the selected inputs.
pub const ValueAndGrad = struct { values: []Array, grads: []Array };

/// Function of arrays -> arrays; `ctx` is passed through untouched.
pub const Fn = *const fn (ctx: *anyopaque, inputs: []const Array) []Array;
/// VJP rule: (primals, cotangents, outputs) -> one cotangent per primal.
pub const VjpFn = *const fn (ctx: *anyopaque, primals: []const Array, cotangents: []const Array, outputs: []const Array) []Array;

const Payload = struct { f: Fn, vjp: ?VjpFn = null, ctx: *anyopaque };

fn setResult(res: [*c]c.mlx_vector_array, out: []Array) void {
    check(c.mlx_vector_array_set_data(res, out.ptr, out.len));
    gpa.free(out);
}
fn trampoline(res: [*c]c.mlx_vector_array, in: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const p: *Payload = @ptrCast(@alignCast(payload.?));
    const args = vecToSlice(in);
    defer gpa.free(args);
    setResult(res, p.f(p.ctx, args));
    return 0;
}
fn vjpTrampoline(res: [*c]c.mlx_vector_array, primals: c.mlx_vector_array, cots: c.mlx_vector_array, outs: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const p: *Payload = @ptrCast(@alignCast(payload.?));
    const a = vecToSlice(primals);
    defer gpa.free(a);
    const b = vecToSlice(cots);
    defer gpa.free(b);
    const o = vecToSlice(outs);
    defer gpa.free(o);
    setResult(res, p.vjp.?(p.ctx, a, b, o));
    return 0;
}
fn freePayload(p: ?*anyopaque) callconv(.c) void {
    gpa.destroy(@as(*Payload, @ptrCast(@alignCast(p.?))));
}

/// A closure with a custom VJP rule. Build once and reuse.
pub const CustomFn = struct {
    cl: c.mlx_closure,

    pub fn init(f: Fn, vjp: VjpFn, ctx: *anyopaque) CustomFn {
        const p = gpa.create(Payload) catch oom();
        p.* = .{ .f = f, .vjp = vjp, .ctx = ctx };
        const q = gpa.create(Payload) catch oom();
        q.* = p.*;
        const base = c.mlx_closure_new_func_payload(trampoline, p, freePayload);
        defer _ = c.mlx_closure_free(base);
        const rule = c.mlx_closure_custom_new_func_payload(vjpTrampoline, q, freePayload);
        defer _ = c.mlx_closure_custom_free(rule);
        var cl = c.mlx_closure_new();
        check(c.mlx_custom_vjp(&cl, base, rule));
        return .{ .cl = cl };
    }
    pub fn apply(self: CustomFn, inputs: []const Array) []Array {
        const in = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = c.mlx_vector_array_free(in);
        var res = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(res);
        check(c.mlx_closure_apply(&res, self.cl, in));
        return vecToSlice(res);
    }
};

/// Values of f(inputs) and gradients of values[0] w.r.t. inputs[argnums].
/// Both slices are allocated with gpa; the arrays are tracked.
pub fn valueAndGrad(f: Fn, ctx: *anyopaque, inputs: []const Array, argnums: []const i32) ValueAndGrad {
    var p = Payload{ .f = f, .ctx = ctx };
    const cl = c.mlx_closure_new_func_payload(trampoline, &p, null);
    defer _ = c.mlx_closure_free(cl);
    var vag = c.mlx_closure_value_and_grad_new();
    defer _ = c.mlx_closure_value_and_grad_free(vag);
    check(c.mlx_value_and_grad(&vag, cl, argnums.ptr, argnums.len));
    const in = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = c.mlx_vector_array_free(in);
    var vals = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(vals);
    var grads = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(grads);
    check(c.mlx_closure_value_and_grad_apply(&vals, &grads, vag, in));
    return .{ .values = vecToSlice(vals), .grads = vecToSlice(grads) };
}

pub fn synchronize() void {
    check(c.mlx_synchronize(stream));
}

const GradPayload = struct { f: Fn, ctx: *anyopaque, argnums: []i32 };

fn gradTrampoline(res: [*c]c.mlx_vector_array, in: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const p: *GradPayload = @ptrCast(@alignCast(payload.?));
    var inner = Payload{ .f = p.f, .ctx = p.ctx };
    const cl = c.mlx_closure_new_func_payload(trampoline, &inner, null);
    defer _ = c.mlx_closure_free(cl);
    var vag = c.mlx_closure_value_and_grad_new();
    defer _ = c.mlx_closure_value_and_grad_free(vag);
    check(c.mlx_value_and_grad(&vag, cl, p.argnums.ptr, p.argnums.len));
    var vals = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(vals);
    var grads = c.mlx_vector_array_new();
    defer _ = c.mlx_vector_array_free(grads);
    check(c.mlx_closure_value_and_grad_apply(&vals, &grads, vag, in));
    const v = vecToSlice(vals);
    defer gpa.free(v);
    const g = vecToSlice(grads);
    defer gpa.free(g);
    const out = gpa.alloc(Array, v.len + g.len) catch oom();
    @memcpy(out[0..v.len], v);
    @memcpy(out[v.len..], g);
    setResult(res, out);
    return 0;
}
fn freeGradPayload(p: ?*anyopaque) callconv(.c) void {
    const q: *GradPayload = @ptrCast(@alignCast(p.?));
    gpa.free(q.argnums);
    gpa.destroy(q);
}

/// Values and gradients as ONE compiled function (`mx.compile(value_and_grad(f))`):
/// the backward graph is fused too, which differentiating a compiled function
/// alone does not do. Returns values ++ gradients.
pub const CompiledGrad = struct {
    cl: c.mlx_closure,
    ngrads: usize,

    pub fn init(f: Fn, ctx: *anyopaque, argnums: []const i32) CompiledGrad {
        const p = gpa.create(GradPayload) catch oom();
        p.* = .{ .f = f, .ctx = ctx, .argnums = gpa.dupe(i32, argnums) catch oom() };
        const base = c.mlx_closure_new_func_payload(gradTrampoline, p, freeGradPayload);
        defer _ = c.mlx_closure_free(base);
        var cl = c.mlx_closure_new();
        check(c.mlx_compile(&cl, base, false));
        return .{ .cl = cl, .ngrads = argnums.len };
    }
    /// Values of f, then one gradient per argnum (all tracked; caller frees the slice).
    pub fn apply(self: CompiledGrad, inputs: []const Array) ValueAndGrad {
        const in = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = c.mlx_vector_array_free(in);
        var res = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(res);
        check(c.mlx_closure_apply(&res, self.cl, in));
        const all = vecToSlice(res);
        return .{ .values = all[0 .. all.len - self.ngrads], .grads = all[all.len - self.ngrads ..] };
    }
};

/// Compiles a function of arrays; MLX fuses its elementwise chains and reuses
/// the plan while the input shapes stay the same.
pub const Compiled = struct {
    cl: c.mlx_closure,

    pub fn init(f: Fn, ctx: *anyopaque) Compiled {
        const p = gpa.create(Payload) catch oom();
        p.* = .{ .f = f, .ctx = ctx };
        const base = c.mlx_closure_new_func_payload(trampoline, p, freePayload);
        defer _ = c.mlx_closure_free(base);
        var cl = c.mlx_closure_new();
        check(c.mlx_compile(&cl, base, false));
        return .{ .cl = cl };
    }
    /// Values and gradients of the compiled function (same contract as valueAndGrad).
    pub fn valueAndGrad(self: Compiled, inputs: []const Array, argnums: []const i32) ValueAndGrad {
        var vag = c.mlx_closure_value_and_grad_new();
        defer _ = c.mlx_closure_value_and_grad_free(vag);
        check(c.mlx_value_and_grad(&vag, self.cl, argnums.ptr, argnums.len));
        const in = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = c.mlx_vector_array_free(in);
        var vals = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(vals);
        var grads = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(grads);
        check(c.mlx_closure_value_and_grad_apply(&vals, &grads, vag, in));
        return .{ .values = vecToSlice(vals), .grads = vecToSlice(grads) };
    }
    pub fn apply(self: Compiled, inputs: []const Array) []Array {
        const in = c.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = c.mlx_vector_array_free(in);
        var res = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(res);
        check(c.mlx_closure_apply(&res, self.cl, in));
        return vecToSlice(res);
    }
};
