//! The V3 checkpoint format (port of src/checkpoint.h). Written byte for byte
//! as the C++ build writes it: a magic word, the configuration text, the
//! training progress, every parameter with its Adam moments, the streaming
//! state and a trailing checksum over everything before it.
const std = @import("std");
const gpu = @import("gpu.zig");
const cfgmod = @import("config.zig");
const model = @import("model.zig");

pub const Error = error{Checkpoint};

const HASH_INIT: u64 = 14695981039346656037;
const MAGIC: u64 = 0x33544b5043544d54; // TMTCPKT3

var msg_buf: [256]u8 = undefined;
var msg_len: usize = 0;
pub fn lastError() []const u8 {
    return msg_buf[0..msg_len];
}
fn fail(comptime fmt: []const u8, args: anytype) Error {
    const s = std.fmt.bufPrint(&msg_buf, fmt, args) catch msg_buf[0..];
    msg_len = s.len;
    return error.Checkpoint;
}

pub fn hashBytes(h: u64, data: []const u8) u64 {
    var x = h;
    for (data) |b| x = (x ^ b) *% 1099511628211;
    return x;
}

pub const Progress = extern struct {
    step: u64 = 0,
    cursor: u64 = 0,
    epoch: u64 = 0,
    carried: u64 = 0,
    data_size: u64 = 0,
    data_hash: u64 = 0,
};

/// One open checkpoint, reading or writing, hashing everything that passes.
const File = struct {
    file: std.Io.File,
    io: std.Io,
    writing: bool,
    hash: u64 = HASH_INIT,
    offset: u64 = 0,
    gpa: std.mem.Allocator,

    fn bytes(f: *File, buf: []u8) !void {
        if (f.writing) {
            f.file.writePositionalAll(f.io, buf, f.offset) catch return fail("checkpoint I/O failed or truncated", .{});
        } else {
            const got = f.file.readPositionalAll(f.io, buf, f.offset) catch
                return fail("checkpoint I/O failed or truncated", .{});
            if (got != buf.len) return fail("checkpoint I/O failed or truncated", .{});
        }
        f.hash = hashBytes(f.hash, buf);
        f.offset += buf.len;
    }
    fn scalar(f: *File, v: anytype) !void {
        try f.bytes(std.mem.asBytes(v));
    }
    /// Device memory through a bounded host staging buffer.
    fn device(f: *File, ptr: *anyopaque, size: usize) !void {
        const buf = try f.gpa.alloc(u8, @min(size, 1 << 20));
        defer f.gpa.free(buf);
        var at: usize = 0;
        while (at < size) : (at += buf.len) {
            const n = @min(buf.len, size - at);
            const dev: *anyopaque = @ptrFromInt(@intFromPtr(ptr) + at);
            if (f.writing) try gpu.download(buf[0..n], dev);
            try f.bytes(buf[0..n]);
            if (!f.writing) try gpu.upload(dev, buf[0..n]);
        }
    }
    fn skip(f: *File, n: u64) void {
        f.offset += n;
    }
    fn finish(f: *File) !void {
        var checksum = f.hash;
        if (f.writing) {
            f.file.writePositionalAll(f.io, std.mem.asBytes(&checksum), f.offset) catch
                return fail("checkpoint flush failed", .{});
            f.file.sync(f.io) catch return fail("checkpoint flush failed", .{});
        } else {
            var expected: u64 = 0;
            const got = f.file.readPositionalAll(f.io, std.mem.asBytes(&expected), f.offset) catch 0;
            const end = (f.file.stat(f.io) catch return fail("checkpoint checksum/trailer mismatch", .{})).size;
            if (got != 8 or expected != checksum or end != f.offset + 8)
                return fail("checkpoint checksum/trailer mismatch", .{});
        }
    }
};

/// The whole file must hash to its trailer before anything is loaded.
pub fn verify(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return fail("cannot read checkpoint", .{});
    defer file.close(io);
    const size = (file.stat(io) catch return fail("cannot read checkpoint", .{})).size;
    if (size < 16) return fail("checkpoint truncated", .{});
    const buf = try gpa.alloc(u8, 65536);
    defer gpa.free(buf);
    var remaining = size - 8;
    var at: u64 = 0;
    var h = HASH_INIT;
    while (remaining != 0) {
        const n = @min(remaining, buf.len);
        const got = file.readPositionalAll(io, buf[0..n], at) catch return fail("checkpoint read failed", .{});
        if (got != n) return fail("checkpoint read failed", .{});
        h = hashBytes(h, buf[0..n]);
        remaining -= n;
        at += n;
    }
    var expected: u64 = 0;
    const got = file.readPositionalAll(io, std.mem.asBytes(&expected), at) catch 0;
    if (got != 8 or h != expected)
        return fail("checkpoint checksum mismatch (old V2 files are unsupported)", .{});
}

/// Magic word and configuration text. When reading, the stored text must be
/// exactly what the decoded configuration produces again, except for keys
/// added later with a behavior-preserving default.
fn header(f: *File, cfg: *cfgmod.Cfg) !void {
    var magic = MAGIC;
    try f.scalar(&magic);
    if (magic != MAGIC) return fail("unsupported checkpoint format/architecture", .{});
    const written = try cfgmod.text(f.gpa, cfg.*);
    defer f.gpa.free(written);
    var length: u64 = written.len;
    try f.scalar(&length);
    if (length == 0 or length > 16384) return fail("invalid checkpoint configuration length", .{});
    const text = try f.gpa.alloc(u8, length);
    defer f.gpa.free(text);
    if (f.writing) @memcpy(text, written);
    try f.bytes(text);
    if (f.writing) return;

    var decoded = cfgmod.Cfg{};
    var stored_keys: std.ArrayList(u8) = .empty;
    defer stored_keys.deinit(f.gpa);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse return fail("invalid checkpoint configuration", .{});
        cfgmod.set(&decoded, line[0..eq], line[eq + 1 ..]) catch return fail("{s}", .{cfgmod.lastError()});
        try stored_keys.appendSlice(f.gpa, " ");
        try stored_keys.appendSlice(f.gpa, line[0..eq]);
        try stored_keys.appendSlice(f.gpa, " ");
    }
    cfgmod.validate(decoded) catch return fail("{s}", .{cfgmod.lastError()});
    const canonical = try cfgmod.text(f.gpa, decoded);
    defer f.gpa.free(canonical);
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(f.gpa);
    var want = std.mem.splitScalar(u8, canonical, '\n');
    while (want.next()) |line| {
        if (line.len == 0) continue;
        const key = line[0 .. std.mem.indexOfScalar(u8, line, '=') orelse line.len];
        var needle_buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&needle_buf, " {s} ", .{key}) catch return fail("checkpoint configuration schema mismatch", .{});
        if (std.mem.indexOf(u8, stored_keys.items, needle) != null) {
            try expected.appendSlice(f.gpa, line);
            try expected.append(f.gpa, '\n');
        } else if (!std.mem.eql(u8, key, "traces") and !std.mem.eql(u8, key, "trace_decay") and
            !std.mem.eql(u8, key, "docsep") and !std.mem.eql(u8, key, "dialog") and
            !std.mem.startsWith(u8, key, "mem") and !std.mem.startsWith(u8, key, "mtp") and
            !std.mem.startsWith(u8, key, "muon") and !std.mem.startsWith(u8, key, "mup") and !std.mem.startsWith(u8, key, "patch") and !std.mem.eql(u8, key, "accum"))
        {
            return fail("checkpoint configuration schema mismatch", .{});
        }
    }
    if (!std.mem.eql(u8, expected.items, text)) return fail("checkpoint configuration schema mismatch", .{});
    cfg.* = decoded;
}

/// The configuration a checkpoint was written with.
pub fn readConfig(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !cfgmod.Cfg {
    try verify(gpa, io, path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return fail("cannot open checkpoint: {s}", .{path});
    defer file.close(io);
    var f = File{ .file = file, .io = io, .writing = false, .gpa = gpa };
    var cfg = cfgmod.Cfg{};
    try header(&f, &cfg);
    return cfg;
}

/// Everything that fixes the weight shapes or the state layout; the runtime
/// keys (batch, seqlen, schedule, losses) may differ when loading.
fn requireSameModel(a: cfgmod.Cfg, b: cfgmod.Cfg) !void {
    const same = a.dim == b.dim and a.layers == b.layers and a.experts == b.experts and
        a.topk == b.topk and a.gated == b.gated and a.half_min == b.half_min and
        a.half_max == b.half_max and a.mla == b.mla and a.mla_heads == b.mla_heads and
        a.mla_dh == b.mla_dh and a.mla_L == b.mla_L and a.mla_R == b.mla_R and
        a.mla_cache == b.mla_cache and a.mla_every == b.mla_every and a.mla_cc == b.mla_cc and
        a.mla_theta == b.mla_theta and a.mtp == b.mtp and a.patch == b.patch and
        a.patch_lo == b.patch_lo and a.patch_hi == b.patch_hi and a.mem == b.mem and
        (a.mem == 0 or (a.mem_len == b.mem_len and a.mem_heads == b.mem_heads and
            a.mem_dh == b.mem_dh and a.mem_every == b.mem_every and a.mem_rdim == b.mem_rdim));
    if (!same) return fail("checkpoint architecture differs; use a new path for a new experiment", .{});
}

fn payload(f: *File, m: *model.Model, state: *model.StreamState, progress: *Progress) !void {
    var stored = m.c;
    try header(f, &stored);
    try requireSameModel(stored, m.c);
    inline for (.{ "step", "cursor", "epoch", "carried", "data_size", "data_hash" }) |name|
        try f.scalar(&@field(progress, name));
    if (progress.step >= 2147483647 or progress.cursor >= progress.data_size or
        progress.carried > std.math.maxInt(i64))
        return fail("invalid checkpoint progress", .{});
    var count: u64 = m.store.values.items.len;
    try f.scalar(&count);
    if (count != m.store.values.items.len) return fail("checkpoint parameter count mismatch", .{});
    for (m.store.values.items) |p| {
        var n: u64 = @intCast(p.n);
        try f.scalar(&n);
        if (n != @as(u64, @intCast(p.n))) return fail("checkpoint parameter shape mismatch", .{});
        try f.device(p.master, @intCast(n * 4));
        try f.device(p.m, @intCast(n * 4));
        try f.device(p.v, @intCast(n * 4));
        if (!f.writing) {
            const g: u32 = @intCast((n + 255) / 256);
            try (try m.kernels.get("copy_bf16")).launch(g, 256, .{ p.master, p.work, p.n });
        }
    }
    try f.scalar(&state.position);
    if (state.position < 0 or @as(u64, @intCast(state.position)) != progress.carried)
        return fail("checkpoint stream position mismatch", .{});
    const bd: usize = @as(usize, @intCast(m.c.batch)) * @as(usize, @intCast(m.c.dim)) * 4;
    for (0..@intCast(m.c.layers)) |l| {
        try f.device(state.carry[l], bd);
        if (!m.ML[l].use) continue;
        const cache = &state.cache[l];
        try f.scalar(&cache.head);
        try f.scalar(&cache.base0);
        if (cache.head < 0 or cache.head > cache.Cmax or cache.base0 < 0 or
            cache.base0 + cache.head != state.position)
            return fail("invalid checkpoint cache position", .{});
        // Only the initialized part of the cache is stored, per stream.
        for (0..@intCast(m.c.batch)) |b| {
            const lat = cache.lat + b * @as(usize, @intCast(cache.Cmax)) * @as(usize, @intCast(cache.L));
            const kr = cache.kr + b * @as(usize, @intCast(cache.Cmax)) * @as(usize, @intCast(cache.R));
            try f.device(lat, @intCast(cache.head * cache.L * 2));
            try f.device(kr, @intCast(cache.head * cache.R * 2));
        }
    }
    if (m.c.traces != 0) { // absent for traces=0, so older V3 files still load
        for (0..@intCast(m.c.layers)) |l| {
            try f.device(state.tdec[l], bd);
            try f.device(state.tgate[l], bd);
        }
        try f.device(state.temb.?, 256 * bd);
    }
    try f.finish();
}

pub fn save(gpa: std.mem.Allocator, io: std.Io, path: []const u8, m: *model.Model,
            state: *model.StreamState, progress: *Progress) !void {
    // Written to a temporary file and renamed, so a crash never truncates the
    // checkpoint that already exists.
    const tmp = try std.fmt.allocPrint(gpa, "{s}.tmp.{d}", .{ path, std.os.linux.getpid() });
    defer gpa.free(tmp);
    const file = std.Io.Dir.cwd().createFile(io, tmp, .{}) catch return fail("cannot open checkpoint: {s}", .{tmp});
    var f = File{ .file = file, .io = io, .writing = true, .gpa = gpa };
    payload(&f, m, state, progress) catch |e| {
        file.close(io);
        std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
        return e;
    };
    file.close(io);
    std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), path, io) catch {
        std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
        return fail("checkpoint rename failed", .{});
    };
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8, m: *model.Model,
            state: *model.StreamState, progress: *Progress) !void {
    try verify(gpa, io, path); // before any model or state is touched
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return fail("cannot open checkpoint: {s}", .{path});
    defer file.close(io);
    var f = File{ .file = file, .io = io, .writing = false, .gpa = gpa };
    try payload(&f, m, state, progress);
}

/// Weights only: the moments and the streaming state are skipped, and the file
/// must end exactly after the checksum, which proves the layout was right.
pub fn loadWeights(gpa: std.mem.Allocator, io: std.Io, path: []const u8, m: *model.Model,
                   file_cfg: cfgmod.Cfg) !void {
    try verify(gpa, io, path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return fail("cannot open checkpoint: {s}", .{path});
    defer file.close(io);
    var f = File{ .file = file, .io = io, .writing = false, .gpa = gpa };
    var stored = m.c;
    try header(&f, &stored);
    try requireSameModel(stored, m.c);
    var pr = Progress{};
    inline for (.{ "step", "cursor", "epoch", "carried", "data_size", "data_hash" }) |name|
        try f.scalar(&@field(pr, name));
    var count: u64 = 0;
    try f.scalar(&count);
    if (count != m.store.values.items.len) return fail("parameter count mismatch", .{});
    for (m.store.values.items) |p| {
        var n: u64 = 0;
        try f.scalar(&n);
        if (n != @as(u64, @intCast(p.n))) return fail("parameter shape mismatch", .{});
        try f.device(p.master, @intCast(n * 4));
        f.skip(n * 4 * 2); // the two Adam moments
        const g: u32 = @intCast((n + 255) / 256);
        try (try m.kernels.get("copy_bf16")).launch(g, 256, .{ p.master, p.work, p.n });
    }
    var spos: i64 = 0;
    try f.scalar(&spos); // the stream state is rebuilt, so its position is ignored
    const fB: u64 = @intCast(file_cfg.batch);
    const fD: u64 = @intCast(file_cfg.dim);
    for (0..@intCast(file_cfg.layers)) |l| {
        f.skip(fB * fD * 4);
        const has_cache = (l < @as(usize, @intCast(m.c.layers)) and m.ML[l].use) or
            (file_cfg.mla != 0 and @rem(@as(i32, @intCast(l)), file_cfg.mla_every) == 0);
        if (!has_cache) continue;
        var head: i64 = 0;
        var base0: i64 = 0;
        try f.scalar(&head);
        try f.scalar(&base0);
        const per_stream: u64 = @intCast(head * (file_cfg.mla_L + file_cfg.mla_R) * 2);
        f.skip(fB * per_stream);
    }
    if (file_cfg.traces != 0) // training state only
        f.skip((2 * @as(u64, @intCast(file_cfg.layers)) + 256) * fB * fD * 4);
    const end = (file.stat(io) catch return fail("seek failed", .{})).size;
    if (end - f.offset != 8) return fail("checkpoint layout mismatch", .{});
    try gpu.synchronize();
}
