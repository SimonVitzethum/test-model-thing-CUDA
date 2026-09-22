// Checkpoint format V3 (TMTCPKT3), byte-compatible with the CUDA build:
//   magic | config text | progress | per parameter (n, master, m, v) |
//   stream position | per layer carry [+ MLA cache head/base0/contents] |
//   [traces] | FNV-1a checksum of everything before it.
// Written through a temporary file with fsync and atomic rename; the checksum
// is verified before anything is restored.
const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const config = @import("config.zig");
const u = @import("util.zig");
const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("unistd.h");
});
const Cfg = config.Cfg;

pub const HASH_INIT: u64 = 14695981039346656037;
pub fn hashBytes(h0: u64, bytes: []const u8) u64 {
    var h = h0;
    for (bytes) |b| h = (h ^ b) *% 1099511628211;
    return h;
}

pub const Progress = struct {
    step: u64 = 0,
    cursor: u64 = 0,
    epoch: u64 = 0,
    carried: u64 = 0,
    data_size: u64 = 0,
    data_hash: u64 = 0,
};

pub const Error = error{ Io, Corrupt, Mismatch };
pub var last_error: []const u8 = "";
fn fail(comptime msg: []const u8, e: Error) Error {
    last_error = msg;
    return e;
}

const MAGIC: u64 = 0x33544b5043544d54;

const File = struct {
    f: *c.FILE,
    writing: bool,
    hash: u64 = HASH_INIT,

    fn open(path: [:0]const u8, write: bool) Error!File {
        const f = c.fopen(path.ptr, if (write) "wb" else "rb") orelse return fail("cannot open checkpoint", error.Io);
        return .{ .f = f, .writing = write };
    }
    fn close(self: *File) void {
        _ = c.fclose(self.f);
    }
    fn bytes(self: *File, buf: []u8) Error!void {
        const got = if (self.writing) c.fwrite(buf.ptr, 1, buf.len, self.f) else c.fread(buf.ptr, 1, buf.len, self.f);
        if (got != buf.len) return fail("checkpoint I/O failed or truncated", error.Io);
        self.hash = hashBytes(self.hash, buf);
    }
    fn scalar(self: *File, comptime T: type, v: *T) Error!void {
        try self.bytes(std.mem.asBytes(v));
    }
    fn f32s(self: *File, v: []f32) Error!void {
        try self.bytes(std.mem.sliceAsBytes(v));
    }
    fn u16s(self: *File, v: []u16) Error!void {
        try self.bytes(std.mem.sliceAsBytes(v));
    }
    fn skip(self: *File, n: u64) Error!void {
        if (c.fseeko(self.f, @intCast(n), c.SEEK_CUR) != 0) return fail("checkpoint skip failed", error.Io);
    }
    fn finish(self: *File) Error!void {
        var checksum = self.hash;
        if (self.writing) {
            if (c.fwrite(&checksum, 8, 1, self.f) != 1 or c.fflush(self.f) != 0 or c.fsync(c.fileno(self.f)) != 0)
                return fail("checkpoint flush failed", error.Io);
        } else {
            var expected: u64 = 0;
            if (c.fread(&expected, 8, 1, self.f) != 1 or expected != checksum or c.fgetc(self.f) != c.EOF)
                return fail("checkpoint checksum/trailer mismatch", error.Corrupt);
        }
    }
};

/// Streams the file and compares the trailing checksum (before any restore).
pub fn verify(path: [:0]const u8) Error!void {
    const f = c.fopen(path.ptr, "rb") orelse return fail("cannot read checkpoint", error.Io);
    defer _ = c.fclose(f);
    if (c.fseeko(f, 0, c.SEEK_END) != 0) return fail("checkpoint read failed", error.Io);
    const total = c.ftello(f);
    if (total < 16) return fail("checkpoint truncated", error.Corrupt);
    _ = c.fseeko(f, 0, c.SEEK_SET);
    var remaining: u64 = @intCast(total - 8);
    var h = HASH_INIT;
    var buf: [1 << 16]u8 = undefined;
    while (remaining > 0) {
        const n: usize = @intCast(@min(remaining, buf.len));
        if (c.fread(&buf, 1, n, f) != n) return fail("checkpoint read failed", error.Io);
        h = hashBytes(h, buf[0..n]);
        remaining -= n;
    }
    var expected: u64 = 0;
    if (c.fread(&expected, 8, 1, f) != 1 or h != expected)
        return fail("checkpoint checksum mismatch (old V2 files are unsupported)", error.Corrupt);
}

fn header(io: *File, cfg: *Cfg) Error!void {
    var magic: u64 = MAGIC;
    try io.scalar(u64, &magic);
    if (magic != MAGIC) return fail("unsupported checkpoint format/architecture", error.Corrupt);
    const text = config.text(mx.gpa, cfg.*);
    defer mx.gpa.free(text);
    var length: u64 = text.len;
    try io.scalar(u64, &length);
    if (length == 0 or length > 16384) return fail("invalid checkpoint configuration length", error.Corrupt);
    if (io.writing) return io.bytes(text);
    const stored = mx.gpa.alloc(u8, length) catch mx.oom();
    defer mx.gpa.free(stored);
    try io.bytes(stored);
    var decoded = Cfg{};
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(mx.gpa);
    var lines = std.mem.splitScalar(u8, stored, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const kv = u.splitKV(line) orelse return fail("invalid checkpoint configuration", error.Corrupt);
        config.set(&decoded, kv.key, kv.value) catch return fail("invalid checkpoint configuration", error.Corrupt);
        keys.append(mx.gpa, kv.key) catch mx.oom();
    }
    config.validate(decoded) catch return fail("invalid checkpoint configuration", error.Corrupt);
    // Keys added later with a behavior-preserving default may be absent in
    // older V3 files; every other key must round-trip exactly.
    const canonical = config.text(mx.gpa, decoded);
    defer mx.gpa.free(canonical);
    var expected: std.ArrayList(u8) = .empty;
    defer expected.deinit(mx.gpa);
    var cl = std.mem.splitScalar(u8, canonical, '\n');
    while (cl.next()) |line| {
        if (line.len == 0) continue;
        const key = line[0..std.mem.indexOfScalar(u8, line, '=').?];
        var present = false;
        for (keys.items) |k| present = present or std.mem.eql(u8, k, key);
        if (present) {
            expected.appendSlice(mx.gpa, line) catch mx.oom();
            expected.append(mx.gpa, '\n') catch mx.oom();
        } else if (!(std.mem.eql(u8, key, "traces") or std.mem.eql(u8, key, "trace_decay") or
            std.mem.eql(u8, key, "docsep") or std.mem.eql(u8, key, "dialog")))
            return fail("checkpoint configuration schema mismatch", error.Corrupt);
    }
    if (!std.mem.eql(u8, expected.items, stored)) return fail("checkpoint configuration schema mismatch", error.Corrupt);
    cfg.* = decoded;
}

/// Configuration stored in a checkpoint (after verifying its checksum).
pub fn readConfig(path: [:0]const u8) Error!Cfg {
    try verify(path);
    var io = try File.open(path, false);
    defer io.close();
    var cfg = Cfg{};
    try header(&io, &cfg);
    return cfg;
}

fn progressIo(io: *File, p: *Progress) Error!void {
    inline for (.{ "step", "cursor", "epoch", "carried", "data_size", "data_hash" }) |f| try io.scalar(u64, &@field(p, f));
}

fn payload(io: *File, m: *model.Model, st: *model.State, p: *Progress) Error!void {
    var stored = m.c;
    try header(io, &stored);
    if (!config.sameModel(stored, m.c)) return fail("checkpoint architecture differs; use a new path for a new experiment", error.Mismatch);
    try progressIo(io, p);
    if (p.step >= std.math.maxInt(i32) or p.cursor >= p.data_size or p.carried > std.math.maxInt(i64))
        return fail("invalid checkpoint progress", error.Corrupt);
    var count: u64 = m.nparams();
    try io.scalar(u64, &count);
    if (count != m.nparams()) return fail("checkpoint parameter count mismatch", error.Mismatch);
    const cfg = m.c;
    for (m.params.items) |*par| {
        var n: u64 = par.n;
        try io.scalar(u64, &n);
        if (n != par.n) return fail("checkpoint parameter shape mismatch", error.Mismatch);
        const buf = mx.gpa.alloc(f32, par.n) catch mx.oom();
        defer mx.gpa.free(buf);
        inline for (.{ "master", "m", "v" }) |field| {
            if (io.writing) mx.toF32(@field(par, field), buf);
            try io.f32s(buf);
            if (!io.writing) {
                const m0 = mx.mark();
                defer mx.release(m0);
                mx.assign(&@field(par, field), mx.fromSlice(f32, buf, par.shp()));
            }
        }
    }
    try io.scalar(i64, &st.position);
    if (st.position < 0 or @as(u64, @intCast(st.position)) != p.carried)
        return fail("checkpoint stream position mismatch", error.Corrupt);
    const B: usize = @intCast(cfg.batch);
    const bd = B * @as(usize, @intCast(cfg.dim));
    const hbuf = mx.gpa.alloc(f32, bd * 256) catch mx.oom();
    defer mx.gpa.free(hbuf);
    for (m.layers, 0..) |ly, l| {
        try stateF32(io, &st.carry[l], hbuf[0..bd], &.{ cfg.batch, cfg.dim });
        if (ly.mla == null) continue;
        const k = &st.cache[l];
        try io.scalar(i64, &k.head);
        try io.scalar(i64, &k.base0);
        if (k.head < 0 or k.head > cfg.mla_cache or k.base0 < 0 or k.base0 + k.head != st.position)
            return fail("invalid checkpoint cache position", error.Corrupt);
        try cacheIo(io, k, B, @intCast(cfg.mla_L), @intCast(cfg.mla_R));
    }
    if (cfg.traces == 1) { // absent for traces=0, so older V3 files load unchanged
        for (m.layers, 0..) |_, l| {
            try stateF32(io, &st.tdec[l], hbuf[0..bd], &.{ cfg.batch, cfg.dim });
            try stateF32(io, &st.tgate[l], hbuf[0..bd], &.{ cfg.batch, cfg.dim });
        }
        try stateF32(io, &st.temb, hbuf, &.{ cfg.batch, 256, cfg.dim });
    }
    try io.finish();
}

fn stateF32(io: *File, a: *mx.Array, buf: []f32, shp: []const i32) Error!void {
    if (io.writing) mx.toF32(a.*, buf);
    try io.f32s(buf);
    if (!io.writing) {
        const m0 = mx.mark();
        defer mx.release(m0);
        mx.assign(a, mx.fromSlice(f32, buf, shp));
    }
}

/// Initialized cache rows: per stream, head latents (L) then head rotated keys (R), bf16.
fn cacheIo(io: *File, k: *model.Cache, B: usize, L: usize, R: usize) Error!void {
    const head: usize = @intCast(k.head);
    if (head == 0) {
        if (!io.writing) {
            mx.free(k.lat);
            mx.free(k.kr);
            k.lat = mx.none;
            k.kr = mx.none;
        }
        return;
    }
    const lat = mx.gpa.alloc(u16, B * head * L) catch mx.oom();
    defer mx.gpa.free(lat);
    const kr = mx.gpa.alloc(u16, B * head * R) catch mx.oom();
    defer mx.gpa.free(kr);
    if (io.writing) {
        mx.toBf16Bits(k.lat, lat);
        mx.toBf16Bits(k.kr, kr);
    }
    for (0..B) |b| {
        try io.u16s(lat[b * head * L ..][0 .. head * L]);
        try io.u16s(kr[b * head * R ..][0 .. head * R]);
    }
    if (!io.writing) {
        const m0 = mx.mark();
        defer mx.release(m0);
        const bi: i32 = @intCast(B);
        const hi: i32 = @intCast(head);
        mx.assign(&k.lat, mx.fromBf16Bits(lat, &.{ bi, hi, @intCast(L) }));
        mx.assign(&k.kr, mx.fromBf16Bits(kr, &.{ bi, hi, @intCast(R) }));
    }
}

pub fn save(path: [:0]const u8, m: *model.Model, st: *model.State, p: *Progress) Error!void {
    var tmpbuf: [4096]u8 = undefined;
    const tmp = std.fmt.bufPrintZ(&tmpbuf, "{s}.tmp.{d}", .{ path, c.getpid() }) catch return fail("checkpoint path too long", error.Io);
    var ok = false;
    defer if (!ok) {
        _ = c.unlink(tmp.ptr);
    };
    {
        var io = try File.open(tmp, true);
        defer io.close();
        try payload(&io, m, st, p);
    }
    if (c.rename(tmp.ptr, path.ptr) != 0) return fail("checkpoint rename failed", error.Io);
    ok = true;
}

pub fn load(path: [:0]const u8, m: *model.Model, st: *model.State, p: *Progress) Error!void {
    try verify(path);
    var io = try File.open(path, false);
    defer io.close();
    try payload(&io, m, st, p);
}

/// Weights only (masters), with full integrity check and a layout proof: the
/// optimizer moments and stream state are skipped, the file end must follow.
pub fn loadWeights(path: [:0]const u8, m: *model.Model, file_cfg: Cfg) Error!void {
    try verify(path);
    var io = try File.open(path, false);
    defer io.close();
    var stored = Cfg{};
    try header(&io, &stored);
    if (!config.sameModel(stored, m.c)) return fail("checkpoint architecture differs; use a new path for a new experiment", error.Mismatch);
    var pr = Progress{};
    try progressIo(&io, &pr);
    var count: u64 = 0;
    try io.scalar(u64, &count);
    if (count != m.nparams()) return fail("parameter count mismatch", error.Mismatch);
    for (m.params.items) |*par| {
        var n: u64 = 0;
        try io.scalar(u64, &n);
        if (n != par.n) return fail("parameter shape mismatch", error.Mismatch);
        const buf = mx.gpa.alloc(f32, par.n) catch mx.oom();
        defer mx.gpa.free(buf);
        try io.f32s(buf);
        const m0 = mx.mark();
        defer mx.release(m0);
        mx.assign(&par.master, mx.fromSlice(f32, buf, par.shp()));
        try io.skip(n * 8);
    }
    var spos: i64 = 0;
    try io.scalar(i64, &spos);
    const fB: u64 = @intCast(file_cfg.batch);
    const fD: u64 = @intCast(file_cfg.dim);
    for (0..@intCast(file_cfg.layers)) |l| {
        try io.skip(fB * fD * 4);
        if (file_cfg.mla == 1 and @mod(@as(i32, @intCast(l)), file_cfg.mla_every) == 0) {
            var head: i64 = 0;
            var base0: i64 = 0;
            try io.scalar(i64, &head);
            try io.scalar(i64, &base0);
            if (head < 0 or head > file_cfg.mla_cache) return fail("invalid checkpoint cache position", error.Corrupt);
            try io.skip(fB * @as(u64, @intCast(head)) * @as(u64, @intCast(file_cfg.mla_L + file_cfg.mla_R)) * 2);
        }
    }
    if (file_cfg.traces == 1) try io.skip((2 * @as(u64, @intCast(file_cfg.layers)) + 256) * fB * fD * 4);
    const pos = c.ftello(io.f);
    _ = c.fseeko(io.f, 0, c.SEEK_END);
    if (c.ftello(io.f) - pos != 8) return fail("checkpoint layout mismatch", error.Mismatch);
}
