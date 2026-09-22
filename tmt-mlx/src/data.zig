// Raw byte datasets via mmap, identified by size and FNV content hash.
const std = @import("std");
const ckpt = @import("checkpoint.zig");
const u = @import("util.zig");
const c = @cImport({
    @cInclude("sys/mman.h");
    @cInclude("sys/stat.h");
    @cInclude("fcntl.h");
    @cInclude("unistd.h");
});

pub const Dataset = struct {
    bytes: []const u8,
    hash: u64,
    fd: c_int,

    pub fn open(path: [:0]const u8) Dataset {
        const fd = c.open(path.ptr, c.O_RDONLY);
        var st: c.struct_stat = undefined;
        if (fd < 0 or c.fstat(fd, &st) != 0 or st.st_size < 2) {
            if (fd >= 0) _ = c.close(fd);
            u.die("dataset missing or shorter than two bytes", .{});
        }
        const size: usize = @intCast(st.st_size);
        const p = c.mmap(null, size, c.PROT_READ, c.MAP_PRIVATE, fd, 0);
        if (p == c.MAP_FAILED) u.die("dataset mmap failed", .{});
        const bytes = @as([*]const u8, @ptrCast(p))[0..size];
        return .{ .bytes = bytes, .hash = ckpt.hashBytes(ckpt.HASH_INIT, bytes), .fd = fd };
    }
    pub fn close(self: *Dataset) void {
        _ = c.munmap(@constCast(@ptrCast(self.bytes.ptr)), self.bytes.len);
        _ = c.close(self.fd);
    }
};

/// Dialog control bytes (tools/dialogprep.zig).
pub const CONV = 0x1E;
pub const USER = 0x02;
pub const ASSISTANT = 0x03;
pub const END = 0x04;
pub fn isMarker(b: u8) bool {
    return b == CONV or b == USER or b == ASSISTANT or b == END;
}
