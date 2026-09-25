//! ckpttool: read and rewrite the fp32 master weights of a V3 checkpoint.
//!
//!   ckpttool layout CKPT
//!   ckpttool dump   CKPT OUT.f32
//!   ckpttool patch  IN.ckpt WEIGHTS.f32 OUT.ckpt [step=N] [moments=keep|zero|scale] [mscale=F] [vscale=F]
//!
//! The trajectory experiment (MOONSHOTS.md A) needs the parameter vector of a
//! series of checkpoints as one flat array, and needs to write a checkpoint
//! back from an extrapolated array. Both are pure file operations - no GPU,
//! no model - so this tool parses the V3 layout directly and leaves every
//! byte it does not own alone, including the configuration text and the
//! streaming state. The trailing checksum is recomputed with the same FNV-1a
//! as `checkpoint.hashBytes`, so the result loads through the ordinary
//! reader with no special case.
//!
//! `step=` rewrites the progress step, which is what a jump forward along the
//! trajectory means for the learning-rate schedule; the data cursor is left
//! where it was, so the jumped run simply reads on from there.
const std = @import("std");
const cli = @import("cli.zig");

const MAGIC: u64 = 0x33544b5043544d54; // TMTCPKT3
const HASH_INIT: u64 = 14695981039346656037;

/// The same FNV-1a as `checkpoint.hashBytes`, repeated here so the tool links
/// without the model and the CUDA runtime.
fn hashBytes(h: u64, data: []const u8) u64 {
    var x = h;
    for (data) |b| x = (x ^ b) *% 1099511628211;
    return x;
}

/// Where every parameter's three fp32 arrays live inside the file image.
const Layout = struct {
    text: []const u8,
    progress: usize, // offset of the six progress words
    counts: std.ArrayList(u64), // element count per parameter
    master: std.ArrayList(usize), // byte offset of each master array
    total: u64 = 0,

    fn deinit(l: *Layout, gpa: std.mem.Allocator) void {
        l.counts.deinit(gpa);
        l.master.deinit(gpa);
    }
};

fn word(image: []const u8, at: usize) !u64 {
    if (at + 8 > image.len) return error.Malformed;
    return std.mem.readInt(u64, image[at..][0..8], .little);
}

fn parse(gpa: std.mem.Allocator, image: []const u8) !Layout {
    if (try word(image, 0) != MAGIC) return error.Malformed;
    const len = try word(image, 8);
    if (len == 0 or len > 16384 or 16 + len > image.len) return error.Malformed;
    var l = Layout{ .text = image[16..][0..@intCast(len)], .progress = 16 + @as(usize, @intCast(len)), .counts = .empty, .master = .empty };
    errdefer l.deinit(gpa);
    var at = l.progress + 6 * 8;
    const count = try word(image, at);
    at += 8;
    if (count > 1 << 20) return error.Malformed;
    for (0..@intCast(count)) |_| {
        const n = try word(image, at);
        at += 8;
        if (n == 0 or at + 3 * n * 4 > image.len) return error.Malformed;
        try l.counts.append(gpa, n);
        try l.master.append(gpa, at);
        l.total += n;
        at += 3 * n * 4;
    }
    return l;
}

/// The three fp32 arrays of one parameter, in file order, as *bytes*.
///
/// They cannot be viewed as `[]f32`: the parameter blocks start at whatever
/// offset the configuration text left them at - 811 in a current checkpoint -
/// and `@alignCast` on an odd address panics. `dump` never noticed because it
/// writes raw bytes; `patch` did, every time it was run on a real file.
fn arrays(image: []u8, l: Layout, i: usize) [3][]u8 {
    const n: usize = @intCast(l.counts.items[i]);
    var out: [3][]u8 = undefined;
    for (0..3) |k| {
        const at = l.master.items[i] + k * n * 4;
        out[k] = image[at..][0 .. n * 4];
    }
    return out;
}

fn getF(b: []const u8, j: usize) f32 {
    return @bitCast(std.mem.readInt(u32, b[j * 4 ..][0..4], .little));
}
fn putF(b: []u8, j: usize, v: f32) void {
    std.mem.writeInt(u32, b[j * 4 ..][0..4], @bitCast(v), .little);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const args = try cli.Args.parse(init.arena.allocator(), init.minimal.args);
    const pos = args.positional.items;
    if (pos.len < 2) return cli.fail(io, "usage: ckpttool layout|dump|patch ...\n", .{});

    const image = try std.Io.Dir.cwd().readFileAlloc(io, pos[1], gpa, .unlimited);
    defer gpa.free(image);
    var l = parse(gpa, image) catch return cli.fail(io, "not a V3 checkpoint: {s}\n", .{pos[1]});
    defer l.deinit(gpa);

    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &out.interface;

    if (std.mem.eql(u8, pos[0], "layout")) {
        try w.print("params={d} total={d} step={d} cursor={d} epoch={d} carried={d}\n", .{
            l.counts.items.len,               l.total,
            try word(image, l.progress),      try word(image, l.progress + 8),
            try word(image, l.progress + 16), try word(image, l.progress + 24),
        });
        for (l.counts.items, l.master.items) |n, at| try w.print("{d} {d}\n", .{ n, at });
        return w.flush();
    }

    if (std.mem.eql(u8, pos[0], "dump")) {
        if (pos.len != 3) return cli.fail(io, "usage: ckpttool dump CKPT OUT.f32\n", .{});
        const file = try std.Io.Dir.cwd().createFile(io, pos[2], .{});
        defer file.close(io);
        var fbuf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &fbuf);
        for (0..l.counts.items.len) |i| {
            const n: usize = @intCast(l.counts.items[i]);
            try fw.interface.writeAll(image[l.master.items[i]..][0 .. n * 4]);
        }
        try fw.interface.flush();
        try w.print("wrote {d} floats\n", .{l.total});
        return w.flush();
    }

    if (!std.mem.eql(u8, pos[0], "patch") or pos.len != 4)
        return cli.fail(io, "usage: ckpttool patch IN.ckpt WEIGHTS.f32 OUT.ckpt [step=N] [moments=keep|zero|scale]\n", .{});

    const weights = try std.Io.Dir.cwd().readFileAlloc(io, pos[2], gpa, .unlimited);
    defer gpa.free(weights);
    if (weights.len != l.total * 4) return cli.fail(io, "weight file is {d} floats, checkpoint holds {d}\n", .{ weights.len / 4, l.total });
    const flat: []const u8 = weights;

    const moments = args.get("moments", "keep");
    const mscale: f32 = @floatCast(try args.float("mscale", 1));
    const vscale: f32 = @floatCast(try args.float("vscale", 1));
    var cursor: usize = 0;
    for (0..l.counts.items.len) |i| {
        const n: usize = @intCast(l.counts.items[i]);
        const a = arrays(image, l, i);
        @memcpy(a[0], flat[cursor * 4 ..][0 .. n * 4]);
        cursor += n;
        if (std.mem.eql(u8, moments, "zero")) {
            @memset(a[1], 0);
            @memset(a[2], 0);
        } else if (std.mem.eql(u8, moments, "scale")) {
            for (0..n) |j| putF(a[1], j, getF(a[1], j) * mscale);
            for (0..n) |j| putF(a[2], j, getF(a[2], j) * vscale);
        } else if (!std.mem.eql(u8, moments, "keep")) {
            return cli.fail(io, "moments must be keep, zero or scale\n", .{});
        }
    }
    if (args.options.get("step")) |_| {
        const step = try args.int(u64, "step", 0);
        std.mem.writeInt(u64, image[l.progress..][0..8], step, .little);
    }
    // Everything but the trailer hashes, exactly as the writer does it.
    const body = image[0 .. image.len - 8];
    std.mem.writeInt(u64, image[image.len - 8 ..][0..8], hashBytes(HASH_INIT, body), .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos[3], .data = image });
    try w.print("patched {d} floats into {s} (moments={s})\n", .{ l.total, pos[3], moments });
    return w.flush();
}
