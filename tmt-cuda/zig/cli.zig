//! Command-line helpers shared by the tools: positional arguments plus
//! `key=value` options, as in the C++ tools.
const std = @import("std");

pub const Args = struct {
    positional: std.ArrayList([]const u8) = .empty,
    options: std.StringHashMap([]const u8),

    pub fn parse(gpa: std.mem.Allocator, raw: std.process.Args) !Args {
        var a = Args{ .options = .init(gpa) };
        var it = raw.iterate();
        _ = it.next(); // program name
        while (it.next()) |arg| {
            if (std.mem.indexOfScalar(u8, arg, '=')) |eq| {
                try a.options.put(arg[0..eq], arg[eq + 1 ..]);
            } else try a.positional.append(gpa, arg);
        }
        return a;
    }

    pub fn get(a: Args, key: []const u8, default: []const u8) []const u8 {
        return a.options.get(key) orelse default;
    }
    pub fn float(a: Args, key: []const u8, default: f64) !f64 {
        const v = a.options.get(key) orelse return default;
        return std.fmt.parseFloat(f64, v) catch error.InvalidOption;
    }
    pub fn int(a: Args, comptime T: type, key: []const u8, default: T) !T {
        const v = a.options.get(key) orelse return default;
        return std.fmt.parseInt(T, v, 10) catch error.InvalidOption;
    }
};

/// Prints `msg` to stderr (unbuffered) and returns error.Usage.
pub fn fail(io: std.Io, comptime fmt: []const u8, args: anytype) error{Usage} {
    var buf: [1024]u8 = undefined;
    var w = std.Io.File.stderr().writer(io, &buf);
    w.interface.print(fmt, args) catch {};
    w.interface.flush() catch {};
    return error.Usage;
}

/// Next line of `r` without the newline, of any length (grows `line`), or null
/// at the end of the stream. The returned slice is valid until the next call.
pub fn readLine(r: *std.Io.Reader, line: *std.Io.Writer.Allocating) !?[]const u8 {
    line.clearRetainingCapacity();
    _ = try r.streamDelimiterEnding(&line.writer, '\n');
    if (r.bufferedLen() > 0) {
        r.toss(1); // the newline
        return line.written();
    }
    return if (line.written().len > 0) line.written() else null;
}
