//! Prepares the LLVM IR that Zig emits for nvptx64 so that llc accepts it.
//!
//! Zig exports a symbol as an alias to the internal definition
//! (`@name = alias ..., ptr @file.name`), which the NVPTX backend rejects
//! ("NVPTX aliasee must be a non-kernel function definition"). This tool drops
//! the alias, renames the definition to the exported name and gives it
//! external linkage. It also writes the kernel names to a second file (for
//! `opt -internalize-public-api-file`) and can set the `nvvm-reflect-ftz`
//! module flag, which selects the flush-to-zero libdevice variants that
//! nvcc's `--use_fast_math` also picks.
//!
//! usage: irpatch IN.ll OUT.ll NAMES.txt [--ftz]
const std = @import("std");

fn warn(io: std.Io, msg: []const u8) void {
    var buf: [128]u8 = undefined;
    var w = std.Io.File.stderr().writerStreaming(io, &buf);
    w.interface.writeAll(msg) catch {};
    w.interface.flush() catch {};
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const gpa = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(gpa);
    if (argv.len < 4) {
        warn(io, "usage: irpatch IN.ll OUT.ll NAMES.txt [--ftz]\n");
        return 1;
    }
    const ftz = argv.len > 4 and std.mem.eql(u8, argv[4], "--ftz");
    const text = try std.Io.Dir.cwd().readFileAlloc(io, argv[1], gpa, .unlimited);

    // Collect the aliases: "@name = alias <type>, ptr @internal.name".
    var names: std.ArrayList([]const u8) = .empty;
    var internal: std.ArrayList([]const u8) = .empty;
    var out: std.ArrayList(u8) = .empty;
    var max_meta: u32 = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "!") and std.mem.indexOf(u8, line, " = ") != null) {
            const id = std.fmt.parseInt(u32, line[1..std.mem.indexOfScalar(u8, line, ' ').?], 10) catch 0;
            max_meta = @max(max_meta, id);
        }
        if (!std.mem.startsWith(u8, line, "@") or std.mem.indexOf(u8, line, " = alias ") == null) continue;
        const eq = std.mem.indexOf(u8, line, " = alias ").?;
        const ptr = std.mem.lastIndexOf(u8, line, "ptr @") orelse continue;
        try names.append(gpa, line[1..eq]);
        try internal.append(gpa, std.mem.trimEnd(u8, line[ptr + 5 ..], " \t\r"));
    }

    // Two kernels with identical bodies end up as one definition with two
    // aliases; renaming would silently drop one of them.
    for (internal.items, 0..) |sym, i| for (internal.items[i + 1 ..], names.items[i + 1 ..]) |other, name| {
        if (std.mem.eql(u8, sym, other)) {
            warn(io, "irpatch: identical kernel bodies share one definition: ");
            warn(io, names.items[i]);
            warn(io, " and ");
            warn(io, name);
            warn(io, "\n");
            return 1;
        }
    };

    lines = std.mem.splitScalar(u8, text, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, " = alias ") != null) continue; // dropped
        if (!first) try out.append(gpa, '\n');
        first = false;
        var patched = line;
        // The definition becomes the exported kernel itself.
        for (names.items, internal.items) |name, sym| {
            var buf: [512]u8 = undefined;
            for ([_][]const u8{ "define private ptx_kernel ", "define internal ptx_kernel " }) |prefix| {
                const needle = try std.fmt.bufPrint(&buf, "{s}", .{prefix});
                if (std.mem.startsWith(u8, patched, needle) and std.mem.indexOf(u8, patched, sym) != null)
                    patched = try std.mem.concat(gpa, u8, &.{ "define ptx_kernel ", patched[needle.len..] });
            }
            const old = try std.fmt.allocPrint(gpa, "@{s}", .{sym});
            const new = try std.fmt.allocPrint(gpa, "@{s}", .{name});
            if (std.mem.indexOf(u8, patched, old) != null)
                patched = try std.mem.replaceOwned(u8, gpa, patched, old, new);
        }
        // nvcc fuses a multiply and an add into one FMA (-fmad=true, its
        // default); in LLVM that needs the `contract` flag per instruction.
        inline for (.{ "fadd", "fsub", "fmul" }) |op| {
            const needle = " = " ++ op ++ " ";
            if (std.mem.indexOf(u8, patched, needle) != null)
                patched = try std.mem.replaceOwned(u8, gpa, patched, needle, " = " ++ op ++ " contract ");
        }
        // --use_fast_math also flushes denormals in plain float arithmetic
        // (add.ftz.f32 instead of add.rn.f32); in LLVM that is a function
        // attribute, not a module flag.
        if (ftz and std.mem.startsWith(u8, patched, "attributes #") and std.mem.endsWith(u8, patched, "}"))
            patched = try std.mem.concat(gpa, u8, &.{ patched[0 .. patched.len - 1], " \"denormal-fp-math-f32\"=\"preserve-sign,preserve-sign\" }" });
        try out.appendSlice(gpa, patched);
    }
    if (ftz) {
        const id = max_meta + 1;
        const flags = "!llvm.module.flags = !{";
        if (std.mem.indexOf(u8, out.items, flags)) |at| {
            const insert = at + flags.len;
            const sep: []const u8 = if (out.items[insert] == '}') "" else ", ";
            const extra = try std.fmt.allocPrint(gpa, "!{d}{s}", .{ id, sep });
            try out.insertSlice(gpa, insert, extra);
        } else try out.appendSlice(gpa, try std.fmt.allocPrint(gpa, "\n!llvm.module.flags = !{{!{d}}}", .{id}));
        try out.appendSlice(gpa, try std.fmt.allocPrint(gpa, "\n!{d} = !{{i32 4, !\"nvvm-reflect-ftz\", i32 1}}\n", .{id}));
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = argv[2], .data = out.items });

    var list: std.ArrayList(u8) = .empty;
    for (names.items) |n| {
        try list.appendSlice(gpa, n);
        try list.append(gpa, '\n');
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = argv[3], .data = list.items });
    if (names.items.len == 0) {
        warn(io, "irpatch: no exported kernels found\n");
        return 1;
    }
    return 0;
}
