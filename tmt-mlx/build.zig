const std = @import("std");

// MLX and its C API (mlx-c) are expected under -Dmlx=PATH (default ~/.local/mlx),
// with include/ and lib/ (libmlx.dylib, libmlxc.dylib, mlx.metallib).
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Optimization mode (default ReleaseFast)") orelse .ReleaseFast;
    const home = b.graph.environ_map.get("HOME") orelse ".";
    const mlx = b.option([]const u8, "mlx", "MLX/mlx-c install prefix") orelse b.pathJoin(&.{ home, ".local/mlx" });
    const mlx_lib = b.pathJoin(&.{ mlx, "lib" });

    const exes = [_]struct { name: []const u8, src: []const u8, mlx: bool = true }{
        .{ .name = "train", .src = "src/train_main.zig" },
        .{ .name = "sample", .src = "src/sample_main.zig" },
        .{ .name = "chat", .src = "src/chat_main.zig" },
        .{ .name = "gradcheck", .src = "src/gradcheck_main.zig" },
        .{ .name = "architecture_test", .src = "src/architecture_test.zig" },
        .{ .name = "bench", .src = "src/bench_main.zig" },
        .{ .name = "dialogprep", .src = "tools/dialogprep.zig", .mlx = false },
    };
    for (exes) |e| {
        const mod = b.createModule(.{
            .root_source_file = b.path(e.src),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        if (e.mlx) {
            mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ mlx, "include" }) });
            mod.addLibraryPath(.{ .cwd_relative = mlx_lib });
            mod.addRPath(.{ .cwd_relative = mlx_lib });
            mod.linkSystemLibrary("mlxc", .{});
            mod.linkSystemLibrary("mlx", .{});
        }
        const exe = b.addExecutable(.{ .name = e.name, .root_module = mod });
        b.installArtifact(exe);
    }

    const check = b.step("check", "Run architecture and CLI tests");
    const arch = b.addSystemCommand(&.{b.getInstallPath(.bin, "architecture_test")});
    arch.step.dependOn(b.getInstallStep());
    const cli = b.addSystemCommand(&.{ "sh", "tests/cli.sh" });
    cli.step.dependOn(&arch.step);
    check.dependOn(&cli.step);
}
