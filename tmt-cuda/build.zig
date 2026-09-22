//! Zig host programs for TMT-CUDA. The CUDA kernels stay in src/*.cu and are
//! compiled by nvcc into one object with a C API (src/capi.h); Zig links it
//! with the CUDA runtime, cuBLAS and libstdc++.
//!
//!   zig build -Doptimize=ReleaseFast      # binaries in zig-out/bin
//!   zig build test                        # Zig unit tests
//!   zig build -Darch=sm_89                # other GPU (default sm_120a)
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });
    const arch = b.option([]const u8, "arch", "CUDA architecture (default sm_120a)") orelse "sm_120a";
    const cuda = b.option([]const u8, "cuda", "CUDA installation (default /opt/cuda)") orelse "/opt/cuda";
    // nvcc's host compiler is g++, so the object needs the system libstdc++
    // (Zig's "c++" would be its own libc++ with a different ABI).
    const libstdcxx = b.option([]const u8, "libstdcxx", "path of libstdc++.so (g++ -print-file-name=libstdc++.so)") orelse "/usr/lib/libstdc++.so";
    const libgcc = b.option([]const u8, "libgcc", "path of libgcc_s (C++ exception unwinding)") orelse "/usr/lib/libgcc_s.so.1";

    // nvcc: the model and its C API as one object.
    const nvcc = b.addSystemCommand(&.{ b.pathJoin(&.{ cuda, "bin", "nvcc" }), "-O3", "-std=c++17", "--use_fast_math", "-diag-suppress", "549,550", "-Xcompiler", "-fPIC,-Wall" });
    const compute = b.fmt("-gencode=arch=compute_{s},code={s}", .{ arch[3..], arch });
    nvcc.addArg(compute);
    nvcc.addArgs(&.{ "-c", "-o" });
    const capi_obj = nvcc.addOutputFileArg("capi.o");
    nvcc.addFileArg(b.path("src/capi.cu"));
    // Rebuild when any CUDA source changes.
    for ([_][]const u8{ "capi.cu", "capi.h", "generate.h", "checkpoint.h", "model.cu", "train.cu", "config.h", "cell.cu", "moe.cu", "mla.cu", "memory.cu", "norm.cu", "linalg.cu", "emb.cu", "loss.cu", "adam.cu", "util.h", "common.h" }) |f|
        nvcc.addFileInput(b.path(b.pathJoin(&.{ "src", f })));

    const Tool = struct { name: []const u8, cuda: bool };
    const tools = [_]Tool{
        .{ .name = "dialogprep", .cuda = false },
        .{ .name = "kgprep", .cuda = false },
        .{ .name = "sample", .cuda = true },
        .{ .name = "chat", .cuda = true },
        .{ .name = "train", .cuda = true },
    };
    for (tools) |t| {
        const mod = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{t.name})),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        if (t.cuda) {
            mod.addObjectFile(capi_obj);
            mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "lib64" }) });
            mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "lib64" }) });
            mod.linkSystemLibrary("cudart", .{});
            mod.linkSystemLibrary("cublas", .{});
            mod.addObjectFile(.{ .cwd_relative = libstdcxx });
            mod.addObjectFile(.{ .cwd_relative = libgcc });
        }
        const exe = b.addExecutable(.{ .name = t.name, .root_module = mod });
        b.installArtifact(exe);
    }

    const test_step = b.step("test", "Run the Zig unit tests");
    for ([_][]const u8{ "json", "stdrand" }) |name| {
        const t = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{name})),
            .target = target,
            .optimize = optimize,
        }) });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
