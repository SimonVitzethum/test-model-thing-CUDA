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

    // The kernels in Zig: nvptx64 IR -> patch -> link libdevice -> PTX.
    // The host loads the PTX through the CUDA driver API (see zig/ptx.zig).
    const irpatch = b.addExecutable(.{ .name = "irpatch", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tools/irpatch.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    }) });
    const emit_ir = b.addSystemCommand(&.{ b.graph.zig_exe, "build-obj", "-target", "nvptx64-cuda", "-mcpu", arch, "-OReleaseFast", "-fstrip", "-fno-emit-bin" });
    const raw_ir = emit_ir.addPrefixedOutputFileArg("-femit-llvm-ir=", "kernels.ll");
    emit_ir.addFileArg(b.path("zig/kernels/kernels.zig"));
    emit_ir.addFileInput(b.path("zig/kernels/cuda.zig"));
    const patch = b.addRunArtifact(irpatch);
    patch.addFileArg(raw_ir);
    const patched_ir = patch.addOutputFileArg("kernels-patched.ll");
    const kernel_names = patch.addOutputFileArg("kernels.txt");
    patch.addArg("--ftz"); // nvcc builds the C++ kernels with --use_fast_math
    const link = b.addSystemCommand(&.{"llvm-link"});
    link.addFileArg(patched_ir);
    link.addFileArg(.{ .cwd_relative = b.pathJoin(&.{ cuda, "nvvm", "libdevice", "libdevice.10.bc" }) });
    link.addArg("-o");
    const linked = link.addOutputFileArg("kernels-linked.bc");
    const optimize_bc = b.addSystemCommand(&.{ "opt", "-passes=nvvm-reflect,internalize,globaldce,default<O3>" });
    optimize_bc.addPrefixedFileArg("-internalize-public-api-file=", kernel_names);
    optimize_bc.addFileArg(linked);
    optimize_bc.addArg("-o");
    const optimized = optimize_bc.addOutputFileArg("kernels-opt.bc");
    const llc = b.addSystemCommand(&.{ "llc", "-march=nvptx64", b.fmt("-mcpu={s}", .{arch}) });
    llc.addFileArg(optimized);
    llc.addArg("-o");
    const ptx = llc.addOutputFileArg("kernels.ptx");
    b.getInstallStep().dependOn(&b.addInstallFile(ptx, "kernels.ptx").step);

    // `reference` links the C++ model through its C API; only the comparison
    // program needs it, every other tool runs the Zig model.
    const Tool = struct { name: []const u8, cuda: bool, reference: bool = false };
    const tools = [_]Tool{
        .{ .name = "dialogprep", .cuda = false },
        .{ .name = "kgprep", .cuda = false },
        .{ .name = "sample", .cuda = true },
        .{ .name = "chat", .cuda = true },
        .{ .name = "train", .cuda = true },
        .{ .name = "gradcheck", .cuda = true },
        .{ .name = "kgtrain", .cuda = true },
        .{ .name = "ktest", .cuda = true, .reference = true },
    };
    for (tools) |t| {
        const mod = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{t.name})),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        if (t.cuda) {
            mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "lib64" }) });
            mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "lib64" }) });
            mod.linkSystemLibrary("cudart", .{});
            mod.linkSystemLibrary("cublas", .{});
        }
        if (t.reference) { // nvcc's object needs the system libstdc++
            mod.addObjectFile(capi_obj);
            mod.addObjectFile(.{ .cwd_relative = libstdcxx });
            mod.addObjectFile(.{ .cwd_relative = libgcc });
        }
        if (t.cuda) { // the kernels are loaded from the embedded PTX module
            mod.addAnonymousImport("kernels.ptx", .{ .root_source_file = ptx });
            mod.linkSystemLibrary("cuda", .{});
        }
        const exe = b.addExecutable(.{ .name = t.name, .root_module = mod });
        b.installArtifact(exe);
    }

    const test_step = b.step("test", "Run the Zig unit tests");
    for ([_][]const u8{ "json", "stdrand", "model/config" }) |name| {
        const t = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{name})),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }) });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
