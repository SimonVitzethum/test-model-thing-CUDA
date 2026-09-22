//! kgtrain: train and evaluate the fact memory on knowledge-graph QA data
//! (port of src/kg_main.cu).
//!   kgtrain train DATA.tsv CKPT [steps=N] [saveevery=N] [nodes=NODES.tsv] [key=value ...]
//!   kgtrain eval  DATA.tsv CKPT [memory=on|off|shuffled|retrieved] [nodes=NODES.tsv] [show=N]
//!   kgtrain ask   CKPT nodes=NODES.tsv ["question"] [top=5] [maxlen=80] [temp=0]
//! DATA.tsv rows: question <tab> answer <tab> memory <tab> subject, as written
//! by kgprep qa. Each example is one window "question answer\n"; the loss
//! covers only the answer bytes and the final newline.
const std = @import("std");
const config = @import("model/config.zig");
const checkpoint = @import("model/checkpoint.zig");
const session = @import("model/session.zig");
const gpu = @import("model/gpu.zig");

const kernels_ptx = @embedFile("kernels.ptx");
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;
const stdrand = @import("stdrand.zig");
const retrieval = @import("retrieval.zig");

const Fatal = error{Fatal};

extern "c" fn exp(x: f64) f64;

const Out = struct {
    w: *std.Io.Writer,
    fn print(o: Out, comptime fmt: [:0]const u8, args: anytype) !void {
        var buf: [4096]u8 = undefined;
        const n = @call(.auto, snprintf, .{ &buf, buf.len, fmt.ptr } ++ args);
        try o.w.writeAll(buf[0..@intCast(@min(n, buf.len - 1))]);
        if (fmt[fmt.len - 1] == '\n') try o.w.flush();
    }
};

var err_msg: [512]u8 = undefined;
var err_len: usize = 0;
fn fail(comptime fmt: []const u8, args: anytype) Fatal {
    const s: []const u8 = std.fmt.bufPrint(&err_msg, fmt, args) catch err_msg[0..];
    err_len = s.len;
    return error.Fatal;
}
fn cfgFail() Fatal {
    return fail("{s}", .{config.lastError()});
}
fn ckptFail() Fatal {
    return fail("{s}", .{checkpoint.lastError()});
}
fn gpuFail() Fatal {
    return fail("{s}", .{gpu.lastError()});
}

const Example = struct { question: []const u8, answer: []const u8, memory: []const u8, subject: []const u8 };
const Node = struct { qid: []const u8, label: []const u8, memory: []const u8 };

/// Tab-separated rows, split like std::getline plus find('\t').
fn readTsv(gpa: std.mem.Allocator, io: std.Io, path: []const u8, size: ?*u64, hash: ?*u64) ![][][]const u8 {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch
        return fail("cannot read {s}", .{path});
    if (size) |p| p.* = text.len;
    if (hash) |p| {
        var h: u64 = 14695981039346656037;
        for (text) |c| h = (h ^ c) *% 1099511628211;
        p.* = h;
    }
    var rows: std.ArrayList([][]const u8) = .empty;
    var pos: usize = 0;
    while (pos < text.len) {
        const nl = std.mem.indexOfScalarPos(u8, text, pos, '\n') orelse text.len;
        const line = text[pos..nl];
        pos = nl + 1;
        var fields: std.ArrayList([]const u8) = .empty;
        var start: usize = 0;
        while (std.mem.indexOfScalarPos(u8, line, start, '\t')) |tab| {
            try fields.append(gpa, line[start..tab]);
            start = tab + 1;
        }
        try fields.append(gpa, line[start..]);
        try rows.append(gpa, try fields.toOwnedSlice(gpa));
    }
    return rows.toOwnedSlice(gpa);
}

fn loadExamples(gpa: std.mem.Allocator, io: std.Io, path: []const u8, size: *u64, hash: *u64) ![]Example {
    var out: std.ArrayList(Example) = .empty;
    for (try readTsv(gpa, io, path, size, hash)) |f|
        if (f.len >= 3 and f[0].len != 0 and f[1].len != 0)
            try out.append(gpa, .{ .question = f[0], .answer = f[1], .memory = f[2], .subject = if (f.len > 3) f[3] else "" });
    if (out.items.len == 0) return fail("no examples in {s}", .{path});
    return out.toOwnedSlice(gpa);
}

fn loadNodes(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]Node {
    var out: std.ArrayList(Node) = .empty;
    for (try readTsv(gpa, io, path, null, null)) |f|
        if (f.len == 3 and f[1].len != 0) try out.append(gpa, .{ .qid = f[0], .label = f[1], .memory = f[2] });
    if (out.items.len == 0) return fail("no nodes in {s}", .{path});
    return out.toOwnedSlice(gpa);
}

/// One batch window of bytes: ids and targets (batch*seqlen), memory bytes
/// (batch*mem_len). Shared by the answer, reading and evaluation passes.
const Window = struct {
    ids: []c_int,
    nxt: []c_int,
    mem: []c_int,
    T: usize,
    M: usize,

    fn init(gpa: std.mem.Allocator, B: usize, T: usize, M: usize) !Window {
        return .{ .ids = try gpa.alloc(c_int, B * T), .nxt = try gpa.alloc(c_int, B * T),
                  .mem = try gpa.alloc(c_int, B * M), .T = T, .M = M };
    }
    /// Answer window b: "question answer\n", loss on the answer. Number of
    /// scored positions, or null if the example does not fit into seqlen.
    fn answer(w: Window, gpa: std.mem.Allocator, e: Example, memory: []const u8, b: usize) !?usize {
        const text = try std.mem.concat(gpa, u8, &.{ e.question, " ", e.answer, "\n" });
        defer gpa.free(text);
        const start = e.question.len + 1;
        if (text.len > w.T) return null;
        var scored: usize = 0;
        for (0..w.T) |t| {
            const i = b * w.T + t;
            w.ids[i] = if (t < text.len) text[t] else ' ';
            const target = t + 1 < text.len and t + 1 >= start;
            w.nxt[i] = if (target) text[t + 1] else -1;
            scored += @intFromBool(target);
        }
        w.fillMemory(memory, b);
        return scored;
    }
    /// Reading window b (question or node label), no loss and no memory.
    /// Position of the last byte (where the state is read), or null if too long.
    fn reading(w: Window, text: []const u8, b: usize) ?usize {
        if (text.len == 0 or text.len > w.T) return null;
        for (0..w.T) |t| {
            w.ids[b * w.T + t] = if (t < text.len) text[t] else ' ';
            w.nxt[b * w.T + t] = -1;
        }
        w.fillMemory("", b);
        return text.len - 1;
    }
    fn fillMemory(w: Window, memory: []const u8, b: usize) void {
        for (0..w.M) |j| w.mem[b * w.M + j] = if (j < memory.len) memory[j] else -1;
    }
    fn blank(w: Window, b: usize) void {
        for (0..w.T) |t| {
            w.ids[b * w.T + t] = ' ';
            w.nxt[b * w.T + t] = -1;
        }
        w.fillMemory("", b);
    }
};

/// Model, window buffers and the passes shared by all three subcommands.
const Kg = struct {
    sess: session.Session,
    w: Window,
    B: usize,
    T: usize,
    M: usize,
    D: usize,
    R: usize,
    gpa: std.mem.Allocator,
    last: []c_int,
    rows: []f32,

    fn init(gpa: std.mem.Allocator, io: std.Io, cfg: config.Cfg) !Kg {
        const B: usize = @intCast(cfg.batch);
        const D: usize = @intCast(cfg.dim);
        var kg = Kg{
            .sess = session.Session.init(gpa, io, cfg, kernels_ptx) catch return gpuFail(),
            .B = B, .T = @intCast(cfg.seqlen), .M = @intCast(cfg.mem_len), .D = D,
            .R = @intCast(cfg.mem_rdim), .gpa = gpa, .w = undefined,
            .last = try gpa.alloc(c_int, B), .rows = try gpa.alloc(f32, B * D),
        };
        kg.sess.attach();
        kg.w = try Window.init(gpa, B, kg.T, kg.M);
        return kg;
    }
    fn forward(kg: *Kg, loss: *f32, ce: *f32) !void {
        const window = kg.sess.forwardMem(kg.w.ids, kg.w.nxt, kg.w.mem) catch return gpuFail();
        loss.* = window.total;
        ce.* = window.ce;
    }
    /// Forward one reading batch (questions or labels); rows at the last bytes.
    fn readBatch(kg: *Kg, texts: []const []const u8) ![]f32 {
        for (0..kg.B) |b| {
            kg.last[b] = -1;
            if (b < texts.len) kg.last[b] = if (kg.w.reading(texts[b], b)) |p| @intCast(p) else -1;
            if (kg.last[b] < 0) _ = kg.w.reading(" ", b);
        }
        var loss: f32 = 0;
        var ce: f32 = 0;
        try kg.forward(&loss, &ce);
        @memset(kg.rows, 0);
        kg.sess.readRows(kg.last, kg.rows) catch return gpuFail();
        return kg.rows;
    }
    fn master(kg: *Kg, which: usize) ![]f32 {
        const j = kg.sess.retrievalParam(which).?;
        const out = try kg.gpa.alloc(f32, kg.sess.paramSize(j));
        kg.sess.paramMaster(j, out) catch return gpuFail();
        return out;
    }
    /// Unit keys (nodes, R) of the whole index, from the label bytes.
    fn indexKeys(kg: *Kg, nodes: []const Node, Wk: []const f32) ![]f32 {
        const keys = try kg.gpa.alloc(f32, nodes.len * kg.R);
        @memset(keys, 0);
        var first: usize = 0;
        while (first < nodes.len) : (first += kg.B) {
            var batch: std.ArrayList([]const u8) = .empty;
            for (first..@min(nodes.len, first + kg.B)) |i| try batch.append(kg.gpa, nodes[i].label);
            const rows = try kg.readBatch(batch.items);
            for (0..batch.items.len) |b| {
                if (kg.last[b] < 0) continue; // label longer than seqlen: never retrieved
                const u = try retrieval.headUnit(kg.gpa, Wk, rows[b * kg.D ..][0..kg.D], kg.R, kg.D);
                defer kg.gpa.free(u);
                @memcpy(keys[(first + b) * kg.R ..][0..kg.R], u);
            }
        }
        return keys;
    }
    /// uniform_int_distribution<size_t>(0, len - 1)
    fn rngPick(kg: *Kg, rng: *stdrand.Mt19937, len: usize) usize {
        _ = kg;
        return @intCast(rng.uniform(len - 1));
    }
    /// Backward of a reading batch with the retrieval gradient on the read rows.
    fn backwardRows(kg: *Kg, texts: []const []const u8, drows: []const f32, dX: []f32, rweight: f32) !void {
        _ = try kg.readBatch(texts);
        @memset(dX, 0);
        for (0..kg.B) |b| {
            if (kg.last[b] < 0) continue;
            const at = (b * kg.T + @as(usize, @intCast(kg.last[b]))) * kg.D;
            for (0..kg.D) |d| dX[at + d] = rweight * drows[b * kg.D + d];
        }
        kg.sess.backwardExt(dX) catch return gpuFail();
        kg.sess.accAdd(null) catch return gpuFail();
    }
    fn score(kg: *Kg, keys: []const f32, q: []const f32, k: usize) f32 {
        var s: f32 = 0;
        for (0..kg.R) |r| s += q[r] * keys[k * kg.R + r];
        return s;
    }
};

const Scored = struct { s: f32, k: usize };

/// Best `top` candidates, ties resolved like the C++ partial_sort/insert.
fn topScores(gpa: std.mem.Allocator, kg: *Kg, keys: []const f32, q: []const f32, nodes_len: usize, top: usize) ![]Scored {
    var best: std.ArrayList(Scored) = .empty;
    for (0..nodes_len) |k| {
        const s = kg.score(keys, q, k);
        if (best.items.len < top or s > best.items[best.items.len - 1].s) {
            var at: usize = best.items.len;
            for (best.items, 0..) |e, i| if (!(e.s > s)) {
                at = i;
                break;
            };
            try best.insert(gpa, at, .{ .s = s, .k = k });
            if (best.items.len > top) _ = best.pop();
        }
    }
    return best.items;
}

// ---- ask ----
fn runAsk(init: std.process.Init, out: Out, argv: []const []const u8) !void {
    const io = init.io;
    const gpa = init.arena.allocator();
    const path = try gpa.dupeZ(u8, argv[2]);
    var nodes_path: []const u8 = "";
    var question: []const u8 = "";
    var top: usize = 5;
    var maxlen: usize = 80;
    var temp: f32 = 0;
    for (argv[3..]) |a| {
        const eq = std.mem.indexOfScalar(u8, a, '=') orelse {
            question = a;
            continue;
        };
        const k = a[0..eq];
        const v = a[eq + 1 ..];
        if (std.mem.eql(u8, k, "nodes")) nodes_path = v
        else if (std.mem.eql(u8, k, "top")) top = @intCast(std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{}))
        else if (std.mem.eql(u8, k, "maxlen")) maxlen = @intCast(std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{}))
        else if (std.mem.eql(u8, k, "temp")) temp = std.fmt.parseFloat(f32, v) catch return fail("stof", .{})
        else return fail("unknown option: {s}", .{k});
    }
    const cfg = checkpoint.readConfig(gpa, io, path) catch return ckptFail();
    const mem: i64 = cfg.mem;
    const rdim: i64 = cfg.mem_rdim;
    if (mem == 0 or rdim <= 0 or nodes_path.len == 0)
        return fail("ask needs a stage-2 checkpoint (mem_rdim > 0) and nodes=NODES.tsv", .{});
    const nodes = try loadNodes(gpa, io, nodes_path);
    var kg = try Kg.init(gpa, io, cfg);
    checkpoint.loadWeights(gpa, io, path, &kg.sess.m, cfg) catch return ckptFail();

    var ebuf: [256]u8 = undefined;
    var ew = std.Io.File.stderr().writerStreaming(io, &ebuf);
    try ew.interface.print("indexing {d} nodes...\n", .{nodes.len});
    try ew.interface.flush();
    const Wk = try kg.master(1);
    const Wq = try kg.master(0);
    const keys = try kg.indexKeys(nodes, Wk);
    var rng = stdrand.Mt19937.init(1);

    const answer = struct {
        fn f(k: *Kg, o: Out, g: std.mem.Allocator, ns: []const Node, ks: []const f32, wq: []const f32,
             r: *stdrand.Mt19937, q: []const u8, topn: usize, mx: usize, tp: f32) !void {
            const rows = try k.readBatch(&.{q});
            if (k.last[0] < 0) {
                try o.print("(question longer than seqlen=%d)\n", .{@as(c_int, @intCast(k.T))});
                return;
            }
            const qv = try retrieval.headUnit(g, wq, rows[0..k.D], k.R, k.D);
            const shown = @min(topn, ns.len);
            const best = try topScores(g, k, ks, qv, ns.len, shown);
            try o.print("retrieved:", .{});
            for (best[0..shown], 0..) |e, i| {
                const label = try g.dupeZ(u8, ns[e.k].label);
                try o.print(" %s (%.3f)%s", .{ label.ptr, @as(f64, e.s), if (i + 1 < shown) ",".ptr else "\n".ptr });
            }
            const winner = ns[best[0].k];
            try o.print("memory:    %s\n", .{(try g.dupeZ(u8, winner.memory)).ptr});
            // Generate "question answer\n": one window forward per byte.
            var buf: std.ArrayList(u8) = .empty;
            try buf.appendSlice(g, q);
            try buf.append(g, ' ');
            var produced: std.ArrayList(u8) = .empty;
            const logits = try g.alloc(f32, k.B * k.T * 256);
            for (0..mx) |_| {
                if (buf.items.len >= k.T) break;
                for (0..k.T) |t| {
                    k.w.ids[t] = if (t < buf.items.len) buf.items[t] else ' ';
                    k.w.nxt[t] = -1;
                }
                k.w.fillMemory(winner.memory, 0);
                for (1..k.B) |b| _ = k.w.reading(" ", b);
                var loss: f32 = 0;
                var ce: f32 = 0;
                try k.forward(&loss, &ce);
                k.sess.logits(logits) catch return gpuFail();
                const row = logits[(buf.items.len - 1) * 256 ..][0..256];
                var pick: usize = 0;
                if (tp <= 0) {
                    for (row, 0..) |v, c| if (v > row[pick]) {
                        pick = c;
                    };
                } else {
                    var p: [256]f64 = undefined;
                    var sum: f64 = 0;
                    var m: f32 = -1e30;
                    for (row) |v| m = @max(m, v);
                    for (row, 0..) |v, c| {
                        p[c] = exp((v - m) / tp);
                        sum += p[c];
                    }
                    var rv = r.canonical() * sum;
                    pick = 0;
                    while (pick < 255) : (pick += 1) {
                        rv -= p[pick];
                        if (!(rv > 0)) break;
                    }
                }
                if (pick == '\n') break;
                try buf.append(g, @intCast(pick));
                try produced.append(g, @intCast(pick));
            }
            try o.print("answer:    %s\n", .{(try g.dupeZ(u8, produced.items)).ptr});
        }
    }.f;

    if (question.len != 0) {
        try answer(&kg, out, gpa, nodes, keys, Wq, &rng, question, top, maxlen, temp);
        return;
    }
    const tty = std.Io.File.stdin().isTty(io) catch false;
    var inbuf: [4096]u8 = undefined;
    var reader = std.Io.File.stdin().readerStreaming(io, &inbuf);
    while (true) {
        if (tty) try out.print("question> ", .{});
        const line = reader.interface.takeDelimiterExclusive('\n') catch break;
        if (std.mem.eql(u8, line, "/quit")) break;
        if (line.len != 0) try answer(&kg, out, gpa, nodes, keys, Wq, &rng, line, top, maxlen, temp);
    }
}

fn run(init: std.process.Init, out: Out) !void {
    const io = init.io;
    const gpa = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(gpa);
    if (argv.len >= 3 and std.mem.eql(u8, argv[1], "ask")) return runAsk(init, out, argv);
    if (argv.len < 4 or (!std.mem.eql(u8, argv[1], "train") and !std.mem.eql(u8, argv[1], "eval"))) {
        try out.print("usage: kgtrain train DATA.tsv CKPT [steps=N] [saveevery=N] [nodes=NODES.tsv] [key=value ...]\n" ++
            "       kgtrain eval  DATA.tsv CKPT [memory=on|off|shuffled|retrieved] [nodes=NODES.tsv] [show=N]\n" ++
            "       kgtrain ask   CKPT nodes=NODES.tsv [\"question\"] [top=5] [maxlen=80] [temp=0]\n", .{});
        return error.Usage;
    }
    const training = std.mem.eql(u8, argv[1], "train");
    const path = try gpa.dupeZ(u8, argv[3]);
    var data_size: u64 = 0;
    var data_hash: u64 = 0;
    const examples = try loadExamples(gpa, io, argv[2], &data_size, &data_hash);
    const exists = blk: {
        const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch break :blk false;
        f.close(io);
        break :blk true;
    };
    if (!training and !exists) return fail("evaluation requires an existing checkpoint", .{});
    var cfg = if (exists) checkpoint.readConfig(gpa, io, path) catch return ckptFail() else config.Cfg{};
    if (!exists) cfg.mem = 1;
    const saved = try config.text(gpa, cfg);
    var steps: u64 = 0;
    var saveevery: u64 = 500;
    var show: usize = 5;
    var negbatches: i64 = 1;
    var tau: f32 = 0.05;
    var rweight: f32 = 1;
    var memory_mode: []const u8 = "on";
    var nodes_path: []const u8 = "";
    for (argv[4..]) |a| {
        const eq = std.mem.indexOfScalar(u8, a, '=') orelse return fail("expected key=value", .{});
        const k = a[0..eq];
        const v = a[eq + 1 ..];
        if (std.mem.eql(u8, k, "steps")) steps = @intCast(std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{}))
        else if (std.mem.eql(u8, k, "saveevery")) saveevery = @intCast(std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{}))
        else if (std.mem.eql(u8, k, "memory")) memory_mode = v
        else if (std.mem.eql(u8, k, "show")) show = @intCast(std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{}))
        else if (std.mem.eql(u8, k, "nodes")) nodes_path = v
        else if (std.mem.eql(u8, k, "negbatches")) negbatches = std.fmt.parseInt(i64, v, 10) catch return fail("stoi", .{})
        else if (std.mem.eql(u8, k, "tau")) tau = std.fmt.parseFloat(f32, v) catch return fail("stof", .{})
        else if (std.mem.eql(u8, k, "rweight")) rweight = std.fmt.parseFloat(f32, v) catch return fail("stof", .{})
        else config.set(&cfg, k, v) catch return cfgFail();
    }
    const modes = [_][]const u8{ "on", "off", "shuffled", "retrieved" };
    var known = false;
    for (modes) |m| known = known or std.mem.eql(u8, memory_mode, m);
    if (!known) return fail("memory must be on, off, shuffled or retrieved", .{});
    config.validate(cfg) catch return cfgFail();
    const text = try config.text(gpa, cfg);
    if (cfg.mem == 0)
        return fail("kgtrain requires mem=1", .{});
    if (exists and !std.mem.eql(u8, text, saved))
        return fail("configuration differs from checkpoint; choose a new checkpoint path", .{});
    const rdim: i64 = cfg.mem_rdim;
    const with_retrieval = rdim > 0 and nodes_path.len != 0;
    if (std.mem.eql(u8, memory_mode, "retrieved") and !with_retrieval)
        return fail("memory=retrieved requires a checkpoint with mem_rdim > 0 and nodes=...", .{});
    if (negbatches < 0 or tau <= 0) return fail("invalid retrieval options", .{});
    var nodes: []Node = &.{};
    var node_of = std.StringHashMap(usize).init(gpa);
    if (with_retrieval) {
        nodes = try loadNodes(gpa, io, nodes_path);
        for (nodes, 0..) |n, i| try node_of.put(n.qid, i);
    }
    var kg = try Kg.init(gpa, io, cfg);
    var progress = checkpoint.Progress{};
    if (exists) checkpoint.load(gpa, io, path, &kg.sess.m, &kg.sess.state, &progress) catch return ckptFail();
    if (training and exists and (progress.data_size != data_size or progress.data_hash != data_hash))
        return fail("resume dataset differs from checkpoint", .{});
    progress.data_size = data_size;
    progress.data_hash = data_hash;
    progress.cursor = 0;
    const B = kg.B;
    const T = kg.T;
    const D = kg.D;
    const R = kg.R;
    const N = B * T;
    const losses = try gpa.alloc(f32, N);
    session.installStopHandler();
    const seed: u32 = @bitCast(cfg.seed);

    if (training) {
        try out.print("kgtrain: %zu examples, D=%d L=%d mem_len=%d heads=%d every=%d B=%d T=%d step=%llu%s\n", .{
            examples.len, @as(c_int, @intCast(D)),
            @as(c_int, cfg.layers),
            @as(c_int, @intCast(kg.M)),
            @as(c_int, cfg.mem_heads),
            @as(c_int, cfg.mem_every),
            @as(c_int, @intCast(B)), @as(c_int, @intCast(T)), @as(c_ulonglong, progress.step),
            if (with_retrieval) " retrieval".ptr else "".ptr,
        });
        const dX = try gpa.alloc(f32, N * D);
        const begin = progress.step;
        var window_ce: f64 = 0;
        var window_rloss: f64 = 0;
        var window_racc: f64 = 0;
        var window_count: i64 = 0;
        var rsteps: i64 = 0;
        while (!session.stopRequested() and (steps == 0 or progress.step - begin < steps)) {
            var rng = stdrand.Mt19937.init(seed *% 2654435761 +% @as(u32, @truncate(progress.step))); // resumable
            var chosen: std.ArrayList(usize) = .empty;
            var seen: std.ArrayList([]const u8) = .empty;
            for (0..B) |b| {
                var tries: usize = 0;
                while (true) : (tries += 1) {
                    if (tries > 10000) return fail("examples do not fit into seqlen", .{});
                    const i = kg.rngPick(&rng, examples.len);
                    const e = examples[i];
                    if (with_retrieval) {
                        const at = node_of.get(e.subject) orelse continue;
                        if (kg.w.reading(e.question, b) == null) continue;
                        if (kg.w.reading(nodes[at].label, b) == null) continue;
                    }
                    if (try kg.w.answer(gpa, e, e.memory, b) == null) continue;
                    try chosen.append(gpa, i);
                    try seen.append(gpa, e.subject);
                    break;
                }
            }
            kg.sess.accZero() catch return gpuFail();
            // Answer pass with the true facts (stage 1).
            for (0..B) |b| _ = try kg.w.answer(gpa, examples[chosen.items[b]], examples[chosen.items[b]].memory, b);
            var loss: f32 = 0;
            var ce: f32 = 0;
            try kg.forward(&loss, &ce);
            kg.sess.losses(losses) catch return gpuFail();
            for (losses, kg.w.nxt) |l, t| if (t >= 0) {
                window_ce += l;
                window_count += 1;
            };
            if (!std.math.isFinite(loss)) return fail("non-finite loss; update refused", .{});
            kg.sess.backwardExt(null) catch return gpuFail();
            kg.sess.accAdd(null) catch return gpuFail();
            if (with_retrieval) {
                // Candidates: each distinct subject once (duplicates share their
                // positive), then random other nodes as negatives.
                var questions: std.ArrayList([]const u8) = .empty;
                var subjects: std.ArrayList([]const u8) = .empty;
                const labels = try gpa.alloc(std.ArrayList([]const u8), @intCast(1 + negbatches));
                @memset(labels, .empty);
                const pos = try gpa.alloc(usize, B);
                for (0..B) |b| {
                    const e = examples[chosen.items[b]];
                    try questions.append(gpa, e.question);
                    pos[b] = subjects.items.len;
                    for (subjects.items, 0..) |s, i| if (std.mem.eql(u8, s, e.subject)) {
                        pos[b] = i;
                        break;
                    };
                    if (pos[b] == subjects.items.len) {
                        try subjects.append(gpa, e.subject);
                        try labels[0].append(gpa, nodes[node_of.get(e.subject).?].label);
                    }
                }
                for (labels) |*batch| {
                    var tries: usize = 0;
                    while (batch.items.len < B) : (tries += 1) {
                        const n = nodes[kg.rngPick(&rng, nodes.len)];
                        var fresh = true;
                        for (seen.items) |s| if (std.mem.eql(u8, s, n.qid)) {
                            fresh = false;
                            break;
                        };
                        if ((fresh or tries > 1000) and n.label.len <= T) try batch.append(gpa, n.label);
                    }
                }
                const xq = try gpa.dupe(f32, try kg.readBatch(questions.items));
                const y = try gpa.alloc(f32, labels.len * B * D);
                for (labels, 0..) |batch, j| @memcpy(y[j * B * D ..][0 .. B * D], try kg.readBatch(batch.items));
                var rb = retrieval.Batch{ .B = B, .C = B * labels.len, .D = D, .R = R, .tau = tau };
                const dx = try gpa.alloc(f32, B * D);
                const dy = try gpa.alloc(f32, labels.len * B * D);
                const dWq = try gpa.alloc(f64, R * D);
                const dWk = try gpa.alloc(f64, R * D);
                @memset(dWq, 0);
                @memset(dWk, 0);
                try retrieval.loss(gpa, &rb, xq, y, pos, try kg.master(0), try kg.master(1), dx, dy, dWq, dWk);
                window_rloss += rb.loss;
                window_racc += rb.accuracy;
                rsteps += 1;
                // Backward of the reading batches with the retrieval gradient on the read rows.
                for (labels, 0..) |batch, j| try kg.backwardRows(batch.items, dy[j * B * D ..][0 .. B * D], dX, rweight);
                try kg.backwardRows(questions.items, dx, dX, rweight);
                for ([_]struct { c_int, []const f64 }{ .{ 0, dWq }, .{ 1, dWk } }) |pair| {
                    const j = kg.sess.retrievalParam(@intCast(pair[0])).?;
                    const h = try gpa.alloc(f32, pair[1].len);
                    for (h, pair[1]) |*e, g| e.* = rweight * @as(f32, @floatCast(g));
                    kg.sess.setParamGrad(j, h) catch return gpuFail();
                    kg.sess.accAdd(j) catch return gpuFail();
                }
            }
            kg.sess.accStore() catch return gpuFail();
            kg.sess.optimizerStep(@intCast(progress.step)) catch |e| return if (e == error.NonFiniteGradient)
                fail("non-finite gradient; update refused", .{})
            else
                gpuFail();
            progress.step += 1;
            if (progress.step % 20 == 0) {
                try out.print("step=%llu answer_ce=%.4f", .{ @as(c_ulonglong, progress.step), window_ce / @as(f64, @floatFromInt(@max(window_count, 1))) });
                if (rsteps != 0) try out.print(" retrieval_loss=%.4f batch_acc=%.3f", .{
                    window_rloss / @as(f64, @floatFromInt(rsteps)), window_racc / @as(f64, @floatFromInt(rsteps)),
                });
                try out.print("\n", .{});
                window_ce = 0;
                window_rloss = 0;
                window_racc = 0;
                window_count = 0;
                rsteps = 0;
            }
            if (saveevery != 0 and progress.step % saveevery == 0) {
                kg.sess.resetState() catch return gpuFail();
                checkpoint.save(gpa, io, path, &kg.sess.m, &kg.sess.state, &progress) catch return ckptFail();
            }
        }
        kg.sess.resetState() catch return gpuFail();
        checkpoint.save(gpa, io, path, &kg.sess.m, &kg.sess.state, &progress) catch return ckptFail();
        try out.print("{\"mode\":\"train\",\"steps\":%llu}\n", .{@as(c_ulonglong, progress.step - begin)});
        return;
    }

    // ---- evaluation ----
    const n = examples.len;
    var keys: []f32 = &.{};
    var Wq: []f32 = &.{};
    if (std.mem.eql(u8, memory_mode, "retrieved")) {
        keys = try kg.indexKeys(nodes, try kg.master(1));
        Wq = try kg.master(0);
    }
    var exact: i64 = 0;
    var evaluated: i64 = 0;
    var skipped: i64 = 0;
    var count: i64 = 0;
    var top1: i64 = 0;
    var top1_label: i64 = 0;
    var top5: i64 = 0;
    var ce_sum: f64 = 0;
    const logits = try gpa.alloc(f32, N * 256);
    const memory = try gpa.alloc([]const u8, B);
    const retrieved_label = try gpa.alloc([]const u8, B);
    const slot = try gpa.alloc(?usize, B);
    var first: usize = 0;
    while (first < n) : (first += B) {
        @memset(memory, "");
        @memset(retrieved_label, "");
        if (std.mem.eql(u8, memory_mode, "retrieved")) {
            var questions: std.ArrayList([]const u8) = .empty;
            for (0..B) |b| if (first + b < n) try questions.append(gpa, examples[first + b].question);
            const rows = try kg.readBatch(questions.items);
            for (0..questions.items.len) |b| {
                if (kg.last[b] < 0) continue;
                const q = try retrieval.headUnit(gpa, Wq, rows[b * D ..][0..D], R, D);
                const best = try topScores(gpa, &kg, keys, q, nodes.len, 5);
                const e = examples[first + b];
                const got = nodes[best[0].k];
                memory[b] = got.memory;
                retrieved_label[b] = got.label;
                top1 += @intFromBool(std.mem.eql(u8, got.qid, e.subject));
                if (node_of.get(e.subject)) |at|
                    top1_label += @intFromBool(std.mem.eql(u8, got.label, nodes[at].label));
                for (best) |c| top5 += @intFromBool(std.mem.eql(u8, nodes[c.k].qid, e.subject));
            }
        } else {
            for (0..B) |b| {
                if (first + b >= n) break;
                const i = first + b;
                if (std.mem.eql(u8, memory_mode, "on")) memory[b] = examples[i].memory
                else if (std.mem.eql(u8, memory_mode, "shuffled")) {
                    // another subject's facts, with a different answer
                    for (1..n) |k| {
                        const o = examples[(i + n / 2 + k) % n];
                        if (!std.mem.eql(u8, o.answer, examples[i].answer) and !std.mem.eql(u8, o.memory, examples[i].memory)) {
                            memory[b] = o.memory;
                            break;
                        }
                    }
                }
            }
        }
        for (0..B) |b| {
            const i = first + b;
            const ok = i < n and (try kg.w.answer(gpa, examples[i], memory[b], b)) != null;
            if (i < n and !ok) skipped += 1;
            slot[b] = if (ok) i else null;
            if (!ok) kg.w.blank(b);
        }
        var loss: f32 = 0;
        var ce: f32 = 0;
        try kg.forward(&loss, &ce);
        kg.sess.losses(losses) catch return gpuFail();
        for (losses, kg.w.nxt) |l, t| if (t >= 0) {
            ce_sum += l;
            count += 1;
        };
        kg.sess.logits(logits) catch return gpuFail();
        for (0..B) |b| {
            const at = slot[b] orelse continue;
            var all = true;
            var predicted: std.ArrayList(u8) = .empty;
            for (0..T) |t| {
                const i = b * T + t;
                if (kg.w.nxt[i] < 0) continue;
                const row = logits[i * 256 ..][0..256];
                var best: usize = 0;
                for (row, 0..) |v, k| if (v > row[best]) {
                    best = k;
                };
                all = all and @as(c_int, @intCast(best)) == kg.w.nxt[i];
                if (best != '\n') try predicted.append(gpa, @intCast(best));
            }
            exact += @intFromBool(all);
            evaluated += 1;
            if (evaluated <= show) {
                try out.print("Q: %s | expected: %s | predicted: %s", .{
                    (try gpa.dupeZ(u8, examples[at].question)).ptr,
                    (try gpa.dupeZ(u8, examples[at].answer)).ptr,
                    (try gpa.dupeZ(u8, predicted.items)).ptr,
                });
                if (std.mem.eql(u8, memory_mode, "retrieved"))
                    try out.print(" | retrieved: %s", .{(try gpa.dupeZ(u8, retrieved_label[b])).ptr});
                try out.print("\n", .{});
            }
        }
    }
    if (evaluated == 0) return fail("no example fits into seqlen", .{});
    try out.print("{\"mode\":\"eval\",\"memory\":\"%s\",\"examples\":%ld,\"skipped\":%ld,\"exact\":%.4f,\"answer_ce\":%.4f", .{
        (try gpa.dupeZ(u8, memory_mode)).ptr, @as(c_long, evaluated), @as(c_long, skipped),
        @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(evaluated)),
        ce_sum / @as(f64, @floatFromInt(@max(count, 1))),
    });
    if (std.mem.eql(u8, memory_mode, "retrieved"))
        try out.print(",\"index_nodes\":%zu,\"retrieval_top1\":%.4f,\"retrieval_top1_label\":%.4f,\"retrieval_top5\":%.4f", .{
            nodes.len,
            @as(f64, @floatFromInt(top1)) / @as(f64, @floatFromInt(n)),
            @as(f64, @floatFromInt(top1_label)) / @as(f64, @floatFromInt(n)),
            @as(f64, @floatFromInt(top5)) / @as(f64, @floatFromInt(n)),
        });
    try out.print("}\n", .{});
}

pub fn main(init: std.process.Init) u8 {
    const io = init.io;
    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    run(init, .{ .w = &stdout.interface }) catch |e| {
        stdout.interface.flush() catch {};
        if (e == error.Usage) return 1;
        var ebuf: [600]u8 = undefined;
        var ew = std.Io.File.stderr().writerStreaming(io, &ebuf);
        if (e == error.Fatal) ew.interface.print("error: {s}\n", .{err_msg[0..err_len]}) catch {} else ew.interface.print("error: {s}\n", .{@errorName(e)}) catch {};
        ew.interface.flush() catch {};
        return 1;
    };
    stdout.interface.flush() catch {};
    return 0;
}
