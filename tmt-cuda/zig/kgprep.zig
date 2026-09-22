//! kgprep: knowledge-graph preparation for kgtrain (port of tools/kgprep.cpp;
//! output is byte-identical, including libstdc++'s mt19937/shuffle order).
//!
//!   kgprep dump OUT.tsv [threads=N] [limit=N]      < Wikidata JSON dump on stdin
//!   kgprep qa GRAPH.tsv OUT_PREFIX [mem_len=256] [test_frac=0.1] [seed=1]
//!
//! dump: one streaming pass over a Wikidata JSON dump (one entity per line).
//!   Keeps the item-valued facts (PROPS) of entities with an English Wikipedia
//!   article and the English labels of all items, then writes nodes and facts
//!   whose endpoints are labeled. Lines are parsed by worker threads.
//! qa: question/answer/memory rows split by SUBJECT, plus PREFIX_nodes.tsv
//!   (every labeled node with the memory it would load, for stage 2).
//!
//! Graph TSV:  N <tab> QID <tab> label   |   F <tab> QID <tab> PID <tab> QID
//! QA TSV:     question <tab> answer <tab> memory <tab> subject QID
//! Keep PROPS in sync with tools/wikidata_kg.py (API fetch).
const std = @import("std");
const json = @import("json.zig");
const cli = @import("cli.zig");
const stdrand = @import("stdrand.zig");
const Mt19937 = stdrand.Mt19937;
const shuffle = stdrand.shuffle;

const Prop = struct { pid: []const u8, name: []const u8, question: []const u8 };
const PROPS = [_]Prop{
    .{ .pid = "P36", .name = "capital", .question = "What is the capital of %s?" },
    .{ .pid = "P30", .name = "continent", .question = "On which continent is %s?" },
    .{ .pid = "P38", .name = "currency", .question = "What is the currency of %s?" },
    .{ .pid = "P37", .name = "official language", .question = "What is the official language of %s?" },
    .{ .pid = "P17", .name = "country", .question = "In which country is %s?" },
    .{ .pid = "P19", .name = "place of birth", .question = "Where was %s born?" },
    .{ .pid = "P20", .name = "place of death", .question = "Where did %s die?" },
    .{ .pid = "P27", .name = "citizenship", .question = "What is the citizenship of %s?" },
    .{ .pid = "P106", .name = "occupation", .question = "What was the occupation of %s?" },
    .{ .pid = "P50", .name = "author", .question = "Who wrote %s?" },
    .{ .pid = "P136", .name = "genre", .question = "What is the genre of %s?" },
    .{ .pid = "P495", .name = "country of origin", .question = "Where does %s come from?" },
    .{ .pid = "P186", .name = "made from", .question = "What is %s made of?" },
    .{ .pid = "P57", .name = "director", .question = "Who directed %s?" },
    .{ .pid = "P1412", .name = "language spoken", .question = "Which language did %s speak?" },
};
const MAX_LABEL = 60;

fn propIndex(pid: []const u8) ?usize {
    for (PROPS, 0..) |p, i| if (std.mem.eql(u8, p.pid, pid)) return i;
    return null;
}

// ------------------------------------------------------------------ dump
fn qnum(id: []const u8) u32 {
    if (id.len < 2 or id[0] != 'Q') return 0;
    var v: u64 = 0;
    for (id[1..]) |c| {
        if (c < '0' or c > '9') return 0;
        v = v * 10 + (c - '0');
        if (v > std.math.maxInt(u32)) return 0;
    }
    return @intCast(v);
}

/// Collapses whitespace; null if empty or longer than MAX_LABEL.
fn cleanLabel(gpa: std.mem.Allocator, label: []const u8) !?[]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var space = false;
    for (label) |c| {
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
            space = out.items.len != 0;
            continue;
        }
        if (space) try out.append(gpa, ' ');
        space = false;
        try out.append(gpa, c);
    }
    if (out.items.len == 0 or out.items.len > MAX_LABEL) {
        out.deinit(gpa);
        return null;
    }
    return try out.toOwnedSlice(gpa);
}

const Fact = struct { s: u32, p: u16, o: u32 };
const Label = struct { q: u32, text: []u8 };
const Batch = struct {
    labels: std.ArrayList(Label) = .empty,
    facts: std.ArrayList(Fact) = .empty,
};

fn parseEntity(gpa: std.mem.Allocator, line: []const u8, out: *Batch) !void {
    const j = json.Json{ .s = line };
    const t = try j.find(0, "type") orelse return;
    const id = try j.find(0, "id") orelse return;
    if (!std.mem.eql(u8, try j.rawString(t), "item")) return;
    const q = qnum(try j.rawString(id));
    if (q == 0) return;
    const en = try j.find(try j.find(try j.find(0, "labels"), "en"), "value") orelse return;
    const raw = try json.unescape(gpa, try j.rawString(en), false);
    defer gpa.free(raw);
    const label = try cleanLabel(gpa, raw) orelse return;
    try out.labels.append(gpa, .{ .q = q, .text = label });
    if (try j.find(try j.find(0, "sitelinks"), "enwiki") == null) return;
    const claims = try j.find(0, "claims");
    for (PROPS, 0..) |prop, pi| {
        var values: std.ArrayList(struct { preferred: bool, o: u32 }) = .empty;
        defer values.deinit(gpa);
        var it = try j.items(try j.find(claims, prop.pid));
        while (try it.next()) |st| {
            const r: []const u8 = if (try j.find(st, "rank")) |rank| try j.rawString(rank) else "normal";
            if (std.mem.eql(u8, r, "deprecated")) continue;
            const snak = try j.find(st, "mainsnak");
            const kind = try j.find(snak, "snaktype") orelse continue;
            if (!std.mem.eql(u8, try j.rawString(kind), "value")) continue;
            const v = try j.find(try j.find(try j.find(snak, "datavalue"), "value"), "id") orelse continue;
            const o = qnum(try j.rawString(v));
            if (o != 0) try values.append(gpa, .{ .preferred = std.mem.eql(u8, r, "preferred"), .o = o });
        }
        var preferred = false;
        for (values.items) |v| preferred = preferred or v.preferred;
        for (values.items) |v| if (!preferred or v.preferred) try out.facts.append(gpa, .{ .s = q, .p = @intCast(pi), .o = v.o });
    }
}

const Shared = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList([][]u8) = .empty,
    done: bool = false,
    threads: usize,
    labels: std.ArrayList(Label) = .empty,
    facts: std.ArrayList(Fact) = .empty,
    errors: usize = 0,
    failure: ?anyerror = null,
};

fn worker(sh: *Shared) void {
    const io = sh.io;
    while (true) {
        sh.mutex.lockUncancelable(io);
        while (sh.queue.items.len == 0 and !sh.done) sh.cond.waitUncancelable(io, &sh.mutex);
        if (sh.queue.items.len == 0) {
            sh.mutex.unlock(io);
            return;
        }
        const work = sh.queue.pop().?;
        sh.mutex.unlock(io);
        sh.cond.broadcast(io);
        var b = Batch{};
        var errors: usize = 0;
        for (work) |line| {
            parseEntity(sh.gpa, line, &b) catch |e| switch (e) {
                error.Malformed => errors += 1,
                else => {
                    sh.failure = e;
                    errors += 1;
                },
            };
            sh.gpa.free(line);
        }
        sh.gpa.free(work);
        sh.mutex.lockUncancelable(io);
        sh.labels.appendSlice(sh.gpa, b.labels.items) catch |e| {
            sh.failure = e;
        };
        sh.facts.appendSlice(sh.gpa, b.facts.items) catch |e| {
            sh.failure = e;
        };
        sh.errors += errors;
        sh.mutex.unlock(io);
        b.labels.deinit(sh.gpa);
        b.facts.deinit(sh.gpa);
    }
}

fn cmdDump(io: std.Io, gpa: std.mem.Allocator, out_path: []const u8, threads: usize, limit: usize) !void {
    var sh = Shared{ .io = io, .gpa = gpa, .threads = threads };
    const pool = try gpa.alloc(std.Thread, threads);
    defer gpa.free(pool);
    for (pool) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&sh});

    var in_buf: [1 << 16]u8 = undefined;
    var in = std.Io.File.stdin().reader(io, &in_buf);
    var line_buf = std.Io.Writer.Allocating.init(gpa);
    defer line_buf.deinit();
    var err_buf: [256]u8 = undefined;
    var err = std.Io.File.stderr().writer(io, &err_buf);
    var batch: std.ArrayList([]u8) = .empty;
    var n: usize = 0;
    while (try cli.readLine(&in.interface, &line_buf)) |raw| {
        var line = raw;
        while (line.len > 0 and (line[line.len - 1] == ',' or line[line.len - 1] == '\r' or line[line.len - 1] == ' ')) line = line[0 .. line.len - 1];
        if (line.len < 2 or line[0] != '{') continue;
        try batch.append(gpa, try gpa.dupe(u8, line));
        if (batch.items.len == 2048) {
            sh.mutex.lockUncancelable(io);
            while (sh.queue.items.len >= threads * 4) sh.cond.waitUncancelable(io, &sh.mutex);
            try sh.queue.append(gpa, try batch.toOwnedSlice(gpa));
            sh.mutex.unlock(io);
            sh.cond.broadcast(io);
        }
        n += 1;
        if (n % 1000000 == 0) {
            try err.interface.print("{d} entities read\n", .{n});
            try err.interface.flush();
        }
        if (limit != 0 and n >= limit) break;
    }
    sh.mutex.lockUncancelable(io);
    if (batch.items.len != 0) try sh.queue.append(gpa, try batch.toOwnedSlice(gpa));
    sh.done = true;
    sh.mutex.unlock(io);
    sh.cond.broadcast(io);
    for (pool) |t| t.join();
    batch.deinit(gpa);
    if (sh.failure) |e| return e;

    const labels = sh.labels.items;
    std.mem.sort(Label, labels, {}, struct {
        fn lt(_: void, a: Label, b: Label) bool {
            return a.q < b.q;
        }
    }.lt);
    const labelOf = struct {
        fn f(ls: []const Label, q: u32) ?[]const u8 {
            var lo: usize = 0;
            var hi: usize = ls.len;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (ls[mid].q < q) lo = mid + 1 else hi = mid;
            }
            return if (lo < ls.len and ls[lo].q == q) ls[lo].text else null;
        }
    }.f;
    const facts = sh.facts.items;
    std.mem.sort(Fact, facts, {}, struct {
        fn lt(_: void, a: Fact, b: Fact) bool {
            if (a.s != b.s) return a.s < b.s;
            if (a.p != b.p) return a.p < b.p;
            return a.o < b.o;
        }
    }.lt);
    var kept: std.ArrayList(Fact) = .empty;
    defer kept.deinit(gpa);
    var used: std.ArrayList(u32) = .empty;
    defer used.deinit(gpa);
    for (facts, 0..) |f, i| {
        if (i > 0 and std.meta.eql(f, facts[i - 1])) continue; // unique
        if (f.s != f.o and labelOf(labels, f.s) != null and labelOf(labels, f.o) != null) {
            try kept.append(gpa, f);
            try used.append(gpa, f.s);
            try used.append(gpa, f.o);
        }
    }
    std.mem.sort(u32, used.items, {}, std.sort.asc(u32));
    var nq: usize = 0; // unique, like std::set
    for (used.items) |q| {
        if (nq > 0 and used.items[nq - 1] == q) continue;
        used.items[nq] = q;
        nq += 1;
    }
    const qs = used.items[0..nq];
    const file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var obuf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &obuf);
    const w = &fw.interface;
    for (qs) |q| try w.print("N\tQ{d}\t{s}\n", .{ q, labelOf(labels, q).? });
    for (kept.items) |f| try w.print("F\tQ{d}\t{s}\tQ{d}\n", .{ f.s, PROPS[f.p].pid, f.o });
    try w.flush();
    try err.interface.print("wrote {s}: {d} nodes, {d} facts ({d} entities, {d} parse errors)\n", .{ out_path, qs.len, kept.items.len, n, sh.errors });
    try err.interface.flush();
    for (labels) |l| gpa.free(l.text);
    sh.labels.deinit(gpa);
    sh.facts.deinit(gpa);
    sh.queue.deinit(gpa);
}

// -------------------------------------------------------------------- QA
const Row = struct { question: []const u8, answer: []const u8, memory: []const u8, subject: []const u8 };

fn cmdQa(io: std.Io, arena: std.mem.Allocator, graph: []const u8, prefix: []const u8, mem_len: usize, test_frac: f64, seed: u32) !void {
    const data = try std.Io.Dir.cwd().readFileAlloc(io, graph, arena, .unlimited);
    var labels = std.StringHashMap([]const u8).init(arena);
    var order: std.ArrayList([]const u8) = .empty; // subjects in file order
    const SF = struct { p: usize, o: []const u8 };
    var by_subject = std.StringHashMap(std.ArrayList(SF)).init(arena);
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        var fields: [5][]const u8 = undefined;
        var nf: usize = 0;
        var parts = std.mem.splitScalar(u8, line, '\t');
        while (parts.next()) |part| {
            if (nf == fields.len) break;
            fields[nf] = part;
            nf += 1;
        }
        if (nf > 0 and fields[nf - 1].len == 0 and nf > 1) nf -= 1; // like std::getline: no trailing empty field
        if (nf == 3 and std.mem.eql(u8, fields[0], "N")) {
            try labels.put(fields[1], fields[2]);
        } else if (nf == 4 and std.mem.eql(u8, fields[0], "F") and !std.mem.eql(u8, fields[1], fields[3])) {
            const p = propIndex(fields[2]) orelse continue;
            const gop = try by_subject.getOrPut(fields[1]);
            if (!gop.found_existing) {
                gop.value_ptr.* = .empty;
                try order.append(arena, fields[1]);
            }
            try gop.value_ptr.append(arena, .{ .p = p, .o = fields[3] });
        }
    }
    var rng = Mt19937.init(seed);
    var rows: [2]std.ArrayList(Row) = .{ .empty, .empty };
    var subjects: [2]std.StringHashMap(void) = .{ .init(arena), .init(arena) };
    var memories = std.StringHashMap([]const u8).init(arena);
    for (order.items) |s| {
        const sf = by_subject.get(s).?.items;
        const slabel = labels.get(s) orelse continue;
        var items: std.ArrayList([]const u8) = .empty;
        for (sf) |f| if (labels.get(f.o)) |ol| try items.append(arena, try std.mem.concat(arena, u8, &.{ PROPS[f.p].name, ": ", ol }));
        shuffle([]const u8, items.items, &rng);
        var memory: std.ArrayList(u8) = .empty;
        try memory.appendSlice(arena, slabel);
        try memory.appendSlice(arena, ": ");
        for (items.items) |item| { // whole facts only, within the memory budget
            if (memory.items.len + item.len + 2 > mem_len) break;
            try memory.appendSlice(arena, item);
            try memory.appendSlice(arena, "; ");
        }
        try memories.put(s, memory.items);
        const split: usize = @intFromBool(json.inTest(s, test_frac));
        for (0..PROPS.len) |p| { // std::map<int, std::set<string>>: by property index
            var only: ?[]const u8 = null;
            var count: usize = 0;
            for (sf) |f| {
                if (f.p != p or labels.get(f.o) == null) continue;
                if (only == null or !std.mem.eql(u8, only.?, f.o)) {
                    if (only != null) count = 2 else count = 1; // two distinct objects: multi-valued
                    if (only == null) only = f.o;
                }
            }
            if (count != 1) continue; // multi-valued or absent: no single correct answer
            const answer = labels.get(only.?).?;
            const needle = try std.mem.concat(arena, u8, &.{ PROPS[p].name, ": ", answer, ";" });
            if (std.mem.indexOf(u8, memory.items, needle) == null) continue;
            const pct = std.mem.indexOf(u8, PROPS[p].question, "%s").?;
            const question = try std.mem.concat(arena, u8, &.{ PROPS[p].question[0..pct], slabel, PROPS[p].question[pct + 2 ..] });
            try rows[split].append(arena, .{ .question = question, .answer = answer, .memory = memory.items, .subject = s });
            try subjects[split].put(s, {});
        }
    }
    var err_buf: [512]u8 = undefined;
    var err = std.Io.File.stderr().writer(io, &err_buf);
    const names = [2][]const u8{ "train", "test" };
    for (0..2) |k| {
        shuffle(Row, rows[k].items, &rng);
        const path = try std.mem.concat(arena, u8, &.{ prefix, "_", names[k], ".tsv" });
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var obuf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &obuf);
        for (rows[k].items) |r| try fw.interface.print("{s}\t{s}\t{s}\t{s}\n", .{ r.question, r.answer, r.memory, r.subject });
        try fw.interface.flush();
        try err.interface.print("wrote {s}: {d} examples, {d} subjects\n", .{ path, rows[k].items.len, subjects[k].count() });
    }
    // Retrieval index: every labeled node, ordered by (length, text) of its QID.
    var ids: std.ArrayList([]const u8) = .empty;
    var it = labels.keyIterator();
    while (it.next()) |q| try ids.append(arena, q.*);
    std.mem.sort([]const u8, ids.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return if (a.len != b.len) a.len < b.len else std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    const npath = try std.mem.concat(arena, u8, &.{ prefix, "_nodes.tsv" });
    const nfile = try std.Io.Dir.cwd().createFile(io, npath, .{});
    defer nfile.close(io);
    var nbuf: [1 << 16]u8 = undefined;
    var nw = nfile.writer(io, &nbuf);
    for (ids.items) |q| {
        const l = labels.get(q).?;
        if (memories.get(q)) |mem| try nw.interface.print("{s}\t{s}\t{s}\n", .{ q, l, mem }) else try nw.interface.print("{s}\t{s}\t{s}: \n", .{ q, l, l });
    }
    try nw.interface.flush();
    try err.interface.print("wrote {s}: {d} nodes\n", .{ npath, ids.items.len });
    try err.interface.flush();
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try cli.Args.parse(init.arena.allocator(), init.minimal.args);
    const pos = args.positional.items;
    if (pos.len == 2 and std.mem.eql(u8, pos[0], "dump")) {
        var threads = try args.int(usize, "threads", 0);
        if (threads == 0) threads = @max(1, std.Thread.getCpuCount() catch 1);
        return cmdDump(io, init.gpa, pos[1], threads, try args.int(usize, "limit", 0));
    }
    if (pos.len == 3 and std.mem.eql(u8, pos[0], "qa"))
        return cmdQa(io, init.arena.allocator(), pos[1], pos[2], try args.int(usize, "mem_len", 256), try args.float("test_frac", 0.1), try args.int(u32, "seed", 1));
    return cli.fail(io, "usage: kgprep dump OUT.tsv [threads=N] [limit=N]   < wikidata-dump.json\n" ++
        "       kgprep qa GRAPH.tsv OUT_PREFIX [mem_len=256] [test_frac=0.1] [seed=1]\n", .{});
}
