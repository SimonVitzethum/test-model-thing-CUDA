//! ngram: what is a look-up into the training corpus worth, before any
//! training at all?
//!
//! Section G of MOONSHOTS.md argues that byte prediction on Wikipedia mixes
//! two problems - linguistic structure, which is compressible and learnable
//! from little data, and factual or verbatim content, which is not and which
//! does not fit in 34M parameters. Gradient descent spends most of its
//! capacity on the second. An index over the corpus makes it free.
//!
//! This measures the ceiling on that claim in the cheapest possible way: no
//! training, no change to the model, no retrieval machinery inside it. The
//! model's per-byte losses come from `train mode=eval lossout=...`, and
//! mixing two distributions at the true byte needs only both of their values
//! there:
//!
//!     p = (1-lambda) * p_model + lambda * p_retrieval
//!
//! so the whole experiment is an interpolation against a number the model
//! already produced. If that does not lower held-out bits per byte, nothing
//! built on top of retrieval will either.
//!
//! The index is an n-gram store at several context lengths, queried longest
//! first - the infinite-gram construction. Each entry packs a 56-bit hash of
//! the context with the byte that followed it, so sorting by the whole u64
//! groups equal contexts together *and* sorts their continuations within the
//! group, which makes counting a scan of a contiguous range.
//!
//! On leakage, which is the trap this whole section stands or falls on: the
//! datastore is the training corpus and the queries are held-out bytes, two
//! disjoint files. Near-duplicate passages across Wikipedia articles are not
//! leakage - exploiting them is the entire point - but retrieving over the
//! evaluation set would be, and this tool cannot do it.
//!
//! usage: ngram DATASTORE HELD LOSSFILE [ms=32,24,16,12,8] [minmatch=1]
//!                  [store=0] [limit=0]
const std = @import("std");
const cli = @import("cli.zig");

extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

/// 56 bits of hash in the high bits, the continuation byte in the low eight.
const KEY_SHIFT = 8;

fn hashBytes(b: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (b) |c| {
        h ^= c;
        h *%= 0x100000001b3;
    }
    // Fold to 56 bits so the low byte can carry the continuation. With a
    // hundred million entries the collision probability is a few percent
    // across the whole table, which inflates a handful of counts and
    // changes no conclusion.
    return (h ^ (h >> 33)) >> KEY_SHIFT;
}

/// One context length's store: entries sorted by (hash, next byte).
const Table = struct {
    m: usize,
    e: []u64,

    fn build(gpa: std.mem.Allocator, data: []const u8, m: usize) !Table {
        const n = if (data.len > m) data.len - m else 0;
        const e = try gpa.alloc(u64, n);
        for (0..n) |i| e[i] = (hashBytes(data[i .. i + m]) << KEY_SHIFT) | data[i + m];
        std.sort.pdq(u64, e, {}, std.sort.asc(u64));
        return .{ .m = m, .e = e };
    }

    /// Continuation counts for one context: the total, and how many of them
    /// were the byte actually observed.
    ///
    /// Both are binary searches rather than a scan. Because the entries sort
    /// by hash *and then* by continuation byte, the occurrences of one byte
    /// under one context are themselves a contiguous range - which matters,
    /// because an eight-byte context in a hundred-megabyte store can have
    /// millions of occurrences and scanning them per query would dominate
    /// everything.
    fn look(t: Table, ctx: []const u8, want: u8) struct { total: u32, hit: u32 } {
        const h = hashBytes(ctx) << KEY_SHIFT;
        const lo = std.sort.lowerBound(u64, t.e, h, order);
        const hi = std.sort.upperBound(u64, t.e, h | 0xFF, order);
        if (lo >= hi) return .{ .total = 0, .hit = 0 };
        const wlo = std.sort.lowerBound(u64, t.e, h | want, order);
        const whi = std.sort.upperBound(u64, t.e, h | want, order);
        return .{ .total = @intCast(hi - lo), .hit = @intCast(whi - wlo) };
    }

    fn order(key: u64, item: u64) std.math.Order {
        return std.math.order(key, item);
    }
};

/// Crude but explicit: a byte counts as markup while inside `<...>`,
/// `{{...}}` or `[[...]]`. Wikipedia bytes are full of XML and template
/// boilerplate that an n-gram predicts almost perfectly, and which is not
/// what anyone means by knowledge - so a headline number that mixes the two
/// says more about the dump's format than about retrieval. The split is
/// built into the measurement rather than argued about afterwards.
fn markMarkup(gpa: std.mem.Allocator, d: []const u8) ![]bool {
    const out = try gpa.alloc(bool, d.len);
    var tag: i32 = 0;
    var brace: i32 = 0;
    var brack: i32 = 0;
    var i: usize = 0;
    while (i < d.len) : (i += 1) {
        const c = d[i];
        const nxt: u8 = if (i + 1 < d.len) d[i + 1] else 0;
        if (c == '<') tag += 1;
        if (c == '{' and nxt == '{') brace += 1;
        if (c == '[' and nxt == '[') brack += 1;
        out[i] = tag > 0 or brace > 0 or brack > 0;
        if (c == '>' and tag > 0) tag -= 1;
        if (c == '}' and nxt == '}' and brace > 0) brace -= 1;
        if (c == ']' and nxt == ']' and brack > 0) brack -= 1;
    }
    return out;
}

fn readAll(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const gpa = init.arena.allocator();
    var buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &stdout.interface;
    var b: [1024]u8 = undefined;
    const p = struct {
        fn f(out: *std.Io.Writer, bb: []u8, comptime fmt: [:0]const u8, a: anytype) !void {
            const c = @call(.auto, snprintf, .{ bb.ptr, bb.len, fmt.ptr } ++ a);
            try out.writeAll(bb[0..@intCast(@min(c, bb.len - 1))]);
            try out.flush();
        }
    }.f;

    const argv = try init.minimal.args.toSlice(gpa);
    if (argv.len < 4) return cli.fail(io, "usage: ngram DATASTORE HELD LOSSFILE [ms=...] [store=N]\n", .{});
    const args = try cli.Args.parse(gpa, init.minimal.args);
    const store_mb: usize = try args.int(usize, "store", 0);
    const minmatch: u32 = try args.int(u32, "minmatch", 1);
    const ms_text = args.get("ms", "32,24,16,12,8");

    var ms: [8]usize = undefined;
    var nms: usize = 0;
    var it = std.mem.splitScalar(u8, ms_text, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        if (nms == ms.len) break;
        ms[nms] = std.fmt.parseInt(usize, tok, 10) catch return cli.fail(io, "bad ms\n", .{});
        nms += 1;
    }
    // Longest first: the infinite-gram rule is to use the longest context
    // that occurs at all, because a long exact match on this data is nearly
    // deterministic and a short one is nearly worthless.
    std.mem.sort(usize, ms[0..nms], {}, comptime std.sort.desc(usize));

    var data = try readAll(gpa, io, argv[1]);
    if (store_mb > 0 and data.len > store_mb << 20) data = data[0 .. store_mb << 20];
    const held = try readAll(gpa, io, argv[2]);
    const loss_bytes = try readAll(gpa, io, argv[3]);
    const loss = std.mem.bytesAsSlice(f32, loss_bytes);
    if (loss.len < held.len) return cli.fail(io, "loss file shorter than held data\n", .{});

    try p(w, &b, "retrieval: datastore %.1f MB, held %.1f MB, contexts", .{
        @as(f64, @floatFromInt(data.len)) / 1048576.0,
        @as(f64, @floatFromInt(held.len)) / 1048576.0,
    });
    for (ms[0..nms]) |m| try p(w, &b, " %zu", .{m});
    try p(w, &b, "\n", .{});

    // Eight bytes per position per context length adds up fast: the full
    // 950 MB corpus at seven lengths would want 53 GB. Refuse rather than
    // discover it by being killed.
    const want_gb = @as(f64, @floatFromInt(data.len * nms * 8)) / (1 << 30);
    const max_gb: f64 = try args.float("maxgb", 8.0);
    try p(w, &b, "  index needs %.1f GB\n", .{want_gb});
    if (want_gb > max_gb)
        return cli.fail(io, "index would need {d:.1} GB; raise maxgb or lower store/ms\n", .{want_gb});

    var tables: [8]Table = undefined;
    for (ms[0..nms], 0..) |m, i| {
        tables[i] = try Table.build(gpa, data, m);
        try p(w, &b, "  m=%-3zu %zu entries\n", .{ m, tables[i].e.len });
    }
    const markup = try markMarkup(gpa, held);

    // A unigram from the datastore itself, as the base of the backoff. Free
    // to compute and strictly better than a uniform prior.
    var ucount: [256]u64 = @splat(0);
    for (data) |c| ucount[c] += 1;
    var uni: [256]f64 = undefined;
    for (0..256) |c| uni[c] = (@as(f64, @floatFromInt(ucount[c])) + 1.0) /
        (@as(f64, @floatFromInt(data.len)) + 256.0);

    // One pass over the held bytes: model probability from the loss file,
    // retrieval probability from the corpus.
    //
    // Two retrieval distributions are scored side by side, because they are
    // different claims. `longest` is the raw maximum likelihood at the
    // longest context that occurs at all - sharp, and zero for anything it
    // has not seen, which is why it cannot stand alone. `backoff` is the
    // recursive interpolation
    //
    //     p_m = (c_m(ctx,y) + alpha * p_{m-1}) / (c_m(ctx) + alpha)
    //
    // from the unigram upward, which has no zeros and is a language model in
    // its own right.
    const NL = 16;
    const lambdas = [NL]f64{ 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    var s_long: [NL]f64 = @splat(0);
    var s_back: [NL]f64 = @splat(0);
    var s_prose: [NL]f64 = @splat(0);
    var s_markup: [NL]f64 = @splat(0);
    var n_prose: u64 = 0;
    var n_markup: u64 = 0;
    var n: u64 = 0;
    var covered: u64 = 0;
    var per_m_n: [8]u64 = @splat(0);
    const longest = ms[0];
    const limit: u64 = try args.int(u64, "limit", 0);
    const alpha: f64 = try args.float("alpha", 2.0);

    for (longest..held.len - 1) |off| {
        const l = loss[off];
        if (!std.math.isFinite(l)) continue;
        if (limit > 0 and n >= limit) break;
        const target = held[off + 1];
        const pm: f64 = @exp(-@as(f64, l));

        var pr_long: f64 = 0;
        var have_long = false;
        var pr_back: f64 = uni[target];
        // Shortest context first, each level refining the one below it.
        var best_i: usize = 0;
        var i: usize = nms;
        while (i > 0) {
            i -= 1;
            const m = ms[i];
            const r = tables[i].look(held[off + 1 - m .. off + 1], target);
            // Every context here ends at the same byte, so a shorter one is
            // a suffix of a longer one: if the short context occurs nowhere,
            // no longer context can, and the remaining levels are skipped.
            // Hash collisions can only invent matches, never hide them, so
            // the shortcut is safe.
            if (r.total == 0) break;
            pr_back = (@as(f64, @floatFromInt(r.hit)) + alpha * pr_back) /
                (@as(f64, @floatFromInt(r.total)) + alpha);
            if (r.total >= minmatch) {
                pr_long = @as(f64, @floatFromInt(r.hit)) / @as(f64, @floatFromInt(r.total));
                best_i = i;
                have_long = true;
            }
        }
        if (have_long) {
            covered += 1;
            per_m_n[best_i] += 1; // the longest that matched, not the first
        }
        n += 1;
        const is_markup = markup[off + 1];
        if (is_markup) n_markup += 1 else n_prose += 1;
        for (lambdas, 0..) |lam, k| {
            // Where nothing was retrieved the model stands alone, which is
            // the honest mixture rather than a penalty for a missing index.
            const ml = if (have_long) (1.0 - lam) * pm + lam * pr_long else pm;
            s_long[k] += -@log2(@max(ml, 1e-30));
            const bk = -@log2(@max((1.0 - lam) * pm + lam * pr_back, 1e-30));
            s_back[k] += bk;
            if (is_markup) s_markup[k] += bk else s_prose[k] += bk;
        }
    }

    const fn_: f64 = @floatFromInt(@max(n, 1));
    try p(w, &b, "\nscored %llu bytes, %.1f%% had a context in the store\n",
        .{ @as(c_ulonglong, n), 100.0 * @as(f64, @floatFromInt(covered)) / fn_ });
    for (ms[0..nms], 0..) |m, i| {
        if (per_m_n[i] == 0) continue;
        try p(w, &b, "  longest match m=%-3zu %7llu (%.1f%%)\n",
            .{ m, @as(c_ulonglong, per_m_n[i]), 100.0 * @as(f64, @floatFromInt(per_m_n[i])) / fn_ });
    }
    try p(w, &b, "\n%8s %10s %10s %10s %10s\n",
        .{ "lambda", "longest", "vs model", "backoff", "vs model" });
    const base = s_long[0] / fn_;
    var best: f64 = base;
    var best_lam: f64 = 0;
    for (lambdas, s_long, s_back) |lam, a, c| {
        const bl = a / fn_;
        const bb = c / fn_;
        if (bb < best) {
            best = bb;
            best_lam = lam;
        }
        try p(w, &b, "%8.2f %10.4f %+9.4f %10.4f %+9.4f\n", .{ lam, bl, bl - base, bb, bb - base });
    }
    try p(w, &b, "\nmodel alone %.4f, best mix %.4f at lambda %.2f (%.1f%% better)\n",
        .{ base, best, best_lam, 100.0 * (base - best) / base });

    // Split, because Wikipedia's markup is where an n-gram is unbeatable and
    // where a gain means least. A headline number that does not separate
    // them is mostly a statement about the dump's format.
    const fp: f64 = @floatFromInt(@max(n_prose, 1));
    const fm: f64 = @floatFromInt(@max(n_markup, 1));
    try p(w, &b, "\n%8s %12s %12s   (prose %.0f%% of bytes)\n",
        .{ "lambda", "prose", "markup", 100.0 * fp / fn_ });
    const bp0 = s_prose[0] / fp;
    const bm0 = s_markup[0] / fm;
    for (lambdas, s_prose, s_markup) |lam, a, c| {
        try p(w, &b, "%8.2f %8.4f %+3.4f %8.4f %+3.4f\n",
            .{ lam, a / fp, a / fp - bp0, c / fm, c / fm - bm0 });
    }
    return 0;
}
