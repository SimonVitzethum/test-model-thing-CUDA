// Byte-by-byte generation with a persistent recurrent state (shared by sample
// and chat). Uses exactly the training forward (forwardWindow with B=1, T=1);
// the stream state (carry + MLA cache) lives as long as the generator.
const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const config = @import("config.zig");
const ckpt = @import("checkpoint.zig");
const u = @import("util.zig");

pub const Generator = struct {
    file_cfg: config.Cfg,
    m: model.Model,
    state: model.State,
    rng: std.Random.DefaultPrng,
    logits: [256]f32 = @splat(0),
    stop_prob: f32 = 0, // sigmoid of the stop head for the last fed byte
    ready: bool = false, // logits hold the prediction after the last fed byte
    fed: u64 = 0, // bytes fed since the last reset

    pub fn init(path: [:0]const u8, seed: u64) Generator {
        const file_cfg = ckpt.readConfig(path) catch u.die("{s}", .{ckpt.last_error});
        var cfg = file_cfg;
        cfg.batch = 1; // single step (runtime keys only)
        cfg.seqlen = 1;
        config.validate(cfg) catch u.die("{s}", .{config.last_error});
        var g = Generator{ .file_cfg = file_cfg, .m = model.build(cfg), .state = undefined, .rng = .init(seed) };
        g.state = model.State.init(&g.m);
        ckpt.loadWeights(path, &g.m, file_cfg) catch u.die("{s}", .{ckpt.last_error});
        return g;
    }
    /// An untrained stop head (stop=0) gives random logits and is ignored.
    pub fn stopTrained(self: *const Generator) bool {
        return self.file_cfg.stop > 0;
    }

    pub fn feed(self: *Generator, byte: u8) void {
        const m0 = mx.mark();
        defer mx.release(m0);
        const ids = [_]i32{byte};
        const nxt = [_]i32{-1};
        const end = [_]i32{0};
        _ = model.forwardWindow(&self.m, &self.state, model.makeBatch(self.m.c, &ids, &nxt, &end), .{});
        mx.toF32(self.m.logits, &self.logits);
        self.stop_prob = 1 / (1 + @exp(-mx.item(self.m.stoplog)));
        self.ready = true;
        self.fed += 1;
    }
    pub fn feedText(self: *Generator, text: []const u8) void {
        for (text) |b| self.feed(b);
    }

    /// Next byte from the prediction after the last fed byte (argmax if temp <= 0).
    pub fn sample(self: *Generator, temp: f32) u8 {
        if (!self.ready) u.die("nothing fed yet", .{});
        var best: usize = 0;
        for (self.logits, 0..) |v, i| if (v > self.logits[best]) {
            best = i;
        };
        _ = &best;
        if (temp <= 0) return @intCast(best);
        const mxv = self.logits[best];
        var p: [256]f64 = undefined;
        var sum: f64 = 0;
        for (self.logits, 0..) |v, i| {
            p[i] = @exp((v - mxv) / temp);
            sum += p[i];
        }
        var r = self.rng.random().float(f64) * sum;
        for (p, 0..) |v, i| {
            r -= v;
            if (r <= 0) return @intCast(i);
        }
        return 255;
    }

    pub fn reset(self: *Generator) void {
        self.state.reset();
        self.ready = false;
        self.fed = 0;
    }
};
