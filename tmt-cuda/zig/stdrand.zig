//! Random numbers that reproduce libstdc++ exactly, so Zig tools produce the
//! same output as the C++ versions for the same seed: std::mt19937,
//! uniform_int_distribution (Lemire), std::shuffle and generate_canonical.
const std = @import("std");

/// std::mt19937 (32-bit Mersenne Twister) with the standard seeding.
pub const Mt19937 = struct {
    mt: [624]u32,
    i: usize = 624,

    pub fn init(seed: u32) Mt19937 {
        var r = Mt19937{ .mt = undefined };
        r.mt[0] = seed;
        for (1..624) |k| r.mt[k] = 1812433253 *% (r.mt[k - 1] ^ (r.mt[k - 1] >> 30)) +% @as(u32, @intCast(k));
        return r;
    }
    pub fn next(r: *Mt19937) u32 {
        if (r.i >= 624) {
            for (0..624) |k| {
                const y = (r.mt[k] & 0x80000000) | (r.mt[(k + 1) % 624] & 0x7fffffff);
                r.mt[k] = r.mt[(k + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            r.i = 0;
        }
        var y = r.mt[r.i];
        r.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }
    /// uniform_int_distribution<unsigned long>{0, b}: Lemire's method on the
    /// 32-bit engine (libstdc++ path for b < 2^32 - 1).
    pub fn uniform(r: *Mt19937, b: u64) u64 {
        if (b == 0xffffffff) return r.next();
        std.debug.assert(b < 0xffffffff);
        const range: u32 = @intCast(b + 1);
        var product: u64 = @as(u64, r.next()) * range;
        var low: u32 = @truncate(product);
        if (low < range) {
            const threshold: u32 = (0 -% range) % range;
            while (low < threshold) {
                product = @as(u64, r.next()) * range;
                low = @truncate(product);
            }
        }
        return product >> 32;
    }
    /// uniform_real_distribution<double>(0, 1): libstdc++ (GCC 16)
    /// generate_canonical for a 32-bit engine: two draws, 64 bits, / 2^64,
    /// redrawn if the result rounds to 1.
    pub fn canonical(r: *Mt19937) f64 {
        while (true) {
            const lo: u64 = r.next();
            const sum = lo | (@as(u64, r.next()) << 32);
            const v = @as(f64, @floatFromInt(sum)) / 18446744073709551616.0;
            if (v < 1.0) return v;
        }
    }
};

/// std::shuffle as implemented by libstdc++ (two swaps per draw when n*n fits
/// the engine's range).
pub fn shuffle(comptime T: type, items: []T, r: *Mt19937) void {
    const n: u64 = items.len;
    if (n == 0) return;
    if (0xffffffff / n >= n) {
        var i: usize = 1;
        if (n % 2 == 0) {
            std.mem.swap(T, &items[1], &items[@intCast(r.uniform(1))]);
            i = 2;
        }
        while (i != items.len) {
            const swap_range: u64 = i + 1;
            const x = r.uniform(swap_range * (swap_range + 1) - 1);
            std.mem.swap(T, &items[i], &items[@intCast(x / (swap_range + 1))]);
            std.mem.swap(T, &items[i + 1], &items[@intCast(x % (swap_range + 1))]);
            i += 2;
        }
        return;
    }
    for (1..items.len) |i| std.mem.swap(T, &items[i], &items[@intCast(r.uniform(i))]);
}

test "mt19937 matches the C++ standard" {
    var r = Mt19937.init(5489);
    for (0..9999) |_| _ = r.next();
    try std.testing.expectEqual(@as(u32, 4123659995), r.next()); // required by [rand.predef]
}

