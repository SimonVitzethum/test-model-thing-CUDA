//! Minimal JSON line scanner shared by the data tools. Navigates one JSON value
//! by byte offsets without building a DOM (port of tools/json_scan.h).
const std = @import("std");

pub const Error = error{Malformed};

pub const Json = struct {
    s: []const u8,

    fn at(self: Json, p: usize) Error!u8 {
        if (p >= self.s.len) return error.Malformed;
        return self.s[p];
    }

    pub fn ws(self: Json, p0: usize) usize {
        var p = p0;
        while (p < self.s.len and (self.s[p] == ' ' or self.s[p] == '\t' or self.s[p] == '\r' or self.s[p] == '\n')) p += 1;
        return p;
    }

    /// `p` at the opening quote; returns the offset after the closing quote.
    pub fn skipString(self: Json, p0: usize) Error!usize {
        var p = p0 + 1;
        while (p < self.s.len) : (p += 1) {
            if (self.s[p] == '\\') {
                p += 1;
            } else if (self.s[p] == '"') return p + 1;
        }
        return error.Malformed;
    }

    /// Offset after the complete value starting at (or after whitespace from) `p0`.
    pub fn skip(self: Json, p0: usize) Error!usize {
        var p = self.ws(p0);
        const c = try self.at(p);
        if (c == '"') return self.skipString(p);
        if (c == '{' or c == '[') {
            var depth: usize = 0;
            while (p < self.s.len) : (p += 1) {
                const d = self.s[p];
                if (d == '"') {
                    p = try self.skipString(p) - 1;
                } else if (d == '{' or d == '[') {
                    depth += 1;
                } else if (d == '}' or d == ']') {
                    depth -= 1;
                    if (depth == 0) return p + 1;
                }
            }
            return error.Malformed;
        }
        while (p < self.s.len and self.s[p] != ',' and self.s[p] != '}' and self.s[p] != ']') p += 1; // number/literal
        return p;
    }

    /// String content between the quotes, escapes kept.
    pub fn rawString(self: Json, p: usize) Error![]const u8 {
        if ((try self.at(p)) != '"') return error.Malformed;
        const e = try self.skipString(p);
        return self.s[p + 1 .. e - 1];
    }

    /// Offset of the value of `key` in the object starting at `p0`, or null.
    pub fn find(self: Json, p0: ?usize, key: []const u8) Error!?usize {
        var p = self.ws(p0 orelse return null);
        if (p >= self.s.len or self.s[p] != '{') return null;
        p = self.ws(p + 1);
        while (p < self.s.len and self.s[p] != '}') {
            const k = try self.rawString(p);
            p = self.ws(try self.skipString(p));
            if ((try self.at(p)) != ':') return error.Malformed;
            const v = self.ws(p + 1);
            if (std.mem.eql(u8, k, key)) return v;
            p = self.ws(try self.skip(v));
            if (p < self.s.len and self.s[p] == ',') p = self.ws(p + 1);
        }
        return null;
    }

    /// Iterates array elements or object values of the container at `p0`.
    pub fn items(self: Json, p0: ?usize) Error!Items {
        const start = p0 orelse return .{ .j = self, .p = 0, .close = 0 };
        const p = self.ws(start);
        const open = try self.at(p);
        if (open != '[' and open != '{') return .{ .j = self, .p = 0, .close = 0 };
        return .{ .j = self, .p = self.ws(p + 1), .close = if (open == '[') ']' else '}', .object = open == '{' };
    }

    pub const Items = struct {
        j: Json,
        p: usize,
        close: u8, // 0: empty iterator
        object: bool = false,

        pub fn next(it: *Items) Error!?usize {
            if (it.close == 0 or it.p >= it.j.s.len or it.j.s[it.p] == it.close) return null;
            if (it.object) {
                it.p = it.j.ws(try it.j.skipString(it.p));
                it.p = it.j.ws(it.p + 1);
            }
            const v = it.p;
            it.p = it.j.ws(try it.j.skip(v));
            if (it.p < it.j.s.len and it.j.s[it.p] == ',') it.p = it.j.ws(it.p + 1);
            return v;
        }
    };
};

pub fn appendUtf8(out: *std.ArrayList(u8), gpa: std.mem.Allocator, cp: u32) !void {
    if (cp < 0x80) {
        try out.append(gpa, @intCast(cp));
    } else if (cp < 0x800) {
        try out.appendSlice(gpa, &.{ @intCast(0xC0 | cp >> 6), @intCast(0x80 | (cp & 63)) });
    } else if (cp < 0x10000) {
        try out.appendSlice(gpa, &.{ @intCast(0xE0 | cp >> 12), @intCast(0x80 | (cp >> 6 & 63)), @intCast(0x80 | (cp & 63)) });
    } else {
        try out.appendSlice(gpa, &.{ @intCast(0xF0 | cp >> 18), @intCast(0x80 | (cp >> 12 & 63)), @intCast(0x80 | (cp >> 6 & 63)), @intCast(0x80 | (cp & 63)) });
    }
}

/// Decodes JSON escapes. \n becomes a newline if `keep_newlines`, else a space;
/// \t \r \b \f become spaces.
pub fn unescape(gpa: std.mem.Allocator, r: []const u8, keep_newlines: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < r.len) : (i += 1) {
        if (r[i] != '\\') {
            try out.append(gpa, r[i]);
            continue;
        }
        i += 1;
        if (i >= r.len) return error.Malformed;
        switch (r[i]) {
            'n' => try out.append(gpa, if (keep_newlines) '\n' else ' '),
            't', 'r', 'b', 'f' => try out.append(gpa, ' '),
            'u' => {
                if (i + 4 >= r.len) return error.Malformed; // four hex digits after 'u'
                var cp: u32 = std.fmt.parseInt(u32, r[i + 1 .. i + 5], 16) catch return error.Malformed;
                i += 4;
                if (cp >= 0xD800 and cp < 0xDC00 and i + 6 < r.len and r[i + 1] == '\\' and r[i + 2] == 'u') {
                    const lo = std.fmt.parseInt(u32, r[i + 3 .. i + 7], 16) catch return error.Malformed;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo -% 0xDC00);
                    i += 6;
                }
                try appendUtf8(&out, gpa, cp);
            },
            else => |c| try out.append(gpa, c),
        }
    }
    return out.toOwnedSlice(gpa);
}

pub fn fnv(s: []const u8) u64 {
    var h: u64 = 14695981039346656037;
    for (s) |c| h = (h ^ c) *% 1099511628211;
    return h;
}

/// Deterministic split by key: fraction `frac` of keys go to the test set.
pub fn inTest(key: []const u8, frac: f64) bool {
    const x: f64 = @floatFromInt(fnv(key) >> 11);
    return x / 9007199254740992.0 < frac; // 2^53
}

test "scanner" {
    const j = Json{ .s = "{\"a\": [1, {\"b\": \"x\\\"y\"}], \"c\": 2}" };
    const a = (try j.find(0, "a")).?;
    var it = try j.items(a);
    try std.testing.expect((try it.next()) != null);
    const obj = (try it.next()).?;
    try std.testing.expectEqualStrings("x\\\"y", try j.rawString((try j.find(obj, "b")).?));
    try std.testing.expect((try it.next()) == null);
    try std.testing.expect((try j.find(0, "zz")) == null);
    const u = try unescape(std.testing.allocator, "a\\u00e4\\nb\\ud83d\\ude00", true);
    defer std.testing.allocator.free(u);
    try std.testing.expectEqualStrings("aä\nb😀", u);
}
