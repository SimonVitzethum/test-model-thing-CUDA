#pragma once
// Minimal JSON line scanner shared by the data tools (C++17, no dependencies).
// Navigates one JSON value by offsets without building a DOM.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

// Navigates one JSON line without building a DOM: positions are offsets into
// the line; skip() jumps over a complete value.
struct Json {
    std::string_view s;
    size_t ws(size_t p) const { while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n')) ++p; return p; }
    size_t skip_string(size_t p) const {  // p at opening quote, returns after closing quote
        for (++p; p < s.size(); ++p) {
            if (s[p] == '\\') ++p;
            else if (s[p] == '"') return p + 1;
        }
        throw std::runtime_error("unterminated string");
    }
    size_t skip(size_t p) const {
        p = ws(p);
        if (p >= s.size()) throw std::runtime_error("unexpected end");
        char c = s[p];
        if (c == '"') return skip_string(p);
        if (c == '{' || c == '[') {
            int depth = 0;
            for (; p < s.size(); ++p) {
                if (s[p] == '"') { p = skip_string(p) - 1; continue; }
                if (s[p] == '{' || s[p] == '[') ++depth;
                else if (s[p] == '}' || s[p] == ']') { if (--depth == 0) return p + 1; }
            }
            throw std::runtime_error("unterminated container");
        }
        while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;  // number/literal
        return p;
    }
    std::string_view raw_string(size_t p) const {  // content between quotes (escapes kept)
        size_t e = skip_string(p);
        return s.substr(p + 1, e - p - 2);
    }
    // Value position of `key` in the object starting at p, or npos.
    size_t find(size_t p, std::string_view key) const {
        if (p == std::string::npos) return p;
        p = ws(p);
        if (p >= s.size() || s[p] != '{') return std::string::npos;
        p = ws(p + 1);
        while (p < s.size() && s[p] != '}') {
            std::string_view k = raw_string(p);
            p = ws(skip_string(p));
            if (s[p] != ':') throw std::runtime_error("expected ':'");
            size_t v = ws(p + 1);
            if (k == key) return v;
            p = ws(skip(v));
            if (s[p] == ',') p = ws(p + 1);
        }
        return std::string::npos;
    }
    template<class F> void each(size_t p, F f) const {  // array elements or object values
        if (p == std::string::npos) return;
        p = ws(p);
        char open = s[p];
        if (open != '[' && open != '{') return;
        p = ws(p + 1);
        char close = open == '[' ? ']' : '}';
        while (p < s.size() && s[p] != close) {
            if (open == '{') { p = ws(skip_string(p)); p = ws(p + 1); }
            f(p);
            p = ws(skip(p));
            if (s[p] == ',') p = ws(p + 1);
        }
    }
};

static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | cp >> 6); out += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | cp >> 12); out += (char)(0x80 | (cp >> 6 & 63)); out += (char)(0x80 | (cp & 63)); }
    else { out += (char)(0xF0 | cp >> 18); out += (char)(0x80 | (cp >> 12 & 63)); out += (char)(0x80 | (cp >> 6 & 63)); out += (char)(0x80 | (cp & 63)); }
}
static std::string unescape(std::string_view r, bool keep_newlines = false) {
    std::string out;
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] != '\\') { out += r[i]; continue; }
        char c = r[++i];
        switch (c) {
            case 'n': out += keep_newlines ? '\n' : ' '; break;
            case 't': case 'r': case 'b': case 'f': out += ' '; break;
            case 'u': {
                uint32_t cp = std::stoul(std::string(r.substr(i + 1, 4)), nullptr, 16); i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < r.size() && r[i + 1] == '\\' && r[i + 2] == 'u') {
                    uint32_t lo = std::stoul(std::string(r.substr(i + 3, 4)), nullptr, 16);
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6;
                }
                append_utf8(out, cp); break;
            }
            default: out += c;
        }
    }
    return out;
}
