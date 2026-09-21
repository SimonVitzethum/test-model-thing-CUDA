#pragma once
// Stage-2 retrieval head (host math; B and C are small, D and R a few hundred).
//
// Query  q_b = Wq x_b  from the model state at the last question byte,
// key    k_c = Wk y_c  from the model state at the last byte of a node label.
// Score  s_bc = cos(q_b, k_c) / tau; InfoNCE loss with positive candidate pos[b]:
//   L = mean_b ( logsumexp_c s_bc - s_b,pos[b] )
// Returns the loss and fills the gradients w.r.t. x, y, Wq and Wk (accumulated
// into dWq/dWk, overwritten for dx/dy). Keys are computed from label bytes, so
// nodes never seen in training get meaningful keys.
#include <algorithm>
#include <cmath>
#include <vector>

struct RetrievalBatch {
    int B = 0, C = 0, D = 0, R = 0;
    float tau = 0.05f;
    double loss = 0, accuracy = 0;
};

// out[r] = sum_d W[r*D + d] v[d]
static void head_apply(const std::vector<float>& W, const float* v, int R, int D, std::vector<double>& out) {
    out.assign(R, 0.0);
    for (int r = 0; r < R; ++r) {
        double a = 0;
        for (int d = 0; d < D; ++d) a += (double)W[(size_t)r * D + d] * v[d];
        out[r] = a;
    }
}
static double normalize(std::vector<double>& v) {
    double n = 0;
    for (double e : v) n += e * e;
    n = std::sqrt(std::max(n, 1e-24));
    for (double& e : v) e /= n;
    return n;
}

// Unit vector of W v (the normalized query or key), for scoring the index.
static std::vector<float> head_unit(const std::vector<float>& W, const float* v, int R, int D) {
    std::vector<double> h; head_apply(W, v, R, D, h); normalize(h);
    return std::vector<float>(h.begin(), h.end());
}

static void retrieval_loss(RetrievalBatch& rb, const std::vector<float>& x, const std::vector<float>& y,
                           const std::vector<int>& pos, const std::vector<float>& Wq,
                           const std::vector<float>& Wk, std::vector<float>& dx, std::vector<float>& dy,
                           std::vector<double>& dWq, std::vector<double>& dWk) {
    const int B = rb.B, C = rb.C, D = rb.D, R = rb.R;
    std::vector<std::vector<double>> q(B), k(C);
    std::vector<double> qn(B), kn(C);
    for (int b = 0; b < B; ++b) { head_apply(Wq, &x[(size_t)b * D], R, D, q[b]); qn[b] = normalize(q[b]); }
    for (int c = 0; c < C; ++c) { head_apply(Wk, &y[(size_t)c * D], R, D, k[c]); kn[c] = normalize(k[c]); }
    std::vector<std::vector<double>> dq(B, std::vector<double>(R, 0.0)), dk(C, std::vector<double>(R, 0.0));
    rb.loss = 0; rb.accuracy = 0;
    std::vector<double> s(C);
    for (int b = 0; b < B; ++b) {
        double mx = -1e300; int best = 0;
        for (int c = 0; c < C; ++c) {
            double dot = 0;
            for (int r = 0; r < R; ++r) dot += q[b][r] * k[c][r];
            s[c] = dot / rb.tau;
            if (s[c] > mx) { mx = s[c]; best = c; }
        }
        double sum = 0;
        for (int c = 0; c < C; ++c) sum += std::exp(s[c] - mx);
        rb.loss += mx + std::log(sum) - s[pos[b]];
        rb.accuracy += best == pos[b];
        for (int c = 0; c < C; ++c) {
            double g = (std::exp(s[c] - mx) / sum - (c == pos[b] ? 1.0 : 0.0)) / (B * rb.tau);
            for (int r = 0; r < R; ++r) { dq[b][r] += g * k[c][r]; dk[c][r] += g * q[b][r]; }
        }
    }
    rb.loss /= B; rb.accuracy /= B;
    // Through the normalization: d(v/|v|) = (g - u (u . g)) / |v|.
    auto through_norm = [&](std::vector<double>& g, const std::vector<double>& u, double n) {
        double dot = 0;
        for (int r = 0; r < R; ++r) dot += u[r] * g[r];
        for (int r = 0; r < R; ++r) g[r] = (g[r] - u[r] * dot) / n;
    };
    dx.assign((size_t)B * D, 0.f); dy.assign((size_t)C * D, 0.f);
    dWq.resize((size_t)R * D, 0.0); dWk.resize((size_t)R * D, 0.0);
    for (int b = 0; b < B; ++b) {
        through_norm(dq[b], q[b], qn[b]);
        for (int r = 0; r < R; ++r)
            for (int d = 0; d < D; ++d) {
                dWq[(size_t)r * D + d] += dq[b][r] * x[(size_t)b * D + d];
                dx[(size_t)b * D + d] += (float)(dq[b][r] * Wq[(size_t)r * D + d]);
            }
    }
    for (int c = 0; c < C; ++c) {
        through_norm(dk[c], k[c], kn[c]);
        for (int r = 0; r < R; ++r)
            for (int d = 0; d < D; ++d) {
                dWk[(size_t)r * D + d] += dk[c][r] * y[(size_t)c * D + d];
                dy[(size_t)c * D + d] += (float)(dk[c][r] * Wk[(size_t)r * D + d]);
            }
    }
}
