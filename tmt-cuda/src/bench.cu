// Standalone normalized recurrence benchmark with CPU reference.
// Aufruf: ./bench [B T D]   (default 16 128 1280 = run2-Fenster)
#include "common.h"
#include <vector>
#include <random>

void cell_forward(const bf16* X, float* S, const float* decay,
                  float* mean, float* rstd, bf16* Y,
                  int B, int T, int D, cudaStream_t stream = 0);

static float sigmoid_h(float x) { return 1.0f / (1.0f + expf(-x)); }

int main(int argc, char** argv) {
    int B = argc > 1 ? atoi(argv[1]) : 16;
    int T = argc > 2 ? atoi(argv[2]) : 128;
    int D = argc > 3 ? atoi(argv[3]) : 1280;
    if (B <= 0 || T <= 0 || D <= 0 || (long)B * T * D > 2147483647L) return 1;
    long N = (long)B * T * D;
    std::printf("cell S1: B=%d T=%d D=%d (%.1fM elem)\n", B, T, D, N / 1e6);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hX(N), hDec(D);
    for (auto& v : hX) v = bf2f(f2bf(dist(rng)));
    for (auto& v : hDec) v = 2.0f;  // wie P1.5-Init

    // CPU-Referenz
    std::vector<float> refY(N);
    for (int b = 0; b < B; ++b)
        for (int d = 0; d < D; ++d) {
            float dec = sigmoid_h(hDec[d]), st = 0.0f;
            for (int t = 0; t < T; ++t) {
                long row = (long)b * T + t;
                st = dec * st + (1.f - dec) * hX[row * D + d];
                refY[row * D + d] = st;  // States zwischenspeichern
            }
        }
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            long row = (long)b * T + t;
            double m = 0, q = 0;
            for (int d = 0; d < D; ++d) { m += refY[row * D + d]; }
            m /= D;
            for (int d = 0; d < D; ++d) {
                double dv = refY[row * D + d] - m; q += dv * dv;
            }
            double rs = 1.0 / sqrt(q / D + 1e-5);
            for (int d = 0; d < D; ++d) {
                float h = (float)((refY[row * D + d] - m) * rs);
                float y = h * sigmoid_h(h) + hX[row * D + d];
                refY[row * D + d] = y;
            }
        }

    // GPU-Buffer
    std::vector<bf16> hXb(N);
    for (long i = 0; i < N; ++i) hXb[i] = f2bf(hX[i]);
    bf16 *X; float *S, *dec, *mean, *rstd; bf16* Y;
    CUDA_CHECK(cudaMalloc(&X, N * 2));
    CUDA_CHECK(cudaMalloc(&S, N * 4));
    CUDA_CHECK(cudaMalloc(&dec, D * 4));
    CUDA_CHECK(cudaMalloc(&mean, (long)B * T * 4));
    CUDA_CHECK(cudaMalloc(&rstd, (long)B * T * 4));
    CUDA_CHECK(cudaMalloc(&Y, N * 2));
    CUDA_CHECK(cudaMemcpy(X, hXb.data(), N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dec, hDec.data(), D * 4, cudaMemcpyHostToDevice));

    // Warmup + Zeit (50 Fenster)
    const int ITERS = 50;
    cell_forward(X, S, dec, mean, rstd, Y, B, T, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t evA, evB;
    CUDA_CHECK(cudaEventCreate(&evA));
    CUDA_CHECK(cudaEventCreate(&evB));
    CUDA_CHECK(cudaEventRecord(evA));
    for (int i = 0; i < ITERS; ++i)
        cell_forward(X, S, dec, mean, rstd, Y, B, T, D);
    CUDA_CHECK(cudaEventRecord(evB));
    CUDA_CHECK(cudaEventSynchronize(evB));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evA, evB));
    std::printf("roh: %.3f ms / %d iters\n", ms, ITERS);
    ms /= ITERS;

    // Korrektheit
    std::vector<bf16> hY(N);
    CUDA_CHECK(cudaMemcpy(hY.data(), Y, N * 2, cudaMemcpyDeviceToHost));
    double mx = 0;
    for (long i = 0; i < N; ++i)
        mx = fmax(mx, fabs(bf2f(hY[i]) - refY[i]));

    // Estimated traffic: X twice, S write + two reads, Y write (excluding stats).
    double bytes = (double)N * (2 + 4 + 4 + 4 + 2 + 2);
    double gbs = bytes / (ms / 1e3) / 1e9;
    std::printf("max-abw vs CPU: %.4f %s\n", mx, mx < 0.05 ? "OK" : "FAIL");
    std::printf("fenster: %.3f ms | estimated %.0f GB/s\n", ms, gbs);
    std::printf("launches/fenster: 3 (statt ~%d eager)\n", T * 2 * 10);
    return mx < 0.05 ? 0 : 2;
}
