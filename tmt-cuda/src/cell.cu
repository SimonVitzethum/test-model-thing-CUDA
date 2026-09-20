// Fused persistente Recurrent-Cell (S1, Forward).
//
// state = sigmoid(decay)*state + x        (persistent über T, Register)
// y = silu(LN(state)) + x                 (Residual wie TMT-Layer)
//
// Layout (B,T,D), D-kontinuierlich -> coalesced. Ein Thread = ein (b,d),
// loopt über die ganze Sequenz: genau 3 Launches pro Fenster (state,
// stats, out) statt ~T*L*10.
#include "common.h"

// Pass 1: Zeit-Loop, schreibt States (fp32) für die Norm.
__global__ void state_pass_kernel(const bf16* __restrict__ X,
                                  float* __restrict__ S,
                                  const float* __restrict__ decay,
                                  int B, int T, int D) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    float dec = sigmoid_f(decay[d]);
    float state = 0.0f;
    for (int t = 0; t < T; ++t) {
        int idx = (b * T + t) * D + d;
        state = dec * state + bf2f(X[idx]);
        S[idx] = state;
    }
}

// Pass 2: ein Block pro (b,t) reduziert Mittelwert + rstd über D.
__global__ void stats_kernel(const float* __restrict__ S,
                             float* __restrict__ mean,
                             float* __restrict__ rstd,
                             int B, int T, int D) {
    int row = blockIdx.x;               // row = b*T + t
    if (row >= B * T) return;
    __shared__ float buf[256];
    __shared__ float total_sum;
    float sum = 0.0f, sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = S[row * D + d];
        sum += v; sq += v * v;
    }
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) total_sum = buf[0];
    __syncthreads();
    buf[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = total_sum / D;
        float v = buf[0] / D - m * m;
        mean[row] = m;
        rstd[row] = rsqrtf(v + 1e-5f);
    }
}

// Pass 3: Norm + SiLU + Residual, wieder persistent über T.
__global__ void out_pass_kernel(const bf16* __restrict__ X,
                                const float* __restrict__ S,
                                const float* __restrict__ mean,
                                const float* __restrict__ rstd,
                                bf16* __restrict__ Y,
                                int B, int T, int D) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;
    for (int t = 0; t < T; ++t) {
        int row = b * T + t;
        int idx = row * D + d;
        float h = (S[idx] - mean[row]) * rstd[row];
        Y[idx] = f2bf(silu_f(h) + bf2f(X[idx]));
    }
}

void cell_forward(const bf16* X, float* S, const float* decay,
                  float* mean, float* rstd, bf16* Y,
                  int B, int T, int D, cudaStream_t stream = 0) {
    const int TPB = 256;
    dim3 grid1(B, (D + TPB - 1) / TPB);
    state_pass_kernel<<<grid1, TPB, 0, stream>>>(X, S, decay, B, T, D);
    stats_kernel<<<B * T, TPB, 0, stream>>>(S, mean, rstd, B, T, D);
    out_pass_kernel<<<grid1, TPB, 0, stream>>>(X, S, mean, rstd, Y, B, T, D);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        std::printf("LAUNCH-FAIL: %s (grid1=%d,%d stats=%d)\n",
                    cudaGetErrorString(le), grid1.x, grid1.y, B * T);
        exit(3);
    }
}
