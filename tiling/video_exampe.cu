#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 16

#define CHECK_CUDA(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// C = A * B
// A is M x K
// B is K x N
// C is M x N
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    // Problem size:
    // (4800 x 192) * (192 x 256) = (4800 x 256)
    int M = 4800;
    int K = 192;
    int N = 256;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // Host memory
    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Generate sample data for A (M x K)
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < K; c++) {
            h_A[r * K + c] = ((r + c) % 10) * 0.1f;
        }
    }

    // Generate sample data for B (K x N)
    for (int r = 0; r < K; r++) {
        for (int c = 0; c < N; c++) {
            h_B[r * N + c] = ((r + c) % 7) * 0.2f;
        }
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytesA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytesB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytesC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    // 16x16 thread block
    dim3 block(TILE, TILE);

    // Grid sized for C which is M x N
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Naive: %f ms\n", ms);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    // Print a few values so you can verify something happened
    printf("C[0,0]   = %f\n", h_C[0 * N + 0]);
    printf("C[0,1]   = %f\n", h_C[0 * N + 1]);
    printf("C[10,20] = %f\n", h_C[10 * N + 20]);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
