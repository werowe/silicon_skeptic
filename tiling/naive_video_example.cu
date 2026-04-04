#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Naive kernel (no tiling)
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Dot product
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main()
{
    int M = 4800;
    int K = 192;
    int N = 256;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* A = (float*)malloc(sizeA);
    float* B = (float*)malloc(sizeB);
    float* C = (float*)malloc(sizeC);

    // Generate simple data
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // 16x16 threads per block
    dim3 block(16, 16);

    // Grid covers C (M x N)
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // Warm-up run
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 20;

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaGetLastError();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Naive avg over %d runs: %f ms\n", runs, ms / runs);

    // Copy result back
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Quick correctness check
    // Each element should be K (since all ones)
    printf("C[0] = %f (expected %d)\n", C[0], K);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(A);
    free(B);
    free(C);

    return 0;
}
