#include <stdio.h>
#include <cuda_runtime.h>

// Whole matrix multiplication using a single thread on the GPU.

__global__ void matmul_single_thread(const float *A,
                                     const float *B,
                                     float *C,
                                     int N)
{

    // For every row i of A
    for (int i = 0; i < N; ++i)
    {
        // For every column j of B
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;

            // Dot product of row i of A and column j of B
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main()
{
    const int N = 3;
    const float A[N * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    const float B[N * N] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1};

    float C[N * N] = {0};

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    matmul_single_thread<<<1, 1>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%.1f ", C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
