#include <stdio.h>
#include <cuda_runtime.h>

// most common approach.  whole matrix multiplication.
// Each thread computes one element of C

__global__ void matmul_basic(const float *A,
                             const float *B,
                             float *C,
                             int N)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y; // global row index in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // global col index in C

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
    {
        float a = A[row * N + k]; // A[row, k]
        float b = B[k * N + col]; // B[k, col]
        sum += a * b;
    }

    // Write the result into the correct location of C
    C[row * N + col] = sum;
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

    dim3 blockDim(N, N); // 3x3 = 9 threads per block
    dim3 gridDim(1, 1);  // one block

    matmul_basic<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result matrix C = A * B:\n");
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
