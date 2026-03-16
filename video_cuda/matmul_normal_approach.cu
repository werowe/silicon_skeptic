#include <stdio.h>
#include <cuda_runtime.h>

// Basic matrix multiplication kernel: C = A x B.
//
// This is the standard introductory CUDA approach where each thread is
// responsible for computing exactly one output element in matrix C.
// To do that, the thread:
// 1. picks one row from A
// 2. picks one column from B
// 3. computes their dot product

__global__ void matmul_basic(const float *A,
                             const float *B,
                             float *C,
                             int N)
{

    // Convert this thread's block-local position into global matrix coordinates.
    // The result identifies which element of C this thread will compute.
    int row = blockIdx.y * blockDim.y + threadIdx.y; // global row index in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // global col index in C

    // Accumulate the dot product of row 'row' from A and column 'col' from B.
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
    {
        // Access A(row, k) and B(k, col). The matrices are stored as 1D arrays,
        // so 2D coordinates must be flattened into row-major indices.
        float a = A[row * N + k]; // A[row, k]
        float b = B[k * N + col]; // B[k, col]
        sum += a * b;
    }

    // Store the finished dot product in the matching position of C.
    C[row * N + col] = sum;
}

int main()
{
    // Matrix dimension: all matrices are 3x3 in this example.
    const int N = 3;

    // Host-side input matrices stored in flattened row-major order.
    const float A[N * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    const float B[N * N] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1};

    // Host-side output matrix, initialized to zero before computation.
    float C[N * N] = {0};

    // Device pointers for matrices stored in GPU global memory.
    float *d_A, *d_B, *d_C;

    // Allocate GPU memory for the input and output matrices.
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy input data from host memory into device memory.
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // For this small example, use one 3x3 block so there is one thread for
    // each output element in C.
    // In larger programs, block dimensions are usually fixed sizes such as
    // 16x16 or 32x32, and the grid contains many blocks.
    dim3 blockDim(N, N); // 3x3 = 9 threads per block
    dim3 gridDim(1, 1);  // one block

    // Launch the kernel on the GPU.
    matmul_basic<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Wait until the GPU finishes before copying results back.
    cudaDeviceSynchronize();

    // Copy the output matrix from device memory back to host memory.
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result as a readable 2D matrix.
    printf("Result matrix C = A * B:\n");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%.1f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Release GPU memory allocations.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
