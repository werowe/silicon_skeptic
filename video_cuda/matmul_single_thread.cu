#include <stdio.h>
#include <cuda_runtime.h>

// This kernel performs the entire matrix multiplication using only one CUDA
// thread. It is intentionally not parallel and is useful as a teaching example
// to show that GPU code can still be written in a sequential style.
//
// Compared with the more common CUDA approach where many threads cooperate,
// this kernel launches just one thread and that single thread computes every
// element of the output matrix C.

__global__ void matmul_single_thread(const float *A,
                                     const float *B,
                                     float *C,
                                     int N)
{

    // Loop over every output row.
    for (int i = 0; i < N; ++i)
    {
        // Loop over every output column.
        for (int j = 0; j < N; ++j)
        {
            // This accumulator will become C[i, j].
            float sum = 0.0f;

            // Compute the dot product of row i from A and column j from B.
            // Because the matrices are flattened into 1D row-major arrays,
            // A[i * N + k] means A(i, k) and B[k * N + j] means B(k, j).
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            // Store the completed value in the output matrix.
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    // All matrices in this example are 3x3.
    const int N = 3;

    // Host-side input matrix A.
    const float A[N * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    // Host-side input matrix B.
    const float B[N * N] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1};

    // Host-side output matrix. It starts at zero and is filled after the
    // kernel finishes and the result is copied back from the GPU.
    float C[N * N] = {0};

    // Device pointers for memory allocated on the GPU.
    float *d_A, *d_B, *d_C;

    // Allocate device memory for the two input matrices and one output matrix.
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy input matrices from host memory to device memory.
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch exactly one block with exactly one thread.
    // That single thread performs the full matrix multiplication by itself.
    matmul_single_thread<<<1, 1>>>(d_A, d_B, d_C, N);

    // Wait for the kernel to finish before accessing the result.
    cudaDeviceSynchronize();

    // Copy the completed output matrix back to the host.
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix in row/column form.
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%.1f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free all allocated GPU memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
