
#include <stdio.h>
#include <cuda_runtime.h>

// This kernel performs square matrix multiplication: C = A x B.
// The matrices are stored in row-major 1D arrays, so a logical element
// at (row, col) must be translated to a linear index when accessed.
//
// For this teaching example, each CUDA thread computes exactly one output
// element C[row, col] by taking the dot product of:
// - one row from A
// - one column from B

__global__ void single_thread(const float *A,
                              const float *B,
                              float *C,
                              int N)

{
    // Map this thread's 2D position within the block/grid to a matrix element.
    // row selects which row of A to use.
    // col selects which column of B to use.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Accumulate the dot product for C[row, col].
    // sum = A[row, 0] * B[0, col]
    //     + A[row, 1] * B[1, col]
    //     + ...
    //     + A[row, N-1] * B[N-1, col]
    float sum = 0.0f;
    int k = 0;
    while (k < N)
    {
        // A[row * N + k] accesses the element in A at (row, k).
        // B[k * N + col] accesses the element in B at (k, col).
        sum += A[row * N + k] * B[k * N + col];
        ++k;
    }

    // Store the finished dot product into the output matrix at (row, col).
    C[row * N + col] = sum;
}


int main()
{

    // Matrix dimension. Because N = 3, each matrix is 3x3.
    const int N = 3;

    // Host-side input matrices.
    // These are flattened into 1D arrays, but represent:
    // A = [1 2 3
    //      4 5 6
    //      7 8 9]
    const float A[N * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    // B = [9 8 7
    //      6 5 4
    //      3 2 1]
    const float B[N * N] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1};

    // Host-side output matrix, initialized to 0 before the GPU fills it.
    float C[N * N] = {0};

    // Device pointers for matrices stored in GPU global memory.
    float *d_A, *d_B, *d_C;

    // Allocate space on the GPU for A, B, and C.
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy the input matrices from host memory to device memory.
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch exactly one block containing N x N threads.
    // With N = 3, this creates a 3x3 thread block, so there is one thread
    // for every output element in C.
    //
    // This is convenient for a tiny example, but for larger matrices you would
    // typically use a fixed tile size such as 16x16 and a grid with many blocks.
    single_thread<<<1, dim3(N, N)>>>(d_A, d_B, d_C, N);

    // Wait for the kernel to finish before reading the result back.
    cudaDeviceSynchronize();

    // Copy the completed output matrix from GPU memory back to host memory.
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix in a 2D layout even though it is stored linearly.
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            printf("%.1f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free all GPU memory allocations.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
