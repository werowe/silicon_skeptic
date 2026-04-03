// ==========================================================
// NAIVE MATRIX MULTIPLICATION KERNEL (NO TILING)
// ==========================================================

__global__ void matmul_naive(float* A, float* B, float* C, int N) {

    // ------------------------------------------------------
    // Each thread computes ONE element of output matrix C
    // ------------------------------------------------------

    // Compute global row index this thread is responsible for
    // blockIdx.y → which block (row direction)
    // threadIdx.y → which thread inside the block
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute global column index this thread is responsible for
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // ------------------------------------------------------
    // Accumulator for dot product
    // ------------------------------------------------------
    float sum = 0.0f;

    // ------------------------------------------------------
    // Compute dot product of:
    // row of A and column of B
    //
    // This is where the inefficiency happens:
    // Every thread repeatedly loads from global memory
    // ------------------------------------------------------
    for (int k = 0; k < N; k++) {

        // Access A[row][k]
        // Row-major indexing: row * N + k
        float a = A[row * N + k];

        // Access B[k][col]
        float b = B[k * N + col];

        // Multiply and accumulate
        sum += a * b;
    }

    // ------------------------------------------------------
    // Write result to global memory
    // Each thread writes one output element
    // ------------------------------------------------------
    C[row * N + col] = sum;
}
