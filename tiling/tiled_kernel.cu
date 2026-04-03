// ==========================================================
// TILED MATRIX MULTIPLICATION KERNEL (USING SHARED MEMORY)
// ==========================================================

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {

    // ------------------------------------------------------
    // SHARED MEMORY (per block, on-chip, fast)
    //
    // Each block gets its own copy of these tiles.
    // All threads in the block can read/write them.
    // ------------------------------------------------------
    __shared__ float As[TILE][TILE];  // Tile from matrix A
    __shared__ float Bs[TILE][TILE];  // Tile from matrix B

    // ------------------------------------------------------
    // Compute global row and column this thread is responsible for
    //
    // Each thread computes ONE element of output matrix C
    // ------------------------------------------------------
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Accumulator for the final result
    float sum = 0.0f;

    // ------------------------------------------------------
    // LOOP OVER TILES
    //
    // Instead of looping over all k (like naive version),
    // we process the matrices in TILE-sized chunks.
    //
    // Each iteration loads a pair of tiles:
    // - One tile from A (row-wise)
    // - One tile from B (column-wise)
    // ------------------------------------------------------
    for (int m = 0; m < N / TILE; m++) {

        // --------------------------------------------------
        // STEP 1: COOPERATIVE LOADING INTO SHARED MEMORY
        //
        // Each thread loads ONE element from global memory
        // into shared memory.
        //
        // Together, all threads fill the entire tile.
        // --------------------------------------------------

        // Load element from A into shared memory tile As
        // Row stays fixed, column moves across tiles
        As[threadIdx.y][threadIdx.x] =
            A[row * N + m * TILE + threadIdx.x];

        // Load element from B into shared memory tile Bs
        // Column stays fixed, row moves across tiles
        Bs[threadIdx.y][threadIdx.x] =
            B[(m * TILE + threadIdx.y) * N + col];

        // --------------------------------------------------
        // STEP 2: SYNCHRONIZE
        //
        // Ensure ALL threads have finished loading data
        // before any thread starts computing.
        //
        // Without this, some threads could read garbage.
        // --------------------------------------------------
        __syncthreads();

        // --------------------------------------------------
        // STEP 3: COMPUTE USING SHARED MEMORY
        //
        // Now each thread performs a partial dot product
        // using the data in shared memory.
        //
        // IMPORTANT:
        // These accesses are FAST (on-chip),
        // and each loaded value is reused many times.
        // --------------------------------------------------
        #pragma unroll
        for (int k = 0; k < TILE; k++) {

            // Multiply corresponding elements
            // from the A tile and B tile
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // --------------------------------------------------
        // STEP 4: SYNCHRONIZE AGAIN
        //
        // Ensure all threads are done using this tile
        // before we overwrite shared memory in next iteration.
        // --------------------------------------------------
        __syncthreads();
    }

    // ------------------------------------------------------
    // FINAL STEP: WRITE RESULT TO GLOBAL MEMORY
    //
    // Each thread writes one element of C
    // ------------------------------------------------------
    C[row * N + col] = sum;
}
