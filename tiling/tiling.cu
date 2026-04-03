__global__ void shared_example(float* input, float* output) {

    // Shared memory (visible to ALL threads in this block)
    __shared__ float tile[16];

    // Thread index within the block
    int tid = threadIdx.x;

    // Global index
    int gid = blockIdx.x * blockDim.x + tid;

    // --------------------------------------------------
    // STEP 1: Each thread loads ONE element
    // from global memory into shared memory
    // --------------------------------------------------
    tile[tid] = input[gid];

    // --------------------------------------------------
    // STEP 2: Synchronize
    // Ensure ALL threads finished loading
    // --------------------------------------------------
    __syncthreads();

    // --------------------------------------------------
    // STEP 3: Use shared memory
    // Now every thread can read ALL elements in tile[]
    // --------------------------------------------------
    float sum = 0.0f;

    for (int i = 0; i < blockDim.x; i++) {
        sum += tile[i];
    }

    // --------------------------------------------------
    // STEP 4: Write result
    // --------------------------------------------------
    output[gid] = sum;
}
