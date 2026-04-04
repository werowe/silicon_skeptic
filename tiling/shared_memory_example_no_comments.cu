__global__ void shared_example(float* input, float* output) {
 
    __shared__ float tile[16];     // Shared memory (visible to ALL threads in this block)

    int tid = threadIdx.x;    // Thread index within the block
   
    int gid = blockIdx.x * blockDim.x + tid;    // Global index

    tile[tid] = input[gid]; // Each thread loads ONE element  from global memory into shared memory
 
    __syncthreads();   // Ensure ALL threads finished loading

    float sum = 0.0f;

    for (int i = 0; i < blockDim.x; i++) {    
        sum += tile[i];     // Now every thread can read ALL elements in tile[]
    }

    output[gid] = sum;  // STEP 4: Write result
}
