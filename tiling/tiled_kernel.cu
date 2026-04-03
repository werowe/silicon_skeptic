__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int m = 0; m < N/TILE; m++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + m*TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(m*TILE + threadIdx.y) * N + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
