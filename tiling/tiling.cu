__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}


cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
matmul_naive<<<grid, block>>>(A, B, C, N);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);

printf("Naive: %f ms\n", ms);



