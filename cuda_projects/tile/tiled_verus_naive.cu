#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_NAIVE 16
#define TILE_TILED 32
#define BLOCK_X 32
#define BLOCK_Y 8
#define ROWS_PER_THREAD 4

// This file now contains three conceptual implementations:
// 1) A naive baseline kernel (`matmul_naive`) used for performance comparison.
// 2) The original tiled kernel design (described in comments below).
// 3) The optimized tiled kernel (`matmul_tiled`) that uses register tiling
//    and improved shared-memory access patterns.
//
// Why keep the old design documented even after replacing it:
// - It provides a readable bridge from the 16x16 textbook tile to a faster
//   production-style kernel.
// - It makes the measured speedup explainable rather than "magic".
// - It highlights the specific bottlenecks removed by the new kernel.

#define CHECK_CUDA(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ==========================================================
// NAIVE MATRIX MULTIPLICATION KERNEL
// A is M x K
// B is K x N
// C is M x N
// Each thread computes one C[row, col]
// ==========================================================
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ==========================================================
// TILED MATRIX MULTIPLICATION KERNEL (OPTIMIZED)
// A is M x K
// B is K x N
// C is M x N
//
// ------------------------------
// OLD TILED KERNEL (WHAT IT DID)
// ------------------------------
// Old block shape: 16 x 16 = 256 threads
// Old tile shape : 16 x 16
// Old per-thread work: 1 output element
//
// For each K-tile:
// - Every thread loaded 1 value of A and 1 value of B into shared memory.
// - Every thread computed a 16-term dot product for one C[row,col].
//
// Strengths:
// - Shared memory reuse reduced global-memory traffic vs naive GEMM.
// - Simple and correct.
//
// Main limits:
// - Low arithmetic intensity per thread (only one output).
// - More blocks and scheduling overhead for same matrix area.
// - Less reuse of B values inside each thread.
//
// ------------------------------------------
// NEW TILED KERNEL (HOW IT IS FASTER NOW)
// ------------------------------------------
// New block shape: 32 x 8 = 256 threads (same thread count)
// New tile shape : 32 x 32
// New per-thread work: 4 outputs in one column (register tiling)
//
// Core ideas behind the speedup:
// 1) Register tiling:
//    - Each thread accumulates 4 independent sums (`sum[4]`).
//    - One loaded B value contributes to 4 multiply-adds, increasing
//      useful math per memory access.
//
// 2) Larger K-tile (32 instead of 16):
//    - More data reuse from shared memory per tile.
//    - Fewer tile iterations over K for the same problem size.
//
// 3) Shared memory layout tuned for B:
//    - `Bs[TILE_TILED][TILE_TILED + 1]` pads one extra column to reduce
//      shared-memory bank conflicts for the access pattern `Bs[k][tx]`.
//
// 4) Loop unrolling:
//    - `#pragma unroll` helps the compiler expose more instruction-level
//      parallelism and reduce loop-control overhead.
//
// 5) Same 256-thread block, different work partitioning:
//    - Keeps occupancy behavior familiar while increasing per-thread work.
//
// Correctness note:
// - Two `__syncthreads()` calls are required per K-tile iteration:
//   one after cooperative loads, and one after compute, before overwriting
//   shared tiles in the next iteration.
// ==========================================================
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K)
{
    __shared__ float As[TILE_TILED][TILE_TILED];
    __shared__ float Bs[TILE_TILED][TILE_TILED + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_TILED + tx;
    int row_base = blockIdx.y * TILE_TILED + ty;

    float sum[ROWS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int m = 0; m < (K + TILE_TILED - 1) / TILE_TILED; m++) {
        int tile_k = m * TILE_TILED;

        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; i++) {
            int row_a = row_base + i * BLOCK_Y;
            int col_a = tile_k + tx;
            As[ty + i * BLOCK_Y][tx] =
                (row_a < M && col_a < K) ? A[row_a * K + col_a] : 0.0f;

            int row_b = tile_k + ty + i * BLOCK_Y;
            Bs[ty + i * BLOCK_Y][tx] =
                (row_b < K && col < N) ? B[row_b * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_TILED; k++) {
            float b = Bs[k][tx];
            #pragma unroll
            for (int i = 0; i < ROWS_PER_THREAD; i++) {
                sum[i] += As[ty + i * BLOCK_Y][k] * b;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
        int row = row_base + i * BLOCK_Y;
        if (row < M && col < N) {
            C[row * N + col] = sum[i];
        }
    }
}

// ==========================================================
// ORIGINAL TILED KERNEL (BASELINE TILED, FOR COMPARISON)
// This is the simpler 16x16 tiled design that preceded the optimized version.
// Kept here to isolate the performance delta from original→optimized.
//
// Design:
// - Block: 16 x 16 = 256 threads
// - Tile:  16 x 16 shared memory
// - Work:  Each thread computes one C[row, col] element
// - Reuse: Within each K-tile, one A and one B value per thread
//
// This serves as intermediate baseline between naive (1.0x) and
// heavily optimized tiled (3.16x).
// ==========================================================
__global__ void matmul_tiled_original(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K)
{
    __shared__ float As_orig[TILE_NAIVE][TILE_NAIVE];
    __shared__ float Bs_orig[TILE_NAIVE][TILE_NAIVE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Move across the K dimension in TILE_NAIVE-sized chunks
        for (int m = 0; m < (K + TILE_NAIVE - 1) / TILE_NAIVE; m++) {
            int a_col = m * TILE_NAIVE + threadIdx.x;
            int b_row = m * TILE_NAIVE + threadIdx.y;

            // Load one element of A tile into shared memory
            if (row < M && a_col < K) {
                As_orig[threadIdx.y][threadIdx.x] = A[row * K + a_col];
            } else {
                As_orig[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Load one element of B tile into shared memory
            if (b_row < K && col < N) {
                Bs_orig[threadIdx.y][threadIdx.x] = B[b_row * N + col];
            } else {
                Bs_orig[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Compute partial dot product using shared-memory tiles
            #pragma unroll
            for (int k = 0; k < TILE_NAIVE; k++) {
                sum += As_orig[threadIdx.y][k] * Bs_orig[k][threadIdx.x];
            }

            __syncthreads();
        }

        C[row * N + col] = sum;
    }
}

// ==========================================================
// CPU REFERENCE FOR CHECKING
// ==========================================================
void matmul_cpu(const float* A, const float* B, float* C,
                int M, int N, int K)
{
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[r * K + i] * B[i * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

// ==========================================================
// COMPARE TWO ARRAYS
// ==========================================================
bool compare_arrays(const float* a, const float* b, int size, float eps = 1e-3f)
{
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > eps) {
            printf("Mismatch at index %d: a=%f b=%f diff=%f\n",
                   i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// ==========================================================
// TIME A KERNEL OVER MULTIPLE RUNS
// ==========================================================
float time_naive(const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K, dim3 grid, dim3 block, int runs)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / runs;
}

float time_tiled(const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K, dim3 grid, dim3 block, int runs)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / runs;
}

float time_tiled_original(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K, dim3 grid, dim3 block, int runs)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    matmul_tiled_original<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        matmul_tiled_original<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / runs;
}

int main()
{
    // Matrix sizes:
    // A = M x K
    // B = K x N
    // C = M x N
    int M = 4800;
    int K = 4800;
    int N = 4800;

    int runs = 50;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* h_A       = (float*)malloc(bytesA);
    float* h_B       = (float*)malloc(bytesB);
    float* h_C_naive = (float*)malloc(bytesC);
    float* h_C_tiled = (float*)malloc(bytesC);
    float* h_C_tiled_original = (float*)malloc(bytesC);
    float* h_C_cpu   = (float*)malloc(bytesC);

    if (!h_A || !h_B || !h_C_naive || !h_C_tiled || !h_C_tiled_original || !h_C_cpu) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Fill A and B with deterministic values
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < K; c++) {
            h_A[r * K + c] = ((r + c) % 10) * 0.1f;
        }
    }

    for (int r = 0; r < K; r++) {
        for (int c = 0; c < N; c++) {
            h_B[r * N + c] = ((r + c) % 7) * 0.2f;
        }
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytesA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytesB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytesC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block_naive(TILE_NAIVE, TILE_NAIVE);
    dim3 grid_naive((N + TILE_NAIVE - 1) / TILE_NAIVE,
                    (M + TILE_NAIVE - 1) / TILE_NAIVE);

    dim3 block_tiled(BLOCK_X, BLOCK_Y);
    dim3 grid_tiled((N + TILE_TILED - 1) / TILE_TILED,
                    (M + TILE_TILED - 1) / TILE_TILED);

    printf("Matrix sizes:\n");
    printf("A = %d x %d\n", M, K);
    printf("B = %d x %d\n", K, N);
    printf("C = %d x %d\n", M, N);
    printf("\nKernel configurations:\n");
    printf("Naive block = %d x %d, grid = %d x %d\n", TILE_NAIVE, TILE_NAIVE, grid_naive.x, grid_naive.y);
    printf("Tiled Original (16x16) block = %d x %d, grid = %d x %d\n", TILE_NAIVE, TILE_NAIVE, grid_naive.x, grid_naive.y);
    printf("Tiled Optimized (32x32 tile, 32x8 block) = %d x %d, grid = %d x %d\n\n", BLOCK_X, BLOCK_Y, grid_tiled.x, grid_tiled.y);

    // Time naive (baseline)
    float naive_ms = time_naive(d_A, d_B, d_C, M, N, K,
                                grid_naive, block_naive, runs);
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C, bytesC, cudaMemcpyDeviceToHost));

    // Time original tiled kernel (16x16 baseline tile)
    float tiled_original_ms = time_tiled_original(d_A, d_B, d_C, M, N, K,
                                                   grid_naive, block_naive, runs);
    CHECK_CUDA(cudaMemcpy(h_C_tiled_original, d_C, bytesC, cudaMemcpyDeviceToHost));

    // Time optimized tiled kernel
    float tiled_ms = time_tiled(d_A, d_B, d_C, M, N, K,
                                grid_tiled, block_tiled, runs);
    CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C, bytesC, cudaMemcpyDeviceToHost));

    // CPU reference
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);

    bool naive_ok = compare_arrays(h_C_naive, h_C_cpu, M * N);
    bool tiled_original_ok = compare_arrays(h_C_tiled_original, h_C_cpu, M * N);
    bool tiled_ok = compare_arrays(h_C_tiled, h_C_cpu, M * N);

    // GEMM does ~2*M*N*K floating-point ops (multiply + add).
    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double naive_gflops = total_flops / (naive_ms * 1e6);
    double tiled_original_gflops = total_flops / (tiled_original_ms * 1e6);
    double tiled_gflops = total_flops / (tiled_ms * 1e6);

    printf("=== PERFORMANCE COMPARISON ===\n\n");
    printf("Method                     | Avg Time (ms) | GFLOP/s   | Speedup vs Naive\n");
    printf("---------------------------------------------------------------------------\n");
    printf("Naive (baseline)           |   %10.3f  | %9.2f | 1.0x\n", naive_ms, naive_gflops);
    printf("Original Tiled (16x16)     |   %10.3f  | %9.2f | %.2fx\n", tiled_original_ms, tiled_original_gflops, naive_ms / tiled_original_ms);
    printf("Optimized Tiled (32x32)    |   %10.3f  | %9.2f | %.2fx\n", tiled_ms, tiled_gflops, naive_ms / tiled_ms);
    printf("Optimized vs Original      |   %10.3f  | %9.2f | %.2fx\n\n", tiled_ms, tiled_gflops, tiled_original_ms / tiled_ms);

    printf("Correctness:\n");
    printf("  Naive correct:            %s\n", naive_ok ? "YES" : "NO");
    printf("  Original Tiled correct:   %s\n", tiled_original_ok ? "YES" : "NO");
    printf("  Optimized Tiled correct:  %s\n", tiled_ok ? "YES" : "NO");

    printf("\nSample outputs (C[0,0]):\n");
    printf("  Naive:           %f\n", h_C_naive[0 * N + 0]);
    printf("  Original Tiled:  %f\n", h_C_tiled_original[0 * N + 0]);
    printf("  Optimized Tiled: %f\n", h_C_tiled[0 * N + 0]);
    printf("  CPU Reference:   %f\n", h_C_cpu[0 * N + 0]);

    printf("\nSample outputs (C[10,20]):\n");
    printf("  Naive:           %f\n", h_C_naive[10 * N + 20]);
    printf("  Original Tiled:  %f\n", h_C_tiled_original[10 * N + 20]);
    printf("  Optimized Tiled: %f\n", h_C_tiled[10 * N + 20]);
    printf("  CPU Reference:   %f\n", h_C_cpu[10 * N + 20]);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled_original);
    free(h_C_tiled);
    free(h_C_cpu);

    return 0;
}