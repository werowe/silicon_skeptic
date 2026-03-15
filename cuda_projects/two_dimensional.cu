

#include <cstdio>
#include <cuda_runtime.h>

// Each thread maps to exactly one (row, col) element in the 2D array.
__global__ void increment(int *a, int rows, int cols)
{

    // Compute this thread's column and row from block/thread coordinates.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    { // bounds check
        // Flatten 2D coordinates into a 1D linear index for contiguous memory.
        int idx = row * cols + col;
        a[idx]++;
    }
}

int main(void)
{

    const int rows = 5;
    const int cols = 2;

    // Input matrix on host and output buffer to receive device results.
    int h_host_array[rows][cols] = {
        {0, 1},
        {2, 3},
        {4, 5},
        {6, 7},
        {8, 9}};

    int c_host_array[rows][cols];

    //before incrementing, print all elements so we can verify the initial values.
     for (int row = 0; row < rows; row++)
    {
        printf("row %d:", row);
        for (int col = 0; col < cols; col++)
        {
            printf(" %d", h_host_array[row][col]);
        }
        printf("\n");
    }

    printf("\n");

    // Allocate a contiguous device buffer large enough for rows * cols integers.
    int *d_device_array;
    cudaMalloc(&d_device_array, sizeof(int) * rows * cols);

    // Copy initial matrix data from host RAM to device global memory.
    cudaMemcpy(d_device_array, h_host_array,
               sizeof(int) * rows * cols,
               cudaMemcpyHostToDevice);

    // Launch one block sized to the matrix dimensions for this small example.
    // For larger matrices, you would typically use a fixed block size (e.g., 16x16)
    // and a multi-block grid.
    dim3 block_size(cols, rows);
    dim3 grid_size((cols + block_size.x - 1) / block_size.x,
                   (rows + block_size.y - 1) / block_size.y);

    // Execute the kernel to increment each element by 1 on the GPU.
    increment<<<grid_size, block_size>>>(d_device_array, rows, cols);

    // Copy updated results back from device memory into the host output array.
    cudaMemcpy(c_host_array, d_device_array, sizeof(int) * rows * cols,
               cudaMemcpyDeviceToHost);

    // Print all elements so we can verify each value was incremented.
    for (int row = 0; row < rows; row++)
    {
        printf("row %d:", row);
        for (int col = 0; col < cols; col++)
        {
            printf(" %d", c_host_array[row][col]);
        }
        printf("\n");
    }

    // Release GPU memory once we're done.
    cudaFree(d_device_array);

    return 0;
}