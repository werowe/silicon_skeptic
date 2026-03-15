
/*
x is a 1 dimension array of threads, each thread processes a single element (vector).
y is a 2 dimension array of threads, each thread processes a 2D location  
z is a 3 dimension array of threads, each thread processes a 3D location  
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void increment(int *a, int rows, int cols) {

    /*
    if we have only one block, we can use threadIdx.x to access the column and 
    threadIdx.y to access the row 

    */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) { // bounds check
        int idx = row * cols + col;
        a[idx]++;
    }
}


int main(void) {

    const int rows = 5;
    const int cols = 2;

    // Allocate host arrays
    int h_host_array[5][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
    int c_host_array[5][2];

    // Allocate device memory for all rows * cols elements
    int *d_device_array;
    cudaMalloc(&d_device_array, sizeof(int) * rows * cols);

    // Copy host data to device
    cudaMemcpy(d_device_array, h_host_array, 
        sizeof(int) * rows * cols, 
        cudaMemcpyHostToDevice);

    /*
   A dim3 variable that specifies the dimensions of a CUDA thread block, 
   defining how many threads are organized along the x, y, and z axes 
   within each block launched on the GPU. 

    block_size = (cols, rows) = (2, 5)
    grid_size.x = (2 + 2 - 1) / 2 = 1
    grid_size.y = (5 + 5 - 1) / 5 = 1
    
    So the kernel launch is effectively <<<(1,1,1), (2,5,1)>>>, 
    which is 1 block containing 10 threads.
  
    grid_size.x is computed as:

    grid_size.x = (cols + block_size.x - 1) / block_size.x

    For your code:

    cols = 2
    block_size.x = 2
    
    So:

    grid_size.x = (2 + 2 - 1) / 2
    grid_size.x = 3 / 2
    grid_size.x = 1 (integer division)  "rounded up"


    */

    dim3 block_size(cols, rows);
    dim3 grid_size((cols + block_size.x - 1) / block_size.x,
                   (rows + block_size.y - 1) / block_size.y);

    // Launch the increment kernel with a 2D thread layout
    increment<<<grid_size, block_size>>>(d_device_array, rows, cols);

    // Copy results back to host
    cudaMemcpy(c_host_array, d_device_array, sizeof(int) * rows * cols, 
    cudaMemcpyDeviceToHost);

    // Print results
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            printf("c_host_array[%d][%d] = %d\n", row, col, c_host_array[row][col]);
        }
    }

    cudaFree(d_device_array);

    return 0;
}