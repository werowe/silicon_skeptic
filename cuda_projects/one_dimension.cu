#include <cstdio>
#include <cuda_runtime.h>


/*
a is a one-dimensional array, and each CUDA thread processes one element of that array.

We can process these in parallel since we are just incrementing each vector 
element.  It doesn't matter what order we do that nor does it depend on the step
that came before it.  
*/

__global__ void increment(int *a, int number_of_threads) {
    int i = threadIdx.x; // get the thread index within the block

    if (i < number_of_threads) // bounds check
        a[i]++;
}



int main(void) {

    int number_of_threads = 5;

    // Allocate host arrays
    int h_host_array[5] = {10, 20, 30, 40, 50};
    int c_host_array[5];

    // Allocate device memory (enough for all 5 ints)
    int *d_device_array;
    cudaMalloc(&d_device_array, sizeof(int) * number_of_threads);

    // Copy host data to device
    cudaMemcpy(d_device_array, h_host_array, 
        sizeof(int) * number_of_threads, 
         cudaMemcpyHostToDevice);

    /*
   A dim3 variable that specifies the dimensions of a CUDA thread block, 
   defining how many threads are organized along the x, y, and z axes 
   within each block launched on the GPU. 
  
    */
    dim3 grid_size(1);
    dim3 block_size(number_of_threads);

    // Launch the increment kernel: 1 block, 5 threads
    increment<<<grid_size, block_size>>>(d_device_array, number_of_threads);

    // Copy results back to host
    cudaMemcpy(c_host_array, d_device_array,
         sizeof(int) * number_of_threads, 
         cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < number_of_threads; i++) {
        printf("c_host_array[%d] = %d\n", i, c_host_array[i]);
    }

    cudaFree(d_device_array);

    return 0;
}