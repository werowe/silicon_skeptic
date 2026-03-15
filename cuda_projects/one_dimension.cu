#include <cstdio>
#include <cuda_runtime.h>

// Each thread increments one element in the 1D device array.
__global__ void increment(int *a, int number_of_threads)
{
    // For a single-block launch, threadIdx.x uniquely identifies the element.
    int i = threadIdx.x; // get the thread index within the block

    if (i < number_of_threads) // bounds check
        a[i]++;
}

int main(void)
{

    int number_of_threads = 5;

    // Input data on host and output buffer for results copied back from device.
    int h_host_array[5] = {10, 20, 30, 40, 50};
    int c_host_array[5];

    // Allocate contiguous GPU memory for all array elements.
    int *d_device_array;
    cudaMalloc(&d_device_array, sizeof(int) * number_of_threads);

    // Transfer input data from host memory to device memory.
    cudaMemcpy(d_device_array, h_host_array,
               sizeof(int) * number_of_threads,
               cudaMemcpyHostToDevice);

    // Launch one block with one thread per array element.
    dim3 grid_size(1);
    dim3 block_size(number_of_threads);

    // Run kernel on GPU: each thread increments one value.
    increment<<<grid_size, block_size>>>(d_device_array, number_of_threads);

    // Copy updated values from device back to host.
    cudaMemcpy(c_host_array, d_device_array,
               sizeof(int) * number_of_threads,
               cudaMemcpyDeviceToHost);

    // Print results to verify each element was incremented.
    for (int i = 0; i < number_of_threads; i++)
    {
        printf("c_host_array[%d] = %d\n", i, c_host_array[i]);
    }

    // Free GPU memory after use.
    cudaFree(d_device_array);

    return 0;
}