// ------------------------------------------------------
// CUDA EVENTS FOR GPU TIMING
// These measure time ON THE GPU (not CPU time)
// ------------------------------------------------------
cudaEvent_t start, stop;

// Create event objects
cudaEventCreate(&start);
cudaEventCreate(&stop);

// ------------------------------------------------------
// Record start event
// ------------------------------------------------------
cudaEventRecord(start);

// Launch kernel
// <<<grid, block>>> defines how many threads run
matmul_naive<<<grid, block>>>(A, B, C, N);

// ------------------------------------------------------
// Record stop event
// Important: kernel launch is async, so we measure properly
// ------------------------------------------------------
cudaEventRecord(stop);

// Wait until GPU finishes executing kernel
cudaEventSynchronize(stop);

// ------------------------------------------------------
// Compute elapsed time in milliseconds
// ------------------------------------------------------
float ms;
cudaEventElapsedTime(&ms, start, stop);

// Print result
printf("Naive: %f ms\n", ms);
