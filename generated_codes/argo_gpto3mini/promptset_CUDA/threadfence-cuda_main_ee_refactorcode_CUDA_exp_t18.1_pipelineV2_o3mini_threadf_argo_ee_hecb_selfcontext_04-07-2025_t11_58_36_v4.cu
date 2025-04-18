
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Kernel phase 1: Each block reduces its input slice and writes its partial sum.
__global__ void sum (
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,      // kept for interface compatibility; no longer used
    volatile float* __restrict__ result)
{
  extern __shared__ float sdata[];
  
  unsigned int bid = blockIdx.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  // Each thread loads its corresponding input element (or 0 if out-of-bounds)
  float mySum = (gid < N) ? array[gid] : 0.0f;
  sdata[lid] = mySum;
  __syncthreads();

  // Perform tree-based reduction in shared memory.
  for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
    if (lid < stride) {
      sdata[lid] += sdata[lid + stride];
    }
    __syncthreads();
  }
  
  // Write the block's partial sum to global memory.
  if (lid == 0) {
    result[bid] = sdata[0];
    // The count parameter is no longer used.
  }
}

// Kernel phase 2: Reduce an array of partial sums into one sum.
__global__ void finalReduction(
    const float* __restrict__ input,
    const unsigned int N,
    float* __restrict__ output)
{
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  float sum = 0.0f;
  
  if (index < N)
    sum = input[index];
  if (index + blockDim.x < N)
    sum += input[index + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  // Unrolled reduction loop can be applied for further energy efficiency.
  for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // The first thread writes the block's reduced result.
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);

  const int blocks = 256;
  // Compute grid dimension for the first-phase kernel.
  const int grids = (N + blocks - 1) / blocks;

  float* h_array = (float*) malloc(N * sizeof(float));
  float h_sum;

  // Allocate memory for partial sums (result array)
  float* d_result;
  cudaMalloc((void**)&d_result, grids * sizeof(float));

  float* d_array;
  cudaMalloc((void**)&d_array, N * sizeof(float));

  // d_count is still allocated for interface compatibility.
  unsigned int* d_count;
  cudaMalloc((void**)&d_count, sizeof(unsigned int));
  cudaMemset(d_count, 0u, sizeof(unsigned int));

  bool ok = true;
  double time = 0.0;

  // Initialize the host array.
  for (int i = 0; i < N; i++) 
    h_array[i] = -1.f;

  // Transfer input data to the device only once.
  cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate temporary buffer for iterative final reduction.
  // It is sized to hold the maximum number of partial sums.
  float* d_temp;
  cudaMalloc((void**)&d_temp, grids * sizeof(float));

  for (int n = 0; n < repeat; n++) {
    // Create CUDA events for asynchronous timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event.
    cudaEventRecord(start, 0);

    // Phase 1: Launch kernel "sum": each block computes its partial sum.
    sum<<<grids, blocks, blocks * sizeof(float)>>>(d_array, N, d_count, d_result);
    // No blocking synchronization here; kernels launched on the default stream are serialized.

    // Phase 2: Iteratively reduce the partial sums from d_result.
    unsigned int numElements = grids;
    float* input = d_result;
    float* output = d_temp;
    while (numElements > 1) {
      // Use 256 threads per block.
      int threads = 256;
      // Each thread block processes two elements per thread.
      int gridFinal = (numElements + threads * 2 - 1) / (threads * 2);
      finalReduction<<<gridFinal, threads, threads * sizeof(float)>>>(input, numElements, output);
      numElements = gridFinal;
      // Swap the input and output for next iteration.
      input = output;
    }
    // Copy the final result into d_result[0] if not already there.
    cudaMemcpy(d_result, input, sizeof(float), cudaMemcpyDeviceToDevice);

    // Record stop event and synchronize on it.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    // Convert elapsed time from ms to ns and add.
    time += (elapsed_ms * 1e6);

    // Clean up events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the final result to the host.
    cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // The expected sum is -1.f * N
    if (h_sum != -1.f * N) {
      ok = false;
      break;
    }
  }

  if (ok) 
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_count);
  cudaFree(d_temp);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
