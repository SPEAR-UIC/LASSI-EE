
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

  // Perform tree-based reduction in shared memory until the number
  // of active threads drops below 32.
  for (unsigned int stride = block_size / 2; stride >= 32; stride /= 2) {
    if (lid < stride) {
      sdata[lid] += sdata[lid + stride];
    }
    __syncthreads();
  }
  
  // Use warp-level reduction for the final 32 threads.
  if (lid < 32) {
    float val = sdata[lid];
    // Perform warp-level reduction without __syncthreads.
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    // Write the block's partial sum to global memory in thread 0.
    if (lid == 0) {
      result[bid] = val;
      // The count parameter is no longer used.
    }
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
  unsigned int index = blockIdx.x * blockDim.x * 2 + tid;
  float sum = 0.0f;
  
  if (index < N)
    sum = input[index];
  if (index + blockDim.x < N)
    sum += input[index + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  // Reduce in shared memory until we get down to 32 threads.
  for (unsigned int s = blockDim.x / 2; s >= 32; s /= 2) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // When we are within a warp, use warp-level reduction.
  if (tid < 32) {
    float val = sdata[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (tid == 0) {
      output[blockIdx.x] = val;
    }
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
  // Allocate pinned (page-locked) host memory for the final sum.
  float* h_sum;
  cudaHostAlloc((void**)&h_sum, sizeof(float), cudaHostAllocDefault);

  // Allocate memory for partial sums (result array)
  float* d_result;
  cudaMalloc((void**)&d_result, grids * sizeof(float));

  float* d_array;
  cudaMalloc((void**)&d_array, N * sizeof(float));

  // d_count is kept in the kernel for interface compatibility,
  // so we no longer allocate device memory for it.
  // Instead, we will pass a nullptr when launching the kernel.

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

  // Create CUDA events once for asynchronous timing.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Create a stream for asynchronous memory copies.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Threshold to decide whether to perform a single final reduction kernel launch.
  const unsigned int singleLaunchThreshold = 512;

  for (int n = 0; n < repeat; n++) {
    // Record start event.
    cudaEventRecord(start, stream);

    // Phase 1: Launch kernel "sum": each block computes its partial sum.
    // Pass nullptr for d_count since it is no longer used.
    sum<<<grids, blocks, blocks * sizeof(float), stream>>>(d_array, N, nullptr, d_result);

    // Phase 2: Final reduction of the partial sums.
    // If the number of partial sums is low, fuse the final reduction into a single kernel launch.
    if (grids <= singleLaunchThreshold) {
      // Determine the number of threads needed so that 2*threads covers grids elements.
      int threads = 1;
      while (threads * 2 < grids)
        threads *= 2;
      // Launch a single block to reduce all partial sums.
      finalReduction<<<1, threads, threads * sizeof(float), stream>>>(d_result, grids, d_result);
    } else {
      // Iteratively reduce the partial sums from d_result.
      unsigned int numElements = grids;
      float* input = d_result;
      float* output = d_temp;
      while (numElements > 1) {
        // Use 256 threads per block.
        int threads = 256;
        // Each thread block processes two elements per thread.
        int gridFinal = (numElements + threads * 2 - 1) / (threads * 2);
        finalReduction<<<gridFinal, threads, threads * sizeof(float), stream>>>(input, numElements, output);
        numElements = gridFinal;
        // Swap the input and output for next iteration.
        input = output;
      }
      // Copy the final result into d_result[0] if not already there.
      cudaMemcpyAsync(d_result, input, sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Record stop event and synchronize on it.
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    // Convert elapsed time from ms to ns and add.
    time += (elapsed_ms * 1e6);

    // Asynchronously copy the final result to the pinned host memory.
    cudaMemcpyAsync(h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // The expected sum is -1.f * N
    if (*h_sum != -1.f * N) {
      ok = false;
      break;
    }
  }

  // Clean up events after all iterations.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  if (ok) 
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  cudaFreeHost(h_sum);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_temp);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
