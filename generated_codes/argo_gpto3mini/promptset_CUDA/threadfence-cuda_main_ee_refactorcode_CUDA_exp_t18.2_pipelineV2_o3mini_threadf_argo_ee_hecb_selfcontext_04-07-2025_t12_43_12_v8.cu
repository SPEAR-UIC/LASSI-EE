
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ void sum (
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,       // not used in this kernel, preserved for interface compatibility
    volatile float* __restrict__ result)     // This global array now stores block–level partial sums
{
  extern __shared__ float sdata[];  // dynamically allocated shared memory for tree reduction

  unsigned int bid = blockIdx.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  // Each thread loads one element (or 0 if beyond N) into shared memory.
  sdata[lid] = (gid < N) ? array[gid] : 0.0f;
  __syncthreads();

  // Perform in–block tree reduction to compute the block’s partial sum.
  for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      sdata[lid] += sdata[lid + s];
    }
    __syncthreads();
  }
  
  // Write this block's partial sum to global memory.
  if (lid == 0) {
    result[bid] = sdata[0];
  }
}

// Updated kernel to perform the final reduction over the per–block partial sums.
// This version of finalReduce replaces the last stage of shared–memory reduction with a warp‐level reduction using __shfl_down_sync.
__global__ void finalReduce(volatile float* result, unsigned int numElements) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  float localSum = 0.0f;

  // Each thread accumulates multiple partial sums from global memory.
  for (unsigned int i = tid; i < numElements; i += block_size) {
    localSum += result[i];
  }
  sdata[tid] = localSum;
  __syncthreads();

  // Reduce using shared memory until only 32 threads remain.
  for (unsigned int s = block_size >> 1; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Final warp-level reduction using __shfl_down_sync.
  if (tid < 32) {
    // Copy the value from shared memory to a register.
    float val = sdata[tid];
    // Unroll warp-level reduction.
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid == 0) {
      result[0] = val;
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
  const int grids = (N + blocks - 1) / blocks;

  float* h_array = (float*) malloc(N * sizeof(float));
  float h_sum;

  float* d_result;
  cudaMalloc((void**)&d_result, grids * sizeof(float));

  float* d_array;
  cudaMalloc((void**)&d_array, N * sizeof(float));

  unsigned int* d_count;
  cudaMalloc((void**)&d_count, sizeof(unsigned int));
  cudaMemset(d_count, 0u, sizeof(unsigned int));

  bool ok = true;
  double time = 0.0;

  // Initialize host array with -1.0f.
  for (int i = 0; i < N; i++) {
    h_array[i] = -1.f;
  }

  // Copy the input array to the device only once since it remains constant.
  cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++) {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch the first kernel to compute per–block partial sums.
    sum<<< grids, blocks, blocks * sizeof(float) >>>(d_array, N, d_count, d_result);

    // Launch the second kernel to perform final reduction.
    finalReduce<<< 1, blocks, blocks * sizeof(float) >>>(d_result, grids);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Copy the final result back from the device.
    cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Verify that the computed sum equals (-1.f * N).
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

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
