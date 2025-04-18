
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ void sum (
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,
    volatile float* __restrict__ result)
{
  __shared__ bool isLastBlockDone;
  __shared__ float partialSum;
  extern __shared__ float sdata[];  // dynamically allocated shared memory for tree reduction

  unsigned int bid = blockIdx.x;
  unsigned int num_blocks = gridDim.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  // Each thread loads one element (or 0 if beyond N) into shared memory
  sdata[lid] = (gid < N) ? array[gid] : 0.0f;
  __syncthreads();

  // Perform in–block tree reduction to compute the block’s partial sum
  for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      sdata[lid] += sdata[lid + s];
    }
    __syncthreads();
  }
  partialSum = sdata[0];

  if (lid == 0) {
    // Store the block partial sum into global memory.
    result[bid] = partialSum;

    // Ensure the write to global memory is visible.
    __threadfence();

    // Signal completion of the block.
    unsigned int value = atomicAdd(count, 1);
    isLastBlockDone = (value == (num_blocks - 1));
  }
  __syncthreads();

  if (isLastBlockDone) {
    // Final reduction: instead of using atomicAdd in a loop,
    // we load the block-level partial sums from global memory into shared memory,
    // then perform a tree reduction within shared memory.
    float localSum = 0.0f;
    // Each thread processes multiple grid elements
    for (unsigned int i = lid; i < num_blocks; i += block_size) {
      localSum += result[i];
    }
    sdata[lid] = localSum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
      if (lid < s) {
        sdata[lid] += sdata[lid + s];
      }
      __syncthreads();
    }

    if (lid == 0) {
      result[0] = sdata[0];
      *count = 0;
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

  for (int i = 0; i < N; i++) h_array[i] = -1.f;

  // Copy the input array to the device only once since it remains constant
  cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++) {

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch kernel with dynamic shared memory allocation for the in–block tree reduction
    sum<<< grids, blocks, blocks * sizeof(float) >>>(d_array, N, d_count, d_result);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    if (h_sum != -1.f * N) {
      ok = false;
      break;
    }
  }

  if (ok) printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_count);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return 0;
}
