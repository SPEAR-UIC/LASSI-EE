
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
  extern __shared__ float sdata[];
  
  // Each block sums a subset of the input array.
  unsigned int bid = blockIdx.x;
  unsigned int num_blocks = gridDim.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  // Each thread loads its input element (or 0 if out-of-bounds) into shared memory.
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
  
  // sdata[0] now holds the block's partial sum.
  if (lid == 0) {
    result[bid] = sdata[0];
    __threadfence(); // Ensure that the global memory store is visible before updating the count.
    unsigned int value = atomicAdd(count, 1);
    isLastBlockDone = (value == (num_blocks - 1));
  }
  
  // Synchronize so that every thread reads the correct value of isLastBlockDone.
  __syncthreads();
  
  if (isLastBlockDone) {
    // The last block to finish performs the final reduction of all partial sums.
    float total = 0.0f;
    for (unsigned int i = lid; i < num_blocks; i += block_size) {
      total += result[i];
    }
    sdata[lid] = total;
    __syncthreads();
    
    // Reduce within the block to obtain the final sum.
    for (unsigned int stride = block_size / 2; stride > 0; stride /= 2) {
      if (lid < stride) {
        sdata[lid] += sdata[lid + stride];
      }
      __syncthreads();
    }
    
    if (lid == 0) {
      result[0] = sdata[0];
      *count = 0; // Reset count for the next kernel call.
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

  // Initialize the host array.
  for (int i = 0; i < N; i++) h_array[i] = -1.f;

  // Transfer input data to the device only once.
  cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  for (int n = 0; n < repeat; n++) {
    // No need to copy d_array again since its contents do not change.

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch kernel with shared memory size equal to block size.
    sum<<<grids, blocks, blocks * sizeof(float)>>>(d_array, N, d_count, d_result);

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
