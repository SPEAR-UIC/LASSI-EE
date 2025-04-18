
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ void sum (
    const float* __restrict__ array,
    const int N,
    unsigned int * __restrict__ count,
    volatile float* __restrict__ result)
{
  __shared__ bool isLastBlockDone;
  extern __shared__ float sdata[];

  // Each block sums a subset of the input array.
  unsigned int bid = blockIdx.x;
  unsigned int num_blocks = gridDim.x;
  unsigned int block_size = blockDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int gid = bid * block_size + tid;

  // Load one element per thread (or 0 if outside bounds)
  float myVal = (gid < N) ? array[gid] : 0.0f;
  sdata[tid] = myVal;
  __syncthreads();

  // In-block tree reduction using shared memory
  for (int s = block_size >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Thread 0 of each block stores the partial sum.
  if (tid == 0) {
    result[bid] = sdata[0];
    __threadfence();

    // Signal completion of block reduction.
    unsigned int value = atomicAdd(count, 1);
    isLastBlockDone = (value == (num_blocks - 1));
  }
  __syncthreads();

  // The last finished block performs a final reduction over all block partial sums.
  if (isLastBlockDone) {
    // Reuse shared memory: load partial sums from result into sdata.
    if (tid < num_blocks)
      sdata[tid] = result[tid];
    else
      sdata[tid] = 0.0f;
    __syncthreads();

    // Perform tree reduction across the partial sums.
    for (int s = block_size >> 1; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
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

  float* h_array = (float*) malloc (N * sizeof(float));
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

  for (int i = 0; i < N; i++) 
    h_array[i] = -1.f;

  for (int n = 0; n < repeat; n++) {
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch the kernel with dynamic shared memory allocation:
    sum <<< grids, blocks, blocks * sizeof(float) >>> (d_array, N, d_count, d_result);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

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
