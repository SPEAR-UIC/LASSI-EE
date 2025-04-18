
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ void sum (
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,       // preserved for interface compatibility, unused in kernel
    volatile float* __restrict__ result)     // Global array to store per–block partial sums
{
  extern __shared__ float sdata[];  // dynamically allocated shared memory for tree reduction

  unsigned int bid = blockIdx.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  // Each thread loads one element (or 0 if beyond N) into shared memory.
  sdata[lid] = (gid < N) ? array[gid] : 0.0f;
  __syncthreads();

  // Perform in–block reduction until only a single warp remains.
  for (unsigned int s = block_size >> 1; s > 32; s >>= 1) {
    if (lid < s) {
      sdata[lid] += sdata[lid + s];
    }
    __syncthreads();
  }

  // Use warp–level primitives for the final reduction stage.
  if (lid < 32) {
    // Use volatile pointer for warp-synchronous behavior.
    volatile float* vsmem = sdata;
    // Unroll the last warp using __shfl_down_sync.
    // First, add using shared memory so that all threads in the warp have updated value.
    for (unsigned int offset = 16; offset > 0; offset /= 2) {
      float shfl_val = __shfl_down_sync(0xffffffff, vsmem[lid], offset);
      vsmem[lid] += shfl_val;
    }
  }

  // Write this block's partial sum to global memory.
  if (lid == 0) {
    result[bid] = sdata[0];
  }
}

// Kernel to perform the final reduction over the per–block partial sums.
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

  // Perform in–shared-memory reduction until only a single warp remains.
  for (unsigned int s = block_size >> 1; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // Use warp–level reduction in the final warp.
  if (tid < 32) {
    volatile float* vsmem = sdata;
    for (unsigned int offset = 16; offset > 0; offset /= 2) {
      float shfl_val = __shfl_down_sync(0xffffffff, vsmem[tid], offset);
      vsmem[tid] += shfl_val;
    }
  }
  
  // Write the final result back to global memory.
  if (tid == 0) {
    result[0] = sdata[0];
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
