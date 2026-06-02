// Copyright (c) 2021, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

// A multiple of thread block size
#define N 16384

#define IDX(i, j) ((i) + (j) * N)

void initialize_data(float* f)
{
  // Set up simple sinusoidal boundary conditions
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {

      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }

    }
  }
}

__global__ __launch_bounds__(256) void jacobi_step(float* __restrict__ f,
                                                   const float* __restrict__ f_old,
                                                   float* __restrict__ partial_error)
{
  __shared__ float f_old_tile[18][18];
  __shared__ float reduction_array[16];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // First read in the "interior" data, one value per thread
  // Note the offset by 1, to reserve space for the "left"/"bottom" halo

  f_old_tile[threadIdx.y+1][threadIdx.x+1] = __ldg(&f_old[IDX(i,j)]);

  // Now read in the halo data; we'll pick the "closest" thread
  // to each element. When we do this, make sure we don't fall
  // off the end of the global memory array. Note that this
  // code does not fill the corners, as they are not used in
  // this stencil.

  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y+1][threadIdx.x+0] = __ldg(&f_old[IDX(i-1,j)]);
  }
  if (threadIdx.x == 15 && i <= N-2) {
    f_old_tile[threadIdx.y+1][threadIdx.x+2] = __ldg(&f_old[IDX(i+1,j)]);
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[threadIdx.y+0][threadIdx.x+1] = __ldg(&f_old[IDX(i,j-1)]);
  }
  if (threadIdx.y == 15 && j <= N-2) {
    f_old_tile[threadIdx.y+2][threadIdx.x+1] = __ldg(&f_old[IDX(i,j+1)]);
  }

  // Synchronize all threads
  __syncthreads();

  float err = 0.0f;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      // Perform the read from shared memory
      float new_value = 0.25f * (f_old_tile[threadIdx.y+1][threadIdx.x+2] +
                                 f_old_tile[threadIdx.y+1][threadIdx.x+0] +
                                 f_old_tile[threadIdx.y+2][threadIdx.x+1] +
                                 f_old_tile[threadIdx.y+0][threadIdx.x+1]);
      f[IDX(i,j)] = new_value;
      float df = new_value - f_old_tile[threadIdx.y+1][threadIdx.x+1];
      err = df * df;
    }
  }

  // Sum over threads in the warp
  // For simplicity, we do this outside the above conditional
  // so that all threads participate
  for (int offset = 8; offset > 0; offset /= 2) {
    err += __shfl_down_sync(0xffffffff, err, offset);
  }

  // If we're thread 0 in the warp, update our value to shared memory
  // Note that we're assuming exactly a 16x16 block and that the warp ID
  // is equivalent to threadIdx.y. For the general case, we would have to
  // write more careful code.
  if (threadIdx.x == 0) {
    reduction_array[threadIdx.y] = err;
  }

  // Synchronize the block before reading any values from smem
  __syncthreads();

  // Using the first warp in the block, reduce over the partial sums
  // in the shared memory array.
  if (threadIdx.y == 0) {
    err = reduction_array[threadIdx.x];
    for (int offset = 8; offset > 0; offset /= 2) {
      err += __shfl_down_sync(0x0000ffff, err, offset);
    }
    if (threadIdx.x == 0) {
      partial_error[blockIdx.x + blockIdx.y * gridDim.x] = err;
    }
  }
}

__global__ __launch_bounds__(256) void reduce_partials_256(const float* __restrict__ input,
                                                           float* __restrict__ output,
                                                           int n)
{
  __shared__ float smem[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  float value = 0.0f;
  if (idx < n) {
    value = __ldg(&input[idx]);
  }

  smem[tid] = value;
  __syncthreads();

  if (tid < 128) {
    smem[tid] += smem[tid + 128];
  }
  __syncthreads();

  if (tid < 64) {
    smem[tid] += smem[tid + 64];
  }
  __syncthreads();

  float sum = smem[tid];

  if (tid < 32) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (tid == 0) {
      output[blockIdx.x] = sum;
    }
  }
}

__global__ __launch_bounds__(256) void reduce_final_256(const float* __restrict__ input,
                                                        float* __restrict__ output,
                                                        int n)
{
  __shared__ float smem[256];

  int tid = threadIdx.x;
  float sum = 0.0f;

  for (int idx = tid; idx < n; idx += blockDim.x) {
    sum += __ldg(&input[idx]);
  }

  smem[tid] = sum;
  __syncthreads();

  if (tid < 128) {
    smem[tid] += smem[tid + 128];
  }
  __syncthreads();

  if (tid < 64) {
    smem[tid] += smem[tid + 64];
  }
  __syncthreads();

  sum = smem[tid];

  if (tid < 32) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (tid == 0) {
      output[0] = sum;
    }
  }
}

__global__ __launch_bounds__(256) void swap_data(const float* __restrict__ f,
                                                 float* __restrict__ f_old)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      f_old[IDX(i,j)] = f[IDX(i,j)];
    }
  }
}

int main()
{
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;
  float* d_partial_error;
  float* d_reduced_error;
  float* h_error;

  const size_t field_bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

  // Reserve space for the scalar field
  float* f = (float*) aligned_alloc(64, field_bytes);

  // Initialize data
  initialize_data(f);

  cudaMalloc((void**)&d_f, field_bytes);
  cudaMemcpy(d_f, f, field_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_f_old, field_bytes);
  cudaMemcpy(d_f_old, d_f, field_bytes, cudaMemcpyDeviceToDevice);

  cudaMalloc((void**)&d_error, sizeof(float));

  dim3 grid(N/16, N/16);
  dim3 block(16, 16);

  const int partial_count = grid.x * grid.y;
  const int reduced_count = (partial_count + 255) / 256;

  cudaMalloc((void**)&d_partial_error, static_cast<size_t>(partial_count) * sizeof(float));
  cudaMalloc((void**)&d_reduced_error, static_cast<size_t>(reduced_count) * sizeof(float));
  cudaMallocHost((void**)&h_error, sizeof(float));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  jacobi_step<<<grid, block, 0, stream>>>(d_f, d_f_old, d_partial_error);
  reduce_partials_256<<<reduced_count, 256, 0, stream>>>(d_partial_error, d_reduced_error, partial_count);
  reduce_final_256<<<1, 256, 0, stream>>>(d_reduced_error, d_error, reduced_count);
  cudaMemcpyAsync(h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, stream);
  swap_data<<<grid, block, 0, stream>>>(d_f, d_f_old);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

  // Initialize error to a large number
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  int num_iters = 0;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Perform a Jacobi relaxation step and hierarchical error reduction.
    cudaGraphLaunch(graph_exec, stream);

    cudaStreamSynchronize(stream);

    error = *h_error;

    // Normalize the L2-norm of the error by the number of data points
    // and then take the square root
    error = sqrtf(error / (N * N));

    // Periodically print out the current error
    if (num_iters % 1000 == 0) {
      std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
    }

    // Increment the iteration count
    ++num_iters;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  // If we took fewer than max_iters steps and the error is below the tolerance,
  // we succeeded. Otherwise, we failed.

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  // CLean up memory allocations
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  cudaFree(d_partial_error);
  cudaFree(d_reduced_error);
  cudaFreeHost(h_error);
  free(f);

  // End wall timing
  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
