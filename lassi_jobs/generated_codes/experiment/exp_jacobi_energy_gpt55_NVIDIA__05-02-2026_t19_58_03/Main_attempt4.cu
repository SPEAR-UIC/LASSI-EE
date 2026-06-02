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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// A multiple of thread block size
#define N 16384

#define IDX(i, j) ((i) + (j) * N)

static void check_cuda(cudaError_t result, const char* file, int line)
{
  if (result != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << ": "
              << cudaGetErrorString(result) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

void initialize_data(float* f)
{
  // Set up simple sinusoidal boundary conditions
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {

      if (i == 0 || i == N - 1) {
        f[IDX(i, j)] = sinf(j * 2 * M_PI / (N - 1));
      }
      else if (j == 0 || j == N - 1) {
        f[IDX(i, j)] = sinf(i * 2 * M_PI / (N - 1));
      }
      else {
        f[IDX(i, j)] = 0.0f;
      }

    }
  }
}

__global__ void jacobi_step(float* __restrict__ f,
                            const float* __restrict__ f_old,
                            float* __restrict__ error)
{
  __shared__ float f_old_tile[18][18];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // First read in the "interior" data, one value per thread
  // Note the offset by 1, to reserve space for the "left"/"bottom" halo

  f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = f_old[IDX(i, j)];

  // Now read in the halo data; we'll pick the "closest" thread
  // to each element. When we do this, make sure we don't fall
  // off the end of the global memory array. Note that this
  // code does not fill the corners, as they are not used in
  // this stencil.

  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 0] = f_old[IDX(i - 1, j)];
  }
  if (threadIdx.x == 15 && i <= N - 2) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 2] = f_old[IDX(i + 1, j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[threadIdx.y + 0][threadIdx.x + 1] = f_old[IDX(i, j - 1)];
  }
  if (threadIdx.y == 15 && j <= N - 2) {
    f_old_tile[threadIdx.y + 2][threadIdx.x + 1] = f_old[IDX(i, j + 1)];
  }

  // Synchronize all threads
  __syncthreads();

  float err = 0.0f;

  if (j >= 1 && j <= N - 2) {
    if (i >= 1 && i <= N - 2) {
      // Perform the read from shared memory
      f[IDX(i, j)] = 0.25f * (f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                              f_old_tile[threadIdx.y + 1][threadIdx.x + 0] +
                              f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                              f_old_tile[threadIdx.y + 0][threadIdx.x + 1]);
      float df = f[IDX(i, j)] - f_old_tile[threadIdx.y + 1][threadIdx.x + 1];
      err = df * df;
    }
  }

  // Sum over threads in each 16-thread row segment.
  // The block is exactly 16x16, so each warp contains two independent rows.
  for (int offset = 8; offset > 0; offset /= 2) {
    err += __shfl_down_sync(0xffffffff, err, offset, 16);
  }

  // If we're thread 0 in the row, update our value to shared memory.
  __shared__ float reduction_array[16];
  if (threadIdx.x == 0) {
    reduction_array[threadIdx.y] = err;
  }

  // Synchronize the block before reading any values from smem
  __syncthreads();

  // Using the first row in the block, reduce over the partial sums
  // in the shared memory array.
  if (threadIdx.y == 0) {
    err = reduction_array[threadIdx.x];
    for (int offset = 8; offset > 0; offset /= 2) {
      err += __shfl_down_sync(0x0000ffff, err, offset, 16);
    }
    if (threadIdx.x == 0) {
      atomicAdd(error, err);
    }
  }
}

__global__ void swap_data(const float* __restrict__ f,
                          float* __restrict__ f_old)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= 1 && j <= N - 2) {
    if (i >= 1 && i <= N - 2) {
      f_old[IDX(i, j)] = f[IDX(i, j)];
    }
  }
}

double compute_checksum(const float* d_data, float* h_buffer)
{
  const size_t count = static_cast<size_t>(N) * static_cast<size_t>(N);

  CHECK_CUDA(cudaMemcpy(h_buffer,
                        d_data,
                        count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  for (size_t idx = 0; idx < count; ++idx) {
    double value = static_cast<double>(h_buffer[idx]);
    checksum += value * value;
  }

  return checksum;
}

int main()
{
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;

  const size_t total_count = static_cast<size_t>(N) * static_cast<size_t>(N);
  const size_t total_bytes = total_count * sizeof(float);

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = static_cast<float*>(std::aligned_alloc(64, total_bytes));
  float* f_old = static_cast<float*>(std::aligned_alloc(64, total_bytes));

  if (f == nullptr || f_old == nullptr) {
    std::cerr << "Host allocation failed" << std::endl;
    std::free(f);
    std::free(f_old);
    return -1;
  }

  // Initialize data (we'll do this on both f and f_old, so that we don't
  // have to worry about the boundary points later)
  initialize_data(f);
  initialize_data(f_old);

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_f), total_bytes));
  CHECK_CUDA(cudaMemcpy(d_f,
                        f,
                        total_bytes,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_f_old), total_bytes));
  CHECK_CUDA(cudaMemcpy(d_f_old,
                        f_old,
                        total_bytes,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_error), sizeof(float)));

  // Initialize error to a large number
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  int num_iters = 0;

  dim3 grid(N / 16, N / 16);
  dim3 block(16, 16);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Initialize error to zero (we'll add to it the following step)
    CHECK_CUDA(cudaMemset(d_error, 0, 4));

    // Perform a Jacobi relaxation step
    jacobi_step<<<grid, block>>>(d_f, d_f_old, d_error);
    CHECK_CUDA(cudaGetLastError());

    // Swap the old data and the new data
    // We're doing this explicitly for pedagogical purposes, even though
    // in this specific application a std::swap would have been OK
    swap_data<<<grid, block>>>(d_f, d_f_old);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost));

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

  CHECK_CUDA(cudaDeviceSynchronize());
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
    CHECK_CUDA(cudaFree(d_f));
    CHECK_CUDA(cudaFree(d_f_old));
    CHECK_CUDA(cudaFree(d_error));
    std::free(f);
    std::free(f_old);
    return -1;
  }

  double checksum = compute_checksum(d_f_old, f_old);
  std::cout << "Verification checksum: " << std::setprecision(17) << checksum << std::endl;

  // CLean up memory allocations
  CHECK_CUDA(cudaFree(d_f));
  CHECK_CUDA(cudaFree(d_f_old));
  CHECK_CUDA(cudaFree(d_error));
  std::free(f);
  std::free(f_old);

  // End wall timing
  double duration = (std::clock() - start_time) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
