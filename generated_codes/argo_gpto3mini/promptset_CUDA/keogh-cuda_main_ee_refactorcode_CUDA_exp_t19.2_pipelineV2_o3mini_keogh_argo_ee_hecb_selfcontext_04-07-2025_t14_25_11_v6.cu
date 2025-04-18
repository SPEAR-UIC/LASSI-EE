
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// Begin replacement for reference.h to fix missing file error
#ifndef REFERENCE_H
#define REFERENCE_H

inline void reference(const float *__restrict__ subject,
                      const float *__restrict__ avgs,
                      const float *__restrict__ stds, 
                            float *__restrict__ lb_keogh,
                      const float *__restrict__ lower_bound,
                      const float *__restrict__ upper_bound,
                      const int M,
                      const int N)
{
  for (int idx = 0; idx < N - M + 1; ++idx) {
    float residues = 0.0f;
    float avg = avgs[idx];
    float std = stds[idx];
    float recStd = 1.0f / std;
    for (int i = 0; i < M; ++i) {
      float value = (subject[idx + i] - avg) * recStd;
      float lower = value - lower_bound[i];
      float upper = value - upper_bound[i];
      residues += (upper > 0 ? upper * upper : 0.0f) + (lower < 0 ? lower * lower : 0.0f);
    }
    lb_keogh[idx] = residues;
  }
}

#endif // REFERENCE_H
// End replacement for reference.h

// calculate LB_Keogh; added repeat parameter to batch iterations in a single kernel launch
__global__
void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
                    float *__restrict__ lb_keogh,
              const float *__restrict__ lower_bound,
              const float *__restrict__ upper_bound,
              const int M,
              const int N,
              const int repeat) 
{
  // shared memory allocation
  extern __shared__ float cache[];

  int lid = threadIdx.x;
  int blockOffset = blockDim.x * blockIdx.x;
  int idx = blockOffset + lid;

  // Optimized shared memory loading:
  // Each thread loads one element from the main block
  if (blockOffset + lid < N) {
    cache[lid] = subject[blockOffset + lid];
  }
  // Threads with lid < M load the extra elements for the halo region
  if (lid < M && (blockOffset + blockDim.x + lid) < N) {
    cache[blockDim.x + lid] = subject[blockOffset + blockDim.x + lid];
  }
  __syncthreads();

  // Only threads with valid indices perform computation
  if (idx < N - M + 1) {
    float avg = avgs[idx];
    float std = stds[idx];
    float recStd = 1.0f / std;
    float result = 0.0f;
    // Batch the computation repeat times; only the final iteration's result is stored.
    for (int rep = 0; rep < repeat; rep++) {
      float residues = 0.0f;
      for (int i = 0; i < M; ++i) {
        // Use shared memory offset directly: note that subject element is at cache[lid + i]
        float value = (cache[lid + i] - avg) * recStd;
        float lowerVal = value - lower_bound[i];
        float upperVal = value - upper_bound[i];
        residues += (upperVal > 0 ? upperVal * upperVal : 0.0f)
                  + (lowerVal < 0 ? lowerVal * lowerVal : 0.0f);
      }
      result = residues;
    }
    lb_keogh[idx] = result;
  }
}

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  // Host side memory allocation using pinned (page-locked) memory
  float *subject = NULL;
  float *lower = NULL;
  float *upper = NULL;
  float *lb = NULL;
  float *lb_h = NULL;
  float *avgs = NULL;
  float *stds = NULL;

  cudaHostAlloc((void**)&subject, sizeof(float) * N, cudaHostAllocDefault);
  cudaHostAlloc((void**)&lower, sizeof(float) * M, cudaHostAllocDefault);
  cudaHostAlloc((void**)&upper, sizeof(float) * M, cudaHostAllocDefault);
  cudaHostAlloc((void**)&lb, sizeof(float) * (N - M + 1), cudaHostAllocDefault);
  cudaHostAlloc((void**)&lb_h, sizeof(float) * (N - M + 1), cudaHostAllocDefault);
  cudaHostAlloc((void**)&avgs, sizeof(float) * (N - M + 1), cudaHostAllocDefault);
  cudaHostAlloc((void**)&stds, sizeof(float) * (N - M + 1), cudaHostAllocDefault);

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  float *d_subject = NULL, *d_avgs = NULL, *d_stds = NULL, 
        *d_lb = NULL, *d_lower = NULL, *d_upper = NULL;

  cudaMalloc(&d_subject, sizeof(float) * N);
  cudaMalloc(&d_avgs, sizeof(float) * (N - M + 1));
  cudaMalloc(&d_stds, sizeof(float) * (N - M + 1));
  cudaMalloc(&d_lb, sizeof(float) * (N - M + 1));
  cudaMalloc(&d_lower, sizeof(float) * M);
  cudaMalloc(&d_upper, sizeof(float) * M);

  // Create a stream for asynchronous memory transfers and kernel launch
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Asynchronously copy host data to device using the created stream
  cudaMemcpyAsync(d_subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_lower, lower, sizeof(float) * M, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_upper, upper, sizeof(float) * M, cudaMemcpyHostToDevice, stream);

  // No extra synchronization is needed because kernel launch in the same stream
  // will wait for these async operations to complete.

  const int blocks = 256;
  const int grids = (N - M + 1 + blocks - 1) / blocks;
  // Updated shared memory size: we need blockDim.x for the base plus M for the halo
  int smem_size = (blocks + M) * sizeof(float);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch a single kernel which internally repeats the computation "repeat" times using the same stream.
  lb_keogh<<<grids, blocks, smem_size, stream>>>
      (d_subject, d_avgs, d_stds, d_lb, d_lower, d_upper, M, N, repeat);

  // Asynchronously copy the results back to the host
  cudaMemcpyAsync(lb, d_lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost, stream);

  // Wait for all operations in the stream to complete before timing ends
  cudaStreamSynchronize(stream);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  // verify results using the reference function
  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N - M + 1; i++) {
    if (fabsf(lb[i] - lb_h[i]) > 1e-2f) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_lb);
  cudaFree(d_avgs);
  cudaFree(d_stds);
  cudaFree(d_subject);
  cudaFree(d_lower);
  cudaFree(d_upper);

  cudaFreeHost(lb);
  cudaFreeHost(lb_h);
  cudaFreeHost(avgs);
  cudaFreeHost(stds);
  cudaFreeHost(subject);
  cudaFreeHost(lower);
  cudaFreeHost(upper);

  cudaStreamDestroy(stream);
  return 0;
}
