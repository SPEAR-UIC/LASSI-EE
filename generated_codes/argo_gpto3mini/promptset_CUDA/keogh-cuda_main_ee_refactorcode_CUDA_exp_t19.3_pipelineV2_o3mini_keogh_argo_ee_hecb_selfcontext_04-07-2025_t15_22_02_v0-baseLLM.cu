
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// Begin reference.h implementation to fix missing header error.
#ifndef REFERENCE_H
#define REFERENCE_H

// Reference CPU implementation for LB_Keogh calculation.
static inline void reference(const float *subject,
                             const float *avgs,
                             const float *stds,
                                   float *lb_keogh,
                             const float *lower,
                             const float *upper,
                             const int M,
                             const int N) 
{
  for (int idx = 0; idx < N - M + 1; idx++) {
    float residues = 0.0f;
    const float avg = avgs[idx];
    const float std = stds[idx];
    const float inv_std = (std != 0.0f) ? 1.0f / std : 0.0f;
    for (int i = 0; i < M; ++i) {
      float norm_val = (subject[idx + i] - avg) * inv_std;
      float lowerDiff = norm_val - lower[i];
      float upperDiff = norm_val - upper[i];
      if (upperDiff > 0)
        residues += upperDiff * upperDiff;
      if (lowerDiff < 0)
        residues += lowerDiff * lowerDiff;
    }
    lb_keogh[idx] = residues;
  }
}

#endif // REFERENCE_H
// End reference.h implementation

// We assume that maximum query length (M) does not exceed this constant.
#define MAX_QUERY_LENGTH 1024

// Place read‐only lower and upper bounds arrays into constant memory for faster access 
// and lower energy consumption. These arrays are only used for reading inside the kernel.
__constant__ float c_lower[MAX_QUERY_LENGTH];
__constant__ float c_upper[MAX_QUERY_LENGTH];

// calculate LB_Keogh
// Note: lower_bound and upper_bound have been replaced by the constant memory arrays.
__global__
void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
                    float *__restrict__ lb_keogh,
              const int M,
              const int N) 
{
  // shared memory is used to cache a tile of subject.
  extern __shared__ float cache[];

  const int lid = threadIdx.x;
  const int blockOffset = blockIdx.x * blockDim.x;
  const int idx = blockOffset + lid;

  // Each block loads blockDim.x + M elements into shared memory.
  // Note: this loop allows coalesced memory reads by all threads.
  for (int k = lid; k < blockDim.x + M; k += blockDim.x) {
    if (blockOffset + k < N)
      cache[k] = subject[blockOffset + k];
  }

  __syncthreads();

  // Each thread computes the LB_Keogh for one window starting at idx
  if (idx < N - M + 1) {
    float residues = 0.0f;
    const float avg = avgs[idx];
    const float std = stds[idx];
    // Precompute inverse for efficiency and to avoid extra division operations.
    const float inv_std = (std != 0.0f) ? 1.0f / std : 0.0f;

    // Compute the residue using the pre-loaded shared memory and constant envelope arrays.
    // Unrolling the division by precomputing inv_std reduces overall operations.
    for (int i = 0; i < M; ++i) {
      // Access the subject value from shared memory.
      float norm_val = (cache[lid + i] - avg) * inv_std;
      float lowerDiff = norm_val - c_lower[i];
      float upperDiff = norm_val - c_upper[i];

      // Use the condition (upperDiff > 0) and (lowerDiff < 0) as multipliers (0 or 1)
      residues += (upperDiff > 0) * (upperDiff * upperDiff) +
                  (lowerDiff < 0) * (lowerDiff * lowerDiff);
    }

    lb_keogh[idx] = residues;
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

  // Validate that M fits in our constant memory arrays.
  if (M > MAX_QUERY_LENGTH) {
    fprintf(stderr, "Error: query length exceeds maximum supported (%d)\n", MAX_QUERY_LENGTH);
    return 1;
  }

  // host side memory allocations
  float *subject = (float*) malloc(sizeof(float) * N);
  float *lower   = (float*) malloc(sizeof(float) * M);
  float *upper   = (float*) malloc(sizeof(float) * M);
  float *lb      = (float*) malloc(sizeof(float) * (N - M + 1));
  float *lb_h    = (float*) malloc(sizeof(float) * (N - M + 1));
  float *avgs    = (float*) malloc(sizeof(float) * (N - M + 1));
  float *stds    = (float*) malloc(sizeof(float) * (N - M + 1));

  // initialize host data
  srand(123);
  for (int i = 0; i < N; ++i) 
    subject[i] = (float)rand() / (float)RAND_MAX;
  
  for (int i = 0; i < N - M + 1; ++i) {
    avgs[i] = (float)rand() / (float)RAND_MAX;
    stds[i] = (float)rand() / (float)RAND_MAX;
  }
  
  for (int i = 0; i < M; ++i) {
    upper[i] = (float)rand() / (float)RAND_MAX;
    lower[i] = (float)rand() / (float)RAND_MAX;
  }

  // Allocate device memory for subject, avgs, stds, and lb arrays.
  float *d_subject = nullptr, *d_avgs = nullptr, *d_stds = nullptr, *d_lb = nullptr;
  cudaMalloc(&d_subject, sizeof(float) * N);
  cudaMalloc(&d_avgs, sizeof(float) * (N - M + 1));
  cudaMalloc(&d_stds, sizeof(float) * (N - M + 1));
  cudaMalloc(&d_lb, sizeof(float) * (N - M + 1));

  // Copy data from host to device.
  cudaMemcpy(d_subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice);

  // Copy constant envelope arrays to constant memory.
  cudaMemcpyToSymbol(c_lower, lower, sizeof(float) * M);
  cudaMemcpyToSymbol(c_upper, upper, sizeof(float) * M);

  // Define execution configuration.
  const int threads_per_block = 256;
  const int numBlocks = (N - M + 1 + threads_per_block - 1) / threads_per_block;
  // Shared memory tile size = blockDim.x + M elements.
  int smem_size = (threads_per_block + M) * sizeof(float);

  // Ensure device synchronization and start timing.
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch the kernel repeat times.
  for (int i = 0; i < repeat; i++) {
    lb_keogh<<<numBlocks, threads_per_block, smem_size>>>
      (d_subject, d_avgs, d_stds, d_lb, M, N);
  }
  
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  // Copy kernel result back to host.
  cudaMemcpy(lb, d_lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost);

  // Verify the results against the reference implementation.
  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N - M + 1; i++) {
    if (fabsf(lb[i] - lb_h[i]) > 1e-2f) {
      printf("Mismatch at %d: GPU = %f, Reference = %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Cleanup device and host memory.
  cudaFree(d_lb);
  cudaFree(d_avgs);
  cudaFree(d_stds);
  cudaFree(d_subject);
  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower);
  free(upper);

  return 0;
}
