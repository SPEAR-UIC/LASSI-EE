
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#define MAX_M 1024

#define CUDA_CALL(call)                                                   \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(err));                                   \
      exit(err);                                                          \
    }                                                                     \
  } while (0)

// constant memory for envelope arrays
__constant__ float const_lower_bound[MAX_M];
__constant__ float const_upper_bound[MAX_M];

// calculate LB_Keogh
__global__
void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
              float *__restrict__ lb_keogh,
              const float *__restrict__ lower, // no longer used
              const float *__restrict__ upper, // no longer used
              const int M,
              const int N) 
{
  // shared memory
  extern __shared__ float cache[];

  int lid = threadIdx.x;
  int blockOffset = blockDim.x * blockIdx.x;
  int idx = blockOffset + lid;

  for (int k = lid; k < blockDim.x + M; k += blockDim.x)
    if (blockOffset + k < N) cache[k] = subject[blockOffset + k];

  __syncthreads();

  if (idx < N - M + 1) {
    float residues = 0.0f;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      float value = (cache[lid + i] - avg) / std;
      float lower_val = value - const_lower_bound[i];
      float upper_val = value - const_upper_bound[i];

      residues += (upper_val > 0 ? upper_val * upper_val : 0.0f) +
                  (lower_val < 0 ? lower_val * lower_val : 0.0f);
    }

    lb_keogh[idx] = residues;
  }
}

// Reference CPU implementation for verification.
// This function reproduces the same logic as the kernel.
void reference(const float *subject,
               const float *avgs,
               const float *stds, 
               float *lb_keogh,
               const float *lower,
               const float *upper,
               const int M,
               const int N) 
{
  for (int idx = 0; idx < N - M + 1; ++idx) {
    float residues = 0.0f;
    float avg = avgs[idx];
    float std = stds[idx];
    for (int i = 0; i < M; ++i) {
      float value = (subject[idx + i] - avg) / std;
      float lower_val = value - lower[i];
      float upper_val = value - upper[i];
      residues += (upper_val > 0 ? upper_val * upper_val : 0.0f) +
                  (lower_val < 0 ? lower_val * lower_val : 0.0f);
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

  if (M > MAX_M) {
    printf("Error: Query length (M=%d) exceeds MAX_M (%d)\n", M, MAX_M);
    return 1;
  }

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  // Allocate pinned host memory for improved transfer performance (energy efficiency)
  float *subject, *lower, *upper, *lb, *lb_h, *avgs, *stds;
  CUDA_CALL(cudaMallocHost((void**)&subject, sizeof(float) * N));
  CUDA_CALL(cudaMallocHost((void**)&lower, sizeof(float) * M));
  CUDA_CALL(cudaMallocHost((void**)&upper, sizeof(float) * M));
  CUDA_CALL(cudaMallocHost((void**)&lb, sizeof(float) * (N - M + 1)));
  CUDA_CALL(cudaMallocHost((void**)&lb_h, sizeof(float) * (N - M + 1)));
  CUDA_CALL(cudaMallocHost((void**)&avgs, sizeof(float) * (N - M + 1)));
  CUDA_CALL(cudaMallocHost((void**)&stds, sizeof(float) * (N - M + 1)));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N - M + 1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  float *d_subject = NULL, *d_avgs = NULL, *d_stds = NULL, *d_lb = NULL;
  // Device allocation for only necessary arrays
  CUDA_CALL(cudaMalloc((void**)&d_subject, sizeof(float) * N));
  CUDA_CALL(cudaMalloc((void**)&d_avgs, sizeof(float) * (N - M + 1)));
  CUDA_CALL(cudaMalloc((void**)&d_stds, sizeof(float) * (N - M + 1)));
  CUDA_CALL(cudaMalloc((void**)&d_lb, sizeof(float) * (N - M + 1)));

  CUDA_CALL(cudaMemcpy(d_subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice));

  // copy envelope arrays to constant memory
  CUDA_CALL(cudaMemcpyToSymbol(const_lower_bound, lower, sizeof(float) * M));
  CUDA_CALL(cudaMemcpyToSymbol(const_upper_bound, upper, sizeof(float) * M));

  const int threadsPerBlock = 256;
  const int grids = (N - M + 1 + threadsPerBlock - 1) / threadsPerBlock;
  int smem_size = (threadsPerBlock + M) * sizeof(float);

  CUDA_CALL(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    lb_keogh<<<grids, threadsPerBlock, smem_size>>>
      (d_subject, d_avgs, d_stds, d_lb, NULL, NULL, M, N);
  }

  CUDA_CALL(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  CUDA_CALL(cudaMemcpy(lb, d_lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost));

  // verify
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

  CUDA_CALL(cudaFree(d_lb));
  CUDA_CALL(cudaFree(d_avgs));
  CUDA_CALL(cudaFree(d_stds));
  CUDA_CALL(cudaFree(d_subject));

  CUDA_CALL(cudaFreeHost(lb));
  CUDA_CALL(cudaFreeHost(lb_h));
  CUDA_CALL(cudaFreeHost(avgs));
  CUDA_CALL(cudaFreeHost(stds));
  CUDA_CALL(cudaFreeHost(subject));
  CUDA_CALL(cudaFreeHost(lower));
  CUDA_CALL(cudaFreeHost(upper));

  return 0;
}
