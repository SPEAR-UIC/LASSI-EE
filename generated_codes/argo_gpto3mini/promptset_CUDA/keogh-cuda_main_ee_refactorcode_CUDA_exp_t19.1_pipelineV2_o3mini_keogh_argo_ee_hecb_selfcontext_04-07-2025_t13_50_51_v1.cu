
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

// Begin inline reference.h implementation to fix missing header.
#ifndef REFERENCE_H
#define REFERENCE_H
void reference(const float *subject, const float *avgs, const float *stds, float *lb, const float *lower, const float *upper, int M, int N) {
    int nLen = N - M + 1;
    for (int idx = 0; idx < nLen; idx++) {
        float residues = 0.0f;
        float avg = avgs[idx];
        float std = stds[idx];
        for (int j = 0; j < M; j++) {
            float normval = (subject[idx + j] - avg) / std;
            float diff_lower = normval - lower[j];
            float diff_upper = normval - upper[j];
            if (diff_upper > 0.f)
                residues += diff_upper * diff_upper;
            if (diff_lower < 0.f)
                residues += diff_lower * diff_lower;
        }
        lb[idx] = residues;
    }
}
#endif
// End inline reference.h implementation.

#define MAX_QUERY_LENGTH 1024  // maximum allowed query length for constant memory

__constant__ float d_lower_const[MAX_QUERY_LENGTH];
__constant__ float d_upper_const[MAX_QUERY_LENGTH];

// Optimized kernel: fused iterations & use of constant memory for lower/upper arrays.
__global__
void lb_keogh_kernel(const float *__restrict__ subject,
                     const float *__restrict__ avgs,
                     const float *__restrict__ stds, 
                           float *__restrict__ lb,
                     const int repeat,
                     const int M,
                     const int N)
{
    extern __shared__ float s_subject[];
    int lid = threadIdx.x;
    int blockStartIdx = blockIdx.x * blockDim.x;
    int globalIdx = blockStartIdx + lid;

    // Cooperative loading of the subject array tile into shared memory.
    for (int i = lid; i < blockDim.x + M; i += blockDim.x) {
        int index = blockStartIdx + i;
        s_subject[i] = (index < N) ? subject[index] : 0.0f;
    }
    __syncthreads();

    // Process the kernel computation "repeat" times.
    for (int r = 0; r < repeat; r++) {
        if (globalIdx < N - M + 1) {
            float residues = 0.0f;
            float avg = avgs[globalIdx];
            float std = stds[globalIdx];
            // Compute lb_keogh measure.
            for (int i = 0; i < M; ++i) {
                float normval = (s_subject[lid + i] - avg) / std;
                float diff_lower = normval - d_lower_const[i];
                float diff_upper = normval - d_upper_const[i];
                residues += (diff_upper > 0.f) ? diff_upper * diff_upper : 0.f;
                residues += (diff_lower < 0.f) ? diff_lower * diff_lower : 0.f;
            }
            lb[globalIdx] = residues;
        }
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

    if(M > MAX_QUERY_LENGTH) {
        printf("Error: Query length (%d) exceeds maximum (%d) allowed for constant memory.\n", M, MAX_QUERY_LENGTH);
        return 1;
    }

    printf("Query length = %d\n", M);
    printf("Subject length = %d\n", N);

    // Allocate pinned host memory using cudaHostAlloc.
    float *subject = NULL, *lower = NULL, *upper = NULL, *lb = NULL, *lb_h = NULL, *avgs = NULL, *stds = NULL;
    cudaHostAlloc((void**)&subject, sizeof(float) * N, cudaHostAllocDefault);
    cudaHostAlloc((void**)&lower,   sizeof(float) * M, cudaHostAllocDefault);
    cudaHostAlloc((void**)&upper,   sizeof(float) * M, cudaHostAllocDefault);
    cudaHostAlloc((void**)&lb,      sizeof(float) * (N - M + 1), cudaHostAllocDefault);
    cudaHostAlloc((void**)&lb_h,    sizeof(float) * (N - M + 1), cudaHostAllocDefault);
    cudaHostAlloc((void**)&avgs,    sizeof(float) * (N - M + 1), cudaHostAllocDefault);
    cudaHostAlloc((void**)&stds,    sizeof(float) * (N - M + 1), cudaHostAllocDefault);

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

    // Allocate device memory.
    float *d_subject = NULL, *d_avgs = NULL, *d_stds = NULL, *d_lb = NULL;
    cudaMalloc(&d_subject, sizeof(float) * N);
    cudaMalloc(&d_avgs, sizeof(float) * (N - M + 1));
    cudaMalloc(&d_stds, sizeof(float) * (N - M + 1));
    cudaMalloc(&d_lb, sizeof(float) * (N - M + 1));

    // Copy data to the device.
    cudaMemcpy(d_subject, subject, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_avgs, avgs, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stds, stds, sizeof(float) * (N - M + 1), cudaMemcpyHostToDevice);
    // Copy the query bounds into constant memory.
    cudaMemcpyToSymbol(d_lower_const, lower, sizeof(float) * M, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_upper_const, upper, sizeof(float) * M, 0, cudaMemcpyHostToDevice);

    // Choose a suitable block size.
    const int threadsPerBlock = 256;
    const int numElements = N - M + 1;
    const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // The shared memory needed is (threadsPerBlock + M) floats.
    int smem_size = (threadsPerBlock + M) * sizeof(float);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch the optimized kernel.
    lb_keogh_kernel<<<numBlocks, threadsPerBlock, smem_size>>>
        (d_subject, d_avgs, d_stds, d_lb, repeat, M, N);
   
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    cudaMemcpy(lb, d_lb, sizeof(float) * (N - M + 1), cudaMemcpyDeviceToHost);

    // Verify results via the CPU reference implementation.
    reference(subject, avgs, stds, lb_h, lower, upper, M, N);
    bool ok = true;
    for (int i = 0; i < N - M + 1; i++) {
        if (fabsf(lb[i] - lb_h[i]) > 1e-2f) {
            printf("Mismatch at index %d: GPU result %f, reference %f\n", i, lb[i], lb_h[i]);
            ok = false;
            break;
        }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    // Free device memory.
    cudaFree(d_lb);
    cudaFree(d_avgs);
    cudaFree(d_stds);
    cudaFree(d_subject);

    // Free pinned host memory.
    cudaFreeHost(lb);
    cudaFreeHost(lb_h);
    cudaFreeHost(avgs);
    cudaFreeHost(stds);
    cudaFreeHost(subject);
    cudaFreeHost(lower);
    cudaFreeHost(upper);

    return 0;
}
