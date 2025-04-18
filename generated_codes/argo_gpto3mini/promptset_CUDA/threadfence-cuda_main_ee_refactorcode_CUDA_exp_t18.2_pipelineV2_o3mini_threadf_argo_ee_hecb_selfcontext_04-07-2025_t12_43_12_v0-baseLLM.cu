
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

// First kernel: Each block computes a partial sum using shared memory reduction.
__global__ void reduceKernel(const float* __restrict__ array, int N, float* __restrict__ partialSums)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    // Using a grid–stride loop loading two elements per thread when possible.
    unsigned int index = 2 * blockIdx.x * blockDim.x + tid;
    float sum = 0;  

    if (index < N)
        sum = array[index];
    if (index + blockDim.x < N)
        sum += array[index + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Do reduction in shared memory. (Assumes blockDim.x is a power of 2)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write the block’s result into the partialSums array.
    if (tid == 0)
        partialSums[blockIdx.x] = sdata[0];
}

// Second kernel: Further reduce the partial sums. (One block is enough here.)
__global__ void finalReduceKernel(float* __restrict__ partialSums, int numPartial)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    // Load values from global memory into shared memory.
    float sum = (tid < numPartial) ? partialSums[tid] : 0;
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partialSums[0] = sdata[0];
}

int main(int argc, char** argv) {

    if (argc != 3) {
        printf("Usage: %s <repeat> <array length>\n", argv[0]);
        return 1;
    }

    const int repeat = atoi(argv[1]);
    const int N = atoi(argv[2]);

    // Choose a block size that is a power-of-two (e.g., 256).
    const int threads = 256;
    // First kernel will load 2 elements per thread.
    const int blocks = (N + threads * 2 - 1) / (threads * 2);

    // Allocate host memory.
    float* h_array = (float*) malloc(N * sizeof(float));
    float h_sum = 0.0f;

    // Allocate device memory.
    float* d_array = nullptr;
    cudaMalloc((void**)&d_array, N * sizeof(float));

    // For partial sums we need an array of size "blocks"
    float* d_partialSums = nullptr;
    cudaMalloc((void**)&d_partialSums, blocks * sizeof(float));

    bool ok = true;
    double totalTime = 0.0;

    // Initialize the array.
    for (int i = 0; i < N; i++) 
        h_array[i] = -1.f;

    for (int n = 0; n < repeat; n++) {

        cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        auto start = std::chrono::steady_clock::now();

        // First reduction kernel launch.
        // Use shared memory equal to "threads * sizeof(float)".
        reduceKernel<<<blocks, threads, threads * sizeof(float)>>>(d_array, N, d_partialSums);

        // If more than one block was used, launch second kernel.
        if (blocks > 1) {
            // We use one block with threads at least equal to 'blocks'.
            int finalThreads = 1;
            while (finalThreads < blocks) finalThreads <<= 1;
            finalReduceKernel<<<1, finalThreads, finalThreads * sizeof(float)>>>(d_partialSums, blocks);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // Copy the final result from device to host.
        cudaMemcpy(&h_sum, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost);

        // Validate the result.
        if (h_sum != -1.f * N) {
            ok = false;
            break;
        }
    }

    if (ok)
        printf("Average kernel execution time: %f (ms)\n", (totalTime * 1e-6f) / repeat);

    free(h_array);
    cudaFree(d_array);
    cudaFree(d_partialSums);

    printf("%s\n", ok ? "PASS" : "FAIL");
    return 0;
}
