
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// First kernel: each block reduces a portion of the input array
__global__ void blockReduce(const float* __restrict__ array, int N, float* __restrict__ partialSums)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Strided access: each thread accumulates multiple elements
    for (int i = gid; i < N; i += gridDim.x * blockDim.x)
    {
        sum += array[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // In-block reduction using a binary tree pattern
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write each block's result to the partialSums array.
    if (tid == 0)
    {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// Second kernel: reduce the partial sums computed by blockReduce
__global__ void finalReduce(float* __restrict__ partialSums, int numElements)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;

    // Load the partial sums into shared memory; threads with tid >= numElements load 0
    if (tid < numElements)
    {
        sum = partialSums[tid];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result back to partialSums[0]
    if (tid == 0)
    {
        partialSums[0] = sdata[0];
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <repeat> <array length>\n", argv[0]);
        return 1;
    }

    const int repeat = atoi(argv[1]);
    const int N = atoi(argv[2]);

    // Use a block size that is a power of two for efficient reduction.
    const int blockSize = 256;
    // Calculate number of blocks needed for the first kernel;
    // ensuring all elements are covered.
    int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory and initialize the array elements.
    float* h_array = (float*) malloc(N * sizeof(float));
    float h_sum = 0.0f;

    // Allocate memory on the device:
    // d_array for input data, d_partialSums will hold per-block sums and result.
    float* d_array;
    cudaMalloc((void**)&d_array, N * sizeof(float));

    float* d_partialSums;
    // Allocate enough space for gridSize partial results; the final answer will be in index 0.
    cudaMalloc((void**)&d_partialSums, gridSize * sizeof(float));

    // Initialize host data.
    for (int i = 0; i < N; i++) {
        h_array[i] = -1.0f;
    }

    bool ok = true;
    double totalTimeNs = 0.0;

    for (int n = 0; n < repeat; n++) {
        // Copy data from host to device.
        cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        auto start = std::chrono::steady_clock::now();

        // Kernel 1: Each block reduces a subset of the array.
        // Dynamic shared memory size is blockSize * sizeof(float)
        blockReduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_array, N, d_partialSums);
        cudaDeviceSynchronize();

        // Kernel 2: If more than one block was used, perform a final reduction.
        if (gridSize > 1) {
            // Use a single block for the final reduction.
            // Pick a block size which is at least the next power-of-two of gridSize.
            int finalBlockSize = 1;
            while (finalBlockSize < gridSize) {
                finalBlockSize <<= 1;
            }
            finalReduce<<<1, finalBlockSize, finalBlockSize * sizeof(float)>>>(d_partialSums, gridSize);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::steady_clock::now();
        totalTimeNs += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // Retrieve the final sum. Regardless of which kernel did the final reduction,
        // the complete sum is stored in d_partialSums[0].
        cudaMemcpy(&h_sum, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost);

        // Validate the result.
        if (h_sum != -1.0f * N) {
            ok = false;
            break;
        }
    }

    if (ok)
        printf("Average kernel execution time: %f (ms)\n", (totalTimeNs * 1e-6f) / repeat);
    else
        printf("Result error: expected %f, got %f\n", -1.0f * N, h_sum);

    // Free the allocated memory.
    free(h_array);
    cudaFree(d_array);
    cudaFree(d_partialSums);

    printf("%s\n", ok ? "PASS" : "FAIL");
    return 0;
}
