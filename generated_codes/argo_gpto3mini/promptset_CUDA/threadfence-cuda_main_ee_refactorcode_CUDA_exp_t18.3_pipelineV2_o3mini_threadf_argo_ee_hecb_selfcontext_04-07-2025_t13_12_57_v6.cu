
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// Phase 1: Compute block-level partial sums.
__global__ void sum(
    const float* __restrict__ array,
    const int N,
    unsigned int* __restrict__ count,  // kept for signature compatibility (unused)
    volatile float* __restrict__ result)
{
    extern __shared__ float sdata[];

    float mySum = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += gridDim.x * blockDim.x)
    {
        mySum += array[idx];
    }
    sdata[threadIdx.x] = mySum;
    __syncthreads();

    // Intra-block reduction: reduce within the block using shared memory.
    for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sdata[threadIdx.x] += __shfl_down_sync(0xffffffff, sdata[threadIdx.x], offset);
        }
    }
    // Write the block's partial sum to global memory.
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sdata[0];
        // The atomic counter is no longer needed for final reduction.
        // We still preserve d_count for compatibility.
    }
}

// Phase 2: Final reduction kernel reduces the partial sums computed by sum().
__global__ void finalReduce(
    volatile float* __restrict__ result,
    const int nPartial)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    // Load partial sums into shared memory. If nPartial < blockDim.x, pad with zeros.
    float val = (tid < nPartial) ? result[tid] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Perform tree reduction in shared memory.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write final result to result[0].
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

    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory.
    float* h_array = (float*) malloc(N * sizeof(float));
    float h_sum;

    // Allocate device memory.
    float* d_result;
    cudaMalloc((void**)&d_result, numBlocks * sizeof(float));

    float* d_array;
    cudaMalloc((void**)&d_array, N * sizeof(float));

    unsigned int* d_count;
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0u, sizeof(unsigned int));

    bool ok = true;
    double time = 0.0;

    // Initialize the host array with -1.f.
    for (int i = 0; i < N; i++) {
        h_array[i] = -1.f;
    }

    // Copy the constant input array from host to device only once.
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Determine launch parameters for final reduction.
    // We'll launch finalReduce with one block. Choose threads that is a power-of-two 
    // and at least as large as numBlocks.
    int finalThreads = 1;
    while (finalThreads < numBlocks) {
        finalThreads *= 2;
    }

    // Run the kernel "repeat" times.
    for (int n = 0; n < repeat; n++) {
        // Record the start event.
        cudaEventRecord(start, 0);

        // Launch the first-phase kernel with dynamic shared memory set to threadsPerBlock*sizeof(float).
        sum<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_array, N, d_count, d_result);

        // If more than one block produced partial sums, reduce them.
        if (numBlocks > 1) {
            finalReduce<<<1, finalThreads, finalThreads * sizeof(float)>>>(d_result, numBlocks);
        }
        
        // Record the stop event.
        cudaEventRecord(stop, 0);
        // Wait for the stop event to complete.
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        time += elapsed_ms;

        // Copy the final sum from device to host.
        cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        // Verify result.
        if (h_sum != -1.f * N) {
            ok = false;
            break;
        }
    }

    // Destroy CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (ok)
        printf("Average kernel execution time: %f (ms)\n", time / repeat);

    // Clean up.
    free(h_array);
    cudaFree(d_result);
    cudaFree(d_array);
    cudaFree(d_count);

    printf("%s\n", ok ? "PASS" : "FAIL");
    return 0;
}
