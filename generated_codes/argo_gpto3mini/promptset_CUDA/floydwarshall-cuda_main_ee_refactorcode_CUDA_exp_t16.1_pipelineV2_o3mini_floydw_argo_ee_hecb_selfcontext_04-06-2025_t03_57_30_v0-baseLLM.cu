
/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
   (License text unchanged)
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

// Define a maximum random distance value.
#define MAXDISTANCE    (200)

// Macro for basic CUDA error checking.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (cudaSuccess != err) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/**
 * Returns the lesser of the two unsigned integers a and b.
 */
unsigned int minimum(unsigned int a, unsigned int b)
{
    return (b < a) ? b : a;
}

/**
 * CPU reference implementation of Floyd–Warshall algorithm.
 * This version computes the shortest path between every pair of nodes.
 */
void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
                               unsigned int * pathMatrix, unsigned int numNodes)
{
    unsigned int distanceYtoX, distanceYtoK, distanceKtoX, indirectDistance;
    unsigned int width = numNodes;
    unsigned int yXwidth;

    for (unsigned int k = 0; k < numNodes; ++k)
    {
        for (unsigned int y = 0; y < numNodes; ++y)
        {
            yXwidth = y * numNodes;
            for (unsigned int x = 0; x < numNodes; ++x)
            {
                distanceYtoX = pathDistanceMatrix[yXwidth + x];
                distanceYtoK = pathDistanceMatrix[yXwidth + k];
                distanceKtoX = pathDistanceMatrix[k * width + x];

                indirectDistance = distanceYtoK + distanceKtoX;
                if (indirectDistance < distanceYtoX)
                {
                    pathDistanceMatrix[yXwidth + x] = indirectDistance;
                    pathMatrix[yXwidth + x] = k;
                }
            }
        }
    }
}

/**
 * GPU kernel for one pass of the Floyd–Warshall algorithm.
 * Each kernel launch processes one intermediate node 'k' (given by 'pass').
 */
__global__ void floydWarshallPass(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes,
    const unsigned int pass)
{
    int xValue = threadIdx.x + blockIdx.x * blockDim.x;
    int yValue = threadIdx.y + blockIdx.y * blockDim.y;

    // Ensure that threads do not access beyond the matrix dimensions.
    if (xValue < numNodes && yValue < numNodes) {
        int k = pass;
        int index = yValue * numNodes + xValue;
        int oldWeight = pathDistanceBuffer[index];
        int tempWeight = pathDistanceBuffer[yValue * numNodes + k] +
                         pathDistanceBuffer[k * numNodes + xValue];

        if (tempWeight < oldWeight)
        {
            pathDistanceBuffer[index] = tempWeight;
            pathBuffer[index] = k;
        }
    }
}

int main(int argc, char** argv) {
    // Expect exactly three command-line arguments.
    if (argc != 4) {
        printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
        return 1;
    }
    unsigned int numNodes = (unsigned int) atoi(argv[1]);
    unsigned int numIterations = (unsigned int) atoi(argv[2]);
    unsigned int blockSize = (unsigned int) atoi(argv[3]);

    // Ensure that numNodes is a multiple of blockSize.
    if (numNodes % blockSize != 0) {
        numNodes = ((numNodes / blockSize) + 1) * blockSize;
    }

    size_t matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);

    // Use pinned (page-locked) host memory for improved data transfer efficiency.
    unsigned int *pathDistanceMatrix = nullptr;
    unsigned int *pathMatrix = nullptr;
    unsigned int *verificationPathDistanceMatrix = nullptr;
    unsigned int *verificationPathMatrix = nullptr;

    CUDA_CHECK(cudaMallocHost((void**)&pathDistanceMatrix, matrixSizeBytes));
    CUDA_CHECK(cudaMallocHost((void**)&pathMatrix, matrixSizeBytes));
    verificationPathDistanceMatrix = (unsigned int *) malloc(matrixSizeBytes);
    assert(verificationPathDistanceMatrix != nullptr);
    verificationPathMatrix = (unsigned int *) malloc(matrixSizeBytes);
    assert(verificationPathMatrix != nullptr);

    // Initialize the distance matrix with random values.
    srand(2);
    for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++) {
            int index = i * numNodes + j;
            pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
        }
    }
    // Set self-loops to 0.
    for (unsigned int i = 0; i < numNodes; i++) {
        pathDistanceMatrix[i * numNodes + i] = 0;
    }

    // Initialize the path matrix such that each entry is the source index.
    for (unsigned int i = 0; i < numNodes; ++i) {
        for (unsigned int j = 0; j < i; ++j) {
            pathMatrix[i * numNodes + j] = i;
            pathMatrix[j * numNodes + i] = j;
        }
        pathMatrix[i * numNodes + i] = i;
    }

    // Copy the initialized matrices to the verification buffers.
    memcpy(verificationPathDistanceMatrix, pathDistanceMatrix, matrixSizeBytes);
    memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);

    unsigned int numPasses = numNodes;

    // Compute grid and block dimensions.
    unsigned int globalThreads[2] = { numNodes, numNodes };
    unsigned int localThreads[2]  = { blockSize, blockSize };

    // Ensure the block does not exceed the maximum threads per block.
    if ((localThreads[0] * localThreads[1]) > 256) {
        blockSize = 16;
        localThreads[0] = blockSize;
        localThreads[1] = blockSize;
    }

    dim3 grids(globalThreads[0] / localThreads[0], globalThreads[1] / localThreads[1]);
    dim3 threads(localThreads[0], localThreads[1]);

    // Allocate device memory once.
    unsigned int *pathDistanceBuffer, *pathBuffer;
    CUDA_CHECK(cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&pathBuffer, matrixSizeBytes));

    // Create a CUDA stream to overlap memory transfers and kernel executions.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create CUDA events for precise timing to allow the CPU to remain less active.
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    float total_time_sec = 0.f;

    // Execute the algorithm for the specified number of iterations.
    for (unsigned int n = 0; n < numIterations; n++) {
        // Asynchronously copy the host matrix to the device.
        CUDA_CHECK(cudaMemcpyAsync(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes,
                                     cudaMemcpyHostToDevice, stream));

        // Record the start event.
        CUDA_CHECK(cudaEventRecord(startEvent, stream));

        // Launch a kernel for each Floyd–Warshall intermediate pass.
        for (unsigned int pass = 0; pass < numPasses; pass++) {
            floydWarshallPass<<<grids, threads, 0, stream>>>(pathDistanceBuffer,
                                                               pathBuffer,
                                                               numNodes,
                                                               pass);
        }
        // Record the stop event.
        CUDA_CHECK(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        // Measure the elapsed time in milliseconds.
        float elapsedTime_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_ms, startEvent, stopEvent));
        total_time_sec += elapsedTime_ms * 1e-3f;
    }

    printf("Average kernel execution time %f s\n", total_time_sec / numIterations);

    // Copy the result back to host.
    CUDA_CHECK(cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes,
                            cudaMemcpyDeviceToHost));

    // Clean up device resources.
    CUDA_CHECK(cudaFree(pathDistanceBuffer));
    CUDA_CHECK(cudaFree(pathBuffer));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // Verification: Run the CPU reference implementation.
    floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
    if (memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
        if (numNodes <= 8) {
            for (unsigned int i = 0; i < numNodes; i++) {
                for (unsigned int j = 0; j < numNodes; j++)
                    printf("host: %u ", verificationPathDistanceMatrix[i * numNodes + j]);
                printf("\n");
            }
            for (unsigned int i = 0; i < numNodes; i++) {
                for (unsigned int j = 0; j < numNodes; j++)
                    printf("device: %u ", pathDistanceMatrix[i * numNodes + j]);
                printf("\n");
            }
        }
    }

    // Free host memory.
    CUDA_CHECK(cudaFreeHost(pathDistanceMatrix));
    CUDA_CHECK(cudaFreeHost(pathMatrix));
    free(verificationPathDistanceMatrix);
    free(verificationPathMatrix);
    
    return 0;
}
