
/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define MAXDISTANCE    (200)

/**
 * Returns the lesser of the two unsigned integers a and b
 */
unsigned int minimum(unsigned int a, unsigned int b) 
{
  return (b < a) ? b : a;
}

/**
 * Reference CPU implementation of FloydWarshall PathFinding
 * for performance comparison.
 */
void floydWarshallCPUReference(unsigned int * pathDistanceMatrix,
    unsigned int * pathMatrix, unsigned int numNodes)
{
  unsigned int distanceYtoX, distanceYtoK, distanceKtoX, indirectDistance;
  unsigned int width = numNodes;
  unsigned int yXwidth;

  for(unsigned int k = 0; k < numNodes; ++k)
  {
    for(unsigned int y = 0; y < numNodes; ++y)
    {
      yXwidth = y * numNodes;
      for(unsigned int x = 0; x < numNodes; ++x)
      {
        distanceYtoX = pathDistanceMatrix[yXwidth + x];
        distanceYtoK = pathDistanceMatrix[yXwidth + k];
        distanceKtoX = pathDistanceMatrix[k * width + x];

        indirectDistance = distanceYtoK + distanceKtoX;
        if(indirectDistance < distanceYtoX)
        {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x] = k;
        }
      }
    }
  }
}

/*
 * Refactored GPU kernel using CUDA Cooperative Groups to fuse all
 * passes inside a single kernel launch. To avoid launching too many blocks,
 * the kernel processes the entire numNodes*numNodes elements using a grid-stride loop.
 *
 * Note: This kernel requires a GPU and CUDA runtime that supports
 * grid-synchronization via cooperative groups.
 */
extern "C" __global__ void floydWarshallKernel(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes)
{
  // Create a grid group for global synchronization.
  grid_group grid = this_grid();

  const unsigned int totalElements = numNodes * numNodes;
  for (unsigned int k = 0; k < numNodes; k++) {
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalElements; idx += blockDim.x * gridDim.x) {
      unsigned int y = idx / numNodes;
      unsigned int x = idx % numNodes;
      int oldWeight = pathDistanceBuffer[y * numNodes + x];
      int tempWeight = pathDistanceBuffer[y * numNodes + k] +
                       pathDistanceBuffer[k * numNodes + x];
      if (tempWeight < oldWeight)
      {
        pathDistanceBuffer[y * numNodes + x] = tempWeight;
        pathBuffer[y * numNodes + x] = k;
      }
    }
    // Global grid synchronization ensures that all threads have
    // updated the matrix before the next iteration.
    grid.sync();
  }
}

int main(int argc, char** argv) {
  // Set device flags to use blocking sync which can help reduce CPU spin.
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }
  // Read and adjust parameters.
  unsigned int numNodes = atoi(argv[1]);
  unsigned int numIterations = atoi(argv[2]);
  unsigned int blockSize = atoi(argv[3]);

  // Adjust numNodes to be a multiple of blockSize.
  if(numNodes % blockSize != 0) {
    numNodes = ((numNodes / blockSize) + 1) * blockSize;
  }

  unsigned int matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
  // Use pinned host memory for energy-efficient transfers.
  unsigned int* pathMatrix = nullptr;
  cudaMallocHost((void**)&pathMatrix, matrixSizeBytes);
  assert(pathMatrix != NULL);

  unsigned int* pathDistanceMatrix = nullptr;
  cudaMallocHost((void**)&pathDistanceMatrix, matrixSizeBytes);
  assert(pathDistanceMatrix != NULL);

  // Initialize the distance matrix.
  srand(2);
  for(unsigned int i = 0; i < numNodes; i++)
    for(unsigned int j = 0; j < numNodes; j++)
    {
      int index = i * numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
  // Distance from a node to itself is zero.
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    unsigned int idx = i * numNodes + i;
    pathDistanceMatrix[idx] = 0;
  }

  // Initialize the path matrix.
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    for(unsigned int j = 0; j < i; ++j)
    {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  // Allocate memory for verification using pinned host memory.
  unsigned int* verificationPathDistanceMatrix = nullptr;
  cudaMallocHost((void**)&verificationPathDistanceMatrix, matrixSizeBytes);
  assert(verificationPathDistanceMatrix != NULL);
  unsigned int* verificationPathMatrix = nullptr;
  cudaMallocHost((void**)&verificationPathMatrix, matrixSizeBytes);
  assert(verificationPathMatrix != NULL);

  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix, matrixSizeBytes);
  memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);

  // Compute total number of elements.
  unsigned int totalElements = numNodes * numNodes;
  // Use a 1D grid launch configuration to reduce the total number of blocks in cooperative launch.
  unsigned int threadsPerBlock = blockSize;
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  // Check if cooperative launch is supported
  if (!deviceProp.cooperativeLaunch) {
    fprintf(stderr, "ERROR: Cooperative launch not supported on this GPU.\n");
    exit(EXIT_FAILURE);
  }
  // Ensure blockSize does not exceed device maximum.
  if (threadsPerBlock > deviceProp.maxThreadsPerBlock) {
    threadsPerBlock = deviceProp.maxThreadsPerBlock;
  }
  unsigned int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

  // Calculate maximum cooperative blocks based on device properties.
  int maxBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / threadsPerBlock;
  unsigned int maxCoopBlocks = deviceProp.multiProcessorCount * maxBlocksPerSM;
  // Adjust grid dimension to meet cooperative launch residency requirement.
  if (numBlocks > maxCoopBlocks) {
    numBlocks = maxCoopBlocks;
  }
  if (numBlocks > deviceProp.maxGridSize[0]) {
    numBlocks = deviceProp.maxGridSize[0];
  }

  dim3 grids(numBlocks, 1, 1);
  dim3 threads(threadsPerBlock, 1, 1);

  // Allocate device buffers.
  unsigned int *pathDistanceBuffer, *pathBuffer;
  cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes);
  cudaMalloc((void**)&pathBuffer, matrixSizeBytes);

  float total_time = 0.f;

  // Use the cooperative group kernel to run all passes in one launch.
  for (unsigned int n = 0; n < numIterations; n++) {
    cudaMemcpy(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    void *kernelArgs[] = { (void *)&pathDistanceBuffer, (void *)&pathBuffer, (void *)&numNodes };
    cudaError_t launchStatus = cudaLaunchCooperativeKernel(reinterpret_cast<void*>(floydWarshallKernel),
                                                           grids, threads,
                                                           kernelArgs, 0, 0);
    if (launchStatus != cudaSuccess) {
      fprintf(stderr, "ERROR: Unable to launch cooperative kernel (must run on GPU with grid "
              "synchronization support): %s\n", cudaGetErrorString(launchStatus));
      exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / numIterations);

  cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
  cudaFree(pathDistanceBuffer);
  cudaFree(pathBuffer);

  // Verify results against the CPU reference implementation.
  floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
  
  if (memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0)
  {
    printf("PASS\n");
  }
  else
  {
    printf("FAIL\n");
    if (numNodes <= 8) 
    {
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

  cudaFreeHost(pathDistanceMatrix);
  cudaFreeHost(pathMatrix);
  cudaFreeHost(verificationPathDistanceMatrix);
  cudaFreeHost(verificationPathMatrix);
  return 0;
}
