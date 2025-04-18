
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
#include <cuda.h>
#include <math.h>

#define MAXDISTANCE    (200)

/**
 * Returns the lesser of the two unsigned integers a and b.
 */
unsigned int minimum(unsigned int a, unsigned int b) 
{
  return (b < a) ? b : a;
}

/**
 * Reference CPU implementation of FloydWarshall PathFinding.
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
      yXwidth =  y * numNodes;
      for(unsigned int x = 0; x < numNodes; ++x)
      {
        distanceYtoX = pathDistanceMatrix[yXwidth + x];
        distanceYtoK = pathDistanceMatrix[yXwidth + k];
        distanceKtoX = pathDistanceMatrix[k * width + x];

        indirectDistance = distanceYtoK + distanceKtoX;

        if(indirectDistance < distanceYtoX)
        {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x]         = k;
        }
      }
    }
  }
}

/**
 * CUDA kernel for the FloydWarshall algorithm.
 * This version performs one intermediate node pass per kernel launch.
 * The 'dummy_k' parameter is used to pass the current intermediate node index.
 */
__global__ void floydWarshallPass(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes,
    const unsigned int dummy_k)  // 'dummy_k' now represents the current k iteration
{
  int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  int yValue = threadIdx.y + blockIdx.y * blockDim.y;

  // Guard against threads outside the problem domain.
  if (xValue >= numNodes || yValue >= numNodes)
    return;

  int index = yValue * numNodes + xValue;
  int oldWeight = pathDistanceBuffer[index];
  int newWeight = pathDistanceBuffer[yValue * numNodes + dummy_k] +
                  pathDistanceBuffer[dummy_k * numNodes + xValue];

  if(newWeight < oldWeight) {
    pathDistanceBuffer[index] = newWeight;
    pathBuffer[index] = dummy_k;
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }
  // Parse command-line arguments.
  unsigned int numNodes = atoi(argv[1]);
  unsigned int numIterations = atoi(argv[2]);
  unsigned int blockSize = atoi(argv[3]);

  // Adjust numNodes to be a multiple of blockSize.
  if(numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  size_t matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);

  // Allocate pinned host memory to improve transfer efficiency.
  unsigned int* pathMatrix = NULL;
  unsigned int* pathDistanceMatrix = NULL;
  unsigned int* verificationPathDistanceMatrix = NULL;
  unsigned int* verificationPathMatrix = NULL;
  cudaError_t err;

  err = cudaMallocHost((void**)&pathDistanceMatrix, matrixSizeBytes);
  assert(err == cudaSuccess && "cudaMallocHost failed for pathDistanceMatrix");
  err = cudaMallocHost((void**)&pathMatrix, matrixSizeBytes);
  assert(err == cudaSuccess && "cudaMallocHost failed for pathMatrix");
  err = cudaMallocHost((void**)&verificationPathDistanceMatrix, matrixSizeBytes);
  assert(err == cudaSuccess && "cudaMallocHost failed for verificationPathDistanceMatrix");
  err = cudaMallocHost((void**)&verificationPathMatrix, matrixSizeBytes);
  assert(err == cudaSuccess && "cudaMallocHost failed for verificationPathMatrix");

  // Initialize the path distance matrix.
  srand(2);
  for(unsigned int i = 0; i < numNodes; i++) {
    for(unsigned int j = 0; j < numNodes; j++)
    {
      int index = i * numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
  }
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    unsigned int iXWidth = i * numNodes;
    pathDistanceMatrix[iXWidth + i] = 0;
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

  // Copy the initial host matrices into the verification arrays.
  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix, matrixSizeBytes);
  memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);

  // Determine grid and block dimensions.
  dim3 threads(blockSize, blockSize);
  unsigned int gridDim = numNodes / blockSize;
  dim3 grids(gridDim, gridDim);
  if(blockSize * blockSize > 256) {
    // Adjust block dimensions for energy and resource efficiency.
    blockSize = 16;
    threads.x = blockSize;
    threads.y = blockSize;
    gridDim = numNodes / blockSize;
    grids.x = gridDim;
    grids.y = gridDim;
  }

  // Allocate device memory.
  unsigned int *pathDistanceBuffer, *pathBuffer;
  cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes);
  cudaMalloc((void**)&pathBuffer, matrixSizeBytes);

  // Create a CUDA stream to enable asynchronous memory copies and kernel launches.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float total_time = 0.f;

  // Run the iterations.
  for (unsigned int n = 0; n < numIterations; n++) {
    // Asynchronously copy the host matrix to device.
    cudaMemcpyAsync(pathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(pathBuffer, pathMatrix, matrixSizeBytes, cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    auto start = std::chrono::steady_clock::now();

    // Loop over all intermediate nodes.
    for (unsigned int k = 0; k < numNodes; k++) {
      unsigned int dummy = k;  // Pass the current intermediate node
      floydWarshallPass<<<grids, threads, 0, stream>>>(pathDistanceBuffer, pathBuffer, numNodes, dummy);
      cudaStreamSynchronize(stream);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / numIterations);

  // Copy the result back to host asynchronously and wait for completion.
  cudaMemcpyAsync(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Free device memory.
  cudaFree(pathDistanceBuffer);
  cudaFree(pathBuffer);
  cudaStreamDestroy(stream);

  // Verify the result against the CPU reference.
  floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
  if(memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0)
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
          printf("host: %u ", verificationPathDistanceMatrix[i*numNodes+j]);
        printf("\n");
      }
      for (unsigned int i = 0; i < numNodes; i++) {
        for (unsigned int j = 0; j < numNodes; j++)
          printf("device: %u ", pathDistanceMatrix[i*numNodes+j]);
        printf("\n");
      }
    }
  }

  // Free pinned host memory.
  cudaFreeHost(pathDistanceMatrix);
  cudaFreeHost(pathMatrix);
  cudaFreeHost(verificationPathDistanceMatrix);
  cudaFreeHost(verificationPathMatrix);

  return 0;
}
