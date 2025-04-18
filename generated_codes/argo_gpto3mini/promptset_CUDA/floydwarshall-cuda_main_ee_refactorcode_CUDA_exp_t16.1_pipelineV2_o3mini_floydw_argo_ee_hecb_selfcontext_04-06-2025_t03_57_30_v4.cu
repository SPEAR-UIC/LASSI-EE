
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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
 * for performance comparison
 * @param pathDistanceMatrix Distance between nodes of a graph
 * @param intermediate node between two nodes of a graph
 * @param number of nodes in the graph
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
      yXwidth =  y*numNodes;
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
 * Fused kernel for performing all Floyd–Warshall passes.
 * Although the kernel signature still includes the 'pass' parameter,
 * it is now unused since all passes are processed in one launch.
 */
__global__ void floydWarshallPass(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes,
    const unsigned int pass)  // pass parameter is kept for compatibility
{
  int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  int yValue = threadIdx.y + blockIdx.y * blockDim.y;

  // Ensure we are not out-of-bounds.
  if (xValue >= numNodes || yValue >= numNodes)
    return;

  // Get the grid-level cooperative group for global synchronization.
  cg::grid_group grid = cg::this_grid();

  // Iterate over every intermediate node (pass)
  for (unsigned int k = 0; k < numNodes; k++) {
    int index = yValue * numNodes + xValue;
    int oldWeight = pathDistanceBuffer[index];
    int tempWeight = pathDistanceBuffer[yValue * numNodes + k] + 
                     pathDistanceBuffer[k * numNodes + xValue];

    if (tempWeight < oldWeight)
    {
      pathDistanceBuffer[index] = tempWeight;
      pathBuffer[index] = k;
    }
    // Ensure all threads have updated the data for the current pass
    grid.sync();
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }
  // There are three required command-line arguments
  unsigned int numNodes = atoi(argv[1]);
  unsigned int numIterations = atoi(argv[2]);
  unsigned int blockSize = atoi(argv[3]);

  // numNodes is a multiple of blockSize
  if(numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  // allocate and init memory used by host using pinned memory for energy efficient transfers
  unsigned int* pathMatrix = NULL;
  unsigned int* pathDistanceMatrix = NULL;
  unsigned int* verificationPathDistanceMatrix = NULL;
  unsigned int* verificationPathMatrix = NULL;
  unsigned int matrixSizeBytes;

  matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);
  cudaMallocHost((void**)&pathDistanceMatrix, matrixSizeBytes);
  assert (pathDistanceMatrix != NULL);

  cudaMallocHost((void**)&pathMatrix, matrixSizeBytes);
  assert (pathMatrix != NULL);

  // input must be initialized; otherwise host and device results may be different
  srand(2);
  for(unsigned int i = 0; i < numNodes; i++)
    for(unsigned int j = 0; j < numNodes; j++)
    {
      int index = i*numNodes + j;
      pathDistanceMatrix[index] = rand() % (MAXDISTANCE + 1);
    }
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    unsigned int iXWidth = i * numNodes;
    pathDistanceMatrix[iXWidth + i] = 0;
  }

  /*
   * pathMatrix is the intermediate node from which the path passes
   * pathMatrix(i,j) = k means the shortest path from i to j
   * passes through an intermediate node k
   * Initialized such that pathMatrix(i,j) = i
   */
  for(unsigned int i = 0; i < numNodes; ++i)
  {
    for(unsigned int j = 0; j < i; ++j)
    {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  cudaMallocHost((void**)&verificationPathDistanceMatrix, numNodes * numNodes * sizeof(int));
  assert (verificationPathDistanceMatrix != NULL);

  cudaMallocHost((void**)&verificationPathMatrix, numNodes * numNodes * sizeof(int));
  assert(verificationPathMatrix != NULL);

  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix,
      numNodes * numNodes * sizeof(int));
  memcpy(verificationPathMatrix, pathMatrix, numNodes*numNodes*sizeof(int));

  unsigned int numPasses = numNodes;

  unsigned int globalThreads[2] = {numNodes, numNodes};
  unsigned int localThreads[2] = {blockSize, blockSize};

  if((unsigned int)(localThreads[0] * localThreads[0]) > 256)
  {
    blockSize = 16;
    localThreads[0] = blockSize;
    localThreads[1] = blockSize;
  }

  dim3 grids( globalThreads[0]/localThreads[0], globalThreads[1]/localThreads[1]);
  dim3 threads (localThreads[0], localThreads[1]);

  unsigned int *pathDistanceBuffer, *pathBuffer;
  cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes);
  cudaMalloc((void**)&pathBuffer, matrixSizeBytes);

  // Allocate an additional device buffer to store the initial matrix so we do not copy from the host repeatedly.
  unsigned int *initialPathDistanceBuffer;
  cudaMalloc((void**)&initialPathDistanceBuffer, matrixSizeBytes);
  cudaMemcpy(initialPathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice);

  float total_time = 0.f;

  // For each iteration, restore the input from the device copy and run the (fused) kernel.
  for (unsigned int n = 0; n < numIterations; n++) {
    // Use device-to-device memcpy, which is generally faster and consumes less energy,
    // instead of copying from the host on every iteration.
    cudaMemcpy(pathDistanceBuffer, initialPathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch a single cooperative kernel that fuses all passes
    floydWarshallPass <<< grids, threads >>> (pathDistanceBuffer, pathBuffer, numNodes, 0);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / numIterations);

  cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost);
  cudaFree(pathDistanceBuffer);
  cudaFree(pathBuffer);
  cudaFree(initialPathDistanceBuffer);

  // verify
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

  cudaFreeHost(pathDistanceMatrix);
  cudaFreeHost(pathMatrix);
  cudaFreeHost(verificationPathDistanceMatrix);
  cudaFreeHost(verificationPathMatrix);
  return 0;
}
