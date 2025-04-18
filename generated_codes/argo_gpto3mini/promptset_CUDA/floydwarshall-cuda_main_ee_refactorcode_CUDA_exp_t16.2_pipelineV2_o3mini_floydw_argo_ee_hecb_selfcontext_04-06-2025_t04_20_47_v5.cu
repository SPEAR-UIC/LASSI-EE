
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <cuda.h>

#define MAXDISTANCE    (200)

// Macro for CUDA API error checking.
#define CHECK_CUDA(call)                                                        \
  {                                                                             \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error at %s:%d code=%d (%s)\n", __FILE__, __LINE__,  \
              err, cudaGetErrorString(err));                                    \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

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

/*!
 * The floyd Warshall algorithm is a multipass algorithm
 * that calculates the shortest path between each pair of
 * nodes represented by pathDistanceBuffer.
 */
__global__ void floydWarshallPass(
    unsigned int *__restrict__ pathDistanceBuffer,
    unsigned int *__restrict__ pathBuffer,
    const unsigned int numNodes,
    const unsigned int pass)
{
  int xValue = threadIdx.x + blockIdx.x * blockDim.x;
  int yValue = threadIdx.y + blockIdx.y * blockDim.y;

  // Add bounds checking to prevent threads from operating on out-of-bound indices
  if(xValue < numNodes && yValue < numNodes)
  {
    int k = pass;
    int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue];
    int tempWeight = pathDistanceBuffer[yValue * numNodes + k] +
                     pathDistanceBuffer[k * numNodes + xValue];

    if (tempWeight < oldWeight)
    {
      pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight;
      pathBuffer[yValue * numNodes + xValue] = k;
    }
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

  if(numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  unsigned int* pathMatrix = NULL;
  unsigned int* pathDistanceMatrix = NULL;
  unsigned int* verificationPathDistanceMatrix = NULL;
  unsigned int* verificationPathMatrix = NULL;
  unsigned int matrixSizeBytes = numNodes * numNodes * sizeof(unsigned int);

  // Allocate host memory using pinned (page-locked) memory.
  CHECK_CUDA(cudaMallocHost((void **)&pathDistanceMatrix, matrixSizeBytes));
  CHECK_CUDA(cudaMallocHost((void **)&pathMatrix, matrixSizeBytes));
  CHECK_CUDA(cudaMallocHost((void **)&verificationPathDistanceMatrix, numNodes * numNodes * sizeof(unsigned int)));
  CHECK_CUDA(cudaMallocHost((void **)&verificationPathMatrix, numNodes * numNodes * sizeof(unsigned int)));

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

  for(unsigned int i = 0; i < numNodes; ++i)
  {
    for(unsigned int j = 0; j < i; ++j)
    {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  memcpy(verificationPathDistanceMatrix, pathDistanceMatrix,
      numNodes * numNodes * sizeof(unsigned int));
  memcpy(verificationPathMatrix, pathMatrix, numNodes * numNodes * sizeof(unsigned int));

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
  dim3 threads (localThreads[0],localThreads[1]);

  unsigned int *pathDistanceBuffer, *pathBuffer;
  CHECK_CUDA(cudaMalloc((void**)&pathDistanceBuffer, matrixSizeBytes));
  CHECK_CUDA(cudaMalloc((void**)&pathBuffer, matrixSizeBytes));

  // Allocate a device memory buffer to hold the original path distance matrix.
  unsigned int *origPathDistanceBuffer;
  CHECK_CUDA(cudaMalloc((void**)&origPathDistanceBuffer, matrixSizeBytes));
  // Copy the initial (static) matrix from host to device once.
  CHECK_CUDA(cudaMemcpy(origPathDistanceBuffer, pathDistanceMatrix, matrixSizeBytes, cudaMemcpyHostToDevice));

  float total_time = 0.f;

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Create a CUDA stream to allow asynchronous operations
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // For each iteration, use an asynchronous device-to-device copy and launch kernels on a stream.
  for (unsigned int n = 0; n < numIterations; n++) {
    // Asynchronously copy from the stored device-resident original matrix 
    // to the working buffer using the custom stream.
    CHECK_CUDA(cudaMemcpyAsync(pathDistanceBuffer, origPathDistanceBuffer, matrixSizeBytes,
                    cudaMemcpyDeviceToDevice, stream));

    // Record the start event on the stream.
    CHECK_CUDA(cudaEventRecord(start, stream));

    // Launch a kernel for each pass asynchronously on the same stream.
    for(unsigned int i = 0; i < numPasses; i++)
    {
      floydWarshallPass<<< grids, threads, 0, stream>>> (pathDistanceBuffer, pathBuffer, numNodes, i);
    }

    // Record the stop event on the stream.
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_time += elapsed_ms;
  }

  // Print average kernel execution time in seconds.
  printf("Average kernel execution time %f (s)\n", (total_time * 1e-3f) / numIterations);

  CHECK_CUDA(cudaMemcpy(pathDistanceMatrix, pathDistanceBuffer, matrixSizeBytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(pathDistanceBuffer));
  CHECK_CUDA(cudaFree(pathBuffer));
  CHECK_CUDA(cudaFree(origPathDistanceBuffer));

  // Destroy CUDA stream and events
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

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

  // Free pinned host memory
  CHECK_CUDA(cudaFreeHost(pathDistanceMatrix));
  CHECK_CUDA(cudaFreeHost(pathMatrix));
  CHECK_CUDA(cudaFreeHost(verificationPathDistanceMatrix));
  CHECK_CUDA(cudaFreeHost(verificationPathMatrix));
  return 0;
}
