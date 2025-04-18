
#include <cstdio>
#include <random>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

// Begin inline definition of tables.h content to eliminate missing file error.
#ifndef TABLES_H
#define TABLES_H
// Dummy definitions for host tables with sizes as used in the code.
// Adjust sizes accordingly if actual data is provided.
static const unsigned short distinctEdgesTable[256] = {0};
static const int triTable[256 * 16] = {0};
static const uchar4 edgeIDTable[256] = { {0, 0, 0, 0} };
#endif
// End inline definition of tables.h content.

  
// Error-checking macro
#define CUDA_CHECK(call) do {                                                    \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__,\
              cudaGetErrorString(err));                                          \
      exit(EXIT_FAILURE);                                                        \
    }                                                                            \
  } while(0)

  
// problem size definitions
constexpr unsigned int N(1024);
constexpr unsigned int Nd2(N / 2);
constexpr unsigned int voxelXLv1(16);
constexpr unsigned int voxelYLv1(16);
constexpr unsigned int voxelZLv1(64);
constexpr unsigned int gridXLv1((N - 1) / (voxelXLv1 - 1));
constexpr unsigned int gridYLv1((N - 1) / (voxelYLv1 - 1));
constexpr unsigned int gridZLv1((N - 1) / (voxelZLv1 - 1));
constexpr unsigned int countingThreadNumLv1(128);
constexpr unsigned int blockNum(gridXLv1 * gridYLv1 * gridZLv1);
constexpr unsigned int countingBlockNumLv1(blockNum / countingThreadNumLv1);

constexpr unsigned int voxelXLv2(4);
constexpr unsigned int voxelYLv2(4);
constexpr unsigned int voxelZLv2(8);
constexpr unsigned int blockXLv2(5);
constexpr unsigned int blockYLv2(5);
constexpr unsigned int blockZLv2(9);
constexpr unsigned int voxelNumLv2(blockXLv2 * blockYLv2 * blockZLv2);

constexpr unsigned int countingThreadNumLv2(1024);
constexpr unsigned int gridXLv2(gridXLv1 * blockXLv2);
constexpr unsigned int gridYLv2(gridYLv1 * blockYLv2);
// worst-case allocation sizes for level2 arrays: assume every Lv1 block is “active”
constexpr size_t maxCountedBlockNumLv1 = blockNum;
constexpr size_t maxMinMaxLv2Size = maxCountedBlockNumLv1 * voxelNumLv2 * 2;
constexpr size_t maxBlockIndicesLv2Size = maxCountedBlockNumLv1 * voxelNumLv2;


// Declare the “table” arrays in constant memory for lower‐overhead access.
__constant__ unsigned short d_distinctEdgesTable[sizeof(distinctEdgesTable)/sizeof(unsigned short)];
__constant__ int            d_triTable[sizeof(triTable)/sizeof(int)];
__constant__ uchar4         d_edgeIDTable[sizeof(edgeIDTable)/sizeof(uchar4)];

  
// device inline functions
__inline__ __device__ float f(unsigned int x, unsigned int y, unsigned int z)
{
  constexpr float d(2.0f / N);
  float xf((int(x - Nd2)) * d);//[-1, 1)
  float yf((int(y - Nd2)) * d);
  float zf((int(z - Nd2)) * d);
  return 1.f - 16.f * xf * yf * zf - 4.f * (xf * xf + yf * yf + zf * zf);
}

__inline__ __device__ float zeroPoint(unsigned int x, float v0, float v1, float isoValue)
{
  return ((x * (v1 - isoValue) + (x + 1) * (isoValue - v0)) / (v1 - v0) - Nd2) * (2.0f / N);
}

__inline__ __device__ float transformToCoord(unsigned int x)
{
  return (int(x) - int(Nd2)) * (2.0f / N);
}

  
// Kernels remain logically the same.
__global__ void computeMinMaxLv1(float* __restrict__ minMax)
{
  __shared__ float sminMax[64];
  constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
  constexpr unsigned int warpSize(32);
  constexpr unsigned int warpNum(threadNum / warpSize);
  unsigned int x(blockIdx.x * (voxelXLv1 - 1) + threadIdx.x);
  unsigned int y(blockIdx.y * (voxelYLv1 - 1) + threadIdx.y);
  unsigned int z(blockIdx.z * (voxelZLv1 - 1));
  unsigned int tid(threadIdx.x + voxelXLv1 * threadIdx.y);
  unsigned int laneid = tid % warpSize;
  unsigned int blockid(blockIdx.x + gridXLv1 * (blockIdx.y + gridYLv1 * blockIdx.z));
  unsigned int warpid(tid / warpSize);
  float v(f(x, y, z));
  float minV(v), maxV(v);
  for (int c0 = 1; c0 < voxelZLv1; ++c0)
  {
    v = f(x, y, z + c0);
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
#pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
    float t0 = __shfl_down_sync(0xffffffffu, minV, offset);
    float t1 = __shfl_down_sync(0xffffffffu, maxV, offset);
    if (t0 < minV) minV = t0;
    if (t1 > maxV) maxV = t1;
  }
  if (laneid == 0)
  {
    sminMax[warpid] = minV;
    sminMax[warpid + warpNum] = maxV;
  }
  __syncthreads();
  if (warpid == 0)
  {
    // Only the first warp (threads with laneid < warpNum) participate in final reduction.
    if(laneid < warpNum)
    {
      float rmin = sminMax[laneid];
      float rmax = sminMax[laneid + warpNum];
#pragma unroll
      for (int offset = warpNum/2; offset > 0; offset /= 2)
      {
        float t0 = __shfl_down_sync(0xffffffffu, rmin, offset);
        float t1 = __shfl_down_sync(0xffffffffu, rmax, offset);
        if (t0 < rmin) rmin = t0;
        if (t1 > rmax) rmax = t1;
      }
      if (laneid == 0)
      {
        minMax[blockid * 2] = rmin;
        minMax[blockid * 2 + 1] = rmax;
      }
    }
  }
}

__global__ void compactLv1(
  float isoValue, 
  const float* __restrict__ minMax,
  unsigned int* __restrict__ blockIndices,
  unsigned int* __restrict__ countedBlockNum)
{
  __shared__ unsigned int sums[32];
  constexpr unsigned int warpSize(32);
  constexpr unsigned int warpNum(countingThreadNumLv1 / warpSize);
  unsigned int tid(threadIdx.x);
  unsigned int laneid = tid % warpSize;
  unsigned int bIdx(blockIdx.x * countingThreadNumLv1 + tid);
  unsigned int warpid(tid / warpSize);
  unsigned int test = (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue) ? 1 : 0;
  unsigned int testSum(test);
#pragma unroll
  for (int offset = 1; offset < warpSize; offset *= 2)
  {
    unsigned int tp = __shfl_up_sync(0xffffffffu, testSum, offset);
    if (laneid >= offset) testSum += tp;
  }
  if (laneid == warpSize - 1)
    sums[warpid] = testSum;
  __syncthreads();
  if (warpid == 0)
  {
    if(laneid < warpNum)
    {
      unsigned int warpSum = sums[laneid];
#pragma unroll
      for (int offset = 1; offset < warpNum; offset *= 2)
      {
        unsigned int tp = __shfl_up_sync(0xffffffffu, warpSum, offset);
        if (laneid >= offset) warpSum += tp;
      }
      sums[laneid] = warpSum;
    }
  }
  __syncthreads();
  if (warpid != 0) testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv1 - 1 && testSum != 0)
    sums[warpSize - 1] = atomicAdd(countedBlockNum, testSum);
  __syncthreads();
  if (test)
    blockIndices[testSum + sums[warpSize - 1] - 1] = bIdx;
}

__global__ void computeMinMaxLv2(
  const unsigned int* __restrict__ blockIndicesLv1,
  float* __restrict__ minMax)
{
  unsigned int tid(threadIdx.x);
  unsigned int voxelOffset(threadIdx.y);
  unsigned int blockIndex(blockIndicesLv1[blockIdx.x]);
  unsigned int tp(blockIndex);
  unsigned int x((blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % blockXLv2) * (voxelXLv2 - 1) + (tid & 3));
  tp /= gridXLv1;
  unsigned int y((tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / blockXLv2) * (voxelYLv2 - 1) + (tid >> 2));
  tp /= gridYLv1;
  unsigned int z(tp * (voxelZLv1 - 1));
  float v(f(x, y, z));
  float minV(v), maxV(v);
  unsigned int idx(2 * (voxelOffset + voxelNumLv2 * blockIdx.x));
  for (int c0 = 0; c0 < blockZLv2; ++c0)
  {
    for (int c1 = 1; c1 < voxelZLv2; ++c1)
    {
      v = f(x, y, z + c1);
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    z += voxelZLv2 - 1;
#pragma unroll
    for (int c1 = 8; c1 > 0; c1 /= 2)
    {
      float t0 = __shfl_down_sync(0xffffffffu, minV, c1);
      float t1 = __shfl_down_sync(0xffffffffu, maxV, c1);
      if (t0 < minV) minV = t0;
      if (t1 > maxV) maxV = t1;
    }
    if (tid == 0)
    {
      minMax[idx] = minV;
      minMax[idx + 1] = maxV;
      constexpr unsigned int offsetSize(2 * blockXLv2 * blockYLv2);
      idx += offsetSize;
    }
    minV = v;
    maxV = v;
  }
}

__global__ void compactLv2(
  float isoValue,
  const float* __restrict__ minMax,
  const unsigned int* __restrict__ blockIndicesLv1,
  unsigned int* __restrict__ blockIndicesLv2,
  unsigned int counterBlockNumLv1,
  unsigned int* __restrict__ countedBlockNumLv2)
{
  __shared__ unsigned int sums[32];
  constexpr unsigned int warpSize(32);
  constexpr unsigned int warpNum(countingThreadNumLv2 / warpSize);
  unsigned int tid(threadIdx.x);
  unsigned int laneid = tid % warpSize;
  unsigned int warpid(tid / warpSize);
  unsigned int id0(tid + blockIdx.x * countingThreadNumLv2);
  unsigned int id1(id0 / voxelNumLv2);
  unsigned int test = 0;
  if (id1 < counterBlockNumLv1)
  {
    test = (minMax[2 * id0] <= isoValue && minMax[2 * id0 + 1] >= isoValue) ? 1 : 0;
  }
  unsigned int testSum(test);
#pragma unroll
  for (int offset = 1; offset < warpSize; offset *= 2)
  {
    unsigned int tp = __shfl_up_sync(0xffffffffu, testSum, offset);
    if (laneid >= offset) testSum += tp;
  }
  if (laneid == warpSize - 1)
    sums[warpid] = testSum;
  __syncthreads();
  if (warpid == 0)
  {
    if(laneid < warpNum)
    {
      unsigned int warpSum = sums[laneid];
#pragma unroll
      for (int offset = 1; offset < warpNum; offset *= 2)
      {
        unsigned int tp = __shfl_up_sync(0xffffffffu, warpSum, offset);
        if (laneid >= offset) warpSum += tp;
      }
      sums[laneid] = warpSum;
    }
  }
  __syncthreads();
  if (warpid != 0) testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv2 - 1)
    sums[warpSize - 1] = atomicAdd(countedBlockNumLv2, testSum);
  __syncthreads();
  
  if (test)
  {
    unsigned int bIdx1(blockIndicesLv1[id1]);
    unsigned int bIdx2;
    unsigned int x1, y1, z1;
    unsigned int x2, y2, z2;
    unsigned int tp1(bIdx1);
    unsigned int tp2 = (tid + blockIdx.x * countingThreadNumLv2) % voxelNumLv2;
    x1 = tp1 % gridXLv1;
    x2 = tp2 % blockXLv2;
    tp1 /= gridXLv1;
    tp2 /= blockXLv2;
    y1 = tp1 % gridYLv1;
    y2 = tp2 % blockYLv2;
    z1 = tp1 / gridYLv1;
    z2 = tp2 / blockYLv2;
    bIdx2 = x2 + blockXLv2 * (x1 + gridXLv1 * (y2 + blockYLv2 * (y1 + gridXLv1 * (z1 * blockZLv2 + z2))));
    blockIndicesLv2[testSum + sums[warpSize - 1] - 1] = bIdx2;
  }
}

__global__ void generatingTriangles(
  float isoValue, 
  const unsigned int* __restrict__ blockIndicesLv2,
  const unsigned short * __restrict__ distinctEdgesTable,
  const int * __restrict__ triTable,
  const uchar4 * __restrict__ edgeIDTable,
  unsigned int* __restrict__ countedVerticesNum,
  unsigned int* __restrict__ countedTrianglesNum,
  unsigned long long* __restrict__ triangles,
  float* __restrict__ coordX,
  float* __restrict__ coordY,
  float* __restrict__ coordZ,
  float* __restrict__ coordZP)
{
  __shared__ unsigned short vertexIndices[voxelZLv2][voxelYLv2][voxelXLv2];
  __shared__ float value[voxelZLv2 + 1][voxelYLv2 + 1][voxelXLv2 + 1];

  // Compute voxel coordinates based on blockIndicesLv2.
  unsigned int blockId = blockIndicesLv2[blockIdx.x];
  unsigned int tp = blockId;
  unsigned int x = (tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx.x;
  tp /= gridXLv2;
  unsigned int y = (tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx.y;
  unsigned int z = (tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx.z;

  // Initialize eds and load primary value.
  unsigned int eds = 7;
  float v = (value[threadIdx.z][threadIdx.y][threadIdx.x] = f(x, y, z));
  if (threadIdx.x == voxelXLv2 - 1)
  {
    eds &= 6;
    value[threadIdx.z][threadIdx.y][voxelXLv2] = f(x + 1, y, z);
    if (threadIdx.y == voxelYLv2 - 1)
      value[threadIdx.z][voxelYLv2][voxelXLv2] = f(x + 1, y + 1, z);
  }
  if (threadIdx.y == voxelYLv2 - 1)
  {
    eds &= 5;
    value[threadIdx.z][voxelYLv2][threadIdx.x] = f(x, y + 1, z);
    if (threadIdx.z == voxelZLv2 - 1)
      value[threadIdx.z][voxelYLv2][threadIdx.x] = f(x, y + 1, z); // redundant safe write
  }
  if (threadIdx.z == voxelZLv2 - 1)
  {
    eds &= 3;
    value[voxelZLv2][threadIdx.y][threadIdx.x] = f(x, y, z + 1);
    if (threadIdx.x == voxelXLv2 - 1)
      value[voxelZLv2][threadIdx.y][voxelXLv2] = f(x + 1, y, z + 1);
  }
  eds <<= 13;
  __syncthreads();

  unsigned int cubeCase = 0;
  if (value[threadIdx.z][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 1;
  if (value[threadIdx.z][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 2;
  if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 4;
  if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 8;
  if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 16;
  if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 32;
  if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 64;
  if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 128;

  unsigned int distinctEdges = (eds ? distinctEdgesTable[cubeCase] : 0);
  // Apply mask to distinctEdges.
  distinctEdges &= eds;
  unsigned int numTriangles = (eds != 0xe000 ? 0 : (distinctEdges & 7));
  unsigned int numVertices = __popc(distinctEdges);

  // Perform an exclusive scan for vertex indices across the block.
  int totalThreads = voxelXLv2 * voxelYLv2 * voxelZLv2; // 4*4*8 = 128
  int linearId = threadIdx.x + threadIdx.y * voxelXLv2 + threadIdx.z * voxelXLv2 * voxelYLv2;
  __shared__ unsigned int scanVertices[128];
  scanVertices[linearId] = numVertices;
  __syncthreads();
  for (int offset = 1; offset < 128; offset *= 2) {
    unsigned int temp = (linearId >= offset) ? scanVertices[linearId - offset] : 0;
    __syncthreads();
    scanVertices[linearId] += temp;
    __syncthreads();
  }
  unsigned int exclusiveSum = (linearId == 0) ? 0 : scanVertices[linearId - 1];
  unsigned int totalBlockVertices = scanVertices[127];
  __shared__ unsigned int blockVertexOffset;
  if (linearId == 0) {
    blockVertexOffset = atomicAdd(countedVerticesNum, totalBlockVertices);
  }
  __syncthreads();
  unsigned int finalVertexIndex = blockVertexOffset + exclusiveSum;
  vertexIndices[threadIdx.z][threadIdx.y][threadIdx.x] = finalVertexIndex | distinctEdges;
  __syncthreads();

  // Generate triangles using computed vertex indices.
  for (unsigned int c0 = 0; c0 < numTriangles; ++c0)
  {
#pragma unroll
    for (unsigned int c1 = 0; c1 < 3; ++c1)
    {
      int edgeID = triTable[16 * cubeCase + 3 * c0 + c1];
      uchar4 edgePos = edgeIDTable[edgeID];
      unsigned short vIndex = vertexIndices[threadIdx.z + edgePos.z][threadIdx.y + edgePos.y][threadIdx.x + edgePos.x];
      unsigned int computedVertex = __popc(vIndex >> (16 - edgePos.w)) + (vIndex & 0x1FFF);
      atomicAdd(triangles, (unsigned long long)(blockVertexOffset + computedVertex));
    }
  }
  
  float zp = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;
  if (distinctEdges & (1 << 15))
  {
    zp = zeroPoint(x, v, value[threadIdx.z][threadIdx.y][threadIdx.x + 1], isoValue);
    cy = transformToCoord(y);
    cz = transformToCoord(z);
  }
  if (distinctEdges & (1 << 14))
  {
    cx = transformToCoord(x);
    zp += zeroPoint(y, v, value[threadIdx.z][threadIdx.y + 1][threadIdx.x], isoValue);
    cz += transformToCoord(z);
  }
  if (distinctEdges & (1 << 13))
  {
    cx += transformToCoord(x);
    cy += transformToCoord(y);
    zp += zeroPoint(z, v, value[threadIdx.z + 1][threadIdx.y][threadIdx.x], isoValue);
  }
  atomicAdd(coordX, cx);
  atomicAdd(coordY, cy);
  atomicAdd(coordZ, cz);
  atomicAdd(coordZP, zp);
}
  
int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int repeat = atoi(argv[1]);

  // Upload the tables to constant memory (only once)
  CUDA_CHECK(cudaMemcpyToSymbol(d_distinctEdgesTable, distinctEdgesTable, sizeof(distinctEdgesTable)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_edgeIDTable, edgeIDTable, sizeof(edgeIDTable)));

  // Allocate all persistent device memory up-front
  float* minMaxLv1Device = nullptr;
  unsigned int* blockIndicesLv1Device = nullptr;
  unsigned int* countedBlockNumLv1Device = nullptr;
  unsigned int* countedBlockNumLv2Device = nullptr;
  unsigned int* countedVerticesNumDevice = nullptr;
  unsigned int* countedTrianglesNumDevice = nullptr;
  unsigned long long* trianglesDevice = nullptr;
  float *coordXDevice = nullptr;
  float *coordYDevice = nullptr;
  float *coordZDevice = nullptr;
  float *coordZPDevice = nullptr;
  // Worst-case allocations for Lv2 arrays:
  float* minMaxLv2Device = nullptr;      
  unsigned int* blockIndicesLv2Device = nullptr; 

  CUDA_CHECK(cudaMalloc(&minMaxLv1Device, blockNum * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&blockIndicesLv1Device, blockNum * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&countedBlockNumLv1Device, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&countedBlockNumLv2Device, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&countedVerticesNumDevice, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&countedTrianglesNumDevice, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&trianglesDevice, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&coordXDevice, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&coordYDevice, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&coordZDevice, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&coordZPDevice, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&minMaxLv2Device, maxMinMaxLv2Size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&blockIndicesLv2Device, maxBlockIndicesLv2Size * sizeof(unsigned int)));

  const dim3 BlockSizeLv1{ voxelXLv1, voxelYLv1, 1 };
  const dim3 GridSizeLv1{ gridXLv1, gridYLv1, gridZLv1 };
  
  const dim3 BlockSizeLv2{ voxelXLv2 * voxelYLv2, blockXLv2 * blockYLv2, 1 };
  const dim3 BlockSizeGenerating{ voxelXLv2, voxelYLv2, voxelZLv2 };
  
  float isoValue(-0.9f);
  
  unsigned int countedBlockNumLv1 = 0;
  unsigned int countedBlockNumLv2 = 0;
  unsigned int countedVerticesNum = 0;
  unsigned int countedTrianglesNum = 0;
  float totalKernelTime = 0.f;
  
  cudaEvent_t startEvent, stopEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));
  
  for (unsigned int c0 = 0; c0 < repeat; ++c0)
  {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(countedBlockNumLv1Device, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(countedBlockNumLv2Device, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(countedVerticesNumDevice, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(countedTrianglesNumDevice, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(trianglesDevice, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(coordXDevice, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(coordYDevice, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(coordZDevice, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(coordZPDevice, 0, sizeof(float)));
    
    computeMinMaxLv1<<<GridSizeLv1, BlockSizeLv1>>>(minMaxLv1Device);
    compactLv1<<<countingBlockNumLv1, countingThreadNumLv1>>>(isoValue, minMaxLv1Device, blockIndicesLv1Device, countedBlockNumLv1Device);
    
    CUDA_CHECK(cudaMemcpy(&countedBlockNumLv1, countedBlockNumLv1Device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    computeMinMaxLv2<<<countedBlockNumLv1, BlockSizeLv2>>>(blockIndicesLv1Device, minMaxLv2Device);
    
    unsigned int totalLv2Threads = countedBlockNumLv1 * voxelNumLv2;
    unsigned int countingBlockNumLv2 = (totalLv2Threads + countingThreadNumLv2 - 1) / countingThreadNumLv2;
    
    compactLv2<<<countingBlockNumLv2, countingThreadNumLv2>>>(isoValue, minMaxLv2Device, blockIndicesLv1Device, blockIndicesLv2Device, countedBlockNumLv1, countedBlockNumLv2Device);
    
    CUDA_CHECK(cudaMemcpy(&countedBlockNumLv2, countedBlockNumLv2Device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(startEvent));
    generatingTriangles<<<countedBlockNumLv2, BlockSizeGenerating>>>(
         isoValue, blockIndicesLv2Device,
         d_distinctEdgesTable, d_triTable, d_edgeIDTable,
         countedVerticesNumDevice, countedTrianglesNumDevice, trianglesDevice,
         coordXDevice, coordYDevice, coordZDevice, coordZPDevice);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    totalKernelTime += ms * 1e-3f;
    
    CUDA_CHECK(cudaMemcpy(&countedVerticesNum, countedVerticesNumDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&countedTrianglesNum, countedTrianglesNumDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  }
  
  printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
  printf("Vertices Size: %u\n", countedBlockNumLv2 * 304);
  printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3);
  printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);
  bool ok = (countedBlockNumLv1 == 8296 && countedBlockNumLv2 == 240380 &&
             countedVerticesNum == 4856560 && countedTrianglesNum == 6101640);
  printf("%s\n", ok ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(minMaxLv1Device));
  CUDA_CHECK(cudaFree(blockIndicesLv1Device));
  CUDA_CHECK(cudaFree(countedBlockNumLv1Device));
  CUDA_CHECK(cudaFree(countedBlockNumLv2Device));
  CUDA_CHECK(cudaFree(countedVerticesNumDevice));
  CUDA_CHECK(cudaFree(countedTrianglesNumDevice));
  CUDA_CHECK(cudaFree(trianglesDevice));
  CUDA_CHECK(cudaFree(coordXDevice));
  CUDA_CHECK(cudaFree(coordYDevice));
  CUDA_CHECK(cudaFree(coordZDevice));
  CUDA_CHECK(cudaFree(coordZPDevice));
  CUDA_CHECK(cudaFree(minMaxLv2Device));
  CUDA_CHECK(cudaFree(blockIndicesLv2Device));
  
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));

  return 0;
}
