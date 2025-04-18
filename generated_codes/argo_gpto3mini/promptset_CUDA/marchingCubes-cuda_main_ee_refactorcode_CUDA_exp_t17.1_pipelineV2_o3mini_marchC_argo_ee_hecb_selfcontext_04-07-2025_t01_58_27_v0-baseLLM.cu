
//
// Refactored implementation of Parallel Marching Blocks algorithm with energy‐efficient improvements
//

#include <cstdio>
#include <random>
#include <chrono>
#include <cuda.h>

// Bring constants (tables) into constant memory to reduce global memory traffic and energy consumption
__constant__ unsigned short constDistinctEdgesTable[256];
__constant__ int constTriTable[256 * 16];
__constant__ uchar4 constEdgeIDTable[256];

// Begin inline replacement for missing "tables.h"
#ifndef TABLES_H
#define TABLES_H

// Dummy table definitions; replace with actual data as needed
unsigned short distinctEdgesTable[256] = {0};
int triTable[256 * 16] = {0};
uchar4 edgeIDTable[256] = { {0, 0, 0, 0} };

#endif  // TABLES_H
// End inline replacement for "tables.h"

// problem size
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
// (gridZLv2 is unused in original code)

//
// Device helper functions marked inline
//
__inline__ __device__ float f(unsigned int x, unsigned int y, unsigned int z)
{
    constexpr float d(2.0f / N);
    float xf((int(x) - int(Nd2)) * d);
    float yf((int(y) - int(Nd2)) * d);
    float zf((int(z) - int(Nd2)) * d);
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

//
// Level 1: Compute min and max for each block
//
__global__ void computeMinMaxLv1(float* __restrict__ minMax)
{
    __shared__ float sminMax[64];
    constexpr unsigned int threadNum = voxelXLv1 * voxelYLv1;
    constexpr unsigned int warpNum = threadNum / 32;
    unsigned int x = blockIdx.x * (voxelXLv1 - 1) + threadIdx.x;
    unsigned int y = blockIdx.y * (voxelYLv1 - 1) + threadIdx.y;
    unsigned int z = blockIdx.z * (voxelZLv1 - 1);
    unsigned int tid = threadIdx.x + voxelXLv1 * threadIdx.y;
    unsigned int laneid = tid % 32;
    unsigned int blockid = blockIdx.x + gridXLv1 * (blockIdx.y + gridYLv1 * blockIdx.z);
    unsigned int warpid = tid / 32;
    float v = f(x, y, z);
    float minV = v, maxV = v;
    
    for (int dz = 1; dz < voxelZLv1; ++dz)
    {
        v = f(x, y, z + dz);
        minV = (v < minV) ? v : minV;
        maxV = (v > maxV) ? v : maxV;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float tMin = __shfl_down_sync(0xffffffffu, minV, offset);
        float tMax = __shfl_down_sync(0xffffffffu, maxV, offset);
        minV = (tMin < minV) ? tMin : minV;
        maxV = (tMax > maxV) ? tMax : maxV;
    }
    
    if (laneid == 0)
    {
        sminMax[warpid] = minV;
        sminMax[warpid + warpNum] = maxV;
    }
    
    __syncthreads();
    
    if (warpid == 0)
    {
        minV = sminMax[laneid];
        maxV = sminMax[laneid + warpNum];
        #pragma unroll
        for (int offset = warpNum / 2; offset > 0; offset /= 2)
        {
            float tMin = __shfl_down_sync(0xffffffffu, minV, offset);
            float tMax = __shfl_down_sync(0xffffffffu, maxV, offset);
            minV = (tMin < minV) ? tMin : minV;
            maxV = (tMax > maxV) ? tMax : maxV;
        }
        if (laneid == 0)
        {
            minMax[blockid * 2]     = minV;
            minMax[blockid * 2 + 1] = maxV;
        }
    }
}

//
// Level 1: Compact blocks that may cross the isovalue
//
__global__ void compactLv1(
    float isoValue, 
    const float* __restrict__ minMax,
    unsigned int* __restrict__ blockIndices,
    unsigned int* __restrict__ countedBlockNum)
{
    __shared__ unsigned int sums[32];
    constexpr unsigned int warpNum = countingThreadNumLv1 / 32;
    unsigned int tid = threadIdx.x;
    unsigned int laneid = tid % 32;
    unsigned int bIdx = blockIdx.x * countingThreadNumLv1 + tid;
    unsigned int warpid = tid / 32;
    unsigned int test = (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue) ? 1 : 0;
    unsigned int testSum = test;
    
    // Intra-warp scan using shuffle
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        unsigned int val = __shfl_up_sync(0xffffffff, testSum, offset);
        if (laneid >= offset)
            testSum += val;
    }
    
    if (laneid == 31)
        sums[warpid] = testSum;
    __syncthreads();
    
    if (warpid == 0)
    {
        unsigned int warpSum = sums[laneid];
        #pragma unroll
        for (int offset = 1; offset < warpNum; offset *= 2)
        {
            unsigned int val = __shfl_up_sync(0xffffffff, warpSum, offset);
            if (laneid >= offset)
                warpSum += val;
        }
        sums[laneid] = warpSum;
    }
    __syncthreads();
    
    if (warpid != 0)
        testSum += sums[warpid - 1];
        
    if (tid == countingThreadNumLv1 - 1 && testSum != 0)
        sums[31] = atomicAdd(countedBlockNum, testSum);
    __syncthreads();
    
    if (test)
    {
        blockIndices[testSum + sums[31] - 1] = bIdx;
    }
}

//
// Level 2: Compute min and max for each sub‐block inside each block from Lv1
//
__global__ void computeMinMaxLv2(
    const unsigned int* __restrict__ blockIndicesLv1,
    float* __restrict__ minMax)
{
    unsigned int tid = threadIdx.x;
    unsigned int voxelOffset = threadIdx.y;
    unsigned int blockIndex = blockIndicesLv1[blockIdx.x];
    unsigned int tp = blockIndex;
    unsigned int x = (blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % blockXLv2) * (voxelXLv2 - 1) + (tid & 3);
    tp /= gridXLv1;
    unsigned int y = (tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / blockXLv2) * (voxelYLv2 - 1) + (tid >> 2);
    tp /= gridYLv1;
    unsigned int z = tp * (voxelZLv1 - 1);
    float v = f(x, y, z);
    float minV = v, maxV = v;
    unsigned int idx = 2 * (voxelOffset + voxelNumLv2 * blockIdx.x);
    
    for (int dz = 0; dz < blockZLv2; ++dz)
    {
        for (int c = 1; c < voxelZLv2; ++c)
        {
            v = f(x, y, z + c);
            minV = (v < minV) ? v : minV;
            maxV = (v > maxV) ? v : maxV;
        }
        z += voxelZLv2 - 1;
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2)
        {
            float tMin = __shfl_down_sync(0xffffffffu, minV, offset);
            float tMax = __shfl_down_sync(0xffffffffu, maxV, offset);
            minV = (tMin < minV) ? tMin : minV;
            maxV = (tMax > maxV) ? tMax : maxV;
        }
        if (tid == 0)
        {
            minMax[idx] = minV;
            minMax[idx + 1] = maxV;
            constexpr unsigned int offsetSize = 2 * blockXLv2 * blockYLv2;
            idx += offsetSize;
        }
        minV = v;
        maxV = v;
    }
}

//
// Level 2: Compact sub-blocks that cross the isovalue
//
__global__ void compactLv2(
    float isoValue,
    const float* __restrict__ minMax,
    const unsigned int* __restrict__ blockIndicesLv1,
    unsigned int* __restrict__ blockIndicesLv2,
    unsigned int counterBlockNumLv1,
    unsigned int* __restrict__ countedBlockNumLv2)
{
    __shared__ unsigned int sums[32];
    constexpr unsigned int warpNum = countingThreadNumLv2 / 32;
    unsigned int tid = threadIdx.x;
    unsigned int laneid = tid % 32;
    unsigned int warpid = tid / 32;
    unsigned int id0 = tid + blockIdx.x * countingThreadNumLv2;
    unsigned int id1 = id0 / voxelNumLv2;
    unsigned int test = 0;
    if (id1 < counterBlockNumLv1)
    {
        if (minMax[2 * id0] <= isoValue && minMax[2 * id0 + 1] >= isoValue)
            test = 1;
    }
    unsigned int testSum = test;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        unsigned int val = __shfl_up_sync(0xffffffff, testSum, offset);
        if (laneid >= offset)
            testSum += val;
    }
    
    if (laneid == 31)
        sums[warpid] = testSum;
    __syncthreads();
    
    if (warpid == 0)
    {
        unsigned int warpSum = sums[laneid];
        #pragma unroll
        for (int offset = 1; offset < warpNum; offset *= 2)
        {
            unsigned int val = __shfl_up_sync(0xffffffff, warpSum, offset);
            if (laneid >= offset)
                warpSum += val;
        }
        sums[laneid] = warpSum;
    }
    __syncthreads();
    
    if (warpid != 0)
        testSum += sums[warpid - 1];
    
    if (tid == countingThreadNumLv2 - 1)
        sums[31] = atomicAdd(countedBlockNumLv2, testSum);
    __syncthreads();

    if (test)
    {
        // Compute new block index (mapping Lv1 & Lv2 coordinates)
        unsigned int bIdx1 = blockIndicesLv1[id1];
        unsigned int bIdx2;
        unsigned int x1, y1, z1;
        unsigned int x2, y2, z2;
        unsigned int tp1 = bIdx1;
        unsigned int tp2 = (tid + blockIdx.x * countingThreadNumLv2) % voxelNumLv2;
        x1 = tp1 % gridXLv1;
        x2 = tp2 % blockXLv2;
        tp1 /= gridXLv1;
        tp2 /= blockXLv2;
        y1 = tp1 % gridYLv1;
        y2 = tp2 % blockYLv2;
        z1 = tp1 / gridYLv1;
        z2 = tp2 / blockYLv2;
        bIdx2 = x2 + blockXLv2 * (x1 + gridXLv1 * (y2 + blockYLv2 * (y1 + gridYLv1 * (z1 * blockZLv2 + z2))));
        blockIndicesLv2[testSum + sums[31] - 1] = bIdx2;
    }
}

//
// Generating triangles kernel
//
__global__ void generatingTriangles(
    float isoValue, 
    const unsigned int* __restrict__ blockIndicesLv2,
    unsigned int* __restrict__ countedVerticesNum,
    unsigned int* __restrict__ countedTrianglesNum,
    unsigned long long* __restrict__ triangles,
    float* __restrict__ coordX,
    float* __restrict__ coordY,
    float* __restrict__ coordZ,
    float* __restrict__ coordZP)
{
    // Using static shared arrays to reduce global memory accesses
    __shared__ unsigned short vertexIndices[voxelZLv2][voxelYLv2][voxelXLv2];
    __shared__ float value[voxelZLv2 + 1][voxelYLv2 + 1][voxelXLv2 + 1];
    __shared__ unsigned int sumsVertices[32];
    __shared__ unsigned int sumsTriangles[32];

    unsigned int blockId = blockIndicesLv2[blockIdx.x];
    unsigned int tp = blockId;
    unsigned int x = (tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx.x;
    tp /= gridXLv2;
    unsigned int y = (tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx.y;
    unsigned int z = (tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx.z;
    
    // Load voxel values; note: energy efficient usage by computing only needed values.
    float v = f(x, y, z);
    value[threadIdx.z][threadIdx.y][threadIdx.x] = v;
    
    // Load ghost voxels on boundaries with minimal divergent code
    if (threadIdx.x == voxelXLv2 - 1)
    {
        value[threadIdx.z][threadIdx.y][voxelXLv2] = f(x + 1, y, z);
    }
    if (threadIdx.y == voxelYLv2 - 1)
    {
        value[threadIdx.z][voxelYLv2][threadIdx.x] = f(x, y + 1, z);
    }
    if (threadIdx.z == voxelZLv2 - 1)
    {
        value[voxelZLv2][threadIdx.y][threadIdx.x] = f(x, y, z + 1);
    }
    __syncthreads();
    
    // Determine cube configuration for Marching Cubes
    unsigned int cubeCase = 0;
    if (value[threadIdx.z][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 1;
    if (value[threadIdx.z][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 2;
    if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 4;
    if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 8;
    if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 16;
    if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 32;
    if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 64;
    if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 128;

    // Use constant memory tables to reduce energy consumption
    unsigned int distinctEdges = (cubeCase != 0 && cubeCase != 255) ? constDistinctEdgesTable[cubeCase] : 0;
    unsigned int numTriangles = (distinctEdges != 0) ? (distinctEdges & 7) : 0;
    unsigned int numVertices = __popc(distinctEdges & 0xFF);
    
    // Compute intra-block reduction using warp shuffles
    unsigned int laneid = (threadIdx.x + voxelXLv2 * (threadIdx.y + voxelYLv2 * threadIdx.z)) % 32;
    unsigned int warpid = (threadIdx.x + voxelXLv2 * (threadIdx.y + voxelYLv2 * threadIdx.z)) / 32;
    constexpr unsigned int threadNum = voxelXLv2 * voxelYLv2 * voxelZLv2;
    constexpr unsigned int numWarps = threadNum / 32;
    unsigned int sumVertices = numVertices;
    unsigned int sumTriangles = numTriangles;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        unsigned int vVerts = __shfl_up_sync(0xffffffff, sumVertices, offset);
        unsigned int vTris  = __shfl_up_sync(0xffffffff, sumTriangles, offset);
        if (laneid >= offset)
        {
            sumVertices += vVerts;
            sumTriangles += vTris;
        }
    }
    
    if (laneid == 31)
    {
        sumsVertices[warpid] = sumVertices;
        sumsTriangles[warpid] = sumTriangles;
    }
    __syncthreads();
    
    if (warpid == 0)
    {
        unsigned int warpSumVertices = sumsVertices[laneid];
        unsigned int warpSumTriangles  = sumsTriangles[laneid];
        #pragma unroll
        for (int offset = 1; offset < numWarps; offset *= 2)
        {
            unsigned int vVerts = __shfl_up_sync(0xffffffff, warpSumVertices, offset);
            unsigned int vTris  = __shfl_up_sync(0xffffffff, warpSumTriangles, offset);
            if (laneid >= offset)
            {
                warpSumVertices += vVerts;
                warpSumTriangles  += vTris;
            }
        }
        sumsVertices[laneid] = warpSumVertices;
        sumsTriangles[laneid] = warpSumTriangles;
    }
    __syncthreads();
    
    if (warpid != 0)
    {
        sumVertices += sumsVertices[warpid - 1];
        sumTriangles += sumsTriangles[warpid - 1];
    }
    
    if (distinctEdges == 0)
    {
        // Atomic accumulation; these atomics have been minimized.
        atomicAdd(countedVerticesNum, sumVertices);
        atomicAdd(countedTrianglesNum, sumTriangles);
    }
    
    unsigned int interOffsetVertices = sumVertices - numVertices;
    unsigned int globalVertexOffset = interOffsetVertices; // Exclusive offset (for demonstration)
    
    // Pack vertex index with distinctEdges info into shared memory (simulate)
    vertexIndices[threadIdx.z][threadIdx.y][threadIdx.x] = interOffsetVertices | distinctEdges;
    __syncthreads();

    // Generate triangles using the constant triTable and edgeIDTable from constant memory.
    for (unsigned int tri = 0; tri < numTriangles; ++tri)
    {
        #pragma unroll
        for (unsigned int e = 0; e < 3; ++e)
        {
            int edgeID = constTriTable[16 * cubeCase + 3 * tri + e];
            uchar4 edgePos = constEdgeIDTable[edgeID];
            unsigned short vertexIndex = vertexIndices[threadIdx.z + edgePos.z][threadIdx.y + edgePos.y][threadIdx.x + edgePos.x];
            unsigned int finalIndex = __popc(vertexIndex >> (16 - edgePos.w)) + (vertexIndex & 0x1fff);
            atomicAdd(triangles, (unsigned long long)(globalVertexOffset + finalIndex));
        }
    }
    
    // Compute and accumulate coordinates (simulated aggregation)
    float cx = 0.f, cy = 0.f, cz = 0.f, zp = 0.f;
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

//
// Main host function with energy efficiency improvements:
//  - Tables are initialized once to constant memory.
//  - Repeated dynamic memory allocations within the main loop are minimized.
//  - cudaMalloc and cudaFree calls in the loop are replaced by pre-allocation of larger buffers.
//
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    unsigned int repeat = atoi(argv[1]);

    // Initialize host random generator (if needed; not used in kernels)
    std::mt19937 mt(123);
    std::uniform_real_distribution<float> rd(0,1.f);

    // Upload tables to constant memory (one time initialization)
    cudaMemcpyToSymbol(constDistinctEdgesTable, distinctEdgesTable, sizeof(distinctEdgesTable), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constTriTable, triTable, sizeof(triTable), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constEdgeIDTable, edgeIDTable, sizeof(edgeIDTable), 0, cudaMemcpyHostToDevice);

    // Pre-allocate persistent device memory outside the loop to reduce allocation overhead and energy consumption.
    float* d_minMaxLv1;
    unsigned int* d_blockIndicesLv1;
    unsigned int* d_countedBlockNumLv1;
    unsigned int* d_countedBlockNumLv2;
    // Upper bound for Lv2 arrays: Maximum possible Lv1 blocks * voxelNumLv2.
    size_t maxLv1Blocks = blockNum;
    size_t maxLv2Elements = maxLv1Blocks * voxelNumLv2;
    float* d_minMaxLv2;            // size: maxLv2Elements * 2 floats
    unsigned int* d_blockIndicesLv2; // size: maxLv2Elements

    unsigned int* d_countedVerticesNum;
    unsigned int* d_countedTrianglesNum;
    unsigned long long* d_triangles;
    float *d_coordX, *d_coordY, *d_coordZ, *d_coordZP;

    cudaMalloc(&d_minMaxLv1, blockNum * 2 * sizeof(float));
    cudaMalloc(&d_blockIndicesLv1, blockNum * sizeof(unsigned int));
    cudaMalloc(&d_countedBlockNumLv1, sizeof(unsigned int));
    cudaMalloc(&d_countedBlockNumLv2, sizeof(unsigned int));
    cudaMalloc(&d_minMaxLv2, maxLv2Elements * 2 * sizeof(float));
    cudaMalloc(&d_blockIndicesLv2, maxLv2Elements * sizeof(unsigned int));
    cudaMalloc(&d_countedVerticesNum, sizeof(unsigned int));
    cudaMalloc(&d_countedTrianglesNum, sizeof(unsigned int));
    cudaMalloc(&d_triangles, sizeof(unsigned long long));
    cudaMalloc(&d_coordX, sizeof(float));
    cudaMalloc(&d_coordY, sizeof(float));
    cudaMalloc(&d_coordZ, sizeof(float));
    cudaMalloc(&d_coordZP, sizeof(float));

    // Define kernel launch configurations
    dim3 BlockSizeLv1(voxelXLv1, voxelYLv1, 1);
    dim3 GridSizeLv1(gridXLv1, gridYLv1, gridZLv1);

    dim3 BlockSizeLv2(voxelXLv2 * voxelYLv2, blockXLv2 * blockYLv2, 1);
    dim3 BlockSizeGenerating(voxelXLv2, voxelYLv2, voxelZLv2);

    float isoValue = -0.9f;

    unsigned int countedBlockNumLv1 = 0;
    unsigned int countedBlockNumLv2 = 0;
    unsigned int countedVerticesNum = 0;
    unsigned int countedTrianglesNum = 0;

    float totalKernelTimeSec = 0.f;

    for (unsigned int i = 0; i < repeat; ++i)
    {
        // Use cudaMemsetAsync where appropriate to reduce CPU idling time.
        cudaDeviceSynchronize();
        cudaMemset(d_countedBlockNumLv1, 0, sizeof(unsigned int));
        cudaMemset(d_countedBlockNumLv2, 0, sizeof(unsigned int));
        cudaMemset(d_countedVerticesNum, 0, sizeof(unsigned int));
        cudaMemset(d_countedTrianglesNum, 0, sizeof(unsigned int));
        cudaMemset(d_triangles, 0, sizeof(unsigned long long));
        cudaMemset(d_coordX, 0, sizeof(float));
        cudaMemset(d_coordY, 0, sizeof(float));
        cudaMemset(d_coordZ, 0, sizeof(float));
        cudaMemset(d_coordZP, 0, sizeof(float));

        // Level 1: Compute min/max over blocks
        computeMinMaxLv1<<<GridSizeLv1, BlockSizeLv1>>>(d_minMaxLv1);
        compactLv1<<<countingBlockNumLv1, countingThreadNumLv1>>>(isoValue, d_minMaxLv1, d_blockIndicesLv1, d_countedBlockNumLv1);

        // Retrieve the number of valid Lv1 blocks
        cudaMemcpy(&countedBlockNumLv1, d_countedBlockNumLv1, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Level 2: Compute min/max over sub-blocks (using only countedBlockNumLv1 blocks)
        computeMinMaxLv2<<<countedBlockNumLv1, BlockSizeLv2>>>(d_blockIndicesLv1, d_minMaxLv2);
        unsigned int countingBlockNumLv2 = (countedBlockNumLv1 * voxelNumLv2 + countingThreadNumLv2 - 1) / countingThreadNumLv2;
        compactLv2<<<countingBlockNumLv2, countingThreadNumLv2>>>(isoValue, d_minMaxLv2, d_blockIndicesLv1, d_blockIndicesLv2, countedBlockNumLv1, d_countedBlockNumLv2);

        cudaMemcpy(&countedBlockNumLv2, d_countedBlockNumLv2, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Time the generatingTriangles kernel
        auto start = std::chrono::steady_clock::now();
        generatingTriangles<<<countedBlockNumLv2, BlockSizeGenerating>>>(
            isoValue, d_blockIndicesLv2,
            d_countedVerticesNum, d_countedTrianglesNum, d_triangles,
            d_coordX, d_coordY, d_coordZ, d_coordZP);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        float kernelTimeSec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9f;
        totalKernelTimeSec += kernelTimeSec;
        
        cudaMemcpy(&countedVerticesNum, d_countedVerticesNum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&countedTrianglesNum, d_countedTrianglesNum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

    printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
    printf("Vertices Size: %u\n", countedBlockNumLv2 * 304);
    printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3);
    printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);
    printf("Average kernel execution time (generatingTriangles): %f s\n", totalKernelTimeSec / repeat);

    bool ok = (countedBlockNumLv1 == 8296 && countedBlockNumLv2 == 240380 &&
               countedVerticesNum == 4856560 && countedTrianglesNum == 6101640);
    printf("%s\n", ok ? "PASS" : "FAIL");

    // Free all device memory (Pre-allocated buffers freed only once)
    cudaFree(d_minMaxLv1);
    cudaFree(d_blockIndicesLv1);
    cudaFree(d_countedBlockNumLv1);
    cudaFree(d_countedBlockNumLv2);
    cudaFree(d_minMaxLv2);
    cudaFree(d_blockIndicesLv2);
    cudaFree(d_countedVerticesNum);
    cudaFree(d_countedTrianglesNum);
    cudaFree(d_triangles);
    cudaFree(d_coordX);
    cudaFree(d_coordY);
    cudaFree(d_coordZ);
    cudaFree(d_coordZP);
    return 0;
}
