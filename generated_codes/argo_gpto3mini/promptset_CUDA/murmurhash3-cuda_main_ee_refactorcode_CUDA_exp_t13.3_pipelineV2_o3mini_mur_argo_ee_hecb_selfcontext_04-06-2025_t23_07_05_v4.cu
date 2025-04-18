
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <cuda.h>

//------------------------------------------------------------------------------
// Energy-efficient refactored version of MurmurHash3_x64_128 CUDA C++ code
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call,           \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256

#define FORCE_INLINE inline __attribute__((always_inline))

// Bit-rotation
__host__ __device__
FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

#define BIG_CONSTANT(x) (x##LU)

//------------------------------------------------------------------------------
// Declare read-only constants in device constant memory.
// These are used only when the code is compiled for the device.
__device__ __constant__ uint64_t devC1;
__device__ __constant__ uint64_t devC2;

// Helper functions to retrieve constants.
// When compiled for the device, they return the constant from device memory,
// while on the host they return the compile-time literal.
__host__ __device__ FORCE_INLINE uint64_t getC1() {
#ifndef __CUDA_ARCH__
    return BIG_CONSTANT(0x87c37b91114253d5);
#else
    return devC1;
#endif
}

__host__ __device__ FORCE_INLINE uint64_t getC2() {
#ifndef __CUDA_ARCH__
    return BIG_CONSTANT(0x4cf5ad432745937f);
#else
    return devC2;
#endif
}

// Read a block of 64 bits from an unaligned byte pointer.
__host__ __device__
FORCE_INLINE uint64_t getblock64(const uint8_t* p, uint32_t i) {
  uint64_t s = 0;
  #pragma unroll
  for (uint32_t n = 0; n < 8; n++) {
    s |= ((uint64_t)p[8 * i + n]) << (n * 8);
  }
  return s;
}

// Finalization mix - force all bits of a hash block to avalanche.
__host__ __device__
FORCE_INLINE uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__host__ __device__
void MurmurHash3_x64_128(const void* key, const uint32_t len,
                         const uint32_t seed, void* out)
{
  const uint8_t* data = reinterpret_cast<const uint8_t*>(key);
  const uint32_t nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  // Get read-only constants (from constant memory on device)
  uint64_t c1 = getC1();
  uint64_t c2 = getC2();

  // Body
  for (uint32_t i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock64(data, i * 2 + 0);
    uint64_t k2 = getblock64(data, i * 2 + 1);

    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  // Tail
  const uint8_t* tail = data + nblocks * 16;
  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (len & 15) {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[9])  << 8;
    case 9:  k2 ^= ((uint64_t)tail[8])  << 0;
             k2 *= c2;
             k2 = rotl64(k2, 33);
             k2 *= c1;
             h2 ^= k2;
    case 8:  k1 ^= ((uint64_t)tail[7])  << 56;
    case 7:  k1 ^= ((uint64_t)tail[6])  << 48;
    case 6:  k1 ^= ((uint64_t)tail[5])  << 40;
    case 5:  k1 ^= ((uint64_t)tail[4])  << 32;
    case 4:  k1 ^= ((uint64_t)tail[3])  << 24;
    case 3:  k1 ^= ((uint64_t)tail[2])  << 16;
    case 2:  k1 ^= ((uint64_t)tail[1])  << 8;
    case 1:  k1 ^= ((uint64_t)tail[0])  << 0;
             k1 *= c1;
             k1 = rotl64(k1, 31);
             k1 *= c2;
             h1 ^= k1;
  };

  // Finalization
  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  reinterpret_cast<uint64_t*>(out)[0] = h1;
  reinterpret_cast<uint64_t*>(out)[1] = h2;
}

// Kernel: each thread hashes one key. The pointers marked __restrict__ help the compiler optimize memory accesses.
__global__
void MurmurHash3_x64_128_kernel(
    const uint8_t* __restrict__ d_keys,
    const uint32_t* __restrict__ d_prefix,   // prefix offsets for each key
    const uint32_t* __restrict__ d_keyLen,     // individual key lengths
    uint64_t* __restrict__ d_out,
    const uint32_t numKeys)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numKeys) {
    // Each key begins at d_keys + d_prefix[i] and has length d_keyLen[i]
    MurmurHash3_x64_128(d_keys + d_prefix[i], d_keyLen[i], i, d_out + i * 2);
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return EXIT_FAILURE;
  }
  uint32_t numKeys = static_cast<uint32_t>(atoi(argv[1]));
  uint32_t repeat  = static_cast<uint32_t>(atoi(argv[2]));

  srand(3);
  uint32_t i;

  // Allocate pinned host memory for key lengths.
  uint32_t* h_keyLen = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_keyLen, sizeof(uint32_t) * numKeys, cudaHostAllocDefault));

  // First, generate key lengths.
  uint32_t total_length = 0;
  for (i = 0; i < numKeys; i++) {
    h_keyLen[i] = rand() % 10000;  // Length between 0 and 9999.
    total_length += h_keyLen[i];
  }

  // Allocate pinned host memory for prefix offsets (numKeys+1 entries).
  uint32_t* h_prefix = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_prefix, sizeof(uint32_t) * (numKeys + 1), cudaHostAllocDefault));
  h_prefix[0] = 0;
  for (i = 0; i < numKeys; i++) {
    h_prefix[i + 1] = h_prefix[i] + h_keyLen[i];
  }

  // Allocate one contiguous pinned buffer for all keys.
  uint8_t* h_allKeys = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_allKeys, sizeof(uint8_t) * total_length, cudaHostAllocDefault));

  // Fill the contiguous key buffer with generated keys.
  for (i = 0; i < numKeys; i++) {
    uint32_t keyLen = h_keyLen[i];
    uint8_t* keyPtr = h_allKeys + h_prefix[i];
    for (uint32_t c = 0; c < keyLen; c++) {
      keyPtr[c] = static_cast<uint8_t>(c % 256);
    }
  }

  // Compute host hashes for verification.
  // Allocate pinned host memory for host computed hashes (2 uint64_t per key).
  uint64_t* h_hostHash = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_hostHash, sizeof(uint64_t) * numKeys * 2, cudaHostAllocDefault));

  for (i = 0; i < numKeys; i++) {
    MurmurHash3_x64_128(h_allKeys + h_prefix[i], h_keyLen[i], i, h_hostHash + i * 2);
#ifdef DEBUG
    printf("Key %u hash: %lu %lu\n", i, h_hostHash[i * 2], h_hostHash[i * 2 + 1]);
#endif
  }

  // Allocate output array on host for kernel result.
  uint64_t* h_outKernel = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_outKernel, sizeof(uint64_t) * (numKeys * 2), cudaHostAllocDefault));

  //------------------------------------------------------------------------------
  // Device memory allocation
  //------------------------------------------------------------------------------
  uint8_t* d_keys = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_keys, sizeof(uint8_t) * total_length));
  uint32_t* d_prefix = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_prefix, sizeof(uint32_t) * (numKeys + 1)));
  uint32_t* d_keyLen = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_keyLen, sizeof(uint32_t) * numKeys));
  uint64_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(uint64_t) * numKeys * 2));

  // Create a CUDA stream for asynchronous operations.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Copy read-only constants to device constant memory.
  uint64_t hC1 = BIG_CONSTANT(0x87c37b91114253d5);
  uint64_t hC2 = BIG_CONSTANT(0x4cf5ad432745937f);
  CUDA_CHECK(cudaMemcpyToSymbol(devC1, &hC1, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(devC2, &hC2, sizeof(uint64_t)));

  // Asynchronously copy host data to the device.
  CUDA_CHECK(cudaMemcpyAsync(d_keys, h_allKeys, sizeof(uint8_t) * total_length,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_prefix, h_prefix, sizeof(uint32_t) * (numKeys + 1),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_keyLen, h_keyLen, sizeof(uint32_t) * numKeys,
                             cudaMemcpyHostToDevice, stream));

  // Set cache configuration (prefer L1) to potentially reduce energy by better caching.
  CUDA_CHECK(cudaFuncSetCacheConfig(MurmurHash3_x64_128_kernel, cudaFuncCachePreferL1));

  // Define grid and block dimensions.
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE);

  //------------------------------------------------------------------------------
  // Use CUDA events for kernel timing measurement.
  //------------------------------------------------------------------------------
  cudaEvent_t startEvent, stopEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  // Ensure that previous asynchronous copies are completed.
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Record start event in our stream.
  CUDA_CHECK(cudaEventRecord(startEvent, stream));

  // Launch kernel repeatedly on the same stream.
  for (uint32_t n = 0; n < repeat; n++) {
    MurmurHash3_x64_128_kernel<<<gridDim, blockDim, 0, stream>>>(d_keys, d_prefix, d_keyLen, d_out, numKeys);
  }

  // Record stop event in the stream.
  CUDA_CHECK(cudaEventRecord(stopEvent, stream));
  CUDA_CHECK(cudaEventSynchronize(stopEvent));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent));

  printf("Average kernel execution time: %f s\n", (elapsed_ms * 1e-3f) / repeat);

  // Asynchronously copy the results back to the host.
  CUDA_CHECK(cudaMemcpyAsync(h_outKernel, d_out, sizeof(uint64_t) * numKeys * 2,
                             cudaMemcpyDeviceToHost, stream));
  // Ensure that all asynchronous operations complete.
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Verification of kernel computed results vs host computed ones.
  bool error = false;
  for (uint32_t i = 0; i < numKeys; i++) {
    if (h_outKernel[2 * i] != h_hostHash[2 * i] ||
        h_outKernel[2 * i + 1] != h_hostHash[2 * i + 1]) {
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "SUCCESS");

  //------------------------------------------------------------------------------
  // Free CPU resources.
  //------------------------------------------------------------------------------
  CUDA_CHECK(cudaFreeHost(h_keyLen));
  CUDA_CHECK(cudaFreeHost(h_prefix));
  CUDA_CHECK(cudaFreeHost(h_allKeys));
  CUDA_CHECK(cudaFreeHost(h_hostHash));
  CUDA_CHECK(cudaFreeHost(h_outKernel));

  //------------------------------------------------------------------------------
  // Free Device resources.
  //------------------------------------------------------------------------------
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_prefix));
  CUDA_CHECK(cudaFree(d_keyLen));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}
