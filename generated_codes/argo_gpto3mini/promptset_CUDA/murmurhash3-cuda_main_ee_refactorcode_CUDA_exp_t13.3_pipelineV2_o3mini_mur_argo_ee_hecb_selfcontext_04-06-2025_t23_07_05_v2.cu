
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

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

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

  // Allocate and initialize CPU arrays.
  // For arrays involved in host-device transfers, we use pinned (page-locked) memory.
  uint32_t* h_keyLen = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_keyLen, sizeof(uint32_t) * numKeys, cudaHostAllocDefault));

  uint8_t** h_keys = (uint8_t**)malloc(sizeof(uint8_t*) * numKeys);
  uint64_t** h_out = (uint64_t**)malloc(sizeof(uint64_t*) * numKeys);

  // Compute keys with variable lengths.
  for (i = 0; i < numKeys; i++) {
    h_keyLen[i] = rand() % 10000;  // Length between 0 and 9999.
    h_keys[i] = (uint8_t*)malloc(h_keyLen[i]);
    h_out[i] = (uint64_t*)malloc(2 * sizeof(uint64_t));
    for (uint32_t c = 0; c < h_keyLen[i]; c++) {
      h_keys[i][c] = static_cast<uint8_t>(c % 256);
    }
    // Calculate hash on host for verification.
    MurmurHash3_x64_128(h_keys[i], h_keyLen[i], i, h_out[i]);
#ifdef DEBUG
    printf("Key %u hash: %lu %lu\n", i, h_out[i][0], h_out[i][1]);
#endif
  }

  // Prepare contiguous arrays for device.
  // 1. Compute prefix offsets for keys.
  uint32_t total_length = 0;
  // We allocate (numKeys+1) entries, where the first is 0.
  uint32_t* h_prefix = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_prefix, sizeof(uint32_t) * (numKeys + 1), cudaHostAllocDefault));
  h_prefix[0] = 0;
  for (uint32_t i = 0; i < numKeys; i++) {
    total_length += h_keyLen[i];
    h_prefix[i + 1] = total_length;
  }

  // 2. Create one contiguous buffer for all keys.
  uint8_t* h_allKeys = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_allKeys, sizeof(uint8_t) * total_length, cudaHostAllocDefault));
  for (uint32_t i = 0; i < numKeys; i++) {
    memcpy(h_allKeys + h_prefix[i], h_keys[i], h_keyLen[i]);
  }

  // Sanity check.
  for (uint32_t i = 0; i < numKeys; i++) {
    assert(0 == memcmp(h_allKeys + h_prefix[i], h_keys[i], h_keyLen[i]));
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
    if (h_outKernel[2 * i] != h_out[i][0] || h_outKernel[2 * i + 1] != h_out[i][1]) {
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "SUCCESS");

  //------------------------------------------------------------------------------  
  // Free CPU resources.
  //------------------------------------------------------------------------------
  for (uint32_t i = 0; i < numKeys; i++) {
    free(h_keys[i]);
    free(h_out[i]);
  }
  free(h_keys);
  free(h_out);
  CUDA_CHECK(cudaFreeHost(h_keyLen));
  CUDA_CHECK(cudaFreeHost(h_prefix));
  CUDA_CHECK(cudaFreeHost(h_allKeys));
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
