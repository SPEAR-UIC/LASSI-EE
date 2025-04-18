
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <cuda.h>

#define BLOCK_SIZE 256
#define FORCE_INLINE inline __attribute__((always_inline))

//---------------------------------------------------------------------------
// MurmurHash3 Implementation by Austin Appleby (public domain)
//---------------------------------------------------------------------------

__host__ __device__
FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r)
{
  return (x << r) | (x >> (64 - r));
}

#define BIG_CONSTANT(x) (x##LU)

// Block read function
__host__ __device__
FORCE_INLINE uint64_t getblock64(const uint8_t *p, uint32_t i)
{
  uint64_t s = 0;
  // Unrolled accumulation for 8 bytes
  for (uint32_t n = 0; n < 8; n++) {
    s |= ((uint64_t)p[8 * i + n]) << (n * 8);
  }
  return s;
}

__host__ __device__
FORCE_INLINE uint64_t fmix64(uint64_t k)
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__host__ __device__
void MurmurHash3_x64_128(const void *key, const uint32_t len,
                           const uint32_t seed, void *out)
{
  const uint8_t *data = (const uint8_t *)key;
  const uint32_t nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  for (uint32_t i = 0; i < nblocks; i++)
  {
    uint64_t k1 = getblock64(data, i * 2);
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

  const uint8_t *tail = data + nblocks * 16;
  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch(len & 15)
  {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[9]) << 8;
    case  9: k2 ^= ((uint64_t)tail[8]) << 0;
             k2 *= c2; 
             k2 = rotl64(k2,33); 
             k2 *= c1; 
             h2 ^= k2;
    case  8: k1 ^= ((uint64_t)tail[7]) << 56;
    case  7: k1 ^= ((uint64_t)tail[6]) << 48;
    case  6: k1 ^= ((uint64_t)tail[5]) << 40;
    case  5: k1 ^= ((uint64_t)tail[4]) << 32;
    case  4: k1 ^= ((uint64_t)tail[3]) << 24;
    case  3: k1 ^= ((uint64_t)tail[2]) << 16;
    case  2: k1 ^= ((uint64_t)tail[1]) << 8;
    case  1: k1 ^= ((uint64_t)tail[0]) << 0;
             k1 *= c1; 
             k1 = rotl64(k1,31); 
             k1 *= c2; 
             h1 ^= k1;
  };

  h1 ^= len; 
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t *)out)[0] = h1;
  ((uint64_t *)out)[1] = h2;
}

__global__
void MurmurHash3_x64_128_kernel(const uint8_t *__restrict__ d_keys,
                                  const uint32_t *__restrict__ d_offsets,
                                  const uint32_t *__restrict__ d_keyLengths,
                                  uint64_t *__restrict__ d_out,
                                  const uint32_t numKeys)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numKeys)
    MurmurHash3_x64_128(d_keys + d_offsets[i], d_keyLengths[i], i, d_out + i * 2);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  uint32_t numKeys = atoi(argv[1]);
  uint32_t repeat = atoi(argv[2]);

  srand(3);
  uint32_t i;

  // allocate host arrays for key lengths, key pointers, and hash outputs
  uint32_t* h_keyLengths = (uint32_t*)malloc(sizeof(uint32_t) * numKeys);
  uint8_t** h_keys = (uint8_t**)malloc(sizeof(uint8_t*) * numKeys);
  uint64_t** h_out = (uint64_t**)malloc(sizeof(uint64_t*) * numKeys);

  // Populate keys and compute host hashes for later verification
  for (i = 0; i < numKeys; i++) {
    h_keyLengths[i] = rand() % 10000;
    h_keys[i] = (uint8_t*)malloc(h_keyLengths[i]);
    h_out[i] = (uint64_t*)malloc(2 * sizeof(uint64_t));
    for (uint32_t c = 0; c < h_keyLengths[i]; c++)
      h_keys[i][c] = c % 256;

    MurmurHash3_x64_128(h_keys[i], h_keyLengths[i], i, h_out[i]);
#ifdef DEBUG
    printf("%lu %lu\n", h_out[i][0], h_out[i][1]);
#endif
  }

  // Create 1D flat arrays for device offloading.
  // First the cumulative offsets array (d_offsets): first element is 0.
  // We place the beginning index for each key.
  uint32_t total_length = 0;
  // Use pinned memory (page-locked) for faster transfers from host to device.
  uint32_t* h_offsets;
  cudaMallocHost((void**)&h_offsets, sizeof(uint32_t) * (numKeys + 1));
  h_offsets[0] = 0;
  for (uint32_t i = 0; i < numKeys; i++) {
    total_length += h_keyLengths[i];
    h_offsets[i + 1] = total_length;
  }

  // Allocate pinned memory for the flat key array that will be copied to the device.
  uint8_t* h_flatKeys;
  cudaMallocHost((void**)&h_flatKeys, sizeof(uint8_t) * total_length);
  for (uint32_t i = 0; i < numKeys; i++) {
    memcpy(h_flatKeys + h_offsets[i], h_keys[i], h_keyLengths[i]);
  }

  // Sanity check that the flat key array matches the original key arrays.
  for (uint32_t i = 0; i < numKeys; i++) {
    assert(0 == memcmp(h_flatKeys + h_offsets[i], h_keys[i], h_keyLengths[i]));
  }

  // Allocate device memory for the keys and both offset and keyLengths arrays.
  uint8_t* d_keys;
  cudaMalloc((void**)&d_keys, sizeof(uint8_t) * total_length);
  cudaMemcpy(d_keys, h_flatKeys, sizeof(uint8_t) * total_length, cudaMemcpyHostToDevice);

  uint32_t* d_offsets;
  cudaMalloc((void**)&d_offsets, sizeof(uint32_t) * (numKeys + 1));
  cudaMemcpy(d_offsets, h_offsets, sizeof(uint32_t) * (numKeys + 1), cudaMemcpyHostToDevice);

  uint32_t* d_keyLengths;
  cudaMalloc((void**)&d_keyLengths, sizeof(uint32_t) * numKeys);
  cudaMemcpy(d_keyLengths, h_keyLengths, sizeof(uint32_t) * numKeys, cudaMemcpyHostToDevice);

  uint64_t* d_out;
  cudaMalloc((void**)&d_out, sizeof(uint64_t) * numKeys * 2);

  // Launch the kernel using only as many blocks as needed.
  uint32_t numBlocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridDim(numBlocks);
  dim3 blockDim(BLOCK_SIZE);

  // Synchronize before timing.
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (uint32_t n = 0; n < repeat; n++)
    MurmurHash3_x64_128_kernel<<<gridDim, blockDim>>>(d_keys, d_offsets, d_keyLengths, d_out, numKeys);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  double avgTime_s = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9 / repeat;
  printf("Average kernel execution time %f (s)\n", avgTime_s);

  // Copy hash outputs from device to host.
  uint64_t* h_outFlat = (uint64_t*)malloc(sizeof(uint64_t) * numKeys * 2);
  cudaMemcpy(h_outFlat, d_out, sizeof(uint64_t) * numKeys * 2, cudaMemcpyDeviceToHost);

  // Verify the results
  bool error = false;
  for (uint32_t i = 0; i < numKeys; i++) {
    if (h_outFlat[i * 2] != h_out[i][0] || h_outFlat[i * 2 + 1] != h_out[i][1]) {
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "SUCCESS");

  // Free host memory.
  for (uint32_t i = 0; i < numKeys; i++) {
    free(h_keys[i]);
    free(h_out[i]);
  }
  free(h_keys);
  free(h_out);
  free(h_keyLengths);
  cudaFreeHost(h_flatKeys);
  cudaFreeHost(h_offsets);
  free(h_outFlat);

  // Free device memory.
  cudaFree(d_keys);
  cudaFree(d_offsets);
  cudaFree(d_keyLengths);
  cudaFree(d_out);

  return 0;
}
