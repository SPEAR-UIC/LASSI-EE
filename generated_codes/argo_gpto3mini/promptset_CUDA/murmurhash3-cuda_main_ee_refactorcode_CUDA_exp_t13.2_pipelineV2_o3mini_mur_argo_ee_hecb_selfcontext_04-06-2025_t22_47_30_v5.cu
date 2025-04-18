
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h> // for cudaMallocHost and cudaFreeHost

#define BLOCK_SIZE 256

#define FORCE_INLINE inline __attribute__((always_inline))

__host__ __device__
inline uint64_t rotl64 ( uint64_t x, int8_t r )
{
  return (x << r) | (x >> (64 - r));
}

#define BIG_CONSTANT(x) (x##LU)

// Optimized block read function: if the pointer p is aligned to 8 bytes,
// load the 64-bit block directly; otherwise, fall back to the byte-by-byte loop.
__host__ __device__
FORCE_INLINE uint64_t getblock64 ( const uint8_t * p, uint32_t i )
{
  // Check for 8-byte alignment (works on both host and device)
  if ((((uintptr_t)p) & 0x7) == 0) {
    // Directly load the 64-bit value
    return ((const uint64_t*)p)[i];
  } else {
    uint64_t s = 0;
    for (uint32_t n = 0; n < 8; n++) {
      s |= ((uint64_t)p[8*i+n] << (n*8));
    }
    return s;
  }
}

// Finalization mix - force all bits of a hash block to avalanche
__host__ __device__
FORCE_INLINE uint64_t fmix64 ( uint64_t k )
{
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

__host__ __device__ 
void MurmurHash3_x64_128 (const void * key, const uint32_t len,
                          const uint32_t seed, void * out)
{
  const uint8_t * data = (const uint8_t*)key;
  const uint32_t nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
  const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

  for(uint32_t i = 0; i < nblocks; i++)
  {
    uint64_t k1 = getblock64(data,i*2+0);
    uint64_t k2 = getblock64(data,i*2+1);

    k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;

    h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;

    k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

    h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
  }

  const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch(len & 15)
  {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[ 9]) << 8;
    case  9: k2 ^= ((uint64_t)tail[ 8]) << 0;
            k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;
    case  8: k1 ^= ((uint64_t)tail[ 7]) << 56;
    case  7: k1 ^= ((uint64_t)tail[ 6]) << 48;
    case  6: k1 ^= ((uint64_t)tail[ 5]) << 40;
    case  5: k1 ^= ((uint64_t)tail[ 4]) << 32;
    case  4: k1 ^= ((uint64_t)tail[ 3]) << 24;
    case  3: k1 ^= ((uint64_t)tail[ 2]) << 16;
    case  2: k1 ^= ((uint64_t)tail[ 1]) << 8;
    case  1: k1 ^= ((uint64_t)tail[ 0]) << 0;
            k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
  };

  h1 ^= len; h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix64(h1);
  h2 = fmix64(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t*)out)[0] = h1;
  ((uint64_t*)out)[1] = h2;
}

__global__
void MurmurHash3_x64_128_kernel (
   const uint8_t *__restrict__ d_keys,
   const uint32_t *__restrict__ d_length,
   const uint32_t *__restrict__ length, 
         uint64_t *__restrict__ d_out,
   const uint32_t numKeys )
{
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numKeys) 
    MurmurHash3_x64_128 (d_keys+d_length[i], length[i], i, d_out+i*2);
}

int main(int argc, char** argv) 
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  } 
  uint32_t numKeys = atoi(argv[1]);
  uint32_t repeat = atoi(argv[2]);

  srand(3);
  uint32_t i;

  // Allocate pinned memory for key lengths
  uint32_t* length;
  cudaMallocHost((void**)&length, sizeof(uint32_t) * numKeys);

  // First, generate key lengths and compute per-key offsets 
  // (we use offsets as the starting position for each key in our contiguous buffer)
  uint32_t* offsets;
  cudaMallocHost((void**)&offsets, sizeof(uint32_t) * (numKeys+1));
  offsets[0] = 0;
  for (i = 0; i < numKeys; i++) {
    length[i] = rand() % 10000;
    offsets[i+1] = offsets[i] + length[i];
  }
  uint32_t total_length = offsets[numKeys];

  // Allocate a single contiguous pinned memory buffer for all keys
  uint8_t* keys_all;
  cudaMallocHost((void**)&keys_all, sizeof(uint8_t) * total_length);

  // Allocate a single contiguous output array for CPU-side hashing results
  uint64_t* out_all;
  cudaMallocHost((void**)&out_all, sizeof(uint64_t) * 2 * numKeys);

  // Fill the keys_all buffer with key data and compute CPU-side hash for verification
  // Each key's data is simply c % 256 for each byte.
  for (i = 0; i < numKeys; i++) {
    for (uint32_t c = 0; c < length[i]; c++) {
      keys_all[offsets[i] + c] = c % 256;
    }
    // Compute CPU side hash for this key using the contiguous keys_all buffer.
    MurmurHash3_x64_128(keys_all + offsets[i], length[i], i, out_all + 2*i);
#ifdef DEBUG
    printf("%lu %lu\n", out_all[2*i], out_all[2*i+1]);
#endif
  }

  // Create the 1D data arrays for device offloading  
  uint64_t* d_out;
  cudaMallocHost((void**)&d_out, sizeof(uint64_t) * 2 * numKeys);
  
  // d_length will hold the per-key offsets for the device
  uint32_t* d_length;
  cudaMallocHost((void**)&d_length, sizeof(uint32_t) * (numKeys+1));
  // Copy our computed offsets to d_length
  memcpy(d_length, offsets, sizeof(uint32_t)*(numKeys+1));

  // Sanity check: verify that keys_all contains the proper data
  for (i = 0; i < numKeys; i++) {
    for (uint32_t c = 0; c < length[i]; c++) {
      assert(keys_all[offsets[i] + c] == (c % 256));
    }
  }

  // Allocate device memory and copy the contiguous key data and offsets
  uint8_t* dev_keys;
  cudaMalloc((void**)&dev_keys, sizeof(uint8_t)*total_length);
  cudaMemcpy(dev_keys, keys_all, sizeof(uint8_t)*total_length, cudaMemcpyHostToDevice);

  uint32_t* dev_length;
  cudaMalloc((void**)&dev_length, sizeof(uint32_t)*(numKeys+1));
  cudaMemcpy(dev_length, d_length, sizeof(uint32_t)*(numKeys+1), cudaMemcpyHostToDevice);

  // Allocate and copy key lengths to device
  uint32_t* key_length;
  cudaMalloc((void**)&key_length, sizeof(uint32_t)*(numKeys));
  cudaMemcpy(key_length, length, sizeof(uint32_t)*(numKeys), cudaMemcpyHostToDevice);

  // Allocate output array on device 
  uint64_t* dev_out;
  cudaMalloc((void**)&dev_out, sizeof(uint64_t)*(numKeys*2));

  dim3 gridDim ((numKeys+BLOCK_SIZE-1)/BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  // Create a CUDA stream for asynchronous operations
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Remove idle sync before timing to let asynchronous operations overlap
  auto start = std::chrono::steady_clock::now();

  for (uint32_t n = 0; n < repeat; n++)  {
    MurmurHash3_x64_128_kernel<<<gridDim, blockDim, 0, stream>>>(
      dev_keys, dev_length, key_length, dev_out, numKeys);
  }

  // Synchronize on the stream to wait for kernel execution to finish
  cudaStreamSynchronize(stream);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  // Asynchronously copy results back from device to host
  cudaMemcpyAsync(d_out, dev_out, sizeof(uint64_t)*(numKeys*2), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Verify: compare device computed results with the CPU hash results in out_all
  bool error = false;
  for (uint32_t i = 0; i < numKeys; i++) {
    if (d_out[2*i] != out_all[2*i] ||  d_out[2*i+1] != out_all[2*i+1]) {
      error = true;
      break;
    }
  }
  if (error) printf("FAIL\n");
  else printf("SUCCESS\n");

  // Free allocated host memory
  cudaFreeHost(length);
  cudaFreeHost(offsets);
  cudaFreeHost(keys_all);
  cudaFreeHost(out_all);
  cudaFreeHost(d_out);
  cudaFreeHost(d_length);

  // Free device memory
  cudaFree(dev_keys);
  cudaFree(dev_out);
  cudaFree(dev_length);
  cudaFree(key_length);

  // Destroy the CUDA stream
  cudaStreamDestroy(stream);
  return 0;
}
