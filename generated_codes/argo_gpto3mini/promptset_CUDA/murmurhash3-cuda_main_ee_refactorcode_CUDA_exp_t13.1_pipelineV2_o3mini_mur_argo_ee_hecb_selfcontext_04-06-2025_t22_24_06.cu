
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda.h>

#define BLOCK_SIZE 256
#define FORCE_INLINE inline __attribute__((always_inline))

// Device constant for repeat count (no longer used in kernel body)
__constant__ uint32_t gRepeat;

// Rotate left (for both device and host)
__host__ __device__
FORCE_INLINE uint64_t rotl64(uint64_t x, int8_t r)
{
    return (x << r) | (x >> (64 - r));
}

#define BIG_CONSTANT(x) (x##LU)

// Optimized: Read 64 bits from pointer p at block index i by performing
// a safe unaligned load using bitwise shifts (works on both CUDA devices and x86 hosts)
__host__ __device__
FORCE_INLINE uint64_t getblock64(const uint8_t *p, uint32_t i)
{
    const uint8_t* ptr = p + 8 * i;
    return ((uint64_t)ptr[0])       |
           ((uint64_t)ptr[1] << 8)  |
           ((uint64_t)ptr[2] << 16) |
           ((uint64_t)ptr[3] << 24) |
           ((uint64_t)ptr[4] << 32) |
           ((uint64_t)ptr[5] << 40) |
           ((uint64_t)ptr[6] << 48) |
           ((uint64_t)ptr[7] << 56);
}

// Finalization mix - force all bits of a hash block to avalanche
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

// Compute the 128-bit MurmurHash3 x64 hash (host and device)
__host__ __device__
void MurmurHash3_x64_128(const void *key, const uint32_t len,
                           const uint32_t seed, void *out)
{
    const uint8_t *data = (const uint8_t*)key;
    const uint32_t nblocks = len / 16;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    // Constants (same for host and device)
    const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
    const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

    // Body
    for (uint32_t i = 0; i < nblocks; i++)
    {
        uint64_t k1 = getblock64(data, i * 2 + 0);
        uint64_t k2 = getblock64(data, i * 2 + 1);

        k1 *= c1; k1  = rotl64(k1, 31); k1 *= c2; h1 ^= k1;

        h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

        k2 *= c2; k2  = rotl64(k2, 33); k2 *= c1; h2 ^= k2;

        h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }

    // Tail
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
                 k2 *= c2; k2  = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
                 // When tail length is >= 9, k1 is computed from the first 8 bytes.
                 // Fall through to compute k1.
        case  8: k1 ^= ((uint64_t)tail[7]) << 56;
        case  7: k1 ^= ((uint64_t)tail[6]) << 48;
        case  6: k1 ^= ((uint64_t)tail[5]) << 40;
        case  5: k1 ^= ((uint64_t)tail[4]) << 32;
        case  4: k1 ^= ((uint64_t)tail[3]) << 24;
        case  3: k1 ^= ((uint64_t)tail[2]) << 16;
        case  2: k1 ^= ((uint64_t)tail[1]) << 8;
        case  1: k1 ^= ((uint64_t)tail[0]) << 0;
                 k1 *= c1; k1  = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    }

    // Finalization mix
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

// -------------------------------------------------------------------------
// Kernel: Each thread computes the hash for one key.
// (Refactored: removed the inner repeat loop. Instead, the host will launch the kernel repeatedly.)
// Note: d_offsets holds cumulative byte offsets for each key.
//       keyLengths holds individual key lengths.
__global__
void MurmurHash3_x64_128_kernel(const uint8_t *__restrict__ d_keys,
                                  const uint32_t *__restrict__ d_offsets,
                                  const uint32_t *__restrict__ keyLengths,
                                  uint64_t *__restrict__ d_out,
                                  const uint32_t numKeys)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numKeys) {
        uint64_t hash[2];
        // Single hash computation per kernel launch.
        MurmurHash3_x64_128(d_keys + d_offsets[i], keyLengths[i], i, hash);
        // Write the result.
        d_out[i * 2]     = hash[0];
        d_out[i * 2 + 1] = hash[1];
    }
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

    // Allocate number of keys and determine their lengths
    uint32_t* length = (uint32_t*)malloc(sizeof(uint32_t) * numKeys);
    uint32_t total_length = 0;
    for (i = 0; i < numKeys; i++) {
        length[i] = rand() % 10000;
        total_length += length[i];
    }

    // Create one-dimensional array for cumulative byte offsets (pinned host memory)
    uint32_t* h_offsets;
    cudaHostAlloc((void**)&h_offsets, sizeof(uint32_t) * (numKeys + 1), cudaHostAllocDefault);
    h_offsets[0] = 0;
    for (uint32_t j = 0; j < numKeys; j++) {
        h_offsets[j + 1] = h_offsets[j] + length[j];
    }

    // Allocate a single contiguous host buffer for keys using pinned memory and
    // generate key data directly into it.
    uint8_t* h_keys;
    cudaHostAlloc((void**)&h_keys, sizeof(uint8_t) * total_length, cudaHostAllocDefault);
    for (uint32_t j = 0; j < numKeys; j++) {
        uint8_t* key_ptr = h_keys + h_offsets[j];
        for (uint32_t c = 0; c < length[j]; c++) {
            key_ptr[c] = static_cast<uint8_t>(c % 256);
        }
    }

    // Precompute keys' hashes for verification, storing results in a contiguous array.
    uint64_t* precomputed = (uint64_t*)malloc(sizeof(uint64_t) * numKeys * 2);
    for (uint32_t j = 0; j < numKeys; j++) {
        uint64_t hash[2];
        MurmurHash3_x64_128(h_keys + h_offsets[j], length[j], j, hash);
        precomputed[j * 2]     = hash[0];
        precomputed[j * 2 + 1] = hash[1];
#ifdef DEBUG
        printf("%lu %lu\n", hash[0], hash[1]);
#endif
    }

    // Sanity check that key data was generated correctly (compare within h_keys)
    for (uint32_t j = 0; j < numKeys; j++) {
        for (uint32_t c = 0; c < length[j]; c++) {
            assert(h_keys[h_offsets[j] + c] == static_cast<uint8_t>(c % 256));
        }
    }

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device memory using cudaMalloc
    uint8_t *dev_keys;
    cudaMalloc((void**)&dev_keys, sizeof(uint8_t) * total_length);
    cudaMemcpyAsync(dev_keys, h_keys, sizeof(uint8_t) * total_length, cudaMemcpyHostToDevice, stream);

    // Copy the offsets to the device (numKeys+1 entries)
    uint32_t *dev_offsets;
    cudaMalloc((void**)&dev_offsets, sizeof(uint32_t) * (numKeys + 1));
    cudaMemcpyAsync(dev_offsets, h_offsets, sizeof(uint32_t) * (numKeys + 1), cudaMemcpyHostToDevice, stream);

    // keyLengths on device: holds each key's length
    uint32_t* d_keyLengths;
    cudaMalloc((void**)&d_keyLengths, sizeof(uint32_t) * numKeys);
    cudaMemcpyAsync(d_keyLengths, length, sizeof(uint32_t) * numKeys, cudaMemcpyHostToDevice, stream);

    // Allocate output array on device
    uint64_t* dev_out;
    cudaMalloc((void**)&dev_out, sizeof(uint64_t) * numKeys * 2);

    // Ensure all asynchronous copies have completed before kernel execution
    cudaStreamSynchronize(stream);

    // Use grid dimensions calculated appropriately (one thread per key)
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaStreamSynchronize(stream);
    cudaEventRecord(startEvent, stream);

    // Launch the kernel repeatedly in a loop from the host.
    // Refactored: Removed per-iteration synchronization to minimize overhead.
    for (uint32_t r = 0; r < repeat; r++) {
        MurmurHash3_x64_128_kernel<<<gridDim, blockDim, 0, stream>>>(dev_keys, dev_offsets, d_keyLengths, dev_out, numKeys);
    }
    // Single stream synchronization after all kernel launches.
    cudaStreamSynchronize(stream);

    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    printf("Average kernel execution time %f (s)\n", (elapsed_ms * 1e-3f) / repeat);

    // Copy device output back to host using pinned memory and asynchronous copy
    uint64_t* h_out;
    cudaHostAlloc((void**)&h_out, sizeof(uint64_t) * numKeys * 2, cudaHostAllocDefault);
    cudaMemcpyAsync(h_out, dev_out, sizeof(uint64_t) * numKeys * 2, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Verify the results match the precomputed host hashes
    bool error = false;
    for (uint32_t j = 0; j < numKeys; j++) {
        if (h_out[2 * j] != precomputed[2 * j] || h_out[2 * j + 1] != precomputed[2 * j + 1]) {
            error = true;
            break;
        }
    }
    if (error)
        printf("FAIL\n");
    else
        printf("SUCCESS\n");

    // Clean up host memory
    free(precomputed);
    free(length);
    cudaFreeHost(h_keys);
    cudaFreeHost(h_out);
    cudaFreeHost(h_offsets);

    // Clean up device memory
    cudaFree(dev_keys);
    cudaFree(dev_offsets);
    cudaFree(dev_out);
    cudaFree(d_keyLengths);

    // Destroy CUDA events and stream
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(stream);

    return 0;
}
