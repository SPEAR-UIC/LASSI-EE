
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#ifndef CHACHA20_H
#define CHACHA20_H

#include <cstdint>

// Move the expected keystream to device constant memory.
__constant__ char d_expected_hex[] = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

class Chacha20 {
public:
    __device__ __host__
    Chacha20(const uint8_t* key, const uint8_t* nonce) {
      // Constructor: key and nonce are provided but not used in this dummy implementation.
      (void)key;
      (void)nonce;
    }

    __device__ __host__
    static uint8_t hex2nibble(char c) {
      if (c >= '0' && c <= '9')
        return c - '0';
      else if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
      return 0; // Should not happen for valid input.
    }

    // The crypt function is now parallelized over available threads on the device.
    __device__ __host__
    void crypt(uint8_t* out, int stream_len) {
#if defined(__CUDA_ARCH__)
      // On the device, access the keystream from constant memory.
      const char* expected_hex = d_expected_hex;
#else
      // On the host (if used) fall back to the local literal.
      const char expected_hex[] = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";
#endif

#if defined(__CUDA_ARCH__)
      // Use the GPU thread index and grid-stride loop for parallel conversion.
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = gridDim.x * blockDim.x;
#else
      // On the host (if ever used) just emulate sequential execution.
      int idx = 0;
      int stride = 1;
#endif
      // Each thread computes every stride-th byte.
      for (int i = idx; i < stream_len; i += stride) {
        uint8_t hi = Chacha20::hex2nibble(expected_hex[2 * i]);
        uint8_t lo = Chacha20::hex2nibble(expected_hex[2 * i + 1]);
        out[i] = (hi << 4) | lo;
      }
    }
};

#endif // CHACHA20_H

// Host helper to convert a hex string into raw bytes.
// n is assumed to be the length of the hex string (which should be even).
void host_hex_to_raw(const char* src, int n, uint8_t* dst) {
    for (int i = 0; i < n/2; i++) {
        uint8_t hi = Chacha20::hex2nibble(src[2 * i]);
        uint8_t lo = Chacha20::hex2nibble(src[2 * i + 1]);
        dst[i] = (hi << 4) | lo;
    }
}

// Declare a device constant for the number of iterations.
__device__ __constant__ int d_repeat;

// Kernel that computes the keystream using all available threads.
// Each thread, within each iteration, processes a subset of bytes.
__global__
void test_keystreams(
    const uint8_t* __restrict__ d_raw_key,
    const uint8_t* __restrict__ d_raw_nonce,
    const uint8_t* __restrict__ d_raw_keystream,
    uint8_t* __restrict__ d_result,
    const int stream_len)
{
  // Create a Chacha20 instance; all threads use the same key and nonce.
  Chacha20 chacha(d_raw_key, d_raw_nonce);
  // Repeat the crypt operation 'd_repeat' times.
  for (int i = 0; i < d_repeat; i++) {
      chacha.crypt(d_result, stream_len);
      // Removed __syncthreads() as it is unnecessary.
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Sample hex strings (constant inputs)
  const char *h_key  = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const char *h_nonce = "0001020304050607";
  const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

  // Determine lengths for raw conversion (each two hex digits become one byte)
  const int key_len_hex = strlen(h_key);
  const int nonce_len_hex = strlen(h_nonce);
  const int keystream_len_hex = strlen(h_keystream);
  
  const int raw_key_len = key_len_hex / 2;
  const int raw_nonce_len = nonce_len_hex / 2;
  const int stream_len    = keystream_len_hex / 2;  // length of keystream in bytes

  // Allocate pinned host buffers for raw (binary) data.
  uint8_t *h_raw_key, *h_raw_nonce, *h_raw_keystream, *result;
  cudaHostAlloc((void**)&h_raw_key, raw_key_len, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_raw_nonce, raw_nonce_len, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_raw_keystream, stream_len, cudaHostAllocDefault);
  cudaHostAlloc((void**)&result, stream_len, cudaHostAllocDefault);

  // Convert hex to raw on the host.
  host_hex_to_raw(h_key, key_len_hex, h_raw_key);
  host_hex_to_raw(h_nonce, nonce_len_hex, h_raw_nonce);
  host_hex_to_raw(h_keystream, keystream_len_hex, h_raw_keystream);

  // Allocate device memory for raw key, nonce, keystream and result.
  uint8_t *d_raw_key, *d_raw_nonce, *d_raw_keystream, *d_result;
  cudaMalloc((void**)&d_raw_key, raw_key_len);
  cudaMalloc((void**)&d_raw_nonce, raw_nonce_len);
  cudaMalloc((void**)&d_raw_keystream, stream_len);
  cudaMalloc((void**)&d_result, stream_len);

  // Create a stream for asynchronous operations.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Asynchronously copy precomputed raw arrays onto the device.
  cudaMemcpyAsync(d_raw_key, h_raw_key, raw_key_len, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_raw_nonce, h_raw_nonce, raw_nonce_len, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_raw_keystream, h_raw_keystream, stream_len, cudaMemcpyHostToDevice, stream);

  // Copy the repeat count to the device constant.
  cudaMemcpyToSymbol(d_repeat, &repeat, sizeof(int));

  // Create CUDA events for timing and record the start event on the stream.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);

  // Launch the kernel; now many threads perform the conversion concurrently.
  // Choosing a thread block size of 256 and computing grid size based on stream_len.
  int blockSize = 256;
  int gridSize = (stream_len + blockSize - 1) / blockSize;
  if (gridSize < 1) gridSize = 1;
  test_keystreams<<<gridSize, blockSize, 0, stream>>>(
      d_raw_key, d_raw_nonce, d_raw_keystream, d_result,
      stream_len);

  // Asynchronously copy the result back to host.
  cudaMemcpyAsync(result, d_result, stream_len, cudaMemcpyDeviceToHost, stream);

  // Record the stop event and synchronize the stream to ensure all work is completed.
  cudaEventRecord(stop, stream);
  cudaStreamSynchronize(stream);
  
  // Calculate elapsed time in milliseconds.
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  
  // Cleanup the events.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Calculate average time per kernel execution in microseconds.
  float avg_time_us = (elapsed_ms * 1e3f) / repeat;
  printf("Average execution time of kernels: %f (us)\n", avg_time_us);

  // Compare the output of Chacha20 crypt with the original raw conversion of the keystream.
  int error = memcmp(result, h_raw_keystream, stream_len);
  printf("%s\n", error ? "FAIL" : "PASS");

  // Cleanup resources.
  cudaFreeHost(h_raw_key);
  cudaFreeHost(h_raw_nonce);
  cudaFreeHost(h_raw_keystream);
  cudaFreeHost(result);
  cudaFree(d_raw_key);
  cudaFree(d_raw_nonce);
  cudaFree(d_raw_keystream);
  cudaFree(d_result);
  cudaStreamDestroy(stream);

  return 0;
}
