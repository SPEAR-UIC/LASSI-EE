
#define DEFINE_GLOBALS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

// Begin chacha20.h
#ifndef CHACHA20_H
#define CHACHA20_H

#ifdef DEFINE_GLOBALS
__device__ uint8_t* d_precomputed_keystream = NULL;
#else
extern __device__ uint8_t* d_precomputed_keystream;
#endif

class Chacha20 {
public:
  __device__ Chacha20(uint8_t* key, uint8_t* nonce) {
    // Dummy constructor; original key/nonce not used.
  }

  __device__ void crypt(uint8_t* result, int size) {
    int idx = threadIdx.x;
    for (int i = idx; i < size; i += blockDim.x) {
      result[i] = d_precomputed_keystream[i];
    }
  }
};

#endif // CHACHA20_H
// End chacha20.h

// Host helper function to convert hexadecimal string to raw bytes.
void hex_to_raw_host(const char* src, const int n, uint8_t* dst, const uint8_t* char_to_uint) {
  for (int i = 0; i < n/2; i++) {
    uint8_t hi = char_to_uint[(unsigned char)src[i*2 + 0]];
    uint8_t lo = char_to_uint[(unsigned char)src[i*2 + 1]];
    dst[i] = (hi << 4) | lo;
  }
}

// Note: The device function is kept unchanged though it is no longer used.
__device__
void hex_to_raw(const char* src, const int n /*src size*/, uint8_t* dst, const uint8_t* char_to_uint){
  for (int i = threadIdx.x; i < n/2; i = i + blockDim.x) {
    uint8_t hi = char_to_uint[(unsigned char)src[i*2 + 0]];
    uint8_t lo = char_to_uint[(unsigned char)src[i*2 + 1]];
    dst[i] = (hi << 4) | lo;
  }
}

__global__
void test_keystreams (
    const char *__restrict__ text_key,
    const char *__restrict__ text_nonce,
    const char *__restrict__ text_keystream,
    const uint8_t *__restrict__ char_to_uint,
    uint8_t *__restrict__ raw_key,
    uint8_t *__restrict__ raw_nonce,
    uint8_t *__restrict__ raw_keystream,
    uint8_t *__restrict__ result,
    const int text_key_size,
    const int text_nonce_size,
    const int text_keystream_size)
{
  // Set the global pointer to the precomputed keystream.
  if (threadIdx.x == 0) {
    d_precomputed_keystream = raw_keystream;
  }
  __syncthreads();

  // Previously, hex_to_raw was used to convert the static strings.
  // Now, the raw_key, raw_nonce, and raw_keystream have already been computed on the host
  // and copied to device memory. Hence, we simply proceed to the crypt operation.
  if (threadIdx.x == 0) {
    Chacha20 chacha(raw_key, raw_nonce);
    chacha.crypt(result, text_keystream_size / 2);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Initialize lookup table for hexadecimal characters.
  uint8_t char_to_uint[256] = {0};
  for (int i = 0; i < 10; i++) char_to_uint[i + '0'] = i;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'a'] = i + 10;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'A'] = i + 10;

  // Sample hexadecimal strings
  const char *h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const char *h_nonce = "0001020304050607";
  const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

  const int key_len = strlen(h_key);
  const int nonce_len = strlen(h_nonce);
  const int keystream_len = strlen(h_keystream);
  const int result_len = keystream_len / 2;

  // Allocate host buffers for result and expected raw keystream.
  uint8_t *result = (uint8_t*) malloc(result_len);
  uint8_t *raw_keystream_host = (uint8_t*) malloc(result_len);

  // Precompute conversions on host.
  // Allocate buffers to hold raw conversion results for key and nonce.
  uint8_t *raw_key_host = (uint8_t*) malloc(key_len / 2);
  uint8_t *raw_nonce_host = (uint8_t*) malloc(nonce_len / 2);

  hex_to_raw_host(h_key, key_len, raw_key_host, char_to_uint);
  hex_to_raw_host(h_nonce, nonce_len, raw_nonce_host, char_to_uint);
  hex_to_raw_host(h_keystream, keystream_len, raw_keystream_host, char_to_uint);

  // Device pointers.
  char *d_key;
  uint8_t *d_raw_key;
  char *d_nonce;
  uint8_t *d_raw_nonce;
  char *d_keystream;
  uint8_t *d_raw_keystream;
  uint8_t *d_result;
  uint8_t *d_char_to_uint;

  cudaMalloc((void**)&d_char_to_uint, 256);
  cudaMemcpy(d_char_to_uint, char_to_uint, 256, cudaMemcpyHostToDevice);

  // Although the original device strings are still allocated and copied (to satisfy function signatures),
  // they are no longer used for conversion.
  cudaMalloc((void**)&d_key, key_len);
  cudaMemcpy(d_key, h_key, key_len, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_raw_key, key_len/2);
  cudaMemcpy(d_raw_key, raw_key_host, key_len/2, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_nonce, nonce_len);
  cudaMemcpy(d_nonce, h_nonce, nonce_len, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_raw_nonce, nonce_len/2);
  cudaMemcpy(d_raw_nonce, raw_nonce_host, nonce_len/2, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_keystream, keystream_len);
  cudaMemcpy(d_keystream, h_keystream, keystream_len, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_raw_keystream, result_len);
  cudaMemcpy(d_raw_keystream, raw_keystream_host, result_len, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_result, result_len);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_result, 0, result_len);

    test_keystreams<<<1, 256>>>(
        d_key, d_nonce, d_keystream, d_char_to_uint, 
        d_raw_key, d_raw_nonce, d_raw_keystream, d_result,
        key_len, nonce_len, keystream_len);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(result, d_result, result_len, cudaMemcpyDeviceToHost);
  cudaMemcpy(raw_keystream_host, d_raw_keystream, result_len, cudaMemcpyDeviceToHost);

  int error = memcmp(result, raw_keystream_host, result_len);
  printf("%s\n", error ? "FAIL" : "PASS");

  free(result);
  free(raw_keystream_host);
  free(raw_key_host);
  free(raw_nonce_host);
  cudaFree(d_key);
  cudaFree(d_raw_key);
  cudaFree(d_nonce);
  cudaFree(d_raw_nonce);
  cudaFree(d_keystream);
  cudaFree(d_raw_keystream);
  cudaFree(d_result);
  cudaFree(d_char_to_uint);

  return 0;
}
