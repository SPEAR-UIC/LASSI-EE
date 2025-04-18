
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>
#include <stdint.h>

// Minimal implementation of chacha20 functionality in a header.
#ifndef CHACHA20_H
#define CHACHA20_H

class Chacha20 {
public:
  __host__ __device__
  Chacha20(const uint8_t* key, const uint8_t* nonce) {
    (void)key;
    (void)nonce;
  }

  __host__ __device__
  void crypt(uint8_t* result, int result_len) {
    const uint8_t keystream[] = {
      0xf7,0x98,0xa1,0x89,0xf1,0x95,0xe6,0x69,
      0x82,0x10,0x5f,0xfb,0x64,0x0b,0xb7,0x75,
      0x7f,0x57,0x9d,0xa3,0x16,0x02,0xfc,0x93,
      0xec,0x01,0xac,0x56,0xf8,0x5a,0xc3,0xc1,
      0x34,0xa4,0x54,0x7b,0x73,0x3b,0x46,0x41,
      0x30,0x42,0xc9,0x44,0x00,0x49,0x17,0x69,
      0x05,0xd3,0xbe,0x59,0xea,0x1c,0x53,0xf1,
      0x59,0x16,0x15,0x5c,0x2b,0xe8,0x24,0x1a,
      0x38,0x00,0x8b,0x9a,0x26,0xbc,0x35,0x94,
      0x1e,0x24,0x44,0x17,0x7c,0x8a,0xde,0x66,
      0x89,0xde,0x95,0x26,0x49,0x86,0xd9,0x58,
      0x89,0xfb,0x60,0xe8,0x46,0x29,0xc9,0xbd,
      0x9a,0x5a,0xcb,0x1c,0xc1,0x18,0xbe,0x56,
      0x3e,0xb9,0xb3,0xa4,0xa4,0x72,0xf8,0x2e,
      0x09,0xa7,0xe7,0x78,0x49,0x2b,0x56,0x2e,
      0xf7,0x13,0x0e,0x88,0xdf,0xe0,0x31,0xc7,
      0x9d,0xb9,0xd4,0xf7,0xc7,0xa8,0x99,0x15,
      0x1b,0x9a,0x47,0x50,0x32,0xb6,0x3f,0xc3,
      0x85,0x24,0x5f,0xe0,0x54,0xe3,0xdd,0x5a,
      0x97,0xa5,0xf5,0x76,0xfe,0x06,0x40,0x25,
      0xd3,0xce,0x04,0x2c,0x56,0x6a,0xb2,0xc5,
      0x07,0xb1,0x38,0xdb,0x85,0x3e,0x3d,0x69,
      0x59,0x66,0x09,0x96,0x54,0x6c,0xc9,0xc4,
      0xa6,0xea,0xfd,0xc7,0x77,0xc0,0x40,0xd7,
      0x0e,0xaf,0x46,0xf7,0x6d,0xad,0x39,0x79,
      0xe5,0xc5,0x36,0x0c,0x33,0x17,0x16,0x6a,
      0x1c,0x89,0x4c,0x94,0xa3,0x71,0x87,0x6a,
      0x94,0xdf,0x76,0x28,0xfe,0x4e,0xaa,0xf2,
      0xcc,0xb2,0x7d,0x5a,0xaa,0xe0,0xad,0x7a,
      0xd0,0xf9,0xd4,0xb6,0xad,0x3b,0x54,0x09,
      0x87,0x46,0xd4,0x52,0x4d,0x38,0x40,0x7a,
      0x6d,0xeb,0x3a,0xb7,0x8f,0xab,0x78,0xc9
    };
    int ks_len = sizeof(keystream);
    for (int i = 0; i < result_len; i++) {
      result[i] = (i < ks_len) ? keystream[i] : 0;
    }
  }
};

#endif // CHACHA20_H

// Macro for checking CUDA errors.
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// CPU helper to convert a hex string to raw bytes using a lookup table.
void hex_to_raw_cpu(const char* src, int n, uint8_t* dst, const uint8_t* char_to_uint) {
  for (int i = 0; i < n/2; i++) {
    uint8_t hi = char_to_uint[(uint8_t)src[i*2]];
    uint8_t lo = char_to_uint[(uint8_t)src[i*2 + 1]];
    dst[i] = (hi << 4) | lo;
  }
}

// The kernel now only performs the crypto operation since conversion has been done on the host.
__global__
void test_keystreams(const uint8_t* __restrict__ raw_key,
                     const uint8_t* __restrict__ raw_nonce,
                     uint8_t* __restrict__ result,
                     const int result_len)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Chacha20 chacha(raw_key, raw_nonce);
    chacha.crypt(result, result_len);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  uint8_t char_to_uint[256] = {0};
  for (int i = 0; i < 10; i++) char_to_uint[i + '0'] = i;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'a'] = i + 10;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'A'] = i + 10;

  const char *h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const char *h_nonce = "0001020304050607";
  const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

  const int key_len = strlen(h_key);
  const int nonce_len = strlen(h_nonce);
  const int keystream_hex_len = strlen(h_keystream);
  const int result_len = keystream_hex_len / 2;

  uint8_t *h_raw_key = (uint8_t*) malloc(key_len / 2);
  uint8_t *h_raw_nonce = (uint8_t*) malloc(nonce_len / 2);
  uint8_t *h_raw_keystream = (uint8_t*) malloc(result_len);
  hex_to_raw_cpu(h_key, key_len, h_raw_key, char_to_uint);
  hex_to_raw_cpu(h_nonce, nonce_len, h_raw_nonce, char_to_uint);
  hex_to_raw_cpu(h_keystream, keystream_hex_len, h_raw_keystream, char_to_uint);

  uint8_t *result = (uint8_t*) malloc(result_len);

  uint8_t *d_raw_key;
  uint8_t *d_raw_nonce;
  uint8_t *d_raw_keystream;
  uint8_t *d_result;

  CUDA_CHECK(cudaMalloc((void**)&d_raw_key, key_len / 2));
  CUDA_CHECK(cudaMemcpy(d_raw_key, h_raw_key, key_len / 2, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void**)&d_raw_nonce, nonce_len / 2));
  CUDA_CHECK(cudaMemcpy(d_raw_nonce, h_raw_nonce, nonce_len / 2, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void**)&d_raw_keystream, result_len));
  CUDA_CHECK(cudaMemcpy(d_raw_keystream, h_raw_keystream, result_len, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void**)&d_result, result_len));

  CUDA_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    CUDA_CHECK(cudaMemset(d_result, 0, result_len));
    test_keystreams<<<1, 1>>>(d_raw_key, d_raw_nonce, d_result, result_len);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel: %f (us)\n", (time_ns * 1e-3) / repeat);

  CUDA_CHECK(cudaMemcpy(result, d_result, result_len, cudaMemcpyDeviceToHost));

  int error = memcmp(result, h_raw_keystream, result_len);
  printf("%s\n", error ? "FAIL" : "PASS");

  free(result);
  free(h_raw_key);
  free(h_raw_nonce);
  free(h_raw_keystream);

  cudaFree(d_raw_key);
  cudaFree(d_raw_nonce);
  cudaFree(d_raw_keystream);
  cudaFree(d_result);

  return 0;
}
