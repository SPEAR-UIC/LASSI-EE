
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include <cstdint>

// Begin inlined chacha20.h
#ifndef CHACHA20_H
#define CHACHA20_H

class Chacha20 {
public:
  __device__ Chacha20(const uint8_t *key, const uint8_t *nonce) {
    // Initialize constants.
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;
    // Load key (32 bytes -> 8 words, little-endian).
    for (int i = 0; i < 8; i++) {
      state[4 + i] = key[4*i] |
                     (static_cast<uint32_t>(key[4*i + 1]) << 8) |
                     (static_cast<uint32_t>(key[4*i + 2]) << 16) |
                     (static_cast<uint32_t>(key[4*i + 3]) << 24);
    }
    // Initialize 64-bit block counter to 0.
    state[12] = 0;
    state[13] = 0;
    // Load nonce (8 bytes -> 2 words, little-endian) into state[14] and state[15].
    state[14] = nonce[0] |
                (static_cast<uint32_t>(nonce[1]) << 8) |
                (static_cast<uint32_t>(nonce[2]) << 16) |
                (static_cast<uint32_t>(nonce[3]) << 24);
    state[15] = nonce[4] |
                (static_cast<uint32_t>(nonce[5]) << 8) |
                (static_cast<uint32_t>(nonce[6]) << 16) |
                (static_cast<uint32_t>(nonce[7]) << 24);
  }

  __device__ void crypt(uint8_t *out, int out_len) {
    int blocks = (out_len + 63) / 64;
    int offset = 0;
    for (int b = 0; b < blocks; b++) {
      uint32_t working_state[16];
      // Copy current state.
      for (int i = 0; i < 16; i++) {
          working_state[i] = state[i];
      }
      // ChaCha20: 20 rounds (10 iterations of 2 rounds).
      for (int i = 0; i < 10; i++) {
          // Column rounds.
          quarterRound(working_state, 0, 4, 8, 12);
          quarterRound(working_state, 1, 5, 9, 13);
          quarterRound(working_state, 2, 6, 10, 14);
          quarterRound(working_state, 3, 7, 11, 15);
          // Diagonal rounds.
          quarterRound(working_state, 0, 5, 10, 15);
          quarterRound(working_state, 1, 6, 11, 12);
          quarterRound(working_state, 2, 7, 8, 13);
          quarterRound(working_state, 3, 4, 9, 14);
      }
      // Add the original state to the working state.
      for (int i = 0; i < 16; i++) {
          working_state[i] += state[i];
      }
      // Serialize the output block in little-endian.
      int bytes_to_write = (out_len - offset > 64) ? 64 : (out_len - offset);
      for (int i = 0; i < bytes_to_write; i++) {
          uint32_t word = working_state[i / 4];
          out[offset + i] = (word >> (8 * (i % 4))) & 0xFF;
      }
      offset += bytes_to_write;
      // Increment the 64-bit counter stored in state[12] and state[13].
      state[12]++;
      if (state[12] == 0) {
         state[13]++;
      }
    }
  }

private:
  uint32_t state[16];

  __device__ static inline uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
  }

  __device__ static inline void quarterRound(uint32_t st[16], int a, int b, int c, int d) {
    st[a] += st[b]; st[d] = rotl(st[d] ^ st[a], 16);
    st[c] += st[d]; st[b] = rotl(st[b] ^ st[c], 12);
    st[a] += st[b]; st[d] = rotl(st[d] ^ st[a], 8);
    st[c] += st[d]; st[b] = rotl(st[b] ^ st[c], 7);
  }
};

#endif // CHACHA20_H
// End inlined chacha20.h

// Store the lookup table in constant memory for better caching and energy efficiency.
__constant__ uint8_t d_char_to_uint_const[256];

__device__
void hex_to_raw(const char* src, const int n, uint8_t* dst) {
  // Each thread processes multiple output bytes (stride by blockDim.x).
  for (int i = threadIdx.x; i < n/2; i += blockDim.x) {
    // Read two input characters.
    uint8_t hi = d_char_to_uint_const[static_cast<uint8_t>(src[i*2 + 0])];
    uint8_t lo = d_char_to_uint_const[static_cast<uint8_t>(src[i*2 + 1])];
    dst[i] = (hi << 4) | lo;
  }
}

// Unified kernel that does data conversion and encryption repeatedly.
// By batching the repetitions inside a single kernel launch, we reduce 
// repeated kernel launch overhead and thereby lower GPU energy consumption.
__global__
void test_keystreams (
    const char *__restrict__ text_key,
    const char *__restrict__ text_nonce,
    const char *__restrict__ text_keystream,
    uint8_t *__restrict__ raw_key,
    uint8_t *__restrict__ raw_nonce,
    uint8_t *__restrict__ raw_keystream,
    uint8_t *__restrict__ result,
    const int text_key_size,
    const int text_nonce_size,
    const int text_keystream_size,
    const int repeat)
{
  // Convert hexadecimal inputs into raw bytes.
  hex_to_raw(text_key, text_key_size, raw_key);
  hex_to_raw(text_nonce, text_nonce_size, raw_nonce);
  hex_to_raw(text_keystream, text_keystream_size, raw_keystream);

  // Ensure all threads have completed the conversions.
  __syncthreads();

  // Only one thread (thread 0) launches the Chacha20 encryption repeatedly.
  if (threadIdx.x == 0) {
    Chacha20 chacha(raw_key, raw_nonce);
    const int out_len = text_keystream_size / 2;
    for (int i = 0; i < repeat; i++) {
      // Reset output buffer.
      for (int j = 0; j < out_len; j++) {
        result[j] = 0;
      }
      chacha.crypt(result, out_len);
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Build the lookup table on host.
  uint8_t host_char_to_uint[256] = {0};
  for (int i = 0; i < 10; i++) host_char_to_uint[i + '0'] = i;
  for (int i = 0; i < 26; i++) host_char_to_uint[i + 'a'] = i + 10;
  for (int i = 0; i < 26; i++) host_char_to_uint[i + 'A'] = i + 10;

  // Sample hexadecimal strings.
  const char *h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const char *h_nonce = "0001020304050607";
  const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

  const int key_len = strlen(h_key);
  const int nonce_len = strlen(h_nonce);
  const int keystream_len = strlen(h_keystream);
  const int result_len = keystream_len / 2;

  // Allocate host memory for final results.
  uint8_t *result = (uint8_t*) malloc(result_len);
  uint8_t *raw_keystream = (uint8_t*) malloc(result_len);

  // Device pointers.
  char *d_key;
  uint8_t *d_raw_key;
  char *d_nonce;
  uint8_t *d_raw_nonce;
  char *d_keystream;
  uint8_t *d_raw_keystream;
  uint8_t *d_result;

  // Copy lookup table to constant memory.
  cudaMemcpyToSymbol(d_char_to_uint_const, host_char_to_uint, 256 * sizeof(uint8_t));

  // Allocate and copy key.
  cudaMalloc((void**)&d_key, key_len); 
  cudaMemcpy(d_key, h_key, key_len, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_raw_key, key_len/2);

  // Allocate and copy nonce.
  cudaMalloc((void**)&d_nonce, nonce_len); 
  cudaMemcpy(d_nonce, h_nonce, nonce_len, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_raw_nonce, nonce_len/2);

  // Allocate and copy keystream.
  cudaMalloc((void**)&d_keystream, keystream_len); 
  cudaMemcpy(d_keystream, h_keystream, keystream_len, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_raw_keystream, result_len);
  
  // Allocate memory for encrypted output.
  cudaMalloc((void**)&d_result, result_len);
  
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch a single kernel that performs all repetitions.
  test_keystreams<<<1, 256>>>(
      d_key, d_nonce, d_keystream, 
      d_raw_key, d_raw_nonce, d_raw_keystream, d_result,
      key_len, nonce_len, keystream_len, repeat);
  
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time per iteration (us): %f\n", (time * 1e-3f) / repeat);

  // Retrieve the outputs from device memory.
  cudaMemcpy(result, d_result, result_len, cudaMemcpyDeviceToHost);
  cudaMemcpy(raw_keystream, d_raw_keystream, result_len, cudaMemcpyDeviceToHost);

  // Compare the final encryption result with the raw keystream.
  int error = memcmp(result, raw_keystream, result_len);
  printf("%s\n", error ? "FAIL" : "PASS");

  // Free host and device memory.
  free(result);
  free(raw_keystream);
  cudaFree(d_key);
  cudaFree(d_raw_key);
  cudaFree(d_nonce);
  cudaFree(d_raw_nonce);
  cudaFree(d_keystream);
  cudaFree(d_raw_keystream);
  cudaFree(d_result);

  return 0;
}
