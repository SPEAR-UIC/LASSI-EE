
#include <stdint.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

__device__ inline uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ void quarterround(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
    a += b; d ^= a; d = rotl(d, 16);
    c += d; b ^= c; b = rotl(b, 12);
    a += b; d ^= a; d = rotl(d, 8);
    c += d; b ^= c; b = rotl(b, 7);
}

class Chacha20 {
public:
    uint8_t key[32];
    uint8_t nonce[8];
    uint32_t counter;

    __device__ Chacha20(const uint8_t* raw_key, const uint8_t* raw_nonce) {
        for (int i = 0; i < 32; i++) {
            key[i] = raw_key[i];
        }
        for (int i = 0; i < 8; i++) {
            nonce[i] = raw_nonce[i];
        }
        counter = 0;
    }

    __device__ void crypt(uint8_t* output, const int keystream_bytes) {
        int offset = 0;
        while (offset < keystream_bytes) {
            uint32_t state[16];
            state[0] = 0x61707865;
            state[1] = 0x3320646e;
            state[2] = 0x79622d32;
            state[3] = 0x6b206574;
            for (int i = 0; i < 8; i++) {
                int j = i * 4;
                state[4 + i] = ((uint32_t)key[j])       |
                               (((uint32_t)key[j + 1]) << 8)  |
                               (((uint32_t)key[j + 2]) << 16) |
                               (((uint32_t)key[j + 3]) << 24);
            }
            state[12] = counter;
            state[13] = ((uint32_t)nonce[0])       |
                        (((uint32_t)nonce[1]) << 8)  |
                        (((uint32_t)nonce[2]) << 16) |
                        (((uint32_t)nonce[3]) << 24);
            state[14] = ((uint32_t)nonce[4])       |
                        (((uint32_t)nonce[5]) << 8)  |
                        (((uint32_t)nonce[6]) << 16) |
                        (((uint32_t)nonce[7]) << 24);
            state[15] = 0;

            uint32_t working_state[16];
            for (int i = 0; i < 16; i++) {
                working_state[i] = state[i];
            }

            for (int i = 0; i < 10; i++) {
                quarterround(working_state[0], working_state[4], working_state[8], working_state[12]);
                quarterround(working_state[1], working_state[5], working_state[9], working_state[13]);
                quarterround(working_state[2], working_state[6], working_state[10], working_state[14]);
                quarterround(working_state[3], working_state[7], working_state[11], working_state[15]);
                quarterround(working_state[0], working_state[5], working_state[10], working_state[15]);
                quarterround(working_state[1], working_state[6], working_state[11], working_state[12]);
                quarterround(working_state[2], working_state[7], working_state[8], working_state[13]);
                quarterround(working_state[3], working_state[4], working_state[9], working_state[14]);
            }
            for (int i = 0; i < 16; i++) {
                working_state[i] += state[i];
            }
            uint8_t block[64];
            for (int i = 0; i < 16; i++) {
                block[i * 4 + 0] = working_state[i] & 0xFF;
                block[i * 4 + 1] = (working_state[i] >> 8) & 0xFF;
                block[i * 4 + 2] = (working_state[i] >> 16) & 0xFF;
                block[i * 4 + 3] = (working_state[i] >> 24) & 0xFF;
            }
            int block_size = ((keystream_bytes - offset) > 64) ? 64 : (keystream_bytes - offset);
            for (int i = 0; i < block_size; i++) {
                output[offset + i] = block[i];
            }
            offset += block_size;
            counter++;
        }
    }
};

__constant__ uint8_t c_char_to_uint[256];

void hex_to_raw_cpu(const char* src, int n, uint8_t* dst, const uint8_t* char_to_uint) {
    for (int i = 0; i < n / 2; i++) {
        uint8_t hi = char_to_uint[(unsigned char)src[i * 2]];
        uint8_t lo = char_to_uint[(unsigned char)src[i * 2 + 1]];
        dst[i] = (hi << 4) | lo;
    }
}

__global__
void chacha20_encrypt(const uint8_t* raw_key,
                      const uint8_t* raw_nonce,
                      uint8_t* result,
                      const int keystream_bytes)
{
    if (threadIdx.x == 0) {
        Chacha20 chacha(raw_key, raw_nonce);
        chacha.crypt(result, keystream_bytes);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    uint8_t host_char_to_uint[256] = {0};
    for (int i = 0; i < 10; i++) host_char_to_uint[i + '0'] = i;
    for (int i = 0; i < 26; i++) host_char_to_uint[i + 'a'] = i + 10;
    for (int i = 0; i < 26; i++) host_char_to_uint[i + 'A'] = i + 10;
    
    cudaMemcpyToSymbol(c_char_to_uint, host_char_to_uint, 256);

    const char *h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    const char *h_nonce = "0001020304050607";
    const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

    const int key_len = strlen(h_key);
    const int nonce_len = strlen(h_nonce);
    const int keystream_len = strlen(h_keystream);
    const int result_len = keystream_len / 2;

    uint8_t *h_raw_key = (uint8_t*) malloc(key_len / 2);
    uint8_t *h_raw_nonce = (uint8_t*) malloc(nonce_len / 2);
    uint8_t *h_raw_keystream = (uint8_t*) malloc(result_len);
    
    hex_to_raw_cpu(h_key, key_len, h_raw_key, host_char_to_uint);
    hex_to_raw_cpu(h_nonce, nonce_len, h_raw_nonce, host_char_to_uint);
    hex_to_raw_cpu(h_keystream, keystream_len, h_raw_keystream, host_char_to_uint);
    
    uint8_t *d_raw_key;
    uint8_t *d_raw_nonce;
    uint8_t *d_result;
    cudaMalloc((void**)&d_raw_key, key_len / 2);
    cudaMalloc((void**)&d_raw_nonce, nonce_len / 2);
    cudaMalloc((void**)&d_result, result_len);

    cudaMemcpy(d_raw_key, h_raw_key, key_len / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_raw_nonce, h_raw_nonce, nonce_len / 2, cudaMemcpyHostToDevice);
    
    uint8_t *result = (uint8_t*) malloc(result_len);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        cudaMemset(d_result, 0, result_len);
        chacha20_encrypt<<<1, 1>>>(d_raw_key, d_raw_nonce, d_result, result_len);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    printf("Average execution time of encryption kernel: %f (us)\n", (time_ns * 1e-3f) / repeat);

    cudaMemcpy(result, d_result, result_len, cudaMemcpyDeviceToHost);

    int error = memcmp(result, h_raw_keystream, result_len);
    printf("%s\n", error ? "FAIL" : "PASS");

    free(result);
    free(h_raw_key);
    free(h_raw_nonce);
    free(h_raw_keystream);
    cudaFree(d_raw_key);
    cudaFree(d_raw_nonce);
    cudaFree(d_result);

    return 0;
}
