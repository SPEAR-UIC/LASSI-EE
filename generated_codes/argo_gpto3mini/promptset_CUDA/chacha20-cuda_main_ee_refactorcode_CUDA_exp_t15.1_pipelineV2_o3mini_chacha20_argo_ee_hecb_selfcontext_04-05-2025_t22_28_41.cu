
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda.h>
#include <cstdint>

//-------------------------------------------------------------
// Begin chacha20.h
#ifndef CHACHA20_H
#define CHACHA20_H

class Chacha20 {
public:
    // Constructor: copies key (32 bytes) and nonce (8 bytes), sets counter to 0.
    __device__ __host__
    Chacha20(const uint8_t* key, const uint8_t* nonce) : counter(0) {
        for (int i = 0; i < 32; i++) {
            this->key[i] = key[i];
        }
        for (int i = 0; i < 8; i++) {
            this->nonce[i] = nonce[i];
        }
    }

    // Encrypts (i.e. generates the keystream and XORs with the input buffer).
    // Here the input buffer is assumed to be all zero (to generate keystream).
    __device__
    void crypt(uint8_t* buffer, int buflen) {
        int offset = 0;
        while (offset < buflen) {
            uint8_t block[64];
            chacha_block(block, counter);
            counter++;  // Increment block counter after each block.
            int chunk = (buflen - offset < 64) ? (buflen - offset) : 64;
            // XOR with plaintext; since plaintext is assumed 0, simply copy the block.
            for (int i = 0; i < chunk; i++) {
                buffer[offset + i] = block[i];
            }
            offset += chunk;
        }
    }

private:
    // 32-byte key and 8-byte nonce.
    uint8_t key[32];
    uint8_t nonce[8];
    uint64_t counter;

    // Rotate left.
    __device__ __host__ static inline uint32_t rotl(uint32_t x, int b) {
        return (x << b) | (x >> (32 - b));
    }

    // Performs the ChaCha20 quarter round on four state words.
    __device__ __host__ static inline void quarterround(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
        a += b; d ^= a; d = rotl(d, 16);
        c += d; b ^= c; b = rotl(b, 12);
        a += b; d ^= a; d = rotl(d, 8);
        c += d; b ^= c; b = rotl(b, 7);
    }

    // Generates a 64-byte keystream block for the given block counter.
    __device__
    void chacha_block(uint8_t output[64], uint64_t block_counter) const {
        // Constants ("expand 32-byte k")
        const uint32_t constants[4] = { 0x61707865, 0x3320646e, 0x79622d32, 0x6b206574 };

        uint32_t state[16];
        uint32_t working[16];

        // Setup state.
        state[0] = constants[0];
        state[1] = constants[1];
        state[2] = constants[2];
        state[3] = constants[3];

        // Load key into state[4..11] (little-endian conversion).
        for (int i = 0; i < 8; i++) {
            int idx = i * 4;
            state[4 + i] = (uint32_t)key[idx] |
                           ((uint32_t)key[idx + 1] << 8) |
                           ((uint32_t)key[idx + 2] << 16) |
                           ((uint32_t)key[idx + 3] << 24);
        }

        // Split 64-bit block_counter into two 32-bit words.
        state[12] = (uint32_t)(block_counter & 0xFFFFFFFF);
        state[13] = (uint32_t)(block_counter >> 32);

        // Load nonce (8 bytes) into state[14..15] (little-endian conversion).
        state[14] = (uint32_t)nonce[0] |
                    ((uint32_t)nonce[1] << 8) |
                    ((uint32_t)nonce[2] << 16) |
                    ((uint32_t)nonce[3] << 24);
        state[15] = (uint32_t)nonce[4] |
                    ((uint32_t)nonce[5] << 8) |
                    ((uint32_t)nonce[6] << 16) |
                    ((uint32_t)nonce[7] << 24);

        // Copy state to working array.
        for (int i = 0; i < 16; i++) {
            working[i] = state[i];
        }

        // 20 rounds: 10 iterations of double round.
        for (int i = 0; i < 10; i++) {
            // Column rounds.
            quarterround(working[0], working[4], working[8],  working[12]);
            quarterround(working[1], working[5], working[9],  working[13]);
            quarterround(working[2], working[6], working[10], working[14]);
            quarterround(working[3], working[7], working[11], working[15]);
            // Diagonal rounds.
            quarterround(working[0], working[5], working[10], working[15]);
            quarterround(working[1], working[6], working[11], working[12]);
            quarterround(working[2], working[7], working[8],  working[13]);
            quarterround(working[3], working[4], working[9],  working[14]);
        }

        // Add the original state to the result.
        for (int i = 0; i < 16; i++) {
            working[i] += state[i];
        }

        // Serialize the state to bytes (little-endian).
        for (int i = 0; i < 16; i++) {
            output[i * 4 + 0] = (uint8_t)(working[i] & 0xFF);
            output[i * 4 + 1] = (uint8_t)((working[i] >> 8) & 0xFF);
            output[i * 4 + 2] = (uint8_t)((working[i] >> 16) & 0xFF);
            output[i * 4 + 3] = (uint8_t)((working[i] >> 24) & 0xFF);
        }
    }
};

#endif // CHACHA20_H
//-------------------------------------------------------------
// End chacha20.h

//---------------------------------------------------------------------
// Utility: Convert a hexadecimal string to raw bytes on the host.
// Assumes src_len is even.
void hex_to_bytes_host(const char* src, int src_len, uint8_t* dst) {
    for (int i = 0; i < src_len / 2; i++) {
        uint8_t hi = (src[i * 2] >= '0' && src[i * 2] <= '9') ?
                      (src[i * 2] - '0') : ((src[i * 2] | 32) - 'a' + 10);
        uint8_t lo = (src[i * 2 + 1] >= '0' && src[i * 2 + 1] <= '9') ?
                      (src[i * 2 + 1] - '0') : ((src[i * 2 + 1] | 32) - 'a' + 10);
        dst[i] = (hi << 4) | lo;
    }
}

//---------------------------------------------------------------------
// New kernel: Only performs the encryption using already-precomputed raw key and raw nonce.
// The repeating loop is done inside a single kernel launch to reduce launch overhead.
// Only a single thread is used for the sequential encryption so as to avoid wasting GPU energy.
__global__
void test_keystreams(
    const uint8_t *__restrict__ raw_key,
    const uint8_t *__restrict__ raw_nonce,
    uint8_t *__restrict__ result,
    const int result_len,
    const int repeat)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < repeat; i++) {
            Chacha20 chacha(raw_key, raw_nonce);
            chacha.crypt(result, result_len);
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

    // Sample input hexadecimal strings.
    const char *h_key    = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
    const char *h_nonce  = "0001020304050607";
    const char *h_keystream = "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

    const int key_len = strlen(h_key);
    const int nonce_len = strlen(h_nonce);
    const int keystream_len = strlen(h_keystream);
    const int result_len = keystream_len / 2;

    uint8_t* h_raw_key = (uint8_t*) malloc(key_len / 2);
    uint8_t* h_raw_nonce = (uint8_t*) malloc(nonce_len / 2);
    uint8_t* h_raw_keystream_expected = (uint8_t*) malloc(result_len);

    hex_to_bytes_host(h_key, key_len, h_raw_key);
    hex_to_bytes_host(h_nonce, nonce_len, h_raw_nonce);
    hex_to_bytes_host(h_keystream, keystream_len, h_raw_keystream_expected);

    uint8_t *result = (uint8_t*) malloc(result_len);

    uint8_t *d_raw_key;
    uint8_t *d_raw_nonce;
    uint8_t *d_result;

    cudaMalloc((void**)&d_raw_key, key_len / 2);
    cudaMalloc((void**)&d_raw_nonce, nonce_len / 2);
    cudaMalloc((void**)&d_result, result_len);

    cudaMemcpy(d_raw_key, h_raw_key, key_len / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_raw_nonce, h_raw_nonce, nonce_len / 2, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    test_keystreams<<<1, 1>>>(d_raw_key, d_raw_nonce, d_result, result_len, repeat);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels: %f (us)\n", (time_ns * 1e-3f) / repeat);

    cudaMemcpy(result, d_result, result_len, cudaMemcpyDeviceToHost);

    int error = memcmp(result, h_raw_keystream_expected, result_len);
    printf("%s\n", error ? "FAIL" : "PASS");

    free(result);
    free(h_raw_key);
    free(h_raw_nonce);
    free(h_raw_keystream_expected);
    cudaFree(d_raw_key);
    cudaFree(d_raw_nonce);
    cudaFree(d_result);

    return 0;
}
