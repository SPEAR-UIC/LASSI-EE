
#include <cstdio>
#include <cuda_runtime.h>

__global__ void dummyKernel() {
    // This kernel is intentionally left blank.
}

int main() {
    // Launch dummyKernel with one block of one thread.
    dummyKernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
