
#include <cuda_runtime.h>

__global__ void kernelFunction() {
    // Original kernel logic remains unchanged.
}

int main() {
    // Launch the kernel with a single thread.
    kernelFunction<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
