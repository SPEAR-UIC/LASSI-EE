
#include <cuda_runtime.h>

__global__ void sampleKernel() {
    // Kernel logic here remains unchanged.
}

int main() {
    // Launch the sampleKernel with one block and one thread, as a minimal invocation.
    sampleKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
