
#include <cuda_runtime.h>
#include <iostream>

__global__ void sampleKernel() {
    // Kernel logic (if any) goes here
}

int main() {
    // Launch the kernel
    sampleKernel<<<1, 1>>>();
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus 
                  << " after launching sampleKernel!" << std::endl;
        return 1;
    }

    return 0;
}
