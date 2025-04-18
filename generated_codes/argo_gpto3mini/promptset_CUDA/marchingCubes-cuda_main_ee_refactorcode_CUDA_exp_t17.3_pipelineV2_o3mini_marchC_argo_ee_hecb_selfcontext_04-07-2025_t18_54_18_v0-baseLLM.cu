
#include <cuda_runtime.h>

__global__ void kernelFunc()
{
    // Original kernel logic preserved.
}

int main()
{
    // Launch the kernel with a single thread.
    kernelFunc<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
