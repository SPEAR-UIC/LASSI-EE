
#define N 1024  // Define the grid size
#define IDX(i, j) ((i) * N + (j))  // Define the indexing macro

__global__ void jacobi_step(float* __restrict__ f, 
                            const float* __restrict__ f_old, 
                            float* __restrict__ error) {
  __shared__ float f_old_tile[18][18];
  __shared__ float block_error[16];  // Shared memory for block-level error reduction

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Initialize block-level error to zero
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_error[threadIdx.y] = 0.0f;
  }

  // Load data into shared memory
  if (i < N && j < N) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = f_old[IDX(i, j)];

    if (threadIdx.x == 0 && i >= 1) {
      f_old_tile[threadIdx.y + 1][threadIdx.x + 0] = f_old[IDX(i - 1, j)];
    }
    if (threadIdx.x == 15 && i <= N - 2) {
      f_old_tile[threadIdx.y + 1][threadIdx.x + 2] = f_old[IDX(i + 1, j)];
    }
    if (threadIdx.y == 0 && j >= 1) {
      f_old_tile[threadIdx.y + 0][threadIdx.x + 1] = f_old[IDX(i, j - 1)];
    }
    if (threadIdx.y == 15 && j <= N - 2) {
      f_old_tile[threadIdx.y + 2][threadIdx.x + 1] = f_old[IDX(i, j + 1)];
    }
  }

  __syncthreads();

  float local_error = 0.0f;

  // Perform Jacobi iteration and calculate local error
  if (j >= 1 && j <= N - 2 && i >= 1 && i <= N - 2) {
    f[IDX(i, j)] = 0.25f * (f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                            f_old_tile[threadIdx.y + 1][threadIdx.x + 0] +
                            f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                            f_old_tile[threadIdx.y + 0][threadIdx.x + 1]);
    float df = f[IDX(i, j)] - f_old_tile[threadIdx.y + 1][threadIdx.x + 1];
    local_error = df * df;
  }

  // Warp-level reduction for local error
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_error += __shfl_down_sync(0xffffffff, local_error, offset);
  }

  // First thread in each warp writes to shared memory
  if (threadIdx.x % warpSize == 0) {
    atomicAdd(&block_error[threadIdx.y], local_error);
  }

  __syncthreads();

  // Final reduction across warps in the block
  if (threadIdx.y == 0 && threadIdx.x < blockDim.y) {
    float block_sum = block_error[threadIdx.x];
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    if (threadIdx.x == 0) {
      atomicAdd(error, block_sum);
    }
  }
}

// Dummy main function to resolve the linker error
int main() {
  return 0;
}
