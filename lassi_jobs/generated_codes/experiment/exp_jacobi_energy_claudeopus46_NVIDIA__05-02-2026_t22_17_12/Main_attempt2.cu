#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cuda.h>

#define N 16384
#define IDX(i, j) ((i) + (j) * N)

#define TILE_X 32
#define TILE_Y 8

void initialize_data(float* f) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            if (i == 0 || i == N-1) {
                f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
            }
            else if (j == 0 || j == N-1) {
                f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
            }
            else {
                f[IDX(i,j)] = 0.0f;
            }
        }
    }
}

// Jacobi step WITHOUT error computation - used for non-check iterations
__global__ __launch_bounds__(256, 8)
void jacobi_step_no_error(const float* __restrict__ f_old,
                          float* __restrict__ f) {
    __shared__ float f_old_tile[TILE_Y + 2][TILE_X + 2];

    int i = threadIdx.x + blockIdx.x * TILE_X;
    int j = threadIdx.y + blockIdx.y * TILE_Y;

    // Load interior
    float old_val = __ldg(&f_old[IDX(i, j)]);
    f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = old_val;

    // Load halos
    if (threadIdx.x == 0 && i >= 1) {
        f_old_tile[threadIdx.y + 1][0] = __ldg(&f_old[IDX(i - 1, j)]);
    }
    if (threadIdx.x == TILE_X - 1 && i <= N - 2) {
        f_old_tile[threadIdx.y + 1][threadIdx.x + 2] = __ldg(&f_old[IDX(i + 1, j)]);
    }
    if (threadIdx.y == 0 && j >= 1) {
        f_old_tile[0][threadIdx.x + 1] = __ldg(&f_old[IDX(i, j - 1)]);
    }
    if (threadIdx.y == TILE_Y - 1 && j <= N - 2) {
        f_old_tile[threadIdx.y + 2][threadIdx.x + 1] = __ldg(&f_old[IDX(i, j + 1)]);
    }

    __syncthreads();

    if (j >= 1 && j <= N - 2) {
        if (i >= 1 && i <= N - 2) {
            float new_val = 0.25f * (f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                                     f_old_tile[threadIdx.y + 1][threadIdx.x + 0] +
                                     f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                                     f_old_tile[threadIdx.y + 0][threadIdx.x + 1]);
            f[IDX(i, j)] = new_val;
        }
    }
}

// Jacobi step WITH error computation - used for check iterations
__global__ __launch_bounds__(256, 8)
void jacobi_step_with_error(const float* __restrict__ f_old,
                            float* __restrict__ f,
                            float* __restrict__ error) {
    __shared__ float f_old_tile[TILE_Y + 2][TILE_X + 2];

    int i = threadIdx.x + blockIdx.x * TILE_X;
    int j = threadIdx.y + blockIdx.y * TILE_Y;

    float old_val = __ldg(&f_old[IDX(i, j)]);
    f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = old_val;

    if (threadIdx.x == 0 && i >= 1) {
        f_old_tile[threadIdx.y + 1][0] = __ldg(&f_old[IDX(i - 1, j)]);
    }
    if (threadIdx.x == TILE_X - 1 && i <= N - 2) {
        f_old_tile[threadIdx.y + 1][threadIdx.x + 2] = __ldg(&f_old[IDX(i + 1, j)]);
    }
    if (threadIdx.y == 0 && j >= 1) {
        f_old_tile[0][threadIdx.x + 1] = __ldg(&f_old[IDX(i, j - 1)]);
    }
    if (threadIdx.y == TILE_Y - 1 && j <= N - 2) {
        f_old_tile[threadIdx.y + 2][threadIdx.x + 1] = __ldg(&f_old[IDX(i, j + 1)]);
    }

    __syncthreads();

    float err = 0.0f;

    if (j >= 1 && j <= N - 2) {
        if (i >= 1 && i <= N - 2) {
            float new_val = 0.25f * (f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                                     f_old_tile[threadIdx.y + 1][threadIdx.x + 0] +
                                     f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                                     f_old_tile[threadIdx.y + 0][threadIdx.x + 1]);
            f[IDX(i, j)] = new_val;
            float df = new_val - old_val;
            err = df * df;
        }
    }

    // Warp-level reduction (32 threads in x-dimension = full warp)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        err += __shfl_down_sync(0xffffffff, err, offset);
    }

    // Now lane 0 of each warp has the partial sum
    // With TILE_X=32, TILE_Y=8, we have 256 threads = 8 warps
    // Each warp corresponds to one row (threadIdx.y)
    __shared__ float reduction_array[TILE_Y];
    if (threadIdx.x == 0) {
        reduction_array[threadIdx.y] = err;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < TILE_Y) {
        err = reduction_array[threadIdx.x];
    } else {
        err = 0.0f;
    }

    if (threadIdx.y == 0) {
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            err += __shfl_down_sync(0xffffffff, err, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(error, err);
        }
    }
}

int main() {
    std::clock_t start_time = std::clock();

    float* d_f;
    float* d_f_old;
    float* d_error;

    float* f = (float*)aligned_alloc(64, N * N * sizeof(float));
    float* f_old = (float*)aligned_alloc(64, N * N * sizeof(float));

    initialize_data(f);
    initialize_data(f_old);

    cudaMalloc((void**)&d_f, N * N * sizeof(float));
    cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
    cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_error, sizeof(float));

    float error = std::numeric_limits<float>::max();
    const float tolerance = 1.e-5f;
    const int max_iters = 10000;
    int num_iters = 0;

    dim3 grid(N / TILE_X, N / TILE_Y);
    dim3 block(TILE_X, TILE_Y);

    const int check_interval = 100;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    while (error > tolerance && num_iters < max_iters) {
        bool do_check = (num_iters % check_interval == 0) || (num_iters == max_iters - 1);

        if (do_check) {
            cudaMemset(d_error, 0, sizeof(float));
            jacobi_step_with_error<<<grid, block>>>(d_f_old, d_f, d_error);
            cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
            error = sqrtf(error / (N * N));
        } else {
            jacobi_step_no_error<<<grid, block>>>(d_f_old, d_f);
        }

        if (num_iters % 1000 == 0) {
            if (!do_check) {
                // Need to compute error for printing
                cudaMemset(d_error, 0, sizeof(float));
                // Re-run with error on the ALREADY computed result - but we already wrote f.
                // Actually the error was already computed if do_check was true at %1000
                // Since 1000 % 100 == 0, do_check is always true at %1000
            }
            std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
        }

        // Swap pointers instead of copying data
        float* tmp = d_f_old;
        d_f_old = d_f;
        d_f = tmp;

        ++num_iters;
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

    if (error <= tolerance && num_iters < max_iters) {
        std::cout << "PASS" << std::endl;
    }
    else {
        std::cout << "FAIL" << std::endl;
        return -1;
    }

    cudaFree(d_f);
    cudaFree(d_f_old);
    cudaFree(d_error);
    free(f);
    free(f_old);

    double duration = (std::clock() - start_time) / (double)CLOCKS_PER_SEC;
    std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

    return 0;
}
