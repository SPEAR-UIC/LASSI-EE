#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t cuda_check_status = (call);                                  \
        if (cuda_check_status != cudaSuccess) {                                  \
            std::fprintf(stderr,                                                \
                         "CUDA error at %s:%d: %s\n",                            \
                         __FILE__,                                               \
                         __LINE__,                                               \
                         cudaGetErrorString(cuda_check_status));                 \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

constexpr int N_MATERIALS = 12;
constexpr int MAX_NUCS = 34;

struct Inputs {
    int n;
    int iterations;
    int block_size;
    bool help;
};

void load_num_nucs(int *num_nucs)
{
    num_nucs[0] = 34;
    num_nucs[1] = 5;
    num_nucs[2] = 4;
    num_nucs[3] = 4;
    num_nucs[4] = 27;
    num_nucs[5] = 21;
    num_nucs[6] = 21;
    num_nucs[7] = 21;
    num_nucs[8] = 21;
    num_nucs[9] = 9;
    num_nucs[10] = 9;
    num_nucs[11] = 9;
}

void load_mats(int *mats)
{
    for (int i = 0; i < N_MATERIALS * MAX_NUCS; ++i) {
        mats[i] = -1;
    }

    const int mat0[MAX_NUCS] = {
        58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 41, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    const int mat1[5] = {
        40, 41, 42, 43, 44
    };
    const int mat2[4] = {
        4, 5, 24, 25
    };
    const int mat3[4] = {
        4, 5, 24, 25
    };
    const int mat4[27] = {
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42
    };
    const int mat5[21] = {
        4, 5, 24, 25, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43
    };
    const int mat6[21] = {
        4, 5, 24, 25, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43
    };
    const int mat7[21] = {
        4, 5, 24, 25, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43
    };
    const int mat8[21] = {
        4, 5, 24, 25, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43
    };
    const int mat9[9] = {
        4, 5, 6, 7, 8, 9, 10, 11, 12
    };
    const int mat10[9] = {
        4, 5, 6, 7, 8, 9, 10, 11, 12
    };
    const int mat11[9] = {
        4, 5, 6, 7, 8, 9, 10, 11, 12
    };

    const int *src[N_MATERIALS] = {
        mat0, mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11
    };

    int num_nucs[N_MATERIALS];
    load_num_nucs(num_nucs);

    for (int m = 0; m < N_MATERIALS; ++m) {
        for (int n = 0; n < num_nucs[m]; ++n) {
            mats[m * MAX_NUCS + n] = src[m][n];
        }
    }
}

void load_concs(double *concs)
{
    for (int i = 0; i < N_MATERIALS * MAX_NUCS; ++i) {
        concs[i] = 0.0;
    }

    const double mat0[MAX_NUCS] = {
        6.118640e-04, 6.118640e-04, 4.481320e-03, 4.481320e-03,
        1.298940e-02, 1.298940e-02, 2.656500e-04, 2.656500e-04,
        8.545800e-05, 8.545800e-05, 1.430000e-06, 1.430000e-06,
        9.540000e-07, 9.540000e-07, 4.520000e-07, 4.520000e-07,
        1.950000e-07, 1.950000e-07, 8.070000e-08, 8.070000e-08,
        4.440000e-08, 4.440000e-08, 4.551950e-02, 4.551950e-02,
        1.223780e-04, 1.223780e-04, 3.024010e-04, 3.024010e-04,
        1.127890e-04, 1.127890e-04, 3.461070e-05, 3.461070e-05,
        4.877500e-06, 4.877500e-06
    };
    const double mat1[5] = {
        4.251960e-02, 8.523540e-03, 5.349720e-03, 5.349720e-03, 8.657260e-04
    };
    const double mat2[4] = {
        6.665000e-02, 3.332500e-02, 1.000000e-05, 4.000000e-05
    };
    const double mat3[4] = {
        6.665000e-02, 3.332500e-02, 1.000000e-05, 4.000000e-05
    };
    const double mat4[27] = {
        7.984480e-03, 7.984480e-03, 7.984480e-03, 7.984480e-03,
        7.984480e-03, 7.984480e-03, 7.984480e-03, 7.984480e-03,
        7.984480e-03, 7.984480e-03, 7.984480e-03, 7.984480e-03,
        7.984480e-03, 7.984480e-03, 7.984480e-03, 7.984480e-03,
        7.984480e-03, 7.984480e-03, 7.984480e-03, 7.984480e-03,
        7.984480e-03, 7.984480e-03, 7.984480e-03, 7.984480e-03,
        7.984480e-03, 7.984480e-03, 7.984480e-03
    };
    const double mat5[21] = {
        4.034780e-02, 2.017390e-02, 4.720000e-05, 1.888000e-04,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03
    };
    const double mat6[21] = {
        4.034780e-02, 2.017390e-02, 4.720000e-05, 1.888000e-04,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03
    };
    const double mat7[21] = {
        4.034780e-02, 2.017390e-02, 4.720000e-05, 1.888000e-04,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03
    };
    const double mat8[21] = {
        4.034780e-02, 2.017390e-02, 4.720000e-05, 1.888000e-04,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03, 1.143380e-03, 1.143380e-03, 1.143380e-03,
        1.143380e-03
    };
    const double mat9[9] = {
        2.000000e-02, 1.000000e-02, 2.500000e-03, 2.500000e-03,
        2.500000e-03, 2.500000e-03, 2.500000e-03, 2.500000e-03,
        2.500000e-03
    };
    const double mat10[9] = {
        2.000000e-02, 1.000000e-02, 2.500000e-03, 2.500000e-03,
        2.500000e-03, 2.500000e-03, 2.500000e-03, 2.500000e-03,
        2.500000e-03
    };
    const double mat11[9] = {
        2.000000e-02, 1.000000e-02, 2.500000e-03, 2.500000e-03,
        2.500000e-03, 2.500000e-03, 2.500000e-03, 2.500000e-03,
        2.500000e-03
    };

    const double *src[N_MATERIALS] = {
        mat0, mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11
    };

    int num_nucs[N_MATERIALS];
    load_num_nucs(num_nucs);

    for (int m = 0; m < N_MATERIALS; ++m) {
        for (int n = 0; n < num_nucs[m]; ++n) {
            concs[m * MAX_NUCS + n] = src[m][n];
        }
    }
}

static int parse_positive_int(const char *text, int fallback)
{
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);

    if (end == text || *end != '\0') {
        return fallback;
    }

    if (value < 1 || value > std::numeric_limits<int>::max()) {
        return fallback;
    }

    return static_cast<int>(value);
}

Inputs read_CLI(int argc, char **argv)
{
    Inputs inputs;
    inputs.n = 1024;
    inputs.iterations = 100;
    inputs.block_size = 16;
    inputs.help = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            inputs.help = true;
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--size") == 0) && i + 1 < argc) {
            ++i;
            inputs.n = parse_positive_int(argv[i], inputs.n);
        } else if ((std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--iterations") == 0) && i + 1 < argc) {
            ++i;
            inputs.iterations = parse_positive_int(argv[i], inputs.iterations);
        } else if ((std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--block") == 0) && i + 1 < argc) {
            ++i;
            inputs.block_size = parse_positive_int(argv[i], inputs.block_size);
        }
    }

    if (inputs.block_size < 1) {
        inputs.block_size = 16;
    }

    if (inputs.block_size > 32) {
        inputs.block_size = 32;
    }

    return inputs;
}

__global__ void initialize_grid_kernel(double *__restrict__ a,
                                       double *__restrict__ b,
                                       int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        int idx = y * n + x;
        double value = 0.0;

        if (x == 0 || y == 0 || x == n - 1 || y == n - 1) {
            double sx = static_cast<double>(x) / static_cast<double>(n);
            double sy = static_cast<double>(y) / static_cast<double>(n);
            value = std::sin(sx) + std::cos(sy);
        }

        a[idx] = value;
        b[idx] = value;
    }
}

__global__ void jacobi_step_kernel(const double *__restrict__ in,
                                   double *__restrict__ out,
                                   int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        int idx = y * n + x;

        if (x > 0 && x < n - 1 && y > 0 && y < n - 1) {
            out[idx] = 0.25 * (in[idx - 1] + in[idx + 1] + in[idx - n] + in[idx + n]);
        } else {
            out[idx] = in[idx];
        }
    }
}

void launch_initialize_grid(double *a,
                            double *b,
                            int n,
                            dim3 grid,
                            dim3 block)
{
    initialize_grid_kernel<<<grid, block>>>(a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_jacobi_step(const double *in,
                        double *out,
                        int n,
                        dim3 grid,
                        dim3 block)
{
    jacobi_step_kernel<<<grid, block>>>(in, out, n);
    CUDA_CHECK(cudaGetLastError());
}

double checksum_host(const std::vector<double> &grid)
{
    double sum = 0.0;

    for (std::size_t i = 0; i < grid.size(); ++i) {
        sum += grid[i];
    }

    return sum;
}

int main(int argc, char **argv)
{
    Inputs inputs = read_CLI(argc, argv);

    if (inputs.help) {
        std::printf("Usage: ./a.out [-n size] [-i iterations] [-b block]\n");
        return 0;
    }

    int num_nucs[N_MATERIALS];
    int mats[N_MATERIALS * MAX_NUCS];
    double concs[N_MATERIALS * MAX_NUCS];

    load_num_nucs(num_nucs);
    load_mats(mats);
    load_concs(concs);

    int n = inputs.n;
    int iterations = inputs.iterations;
    int block_size = inputs.block_size;

    std::size_t total_cells = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::size_t bytes = total_cells * sizeof(double);

    double *d_a = nullptr;
    double *d_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    dim3 block(block_size, block_size, 1);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);

    launch_initialize_grid(d_a, d_b, n, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    double *current = d_a;
    double *next = d_b;

    for (int iter = 0; iter < iterations; ++iter) {
        launch_jacobi_step(current, next, n, grid, block);
        std::swap(current, next);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> host_grid(total_cells);
    CUDA_CHECK(cudaMemcpy(host_grid.data(), current, bytes, cudaMemcpyDeviceToHost));

    double sum = checksum_host(host_grid);

    std::printf("Jacobi CUDA simulation\n");
    std::printf("Grid size: %d\n", n);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Checksum: %.12e\n", sum);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
