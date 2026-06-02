#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t cuda_check_status = (call);                                           \
        if (cuda_check_status != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorString(cuda_check_status));                          \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

struct CLIOptions {
    int n;
    int iterations;
    bool help;
};

CLIOptions read_CLI(int argc, char **argv)
{
    CLIOptions options;
    options.n = 1024;
    options.iterations = 100;
    options.help = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            options.help = true;
        } else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--size") == 0) && i + 1 < argc) {
            options.n = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "-i") == 0 || std::strcmp(argv[i], "--iterations") == 0) && i + 1 < argc) {
            options.iterations = std::atoi(argv[++i]);
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            if (options.n == 1024) {
                options.n = std::atoi(argv[i]);
            } else {
                options.iterations = std::atoi(argv[i]);
            }
        }
    }

    if (options.n < 1) {
        options.n = 1;
    }

    if (options.iterations < 0) {
        options.iterations = 0;
    }

    return options;
}

int *load_num_nucs()
{
    int *num_nucs = static_cast<int *>(std::malloc(12 * sizeof(int)));

    if (num_nucs == nullptr) {
        std::fprintf(stderr, "Failed to allocate num_nucs\n");
        std::exit(EXIT_FAILURE);
    }

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

    return num_nucs;
}

int **load_mats(int *num_nucs, int n_isotopes)
{
    int **mats = static_cast<int **>(std::malloc(12 * sizeof(int *)));

    if (mats == nullptr) {
        std::fprintf(stderr, "Failed to allocate mats\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 12; ++i) {
        mats[i] = static_cast<int *>(std::malloc(static_cast<size_t>(num_nucs[i]) * sizeof(int)));
        if (mats[i] == nullptr) {
            std::fprintf(stderr, "Failed to allocate mats[%d]\n", i);
            std::exit(EXIT_FAILURE);
        }
    }

    const int mat0[34] = {
        58, 59, 60, 61, 40, 42, 43, 44, 46, 48, 49, 50, 52, 54, 55, 56, 57,
        41, 47, 53, 45, 51, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
    };
    const int mat1[5] = {
        63, 64, 65, 66, 67
    };
    const int mat2[4] = {
        24, 41, 4, 5
    };
    const int mat3[4] = {
        24, 41, 4, 5
    };
    const int mat4[27] = {
        19, 20, 21, 22, 35, 36, 37, 38, 39,
        24, 41, 4, 5, 23, 42, 43, 44, 46,
        48, 49, 50, 52, 54, 55, 56, 57, 47
    };
    const int mat5[21] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37,
        38, 39, 23, 42, 43, 44, 46, 48, 49, 50
    };
    const int mat6[21] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37,
        38, 39, 23, 42, 43, 44, 46, 48, 49, 50
    };
    const int mat7[21] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37,
        38, 39, 23, 42, 43, 44, 46, 48, 49, 50
    };
    const int mat8[21] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37,
        38, 39, 23, 42, 43, 44, 46, 48, 49, 50
    };
    const int mat9[9] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35
    };
    const int mat10[9] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35
    };
    const int mat11[9] = {
        24, 41, 4, 5, 19, 20, 21, 22, 35
    };

    const int *source[12] = {
        mat0, mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8, mat9, mat10, mat11
    };

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < num_nucs[i]; ++j) {
            int value = source[i][j];
            if (n_isotopes > 0 && value >= n_isotopes) {
                value = value % n_isotopes;
            }
            mats[i][j] = value;
        }
    }

    return mats;
}

double **load_concs(int *num_nucs)
{
    double **concs = static_cast<double **>(std::malloc(12 * sizeof(double *)));

    if (concs == nullptr) {
        std::fprintf(stderr, "Failed to allocate concs\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 12; ++i) {
        concs[i] = static_cast<double *>(std::malloc(static_cast<size_t>(num_nucs[i]) * sizeof(double)));
        if (concs[i] == nullptr) {
            std::fprintf(stderr, "Failed to allocate concs[%d]\n", i);
            std::exit(EXIT_FAILURE);
        }
    }

    const double fuel[34] = {
        6.910000000000000e-06, 1.660000000000000e-03, 2.226000000000000e-02,
        1.890000000000000e-04, 4.573000000000000e-02, 3.447000000000000e-06,
        3.447000000000000e-06, 3.447000000000000e-06, 3.447000000000000e-06,
        3.447000000000000e-06, 3.447000000000000e-06, 3.447000000000000e-06,
        3.447000000000000e-06, 3.447000000000000e-06, 3.447000000000000e-06,
        3.447000000000000e-06, 3.447000000000000e-06, 7.800000000000000e-04,
        7.800000000000000e-04, 7.800000000000000e-04, 7.800000000000000e-04,
        7.800000000000000e-04, 4.500000000000000e-04, 4.500000000000000e-04,
        4.500000000000000e-04, 4.500000000000000e-04, 4.500000000000000e-04,
        4.500000000000000e-04, 4.500000000000000e-04, 4.500000000000000e-04,
        4.500000000000000e-04, 4.500000000000000e-04, 4.500000000000000e-04,
        4.500000000000000e-04
    };
    const double clad[5] = {
        4.230000000000000e-02, 2.200000000000000e-02, 1.000000000000000e-02,
        8.000000000000000e-03, 5.000000000000000e-03
    };
    const double water[4] = {
        6.660000000000000e-02, 3.330000000000000e-02,
        1.000000000000000e-05, 1.000000000000000e-05
    };
    const double steel27[27] = {
        5.900000000000000e-02, 9.000000000000000e-04, 8.000000000000000e-04,
        7.000000000000000e-04, 6.000000000000000e-04, 5.000000000000000e-04,
        4.000000000000000e-04, 3.000000000000000e-04, 2.000000000000000e-04,
        1.000000000000000e-04, 9.000000000000000e-05, 8.000000000000000e-05,
        7.000000000000000e-05, 6.000000000000000e-05, 5.000000000000000e-05,
        4.000000000000000e-05, 3.000000000000000e-05, 2.000000000000000e-05,
        1.000000000000000e-05, 9.000000000000000e-06, 8.000000000000000e-06,
        7.000000000000000e-06, 6.000000000000000e-06, 5.000000000000000e-06,
        4.000000000000000e-06, 3.000000000000000e-06, 2.000000000000000e-06
    };
    const double refl21[21] = {
        3.330000000000000e-02, 6.660000000000000e-02, 1.000000000000000e-05,
        1.000000000000000e-05, 7.000000000000000e-04, 6.000000000000000e-04,
        5.000000000000000e-04, 4.000000000000000e-04, 3.000000000000000e-04,
        2.000000000000000e-04, 1.000000000000000e-04, 9.000000000000000e-05,
        8.000000000000000e-05, 7.000000000000000e-05, 6.000000000000000e-05,
        5.000000000000000e-05, 4.000000000000000e-05, 3.000000000000000e-05,
        2.000000000000000e-05, 1.000000000000000e-05, 9.000000000000000e-06
    };
    const double plate9[9] = {
        3.330000000000000e-02, 6.660000000000000e-02, 1.000000000000000e-05,
        1.000000000000000e-05, 7.000000000000000e-04, 6.000000000000000e-04,
        5.000000000000000e-04, 4.000000000000000e-04, 3.000000000000000e-04
    };

    const double *source[12] = {
        fuel, clad, water, water, steel27, refl21, refl21, refl21, refl21, plate9, plate9, plate9
    };

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < num_nucs[i]; ++j) {
            concs[i][j] = source[i][j];
        }
    }

    return concs;
}

void free_material_data(int *num_nucs, int **mats, double **concs)
{
    if (num_nucs != nullptr) {
        for (int i = 0; i < 12; ++i) {
            if (mats != nullptr) {
                std::free(mats[i]);
            }
            if (concs != nullptr) {
                std::free(concs[i]);
            }
        }
    }

    std::free(mats);
    std::free(concs);
    std::free(num_nucs);
}

__global__ void initialize_kernel(float *a, float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;

    if (idx < total) {
        int row = idx / n;
        int col = idx - row * n;
        float value = static_cast<float>((row + 1) * (col + 1)) * 0.000001f;
        a[idx] = value;
        b[idx] = value;
    }
}

__global__ void jacobi_kernel(const float *__restrict__ input, float *__restrict__ output, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
        int idx = row * n + col;
        output[idx] = 0.25f * (input[idx - 1] + input[idx + 1] + input[idx - n] + input[idx + n]);
    }
}

float *run_simulation(int n, int iterations)
{
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t bytes = total * sizeof(float);

    float *d_a = nullptr;
    float *d_b = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));

    int init_threads = 256;
    int init_blocks = static_cast<int>((total + static_cast<size_t>(init_threads) - 1) / static_cast<size_t>(init_threads));
    initialize_kernel<<<init_blocks, init_threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    for (int iter = 0; iter < iterations; ++iter) {
        jacobi_kernel<<<grid, block>>>(d_a, d_b, n);
        CUDA_CHECK(cudaGetLastError());

        float *tmp = d_a;
        d_a = d_b;
        d_b = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_b));

    return d_a;
}

double compute_checksum(const float *d_data, int n)
{
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    float *h_data = static_cast<float *>(std::malloc(total * sizeof(float)));

    if (h_data == nullptr) {
        std::fprintf(stderr, "Failed to allocate host checksum buffer\n");
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(h_data, d_data, total * sizeof(float), cudaMemcpyDeviceToHost));

    double checksum = 0.0;

    for (size_t i = 0; i < total; ++i) {
        checksum += static_cast<double>(h_data[i]);
    }

    std::free(h_data);

    return checksum;
}

int main(int argc, char **argv)
{
    CLIOptions options = read_CLI(argc, argv);

    if (options.help) {
        std::printf("Usage: %s [-n size] [-i iterations]\n", argv[0]);
        return 0;
    }

    int *num_nucs = load_num_nucs();
    int **mats = load_mats(num_nucs, 68);
    double **concs = load_concs(num_nucs);

    float *d_result = run_simulation(options.n, options.iterations);
    double checksum = compute_checksum(d_result, options.n);

    CUDA_CHECK(cudaFree(d_result));

    std::printf("Checksum: %.12e\n", checksum);

    free_material_data(num_nucs, mats, concs);
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
