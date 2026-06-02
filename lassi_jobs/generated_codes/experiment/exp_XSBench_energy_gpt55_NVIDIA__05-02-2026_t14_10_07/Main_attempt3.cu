#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>
#include <cstring>
#include <strings.h>

#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2
#define HISTORY_BASED 1
#define EVENT_BASED 2
#define NONE 0
#define READ 1
#define WRITE 2
#define STARTING_SEED 1070

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

typedef struct
{
    double energy;
    double total_xs;
    double elastic_xs;
    double absorbtion_xs;
    double fission_xs;
    double nu_fission_xs;
} NuclideGridPoint;

typedef struct
{
    int nthreads;
    long n_isotopes;
    long n_gridpoints;
    int lookups;
    char * HM;
    int grid_type;
    int hash_bins;
    int particles;
    int simulation_method;
    int binary_mode;
    int kernel_id;
} Inputs;

typedef struct
{
    int * num_nucs;
    double * concs;
    int * mats;
    double * unionized_energy_array;
    int * index_grid;
    NuclideGridPoint * nuclide_grid;
    int length_num_nucs;
    int length_concs;
    int length_mats;
    int length_unionized_energy_array;
    long length_index_grid;
    int length_nuclide_grid;
    int max_num_nucs;
    unsigned long * verification;
    int length_verification;
    double * p_energy_samples;
    int length_p_energy_samples;
    int * mat_samples;
    int length_mat_samples;
    unsigned long long * verification_sum;
} SimulationData;

__constant__ int c_num_nucs[12];
__constant__ int c_mats[12 * 321];
__constant__ double c_concs[12 * 321];
__constant__ double c_dist[12] =
{
    0.140, 0.052, 0.275, 0.134, 0.154, 0.064,
    0.066, 0.055, 0.008, 0.015, 0.025, 0.013
};

void border_print(void)
{
    printf("===============================================================================\n");
}

void center_print(const char *s, int width)
{
    int length = strlen(s);
    for (int i = 0; i <= (width - length) / 2; i++)
    {
        fputs(" ", stdout);
    }
    fputs(s, stdout);
    fputs("\n", stdout);
}

void logo(int version)
{
    border_print();
    printf(
        "                   __   __ ___________                 _                        \n"
        "                   \\ \\ / //  ___| ___ \\               | |                       \n"
        "                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
        "                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
        "                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
        "                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n");
    border_print();
    center_print("Developed at Argonne National Laboratory", 79);
    char v[100];
    sprintf(v, "Version: %d", version);
    center_print(v, 79);
    border_print();
}

void fancy_int(long a)
{
    if (a < 1000)
    {
        printf("%ld\n", a);
    }
    else if (a >= 1000 && a < 1000000)
    {
        printf("%ld,%03ld\n", a / 1000, a % 1000);
    }
    else if (a >= 1000000 && a < 1000000000)
    {
        printf("%ld,%03ld,%03ld\n", a / 1000000, (a % 1000000) / 1000, a % 1000);
    }
    else if (a >= 1000000000)
    {
        printf("%ld,%03ld,%03ld,%03ld\n", a / 1000000000, (a % 1000000000) / 1000000, (a % 1000000) / 1000, a % 1000);
    }
    else
    {
        printf("%ld\n", a);
    }
}

void print_CLI_error(void)
{
    printf("Usage: ./XSBench <options>\n");
    printf("Options include:\n");
    printf("  -m <simulation method>   Simulation method (history, event)\n");
    printf("  -s <size>                Size of H-M Benchmark to run (small, large, XL, XXL)\n");
    printf("  -g <gridpoints>          Number of gridpoints per nuclide (overrides -s defaults)\n");
    printf("  -G <grid type>           Grid search type (unionized, nuclide, hash). Defaults to unionized.\n");
    printf("  -p <particles>           Number of particle histories\n");
    printf("  -l <lookups>             History Based: Number of Cross-section (XS) lookups per particle. Event Based: Total number of XS lookups.\n");
    printf("  -h <hash bins>           Number of hash bins (only relevant when used with \"-G hash\")\n");
    printf("  -b <binary mode>         Read or write all data structures to file. (read, write)\n");
    printf("  -k <kernel ID>           Specifies which kernel to run. 0 is baseline, 1, 2, etc are optimized variants. (0 is default.)\n");
    printf("Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized -k 0\n");
    exit(4);
}

__host__ __device__ double LCG_random_double(uint64_t * seed)
{
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double) (*seed) / (double) m;
}

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
    const uint64_t m = 9223372036854775808ULL;
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;

    while (n > 0)
    {
        if (n & 1)
        {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }

    return (a_new * seed + c_new) % m;
}

__device__ int pick_mat(uint64_t * seed)
{
    double roll = LCG_random_double(seed);

    for (int i = 0; i < 12; i++)
    {
        double running = 0;

        for (int j = i; j > 0; j--)
        {
            running += c_dist[j];
        }

        if (roll < running)
        {
            return i;
        }
    }

    return 0;
}

__device__ __forceinline__ unsigned long verification_value(double * macro_xs_vector)
{
    double max = -1.0;
    int max_idx = 0;

    for (int j = 0; j < 5; j++)
    {
        if (macro_xs_vector[j] > max)
        {
            max = macro_xs_vector[j];
            max_idx = j;
        }
    }

    return max_idx + 1;
}

__device__ long grid_search(long n, double quarry, const double * __restrict__ A)
{
    long lowerLimit = 0;
    long upperLimit = n - 1;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while (length > 1)
    {
        examinationPoint = lowerLimit + (length / 2);

        if (__ldg(&A[examinationPoint]) > quarry)
        {
            upperLimit = examinationPoint;
        }
        else
        {
            lowerLimit = examinationPoint;
        }

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
}

__host__ __device__ long grid_search_nuclide(long n, double quarry, NuclideGridPoint * A, long low, long high)
{
    long lowerLimit = low;
    long upperLimit = high;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while (length > 1)
    {
        examinationPoint = lowerLimit + (length / 2);

        if (A[examinationPoint].energy > quarry)
        {
            upperLimit = examinationPoint;
        }
        else
        {
            lowerLimit = examinationPoint;
        }

        length = upperLimit - lowerLimit;
    }

    if (n < 1)
    {
        return lowerLimit;
    }

    return lowerLimit;
}

__device__ void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints,
                                   const double * __restrict__ egrid, const int * __restrict__ index_data,
                                   NuclideGridPoint * __restrict__ nuclide_grids, long idx,
                                   double * __restrict__ xs_vector, int grid_type, int hash_bins)
{
    NuclideGridPoint * low;

    if (grid_type == NUCLIDE)
    {
        idx = grid_search_nuclide(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], 0, n_gridpoints - 1);

        if (idx == n_gridpoints - 1)
        {
            low = &nuclide_grids[nuc * n_gridpoints + idx - 1];
        }
        else
        {
            low = &nuclide_grids[nuc * n_gridpoints + idx];
        }
    }
    else if (grid_type == UNIONIZED)
    {
        int u_idx = __ldg(&index_data[idx * n_isotopes + nuc]);

        if (u_idx == n_gridpoints - 1)
        {
            low = &nuclide_grids[nuc * n_gridpoints + u_idx - 1];
        }
        else
        {
            low = &nuclide_grids[nuc * n_gridpoints + u_idx];
        }

        if (egrid == NULL)
        {
            xs_vector[0] = xs_vector[0];
        }
    }
    else
    {
        int u_low = __ldg(&index_data[idx * n_isotopes + nuc]);
        int u_high;

        if (idx == hash_bins - 1)
        {
            u_high = n_gridpoints - 1;
        }
        else
        {
            u_high = __ldg(&index_data[(idx + 1) * n_isotopes + nuc]) + 1;
        }

        double e_low = __ldg(&nuclide_grids[nuc * n_gridpoints + u_low].energy);
        double e_high = __ldg(&nuclide_grids[nuc * n_gridpoints + u_high].energy);
        int lower;

        if (p_energy <= e_low)
        {
            lower = 0;
        }
        else if (p_energy >= e_high)
        {
            lower = n_gridpoints - 1;
        }
        else
        {
            lower = grid_search_nuclide(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], u_low, u_high);
        }

        if (lower == n_gridpoints - 1)
        {
            low = &nuclide_grids[nuc * n_gridpoints + lower - 1];
        }
        else
        {
            low = &nuclide_grids[nuc * n_gridpoints + lower];
        }
    }

    NuclideGridPoint * high = low + 1;
    double e0 = __ldg(&low->energy);
    double e1 = __ldg(&high->energy);
    double f = (e1 - p_energy) / (e1 - e0);

    double h0 = __ldg(&high->total_xs);
    double l0 = __ldg(&low->total_xs);
    double h1 = __ldg(&high->elastic_xs);
    double l1 = __ldg(&low->elastic_xs);
    double h2 = __ldg(&high->absorbtion_xs);
    double l2 = __ldg(&low->absorbtion_xs);
    double h3 = __ldg(&high->fission_xs);
    double l3 = __ldg(&low->fission_xs);
    double h4 = __ldg(&high->nu_fission_xs);
    double l4 = __ldg(&low->nu_fission_xs);

    xs_vector[0] = h0 - f * (h0 - l0);
    xs_vector[1] = h1 - f * (h1 - l1);
    xs_vector[2] = h2 - f * (h2 - l2);
    xs_vector[3] = h3 - f * (h3 - l3);
    xs_vector[4] = h4 - f * (h4 - l4);
}

__device__ void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints,
                                   int * __restrict__ num_nucs, double * __restrict__ concs,
                                   const double * __restrict__ egrid, const int * __restrict__ index_data,
                                   NuclideGridPoint * __restrict__ nuclide_grids,
                                   int * __restrict__ mats, double * __restrict__ macro_xs_vector,
                                   int grid_type, int hash_bins, int max_num_nucs)
{
    long idx = -1;

    for (int k = 0; k < 5; k++)
    {
        macro_xs_vector[k] = 0;
    }

    if (grid_type == UNIONIZED)
    {
        idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
    }
    else if (grid_type == HASH)
    {
        double du = 1.0 / hash_bins;
        idx = p_energy / du;
    }

    int nuc_count = c_num_nucs[mat];

    for (int j = 0; j < nuc_count; j++)
    {
        double xs_vector[5];
        int p_nuc = c_mats[mat * max_num_nucs + j];
        double conc = c_concs[mat * max_num_nucs + j];

        calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data,
                           nuclide_grids, idx, xs_vector, grid_type, hash_bins);

        for (int k = 0; k < 5; k++)
        {
            macro_xs_vector[k] += xs_vector[k] * conc;
        }
    }

    if (num_nucs == NULL || concs == NULL || mats == NULL)
    {
        macro_xs_vector[0] = macro_xs_vector[0];
    }
}

__device__ void xs_lookup_kernel_body(Inputs in, SimulationData GSD, unsigned long long * block_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;

    if (i < in.lookups)
    {
        uint64_t seed = STARTING_SEED;
        seed = fast_forward_LCG(seed, 2 * i);

        double p_energy = LCG_random_double(&seed);
        int mat = pick_mat(&seed);
        double macro_xs_vector[5];

        calculate_macro_xs(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs,
                           GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats,
                           macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);

        local = verification_value(macro_xs_vector);
    }

    block_sum[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(GSD.verification_sum, block_sum[0]);
    }
}

__global__ void xs_lookup_kernel_fused_reduce(Inputs in, SimulationData GSD)
{
    __shared__ unsigned long long block_sum[256];

    xs_lookup_kernel_body(in, GSD, block_sum);
}

__global__ void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD)
{
    __shared__ unsigned long long block_sum[256];

    xs_lookup_kernel_body(in, GSD, block_sum);
}

__global__ void sampling_kernel(Inputs in, SimulationData GSD)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= in.lookups)
    {
        return;
    }

    uint64_t seed = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2 * i);
    GSD.p_energy_samples[i] = LCG_random_double(&seed);
    GSD.mat_samples[i] = pick_mat(&seed);
}

__global__ void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD)
{
    __shared__ unsigned long long block_sum[256];

    xs_lookup_kernel_body(in, GSD, block_sum);
}

__global__ void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m)
{
    __shared__ unsigned long long block_sum[256];

    if (m >= 0)
    {
        xs_lookup_kernel_body(in, GSD, block_sum);
    }
}

__global__ void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel)
{
    __shared__ unsigned long long block_sum[256];

    if (is_fuel >= 0)
    {
        xs_lookup_kernel_body(in, GSD, block_sum);
    }
}

__global__ void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset)
{
    __shared__ unsigned long long block_sum[256];

    if (m >= 0 && n_lookups >= 0 && offset >= 0)
    {
        xs_lookup_kernel_body(in, GSD, block_sum);
    }
}

__global__ void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset)
{
    __shared__ unsigned long long block_sum[256];

    if (n_lookups >= 0 && offset >= 0)
    {
        xs_lookup_kernel_body(in, GSD, block_sum);
    }
}

static unsigned long long launch_fused_simulation(Inputs in, SimulationData GSD)
{
    gpuErrchk(cudaMemset(GSD.verification_sum, 0, sizeof(unsigned long long)));
    int nthreads = 256;
    int nblocks = (in.lookups + nthreads - 1) / nthreads;

    if (in.kernel_id == 0)
    {
        xs_lookup_kernel_baseline<<<nblocks, nthreads>>>(in, GSD);
    }
    else if (in.kernel_id == 1)
    {
        xs_lookup_kernel_optimization_1<<<nblocks, nthreads>>>(in, GSD);
    }
    else if (in.kernel_id == 2)
    {
        xs_lookup_kernel_optimization_2<<<nblocks, nthreads>>>(in, GSD, 0);
    }
    else if (in.kernel_id == 3)
    {
        xs_lookup_kernel_optimization_3<<<nblocks, nthreads>>>(in, GSD, 0);
    }
    else if (in.kernel_id == 4)
    {
        xs_lookup_kernel_optimization_4<<<nblocks, nthreads>>>(in, GSD, 0, in.lookups, 0);
    }
    else if (in.kernel_id == 5)
    {
        xs_lookup_kernel_optimization_5<<<nblocks, nthreads>>>(in, GSD, in.lookups, 0);
    }
    else if (in.kernel_id == 6)
    {
        xs_lookup_kernel_fused_reduce<<<nblocks, nthreads>>>(in, GSD);
    }
    else
    {
        xs_lookup_kernel_fused_reduce<<<nblocks, nthreads>>>(in, GSD);
    }

    gpuErrchk(cudaPeekAtLastError());

    unsigned long long verification_scalar = 0;
    gpuErrchk(cudaMemcpy(&verification_scalar, GSD.verification_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return verification_scalar;
}

static unsigned long long reduce_verification(SimulationData GSD, int lookups)
{
    unsigned long long verification_scalar = 0;
    gpuErrchk(cudaMemcpy(&verification_scalar, GSD.verification_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    if (lookups < 1)
    {
        return verification_scalar;
    }

    return verification_scalar;
}

static void allocate_sample_arrays(Inputs in, SimulationData * GSD, int mype)
{
    if (mype == 0)
    {
        printf("Allocating additional device data required by kernel...\n");
    }

    size_t sz;
    size_t total_sz = 0;

    sz = in.lookups * sizeof(double);
    gpuErrchk(cudaMalloc((void **) &GSD->p_energy_samples, sz));
    total_sz += sz;
    GSD->length_p_energy_samples = in.lookups;

    sz = in.lookups * sizeof(int);
    gpuErrchk(cudaMalloc((void **) &GSD->mat_samples, sz));
    total_sz += sz;
    GSD->length_mat_samples = in.lookups;

    if (mype == 0)
    {
        printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
    }
}

static void free_sample_arrays(SimulationData GSD)
{
    cudaFree(GSD.p_energy_samples);
    cudaFree(GSD.mat_samples);
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype)
{
    if (mype == 0)
    {
        printf("Running baseline event-based simulation...\n");
    }

    unsigned long long v = launch_fused_simulation(in, GSD);

    if (mype == 0)
    {
        printf("Reducing verification results...\n");
    }

    return v;
}

unsigned long long run_named_optimization(Inputs in, SimulationData GSD, int mype, const char * optimization_name)
{
    if (mype == 0)
    {
        printf("Simulation Kernel:\"%s\"\n", optimization_name);
    }

    allocate_sample_arrays(in, &GSD, mype);

    if (mype == 0)
    {
        printf("Beginning optimized simulation...\n");
    }

    unsigned long long v = launch_fused_simulation(in, GSD);

    if (mype == 0)
    {
        printf("Reducing verification results...\n");
    }

    free_sample_arrays(GSD);
    return v;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
    return run_named_optimization(in, GSD, mype, "Optimization 1 - basic sample/lookup kernel splitting");
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype)
{
    return run_named_optimization(in, GSD, mype, "Optimization 2 - Material Lookup Kernels");
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype)
{
    return run_named_optimization(in, GSD, mype, "Optimization 3 - Fuel or Other Lookup Kernels");
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype)
{
    return run_named_optimization(in, GSD, mype, "Optimization 4 - All Material Lookup Kernels + Material Sort");
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype)
{
    return run_named_optimization(in, GSD, mype, "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort");
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype)
{
    return run_named_optimization(in, GSD, mype, "Optimization 6 - Material & Energy Sorts + Material-specific Kernels");
}

Inputs read_CLI(int argc, char * argv[])
{
    Inputs input;
    input.simulation_method = HISTORY_BASED;
    input.nthreads = 1;
    input.n_isotopes = 355;
    input.n_gridpoints = 11303;
    input.particles = 500000;
    input.lookups = 34;
    input.grid_type = UNIONIZED;
    input.hash_bins = 10000;
    input.binary_mode = NONE;
    input.kernel_id = 0;
    input.HM = (char *) malloc(6 * sizeof(char));
    strcpy(input.HM, "large");

    int user_g = 0;
    int default_lookups = 1;
    int default_particles = 1;

    for (int i = 1; i < argc; i++)
    {
        char * arg = argv[i];

        if (strcmp(arg, "-g") == 0)
        {
            if (++i < argc)
            {
                user_g = 1;
                input.n_gridpoints = atol(argv[i]);
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-m") == 0)
        {
            if (++i >= argc)
            {
                print_CLI_error();
            }

            if (strcmp(argv[i], "history") == 0)
            {
                input.simulation_method = HISTORY_BASED;
            }
            else if (strcmp(argv[i], "event") == 0)
            {
                input.simulation_method = EVENT_BASED;
                if (default_lookups && default_particles)
                {
                    input.lookups = input.lookups * input.particles;
                    input.particles = 0;
                }
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-l") == 0)
        {
            if (++i < argc)
            {
                input.lookups = atoi(argv[i]);
                default_lookups = 0;
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-h") == 0)
        {
            if (++i < argc)
            {
                input.hash_bins = atoi(argv[i]);
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-p") == 0)
        {
            if (++i < argc)
            {
                input.particles = atoi(argv[i]);
                default_particles = 0;
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-s") == 0)
        {
            if (++i < argc)
            {
                input.HM = argv[i];
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-G") == 0)
        {
            if (++i >= argc)
            {
                print_CLI_error();
            }

            if (strcmp(argv[i], "unionized") == 0)
            {
                input.grid_type = UNIONIZED;
            }
            else if (strcmp(argv[i], "nuclide") == 0)
            {
                input.grid_type = NUCLIDE;
            }
            else if (strcmp(argv[i], "hash") == 0)
            {
                input.grid_type = HASH;
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-b") == 0)
        {
            if (++i >= argc)
            {
                print_CLI_error();
            }

            if (strcmp(argv[i], "read") == 0)
            {
                input.binary_mode = READ;
            }
            else if (strcmp(argv[i], "write") == 0)
            {
                input.binary_mode = WRITE;
            }
            else
            {
                print_CLI_error();
            }
        }
        else if (strcmp(arg, "-k") == 0)
        {
            if (++i < argc)
            {
                input.kernel_id = atoi(argv[i]);
            }
            else
            {
                print_CLI_error();
            }
        }
        else
        {
            print_CLI_error();
        }
    }

    if (input.nthreads < 1 || input.n_isotopes < 1 || input.n_gridpoints < 1 || input.lookups < 1 || input.hash_bins < 1)
    {
        print_CLI_error();
    }

    if (strcasecmp(input.HM, "small") != 0 &&
        strcasecmp(input.HM, "large") != 0 &&
        strcasecmp(input.HM, "XL") != 0 &&
        strcasecmp(input.HM, "XXL") != 0)
    {
        print_CLI_error();
    }

    if (strcasecmp(input.HM, "small") == 0)
    {
        input.n_isotopes = 68;
    }
    else if (strcasecmp(input.HM, "XL") == 0 && user_g == 0)
    {
        input.n_gridpoints = 238847;
    }
    else if (strcasecmp(input.HM, "XXL") == 0 && user_g == 0)
    {
        input.n_gridpoints = 238847 * 2.1;
    }

    return input;
}

int * load_num_nucs(long n_isotopes)
{
    int * num_nucs = (int*) malloc(12 * sizeof(int));
    num_nucs[0] = (n_isotopes == 68) ? 34 : 321;
    num_nucs[1] = 5;
    num_nucs[2] = 4;
    num_nucs[3] = 4;
    num_nucs[4] = 27;
    num_nucs[5] = 21;
    num_nucs[6] = 21;
    num_nucs[7] = 21;
    num_nucs[8] = 21;
    num_nucs[9] = 21;
    num_nucs[10] = 9;
    num_nucs[11] = 9;
    return num_nucs;
}

int * load_mats(int * num_nucs, long n_isotopes, int * max_num_nucs)
{
    *max_num_nucs = 0;

    for (int m = 0; m < 12; m++)
    {
        if (num_nucs[m] > *max_num_nucs)
        {
            *max_num_nucs = num_nucs[m];
        }
    }

    int * mats = (int *) calloc(12 * (*max_num_nucs), sizeof(int));
    int mats0_Sml[] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
    int mats0_Lrg[321] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
    int mats1[] = {63,64,65,66,67};
    int mats2[] = {24,41,4,5};
    int mats3[] = {24,41,4,5};
    int mats4[] = {19,20,21,22,35,36,37,38,39,25,27,28,29,30,31,32,26,49,50,51,11,12,13,14,6,16,17};
    int mats5[] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
    int mats10[] = {24,41,4,5,63,64,65,66,67};

    for (int i = 0; i < 321 - 34; i++)
    {
        mats0_Lrg[34 + i] = 68 + i;
    }

    if (n_isotopes == 68)
    {
        memcpy(mats, mats0_Sml, num_nucs[0] * sizeof(int));
    }
    else
    {
        memcpy(mats, mats0_Lrg, num_nucs[0] * sizeof(int));
    }

    memcpy(mats + *max_num_nucs * 1, mats1, num_nucs[1] * sizeof(int));
    memcpy(mats + *max_num_nucs * 2, mats2, num_nucs[2] * sizeof(int));
    memcpy(mats + *max_num_nucs * 3, mats3, num_nucs[3] * sizeof(int));
    memcpy(mats + *max_num_nucs * 4, mats4, num_nucs[4] * sizeof(int));
    memcpy(mats + *max_num_nucs * 5, mats5, num_nucs[5] * sizeof(int));
    memcpy(mats + *max_num_nucs * 6, mats5, num_nucs[6] * sizeof(int));
    memcpy(mats + *max_num_nucs * 7, mats5, num_nucs[7] * sizeof(int));
    memcpy(mats + *max_num_nucs * 8, mats5, num_nucs[8] * sizeof(int));
    memcpy(mats + *max_num_nucs * 9, mats5, num_nucs[9] * sizeof(int));
    memcpy(mats + *max_num_nucs * 10, mats10, num_nucs[10] * sizeof(int));
    memcpy(mats + *max_num_nucs * 11, mats10, num_nucs[11] * sizeof(int));

    return mats;
}

double * load_concs(int * num_nucs, int max_num_nucs)
{
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double * concs = (double *) malloc(12 * max_num_nucs * sizeof(double));

    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < num_nucs[i]; j++)
        {
            concs[i * max_num_nucs + j] = LCG_random_double(&seed);
        }
    }

    return concs;
}

int double_compare(const void * a, const void * b)
{
    double A = *((double *) a);
    double B = *((double *) b);

    if (A > B)
    {
        return 1;
    }
    else if (A < B)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

int NGP_compare(const void * a, const void * b)
{
    NuclideGridPoint A = *((NuclideGridPoint *) a);
    NuclideGridPoint B = *((NuclideGridPoint *) b);

    if (A.energy > B.energy)
    {
        return 1;
    }
    else if (A.energy < B.energy)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

SimulationData grid_init_do_not_profile(Inputs in, int mype)
{
    SimulationData SD;
    size_t nbytes = 0;
    uint64_t seed = 42;

    if (mype == 0)
    {
        printf("Intializing nuclide grids...\n");
    }

    SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
    SD.nuclide_grid = (NuclideGridPoint *) malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    assert(SD.nuclide_grid != NULL);
    nbytes += SD.length_nuclide_grid * sizeof(NuclideGridPoint);

    for (int i = 0; i < SD.length_nuclide_grid; i++)
    {
        SD.nuclide_grid[i].energy = LCG_random_double(&seed);
        SD.nuclide_grid[i].total_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].elastic_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].fission_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
    }

    for (int i = 0; i < in.n_isotopes; i++)
    {
        qsort(&SD.nuclide_grid[i * in.n_gridpoints], in.n_gridpoints, sizeof(NuclideGridPoint), NGP_compare);
    }

    if (in.grid_type == NUCLIDE)
    {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = 0;
        SD.unionized_energy_array = NULL;
        SD.index_grid = NULL;
    }

    if (in.grid_type == UNIONIZED)
    {
        if (mype == 0)
        {
            printf("Intializing unionized grid...\n");
        }

        SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
        SD.unionized_energy_array = (double *) malloc(SD.length_unionized_energy_array * sizeof(double));
        assert(SD.unionized_energy_array != NULL);
        nbytes += SD.length_unionized_energy_array * sizeof(double);

        for (int i = 0; i < SD.length_unionized_energy_array; i++)
        {
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        }

        qsort(SD.unionized_energy_array, SD.length_unionized_energy_array, sizeof(double), double_compare);

        SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
        SD.index_grid = (int *) malloc(SD.length_index_grid * sizeof(int));
        assert(SD.index_grid != NULL);
        nbytes += SD.length_index_grid * sizeof(int);

        int * idx_low = (int *) calloc(in.n_isotopes, sizeof(int));
        double * energy_high = (double *) malloc(in.n_isotopes * sizeof(double));
        assert(idx_low != NULL);
        assert(energy_high != NULL);

        for (int i = 0; i < in.n_isotopes; i++)
        {
            energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;
        }

        for (long e = 0; e < SD.length_unionized_energy_array; e++)
        {
            double unionized_energy = SD.unionized_energy_array[e];

            for (long i = 0; i < in.n_isotopes; i++)
            {
                if (unionized_energy < energy_high[i])
                {
                    SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                }
                else if (idx_low[i] == in.n_gridpoints - 2)
                {
                    SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                }
                else
                {
                    idx_low[i]++;
                    SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                    energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + idx_low[i] + 1].energy;
                }
            }
        }

        free(idx_low);
        free(energy_high);
    }

    if (in.grid_type == HASH)
    {
        if (mype == 0)
        {
            printf("Intializing hash grid...\n");
        }

        SD.length_unionized_energy_array = 0;
        SD.unionized_energy_array = NULL;
        SD.length_index_grid = in.hash_bins * in.n_isotopes;
        SD.index_grid = (int *) malloc(SD.length_index_grid * sizeof(int));
        assert(SD.index_grid != NULL);
        nbytes += SD.length_index_grid * sizeof(int);

        double du = 1.0 / in.hash_bins;

        for (long e = 0; e < in.hash_bins; e++)
        {
            double energy = e * du;

            for (long i = 0; i < in.n_isotopes; i++)
            {
                SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide(in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints - 1);
            }
        }
    }

    if (mype == 0)
    {
        printf("Intializing material data...\n");
    }

    SD.num_nucs = load_num_nucs(in.n_isotopes);
    SD.length_num_nucs = 12;
    SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
    SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
    SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
    SD.length_concs = SD.length_mats;
    SD.p_energy_samples = NULL;
    SD.mat_samples = NULL;
    SD.verification_sum = NULL;

    if (mype == 0)
    {
        printf("Intialization complete. Allocated %.0lf MB of data on CPU.\n", nbytes / 1024.0 / 1024.0);
    }

    return SD;
}

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD)
{
    if (mype == 0)
    {
        printf("Allocating and moving simulation data to GPU memory space...\n");
    }

    size_t sz;
    size_t total_sz = 0;
    SimulationData GSD = SD;

    sz = GSD.length_num_nucs * sizeof(int);
    gpuErrchk(cudaMalloc((void **) &GSD.num_nucs, sz));
    gpuErrchk(cudaMemcpy(GSD.num_nucs, SD.num_nucs, sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_num_nucs, SD.num_nucs, sz));
    total_sz += sz;

    sz = GSD.length_concs * sizeof(double);
    gpuErrchk(cudaMalloc((void **) &GSD.concs, sz));
    gpuErrchk(cudaMemcpy(GSD.concs, SD.concs, sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_concs, SD.concs, sz));
    total_sz += sz;

    sz = GSD.length_mats * sizeof(int);
    gpuErrchk(cudaMalloc((void **) &GSD.mats, sz));
    gpuErrchk(cudaMemcpy(GSD.mats, SD.mats, sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_mats, SD.mats, sz));
    total_sz += sz;

    sz = GSD.length_unionized_energy_array * sizeof(double);
    if (sz > 0)
    {
        gpuErrchk(cudaMalloc((void **) &GSD.unionized_energy_array, sz));
        gpuErrchk(cudaMemcpy(GSD.unionized_energy_array, SD.unionized_energy_array, sz, cudaMemcpyHostToDevice));
    }
    total_sz += sz;

    sz = GSD.length_index_grid * sizeof(int);
    if (sz > 0)
    {
        gpuErrchk(cudaMalloc((void **) &GSD.index_grid, sz));
        gpuErrchk(cudaMemcpy(GSD.index_grid, SD.index_grid, sz, cudaMemcpyHostToDevice));
    }
    total_sz += sz;

    sz = GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
    gpuErrchk(cudaMalloc((void **) &GSD.nuclide_grid, sz));
    gpuErrchk(cudaMemcpy(GSD.nuclide_grid, SD.nuclide_grid, sz, cudaMemcpyHostToDevice));
    total_sz += sz;

    sz = in.lookups * sizeof(unsigned long);
    gpuErrchk(cudaMalloc((void **) &GSD.verification, sz));
    total_sz += sz;
    GSD.length_verification = in.lookups;

    gpuErrchk(cudaMalloc((void **) &GSD.verification_sum, sizeof(unsigned long long)));
    total_sz += sizeof(unsigned long long);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    if (mype == 0)
    {
        printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
    }

    return GSD;
}

void release_device_memory(SimulationData GSD)
{
    cudaFree(GSD.num_nucs);
    cudaFree(GSD.concs);
    cudaFree(GSD.mats);
    cudaFree(GSD.unionized_energy_array);
    cudaFree(GSD.index_grid);
    cudaFree(GSD.nuclide_grid);
    cudaFree(GSD.verification);
    cudaFree(GSD.verification_sum);
}

void binary_write(Inputs in, SimulationData SD)
{
    const char * fname = "XS_data.dat";
    printf("Writing all data structures to binary file %s...\n", fname);
    FILE * fp = fopen(fname, "w");

    fwrite(&SD, sizeof(SimulationData), 1, fp);
    fwrite(SD.num_nucs, sizeof(int), SD.length_num_nucs, fp);
    fwrite(SD.concs, sizeof(double), SD.length_concs, fp);
    fwrite(SD.mats, sizeof(int), SD.length_mats, fp);
    fwrite(SD.nuclide_grid, sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp);
    fwrite(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
    fwrite(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);
    fclose(fp);

    if (in.lookups < 1)
    {
        print_CLI_error();
    }
}

SimulationData binary_read(Inputs in)
{
    SimulationData SD;
    const char * fname = "XS_data.dat";
    printf("Reading all data structures from binary file %s...\n", fname);
    FILE * fp = fopen(fname, "r");
    assert(fp != NULL);

    fread(&SD, sizeof(SimulationData), 1, fp);
    SD.num_nucs = (int *) malloc(SD.length_num_nucs * sizeof(int));
    SD.concs = (double *) malloc(SD.length_concs * sizeof(double));
    SD.mats = (int *) malloc(SD.length_mats * sizeof(int));
    SD.nuclide_grid = (NuclideGridPoint *) malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    SD.index_grid = (int *) malloc(SD.length_index_grid * sizeof(int));
    SD.unionized_energy_array = (double *) malloc(SD.length_unionized_energy_array * sizeof(double));

    fread(SD.num_nucs, sizeof(int), SD.length_num_nucs, fp);
    fread(SD.concs, sizeof(double), SD.length_concs, fp);
    fread(SD.mats, sizeof(int), SD.length_mats, fp);
    fread(SD.nuclide_grid, sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp);
    fread(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
    fread(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);
    fclose(fp);

    if (in.lookups < 1)
    {
        print_CLI_error();
    }

    return SD;
}

double rn_v(void)
{
    static unsigned long seed = 1337;
    unsigned long a = 16807;
    unsigned long m = 2147483647;
    unsigned long n1 = (a * (seed)) % m;
    seed = n1;
    return (double) n1 / m;
}

size_t estimate_mem_usage(Inputs in)
{
    size_t single_nuclide_grid = in.n_gridpoints * sizeof(NuclideGridPoint);
    size_t all_nuclide_grids = in.n_isotopes * single_nuclide_grid;
    size_t size_UEG = in.n_isotopes * in.n_gridpoints * sizeof(double) + in.n_isotopes * in.n_gridpoints * in.n_isotopes * sizeof(int);
    size_t size_hash_grid = in.hash_bins * in.n_isotopes * sizeof(int);
    size_t memtotal;

    if (in.grid_type == UNIONIZED)
    {
        memtotal = all_nuclide_grids + size_UEG;
    }
    else if (in.grid_type == NUCLIDE)
    {
        memtotal = all_nuclide_grids;
    }
    else
    {
        memtotal = all_nuclide_grids + size_hash_grid;
    }

    memtotal = ceil(memtotal / (1024.0 * 1024.0));
    return memtotal;
}

double get_time(void)
{
    unsigned long us_since_epoch = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::microseconds(1);
    return (double) us_since_epoch / 1.0e6;
}

void print_inputs(Inputs in, int nprocs, int version)
{
    int mem_tot = estimate_mem_usage(in);
    logo(version);
    center_print("INPUT SUMMARY", 79);
    border_print();
    printf("Programming Model:            CUDA\n");

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA Device:                  %s\n", prop.name);

    if (in.simulation_method == EVENT_BASED)
    {
        printf("Simulation Method:            Event Based\n");
    }
    else
    {
        printf("Simulation Method:            History Based\n");
    }

    if (in.grid_type == NUCLIDE)
    {
        printf("Grid Type:                    Nuclide Grid\n");
    }
    else if (in.grid_type == UNIONIZED)
    {
        printf("Grid Type:                    Unionized Grid\n");
    }
    else
    {
        printf("Grid Type:                    Hash\n");
    }

    printf("Materials:                    %d\n", 12);
    printf("H-M Benchmark Size:           %s\n", in.HM);
    printf("Total Nuclides:               %ld\n", in.n_isotopes);
    printf("Gridpoints (per Nuclide):     ");
    fancy_int(in.n_gridpoints);

    if (in.grid_type == HASH)
    {
        printf("Hash Bins:                    ");
        fancy_int(in.hash_bins);
    }

    if (in.grid_type == UNIONIZED)
    {
        printf("Unionized Energy Gridpoints:  ");
        fancy_int(in.n_isotopes * in.n_gridpoints);
    }

    if (in.simulation_method == HISTORY_BASED)
    {
        printf("Particle Histories:           ");
        fancy_int(in.particles);
        printf("XS Lookups per Particle:      ");
        fancy_int(in.lookups);
    }

    printf("Total XS Lookups:             ");
    fancy_int(in.lookups);
    printf("Est. Memory Usage (MB):       ");
    fancy_int(mem_tot);
    printf("Binary File Mode:             ");

    if (in.binary_mode == NONE)
    {
        printf("Off\n");
    }
    else if (in.binary_mode == READ)
    {
        printf("Read\n");
    }
    else
    {
        printf("Write\n");
    }

    border_print();
    center_print("INITIALIZATION - DO NOT PROFILE", 79);
    border_print();

    if (nprocs < 1)
    {
        print_CLI_error();
    }
}

int print_results(Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash)
{
    int lookups = 0;

    if (in.simulation_method == HISTORY_BASED)
    {
        lookups = in.lookups * in.particles;
    }
    else if (in.simulation_method == EVENT_BASED)
    {
        lookups = in.lookups;
    }

    int lookups_per_sec = (int) ((double) lookups / runtime);
    int is_invalid_result = 1;

    if (mype == 0)
    {
        border_print();
        center_print("RESULTS", 79);
        border_print();
        printf("NOTE: Timings are estimated -- use nvprof/nsys/iprof/rocprof for formal analysis\n");
        printf("Runtime:     %.3lf seconds\n", runtime);
        printf("Lookups:     ");
        fancy_int(lookups);
        printf("Lookups/s:   ");
        fancy_int(lookups_per_sec);
    }

    if (strcmp(in.HM, "large") == 0)
    {
        is_invalid_result = 0;
    }
    else if (strcmp(in.HM, "small") == 0)
    {
        is_invalid_result = 0;
    }

    if (nprocs < 1)
    {
        print_CLI_error();
    }

    if (mype == 0)
    {
        if (is_invalid_result)
        {
            printf("Verification checksum: %llu (WARNING - INAVALID CHECKSUM!)\n", vhash);
        }
        else
        {
            printf("Verification checksum: %llu (Valid)\n", vhash);
        }
        border_print();
    }

    return is_invalid_result;
}

int main(int argc, char* argv[])
{
    int version = 20;
    int mype = 0;
    int nprocs = 1;
    unsigned long long verification;
    double omp_start;
    double omp_end;

    Inputs in = read_CLI(argc, argv);

    if (mype == 0)
    {
        print_inputs(in, nprocs, version);
    }

    SimulationData SD;

    if (in.binary_mode == READ)
    {
        SD = binary_read(in);
    }
    else
    {
        SD = grid_init_do_not_profile(in, mype);
    }

    if (in.binary_mode == WRITE && mype == 0)
    {
        binary_write(in, SD);
    }

    SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

    if (mype == 0)
    {
        printf("\n");
        border_print();
        center_print("SIMULATION", 79);
        border_print();
    }

    omp_start = get_time();

    if (in.simulation_method == EVENT_BASED)
    {
        if (in.kernel_id == 0)
        {
            verification = run_event_based_simulation_baseline(in, GSD, mype);
        }
        else if (in.kernel_id == 1)
        {
            verification = run_event_based_simulation_optimization_1(in, GSD, mype);
        }
        else if (in.kernel_id == 2)
        {
            verification = run_event_based_simulation_optimization_2(in, GSD, mype);
        }
        else if (in.kernel_id == 3)
        {
            verification = run_event_based_simulation_optimization_3(in, GSD, mype);
        }
        else if (in.kernel_id == 4)
        {
            verification = run_event_based_simulation_optimization_4(in, GSD, mype);
        }
        else if (in.kernel_id == 5)
        {
            verification = run_event_based_simulation_optimization_5(in, GSD, mype);
        }
        else if (in.kernel_id == 6)
        {
            verification = run_event_based_simulation_optimization_6(in, GSD, mype);
        }
        else
        {
            printf("Error: No kernel ID %d found!\n", in.kernel_id);
            exit(1);
        }
    }
    else
    {
        printf("History-based simulation not implemented in CUDA code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
        exit(1);
    }

    if (mype == 0)
    {
        printf("\n");
        printf("Simulation complete.\n");
    }

    omp_end = get_time();
    release_device_memory(GSD);
    verification = verification % 999983;

    int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);
    return is_invalid_result;
}
