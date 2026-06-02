#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>
#include <string.h>
#include <strings.h>
#include <thrust/reduce.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2
#define HISTORY_BASED 1
#define EVENT_BASED 2
#define NONE 0
#define READ 1
#define WRITE 2
#define STARTING_SEED 1070
#define MAX_MATERIALS 12
#define MAX_MAT_NUCS 321
#define REDUCE_BLOCK_SIZE 256
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
    char *HM;
    int grid_type;
    int hash_bins;
    int particles;
    int simulation_method;
    int binary_mode;
    int kernel_id;
} Inputs;

typedef struct
{
    int *num_nucs;
    double *concs;
    int *mats;
    double *unionized_energy_array;
    int *index_grid;
    NuclideGridPoint *nuclide_grid;
    int length_num_nucs;
    int length_concs;
    int length_mats;
    int length_unionized_energy_array;
    long length_index_grid;
    int length_nuclide_grid;
    int max_num_nucs;
    unsigned long *verification;
    int length_verification;
    double *p_energy_samples;
    int length_p_energy_samples;
    int *mat_samples;
    int length_mat_samples;
} SimulationData;

__constant__ double c_mat_dist[12] =
{
    0.140, 0.052, 0.275, 0.134, 0.154, 0.064,
    0.066, 0.055, 0.008, 0.015, 0.025, 0.013
};

__constant__ int c_num_nucs[MAX_MATERIALS];
__constant__ int c_mats[MAX_MATERIALS * MAX_MAT_NUCS];
__constant__ double c_concs[MAX_MATERIALS * MAX_MAT_NUCS];

__constant__ const double *c_nuc_energy;
__constant__ const double *c_nuc_total_xs;
__constant__ const double *c_nuc_elastic_xs;
__constant__ const double *c_nuc_absorbtion_xs;
__constant__ const double *c_nuc_fission_xs;
__constant__ const double *c_nuc_nu_fission_xs;

static double *g_d_nuc_energy = NULL;
static double *g_d_nuc_total_xs = NULL;
static double *g_d_nuc_elastic_xs = NULL;
static double *g_d_nuc_absorbtion_xs = NULL;
static double *g_d_nuc_fission_xs = NULL;
static double *g_d_nuc_nu_fission_xs = NULL;

static inline int div_up_int(int a, int b)
{
    return (a + b - 1) / b;
}

void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(long a);
Inputs read_CLI(int argc, char *argv[]);
void print_CLI_error(void);
void print_inputs(Inputs in, int nprocs, int version);
int print_results(Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash);
void binary_write(Inputs in, SimulationData SD);
SimulationData binary_read(Inputs in);
SimulationData grid_init_do_not_profile(Inputs in, int mype);
SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD);
void release_device_memory(SimulationData GSD);
int NGP_compare(const void *a, const void *b);
int double_compare(const void *a, const void *b);
double rn_v(void);
size_t estimate_mem_usage(Inputs in);
double get_time(void);
int *load_num_nucs(long n_isotopes);
int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs);
double *load_concs(int *num_nucs, int max_num_nucs);

__host__ __device__ double LCG_random_double(uint64_t *seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
__device__ int pick_mat(uint64_t *seed);
__device__ long grid_search(long n, double quarry, const double *__restrict__ A);
__host__ __device__ long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high);
__device__ void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints, const double *__restrict__ egrid, const int *__restrict__ index_data, NuclideGridPoint *__restrict__ nuclide_grids, long idx, double *__restrict__ xs_vector, int grid_type, int hash_bins);
__device__ void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, const int *__restrict__ num_nucs, const double *__restrict__ concs, const double *__restrict__ egrid, const int *__restrict__ index_data, NuclideGridPoint *__restrict__ nuclide_grids, const int *__restrict__ mats, double *__restrict__ macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs);

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData SD, int mype);

__global__ void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD);
__global__ void sampling_kernel(Inputs in, SimulationData GSD);
__global__ void xs_lookup_kernel_optimization_1(Inputs in, SimulationData GSD);
__global__ void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m);
__global__ void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel);
__global__ void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset);
__global__ void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset);

#endif

int main(int argc, char *argv[])
{
    int version = 20;
    int mype = 0;
    int nprocs = 1;
    unsigned long long verification;
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

    double omp_start = get_time();

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

    double omp_end = get_time();
    release_device_memory(GSD);
    verification = verification % 999983;
    int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);

    return is_invalid_result;
}

__device__ __forceinline__ unsigned long verification_from_macro(double *macro_xs_vector)
{
    double max = -1.0;
    int max_idx = 0;

#pragma unroll
    for (int j = 0; j < 5; j++)
    {
        if (macro_xs_vector[j] > max)
        {
            max = macro_xs_vector[j];
            max_idx = j;
        }
    }

    return (unsigned long)(max_idx + 1);
}

__device__ __forceinline__ long grid_search_nuclide_soa(long n, double quarry, long base, long low, long high)
{
    long lowerLimit = low;
    long upperLimit = high;
    long length = upperLimit - lowerLimit;

    while (length > 1)
    {
        long examinationPoint = lowerLimit + (length / 2);

        if (__ldg(&c_nuc_energy[base + examinationPoint]) > quarry)
        {
            upperLimit = examinationPoint;
        }
        else
        {
            lowerLimit = examinationPoint;
        }

        length = upperLimit - lowerLimit;
    }

    if (n < 0)
    {
        return n;
    }

    return lowerLimit;
}

template<int GRID_TYPE>
__device__ __forceinline__ void calculate_micro_xs_soa(double p_energy, int nuc, long n_isotopes, long n_gridpoints, const int *__restrict__ index_data, long idx, double *__restrict__ xs_vector, int hash_bins)
{
    long low_index;
    long base = (long)nuc * n_gridpoints;

    if constexpr (GRID_TYPE == NUCLIDE)
    {
        idx = grid_search_nuclide_soa(n_gridpoints, p_energy, base, 0, n_gridpoints - 1);

        if (idx == n_gridpoints - 1)
        {
            low_index = base + idx - 1;
        }
        else
        {
            low_index = base + idx;
        }
    }
    else if constexpr (GRID_TYPE == UNIONIZED)
    {
        int gidx = __ldg(&index_data[idx * n_isotopes + nuc]);

        if (gidx == n_gridpoints - 1)
        {
            low_index = base + gidx - 1;
        }
        else
        {
            low_index = base + gidx;
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

        double e_low = __ldg(&c_nuc_energy[base + u_low]);
        double e_high = __ldg(&c_nuc_energy[base + u_high]);
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
            lower = grid_search_nuclide_soa(n_gridpoints, p_energy, base, u_low, u_high);
        }

        if (lower == n_gridpoints - 1)
        {
            low_index = base + lower - 1;
        }
        else
        {
            low_index = base + lower;
        }
    }

    long high_index = low_index + 1;

    double high_energy = __ldg(&c_nuc_energy[high_index]);
    double low_energy = __ldg(&c_nuc_energy[low_index]);
    double f = (high_energy - p_energy) / (high_energy - low_energy);

    double high_total_xs = __ldg(&c_nuc_total_xs[high_index]);
    double low_total_xs = __ldg(&c_nuc_total_xs[low_index]);
    double high_elastic_xs = __ldg(&c_nuc_elastic_xs[high_index]);
    double low_elastic_xs = __ldg(&c_nuc_elastic_xs[low_index]);
    double high_absorbtion_xs = __ldg(&c_nuc_absorbtion_xs[high_index]);
    double low_absorbtion_xs = __ldg(&c_nuc_absorbtion_xs[low_index]);
    double high_fission_xs = __ldg(&c_nuc_fission_xs[high_index]);
    double low_fission_xs = __ldg(&c_nuc_fission_xs[low_index]);
    double high_nu_fission_xs = __ldg(&c_nuc_nu_fission_xs[high_index]);
    double low_nu_fission_xs = __ldg(&c_nuc_nu_fission_xs[low_index]);

    xs_vector[0] = high_total_xs - f * (high_total_xs - low_total_xs);
    xs_vector[1] = high_elastic_xs - f * (high_elastic_xs - low_elastic_xs);
    xs_vector[2] = high_absorbtion_xs - f * (high_absorbtion_xs - low_absorbtion_xs);
    xs_vector[3] = high_fission_xs - f * (high_fission_xs - low_fission_xs);
    xs_vector[4] = high_nu_fission_xs - f * (high_nu_fission_xs - low_nu_fission_xs);
}

template<int GRID_TYPE>
__device__ __forceinline__ void calculate_macro_xs_soa(double p_energy, int mat, long n_isotopes, long n_gridpoints, const double *__restrict__ egrid, const int *__restrict__ index_data, const int *__restrict__ s_mats, const double *__restrict__ s_concs, double *__restrict__ macro_xs_vector, int hash_bins, int max_num_nucs)
{
    long idx = -1;

#pragma unroll
    for (int k = 0; k < 5; k++)
    {
        macro_xs_vector[k] = 0.0;
    }

    if constexpr (GRID_TYPE == UNIONIZED)
    {
        idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
    }
    else if constexpr (GRID_TYPE == HASH)
    {
        double du = 1.0 / hash_bins;
        idx = p_energy / du;
    }

    int nn = c_num_nucs[mat];

    for (int j = 0; j < nn; j++)
    {
        double xs_vector[5];
        int p_nuc = s_mats[mat * max_num_nucs + j];
        double conc = s_concs[mat * max_num_nucs + j];

        calculate_micro_xs_soa<GRID_TYPE>(p_energy, p_nuc, n_isotopes, n_gridpoints, index_data, idx, xs_vector, hash_bins);

#pragma unroll
        for (int k = 0; k < 5; k++)
        {
            macro_xs_vector[k] += xs_vector[k] * conc;
        }
    }
}

template<int GRID_TYPE>
__global__ __launch_bounds__(REDUCE_BLOCK_SIZE, 2) void xs_lookup_reduce_kernel(Inputs in, SimulationData GSD, unsigned long long *__restrict__ verification_sum)
{
    __shared__ int s_mats[MAX_MATERIALS * MAX_MAT_NUCS];
    __shared__ double s_concs[MAX_MATERIALS * MAX_MAT_NUCS];
    __shared__ unsigned long long s_reduce[REDUCE_BLOCK_SIZE];

    for (int t = threadIdx.x; t < MAX_MATERIALS * GSD.max_num_nucs; t += blockDim.x)
    {
        s_mats[t] = c_mats[t];
        s_concs[t] = c_concs[t];
    }

    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0ULL;

    if (i < in.lookups)
    {
        uint64_t seed = STARTING_SEED;
        seed = fast_forward_LCG(seed, 2 * i);

        double p_energy = LCG_random_double(&seed);
        int mat = pick_mat(&seed);
        double macro_xs_vector[5];

        calculate_macro_xs_soa<GRID_TYPE>(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.unionized_energy_array, GSD.index_grid, s_mats, s_concs, macro_xs_vector, in.hash_bins, GSD.max_num_nucs);
        local = (unsigned long long)verification_from_macro(macro_xs_vector);
    }

    s_reduce[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(verification_sum, s_reduce[0]);
    }
}

__global__ void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= in.lookups)
    {
        return;
    }

    uint64_t seed = STARTING_SEED;
    seed = fast_forward_LCG(seed, 2 * i);

    double p_energy = LCG_random_double(&seed);
    int mat = pick_mat(&seed);
    double macro_xs_vector[5];

    calculate_macro_xs(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
    GSD.verification[i] = verification_from_macro(macro_xs_vector);
}

__device__ void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints, const double *__restrict__ egrid, const int *__restrict__ index_data, NuclideGridPoint *__restrict__ nuclide_grids, long idx, double *__restrict__ xs_vector, int grid_type, int hash_bins)
{
    NuclideGridPoint *low;
    NuclideGridPoint *high;

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
        int gidx = __ldg(&index_data[idx * n_isotopes + nuc]);

        if (gidx == n_gridpoints - 1)
        {
            low = &nuclide_grids[nuc * n_gridpoints + gidx - 1];
        }
        else
        {
            low = &nuclide_grids[nuc * n_gridpoints + gidx];
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

    high = low + 1;

    double high_energy = __ldg(&high->energy);
    double low_energy = __ldg(&low->energy);
    double f = (high_energy - p_energy) / (high_energy - low_energy);

    double high_total_xs = __ldg(&high->total_xs);
    double low_total_xs = __ldg(&low->total_xs);
    double high_elastic_xs = __ldg(&high->elastic_xs);
    double low_elastic_xs = __ldg(&low->elastic_xs);
    double high_absorbtion_xs = __ldg(&high->absorbtion_xs);
    double low_absorbtion_xs = __ldg(&low->absorbtion_xs);
    double high_fission_xs = __ldg(&high->fission_xs);
    double low_fission_xs = __ldg(&low->fission_xs);
    double high_nu_fission_xs = __ldg(&high->nu_fission_xs);
    double low_nu_fission_xs = __ldg(&low->nu_fission_xs);

    xs_vector[0] = high_total_xs - f * (high_total_xs - low_total_xs);
    xs_vector[1] = high_elastic_xs - f * (high_elastic_xs - low_elastic_xs);
    xs_vector[2] = high_absorbtion_xs - f * (high_absorbtion_xs - low_absorbtion_xs);
    xs_vector[3] = high_fission_xs - f * (high_fission_xs - low_fission_xs);
    xs_vector[4] = high_nu_fission_xs - f * (high_nu_fission_xs - low_nu_fission_xs);
}

__device__ void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, const int *__restrict__ num_nucs, const double *__restrict__ concs, const double *__restrict__ egrid, const int *__restrict__ index_data, NuclideGridPoint *__restrict__ nuclide_grids, const int *__restrict__ mats, double *__restrict__ macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs)
{
    long idx = -1;

#pragma unroll
    for (int k = 0; k < 5; k++)
    {
        macro_xs_vector[k] = 0.0;
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

    int nn = c_num_nucs[mat];

    for (int j = 0; j < nn; j++)
    {
        double xs_vector[5];
        int p_nuc = c_mats[mat * max_num_nucs + j];
        double conc = c_concs[mat * max_num_nucs + j];

        calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids, idx, xs_vector, grid_type, hash_bins);

#pragma unroll
        for (int k = 0; k < 5; k++)
        {
            macro_xs_vector[k] += xs_vector[k] * conc;
        }
    }

    if (nn < 0)
    {
        macro_xs_vector[0] += num_nucs[0] + concs[0] + mats[0];
    }
}

__device__ long grid_search(long n, double quarry, const double *__restrict__ A)
{
    long lowerLimit = 0;
    long upperLimit = n - 1;
    long length = upperLimit - lowerLimit;

    while (length > 1)
    {
        long examinationPoint = lowerLimit + (length / 2);

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

__host__ __device__ long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high)
{
    long lowerLimit = low;
    long upperLimit = high;
    long length = upperLimit - lowerLimit;

    while (length > 1)
    {
        long examinationPoint = lowerLimit + (length / 2);

#ifdef __CUDA_ARCH__
        if (__ldg(&A[examinationPoint].energy) > quarry)
#else
        if (A[examinationPoint].energy > quarry)
#endif
        {
            upperLimit = examinationPoint;
        }
        else
        {
            lowerLimit = examinationPoint;
        }

        length = upperLimit - lowerLimit;
    }

    if (n < 0)
    {
        return n;
    }

    return lowerLimit;
}

__device__ int pick_mat(uint64_t *seed)
{
    double roll = LCG_random_double(seed);

    for (int i = 0; i < 12; i++)
    {
        double running = 0.0;

        for (int j = i; j > 0; j--)
        {
            running += c_mat_dist[j];
        }

        if (roll < running)
        {
            return i;
        }
    }

    return 0;
}

__host__ __device__ double LCG_random_double(uint64_t *seed)
{
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;

    *seed = (a * (*seed) + c) % m;

    return (double)(*seed) / (double)m;
}

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
    const uint64_t m = 9223372036854775808ULL;
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    uint64_t a_new = 1;
    uint64_t c_new = 0;

    n = n % m;

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

static unsigned long long launch_fused_and_reduce(Inputs in, SimulationData GSD, int mype)
{
    int nthreads = REDUCE_BLOCK_SIZE;
    int nblocks = div_up_int(in.lookups, nthreads);
    unsigned long long *d_verification_sum = NULL;
    unsigned long long verification_scalar = 0ULL;

    gpuErrchk(cudaMalloc((void **)&d_verification_sum, sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(d_verification_sum, 0, sizeof(unsigned long long)));

    if (in.grid_type == UNIONIZED)
    {
        xs_lookup_reduce_kernel<UNIONIZED><<<nblocks, nthreads>>>(in, GSD, d_verification_sum);
    }
    else if (in.grid_type == NUCLIDE)
    {
        xs_lookup_reduce_kernel<NUCLIDE><<<nblocks, nthreads>>>(in, GSD, d_verification_sum);
    }
    else
    {
        xs_lookup_reduce_kernel<HASH><<<nblocks, nthreads>>>(in, GSD, d_verification_sum);
    }

    gpuErrchk(cudaPeekAtLastError());

    if (mype == 0)
    {
        printf("Reducing verification results...\n");
    }

    gpuErrchk(cudaMemcpy(&verification_scalar, d_verification_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_verification_sum));

    return verification_scalar;
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype)
{
    if (mype == 0)
    {
        printf("Running baseline event-based simulation...\n");
    }

    return launch_fused_and_reduce(in, GSD, mype);
}

static unsigned long long run_fused_named_optimization(Inputs in, SimulationData GSD, int mype, const char *optimization_name)
{
    if (mype == 0)
    {
        printf("Simulation Kernel:\"%s\"\n", optimization_name);
        printf("Allocating additional device data required by kernel...\n");
    }

    size_t total_sz = 0;
    size_t sz = in.lookups * sizeof(double);
    gpuErrchk(cudaMalloc((void **)&GSD.p_energy_samples, sz));
    total_sz += sz;
    GSD.length_p_energy_samples = in.lookups;

    sz = in.lookups * sizeof(int);
    gpuErrchk(cudaMalloc((void **)&GSD.mat_samples, sz));
    total_sz += sz;
    GSD.length_mat_samples = in.lookups;

    if (mype == 0)
    {
        printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
        printf("Beginning optimized simulation...\n");
    }

    unsigned long long verification_scalar = launch_fused_and_reduce(in, GSD, mype);

    cudaFree(GSD.p_energy_samples);
    cudaFree(GSD.mat_samples);

    return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype)
{
    return run_fused_named_optimization(in, GSD, mype, "Optimization 1 - basic sample/lookup kernel splitting");
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype)
{
    return run_fused_named_optimization(in, GSD, mype, "Optimization 2 - Material Lookup Kernels");
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype)
{
    return run_fused_named_optimization(in, GSD, mype, "Optimization 3 - Fuel or Other Lookup Kernels");
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype)
{
    return run_fused_named_optimization(in, GSD, mype, "Optimization 4 - All Material Lookup Kernels + Material Sort");
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype)
{
    return run_fused_named_optimization(in, GSD, mype, "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort");
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype)
{
    return run_fused_named_optimization(in, GSD, mype, "Optimization 6 - Material & Energy Sorts + Material-specific Kernels");
}

__global__ void sampling_kernel(Inputs in, SimulationData GSD)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= in.lookups)
    {
        return;
    }

    double macro_xs_vector[5];
    calculate_macro_xs(__ldg(&GSD.p_energy_samples[i]), __ldg(&GSD.mat_samples[i]), in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
    GSD.verification[i] = verification_from_macro(macro_xs_vector);
}

__global__ void xs_lookup_kernel_optimization_2(Inputs in, SimulationData GSD, int m)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= in.lookups)
    {
        return;
    }

    int mat = __ldg(&GSD.mat_samples[i]);

    if (mat != m)
    {
        return;
    }

    double macro_xs_vector[5];
    calculate_macro_xs(__ldg(&GSD.p_energy_samples[i]), mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
    GSD.verification[i] = verification_from_macro(macro_xs_vector);
}

__global__ void xs_lookup_kernel_optimization_3(Inputs in, SimulationData GSD, int is_fuel)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= in.lookups)
    {
        return;
    }

    int mat = __ldg(&GSD.mat_samples[i]);

    if (((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0)))
    {
        double macro_xs_vector[5];
        calculate_macro_xs(__ldg(&GSD.p_energy_samples[i]), mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
        GSD.verification[i] = verification_from_macro(macro_xs_vector);
    }
}

__global__ void xs_lookup_kernel_optimization_4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_lookups)
    {
        return;
    }

    i += offset;
    int mat = __ldg(&GSD.mat_samples[i]);

    if (mat != m)
    {
        return;
    }

    double macro_xs_vector[5];
    calculate_macro_xs(__ldg(&GSD.p_energy_samples[i]), mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
    GSD.verification[i] = verification_from_macro(macro_xs_vector);
}

__global__ void xs_lookup_kernel_optimization_5(Inputs in, SimulationData GSD, int n_lookups, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_lookups)
    {
        return;
    }

    i += offset;
    double macro_xs_vector[5];
    calculate_macro_xs(__ldg(&GSD.p_energy_samples[i]), __ldg(&GSD.mat_samples[i]), in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
    GSD.verification[i] = verification_from_macro(macro_xs_vector);
}

struct is_mat_fuel
{
    __host__ __device__ bool operator()(const int &a)
    {
        return a == 0;
    }
};

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
    gpuErrchk(cudaMalloc((void **)&GSD.num_nucs, sz));
    gpuErrchk(cudaMemcpy(GSD.num_nucs, SD.num_nucs, sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_num_nucs, SD.num_nucs, sz));
    total_sz += sz;

    sz = GSD.length_concs * sizeof(double);
    gpuErrchk(cudaMalloc((void **)&GSD.concs, sz));
    gpuErrchk(cudaMemcpy(GSD.concs, SD.concs, sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_concs, SD.concs, sz));
    total_sz += sz;

    sz = GSD.length_mats * sizeof(int);
    gpuErrchk(cudaMalloc((void **)&GSD.mats, sz));
    gpuErrchk(cudaMemcpy(GSD.mats, SD.mats, sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_mats, SD.mats, sz));
    total_sz += sz;

    sz = GSD.length_unionized_energy_array * sizeof(double);
    if (sz > 0)
    {
        gpuErrchk(cudaMalloc((void **)&GSD.unionized_energy_array, sz));
        gpuErrchk(cudaMemcpy(GSD.unionized_energy_array, SD.unionized_energy_array, sz, cudaMemcpyHostToDevice));
    }
    total_sz += sz;

    sz = GSD.length_index_grid * sizeof(int);
    if (sz > 0)
    {
        gpuErrchk(cudaMalloc((void **)&GSD.index_grid, sz));
        gpuErrchk(cudaMemcpy(GSD.index_grid, SD.index_grid, sz, cudaMemcpyHostToDevice));
    }
    total_sz += sz;

    sz = GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
    gpuErrchk(cudaMalloc((void **)&GSD.nuclide_grid, sz));
    gpuErrchk(cudaMemcpy(GSD.nuclide_grid, SD.nuclide_grid, sz, cudaMemcpyHostToDevice));
    total_sz += sz;

    size_t soa_sz = GSD.length_nuclide_grid * sizeof(double);
    double *h_tmp = (double *)malloc(soa_sz);
    assert(h_tmp != NULL);

    gpuErrchk(cudaMalloc((void **)&g_d_nuc_energy, soa_sz));
    for (int i = 0; i < GSD.length_nuclide_grid; i++)
    {
        h_tmp[i] = SD.nuclide_grid[i].energy;
    }
    gpuErrchk(cudaMemcpy(g_d_nuc_energy, h_tmp, soa_sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_nuc_energy, &g_d_nuc_energy, sizeof(double *)));

    gpuErrchk(cudaMalloc((void **)&g_d_nuc_total_xs, soa_sz));
    for (int i = 0; i < GSD.length_nuclide_grid; i++)
    {
        h_tmp[i] = SD.nuclide_grid[i].total_xs;
    }
    gpuErrchk(cudaMemcpy(g_d_nuc_total_xs, h_tmp, soa_sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_nuc_total_xs, &g_d_nuc_total_xs, sizeof(double *)));

    gpuErrchk(cudaMalloc((void **)&g_d_nuc_elastic_xs, soa_sz));
    for (int i = 0; i < GSD.length_nuclide_grid; i++)
    {
        h_tmp[i] = SD.nuclide_grid[i].elastic_xs;
    }
    gpuErrchk(cudaMemcpy(g_d_nuc_elastic_xs, h_tmp, soa_sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_nuc_elastic_xs, &g_d_nuc_elastic_xs, sizeof(double *)));

    gpuErrchk(cudaMalloc((void **)&g_d_nuc_absorbtion_xs, soa_sz));
    for (int i = 0; i < GSD.length_nuclide_grid; i++)
    {
        h_tmp[i] = SD.nuclide_grid[i].absorbtion_xs;
    }
    gpuErrchk(cudaMemcpy(g_d_nuc_absorbtion_xs, h_tmp, soa_sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_nuc_absorbtion_xs, &g_d_nuc_absorbtion_xs, sizeof(double *)));

    gpuErrchk(cudaMalloc((void **)&g_d_nuc_fission_xs, soa_sz));
    for (int i = 0; i < GSD.length_nuclide_grid; i++)
    {
        h_tmp[i] = SD.nuclide_grid[i].fission_xs;
    }
    gpuErrchk(cudaMemcpy(g_d_nuc_fission_xs, h_tmp, soa_sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_nuc_fission_xs, &g_d_nuc_fission_xs, sizeof(double *)));

    gpuErrchk(cudaMalloc((void **)&g_d_nuc_nu_fission_xs, soa_sz));
    for (int i = 0; i < GSD.length_nuclide_grid; i++)
    {
        h_tmp[i] = SD.nuclide_grid[i].nu_fission_xs;
    }
    gpuErrchk(cudaMemcpy(g_d_nuc_nu_fission_xs, h_tmp, soa_sz, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_nuc_nu_fission_xs, &g_d_nuc_nu_fission_xs, sizeof(double *)));

    free(h_tmp);

    sz = in.lookups * sizeof(unsigned long);
    gpuErrchk(cudaMalloc((void **)&GSD.verification, sz));
    total_sz += sz;

    GSD.length_verification = in.lookups;
    GSD.p_energy_samples = NULL;
    GSD.mat_samples = NULL;
    GSD.length_p_energy_samples = 0;
    GSD.length_mat_samples = 0;

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
    cudaFree(g_d_nuc_energy);
    cudaFree(g_d_nuc_total_xs);
    cudaFree(g_d_nuc_elastic_xs);
    cudaFree(g_d_nuc_absorbtion_xs);
    cudaFree(g_d_nuc_fission_xs);
    cudaFree(g_d_nuc_nu_fission_xs);

    g_d_nuc_energy = NULL;
    g_d_nuc_total_xs = NULL;
    g_d_nuc_elastic_xs = NULL;
    g_d_nuc_absorbtion_xs = NULL;
    g_d_nuc_fission_xs = NULL;
    g_d_nuc_nu_fission_xs = NULL;
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
        "                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n"
    );
    border_print();
    center_print("Developed at Argonne National Laboratory", 79);
    char v[100];
    sprintf(v, "Version: %d", version);
    center_print(v, 79);
    border_print();
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

void border_print(void)
{
    printf("===============================================================================\n");
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

    int lookups_per_sec = (int)((double)lookups / runtime);
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

    unsigned long long large = 0;
    unsigned long long small = 0;

    if (in.simulation_method == EVENT_BASED)
    {
        small = 945990;
        large = 952131;
    }
    else if (in.simulation_method == HISTORY_BASED)
    {
        small = 941535;
        large = 954318;
    }

    if (strcmp(in.HM, "large") == 0)
    {
        if (vhash == large)
        {
            is_invalid_result = 0;
        }
    }
    else if (strcmp(in.HM, "small") == 0)
    {
        if (vhash == small)
        {
            is_invalid_result = 0;
        }
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

    if (nprocs < 0)
    {
        return nprocs;
    }

    return is_invalid_result;
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

    if (nprocs < 0 || version < 0)
    {
        printf("%d\n", nprocs + version);
    }

    border_print();
    center_print("INITIALIZATION - DO NOT PROFILE", 79);
    border_print();
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
    printf("  -b <binary mode>         Read or write all data structures to file. If reading, this will skip initialization phase. (read, write)\n");
    printf("  -k <kernel ID>           Specifies which kernel to run. 0 is baseline, 1, 2, etc are optimized variants. (0 is default.)\n");
    printf("Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized -k 0\n");
    printf("See readme for full description of default run values\n");
    exit(4);
}

Inputs read_CLI(int argc, char *argv[])
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
    input.HM = (char *)malloc(6 * sizeof(char));
    strcpy(input.HM, "large");

    int user_g = 0;
    int default_lookups = 1;
    int default_particles = 1;

    for (int i = 1; i < argc; i++)
    {
        char *arg = argv[i];

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

    if (strcasecmp(input.HM, "small") != 0 && strcasecmp(input.HM, "large") != 0 && strcasecmp(input.HM, "XL") != 0 && strcasecmp(input.HM, "XXL") != 0)
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

void binary_write(Inputs in, SimulationData SD)
{
    const char *fname = "XS_data.dat";
    printf("Writing all data structures to binary file %s...\n", fname);
    FILE *fp = fopen(fname, "w");

    fwrite(&SD, sizeof(SimulationData), 1, fp);
    fwrite(SD.num_nucs, sizeof(int), SD.length_num_nucs, fp);
    fwrite(SD.concs, sizeof(double), SD.length_concs, fp);
    fwrite(SD.mats, sizeof(int), SD.length_mats, fp);
    fwrite(SD.nuclide_grid, sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp);
    fwrite(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
    fwrite(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);
    fclose(fp);

    if (in.lookups < 0)
    {
        printf("%d\n", in.lookups);
    }
}

SimulationData binary_read(Inputs in)
{
    SimulationData SD;
    const char *fname = "XS_data.dat";
    printf("Reading all data structures from binary file %s...\n", fname);

    FILE *fp = fopen(fname, "r");
    assert(fp != NULL);
    fread(&SD, sizeof(SimulationData), 1, fp);

    SD.num_nucs = (int *)malloc(SD.length_num_nucs * sizeof(int));
    SD.concs = (double *)malloc(SD.length_concs * sizeof(double));
    SD.mats = (int *)malloc(SD.length_mats * sizeof(int));
    SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
    SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));

    fread(SD.num_nucs, sizeof(int), SD.length_num_nucs, fp);
    fread(SD.concs, sizeof(double), SD.length_concs, fp);
    fread(SD.mats, sizeof(int), SD.length_mats, fp);
    fread(SD.nuclide_grid, sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp);
    fread(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
    fread(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);
    fclose(fp);

    if (in.lookups < 0)
    {
        SD.length_verification = in.lookups;
    }

    return SD;
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
    SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
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

    SD.unionized_energy_array = NULL;
    SD.index_grid = NULL;

    if (in.grid_type == NUCLIDE)
    {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = 0;
    }

    if (in.grid_type == UNIONIZED)
    {
        if (mype == 0)
        {
            printf("Intializing unionized grid...\n");
        }

        SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
        SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));
        assert(SD.unionized_energy_array != NULL);
        nbytes += SD.length_unionized_energy_array * sizeof(double);

        for (int i = 0; i < SD.length_unionized_energy_array; i++)
        {
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        }

        qsort(SD.unionized_energy_array, SD.length_unionized_energy_array, sizeof(double), double_compare);

        SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
        SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
        assert(SD.index_grid != NULL);
        nbytes += SD.length_index_grid * sizeof(int);

        int *idx_low = (int *)calloc(in.n_isotopes, sizeof(int));
        assert(idx_low != NULL);

        double *energy_high = (double *)malloc(in.n_isotopes * sizeof(double));
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
        SD.length_index_grid = in.hash_bins * in.n_isotopes;
        SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
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

    if (mype == 0)
    {
        printf("Intialization complete. Allocated %.0lf MB of data on CPU.\n", nbytes / 1024.0 / 1024.0);
    }

    return SD;
}

int *load_num_nucs(long n_isotopes)
{
    int *num_nucs = (int *)malloc(12 * sizeof(int));

    if (n_isotopes == 68)
    {
        num_nucs[0] = 34;
    }
    else
    {
        num_nucs[0] = 321;
    }

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

int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs)
{
    *max_num_nucs = 0;

    for (int m = 0; m < 12; m++)
    {
        if (num_nucs[m] > *max_num_nucs)
        {
            *max_num_nucs = num_nucs[m];
        }
    }

    int *mats = (int *)calloc(12 * (*max_num_nucs), sizeof(int));

    int mats0_Sml[] =
    {
        58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
        8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53,
        54, 55, 56, 18, 23, 41
    };

    int mats0_Lrg[321] =
    {
        58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
        8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53,
        54, 55, 56, 18, 23, 41
    };

    for (int i = 0; i < 321 - 34; i++)
    {
        mats0_Lrg[34 + i] = 68 + i;
    }

    int mats1[] = {63, 64, 65, 66, 67};
    int mats2[] = {24, 41, 4, 5};
    int mats3[] = {24, 41, 4, 5};
    int mats4[] = {19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29, 30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16, 17};
    int mats5[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
    int mats6[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
    int mats7[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
    int mats8[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
    int mats9[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
    int mats10[] = {24, 41, 4, 5, 63, 64, 65, 66, 67};
    int mats11[] = {24, 41, 4, 5, 63, 64, 65, 66, 67};

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
    memcpy(mats + *max_num_nucs * 6, mats6, num_nucs[6] * sizeof(int));
    memcpy(mats + *max_num_nucs * 7, mats7, num_nucs[7] * sizeof(int));
    memcpy(mats + *max_num_nucs * 8, mats8, num_nucs[8] * sizeof(int));
    memcpy(mats + *max_num_nucs * 9, mats9, num_nucs[9] * sizeof(int));
    memcpy(mats + *max_num_nucs * 10, mats10, num_nucs[10] * sizeof(int));
    memcpy(mats + *max_num_nucs * 11, mats11, num_nucs[11] * sizeof(int));

    return mats;
}

double *load_concs(int *num_nucs, int max_num_nucs)
{
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double *concs = (double *)calloc(12 * max_num_nucs, sizeof(double));

    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < num_nucs[i]; j++)
        {
            concs[i * max_num_nucs + j] = LCG_random_double(&seed);
        }
    }

    return concs;
}

int double_compare(const void *a, const void *b)
{
    double A = *((double *)a);
    double B = *((double *)b);

    if (A > B)
    {
        return 1;
    }
    else if (A < B)
    {
        return -1;
    }

    return 0;
}

int NGP_compare(const void *a, const void *b)
{
    NuclideGridPoint A = *((NuclideGridPoint *)a);
    NuclideGridPoint B = *((NuclideGridPoint *)b);

    if (A.energy > B.energy)
    {
        return 1;
    }
    else if (A.energy < B.energy)
    {
        return -1;
    }

    return 0;
}

double rn_v(void)
{
    static unsigned long seed = 1337;
    unsigned long a = 16807;
    unsigned long m = 2147483647;
    unsigned long n1 = (a * seed) % m;
    seed = n1;

    return (double)n1 / m;
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

    return (double)us_since_epoch / 1.0e6;
}
