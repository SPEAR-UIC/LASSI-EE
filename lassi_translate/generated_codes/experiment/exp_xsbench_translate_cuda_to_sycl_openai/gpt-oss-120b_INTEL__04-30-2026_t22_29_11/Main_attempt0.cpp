// ==================== XSBench SYCL Port ====================
// Single translation unit, SYCL 2020, compiled with icpx
// ---------------------------------------------------------

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

// Global SYCL queue (single device)
sycl::queue q{sycl::gpu_selector_v};

// Grid types
#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2

// Simulation types
#define HISTORY_BASED 1
#define EVENT_BASED 2

// Binary Mode Type
#define NONE 0
#define READ 1
#define WRITE 2

// Starting Seed
#define STARTING_SEED 1070

// Structures
typedef struct{
    double energy;
    double total_xs;
    double elastic_xs;
    double absorbtion_xs;
    double fission_xs;
    double nu_fission_xs;
} NuclideGridPoint;

typedef struct{
    int nthreads;
    long n_isotopes;
    long n_gridpoints;
    int lookups;
    char * HM;
    int grid_type; // 0: Unionized Grid (default)    1: Nuclide Grid
    int hash_bins;
    int particles;
    int simulation_method;
    int binary_mode;
    int kernel_id;
} Inputs;

typedef struct{
    int * num_nucs;                     // Length = length_num_nucs;
    double * concs;                     // Length = length_concs
    int * mats;                         // Length = length_mats
    double * unionized_energy_array;    // Length = length_unionized_energy_array
    int * index_grid;                   // Length = length_index_grid
    NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
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
} SimulationData;

// ------------------------------------------------------------------
// Utility functions (device compatible)
inline double LCG_random_double(uint64_t * seed)
{
    const uint64_t m = 9223372036854775808ULL; // 2^63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double)(*seed) / (double)m;
}

inline uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
    const uint64_t m = 9223372036854775808ULL; // 2^63
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;
    while (n > 0) {
        if (n & 1) {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }
    return (a_new * seed + c_new) % m;
}

inline int pick_mat(uint64_t * seed)
{
    double dist[12] = {0.140, 0.052, 0.275, 0.134, 0.154,
                       0.064, 0.066, 0.055, 0.008, 0.015,
                       0.025, 0.013};
    double roll = LCG_random_double(seed);
    for (int i = 0; i < 12; ++i) {
        double running = 0.0;
        for (int j = i; j > 0; --j)
            running += dist[j];
        if (roll < running)
            return i;
    }
    return 0;
}

inline long grid_search(long n, double quarry, const double * A)
{
    long lower = 0;
    long upper = n - 1;
    long len = upper - lower;
    while (len > 1) {
        long mid = lower + (len / 2);
        if (A[mid] > quarry)
            upper = mid;
        else
            lower = mid;
        len = upper - lower;
    }
    return lower;
}

inline long grid_search_nuclide(long n, double quarry,
                                const NuclideGridPoint * A,
                                long low, long high)
{
    long lower = low;
    long upper = high;
    long len = upper - lower;
    while (len > 1) {
        long mid = lower + (len / 2);
        if (A[mid].energy > quarry)
            upper = mid;
        else
            lower = mid;
        len = upper - lower;
    }
    return lower;
}

// ------------------------------------------------------------------
// Core calculation functions (device callable)
inline void calculate_micro_xs(double p_energy, int nuc,
                               long n_isotopes, long n_gridpoints,
                               const double * egrid, const int * index_data,
                               const NuclideGridPoint * nuclide_grids,
                               long idx, double * xs_vector,
                               int grid_type, int hash_bins)
{
    double f;
    const NuclideGridPoint * low;
