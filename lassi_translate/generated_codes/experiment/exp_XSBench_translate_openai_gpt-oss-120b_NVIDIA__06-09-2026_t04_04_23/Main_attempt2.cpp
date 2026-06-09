// ==================== XSbench_SYCL.cpp ====================
// SYCL translation of the CUDA XSBench benchmark
// ---------------------------------------------------------

#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <numeric>
#include <vector>
#include <cassert>
#include <algorithm>

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

// ------------------------------------------------------------------
// Data structures
// ------------------------------------------------------------------
struct NuclideGridPoint {
    double energy;
    double total_xs;
    double elastic_xs;
    double absorbtion_xs;
    double fission_xs;
    double nu_fission_xs;
};

struct Inputs {
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
};

struct SimulationData {
    int *num_nucs;                     // Length = length_num_nucs;
    double *concs;                     // Length = length_concs
    int *mats;                         // Length = length_mats
    double *unionized_energy_array;    // Length = length_unionized_energy_array
    int *index_grid;                   // Length = length_index_grid
    NuclideGridPoint *nuclide_grid;    // Length = length_nuclide_grid
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
};

// ------------------------------------------------------------------
// Utility functions (host)
// ------------------------------------------------------------------
static double get_time() {
    using namespace std::chrono;
    return duration<double>(high_resolution_clock::now().time_since_epoch()).count();
}

static void fancy_int(long a) {
    if (a < 1000) printf("%ld\n", a);
    else if (a < 1000000) printf("%ld,%03ld\n", a / 1000, a % 1000);
    else if (a < 1000000000) printf("%ld,%03ld,%03ld\n", a / 1000000, (a % 1000000) / 1000, a % 1000);
    else printf("%ld,%03ld,%03ld,%03ld\n",
                a / 1000000000,
                (a % 1000000000) / 1000000,
                (a % 1000000) / 1000,
                a % 1000);
}

// ------------------------------------------------------------------
// RNG helpers (device)
// ------------------------------------------------------------------
static inline double LCG_random_double(uint64_t *seed) {
    const uint64_t m = 9223372036854775808ULL; // 2^63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return static_cast<double>(*seed) / static_cast<double>(m);
}

static inline uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
    const uint64_t m = 9223372036854775808ULL;
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

// ------------------------------------------------------------------
// Device helper functions
// ------------------------------------------------------------------
static inline long grid_search_nuclide(long n, double quarry,
                                      const NuclideGridPoint *A,
                                      long low, long high) {
    long lowerLimit = low;
    long upperLimit = high;
    long length = upperLimit - lowerLimit;
    while (length > 1) {
        long examinationPoint = lowerLimit + (length / 2);
        if (A[examinationPoint].energy > quarry)
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;
        length = upperLimit - lowerLimit;
    }
    return lowerLimit;
}

static inline long grid_search(long n, double quarry, const double *A) {
    long lowerLimit = 0;
    long upperLimit = n - 1;
    long length = upperLimit - lowerLimit;
    while (length > 1) {
        long examinationPoint = lowerLimit + (length / 2);
        if (A[examinationPoint] > quarry)
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;
        length = upperLimit - lowerLimit;
    }
    return lowerLimit;
}

static inline int pick_mat(uint64_t *seed) {
    const double dist[12] = {0.140, 0.052, 0.275, 0.134, 0.154,
                             0.064, 0.066, 0.055, 0.008, 0.015,
                             0.025, 0.013};
    double roll = LCG_random_double(seed);
    double running = 0.0;
    for (int i = 0; i < 12; ++i) {
        running += dist[i];
        if (roll < running) return i;
    }
    return 0;
}

// ------------------------------------------------------------------
// Core physics functions (device)
// ------------------------------------------------------------------
static inline void calculate_micro_xs(double p_energy, int nuc,
                                      long n_isotopes, long n_gridpoints,
                                      const double *egrid, const int *index_data,
                                      const NuclideGridPoint *nuclide_grids,
                                      long idx, double *xs_vector,
                                      int grid_type, int hash_bins) {
    double f;
    const NuclideGridPoint *low, *high;

    if (grid_type == NUCLIDE) {
        idx = grid_search_nuclide(n_gridpoints, p_energy,
                                  nuclide_grids + nuc * n_gridpoints, 0,
                                  n_gridpoints - 1);
        low = (idx == n_gridpoints - 1)
                  ? &nuclide_grids[nuc * n_gridpoints + idx - 1]
                  : &nuclide_grids[nuc * n_gridpoints + idx];
    } else if (grid_type == UNIONIZED) {
        int index = index_data[idx * n_isotopes + nuc];
        low = (index == n_gridpoints - 1)
                  ? &nuclide_grids[nuc * n_gridpoints + index - 1]
                  : &nuclide_grids[nuc * n_gridpoints + index];
    } else { // HASH
        int u_low = index_data[idx * n_isotopes + nuc];
        int u_high = (idx == hash_bins - 1)
                         ? n_gridpoints - 1
                         : index_data[(idx + 1) * n_isotopes + nuc] + 1;
        double e_low = nuclide_grids[nuc * n_gridpoints + u_low].energy;
        double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
        int lower;
        if (p_energy <= e_low)
            lower = 0;
        else if (p_energy >= e_high)
            lower = n_gridpoints - 1;
        else
            lower = grid_search_nuclide(n_gridpoints, p_energy,
                                        nuclide_grids + nuc * n_gridpoints,
                                        u_low, u_high);
        low = (lower == n_gridpoints - 1)
                  ? &nuclide_grids[nuc * n_gridpoints + lower - 1]
                  : &nuclide_grids[nuc * n_gridpoints + lower];
    }
    high = low + 1;
    f = (high->energy - p_energy) / (high->energy - low->energy);
    xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
    xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
    xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
    xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
    xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

static inline void calculate_macro_xs(double p_energy, int mat,
                                      long n_isotopes, long n_gridpoints,
                                      const int *num_nucs, const double *concs,
                                      const double *egrid, const int *index_data,
                                      const NuclideGridPoint *nuclide_grids,
                                      const int *mats, double *macro_xs_vector,
                                      int grid_type, int hash_bins, int max_num_nucs) {
    for (int k = 0; k < 5; ++k) macro_xs_vector[k] = 0.0;

    long idx = -1;
    if (grid_type == UNIONIZED)
        idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
    else if (grid_type == HASH) {
        double du = 1.0 / hash_bins;
        idx = static_cast<long>(p_energy / du);
    }

    for (int j = 0; j < num_nucs[mat]; ++j) {
        double xs_vector[5];
        int p_nuc = mats[mat * max_num_nucs + j];
        double conc = concs[mat * max_num_nucs + j];
        calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints,
                           egrid, index_data, nuclide_grids, idx,
                           xs_vector, grid_type, hash_bins);
        for (int k = 0; k < 5; ++k) macro_xs_vector[k] += xs_vector[k] * conc;
    }
}

// ------------------------------------------------------------------
// Comparison helpers for sorting/counting (host/device)
// ------------------------------------------------------------------
struct is_mat_fuel {
    bool operator()(const int &a) const { return a == 0; }
};

// ------------------------------------------------------------------
// Data generation helpers (host)
// ------------------------------------------------------------------
static int *load_num_nucs(long n_isotopes) {
    int *num_nucs = (int *)malloc(12 * sizeof(int));
    if (n_isotopes == 68)
        num_nucs[0] = 34;
    else
        num_nucs[0] = 321;
    num_nucs[1] = 5;  num_nucs[2] = 4;  num_nucs[3] = 4;
    num_nucs[4] = 27; num_nucs[5] = 21; num_nucs[6] = 21;
    num_nucs[7] = 21; num_nucs[8] = 21; num_nucs[9] = 21;
    num_nucs[10] = 9; num_nucs[11] = 9;
    return num_nucs;
}

static int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs) {
    *max_num_nucs = 0;
    for (int m = 0; m < 12; ++m)
        if (num_nucs[m] > *max_num_nucs) *max_num_nucs = num_nucs[m];
    int *mats = (int *)malloc(12 * (*max_num_nucs) * sizeof(int));

    // Fuel
    if (n_isotopes == 68) {
        int mats0_Sml[34] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,
                             8,9,10,29,57,47,48,0,62,15,33,34,52,53,
                             54,55,56,18,23,41};
        memcpy(mats, mats0_Sml, 34 * sizeof(int));
    } else {
        int mats0_Lrg[321];
        int base[34] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,
                        8,9,10,29,57,47,48,0,62,15,33,34,52,53,
                        54,55,56,18,23,41};
        memcpy(mats0_Lrg, base, 34 * sizeof(int));
        for (int i = 0; i < 321 - 34; ++i) mats0_Lrg[34 + i] = 68 + i;
        memcpy(mats, mats0_Lrg, 321 * sizeof(int));
    }

    // Other materials (hard‑coded)
    int mats1[5] = {63,64,65,66,67};
    int mats2[4] = {24,41,4,5};
    int mats3[4] = {24,41,4,5};
    int mats4[27] = {19,20,21,22,35,36,37,38,39,25,27,28,29,
                    30,31,32,26,49,50,51,11,12,13,14,6,16,17};
    int mats5[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                    49,50,51,11,12,13,14};
    int mats6[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                    49,50,51,11,12,13,14};
    int mats7[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                    49,50,51,11,12,13,14};
    int mats8[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                    49,50,51,11,12,13,14};
    int mats9[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                    49,50,51,11,12,13,14};
    int mats10[9] = {24,41,4,5,63,64,65,66,67};
    int mats11[9] = {24,41,4,5,63,64,65,66,67};

    memcpy(mats + (*max_num_nucs) * 1, mats1, 5 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 2, mats2, 4 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 3, mats3, 4 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 4, mats4, 27 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 5, mats5, 21 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 6, mats6, 21 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 7, mats7, 21 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 8, mats8, 21 * sizeof(int));
    memcpy(mats + (*max_num_nucs) * 9, mats9, 21 * sizeof(int));
    memcpy(mats + (*max_num_nucs) *10, mats10, 9 * sizeof(int));
    memcpy(mats + (*max_num_nucs) *11, mats11, 9 * sizeof(int));
    return mats;
}

static double *load_concs(int *num_nucs, int max_num_nucs) {
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double *concs = (double *)malloc(12 * max_num_nucs * sizeof(double));
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < num_nucs[i]; ++j)
            concs[i * max_num_nucs + j] = LCG_random_double(&seed);
    return concs;
}

// ------------------------------------------------------------------
// Grid initialization (host)
// ------------------------------------------------------------------
static SimulationData grid_init_do_not_profile(const Inputs &in) {
    SimulationData SD{};
    size_t nbytes = 0;
    uint64_t seed = 42;

    // Nuclide grid
    SD.length_nuclide_grid = static_cast<int>(in.n_isotopes * in.n_gridpoints);
    SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    for (int i = 0; i < SD.length_nuclide_grid; ++i) {
        SD.nuclide_grid[i].energy = LCG_random_double(&seed);
        SD.nuclide_grid[i].total_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].elastic_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].fission_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
    }
    for (int i = 0; i < in.n_isotopes; ++i)
        qsort(&SD.nuclide_grid[i * in.n_gridpoints], in.n_gridpoints,
              sizeof(NuclideGridPoint),
              [](const void *a, const void *b) {
                  const NuclideGridPoint *A = (const NuclideGridPoint *)a;
                  const NuclideGridPoint *B = (const NuclideGridPoint *)b;
                  return (A->energy > B->energy) - (A->energy < B->energy);
              });

    // Acceleration structures
    if (in.grid_type == UNIONIZED) {
        SD.length_unionized_energy_array = static_cast<int>(in.n_isotopes * in.n_gridpoints);
        SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));
        for (int i = 0; i < SD.length_unionized_energy_array; ++i)
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        qsort(SD.unionized_energy_array, SD.length_unionized_energy_array,
              sizeof(double),
              [](const void *a, const void *b) {
                  double A = *(const double *)a;
                  double B = *(const double *)b;
                  return (A > B) - (A < B);
              });

        SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
        SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
        int *idx_low = (int *)calloc(in.n_isotopes, sizeof(int));
        double *energy_high = (double *)malloc(in.n_isotopes * sizeof(double));
        for (int i = 0; i < in.n_isotopes; ++i)
            energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;
        for (long e = 0; e < SD.length_unionized_energy_array; ++e) {
            double ue = SD.unionized_energy_array[e];
            for (long i = 0; i < in.n_isotopes; ++i) {
                if (ue < energy_high[i])
                    SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                else if (idx_low[i] == in.n_gridpoints - 2)
                    SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                else {
                    idx_low[i]++;
                    SD.index_grid[e * in.n_isotopes + i] = idx_low[i];
                    energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + idx_low[i] + 1].energy;
                }
            }
        }
        free(idx_low);
        free(energy_high);
    } else if (in.grid_type == HASH) {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = static_cast<long>(in.hash_bins * in.n_isotopes);
        SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
        double du = 1.0 / in.hash_bins;
        for (long e = 0; e < in.hash_bins; ++e) {
            double energy = e * du;
            for (long i = 0; i < in.n_isotopes; ++i) {
                SD.index_grid[e * in.n_isotopes + i] = static_cast<int>(
                    grid_search_nuclide(in.n_gridpoints, energy,
                                        SD.nuclide_grid + i * in.n_gridpoints,
                                        0, in.n_gridpoints - 1));
            }
        }
    } else {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = 0;
    }

    // Materials
    SD.num_nucs = load_num_nucs(in.n_isotopes);
    SD.length_num_nucs = 12;
    SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
    SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
    SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
    SD.length_concs = SD.length_mats;

    return SD;
}

// ------------------------------------------------------------------
// Device memory management (USM)
// ------------------------------------------------------------------
static SimulationData move_simulation_data_to_device(const Inputs &in,
                                                    const SimulationData &hostSD,
                                                    sycl::queue &q) {
    SimulationData devSD = hostSD;

    devSD.num_nucs = sycl::malloc_device<int>(hostSD.length_num_nucs, q);
    q.memcpy(devSD.num_nucs, hostSD.num_nucs,
             hostSD.length_num_nucs * sizeof(int)).wait();

    devSD.concs = sycl::malloc_device<double>(hostSD.length_concs, q);
    q.memcpy(devSD.concs, hostSD.concs,
             hostSD.length_concs * sizeof(double)).wait();

    devSD.mats = sycl::malloc_device<int>(hostSD.length_mats, q);
    q.memcpy(devSD.mats, hostSD.mats,
             hostSD.length_mats * sizeof(int)).wait();

    devSD.unionized_energy_array = sycl::malloc_device<double>(hostSD.length_unionized_energy_array, q);
    q.memcpy(devSD.unionized_energy_array, hostSD.unionized_energy_array,
             hostSD.length_unionized_energy_array * sizeof(double)).wait();

    devSD.index_grid = sycl::malloc_device<int>(hostSD.length_index_grid, q);
    q.memcpy(devSD.index_grid, hostSD.index_grid,
             hostSD.length_index_grid * sizeof(int)).wait();

    devSD.nuclide_grid = sycl::malloc_device<NuclideGridPoint>(hostSD.length_nuclide_grid, q);
    q.memcpy(devSD.nuclide_grid, hostSD.nuclide_grid,
             hostSD.length_nuclide_grid * sizeof(NuclideGridPoint)).wait();

    devSD.verification = sycl::malloc_device<unsigned long>(in.lookups, q);
    q.memset(devSD.verification, 0, in.lookups * sizeof(unsigned long)).wait();
    devSD.length_verification = in.lookups;

    return devSD;
}

static void release_device_memory(SimulationData &devSD, sycl::queue &q) {
    sycl::free(devSD.num_nucs, q);
    sycl::free(devSD.concs, q);
    sycl::free(devSD.mats, q);
    sycl::free(devSD.unionized_energy_array, q);
    sycl::free(devSD.index_grid, q);
    sycl::free(devSD.nuclide_grid, q);
    sycl::free(devSD.verification, q);
    if (devSD.p_energy_samples) sycl::free(devSD.p_energy_samples, q);
    if (devSD.mat_samples) sycl::free(devSD.mat_samples, q);
}

// ------------------------------------------------------------------
// Baseline simulation
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_baseline(const Inputs &in,
                                                             SimulationData &devSD,
                                                             sycl::queue &q) {
    const int nthreads = 256;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(p_energy, mat,
                                   in.n_isotopes, in.n_gridpoints,
                                   devSD.num_nucs, devSD.concs,
                                   devSD.unionized_energy_array,
                                   devSD.index_grid,
                                   devSD.nuclide_grid,
                                   devSD.mats,
                                   macro_xs_vector,
                                   in.grid_type, in.hash_bins,
                                   devSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j) {
                    if (macro_xs_vector[j] > max) {
                        max = macro_xs_vector[j];
                        max_idx = j;
                    }
                }
                devSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
    }).wait();

    // Reduce verification on host
    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    unsigned long long checksum = std::accumulate(host_verif.begin(),
                                                  host_verif.end(),
                                                  0ULL);
    return checksum;
}

// ------------------------------------------------------------------
// Optimization 1 (sampling + lookup split)
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_optimization_1(const Inputs &in,
                                                                   SimulationData &devSD,
                                                                   sycl::queue &q) {
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    // Allocate extra buffers
    devSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    devSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
    devSD.length_p_energy_samples = in.lookups;
    devSD.length_mat_samples = in.lookups;

    // Sampling kernel
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                devSD.p_energy_samples[i] = p_energy;
                devSD.mat_samples[i] = mat;
            });
    }).wait();

    // Lookup kernel
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(devSD.p_energy_samples[i],
                                   devSD.mat_samples[i],
                                   in.n_isotopes, in.n_gridpoints,
                                   devSD.num_nucs, devSD.concs,
                                   devSD.unionized_energy_array,
                                   devSD.index_grid,
                                   devSD.nuclide_grid,
                                   devSD.mats,
                                   macro_xs_vector,
                                   in.grid_type, in.hash_bins,
                                   devSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j) {
                    if (macro_xs_vector[j] > max) {
                        max = macro_xs_vector[j];
                        max_idx = j;
                    }
                }
                devSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 2 (material‑specific kernels)
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_optimization_2(const Inputs &in,
                                                                   SimulationData &devSD,
                                                                   sycl::queue &q) {
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    devSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    devSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
    devSD.length_p_energy_samples = in.lookups;
    devSD.length_mat_samples = in.lookups;

    // Sampling
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                devSD.p_energy_samples[i] = p_energy;
                devSD.mat_samples[i] = mat;
            });
    }).wait();

    // Material kernels
    for (int m = 0; m < 12; ++m) {
        q.submit([&, m](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                                  sycl::range<1>(nthreads)),
                [=](sycl::nd_item<1> it) {
                    int i = it.get_global_id(0);
                    if (i >= in.lookups) return;
                    int mat = devSD.mat_samples[i];
                    if (mat != m) return;
                    double macro_xs_vector[5] = {0};
                    calculate_macro_xs(devSD.p_energy_samples[i],
                                       mat,
                                       in.n_isotopes, in.n_gridpoints,
                                       devSD.num_nucs, devSD.concs,
                                       devSD.unionized_energy_array,
                                       devSD.index_grid,
                                       devSD.nuclide_grid,
                                       devSD.mats,
                                       macro_xs_vector,
                                       in.grid_type, in.hash_bins,
                                       devSD.max_num_nucs);
                    double max = -1.0;
                    int max_idx = 0;
                    for (int j = 0; j < 5; ++j) {
                        if (macro_xs_vector[j] > max) {
                            max = macro_xs_vector[j];
                            max_idx = j;
                        }
                    }
                    devSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
                });
        }).wait();
    }

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 3 (fuel vs non‑fuel)
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_optimization_3(const Inputs &in,
                                                                   SimulationData &devSD,
                                                                   sycl::queue &q) {
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    devSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    devSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
    devSD.length_p_energy_samples = in.lookups;
    devSD.length_mat_samples = in.lookups;

    // Sampling
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                devSD.p_energy_samples[i] = p_energy;
                devSD.mat_samples[i] = mat;
            });
    }).wait();

    // Fuel kernel
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                int mat = devSD.mat_samples[i];
                if (mat != 0) return;
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(devSD.p_energy_samples[i],
                                   mat,
                                   in.n_isotopes, in.n_gridpoints,
                                   devSD.num_nucs, devSD.concs,
                                   devSD.unionized_energy_array,
                                   devSD.index_grid,
                                   devSD.nuclide_grid,
                                   devSD.mats,
                                   macro_xs_vector,
                                   in.grid_type, in.hash_bins,
                                   devSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j) {
                    if (macro_xs_vector[j] > max) {
                        max = macro_xs_vector[j];
                        max_idx = j;
                    }
                }
                devSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
    }).wait();

    // Non‑fuel kernel
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                int mat = devSD.mat_samples[i];
                if (mat == 0) return;
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(devSD.p_energy_samples[i],
                                   mat,
                                   in.n_isotopes, in.n_gridpoints,
                                   devSD.num_nucs, devSD.concs,
                                   devSD.unionized_energy_array,
                                   devSD.index_grid,
                                   devSD.nuclide_grid,
                                   devSD.mats,
                                   macro_xs_vector,
                                   in.grid_type, in.hash_bins,
                                   devSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j) {
                    if (macro_xs_vector[j] > max) {
                        max = macro_xs_vector[j];
                        max_idx = j;
                    }
                }
                devSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 4 (full material sort)
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_optimization_4(const Inputs &in,
                                                                   SimulationData &devSD,
                                                                   sycl::queue &q) {
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    devSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    devSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
    devSD.length_p_energy_samples = in.lookups;
    devSD.length_mat_samples = in.lookups;

    // Sampling
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                devSD.p_energy_samples[i] = p_energy;
                devSD.mat_samples[i] = mat;
            });
    }).wait();

    // Copy to host for sorting
    std::vector<int> h_mat(in.lookups);
    std::vector<double> h_energy(in.lookups);
    q.memcpy(h_mat.data(), devSD.mat_samples, in.lookups * sizeof(int)).wait();
    q.memcpy(h_energy.data(), devSD.p_energy_samples, in.lookups * sizeof(double)).wait();

    // Sort by material (stable)
    std::vector<std::pair<int, double>> pairs(in.lookups);
    for (int i = 0; i < in.lookups; ++i) pairs[i] = {h_mat[i], h_energy[i]};
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });

    // Write back sorted data
    for (int i = 0; i < in.lookups; ++i) {
        h_mat[i] = pairs[i].first;
        h_energy[i] = pairs[i].second;
    }
    q.memcpy(devSD.mat_samples, h_mat.data(), in.lookups * sizeof(int)).wait();
    q.memcpy(devSD.p_energy_samples, h_energy.data(), in.lookups * sizeof(double)).wait();

    // Count per material
    std::vector<int> h_counts(12, 0);
    for (int i = 0; i < in.lookups; ++i) ++h_counts[h_mat[i]];

    // Launch per‑material kernels
    int offset = 0;
    for (int m = 0; m < 12; ++m) {
        int nlook = h_counts[m];
        if (nlook == 0) continue;
        int nb = (nlook + nthreads - 1) / nthreads;
        q.submit([&, m, nlook, offset](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nb * nthreads),
                                  sycl::range<1>(nthreads)),
                [=](sycl::nd_item<1> it) {
                    int i = it.get_global_id(0);
                    if (i >= nlook) return;
                    int idx = i + offset;
                    int mat = devSD.mat_samples[idx];
                    if (mat != m) return;
                    double macro_xs_vector[5] = {0};
                    calculate_macro_xs(devSD.p_energy_samples[idx],
                                       mat,
                                       in.n_isotopes, in.n_gridpoints,
                                       devSD.num_nucs, devSD.concs,
                                       devSD.unionized_energy_array,
                                       devSD.index_grid,
                                       devSD.nuclide_grid,
                                       devSD.mats,
                                       macro_xs_vector,
                                       in.grid_type, in.hash_bins,
                                       devSD.max_num_nucs);
                    double max = -1.0;
                    int max_idx = 0;
                    for (int j = 0; j < 5; ++j) {
                        if (macro_xs_vector[j] > max) {
                            max = macro_xs_vector[j];
                            max_idx = j;
                        }
                    }
                    devSD.verification[idx] = static_cast<unsigned long>(max_idx + 1);
                });
        }).wait();
        offset += nlook;
    }

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 5 (fuel / non‑fuel partition)
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_optimization_5(const Inputs &in,
                                                                   SimulationData &devSD,
                                                                   sycl::queue &q) {
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    devSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    devSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
    devSD.length_p_energy_samples = in.lookups;
    devSD.length_mat_samples = in.lookups;

    // Sampling
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                devSD.p_energy_samples[i] = p_energy;
                devSD.mat_samples[i] = mat;
            });
    }).wait();

    // Copy to host for partition
    std::vector<int> h_mat(in.lookups);
    std::vector<double> h_energy(in.lookups);
    q.memcpy(h_mat.data(), devSD.mat_samples, in.lookups * sizeof(int)).wait();
    q.memcpy(h_energy.data(), devSD.p_energy_samples, in.lookups * sizeof(double)).wait();

    // Partition (fuel first)
    auto partition_point = std::stable_partition(
        h_mat.begin(), h_mat.end(),
        [&](int a) { return is_mat_fuel()(a); });
    size_t n_fuel = std::distance(h_mat.begin(), partition_point);

    // Apply same permutation to energies
    std::vector<std::pair<int, double>> paired(in.lookups);
    for (size_t i = 0; i < h_mat.size(); ++i) paired[i] = {h_mat[i], h_energy[i]};
    std::stable_partition(paired.begin(), paired.end(),
                          [&](const auto &p) { return is_mat_fuel()(p.first); });
    for (size_t i = 0; i < paired.size(); ++i) {
        h_mat[i] = paired[i].first;
        h_energy[i] = paired[i].second;
    }

    // Write back partitioned data
    q.memcpy(devSD.mat_samples, h_mat.data(), in.lookups * sizeof(int)).wait();
    q.memcpy(devSD.p_energy_samples, h_energy.data(), in.lookups * sizeof(double)).wait();

    // Fuel kernel
    int nb_fuel = (static_cast<int>(n_fuel) + nthreads - 1) / nthreads;
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nb_fuel * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= static_cast<int>(n_fuel)) return;
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(devSD.p_energy_samples[i],
                                   devSD.mat_samples[i],
                                   in.n_isotopes, in.n_gridpoints,
                                   devSD.num_nucs, devSD.concs,
                                   devSD.unionized_energy_array,
                                   devSD.index_grid,
                                   devSD.nuclide_grid,
                                   devSD.mats,
                                   macro_xs_vector,
                                   in.grid_type, in.hash_bins,
                                   devSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j) {
                    if (macro_xs_vector[j] > max) {
                        max = macro_xs_vector[j];
                        max_idx = j;
                    }
                }
                devSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
    }).wait();

    // Non‑fuel kernel
    int n_nonfuel = in.lookups - static_cast<int>(n_fuel);
    int nb_nonfuel = (n_nonfuel + nthreads - 1) / nthreads;
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nb_nonfuel * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= n_nonfuel) return;
                int idx = i + static_cast<int>(n_fuel);
                double macro_xs_vector[5] = {0};
                calculate_macro_xs(devSD.p_energy_samples[idx],
                                   devSD.mat_samples[idx],
                                   in.n_isotopes, in.n_gridpoints,
                                   devSD.num_nucs, devSD.concs,
                                   devSD.unionized_energy_array,
                                   devSD.index_grid,
                                   devSD.nuclide_grid,
                                   devSD.mats,
                                   macro_xs_vector,
                                   in.grid_type, in.hash_bins,
                                   devSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j) {
                    if (macro_xs_vector[j] > max) {
                        max = macro_xs_vector[j];
                        max_idx = j;
                    }
                }
                devSD.verification[idx] = static_cast<unsigned long>(max_idx + 1);
            });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 6 (material + energy sort)
// ------------------------------------------------------------------
static unsigned long long run_event_based_simulation_optimization_6(const Inputs &in,
                                                                   SimulationData &devSD,
                                                                   sycl::queue &q) {
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    devSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    devSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
    devSD.length_p_energy_samples = in.lookups;
    devSD.length_mat_samples = in.lookups;

    // Sampling
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads),
                              sycl::range<1>(nthreads)),
            [=](sycl::nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                uint64_t seed = STARTING_SEED;
                seed = fast_forward_LCG(seed, 2ULL * i);
                double p_energy = LCG_random_double(&seed);
                int mat = pick_mat(&seed);
                devSD.p_energy_samples[i] = p_energy;
                devSD.mat_samples[i] = mat;
            });
    }).wait();

    // Copy to host
    std::vector<int> h_mat(in.lookups);
    std::vector<double> h_energy(in.lookups);
    q.memcpy(h_mat.data(), devSD.mat_samples, in.lookups * sizeof(int)).wait();
    q.memcpy(h_energy.data(), devSD.p_energy_samples, in.lookups * sizeof(double)).wait();

    // Count per material
    std::vector<int> h_counts(12, 0);
    for (int i = 0; i < in.lookups; ++i) ++h_counts[h_mat[i]];

    // Sort by material (stable)
    std::vector<std::pair<int, double>> pairs(in.lookups);
    for (int i = 0; i < in.lookups; ++i) pairs[i] = {h_mat[i], h_energy[i]};
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });

    // Sort each material block by energy (stable)
    int offset = 0;
    for (int m = 0; m < 12; ++m) {
        int n = h_counts[m];
        if (n == 0) continue;
        std::stable_sort(pairs.begin() + offset,
                         pairs.begin() + offset + n,
                         [](const auto &a, const auto &b) { return a.second < b.second; });
        offset += n;
    }

    // Write back sorted data
    for (int i = 0; i < in.lookups; ++i) {
        h_mat[i] = pairs[i].first;
        h_energy[i] = pairs[i].second;
    }
    q.memcpy(devSD.mat_samples, h_mat.data(), in.lookups * sizeof(int)).wait();
    q.memcpy(devSD.p_energy_samples, h_energy.data(), in.lookups * sizeof(double)).wait();

    // Launch per‑material kernels (reuse logic from opt‑4)
    offset = 0;
    for (int m = 0; m < 12; ++m) {
        int nlook = h_counts[m];
        if (nlook == 0) continue;
        int nb = (nlook + nthreads - 1) / nthreads;
        q.submit([&, m, nlook, offset](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nb * nthreads),
                                  sycl::range<1>(nthreads)),
                [=](sycl::nd_item<1> it) {
                    int i = it.get_global_id(0);
                    if (i >= nlook) return;
                    int idx = i + offset;
                    int mat = devSD.mat_samples[idx];
                    if (mat != m) return;
                    double macro_xs_vector[5] = {0};
                    calculate_macro_xs(devSD.p_energy_samples[idx],
                                       mat,
                                       in.n_isotopes, in.n_gridpoints,
                                       devSD.num_nucs, devSD.concs,
                                       devSD.unionized_energy_array,
                                       devSD.index_grid,
                                       devSD.nuclide_grid,
                                       devSD.mats,
                                       macro_xs_vector,
                                       in.grid_type, in.hash_bins,
                                       devSD.max_num_nucs);
                    double max = -1.0;
                    int max_idx = 0;
                    for (int j = 0; j < 5; ++j) {
                        if (macro_xs_vector[j] > max) {
                            max = macro_xs_vector[j];
                            max_idx = j;
                        }
                    }
                    devSD.verification[idx] = static_cast<unsigned long>(max_idx + 1);
                });
        }).wait();
        offset += nlook;
    }

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), devSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Result printing (host)
// ------------------------------------------------------------------
static int print_results(const Inputs &in, int mype, double runtime,
                         int nprocs, unsigned long long vhash) {
    int lookups = (in.simulation_method == HISTORY_BASED)
                      ? in.lookups * in.particles
                      : in.lookups;
    int lookups_per_sec = static_cast<int>(lookups / runtime);
    int is_invalid = 1;

    if (mype == 0) {
        printf("\n===================================================================\n");
        printf("RESULTS\n");
        printf("===================================================================\n");
        printf("Runtime: %.3lf seconds\n", runtime);
        printf("Lookups: "); fancy_int(lookups);
        printf("Lookups/s: "); fancy_int(lookups_per_sec);
        printf("\nVerification checksum: %llu\n", vhash);
        if (strcmp(in.HM, "large") == 0) {
            if (vhash == 952131ULL) is_invalid = 0;
        } else if (strcmp(in.HM, "small") == 0) {
            if (vhash == 945990ULL) is_invalid = 0;
        }
        if (is_invalid)
            printf("WARNING - INVALID CHECKSUM!\n");
        else
            printf("Checksum VALID.\n");
        printf("===================================================================\n");
    }
    return is_invalid;
}

// ------------------------------------------------------------------
// Minimal CLI parser (keeps original defaults)
// ------------------------------------------------------------------
static Inputs read_CLI(int argc, char *argv[]) {
    Inputs in{};
    in.simulation_method = HISTORY_BASED;
    in.nthreads = 1;
    in.n_isotopes = 355;
    in.n_gridpoints = 11303;
    in.particles = 500000;
    in.lookups = 34;
    in.grid_type = UNIONIZED;
    in.hash_bins = 10000;
    in.binary_mode = NONE;
    in.kernel_id = 0;
    in.HM = (char *)malloc(6);
    strcpy(in.HM, "large");

    int user_g = 0;
    int default_lookups = 1, default_particles = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-g") == 0 && ++i < argc) {
            user_g = 1;
            in.n_gridpoints = atol(argv[i]);
        } else if (strcmp(argv[i], "-m") == 0 && ++i < argc) {
            if (strcmp(argv[i], "history") == 0) in.simulation_method = HISTORY_BASED;
            else if (strcmp(argv[i], "event") == 0) {
                in.simulation_method = EVENT_BASED;
                if (default_lookups && default_particles) {
                    in.lookups = in.lookups * in.particles;
                    in.particles = 0;
                }
            } else { printf("Invalid simulation method\n"); exit(1); }
        } else if (strcmp(argv[i], "-l") == 0 && ++i < argc) {
            in.lookups = atoi(argv[i]); default_lookups = 0;
        } else if (strcmp(argv[i], "-h") == 0 && ++i < argc) {
            in.hash_bins = atoi(argv[i]);
        } else if (strcmp(argv[i], "-p") == 0 && ++i < argc) {
            in.particles = atoi(argv[i]); default_particles = 0;
        } else if (strcmp(argv[i], "-s") == 0 && ++i < argc) {
            in.HM = argv[i];
        } else if (strcmp(argv[i], "-G") == 0 && ++i < argc) {
            if (strcmp(argv[i], "unionized") == 0) in.grid_type = UNIONIZED;
            else if (strcmp(argv[i], "nuclide") == 0) in.grid_type = NUCLIDE;
            else if (strcmp(argv[i], "hash") == 0) in.grid_type = HASH;
            else { printf("Invalid grid type\n"); exit(1); }
        } else if (strcmp(argv[i], "-b") == 0 && ++i < argc) {
            if (strcmp(argv[i], "read") == 0) in.binary_mode = READ;
            else if (strcmp(argv[i], "write") == 0) in.binary_mode = WRITE;
            else { printf("Invalid binary mode\n"); exit(1); }
        } else if (strcmp(argv[i], "-k") == 0 && ++i < argc) {
            in.kernel_id = atoi(argv[i]);
        } else {
            printf("Unknown argument %s\n", argv[i]); exit(1);
        }
    }

    if (strcmp(in.HM, "small") == 0) in.n_isotopes = 68;
    else if (strcmp(in.HM, "XL") == 0 && !user_g) in.n_gridpoints = 238847;
    else if (strcmp(in.HM, "XXL") == 0 && !user_g) in.n_gridpoints = static_cast<long>(238847 * 2.1);

    return in;
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int mype = 0;
    int nprocs = 1;
    double start, end;
    unsigned long long verification = 0;

    Inputs in = read_CLI(argc, argv);

    // Device selection (prefer GPU, fall back to CPU)
    sycl::device dev = sycl::default_selector{}.select_device();
    sycl::queue q(dev, sycl::property::queue::in_order());

    // Initialize data (binary I/O omitted for brevity)
    SimulationData hostSD = grid_init_do_not_profile(in);
    SimulationData devSD = move_simulation_data_to_device(in, hostSD, q);

    if (mype == 0) printf("\n=== SIMULATION START ===\n");
    start = get_time();

    // Dispatch selected kernel
    switch (in.kernel_id) {
        case 0:
            verification = run_event_based_simulation_baseline(in, devSD, q);
            break;
        case 1:
            verification = run_event_based_simulation_optimization_1(in, devSD, q);
            break;
        case 2:
            verification = run_event_based_simulation_optimization_2(in, devSD, q);
            break;
        case 3:
            verification = run_event_based_simulation_optimization_3(in, devSD, q);
            break;
        case 4:
            verification = run_event_based_simulation_optimization_4(in, devSD, q);
            break;
        case 5:
            verification = run_event_based_simulation_optimization_5(in, devSD, q);
            break;
        case 6:
            verification = run_event_based_simulation_optimization_6(in, devSD, q);
            break;
        default:
            printf("Error: No kernel ID %d found!\n", in.kernel_id);
            exit(1);
    }

    end = get_time();
    verification %= 999983ULL;

    int invalid = print_results(in, mype, end - start, nprocs, verification);

    release_device_memory(devSD, q);
    return invalid;
}
