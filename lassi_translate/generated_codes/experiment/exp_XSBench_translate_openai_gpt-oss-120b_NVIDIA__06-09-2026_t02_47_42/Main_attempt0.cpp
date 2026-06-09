// ==================== SYCL XSBench ====================
//   Translated from CUDA to SYCL (oneAPI)
//   Energy‑aware guidelines applied where reasonable
// =====================================================

#include <CL/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

// ------------------- Constants -------------------
#define UNIONIZED 0
#define NUCLIDE  1
#define HASH     2

#define HISTORY_BASED 1
#define EVENT_BASED   2

#define NONE 0
#define READ 1
#define WRITE 2

#define STARTING_SEED 1070ULL

// ------------------- Types -------------------
struct NuclideGridPoint {
    double energy;
    double total_xs;
    double elastic_xs;
    double absorbtion_xs;
    double fission_xs;
    double nu_fission_xs;
};

struct Inputs {
    int   nthreads;
    long  n_isotopes;
    long  n_gridpoints;
    int   lookups;
    char* HM;
    int   grid_type;
    int   hash_bins;
    int   particles;
    int   simulation_method;
    int   binary_mode;
    int   kernel_id;
};

struct SimulationData {
    int*    num_nucs;                     // length_num_nucs
    double* concs;                        // length_concs
    int*    mats;                         // length_mats
    double* unionized_energy_array;       // length_unionized_energy_array
    int*    index_grid;                   // length_index_grid
    NuclideGridPoint* nuclide_grid;       // length_nuclide_grid

    int    length_num_nucs;
    int    length_concs;
    int    length_mats;
    int    length_unionized_energy_array;
    long   length_index_grid;
    int    length_nuclide_grid;
    int    max_num_nucs;

    unsigned long* verification;          // length_verification
    int    length_verification;

    double* p_energy_samples;
    int    length_p_energy_samples;
    int*   mat_samples;
    int    length_mat_samples;
};

// ------------------- Utility Functions -------------------
static inline void gpuAssert(sycl::exception const& e, const char* file, int line) {
    std::cerr << "SYCL exception: " << e.what() << " at " << file << ":" << line << "\n";
    std::exit(1);
}
#define gpuErrchk(call) \
    try { call; } catch (sycl::exception const& e) { gpuAssert(e, __FILE__, __LINE__); }

// ------------------- RNG -------------------
static inline double LCG_random_double(uint64_t* seed) {
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
    n %= m;
    uint64_t a_new = 1, c_new = 0;
    while (n) {
        if (n & 1) {
            a_new = a_new * a;
            c_new = c_new * a + c;
        }
        c = c * (a + 1);
        a = a * a;
        n >>= 1;
    }
    return (a_new * seed + c_new) % m;
}

// ------------------- Comparison -------------------
int double_compare(const void* a, const void* b) {
    double A = *(const double*)a;
    double B = *(const double*)b;
    return (A > B) - (A < B);
}
int NGP_compare(const void* a, const void* b) {
    const NuclideGridPoint* A = (const NuclideGridPoint*)a;
    const NuclideGridPoint* B = (const NuclideGridPoint*)b;
    return (A->energy > B->energy) - (A->energy < B->energy);
}

// ------------------- Search Functions -------------------
static inline long grid_search(long n, double quarry, const double* A) {
    long lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        long mid = lo + (hi - lo) / 2;
        if (A[mid] > quarry) hi = mid;
        else lo = mid;
    }
    return lo;
}
static inline long grid_search_nuclide(long n, double quarry,
                                      const NuclideGridPoint* A,
                                      long low, long high) {
    long lo = low, hi = high;
    while (hi - lo > 1) {
        long mid = lo + (hi - lo) / 2;
        if (A[mid].energy > quarry) hi = mid;
        else lo = mid;
    }
    return lo;
}

// ------------------- Material Picker -------------------
static inline int pick_mat(uint64_t* seed) {
    const double dist[12] = {0.140,0.052,0.275,0.134,0.154,0.064,
                             0.066,0.055,0.008,0.015,0.025,0.013};
    double roll = LCG_random_double(seed);
    double cum = 0.0;
    for (int i = 0; i < 12; ++i) {
        cum += dist[i];
        if (roll < cum) return i;
    }
    return 0;
}

// ------------------- Core XS Calculations -------------------
static inline void calculate_micro_xs(double p_energy, int nuc,
                                     long n_isotopes, long n_gridpoints,
                                     const double* egrid, const int* index_data,
                                     const NuclideGridPoint* nuclide_grids,
                                     long idx, double* xs_vector,
                                     int grid_type, int hash_bins) {
    const NuclideGridPoint* low;
    const NuclideGridPoint* high;
    if (grid_type == NUCLIDE) {
        idx = grid_search_nuclide(n_gridpoints, p_energy,
                                  nuclide_grids + nuc * n_gridpoints, 0, n_gridpoints - 1);
        low = (idx == n_gridpoints - 1) ?
              &nuclide_grids[nuc * n_gridpoints + idx - 1] :
              &nuclide_grids[nuc * n_gridpoints + idx];
    } else if (grid_type == UNIONIZED) {
        int id = index_data[idx * n_isotopes + nuc];
        low = (id == n_gridpoints - 1) ?
              &nuclide_grids[nuc * n_gridpoints + id - 1] :
              &nuclide_grids[nuc * n_gridpoints + id];
    } else { // HASH
        int u_low = index_data[idx * n_isotopes + nuc];
        int u_high = (idx == hash_bins - 1) ?
                     n_gridpoints - 1 :
                     index_data[(idx + 1) * n_isotopes + nuc] + 1;
        double e_low  = nuclide_grids[nuc * n_gridpoints + u_low].energy;
        double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
        int lower;
        if (p_energy <= e_low) lower = 0;
        else if (p_energy >= e_high) lower = n_gridpoints - 1;
        else lower = grid_search_nuclide(n_gridpoints, p_energy,
                                         nuclide_grids + nuc * n_gridpoints,
                                         u_low, u_high);
        low = (lower == n_gridpoints - 1) ?
              &nuclide_grids[nuc * n_gridpoints + lower - 1] :
              &nuclide_grids[nuc * n_gridpoints + lower];
    }
    high = low + 1;
    double f = (high->energy - p_energy) / (high->energy - low->energy);
    xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
    xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
    xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
    xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
    xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

static inline void calculate_macro_xs(double p_energy, int mat,
                                      long n_isotopes, long n_gridpoints,
                                      const int* num_nucs, const double* concs,
                                      const double* egrid, const int* index_data,
                                      const NuclideGridPoint* nuclide_grids,
                                      const int* mats, double* macro_xs_vector,
                                      int grid_type, int hash_bins, int max_num_nucs) {
    for (int k = 0; k < 5; ++k) macro_xs_vector[k] = 0.0;
    long idx = -1;
    if (grid_type == UNIONIZED) idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
    else if (grid_type == HASH) {
        double du = 1.0 / hash_bins;
        idx = static_cast<long>(p_energy / du);
    }
    for (int j = 0; j < num_nucs[mat]; ++j) {
        double xs_vec[5];
        int p_nuc = mats[mat * max_num_nucs + j];
        double conc = concs[mat * max_num_nucs + j];
        calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints,
                           egrid, index_data, nuclide_grids,
                           idx, xs_vec, grid_type, hash_bins);
        for (int k = 0; k < 5; ++k) macro_xs_vector[k] += xs_vec[k] * conc;
    }
}

// ------------------- Data Generation -------------------
int* load_num_nucs(long n_isotopes) {
    int* num_nucs = (int*)malloc(12 * sizeof(int));
    num_nucs[0] = (n_isotopes == 68) ? 34 : 321;
    num_nucs[1] = 5;  num_nucs[2] = 4;  num_nucs[3] = 4;
    num_nucs[4] = 27; num_nucs[5] = 21; num_nucs[6] = 21;
    num_nucs[7] = 21; num_nucs[8] = 21; num_nucs[9] = 21;
    num_nucs[10] = 9; num_nucs[11] = 9;
    return num_nucs;
}
int* load_mats(int* num_nucs, long n_isotopes, int* max_num_nucs) {
    *max_num_nucs = 0;
    for (int i = 0; i < 12; ++i) *max_num_nucs = std::max(*max_num_nucs, num_nucs[i]);
    int* mats = (int*)malloc(12 * (*max_num_nucs) * sizeof(int));
    // Fuel
    if (n_isotopes == 68) {
        int fuel[34] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
        std::copy(fuel, fuel + 34, mats);
    } else {
        int fuel[321];
        int base[34] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
        std::copy(base, base + 34, fuel);
        for (int i = 0; i < 321 - 34; ++i) fuel[34 + i] = 68 + i;
        std::copy(fuel, fuel + 321, mats);
    }
    // Non‑fuel (same as original)
    int mats1[5] = {63,64,65,66,67};
    int mats2[4] = {24,41,4,5};
    int mats3[4] = {24,41,4,5};
    int mats4[27] = {19,20,21,22,35,36,37,38,39,25,27,28,29,30,31,32,26,49,50,51,11,12,13,14,6,16,17};
    int mats5[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
    int mats6[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
    int mats7[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
    int mats8[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
    int mats9[21] = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,49,50,51,11,12,13,14};
    int mats10[9] = {24,41,4,5,63,64,65,66,67};
    int mats11[9] = {24,41,4,5,63,64,65,66,67};

    int offset = *max_num_nucs;
    std::copy(mats1, mats1 + 5, mats + offset * 1);
    std::copy(mats2, mats2 + 4, mats + offset * 2);
    std::copy(mats3, mats3 + 4, mats + offset * 3);
    std::copy(mats4, mats4 + 27, mats + offset * 4);
    std::copy(mats5, mats5 + 21, mats + offset * 5);
    std::copy(mats6, mats6 + 21, mats + offset * 6);
    std::copy(mats7, mats7 + 21, mats + offset * 7);
    std::copy(mats8, mats8 + 21, mats + offset * 8);
    std::copy(mats9, mats9 + 21, mats + offset * 9);
    std::copy(mats10, mats10 + 9, mats + offset *10);
    std::copy(mats11, mats11 + 9, mats + offset *11);
    return mats;
}
double* load_concs(int* num_nucs, int max_num_nucs) {
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double* concs = (double*)malloc(12 * max_num_nucs * sizeof(double));
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < num_nucs[i]; ++j)
            concs[i * max_num_nucs + j] = LCG_random_double(&seed);
    return concs;
}

// ------------------- Memory Management -------------------
SimulationData move_simulation_data_to_device(sycl::queue& q, const Inputs& in,
                                             const SimulationData& hostSD) {
    SimulationData devSD = hostSD; // shallow copy of pointers
    // Allocate device buffers (USM)
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
    devSD.length_verification = in.lookups;

    // optional extra buffers (set to nullptr)
    devSD.p_energy_samples = nullptr;
    devSD.mat_samples      = nullptr;
    return devSD;
}
void release_device_memory(sycl::queue& q, SimulationData& devSD) {
    sycl::free(devSD.num_nucs, q);
    sycl::free(devSD.concs, q);
    sycl::free(devSD.mats, q);
    sycl::free(devSD.unionized_energy_array, q);
    sycl::free(devSD.index_grid, q);
    sycl::free(devSD.nuclide_grid, q);
    sycl::free(devSD.verification, q);
    if (devSD.p_energy_samples) sycl::free(devSD.p_energy_samples, q);
    if (devSD.mat_samples)      sycl::free(devSD.mat_samples, q);
}

// ------------------- Baseline Simulation -------------------
unsigned long long run_event_based_simulation_baseline(sycl::queue& q,
                                                       const Inputs& in,
                                                       SimulationData& GSD) {
    const int nthreads = 256;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);
            double macro_xs[5] = {0};
            calculate_macro_xs(p_energy, mat,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array, GSD.index_grid,
                               GSD.nuclide_grid, GSD.mats,
                               macro_xs, in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    // Reduction using oneDPL
    auto dev_ptr = GSD.verification;
    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      dev_ptr, dev_ptr + in.lookups, 0UL);
    return result;
}

// ------------------- Optimization 1 (sample/lookup split) -------------------
unsigned long long run_event_based_simulation_optimization_1(sycl::queue& q,
                                                            const Inputs& in,
                                                            SimulationData& GSD) {
    // allocate extra buffers
    GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    GSD.mat_samples      = sycl::malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    // sampling kernel
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // lookup kernel
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i], GSD.mat_samples[i],
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array, GSD.index_grid,
                               GSD.nuclide_grid, GSD.mats,
                               macro_xs, in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      GSD.verification, GSD.verification + in.lookups, 0UL);
    return result;
}

// ------------------- Optimization 2 (material‑specific kernels) -------------------
unsigned long long run_event_based_simulation_optimization_2(sycl::queue& q,
                                                            const Inputs& in,
                                                            SimulationData& GSD) {
    // allocate extra buffers
    GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    GSD.mat_samples      = sycl::malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    // sampling
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // material kernels
    for (int m = 0; m < 12; ++m) {
        q.submit([&, m](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
                int i = idx[0];
                if (i >= in.lookups) return;
                if (GSD.mat_samples[i] != m) return;
                double macro_xs[5] = {0};
                calculate_macro_xs(GSD.p_energy_samples[i], m,
                                   in.n_isotopes, in.n_gridpoints,
                                   GSD.num_nucs, GSD.concs,
                                   GSD.unionized_energy_array, GSD.index_grid,
                                   GSD.nuclide_grid, GSD.mats,
                                   macro_xs, in.grid_type, in.hash_bins,
                                   GSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j)
                    if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
                GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
        }).wait();
    }

    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      GSD.verification, GSD.verification + in.lookups, 0UL);
    return result;
}

// ------------------- Optimization 3 (fuel / non‑fuel) -------------------
unsigned long long run_event_based_simulation_optimization_3(sycl::queue& q,
                                                            const Inputs& in,
                                                            SimulationData& GSD) {
    // allocate extra buffers
    GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    GSD.mat_samples      = sycl::malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    // sampling
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // fuel kernel (material 0)
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            int mat = GSD.mat_samples[i];
            if (mat != 0) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i], mat,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array, GSD.index_grid,
                               GSD.nuclide_grid, GSD.mats,
                               macro_xs, in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    // non‑fuel kernel
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            int mat = GSD.mat_samples[i];
            if (mat == 0) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i], mat,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array, GSD.index_grid,
                               GSD.nuclide_grid, GSD.mats,
                               macro_xs, in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      GSD.verification, GSD.verification + in.lookups, 0UL);
    return result;
}

// ------------------- Optimization 4 (material sort) -------------------
unsigned long long run_event_based_simulation_optimization_4(sycl::queue& q,
                                                            const Inputs& in,
                                                            SimulationData& GSD) {
    // allocate extra buffers
    GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    GSD.mat_samples      = sycl::malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    // sampling
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // count per material
    int lookups_per_mat[12];
    for (int m = 0; m < 12; ++m) {
        lookups_per_mat[m] = std::count_if(
            oneapi::dpl::execution::make_device_policy(q),
            GSD.mat_samples, GSD.mat_samples + in.lookups,
            [=](int v){ return v == m; });
    }

    // sort by material (key = mat, value = energy)
    oneapi::dpl::sort_by_key(oneapi::dpl::execution::make_device_policy(q),
                             GSD.mat_samples, GSD.mat_samples + in.lookups,
                             GSD.p_energy_samples);

    // launch per‑material kernels
    int offset = 0;
    for (int m = 0; m < 12; ++m) {
        int nlook = lookups_per_mat[m];
        if (nlook == 0) continue;
        int nb = (nlook + nthreads - 1) / nthreads;
        q.submit([&, m, nlook, offset](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nb * nthreads), [=](sycl::id<1> idx) {
                int i = idx[0];
                if (i >= nlook) return;
                i += offset;
                double macro_xs[5] = {0};
                calculate_macro_xs(GSD.p_energy_samples[i], m,
                                   in.n_isotopes, in.n_gridpoints,
                                   GSD.num_nucs, GSD.concs,
                                   GSD.unionized_energy_array, GSD.index_grid,
                                   GSD.nuclide_grid, GSD.mats,
                                   macro_xs, in.grid_type, in.hash_bins,
                                   GSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j)
                    if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
                GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
        }).wait();
        offset += nlook;
    }

    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      GSD.verification, GSD.verification + in.lookups, 0UL);
    return result;
}

// ------------------- Optimization 5 (fuel / non‑fuel partition) -------------------
struct is_fuel {
    bool operator()(int v) const { return v == 0; }
};

unsigned long long run_event_based_simulation_optimization_5(sycl::queue& q,
                                                            const Inputs& in,
                                                            SimulationData& GSD) {
    // allocate extra buffers
    GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    GSD.mat_samples      = sycl::malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    // sampling
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // count fuel
    int n_fuel = std::count_if(
        oneapi::dpl::execution::make_device_policy(q),
        GSD.mat_samples, GSD.mat_samples + in.lookups,
        [](int v){ return v == 0; });

    // partition fuel to front
    oneapi::dpl::partition(oneapi::dpl::execution::make_device_policy(q),
                           GSD.mat_samples, GSD.mat_samples + in.lookups,
                           GSD.p_energy_samples, is_fuel());

    // fuel kernel
    int nb_fuel = (n_fuel + nthreads - 1) / nthreads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nb_fuel * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= n_fuel) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i], 0,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array, GSD.index_grid,
                               GSD.nuclide_grid, GSD.mats,
                               macro_xs, in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    // non‑fuel kernel
    int n_nonfuel = in.lookups - n_fuel;
    int nb_nonfuel = (n_nonfuel + nthreads - 1) / nthreads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nb_nonfuel * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= n_nonfuel) return;
            i += n_fuel;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i], GSD.mat_samples[i],
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array, GSD.index_grid,
                               GSD.nuclide_grid, GSD.mats,
                               macro_xs, in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      GSD.verification, GSD.verification + in.lookups, 0UL);
    return result;
}

// ------------------- Optimization 6 (material + energy sort) -------------------
unsigned long long run_event_based_simulation_optimization_6(sycl::queue& q,
                                                            const Inputs& in,
                                                            SimulationData& GSD) {
    // allocate extra buffers
    GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
    GSD.mat_samples      = sycl::malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks  = (in.lookups + nthreads - 1) / nthreads;

    // sampling
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // count per material
    int lookups_per_mat[12];
    for (int m = 0; m < 12; ++m) {
        lookups_per_mat[m] = std::count_if(
            oneapi::dpl::execution::make_device_policy(q),
            GSD.mat_samples, GSD.mat_samples + in.lookups,
            [=](int v){ return v == m; });
    }

    // sort by material
    oneapi::dpl::sort_by_key(oneapi::dpl::execution::make_device_policy(q),
                             GSD.mat_samples, GSD.mat_samples + in.lookups,
                             GSD.p_energy_samples);

    // per‑material energy sort
    int offset = 0;
    for (int m = 0; m < 12; ++m) {
        int cnt = lookups_per_mat[m];
        if (cnt == 0) continue;
        oneapi::dpl::sort_by_key(oneapi::dpl::execution::make_device_policy(q),
                                 GSD.p_energy_samples + offset,
                                 GSD.p_energy_samples + offset + cnt,
                                 GSD.mat_samples + offset);
        offset += cnt;
    }

    // launch per‑material kernels (reuse kernel from opt‑4)
    offset = 0;
    for (int m = 0; m < 12; ++m) {
        int nlook = lookups_per_mat[m];
        if (nlook == 0) continue;
        int nb = (nlook + nthreads - 1) / nthreads;
        q.submit([&, m, nlook, offset](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nb * nthreads), [=](sycl::id<1> idx) {
                int i = idx[0];
                if (i >= nlook) return;
                i += offset;
                double macro_xs[5] = {0};
                calculate_macro_xs(GSD.p_energy_samples[i], m,
                                   in.n_isotopes, in.n_gridpoints,
                                   GSD.num_nucs, GSD.concs,
                                   GSD.unionized_energy_array, GSD.index_grid,
                                   GSD.nuclide_grid, GSD.mats,
                                   macro_xs, in.grid_type, in.hash_bins,
                                   GSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j)
                    if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
                GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
        }).wait();
        offset += nlook;
    }

    unsigned long result = std::reduce(oneapi::dpl::execution::make_device_policy(q),
                                      GSD.verification, GSD.verification + in.lookups, 0UL);
    return result;
}

// ------------------- Grid Initialization (host) -------------------
SimulationData grid_init_do_not_profile(const Inputs& in) {
    SimulationData SD{};
    uint64_t seed = 42ULL;

    // nuclide grid
    SD.length_nuclide_grid = static_cast<int>(in.n_isotopes * in.n_gridpoints);
    SD.nuclide_grid = (NuclideGridPoint*)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    for (int i = 0; i < SD.length_nuclide_grid; ++i) {
        SD.nuclide_grid[i].energy        = LCG_random_double(&seed);
        SD.nuclide_grid[i].total_xs      = LCG_random_double(&seed);
        SD.nuclide_grid[i].elastic_xs    = LCG_random_double(&seed);
        SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].fission_xs    = LCG_random_double(&seed);
        SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
    }
    for (int i = 0; i < in.n_isotopes; ++i)
        qsort(&SD.nuclide_grid[i * in.n_gridpoints], in.n_gridpoints,
              sizeof(NuclideGridPoint), NGP_compare);

    // acceleration structures
    if (in.grid_type == UNIONIZED) {
        SD.length_unionized_energy_array = static_cast<int>(in.n_isotopes * in.n_gridpoints);
        SD.unionized_energy_array = (double*)malloc(SD.length_unionized_energy_array * sizeof(double));
        for (int i = 0; i < SD.length_unionized_energy_array; ++i)
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        qsort(SD.unionized_energy_array, SD.length_unionized_energy_array,
              sizeof(double), double_compare);

        SD.length_index_grid = static_cast<long>(SD.length_unionized_energy_array * in.n_isotopes);
        SD.index_grid = (int*)malloc(SD.length_index_grid * sizeof(int));
        int* idx_low = (int*)calloc(in.n_isotopes, sizeof(int));
        double* energy_high = (double*)malloc(in.n_isotopes * sizeof(double));
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
                    ++idx_low[i];
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
        SD.index_grid = (int*)malloc(SD.length_index_grid * sizeof(int));
        double du = 1.0 / in.hash_bins;
        for (long e = 0; e < in.hash_bins; ++e) {
            double energy = e * du;
            for (long i = 0; i < in.n_isotopes; ++i) {
                SD.index_grid[e * in.n_isotopes + i] =
                    grid_search_nuclide(in.n_gridpoints, energy,
                                        SD.nuclide_grid + i * in.n_gridpoints, 0,
                                        in.n_gridpoints - 1);
            }
        }
    } else {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = 0;
    }

    // materials & concentrations
    SD.num_nucs = load_num_nucs(in.n_isotopes);
    SD.length_num_nucs = 12;
    SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
    SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
    SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
    SD.length_concs = SD.length_mats;

    return SD;
}

// ------------------- Binary I/O (simplified) -------------------
void binary_write(const Inputs& in, const SimulationData& SD) {
    std::ofstream fp("XS_data.dat", std::ios::binary);
    fp.write(reinterpret_cast<const char*>(&SD), sizeof(SimulationData));
    fp.write(reinterpret_cast<const char*>(SD.num_nucs), SD.length_num_nucs * sizeof(int));
    fp.write(reinterpret_cast<const char*>(SD.concs), SD.length_concs * sizeof(double));
    fp.write(reinterpret_cast<const char*>(SD.mats), SD.length_mats * sizeof(int));
    fp.write(reinterpret_cast<const char*>(SD.nuclide_grid),
             SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    fp.write(reinterpret_cast<const char*>(SD.index_grid),
             SD.length_index_grid * sizeof(int));
    fp.write(reinterpret_cast<const char*>(SD.unionized_energy_array),
             SD.length_unionized_energy_array * sizeof(double));
}
SimulationData binary_read(const Inputs& in) {
    SimulationData SD{};
    std::ifstream fp("XS_data.dat", std::ios::binary);
    fp.read(reinterpret_cast<char*>(&SD), sizeof(SimulationData));
    SD.num_nucs = (int*)malloc(SD.length_num_nucs * sizeof(int));
    SD.concs = (double*)malloc(SD.length_concs * sizeof(double));
    SD.mats = (int*)malloc(SD.length_mats * sizeof(int));
    SD.nuclide_grid = (NuclideGridPoint*)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    SD.index_grid = (int*)malloc(SD.length_index_grid * sizeof(int));
    SD.unionized_energy_array = (double*)malloc(SD.length_unionized_energy_array * sizeof(double));
    fp.read(reinterpret_cast<char*>(SD.num_nucs), SD.length_num_nucs * sizeof(int));
    fp.read(reinterpret_cast<char*>(SD.concs), SD.length_concs * sizeof(double));
    fp.read(reinterpret_cast<char*>(SD.mats), SD.length_mats * sizeof(int));
    fp.read(reinterpret_cast<char*>(SD.nuclide_grid),
            SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    fp.read(reinterpret_cast<char*>(SD.index_grid),
            SD.length_index_grid * sizeof(int));
    fp.read(reinterpret_cast<char*>(SD.unionized_energy_array),
            SD.length_unionized_energy_array * sizeof(double));
    return SD;
}

// ------------------- Timing -------------------
double get_time() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ------------------- CLI Parsing (minimal) -------------------
Inputs read_CLI(int argc, char* argv[]) {
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
    in.HM = (char*)malloc(6);
    strcpy(in.HM, "large");

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-g")) in.n_gridpoints = atol(argv[++i]);
        else if (!strcmp(argv[i], "-m")) {
            const char* t = argv[++i];
            if (!strcmp(t, "history")) in.simulation_method = HISTORY_BASED;
            else if (!strcmp(t, "event")) {
                in.simulation_method = EVENT_BASED;
                in.lookups = in.lookups * in.particles;
                in.particles = 0;
            }
        } else if (!strcmp(argv[i], "-l")) in.lookups = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) in.hash_bins = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p")) in.particles = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")) in.HM = argv[++i];
        else if (!strcmp(argv[i], "-G")) {
            const char* t = argv[++i];
            if (!strcmp(t, "unionized")) in.grid_type = UNIONIZED;
            else if (!strcmp(t, "nuclide")) in.grid_type = NUCLIDE;
            else if (!strcmp(t, "hash")) in.grid_type = HASH;
        } else if (!strcmp(argv[i], "-b")) {
            const char* t = argv[++i];
            if (!strcmp(t, "read")) in.binary_mode = READ;
            else if (!strcmp(t, "write")) in.binary_mode = WRITE;
        } else if (!strcmp(argv[i], "-k")) in.kernel_id = atoi(argv[++i]);
        else { printf("Invalid option %s\n", argv[i]); exit(1); }
    }

    if (!strcasecmp(in.HM, "small")) in.n_isotopes = 68;
    else if (!strcasecmp(in.HM, "XL") && in.n_gridpoints == 11303) in.n_gridpoints = 238847;
    else if (!strcasecmp(in.HM, "XXL") && in.n_gridpoints == 11303) in.n_gridpoints = static_cast<long>(238847 * 2.1);
    return in;
}

// ------------------- Result Printing -------------------
int print_results(const Inputs& in, double runtime, unsigned long long vhash) {
    int lookups = (in.simulation_method == EVENT_BASED) ? in.lookups
                                                       : in.lookups * in.particles;
    int lookups_per_sec = static_cast<int>(lookups / runtime);
    printf("\nRuntime: %.3lf s\nLookups: %d\nLookups/s: %d\n", runtime, lookups, lookups_per_sec);
    unsigned long long ref = (strcmp(in.HM, "large") == 0) ? 952131ULL : 945990ULL;
    if (vhash == ref) printf("Verification checksum: %llu (Valid)\n", vhash);
    else printf("Verification checksum: %llu (Invalid)\n", vhash);
    return (vhash == ref) ? 0 : 1;
}

// ------------------- Main -------------------
int main(int argc, char* argv[]) {
    // Select low‑power device (example: CPU or integrated GPU)
    sycl::device dev = sycl::default_selector{}.select_device();
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::cout << "Using device: " << dev.get_info<sycl::info::device::name>() << "\n";

    Inputs in = read_CLI(argc, argv);
    SimulationData SD;

    if (in.binary_mode == READ) SD = binary_read(in);
    else SD = grid_init_do_not_profile(in);

    if (in.binary_mode == WRITE) binary_write(in, SD);

    SimulationData GSD = move_simulation_data_to_device(q, in, SD);

    double t0 = get_time();
    unsigned long long verification = 0;
    if (in.simulation_method == EVENT_BASED) {
        switch (in.kernel_id) {
            case 0: verification = run_event_based_simulation_baseline(q, in, GSD); break;
            case 1: verification = run_event_based_simulation_optimization_1(q, in, GSD); break;
            case 2: verification = run_event_based_simulation_optimization_2(q, in, GSD); break;
            case 3: verification = run_event_based_simulation_optimization_3(q, in, GSD); break;
            case 4: verification = run_event_based_simulation_optimization_4(q, in, GSD); break;
            case 5: verification = run_event_based_simulation_optimization_5(q, in, GSD); break;
            case 6: verification = run_event_based_simulation_optimization_6(q, in, GSD); break;
            default: std::cerr << "Invalid kernel ID\n"; std::exit(1);
        }
    } else {
        std::cerr << "Only event‑based method supported in SYCL version.\n";
        std::exit(1);
    }
    double t1 = get_time();

    release_device_memory(q, GSD);
    verification %= 999983ULL;
    return print_results(in, t1 - t0, verification);
}
