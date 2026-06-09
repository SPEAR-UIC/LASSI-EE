// ==================== SYCL XSBench ====================
//  Single‑file SYCL implementation of the XSBench CUDA benchmark
//  (energy‑efficient translation, see context knowledge for guidelines)

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>
#include <numeric>
#include <fstream>

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

using namespace sycl;

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
    int *num_nucs{};
    double *concs{};
    int *mats{};
    double *unionized_energy_array{};
    int *index_grid{};
    NuclideGridPoint *nuclide_grid{};
    int length_num_nucs{};
    int length_concs{};
    int length_mats{};
    int length_unionized_energy_array{};
    long length_index_grid{};
    int length_nuclide_grid{};
    int max_num_nucs{};
    unsigned long *verification{};
    int length_verification{};
    double *p_energy_samples{};
    int length_p_energy_samples{};
    int *mat_samples{};
    int length_mat_samples{};
};

// ------------------------------------------------------------------
// Utility functions (host)
// ------------------------------------------------------------------
static inline double get_time() {
    auto us = std::chrono::high_resolution_clock::now().time_since_epoch() /
              std::chrono::microseconds(1);
    return static_cast<double>(us) * 1e-6;
}

static void fancy_int(long a) {
    if (a < 1000) printf("%ld\n", a);
    else if (a < 1000000) printf("%ld,%03ld\n", a / 1000, a % 1000);
    else if (a < 1000000000) printf("%ld,%03ld,%03ld\n",
        a / 1000000, (a % 1000000) / 1000, a % 1000);
    else printf("%ld,%03ld,%03ld,%03ld\n",
        a / 1000000000, (a % 1000000000) / 1000000,
        (a % 1000000) / 1000, a % 1000);
}

// ------------------------------------------------------------------
// RNG helpers (device)
// ------------------------------------------------------------------
inline double LCG_random_double(uint64_t *seed) {
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return static_cast<double>(*seed) / static_cast<double>(m);
}
inline uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
    const uint64_t m = 9223372036854775808ULL;
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n %= m;
    uint64_t a_new = 1, c_new = 0;
    while (n) {
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
inline int pick_mat(uint64_t *seed) {
    const double dist[12] = {0.140,0.052,0.275,0.134,0.154,0.064,
                             0.066,0.055,0.008,0.015,0.025,0.013};
    double roll = LCG_random_double(seed);
    for (int i = 0; i < 12; ++i) {
        double running = 0.0;
        for (int j = i; j > 0; --j) running += dist[j];
        if (roll < running) return i;
    }
    return 0;
}

// ------------------------------------------------------------------
// Search helpers (device)
// ------------------------------------------------------------------
inline long grid_search(long n, double quarry, const double *A) {
    long low = 0, high = n - 1;
    while (high - low > 1) {
        long mid = low + (high - low) / 2;
        if (A[mid] > quarry) high = mid;
        else low = mid;
    }
    return low;
}
inline long grid_search_nuclide(long n, double quarry,
                               const NuclideGridPoint *A,
                               long low, long high) {
    while (high - low > 1) {
        long mid = low + (high - low) / 2;
        if (A[mid].energy > quarry) high = mid;
        else low = mid;
    }
    return low;
}

// ------------------------------------------------------------------
// Cross‑section calculations (device)
// ------------------------------------------------------------------
inline void calculate_micro_xs(double p_energy, int nuc,
                               long n_isotopes, long n_gridpoints,
                               const double *egrid, const int *index_data,
                               const NuclideGridPoint *nuclide_grids,
                               long idx, double *xs_vector,
                               int grid_type, int hash_bins) {
    const NuclideGridPoint *low, *high;
    if (grid_type == NUCLIDE) {
        idx = grid_search_nuclide(n_gridpoints, p_energy,
                                  nuclide_grids + nuc * n_gridpoints, 0,
                                  n_gridpoints - 1);
        low = (idx == n_gridpoints - 1) ?
              &nuclide_grids[nuc * n_gridpoints + idx - 1] :
              &nuclide_grids[nuc * n_gridpoints + idx];
    } else if (grid_type == UNIONIZED) {
        long ind = index_data[idx * n_isotopes + nuc];
        low = (ind == n_gridpoints - 1) ?
              &nuclide_grids[nuc * n_gridpoints + ind - 1] :
              &nuclide_grids[nuc * n_gridpoints + ind];
    } else { // HASH
        int u_low = index_data[idx * n_isotopes + nuc];
        int u_high = (idx == hash_bins - 1) ?
                     n_gridpoints - 1 :
                     index_data[(idx + 1) * n_isotopes + nuc] + 1;
        double e_low = nuclide_grids[nuc * n_gridpoints + u_low].energy;
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
inline void calculate_macro_xs(double p_energy, int mat,
                               long n_isotopes, long n_gridpoints,
                               const int *num_nucs, const double *concs,
                               const double *egrid, const int *index_data,
                               const NuclideGridPoint *nuclide_grids,
                               const int *mats, double *macro_xs_vector,
                               int grid_type, int hash_bins, int max_num_nucs) {
    for (int k = 0; k < 5; ++k) macro_xs_vector[k] = 0.0;
    long idx = -1;
    if (grid_type == UNIONIZED) idx = grid_search(n_isotopes * n_gridpoints,
                                                 p_energy, egrid);
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

// ------------------------------------------------------------------
// Memory management (SYCL)
// ------------------------------------------------------------------
SimulationData move_simulation_data_to_device(const Inputs &in,
                                              SimulationData &SD,
                                              queue &q) {
    SimulationData GSD = SD;
    GSD.max_num_nucs = SD.max_num_nucs;

    GSD.num_nucs = malloc_device<int>(SD.length_num_nucs, q);
    q.memcpy(GSD.num_nucs, SD.num_nucs,
             SD.length_num_nucs * sizeof(int)).wait();

    GSD.concs = malloc_device<double>(SD.length_concs, q);
    q.memcpy(GSD.concs, SD.concs,
             SD.length_concs * sizeof(double)).wait();

    GSD.mats = malloc_device<int>(SD.length_mats, q);
    q.memcpy(GSD.mats, SD.mats,
             SD.length_mats * sizeof(int)).wait();

    GSD.unionized_energy_array =
        malloc_device<double>(SD.length_unionized_energy_array, q);
    q.memcpy(GSD.unionized_energy_array, SD.unionized_energy_array,
             SD.length_unionized_energy_array * sizeof(double)).wait();

    GSD.index_grid = malloc_device<int>(SD.length_index_grid, q);
    q.memcpy(GSD.index_grid, SD.index_grid,
             SD.length_index_grid * sizeof(int)).wait();

    GSD.nuclide_grid =
        malloc_device<NuclideGridPoint>(SD.length_nuclide_grid, q);
    q.memcpy(GSD.nuclide_grid, SD.nuclide_grid,
             SD.length_nuclide_grid * sizeof(NuclideGridPoint)).wait();

    GSD.verification = malloc_device<unsigned long>(in.lookups, q);
    q.memset(GSD.verification, 0, in.lookups * sizeof(unsigned long)).wait();
    GSD.length_verification = in.lookups;

    return GSD;
}
void release_device_memory(SimulationData &GSD, queue &q) {
    free(GSD.num_nucs, q);
    free(GSD.concs, q);
    free(GSD.mats, q);
    free(GSD.unionized_energy_array, q);
    free(GSD.index_grid, q);
    free(GSD.nuclide_grid, q);
    free(GSD.verification, q);
    if (GSD.p_energy_samples) free(GSD.p_energy_samples, q);
    if (GSD.mat_samples) free(GSD.mat_samples, q);
}

// ------------------------------------------------------------------
// Baseline kernel
// ------------------------------------------------------------------
unsigned long long run_event_based_simulation_baseline(const Inputs &in,
                                                       SimulationData &GSD,
                                                       queue &q) {
    const int nthreads = 256;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);
            double macro_xs[5] = {0};
            calculate_macro_xs(p_energy, mat, in.n_isotopes,
                               in.n_gridpoints, GSD.num_nucs,
                               GSD.concs, GSD.unionized_energy_array,
                               GSD.index_grid, GSD.nuclide_grid,
                               GSD.mats, macro_xs,
                               in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    unsigned long long checksum = std::accumulate(host_verif.begin(),
                                                  host_verif.end(),
                                                  0ULL);
    return checksum;
}

// ------------------------------------------------------------------
// Optimization 1: sampling + lookup split
// ------------------------------------------------------------------
unsigned long long run_event_based_simulation_optimization_1(const Inputs &in,
                                                            SimulationData &GSD,
                                                            queue &q) {
    GSD.p_energy_samples = malloc_device<double>(in.lookups, q);
    GSD.mat_samples = malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);
            GSD.p_energy_samples[i] = p_energy;
            GSD.mat_samples[i] = mat;
        });
    }).wait();

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i],
                               GSD.mat_samples[i],
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array,
                               GSD.index_grid, GSD.nuclide_grid,
                               GSD.mats, macro_xs,
                               in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    unsigned long long checksum = std::accumulate(host_verif.begin(),
                                                  host_verif.end(),
                                                  0ULL);
    return checksum;
}

// ------------------------------------------------------------------
// Optimization 2: material‑specific kernels
// ------------------------------------------------------------------
unsigned long long run_event_based_simulation_optimization_2(const Inputs &in,
                                                            SimulationData &GSD,
                                                            queue &q) {
    GSD.p_energy_samples = malloc_device<double>(in.lookups, q);
    GSD.mat_samples = malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    for (int m = 0; m < 12; ++m) {
        q.submit([&, m](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                       range<1>(nthreads)),
                           [=](nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= in.lookups) return;
                if (GSD.mat_samples[i] != m) return;
                double macro_xs[5] = {0};
                calculate_macro_xs(GSD.p_energy_samples[i],
                                   m,
                                   in.n_isotopes, in.n_gridpoints,
                                   GSD.num_nucs, GSD.concs,
                                   GSD.unionized_energy_array,
                                   GSD.index_grid, GSD.nuclide_grid,
                                   GSD.mats, macro_xs,
                                   in.grid_type, in.hash_bins,
                                   GSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j)
                    if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
                GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
            });
        }).wait();
    }

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 3: fuel vs non‑fuel kernels
// ------------------------------------------------------------------
unsigned long long run_event_based_simulation_optimization_3(const Inputs &in,
                                                            SimulationData &GSD,
                                                            queue &q) {
    GSD.p_energy_samples = malloc_device<double>(in.lookups, q);
    GSD.mat_samples = malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            if (GSD.mat_samples[i] != 0) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i],
                               0,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array,
                               GSD.index_grid, GSD.nuclide_grid,
                               GSD.mats, macro_xs,
                               in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            int mat = GSD.mat_samples[i];
            if (mat == 0) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i],
                               mat,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array,
                               GSD.index_grid, GSD.nuclide_grid,
                               GSD.mats, macro_xs,
                               in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 4: full material sort + per‑material kernels
// ------------------------------------------------------------------
unsigned long long run_event_based_simulation_optimization_4(const Inputs &in,
                                                            SimulationData &GSD,
                                                            queue &q) {
    GSD.p_energy_samples = malloc_device<double>(in.lookups, q);
    GSD.mat_samples = malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // Copy to host for sorting/counting
    std::vector<int> h_mat(in.lookups);
    std::vector<double> h_energy(in.lookups);
    q.memcpy(h_mat.data(), GSD.mat_samples, in.lookups * sizeof(int)).wait();
    q.memcpy(h_energy.data(), GSD.p_energy_samples, in.lookups * sizeof(double)).wait();

    int lookups_per_mat[12] = {0};
    for (int m = 0; m < 12; ++m)
        lookups_per_mat[m] = std::count(h_mat.begin(), h_mat.end(), m);

    // sort by material (stable)
    std::vector<std::pair<int,double>> pairs(in.lookups);
    for (int i = 0; i < in.lookups; ++i) pairs[i] = {h_mat[i], h_energy[i]};
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const auto &a, const auto &b){ return a.first < b.first; });
    for (int i = 0; i < in.lookups; ++i) {
        h_mat[i] = pairs[i].first;
        h_energy[i] = pairs[i].second;
    }

    // copy sorted data back to device
    q.memcpy(GSD.mat_samples, h_mat.data(), in.lookups * sizeof(int)).wait();
    q.memcpy(GSD.p_energy_samples, h_energy.data(), in.lookups * sizeof(double)).wait();

    int offset = 0;
    for (int m = 0; m < 12; ++m) {
        int nlook = lookups_per_mat[m];
        if (nlook == 0) continue;
        int nb = (nlook + nthreads - 1) / nthreads;
        q.submit([&, m, nlook, offset](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(nb * nthreads),
                                       range<1>(nthreads)),
                           [=](nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= nlook) return;
                int idx = i + offset;
                double macro_xs[5] = {0};
                calculate_macro_xs(GSD.p_energy_samples[idx],
                                   GSD.mat_samples[idx],
                                   in.n_isotopes, in.n_gridpoints,
                                   GSD.num_nucs, GSD.concs,
                                   GSD.unionized_energy_array,
                                   GSD.index_grid, GSD.nuclide_grid,
                                   GSD.mats, macro_xs,
                                   in.grid_type, in.hash_bins,
                                   GSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j)
                    if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
                GSD.verification[idx] = static_cast<unsigned long>(max_idx + 1);
            });
        }).wait();
        offset += nlook;
    }

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 5: fuel / non‑fuel partition
// ------------------------------------------------------------------
struct is_fuel {
    bool operator()(const int &v) const { return v == 0; }
};
unsigned long long run_event_based_simulation_optimization_5(const Inputs &in,
                                                            SimulationData &GSD,
                                                            queue &q) {
    GSD.p_energy_samples = malloc_device<double>(in.lookups, q);
    GSD.mat_samples = malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // Copy to host for partitioning
    std::vector<int> h_mat(in.lookups);
    std::vector<double> h_energy(in.lookups);
    q.memcpy(h_mat.data(), GSD.mat_samples, in.lookups * sizeof(int)).wait();
    q.memcpy(h_energy.data(), GSD.p_energy_samples, in.lookups * sizeof(double)).wait();

    int n_fuel = std::count(h_mat.begin(), h_mat.end(), 0);

    // stable partition fuel to front
    std::stable_partition(h_mat.begin(), h_mat.end(),
                          [&](int v){ return v == 0; });
    std::stable_partition(h_energy.begin(), h_energy.end(),
                          [&](int idx){ return h_mat[idx] == 0; });

    // copy partitioned data back
    q.memcpy(GSD.mat_samples, h_mat.data(), in.lookups * sizeof(int)).wait();
    q.memcpy(GSD.p_energy_samples, h_energy.data(), in.lookups * sizeof(double)).wait();

    int nb_fuel = (n_fuel + nthreads - 1) / nthreads;
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nb_fuel * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= n_fuel) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[i],
                               0,
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array,
                               GSD.index_grid, GSD.nuclide_grid,
                               GSD.mats, macro_xs,
                               in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[i] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    int n_nonfuel = in.lookups - n_fuel;
    int nb_nonfuel = (n_nonfuel + nthreads - 1) / nthreads;
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nb_nonfuel * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= n_nonfuel) return;
            int idx = i + n_fuel;
            double macro_xs[5] = {0};
            calculate_macro_xs(GSD.p_energy_samples[idx],
                               GSD.mat_samples[idx],
                               in.n_isotopes, in.n_gridpoints,
                               GSD.num_nucs, GSD.concs,
                               GSD.unionized_energy_array,
                               GSD.index_grid, GSD.nuclide_grid,
                               GSD.mats, macro_xs,
                               in.grid_type, in.hash_bins,
                               GSD.max_num_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            GSD.verification[idx] = static_cast<unsigned long>(max_idx + 1);
        });
    }).wait();

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Optimization 6: material + energy sorts
// ------------------------------------------------------------------
unsigned long long run_event_based_simulation_optimization_6(const Inputs &in,
                                                            SimulationData &GSD,
                                                            queue &q) {
    GSD.p_energy_samples = malloc_device<double>(in.lookups, q);
    GSD.mat_samples = malloc_device<int>(in.lookups, q);
    const int nthreads = 32;
    const int nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                   range<1>(nthreads)),
                       [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            GSD.p_energy_samples[i] = LCG_random_double(&seed);
            GSD.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();

    // Host copies
    std::vector<int> h_mat(in.lookups);
    std::vector<double> h_energy(in.lookups);
    q.memcpy(h_mat.data(), GSD.mat_samples, in.lookups * sizeof(int)).wait();
    q.memcpy(h_energy.data(), GSD.p_energy_samples, in.lookups * sizeof(double)).wait();

    int lookups_per_mat[12] = {0};
    for (int m = 0; m < 12; ++m)
        lookups_per_mat[m] = std::count(h_mat.begin(), h_mat.end(), m);

    // sort by material
    std::vector<std::pair<int,double>> pairs(in.lookups);
    for (int i = 0; i < in.lookups; ++i) pairs[i] = {h_mat[i], h_energy[i]};
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const auto &a, const auto &b){ return a.first < b.first; });
    for (int i = 0; i < in.lookups; ++i) {
        h_mat[i] = pairs[i].first;
        h_energy[i] = pairs[i].second;
    }

    // sort each material by energy
    int offset = 0;
    for (int m = 0; m < 12; ++m) {
        int cnt = lookups_per_mat[m];
        if (cnt == 0) continue;
        std::sort(h_energy.begin() + offset,
                  h_energy.begin() + offset + cnt);
        offset += cnt;
    }

    // copy back
    q.memcpy(GSD.mat_samples, h_mat.data(), in.lookups * sizeof(int)).wait();
    q.memcpy(GSD.p_energy_samples, h_energy.data(), in.lookups * sizeof(double)).wait();

    // launch per‑material kernels (reuse logic from opt‑4)
    offset = 0;
    for (int m = 0; m < 12; ++m) {
        int nlook = lookups_per_mat[m];
        if (nlook == 0) continue;
        int nb = (nlook + nthreads - 1) / nthreads;
        q.submit([&, m, nlook, offset](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(nb * nthreads),
                                       range<1>(nthreads)),
                           [=](nd_item<1> it) {
                int i = it.get_global_id(0);
                if (i >= nlook) return;
                int idx = i + offset;
                double macro_xs[5] = {0};
                calculate_macro_xs(GSD.p_energy_samples[idx],
                                   GSD.mat_samples[idx],
                                   in.n_isotopes, in.n_gridpoints,
                                   GSD.num_nucs, GSD.concs,
                                   GSD.unionized_energy_array,
                                   GSD.index_grid, GSD.nuclide_grid,
                                   GSD.mats, macro_xs,
                                   in.grid_type, in.hash_bins,
                                   GSD.max_num_nucs);
                double max = -1.0;
                int max_idx = 0;
                for (int j = 0; j < 5; ++j)
                    if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
                GSD.verification[idx] = static_cast<unsigned long>(max_idx + 1);
            });
        }).wait();
        offset += nlook;
    }

    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), GSD.verification,
             in.lookups * sizeof(unsigned long)).wait();
    return std::accumulate(host_verif.begin(), host_verif.end(), 0ULL);
}

// ------------------------------------------------------------------
// Placeholder implementations for the remaining helper functions
// ------------------------------------------------------------------
int *load_num_nucs(long n_isotopes) {
    int *num_nucs = (int *)malloc(12 * sizeof(int));
    if (n_isotopes == 68) num_nucs[0] = 34;
    else num_nucs[0] = 321;
    num_nucs[1] = 5;  num_nucs[2] = 4;  num_nucs[3] = 4;
    num_nucs[4] = 27; num_nucs[5] = 21; num_nucs[6] = 21;
    num_nucs[7] = 21; num_nucs[8] = 21; num_nucs[9] = 21;
    num_nucs[10] = 9; num_nucs[11] = 9;
    return num_nucs;
}
int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs) {
    *max_num_nucs = 0;
    for (int m = 0; m < 12; ++m)
        *max_num_nucs = std::max(*max_num_nucs, num_nucs[m]);
    int *mats = (int *)malloc(12 * (*max_num_nucs) * sizeof(int));
    memset(mats, 0, 12 * (*max_num_nucs) * sizeof(int));
    return mats;
}
double *load_concs(int *num_nucs, int max_num_nucs) {
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double *concs = (double *)malloc(12 * max_num_nucs * sizeof(double));
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < num_nucs[i]; ++j)
            concs[i * max_num_nucs + j] = LCG_random_double(&seed);
    return concs;
}
int NGP_compare(const void *a, const void *b) {
    const NuclideGridPoint *A = (const NuclideGridPoint *)a;
    const NuclideGridPoint *B = (const NuclideGridPoint *)b;
    return (A->energy > B->energy) - (A->energy < B->energy);
}
int double_compare(const void *a, const void *b) {
    double A = *(const double *)a;
    double B = *(const double *)b;
    return (A > B) - (A < B);
}
SimulationData grid_init_do_not_profile(const Inputs &in, int mype) {
    SimulationData SD{};
    uint64_t seed = 42;
    SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
    SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid *
                                                sizeof(NuclideGridPoint));
    assert(SD.nuclide_grid != nullptr);
    for (int i = 0; i < SD.length_nuclide_grid; ++i) {
        SD.nuclide_grid[i].energy = LCG_random_double(&seed);
        SD.nuclide_grid[i].total_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].elastic_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].fission_xs = LCG_random_double(&seed);
        SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
    }
    for (int i = 0; i < in.n_isotopes; i++)
        qsort(&SD.nuclide_grid[i*in.n_gridpoints], in.n_gridpoints, sizeof(NuclideGridPoint), NGP_compare);
    if (in.grid_type == UNIONIZED) {
        SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
        SD.unionized_energy_array = (double *)malloc(
            SD.length_unionized_energy_array * sizeof(double));
        assert(SD.unionized_energy_array != nullptr);
        for (int i = 0; i < SD.length_unionized_energy_array; ++i)
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        qsort(SD.unionized_energy_array,
              SD.length_unionized_energy_array, sizeof(double), double_compare);
        SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
        SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
        assert(SD.index_grid != nullptr);
        int *idx_low = (int *)calloc(in.n_isotopes, sizeof(int));
        double *energy_high = (double *)malloc(in.n_isotopes * sizeof(double));
        for (int i = 0; i < in.n_isotopes; i++)
            energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;
        for (long e = 0; e < SD.length_unionized_energy_array; e++) {
            double ue = SD.unionized_energy_array[e];
            for (long i = 0; i < in.n_isotopes; i++) {
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
    }
    if (in.grid_type == HASH) {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid  = in.hash_bins * in.n_isotopes;
        SD.index_grid = (int *)malloc( SD.length_index_grid * sizeof(int));
        double du = 1.0 / in.hash_bins;
        for (long e = 0; e < in.hash_bins; e++) {
            double energy = e * du;
            for (long i = 0; i < in.n_isotopes; i++) {
                SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide(
                    in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints-1);
            }
        }
    }
    SD.num_nucs  = load_num_nucs(in.n_isotopes);
    SD.length_num_nucs = 12;
    SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
    SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
    SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
    SD.length_concs = SD.length_mats;
    return SD;
}
SimulationData binary_read(const Inputs &in) { return {}; }
void binary_write(const Inputs &in, const SimulationData &SD) {}
int print_results(const Inputs &in, int mype, double runtime,
                  int nprocs, unsigned long long vhash) {
    int lookups = (in.simulation_method == EVENT_BASED) ? in.lookups
                                                       : in.lookups * in.particles;
    int lookups_per_sec = static_cast<int>(static_cast<double>(lookups) / runtime);
    int is_invalid_result = 1;
    if (mype == 0) {
        printf("Runtime: %.3lf seconds\n", runtime);
        printf("Lookups: "); fancy_int(lookups);
        printf("Lookups/s: "); fancy_int(lookups_per_sec);
        unsigned long long large = 952131, small = 945990;
        if (strcmp(in.HM, "large") == 0) {
            if (vhash == large) is_invalid_result = 0;
        } else if (strcmp(in.HM, "small") == 0) {
            if (vhash == small) is_invalid_result = 0;
        }
        if (is_invalid_result)
            printf("Verification checksum: %llu (WARNING - INVALID CHECKSUM!)\n", vhash);
        else
            printf("Verification checksum: %llu (Valid)\n", vhash);
    }
    return is_invalid_result;
}
void print_inputs(const Inputs &in, int nprocs, int version) {}
void logo(int version) {}
void border_print(void) {}
void center_print(const char *s, int width) {}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
int main(int argc, char *argv[]) {
    try {
        queue q{gpu_selector{}, property::queue::in_order{}};

        Inputs in{};
        in.simulation_method = EVENT_BASED;
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

        if (in.simulation_method == EVENT_BASED) {
            in.lookups = in.lookups * in.particles;
            in.particles = 0;
        }

        SimulationData SD;
        if (in.binary_mode == READ) SD = binary_read(in);
        else SD = grid_init_do_not_profile(in, 0);
        if (in.binary_mode == WRITE) binary_write(in, SD);

        SimulationData GSD = move_simulation_data_to_device(in, SD, q);

        double start = get_time();
        unsigned long long verification = 0;
        if (in.simulation_method == EVENT_BASED) {
            switch (in.kernel_id) {
                case 0: verification = run_event_based_simulation_baseline(in, GSD, q); break;
                case 1: verification = run_event_based_simulation_optimization_1(in, GSD, q); break;
                case 2: verification = run_event_based_simulation_optimization_2(in, GSD, q); break;
                case 3: verification = run_event_based_simulation_optimization_3(in, GSD, q); break;
                case 4: verification = run_event_based_simulation_optimization_4(in, GSD, q); break;
                case 5: verification = run_event_based_simulation_optimization_5(in, GSD, q); break;
                case 6: verification = run_event_based_simulation_optimization_6(in, GSD, q); break;
                default:
                    printf("Error: No kernel ID %d found!\n", in.kernel_id);
                    return 1;
            }
        } else {
            printf("History‑based simulation not implemented.\n");
            return 1;
        }
        double end = get_time();

        release_device_memory(GSD, q);
        verification %= 999983;

        print_results(in, 0, end - start, 1, verification);
        return 0;
    } catch (exception const &e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }
}
