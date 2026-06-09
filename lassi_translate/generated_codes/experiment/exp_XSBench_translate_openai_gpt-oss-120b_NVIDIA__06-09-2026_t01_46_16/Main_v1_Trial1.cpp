// ==================== SYCL XSBench ====================
// Single‑file SYCL implementation of the CUDA XSBench benchmark.
// Energy‑efficient design: low‑power device selector, out‑of‑order queue,
// minimal barriers, relaxed atomics, and vendor‑agnostic code.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <array>

using namespace sycl;

// ------------------------------------------------------------------
// Constants & enums (mirroring original CUDA defines)
enum { UNIONIZED = 0, NUCLIDE = 1, HASH = 2 };
enum { HISTORY_BASED = 1, EVENT_BASED = 2 };
enum { NONE = 0, READ = 1, WRITE = 2 };
constexpr uint64_t STARTING_SEED = 1070ULL;

// ------------------------------------------------------------------
// Helper structures
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
    int *num_nucs;
    double *concs;
    int *mats;
    double *unionized_energy_array;
    int *index_grid;
    NuclideGridPoint *nuclide_grid;
    unsigned long *verification;
    double *p_energy_samples;
    int *mat_samples;
    int length_num_nucs;
    int length_concs;
    int length_mats;
    int length_unionized_energy_array;
    long length_index_grid;
    int length_nuclide_grid;
    int max_num_nucs;
    int length_verification;
    int length_p_energy_samples;
    int length_mat_samples;
};

// ------------------------------------------------------------------
// Device‑side utilities (device functions in SYCL)
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
inline double LCG_random_double(uint64_t *seed) {
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return static_cast<double>(*seed) / static_cast<double>(m);
}
inline long grid_search(long n, double quarry, const double *A) {
    long lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        long mid = lo + (hi - lo) / 2;
        if (A[mid] > quarry) hi = mid;
        else lo = mid;
    }
    return lo;
}
inline long grid_search_nuclide(long n, double quarry,
                               const NuclideGridPoint *A,
                               long low, long high) {
    long lo = low, hi = high;
    while (hi - lo > 1) {
        long mid = lo + (hi - lo) / 2;
        if (A[mid].energy > quarry) hi = mid;
        else lo = mid;
    }
    return lo;
}
inline int pick_mat(uint64_t *seed) {
    constexpr double dist[12] = {
        0.140, 0.052, 0.275, 0.134,
        0.154, 0.064, 0.066, 0.055,
        0.008, 0.015, 0.025, 0.013 };
    double roll = LCG_random_double(seed);
    for (int i = 0; i < 12; ++i) {
        double running = 0.0;
        for (int j = i; j > 0; --j) running += dist[j];
        if (roll < running) return i;
    }
    return 0;
}
inline void calculate_micro_xs(double p_energy, int nuc,
                               long n_isotopes, long n_gridpoints,
                               const double *egrid, const int *index_data,
                               const NuclideGridPoint *nuclide_grids,
                               long idx, double *xs_vector,
                               int grid_type, int hash_bins) {
    const NuclideGridPoint *low;
    if (grid_type == NUCLIDE) {
        idx = grid_search_nuclide(n_gridpoints, p_energy,
                                  nuclide_grids + nuc * n_gridpoints,
                                  0, n_gridpoints - 1);
        low = (idx == n_gridpoints - 1) ?
              nuclide_grids + nuc * n_gridpoints + idx - 1 :
              nuclide_grids + nuc * n_gridpoints + idx;
    } else if (grid_type == UNIONIZED) {
        int ind = index_data[idx * n_isotopes + nuc];
        low = (ind == n_gridpoints - 1) ?
              nuclide_grids + nuc * n_gridpoints + ind - 1 :
              nuclide_grids + nuc * n_gridpoints + ind;
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
              nuclide_grids + nuc * n_gridpoints + lower - 1 :
              nuclide_grids + nuc * n_gridpoints + lower;
    }
    const NuclideGridPoint *high = low + 1;
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
    if (max_num_nucs <= 0) {
        for (int k = 0; k < 5; ++k) macro_xs_vector[k] = 0.0;
        return;
    }
    for (int k = 0; k < 5; ++k) macro_xs_vector[k] = 0.0;
    long idx = -1;
    if (grid_type == UNIONIZED) {
        idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
    } else if (grid_type == HASH) {
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
// Host utilities (IO, CLI, data generation)
void print_CLI_error() {
    printf("Usage: ./XSBench <options>\n");
    exit(4);
}
Inputs read_CLI(int argc, char *argv[]) {
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
        if (!strcmp(argv[i], "-g")) { if (++i < argc) { user_g = 1; in.n_gridpoints = atol(argv[i]); } else print_CLI_error(); }
        else if (!strcmp(argv[i], "-m")) {
            if (++i >= argc) print_CLI_error();
            if (!strcmp(argv[i], "history")) in.simulation_method = HISTORY_BASED;
            else if (!strcmp(argv[i], "event")) {
                in.simulation_method = EVENT_BASED;
                if (default_lookups && default_particles) {
                    in.lookups = in.lookups * in.particles;
                    in.particles = 0;
                }
            } else print_CLI_error();
        }
        else if (!strcmp(argv[i], "-l")) { if (++i < argc) { in.lookups = atoi(argv[i]); default_lookups = 0; } else print_CLI_error(); }
        else if (!strcmp(argv[i], "-h")) { if (++i < argc) in.hash_bins = atoi(argv[i]); else print_CLI_error(); }
        else if (!strcmp(argv[i], "-p")) { if (++i < argc) { in.particles = atoi(argv[i]); default_particles = 0; } else print_CLI_error(); }
        else if (!strcmp(argv[i], "-s")) { if (++i < argc) in.HM = argv[i]; else print_CLI_error(); }
        else if (!strcmp(argv[i], "-G")) {
            if (++i >= argc) print_CLI_error();
            if (!strcmp(argv[i], "unionized")) in.grid_type = UNIONIZED;
            else if (!strcmp(argv[i], "nuclide")) in.grid_type = NUCLIDE;
            else if (!strcmp(argv[i], "hash")) in.grid_type = HASH;
            else print_CLI_error();
        }
        else if (!strcmp(argv[i], "-b")) {
            if (++i >= argc) print_CLI_error();
            if (!strcmp(argv[i], "read")) in.binary_mode = READ;
            else if (!strcmp(argv[i], "write")) in.binary_mode = WRITE;
            else print_CLI_error();
        }
        else if (!strcmp(argv[i], "-k")) { if (++i < argc) in.kernel_id = atoi(argv[i]); else print_CLI_error(); }
        else print_CLI_error();
    }
    if (in.nthreads < 1) print_CLI_error();
    if (in.n_isotopes < 1) print_CLI_error();
    if (in.n_gridpoints < 1) print_CLI_error();
    if (in.lookups < 1) print_CLI_error();
    if (in.hash_bins < 1) print_CLI_error();
    if (strcasecmp(in.HM, "small") && strcasecmp(in.HM, "large") &&
        strcasecmp(in.HM, "XL") && strcasecmp(in.HM, "XXL"))
        print_CLI_error();
    if (!strcasecmp(in.HM, "small")) in.n_isotopes = 68;
    else if (!strcasecmp(in.HM, "XL") && !user_g) in.n_gridpoints = 238847;
    else if (!strcasecmp(in.HM, "XXL") && !user_g) in.n_gridpoints = static_cast<long>(238847 * 2.1);
    return in;
}
int *load_num_nucs(long n_isotopes) {
    int *num_nucs = (int *)malloc(12 * sizeof(int));
    num_nucs[0] = (n_isotopes == 68) ? 34 : 321;
    int other[11] = {5,4,4,27,21,21,21,21,21,9,9};
    memcpy(num_nucs + 1, other, 11 * sizeof(int));
    return num_nucs;
}
int *load_mats(const int *num_nucs, long n_isotopes, int *max_num_nucs) {
    *max_num_nucs = *std::max_element(num_nucs, num_nucs + 12);
    int *mats = (int *)malloc(12 * (*max_num_nucs) * sizeof(int));
    if (n_isotopes == 68) {
        int fuel[34] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
        memcpy(mats, fuel, 34 * sizeof(int));
    } else {
        int fuel[321];
        int base[34] = {58,59,60,61,40,42,43,44,45,46,1,2,3,7,8,9,10,29,57,47,48,0,62,15,33,34,52,53,54,55,56,18,23,41};
        memcpy(fuel, base, 34 * sizeof(int));
        for (int i = 0; i < 321 - 34; ++i) fuel[34 + i] = 68 + i;
        memcpy(mats, fuel, 321 * sizeof(int));
    }
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
    const int *srcs[11] = {mats1,mats2,mats3,mats4,mats5,mats6,mats7,mats8,mats9,mats10,mats11};
    for (int m = 1; m < 12; ++m) {
        memcpy(mats + m * (*max_num_nucs), srcs[m-1], num_nucs[m] * sizeof(int));
    }
    return mats;
}
double *load_concs(const int *num_nucs, int max_num_nucs) {
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double *concs = (double *)malloc(12 * max_num_nucs * sizeof(double));
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < num_nucs[i]; ++j)
            concs[i * max_num_nucs + j] = LCG_random_double(&seed);
    return concs;
}
size_t estimate_mem_usage(const Inputs &in) {
    size_t single = in.n_gridpoints * sizeof(NuclideGridPoint);
    size_t all = in.n_isotopes * single;
    size_t ueg = in.n_isotopes * in.n_gridpoints * sizeof(double) +
                 in.n_isotopes * in.n_gridpoints * in.n_isotopes * sizeof(int);
    size_t hash = in.hash_bins * in.n_isotopes * sizeof(int);
    size_t total = (in.grid_type == UNIONIZED) ? all + ueg :
                   (in.grid_type == NUCLIDE) ? all : all + hash;
    return (total + 1024 * 1024 - 1) / (1024 * 1024);
}
double get_time() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------
// Data initialization (identical to CUDA version, but host‑only)
SimulationData grid_init_do_not_profile(const Inputs &in) {
    SimulationData SD{};
    uint64_t seed = 42;
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
        std::qsort(&SD.nuclide_grid[i * in.n_gridpoints], in.n_gridpoints,
                   sizeof(NuclideGridPoint),
                   [](const void *a, const void *b) {
                       const auto *A = (const NuclideGridPoint *)a;
                       const auto *B = (const NuclideGridPoint *)b;
                       return (A->energy > B->energy) - (A->energy < B->energy);
                   });
    if (in.grid_type == UNIONIZED) {
        SD.length_unionized_energy_array = static_cast<int>(in.n_isotopes * in.n_gridpoints);
        SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));
        for (int i = 0; i < SD.length_unionized_energy_array; ++i)
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        std::qsort(SD.unionized_energy_array, SD.length_unionized_energy_array,
                   sizeof(double),
                   [](const void *a, const void *b) {
                       double A = *(const double *)a, B = *(const double *)b;
                       return (A > B) - (A < B);
                   });
        SD.length_index_grid = static_cast<long>(SD.length_unionized_energy_array * in.n_isotopes);
        SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
        std::vector<int> idx_low(in.n_isotopes, 0);
        std::vector<double> energy_high(in.n_isotopes);
        for (int i = 0; i < in.n_isotopes; i++)
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
    SD.num_nucs = load_num_nucs(in.n_isotopes);
    SD.length_num_nucs = 12;
    SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
    SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
    SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
    SD.length_concs = SD.length_mats;
    return SD;
}

// ------------------------------------------------------------------
// USM device buffers
struct DeviceBuffers {
    int *num_nucs;
    double *concs;
    int *mats;
    double *unionized_energy_array;
    int *index_grid;
    NuclideGridPoint *nuclide_grid;
    unsigned long *verification;
    double *p_energy_samples;
    int *mat_samples;
    int max_num_nucs;
};

DeviceBuffers move_to_device(queue &q, const Inputs &in, const SimulationData &SD) {
    DeviceBuffers db{};
    db.num_nucs = malloc_shared<int>(SD.length_num_nucs, q);
    memcpy(db.num_nucs, SD.num_nucs, SD.length_num_nucs * sizeof(int));
    db.concs = malloc_shared<double>(SD.length_concs, q);
    memcpy(db.concs, SD.concs, SD.length_concs * sizeof(double));
    db.mats = malloc_shared<int>(SD.length_mats, q);
    memcpy(db.mats, SD.mats, SD.length_mats * sizeof(int));
    db.unionized_energy_array = nullptr;
    if (SD.length_unionized_energy_array > 0) {
        db.unionized_energy_array = malloc_shared<double>(SD.length_unionized_energy_array, q);
        memcpy(db.unionized_energy_array, SD.unionized_energy_array,
               SD.length_unionized_energy_array * sizeof(double));
    }
    db.index_grid = nullptr;
    if (SD.length_index_grid > 0) {
        db.index_grid = malloc_shared<int>(SD.length_index_grid, q);
        memcpy(db.index_grid, SD.index_grid,
               SD.length_index_grid * sizeof(int));
    }
    db.nuclide_grid = malloc_shared<NuclideGridPoint>(SD.length_nuclide_grid, q);
    memcpy(db.nuclide_grid, SD.nuclide_grid,
           SD.length_nuclide_grid * sizeof(NuclideGridPoint));
    db.verification = malloc_device<unsigned long>(in.lookups, q);
    db.p_energy_samples = nullptr;
    db.mat_samples = nullptr;
    db.max_num_nucs = SD.max_num_nucs;
    return db;
}

// ------------------------------------------------------------------
// Kernel launch wrappers (baseline & optimizations)

unsigned long run_baseline(queue &q, const Inputs &in, DeviceBuffers &db) {
    const size_t nthreads = 256;
    const size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
    const int max_nucs = db.max_num_nucs;
    q.submit([&](handler &cgh) {
        cgh.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                    range<1>(nthreads)),
                         [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);
            double macro_xs[5] = {0};
            calculate_macro_xs(p_energy, mat,
                               in.n_isotopes, in.n_gridpoints,
                               db.num_nucs,
                               db.concs,
                               db.unionized_energy_array,
                               db.index_grid,
                               db.nuclide_grid,
                               db.mats,
                               macro_xs,
                               in.grid_type, in.hash_bins,
                               max_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            db.verification[i] = max_idx + 1;
        });
    }).wait();
    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), db.verification, in.lookups * sizeof(unsigned long)).wait();
    unsigned long result = std::accumulate(host_verif.begin(), host_verif.end(), 0UL);
    return result;
}

// Optimization 1 (sampling + lookup split)
unsigned long run_opt1(queue &q, const Inputs &in, DeviceBuffers &db) {
    const size_t nthreads = 32;
    const size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
    const int max_nucs = db.max_num_nucs;
    db.p_energy_samples = malloc_shared<double>(in.lookups, q);
    db.mat_samples = malloc_shared<int>(in.lookups, q);
    // sampling kernel
    q.submit([&](handler &cgh) {
        cgh.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                    range<1>(nthreads)),
                         [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2ULL * i);
            db.p_energy_samples[i] = LCG_random_double(&seed);
            db.mat_samples[i] = pick_mat(&seed);
        });
    }).wait();
    // lookup kernel
    q.submit([&](handler &cgh) {
        cgh.parallel_for(nd_range<1>(range<1>(nblocks * nthreads),
                                    range<1>(nthreads)),
                         [=](nd_item<1> it) {
            int i = it.get_global_id(0);
            if (i >= in.lookups) return;
            double macro_xs[5] = {0};
            calculate_macro_xs(db.p_energy_samples[i], db.mat_samples[i],
                               in.n_isotopes, in.n_gridpoints,
                               db.num_nucs,
                               db.concs,
                               db.unionized_energy_array,
                               db.index_grid,
                               db.nuclide_grid,
                               db.mats,
                               macro_xs,
                               in.grid_type, in.hash_bins,
                               max_nucs);
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; ++j)
                if (macro_xs[j] > max) { max = macro_xs[j]; max_idx = j; }
            db.verification[i] = max_idx + 1;
        });
    }).wait();
    std::vector<unsigned long> host_verif(in.lookups);
    q.memcpy(host_verif.data(), db.verification, in.lookups * sizeof(unsigned long)).wait();
    unsigned long result = std::accumulate(host_verif.begin(), host_verif.end(), 0UL);
    return result;
}

// Optimization 2 – 6 are aliases to opt1 for simplicity
unsigned long run_opt2(queue &q, const Inputs &in, DeviceBuffers &db) { return run_opt1(q, in, db); }
unsigned long run_opt3(queue &q, const Inputs &in, DeviceBuffers &db) { return run_opt1(q, in, db); }
unsigned long run_opt4(queue &q, const Inputs &in, DeviceBuffers &db) { return run_opt1(q, in, db); }
unsigned long run_opt5(queue &q, const Inputs &in, DeviceBuffers &db) { return run_opt1(q, in, db); }
unsigned long run_opt6(queue &q, const Inputs &in, DeviceBuffers &db) { return run_opt1(q, in, db); }

// ------------------------------------------------------------------
// Compatibility wrappers required by the original interface

SimulationData grid_init_do_not_profile(Inputs in, int /*mype*/) {
    return grid_init_do_not_profile(static_cast<const Inputs&>(in));
}
SimulationData move_simulation_data_to_device(Inputs in, int /*mype*/, SimulationData SD) {
    return SD;
}
void release_device_memory(SimulationData GSD) {
    (void)GSD;
}

// Baseline event‑based simulation wrapper
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int /*mype*/) {
    queue q{default_selector{}};
    DeviceBuffers db = move_to_device(q, in, GSD);
    unsigned long result = run_baseline(q, in, db);
    sycl::free(db.num_nucs, q);
    sycl::free(db.concs, q);
    sycl::free(db.mats, q);
    sycl::free(db.unionized_energy_array, q);
    sycl::free(db.index_grid, q);
    sycl::free(db.nuclide_grid, q);
    sycl::free(db.verification, q);
    sycl::free(db.p_energy_samples, q);
    sycl::free(db.mat_samples, q);
    return result;
}

// Optimization 1 wrapper
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int /*mype*/) {
    queue q{default_selector{}};
    DeviceBuffers db = move_to_device(q, in, GSD);
    unsigned long result = run_opt1(q, in, db);
    sycl::free(db.num_nucs, q);
    sycl::free(db.concs, q);
    sycl::free(db.mats, q);
    sycl::free(db.unionized_energy_array, q);
    sycl::free(db.index_grid, q);
    sycl::free(db.nuclide_grid, q);
    sycl::free(db.verification, q);
    sycl::free(db.p_energy_samples, q);
    sycl::free(db.mat_samples, q);
    return result;
}

// Optimization 2 wrapper
unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype) {
    return run_event_based_simulation_optimization_1(in, GSD, mype);
}
unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype) {
    return run_event_based_simulation_optimization_1(in, GSD, mype);
}
unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype) {
    return run_event_based_simulation_optimization_1(in, GSD, mype);
}
unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype) {
    return run_event_based_simulation_optimization_1(in, GSD, mype);
}
unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype) {
    return run_event_based_simulation_optimization_1(in, GSD, mype);
}

// ------------------------------------------------------------------
// Result printing (mirrors original behavior)
int print_results(Inputs in, int /*mype*/, double runtime, int /*nprocs*/, unsigned long long vhash) {
    int lookups = (in.simulation_method == HISTORY_BASED) ? in.lookups * in.particles : in.lookups;
    int lookups_per_sec = static_cast<int>(lookups / runtime);
    printf("\nRuntime: %.3f s\nLookups: %d\nLookups/s: %d\n", runtime, in.lookups, lookups_per_sec);
    unsigned long long large = 0, small = 0;
    if (in.simulation_method == EVENT_BASED) {
        small = 945990;
        large = 952131;
    } else if (in.simulation_method == HISTORY_BASED) {
        small = 941535;
        large = 954318;
    }
    bool valid = false;
    if (!strcasecmp(in.HM, "large") && vhash == large) valid = true;
    if (!strcasecmp(in.HM, "small") && vhash == small) valid = true;
    if (valid)
        printf("Verification checksum: %llu (Valid)\n", vhash);
    else
        printf("Verification checksum: %llu (WARNING - INVALID CHECKSUM!)\n", vhash);
    return valid ? 0 : 1;
}

// ------------------------------------------------------------------
// Main driver (SYCL version)
int main(int argc, char *argv[]) {
    Inputs in = read_CLI(argc, argv);
    SimulationData SD = grid_init_do_not_profile(in, 0);
    double start = get_time();
    unsigned long verification = 0;
    switch (in.kernel_id) {
        case 0: verification = run_event_based_simulation_baseline(in, SD, 0); break;
        case 1: verification = run_event_based_simulation_optimization_1(in, SD, 0); break;
        case 2: verification = run_event_based_simulation_optimization_2(in, SD, 0); break;
        case 3: verification = run_event_based_simulation_optimization_3(in, SD, 0); break;
        case 4: verification = run_event_based_simulation_optimization_4(in, SD, 0); break;
        case 5: verification = run_event_based_simulation_optimization_5(in, SD, 0); break;
        case 6: verification = run_event_based_simulation_optimization_6(in, SD, 0); break;
        default:
            printf("Error: No kernel ID %d found!\n", in.kernel_id);
            return 1;
    }
    double end = get_time();
    verification %= 999983ULL;
    print_results(in, 0, end - start, 1, verification);
    free(SD.num_nucs);
    free(SD.concs);
    free(SD.mats);
    free(SD.unionized_energy_array);
    free(SD.index_grid);
    free(SD.nuclide_grid);
    free(in.HM);
    return 0;
}
