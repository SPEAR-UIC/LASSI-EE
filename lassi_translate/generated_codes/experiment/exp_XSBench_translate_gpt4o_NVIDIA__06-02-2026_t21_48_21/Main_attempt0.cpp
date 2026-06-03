#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>

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

// Global SYCL queue
sycl::queue q{sycl::gpu_selector_v};

// Function prototypes
Inputs read_CLI(int argc, char *argv[]);
SimulationData grid_init_do_not_profile(Inputs in, int mype);
SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD);
void release_device_memory(SimulationData GSD);
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype);
void xs_lookup_kernel_baseline(sycl::queue &q, Inputs in, SimulationData GSD);
void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, long idx, double *xs_vector, int grid_type, int hash_bins);
void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, int *num_nucs, double *concs, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs);
long grid_search(long n, double quarry, double *A);
long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high);
int pick_mat(uint64_t *seed);
double LCG_random_double(uint64_t *seed);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);

int main(int argc, char *argv[]) {
    int version = 20;
    int mype = 0;
    double omp_start, omp_end;
    int nprocs = 1;
    unsigned long long verification;

    Inputs in = read_CLI(argc, argv);

    SimulationData SD = grid_init_do_not_profile(in, mype);
    SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

    omp_start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    if (in.simulation_method == EVENT_BASED) {
        verification = run_event_based_simulation_baseline(in, GSD, mype);
    } else {
        printf("History-based simulation not implemented.\n");
        exit(1);
    }

    omp_end = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    release_device_memory(GSD);

    verification = verification % 999983;

    printf("Verification checksum: %llu\n", verification);

    return 0;
}

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD) {
    SimulationData GSD = SD;

    GSD.num_nucs = sycl::malloc_device<int>(GSD.length_num_nucs, q);
    q.memcpy(GSD.num_nucs, SD.num_nucs, GSD.length_num_nucs * sizeof(int)).wait();

    GSD.concs = sycl::malloc_device<double>(GSD.length_concs, q);
    q.memcpy(GSD.concs, SD.concs, GSD.length_concs * sizeof(double)).wait();

    GSD.mats = sycl::malloc_device<int>(GSD.length_mats, q);
    q.memcpy(GSD.mats, SD.mats, GSD.length_mats * sizeof(int)).wait();

    GSD.unionized_energy_array = sycl::malloc_device<double>(GSD.length_unionized_energy_array, q);
    q.memcpy(GSD.unionized_energy_array, SD.unionized_energy_array, GSD.length_unionized_energy_array * sizeof(double)).wait();

    GSD.index_grid = sycl::malloc_device<int>(GSD.length_index_grid, q);
    q.memcpy(GSD.index_grid, SD.index_grid, GSD.length_index_grid * sizeof(int)).wait();

    GSD.nuclide_grid = sycl::malloc_device<NuclideGridPoint>(GSD.length_nuclide_grid, q);
    q.memcpy(GSD.nuclide_grid, SD.nuclide_grid, GSD.length_nuclide_grid * sizeof(NuclideGridPoint)).wait();

    GSD.verification = sycl::malloc_device<unsigned long>(in.lookups, q);

    return GSD;
}

void release_device_memory(SimulationData GSD) {
    sycl::free(GSD.num_nucs, q);
    sycl::free(GSD.concs, q);
    sycl::free(GSD.mats, q);
    sycl::free(GSD.unionized_energy_array, q);
    sycl::free(GSD.index_grid, q);
    sycl::free(GSD.nuclide_grid, q);
    sycl::free(GSD.verification, q);
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype) {
    xs_lookup_kernel_baseline(q, in, GSD);

    unsigned long long verification_scalar = 0;
    unsigned long *verification_host = (unsigned long *)malloc(in.lookups * sizeof(unsigned long));
    q.memcpy(verification_host, GSD.verification, in.lookups * sizeof(unsigned long)).wait();

    for (int i = 0; i < in.lookups; i++) {
        verification_scalar += verification_host[i];
    }

    free(verification_host);
    return verification_scalar;
}

void xs_lookup_kernel_baseline(sycl::queue &q, Inputs in, SimulationData GSD) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(in.lookups), [=](sycl::id<1> id) {
            int i = id[0];
            if (i >= in.lookups) return;

            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2 * i);

            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);

            double macro_xs_vector[5] = {0};

            calculate_macro_xs(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);

            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; j++) {
                if (macro_xs_vector[j] > max) {
                    max = macro_xs_vector[j];
                    max_idx = j;
                }
            }
            GSD.verification[i] = max_idx + 1;
        });
    }).wait();
}

void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, long idx, double *xs_vector, int grid_type, int hash_bins) {
    // Implementation here...
}

void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, int *num_nucs, double *concs, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs) {
    // Implementation here...
}

long grid_search(long n, double quarry, double *A) {
    // Implementation here...
    return 0;
}

long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high) {
    // Implementation here...
    return 0;
}

int pick_mat(uint64_t *seed) {
    // Implementation here...
    return 0;
}

double LCG_random_double(uint64_t *seed) {
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double)(*seed) / (double)m;
}

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
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
