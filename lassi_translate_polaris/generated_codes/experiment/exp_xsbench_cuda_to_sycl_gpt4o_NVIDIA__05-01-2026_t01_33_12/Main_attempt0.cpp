#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <chrono>
#include <cstring>

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
SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD, sycl::queue &q);
void release_device_memory(SimulationData GSD, sycl::queue &q);
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype, sycl::queue &q);
void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, int *num_nucs, double *concs, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs, sycl::nd_item<1> item);
long grid_search(long n, double quarry, double *A);
long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high);
double LCG_random_double(uint64_t *seed);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
int pick_mat(uint64_t *seed);

int main(int argc, char *argv[]) {
    int version = 20;
    int mype = 0;
    double omp_start, omp_end;
    int nprocs = 1;
    unsigned long long verification;

    Inputs in = read_CLI(argc, argv);

    SimulationData SD = grid_init_do_not_profile(in, mype);

    SimulationData GSD = move_simulation_data_to_device(in, mype, SD, q);

    omp_start = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1.0e6;

    verification = run_event_based_simulation_baseline(in, GSD, mype, q);

    omp_end = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1.0e6;

    release_device_memory(GSD, q);

    verification = verification % 999983;

    printf("Verification checksum: %llu\n", verification);

    return 0;
}

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD, sycl::queue &q) {
    SimulationData GSD = SD;

    GSD.num_nucs = sycl::malloc_device<int>(SD.length_num_nucs, q);
    q.memcpy(GSD.num_nucs, SD.num_nucs, SD.length_num_nucs * sizeof(int)).wait();

    GSD.concs = sycl::malloc_device<double>(SD.length_concs, q);
    q.memcpy(GSD.concs, SD.concs, SD.length_concs * sizeof(double)).wait();

    GSD.mats = sycl::malloc_device<int>(SD.length_mats, q);
    q.memcpy(GSD.mats, SD.mats, SD.length_mats * sizeof(int)).wait();

    GSD.unionized_energy_array = sycl::malloc_device<double>(SD.length_unionized_energy_array, q);
    q.memcpy(GSD.unionized_energy_array, SD.unionized_energy_array, SD.length_unionized_energy_array * sizeof(double)).wait();

    GSD.index_grid = sycl::malloc_device<int>(SD.length_index_grid, q);
    q.memcpy(GSD.index_grid, SD.index_grid, SD.length_index_grid * sizeof(int)).wait();

    GSD.nuclide_grid = sycl::malloc_device<NuclideGridPoint>(SD.length_nuclide_grid, q);
    q.memcpy(GSD.nuclide_grid, SD.nuclide_grid, SD.length_nuclide_grid * sizeof(NuclideGridPoint)).wait();

    GSD.verification = sycl::malloc_device<unsigned long>(in.lookups, q);

    return GSD;
}

void release_device_memory(SimulationData GSD, sycl::queue &q) {
    sycl::free(GSD.num_nucs, q);
    sycl::free(GSD.concs, q);
    sycl::free(GSD.mats, q);
    sycl::free(GSD.unionized_energy_array, q);
    sycl::free(GSD.index_grid, q);
    sycl::free(GSD.nuclide_grid, q);
    sycl::free(GSD.verification, q);
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype, sycl::queue &q) {
    unsigned long long verification_scalar = 0;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(in.lookups), [=](sycl::id<1> i) {
            if (i >= in.lookups) return;

            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2 * i);

            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);

            double macro_xs_vector[5] = {0};

            calculate_macro_xs(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs, h.get_nd_item<1>());
        });
    }).wait();

    q.memcpy(&verification_scalar, GSD.verification, sizeof(unsigned long long)).wait();

    return verification_scalar;
}

void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, int *num_nucs, double *concs, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs, sycl::nd_item<1> item) {
    // Implementation of calculate_macro_xs logic
}

long grid_search(long n, double quarry, double *A) {
    long lowerLimit = 0;
    long upperLimit = n - 1;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while (length > 1) {
        examinationPoint = lowerLimit + (length / 2);

        if (A[examinationPoint] > quarry)
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
}

long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high) {
    long lowerLimit = low;
    long upperLimit = high;
    long examinationPoint;
    long length = upperLimit - lowerLimit;

    while (length > 1) {
        examinationPoint = lowerLimit + (length / 2);

        if (A[examinationPoint].energy > quarry)
            upperLimit = examinationPoint;
        else
            lowerLimit = examinationPoint;

        length = upperLimit - lowerLimit;
    }

    return lowerLimit;
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

int pick_mat(uint64_t *seed) {
    double dist[12] = {0.140, 0.052, 0.275, 0.134, 0.154, 0.064, 0.066, 0.055, 0.008, 0.015, 0.025, 0.013};
    double roll = LCG_random_double(seed);

    for (int i = 0; i < 12; i++) {
        double running = 0;
        for (int j = i; j > 0; j--)
            running += dist[j];
        if (roll < running)
            return i;
    }

    return 0;
}
