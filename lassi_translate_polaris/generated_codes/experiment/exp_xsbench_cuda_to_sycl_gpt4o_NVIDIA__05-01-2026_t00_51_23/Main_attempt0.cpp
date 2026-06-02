#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>

#define UNIONIZED 0
#define NUCLIDE 1
#define HASH 2

#define HISTORY_BASED 1
#define EVENT_BASED 2

#define NONE 0
#define READ 1
#define WRITE 2

#define STARTING_SEED 1070

typedef struct {
    double energy;
    double total_xs;
    double elastic_xs;
    double absorbtion_xs;
    double fission_xs;
    double nu_fission_xs;
} NuclideGridPoint;

typedef struct {
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

typedef struct {
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

sycl::queue q{sycl::gpu_selector_v};

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype);
void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, long idx, double *xs_vector, int grid_type, int hash_bins, sycl::queue &q);
void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, int *num_nucs, double *concs, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids, int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs, sycl::queue &q);
long grid_search(long n, double quarry, double *A, sycl::queue &q);
long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high, sycl::queue &q);
int pick_mat(uint64_t *seed, sycl::queue &q);
double LCG_random_double(uint64_t *seed, sycl::queue &q);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n, sycl::queue &q);

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD, sycl::queue &q);
void release_device_memory(SimulationData GSD, sycl::queue &q);

int main(int argc, char *argv[]) {
    // Initialization
    int version = 20;
    int mype = 0;
    double omp_start, omp_end;
    int nprocs = 1;
    unsigned long long verification;

    Inputs in;
    // Populate `in` with default values or parsed CLI arguments (not shown here for brevity)

    SimulationData SD;
    SimulationData GSD = move_simulation_data_to_device(in, mype, SD, q);

    omp_start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    verification = run_event_based_simulation_baseline(in, GSD, mype);

    omp_end = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    release_device_memory(GSD, q);

    verification = verification % 999983;

    printf("Verification checksum: %llu\n", verification);

    return 0;
}

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD, sycl::queue &q) {
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

void release_device_memory(SimulationData GSD, sycl::queue &q) {
    sycl::free(GSD.num_nucs, q);
    sycl::free(GSD.concs, q);
    sycl::free(GSD.mats, q);
    sycl::free(GSD.unionized_energy_array, q);
    sycl::free(GSD.index_grid, q);
    sycl::free(GSD.nuclide_grid, q);
    sycl::free(GSD.verification, q);
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype) {
    size_t nthreads = 256;
    size_t nblocks = (in.lookups + nthreads - 1) / nthreads;

    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> id) {
            int i = id[0];
            if (i >= in.lookups) return;

            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2 * i, q);

            double p_energy = LCG_random_double(&seed, q);
            int mat = pick_mat(&seed, q);

            double macro_xs_vector[5] = {0};

            calculate_macro_xs(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs, q);

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

    unsigned long long verification_scalar = 0;
    q.memcpy(&verification_scalar, GSD.verification, sizeof(unsigned long long)).wait();

    return verification_scalar;
}
