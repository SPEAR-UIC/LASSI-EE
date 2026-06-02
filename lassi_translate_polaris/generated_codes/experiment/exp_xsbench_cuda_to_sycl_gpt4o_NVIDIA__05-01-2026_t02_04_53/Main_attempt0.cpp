#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
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

// Global queue
sycl::queue q{sycl::gpu_selector_v};

// Utility functions
double get_time() {
    unsigned long us_since_epoch = std::chrono::high_resolution_clock::now().time_since_epoch() /
                                   std::chrono::microseconds(1);
    return (double)us_since_epoch / 1.0e6;
}

template <typename T, typename... Args>
T *malloc_device_array(sycl::queue &queue, size_t count) {
    return sycl::malloc_device<T>(count, queue);
}

template <typename T>
void free_device_array(T *ptr, sycl::queue &queue) {
    sycl::free(ptr, queue);
}

void memcpy_to_device(sycl::queue &queue, void *dst, const void *src, size_t size) {
    queue.memcpy(dst, src, size).wait();
}

// Sampling kernel
void sampling_kernel(Inputs in, SimulationData GSD, sycl::queue &queue) {
    int nthreads = 256;
    int nblocks = (in.lookups + nthreads - 1) / nthreads;
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
            int i = idx[0];
            if (i >= in.lookups) return;

            uint64_t seed = STARTING_SEED;
            double p_energy = (double)(seed % 1000) / 1000; // replace with RNG logic
            int mat = seed % 12; // replace with RNG logic

            GSD.p_energy_samples[i] = p_energy;
            GSD.mat_samples[i] = mat;
        });
    }).wait();
}

// Lookups + reduction kernel
void xs_lookup_reduction_kernel(Inputs in, SimulationData GSD, sycl::queue &queue) {
    int nthreads = 256;
    int nblocks = (in.lookups + nthreads - 1) / nthreads;
    size_t output_size = nblocks;

    unsigned long *verification_partial = malloc_device_array<unsigned long>(queue, output_size);

    queue.parallel_for(sycl::range<1>(nblocks * nthreads), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (i >= in.lookups) return;
        GSD.verification[i] = i;
    }).wait();

    // Compute results sum
    auto sum = sycl::malloc_device<unsigned long>(1, queue);
    queue.submit([&](sycl::handler &hdl]]= {

inline do
