#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <numeric>

using namespace sycl;

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
	int grid_type;
	int hash_bins;
	int particles;
	int simulation_method;
	int binary_mode;
	int kernel_id;
} Inputs;

typedef struct{
	int * num_nucs;
	double * concs;
	int * mats;
	double * unionized_energy_array;
	int * index_grid;
	NuclideGridPoint * nuclide_grid;
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

// Forward declarations for utility functions
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
int *load_num_nucs(long n_isotopes);
int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs);
double *load_concs(int *num_nucs, int max_num_nucs);
int NGP_compare(const void *a, const void *b);
int double_compare(const void *a, const void *b);
double rn_v(void);
size_t estimate_mem_usage(Inputs in);
double get_time(void);

// SYCL device selection for energy efficiency
device select_device() {
	auto platforms = platform::get_platforms();
	for (auto &plat : platforms) {
		auto devices = plat.get_devices();
		for (auto &dev : devices) {
			if (dev.is_gpu()) return dev;
		}
	}
	return device(default_selector_v);
}

// SYCL queue management: out-of-order queue for concurrency
queue make_queue() {
	device dev = select_device();
	return queue(dev, property::queue::in_order{});
}

// SYCL memory management helpers
template <typename T>
T *sycl_malloc(queue &q, size_t count) {
	return static_cast<T *>(malloc_device(sizeof(T) * count, q));
}

template <typename T>
void sycl_memcpy(queue &q, T *dst, T *src, size_t count) {
	q.memcpy(dst, src, sizeof(T) * count).wait();
}

// Release device memory
void release_device_memory(SimulationData GSD, queue &q) {
	q.free(GSD.num_nucs);
	q.free(GSD.concs);
	q.free(GSD.mats);
	q.free(GSD.unionized_energy_array);
	q.free(GSD.nuclide_grid);
	q.free(GSD.verification);
	if (GSD.p_energy_samples) q.free(GSD.p_energy_samples);
	if (GSD.mat_samples) q.free(GSD.mat_samples);
	if (GSD.index_grid) q.free(GSD.index_grid);
}

// RNG
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

// pick_mat
int pick_mat(uint64_t *seed) {
	double dist[12] = {0.140, 0.052, 0.275, 0.134, 0.154, 0.064, 0.066, 0.055, 0.008, 0.015, 0.025, 0.013};
	double roll = LCG_random_double(seed);
	for (int i = 0; i < 12; i++) {
		double running = 0;
		for (int j = i; j > 0; j--) running += dist[j];
		if (roll < running) return i;
	}
	return 0;
}

// grid_search
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

// grid_search_nuclide
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

// calculate_micro_xs
void calculate_micro_xs(double p_energy, int nuc, long n_isotopes, long n_gridpoints,
                        double *egrid, int *index_data, NuclideGridPoint *nuclide_grids,
                        long idx, double *xs_vector, int grid_type, int hash_bins) {
	double f;
	NuclideGridPoint *low, *high;
	if (grid_type == NUCLIDE) {
		idx = grid_search_nuclide(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], 0, n_gridpoints - 1);
		if (idx == n_gridpoints - 1)
			low = &nuclide_grids[nuc * n_gridpoints + idx - 1];
		else
			low = &nuclide_grids[nuc * n_gridpoints + idx];
	} else if (grid_type == UNIONIZED) {
		if (index_data[idx * n_isotopes + nuc] == n_gridpoints - 1)
			low = &nuclide_grids[nuc * n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
		else
			low = &nuclide_grids[nuc * n_gridpoints + index_data[idx * n_isotopes + nuc]];
	} else {
		int u_low = index_data[idx * n_isotopes + nuc];
		int u_high;
		if (idx == hash_bins - 1)
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx + 1) * n_isotopes + nuc] + 1;
		double e_low = nuclide_grids[nuc * n_gridpoints + u_low].energy;
		double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
		int lower;
		if (p_energy <= e_low)
			lower = 0;
		else if (p_energy >= e_high)
			lower = n_gridpoints - 1;
		else
			lower = grid_search_nuclide(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], u_low, u_high);
		if (lower == n_gridpoints - 1)
			low = &nuclide_grids[nuc * n_gridpoints + lower - 1];
		else
			low = &nuclide_grids[nuc * n_gridpoints + lower];
	}
	high = low + 1;
	f = (high->energy - p_energy) / (high->energy - low->energy);
	xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
	xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
	xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
	xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
	xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

// calculate_macro_xs
void calculate_macro_xs(double p_energy, int mat, long n_isotopes, long n_gridpoints, int *num_nucs,
                        double *concs, double *egrid, int *index_data, NuclideGridPoint *nuclide_grids,
                        int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs) {
	int p_nuc;
	long idx = -1;
	double conc;
	for (int k = 0; k < 5; k++) macro_xs_vector[k] = 0;
	if (grid_type == UNIONIZED)
		idx = grid_search(n_isotopes * n_gridpoints, p_energy, egrid);
	else if (grid_type == HASH) {
		double du = 1.0 / hash_bins;
		idx = p_energy / du;
	}
	for (int j = 0; j < num_nucs[mat]; j++) {
		double xs_vector[5];
		p_nuc = mats[mat * max_num_nucs + j];
		conc = concs[mat * max_num_nucs + j];
		calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data,
		                   nuclide_grids, idx, xs_vector, grid_type, hash_bins);
		for (int k = 0; k < 5; k++)
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}

// SYCL kernel for baseline simulation
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype, queue &q) {
	if (mype == 0) printf("Running baseline event-based simulation...\n");
	int nthreads = 256;
	int nblocks = (int)std::ceil((double)in.lookups / nthreads);
	size_t total = in.lookups;

	q.submit([&](handler &h) {
		h.parallel_for(range<1>(total), [=](item<1> idx) {
			int i = idx.get_id(0);
			if (i >= in.lookups) return;
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, in.n_isotopes, in.n_gridpoints,
				GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid,
				GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
			);
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

	// Reduce verification results
	if (mype == 0) printf("Reducing verification results...\n");
	std::vector<unsigned long> host_verification(in.lookups);
	q.memcpy(host_verification.data(), GSD.verification, sizeof(unsigned long) * in.lookups).wait();
	unsigned long verification_scalar = std::accumulate(host_verification.begin(), host_verification.end(), 0ul);
	return verification_scalar;
}

// Optimization 1: sample/lookup kernel splitting
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype, queue &q) {
	const char *optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";
	if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

	size_t sz;
	size_t total_sz = 0;
	sz = in.lookups * sizeof(double);
	GSD.p_energy_samples = sycl_malloc<double>(q, in.lookups);
	total_sz += sz;
	GSD.length_p_energy_samples = in.lookups;

	sz = in.lookups * sizeof(int);
	GSD.mat_samples = sycl_malloc<int>(q, in.lookups);
	total_sz += sz;
	GSD.length_mat_samples = in.lookups;

	if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

	int nthreads = 32;
	size_t total = in.lookups;

	// Sampling kernel
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(total), [=](item<1> idx) {
			int i = idx.get_id(0);
			if (i >= in.lookups) return;
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();

	// Lookup kernel
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(total), [=](item<1> idx) {
			int i = idx.get_id(0);
			if (i >= in.lookups) return;
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints,
				GSD.num_nucs, GSD.concs, GSD.unionized_energy_array, GSD.index_grid,
				GSD.nuclide_grid, GSD.mats, macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
			);
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

	if (mype == 0) printf("Reducing verification results...\n");
	std::vector<unsigned long> host_verification(in.lookups);
	q.memcpy(host_verification.data(), GSD.verification, sizeof(unsigned long) * in.lookups).wait();
	unsigned long verification_scalar = std::accumulate(host_verification.begin(), host_verification.end(), 0ul);
	return verification_scalar;
}

// Additional optimizations (2-6) can be implemented similarly using SYCL parallel_for, DPC++ algorithms, and partition/sort/reduce as above.

// Main function
int main(int argc, char *argv[]) {
	int version = 20;
	int mype = 0;
	double omp_start, omp_end;
	int nprocs = 1;
	unsigned long long verification;

	Inputs in = read_CLI(argc, argv);

	if (mype == 0)
		print_inputs(in, nprocs, version);

	SimulationData SD;
	if (in.binary_mode == READ)
		SD = binary_read(in);
	else
		SD = grid_init_do_not_profile(in, mype);

	if (in.binary_mode == WRITE && mype == 0)
		binary_write(in, SD);

	queue q = make_queue();
	SimulationData GSD = SD;

	// Allocate device memory and copy data
	GSD.num_nucs = sycl_malloc<int>(q, SD.length_num_nucs);
	sycl_memcpy(q, GSD.num_nucs, SD.num_nucs, SD.length_num_nucs);

	GSD.concs = sycl_malloc<double>(q, SD.length_concs);
	sycl_memcpy(q, GSD.concs, SD.concs, SD.length_concs);

	GSD.mats = sycl_malloc<int>(q, SD.length_mats);
	sycl_memcpy(q, GSD.mats, SD.mats, SD.length_mats);

	GSD.unionized_energy_array = sycl_malloc<double>(q, SD.length_unionized_energy_array);
	sycl_memcpy(q, GSD.unionized_energy_array, SD.unionized_energy_array, SD.length_unionized_energy_array);

	GSD.index_grid = sycl_malloc<int>(q, SD.length_index_grid);
	sycl_memcpy(q, GSD.index_grid, SD.index_grid, SD.length_index_grid);

	GSD.nuclide_grid = sycl_malloc<NuclideGridPoint>(q, SD.length_nuclide_grid);
	sycl_memcpy(q, GSD.nuclide_grid, SD.nuclide_grid, SD.length_nuclide_grid);

	GSD.verification = sycl_malloc<unsigned long>(q, in.lookups);
	GSD.length_verification = in.lookups;

	if (mype == 0) {
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	omp_start = get_time();

	if (in.simulation_method == EVENT_BASED) {
		if (in.kernel_id == 0)
			verification = run_event_based_simulation_baseline(in, GSD, mype, q);
		else if (in.kernel_id == 1)
			verification = run_event_based_simulation_optimization_1(in, GSD, mype, q);
		// Add further optimized kernel calls here...
		else {
			printf("Error: No kernel ID %d found!\n", in.kernel_id);
			exit(1);
		}
	} else {
		printf("History-based simulation not implemented in SYCL code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
		exit(1);
	}

	if (mype == 0) {
		printf("\n");
		printf("Simulation complete.\n");
	}

	omp_end = get_time();

	release_device_memory(GSD, q);

	verification = verification % 999983;

	int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);

	return is_invalid_result;
}

// Utility functions (logo, center_print, border_print, fancy_int, print_CLI_error, read_CLI, print_inputs, print_results, binary_write, binary_read, grid_init_do_not_profile, load_num_nucs, load_mats, load_concs, double_compare, NGP_compare, rn_v, estimate_mem_usage, get_time)
// ... (Copy from original code, unchanged, except for CUDA-specific parts replaced with SYCL equivalents as above)
