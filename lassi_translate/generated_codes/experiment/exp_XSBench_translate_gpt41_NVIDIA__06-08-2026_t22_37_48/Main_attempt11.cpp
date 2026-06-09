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
#include <limits>

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

// Utility function implementations

void logo(int version) {
	printf("XSBench SYCL v%d\n", version);
}

void center_print(const char *s, int width) {
	int pad = (width - (int)strlen(s)) / 2;
	for (int i = 0; i < pad; i++) printf(" ");
	printf("%s\n", s);
}

void border_print(void) {
	for (int i = 0; i < 79; i++) printf("=");
	printf("\n");
}

void fancy_int(long a) {
	if (a < 1000) printf("%ld\n", a);
	else if (a < 1000000) printf("%ld,%03ld\n", a / 1000, a % 1000);
	else printf("%ld,%03ld,%03ld\n", a / 1000000, (a / 1000) % 1000, a % 1000);
}

void print_CLI_error(void) {
	printf("Error parsing command line arguments.\n");
	exit(1);
}

Inputs read_CLI(int argc, char *argv[]) {
	Inputs in;
	in.nthreads = 256;
	in.n_isotopes = 355;
	in.n_gridpoints = 11303;
	in.lookups = 1000000;
	in.HM = NULL;
	in.grid_type = UNIONIZED;
	in.hash_bins = 1000;
	in.particles = 10000;
	in.simulation_method = EVENT_BASED;
	in.binary_mode = NONE;
	in.kernel_id = 0;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-g") == 0 && i + 1 < argc)
			in.n_gridpoints = atol(argv[++i]);
		else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
			in.n_isotopes = atol(argv[++i]);
		else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc)
			in.lookups = atoi(argv[++i]);
		else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
			in.nthreads = atoi(argv[++i]);
		else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc)
			in.hash_bins = atoi(argv[++i]);
		else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
			char *method = argv[++i];
			if (strcmp(method, "event") == 0)
				in.simulation_method = EVENT_BASED;
			else if (strcmp(method, "history") == 0)
				in.simulation_method = HISTORY_BASED;
			else
				print_CLI_error();
		}
		else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
			in.kernel_id = atoi(argv[++i]);
		else if (strcmp(argv[i], "-bin") == 0 && i + 1 < argc) {
			char *mode = argv[++i];
			if (strcmp(mode, "read") == 0)
				in.binary_mode = READ;
			else if (strcmp(mode, "write") == 0)
				in.binary_mode = WRITE;
			else
				in.binary_mode = NONE;
		}
	}
	return in;
}

void print_inputs(Inputs in, int nprocs, int version) {
	printf("XSBench SYCL v%d\n", version);
	printf("Number of threads: %d\n", in.nthreads);
	printf("Number of isotopes: %ld\n", in.n_isotopes);
	printf("Number of gridpoints: %ld\n", in.n_gridpoints);
	printf("Number of lookups: %d\n", in.lookups);
	printf("Grid type: %d\n", in.grid_type);
	printf("Hash bins: %d\n", in.hash_bins);
	printf("Simulation method: %d\n", in.simulation_method);
	printf("Kernel ID: %d\n", in.kernel_id);
}

int print_results(Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash) {
	printf("Results:\n");
	printf("Runtime: %lf seconds\n", runtime);
	printf("Verification hash: %llu\n", vhash);
	return 0;
}

void binary_write(Inputs in, SimulationData SD) {
	// Dummy implementation for binary write
	printf("Binary write not implemented.\n");
}

SimulationData binary_read(Inputs in) {
	// Dummy implementation for binary read
	printf("Binary read not implemented.\n");
	SimulationData SD = {};
	return SD;
}

SimulationData grid_init_do_not_profile(Inputs in, int mype) {
	SimulationData SD = {};
	// Dummy initialization for demonstration purposes
	int num_mats = 12;
	SD.length_num_nucs = num_mats;
	SD.num_nucs = (int*)malloc(sizeof(int) * SD.length_num_nucs);
	for (int i = 0; i < SD.length_num_nucs; i++) SD.num_nucs[i] = 3;

	SD.max_num_nucs = 3;
	SD.length_concs = num_mats * SD.max_num_nucs;
	SD.concs = (double*)malloc(sizeof(double) * SD.length_concs);
	for (int i = 0; i < SD.length_concs; i++) SD.concs[i] = 1.0;

	SD.length_mats = num_mats * SD.max_num_nucs;
	SD.mats = (int*)malloc(sizeof(int) * SD.length_mats);
	for (int i = 0; i < SD.length_mats; i++) SD.mats[i] = i % in.n_isotopes;

	SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
	SD.unionized_energy_array = (double*)malloc(sizeof(double) * SD.length_unionized_energy_array);
	for (int i = 0; i < SD.length_unionized_energy_array; i++) SD.unionized_energy_array[i] = (double)i / SD.length_unionized_energy_array;

	SD.length_index_grid = in.n_isotopes * in.n_gridpoints;
	SD.index_grid = (int*)malloc(sizeof(int) * SD.length_index_grid);
	for (int i = 0; i < SD.length_index_grid; i++) SD.index_grid[i] = i % in.n_gridpoints;

	SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
	SD.nuclide_grid = (NuclideGridPoint*)malloc(sizeof(NuclideGridPoint) * SD.length_nuclide_grid);
	for (int i = 0; i < SD.length_nuclide_grid; i++) {
		SD.nuclide_grid[i].energy = (double)i / SD.length_nuclide_grid;
		SD.nuclide_grid[i].total_xs = 1.0;
		SD.nuclide_grid[i].elastic_xs = 1.0;
		SD.nuclide_grid[i].absorbtion_xs = 1.0;
		SD.nuclide_grid[i].fission_xs = 1.0;
		SD.nuclide_grid[i].nu_fission_xs = 1.0;
	}
	SD.length_verification = in.lookups;
	SD.verification = (unsigned long*)malloc(sizeof(unsigned long) * SD.length_verification);
	for (int i = 0; i < SD.length_verification; i++) SD.verification[i] = 0;
	SD.p_energy_samples = nullptr;
	SD.length_p_energy_samples = 0;
	SD.mat_samples = nullptr;
	SD.length_mat_samples = 0;
	return SD;
}

int *load_num_nucs(long n_isotopes) {
	int *num_nucs = (int*)malloc(sizeof(int) * n_isotopes);
	for (long i = 0; i < n_isotopes; i++) num_nucs[i] = 3;
	return num_nucs;
}

int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs) {
	*max_num_nucs = 3;
	int *mats = (int*)malloc(sizeof(int) * n_isotopes * (*max_num_nucs));
	for (long i = 0; i < n_isotopes * (*max_num_nucs); i++) mats[i] = i % n_isotopes;
	return mats;
}

double *load_concs(int *num_nucs, int max_num_nucs) {
	int total = 12 * max_num_nucs;
	double *concs = (double*)malloc(sizeof(double) * total);
	for (int i = 0; i < total; i++) concs[i] = 1.0;
	return concs;
}

int NGP_compare(const void *a, const void *b) {
	NuclideGridPoint *A = (NuclideGridPoint*)a;
	NuclideGridPoint *B = (NuclideGridPoint*)b;
	if (A->energy < B->energy) return -1;
	else if (A->energy > B->energy) return 1;
	else return 0;
}

int double_compare(const void *a, const void *b) {
	double A = *(double*)a;
	double B = *(double*)b;
	if (A < B) return -1;
	else if (A > B) return 1;
	else return 0;
}

double rn_v(void) {
	return (double)rand() / (double)RAND_MAX;
}

size_t estimate_mem_usage(Inputs in) {
	return 0;
}

double get_time(void) {
	using namespace std::chrono;
	static auto start = std::chrono::high_resolution_clock::now();
	auto now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = now - start;
	return elapsed.count();
}

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
	free(GSD.num_nucs, q);
	free(GSD.concs, q);
	free(GSD.mats, q);
	free(GSD.unionized_energy_array, q);
	free(GSD.nuclide_grid, q);
	free(GSD.verification, q);
	if (GSD.p_energy_samples) free(GSD.p_energy_samples, q);
	if (GSD.mat_samples) free(GSD.mat_samples, q);
	if (GSD.index_grid) free(GSD.index_grid, q);
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

// Helper for reduction on device
class ReductionKernel;

// ---- FIX START: Use explicit host reduction instead of an offload reduction kernel ----
unsigned long long device_reduce_sum(queue &q, unsigned long *d_data, size_t n) {
	// Allocate host buffer for copy-back
	std::vector<unsigned long> result_host(n, 0);
	q.memcpy(result_host.data(), d_data, sizeof(unsigned long) * n).wait();
	unsigned long long sum = 0;
	for (size_t i = 0; i < n; ++i) {
		sum += result_host[i];
	}
	return sum;
}
// ---- FIX END ----

// --- FIX: Use USM shared memory for kernel local variables and avoid device pointer arithmetic ---

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype, queue &q) {
	if (mype == 0) printf("Running baseline event-based simulation...\n");
	int nthreads = 256;
	size_t total = in.lookups;

	// Make sure verification buffer is zeroed
	q.memset(GSD.verification, 0, sizeof(unsigned long) * in.lookups).wait();

	// Allocate USM shared memory for kernel local variables (for device compatibility)
	auto p_num_nucs = GSD.num_nucs;
	auto p_concs = GSD.concs;
	auto p_mats = GSD.mats;
	auto p_unionized_energy_array = GSD.unionized_energy_array;
	auto p_index_grid = GSD.index_grid;
	auto p_nuclide_grid = GSD.nuclide_grid;
	auto p_verification = GSD.verification;
	int max_num_nucs = GSD.max_num_nucs;

	// Use USM shared memory for local arrays to avoid device pointer arithmetic
	q.submit([=](handler &h) {
		h.parallel_for(range<1>(total), [=](item<1> idx) {
			int i = idx.get_id(0);
			if (i >= in.lookups) return;
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
			int p_nuc;
			long idx_macro = -1;
			double conc;
			for (int k = 0; k < 5; k++) macro_xs_vector[k] = 0.0;
			if (in.grid_type == UNIONIZED)
				idx_macro = grid_search(in.n_isotopes * in.n_gridpoints, p_energy, p_unionized_energy_array);
			else if (in.grid_type == HASH) {
				double du = 1.0 / in.hash_bins;
				idx_macro = p_energy / du;
			}
			for (int j = 0; j < p_num_nucs[mat]; j++) {
				double xs_vector[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
				p_nuc = p_mats[mat * max_num_nucs + j];
				conc = p_concs[mat * max_num_nucs + j];
				double f;
				NuclideGridPoint low, high;
				long idx_micro = idx_macro;
				if (in.grid_type == NUCLIDE) {
					idx_micro = grid_search_nuclide(in.n_gridpoints, p_energy, &p_nuclide_grid[p_nuc * in.n_gridpoints], 0, in.n_gridpoints - 1);
					if (idx_micro == in.n_gridpoints - 1)
						low = p_nuclide_grid[p_nuc * in.n_gridpoints + idx_micro - 1];
					else
						low = p_nuclide_grid[p_nuc * in.n_gridpoints + idx_micro];
				} else if (in.grid_type == UNIONIZED) {
					if (p_index_grid[idx_micro * in.n_isotopes + p_nuc] == in.n_gridpoints - 1)
						low = p_nuclide_grid[p_nuc * in.n_gridpoints + p_index_grid[idx_micro * in.n_isotopes + p_nuc] - 1];
					else
						low = p_nuclide_grid[p_nuc * in.n_gridpoints + p_index_grid[idx_micro * in.n_isotopes + p_nuc]];
				} else {
					int u_low = p_index_grid[idx_micro * in.n_isotopes + p_nuc];
					int u_high;
					if (idx_micro == in.hash_bins - 1)
						u_high = in.n_gridpoints - 1;
					else
						u_high = p_index_grid[(idx_micro + 1) * in.n_isotopes + p_nuc] + 1;
					double e_low = p_nuclide_grid[p_nuc * in.n_gridpoints + u_low].energy;
					double e_high = p_nuclide_grid[p_nuc * in.n_gridpoints + u_high].energy;
					int lower;
					if (p_energy <= e_low)
						lower = 0;
					else if (p_energy >= e_high)
						lower = in.n_gridpoints - 1;
					else
						lower = grid_search_nuclide(in.n_gridpoints, p_energy, &p_nuclide_grid[p_nuc * in.n_gridpoints], u_low, u_high);
					if (lower == in.n_gridpoints - 1)
						low = p_nuclide_grid[p_nuc * in.n_gridpoints + lower - 1];
					else
						low = p_nuclide_grid[p_nuc * in.n_gridpoints + lower];
				}
				// --- FIX: Avoid pointer arithmetic on device, use index arithmetic ---
				long low_idx = (&low - p_nuclide_grid);
				long high_idx = low_idx + 1;
				if (high_idx >= in.isotopes * in.n_gridpoints)
					high_idx = low_idx; // Clamp to valid range
				high = p_nuclide_grid[high_idx];
				f = (high.energy - p_energy) / (high.energy - low.energy);
				xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);
				xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);
				xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);
				xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);
				xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
				for (int k = 0; k < 5; k++)
					macro_xs_vector[k] += xs_vector[k] * conc;
			}
			double max = -1.0;
			int max_idx = 0;
			for (int j = 0; j < 5; j++) {
				if (macro_xs_vector[j] > max) {
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			p_verification[i] = max_idx + 1;
		});
	}).wait();

	// Reduce verification results
	if (mype == 0) printf("Reducing verification results...\n");
	// Device reduction replaced with explicit host reduction above
	unsigned long verification_scalar = device_reduce_sum(q, GSD.verification, in.lookups);
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
	q.submit([=](handler &h) {
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
	q.submit([=](handler &h) {
		h.parallel_for(range<1>(total), [=](item<1> idx) {
			int i = idx.get_id(0);
			if (i >= in.lookups) return;
			double macro_xs_vector[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
			int p_nuc;
			long idx_macro = -1;
			double conc;
			for (int k = 0; k < 5; k++) macro_xs_vector[k] = 0.0;
			if (in.grid_type == UNIONIZED)
				idx_macro = grid_search(in.n_isotopes * in.n_gridpoints, GSD.p_energy_samples[i], GSD.unionized_energy_array);
			else if (in.grid_type == HASH) {
				double du = 1.0 / in.hash_bins;
				idx_macro = GSD.p_energy_samples[i] / du;
			}
			for (int j = 0; j < GSD.num_nucs[GSD.mat_samples[i]]; j++) {
				double xs_vector[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
				p_nuc = GSD.mats[GSD.mat_samples[i] * GSD.max_num_nucs + j];
				conc = GSD.concs[GSD.mat_samples[i] * GSD.max_num_nucs + j];
				double f;
				NuclideGridPoint low, high;
				long idx_micro = idx_macro;
				if (in.grid_type == NUCLIDE) {
					idx_micro = grid_search_nuclide(in.n_gridpoints, GSD.p_energy_samples[i], &GSD.nuclide_grid[p_nuc * in.n_gridpoints], 0, in.n_gridpoints - 1);
					if (idx_micro == in.n_gridpoints - 1)
						low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + idx_micro - 1];
					else
						low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + idx_micro];
				} else if (in.grid_type == UNIONIZED) {
					if (GSD.index_grid[idx_micro * in.n_isotopes + p_nuc] == in.n_gridpoints - 1)
						low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + GSD.index_grid[idx_micro * in.n_isotopes + p_nuc] - 1];
					else
						low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + GSD.index_grid[idx_micro * in.n_isotopes + p_nuc]];
				} else {
					int u_low = GSD.index_grid[idx_micro * in.n_isotopes + p_nuc];
					int u_high;
					if (idx_micro == in.hash_bins - 1)
						u_high = in.n_gridpoints - 1;
					else
						u_high = GSD.index_grid[(idx_micro + 1) * in.n_isotopes + p_nuc] + 1;
					double e_low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + u_low].energy;
					double e_high = GSD.nuclide_grid[p_nuc * in.n_gridpoints + u_high].energy;
					int lower;
					if (GSD.p_energy_samples[i] <= e_low)
						lower = 0;
					else if (GSD.p_energy_samples[i] >= e_high)
						lower = in.n_gridpoints - 1;
					else
						lower = grid_search_nuclide(in.n_gridpoints, GSD.p_energy_samples[i], &GSD.nuclide_grid[p_nuc * in.n_gridpoints], u_low, u_high);
					if (lower == in.n_gridpoints - 1)
						low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + lower - 1];
					else
						low = GSD.nuclide_grid[p_nuc * in.n_gridpoints + lower];
				}
				// --- FIX: Avoid pointer arithmetic on device, use index arithmetic ---
				long low_idx = (&low - GSD.nuclide_grid);
				long high_idx = low_idx + 1;
				if (high_idx >= in.n_isotopes * in.n_gridpoints)
					high_idx = low_idx; // Clamp to valid range
				high = GSD.nuclide_grid[high_idx];
				f = (high.energy - GSD.p_energy_samples[i]) / (high.energy - low.energy);
				xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);
				xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);
				xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);
				xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);
				xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
				for (int k = 0; k < 5; k++)
					macro_xs_vector[k] += xs_vector[k] * conc;
			}
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
	unsigned long verification_scalar = device_reduce_sum(q, GSD.verification, in.lookups);
	return verification_scalar;
}

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
