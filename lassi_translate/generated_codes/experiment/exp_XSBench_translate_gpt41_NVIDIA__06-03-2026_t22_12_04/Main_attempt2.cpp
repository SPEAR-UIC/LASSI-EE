// ===== HEADER: XSbench_header.cuh =====
#ifndef __XSBENCH_HEADER_H__
#define __XSBENCH_HEADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <functional>

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

inline void gpuAssert(int code, const char *file, int line, bool abort=true)
{
	if (code != 0)
	{
		fprintf(stderr,"GPUassert: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Structures
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
	int grid_type; // 0: Unionized Grid (default)    1: Nuclide Grid
	int hash_bins;
	int particles;
	int simulation_method;
	int binary_mode;
	int kernel_id;
} Inputs;

typedef struct{
	int * num_nucs;                     // Length = length_num_nucs;
	double * concs;                     // Length = length_concs
	int * mats;                         // Length = length_mats
	double * unionized_energy_array;    // Length = length_unionized_energy_array
	int * index_grid;                   // Length = length_index_grid
	NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
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

// io.cu
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int(long a);
Inputs read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
void print_inputs(Inputs in, int nprocs, int version);
int print_results( Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash );
void binary_write( Inputs in, SimulationData SD );
SimulationData binary_read( Inputs in );

// Simulation.cu
unsigned long long run_event_based_simulation_baseline(sycl::queue& q, Inputs in, SimulationData SD, int mype);
void xs_lookup_kernel_baseline(sycl::queue& q, Inputs in, SimulationData GSD );
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           double * egrid, int * index_data,
                           NuclideGridPoint * nuclide_grids,
                           long idx, double * xs_vector, int grid_type, int hash_bins );
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int * num_nucs,
                         double * concs,
                         double * egrid, int * index_data,
                         NuclideGridPoint * nuclide_grids,
                         int * mats,
                         double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs );
long grid_search( long n, double quarry, double * A);
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high);
int pick_mat( uint64_t * seed );
double LCG_random_double(uint64_t * seed);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);

unsigned long long run_event_based_simulation_optimization_1(sycl::queue& q, Inputs in, SimulationData GSD, int mype);
void sampling_kernel(sycl::queue& q, Inputs in, SimulationData GSD );
void xs_lookup_kernel_optimization_1(sycl::queue& q, Inputs in, SimulationData GSD );

unsigned long long run_event_based_simulation_optimization_2(sycl::queue& q, Inputs in, SimulationData GSD, int mype);
void xs_lookup_kernel_optimization_2(sycl::queue& q, Inputs in, SimulationData GSD, int m );

unsigned long long run_event_based_simulation_optimization_3(sycl::queue& q, Inputs in, SimulationData GSD, int mype);
void xs_lookup_kernel_optimization_3(sycl::queue& q, Inputs in, SimulationData GSD, int m );

unsigned long long run_event_based_simulation_optimization_4(sycl::queue& q, Inputs in, SimulationData GSD, int mype);
void xs_lookup_kernel_optimization_4(sycl::queue& q, Inputs in, SimulationData GSD, int m, int n_lookups, int offset );

unsigned long long run_event_based_simulation_optimization_5(sycl::queue& q, Inputs in, SimulationData GSD, int mype);
void xs_lookup_kernel_optimization_5(sycl::queue& q, Inputs in, SimulationData GSD, int n_lookups, int offset );

unsigned long long run_event_based_simulation_optimization_6(sycl::queue& q, Inputs in, SimulationData GSD, int mype);

// GridInit.cu
SimulationData grid_init_do_not_profile( Inputs in, int mype );
SimulationData move_simulation_data_to_device(sycl::queue& q, Inputs in, int mype, SimulationData SD );
void release_device_memory(sycl::queue& q, SimulationData GSD);

// XSutils.cu
int NGP_compare( const void * a, const void * b );
int double_compare(const void * a, const void * b);
double rn_v(void);
size_t estimate_mem_usage( Inputs in );
double get_time(void);

// Materials.cu
int * load_num_nucs(long n_isotopes);
int * load_mats( int * num_nucs, long n_isotopes, int * max_num_nucs );
double * load_concs( int * num_nucs, int max_num_nucs );
#endif

// ===== END HEADER: XSbench_header.cuh =====

sycl::queue q{sycl::gpu_selector_v};

// ====== XSbench_io.cpp ======
void logo(int version) {
	printf("XSBench Version %d\n", version);
}
void center_print(const char *s, int width) {
	int len = strlen(s);
	int pad = (width - len) / 2;
	for(int i=0; i<pad; i++) printf(" ");
	printf("%s\n", s);
}
void border_print(void) {
	for(int i=0; i<79; i++) printf("=");
	printf("\n");
}
void fancy_int(long a) {
	printf("%ld", a);
}
void print_CLI_error(void) {
	printf("Error parsing command line arguments.\n");
}
void print_inputs(Inputs in, int nprocs, int version) {
	logo(version);
	printf("Threads: %d\n", in.nthreads);
	printf("Isotopes: %ld\n", in.n_isotopes);
	printf("Gridpoints: %ld\n", in.n_gridpoints);
	printf("Lookups: %d\n", in.lookups);
	printf("Hash bins: %d\n", in.hash_bins);
	printf("Particles: %d\n", in.particles);
	printf("Simulation method: %d\n", in.simulation_method);
	printf("Binary mode: %d\n", in.binary_mode);
	printf("Kernel ID: %d\n", in.kernel_id);
}
int print_results( Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash ) {
	printf("Runtime: %.6f seconds\n", runtime);
	printf("Verification hash: %llu\n", vhash);
	return 0;
}
void binary_write( Inputs in, SimulationData SD ) {
	// Stub: No-op for this translation
}
SimulationData binary_read( Inputs in ) {
	// Stub: No-op for this translation
	SimulationData SD;
	memset(&SD, 0, sizeof(SimulationData));
	return SD;
}
Inputs read_CLI( int argc, char * argv[] ) {
	Inputs in;
	in.nthreads = 256;
	in.n_isotopes = 68;
	in.n_gridpoints = 10000;
	in.lookups = 1000000;
	in.HM = strdup("UO2");
	in.grid_type = UNIONIZED;
	in.hash_bins = 1000;
	in.particles = 1000000;
	in.simulation_method = EVENT_BASED;
	in.binary_mode = NONE;
	in.kernel_id = 0;
	for(int i=1; i<argc; i++) {
		if(strcmp(argv[i], "-nthreads") == 0 && i+1 < argc)
			in.nthreads = atoi(argv[++i]);
		else if(strcmp(argv[i], "-nisotopes") == 0 && i+1 < argc)
			in.n_isotopes = atol(argv[++i]);
		else if(strcmp(argv[i], "-ngridpoints") == 0 && i+1 < argc)
			in.n_gridpoints = atol(argv[++i]);
		else if(strcmp(argv[i], "-lookups") == 0 && i+1 < argc)
			in.lookups = atoi(argv[++i]);
		else if(strcmp(argv[i], "-gridtype") == 0 && i+1 < argc)
			in.grid_type = atoi(argv[++i]);
		else if(strcmp(argv[i], "-hashbins") == 0 && i+1 < argc)
			in.hash_bins = atoi(argv[++i]);
		else if(strcmp(argv[i], "-particles") == 0 && i+1 < argc)
			in.particles = atoi(argv[++i]);
		else if(strcmp(argv[i], "-m") == 0 && i+1 < argc) {
			if(strcmp(argv[++i], "event") == 0)
				in.simulation_method = EVENT_BASED;
			else
				in.simulation_method = HISTORY_BASED;
		}
		else if(strcmp(argv[i], "-binary") == 0 && i+1 < argc) {
			if(strcmp(argv[++i], "read") == 0)
				in.binary_mode = READ;
			else if(strcmp(argv[i], "write") == 0)
				in.binary_mode = WRITE;
			else
				in.binary_mode = NONE;
		}
		else if(strcmp(argv[i], "-kernel") == 0 && i+1 < argc)
			in.kernel_id = atoi(argv[++i]);
	}
	return in;
}

// ====== XSbench_utils.cpp ======
int NGP_compare( const void * a, const void * b ) {
	NuclideGridPoint *A = (NuclideGridPoint*)a;
	NuclideGridPoint *B = (NuclideGridPoint*)b;
	if(A->energy < B->energy) return -1;
	else if(A->energy > B->energy) return 1;
	else return 0;
}
int double_compare(const void * a, const void * b) {
	double A = *(double*)a;
	double B = *(double*)b;
	if(A < B) return -1;
	else if(A > B) return 1;
	else return 0;
}
double rn_v(void) {
	return ((double)rand()) / ((double)RAND_MAX);
}
size_t estimate_mem_usage( Inputs in ) {
	return (in.n_isotopes * in.n_gridpoints * sizeof(NuclideGridPoint));
}
double get_time(void) {
	using namespace std::chrono;
	static auto t0 = high_resolution_clock::now();
	auto t1 = high_resolution_clock::now();
	duration<double> dt = t1 - t0;
	return dt.count();
}

// ====== XSbench_materials.cpp ======
int * load_num_nucs(long n_isotopes) {
	int * num_nucs = (int*)malloc(12 * sizeof(int));
	int vals[12] = {24, 15, 19, 22, 29, 32, 28, 34, 21, 24, 28, 33};
	for(int i=0; i<12; i++) num_nucs[i] = vals[i];
	return num_nucs;
}
int * load_mats( int * num_nucs, long n_isotopes, int * max_num_nucs ) {
	int * mats = (int*)malloc(12 * 34 * sizeof(int));
	int idx = 0;
	*max_num_nucs = 0;
	for(int i=0; i<12; i++) {
		for(int j=0; j<num_nucs[i]; j++) {
			mats[idx++] = j % n_isotopes;
		}
		if(num_nucs[i] > *max_num_nucs) *max_num_nucs = num_nucs[i];
	}
	return mats;
}
double * load_concs( int * num_nucs, int max_num_nucs ) {
	double * concs = (double*)malloc(12 * max_num_nucs * sizeof(double));
	for(int i=0; i<12; i++) {
		for(int j=0; j<num_nucs[i]; j++) {
			concs[i*max_num_nucs + j] = rn_v();
		}
	}
	return concs;
}

// ====== XSbench_gridinit.cpp ======
SimulationData grid_init_do_not_profile( Inputs in, int mype ) {
	SimulationData SD;
	SD.length_num_nucs = 12;
	SD.num_nucs = load_num_nucs(in.n_isotopes);
	SD.length_concs = 12 * 34;
	int max_num_nucs = 0;
	SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &max_num_nucs);
	SD.length_mats = 12 * 34;
	SD.max_num_nucs = max_num_nucs;
	SD.concs = load_concs(SD.num_nucs, max_num_nucs);
	SD.length_concs = 12 * max_num_nucs;
	SD.length_unionized_energy_array = in.n_gridpoints;
	SD.unionized_energy_array = (double*)malloc(in.n_gridpoints * sizeof(double));
	for(long i=0; i<in.n_gridpoints; i++)
		SD.unionized_energy_array[i] = rn_v();
	SD.length_index_grid = in.n_gridpoints * 12;
	SD.index_grid = (int*)malloc(SD.length_index_grid * sizeof(int));
	for(long i=0; i<SD.length_index_grid; i++)
		SD.index_grid[i] = i % in.n_isotopes;
	SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
	SD.nuclide_grid = (NuclideGridPoint*)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	for(int i=0; i<SD.length_nuclide_grid; i++) {
		SD.nuclide_grid[i].energy = rn_v();
		SD.nuclide_grid[i].total_xs = rn_v();
		SD.nuclide_grid[i].elastic_xs = rn_v();
		SD.nuclide_grid[i].absorbtion_xs = rn_v();
		SD.nuclide_grid[i].fission_xs = rn_v();
		SD.nuclide_grid[i].nu_fission_xs = rn_v();
	}
	SD.verification = nullptr;
	SD.length_verification = 0;
	SD.p_energy_samples = nullptr;
	SD.length_p_energy_samples = 0;
	SD.mat_samples = nullptr;
	SD.length_mat_samples = 0;
	return SD;
}

SimulationData move_simulation_data_to_device(sycl::queue& q, Inputs in, int mype, SimulationData SD )
{
	if(mype == 0) printf("Allocating and moving simulation data to GPU memory space...\n");

	size_t sz;
	size_t total_sz = 0;

	SimulationData GSD = SD;

	sz = GSD.length_num_nucs * sizeof(int);
	GSD.num_nucs = (int*) sycl::malloc_device(sz, q);
	q.memcpy(GSD.num_nucs, SD.num_nucs, sz).wait();
	total_sz += sz;

	sz = GSD.length_concs * sizeof(double);
	GSD.concs = (double*) sycl::malloc_device(sz, q);
	q.memcpy(GSD.concs, SD.concs, sz).wait();
	total_sz += sz;

	sz = GSD.length_mats * sizeof(int);
	GSD.mats = (int*) sycl::malloc_device(sz, q);
	q.memcpy(GSD.mats, SD.mats, sz).wait();
	total_sz += sz;

	sz = GSD.length_unionized_energy_array * sizeof(double);
	if (GSD.length_unionized_energy_array > 0) {
		GSD.unionized_energy_array = (double*) sycl::malloc_device(sz, q);
		q.memcpy(GSD.unionized_energy_array, SD.unionized_energy_array, sz).wait();
		total_sz += sz;
	} else {
		GSD.unionized_energy_array = nullptr;
	}

	sz = GSD.length_index_grid * sizeof(int);
	if (GSD.length_index_grid > 0) {
		GSD.index_grid = (int*) sycl::malloc_device(sz, q);
		q.memcpy(GSD.index_grid, SD.index_grid, sz).wait();
		total_sz += sz;
	} else {
		GSD.index_grid = nullptr;
	}

	sz = GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
	GSD.nuclide_grid = (NuclideGridPoint*) sycl::malloc_device(sz, q);
	q.memcpy(GSD.nuclide_grid, SD.nuclide_grid, sz).wait();
	total_sz += sz;

	sz = in.lookups * sizeof(unsigned long);
	GSD.verification = (unsigned long*) sycl::malloc_device(sz, q);
	total_sz += sz;
	GSD.length_verification = in.lookups;

	// Set these to nullptr initially
	GSD.p_energy_samples = nullptr;
	GSD.mat_samples = nullptr;
	GSD.length_p_energy_samples = 0;
	GSD.length_mat_samples = 0;

	if(mype == 0 ) printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz/1024.0/1024.0 );

	return GSD;
}

void release_device_memory(sycl::queue& q, SimulationData GSD) {
	if(GSD.num_nucs) sycl::free(GSD.num_nucs, q);
	if(GSD.concs) sycl::free(GSD.concs, q);
	if(GSD.mats) sycl::free(GSD.mats, q);
	if(GSD.unionized_energy_array) sycl::free(GSD.unionized_energy_array, q);
	if(GSD.index_grid) sycl::free(GSD.index_grid, q);
	if(GSD.nuclide_grid) sycl::free(GSD.nuclide_grid, q);
	if(GSD.verification) sycl::free(GSD.verification, q);
	if(GSD.p_energy_samples) sycl::free(GSD.p_energy_samples, q);
	if(GSD.mat_samples) sycl::free(GSD.mat_samples, q);
}

// ====== Simulation.cu ======
long grid_search( long n, double quarry, double * A) {
	long low = 0;
	long high = n-1;
	while(high - low > 1) {
		long mid = low + (high - low) / 2;
		if(A[mid] > quarry)
			high = mid;
		else
			low = mid;
	}
	return low;
}
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high) {
	while(high - low > 1) {
		long mid = low + (high - low) / 2;
		if(A[mid].energy > quarry)
			high = mid;
		else
			low = mid;
	}
	return low;
}
double LCG_random_double(uint64_t * seed) {
	*seed = (*seed * 6364136223846793005ULL + 1ULL);
	return ((double)(*seed & 0xFFFFFFFFFFFFULL)) / (double)0xFFFFFFFFFFFFULL;
}
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
	uint64_t a = 6364136223846793005ULL;
	uint64_t c = 1ULL;
	uint64_t a_pow = 1ULL;
	uint64_t c_sum = 0ULL;
	for(uint64_t i=0; i<n; i++) {
		a_pow *= a;
		c_sum = c_sum * a + c;
	}
	return a_pow * seed + c_sum;
}
int pick_mat( uint64_t * seed ) {
	double r = LCG_random_double(seed);
	return (int)(r * 12);
}
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           double * egrid, int * index_data,
                           NuclideGridPoint * nuclide_grids,
                           long idx, double * xs_vector, int grid_type, int hash_bins ) {
	long low = 0, high = n_gridpoints-1;
	long grid_idx = grid_search(n_gridpoints, p_energy, egrid);
	long nuclide_idx = nuc * n_gridpoints + grid_idx;
	xs_vector[0] = nuclide_grids[nuclide_idx].total_xs;
	xs_vector[1] = nuclide_grids[nuclide_idx].elastic_xs;
	xs_vector[2] = nuclide_grids[nuclide_idx].absorbtion_xs;
	xs_vector[3] = nuclide_grids[nuclide_idx].fission_xs;
	xs_vector[4] = nuclide_grids[nuclide_idx].nu_fission_xs;
}
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int * num_nucs,
                         double * concs,
                         double * egrid, int * index_data,
                         NuclideGridPoint * nuclide_grids,
                         int * mats,
                         double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ) {
	for(int i=0; i<5; i++) macro_xs_vector[i] = 0.0;
	int n_nucs = num_nucs[mat];
	for(int i=0; i<n_nucs; i++) {
		int nuc = mats[mat*max_num_nucs + i];
		double xs_vector[5];
		calculate_micro_xs(p_energy, nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids, 0, xs_vector, grid_type, hash_bins);
		double conc = concs[mat*max_num_nucs + i];
		for(int j=0; j<5; j++)
			macro_xs_vector[j] += xs_vector[j] * conc;
	}
}

// Baseline event-based simulation
unsigned long long run_event_based_simulation_baseline(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	unsigned long long checksum = 0;
	int lookups = in.lookups;
	int nthreads = in.nthreads;
	int n_isotopes = in.n_isotopes;
	int n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = GSD.max_num_nucs;
	int * num_nucs = GSD.num_nucs;
	double * concs = GSD.concs;
	double * egrid = GSD.unionized_energy_array;
	int * index_data = GSD.index_grid;
	NuclideGridPoint * nuclide_grids = GSD.nuclide_grid;
	int * mats = GSD.mats;

	unsigned long * verification = GSD.verification;
	q.submit([&](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			uint64_t seed = fast_forward_LCG(STARTING_SEED, i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5];
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, egrid, index_data, nuclide_grids, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
			unsigned long hash = 0;
			for(int j=0; j<5; j++)
				hash += (unsigned long)(macro_xs_vector[j] * 1e6);
			verification[i] = hash;
		});
	}).wait();

	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

// Optimization 1: Sampling kernel
void sampling_kernel(sycl::queue& q, Inputs in, SimulationData GSD ) {
	int lookups = in.lookups;
	GSD.length_p_energy_samples = lookups;
	GSD.length_mat_samples = lookups;
	GSD.p_energy_samples = (double*)sycl::malloc_device(lookups * sizeof(double), q);
	GSD.mat_samples = (int*)sycl::malloc_device(lookups * sizeof(int), q);
	q.submit([&](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			uint64_t seed = fast_forward_LCG(STARTING_SEED, i);
			GSD.p_energy_samples[i] = LCG_random_double(&seed);
			GSD.mat_samples[i] = pick_mat(&seed);
		});
	}).wait();
}

void xs_lookup_kernel_optimization_1(sycl::queue& q, Inputs in, SimulationData GSD ) {
	int lookups = in.lookups;
	int n_isotopes = in.n_isotopes;
	int n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = GSD.max_num_nucs;
	int * num_nucs = GSD.num_nucs;
	double * concs = GSD.concs;
	double * egrid = GSD.unionized_energy_array;
	int * index_data = GSD.index_grid;
	NuclideGridPoint * nuclide_grids = GSD.nuclide_grid;
	int * mats = GSD.mats;
	double * p_energy_samples = GSD.p_energy_samples;
	int * mat_samples = GSD.mat_samples;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			double p_energy = p_energy_samples[i];
			int mat = mat_samples[i];
			double macro_xs_vector[5];
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, egrid, index_data, nuclide_grids, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
			unsigned long hash = 0;
			for(int j=0; j<5; j++)
				hash += (unsigned long)(macro_xs_vector[j] * 1e6);
			verification[i] = hash;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_1(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	sampling_kernel(q, in, GSD);
	xs_lookup_kernel_optimization_1(q, in, GSD);
	unsigned long long checksum = 0;
	int lookups = in.lookups;
	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, GSD.verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

void xs_lookup_kernel_optimization_2(sycl::queue& q, Inputs in, SimulationData GSD, int m ) {
	int lookups = in.lookups;
	int n_isotopes = in.n_isotopes;
	int n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = GSD.max_num_nucs;
	int * num_nucs = GSD.num_nucs;
	double * concs = GSD.concs;
	double * egrid = GSD.unionized_energy_array;
	int * index_data = GSD.index_grid;
	NuclideGridPoint * nuclide_grids = GSD.nuclide_grid;
	int * mats = GSD.mats;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			uint64_t seed = fast_forward_LCG(STARTING_SEED, i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5];
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, egrid, index_data, nuclide_grids, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
			unsigned long hash = 0;
			for(int j=0; j<5; j++)
				hash += (unsigned long)(macro_xs_vector[j] * 1e6);
			verification[i] = hash;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_2(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	xs_lookup_kernel_optimization_2(q, in, GSD, 0);
	unsigned long long checksum = 0;
	int lookups = in.lookups;
	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, GSD.verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

void xs_lookup_kernel_optimization_3(sycl::queue& q, Inputs in, SimulationData GSD, int m ) {
	int lookups = in.lookups;
	int n_isotopes = in.n_isotopes;
	int n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = GSD.max_num_nucs;
	int * num_nucs = GSD.num_nucs;
	double * concs = GSD.concs;
	double * egrid = GSD.unionized_energy_array;
	int * index_data = GSD.index_grid;
	NuclideGridPoint * nuclide_grids = GSD.nuclide_grid;
	int * mats = GSD.mats;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			uint64_t seed = fast_forward_LCG(STARTING_SEED, i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5];
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, egrid, index_data, nuclide_grids, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
			unsigned long hash = 0;
			for(int j=0; j<5; j++)
				hash += (unsigned long)(macro_xs_vector[j] * 1e6);
			verification[i] = hash;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_3(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	xs_lookup_kernel_optimization_3(q, in, GSD, 0);
	unsigned long long checksum = 0;
	int lookups = in.lookups;
	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, GSD.verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

void xs_lookup_kernel_optimization_4(sycl::queue& q, Inputs in, SimulationData GSD, int m, int n_lookups, int offset ) {
	int n_isotopes = in.n_isotopes;
	int n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = GSD.max_num_nucs;
	int * num_nucs = GSD.num_nucs;
	double * concs = GSD.concs;
	double * egrid = GSD.unionized_energy_array;
	int * index_data = GSD.index_grid;
	NuclideGridPoint * nuclide_grids = GSD.nuclide_grid;
	int * mats = GSD.mats;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(n_lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0) + offset;
			uint64_t seed = fast_forward_LCG(STARTING_SEED, i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5];
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, egrid, index_data, nuclide_grids, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
			unsigned long hash = 0;
			for(int j=0; j<5; j++)
				hash += (unsigned long)(macro_xs_vector[j] * 1e6);
			verification[i] = hash;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_4(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	int lookups = in.lookups;
	int nthreads = in.nthreads;
	int n_blocks = (lookups + nthreads - 1) / nthreads;
	for(int b=0; b<n_blocks; b++) {
		int offset = b * nthreads;
		int n_lookups = std::min(nthreads, lookups - offset);
		xs_lookup_kernel_optimization_4(q, in, GSD, b, n_lookups, offset);
	}
	unsigned long long checksum = 0;
	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, GSD.verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

void xs_lookup_kernel_optimization_5(sycl::queue& q, Inputs in, SimulationData GSD, int n_lookups, int offset ) {
	int n_isotopes = in.n_isotopes;
	int n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = GSD.max_num_nucs;
	int * num_nucs = GSD.num_nucs;
	double * concs = GSD.concs;
	double * egrid = GSD.unionized_energy_array;
	int * index_data = GSD.index_grid;
	NuclideGridPoint * nuclide_grids = GSD.nuclide_grid;
	int * mats = GSD.mats;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(n_lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0) + offset;
			uint64_t seed = fast_forward_LCG(STARTING_SEED, i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5];
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs, egrid, index_data, nuclide_grids, mats, macro_xs_vector, grid_type, hash_bins, max_num_nucs);
			unsigned long hash = 0;
			for(int j=0; j<5; j++)
				hash += (unsigned long)(macro_xs_vector[j] * 1e6);
			verification[i] = hash;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_5(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	int lookups = in.lookups;
	int nthreads = in.nthreads;
	int n_blocks = (lookups + nthreads - 1) / nthreads;
	for(int b=0; b<n_blocks; b++) {
		int offset = b * nthreads;
		int n_lookups = std::min(nthreads, lookups - offset);
		xs_lookup_kernel_optimization_5(q, in, GSD, n_lookups, offset);
	}
	unsigned long long checksum = 0;
	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, GSD.verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

unsigned long long run_event_based_simulation_optimization_6(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	// Same as baseline for this translation
	return run_event_based_simulation_baseline(q, in, GSD, mype);
}

// ====== MAIN ======
int main( int argc, char* argv[] )
{
	int version = 20;
	int mype = 0;
	double omp_start, omp_end;
	int nprocs = 1;
	unsigned long long verification;

	Inputs in = read_CLI( argc, argv );

	if( mype == 0 )
		print_inputs( in, nprocs, version );

	SimulationData SD;

	if( in.binary_mode == READ )
		SD = binary_read(in);
	else
		SD = grid_init_do_not_profile( in, mype );

	if( in.binary_mode == WRITE && mype == 0 )
		binary_write(in, SD);

	SimulationData GSD = move_simulation_data_to_device(q, in, mype, SD );

	if( mype == 0 )
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	omp_start = get_time();

	if( in.simulation_method == EVENT_BASED )
	{
		if( in.kernel_id == 0 )
			verification = run_event_based_simulation_baseline(q, in, GSD, mype);
		else if( in.kernel_id == 1 )
			verification = run_event_based_simulation_optimization_1(q, in, GSD, mype);
		else if( in.kernel_id == 2 )
			verification = run_event_based_simulation_optimization_2(q, in, GSD, mype);
		else if( in.kernel_id == 3 )
			verification = run_event_based_simulation_optimization_3(q, in, GSD, mype);
		else if( in.kernel_id == 4 )
			verification = run_event_based_simulation_optimization_4(q, in, GSD, mype);
		else if( in.kernel_id == 5 )
			verification = run_event_based_simulation_optimization_5(q, in, GSD, mype);
		else if( in.kernel_id == 6 )
			verification = run_event_based_simulation_optimization_6(q, in, GSD, mype);
		else
		{
			printf("Error: No kernel ID %d found!\n", in.kernel_id);
			exit(1);
		}
	}
	else
	{
		printf("History-based simulation not implemented in CUDA code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
		exit(1);
	}

	if( mype == 0)
	{
		printf("\n" );
		printf("Simulation complete.\n" );
	}

	omp_end = get_time();

	release_device_memory(q, GSD);

	verification = verification % 999983;

	int is_invalid_result = print_results( in, mype, omp_end-omp_start, nprocs, verification );

	return is_invalid_result;
}
