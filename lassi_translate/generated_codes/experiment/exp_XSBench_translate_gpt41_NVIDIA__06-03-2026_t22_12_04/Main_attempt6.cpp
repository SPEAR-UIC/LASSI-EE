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

// ====== XSbench_utils.cpp ======
int NGP_compare( const void * a, const void * b ) {
	const NuclideGridPoint *A = (const NuclideGridPoint*)a;
	const NuclideGridPoint *B = (const NuclideGridPoint*)b;
	if(A->energy < B->energy) return -1;
	else if(A->energy > B->energy) return 1;
	else return 0;
}
int double_compare(const void * a, const void * b) {
	const double A = *(const double*)a;
	const double B = *(const double*)b;
	if(A < B) return -1;
	else if(A > B) return 1;
	else return 0;
}
double rn_v(void) {
	static unsigned long seed = 1337;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (seed) ) % m;
	seed = n1;
	return (double) n1 / m;
}
size_t estimate_mem_usage( Inputs in ) {
	size_t single_nuclide_grid = in.n_gridpoints * sizeof( NuclideGridPoint );
	size_t all_nuclide_grids   = in.n_isotopes * single_nuclide_grid;
	size_t size_UEG            = in.n_isotopes*in.n_gridpoints*sizeof(double) + in.n_isotopes*in.n_gridpoints*in.n_isotopes*sizeof(int);
	size_t size_hash_grid      = in.hash_bins * in.n_isotopes * sizeof(int);
	size_t memtotal;

	if( in.grid_type == UNIONIZED )
		memtotal          = all_nuclide_grids + size_UEG;
	else if( in.grid_type == NUCLIDE )
		memtotal          = all_nuclide_grids;
	else
		memtotal          = all_nuclide_grids + size_hash_grid;

	memtotal          = ceil(memtotal / (1024.0*1024.0));
	return memtotal;
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
	if( n_isotopes == 68 )
		num_nucs[0]  = 34;
	else
		num_nucs[0]  = 321;
	num_nucs[1]  = 5;
	num_nucs[2]  = 4;
	num_nucs[3]  = 4;
	num_nucs[4]  = 27;
	num_nucs[5]  = 21;
	num_nucs[6]  = 21;
	num_nucs[7]  = 21;
	num_nucs[8]  = 21;
	num_nucs[9]  = 21;
	num_nucs[10] = 9;
	num_nucs[11] = 9;
	return num_nucs;
}
int * load_mats( int * num_nucs, long n_isotopes, int * max_num_nucs ) {
	*max_num_nucs = 0;
	int num_mats = 12;
	for( int m = 0; m < num_mats; m++ )
	{
		if( num_nucs[m] > *max_num_nucs )
			*max_num_nucs = num_nucs[m];
	}
	int * mats = (int *) malloc( num_mats * (*max_num_nucs) * sizeof(int) );
	int mats0_Sml[] =  { 58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
	                 8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53, 
	                 54, 55, 56, 18, 23, 41 };
	int mats0_Lrg[321] =  { 58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
	                 8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53,
	                 54, 55, 56, 18, 23, 41 };
	for( int i = 0; i < 321-34; i++ )
		mats0_Lrg[34+i] = 68 + i;
	int mats1[] =  { 63, 64, 65, 66, 67 };
	int mats2[] =  { 24, 41, 4, 5 };
	int mats3[] =  { 24, 41, 4, 5 };
	int mats4[] =  { 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29,
	                 30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16,
	                 17 };
	int mats5[] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
	int mats6[] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
	int mats7[] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
	int mats8[] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
	int mats9[] =  { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
	int mats10[] = { 24, 41, 4, 5, 63, 64, 65, 66, 67 };
	int mats11[] = { 24, 41, 4, 5, 63, 64, 65, 66, 67 };
	if( n_isotopes == 68 )
		memcpy( mats,  mats0_Sml,  num_nucs[0]  * sizeof(int) );	
	else
		memcpy( mats,  mats0_Lrg,  num_nucs[0]  * sizeof(int) );
	memcpy( mats + *max_num_nucs * 1,  mats1,  num_nucs[1]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 2,  mats2,  num_nucs[2]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 3,  mats3,  num_nucs[3]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 4,  mats4,  num_nucs[4]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 5,  mats5,  num_nucs[5]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 6,  mats6,  num_nucs[6]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 7,  mats7,  num_nucs[7]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 8,  mats8,  num_nucs[8]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 9,  mats9,  num_nucs[9]  * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 10, mats10, num_nucs[10] * sizeof(int) );	
	memcpy( mats + *max_num_nucs * 11, mats11, num_nucs[11] * sizeof(int) );	
	return mats;
}
double * load_concs( int * num_nucs, int max_num_nucs ) {
	uint64_t seed = STARTING_SEED * STARTING_SEED;
	double * concs = (double *) malloc( 12 * max_num_nucs * sizeof( double ) );
	for( int i = 0; i < 12; i++ )
		for( int j = 0; j < num_nucs[i]; j++ )
			concs[i * max_num_nucs + j] = rn_v();
	return concs;
}

// ====== XSbench_gridinit.cpp ======
SimulationData grid_init_do_not_profile( Inputs in, int mype ) {
	SimulationData SD;
	SD.length_num_nucs = 12;
	SD.num_nucs = load_num_nucs(in.n_isotopes);
	int max_num_nucs = 0;
	SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &max_num_nucs);
	SD.max_num_nucs = max_num_nucs;
	SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
	SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
	SD.length_concs = SD.length_mats;
	SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
	SD.nuclide_grid = (NuclideGridPoint*)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	uint64_t seed = 42;
	for(int i=0; i<SD.length_nuclide_grid; i++) {
		SD.nuclide_grid[i].energy = rn_v();
		SD.nuclide_grid[i].total_xs = rn_v();
		SD.nuclide_grid[i].elastic_xs = rn_v();
		SD.nuclide_grid[i].absorbtion_xs = rn_v();
		SD.nuclide_grid[i].fission_xs = rn_v();
		SD.nuclide_grid[i].nu_fission_xs = rn_v();
	}
	SD.length_unionized_energy_array = in.n_gridpoints;
	SD.unionized_energy_array = (double*)malloc(in.n_gridpoints * sizeof(double));
	for(long i=0; i<in.n_gridpoints; i++)
		SD.unionized_energy_array[i] = rn_v();
	SD.length_index_grid = in.n_gridpoints * 12;
	SD.index_grid = (int*)malloc(SD.length_index_grid * sizeof(int));
	for(long i=0; i<SD.length_index_grid; i++)
		SD.index_grid[i] = i % in.n_isotopes;
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

// ====== XSbench_io.cpp ======
void logo(int version) {
	border_print();
	printf(
	"                   __   __ ___________                 _                        \n"
	"                   \\ \\ / //  ___| ___ \\               | |                       \n"
	"                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
	"                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
	"                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
	"                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n"
    );
	border_print();
	center_print("Developed at Argonne National Laboratory", 79);
	char v[100];
	sprintf(v, "Version: %d", version);
	center_print(v, 79);
	border_print();
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
    if( a < 1000 )
        printf("%ld\n",a);

    else if( a >= 1000 && a < 1000000 )
        printf("%ld,%03ld\n", a / 1000, a % 1000);

    else if( a >= 1000000 && a < 1000000000 )
        printf("%ld,%03ld,%03ld\n",a / 1000000,(a % 1000000) / 1000,a % 1000 );

    else if( a >= 1000000000 )
        printf("%ld,%03ld,%03ld,%03ld\n",
               a / 1000000000,
               (a % 1000000000) / 1000000,
               (a % 1000000) / 1000,
               a % 1000 );
    else
        printf("%ld\n",a);
}
void print_CLI_error(void) {
	printf("Error parsing command line arguments.\n");
	exit(4);
}
void print_inputs(Inputs in, int nprocs, int version) {
	logo(version);
	center_print("INPUT SUMMARY", 79);
	border_print();
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
	int lookups = 0;
	if( in.simulation_method == HISTORY_BASED )
		lookups = in.lookups * in.particles;
	else if( in.simulation_method == EVENT_BASED )
		lookups = in.lookups;
	int lookups_per_sec = (int) ((double) lookups / runtime);

	int is_invalid_result = 1;

	if( mype == 0 )
	{
		border_print();
		center_print("RESULTS", 79);
		border_print();

		printf("Runtime:     %.3lf seconds\n", runtime);
		printf("Lookups:     "); fancy_int(lookups);
		printf("Lookups/s:   ");
		fancy_int(lookups_per_sec);
	}

	unsigned long long large = 0;
	unsigned long long small = 0; 
	if( in.simulation_method == EVENT_BASED )
	{
		small = 945990;
		large = 952131;
	}
	else if( in.simulation_method == HISTORY_BASED )
	{
		small = 941535;
		large = 954318; 
	}
	if( in.HM && strcmp(in.HM, "large") == 0 )
	{
		if( vhash == large )
			is_invalid_result = 0;
	}
	else if( in.HM && strcmp(in.HM, "small") == 0 )
	{
		if( vhash == small )
			is_invalid_result = 0;
	}

	if(mype == 0 )
	{
		if( is_invalid_result )
			printf("Verification hash: %llu\n", vhash);
		else
			printf("Verification hash: %llu (Valid)\n", vhash);
		border_print();
	}

	return is_invalid_result;
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
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0) 
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;
}
int pick_mat( uint64_t * seed ) {
	double dist[12] = {0.140,0.052,0.275,0.134,0.154,0.064,0.066,0.055,0.008,0.015,0.025,0.013};
	double roll = LCG_random_double(seed);
	for( int i = 0; i < 12; i++ )
	{
		double running = 0;
		for( int j = i; j > 0; j-- )
			running += dist[j];
		if( roll < running )
			return i;
	}
	return 0;
}
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           double * egrid, int * index_data,
                           NuclideGridPoint * nuclide_grids,
                           long idx, double * xs_vector, int grid_type, int hash_bins ) {
	double f;
	NuclideGridPoint * low, * high;
	if( grid_type == NUCLIDE )
	{
		idx = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], 0, n_gridpoints-1);
		if( idx == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + idx - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + idx];
	}
	else if( grid_type == UNIONIZED)
	{
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc]];
	}
	else
	{
		int u_low = index_data[idx * n_isotopes + nuc];
		int u_high;
		if( idx == hash_bins - 1 )
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;
		double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
		double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
		int lower;
		if( p_energy <= e_low )
			lower = 0;
		else if( p_energy >= e_high )
			lower = n_gridpoints - 1;
		else
			lower = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], u_low, u_high);
		if( lower == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + lower - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + lower];
	}
	high = low + 1;
	f = (high->energy - p_energy) / (high->energy - low->energy);
	xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
	xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
	xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
	xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
	xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int * num_nucs,
                         double * concs,
                         double * egrid, int * index_data,
                         NuclideGridPoint * nuclide_grids,
                         int * mats,
                         double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ) {
	long idx = -1;
	for(int k=0; k<5; k++) macro_xs_vector[k] = 0.0;
	if( grid_type == UNIONIZED )
		idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);
	else if( grid_type == HASH )
	{
		double du = 1.0 / hash_bins;
		idx = p_energy / du;
	}
	int n_nucs = num_nucs[mat];
	for(int j=0; j<n_nucs; j++) {
		int p_nuc = mats[mat*max_num_nucs + j];
		double conc = concs[mat*max_num_nucs + j];
		double xs_vector[5];
		calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids, idx, xs_vector, grid_type, hash_bins);
		for(int k=0; k<5; k++)
			macro_xs_vector[k] += xs_vector[k] * conc;
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
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2*i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs,
				egrid, index_data, nuclide_grids, mats, macro_xs_vector,
				grid_type, hash_bins, max_num_nucs
			);
			double max = -1.0;
			int max_idx = 0;
			for(int j=0; j<5; j++) {
				if(macro_xs_vector[j] > max) {
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification[i] = max_idx+1;
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
	if(GSD.p_energy_samples) sycl::free(GSD.p_energy_samples, q);
	if(GSD.mat_samples) sycl::free(GSD.mat_samples, q);
	GSD.length_p_energy_samples = lookups;
	GSD.length_mat_samples = lookups;
	GSD.p_energy_samples = (double*)sycl::malloc_device(lookups * sizeof(double), q);
	GSD.mat_samples = (int*)sycl::malloc_device(lookups * sizeof(int), q);
	q.submit([&](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2*i);
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
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs,
				egrid, index_data, nuclide_grids, mats, macro_xs_vector,
				grid_type, hash_bins, max_num_nucs
			);
			double max = -1.0;
			int max_idx = 0;
			for(int j=0; j<5; j++) {
				if(macro_xs_vector[j] > max) {
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification[i] = max_idx+1;
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
	double * p_energy_samples = GSD.p_energy_samples;
	int * mat_samples = GSD.mat_samples;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0);
			int mat = mat_samples[i];
			if(mat != m) return;
			double p_energy = p_energy_samples[i];
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs,
				egrid, index_data, nuclide_grids, mats, macro_xs_vector,
				grid_type, hash_bins, max_num_nucs
			);
			double max = -1.0;
			int max_idx = 0;
			for(int j=0; j<5; j++) {
				if(macro_xs_vector[j] > max) {
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification[i] = max_idx+1;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_2(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	sampling_kernel(q, in, GSD);
	for(int m=0; m<12; m++)
		xs_lookup_kernel_optimization_2(q, in, GSD, m);
	unsigned long long checksum = 0;
	int lookups = in.lookups;
	unsigned long * host_verification = (unsigned long*)malloc(lookups * sizeof(unsigned long));
	q.memcpy(host_verification, GSD.verification, lookups * sizeof(unsigned long)).wait();
	for(int i=0; i<lookups; i++)
		checksum += host_verification[i];
	free(host_verification);
	return checksum;
}

void xs_lookup_kernel_optimization_3(sycl::queue& q, Inputs in, SimulationData GSD, int is_fuel ) {
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
			int mat = mat_samples[i];
			if( ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0 )) ) {
				double p_energy = p_energy_samples[i];
				double macro_xs_vector[5] = {0};
				calculate_macro_xs(
					p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs,
					egrid, index_data, nuclide_grids, mats, macro_xs_vector,
					grid_type, hash_bins, max_num_nucs
				);
				double max = -1.0;
				int max_idx = 0;
				for(int j=0; j<5; j++) {
					if(macro_xs_vector[j] > max) {
						max = macro_xs_vector[j];
						max_idx = j;
					}
				}
				verification[i] = max_idx+1;
			}
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_3(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	sampling_kernel(q, in, GSD);
	xs_lookup_kernel_optimization_3(q, in, GSD, 0);
	xs_lookup_kernel_optimization_3(q, in, GSD, 1);
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
	double * p_energy_samples = GSD.p_energy_samples;
	int * mat_samples = GSD.mat_samples;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(n_lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0) + offset;
			int mat = mat_samples[i];
			if(mat != m) return;
			double p_energy = p_energy_samples[i];
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs,
				egrid, index_data, nuclide_grids, mats, macro_xs_vector,
				grid_type, hash_bins, max_num_nucs
			);
			double max = -1.0;
			int max_idx = 0;
			for(int j=0; j<5; j++) {
				if(macro_xs_vector[j] > max) {
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification[i] = max_idx+1;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_4(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	sampling_kernel(q, in, GSD);
	int lookups = in.lookups;
	int nthreads = in.nthreads;
	int n_blocks = (lookups + nthreads - 1) / nthreads;
	for(int b=0; b<n_blocks; b++) {
		int offset = b * nthreads;
		int n_lookups = std::min(nthreads, lookups - offset);
		for(int m=0; m<12; m++)
			xs_lookup_kernel_optimization_4(q, in, GSD, m, n_lookups, offset);
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
	double * p_energy_samples = GSD.p_energy_samples;
	int * mat_samples = GSD.mat_samples;
	unsigned long * verification = GSD.verification;
	q.submit([=](sycl::handler& h) {
		h.parallel_for(sycl::range<1>(n_lookups), [=](sycl::item<1> item) {
			int i = item.get_id(0) + offset;
			double p_energy = p_energy_samples[i];
			int mat = mat_samples[i];
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, n_isotopes, n_gridpoints, num_nucs, concs,
				egrid, index_data, nuclide_grids, mats, macro_xs_vector,
				grid_type, hash_bins, max_num_nucs
			);
			double max = -1.0;
			int max_idx = 0;
			for(int j=0; j<5; j++) {
				if(macro_xs_vector[j] > max) {
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
			verification[i] = max_idx+1;
		});
	}).wait();
}

unsigned long long run_event_based_simulation_optimization_5(sycl::queue& q, Inputs in, SimulationData GSD, int mype) {
	sampling_kernel(q, in, GSD);
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
