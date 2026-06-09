/*
 * UNIFIED MINIMDOCK KERNEL - SYCL VERSION
 * This file contains all kernel implementations merged into a single unit.
 * IMPORTANT: This is a SINGLE, MONOLITHIC code file. All code sections are 
 * interdependent and must be regenerated together as one complete unit.
 * DO NOT treat sections as separate files - this is ONE continuous codebase.
 */

#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdint.h>

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

// Forward declarations
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

// RNG functions
double LCG_random_double(uint64_t *seed);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);

// Device-side helpers
long grid_search(long n, double quarry, double *A);
long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high);
int pick_mat(uint64_t *seed);

// Kernel helpers
void calculate_micro_xs(double p_energy, int nuc, long n_isotopes,
                        long n_gridpoints, double *egrid, int *index_data,
                        NuclideGridPoint *nuclide_grids, long idx, double *xs_vector,
                        int grid_type, int hash_bins);

void calculate_macro_xs(double p_energy, int mat, long n_isotopes,
                        long n_gridpoints, int *num_nucs, double *concs,
                        double *egrid, int *index_data, NuclideGridPoint *nuclide_grids,
                        int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs);

// SYCL kernel wrappers
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype, queue &q);
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype, queue &q);
unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype, queue &q);
unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype, queue &q);
unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype, queue &q);
unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype, queue &q);
unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype, queue &q);

// Main function
int main(int argc, char *argv[])
{
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

	// SYCL device selection for energy efficiency
	auto devices = device::get_devices();
	device best_device = devices[0];
	for (auto &dev : devices) {
		// Example: select GPU if available, else CPU
		if (dev.is_gpu()) {
			best_device = dev;
			break;
		}
	}
	queue q(best_device, property::queue::in_order());

	SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

	if (mype == 0) {
		std::cout << "\n";
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
		else if (in.kernel_id == 2)
			verification = run_event_based_simulation_optimization_2(in, GSD, mype, q);
		else if (in.kernel_id == 3)
			verification = run_event_based_simulation_optimization_3(in, GSD, mype, q);
		else if (in.kernel_id == 4)
			verification = run_event_based_simulation_optimization_4(in, GSD, mype, q);
		else if (in.kernel_id == 5)
			verification = run_event_based_simulation_optimization_5(in, GSD, mype, q);
		else if (in.kernel_id == 6)
			verification = run_event_based_simulation_optimization_6(in, GSD, mype, q);
		else {
			std::cout << "Error: No kernel ID " << in.kernel_id << " found!\n";
			exit(1);
		}
	} else {
		std::cout << "History-based simulation not implemented in CUDA code. Instead,\nuse the event-based method with \"-m event\" argument.\n";
		exit(1);
	}

	if (mype == 0) {
		std::cout << "\nSimulation complete.\n";
	}

	omp_end = get_time();

	release_device_memory(GSD);

	verification = verification % 999983;

	int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);

	return is_invalid_result;
}

// Prints program logo
void logo(int version)
{
	border_print();
	std::cout <<
	"                   __   __ ___________                 _                        \n"
	"                   \\ \\ / //  ___| ___ \\               | |                       \n"
	"                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
	"                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
	"                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
	"                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n";
	border_print();
	center_print("Developed at Argonne National Laboratory", 79);
	char v[100];
	sprintf(v, "Version: %d", version);
	center_print(v, 79);
	border_print();
}

void center_print(const char *s, int width)
{
	int length = strlen(s);
	int i;
	for (i = 0; i <= (width - length) / 2; i++) {
		std::cout << " ";
	}
	std::cout << s << std::endl;
}

void border_print(void)
{
	std::cout << "==================================================================="
	"=============" << std::endl;
}

void fancy_int(long a)
{
    if( a < 1000 )
        std::cout << a << std::endl;
    else if( a >= 1000 && a < 1000000 )
        std::cout << a / 1000 << "," << std::setw(3) << std::setfill('0') << a % 1000 << std::endl;
    else if( a >= 1000000 && a < 1000000000 )
        std::cout << a / 1000000 << "," << std::setw(3) << std::setfill('0') << (a % 1000000) / 1000 << "," << std::setw(3) << std::setfill('0') << a % 1000 << std::endl;
    else if( a >= 1000000000 )
        std::cout << a / 1000000000 << "," << std::setw(3) << std::setfill('0') << (a % 1000000000) / 1000000 << "," << std::setw(3) << std::setfill('0') << (a % 1000000) / 1000 << "," << std::setw(3) << std::setfill('0') << a % 1000 << std::endl;
    else
        std::cout << a << std::endl;
}

void print_CLI_error(void)
{
	std::cout << "Usage: ./XSBench <options>\n";
	std::cout << "Options include:\n";
	std::cout << "  -m <simulation method>   Simulation method (history, event)\n";
	std::cout << "  -s <size>                Size of H-M Benchmark to run (small, large, XL, XXL)\n";
	std::cout << "  -g <gridpoints>          Number of gridpoints per nuclide (overrides -s defaults)\n";
	std::cout << "  -G <grid type>           Grid search type (unionized, nuclide, hash). Defaults to unionized.\n";
	std::cout << "  -p <particles>           Number of particle histories\n";
	std::cout << "  -l <lookups>             History Based: Number of Cross-section (XS) lookups per particle. Event Based: Total number of XS lookups.\n";
	std::cout << "  -h <hash bins>           Number of hash bins (only relevant when used with \"-G hash\")\n";
	std::cout << "  -b <binary mode>         Read or write all data structures to file. If reading, this will skip initialization phase. (read, write)\n";
	std::cout << "  -k <kernel ID>           Specifies which kernel to run. 0 is baseline, 1, 2, etc are optimized variants. (0 is default.)\n";
	std::cout << "Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized -k 0\n";
	std::cout << "See readme for full description of default run values\n";
	exit(4);
}

Inputs read_CLI(int argc, char *argv[])
{
	Inputs input;

	input.simulation_method = HISTORY_BASED;
	input.nthreads = 1;
	input.n_isotopes = 355;
	input.n_gridpoints = 11303;
	input.particles = 500000;
	input.lookups = 34;
	input.grid_type = UNIONIZED;
	input.hash_bins = 10000;
	input.binary_mode = NONE;
	input.kernel_id = 0;
	input.HM = (char *)malloc(6 * sizeof(char));
	strcpy(input.HM, "large");

	int user_g = 0;
	int default_lookups = 1;
	int default_particles = 1;

	for (int i = 1; i < argc; i++) {
		char *arg = argv[i];
		if (strcmp(arg, "-g") == 0) {
			if (++i < argc) {
				user_g = 1;
				input.n_gridpoints = atol(argv[i]);
			} else
				print_CLI_error();
		} else if (strcmp(arg, "-m") == 0) {
			char *sim_type;
			if (++i < argc)
				sim_type = argv[i];
			else
				print_CLI_error();

			if (strcmp(sim_type, "history") == 0)
				input.simulation_method = HISTORY_BASED;
			else if (strcmp(sim_type, "event") == 0) {
				input.simulation_method = EVENT_BASED;
				if (default_lookups && default_particles) {
					input.lookups = input.lookups * input.particles;
					input.particles = 0;
				}
			} else
				print_CLI_error();
		} else if (strcmp(arg, "-l") == 0) {
			if (++i < argc) {
				input.lookups = atoi(argv[i]);
				default_lookups = 0;
			} else
				print_CLI_error();
		} else if (strcmp(arg, "-h") == 0) {
			if (++i < argc)
				input.hash_bins = atoi(argv[i]);
			else
				print_CLI_error();
		} else if (strcmp(arg, "-p") == 0) {
			if (++i < argc) {
				input.particles = atoi(argv[i]);
				default_particles = 0;
			} else
				print_CLI_error();
		} else if (strcmp(arg, "-s") == 0) {
			if (++i < argc)
				input.HM = argv[i];
			else
				print_CLI_error();
		} else if (strcmp(arg, "-G") == 0) {
			char *grid_type;
			if (++i < argc)
				grid_type = argv[i];
			else
				print_CLI_error();

			if (strcmp(grid_type, "unionized") == 0)
				input.grid_type = UNIONIZED;
			else if (strcmp(grid_type, "nuclide") == 0)
				input.grid_type = NUCLIDE;
			else if (strcmp(grid_type, "hash") == 0)
				input.grid_type = HASH;
			else
				print_CLI_error();
		} else if (strcmp(arg, "-b") == 0) {
			char *binary_mode;
			if (++i < argc)
				binary_mode = argv[i];
			else
				print_CLI_error();

			if (strcmp(binary_mode, "read") == 0)
				input.binary_mode = READ;
			else if (strcmp(binary_mode, "write") == 0)
				input.binary_mode = WRITE;
			else
				print_CLI_error();
		} else if (strcmp(arg, "-k") == 0) {
			if (++i < argc) {
				input.kernel_id = atoi(argv[i]);
			} else
				print_CLI_error();
		} else
			print_CLI_error();
	}

	if (input.nthreads < 1)
		print_CLI_error();
	if (input.n_isotopes < 1)
		print_CLI_error();
	if (input.n_gridpoints < 1)
		print_CLI_error();
	if (input.lookups < 1)
		print_CLI_error();
	if (input.hash_bins < 1)
		print_CLI_error();
	if (strcasecmp(input.HM, "small") != 0 &&
		strcasecmp(input.HM, "large") != 0 &&
		strcasecmp(input.HM, "XL") != 0 &&
		strcasecmp(input.HM, "XXL") != 0)
		print_CLI_error();

	if (strcasecmp(input.HM, "small") == 0)
		input.n_isotopes = 68;
	else if (strcasecmp(input.HM, "XL") == 0 && user_g == 0)
		input.n_gridpoints = 238847;
	else if (strcasecmp(input.HM, "XXL") == 0 && user_g == 0)
		input.n_gridpoints = 238847 * 2.1;

	return input;
}

void print_inputs(Inputs in, int nprocs, int version)
{
	int mem_tot = estimate_mem_usage(in);
	logo(version);
	center_print("INPUT SUMMARY", 79);
	border_print();
	std::cout << "Programming Model:            SYCL\n";
	std::cout << "SYCL Device:                  ";
	auto dev = device::get_devices()[0];
	std::cout << dev.get_info<info::device::name>() << std::endl;
	if (in.simulation_method == EVENT_BASED)
		std::cout << "Simulation Method:            Event Based\n";
	else
		std::cout << "Simulation Method:            History Based\n";
	if (in.grid_type == NUCLIDE)
		std::cout << "Grid Type:                    Nuclide Grid\n";
	else if (in.grid_type == UNIONIZED)
		std::cout << "Grid Type:                    Unionized Grid\n";
	else
		std::cout << "Grid Type:                    Hash\n";
	std::cout << "Materials:                    12\n";
	std::cout << "H-M Benchmark Size:           " << in.HM << std::endl;
	std::cout << "Total Nuclides:               " << in.n_isotopes << std::endl;
	std::cout << "Gridpoints (per Nuclide):     "; fancy_int(in.n_gridpoints);
	if (in.grid_type == HASH) {
		std::cout << "Hash Bins:                    "; fancy_int(in.hash_bins);
	}
	if (in.grid_type == UNIONIZED) {
		std::cout << "Unionized Energy Gridpoints:  "; fancy_int(in.n_isotopes * in.n_gridpoints);
	}
	if (in.simulation_method == HISTORY_BASED) {
		std::cout << "Particle Histories:           "; fancy_int(in.particles);
		std::cout << "XS Lookups per Particle:      "; fancy_int(in.lookups);
	}
	std::cout << "Total XS Lookups:             "; fancy_int(in.lookups);
	std::cout << "Est. Memory Usage (MB):       "; fancy_int(mem_tot);
	std::cout << "Binary File Mode:             ";
	if (in.binary_mode == NONE)
		std::cout << "Off\n";
	else if (in.binary_mode == READ)
		std::cout << "Read\n";
	else
		std::cout << "Write\n";
	border_print();
	center_print("INITIALIZATION - DO NOT PROFILE", 79);
	border_print();
}

int print_results(Inputs in, int mype, double runtime, int nprocs,
	unsigned long long vhash)
{
	int lookups = 0;
	if (in.simulation_method == HISTORY_BASED)
		lookups = in.lookups * in.particles;
	else if (in.simulation_method == EVENT_BASED)
		lookups = in.lookups;
	int lookups_per_sec = (int)((double)lookups / runtime);

	int is_invalid_result = 1;

	if (mype == 0) {
		border_print();
		center_print("RESULTS", 79);
		border_print();
		std::cout << "NOTE: Timings are estimated -- use profilers for formal analysis\n";
		std::cout << "Runtime:     " << runtime << " seconds\n";
		std::cout << "Lookups:     "; fancy_int(lookups);
		std::cout << "Lookups/s:   "; fancy_int(lookups_per_sec);
	}

	unsigned long long large = 0;
	unsigned long long small = 0;
	if (in.simulation_method == EVENT_BASED) {
		small = 945990;
		large = 952131;
	} else if (in.simulation_method == HISTORY_BASED) {
		small = 941535;
		large = 954318;
	}
	if (strcmp(in.HM, "large") == 0) {
		if (vhash == large)
			is_invalid_result = 0;
	} else if (strcmp(in.HM, "small") == 0) {
		if (vhash == small)
			is_invalid_result = 0;
	}

	if (mype == 0) {
		if (is_invalid_result)
			std::cout << "Verification checksum: " << vhash << " (WARNING - INAVALID CHECKSUM!)\n";
		else
			std::cout << "Verification checksum: " << vhash << " (Valid)\n";
		border_print();
	}

	return is_invalid_result;
}

void binary_write(Inputs in, SimulationData SD)
{
	const char *fname = "XS_data.dat";
	std::cout << "Writing all data structures to binary file " << fname << "...\n";
	std::ofstream fp(fname, std::ios::binary);

	fp.write((char *)&SD, sizeof(SimulationData));
	fp.write((char *)SD.num_nucs, sizeof(int) * SD.length_num_nucs);
	fp.write((char *)SD.concs, sizeof(double) * SD.length_concs);
	fp.write((char *)SD.mats, sizeof(int) * SD.length_mats);
	fp.write((char *)SD.nuclide_grid, sizeof(NuclideGridPoint) * SD.length_nuclide_grid);
	fp.write((char *)SD.index_grid, sizeof(int) * SD.length_index_grid);
	fp.write((char *)SD.unionized_energy_array, sizeof(double) * SD.length_unionized_energy_array);

	fp.close();
}

SimulationData binary_read(Inputs in)
{
	SimulationData SD;
	const char *fname = "XS_data.dat";
	std::cout << "Reading all data structures from binary file " << fname << "...\n";
	std::ifstream fp(fname, std::ios::binary);
	assert(fp.is_open());

	fp.read((char *)&SD, sizeof(SimulationData));
	SD.num_nucs = (int *)malloc(SD.length_num_nucs * sizeof(int));
	SD.concs = (double *)malloc(SD.length_concs * sizeof(double));
	SD.mats = (int *)malloc(SD.length_mats * sizeof(int));
	SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
	SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));

	fp.read((char *)SD.num_nucs, sizeof(int) * SD.length_num_nucs);
	fp.read((char *)SD.concs, sizeof(double) * SD.length_concs);
	fp.read((char *)SD.mats, sizeof(int) * SD.length_mats);
	fp.read((char *)SD.nuclide_grid, sizeof(NuclideGridPoint) * SD.length_nuclide_grid);
	fp.read((char *)SD.index_grid, sizeof(int) * SD.length_index_grid);
	fp.read((char *)SD.unionized_energy_array, sizeof(double) * SD.length_unionized_energy_array);

	fp.close();

	return SD;
}

SimulationData grid_init_do_not_profile(Inputs in, int mype)
{
	SimulationData SD;
	size_t nbytes = 0;
	uint64_t seed = 42;

	SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
	SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	assert(SD.nuclide_grid != NULL);
	nbytes += SD.length_nuclide_grid * sizeof(NuclideGridPoint);
	for (int i = 0; i < SD.length_nuclide_grid; i++) {
		SD.nuclide_grid[i].energy = LCG_random_double(&seed);
		SD.nuclide_grid[i].total_xs = LCG_random_double(&seed);
		SD.nuclide_grid[i].elastic_xs = LCG_random_double(&seed);
		SD.nuclide_grid[i].absorbtion_xs = LCG_random_double(&seed);
		SD.nuclide_grid[i].fission_xs = LCG_random_double(&seed);
		SD.nuclide_grid[i].nu_fission_xs = LCG_random_double(&seed);
	}
	for (int i = 0; i < in.n_isotopes; i++)
		std::sort(&SD.nuclide_grid[i * in.n_gridpoints], &SD.nuclide_grid[(i + 1) * in.n_gridpoints], [](const NuclideGridPoint &a, const NuclideGridPoint &b) { return a.energy < b.energy; });

	if (in.grid_type == NUCLIDE) {
		SD.length_unionized_energy_array = 0;
		SD.length_index_grid = 0;
	}
	if (in.grid_type == UNIONIZED) {
		SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
		SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));
		assert(SD.unionized_energy_array != NULL);
		nbytes += SD.length_unionized_energy_array * sizeof(double);
		for (int i = 0; i < SD.length_unionized_energy_array; i++)
			SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
		std::sort(SD.unionized_energy_array, SD.unionized_energy_array + SD.length_unionized_energy_array);
		SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
		SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
		assert(SD.index_grid != NULL);
		nbytes += SD.length_index_grid * sizeof(int);
		int *idx_low = (int *)calloc(in.n_isotopes, sizeof(int));
		assert(idx_low != NULL);
		double *energy_high = (double *)malloc(in.n_isotopes * sizeof(double));
		assert(energy_high != NULL);
		for (int i = 0; i < in.n_isotopes; i++)
			energy_high[i] = SD.nuclide_grid[i * in.n_gridpoints + 1].energy;
		for (long e = 0; e < SD.length_unionized_energy_array; e++) {
			double unionized_energy = SD.unionized_energy_array[e];
			for (long i = 0; i < in.n_isotopes; i++) {
				if (unionized_energy < energy_high[i])
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
		SD.length_index_grid = in.hash_bins * in.n_isotopes;
		SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
		assert(SD.index_grid != NULL);
		nbytes += SD.length_index_grid * sizeof(int);
		double du = 1.0 / in.hash_bins;
		for (long e = 0; e < in.hash_bins; e++) {
			double energy = e * du;
			for (long i = 0; i < in.n_isotopes; i++) {
				SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide(in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints - 1);
			}
		}
	}
	SD.num_nucs = load_num_nucs(in.n_isotopes);
	SD.length_num_nucs = 12;
	SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
	SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
	SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
	SD.length_concs = SD.length_mats;
	return SD;
}

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD)
{
	SimulationData GSD = SD;
	GSD.num_nucs = SD.num_nucs;
	GSD.concs = SD.concs;
	GSD.mats = SD.mats;
	GSD.unionized_energy_array = SD.unionized_energy_array;
	GSD.index_grid = SD.index_grid;
	GSD.nuclide_grid = SD.nuclide_grid;
	GSD.verification = (unsigned long *)malloc(in.lookups * sizeof(unsigned long));
	GSD.length_verification = in.lookups;
	GSD.p_energy_samples = nullptr;
	GSD.mat_samples = nullptr;
	GSD.length_p_energy_samples = 0;
	GSD.length_mat_samples = 0;
	return GSD;
}

void release_device_memory(SimulationData GSD)
{
	free(GSD.num_nucs);
	free(GSD.concs);
	free(GSD.mats);
	free(GSD.unionized_energy_array);
	free(GSD.nuclide_grid);
	free(GSD.verification);
	if (GSD.p_energy_samples) free(GSD.p_energy_samples);
	if (GSD.mat_samples) free(GSD.mat_samples);
}

int *load_num_nucs(long n_isotopes)
{
	int *num_nucs = (int *)malloc(12 * sizeof(int));
	if (n_isotopes == 68)
		num_nucs[0] = 34;
	else
		num_nucs[0] = 321;
	num_nucs[1] = 5;
	num_nucs[2] = 4;
	num_nucs[3] = 4;
	num_nucs[4] = 27;
	num_nucs[5] = 21;
	num_nucs[6] = 21;
	num_nucs[7] = 21;
	num_nucs[8] = 21;
	num_nucs[9] = 21;
	num_nucs[10] = 9;
	num_nucs[11] = 9;
	return num_nucs;
}

int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs)
{
	*max_num_nucs = 0;
	int num_mats = 12;
	for (int m = 0; m < num_mats; m++) {
		if (num_nucs[m] > *max_num_nucs)
			*max_num_nucs = num_nucs[m];
	}
	int *mats = (int *)malloc(num_mats * (*max_num_nucs) * sizeof(int));
	int mats0_Sml[] = {58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7, 8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53, 54, 55, 56, 18, 23, 41};
	int mats0_Lrg[321] = {58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7, 8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53, 54, 55, 56, 18, 23, 41};
	for (int i = 0; i < 321 - 34; i++)
		mats0_Lrg[34 + i] = 68 + i;
	int mats1[] = {63, 64, 65, 66, 67};
	int mats2[] = {24, 41, 4, 5};
	int mats3[] = {24, 41, 4, 5};
	int mats4[] = {19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29, 30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16, 17};
	int mats5[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
	int mats6[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
	int mats7[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
	int mats8[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
	int mats9[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
	int mats10[] = {24, 41, 4, 5, 63, 64, 65, 66, 67};
	int mats11[] = {24, 41, 4, 5, 63, 64, 65, 66, 67};
	if (n_isotopes == 68)
		memcpy(mats, mats0_Sml, num_nucs[0] * sizeof(int));
	else
		memcpy(mats, mats0_Lrg, num_nucs[0] * sizeof(int));
	memcpy(mats + *max_num_nucs * 1, mats1, num_nucs[1] * sizeof(int));
	memcpy(mats + *max_num_nucs * 2, mats2, num_nucs[2] * sizeof(int));
	memcpy(mats + *max_num_nucs * 3, mats3, num_nucs[3] * sizeof(int));
	memcpy(mats + *max_num_nucs * 4, mats4, num_nucs[4] * sizeof(int));
	memcpy(mats + *max_num_nucs * 5, mats5, num_nucs[5] * sizeof(int));
	memcpy(mats + *max_num_nucs * 6, mats6, num_nucs[6] * sizeof(int));
	memcpy(mats + *max_num_nucs * 7, mats7, num_nucs[7] * sizeof(int));
	memcpy(mats + *max_num_nucs * 8, mats8, num_nucs[8] * sizeof(int));
	memcpy(mats + *max_num_nucs * 9, mats9, num_nucs[9] * sizeof(int));
	memcpy(mats + *max_num_nucs * 10, mats10, num_nucs[10] * sizeof(int));
	memcpy(mats + *max_num_nucs * 11, mats11, num_nucs[11] * sizeof(int));
	return mats;
}

double *load_concs(int *num_nucs, int max_num_nucs)
{
	uint64_t seed = STARTING_SEED * STARTING_SEED;
	double *concs = (double *)malloc(12 * max_num_nucs * sizeof(double));
	for (int i = 0; i < 12; i++)
		for (int j = 0; j < num_nucs[i]; j++)
			concs[i * max_num_nucs + j] = LCG_random_double(&seed);
	return concs;
}

int double_compare(const void *a, const void *b)
{
	double A = *((double *)a);
	double B = *((double *)b);
	if (A > B)
		return 1;
	else if (A < B)
		return -1;
	else
		return 0;
}

int NGP_compare(const void *a, const void *b)
{
	NuclideGridPoint A = *((NuclideGridPoint *)a);
	NuclideGridPoint B = *((NuclideGridPoint *)b);
	if (A.energy > B.energy)
		return 1;
	else if (A.energy < B.energy)
		return -1;
	else
		return 0;
}

double rn_v(void)
{
	static unsigned long seed = 1337;
	double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = (a * (seed)) % m;
	seed = n1;
	ret = (double)n1 / m;
	return ret;
}

size_t estimate_mem_usage(Inputs in)
{
	size_t single_nuclide_grid = in.n_gridpoints * sizeof(NuclideGridPoint);
	size_t all_nuclide_grids = in.n_isotopes * single_nuclide_grid;
	size_t size_UEG = in.n_isotopes * in.n_gridpoints * sizeof(double) + in.n_isotopes * in.n_gridpoints * in.n_isotopes * sizeof(int);
	size_t size_hash_grid = in.hash_bins * in.n_isotopes * sizeof(int);
	size_t memtotal;
	if (in.grid_type == UNIONIZED)
		memtotal = all_nuclide_grids + size_UEG;
	else if (in.grid_type == NUCLIDE)
		memtotal = all_nuclide_grids;
	else
		memtotal = all_nuclide_grids + size_hash_grid;
	memtotal = ceil(memtotal / (1024.0 * 1024.0));
	return memtotal;
}

double get_time(void)
{
	unsigned long us_since_epoch = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000;
	return (double)us_since_epoch / 1.0e6;
}

// RNG
double LCG_random_double(uint64_t *seed)
{
	const uint64_t m = 9223372036854775808ULL;
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double)(*seed) / (double)m;
}

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
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

// Device-side helpers
long grid_search(long n, double quarry, double *A)
{
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

long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high)
{
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

int pick_mat(uint64_t *seed)
{
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

void calculate_micro_xs(double p_energy, int nuc, long n_isotopes,
                        long n_gridpoints, double *egrid, int *index_data,
                        NuclideGridPoint *nuclide_grids, long idx, double *xs_vector,
                        int grid_type, int hash_bins)
{
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

void calculate_macro_xs(double p_energy, int mat, long n_isotopes,
                        long n_gridpoints, int *num_nucs, double *concs,
                        double *egrid, int *index_data, NuclideGridPoint *nuclide_grids,
                        int *mats, double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs)
{
	int p_nuc;
	long idx = -1;
	double conc;
	for (int k = 0; k < 5; k++)
		macro_xs_vector[k] = 0;
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
		calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids, idx, xs_vector, grid_type, hash_bins);
		for (int k = 0; k < 5; k++)
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}

// SYCL kernel wrappers

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Running baseline event-based simulation...\n";
	std::vector<unsigned long> verification(in.lookups, 0);
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				p_energy, mat, in.n_isotopes, in.n_gridpoints,
				GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
				GSD.index_grid, GSD.nuclide_grid, GSD.mats,
				macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}

// Optimization 1: Basic kernel splitting of sampling & lookup routines
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Simulation Kernel:\"Optimization 1 - basic sample/lookup kernel splitting\"\n";
	GSD.p_energy_samples = (double *)malloc(in.lookups * sizeof(double));
	GSD.length_p_energy_samples = in.lookups;
	GSD.mat_samples = (int *)malloc(in.lookups * sizeof(int));
	GSD.length_mat_samples = in.lookups;
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints,
				GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
				GSD.index_grid, GSD.nuclide_grid, GSD.mats,
				macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}

// Optimization 2: Material-specific kernels
unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Simulation Kernel:\"Optimization 2 - Material Lookup Kernels\"\n";
	GSD.p_energy_samples = (double *)malloc(in.lookups * sizeof(double));
	GSD.length_p_energy_samples = in.lookups;
	GSD.mat_samples = (int *)malloc(in.lookups * sizeof(int));
	GSD.length_mat_samples = in.lookups;
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();
	for (int m = 0; m < 12; m++) {
		q.submit([&](handler &h) {
			h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
				int i = idx[0];
				int mat = GSD.mat_samples[i];
				if (mat != m) return;
				double macro_xs_vector[5] = {0};
				calculate_macro_xs(
					GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints,
					GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
					GSD.index_grid, GSD.nuclide_grid, GSD.mats,
					macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
	}
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}

// Optimization 3: Fuel/non-fuel kernels
unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Simulation Kernel:\"Optimization 3 - Fuel or Other Lookup Kernels\"\n";
	GSD.p_energy_samples = (double *)malloc(in.lookups * sizeof(double));
	GSD.length_p_energy_samples = in.lookups;
	GSD.mat_samples = (int *)malloc(in.lookups * sizeof(int));
	GSD.length_mat_samples = in.lookups;
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();
	for (int is_fuel = 0; is_fuel <= 1; is_fuel++) {
		q.submit([&](handler &h) {
			h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
				int i = idx[0];
				int mat = GSD.mat_samples[i];
				if (((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0))) {
					double macro_xs_vector[5] = {0};
					calculate_macro_xs(
						GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints,
						GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
						GSD.index_grid, GSD.nuclide_grid, GSD.mats,
						macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
				}
			});
		}).wait();
	}
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}

// Optimization 4/5/6: Material/energy sort + kernels
unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Simulation Kernel:\"Optimization 4 - All Material Lookup Kernels + Material Sort\"\n";
	GSD.p_energy_samples = (double *)malloc(in.lookups * sizeof(double));
	GSD.length_p_energy_samples = in.lookups;
	GSD.mat_samples = (int *)malloc(in.lookups * sizeof(int));
	GSD.length_mat_samples = in.lookups;
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();
	std::vector<std::pair<int, double>> mat_energy_pairs(in.lookups);
	for (int i = 0; i < in.lookups; ++i)
		mat_energy_pairs[i] = std::make_pair(GSD.mat_samples[i], GSD.p_energy_samples[i]);
	std::sort(mat_energy_pairs.begin(), mat_energy_pairs.end());
	for (int i = 0; i < in.lookups; ++i) {
		GSD.mat_samples[i] = mat_energy_pairs[i].first;
		GSD.p_energy_samples[i] = mat_energy_pairs[i].second;
	}
	int n_lookups_per_material[12] = {0};
	for (int i = 0; i < in.lookups; ++i)
		n_lookups_per_material[GSD.mat_samples[i]]++;
	int offset = 0;
	for (int m = 0; m < 12; m++) {
		q.submit([&](handler &h) {
			h.parallel_for(range<1>(n_lookups_per_material[m]), [=](id<1> idx) {
				int i = idx[0] + offset;
				int mat = GSD.mat_samples[i];
				if (mat != m) return;
				double macro_xs_vector[5] = {0};
				calculate_macro_xs(
					GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints,
					GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
					GSD.index_grid, GSD.nuclide_grid, GSD.mats,
					macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
		offset += n_lookups_per_material[m];
	}
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Simulation Kernel:\"Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort\"\n";
	GSD.p_energy_samples = (double *)malloc(in.lookups * sizeof(double));
	GSD.length_p_energy_samples = in.lookups;
	GSD.mat_samples = (int *)malloc(in.lookups * sizeof(int));
	GSD.length_mat_samples = in.lookups;
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();
	std::vector<std::pair<int, double>> mat_energy_pairs(in.lookups);
	for (int i = 0; i < in.lookups; ++i)
		mat_energy_pairs[i] = std::make_pair(GSD.mat_samples[i], GSD.p_energy_samples[i]);
	std::stable_sort(mat_energy_pairs.begin(), mat_energy_pairs.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) { return a.first == 0 && b.first != 0; });
	for (int i = 0; i < in.lookups; ++i) {
		GSD.mat_samples[i] = mat_energy_pairs[i].first;
		GSD.p_energy_samples[i] = mat_energy_pairs[i].second;
	}
	int n_fuel_lookups = std::count(GSD.mat_samples, GSD.mat_samples + in.lookups, 0);
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(n_fuel_lookups), [=](id<1> idx) {
			int i = idx[0];
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints,
				GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
				GSD.index_grid, GSD.nuclide_grid, GSD.mats,
				macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups - n_fuel_lookups), [=](id<1> idx) {
			int i = idx[0] + n_fuel_lookups;
			double macro_xs_vector[5] = {0};
			calculate_macro_xs(
				GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints,
				GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
				GSD.index_grid, GSD.nuclide_grid, GSD.mats,
				macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype, queue &q)
{
	if (mype == 0) std::cout << "Simulation Kernel:\"Optimization 6 - Material & Energy Sorts + Material-specific Kernels\"\n";
	GSD.p_energy_samples = (double *)malloc(in.lookups * sizeof(double));
	GSD.length_p_energy_samples = in.lookups;
	GSD.mat_samples = (int *)malloc(in.lookups * sizeof(int));
	GSD.length_mat_samples = in.lookups;
	q.submit([&](handler &h) {
		h.parallel_for(range<1>(in.lookups), [=](id<1> idx) {
			int i = idx[0];
			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2 * i);
			double p_energy = LCG_random_double(&seed);
			int mat = pick_mat(&seed);
			GSD.p_energy_samples[i] = p_energy;
			GSD.mat_samples[i] = mat;
		});
	}).wait();
	std::vector<std::pair<int, double>> mat_energy_pairs(in.lookups);
	for (int i = 0; i < in.lookups; ++i)
		mat_energy_pairs[i] = std::make_pair(GSD.mat_samples[i], GSD.p_energy_samples[i]);
	std::sort(mat_energy_pairs.begin(), mat_energy_pairs.end());
	for (int i = 0; i < in.lookups; ++i) {
		GSD.mat_samples[i] = mat_energy_pairs[i].first;
		GSD.p_energy_samples[i] = mat_energy_pairs[i].second;
	}
	int n_lookups_per_material[12] = {0};
	for (int i = 0; i < in.lookups; ++i)
		n_lookups_per_material[GSD.mat_samples[i]]++;
	int offset = 0;
	for (int m = 0; m < 12; m++) {
		std::vector<double> energies;
		std::vector<int> mats;
		for (int i = offset; i < offset + n_lookups_per_material[m]; ++i) {
			energies.push_back(GSD.p_energy_samples[i]);
			mats.push_back(GSD.mat_samples[i]);
		}
		std::vector<std::pair<double, int>> energy_mat_pairs(n_lookups_per_material[m]);
		for (int i = 0; i < n_lookups_per_material[m]; ++i)
			energy_mat_pairs[i] = std::make_pair(energies[i], mats[i]);
		std::sort(energy_mat_pairs.begin(), energy_mat_pairs.end());
		for (int i = 0; i < n_lookups_per_material[m]; ++i) {
			GSD.p_energy_samples[offset + i] = energy_mat_pairs[i].first;
			GSD.mat_samples[offset + i] = energy_mat_pairs[i].second;
		}
		q.submit([&](handler &h) {
			h.parallel_for(range<1>(n_lookups_per_material[m]), [=](id<1> idx) {
				int i = idx[0] + offset;
				int mat = GSD.mat_samples[i];
				if (mat != m) return;
				double macro_xs_vector[5] = {0};
				calculate_macro_xs(
					GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints,
					GSD.num_nucs, GSD.concs, GSD.unionized_energy_array,
					GSD.index_grid, GSD.nuclide_grid, GSD.mats,
					macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs
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
		offset += n_lookups_per_material[m];
	}
	unsigned long verification_scalar = std::accumulate(GSD.verification, GSD.verification + in.lookups, 0UL);
	return verification_scalar;
}
