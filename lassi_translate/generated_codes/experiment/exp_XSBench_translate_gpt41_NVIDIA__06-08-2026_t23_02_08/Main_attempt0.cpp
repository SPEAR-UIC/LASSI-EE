#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <memory>

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
    std::string HM;
    int grid_type;
    int hash_bins;
    int particles;
    int simulation_method;
    int binary_mode;
    int kernel_id;
};

struct SimulationData {
    std::vector<int> num_nucs;
    std::vector<double> concs;
    std::vector<int> mats;
    std::vector<double> unionized_energy_array;
    std::vector<int> index_grid;
    std::vector<NuclideGridPoint> nuclide_grid;
    int length_num_nucs;
    int length_concs;
    int length_mats;
    int length_unionized_energy_array;
    long length_index_grid;
    int length_nuclide_grid;
    int max_num_nucs;
    std::vector<unsigned long> verification;
    int length_verification;
    std::vector<double> p_energy_samples;
    int length_p_energy_samples;
    std::vector<int> mat_samples;
    int length_mat_samples;
};

// Utility functions
void border_print() {
    std::cout << "==================================================================="
                 "=============" << std::endl;
}

void center_print(const char *s, int width) {
    int length = strlen(s);
    int i;
    for (i = 0; i <= (width - length) / 2; i++) {
        std::cout << " ";
    }
    std::cout << s << std::endl;
}

void fancy_int(long a) {
    if (a < 1000)
        std::cout << a << std::endl;
    else if (a >= 1000 && a < 1000000)
        std::cout << a / 1000 << "," << std::setw(3) << std::setfill('0') << a % 1000 << std::endl;
    else if (a >= 1000000 && a < 1000000000)
        std::cout << a / 1000000 << "," << std::setw(3) << std::setfill('0') << (a % 1000000) / 1000 << "," << std::setw(3) << std::setfill('0') << a % 1000 << std::endl;
    else if (a >= 1000000000)
        std::cout << a / 1000000000 << "," << std::setw(3) << std::setfill('0') << (a % 1000000000) / 1000000 << "," << std::setw(3) << std::setfill('0') << (a % 1000000) / 1000 << "," << std::setw(3) << std::setfill('0') << a % 1000 << std::endl;
    else
        std::cout << a << std::endl;
}

void logo(int version) {
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

void print_CLI_error() {
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

Inputs read_CLI(int argc, char *argv[]) {
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
    input.HM = "large";
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
    if (strcasecmp(input.HM.c_str(), "small") != 0 &&
        strcasecmp(input.HM.c_str(), "large") != 0 &&
        strcasecmp(input.HM.c_str(), "XL") != 0 &&
        strcasecmp(input.HM.c_str(), "XXL") != 0)
        print_CLI_error();
    if (strcasecmp(input.HM.c_str(), "small") == 0)
        input.n_isotopes = 68;
    else if (strcasecmp(input.HM.c_str(), "XL") == 0 && user_g == 0)
        input.n_gridpoints = 238847;
    else if (strcasecmp(input.HM.c_str(), "XXL") == 0 && user_g == 0)
        input.n_gridpoints = static_cast<long>(238847 * 2.1);
    return input;
}

size_t estimate_mem_usage(Inputs in) {
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
    memtotal = static_cast<size_t>(std::ceil(memtotal / (1024.0 * 1024.0)));
    return memtotal;
}

void print_inputs(Inputs in, int nprocs, int version) {
    int mem_tot = estimate_mem_usage(in);
    logo(version);
    center_print("INPUT SUMMARY", 79);
    border_print();
    std::cout << "Programming Model:            SYCL\n";
    std::cout << "SYCL Device:                  ";
    try {
        auto devices = sycl::device::get_devices();
        for (auto &dev : devices) {
            if (dev.is_gpu()) {
                std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
                break;
            }
        }
    } catch (...) {
        std::cout << "Unknown SYCL device\n";
    }
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
    std::cout << "Materials:                    " << 12 << std::endl;
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

double get_time() {
    auto us_since_epoch = std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::microseconds(1);
    return static_cast<double>(us_since_epoch) / 1.0e6;
}

int double_compare(const void *a, const void *b) {
    double A = *((double *)a);
    double B = *((double *)b);
    if (A > B)
        return 1;
    else if (A < B)
        return -1;
    else
        return 0;
}

int NGP_compare(const void *a, const void *b) {
    NuclideGridPoint A = *((NuclideGridPoint *)a);
    NuclideGridPoint B = *((NuclideGridPoint *)b);
    if (A.energy > B.energy)
        return 1;
    else if (A.energy < B.energy)
        return -1;
    else
        return 0;
}

double rn_v() {
    static unsigned long seed = 1337;
    double ret;
    unsigned long n1;
    unsigned long a = 16807;
    unsigned long m = 2147483647;
    n1 = (a * (seed)) % m;
    seed = n1;
    ret = static_cast<double>(n1) / m;
    return ret;
}

int *load_num_nucs(long n_isotopes) {
    int *num_nucs = new int[12];
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

int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs) {
    *max_num_nucs = 0;
    int num_mats = 12;
    for (int m = 0; m < num_mats; m++) {
        if (num_nucs[m] > *max_num_nucs)
            *max_num_nucs = num_nucs[m];
    }
    int *mats = new int[num_mats * (*max_num_nucs)];
    int mats0_Sml[] = {58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
                       8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53,
                       54, 55, 56, 18, 23, 41};
    int mats0_Lrg[321] = {58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
                          8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53,
                          54, 55, 56, 18, 23, 41};
    for (int i = 0; i < 321 - 34; i++)
        mats0_Lrg[34 + i] = 68 + i;
    int mats1[] = {63, 64, 65, 66, 67};
    int mats2[] = {24, 41, 4, 5};
    int mats3[] = {24, 41, 4, 5};
    int mats4[] = {19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29,
                   30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16,
                   17};
    int mats5[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
                   49, 50, 51, 11, 12, 13, 14};
    int mats6[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
                   49, 50, 51, 11, 12, 13, 14};
    int mats7[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
                   49, 50, 51, 11, 12, 13, 14};
    int mats8[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
                   49, 50, 51, 11, 12, 13, 14};
    int mats9[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
                   49, 50, 51, 11, 12, 13, 14};
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

double *load_concs(int *num_nucs, int max_num_nucs) {
    uint64_t seed = STARTING_SEED * STARTING_SEED;
    double *concs = new double[12 * max_num_nucs];
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < num_nucs[i]; j++)
            concs[i * max_num_nucs + j] = ((double)(seed = (2806196910506780709ULL * seed + 1ULL) % 9223372036854775808ULL)) / 9223372036854775808ULL;
    return concs;
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

void calculate_micro_xs(double p_energy, int nuc, long n_isotopes,
                        long n_gridpoints,
                        double *egrid, int *index_data,
                        NuclideGridPoint *nuclide_grids,
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

void calculate_macro_xs(double p_energy, int mat, long n_isotopes,
                        long n_gridpoints, int *num_nucs,
                        double *concs,
                        double *egrid, int *index_data,
                        NuclideGridPoint *nuclide_grids,
                        int *mats,
                        double *macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs) {
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
        calculate_micro_xs(p_energy, p_nuc, n_isotopes,
                           n_gridpoints, egrid, index_data,
                           nuclide_grids, idx, xs_vector, grid_type, hash_bins);
        for (int k = 0; k < 5; k++)
            macro_xs_vector[k] += xs_vector[k] * conc;
    }
}

SimulationData grid_init_do_not_profile(Inputs in, int mype) {
    SimulationData SD;
    size_t nbytes = 0;
    uint64_t seed = 42;
    SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
    SD.nuclide_grid.resize(SD.length_nuclide_grid);
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
        std::sort(SD.nuclide_grid.begin() + i * in.n_gridpoints, SD.nuclide_grid.begin() + (i + 1) * in.n_gridpoints,
                  [](const NuclideGridPoint &a, const NuclideGridPoint &b) { return a.energy < b.energy; });
    if (in.grid_type == NUCLIDE) {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = 0;
    }
    if (in.grid_type == UNIONIZED) {
        SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
        SD.unionized_energy_array.resize(SD.length_unionized_energy_array);
        nbytes += SD.length_unionized_energy_array * sizeof(double);
        for (int i = 0; i < SD.length_unionized_energy_array; i++)
            SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;
        std::sort(SD.unionized_energy_array.begin(), SD.unionized_energy_array.end());
        SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
        SD.index_grid.resize(SD.length_index_grid);
        nbytes += SD.length_index_grid * sizeof(int);
        std::vector<int> idx_low(in.n_isotopes, 0);
        std::vector<double> energy_high(in.n_isotopes);
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
    }
    if (in.grid_type == HASH) {
        SD.length_unionized_energy_array = 0;
        SD.length_index_grid = in.hash_bins * in.n_isotopes;
        SD.index_grid.resize(SD.length_index_grid);
        nbytes += SD.length_index_grid * sizeof(int);
        double du = 1.0 / in.hash_bins;
        for (long e = 0; e < in.hash_bins; e++) {
            double energy = e * du;
            for (long i = 0; i < in.n_isotopes; i++) {
                SD.index_grid[e * in.n_isotopes + i] = grid_search_nuclide(in.n_gridpoints, energy, SD.nuclide_grid.data() + i * in.n_gridpoints, 0, in.n_gridpoints - 1);
            }
        }
    }
    SD.num_nucs.resize(12);
    int *num_nucs_ptr = load_num_nucs(in.n_isotopes);
    for (int i = 0; i < 12; i++) SD.num_nucs[i] = num_nucs_ptr[i];
    delete[] num_nucs_ptr;
    SD.length_num_nucs = 12;
    int max_num_nucs;
    int *mats_ptr = load_mats(SD.num_nucs.data(), in.n_isotopes, &max_num_nucs);
    SD.max_num_nucs = max_num_nucs;
    SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;
    SD.mats.resize(SD.length_mats);
    for (int i = 0; i < SD.length_mats; i++) SD.mats[i] = mats_ptr[i];
    delete[] mats_ptr;
    double *concs_ptr = load_concs(SD.num_nucs.data(), SD.max_num_nucs);
    SD.length_concs = SD.length_mats;
    SD.concs.resize(SD.length_concs);
    for (int i = 0; i < SD.length_concs; i++) SD.concs[i] = concs_ptr[i];
    delete[] concs_ptr;
    return SD;
}

int print_results(Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash) {
    int lookups = 0;
    if (in.simulation_method == HISTORY_BASED)
        lookups = in.lookups * in.particles;
    else if (in.simulation_method == EVENT_BASED)
        lookups = in.lookups;
    int lookups_per_sec = static_cast<int>((double)lookups / runtime);
    int is_invalid_result = 1;
    if (mype == 0) {
        border_print();
        center_print("RESULTS", 79);
        border_print();
        std::cout << "NOTE: Timings are estimated -- use nsys/iprof/rocprof for formal analysis\n";
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
    if (in.HM == "large") {
        if (vhash == large)
            is_invalid_result = 0;
    } else if (in.HM == "small") {
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

// SYCL kernel for baseline event-based simulation
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData &SD, int mype, sycl::queue &q) {
    if (mype == 0) std::cout << "Running baseline event-based simulation...\n";
    std::vector<unsigned long> verification(in.lookups, 0);
    sycl::buffer<int> num_nucs_buf(SD.num_nucs.data(), SD.length_num_nucs);
    sycl::buffer<double> concs_buf(SD.concs.data(), SD.length_concs);
    sycl::buffer<int> mats_buf(SD.mats.data(), SD.length_mats);
    sycl::buffer<double> unionized_energy_array_buf(SD.unionized_energy_array.data(), SD.length_unionized_energy_array);
    sycl::buffer<int> index_grid_buf(SD.index_grid.data(), SD.length_index_grid);
    sycl::buffer<NuclideGridPoint> nuclide_grid_buf(SD.nuclide_grid.data(), SD.length_nuclide_grid);
    sycl::buffer<unsigned long> verification_buf(verification.data(), in.lookups);
    q.submit([&](sycl::handler &h) {
        auto num_nucs = num_nucs_buf.get_access<sycl::access::mode::read>(h);
        auto concs = concs_buf.get_access<sycl::access::mode::read>(h);
        auto mats = mats_buf.get_access<sycl::access::mode::read>(h);
        auto unionized_energy_array = unionized_energy_array_buf.get_access<sycl::access::mode::read>(h);
        auto index_grid = index_grid_buf.get_access<sycl::access::mode::read>(h);
        auto nuclide_grid = nuclide_grid_buf.get_access<sycl::access::mode::read>(h);
        auto verification = verification_buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(in.lookups), [=](sycl::id<1> idx) {
            int i = idx[0];
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2 * i);
            double p_energy = LCG_random_double(&seed);
            int mat = pick_mat(&seed);
            double macro_xs_vector[5] = {0};
            calculate_macro_xs(
                p_energy,
                mat,
                in.n_isotopes,
                in.n_gridpoints,
                num_nucs.get_pointer(),
                concs.get_pointer(),
                unionized_energy_array.get_pointer(),
                index_grid.get_pointer(),
                nuclide_grid.get_pointer(),
                mats.get_pointer(),
                macro_xs_vector,
                in.grid_type,
                in.hash_bins,
                SD.max_num_nucs
            );
            double max = -1.0;
            int max_idx = 0;
            for (int j = 0; j < 5; j++) {
                if (macro_xs_vector[j] > max) {
                    max = macro_xs_vector[j];
                    max_idx = j;
                }
            }
            verification[i] = max_idx + 1;
        });
    });
    q.wait();
    unsigned long verification_scalar = 0;
    {
        auto verification_acc = verification_buf.get_access<sycl::access::mode::read>();
        for (int i = 0; i < in.lookups; i++)
            verification_scalar += verification_acc[i];
    }
    return verification_scalar;
}

int main(int argc, char *argv[]) {
    int version = 20;
    int mype = 0;
    int nprocs = 1;
    unsigned long long verification;
    Inputs in = read_CLI(argc, argv);
    if (mype == 0)
        print_inputs(in, nprocs, version);
    SimulationData SD;
    if (in.binary_mode == READ) {
        // Not implemented for SYCL, fallback to grid_init
        SD = grid_init_do_not_profile(in, mype);
    } else {
        SD = grid_init_do_not_profile(in, mype);
    }
    if (in.binary_mode == WRITE && mype == 0) {
        // Not implemented for SYCL, fallback to no-op
    }
    sycl::queue q(sycl::default_selector{});
    if (mype == 0) {
        std::cout << "\n";
        border_print();
        center_print("SIMULATION", 79);
        border_print();
    }
    double omp_start = get_time();
    if (in.simulation_method == EVENT_BASED) {
        if (in.kernel_id == 0)
            verification = run_event_based_simulation_baseline(in, SD, mype, q);
        else {
            std::cout << "Error: Only baseline kernel implemented in SYCL translation!\n";
            exit(1);
        }
    } else {
        std::cout << "History-based simulation not implemented in SYCL code. Instead,\nuse the event-based method with \"-m event\" argument.\n";
        exit(1);
    }
    if (mype == 0) {
        std::cout << "\n";
        std::cout << "Simulation complete.\n";
    }
    double omp_end = get_time();
    verification = verification % 999983;
    int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);
    return is_invalid_result;
}
