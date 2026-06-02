// SYCL 2020 translation of the provided CUDA XSBench unified file (single TU)
// Compilable with AdaptiveCpp (acpp). Standard C++17 + core SYCL only.

#include <sycl/sycl.hpp>

#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#define STARTING_SEED 1070ULL

// Simulation method
enum SimulationMethod { HISTORY_BASED = 0, EVENT_BASED = 1 };
// Grid type
enum GridType { UNIONIZED = 0, NUCLIDE = 1, HASH = 2 };
// Binary mode
enum BinaryMode { NONE = 0, READ = 1, WRITE = 2 };

struct Inputs {
  int simulation_method;
  int nthreads;
  long n_isotopes;
  long n_gridpoints;
  int particles;
  int lookups;
  int grid_type;
  int hash_bins;
  int binary_mode;
  int kernel_id;
  char* HM;
};

struct NuclideGridPoint {
  double energy;
  double total_xs;
  double elastic_xs;
  double absorbtion_xs;
  double fission_xs;
  double nu_fission_xs;
};

struct SimulationData {
  int* num_nucs = nullptr;
  double* concs = nullptr;
  int* mats = nullptr;
  double* unionized_energy_array = nullptr;
  int* index_grid = nullptr;
  NuclideGridPoint* nuclide_grid = nullptr;

  // verification output per lookup
  unsigned long* verification = nullptr;

  // optimized kernels
  double* p_energy_samples = nullptr;
  int* mat_samples = nullptr;

  // lengths (number of elements)
  long length_num_nucs = 0;
  long length_concs = 0;
  long length_mats = 0;
  long length_unionized_energy_array = 0;
  long length_index_grid = 0;
  long length_nuclide_grid = 0;
  long length_verification = 0;
  long length_p_energy_samples = 0;
  long length_mat_samples = 0;

  int max_num_nucs = 0;
};

// Global SYCL queue
static sycl::queue& get_queue() {
  static sycl::queue q{sycl::default_selector_v};
  return q;
}

// ============================================================================
// Utility printing and CLI
// ============================================================================
void border_print(void) {
  printf("===============================================================================\n");
}

void center_print(const char* s, int width) {
  int length = (int)std::strlen(s);
  for (int i = 0; i <= (width - length) / 2; i++) fputs(" ", stdout);
  fputs(s, stdout);
  fputs("\n", stdout);
}

void fancy_int(long a) {
  if (a < 1000)
    printf("%ld\n", a);
  else if (a >= 1000 && a < 1000000)
    printf("%ld,%03ld\n", a / 1000, a % 1000);
  else if (a >= 1000000 && a < 1000000000)
    printf("%ld,%03ld,%03ld\n", a / 1000000, (a % 1000000) / 1000, a % 1000);
  else if (a >= 1000000000)
    printf("%ld,%03ld,%03ld,%03ld\n", a / 1000000000, (a % 1000000000) / 1000000,
           (a % 1000000) / 1000, a % 1000);
  else
    printf("%ld\n", a);
}

void logo(int version) {
  border_print();
  printf(
      "                   __   __ ___________                 _                        \n"
      "                   \\ \\ / //  ___| ___ \\               | |                       \n"
      "                    \\ V / \\ `--.| |_/ / ___ _ __   ___| |__                     \n"
      "                    /   \\  `--. \\ ___ \\/ _ \\ '_ \\ / __| '_ \\                    \n"
      "                   / /^\\ \\/\\__/ / |_/ /  __/ | | | (__| | | |                   \n"
      "                   \\/   \\/\\____/\\____/ \\___|_| |_|\\___|_| |_|                   \n\n");
  border_print();
  center_print("Developed at Argonne National Laboratory", 79);
  char v[100];
  std::snprintf(v, sizeof(v), "Version: %d", version);
  center_print(v, 79);
  border_print();
}

void print_CLI_error(void) {
  printf("Usage: ./XSBench <options>\n");
  printf("Options include:\n");
  printf("  -m <simulation method>   Simulation method (history, event)\n");
  printf("  -s <size>                Size of H-M Benchmark to run (small, large, XL, XXL)\n");
  printf("  -g <gridpoints>          Number of gridpoints per nuclide (overrides -s defaults)\n");
  printf("  -G <grid type>           Grid search type (unionized, nuclide, hash). Defaults to unionized.\n");
  printf("  -p <particles>           Number of particle histories\n");
  printf("  -l <lookups>             History Based: Number of Cross-section (XS) lookups per particle. Event Based: Total number of XS lookups.\n");
  printf("  -h <hash bins>           Number of hash bins (only relevant when used with \"-G hash\")\n");
  printf("  -b <binary mode>         Read or write all data structures to file. If reading, this will skip initialization phase. (read, write)\n");
  printf("  -k <kernel ID>           Specifies which kernel to run. 0 is baseline, 1, 2, etc are optimized variants. (0 is default.)\n");
  printf("Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized -k 0\n");
  printf("See readme for full description of default run values\n");
  std::exit(4);
}

static int strcasecmp_local(const char* a, const char* b) {
  while (*a && *b) {
    int ca = std::tolower((unsigned char)*a);
    int cb = std::tolower((unsigned char)*b);
    if (ca != cb) return ca - cb;
    ++a;
    ++b;
  }
  return std::tolower((unsigned char)*a) - std::tolower((unsigned char)*b);
}

size_t estimate_mem_usage(Inputs in) {
  size_t single_nuclide_grid = (size_t)in.n_gridpoints * sizeof(NuclideGridPoint);
  size_t all_nuclide_grids = (size_t)in.n_isotopes * single_nuclide_grid;
  size_t size_UEG = (size_t)in.n_isotopes * (size_t)in.n_gridpoints * sizeof(double) +
                    (size_t)in.n_isotopes * (size_t)in.n_gridpoints * (size_t)in.n_isotopes * sizeof(int);
  size_t size_hash_grid = (size_t)in.hash_bins * (size_t)in.n_isotopes * sizeof(int);
  size_t memtotal;
  if (in.grid_type == UNIONIZED)
    memtotal = all_nuclide_grids + size_UEG;
  else if (in.grid_type == NUCLIDE)
    memtotal = all_nuclide_grids;
  else
    memtotal = all_nuclide_grids + size_hash_grid;

  memtotal = (size_t)std::ceil(memtotal / (1024.0 * 1024.0));
  return memtotal;
}

double get_time(void) {
  unsigned long us_since_epoch =
      (unsigned long)(std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::microseconds(1));
  return (double)us_since_epoch / 1.0e6;
}

void print_inputs(Inputs in, int nprocs, int version) {
  int mem_tot = (int)estimate_mem_usage(in);
  logo(version);
  center_print("INPUT SUMMARY", 79);
  border_print();
  printf("Programming Model:            SYCL\n");
  try {
    auto dev = get_queue().get_device();
    auto name = dev.get_info<sycl::info::device::name>();
    printf("SYCL Device:                  %s\n", name.c_str());
  } catch (...) {
    printf("SYCL Device:                  (unknown)\n");
  }
  if (in.simulation_method == EVENT_BASED)
    printf("Simulation Method:            Event Based\n");
  else
    printf("Simulation Method:            History Based\n");

  if (in.grid_type == NUCLIDE)
    printf("Grid Type:                    Nuclide Grid\n");
  else if (in.grid_type == UNIONIZED)
    printf("Grid Type:                    Unionized Grid\n");
  else
    printf("Grid Type:                    Hash\n");

  printf("Materials:                    %d\n", 12);
  printf("H-M Benchmark Size:           %s\n", in.HM);
  printf("Total Nuclides:               %ld\n", in.n_isotopes);
  printf("Gridpoints (per Nuclide):     ");
  fancy_int(in.n_gridpoints);
  if (in.grid_type == HASH) {
    printf("Hash Bins:                    ");
    fancy_int(in.hash_bins);
  }
  if (in.grid_type == UNIONIZED) {
    printf("Unionized Energy Gridpoints:  ");
    fancy_int(in.n_isotopes * in.n_gridpoints);
  }
  if (in.simulation_method == HISTORY_BASED) {
    printf("Particle Histories:           ");
    fancy_int(in.particles);
    printf("XS Lookups per Particle:      ");
    fancy_int(in.lookups);
  }
  printf("Total XS Lookups:             ");
  fancy_int(in.lookups);
  printf("Est. Memory Usage (MB):       ");
  fancy_int(mem_tot);
  printf("Binary File Mode:             ");
  if (in.binary_mode == NONE)
    printf("Off\n");
  else if (in.binary_mode == READ)
    printf("Read\n");
  else
    printf("Write\n");
  border_print();
  center_print("INITIALIZATION - DO NOT PROFILE", 79);
  border_print();
}

int print_results(Inputs in, int mype, double runtime, int nprocs, unsigned long long vhash) {
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
    printf("NOTE: Timings are estimated -- use vendor profiler tools for formal analysis\n");
    printf("Runtime:     %.3lf seconds\n", runtime);
    printf("Lookups:     ");
    fancy_int(lookups);
    printf("Lookups/s:   ");
    fancy_int(lookups_per_sec);
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
  if (std::strcmp(in.HM, "large") == 0) {
    if (vhash == large) is_invalid_result = 0;
  } else if (std::strcmp(in.HM, "small") == 0) {
    if (vhash == small) is_invalid_result = 0;
  }

  if (mype == 0) {
    if (is_invalid_result)
      printf("Verification checksum: %llu (WARNING - INAVALID CHECKSUM!)\n", vhash);
    else
      printf("Verification checksum: %llu (Valid)\n", vhash);
    border_print();
  }

  return is_invalid_result;
}

Inputs read_CLI(int argc, char* argv[]) {
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

  input.HM = (char*)std::malloc(6 * sizeof(char));
  input.HM[0] = 'l';
  input.HM[1] = 'a';
  input.HM[2] = 'r';
  input.HM[3] = 'g';
  input.HM[4] = 'e';
  input.HM[5] = '\0';

  int user_g = 0;
  int default_lookups = 1;
  int default_particles = 1;

  for (int i = 1; i < argc; i++) {
    char* arg = argv[i];

    if (std::strcmp(arg, "-g") == 0) {
      if (++i < argc) {
        user_g = 1;
        input.n_gridpoints = std::atol(argv[i]);
      } else
        print_CLI_error();
    } else if (std::strcmp(arg, "-m") == 0) {
      char* sim_type;
      if (++i < argc)
        sim_type = argv[i];
      else
        print_CLI_error();

      if (std::strcmp(sim_type, "history") == 0)
        input.simulation_method = HISTORY_BASED;
      else if (std::strcmp(sim_type, "event") == 0) {
        input.simulation_method = EVENT_BASED;
        if (default_lookups && default_particles) {
          input.lookups = input.lookups * input.particles;
          input.particles = 0;
        }
      } else
        print_CLI_error();
    } else if (std::strcmp(arg, "-l") == 0) {
      if (++i < argc) {
        input.lookups = std::atoi(argv[i]);
        default_lookups = 0;
      } else
        print_CLI_error();
    } else if (std::strcmp(arg, "-h") == 0) {
      if (++i < argc)
        input.hash_bins = std::atoi(argv[i]);
      else
        print_CLI_error();
    } else if (std::strcmp(arg, "-p") == 0) {
      if (++i < argc) {
        input.particles = std::atoi(argv[i]);
        default_particles = 0;
      } else
        print_CLI_error();
    } else if (std::strcmp(arg, "-s") == 0) {
      if (++i < argc)
        input.HM = argv[i];
      else
        print_CLI_error();
    } else if (std::strcmp(arg, "-G") == 0) {
      char* grid_type;
      if (++i < argc)
        grid_type = argv[i];
      else
        print_CLI_error();

      if (std::strcmp(grid_type, "unionized") == 0)
        input.grid_type = UNIONIZED;
      else if (std::strcmp(grid_type, "nuclide") == 0)
        input.grid_type = NUCLIDE;
      else if (std::strcmp(grid_type, "hash") == 0)
        input.grid_type = HASH;
      else
        print_CLI_error();
    } else if (std::strcmp(arg, "-b") == 0) {
      char* binary_mode;
      if (++i < argc)
        binary_mode = argv[i];
      else
        print_CLI_error();

      if (std::strcmp(binary_mode, "read") == 0)
        input.binary_mode = READ;
      else if (std::strcmp(binary_mode, "write") == 0)
        input.binary_mode = WRITE;
      else
        print_CLI_error();
    } else if (std::strcmp(arg, "-k") == 0) {
      if (++i < argc) {
        input.kernel_id = std::atoi(argv[i]);
      } else
        print_CLI_error();
    } else
      print_CLI_error();
  }

  if (input.nthreads < 1) print_CLI_error();
  if (input.n_isotopes < 1) print_CLI_error();
  if (input.n_gridpoints < 1) print_CLI_error();
  if (input.lookups < 1) print_CLI_error();
  if (input.hash_bins < 1) print_CLI_error();

  if (strcasecmp_local(input.HM, "small") != 0 && strcasecmp_local(input.HM, "large") != 0 &&
      strcasecmp_local(input.HM, "XL") != 0 && strcasecmp_local(input.HM, "XXL") != 0)
    print_CLI_error();

  if (strcasecmp_local(input.HM, "small") == 0)
    input.n_isotopes = 68;
  else if (strcasecmp_local(input.HM, "XL") == 0 && user_g == 0)
    input.n_gridpoints = 238847;
  else if (strcasecmp_local(input.HM, "XXL") == 0 && user_g == 0)
    input.n_gridpoints = (long)(238847 * 2.1);

  return input;
}

// ============================================================================
// Comparators and RNG
// ============================================================================
int double_compare(const void* a, const void* b) {
  double A = *((double*)a);
  double B = *((double*)b);
  if (A > B) return 1;
  if (A < B) return -1;
  return 0;
}

int NGP_compare(const void* a, const void* b) {
  NuclideGridPoint A = *((NuclideGridPoint*)a);
  NuclideGridPoint B = *((NuclideGridPoint*)b);
  if (A.energy > B.energy) return 1;
  if (A.energy < B.energy) return -1;
  return 0;
}

double rn_v(void) {
  static unsigned long seed = 1337;
  unsigned long a = 16807;
  unsigned long m = 2147483647;
  seed = (a * seed) % m;
  return (double)seed / (double)m;
}

inline double LCG_random_double_host(uint64_t* seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

inline uint64_t fast_forward_LCG_host(uint64_t seed, uint64_t n) {
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

// Device-callable versions (SYCL)
static inline double LCG_random_double_dev(uint64_t& seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  seed = (a * seed + c) % m;
  return (double)seed / (double)m;
}

static inline uint64_t fast_forward_LCG_dev(uint64_t seed, uint64_t n) {
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

// ============================================================================
// Material data
// ============================================================================
int* load_num_nucs(long n_isotopes) {
  int* num_nucs = (int*)std::malloc(12 * sizeof(int));
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

int* load_mats(int* num_nucs, long n_isotopes, int* max_num_nucs) {
  *max_num_nucs = 0;
  int num_mats = 12;
  for (int m = 0; m < num_mats; m++) *max_num_nucs = std::max(*max_num_nucs, num_nucs[m]);
  int* mats = (int*)std::malloc((size_t)num_mats * (size_t)(*max_num_nucs) * sizeof(int));
  std::memset(mats, 0, (size_t)num_mats * (size_t)(*max_num_nucs) * sizeof(int));

  int mats0_Sml[] = {58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1,  2,  3,  7,  8,  9,  10,
                     29, 57, 47, 48, 0,  62, 15, 33, 34, 52, 53, 54, 55, 56, 18, 23, 41};
  int mats0_Lrg[321] = {58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1,  2,  3,  7,  8,  9,  10,
                        29, 57, 47, 48, 0,  62, 15, 33, 34, 52, 53, 54, 55, 56, 18, 23, 41};
  for (int i = 0; i < 321 - 34; i++) mats0_Lrg[34 + i] = 68 + i;

  int mats1[] = {63, 64, 65, 66, 67};
  int mats2[] = {24, 41, 4, 5};
  int mats3[] = {24, 41, 4, 5};
  int mats4[] = {19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29, 30,
                 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6,  16, 17};
  int mats5[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
  int mats6[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
  int mats7[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
  int mats8[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
  int mats9[] = {24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 49, 50, 51, 11, 12, 13, 14};
  int mats10[] = {24, 41, 4, 5, 63, 64, 65, 66, 67};
  int mats11[] = {24, 41, 4, 5, 63, 64, 65, 66, 67};

  if (n_isotopes == 68)
    std::memcpy(mats, mats0_Sml, (size_t)num_nucs[0] * sizeof(int));
  else
    std::memcpy(mats, mats0_Lrg, (size_t)num_nucs[0] * sizeof(int));

  std::memcpy(mats + (*max_num_nucs) * 1, mats1, (size_t)num_nucs[1] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 2, mats2, (size_t)num_nucs[2] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 3, mats3, (size_t)num_nucs[3] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 4, mats4, (size_t)num_nucs[4] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 5, mats5, (size_t)num_nucs[5] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 6, mats6, (size_t)num_nucs[6] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 7, mats7, (size_t)num_nucs[7] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 8, mats8, (size_t)num_nucs[8] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 9, mats9, (size_t)num_nucs[9] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 10, mats10, (size_t)num_nucs[10] * sizeof(int));
  std::memcpy(mats + (*max_num_nucs) * 11, mats11, (size_t)num_nucs[11] * sizeof(int));

  return mats;
}

double* load_concs(int* num_nucs, int max_num_nucs) {
  uint64_t seed = STARTING_SEED * STARTING_SEED;
  double* concs = (double*)std::malloc((size_t)12 * (size_t)max_num_nucs * sizeof(double));
  std::memset(concs, 0, (size_t)12 * (size_t)max_num_nucs * sizeof(double));
  for (int i = 0; i < 12; i++)
    for (int j = 0; j < num_nucs[i]; j++) concs[i * max_num_nucs + j] = LCG_random_double_host(&seed);
  return concs;
}

// ============================================================================
// Grid search (host and device-callable)
// ============================================================================
static inline long grid_search_host(long n, double quarry, const double* A) {
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

static inline long grid_search_nuclide_host(long n, double quarry, const NuclideGridPoint* A, long low, long high) {
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

static inline long grid_search_dev(long n, double quarry, const double* A) {
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

static inline long grid_search_nuclide_dev(long n, double quarry, const NuclideGridPoint* A, long low, long high) {
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

// ============================================================================
// Pick material (device-callable)
// ============================================================================
static inline int pick_mat_dev(uint64_t& seed) {
  double dist[12];
  dist[0] = 0.140;
  dist[1] = 0.052;
  dist[2] = 0.275;
  dist[3] = 0.134;
  dist[4] = 0.154;
  dist[5] = 0.064;
  dist[6] = 0.066;
  dist[7] = 0.055;
  dist[8] = 0.008;
  dist[9] = 0.015;
  dist[10] = 0.025;
  dist[11] = 0.013;

  double roll = LCG_random_double_dev(seed);

  for (int i = 0; i < 12; i++) {
    double running = 0;
    for (int j = i; j > 0; j--) running += dist[j];
    if (roll < running) return i;
  }
  return 0;
}

// ============================================================================
// XS calculations (device-callable)
// ============================================================================
static inline void calculate_micro_xs_dev(double p_energy, int nuc, long n_isotopes, long n_gridpoints,
                                         const double* egrid, const int* index_data,
                                         const NuclideGridPoint* nuclide_grids, long idx,
                                         double* xs_vector, int grid_type, int hash_bins) {
  double f;
  const NuclideGridPoint* low;
  const NuclideGridPoint* high;

  if (grid_type == NUCLIDE) {
    idx = grid_search_nuclide_dev(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], 0, n_gridpoints - 1);
    if (idx == n_gridpoints - 1)
      low = &nuclide_grids[nuc * n_gridpoints + idx - 1];
    else
      low = &nuclide_grids[nuc * n_gridpoints + idx];
  } else if (grid_type == UNIONIZED) {
    long gd = index_data[idx * n_isotopes + nuc];
    if (gd == n_gridpoints - 1)
      low = &nuclide_grids[nuc * n_gridpoints + gd - 1];
    else
      low = &nuclide_grids[nuc * n_gridpoints + gd];
  } else {
    int u_low = index_data[idx * n_isotopes + nuc];
    int u_high;
    if (idx == hash_bins - 1)
      u_high = (int)n_gridpoints - 1;
    else
      u_high = index_data[(idx + 1) * n_isotopes + nuc] + 1;

    double e_low = nuclide_grids[nuc * n_gridpoints + u_low].energy;
    double e_high = nuclide_grids[nuc * n_gridpoints + u_high].energy;
    int lower;
    if (p_energy <= e_low)
      lower = 0;
    else if (p_energy >= e_high)
      lower = (int)n_gridpoints - 1;
    else
      lower = (int)grid_search_nuclide_dev(n_gridpoints, p_energy, &nuclide_grids[nuc * n_gridpoints], u_low, u_high);

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

static inline void calculate_macro_xs_dev(double p_energy, int mat, long n_isotopes, long n_gridpoints,
                                         const int* num_nucs, const double* concs,
                                         const double* egrid, const int* index_data,
                                         const NuclideGridPoint* nuclide_grids, const int* mats,
                                         double* macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs) {
  int p_nuc;
  long idx = -1;
  double conc;

  for (int k = 0; k < 5; k++) macro_xs_vector[k] = 0.0;

  if (grid_type == UNIONIZED)
    idx = grid_search_dev(n_isotopes * n_gridpoints, p_energy, egrid);
  else if (grid_type == HASH) {
    double du = 1.0 / (double)hash_bins;
    idx = (long)(p_energy / du);
    if (idx < 0) idx = 0;
    if (idx >= hash_bins) idx = hash_bins - 1;
  }

  for (int j = 0; j < num_nucs[mat]; j++) {
    double xs_vector[5];
    p_nuc = mats[mat * max_num_nucs + j];
    conc = concs[mat * max_num_nucs + j];
    calculate_micro_xs_dev(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid, index_data, nuclide_grids, idx, xs_vector,
                           grid_type, hash_bins);
    for (int k = 0; k < 5; k++) macro_xs_vector[k] += xs_vector[k] * conc;
  }
}

// ============================================================================
// Binary I/O
// ============================================================================
void binary_write(Inputs in, SimulationData SD) {
  const char* fname = "XS_data.dat";
  printf("Writing all data structures to binary file %s...\n", fname);
  FILE* fp = std::fopen(fname, "w");
  if (!fp) {
    printf("Error opening %s for write\n", fname);
    std::exit(1);
  }

  std::fwrite(&SD, sizeof(SimulationData), 1, fp);
  std::fwrite(SD.num_nucs, sizeof(int), (size_t)SD.length_num_nucs, fp);
  std::fwrite(SD.concs, sizeof(double), (size_t)SD.length_concs, fp);
  std::fwrite(SD.mats, sizeof(int), (size_t)SD.length_mats, fp);
  std::fwrite(SD.nuclide_grid, sizeof(NuclideGridPoint), (size_t)SD.length_nuclide_grid, fp);
  std::fwrite(SD.index_grid, sizeof(int), (size_t)SD.length_index_grid, fp);
  std::fwrite(SD.unionized_energy_array, sizeof(double), (size_t)SD.length_unionized_energy_array, fp);

  std::fclose(fp);
}

SimulationData binary_read(Inputs in) {
  SimulationData SD;
  const char* fname = "XS_data.dat";
  printf("Reading all data structures from binary file %s...\n", fname);

  FILE* fp = std::fopen(fname, "r");
  assert(fp != nullptr);

  std::fread(&SD, sizeof(SimulationData), 1, fp);

  SD.num_nucs = (int*)std::malloc((size_t)SD.length_num_nucs * sizeof(int));
  SD.concs = (double*)std::malloc((size_t)SD.length_concs * sizeof(double));
  SD.mats = (int*)std::malloc((size_t)SD.length_mats * sizeof(int));
  SD.nuclide_grid = (NuclideGridPoint*)std::malloc((size_t)SD.length_nuclide_grid * sizeof(NuclideGridPoint));
  SD.index_grid = (int*)std::malloc((size_t)SD.length_index_grid * sizeof(int));
  SD.unionized_energy_array = (double*)std::malloc((size_t)SD.length_unionized_energy_array * sizeof(double));

  std::fread(SD.num_nucs, sizeof(int), (size_t)SD.length_num_nucs, fp);
  std::fread(SD.concs, sizeof(double), (size_t)SD.length_concs, fp);
  std::fread(SD.mats, sizeof(int), (size_t)SD.length_mats, fp);
  std::fread(SD.nuclide_grid, sizeof(NuclideGridPoint), (size_t)SD.length_nuclide_grid, fp);
  std::fread(SD.index_grid, sizeof(int), (size_t)SD.length_index_grid, fp);
  std::fread(SD.unionized_energy_array, sizeof(double), (size_t)SD.length_unionized_energy_array, fp);

  std::fclose(fp);
  return SD;
}

// ============================================================================
// Initialization
// ============================================================================
SimulationData grid_init_do_not_profile(Inputs in, int mype) {
  SimulationData SD;
  size_t nbytes = 0;
  uint64_t seed = 42;

  if (mype == 0) printf("Intializing nuclide grids...\n");

  SD.length_nuclide_grid = in.n_isotopes * in.n_gridpoints;
  SD.nuclide_grid =
      (NuclideGridPoint*)std::malloc((size_t)SD.length_nuclide_grid * sizeof(NuclideGridPoint));
  assert(SD.nuclide_grid != nullptr);
  nbytes += (size_t)SD.length_nuclide_grid * sizeof(NuclideGridPoint);

  for (long i = 0; i < SD.length_nuclide_grid; i++) {
    SD.nuclide_grid[i].energy = LCG_random_double_host(&seed);
    SD.nuclide_grid[i].total_xs = LCG_random_double_host(&seed);
    SD.nuclide_grid[i].elastic_xs = LCG_random_double_host(&seed);
    SD.nuclide_grid[i].absorbtion_xs = LCG_random_double_host(&seed);
    SD.nuclide_grid[i].fission_xs = LCG_random_double_host(&seed);
    SD.nuclide_grid[i].nu_fission_xs = LCG_random_double_host(&seed);
  }

  for (int i = 0; i < in.n_isotopes; i++)
    std::qsort(&SD.nuclide_grid[(size_t)i * (size_t)in.n_gridpoints], (size_t)in.n_gridpoints, sizeof(NuclideGridPoint),
               NGP_compare);

  if (in.grid_type == NUCLIDE) {
    SD.length_unionized_energy_array = 0;
    SD.length_index_grid = 0;
    SD.unionized_energy_array = nullptr;
    SD.index_grid = nullptr;
  }

  if (in.grid_type == UNIONIZED) {
    if (mype == 0) printf("Intializing unionized grid...\n");
    SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
    SD.unionized_energy_array = (double*)std::malloc((size_t)SD.length_unionized_energy_array * sizeof(double));
    assert(SD.unionized_energy_array != nullptr);
    nbytes += (size_t)SD.length_unionized_energy_array * sizeof(double);

    for (long i = 0; i < SD.length_unionized_energy_array; i++) SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;

    std::qsort(SD.unionized_energy_array, (size_t)SD.length_unionized_energy_array, sizeof(double), double_compare);

    SD.length_index_grid = SD.length_unionized_energy_array * in.n_isotopes;
    SD.index_grid = (int*)std::malloc((size_t)SD.length_index_grid * sizeof(int));
    assert(SD.index_grid != nullptr);
    nbytes += (size_t)SD.length_index_grid * sizeof(int);

    int* idx_low = (int*)std::calloc((size_t)in.n_isotopes, sizeof(int));
    assert(idx_low != nullptr);
    double* energy_high = (double*)std::malloc((size_t)in.n_isotopes * sizeof(double));
    assert(energy_high != nullptr);

    for (int i = 0; i < in.n_isotopes; i++) energy_high[i] = SD.nuclide_grid[(size_t)i * (size_t)in.n_gridpoints + 1].energy;

    for (long e = 0; e < SD.length_unionized_energy_array; e++) {
      double unionized_energy = SD.unionized_energy_array[e];
      for (long i = 0; i < in.n_isotopes; i++) {
        if (unionized_energy < energy_high[i])
          SD.index_grid[(size_t)e * (size_t)in.n_isotopes + (size_t)i] = idx_low[i];
        else if (idx_low[i] == in.n_gridpoints - 2)
          SD.index_grid[(size_t)e * (size_t)in.n_isotopes + (size_t)i] = idx_low[i];
        else {
          idx_low[i]++;
          SD.index_grid[(size_t)e * (size_t)in.n_isotopes + (size_t)i] = idx_low[i];
          energy_high[i] = SD.nuclide_grid[(size_t)i * (size_t)in.n_gridpoints + (size_t)idx_low[i] + 1].energy;
        }
      }
    }

    std::free(idx_low);
    std::free(energy_high);
  }

  if (in.grid_type == HASH) {
    if (mype == 0) printf("Intializing hash grid...\n");
    SD.length_unionized_energy_array = 0;
    SD.unionized_energy_array = nullptr;

    SD.length_index_grid = (long)in.hash_bins * in.n_isotopes;
    SD.index_grid = (int*)std::malloc((size_t)SD.length_index_grid * sizeof(int));
    assert(SD.index_grid != nullptr);
    nbytes += (size_t)SD.length_index_grid * sizeof(int);

    double du = 1.0 / (double)in.hash_bins;
    for (long e = 0; e < in.hash_bins; e++) {
      double energy = e * du;
      for (long i = 0; i < in.n_isotopes; i++) {
        SD.index_grid[(size_t)e * (size_t)in.n_isotopes + (size_t)i] =
            (int)grid_search_nuclide_host(in.n_gridpoints, energy, SD.nuclide_grid + (size_t)i * (size_t)in.n_gridpoints, 0,
                                          in.n_gridpoints - 1);
      }
    }
  }

  if (mype == 0) printf("Intializing material data...\n");

  SD.num_nucs = load_num_nucs(in.n_isotopes);
  SD.length_num_nucs = 12;

  SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
  SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;

  SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
  SD.length_concs = SD.length_mats;

  if (mype == 0) printf("Intialization complete. Allocated %.0lf MB of data on CPU.\n", nbytes / 1024.0 / 1024.0);

  return SD;
}

// ============================================================================
// Device memory management (USM)
// ============================================================================
SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD) {
  if (mype == 0) printf("Allocating and moving simulation data to GPU memory space...\n");

  auto& q = get_queue();
  size_t total_sz = 0;

  SimulationData GSD = SD;  // copy metadata, then replace pointers with device allocations

  auto usm_alloc = [&](size_t bytes) -> void* {
    void* p = sycl::malloc_device(bytes, q);
    if (!p) {
      printf("USM device allocation failed (%zu bytes)\n", bytes);
      std::exit(1);
    }
    return p;
  };

  size_t sz;

  sz = (size_t)GSD.length_num_nucs * sizeof(int);
  GSD.num_nucs = (int*)usm_alloc(sz);
  q.memcpy(GSD.num_nucs, SD.num_nucs, sz);
  total_sz += sz;

  sz = (size_t)GSD.length_concs * sizeof(double);
  GSD.concs = (double*)usm_alloc(sz);
  q.memcpy(GSD.concs, SD.concs, sz);
  total_sz += sz;

  sz = (size_t)GSD.length_mats * sizeof(int);
  GSD.mats = (int*)usm_alloc(sz);
  q.memcpy(GSD.mats, SD.mats, sz);
  total_sz += sz;

  sz = (size_t)GSD.length_unionized_energy_array * sizeof(double);
  if (GSD.length_unionized_energy_array > 0) {
    GSD.unionized_energy_array = (double*)usm_alloc(sz);
    q.memcpy(GSD.unionized_energy_array, SD.unionized_energy_array, sz);
    total_sz += sz;
  } else {
    GSD.unionized_energy_array = nullptr;
  }

  sz = (size_t)GSD.length_index_grid * sizeof(int);
  if (GSD.length_index_grid > 0) {
    GSD.index_grid = (int*)usm_alloc(sz);
    q.memcpy(GSD.index_grid, SD.index_grid, sz);
    total_sz += sz;
  } else {
    GSD.index_grid = nullptr;
  }

  sz = (size_t)GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
  GSD.nuclide_grid = (NuclideGridPoint*)usm_alloc(sz);
  q.memcpy(GSD.nuclide_grid, SD.nuclide_grid, sz);
  total_sz += sz;

  sz = (size_t)in.lookups * sizeof(unsigned long);
  GSD.verification = (unsigned long*)usm_alloc(sz);
  GSD.length_verification = in.lookups;
  total_sz += sz;

  GSD.p_energy_samples = nullptr;
  GSD.mat_samples = nullptr;
  GSD.length_p_energy_samples = 0;
  GSD.length_mat_samples = 0;

  q.wait();

  if (mype == 0)
    printf("GPU Intialization complete. Allocated %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

  return GSD;
}

void release_device_memory(SimulationData GSD) {
  auto& q = get_queue();
  if (GSD.num_nucs) sycl::free(GSD.num_nucs, q);
  if (GSD.concs) sycl::free(GSD.concs, q);
  if (GSD.mats) sycl::free(GSD.mats, q);
  if (GSD.unionized_energy_array) sycl::free(GSD.unionized_energy_array, q);
  if (GSD.index_grid) sycl::free(GSD.index_grid, q);
  if (GSD.nuclide_grid) sycl::free(GSD.nuclide_grid, q);
  if (GSD.verification) sycl::free(GSD.verification, q);
  if (GSD.p_energy_samples) sycl::free(GSD.p_energy_samples, q);
  if (GSD.mat_samples) sycl::free(GSD.mat_samples, q);
}

// ============================================================================
// Simple device reductions and sorting utilities (replace thrust)
// ============================================================================
static unsigned long reduce_verification_device(unsigned long* d, size_t n) {
  auto& q = get_queue();
  unsigned long* d_out = (unsigned long*)sycl::malloc_device(sizeof(unsigned long), q);
  if (!d_out) {
    printf("USM device allocation failed for reduction output\n");
    std::exit(1);
  }
  q.memset(d_out, 0, sizeof(unsigned long));

  // Parallel reduction using atomic add
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
       sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
           atom(*d_out);
       atom.fetch_add(d[tid]);
     });
   }).wait();

  unsigned long out = 0;
  q.memcpy(&out, d_out, sizeof(unsigned long)).wait();
  sycl::free(d_out, q);
  return out;
}

static void sample_kernel(const Inputs& in, SimulationData GSD) {
  auto& q = get_queue();
  size_t n = (size_t)in.lookups;
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
       size_t i = tid[0];
       uint64_t seed = STARTING_SEED;
       seed = fast_forward_LCG_dev(seed, (uint64_t)(2 * i));
       double p_energy = LCG_random_double_dev(seed);
       int mat = pick_mat_dev(seed);
       GSD.p_energy_samples[i] = p_energy;
       GSD.mat_samples[i] = mat;
     });
   });
}

// Stable sort by key on host by copying device arrays back and forth.
// Used for optimizations 4/6 (matches behavior; not performance-focused).
static void sort_by_key_host(int* d_keys, double* d_vals, size_t n) {
  auto& q = get_queue();
  std::vector<int> keys(n);
  std::vector<double> vals(n);
  q.memcpy(keys.data(), d_keys, n * sizeof(int)).wait();
  q.memcpy(vals.data(), d_vals, n * sizeof(double)).wait();

  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; i++) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });

  std::vector<int> keys2(n);
  std::vector<double> vals2(n);
  for (size_t i = 0; i < n; i++) {
    keys2[i] = keys[idx[i]];
    vals2[i] = vals[idx[i]];
  }

  q.memcpy(d_keys, keys2.data(), n * sizeof(int));
  q.memcpy(d_vals, vals2.data(), n * sizeof(double)).wait();
}

static void sort_by_key_host(double* d_keys, int* d_vals, size_t n) {
  auto& q = get_queue();
  std::vector<double> keys(n);
  std::vector<int> vals(n);
  q.memcpy(keys.data(), d_keys, n * sizeof(double)).wait();
  q.memcpy(vals.data(), d_vals, n * sizeof(int)).wait();

  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; i++) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });

  std::vector<double> keys2(n);
  std::vector<int> vals2(n);
  for (size_t i = 0; i < n; i++) {
    keys2[i] = keys[idx[i]];
    vals2[i] = vals[idx[i]];
  }

  q.memcpy(d_keys, keys2.data(), n * sizeof(double));
  q.memcpy(d_vals, vals2.data(), n * sizeof(int)).wait();
}

static int count_value_host(const int* d_arr, size_t n, int value) {
  auto& q = get_queue();
  std::vector<int> arr(n);
  q.memcpy(arr.data(), d_arr, n * sizeof(int)).wait();
  int c = 0;
  for (size_t i = 0; i < n; i++)
    if (arr[i] == value) c++;
  return c;
}

// Partition fuel (==0) to front, preserving relative order (stable partition)
static void stable_partition_fuel_host(int* d_mats, double* d_energy, size_t n) {
  auto& q = get_queue();
  std::vector<int> mats(n);
  std::vector<double> en(n);
  q.memcpy(mats.data(), d_mats, n * sizeof(int)).wait();
  q.memcpy(en.data(), d_energy, n * sizeof(double)).wait();

  std::vector<int> mats2;
  std::vector<double> en2;
  mats2.reserve(n);
  en2.reserve(n);

  // fuel first
  for (size_t i = 0; i < n; i++) {
    if (mats[i] == 0) {
      mats2.push_back(mats[i]);
      en2.push_back(en[i]);
    }
  }
  // then others
  for (size_t i = 0; i < n; i++) {
    if (mats[i] != 0) {
      mats2.push_back(mats[i]);
      en2.push_back(en[i]);
    }
  }

  q.memcpy(d_mats, mats2.data(), n * sizeof(int));
  q.memcpy(d_energy, en2.data(), n * sizeof(double)).wait();
}

// ============================================================================
// Kernels: baseline and optimized variants
// ============================================================================
unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype) {
  if (mype == 0) printf("Running baseline event-based simulation...\n");

  auto& q = get_queue();
  size_t n = (size_t)in.lookups;

  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
       size_t i = tid[0];
       uint64_t seed = STARTING_SEED;
       seed = fast_forward_LCG_dev(seed, (uint64_t)(2 * i));

       double p_energy = LCG_random_double_dev(seed);
       int mat = pick_mat_dev(seed);

       double macro_xs_vector[5];
       calculate_macro_xs_dev(p_energy, mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs,
                              GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                              in.grid_type, in.hash_bins, GSD.max_num_nucs);

       double maxv = -1.0;
       int max_idx = 0;
       for (int j = 0; j < 5; j++) {
         if (macro_xs_vector[j] > maxv) {
           maxv = macro_xs_vector[j];
           max_idx = j;
         }
       }
       GSD.verification[i] = (unsigned long)(max_idx + 1);
     });
   }).wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, n);
  return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype) {
  const char* optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";
  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  auto& q = get_queue();

  if (mype == 0) printf("Allocating additional device data required by kernel...\n");
  size_t total_sz = 0;

  GSD.p_energy_samples = (double*)sycl::malloc_device((size_t)in.lookups * sizeof(double), q);
  GSD.mat_samples = (int*)sycl::malloc_device((size_t)in.lookups * sizeof(int), q);
  if (!GSD.p_energy_samples || !GSD.mat_samples) {
    printf("USM device allocation failed for optimization 1 arrays\n");
    std::exit(1);
  }
  GSD.length_p_energy_samples = in.lookups;
  GSD.length_mat_samples = in.lookups;
  total_sz += (size_t)in.lookups * (sizeof(double) + sizeof(int));
  if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  sample_kernel(in, GSD);
  q.wait();

  size_t n = (size_t)in.lookups;
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
       size_t i = tid[0];
       double macro_xs_vector[5];
       calculate_macro_xs_dev(GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints, GSD.num_nucs,
                              GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats,
                              macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);

       double maxv = -1.0;
       int max_idx = 0;
       for (int j = 0; j < 5; j++) {
         if (macro_xs_vector[j] > maxv) {
           maxv = macro_xs_vector[j];
           max_idx = j;
         }
       }
       GSD.verification[i] = (unsigned long)(max_idx + 1);
     });
   }).wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, n);
  return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype) {
  const char* optimization_name = "Optimization 2 - Material Lookup Kernels";
  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  auto& q = get_queue();

  if (mype == 0) printf("Allocating additional device data required by kernel...\n");
  size_t total_sz = 0;

  GSD.p_energy_samples = (double*)sycl::malloc_device((size_t)in.lookups * sizeof(double), q);
  GSD.mat_samples = (int*)sycl::malloc_device((size_t)in.lookups * sizeof(int), q);
  if (!GSD.p_energy_samples || !GSD.mat_samples) {
    printf("USM device allocation failed for optimization 2 arrays\n");
    std::exit(1);
  }
  GSD.length_p_energy_samples = in.lookups;
  GSD.length_mat_samples = in.lookups;
  total_sz += (size_t)in.lookups * (sizeof(double) + sizeof(int));
  if (mype == 0) printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  sample_kernel(in, GSD);
  q.wait();

  size_t n = (size_t)in.lookups;
  for (int m = 0; m < 12; m++) {
    q.submit([&](sycl::handler& h) {
       h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
         size_t i = tid[0];
         int mat = GSD.mat_samples[i];
         if (mat != m) return;

         double macro_xs_vector[5];
         calculate_macro_xs_dev(GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs,
                                GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                                in.grid_type, in.hash_bins, GSD.max_num_nucs);

         double maxv = -1.0;
         int max_idx = 0;
         for (int j = 0; j < 5; j++) {
           if (macro_xs_vector[j] > maxv) {
             maxv = macro_xs_vector[j];
             max_idx = j;
           }
         }
         GSD.verification[i] = (unsigned long)(max_idx + 1);
       });
     });
  }
  q.wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, n);
  return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype) {
  const char* optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";
  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  auto& q = get_queue();
  if (mype == 0) printf("Allocating additional device data required by kernel...\n");

  GSD.p_energy_samples = (double*)sycl::malloc_device((size_t)in.lookups * sizeof(double), q);
  GSD.mat_samples = (int*)sycl::malloc_device((size_t)in.lookups * sizeof(int), q);
  if (!GSD.p_energy_samples || !GSD.mat_samples) {
    printf("USM device allocation failed for optimization 3 arrays\n");
    std::exit(1);
  }
  GSD.length_p_energy_samples = in.lookups;
  GSD.length_mat_samples = in.lookups;

  if (mype == 0) printf("Beginning optimized simulation...\n");

  sample_kernel(in, GSD);
  q.wait();

  size_t n = (size_t)in.lookups;
  for (int is_fuel = 0; is_fuel <= 1; is_fuel++) {
    q.submit([&](sycl::handler& h) {
       h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
         size_t i = tid[0];
         int mat = GSD.mat_samples[i];
         bool do_lookup = ((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0));
         if (!do_lookup) return;

         double macro_xs_vector[5];
         calculate_macro_xs_dev(GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs,
                                GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                                in.grid_type, in.hash_bins, GSD.max_num_nucs);

         double maxv = -1.0;
         int max_idx = 0;
         for (int j = 0; j < 5; j++) {
           if (macro_xs_vector[j] > maxv) {
             maxv = macro_xs_vector[j];
             max_idx = j;
           }
         }
         GSD.verification[i] = (unsigned long)(max_idx + 1);
       });
     });
  }
  q.wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, n);
  return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype) {
  const char* optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";
  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  auto& q = get_queue();

  if (mype == 0) printf("Allocating additional device data required by kernel...\n");
  GSD.p_energy_samples = (double*)sycl::malloc_device((size_t)in.lookups * sizeof(double), q);
  GSD.mat_samples = (int*)sycl::malloc_device((size_t)in.lookups * sizeof(int), q);
  if (!GSD.p_energy_samples || !GSD.mat_samples) {
    printf("USM device allocation failed for optimization 4 arrays\n");
    std::exit(1);
  }
  GSD.length_p_energy_samples = in.lookups;
  GSD.length_mat_samples = in.lookups;

  if (mype == 0) printf("Beginning optimized simulation...\n");

  sample_kernel(in, GSD);
  q.wait();

  int n_lookups_per_material[12];
  for (int m = 0; m < 12; m++) n_lookups_per_material[m] = count_value_host(GSD.mat_samples, (size_t)in.lookups, m);

  // Sort materials by key, carrying energies
  sort_by_key_host(GSD.mat_samples, GSD.p_energy_samples, (size_t)in.lookups);

  size_t offset = 0;
  for (int m = 0; m < 12; m++) {
    size_t n = (size_t)n_lookups_per_material[m];
    size_t off = offset;
    q.submit([&](sycl::handler& h) {
       h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
         size_t li = tid[0];
         size_t i = li + off;

         int mat = GSD.mat_samples[i];
         if (mat != m) return;

         double macro_xs_vector[5];
         calculate_macro_xs_dev(GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs,
                                GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                                in.grid_type, in.hash_bins, GSD.max_num_nucs);

         double maxv = -1.0;
         int max_idx = 0;
         for (int j = 0; j < 5; j++) {
           if (macro_xs_vector[j] > maxv) {
             maxv = macro_xs_vector[j];
             max_idx = j;
           }
         }
         GSD.verification[i] = (unsigned long)(max_idx + 1);
       });
     });
    offset += n;
  }
  q.wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, (size_t)in.lookups);
  return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype) {
  const char* optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";
  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  auto& q = get_queue();

  if (mype == 0) printf("Allocating additional device data required by kernel...\n");
  GSD.p_energy_samples = (double*)sycl::malloc_device((size_t)in.lookups * sizeof(double), q);
  GSD.mat_samples = (int*)sycl::malloc_device((size_t)in.lookups * sizeof(int), q);
  if (!GSD.p_energy_samples || !GSD.mat_samples) {
    printf("USM device allocation failed for optimization 5 arrays\n");
    std::exit(1);
  }
  GSD.length_p_energy_samples = in.lookups;
  GSD.length_mat_samples = in.lookups;

  if (mype == 0) printf("Beginning optimized simulation...\n");

  sample_kernel(in, GSD);
  q.wait();

  int n_fuel_lookups = count_value_host(GSD.mat_samples, (size_t)in.lookups, 0);

  // stable partition fuel to front (materials and energies aligned)
  stable_partition_fuel_host(GSD.mat_samples, GSD.p_energy_samples, (size_t)in.lookups);

  // fuel kernel
  size_t n_fuel = (size_t)n_fuel_lookups;
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>(n_fuel), [=](sycl::id<1> tid) {
       size_t i = tid[0];
       double macro_xs_vector[5];
       calculate_macro_xs_dev(GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints, GSD.num_nucs,
                              GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats,
                              macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
       double maxv = -1.0;
       int max_idx = 0;
       for (int j = 0; j < 5; j++) {
         if (macro_xs_vector[j] > maxv) {
           maxv = macro_xs_vector[j];
           max_idx = j;
         }
       }
       GSD.verification[i] = (unsigned long)(max_idx + 1);
     });
   });

  // other kernel
  size_t n_other = (size_t)in.lookups - n_fuel;
  size_t off = n_fuel;
  q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::range<1>(n_other), [=](sycl::id<1> tid) {
       size_t li = tid[0];
       size_t i = li + off;
       double macro_xs_vector[5];
       calculate_macro_xs_dev(GSD.p_energy_samples[i], GSD.mat_samples[i], in.n_isotopes, in.n_gridpoints, GSD.num_nucs,
                              GSD.concs, GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats,
                              macro_xs_vector, in.grid_type, in.hash_bins, GSD.max_num_nucs);
       double maxv = -1.0;
       int max_idx = 0;
       for (int j = 0; j < 5; j++) {
         if (macro_xs_vector[j] > maxv) {
           maxv = macro_xs_vector[j];
           max_idx = j;
         }
       }
       GSD.verification[i] = (unsigned long)(max_idx + 1);
     });
   });

  q.wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, (size_t)in.lookups);
  return verification_scalar;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype) {
  const char* optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";
  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  auto& q = get_queue();

  if (mype == 0) printf("Allocating additional device data required by kernel...\n");
  GSD.p_energy_samples = (double*)sycl::malloc_device((size_t)in.lookups * sizeof(double), q);
  GSD.mat_samples = (int*)sycl::malloc_device((size_t)in.lookups * sizeof(int), q);
  if (!GSD.p_energy_samples || !GSD.mat_samples) {
    printf("USM device allocation failed for optimization 6 arrays\n");
    std::exit(1);
  }
  GSD.length_p_energy_samples = in.lookups;
  GSD.length_mat_samples = in.lookups;

  if (mype == 0) printf("Beginning optimized simulation...\n");

  sample_kernel(in, GSD);
  q.wait();

  int n_lookups_per_material[12];
  for (int m = 0; m < 12; m++) n_lookups_per_material[m] = count_value_host(GSD.mat_samples, (size_t)in.lookups, m);

  // Sort by material
  sort_by_key_host(GSD.mat_samples, GSD.p_energy_samples, (size_t)in.lookups);

  // Sort each material by energy (subranges)
  size_t offset = 0;
  for (int m = 0; m < 12; m++) {
    size_t n = (size_t)n_lookups_per_material[m];
    if (n > 1) {
      // sort energy keys with mats as values (to keep aligned)
      sort_by_key_host(GSD.p_energy_samples + offset, GSD.mat_samples + offset, n);
    }
    offset += n;
  }

  // Launch material kernels as in opt4
  offset = 0;
  for (int m = 0; m < 12; m++) {
    size_t n = (size_t)n_lookups_per_material[m];
    size_t off = offset;
    q.submit([&](sycl::handler& h) {
       h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> tid) {
         size_t li = tid[0];
         size_t i = li + off;
         int mat = GSD.mat_samples[i];
         if (mat != m) return;

         double macro_xs_vector[5];
         calculate_macro_xs_dev(GSD.p_energy_samples[i], mat, in.n_isotopes, in.n_gridpoints, GSD.num_nucs, GSD.concs,
                                GSD.unionized_energy_array, GSD.index_grid, GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                                in.grid_type, in.hash_bins, GSD.max_num_nucs);

         double maxv = -1.0;
         int max_idx = 0;
         for (int j = 0; j < 5; j++) {
           if (macro_xs_vector[j] > maxv) {
             maxv = macro_xs_vector[j];
             max_idx = j;
           }
         }
         GSD.verification[i] = (unsigned long)(max_idx + 1);
       });
     });
    offset += n;
  }
  q.wait();

  if (mype == 0) printf("Reducing verification results...\n");
  unsigned long verification_scalar = reduce_verification_device(GSD.verification, (size_t)in.lookups);
  return verification_scalar;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
  int version = 20;
  int mype = 0;
  double omp_start, omp_end;
  int nprocs = 1;
  unsigned long long verification = 0ULL;

  Inputs in = read_CLI(argc, argv);

  if (mype == 0) print_inputs(in, nprocs, version);

  SimulationData SD;
  if (in.binary_mode == READ)
    SD = binary_read(in);
  else
    SD = grid_init_do_not_profile(in, mype);

  if (in.binary_mode == WRITE && mype == 0) binary_write(in, SD);

  SimulationData GSD = move_simulation_data_to_device(in, mype, SD);

  if (mype == 0) {
    printf("\n");
    border_print();
    center_print("SIMULATION", 79);
    border_print();
  }

  omp_start = get_time();

  if (in.simulation_method == EVENT_BASED) {
    if (in.kernel_id == 0)
      verification = run_event_based_simulation_baseline(in, GSD, mype);
    else if (in.kernel_id == 1)
      verification = run_event_based_simulation_optimization_1(in, GSD, mype);
    else if (in.kernel_id == 2)
      verification = run_event_based_simulation_optimization_2(in, GSD, mype);
    else if (in.kernel_id == 3)
      verification = run_event_based_simulation_optimization_3(in, GSD, mype);
    else if (in.kernel_id == 4)
      verification = run_event_based_simulation_optimization_4(in, GSD, mype);
    else if (in.kernel_id == 5)
      verification = run_event_based_simulation_optimization_5(in, GSD, mype);
    else if (in.kernel_id == 6)
      verification = run_event_based_simulation_optimization_6(in, GSD, mype);
    else {
      printf("Error: No kernel ID %d found!\n", in.kernel_id);
      std::exit(1);
    }
  } else {
    printf("History-based simulation not implemented in CUDA code. Instead,\nuse the event-based method with \"-m event\" argument.\n");
    std::exit(1);
  }

  if (mype == 0) {
    printf("\n");
    printf("Simulation complete.\n");
  }

  omp_end = get_time();

  release_device_memory(GSD);

  verification = verification % 999983ULL;

  int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);
  return is_invalid_result;
}
