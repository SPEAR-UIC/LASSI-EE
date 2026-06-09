#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

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

static int str_case_cmp(const char *a, const char *b) {
  while (*a && *b) {
    char ca = (*a >= 'A' && *a <= 'Z') ? *a - 'A' + 'a' : *a;
    char cb = (*b >= 'A' && *b <= 'Z') ? *b - 'A' + 'a' : *b;
    if (ca != cb) return ca - cb;
    ++a;
    ++b;
  }
  return *a - *b;
}

sycl::queue &get_queue() {
  static sycl::queue *q = []() {
    auto async_handler = [](sycl::exception_list exceptions) {
      for (auto &e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const sycl::exception &ex) {
          std::cerr << "Asynchronous SYCL exception: " << ex.what() << "\n";
        }
      }
      std::terminate();
    };

    try {
      return new sycl::queue(sycl::gpu_selector_v, async_handler);
    } catch (...) {
      return new sycl::queue(sycl::default_selector_v, async_handler);
    }
  }();
  return *q;
}

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

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData SD, int mype);
unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype);
unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype);

SimulationData grid_init_do_not_profile(Inputs in, int mype);
SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD);
void release_device_memory(SimulationData GSD);

int NGP_compare(const void *a, const void *b);
int double_compare(const void *a, const void *b);
double rn_v(void);
size_t estimate_mem_usage(Inputs in);
double get_time(void);

int *load_num_nucs(long n_isotopes);
int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs);
double *load_concs(int *num_nucs, int max_num_nucs);

inline long grid_search(long n, double quarry, double *A) {
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

inline long grid_search_nuclide(long n, double quarry, NuclideGridPoint *A, long low, long high) {
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

inline double LCG_random_double(uint64_t *seed) {
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double)(*seed) / (double)m;
}

inline uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
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

inline int pick_mat(uint64_t *seed) {
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

inline void calculate_micro_xs(double p_energy, int nuc, long n_isotopes,
                               long n_gridpoints, double *egrid, int *index_data,
                               NuclideGridPoint *nuclide_grids, long idx,
                               double *xs_vector, int grid_type, int hash_bins) {
  double f;
  NuclideGridPoint *low;

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

  NuclideGridPoint *high = low + 1;

  f = (high->energy - p_energy) / (high->energy - low->energy);

  xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
  xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
  xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
  xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
  xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

inline void calculate_macro_xs(double p_energy, int mat, long n_isotopes,
                               long n_gridpoints, int *num_nucs, double *concs,
                               double *egrid, int *index_data,
                               NuclideGridPoint *nuclide_grids, int *mats,
                               double *macro_xs_vector, int grid_type,
                               int hash_bins, int max_num_nucs) {
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

    calculate_micro_xs(p_energy, p_nuc, n_isotopes, n_gridpoints, egrid,
                       index_data, nuclide_grids, idx, xs_vector, grid_type,
                       hash_bins);

    for (int k = 0; k < 5; k++)
      macro_xs_vector[k] += xs_vector[k] * conc;
  }
}

inline unsigned long verification_from_macro(double *macro_xs_vector) {
  double max = -1.0;
  int max_idx = 0;
  for (int j = 0; j < 5; j++) {
    if (macro_xs_vector[j] > max) {
      max = macro_xs_vector[j];
      max_idx = j;
    }
  }
  return (unsigned long)(max_idx + 1);
}

static void launch_sampling_kernel(Inputs in, SimulationData GSD, int nthreads) {
  sycl::queue &q = get_queue();
  size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
  if (nblocks == 0) return;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= in.lookups)
           return;

         uint64_t seed = STARTING_SEED;
         seed = fast_forward_LCG(seed, 2ULL * (uint64_t)i);

         double p_energy = LCG_random_double(&seed);
         int mat = pick_mat(&seed);

         GSD.p_energy_samples[i] = p_energy;
         GSD.mat_samples[i] = mat;
       })
      .wait();
}

static void launch_lookup_baseline(Inputs in, SimulationData GSD, int nthreads) {
  sycl::queue &q = get_queue();
  size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
  if (nblocks == 0) return;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= in.lookups)
           return;

         uint64_t seed = STARTING_SEED;
         seed = fast_forward_LCG(seed, 2ULL * (uint64_t)i);

         double p_energy = LCG_random_double(&seed);
         int mat = pick_mat(&seed);

         double macro_xs_vector[5] = {0};

         calculate_macro_xs(p_energy, mat, in.n_isotopes, in.n_gridpoints,
                            GSD.num_nucs, GSD.concs,
                            GSD.unionized_energy_array, GSD.index_grid,
                            GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                            in.grid_type, in.hash_bins, GSD.max_num_nucs);

         GSD.verification[i] = verification_from_macro(macro_xs_vector);
       })
      .wait();
}

static void launch_lookup_opt1(Inputs in, SimulationData GSD, int nthreads) {
  sycl::queue &q = get_queue();
  size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
  if (nblocks == 0) return;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= in.lookups)
           return;

         double macro_xs_vector[5] = {0};

         calculate_macro_xs(GSD.p_energy_samples[i], GSD.mat_samples[i],
                            in.n_isotopes, in.n_gridpoints, GSD.num_nucs,
                            GSD.concs, GSD.unionized_energy_array,
                            GSD.index_grid, GSD.nuclide_grid, GSD.mats,
                            macro_xs_vector, in.grid_type, in.hash_bins,
                            GSD.max_num_nucs);

         GSD.verification[i] = verification_from_macro(macro_xs_vector);
       })
      .wait();
}

static void launch_lookup_opt2(Inputs in, SimulationData GSD, int m, int nthreads) {
  sycl::queue &q = get_queue();
  size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
  if (nblocks == 0) return;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= in.lookups)
           return;

         int mat = GSD.mat_samples[i];
         if (mat != m)
           return;

         double macro_xs_vector[5] = {0};

         calculate_macro_xs(GSD.p_energy_samples[i], mat, in.n_isotopes,
                            in.n_gridpoints, GSD.num_nucs, GSD.concs,
                            GSD.unionized_energy_array, GSD.index_grid,
                            GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                            in.grid_type, in.hash_bins, GSD.max_num_nucs);

         GSD.verification[i] = verification_from_macro(macro_xs_vector);
       })
      .wait();
}

static void launch_lookup_opt3(Inputs in, SimulationData GSD, int is_fuel, int nthreads) {
  sycl::queue &q = get_queue();
  size_t nblocks = (in.lookups + nthreads - 1) / nthreads;
  if (nblocks == 0) return;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= in.lookups)
           return;

         int mat = GSD.mat_samples[i];

         if (((is_fuel == 1) && (mat == 0)) || ((is_fuel == 0) && (mat != 0))) {
           double macro_xs_vector[5] = {0};

           calculate_macro_xs(GSD.p_energy_samples[i], mat, in.n_isotopes,
                              in.n_gridpoints, GSD.num_nucs, GSD.concs,
                              GSD.unionized_energy_array, GSD.index_grid,
                              GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                              in.grid_type, in.hash_bins, GSD.max_num_nucs);

           GSD.verification[i] = verification_from_macro(macro_xs_vector);
         }
       })
      .wait();
}

static void launch_lookup_opt4(Inputs in, SimulationData GSD, int m, int n_lookups, int offset, int nthreads) {
  if (n_lookups <= 0) return;

  sycl::queue &q = get_queue();
  size_t nblocks = (n_lookups + nthreads - 1) / nthreads;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= n_lookups)
           return;

         i += offset;

         int mat = GSD.mat_samples[i];
         if (mat != m)
           return;

         double macro_xs_vector[5] = {0};

         calculate_macro_xs(GSD.p_energy_samples[i], mat, in.n_isotopes,
                            in.n_gridpoints, GSD.num_nucs, GSD.concs,
                            GSD.unionized_energy_array, GSD.index_grid,
                            GSD.nuclide_grid, GSD.mats, macro_xs_vector,
                            in.grid_type, in.hash_bins, GSD.max_num_nucs);

         GSD.verification[i] = verification_from_macro(macro_xs_vector);
       })
      .wait();
}

static void launch_lookup_opt5(Inputs in, SimulationData GSD, int n_lookups, int offset, int nthreads) {
  if (n_lookups <= 0) return;

  sycl::queue &q = get_queue();
  size_t nblocks = (n_lookups + nthreads - 1) / nthreads;

  q.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(nblocks * nthreads), sycl::range<1>(nthreads)),
       [=](sycl::nd_item<1> item) {
         int i = item.get_global_linear_id();

         if (i >= n_lookups)
           return;

         i += offset;

         double macro_xs_vector[5] = {0};

         calculate_macro_xs(GSD.p_energy_samples[i], GSD.mat_samples[i],
                            in.n_isotopes, in.n_gridpoints, GSD.num_nucs,
                            GSD.concs, GSD.unionized_energy_array,
                            GSD.index_grid, GSD.nuclide_grid, GSD.mats,
                            macro_xs_vector, in.grid_type, in.hash_bins,
                            GSD.max_num_nucs);

         GSD.verification[i] = verification_from_macro(macro_xs_vector);
       })
      .wait();
}

static unsigned long long reduce_verification(SimulationData GSD, int n) {
  sycl::queue &q = get_queue();
  unsigned long long *sum = sycl::malloc_shared<unsigned long long>(1, q);
  *sum = 0;

  q.submit([&](sycl::handler &h) {
     auto red = sycl::reduction(sum, static_cast<unsigned long long>(0), sycl::plus<unsigned long long>());
     h.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> idx, auto &acc) {
       acc += (unsigned long long)GSD.verification[idx[0]];
     });
   }).wait();

  unsigned long long ret = *sum;
  sycl::free(sum, q);
  return ret;
}

static void allocate_sample_arrays(Inputs in, SimulationData &GSD, int mype) {
  sycl::queue &q = get_queue();
  if (mype == 0) printf("Allocating additional device data required by kernel...\n");

  size_t total_sz = 0;

  GSD.p_energy_samples = sycl::malloc_device<double>(in.lookups, q);
  total_sz += in.lookups * sizeof(double);
  GSD.length_p_energy_samples = in.lookups;

  GSD.mat_samples = sycl::malloc_device<int>(in.lookups, q);
  total_sz += in.lookups * sizeof(int);
  GSD.length_mat_samples = in.lookups;

  if (mype == 0)
    printf("Allocated an additional %.0lf MB of data on GPU.\n", total_sz / 1024.0 / 1024.0);
}

static void free_sample_arrays(SimulationData &GSD) {
  sycl::queue &q = get_queue();
  if (GSD.p_energy_samples) sycl::free(GSD.p_energy_samples, q);
  if (GSD.mat_samples) sycl::free(GSD.mat_samples, q);
  GSD.p_energy_samples = nullptr;
  GSD.mat_samples = nullptr;
}

struct SamplePair {
  int mat;
  double energy;
};

static void copy_samples_to_host(SimulationData GSD, int n, std::vector<SamplePair> &samples) {
  sycl::queue &q = get_queue();
  std::vector<int> mats(n);
  std::vector<double> energies(n);

  q.memcpy(mats.data(), GSD.mat_samples, n * sizeof(int)).wait();
  q.memcpy(energies.data(), GSD.p_energy_samples, n * sizeof(double)).wait();

  samples.resize(n);
  for (int i = 0; i < n; i++) {
    samples[i].mat = mats[i];
    samples[i].energy = energies[i];
  }
}

static void copy_samples_to_device(SimulationData GSD, const std::vector<SamplePair> &samples) {
  sycl::queue &q = get_queue();
  int n = (int)samples.size();
  std::vector<int> mats(n);
  std::vector<double> energies(n);

  for (int i = 0; i < n; i++) {
    mats[i] = samples[i].mat;
    energies[i] = samples[i].energy;
  }

  q.memcpy(GSD.mat_samples, mats.data(), n * sizeof(int)).wait();
  q.memcpy(GSD.p_energy_samples, energies.data(), n * sizeof(double)).wait();
}

int main(int argc, char *argv[]) {
  try {
    int version = 20;
    int mype = 0;
    double omp_start, omp_end;
    int nprocs = 1;
    unsigned long long verification = 0;

    (void)get_queue();

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

    release_device_memory(GSD);

    verification = verification % 999983;

    int is_invalid_result = print_results(in, mype, omp_end - omp_start, nprocs, verification);

    return is_invalid_result;
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 1;
  }
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
  sprintf(v, "Version: %d", version);
  center_print(v, 79);
  border_print();
}

void center_print(const char *s, int width) {
  int length = strlen(s);
  for (int i = 0; i <= (width - length) / 2; i++)
    fputs(" ", stdout);
  fputs(s, stdout);
  fputs("\n", stdout);
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

    printf("NOTE: Timings are estimated -- use profiling tools for formal analysis\n");
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

  if (strcmp(in.HM, "large") == 0) {
    if (vhash == large)
      is_invalid_result = 0;
  } else if (strcmp(in.HM, "small") == 0) {
    if (vhash == small)
      is_invalid_result = 0;
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

void print_inputs(Inputs in, int nprocs, int version) {
  int mem_tot = estimate_mem_usage(in);
  logo(version);
  center_print("INPUT SUMMARY", 79);
  border_print();
  printf("Programming Model:            SYCL\n");
  printf("SYCL Device:                  %s\n", get_queue().get_device().get_info<sycl::info::device::name>().c_str());

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

void border_print(void) {
  printf("================================================================================\n");
}

void fancy_int(long a) {
  if (a < 1000)
    printf("%ld\n", a);
  else if (a >= 1000 && a < 1000000)
    printf("%ld,%03ld\n", a / 1000, a % 1000);
  else if (a >= 1000000 && a < 1000000000)
    printf("%ld,%03ld,%03ld\n", a / 1000000, (a % 1000000) / 1000, a % 1000);
  else if (a >= 1000000000)
    printf("%ld,%03ld,%03ld,%03ld\n", a / 1000000000, (a % 1000000000) / 1000000, (a % 1000000) / 1000, a % 1000);
  else
    printf("%ld\n", a);
}

void print_CLI_error(void) {
  printf("Usage: ./XSBench <options>\n");
  printf("Options include:\n");
  printf("  -m <simulation method>   Simulation method (history, event)\n");
  printf("  -s <size>                Size of H-M Benchmark to run (small, large, XL, XXL)\n");
  printf("  -g <gridpoints>          Number of gridpoints per nuclide (overrides -s defaults)\n");
  printf("  -G <grid type>           Grid search type (unionized, nuclide, hash). Defaults to unionized.\n");
  printf("  -p <particles>           Number of particle histories\n");
  printf("  -l <lookups>             History Based: Number of XS lookups per particle. Event Based: Total number of XS lookups.\n");
  printf("  -h <hash bins>           Number of hash bins (only relevant when used with \"-G hash\")\n");
  printf("  -b <binary mode>         Read or write all data structures to file. (read, write)\n");
  printf("  -k <kernel ID>           Specifies which kernel to run. 0 is baseline.\n");
  printf("Default is equivalent to: -m history -s large -l 34 -p 500000 -G unionized -k 0\n");
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

  input.HM = (char *)malloc(6 * sizeof(char));
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
      if (++i < argc)
        input.kernel_id = atoi(argv[i]);
      else
        print_CLI_error();
    } else
      print_CLI_error();
  }

  if (input.nthreads < 1) print_CLI_error();
  if (input.n_isotopes < 1) print_CLI_error();
  if (input.n_gridpoints < 1) print_CLI_error();
  if (input.lookups < 1) print_CLI_error();
  if (input.hash_bins < 1) print_CLI_error();

  if (str_case_cmp(input.HM, "small") != 0 &&
      str_case_cmp(input.HM, "large") != 0 &&
      str_case_cmp(input.HM, "XL") != 0 &&
      str_case_cmp(input.HM, "XXL") != 0)
    print_CLI_error();

  if (str_case_cmp(input.HM, "small") == 0)
    input.n_isotopes = 68;
  else if (str_case_cmp(input.HM, "XL") == 0 && user_g == 0)
    input.n_gridpoints = 238847;
  else if (str_case_cmp(input.HM, "XXL") == 0 && user_g == 0)
    input.n_gridpoints = 238847 * 2.1;

  return input;
}

void binary_write(Inputs in, SimulationData SD) {
  const char *fname = "XS_data.dat";
  printf("Writing all data structures to binary file %s...\n", fname);
  FILE *fp = fopen(fname, "wb");
  assert(fp != NULL);

  fwrite(&SD, sizeof(SimulationData), 1, fp);
  fwrite(SD.num_nucs, sizeof(int), SD.length_num_nucs, fp);
  fwrite(SD.concs, sizeof(double), SD.length_concs, fp);
  fwrite(SD.mats, sizeof(int), SD.length_mats, fp);
  fwrite(SD.nuclide_grid, sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp);
  fwrite(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
  fwrite(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);

  fclose(fp);
}

SimulationData binary_read(Inputs in) {
  SimulationData SD = {};

  const char *fname = "XS_data.dat";
  printf("Reading all data structures from binary file %s...\n", fname);

  FILE *fp = fopen(fname, "rb");
  assert(fp != NULL);

  fread(&SD, sizeof(SimulationData), 1, fp);

  SD.num_nucs = (int *)malloc(SD.length_num_nucs * sizeof(int));
  SD.concs = (double *)malloc(SD.length_concs * sizeof(double));
  SD.mats = (int *)malloc(SD.length_mats * sizeof(int));
  SD.nuclide_grid = (NuclideGridPoint *)malloc(SD.length_nuclide_grid * sizeof(NuclideGridPoint));
  SD.index_grid = SD.length_index_grid ? (int *)malloc(SD.length_index_grid * sizeof(int)) : nullptr;
  SD.unionized_energy_array = SD.length_unionized_energy_array ? (double *)malloc(SD.length_unionized_energy_array * sizeof(double)) : nullptr;

  fread(SD.num_nucs, sizeof(int), SD.length_num_nucs, fp);
  fread(SD.concs, sizeof(double), SD.length_concs, fp);
  fread(SD.mats, sizeof(int), SD.length_mats, fp);
  fread(SD.nuclide_grid, sizeof(NuclideGridPoint), SD.length_nuclide_grid, fp);
  fread(SD.index_grid, sizeof(int), SD.length_index_grid, fp);
  fread(SD.unionized_energy_array, sizeof(double), SD.length_unionized_energy_array, fp);

  fclose(fp);

  SD.verification = nullptr;
  SD.p_energy_samples = nullptr;
  SD.mat_samples = nullptr;
  SD.length_verification = 0;
  SD.length_p_energy_samples = 0;
  SD.length_mat_samples = 0;

  return SD;
}

SimulationData move_simulation_data_to_device(Inputs in, int mype, SimulationData SD) {
  sycl::queue &q = get_queue();

  if (mype == 0)
    printf("Allocating and moving simulation data to GPU memory space...\n");

  size_t total_sz = 0;
  SimulationData GSD = SD;

  auto copy_int = [&](int *src, size_t n) -> int * {
    if (n == 0) return nullptr;
    int *dst = sycl::malloc_device<int>(n, q);
    q.memcpy(dst, src, n * sizeof(int)).wait();
    total_sz += n * sizeof(int);
    return dst;
  };

  auto copy_double = [&](double *src, size_t n) -> double * {
    if (n == 0) return nullptr;
    double *dst = sycl::malloc_device<double>(n, q);
    q.memcpy(dst, src, n * sizeof(double)).wait();
    total_sz += n * sizeof(double);
    return dst;
  };

  GSD.num_nucs = copy_int(SD.num_nucs, GSD.length_num_nucs);
  GSD.concs = copy_double(SD.concs, GSD.length_concs);
  GSD.mats = copy_int(SD.mats, GSD.length_mats);
  GSD.unionized_energy_array = copy_double(SD.unionized_energy_array, GSD.length_unionized_energy_array);
  GSD.index_grid = copy_int(SD.index_grid, GSD.length_index_grid);

  if (GSD.length_nuclide_grid) {
    GSD.nuclide_grid = sycl::malloc_device<NuclideGridPoint>(GSD.length_nuclide_grid, q);
    q.memcpy(GSD.nuclide_grid, SD.nuclide_grid, GSD.length_nuclide_grid * sizeof(NuclideGridPoint)).wait();
    total_sz += GSD.length_nuclide_grid * sizeof(NuclideGridPoint);
  } else
    GSD.nuclide_grid = nullptr;

  GSD.verification = sycl::malloc_device<unsigned long>(in.lookups, q);
  total_sz += in.lookups * sizeof(unsigned long);
  GSD.length_verification = in.lookups;

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
  sycl::queue &q = get_queue();

  if (GSD.num_nucs) sycl::free(GSD.num_nucs, q);
  if (GSD.concs) sycl::free(GSD.concs, q);
  if (GSD.mats) sycl::free(GSD.mats, q);
  if (GSD.unionized_energy_array) sycl::free(GSD.unionized_energy_array, q);
  if (GSD.index_grid) sycl::free(GSD.index_grid, q);
  if (GSD.nuclide_grid) sycl::free(GSD.nuclide_grid, q);
  if (GSD.verification) sycl::free(GSD.verification, q);
}

SimulationData grid_init_do_not_profile(Inputs in, int mype) {
  SimulationData SD = {};
  size_t nbytes = 0;
  uint64_t seed = 42;

  if (mype == 0) printf("Intializing nuclide grids...\n");

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
    qsort(&SD.nuclide_grid[i * in.n_gridpoints], in.n_gridpoints, sizeof(NuclideGridPoint), NGP_compare);

  if (in.grid_type == NUCLIDE) {
    SD.length_unionized_energy_array = 0;
    SD.length_index_grid = 0;
    SD.unionized_energy_array = nullptr;
    SD.index_grid = nullptr;
  }

  if (in.grid_type == UNIONIZED) {
    if (mype == 0) printf("Intializing unionized grid...\n");

    SD.length_unionized_energy_array = in.n_isotopes * in.n_gridpoints;
    SD.unionized_energy_array = (double *)malloc(SD.length_unionized_energy_array * sizeof(double));
    assert(SD.unionized_energy_array != NULL);
    nbytes += SD.length_unionized_energy_array * sizeof(double);

    for (int i = 0; i < SD.length_unionized_energy_array; i++)
      SD.unionized_energy_array[i] = SD.nuclide_grid[i].energy;

    qsort(SD.unionized_energy_array, SD.length_unionized_energy_array, sizeof(double), double_compare);

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
    if (mype == 0) printf("Intializing hash grid...\n");

    SD.length_unionized_energy_array = 0;
    SD.unionized_energy_array = nullptr;
    SD.length_index_grid = in.hash_bins * in.n_isotopes;
    SD.index_grid = (int *)malloc(SD.length_index_grid * sizeof(int));
    assert(SD.index_grid != NULL);
    nbytes += SD.length_index_grid * sizeof(int);

    double du = 1.0 / in.hash_bins;

    for (long e = 0; e < in.hash_bins; e++) {
      double energy = e * du;
      for (long i = 0; i < in.n_isotopes; i++)
        SD.index_grid[e * in.n_isotopes + i] =
            grid_search_nuclide(in.n_gridpoints, energy, SD.nuclide_grid + i * in.n_gridpoints, 0, in.n_gridpoints - 1);
    }
  }

  if (mype == 0) printf("Intializing material data...\n");

  SD.num_nucs = load_num_nucs(in.n_isotopes);
  SD.length_num_nucs = 12;

  SD.mats = load_mats(SD.num_nucs, in.n_isotopes, &SD.max_num_nucs);
  SD.length_mats = SD.length_num_nucs * SD.max_num_nucs;

  SD.concs = load_concs(SD.num_nucs, SD.max_num_nucs);
  SD.length_concs = SD.length_mats;

  SD.verification = nullptr;
  SD.length_verification = 0;
  SD.p_energy_samples = nullptr;
  SD.length_p_energy_samples = 0;
  SD.mat_samples = nullptr;
  SD.length_mat_samples = 0;

  if (mype == 0)
    printf("Intialization complete. Allocated %.0lf MB of data on CPU.\n", nbytes / 1024.0 / 1024.0);

  return SD;
}

int *load_num_nucs(long n_isotopes) {
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

int *load_mats(int *num_nucs, long n_isotopes, int *max_num_nucs) {
  *max_num_nucs = 0;
  int num_mats = 12;

  for (int m = 0; m < num_mats; m++)
    if (num_nucs[m] > *max_num_nucs)
      *max_num_nucs = num_nucs[m];

  int *mats = (int *)calloc(num_mats * (*max_num_nucs), sizeof(int));

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
                 30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16, 17};
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
  double *concs = (double *)calloc(12 * max_num_nucs, sizeof(double));

  for (int i = 0; i < 12; i++)
    for (int j = 0; j < num_nucs[i]; j++)
      concs[i * max_num_nucs + j] = LCG_random_double(&seed);

  return concs;
}

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype) {
  if (mype == 0) printf("Running baseline event-based simulation...\n");

  launch_lookup_baseline(in, GSD, 256);

  if (mype == 0) printf("Reducing verification results...\n");

  return reduce_verification(GSD, in.lookups);
}

unsigned long long run_event_based_simulation_optimization_1(Inputs in, SimulationData GSD, int mype) {
  const char *optimization_name = "Optimization 1 - basic sample/lookup kernel splitting";

  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  allocate_sample_arrays(in, GSD, mype);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  launch_sampling_kernel(in, GSD, 32);
  launch_lookup_opt1(in, GSD, 32);

  if (mype == 0) printf("Reducing verification results...\n");

  unsigned long long ret = reduce_verification(GSD, in.lookups);
  free_sample_arrays(GSD);
  return ret;
}

unsigned long long run_event_based_simulation_optimization_2(Inputs in, SimulationData GSD, int mype) {
  const char *optimization_name = "Optimization 2 - Material Lookup Kernels";

  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  allocate_sample_arrays(in, GSD, mype);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  launch_sampling_kernel(in, GSD, 32);

  for (int m = 0; m < 12; m++)
    launch_lookup_opt2(in, GSD, m, 32);

  if (mype == 0) printf("Reducing verification results...\n");

  unsigned long long ret = reduce_verification(GSD, in.lookups);
  free_sample_arrays(GSD);
  return ret;
}

unsigned long long run_event_based_simulation_optimization_3(Inputs in, SimulationData GSD, int mype) {
  const char *optimization_name = "Optimization 3 - Fuel or Other Lookup Kernels";

  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  allocate_sample_arrays(in, GSD, mype);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  launch_sampling_kernel(in, GSD, 32);
  launch_lookup_opt3(in, GSD, 0, 32);
  launch_lookup_opt3(in, GSD, 1, 32);

  if (mype == 0) printf("Reducing verification results...\n");

  unsigned long long ret = reduce_verification(GSD, in.lookups);
  free_sample_arrays(GSD);
  return ret;
}

unsigned long long run_event_based_simulation_optimization_4(Inputs in, SimulationData GSD, int mype) {
  const char *optimization_name = "Optimization 4 - All Material Lookup Kernels + Material Sort";

  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  allocate_sample_arrays(in, GSD, mype);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  launch_sampling_kernel(in, GSD, 32);

  std::vector<SamplePair> samples;
  copy_samples_to_host(GSD, in.lookups, samples);

  int n_lookups_per_material[12] = {};
  for (const auto &s : samples)
    n_lookups_per_material[s.mat]++;

  std::sort(samples.begin(), samples.end(), [](const SamplePair &a, const SamplePair &b) {
    return a.mat < b.mat;
  });

  copy_samples_to_device(GSD, samples);

  int offset = 0;
  for (int m = 0; m < 12; m++) {
    launch_lookup_opt4(in, GSD, m, n_lookups_per_material[m], offset, 32);
    offset += n_lookups_per_material[m];
  }

  if (mype == 0) printf("Reducing verification results...\n");

  unsigned long long ret = reduce_verification(GSD, in.lookups);
  free_sample_arrays(GSD);
  return ret;
}

unsigned long long run_event_based_simulation_optimization_5(Inputs in, SimulationData GSD, int mype) {
  const char *optimization_name = "Optimization 5 - Fuel/No Fuel Lookup Kernels + Fuel/No Fuel Sort";

  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  allocate_sample_arrays(in, GSD, mype);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  launch_sampling_kernel(in, GSD, 32);

  std::vector<SamplePair> samples;
  copy_samples_to_host(GSD, in.lookups, samples);

  int n_fuel_lookups = 0;
  for (const auto &s : samples)
    if (s.mat == 0) n_fuel_lookups++;

  std::partition(samples.begin(), samples.end(), [](const SamplePair &s) {
    return s.mat == 0;
  });

  copy_samples_to_device(GSD, samples);

  launch_lookup_opt5(in, GSD, n_fuel_lookups, 0, 32);
  launch_lookup_opt5(in, GSD, in.lookups - n_fuel_lookups, n_fuel_lookups, 32);

  if (mype == 0) printf("Reducing verification results...\n");

  unsigned long long ret = reduce_verification(GSD, in.lookups);
  free_sample_arrays(GSD);
  return ret;
}

unsigned long long run_event_based_simulation_optimization_6(Inputs in, SimulationData GSD, int mype) {
  const char *optimization_name = "Optimization 6 - Material & Energy Sorts + Material-specific Kernels";

  if (mype == 0) printf("Simulation Kernel:\"%s\"\n", optimization_name);

  allocate_sample_arrays(in, GSD, mype);

  if (mype == 0) printf("Beginning optimized simulation...\n");

  launch_sampling_kernel(in, GSD, 32);

  std::vector<SamplePair> samples;
  copy_samples_to_host(GSD, in.lookups, samples);

  int n_lookups_per_material[12] = {};
  for (const auto &s : samples)
    n_lookups_per_material[s.mat]++;

  std::sort(samples.begin(), samples.end(), [](const SamplePair &a, const SamplePair &b) {
    if (a.mat != b.mat) return a.mat < b.mat;
    return a.energy < b.energy;
  });

  copy_samples_to_device(GSD, samples);

  int offset = 0;
  for (int m = 0; m < 12; m++) {
    launch_lookup_opt4(in, GSD, m, n_lookups_per_material[m], offset, 32);
    offset += n_lookups_per_material[m];
  }

  if (mype == 0) printf("Reducing verification results...\n");

  unsigned long long ret = reduce_verification(GSD, in.lookups);
  free_sample_arrays(GSD);
  return ret;
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

double rn_v(void) {
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

size_t estimate_mem_usage(Inputs in) {
  size_t single_nuclide_grid = in.n_gridpoints * sizeof(NuclideGridPoint);
  size_t all_nuclide_grids = in.n_isotopes * single_nuclide_grid;
  size_t size_UEG = in.n_isotopes * in.n_gridpoints * sizeof(double) +
                    in.n_isotopes * in.n_gridpoints * in.n_isotopes * sizeof(int);
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

double get_time(void) {
  unsigned long us_since_epoch =
      std::chrono::high_resolution_clock::now().time_since_epoch() / std::chrono::microseconds(1);
  return (double)us_since_epoch / 1.0e6;
}
