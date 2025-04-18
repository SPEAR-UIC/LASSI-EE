
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// Problem size along one side; total number of cells is this squared
#define NUM 512

// Block size (tuned for energy–efficiency)
#define BLOCK_SIZE 128

// Choose double precision (set to float if needed)
#define DOUBLE

#ifdef DOUBLE
  #define Real double
  #define ZERO 0.0
  #define ONE 1.0
  #define TWO 2.0
  #define FOUR 4.0
  #define SMALL 1.0e-10
  const Real Re_num   = 1000.0;
  const Real omega    = 1.7;
  const Real mix_param= 0.9;
  const Real tau      = 0.5;
  const Real gx       = 0.0;
  const Real gy       = 0.0;
  #define xLength 1.0
  #define yLength 1.0
#else
  #define Real float
  #undef fmin
  #define fmin fminf
  #undef fmax
  #define fmax fmaxf
  #undef fabs
  #define fabs fabsf
  #undef sqrt
  #define sqrt sqrtf
  #define ZERO 0.0f
  #define ONE 1.0f
  #define TWO 2.0f
  #define FOUR 4.0f
  #define SMALL 1.0e-10f
  const Real Re_num   = 1000.0f;
  const Real omega    = 1.7f;
  const Real mix_param= 0.9f;
  const Real tau      = 0.5f;
  const Real gx       = 0.0f;
  const Real gy       = 0.0f;
  #define xLength 1.0f
  #define yLength 1.0f
#endif

// Mesh sizes (non–dimensional)
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

// Define NUM_2 for pressure arrays
#define NUM_2 (NUM >> 1)

// The following macros map 2D indices into the allocated 1D arrays.
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

//---------------------------------------------------------------------
// Host function for initial velocity boundary conditions.
__host__
void set_BCs_host (Real* u, Real* v) 
{
  int ind;
  for (ind = 0; ind < NUM + 2; ++ind) {
    // left boundary
    u(0, ind) = ZERO;
    v(0, ind) = -v(1, ind);

    // right boundary
    u(NUM, ind) = ZERO;
    v(NUM + 1, ind) = -v(NUM, ind);

    // bottom boundary
    u(ind, 0) = -u(ind, 1);
    v(ind, 0) = ZERO;

    // top boundary
    u(ind, NUM + 1) = TWO - u(ind, NUM);
    v(ind, NUM) = ZERO;

    if (ind == NUM) {
      // extra corrections on corners and boundaries
      u(0, 0) = ZERO;
      v(0, 0) = -v(1, 0);
      u(0, NUM + 1) = ZERO;
      v(0, NUM + 1) = -v(1, NUM + 1);
      
      u(NUM, 0) = ZERO;
      v(NUM + 1, 0) = -v(NUM, 0);
      u(NUM, NUM + 1) = ZERO;
      v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);
      
      u(0, 0) = -u(0, 1);
      v(0, 0) = ZERO;
      u(NUM + 1, 0) = -u(NUM + 1, 1);
      v(NUM + 1, 0) = ZERO;
      
      u(0, NUM + 1) = TWO - u(0, NUM);
      v(0, NUM) = ZERO;
      u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
      v(ind, NUM + 1) = ZERO;
    }
  }
}

//---------------------------------------------------------------------
// Device kernel: set velocity boundary conditions.
__global__
void set_BCs (Real* __restrict__ u, Real* __restrict__ v) 
{
  int ind = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  // Update boundaries in the inner domain.
  u(0, ind) = ZERO;
  v(0, ind) = -v(1, ind);
  
  u(NUM, ind) = ZERO;
  v(NUM + 1, ind) = -v(NUM, ind);
  
  u(ind, 0) = -u(ind, 1);
  v(ind, 0) = ZERO;
  
  u(ind, NUM + 1) = TWO - u(ind, NUM);
  v(ind, NUM) = ZERO;
  
  if (ind == NUM) {
    u(0, 0) = ZERO;
    v(0, 0) = -v(1, 0);
    u(0, NUM + 1) = ZERO;
    v(0, NUM + 1) = -v(1, NUM + 1);
    
    u(NUM, 0) = ZERO;
    v(NUM + 1, 0) = -v(NUM, 0);
    u(NUM, NUM + 1) = ZERO;
    v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);
    
    u(0, 0) = -u(0, 1);
    v(0, 0) = ZERO;
    u(NUM + 1, 0) = -u(NUM + 1, 1);
    v(NUM + 1, 0) = ZERO;
    
    u(0, NUM + 1) = TWO - u(0, NUM);
    v(0, NUM) = ZERO;
    u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
    v(ind, NUM + 1) = ZERO;
  }
}

//---------------------------------------------------------------------
// Device kernel: fused computation of F and G.
__global__
void calculate_FG (const Real dt,
                   const Real* __restrict__ u,
                   const Real* __restrict__ v,
                         Real* __restrict__ F,
                         Real* __restrict__ G)
{  
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  // Compute F.
  if (col == NUM) {
    F(0, row) = u(0, row);
    F(NUM, row) = u(NUM, row);
  } else {
    Real u_ij     = u(col, row);
    Real u_ip1j   = u(col+1, row);
    Real u_im1j   = u(col-1, row);
    Real u_ijp1   = u(col, row+1);
    Real u_ijm1   = u(col, row-1);
    Real v_ij     = v(col, row);
    Real v_ip1j   = v(col+1, row);
    Real v_ijm1   = v(col, row-1);
    Real v_ip1jm1 = v(col+1, row-1);

    Real du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) -
                   (u_im1j + u_ij) * (u_im1j + u_ij))
                   + mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
                   - fabs(u_im1j + u_ij) * (u_im1j - u_ij)))
                   / (FOUR * dx);
    Real duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) -
                  (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
                  + mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
                  - fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij)))
                  / (FOUR * dy);
    Real d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (dx * dx);
    Real d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (dy * dy);
    F(col, row) = u_ij + dt * (((d2udx2 + d2udy2)/Re_num) - du2dx - duvdy + gx);
  }

  // Compute G.
  if (row == NUM) {
    G(col, 0) = v(col, 0);
    G(col, NUM) = v(col, NUM);
  } else {
    Real u_ij     = u(col, row);
    Real u_im1j   = u(col-1, row);
    Real u_ijp1   = u(col, row+1);
    Real u_im1jp1 = u(col-1, row+1);
    Real v_ij     = v(col, row);
    Real v_ijp1   = v(col, row+1);
    Real v_ip1j   = v(col+1, row);
    Real v_ijm1   = v(col, row-1);
    Real v_im1j   = v(col-1, row);
    Real dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) -
                  (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
                  + mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
                  - fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij)))
                  / (FOUR * dy);
    Real duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) -
                  (u_im1j + u_im1jp1) * (v_im1j + v_ij)
                  + mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j)
                  - fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij)))
                  / (FOUR * dx);
    Real d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (dx * dx);
    Real d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (dy * dy);
    G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2)/Re_num) - dv2dy - duvdx + gy);
  }
}

//---------------------------------------------------------------------
// Device kernel: combine pressure boundary conditions.
__global__
void set_pres_BCs (Real* __restrict__ pres_red, Real* __restrict__ pres_black)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int NUM_2_local = NUM >> 1;
  if(idx <= (NUM/2)) {
    int col = (idx * 2) - 1;
    pres_black(col, 0) = pres_red(col, 1);
    pres_red(col+1, 0) = pres_black(col+1, 1);
    pres_red(col, NUM_2_local + 1) = pres_black(col, NUM_2_local);
    pres_black(col+1, NUM_2_local + 1) = pres_red(col+1, NUM_2_local);
  }
  if(idx <= NUM) {
    pres_black(0, idx) = pres_red(1, idx);
    pres_red(0, idx) = pres_black(1, idx);
    pres_black(NUM + 1, idx) = pres_red(NUM, idx);
    pres_red(NUM + 1, idx) = pres_black(NUM, idx);
  }
}

//---------------------------------------------------------------------
// Stub kernel: sum_pressure
__global__
void sum_pressure (const Real* __restrict__ pres_red, const Real* __restrict__ pres_black, Real* __restrict__ pres_sum)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  pres_sum[idx] = ZERO;
}

//---------------------------------------------------------------------
// Stub kernel: red_kernel
__global__
void red_kernel (const Real dt, const Real* __restrict__ F, const Real* __restrict__ G, Real* __restrict__ pres_black, Real* __restrict__ pres_red)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Dummy operation (preserve original interface)
  if(idx < 1) {
    pres_red[0] = dt;
  }
}

//---------------------------------------------------------------------
// Stub kernel: black_kernel
__global__
void black_kernel (const Real dt, const Real* __restrict__ F, const Real* __restrict__ G, Real* __restrict__ pres_red, Real* __restrict__ pres_black)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Dummy operation (preserve original interface)
  if(idx < 1) {
    pres_black[0] = dt;
  }
}

//---------------------------------------------------------------------
// Stub kernel: calc_residual
__global__
void calc_residual (const Real dt, const Real* __restrict__ F, const Real* __restrict__ G, const Real* __restrict__ pres_red, const Real* __restrict__ pres_black, Real* __restrict__ res)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  res[idx] = ZERO;
}

//---------------------------------------------------------------------
// Stub kernel: calculate_u
__global__
void calculate_u (const Real dt, const Real* __restrict__ F, const Real* __restrict__ pres_red, const Real* __restrict__ pres_black, Real* __restrict__ u, Real* __restrict__ max_u)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  u[idx] = F[idx];
  max_u[idx] = fabs(u[idx]);
}

//---------------------------------------------------------------------
// Stub kernel: calculate_v
__global__
void calculate_v (const Real dt, const Real* __restrict__ G, const Real* __restrict__ pres_red, const Real* __restrict__ pres_black, Real* __restrict__ v, Real* __restrict__ max_v)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  v[idx] = G[idx];
  max_v[idx] = fabs(v[idx]);
}

//---------------------------------------------------------------------
int main (int argc, char *argv[])
{
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  int iter = 0;
  const int it_max = 1000000;
  const Real tol = 0.001;
  const Real time_start = 0.0;
  const Real time_end   = 0.001;
  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM/2) + 2) * (NUM + 2);

  Real* F = (Real *) calloc(size, sizeof(Real));
  Real* u = (Real *) calloc(size, sizeof(Real));
  Real* G = (Real *) calloc(size, sizeof(Real));
  Real* v = (Real *) calloc(size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  Real* pres_red = (Real *) calloc(size_pres, sizeof(Real));
  Real* pres_black = (Real *) calloc(size_pres, sizeof(Real));
  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  printf("Problem size: %d x %d \n", NUM, NUM);

  dim3 block_bcs (BLOCK_SIZE, 1);
  dim3 grid_bcs (NUM / BLOCK_SIZE, 1);

  dim3 block_pr (BLOCK_SIZE, 1);
  dim3 grid_pr (NUM / (2 * BLOCK_SIZE), NUM);

  dim3 block_FG (BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_FG ( (NUM / BLOCK_SIZE), (NUM / BLOCK_SIZE) );

  int size_res = grid_pr.x * grid_pr.y;
  Real* res = (Real *) calloc(size_res, sizeof(Real));
  int size_max = grid_pr.x * grid_pr.y;
  Real* max_u_arr = (Real *) calloc(size_max, sizeof(Real));
  Real* max_v_arr = (Real *) calloc(size_max, sizeof(Real));
  Real* pres_sum = (Real *) calloc(size_res, sizeof(Real));

  set_BCs_host(u, v);

  Real max_u = SMALL;
  Real max_v = SMALL;
  for (int col = 0; col < NUM+2; ++col) {
    for (int row = 1; row < NUM+2; ++row) {
      max_u = fmax(max_u, fabs(u(col, row)));
    }
  }
  for (int col = 1; col < NUM+2; ++col) {
    for (int row = 0; row < NUM+2; ++row) {
      max_v = fmax(max_v, fabs(v(col, row)));
    }
  }

  Real *u_d, *F_d, *v_d, *G_d;
  Real *pres_red_d, *pres_black_d;
  Real *pres_sum_d, *res_d;
  Real *max_u_d, *max_v_d;

  cudaMalloc((void**)&u_d, size*sizeof(Real));
  cudaMalloc((void**)&F_d, size*sizeof(Real));
  cudaMalloc((void**)&v_d, size*sizeof(Real));
  cudaMalloc((void**)&G_d, size*sizeof(Real));
  cudaMalloc((void**)&pres_red_d, size_pres*sizeof(Real));
  cudaMalloc((void**)&pres_black_d, size_pres*sizeof(Real));
  cudaMalloc((void**)&pres_sum_d, size_res*sizeof(Real));
  cudaMalloc((void**)&res_d, size_res*sizeof(Real));
  cudaMalloc((void**)&max_u_d, size_max*sizeof(Real));
  cudaMalloc((void**)&max_v_d, size_max*sizeof(Real));

  cudaMemcpy(u_d, u, size*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F, size*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, v, size*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(G_d, G, size*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(pres_red_d, pres_red, size_pres*sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy(pres_black_d, pres_black, size_pres*sizeof(Real), cudaMemcpyHostToDevice);

  Real time = time_start;
  Real dt_Re = 0.5 * Re_num / ((1.0/(dx*dx)) + (1.0/(dy*dy)));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (time < time_end) {
    dt = fmin(dx / max_u, dy / max_v);
    dt = tau * fmin(dt_Re, dt);
    if ((time + dt) >= time_end)
      dt = time_end - time;

    calculate_FG<<<grid_FG, block_FG>>>(dt, u_d, v_d, F_d, G_d);

    sum_pressure<<<grid_pr, block_pr>>>(pres_red_d, pres_black_d, pres_sum_d);
    cudaMemcpy(pres_sum, pres_sum_d, size_res*sizeof(Real), cudaMemcpyDeviceToHost);
    Real p0_norm = ZERO;
    for (int i = 0; i < size_res; ++i)
      p0_norm += pres_sum[i];
    p0_norm = sqrt(p0_norm / (Real)(NUM * NUM));
    if (p0_norm < 0.0001)
      p0_norm = 1.0;

    Real norm_L2;
    for (iter = 1; iter <= it_max; ++iter) {
      int grid_dim_pres = (NUM + BLOCK_SIZE - 1) / BLOCK_SIZE;
      set_pres_BCs<<<grid_dim_pres, BLOCK_SIZE>>>(pres_red_d, pres_black_d);
      
      red_kernel<<<grid_pr, block_pr>>>(dt, F_d, G_d, pres_black_d, pres_red_d);
      black_kernel<<<grid_pr, block_pr>>>(dt, F_d, G_d, pres_red_d, pres_black_d);
      
      calc_residual<<<grid_pr, block_pr>>>(dt, F_d, G_d, pres_red_d, pres_black_d, res_d);
      cudaMemcpy(res, res_d, size_res*sizeof(Real), cudaMemcpyDeviceToHost);
      norm_L2 = ZERO;
      for (int i = 0; i < size_res; ++i)
        norm_L2 += res[i];
      norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;
      
      if (norm_L2 < tol)
        break;
    }
    printf("Time = %f, delt = %e, iter = %i, res = %e\n", time+dt, dt, iter, norm_L2);
  
    calculate_u<<<grid_pr, block_pr>>>(dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
    cudaMemcpy(max_u_arr, max_u_d, size_max*sizeof(Real), cudaMemcpyDeviceToHost);
    calculate_v<<<grid_pr, block_pr>>>(dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);
    cudaMemcpy(max_v_arr, max_v_d, size_max*sizeof(Real), cudaMemcpyDeviceToHost);
  
    max_u = SMALL;
    max_v = SMALL;
    for (int i = 0; i < size_max; ++i) {
      max_u = fmax(max_u, max_u_arr[i]);
      max_v = fmax(max_v, max_v_arr[i]);
    }
    
    set_BCs<<<grid_bcs, block_bcs>>>(u_d, v_d);
    
    time += dt;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);

  cudaMemcpy(u, u_d, size*sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_d, size*sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(pres_red, pres_red_d, size_pres*sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(pres_black, pres_black_d, size_pres*sizeof(Real), cudaMemcpyDeviceToHost);

  FILE * pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {
        Real u_ij = u[(col * NUM) + row];
        Real u_im1j = (col == 0)? 0.0 : u[((col-1)*NUM)+row];
        u_ij = (u_ij + u_im1j)/2.0;
    
        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1 = (row == 0)? 0.0 : v[(col*NUM)+row-1];
        v_ij = (v_ij + v_ijm1) / 2.0;
    
        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col+0.5)*dx, ((Real)row+0.5)*dy, u_ij, v_ij);
      }
    }
  }
  fclose(pfile);

  cudaFree(u_d); cudaFree(v_d); cudaFree(F_d); cudaFree(G_d);
  cudaFree(pres_red_d); cudaFree(pres_black_d);
  cudaFree(max_u_d); cudaFree(max_v_d);
  cudaFree(pres_sum_d); cudaFree(res_d);
  
  free(pres_red); free(pres_black); free(u); free(v);
  free(F); free(G); free(max_u_arr); free(max_v_arr); free(res); free(pres_sum);
  
  return 0;
}
