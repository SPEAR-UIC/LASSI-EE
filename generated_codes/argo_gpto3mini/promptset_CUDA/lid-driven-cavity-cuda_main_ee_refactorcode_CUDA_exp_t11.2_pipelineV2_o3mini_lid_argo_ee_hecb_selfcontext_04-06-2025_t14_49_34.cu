
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

/** Problem size along one side; total number of cells is this squared */
#define NUM 512

// block size for one-dimensional kernels (unchanged)
#define BLOCK_SIZE 128

// block dimensions for the new 2D calculate_F kernel
#define BLOCK_F_DIM 16

// block dimensions for the new 2D calculate_G kernel
#define BLOCK_G_DIM 16

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
#define Real double

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

#define SMALL 1.0e-10

/** Domain size (non-dimensional) */
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

/** Domain size (non-dimensional) */
#define xLength 1.0f
#define yLength 1.0f
#endif

/** Original host constants */
#ifdef DOUBLE
const Real Re_num = 1000.0;
const Real omega = 1.7;
const Real mix_param = 0.9;
const Real tau = 0.5;
const Real gx = 0.0;
const Real gy = 0.0;
#else
const Real Re_num = 1000.0f;
const Real omega = 1.7f;
const Real mix_param = 0.9f;
const Real tau = 0.5f;
const Real gx = 0.0f;
const Real gy = 0.0f;
#endif

/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

// Map two-dimensional indices to one-dimensional memory
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

//---------------------------------------------------------------------
// Place simulation parameters in constant memory for the device kernels.
__constant__ Real d_Re_num;
__constant__ Real d_omega;
__constant__ Real d_mix_param;
__constant__ Real d_gx;
__constant__ Real d_gy;
__constant__ Real d_dx;
__constant__ Real d_dy;
__constant__ Real d_tau; // not used in device kernels, but available if needed

///////////////////////////////////////////////////////////////////////////////
__host__
void set_BCs_host (Real* u, Real* v) 
{
  int ind;

  // loop through rows and columns
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
      // left boundary
      u(0, 0) = ZERO;
      v(0, 0) = -v(1, 0);
      u(0, NUM + 1) = ZERO;
      v(0, NUM + 1) = -v(1, NUM + 1);

      // right boundary
      u(NUM, 0) = ZERO;
      v(NUM + 1, 0) = -v(NUM, 0);
      u(NUM, NUM + 1) = ZERO;
      v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

      // bottom boundary
      u(0, 0) = -u(0, 1);
      v(0, 0) = ZERO;
      u(NUM + 1, 0) = -u(NUM + 1, 1);
      v(NUM + 1, 0) = ZERO;

      // top boundary
      u(0, NUM + 1) = TWO - u(0, NUM);
      v(0, NUM) = ZERO;
      u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
      v(ind, NUM + 1) = ZERO;
    } // end if

  } // end for

} // end set_BCs_host

///////////////////////////////////////////////////////////////////////////////
/** 
 * Consolidated boundary condition update kernel.
 *
 * This new 2D kernel replaces the previous one-dimensional version. Each thread
 * maps to a grid location (i,j) in the (NUM+2)x(NUM+2) domain (including ghost cells)
 * and, if it is on a boundary, applies the corresponding update.
 */
__global__
void set_BCs (Real* __restrict__ u, Real* __restrict__ v) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // i in [0, NUM+1]
  int j = blockIdx.y * blockDim.y + threadIdx.y;    // j in [0, NUM+1]

  if (i >= (NUM + 2) || j >= (NUM + 2))
    return;

  // Left boundary: i == 0
  if (i == 0) {
    u[0*(NUM+2) + j] = ZERO;
    v[0*(NUM+2) + j] = - v[1*(NUM+2) + j];
  }
  // Right boundary for u: i == NUM
  if (i == NUM) {
    u[i*(NUM+2) + j] = ZERO;
  }
  // Right boundary for v: i == NUM+1
  if (i == NUM + 1) {
    v[i*(NUM+2) + j] = - v[NUM*(NUM+2) + j];
  }
  // Bottom boundary: j == 0
  if (j == 0) {
    u[i*(NUM+2) + 0] = - u[i*(NUM+2) + 1];
    v[i*(NUM+2) + 0] = ZERO;
  }
  // Top boundary for u: j == NUM+1
  if (j == NUM + 1) {
    u[i*(NUM+2) + (NUM+1)] = TWO - u[i*(NUM+2) + NUM];
  }
  // Top boundary for v: j == NUM
  if (j == NUM) {
    v[i*(NUM+2) + j] = ZERO;
  }
} // end set_BCs

///////////////////////////////////////////////////////////////////////////////
/** 
 * Updated calculate_F kernel using shared memory tiling for improved memory reuse.
 */
__global__ 
void calculate_F (const Real dt,
                  const Real* __restrict__ u,
                  const Real* __restrict__ v,
                        Real* __restrict__ F) 
{  
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tid_x + 1;
    int row = blockIdx.y * blockDim.y + tid_y + 1;
    
    __shared__ Real sh_u[BLOCK_F_DIM + 2][BLOCK_F_DIM + 2];
    __shared__ Real sh_v[BLOCK_F_DIM + 2][BLOCK_F_DIM + 2];
    
    int global_col = col - 1;
    int global_row = row - 1;
    
    if ((global_col >= 0) && (global_col < NUM+2) &&
        (global_row >= 0) && (global_row < NUM+2))
    {
        sh_u[tid_y+1][tid_x+1] = u(global_col, global_row);
        sh_v[tid_y+1][tid_x+1] = v(global_col, global_row);
    }
    
    if (tid_x == 0 && (col - 1) > 0)
    {
        sh_u[tid_y+1][0] = u(col - 2, row);
        sh_v[tid_y+1][0] = v(col - 2, row);
    }
    if (tid_x == blockDim.x - 1 && (col + 1) < NUM+2)
    {
        sh_u[tid_y+1][tid_x+2] = u(col + 1, row);
        sh_v[tid_y+1][tid_x+2] = v(col + 1, row);
    }
    if (tid_y == 0 && (row - 1) > 0)
    {
        sh_u[0][tid_x+1] = u(col, row - 1);
        sh_v[0][tid_x+1] = v(col, row - 1);
    }
    if (tid_y == blockDim.y - 1 && (row + 1) < NUM+2)
    {
        sh_u[tid_y+2][tid_x+1] = u(col, row + 1);
        sh_v[tid_y+2][tid_x+1] = v(col, row + 1);
    }
    if (tid_x == 0 && tid_y == 0 && (col - 1) > 0 && (row - 1) > 0)
    {
        sh_u[0][0] = u(col - 1, row - 1);
        sh_v[0][0] = v(col - 1, row - 1);
    }
    if (tid_x == blockDim.x - 1 && tid_y == 0 && (col + 1) < NUM+2 && (row - 1) > 0)
    {
        sh_u[0][tid_x+2] = u(col + 1, row - 1);
        sh_v[0][tid_x+2] = v(col + 1, row - 1);
    }
    if (tid_x == 0 && tid_y == blockDim.y - 1 && (col - 1) > 0 && (row + 1) < NUM+2)
    {
        sh_u[tid_y+2][0] = u(col - 1, row + 1);
        sh_v[tid_y+2][0] = v(col - 1, row + 1);
    }
    if (tid_x == blockDim.x - 1 && tid_y == blockDim.y - 1 && (col + 1) < NUM+2 && (row + 1) < NUM+2)
    {
        sh_u[tid_y+2][tid_x+2] = u(col + 1, row + 1);
        sh_v[tid_y+2][tid_x+2] = v(col + 1, row + 1);
    }
    
    __syncthreads();
    
    if (col == NUM) {
        F(0, row) = u(0, row);
        F(NUM, row) = u(NUM, row);
        return;
    }
    
    Real u_ij = sh_u[tid_y+1][tid_x+1];
    Real u_ip1j = sh_u[tid_y+1][tid_x+2];
    Real u_im1j = sh_u[tid_y+1][tid_x];
    Real u_ijp1 = sh_u[tid_y+2][tid_x+1];
    Real u_ijm1 = sh_u[tid_y][tid_x+1];

    Real v_ij = sh_v[tid_y+1][tid_x+1];
    Real v_ip1j = sh_v[tid_y+1][tid_x+2];
    Real v_ijm1 = sh_v[tid_y][tid_x+1];
    Real v_ip1jm1 = sh_v[tid_y][tid_x+2];

    Real du2dx, duvdy, d2udx2, d2udy2;

    du2dx = (((u_ij + u_ip1j) * (u_ij + u_ip1j) - (u_im1j + u_ij) * (u_im1j + u_ij))
             + d_mix_param * (fabs(u_ij + u_ip1j) * (u_ij - u_ip1j)
                            - fabs(u_im1j + u_ij) * (u_im1j - u_ij))) / (FOUR * d_dx);
    duvdy = ((v_ij + v_ip1j) * (u_ij + u_ijp1) - (v_ijm1 + v_ip1jm1) * (u_ijm1 + u_ij)
             + d_mix_param * (fabs(v_ij + v_ip1j) * (u_ij - u_ijp1)
                            - fabs(v_ijm1 + v_ip1jm1) * (u_ijm1 - u_ij))) / (FOUR * d_dy);
    d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) / (d_dx * d_dx);
    d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) / (d_dy * d_dy);

    F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / d_Re_num) - du2dx - duvdy + d_gx);
} // end calculate_F

///////////////////////////////////////////////////////////////////////////////
/** 
 * Updated calculate_G kernel rewritten for a 2D thread block configuration.
 */
__global__ 
void calculate_G (const Real dt,
                  const Real* __restrict__ u,
                  const Real* __restrict__ v,
                        Real* __restrict__ G) 
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + tx + 1;
  int row = blockIdx.y * blockDim.y + ty + 1;

  if(row > NUM || col > NUM) return;

  if (row == NUM) {
    G(col, 0) = v(col, 0);
    G(col, NUM) = v(col, NUM);
    return;
  }

  Real u_ij = u(col, row);
  Real u_ijp1 = u(col, row + 1);
  Real u_im1j = u(col - 1, row);
  Real u_im1jp1 = u(col - 1, row + 1);

  Real v_ij = v(col, row);
  Real v_ijp1 = v(col, row + 1);
  Real v_ip1j = v(col + 1, row);
  Real v_ijm1 = v(col, row - 1);
  Real v_im1j = v(col - 1, row);

  Real dv2dy, duvdx, d2vdx2, d2vdy2;

  dv2dy = ((v_ij + v_ijp1) * (v_ij + v_ijp1) - (v_ijm1 + v_ij) * (v_ijm1 + v_ij)
        + d_mix_param * (fabs(v_ij + v_ijp1) * (v_ij - v_ijp1)
          - fabs(v_ijm1 + v_ij) * (v_ijm1 - v_ij))) / (FOUR * d_dy);
  duvdx = ((u_ij + u_ijp1) * (v_ij + v_ip1j) - (u_im1j + u_im1jp1) * (v_im1j + v_ij)
        + d_mix_param * (fabs(u_ij + u_ijp1) * (v_ij - v_ip1j) 
          - fabs(u_im1j + u_im1jp1) * (v_im1j - v_ij))) / (FOUR * d_dx);
  d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) / (d_dx * d_dx);
  d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) / (d_dy * d_dy);

  G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / d_Re_num) - dv2dy - duvdx + d_gy);
} // end calculate_G

///////////////////////////////////////////////////////////////////////////////
__global__
void sum_pressure (const Real* __restrict__ pres_red,
                   const Real* __restrict__ pres_black, 
                         Real* __restrict__ pres_sum)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  int NUM_2 = NUM >> 1;

  Real pres_r = pres_red(col, row);
  Real pres_b = pres_black(col, row);
  Real val = (pres_r * pres_r) + (pres_b * pres_b);

  unsigned int mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
  }

  __shared__ Real warpSum[BLOCK_SIZE/32];
  int lane = threadIdx.x & 31;
  int warpId = threadIdx.x >> 5;
  if(lane == 0)
      warpSum[warpId] = val;
  __syncthreads();

  if(warpId == 0) {
      val = (lane < (blockDim.x / 32)) ? warpSum[lane] : 0.0;
      for (int offset = warpSize/2; offset > 0; offset /= 2) {
          val += __shfl_down_sync(mask, val, offset);
      }
      if(lane == 0)
          pres_sum[blockIdx.y + (gridDim.y * blockIdx.x)] = val;
  }
} // end sum_pressure

///////////////////////////////////////////////////////////////////////////////
__global__
void final_reduce (const Real* __restrict__ input, Real* __restrict__ output, const int N)
{
    __shared__ Real sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = tid;
    Real sum = 0.0;

    while(i < N){
        sum += input[i];
        i += blockDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        output[0] = sdata[0];
} // end final_reduce

///////////////////////////////////////////////////////////////////////////////
__global__ 
void set_horz_pres_BCs (Real* __restrict__ pres_red, Real* __restrict__ pres_black) 
{
  int col = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  col = (col * 2) - 1;

  int NUM_2 = NUM >> 1;

  pres_black(col, 0) = pres_red(col, 1);
  pres_red(col + 1, 0) = pres_black(col + 1, 1);

  pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
  pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);
} // end set_horz_pres_BCs

//////////////////////////////////////////////////////////////////////////////
__global__
void set_vert_pres_BCs (Real* __restrict__ pres_red, Real* __restrict__ pres_black) 
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

  int NUM_2 = NUM >> 1;

  pres_black(0, row) = pres_red(1, row);
  pres_red(0, row) = pres_black(1, row);

  pres_black(NUM + 1, row) = pres_red(NUM, row);
  pres_red(NUM + 1, row) = pres_black(NUM, row);
} // end set_vert_pres_BCs

///////////////////////////////////////////////////////////////////////////////
__global__
void red_kernel (const Real dt,
                 const Real* __restrict__ F, 
                 const Real* __restrict__ G,
                 const Real* __restrict__ pres_black,
                       Real* __restrict__ pres_red) 
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  int NUM_2 = NUM >> 1;      

  Real p_ij = pres_red(col, row);

  Real p_im1j = pres_black(col - 1, row);
  Real p_ip1j = pres_black(col + 1, row);
  Real p_ijm1 = pres_black(col, row - (col & 1));
  Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

  Real rhs = (((F(col, (2 * row) - (col & 1))
          - F(col - 1, (2 * row) - (col & 1))) / d_dx)
      + ((G(col, (2 * row) - (col & 1))
          - G(col, (2 * row) - (col & 1) - 1)) / d_dy)) / dt;

  pres_red(col, row) = p_ij * (ONE - d_omega) + d_omega * 
    (((p_ip1j + p_im1j) / (d_dx * d_dx)) + ((p_ijp1 + p_ijm1) / (d_dy * d_dy)) - 
     rhs) / ((TWO / (d_dx * d_dx)) + (TWO / (d_dy * d_dy)));
} // end red_kernel

///////////////////////////////////////////////////////////////////////////////
__global__ 
void black_kernel (const Real dt,
                   const Real* __restrict__ F, 
                   const Real* __restrict__ G,
                   const Real* __restrict__ pres_red, 
                         Real* __restrict__ pres_black) 
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  int NUM_2 = NUM >> 1;

  Real p_ij = pres_black(col, row);

  Real p_im1j = pres_red(col - 1, row);
  Real p_ip1j = pres_red(col + 1, row);
  Real p_ijm1 = pres_red(col, row - ((col + 1) & 1));
  Real p_ijp1 = pres_red(col, row + (col & 1));

  Real rhs = (((F(col, (2 * row) - ((col + 1) & 1))
          - F(col - 1, (2 * row) - ((col + 1) & 1))) / d_dx)
      + ((G(col, (2 * row) - ((col + 1) & 1))
          - G(col, (2 * row) - ((col + 1) & 1) - 1)) / d_dy)) / dt;

  pres_black(col, row) = p_ij * (ONE - d_omega) + d_omega * 
    (((p_ip1j + p_im1j) / (d_dx * d_dx)) + ((p_ijp1 + p_ijm1) / (d_dy * d_dy)) - 
     rhs) / ((TWO / (d_dx * d_dx)) + (TWO / (d_dy * d_dy)));
} // end black_kernel

///////////////////////////////////////////////////////////////////////////////
__global__
void calc_residual (const Real dt,
                    const Real* __restrict__ F,
                    const Real* __restrict__ G, 
                    const Real* __restrict__ pres_red,
                    const Real* __restrict__ pres_black,
                          Real* __restrict__ res_array)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  int NUM_2 = NUM >> 1;

  Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res, res2;

  p_ij = pres_red(col, row);
  p_im1j = pres_black(col - 1, row);
  p_ip1j = pres_black(col + 1, row);
  p_ijm1 = pres_black(col, row - (col & 1));
  p_ijp1 = pres_black(col, row + ((col + 1) & 1));

  rhs = (((F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) / d_dx)
      +  ((G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) / d_dy)) / dt;

  res = ((p_ip1j - (TWO * p_ij) + p_im1j) / (d_dx * d_dx))
    + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (d_dy * d_dy)) - rhs;

  p_ij = pres_black(col, row);
  p_im1j = pres_red(col - 1, row);
  p_ip1j = pres_red(col + 1, row);
  p_ijm1 = pres_red(col, row - ((col + 1) & 1));
  p_ijp1 = pres_red(col, row + (col & 1));

  rhs = (((F(col, (2 * row) - ((col + 1) & 1)) - F(col - 1, (2 * row) - ((col + 1) & 1))) / d_dx)
      +  ((G(col, (2 * row) - ((col + 1) & 1)) - G(col, (2 * row) - ((col + 1) & 1) - 1)) / d_dy)) / dt;

  res2 = ((p_ip1j - (TWO * p_ij) + p_im1j) / (d_dx * d_dx))
    + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (d_dy * d_dy)) - rhs;

  __shared__ Real sum_cache[BLOCK_SIZE];

  sum_cache[threadIdx.x] = (res * res) + (res2 * res2);

  __syncthreads();

  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (threadIdx.x < i) {
      sum_cache[threadIdx.x] += sum_cache[threadIdx.x + i];
    }
    __syncthreads();
    i >>= 1;
  }

  if (threadIdx.x == 0) {
    res_array[blockIdx.y + (gridDim.y * blockIdx.x)] = sum_cache[0];
  }
} 

///////////////////////////////////////////////////////////////////////////////
__global__ 
void calculate_u (const Real dt,
                  const Real* __restrict__ F, 
                  const Real* __restrict__ pres_red,
                  const Real* __restrict__ pres_black, 
                        Real* __restrict__ u,
                        Real* __restrict__ max_u)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  __shared__ Real max_cache[BLOCK_SIZE];
  max_cache[threadIdx.x] = ZERO;

  int NUM_2 = NUM >> 1;
  Real new_u = ZERO;

  if (col != NUM) {

    Real p_ij, p_ip1j, new_u2;

    p_ij = pres_red(col, row);
    p_ip1j = pres_black(col + 1, row);

    new_u = F(col, (2 * row) - (col & 1)) - (dt * (p_ip1j - p_ij) / d_dx);
    u(col, (2 * row) - (col & 1)) = new_u;

    p_ij = pres_black(col, row);
    p_ip1j = pres_red(col + 1, row);

    new_u2 = F(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ip1j - p_ij) / d_dx);
    u(col, (2 * row) - ((col + 1) & 1)) = new_u2;

    new_u = fmax(fabs(new_u), fabs(new_u2));

    if ((2 * row) == NUM) {
      new_u = fmax(new_u, fabs( u(col, NUM + 1) ));
    }
  } else {
    new_u = fmax(fabs( u(NUM, (2 * row)) ), fabs( u(0, (2 * row)) ));
    new_u = fmax(fabs( u(NUM, (2 * row) - 1) ), new_u);
    new_u = fmax(fabs( u(0, (2 * row) - 1) ), new_u);

    new_u = fmax(fabs( u(NUM + 1, (2 * row)) ), new_u);
    new_u = fmax(fabs( u(NUM + 1, (2 * row) - 1) ), new_u);
  }

  max_cache[threadIdx.x] = new_u;

  __syncthreads();

  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (threadIdx.x < i) {
      max_cache[threadIdx.x] = fmax(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
    }
    __syncthreads();
    i >>= 1;
  }

  if (threadIdx.x == 0) {
    max_u[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
  }
} // end calculate_u

///////////////////////////////////////////////////////////////////////////////
__global__ 
void calculate_v (const Real dt,
                  const Real* __restrict__ G, 
                  const Real* __restrict__ pres_red,
                  const Real* __restrict__ pres_black, 
                        Real* __restrict__ v,
                        Real* __restrict__ max_v)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

  __shared__ Real max_cache[BLOCK_SIZE];
  max_cache[threadIdx.x] = ZERO;

  int NUM_2 = NUM >> 1;
  Real new_v = ZERO;

  if (row != NUM_2) {
    Real p_ij, p_ijp1, new_v2;

    p_ij = pres_red(col, row);
    p_ijp1 = pres_black(col, row + ((col + 1) & 1));

    new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / d_dy);
    v(col, (2 * row) - (col & 1)) = new_v;

    p_ij = pres_black(col, row);
    p_ijp1 = pres_red(col, row + (col & 1));

    new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / d_dy);
    v(col, (2 * row) - ((col + 1) & 1)) = new_v2;

    new_v = fmax(fabs(new_v), fabs(new_v2));

    if (col == NUM) {
      new_v = fmax(new_v, fabs( v(NUM + 1, (2 * row)) ));
    }

  } else {

    if ((col & 1) == 1) {
      Real p_ij = pres_red(col, row);
      Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

      new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / d_dy);
      v(col, (2 * row) - (col & 1)) = new_v;
    } else {
      Real p_ij = pres_black(col, row);
      Real p_ijp1 = pres_red(col, row + (col & 1));

      new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / d_dy);
      v(col, (2 * row) - ((col + 1) & 1)) = new_v;
    }

    new_v = fabs(new_v);

    new_v = fmax(fabs( v(col, NUM) ), new_v);
    new_v = fmax(fabs( v(col, 0) ), new_v);

    new_v = fmax(fabs( v(col, NUM + 1) ), new_v);
  }

  max_cache[threadIdx.x] = new_v;

  __syncthreads();

  int i = BLOCK_SIZE >> 1;
  while (i != 0) {
    if (threadIdx.x < i) {
      max_cache[threadIdx.x] = fmax(max_cache[threadIdx.x], max_cache[threadIdx.x + i]);
    }
    __syncthreads();
    i >>= 1;
  }

  if (threadIdx.x == 0) {
    max_v[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
  }
} // end calculate_v

///////////////////////////////////////////////////////////////////////////////

int main (int argc, char *argv[])
{
  int iter = 0;
  const int it_max = 1000000;

  const Real tol = 0.001;

  const Real time_start = 0.0;
  const Real time_end = 0.001;

  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM / 2) + 2) * (NUM + 2);

  Real* F;
  Real* u;
  Real* G;
  Real* v;

  F = (Real *) calloc (size, sizeof(Real));
  u = (Real *) calloc (size, sizeof(Real));
  G = (Real *) calloc (size, sizeof(Real));
  v = (Real *) calloc (size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  Real* pres_red;
  Real* pres_black;

  pres_red = (Real *) calloc (size_pres, sizeof(Real));
  pres_black = (Real *) calloc (size_pres, sizeof(Real));

  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  printf("Problem size: %d x %d \n", NUM, NUM);

  dim3 block_bcs_new(16, 16);
  dim3 grid_bcs_new((NUM + 2 + block_bcs_new.x - 1) / block_bcs_new.x,
                    (NUM + 2 + block_bcs_new.y - 1) / block_bcs_new.y);

  dim3 block_pr (BLOCK_SIZE, 1);
  dim3 grid_pr (NUM / (2 * BLOCK_SIZE), NUM);

  dim3 block_F (BLOCK_F_DIM, BLOCK_F_DIM);
  dim3 grid_F ((NUM + BLOCK_F_DIM - 1) / BLOCK_F_DIM, (NUM + BLOCK_F_DIM - 1) / BLOCK_F_DIM);

  dim3 block_G (BLOCK_G_DIM, BLOCK_G_DIM);
  dim3 grid_G ((NUM + BLOCK_G_DIM - 1) / BLOCK_G_DIM, (NUM + BLOCK_G_DIM - 1) / BLOCK_G_DIM);

  dim3 block_hpbc (BLOCK_SIZE, 1);
  dim3 grid_hpbc (NUM / (2 * BLOCK_SIZE), 1);

  dim3 block_vpbc (BLOCK_SIZE, 1);
  dim3 grid_vpbc (NUM / (2 * BLOCK_SIZE), 1);

  Real host_pres_sum = 0.0;

  Real* max_u_arr;
  Real* max_v_arr;
  int size_max = grid_pr.x * grid_pr.y;
  cudaHostAlloc((void**)&max_u_arr, size_max * sizeof(Real), cudaHostAllocDefault);
  cudaHostAlloc((void**)&max_v_arr, size_max * sizeof(Real), cudaHostAllocDefault);

  Real* pres_sum_d;
  Real* final_sum_d; 
  int size_res = grid_pr.x * grid_pr.y;
  cudaMalloc ((void**) &final_sum_d, sizeof(Real));

  Real* res;
  cudaHostAlloc((void**)&res, size_res * sizeof(Real), cudaHostAllocDefault);

  // Declare missing device pointers
  Real* res_d;
  Real* max_u_d;
  Real* max_v_d;

  cudaMalloc ((void**) &pres_sum_d, size_res * sizeof(Real));
  cudaMalloc ((void**) &res_d, size_res * sizeof(Real));
  cudaMalloc ((void**) &max_u_d, size_max * sizeof(Real));
  cudaMalloc ((void**) &max_v_d, size_max * sizeof(Real));

  set_BCs_host (u, v);

  Real max_u = SMALL;
  Real max_v = SMALL;
  #pragma unroll
  for (int col = 0; col < NUM + 2; ++col) {
    #pragma unroll
    for (int row = 1; row < NUM + 2; ++row) {
      max_u = fmax(max_u, fabs( u(col, row) ));
    }
  }
  #pragma unroll
  for (int col = 1; col < NUM + 2; ++col) {
    #pragma unroll
    for (int row = 0; row < NUM + 2; ++row) {
      max_v = fmax(max_v, fabs( v(col, row) ));
    }
  }

  Real* u_d;
  Real* F_d;
  Real* v_d;
  Real* G_d;

  Real* pres_red_d;
  Real* pres_black_d;

  cudaMalloc ((void**) &u_d, size * sizeof(Real));
  cudaMalloc ((void**) &F_d, size * sizeof(Real));
  cudaMalloc ((void**) &v_d, size * sizeof(Real));
  cudaMalloc ((void**) &G_d, size * sizeof(Real));

  cudaMalloc ((void**) &pres_red_d, size_pres * sizeof(Real));
  cudaMalloc ((void**) &pres_black_d, size_pres * sizeof(Real));

  cudaMemcpy (u_d, u, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (F_d, F, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (v_d, v, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (G_d, G, size * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (pres_red_d, pres_red, size_pres * sizeof(Real), cudaMemcpyHostToDevice);
  cudaMemcpy (pres_black_d, pres_black, size_pres * sizeof(Real), cudaMemcpyHostToDevice);

  // Copy host simulation constants to GPU constant memory.
  cudaMemcpyToSymbol(d_Re_num, &Re_num, sizeof(Real));
  cudaMemcpyToSymbol(d_omega, &omega, sizeof(Real));
  cudaMemcpyToSymbol(d_mix_param, &mix_param, sizeof(Real));
  cudaMemcpyToSymbol(d_gx, &gx, sizeof(Real));
  cudaMemcpyToSymbol(d_gy, &gy, sizeof(Real));
  cudaMemcpyToSymbol(d_dx, &dx, sizeof(Real));
  cudaMemcpyToSymbol(d_dy, &dy, sizeof(Real));
  cudaMemcpyToSymbol(d_tau, &tau, sizeof(Real));

  Real time = time_start;

  Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (time < time_end) {

    dt = fmin((dx / max_u), (dy / max_v));
    dt = tau * fmin(dt_Re, dt);

    if ((time + dt) >= time_end) {
      dt = time_end - time;
    }

    calculate_F <<<grid_F, block_F>>> (dt, u_d, v_d, F_d);
    calculate_G <<<grid_G, block_G>>> (dt, u_d, v_d, G_d);

    sum_pressure <<<grid_pr, block_pr>>> (pres_red_d, pres_black_d, pres_sum_d);
    final_reduce<<<1, BLOCK_SIZE>>>(pres_sum_d, final_sum_d, size_res);
    cudaMemcpy(&host_pres_sum, final_sum_d, sizeof(Real), cudaMemcpyDeviceToHost);

    Real p0_norm = sqrt(host_pres_sum / ((Real)(NUM * NUM)));
    if (p0_norm < 0.0001) {
      p0_norm = 1.0;
    }

    Real norm_L2;

    for (iter = 1; iter <= it_max; ++iter) {

      set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (pres_red_d, pres_black_d);
      set_vert_pres_BCs <<<grid_vpbc, block_vpbc>>> (pres_red_d, pres_black_d);

      red_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_black_d, pres_red_d);

      black_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d);

      calc_residual <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);

      cudaMemcpyAsync (res, res_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(0);

      norm_L2 = ZERO;
      #pragma unroll
      for (int i = 0; i < size_res; ++i) {
        norm_L2 += res[i];
      }

      norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

      if (norm_L2 < tol) {
        break;
      }  
    }

    printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

    calculate_u <<<grid_pr, block_pr>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
    cudaMemcpyAsync (max_u_arr, max_u_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);

    calculate_v <<<grid_pr, block_pr>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);
    cudaMemcpyAsync (max_v_arr, max_v_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);

    max_v = SMALL;
    max_u = SMALL;

    #pragma unroll
    for (int i = 0; i < size_max; ++i) {
      Real test_u = max_u_arr[i];
      max_u = fmax(max_u, test_u);

      Real test_v = max_v_arr[i];
      max_v = fmax(max_v, test_v);
    }

    set_BCs<<<grid_bcs_new, block_bcs_new>>>(u_d, v_d);

    time += dt;

  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);

  cudaMemcpy (u, u_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (v, v_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (pres_red, pres_red_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy (pres_black, pres_black_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost);

  FILE * pfile;
  pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {

        Real u_ij = u[(col * NUM) + row];
        Real u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        } else {
          u_im1j = u[(col - 1) * NUM + row];
        }

        u_ij = (u_ij + u_im1j) / 2.0;

        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        } else {
          v_ijm1 = v[(col * NUM) + row - 1];
        }

        v_ij = (v_ij + v_ijm1) / 2.0;

        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
      }
    }
  }

  fclose(pfile);

  cudaFree(u_d);
  cudaFree(v_d);
  cudaFree(F_d);
  cudaFree(G_d);
  cudaFree(pres_red_d);
  cudaFree(pres_black_d);
  cudaFree(max_u_d);
  cudaFree(max_v_d);
  cudaFree(pres_sum_d);
  cudaFree(res_d);
  cudaFree(final_sum_d);

  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  cudaFreeHost(max_u_arr);
  cudaFreeHost(max_v_arr);
  cudaFreeHost(res);

  return 0;
}
