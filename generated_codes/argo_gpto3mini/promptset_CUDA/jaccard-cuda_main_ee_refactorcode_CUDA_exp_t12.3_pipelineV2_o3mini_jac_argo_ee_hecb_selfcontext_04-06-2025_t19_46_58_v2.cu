
/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <stdio.h>
#include <algorithm> 
#include <iostream> 
#include <vector> 
#include <chrono>

using namespace std; 

#define MAX_KERNEL_THREADS 256
#define mask 0xFFFFFFFF

// float or double 
typedef float vtype;
typedef vector<vector<vtype>> matrix; 

template<typename T>
__device__
T parallel_prefix_sum(const int n, const int *ind, const T *w) 
{

  T sum = 0.0;
  T last;

  int mn = (((n+blockDim.x-1)/blockDim.x)*blockDim.x); // n padded to multiple of blockDim.x
  for (int i = threadIdx.x; i < mn; i += blockDim.x) {
    // All threads participate in the shfl instruction.
    bool valid = i < n;
    // Get the value from the last thread in previous iteration.
    last = __shfl_sync(mask, sum, blockDim.x - 1, blockDim.x);
    // Read value or set to zero if invalid.
    sum = (valid) ? w[ind[i]] : 0.0;
    // Do prefix sum (for warp size = blockDim.x <= 32).
    for (int j = 1; j < blockDim.x; j *= 2) {
      T v = __shfl_up_sync(mask, sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    // Accumulate with last value.
    sum += last;
  }
  last = __shfl_sync(mask, sum, blockDim.x - 1, blockDim.x);
  return last;
}

// Volume of neighbors (*weight_s)
template<bool weighted, typename T>
__global__ void 
jaccard_row_sum(const int n,
                const int *__restrict__ csrPtr,
                const int *__restrict__ csrInd,
                const T *__restrict__ w,
                      T *__restrict__ work)
{
  for (int row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
    int start = csrPtr[row];
    int end   = csrPtr[row+1];
    int length = end - start;
    if (weighted) {
      T sum = parallel_prefix_sum(length, csrInd + start, w); 
      if (threadIdx.x == 0) work[row] = sum;
    }
    else {
      work[row] = (T)length;
    }
  }
}

// Volume of intersections (*weight_i) and cumulative volume of neighbors (*weight_s)
template<bool weighted, typename T>
__global__ void
jaccard_is(const int n, const int e,
           const int *__restrict__ csrPtr,
           const int *__restrict__ csrInd,
           const T *__restrict__ v,
           const T *__restrict__ work,
                 T *__restrict__ weight_i,
                 T *__restrict__ weight_s) 
{
  for (int row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {  
    for (int j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y;
             j < csrPtr[row+1]; j += gridDim.y * blockDim.y) { 
      int col = csrInd[j];
      int Ni = csrPtr[row+1] - csrPtr[row];
      int Nj = csrPtr[col+1] - csrPtr[col];
      int ref = (Ni < Nj) ? row : col;
      int cur = (Ni < Nj) ? col : row;
      weight_s[j] = work[row] + work[col];

      for (int i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x;
           i < csrPtr[ref+1]; i += gridDim.x * blockDim.x) {
        int match = -1;           
        int ref_col = csrInd[i];
        T ref_val = weighted ? v[ref_col] : (T)1.0;

        // Binary search in the current row.
        int left = csrPtr[cur]; 
        int right = csrPtr[cur+1] - 1; 
        while (left <= right){
          int middle = (left + right) >> 1; 
          int cur_col = csrInd[middle];
          if (cur_col > ref_col) {
            right = middle - 1;
          }
          else if (cur_col < ref_col) {
            left = middle + 1;
          }
          else {
            match = middle; 
            break; 
          }
        }
        if (match != -1){
          atomicAdd(&weight_i[j], ref_val);
        }
      }
    }
  }
}

template<bool weighted, typename T>
__global__ void 
jaccard_jw(const int e,
    const T *__restrict__ csrVal,
    const T gamma,
    const T *__restrict__ weight_i,
    const T *__restrict__ weight_s,
          T *__restrict__ weight_j) 
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {  
    T Wi = weight_i[j];
    T Ws = weight_s[j];
    weight_j[j] = (gamma * csrVal[j]) * (Wi / (Ws - Wi));
  }
}

template <bool weighted, typename T>
__global__ void 
fill(const int e, T* w, const T value) 
{
  for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {  
    w[j] = weighted ? (T)(j + 1) / e : value; 
  }
}

// Helper function: returns optimal block dimension, a multiple of 32, not exceeding maxThreads.
inline int getOptimalBlockDim(const int N, const int maxThreads = MAX_KERNEL_THREADS) {
  if (N >= maxThreads) return maxThreads;
  // Round up to the next multiple of 32 (warp size)
  return ((N + 31) / 32) * 32;
}

template <bool weighted, typename T>
void jaccard_weight (const int iteration, const int n, const int e, 
    int* csr_ptr, int* csr_ind, T* csr_val)
{
  const T gamma = (T)0.46;  // arbitrary

  T *d_weight_i, *d_weight_s, *d_weight_j, *d_work;
  int *d_csrInd;
  int *d_csrPtr;
  T *d_csrVal;

#ifdef DEBUG
  T *weight_i, *weight_s, *work;
  // Allocate pinned memory for asynchronous debug copies
  cudaMallocHost((void**)&weight_i, sizeof(T) * e);
  cudaMallocHost((void**)&weight_s, sizeof(T) * e);
  cudaMallocHost((void**)&work, sizeof(T) * n);
#endif
  T* weight_j = (T*) malloc (sizeof(T) * e);

  cudaMalloc((void**)&d_work, sizeof(T) * n);
  cudaMalloc((void**)&d_weight_i, sizeof(T) * e);
  cudaMalloc((void**)&d_weight_s, sizeof(T) * e);
  cudaMalloc((void**)&d_weight_j, sizeof(T) * e);
  cudaMalloc((void**)&d_csrVal, sizeof(T) * e);
  cudaMalloc((void**)&d_csrPtr, sizeof(int) * (n+1));
  cudaMalloc((void**)&d_csrInd, sizeof(int) * e);

  cudaMemcpy(d_csrPtr, csr_ptr, sizeof(int) * (n+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Create a stream for DEBUG asynchronous copies
#ifdef DEBUG
  cudaStream_t debugStream;
  cudaStreamCreate(&debugStream);
#endif

  for (int i = 0; i < iteration; i++) {
    dim3 nthreads, nblocks;

    // Launch fill kernel for weight_j
    int optimalThreads = getOptimalBlockDim(e);
    nthreads.x = optimalThreads;
    nthreads.y = 1; 
    nthreads.z = 1; 
    nblocks.x  = (e + optimalThreads - 1) / optimalThreads;
    nblocks.y  = 1;
    nblocks.z  = 1; 

    fill<weighted, T><<<nblocks, nthreads>>>(e, d_weight_j, (T)1.0);
#ifdef DEBUG
    cudaMemcpyAsync(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost, debugStream);
    cudaStreamSynchronize(debugStream);
    for (int i = 0; i < e; i++) 
      printf("wj: %d %f\n", i, weight_j[i]);
#endif

    // Initialize volume of intersections
    fill<false, T><<<nblocks, nthreads>>>(e, d_weight_i, (T)0.0);

    // Compute row sum with prefix sum
    const int y = 4;
    nthreads.x = 64 / y;
    nthreads.y = y; 
    nthreads.z = 1; 
    nblocks.x  = 1; 
    nblocks.y  = (n + nthreads.y - 1) / nthreads.y;
    nblocks.z  = 1; 
    jaccard_row_sum<weighted,T><<<nblocks, nthreads>>>(n, d_csrPtr, d_csrInd, d_weight_j, d_work);

#ifdef DEBUG
    cudaMemcpyAsync(work, d_work, sizeof(T) * n, cudaMemcpyDeviceToHost, debugStream);
    cudaStreamSynchronize(debugStream);
    for (int i = 0; i < n; i++) 
      printf("work: %d %f\n", i, work[i]);
#endif

    // Compute volume of intersections (*weight_i) and cumulative volume of neighbors (*weight_s)
    nthreads.x = 32 / y;
    nthreads.y = y;
    nthreads.z = 8;
    nblocks.x  = 1;
    nblocks.y  = 1;
    nblocks.z  = (n + nthreads.z - 1) / nthreads.z;
    jaccard_is<weighted,T><<<nblocks, nthreads>>>(n, e, d_csrPtr, d_csrInd, d_weight_j, d_work, d_weight_i, d_weight_s);

#ifdef DEBUG
    cudaMemcpyAsync(weight_i, d_weight_i, sizeof(T) * e, cudaMemcpyDeviceToHost, debugStream);
    cudaMemcpyAsync(weight_s, d_weight_s, sizeof(T) * e, cudaMemcpyDeviceToHost, debugStream);
    cudaStreamSynchronize(debugStream);
    for (int i = 0; i < e; i++) 
      printf("wi: %d %f\n", i, weight_i[i]);
    for (int i = 0; i < e; i++) 
      printf("ws: %d %f\n", i, weight_s[i]);
#endif

    // Compute jaccard weights with optimized thread block configuration
    optimalThreads = getOptimalBlockDim(e);
    nthreads.x = optimalThreads;
    nthreads.y = 1; 
    nthreads.z = 1;  
    nblocks.x  = (e + optimalThreads - 1) / optimalThreads;
    nblocks.y  = 1; 
    nblocks.z  = 1;
    jaccard_jw<weighted,T><<<nblocks, nthreads>>>(e, d_csrVal, gamma, d_weight_i, d_weight_s, d_weight_j);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cout << "Average execution time of kernels: " << (time * 1e-9f) / iteration << " (s)\n";

  cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    cudaMemcpyAsync(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost, debugStream);
    cudaStreamSynchronize(debugStream);
    float error; 
    if (weighted)
      error = std::fabs(weight_j[0] - 0.306667) +
              std::fabs(weight_j[1] - 0.000000) +
              std::fabs(weight_j[2] - 3.680000) +
              std::fabs(weight_j[3] - 1.380000) +
              std::fabs(weight_j[4] - 0.788571) +
              std::fabs(weight_j[5] - 0.460000);
    else
      error = std::fabs(weight_j[0] - 0.230000) +
              std::fabs(weight_j[1] - 0.000000) +
              std::fabs(weight_j[2] - 3.680000) +
              std::fabs(weight_j[3] - 1.380000) +
              std::fabs(weight_j[4] - 0.920000) +
              std::fabs(weight_j[5] - 0.460000);
    if (error > 1e-5) {
      for (int i = 0; i < e; i++) 
        printf("wj: %d %f\n", i, weight_j[i]);
      printf("FAIL");
    } else {
      printf("PASS");
    }
    printf("\n");
#endif

  cudaFree(d_work);
  cudaFree(d_weight_i);
  cudaFree(d_weight_s);
  cudaFree(d_weight_j);
  cudaFree(d_csrInd);
  cudaFree(d_csrVal);
  cudaFree(d_csrPtr);
  free(weight_j);
#ifdef DEBUG
  cudaFreeHost(weight_i);
  cudaFreeHost(weight_s);
  cudaFreeHost(work);
  cudaStreamDestroy(debugStream);
#endif
}

// Utilities
void printMatrix(const matrix& M) 
{ 
  int m = M.size(); 
  int n = M[0].size(); 
  for (int i = 0; i < m; i++) { 
    for (int j = 0; j < n; j++) 
      cout << M[i][j] << " ";     
    cout << endl; 
  } 
} 

template <typename T>
void printVector(const vector<T>& V, char* msg) 
{ 
  cout << msg << "[ "; 
  for_each(V.begin(), V.end(), [](int a) { cout << a << " "; }); 
  cout << "]" << endl; 
} 

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv) 
{ 
  int iteration = 10;

#ifdef DEBUG
  matrix M  = { 
    { 0, 0, 0, 1}, 
    { 5, 8, 0, 0}, 
    { 0, 0, 3, 0}, 
    { 0, 6, 0, 1} 
  }; 
#else
  int numRow = atoi(argv[1]);
  int numCol = atoi(argv[2]);
  iteration = atoi(argv[3]);

  srand(2);

  matrix M;
  vector<vtype> rowElems(numCol);
  for (int r = 0; r < numRow; r++) {
    for (int c = 0; c < numCol; c++)
      rowElems[c] = rand() % 10;
    M.push_back(rowElems);
  }
#endif

  int row = M.size();
  int col = M[0].size();
  printf("Number of matrix rows and cols: %d %d\n", row, col);
  vector<vtype> csr_val;
  vector<int> csr_ptr = { 0 };
  vector<int> csr_ind;
  int nnz = 0;

  for (int i = 0; i < row; i++) { 
    for (int j = 0; j < col; j++) { 
      if (M[i][j] != (vtype)0) { 
        csr_val.push_back(M[i][j]); 
        csr_ind.push_back(j); 
        nnz++; 
      } 
    } 
    csr_ptr.push_back(nnz); 
  } 

  if (row <= 16 && col <= 16) {
    printMatrix(M); 
    printVector(csr_val, (char*)"values = "); 
    printVector(csr_ptr, (char*)"row pointer = "); 
    printVector(csr_ind, (char*)"col indices = "); 
  }

  jaccard_weight<true, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
  jaccard_weight<false, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());

  return 0; 
} 
