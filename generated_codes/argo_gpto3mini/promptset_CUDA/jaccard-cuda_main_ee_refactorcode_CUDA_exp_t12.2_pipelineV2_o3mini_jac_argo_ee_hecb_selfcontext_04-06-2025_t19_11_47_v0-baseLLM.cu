
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
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace std;

#define MAX_KERNEL_THREADS 256
#define mask 0xFFFFFFFF

// Simple error-checking macro
#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

// float or double 
typedef float vtype;
typedef vector<vector<vtype>> matrix;

//---------------------------------------------------------------------
// Using __ldg() for read-only accesses can reduce energy by enabling
// read-only cache paths.
template<typename T>
__device__
T parallel_prefix_sum(const int n, const int *ind, const T *w)
{
    T sum = 0.0;
    T last;
    int mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x); // round-up n to multiple of blockDim.x
    for (int i = threadIdx.x; i < mn; i += blockDim.x) {
        bool valid = (i < n);
        // propagate sum from previous iteration
        last = __shfl_sync(mask, sum, blockDim.x - 1, blockDim.x);
        // use read-only load to lower energy consumption on global memory accesses
        sum = valid ? __ldg(&w[ind[i]]) : 0.0;
        // intra-warp prefix sum
        for (int j = 1; j < blockDim.x; j *= 2) {
            T v = __shfl_up_sync(mask, sum, j, blockDim.x);
            if (threadIdx.x >= j)
                sum += v;
        }
        sum += last;
    }
    last = __shfl_sync(mask, sum, blockDim.x - 1, blockDim.x);
    return last;
}

//---------------------------------------------------------------------
// Fused kernel for initializing weight_j and weight_i.
// This reduces the number of kernel launches and thereby saves energy.
template <bool weighted, typename T>
__global__ void initialize_weights(const int e, T* weight_j, T* weight_i, const T fill_val)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < e)
    {
        weight_j[j] = weighted ? ((T)(j + 1) / (T)e) : fill_val;
        weight_i[j] = (T)0.0;
    }
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
        int end = csrPtr[row + 1];
        int length = end - start;
        // compute row sum using prefix sum if weighted, else count number of non-zero elements
        if (weighted) {
            T sum = parallel_prefix_sum(length, csrInd + start, w);
            if (threadIdx.x == 0) work[row] = sum;
        }
        else {
            work[row] = (T)length;
        }
    }
}

// Volume of intersections (*weight_i) and cumulated volume of neighbors (*weight_s)
// Note: number of columns is constrained by the number of rows.
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
                 j < csrPtr[row + 1]; j += gridDim.y * blockDim.y) {
            int col = csrInd[j];
            int Ni = csrPtr[row + 1] - csrPtr[row];
            int Nj = csrPtr[col + 1] - csrPtr[col];
            int ref = (Ni < Nj) ? row : col;
            int cur = (Ni < Nj) ? col : row;

            weight_s[j] = work[row] + work[col];

            // search for a match using binary search in the 'cur' row.
            for (int i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x;
                     i < csrPtr[ref + 1]; i += gridDim.x * blockDim.x) {
                int match = -1;
                int ref_col = __ldg(&csrInd[i]);
                T ref_val = weighted ? __ldg(&v[ref_col]) : (T)1.0;

                int left = csrPtr[cur];
                int right = csrPtr[cur + 1] - 1;
                while (left <= right) {
                    int middle = (left + right) >> 1;
                    int cur_col = __ldg(&csrInd[middle]);
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
                if (match != -1) {
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
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < e) {
        T Wi = weight_i[j];
        T Ws = weight_s[j];
        // Compute jaccard weight.
        weight_j[j] = (gamma * csrVal[j]) * (Wi / (Ws - Wi));
    }
}

template <bool weighted, typename T>
void jaccard_weight(const int iteration, const int n, const int e,
                    int* csr_ptr, int* csr_ind, T* csr_val)
{
    const T gamma = (T)0.46;  // arbitrary constant

    T *d_weight_i, *d_weight_s, *d_weight_j, *d_work;
    int *d_csrInd, *d_csrPtr;
    T *d_csrVal;

#ifdef DEBUG
    T* weight_i = (T*) malloc(sizeof(T) * e);
    T* weight_s = (T*) malloc(sizeof(T) * e);
    T* work = (T*) malloc(sizeof(T) * n);
#endif
    T* weight_j = (T*) malloc(sizeof(T) * e);

    // Allocate device memory.
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(T) * n));
    CHECK_CUDA(cudaMalloc((void**)&d_weight_i, sizeof(T) * e));
    CHECK_CUDA(cudaMalloc((void**)&d_weight_s, sizeof(T) * e));
    CHECK_CUDA(cudaMalloc((void**)&d_weight_j, sizeof(T) * e));
    CHECK_CUDA(cudaMalloc((void**)&d_csrVal, sizeof(T) * e));
    CHECK_CUDA(cudaMalloc((void**)&d_csrPtr, sizeof(int) * (n + 1)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrInd, sizeof(int) * e));

    CHECK_CUDA(cudaMemcpy(d_csrPtr, csr_ptr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice));

    // Set device configuration to prefer L1 cache, reducing energy by lowering memory traffic.
    CHECK_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Wait for transfers to finish.
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iteration; i++) {
        dim3 nthreads, nblocks; // reusable launch configuration

        // Instead of launching two separate fill kernels,
        // fuse the initialization for weight_j and weight_i.
        nthreads.x = MAX_KERNEL_THREADS;
        nthreads.y = 1;
        nthreads.z = 1;
        nblocks.x = (e + MAX_KERNEL_THREADS - 1) / MAX_KERNEL_THREADS;
        nblocks.y = 1;
        nblocks.z = 1;
        // For weight_j: fill value is 1.0 when not weighted.
        initialize_weights<weighted, T><<<nblocks, nthreads>>>(e, d_weight_j, d_weight_i, (T)1.0);
        // Also initialize weight_s to zeros.
        CHECK_CUDA(cudaMemset(d_weight_s, 0, sizeof(T) * e));

        // Compute row sums.
        const int y = 4;
        nthreads.x = 64 / y;
        nthreads.y = y;
        nthreads.z = 1;
        nblocks.x = 1;
        nblocks.y = (n + nthreads.y - 1) / nthreads.y;
        nblocks.z = 1;
        jaccard_row_sum<weighted, T><<<nblocks, nthreads>>>(n, d_csrPtr, d_csrInd, d_weight_j, d_work);

        // Compute volume of intersections and cumulative volume of neighbors.
        nthreads.x = 32 / y;
        nthreads.y = y;
        nthreads.z = 8;
        nblocks.x = 1;
        nblocks.y = 1;
        nblocks.z = (n + nthreads.z - 1) / nthreads.z;
        jaccard_is<weighted, T><<<nblocks, nthreads>>>(n, e, d_csrPtr,
            d_csrInd, d_weight_j, d_work, d_weight_i, d_weight_s);

        // Compute jaccard weights.
        nthreads.x = std::min(e, MAX_KERNEL_THREADS);
        nthreads.y = 1;
        nthreads.z = 1;
        nblocks.x = (e + nthreads.x - 1) / nthreads.x;
        nblocks.y = 1;
        nblocks.z = 1;
        jaccard_jw<weighted, T><<<nblocks, nthreads>>>(e,
            d_csrVal, gamma, d_weight_i, d_weight_s, d_weight_j);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cout << "Average execution time of kernels: " << (time * 1e-9f) / iteration << " (s)\n";

    CHECK_CUDA(cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost));

#ifdef DEBUG
    // Verification for debugging (example expected values).
    float error;
    if (weighted)
        error = std::fabs(weight_j[0] - 0.306667f) +
                std::fabs(weight_j[1] - 0.000000f) +
                std::fabs(weight_j[2] - 3.680000f) +
                std::fabs(weight_j[3] - 1.380000f) +
                std::fabs(weight_j[4] - 0.788571f) +
                std::fabs(weight_j[5] - 0.460000f);
    else
        error = std::fabs(weight_j[0] - 0.230000f) +
                std::fabs(weight_j[1] - 0.000000f) +
                std::fabs(weight_j[2] - 3.680000f) +
                std::fabs(weight_j[3] - 1.380000f) +
                std::fabs(weight_j[4] - 0.920000f) +
                std::fabs(weight_j[5] - 0.460000f);

    if (error > 1e-5) {
        for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
        printf("FAIL");
    } else {
        printf("PASS");
    }
    printf("\n");
#endif

    // Free device memory.
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_weight_i));
    CHECK_CUDA(cudaFree(d_weight_s));
    CHECK_CUDA(cudaFree(d_weight_j));
    CHECK_CUDA(cudaFree(d_csrInd));
    CHECK_CUDA(cudaFree(d_csrVal));
    CHECK_CUDA(cudaFree(d_csrPtr));
    free(weight_j);
#ifdef DEBUG
    free(weight_i);
    free(weight_s);
    free(work);
#endif
}

//---------------------------------------------------------------------
// Utility functions.
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
void printVector(const vector<T>& V, const char* msg)
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
    matrix M = {
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
        printVector(csr_val, "values = ");
        printVector(csr_ptr, "row pointer = ");
        printVector(csr_ind, "col indices = ");
    }

    // Launch both weighted and non-weighted versions.
    jaccard_weight<true, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
    jaccard_weight<false, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
    return 0;
}
