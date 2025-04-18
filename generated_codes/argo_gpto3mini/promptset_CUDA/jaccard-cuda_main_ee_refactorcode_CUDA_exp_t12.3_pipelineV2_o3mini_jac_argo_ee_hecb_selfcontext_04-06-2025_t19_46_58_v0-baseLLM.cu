
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
 *
 * Refactored for energy efficiency by fusing kernel launches where possible,
 * reducing redundant synchronizations and memory copy overhead, and adding error
 * checking. The functionality remains identical.
 */

#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace std;

#define MAX_KERNEL_THREADS 256
#define mask 0xFFFFFFFF

// Error checking macro for CUDA calls.
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// float or double
typedef float vtype;
typedef vector<vector<vtype>> matrix;

// Parallel prefix sum using warp shuffle instructions.
// (The algorithm launches sufficient iterations to cover all n elements.)
template<typename T>
__device__ T parallel_prefix_sum(const int n, const int *ind, const T *w)
{
    T sum = 0.0;
    T last;
    // Round n up to a multiple of blockDim.x.
    int mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x);
    for (int i = threadIdx.x; i < mn; i += blockDim.x) {
        // Ensure all threads participate in the __shfl_sync instructions.
        bool valid = (i < n);
        last = __shfl_sync(mask, sum, blockDim.x - 1, blockDim.x);
        sum = valid ? w[ind[i]] : 0.0;
        for (int j = 1; j < blockDim.x; j *= 2) {
            T v = __shfl_up_sync(mask, sum, j, blockDim.x);
            if (threadIdx.x >= j) sum += v;
        }
        sum += last;
    }
    last = __shfl_sync(mask, sum, blockDim.x - 1, blockDim.x);
    return last;
}


// Kernel to compute the row sums (volume of neighbors).
// If weighted is true, the row sum is computed using prefix sum; otherwise the count is used.
template<bool weighted, typename T>
__global__ void jaccard_row_sum(const int n,
                                const int *__restrict__ csrPtr,
                                const int *__restrict__ csrInd,
                                const T *__restrict__ w,
                                T *__restrict__ work)
{
    for (int row = threadIdx.y + blockIdx.y * blockDim.y; row < n;
         row += gridDim.y * blockDim.y) {
        int start = csrPtr[row];
        int end   = csrPtr[row + 1];
        int length = end - start;
        if (weighted) {
            T sum = parallel_prefix_sum(length, csrInd + start, w);
            if (threadIdx.x == 0) work[row] = sum;
        }
        else {
            work[row] = static_cast<T>(length);
        }
    }
}

// Compute the volume of intersections (weight_i) and cumulated volume (weight_s)
// for each edge. Uses binary search on the sorted CSR indices.
template<bool weighted, typename T>
__global__ void jaccard_is(const int n, const int e,
                           const int *__restrict__ csrPtr,
                           const int *__restrict__ csrInd,
                           const T *__restrict__ v,
                           const T *__restrict__ work,
                           T *__restrict__ weight_i,
                           T *__restrict__ weight_s)
{
    for (int row = threadIdx.z + blockIdx.z * blockDim.z; row < n;
         row += gridDim.z * blockDim.z) {
        for (int j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y;
             j < csrPtr[row + 1];
             j += gridDim.y * blockDim.y) {
            int col = csrInd[j];
            // Determine the reference row (with the smaller number of elements).
            int Ni = csrPtr[row + 1] - csrPtr[row];
            int Nj = csrPtr[col + 1] - csrPtr[col];
            int ref = (Ni < Nj) ? row : col;
            int cur = (Ni < Nj) ? col : row;

            // Save cumulative volume from both rows.
            weight_s[j] = work[row] + work[col];

            // Look for intersection in the reference row.
            for (int i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x;
                 i < csrPtr[ref + 1];
                 i += gridDim.x * blockDim.x) {
                int match = -1;
                int ref_col = csrInd[i];
                T ref_val = weighted ? v[ref_col] : static_cast<T>(1.0);

                // Binary search in the current row.
                int left = csrPtr[cur];
                int right = csrPtr[cur + 1] - 1;
                while (left <= right) {
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
                if (match != -1) {
                    atomicAdd(&weight_i[j], ref_val);
                }
            }
        }
    }
}

// Compute final Jaccard weights.
template<bool weighted, typename T>
__global__ void jaccard_jw(const int e,
                           const T *__restrict__ csrVal,
                           const T gamma,
                           const T *__restrict__ weight_i,
                           const T *__restrict__ weight_s,
                           T *__restrict__ weight_j)
{
    for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e;
         j += gridDim.x * blockDim.x) {
        T Wi = weight_i[j];
        T Ws = weight_s[j];
        weight_j[j] = (gamma * csrVal[j]) * (Wi / (Ws - Wi));
    }
}

//---------------------------------------------------------------------
// Combined fill kernel to initialize two arrays.
// This fuses two separate kernel launches, reducing kernel launch overhead and power usage.
// For weight_j, if weighted==true the value is computed as (idx+1)/e; if false a constant value is used.
// For weight_i, always assign the provided constant value.
template<bool weighted, typename T>
__global__ void fill_two(const int e, T *w_j, T *w_i, T value_j, T value_i)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < e) {
        w_j[idx] = weighted ? static_cast<T>(idx + 1) / e : value_j;
        w_i[idx] = value_i;
    }
}

//---------------------------------------------------------------------
// Main function to execute and coordinate kernels.
// Minor modifications include fused initialization and error checking to reduce
// redundant operations and keep the GPU busy only as needed.
template <bool weighted, typename T>
void jaccard_weight (const int iteration, const int n, const int e,
                     int* csr_ptr, int* csr_ind, T* csr_val)
{
    const T gamma = static_cast<T>(0.46);

    T *d_weight_i, *d_weight_s, *d_weight_j, *d_work;
    int *d_csrInd, *d_csrPtr;
    T *d_csrVal;

#ifdef DEBUG
    T* weight_i = (T*) malloc(sizeof(T) * e);
    T* weight_s = (T*) malloc(sizeof(T) * e);
    T* work    = (T*) malloc(sizeof(T) * n);
#endif
    T* weight_j = (T*) malloc(sizeof(T) * e);

    CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(T) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_weight_i, sizeof(T) * e));
    CUDA_CHECK(cudaMalloc((void**)&d_weight_s, sizeof(T) * e));
    CUDA_CHECK(cudaMalloc((void**)&d_weight_j, sizeof(T) * e));
    CUDA_CHECK(cudaMalloc((void**)&d_csrVal, sizeof(T) * e));
    CUDA_CHECK(cudaMalloc((void**)&d_csrPtr, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrInd, sizeof(int) * e));

    CUDA_CHECK(cudaMemcpy(d_csrPtr, csr_ptr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice));

    // Ensure previous operations are finished before starting timer.
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int it = 0; it < iteration; it++) {
        // Reuse the kernel configuration variable.
        dim3 nthreads, nblocks;

        // Fuse two fill operations into one kernel call.
        nthreads.x = MAX_KERNEL_THREADS;
        nthreads.y = 1;
        nthreads.z = 1;
        nblocks.x = (e + MAX_KERNEL_THREADS - 1) / MAX_KERNEL_THREADS;
        // Initialize d_weight_j and d_weight_i concurrently.
        fill_two<weighted, T><<<nblocks, nthreads>>>(e, d_weight_j, d_weight_i,
                                                     static_cast<T>(1.0), static_cast<T>(0.0));
        CUDA_CHECK(cudaGetLastError());

        // Compute row sum (volume of neighbors) using prefix sum.
        const int y = 4;
        nthreads.x = 64 / y;
        nthreads.y = y;
        nthreads.z = 1;
        nblocks.x  = 1;
        nblocks.y  = (n + nthreads.y - 1) / nthreads.y;
        nblocks.z  = 1;
        jaccard_row_sum<weighted, T><<<nblocks, nthreads>>>(n, d_csrPtr, d_csrInd, d_weight_j, d_work);
        CUDA_CHECK(cudaGetLastError());

        // Compute volume of intersections and cumulative volume.
        nthreads.x = 32 / y;
        nthreads.y = y;
        nthreads.z = 8;
        nblocks.x  = 1;
        nblocks.y  = 1;
        nblocks.z  = (n + nthreads.z - 1) / nthreads.z;
        jaccard_is<weighted, T><<<nblocks, nthreads>>>(n, e, d_csrPtr, d_csrInd,
                                                       d_weight_j, d_work, d_weight_i, d_weight_s);
        CUDA_CHECK(cudaGetLastError());

        // Compute final jaccard weights.
        nthreads.x = min(e, MAX_KERNEL_THREADS);
        nthreads.y = 1;
        nthreads.z = 1;
        nblocks.x  = (e + nthreads.x - 1) / nthreads.x;
        nblocks.y  = 1;
        nblocks.z  = 1;
        jaccard_jw<weighted, T><<<nblocks, nthreads>>>(e, d_csrVal, gamma,
                                                       d_weight_i, d_weight_s, d_weight_j);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    cout << "Average execution time of kernels: " << (time * 1e-9f) / iteration << " (s)\n";

    CUDA_CHECK(cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost));

#ifdef DEBUG
    // Verification using known values.
    float error;
    if (weighted)
        error = fabs(weight_j[0] - 0.306667) +
                fabs(weight_j[1] - 0.000000) +
                fabs(weight_j[2] - 3.680000) +
                fabs(weight_j[3] - 1.380000) +
                fabs(weight_j[4] - 0.788571) +
                fabs(weight_j[5] - 0.460000);
    else
        error = fabs(weight_j[0] - 0.230000) +
                fabs(weight_j[1] - 0.000000) +
                fabs(weight_j[2] - 3.680000) +
                fabs(weight_j[3] - 1.380000) +
                fabs(weight_j[4] - 0.920000) +
                fabs(weight_j[5] - 0.460000);

    if (error > 1e-5) {
        for (int i = 0; i < e; i++)
            printf("wj: %d %f\n", i, weight_j[i]);
        printf("FAIL");
    }
    else {
        printf("PASS");
    }
    printf("\n");
#endif

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_weight_i));
    CUDA_CHECK(cudaFree(d_weight_s));
    CUDA_CHECK(cudaFree(d_weight_j));
    CUDA_CHECK(cudaFree(d_csrInd));
    CUDA_CHECK(cudaFree(d_csrVal));
    CUDA_CHECK(cudaFree(d_csrPtr));
    free(weight_j);
#ifdef DEBUG
    free((void*)NULL); // dummy, if additional host-side arrays were allocated.
#endif
}

//---------------------------------------------------------------------
// Utility functions to print a matrix and a vector.
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
    for (auto item : V)
        cout << item << " ";
    cout << "]" << endl;
}

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv)
{
    int iteration = 10;
#ifdef DEBUG
    matrix M = {
        { 0, 0, 0, 1 },
        { 5, 8, 0, 0 },
        { 0, 0, 3, 0 },
        { 0, 6, 0, 1 }
    };
#else
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <numRow> <numCol> <iteration>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
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
    // Build CSR representation.
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (M[i][j] != static_cast<vtype>(0)) {
                csr_val.push_back(M[i][j]);
                csr_ind.push_back(j);
                nnz++;
            }
        }
        csr_ptr.push_back(nnz);
    }

    // Print small matrices for verification.
    if (row <= 16 && col <= 16) {
        printMatrix(M);
        printVector(csr_val, "values = ");
        printVector(csr_ptr, "row pointer = ");
        printVector(csr_ind, "col indices = ");
    }

    // Execute the two variants.
    jaccard_weight<true, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
    jaccard_weight<false, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());

    return 0;
}
