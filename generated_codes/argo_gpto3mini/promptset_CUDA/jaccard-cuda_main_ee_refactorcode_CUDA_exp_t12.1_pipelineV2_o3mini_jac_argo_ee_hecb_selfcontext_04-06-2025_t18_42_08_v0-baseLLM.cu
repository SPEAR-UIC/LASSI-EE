
/*
 * Refactored for Energy Efficiency:
 * - Added __launch_bounds__ to kernels to restrict occupancy.
 * - Replaced chrono timing (active CPU polling) with CUDA events for lower CPU overhead.
 * - Used __ldg for read-only global memory fetches to reduce memory power usage.
 * - Combined some unnecessary synchronizations and kept kernel configurations minimal.
 *
 * All functionality is preserved.
 */

#include <cuda.h>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

#define MAX_KERNEL_THREADS 256
#define MASK 0xFFFFFFFF

// float or double
typedef float vtype;
typedef vector<vector<vtype>> matrix;

// Removed __launch_bounds__ from non-global function.
template<typename T>
__device__
T parallel_prefix_sum(const int n, const int *ind, const T *w)
{
    T sum = 0.0;
    T last;
    int mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x); // n rounded up to a multiple of blockDim.x
    for (int i = threadIdx.x; i < mn; i += blockDim.x) {
        // Ensure all threads participate for __shfl_sync.
        bool valid = (i < n);
        // Use __ldg for read-only accesses to global memory.
        last = __shfl_sync(MASK, sum, blockDim.x - 1, blockDim.x);
        sum = valid ? w[__ldg(&ind[i])] : 0.0;
        for (int j = 1; j < blockDim.x; j *= 2) {
            T v = __shfl_up_sync(MASK, sum, j, blockDim.x);
            if (threadIdx.x >= j)
                sum += v;
        }
        sum += last;
    }
    last = __shfl_sync(MASK, sum, blockDim.x - 1, blockDim.x);
    return last;
}

template<bool weighted, typename T>
__global__ __launch_bounds__(MAX_KERNEL_THREADS)
void jaccard_row_sum(const int n,
                     const int *__restrict__ csrPtr,
                     const int *__restrict__ csrInd,
                     const T *__restrict__ w,
                           T *__restrict__ work)
{
    // Each block processes a subset of rows.
    for (int row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
        int start = __ldg(&csrPtr[row]);
        int end   = __ldg(&csrPtr[row+1]);
        int length = end - start;
        if (weighted) {
            T sum = parallel_prefix_sum(length, csrInd + start, w);
            if (threadIdx.x == 0)
                work[row] = sum;
        }
        else {
            work[row] = (T)length;
        }
    }
}

template<bool weighted, typename T>
__global__ __launch_bounds__(MAX_KERNEL_THREADS)
void jaccard_is(const int n, const int e,
                const int *__restrict__ csrPtr,
                const int *__restrict__ csrInd,
                const T *__restrict__ v,
                const T *__restrict__ work,
                      T *__restrict__ weight_i,
                      T *__restrict__ weight_s)
{
    for (int row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
        for (int j = __ldg(&csrPtr[row]) + threadIdx.y + blockIdx.y * blockDim.y;
                 j < __ldg(&csrPtr[row + 1]);
                 j += gridDim.y * blockDim.y) {
            int col = __ldg(&csrInd[j]);
            int Ni = __ldg(&csrPtr[row + 1]) - __ldg(&csrPtr[row]);
            int Nj = __ldg(&csrPtr[col + 1]) - __ldg(&csrPtr[col]);
            int ref = (Ni < Nj) ? row : col;
            int cur = (Ni < Nj) ? col : row;
            weight_s[j] = work[row] + work[col];
            // For each element in the reference row, search in the other row.
            for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
                     i < __ldg(&csrPtr[ref+1]) - __ldg(&csrPtr[ref]); 
                     i += gridDim.x * blockDim.x) {
                int ref_idx = __ldg(&csrPtr[ref]) + i;
                int ref_col = __ldg(&csrInd[ref_idx]);
                T ref_val = weighted ? v[ref_col] : (T)1.0;
                int left = __ldg(&csrPtr[cur]);
                int right = __ldg(&csrPtr[cur+1]) - 1;
                int match = -1;
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
__global__ __launch_bounds__(MAX_KERNEL_THREADS)
void jaccard_jw(const int e,
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

template<bool weighted, typename T>
__global__ __launch_bounds__(MAX_KERNEL_THREADS)
void fill(const int e, T* w, const T value)
{
    for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
        w[j] = weighted ? (T)(j + 1) / e : value;
    }
}

template<bool weighted, typename T>
void jaccard_weight(const int iteration, const int n, const int e,
                    int* csr_ptr, int* csr_ind, T* csr_val)
{
    const T gamma = (T)0.46;  // arbitrary

    T *d_weight_i, *d_weight_s, *d_weight_j, *d_work;
    int *d_csrInd;
    int *d_csrPtr;
    T *d_csrVal;

#ifdef DEBUG
    T* weight_i = (T*) malloc(sizeof(T) * e);
    T* weight_s = (T*) malloc(sizeof(T) * e);
    T* work = (T*) malloc(sizeof(T) * n);
#endif
    T* weight_j = (T*) malloc(sizeof(T) * e);

    cudaMalloc((void**)&d_work, sizeof(T) * n);
    cudaMalloc((void**)&d_weight_i, sizeof(T) * e);
    cudaMalloc((void**)&d_weight_s, sizeof(T) * e);
    cudaMalloc((void**)&d_weight_j, sizeof(T) * e);
    cudaMalloc((void**)&d_csrVal, sizeof(T) * e);
    cudaMalloc((void**)&d_csrPtr, sizeof(int) * (n + 1));
    cudaMalloc((void**)&d_csrInd, sizeof(int) * e);

    cudaMemcpy(d_csrPtr, csr_ptr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice);

    // Use CUDA events for timing to reduce CPU busy waiting.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iteration; i++) {
        dim3 nthreads, nblocks;

        // Kernel: fill weight_j
        nthreads.x = MAX_KERNEL_THREADS;
        nthreads.y = nthreads.z = 1;
        nblocks.x = (e + MAX_KERNEL_THREADS - 1) / MAX_KERNEL_THREADS;
        nblocks.y = nblocks.z = 1;
        fill<weighted, T><<<nblocks, nthreads>>>(e, d_weight_j, (T)1.0);
#ifdef DEBUG
        cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost);
        for (int k = 0; k < e; k++) printf("wj: %d %f\n", k, weight_j[k]);
#endif

        // Kernel: initialize volume of intersections
        fill<false, T><<<nblocks, nthreads>>>(e, d_weight_i, (T)0.0);

        // Kernel: jaccard_row_sum, using a 2D grid configuration.
        const int y = 4;
        nthreads.x = 64 / y;
        nthreads.y = y;
        nthreads.z = 1;
        nblocks.x = 1;
        nblocks.y = (n + nthreads.y - 1) / nthreads.y;
        nblocks.z = 1;
        jaccard_row_sum<weighted, T><<<nblocks, nthreads>>>(n, d_csrPtr, d_csrInd, d_weight_j, d_work);

        // Kernel: jaccard_is, using a 3D grid configuration.
        nthreads.x = 32 / y;
        nthreads.y = y;
        nthreads.z = 8;
        nblocks.x = 1;
        nblocks.y = 1;
        nblocks.z = (n + nthreads.z - 1) / nthreads.z;
        jaccard_is<weighted, T><<<nblocks, nthreads>>>(n, e, d_csrPtr, d_csrInd, d_weight_j, d_work, d_weight_i, d_weight_s);

        // Kernel: jaccard_jw for final weights computation.
        nthreads.x = min(e, MAX_KERNEL_THREADS);
        nthreads.y = nthreads.z = 1;
        nblocks.x = (e + nthreads.x - 1) / nthreads.x;
        nblocks.y = nblocks.z = 1;
        jaccard_jw<weighted, T><<<nblocks, nthreads>>>(e, d_csrVal, gamma, d_weight_i, d_weight_s, d_weight_j);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Average execution time of kernels: " << (milliseconds / 1000.0f) / iteration << " (s)\n";

    cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost);
#ifdef DEBUG
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
        for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
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
    free(work);
    free(weight_i);
    free(weight_s);
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Utility function: print a matrix.
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

// Utility function: print a vector.
template <typename T>
void printVector(const vector<T>& V, const char* msg)
{
    cout << msg << "[ ";
    for (const auto &a : V)
        cout << a << " ";
    cout << "]" << endl;
}

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv)
{
    int iteration = 10;

#ifdef DEBUG
    matrix M = {
        {0, 0, 0, 1},
        {5, 8, 0, 0},
        {0, 0, 3, 0},
        {0, 6, 0, 1}
    };
#else
    if (argc < 4) {
        printf("Usage: %s <numRow> <numCol> <iteration>\n", argv[0]);
        return -1;
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
    vector<int> csr_ptr = {0}; // starting pointer
    vector<int> csr_ind;
    int nnz = 0; // number of non-zero elements

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

    // Optionally print matrix and CSR representation if small.
    if (row <= 16 && col <= 16) {
        printMatrix(M);
        printVector(csr_val, "values = ");
        printVector(csr_ptr, "row pointer = ");
        printVector(csr_ind, "col indices = ");
    }

    jaccard_weight<true, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
    jaccard_weight<false, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());

    return 0;
}
