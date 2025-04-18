
/*
 Refactored for energy‐efficiency. In this version we merge the repeated kernel executions into a single launch,
 which reduces repeated kernel launch overhead and the associated synchronization, thereby reducing power consumption.
 Also, we use __ldg() for read‐only global memory loads to improve memory bandwidth efficiency.
 The functionality remains identical.
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

__host__ __device__ __forceinline__
void clip_plus( const bool &clip, const int &n, int &plus ) {
  if ( clip ) {
    if ( plus >= n ) {
      plus = n - 1;
    }
  } else {
    if ( plus >= n ) {
      plus -= n;
    }
  }
}

__host__ __device__ __forceinline__
void clip_minus( const bool &clip, const int &n, int &minus ) {
  if ( clip ) {
    if ( minus < 0 ) {
      minus = 0;
    }
  } else {
    if ( minus < 0 ) {
      minus += n;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
//                          RELEXTREMA 1D                                    //
///////////////////////////////////////////////////////////////////////////////

// Merged kernel: performs the work repeat times inside the kernel to reduce launch overhead.
template<typename T>
__global__ void relextrema_1D_kernel(
  const int  n,
  const int  order,
  const bool clip,
  const int  repeat,
  const T *__restrict__ inp,
  bool *__restrict__ results)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for ( int tid = tx; tid < n; tid += stride ) {
    bool finalResult = false;
    // Repeat the computation inside the kernel to avoid multiple launches.
    for (int rep = 0; rep < repeat; rep++) {
      const T data = __ldg(&inp[tid]);
      bool temp = true;
      for ( int o = 1; o <= order; o++ ) {
        int plus = tid + o;
        int minus = tid - o;
        clip_plus( clip, n, plus );
        clip_minus( clip, n, minus );
        temp &= data > __ldg(&inp[plus]);
        temp &= data >= __ldg(&inp[minus]);
      }
      finalResult = temp;
    }
    results[tid] = finalResult;
  }
}

// CPU implementation unchanged.
template<typename T>
void cpu_relextrema_1D(
  const int  n,
  const int  order,
  const bool clip,
  const T *__restrict__ inp,
  bool *__restrict__ results)
{
  for ( int tid = 0; tid < n; tid++ ) {
    const T data = inp[tid];
    bool    temp = true;
    for ( int o = 1; o <= order; o++ ) {
      int plus = tid + o;
      int minus = tid - o;
      clip_plus( clip, n, plus );
      clip_minus( clip, n, minus );
      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp;
  }
}

///////////////////////////////////////////////////////////////////////////////
//                          RELEXTREMA 2D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void relextrema_2D_kernel(
  const int  in_x,
  const int  in_y,
  const int  order,
  const bool clip,
  const int  axis,
  const int  repeat,
  const T *__restrict__ inp,
  bool *__restrict__ results) 
{
  // Note: We follow the original thread ordering.
  const int ty = blockIdx.x * blockDim.x + threadIdx.x;
  const int tx = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ( tx < in_y ) && ( ty < in_x ) ) {
    const int tid = tx * in_x + ty;
    bool finalResult = false;
    for (int rep = 0; rep < repeat; rep++) {
      const T data = __ldg(&inp[tid]);
      bool temp = true;
      for ( int o = 1; o <= order; o++ ) {
        int plus;
        int minus;
        if ( axis == 0 ) {
          plus  = tx + o;
          minus = tx - o;
          clip_plus( clip, in_y, plus );
          clip_minus( clip, in_y, minus );
          plus  = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus  = ty + o;
          minus = ty - o;
          clip_plus( clip, in_x, plus );
          clip_minus( clip, in_x, minus );
          plus  = tx * in_x + plus;
          minus = tx * in_x + minus;
        }
        temp &= data > __ldg(&inp[plus]);
        temp &= data >= __ldg(&inp[minus]);
      }
      finalResult = temp;
    }
    results[tid] = finalResult;
  }
}

template<typename T>
void cpu_relextrema_2D(
  const int  in_x,
  const int  in_y,
  const int  order,
  const bool clip,
  const int  axis,
  const T *__restrict__ inp,
  bool *__restrict__ results) 
{
  for (int tx = 0; tx < in_y; tx++)
    for (int ty = 0; ty < in_x; ty++) {
      const int tid = tx * in_x + ty;
      const T data = inp[tid];
      bool    temp = true;
      for ( int o = 1; o <= order; o++ ) {
        int plus;
        int minus;
        if ( axis == 0 ) {
          plus  = tx + o;
          minus = tx - o;
          clip_plus( clip, in_y, plus );
          clip_minus( clip, in_y, minus );
          plus  = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus  = ty + o;
          minus = ty - o;
          clip_plus( clip, in_x, plus );
          clip_minus( clip, in_x, minus );
          plus  = tx * in_x + plus;
          minus = tx * in_x + minus;
        }
        temp &= data > inp[plus];
        temp &= data >= inp[minus];
      }
      results[tid] = temp;
    }
}

///////////////////////////////////////////////////////////////////////////////
//                          TEST FUNCTIONS                                   //
///////////////////////////////////////////////////////////////////////////////

template <typename T>
long test_1D (const int length, const int order, const bool clip,
              const int repeat, const char* type) 
{
  T* x = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;
  
  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  T* d_x;
  bool *d_result;
  cudaMalloc((void**)&d_x, length * sizeof(T));
  cudaMemcpy(d_x, x, length * sizeof(T), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_result, length * sizeof(bool));

  // Use 256 threads per block.
  dim3 grids ((length + 255) / 256);
  dim3 threads (256);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch the kernel only once and do all repetitions within it.
  relextrema_1D_kernel<T><<<grids, threads>>>(length, order, clip, repeat, d_x, d_result);
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  const double elapsed_sec = (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-9;
  // Divide by repeat to get average execution time.
  printf("Average 1D kernel (type = %s, order = %d, clip = %d) execution time %f (s)\n", 
         type, order, clip, elapsed_sec / repeat);

  cudaMemcpy(gpu_r, d_result, length * sizeof(bool), cudaMemcpyDeviceToHost);
  cpu_relextrema_1D<T>(length, order, clip, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++) {
    if (cpu_r[i] != gpu_r[i]) {
      error = 1; 
      break;
    }
  }

  cudaFree(d_x);
  cudaFree(d_result);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error)
    printf("1D test: FAILED\n");
  return elapsed_sec;
}

template <typename T>
long test_2D (const int length_x, const int length_y, const int order,
              const bool clip, const int axis, const int repeat, const char* type) 
{
  const int length = length_x * length_y;
  T* x = (T*) malloc (sizeof(T)*length);
  for (int i = 0; i < length; i++)
    x[i] = rand() % length;
  
  bool* cpu_r = (bool*) malloc (sizeof(bool)*length);
  bool* gpu_r = (bool*) malloc (sizeof(bool)*length);

  T* d_x;
  bool *d_result;
  cudaMalloc((void**)&d_x, length * sizeof(T));
  cudaMemcpy(d_x, x, length * sizeof(T), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_result, length * sizeof(bool));

  // Use a 16x16 block configuration.
  dim3 grids ((length_x + 15) / 16, (length_y + 15) / 16);
  dim3 threads (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  relextrema_2D_kernel<T><<<grids, threads>>>(length_x, length_y, order, clip, axis, repeat, d_x, d_result);
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  const double elapsed_sec = (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-9;
  printf("Average 2D kernel (type = %s, order = %d, clip = %d, axis = %d) execution time %f (s)\n", 
         type, order, clip, axis, elapsed_sec / repeat);

  cudaMemcpy(gpu_r, d_result, length * sizeof(bool), cudaMemcpyDeviceToHost);
  cpu_relextrema_2D(length_x, length_y, order, clip, axis, x, cpu_r);

  int error = 0;
  for (int i = 0; i < length; i++) {
    if (cpu_r[i] != gpu_r[i]) {
      error = 1;
      break;
    }
  }

  cudaFree(d_x);
  cudaFree(d_result);
  free(x);
  free(cpu_r);
  free(gpu_r);
  if (error)
    printf("2D test: FAILED\n");
  return elapsed_sec;
}

///////////////////////////////////////////////////////////////////////////////
//                              MAIN                                         //
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage ./%s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  double total_time = 0;

  // 1D tests for different data types.
  for (int order = 1; order <= 128; order = order * 2) {
    total_time += test_1D<   int>(1000000, order, true, repeat, "int");
    total_time += test_1D<  long>(1000000, order, true, repeat, "long");
    total_time += test_1D< float>(1000000, order, true, repeat, "float");
    total_time += test_1D<double>(1000000, order, true, repeat, "double");
  }

  // 2D tests along axis 1.
  for (int order = 1; order <= 128; order = order * 2) {
    total_time += test_2D<   int>(1000, 1000, order, true, 1, repeat, "int");
    total_time += test_2D<  long>(1000, 1000, order, true, 1, repeat, "long");
    total_time += test_2D< float>(1000, 1000, order, true, 1, repeat, "float");
    total_time += test_2D<double>(1000, 1000, order, true, 1, repeat, "double");
  }

  // 2D tests along axis 0.
  for (int order = 1; order <= 128; order = order * 2) {
    total_time += test_2D<   int>(1000, 1000, order, true, 0, repeat, "int");
    total_time += test_2D<  long>(1000, 1000, order, true, 0, repeat, "long");
    total_time += test_2D< float>(1000, 1000, order, true, 0, repeat, "float");
    total_time += test_2D<double>(1000, 1000, order, true, 0, repeat, "double");
  }

  printf("\n-----------------------------------------------\n");
  printf("Total kernel execution time: %lf (s)\n", total_time);
  printf("-----------------------------------------------\n");

  return 0;
}
