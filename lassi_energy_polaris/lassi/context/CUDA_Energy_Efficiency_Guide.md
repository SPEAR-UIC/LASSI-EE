# CUDA GPU Energy Efficiency Guide
## Optimizing CUDA Programs for Reduced Energy Consumption on NVIDIA GPUs

This guide covers techniques for reducing the energy consumption (Joules) of
CUDA programs running on NVIDIA data-center GPUs, focusing on the A100
(sm_80, Ampere architecture).  Energy = Power × Time; any reduction in
average power draw OR execution time reduces total energy.

---

## 1. Understanding GPU Energy Consumption

### 1.1 Where GPU Power Goes

An NVIDIA A100 GPU has a 400 W Thermal Design Power (TDP).  Power is
consumed by:

- **DRAM (HBM2e)**: Reading or writing GPU memory is the single largest
  power consumer in memory-bound kernels.  HBM2e has ~2 TB/s bandwidth but
  consumes significant power per byte transferred.
- **L2 cache**: Smaller than DRAM accesses but still significant; 40 MB on A100.
- **Compute units (CUDA cores, Tensor cores)**: Active computation draws power
  proportional to instruction throughput and operand switching activity.
- **Clock distribution and I/O**: Fixed overhead regardless of workload.
- **Idle SM overhead**: SMs that are scheduled but have no active warps still
  consume static power.

### 1.2 Memory-Bound vs Compute-Bound

- **Memory-bound kernels** (like XSBench's random cross-section lookup) spend
  most of their time waiting for data from DRAM.  Their energy is dominated by
  DRAM access energy.  Reducing memory traffic is the highest-leverage strategy.
- **Compute-bound kernels** spend most time executing arithmetic.  Their energy
  is dominated by FP/INT unit switching.  Reducing instruction count and
  switching to lower-precision arithmetic helps.

**To classify your kernel**: if DRAM bandwidth utilization (from Nsight Compute)
is close to the roofline peak, the kernel is memory-bound.

---

## 2. Memory Access Optimization (Highest Impact for Memory-Bound Kernels)

### 2.1 Global Memory Coalescing

**Problem**: When threads in a warp access non-consecutive memory addresses,
the hardware cannot combine the accesses into a single DRAM transaction.
Instead, it issues multiple smaller transactions, multiplying DRAM traffic and
energy.

**Rule**: Warp thread `i` should access address `base + i * stride`, with
`stride == 1` (sizeof(element)) for perfect coalescing.

**Diagnostic**: If an access pattern is `arr[threadIdx.x * N + j]` inside a
loop over `j`, threads in the warp are strided — uncoalesced.  Transposing
to `arr[j * blockDim.x + threadIdx.x]` achieves coalescing.

**Array-of-Structs (AoS) → Struct-of-Arrays (SoA)**:
```cpp
// AoS — uncoalesced (stride = sizeof(Particle))
struct Particle { float x, y, z, vx, vy, vz; };
Particle particles[N];
// Thread i reads: particles[i].x  →  strided access

// SoA — coalesced (stride = 1)
struct Particles { float* x; float* y; float* z; float* vx; float* vy; float* vz; };
// Thread i reads: particles.x[i]  →  consecutive access
```

**Energy impact**: Perfect coalescing can reduce DRAM transactions by 32×
vs fully uncoalesced access, directly reducing DRAM power.

### 2.2 Shared Memory Tiling

**Problem**: When multiple threads in a block repeatedly read overlapping
regions of global memory, each read issues a separate DRAM transaction.
Shared memory (SRAM) is on-chip, ~100× faster than HBM2e, and consumes
far less energy per access.

**Pattern** (tile-based reduction of global memory traffic):
```cpp
__global__ void tiled_kernel(const float* __restrict__ g_data, ...) {
    __shared__ float s_tile[TILE_SIZE];

    // Cooperatively load tile from global memory (once per block)
    if (threadIdx.x < TILE_SIZE)
        s_tile[threadIdx.x] = g_data[blockIdx.x * TILE_SIZE + threadIdx.x];
    __syncthreads();

    // Each thread reads from shared memory (no DRAM access)
    float val = s_tile[some_index];
    ...
}
```

**When applicable**: When threads in the same block access the same or
nearby global memory addresses (spatial reuse within a block).

**A100 shared memory**: Up to 164 KB per SM (configurable up to 163 KB).
Use `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, N)`
to request more than the default 48 KB.

**Bank conflicts**: Shared memory is divided into 32 banks.  If multiple
threads access the same bank simultaneously, accesses serialize.  Ensure
thread `i` accesses `s[i * stride]` where `stride` is not a multiple of 32.

### 2.3 Read-Only Cache (`__ldg` and `const __restrict__`)

**Problem**: Regular global memory reads go through L1/L2 caches but compete
with write traffic.  The A100 has a separate 256 KB L1 texture cache path
that is read-only and often underutilized.

**Solution**: For arrays that are only read (never written) inside a kernel,
route reads through the texture cache using `__ldg()` or `const __restrict__`:

```cpp
// Method 1: __ldg() intrinsic (explicit)
float val = __ldg(&arr[i]);

// Method 2: const __restrict__ pointer (compiler uses texture cache path)
__global__ void kernel(const float* __restrict__ arr, ...) {
    float val = arr[i];   // compiler may generate LDG instruction
}
```

**When applicable**: Read-only arrays that have spatial or temporal locality
in access patterns.  For XSBench-style random accesses, __ldg still helps
by routing through the larger texture cache path.

**Energy impact**: The texture cache reduces L2 pressure and DRAM traffic for
repeated reads to the same cache line.

---

## 3. Occupancy Optimization

### 3.1 What Is Occupancy

**Occupancy** = (active warps per SM) / (maximum warps per SM).
On an A100, max warps per SM = 64 (= 2048 threads / 32 threads-per-warp).

Low occupancy means some SMs are idle even while others are oversubscribed.
Idle SMs still draw static power — better occupancy means the same work is
done with fewer idle SMs, reducing energy.

### 3.2 Factors Limiting Occupancy

1. **Registers per thread**: A100 has 65,536 registers per SM.  If each thread
   uses 64 registers, max threads per SM = 65536/64 = 1024 (50% occupancy
   with 256-thread blocks).
2. **Shared memory per block**: 164 KB total per SM.  A 96 KB shared memory
   allocation allows only one block per SM.
3. **Block size**: Must be a multiple of 32 (warp size).  Very small blocks
   (e.g., 32 threads) waste scheduler slots; very large blocks (e.g., 1024)
   may limit occupancy due to registers and shared memory.

### 3.3 Choosing Block Size

For most workloads, **128–256 threads per block** maximizes occupancy while
minimizing register spilling.  Use CUDA Occupancy Calculator or:

```cpp
// Query optimal block size at runtime
int min_grid, block;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, my_kernel, 0, 0);
```

### 3.4 Avoiding Register Spilling

When a kernel uses more registers than the per-thread budget, the compiler
spills to local memory (which lives in DRAM — expensive).  To reduce spills:
- Break large kernels into smaller kernels
- Use `__launch_bounds__(MAX_THREADS, MIN_BLOCKS)` to guide register allocation
- Move intermediate values into shared memory

```cpp
__global__ __launch_bounds__(256, 4)   // 256 threads/block, min 4 blocks/SM
void my_kernel(...) { ... }
```

---

## 4. Warp Divergence Reduction

### 4.1 What Is Warp Divergence

All 32 threads in a warp execute the same instruction in lockstep (SIMT).
When threads take different branches of an `if/else`, the warp executes both
branches serially, with inactive threads masked.  This doubles (or worse)
the instruction count for that code region and wastes execution units.

**Energy cost**: Divergence increases active cycles per iteration and prevents
the SM from issuing new work to diverged lanes.

### 4.2 Strategies

**Branchless arithmetic**:
```cpp
// Divergent
if (x > 0) result = a;
else result = b;

// Branchless
result = (x > 0) ? a : b;           // compiler generates SELP instruction
result = b + (a - b) * (x > 0);     // arithmetic version
```

**Warp-level predication**: For short branches (1–4 instructions), the compiler
may already use predicated instructions instead of branches.  Ensure the branch
body is short.

**Sort inputs to group similar paths**: If threads process different work types
based on an array value, sorting inputs by type before launching the kernel
ensures entire warps process the same type, eliminating divergence.

**Minimize random lookups that produce divergent table entries**: XSBench
performs random cross-section lookups; the lookup itself cannot be made
non-divergent, but the surrounding control flow can.

---

## 5. Arithmetic Precision

### 5.1 Float vs Double

Double-precision (64-bit) floating-point operations:
- Use 2× more registers and shared memory than single-precision
- Are 2× slower on most CUDA cores (A100 has 312 TFLOPS FP16, 156 TFLOPS BF16,
  77.6 TFLOPS FP32, 9.7 TFLOPS FP64 — FP64 is ~8× less throughput than FP32)
- Increase data movement cost (8 bytes vs 4 bytes per value)

**When safe to use float**:
- Physics quantities with relative accuracy > 1e-6
- Intermediate accumulations that are later reduced and don't require
  inter-operation consistency with FP64 results

**CUDA fast-math intrinsics** (use only when precision loss is acceptable):
```cpp
__expf(x)    // vs expf(x) — uses hardware approximation (~22-bit precision)
__logf(x)    // vs logf(x)
__sinf(x), __cosf(x)
__fdividef(a, b)  // fast single-precision divide
__fmaf_rn(a, b, c)  // fused multiply-add (also available in standard math)
```

**Note for XSBench**: The simulation uses lookup tables with float or double.
Check whether changing intermediate values to float changes the verification
checksum — if it does, you must keep the original precision.

### 5.2 Fused Multiply-Add (FMA)

FMA computes `a*b + c` in a single instruction with a single rounding, which
is more accurate AND faster than separate multiply and add.  The CUDA compiler
uses FMA by default when possible; explicit `fmaf()` / `fma()` ensures it.

---

## 6. Kernel Fusion

**Problem**: Launching multiple consecutive kernels that operate on the same
data forces intermediate results to be written to DRAM and read back.
Each kernel launch also incurs a fixed overhead (~5–15 µs).

**Solution**: Merge consecutive kernels into a single kernel, keeping
intermediate results in registers or shared memory.

```cpp
// Before: two kernel launches, data written and read from global memory
launch_kernel_A<<<grid, block>>>(d_in, d_tmp);
launch_kernel_B<<<grid, block>>>(d_tmp, d_out);

// After: fused into one, d_tmp stays in registers
launch_kernel_AB<<<grid, block>>>(d_in, d_out);
__global__ void launch_kernel_AB(float* in, float* out) {
    float tmp = compute_A(in[i]);  // was kernel A
    out[i] = compute_B(tmp);       // was kernel B — no DRAM round-trip
}
```

**Constraint**: Only fuse kernels that have compatible launch configurations
and data dependencies that can be resolved within a single kernel.

---

## 7. Synchronization Minimization

### 7.1 `__syncthreads()` Cost

`__syncthreads()` causes all threads in a block to stall until every thread
reaches the barrier.  It serializes execution and prevents the scheduler from
issuing new warps during the wait.

**Minimize by**:
- Moving `__syncthreads()` outside of loops when data is not modified each iteration
- Restructuring shared memory access patterns to avoid the need for mid-loop barriers
- Using warp-level primitives (`__syncwarp()`, `__ballot_sync()`, `__reduce_sync()`)
  when only intra-warp consistency is needed

### 7.2 Atomic Operations

Atomic operations (`atomicAdd`, `atomicCAS`, etc.) serialize concurrent updates
to the same address.  On A100, native FP32 atomics are efficient, but
high-contention atomics on a single address (reduction pattern) waste energy.

**Better reduction pattern** (tree reduction using shared memory):
```cpp
__shared__ float s_partial[BLOCK_SIZE];
s_partial[threadIdx.x] = local_sum;
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride)
        s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
}
// Only one atomic per block (instead of one per thread)
if (threadIdx.x == 0) atomicAdd(global_result, s_partial[0]);
```

---

## 8. Memory Transfer Optimization

### 8.1 Minimize Host-Device Transfers

PCIe transfers between CPU and GPU are slow (~32 GB/s bidirectional) and
consume significant energy.  Minimize by:
- Keeping data on the GPU for multiple kernel invocations
- Using pinned (page-locked) host memory for faster transfers when transfers
  are necessary (`cudaMallocHost`)
- Overlapping transfers with computation using streams and `cudaMemcpyAsync`

### 8.2 Unified Memory (cudaMallocManaged)

Unified memory can cause implicit page migrations that are expensive.  For
best energy efficiency, use explicit `cudaMalloc` and `cudaMemcpy` to control
exactly when transfers occur.

---

## 9. Loop Unrolling

The `#pragma unroll` directive tells the compiler to unroll a loop by a
specified factor (or fully if the trip count is known at compile time).
Unrolling:
- Eliminates loop-counter increment and branch instructions
- Enables the compiler to schedule more independent instructions in flight
- Increases register pressure — balance against occupancy

```cpp
// Fully unroll a short inner loop (trip count known at compile time)
#pragma unroll
for (int k = 0; k < 4; ++k)
    sum += a[i + k] * b[j + k];

// Partial unroll (factor 4) for longer loops
#pragma unroll 4
for (int k = 0; k < N; ++k)
    sum += a[k] * b[k];
```

---

## 10. Profiling-Guided Optimization

The most impactful optimizations are identified by profiling, not guessing.

### Key Nsight Compute Metrics

| Metric | What it reveals |
|---|---|
| `dram__bytes_read.sum` | Total DRAM bytes read — high = memory-bound |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | L1 cache sectors for global loads |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | Coalescing efficiency |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Occupancy (warp-based) |
| `smsp__thread_inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed` | FMA utilization |
| `smsp__sass_inst_executed_op_branch.sum` | Branch instructions (divergence proxy) |
| `gpu__time_duration.sum` | Wall-clock execution time |

### nvidia-smi Power Monitoring

During profiling, capture power with:
```bash
nvidia-smi dmon -s p -d 1 -i 0   # sample power (W) every second, GPU 0
nvidia-smi --query-gpu=power.draw --format=csv,noheader -l 1  # alternative
```

---

## 11. XSBench-Specific Considerations

XSBench simulates Monte Carlo neutron transport cross-section lookups.
Its GPU kernels perform:

1. **Random nuclide and energy sampling** — generates random indices into large
   cross-section tables stored in GPU global memory.
2. **Binary search on energy grids** (`grid_search`) — each thread binary-searches
   an array of size ~100k floats.  Access pattern is logarithmic, predictable.
3. **Cross-section interpolation** — reads nuclide-specific cross-section data at
   the found grid point.  Access is random (data-dependent on sampled nuclide).

**Key characteristics**:
- **Highly memory-bound**: random access to ~10 GB of cross-section tables means
  most accesses miss L1 and L2 — they go to HBM2e.
- **Low arithmetic intensity**: a few FP operations per DRAM word fetched.
- **Divergence**: different threads sample different nuclides and may take
  different branches in interpolation code.

**Most impactful strategies for XSBench**:
1. `__ldg()` for read-only cross-section arrays (routes through texture cache)
2. Block-level shared memory for the energy grid search (the grid is shared
   across all lookups within a simulation step if multiple lookups share energy)
3. `const __restrict__` on all input arrays
4. Occupancy tuning (block size 128–256 threads)
5. Replace any remaining `double` intermediate values with `float` if the
   checksum is insensitive to the precision change

**Constraints to never violate in XSBench**:
- Do NOT change the binary search algorithm or loop bounds (correctness-critical)
- Do NOT change the interpolation formula
- The `Verification checksum:` output line must remain `(Valid)` — any change
  to numerical behavior will invalidate the checksum and be automatically rejected

---

## 12. Quick Reference Checklist

Before submitting an optimized CUDA kernel, verify:

- [ ] Global memory accesses are coalesced (warp threads access consecutive addresses)
- [ ] Read-only input arrays use `const __restrict__` or `__ldg()`
- [ ] Shared memory is used for data reused within a thread block
- [ ] Block size is a multiple of 32, and 128–256 threads per block is typical
- [ ] `__syncthreads()` is only where strictly required by data dependencies
- [ ] No unnecessary double-precision operations if float suffices
- [ ] Loop bodies that iterate small, fixed counts use `#pragma unroll`
- [ ] Consecutive kernels on the same data are fused where possible
- [ ] No function stubs (`(void)param;` casts in simulation functions)
- [ ] Verification checksum unchanged (XSBench `(Valid)` line present)
