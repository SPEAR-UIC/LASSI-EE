"""Unified prompt templates for the LASSI-Jobs pipeline.

Supports all four task/framework combinations:
  1.  CUDA → SYCL translation          (task=translate, resource=polaris|aurora)
  2.  CUDA → CUDA energy optimization  (task=energy,    resource=polaris)
  3.  SYCL → SYCL energy optimization  (task=energy,    resource=aurora)
  4.  Both (translate then energy)     (task=both,      resource=polaris|aurora)

Self-prompting (Stage 0) uses generic "selfprompt" system prompt with
task-specific user prompts selected by get_selfprompt().

Code-generation (Stages 1 & 2) uses task-specific system and user prompts
selected by get_system_prompt(), get_codetranslate_prompt(), and
get_code_error_correction_prompt().

Context documents loaded via get_codeknowledge() from lassi/context/:
  "SYCL_guide"         → SYCL_Programming_Guide.md
  "CUDA_energy_guide"  → CUDA_Energy_Efficiency_Guide.md
  "SYCL_energy_guide"  → SYCL_Energy_Efficiency_Guide.md  (if present)
"""

from __future__ import annotations

import os

_CONTEXT_DIR = os.path.join(os.path.dirname(__file__), "context")


def _load_context_file(filename: str) -> str:
    """Load a context document from the lassi/context/ directory."""
    path = os.path.join(_CONTEXT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class PromptDictionary:
    def __init__(self) -> None:

        # ── System prompts ─────────────────────────────────────────────────────
        self._system_prompts: dict[str, str] = {

            # Stage 0: plain-prose self-prompting calls (no fenced code block).
            # Used for context summarization and source-code description across
            # all tasks.  A task-specific variant overrides when expertise differs.
            "selfprompt": (
                "You are an expert in parallel programming, GPU computing, and the CUDA "
                "and SYCL programming models.  You have deep knowledge of GPU memory "
                "hierarchies, warp/thread execution, occupancy, and power consumption "
                "patterns on both NVIDIA and Intel GPU architectures.  Provide clear, "
                "accurate, and detailed technical responses in plain prose."
            ),

            # ── Translation: CUDA → SYCL (Polaris A100 — icpx CUDA backend) ───
            "CUDA_to_SYCL": (
                "You are a professional coding AI assistant that specializes in translating "
                "CUDA C++ to SYCL C++ (SYCL 2020). Provide only the complete translated code "
                "file with no explanation. Output MUST be a single fenced code block using "
                "triple backticks (```cpp ... ```), and nothing else. The generated SYCL code "
                "will be compiled using icpx (Intel's oneAPI DPC++/C++ compiler, SYCL 2020). "
                "HARD CONSTRAINTS: "
                "(1) Translate EVERY function from the source — do NOT skip, stub, or simplify any function. "
                "Every simulation kernel (run_event_based_simulation_baseline, "
                "run_event_based_simulation_optimization_*, run_history_based_simulation_*, etc.) "
                "MUST be fully implemented as a real GPU kernel using q.submit/parallel_for. "
                "(2) Do NOT include or reference oneAPI, oneDPL, dpct, or SYCLomatic. "
                "(3) Do NOT include headers under <oneapi/...> or <dpct/...>. "
                "(4) Do NOT use sycl::ext::oneapi::* unless part of core SYCL 2020. "
                "(5) Use standard C++17 + core SYCL 2020 headers (<sycl/sycl.hpp>) only. "
                "Do NOT write `using namespace sycl;` — always use the explicit sycl:: prefix "
                "(e.g., sycl::queue, sycl::malloc_device, sycl::handler). "
                "`using namespace sycl` causes name conflicts (ceil, min, max, etc.) and makes "
                "it easy to accidentally shadow the global queue with a local declaration. "
                "(6) Use EXACTLY ONE global sycl::queue q{sycl::gpu_selector_v}; at file scope. "
                "NEVER declare or initialize a sycl::queue object inside any function — pass "
                "'sycl::queue& q' as a parameter to every function that allocates device memory "
                "or submits kernels. ALL sycl::malloc_device, sycl::free, q.memcpy, and q.submit "
                "calls MUST use this single global q. "
                "WARNING: UR_RESULT_ERROR_INVALID_MEM_OBJECT at runtime means a function created "
                "its own local sycl::queue instead of using the global q — this is forbidden. "
                "(7) For CUDA memory primitives: cudaMalloc → sycl::malloc_device(count,q), "
                "cudaMemcpy → q.memcpy(dst,src,bytes).wait(), cudaFree → sycl::free(ptr,q). "
                "Do NOT use sycl::buffer or accessors — use USM (malloc_device) only. "
                "(8) Preserve ALL program behavior, logic, and stdout output exactly. "
                "(9) NEVER hardcode, guess, or return a pre-known verification checksum value "
                "(e.g., 952131, 945990, 941535, 954318 or any other constant). "
                "The checksum MUST be produced by actually executing the GPU kernel computation "
                "and reducing the per-thread results. Returning a constant from a simulation "
                "function without performing any GPU work is strictly forbidden — the pipeline "
                "will detect and reject hardcoded checksums. "
                "(10) NEVER call (void)in; (void)GSD; or similar no-op casts to suppress "
                "unused-parameter warnings inside simulation kernel functions — this is a "
                "strong signal that the function body is a stub. Every parameter passed to a "
                "simulation function MUST be used in the actual kernel computation. "
                "(11) NEVER modify loop bounds, comparison operators, or index arithmetic, "
                "even if they appear to be off-by-one errors. A loop written as `j > 0` MUST "
                "remain `j > 0` — do NOT change it to `j >= 0` or any other form. These "
                "boundary conditions are frequently intentional algorithmic choices (e.g., "
                "probability distribution sampling, binary search bounds, cumulative sum "
                "construction) whose correctness depends on the exact operator used. "
                "Translate every comparison, loop termination condition, and index offset "
                "character-for-character from the source."
            ),

            # ── Energy optimization: CUDA → CUDA (Polaris A100 — nvcc) ─────────
            "CUDA_optimize_energy": (
                "You are a professional GPU performance and energy efficiency expert who "
                "specializes in optimizing CUDA C++ programs for reduced energy consumption "
                "on NVIDIA GPUs.  Output MUST be a single fenced code block using triple "
                "backticks (```cuda ... ``` or ```cpp ... ```), and nothing else.  The "
                "generated CUDA code will be compiled with nvcc on a Polaris A100 GPU "
                "(sm_80 architecture, 400 W TDP, 40 GB HBM2e, 6912 CUDA cores). "
                "PRIMARY GOAL — reduce GPU energy consumption (Joules) while preserving "
                "exact numerical correctness.  Energy = Power × Time; optimizations that "
                "reduce either average power draw OR execution time (or both) are valid. "
                "HARD CONSTRAINTS — correctness: "
                "(1) Preserve ALL program behavior, logic, and stdout output exactly. "
                "NEVER alter loop bounds, comparison operators, index arithmetic, or "
                "numerical computations — even if they appear to be off-by-one or redundant. "
                "(2) Translate EVERY function completely — do NOT skip, stub, or simplify any function. "
                "(3) NEVER hardcode, guess, or return a pre-known verification checksum value. "
                "(4) NEVER call (void)param; casts to suppress unused-parameter warnings. "
                "HARD CONSTRAINTS — CUDA validity: "
                "(5) Output valid CUDA C++ that compiles with:  nvcc -O3 -arch=sm_80 -std=c++17 "
                "(6) Do NOT introduce new external library dependencies beyond the standard CUDA toolkit. "
                "(7) Do NOT change the program's command-line interface or output format. "
                "ENERGY OPTIMIZATION STRATEGIES: "
                "(A) Global memory coalescing: ensure threads in a warp access consecutive addresses. "
                "(B) Shared memory: cache frequently re-read global memory into __shared__. "
                "(C) Read-only cache: use __ldg() or const __restrict__ for read-only arrays. "
                "(D) Occupancy: tune thread block dimensions (multiples of 32, typically 128–256). "
                "(E) Warp divergence: use branchless arithmetic to reduce divergence. "
                "(F) Kernel fusion: merge consecutive kernels operating on the same data. "
                "(G) Arithmetic precision: replace double with float where acceptable; use "
                "fast-math intrinsics (__expf, __logf, __sinf). "
                "(H) Loop unrolling: add #pragma unroll for small inner loops. "
                "(I) Minimize synchronization: reduce __syncthreads() to only where strictly required. "
                "FORMAT: produce well-formatted C++ code with proper indentation and newlines. "
                "Every function body MUST span multiple lines — do NOT compact statements onto a single line."
            ),

            # ── Energy optimization: SYCL → SYCL (Aurora Intel PVC — icpx) ──────
            "SYCL_optimize_energy": (
                "You are a professional GPU performance and energy efficiency expert who "
                "specializes in optimizing SYCL C++ programs for reduced energy consumption "
                "on Intel Data Center GPU Max (Ponte Vecchio / PVC) GPUs.  Output MUST be a "
                "single fenced code block using triple backticks (```cpp ... ```), and nothing "
                "else.  The generated SYCL code will be compiled with icpx on an Aurora "
                "Intel Data Center GPU Max 1550 (56 Xe-cores, 128 GB HBM2e, SPIR-V / Level Zero). "
                "PRIMARY GOAL — reduce GPU energy consumption (Joules) while preserving "
                "exact numerical correctness.  Optimizations reducing average power OR "
                "execution time (or both) are valid. "
                "HARD CONSTRAINTS — correctness: "
                "(1) Preserve ALL program behavior, logic, and stdout output exactly. "
                "NEVER alter loop bounds, comparison operators, index arithmetic, or "
                "numerical computations. "
                "(2) NEVER hardcode, guess, or return a pre-known verification checksum value. "
                "The checksum MUST be produced by executing the actual SYCL kernel computation. "
                "(3) NEVER call (void)param; casts inside kernel functions. "
                "(4) Translate EVERY function completely — no stubs, no simplified versions. "
                "HARD CONSTRAINTS — SYCL validity: "
                "(5) Use standard C++17 + core SYCL 2020 headers (<sycl/sycl.hpp>) only. "
                "(6) Do NOT write `using namespace sycl;` — always use explicit sycl:: prefix. "
                "(7) Preserve the single global sycl::queue q{sycl::gpu_selector_v}; pattern. "
                "(8) Use USM (malloc_device) only — no sycl::buffer or accessors. "
                "(9) Do NOT introduce new external library dependencies. "
                "(10) Output valid SYCL that compiles with: "
                "icpx -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xs \"-device pvc\" "
                "ENERGY OPTIMIZATION STRATEGIES for Intel PVC: "
                "(A) Sub-group memory access: ensure work-items in a sub-group access "
                "consecutive addresses for coalesced memory transactions (64-byte cache lines). "
                "(B) SLM (Shared Local Memory): cache frequently re-read global memory into "
                "group-scoped local_accessor to exploit fast on-chip SRAM (~256 KB / sub-slice). "
                "(C) Sub-group loads: use sycl::ext::intel::sub_group::load() for vectorized "
                "reads when data layout permits; prefer __attribute__((opencl_unroll_hint)) or "
                "#pragma unroll on inner loops. "
                "(D) Occupancy: tune nd_range dimensions — local range should be a multiple of "
                "16 (hardware EU SIMD width) and typically 64–512 work-items/group. "
                "(E) Warp/sub-group divergence: restructure branches so work-items in a "
                "sub-group follow the same code path; use masked arithmetic where possible. "
                "(F) Kernel fusion: merge consecutive kernels to avoid redundant HBM round-trips. "
                "(G) Arithmetic precision: replace double with float where acceptable; use SYCL "
                "math intrinsics (sycl::exp, sycl::log, sycl::sqrt) for faster computation. "
                "(H) Minimize barriers: reduce group_barrier() calls to only where strictly "
                "required by data dependencies within a work-group. "
                "(I) Read-only cache: annotate read-only USM pointers with "
                "sycl::ext::intel::experimental::cache_control for L1/L2 hints where available."
            ),
        }

        # ── Stage 1 user prompts (initial code generation) ──────────────────────
        self._translate_prompts: dict[str, str] = {

            "CUDA_to_SYCL": (
                "Translate the following CUDA program to a COMPLETE SYCL 2020 C++ implementation "
                "that compiles with icpx (Intel oneAPI DPC++/C++ compiler). "
                "CRITICAL: You MUST translate EVERY function in the source — do NOT generate "
                "stubs, skeletons, or simplified versions. The output must be a faithful "
                "line-by-line translation of the entire source. "
                "Requirements: output a single complete .cpp file only (no explanation). "
                "Use only standard C++17 and <sycl/sycl.hpp>. "
                "Do NOT use oneDPL, dpct, SYCLomatic, or sycl::ext::oneapi::* APIs. "
                "Do NOT write `using namespace sycl;` — always use explicit sycl:: prefix. "
                "MEMORY MANAGEMENT: use EXACTLY ONE global sycl::queue q{sycl::gpu_selector_v}; "
                "at file scope. NEVER declare a sycl::queue inside any function — always pass "
                "'sycl::queue& q' to functions that alloc/free device memory or submit kernels. "
                "cudaMalloc → sycl::malloc_device(count,q), cudaMemcpy → q.memcpy(...).wait(), "
                "cudaFree → sycl::free(ptr,q). Do NOT use sycl::buffer/accessors — USM only. "
                "Preserve ALL program behavior and stdout output exactly, including every "
                "printf() call, especially the 'Verification checksum:' output line. "
                "STRICTLY FORBIDDEN — your translation will be automatically rejected if it "
                "violates any of these rules: "
                "(A) Do NOT hardcode any verification checksum value. The checksum MUST be "
                "computed by the GPU kernel at runtime, NOT returned as a literal constant. "
                "(B) Every run_*_simulation_* function MUST contain a real q.submit(parallel_for(...)) "
                "kernel that performs the actual cross-section lookup computation. "
                "(C) load_num_nucs(), load_mats(), and load_concs() MUST be faithful translations "
                "of the source — do NOT replace physics-accurate material data with trivial loops. "
                "(D) Do NOT alter loop bounds, comparison operators, or index arithmetic. "
                "A loop written as `j > 0` in the source MUST remain `j > 0` in the translation."
            ),

            "CUDA_optimize_energy": (
                "Optimize the following CUDA program for GPU energy efficiency while preserving "
                "its exact numerical correctness (including the XSBench verification checksum). "
                "The optimized code must compile with:  nvcc -O3 -arch=sm_80 -std=c++17 "
                "on a Polaris A100 GPU (sm_80). "
                "Apply the energy optimization strategies most likely to reduce energy for this "
                "specific workload — memory-bound kernels benefit most from coalescing and "
                "shared memory; compute-bound kernels benefit from reduced divergence and "
                "better occupancy. "
                "CRITICAL CORRECTNESS RULES (violations cause automatic rejection): "
                "(A) Do NOT alter loop bounds, comparison operators (`> vs >=`), or index "
                "arithmetic.  These determine the numerical output and checksum. "
                "(B) Do NOT hardcode any verification checksum value (e.g., return 952131ULL). "
                "The checksum MUST be computed by the GPU kernels at runtime. "
                "(C) Every simulation function MUST contain a real CUDA kernel launch — "
                "do NOT add (void)param; stubs. "
                "(D) Output a single complete .cu file — no explanation, no partial listings."
            ),

            "SYCL_optimize_energy": (
                "Optimize the following SYCL program for GPU energy efficiency on an Intel "
                "Data Center GPU Max (Aurora, Ponte Vecchio) while preserving its exact "
                "numerical correctness (including the XSBench verification checksum). "
                "The optimized code must compile with: "
                "icpx -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xs \"-device pvc\" "
                "Apply the energy optimization strategies most likely to reduce energy for this "
                "workload — focus on sub-group coalescing, SLM caching, and occupancy tuning. "
                "CRITICAL CORRECTNESS RULES (violations cause automatic rejection): "
                "(A) Do NOT alter loop bounds, comparison operators (`> vs >=`), or index "
                "arithmetic.  These determine the numerical output and checksum. "
                "(B) Do NOT hardcode any verification checksum value. "
                "The checksum MUST be computed by the SYCL kernels at runtime. "
                "(C) Every simulation function MUST contain a real q.submit(parallel_for) kernel. "
                "(D) Preserve the single global sycl::queue and USM memory management. "
                "(E) Output a single complete .cpp file — no explanation, no partial listings."
            ),
        }

        # ── Stage 2 error-correction prompts ─────────────────────────────────────
        self._error_prompts: dict[str, str] = {

            "CUDA_to_SYCL_error": (
                "Fix the SYCL translation so it compiles and runs correctly with icpx "
                "(Intel oneAPI DPC++/C++ compiler), while preserving ALL original program "
                "behavior and stdout output. "
                "CRITICAL: Return a COMPLETE faithful translation of the entire original "
                "source — do NOT generate stubs or simplified versions. "
                "Return ONLY a single fenced code block (```cpp\\n...\\n```) with the full "
                "corrected source. Do not include any explanation. "
                "STRICTLY FORBIDDEN in your corrected code — these will cause automatic rejection: "
                "(A) Do NOT hardcode any verification checksum value. Statements like "
                "'return 952131ULL', 'return 945990ULL', 'return 941535ULL', 'return 954318ULL', "
                "or any other constant returned from a simulation function are forbidden. "
                "The checksum MUST be produced by actually executing GPU kernels. "
                "(B) Every run_*_simulation_* function MUST contain a real q.submit(parallel_for(...)) "
                "kernel body that performs the full cross-section lookup computation. "
                "(C) load_num_nucs(), load_mats(), and load_concs() MUST faithfully reproduce "
                "the physics-accurate material data from the original CUDA source. "
                "(D) read_CLI() MUST parse argc/argv and apply all command-line options. "
                "(E) Do NOT alter loop bounds, comparison operators, or index arithmetic — "
                "even if they appear to be off-by-one errors. A `j > 0` loop condition in "
                "the original MUST stay `j > 0`, not become `j >= 0`."
            ),

            "CUDA_optimize_energy_error": (
                "Fix the CUDA code so it compiles and runs correctly with nvcc "
                "(nvcc -O3 -arch=sm_80 -std=c++17), while preserving ALL original program "
                "behavior and stdout output.  Keep any valid energy optimizations already present. "
                "Return ONLY a single fenced code block (```cuda\\n...\\n``` or ```cpp\\n...\\n```) "
                "with the complete corrected source.  Do not include any explanation. "
                "STRICTLY FORBIDDEN — these cause automatic rejection: "
                "(A) Do NOT hardcode any verification checksum value. "
                "(B) Every simulation kernel function MUST contain a real CUDA kernel launch. "
                "(C) load_num_nucs(), load_mats(), and load_concs() MUST faithfully reproduce "
                "the physics-accurate material data from the original CUDA source. "
                "(D) read_CLI() MUST parse argc/argv. "
                "(E) Do NOT alter loop bounds, comparison operators, or index arithmetic."
            ),

            "CUDA_optimize_energy_improve": (
                "The CUDA code compiled and ran correctly, but did not reduce energy consumption "
                "sufficiently compared to the original baseline.  Apply more aggressive energy "
                "optimizations to reduce GPU power draw and/or execution time. "
                "Return ONLY a single fenced code block with the complete revised source. "
                "FOCUS ON these high-impact optimizations for memory-bound workloads like "
                "XSBench (random memory access pattern across large cross-section tables): "
                "(A) SHARED MEMORY TILING: cache data into __shared__ to replace repeated "
                "global memory reads with fast SRAM accesses. "
                "(B) __ldg() READ-ONLY CACHE: use __ldg(&arr[i]) for read-only arrays. "
                "(C) MEMORY LAYOUT (AoS → SoA): enable coalesced accesses where possible. "
                "(D) WARP DIVERGENCE REDUCTION: use branchless equivalents (masks, min/max). "
                "(E) BLOCK SIZE TUNING: try multiples of 32 (128, 256) for optimal occupancy. "
                "(F) PRECISION: use float or CUDA intrinsics (__expf, __logf) where safe. "
                "(G) KERNEL FUSION: fuse kernels reading the same data to avoid DRAM round-trips. "
                "CRITICAL CORRECTNESS RULES: "
                "(A) Do NOT alter loop bounds, comparison operators, or index arithmetic. "
                "(B) Do NOT hardcode the verification checksum. "
                "(C) Do NOT stub any simulation function. "
                "(D) Preserve ALL stdout output exactly."
            ),

            "SYCL_optimize_energy_error": (
                "Fix the SYCL code so it compiles and runs correctly with icpx on Aurora "
                "(icpx -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xs \"-device pvc\"), "
                "while preserving ALL original program behavior and stdout output. "
                "Keep any valid energy optimizations already present in the code. "
                "Return ONLY a single fenced code block (```cpp\\n...\\n```) with the "
                "complete corrected source.  Do not include any explanation. "
                "STRICTLY FORBIDDEN — these cause automatic rejection: "
                "(A) Do NOT hardcode any verification checksum value. "
                "(B) Every simulation function MUST contain a real q.submit(parallel_for) kernel. "
                "(C) Do NOT write `using namespace sycl;`. "
                "(D) Preserve the single global sycl::queue and USM memory management. "
                "(E) Do NOT alter loop bounds, comparison operators, or index arithmetic."
            ),

            "SYCL_optimize_energy_improve": (
                "The SYCL code compiled and ran correctly, but did not reduce energy consumption "
                "sufficiently compared to the baseline on Aurora (Intel Data Center GPU Max). "
                "Apply more aggressive energy optimizations targeting the Intel PVC architecture. "
                "Return ONLY a single fenced code block with the complete revised source. "
                "FOCUS ON high-impact optimizations for Intel PVC: "
                "(A) SLM CACHING: load tiles of frequently accessed global data into "
                "local_accessor<T, 1, sycl::access::mode::read_write> (shared local memory). "
                "(B) SUB-GROUP COALESCING: ensure consecutive work-items in a sub-group "
                "access consecutive memory addresses (64-byte L1 cache lines on PVC). "
                "(C) ND-RANGE TUNING: try local work-group sizes that are multiples of 16 "
                "(SIMD-16 hardware) — typically 64, 128, or 256 for good sub-slice occupancy. "
                "(D) SUB-GROUP DIVERGENCE: refactor conditional branches inside kernels to "
                "branchless forms (masked arithmetic, sycl::min/max) so all work-items in a "
                "sub-group execute the same instructions. "
                "(E) KERNEL FUSION: merge adjacent kernels that operate on the same buffers. "
                "(F) PRECISION: replace double with float or use sycl::half where precision allows. "
                "(G) MINIMIZE BARRIERS: remove unnecessary group_barrier() calls. "
                "CRITICAL CORRECTNESS RULES: "
                "(A) Do NOT alter loop bounds, comparison operators, or index arithmetic. "
                "(B) Do NOT hardcode the verification checksum. "
                "(C) Do NOT stub any simulation function. "
                "(D) Preserve the global sycl::queue and USM memory model. "
                "(E) Preserve ALL stdout output exactly."
            ),
        }

        # ── Stage 0 self-prompting user prompts ───────────────────────────────────
        self._selfprompt_prompts: dict[str, str] = {

            # Shared: summarize context knowledge (guide doc injected by caller)
            "summarize_context_sycl": (
                "You have been provided with a reference guide on SYCL 2020 programming "
                "and migrating CUDA code to SYCL. "
                "Summarize the key concepts from this guide that are most relevant for "
                "translating CUDA C++ code to SYCL 2020 C++ targeting Intel GPUs compiled "
                "with icpx. Cover: memory model and USM (malloc_device/shared/host), "
                "queue and device setup, kernel dispatch (q.submit / parallel_for), "
                "nd_range and thread indexing (get_global_id / get_local_id / get_group_id), "
                "local/shared memory (local_accessor), synchronization (group_barrier), "
                "atomics, and common pitfalls when migrating from CUDA. "
                "Output only the summary."
            ),

            "summarize_context_cuda_energy": (
                "You have been provided with a reference guide on CUDA GPU energy efficiency "
                "optimization techniques. "
                "Summarize the key actionable techniques from this guide that are most relevant "
                "for reducing the energy consumption of a CUDA program running on an NVIDIA A100 "
                "GPU (sm_80, 40 GB HBM2e, 6912 CUDA cores). "
                "Cover: memory access coalescing, shared memory use, occupancy optimization, "
                "warp divergence reduction, kernel fusion, read-only cache (__ldg), arithmetic "
                "precision tradeoffs, synchronization minimization, and prefetching. "
                "For each technique, briefly state when it is most impactful. "
                "Output only the summary."
            ),

            "summarize_context_sycl_energy": (
                "You have been provided with a reference guide on SYCL GPU energy efficiency "
                "optimization for Intel Data Center GPU Max (Ponte Vecchio / PVC). "
                "Summarize the key actionable techniques from this guide that are most relevant "
                "for reducing the energy consumption of a SYCL program on Aurora Intel PVC GPUs. "
                "Cover: sub-group coalescing, shared local memory (SLM) caching, nd-range "
                "occupancy tuning, sub-group divergence reduction, kernel fusion, L1/L2 cache "
                "hints, arithmetic precision, and synchronization minimization. "
                "For each technique, briefly state when it is most impactful. "
                "Output only the summary."
            ),

            # Describe source for translation tasks
            "describe_source_translate": (
                "Analyze the following CUDA C++ source code and produce a structured "
                "technical description covering: "
                "(1) the overall purpose and algorithm, "
                "(2) all CUDA-specific constructs used (cudaMalloc, cudaMemcpy, device "
                "kernels, __shared__, atomics, etc.) and their role in the program, "
                "(3) the structure of GPU kernels and their thread/block organization, "
                "(4) which host functions allocate device memory or launch kernels — "
                "these will require a sycl::queue& q parameter in the SYCL translation, "
                "(5) any loop bounds, comparison operators, or index offsets that appear "
                "asymmetric or potentially incorrect (e.g., `j > 0` instead of `j >= 0` "
                "in a loop starting at j=i) — explicitly flag these as intentional and "
                "note that they MUST be preserved verbatim in the translation. "
                "Output only the structured description, no code."
            ),

            # Describe source for CUDA energy tasks
            "describe_source_cuda_energy": (
                "Analyze the following CUDA C++ source code and produce a structured technical "
                "description focused on energy efficiency opportunities.  Cover: "
                "(1) The overall algorithm and its GPU resource utilization profile "
                "(compute-bound vs memory-bound, random vs sequential access). "
                "(2) All CUDA kernel launch configurations (block/grid dimensions) and whether "
                "they are likely to achieve high occupancy on an A100. "
                "(3) Global memory access patterns in each kernel — are accesses coalesced "
                "(consecutive thread → consecutive address) or scattered? "
                "(4) Shared memory usage (current and potential). "
                "(5) Warp divergence: identify conditional branches inside kernels. "
                "(6) Data types: identify double-precision operations replaceable with float. "
                "(7) Synchronization overhead: count __syncthreads() calls. "
                "(8) Any loop bounds or comparison operators that appear unusual — flag these "
                "as INTENTIONAL and note they MUST be preserved verbatim. "
                "Output only the structured description, no code."
            ),

            # Describe source for SYCL energy tasks
            "describe_source_sycl_energy": (
                "Analyze the following SYCL C++ source code and produce a structured technical "
                "description focused on energy efficiency opportunities on Intel Data Center GPU "
                "Max (Ponte Vecchio / PVC).  Cover: "
                "(1) The overall algorithm and GPU resource utilization profile "
                "(compute-bound vs memory-bound, random vs sequential access). "
                "(2) All SYCL kernel nd_range configurations and likely occupancy on PVC. "
                "(3) Global (USM) memory access patterns — are sub-group accesses coalesced? "
                "(4) Shared local memory (SLM) usage (current and potential). "
                "(5) Sub-group divergence: identify conditional branches inside kernels. "
                "(6) Data types: identify double-precision operations replaceable with float. "
                "(7) Barrier usage: count group_barrier() calls per kernel. "
                "(8) Any loop bounds or comparison operators that appear unusual — flag as "
                "INTENTIONAL and note they MUST be preserved verbatim. "
                "Output only the structured description, no code."
            ),
        }

    # ── Public accessors ─────────────────────────────────────────────────────────

    def get_system_prompt(self, key: str) -> str:
        return self._system_prompts[key]

    def get_codetranslate_prompt(self, key: str) -> str:
        return self._translate_prompts[key]

    def get_code_error_correction_prompt(self, key: str) -> str:
        return self._error_prompts[key]

    def get_selfprompt(self, key: str) -> str:
        return self._selfprompt_prompts[key]

    def get_codeknowledge(self, key: str) -> str:
        """Load a context knowledge document from lassi/context/."""
        filemap = {
            "SYCL_guide":        "SYCL_Programming_Guide.md",
            "CUDA_energy_guide": "CUDA_Energy_Efficiency_Guide.md",
            "SYCL_energy_guide": "SYCL_Energy_Efficiency_Guide.md",
        }
        if key not in filemap:
            raise KeyError(f"Unknown codeknowledge key: {key!r}")
        return _load_context_file(filemap[key])
