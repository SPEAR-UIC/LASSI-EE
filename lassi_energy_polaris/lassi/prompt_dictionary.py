"""Prompt templates used by the CUDA energy optimization pipeline.

This standalone bundle supports CUDA energy efficiency optimization and
compile/run self-correction.  The target language is still CUDA — the goal
is to produce a more energy-efficient CUDA program that compiles with nvcc,
runs correctly on an NVIDIA A100 GPU, and consumes less GPU energy than the
original.
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
        self._system_prompts = {
            # Stage 0 self-prompting calls produce plain prose.
            # Must NOT instruct the model to output a fenced code block.
            "selfprompt": (
                "You are an expert in GPU computing, CUDA C++ programming, and GPU energy "
                "efficiency optimization.  You have deep knowledge of NVIDIA GPU architecture "
                "(Ampere/A100 and earlier), the CUDA programming model, GPU memory hierarchies, "
                "warp execution, occupancy, and power consumption patterns.  Provide clear, "
                "accurate, and detailed technical responses in plain prose."
            ),

            # System prompt for all code-generation calls (Stages 1 and 2).
            "CUDA_optimize_energy": (
                "You are a professional GPU performance and energy efficiency expert who "
                "specializes in optimizing CUDA C++ programs for reduced energy consumption "
                "on NVIDIA GPUs.  Output MUST be a single fenced code block using triple "
                "backticks (```cuda ... ``` or ```cpp ... ```), and nothing else.  The "
                "generated CUDA code will be compiled with nvcc on a Polaris A100 GPU "
                "(sm_80 architecture, 400 W TDP, 40 GB HBM2e, 6912 CUDA cores). "
                ""
                "PRIMARY GOAL — reduce GPU energy consumption (Joules) while preserving "
                "exact numerical correctness.  Energy = Power × Time; optimizations that "
                "reduce either average power draw OR execution time (or both) are valid. "
                ""
                "HARD CONSTRAINTS — correctness: "
                "(1) Preserve ALL program behavior, logic, and stdout output exactly. "
                "NEVER alter loop bounds, comparison operators, index arithmetic, or "
                "numerical computations — even if they appear to be off-by-one or redundant. "
                "These boundary conditions and calculations are intentional algorithmic choices "
                "that directly determine the output checksum. "
                "(2) Translate EVERY function completely — do NOT skip, stub, or simplify "
                "any function.  Every simulation kernel MUST be fully implemented. "
                "(3) NEVER hardcode, guess, or return a pre-known verification checksum value "
                "(e.g., 952131, 945990, 941535, or any other constant).  The checksum MUST "
                "be produced by actually executing all GPU kernel computations at runtime. "
                "(4) NEVER call (void)param; casts to suppress unused-parameter warnings "
                "inside simulation functions — this signals a function stub. "
                ""
                "HARD CONSTRAINTS — CUDA validity: "
                "(5) Output valid CUDA C++ that compiles with:  nvcc -O3 -arch=sm_80 -std=c++17 "
                "(6) Do NOT introduce new external library dependencies beyond the standard "
                "CUDA toolkit headers already included in the source. "
                "(7) Do NOT change the program's command-line interface or output format. "
                ""
                "ENERGY OPTIMIZATION STRATEGIES to apply (use judgment — not all are always "
                "applicable or beneficial): "
                "(A) Global memory coalescing: ensure threads in a warp access consecutive "
                "memory addresses.  Reorganize data layouts (AoS → SoA) when beneficial. "
                "(B) Shared memory: cache frequently re-read global memory into __shared__ "
                "to avoid repeated DRAM accesses across the tile. "
                "(C) Read-only cache: use __ldg() or const __restrict__ for read-only "
                "arrays to exploit the L1 texture cache path. "
                "(D) Occupancy: tune thread block dimensions (multiples of 32, typically "
                "128–256 threads/block) to maximize SM utilization without excessive "
                "register spilling. "
                "(E) Warp divergence: restructure branches so threads within a warp take "
                "the same path when possible.  Use branchless arithmetic (e.g., ternary "
                "expressions, masked operations) to reduce divergence. "
                "(F) Kernel fusion: merge consecutive kernels that operate on the same data "
                "to eliminate redundant global memory round-trips. "
                "(G) Arithmetic precision: replace double with float where the application "
                "permits, and use fast-math intrinsics (__expf, __logf, __sinf, etc.) "
                "when precision loss is acceptable. "
                "(H) Loop unrolling: add #pragma unroll for small inner loops to reduce "
                "branch overhead and expose more instruction-level parallelism. "
                "(I) Asynchronous prefetching: use __pipeline_memcpy_async or cudaMemcpyAsync "
                "to overlap data transfers with computation where applicable. "
                "(J) Minimize synchronization: reduce __syncthreads() calls to only where "
                "strictly required by data dependencies. "
                "(K) Minimize idle cycles: avoid unnecessary global memory barriers, "
                "redundant atomics, or serialization points that stall warps."
            ),
        }

        # Stage 1 prompt: initial energy optimization request.
        self._translate_prompts = {
            "CUDA_optimize_energy": (
                "Optimize the following CUDA program for GPU energy efficiency while preserving "
                "its exact numerical correctness (including the XSBench verification checksum). "
                "The optimized code must compile with:  nvcc -O3 -arch=sm_80 -std=c++17 "
                "on a Polaris A100 GPU (sm_80). "
                ""
                "Apply the energy optimization strategies most likely to reduce energy for this "
                "specific workload — memory-bound kernels benefit most from coalescing and "
                "shared memory; compute-bound kernels benefit from reduced divergence and "
                "better occupancy. "
                ""
                "CRITICAL CORRECTNESS RULES (violations cause automatic rejection): "
                "(A) Do NOT alter loop bounds, comparison operators (`> vs >=`), or index "
                "arithmetic.  These determine the numerical output and checksum. "
                "(B) Do NOT hardcode any verification checksum value (e.g., return 952131ULL). "
                "The checksum MUST be computed by the GPU kernels at runtime. "
                "(C) Every simulation function MUST contain a real CUDA kernel launch that "
                "performs the full computation — do NOT add (void)param; stubs. "
                "(D) Output a single complete .cu file — no explanation, no partial listings."
            ),
        }

        # Stage 2 correction prompts.
        self._error_prompts = {
            # Used when there is a compile error, runtime error, or checksum mismatch.
            "CUDA_optimize_energy_error": (
                "Fix the CUDA code so it compiles and runs correctly with nvcc "
                "(nvcc -O3 -arch=sm_80 -std=c++17), while preserving ALL original program "
                "behavior and stdout output.  Keep any valid energy optimizations that were "
                "already present in the code. "
                "Return ONLY a single fenced code block (```cuda\\n...\\n``` or ```cpp\\n...\\n```) "
                "with the complete corrected source.  Do not include any explanation. "
                ""
                "STRICTLY FORBIDDEN — these cause automatic rejection: "
                "(A) Do NOT hardcode any verification checksum value.  Statements like "
                "'return 952131ULL', 'return 945990ULL', or any other constant returned from "
                "a simulation function are forbidden.  The checksum MUST be produced by "
                "running the actual CUDA kernels. "
                "(B) Every simulation kernel function MUST contain a real CUDA kernel launch "
                "(__global__ function invocation via <<<...>>>) that performs the full "
                "cross-section lookup computation.  Do NOT stub these out with (void)param; casts. "
                "(C) load_num_nucs(), load_mats(), and load_concs() MUST faithfully reproduce "
                "the physics-accurate material data from the original CUDA source. "
                "(D) read_CLI() MUST parse argc/argv — do NOT add (void)argc; (void)argv; stubs. "
                "(E) Do NOT alter loop bounds, comparison operators, or index arithmetic — "
                "a `j > 0` loop in the original MUST remain `j > 0`."
            ),

            # Used when compile and run succeed but energy improvement is insufficient.
            "CUDA_optimize_energy_improve": (
                "The CUDA code compiled and ran correctly, but did not reduce energy consumption "
                "sufficiently compared to the original baseline.  Apply more aggressive energy "
                "optimizations to reduce GPU power draw and/or execution time. "
                "Return ONLY a single fenced code block with the complete revised source. "
                ""
                "FOCUS ON these high-impact optimizations for memory-bound workloads like "
                "XSBench (random memory access pattern across large cross-section tables): "
                "(A) SHARED MEMORY TILING: identify loops where multiple threads in a block "
                "access overlapping or nearby memory regions — cache these into __shared__ "
                "to replace repeated global memory reads with fast SRAM accesses. "
                "(B) __ldg() READ-ONLY CACHE: for arrays that are only read (never written) "
                "inside kernels, use __ldg(&arr[i]) to route reads through the L1 texture "
                "cache (256 KB on A100) instead of L2/DRAM. "
                "(C) MEMORY LAYOUT (AoS → SoA): if structs-of-arrays reorganization enables "
                "coalesced accesses, apply it consistently throughout the code. "
                "(D) WARP DIVERGENCE REDUCTION: replace if/else branches inside kernels with "
                "branchless equivalents (masks, min/max, ternary) where correctness allows. "
                "(E) BLOCK SIZE TUNING: experiment with block dimensions that are multiples of "
                "32 (e.g., 128, 256) to maximize SM occupancy without excessive register spills. "
                "(F) PRECISION: replace double-precision floating-point operations with float "
                "where the algorithm tolerates it, and use CUDA intrinsics (__expf, __logf, "
                "__rsqrtf) for faster, lower-power computation. "
                "(G) KERNEL FUSION: if multiple kernels read the same data, fuse them to avoid "
                "redundant DRAM round-trips. "
                ""
                "CRITICAL CORRECTNESS RULES (must not be violated): "
                "(A) Do NOT alter loop bounds, comparison operators, or index arithmetic. "
                "(B) Do NOT hardcode the verification checksum — it must be computed at runtime. "
                "(C) Do NOT stub any simulation function. "
                "(D) Preserve ALL stdout output exactly."
            ),
        }

        # Stage 0 self-prompting prompts (plain prose, no code blocks).
        self._selfprompt_prompts = {
            "summarize_context": (
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
            "describe_source": (
                "Analyze the following CUDA C++ source code and produce a structured technical "
                "description focused on energy efficiency opportunities.  Cover: "
                "(1) The overall algorithm and its GPU resource utilization profile "
                "(compute-bound vs memory-bound, random vs sequential access). "
                "(2) All CUDA kernel launch configurations (block/grid dimensions) and whether "
                "they are likely to achieve high occupancy on an A100. "
                "(3) Global memory access patterns in each kernel — are accesses coalesced "
                "(consecutive thread → consecutive address) or scattered?  Identify the "
                "highest-bandwidth kernels. "
                "(4) Shared memory usage (current and potential).  Flag kernels where multiple "
                "threads in a block re-read the same global memory locations — these are "
                "candidates for shared memory caching. "
                "(5) Warp divergence: identify conditional branches inside kernels where threads "
                "in the same warp may diverge, causing serialization. "
                "(6) Data types: identify any double-precision operations that might be "
                "replaceable with float or CUDA intrinsics. "
                "(7) Synchronization overhead: count __syncthreads() calls and note whether any "
                "appear unnecessary given the data flow. "
                "(8) Any loop bounds, comparison operators, or index offsets that appear "
                "asymmetric or unusual — flag these as INTENTIONAL and note they MUST be "
                "preserved verbatim (e.g., `j > 0` not `j >= 0`). "
                "Output only the structured description, no code."
            ),
        }

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
            "CUDA_energy_guide": "CUDA_Energy_Efficiency_Guide.md",
        }
        if key not in filemap:
            raise KeyError(f"Unknown codeknowledge key: {key!r}")
        return _load_context_file(filemap[key])
