"""Prompt templates used by the translation pipeline.

This standalone bundle supports only CUDA→SYCL translation and compile/run
self-correction.
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
            "CUDA_to_SYCL": (
                "You are a professional coding AI assistant that specializes in translating "
                "CUDA C++ to SYCL C++ (SYCL 2020). Provide only the complete translated code "
                "file with no explanation. Output MUST be a single fenced code block using "
                "triple backticks (```cpp ... ```), and nothing else. The generated SYCL code "
                "will be compiled using icpx (Intel's oneAPI DPC++/C++ compiler, SYCL 2020). "
                "HARD CONSTRAINTS: "
                "(1) Translate EVERY function from the source — do NOT skip, stub, or simplify any function. "
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
                "(8) Preserve ALL program behavior, logic, and stdout output exactly."
            )
        }

        self._translate_prompts = {
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
                "printf() call, especially the 'Verification checksum:' output line."
            )
        }

        self._error_prompts = {
            "CUDA_to_SYCL_error": (
                "Fix the SYCL translation so it compiles and runs correctly with icpx "
                "(Intel oneAPI DPC++/C++ compiler), while preserving ALL original program "
                "behavior and stdout output. "
                "CRITICAL: Return a COMPLETE faithful translation of the entire original "
                "source — do NOT generate stubs or simplified versions. "
                "Return ONLY a single fenced code block (```cpp\\n...\\n```) with the full "
                "corrected source. Do not include any explanation."
            )
        }

        # Self-prompting prompts — used before Stage 1 to build enriched context,
        # following the methodology from the original LASSI paper (IEEE CLUSTER 2024).
        self._selfprompt_prompts = {
            "summarize_context": (
                "You are an expert in parallel programming and GPU computing. "
                "You have been provided with a reference guide on SYCL 2020 programming "
                "and migrating CUDA code to SYCL. "
                "Summarize the key concepts from this guide that are most relevant for "
                "translating CUDA C++ code to SYCL 2020 C++ targeting Intel GPUs compiled "
                "with icpx. Focus on: memory model and USM, queue and device setup, kernel "
                "dispatch syntax, thread indexing equivalents, local/shared memory, "
                "synchronization primitives, and common pitfalls to avoid. "
                "Be concise and actionable. Output only the summary."
            ),
            "describe_source": (
                "You are an expert in parallel programming. Analyze the following CUDA C++ "
                "source code and produce a structured description of it. Identify: "
                "(1) the overall purpose and algorithm, "
                "(2) all CUDA-specific constructs used (cudaMalloc, cudaMemcpy, kernel "
                "launches, __global__, __device__, __shared__, threadIdx/blockIdx/blockDim, "
                "atomics, synchronization), "
                "(3) every function and whether it will need a sycl::queue& q parameter "
                "when translated to SYCL, "
                "(4) any potential translation challenges or pitfalls. "
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
            "SYCL_guide": "SYCL_Programming_Guide.md",
        }
        if key not in filemap:
            raise KeyError(f"Unknown codeknowledge key: {key!r}")
        return _load_context_file(filemap[key])
