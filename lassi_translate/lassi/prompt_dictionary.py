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
            # Used for Stage 0 self-prompting calls (context summarization and
            # source code description).  Must NOT instruct the model to output
            # a fenced code block — these calls produce plain prose.
            "selfprompt": (
                "You are an expert in parallel programming, GPU computing, and the CUDA "
                "and SYCL programming models. Provide clear, accurate, and detailed "
                "technical responses in plain prose."
            ),
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
                "printf() call, especially the 'Verification checksum:' output line. "
                "STRICTLY FORBIDDEN — your translation will be automatically rejected if it "
                "violates any of these rules: "
                "(A) Do NOT hardcode any verification checksum value. The checksum (e.g., the "
                "value printed on the 'Verification checksum:' line) MUST be computed by the "
                "GPU kernel at runtime, NOT returned as a literal constant (e.g., 'return 952131ULL'). "
                "(B) Every run_*_simulation_* function MUST contain a real q.submit(parallel_for(...)) "
                "kernel that performs the actual cross-section lookup computation — do NOT delegate "
                "to another function that returns a constant, and do NOT add (void)GSD; or (void)in; "
                "casts to silence unused-parameter warnings. "
                "(C) load_num_nucs(), load_mats(), and load_concs() MUST be faithful translations "
                "of the source — do NOT replace physics-accurate material data with trivial loops "
                "like 'arr[i] = i % 12' or 'arr[i] = 1.0'. "
                "(D) Do NOT alter loop bounds, comparison operators, or index arithmetic. "
                "A loop written as `j > 0` in the source MUST remain `j > 0` in the translation — "
                "never change `>` to `>=`, `<` to `<=`, or any similar operator. These boundary "
                "conditions directly affect numerical results and are intentional."
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
                "corrected source. Do not include any explanation. "
                "STRICTLY FORBIDDEN in your corrected code — these will cause automatic rejection: "
                "(A) Do NOT hardcode any verification checksum value. Statements like "
                "'return 952131ULL', 'return 945990ULL', 'return 941535ULL', 'return 954318ULL', "
                "or any other constant returned from a simulation function are forbidden. "
                "The checksum MUST be produced by actually executing GPU kernels and reducing "
                "the per-thread verification results. "
                "(B) Every run_*_simulation_* function MUST contain a real q.submit(parallel_for(...)) "
                "kernel body that performs the full cross-section lookup computation. "
                "Do NOT stub these out with (void)GSD; (void)in; or by forwarding to another "
                "function that returns a constant. "
                "(C) load_num_nucs(), load_mats(), and load_concs() MUST faithfully reproduce "
                "the physics-accurate material data from the original CUDA source — do NOT "
                "replace them with trivial loops or placeholder values. "
                "(D) read_CLI() MUST parse argc/argv and apply all command-line options — "
                "do NOT ignore argc/argv with (void)argc; (void)argv;. "
                "(E) Do NOT alter loop bounds, comparison operators, or index arithmetic — "
                "even if they appear to be off-by-one errors. A `j > 0` loop condition in "
                "the original MUST stay `j > 0`, not become `j >= 0`. These are numerically "
                "significant algorithmic decisions, not typos."
            )
        }

        # Self-prompting prompts — used before Stage 1 to build enriched context,
        # following the methodology from the original LASSI paper (IEEE CLUSTER 2024).
        # These are USER prompts; they are paired with the "selfprompt" system prompt,
        # NOT the code-translation system prompt.
        self._selfprompt_prompts = {
            "summarize_context": (
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
            "describe_source": (
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
