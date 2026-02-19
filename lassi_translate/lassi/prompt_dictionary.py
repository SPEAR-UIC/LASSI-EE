"""Prompt templates used by the translation pipeline.

This standalone bundle supports only CUDA→SYCL translation and compile/run
self-correction.
"""

from __future__ import annotations


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
                "(6) Use sycl::queue q{sycl::gpu_selector_v}; (not the deprecated sycl::default_selector). "
                "(7) Preserve ALL program behavior, logic, and stdout output exactly."
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

    def get_system_prompt(self, key: str) -> str:
        return self._system_prompts[key]

    def get_codetranslate_prompt(self, key: str) -> str:
        return self._translate_prompts[key]

    def get_code_error_correction_prompt(self, key: str) -> str:
        return self._error_prompts[key]
