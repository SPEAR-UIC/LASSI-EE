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
                "triple backticks (```), and nothing else. The generated SYCL code will be "
                "compiled using AdaptiveCpp (acpp). HARD CONSTRAINTS: (1) Do NOT include or "
                "reference oneAPI, oneDPL, dpct, or SYCLomatic. (2) Do NOT include any headers "
                "under <oneapi/...> or <dpct/...>. (3) Do NOT use oneAPI extensions like "
                "sycl::ext::oneapi::* unless they are part of core SYCL 2020. (4) Use only "
                "standard C++17 + core SYCL headers (e.g., <sycl/sycl.hpp> or <CL/sycl.hpp>) "
                "and implement any needed algorithms manually. (5) Preserve program behavior "
                "and stdout output."
            )
        }

        self._translate_prompts = {
            "CUDA_to_SYCL": (
                "Translate the following parallelized CUDA program to SYCL 2020 C++ that "
                "compiles with AdaptiveCpp (acpp). Requirements: output a single complete .cpp "
                "file only (no explanation). Use only standard C++17 and core SYCL headers "
                "(<sycl/sycl.hpp> or <CL/sycl.hpp>). Do NOT use oneDPL / oneAPI headers "
                "(<oneapi/dpl/...>), dpct, SYCLomatic, or sycl::ext::oneapi::* APIs. Preserve "
                "the original program’s behavior and stdout output."
            )
        }

        self._error_prompts = {
            "CUDA_to_SYCL_error": (
                "Fix the SYCL translation below so it compiles and runs correctly with "
                "AdaptiveCpp (acpp), while preserving the original program behavior and stdout. "
                "Return ONLY a single fenced code block (```...```) containing the full corrected "
                "source file. Do not include any explanation."
            )
        }

    def get_system_prompt(self, key: str) -> str:
        return self._system_prompts[key]

    def get_codetranslate_prompt(self, key: str) -> str:
        return self._translate_prompts[key]

    def get_code_error_correction_prompt(self, key: str) -> str:
        return self._error_prompts[key]
