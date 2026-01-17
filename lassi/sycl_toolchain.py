"""SYCL toolchain helpers.

This project currently assumes CUDA/HIP toolchains are installed.
For SYCL, we provide *placeholders* so the notebook can run end-to-end
up to the point where a SYCL compiler is required.

Once you install a SYCL compiler, update the defaults in
[`lassi.sycl_toolchain.get_sycl_toolchain()`](LASSI-EE/lassi/sycl_toolchain.py:1).

Supported compilers (common options):
- Intel oneAPI DPC++ (icpx)
- LLVM/Clang SYCL (clang++ with -fsycl)
- AdaptiveCpp (acpp)

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SYCLToolchain:
    """Configuration for compiling SYCL code."""

    compiler: str
    cxxflags: str
    ldflags: str = ""
    env_hint: str = ""


def get_sycl_toolchain() -> SYCLToolchain:
    """Return the SYCL toolchain configuration.

    This is intentionally a placeholder.

    Edit this function after installing a SYCL compiler.

    Examples:

    1) Intel oneAPI DPC++ (icpx):
       compiler = "icpx"
       cxxflags = "-fsycl -O3 -std=c++17"

       For NVIDIA backend (requires oneAPI + CUDA plugin support):
       cxxflags = "-fsycl -O3 -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda"

    2) LLVM/Clang SYCL:
       compiler = "clang++"
       cxxflags = "-fsycl -O3 -std=c++17"

    3) AdaptiveCpp:
       compiler = "acpp"
       cxxflags = "-O3 -std=c++17"

    return SYCLToolchain(
        compiler="<SYCL_COMPILER>",
        cxxflags="<SYCL_CXXFLAGS>",
        ldflags="<SYCL_LDFLAGS>",
        env_hint=(
            "Set your SYCL environment here if needed (e.g., ONEAPI_DEVICE_SELECTOR). "
            "This is a placeholder until a SYCL toolchain is installed."
        ),
    )"""

    # AdaptiveCpp (user-space install) default for this environment.
    # If you installed AdaptiveCpp elsewhere, update this path.
    return SYCLToolchain(
        compiler="/home/vchirio/adaptivecpp/bin/acpp",
        cxxflags="-O3 -std=c++17",
        ldflags="",
        env_hint="export PATH=/home/vchirio/adaptivecpp/bin:$PATH",
    )
