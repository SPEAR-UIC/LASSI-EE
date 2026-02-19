"""SYCL toolchain configuration.

The runner uses this to decide how to compile generated SYCL code.

Environment variable overrides (useful for Aurora/icpx):
  SYCL_COMPILER   - path or name of the SYCL compiler (e.g. "icpx")
  SYCL_CXXFLAGS   - compiler flags (default: "-O2 -std=c++17 -fsycl" when SYCL_COMPILER is set)
  SYCL_LDFLAGS    - linker flags (default: "")
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SYCLToolchain:
    compiler: str
    cxxflags: str
    ldflags: str = ""


def get_sycl_toolchain() -> SYCLToolchain:
    """Return the SYCL toolchain used for generated code compilation."""

    compiler = os.environ.get("SYCL_COMPILER")
    if compiler:
        cxxflags = os.environ.get("SYCL_CXXFLAGS", "-O2 -std=c++17 -fsycl")
        ldflags = os.environ.get("SYCL_LDFLAGS", "")
        return SYCLToolchain(compiler=compiler, cxxflags=cxxflags, ldflags=ldflags)

    # AdaptiveCpp default for non-Aurora environments.
    return SYCLToolchain(
        compiler="/home/vchirio/adaptivecpp/bin/acpp",
        cxxflags="-O3 -std=c++17",
        ldflags="",
    )
