from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SYCLToolchain:
    compiler: str
    cxxflags: str
    ldflags: str = ""


def get_sycl_toolchain() -> SYCLToolchain:
    compiler = os.environ.get("SYCL_COMPILER")
    if compiler:
        return SYCLToolchain(
            compiler=compiler,
            cxxflags=os.environ.get("SYCL_CXXFLAGS", "-O2 -std=c++17 -fsycl"),
            ldflags=os.environ.get("SYCL_LDFLAGS", ""),
        )
    # Default for non-Aurora environments (AdaptiveCpp).
    return SYCLToolchain(
        compiler="/home/vchirio/adaptivecpp/bin/acpp",
        cxxflags="-O3 -std=c++17",
    )
