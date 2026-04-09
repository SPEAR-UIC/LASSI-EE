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
    # Fallback for local development (AdaptiveCpp).
    # On ALCF resources (Aurora, Polaris) SYCL_COMPILER is always set by the
    # corresponding run_*_demo.sh wrapper before the pipeline is invoked.
    return SYCLToolchain(
        compiler="/home/vchirio/adaptivecpp/bin/acpp",
        cxxflags="-O3 -std=c++17",
    )
