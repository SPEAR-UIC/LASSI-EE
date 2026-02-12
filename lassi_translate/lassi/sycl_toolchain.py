"""SYCL toolchain configuration.

The runner uses this to decide how to compile generated SYCL code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SYCLToolchain:
    compiler: str
    cxxflags: str
    ldflags: str = ""


def get_sycl_toolchain() -> SYCLToolchain:
    """Return the SYCL toolchain used for generated code compilation."""

    # AdaptiveCpp default for this environment.
    return SYCLToolchain(
        compiler="/home/vchirio/adaptivecpp/bin/acpp",
        cxxflags="-O3 -std=c++17",
        ldflags="",
    )
