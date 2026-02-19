"""GPU detection.

Used by the runner and pipeline to report what GPU vendor(s) are available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Optional


def _run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return p.stdout or ""


def _find_tool(name: str) -> Optional[str]:
    """Find a tool by name.

    Checks PATH first, then the directory containing icpx (on Aurora the
    Intel oneAPI tools share a bin/ directory that may not always be on PATH
    after conda activate).
    """
    path = shutil.which(name)
    if path:
        return path
    # Fallback: look in the same bin/ as icpx (common on Aurora).
    icpx = shutil.which("icpx")
    if icpx:
        candidate = os.path.join(os.path.dirname(icpx), name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def detect_gpu_vendors() -> list[dict[str, Any]]:
    """Return a list like: [{'vendor': 'nvidia', 'count': 1, 'tool': 'nvidia-smi'}]."""

    vendors: list[dict[str, Any]] = []

    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "-L"])
        count = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        if count:
            vendors.append({"vendor": "nvidia", "count": count, "tool": "nvidia-smi"})

    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showid"])
        # rocm-smi output formats vary; count lines containing "GPU" and ":".
        count = sum(1 for line in out.splitlines() if "GPU" in line and ":" in line)
        if count:
            vendors.append({"vendor": "amd", "count": count, "tool": "rocm-smi"})

    # Intel: try sycl-ls (may live next to icpx rather than on PATH on Aurora).
    sycl_ls = _find_tool("sycl-ls")
    if sycl_ls:
        out = _run([sycl_ls])
        # sycl-ls lists SYCL platforms/devices, e.g.:
        #   [ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Data Center GPU Max 1550 ...
        count = sum(1 for line in out.splitlines() if ":gpu:" in line.lower())
        if count:
            vendors.append({"vendor": "intel", "count": count, "tool": "sycl-ls"})

    # Intel fallback: xpu-smi is Aurora's native Data Center GPU management CLI.
    if not any(v["vendor"] == "intel" for v in vendors) and shutil.which("xpu-smi"):
        out = _run(["xpu-smi", "discovery"])
        # Each device appears as one "Device Name:" line in xpu-smi discovery output.
        count = sum(1 for line in out.splitlines() if "Device Name:" in line)
        if count:
            vendors.append({"vendor": "intel", "count": count, "tool": "xpu-smi"})

    return vendors
