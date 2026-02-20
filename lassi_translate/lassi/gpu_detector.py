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
    path = shutil.which(name)
    if path:
        return path
    # On Aurora, sycl-ls lives next to icpx rather than on PATH after conda activate.
    icpx = shutil.which("icpx")
    if icpx:
        candidate = os.path.join(os.path.dirname(icpx), name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def detect_gpu_vendors() -> list[dict[str, Any]]:
    vendors: list[dict[str, Any]] = []

    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "-L"])
        count = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        if count:
            vendors.append({"vendor": "nvidia", "count": count, "tool": "nvidia-smi"})

    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showid"])
        count = sum(1 for line in out.splitlines() if "GPU" in line and ":" in line)
        if count:
            vendors.append({"vendor": "amd", "count": count, "tool": "rocm-smi"})

    sycl_ls = _find_tool("sycl-ls")
    if sycl_ls:
        out = _run([sycl_ls])
        count = sum(1 for line in out.splitlines() if ":gpu" in line.lower())
        if count:
            vendors.append({"vendor": "intel", "count": count, "tool": "sycl-ls"})

    if not any(v["vendor"] == "intel" for v in vendors) and shutil.which("xpu-smi"):
        out = _run(["xpu-smi", "discovery"])
        count = sum(1 for line in out.splitlines() if "Device Name:" in line)
        if count:
            vendors.append({"vendor": "intel", "count": count, "tool": "xpu-smi"})

    return vendors
