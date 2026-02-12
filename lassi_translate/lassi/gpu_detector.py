"""GPU detection.

Used by the runner and pipeline to report what GPU vendor(s) are available.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    return p.stdout or ""


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

    return vendors
