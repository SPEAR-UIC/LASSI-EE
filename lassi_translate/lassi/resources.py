"""ALCF resource definitions and worker-script lookup tables."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceConfig:
    name:          str
    host:          str
    amsc_resource: str
    amsc_fs:       str
    default_dir:   str
    filesystems:   str
    gpu_vendor:    str


RESOURCES: dict[str, ResourceConfig] = {
    "polaris": ResourceConfig(
        name          = "polaris",
        host          = "polaris.alcf.anl.gov",
        amsc_resource = "Polaris",
        amsc_fs       = "Eagle",
        # Must live under /eagle (or /lus/eagle) — the amsc Eagle filesystem
        # resource rejects any other path with a 400 ("Allowed base paths for
        # filesystem Eagle are: /eagle, /lus/eagle").
        default_dir   = "/eagle/EE-ECP/LASSI-EE/lassi_jobs",
        filesystems   = "home:eagle",
        gpu_vendor    = "NVIDIA",
    ),
    "aurora": ResourceConfig(
        name          = "aurora",
        host          = "aurora.alcf.anl.gov",
        amsc_resource = "Aurora",
        amsc_fs       = "Flare",
        # Flare's IRI registration is incomplete (see lassi/alcf.py — the
        # fs_resource lookup falls back to None on Aurora), but we still pin
        # to /flare so the path is sensible for SSH and PBS submission.
        default_dir   = "/flare/EE-ECP/LASSI-EE/lassi_jobs",
        filesystems   = "home:flare",
        gpu_vendor    = "Intel",
    ),
}

# Worker-script mapping per resource.  Each value is (script_filename, task_arg).
# task_arg is the first positional argument to the one-shot worker;
# persistent_worker uses "" because the task arrives via trigger.json.
WORKERS: dict[str, dict[str, tuple[str, str]]] = {
    "polaris": {
        "translate":         ("polaris_worker.sh",            "translate"),
        "energy_cuda":       ("polaris_worker.sh",            "cuda_energy"),
        "energy_sycl":       ("polaris_worker.sh",            "sycl_energy"),
        "persistent_worker": ("polaris_persistent_worker.sh", ""),
    },
    "aurora": {
        "translate":         ("aurora_worker.sh",             "translate"),
        "energy_sycl":       ("aurora_worker.sh",             "sycl_energy"),
        "persistent_worker": ("aurora_persistent_worker.sh",  ""),
    },
}

# Which source frameworks each resource can natively compile + execute.
# Used by the demo to decide whether a baseline measurement of the original
# source is possible (e.g. CUDA can't run on Aurora's Intel PVC GPUs).
FRAMEWORKS: dict[str, tuple[str, ...]] = {
    "polaris": ("cuda", "sycl"),   # SYCL on Polaris uses icpx's CUDA backend
    "aurora":  ("sycl",),
}


def can_run(resource: str, framework: str) -> bool:
    """Return True if `resource` can natively compile + execute `framework` code."""
    return framework in FRAMEWORKS.get(resource, ())
