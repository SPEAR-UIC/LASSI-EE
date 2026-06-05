"""ALCF resource definitions: hosts, default paths, worker scripts, framework compatibility."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceConfig:
    name: str
    host: str
    amsc_resource: str
    amsc_fs: str
    default_dir: str
    filesystems: str
    gpu_vendor: str


RESOURCES = {
    "polaris": ResourceConfig(
        "polaris", "polaris.alcf.anl.gov", "Polaris", "Eagle",
        "/eagle/EE-ECP/LASSI-EE/lassi_jobs", "home:eagle", "NVIDIA",
    ),
    "aurora": ResourceConfig(
        "aurora", "aurora.alcf.anl.gov", "Aurora", "Flare",
        "/flare/EE-ECP/LASSI-EE/lassi_jobs", "home:flare", "Intel",
    ),
}

# Worker script per (resource, job_type).  Value: (script_filename, task_arg).
WORKERS = {
    "polaris": {
        "translate":         ("polaris_worker.sh",            "translate"),
        "energy_cuda":       ("polaris_worker.sh",            "cuda_energy"),
        "persistent_worker": ("polaris_persistent_worker.sh", ""),
    },
    "aurora": {
        "translate":         ("aurora_worker.sh",             "translate"),
        "persistent_worker": ("aurora_persistent_worker.sh",  ""),
    },
}

# Which source frameworks each resource can natively compile + run.
FRAMEWORKS = {"polaris": ("cuda", "sycl"), "aurora": ("sycl",)}


def can_run(resource: str, framework: str) -> bool:
    """True iff `framework` source can be compiled+run on `resource`."""
    return framework in FRAMEWORKS.get(resource, ())
