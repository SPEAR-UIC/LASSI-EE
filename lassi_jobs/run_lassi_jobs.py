#!/usr/bin/env python3
"""LASSI-Jobs — Unified LLM-driven GPU code translation and energy optimization.

Combines the capabilities of lassi_translate_polaris and lassi_energy_polaris into
a single command-line tool that supports multiple HPC resources, framework pairs,
and task modes.

LLM backend
-----------
Uses the ANL Argo gateway (apps.inside.anl.gov/argoapi/...).
Accessible from any machine on the ANL network — no ALCF login needed for LLM.
Set ARGO_USER to your ANL username (default: $USER).
Set ARGO_API_KEY if your deployment requires a Bearer token.

Architecture
------------
  Stage 0 : Self-prompting context builder     — local Argo LLM call
  Stage 1 : Initial code generation            — local Argo LLM call
  Stage 2 : Self-correction loop (up to 15 iterations):
              1. Write generated code locally
              2. rsync code to Eagle/Flare filesystem on the target resource
              3. Submit PBS job: worker.sh compiles + runs (+ measures energy)
              4. Poll until job reaches terminal state
              5. Read build / run / energy logs from filesystem via amsc_client
              6. Parse: compile error / runtime error / checksum mismatch /
                        insufficient energy improvement
              7. If error: generate correction via local Argo LLM → repeat
              8. If success criteria met: done

Tasks
-----
  translate : CUDA → SYCL translation with correctness verification
  energy    : GPU energy optimization of existing code:
                - Polaris: CUDA → CUDA (nvcc, nvidia-smi power monitoring)
                - Aurora:  SYCL → SYCL (icpx, xpu-smi power monitoring)
  both      : translate first (CUDA → SYCL), then measure energy of SYCL
              vs original CUDA baseline on the same resource

Resources
---------
  polaris : Polaris @ ALCF — NVIDIA A100 GPUs, Eagle filesystem, icpx / nvcc
  aurora  : Aurora  @ ALCF — Intel Data Center GPU Max (PVC), Flare filesystem,
            icpx + Level Zero

Supported combinations
----------------------
  --resource polaris --task translate  : CUDA → SYCL, icpx (CUDA backend, sm_80)
  --resource polaris --task energy     : CUDA energy optimization, nvcc + nvidia-smi
  --resource polaris --task both       : translate then SYCL energy on Polaris A100
  --resource aurora  --task translate  : CUDA → SYCL, icpx (Level Zero / PVC)
  --resource aurora  --task energy     : SYCL energy optimization, icpx + xpu-smi
  --resource aurora  --task both       : translate then SYCL energy on Aurora PVC

Usage
-----
  # Polaris: translate XSBench CUDA → SYCL
  python run_lassi_jobs.py --resource polaris --task translate \\
      --alcf-username vchirio --model gpt4o

  # Polaris: optimize XSBench CUDA for energy efficiency
  python run_lassi_jobs.py --resource polaris --task energy \\
      --alcf-username vchirio --model gpt4o --min-energy-gain 5.0

  # Aurora: translate XSBench CUDA → SYCL
  python run_lassi_jobs.py --resource aurora --task translate \\
      --alcf-username vchirio --model gpt4o

  # Aurora: SYCL energy optimization (requires pre-existing SYCL source)
  python run_lassi_jobs.py --resource aurora --task energy \\
      --alcf-username vchirio --model gpt4o --sycl-source /path/to/Main.cpp

  # Polaris: full pipeline (translate + energy measurement of SYCL vs CUDA)
  python run_lassi_jobs.py --resource polaris --task both \\
      --alcf-username vchirio --model gpt4o --skip-sync

  # Disable self-prompting, custom queue
  python run_lassi_jobs.py --resource polaris --task translate \\
      --alcf-username vchirio --no-self-prompting --queue prod

Argo model names (examples)
---------------------------
  gpt4o         GPT-4o via ANL Argo gateway
  gpt41         GPT-4.1
  claudesonnet4 Claude Sonnet 4
  gpto3mini     o3-mini
  gpt55         GPT-5 (when available)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lassi.llm_client as _llm
from lassi.format_colors import color

# ──────────────────────────────────────────────────────────────────────────────
# Config file loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    """Execute a Python config file; return its top-level variables as a dict."""
    if not path.exists():
        return {}
    try:
        ns: dict = {}
        exec(path.read_text(), ns)  # noqa: S102
        return ns.get("config", {})
    except Exception as exc:
        print(f"Warning: could not load config {path}: {exc}", file=sys.stderr)
        return {}

from lassi.hpc_session import HPCSession, ResourceConfig
from lassi.prompt_dictionary import PromptDictionary

_B = color.BOLD
_G = color.GREEN
_R = color.RED
_E = color.END
_SEP  = "=" * 72
_THIN = "─" * 72

_SCRIPT_DIR = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# Resource configuration
# ──────────────────────────────────────────────────────────────────────────────

_RESOURCES: dict[str, ResourceConfig] = {
    "polaris": ResourceConfig(
        name          = "polaris",
        host          = "polaris.alcf.anl.gov",
        amsc_resource = "Polaris",
        amsc_fs       = "Eagle",
        default_dir   = "/eagle/EE-ECP/LASSI-EE/lassi_jobs",
        filesystems   = "home:eagle",
        gpu_vendor    = "NVIDIA",
    ),
    "aurora": ResourceConfig(
        name          = "aurora",
        host          = "aurora.alcf.anl.gov",
        amsc_resource = "Aurora",
        amsc_fs       = "Flare",
        default_dir   = "/flare/EE-ECP/LASSI-EE/lassi_jobs",
        filesystems   = "home:flare",
        gpu_vendor    = "Intel",
    ),
}

# Worker scripts (relative to _SCRIPT_DIR/workers/)
# Each entry is (script_filename, task_arg).
# task_arg is the first positional argument passed to the one-shot worker;
# persistent_worker entries use "" because the task arrives via trigger.json.
_WORKERS: dict[str, dict[str, tuple[str, str]]] = {
    "polaris": {
        "translate":          ("polaris_worker.sh",            "translate"),
        "energy_cuda":        ("polaris_worker.sh",            "cuda_energy"),
        "energy_sycl":        ("polaris_worker.sh",            "sycl_energy"),
        "persistent_worker":  ("polaris_persistent_worker.sh", ""),
    },
    "aurora": {
        "translate":          ("aurora_worker.sh",             "translate"),
        "energy_sycl":        ("aurora_worker.sh",             "sycl_energy"),
        "persistent_worker":  ("aurora_persistent_worker.sh",  ""),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL        = "gpt55"
_DEFAULT_ACCOUNT      = "EE-ECP"
_DEFAULT_QUEUE        = "debug"
_DEFAULT_MIN_GAIN_PCT = 1.0
_MAX_ATTEMPTS         = 15
_JOB_DURATION_S           = 300  # seconds; 1 h = Polaris debug queue max
_JOB_WAIT_TIMEOUT_S       = 900
_JOB_POLL_INTERVAL_S      = 15
_PERSISTENT_START_TIMEOUT = 600   # seconds to wait for job_started
_PERSISTENT_RESULT_TIMEOUT = 1800  # seconds to wait for result_ready per attempt
_EAGLE_READ_TIMEOUT   = 60
_EAGLE_PROPAGATION_S  = 5


# ──────────────────────────────────────────────────────────────────────────────
# Source loading
# ──────────────────────────────────────────────────────────────────────────────

def _find_source_dir(app_folder: str, category: str = "") -> Path:
    """Locate an application source directory inside lassi_jobs/source_codes/.

    Supports both flat layouts (source_codes/XSBench/) and nested layouts
    (source_codes/Miniapps/XSBench/).
    """
    source_codes = _SCRIPT_DIR / "source_codes"
    suffixes = [f"{category}/{app_folder}", app_folder] if category else [app_folder]
    for suffix in suffixes:
        candidate = source_codes / suffix
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Source directory not found for app_folder={app_folder!r} "
        f"(category={category!r}).\n"
        f"Expected inside: {source_codes}\n"
        f"Copy or symlink the source_codes/ directory into {_SCRIPT_DIR}."
    )


def _find_apps_json() -> Path:
    """Locate source_apps.json inside lassi_jobs/."""
    candidate = _SCRIPT_DIR / "source_apps.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"source_apps.json not found in {_SCRIPT_DIR}.\n"
        f"Copy the file into {_SCRIPT_DIR}."
    )


def _load_cuda_source(app_category: str, app_name: str) -> tuple[str, list[str], Path, str]:
    """Load CUDA source code and parameters for the given app.

    Returns (combined_source_code, run_args, main_cu_path, verification).
    verification is the correctness-check strategy for this app (e.g. "xsbench" or "none").
    Headers (*.cuh, *.h) are prepended so the full context is visible to the LLM.
    """
    apps_json  = _find_apps_json()
    catalog    = json.loads(apps_json.read_text())

    cat_block = catalog.get(app_category)
    if cat_block is None:
        raise KeyError(f"App category {app_category!r} not found in source_apps.json")

    app_cfg = cat_block.get(app_name)
    if app_cfg is None:
        raise KeyError(f"App {app_name!r} not found in category {app_category!r}")

    app_folder   = app_cfg["app_folder"]
    app_params   = app_cfg.get("app_params", [])
    verification = app_cfg.get("verification", "none")
    src_dir      = _find_source_dir(app_folder, category=app_category)

    # Find the main CUDA file (Main.cu or first .cu file)
    main_cu = src_dir / "Main.cu"
    if not main_cu.exists():
        cu_files = sorted(src_dir.glob("*.cu"))
        if not cu_files:
            raise FileNotFoundError(f"No .cu files found in {src_dir}")
        main_cu = cu_files[0]

    code = main_cu.read_text(encoding="utf-8")

    headers: list[str] = []
    for pattern in ("*.cuh", "*.h"):
        for hdr in sorted(src_dir.glob(pattern)):
            headers.append(
                f"// ===== HEADER: {hdr.name} =====\n"
                + hdr.read_text(encoding="utf-8")
                + f"\n// ===== END HEADER: {hdr.name} =====\n"
            )
    if headers:
        code = "\n".join(headers) + "\n" + code

    return code, app_params, main_cu, verification


def _load_sycl_source(sycl_path: Path) -> str:
    """Load SYCL source from an explicit file path (for Aurora energy task)."""
    if not sycl_path.exists():
        raise FileNotFoundError(f"SYCL source file not found: {sycl_path}")
    code = sycl_path.read_text(encoding="utf-8")
    # Include any sibling headers
    src_dir = sycl_path.parent
    headers: list[str] = []
    for pattern in ("*.hpp", "*.h"):
        for hdr in sorted(src_dir.glob(pattern)):
            if hdr != sycl_path:
                headers.append(
                    f"// ===== HEADER: {hdr.name} =====\n"
                    + hdr.read_text(encoding="utf-8")
                    + f"\n// ===== END HEADER: {hdr.name} =====\n"
                )
    if headers:
        code = "\n".join(headers) + "\n" + code
    return code


# ──────────────────────────────────────────────────────────────────────────────
# ALCF connection (amsc_client)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_ipv4_patch() -> None:
    """Prefer IPv4 to avoid hangs with broken IPv6 on some ANL workstations."""
    orig = socket.getaddrinfo
    def _ipv4_first(host, port, family=0, *a, **kw):
        results = orig(host, port, family, *a, **kw)
        ipv4 = [r for r in results if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else results
    socket.getaddrinfo = _ipv4_first
    socket.setdefaulttimeout(30)


def _amsc_connect(resource_name: str, fs_name: str):
    """Connect to ALCF via Globus; return (compute_resource, fs_resource).

    A browser window opens automatically on first call for Globus login.
    Subsequent calls use cached credentials (~/.amsc/credentials.json).
    """
    _apply_ipv4_patch()

    try:
        from amsc_client import Client
    except ImportError:
        for parent in [_SCRIPT_DIR, _SCRIPT_DIR.parent]:
            candidate = parent / ".venv" / "bin" / "python"
            if not candidate.is_file():
                continue
            try:
                if candidate.samefile(sys.executable):
                    continue
            except OSError:
                pass
            check = subprocess.run(
                [str(candidate), "-c", "import amsc_client"],
                capture_output=True,
            )
            if check.returncode == 0:
                os.execv(str(candidate), [str(candidate)] + sys.argv)
        raise SystemExit(
            "amsc_client not found.\n"
            "Install with:  pip install amsc-client\n"
            "Or activate:   source .venv/bin/activate"
        )

    print("  Creating AmSC client...", flush=True)
    client   = Client(token="not-needed")
    print("  Fetching ALCF facility...", flush=True)
    alcf     = client.facility("alcf")
    print(f"  Getting {resource_name} resource...", flush=True)
    compute  = alcf.resource(resource_name)
    print(f"  Getting {fs_name} resource...", flush=True)
    try:
        fs = alcf.resource(fs_name)
    except Exception as _fs_exc:
        # Some resources (e.g. Flare/Aurora) are not registered in the IRI API.
        # Return None; callers fall back to SSH-based file operations.
        print(f"  {_B}WARNING:{_E} {fs_name} not found in IRI API ({_fs_exc}) — "
              f"filesystem operations will fall back to SSH.", flush=True)
        fs = None

    status = compute.status
    print(f"  {resource_name} status: {status}", flush=True)
    if status.lower() not in ("up", "available", "running"):
        print(f"  {_B}WARNING:{_E} {resource_name} reports '{status}' — job may queue longer.")

    return compute, fs


# ──────────────────────────────────────────────────────────────────────────────
# SSH ControlMaster lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def _ssh_master_start(host: str, alcf_username: str) -> str:
    """Open a persistent SSH ControlMaster connection to the target host."""
    import tempfile
    ctl_dir  = Path(tempfile.mkdtemp(prefix="lassi_ssh_"))
    ctl_path = str(ctl_dir / "ctl")
    cmd = [
        "ssh",
        "-o", f"ControlPath={ctl_path}",
        "-o", "ControlMaster=yes",
        "-o", "ControlPersist=yes",
        "-N",
        "-f",
        f"{alcf_username}@{host}",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"SSH ControlMaster to {host} failed (rc={result.returncode})")
    return ctl_path


def _ssh_master_stop(ctl_path: str, host: str, alcf_username: str) -> None:
    subprocess.run(
        ["ssh", "-o", f"ControlPath={ctl_path}", "-O", "exit",
         f"{alcf_username}@{host}"],
        capture_output=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# File sync helper
# ──────────────────────────────────────────────────────────────────────────────

def _sync_file_to_remote(
    session: HPCSession,
    local_file: Path,
    remote_dir: str,
    exp_name: str,
    filename: str,
) -> bool:
    """rsync a single generated source file to its experiment directory on the remote FS."""
    remote = f"{remote_dir}/generated_codes/experiment/{exp_name}/{filename}"
    return session.rsync(str(local_file), remote)


# ──────────────────────────────────────────────────────────────────────────────
# XSBench checksum validation
# ──────────────────────────────────────────────────────────────────────────────

def _extract_xsbench_checksum(stdout: str) -> str:
    for line in (stdout or "").splitlines():
        if "Verification checksum:" in line:
            return line.strip()
    return ""


def _xsbench_checksum_valid(stdout: str) -> bool:
    line = _extract_xsbench_checksum(stdout)
    return bool(line) and "(Valid)" in line


def _tail5(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(f"    {ln}" for ln in lines[-5:])


# ──────────────────────────────────────────────────────────────────────────────
# PBS job submission and monitoring
# ──────────────────────────────────────────────────────────────────────────────

class _SshPbsJob:
    """Minimal job-like wrapper around a PBS job ID obtained via SSH qsub.

    Provides the same .id / .state / .is_terminal / .status interface used by
    amsc_client job objects so the rest of the code needs no conditional logic.
    """

    def __init__(self, job_id: str, ssh_ctl: str, host: str, user: str) -> None:
        self._id      = job_id
        self._ssh_ctl = ssh_ctl
        self._host    = host
        self._user    = user

    @property
    def id(self) -> str:
        return self._id

    def _qstat_state(self) -> tuple[str, int]:
        """Return (pbs_state_char, ssh_returncode).

        PBS Pro: Q=queued, R=running, E=exiting, F=finished, X=expired.
        If qstat exits non-zero the job has been purged → treat as finished.
        """
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={self._ssh_ctl}",
             f"{self._user}@{self._host}",
             f"qstat -f {shlex.quote(self._id)} 2>/dev/null"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return "F", r.returncode
        for line in r.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("job_state"):
                return stripped.split("=", 1)[1].strip(), 0
        return "R", 0  # running (state line not yet visible)

    @property
    def state(self) -> str:
        pbs_state, _ = self._qstat_state()
        mapping = {
            "Q": "queued", "H": "queued",
            "R": "running", "E": "exiting",
            "F": "completed", "X": "completed",
        }
        return mapping.get(pbs_state, pbs_state.lower())

    @property
    def is_terminal(self) -> bool:
        pbs_state, rc = self._qstat_state()
        return rc != 0 or pbs_state in ("F", "X")

    @property
    def status(self):
        """Compatibility shim — amsc_client code accesses .status before .is_terminal."""
        return self.state


def _ssh_pbs_submit(
    *,
    ssh_ctl: str,
    host: str,
    user: str,
    job_name: str,
    queue: str,
    account: str,
    walltime_s: int,
    nodes: int = 1,
    filesystems: str = "",
    remote_dir: str,
    stdout_path: str,
    stderr_path: str,
    cmd: str,
) -> _SshPbsJob:
    """Submit a PBS job via SSH qsub and return an _SshPbsJob object.

    Used as a fallback when amsc_client's compute_resource.submit() is
    unavailable (e.g. Aurora's IRI API does not expose job submission).
    """
    h = walltime_s // 3600
    m = (walltime_s % 3600) // 60
    s = walltime_s % 60
    walltime_str = f"{h:02d}:{m:02d}:{s:02d}"

    fs_line = f"#PBS -l filesystems={filesystems}" if filesystems else ""
    script = (
        "#!/bin/bash\n"
        f"#PBS -N {job_name}\n"
        f"#PBS -q {queue}\n"
        f"#PBS -A {account}\n"
        f"#PBS -l select={nodes}\n"
        f"#PBS -l walltime={walltime_str}\n"
        f"{fs_line}\n"
        f"#PBS -o {stdout_path}\n"
        f"#PBS -e {stderr_path}\n"
        f"cd {shlex.quote(remote_dir)}\n"
        f"{cmd}\n"
    )

    r = subprocess.run(
        ["ssh", "-o", f"ControlPath={ssh_ctl}", f"{user}@{host}", "qsub"],
        input=script,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"SSH qsub failed (rc={r.returncode}): {r.stderr.strip()}"
        )
    job_id = r.stdout.strip()
    if not job_id:
        raise RuntimeError("SSH qsub returned empty job ID")
    return _SshPbsJob(job_id, ssh_ctl, host, user)


def _submit_job(
    *,
    compute_resource,
    remote_dir: str,
    worker_script: str,
    task_arg: str = "",
    src_remote_path: str,
    results_remote_dir: str,
    run_args: list[str],
    job_name: str,
    account: str,
    queue: str,
    filesystems: str,
    walltime_s: int = _JOB_DURATION_S,
    compiler_env: dict[str, str] | None = None,
    ssh_ctl: str = "",
    host: str = "",
    user: str = "",
):
    """Submit a PBS job that runs the given worker script on a compute node.

    compiler_env: optional dict of environment variable overrides forwarded as
                  VAR=val prefixes on the command line (e.g., SYCL_FLAGS, CUDA_FLAGS).

    Falls back to SSH qsub when compute_resource.submit() is unavailable
    (e.g. Aurora's IRI API does not expose job submission).
    """
    run_args_quoted = " ".join(shlex.quote(a) for a in run_args)
    env_parts: list[str] = []
    for var, val in (compiler_env or {}).items():
        if val:
            env_parts.append(f"{var}={shlex.quote(val)}")
    env_prefix = (" ".join(env_parts) + " ") if env_parts else ""

    cmd = (
        f"{env_prefix}bash {shlex.quote(worker_script)}"
        + (f" {shlex.quote(task_arg)}" if task_arg else "")
        + f" {shlex.quote(src_remote_path)}"
        + f" {shlex.quote(results_remote_dir)}"
        + f" {run_args_quoted}"
    )

    try:
        return compute_resource.submit(
            executable  = "/bin/bash",
            arguments   = ["-c", cmd],
            directory   = remote_dir,
            name        = job_name,
            queue       = queue,
            account     = account,
            duration    = walltime_s,
            nodes       = 1,
            stdout_path = f"{results_remote_dir}/pbs_stdout.log",
            stderr_path = f"{results_remote_dir}/pbs_stderr.log",
            filesystems = filesystems,
        )
    except Exception as amsc_exc:
        if not (ssh_ctl and host and user):
            raise
        print(f"  [amsc fallback → SSH qsub: {amsc_exc}]", flush=True)
        return _ssh_pbs_submit(
            ssh_ctl      = ssh_ctl,
            host         = host,
            user         = user,
            job_name     = job_name,
            queue        = queue,
            account      = account,
            walltime_s   = walltime_s,
            filesystems  = filesystems,
            remote_dir   = remote_dir,
            stdout_path  = f"{results_remote_dir}/pbs_stdout.log",
            stderr_path  = f"{results_remote_dir}/pbs_stderr.log",
            cmd          = cmd,
        )


def _wait_for_job(job, timeout_s: int = _JOB_WAIT_TIMEOUT_S) -> str:
    """Poll job status until terminal state or timeout."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            _ = job.status
            if job.is_terminal:
                return job.state
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "400" in msg:
                return "completed"
        time.sleep(_JOB_POLL_INTERVAL_S)
    try:
        return job.state
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Persistent worker
# ──────────────────────────────────────────────────────────────────────────────

class PersistentWorker:
    """Manages a long-running PBS job that receives work via IPC files.

    IPC protocol (all files in ipc_dir on the remote FS):
      job_started       written by worker at startup (content = hostname)
      trigger.json      written by us to request work
      result_ready.json written by worker when work is done
      exit              written by us to shut the worker down
    """

    def __init__(self, job, ipc_dir: str, session: HPCSession) -> None:
        self._job     = job
        self._ipc_dir = ipc_dir
        self._session = session

    @classmethod
    def launch(
        cls,
        *,
        session: HPCSession,
        compute_resource,
        remote_dir: str,
        ipc_dir: str,
        account: str,
        queue: str,
        filesystems: str,
        walltime_s: int,
        resource_name: str,
        start_timeout_s: int = _PERSISTENT_START_TIMEOUT,
    ) -> "PersistentWorker":
        """Submit the persistent worker PBS job and wait for it to signal readiness."""
        worker_script = f"{remote_dir}/workers/{_WORKERS[resource_name]['persistent_worker'][0]}"
        job_name = f"lassi-persistent-{int(time.time())}"
        cmd = f"bash {shlex.quote(worker_script)} {shlex.quote(ipc_dir)}"

        print(f"  Submitting persistent worker PBS job ({job_name})...", end="  ", flush=True)
        try:
            job = compute_resource.submit(
                executable  = "/bin/bash",
                arguments   = ["-c", cmd],
                directory   = remote_dir,
                name        = job_name,
                queue       = queue,
                account     = account,
                duration    = walltime_s,
                nodes       = 1,
                stdout_path = f"{ipc_dir}/pbs_stdout.log",
                stderr_path = f"{ipc_dir}/pbs_stderr.log",
                filesystems = filesystems,
            )
        except Exception as amsc_exc:
            if not session.ssh_ctl:
                raise
            print(f"\n  [amsc fallback → SSH qsub: {amsc_exc}]", flush=True)
            print(f"  Submitting via SSH qsub...", end="  ", flush=True)
            job = _ssh_pbs_submit(
                ssh_ctl     = session.ssh_ctl,
                host        = session.host,
                user        = session.user,
                job_name    = job_name,
                queue       = queue,
                account     = account,
                walltime_s  = walltime_s,
                filesystems = filesystems,
                remote_dir  = remote_dir,
                stdout_path = f"{ipc_dir}/pbs_stdout.log",
                stderr_path = f"{ipc_dir}/pbs_stderr.log",
                cmd         = cmd,
            )
        print(f"{_G}submitted{_E}  (PBS ID: {job.id})")

        started_path = f"{ipc_dir}/job_started"
        deadline     = time.time() + start_timeout_s
        print(f"  Waiting for persistent worker to start (up to {start_timeout_s}s)...",
              end="  ", flush=True)
        while time.time() < deadline:
            if session.file_exists(started_path):
                break
            try:
                if job.is_terminal:
                    raise RuntimeError(
                        f"Persistent worker job {job.id} reached terminal state before "
                        f"writing job_started. Check {ipc_dir}/pbs_stderr.log for errors."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(10)
        else:
            raise TimeoutError(
                f"Persistent worker failed to write job_started within {start_timeout_s}s. "
                f"Check {ipc_dir}/pbs_stderr.log for errors."
            )

        hostname = session.read_file(started_path).strip()
        print(f"{_G}ready{_E}  (compute node: {hostname})")
        return cls(job, ipc_dir, session)

    def signal_work(self, trigger: dict) -> None:
        """Deliver a work trigger to the persistent worker atomically via rsync + mv."""
        import tempfile

        session = self._session
        ipc_dir = self._ipc_dir

        session.run(f"rm -f {shlex.quote(ipc_dir + '/result_ready.json')}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="lassi_trigger_"
        ) as tf:
            json.dump(trigger, tf)
            tmp_path = tf.name

        try:
            remote_tmp = f"{ipc_dir}/trigger.json.tmp"
            if not session.rsync(tmp_path, remote_tmp):
                raise RuntimeError("rsync of trigger.json.tmp to remote failed")
            rc = session.run(
                f"mv {shlex.quote(remote_tmp)} {shlex.quote(ipc_dir + '/trigger.json')}"
            )
            if rc != 0:
                raise RuntimeError("Remote mv trigger.json.tmp → trigger.json failed")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def wait_for_result(
        self,
        timeout_s: int = _PERSISTENT_RESULT_TIMEOUT,
        poll_s: float = 3.0,
    ) -> dict:
        """Poll for result_ready.json; return its parsed content."""
        session     = self._session
        ipc_dir     = self._ipc_dir
        result_path = f"{ipc_dir}/result_ready.json"
        deadline    = time.time() + timeout_s
        last_job_check = time.time()

        while time.time() < deadline:
            if session.file_exists(result_path):
                raw = session.read_file(result_path)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    time.sleep(0.5)
                    raw = session.read_file(result_path)
                    return json.loads(raw)

            if time.time() - last_job_check > 30:
                try:
                    if self._job.is_terminal:
                        raise RuntimeError(
                            f"Persistent worker job {self._job.id} died unexpectedly "
                            f"while waiting for result_ready. "
                            f"Check {ipc_dir}/pbs_stderr.log for errors."
                        )
                except RuntimeError:
                    raise
                except Exception:
                    pass
                last_job_check = time.time()

            time.sleep(poll_s)

        raise TimeoutError(
            f"Timed out after {timeout_s}s waiting for result_ready.json in {ipc_dir}. "
            f"Check {ipc_dir}/worker.log for diagnostics."
        )

    def cleanup_result(self) -> None:
        """Remove result_ready.json so the worker is ready for the next trigger."""
        self._session.run(
            f"rm -f {shlex.quote(self._ipc_dir + '/result_ready.json')}"
        )

    def shutdown(self) -> None:
        """Send exit signal to the persistent worker and wait for the job to finish."""
        import tempfile

        session = self._session
        ipc_dir = self._ipc_dir

        print(f"\n  Sending exit signal to persistent worker...", end="  ", flush=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".exit", delete=False, prefix="lassi_exit_"
        ) as tf:
            tf.write("")
            tmp_path = tf.name

        try:
            if not session.rsync(tmp_path, f"{ipc_dir}/exit"):
                print(f"{_R}rsync of exit signal failed (job may linger){_E}")
                return
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if self._job.is_terminal:
                    print(f"{_G}done{_E}  (state: {self._job.state})")
                    return
            except Exception:
                pass
            time.sleep(5)

        print(f"{_R}timeout — job may still be running{_E}")


# ──────────────────────────────────────────────────────────────────────────────
# Job runner (per-attempt PBS job or persistent worker dispatch)
# ──────────────────────────────────────────────────────────────────────────────

class JobRunner:
    """Executes one compile+run task via either a PersistentWorker or a fresh PBS job.

    Construct once per pipeline run; call run() for each attempt.
    """

    def __init__(
        self,
        *,
        session: HPCSession,
        compute_resource,
        remote_dir: str,
        account: str,
        queue: str,
        filesystems: str,
        worker_script: str,
        worker_task_arg: str,
        compiler_env: dict[str, str],
        walltime_s: int = _JOB_DURATION_S,
        persistent: PersistentWorker | None = None,
    ) -> None:
        self._session       = session
        self._compute       = compute_resource
        self._remote_dir    = remote_dir
        self._account       = account
        self._queue         = queue
        self._filesystems   = filesystems
        self._worker_script = worker_script
        self._task_arg      = worker_task_arg
        self._compiler_env  = compiler_env
        self._walltime_s    = walltime_s
        self._persistent    = persistent

    def run(
        self,
        *,
        src_remote: str,
        results_dir: str,
        run_args: list[str],
        job_name: str,
        verbose: bool = False,
    ) -> None:
        """Execute one compile+run task. Raises on failure.

        RuntimeError — fatal (persistent worker died); caller should break loop.
        Exception    — retriable; caller should continue loop.
        """
        if self._persistent is not None:
            self._run_persistent(src_remote, results_dir, run_args, job_name)
        else:
            self._run_per_job(src_remote, results_dir, run_args, job_name, verbose)

    def _run_persistent(
        self, src_remote: str, results_dir: str, run_args: list[str], job_name: str
    ) -> None:
        worker  = self._persistent
        trigger = {
            "task":         self._task_arg,
            "src":          src_remote,
            "outdir":       results_dir,
            "run_args":     run_args,
            "compiler_env": self._compiler_env,
        }
        print(f"  Signalling work to persistent worker ({self._task_arg}, {job_name})...",
              end="  ", flush=True)
        try:
            worker.signal_work(trigger)
        except Exception as exc:
            print(f"{_R}FAILED: {exc}{_E}")
            raise Exception(f"Persistent worker signal error: {exc}") from exc
        print(f"{_G}OK{_E}")

        print(f"  Waiting for result ({job_name})...", end="  ", flush=True)
        try:
            worker.wait_for_result()
        except RuntimeError as exc:
            print(f"{_R}WORKER DIED: {exc}{_E}")
            raise RuntimeError(f"Persistent worker died: {exc}") from exc
        except TimeoutError as exc:
            print(f"{_R}TIMEOUT: {exc}{_E}")
            raise Exception(f"Timed out waiting for worker result: {exc}") from exc
        worker.cleanup_result()
        print(f"{_G}done{_E}")

    def _run_per_job(
        self, src_remote: str, results_dir: str, run_args: list[str],
        job_name: str, verbose: bool
    ) -> None:
        session = self._session
        print(f"  Submitting PBS job ({job_name})...", end="  ", flush=True)
        try:
            job = _submit_job(
                compute_resource   = self._compute,
                remote_dir         = self._remote_dir,
                worker_script      = self._worker_script,
                task_arg           = self._task_arg,
                src_remote_path    = src_remote,
                results_remote_dir = results_dir,
                run_args           = run_args,
                job_name           = job_name,
                account            = self._account,
                queue              = self._queue,
                filesystems        = self._filesystems,
                walltime_s         = self._walltime_s,
                compiler_env       = self._compiler_env,
                ssh_ctl            = session.ssh_ctl,
                host               = session.host,
                user               = session.user,
            )
        except Exception as exc:
            print(f"{_R}FAILED: {exc}{_E}")
            raise Exception(f"PBS submission error: {exc}") from exc

        pbs_id = job.id.split(".")[0]
        print(f"{_G}submitted{_E}  (PBS ID: {job.id})")
        if verbose:
            print(f"    Monitor: ssh {session.host} qstat {pbs_id}")

        print(f"  Waiting for PBS job {pbs_id}...", end="  ", flush=True)
        final_state = _wait_for_job(job)
        state_color = _G if final_state == "completed" else _R
        print(f"{state_color}{final_state}{_E}")

        if final_state not in ("completed", "active"):
            time.sleep(_EAGLE_PROPAGATION_S)
            raise Exception(f"PBS job ended in unexpected state: {final_state!r}")

        time.sleep(_EAGLE_PROPAGATION_S)



def _parse_attempt_results(
    session: HPCSession,
    results_remote_dir: str,
    has_energy: bool = False,
) -> dict:
    """Read compile/run (and optionally energy) result files from the remote FS."""
    def _rfile(fname: str, max_lines: int = 10000) -> str:
        return session.read_remote_file(f"{results_remote_dir}/{fname}", max_lines)

    def _rc(fname: str) -> int:
        raw = _rfile(fname, max_lines=5).strip()
        return int(raw) if re.fullmatch(r"-?\d+", raw) else -1

    result = {
        "build_ok":     _rc("build_rc.txt") == 0,
        "run_ok":       _rc("run_rc.txt")   == 0,
        "build_stdout": _rfile("build_stdout.log"),
        "build_stderr": _rfile("build_stderr.log"),
        "run_stdout":   _rfile("run_stdout.log"),
        "run_stderr":   _rfile("run_stderr.log"),
    }

    if has_energy:
        energy_raw = _rfile("energy_results.txt")
        energy: dict = {}
        try:
            energy = json.loads(energy_raw)
        except Exception:
            energy = {}
        result.update({
            "energy_joules": energy.get("energy_joules", -1.0),
            "avg_power_w":   energy.get("avg_power_w",   -1.0),
            "duration_s":    energy.get("duration_s",    -1.0),
            "num_samples":   energy.get("num_samples",    0),
        })

    return result


def _save_attempt_logs(results: dict, experiment_dir: Path, prefix: str) -> None:
    """Mirror remote result files to the local experiment directory."""
    for key, fname in [
        ("build_stdout", "build_stdout.log"),
        ("build_stderr", "build_stderr.log"),
        ("run_stdout",   "run_stdout.log"),
        ("run_stderr",   "run_stderr.log"),
    ]:
        content = results.get(key) or ""
        if content and not content.startswith("[Remote FS read error"):
            (experiment_dir / (prefix + fname)).write_text(content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Self-prompting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_selfprompt(
    *,
    model: str,
    prompts: PromptDictionary,
    input_code: str,
    task: str,
    resource: str,
    verifyprompt: bool,
    use_self_prompting: bool,
) -> tuple[str, str]:
    """Stage 0: run self-prompting LLM calls to build enriched context.

    Returns (guide_summary, source_description). Either may be empty if
    self-prompting is disabled or the context document is unavailable.

    Selects prompts based on task and resource:
      translate         → SYCL guide summary + CUDA source description for translation
      energy/polaris    → CUDA energy guide summary + CUDA source description
      energy/aurora     → SYCL energy guide summary + SYCL source description
    """
    if not use_self_prompting:
        return "", ""

    print(f"\n{_B}----- STAGE 0: SELF-PROMPTING  (local Argo) -----{_E}\n")

    # Select the appropriate guide key and prompts based on task+resource.
    if task == "translate":
        guide_key     = "SYCL_guide"
        summarize_key = "summarize_context_sycl"
        describe_key  = "describe_source_translate"
        guide_label   = "SYCL reference guide"
    elif task == "energy" and resource == "polaris":
        guide_key     = "CUDA_energy_guide"
        summarize_key = "summarize_context_cuda_energy"
        describe_key  = "describe_source_cuda_energy"
        guide_label   = "CUDA energy efficiency guide"
    else:
        # energy on aurora, or "sycl_energy" sub-task of "both"
        guide_key     = "SYCL_energy_guide"
        summarize_key = "summarize_context_sycl_energy"
        describe_key  = "describe_source_sycl_energy"
        guide_label   = "SYCL energy efficiency guide"

    system_prompt = prompts.get_system_prompt("selfprompt")
    guide_summary = ""

    # Step 1: summarize context guide (may not exist — skip gracefully)
    try:
        guide_raw = prompts.get_codeknowledge(guide_key)
    except (KeyError, FileNotFoundError) as exc:
        print(f"  {_B}WARN:{_E} Could not load {guide_label}: {exc}.  Skipping summary.")
        guide_raw = ""

    if guide_raw:
        print(f"  Step 1/2: Summarizing {guide_label}...")
        summarize_prompt = (
            prompts.get_selfprompt(summarize_key)
            + f"\n\n[BEGIN GUIDE]\n{guide_raw}\n[END GUIDE]"
        )
        guide_summary = _llm.generate_text(
            model,
            system_prompt,
            summarize_prompt,
            intent=f"{guide_label} summarization",
            verifyprompt=verifyprompt,
            label=f"[LLM] Summarizing {guide_label}",
        )
    else:
        print(f"  Step 1/2: Skipping guide summarization (no guide file available).")

    # Step 2: describe source code
    print("  Step 2/2: Analyzing source code...")
    describe_prompt = (
        prompts.get_selfprompt(describe_key)
        + "\n\n[BEGIN SOURCE CODE]\n"
        + input_code
        + "\n[END SOURCE CODE]"
    )
    source_description = _llm.generate_text(
        model,
        system_prompt,
        describe_prompt,
        intent="source code analysis",
        verifyprompt=verifyprompt,
        label="[LLM] Analyzing source code",
    )

    print(f"\n  {_B}{_G}✓{_E}  Self-prompting complete.\n")
    return guide_summary, source_description


def _build_stage1_prompt(
    *,
    prompts: PromptDictionary,
    task: str,
    resource: str,
    input_code: str,
    guide_summary: str,
    source_description: str,
    baseline_energy_j: float = -1.0,
    baseline_power_w: float = -1.0,
) -> tuple[str, str]:
    """Construct the system prompt and user prompt for Stage 1 (initial generation).

    Returns (system_prompt, user_prompt).
    """
    if task == "translate":
        sys_key   = "CUDA_to_SYCL"
        usr_key   = "CUDA_to_SYCL"
        block_tag = "CODE TO REFACTOR"
    elif task == "energy" and resource == "polaris":
        sys_key   = "CUDA_optimize_energy"
        usr_key   = "CUDA_optimize_energy"
        block_tag = "CUDA CODE TO OPTIMIZE"
    else:
        sys_key   = "SYCL_optimize_energy"
        usr_key   = "SYCL_optimize_energy"
        block_tag = "SYCL CODE TO OPTIMIZE"

    system_prompt   = prompts.get_system_prompt(sys_key)
    translate_user  = prompts.get_codetranslate_prompt(usr_key)

    # Build preamble from self-prompting results
    preamble = ""
    if guide_summary:
        preamble += (
            "[CONTEXT KNOWLEDGE]\n"
            + guide_summary
            + "\n[/CONTEXT KNOWLEDGE]\n\n"
        )
    if source_description:
        preamble += (
            "[YOUR ANALYSIS OF THE SOURCE CODE]\n"
            + source_description
            + "\n[/YOUR ANALYSIS OF THE SOURCE CODE]\n\n"
        )

    think_prefix = (
        "Think carefully, using your analysis above.  Now, "
        if source_description else ""
    )

    user_prompt = (
        preamble
        + think_prefix
        + translate_user
        + f"\n\n[START OF {block_tag}]\n"
        + input_code
        + f"\n[/END OF {block_tag}]"
    )

    # Append baseline energy reference for energy tasks, matching base LASSI-EE behaviour.
    if task in ("energy", "sycl_energy") and baseline_energy_j > 0:
        user_prompt += (
            f"\n\nFor reference, the {block_tag} provided above was executed on this system "
            f"and consumed a total energy of {baseline_energy_j:.2f} Joules at an average power "
            f"of {baseline_power_w:.2f} Watts, which includes the idle system energy consumption "
            f"and average power.\n"
        )

    return system_prompt, user_prompt


def _build_correction_prompt(
    *,
    prompts: PromptDictionary,
    task: str,
    resource: str,
    code_block: str,
    compile_err: str,
    run_err: str,
    run_stdout: str,
    input_code: str,
    attempt: int,
    guide_summary: str = "",
    source_description: str = "",
    best_code_block: str = "",
    opt_power_w: float = -1.0,
    opt_dur_s: float = -1.0,
) -> tuple[str, str]:
    """Construct the correction user prompt for Stage 2.

    Returns (err_type_label, user_prompt).
    """
    if task == "translate":
        base_key = "CUDA_to_SYCL_error"
    elif task == "energy" and resource == "polaris":
        base_key = "CUDA_optimize_energy_error"
    else:
        base_key = "SYCL_optimize_energy_error"

    improve_key = None
    if task == "energy":
        if resource == "polaris":
            improve_key = "CUDA_optimize_energy_improve"
        else:
            improve_key = "SYCL_optimize_energy_improve"

    if compile_err == "execute_error":
        err_type    = "runtime error"
        error_msg   = run_err
        if run_stdout.strip():
            error_msg += "\n\nProgram stdout before crash:\n" + run_stdout
        error_msg  += "\n\n[ORIGINAL SOURCE FOR REFERENCE]\n" + input_code + "\n[/ORIGINAL SOURCE]\n"
        error_intro = "\n -- Runtime error: "
        prompt_key  = base_key
    elif compile_err.startswith("VERIFICATION MISMATCH"):
        err_type    = "checksum mismatch"
        error_msg   = compile_err
        error_intro = "\n -- Verification mismatch: "
        prompt_key  = base_key
    elif compile_err.startswith("ENERGY_KEEP_IMPROVING") and improve_key:
        err_type    = "energy keep improving"
        error_msg   = compile_err
        if opt_power_w > 0 and opt_dur_s > 0:
            error_msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        error_intro = "\n -- Energy result: "
        prompt_key  = improve_key
        if best_code_block:
            code_block = best_code_block
    elif compile_err.startswith("ENERGY_NOT_IMPROVED") and improve_key:
        err_type    = "energy improvement insufficient"
        error_msg   = compile_err
        # Append power and duration if available so the model can reason about
        # whether execution time or power draw is the dominant cost.
        if opt_power_w > 0 and opt_dur_s > 0:
            error_msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        error_intro = "\n -- Energy result: "
        prompt_key  = improve_key
        # Re-use the best-performing code seen so far as the starting point,
        # matching base LASSI-EE behaviour (code_blocks[tracking_best_version+1]).
        if best_code_block:
            code_block = best_code_block
    else:
        err_type    = "compile error"
        error_msg   = compile_err
        error_intro = "\n -- Compile error: "
        prompt_key  = base_key

    correction_base = prompts.get_code_error_correction_prompt(prompt_key)

    # For energy corrections, re-inject Stage 0 context so the model retains
    # the knowledge summary and source analysis, matching base LASSI-EE behaviour.
    if err_type in ("energy improvement insufficient", "energy keep improving"):
        preamble = ""
        if guide_summary:
            preamble += (
                "[CONTEXT KNOWLEDGE]\n"
                + guide_summary
                + "\n[/CONTEXT KNOWLEDGE]\n\n"
            )
        if source_description:
            preamble += (
                "[YOUR ANALYSIS OF THE SOURCE CODE]\n"
                + source_description
                + "\n[/YOUR ANALYSIS OF THE SOURCE CODE]\n\n"
            )
        correction_base = preamble + correction_base

    user_prompt = (
        correction_base
        + "\n\n[START OF CODE TO FIX]\n"
        + code_block
        + "\n[/END OF CODE TO FIX]"
        + error_intro + "\n"
        + error_msg
    )

    return err_type, user_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Baseline measurement (energy task only)
# ──────────────────────────────────────────────────────────────────────────────

def _run_baseline(
    *,
    session: HPCSession,
    runner: JobRunner,
    remote_dir: str,
    original_src_path: Path,
    src_extension: str,
    experiment_name: str,
    experiment_dir: Path,
    run_args: list[str],
    run_id: str,
    verification: str = "none",
) -> tuple[float, float, str]:
    """Compile and run the original source to establish a baseline energy.

    Returns (energy_joules, avg_power_w, run_stdout).
    Energy and power are -1.0 on failure; stdout is "" on failure.
    """
    print(f"\n{_B}----- BASELINE: compiling + running original source  (PBS) -----{_E}\n")

    baseline_name = f"Main_baseline{src_extension}"
    exp_remote    = f"{remote_dir}/generated_codes/experiment/{experiment_name}"

    print(f"  Uploading {baseline_name} to remote FS...", end="  ", flush=True)
    if not _sync_file_to_remote(session, original_src_path, remote_dir, experiment_name, baseline_name):
        print(f"{_R}FAILED{_E}")
        return -1.0, -1.0, ""
    print(f"{_G}OK{_E}")

    # Sync companion headers so the compiler can resolve #include directives.
    src_dir = original_src_path.parent
    for pattern in ("*.cuh", "*.h", "*.hpp"):
        for hdr in sorted(src_dir.glob(pattern)):
            session.rsync(str(hdr), f"{exp_remote}/{hdr.name}")

    src_remote     = f"{exp_remote}/{baseline_name}"
    results_remote = f"{exp_remote}/baseline_results"
    session.mkdir(results_remote)

    try:
        runner.run(
            src_remote  = src_remote,
            results_dir = results_remote,
            run_args    = run_args,
            job_name    = f"lassi-baseline-{run_id}",
        )
    except Exception as exc:
        print(f"  {_R}Baseline job failed: {exc}{_E}")
        return -1.0, -1.0, ""

    results = _parse_attempt_results(session, results_remote, has_energy=True)
    _save_attempt_logs(results, experiment_dir, "baseline_")

    if not results["build_ok"]:
        print(f"  {_R}Baseline compile FAILED — cannot establish energy reference.{_E}")
        return -1.0, -1.0, ""
    if not results["run_ok"]:
        print(f"  {_R}Baseline run FAILED — cannot establish energy reference.{_E}")
        return -1.0, -1.0, ""

    run_stdout_str = results.get("run_stdout") or ""

    if verification == "xsbench" and not _xsbench_checksum_valid(run_stdout_str):
        cs = _extract_xsbench_checksum(run_stdout_str)
        print(f"  {_R}Baseline checksum invalid ({cs!r}) — energy reference may be unreliable.{_E}")

    ej  = results.get("energy_joules", -1.0)
    apw = results.get("avg_power_w",   -1.0)
    ds  = results.get("duration_s",    -1.0)
    ns  = results.get("num_samples",    0)

    if ej > 0:
        print(f"  {_B}{_G}Baseline energy : {ej:.1f} J  ({apw:.1f} W avg,  {ds:.1f} s,  {ns} samples){_E}")
    else:
        print(f"  {_R}Warning: energy measurement unavailable (power monitor not found?).{_E}")
        print(f"  Pipeline will proceed verifying correctness only.")

    baseline_json = {"energy_joules": ej, "avg_power_w": apw, "duration_s": ds, "num_samples": ns}
    (experiment_dir / "baseline_energy.json").write_text(
        json.dumps(baseline_json, indent=2), encoding="utf-8"
    )
    return ej, apw, run_stdout_str


# ──────────────────────────────────────────────────────────────────────────────
# LLM judge (mirrors base LASSI-EE llm_judge)
# ──────────────────────────────────────────────────────────────────────────────

def _run_judge(
    *,
    model: str,
    system_prompt: str,
    source_code: str,
    source_stdout: str,
    source_metrics: str,
    refactored_code: str,
    refactored_stdout: str,
    refactored_metrics: str,
) -> str:
    """Ask the LLM whether the refactored code is functionally equivalent.

    Returns "VALID", "INVALID", or "NONE".
    """
    print(f"\n  {_B}[Judge] Verifying functional equivalence via LLM...{_E}")

    content_prompt = (
        "Please compare the difference between the following two CUDA C++ codes.\n\n"
        "First, I will provide you with the SOURCE CODE followed by its STANDARD OUTPUT "
        "and METRICS measured during code runtime.\n"
        "Next, I will provide you with a REFACTORED CODE followed by its STANDARD OUTPUT "
        "and METRICS measured during code runtime.\n\n"
        "After you have reviewed both codes and their output, the REFACTORED CODE is expected "
        "to produce equivalent functionality and reasonably similar standard output in its "
        "layout and content. However, the REFACTORED CODE may not meet these expectations.\n\n"
        "Your task is to strictly evaluate whether the REFACTORED CODE (1) preserves all "
        "intended functionality of the SOURCE CODE and (2) produces printed nearly equivalent "
        "standard output, including its displayed layout and content, compared to the SOURCE CODE.\n\n"
        "Your careful evaluation should focus on the functional equivalence of the two codes "
        "and nearly exact printed standard output of the two codes. Minor differences in "
        "numerical values are acceptable in the printed standard output only if they are "
        "consistent with the REFACTORED CODE improving energy efficiency and these specific "
        "values are reasonable and consistent with the SOURCE CODE's intended purpose. "
        "Formatting differences are acceptable as long as key details are included that are "
        "seen in the standard output of the SOURCE CODE.\n\n"
        "Please explain your reasoning, and then conclude your response with VALID if the "
        "REFACTORED CODE is functionally equivalent, or INVALID if it is not."
    )

    content_prompt += f"\n\n# [SOURCE CODE]\n{source_code}"
    content_prompt += f"\n\n## [STANDARD OUTPUT]\n{source_stdout}"
    content_prompt += f"\n\n## [METRICS]\n{source_metrics}"
    content_prompt += f"\n\n# [REFACTORED CODE]\n{refactored_code}"
    content_prompt += f"\n\n## [STANDARD OUTPUT]\n{refactored_stdout}"
    content_prompt += f"\n\n## [METRICS]\n{refactored_metrics}"

    response = _llm.generate_text(
        model,
        system_prompt,
        content_prompt,
        intent="code validation judge",
        label="[LLM] Judge",
        temperature=0.2,
    )

    if "INVALID" in response:
        verdict = "INVALID"
    elif "VALID" in response:
        verdict = "VALID"
    else:
        verdict = "NONE"

    print(f"  {_B}[Judge] verdict: {verdict}{_E}")
    return verdict


# ──────────────────────────────────────────────────────────────────────────────
# Metadata artifact writer (mirrors base LASSI-EE pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

_CODE_COMPARISON_PROMPT = (
    "Below are two sets of parallel code written with CUDA in C++ that we want to compare their "
    "differences. The first is labeled SOURCE CODE and it is followed by its output to the stdout "
    "when it was executed as well as the total energy consumed and average power. The second is "
    "labeled REFACTORED CODE followed by its output to the stdout when executed on the same GPU "
    "and its energy measurements. Both codes are expected to perform the same function. Both codes "
    "are run on the same GPU and compute system. However, the REFACTORED CODE is a refactoring of "
    "the SOURCE CODE generated by an LLM that was prompted to improve the energy efficiency of the "
    "SOURCE CODE by reducing energy consumption and/or reducing the average power utilized by the "
    "GPU while the code is executed. Please compare the differences in the REFACTORED CODE from "
    "the SOURCE CODE by highlighting any changes in the REFACTORED CODE that might be related to "
    "improving its energy efficiency. Avoid repeating the entire code for reference. Conclude your "
    "response with a simple list of the optimization methods, if any, used in the REFACTORED CODE "
    "not included in the SOURCE CODE that improves energy efficiency. Surround this list of "
    "optimization methods with the tags [OPTIMIZATION LIST] and [/OPTIMIZATION LIST]"
)


def _write_metadata(
    *,
    experiment_dir: Path,
    experiment_name: str,
    runtime_s: float,
    attempt_count: int,
    energy_efficiency_iteration: int,
    has_energy: bool,
    baseline_stdout: str,
    best_run_stdout: str,
    baseline_energy_j: float,
    baseline_power_w: float,
    best_energy: float,
    best_power_w: float,
    best_dur_s: float,
    energy_efficiency_success: bool,
    input_code: str,
    best_code_block: str,
    model: str,
    system_prompt: str,
) -> None:
    """Write metadata_{experiment_name}.txt matching base LASSI-EE artifact format."""
    from datetime import datetime as _dt

    dt_string = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read baseline duration from the JSON written by _run_baseline
    baseline_dur_s = -1.0
    baseline_json_path = experiment_dir / "baseline_energy.json"
    if baseline_json_path.exists():
        try:
            import json as _json
            bj = _json.loads(baseline_json_path.read_text(encoding="utf-8"))
            baseline_dur_s = float(bj.get("duration_s", -1.0))
        except Exception:
            pass

    meta: list[str] = []

    meta.append("\n\n----- METRICS -----")
    meta.append(f"\nExperiment date-time: {dt_string}")
    meta.append(f"\nTime (sec): {round(runtime_s, 2)}")
    meta.append(f"\nSelf-correction iteration count: {attempt_count}")
    if has_energy:
        meta.append(f"\nEnergy efficiency improvement count: {energy_efficiency_iteration}")

    if baseline_stdout:
        meta.append("\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n")
        meta.append(baseline_stdout)

    if best_run_stdout:
        meta.append("\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n")
        meta.append(best_run_stdout)

    if has_energy and baseline_energy_j > 0:
        meta.append("\n\n\n----- ENERGY EFFICIENCY METRICS -----\n")
        meta.append("Total Energy (J):")
        meta.append(f"  Source code: {round(baseline_energy_j, 2)}")
        if best_energy > 0:
            meta.append(f"  Refactored code: {round(best_energy, 2)}")
        meta.append("\nAverage Power (W):")
        meta.append(f"  Source code: {round(baseline_power_w, 2)}")
        if best_power_w > 0:
            meta.append(f"  Refactored code: {round(best_power_w, 2)}")
        meta.append("\nRuntime (s):")
        if baseline_dur_s > 0:
            meta.append(f"  Source code: {round(baseline_dur_s, 2)}")
        if best_dur_s > 0:
            meta.append(f"  Refactored code: {round(best_dur_s, 2)}")

    if energy_efficiency_success and best_code_block and baseline_energy_j > 0 and best_energy > 0:
        print(f"\n{_B}----- BEGIN CODE COMPARISON -----{_E}\n")
        compare_prompt = _CODE_COMPARISON_PROMPT
        compare_prompt += "\n\n[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_code
        compare_prompt += "\n\nOUTPUT: \n\n" + baseline_stdout
        if baseline_dur_s > 0:
            compare_prompt += f"\n\nCode execution duration: {round(baseline_dur_s, 2)} s"
        compare_prompt += f"\nTotal Energy Consumed: {round(baseline_energy_j, 2)} J"
        compare_prompt += f"\nAverage Power: {round(baseline_power_w, 2)} W"
        compare_prompt += "\n\n[/END SOURCE CODE AND RESULTS]"
        compare_prompt += "\n\n[BEGIN REFACTORED CODE AND RESULTS]\n\n" + best_code_block
        compare_prompt += "\n\nOUTPUT: \n\n" + best_run_stdout
        if best_dur_s > 0:
            compare_prompt += f"\n\nCode execution duration: {round(best_dur_s, 2)} s"
        compare_prompt += f"\nTotal Energy Consumed: {round(best_energy, 2)} J"
        compare_prompt += f"\nAverage Power: {round(best_power_w, 2)} W"
        compare_prompt += "\n\n[/END REFACTORED CODE AND RESULTS]"

        code_comparison_response = _llm.generate_text(
            model,
            system_prompt,
            compare_prompt,
            intent="code comparison",
            label="[LLM] Code comparison",
        )
        meta.append("\n\n\n----- CODE COMPARISON -----\n")
        meta.append(code_comparison_response)

    metadata_text = "".join(meta)
    metadata_path = experiment_dir / f"metadata_{experiment_name}.txt"
    metadata_path.write_text(metadata_text, encoding="utf-8")
    print(f"\n  Metadata file: {metadata_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline (shared by translate, energy, and both)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    model: str,
    task: str,
    resource: str,
    input_code: str,
    session: HPCSession,
    runner: JobRunner,
    remote_dir: str,
    src_extension: str,
    run_args: list[str],
    experiment_name: str,
    experiment_dir: Path,
    csv_path: Path,
    experiment_data: dict,
    trial_number: int,
    run_id: str,
    use_self_prompting: bool = True,
    verifyprompt: bool = False,
    verbose: bool = False,
    baseline_energy_j: float = -1.0,
    baseline_power_w: float = -1.0,
    baseline_stdout: str = "",
    min_energy_gain_pct: float = _DEFAULT_MIN_GAIN_PCT,
    has_energy: bool = False,
    use_judge: bool = True,
    verification: str = "none",
) -> tuple[bool, str]:
    """Run the full LASSI pipeline (Stage 0 → Stage 1 → Stage 2 loop).

    Returns (success, final_code_block).
    final_code_block is the last successfully generated code on success,
    or the last attempt's code on failure (useful for chaining in "both" mode).
    """
    prompts    = PromptDictionary()
    start_time = time.time()

    # ── Stage 0: self-prompting ───────────────────────────────────────────────
    guide_summary, source_description = _run_selfprompt(
        model              = model,
        prompts            = prompts,
        input_code         = input_code,
        task               = task,
        resource           = resource,
        verifyprompt       = verifyprompt,
        use_self_prompting = use_self_prompting,
    )

    # ── Build Stage 1 prompts ─────────────────────────────────────────────────
    system_prompt, stage1_user_prompt = _build_stage1_prompt(
        prompts            = prompts,
        task               = task,
        resource           = resource,
        input_code         = input_code,
        guide_summary      = guide_summary,
        source_description = source_description,
        baseline_energy_j  = baseline_energy_j,
        baseline_power_w   = baseline_power_w,
    )

    # Ensure experiment directory exists on remote FS
    session.mkdirs(
        f"{remote_dir}/generated_codes",
        f"{remote_dir}/generated_codes/experiment",
        f"{remote_dir}/generated_codes/experiment/{experiment_name}",
    )

    # Energy-enabled flag for result parsing
    energy_check_enabled = has_energy and baseline_energy_j > 0

    # ── Stages 1 + 2 loop ────────────────────────────────────────────────────
    compile_err           = ""
    run_err               = ""
    run_stdout            = ""
    code_block            = ""
    best_energy                    = -1.0
    best_code_block                = ""     # best-performing code seen so far (base LASSI-EE alignment)
    best_run_stdout                = ""     # stdout from the best-energy run (for metadata artifact)
    best_power_w                   = -1.0   # power at best energy point (for final reporting)
    best_dur_s                     = -1.0   # duration at best energy point
    last_power_w                   = -1.0   # power from the most recent successful run
    last_dur_s                     = -1.0   # duration from the most recent successful run
    code_gen_temp                  = 0.2    # LLM temperature; increments on energy-insufficient iterations
    energy_correction_cnt          = 0      # counts energy-insufficient corrections for temp ceiling
    energy_efficiency_iteration    = 0      # counts successful energy improvements (stagnation detection)
    energy_efficiency_success      = False  # True once any energy improvement has been seen

    for attempt in range(_MAX_ATTEMPTS + 1):

        # ── Local Argo LLM call ───────────────────────────────────────────────
        if attempt == 0:
            stage_label = {
                "translate":    "CUDA→SYCL TRANSLATION",
                "energy":       "CUDA ENERGY OPTIMIZATION" if resource == "polaris" else "SYCL ENERGY OPTIMIZATION",
                "sycl_energy":  "SYCL ENERGY OPTIMIZATION",
            }.get(task, "CODE GENERATION")
            print(f"\n{_B}----- STAGE 1: {stage_label}  (local Argo) -----{_E}\n")
            code_block = _llm.generate_code(
                model,
                system_prompt,
                stage1_user_prompt,
                verifyprompt=verifyprompt,
                label="[LLM] Generating initial code",
                temperature=0.2,
            )
        else:
            if attempt == 1:
                print(f"\n{_B}----- STAGE 2: SELF-CORRECTION  (local Argo + PBS) -----{_E}\n")

            err_type, error_user_prompt = _build_correction_prompt(
                prompts            = prompts,
                task               = task,
                resource           = resource,
                code_block         = code_block,
                compile_err        = compile_err,
                run_err            = run_err,
                run_stdout         = run_stdout,
                input_code         = input_code,
                attempt            = attempt,
                guide_summary      = guide_summary,
                source_description = source_description,
                best_code_block    = best_code_block,
                opt_power_w        = last_power_w,
                opt_dur_s          = last_dur_s,
            )
            print(f"  {_B}Attempt {attempt}/{_MAX_ATTEMPTS}{_E} — correcting {err_type}")

            # Temperature strategy (mirrors base LASSI-EE):
            # - Energy-insufficient (no prior success): increment by 0.2 each time; break at ceiling.
            # - Energy keep-improving (prior success): stay at 0.2 (reset was done on improvement).
            # - Compile/runtime errors: always 0.2.
            if err_type == "energy improvement insufficient":
                energy_correction_cnt += 1
                call_temp      = code_gen_temp
                code_gen_temp  = round(min(code_gen_temp + 0.2, 1.0), 1)
                if call_temp >= 1.0:
                    print(f"  {_B}Max LLM temperature reached — stopping self-correction.{_E}")
                    break
            elif err_type == "energy keep improving":
                call_temp = code_gen_temp  # already reset to 0.2 on last improvement
            else:
                call_temp = 0.2  # match base LASSI-EE: compile/runtime corrections at 0.2

            code_block = _llm.generate_code(
                model,
                system_prompt,
                error_user_prompt,
                verifyprompt=verifyprompt,
                label=f"[LLM] Corrected code (attempt {attempt})",
                temperature=call_temp,
            )

        # ── Write generated code locally ──────────────────────────────────────
        src_name  = f"Main_attempt{attempt}{src_extension}"
        local_src = experiment_dir / src_name
        local_src.write_text(code_block, encoding="utf-8")

        # ── Upload source to remote FS ─────────────────────────────────────────
        print(f"  Uploading {src_name} to remote FS...", end="  ", flush=True)
        if not _sync_file_to_remote(session, local_src, remote_dir, experiment_name, src_name):
            print(f"{_R}FAILED{_E}")
            compile_err = f"rsync failed — could not upload {src_name}"
            run_stdout  = ""
            continue
        print(f"{_G}OK{_E}")

        src_remote_path    = f"{remote_dir}/generated_codes/experiment/{experiment_name}/{src_name}"
        results_remote_dir = f"{remote_dir}/generated_codes/experiment/{experiment_name}/attempt{attempt}_results"
        session.mkdir(results_remote_dir)

        try:
            runner.run(
                src_remote  = src_remote_path,
                results_dir = results_remote_dir,
                run_args    = run_args,
                job_name    = f"lassi-{attempt}-{run_id}",
                verbose     = verbose,
            )
        except RuntimeError as exc:
            compile_err = str(exc)
            run_stdout  = ""
            break
        except Exception as exc:
            compile_err = str(exc)
            run_stdout  = ""
            continue

        # ── Read results from remote FS ────────────────────────────────────────
        results = _parse_attempt_results(session, results_remote_dir, has_energy)
        _save_attempt_logs(results, experiment_dir, f"attempt{attempt}_")

        # ── Compile result ────────────────────────────────────────────────────
        if not results["build_ok"]:
            tail = _tail5(results["build_stderr"])
            print(f"  Compiling...  {_B}{_R}✗  compile error:{_E}")
            if tail:
                print(tail)
            compile_err = results["build_stderr"] or "Compilation failed (no stderr captured)"
            run_stdout  = ""
            continue
        print(f"  Compiling...  {_B}{_G}✓  OK{_E}")

        # ── Run result ────────────────────────────────────────────────────────
        run_stdout = results["run_stdout"] or ""
        if not results["run_ok"]:
            combined = (results["run_stderr"] or "") + "\n" + run_stdout
            tail = _tail5(combined)
            print(f"  Running...    {_B}{_R}✗  runtime error:{_E}")
            if tail:
                print(tail)
            compile_err = "execute_error"
            run_err     = results["run_stderr"] or "Exited with non-zero exit code"
            continue
        print(f"  Running...    {_B}{_G}✓  OK{_E}")

        # ── Checksum verification ─────────────────────────────────────────────
        if verification == "xsbench":
            checksum_line = _extract_xsbench_checksum(run_stdout)
            if not _xsbench_checksum_valid(run_stdout):
                print(f"  Checksum...   {_B}{_R}✗  {checksum_line or 'missing'}{_E}")
                snippet = (run_stdout[:1000].strip() or "(no output)") + (
                    "\n...(truncated)" if len(run_stdout) > 1000 else ""
                )
                detail = (
                    f"Checksum missing or invalid.\n"
                    f"Checksum line: {checksum_line!r}\n"
                    f"\nActual stdout:\n{snippet}\n"
                    f"\nYour code MUST faithfully implement ALL functions from "
                    f"the original source, including print_results(), "
                    f"run_event_based_simulation_baseline(), and grid_init_do_not_profile().\n"
                    f"Do NOT generate a simplified stub.\n"
                    f"\n[ORIGINAL SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL SOURCE]\n"
                )
                compile_err = "VERIFICATION MISMATCH: " + detail
                continue
            print(f"  Checksum...   {_B}{_G}✓  {checksum_line}{_E}")

        # ── LLM judge: verify functional equivalence ──────────────────────────
        if use_judge and baseline_stdout:
            src_metrics_str = (
                f"Energy: {baseline_energy_j:.1f} J, Avg Power: {baseline_power_w:.1f} W"
                if baseline_energy_j > 0 else "(not measured)"
            )
            opt_energy_j_pre = results.get("energy_joules", -1.0)
            opt_power_w_pre  = results.get("avg_power_w",   -1.0)
            ref_metrics_str  = (
                f"Energy: {opt_energy_j_pre:.1f} J, Avg Power: {opt_power_w_pre:.1f} W"
                if opt_energy_j_pre > 0 else "(not measured)"
            )
            verdict = _run_judge(
                model             = model,
                system_prompt     = system_prompt,
                source_code       = input_code,
                source_stdout     = baseline_stdout,
                source_metrics    = src_metrics_str,
                refactored_code   = code_block,
                refactored_stdout = run_stdout,
                refactored_metrics = ref_metrics_str,
            )
            if verdict != "VALID":
                # Base LASSI-EE requires valid_code_flag == "VALID" for an improvement to count;
                # INVALID and NONE both block the attempt and trigger re-generation.
                reason = (
                    "the refactored code does not preserve functional equivalence or standard output"
                    if verdict == "INVALID" else
                    "judge could not determine functional equivalence (NONE verdict)"
                )
                compile_err = (
                    f"VERIFICATION MISMATCH (LLM judge): {reason}. "
                    "Ensure ALL original logic, print statements, and computation are preserved."
                )
                run_stdout = ""
                continue

        # ── Energy check (energy task only) ───────────────────────────────────
        if has_energy:
            opt_energy_j = results.get("energy_joules", -1.0)
            opt_power_w  = results.get("avg_power_w",   -1.0)
            opt_dur_s    = results.get("duration_s",    -1.0)
            opt_samples  = results.get("num_samples",    0)

            if opt_energy_j > 0:
                print(
                    f"  Energy...     {_B}{opt_energy_j:.1f} J{_E}  "
                    f"({opt_power_w:.1f} W avg,  {opt_dur_s:.1f} s,  {opt_samples} samples)"
                )
                last_power_w = opt_power_w
                last_dur_s   = opt_dur_s
            else:
                print(f"  Energy...     {_B}(measurement unavailable){_E}")

            if energy_check_enabled and opt_energy_j > 0:
                reduction_pct = (baseline_energy_j - opt_energy_j) / baseline_energy_j * 100.0

                comparison_j = best_energy if best_energy > 0 else baseline_energy_j
                if opt_energy_j < comparison_j:
                    # ── Energy improved vs best seen so far (or vs baseline) ──
                    prev_best       = best_energy
                    best_energy     = opt_energy_j
                    best_code_block = code_block
                    best_run_stdout = run_stdout
                    best_power_w    = opt_power_w
                    best_dur_s      = opt_dur_s
                    energy_efficiency_success = True
                    energy_correction_cnt     = 0
                    code_gen_temp             = 0.2   # reset temp on improvement (base LASSI-EE)
                    energy_efficiency_iteration += 1

                    print(
                        f"  Reduction...  {_B}{_G}↓  {reduction_pct:.1f}%  "
                        f"({baseline_energy_j:.1f} J → {opt_energy_j:.1f} J)  "
                        f"[new best, iteration {energy_efficiency_iteration}]{_E}"
                    )

                    # ── Stagnation stop (mirrors base LASSI-EE) ───────────────
                    if prev_best > 0 and energy_efficiency_iteration > 4:
                        marginal = (prev_best - opt_energy_j) / prev_best
                        if marginal < 0.011 and opt_energy_j < baseline_energy_j:
                            print(
                                f"  {_B}Energy improvement stagnating "
                                f"({marginal*100:.2f}% marginal gain) — stopping self-correction.{_E}"
                            )
                            break

                    # ── Keep optimizing ───────────────────────────────────────
                    from_j = prev_best if prev_best > 0 else baseline_energy_j
                    _correctness_note = (
                        "CRITICAL: preserve the exact XSBench verification checksum — "
                        "do NOT alter numerical computation, loop bounds, or comparison operators."
                        if verification == "xsbench" else
                        "CRITICAL: preserve the exact behavior and output of the original source code."
                    )
                    compile_err = (
                        f"ENERGY_KEEP_IMPROVING: Code improved from "
                        f"{from_j:.1f} J to {opt_energy_j:.1f} J "
                        f"({reduction_pct:+.1f}% vs baseline of {baseline_energy_j:.1f} J). "
                        f"Apply additional aggressive energy optimizations to reduce further.\n"
                        f"{_correctness_note}"
                    )
                    continue

                else:
                    # ── No improvement vs best (or baseline if no best yet) ───
                    ref_label = f"best {best_energy:.1f} J" if best_energy > 0 else f"baseline {baseline_energy_j:.1f} J"
                    print(
                        f"  Reduction...  {_B}{_R}✗  {reduction_pct:.1f}%  "
                        f"(no improvement vs {ref_label}){_E}"
                    )
                    _correctness_note = (
                        "CRITICAL: preserve the exact XSBench verification checksum — "
                        "do NOT alter numerical computation, loop bounds, or comparison operators."
                        if verification == "xsbench" else
                        "CRITICAL: preserve the exact behavior and output of the original source code."
                    )
                    compile_err = (
                        f"ENERGY_NOT_IMPROVED: Current code uses {opt_energy_j:.1f} J "
                        f"versus baseline {baseline_energy_j:.1f} J "
                        f"({reduction_pct:+.1f}% change, need ≥{min_energy_gain_pct:.1f}% reduction).\n"
                        f"Apply more aggressive energy optimizations.\n"
                        f"{_correctness_note}"
                    )
                    continue

            elif energy_check_enabled and opt_energy_j <= 0:
                # Measurement unavailable — keep iterating; preserve last compile_err.
                if not compile_err:
                    compile_err = "ENERGY_NOT_IMPROVED: Energy measurement unavailable on this run. Retry."
                continue
            else:
                # No baseline available — treat correctness alone as success.
                pass  # fall through to the success block below

        # ── Success (translate task or energy task without baseline) ──────────
        if has_energy and energy_check_enabled:
            # Energy tasks with a baseline never reach here — they always
            # continue or break.  This guard is a safety net only.
            pass
        else:
            task_label   = "translate" if task == "translate" else "energy"
            versioned    = experiment_dir / f"Main_v1_Trial{trial_number}{src_extension}"
            versioned.write_text(code_block, encoding="utf-8")

            runtime_min    = round((time.time() - start_time) / 60, 2)
            attempts_label = f"attempt {attempt + 1}" if attempt > 0 else "first attempt"

            print(
                f"\n  {_B}{_G}✅  {task_label.capitalize()} complete{_E}"
                f"  —  {attempts_label}  |  {runtime_min} min total\n"
            )
            print(f"  Successful code : {versioned}")
            print(f"  Remote results  : {results_remote_dir}")

            experiment_data["success"]     = True
            experiment_data["runtime_min"] = runtime_min

            with csv_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writerow(experiment_data)

            _write_metadata(
                experiment_dir            = experiment_dir,
                experiment_name           = experiment_name,
                runtime_s                 = time.time() - start_time,
                attempt_count             = attempt,
                energy_efficiency_iteration = energy_efficiency_iteration,
                has_energy                = has_energy,
                baseline_stdout           = baseline_stdout,
                best_run_stdout           = run_stdout,
                baseline_energy_j         = baseline_energy_j,
                baseline_power_w          = baseline_power_w,
                best_energy               = -1.0,
                best_power_w              = -1.0,
                best_dur_s                = -1.0,
                energy_efficiency_success = False,
                input_code                = input_code,
                best_code_block           = code_block,
                model                     = model,
                system_prompt             = system_prompt,
            )

            return True, code_block

    # ── Post-loop energy verdict ──────────────────────────────────────────────
    # Reached when the loop exits via break (stagnation/temperature ceiling)
    # or exhausts all attempts.  Evaluate the best code found against threshold.
    if has_energy and energy_check_enabled and best_energy > 0:
        best_reduction_pct = (baseline_energy_j - best_energy) / baseline_energy_j * 100.0
        if best_reduction_pct >= min_energy_gain_pct:
            task_label     = "energy"
            versioned      = experiment_dir / f"Main_v1_Trial{trial_number}{src_extension}"
            versioned.write_text(best_code_block, encoding="utf-8")

            runtime_min    = round((time.time() - start_time) / 60, 2)
            energy_summary = (
                f"  Energy   : {baseline_energy_j:.1f} J → {best_energy:.1f} J  "
                f"({best_reduction_pct:+.1f}%)  "
                f"[{energy_efficiency_iteration} improvement(s)]\n"
            )
            print(
                f"\n  {_B}{_G}✅  Energy complete{_E}"
                f"  —  {runtime_min} min total\n"
                + energy_summary
            )
            print(f"  Best code saved : {versioned}")

            experiment_data["success"]              = True
            experiment_data["runtime_min"]          = runtime_min
            experiment_data["baseline_energy_j"]    = baseline_energy_j
            experiment_data["optimized_energy_j"]   = best_energy
            experiment_data["energy_reduction_pct"] = round(best_reduction_pct, 2)

            with csv_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writerow(experiment_data)

            _write_metadata(
                experiment_dir            = experiment_dir,
                experiment_name           = experiment_name,
                runtime_s                 = time.time() - start_time,
                attempt_count             = attempt,
                energy_efficiency_iteration = energy_efficiency_iteration,
                has_energy                = has_energy,
                baseline_stdout           = baseline_stdout,
                best_run_stdout           = best_run_stdout,
                baseline_energy_j         = baseline_energy_j,
                baseline_power_w          = baseline_power_w,
                best_energy               = best_energy,
                best_power_w              = best_power_w,
                best_dur_s                = best_dur_s,
                energy_efficiency_success = energy_efficiency_success,
                input_code                = input_code,
                best_code_block           = best_code_block,
                model                     = model,
                system_prompt             = system_prompt,
            )

            return True, best_code_block

    print(f"\n{_B}{_R}❌  PIPELINE: no sufficient energy improvement found "
          f"(best: {best_energy:.1f} J, baseline: {baseline_energy_j:.1f} J).{_E}\n"
          if has_energy and energy_check_enabled else
          f"\n{_B}{_R}❌  PIPELINE ERROR: exceeded {_MAX_ATTEMPTS} self-correction attempts.{_E}\n")

    _write_metadata(
        experiment_dir            = experiment_dir,
        experiment_name           = experiment_name,
        runtime_s                 = time.time() - start_time,
        attempt_count             = attempt,
        energy_efficiency_iteration = energy_efficiency_iteration,
        has_energy                = has_energy,
        baseline_stdout           = baseline_stdout,
        best_run_stdout           = best_run_stdout,
        baseline_energy_j         = baseline_energy_j,
        baseline_power_w          = baseline_power_w,
        best_energy               = best_energy,
        best_power_w              = best_power_w,
        best_dur_s                = best_dur_s,
        energy_efficiency_success = False,  # no metadata comparison on failure
        input_code                = input_code,
        best_code_block           = best_code_block,
        model                     = model,
        system_prompt             = system_prompt,
    )

    return False, best_code_block if best_code_block else code_block


# ──────────────────────────────────────────────────────────────────────────────
# "Both" pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_both_pipeline(
    *,
    model: str,
    resource: str,
    resource_cfg: ResourceConfig,
    session: HPCSession,
    compute_resource,
    remote_dir: str,
    account: str,
    queue: str,
    run_args: list[str],
    cuda_input_code: str,
    original_cu_path: Path,
    experiment_dir: Path,
    experiment_name: str,
    csv_path: Path,
    run_id: str,
    sycl_compiler_env: dict[str, str],
    sycl_energy_compiler_env: dict[str, str],
    use_self_prompting: bool,
    verifyprompt: bool,
    verbose: bool,
    min_energy_gain_pct: float,
    use_judge: bool = True,
    verification: str = "none",
    walltime_s: int = _JOB_DURATION_S,
    persistent: PersistentWorker | None = None,
) -> bool:
    """Run translate then energy measurement of the translated SYCL code.

    Step 1: Translate CUDA → SYCL (translate pipeline).
    Step 2: Measure energy of the SYCL code vs original CUDA baseline.
    """
    def _make_runner(worker_key: str, compiler_env: dict[str, str]) -> JobRunner:
        _w = _WORKERS[resource][worker_key]
        return JobRunner(
            session         = session,
            compute_resource = compute_resource,
            remote_dir      = remote_dir,
            account         = account,
            queue           = queue,
            filesystems     = resource_cfg.filesystems,
            worker_script   = f"{remote_dir}/workers/{_w[0]}",
            worker_task_arg = _w[1],
            compiler_env    = compiler_env,
            walltime_s      = walltime_s,
            persistent      = persistent,
        )

    # ── Step 1: Translation ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {_B}BOTH TASK — Step 1 of 2: CUDA→SYCL Translation{_E}")
    print(f"{'='*72}")

    translate_exp_name = experiment_name + "_translate"
    translate_exp_dir  = experiment_dir / "translate"
    translate_exp_dir.mkdir(parents=True, exist_ok=True)

    translate_csv  = translate_exp_dir / "results_translate.csv"
    translate_data = {
        "timestamp": experiment_name.split("__")[-1] if "__" in experiment_name else "",
        "app": "XSBench", "model": model, "gpu_vendor": resource_cfg.gpu_vendor,
        "success": False, "runtime_min": 0.0,
    }
    with translate_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(translate_data.keys())).writeheader()

    _llm.configure(log_dir=str(translate_exp_dir / "llm_calls"), truncate_chars=4000, debug=verbose)

    translate_ok, sycl_code = run_pipeline(
        model              = model,
        task               = "translate",
        resource           = resource,
        input_code         = cuda_input_code,
        session            = session,
        runner             = _make_runner("translate", sycl_compiler_env),
        remote_dir         = remote_dir,
        src_extension      = ".cpp",
        run_args           = run_args,
        experiment_name    = translate_exp_name,
        experiment_dir     = translate_exp_dir,
        csv_path           = translate_csv,
        experiment_data    = translate_data,
        trial_number       = 1,
        run_id             = run_id,
        use_self_prompting = use_self_prompting,
        verifyprompt       = verifyprompt,
        verbose            = verbose,
        has_energy         = False,
    )

    if not translate_ok:
        print(f"\n  {_R}Translation step failed — skipping energy measurement.{_E}")
        return False

    sycl_path = experiment_dir / "Main_translated.cpp"
    sycl_path.write_text(sycl_code, encoding="utf-8")
    print(f"\n  Translated SYCL code saved to: {sycl_path}")

    # ── Step 2: Energy measurement ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {_B}BOTH TASK — Step 2 of 2: SYCL Energy Optimization{_E}")
    print(f"{'='*72}")

    energy_runner = _make_runner("energy_sycl", sycl_energy_compiler_env)

    energy_exp_name = experiment_name + "_sycl_energy"
    energy_exp_dir  = experiment_dir / "sycl_energy"
    energy_exp_dir.mkdir(parents=True, exist_ok=True)

    _llm.configure(log_dir=str(energy_exp_dir / "llm_calls"), truncate_chars=4000, debug=verbose)

    # Baseline: CUDA on Polaris, or initial SYCL translation on Aurora.
    if resource == "polaris":
        baseline_runner   = _make_runner("energy_cuda", {})
        baseline_src_path = original_cu_path
        baseline_ext      = ".cu"
    else:
        baseline_runner   = energy_runner
        baseline_src_path = sycl_path
        baseline_ext      = ".cpp"

    baseline_energy_j, baseline_power_w, baseline_stdout = _run_baseline(
        session           = session,
        runner            = baseline_runner,
        remote_dir        = remote_dir,
        original_src_path = baseline_src_path,
        src_extension     = baseline_ext,
        experiment_name   = energy_exp_name,
        experiment_dir    = energy_exp_dir,
        run_args          = run_args,
        run_id            = run_id + "b",
        verification      = verification,
    )

    energy_csv_fields = [
        "timestamp", "app", "model", "gpu_vendor",
        "success", "runtime_min",
        "baseline_energy_j", "optimized_energy_j", "energy_reduction_pct",
    ]
    energy_csv  = energy_exp_dir / "results_energy.csv"
    energy_data = {
        "timestamp":            experiment_name.split("__")[-1] if "__" in experiment_name else "",
        "app":                  "XSBench",
        "model":                model,
        "gpu_vendor":           resource_cfg.gpu_vendor,
        "success":              False,
        "runtime_min":          0.0,
        "baseline_energy_j":    baseline_energy_j,
        "optimized_energy_j":   -1.0,
        "energy_reduction_pct": 0.0,
    }
    with energy_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=energy_csv_fields).writeheader()

    energy_ok, _ = run_pipeline(
        model               = model,
        task                = "sycl_energy",
        resource            = resource,
        input_code          = sycl_code,
        session             = session,
        runner              = energy_runner,
        remote_dir          = remote_dir,
        src_extension       = ".cpp",
        run_args            = run_args,
        experiment_name     = energy_exp_name,
        experiment_dir      = energy_exp_dir,
        csv_path            = energy_csv,
        experiment_data     = energy_data,
        trial_number        = 1,
        run_id              = run_id,
        use_self_prompting  = use_self_prompting,
        verifyprompt        = verifyprompt,
        verbose             = verbose,
        baseline_energy_j   = baseline_energy_j,
        baseline_power_w    = baseline_power_w,
        baseline_stdout     = baseline_stdout,
        min_energy_gain_pct = min_energy_gain_pct,
        has_energy          = True,
        use_judge           = use_judge,
        verification        = verification,
    )

    return translate_ok and energy_ok


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ── Pre-parse --config before building the real parser ───────────────────
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default="")
    _pre_args, _ = _pre.parse_known_args()
    _config_path = (
        Path(_pre_args.config) if _pre_args.config
        else _SCRIPT_DIR / "lassi_config.py"
    )
    _cfg = _load_config(_config_path)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    p.add_argument(
        "--config",
        default="",
        metavar="PATH",
        help=(
            "Path to a Python config file (default: lassi_config.py in the script directory). "
            "Config values act as defaults; CLI arguments override them."
        ),
    )

    # ── Resource + task ───────────────────────────────────────────────────────
    p.add_argument(
        "--resource",
        default=None,
        choices=["polaris", "aurora"],
        help="Target HPC resource: polaris (NVIDIA A100) or aurora (Intel PVC)",
    )
    p.add_argument(
        "--task",
        default=None,
        choices=["translate", "energy", "both"],
        help=(
            "translate: CUDA→SYCL translation with correctness verification. "
            "energy: GPU energy optimization (CUDA→CUDA on Polaris, SYCL→SYCL on Aurora). "
            "both: translate CUDA→SYCL then measure SYCL energy vs CUDA baseline."
        ),
    )
    p.add_argument(
        "--input-framework",
        default="cuda",
        choices=["cuda", "sycl"],
        help=(
            "Input code framework (default: cuda). "
            "Use sycl for Aurora energy task with an existing SYCL source. "
            "Ignored for translate and both tasks."
        ),
    )

    # ── Source application ────────────────────────────────────────────────────
    p.add_argument(
        "--app-category",
        default="Miniapps",
        metavar="CATEGORY",
        help="Application category from source_apps.json (default: Miniapps)",
    )
    p.add_argument(
        "--app-name",
        default="XSBench",
        metavar="APP",
        help="Application name within the category (default: XSBench)",
    )
    p.add_argument(
        "--sycl-source",
        default="",
        metavar="PATH",
        help=(
            "Path to an existing SYCL .cpp file (for Aurora energy task with "
            "--input-framework sycl). Required when --resource aurora --task energy."
        ),
    )

    # ── Identity / deployment ─────────────────────────────────────────────────
    p.add_argument(
        "--alcf-username",
        default=os.environ.get("USER", ""),
        metavar="USER",
        help="ALCF login name for rsync/SSH (default: $USER)",
    )
    p.add_argument(
        "--remote-dir",
        default="",
        metavar="PATH",
        help=(
            "Absolute remote filesystem path for this project. "
            "Defaults to /eagle/EE-ECP/LASSI-EE/lassi_jobs (Polaris) or "
            "/flare/EE-ECP/LASSI-EE/lassi_jobs (Aurora)."
        ),
    )
    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip initial rsync of lassi_jobs/ to the remote filesystem.",
    )

    # ── PBS job parameters ────────────────────────────────────────────────────
    p.add_argument("--account", default=_DEFAULT_ACCOUNT,
                   help=f"PBS account (default: {_DEFAULT_ACCOUNT})")
    p.add_argument("--queue",   default=_DEFAULT_QUEUE,
                   help=f"PBS queue (default: {_DEFAULT_QUEUE})")
    p.add_argument("--walltime", type=int, default=_JOB_DURATION_S, metavar="SECONDS",
                   help=(
                       f"PBS walltime in seconds (default: {_JOB_DURATION_S}). "
                       "For the persistent worker this is the total budget for the whole pipeline run."
                   ))
    p.add_argument(
        "--no-persistent-worker",
        dest="persistent_worker",
        action="store_false",
        default=True,
        help=(
            "Disable persistent worker mode and submit one PBS job per attempt "
            "(slower due to repeated queue wait; useful for debugging)."
        ),
    )

    # ── LLM parameters ────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Argo model name (default: {_DEFAULT_MODEL}). Examples: gpt4o, gpt41, claudesonnet4",
    )

    # ── Pipeline options ──────────────────────────────────────────────────────
    p.add_argument(
        "--no-self-prompting",
        dest="self_prompting",
        action="store_false",
        help="Disable Stage 0 self-prompting context builder",
    )
    p.add_argument(
        "--no-judge",
        dest="use_judge",
        action="store_false",
        help="Disable LLM-as-judge functional-equivalence check after each successful run",
    )
    p.add_argument(
        "--min-energy-gain",
        type=float,
        default=_DEFAULT_MIN_GAIN_PCT,
        metavar="PCT",
        help=(
            f"Minimum energy reduction (%%) required to declare success for energy tasks "
            f"(default: {_DEFAULT_MIN_GAIN_PCT:.1f}%%). Use 0 to accept any correct optimization."
        ),
    )
    p.add_argument(
        "--verifyprompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print full prompts before each LLM call",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: print prompt sizes and extra diagnostics",
    )

    # ── Compiler overrides (forwarded to worker scripts) ──────────────────────
    p.add_argument(
        "--sycl-compiler",
        default="",
        metavar="PATH",
        help="Override SYCL compiler on the compute node (sets SYCL_COMPILER; default: icpx)",
    )
    p.add_argument(
        "--sycl-flags",
        default="",
        metavar="FLAGS",
        help="Override SYCL compilation flags (sets SYCL_FLAGS)",
    )
    p.add_argument(
        "--sycl-ldflags",
        default="",
        metavar="FLAGS",
        help="Extra SYCL linker flags (sets SYCL_LDFLAGS)",
    )
    p.add_argument(
        "--cuda-compiler",
        default="",
        metavar="PATH",
        help="Override CUDA compiler on the compute node (sets CUDA_COMPILER; default: nvcc)",
    )
    p.add_argument(
        "--cuda-flags",
        default="",
        metavar="FLAGS",
        help="Override CUDA compilation flags (sets CUDA_FLAGS)",
    )
    p.add_argument(
        "--cuda-ldflags",
        default="",
        metavar="FLAGS",
        help="Extra CUDA linker flags (sets CUDA_LDFLAGS)",
    )

    p.set_defaults(self_prompting=True, use_judge=True)
    # Apply config file values as defaults (CLI args override these).
    if _cfg:
        p.set_defaults(**_cfg)
    args = p.parse_args()

    if not args.resource:
        p.error("--resource is required (or set resource = 'polaris' in lassi_config.py)")
    if not args.task:
        p.error("--task is required (or set task = 'energy' in lassi_config.py)")
    if not args.alcf_username:
        p.error("--alcf-username is required (or set alcf_username = '...' in lassi_config.py)")

    if args.debug:
        _llm.configure(debug=True)

    # ── Validate task/resource combinations ───────────────────────────────────
    resource     = args.resource
    task         = args.task
    resource_cfg = _RESOURCES[resource]

    if task == "energy" and resource == "aurora" and args.input_framework == "cuda" and not args.sycl_source:
        p.error(
            "Aurora energy task requires a SYCL input source.\n"
            "Either:\n"
            "  (a) use --input-framework sycl --sycl-source /path/to/Main.cpp, or\n"
            "  (b) use --task both to translate CUDA→SYCL first, then optimize energy."
        )

    # ── Resolve remote directory ───────────────────────────────────────────────
    remote_dir = args.remote_dir or resource_cfg.default_dir

    # ── Banner ─────────────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {_B}LASSI-Jobs{_E}  |  resource={resource}  task={task}")
    print(_SEP)
    print(f"  LLM        : {_B}{args.model}{_E}  via Argo  (apps-dev.inside.anl.gov)")
    print(f"  Resource   : {_B}{resource}{_E} ({resource_cfg.gpu_vendor} GPU)")
    print(f"  Queue      : {_B}{args.queue}{_E}  |  account: {args.account}")
    print(f"  Remote dir : {remote_dir}")
    if task == "energy":
        print(f"  Min gain   : {args.min_energy_gain:.1f}% energy reduction required")
    mode_labels = {
        "translate": f"CUDA→SYCL translation + self-correction ({'with' if args.self_prompting else 'no'} self-prompting)",
        "energy":    f"Energy optimization + self-correction ({'with' if args.self_prompting else 'no'} self-prompting)",
        "both":      f"Translate CUDA→SYCL + SYCL energy optimization",
    }
    print(f"  Mode       : {mode_labels[task]}")
    if args.debug:
        print(f"  Debug      : ON")
    print(_SEP)

    # ── Load source ───────────────────────────────────────────────────────────
    original_cu_path: Path | None = None

    verification: str = "none"

    if task in ("translate", "both") or (task == "energy" and args.input_framework == "cuda"):
        print(f"\n  Loading CUDA source ({args.app_category}/{args.app_name})...")
        input_code, run_args, original_cu_path, verification = _load_cuda_source(
            args.app_category, args.app_name
        )
        print(f"  Source: {len(input_code.splitlines())} lines  |  run args: {run_args}")
    else:
        # Aurora energy with SYCL input
        print(f"\n  Loading SYCL source from: {args.sycl_source}")
        sycl_src_path = Path(args.sycl_source).resolve()
        input_code    = _load_sycl_source(sycl_src_path)
        apps_json     = _find_apps_json()
        catalog       = json.loads(apps_json.read_text())
        app_entry     = catalog.get(args.app_category, {}).get(args.app_name, {})
        run_args      = app_entry.get("app_params", [])
        verification  = app_entry.get("verification", "none")
        original_cu_path = sycl_src_path
        print(f"  Source: {len(input_code.splitlines())} lines  |  run args: {run_args}")

    # ── Connect to ALCF ────────────────────────────────────────────────────────
    print(f"\n  Connecting to ALCF via Globus ({resource_cfg.amsc_resource} + {resource_cfg.amsc_fs})...")
    print("  (A browser window will open if Globus login is required.)\n")
    compute_resource, fs_resource = _amsc_connect(
        resource_cfg.amsc_resource,
        resource_cfg.amsc_fs,
    )

    # ── SSH ControlMaster ─────────────────────────────────────────────────────
    print(f"\n  Opening SSH connection to {resource_cfg.host} (one-time auth)...")
    try:
        ssh_ctl = _ssh_master_start(resource_cfg.host, args.alcf_username)
        print("  SSH connection established.")
    except Exception as exc:
        print(f"  {_R}Warning: SSH ControlMaster failed ({exc}); rsync will prompt per call.{_E}")
        ssh_ctl = ""

    # ── Build HPCSession ───────────────────────────────────────────────────────
    session = HPCSession(
        host        = resource_cfg.host,
        user        = args.alcf_username,
        ssh_ctl     = ssh_ctl,
        fs_resource = fs_resource,
    )

    # ── Initial rsync ──────────────────────────────────────────────────────────
    if not args.skip_sync:
        print(f"\n  Syncing lassi_jobs/ to {resource} remote filesystem...")
        print(f"  {_SCRIPT_DIR}/  →  {args.alcf_username}@{resource_cfg.host}:{remote_dir}/")
        if not session.rsync(str(_SCRIPT_DIR), remote_dir, is_dir=True):
            if ssh_ctl:
                _ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)
            raise SystemExit("Initial rsync to remote host failed.  Check SSH key auth.")
        print("  Sync complete.")
    else:
        print("  --skip-sync: skipping initial rsync.")

    session.mkdirs(
        f"{remote_dir}/generated_codes",
        f"{remote_dir}/generated_codes/experiment",
    )

    # ── Experiment setup ───────────────────────────────────────────────────────
    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id    = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_slug      = re.sub(r"[/: ]", "_", args.model)
    app_slug        = re.sub(r"[/: ]", "_", args.app_name)
    experiment_name = (
        f"exp_{app_slug}_{task}_{model_slug}"
        f"_{resource_cfg.gpu_vendor}__{dt_string}"
    )

    experiment_dir = _SCRIPT_DIR / "generated_codes" / "experiment" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Experiment : {experiment_name}")
    print(f"  Local dir  : {experiment_dir}")
    print(f"  Remote dir : {remote_dir}/generated_codes/experiment/{experiment_name}")
    print()

    # ── Build compiler environment dicts ──────────────────────────────────────
    sycl_compiler_env = {
        k: v for k, v in {
            "SYCL_COMPILER": args.sycl_compiler,
            "SYCL_FLAGS":    args.sycl_flags,
            "SYCL_LDFLAGS":  args.sycl_ldflags,
        }.items() if v
    }
    cuda_compiler_env = {
        k: v for k, v in {
            "CUDA_COMPILER": args.cuda_compiler,
            "CUDA_FLAGS":    args.cuda_flags,
            "CUDA_LDFLAGS":  args.cuda_ldflags,
        }.items() if v
    }

    # ── Launch persistent worker ──────────────────────────────────────────────
    ipc_dir   = f"{remote_dir}/generated_codes/experiment/{experiment_name}/ipc"
    persistent: PersistentWorker | None = None

    if args.persistent_worker:
        if not ssh_ctl:
            print(f"\n  {_R}Warning: SSH ControlMaster not available — "
                  f"falling back to per-job PBS mode.{_E}")
        else:
            print(f"\n  Launching persistent worker on {resource} "
                  f"(walltime {args.walltime}s)...")
            try:
                persistent = PersistentWorker.launch(
                    session          = session,
                    compute_resource = compute_resource,
                    remote_dir       = remote_dir,
                    ipc_dir          = ipc_dir,
                    account          = args.account,
                    queue            = args.queue,
                    filesystems      = resource_cfg.filesystems,
                    walltime_s       = args.walltime,
                    resource_name    = resource,
                )
            except Exception as exc:
                print(f"  {_R}Persistent worker launch failed: {exc}{_E}")
                print(f"  Falling back to per-job PBS mode.")
                persistent = None
    else:
        print(f"\n  --no-persistent-worker: using per-job PBS mode.")

    # ── Helper: build a JobRunner for a given worker key ─────────────────────
    def _make_runner(worker_key: str, compiler_env: dict[str, str]) -> JobRunner:
        _w = _WORKERS[resource][worker_key]
        return JobRunner(
            session         = session,
            compute_resource = compute_resource,
            remote_dir      = remote_dir,
            account         = args.account,
            queue           = args.queue,
            filesystems     = resource_cfg.filesystems,
            worker_script   = f"{remote_dir}/workers/{_w[0]}",
            worker_task_arg = _w[1],
            compiler_env    = compiler_env,
            walltime_s      = args.walltime,
            persistent      = persistent,
        )

    # ── Run the appropriate pipeline ──────────────────────────────────────────
    success = False

    if task == "both":
        assert original_cu_path is not None
        success = run_both_pipeline(
            model                    = args.model,
            resource                 = resource,
            resource_cfg             = resource_cfg,
            session                  = session,
            compute_resource         = compute_resource,
            remote_dir               = remote_dir,
            account                  = args.account,
            queue                    = args.queue,
            run_args                 = run_args,
            cuda_input_code          = input_code,
            original_cu_path         = original_cu_path,
            experiment_dir           = experiment_dir,
            experiment_name          = experiment_name,
            csv_path                 = experiment_dir / "results.csv",
            run_id                   = run_id,
            walltime_s               = args.walltime,
            sycl_compiler_env        = sycl_compiler_env,
            sycl_energy_compiler_env = sycl_compiler_env,
            use_self_prompting       = args.self_prompting,
            verifyprompt             = args.verifyprompt,
            verbose                  = args.debug,
            min_energy_gain_pct      = args.min_energy_gain,
            use_judge                = args.use_judge,
            verification             = verification,
            persistent               = persistent,
        )

    elif task == "energy":
        if resource == "polaris" and args.input_framework == "cuda":
            worker_key = "energy_cuda"
            src_ext    = ".cu"
            comp_env   = cuda_compiler_env
        else:
            worker_key = "energy_sycl"
            src_ext    = ".cpp"
            comp_env   = sycl_compiler_env

        runner = _make_runner(worker_key, comp_env)

        assert original_cu_path is not None
        baseline_energy_j, baseline_power_w, baseline_stdout = _run_baseline(
            session           = session,
            runner            = runner,
            remote_dir        = remote_dir,
            original_src_path = original_cu_path,
            src_extension     = src_ext,
            experiment_name   = experiment_name,
            experiment_dir    = experiment_dir,
            run_args          = run_args,
            run_id            = run_id + "b",
            verification      = verification,
        )

        csv_path = experiment_dir / "results.csv"
        experiment_data = {
            "timestamp":            dt_string,
            "app":                  args.app_name,
            "model":                args.model,
            "gpu_vendor":           resource_cfg.gpu_vendor,
            "success":              False,
            "runtime_min":          0.0,
            "baseline_energy_j":    baseline_energy_j,
            "optimized_energy_j":   -1.0,
            "energy_reduction_pct": 0.0,
        }
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writeheader()

        _llm.configure(log_dir=str(experiment_dir / "llm_calls"), truncate_chars=4000, debug=args.debug)

        success, _ = run_pipeline(
            model               = args.model,
            task                = "energy",
            resource            = resource,
            input_code          = input_code,
            session             = session,
            runner              = runner,
            remote_dir          = remote_dir,
            src_extension       = src_ext,
            run_args            = run_args,
            experiment_name     = experiment_name,
            experiment_dir      = experiment_dir,
            csv_path            = csv_path,
            experiment_data     = experiment_data,
            trial_number        = 1,
            run_id              = run_id,
            use_self_prompting  = args.self_prompting,
            verifyprompt        = args.verifyprompt,
            verbose             = args.debug,
            baseline_energy_j   = baseline_energy_j,
            baseline_power_w    = baseline_power_w,
            baseline_stdout     = baseline_stdout,
            min_energy_gain_pct = args.min_energy_gain,
            has_energy          = True,
            use_judge           = args.use_judge,
            verification        = verification,
        )

    else:
        # task == "translate"
        csv_path = experiment_dir / "results.csv"
        experiment_data = {
            "timestamp":   dt_string,
            "app":         args.app_name,
            "model":       args.model,
            "gpu_vendor":  resource_cfg.gpu_vendor,
            "success":     False,
            "runtime_min": 0.0,
        }
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writeheader()

        _llm.configure(log_dir=str(experiment_dir / "llm_calls"), truncate_chars=4000, debug=args.debug)

        success, _ = run_pipeline(
            model              = args.model,
            task               = "translate",
            resource           = resource,
            input_code         = input_code,
            session            = session,
            runner             = _make_runner("translate", sycl_compiler_env),
            remote_dir         = remote_dir,
            src_extension      = ".cpp",
            run_args           = run_args,
            experiment_name    = experiment_name,
            experiment_dir     = experiment_dir,
            csv_path           = csv_path,
            experiment_data    = experiment_data,
            trial_number       = 1,
            run_id             = run_id,
            use_self_prompting = args.self_prompting,
            verifyprompt       = args.verifyprompt,
            verbose            = args.debug,
            has_energy         = False,
        )

    if persistent:
        persistent.shutdown()

    if ssh_ctl:
        _ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
