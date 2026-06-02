#!/usr/bin/env python3
"""LASSI XSBench CUDA Energy Optimizer — LLM calls run locally via Argo,
compile + run executed on Polaris via PBS job submission (amsc_client).

Goal
----
Optimize an existing CUDA program (XSBench) for GPU energy efficiency using
an LLM-driven self-correction loop, while preserving numerical correctness.
The optimized code stays in CUDA (not translated to another language).

LLM backend
-----------
Uses the ANL Argo gateway (apps-dev.inside.anl.gov/argoapi/...).
Accessible from any machine on the ANL network — no ALCF login needed for LLM.
Set ARGO_USER to your ANL username (default: $USER).
Set ARGO_API_KEY if your deployment requires a Bearer token.

Architecture
------------
  Baseline : Compile + run the original CUDA source to measure reference energy
             (one Polaris PBS job, no LLM call)

  Stage 0  : Self-prompting context builder     — local Argo LLM call
               1. Summarize CUDA energy efficiency reference guide
               2. Analyze the CUDA source for energy inefficiency patterns

  Stage 1  : Initial energy optimization         — local Argo LLM call
               LLM generates energy-optimized CUDA code

  Stage 2  : Self-correction loop (up to 15 iterations):
               1. Write generated .cu locally
               2. rsync .cu to Eagle filesystem on Polaris
               3. Submit PBS job: polaris_energy_worker.sh compiles with nvcc
                  + runs XSBench with nvidia-smi dmon power monitoring
               4. Poll until job reaches terminal state
               5. Read build / run / energy logs from Eagle via amsc_client
               6. Parse: compile error / runtime error / checksum mismatch /
                         insufficient energy improvement
               7. If correctness error: generate correction via local Argo LLM → repeat
               8. If energy not improved: request more aggressive optimization → repeat
               9. If checksum valid AND energy improved: success

Usage
-----
  python run_lassi_energy_polaris_v1.py --alcf-username vchirio [options]

  # Custom model, skip initial sync
  python run_lassi_energy_polaris_v1.py --alcf-username vchirio --model gpt4o --skip-sync

  # Disable self-prompting, require 5% energy improvement to declare success
  python run_lassi_energy_polaris_v1.py --alcf-username vchirio \\
      --no-self-prompting --min-energy-gain 5.0

  # Custom nvcc flags on Polaris
  python run_lassi_energy_polaris_v1.py --alcf-username vchirio \\
      --cuda-flags "-O3 -arch=sm_80 -std=c++17 --use_fast_math"

Argo model names (examples)
---------------------------
  gpt4o         GPT-4o via ANL Argo gateway
  gpt41         GPT-4.1
  claudesonnet4 Claude Sonnet 4
  gpto3mini     o3-mini
  gpt52         GPT-5 (when available)
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

# ── Local lassi module (self-contained copy in this directory) ─────────────────
import lassi.llm_methods as _llm
from lassi.format_colors import color
from lassi.llm_call_logger import LLMCallLogger
from lassi.prompt_dictionary import PromptDictionary

_B = color.BOLD
_G = color.GREEN
_R = color.RED
_E = color.END
_SEP  = "=" * 72
_THIN = "─" * 72

_SCRIPT_DIR = Path(__file__).resolve().parent

# ── Defaults ────────────────────────────────────────────────────────────────────
_DEFAULT_POLARIS_DIR  = "/eagle/EE-ECP/LASSI-EE/lassi_energy_polaris"
_DEFAULT_MODEL        = "gpt55"
_DEFAULT_ACCOUNT      = "EE-ECP"
_DEFAULT_QUEUE        = "debug"
_DEFAULT_MIN_GAIN_PCT = 1.0    # minimum energy reduction (%) to declare success
_POLARIS_HOST         = "polaris.alcf.anl.gov"
_MAX_ATTEMPTS         = 15
_JOB_DURATION_S       = 600    # 10 min per compile+run attempt
_JOB_WAIT_TIMEOUT_S   = 900    # 15 min total before giving up on a single job
_JOB_POLL_INTERVAL_S  = 15     # seconds between status polls
_EAGLE_READ_TIMEOUT   = 60     # seconds for eagle.fs.head()
_EAGLE_PROPAGATION_S  = 5      # seconds to wait after job ends before reading Eagle


# ──────────────────────────────────────────────────────────────────────────────
# Source loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_source() -> tuple[str, list[str]]:
    """Load XSBench CUDA source (Main.cu + headers) and exe parameters.

    Headers (*.cuh, *.h) are prepended to Main.cu so the full optimization
    context is visible to the LLM in one shot.

    Falls back to the sibling lassi_translate_polaris/source_codes/ directory
    if no local source_codes/ exists, so the two pipelines can share source.
    """
    search_bases = [_SCRIPT_DIR, _SCRIPT_DIR.parent / "lassi_translate_polaris"]
    src_dir: Path | None = None
    for base in search_bases:
        candidate = base / "source_codes" / "Miniapps" / "XSBench" / "cuda"
        if (candidate / "Main.cu").exists():
            src_dir = candidate
            break

    if src_dir is None:
        raise FileNotFoundError(
            "XSBench CUDA source not found.  Expected:\n"
            f"  {_SCRIPT_DIR / 'source_codes' / 'Miniapps' / 'XSBench' / 'cuda' / 'Main.cu'}\n"
            "Copy or symlink the source_codes/ directory here, or place it under "
            "lassi_translate_polaris/ (sibling directory)."
        )

    code = (src_dir / "Main.cu").read_text(encoding="utf-8")

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

    apps_json_paths = [
        _SCRIPT_DIR / "source_apps.json",
        _SCRIPT_DIR.parent / "lassi_translate_polaris" / "source_apps.json",
    ]
    apps_json = next((p for p in apps_json_paths if p.exists()), None)
    if apps_json is None:
        raise FileNotFoundError("source_apps.json not found")
    app_params: list[str] = json.loads(apps_json.read_text())["Miniapps"]["XSBench"]["app_params"]
    return code, app_params


# ──────────────────────────────────────────────────────────────────────────────
# ALCF connection (amsc_client) — for job submission and Eagle file I/O only
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


def _amsc_connect():
    """Connect to ALCF via Globus; return (polaris_resource, eagle_resource).

    A browser window opens automatically on first call for Globus login.
    Subsequent calls use cached credentials (~/.amsc/credentials.json).

    Note: this auth is for PBS job submission and Eagle filesystem access only.
          LLM calls use the Argo gateway and need no ALCF credentials.
    """
    _apply_ipv4_patch()

    try:
        from amsc_client import Client
    except ImportError:
        for parent in [_SCRIPT_DIR, _SCRIPT_DIR.parent, _SCRIPT_DIR.parent.parent]:
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
    client  = Client(token="not-needed")
    print("  Fetching ALCF facility...", flush=True)
    alcf    = client.facility("alcf")
    print("  Getting Polaris resource...", flush=True)
    polaris = alcf.resource("Polaris")
    print("  Getting Eagle resource...", flush=True)
    eagle   = alcf.resource("Eagle")

    status = polaris.status
    print(f"  Polaris status: {status}", flush=True)
    if status.lower() not in ("up", "available", "running"):
        print(f"  {_B}WARNING:{_E} Polaris reports '{status}' — job may queue longer.")

    return polaris, eagle


# ──────────────────────────────────────────────────────────────────────────────
# Eagle filesystem helpers
# ──────────────────────────────────────────────────────────────────────────────

def _eagle_mkdir(eagle, path: str) -> None:
    """Create a single directory on Eagle (non-fatal if it already exists)."""
    try:
        eagle.fs.stat(path).wait(timeout=20)
        return
    except Exception:
        pass
    try:
        eagle.fs.mkdir(path).wait(timeout=30)
    except Exception:
        pass


def _eagle_mkdirs(eagle, *paths: str) -> None:
    """Create a sequence of directories on Eagle, one level at a time."""
    for path in paths:
        _eagle_mkdir(eagle, path)


def _read_eagle_file(eagle, path: str, max_lines: int = 10000) -> str:
    """Read a file from Eagle; return content as a string."""
    try:
        task = eagle.fs.head(path, lines=max_lines)
        task.wait(timeout=_EAGLE_READ_TIMEOUT)
        result = task.result
        if isinstance(result, dict):
            return (
                result.get("output", {}).get("content", "")
                or result.get("content", "")
                or str(result)
            )
        return str(result) if result else ""
    except Exception as exc:
        return f"[Eagle read error for {path!r}: {exc}]"


# ──────────────────────────────────────────────────────────────────────────────
# SSH ControlMaster — one persistent connection, reused for all rsync calls
# ──────────────────────────────────────────────────────────────────────────────

def _ssh_master_start(alcf_username: str) -> str:
    """Open a persistent SSH ControlMaster connection to Polaris."""
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
        f"{alcf_username}@{_POLARIS_HOST}",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH ControlMaster to {_POLARIS_HOST} failed (rc={result.returncode})"
        )
    return ctl_path


def _ssh_master_stop(ctl_path: str, alcf_username: str) -> None:
    """Close the ControlMaster connection and remove the socket."""
    subprocess.run(
        ["ssh", "-o", f"ControlPath={ctl_path}", "-O", "exit",
         f"{alcf_username}@{_POLARIS_HOST}"],
        capture_output=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# File sync to Polaris Eagle via rsync
# ──────────────────────────────────────────────────────────────────────────────

def _rsync(
    local_src: str,
    remote_dst: str,
    alcf_username: str,
    is_dir: bool = False,
    ssh_ctl: str = "",
) -> bool:
    """rsync local_src to remote_dst on Polaris Eagle.  Returns True on success."""
    dst = f"{alcf_username}@{_POLARIS_HOST}:{remote_dst}"
    ssh_cmd = f"ssh -o ControlPath={ssh_ctl}" if ssh_ctl else "ssh"
    cmd = ["rsync", "-av", "--mkpath", "-e", ssh_cmd]
    if is_dir:
        local_src = local_src.rstrip("/") + "/"
        cmd += [
            "--delete",
            "--exclude=__pycache__/", "--exclude=*.pyc",
            "--exclude=generated_codes/",
            "--exclude=.git/",
        ]
    cmd += [local_src, dst]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd2 = [c for c in cmd if c != "--mkpath"]
        result = subprocess.run(cmd2, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  {_R}rsync failed (rc={result.returncode}):{_E}")
        if result.stderr.strip():
            print(f"    {result.stderr.strip()[:400]}")
        return False
    return True


def _initial_sync(alcf_username: str, polaris_dir: str, ssh_ctl: str = "") -> bool:
    """rsync the entire lassi_energy_polaris/ directory to Eagle."""
    print(f"  {_SCRIPT_DIR}/  →  {alcf_username}@polaris:{polaris_dir}/")
    return _rsync(str(_SCRIPT_DIR), polaris_dir, alcf_username, is_dir=True, ssh_ctl=ssh_ctl)


def _sync_cu_to_eagle(
    local_cu: Path,
    polaris_dir: str,
    exp_name: str,
    cu_filename: str,
    alcf_username: str,
    ssh_ctl: str = "",
) -> bool:
    """rsync a single generated .cu file to its experiment directory on Eagle."""
    remote = f"{polaris_dir}/generated_codes/experiment/{exp_name}/{cu_filename}"
    return _rsync(str(local_cu), remote, alcf_username, ssh_ctl=ssh_ctl)


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
    """Return the last 5 non-empty lines, indented with 4 spaces."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(f"    {ln}" for ln in lines[-5:])


# ──────────────────────────────────────────────────────────────────────────────
# PBS job submission and monitoring
# ──────────────────────────────────────────────────────────────────────────────

def _submit_compile_run_job(
    *,
    polaris,
    polaris_dir: str,
    cu_eagle_path: str,
    results_eagle_dir: str,
    run_args: list[str],
    job_name: str,
    account: str,
    queue: str,
    cuda_compiler: str = "",
    cuda_flags: str = "",
    cuda_ldflags: str = "",
):
    """Submit a PBS job that runs polaris_energy_worker.sh on a Polaris compute node.

    polaris_energy_worker.sh will:
      1. Load the CUDA toolkit module to get nvcc
      2. Compile <cu_eagle_path> to <results_eagle_dir>/Main
      3. Run Main with <run_args>, measuring GPU power via nvidia-smi dmon
      4. Write build_rc.txt, build_stdout/stderr.log, run_rc.txt,
         run_stdout/stderr.log, power_samples.log, energy_results.txt
         to <results_eagle_dir>

    PBS stdout/stderr → <results_eagle_dir>/pbs_{stdout,stderr}.log
    """
    worker           = f"{polaris_dir}/polaris_energy_worker.sh"
    run_args_quoted  = " ".join(shlex.quote(a) for a in run_args)

    env_parts: list[str] = []
    if cuda_compiler:
        env_parts.append(f"CUDA_COMPILER={shlex.quote(cuda_compiler)}")
    if cuda_flags:
        env_parts.append(f"CUDA_FLAGS={shlex.quote(cuda_flags)}")
    if cuda_ldflags:
        env_parts.append(f"CUDA_LDFLAGS={shlex.quote(cuda_ldflags)}")
    env_prefix = (" ".join(env_parts) + " ") if env_parts else ""

    cmd = (
        f"{env_prefix}bash {shlex.quote(worker)}"
        f" {shlex.quote(cu_eagle_path)}"
        f" {shlex.quote(results_eagle_dir)}"
        f" {run_args_quoted}"
    )

    return polaris.submit(
        executable  = "/bin/bash",
        arguments   = ["-c", cmd],
        directory   = polaris_dir,
        name        = job_name,
        queue       = queue,
        account     = account,
        duration    = _JOB_DURATION_S,
        nodes       = 1,
        stdout_path = f"{results_eagle_dir}/pbs_stdout.log",
        stderr_path = f"{results_eagle_dir}/pbs_stderr.log",
        filesystems = "home:eagle",
    )


def _wait_for_job(job, timeout_s: int = _JOB_WAIT_TIMEOUT_S) -> str:
    """Poll job status every _JOB_POLL_INTERVAL_S seconds until terminal.

    HTTP 400 / 'not found' means the job finished and left the PBS queue.
    """
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


def _parse_attempt_results(eagle, results_eagle_dir: str) -> dict:
    """Read compile/run/energy result files from Eagle and return a parsed dict."""
    def _rc(fname: str) -> int:
        raw = _read_eagle_file(eagle, f"{results_eagle_dir}/{fname}", max_lines=5).strip()
        return int(raw) if re.fullmatch(r"-?\d+", raw) else -1

    energy_raw = _read_eagle_file(eagle, f"{results_eagle_dir}/energy_results.txt")
    energy: dict = {}
    try:
        energy = json.loads(energy_raw)
    except Exception:
        energy = {}

    return {
        "build_ok":      _rc("build_rc.txt") == 0,
        "run_ok":        _rc("run_rc.txt")   == 0,
        "build_stdout":  _read_eagle_file(eagle, f"{results_eagle_dir}/build_stdout.log"),
        "build_stderr":  _read_eagle_file(eagle, f"{results_eagle_dir}/build_stderr.log"),
        "run_stdout":    _read_eagle_file(eagle, f"{results_eagle_dir}/run_stdout.log"),
        "run_stderr":    _read_eagle_file(eagle, f"{results_eagle_dir}/run_stderr.log"),
        "energy_joules": energy.get("energy_joules", -1.0),
        "avg_power_w":   energy.get("avg_power_w",   -1.0),
        "duration_s":    energy.get("duration_s",    -1.0),
        "num_samples":   energy.get("num_samples",    0),
    }


def _save_attempt_logs(results: dict, experiment_dir: Path, prefix: str) -> None:
    """Mirror Eagle result files to the local experiment directory."""
    for key, fname in [
        ("build_stdout", "build_stdout.log"),
        ("build_stderr", "build_stderr.log"),
        ("run_stdout",   "run_stdout.log"),
        ("run_stderr",   "run_stderr.log"),
    ]:
        content = results.get(key) or ""
        if content and not content.startswith("[Eagle read error"):
            (experiment_dir / (prefix + fname)).write_text(content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Baseline measurement
# ──────────────────────────────────────────────────────────────────────────────

def _run_baseline(
    *,
    polaris,
    eagle,
    polaris_dir: str,
    original_cu_path: Path,
    experiment_name: str,
    experiment_dir: Path,
    run_args: list[str],
    account: str,
    queue: str,
    alcf_username: str,
    cuda_compiler: str,
    cuda_flags: str,
    cuda_ldflags: str,
    ssh_ctl: str,
    run_id: str,
) -> float:
    """Compile and run the ORIGINAL XSBench source to establish a baseline energy.

    Returns baseline energy in Joules, or -1.0 if the measurement failed.
    """
    print(f"\n{_B}----- BASELINE: compiling + running original CUDA  (Polaris PBS) -----{_E}\n")

    baseline_cu_name = "Main_baseline.cu"
    baseline_dir     = f"{polaris_dir}/generated_codes/experiment/{experiment_name}"

    # Upload the original .cu to Eagle under the experiment directory.
    print(f"  Uploading {baseline_cu_name} to Eagle...", end="  ", flush=True)
    ok = _sync_cu_to_eagle(
        local_cu      = original_cu_path,
        polaris_dir   = polaris_dir,
        exp_name      = experiment_name,
        cu_filename   = baseline_cu_name,
        alcf_username = alcf_username,
        ssh_ctl       = ssh_ctl,
    )
    if not ok:
        print(f"{_R}FAILED{_E}")
        return -1.0
    print(f"{_G}OK{_E}")

    cu_eagle_path    = f"{baseline_dir}/{baseline_cu_name}"
    results_eagle_dir = f"{baseline_dir}/baseline_results"
    _eagle_mkdir(eagle, results_eagle_dir)

    job_name = f"lassi-baseline-{run_id}"
    print(f"  Submitting baseline PBS job ({job_name})...", end="  ", flush=True)
    try:
        job = _submit_compile_run_job(
            polaris           = polaris,
            polaris_dir       = polaris_dir,
            cu_eagle_path     = cu_eagle_path,
            results_eagle_dir = results_eagle_dir,
            run_args          = run_args,
            job_name          = job_name,
            account           = account,
            queue             = queue,
            cuda_compiler     = cuda_compiler,
            cuda_flags        = cuda_flags,
            cuda_ldflags      = cuda_ldflags,
        )
    except Exception as exc:
        print(f"{_R}FAILED: {exc}{_E}")
        return -1.0

    pbs_id = job.id.split(".")[0]
    print(f"{_G}submitted{_E}  (PBS ID: {job.id})")

    print(f"  Waiting for baseline job {pbs_id}...", end="  ", flush=True)
    final_state = _wait_for_job(job)
    state_color = _G if final_state == "completed" else _R
    print(f"{state_color}{final_state}{_E}")

    time.sleep(_EAGLE_PROPAGATION_S)
    results = _parse_attempt_results(eagle, results_eagle_dir)
    _save_attempt_logs(results, experiment_dir, "baseline_")

    if not results["build_ok"]:
        print(f"  {_R}Baseline compile FAILED — cannot establish energy reference.{_E}")
        tail = _tail5(results["build_stderr"])
        if tail:
            print(tail)
        return -1.0

    if not results["run_ok"]:
        print(f"  {_R}Baseline run FAILED — cannot establish energy reference.{_E}")
        return -1.0

    if not _xsbench_checksum_valid(results["run_stdout"]):
        cs = _extract_xsbench_checksum(results["run_stdout"])
        print(f"  {_R}Baseline checksum invalid ({cs!r}) — energy reference unreliable.{_E}")

    ej  = results["energy_joules"]
    apw = results["avg_power_w"]
    ds  = results["duration_s"]
    ns  = results["num_samples"]

    if ej > 0:
        print(
            f"  {_B}{_G}Baseline energy : {ej:.1f} J  "
            f"({apw:.1f} W avg,  {ds:.1f} s,  {ns} samples){_E}"
        )
    else:
        print(
            f"  {_R}Warning: energy measurement unavailable (nvidia-smi not found?).{_E}\n"
            f"  Pipeline will proceed and check correctness only."
        )

    # Save baseline energy to a local JSON file for record-keeping.
    baseline_json = {
        "energy_joules": ej,
        "avg_power_w":   apw,
        "duration_s":    ds,
        "num_samples":   ns,
    }
    (experiment_dir / "baseline_energy.json").write_text(
        json.dumps(baseline_json, indent=2), encoding="utf-8"
    )

    return ej


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    model: str,
    alcf_username: str,
    polaris_dir: str,
    account: str,
    queue: str,
    run_args: list[str],
    input_code: str,
    original_cu_path: Path,
    experiment_name: str,
    experiment_dir: Path,
    csv_path: Path,
    experiment_data: dict,
    trial_number: int,
    polaris,
    eagle,
    run_id: str,
    use_self_prompting: bool = True,
    verifyprompt: bool = False,
    verbose: bool = False,
    cuda_compiler: str = "",
    cuda_flags: str = "",
    cuda_ldflags: str = "",
    ssh_ctl: str = "",
    min_energy_gain_pct: float = _DEFAULT_MIN_GAIN_PCT,
) -> bool:
    """Run the full energy optimization pipeline.

    Baseline : compile + run original CUDA to measure reference energy
    Stage 0  : self-prompting context builder (optional)
    Stage 1  : initial energy optimization via local Argo LLM call
    Stage 2  : self-correction loop — per-attempt: local Argo call
               + Polaris PBS job (nvcc compile + run + energy measurement)
               + Eagle result read

    Returns True when the optimized code is correct (valid XSBench checksum)
    and achieves at least min_energy_gain_pct% energy reduction vs baseline.
    Also returns True if energy measurement is unavailable (baseline == -1)
    and correctness is verified — correctness alone is sufficient fallback.
    """
    prompts    = PromptDictionary()
    start_time = time.time()

    # ── Ensure Eagle experiment directories exist ─────────────────────────────
    _eagle_mkdirs(
        eagle,
        f"{polaris_dir}/generated_codes",
        f"{polaris_dir}/generated_codes/experiment",
        f"{polaris_dir}/generated_codes/experiment/{experiment_name}",
    )

    # ── Baseline measurement ──────────────────────────────────────────────────
    baseline_energy_j = _run_baseline(
        polaris           = polaris,
        eagle             = eagle,
        polaris_dir       = polaris_dir,
        original_cu_path  = original_cu_path,
        experiment_name   = experiment_name,
        experiment_dir    = experiment_dir,
        run_args          = run_args,
        account           = account,
        queue             = queue,
        alcf_username     = alcf_username,
        cuda_compiler     = cuda_compiler,
        cuda_flags        = cuda_flags,
        cuda_ldflags      = cuda_ldflags,
        ssh_ctl           = ssh_ctl,
        run_id            = run_id,
    )

    energy_check_enabled = baseline_energy_j > 0
    if not energy_check_enabled:
        print(
            f"\n  {_B}Note:{_E} Baseline energy unavailable — "
            "pipeline will verify correctness only (no energy comparison)."
        )

    # ── STAGE 0: Self-prompting (local Argo LLM calls) ────────────────────────
    energy_guide_summary    = ""
    source_code_description = ""

    if use_self_prompting:
        print(f"\n{_B}----- STAGE 0: SELF-PROMPTING  (local Argo) -----{_E}\n")

        try:
            energy_guide_raw = prompts.get_codeknowledge("CUDA_energy_guide")
        except Exception as exc:
            print(f"  {_B}WARN:{_E} Could not load CUDA energy guide: {exc}.  Skipping.")
            energy_guide_raw = ""

        if energy_guide_raw:
            summarize_prompt = (
                prompts.get_selfprompt("summarize_context")
                + "\n\n[BEGIN CUDA ENERGY EFFICIENCY GUIDE]\n"
                + energy_guide_raw
                + "\n[END CUDA ENERGY EFFICIENCY GUIDE]"
            )
            print("  Step 1/2: Summarizing CUDA energy efficiency guide...")
            energy_guide_summary = _llm.generate_text(
                model,
                prompts.get_system_prompt("selfprompt"),
                summarize_prompt,
                intent="energy guide summarization",
                verifyprompt=verifyprompt,
                label="[LLM] Summarizing CUDA energy efficiency guide",
            )

        describe_prompt = (
            prompts.get_selfprompt("describe_source")
            + "\n\n[BEGIN CUDA SOURCE CODE]\n"
            + input_code
            + "\n[END CUDA SOURCE CODE]"
        )
        print("  Step 2/2: Analyzing source code for energy inefficiencies...")
        source_code_description = _llm.generate_text(
            model,
            prompts.get_system_prompt("selfprompt"),
            describe_prompt,
            intent="source code energy analysis",
            verifyprompt=verifyprompt,
            label="[LLM] Analyzing source code energy patterns",
        )

        print(f"\n  {_B}{_G}✓{_E}  Self-prompting complete.  Context incorporated into Stage 1 prompt.\n")

    # Build the Stage 1 user prompt.
    system_prompt   = prompts.get_system_prompt("CUDA_optimize_energy")
    optimize_prompt = prompts.get_codetranslate_prompt("CUDA_optimize_energy")

    if use_self_prompting and (energy_guide_summary or source_code_description):
        preamble = ""
        if energy_guide_summary:
            preamble += (
                "[CUDA ENERGY EFFICIENCY KNOWLEDGE]\n"
                + energy_guide_summary
                + "\n[/CUDA ENERGY EFFICIENCY KNOWLEDGE]\n\n"
            )
        if source_code_description:
            preamble += (
                "[YOUR ANALYSIS OF ENERGY INEFFICIENCIES IN THE SOURCE CODE]\n"
                + source_code_description
                + "\n[/YOUR ANALYSIS OF ENERGY INEFFICIENCIES IN THE SOURCE CODE]\n\n"
            )
        think_prefix = (
            "Think carefully, using your energy analysis above.  Now, "
            if source_code_description else ""
        )
        optimize_user_prompt = (
            preamble
            + think_prefix
            + optimize_prompt
            + "\n\n[START OF CUDA CODE TO OPTIMIZE]\n"
            + input_code
            + "\n[/END OF CUDA CODE TO OPTIMIZE]"
        )
    else:
        optimize_user_prompt = (
            optimize_prompt
            + "\n\n[START OF CUDA CODE TO OPTIMIZE]\n"
            + input_code
            + "\n[/END OF CUDA CODE TO OPTIMIZE]"
        )

    # ── STAGES 1 + 2: generate → (Polaris compile+run → correct) loop ─────────
    compile_err   = ""
    run_err       = ""
    run_stdout    = ""
    code_block    = ""
    best_energy_j = -1.0

    for attempt in range(_MAX_ATTEMPTS + 1):

        # ── Local Argo LLM call ────────────────────────────────────────────────
        if attempt == 0:
            print(f"\n{_B}----- STAGE 1: CUDA ENERGY OPTIMIZATION  (local Argo) -----{_E}\n")
            code_block = _llm.generate_code(
                model,
                system_prompt,
                optimize_user_prompt,
                verifyprompt,
                label="[LLM] Generating initial energy-optimized CUDA",
            )
        else:
            if attempt == 1:
                print(f"\n{_B}----- STAGE 2: SELF-CORRECTION  (local Argo + Polaris PBS) -----{_E}\n")

            # Select the correction type and build the correction prompt.
            if compile_err.startswith("ENERGY_NOT_IMPROVED"):
                # Energy correct but not improved enough — ask for deeper optimization.
                err_type    = "energy improvement insufficient"
                error_msg   = compile_err
                error_intro = "\n -- Energy result: "
                prompt_key  = "CUDA_optimize_energy_improve"
            elif compile_err == "execute_error":
                err_type    = "runtime error"
                error_msg   = run_err
                if run_stdout.strip():
                    error_msg += "\n\nProgram stdout before crash:\n" + run_stdout
                error_msg  += "\n\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n" + input_code + "\n[/ORIGINAL CUDA SOURCE]\n"
                error_intro = "\n -- Runtime error: "
                prompt_key  = "CUDA_optimize_energy_error"
            elif compile_err.startswith("VERIFICATION MISMATCH"):
                err_type    = "checksum mismatch"
                error_msg   = compile_err
                error_intro = "\n -- Verification mismatch: "
                prompt_key  = "CUDA_optimize_energy_error"
            else:
                err_type    = "compile error"
                error_msg   = compile_err
                error_intro = "\n -- Compile error: "
                prompt_key  = "CUDA_optimize_energy_error"

            print(f"  {_B}Attempt {attempt}/{_MAX_ATTEMPTS}{_E} — correcting {err_type}")

            error_prompt = (
                prompts.get_code_error_correction_prompt(prompt_key)
                + "\n\n[START OF CODE TO FIX]\n"
                + code_block
                + "\n[/END OF CODE TO FIX]"
                + error_intro + "\n"
                + error_msg
            )
            code_block = _llm.generate_code(
                model,
                system_prompt,
                error_prompt,
                verifyprompt,
                label=f"[LLM] Corrected energy-optimized CUDA (attempt {attempt})",
            )

        # ── Write generated code locally ───────────────────────────────────────
        cu_name  = f"Main_attempt{attempt}.cu"
        local_cu = experiment_dir / cu_name
        local_cu.write_text(code_block, encoding="utf-8")

        # ── Upload .cu to Eagle ────────────────────────────────────────────────
        print(f"  Uploading {cu_name} to Eagle...", end="  ", flush=True)
        ok = _sync_cu_to_eagle(
            local_cu      = local_cu,
            polaris_dir   = polaris_dir,
            exp_name      = experiment_name,
            cu_filename   = cu_name,
            alcf_username = alcf_username,
            ssh_ctl       = ssh_ctl,
        )
        if not ok:
            print(f"{_R}FAILED{_E}")
            compile_err = f"rsync failed — could not upload {cu_name} to Eagle"
            run_stdout  = ""
            continue
        print(f"{_G}OK{_E}")

        cu_eagle_path     = f"{polaris_dir}/generated_codes/experiment/{experiment_name}/{cu_name}"
        results_eagle_dir = f"{polaris_dir}/generated_codes/experiment/{experiment_name}/attempt{attempt}_results"
        _eagle_mkdir(eagle, results_eagle_dir)

        # ── Submit PBS compile+run+energy job ─────────────────────────────────
        job_name = f"lassi-e{attempt}-{run_id}"
        print(f"  Submitting PBS job ({job_name})...", end="  ", flush=True)
        try:
            job = _submit_compile_run_job(
                polaris           = polaris,
                polaris_dir       = polaris_dir,
                cu_eagle_path     = cu_eagle_path,
                results_eagle_dir = results_eagle_dir,
                run_args          = run_args,
                job_name          = job_name,
                account           = account,
                queue             = queue,
                cuda_compiler     = cuda_compiler,
                cuda_flags        = cuda_flags,
                cuda_ldflags      = cuda_ldflags,
            )
        except Exception as exc:
            print(f"{_R}FAILED: {exc}{_E}")
            compile_err = f"PBS submission error: {exc}"
            run_stdout  = ""
            continue

        pbs_id = job.id.split(".")[0]
        print(f"{_G}submitted{_E}  (PBS ID: {job.id})")
        if verbose:
            print(f"    Monitor: ssh {_POLARIS_HOST} qstat {pbs_id}")

        # ── Poll until job finishes ────────────────────────────────────────────
        print(f"  Waiting for Polaris job {pbs_id}...", end="  ", flush=True)
        final_state = _wait_for_job(job)
        state_color = _G if final_state == "completed" else _R
        print(f"{state_color}{final_state}{_E}")

        if final_state not in ("completed", "active"):
            compile_err = f"PBS job ended in unexpected state: {final_state!r}"
            run_stdout  = ""
            time.sleep(_EAGLE_PROPAGATION_S)
            partial = _parse_attempt_results(eagle, results_eagle_dir)
            _save_attempt_logs(partial, experiment_dir, f"attempt{attempt}_")
            continue

        time.sleep(_EAGLE_PROPAGATION_S)

        # ── Read build/run/energy results from Eagle ───────────────────────────
        results = _parse_attempt_results(eagle, results_eagle_dir)
        _save_attempt_logs(results, experiment_dir, f"attempt{attempt}_")

        # ── Interpret compile result ───────────────────────────────────────────
        if not results["build_ok"]:
            tail = _tail5(results["build_stderr"])
            print(f"  Compiling on Polaris...  {_B}{_R}✗  compile error:{_E}")
            if tail:
                print(tail)
            compile_err = results["build_stderr"] or "Compilation failed (no stderr captured)"
            run_stdout  = ""
            continue
        print(f"  Compiling on Polaris...  {_B}{_G}✓  OK{_E}")

        # ── Interpret run result ───────────────────────────────────────────────
        run_stdout = results["run_stdout"] or ""
        if not results["run_ok"]:
            combined = (results["run_stderr"] or "") + "\n" + run_stdout
            tail = _tail5(combined)
            print(f"  Running on Polaris...    {_B}{_R}✗  runtime error:{_E}")
            if tail:
                print(tail)
            compile_err = "execute_error"
            run_err     = results["run_stderr"] or "Exited with non-zero exit code"
            continue
        print(f"  Running on Polaris...    {_B}{_G}✓  OK{_E}")

        # ── XSBench checksum verification ──────────────────────────────────────
        checksum_line = _extract_xsbench_checksum(run_stdout)
        if not _xsbench_checksum_valid(run_stdout):
            print(f"  Checksum...              {_B}{_R}✗  {checksum_line or 'missing'}{_E}")
            snippet = (run_stdout[:1000].strip() or "(no output)") + (
                "\n...(truncated)" if len(run_stdout) > 1000 else ""
            )
            detail = (
                f"XSBench checksum missing or invalid.\n"
                f"Checksum line: {checksum_line!r}\n"
                f"\nActual stdout:\n{snippet}\n"
                f"\nYour CUDA optimization MUST faithfully implement ALL functions from "
                f"the original CUDA source, including print_results(), "
                f"run_event_based_simulation_baseline(), and grid_init_do_not_profile().\n"
                f"Do NOT simplify or stub any function — preserve ALL numerical behavior exactly.\n"
                f"\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL CUDA SOURCE]\n"
            )
            compile_err = "VERIFICATION MISMATCH: " + detail
            continue
        print(f"  Checksum...              {_B}{_G}✓  {checksum_line}{_E}")

        # ── Energy comparison ──────────────────────────────────────────────────
        opt_energy_j = results["energy_joules"]
        opt_power_w  = results["avg_power_w"]
        opt_dur_s    = results["duration_s"]
        opt_samples  = results["num_samples"]

        if opt_energy_j > 0:
            print(
                f"  Optimized energy...      {_B}{opt_energy_j:.1f} J{_E}  "
                f"({opt_power_w:.1f} W avg,  {opt_dur_s:.1f} s,  {opt_samples} samples)"
            )
            if best_energy_j < 0 or opt_energy_j < best_energy_j:
                best_energy_j = opt_energy_j
        else:
            print(f"  Optimized energy...      {_B}(measurement unavailable){_E}")

        if energy_check_enabled and opt_energy_j > 0:
            reduction_pct = (baseline_energy_j - opt_energy_j) / baseline_energy_j * 100.0
            if reduction_pct >= min_energy_gain_pct:
                print(
                    f"  Energy reduction...      {_B}{_G}✓  {reduction_pct:.1f}%  "
                    f"({baseline_energy_j:.1f} J → {opt_energy_j:.1f} J){_E}"
                )
                # Fall through to success block below.
            else:
                direction = "improvement" if reduction_pct > 0 else "regression"
                print(
                    f"  Energy reduction...      {_B}{_R}✗  {reduction_pct:.1f}%  "
                    f"({direction}; need ≥{min_energy_gain_pct:.1f}%){_E}"
                )
                compile_err = (
                    f"ENERGY_NOT_IMPROVED: Current optimized code uses {opt_energy_j:.1f} J "
                    f"versus the baseline {baseline_energy_j:.1f} J "
                    f"({reduction_pct:+.1f}% change, need ≥{min_energy_gain_pct:.1f}% reduction).\n"
                    f"Apply more aggressive CUDA energy optimizations — for example:\n"
                    f"  - Improve global memory access coalescing\n"
                    f"  - Increase kernel occupancy (tune thread block size)\n"
                    f"  - Use shared memory to reduce repeated global memory reads\n"
                    f"  - Reduce warp divergence in conditional branches\n"
                    f"  - Fuse separate kernels to eliminate redundant memory traffic\n"
                    f"  - Use __ldg() for read-only data (L1 texture cache)\n"
                    f"  - Replace double precision with float where precision allows\n"
                    f"CRITICAL: preserve the exact XSBench verification checksum — "
                    f"do NOT alter any numerical computation, loop bounds, or comparison operators."
                )
                continue
        # If energy check is disabled (no baseline), correctness alone is success.

        # ── Success ────────────────────────────────────────────────────────────
        versioned = experiment_dir / f"Main_v1_Trial{trial_number}.cu"
        versioned.write_text(code_block, encoding="utf-8")

        runtime_min    = round((time.time() - start_time) / 60, 2)
        attempts_label = f"attempt {attempt + 1}" if attempt > 0 else "first attempt"

        energy_summary = ""
        if energy_check_enabled and opt_energy_j > 0:
            reduction_pct = (baseline_energy_j - opt_energy_j) / baseline_energy_j * 100.0
            energy_summary = (
                f"  Energy   : {baseline_energy_j:.1f} J → {opt_energy_j:.1f} J  "
                f"({reduction_pct:+.1f}%)\n"
            )

        print(
            f"\n  {_B}{_G}✅  Optimization complete{_E}"
            f"  —  {attempts_label}  |  {runtime_min} min total\n"
            + energy_summary
        )
        print(f"  Optimized code : {versioned}")
        print(f"  Eagle results  : {results_eagle_dir}")

        experiment_data["success"]          = True
        experiment_data["runtime_min"]      = runtime_min
        experiment_data["baseline_energy_j"] = baseline_energy_j
        experiment_data["optimized_energy_j"] = opt_energy_j if opt_energy_j > 0 else -1.0
        if energy_check_enabled and opt_energy_j > 0:
            experiment_data["energy_reduction_pct"] = round(
                (baseline_energy_j - opt_energy_j) / baseline_energy_j * 100.0, 2
            )

        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writerow(experiment_data)

        return True

    print(f"\n{_B}{_R}❌  PIPELINE ERROR: exceeded {_MAX_ATTEMPTS} self-correction attempts.{_E}\n")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Identity / deployment ─────────────────────────────────────────────────
    p.add_argument(
        "--alcf-username",
        default=os.environ.get("USER", ""),
        metavar="USER",
        help="ALCF/Polaris login name for rsync/scp (default: $USER)",
    )
    p.add_argument(
        "--polaris-dir",
        default=_DEFAULT_POLARIS_DIR,
        metavar="PATH",
        help=f"Absolute Eagle path for this project (default: {_DEFAULT_POLARIS_DIR})",
    )
    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip initial rsync of this directory to Eagle (code already deployed)",
    )

    # ── PBS job parameters ────────────────────────────────────────────────────
    p.add_argument("--account", default=_DEFAULT_ACCOUNT, help=f"PBS account (default: {_DEFAULT_ACCOUNT})")
    p.add_argument("--queue",   default=_DEFAULT_QUEUE,   help=f"PBS queue (default: {_DEFAULT_QUEUE})")

    # ── Argo LLM parameters ───────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            f"Argo model name (default: {_DEFAULT_MODEL}).  Examples: "
            "gpt4o, gpt41, claudesonnet4, gpto3mini"
        ),
    )

    # ── Pipeline options ──────────────────────────────────────────────────────
    p.add_argument(
        "--no-self-prompting",
        dest="self_prompting",
        action="store_false",
        help="Disable Stage 0 self-prompting context builder",
    )
    p.add_argument(
        "--min-energy-gain",
        type=float,
        default=_DEFAULT_MIN_GAIN_PCT,
        metavar="PCT",
        help=(
            f"Minimum energy reduction (%) required to declare success "
            f"(default: {_DEFAULT_MIN_GAIN_PCT:.1f}%%). "
            "Use 0 to accept any correctness-preserving optimization."
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

    # ── CUDA toolchain overrides (forwarded to polaris_energy_worker.sh) ──────
    p.add_argument(
        "--cuda-compiler",
        default="",
        metavar="PATH",
        help=(
            "Override CUDA compiler on Polaris (sets CUDA_COMPILER; default: nvcc). "
            "Example: --cuda-compiler /path/to/nvcc"
        ),
    )
    p.add_argument(
        "--cuda-flags",
        default="",
        metavar="FLAGS",
        help=(
            "Override CUDA compilation flags (sets CUDA_FLAGS). "
            "Default: -O3 -arch=sm_80 -std=c++17"
        ),
    )
    p.add_argument(
        "--cuda-ldflags",
        default="",
        metavar="FLAGS",
        help="Extra linker flags (sets CUDA_LDFLAGS; default: empty)",
    )

    p.set_defaults(self_prompting=True)
    args = p.parse_args()

    if not args.alcf_username:
        p.error("--alcf-username is required (or export USER=<your-alcf-login>)")

    if args.debug:
        _llm.DEBUG_MODE = True

    # ── Banner ─────────────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {_B}LASSI-Energy-Polaris v1{_E}  |  XSBench CUDA Energy Optimization")
    print(_SEP)
    print(f"  LLM       : {_B}{args.model}{_E}  via Argo  (apps-dev.inside.anl.gov)")
    print(f"  Polaris   : {_B}{args.queue}{_E} queue  |  account: {args.account}")
    print(f"  Eagle dir : {args.polaris_dir}")
    print(f"  Min gain  : {args.min_energy_gain:.1f}% energy reduction required")
    mode_str = (
        "CUDA energy optimization  self-prompting + self-correction  (Polaris nvcc+energy)"
        if args.self_prompting else
        "CUDA energy optimization  self-correction  (Polaris nvcc+energy, no self-prompting)"
    )
    print(f"  Mode      : {mode_str}")
    if args.debug:
        print(f"  Debug     : ON")
    print(_SEP)

    # ── Load XSBench source ────────────────────────────────────────────────────
    print("\n  Loading XSBench CUDA source...")
    input_code, run_args = _load_source()
    print(f"  Source: {len(input_code.splitlines())} lines  |  run args: {run_args}")

    # Resolve the path to the original Main.cu file (needed for baseline upload).
    search_bases = [_SCRIPT_DIR, _SCRIPT_DIR.parent / "lassi_translate_polaris"]
    original_cu_path: Path | None = None
    for base in search_bases:
        candidate = base / "source_codes" / "Miniapps" / "XSBench" / "cuda" / "Main.cu"
        if candidate.exists():
            original_cu_path = candidate
            break
    assert original_cu_path is not None  # _load_source() would have raised first

    # ── Connect to ALCF ────────────────────────────────────────────────────────
    print("\n  Connecting to ALCF via Globus (for PBS job submission)...")
    print("  (A browser window will open if Globus login is required.)\n")
    polaris, eagle = _amsc_connect()

    # ── SSH ControlMaster ─────────────────────────────────────────────────────
    print("\n  Opening SSH connection to Polaris (one-time auth)...")
    try:
        ssh_ctl = _ssh_master_start(args.alcf_username)
        print("  SSH connection established.")
    except Exception as exc:
        print(f"  {_R}Warning: SSH ControlMaster failed ({exc}); rsync will prompt per call.{_E}")
        ssh_ctl = ""

    # ── Initial rsync ──────────────────────────────────────────────────────────
    if not args.skip_sync:
        print(f"\n  Syncing lassi_energy_polaris/ to Eagle...")
        if not _initial_sync(args.alcf_username, args.polaris_dir, ssh_ctl=ssh_ctl):
            _ssh_master_stop(ssh_ctl, args.alcf_username)
            raise SystemExit("Initial rsync to Polaris failed.  Check SSH key auth.")
        print("  Sync complete.")
    else:
        print("  --skip-sync: skipping initial rsync.")

    # Ensure generated_codes/ hierarchy exists on Eagle.
    _eagle_mkdirs(
        eagle,
        f"{args.polaris_dir}/generated_codes",
        f"{args.polaris_dir}/generated_codes/experiment",
    )

    # ── Experiment setup ───────────────────────────────────────────────────────
    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id    = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_slug      = re.sub(r"[/: ]", "_", args.model)
    experiment_name = f"exp_xsbench_cuda_energy_{model_slug}_NVIDIA__{dt_string}"

    experiment_dir  = _SCRIPT_DIR / "generated_codes" / "experiment" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _llm.LLM_CALL_LOGGER = LLMCallLogger(
        str(experiment_dir / "llm_calls"), truncate_chars=4000
    )

    csv_path   = experiment_dir / "results.csv"
    csv_fields = [
        "timestamp", "app", "model", "gpu_vendor",
        "success", "runtime_min",
        "baseline_energy_j", "optimized_energy_j", "energy_reduction_pct",
    ]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    experiment_data = {
        "timestamp":           dt_string,
        "app":                 "XSBench",
        "model":               args.model,
        "gpu_vendor":          "NVIDIA",   # Polaris A100
        "success":             False,
        "runtime_min":         0.0,
        "baseline_energy_j":   -1.0,
        "optimized_energy_j":  -1.0,
        "energy_reduction_pct": 0.0,
    }

    print(f"\n  Experiment : {experiment_name}")
    print(f"  Local dir  : {experiment_dir}")
    print(f"  Eagle dir  : {args.polaris_dir}/generated_codes/experiment/{experiment_name}")
    print()

    # ── Run pipeline ───────────────────────────────────────────────────────────
    success = run_pipeline(
        model               = args.model,
        alcf_username       = args.alcf_username,
        polaris_dir         = args.polaris_dir,
        account             = args.account,
        queue               = args.queue,
        run_args            = run_args,
        input_code          = input_code,
        original_cu_path    = original_cu_path,
        experiment_name     = experiment_name,
        experiment_dir      = experiment_dir,
        csv_path            = csv_path,
        experiment_data     = experiment_data,
        trial_number        = 1,
        polaris             = polaris,
        eagle               = eagle,
        run_id              = run_id,
        use_self_prompting  = args.self_prompting,
        verifyprompt        = args.verifyprompt,
        verbose             = args.debug,
        cuda_compiler       = args.cuda_compiler,
        cuda_flags          = args.cuda_flags,
        cuda_ldflags        = args.cuda_ldflags,
        ssh_ctl             = ssh_ctl,
        min_energy_gain_pct = args.min_energy_gain,
    )

    if ssh_ctl:
        _ssh_master_stop(ssh_ctl, args.alcf_username)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
