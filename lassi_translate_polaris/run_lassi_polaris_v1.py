#!/usr/bin/env python3
"""LASSI XSBench CUDA→SYCL — LLM calls run locally via Argo,
compile + run executed on Polaris via PBS job submission (amsc_client).

LLM backend
-----------
Uses the ANL Argo gateway (apps-dev.inside.anl.gov/argoapi/...).
Accessible from any machine on the ANL network — no ALCF login needed for LLM.
Set ARGO_USER to your ANL username (default: $USER).
Set ARGO_API_KEY if your deployment requires a Bearer token.

Architecture
------------
  Stage 0 : Self-prompting context builder     — local Argo LLM call
  Stage 1 : Initial CUDA→SYCL translation      — local Argo LLM call
  Stage 2 : Self-correction loop (up to 15 iterations):
              1. Write generated .cpp locally
              2. rsync .cpp to Eagle filesystem on Polaris
              3. Submit PBS job: polaris_worker.sh compiles with icpx + runs XSBench
              4. Poll until job reaches terminal state
              5. Read build / run logs from Eagle via amsc_client filesystem API
              6. Parse: compile error / runtime error / checksum mismatch
              7. If error: generate correction via local Argo LLM call → repeat
              8. If checksum valid: success

Usage
-----
  python run_lassi_polaris_v1.py --alcf-username vchirio [options]

  # Custom model, skip initial sync (code already on Eagle)
  python run_lassi_polaris_v1.py --alcf-username vchirio --model gpt4o --skip-sync

  # Disable self-prompting, show all prompts
  python run_lassi_polaris_v1.py --alcf-username vchirio --no-self-prompting --verifyprompt

  # Custom SYCL compiler / flags on Polaris
  python run_lassi_polaris_v1.py --alcf-username vchirio \\
      --sycl-compiler acpp \\
      --sycl-flags "--acpp-targets=cuda:sm_80 -O3 -std=c++17"

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
_DEFAULT_POLARIS_DIR = "/eagle/EE-ECP/LASSI-EE/lassi_translate_polaris"
_DEFAULT_MODEL       = "gpt55"
_DEFAULT_ACCOUNT     = "EE-ECP"
_DEFAULT_QUEUE       = "debug"
_POLARIS_HOST        = "polaris.alcf.anl.gov"
_MAX_ATTEMPTS        = 15
_JOB_DURATION_S      = 600    # 10 min per compile+run attempt (within debug queue max)
_JOB_WAIT_TIMEOUT_S  = 900    # 15 min total before giving up on a single job
_JOB_POLL_INTERVAL_S = 15     # seconds between status polls
_EAGLE_READ_TIMEOUT  = 60     # seconds for eagle.fs.head()
_EAGLE_PROPAGATION_S = 5      # seconds to wait after job ends before reading Eagle


# ──────────────────────────────────────────────────────────────────────────────
# Source loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_source() -> tuple[str, list[str]]:
    """Load XSBench CUDA source (Main.cu + headers) and exe parameters.

    Headers (*.cuh, *.h) are prepended to Main.cu so the full translation
    context is visible to the LLM in one shot.
    """
    src_dir = _SCRIPT_DIR / "source_codes" / "Miniapps" / "XSBench" / "cuda"
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

    apps_json  = _SCRIPT_DIR / "source_apps.json"
    app_params: list[str] = json.loads(apps_json.read_text())["Miniapps"]["XSBench"]["app_params"]
    return code, app_params


# ──────────────────────────────────────────────────────────────────────────────
# ALCF connection (amsc_client) — for job submission and Eagle file I/O only
# ──────────────────────────────────────────────────────────────────────────────

def _apply_ipv4_patch() -> None:
    """Prefer IPv4 to avoid hangs with broken IPv6 on some ANL workstations.

    The amsc_client autogen code uses no socket timeout, so a stalled IPv6
    connection hangs forever.  Reordering getaddrinfo results to prefer AF_INET
    and adding a 30-second global socket timeout prevents this.
    """
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
        # Walk up looking for a venv that has amsc_client installed.
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
        return   # already exists
    except Exception:
        pass
    try:
        eagle.fs.mkdir(path).wait(timeout=30)
    except Exception:
        pass   # may have been created concurrently — not fatal


def _eagle_mkdirs(eagle, *paths: str) -> None:
    """Create a sequence of directories on Eagle, one level at a time."""
    for path in paths:
        _eagle_mkdir(eagle, path)


def _read_eagle_file(eagle, path: str, max_lines: int = 10000) -> str:
    """Read a file from Eagle; return content as a string.

    Returns an error-prefixed string (not an exception) if the read fails so
    the caller can include it in LLM error prompts.
    """
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
    """Open a persistent SSH ControlMaster connection to Polaris.

    The first call prompts for credentials (password / MFA) once.  All
    subsequent SSH/rsync calls reuse the shared socket without re-prompting.

    Returns the ControlPath socket path (pass it to _ssh_master_stop later).
    """
    import tempfile
    ctl_dir  = Path(tempfile.mkdtemp(prefix="lassi_ssh_"))
    ctl_path = str(ctl_dir / "ctl")
    cmd = [
        "ssh",
        "-o", f"ControlPath={ctl_path}",
        "-o", "ControlMaster=yes",
        "-o", "ControlPersist=yes",   # keep socket alive after this process exits
        "-N",                         # no remote command — just open the tunnel
        "-f",                         # go to background after auth succeeds
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
            "--exclude=generated_codes/",  # managed separately; --delete would wipe Eagle results
            "--exclude=.git/",
        ]
    cmd += [local_src, dst]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # --mkpath was added in rsync 3.2.3; retry without it on older systems
        cmd2 = [c for c in cmd if c != "--mkpath"]
        result = subprocess.run(cmd2, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  {_R}rsync failed (rc={result.returncode}):{_E}")
        if result.stderr.strip():
            print(f"    {result.stderr.strip()[:400]}")
        return False
    return True


def _initial_sync(alcf_username: str, polaris_dir: str, ssh_ctl: str = "") -> bool:
    """rsync the entire lassi_translate_polaris/ directory to Eagle."""
    print(f"  {_SCRIPT_DIR}/  →  {alcf_username}@polaris:{polaris_dir}/")
    return _rsync(str(_SCRIPT_DIR), polaris_dir, alcf_username, is_dir=True, ssh_ctl=ssh_ctl)


def _sync_cpp_to_eagle(
    local_cpp: Path,
    polaris_dir: str,
    exp_name: str,
    cpp_filename: str,
    alcf_username: str,
    ssh_ctl: str = "",
) -> bool:
    """rsync a single generated .cpp file to its experiment directory on Eagle."""
    remote = f"{polaris_dir}/generated_codes/experiment/{exp_name}/{cpp_filename}"
    return _rsync(str(local_cpp), remote, alcf_username, ssh_ctl=ssh_ctl)


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
    cpp_eagle_path: str,
    results_eagle_dir: str,
    run_args: list[str],
    job_name: str,
    account: str,
    queue: str,
    sycl_compiler: str = "",
    sycl_flags: str = "",
    sycl_ldflags: str = "",
):
    """Submit a PBS job that runs polaris_worker.sh on a Polaris compute node.

    polaris_worker.sh will:
      1. Load the oneAPI module (icpx with CUDA backend for A100 / sm_80)
      2. Compile <cpp_eagle_path> to <results_eagle_dir>/Main
      3. Run Main with <run_args> and capture stdout / stderr
      4. Write build_rc.txt, build_stdout/stderr.log, run_rc.txt,
         run_stdout/stderr.log to <results_eagle_dir>

    PBS stdout/stderr → <results_eagle_dir>/pbs_{stdout,stderr}.log
    """
    worker           = f"{polaris_dir}/polaris_worker.sh"
    run_args_quoted  = " ".join(shlex.quote(a) for a in run_args)

    env_parts: list[str] = []
    if sycl_compiler:
        env_parts.append(f"SYCL_COMPILER={shlex.quote(sycl_compiler)}")
    if sycl_flags:
        env_parts.append(f"SYCL_FLAGS={shlex.quote(sycl_flags)}")
    if sycl_ldflags:
        env_parts.append(f"SYCL_LDFLAGS={shlex.quote(sycl_ldflags)}")
    env_prefix = (" ".join(env_parts) + " ") if env_parts else ""

    cmd = (
        f"{env_prefix}bash {shlex.quote(worker)}"
        f" {shlex.quote(cpp_eagle_path)}"
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

    When a PBS job finishes and leaves the scheduler queue, the IRI API
    returns HTTP 400 / "not found".  That is treated as "completed" here.

    Returns the final state string (e.g. 'completed', 'failed').
    """
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            _ = job.status      # forces an API refresh; raises on 400 when job is gone
            if job.is_terminal:
                return job.state
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "400" in msg:
                return "completed"  # job left PBS scheduler queue — normal completion
            # transient network error; keep polling
        time.sleep(_JOB_POLL_INTERVAL_S)
    # timeout reached — return whatever state we last saw
    try:
        return job.state
    except Exception:
        return "unknown"


def _parse_attempt_results(eagle, results_eagle_dir: str) -> dict:
    """Read compile/run result files from Eagle and return a parsed dict."""
    def _rc(fname: str) -> int:
        raw = _read_eagle_file(eagle, f"{results_eagle_dir}/{fname}", max_lines=5).strip()
        return int(raw) if re.fullmatch(r"-?\d+", raw) else -1

    return {
        "build_ok":     _rc("build_rc.txt") == 0,
        "run_ok":       _rc("run_rc.txt")   == 0,
        "build_stdout": _read_eagle_file(eagle, f"{results_eagle_dir}/build_stdout.log"),
        "build_stderr": _read_eagle_file(eagle, f"{results_eagle_dir}/build_stderr.log"),
        "run_stdout":   _read_eagle_file(eagle, f"{results_eagle_dir}/run_stdout.log"),
        "run_stderr":   _read_eagle_file(eagle, f"{results_eagle_dir}/run_stderr.log"),
    }


def _save_attempt_logs(results: dict, experiment_dir: Path, attempt: int) -> None:
    """Mirror Eagle result files to the local experiment directory."""
    prefix = f"attempt{attempt}_"
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
    sycl_compiler: str = "",
    sycl_flags: str = "",
    sycl_ldflags: str = "",
    ssh_ctl: str = "",
) -> bool:
    """Run the full three-stage pipeline.

    Stage 0 (optional): self-prompting — two local Argo calls to build context
    Stage 1:            initial translation — one local Argo call
    Stage 2:            self-correction loop — per-attempt: local Argo call
                        + Polaris PBS job (compile + run) + Eagle result read

    Returns True on success (valid XSBench checksum verified on Polaris).
    """
    prompts    = PromptDictionary()
    start_time = time.time()

    # ── STAGE 0: Self-prompting (local Argo LLM calls) ────────────────────────
    sycl_context_summary    = ""
    source_code_description = ""

    if use_self_prompting:
        print(f"\n{_B}----- STAGE 0: SELF-PROMPTING  (local Argo) -----{_E}\n")

        try:
            sycl_context_raw = prompts.get_codeknowledge("SYCL_guide")
        except Exception as exc:
            print(f"  {_B}WARN:{_E} Could not load SYCL guide: {exc}.  Skipping.")
            sycl_context_raw = ""

        if sycl_context_raw:
            summarize_prompt = (
                prompts.get_selfprompt("summarize_context")
                + "\n\n[BEGIN SYCL REFERENCE GUIDE]\n"
                + sycl_context_raw
                + "\n[END SYCL REFERENCE GUIDE]"
            )
            print("  Step 1/2: Summarizing SYCL reference context...")
            sycl_context_summary = _llm.generate_text(
                model,
                prompts.get_system_prompt("selfprompt"),
                summarize_prompt,
                intent="context summarization",
                verifyprompt=verifyprompt,
                label="[LLM] Summarizing SYCL context knowledge",
            )

        describe_prompt = (
            prompts.get_selfprompt("describe_source")
            + "\n\n[BEGIN CUDA SOURCE CODE]\n"
            + input_code
            + "\n[END CUDA SOURCE CODE]"
        )
        print("  Step 2/2: Generating source code description...")
        source_code_description = _llm.generate_text(
            model,
            prompts.get_system_prompt("selfprompt"),
            describe_prompt,
            intent="source code description",
            verifyprompt=verifyprompt,
            label="[LLM] Describing source code",
        )

        print(f"\n  {_B}{_G}✓{_E}  Self-prompting complete.  Context incorporated into Stage 1 prompt.\n")

    # Build the Stage 1 user prompt with optional self-prompting preamble.
    system_prompt    = prompts.get_system_prompt("CUDA_to_SYCL")
    translate_prompt = prompts.get_codetranslate_prompt("CUDA_to_SYCL")

    if use_self_prompting and (sycl_context_summary or source_code_description):
        preamble = ""
        if sycl_context_summary:
            preamble += (
                "[SYCL CONTEXT KNOWLEDGE]\n"
                + sycl_context_summary
                + "\n[/SYCL CONTEXT KNOWLEDGE]\n\n"
            )
        if source_code_description:
            preamble += (
                "[YOUR ANALYSIS OF THE SOURCE CODE TO TRANSLATE]\n"
                + source_code_description
                + "\n[/YOUR ANALYSIS OF THE SOURCE CODE TO TRANSLATE]\n\n"
            )
        think_prefix = "Think carefully, using your analysis above.  Now, " if source_code_description else ""
        translate_user_prompt = (
            preamble
            + think_prefix
            + translate_prompt
            + "\n\n[START OF CODE TO REFACTOR]\n"
            + input_code
            + "\n[/END OF CODE TO REFACTOR]"
        )
    else:
        translate_user_prompt = (
            translate_prompt
            + "\n\n[START OF CODE TO REFACTOR]\n"
            + input_code
            + "\n[/END OF CODE TO REFACTOR]"
        )

    # Ensure the experiment directory exists on Eagle before first iteration.
    _eagle_mkdirs(
        eagle,
        f"{polaris_dir}/generated_codes",
        f"{polaris_dir}/generated_codes/experiment",
        f"{polaris_dir}/generated_codes/experiment/{experiment_name}",
    )

    # ── STAGES 1 + 2: generate → (Polaris compile+run → correct) loop ─────────
    compile_err = ""
    run_err     = ""
    run_stdout  = ""
    code_block  = ""

    for attempt in range(_MAX_ATTEMPTS + 1):

        # ── Local Argo LLM call ────────────────────────────────────────────────
        if attempt == 0:
            print(f"\n{_B}----- STAGE 1: CUDA→SYCL TRANSLATION  (local Argo) -----{_E}\n")
            code_block = _llm.generate_code(
                model,
                system_prompt,
                translate_user_prompt,
                verifyprompt,
                label="[LLM] Generating initial SYCL translation",
            )
        else:
            if attempt == 1:
                print(f"\n{_B}----- STAGE 2: SELF-CORRECTION  (local Argo + Polaris PBS) -----{_E}\n")

            # Determine error type and build the correction prompt.
            if compile_err == "execute_error":
                err_type    = "runtime error"
                error_msg   = run_err
                if run_stdout.strip():
                    error_msg += "\n\nProgram stdout before crash:\n" + run_stdout
                error_msg  += "\n\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n" + input_code + "\n[/ORIGINAL CUDA SOURCE]\n"
                error_intro = "\n -- Runtime error: "
            elif compile_err.startswith("VERIFICATION MISMATCH"):
                err_type    = "checksum mismatch"
                error_msg   = compile_err
                error_intro = "\n -- Verification mismatch: "
            else:
                err_type    = "compile error"
                error_msg   = compile_err
                error_intro = "\n -- Compile error: "

            print(f"  {_B}Attempt {attempt}/{_MAX_ATTEMPTS}{_E} — correcting {err_type}")

            error_prompt = (
                prompts.get_code_error_correction_prompt("CUDA_to_SYCL_error")
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
                label=f"[LLM] Corrected SYCL code (attempt {attempt})",
            )

        # ── Write generated code locally ───────────────────────────────────────
        cpp_name  = f"Main_attempt{attempt}.cpp"
        local_cpp = experiment_dir / cpp_name
        local_cpp.write_text(code_block, encoding="utf-8")

        # ── Upload .cpp to Eagle ───────────────────────────────────────────────
        print(f"  Uploading {cpp_name} to Eagle...", end="  ", flush=True)
        ok = _sync_cpp_to_eagle(
            local_cpp     = local_cpp,
            polaris_dir   = polaris_dir,
            exp_name      = experiment_name,
            cpp_filename  = cpp_name,
            alcf_username = alcf_username,
            ssh_ctl       = ssh_ctl,
        )
        if not ok:
            print(f"{_R}FAILED{_E}")
            compile_err = f"rsync failed — could not upload {cpp_name} to Eagle"
            run_stdout  = ""
            continue
        print(f"{_G}OK{_E}")

        # Paths on Eagle (absolute, as seen from both Polaris nodes and the Eagle API)
        cpp_eagle_path    = f"{polaris_dir}/generated_codes/experiment/{experiment_name}/{cpp_name}"
        results_eagle_dir = f"{polaris_dir}/generated_codes/experiment/{experiment_name}/attempt{attempt}_results"

        # Create the results dir on Eagle so PBS can write stdout/stderr there.
        _eagle_mkdir(eagle, results_eagle_dir)

        # ── Submit PBS compile+run job ─────────────────────────────────────────
        job_name = f"lassi-c{attempt}-{run_id}"
        print(f"  Submitting PBS job ({job_name})...", end="  ", flush=True)
        try:
            job = _submit_compile_run_job(
                polaris           = polaris,
                polaris_dir       = polaris_dir,
                cpp_eagle_path    = cpp_eagle_path,
                results_eagle_dir = results_eagle_dir,
                run_args          = run_args,
                job_name          = job_name,
                account           = account,
                queue             = queue,
                sycl_compiler     = sycl_compiler,
                sycl_flags        = sycl_flags,
                sycl_ldflags      = sycl_ldflags,
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
            # Best-effort read of whatever the worker managed to write before dying
            time.sleep(_EAGLE_PROPAGATION_S)
            partial = _parse_attempt_results(eagle, results_eagle_dir)
            _save_attempt_logs(partial, experiment_dir, attempt)
            continue

        # Small delay so Eagle propagates files written in the closing moments of the job.
        time.sleep(_EAGLE_PROPAGATION_S)

        # ── Read build/run results from Eagle ──────────────────────────────────
        results = _parse_attempt_results(eagle, results_eagle_dir)
        _save_attempt_logs(results, experiment_dir, attempt)

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
                f"\nYour SYCL translation MUST faithfully implement ALL functions from "
                f"the original CUDA source, including print_results(), "
                f"run_event_based_simulation_baseline(), and grid_init_do_not_profile().\n"
                f"Do NOT generate a simplified stub — translate the complete source.\n"
                f"\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL CUDA SOURCE]\n"
            )
            compile_err = "VERIFICATION MISMATCH: " + detail
            continue
        print(f"  Checksum...              {_B}{_G}✓  {checksum_line}{_E}")

        # ── Success ────────────────────────────────────────────────────────────
        versioned = experiment_dir / f"Main_v1_Trial{trial_number}.cpp"
        versioned.write_text(code_block, encoding="utf-8")

        runtime_min    = round((time.time() - start_time) / 60, 2)
        attempts_label = f"attempt {attempt + 1}" if attempt > 0 else "first attempt"
        print(
            f"\n  {_B}{_G}✅  Translation complete{_E}"
            f"  —  {attempts_label}  |  {runtime_min} min total\n"
        )
        print(f"  Successful code : {versioned}")
        print(f"  Eagle results   : {results_eagle_dir}")

        experiment_data["success"]     = True
        experiment_data["runtime_min"] = runtime_min
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

    # ── SYCL toolchain overrides (forwarded to polaris_worker.sh) ─────────────
    p.add_argument(
        "--sycl-compiler",
        default="",
        metavar="PATH",
        help=(
            "Override SYCL compiler on Polaris (sets SYCL_COMPILER; default: icpx). "
            "Example: --sycl-compiler acpp"
        ),
    )
    p.add_argument(
        "--sycl-flags",
        default="",
        metavar="FLAGS",
        help=(
            "Override SYCL compilation flags (sets SYCL_FLAGS). "
            "Default: -O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda "
            "-Xsycl-target-backend --cuda-gpu-arch=sm_80"
        ),
    )
    p.add_argument(
        "--sycl-ldflags",
        default="",
        metavar="FLAGS",
        help="Extra linker flags (sets SYCL_LDFLAGS; default: empty)",
    )

    p.set_defaults(self_prompting=True)
    args = p.parse_args()

    if not args.alcf_username:
        p.error("--alcf-username is required (or export USER=<your-alcf-login>)")

    if args.debug:
        _llm.DEBUG_MODE = True

    # ── Banner ─────────────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {_B}LASSI-Polaris v1{_E}  |  XSBench CUDA→SYCL")
    print(_SEP)
    print(f"  LLM       : {_B}{args.model}{_E}  via Argo  (apps-dev.inside.anl.gov)")
    print(f"  Polaris   : {_B}{args.queue}{_E} queue  |  account: {args.account}")
    print(f"  Eagle dir : {args.polaris_dir}")
    mode_str = (
        "CUDA→SYCL  self-prompting + self-correction  (Polaris compile/run)"
        if args.self_prompting else
        "CUDA→SYCL  self-correction  (Polaris compile/run, no self-prompting)"
    )
    print(f"  Mode      : {mode_str}")
    if args.debug:
        print(f"  Debug     : ON")
    print(_SEP)

    # ── Load XSBench source ────────────────────────────────────────────────────
    print("\n  Loading XSBench CUDA source...")
    input_code, run_args = _load_source()
    print(f"  Source: {len(input_code.splitlines())} lines  |  run args: {run_args}")

    # ── Connect to ALCF (for PBS + Eagle only — LLM uses Argo separately) ──────
    print("\n  Connecting to ALCF via Globus (for PBS job submission)...")
    print("  (A browser window will open if Globus login is required.)\n")
    polaris, eagle = _amsc_connect()

    # ── SSH ControlMaster — one auth for all rsync calls ──────────────────────
    print("\n  Opening SSH connection to Polaris (one-time auth)...")
    try:
        ssh_ctl = _ssh_master_start(args.alcf_username)
        print("  SSH connection established.")
    except Exception as exc:
        print(f"  {_R}Warning: SSH ControlMaster failed ({exc}); rsync will prompt per call.{_E}")
        ssh_ctl = ""

    # ── Initial rsync ──────────────────────────────────────────────────────────
    if not args.skip_sync:
        print(f"\n  Syncing lassi_translate_polaris/ to Eagle...")
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
    experiment_name = f"exp_xsbench_cuda_to_sycl_{model_slug}_NVIDIA__{dt_string}"

    experiment_dir  = _SCRIPT_DIR / "generated_codes" / "experiment" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    _llm.LLM_CALL_LOGGER = LLMCallLogger(
        str(experiment_dir / "llm_calls"), truncate_chars=4000
    )

    csv_path   = experiment_dir / "results.csv"
    csv_fields = ["timestamp", "app", "model", "gpu_vendor", "success", "runtime_min"]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    experiment_data = {
        "timestamp":   dt_string,
        "app":         "XSBench",
        "model":       args.model,
        "gpu_vendor":  "NVIDIA",   # Polaris A100
        "success":     False,
        "runtime_min": 0.0,
    }

    print(f"\n  Experiment : {experiment_name}")
    print(f"  Local dir  : {experiment_dir}")
    print(f"  Eagle dir  : {args.polaris_dir}/generated_codes/experiment/{experiment_name}")
    print()

    # ── Run pipeline ───────────────────────────────────────────────────────────
    success = run_pipeline(
        model             = args.model,
        alcf_username     = args.alcf_username,
        polaris_dir       = args.polaris_dir,
        account           = args.account,
        queue             = args.queue,
        run_args          = run_args,
        input_code        = input_code,
        experiment_name   = experiment_name,
        experiment_dir    = experiment_dir,
        csv_path          = csv_path,
        experiment_data   = experiment_data,
        trial_number      = 1,
        polaris           = polaris,
        eagle             = eagle,
        run_id            = run_id,
        use_self_prompting = args.self_prompting,
        verifyprompt      = args.verifyprompt,
        verbose           = args.debug,
        sycl_compiler     = args.sycl_compiler,
        sycl_flags        = args.sycl_flags,
        sycl_ldflags      = args.sycl_ldflags,
        ssh_ctl           = ssh_ctl,
    )

    if ssh_ctl:
        _ssh_master_stop(ssh_ctl, args.alcf_username)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
