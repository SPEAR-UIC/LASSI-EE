#!/usr/bin/env python3
"""LASSI-EE — CUDA→SYCL translation + GPU energy optimization on Polaris/Aurora.

CLI entry point.  main() dispatches to one of three modes:
    (default) remote   — workstation orchestrates; PBS persistent worker; Argo LLM
    --on-resource      — runs inside qsub -I on the compute node; ALCF Inference
    --submit           — workstation submits one PBS job that runs --on-resource;
                         streams log back via amsc.fs.view; ALCF Inference
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lassi.llm_client as _llm
from lassi.alcf         import amsc_connect
from lassi.format_colors import B, G, R, E
from lassi.hpc_jobs     import JOB_DURATION_S, PersistentWorker, build_runner
from lassi.hpc_session  import HPCSession
from lassi.local_runner  import build_local_runner
from lassi.local_session import LocalSession
from lassi.pipeline     import (print_translate_summary, run_baseline,
                                run_pipeline, run_test_run)
from lassi.resources    import RESOURCES, can_run
from lassi.source_loader import find_apps_json, load_cuda_source

_SEP = "=" * 72
_SCRIPT_DIR = Path(__file__).resolve().parent

_DEFAULT_ARGO_MODEL      = "gpt41"
_DEFAULT_INFERENCE_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
_DEFAULT_ACCOUNT         = "EE-ECP"
_DEFAULT_QUEUE           = "debug"
_DEFAULT_MIN_GAIN_PCT    = 1.0
_COMPUTE_HTTPS_PROXY     = "http://proxy.alcf.anl.gov:3128"

_AUTH_CLIENT_ID  = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"
_APP_NAME        = "inference_app"


def _new_exp_data(dt, args, gpu_vendor, *, energy=False):
    """Pre-declare every column results.csv will contain so the header matches
    rows written by pipeline.py.  Numeric defaults are sentinels (-1.0 / 0)."""
    d = {"timestamp": dt, "app": args.app_name, "model": args.model,
         "gpu_vendor": gpu_vendor, "success": False,
         "attempts": 0, "runtime_min": 0.0,
         "baseline_runtime_s": -1.0,  "baseline_throughput":   "",
         "translation_runtime_s": -1.0, "translation_throughput": ""}
    if energy:
        d.update({"baseline_energy_j": -1.0, "optimized_energy_j": -1.0,
                  "energy_reduction_pct": 0.0})
    return d


# ── Config + CLI ─────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    """Execute lassi_config.py and return its `config` dict (legacy `model` → `argo_model`)."""
    if not path.exists(): return {}
    try:
        ns: dict = {}; exec(path.read_text(), ns)
        cfg = dict(ns.get("config", {}))
    except Exception as exc:
        print(f"Warning: could not load {path}: {exc}", file=sys.stderr); return {}
    if (legacy := cfg.pop("model", None)) and "argo_model" not in cfg:
        cfg["argo_model"] = legacy
    return cfg


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    """Build the argparse parser; `defaults` (from lassi_config.py) seeds flag defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")

    # Resource + task
    p.add_argument("--resource", default=None, choices=["polaris", "aurora"])
    p.add_argument("--task", default=None, choices=["translate", "energy"])
    p.add_argument("--app-category", default="Miniapps")
    p.add_argument("--app-name", default="XSBench")

    # Identity / remote
    p.add_argument("--alcf-username", default=os.environ.get("USER", ""))
    p.add_argument("--remote-dir", default="")
    p.add_argument("--skip-sync", action="store_true")
    # Override generated_codes/ location (default = next to run_lassi.py).
    p.add_argument("--output-dir", default="")

    # PBS
    p.add_argument("--account",  default=_DEFAULT_ACCOUNT)
    p.add_argument("--queue",    default=_DEFAULT_QUEUE)
    p.add_argument("--walltime", type=int, default=JOB_DURATION_S)
    p.add_argument("--no-persistent-worker", dest="persistent_worker",
                   action="store_false", default=True)

    # LLM
    p.add_argument("--model", default=None,
                   help="alias for --argo-model (remote) or --inference-model (--on-resource/--submit)")
    p.add_argument("--argo-model",      default=_DEFAULT_ARGO_MODEL)
    p.add_argument("--inference-model", default=_DEFAULT_INFERENCE_MODEL)

    # Modes
    p.add_argument("--on-resource", action="store_true", default=False)
    p.add_argument("--submit",      action="store_true", default=False)
    p.add_argument("--inference-cluster", default="sophia", choices=["sophia", "metis"])
    p.add_argument("--remote-conda-env", default="base")
    p.add_argument("--remote-python",    default="")
    # Auto-fetch the experiment dir after --submit completes (default on).
    p.add_argument("--no-fetch-results", dest="fetch_results",
                   action="store_false", default=True)
    p.add_argument("--fetch-dest", default="")

    # Pipeline
    p.add_argument("--no-self-prompting", dest="self_prompting", action="store_false")
    # LLM-as-judge equivalence check (default from lassi_config.py "use_judge").
    p.add_argument("--judge", dest="use_judge", action=argparse.BooleanOptionalAction)
    # Temperature ratchet — bump on stuck errors (default from lassi_config.py
    # "ratchet_temperature").  --no-temp-ratchet holds the call temp at 0.2.
    p.add_argument("--temp-ratchet", dest="ratchet_temperature",
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--min-energy-gain", type=float, default=_DEFAULT_MIN_GAIN_PCT)
    p.add_argument("--verifyprompt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug", action="store_true", default=False)

    # Compiler env
    p.add_argument("--sycl-compiler", default="")
    p.add_argument("--sycl-flags",    default="")
    p.add_argument("--sycl-ldflags",  default="")
    p.add_argument("--cuda-compiler", default="")
    p.add_argument("--cuda-flags",    default="")
    p.add_argument("--cuda-ldflags",  default="")

    p.set_defaults(self_prompting=True, use_judge=True, ratchet_temperature=True)
    if defaults: p.set_defaults(**defaults)
    return p


def _load_source(args):
    """Load the CUDA source — returns (code, run_args, src_path, verification)."""
    print(f"\n  Loading CUDA source ({args.app_category}/{args.app_name})...")
    code, run_args, src_path, ver = load_cuda_source(_SCRIPT_DIR, args.app_category, args.app_name)
    print(f"  Source: {len(code.splitlines())} lines  |  run args: {run_args}")
    return code, run_args, src_path, ver


def _compiler_envs(args):
    """Bundle --sycl-* / --cuda-* CLI overrides into env-var dicts for the worker."""
    sycl = {k: v for k, v in {"SYCL_COMPILER": args.sycl_compiler,
                              "SYCL_FLAGS":    args.sycl_flags,
                              "SYCL_LDFLAGS":  args.sycl_ldflags}.items() if v}
    cuda = {k: v for k, v in {"CUDA_COMPILER": args.cuda_compiler,
                              "CUDA_FLAGS":    args.cuda_flags,
                              "CUDA_LDFLAGS":  args.cuda_ldflags}.items() if v}
    return sycl, cuda


def _banner(args, resource_cfg, remote_dir: str,
            llm_endpoint: str = "Argo  (apps.inside.anl.gov)"):
    """Print the per-run header (resource, task, LLM, queue, paths)."""
    print(_SEP)
    print(f"  {B}LASSI-Jobs{E}  |  resource={args.resource}  task={args.task}")
    print(_SEP)
    print(f"  LLM        : {B}{args.model}{E}  via {llm_endpoint}")
    print(f"  Resource   : {B}{args.resource}{E} ({resource_cfg.gpu_vendor} GPU)")
    print(f"  Queue      : {B}{args.queue}{E}  |  account: {args.account}")
    print(f"  Remote dir : {remote_dir}")
    print(f"  Judge      : {B}{'enabled' if args.use_judge else 'disabled'}{E}"
          f"  (LLM-as-judge functional-equivalence check)")
    print(f"  Temp ratchet: {B}{'on' if args.ratchet_temperature else 'off'}{E}"
          f"  (auto-bump LLM temp on stuck errors)")
    if args.task == "energy":
        print(f"  Min gain   : {args.min_energy_gain:.1f}%")
    print(_SEP)


# ── Auth helpers (--submit path) ─────────────────────────────────────────────
# Globus auth runs on the workstation (browser); tokens.json is rsynced to
# the remote home dir.  Compute nodes read cached tokens with stdlib JSON only.

def _local_tokens_dir() -> Path:
    """Canonical path where the vendored auth helper caches Globus tokens."""
    return Path.home() / ".globus" / "app" / _AUTH_CLIENT_ID / _APP_NAME


def _ensure_local_inference_auth():
    """Run the auth helper's browser flow on first ever use; no-op afterwards."""
    tokens = _local_tokens_dir() / "tokens.json"
    if tokens.exists(): return
    print(f"  {B}First-time ALCF Inference setup — browser link will print.{E}\n")
    rc = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "inference_auth_token.py"), "authenticate"]).returncode
    if rc != 0 or not tokens.exists():
        raise SystemExit(f"  {R}Auth failed.{E}")
    print(f"  {G}✓{E}  Authenticated.\n")


def _refresh_local_inference_token():
    """Touch get_access_token so tokens.json gets refreshed before we sync it."""
    try:
        from lassi.alcf_inference import get_access_token
        get_access_token(_SCRIPT_DIR)
    except SystemExit: raise
    except Exception: pass


def _force_reauth():
    """Interactive re-auth — needed every 30 days per ALCF high-assurance policy."""
    print(f"  {B}Running `authenticate --force`:{E}\n")
    rc = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "inference_auth_token.py"),
         "authenticate", "--force"]).returncode
    if rc != 0:
        raise SystemExit(f"  {R}Re-auth failed.  Logout first at https://app.globus.org/logout{E}")


def _verify_inference_token():
    """Pre-flight GET to the ALCF API to confirm the token works.  Auto-runs
    `authenticate --force` if it detects the 30-day high-assurance timeout."""
    from lassi.alcf_inference import _read_cached_access_token, verify_token
    print(f"  Verifying ALCF Inference token...", end="  ", flush=True)
    tok = _read_cached_access_token()
    if not tok:
        print(f"{R}no cached token{E}")
        raise SystemExit("Run: python lassi_translate/inference_auth_token.py authenticate")
    ok, err = verify_token(tok)
    if ok: print(f"{G}OK{E}"); return
    if "high-assurance" in err.lower() or "internal policies" in err.lower():
        print(f"{R}30-day re-auth required{E}")
        print(f"  Logout first: https://app.globus.org/logout")
        _force_reauth()
        tok = _read_cached_access_token()
        ok, err = verify_token(tok) if tok else (False, "no token after re-auth")
        if not ok: raise SystemExit(f"  {R}Token still rejected: {err}{E}")
        print(f"  {G}✓{E}  Re-auth verified.")
        return
    raise SystemExit(f"  {R}Token check failed:{E}  {err}")


def _sync_globus_tokens_to_remote(session, user):
    """rsync local tokens.json → remote ~/.globus (rsync --mkpath creates dirs;
    amsc fs can't reach /home/* since Eagle is /eagle-or-/lus/eagle only)."""
    local = _local_tokens_dir()
    if not local.is_dir(): raise SystemExit("local tokens missing")
    remote = f"/home/{user}/.globus/app/{_AUTH_CLIENT_ID}/{_APP_NAME}"
    print(f"  Syncing Globus tokens to {remote}...", end="  ", flush=True)
    if not session.rsync(str(local), remote, is_dir=True):
        raise SystemExit("token sync failed")
    print(f"{G}OK{E}")


# ── --submit: build + tail a PBS job that runs --on-resource ────────────────

def _build_submit_pbs_script(*, args, resource_cfg, remote_dir, remote_expdir_marker,
                              remote_log_path):
    """Bash body for the amsc-submitted PBS job: redirect stdout/stderr to
    `remote_log_path` on Eagle (bypassing PBS's local-disk spool), set up
    conda (or use --remote-python), then exec `run_lassi.py --on-resource`."""
    inner = ["--on-resource", "--task", args.task, "--resource", args.resource,
             "--app-category", args.app_category, "--app-name", args.app_name,
             "--model", args.model, "--inference-cluster", args.inference_cluster,
             "--min-energy-gain", str(args.min_energy_gain)]
    if not args.self_prompting:      inner += ["--no-self-prompting"]
    if not args.use_judge:            inner += ["--no-judge"]
    if not args.ratchet_temperature:  inner += ["--no-temp-ratchet"]
    if args.verifyprompt:       inner += ["--verifyprompt"]
    if args.debug:              inner += ["--debug"]
    if args.output_dir:         inner += ["--output-dir", args.output_dir]
    for flag, val in [("--sycl-compiler", args.sycl_compiler),
                      ("--sycl-flags",    args.sycl_flags),
                      ("--sycl-ldflags",  args.sycl_ldflags),
                      ("--cuda-compiler", args.cuda_compiler),
                      ("--cuda-flags",    args.cuda_flags),
                      ("--cuda-ldflags",  args.cuda_ldflags)]:
        if val: inner += [flag, val]
    inner_cmd = " ".join(shlex.quote(a) for a in inner)

    if args.remote_python:
        py = shlex.quote(args.remote_python)
        env = (f"if [ ! -x {py} ]; then echo 'ERROR: {args.remote_python} missing' >&2; exit 127; fi\n"
               f"PYTHON={py}\n")
    else:
        env = (
            "if ! command -v module &>/dev/null; then\n"
            "    for _i in /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh \\\n"
            "                 /usr/share/lmod/lmod/init/bash; do\n"
            "        [ -f \"$_i\" ] && source \"$_i\" && break\n"
            "    done\n"
            "fi\n"
            "if command -v module &>/dev/null; then\n"
            "    module use /soft/modulefiles 2>/dev/null || true\n"
            "    module load conda 2>&1 | sed 's/^/  [module] /' || true\n"
            "fi\n"
            "if ! command -v conda &>/dev/null; then\n"
            "    for _cs in /soft/applications/conda/*/mconda3/etc/profile.d/conda.sh; do\n"
            "        [ -f \"$_cs\" ] && source \"$_cs\" && break\n"
            "    done\n"
            "fi\n"
            "if ! command -v conda &>/dev/null; then\n"
            "    echo 'ERROR: conda not found. Use --remote-python /abs/path/to/python3' >&2\n"
            "    exit 127\n"
            "fi\n"
            "eval \"$(conda shell.bash hook 2>/dev/null)\" || true\n"
            f"conda activate {shlex.quote(args.remote_conda_env)} 2>&1 | sed 's/^/  [conda] /' || true\n"
            "PYTHON=python3\n"
        )

    return (
        # Bypass PBS spooling: write stdout+stderr directly to Eagle so the
        # workstation can poll the log via amsc.fs.view while the job runs.
        f"exec > {shlex.quote(remote_log_path)} 2>&1\n"
        "export PYTHONUNBUFFERED=1\n"
        "export LASSI_FORCE_TTY=1\n"
        f"export LASSI_EXPDIR_FILE={shlex.quote(remote_expdir_marker)}\n\n"
        "echo '=== lassi-translate (--submit) ==='\n"
        "echo \"Hostname: $(hostname)\"\n"
        "echo \"Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')\"\n"
        f"echo \"Cwd:      {remote_dir}\"\necho ''\n\n"
        + env +
        "echo \"python: $($PYTHON -c 'import sys; print(sys.executable)' 2>/dev/null || echo NOT FOUND)\"\n"
        "echo ''\n\n"
        f"cd {shlex.quote(remote_dir)}\n"
        f"$PYTHON run_lassi.py {inner_cmd}\n"
        "_rc=$?\necho ''\n"
        "echo \"=== exit code: $_rc ===\"\nexit $_rc\n"
    )


def _tail_until_ctrl_c(session, job, log_path, host, user):
    """Stream PBS log via amsc fs.view; heartbeat while empty.  Returns True
    on natural job completion, False on Ctrl-C detach.  Takes the live Job
    object — ALCF's compute.jobs([id]) lookup is 501."""
    job_id = job.id
    print(f"  {B}Waiting for PBS job; log will stream once content appears.{E}")
    print(f"  {B}Ctrl-C detaches without killing the job.{E}\n")
    print("─" * 72)
    offset = 0
    t0 = time.time()
    streaming = False

    def _drain() -> int:
        """Read new log bytes via fs.view; return bytes appended this tick."""
        nonlocal offset, streaming
        try: chunk = session.read_bytes(log_path, offset=offset, size=1_000_000)
        except Exception: return 0
        if not chunk: return 0
        sys.stdout.write(chunk); sys.stdout.flush()
        offset += len(chunk.encode("utf-8", errors="replace"))
        streaming = True
        return len(chunk)

    try:
        while True:
            _drain()
            # NotFound/404 = PBS purged the job entry → treat as completed.
            try: state = job.refresh()
            except Exception as exc:
                m = str(exc).lower()
                state = "completed" if ("not found" in m or "404" in m) else "?"

            # Heartbeat only while the log is empty.
            if not streaming:
                elapsed = int(time.time() - t0)
                msg = ("queued" if state == "queued" else
                       "active, waiting for log..." if state == "active" else state)
                print(f"    [{elapsed:>4}s] PBS state={msg}", flush=True)

            if state in ("completed", "failed", "cancelled", "canceled"):
                time.sleep(2); _drain()
                print(f"\n{'─' * 72}")
                col = G if state == "completed" else R
                print(f"  {col}PBS job {B}{job_id}{E} {state}.{E}")
                return state == "completed"
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\n{'─' * 72}")
        print(f"  {B}Detached.{E}  PBS job {B}{job_id}{E} still running.")
        print(f"  Cancel via SSH:  ssh {user}@{host} 'qdel {job_id.split('.')[0]}'\n")
        return False


def _fetch_results(*, session, marker_path, dest_dir):
    """Read the expdir marker (via amsc) and rsync_pull the experiment dir back."""
    try: remote_expdir = session.read_file(marker_path).strip()
    except Exception as exc:
        print(f"  {R}Could not read expdir marker ({marker_path}): {exc}{E}")
        return
    if not remote_expdir:
        print(f"  {R}Expdir marker {marker_path} is empty; skipping fetch.{E}")
        return
    dest_root = Path(dest_dir).expanduser() / Path(remote_expdir).name
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"  Fetching {B}{Path(remote_expdir).name}{E} to {dest_root}...", end="  ", flush=True)
    if session.rsync_pull(remote_expdir, dest_root, is_dir=True):
        print(f"{G}OK{E}\n  Local: {B}{dest_root}{E}")
    else:
        print(f"{R}rsync pull failed{E}")


def _submit_main(args, parser) -> int:
    """--submit dispatch: sync auth + code to remote, amsc-submit one PBS job
    that runs --on-resource, poll its log via amsc.fs.view."""
    resource_cfg = RESOURCES[args.resource]
    remote_dir = args.remote_dir or resource_cfg.default_dir or _SCRIPT_DIR.name

    try:
        compute, fs = amsc_connect(resource_cfg.amsc_resource, resource_cfg.amsc_fs, _SCRIPT_DIR)
    except Exception as exc:
        print(f"  {R}Connection failed: {exc}{E}", file=sys.stderr); return 2

    session = HPCSession(resource_cfg.host, args.alcf_username, fs)
    print(f"  {G}✓{E}  Connected to {B}{resource_cfg.host}{E}")

    _ensure_local_inference_auth()
    _refresh_local_inference_token()
    _verify_inference_token()
    _sync_globus_tokens_to_remote(session, args.alcf_username)

    session.mkdir(remote_dir)
    if not args.skip_sync:
        print(f"  Syncing lassi_translate to {remote_dir} (following symlinks)...", end="  ", flush=True)
        if not session.rsync(str(_SCRIPT_DIR), remote_dir, is_dir=True, dereference=True):
            raise SystemExit("rsync failed")
        print(f"{G}OK{E}")

    remote_dir = session.resolve_abs(remote_dir)
    dt = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    log_dir = f"{remote_dir}/generated_codes/experiment/submit_logs"
    session.mkdir(log_dir)
    log_path = f"{log_dir}/lassi_submit_{dt}.log"
    expdir_marker = f"{log_dir}/lassi_submit_{dt}.expdir"

    body = _build_submit_pbs_script(args=args, resource_cfg=resource_cfg,
                                     remote_dir=remote_dir,
                                     remote_expdir_marker=expdir_marker,
                                     remote_log_path=log_path)
    # Upload the script to a file rather than embed in `arguments` — amsc
    # serializes arguments into GraphQL and bash specials trip the parser.
    remote_script = f"{log_dir}/lassi_submit_{dt}.sh"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                      prefix="lassi_submit_") as tf:
        tf.write("#!/bin/bash\n" + body); local_script = tf.name
    try:
        if not session.rsync(local_script, remote_script):
            raise SystemExit("submit-script rsync failed")
    finally:
        Path(local_script).unlink(missing_ok=True)

    # PBS Pro overwrites stdout_path with its (empty) spool at job-end; our
    # `exec > log_path` writes the live log, so PBS must spool elsewhere.
    print(f"  Submitting PBS job ({B}{args.queue}{E}, walltime {args.walltime}s)...", end="  ", flush=True)
    try:
        job = compute.submit(
            executable="/bin/bash", arguments=[remote_script],
            directory=remote_dir, name=f"lassi-translate-{int(time.time())}",
            queue=args.queue, account=args.account, duration=args.walltime, nodes=1,
            stdout_path=log_path + ".pbs_stdout", stderr_path=log_path + ".pbs_stderr",
            filesystems=resource_cfg.filesystems)
        print(f"{G}submitted{E}  (PBS ID: {B}{job.id}{E})")
    except Exception as exc:
        print(f"{R}FAILED: {exc}{E}", file=sys.stderr); return 2

    completed = _tail_until_ctrl_c(session, job, log_path,
                                    resource_cfg.host, args.alcf_username)
    if completed and args.fetch_results:
        dest = args.fetch_dest or str(_SCRIPT_DIR / "generated_codes" / "experiment")
        _fetch_results(session=session, marker_path=expdir_marker, dest_dir=dest)
    return 0


# ── --on-resource: pipeline runs locally on the compute node ─────────────────

def _on_resource_main(args, parser) -> int:
    """--on-resource dispatch: run the pipeline locally on the compute node
    (LocalSession + LocalRunner + ALCF Inference)."""
    resource_cfg = RESOURCES[args.resource]

    # Compute nodes need a proxy for inference-api.alcf.anl.gov
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(k, _COMPUTE_HTTPS_PROXY)

    if args.task == "energy" and args.resource != "polaris":
        parser.error("--task energy only supports --resource polaris (CUDA energy optimization)")

    input_code, run_args, original_src, verification = _load_source(args)

    from lassi.alcf_inference import get_access_token
    print(f"  Checking ALCF Inference auth...", end="  ", flush=True)
    get_access_token(_SCRIPT_DIR)
    print(f"{G}OK{E}")

    session = LocalSession()
    cwd_dir = str(_SCRIPT_DIR)
    workers_dir = f"{cwd_dir}/workers"

    print(f"  {G}✓{E}  Ready on-resource  (host: {B}{os.uname().nodename}{E}, "
          f"inference: {B}{args.inference_cluster}{E})")
    t_connected = time.monotonic()

    _banner(args, resource_cfg, cwd_dir,
            llm_endpoint=f"ALCF Inference ({args.inference_cluster})")

    dt = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[/: ]", "_", args.model)
    app_slug   = re.sub(r"[/: ]", "_", args.app_name)
    exp = f"exp_{app_slug}_{args.task}_{model_slug}_{resource_cfg.gpu_vendor}__{dt}"
    out_base = Path(args.output_dir).expanduser() if args.output_dir \
               else _SCRIPT_DIR / "generated_codes"
    exp_dir = out_base / "experiment" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Experiment : {exp}")
    print(f"  Local dir  : {exp_dir}\n")

    marker = os.environ.get("LASSI_EXPDIR_FILE")
    if marker:
        try: Path(marker).write_text(str(exp_dir.resolve()) + "\n")
        except Exception as exc: print(f"  [warn] could not write expdir marker: {exc}")

    sycl_env, cuda_env = _compiler_envs(args)

    def _runner(key, env):
        return build_local_runner(resource=args.resource, worker_key=key,
                                  workers_dir=workers_dir, compiler_env=env)

    _llm.configure(log_dir=str(exp_dir / "llm_calls"), debug=args.debug,
                   stream_display=True, backend="alcf_inference",
                   inference_cluster=args.inference_cluster,
                   inference_token_provider=lambda: get_access_token(_SCRIPT_DIR))

    if args.task != "translate":
        parser.error("--on-resource supports --task translate only")

    csv_path = exp_dir / "results.csv"
    exp_data = _new_exp_data(dt, args, resource_cfg.gpu_vendor)
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(exp_data.keys())).writeheader()

    baseline_metrics = None
    if can_run(args.resource, "cuda"):
        baseline_metrics = run_test_run(
            session=session, runner=_runner("energy_cuda", cuda_env),
            remote_dir=cwd_dir, original_src_path=original_src, src_extension=".cu",
            experiment_name=exp, experiment_dir=exp_dir,
            run_args=run_args, run_id=run_id + "b", verification=verification)
    else:
        print(f"\n  {B}Note:{E} CUDA can't run on {args.resource} — skipping test run.")

    success, _, gen = run_pipeline(
        model=args.model, task="translate", resource=args.resource,
        input_code=input_code, session=session, runner=_runner("translate", sycl_env),
        remote_dir=cwd_dir, src_extension=".cpp", run_args=run_args,
        experiment_name=exp, experiment_dir=exp_dir,
        csv_path=csv_path, experiment_data=exp_data,
        trial_number=1, run_id=run_id,
        use_self_prompting=args.self_prompting, verifyprompt=args.verifyprompt,
        verbose=args.debug, has_energy=False,
        use_judge=args.use_judge, verification=verification, measure_perf=True,
        baseline_metrics=baseline_metrics, ratchet_temperature=args.ratchet_temperature)

    if success:
        print_translate_summary(resource=args.resource,
            original_framework="cuda", target_framework="sycl",
            baseline=baseline_metrics, generated=gen,
            total_elapsed_s=time.monotonic() - t_connected,
            versioned_path=exp_dir / "Main_v1_Trial1.cpp",
            verification=verification)

    if not os.environ.get("LASSI_EXPDIR_FILE"):
        print(f"\n  {B}To retrieve results to your workstation, run there:{E}")
        print(f"    rsync -avz {args.alcf_username}@{resource_cfg.host}:"
              f"{shlex.quote(str(exp_dir.resolve()))} ./pulled_results/\n")
    return 0 if success else 1


# ── main: remote (default) ──────────────────────────────────────────────────

def main() -> int:
    """Parse args, validate mode flags, dispatch to remote/--on-resource/--submit."""
    try: sys.stdout.reconfigure(line_buffering=True)
    except Exception: pass

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(Path(pre_args.config) if pre_args.config else _SCRIPT_DIR / "lassi_config.py")

    parser = _build_parser(cfg)
    args = parser.parse_args()

    if not args.resource:      parser.error("--resource is required")
    if not args.task:          parser.error("--task is required")
    if not args.alcf_username: parser.error("--alcf-username is required")

    if args.submit and args.on_resource:
        parser.error("--submit and --on-resource are mutually exclusive")

    if args.model is None:
        args.model = args.inference_model if (args.on_resource or args.submit) else args.argo_model

    if args.on_resource:
        if args.remote_dir or args.skip_sync or not args.persistent_worker:
            parser.error("--on-resource ignores --remote-dir, --skip-sync, --no-persistent-worker")
        return _on_resource_main(args, parser)

    if args.submit:
        if args.task != "translate":
            parser.error("--submit supports --task translate only")
        return _submit_main(args, parser)

    # ── Remote mode ─────────────────────────────────────────────────────────
    resource_cfg = RESOURCES[args.resource]
    if args.task == "energy" and args.resource != "polaris":
        parser.error("--task energy only supports --resource polaris (CUDA energy optimization)")
    remote_dir = args.remote_dir or resource_cfg.default_dir

    try:
        compute, fs = amsc_connect(resource_cfg.amsc_resource, resource_cfg.amsc_fs, _SCRIPT_DIR)
    except Exception as exc:
        print(f"  {R}Connection failed: {exc}{E}", file=sys.stderr); return 2

    session = HPCSession(resource_cfg.host, args.alcf_username, fs)

    input_code, run_args, original_src, verification = _load_source(args)

    session.mkdir(remote_dir)
    if not args.skip_sync:
        if not session.rsync(str(_SCRIPT_DIR), remote_dir, is_dir=True):
            raise SystemExit("rsync failed")

    remote_dir = session.resolve_abs(remote_dir)
    session.mkdirs(f"{remote_dir}/generated_codes",
                   f"{remote_dir}/generated_codes/experiment")

    dt = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[/: ]", "_", args.model)
    app_slug   = re.sub(r"[/: ]", "_", args.app_name)
    exp = f"exp_{app_slug}_{args.task}_{model_slug}_{resource_cfg.gpu_vendor}__{dt}"
    out_base = Path(args.output_dir).expanduser() if args.output_dir \
               else _SCRIPT_DIR / "generated_codes"
    exp_dir = out_base / "experiment" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    sycl_env, cuda_env = _compiler_envs(args)
    ipc_dir = f"{remote_dir}/generated_codes/experiment/{exp}/ipc"
    # Pre-create the IPC dir so PBS can open stdout/stderr log paths inside it.
    session.mkdirs(f"{remote_dir}/generated_codes/experiment/{exp}", ipc_dir)

    persistent = None
    if args.persistent_worker:
        try:
            persistent = PersistentWorker.launch(
                session=session, compute_resource=compute, remote_dir=remote_dir,
                ipc_dir=ipc_dir, account=args.account, queue=args.queue,
                filesystems=resource_cfg.filesystems, walltime_s=args.walltime,
                resource_name=args.resource, quiet=False)
        except Exception as exc:
            print(f"  {R}Persistent worker launch failed: {exc}{E}")
            print(f"  Falling back to per-job PBS mode.")

    if persistent:
        print(f"  {G}✓{E}  Connected to {B}{resource_cfg.host}{E}  "
              f"(persistent worker on {B}{persistent.hostname}{E})")
    else:
        print(f"  {G}✓{E}  Connected to {B}{resource_cfg.host}{E}  (per-job PBS mode)")
    t_connected = time.monotonic()

    _banner(args, resource_cfg, remote_dir)
    print(f"  Experiment : {exp}")
    print(f"  Local dir  : {exp_dir}")
    print(f"  Remote dir : {remote_dir}/generated_codes/experiment/{exp}\n")

    def _runner(key, env):
        return build_runner(session=session, compute_resource=compute, resource=args.resource,
            remote_dir=remote_dir, account=args.account, queue=args.queue,
            filesystems=resource_cfg.filesystems, worker_key=key, compiler_env=env,
            walltime_s=args.walltime, persistent=persistent)

    success = False
    if args.task == "energy":
        key, ext, env = "energy_cuda", ".cu", cuda_env
        runner = _runner(key, env)
        bej, bpw, bso = run_baseline(
            session=session, runner=runner, remote_dir=remote_dir,
            original_src_path=original_src, src_extension=ext,
            experiment_name=exp, experiment_dir=exp_dir,
            run_args=run_args, run_id=run_id + "b", verification=verification)
        csv_path = exp_dir / "results.csv"
        exp_data = _new_exp_data(dt, args, resource_cfg.gpu_vendor, energy=True)
        exp_data["baseline_energy_j"] = bej
        # Synthesize a baseline_metrics dict from run_baseline's return so
        # run_pipeline can record baseline_runtime_s + throughput in the csv.
        baseline_metrics = {"run_stdout": bso, "duration_s": -1.0}
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(exp_data.keys())).writeheader()
        _llm.configure(log_dir=str(exp_dir / "llm_calls"), debug=args.debug,
                       stream_display=True)
        success, _, _ = run_pipeline(
            model=args.model, task="energy", resource=args.resource,
            input_code=input_code, session=session, runner=runner,
            remote_dir=remote_dir, src_extension=ext, run_args=run_args,
            experiment_name=exp, experiment_dir=exp_dir,
            csv_path=csv_path, experiment_data=exp_data,
            trial_number=1, run_id=run_id,
            use_self_prompting=args.self_prompting, verifyprompt=args.verifyprompt,
            verbose=args.debug, baseline_energy_j=bej, baseline_power_w=bpw,
            baseline_stdout=bso, min_energy_gain_pct=args.min_energy_gain,
            has_energy=True, use_judge=args.use_judge, verification=verification,
            baseline_metrics=baseline_metrics, ratchet_temperature=args.ratchet_temperature)

    else:  # translate
        csv_path = exp_dir / "results.csv"
        exp_data = _new_exp_data(dt, args, resource_cfg.gpu_vendor)
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(exp_data.keys())).writeheader()
        _llm.configure(log_dir=str(exp_dir / "llm_calls"), debug=args.debug,
                       stream_display=True)
        baseline_metrics = None
        if can_run(args.resource, "cuda"):
            baseline_metrics = run_test_run(
                session=session, runner=_runner("energy_cuda", cuda_env),
                remote_dir=remote_dir, original_src_path=original_src, src_extension=".cu",
                experiment_name=exp, experiment_dir=exp_dir,
                run_args=run_args, run_id=run_id + "b", verification=verification)
        else:
            print(f"\n  {B}Note:{E} CUDA can't run on {args.resource} — skipping test run.")
        success, _, gen = run_pipeline(
            model=args.model, task="translate", resource=args.resource,
            input_code=input_code, session=session, runner=_runner("translate", sycl_env),
            remote_dir=remote_dir, src_extension=".cpp", run_args=run_args,
            experiment_name=exp, experiment_dir=exp_dir,
            csv_path=csv_path, experiment_data=exp_data,
            trial_number=1, run_id=run_id,
            use_self_prompting=args.self_prompting, verifyprompt=args.verifyprompt,
            verbose=args.debug, has_energy=False,
            use_judge=args.use_judge, verification=verification, measure_perf=True,
            baseline_metrics=baseline_metrics)
        if success:
            print_translate_summary(resource=args.resource,
                original_framework="cuda", target_framework="sycl",
                baseline=baseline_metrics, generated=gen,
                total_elapsed_s=time.monotonic() - t_connected,
                versioned_path=exp_dir / "Main_v1_Trial1.cpp",
                verification=verification)

    if persistent: persistent.shutdown()
    return 0 if success else 1


def _safe_main() -> int:
    """Wrap main() with a clean error banner for unhandled exceptions."""
    try: return main()
    except KeyboardInterrupt: print(f"\n  {R}Interrupted.{E}"); return 130
    except SystemExit: raise
    except Exception as exc:
        bar = "=" * 72
        print(f"\n{bar}\n  {R}PIPELINE ERROR ({type(exc).__name__}){E}: {exc}\n{bar}")
        import traceback; traceback.print_exc(); return 1


if __name__ == "__main__":
    raise SystemExit(_safe_main())
