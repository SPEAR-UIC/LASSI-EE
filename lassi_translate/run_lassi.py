#!/usr/bin/env python3
"""LASSI-EE — CUDA→SYCL translation + GPU energy optimization on Polaris/Aurora.

CLI entry point; dispatches to one of three execution modes:

    (default) remote      workstation orchestrates; pipeline runs on the compute
                          node via a persistent PBS worker; LLM = Argo
    --on-resource         inside an interactive qsub -I shell on the compute
                          node; LLM = ALCF Inference (Argo unreachable there)
    --submit              workstation submits a PBS job that itself runs
                          --on-resource; tails its stdout back live; LLM = ALCF
                          Inference; auth + token sync handled automatically

Per-mode dispatch: main() → _submit_main / _on_resource_main / (inline remote).
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
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lassi.llm_client as _llm
from lassi.alcf         import amsc_connect, ssh_master_start, ssh_master_stop
from lassi.format_colors import B, G, R, E
from lassi.hpc_jobs     import JOB_DURATION_S, PersistentWorker, build_runner
from lassi.hpc_session  import HPCSession
from lassi.local_runner  import build_local_runner
from lassi.local_session import LocalSession
from lassi.pipeline     import (print_translate_summary, run_baseline, run_both_pipeline,
                                run_pipeline, run_test_run)
from lassi.resources    import RESOURCES, can_run
from lassi.source_loader import find_apps_json, load_cuda_source, load_sycl_source

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


# ── Config + CLI ─────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    """Execute lassi_config.py and return its top-level `config` dict.

    Legacy key `model` (pre mode-aware schema) maps to `argo_model`.
    """
    if not path.exists(): return {}
    try:
        ns: dict = {}
        exec(path.read_text(), ns)
        cfg = dict(ns.get("config", {}))
    except Exception as exc:
        print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
        return {}
    legacy = cfg.pop("model", None)
    if legacy and "argo_model" not in cfg: cfg["argo_model"] = legacy
    return cfg


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    """Build the argparse parser; `defaults` (from lassi_config.py) seeds flag defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")

    # Resource + task
    p.add_argument("--resource", default=None, choices=["polaris", "aurora"])
    p.add_argument("--task", default=None, choices=["translate", "energy", "both"])
    p.add_argument("--input-framework", default="cuda", choices=["cuda", "sycl"])
    p.add_argument("--app-category", default="Miniapps")
    p.add_argument("--app-name", default="XSBench")
    p.add_argument("--sycl-source", default="")

    # Identity / remote
    p.add_argument("--alcf-username", default=os.environ.get("USER", ""))
    p.add_argument("--remote-dir", default="")
    p.add_argument("--skip-sync", action="store_true")

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

    # Pipeline
    p.add_argument("--no-self-prompting", dest="self_prompting", action="store_false")
    p.add_argument("--no-judge",          dest="use_judge",      action="store_false")
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

    p.set_defaults(self_prompting=True, use_judge=True)
    if defaults: p.set_defaults(**defaults)
    return p


def _load_source(args, holder):
    """Read CUDA or SYCL source into memory; record the source Path in holder[0]."""
    if args.task in ("translate", "both") or (args.task == "energy" and args.input_framework == "cuda"):
        print(f"\n  Loading CUDA source ({args.app_category}/{args.app_name})...")
        code, run_args, src_path, ver = load_cuda_source(_SCRIPT_DIR, args.app_category, args.app_name)
    else:
        print(f"\n  Loading SYCL source from: {args.sycl_source}")
        src_path = Path(args.sycl_source).resolve()
        code = load_sycl_source(src_path)
        catalog = json.loads(find_apps_json(_SCRIPT_DIR).read_text())
        entry = catalog.get(args.app_category, {}).get(args.app_name, {})
        run_args = entry.get("app_params", [])
        ver = entry.get("verification", "none")
    holder[0] = src_path
    print(f"  Source: {len(code.splitlines())} lines  |  run args: {run_args}")
    return code, run_args, ver


def _compiler_envs(args):
    """Bundle the --sycl-* / --cuda-* CLI overrides into two env-var dicts
    that get forwarded to worker scripts as SYCL_FLAGS=..., etc."""
    sycl = {k: v for k, v in {"SYCL_COMPILER": args.sycl_compiler,
                              "SYCL_FLAGS": args.sycl_flags,
                              "SYCL_LDFLAGS": args.sycl_ldflags}.items() if v}
    cuda = {k: v for k, v in {"CUDA_COMPILER": args.cuda_compiler,
                              "CUDA_FLAGS": args.cuda_flags,
                              "CUDA_LDFLAGS": args.cuda_ldflags}.items() if v}
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
    if args.task == "energy":
        print(f"  Min gain   : {args.min_energy_gain:.1f}%")
    print(_SEP)


# ── Auth helpers (--submit path) ─────────────────────────────────────────────
# Goal: zero login-node setup.  All Globus auth happens on the workstation
# (which has a browser), then tokens.json is rsynced to the remote home dir
# where the PBS job can read it.  The compute node only ever reads cached
# tokens with stdlib JSON — no globus_sdk required there.

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


def _sync_globus_tokens_to_remote(session, host, user):
    """Mirror local tokens.json to the matching ~/.globus path on the remote.
    rsync -u preserves a newer remote copy if Polaris auto-refreshed since last submit."""
    local = _local_tokens_dir()
    if not local.is_dir(): raise SystemExit("local tokens missing")
    rc, home = session.run_capture("echo $HOME")
    remote_home = home.strip() if rc == 0 and home.strip() else "~"
    remote = f"{remote_home}/.globus/app/{_AUTH_CLIENT_ID}/{_APP_NAME}"
    print(f"  Syncing Globus tokens to {remote}...", end="  ", flush=True)
    session.run(f"mkdir -p {shlex.quote(remote)}")
    src = str(local).rstrip("/") + "/"
    cmd = ["rsync", "-au", "-e", f"ssh -o ControlPath={session.ssh_ctl}",
           src, f"{user}@{host}:{remote}/"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{R}FAILED{E}\n    {r.stderr.strip()[:300]}")
        raise SystemExit("token sync failed")
    print(f"{G}OK{E}")


# ── --submit: build + tail a PBS job that runs --on-resource ────────────────

def _build_submit_pbs_script(*, args, resource_cfg, remote_dir, remote_log_path):
    """Generate the bash PBS script body for --submit.

    The script: discovers conda (or uses --remote-python), then invokes
    `python3 run_lassi.py --on-resource ...` with all of the user's flags
    forwarded.  LASSI_FORCE_TTY=1 makes SlidingWindow render ANSI even
    though stdout is the PBS log file (the workstation tail-Fs it back).
    """
    inner = ["--on-resource", "--task", args.task, "--resource", args.resource,
             "--app-category", args.app_category, "--app-name", args.app_name,
             "--model", args.model, "--inference-cluster", args.inference_cluster,
             "--min-energy-gain", str(args.min_energy_gain)]
    if args.input_framework != "cuda": inner += ["--input-framework", args.input_framework]
    if args.sycl_source: inner += ["--sycl-source", args.sycl_source]
    if not args.self_prompting: inner += ["--no-self-prompting"]
    if not args.use_judge:      inner += ["--no-judge"]
    if args.verifyprompt:       inner += ["--verifyprompt"]
    if args.debug:              inner += ["--debug"]
    for flag, val in [("--sycl-compiler", args.sycl_compiler),
                      ("--sycl-flags",    args.sycl_flags),
                      ("--sycl-ldflags",  args.sycl_ldflags),
                      ("--cuda-compiler", args.cuda_compiler),
                      ("--cuda-flags",    args.cuda_flags),
                      ("--cuda-ldflags",  args.cuda_ldflags)]:
        if val: inner += [flag, val]
    inner_cmd = " ".join(shlex.quote(a) for a in inner)

    h, rem = divmod(args.walltime, 3600); m, s = divmod(rem, 60)
    wt = f"{h:02d}:{m:02d}:{s:02d}"
    fs = f"#PBS -l filesystems={resource_cfg.filesystems}" if resource_cfg.filesystems else ""

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
        "#!/bin/bash\n"
        f"#PBS -N lassi-translate-{int(time.time())}\n"
        f"#PBS -q {args.queue}\n#PBS -A {args.account}\n"
        f"#PBS -l select=1\n#PBS -l walltime={wt}\n{fs}\n"
        f"#PBS -o {remote_log_path}\n#PBS -e {remote_log_path}.err\n\n"
        "export PYTHONUNBUFFERED=1\n"
        "export LASSI_FORCE_TTY=1\n\n"
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


def _tail_until_ctrl_c(session, log_path, job_id, host, user):
    """ssh tail -F the PBS log to local stdout.  Ctrl-C detaches without killing
    the job and prints reattach/qdel commands the user can paste later."""
    print(f"  {B}Streaming {log_path} (Ctrl-C to detach).{E}\n")
    print("─" * 72)
    tail = subprocess.Popen(
        ["ssh", "-o", f"ControlPath={session.ssh_ctl}", f"{user}@{host}",
         f"tail -n +1 -F {shlex.quote(log_path)} 2>/dev/null"],
        stdout=sys.stdout, stderr=sys.stderr)
    try: tail.wait()
    except KeyboardInterrupt:
        tail.terminate()
        try: tail.wait(timeout=2)
        except subprocess.TimeoutExpired: tail.kill()
        print(f"\n{'─' * 72}")
        print(f"  {B}Detached.{E}  PBS job {B}{job_id}{E} still running.")
        print(f"  Re-attach:  ssh {user}@{host} 'tail -F {log_path}'")
        print(f"  Kill:       ssh {user}@{host} 'qdel {job_id}'\n")


def _submit_main(args, parser) -> int:
    """--submit dispatch: from the workstation, set up auth + tokens + code on
    the remote, submit one PBS job that runs --on-resource, tail its stdout."""
    resource_cfg = RESOURCES[args.resource]
    remote_dir = args.remote_dir or resource_cfg.default_dir or _SCRIPT_DIR.name

    try:
        compute, fs = amsc_connect(resource_cfg.amsc_resource, resource_cfg.amsc_fs, _SCRIPT_DIR)
        ssh_ctl = ssh_master_start(resource_cfg.host, args.alcf_username)
    except Exception as exc:
        print(f"  {R}Connection failed: {exc}{E}", file=sys.stderr); return 2

    session = HPCSession(resource_cfg.host, args.alcf_username, ssh_ctl, fs)
    print(f"  {G}✓{E}  Connected to {B}{resource_cfg.host}{E}")

    _ensure_local_inference_auth()
    _refresh_local_inference_token()
    _verify_inference_token()
    _sync_globus_tokens_to_remote(session, resource_cfg.host, args.alcf_username)

    session.mkdir(remote_dir)
    if not args.skip_sync:
        print(f"  Syncing lassi_translate to {remote_dir} (following symlinks)...", end="  ", flush=True)
        if not session.rsync(str(_SCRIPT_DIR), remote_dir, is_dir=True, dereference=True):
            ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)
            raise SystemExit("rsync failed")
        print(f"{G}OK{E}")

    remote_dir = session.resolve_abs(remote_dir)
    dt = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    log_dir = f"{remote_dir}/generated_codes/experiment/submit_logs"
    session.mkdir(log_dir)
    log_path = f"{log_dir}/lassi_submit_{dt}.log"

    script = _build_submit_pbs_script(args=args, resource_cfg=resource_cfg,
                                       remote_dir=remote_dir, remote_log_path=log_path)
    print(f"  Submitting PBS job ({B}{args.queue}{E}, walltime {args.walltime}s)...", end="  ", flush=True)
    try:
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={ssh_ctl}",
             f"{args.alcf_username}@{resource_cfg.host}", "qsub"],
            input=script, capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            print(f"{R}FAILED{E}\n  qsub stderr: {r.stderr.strip()}", file=sys.stderr)
            ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username); return 2
        job_id = r.stdout.strip()
        print(f"{G}submitted{E}  (PBS ID: {B}{job_id}{E})")
    except Exception as exc:
        print(f"{R}FAILED: {exc}{E}", file=sys.stderr)
        ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username); return 2

    _tail_until_ctrl_c(session, log_path, job_id, resource_cfg.host, args.alcf_username)
    ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)
    return 0


# ── --on-resource: pipeline runs locally on the compute node ─────────────────

def _on_resource_main(args, parser) -> int:
    """--on-resource dispatch: run the whole pipeline locally on the compute
    node (LocalSession + LocalRunner + ALCF Inference LLM).  Called directly
    in an interactive qsub -I shell, or from inside a --submit PBS job."""
    resource_cfg = RESOURCES[args.resource]

    # Compute nodes need a proxy for inference-api.alcf.anl.gov
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(k, _COMPUTE_HTTPS_PROXY)

    if (args.task == "energy" and args.resource == "aurora"
            and args.input_framework == "cuda" and not args.sycl_source):
        parser.error("Aurora energy task needs --input-framework sycl --sycl-source PATH")

    holder = [None]
    input_code, run_args, verification = _load_source(args, holder)
    original_src = holder[0]

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
    exp_dir = _SCRIPT_DIR / "generated_codes" / "experiment" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Experiment : {exp}")
    print(f"  Local dir  : {exp_dir}\n")

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
    exp_data = {"timestamp": dt, "app": args.app_name, "model": args.model,
                "gpu_vendor": resource_cfg.gpu_vendor, "success": False, "runtime_min": 0.0}
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
        use_judge=args.use_judge, verification=verification, measure_perf=True)

    if success:
        print_translate_summary(resource=args.resource,
            original_framework=args.input_framework, target_framework="sycl",
            baseline=baseline_metrics, generated=gen,
            total_elapsed_s=time.monotonic() - t_connected,
            versioned_path=exp_dir / "Main_v1_Trial1.cpp",
            verification=verification)
    return 0 if success else 1


# ── main: remote (default) ──────────────────────────────────────────────────

def main() -> int:
    """Parse args + config, validate mode flags, dispatch to one of three modes:
       --on-resource → _on_resource_main, --submit → _submit_main, else remote (inline)."""
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
    if (args.task == "energy" and args.resource == "aurora"
            and args.input_framework == "cuda" and not args.sycl_source):
        parser.error("Aurora energy task needs --input-framework sycl --sycl-source PATH")
    remote_dir = args.remote_dir or resource_cfg.default_dir

    try:
        compute, fs = amsc_connect(resource_cfg.amsc_resource, resource_cfg.amsc_fs, _SCRIPT_DIR)
        ssh_ctl = ssh_master_start(resource_cfg.host, args.alcf_username)
    except Exception as exc:
        print(f"  {R}Connection failed: {exc}{E}", file=sys.stderr); return 2

    session = HPCSession(resource_cfg.host, args.alcf_username, ssh_ctl, fs)

    holder = [None]
    input_code, run_args, verification = _load_source(args, holder)
    original_src = holder[0]

    session.mkdir(remote_dir)
    if not args.skip_sync:
        if not session.rsync(str(_SCRIPT_DIR), remote_dir, is_dir=True):
            ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)
            raise SystemExit("rsync failed")

    remote_dir = session.resolve_abs(remote_dir)
    session.mkdirs(f"{remote_dir}/generated_codes",
                   f"{remote_dir}/generated_codes/experiment")

    dt = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[/: ]", "_", args.model)
    app_slug   = re.sub(r"[/: ]", "_", args.app_name)
    exp = f"exp_{app_slug}_{args.task}_{model_slug}_{resource_cfg.gpu_vendor}__{dt}"
    exp_dir = _SCRIPT_DIR / "generated_codes" / "experiment" / exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    sycl_env, cuda_env = _compiler_envs(args)
    ipc_dir = f"{remote_dir}/generated_codes/experiment/{exp}/ipc"

    print(f"  Establishing connection to {B}{resource_cfg.host}{E} "
          f"(persistent worker submitting)...", flush=True)
    persistent = None
    if args.persistent_worker and ssh_ctl:
        try:
            persistent = PersistentWorker.launch(
                session=session, compute_resource=compute, remote_dir=remote_dir,
                ipc_dir=ipc_dir, account=args.account, queue=args.queue,
                filesystems=resource_cfg.filesystems, walltime_s=args.walltime,
                resource_name=args.resource, quiet=True)
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
    if args.task == "both":
        success = run_both_pipeline(
            model=args.model, resource=args.resource, resource_cfg=resource_cfg,
            session=session, compute_resource=compute,
            remote_dir=remote_dir, account=args.account, queue=args.queue,
            run_args=run_args, cuda_input_code=input_code, original_cu_path=original_src,
            experiment_dir=exp_dir, experiment_name=exp,
            csv_path=exp_dir / "results.csv", run_id=run_id, walltime_s=args.walltime,
            sycl_compiler_env=sycl_env, sycl_energy_compiler_env=sycl_env,
            use_self_prompting=args.self_prompting,
            verifyprompt=args.verifyprompt, verbose=args.debug,
            min_energy_gain_pct=args.min_energy_gain,
            use_judge=args.use_judge, verification=verification, persistent=persistent)

    elif args.task == "energy":
        if args.resource == "polaris" and args.input_framework == "cuda":
            key, ext, env = "energy_cuda", ".cu", cuda_env
        else:
            key, ext, env = "energy_sycl", ".cpp", sycl_env
        runner = _runner(key, env)
        bej, bpw, bso = run_baseline(
            session=session, runner=runner, remote_dir=remote_dir,
            original_src_path=original_src, src_extension=ext,
            experiment_name=exp, experiment_dir=exp_dir,
            run_args=run_args, run_id=run_id + "b", verification=verification)
        csv_path = exp_dir / "results.csv"
        exp_data = {"timestamp": dt, "app": args.app_name, "model": args.model,
                    "gpu_vendor": resource_cfg.gpu_vendor, "success": False, "runtime_min": 0.0,
                    "baseline_energy_j": bej, "optimized_energy_j": -1.0,
                    "energy_reduction_pct": 0.0}
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
            has_energy=True, use_judge=args.use_judge, verification=verification)

    else:  # translate
        csv_path = exp_dir / "results.csv"
        exp_data = {"timestamp": dt, "app": args.app_name, "model": args.model,
                    "gpu_vendor": resource_cfg.gpu_vendor, "success": False, "runtime_min": 0.0}
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
            use_judge=args.use_judge, verification=verification, measure_perf=True)
        if success:
            print_translate_summary(resource=args.resource,
                original_framework=args.input_framework, target_framework="sycl",
                baseline=baseline_metrics, generated=gen,
                total_elapsed_s=time.monotonic() - t_connected,
                versioned_path=exp_dir / "Main_v1_Trial1.cpp",
                verification=verification)

    if persistent: persistent.shutdown()
    if ssh_ctl:    ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)
    return 0 if success else 1


def _safe_main() -> int:
    """Wrap main() so unhandled exceptions print a clean banner instead of a
    bare Python traceback (matters most in --submit where the user tails a log)."""
    try: return main()
    except KeyboardInterrupt:
        print(f"\n  {R}Interrupted.{E}"); return 130
    except SystemExit: raise
    except Exception as exc:
        bar = "=" * 72
        print(f"\n{bar}\n  {R}PIPELINE ERROR ({type(exc).__name__}){E}: {exc}\n{bar}")
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_safe_main())
