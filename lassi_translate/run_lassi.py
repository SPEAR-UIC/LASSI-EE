#!/usr/bin/env python3
"""LASSI-EE — LLM-driven CUDA→SYCL translation and GPU energy optimization.

Entry point: parses CLI args, loads source, opens ALCF/SSH sessions, launches
the persistent worker (if enabled), and dispatches to the appropriate pipeline
runner in lassi/pipeline.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lassi.llm_client as _llm
from lassi.alcf         import amsc_connect, ssh_master_start, ssh_master_stop
from lassi.format_colors import color
from lassi.hpc_jobs     import (
    JOB_DURATION_S, PersistentWorker, build_runner,
)
from lassi.hpc_session  import HPCSession
from lassi.local_runner  import build_local_runner
from lassi.local_session import LocalSession
from lassi.pipeline     import (
    print_translate_summary, read_baseline_metrics,
    run_baseline, run_both_pipeline, run_pipeline, run_test_run,
)
from lassi.resources    import RESOURCES, can_run
from lassi.source_loader import find_apps_json, load_cuda_source, load_sycl_source

_B, _G, _R, _E = color.BOLD, color.GREEN, color.RED, color.END
_SEP = "=" * 72

_SCRIPT_DIR = Path(__file__).resolve().parent

_DEFAULT_MODEL        = "gpt55"
_DEFAULT_ACCOUNT      = "EE-ECP"
_DEFAULT_QUEUE        = "debug"
_DEFAULT_MIN_GAIN_PCT = 1.0


def _load_config(path: Path) -> dict:
    """Execute a Python config file; return its top-level `config` dict."""
    if not path.exists():
        return {}
    try:
        ns: dict = {}
        exec(path.read_text(), ns)  # noqa: S102
        return ns.get("config", {})
    except Exception as exc:
        print(f"Warning: could not load config {path}: {exc}", file=sys.stderr)
        return {}


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="", metavar="PATH",
                   help="Path to a Python config file (default: lassi_config.py "
                        "in the script directory). Config values act as defaults; "
                        "CLI arguments override them.")

    # Resource + task
    p.add_argument("--resource", default=None, choices=["polaris", "aurora"],
                   help="Target HPC resource: polaris (NVIDIA A100) or aurora (Intel PVC)")
    p.add_argument("--task", default=None, choices=["translate", "energy", "both"],
                   help="translate: CUDA→SYCL with correctness verification. "
                        "energy: GPU energy optimization (CUDA→CUDA on Polaris, SYCL→SYCL on Aurora). "
                        "both: translate CUDA→SYCL then measure SYCL energy vs CUDA baseline.")
    p.add_argument("--input-framework", default="cuda", choices=["cuda", "sycl"],
                   help="Input code framework (default: cuda). Use sycl for Aurora "
                        "energy task with an existing SYCL source. Ignored for translate "
                        "and both tasks.")

    # Source application
    p.add_argument("--app-category", default="Miniapps", metavar="CATEGORY",
                   help="Application category from source_apps.json (default: Miniapps)")
    p.add_argument("--app-name", default="XSBench", metavar="APP",
                   help="Application name within the category (default: XSBench)")
    p.add_argument("--sycl-source", default="", metavar="PATH",
                   help="Path to an existing SYCL .cpp file (for Aurora energy task with "
                        "--input-framework sycl). Required when --resource aurora --task energy.")

    # Identity / deployment
    p.add_argument("--alcf-username", default=os.environ.get("USER", ""), metavar="USER",
                   help="ALCF login name for rsync/SSH (default: $USER)")
    p.add_argument("--remote-dir", default="", metavar="PATH",
                   help="Absolute remote filesystem path for this project. "
                        "Defaults to /eagle/EE-ECP/LASSI-EE/lassi_jobs (Polaris) or "
                        "/flare/EE-ECP/LASSI-EE/lassi_jobs (Aurora).")
    p.add_argument("--skip-sync", action="store_true",
                   help="Skip initial rsync of lassi_jobs/ to the remote filesystem.")

    # PBS job parameters
    p.add_argument("--account", default=_DEFAULT_ACCOUNT,
                   help=f"PBS account (default: {_DEFAULT_ACCOUNT})")
    p.add_argument("--queue", default=_DEFAULT_QUEUE,
                   help=f"PBS queue (default: {_DEFAULT_QUEUE})")
    p.add_argument("--walltime", type=int, default=JOB_DURATION_S, metavar="SECONDS",
                   help=f"PBS walltime in seconds (default: {JOB_DURATION_S}). "
                        "For the persistent worker this is the total budget for the whole pipeline run.")
    p.add_argument("--no-persistent-worker", dest="persistent_worker",
                   action="store_false", default=True,
                   help="Disable persistent worker mode and submit one PBS job per attempt "
                        "(slower due to repeated queue wait; useful for debugging).")

    # LLM parameters
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"LLM model name.  Argo default: {_DEFAULT_MODEL} "
                        "(gpt4o, gpt41, claudesonnet4 also valid).  "
                        "ALCF Inference default: meta-llama/Meta-Llama-3.1-70B-Instruct.")

    # On-resource mode (run the pipeline directly on a Polaris/Aurora compute
    # node — no SSH, no PBS, no Globus, no persistent worker).
    p.add_argument("--on-resource", action="store_true", default=False,
                   help="Run the pipeline locally on a Polaris/Aurora compute node. "
                        "Skips SSH/Globus/PBS entirely and uses the ALCF Inference "
                        "Endpoints API for LLM calls (Argo is unreachable on the "
                        "compute nodes).  Prereq:  python lassi_translate/"
                        "inference_auth_token.py authenticate")
    p.add_argument("--inference-cluster", default="sophia",
                   choices=["sophia", "metis"],
                   help="ALCF Inference cluster (--on-resource only). "
                        "sophia=vLLM, metis=SambaNova. (default: sophia)")

    # Pipeline options
    p.add_argument("--no-self-prompting", dest="self_prompting", action="store_false",
                   help="Disable Stage 0 self-prompting context builder")
    p.add_argument("--no-judge", dest="use_judge", action="store_false",
                   help="Disable LLM-as-judge functional-equivalence check after each successful run")
    p.add_argument("--min-energy-gain", type=float, default=_DEFAULT_MIN_GAIN_PCT, metavar="PCT",
                   help=f"Minimum energy reduction (%%) required to declare success for energy tasks "
                        f"(default: {_DEFAULT_MIN_GAIN_PCT:.1f}%%). Use 0 to accept any correct optimization.")
    p.add_argument("--verifyprompt", action=argparse.BooleanOptionalAction, default=False,
                   help="Print full prompts before each LLM call")
    p.add_argument("--debug", action="store_true", default=False,
                   help="Enable debug mode: print prompt sizes and extra diagnostics")

    # Compiler overrides (forwarded to worker scripts)
    p.add_argument("--sycl-compiler", default="", metavar="PATH",
                   help="Override SYCL compiler on the compute node (sets SYCL_COMPILER; default: icpx)")
    p.add_argument("--sycl-flags", default="", metavar="FLAGS",
                   help="Override SYCL compilation flags (sets SYCL_FLAGS)")
    p.add_argument("--sycl-ldflags", default="", metavar="FLAGS",
                   help="Extra SYCL linker flags (sets SYCL_LDFLAGS)")
    p.add_argument("--cuda-compiler", default="", metavar="PATH",
                   help="Override CUDA compiler on the compute node (sets CUDA_COMPILER; default: nvcc)")
    p.add_argument("--cuda-flags", default="", metavar="FLAGS",
                   help="Override CUDA compilation flags (sets CUDA_FLAGS)")
    p.add_argument("--cuda-ldflags", default="", metavar="FLAGS",
                   help="Extra CUDA linker flags (sets CUDA_LDFLAGS)")

    p.set_defaults(self_prompting=True, use_judge=True)
    if defaults:
        p.set_defaults(**defaults)
    return p


def _load_source(args, original_cu_path_holder: list[Path | None]) -> tuple[str, list, str]:
    """Resolve and load CUDA or SYCL input code based on task/framework.

    Returns (input_code, run_args, verification). Side-effect: appends the
    resolved source path to `original_cu_path_holder[0]` for later use.
    """
    if args.task in ("translate", "both") or (args.task == "energy" and args.input_framework == "cuda"):
        print(f"\n  Loading CUDA source ({args.app_category}/{args.app_name})...")
        input_code, run_args, src_path, verification = load_cuda_source(
            _SCRIPT_DIR, args.app_category, args.app_name,
        )
    else:
        # Aurora energy with SYCL input.
        print(f"\n  Loading SYCL source from: {args.sycl_source}")
        src_path   = Path(args.sycl_source).resolve()
        input_code = load_sycl_source(src_path)
        catalog    = json.loads(find_apps_json(_SCRIPT_DIR).read_text())
        app_entry  = catalog.get(args.app_category, {}).get(args.app_name, {})
        run_args     = app_entry.get("app_params", [])
        verification = app_entry.get("verification", "none")

    original_cu_path_holder[0] = src_path
    print(f"  Source: {len(input_code.splitlines())} lines  |  run args: {run_args}")
    return input_code, run_args, verification


def _compiler_envs(args) -> tuple[dict, dict]:
    sycl_env = {k: v for k, v in {
        "SYCL_COMPILER": args.sycl_compiler,
        "SYCL_FLAGS":    args.sycl_flags,
        "SYCL_LDFLAGS":  args.sycl_ldflags,
    }.items() if v}
    cuda_env = {k: v for k, v in {
        "CUDA_COMPILER": args.cuda_compiler,
        "CUDA_FLAGS":    args.cuda_flags,
        "CUDA_LDFLAGS":  args.cuda_ldflags,
    }.items() if v}
    return sycl_env, cuda_env


def _print_banner(args, resource_cfg, remote_dir: str) -> None:
    print(_SEP)
    print(f"  {_B}LASSI-Jobs{_E}  |  resource={args.resource}  task={args.task}")
    print(_SEP)
    print(f"  LLM        : {_B}{args.model}{_E}  via Argo  (apps-dev.inside.anl.gov)")
    print(f"  Resource   : {_B}{args.resource}{_E} ({resource_cfg.gpu_vendor} GPU)")
    print(f"  Queue      : {_B}{args.queue}{_E}  |  account: {args.account}")
    print(f"  Remote dir : {remote_dir}")
    if args.task == "energy":
        print(f"  Min gain   : {args.min_energy_gain:.1f}% energy reduction required")
    mode_labels = {
        "translate": f"CUDA→SYCL translation + self-correction "
                     f"({'with' if args.self_prompting else 'no'} self-prompting)",
        "energy":    f"Energy optimization + self-correction "
                     f"({'with' if args.self_prompting else 'no'} self-prompting)",
        "both":      f"Translate CUDA→SYCL + SYCL energy optimization",
    }
    print(f"  Mode       : {mode_labels[args.task]}")
    if args.debug:
        print(f"  Debug      : ON")
    print(_SEP)


def _launch_persistent_worker(args, session, compute_resource, remote_dir, ipc_dir,
                              resource_cfg) -> PersistentWorker | None:
    """Try to launch the persistent worker; return None if disabled or it fails.

    Runs quietly — callers are expected to surface a single line about whether
    we ended up in persistent or per-job mode (using `.hostname` on success).
    """
    if not args.persistent_worker:
        return None
    if not session.ssh_ctl:
        print(f"  {_R}Warning: SSH ControlMaster not available — "
              f"falling back to per-job PBS mode.{_E}")
        return None
    try:
        return PersistentWorker.launch(
            session=session, compute_resource=compute_resource,
            remote_dir=remote_dir, ipc_dir=ipc_dir,
            account=args.account, queue=args.queue,
            filesystems=resource_cfg.filesystems,
            walltime_s=args.walltime, resource_name=args.resource,
            quiet=True,
        )
    except Exception as exc:
        print(f"  {_R}Persistent worker launch failed: {exc}{_E}")
        print(f"  Falling back to per-job PBS mode.")
        return None


def _on_resource_main(args, parser) -> int:
    """Pipeline entry point for --on-resource mode (no SSH/PBS/Globus)."""
    resource_cfg = RESOURCES[args.resource]

    # Pick an ALCF Inference default model if the user kept the Argo default.
    if args.model == _DEFAULT_MODEL:
        from lassi.alcf_inference import DEFAULT_MODEL as _INF_DEFAULT
        args.model = _INF_DEFAULT

    # ── Validate the same task/framework rules as remote mode ────────────────
    if (args.task == "energy" and args.resource == "aurora"
            and args.input_framework == "cuda" and not args.sycl_source):
        parser.error(
            "Aurora energy task requires a SYCL input source.\n"
            "Use --input-framework sycl --sycl-source /path/to/Main.cpp."
        )

    # ── Load source ──────────────────────────────────────────────────────────
    src_path_holder: list[Path | None] = [None]
    input_code, run_args, verification = _load_source(args, src_path_holder)
    original_src_path = src_path_holder[0]
    assert original_src_path is not None

    # ── Auth check (fail fast if not yet authenticated) ──────────────────────
    from lassi.alcf_inference import get_access_token
    print(f"  Checking ALCF Inference auth...", end="  ", flush=True)
    _ = get_access_token(_SCRIPT_DIR)   # raises SystemExit with guidance
    print(f"{_G}OK{_E}")

    # Token provider — fetches a fresh token (cached/refreshed by the helper)
    # each LLM call.  Cheap, just reads the local tokens.json.
    _token_provider = lambda: get_access_token(_SCRIPT_DIR)

    # ── Use $CWD as remote_dir, build session/runners locally ────────────────
    session  = LocalSession()
    cwd_dir  = str(_SCRIPT_DIR)        # everything happens inside lassi_translate/
    workers_dir = f"{cwd_dir}/workers"

    print(f"  {_G}✓{_E}  Ready on-resource  "
          f"(host: {_B}{os.uname().nodename}{_E}, "
          f"inference: {_B}{args.inference_cluster}{_E})")
    t_connected = time.monotonic()

    _print_banner(args, resource_cfg, cwd_dir)

    # ── Experiment setup (local only) ────────────────────────────────────────
    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id    = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[/: ]", "_", args.model)
    app_slug   = re.sub(r"[/: ]", "_", args.app_name)
    experiment_name = (
        f"exp_{app_slug}_{args.task}_{model_slug}"
        f"_{resource_cfg.gpu_vendor}__{dt_string}"
    )
    experiment_dir = _SCRIPT_DIR / "generated_codes" / "experiment" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Experiment : {experiment_name}")
    print(f"  Local dir  : {experiment_dir}\n")

    sycl_env, cuda_env = _compiler_envs(args)

    def _runner(worker_key: str, compiler_env: dict[str, str]):
        return build_local_runner(
            resource=args.resource, worker_key=worker_key,
            workers_dir=workers_dir, compiler_env=compiler_env,
        )

    # ── Configure LLM client (ALCF Inference backend, streaming display) ────
    _llm.configure(
        log_dir=str(experiment_dir / "llm_calls"),
        truncate_chars=4000, debug=args.debug,
        stream_display=True,
        backend="alcf_inference",
        inference_cluster=args.inference_cluster,
        inference_token_provider=_token_provider,
    )

    # ── Dispatch (only translate is wired here for now — energy/both reuse
    # the same shape; add when needed) ───────────────────────────────────────
    if args.task != "translate":
        parser.error(
            "--on-resource currently supports --task translate only.  "
            "Energy/both on-resource is straightforward to add — ask if needed."
        )

    csv_path = experiment_dir / "results.csv"
    exp_data = {
        "timestamp":   dt_string,
        "app":         args.app_name,
        "model":       args.model,
        "gpu_vendor":  resource_cfg.gpu_vendor,
        "success":     False,
        "runtime_min": 0.0,
    }
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(exp_data.keys())).writeheader()

    # Test run of the original source (no EE — translate-only flow)
    baseline_metrics = None
    if can_run(args.resource, "cuda"):
        baseline_metrics = run_test_run(
            session=session,
            runner=_runner("energy_cuda", cuda_env),
            remote_dir=cwd_dir,
            original_src_path=original_src_path, src_extension=".cu",
            experiment_name=experiment_name, experiment_dir=experiment_dir,
            run_args=run_args, run_id=run_id + "b",
            verification=verification,
        )
    else:
        print(f"\n  {_B}Note:{_E} CUDA cannot run on {args.resource} "
              f"({resource_cfg.gpu_vendor} GPU) — skipping test run.")

    success, _, gen_metrics = run_pipeline(
        model=args.model, task="translate", resource=args.resource,
        input_code=input_code,
        session=session, runner=_runner("translate", sycl_env),
        remote_dir=cwd_dir, src_extension=".cpp",
        run_args=run_args,
        experiment_name=experiment_name, experiment_dir=experiment_dir,
        csv_path=csv_path, experiment_data=exp_data,
        trial_number=1, run_id=run_id,
        use_self_prompting=args.self_prompting,
        verifyprompt=args.verifyprompt, verbose=args.debug,
        has_energy=False,
        use_judge=args.use_judge, verification=verification,
        measure_perf=True,
    )

    if success:
        print_translate_summary(
            resource=args.resource,
            original_framework=args.input_framework,
            target_framework="sycl",
            baseline=baseline_metrics,
            generated=gen_metrics,
            total_elapsed_s=time.monotonic() - t_connected,
            versioned_path=experiment_dir / "Main_v1_Trial1.cpp",
            verification=verification,
        )

    return 0 if success else 1


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Pre-parse --config to seed the real parser's defaults.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default="")
    _pre_args, _ = _pre.parse_known_args()
    config_path = Path(_pre_args.config) if _pre_args.config else _SCRIPT_DIR / "lassi_config.py"
    cfg = _load_config(config_path)

    parser = _build_parser(cfg)
    args   = parser.parse_args()

    if not args.resource:
        parser.error("--resource is required (or set resource = 'polaris' in lassi_config.py)")
    if not args.task:
        parser.error("--task is required (or set task = 'energy' in lassi_config.py)")
    if not args.alcf_username:
        parser.error("--alcf-username is required (or set alcf_username = '...' in lassi_config.py)")

    if args.debug:
        _llm.configure(debug=True)

    # ── --on-resource: alternate entry point with no SSH/PBS/Globus ─────────
    if args.on_resource:
        # Sanity: reject conflicting remote-mode flags.
        if args.remote_dir or args.skip_sync or not args.persistent_worker:
            parser.error(
                "--on-resource cannot be combined with --remote-dir, "
                "--skip-sync, or --no-persistent-worker (those are remote-mode "
                "concerns; on-resource runs everything locally)."
            )
        return _on_resource_main(args, parser)

    resource_cfg = RESOURCES[args.resource]

    if (args.task == "energy" and args.resource == "aurora"
            and args.input_framework == "cuda" and not args.sycl_source):
        parser.error(
            "Aurora energy task requires a SYCL input source.\n"
            "Either:\n"
            "  (a) use --input-framework sycl --sycl-source /path/to/Main.cpp, or\n"
            "  (b) use --task both to translate CUDA→SYCL first, then optimize energy."
        )

    # Empty `default_dir` (the on-disk default) is treated as "no fixed remote
    # path — derive one from the local script-directory name".
    remote_dir = args.remote_dir or resource_cfg.default_dir or _SCRIPT_DIR.name

    # ── Load source (fast, local — fail early on missing files) ──────────────
    src_path_holder: list[Path | None] = [None]
    input_code, run_args, verification = _load_source(args, src_path_holder)
    original_cu_path = src_path_holder[0]
    assert original_cu_path is not None

    # ── Connect to ALCF + SSH (silent) ───────────────────────────────────────
    try:
        compute_resource, fs_resource = amsc_connect(
            resource_cfg.amsc_resource, resource_cfg.amsc_fs, _SCRIPT_DIR,
        )
        ssh_ctl = ssh_master_start(resource_cfg.host, args.alcf_username)
    except Exception as exc:
        print(f"  {_R}Connection failed: {exc}{_E}", file=sys.stderr)
        return 2

    session = HPCSession(
        host=resource_cfg.host, user=args.alcf_username,
        ssh_ctl=ssh_ctl, fs_resource=fs_resource,
    )

    # ── Ensure remote dir exists, then rsync (silent) ────────────────────────
    # Explicit mkdir guards against rsync versions without --mkpath.
    session.mkdir(remote_dir)
    if not args.skip_sync:
        if not session.rsync(str(_SCRIPT_DIR), remote_dir, is_dir=True):
            if ssh_ctl:
                ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)
            raise SystemExit("Initial rsync to remote host failed.  Check SSH key auth.")

    # Resolve remote_dir to an absolute path.  The amsc filesystem API rejects
    # relative paths ("Path must be absolute" 400 errors), so every downstream
    # f"{remote_dir}/..." string must already be absolute.
    remote_dir = session.resolve_abs(remote_dir)

    session.mkdirs(
        f"{remote_dir}/generated_codes",
        f"{remote_dir}/generated_codes/experiment",
    )

    # ── Experiment setup ─────────────────────────────────────────────────────
    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    run_id    = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_slug = re.sub(r"[/: ]", "_", args.model)
    app_slug   = re.sub(r"[/: ]", "_", args.app_name)
    experiment_name = (
        f"exp_{app_slug}_{args.task}_{model_slug}"
        f"_{resource_cfg.gpu_vendor}__{dt_string}"
    )
    experiment_dir = _SCRIPT_DIR / "generated_codes" / "experiment" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    sycl_env, cuda_env = _compiler_envs(args)

    # ── Launch persistent worker (the real "establishing a connection") ──────
    ipc_dir    = f"{remote_dir}/generated_codes/experiment/{experiment_name}/ipc"
    print(f"  Establishing connection to {_B}{resource_cfg.host}{_E} "
          f"(persistent worker submitting to queue)...", flush=True)
    persistent = _launch_persistent_worker(
        args, session, compute_resource, remote_dir, ipc_dir, resource_cfg,
    )

    # ── One-line "ready" + start the demo timer ──────────────────────────────
    if persistent is not None:
        print(f"  {_G}✓{_E}  Connected to {_B}{resource_cfg.host}{_E}  "
              f"(persistent worker active on {_B}{persistent.hostname}{_E})")
    else:
        mode = "per-job PBS mode" if args.persistent_worker is False \
               else "per-job PBS mode (persistent worker unavailable)"
        print(f"  {_G}✓{_E}  Connected to {_B}{resource_cfg.host}{_E}  ({mode})")
    t_connected = time.monotonic()   # demo "total time" starts here

    # ── Specs overview ───────────────────────────────────────────────────────
    _print_banner(args, resource_cfg, remote_dir)
    print(f"  Experiment : {experiment_name}")
    print(f"  Local dir  : {experiment_dir}")
    print(f"  Remote dir : {remote_dir}/generated_codes/experiment/{experiment_name}\n")

    def _runner(worker_key: str, compiler_env: dict[str, str]):
        return build_runner(
            session=session, compute_resource=compute_resource, resource=args.resource,
            remote_dir=remote_dir, account=args.account, queue=args.queue,
            filesystems=resource_cfg.filesystems,
            worker_key=worker_key, compiler_env=compiler_env,
            walltime_s=args.walltime, persistent=persistent,
        )

    success = False

    # ── Dispatch ─────────────────────────────────────────────────────────────
    if args.task == "both":
        success = run_both_pipeline(
            model=args.model, resource=args.resource, resource_cfg=resource_cfg,
            session=session, compute_resource=compute_resource,
            remote_dir=remote_dir, account=args.account, queue=args.queue,
            run_args=run_args,
            cuda_input_code=input_code, original_cu_path=original_cu_path,
            experiment_dir=experiment_dir, experiment_name=experiment_name,
            csv_path=experiment_dir / "results.csv",
            run_id=run_id, walltime_s=args.walltime,
            sycl_compiler_env=sycl_env, sycl_energy_compiler_env=sycl_env,
            use_self_prompting=args.self_prompting,
            verifyprompt=args.verifyprompt, verbose=args.debug,
            min_energy_gain_pct=args.min_energy_gain,
            use_judge=args.use_judge, verification=verification,
            persistent=persistent,
        )

    elif args.task == "energy":
        if args.resource == "polaris" and args.input_framework == "cuda":
            worker_key, src_ext, comp_env = "energy_cuda", ".cu",  cuda_env
        else:
            worker_key, src_ext, comp_env = "energy_sycl", ".cpp", sycl_env

        runner = _runner(worker_key, comp_env)

        baseline_energy_j, baseline_power_w, baseline_stdout = run_baseline(
            session=session, runner=runner, remote_dir=remote_dir,
            original_src_path=original_cu_path, src_extension=src_ext,
            experiment_name=experiment_name, experiment_dir=experiment_dir,
            run_args=run_args, run_id=run_id + "b",
            verification=verification,
        )

        csv_path = experiment_dir / "results.csv"
        exp_data = {
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
            csv.DictWriter(f, fieldnames=list(exp_data.keys())).writeheader()

        _llm.configure(log_dir=str(experiment_dir / "llm_calls"),
                       truncate_chars=4000, debug=args.debug,
                       stream_display=True)

        success, _, _ = run_pipeline(
            model=args.model, task="energy", resource=args.resource,
            input_code=input_code,
            session=session, runner=runner,
            remote_dir=remote_dir, src_extension=src_ext,
            run_args=run_args,
            experiment_name=experiment_name, experiment_dir=experiment_dir,
            csv_path=csv_path, experiment_data=exp_data,
            trial_number=1, run_id=run_id,
            use_self_prompting=args.self_prompting,
            verifyprompt=args.verifyprompt, verbose=args.debug,
            baseline_energy_j=baseline_energy_j,
            baseline_power_w=baseline_power_w,
            baseline_stdout=baseline_stdout,
            min_energy_gain_pct=args.min_energy_gain,
            has_energy=True,
            use_judge=args.use_judge, verification=verification,
        )

    else:  # task == "translate"
        csv_path = experiment_dir / "results.csv"
        exp_data = {
            "timestamp":   dt_string,
            "app":         args.app_name,
            "model":       args.model,
            "gpu_vendor":  resource_cfg.gpu_vendor,
            "success":     False,
            "runtime_min": 0.0,
        }
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(exp_data.keys())).writeheader()

        _llm.configure(log_dir=str(experiment_dir / "llm_calls"),
                       truncate_chars=4000, debug=args.debug,
                       stream_display=True)

        # ── Test run of the ORIGINAL source (no EE — translate is not an
        # energy task; we only need stdout + duration for the comparison) ────
        baseline_metrics = None
        if can_run(args.resource, "cuda"):
            baseline_metrics = run_test_run(
                session=session,
                runner=_runner("energy_cuda", cuda_env),
                remote_dir=remote_dir,
                original_src_path=original_cu_path, src_extension=".cu",
                experiment_name=experiment_name, experiment_dir=experiment_dir,
                run_args=run_args, run_id=run_id + "b",
                verification=verification,
            )
        else:
            print(f"\n  {_B}Note:{_E} CUDA source cannot run on "
                  f"{args.resource} ({resource_cfg.gpu_vendor} GPU) — "
                  f"skipping test run of the original source.")

        # Translation uses the plain `translate` worker — no energy monitor.
        # measure_perf=True still reads energy_results.txt so we get the
        # duration field the worker now always writes.
        success, _, gen_metrics = run_pipeline(
            model=args.model, task="translate", resource=args.resource,
            input_code=input_code,
            session=session, runner=_runner("translate", sycl_env),
            remote_dir=remote_dir, src_extension=".cpp",
            run_args=run_args,
            experiment_name=experiment_name, experiment_dir=experiment_dir,
            csv_path=csv_path, experiment_data=exp_data,
            trial_number=1, run_id=run_id,
            use_self_prompting=args.self_prompting,
            verifyprompt=args.verifyprompt, verbose=args.debug,
            has_energy=False,
            use_judge=args.use_judge, verification=verification,
            measure_perf=True,
        )

        if success:
            print_translate_summary(
                resource=args.resource,
                original_framework=args.input_framework,
                target_framework="sycl",
                baseline=baseline_metrics,
                generated=gen_metrics,
                total_elapsed_s=time.monotonic() - t_connected,
                versioned_path=experiment_dir / f"Main_v1_Trial1.cpp",
                verification=verification,
            )

    if persistent:
        persistent.shutdown()
    if ssh_ctl:
        ssh_master_stop(ssh_ctl, resource_cfg.host, args.alcf_username)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
