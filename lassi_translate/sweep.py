#!/usr/bin/env python3
"""Run a fleet of unattended pipeline invocations.

Default sweep (30 runs total):
    10 × remote mode + Argo gpt55
    10 × remote mode + Argo gpt41
    10 × --submit mode + ALCF Inference openai/gpt-oss-120b

How OTP-prompt deduplication works (for MobilePASS+ users without SSH keys):
    On startup, this script opens ONE persistent SSH ControlMaster connection
    to Polaris — you enter your MobilePASS OTP exactly once.  The socket path
    is exported as LASSI_SSH_CTL so every child `run_lassi.py` reuses it
    instead of opening its own master.  After the sweep finishes (or on
    Ctrl-C), the master is closed cleanly.

Each run still writes its own timestamp-based experiment dir under
generated_codes/experiment/ (results.csv, attempts.json, metadata, llm_calls/,
per-attempt logs — same as a single run).  This script just orchestrates the
30 invocations and records which experiment dir each run produced
(sweeps/<sweep_id>/manifest.json).

Per-run stdout/stderr → sweeps/<sweep_id>/run_<NN>_<label>.log.

Usage:
    python sweep.py                    # run the default sweep
    python sweep.py --dry-run          # print the plan + token check, exit
    python sweep.py --skip-preflight   # bypass the ALCF Inference token check

Before running, ensure the ALCF Inference token is fresh:
    python inference_auth_token.py authenticate   # only if `debug.py` tier 4 is red
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
EXP_BASE = ROOT / "generated_codes" / "experiment"
SWEEPS   = ROOT / "sweeps"

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; D = "\033[2m"; E = "\033[0m"

POLARIS_HOST = "polaris.alcf.anl.gov"
POLARIS_USER = os.environ.get("USER", "vchirio")

# Sweep plan: (label, extra_args).  --task/--resource etc. default from lassi_config.py.
SWEEP_PLAN = (
    [("remote-gpt55",   ["--argo-model",    "gpt55"])]                              * 10 +
    [("remote-gpt41",   ["--argo-model",    "gpt41"])]                              * 10 +
    [("submit-oss120b", ["--submit", "--inference-model", "openai/gpt-oss-120b"])]  * 10
)

# Politeness delay between runs (avoid hammering the LLM gateway).
INTER_RUN_SLEEP_S = 30


# ── Pre-flight: ALCF Inference token (Argo + SSH are checked live via master) ─

def _inference_token_valid() -> tuple[bool, str]:
    try:
        sys.path.insert(0, str(ROOT))
        from lassi.alcf_inference import _read_cached_access_token, verify_token
        tok = _read_cached_access_token()
        if not tok:
            return False, "no cached token (run `python inference_auth_token.py authenticate`)"
        ok, err = verify_token(tok)
        return (True, "valid") if ok else (False, err[:200])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _preflight(need_submit: bool) -> bool:
    print(f"{B}Pre-flight checks:{E}")
    print(f"  {D}info{E} Argo (remote mode): assumed reachable from this Argonne host")
    print(f"  {D}info{E} SSH to Polaris: will prompt for MobilePASS OTP when master opens")
    ok = True
    if need_submit:
        valid, msg = _inference_token_valid()
        mark = f"{G}OK{E}" if valid else f"{R}FAIL{E}"
        print(f"  {mark}   ALCF Inference token: {msg}")
        if not valid:
            print(f"  {Y}  → fix: python {ROOT}/inference_auth_token.py authenticate{E}")
            ok = False
    print()
    return ok


# ── Shared ControlMaster (one OTP prompt for the whole sweep) ────────────────

def _open_master(ctl_path: str) -> bool:
    """Open a persistent SSH ControlMaster to Polaris.  Prompts for the
    MobilePASS OTP once, then daemonizes and returns."""
    print(f"{B}Opening shared SSH ControlMaster to {POLARIS_HOST}...{E}")
    print(f"  {D}(you'll be prompted for your MobilePASS OTP once here){E}\n")
    Path(ctl_path).parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call([
        "ssh", "-N", "-f",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={ctl_path}",
        "-o", "ControlPersist=yes",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{POLARIS_USER}@{POLARIS_HOST}",
    ])
    if rc != 0:
        print(f"{R}Failed to open SSH master (rc={rc}).{E}")
        return False
    # Verify the master is usable
    chk = subprocess.run(
        ["ssh", "-o", f"ControlPath={ctl_path}", f"{POLARIS_USER}@{POLARIS_HOST}",
         "hostname"], capture_output=True, text=True, timeout=15)
    if chk.returncode != 0:
        print(f"{R}Master opened but unreachable: {chk.stderr.strip()[:200]}{E}")
        return False
    print(f"  {G}✓{E}  Master live on {chk.stdout.strip()}\n")
    return True


def _close_master(ctl_path: str):
    if not os.path.exists(ctl_path): return
    try:
        subprocess.run(
            ["ssh", "-o", f"ControlPath={ctl_path}", "-O", "exit",
             f"{POLARIS_USER}@{POLARIS_HOST}"],
            capture_output=True, timeout=5)
    except Exception: pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _snapshot_exp_dirs() -> set[str]:
    if not EXP_BASE.exists(): return set()
    return {d.name for d in EXP_BASE.iterdir() if d.is_dir()}


def _new_exp_dirs(before: set[str]) -> list[str]:
    return sorted(_snapshot_exp_dirs() - before)


def _print_plan(plan):
    print(f"{B}Sweep plan ({len(plan)} runs total):{E}")
    counts: dict = {}
    for label, _ in plan: counts[label] = counts.get(label, 0) + 1
    for label, n in counts.items(): print(f"  {n:>3}× {label}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",        action="store_true",
                    help="print the plan + pre-flight, don't open master or run")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="bypass the inference token check (use with caution)")
    ap.add_argument("--sleep-between",  type=int, default=INTER_RUN_SLEEP_S,
                    help=f"seconds between runs (default {INTER_RUN_SLEEP_S})")
    args = ap.parse_args()

    _print_plan(SWEEP_PLAN)
    need_submit = any("--submit" in a for _, a in SWEEP_PLAN)

    if args.dry_run:
        _preflight(need_submit)
        print(f"{D}(dry-run; not executing){E}")
        return 0

    if not args.skip_preflight:
        if not _preflight(need_submit):
            print(f"{R}Pre-flight failed.  Re-run with --skip-preflight to bypass.{E}")
            return 2

    sweep_id  = datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    sweep_dir = SWEEPS / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    # ControlMaster socket lives in /tmp because Unix-domain socket paths are
    # capped at ~108 chars on Linux and SSH appends a temp suffix during the
    # multiplex handshake — putting the socket in the project dir overflows.
    ctl_path = str(Path(tempfile.mkdtemp(prefix="lassi_sweep_")) / "ctl")
    (sweep_dir / "ssh_ctl_path.txt").write_text(ctl_path + "\n")

    if need_submit and not _open_master(ctl_path):
        return 2

    child_env = os.environ.copy()
    child_env["LASSI_SSH_CTL"] = ctl_path

    print(f"{B}Sweep ID:{E} {sweep_id}")
    print(f"  Per-run logs:        {sweep_dir}")
    print(f"  Manifest (live):     {sweep_dir / 'manifest.json'}")
    print(f"  Shared SSH socket:   {ctl_path}  ({D}path also in ssh_ctl_path.txt{E})")
    print(f"  Experiment dirs go:  {EXP_BASE}/exp_*\n")

    manifest = []
    sweep_started = time.time()
    try:
        for i, (label, extra_args) in enumerate(SWEEP_PLAN, start=1):
            log_path = sweep_dir / f"run_{i:02d}_{label}.log"
            before   = _snapshot_exp_dirs()
            cmd      = [sys.executable, "run_lassi.py", *extra_args]
            t0 = time.time()
            print(f"[{i:02d}/{len(SWEEP_PLAN)}] {label}  →  {log_path.name}", flush=True)
            with log_path.open("w") as f:
                f.write(f"# sweep_id : {sweep_id}\n")
                f.write(f"# run_idx  : {i}/{len(SWEEP_PLAN)}\n")
                f.write(f"# label    : {label}\n")
                f.write(f"# started  : {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"# cmd      : {' '.join(cmd)}\n\n")
                f.flush()
                try:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                         cwd=ROOT, env=child_env).returncode
                except Exception as exc:
                    rc = -1
                    f.write(f"\n# launcher exception: {type(exc).__name__}: {exc}\n")

            elapsed  = time.time() - t0
            new_dirs = _new_exp_dirs(before)
            manifest.append({
                "run": i, "label": label, "args": extra_args,
                "rc": rc, "elapsed_s": round(elapsed, 1),
                "started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                "exp_dirs": new_dirs,
                "log": log_path.name,
            })
            status = f"{G}✓{E}" if rc == 0 else f"{R}✗ rc={rc}{E}"
            exps   = ", ".join(new_dirs) or "(no new dir)"
            print(f"     {status}  elapsed={elapsed:.0f}s  exp_dirs=[{exps}]")
            (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            if i < len(SWEEP_PLAN) and args.sleep_between > 0:
                time.sleep(args.sleep_between)

    except KeyboardInterrupt:
        print(f"\n  {Y}Interrupted.  Manifest preserved through last completed run.{E}")
        (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _close_master(ctl_path)
        return 130

    _close_master(ctl_path)

    total_min = (time.time() - sweep_started) / 60
    ok = sum(1 for m in manifest if m["rc"] == 0)
    print(f"\n{B}Sweep complete.{E}  {ok}/{len(manifest)} runs succeeded.")
    print(f"  Total time: {total_min:.1f} min")
    print(f"  Manifest:   {sweep_dir / 'manifest.json'}")
    print(f"  All experiment dirs: {EXP_BASE}/exp_*  (see manifest for run→dir mapping)")
    return 0 if ok == len(manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
