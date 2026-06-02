#!/usr/bin/env python3
"""Submit LASSI-translate pipeline to Polaris via the AmSC Python Client.

This script handles all prerequisites and submits run_polaris_demo.sh as a PBS
batch job to Polaris, replacing the manual `qsub -I` interactive workflow.

Prerequisites (handled automatically):
  - Syncs local lassi_translate code to eagle via rsync
  - Deploys inference_auth_token.py to ~/  on Polaris via scp
  - Ensures job_outputs/ directory exists on eagle via filesystem API
  - Python venv is created by run_polaris_demo.sh on first run

Usage:
  python submit_lassi_polaris.py [options]

Examples:
  # Basic submission with defaults
  python submit_lassi_polaris.py --alcf-username vchirio

  # Custom model, skip code sync (already deployed)
  python submit_lassi_polaris.py --alcf-username vchirio --model openai/gpt-oss-70b --skip-sync

  # No self-prompting, debug mode
  python submit_lassi_polaris.py --alcf-username vchirio --no-self-prompting --debug
"""
from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_POLARIS_DIR = "/eagle/EE-ECP/LASSI-EE/lassi_translate"
_DEFAULT_ALCF_VENV   = "/eagle/EE-ECP/LASSI-EE/venvs/inference"
_DEFAULT_MODEL       = "openai/gpt-oss-120b"
_DEFAULT_ACCOUNT     = "EE-ECP"
_DEFAULT_QUEUE       = "debug"
_DEFAULT_DURATION    = 3600          # 1 hour (debug queue maximum)
_DEFAULT_NODES       = 1
_DEFAULT_LOGIN_NODE  = "polaris-login-04.hsn.cm.polaris.alcf.anl.gov"
_POLARIS_HOST        = "polaris.alcf.anl.gov"

_SEP = "=" * 56
_THIN = "─" * 56


def _run_shell(cmd: list[str], step_label: str) -> bool:
    """Run a subprocess, stream output, return True on success."""
    print(f"      $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"\n  ERROR: {step_label} failed (exit {result.returncode}).\n"
            f"  Make sure SSH key auth is configured for {_POLARIS_HOST}.\n"
            f"  See: https://docs.alcf.anl.gov/polaris/getting-started/ssh-setup/\n",
            file=sys.stderr,
        )
        return False
    return True


def _sync_code(local_dir: Path, user: str, polaris_dir: str, skip: bool) -> bool:
    """rsync local lassi_translate/ to eagle."""
    print(f"\n[1/4] {'Skipping code sync (--skip-sync).' if skip else 'Syncing lassi_translate to eagle...'}")
    if skip:
        return True
    src = str(local_dir).rstrip("/") + "/"
    dst = f"{user}@{_POLARIS_HOST}:{polaris_dir}/"
    print(f"      rsync  {src}  →  {dst}")
    return _run_shell(
        ["rsync", "-av", "--delete",
         "--exclude=__pycache__", "--exclude=*.pyc",
         "--exclude=generated_codes/", "--exclude=.git/",
         "--exclude=job_outputs/",
         src, dst],
        "rsync",
    )


def _deploy_token(local_dir: Path, user: str, skip: bool) -> None:
    """scp inference_auth_token.py to ~/  on Polaris (non-fatal if missing)."""
    step = "[2/4]"
    if skip:
        print(f"\n{step} Skipping token deploy (--skip-sync).")
        return

    # Look for the token helper relative to this repo
    candidates = [
        local_dir / "inference_auth_token.py",
        local_dir.parent / "amsc-client-tutorial" / "inference_auth_token.py",
        local_dir.parent.parent / "amsc-client-tutorial" / "inference_auth_token.py",
    ]
    token_src = next((p for p in candidates if p.is_file()), None)

    print(f"\n{step} Deploying inference_auth_token.py to Polaris home...")
    if token_src is None:
        print(
            "  WARNING: inference_auth_token.py not found locally.\n"
            "  run_polaris_demo.sh will fail if it is also absent on Polaris.\n"
            "  Download it manually:\n"
            "    wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints"
            "/refs/heads/main/inference_auth_token.py -O ~/inference_auth_token.py\n"
            "  Then run: ssh polaris.alcf.anl.gov "
            "'python ~/inference_auth_token.py authenticate'"
        )
        return

    dst = f"{user}@{_POLARIS_HOST}:~/inference_auth_token.py"
    print(f"      scp  {token_src}  →  {dst}")
    ok = _run_shell(["scp", str(token_src), dst], "scp inference_auth_token.py")
    if not ok:
        print(
            "  WARNING: scp failed — continuing anyway.\n"
            "  The job will attempt to copy inference_auth_token.py from the\n"
            "  deployed script directory as a fallback.",
        )


def _ensure_job_outputs_dir(eagle, polaris_dir: str) -> None:
    """Create job_outputs/ on eagle if it does not exist."""
    outputs_dir = f"{polaris_dir}/job_outputs"
    try:
        eagle.fs.stat(outputs_dir).wait(timeout=30)
    except Exception:
        try:
            eagle.fs.mkdir(outputs_dir).wait(timeout=30)
        except Exception as exc:
            # Directory may have been created by a concurrent call — not fatal.
            print(f"  Note: mkdir({outputs_dir}) returned: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Deployment args ──────────────────────────────────────────────────────
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
        help=f"Absolute path to lassi_translate on eagle (default: {_DEFAULT_POLARIS_DIR})",
    )
    p.add_argument(
        "--alcf-venv",
        default=_DEFAULT_ALCF_VENV,
        metavar="PATH",
        help=f"Python venv path on Polaris (default: {_DEFAULT_ALCF_VENV})",
    )
    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip rsync and scp steps (code already deployed on eagle)",
    )

    # ── PBS job args ─────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"LLM model for inference (default: {_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--account",
        default=_DEFAULT_ACCOUNT,
        help=f"PBS account (default: {_DEFAULT_ACCOUNT})",
    )
    p.add_argument(
        "--queue",
        default=_DEFAULT_QUEUE,
        help=f"PBS queue (default: {_DEFAULT_QUEUE})",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=_DEFAULT_DURATION,
        help=f"Wall time in seconds (default: {_DEFAULT_DURATION})",
    )
    p.add_argument(
        "--nodes",
        type=int,
        default=_DEFAULT_NODES,
        help=f"Number of compute nodes (default: {_DEFAULT_NODES})",
    )
    p.add_argument(
        "--login-node",
        default=_DEFAULT_LOGIN_NODE,
        metavar="HOSTNAME",
        help=(
            "Polaris login node hostname for PBS_O_HOST override "
            f"(default: {_DEFAULT_LOGIN_NODE})"
        ),
    )

    # ── Pipeline pass-through args ───────────────────────────────────────────
    p.add_argument(
        "--no-self-prompting",
        action="store_true",
        help="Disable Stage 0 self-prompting (passed to run_lassi_translate_v3.py)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in pipeline (passed to run_lassi_translate_v3.py)",
    )

    args = p.parse_args()

    if not args.alcf_username:
        p.error("--alcf-username is required (could not determine from $USER)")

    local_dir = Path(__file__).resolve().parent

    # ── Step 1: Sync code ────────────────────────────────────────────────────
    if not _sync_code(local_dir, args.alcf_username, args.polaris_dir, args.skip_sync):
        return 1

    # ── Step 2: Deploy inference_auth_token.py ───────────────────────────────
    _deploy_token(local_dir, args.alcf_username, args.skip_sync)

    # ── Step 3: Connect via amsc_client ─────────────────────────────────────
    print("\n[3/4] Connecting to ALCF via Globus...")
    print("      (A browser window will open if Globus login is required.)")
    try:
        from amsc_client import Client
    except ImportError:
        # amsc_client not in this interpreter — find a .venv that actually has it.
        # Walk up to 3 directory levels looking for a .venv/bin/python that can
        # import amsc_client, skipping the current interpreter to avoid loops.
        script_dir = Path(__file__).resolve().parent

        for parent in [script_dir, script_dir.parent, script_dir.parent.parent]:
            candidate = parent / ".venv" / "bin" / "python"
            if not candidate.is_file():
                continue
            # Use samefile() (inode comparison) so symlinks don't fool us into
            # re-execing with the same binary we're already running.
            try:
                if candidate.samefile(sys.executable):
                    continue
            except OSError:
                pass
            check = subprocess.run(
                [str(candidate), "-c", "import amsc_client"],
                capture_output=True,
            )
            if check.returncode != 0:
                continue  # this venv also lacks amsc_client — keep searching

            # Found a good interpreter. Re-exec with --skip-sync so rsync/scp
            # don't repeat (they already completed in this process).
            new_argv = list(sys.argv)
            if "--skip-sync" not in new_argv:
                new_argv.insert(1, "--skip-sync")
            print(
                f"\n  amsc_client not in {sys.executable!r}.\n"
                f"  Re-running with: {candidate}\n"
            )
            os.execv(str(candidate), [str(candidate)] + new_argv)
            # execv replaces the process — unreachable

        print(
            f"\n  ERROR: amsc_client not found (tried {sys.executable}).\n"
            "  Install it with:  pip install amsc-client\n"
            "  Or activate the workspace venv:\n"
            "    source /path/to/lassi-workspace/.venv/bin/activate\n",
            file=sys.stderr,
        )
        return 1

    # The ALCF IRI API resolves to both IPv4 and IPv6 addresses, but IPv6
    # connectivity from some ANL workstations is broken — httpx tries addresses
    # in getaddrinfo order (IPv6 first) and hangs indefinitely because the
    # autogen client sets no socket timeout.  Patch getaddrinfo to prefer IPv4
    # and set a global socket timeout so any remaining stall fails quickly.
    import socket as _socket
    _orig_getaddrinfo = _socket.getaddrinfo
    def _ipv4_preferred_getaddrinfo(host, port, family=0, *args, **kwargs):
        results = _orig_getaddrinfo(host, port, family, *args, **kwargs)
        ipv4 = [r for r in results if r[0] == _socket.AF_INET]
        return ipv4 if ipv4 else results  # fall back to full list if no IPv4
    _socket.getaddrinfo = _ipv4_preferred_getaddrinfo
    _socket.setdefaulttimeout(30)

    print("      Creating client...", flush=True)
    client  = Client(token="not-needed")
    print("      Fetching ALCF facility...", flush=True)
    alcf    = client.facility("alcf")
    print("      Getting Polaris resource...", flush=True)
    try:
        polaris = alcf.resource("Polaris")
    except Exception as exc:
        print(
            f"\n  ERROR: Could not reach the ALCF IRI API within 30s: {exc}\n"
            "  The API endpoint https://api.alcf.anl.gov may be unreachable.\n"
            "  Check network/VPN connectivity, then retry.\n",
            file=sys.stderr,
        )
        return 1
    print("      Getting Eagle resource...", flush=True)
    try:
        eagle = alcf.resource("Eagle")
    except Exception as exc:
        print(f"  WARNING: Could not fetch Eagle resource: {exc} — skipping filesystem check.", flush=True)
        eagle = None
    print("      Checking Polaris status...", flush=True)
    polaris_status = polaris.status
    print(f"      Polaris: {polaris_status}", flush=True)
    if polaris_status.lower() not in ("up", "available", "running"):
        print(
            f"  WARNING: Polaris reports status '{polaris_status}'.\n"
            "  The job will be queued but may not start until the system is available.",
        )

    # ── Step 4: Ensure job_outputs/ exists on eagle ──────────────────────────
    print("\n[4/4] Ensuring job_outputs/ directory exists on eagle...", flush=True)
    if eagle is not None:
        _ensure_job_outputs_dir(eagle, args.polaris_dir)
        print("      job_outputs/ ready.", flush=True)
    else:
        print("      Skipped (Eagle resource unavailable).", flush=True)

    # ── Step 5: Submit PBS job ───────────────────────────────────────────────
    print("      Submitting PBS job...", flush=True)

    run_id   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"lassi-translate-{run_id}"

    passthrough: list[str] = []
    if args.no_self_prompting:
        passthrough.append("--no-self-prompting")
    if args.debug:
        passthrough.append("--debug")

    stdout_path = f"{args.polaris_dir}/job_outputs/{job_name}.stdout"
    stderr_path = f"{args.polaris_dir}/job_outputs/{job_name}.stderr"

    # 'environment' is not yet supported by the ALCF IRI API (returns 501).
    # Pass env vars as inline assignments in the shell command instead.
    env_prefix = " ".join([
        f"LLM_MODEL={shlex.quote(args.model)}",
        f"ALCF_VENV={shlex.quote(args.alcf_venv)}",
        f"POLARIS_LOGIN_NODE={shlex.quote(args.login_node)}",
    ])
    script_cmd = " ".join(
        [shlex.quote(f"{args.polaris_dir}/run_polaris_demo.sh")]
        + [shlex.quote(a) for a in passthrough]
    )

    job = polaris.submit(
        executable  = "/bin/bash",
        arguments   = ["-c", f"{env_prefix} exec {script_cmd}"],
        directory   = args.polaris_dir,
        name        = job_name,
        queue       = args.queue,
        account     = args.account,
        duration    = args.duration,
        nodes       = args.nodes,
        stdout_path = stdout_path,
        stderr_path = stderr_path,
        filesystems = "home:eagle",
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  Job submitted to Polaris")
    print(f"  {_THIN}")
    print(f"  Job ID   : {job.id}")
    print(f"  State    : {job.state}")
    print(f"  Queue    : {args.queue}  (account: {args.account})")
    print(f"  Model    : {args.model}")
    print(f"  Nodes    : {args.nodes}  |  Duration: {args.duration}s")
    print(f"  {_THIN}")
    print(f"  PBS stdout : {stdout_path}")
    print(f"  PBS stderr : {stderr_path}")
    print(f"  Results    : {args.polaris_dir}/generated_codes/experiment/")
    print(f"  {_THIN}")
    # Strip to the numeric part for cleaner qstat display
    pbs_id = job.id.split(".")[0]
    print(f"  Monitor  : ssh {_POLARIS_HOST} qstat {pbs_id}")
    print(_SEP)

    # ── Wait for job completion ──────────────────────────────────────────────
    wait_timeout = args.duration + 600   # walltime + 10-min queue/teardown buffer
    print(
        f"\n  Waiting for job to complete (up to {wait_timeout}s, polling every 15s)."
        "\n  Press Ctrl+C to stop waiting — the job will keep running on Polaris."
    )
    try:
        job.wait(timeout=wait_timeout, poll_interval=15)
    except TimeoutError:
        print(
            f"\n  WARNING: Job did not complete within {wait_timeout}s.\n"
            f"  Monitor : ssh {_POLARIS_HOST} qstat {pbs_id}\n"
            f"  Stdout  : {stdout_path}\n"
        )
        return 0
    except KeyboardInterrupt:
        print(f"\n  Interrupted. Job {job.id} is still running on Polaris.")
        return 0

    final_state = job.state
    exit_code   = job.exit_code
    ec_str      = f", exit code: {exit_code}" if exit_code is not None else ""
    print(f"\n  Job finished — state: {final_state}{ec_str}")

    # ── Stream output from eagle ─────────────────────────────────────────────
    if eagle is None:
        print(
            "\n  Eagle resource unavailable — cannot retrieve output files.\n"
            f"  Retrieve manually: ssh {_POLARIS_HOST} cat {stdout_path}"
        )
        return 0 if final_state in ("completed", "active") else 1

    def _print_file(path: str, label: str) -> None:
        try:
            task = eagle.fs.head(path, lines=10000)
            task.wait(timeout=120)
            result = task.result
            content = (
                result.get("output", {}).get("content", "")
                if isinstance(result, dict)
                else str(result)
            )
            print(f"\n{_THIN}\n  {label}: {path}\n{_THIN}")
            print(content, end="")
        except Exception as exc:
            print(f"\n  WARNING: could not read {label}: {exc}")

    _print_file(stdout_path, "stdout")
    _print_file(stderr_path, "stderr")

    return 0 if final_state in ("completed", "active") else 1


if __name__ == "__main__":
    raise SystemExit(main())
