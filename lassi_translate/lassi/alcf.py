"""ALCF connection helpers: Globus (amsc_client) + persistent SSH ControlMaster."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from lassi.format_colors import color as _c


def apply_ipv4_patch() -> None:
    """Prefer IPv4 to avoid hangs with broken IPv6 on some ANL workstations."""
    orig = socket.getaddrinfo

    def _ipv4_first(host, port, family=0, *a, **kw):
        results = orig(host, port, family, *a, **kw)
        ipv4 = [r for r in results if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else results

    socket.getaddrinfo = _ipv4_first
    socket.setdefaulttimeout(30)


def amsc_connect(resource_name: str, fs_name: str, script_dir: Path, quiet: bool = True):
    """Connect to ALCF via Globus; return (compute_resource, fs_resource).

    On first call a browser window opens for Globus login; subsequent calls
    use cached credentials (~/.amsc/credentials.json).  `quiet=True` (default)
    suppresses step-by-step chatter for the demo UX — only warnings print.

    Re-execs into a sibling .venv if amsc_client is not importable in the
    current interpreter but is present in a project virtualenv.
    """
    apply_ipv4_patch()

    try:
        from amsc_client import Client
    except ImportError:
        for parent in [script_dir, script_dir.parent]:
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

    def _say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    _say("  Creating AmSC client...")
    client = Client(token="not-needed")
    _say("  Fetching ALCF facility...")
    alcf = client.facility("alcf")
    _say(f"  Getting {resource_name} resource...")
    compute = alcf.resource(resource_name)
    _say(f"  Getting {fs_name} resource...")
    try:
        fs = alcf.resource(fs_name)
    except Exception as exc:
        # Some resources (e.g. Flare/Aurora) are not registered in the IRI API;
        # the SSH fallback in HPCSession handles file operations.  Always warn,
        # quiet or not — the user should know SSH fallback is in play.
        print(f"  {_c.BOLD}WARNING:{_c.END} {fs_name} not found in IRI API ({exc}) — "
              f"filesystem operations will fall back to SSH.", flush=True)
        fs = None

    status = compute.status
    _say(f"  {resource_name} status: {status}")
    if status.lower() not in ("up", "available", "running"):
        print(f"  {_c.BOLD}WARNING:{_c.END} {resource_name} reports '{status}' — "
              f"job may queue longer.")

    return compute, fs


def ssh_master_start(host: str, user: str) -> str:
    """Open a persistent SSH ControlMaster connection; return the control socket path."""
    ctl_dir  = Path(tempfile.mkdtemp(prefix="lassi_ssh_"))
    ctl_path = str(ctl_dir / "ctl")
    r = subprocess.run([
        "ssh",
        "-o", f"ControlPath={ctl_path}",
        "-o", "ControlMaster=yes",
        "-o", "ControlPersist=yes",
        "-N", "-f",
        f"{user}@{host}",
    ])
    if r.returncode != 0:
        raise RuntimeError(f"SSH ControlMaster to {host} failed (rc={r.returncode})")
    return ctl_path


def ssh_master_stop(ctl_path: str, host: str, user: str) -> None:
    subprocess.run(
        ["ssh", "-o", f"ControlPath={ctl_path}", "-O", "exit", f"{user}@{host}"],
        capture_output=True,
    )
