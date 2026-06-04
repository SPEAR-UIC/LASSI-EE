"""ALCF connection setup: Globus (amsc_client) + persistent SSH ControlMaster.

Only used in remote mode and --submit mode (workstation side).  --on-resource
runs entirely on the compute node and doesn't need any of this.
"""
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from lassi.format_colors import B, E


def apply_ipv4_patch():
    """Prefer IPv4 (some Argonne workstations have broken IPv6 that hangs Globus)."""
    orig = socket.getaddrinfo
    def first(host, port, family=0, *a, **kw):
        r = orig(host, port, family, *a, **kw)
        v4 = [x for x in r if x[0] == socket.AF_INET]
        return v4 if v4 else r
    socket.getaddrinfo = first
    socket.setdefaulttimeout(30)


def amsc_connect(resource_name, fs_name, script_dir: Path, quiet=True):
    """Globus-auth and return (compute_resource, fs_resource).  First call opens
    a browser; subsequent calls reuse cached credentials.  Re-execs into a
    sibling .venv if amsc_client isn't importable in the current interpreter."""
    apply_ipv4_patch()
    try:
        from amsc_client import Client
    except ImportError:
        for parent in [script_dir, script_dir.parent]:
            cand = parent / ".venv" / "bin" / "python"
            if not cand.is_file(): continue
            try:
                if cand.samefile(sys.executable): continue
            except OSError: pass
            r = subprocess.run([str(cand), "-c", "import amsc_client"], capture_output=True)
            if r.returncode == 0:
                os.execv(str(cand), [str(cand)] + sys.argv)
        raise SystemExit("amsc_client not found. Run: pip install amsc-client")

    say = (lambda m: None) if quiet else (lambda m: print(m, flush=True))
    say("  Creating AmSC client...")
    client = Client(token="not-needed")
    alcf = client.facility("alcf")
    compute = alcf.resource(resource_name)
    try:
        fs = alcf.resource(fs_name)
    except Exception as exc:
        print(f"  {B}WARNING:{E} {fs_name} unavailable ({exc}) — using SSH fallback.")
        fs = None
    status = compute.status
    if status.lower() not in ("up", "available", "running"):
        print(f"  {B}WARNING:{E} {resource_name} reports '{status}'")
    return compute, fs


def ssh_master_start(host, user):
    """Open a persistent SSH ControlMaster connection; return its socket path."""
    ctl = str(Path(tempfile.mkdtemp(prefix="lassi_ssh_")) / "ctl")
    r = subprocess.run(["ssh", "-o", f"ControlPath={ctl}",
                        "-o", "ControlMaster=yes", "-o", "ControlPersist=yes",
                        "-N", "-f", f"{user}@{host}"])
    if r.returncode != 0: raise RuntimeError(f"SSH master to {host} failed (rc={r.returncode})")
    return ctl


def ssh_master_stop(ctl, host, user):
    """Cleanly shut down a ControlMaster opened by `ssh_master_start`."""
    subprocess.run(["ssh", "-o", f"ControlPath={ctl}", "-O", "exit", f"{user}@{host}"],
                   capture_output=True)
