"""ALCF connection via Globus (amsc_client) — workstation side only.
Raises loudly if the filesystem resource isn't reachable; no SSH fallback."""
import os
import socket
import subprocess
import sys
from pathlib import Path


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
    """Globus-auth, return (compute, fs).  First call opens a browser; later
    calls reuse cached credentials.  Re-execs into sibling .venv if needed."""
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
    try: fs = alcf.resource(fs_name)
    except Exception as exc:
        raise RuntimeError(f"amsc cannot reach filesystem {fs_name!r}: {exc}. "
                           f"Fix the IRI registration or use --on-resource.") from exc
    status = compute.status
    if status.lower() not in ("up", "available", "running"):
        print(f"  WARNING: {resource_name} reports '{status}'")
    return compute, fs
