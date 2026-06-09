"""amsc-client filesystem wrapper for Polaris/Aurora.

Every wrapper calls `fs.X`, waits for the Task, requires state == 'completed',
then extracts result (dict-or-object).  Failures raise FsTaskError.

ALCF IRI fs endpoints (empirically verified via tools/amsc_probe.py):
    ls / mkdir / head / view / rm  ✓ work
    stat / mv / download           ✗ 501 NOT IMPLEMENTED — we work around via
                                     ls-of-parent (stat), atomic rsync (mv),
                                     and rsync_pull (download)

Uploads + downloads use rsync over SSH (amsc has no upload API; download is 501).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from lassi.format_colors import R, E

_FS_TIMEOUT = 60   # seconds


class FsTaskError(RuntimeError):
    """Raised when an amsc fs task reaches a non-completed terminal state."""


def _result(task) -> Any:
    """Extract result payload (dict from `additional_properties`, or raw)."""
    raw = task.result
    if raw is None: return None
    if hasattr(raw, "additional_properties") and raw.additional_properties:
        return raw.additional_properties
    return raw


def _await(task, *, op: str, path: str, timeout: int = _FS_TIMEOUT):
    """Wait + require state == 'completed', else raise FsTaskError with payload."""
    task.wait(timeout=timeout)
    if task.state == "completed":
        return _result(task)
    payload = _result(task)
    msg = payload if isinstance(payload, (str, dict)) else repr(payload)
    raise FsTaskError(f"amsc {op}({path!r}) ended in state={task.state!r}: {msg}")


class HPCSession:
    """amsc fs + (upload-only) rsync wrapper.  `fs_resource` is required."""

    def __init__(self, host, user, fs_resource):
        if fs_resource is None:
            raise RuntimeError("amsc filesystem resource is required (no SSH fallback).")
        self.host, self.user, self.fs_resource = host, user, fs_resource
        self._fs = fs_resource.fs
        # ControlMaster socket: reuse the parent's (set by sweep.py) if
        # LASSI_SSH_CTL is exported — otherwise create our own.  Sharing a
        # socket across multiple Python processes means only ONE OTP prompt
        # across an entire unattended sweep.
        parent_ctl = os.environ.get("LASSI_SSH_CTL")
        if parent_ctl:
            self._ssh_ctl, self._owns_master = parent_ctl, False
        else:
            self._ssh_ctl = str(Path(tempfile.mkdtemp(prefix="lassi_ssh_")) / "ctl")
            self._owns_master = True
            import atexit; atexit.register(self.close)

    def close(self):
        """Shut down the ControlMaster — only if we created it (don't close
        a parent-owned socket, e.g. the sweep's shared master)."""
        if not self._ssh_ctl or not self._owns_master: return
        try:
            subprocess.run(
                ["ssh", "-o", f"ControlPath={self._ssh_ctl}",
                 "-O", "exit", f"{self.user}@{self.host}"],
                capture_output=True, timeout=5)
        except Exception: pass
        self._ssh_ctl = ""

    # ── Directory ops ───────────────────────────────────────────────────────

    def mkdir(self, path: str) -> None:
        """Create directory (with parents).  Idempotent — `mkdir -p` semantics."""
        try:
            _await(self._fs.mkdir(path, parents=True), op="mkdir", path=path)
        except Exception as exc:
            if "already exists" in str(exc).lower(): return
            raise

    def mkdirs(self, *paths) -> None:
        for p in paths: self.mkdir(p)

    def ls(self, path: str) -> list[dict]:
        """List directory entries.  ALCF IRI returns them under `result['output']`."""
        r = _await(self._fs.ls(path), op="ls", path=path)
        if isinstance(r, dict) and isinstance(r.get("output"), list): return r["output"]
        return r if isinstance(r, list) else []

    @staticmethod
    def _entry_field(entry, field):
        return entry.get(field) if isinstance(entry, dict) else getattr(entry, field, None)

    def file_exists(self, path: str) -> bool:
        """Exists check via ls (fs.stat is 501 on ALCF)."""
        parent, _, name = path.rpartition("/")
        if not parent: return False
        try: entries = self.ls(parent)
        except Exception: return False
        return any(self._entry_field(e, "name") == name for e in entries)

    def file_size(self, path: str) -> int:
        """Size in bytes via ls-of-parent (fs.stat is 501 on ALCF); 0 if missing."""
        parent, _, name = path.rpartition("/")
        if not parent: return 0
        try: entries = self.ls(parent)
        except Exception: return 0
        for e in entries:
            if self._entry_field(e, "name") != name: continue
            try: return int(self._entry_field(e, "size"))
            except (TypeError, ValueError): return 0
        return 0

    # ── File reads ──────────────────────────────────────────────────────────

    def _extract_content(self, r: Any) -> str:
        """Pull text content from an fs.head / fs.view result."""
        if r is None: return ""
        if isinstance(r, (str, bytes)):
            return r.decode("utf-8", "replace") if isinstance(r, bytes) else r
        candidates = [r] if isinstance(r, dict) else []
        if candidates and isinstance(r.get("output"), dict): candidates.append(r["output"])
        for d in candidates:
            for k in ("content", "text", "data"):
                v = d.get(k)
                if isinstance(v, (str, bytes)):
                    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v
        return str(r)

    def read_file(self, path: str, max_lines: int = 10000) -> str:
        """Read first `max_lines` lines via fs.head; raises FsTaskError on failure."""
        return self._extract_content(_await(self._fs.head(path, lines=max_lines),
                                             op="head", path=path))

    def read_bytes(self, path: str, *, offset: int = 0, size: int | None = None) -> str:
        """Byte-range read via fs.view (used to tail-follow growing logs)."""
        kw = {"offset": offset} | ({"size": size} if size is not None else {})
        return self._extract_content(_await(self._fs.view(path, **kw),
                                             op="view", path=path))

    # ── File writes / removes ───────────────────────────────────────────────

    def rm(self, path: str) -> None:
        """Delete a file; missing files are not an error (idempotent)."""
        from amsc_client.core.exceptions import NotFoundError
        try: _await(self._fs.rm(path), op="rm", path=path)
        except NotFoundError: pass
        except Exception as exc:
            m = str(exc).lower()
            if any(s in m for s in ("does not exist", "no such file", "not found")): return
            raise

    # ── Upload (rsync over plain SSH — amsc has no upload API) ──────────────

    @property
    def _ssh_transport(self) -> str:
        """ssh -e command for rsync; opens ControlMaster on first use, reused
        after.  ControlPersist=yes (master survives idle gaps between rsync
        calls, e.g. while a PBS job runs); ServerAliveInterval keeps the TCP
        connection alive against server-side timeouts.  Cleanup via atexit."""
        return (f"ssh -o StrictHostKeyChecking=accept-new "
                f"-o ControlMaster=auto -o ControlPath={self._ssh_ctl} "
                f"-o ControlPersist=yes "
                f"-o ServerAliveInterval=60 -o ServerAliveCountMax=10")

    def rsync(self, src, dst, is_dir: bool = False, dereference: bool = False) -> bool:
        """Upload (workstation → remote)."""
        target = f"{self.user}@{self.host}:{dst}"
        cmd = ["rsync", "-av", "--mkpath", "-e", self._ssh_transport]
        if dereference: cmd.append("-L")
        if is_dir:
            src = src.rstrip("/") + "/"
            cmd += ["--delete", "--exclude=__pycache__/", "--exclude=*.pyc",
                    "--exclude=generated_codes/", "--exclude=.git/"]
        cmd += [src, target]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run([c for c in cmd if c != "--mkpath"], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {R}rsync failed (rc={r.returncode}): {r.stderr.strip()[:300]}{E}")
            return False
        return True

    def rsync_pull(self, remote_src: str, local_dst: str, is_dir: bool = True) -> bool:
        """Download (remote → workstation) via the same ControlMaster as `rsync`."""
        src = remote_src.rstrip("/") + ("/" if is_dir else "")
        dst = str(local_dst).rstrip("/") + ("/" if is_dir else "")
        cmd = ["rsync", "-az", "-e", self._ssh_transport,
               f"{self.user}@{self.host}:{src}", dst]
        r = subprocess.run(cmd)
        return r.returncode == 0

    # ── Path helper ─────────────────────────────────────────────────────────

    def resolve_abs(self, path: str) -> str:
        """amsc requires absolute paths; reject relative paths with a clear error."""
        if path.startswith("/"): return path
        raise ValueError(f"Relative remote path {path!r} not supported — pass an "
                         f"absolute path (e.g. /eagle/...) via --remote-dir.")
