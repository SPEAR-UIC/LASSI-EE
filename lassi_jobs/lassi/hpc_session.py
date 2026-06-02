"""HPC connection layer — SSH ControlMaster + remote filesystem I/O."""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

_EAGLE_READ_TIMEOUT = 60


@dataclass(frozen=True)
class ResourceConfig:
    name:          str
    host:          str
    amsc_resource: str
    amsc_fs:       str
    default_dir:   str
    filesystems:   str
    gpu_vendor:    str


class HPCSession:
    """Bundles SSH ControlMaster + remote filesystem access for one HPC resource.

    Provides a uniform interface over two backends:
      - amsc_client fs_resource  (Polaris / Eagle)
      - SSH fallback via ControlMaster (Aurora / Flare, or when amsc is unavailable)

    Construct once in main(); pass a single `session` argument instead of
    threading ssh_ctl, host, user, and fs_resource through every function.
    """

    def __init__(
        self,
        host: str,
        user: str,
        ssh_ctl: str,
        fs_resource,         # amsc fs resource object, or None
    ) -> None:
        self.host        = host
        self.user        = user
        self.ssh_ctl     = ssh_ctl
        self.fs_resource = fs_resource

    # ── SSH primitives ─────────────────────────────────────────────────────────

    def run(self, cmd: str) -> int:
        """Run a command on the remote host. Returns exit code."""
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={self.ssh_ctl}",
             f"{self.user}@{self.host}", cmd],
            capture_output=True,
        )
        return r.returncode

    def file_exists(self, remote_path: str) -> bool:
        """Return True if the file exists on the remote host."""
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={self.ssh_ctl}",
             f"{self.user}@{self.host}",
             f"test -f {shlex.quote(remote_path)}"],
            capture_output=True,
        )
        return r.returncode == 0

    def read_file(self, remote_path: str) -> str:
        """Read a small file from the remote host. Returns content or ''."""
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={self.ssh_ctl}",
             f"{self.user}@{self.host}",
             f"cat {shlex.quote(remote_path)}"],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else ""

    # ── rsync ──────────────────────────────────────────────────────────────────

    def rsync(
        self,
        local_src: str,
        remote_dst: str,
        is_dir: bool = False,
    ) -> bool:
        """rsync local_src to remote_dst. Returns True on success."""
        dst     = f"{self.user}@{self.host}:{remote_dst}"
        ssh_cmd = f"ssh -o ControlPath={self.ssh_ctl}" if self.ssh_ctl else "ssh"
        cmd     = ["rsync", "-av", "--mkpath", "-e", ssh_cmd]
        if is_dir:
            local_src = local_src.rstrip("/") + "/"
            cmd += [
                "--delete",
                "--exclude=__pycache__/", "--exclude=*.pyc",
                "--exclude=generated_codes/",
                "--exclude=.git/",
            ]
        cmd += [local_src, dst]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            cmd2   = [c for c in cmd if c != "--mkpath"]
            result = subprocess.run(cmd2, capture_output=True, text=True)

        if result.returncode != 0:
            from lassi.format_colors import color as _c
            print(f"  {_c.RED}rsync failed (rc={result.returncode}):{_c.END}")
            if result.stderr.strip():
                print(f"    {result.stderr.strip()[:400]}")
            return False
        return True

    # ── Remote filesystem ──────────────────────────────────────────────────────

    def mkdir(self, path: str) -> None:
        """Create a directory on the remote filesystem (non-fatal if exists)."""
        if self.fs_resource is None:
            if self.ssh_ctl:
                self.run(f"mkdir -p {shlex.quote(path)}")
            return
        try:
            self.fs_resource.fs.stat(path).wait(timeout=20)
            return
        except Exception:
            pass
        try:
            self.fs_resource.fs.mkdir(path).wait(timeout=30)
        except Exception:
            pass

    def mkdirs(self, *paths: str) -> None:
        for path in paths:
            self.mkdir(path)

    def read_remote_file(self, path: str, max_lines: int = 10000) -> str:
        """Read a file from the remote filesystem; return content as a string."""
        if self.fs_resource is None:
            if self.ssh_ctl:
                return self.read_file(path)
            return f"[Remote FS read error for {path!r}: fs_resource is None and no SSH fallback]"
        try:
            task   = self.fs_resource.fs.head(path, lines=max_lines)
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
            return f"[Remote FS read error for {path!r}: {exc}]"
