"""HPC connection layer — SSH ControlMaster + remote filesystem I/O."""
from __future__ import annotations

import shlex
import subprocess

from lassi.format_colors import color as _c

_EAGLE_READ_TIMEOUT = 60


class HPCSession:
    """Bundles SSH ControlMaster + remote filesystem access for one HPC resource.

    Provides a uniform interface over two backends:
      - amsc_client fs_resource  (Polaris / Eagle)
      - SSH fallback via ControlMaster (Aurora / Flare, or when amsc is unavailable)
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

    def _ssh(self, cmd: str, *, text: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-o", f"ControlPath={self.ssh_ctl}",
             f"{self.user}@{self.host}", cmd],
            capture_output=True, text=text,
        )

    def run(self, cmd: str) -> int:
        return self._ssh(cmd).returncode

    def run_capture(self, cmd: str) -> tuple[int, str]:
        """Run a command on the remote host; return (exit_code, stdout)."""
        r = self._ssh(cmd, text=True)
        return r.returncode, r.stdout

    def file_exists(self, remote_path: str) -> bool:
        return self._ssh(f"test -f {shlex.quote(remote_path)}").returncode == 0

    def read_file(self, remote_path: str) -> str:
        r = self._ssh(f"cat {shlex.quote(remote_path)}", text=True)
        return r.stdout if r.returncode == 0 else ""

    def resolve_abs(self, path: str) -> str:
        """Resolve `path` to its absolute path on the remote host.

        Uses `cd <path> && pwd` so it verifies the directory exists and follows
        symlinks the same way the shell would.  Falls back to the input if SSH
        capture fails (caller may then choose to error out separately).
        """
        if path.startswith("/"):
            return path
        rc, out = self.run_capture(f"cd {shlex.quote(path)} && pwd")
        if rc == 0 and out.strip():
            return out.strip()
        return path

    # ── rsync ──────────────────────────────────────────────────────────────────

    def rsync(self, local_src: str, remote_dst: str, is_dir: bool = False) -> bool:
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
            # Retry without --mkpath in case rsync version lacks it.
            result = subprocess.run(
                [c for c in cmd if c != "--mkpath"],
                capture_output=True, text=True,
            )
        if result.returncode != 0:
            print(f"  {_c.RED}rsync failed (rc={result.returncode}):{_c.END}")
            if result.stderr.strip():
                print(f"    {result.stderr.strip()[:400]}")
            return False
        return True

    # ── Remote filesystem ──────────────────────────────────────────────────────

    def mkdir(self, path: str) -> None:
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
        if self.fs_resource is None:
            if self.ssh_ctl:
                return self.read_file(path)
            return f"[Remote FS read error for {path!r}: fs_resource is None and no SSH fallback]"
        try:
            task = self.fs_resource.fs.head(path, lines=max_lines)
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
