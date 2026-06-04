"""SSH + amsc filesystem wrapper for talking to Polaris/Aurora from a workstation.

Used by remote mode and by --submit's workstation half.  Methods like
`mkdir`/`read_remote_file` prefer the amsc fs API when available, falling
back to SSH-over-ControlMaster otherwise.
"""
import shlex
import subprocess

from lassi.format_colors import R, E

_READ_TIMEOUT = 60


class HPCSession:
    """Remote-FS + remote-shell access via SSH ControlMaster (+ optional amsc fs)."""

    def __init__(self, host, user, ssh_ctl, fs_resource):
        self.host, self.user, self.ssh_ctl, self.fs_resource = host, user, ssh_ctl, fs_resource

    def _ssh(self, cmd, text=False):
        return subprocess.run(
            ["ssh", "-o", f"ControlPath={self.ssh_ctl}",
             f"{self.user}@{self.host}", cmd],
            capture_output=True, text=text,
        )

    def run(self, cmd): return self._ssh(cmd).returncode
    def run_capture(self, cmd):
        r = self._ssh(cmd, text=True); return r.returncode, r.stdout
    def file_exists(self, path): return self._ssh(f"test -f {shlex.quote(path)}").returncode == 0

    def read_file(self, path):
        r = self._ssh(f"cat {shlex.quote(path)}", text=True)
        return r.stdout if r.returncode == 0 else ""

    def resolve_abs(self, path):
        """Turn a relative remote path into an absolute one (amsc fs API
        rejects relative paths, so every call site needs absolute)."""
        if path.startswith("/"): return path
        rc, out = self.run_capture(f"cd {shlex.quote(path)} && pwd")
        return out.strip() if rc == 0 and out.strip() else path

    def rsync(self, src, dst, is_dir=False, dereference=False):
        target  = f"{self.user}@{self.host}:{dst}"
        ssh_cmd = f"ssh -o ControlPath={self.ssh_ctl}" if self.ssh_ctl else "ssh"
        cmd = ["rsync", "-av", "--mkpath", "-e", ssh_cmd]
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

    def mkdir(self, path):
        if self.fs_resource is None:
            if self.ssh_ctl: self.run(f"mkdir -p {shlex.quote(path)}")
            return
        try: self.fs_resource.fs.stat(path).wait(timeout=20); return
        except Exception: pass
        try: self.fs_resource.fs.mkdir(path).wait(timeout=30)
        except Exception: pass

    def mkdirs(self, *paths):
        for p in paths: self.mkdir(p)

    def read_remote_file(self, path, max_lines=10000):
        if self.fs_resource is None:
            return self.read_file(path) if self.ssh_ctl else f"[FS error: no backend for {path!r}]"
        try:
            task = self.fs_resource.fs.head(path, lines=max_lines)
            task.wait(timeout=_READ_TIMEOUT)
            r = task.result
            if isinstance(r, dict):
                return r.get("output", {}).get("content", "") or r.get("content", "") or str(r)
            return str(r) if r else ""
        except Exception as exc:
            return f"[FS read error {path!r}: {exc}]"
