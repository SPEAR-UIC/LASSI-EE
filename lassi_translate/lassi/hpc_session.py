"""amsc-client filesystem wrapper for talking to Polaris/Aurora from a workstation.

All read-side operations (mkdir, stat, read, rm, mv) go through the amsc fs API.
Uploads go through `rsync` over plain SSH because amsc has no upload API — that
is the only remaining SSH usage in the pipeline.
"""
import subprocess

from lassi.format_colors import R, E

_FS_TIMEOUT = 60


class HPCSession:
    """amsc fs + (upload-only) rsync wrapper.  `fs_resource` is required."""

    def __init__(self, host, user, fs_resource):
        if fs_resource is None:
            raise RuntimeError(
                "amsc filesystem resource is required.  This codebase no "
                "longer supports SSH fallback for FS operations.")
        self.host, self.user, self.fs_resource = host, user, fs_resource

    # ── amsc-backed FS operations ───────────────────────────────────────────

    def mkdir(self, path):
        self.fs_resource.fs.mkdir(path, parents=True).wait(timeout=_FS_TIMEOUT)

    def mkdirs(self, *paths):
        for p in paths: self.mkdir(p)

    def file_exists(self, path):
        try: self.fs_resource.fs.stat(path).wait(timeout=_FS_TIMEOUT); return True
        except Exception: return False

    def file_size(self, path):
        try:
            t = self.fs_resource.fs.stat(path); t.wait(timeout=_FS_TIMEOUT)
            r = t.result
            if isinstance(r, dict): return int(r.get("size", r.get("output", {}).get("size", 0)) or 0)
            return int(getattr(r, "size", 0) or 0)
        except Exception: return 0

    def read_file(self, path, max_lines=10000):
        try:
            t = self.fs_resource.fs.head(path, lines=max_lines); t.wait(timeout=_FS_TIMEOUT)
            r = t.result
            if isinstance(r, dict):
                return r.get("output", {}).get("content", "") or r.get("content", "") or str(r)
            return str(r) if r else ""
        except Exception as exc:
            return f"[FS read error {path!r}: {exc}]"

    def read_bytes(self, path, *, offset=0, size=None):
        """Read a byte range — used for tail-following the submit log."""
        kw = {"offset": offset}
        if size is not None: kw["size"] = size
        t = self.fs_resource.fs.view(path, **kw); t.wait(timeout=_FS_TIMEOUT)
        r = t.result
        if isinstance(r, dict):
            return r.get("output", {}).get("content", "") or r.get("content", "") or ""
        return str(r) if r else ""

    def rm(self, path):
        try: self.fs_resource.fs.rm(path).wait(timeout=_FS_TIMEOUT)
        except Exception: pass  # rm of nonexistent is fine

    def mv(self, src, dst):
        self.fs_resource.fs.mv(src, dst).wait(timeout=_FS_TIMEOUT)

    def ls(self, path):
        t = self.fs_resource.fs.ls(path); t.wait(timeout=_FS_TIMEOUT)
        r = t.result
        if isinstance(r, dict):
            return r.get("output", {}).get("entries", []) or r.get("entries", []) or []
        return r or []

    def download(self, path):
        """Return the bytes of a remote file via amsc.fs.download."""
        t = self.fs_resource.fs.download(path); t.wait(timeout=_FS_TIMEOUT)
        r = t.result
        if isinstance(r, dict):
            return r.get("output", {}).get("content", "") or r.get("content", "") or ""
        return str(r) if r else ""

    # ── Upload (rsync over plain SSH — amsc has no upload API) ──────────────

    def rsync(self, src, dst, is_dir=False, dereference=False):
        """Upload a file or directory via rsync.  Uses plain ssh transport with
        `accept-new` host-key policy (no ControlMaster)."""
        target  = f"{self.user}@{self.host}:{dst}"
        ssh_cmd = "ssh -o StrictHostKeyChecking=accept-new"
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

    # ── Path helper (no shell required) ─────────────────────────────────────

    def resolve_abs(self, path):
        """amsc requires absolute paths.  Reject relative paths with a clear error."""
        if path.startswith("/"): return path
        raise ValueError(
            f"Relative remote path {path!r} not supported.  Pass an absolute "
            f"path (e.g. /eagle/EE-ECP/LASSI-EE/lassi_jobs) via --remote-dir.")
