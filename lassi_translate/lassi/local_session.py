"""Drop-in replacement for HPCSession when the pipeline runs directly on the
compute node (--on-resource / --submit). Same method names; backed by stdlib
filesystem calls instead of SSH/amsc."""
import shutil
import subprocess
from pathlib import Path


class LocalSession:
    """Same interface as HPCSession; ops touch the local filesystem."""

    def __init__(self):
        self.host, self.user, self.ssh_ctl, self.fs_resource = "localhost", "", "", None

    def run(self, cmd): return subprocess.run(["bash", "-c", cmd]).returncode
    def run_capture(self, cmd):
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        return r.returncode, r.stdout
    def file_exists(self, path): return Path(path).is_file()
    def read_file(self, path):
        try: return Path(path).read_text(encoding="utf-8")
        except Exception: return ""

    def resolve_abs(self, path):
        p = Path(path)
        try: return str(p.resolve())
        except Exception: return str(p)

    def rsync(self, src, dst, is_dir=False, dereference=False):
        src, dst = Path(src).resolve(), Path(dst).resolve()
        if src == dst: return True
        try:
            if is_dir:
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    tgt = dst / item.name
                    if item.is_dir():
                        shutil.copytree(item, tgt, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, tgt)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return True
        except Exception as exc:
            print(f"  local copy failed ({src} → {dst}): {exc}")
            return False

    def mkdir(self, path): Path(path).mkdir(parents=True, exist_ok=True)
    def mkdirs(self, *paths):
        for p in paths: self.mkdir(p)

    def read_remote_file(self, path, max_lines=10000):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[Local FS read error {path!r}: {exc}]"
        if max_lines and max_lines > 0:
            lines = text.splitlines(keepends=True)
            if len(lines) > max_lines: return "".join(lines[:max_lines])
        return text
