"""Drop-in HPCSession replacement for --on-resource: stdlib FS calls, no amsc/SSH."""
import shutil
from pathlib import Path


class LocalSession:
    """Same interface as HPCSession; ops touch the local filesystem."""

    def __init__(self):
        self.host, self.user, self.fs_resource = "localhost", "", None

    # ── FS reads ────────────────────────────────────────────────────────────

    def mkdir(self, path):       Path(path).mkdir(parents=True, exist_ok=True)
    def mkdirs(self, *paths):    [self.mkdir(p) for p in paths]
    def file_exists(self, path): return Path(path).is_file()
    def file_size(self, path):
        try: return Path(path).stat().st_size
        except Exception: return 0

    def read_file(self, path, max_lines=10000):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[Local FS read error {path!r}: {exc}]"
        if max_lines and max_lines > 0:
            lines = text.splitlines(keepends=True)
            if len(lines) > max_lines: return "".join(lines[:max_lines])
        return text

    def read_bytes(self, path, *, offset=0, size=None):
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read() if size is None else f.read(size)
        return data.decode("utf-8", errors="replace")

    def ls(self, path):
        return [{"name": p.name, "type": "directory" if p.is_dir() else "file",
                 "size": str(p.stat().st_size) if p.is_file() else "0"}
                for p in Path(path).iterdir()]

    # ── FS writes ───────────────────────────────────────────────────────────

    def rm(self, path):
        try: Path(path).unlink()
        except Exception: pass

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

    def rsync_pull(self, remote_src, local_dst, is_dir=True):
        return self.rsync(remote_src, local_dst, is_dir=is_dir)

    def resolve_abs(self, path):
        try: return str(Path(path).resolve())
        except Exception: return str(Path(path))
