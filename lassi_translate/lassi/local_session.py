"""Local-filesystem stand-in for HPCSession.

Drop-in replacement used in `--on-resource` mode: the pipeline thinks it's
talking to a remote machine via SSH/amsc, but every operation just touches
the local filesystem.  No Globus, no SSH ControlMaster.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path


class LocalSession:
    """HPCSession-shaped object backed by `pathlib` and `subprocess`.

    The pipeline code treats `remote_dir` as a path; here it's literally a
    local directory.  All methods mirror HPCSession's signatures so the rest
    of the codebase needs no awareness of which session is in play.
    """

    def __init__(self) -> None:
        # Matches HPCSession's public attributes so callers that read them
        # (e.g. JobRunner) don't need to branch.  Empty values say "no SSH".
        self.host        = "localhost"
        self.user        = ""
        self.ssh_ctl     = ""
        self.fs_resource = None

    # ── Shell primitives ─────────────────────────────────────────────────────

    def run(self, cmd: str) -> int:
        return subprocess.run(["bash", "-c", cmd]).returncode

    def run_capture(self, cmd: str) -> tuple[int, str]:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        return r.returncode, r.stdout

    def file_exists(self, path: str) -> bool:
        return Path(path).is_file()

    def read_file(self, path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception:
            return ""

    def resolve_abs(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        try:
            return str(p.resolve())
        except Exception:
            return str(p)

    # ── "Sync" — local copy ──────────────────────────────────────────────────

    def rsync(self, local_src: str, remote_dst: str, is_dir: bool = False) -> bool:
        """Copy `local_src` to `remote_dst` on the local filesystem.

        When `local_src` and `remote_dst` resolve to the same path (the common
        case in --on-resource mode), this is a no-op.
        """
        src = Path(local_src).resolve()
        dst = Path(remote_dst).resolve()
        if src == dst:
            return True  # already in place
        try:
            if is_dir:
                dst.mkdir(parents=True, exist_ok=True)
                # Mirror rsync -av --delete-style semantics with shutil.copytree
                # for new content; for an existing dst we walk and copy.
                for item in src.iterdir():
                    target = dst / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return True
        except Exception as exc:
            print(f"  local copy failed ({src} → {dst}): {exc}")
            return False

    # ── Filesystem ───────────────────────────────────────────────────────────

    def mkdir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def mkdirs(self, *paths: str) -> None:
        for p in paths:
            self.mkdir(p)

    def read_remote_file(self, path: str, max_lines: int = 10000) -> str:
        """Read a file from the local filesystem with optional line cap."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[Local FS read error for {path!r}: {exc}]"
        if max_lines and max_lines > 0:
            lines = text.splitlines(keepends=True)
            if len(lines) > max_lines:
                return "".join(lines[:max_lines])
        return text
