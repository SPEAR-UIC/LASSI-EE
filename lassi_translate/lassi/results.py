"""Per-attempt result parsing and log archival."""
from __future__ import annotations

import json
import re
from pathlib import Path

from lassi.hpc_session import HPCSession


def xsbench_checksum_status(stdout: str) -> tuple[str, bool]:
    """Return (checksum_line, is_valid) extracted from XSBench stdout.

    Returns ("", False) if no checksum line is present.
    """
    for line in (stdout or "").splitlines():
        if "Verification checksum:" in line:
            line = line.strip()
            return line, "(Valid)" in line
    return "", False


def tail5(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(f"    {ln}" for ln in lines[-5:])


_INT_RE = re.compile(r"-?\d+")


def parse_attempt_results(
    session: HPCSession,
    results_remote_dir: str,
    has_energy: bool = False,
) -> dict:
    """Read compile/run (and optionally energy) result files from the remote FS."""
    def rfile(fname: str, max_lines: int = 10000) -> str:
        return session.read_remote_file(f"{results_remote_dir}/{fname}", max_lines)

    def rc(fname: str) -> int:
        raw = rfile(fname, max_lines=5).strip()
        return int(raw) if _INT_RE.fullmatch(raw) else -1

    result = {
        "build_ok":     rc("build_rc.txt") == 0,
        "run_ok":       rc("run_rc.txt")   == 0,
        "build_stdout": rfile("build_stdout.log"),
        "build_stderr": rfile("build_stderr.log"),
        "run_stdout":   rfile("run_stdout.log"),
        "run_stderr":   rfile("run_stderr.log"),
    }

    if has_energy:
        try:
            energy = json.loads(rfile("energy_results.txt"))
        except Exception:
            energy = {}
        result.update({
            "energy_joules": energy.get("energy_joules", -1.0),
            "avg_power_w":   energy.get("avg_power_w",   -1.0),
            "duration_s":    energy.get("duration_s",    -1.0),
            "num_samples":   energy.get("num_samples",    0),
        })

    return result


def save_attempt_logs(results: dict, experiment_dir: Path, prefix: str) -> None:
    """Mirror remote result files to the local experiment directory."""
    for key, fname in [
        ("build_stdout", "build_stdout.log"),
        ("build_stderr", "build_stderr.log"),
        ("run_stdout",   "run_stdout.log"),
        ("run_stderr",   "run_stderr.log"),
    ]:
        content = results.get(key) or ""
        if content and not content.startswith("[Remote FS read error"):
            (experiment_dir / (prefix + fname)).write_text(content, encoding="utf-8")
