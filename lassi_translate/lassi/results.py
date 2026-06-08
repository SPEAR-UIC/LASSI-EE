"""Parse per-attempt result files written by the worker scripts on the compute node."""
import json
import re
from pathlib import Path

_INT_RE        = re.compile(r"-?\d+")
_THROUGHPUT_RE = re.compile(r"Lookups/s:\s*([\d,]+)")


def xsbench_checksum_status(stdout):
    """Return (line, is_valid) for the XSBench 'Verification checksum:' line."""
    for line in (stdout or "").splitlines():
        if "Verification checksum:" in line:
            line = line.strip()
            return line, "(Valid)" in line
    return "", False


def parse_throughput(stdout):
    """Return (lookups_per_second, formatted_str) from XSBench 'Lookups/s:' line.
    (-1, '') if not present (e.g. non-XSBench apps)."""
    m = _THROUGHPUT_RE.search(stdout or "")
    if not m: return -1, ""
    return int(m.group(1).replace(",", "")), m.group(1)


def tail5(text):
    """Last 5 non-blank lines, indented — used for compile/runtime error previews."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    return "\n".join(f"    {l}" for l in lines[-5:])


def parse_attempt_results(session, results_dir, has_energy=False):
    """Read build/run rc + logs (and energy if requested); missing files → ""."""
    def rfile(f, lim=10000):
        try: return session.read_file(f"{results_dir}/{f}", lim)
        except Exception: return ""
    def rc(f):
        raw = rfile(f, lim=5).strip()
        return int(raw) if _INT_RE.fullmatch(raw) else -1

    out = {
        "build_ok":     rc("build_rc.txt") == 0,
        "run_ok":       rc("run_rc.txt") == 0,
        "build_stdout": rfile("build_stdout.log"),
        "build_stderr": rfile("build_stderr.log"),
        "run_stdout":   rfile("run_stdout.log"),
        "run_stderr":   rfile("run_stderr.log"),
    }
    if has_energy:
        try: e = json.loads(rfile("energy_results.txt"))
        except Exception: e = {}
        out.update({
            "energy_joules": e.get("energy_joules", -1.0),
            "avg_power_w":   e.get("avg_power_w",   -1.0),
            "duration_s":    e.get("duration_s",    -1.0),
            "num_samples":   e.get("num_samples",    0),
        })
    return out


def save_attempt_logs(results, experiment_dir: Path, prefix: str):
    """Mirror the four build/run logs from `results` into `experiment_dir/<prefix>*.log`."""
    for key, fname in [("build_stdout","build_stdout.log"),
                       ("build_stderr","build_stderr.log"),
                       ("run_stdout","run_stdout.log"),
                       ("run_stderr","run_stderr.log")]:
        content = results.get(key) or ""
        if content:
            (experiment_dir / (prefix + fname)).write_text(content, encoding="utf-8")
