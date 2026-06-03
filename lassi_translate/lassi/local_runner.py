"""Local subprocess runner — drop-in for JobRunner in --on-resource mode.

When the pipeline is already executing inside a Polaris/Aurora compute job
(e.g. via `qsub -I`), there's no need for PBS submission or the persistent
worker.  This runner just shells out to the worker script the same way
PersistentWorker would, only directly.
"""
from __future__ import annotations

import os
import subprocess

from lassi.format_colors import color as _c
from lassi.resources     import WORKERS

_G, _R, _E = _c.GREEN, _c.RED, _c.END


class LocalRunner:
    """One compile+run task via a direct `bash worker.sh ...` invocation.

    Matches JobRunner.run() signature so run_pipeline doesn't branch.  The
    worker script writes build_rc.txt / run_rc.txt / *.log into `results_dir`
    exactly as it does under PBS — `parse_attempt_results` reads them via
    LocalSession the same way it'd read them via HPCSession.
    """

    def __init__(
        self, *,
        worker_script: str, worker_task_arg: str,
        compiler_env: dict[str, str],
    ) -> None:
        self._worker_script = worker_script
        self._task_arg      = worker_task_arg
        self._compiler_env  = compiler_env

    def run(
        self, *,
        src_remote: str, results_dir: str,
        run_args: list[str], job_name: str,
        verbose: bool = False,
    ) -> None:
        env = os.environ.copy()
        for k, v in self._compiler_env.items():
            if v:
                env[k] = v

        cmd = ["bash", self._worker_script]
        if self._task_arg:
            cmd.append(self._task_arg)
        cmd.extend([src_remote, results_dir, *run_args])

        print(f"  Running {self._task_arg or 'worker'} ({job_name})...",
              end="  ", flush=True)
        try:
            # The worker captures its own stdout/stderr to log files; we
            # suppress its console output here so the demo UI stays clean.
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        except Exception as exc:
            print(f"{_R}FAILED: {exc}{_E}")
            raise Exception(f"Local worker invocation error: {exc}") from exc

        # Worker exit codes: 0 = build+run OK, 1 = compile fail, 2 = run fail.
        # We treat all three as "completed" — run_pipeline's downstream parse
        # of build_rc.txt / run_rc.txt is the source of truth.
        if verbose and r.stderr.strip():
            print(f"\n    [worker stderr] {r.stderr.strip()[:400]}")
        print(f"{_G}done{_E}  (rc={r.returncode})")


def build_local_runner(
    *,
    resource: str, worker_key: str,
    workers_dir: str,
    compiler_env: dict[str, str],
) -> LocalRunner:
    """Construct a LocalRunner for the given (resource, worker_key)."""
    script, task_arg = WORKERS[resource][worker_key]
    return LocalRunner(
        worker_script   = f"{workers_dir}/{script}",
        worker_task_arg = task_arg,
        compiler_env    = compiler_env,
    )
