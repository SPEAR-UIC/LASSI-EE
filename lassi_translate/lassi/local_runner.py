"""Drop-in replacement for JobRunner when on the compute node — invokes the
worker shell script as a subprocess instead of submitting a PBS job."""
import os
import subprocess

from lassi.format_colors import G, R, E
from lassi.resources     import WORKERS


class LocalRunner:
    """Same interface as JobRunner; runs `bash worker.sh ...` locally."""

    def __init__(self, worker_script, worker_task_arg, compiler_env):
        self._script  = worker_script
        self._task    = worker_task_arg
        self._env     = compiler_env

    def run(self, *, src_remote, results_dir, run_args, job_name, verbose=False):
        env = os.environ.copy()
        env.update({k: v for k, v in self._env.items() if v})
        cmd = ["bash", self._script]
        if self._task: cmd.append(self._task)
        cmd.extend([src_remote, results_dir, *run_args])
        print(f"  Running {self._task or 'worker'} ({job_name})...", end="  ", flush=True)
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        except Exception as exc:
            print(f"{R}FAILED: {exc}{E}")
            raise Exception(f"Local worker error: {exc}") from exc
        if verbose and r.stderr.strip():
            print(f"\n    [worker stderr] {r.stderr.strip()[:400]}")
        print(f"{G}done{E}  (rc={r.returncode})")


def build_local_runner(*, resource, worker_key, workers_dir, compiler_env):
    """Look up (script, task_arg) in WORKERS and construct a LocalRunner."""
    script, task = WORKERS[resource][worker_key]
    return LocalRunner(f"{workers_dir}/{script}", task, compiler_env)
