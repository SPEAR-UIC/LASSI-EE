"""PBS job submission, monitoring, persistent worker, and per-attempt JobRunner."""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from lassi.format_colors import color as _c
from lassi.hpc_session    import HPCSession
from lassi.resources      import WORKERS

_B, _G, _R, _E = _c.BOLD, _c.GREEN, _c.RED, _c.END

# ── Timing constants ─────────────────────────────────────────────────────────
JOB_DURATION_S            = 300    # default per-job walltime; 1 h = Polaris debug max
JOB_WAIT_TIMEOUT_S        = 900
JOB_POLL_INTERVAL_S       = 15
PERSISTENT_START_TIMEOUT  = 600    # seconds to wait for job_started
PERSISTENT_RESULT_TIMEOUT = 1800   # seconds to wait for result_ready per attempt
EAGLE_PROPAGATION_S       = 5


# ─────────────────────────────────────────────────────────────────────────────
# SSH-qsub fallback: amsc-client-compatible job wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SshPbsJob:
    """Minimal job-like wrapper around a PBS job ID obtained via SSH qsub.

    Exposes .id / .state / .is_terminal / .status matching amsc_client jobs so
    the rest of the code doesn't need to branch on backend.
    """

    def __init__(self, job_id: str, ssh_ctl: str, host: str, user: str) -> None:
        self._id      = job_id
        self._ssh_ctl = ssh_ctl
        self._host    = host
        self._user    = user

    @property
    def id(self) -> str:
        return self._id

    def _qstat_state(self) -> tuple[str, int]:
        """Return (pbs_state_char, ssh_returncode).

        PBS Pro: Q=queued, R=running, E=exiting, F=finished, X=expired.
        Non-zero rc means the job was purged → treat as finished.
        """
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={self._ssh_ctl}",
             f"{self._user}@{self._host}",
             f"qstat -f {shlex.quote(self._id)} 2>/dev/null"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return "F", r.returncode
        for line in r.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("job_state"):
                return stripped.split("=", 1)[1].strip(), 0
        return "R", 0  # state line not yet visible

    @property
    def state(self) -> str:
        pbs_state, _rc = self._qstat_state()
        return {
            "Q": "queued",   "H": "queued",
            "R": "running",  "E": "exiting",
            "F": "completed","X": "completed",
        }.get(pbs_state, pbs_state.lower())

    @property
    def is_terminal(self) -> bool:
        pbs_state, rc = self._qstat_state()
        return rc != 0 or pbs_state in ("F", "X")

    @property
    def status(self):
        # amsc_client checks .status before .is_terminal; provide same surface.
        return self.state


def ssh_pbs_submit(
    *,
    ssh_ctl: str, host: str, user: str,
    job_name: str, queue: str, account: str,
    walltime_s: int, nodes: int = 1, filesystems: str = "",
    remote_dir: str, stdout_path: str, stderr_path: str, cmd: str,
) -> SshPbsJob:
    """Submit a PBS job via SSH qsub. Used when amsc_client cannot submit
    (e.g. Aurora's IRI API does not expose job submission)."""
    h, rem = divmod(walltime_s, 3600)
    m, s   = divmod(rem, 60)
    walltime_str = f"{h:02d}:{m:02d}:{s:02d}"

    fs_line = f"#PBS -l filesystems={filesystems}" if filesystems else ""
    script = (
        "#!/bin/bash\n"
        f"#PBS -N {job_name}\n"
        f"#PBS -q {queue}\n"
        f"#PBS -A {account}\n"
        f"#PBS -l select={nodes}\n"
        f"#PBS -l walltime={walltime_str}\n"
        f"{fs_line}\n"
        f"#PBS -o {stdout_path}\n"
        f"#PBS -e {stderr_path}\n"
        f"cd {shlex.quote(remote_dir)}\n"
        f"{cmd}\n"
    )

    r = subprocess.run(
        ["ssh", "-o", f"ControlPath={ssh_ctl}", f"{user}@{host}", "qsub"],
        input=script, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"SSH qsub failed (rc={r.returncode}): {r.stderr.strip()}")
    job_id = r.stdout.strip()
    if not job_id:
        raise RuntimeError("SSH qsub returned empty job ID")
    return SshPbsJob(job_id, ssh_ctl, host, user)


def submit_job(
    *,
    compute_resource, remote_dir: str,
    worker_script: str, task_arg: str = "",
    src_remote_path: str, results_remote_dir: str,
    run_args: list[str], job_name: str,
    account: str, queue: str, filesystems: str,
    walltime_s: int = JOB_DURATION_S,
    compiler_env: dict[str, str] | None = None,
    ssh_ctl: str = "", host: str = "", user: str = "",
):
    """Submit a PBS job that runs the given worker script on a compute node.

    Falls back to SSH qsub when compute_resource.submit() is unavailable.
    """
    run_args_quoted = " ".join(shlex.quote(a) for a in run_args)
    env_parts = [f"{k}={shlex.quote(v)}" for k, v in (compiler_env or {}).items() if v]
    env_prefix = (" ".join(env_parts) + " ") if env_parts else ""

    cmd = (
        f"{env_prefix}bash {shlex.quote(worker_script)}"
        + (f" {shlex.quote(task_arg)}" if task_arg else "")
        + f" {shlex.quote(src_remote_path)}"
        + f" {shlex.quote(results_remote_dir)}"
        + f" {run_args_quoted}"
    )
    stdout_path = f"{results_remote_dir}/pbs_stdout.log"
    stderr_path = f"{results_remote_dir}/pbs_stderr.log"

    try:
        return compute_resource.submit(
            executable  = "/bin/bash",
            arguments   = ["-c", cmd],
            directory   = remote_dir,
            name        = job_name,
            queue       = queue,
            account     = account,
            duration    = walltime_s,
            nodes       = 1,
            stdout_path = stdout_path,
            stderr_path = stderr_path,
            filesystems = filesystems,
        )
    except Exception as amsc_exc:
        if not (ssh_ctl and host and user):
            raise
        print(f"  [amsc fallback → SSH qsub: {amsc_exc}]", flush=True)
        return ssh_pbs_submit(
            ssh_ctl=ssh_ctl, host=host, user=user,
            job_name=job_name, queue=queue, account=account,
            walltime_s=walltime_s, filesystems=filesystems,
            remote_dir=remote_dir,
            stdout_path=stdout_path, stderr_path=stderr_path,
            cmd=cmd,
        )


def wait_for_job(job, timeout_s: int = JOB_WAIT_TIMEOUT_S) -> str:
    """Poll job status until terminal state or timeout."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            _ = job.status
            if job.is_terminal:
                return job.state
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "400" in msg:
                return "completed"
        time.sleep(JOB_POLL_INTERVAL_S)
    try:
        return job.state
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Persistent worker (one long-running PBS job receives work via IPC files)
# ─────────────────────────────────────────────────────────────────────────────

class PersistentWorker:
    """Manages a long-running PBS job that receives work via IPC files.

    IPC protocol (all files in ipc_dir on the remote FS):
      job_started       written by worker at startup (content = hostname)
      trigger.json      written by us to request work
      result_ready.json written by worker when work is done
      exit              written by us to shut the worker down
    """

    def __init__(self, job, ipc_dir: str, session: HPCSession,
                 hostname: str = "") -> None:
        self._job     = job
        self._ipc_dir = ipc_dir
        self._session = session
        self.hostname = hostname   # compute node where the worker is running

    @classmethod
    def launch(
        cls, *,
        session: HPCSession, compute_resource,
        remote_dir: str, ipc_dir: str,
        account: str, queue: str, filesystems: str,
        walltime_s: int, resource_name: str,
        start_timeout_s: int = PERSISTENT_START_TIMEOUT,
        quiet: bool = False,
    ) -> "PersistentWorker":
        """Submit the persistent worker PBS job and wait for it to signal readiness.

        `quiet=True` suppresses the step-by-step submit/wait prints (the caller
        is expected to surface its own one-line confirmation including the
        returned worker's `.hostname`).
        """
        worker_script = f"{remote_dir}/workers/{WORKERS[resource_name]['persistent_worker'][0]}"
        job_name = f"lassi-persistent-{int(time.time())}"
        cmd = f"bash {shlex.quote(worker_script)} {shlex.quote(ipc_dir)}"

        def _say(msg: str, *, end: str = "\n") -> None:
            if not quiet:
                print(msg, end=end, flush=True)

        _say(f"  Submitting persistent worker PBS job ({job_name})...", end="  ")
        try:
            job = compute_resource.submit(
                executable="/bin/bash", arguments=["-c", cmd],
                directory=remote_dir, name=job_name,
                queue=queue, account=account, duration=walltime_s, nodes=1,
                stdout_path=f"{ipc_dir}/pbs_stdout.log",
                stderr_path=f"{ipc_dir}/pbs_stderr.log",
                filesystems=filesystems,
            )
        except Exception as amsc_exc:
            if not session.ssh_ctl:
                raise
            _say(f"\n  [amsc fallback → SSH qsub: {amsc_exc}]")
            _say(f"  Submitting via SSH qsub...", end="  ")
            job = ssh_pbs_submit(
                ssh_ctl=session.ssh_ctl, host=session.host, user=session.user,
                job_name=job_name, queue=queue, account=account,
                walltime_s=walltime_s, filesystems=filesystems,
                remote_dir=remote_dir,
                stdout_path=f"{ipc_dir}/pbs_stdout.log",
                stderr_path=f"{ipc_dir}/pbs_stderr.log",
                cmd=cmd,
            )
        _say(f"{_G}submitted{_E}  (PBS ID: {job.id})")

        started_path = f"{ipc_dir}/job_started"
        deadline     = time.time() + start_timeout_s
        _say(f"  Waiting for persistent worker to start (up to {start_timeout_s}s)...",
             end="  ")
        while time.time() < deadline:
            if session.file_exists(started_path):
                break
            try:
                if job.is_terminal:
                    raise RuntimeError(
                        f"Persistent worker job {job.id} reached terminal state before "
                        f"writing job_started. Check {ipc_dir}/pbs_stderr.log for errors."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(10)
        else:
            raise TimeoutError(
                f"Persistent worker failed to write job_started within {start_timeout_s}s. "
                f"Check {ipc_dir}/pbs_stderr.log for errors."
            )

        hostname = session.read_file(started_path).strip()
        _say(f"{_G}ready{_E}  (compute node: {hostname})")
        return cls(job, ipc_dir, session, hostname=hostname)

    def signal_work(self, trigger: dict) -> None:
        """Deliver a work trigger to the persistent worker atomically (rsync + mv)."""
        session = self._session
        ipc_dir = self._ipc_dir

        session.run(f"rm -f {shlex.quote(ipc_dir + '/result_ready.json')}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="lassi_trigger_"
        ) as tf:
            json.dump(trigger, tf)
            tmp_path = tf.name

        try:
            remote_tmp = f"{ipc_dir}/trigger.json.tmp"
            if not session.rsync(tmp_path, remote_tmp):
                raise RuntimeError("rsync of trigger.json.tmp to remote failed")
            rc = session.run(
                f"mv {shlex.quote(remote_tmp)} {shlex.quote(ipc_dir + '/trigger.json')}"
            )
            if rc != 0:
                raise RuntimeError("Remote mv trigger.json.tmp → trigger.json failed")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def wait_for_result(
        self,
        timeout_s: int = PERSISTENT_RESULT_TIMEOUT,
        poll_s: float = 3.0,
    ) -> dict:
        """Poll for result_ready.json; return its parsed content."""
        session     = self._session
        ipc_dir     = self._ipc_dir
        result_path = f"{ipc_dir}/result_ready.json"
        deadline    = time.time() + timeout_s
        last_job_check = time.time()

        while time.time() < deadline:
            if session.file_exists(result_path):
                raw = session.read_file(result_path)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    # Briefly retry: rsync + mv is atomic but FS propagation
                    # can race the read by a few hundred ms.
                    time.sleep(0.5)
                    return json.loads(session.read_file(result_path))

            if time.time() - last_job_check > 30:
                try:
                    if self._job.is_terminal:
                        raise RuntimeError(
                            f"Persistent worker job {self._job.id} died unexpectedly "
                            f"while waiting for result_ready. "
                            f"Check {ipc_dir}/pbs_stderr.log for errors."
                        )
                except RuntimeError:
                    raise
                except Exception:
                    pass
                last_job_check = time.time()

            time.sleep(poll_s)

        raise TimeoutError(
            f"Timed out after {timeout_s}s waiting for result_ready.json in {ipc_dir}. "
            f"Check {ipc_dir}/worker.log for diagnostics."
        )

    def cleanup_result(self) -> None:
        self._session.run(
            f"rm -f {shlex.quote(self._ipc_dir + '/result_ready.json')}"
        )

    def shutdown(self) -> None:
        """Send exit signal to the persistent worker and wait for the job to finish."""
        session = self._session
        ipc_dir = self._ipc_dir

        print(f"\n  Sending exit signal to persistent worker...", end="  ", flush=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".exit", delete=False, prefix="lassi_exit_"
        ) as tf:
            tf.write("")
            tmp_path = tf.name

        try:
            if not session.rsync(tmp_path, f"{ipc_dir}/exit"):
                print(f"{_R}rsync of exit signal failed (job may linger){_E}")
                return
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if self._job.is_terminal:
                    print(f"{_G}done{_E}  (state: {self._job.state})")
                    return
            except Exception:
                pass
            time.sleep(5)

        print(f"{_R}timeout — job may still be running{_E}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-attempt job runner — dispatches to persistent worker or fresh PBS job
# ─────────────────────────────────────────────────────────────────────────────

class JobRunner:
    """One compile+run task via either a PersistentWorker or a fresh PBS job."""

    def __init__(
        self, *,
        session: HPCSession, compute_resource,
        remote_dir: str, account: str, queue: str, filesystems: str,
        worker_script: str, worker_task_arg: str,
        compiler_env: dict[str, str],
        walltime_s: int = JOB_DURATION_S,
        persistent: PersistentWorker | None = None,
    ) -> None:
        self._session       = session
        self._compute       = compute_resource
        self._remote_dir    = remote_dir
        self._account       = account
        self._queue         = queue
        self._filesystems   = filesystems
        self._worker_script = worker_script
        self._task_arg      = worker_task_arg
        self._compiler_env  = compiler_env
        self._walltime_s    = walltime_s
        self._persistent    = persistent

    def run(
        self, *,
        src_remote: str, results_dir: str,
        run_args: list[str], job_name: str,
        verbose: bool = False,
    ) -> None:
        """Execute one compile+run task. Raises on failure.

        RuntimeError — fatal (persistent worker died); caller should break loop.
        Exception    — retriable; caller should continue loop.
        """
        if self._persistent is not None:
            self._run_persistent(src_remote, results_dir, run_args, job_name)
        else:
            self._run_per_job(src_remote, results_dir, run_args, job_name, verbose)

    def _run_persistent(
        self, src_remote: str, results_dir: str, run_args: list[str], job_name: str,
    ) -> None:
        worker = self._persistent
        trigger = {
            "task":         self._task_arg,
            "src":          src_remote,
            "outdir":       results_dir,
            "run_args":     run_args,
            "compiler_env": self._compiler_env,
        }
        print(f"  Signalling work to persistent worker ({self._task_arg}, {job_name})...",
              end="  ", flush=True)
        try:
            worker.signal_work(trigger)
        except Exception as exc:
            print(f"{_R}FAILED: {exc}{_E}")
            raise Exception(f"Persistent worker signal error: {exc}") from exc
        print(f"{_G}OK{_E}")

        print(f"  Waiting for result ({job_name})...", end="  ", flush=True)
        try:
            worker.wait_for_result()
        except RuntimeError as exc:
            print(f"{_R}WORKER DIED: {exc}{_E}")
            raise RuntimeError(f"Persistent worker died: {exc}") from exc
        except TimeoutError as exc:
            print(f"{_R}TIMEOUT: {exc}{_E}")
            raise Exception(f"Timed out waiting for worker result: {exc}") from exc
        worker.cleanup_result()
        print(f"{_G}done{_E}")

    def _run_per_job(
        self, src_remote: str, results_dir: str, run_args: list[str],
        job_name: str, verbose: bool,
    ) -> None:
        session = self._session
        print(f"  Submitting PBS job ({job_name})...", end="  ", flush=True)
        try:
            job = submit_job(
                compute_resource   = self._compute,
                remote_dir         = self._remote_dir,
                worker_script      = self._worker_script,
                task_arg           = self._task_arg,
                src_remote_path    = src_remote,
                results_remote_dir = results_dir,
                run_args           = run_args,
                job_name           = job_name,
                account            = self._account,
                queue              = self._queue,
                filesystems        = self._filesystems,
                walltime_s         = self._walltime_s,
                compiler_env       = self._compiler_env,
                ssh_ctl            = session.ssh_ctl,
                host               = session.host,
                user               = session.user,
            )
        except Exception as exc:
            print(f"{_R}FAILED: {exc}{_E}")
            raise Exception(f"PBS submission error: {exc}") from exc

        pbs_id = job.id.split(".")[0]
        print(f"{_G}submitted{_E}  (PBS ID: {job.id})")
        if verbose:
            print(f"    Monitor: ssh {session.host} qstat {pbs_id}")

        print(f"  Waiting for PBS job {pbs_id}...", end="  ", flush=True)
        final_state = wait_for_job(job)
        state_color = _G if final_state == "completed" else _R
        print(f"{state_color}{final_state}{_E}")

        if final_state not in ("completed", "active"):
            time.sleep(EAGLE_PROPAGATION_S)
            raise Exception(f"PBS job ended in unexpected state: {final_state!r}")

        time.sleep(EAGLE_PROPAGATION_S)


def build_runner(
    *,
    session: HPCSession, compute_resource, resource: str,
    remote_dir: str, account: str, queue: str, filesystems: str,
    worker_key: str, compiler_env: dict[str, str],
    walltime_s: int = JOB_DURATION_S,
    persistent: PersistentWorker | None = None,
) -> JobRunner:
    """Construct a JobRunner for `resource`/`worker_key` (per WORKERS table)."""
    script, task_arg = WORKERS[resource][worker_key]
    return JobRunner(
        session         = session,
        compute_resource = compute_resource,
        remote_dir      = remote_dir,
        account         = account,
        queue           = queue,
        filesystems     = filesystems,
        worker_script   = f"{remote_dir}/workers/{script}",
        worker_task_arg = task_arg,
        compiler_env    = compiler_env,
        walltime_s      = walltime_s,
        persistent      = persistent,
    )
