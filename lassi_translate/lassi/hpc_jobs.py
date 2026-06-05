"""PBS submission, the long-running PersistentWorker, and the per-attempt JobRunner.

amsc-client is required.  No SSH fallbacks: job submission, state polling, and
all IPC reads go through amsc.  The only SSH usage is `rsync` (upload) for the
IPC trigger.json + exit signal — amsc has no upload API.
"""
import json
import shlex
import tempfile
import time
from pathlib import Path

from lassi.format_colors import B, G, R, E
from lassi.hpc_session   import HPCSession
from lassi.resources     import WORKERS

JOB_DURATION_S            = 300
JOB_WAIT_TIMEOUT_S        = None
JOB_POLL_INTERVAL_S       = 15
PERSISTENT_START_TIMEOUT  = None
PERSISTENT_RESULT_TIMEOUT = 1800
EAGLE_PROPAGATION_S       = 5


def submit_job(*, compute_resource, remote_dir, worker_script, task_arg="",
               src_remote_path, results_remote_dir, run_args, job_name,
               account, queue, filesystems, walltime_s=JOB_DURATION_S,
               compiler_env=None):
    """Submit one PBS job that runs `worker.sh <task> <src> <outdir> ...` via amsc."""
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in (compiler_env or {}).items() if v)
    env = (env + " ") if env else ""
    args = " ".join(shlex.quote(a) for a in run_args)
    cmd = (f"{env}bash {shlex.quote(worker_script)}"
           f"{' '+shlex.quote(task_arg) if task_arg else ''}"
           f" {shlex.quote(src_remote_path)} {shlex.quote(results_remote_dir)} {args}")
    so, se = f"{results_remote_dir}/pbs_stdout.log", f"{results_remote_dir}/pbs_stderr.log"
    return compute_resource.submit(
        executable="/bin/bash", arguments=["-c", cmd],
        directory=remote_dir, name=job_name, queue=queue, account=account,
        duration=walltime_s, nodes=1, stdout_path=so, stderr_path=se,
        filesystems=filesystems)


def wait_for_job(job, timeout_s=JOB_WAIT_TIMEOUT_S):
    """Poll job.is_terminal every JOB_POLL_INTERVAL_S until done (or timeout)."""
    start = time.time()
    while timeout_s is None or time.time() - start < timeout_s:
        try:
            _ = job.status
            if job.is_terminal: return job.state
        except Exception as exc:
            m = str(exc).lower()
            if "not found" in m or "400" in m: return "completed"
        time.sleep(JOB_POLL_INTERVAL_S)
    try: return job.state
    except Exception: return "unknown"


class PersistentWorker:
    """A long-running PBS job that processes compile-and-run requests via file-IPC.

    Avoids queueing a fresh PBS job per attempt (which would burn minutes of
    queue wait each time).  IPC files in `ipc_dir` on the remote FS:
        job_started        worker writes (hostname) when ready
        trigger.json       driver writes to request work; worker polls for it
        result_ready.json  worker writes when work is done; driver polls for it
        exit               driver writes to ask worker to shut down

    Driver-side: trigger.json + exit are uploads (rsync); all reads/moves/rms
    go through amsc.  See workers/*_persistent_worker.sh for the worker loop.
    """

    def __init__(self, job, ipc_dir, session: HPCSession, hostname=""):
        self._job, self._ipc, self._session, self.hostname = job, ipc_dir, session, hostname

    @classmethod
    def launch(cls, *, session, compute_resource, remote_dir, ipc_dir,
               account, queue, filesystems, walltime_s, resource_name,
               start_timeout_s=PERSISTENT_START_TIMEOUT, quiet=False):
        """Submit the worker PBS job and wait for it to write job_started."""
        script = f"{remote_dir}/workers/{WORKERS[resource_name]['persistent_worker'][0]}"
        name = f"lassi-persistent-{int(time.time())}"
        cmd = f"bash {shlex.quote(script)} {shlex.quote(ipc_dir)}"
        say = (lambda m, end="\n": None) if quiet else (lambda m, end="\n": print(m, end=end, flush=True))

        say(f"  Submitting persistent worker ({name})...", end="  ")
        job = compute_resource.submit(
            executable="/bin/bash", arguments=["-c", cmd],
            directory=remote_dir, name=name, queue=queue, account=account,
            duration=walltime_s, nodes=1,
            stdout_path=f"{ipc_dir}/pbs_stdout.log",
            stderr_path=f"{ipc_dir}/pbs_stderr.log",
            filesystems=filesystems)
        say(f"{G}submitted{E}  (PBS ID: {job.id})")

        started = f"{ipc_dir}/job_started"
        deadline = None if start_timeout_s is None else time.time() + start_timeout_s
        say(f"  Waiting for worker to start...", end="  ")
        while True:
            if session.file_exists(started): break
            try:
                if job.is_terminal:
                    raise RuntimeError(
                        f"Worker {job.id} terminated before startup; "
                        f"check {ipc_dir}/pbs_stderr.log")
            except RuntimeError: raise
            except Exception: pass
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"Worker startup timed out; check {ipc_dir}/pbs_stderr.log")
            time.sleep(10)

        host = session.read_file(started).strip()
        say(f"{G}ready{E}  (node: {host})")
        return cls(job, ipc_dir, session, hostname=host)

    def signal_work(self, trigger):
        """Atomically deliver `trigger` dict to the worker: rsync upload + amsc mv."""
        s = self._session
        s.rm(self._ipc + "/result_ready.json")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                          prefix="lassi_trigger_") as tf:
            json.dump(trigger, tf); tmp = tf.name
        try:
            remote_tmp = f"{self._ipc}/trigger.json.tmp"
            if not s.rsync(tmp, remote_tmp):
                raise RuntimeError("trigger rsync failed")
            s.mv(remote_tmp, f"{self._ipc}/trigger.json")
        finally:
            Path(tmp).unlink(missing_ok=True)

    def wait_for_result(self, timeout_s=PERSISTENT_RESULT_TIMEOUT, poll_s=3.0):
        """Poll for result_ready.json; raise if the worker died mid-attempt."""
        s = self._session
        path = f"{self._ipc}/result_ready.json"
        deadline = time.time() + timeout_s
        last_check = time.time()
        while time.time() < deadline:
            if s.file_exists(path):
                raw = s.read_file(path)
                try: return json.loads(raw)
                except json.JSONDecodeError:
                    time.sleep(0.5)
                    return json.loads(s.read_file(path))
            if time.time() - last_check > 30:
                try:
                    if self._job.is_terminal:
                        raise RuntimeError(f"Worker {self._job.id} died; "
                                           f"check {self._ipc}/pbs_stderr.log")
                except RuntimeError: raise
                except Exception: pass
                last_check = time.time()
            time.sleep(poll_s)
        raise TimeoutError(f"Worker result timed out after {timeout_s}s")

    def cleanup_result(self):
        self._session.rm(self._ipc + "/result_ready.json")

    def shutdown(self):
        """Upload the exit signal via rsync; wait for the worker job to terminate."""
        s = self._session
        print(f"\n  Sending exit signal...", end="  ", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".exit", delete=False,
                                          prefix="lassi_exit_") as tf:
            tmp = tf.name
        try:
            if not s.rsync(tmp, f"{self._ipc}/exit"):
                print(f"{R}exit signal rsync failed{E}"); return
        finally:
            Path(tmp).unlink(missing_ok=True)
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if self._job.is_terminal:
                    print(f"{G}done{E}  ({self._job.state})"); return
            except Exception: pass
            time.sleep(5)
        print(f"{R}timeout{E}")


class JobRunner:
    """One compile+run attempt — uses PersistentWorker if given, else fresh PBS job."""

    def __init__(self, *, session, compute_resource, remote_dir, account, queue,
                 filesystems, worker_script, worker_task_arg, compiler_env,
                 walltime_s=JOB_DURATION_S, persistent: PersistentWorker | None = None):
        self._session, self._compute = session, compute_resource
        self._dir, self._account, self._queue = remote_dir, account, queue
        self._fs, self._script, self._task = filesystems, worker_script, worker_task_arg
        self._env, self._walltime, self._persistent = compiler_env, walltime_s, persistent

    def run(self, *, src_remote, results_dir, run_args, job_name, verbose=False):
        """Compile + run.  Raises RuntimeError on fatal (worker died),
        generic Exception on retriable (compile error to feed back to LLM)."""
        if self._persistent is not None:
            self._run_persistent(src_remote, results_dir, run_args, job_name)
        else:
            self._run_per_job(src_remote, results_dir, run_args, job_name, verbose)

    def _run_persistent(self, src_remote, results_dir, run_args, job_name):
        w = self._persistent
        trigger = {"task": self._task, "src": src_remote, "outdir": results_dir,
                   "run_args": run_args, "compiler_env": self._env}
        print(f"  Signalling work ({self._task}, {job_name})...", end="  ", flush=True)
        try: w.signal_work(trigger)
        except Exception as exc:
            print(f"{R}FAILED: {exc}{E}")
            raise Exception(f"Worker signal error: {exc}") from exc
        print(f"{G}OK{E}")
        print(f"  Waiting for result ({job_name})...", end="  ", flush=True)
        try: w.wait_for_result()
        except RuntimeError as exc:
            print(f"{R}WORKER DIED: {exc}{E}")
            raise RuntimeError(f"Worker died: {exc}") from exc
        except TimeoutError as exc:
            print(f"{R}TIMEOUT: {exc}{E}")
            raise Exception(f"Worker timeout: {exc}") from exc
        w.cleanup_result()
        print(f"{G}done{E}")

    def _run_per_job(self, src_remote, results_dir, run_args, job_name, verbose):
        print(f"  Submitting PBS job ({job_name})...", end="  ", flush=True)
        try:
            job = submit_job(
                compute_resource=self._compute, remote_dir=self._dir,
                worker_script=self._script, task_arg=self._task,
                src_remote_path=src_remote, results_remote_dir=results_dir,
                run_args=run_args, job_name=job_name,
                account=self._account, queue=self._queue, filesystems=self._fs,
                walltime_s=self._walltime, compiler_env=self._env)
        except Exception as exc:
            print(f"{R}FAILED: {exc}{E}")
            raise Exception(f"PBS error: {exc}") from exc
        pbs = job.id.split(".")[0]
        print(f"{G}submitted{E}  (PBS ID: {job.id})")
        if verbose: print(f"    Monitor: qstat {pbs}")
        print(f"  Waiting for PBS job {pbs}...", end="  ", flush=True)
        state = wait_for_job(job)
        col = G if state == "completed" else R
        print(f"{col}{state}{E}")
        if state not in ("completed", "active"):
            time.sleep(EAGLE_PROPAGATION_S)
            raise Exception(f"PBS state: {state!r}")
        time.sleep(EAGLE_PROPAGATION_S)


def build_runner(*, session, compute_resource, resource, remote_dir, account, queue,
                 filesystems, worker_key, compiler_env, walltime_s=JOB_DURATION_S,
                 persistent=None):
    """Construct a JobRunner from a (resource, worker_key) lookup in WORKERS."""
    script, task = WORKERS[resource][worker_key]
    return JobRunner(
        session=session, compute_resource=compute_resource,
        remote_dir=remote_dir, account=account, queue=queue, filesystems=filesystems,
        worker_script=f"{remote_dir}/workers/{script}", worker_task_arg=task,
        compiler_env=compiler_env, walltime_s=walltime_s, persistent=persistent)
