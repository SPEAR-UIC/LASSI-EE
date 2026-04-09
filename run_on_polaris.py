#!/usr/bin/env python3
"""
Submit lassi_translate to Aurora or Polaris, stream stdout, and download the result.

Usage:
    python run_on_polaris.py

Prerequisites:
    pip install amsc-client[globus]

Set RESOURCE_NAME to "Aurora" or "Polaris" to switch targets. The SYCL toolchain
(compiler, flags, module loads, and wrapper script) is selected automatically.
"""

import datetime
import pathlib
import time

# ── Configuration — edit these before running ─────────────────────────────────

ACCOUNT       = "EE-ECP"               # your PBS allocation account name
QUEUE         = "debug"                # queue name (debug, prod, etc.)
NODES         = 1
DURATION      = 3600                   # wall-clock limit in seconds
POLL_INTERVAL = 10                     # seconds between stdout reads

RESOURCE_NAME = "Polaris"              # "Aurora" or "Polaris"

# Remote base directory on the ALCF Eagle filesystem
REMOTE_BASE = f"/eagle/{ACCOUNT}/LASSI-EE/lassi_translate"

# Unique per-run paths prevent stale output from a previous run being displayed
# when a job is rejected before writing anything (e.g. script not executable).
_RUN_ID    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
STDOUT_PATH = f"{REMOTE_BASE}/job_{_RUN_ID}.stdout"
STDERR_PATH = f"{REMOTE_BASE}/job_{_RUN_ID}.stderr"


# ── Per-resource configuration ────────────────────────────────────────────────
# Each resource has a wrapper script that owns all toolchain knowledge
# (module loads, SYCL compiler flags, Python resolution). Adding a new
# resource means adding a script and a corresponding entry here.

RESOURCE_SCRIPTS: dict[str, str] = {
    "Aurora":  f"{REMOTE_BASE}/run_aurora_demo_interactive.sh",
    "Polaris": f"{REMOTE_BASE}/run_polaris_demo.sh",
}

RESOURCE_FILESYSTEMS: dict[str, str] = {
    "Aurora":  "home:flare",
    "Polaris": "home:eagle",
}

# Local path where the downloaded SYCL result will be saved
LOCAL_OUTPUT  = pathlib.Path(__file__).parent / "output"


# ── Pre-job setup (NOT YET FULLY AUTOMATED) ──────────────────────────────────
#
# The following steps currently require manual action on a Polaris login node.
# Once amsc_client matures, this script should handle them automatically:
#
#   1. File transfer — upload source files to REMOTE_BASE on Eagle:
#        rsync -av --exclude=__pycache__ --exclude=generated_codes \
#              --exclude=Main --exclude=XSBench --exclude='*.o' \
#              lassi_translate/ <user>@polaris.alcf.anl.gov:/eagle/EE-ECP/LASSI-EE/lassi_translate/
#      Blocked by:
#        - filesystem mkdir:   returns 501 "Command mkdir option not implemented yet"
#        - filesystem upload:  returns 501 "Command upload option not implemented yet"
#
#   2. ALCF Inference token — authenticate once on the Polaris login node so
#      that compute nodes can call `inference_auth_token.py get_access_token`:
#        scp inference_auth_token.py <user>@polaris.alcf.anl.gov:~/
#        ssh <user>@polaris.alcf.anl.gov \
#            "conda run -n globus_env python3 ~/inference_auth_token.py authenticate"
#      Blocked by:
#        - job submit environment: returns 501 "'environment' not supported yet"
#          (needed to inject a pre-obtained token into the job environment)
#        - tokens expire after 48 h / require re-auth every 30 days, so a
#          one-time injection is not a durable solution regardless
#
def upload_files() -> None:
    raise NotImplementedError(
        "amsc_client does not yet support file upload (501). "
        "Transfer files manually via rsync/scp before running this script."
    )


# ── Filesystem view result helper ────────────────────────────────────────────

def _extract_view_content(result) -> str | None:
    """Extract text from an fs.view() task result.

    The API returns different structures depending on whether the file exists:
      {'error': '...'}                          — file not found (no content)
      {'output': {'content': '...', ...}}       — file content
      str / bytes                               — raw content (fallback)
    Returns the content string, or None if there is nothing to display.
    """
    if result is None:
        return None
    if isinstance(result, dict):
        output = result.get("output")
        if output:
            content = output.get("content", "")
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")
            return content or None
        return None  # 'error' key or unrecognised format
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, str):
        return result
    return None


# ── Stdout streaming ──────────────────────────────────────────────────────────

def stream_stdout(fs_resource, stdout_path: str, job) -> None:
    """Poll the job's stdout file and print new content until the job finishes.

    fs_resource must be the storage resource that owns the path (e.g. Eagle for
    /eagle/... paths) — NOT the compute resource used to submit the job.

    Uses fs.view() with an advancing byte offset — the equivalent of `tail -f`
    over repeated HTTP calls. Each call dispatches a Globus Compute task on
    the facility, so expect several seconds of latency per poll.

    PBS may buffer stdout until the process flushes or fills its buffer. If
    your script doesn't flush explicitly, output may appear in bursts.
    """
    offset = 0

    print("\n" + "=" * 60)
    print("JOB STDOUT (polling every %ds)" % POLL_INTERVAL)
    print("=" * 60 + "\n")

    while not job.is_terminal:
        try:
            job.refresh()
        except Exception as exc:
            # PBS removes completed jobs from the queue, causing the API to
            # return 400 "not found" (same behaviour Job.wait() already handles).
            # Treat any not-found response as terminal and drain remaining output.
            if "not found" in str(exc).lower() or getattr(exc, "status_code", 0) in (400, 404):
                break
            raise

        # Read up to 64 KB of new content from the current offset.
        # PBS creates the stdout file only once the job starts running, so
        # view() may return an error dict while the file doesn't yet exist.
        try:
            t = fs_resource.fs.view(stdout_path, offset=offset, size=65536)
            t.wait(timeout=30)
            chunk = _extract_view_content(t.result)
            if chunk:
                print(chunk, end="", flush=True)
                offset += len(chunk.encode("utf-8"))
        except Exception:
            pass  # transient error — keep polling

        time.sleep(POLL_INTERVAL)

    # Drain any output written after the final job.refresh()
    try:
        t = fs_resource.fs.view(stdout_path, offset=offset, size=65536)
        t.wait(timeout=30)
        chunk = _extract_view_content(t.result)
        if chunk:
            print(chunk, end="", flush=True)
    except Exception:
        pass

    print("\n\n" + "=" * 60)


# ── Stderr drain ──────────────────────────────────────────────────────────────

def drain_stderr(fs_resource, stderr_path: str) -> bool:
    """Print the last 100 lines of job stderr; return True if non-empty.

    fs_resource must be the storage resource that owns the path (e.g. Eagle).
    Called unconditionally after the job finishes to surface any error output.
    """
    print("\nChecking stderr...")
    try:
        t = fs_resource.fs.view(stderr_path, offset=0, size=65536)
        t.wait(timeout=30)
        content = _extract_view_content(t.result)
        if content and content.strip():
            print("=" * 60)
            print("JOB STDERR")
            print("=" * 60)
            print(content)
            return True
        print("  (stderr is empty)")
        return False
    except Exception as e:
        print(f"  Could not read stderr: {e}")
        return False


# ── Result discovery ──────────────────────────────────────────────────────────

def find_sycl_output(fs_resource) -> str | None:
    """Locate Main_v1_Trial1.cpp in the most recent experiment directory.

    The pipeline writes output to:
        generated_codes/experiment/exp_xsbench_translate_cuda_to_sycl_<model>_<vendor>__<timestamp>/
    Since directories are timestamp-suffixed, the last alphabetically is the newest.
    """
    exp_base = f"{REMOTE_BASE}/generated_codes/experiment"
    try:
        t = fs_resource.fs.ls(exp_base)
        t.wait(timeout=30)
        entries = t.result or []
    except Exception as e:
        print(f"Warning: could not list experiment directory: {e}")
        return None

    # entries may be autogen objects, dicts, or plain strings depending on
    # what the facility returns — extract the name field defensively
    names = []
    for e in entries:
        if isinstance(e, str):
            names.append(e)
        elif isinstance(e, dict):
            names.append(e.get("name") or e.get("path") or str(e))
        elif hasattr(e, "name"):
            names.append(e.name)
        else:
            names.append(str(e))

    exp_dirs = sorted(n for n in names if "exp_xsbench" in n)
    if not exp_dirs:
        return None

    # Main_v1_Trial1.cpp is written only on a fully successful translation
    return f"{exp_base}/{exp_dirs[-1]}/Main_v1_Trial1.cpp"


# ── Download ──────────────────────────────────────────────────────────────────

def download_file(fs_resource, remote_path: str, local_dest: pathlib.Path) -> None:
    """Download a remote file and save it locally."""
    t = fs_resource.fs.download(remote_path)
    t.wait(timeout=60)

    content = t.result
    if content is None:
        raise RuntimeError(f"Download returned no content for {remote_path}")
    if isinstance(content, str):
        content = content.encode("utf-8")

    local_dest.parent.mkdir(parents=True, exist_ok=True)
    local_dest.write_bytes(content)
    print(f"Saved to: {local_dest}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from amsc_client import Client

    # Authenticate — triggers an interactive Globus browser login on first run;
    # subsequent runs reuse the cached token at ~/.amsc/credentials.json.
    print("Authenticating with ALCF via Globus...")
    client = Client(auth_method="globus")
    alcf = client.facility("alcf")
    resource = alcf.resource(RESOURCE_NAME)
    print(f"Resource: {resource.name}  status={resource.status}")

    # Eagle is the storage resource for all /eagle/... filesystem operations.
    # Filesystem API calls must go through a storage resource — using the compute
    # resource (Polaris/Aurora) returns 501 "not supported yet".
    eagle = alcf.resource("Eagle")
    print(f"Filesystem: {eagle.name}  status={eagle.status}")

    if RESOURCE_NAME not in RESOURCE_SCRIPTS:
        raise ValueError(
            f"No configuration for {RESOURCE_NAME!r}. "
            f"Add an entry to RESOURCE_SCRIPTS and RESOURCE_FILESYSTEMS."
        )
    script     = RESOURCE_SCRIPTS[RESOURCE_NAME]
    filesystem = RESOURCE_FILESYSTEMS[RESOURCE_NAME]
    print(f"Script: {script}")

    # ── 1. File transfer
    # TODO: replace this block with upload_files(resource) once amsc_client
    # implements filesystem mkdir and upload (currently both return 501).
    print("\nSkipping automated file transfer — transfer files manually first.")
    print(f"  Target: {REMOTE_BASE}/")

    # ── 2. Submit job
    print("\nSubmitting job...")
    job = resource.submit(
        executable=script,
        directory=REMOTE_BASE,
        name="lassi_translate",
        queue=QUEUE,
        account=ACCOUNT,
        duration=DURATION,
        nodes=NODES,
        stdout_path=STDOUT_PATH,
        stderr_path=STDERR_PATH,
        filesystems=filesystem,
    )
    print(f"Job ID: {job.id}  initial state: {job.state}")

    # ── 3. Stream stdout while job runs
    stream_stdout(eagle, STDOUT_PATH, job)

    # ── 4. Read stderr unconditionally
    # PBS removes completed jobs from the queue, so a final state of 'active'
    # means the job ended before the next poll — not that it is still running.
    # Use exit_code (available post-completion) rather than state for failure detection.
    exit_code = getattr(job, "exit_code", None)
    print(f"Job finished — final state: {job.state}  exit_code: {exit_code}")
    has_stderr = drain_stderr(eagle, STDERR_PATH)

    # Stop if the job failed. Common causes when state stays 'queued' with
    # exit_code=0: the executable script is not chmod +x on the remote filesystem.
    if job.state == "failed" or (exit_code is not None and exit_code != 0) or has_stderr:
        return

    # ── 5. Download the translated SYCL file
    print("\nSearching for translated SYCL output...")
    sycl_remote = find_sycl_output(eagle)

    if sycl_remote:
        print(f"Found: {sycl_remote}")
        local_dest = LOCAL_OUTPUT / "Main_v1_Trial1.cpp"
        download_file(eagle, sycl_remote, local_dest)
    else:
        print("No successful translation found (all 15 self-correction attempts may have failed).")
        print(f"Inspect partial results at: {REMOTE_BASE}/generated_codes/experiment/")


if __name__ == "__main__":
    main()
