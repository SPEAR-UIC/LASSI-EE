# Polaris Job Runner — Setup, API Notes & Known Issues

`run_on_polaris.py` (one level up) submits the `lassi_translate` pipeline to ALCF Polaris
or Aurora via `amsc_client`, streams stdout, and downloads the resulting SYCL file.

---

## One-time setup on Polaris

All steps below are done once on a Polaris login node and persist across job submissions.

### 1. Create the inference Python venv

The compute job uses a venv built on top of Polaris's base conda environment.
`conda` is not in PATH by default — load it via the module system first.

```bash
module use /soft/modulefiles; module load conda; conda activate base
python -m venv /eagle/EE-ECP/LASSI-EE/venvs/inference --system-site-packages
source /eagle/EE-ECP/LASSI-EE/venvs/inference/bin/activate
pip install globus_sdk openai
```

### 2. Install and authenticate the ALCF Inference token helper

The pipeline calls the ALCF Inference API from compute nodes and needs a token
helper script at `~/inference_auth_token.py` on the Polaris home filesystem.

```bash
# Download to home directory
wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py \
     -O ~/inference_auth_token.py

# Activate the inference venv, then authenticate
module use /soft/modulefiles; module load conda; conda activate base
source /eagle/EE-ECP/LASSI-EE/venvs/inference/bin/activate
python ~/inference_auth_token.py authenticate
```

Tokens are valid for 48 hours; `get_access_token` refreshes them automatically.
Full re-authentication is required every 30 days.

**Desired future behaviour:** authenticate locally and inject the token into the job
environment via `amsc_client`. Currently blocked — `job submit environment` returns
501 `"'environment' not supported yet"`, and tokens expire in 48 h making a one-time
injection non-durable regardless. When both issues are resolved this step can be
eliminated entirely.

### 3. Transfer source files to Eagle

`amsc_client` does not yet support remote file upload (returns 501). Transfer manually:

```bash
rsync -av --exclude=__pycache__ --exclude=generated_codes \
      --exclude=Main --exclude=XSBench --exclude='*.o' \
      lassi_translate/ <user>@polaris.alcf.anl.gov:/eagle/EE-ECP/LASSI-EE/lassi_translate/
```

---

## amsc_client known limitations (501 Not Implemented)

| Operation | Error | Workaround |
|-----------|-------|------------|
| `filesystem mkdir` (parent option) | 501 | Create directories manually on Polaris |
| `filesystem upload` | 501 | Use rsync/scp manually |
| `filesystem tail` | 501 on Eagle | Use `fs.view(offset=0, size=N)` instead |

## SYCL toolchain on Polaris

Polaris uses Intel oneAPI DPC++ for SYCL, provided by ALCF (not HPE). The correct
module is `oneapi/release` — plain `module load oneapi` does not exist.

Compiler and flags:
```bash
module load oneapi/release
icpx -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
     -Xsycl-target-backend --cuda-gpu-arch=sm_80 <source>
```

`run_polaris_demo.sh` exports these as `SYCL_COMPILER` and `SYCL_CXXFLAGS` for the
pipeline to pick up via `lassi/sycl_toolchain.py`.
| `job submit environment` | 501 | Shell scripts set their own env defaults |

---

## amsc_client API behaviour notes

### Use the Eagle storage resource for filesystem operations
All `fs.*` calls on `/eagle/...` paths must go through `alcf.resource("Eagle")`,
**not** `alcf.resource("Polaris")`. Using the compute resource returns:
```
400: "Polaris not supported yet"
```

### PBS job completion and job state
PBS removes completed jobs from its queue. After removal, `job.refresh()` raises a
400/404 "not found" error — this is treated as terminal (the job has finished).
Because the removal happens before the next poll, `job.state` will read `"active"`
(the last successfully polled state) rather than `"completed"`.

Use `job.exit_code` (0 = success, non-zero = failure) rather than `job.state` for
reliable failure detection.

### fs.view() return format
`fs.view()` returns a dict rather than raising an exception when a file is missing:

| Situation | Result |
|-----------|--------|
| File does not exist | `{'error': 'File /path/... does not exist.'}` |
| File exists | `{'output': {'content': '...', 'content_type': 'bytes', 'start_position': N, 'end_position': N}}` |

The `_extract_view_content()` helper in `run_on_polaris.py` handles both cases.

---

## Common errors and fixes

### "job file is not executable"
`run_polaris_demo.sh` lacks execute permissions on the remote filesystem.

The executable bit is now tracked in git — `git pull` on Polaris sets it automatically.
For an immediate fix without an interactive login:

```bash
ssh <user>@polaris.alcf.anl.gov "chmod +x /eagle/EE-ECP/LASSI-EE/lassi_translate/run_polaris_demo.sh"
```

### "ALCF token helper not found at ~/inference_auth_token.py"
`inference_auth_token.py` is missing from the Polaris home directory.
See **One-time setup, step 2** above.

### "would exceed queue generic's per-user limit of jobs in 'Q' state"
The PBS debug queue has a per-user queued job limit. Clear stale jobs from previous
runs before resubmitting:

```bash
qselect -u <username> -s Q | xargs qdel
```

### Job exits with non-zero exit_code and empty stdout
If the stdout file is never created, the job likely failed before the pipeline ran
(e.g., missing source files, missing conda env, or token auth failure). Check stderr —
`run_on_polaris.py` reads and prints it automatically after every job.
