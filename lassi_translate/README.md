# lassi_translate

LLM-driven CUDA → SYCL translation pipeline for ALCF Polaris / Aurora.
An LLM proposes a SYCL translation of the provided CUDA source; the pipeline
compiles + runs it on a real GPU, verifies output against the original, and
iterates with the LLM on each error (up to 15 self-correction attempts).
An LLM-as-judge step verifies functional equivalence on each successful run.

---

## Modes

| Mode | What runs where | LLM |
|---|---|---|
| **Remote** (default) | Workstation orchestrates via amsc; persistent PBS worker compiles + runs | Argo |
| **`--on-resource`** | Everything runs inside an interactive `qsub -I` shell on a compute node | ALCF Inference |
| **`--submit`** | Workstation submits one PBS job via amsc that runs `--on-resource`; log streams back live | ALCF Inference |

Filesystem operations + job submission + state polling all go through
[amsc-client](https://pypi.org/project/amsc-client/). `rsync` over SSH is used
only for uploads (amsc has no upload API) and for fetching results.

---

## Setup (one-time, ~5 min)

**1. Argonne-managed machine, on-site or VPN, Python 3.11+.** Create a venv
   at the project root and install dependencies:
   ```bash
   cd /path/to/LASSI-EE/lassi_translate
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Activate the venv (`source .venv/bin/activate`) in any new shell.

**2. Set your ALCF username in `lassi_config.py`:**
   ```python
   "alcf_username": "your_username",
   ```

**3. (Strongly recommended) Register an SSH key with ALCF** so you don't get
   prompted for a password on every run. Generate `~/.ssh/id_ed25519` if you
   don't have one, then upload its `.pub` at
   <https://accounts.alcf.anl.gov> → "SSH Public Keys". Propagation: ~10 min.

That's it for **remote** and **`--submit`** modes — they handle Polaris-side
auth (Globus, browser flow on first use, cached for 24h+) automatically.

**4. (Only for `--on-resource`)** Put the project on Eagle and authenticate
   to ALCF Inference once:
   ```bash
   # From your workstation:
   rsync -avzL --delete --exclude='__pycache__/' --exclude='generated_codes/' --exclude='.git/' \
     /path/to/LASSI-EE/lassi_translate/ \
     your_username@polaris.alcf.anl.gov:/eagle/EE-ECP/LASSI-EE/lassi_translate/
   ssh your_username@polaris.alcf.anl.gov \
     'module use /soft/modulefiles && module load conda && conda activate base &&
      python3 /eagle/EE-ECP/LASSI-EE/lassi_translate/inference_auth_token.py authenticate'
   ```
   (Browser flow; tokens cache for 30 days.)

---

## Verify setup

Before any real run, run the preflight check from the project root:
```bash
python debug.py            # ~10s, covers static + amsc + SSH + inference auth
python debug.py --full     # ~5min, also queues a tiny PBS test job
python debug.py --no-amsc  # offline only (skips ALCF connectivity)
```
The output ends with a per-mode readiness verdict — fix any `FAIL` rows before
running the pipeline.

---

## Run

### Remote (default)
From your workstation:
```bash
python run_lassi.py
```
Uses defaults from `lassi_config.py` (polaris, translate, XSBench, Argo
`gpt41`). Streams Stage 0 (self-prompt) → Stage 1 (initial translation) →
Stage 2 (self-correction loop), then prints a side-by-side summary.

### `--submit` (fire-and-forget, recommended for real runs)
From your workstation:
```bash
python run_lassi.py --submit --task translate --app-name XSBench --walltime 1800
```
The pipeline auto-syncs the project + Globus tokens to Eagle, submits one PBS
job via amsc, and polls the job's stdout via `amsc.fs.view` so you see live
log output. On completion, results are auto-rsynced back to
`generated_codes/experiment/<exp>/`.

**Ctrl-C** detaches without killing the job. Cancel via:
```bash
ssh your_username@polaris.alcf.anl.gov 'qdel <pbs_id>'
```

### `--on-resource` (interactive PBS shell)
On a Polaris login node:
```bash
ssh your_username@polaris.alcf.anl.gov
qsub -I -A EE-ECP -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle
# Then on the compute node:
cd /eagle/EE-ECP/LASSI-EE/lassi_translate
module use /soft/modulefiles && module load conda && conda activate base
python3 run_lassi.py --on-resource --task translate --app-name XSBench
```
At the end, the pipeline prints a ready-to-paste `rsync` command for pulling
results back to your workstation.

---

## Common overrides

```bash
# Different Argo model (remote mode)
python run_lassi.py --argo-model claudesonnet46

# Different ALCF Inference model
python run_lassi.py --submit --inference-model meta-llama/Meta-Llama-3.1-405B-Instruct

# Different inference cluster
python run_lassi.py --submit --inference-cluster metis

# Custom Python on the compute node (skip conda discovery)
python run_lassi.py --submit --remote-python ~/.conda/envs/myenv/bin/python3

# Toggle LLM-as-judge (default from lassi_config.py "use_judge")
python run_lassi.py --no-judge       # skip judge this run
python run_lassi.py --judge          # force judge on this run

# Skip auto-fetch (--submit only); results stay on Eagle
python run_lassi.py --submit --no-fetch-results

# Override fetch destination
python run_lassi.py --submit --fetch-dest /scratch/lassi_runs
```

Full flag list: `python run_lassi.py --help`.

---

## Output

Each run writes to `generated_codes/experiment/exp_<app>_<task>_<model>_<vendor>__<timestamp>/`:
```
├── Main_v1_Trial1.cpp         ← final successful translation (on success)
├── Main_attemptN.cpp          ← what the LLM produced on attempt N
├── attemptN_*.log             ← build/run logs per attempt
├── baseline_*.log             ← test run of the original source
├── metadata_<exp>.txt         ← formatted summary
├── results.csv                ← single-row outcome
└── llm_calls/call_NNNN/       ← every LLM prompt + response (debug)
```

For **`--submit`** the PBS log also lives at:
```
/eagle/EE-ECP/LASSI-EE/lassi_jobs/generated_codes/experiment/submit_logs/lassi_submit_<timestamp>.log
```

---

## Debug

| Symptom | First step |
|---|---|
| Anything fails on startup | `python debug.py` — pinpoints which check broke |
| `[LLM ERROR]` block streams | Read the one-line hint in the block; fix and retry |
| Password prompts on every rsync | Register an SSH key (see Setup step 3) |
| Mid-run pipeline silent | Check the streamed log; PBS state heartbeats appear when the log is empty |
| `--submit` job state stuck | `ssh polaris qstat -u $USER` — if gone, the host-side poll loop will detect it within ~15s |
| Stranded PBS job after Ctrl-C | `ssh polaris 'qdel <jobid>'` |
| Want to deep-dive amsc API behavior | `python tools/amsc_smoke.py` (fs round-trip) or `python tools/amsc_probe.py` (raw result shapes) |

Every LLM call's prompt + response is saved under `llm_calls/call_NNNN/` in
the experiment dir — useful for debugging model behavior.

---

## Scope and limitations

| Task | Polaris (NVIDIA) | Aurora (Intel) |
|---|---|---|
| `--task translate` (CUDA → SYCL) | ✓ via icpx CUDA backend | ✓ via icpx PVC backend |
| `--task energy` (CUDA energy opt) | ✓ (remote mode only) | ✗ |

Other limits:
- `--task energy` works only in remote mode (not `--on-resource` / `--submit`).
- `gpt55` (Argo) requires `temperature=1`; the pipeline sends `0.2`, so it doesn't work yet.
- ALCF amsc-client filesystem endpoints `stat`, `mv`, `download` return 501 ("not implemented"). The pipeline works around all three via `ls` (for stat), atomic `rsync` (for mv), and `rsync` pull (for download). Don't add code that depends on these endpoints.
- Only XSBench has a built-in checksum check; other apps rely on the LLM-as-judge step.

---

## Files

```
lassi_translate/
├── run_lassi.py                CLI + mode dispatch (~800 lines)
├── debug.py                    Preflight check for all modes
├── lassi_config.py             Runtime defaults
├── source_apps.json            App catalog
├── requirements.txt            Workstation deps (amsc-client, globus-sdk)
├── inference_auth_token.py     Vendored Globus auth helper for ALCF Inference
├── source_codes -> ../source_codes
├── lassi/                      Pipeline library (14 small modules + context/)
├── workers/                    Bash scripts that run on the compute node
└── tools/                      amsc smoke test + raw-shape probe
```
