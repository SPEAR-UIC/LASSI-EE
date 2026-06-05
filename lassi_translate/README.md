# lassi_translate

LLM-driven CUDA → SYCL translation pipeline for ALCF Polaris / Aurora.
Given a CUDA source, an LLM proposes a SYCL translation; the pipeline
compiles + runs it on the target GPU, verifies output, and iterates with
the LLM on any error — up to 15 self-correction attempts.

Three ways to run it:

| Mode | What runs where | LLM |
|---|---|---|
| **Remote** (default) | Workstation orchestrates; persistent PBS job handles compile+run | Argo |
| **`--on-resource`** | Everything runs inside an interactive `qsub -I` shell on the compute node | ALCF Inference |
| **`--submit`** | Workstation submits one PBS job that runs the whole pipeline; output streams back live | ALCF Inference |

---

## Setup

### One-time on your workstation (all modes)

Argonne-managed machine, on-site or VPN, Python 3.11+. Create a venv at the
project root and install the dependencies pinned in `requirements.txt`:
```bash
cd /path/to/LASSI-EE/lassi_translate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the venv (`source .venv/bin/activate`) in any new shell before running
`run_lassi.py`. Deactivate with `deactivate`.

Set your ALCF username in `lassi_config.py`:
```python
"alcf_username": "your_username",
```

That's it — `--submit` mode handles its own Polaris-side setup automatically.

---

## Recipe 1 — Remote mode (default)

From your workstation:
```bash
python lassi_translate/run_lassi.py
```
This uses the defaults in `lassi_config.py` (Polaris, translate task, XSBench,
Argo model `gpt41`). You'll see the connect banner, a test run of the
original CUDA, Stage 0/1/2 with a live sliding-window of the LLM stream,
and a final side-by-side summary.

---

## Recipe 2 — On-resource (interactive PBS shell)

### One-time on Polaris (only for `--on-resource` mode)

**Step 1.** From your workstation, rsync the project to Eagle (`-L`
dereferences the `source_codes` symlink so the app sources go along too):
```bash
rsync -avzL --delete --exclude='__pycache__/' --exclude='generated_codes/' --exclude='.git/' \
  /path/to/LASSI-EE/lassi_translate/ \
  vchirio@polaris.alcf.anl.gov:/eagle/EE-ECP/LASSI-EE/lassi_translate/
```
(Replace `vchirio` with your ALCF username and `/path/to/LASSI-EE/` with
wherever the repo lives on your workstation.)

**Step 2.** SSH to a Polaris login node and authenticate to ALCF Inference
(browser flow; tokens cache for 30 days):
```bash
ssh vchirio@polaris.alcf.anl.gov
module use /soft/modulefiles
module load conda
conda activate base
python3 /eagle/EE-ECP/LASSI-EE/lassi_translate/inference_auth_token.py authenticate
```

(`--submit` mode does both of these from your workstation automatically and
rsyncs the result to Polaris — no manual Polaris-side setup needed.)

Grab a compute node:
```bash
ssh vchirio@polaris.alcf.anl.gov
qsub -I -A EE-ECP -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle
```

Run on the compute node (note: `module use /soft/modulefiles` is required
on Polaris compute nodes — the conda module isn't on the default MODULEPATH;
and the conda env exposes `python3`, not `python`):
```bash
cd /eagle/EE-ECP/LASSI-EE/lassi_translate
module use /soft/modulefiles
module load conda
conda activate base
python3 run_lassi.py --on-resource --task translate --app-name XSBench
```

(If for some reason you must keep the project in `$HOME`, redirect output
to Eagle explicitly with `--output-dir /eagle/EE-ECP/LASSI-EE/lassi_runs`.
But Eagle-resident is the cleaner setup.)

---

## Recipe 3 — Submit (fire-and-forget, recommended for real runs)

From your workstation — nothing else, no manual Polaris setup:
```bash
python lassi_translate/run_lassi.py --submit \
    --task translate --app-name XSBench --walltime 1800
```

The pipeline:
1. Authenticates to ALCF via Globus (amsc-client) — opens a browser the first
   time, then caches tokens for ~24 h.
2. **First-ever ALCF Inference run only:** a second Globus auth URL prints
   for the inference scope; paste it into a browser, then paste back the code.
3. Every ~30 days the inference token re-auths automatically (browser).
4. Uploads tokens + code to the remote via `rsync` (the only remaining SSH
   usage — amsc has no upload API), submits one PBS job via `amsc`, then
   polls the job's stdout log via `amsc.fs.view` and streams it to your terminal.

**Ctrl-C** detaches without killing the job. To cancel the running job, use
the amsc Job API from a Python shell:
```python
from amsc_client import Client
job = Client(token='not-needed').facility('alcf').resource('Polaris').jobs(['<pbs_id>'])[0]
job.cancel()
```

---

## Common overrides

```bash
# Different Argo model (remote mode)
python lassi_translate/run_lassi.py --argo-model claudesonnet46

# Different ALCF Inference model (--on-resource / --submit)
python lassi_translate/run_lassi.py --submit \
    --inference-model meta-llama/Meta-Llama-3.1-405B-Instruct

# Different ALCF Inference cluster
python lassi_translate/run_lassi.py --submit --inference-cluster metis

# Custom Python interpreter on the compute node (skips conda discovery)
python lassi_translate/run_lassi.py --submit \
    --remote-python ~/.conda/envs/myenv/bin/python3

# Toggle the LLM-as-judge functional-equivalence check.  Default comes from
# lassi_config.py ("use_judge": True); --judge / --no-judge override per run.
python lassi_translate/run_lassi.py --no-judge      # skip judge this run
python lassi_translate/run_lassi.py --judge         # force judge on this run
```

Full flag list: `python lassi_translate/run_lassi.py --help`.

---

## Output

Every run writes to:
```
generated_codes/experiment/exp_<app>_<task>_<model>_<vendor>__<timestamp>/
├── Main_v1_Trial1.cpp        ← final successful translation (on success)
├── Main_attemptN.cpp         ← what the LLM produced on attempt N
├── attemptN_*.log            ← build/run logs per attempt
├── baseline_*.log            ← test run of the original source
├── metadata_<exp>.txt        ← formatted summary
├── results.csv               ← single-row outcome
└── llm_calls/call_NNNN/      ← every LLM prompt + response, for debugging
```

For `--submit` mode, the PBS stdout also lives at:
```
/eagle/EE-ECP/LASSI-EE/lassi_jobs/generated_codes/experiment/submit_logs/lassi_submit_<timestamp>.log
```

### Retrieving results back to your workstation

**Remote mode** — already on your workstation under `generated_codes/`. Nothing to fetch.

**`--submit` mode** — automatic. When the PBS job finishes, the workstation
detects completion (background `qstat` poll), then rsyncs the experiment
directory back to `lassi_translate/generated_codes/experiment/<exp>/`. No
extra step needed.

Override the local destination with `--fetch-dest`, or disable auto-fetch
entirely with `--no-fetch-results` (the data still lives on Eagle —
retrieve it manually with the `--on-resource` recipe below).

If you Ctrl-C to detach before the job finishes, auto-fetch is skipped (the
job is still running). After it completes on its own, pull manually:
```bash
rsync -avz \
  vchirio@polaris.alcf.anl.gov:/eagle/EE-ECP/LASSI-EE/lassi_translate/generated_codes/experiment/<exp_dir>/ \
  ./generated_codes/experiment/<exp_dir>/
```

**`--on-resource` mode** — at the end of the run, the pipeline prints the
exact `rsync` command (with your username and the experiment path filled
in) to paste into a *workstation* shell. Example:
```bash
rsync -avz \
  vchirio@polaris.alcf.anl.gov:/eagle/EE-ECP/LASSI-EE/lassi_translate/generated_codes/experiment/exp_XSBench_translate_..._t11_30_36 \
  ./pulled_results/
```

Quick peek without pulling — read the one-line outcome remotely:
```bash
ssh vchirio@polaris.alcf.anl.gov \
  'cat /eagle/EE-ECP/LASSI-EE/lassi_translate/generated_codes/experiment/<exp_dir>/results.csv'
```

The exact experiment directory name is printed in the run banner as
`Experiment : exp_...`.

---

## Troubleshooting

**`[LLM ERROR]` block appears in the stream**
The block tells you exactly what failed (backend, model, cluster, reason) and a one-line hint. Follow the hint.

**`30-day re-auth required`** (`--submit` mode)
The workstation auto-runs `authenticate --force`. Visit
https://app.globus.org/logout in a browser if it complains, then re-run.

**`module load conda did not work`** (PBS log)
Normal on Polaris — the script falls back to sourcing `conda.sh` directly.

**`pip install` fails on the compute node**
Compute nodes can't reach PyPI. The pipeline doesn't try to install anything
there; if you've been editing the PBS script and added a pip line, remove it.

**Persistent worker / submit job stuck after Ctrl-C**
```bash
ssh your_username@polaris.alcf.anl.gov 'qstat -u $USER -w | grep lassi'
ssh your_username@polaris.alcf.anl.gov 'qdel <jobid>'
```

---

## Scope and limitations

**Supported tasks (CUDA input only):**
| `--task` | Polaris | Aurora |
|---|---|---|
| `translate` (CUDA → SYCL) | ✓ (verify via icpx CUDA backend) | ✓ (verify via icpx PVC backend) |
| `energy` (CUDA → CUDA optimization) | ✓ (remote mode only) | ✗ |

**Other limits:**
- `--task energy` only works in **remote mode** (not `--on-resource` / `--submit`).
- `gpt55` (Argo) requires `temperature=1`; the pipeline sends `0.2`, so it doesn't work yet.
- Both resources require their filesystems to be registered in ALCF's IRI (Polaris/Eagle, Aurora/Flare). The pipeline does not fall back to SSH for filesystem operations — if amsc can't reach the configured filesystem, the run fails immediately with a clear error.
- Only XSBench has a built-in checksum check; other apps rely on the LLM-as-judge step.

---

## Files

```
lassi_translate/
├── run_lassi.py                  CLI + dispatch (~1100 lines)
├── lassi_config.py               runtime defaults
├── source_apps.json              app catalog
├── inference_auth_token.py       vendored Globus auth helper (ALCF Inference)
├── source_codes -> ../source_codes
├── lassi/                        pipeline library (10 small modules)
└── workers/                      bash scripts that run on the compute node
```
