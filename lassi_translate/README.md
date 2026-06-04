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

Argonne-managed machine, on-site or VPN, Python 3.11+:
```bash
pip install amsc-client>=0.4.1 globus-sdk>=3.0
```

Set your ALCF username in `lassi_config.py`:
```python
"alcf_username": "your_username",
```

That's it — `--submit` mode handles its own Polaris-side setup automatically.

### One-time on Polaris (only for `--on-resource` mode)

```bash
ssh your_username@polaris.alcf.anl.gov
module load conda
conda activate base
python ~/lassi_translate/inference_auth_token.py authenticate
```
(Browser flow; tokens cache for 30 days.)

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

Rsync the project to Polaris first:
```bash
# Preserves the source_codes symlink layout
rsync -avz --delete --exclude='__pycache__/' --exclude='generated_codes/' --exclude='.git/' \
  /path/to/LASSI-EE/lassi_translate/ your_username@polaris.alcf.anl.gov:~/lassi_translate/
rsync -avz --delete /path/to/LASSI-EE/source_codes/ \
  your_username@polaris.alcf.anl.gov:~/source_codes/
```

Grab a compute node:
```bash
ssh your_username@polaris.alcf.anl.gov
qsub -I -A EE-ECP -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle
```

Run on the compute node:
```bash
cd ~/lassi_translate
module load conda && conda activate base
python run_lassi.py --on-resource --task translate --app-name XSBench
```

---

## Recipe 3 — Submit (fire-and-forget, recommended for real runs)

From your workstation — nothing else, no manual Polaris setup:
```bash
python lassi_translate/run_lassi.py --submit \
    --task translate --app-name XSBench --walltime 1800
```

The pipeline:
1. Connects to Polaris over SSH (one password prompt).
2. **First-ever run only:** a Globus auth URL prints; paste it into a browser,
   then paste back the resulting code. Tokens cache locally.
3. Every ~30 days, prompts you to re-auth automatically (also browser).
4. Syncs tokens + code to Polaris, submits one PBS job, streams its stdout
   back to your terminal.

**Ctrl-C** detaches without killing the job; reattach with the printed command:
```
ssh your_username@polaris.alcf.anl.gov \
    'tail -F /eagle/.../submit_logs/lassi_submit_<timestamp>.log'
```

Kill a stranded job:
```bash
ssh your_username@polaris.alcf.anl.gov 'qdel <pbs_id>'
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

## What's not supported

- `--task energy` and `--task both` only work in **remote mode** (not `--on-resource` / `--submit`).
- `gpt55` (Argo) requires `temperature=1`; the pipeline sends `0.2`, so it doesn't work yet.
- Aurora's Flare filesystem isn't in IRI — file reads fall back to SSH (slower but transparent).
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
