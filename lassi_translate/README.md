# lassi_translate

LLM-driven CUDA → SYCL translation and GPU energy optimization pipeline for
ALCF Polaris and Aurora.

Given a CUDA source file, `lassi_translate` will:

1. Ask an LLM to translate it to SYCL (or optimize an existing CUDA/SYCL file
   for energy efficiency).
2. Compile and run the generated code on the target ALCF compute resource.
3. Verify correctness (XSBench checksum + LLM-as-judge functional equivalence).
4. On any failure, feed the error back to the LLM and try again, up to 15
   self-correction attempts.

The pipeline runs end-to-end without human intervention. Each run produces a
versioned source file, a metadata report, full per-attempt logs, and a CSV of
the experiment outcome.

---

## Two ways to run it

You pick one of two execution modes depending on where you're running from.

### 1. Remote mode (default) — from your workstation

Your workstation orchestrates everything. The pipeline opens a Globus + SSH
connection to Polaris/Aurora, rsyncs the project to the remote `/eagle/...`
or `/flare/...` directory, submits one long-running "persistent worker" PBS
job, and uses it to compile + run each generated attempt. LLM calls go to
Argo (`apps.inside.anl.gov`).

```
python lassi_translate/run_lassi.py
```

### 2. On-resource mode — from an interactive PBS shell on the compute node

Run the whole pipeline directly inside a `qsub -I` session. No SSH dance, no
PBS submission per attempt, no persistent worker — the worker scripts are
invoked as plain subprocesses. LLM calls go to the ALCF Inference Endpoints
API (Argo is not reachable from compute nodes).

```
python lassi_translate/run_lassi.py --on-resource
```

---

## Installation

### Workstation (for remote mode)

You need an Argonne-managed machine on-site or on VPN, plus a Python 3.11+
environment:

```bash
pip install amsc-client>=0.4.1 globus-sdk>=3.0
```

The default `lassi_config.py` already points at Polaris and the EE-ECP
allocation. Edit it if you want different defaults (model, queue, app, etc.).

### Polaris/Aurora compute node (for on-resource mode)

After grabbing an interactive node, you need:

```bash
module load conda
conda activate base    # or any env with Python 3.11+

pip install openai globus_sdk

# One-time Globus authentication (opens a browser auth flow)
python lassi_translate/inference_auth_token.py authenticate

# Verify
python lassi_translate/inference_auth_token.py get_time_until_token_expiration --units hours
```

Tokens are cached in `~/.globus/app/<client>/inference_app/tokens.json` and
refreshed automatically. An internal policy forces re-authentication every
30 days; if you hit a permission error, re-run with `--force`.

---

## Quick start

### Translate XSBench from CUDA to SYCL on Polaris (remote mode)

```bash
# From your workstation
python lassi_translate/run_lassi.py
```

This runs with the defaults in `lassi_config.py`: Polaris, translate task,
XSBench, model `gpt41`. You'll see a test run of the original CUDA source,
then the Stage 0/1/2 LLM loop, then a final side-by-side summary.

### Same thing, on-resource

```bash
# 1. From your workstation: rsync the code over
rsync -avzL --delete \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='generated_codes/' --exclude='.git/' \
  /path/to/LASSI-EE/lassi_translate/ \
  vchirio@polaris.alcf.anl.gov:~/lassi_translate/
rsync -avz --delete \
  /path/to/LASSI-EE/source_codes/ \
  vchirio@polaris.alcf.anl.gov:~/source_codes/

# 2. SSH in and grab a compute node
ssh vchirio@polaris.alcf.anl.gov
qsub -I -A EE-ECP -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle

# 3. On the compute node
cd ~/lassi_translate
module load conda
python run_lassi.py --on-resource --task translate --app-name XSBench
```

---

## Tasks

Three task types, selected with `--task`:

| Task | Source → Target | Resources | What it does |
|---|---|---|---|
| `translate` | CUDA → SYCL | Polaris, Aurora | Translate CUDA to SYCL with self-correction. Reports duration + stdout for both. |
| `energy` | CUDA → CUDA (Polaris) or SYCL → SYCL (Aurora) | Polaris, Aurora | Iteratively optimize for GPU energy efficiency. Reports energy reduction vs baseline. |
| `both` | CUDA → SYCL → optimized SYCL | Polaris, Aurora | Translate first, then run the energy task on the translated SYCL. |

Aurora cannot compile or run CUDA. For `--task energy` on Aurora you must
pass an existing SYCL source via `--sycl-source PATH` and
`--input-framework sycl`, or use `--task both` to translate first.

The `--task` framework compatibility matrix is encoded in
`lassi/resources.py:FRAMEWORKS`.

---

## Configuration

### `lassi_config.py` — defaults file

A plain Python file loaded at startup. Edit values here to change defaults;
CLI flags still override anything you set:

```python
config = {
    "resource":        "polaris",     # polaris (NVIDIA A100) | aurora (Intel PVC)
    "task":            "translate",   # translate | energy | both
    "input_framework": "cuda",        # cuda | sycl
    "alcf_username":   "vchirio",
    "app_category":    "Miniapps",    # category in source_apps.json
    "app_name":        "XSBench",
    "account":         "EE-ECP",      # PBS allocation
    "queue":           "debug",       # PBS queue
    "walltime":        3600,          # persistent worker walltime (seconds)
    "model":           "gpt41",       # see "Models" below
    "min_energy_gain": 5.0,           # % reduction required for energy success
    "use_judge":       True,          # run LLM-as-judge after each attempt
}
```

Use `--config PATH` to load a different file.

### CLI flags — full list

Run `python run_lassi.py --help` for the authoritative list. The flags below
are the ones you reach for most often.

**Resource and task**
- `--resource {polaris,aurora}` — target HPC
- `--task {translate,energy,both}` — what to do
- `--input-framework {cuda,sycl}` — source language
- `--app-category CATEGORY` and `--app-name APP` — pick an app from `source_apps.json`
- `--sycl-source PATH` — explicit SYCL file (Aurora energy with pre-existing SYCL)

**Identity and PBS**
- `--alcf-username USER` (default: `$USER`)
- `--remote-dir PATH` — override the default `/eagle/...` or `/flare/...`
- `--skip-sync` — don't rsync; assume the remote tree is already current
- `--account NAME` and `--queue NAME` — PBS allocation and queue
- `--walltime SECONDS` — persistent worker walltime budget
- `--no-persistent-worker` — submit one PBS job per attempt instead (slow; useful for debugging)

**LLM and pipeline**
- `--model NAME` — see "Models" below
- `--no-self-prompting` — skip Stage 0 (context guide + source analysis)
- `--no-judge` — skip LLM-as-judge after each successful attempt
- `--min-energy-gain PCT` — minimum % reduction to declare energy success
- `--verifyprompt` — print full prompts before each LLM call (debugging)
- `--debug` — print prompt sizes + extra diagnostics

**On-resource mode**
- `--on-resource` — run locally on the compute node (no SSH/PBS/Globus)
- `--inference-cluster {sophia,metis}` — which ALCF Inference cluster to call

**Compiler overrides** (passed through to worker scripts)
- `--sycl-compiler PATH`, `--sycl-flags FLAGS`, `--sycl-ldflags FLAGS`
- `--cuda-compiler PATH`, `--cuda-flags FLAGS`, `--cuda-ldflags FLAGS`

---

## Models

### Remote mode (Argo)

The model name maps directly to an Argo model ID. Common picks:

| Model | When to use |
|---|---|
| `gpt55` | Strongest available; usually solves the translation in one shot. |
| `gpt41` | Default. Good code generation, makes real translation mistakes that the correction loop fixes in 1-3 attempts. Best for demos. |
| `claudesonnet46` / `claudeopus47` | Strong Anthropic models; good if GPT is struggling. |
| `gemini25pro` | Google. 1M context. |
| `gpt4o` | Avoid — known to produce stubbed translations of large files. |

See the Argo API documentation for the full list. Some models have parameter
restrictions (gpt55 requires `temperature=1`; Haiku/Sonnet 4.5+ reject having
both `temperature` and `top_p` set). The pipeline only sends `temperature`,
so the second restriction is satisfied automatically.

### On-resource mode (ALCF Inference Endpoints)

When `--on-resource` is set, the default model becomes
`meta-llama/Meta-Llama-3.1-70B-Instruct` on the Sophia cluster. Override
either piece independently:

```bash
python run_lassi.py --on-resource \
  --inference-cluster sophia \
  --model meta-llama/Meta-Llama-3.1-405B-Instruct
```

Available models per cluster: `curl` the discovery endpoint with your token,
or check the ALCF docs.

---

## How it works

The pipeline runs in three stages per attempt:

### Stage 0 — Self-prompting (Argo + local)

Two LLM calls that don't generate code:

1. Summarize the relevant context guide (SYCL programming, CUDA energy
   efficiency, or SYCL energy efficiency).
2. Analyze the source code structure and call out non-obvious algorithmic
   details (loop bounds, checksums, kernel layouts) so the model preserves
   them faithfully in Stage 1.

The two summaries are prepended to every subsequent prompt. Skipped if
`--no-self-prompting` is set.

### Stage 1 — Initial generation

One LLM call with the system prompt + Stage 0 context + the source code.
Output is parsed as a fenced code block (` ```cpp ... ``` `), written locally,
rsynced to the remote, and submitted to the persistent worker for compile +
run.

### Stage 2 — Self-correction loop (up to 15 attempts)

Each attempt:

1. Read `build_rc.txt` / `run_rc.txt` from the remote, parse compile and
   runtime status, and pull the relevant stderr/stdout.
2. Classify the failure: compile error, runtime error, XSBench checksum
   mismatch, LLM-judge `INVALID`, or (for energy tasks) energy improvement
   below threshold.
3. Build a structured correction prompt using a different template per
   classification, and call the LLM again at the same temperature (or
   escalated for energy-insufficient cases).
4. Submit the new code and loop.

Termination:

- Translate task: succeeds on the first compile-run-judge-pass attempt.
- Energy task: succeeds when the best attempt beats baseline by at least
  `--min-energy-gain` and stagnates (marginal gain < 1.1% for 4 consecutive
  improvements).
- Either: fails after 15 attempts.

### Persistent worker

To avoid waiting in the PBS queue once per attempt, remote mode submits a
single long-running PBS job that opens a file-based RPC channel on the
remote filesystem. The driver writes `trigger.json` (the work request) and
polls for `result_ready.json` (the compile + run outcome). The worker stays
alive for the full walltime budget, so subsequent attempts hand off in
seconds rather than queue waits. Disable with `--no-persistent-worker` to
fall back to per-attempt PBS submission.

In on-resource mode, the worker script is invoked as a direct subprocess —
no IPC, no PBS, no queue wait at all.

---

## Project layout

```
lassi_translate/
├── run_lassi.py              CLI + main dispatch (~500 lines)
├── lassi_config.py           Runtime defaults
├── source_apps.json          App catalog (Miniapps, HeCBench, etc.)
├── inference_auth_token.py   ALCF Globus auth helper (vendored, on-resource mode only)
├── source_codes              → symlink to ../source_codes (CUDA/SYCL app sources)
├── README.md                 This file
│
├── lassi/
│   ├── __init__.py
│   ├── resources.py          ResourceConfig, RESOURCES, WORKERS, FRAMEWORKS
│   ├── hpc_session.py        SSH + amsc filesystem wrapper (remote mode)
│   ├── local_session.py      Local FS wrapper (on-resource mode)
│   ├── alcf.py               Globus connect + SSH ControlMaster
│   ├── hpc_jobs.py           PBS submission, PersistentWorker, JobRunner
│   ├── local_runner.py       Subprocess runner (on-resource mode)
│   ├── source_loader.py      Find + load CUDA/SYCL source code
│   ├── results.py            Parse per-attempt result files
│   ├── llm_client.py         Argo + ALCF Inference backends, sliding-window display
│   ├── alcf_inference.py     ALCF Inference Endpoints client (OpenAI SDK)
│   ├── prompt_dictionary.py  All LLM prompt templates (do not edit casually)
│   ├── prompt_builder.py     Per-stage prompt assembly + LLM judge
│   ├── pipeline.py           run_pipeline, run_both_pipeline, summaries
│   ├── format_colors.py      ANSI color constants
│   └── context/              Reference markdown for self-prompting
│
├── workers/                  Bash scripts that run on the compute node
│   ├── _common.sh            Shared helpers (Lmod, oneAPI, CUDA, monitors)
│   ├── polaris_worker.sh             one-shot job
│   ├── polaris_persistent_worker.sh  long-running daemon
│   ├── aurora_worker.sh              one-shot job
│   └── aurora_persistent_worker.sh   long-running daemon
│
└── generated_codes/          Output directory (gitignored)
    └── experiment/
        └── exp_<app>_<task>_<model>_<vendor>__<timestamp>/
            ├── Main_attemptN.cpp      generated code per attempt
            ├── Main_v1_Trial1.cpp     final successful version
            ├── attemptN_*.log         per-attempt build/run logs
            ├── baseline_*.log         baseline test run logs
            ├── baseline_energy.json   baseline metrics
            ├── results.csv            single-row experiment summary
            ├── metadata_<exp>.txt     full experiment report
            └── llm_calls/             every LLM call's prompts + response
                └── call_NNNN/
                    ├── meta.json
                    ├── system.txt
                    ├── user.txt
                    └── response.txt
```

---

## Output

Every run lands in
`generated_codes/experiment/exp_<app>_<task>_<model>_<vendor>__<timestamp>/`.
The most useful files:

- `Main_v1_Trial1.cpp` — the final successful translation (only on success)
- `Main_attemptN.cpp` — what the LLM produced on attempt N (all attempts)
- `attemptN_build_stderr.log` — compile errors per attempt
- `attemptN_run_stdout.log` — program output per attempt
- `metadata_<exp>.txt` — formatted summary with timings, energy metrics, LLM-generated code comparison
- `results.csv` — single-row CSV suitable for aggregating across runs
- `llm_calls/call_NNNN/` — exact prompt and response for every LLM call, useful for prompt debugging

---

## Troubleshooting

**"Connection failed: ..." on startup**
Check VPN / on-site network, and that your ALCF SSH keys work for plain
`ssh polaris.alcf.anl.gov`. The pipeline opens one SSH ControlMaster — if
that fails, rsync would prompt per call and everything stalls.

**Every attempt shows `compile error: [Remote FS read error ... Path must be absolute]`**
The amsc Eagle filesystem API only accepts paths under `/eagle/` or
`/lus/eagle/`. Make sure `--remote-dir` (or `default_dir` in
`lassi/resources.py`) starts with `/eagle/`.

**Every attempt shows `compile error: [Remote FS read error ... Allowed base paths for filesystem Eagle are: /eagle, /lus/eagle]`**
Same root cause as above — you pointed `remote_dir` somewhere outside Eagle.

**The LLM keeps producing a 350-line stub instead of translating the whole file**
You're using `gpt4o`. Switch to `gpt41` or `gpt55` in `lassi_config.py`.

**`SystemExit: ALCF Inference: not authenticated yet` (on-resource mode)**
Run `python lassi_translate/inference_auth_token.py authenticate` once,
follow the Globus auth flow, then retry.

**A persistent worker PBS job is hanging around after a failed run**
```bash
ssh vchirio@polaris.alcf.anl.gov 'qstat -u $USER -w | grep lassi-persistent'
ssh vchirio@polaris.alcf.anl.gov 'qdel <jobid>'
```

**Job times out at 15 attempts with the same error every time**
The correction prompt isn't giving the model new information. Inspect
`llm_calls/call_NNNN/user.txt` to see what's actually being sent. If the
error stays identical across attempts, the issue is likely in the worker
script or compiler environment rather than the LLM.

---

## What's not supported

- `--task energy` and `--task both` are not yet wired through `--on-resource`;
  only `--task translate` works in on-resource mode.
- `gpt55` requires `temperature=1`, but the pipeline sends `temperature=0.2`.
  Other models that have hard parameter restrictions (`gpto1mini` etc.) are
  not wrapper-compatible.
- Aurora's Flare filesystem isn't registered in the IRI API, so file reads
  fall back to SSH. This is slower but transparent.
- Only XSBench has a built-in correctness check (`verification: "xsbench"` in
  `source_apps.json`). Other apps rely on the LLM-as-judge step alone.

---

## See also

- Parent project: [LASSI](../README.md)
- Argo Gateway API: https://apps.inside.anl.gov/argoapi/docs
- ALCF Inference Endpoints: https://inference.alcf.anl.gov/
- Prompt templates: [`lassi/prompt_dictionary.py`](lassi/prompt_dictionary.py)
  (read before editing — wording is empirically tuned)
