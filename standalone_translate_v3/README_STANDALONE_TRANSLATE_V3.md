# Standalone LASSI v3 CUDA→SYCL translation runner (Argo-only)

This folder is a **self-contained** subset of the original `LASSI-EE/` tree, intended to run the translation pipeline end-to-end:

- Stage 0: build+run CUDA XSBench (via its Makefile)
- Stage 1+: translate CUDA→SYCL using **Argo** LLM calls, with compile/run self-correction
- **Stops immediately** once a SYCL translation compiles and executes successfully

## What’s included

- Entrypoint: [`run_lassi_translate_v3.py`](run_lassi_translate_v3.py:1)
- Pipeline: [`lassi.pipeline_v3.auto_code_llmgeneration_pipeline()`](lassi/pipeline_v3.py:24)
- Argo backend: [`lassi.llm_config.argo_llm()`](lassi/llm_config.py:18)
- CUDA XSBench source: `source_codes/Miniapps/XSBench/cuda/`

## Requirements

System tools:

- CUDA Toolkit (`nvcc` available; the runner prepends `/usr/local/cuda/bin` to `PATH`)
- A SYCL compiler (AdaptiveCpp expected at `/home/vchirio/adaptivecpp/bin/acpp` or on `PATH`)

Python:

- Install dependencies from [`requirements.txt`](requirements.txt:1)

Argo access:

- Must be on the ANL network (same assumption as the original code).
- If your gateway requires auth, set `ARGO_API_KEY` and it will be sent as a Bearer token.

## Run

From this directory:

```bash
python3 -u run_lassi_translate_v3.py --log-level info --llm-source argo --no-display-response
```

Artifacts are written under `generated_codes/experiment/...`.
