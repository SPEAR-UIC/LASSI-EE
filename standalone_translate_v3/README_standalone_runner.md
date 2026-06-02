# LASSI v3 standalone translation runner

This repo includes a standalone Python entrypoint that refactors the notebook flow in [`LASSI-EE/LASSI_power_pipeline_v3.ipynb`](LASSI-EE/LASSI_power_pipeline_v3.ipynb:1) into a runnable script.

## Entry point

- [`LASSI-EE/run_lassi_translate_v3.py`](LASSI-EE/run_lassi_translate_v3.py)

## What it does

1. **Device inspection**
   - Uses [`lassi.gpu_detector.detect_gpu_vendors()`](LASSI-EE/lassi/gpu_detector.py) and best-effort `nvidia-smi`/`rocm-smi` calls.
2. **Stage 0: build + run source app (CUDA)**
   - Builds XSBench CUDA via its Makefile and runs it with the CLI args from `source_apps.json`.
   - Captures and preserves stdout for later comparison.
3. **Stage 1+: translate CUDA → SYCL with self-correction loops**
   - Uses the v3 pipeline [`lassi.pipeline_v3.auto_code_llmgeneration_pipeline()`](LASSI-EE/lassi/pipeline_v3.py) to:
     - collect prompt dictionary context,
     - prompt an LLM to generate SYCL,
     - compile generated code,
     - if compile fails: apply the pipeline’s self-correction prompting protocol and retry,
     - run generated code,
     - if runtime fails: apply the pipeline’s self-correction prompting protocol, regenerate, recompile, and retry.
4. **Compare stdout**
   - Extracts the pipeline’s recorded source and generated stdout from the experiment metadata file and writes:
     - `source_stdout.txt`
     - `sycl_stdout.txt`
     - `stdout_diff.txt` (unified diff)

## LLM call logging

Every LLM call is logged to disk and summarized to stdout.

- Logger: [`lassi.llm_call_logger.LLMCallLogger`](LASSI-EE/lassi/llm_call_logger.py)
- Hooked into: [`lassi.llm_methods.llm_call()`](LASSI-EE/lassi/llm_methods.py) and [`lassi.llm_methods.code_generator_llm()`](LASSI-EE/lassi/llm_methods.py)

Artifacts are written under the experiment directory:

- `llm_calls/call_0001/meta.json`
- `llm_calls/call_0001/system.txt`
- `llm_calls/call_0001/user.txt`
- `llm_calls/call_0001/response.txt`

`meta.json` includes truncated previews for readability; the full prompts/responses are preserved in the `.txt` files.

## Usage

From the repo root:

```bash
cd /home/vchirio/Desktop/LASSI-EE
python3 ./run_lassi_translate_v3.py \
  --source-app XSBench \
  --source-lang CUDA \
  --target-lang SYCL \
  --llm-source argo \
  --no-display-response
```

Common flags:

- `--llm-source {auto,argo,ollama,cerebras,lmstudio}`
- `--model`, `--model-coder`, `--model-judge`
- `--num-ctx`, `--num-ctx-coder`
- `--truncate-chars` (LLM context preview truncation)

## Notes / environment

- CUDA build expects `nvcc` on `PATH`.
- SYCL build uses the toolchain returned by [`lassi.sycl_toolchain.get_sycl_toolchain()`](LASSI-EE/lassi/sycl_toolchain.py) (AdaptiveCpp by default).
- Several notebook-only Python dependencies were made optional so the runner can execute in a minimal environment:
  - [`lassi.llm_config`](LASSI-EE/lassi/llm_config.py) now guards optional imports (`ollama`, `cerebras`, `httpx`, `aiohttp`, etc.).
  - [`lassi.argostream_curl`](LASSI-EE/lassi/argostream_curl.py) no longer requires `IPython`.
  - [`lassi.token_analysis`](LASSI-EE/lassi/token_analysis.py) no longer requires `tiktoken`.
  - [`lassi.get_power.report_gpu_memory()`](LASSI-EE/lassi/get_power.py) becomes a no-op if `pynvml` is not installed.

## Exit codes

- `0`: script ran; stdout comparison step completed (diff file written). If the pipeline fails early (e.g., Stage 0 runtime error), the comparison may still be “MATCH” because both outputs are empty.
- `2`: script ran and both outputs were captured, but stdout differs.
