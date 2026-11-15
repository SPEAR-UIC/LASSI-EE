# LASSI-EE

<img src="lassi/lassi-ee-logo.png" alt="LASSI-EE Logo" width="300"/>

## What is LASSI-EE?

LASSI-EE (LLM-based Automated Self-correcting pipeline for generating parallel ScIentific codes for Energy Efficiency) is an automated framework that refactors parallel scientific GPU codes to consume less energy. It closes the loop between LLM code generation and empirical measurement: every candidate is compiled, executed and power-profiled on the target GPU, and those measurements, together with compiler and runtime errors, are fed back into the next generation attempt.
Energy-aware prompting, iterative self-correction and an LLM-as-a-Judge screening step turn a single refactoring request into a search that keeps the lowest-energy version that still behaves like the source.

The paper evaluates LASSI-EE on 22 applications on NVIDIA A100 and AMD MI100 GPUs, with 30 independent trials per application and device, and introduces **energy-reduction@*k***, the expected energy reduction when generating *k* candidates and keeping the most efficient one that passes screening.

## Results at a Glance

Across the 22 applications and both GPUs (1,320 trials, all compiled at `-O3`):

- **Expected energy reduction:** 29.1% at *k* = 1, 48.2% at *k* = 3 and 54.6% at *k* = 5, averaged over the A100 and MI100. At *k* = 1 this is 2.8× the unassisted ("vanilla") LLM. These figures count every attempt, including failures; the refactorings that pass screening average 34% to 36% energy reduction.
- **Runtime is preserved:** 93.2% of application-device pairs reduce both energy and runtime, and the other 6.8% reduce energy through lower power alone. Mean runtime reductions are 26.2% on the A100 and 29.6% on the MI100.
- **No precision loss:** none of the roughly 3,500 optimizations reduced numerical precision.
- **Hardware-specific code:** only 3.5% of optimizations are identical across the AMD and NVIDIA runs.

See the paper cited below for the full methodology and per-application results.

## How It Works

LASSI-EE proceeds through five stages for each application:

0. **Code profiling.** The source code is compiled, executed and power-profiled on the target GPU to establish the baseline for runtime, average power and net energy.
1. **Baseline refactoring ("vanilla LLM").** A zero-shot prompt asks the LLM to refactor the code for energy efficiency, with no context and no self-correction. The result is compiled, executed once, power-profiled and screened by the Judge. This is the unassisted baseline that the later stages are compared against.
2. **Context building and planning.** The LLM summarizes background knowledge (CUDA or HIP programming-guide excerpts, in `lassi/context/`), analyzes the code for inefficiencies, and writes a structured refactoring plan.
3. **Iterative self-prompted refactoring with self-correction.** The LLM generates a candidate from the plan. Compiler and runtime errors are fed back until the code runs cleanly (the self-correction loop); the candidate is then power-profiled, screened by the Judge, and compared with the current best version. An improvement triggers a new plan from the latest code. Consecutive non-improving iterations are tracked by a counter that starts at 0.2, rises by 0.2 as unsuccessful iterations accumulate, and resets when a new best version is found; Stage 3 ends when it reaches 1.0, or at the configured iteration caps. For models that accept a sampling temperature, the same value is used as the temperature.
4. **Termination.** The best version is saved with all intermediate versions, power profiles, Judge verdicts, plans and iteration counts, plus an LLM-written comparison of the source and the best refactoring. If nothing passed screening and reduced energy, the source is kept and no improvement is reported.

The **LLM-as-a-Judge** compares the source and refactored code together with their standard outputs and energy metrics, and returns `VALID` or `INVALID`. It is a practical screen for large-scale evaluation, not a substitute for formal verification. All prompt templates are in `lassi/prompt_dictionary.py`.

**Power measurement.** Power is sampled every 0.01 s from NVML on NVIDIA GPUs and from `rocm-smi --showpower` on AMD GPUs. Idle power is estimated from readings taken before and after each execution (with 15 s of post-run capture) and subtracted from every sample to give net energy; negative net samples (measurement noise) are clamped to zero. Vendor counters update coarsely — on the A100, NVML repeats each reading across about ten 10 ms samples — so the paper's applications all run for at least a second, and results are averaged over 30 trials. Very short runs cannot be resolved.

## Repository Layout

- `LASSI_power_pipeline_v3.ipynb` — the entry point: configure `experimental_setup()` and run the pipeline
- `lassi/` — core package: `pipeline_v3.py` (orchestration), `pipeline_methods.py`, `llm_methods.py` and `llm_config.py` (LLM calls and backends), `code_manager.py` (compile and execute), `get_power.py` and `amd_power.py` (NVML and rocm-smi power measurement), `gpu_detector.py`, `prompt_dictionary.py` (prompt templates), `context/` (background-knowledge documents)
- `source_codes/HeCBench/` — the benchmark kernels (see `source_apps.json` for per-kernel input parameters)
- `generated_codes/` — run outputs (gitignored)
- `test_amd_integration.py`, `hip_code_test.py`, `gpu_state_checker.py` — standalone AMD/HIP and GPU-state helper scripts

## Benchmark Applications

The 20 HeCBench kernels used in the reported experiment (exec params are the positional arguments passed to each benchmark binary):

| Category | Application | Lines of Code | Exec Params | Local Header |
|---|---|---|---|---|
| Bandwidth | randomAccess | 158 | `10` | — |
| Bioinformatics | all-pairs-distance | 328 | `10000` | — |
| Computer vision and image processing | colorwheel | 154 | `10000 256 500000` | — |
| | marchingCubes | 574 | `100` | `tables.h` |
| Cryptography | chacha20 | 130 | `300000` | `chacha20.h` |
| Data compression and reduction | segment-reduce | 95 | `16384 100` | — |
| Data encoding, decoding, or verification | entropy | 171 | `10000 1024 1` | `reference.h` |
| | murmurhash3 | 245 | `750000 500` | — |
| Graph and tree | floydwarshall | 295 | `2048 100 256` | — |
| Language and kernel features | layout | 197 | `2500` | — |
| | threadfence | 142 | `100 250000000` | — |
| Machine learning | dense-embedding § | 193 | `100000 128 100` | — |
| Math | jacobi † | 235 | (none) | — |
| | jaccard | 417 | `1024 512 1000` | — |
| | stu-matrix-rotate ‡ | 67 | `30000 100` | — |
| Search | bsearch | 279 | `8388608 1000` | — |
| | keogh | 143 | `256 22500000 100` | `reference.h` |
| Signal processing | extrema | 349 | `750` | `kernels.h`, `reference.h` |
| Simulation | pathfinder | 286 | `100000 10000 2500` | — |
| | lid-driven-cavity | 1079 | (none) | — |

† `jacobi` takes no runtime arguments; its compiled-in size constant `N` was raised to 16384 in the source (see the paper).

‡ This is the paper's `matrix-rotate`. The application has parallel and serial components, and the serial portion was removed (hence 67 lines). `source_apps.json` also keeps the original, longer `matrix-rotate` and `atomicCost`, which are not part of the reported set.

§ **Erratum to the paper:** Table I prints this application's first argument as `10000`. That is a typographical error — with `10000` the program aborts on its own assertion (`nrows > batch_size * batch_size`). Use `100000`, as in `source_apps.json`.

Kernels with a local header compile correctly via the `-I` flag described in [Adding a New Kernel](#adding-a-new-kernel). The header's contents are not part of the LLM's refactoring prompt — only the `.cu` file's own text is sent.

The two miniApps, XSBench (1,472 lines; `-m event -l 1700000000`) and miniMDock (2,394 lines; `-lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 250`), are built from merged source files through their own make scripts.

> **Not yet included:** the merged miniApp sources (XSBench, miniMDock) are not in this repository yet, so those two applications cannot be run from a fresh clone. The 20 HeCBench applications above are fully included.

## Getting Started

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (this includes JupyterLab)
pip install -r requirements.txt

# Launch Jupyter and open LASSI_power_pipeline_v3.ipynb
jupyter lab
```

Requirements: Python 3.12+ on Linux, and a GPU that exposes power telemetry.

- **NVIDIA:** a CUDA toolkit providing `nvcc` that is compatible with the installed driver. `nvidia-ml-py` (installed by `requirements.txt`) provides the `pynvml` module used for power measurement; do not also install the deprecated standalone `pynvml` package, which shares the module name.
- **AMD:** ROCm providing `hipcc` and `rocm-smi`.

The reported experiments used NVIDIA A100 (80 GB, driver 560.35.05, CUDA 12.6) and AMD MI100 (32 GB, ROCm 6.4.1) GPUs on the Chameleon testbed. Other CUDA and ROCm versions should work, but only these were tested for the paper.

> **Power measurement needs a dedicated GPU with stable telemetry.** On a laptop GPU that also drives a desktop session and uses runtime power management, NVML can stall for seconds and return spurious readings, which corrupts the idle-power estimate and therefore every energy value. Use such a system for functional checks only.

### LLM Backend Setup

The `llm_source` setting in the notebook's `experimental_setup()` selects the LLM backend. The models for the three pipeline roles are set by `model` (context building and other inference), `model_coder` (code generation) and `model_judge` (LLM-as-a-Judge).

| Backend | Who can use it | Required setup |
|---|---|---|
| `openai_compatible` | Anyone — OpenAI itself, an institutional LLM gateway that exposes an OpenAI-compatible endpoint, or any self-hosted server implementing the same REST contract (vLLM, llama.cpp server, LM Studio, OpenRouter, etc.) | The base URL in `experimental_setup()` (default `https://api.openai.com/v1`) and an API key, from the `OPENAI_API_KEY` environment variable or `lassi.llm_config.openai_api_key` |
| `ollama` | Anyone running a local [Ollama](https://ollama.com) server | `ollama serve`, and `ollama pull <model>` for each model used |
| `argo` | Argonne National Laboratory personnel only — a legacy path to the Argo Gateway through a localhost tunnel | Your Argonne domain username in `experimental_setup()` |

LASSI-EE sends `temperature` and `top_p=0.9` with each request. If a model rejects one of them, as OpenAI's reasoning models do, that parameter is dropped automatically for the model. The temperature also acts as the pipeline's iteration counter: it rises by 0.2 when refactoring stalls and the run stops once it reaches 1.0, so the stopping rule still works when the model ignores the value.

### Running the Pipeline

In `experimental_setup()`, set the application (`source_collection`, `source_app`), the backend and models, and `total_trials`, then run the notebook cells top to bottom. `jacobi`, the notebook's default, is a good first application.

Each run writes to `generated_codes/experiment/<label>__<timestamp>/`: every generated code version, a power plot and power CSV for each version, the per-trial results (`lassi-ee_pipeline_data.csv`), and a metadata summary. A trial typically takes tens of minutes, depending on how many iterations the LLM needs.

## Reproducing the Paper's Setup

- **Models:** `o4-mini` for context building and code generation on the HeCBench applications, `gpt-5` for code generation on the two miniApps, and `gpt-4.1` as the Judge. The notebook defaults to these OpenAI model IDs.
- **Sampling temperature:** neither o4-mini nor GPT-5 accepts one, so in all reported experiments the 0.2-to-1.0 value served only as the stopping counter; LASSI-EE drops the parameters these models reject automatically.
- **Trials:** 30 independent trials per application and device. The notebook defaults to `total_trials = 1` to avoid an accidental long run; set it to 30 to match the paper.
- **Compile flags:** on the A100, `nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3`; on the MI100, `hipcc -std=c++14 -Wall -O3 --amdgpu-target=gfx908`. The notebook detects the NVIDIA architecture flag automatically (`sm_80` on an A100).
- **Input parameters:** use the values in `source_apps.json` unchanged.
- **Non-determinism:** because LLM inference is non-deterministic, a single run does not reproduce a table row. The paper's tables are averages over 30 trials per application and device.

> **Not yet included:** the per-trial records and the analysis script that computes the paper's pass@*k* and energy-reduction@*k* values are not in this repository yet.

## Adding a New Kernel

No code changes are required to add a kernel from the HeCBench collection that is not listed above. Register it in a new `source_apps.json` entry.

1. **Layout.** HeCBench nests kernels as `src/<kernel>-cuda/main.cu`. This pipeline requires a flat `source_codes/HeCBench/<kernel>/<kernel>-cuda_main.cu` (and `<kernel>-hip_main.cu` for HIP). Rename the main file accordingly.
2. **Local headers.** Place any `#include`d header in the same kernel folder, referenced by filename only (no path). `code_manager.py` passes an `-I` flag for the kernel's source folder on every compile, so the include resolves regardless of where a generated copy is compiled.
3. **`source_apps.json` entry**, under `"HeCBench"`:
   - `app_folder` — kernel folder name
   - `app_params` — positional CLI arguments (see the kernel's `argv` parsing or README)
   - `code_lines` — informational
4. Select it in the notebook with `source_collection` and `source_app`.

## Related Projects

LASSI-EE is part of a suite of tools from the SPEAR research group exploring LLM-based automated refactoring of parallel scientific GPU/HPC code:

- **LASSI** — the original self-correcting pipeline for translating parallel scientific code between languages (e.g., CUDA to OpenMP) via an iterative LLM self-correction loop. LASSI-EE builds directly on this design.
- **LASSI-EE** (this repository) — re-targets the same self-correction loop at energy-efficiency refactoring, using runtime GPU power profiling and an LLM-as-a-Judge screening step.
- **LASSI-Verify** — replaces human review / LLM-judge screening with a three-tier empirical verification framework plus a witness chain for cheat-resistant loop-deletion detection.

See the Citation section below for references to all three.

## Citation

If you use LASSI-EE in your research, please cite:

> M. T. Dearing, Y. Tao, X. Wu, Z. Lan, and V. Taylor, "LASSI-EE: LLM-based automated refactoring of parallel scientific codes for energy efficiency," in *Proceedings of the 3rd International Workshop on Energy Efficiency with Sustainable Performance (EESP '26), held with SC26: The International Conference for High Performance Computing, Networking, Storage and Analysis*, Nov. 2026 (to appear).

### Related Publications

> M. T. Dearing, Y. Tao, X. Wu, Z. Lan, and V. Taylor, "LASSI: An LLM-based Automated Self-Correcting Pipeline for Translating Parallel Scientific Codes," in *International Workshop on Large Language Models and HPC (LLMxHPC), IEEE International Conference on Cluster Computing (CLUSTER)*, 2024. DOI: [10.1109/CLUSTERWorkshops61563.2024.00029](https://doi.org/10.1109/CLUSTERWorkshops61563.2024.00029)

> M. T. Dearing, X. Wu, V. Taylor, and Z. Lan, "LASSI-Verify: Adaptive Empirical Verification of LLM-Generated Parallel Scientific Code Transformations," in *SC26 Workshops (Correctness 2026)*, Chicago, IL, USA, Nov. 2026.

## License

Licensed under the Apache License 2.0; see [LICENSE](LICENSE). Vendored third-party kernels in `source_codes/` retain their original licenses.
