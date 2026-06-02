#!/usr/bin/env python3

"""Standalone runner for the LASSI v3 CUDA→SYCL translation pipeline.

This script is a refactor of the notebook flow in
[`LASSI-EE/LASSI_power_pipeline_v3.ipynb`](LASSI-EE/LASSI_power_pipeline_v3.ipynb:1).

It:
1) Inspects and reports available devices.
2) Builds + runs the original source app (CUDA XSBench) and preserves stdout.
3) Runs the LASSI translation pipeline (CUDA→SYCL) with compile/execute self-correction.
4) Captures generated stdout and compares it to the preserved source stdout.
5) Logs each LLM call context (truncated) and saves full prompts/responses to disk.

Notes:
- Stage 0 (source CUDA) uses the app's Makefile (nvcc under the hood).
- Stage 1+ (generated SYCL) uses a Makefile written into the experiment directory by the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

import lassi.pipeline_v3
from lassi.gpu_detector import detect_gpu_vendors
from lassi.llm_call_logger import LLMCallLogger
from lassi.prompt_dictionary import PromptDictionary
from lassi.sycl_toolchain import get_sycl_toolchain


@dataclass(frozen=True)
class RunnerConfig:
    repo_root: Path
    source_apps_json: Path
    source_collection: str
    source_app: str
    source_lang: str
    target_lang: str

    model: str
    model_coder: str
    model_judge: str
    llm_source: str
    num_ctx: int
    num_ctx_coder: int

    display_response: bool
    verifyprompt: bool
    truncate_chars: int


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [RUN] {msg}")


def _init_data_collection(dir_path: Path, fieldnames: List[str], filename: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    fp = dir_path / filename
    if not fp.exists():
        with fp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    return fp


def _infer_llm_source_from_model(model: str) -> str:
    if model in [
        "gpt52",
        "gpt5",
        "gpt35",
        "gpt35large",
        "gpt4",
        "gpt4large",
        "gpt4turbo",
        "gpt4o",
        "gpt4olatest",
        "gpt41",
        "gpto1preview",
        "gpto1mini",
        "gpto3mini",
        "gpto4mini",
        "claudesonnet4",
    ]:
        return "argo"
    if model in ["gpt-oss-120b", "qwen-3-coder-480b"]:
        return "cerebras"
    if "GGUF" in model and "hf.co" not in model:
        return "lmstudio"
    return "ollama"


def _device_report() -> Tuple[str, int]:
    vendors = detect_gpu_vendors()
    if not vendors:
        raise SystemExit("Pipeline stopped - no GPU hardware found")

    detected_vendor = vendors[0]["vendor"].upper()
    gpu_count = vendors[0]["count"]
    _log(f"Detected GPU vendor={detected_vendor} count={gpu_count}")

    # Extra info (best-effort)
    if detected_vendor == "NVIDIA" and shutil.which("nvidia-smi"):
        _log("nvidia-smi summary:")
        os.system("nvidia-smi -L")
    if detected_vendor == "AMD" and shutil.which("rocm-smi"):
        _log("rocm-smi summary:")
        os.system("rocm-smi --showid")

    return detected_vendor, gpu_count


def _load_source_app_config(cfg: RunnerConfig) -> dict:
    with cfg.source_apps_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_source_code(source_apps: dict, source_collection: str, source_app: str) -> Tuple[str, Path, str, str, int, List[str]]:
    source_folder = source_apps[source_collection]["source_folder"]
    app_folder_name = source_apps[source_collection][source_app]["app_folder"]
    lines_of_code = source_apps[source_collection][source_app]["code_lines"]
    exe_input_parameters = source_apps[source_collection][source_app]["app_params"]

    file_name = "Main"
    input_code_extension = "cu"

    source_file_directory = Path("source_codes") / source_folder / app_folder_name
    source_file_path = (source_file_directory / f"{file_name}.{input_code_extension}").resolve()

    with source_file_path.open("r", encoding="utf-8") as f:
        input_code = f.read()

    return input_code, source_file_path, source_folder, app_folder_name, lines_of_code, exe_input_parameters


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _diff_text(a: str, b: str, a_name: str, b_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=a_name,
            tofile=b_name,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--source-apps-json", default="source_apps.json")
    parser.add_argument("--source-collection", default="Miniapps")
    parser.add_argument("--source-app", default="XSBench")
    parser.add_argument("--source-lang", default="CUDA")
    parser.add_argument("--target-lang", default="SYCL")

    parser.add_argument("--model", default="gpt52")
    parser.add_argument("--model-coder", default="gpt52")
    parser.add_argument("--model-judge", default="gpt52")
    parser.add_argument("--llm-source", default="argo", help="LLM backend (argo/ollama/cerebras/lmstudio/auto)")
    parser.add_argument("--num-ctx", type=int, default=200000)
    parser.add_argument("--num-ctx-coder", type=int, default=200000)

    parser.add_argument("--display-response", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verifyprompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--truncate-chars", type=int, default=2000)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["quiet", "info", "debug"],
        help="Runner verbosity (pipeline has its own prints).",
    )

    args = parser.parse_args()

    cfg = RunnerConfig(
        repo_root=Path(args.repo_root).resolve(),
        source_apps_json=Path(args.source_apps_json),
        source_collection=args.source_collection,
        source_app=args.source_app,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        model=args.model,
        model_coder=args.model_coder,
        model_judge=args.model_judge,
        llm_source=args.llm_source,
        num_ctx=args.num_ctx,
        num_ctx_coder=args.num_ctx_coder,
        display_response=args.display_response,
        verifyprompt=args.verifyprompt,
        truncate_chars=args.truncate_chars,
    )

    log_level = args.log_level

    os.chdir(cfg.repo_root)
    _log(f"CWD={Path.cwd()}")
    _log(
        "Mode: translation-only (energy-efficiency optimization disabled: "
        "power_measure=False, yes_power_feedback=False, baseline_oneshot_goal=True, "
        "use_self_prompting=False, vary_llm_temp=False)"
    )

    # Ensure CUDA Toolkit nvcc is on PATH for Stage 0.
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    _log(f"nvcc on PATH: {shutil.which('nvcc')}")

    # Device inspection
    detected_vendor, _gpu_count = _device_report()

    # Toolchain
    sycl_tc = get_sycl_toolchain()
    _log(f"SYCL toolchain compiler={sycl_tc.compiler} cxxflags={sycl_tc.cxxflags} ldflags={sycl_tc.ldflags}")

    # Load app config + source code
    source_apps = _load_source_app_config(cfg)
    input_code, source_file_path, source_folder, app_folder_name, lines_of_code, exe_input_parameters = _load_source_code(
        source_apps, cfg.source_collection, cfg.source_app
    )
    _log(f"Loaded source: {source_file_path}")
    _log(f"Source LOC: {len(input_code.splitlines())}")

    prompts = PromptDictionary()

    # Prompts: translation only
    this_llm_system_prompt = prompts.get_system_prompt("CUDA_to_SYCL")
    this_llm_translate_prompt = prompts.get_codetranslate_prompt("CUDA_to_SYCL")

    # Experiment naming
    experiment = f"exp_xsbench_translate_cuda_to_sycl_{cfg.model_coder}_{detected_vendor}"
    now = datetime.now(ZoneInfo("America/Chicago"))
    dt_string = now.strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_data_directory = (Path("generated_codes") / "experiment" / f"{experiment}__{dt_string}").resolve()

    # LLM call logging
    llm_calls_dir = experiment_data_directory / "llm_calls"
    llm_logger = LLMCallLogger(str(llm_calls_dir), truncate_chars=cfg.truncate_chars)
    import lassi.llm_methods as llm_methods

    llm_methods.LLM_CALL_LOGGER = llm_logger

    # Data collection CSV (minimal fields)
    DATAFIELDNAMES = [
        "trial_number",
        "timestamp",
        "experiment",
        "gpu_vendor",
        "app_name",
        "code_line_count",
        "llm",
        "llm_coder",
        "llm_judge",
        "source_stdout",
        "sycl_compile_ok",
        "sycl_run_ok",
        "sycl_stdout",
        "pipeline_runtime_min",
    ]

    experiment_data_file_object = _init_data_collection(
        experiment_data_directory, DATAFIELDNAMES, filename="lassi-translate_pipeline_data.csv"
    )

    experiment_data = dict.fromkeys(DATAFIELDNAMES, None)
    experiment_data["gpu_vendor"] = detected_vendor
    experiment_data["experiment"] = experiment
    experiment_data["app_name"] = cfg.source_app
    experiment_data["code_line_count"] = lines_of_code
    experiment_data["llm"] = cfg.model
    experiment_data["llm_coder"] = cfg.model_coder
    experiment_data["llm_judge"] = cfg.model_judge

    meta_data_file = "***** EXPERIMENT CONFIGURATION SUMMARY *****\n"
    meta_data_file += f"\nGPU Vendor: {detected_vendor}"
    meta_data_file += f"\nDate-time: {dt_string}"
    meta_data_file += f"\nApplication: {app_folder_name}"
    meta_data_file += f"\n\nLLM Source: {cfg.llm_source}"
    meta_data_file += f"\nModel: {cfg.model}"
    meta_data_file += f"\nModel coder: {cfg.model_coder}"
    meta_data_file += f"\nModel judge: {cfg.model_judge}"
    meta_data_file += "\n\nPower measurement: False\n"

    _log("Experiment directory: " + str(experiment_data_directory))
    _log("Artifacts will be written under this directory (metadata, llm_calls, stdout files, diff)")
    _write_text(experiment_data_directory / "metadata_summary.txt", meta_data_file)

    # Output directories (generated code)
    output_code_directory = f"generated_codes/{source_folder}/{app_folder_name}/sycl"
    output_code_file_name = "Main"

    # Stage 0 + pipeline run
    trial_number = 1
    all_trials_start = time.time()

    # Stage 0 uses make/nvcc; Stage 1+ uses SYCL toolchain.
    cuda_code_compiler = "make"
    cuda_code_compiler_kwds = "-j SM_VERSION=80 CFLAGS=-std=c++17"

    sycl_code_compiler = sycl_tc.compiler
    sycl_code_compiler_kwds = f"{sycl_tc.cxxflags} {sycl_tc.ldflags}".strip()

    # No extra context for this minimal run
    input_context = ""

    # Pipeline flags (translation-only)
    use_context = False
    use_context_summaries_only = True
    few_shot_learning = False
    run_code_analysis = False

    build_script = False
    make_script = True
    code_build_script = ""

    yes_execute = True
    power_measure = False
    test_code_first = True
    yes_power_feedback = False
    baseline_oneshot_goal = True
    use_self_prompting = False
    vary_llm_temp = False

    test_with_error = False
    generate_test_case = False

    verifyprompt = cfg.verifyprompt

    # If user didn't override llm_source, infer from model (but default is argo per request).
    if cfg.llm_source == "auto":
        llm_source = _infer_llm_source_from_model(cfg.model)
    else:
        llm_source = cfg.llm_source

    _log(f"CUDA build: {cuda_code_compiler} {cuda_code_compiler_kwds}")
    _log(f"SYCL build: {sycl_code_compiler} {sycl_code_compiler_kwds}")

    _log("Starting pipeline (Stage 0: build+run source; Stage 1+: translate+compile+run with self-correction)")
    complete = lassi.pipeline_v3.auto_code_llmgeneration_pipeline(
        prompts,
        this_llm_system_prompt,
        this_llm_translate_prompt,
        llm_source,
        cfg.model,
        cfg.model_coder,
        cfg.model_judge,
        cfg.num_ctx,
        cfg.num_ctx_coder,
        use_context,
        use_context_summaries_only,
        few_shot_learning,
        run_code_analysis,
        input_context,
        "",
        input_code,
        lines_of_code,
        cfg.target_lang,
        "cpp",
        cfg.source_lang,
        # Generated code compilation toolchain
        sycl_code_compiler,
        sycl_code_compiler_kwds,
        build_script,
        True,  # make_script for generated SYCL
        code_build_script,
        output_code_directory,
        output_code_file_name,
        str(source_file_path),
        cfg.display_response,
        test_with_error,
        generate_test_case,
        yes_execute,
        exe_input_parameters,
        meta_data_file,
        power_measure,
        test_code_first,
        yes_power_feedback,
        dt_string,
        baseline_oneshot_goal,
        use_self_prompting,
        vary_llm_temp,
        experiment,
        verifyprompt,
        trial_number,
        str(experiment_data_directory),
        str(experiment_data_file_object),
        experiment_data,
        cfg.source_app,
    )

    runtime_min = round((time.time() - all_trials_start) / 60, 2)
    _log(f"Pipeline completed={complete} runtime_min={runtime_min}")

    # Key locations
    _log(f"Source code path: {source_file_path}")
    _log(f"Generated SYCL code path (pipeline output dir): {Path(output_code_directory) / (output_code_file_name + '.cpp')}")
    _log(f"Experiment dir (build logs, Makefile, llm_calls): {experiment_data_directory}")

    # Collect stdout artifacts from pipeline outputs.
    # The v3 pipeline stores stdout in the llm-judge dict keys:
    # - input_content_llmjudge_dict["source_stdout"]
    # - input_content_llmjudge_dict["refactored_stdout"]
    # but it does not currently copy those into experiment_data for translation-only runs.
    # Therefore, parse the metadata file written by the pipeline.

    meta_files = sorted(experiment_data_directory.glob("metadata_*_Trial*.txt"))
    meta_text = meta_files[-1].read_text(encoding="utf-8") if meta_files else ""

    def _extract_block(tag: str) -> str:
        if not meta_text:
            return ""
        start = meta_text.find(tag)
        if start < 0:
            return ""
        start += len(tag)
        # Next section delimiter
        next_delim = meta_text.find("----- ", start)
        if next_delim < 0:
            return meta_text[start:].strip() + "\n"
        return meta_text[start:next_delim].strip() + "\n"

    source_stdout = _extract_block("----- SOURCE: EXECUTION OUTPUT -----")
    sycl_stdout = _extract_block("----- GENERATED: EXECUTION OUTPUT -----")

    _write_text(experiment_data_directory / "source_stdout.txt", source_stdout)
    _write_text(experiment_data_directory / "sycl_stdout.txt", sycl_stdout)

    diff = _diff_text(source_stdout, sycl_stdout, "source_stdout", "sycl_stdout")
    _write_text(experiment_data_directory / "stdout_diff.txt", diff)

    # Console summary + explicit artifact pointers
    _log(f"Saved source stdout: {experiment_data_directory / 'source_stdout.txt'}")
    _log(f"Saved generated stdout: {experiment_data_directory / 'sycl_stdout.txt'}")
    _log(f"Saved stdout diff: {experiment_data_directory / 'stdout_diff.txt'}")

    if diff.strip():
        _log("STDOUT comparison: DIFFERENT")
        return 2

    _log("STDOUT comparison: MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
