#!/usr/bin/env python3

"""Standalone translation runner.

Workflow:
1) Report available GPU devices.
2) Build+run the selected source app (CUDA) and preserve stdout.
3) Translate to target language (SYCL) via Argo LLM.
4) Compile+run generated code with a self-correction loop on errors.

Argo is the only supported LLM backend.
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
    print(f"[{ts}] [RUN] {msg}", flush=True)


def _init_csv(dir_path: Path, fieldnames: List[str], filename: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    fp = dir_path / filename
    if not fp.exists():
        with fp.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return fp


def _device_report() -> Tuple[str, int]:
    vendors = detect_gpu_vendors()
    if not vendors:
        raise SystemExit("Pipeline stopped - no GPU hardware found")

    vendor = vendors[0]["vendor"].upper()
    count = vendors[0]["count"]
    _log(f"Detected GPU vendor={vendor} count={count}")

    # Best-effort extra info.
    if vendor == "NVIDIA" and shutil.which("nvidia-smi"):
        _log("nvidia-smi summary:")
        os.system("nvidia-smi -L")
    if vendor == "AMD" and shutil.which("rocm-smi"):
        _log("rocm-smi summary:")
        os.system("rocm-smi --showid")

    return vendor, count


def _load_source_app_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_source_code(source_apps: dict, source_collection: str, source_app: str) -> Tuple[str, Path, str, str, int, List[str]]:
    source_folder = source_apps[source_collection]["source_folder"]
    app_folder_name = source_apps[source_collection][source_app]["app_folder"]
    lines_of_code = source_apps[source_collection][source_app]["code_lines"]
    exe_input_parameters = source_apps[source_collection][source_app]["app_params"]

    file_name = "Main"
    input_code_extension = "cu"

    source_dir = Path("source_codes") / source_folder / app_folder_name
    source_path = (source_dir / f"{file_name}.{input_code_extension}").resolve()

    return (
        source_path.read_text(encoding="utf-8"),
        source_path,
        source_folder,
        app_folder_name,
        lines_of_code,
        exe_input_parameters,
    )


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
    # Ensure we see logs even when stdout is buffered (e.g., when run under some terminals).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    p = argparse.ArgumentParser()

    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--source-apps-json", default="source_apps.json")
    p.add_argument("--source-collection", default="Miniapps")
    p.add_argument("--source-app", default="XSBench")
    p.add_argument("--source-lang", default="CUDA")
    p.add_argument("--target-lang", default="SYCL")

    p.add_argument("--model", default="gpt52")
    p.add_argument("--model-coder", default="gpt52")
    p.add_argument("--model-judge", default="gpt52")
    p.add_argument("--llm-source", default="argo")
    p.add_argument("--num-ctx", type=int, default=200000)
    p.add_argument("--num-ctx-coder", type=int, default=200000)

    # Never print full generated code to console; full responses are preserved under experiment_dir/llm_calls/.
    p.add_argument("--display-response", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--verifyprompt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--truncate-chars", type=int, default=2000)

    args = p.parse_args()

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

    if cfg.llm_source != "argo":
        raise SystemExit("This standalone translation runner supports --llm-source=argo only")

    os.chdir(cfg.repo_root)
    _log(f"CWD={Path.cwd()}")
    _log("Mode: device report → build+run source → translate → compile+run with self-correction")

    # Ensure CUDA Toolkit nvcc is on PATH for Stage 0.
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    _log(f"nvcc on PATH: {shutil.which('nvcc')}")

    detected_vendor, _gpu_count = _device_report()

    sycl_tc = get_sycl_toolchain()
    _log(f"SYCL toolchain compiler={sycl_tc.compiler} cxxflags={sycl_tc.cxxflags} ldflags={sycl_tc.ldflags}")

    source_apps = _load_source_app_config(cfg.source_apps_json)
    input_code, source_file_path, source_folder, app_folder_name, lines_of_code, exe_input_parameters = _load_source_code(
        source_apps, cfg.source_collection, cfg.source_app
    )
    _log(f"Loaded source: {source_file_path}")

    prompts = PromptDictionary()
    system_prompt = prompts.get_system_prompt("CUDA_to_SYCL")
    translate_prompt = prompts.get_codetranslate_prompt("CUDA_to_SYCL")

    experiment = f"exp_xsbench_translate_cuda_to_sycl_{cfg.model_coder}_{detected_vendor}"
    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_dir = (Path("generated_codes") / "experiment" / f"{experiment}__{dt_string}").resolve()

    # LLM call logging.
    llm_calls_dir = experiment_dir / "llm_calls"
    import lassi.llm_methods as llm_methods

    llm_methods.LLM_CALL_LOGGER = LLMCallLogger(str(llm_calls_dir), truncate_chars=cfg.truncate_chars)

    # Minimal CSV.
    fields = [
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
    csv_path = _init_csv(experiment_dir, fields, filename="lassi-translate_pipeline_data.csv")

    experiment_data = dict.fromkeys(fields, None)
    experiment_data["gpu_vendor"] = detected_vendor
    experiment_data["experiment"] = experiment
    experiment_data["app_name"] = cfg.source_app
    experiment_data["code_line_count"] = lines_of_code
    experiment_data["llm"] = cfg.model
    experiment_data["llm_coder"] = cfg.model_coder
    experiment_data["llm_judge"] = cfg.model_judge

    meta = "***** EXPERIMENT CONFIGURATION SUMMARY *****\n"
    meta += f"\nGPU Vendor: {detected_vendor}"
    meta += f"\nDate-time: {dt_string}"
    meta += f"\nApplication: {app_folder_name}"
    meta += f"\n\nLLM Source: {cfg.llm_source}"
    meta += f"\nModel: {cfg.model}"
    meta += f"\nModel coder: {cfg.model_coder}"
    meta += f"\nModel judge: {cfg.model_judge}"
    meta += "\n\nPower measurement: False\n"

    _log("Experiment directory: " + str(experiment_dir))
    _write_text(experiment_dir / "metadata_summary.txt", meta)

    # Toolchains.
    cuda_make = "make"
    cuda_make_args = "-j SM_VERSION=80 CFLAGS=-std=c++17"

    sycl_compiler = sycl_tc.compiler
    sycl_flags = f"{sycl_tc.cxxflags} {sycl_tc.ldflags}".strip()

    trial_number = 1
    start = time.time()

    complete = lassi.pipeline_v3.auto_code_llmgeneration_pipeline(
        prompts,
        system_prompt,
        translate_prompt,
        cfg.llm_source,
        cfg.model,
        cfg.model_coder,
        cfg.model_judge,
        cfg.num_ctx,
        cfg.num_ctx_coder,
        False,
        True,
        False,
        False,
        "",
        "",
        input_code,
        lines_of_code,
        cfg.target_lang,
        "cpp",
        cfg.source_lang,
        sycl_compiler,
        sycl_flags,
        False,
        True,
        "",
        f"generated_codes/{source_folder}/{app_folder_name}/sycl",
        "Main",
        str(source_file_path),
        cfg.display_response,
        False,
        False,
        True,
        exe_input_parameters,
        meta,
        False,
        True,
        False,
        dt_string,
        True,
        False,
        False,
        experiment,
        cfg.verifyprompt,
        trial_number,
        str(experiment_dir),
        str(csv_path),
        experiment_data,
        cfg.source_app,
    )

    runtime_min = round((time.time() - start) / 60, 2)
    _log(f"Pipeline completed={complete} runtime_min={runtime_min}")

    # Extract stdout blocks from the metadata file written by the pipeline.
    meta_files = sorted(experiment_dir.glob("metadata_*_Trial*.txt"))
    meta_text = meta_files[-1].read_text(encoding="utf-8") if meta_files else ""

    def _extract_block(tag: str) -> str:
        if not meta_text:
            return ""
        start_i = meta_text.find(tag)
        if start_i < 0:
            return ""
        start_i += len(tag)
        next_delim = meta_text.find("----- ", start_i)
        if next_delim < 0:
            return meta_text[start_i:].strip() + "\n"
        return meta_text[start_i:next_delim].strip() + "\n"

    source_stdout = _extract_block("----- SOURCE: EXECUTION OUTPUT -----")
    sycl_stdout = _extract_block("----- GENERATED: EXECUTION OUTPUT -----")

    _write_text(experiment_dir / "source_stdout.txt", source_stdout)
    _write_text(experiment_dir / "sycl_stdout.txt", sycl_stdout)
    _write_text(experiment_dir / "stdout_diff.txt", _diff_text(source_stdout, sycl_stdout, "source_stdout", "sycl_stdout"))

    # Note: pipeline verification is app-specific (e.g., XSBench compares only the checksum line).
    # The diff here is informational only.
    if (experiment_dir / "stdout_diff.txt").read_text(encoding="utf-8").strip():
        _log("STDOUT diff: present (informational; pipeline verification may still have passed)")
    else:
        _log("STDOUT diff: empty")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
