#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lassi.llm_methods
import lassi.pipeline_v3
from lassi.gpu_detector import detect_gpu_vendors
from lassi.llm_call_logger import LLMCallLogger
from lassi.prompt_dictionary import PromptDictionary
from lassi.sycl_toolchain import get_sycl_toolchain


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [RUN] {msg}", flush=True)


def _load_source() -> tuple[str, list[str]]:
    src_dir = Path("source_codes/Miniapps/XSBench/cuda")
    code = (src_dir / "Main.cu").read_text(encoding="utf-8")
    headers = []
    for pattern in ("*.cuh", "*.h"):
        for hdr in sorted(src_dir.glob(pattern)):
            headers.append(f"// ===== HEADER: {hdr.name} =====\n{hdr.read_text()}\n// ===== END HEADER: {hdr.name} =====\n")
    if headers:
        code = "\n".join(headers) + "\n" + code
    app_params = json.loads(Path("source_apps.json").read_text())["Miniapps"]["XSBench"]["app_params"]
    return code, app_params


def _intel_gpu_summary() -> None:
    sycl_ls = shutil.which("sycl-ls")
    if not sycl_ls:
        icpx = shutil.which("icpx")
        if icpx:
            candidate = os.path.join(os.path.dirname(icpx), "sycl-ls")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                sycl_ls = candidate
    if sycl_ls:
        out = subprocess.run([sycl_ls], capture_output=True, text=True).stdout.strip()
        gpu_lines = [ln for ln in out.splitlines() if ":gpu" in ln.lower()]
        if gpu_lines:
            suffix = f" (×{len(gpu_lines)} tiles)" if len(gpu_lines) > 1 else ""
            _log(f"sycl-ls: {gpu_lines[0]}{suffix}")


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--verifyprompt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--no-self-prompting", dest="self_prompting", action="store_false",
        help="Disable Stage 0 self-prompting context builder (faster but less context-aware).",
    )
    p.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug mode: print prompt sizes before each LLM call and full error details.",
    )
    p.set_defaults(self_prompting=True)
    args = p.parse_args()

    # Apply debug mode globally
    if args.debug:
        import lassi.llm_methods as _lm
        _lm.DEBUG_MODE = True

    os.chdir(Path(__file__).resolve().parent)
    _log(f"CWD={Path.cwd()}")
    _log("LLM backend: ALCF Inference Endpoints")
    _log("Mode: translate CUDA→SYCL with self-correction")

    vendors = detect_gpu_vendors()
    if not vendors:
        raise SystemExit("No GPU hardware found")
    vendor = vendors[0]["vendor"].upper()
    _log(f"Detected GPU vendor={vendor} count={vendors[0]['count']}")
    if vendor == "INTEL":
        _intel_gpu_summary()

    tc = get_sycl_toolchain()
    _log(f"SYCL toolchain compiler={tc.compiler} cxxflags={tc.cxxflags} ldflags={tc.ldflags}")

    input_code, exe_args = _load_source()
    _log(f"Loaded source: {Path('source_codes/Miniapps/XSBench/cuda/Main.cu').resolve()}")

    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_name = f"exp_xsbench_translate_cuda_to_sycl_{args.model}_{vendor}__{dt_string}"
    experiment_dir = Path("generated_codes/experiment") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Experiment directory: {experiment_dir}")

    lassi.llm_methods.LLM_CALL_LOGGER = LLMCallLogger(str(experiment_dir / "llm_calls"), truncate_chars=4000)

    csv_path = experiment_dir / "results.csv"
    csv_fields = ["timestamp", "app", "model", "gpu_vendor", "success", "runtime_min"]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()
    experiment_data = {
        "timestamp": dt_string,
        "app": "XSBench",
        "model": args.model,
        "gpu_vendor": vendor,
        "success": False,
        "runtime_min": 0.0,
    }

    prompts = PromptDictionary()
    sycl_flags = f"{tc.cxxflags} {tc.ldflags}".strip()

    _log(f"Self-prompting: {'enabled' if args.self_prompting else 'disabled (--no-self-prompting)'}")
    if args.debug:
        _log("Debug mode: ON (prompt sizes logged before each LLM call)")

    cfg = lassi.pipeline_v3.PipelineConfig(
        model=args.model,
        system_prompt=prompts.get_system_prompt("CUDA_to_SYCL"),
        translate_prompt=prompts.get_codetranslate_prompt("CUDA_to_SYCL"),
        prompts=prompts,
        input_code=input_code,
        sycl_compiler=tc.compiler,
        sycl_flags=sycl_flags,
        exe_args=exe_args,
        app_name="XSBench",
        experiment_dir=str(experiment_dir.resolve()),
        csv_path=str(csv_path),
        experiment_data=experiment_data,
        experiment_name=experiment_name,
        trial_number=1,
        verifyprompt=args.verifyprompt,
        use_self_prompting=args.self_prompting,
    )

    start = time.time()
    success = lassi.pipeline_v3.run_pipeline(cfg)
    runtime_min = round((time.time() - start) / 60, 2)
    _log(f"Pipeline success={success} runtime_min={runtime_min}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
