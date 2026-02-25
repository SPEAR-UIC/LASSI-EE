#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import lassi.llm_methods
import lassi.pipeline_v3
from lassi.format_colors import color
from lassi.gpu_detector import detect_gpu_vendors
from lassi.llm_call_logger import LLMCallLogger
from lassi.prompt_dictionary import PromptDictionary
from lassi.sycl_toolchain import get_sycl_toolchain

_SEP = "=" * 72
_B = color.BOLD
_E = color.END


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


def _gpu_info() -> tuple[str, int]:
    """Return (short GPU name, tile count) from sycl-ls, or ("unknown", 0)."""
    sycl_ls = shutil.which("sycl-ls")
    if not sycl_ls:
        icpx = shutil.which("icpx")
        if icpx:
            candidate = os.path.join(os.path.dirname(icpx), "sycl-ls")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                sycl_ls = candidate
    if not sycl_ls:
        return "unknown", 0
    out = subprocess.run([sycl_ls], capture_output=True, text=True).stdout.strip()
    gpu_lines = [ln for ln in out.splitlines() if ":gpu" in ln.lower()]
    if not gpu_lines:
        return "unknown", 0
    line = gpu_lines[0]
    # Extract model name: "Intel(R) Data Center GPU Max 1550" → "Intel Data Center GPU Max 1550"
    m = re.search(r"Intel\(R\)\s+(Data Center GPU\s+\w+(?:\s+\d+)?)", line)
    if m:
        name = "Intel " + m.group(1).strip()
    else:
        # Fallback: take text after last comma
        name = line.split(",")[-1].strip().split("[")[0].strip()[:50]
    return name, len(gpu_lines)


def _icpx_version() -> str:
    try:
        out = subprocess.run(["icpx", "--version"], capture_output=True, text=True).stdout.strip()
        m = re.search(r"(\d{4}\.\d+\.\d+)", out)
        return m.group(1) if m else out.split("\n")[0][:30]
    except Exception:
        return "unknown"


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
        help="Disable Stage 0 self-prompting context builder.",
    )
    p.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable debug mode: print prompt sizes before each LLM call.",
    )
    p.set_defaults(self_prompting=True)
    args = p.parse_args()

    if args.debug:
        import lassi.llm_methods as _lm
        _lm.DEBUG_MODE = True

    os.chdir(Path(__file__).resolve().parent)

    # Detect hardware and toolchain
    vendors = detect_gpu_vendors()
    if not vendors:
        raise SystemExit("No GPU hardware found")
    vendor = vendors[0]["vendor"].upper()

    tc = get_sycl_toolchain()
    gpu_name, gpu_tiles = _gpu_info()
    icpx_ver = _icpx_version()
    cluster = os.environ.get("ALCF_CLUSTER", "metis")

    # ── Config header ────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  {_B}LASSI{_E}  |  XSBench  |  CUDA → SYCL  |  {args.model}")
    print(_SEP)
    tile_str = f"  ×{gpu_tiles} tiles" if gpu_tiles > 1 else ""
    print(f"  LLM       : {_B}{args.model}{_E}  (ALCF Inference — {cluster})")
    print(f"  Hardware  : {_B}{gpu_name}{tile_str}{_E}")
    print(f"  Compiler  : icpx {icpx_ver}  ({tc.cxxflags})")
    mode_str = "CUDA→SYCL with self-prompting + self-correction" if args.self_prompting else "CUDA→SYCL with self-correction"
    print(f"  Mode      : {mode_str}")
    if args.debug:
        print(f"  Debug     : ON")
    print(_SEP)

    input_code, exe_args = _load_source()

    dt_string = datetime.now(ZoneInfo("America/Chicago")).strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_name = f"exp_xsbench_translate_cuda_to_sycl_{args.model}_{vendor}__{dt_string}"
    experiment_dir = Path("generated_codes/experiment") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

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
        verbose=args.debug,
    )

    success = lassi.pipeline_v3.run_pipeline(cfg)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
