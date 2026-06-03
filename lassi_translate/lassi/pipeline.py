"""Core LASSI pipelines: baseline measurement, run_pipeline, run_both_pipeline."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import lassi.llm_client as _llm
from lassi.format_colors    import color as _c
from lassi.hpc_jobs         import JobRunner, JOB_DURATION_S, PersistentWorker, build_runner
from lassi.hpc_session      import HPCSession
from lassi.prompt_builder   import (
    CODE_COMPARISON_PROMPT, build_correction_prompt, build_stage1_prompt,
    run_judge, run_selfprompt,
)
from lassi.prompt_dictionary import PromptDictionary
from lassi.resources         import ResourceConfig
from lassi.results           import (
    parse_attempt_results, save_attempt_logs, tail5, xsbench_checksum_status,
)
from lassi.source_loader     import sync_file_to_remote

_B, _G, _R, _E = _c.BOLD, _c.GREEN, _c.RED, _c.END

_DEFAULT_MIN_GAIN_PCT = 1.0
_MAX_ATTEMPTS         = 15


# ─────────────────────────────────────────────────────────────────────────────
# Demo terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _fmt_stdout_preview(stdout: str, max_lines: int = 8) -> str:
    """Last `max_lines` non-blank lines, indented for the summary block."""
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return "      (no output)"
    return "\n".join(f"      {ln}" for ln in lines[-max_lines:])


def print_translate_summary(
    *,
    resource: str,
    original_framework: str,
    target_framework: str,
    baseline: dict | None,
    generated: dict,
    total_elapsed_s: float,
    versioned_path: Path,
    verification: str,
) -> None:
    """Side-by-side perf comparison shown at the end of a translate demo."""
    rule = "─" * 72
    print(f"\n{rule}")
    print(f"  {_B}{_G}✅  TRANSLATION COMPLETE{_E}  —  total time: "
          f"{_B}{_fmt_duration(total_elapsed_s)}{_E}")
    print(rule)

    # Translate summary intentionally omits energy/power per the broader-codebase
    # convention (lassi/pipeline_v3.py): EE metrics are an "energy task" concern.
    def _block(label: str, m: dict | None) -> list[str]:
        if m is None:
            return [
                f"  {_B}{label}{_E}",
                f"      (not measured — incompatible with {resource})",
            ]
        lines = [f"  {_B}{label}{_E}"]
        if m.get("duration_s", -1) > 0:
            lines.append(f"      duration : {m['duration_s']:.2f} s")
        if verification == "xsbench":
            cs_line, cs_valid = xsbench_checksum_status(m.get("run_stdout", ""))
            mark = f"{_G}✓{_E}" if cs_valid else f"{_R}✗{_E}"
            lines.append(f"      checksum : {mark} {cs_line or '(missing)'}")
        return lines

    blk_a = _block(
        f"Original ({original_framework.upper()} on {resource})",
        baseline,
    )
    blk_b = _block(
        f"Generated ({target_framework.upper()} on {resource})",
        generated,
    )
    for line in blk_a:
        print(line)
    print()
    for line in blk_b:
        print(line)

    # Last-N-lines stdout preview
    if baseline is not None and baseline.get("run_stdout"):
        print(f"\n  {_B}Original stdout (tail):{_E}")
        print(_fmt_stdout_preview(baseline.get("run_stdout", "")))
    if generated.get("run_stdout"):
        print(f"\n  {_B}Generated stdout (tail):{_E}")
        print(_fmt_stdout_preview(generated.get("run_stdout", "")))

    print(f"\n  {_B}Saved:{_E} {versioned_path}")
    print(f"{rule}")
    print(f"  {_B}{_G}Job terminated.{_E}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline measurement (energy task only)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    *,
    session: HPCSession, runner: JobRunner,
    remote_dir: str,
    original_src_path: Path, src_extension: str,
    experiment_name: str, experiment_dir: Path,
    run_args: list[str], run_id: str,
    verification: str = "none",
) -> tuple[float, float, str]:
    """Compile + run the original source to establish a baseline energy.

    Returns (energy_joules, avg_power_w, run_stdout); -1.0/"" on failure.
    """
    print(f"\n{_B}----- BASELINE: compiling + running original source  (PBS) -----{_E}\n")

    baseline_name = f"Main_baseline{src_extension}"
    exp_remote    = f"{remote_dir}/generated_codes/experiment/{experiment_name}"

    print(f"  Uploading {baseline_name} to remote FS...", end="  ", flush=True)
    if not sync_file_to_remote(session, original_src_path, remote_dir, experiment_name, baseline_name):
        print(f"{_R}FAILED{_E}")
        return -1.0, -1.0, ""
    print(f"{_G}OK{_E}")

    # Sync companion headers so the compiler can resolve #include directives.
    src_dir = original_src_path.parent
    for pattern in ("*.cuh", "*.h", "*.hpp"):
        for hdr in sorted(src_dir.glob(pattern)):
            session.rsync(str(hdr), f"{exp_remote}/{hdr.name}")

    results_remote = f"{exp_remote}/baseline_results"
    session.mkdir(results_remote)

    try:
        runner.run(
            src_remote  = f"{exp_remote}/{baseline_name}",
            results_dir = results_remote,
            run_args    = run_args,
            job_name    = f"lassi-baseline-{run_id}",
        )
    except Exception as exc:
        print(f"  {_R}Baseline job failed: {exc}{_E}")
        return -1.0, -1.0, ""

    results = parse_attempt_results(session, results_remote, has_energy=True)
    save_attempt_logs(results, experiment_dir, "baseline_")

    if not results["build_ok"]:
        print(f"  {_R}Baseline compile FAILED — cannot establish energy reference.{_E}")
        return -1.0, -1.0, ""
    if not results["run_ok"]:
        print(f"  {_R}Baseline run FAILED — cannot establish energy reference.{_E}")
        return -1.0, -1.0, ""

    run_stdout = results.get("run_stdout") or ""
    if verification == "xsbench":
        cs_line, cs_valid = xsbench_checksum_status(run_stdout)
        if not cs_valid:
            print(f"  {_R}Baseline checksum invalid ({cs_line!r}) — energy reference may be unreliable.{_E}")

    ej  = results.get("energy_joules", -1.0)
    apw = results.get("avg_power_w",   -1.0)
    ds  = results.get("duration_s",    -1.0)
    ns  = results.get("num_samples",    0)

    if ej > 0:
        print(f"  {_B}{_G}Baseline energy : {ej:.1f} J  ({apw:.1f} W avg,  {ds:.1f} s,  {ns} samples){_E}")
    else:
        print(f"  {_R}Warning: energy measurement unavailable (power monitor not found?).{_E}")
        print(f"  Pipeline will proceed verifying correctness only.")

    (experiment_dir / "baseline_energy.json").write_text(
        json.dumps({"energy_joules": ej, "avg_power_w": apw,
                    "duration_s": ds, "num_samples": ns,
                    "run_stdout": run_stdout}, indent=2),
        encoding="utf-8",
    )
    return ej, apw, run_stdout


def read_baseline_metrics(experiment_dir: Path) -> dict | None:
    """Read baseline_energy.json into a metrics dict (or None if missing/empty).

    A baseline is considered usable if it has either a positive energy reading
    or a positive duration — supports both energy-monitored and no-monitor runs.
    """
    p = experiment_dir / "baseline_energy.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if d.get("energy_joules", -1) <= 0 and d.get("duration_s", -1) <= 0:
        return None
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Translate-only test run: compile + run the original source for output
# comparison.  No energy framing — runtime + stdout + checksum is enough.
# Mirrors the broader codebase's `power_measure=False` gate (lassi/pipeline_v3.py).
# ─────────────────────────────────────────────────────────────────────────────

def run_test_run(
    *,
    session: HPCSession, runner: JobRunner,
    remote_dir: str,
    original_src_path: Path, src_extension: str,
    experiment_name: str, experiment_dir: Path,
    run_args: list[str], run_id: str,
    verification: str = "none",
) -> dict | None:
    """Compile and run the original source once to capture its output + duration.

    Returns a metrics dict ({duration_s, run_stdout, num_samples, ...}) or None
    on failure.  Used by the translate demo to display side-by-side comparison.
    Energy/power fields may be absent or -1; callers should not surface them.
    """
    print(f"\n{_B}----- TEST RUN: compiling + running original source -----{_E}\n")

    test_name = f"Main_baseline{src_extension}"
    exp_remote = f"{remote_dir}/generated_codes/experiment/{experiment_name}"

    print(f"  Uploading {test_name} to remote FS...", end="  ", flush=True)
    if not sync_file_to_remote(session, original_src_path, remote_dir, experiment_name, test_name):
        print(f"{_R}FAILED{_E}")
        return None
    print(f"{_G}OK{_E}")

    src_dir = original_src_path.parent
    for pattern in ("*.cuh", "*.h", "*.hpp"):
        for hdr in sorted(src_dir.glob(pattern)):
            session.rsync(str(hdr), f"{exp_remote}/{hdr.name}")

    results_remote = f"{exp_remote}/baseline_results"
    session.mkdir(results_remote)

    try:
        runner.run(
            src_remote  = f"{exp_remote}/{test_name}",
            results_dir = results_remote,
            run_args    = run_args,
            job_name    = f"lassi-testrun-{run_id}",
        )
    except Exception as exc:
        print(f"  {_R}Test run failed: {exc}{_E}")
        return None

    # has_energy=True only so parse_attempt_results reads energy_results.txt
    # (the worker now always writes at least duration_s there, even with no monitor).
    results = parse_attempt_results(session, results_remote, has_energy=True)
    save_attempt_logs(results, experiment_dir, "baseline_")

    if not results["build_ok"]:
        print(f"  Compiling...  {_B}{_R}✗  compile error{_E}")
        return None
    print(f"  Compiling...  {_B}{_G}✓  OK{_E}")

    if not results["run_ok"]:
        print(f"  Running...    {_B}{_R}✗  runtime error{_E}")
        return None
    run_stdout = results.get("run_stdout") or ""
    duration   = results.get("duration_s", -1.0)
    print(f"  Running...    {_B}{_G}✓  OK{_E}"
          + (f"  ({duration:.2f}s)" if duration > 0 else ""))

    if verification == "xsbench":
        cs_line, cs_valid = xsbench_checksum_status(run_stdout)
        mark = f"{_G}✓{_E}" if cs_valid else f"{_R}✗{_E}"
        print(f"  Checksum...   {mark} {cs_line or '(missing)'}")

    metrics = {
        "run_stdout":  run_stdout,
        "duration_s":  duration,
        # Energy fields preserved for callers that DO want them (energy task),
        # but the translate summary intentionally ignores them.
        "energy_joules": results.get("energy_joules", -1.0),
        "avg_power_w":   results.get("avg_power_w",   -1.0),
        "num_samples":   results.get("num_samples",    0),
    }
    (experiment_dir / "baseline_energy.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metadata artifact writer (mirrors base LASSI-EE pipeline.py format)
# ─────────────────────────────────────────────────────────────────────────────

def write_metadata(
    *,
    experiment_dir: Path, experiment_name: str,
    runtime_s: float, attempt_count: int,
    energy_efficiency_iteration: int,
    has_energy: bool,
    baseline_stdout: str, best_run_stdout: str,
    baseline_energy_j: float, baseline_power_w: float,
    best_energy: float, best_power_w: float, best_dur_s: float,
    energy_efficiency_success: bool,
    input_code: str, best_code_block: str,
    model: str, system_prompt: str,
) -> None:
    """Write metadata_{experiment_name}.txt matching base LASSI-EE artifact format."""
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    baseline_dur_s = -1.0
    baseline_json_path = experiment_dir / "baseline_energy.json"
    if baseline_json_path.exists():
        try:
            baseline_dur_s = float(
                json.loads(baseline_json_path.read_text(encoding="utf-8")).get("duration_s", -1.0)
            )
        except Exception:
            pass

    meta: list[str] = ["\n\n----- METRICS -----"]
    meta.append(f"\nExperiment date-time: {dt_string}")
    meta.append(f"\nTime (sec): {round(runtime_s, 2)}")
    meta.append(f"\nSelf-correction iteration count: {attempt_count}")
    if has_energy:
        meta.append(f"\nEnergy efficiency improvement count: {energy_efficiency_iteration}")

    if baseline_stdout:
        meta.append("\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n")
        meta.append(baseline_stdout)
    if best_run_stdout:
        meta.append("\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n")
        meta.append(best_run_stdout)

    if has_energy and baseline_energy_j > 0:
        meta.append("\n\n\n----- ENERGY EFFICIENCY METRICS -----\n")
        meta.append("Total Energy (J):")
        meta.append(f"  Source code: {round(baseline_energy_j, 2)}")
        if best_energy > 0:
            meta.append(f"  Refactored code: {round(best_energy, 2)}")
        meta.append("\nAverage Power (W):")
        meta.append(f"  Source code: {round(baseline_power_w, 2)}")
        if best_power_w > 0:
            meta.append(f"  Refactored code: {round(best_power_w, 2)}")
        meta.append("\nRuntime (s):")
        if baseline_dur_s > 0:
            meta.append(f"  Source code: {round(baseline_dur_s, 2)}")
        if best_dur_s > 0:
            meta.append(f"  Refactored code: {round(best_dur_s, 2)}")

    if energy_efficiency_success and best_code_block and baseline_energy_j > 0 and best_energy > 0:
        print(f"\n{_B}----- BEGIN CODE COMPARISON -----{_E}\n")
        compare_prompt = (
            CODE_COMPARISON_PROMPT
            + "\n\n[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_code
            + "\n\nOUTPUT: \n\n" + baseline_stdout
        )
        if baseline_dur_s > 0:
            compare_prompt += f"\n\nCode execution duration: {round(baseline_dur_s, 2)} s"
        compare_prompt += (
            f"\nTotal Energy Consumed: {round(baseline_energy_j, 2)} J"
            f"\nAverage Power: {round(baseline_power_w, 2)} W"
            "\n\n[/END SOURCE CODE AND RESULTS]"
            "\n\n[BEGIN REFACTORED CODE AND RESULTS]\n\n" + best_code_block
            + "\n\nOUTPUT: \n\n" + best_run_stdout
        )
        if best_dur_s > 0:
            compare_prompt += f"\n\nCode execution duration: {round(best_dur_s, 2)} s"
        compare_prompt += (
            f"\nTotal Energy Consumed: {round(best_energy, 2)} J"
            f"\nAverage Power: {round(best_power_w, 2)} W"
            "\n\n[/END REFACTORED CODE AND RESULTS]"
        )

        response = _llm.generate_text(
            model, system_prompt, compare_prompt,
            intent="code comparison",
            label="[LLM] Code comparison",
        )
        meta.append("\n\n\n----- CODE COMPARISON -----\n")
        meta.append(response)

    metadata_path = experiment_dir / f"metadata_{experiment_name}.txt"
    metadata_path.write_text("".join(meta), encoding="utf-8")
    print(f"\n  Metadata file: {metadata_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers for run_pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _correctness_note(verification: str) -> str:
    """Reminder appended to energy correction prompts."""
    if verification == "xsbench":
        return (
            "CRITICAL: preserve the exact XSBench verification checksum — "
            "do NOT alter numerical computation, loop bounds, or comparison operators."
        )
    return "CRITICAL: preserve the exact behavior and output of the original source code."


def _save_success(
    *,
    experiment_dir: Path, csv_path: Path, experiment_data: dict,
    trial_number: int, src_extension: str,
    saved_code: str, start_time: float,
    energy_fields: dict | None = None,
) -> tuple[Path, float]:
    """Save Main_v1_TrialN, append CSV row; return (versioned_path, runtime_min)."""
    versioned   = experiment_dir / f"Main_v1_Trial{trial_number}{src_extension}"
    versioned.write_text(saved_code, encoding="utf-8")

    runtime_min = round((time.time() - start_time) / 60, 2)
    experiment_data["success"]     = True
    experiment_data["runtime_min"] = runtime_min
    if energy_fields:
        experiment_data.update(energy_fields)

    with csv_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writerow(experiment_data)
    return versioned, runtime_min


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline: Stage 0 → Stage 1 → Stage 2 loop
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    *,
    model: str, task: str, resource: str,
    input_code: str,
    session: HPCSession, runner: JobRunner, remote_dir: str,
    src_extension: str, run_args: list[str],
    experiment_name: str, experiment_dir: Path,
    csv_path: Path, experiment_data: dict,
    trial_number: int, run_id: str,
    use_self_prompting: bool = True,
    verifyprompt: bool = False, verbose: bool = False,
    baseline_energy_j: float = -1.0, baseline_power_w: float = -1.0,
    baseline_stdout: str = "",
    min_energy_gain_pct: float = _DEFAULT_MIN_GAIN_PCT,
    has_energy: bool = False,
    use_judge: bool = True,
    verification: str = "none",
    measure_perf: bool = False,
) -> tuple[bool, str, dict]:
    """Run the full LASSI pipeline (Stage 0 → Stage 1 → Stage 2 loop).

    Returns (success, final_code_block, final_metrics).
    `final_metrics` contains energy/power/duration/stdout of the most recent
    successful attempt (empty dict if no attempt succeeded).
    `measure_perf=True` enables energy reading on the worker results even when
    `has_energy=False` — used by translate demos that want timing+energy on
    the generated code without triggering the energy-improvement loop.
    """
    prompts    = PromptDictionary()
    start_time = time.time()

    # ── Stage 0: self-prompting ──────────────────────────────────────────────
    guide_summary, source_description = run_selfprompt(
        model=model, prompts=prompts, input_code=input_code,
        task=task, resource=resource,
        verifyprompt=verifyprompt, use_self_prompting=use_self_prompting,
    )

    # ── Build Stage 1 prompts ────────────────────────────────────────────────
    system_prompt, stage1_user_prompt = build_stage1_prompt(
        prompts=prompts, task=task, resource=resource,
        input_code=input_code,
        guide_summary=guide_summary, source_description=source_description,
        baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
    )

    session.mkdirs(
        f"{remote_dir}/generated_codes",
        f"{remote_dir}/generated_codes/experiment",
        f"{remote_dir}/generated_codes/experiment/{experiment_name}",
    )

    energy_check_enabled = has_energy and baseline_energy_j > 0
    # When the caller only wants to measure perf (no energy gating), still
    # read energy results so the demo summary can show them.
    collect_energy = has_energy or measure_perf

    # ── Loop state ────────────────────────────────────────────────────────────
    compile_err = run_err = run_stdout = code_block = ""
    last_metrics: dict = {}
    best_energy                 = -1.0
    best_code_block             = ""    # best-performing code so far
    best_run_stdout             = ""
    best_power_w                = -1.0
    best_dur_s                  = -1.0
    last_power_w                = -1.0
    last_dur_s                  = -1.0
    code_gen_temp               = 0.2   # increments on energy-insufficient iterations
    energy_correction_cnt       = 0
    energy_efficiency_iteration = 0     # counts successful energy improvements
    energy_efficiency_success   = False
    attempt                     = 0

    for attempt in range(_MAX_ATTEMPTS + 1):

        # ── LLM call ──────────────────────────────────────────────────────────
        if attempt == 0:
            stage_label = {
                "translate":   "CUDA→SYCL TRANSLATION",
                "energy":      "CUDA ENERGY OPTIMIZATION" if resource == "polaris" else "SYCL ENERGY OPTIMIZATION",
                "sycl_energy": "SYCL ENERGY OPTIMIZATION",
            }.get(task, "CODE GENERATION")
            print(f"\n{_B}----- STAGE 1: {stage_label}  (local Argo) -----{_E}\n")
            code_block = _llm.generate_code(
                model, system_prompt, stage1_user_prompt,
                verifyprompt=verifyprompt,
                label="[LLM] Generating initial code",
                temperature=0.2,
            )
        else:
            if attempt == 1:
                print(f"\n{_B}----- STAGE 2: SELF-CORRECTION  (local Argo + PBS) -----{_E}\n")

            err_type, error_user_prompt = build_correction_prompt(
                prompts=prompts, task=task, resource=resource,
                code_block=code_block,
                compile_err=compile_err, run_err=run_err, run_stdout=run_stdout,
                input_code=input_code, attempt=attempt,
                guide_summary=guide_summary, source_description=source_description,
                best_code_block=best_code_block,
                opt_power_w=last_power_w, opt_dur_s=last_dur_s,
            )
            print(f"  {_B}Attempt {attempt}/{_MAX_ATTEMPTS}{_E} — correcting {err_type}")

            # Temperature strategy (mirrors base LASSI-EE):
            #  - energy-insufficient (no prior success): increment by 0.2; stop at ceiling.
            #  - energy keep-improving (prior success): stay at 0.2 (reset on improvement).
            #  - compile/runtime errors: always 0.2.
            if err_type == "energy improvement insufficient":
                energy_correction_cnt += 1
                call_temp     = code_gen_temp
                code_gen_temp = round(min(code_gen_temp + 0.2, 1.0), 1)
                if call_temp >= 1.0:
                    print(f"  {_B}Max LLM temperature reached — stopping self-correction.{_E}")
                    break
            elif err_type == "energy keep improving":
                call_temp = code_gen_temp  # already reset to 0.2 on last improvement
            else:
                call_temp = 0.2

            code_block = _llm.generate_code(
                model, system_prompt, error_user_prompt,
                verifyprompt=verifyprompt,
                label=f"[LLM] Corrected code (attempt {attempt})",
                temperature=call_temp,
            )

        # ── Write + upload generated source ───────────────────────────────────
        src_name  = f"Main_attempt{attempt}{src_extension}"
        local_src = experiment_dir / src_name
        local_src.write_text(code_block, encoding="utf-8")

        print(f"  Uploading {src_name} to remote FS...", end="  ", flush=True)
        if not sync_file_to_remote(session, local_src, remote_dir, experiment_name, src_name):
            print(f"{_R}FAILED{_E}")
            compile_err = f"rsync failed — could not upload {src_name}"
            run_stdout  = ""
            continue
        print(f"{_G}OK{_E}")

        src_remote_path    = f"{remote_dir}/generated_codes/experiment/{experiment_name}/{src_name}"
        results_remote_dir = f"{remote_dir}/generated_codes/experiment/{experiment_name}/attempt{attempt}_results"
        session.mkdir(results_remote_dir)

        try:
            runner.run(
                src_remote=src_remote_path,
                results_dir=results_remote_dir,
                run_args=run_args,
                job_name=f"lassi-{attempt}-{run_id}",
                verbose=verbose,
            )
        except RuntimeError as exc:
            compile_err = str(exc)
            run_stdout  = ""
            break
        except Exception as exc:
            compile_err = str(exc)
            run_stdout  = ""
            continue

        # ── Read results from remote FS ───────────────────────────────────────
        results = parse_attempt_results(session, results_remote_dir, collect_energy)
        save_attempt_logs(results, experiment_dir, f"attempt{attempt}_")

        # ── Compile result ────────────────────────────────────────────────────
        if not results["build_ok"]:
            tail = tail5(results["build_stderr"])
            print(f"  Compiling...  {_B}{_R}✗  compile error:{_E}")
            if tail:
                print(tail)
            compile_err = results["build_stderr"] or "Compilation failed (no stderr captured)"
            run_stdout  = ""
            continue
        print(f"  Compiling...  {_B}{_G}✓  OK{_E}")

        # ── Run result ────────────────────────────────────────────────────────
        run_stdout = results["run_stdout"] or ""
        if not results["run_ok"]:
            combined = (results["run_stderr"] or "") + "\n" + run_stdout
            tail = tail5(combined)
            print(f"  Running...    {_B}{_R}✗  runtime error:{_E}")
            if tail:
                print(tail)
            compile_err = "execute_error"
            run_err     = results["run_stderr"] or "Exited with non-zero exit code"
            continue
        print(f"  Running...    {_B}{_G}✓  OK{_E}")

        # ── Checksum verification ─────────────────────────────────────────────
        if verification == "xsbench":
            cs_line, cs_valid = xsbench_checksum_status(run_stdout)
            if not cs_valid:
                print(f"  Checksum...   {_B}{_R}✗  {cs_line or 'missing'}{_E}")
                snippet = (run_stdout[:1000].strip() or "(no output)") + (
                    "\n...(truncated)" if len(run_stdout) > 1000 else ""
                )
                compile_err = (
                    "VERIFICATION MISMATCH: Checksum missing or invalid.\n"
                    f"Checksum line: {cs_line!r}\n"
                    f"\nActual stdout:\n{snippet}\n"
                    "\nYour code MUST faithfully implement ALL functions from "
                    "the original source, including print_results(), "
                    "run_event_based_simulation_baseline(), and grid_init_do_not_profile().\n"
                    "Do NOT generate a simplified stub.\n"
                    f"\n[ORIGINAL SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL SOURCE]\n"
                )
                continue
            print(f"  Checksum...   {_B}{_G}✓  {cs_line}{_E}")

        # ── LLM judge ─────────────────────────────────────────────────────────
        if use_judge and baseline_stdout:
            src_metrics = (
                f"Energy: {baseline_energy_j:.1f} J, Avg Power: {baseline_power_w:.1f} W"
                if baseline_energy_j > 0 else "(not measured)"
            )
            opt_ej_pre = results.get("energy_joules", -1.0)
            opt_pw_pre = results.get("avg_power_w",   -1.0)
            ref_metrics = (
                f"Energy: {opt_ej_pre:.1f} J, Avg Power: {opt_pw_pre:.1f} W"
                if opt_ej_pre > 0 else "(not measured)"
            )
            verdict = run_judge(
                model=model, system_prompt=system_prompt,
                source_code=input_code, source_stdout=baseline_stdout,
                source_metrics=src_metrics,
                refactored_code=code_block, refactored_stdout=run_stdout,
                refactored_metrics=ref_metrics,
            )
            if verdict != "VALID":
                # Base LASSI-EE requires VALID for an improvement to count;
                # INVALID and NONE both block the attempt.
                reason = (
                    "the refactored code does not preserve functional equivalence or standard output"
                    if verdict == "INVALID" else
                    "judge could not determine functional equivalence (NONE verdict)"
                )
                compile_err = (
                    f"VERIFICATION MISMATCH (LLM judge): {reason}. "
                    "Ensure ALL original logic, print statements, and computation are preserved."
                )
                run_stdout = ""
                continue

        # ── Energy check (energy task only) ───────────────────────────────────
        if has_energy:
            opt_energy_j = results.get("energy_joules", -1.0)
            opt_power_w  = results.get("avg_power_w",   -1.0)
            opt_dur_s    = results.get("duration_s",    -1.0)
            opt_samples  = results.get("num_samples",    0)

            if opt_energy_j > 0:
                print(
                    f"  Energy...     {_B}{opt_energy_j:.1f} J{_E}  "
                    f"({opt_power_w:.1f} W avg,  {opt_dur_s:.1f} s,  {opt_samples} samples)"
                )
                last_power_w, last_dur_s = opt_power_w, opt_dur_s
            else:
                print(f"  Energy...     {_B}(measurement unavailable){_E}")

            if energy_check_enabled and opt_energy_j > 0:
                reduction_pct = (baseline_energy_j - opt_energy_j) / baseline_energy_j * 100.0
                comparison_j  = best_energy if best_energy > 0 else baseline_energy_j

                if opt_energy_j < comparison_j:
                    # ── Energy improved ──
                    prev_best       = best_energy
                    best_energy     = opt_energy_j
                    best_code_block = code_block
                    best_run_stdout = run_stdout
                    best_power_w    = opt_power_w
                    best_dur_s      = opt_dur_s
                    energy_efficiency_success    = True
                    energy_correction_cnt        = 0
                    code_gen_temp                = 0.2  # reset on improvement
                    energy_efficiency_iteration += 1

                    print(
                        f"  Reduction...  {_B}{_G}↓  {reduction_pct:.1f}%  "
                        f"({baseline_energy_j:.1f} J → {opt_energy_j:.1f} J)  "
                        f"[new best, iteration {energy_efficiency_iteration}]{_E}"
                    )

                    # ── Stagnation stop (mirrors base LASSI-EE) ──
                    if prev_best > 0 and energy_efficiency_iteration > 4:
                        marginal = (prev_best - opt_energy_j) / prev_best
                        if marginal < 0.011 and opt_energy_j < baseline_energy_j:
                            print(
                                f"  {_B}Energy improvement stagnating "
                                f"({marginal*100:.2f}% marginal gain) — stopping self-correction.{_E}"
                            )
                            break

                    from_j = prev_best if prev_best > 0 else baseline_energy_j
                    compile_err = (
                        f"ENERGY_KEEP_IMPROVING: Code improved from "
                        f"{from_j:.1f} J to {opt_energy_j:.1f} J "
                        f"({reduction_pct:+.1f}% vs baseline of {baseline_energy_j:.1f} J). "
                        f"Apply additional aggressive energy optimizations to reduce further.\n"
                        f"{_correctness_note(verification)}"
                    )
                    continue
                else:
                    # ── No improvement vs best (or baseline) ──
                    ref_label = (
                        f"best {best_energy:.1f} J" if best_energy > 0
                        else f"baseline {baseline_energy_j:.1f} J"
                    )
                    print(
                        f"  Reduction...  {_B}{_R}✗  {reduction_pct:.1f}%  "
                        f"(no improvement vs {ref_label}){_E}"
                    )
                    compile_err = (
                        f"ENERGY_NOT_IMPROVED: Current code uses {opt_energy_j:.1f} J "
                        f"versus baseline {baseline_energy_j:.1f} J "
                        f"({reduction_pct:+.1f}% change, need ≥{min_energy_gain_pct:.1f}% reduction).\n"
                        f"Apply more aggressive energy optimizations.\n"
                        f"{_correctness_note(verification)}"
                    )
                    continue

            elif energy_check_enabled and opt_energy_j <= 0:
                # Measurement unavailable — keep iterating; preserve last compile_err.
                if not compile_err:
                    compile_err = "ENERGY_NOT_IMPROVED: Energy measurement unavailable on this run. Retry."
                continue
            # else: no baseline — fall through to success block

        # ── Success (translate, or energy w/o baseline) ──────────────────────
        if has_energy and energy_check_enabled:
            pass  # safety net — energy tasks with baseline always continue/break above
        else:
            task_label     = "translate" if task == "translate" else "energy"
            last_metrics = {
                "run_stdout":    run_stdout,
                "energy_joules": results.get("energy_joules", -1.0),
                "avg_power_w":   results.get("avg_power_w",   -1.0),
                "duration_s":    results.get("duration_s",    -1.0),
                "num_samples":   results.get("num_samples",    0),
                "attempt_count": attempt,
            }
            versioned, runtime_min = _save_success(
                experiment_dir=experiment_dir, csv_path=csv_path,
                experiment_data=experiment_data,
                trial_number=trial_number, src_extension=src_extension,
                saved_code=code_block, start_time=start_time,
            )
            attempts_label = f"attempt {attempt + 1}" if attempt > 0 else "first attempt"
            print(
                f"\n  {_B}{_G}✅  {task_label.capitalize()} complete{_E}"
                f"  —  {attempts_label}  |  {runtime_min} min total\n"
            )
            print(f"  Successful code : {versioned}")
            print(f"  Remote results  : {results_remote_dir}")

            write_metadata(
                experiment_dir=experiment_dir, experiment_name=experiment_name,
                runtime_s=time.time() - start_time,
                attempt_count=attempt,
                energy_efficiency_iteration=energy_efficiency_iteration,
                has_energy=has_energy,
                baseline_stdout=baseline_stdout, best_run_stdout=run_stdout,
                baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
                best_energy=-1.0, best_power_w=-1.0, best_dur_s=-1.0,
                energy_efficiency_success=False,
                input_code=input_code, best_code_block=code_block,
                model=model, system_prompt=system_prompt,
            )
            return True, code_block, last_metrics

    # ── Post-loop energy verdict ─────────────────────────────────────────────
    # Reached when the loop exits via break (stagnation/temperature ceiling)
    # or exhausts all attempts.
    if has_energy and energy_check_enabled and best_energy > 0:
        best_reduction_pct = (baseline_energy_j - best_energy) / baseline_energy_j * 100.0
        if best_reduction_pct >= min_energy_gain_pct:
            versioned, runtime_min = _save_success(
                experiment_dir=experiment_dir, csv_path=csv_path,
                experiment_data=experiment_data,
                trial_number=trial_number, src_extension=src_extension,
                saved_code=best_code_block, start_time=start_time,
                energy_fields={
                    "baseline_energy_j":    baseline_energy_j,
                    "optimized_energy_j":   best_energy,
                    "energy_reduction_pct": round(best_reduction_pct, 2),
                },
            )
            energy_summary = (
                f"  Energy   : {baseline_energy_j:.1f} J → {best_energy:.1f} J  "
                f"({best_reduction_pct:+.1f}%)  "
                f"[{energy_efficiency_iteration} improvement(s)]\n"
            )
            print(
                f"\n  {_B}{_G}✅  Energy complete{_E}"
                f"  —  {runtime_min} min total\n"
                + energy_summary
            )
            print(f"  Best code saved : {versioned}")

            write_metadata(
                experiment_dir=experiment_dir, experiment_name=experiment_name,
                runtime_s=time.time() - start_time,
                attempt_count=attempt,
                energy_efficiency_iteration=energy_efficiency_iteration,
                has_energy=has_energy,
                baseline_stdout=baseline_stdout, best_run_stdout=best_run_stdout,
                baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
                best_energy=best_energy, best_power_w=best_power_w, best_dur_s=best_dur_s,
                energy_efficiency_success=energy_efficiency_success,
                input_code=input_code, best_code_block=best_code_block,
                model=model, system_prompt=system_prompt,
            )
            last_metrics = {
                "run_stdout":    best_run_stdout,
                "energy_joules": best_energy,
                "avg_power_w":   best_power_w,
                "duration_s":    best_dur_s,
                "num_samples":   0,
                "attempt_count": attempt,
            }
            return True, best_code_block, last_metrics

    print(
        f"\n{_B}{_R}❌  PIPELINE: no sufficient energy improvement found "
        f"(best: {best_energy:.1f} J, baseline: {baseline_energy_j:.1f} J).{_E}\n"
        if has_energy and energy_check_enabled else
        f"\n{_B}{_R}❌  PIPELINE ERROR: exceeded {_MAX_ATTEMPTS} self-correction attempts.{_E}\n"
    )

    write_metadata(
        experiment_dir=experiment_dir, experiment_name=experiment_name,
        runtime_s=time.time() - start_time,
        attempt_count=attempt,
        energy_efficiency_iteration=energy_efficiency_iteration,
        has_energy=has_energy,
        baseline_stdout=baseline_stdout, best_run_stdout=best_run_stdout,
        baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
        best_energy=best_energy, best_power_w=best_power_w, best_dur_s=best_dur_s,
        energy_efficiency_success=False,
        input_code=input_code, best_code_block=best_code_block,
        model=model, system_prompt=system_prompt,
    )
    return False, (best_code_block if best_code_block else code_block), last_metrics


# ─────────────────────────────────────────────────────────────────────────────
# "Both" pipeline: translate CUDA→SYCL then measure SYCL energy
# ─────────────────────────────────────────────────────────────────────────────

def run_both_pipeline(
    *,
    model: str, resource: str, resource_cfg: ResourceConfig,
    session: HPCSession, compute_resource,
    remote_dir: str, account: str, queue: str,
    run_args: list[str],
    cuda_input_code: str, original_cu_path: Path,
    experiment_dir: Path, experiment_name: str,
    csv_path: Path, run_id: str,
    sycl_compiler_env: dict[str, str],
    sycl_energy_compiler_env: dict[str, str],
    use_self_prompting: bool, verifyprompt: bool, verbose: bool,
    min_energy_gain_pct: float,
    use_judge: bool = True, verification: str = "none",
    walltime_s: int = JOB_DURATION_S,
    persistent: PersistentWorker | None = None,
) -> bool:
    """Run translate then energy measurement of the translated SYCL code."""

    def _runner(worker_key: str, compiler_env: dict[str, str]) -> JobRunner:
        return build_runner(
            session=session, compute_resource=compute_resource, resource=resource,
            remote_dir=remote_dir, account=account, queue=queue,
            filesystems=resource_cfg.filesystems,
            worker_key=worker_key, compiler_env=compiler_env,
            walltime_s=walltime_s, persistent=persistent,
        )

    # ── Step 1: Translation ──────────────────────────────────────────────────
    print(f"\n{'='*72}\n  {_B}BOTH TASK — Step 1 of 2: CUDA→SYCL Translation{_E}\n{'='*72}")

    translate_exp_name = experiment_name + "_translate"
    translate_exp_dir  = experiment_dir / "translate"
    translate_exp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = experiment_name.split("__")[-1] if "__" in experiment_name else ""
    translate_csv  = translate_exp_dir / "results_translate.csv"
    translate_data = {
        "timestamp": timestamp, "app": "XSBench",
        "model": model, "gpu_vendor": resource_cfg.gpu_vendor,
        "success": False, "runtime_min": 0.0,
    }
    with translate_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(translate_data.keys())).writeheader()

    _llm.configure(log_dir=str(translate_exp_dir / "llm_calls"),
                   truncate_chars=4000, debug=verbose, stream_display=True)

    translate_ok, sycl_code, _ = run_pipeline(
        model=model, task="translate", resource=resource,
        input_code=cuda_input_code,
        session=session, runner=_runner("translate", sycl_compiler_env),
        remote_dir=remote_dir, src_extension=".cpp",
        run_args=run_args,
        experiment_name=translate_exp_name, experiment_dir=translate_exp_dir,
        csv_path=translate_csv, experiment_data=translate_data,
        trial_number=1, run_id=run_id,
        use_self_prompting=use_self_prompting,
        verifyprompt=verifyprompt, verbose=verbose,
        has_energy=False,
    )

    if not translate_ok:
        print(f"\n  {_R}Translation step failed — skipping energy measurement.{_E}")
        return False

    sycl_path = experiment_dir / "Main_translated.cpp"
    sycl_path.write_text(sycl_code, encoding="utf-8")
    print(f"\n  Translated SYCL code saved to: {sycl_path}")

    # ── Step 2: Energy measurement ────────────────────────────────────────────
    print(f"\n{'='*72}\n  {_B}BOTH TASK — Step 2 of 2: SYCL Energy Optimization{_E}\n{'='*72}")

    energy_runner   = _runner("energy_sycl", sycl_energy_compiler_env)
    energy_exp_name = experiment_name + "_sycl_energy"
    energy_exp_dir  = experiment_dir / "sycl_energy"
    energy_exp_dir.mkdir(parents=True, exist_ok=True)

    _llm.configure(log_dir=str(energy_exp_dir / "llm_calls"),
                   truncate_chars=4000, debug=verbose, stream_display=True)

    # Baseline: CUDA on Polaris, or initial SYCL translation on Aurora.
    if resource == "polaris":
        baseline_runner   = _runner("energy_cuda", {})
        baseline_src_path = original_cu_path
        baseline_ext      = ".cu"
    else:
        baseline_runner   = energy_runner
        baseline_src_path = sycl_path
        baseline_ext      = ".cpp"

    baseline_energy_j, baseline_power_w, baseline_stdout = run_baseline(
        session=session, runner=baseline_runner, remote_dir=remote_dir,
        original_src_path=baseline_src_path, src_extension=baseline_ext,
        experiment_name=energy_exp_name, experiment_dir=energy_exp_dir,
        run_args=run_args, run_id=run_id + "b",
        verification=verification,
    )

    energy_csv  = energy_exp_dir / "results_energy.csv"
    energy_data = {
        "timestamp":            timestamp,
        "app":                  "XSBench",
        "model":                model,
        "gpu_vendor":           resource_cfg.gpu_vendor,
        "success":              False,
        "runtime_min":          0.0,
        "baseline_energy_j":    baseline_energy_j,
        "optimized_energy_j":   -1.0,
        "energy_reduction_pct": 0.0,
    }
    with energy_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=list(energy_data.keys())).writeheader()

    energy_ok, _, _ = run_pipeline(
        model=model, task="sycl_energy", resource=resource,
        input_code=sycl_code,
        session=session, runner=energy_runner,
        remote_dir=remote_dir, src_extension=".cpp",
        run_args=run_args,
        experiment_name=energy_exp_name, experiment_dir=energy_exp_dir,
        csv_path=energy_csv, experiment_data=energy_data,
        trial_number=1, run_id=run_id,
        use_self_prompting=use_self_prompting,
        verifyprompt=verifyprompt, verbose=verbose,
        baseline_energy_j=baseline_energy_j,
        baseline_power_w=baseline_power_w,
        baseline_stdout=baseline_stdout,
        min_energy_gain_pct=min_energy_gain_pct,
        has_energy=True,
        use_judge=use_judge, verification=verification,
    )

    return translate_ok and energy_ok
