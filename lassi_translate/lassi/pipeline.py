"""Pipeline state machine: Stage 0 (self-prompt) → Stage 1 (initial) → Stage 2 (correct).

run_pipeline   is the per-task loop (translate OR energy)
run_baseline / run_test_run measure the original source before translation
write_metadata writes the per-experiment report on success/failure
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import lassi.llm_client as _llm
from lassi.format_colors    import B, G, R, E
from lassi.hpc_jobs         import JobRunner, JOB_DURATION_S, PersistentWorker, build_runner
from lassi.hpc_session      import HPCSession
from lassi.prompt_builder   import (build_correction_prompt, build_stage1_prompt,
                                    run_judge, run_selfprompt)
from lassi.prompt_dictionary import PromptDictionary
from lassi.resources         import ResourceConfig
from lassi.results           import (parse_attempt_results, parse_throughput,
                                    save_attempt_logs, tail5, xsbench_checksum_status)
from lassi.source_loader     import sync_file_to_remote

_DEFAULT_MIN_GAIN_PCT = 1.0
_MAX_ATTEMPTS         = 15


def _fmt_duration(s):
    if s < 60: return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60: return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _fmt_tail(stdout, n=8):
    lines = [l for l in (stdout or "").splitlines() if l.strip()]
    return "\n".join(f"      {l}" for l in lines[-n:]) if lines else "      (no output)"


def print_translate_summary(*, resource, original_framework, target_framework,
                             baseline, generated, total_elapsed_s,
                             versioned_path: Path, verification):
    """Side-by-side original vs generated metrics + stdout, printed on success."""
    rule = "─" * 72
    print(f"\n{rule}")
    print(f"  {B}{G}✅  TRANSLATION COMPLETE{E}  —  total time: "
          f"{B}{_fmt_duration(total_elapsed_s)}{E}")
    print(rule)

    def block(label, m):
        if m is None:
            return [f"  {B}{label}{E}", f"      (not measured — incompatible with {resource})"]
        out = [f"  {B}{label}{E}"]
        if m.get("duration_s", -1) > 0:
            out.append(f"      duration : {m['duration_s']:.2f} s")
        if verification == "xsbench":
            line, ok = xsbench_checksum_status(m.get("run_stdout", ""))
            mark = f"{G}✓{E}" if ok else f"{R}✗{E}"
            out.append(f"      checksum : {mark} {line or '(missing)'}")
        return out

    for ln in block(f"Original ({original_framework.upper()} on {resource})", baseline): print(ln)
    print()
    for ln in block(f"Generated ({target_framework.upper()} on {resource})", generated): print(ln)
    if baseline and baseline.get("run_stdout"):
        print(f"\n  {B}Original stdout (tail):{E}")
        print(_fmt_tail(baseline["run_stdout"]))
    if generated.get("run_stdout"):
        print(f"\n  {B}Generated stdout (tail):{E}")
        print(_fmt_tail(generated["run_stdout"]))
    print(f"\n  {B}Saved:{E} {versioned_path}")
    print(f"{rule}")
    print(f"  {B}{G}Job terminated.{E}\n")


def run_baseline(*, session: HPCSession, runner: JobRunner, remote_dir,
                 original_src_path: Path, src_extension, experiment_name,
                 experiment_dir: Path, run_args, run_id, verification="none"):
    """Compile + run the original source with energy monitoring (energy task).
    Returns (energy_joules, avg_power_w, run_stdout); -1.0/"" on failure."""
    print(f"\n{B}----- BASELINE -----{E}\n")
    name = f"Main_baseline{src_extension}"
    exp_remote = f"{remote_dir}/generated_codes/experiment/{experiment_name}"

    print(f"  Uploading {name}...", end="  ", flush=True)
    if not sync_file_to_remote(session, original_src_path, remote_dir, experiment_name, name):
        print(f"{R}FAILED{E}"); return -1.0, -1.0, ""
    print(f"{G}OK{E}")

    for pat in ("*.cuh", "*.h", "*.hpp"):
        for hdr in sorted(original_src_path.parent.glob(pat)):
            session.rsync(str(hdr), f"{exp_remote}/{hdr.name}")

    results_dir = f"{exp_remote}/baseline_results"
    session.mkdir(results_dir)
    try:
        runner.run(src_remote=f"{exp_remote}/{name}", results_dir=results_dir,
                   run_args=run_args, job_name=f"lassi-baseline-{run_id}")
    except Exception as exc:
        print(f"  {R}Baseline failed: {exc}{E}"); return -1.0, -1.0, ""

    results = parse_attempt_results(session, results_dir, has_energy=True)
    save_attempt_logs(results, experiment_dir, "baseline_")
    if not results["build_ok"]:
        print(f"  {R}Baseline compile FAILED:{E}")
        t = tail5(results.get("build_stderr") or "")
        if t: print(t)
        return -1.0, -1.0, ""
    if not results["run_ok"]:
        print(f"  {R}Baseline run FAILED:{E}")
        t = tail5((results.get("run_stderr") or "") + "\n" + (results.get("run_stdout") or ""))
        if t: print(t)
        return -1.0, -1.0, ""

    stdout = results.get("run_stdout") or ""
    if verification == "xsbench":
        line, ok = xsbench_checksum_status(stdout)
        if not ok: print(f"  {R}Baseline checksum invalid ({line!r}){E}")

    ej, apw = results.get("energy_joules", -1.0), results.get("avg_power_w", -1.0)
    ds, ns  = results.get("duration_s", -1.0),    results.get("num_samples", 0)
    if ej > 0:
        print(f"  {B}{G}Baseline: {ej:.1f} J ({apw:.1f} W, {ds:.1f} s, {ns} samples){E}")
    else:
        print(f"  {R}Energy unavailable — correctness only.{E}")
    (experiment_dir / "baseline_energy.json").write_text(
        json.dumps({"energy_joules": ej, "avg_power_w": apw,
                    "duration_s": ds, "num_samples": ns, "run_stdout": stdout}, indent=2))
    return ej, apw, stdout


def run_test_run(*, session, runner, remote_dir, original_src_path: Path, src_extension,
                 experiment_name, experiment_dir, run_args, run_id, verification="none"):
    """Compile + run the original source for the translate-task summary (no EE).
    Returns a metrics dict, or None on failure."""
    print(f"\n{B}----- TEST RUN -----{E}\n")
    name = f"Main_baseline{src_extension}"
    exp_remote = f"{remote_dir}/generated_codes/experiment/{experiment_name}"

    print(f"  Uploading {name}...", end="  ", flush=True)
    if not sync_file_to_remote(session, original_src_path, remote_dir, experiment_name, name):
        print(f"{R}FAILED{E}"); return None
    print(f"{G}OK{E}")

    for pat in ("*.cuh", "*.h", "*.hpp"):
        for hdr in sorted(original_src_path.parent.glob(pat)):
            session.rsync(str(hdr), f"{exp_remote}/{hdr.name}")

    results_dir = f"{exp_remote}/baseline_results"
    session.mkdir(results_dir)
    try:
        runner.run(src_remote=f"{exp_remote}/{name}", results_dir=results_dir,
                   run_args=run_args, job_name=f"lassi-testrun-{run_id}")
    except Exception as exc:
        print(f"  {R}Test run failed: {exc}{E}"); return None

    results = parse_attempt_results(session, results_dir, has_energy=True)
    save_attempt_logs(results, experiment_dir, "baseline_")
    if not results["build_ok"]:
        print(f"  Compiling...  {B}{R}✗  compile error:{E}")
        t = tail5(results.get("build_stderr") or "")
        if t: print(t)
        return None
    print(f"  Compiling...  {B}{G}✓  OK{E}")
    if not results["run_ok"]:
        print(f"  Running...    {B}{R}✗  runtime error:{E}")
        t = tail5((results.get("run_stderr") or "") + "\n" + (results.get("run_stdout") or ""))
        if t: print(t)
        return None
    stdout = results.get("run_stdout") or ""
    dur = results.get("duration_s", -1.0)
    print(f"  Running...    {B}{G}✓  OK{E}" + (f"  ({dur:.2f}s)" if dur > 0 else ""))
    if verification == "xsbench":
        line, ok = xsbench_checksum_status(stdout)
        mark = f"{G}✓{E}" if ok else f"{R}✗{E}"
        print(f"  Checksum...   {mark} {line or '(missing)'}")
    m = {"run_stdout": stdout, "duration_s": dur,
         "energy_joules": results.get("energy_joules", -1.0),
         "avg_power_w":   results.get("avg_power_w",   -1.0),
         "num_samples":   results.get("num_samples",    0)}
    (experiment_dir / "baseline_energy.json").write_text(json.dumps(m, indent=2))
    return m


def write_metadata(*, experiment_dir: Path, experiment_name, runtime_s, attempt_count,
                   energy_efficiency_iteration, has_energy, baseline_stdout, best_run_stdout,
                   baseline_energy_j, baseline_power_w, best_energy, best_power_w, best_dur_s,
                   energy_efficiency_success, input_code, best_code_block, model, system_prompt):
    """Write the per-experiment metadata_*.txt summary (mirrors base LASSI-EE format)."""
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    baseline_dur_s = -1.0
    bj = experiment_dir / "baseline_energy.json"
    if bj.exists():
        try: baseline_dur_s = float(json.loads(bj.read_text()).get("duration_s", -1.0))
        except Exception: pass

    parts = ["\n\n----- METRICS -----"]
    parts.append(f"\nExperiment date-time: {dt}")
    parts.append(f"\nTime (sec): {round(runtime_s, 2)}")
    parts.append(f"\nSelf-correction iteration count: {attempt_count}")
    if has_energy:
        parts.append(f"\nEnergy efficiency improvement count: {energy_efficiency_iteration}")
    if baseline_stdout:
        parts.append("\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n" + baseline_stdout)
    if best_run_stdout:
        parts.append("\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n" + best_run_stdout)
    if has_energy and baseline_energy_j > 0:
        parts.append("\n\n\n----- ENERGY EFFICIENCY METRICS -----\n")
        parts.append("Total Energy (J):")
        parts.append(f"  Source code: {round(baseline_energy_j, 2)}")
        if best_energy > 0: parts.append(f"  Refactored code: {round(best_energy, 2)}")
        parts.append("\nAverage Power (W):")
        parts.append(f"  Source code: {round(baseline_power_w, 2)}")
        if best_power_w > 0: parts.append(f"  Refactored code: {round(best_power_w, 2)}")
        parts.append("\nRuntime (s):")
        if baseline_dur_s > 0: parts.append(f"  Source code: {round(baseline_dur_s, 2)}")
        if best_dur_s > 0: parts.append(f"  Refactored code: {round(best_dur_s, 2)}")

    if energy_efficiency_success and best_code_block and baseline_energy_j > 0 and best_energy > 0:
        print(f"\n{B}----- BEGIN CODE COMPARISON -----{E}\n")
        prompts = PromptDictionary()
        cp = (prompts.get_code_comparison("CUDA_cpp")
              + "\n\n[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_code
              + "\n\nOUTPUT: \n\n" + baseline_stdout)
        if baseline_dur_s > 0: cp += f"\n\nCode execution duration: {round(baseline_dur_s, 2)} s"
        cp += (f"\nTotal Energy Consumed: {round(baseline_energy_j, 2)} J"
               f"\nAverage Power: {round(baseline_power_w, 2)} W"
               "\n\n[/END SOURCE CODE AND RESULTS]"
               "\n\n[BEGIN REFACTORED CODE AND RESULTS]\n\n" + best_code_block
               + "\n\nOUTPUT: \n\n" + best_run_stdout)
        if best_dur_s > 0: cp += f"\n\nCode execution duration: {round(best_dur_s, 2)} s"
        cp += (f"\nTotal Energy Consumed: {round(best_energy, 2)} J"
               f"\nAverage Power: {round(best_power_w, 2)} W"
               "\n\n[/END REFACTORED CODE AND RESULTS]")
        response = _llm.generate_text(model, system_prompt, cp,
                                      intent="code comparison", label="[LLM] Code comparison")
        parts.append("\n\n\n----- CODE COMPARISON -----\n" + response)

    p = experiment_dir / f"metadata_{experiment_name}.txt"
    p.write_text("".join(parts))
    print(f"\n  Metadata file: {p}")


def _correctness_note(verification):
    if verification == "xsbench":
        return ("CRITICAL: preserve the exact XSBench verification checksum — "
                "do NOT alter numerical computation, loop bounds, or comparison operators.")
    return "CRITICAL: preserve the exact behavior and output of the original source code."


def _record_outcome(*, csv_path: Path, experiment_data, success: bool,
                    start_time: float, extra_fields: dict | None = None):
    """Write one row to results.csv (called once per run — success OR failure)."""
    experiment_data["success"]     = success
    experiment_data["runtime_min"] = round((time.time() - start_time) / 60, 2)
    if extra_fields: experiment_data.update(extra_fields)
    with csv_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(experiment_data.keys())).writerow(experiment_data)
    return experiment_data["runtime_min"]


def _save_success(*, experiment_dir: Path, csv_path: Path, experiment_data, trial_number,
                  src_extension, saved_code, start_time, extra_fields=None):
    versioned = experiment_dir / f"Main_v1_Trial{trial_number}{src_extension}"
    versioned.write_text(saved_code, encoding="utf-8")
    runtime_min = _record_outcome(csv_path=csv_path, experiment_data=experiment_data,
                                   success=True, start_time=start_time,
                                   extra_fields=extra_fields)
    return versioned, runtime_min


def _write_attempts_log(experiment_dir: Path, attempt_log: list[dict]):
    """Dump per-attempt outcomes (compile/runtime/checksum/judge/success) as JSON."""
    (experiment_dir / "attempts.json").write_text(
        json.dumps(attempt_log, indent=2), encoding="utf-8")


def run_pipeline(*, model, task, resource, input_code, session, runner: JobRunner,
                 remote_dir, src_extension, run_args, experiment_name, experiment_dir,
                 csv_path, experiment_data, trial_number, run_id,
                 use_self_prompting=True, verifyprompt=False, verbose=False,
                 baseline_energy_j=-1.0, baseline_power_w=-1.0, baseline_stdout="",
                 min_energy_gain_pct=_DEFAULT_MIN_GAIN_PCT, has_energy=False,
                 use_judge=True, verification="none", measure_perf=False,
                 baseline_metrics: dict | None = None, ratchet_temperature=True):
    """Core loop: Stage 0 + 1, then up to _MAX_ATTEMPTS of Stage 2.  Each
    attempt: LLM generates code → upload → compile+run → parse → judge.
    Temperature ratchets up on stuck errors (cap 1.0 = terminate).
    Returns (success, code_block, final_metrics)."""
    prompts = PromptDictionary()
    start_time = time.time()

    guide_summary, source_description = run_selfprompt(
        model=model, prompts=prompts, input_code=input_code,
        task=task, resource=resource,
        verifyprompt=verifyprompt, use_self_prompting=use_self_prompting)

    system_prompt, stage1_user_prompt = build_stage1_prompt(
        prompts=prompts, task=task, resource=resource, input_code=input_code,
        guide_summary=guide_summary, source_description=source_description,
        baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w)

    session.mkdirs(
        f"{remote_dir}/generated_codes",
        f"{remote_dir}/generated_codes/experiment",
        f"{remote_dir}/generated_codes/experiment/{experiment_name}")

    energy_check_enabled = has_energy and baseline_energy_j > 0
    collect_energy = has_energy or measure_perf

    compile_err = run_err = run_stdout = code_block = ""
    last_metrics: dict = {}
    best_energy = -1.0
    best_code_block = best_run_stdout = ""
    best_power_w = best_dur_s = -1.0
    last_power_w = last_dur_s = -1.0
    code_gen_temp = 0.2
    energy_correction_cnt = 0
    prev_code_err_type, same_type_streak = None, 0
    energy_efficiency_iteration = 0
    energy_efficiency_success = False
    attempt = 0
    attempt_log: list[dict] = []   # per-attempt outcomes for attempts.json

    def _summary_fields(translation_stdout: str = "", translation_duration_s: float = -1.0):
        """Bundle the comparable baseline + translation perf into csv-row dict."""
        b_dur = baseline_metrics.get("duration_s", -1.0) if baseline_metrics else -1.0
        _, b_thru = parse_throughput(baseline_metrics.get("run_stdout", "")
                                      if baseline_metrics else "")
        _, t_thru = parse_throughput(translation_stdout)
        return {
            "attempts":               attempt + 1,
            "baseline_runtime_s":     round(b_dur, 3) if b_dur > 0 else -1.0,
            "baseline_throughput":    b_thru,
            "translation_runtime_s":  round(translation_duration_s, 3) if translation_duration_s > 0 else -1.0,
            "translation_throughput": t_thru,
        }

    for attempt in range(_MAX_ATTEMPTS + 1):
        if attempt == 0:
            label = {"translate": "CUDA→SYCL TRANSLATION",
                     "energy":    "CUDA ENERGY OPTIMIZATION"}.get(task, "CODE GENERATION")
            print(f"\n{B}----- STAGE 1: {label} -----{E}\n")
            code_block = _llm.generate_code(model, system_prompt, stage1_user_prompt,
                verifyprompt=verifyprompt, label="[LLM] Generating initial code", temperature=0.2)
        else:
            if attempt == 1: print(f"\n{B}----- STAGE 2: SELF-CORRECTION -----{E}\n")
            err_type, error_prompt = build_correction_prompt(
                prompts=prompts, task=task, resource=resource, code_block=code_block,
                compile_err=compile_err, run_err=run_err, run_stdout=run_stdout,
                input_code=input_code, attempt=attempt,
                guide_summary=guide_summary, source_description=source_description,
                best_code_block=best_code_block, opt_power_w=last_power_w, opt_dur_s=last_dur_s)
            print(f"  {B}Attempt {attempt}/{_MAX_ATTEMPTS}{E} — correcting {err_type}")

            # Temperature ratchet: energy-insufficient bumps every attempt;
            # code errors bump only after 3 consecutive of the same type.
            # Disabled via --no-temp-ratchet → always use the initial temp.
            if not ratchet_temperature:
                call_temp = code_gen_temp
            elif err_type == "energy improvement insufficient":
                energy_correction_cnt += 1
                call_temp, code_gen_temp = code_gen_temp, round(min(code_gen_temp + 0.2, 1.0), 1)
                if call_temp >= 1.0:
                    print(f"  {B}Max temp reached.{E}"); break
            elif err_type == "energy keep improving":
                call_temp = code_gen_temp
            else:
                same_type_streak = same_type_streak + 1 if err_type == prev_code_err_type else 1
                prev_code_err_type = err_type
                if same_type_streak >= 3:
                    call_temp, code_gen_temp = code_gen_temp, round(min(code_gen_temp + 0.2, 1.0), 1)
                    if call_temp >= 1.0:
                        print(f"  {B}Max temp reached.{E}"); break
                    print(f"  {B}Stuck on {err_type} ×{same_type_streak} — "
                          f"bumping temp to {code_gen_temp}.{E}")
                else:
                    call_temp = code_gen_temp

            code_block = _llm.generate_code(model, system_prompt, error_prompt,
                verifyprompt=verifyprompt, label=f"[LLM] Corrected code (attempt {attempt})",
                temperature=call_temp)

        src_name = f"Main_attempt{attempt}{src_extension}"
        local_src = experiment_dir / src_name
        local_src.write_text(code_block, encoding="utf-8")

        print(f"  Uploading {src_name}...", end="  ", flush=True)
        if not sync_file_to_remote(session, local_src, remote_dir, experiment_name, src_name):
            print(f"{R}FAILED{E}")
            compile_err, run_stdout = f"rsync failed — {src_name}", ""
            continue
        print(f"{G}OK{E}")

        src_remote = f"{remote_dir}/generated_codes/experiment/{experiment_name}/{src_name}"
        results_dir = f"{remote_dir}/generated_codes/experiment/{experiment_name}/attempt{attempt}_results"
        session.mkdir(results_dir)

        try:
            runner.run(src_remote=src_remote, results_dir=results_dir,
                       run_args=run_args, job_name=f"lassi-{attempt}-{run_id}", verbose=verbose)
        except RuntimeError as exc:
            compile_err, run_stdout = str(exc), ""; break
        except Exception as exc:
            compile_err, run_stdout = str(exc), ""; continue

        results = parse_attempt_results(session, results_dir, collect_energy)
        save_attempt_logs(results, experiment_dir, f"attempt{attempt}_")

        if not results["build_ok"]:
            t = tail5(results["build_stderr"])
            print(f"  Compiling...  {B}{R}✗  compile error:{E}")
            if t: print(t)
            compile_err = results["build_stderr"] or "Compilation failed"
            run_stdout = ""
            attempt_log.append({"attempt": attempt, "outcome": "compile_error",
                                 "summary": (results["build_stderr"] or "")[:300]})
            continue
        print(f"  Compiling...  {B}{G}✓  OK{E}")

        run_stdout = results["run_stdout"] or ""
        if not results["run_ok"]:
            t = tail5((results["run_stderr"] or "") + "\n" + run_stdout)
            print(f"  Running...    {B}{R}✗  runtime error:{E}")
            if t: print(t)
            compile_err, run_err = "execute_error", results["run_stderr"] or "non-zero exit"
            attempt_log.append({"attempt": attempt, "outcome": "runtime_error",
                                 "summary": (results["run_stderr"] or "non-zero exit")[:300]})
            continue
        print(f"  Running...    {B}{G}✓  OK{E}")

        if verification == "xsbench":
            cs_line, cs_valid = xsbench_checksum_status(run_stdout)
            if not cs_valid:
                print(f"  Checksum...   {B}{R}✗  {cs_line or 'missing'}{E}")
                snippet = (run_stdout[:1000].strip() or "(no output)") + (
                    "\n...(truncated)" if len(run_stdout) > 1000 else "")
                compile_err = (
                    "VERIFICATION MISMATCH: Checksum missing or invalid.\n"
                    f"Checksum line: {cs_line!r}\nActual stdout:\n{snippet}\n"
                    "Your code MUST faithfully implement ALL functions from the original source, "
                    "including print_results(), run_event_based_simulation_baseline(), and "
                    "grid_init_do_not_profile(). Do NOT generate a simplified stub.\n"
                    f"\n[ORIGINAL SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL SOURCE]\n")
                attempt_log.append({"attempt": attempt, "outcome": "checksum_mismatch",
                                     "summary": cs_line or "(missing)"})
                continue
            print(f"  Checksum...   {B}{G}✓  {cs_line}{E}")

        if use_judge and baseline_stdout:
            src_m = (f"Energy: {baseline_energy_j:.1f} J, Avg Power: {baseline_power_w:.1f} W"
                     if baseline_energy_j > 0 else "(not measured)")
            opt_ej, opt_pw = results.get("energy_joules", -1.0), results.get("avg_power_w", -1.0)
            ref_m = (f"Energy: {opt_ej:.1f} J, Avg Power: {opt_pw:.1f} W"
                     if opt_ej > 0 else "(not measured)")
            verdict = run_judge(model=model, system_prompt=system_prompt,
                source_code=input_code, source_stdout=baseline_stdout, source_metrics=src_m,
                refactored_code=code_block, refactored_stdout=run_stdout, refactored_metrics=ref_m)
            if verdict != "VALID":
                reason = ("not functionally equivalent" if verdict == "INVALID"
                          else "judge couldn't determine equivalence")
                compile_err = (f"VERIFICATION MISMATCH (LLM judge): {reason}. "
                               "Ensure ALL original logic, prints, and computation are preserved.")
                run_stdout = ""
                attempt_log.append({"attempt": attempt, "outcome": "judge_rejected",
                                     "summary": f"verdict={verdict}: {reason}"})
                continue

        if has_energy:
            opt_ej = results.get("energy_joules", -1.0)
            opt_pw = results.get("avg_power_w",   -1.0)
            opt_ds = results.get("duration_s",    -1.0)
            opt_ns = results.get("num_samples",    0)
            if opt_ej > 0:
                print(f"  Energy...     {B}{opt_ej:.1f} J{E}  "
                      f"({opt_pw:.1f} W, {opt_ds:.1f} s, {opt_ns} samples)")
                last_power_w, last_dur_s = opt_pw, opt_ds
            else:
                print(f"  Energy...     {B}(unavailable){E}")

            if energy_check_enabled and opt_ej > 0:
                reduction = (baseline_energy_j - opt_ej) / baseline_energy_j * 100.0
                compare = best_energy if best_energy > 0 else baseline_energy_j
                if opt_ej < compare:
                    prev_best = best_energy
                    best_energy, best_code_block, best_run_stdout = opt_ej, code_block, run_stdout
                    best_power_w, best_dur_s = opt_pw, opt_ds
                    energy_efficiency_success, energy_correction_cnt = True, 0
                    code_gen_temp = 0.2
                    prev_code_err_type, same_type_streak = None, 0
                    energy_efficiency_iteration += 1
                    print(f"  Reduction...  {B}{G}↓  {reduction:.1f}%  "
                          f"({baseline_energy_j:.1f} → {opt_ej:.1f} J)  "
                          f"[best, iter {energy_efficiency_iteration}]{E}")
                    if prev_best > 0 and energy_efficiency_iteration > 4:
                        marginal = (prev_best - opt_ej) / prev_best
                        if marginal < 0.011 and opt_ej < baseline_energy_j:
                            print(f"  {B}Stagnation ({marginal*100:.2f}%) — stopping.{E}")
                            break
                    from_j = prev_best if prev_best > 0 else baseline_energy_j
                    compile_err = (
                        f"ENERGY_KEEP_IMPROVING: Code improved from {from_j:.1f} J to "
                        f"{opt_ej:.1f} J ({reduction:+.1f}% vs baseline {baseline_energy_j:.1f} J). "
                        f"Apply more aggressive optimizations.\n{_correctness_note(verification)}")
                    continue
                else:
                    ref = f"best {best_energy:.1f} J" if best_energy > 0 else f"baseline {baseline_energy_j:.1f} J"
                    print(f"  Reduction...  {B}{R}✗  {reduction:.1f}%  (vs {ref}){E}")
                    compile_err = (
                        f"ENERGY_NOT_IMPROVED: Current code uses {opt_ej:.1f} J vs baseline "
                        f"{baseline_energy_j:.1f} J ({reduction:+.1f}%, need ≥{min_energy_gain_pct:.1f}% reduction).\n"
                        f"Apply more aggressive optimizations.\n{_correctness_note(verification)}")
                    continue
            elif energy_check_enabled and opt_ej <= 0:
                if not compile_err:
                    compile_err = "ENERGY_NOT_IMPROVED: Measurement unavailable. Retry."
                continue

        # Success path (translate, or energy without baseline)
        if has_energy and energy_check_enabled:
            pass
        else:
            task_label = "translate" if task == "translate" else "energy"
            t_dur = results.get("duration_s", -1.0)
            last_metrics = {"run_stdout": run_stdout,
                            "energy_joules": results.get("energy_joules", -1.0),
                            "avg_power_w":   results.get("avg_power_w",   -1.0),
                            "duration_s":    t_dur,
                            "num_samples":   results.get("num_samples",    0),
                            "attempt_count": attempt}
            attempt_log.append({"attempt": attempt, "outcome": "success",
                                 "summary": f"runtime {t_dur:.2f}s" if t_dur > 0 else "ok"})
            _write_attempts_log(experiment_dir, attempt_log)
            versioned, runtime_min = _save_success(
                experiment_dir=experiment_dir, csv_path=csv_path,
                experiment_data=experiment_data, trial_number=trial_number,
                src_extension=src_extension, saved_code=code_block, start_time=start_time,
                extra_fields=_summary_fields(translation_stdout=run_stdout,
                                              translation_duration_s=t_dur))
            attempts_label = f"attempt {attempt + 1}" if attempt > 0 else "first attempt"
            print(f"\n  {B}{G}✅  {task_label.capitalize()} complete{E}  —  "
                  f"{attempts_label}  |  {runtime_min} min total\n")
            print(f"  Successful code : {versioned}")
            print(f"  Remote results  : {results_dir}")
            write_metadata(experiment_dir=experiment_dir, experiment_name=experiment_name,
                runtime_s=time.time() - start_time, attempt_count=attempt,
                energy_efficiency_iteration=energy_efficiency_iteration, has_energy=has_energy,
                baseline_stdout=baseline_stdout, best_run_stdout=run_stdout,
                baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
                best_energy=-1.0, best_power_w=-1.0, best_dur_s=-1.0,
                energy_efficiency_success=False, input_code=input_code,
                best_code_block=code_block, model=model, system_prompt=system_prompt)
            return True, code_block, last_metrics

    # Post-loop verdict
    if has_energy and energy_check_enabled and best_energy > 0:
        red = (baseline_energy_j - best_energy) / baseline_energy_j * 100.0
        if red >= min_energy_gain_pct:
            attempt_log.append({"attempt": attempt, "outcome": "success_post_loop",
                                 "summary": f"best energy {best_energy:.1f} J ({red:+.1f}%)"})
            _write_attempts_log(experiment_dir, attempt_log)
            extras = _summary_fields(translation_stdout=best_run_stdout,
                                      translation_duration_s=best_dur_s)
            extras.update({"baseline_energy_j": baseline_energy_j,
                           "optimized_energy_j": best_energy,
                           "energy_reduction_pct": round(red, 2)})
            versioned, runtime_min = _save_success(
                experiment_dir=experiment_dir, csv_path=csv_path,
                experiment_data=experiment_data, trial_number=trial_number,
                src_extension=src_extension, saved_code=best_code_block, start_time=start_time,
                extra_fields=extras)
            print(f"\n  {B}{G}✅  Energy complete{E}  —  {runtime_min} min total\n"
                  f"  Energy: {baseline_energy_j:.1f} J → {best_energy:.1f} J  "
                  f"({red:+.1f}%)  [{energy_efficiency_iteration} improvement(s)]\n")
            print(f"  Best code: {versioned}")
            write_metadata(experiment_dir=experiment_dir, experiment_name=experiment_name,
                runtime_s=time.time() - start_time, attempt_count=attempt,
                energy_efficiency_iteration=energy_efficiency_iteration, has_energy=has_energy,
                baseline_stdout=baseline_stdout, best_run_stdout=best_run_stdout,
                baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
                best_energy=best_energy, best_power_w=best_power_w, best_dur_s=best_dur_s,
                energy_efficiency_success=energy_efficiency_success,
                input_code=input_code, best_code_block=best_code_block,
                model=model, system_prompt=system_prompt)
            last_metrics = {"run_stdout": best_run_stdout, "energy_joules": best_energy,
                            "avg_power_w": best_power_w, "duration_s": best_dur_s,
                            "num_samples": 0, "attempt_count": attempt}
            return True, best_code_block, last_metrics

    print(f"\n{B}{R}❌  PIPELINE: no sufficient energy improvement "
          f"(best: {best_energy:.1f} J, baseline: {baseline_energy_j:.1f} J).{E}\n"
          if has_energy and energy_check_enabled else
          f"\n{B}{R}❌  PIPELINE ERROR: exceeded {_MAX_ATTEMPTS} self-correction attempts.{E}\n")

    # Record the failure: attempts.json + a success=False row in results.csv
    _write_attempts_log(experiment_dir, attempt_log)
    _record_outcome(csv_path=csv_path, experiment_data=experiment_data,
                    success=False, start_time=start_time,
                    extra_fields=_summary_fields(translation_stdout=best_run_stdout or run_stdout,
                                                  translation_duration_s=best_dur_s))

    write_metadata(experiment_dir=experiment_dir, experiment_name=experiment_name,
        runtime_s=time.time() - start_time, attempt_count=attempt,
        energy_efficiency_iteration=energy_efficiency_iteration, has_energy=has_energy,
        baseline_stdout=baseline_stdout, best_run_stdout=best_run_stdout,
        baseline_energy_j=baseline_energy_j, baseline_power_w=baseline_power_w,
        best_energy=best_energy, best_power_w=best_power_w, best_dur_s=best_dur_s,
        energy_efficiency_success=False, input_code=input_code,
        best_code_block=best_code_block, model=model, system_prompt=system_prompt)
    return False, (best_code_block if best_code_block else code_block), last_metrics


