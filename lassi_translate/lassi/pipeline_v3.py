"""Translation pipeline (CUDA→SYCL).

Workflow:
- Stage 0: build+run the source CUDA app (captures stdout).
- Stage 1+: generate SYCL code via Argo, then compile+run.
  If compile or run fails, re-prompt with the error and retry.
  If run succeeds but output verification fails (e.g., checksum mismatch),
  re-prompt with a "verification mismatch" error and retry.

This bundle intentionally omits all energy/power logic and all non-Argo LLM backends.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import lassi.code_manager
import lassi.llm_methods
import lassi.pipeline_methods
from lassi.format_colors import color
from lassi.gpu_detector import detect_gpu_vendors


def _write_sycl_makefile(*, experiment_dir: str, sycl_compiler: str, sycl_flags: str, src: str, out: str) -> None:
    """Write a Makefile into `experiment_dir` for building a single SYCL file."""

    template = os.path.join(os.path.dirname(__file__), "sycl_makefile_hardcoded.mk")
    dst = os.path.join(experiment_dir, "Makefile")

    with open(template, "r", encoding="utf-8") as f:
        mk = f.read()

    # Convenience: allow passing "acpp" and resolve to the default install path.
    if sycl_compiler in {"acpp", "<SYCL_COMPILER>"}:
        acpp_default = "/home/vchirio/adaptivecpp/bin/acpp"
        if os.path.exists(acpp_default):
            sycl_compiler = acpp_default

    mk = mk.replace("<SYCL_COMPILER>", sycl_compiler)
    mk = mk.replace("<SYCL_FLAGS>", sycl_flags)
    mk = mk.replace("<SYCL_LDFLAGS>", "")
    mk = mk.replace("<SYCL_SRC>", src)
    mk = mk.replace("<SYCL_OUT>", out)

    with open(dst, "w", encoding="utf-8") as f:
        f.write(mk)


def _normalize_stdout_for_compare(s: str) -> str:
    """Normalize stdout for comparison.

    Keep this conservative: strip trailing whitespace and normalize newlines.
    """

    return "\n".join(line.rstrip() for line in (s or "").splitlines()).strip() + "\n"


def _extract_xsbench_checksum(stdout_text: str) -> str:
    """Extract the XSBench verification checksum line.

    Example:
      Verification checksum: 711949 (WARNING - INAVALID CHECKSUM!)

    Returns "" if not found.
    """

    for line in (stdout_text or "").splitlines():
        if "Verification checksum:" in line:
            return line.strip()
    return ""


def _xsbench_checksum_valid(stdout_text: str) -> bool:
    """Return True if XSBench reports a passing checksum (no INVALID in checksum line)."""
    line = _extract_xsbench_checksum(stdout_text)
    if not line:
        return False
    return "INVALID" not in line.upper()


def _stdout_matches(app_name: str, source_stdout: str, generated_stdout: str) -> bool:
    """Return True if the user-defined verification output matches.

    If source_stdout is empty (Stage 0 was skipped, e.g. on Aurora where CUDA
    source cannot be executed), XSBench is still verified for a valid (warning-free)
    checksum line; all other apps skip verification and return True.

    Default behavior: compare full normalized stdout.

    App-specific behavior:
    - XSBench: if a reference stdout is available, compare checksum lines.
               If no reference (Stage 0 skipped), verify the generated output
               has a passing checksum (no WARNING - INAVALID CHECKSUM! message).
    """

    if app_name == "XSBench":
        if not (source_stdout or "").strip():
            # No CUDA reference - verify generated checksum is clean (no WARNING).
            valid = _xsbench_checksum_valid(generated_stdout)
            checksum_line = _extract_xsbench_checksum(generated_stdout)
            if valid:
                print(f"[VERIFY] XSBench checksum OK (no reference run): {checksum_line}")
            else:
                print(f"[VERIFY] XSBench checksum FAILED (invalid or missing): {checksum_line!r}")
            return valid
        src_key = _extract_xsbench_checksum(source_stdout)
        gen_key = _extract_xsbench_checksum(generated_stdout)
        match = src_key == gen_key
        print(f"[VERIFY] XSBench checksum: source={src_key!r}  generated={gen_key!r}  match={match}")
        return match

    if not (source_stdout or "").strip():
        print("[VERIFY] No source reference stdout available - skipping stdout verification.")
        return True

    return _normalize_stdout_for_compare(source_stdout) == _normalize_stdout_for_compare(generated_stdout)


def auto_code_llmgeneration_pipeline(
    prompts,
    this_llm_systemcode_prompt,
    this_llm_prompt,
    llm_source,
    model,
    model_coder,
    model_judge,
    num_ctx,
    num_ctx_coder,
    use_context,
    use_context_summaries_only,
    few_shot_learning,
    run_code_analysis,
    input_context,
    input_code_file_subdirectory,
    input_code,
    lines_of_code,
    target_lang,
    refactored_code_extension,
    source_lang,
    code_compiler,
    code_compiler_kwds,
    build_script,
    make_script,
    code_build_script,
    directory,
    file_name,
    source_file_path,
    display_response,
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
    experiment_data_directory,
    experiment_data_file_object,
    experiment_data,
    app_name,
):
    """Run the translation-only pipeline.

    The signature is kept compatible with the original project.
    """

    if llm_source not in ("argo", "alcf"):
        raise ValueError(f"Unsupported llm_source: {llm_source!r}. Use 'argo' or 'alcf'")

    # Minimal experiment bookkeeping.
    now = datetime.now(ZoneInfo("America/Chicago"))
    experiment_data["trial_number"] = trial_number
    experiment_data["timestamp"] = now.strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_data["experiment"] = experiment
    experiment_data["app_name"] = file_name
    experiment_data["code_line_count"] = lines_of_code
    experiment_data["llm"] = model
    experiment_data["llm_coder"] = model_coder
    experiment_data["llm_judge"] = model_judge

    print("\n----- PIPELINE v3 (translation-only) -----\n")
    print(f"[INFO] Writing all artifacts under: {experiment_data_directory}")
    print(f"[INFO] Translating {source_lang} → {target_lang} for app: {app_name}")

    vendors = detect_gpu_vendors()
    if not vendors:
        print("❌ PIPELINE ERROR: No GPUs detected. Stopping.")
        return False
    print(f"🔍 Detected GPU vendor: {vendors[0]['vendor'].upper()}")

    # Stage 0: build+run source.
    input_content = {
        "source_code": input_code,
        "source_stdout": "",
        "source_metrics": "",
        "refactored_code_lang": target_lang,
        "refactored_code": "",
        "refactored_stdout": "",
        "refactored_metrics": "",
    }

    energy_history: list[float] = []
    power_history: list[float] = []
    runtime_history: list[float] = []
    valid_code_history: list[str] = []

    if test_code_first:
        print("\n----- STAGE 0: BUILD+RUN SOURCE -----\n")
        print("[STAGE0] Building and running the original (source) program to capture its stdout for later verification.")
        print(f"[STAGE0] Source file: {source_file_path}")
        stage0_make_script = make_script
        stage0_build_script = build_script
        stage0_compiler = code_compiler
        stage0_flags = code_compiler_kwds

        # For CUDA miniapps, Stage 0 must use the app's Makefile (nvcc).
        if source_lang == "CUDA":
            stage0_make_script = True
            stage0_build_script = False
            stage0_compiler = "make"
            stage0_flags = "-j SM_VERSION=80 CFLAGS=-std=c++17"

        print(f"[STAGE0] Build tool: {stage0_compiler} ({'Makefile build' if stage0_make_script else 'single-file compile'})")
        print(f"[STAGE0] Build flags: {stage0_flags}")

        test_result = lassi.pipeline_methods.test_source_code(
            app_name,
            experiment_data_directory,
            0,
            trial_number,
            source_file_path,
            stage0_build_script,
            stage0_make_script,
            directory,
            source_lang,
            target_lang,
            exe_input_parameters,
            stage0_compiler,
            stage0_flags,
            yes_execute,
            False,
            input_content,
            energy_history,
            power_history,
            runtime_history,
            valid_code_history,
        )

        if test_result is False:
            print(color.BOLD + color.RED + "\n❌ PIPELINE ERROR: Stage 0 failed.\n" + color.END)
            return False

        # Unpack updated dict.
        (_ie, _ip, input_content, _se, _sp, _dur, *_rest) = test_result

    # Stage 1+: translate + self-correct.
    start_time = time.time()
    max_attempts = 15

    compile_err = ""
    run_err = ""
    run_stdout = ""
    code_block = ""

    translate_prompt = (
        this_llm_prompt
        + "\n\n[START OF CODE TO REFACTOR]\n"
        + input_code
        + "\n[/END OF CODE TO REFACTOR]"
    )

    for attempt in range(max_attempts + 1):
        if attempt == 0:
            print(f"\n----- STAGE 1: CUDA→SYCL TRANSLATION ({llm_source.upper()}) -----\n")
            print(f"  Prompts : system=[CUDA_to_SYCL]  |  user=[CUDA_to_SYCL]")
            print(f"  Context : source code")
            with lassi.llm_methods.Spinner(f"[LLM] Generating initial SYCL translation"):
                _raw, code_block = lassi.llm_methods.code_generator_llm(
                    llm_source,
                    model_coder,
                    num_ctx_coder,
                    0.2,
                    this_llm_systemcode_prompt,
                    translate_prompt,
                    False,  # never print full response to console
                    verifyprompt,
                )
        else:
            print(f"\n----- STAGE 2: SELF-CORRECTION (attempt {attempt}/{max_attempts}) -----\n")
            if compile_err == "execute_error":
                error_intro = "\n -- Runtime error: "
                error_message = run_err
                stdout_ctx = " + stdout" if run_stdout.strip() else ""
                if run_stdout.strip():
                    error_message += "\n\nProgram stdout before crash:\n" + run_stdout
                error_message += (
                    "\n\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n"
                    + input_code
                    + "\n[/ORIGINAL CUDA SOURCE]\n"
                )
                print(f"  Prompts : system=[CUDA_to_SYCL]  |  user=[CUDA_to_SYCL_error]")
                print(f"  Context : generated code + runtime error{stdout_ctx} + CUDA source")
            elif compile_err.startswith("VERIFICATION MISMATCH"):
                error_intro = "\n -- Verification mismatch: "
                error_message = compile_err
                print(f"  Prompts : system=[CUDA_to_SYCL]  |  user=[CUDA_to_SYCL_error]")
                print(f"  Context : generated code + verification mismatch + CUDA source")
            else:
                error_intro = "\n -- Compile error: "
                error_message = compile_err
                print(f"  Prompts : system=[CUDA_to_SYCL]  |  user=[CUDA_to_SYCL_error]")
                print(f"  Context : generated code + compile error")

            error_prompt = (
                prompts.get_code_error_correction_prompt("CUDA_to_SYCL_error")
                + "\n\n[START OF CODE TO FIX]\n"
                + code_block
                + "\n[/END OF CODE TO FIX]"
                + error_intro
                + "\n"
                + error_message
            )

            with lassi.llm_methods.Spinner(f"[LLM] Generating corrected code (attempt {attempt}/{max_attempts})"):
                _raw, code_block = lassi.llm_methods.code_generator_llm(
                    llm_source,
                    model_coder,
                    num_ctx_coder,
                    0.2,
                    this_llm_systemcode_prompt,
                    error_prompt,
                    False,  # never print full response to console
                    verifyprompt,
                )

        # Show animated preview of the generated code.
        lassi.llm_methods.show_code_preview(
            code_block,
            title=f"Stage {'1' if attempt == 0 else '2'} generated code — attempt {attempt + 1}",
        )

        # Write generated code.
        out_stem = input_code_file_subdirectory + file_name
        out_src = os.path.join(os.getcwd(), out_stem + "." + refactored_code_extension)
        os.makedirs(os.path.dirname(out_src), exist_ok=True)
        with open(out_src, "w", encoding="utf-8") as f:
            f.write(code_block)
        print(f"[GEN] Wrote generated {target_lang} code to: {out_src}")

        # Compile.
        if make_script:
            if target_lang != "SYCL":
                raise ValueError("translation-only pipeline expects target_lang='SYCL' when make_script=True")

            _write_sycl_makefile(
                experiment_dir=experiment_data_directory,
                sycl_compiler=code_compiler,
                sycl_flags=code_compiler_kwds,
                src=out_src,
                out=out_stem,
            )
            print(f"[BUILD] Wrote a Makefile for building the generated code to: {os.path.join(experiment_data_directory, 'Makefile')}")

            compile_err = lassi.code_manager.make_script_code(
                ".",
                f"-C {experiment_data_directory}",
                "SYCL",
                experiment_dir=experiment_data_directory,
                log_prefix=f"sycl_build_attempt{attempt}_",
            )
            exe_path = os.path.join(experiment_data_directory, out_stem)
        else:
            compile_err, exe_path = lassi.code_manager.compile_code(
                out_src,
                target_lang,
                code_compiler,
                code_compiler_kwds,
                experiment_dir=experiment_data_directory,
                log_prefix=f"sycl_compile_attempt{attempt}_",
            )

        if compile_err:
            continue

        # Run.
        if yes_execute:
            run_result, run_stdout, *_rest = lassi.code_manager.execute_code_selectvendor(
                experiment_data_directory,
                1,
                True,
                trial_number,
                directory,
                exe_path,
                exe_input_parameters,
                False,
                "generated",
                target_lang,
                build_script,
                make_script,
                gpu_vendor="auto",
                wait=5,
            )

            if run_result != "0":
                compile_err = "execute_error"
                run_err = run_result
                continue

            # Verification: compare stdout to source stdout.
            if not _stdout_matches(app_name, input_content["source_stdout"], run_stdout):
                if app_name == "XSBench":
                    src_key = _extract_xsbench_checksum(input_content["source_stdout"])
                    gen_key = _extract_xsbench_checksum(run_stdout)
                    stdout_snippet = (run_stdout[:1000].strip() or "(program produced no output)") + ("\n...(truncated)" if len(run_stdout) > 1000 else "")
                    if src_key:
                        detail = (
                            f"XSBench checksum line mismatch\n"
                            f"SOURCE:    {src_key}\n"
                            f"GENERATED: {gen_key}\n"
                            f"\nYour program's actual stdout:\n{stdout_snippet}\n"
                        )
                    else:
                        detail = (
                            f"XSBench checksum missing or invalid.\n"
                            f"The program compiled and ran, but did NOT print the required "
                            f"'Verification checksum: N' line.\n"
                            f"\nYour program's actual stdout:\n{stdout_snippet}\n"
                            f"\nThe print_results() function in the original source is responsible "
                            f"for printing:\n"
                            f'  printf("Verification checksum: %llu (WARNING - INAVALID CHECKSUM!)\\n", vhash);\n'
                            f"or\n"
                            f'  printf("Verification checksum: %llu (Valid)\\n", vhash);\n'
                            f"\nYour SYCL translation MUST faithfully implement ALL functions from "
                            f"the original CUDA source, including print_results(), "
                            f"run_event_based_simulation_baseline(), and grid_init_do_not_profile().\n"
                            f"Do NOT generate a simplified stub — translate the complete source.\n"
                            f"\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n"
                            f"{input_code}\n"
                            f"[/ORIGINAL CUDA SOURCE]\n"
                        )
                else:
                    detail = (
                        "--- SOURCE STDOUT ---\n"
                        + input_content["source_stdout"]
                        + "\n--- GENERATED STDOUT ---\n"
                        + run_stdout
                    )

                compile_err = "VERIFICATION MISMATCH: generated output does not match the verification criteria.\n\n" + detail
                continue

            # Success: persist a versioned copy and stop.
            versioned = os.path.join(
                experiment_data_directory,
                f"{out_stem}_v1-baseLLM_Trial{trial_number}.{refactored_code_extension}",
            )
            with open(versioned, "w", encoding="utf-8") as f:
                f.write(code_block)

            input_content["refactored_code"] = code_block
            input_content["refactored_stdout"] = run_stdout

            meta_data_file += "\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n" + input_content["source_stdout"]
            meta_data_file += "\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n" + input_content["refactored_stdout"]

            meta_path = os.path.join(
                experiment_data_directory,
                "metadata" + "_" + experiment.replace("/", "_") + "_Trial" + str(trial_number) + ".txt",
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(meta_data_file)
            print("\nMetadata file: " + str(meta_path))

            # Append row to CSV.
            with open(experiment_data_file_object, "a", newline="") as exp_file:
                writer = csv.DictWriter(exp_file, fieldnames=list(experiment_data.keys()))
                writer.writerow(experiment_data)

            runtime_min = round((time.time() - start_time) / 60, 2)
            print("\n✅ Translation complete (compile+run+verify OK). Pipeline runtime (min):", runtime_min)
            return True

        return True

    print(color.BOLD + color.RED + "\n❌ PIPELINE ERROR: exceeded self-correction attempts.\n" + color.END)
    return False
