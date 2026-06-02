"""Minimal LASSI v3 translation pipeline (CUDA→SYCL).

This is a **translation-only** subset of the original LASSI-EE v3 pipeline.

Behavior:
- Stage 0: build+run the original CUDA app (via its Makefile).
- Stage 1+: translate CUDA→SYCL using Argo LLM calls with compile/execute self-correction.
- **Stop immediately** once generated SYCL code compiles and executes successfully.

Notes:
- This module intentionally omits all energy-efficiency / power-measurement logic.
- This module intentionally omits LLM-as-a-judge and code-comparison steps.
- The public entrypoint remains [`auto_code_llmgeneration_pipeline()`](lassi/pipeline_v3.py:24).
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
from lassi.get_power import report_gpu_memory


def auto_code_llmgeneration_pipeline(
    prompts,
    this_llm_systemcode_prompt,
    this_llm_prompt,  # translation instruction prompt
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
    input_code,  # source code
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

    The signature is kept compatible with the original pipeline so the runner can
    call it without special-casing.
    """

    if llm_source != "argo":
        raise ValueError("This standalone translation pipeline supports llm_source='argo' only")

    # -----------------
    # ----- SETUP -----
    # -----------------
    now = datetime.now(ZoneInfo("America/Chicago"))
    dt_string_data = now.strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_data["trial_number"] = trial_number
    experiment_data["timestamp"] = dt_string_data
    experiment_data["experiment"] = experiment
    experiment_data["app_name"] = file_name
    experiment_data["code_line_count"] = lines_of_code
    experiment_data["llm"] = model
    experiment_data["llm_coder"] = model_coder
    experiment_data["llm_judge"] = model_judge

    code_version = 0
    self_correction_max = 10

    input_content_llmjudge_dict = {
        "source_code": input_code,
        "source_stdout": "",
        "source_metrics": "",
        "refactored_code_lang": target_lang,
        "refactored_code": "",
        "refactored_stdout": "",
        "refactored_metrics": "",
    }

    # Required by `test_source_code()` signature.
    energy_history: list[float] = []
    power_history: list[float] = []
    runtime_history: list[float] = []
    valid_code_history: list[str] = []

    print("\n----- WELCOME TO PIPELINE v3 (translation-only) -----\n")

    vendors = detect_gpu_vendors()
    if not vendors:
        print("❌ PIPELINE ERROR: No GPUs detected. Stopping LASSI.")
        return False

    detected_vendor = vendors[0]["vendor"]
    print(f"🔍 Detected GPU vendor: {detected_vendor.upper()}")
    report_gpu_memory()

    # --------------------------------
    # ----- STAGE 0: TEST SOURCE -----
    # --------------------------------
    if test_code_first:
        stage0_make_script = make_script
        stage0_build_script = build_script
        stage0_code_compiler = code_compiler
        stage0_code_compiler_kwds = code_compiler_kwds

        # For CUDA miniapps, Stage 0 must use the app's Makefile (nvcc).
        if source_lang == "CUDA":
            stage0_make_script = True
            stage0_build_script = False
            stage0_code_compiler = "make"
            stage0_code_compiler_kwds = "-j SM_VERSION=80 CFLAGS=-std=c++17"

        test_result = lassi.pipeline_methods.test_source_code(
            app_name,
            experiment_data_directory,
            code_version,
            trial_number,
            source_file_path,
            stage0_build_script,
            stage0_make_script,
            directory,
            source_lang,
            target_lang,
            exe_input_parameters,
            stage0_code_compiler,
            stage0_code_compiler_kwds,
            yes_execute,
            False,  # power_measure disabled
            input_content_llmjudge_dict,
            energy_history,
            power_history,
            runtime_history,
            valid_code_history,
        )

        if test_result is False:
            print(color.BOLD + color.RED + "\n❌ PIPELINE ERROR: Source code failed to compile and/or execute during Stage 0.\n" + color.END)
            return False

        # Unpack only what we need.
        (
            _idle_energy_joules,
            _idle_ave_power,
            input_content_llmjudge_dict,
            _sourcecode_total_energy_joules,
            _sourcecode_ave_power,
            _duration,
            energy_history,
            power_history,
            runtime_history,
            valid_code_history,
        ) = test_result

        print("\n----- END OF STAGE 0: TEST SOURCE -----\n")

    # ------------------------------------------
    # ----- STAGE 1+: TRANSLATE + SELF-FIX -----
    # ------------------------------------------
    start_time = time.time()

    compile_result = ""
    execute_result = ""
    code_block = ""
    self_correction_counter = 0

    # Build the translation prompt once.
    instructions_prompt = this_llm_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]"

    while self_correction_counter <= self_correction_max:
        # First attempt uses the translation prompt; subsequent attempts use error prompt.
        if self_correction_counter == 0:
            print("\n----- STAGE 1: CUDA→SYCL TRANSLATION (Argo) -----\n")
            _code_response, code_block = lassi.llm_methods.code_generator_llm(
                llm_source,
                model_coder,
                num_ctx_coder,
                0.2,
                this_llm_systemcode_prompt,
                instructions_prompt,
                display_response,
                verifyprompt,
            )
        else:
            print("\n----- SELF-CORRECTION (compile/run error) -----\n")
            if compile_result == "execute_error":
                error_intro = "\n -- The above CODE TO FIX executed and produced the following runtime error: "
                error_message = execute_result
            else:
                error_intro = "\n\n -- The above CODE TO FIX was compiled and produced the following compile error: "
                error_message = compile_result

            error_correction_prompt = prompts.get_code_error_correction_prompt("CUDA_to_SYCL_error")
            error_prompt = (
                error_correction_prompt
                + "\n\n[START OF CODE TO FIX] "
                + code_block
                + "\n\n[/END OF CODE TO FIX] "
                + error_intro
                + "\n\n"
                + error_message
            )

            _code_response, code_block = lassi.llm_methods.code_generator_llm(
                llm_source,
                model_coder,
                num_ctx_coder,
                0.2,
                this_llm_systemcode_prompt,
                error_prompt,
                display_response,
                verifyprompt,
            )

        # Normalize fenced code blocks.
        if code_block.startswith("c++") or code_block.startswith("cpp"):
            code_block = code_block[3:]
        if code_block.startswith("c"):
            code_block = code_block[1:]

        # Save generated code to the source tree (for compatibility with existing build logic).
        file_name_save = input_code_file_subdirectory + file_name
        file_path = os.getcwd() + "/" + file_name_save + "." + refactored_code_extension
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(code_block)

        # Ensure a Makefile exists in the experiment directory.
        if target_lang == "SYCL" and make_script:
            hardcoded_mk = os.path.join(os.path.dirname(__file__), "sycl_makefile_hardcoded.mk")
            makefile_dst = os.path.join(experiment_data_directory, "Makefile")
            with open(hardcoded_mk, "r") as src_f:
                mk = src_f.read()

            sycl_compiler = code_compiler
            if sycl_compiler in ["acpp", "<SYCL_COMPILER>"]:
                acpp_default = "/home/vchirio/adaptivecpp/bin/acpp"
                if os.path.exists(acpp_default):
                    sycl_compiler = acpp_default

            mk = mk.replace("<SYCL_COMPILER>", sycl_compiler)
            mk = mk.replace("<SYCL_FLAGS>", code_compiler_kwds)
            mk = mk.replace("<SYCL_LDFLAGS>", "")
            mk = mk.replace("<SYCL_SRC>", file_path)
            mk = mk.replace("<SYCL_OUT>", file_name_save)
            with open(makefile_dst, "w") as dst_f:
                dst_f.write(mk)

        # Compile.
        if make_script:
            if target_lang != "SYCL":
                raise ValueError("translation-only pipeline expects target_lang='SYCL' when make_script=True")

            make_args = f"-C {experiment_data_directory}"
            compile_result = lassi.code_manager.make_script_code(".", make_args, "SYCL")
            execute_file = os.path.join(experiment_data_directory, file_name_save)
        else:
            compile_result, execute_file = lassi.code_manager.compile_code(file_path, target_lang, code_compiler, code_compiler_kwds)

        if compile_result != "":
            self_correction_counter += 1
            continue

        # Execute.
        if yes_execute:
            execute_result, exe_output_generated, *_rest = lassi.code_manager.execute_code_selectvendor(
                experiment_data_directory,
                code_version + 1,
                True,
                trial_number,
                directory,
                execute_file,
                exe_input_parameters,
                False,  # power_measure disabled
                "generated",
                target_lang,
                build_script,
                make_script,
                gpu_vendor="auto",
                wait=5,
            )

            if execute_result != "0":
                compile_result = "execute_error"
                self_correction_counter += 1
                continue

            # Success: persist a versioned copy and stop immediately.
            versioned = os.path.join(
                experiment_data_directory,
                f"{file_name_save}_v1-baseLLM_Trial{trial_number}.{refactored_code_extension}",
            )
            with open(versioned, "w") as f:
                f.write(code_block)

            input_content_llmjudge_dict["refactored_code"] = code_block
            input_content_llmjudge_dict["refactored_stdout"] = exe_output_generated

            meta_data_file += "\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n" + input_content_llmjudge_dict["source_stdout"]
            meta_data_file += "\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n" + input_content_llmjudge_dict["refactored_stdout"]

            filename = os.path.join(
                experiment_data_directory,
                "metadata" + "_" + experiment.replace("/", "_") + "_Trial" + str(trial_number) + ".txt",
            )
            with open(filename, "w") as f:
                f.write(meta_data_file)
            print("\nMetadata file: " + str(filename))

            # Append collected data to experiment spreadsheet (if any fields exist).
            fieldnames = list(experiment_data.keys())
            with open(experiment_data_file_object, "a", newline="") as exp_file:
                writer = csv.DictWriter(exp_file, fieldnames=fieldnames)
                writer.writerow(experiment_data)

            end_time_minutes = round((time.time() - start_time) / 60, 2)
            print("\n✅ Translation complete (compile+run OK). Pipeline runtime (min):", end_time_minutes)
            return True

        # If execution is disabled, treat compile success as success.
        return True

    print(color.BOLD + color.RED + "\n❌ PIPELINE ERROR: exceeded self-correction attempts without a successful compile+run.\n" + color.END)
    return False
