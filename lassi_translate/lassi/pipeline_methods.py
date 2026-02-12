"""Helpers used by the translation pipeline.

Only Stage 0 (build+run source app) is implemented here.
"""

from __future__ import annotations

import os

import lassi.code_manager
from lassi.format_colors import color


def test_source_code(
    app_name,
    experiment_data_directory,
    code_version,
    trial_number,
    source_file_path,
    build_script,
    make_script,
    directory,
    source_lang,
    target_lang,
    exe_input_parameters,
    code_compiler,
    code_compiler_kwds,
    yes_execute,
    power_measure,
    input_content_llmjudge_dict,
    energy_history,
    power_history,
    runtime_history,
    valid_code_history,
):
    """Compile + run the source app and store stdout in `input_content_llmjudge_dict`.

    Power/energy measurement is intentionally not supported; `power_measure` is ignored.

    Returns False on failure, otherwise returns a tuple compatible with the original
    pipeline signature.
    """

    print("\n\n----- " + color.BLUE + "SOURCE CODE: build+run" + color.END + " -----")
    print("\nSource: " + str(source_file_path))

    # Build.
    if make_script:
        make_dir = os.path.dirname(source_file_path)
        compile_err = lassi.code_manager.make_script_code(make_dir, code_compiler_kwds, source_lang)
        if compile_err:
            print("\n\nFailed compile result: " + str(compile_err))
            return False

        # Known miniapp binary names.
        if source_lang in {"CUDA", "HIP"} and app_name == "XSBench":
            execute_file = "XSBench"
        else:
            execute_file = app_name
    else:
        compile_err, execute_file = lassi.code_manager.compile_code(
            source_file_path, source_lang, code_compiler, code_compiler_kwds
        )
        make_dir = directory
        if compile_err:
            print("\n\nFailed compile result: " + str(compile_err))
            return False

    if not yes_execute:
        input_content_llmjudge_dict["source_stdout"] = ""
        input_content_llmjudge_dict["source_metrics"] = ""
        return (0.0, 0.0, input_content_llmjudge_dict, 0.0, 0.0, 0.0, energy_history, power_history, runtime_history, valid_code_history)

    # Run.
    print("\nRun: " + str(execute_file))
    execute_result, stdout, _e, _p, _sp, duration, _m = lassi.code_manager.execute_code_selectvendor(
        experiment_data_directory,
        code_version,
        False,
        trial_number,
        make_dir if make_script else directory,
        execute_file,
        exe_input_parameters,
        False,
        "source",
        target_lang,
        build_script,
        make_script,
        gpu_vendor="auto",
        wait=1,
    )

    input_content_llmjudge_dict["source_stdout"] = stdout
    input_content_llmjudge_dict["source_metrics"] = ""

    if execute_result != "0":
        return False

    # No power measurement in this bundle.
    energy_history.append(0.0)
    power_history.append(0.0)
    runtime_history.append(round(float(duration), 2))
    valid_code_history.append("SOURCE")

    return (
        0.0,
        0.0,
        input_content_llmjudge_dict,
        0.0,
        0.0,
        duration,
        energy_history,
        power_history,
        runtime_history,
        valid_code_history,
    )
