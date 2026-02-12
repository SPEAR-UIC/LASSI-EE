"""Build and execution helpers.

Console logging policy:
- Always print a concise one-line summary of the command being run.
- On failure, print only the last ~50 lines of output to the console.
- Always write full stdout/stderr logs to files under the experiment directory.

Power/energy measurement is not supported.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from lassi.format_colors import color


def _tail_lines(text: str, n: int = 50) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= n:
        return (text or "").strip() + ("\n" if text and not text.endswith("\n") else "")
    return "\n".join(lines[-n:]).strip() + "\n"


def _write_log(experiment_dir: Optional[str], filename: str, content: str) -> None:
    if not experiment_dir:
        return
    p = Path(experiment_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content or "", encoding="utf-8")


def _run_cmd(
    *,
    cmd: List[str],
    cwd: str,
    env: Optional[dict] = None,
    experiment_dir: Optional[str] = None,
    log_prefix: str = "",
) -> subprocess.CompletedProcess:
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[CMD] {cmd_str} (cwd={cwd})")

    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)

    if experiment_dir:
        _write_log(experiment_dir, f"{log_prefix}stdout.log", result.stdout)
        _write_log(experiment_dir, f"{log_prefix}stderr.log", result.stderr)
        _write_log(
            experiment_dir,
            f"{log_prefix}cmd.txt",
            cmd_str + "\n" + f"cwd={cwd}\n",
        )

    return result


def make_script_code(
    folder: str,
    make_args: str,
    source_lang: str,
    *,
    experiment_dir: Optional[str] = None,
    log_prefix: str = "build_",
) -> str:
    """Run `make`.

    Returns "" on success, otherwise returns an error string.

    If `experiment_dir` is provided, full logs are written there.
    """

    target_dir = folder if os.path.isabs(folder) else os.path.join(os.getcwd(), folder)

    env = os.environ.copy()
    if source_lang == "CUDA":
        env["GPU_INCLUDE_PATH"] = "/usr/local/cuda/include"
        env["GPU_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
        env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    cmd = ["make"] + shlex.split(make_args)

    # Clean first to avoid stale objects when the pipeline overwrites sources.
    _run_cmd(
        cmd=["make", "clean"],
        cwd=target_dir,
        env=env,
        experiment_dir=experiment_dir,
        log_prefix=log_prefix + "clean_",
    )

    # If caller uses `-C <dir>`, don't also set cwd to that same dir.
    run_cwd = os.getcwd() if any(tok == "-C" or tok.startswith("-C") for tok in cmd) else target_dir

    result = _run_cmd(cmd=cmd, cwd=run_cwd, env=env, experiment_dir=experiment_dir, log_prefix=log_prefix)

    if result.returncode == 0:
        print(color.BOLD + "Make build OK" + color.END)
        return ""

    combined = (result.stderr or "") + ("\n" if result.stderr and result.stdout else "") + (result.stdout or "")
    print(color.BOLD + color.RED + "Make build FAILED (showing last 50 lines)" + color.END)
    print(_tail_lines(combined, 50), end="")

    err = (result.stderr or "").strip() or (result.stdout or "").strip()
    return (err + "\n") if err else "make failed\n"


def compile_code(
    source_file: str,
    lang: str,
    compiler: str,
    compiler_flags: str,
    *,
    experiment_dir: Optional[str] = None,
    log_prefix: str = "compile_",
) -> Tuple[str, str]:
    """Compile a single file.

    Returns (error_string, output_file_path). error_string is "" on success.

    If `experiment_dir` is provided, full logs are written there.
    """

    flags = shlex.split(compiler_flags)

    if lang in {"OMP", "SYCL", "HIP"}:
        output_file = source_file.rsplit(".cpp", 1)[0] if source_file.endswith(".cpp") else source_file + "_compiled"
    elif lang == "CUDA":
        output_file = source_file.rsplit(".cu", 1)[0] if source_file.endswith(".cu") else source_file + "_compiled"
    else:
        raise ValueError(f"Unsupported language for compilation: {lang}")

    cmd = [compiler] + flags + [source_file, "-o", output_file]
    result = _run_cmd(cmd=cmd, cwd=os.getcwd(), env=None, experiment_dir=experiment_dir, log_prefix=log_prefix)

    if result.returncode == 0:
        print(color.BOLD + "Compilation OK" + color.END)
        return "", output_file

    combined = (result.stderr or "") + ("\n" if result.stderr and result.stdout else "") + (result.stdout or "")
    print(color.BOLD + color.RED + "Compilation FAILED (showing last 50 lines)" + color.END)
    print(_tail_lines(combined, 50), end="")

    return (result.stderr or result.stdout or "compile failed") + "\n", output_file


def execute_code_selectvendor(
    experiment_data_directory: str,
    code_version: int,
    zero_shot: bool,
    trial_number: int,
    directory: str,
    executable_file: str,
    exe_input_parameters: List[str],
    power_measure: bool,
    code_source: str,
    target_lang: str,
    build_script: bool,
    make_script: bool,
    gpu_vendor: str = "auto",
    wait: int = 5,
):
    """Compatibility wrapper; vendor/power args are ignored."""

    return execute_code(
        experiment_data_directory,
        code_version,
        zero_shot,
        trial_number,
        directory,
        executable_file,
        exe_input_parameters,
        False,
        code_source,
        target_lang,
        build_script,
        make_script,
        wait=wait,
    )


def execute_code(
    experiment_data_directory: str,
    code_version: int,
    zero_shot: bool,
    trial_number: int,
    directory: str,
    executable_file: str,
    exe_input_parameters: List[str],
    power_measure: bool,
    code_source: str,
    target_lang: str,
    build_script: bool,
    make_script: bool,
    wait: int = 5,
    execution_timeout: int = 300,
):
    """Run an executable and capture stdout/stderr.

    Console:
    - prints the command and cwd
    - prints only the last ~50 lines of output on failure

    Files:
    - writes full stdout/stderr to the experiment directory

    Returns:
      (return_result, stdout_output, total_energy_joules, ave_power, std_power, duration, metrics_results_string)

    Power/energy fields are always 0.0 in this bundle.
    """

    # Optional settle time.
    for _ in range(wait):
        time.sleep(1)

    # Determine working directory.
    if make_script:
        if os.path.isabs(executable_file):
            workdir = os.path.dirname(executable_file) or os.getcwd()
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            workdir = directory if os.path.isabs(directory) else os.path.join(repo_root, directory) if directory else repo_root
    else:
        workdir = os.path.dirname(executable_file) or os.getcwd()

    executable = os.path.basename(executable_file)
    os.makedirs(workdir, exist_ok=True)

    # Ensure the binary has execute permissions.
    subprocess.run(["chmod", "+x", os.path.join(workdir, executable)], check=False)

    if build_script:
        args = exe_input_parameters[0].split(" ") if exe_input_parameters else []
    else:
        args = exe_input_parameters or []

    cmd = ["./" + executable] + args

    if not os.path.exists(os.path.join(workdir, executable)):
        msg = f"Pipeline error | The file {executable} does not exist."
        return msg, "", 0.0, 0.0, 0.0, 0.0, ""

    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"[CMD] {cmd_str} (cwd={workdir})")

    start = time.time()
    try:
        p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate(timeout=execution_timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        stdout, stderr, rc = "", f"Execution timeout after {execution_timeout} seconds", -999

    duration = time.time() - start

    # Always persist full logs.
    _write_log(experiment_data_directory, f"run_trial{trial_number}_v{code_version}_stdout.log", stdout)
    _write_log(experiment_data_directory, f"run_trial{trial_number}_v{code_version}_stderr.log", stderr)
    _write_log(experiment_data_directory, f"run_trial{trial_number}_v{code_version}_cmd.txt", cmd_str + "\n" + f"cwd={workdir}\n")

    # Some miniapps (e.g., XSBench) may return non-zero even when they printed RESULTS.
    if rc == 0 or "RESULTS" in (stdout or ""):
        print(color.BOLD + "Run OK" + color.END)
        return "0", stdout, 0.0, 0.0, 0.0, duration, ""

    combined = (stderr or "") + ("\n" if stderr and stdout else "") + (stdout or "")
    err = f"Program exited with return code: {rc}"
    if (stderr or "").strip():
        err += "\nStandard Error: " + stderr

    print(color.BOLD + color.RED + "Run FAILED (showing last 50 lines)" + color.END)
    print(_tail_lines(combined, 50), end="")

    return err, stdout, 0.0, 0.0, 0.0, duration, ""
