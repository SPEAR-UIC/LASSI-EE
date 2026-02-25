from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import List

import lassi.code_manager
import lassi.llm_methods
from lassi.format_colors import color
from lassi.gpu_detector import detect_gpu_vendors
from lassi.prompt_dictionary import PromptDictionary

_B  = color.BOLD
_G  = color.GREEN
_R  = color.RED
_E  = color.END


@dataclass
class PipelineConfig:
    model: str
    system_prompt: str
    translate_prompt: str
    prompts: PromptDictionary
    input_code: str
    sycl_compiler: str
    sycl_flags: str
    exe_args: List[str]
    app_name: str
    experiment_dir: str
    csv_path: str
    experiment_data: dict
    experiment_name: str
    trial_number: int
    verifyprompt: bool = False
    use_self_prompting: bool = True
    verbose: bool = False


def _write_sycl_makefile(experiment_dir: str, sycl_compiler: str, sycl_flags: str, src: str, out: str) -> None:
    template = os.path.join(os.path.dirname(__file__), "sycl_makefile_hardcoded.mk")
    with open(template, "r") as f:
        mk = f.read()
    mk = mk.replace("<SYCL_COMPILER>", sycl_compiler)
    mk = mk.replace("<SYCL_FLAGS>", sycl_flags)
    mk = mk.replace("<SYCL_LDFLAGS>", "")
    mk = mk.replace("<SYCL_SRC>", src)
    mk = mk.replace("<SYCL_OUT>", out)
    with open(os.path.join(experiment_dir, "Makefile"), "w") as f:
        f.write(mk)


def _extract_xsbench_checksum(stdout: str) -> str:
    for line in (stdout or "").splitlines():
        if "Verification checksum:" in line:
            return line.strip()
    return ""


def _xsbench_checksum_valid(stdout: str) -> bool:
    line = _extract_xsbench_checksum(stdout)
    return bool(line) and "INVALID" not in line.upper()


def _tail5(text: str) -> str:
    """Return the last 5 non-empty lines, indented."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    return "\n".join(f"    {l}" for l in lines[-5:])


def run_pipeline(cfg: PipelineConfig) -> bool:
    max_attempts = 15
    exe_path = os.path.join(cfg.experiment_dir, "Main")
    start_time = time.time()

    vendors = detect_gpu_vendors()
    if not vendors:
        print(f"{_B}{_R}❌  No GPUs detected.{_E}")
        return False

    compile_err = ""
    run_err = ""
    run_stdout = ""
    code_block = ""

    # ----- STAGE 0: SELF-PROMPTING CONTEXT BUILDER -----
    # Following the methodology from the original LASSI paper (IEEE CLUSTER 2024):
    # (1) LLM summarizes target-language reference documentation
    # (2) LLM describes the source code to translate
    # (3) Both summaries are incorporated into the Stage 1 translation prompt
    sycl_context_summary = ""
    source_code_description = ""

    if cfg.use_self_prompting:
        print(f"\n{_B}----- STAGE 0: SELF-PROMPTING -----{_E}\n")

        # Load SYCL reference context
        try:
            sycl_context_raw = cfg.prompts.get_codeknowledge("SYCL_guide")
        except Exception as e:
            print(f"  [WARN] Could not load SYCL context document: {e}. Skipping.")
            sycl_context_raw = ""

        if sycl_context_raw:
            summarize_prompt = (
                cfg.prompts.get_selfprompt("summarize_context")
                + "\n\n[BEGIN SYCL REFERENCE GUIDE]\n"
                + sycl_context_raw
                + "\n[END SYCL REFERENCE GUIDE]"
            )
            print("  Step 1/2: Summarizing SYCL reference context...")
            sycl_context_summary = lassi.llm_methods.generate_text(
                cfg.model, cfg.prompts.get_system_prompt("selfprompt"), summarize_prompt,
                intent="context summarization", verifyprompt=cfg.verifyprompt,
                label="[LLM] Summarizing SYCL context knowledge",
            )

        describe_prompt = (
            cfg.prompts.get_selfprompt("describe_source")
            + "\n\n[BEGIN CUDA SOURCE CODE]\n"
            + cfg.input_code
            + "\n[END CUDA SOURCE CODE]"
        )
        print("  Step 2/2: Generating source code description...")
        source_code_description = lassi.llm_methods.generate_text(
            cfg.model, cfg.prompts.get_system_prompt("selfprompt"), describe_prompt,
            intent="source code description", verifyprompt=cfg.verifyprompt,
            label="[LLM] Describing source code",
        )

        print(f"\n  {_B}{_G}✓{_E}  Self-prompting complete. Context incorporated into Stage 1 prompt.\n")

    # Build the Stage 1 translation prompt with optional self-prompting context.
    # Prompt structure follows the original LASSI paper methodology:
    # (1) LLM-generated SYCL context summary  (2) LLM-generated source code description
    # (3) "Think carefully... Now, [translation task]. [source code]"
    if cfg.use_self_prompting and (sycl_context_summary or source_code_description):
        context_preamble = ""
        if sycl_context_summary:
            context_preamble += (
                "[SYCL CONTEXT KNOWLEDGE]\n"
                + sycl_context_summary
                + "\n[/SYCL CONTEXT KNOWLEDGE]\n\n"
            )
        if source_code_description:
            context_preamble += (
                "[YOUR ANALYSIS OF THE SOURCE CODE TO TRANSLATE]\n"
                + source_code_description
                + "\n[/YOUR ANALYSIS OF THE SOURCE CODE TO TRANSLATE]\n\n"
            )
        think_prefix = "Think carefully, using your analysis above. Now, " if source_code_description else ""
        translate_user_prompt = (
            context_preamble
            + think_prefix
            + cfg.translate_prompt
            + "\n\n[START OF CODE TO REFACTOR]\n"
            + cfg.input_code
            + "\n[/END OF CODE TO REFACTOR]"
        )
    else:
        translate_user_prompt = (
            cfg.translate_prompt
            + "\n\n[START OF CODE TO REFACTOR]\n"
            + cfg.input_code
            + "\n[/END OF CODE TO REFACTOR]"
        )

    for attempt in range(max_attempts + 1):
        if attempt == 0:
            print(f"\n{_B}----- STAGE 1: CUDA→SYCL TRANSLATION -----{_E}\n")
            if cfg.verbose:
                ctx_parts = ["source code"]
                if sycl_context_summary:
                    ctx_parts.append("SYCL context summary")
                if source_code_description:
                    ctx_parts.append("source code description")
                print(f"  Context : {' + '.join(ctx_parts)}")
            code_block = lassi.llm_methods.generate_code(
                cfg.model, cfg.system_prompt, translate_user_prompt, cfg.verifyprompt,
                label="[LLM] Generating initial SYCL translation",
            )
        else:
            # Print the Stage 2 banner only on the first correction attempt
            if attempt == 1:
                print(f"\n{_B}----- STAGE 2: SELF-CORRECTION -----{_E}\n")

            # Determine what kind of error we're correcting
            if compile_err == "execute_error":
                err_type = "runtime error"
                error_msg = run_err
                stdout_ctx = " + stdout" if run_stdout.strip() else ""
                if run_stdout.strip():
                    error_msg += "\n\nProgram stdout before crash:\n" + run_stdout
                error_msg += "\n\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n" + cfg.input_code + "\n[/ORIGINAL CUDA SOURCE]\n"
                error_intro = "\n -- Runtime error: "
            elif compile_err.startswith("VERIFICATION MISMATCH"):
                err_type = "checksum mismatch"
                error_msg = compile_err
                error_intro = "\n -- Verification mismatch: "
            else:
                err_type = "compile error"
                error_msg = compile_err
                error_intro = "\n -- Compile error: "

            print(f"  {_B}Attempt {attempt}/{max_attempts}{_E} — correcting {err_type}")

            if cfg.verbose:
                print(f"  Prompts : system=[CUDA_to_SYCL]  |  user=[CUDA_to_SYCL_error]")

            error_prompt = (
                cfg.prompts.get_code_error_correction_prompt("CUDA_to_SYCL_error")
                + "\n\n[START OF CODE TO FIX]\n"
                + code_block
                + "\n[/END OF CODE TO FIX]"
                + error_intro + "\n"
                + error_msg
            )

            code_block = lassi.llm_methods.generate_code(
                cfg.model, cfg.system_prompt, error_prompt, cfg.verifyprompt,
                label=f"[LLM] Generating corrected SYCL code",
            )

        if cfg.verbose:
            lassi.llm_methods.show_code_preview(
                code_block,
                title=f"Stage {'1' if attempt == 0 else '2'} generated code — attempt {attempt + 1}",
            )

        # Write generated code to a per-attempt file in the experiment directory.
        attempt_src = os.path.join(cfg.experiment_dir, f"Main_attempt{attempt}.cpp")
        with open(attempt_src, "w", encoding="utf-8") as f:
            f.write(code_block)
        if cfg.verbose:
            print(f"  [GEN] Wrote generated SYCL code to: {attempt_src}")

        # Compile via Makefile.
        _write_sycl_makefile(cfg.experiment_dir, cfg.sycl_compiler, cfg.sycl_flags, attempt_src, "Main")
        if cfg.verbose:
            print(f"  [BUILD] Wrote Makefile to experiment directory")

        compile_err = lassi.code_manager.make_script_code(
            cfg.experiment_dir,
            "",
            "SYCL",
            experiment_dir=cfg.experiment_dir,
            log_prefix=f"build_attempt{attempt}_",
        )
        if compile_err:
            tail = _tail5(compile_err)
            print(f"  Compiling...  {_B}{_R}✗  Compile error:{_E}")
            if tail:
                print(tail)
            continue
        print(f"  Compiling...  {_B}{_G}✓  OK{_E}")

        # Run.
        run_result, run_stdout, _, _, _, _, run_combined = lassi.code_manager.execute_code(
            cfg.experiment_dir,
            1,
            True,
            cfg.trial_number,
            "",
            exe_path,
            cfg.exe_args,
            False,
            "generated",
            "SYCL",
            False,
            True,
            wait=5,
        )

        if run_result != "0":
            compile_err = "execute_error"
            run_err = run_result
            tail = _tail5(run_combined or run_result)
            print(f"  Running...    {_B}{_R}✗  Runtime error:{_E}")
            if tail:
                print(tail)
            continue
        print(f"  Running...    {_B}{_G}✓  OK{_E}")

        # Verify XSBench checksum.
        checksum_line = _extract_xsbench_checksum(run_stdout)
        if not _xsbench_checksum_valid(run_stdout):
            print(f"  Checksum:     {_B}{_R}✗  {checksum_line or 'missing'}{_E}")
            stdout_snippet = (run_stdout[:1000].strip() or "(no output)") + ("\n...(truncated)" if len(run_stdout) > 1000 else "")
            detail = (
                f"XSBench checksum missing or invalid.\n"
                f"Checksum line: {checksum_line!r}\n"
                f"\nActual stdout:\n{stdout_snippet}\n"
                f"\nYour SYCL translation MUST faithfully implement ALL functions from "
                f"the original CUDA source, including print_results(), "
                f"run_event_based_simulation_baseline(), and grid_init_do_not_profile().\n"
                f"Do NOT generate a simplified stub — translate the complete source.\n"
                f"\n[ORIGINAL CUDA SOURCE FOR REFERENCE]\n{cfg.input_code}\n[/ORIGINAL CUDA SOURCE]\n"
            )
            compile_err = "VERIFICATION MISMATCH: " + detail
            continue

        print(f"  Checksum:     {_B}{_G}✓  {checksum_line}{_E}")

        # Success — write versioned copy and CSV row.
        versioned = os.path.join(cfg.experiment_dir, f"Main_v1_Trial{cfg.trial_number}.cpp")
        with open(versioned, "w") as f:
            f.write(code_block)

        runtime_min = round((time.time() - start_time) / 60, 2)
        attempts_label = f"attempt {attempt + 1}" if attempt > 0 else "first attempt"
        print(f"\n  {_B}{_G}✅  Translation complete{_E}  —  {attempts_label}  |  {runtime_min} min total\n")

        cfg.experiment_data["success"] = True
        cfg.experiment_data["runtime_min"] = runtime_min
        with open(cfg.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cfg.experiment_data.keys()))
            writer.writerow(cfg.experiment_data)

        return True

    print(f"{_B}{_R}\n❌  PIPELINE ERROR: exceeded self-correction attempts ({max_attempts}).\n{_E}")
    return False
