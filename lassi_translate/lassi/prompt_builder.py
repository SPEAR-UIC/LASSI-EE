"""LLM-prompt assembly: self-prompting, Stage 1, Stage 2 correction, judge."""
from __future__ import annotations

import lassi.llm_client as _llm
from lassi.format_colors    import color as _c
from lassi.prompt_dictionary import PromptDictionary

_B, _G, _R, _E = _c.BOLD, _c.GREEN, _c.RED, _c.END

# Task→key tables: keyed by (task, resource) where 'resource' only matters
# for the "energy" task. "sycl_energy" is a sub-task of "both".
_TASK_KEYS = {
    # task, resource → (guide_key, summarize_key, describe_key, guide_label,
    #                   sys_key, usr_key, base_err_key, improve_key, block_tag)
    ("translate",  "polaris"): (
        "SYCL_guide", "summarize_context_sycl", "describe_source_translate",
        "SYCL reference guide",
        "CUDA_to_SYCL", "CUDA_to_SYCL", "CUDA_to_SYCL_error", None,
        "CODE TO REFACTOR",
    ),
    ("translate",  "aurora"):  (
        "SYCL_guide", "summarize_context_sycl", "describe_source_translate",
        "SYCL reference guide",
        "CUDA_to_SYCL", "CUDA_to_SYCL", "CUDA_to_SYCL_error", None,
        "CODE TO REFACTOR",
    ),
    ("energy",     "polaris"): (
        "CUDA_energy_guide", "summarize_context_cuda_energy", "describe_source_cuda_energy",
        "CUDA energy efficiency guide",
        "CUDA_optimize_energy", "CUDA_optimize_energy",
        "CUDA_optimize_energy_error", "CUDA_optimize_energy_improve",
        "CUDA CODE TO OPTIMIZE",
    ),
    ("energy",     "aurora"):  (
        "SYCL_energy_guide", "summarize_context_sycl_energy", "describe_source_sycl_energy",
        "SYCL energy efficiency guide",
        "SYCL_optimize_energy", "SYCL_optimize_energy",
        "SYCL_optimize_energy_error", "SYCL_optimize_energy_improve",
        "SYCL CODE TO OPTIMIZE",
    ),
    # NOTE: original code sets improve_key=None for task=="sycl_energy" (only
    # task=="energy" gets a non-None improve_key).  Preserve that behavior so
    # energy-keep/not-improved corrections in the "both" pipeline's Step 2 fall
    # through to the base error prompt, matching the unrefactored version.
    ("sycl_energy","polaris"): (
        "SYCL_energy_guide", "summarize_context_sycl_energy", "describe_source_sycl_energy",
        "SYCL energy efficiency guide",
        "SYCL_optimize_energy", "SYCL_optimize_energy",
        "SYCL_optimize_energy_error", None,
        "SYCL CODE TO OPTIMIZE",
    ),
    ("sycl_energy","aurora"):  (
        "SYCL_energy_guide", "summarize_context_sycl_energy", "describe_source_sycl_energy",
        "SYCL energy efficiency guide",
        "SYCL_optimize_energy", "SYCL_optimize_energy",
        "SYCL_optimize_energy_error", None,
        "SYCL CODE TO OPTIMIZE",
    ),
}


def _task_keys(task: str, resource: str) -> tuple:
    return _TASK_KEYS.get((task, resource), _TASK_KEYS[("sycl_energy", resource)])


def _selfprompt_preamble(guide_summary: str, source_description: str) -> str:
    """Build the [CONTEXT KNOWLEDGE] + [ANALYSIS] preamble shared by Stage 1
    and energy-correction Stage 2 prompts."""
    parts: list[str] = []
    if guide_summary:
        parts.append(
            "[CONTEXT KNOWLEDGE]\n" + guide_summary + "\n[/CONTEXT KNOWLEDGE]\n\n"
        )
    if source_description:
        parts.append(
            "[YOUR ANALYSIS OF THE SOURCE CODE]\n"
            + source_description
            + "\n[/YOUR ANALYSIS OF THE SOURCE CODE]\n\n"
        )
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: self-prompting (context summary + source analysis)
# ─────────────────────────────────────────────────────────────────────────────

def run_selfprompt(
    *, model: str, prompts: PromptDictionary,
    input_code: str, task: str, resource: str,
    verifyprompt: bool, use_self_prompting: bool,
) -> tuple[str, str]:
    """Run Stage 0 LLM calls and return (guide_summary, source_description).

    Either may be empty if self-prompting is disabled or the context document
    is unavailable.
    """
    if not use_self_prompting:
        return "", ""

    print(f"\n{_B}----- STAGE 0: SELF-PROMPTING  (local Argo) -----{_E}\n")

    guide_key, summarize_key, describe_key, guide_label, *_ = _task_keys(task, resource)
    system_prompt = prompts.get_system_prompt("selfprompt")
    guide_summary = ""

    try:
        guide_raw = prompts.get_codeknowledge(guide_key)
    except (KeyError, FileNotFoundError) as exc:
        print(f"  {_B}WARN:{_E} Could not load {guide_label}: {exc}.  Skipping summary.")
        guide_raw = ""

    if guide_raw:
        print(f"  Step 1/2: Summarizing {guide_label}...")
        guide_summary = _llm.generate_text(
            model, system_prompt,
            prompts.get_selfprompt(summarize_key)
                + f"\n\n[BEGIN GUIDE]\n{guide_raw}\n[END GUIDE]",
            intent=f"{guide_label} summarization",
            verifyprompt=verifyprompt,
            label=f"[LLM] Summarizing {guide_label}",
        )
    else:
        print(f"  Step 1/2: Skipping guide summarization (no guide file available).")

    print("  Step 2/2: Analyzing source code...")
    source_description = _llm.generate_text(
        model, system_prompt,
        prompts.get_selfprompt(describe_key)
            + "\n\n[BEGIN SOURCE CODE]\n" + input_code + "\n[END SOURCE CODE]",
        intent="source code analysis",
        verifyprompt=verifyprompt,
        label="[LLM] Analyzing source code",
    )

    print(f"\n  {_B}{_G}✓{_E}  Self-prompting complete.\n")
    return guide_summary, source_description


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: initial generation prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_stage1_prompt(
    *, prompts: PromptDictionary,
    task: str, resource: str, input_code: str,
    guide_summary: str, source_description: str,
    baseline_energy_j: float = -1.0, baseline_power_w: float = -1.0,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the initial Stage 1 LLM call."""
    _, _, _, _, sys_key, usr_key, _, _, block_tag = _task_keys(task, resource)

    system_prompt  = prompts.get_system_prompt(sys_key)
    translate_user = prompts.get_codetranslate_prompt(usr_key)

    preamble     = _selfprompt_preamble(guide_summary, source_description)
    think_prefix = "Think carefully, using your analysis above.  Now, " if source_description else ""

    user_prompt = (
        preamble + think_prefix + translate_user
        + f"\n\n[START OF {block_tag}]\n" + input_code
        + f"\n[/END OF {block_tag}]"
    )

    if task in ("energy", "sycl_energy") and baseline_energy_j > 0:
        user_prompt += (
            f"\n\nFor reference, the {block_tag} provided above was executed on this system "
            f"and consumed a total energy of {baseline_energy_j:.2f} Joules at an average power "
            f"of {baseline_power_w:.2f} Watts, which includes the idle system energy consumption "
            f"and average power.\n"
        )
    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: error-correction prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_correction_prompt(
    *, prompts: PromptDictionary,
    task: str, resource: str, code_block: str,
    compile_err: str, run_err: str, run_stdout: str, input_code: str,
    attempt: int,
    guide_summary: str = "", source_description: str = "",
    best_code_block: str = "",
    opt_power_w: float = -1.0, opt_dur_s: float = -1.0,
) -> tuple[str, str]:
    """Classify the failure and build the Stage 2 (correction) user prompt.

    Returns (err_type_label, user_prompt).
    """
    *_, base_err_key, improve_key, _ = _task_keys(task, resource)

    if compile_err == "execute_error":
        err_type    = "runtime error"
        error_msg   = run_err
        if run_stdout.strip():
            error_msg += "\n\nProgram stdout before crash:\n" + run_stdout
        error_msg  += "\n\n[ORIGINAL SOURCE FOR REFERENCE]\n" + input_code + "\n[/ORIGINAL SOURCE]\n"
        error_intro = "\n -- Runtime error: "
        prompt_key  = base_err_key
    elif compile_err.startswith("VERIFICATION MISMATCH"):
        err_type    = "checksum mismatch"
        error_msg   = compile_err
        error_intro = "\n -- Verification mismatch: "
        prompt_key  = base_err_key
    elif compile_err.startswith("ENERGY_KEEP_IMPROVING") and improve_key:
        err_type    = "energy keep improving"
        error_msg   = compile_err
        if opt_power_w > 0 and opt_dur_s > 0:
            error_msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        error_intro = "\n -- Energy result: "
        prompt_key  = improve_key
        if best_code_block:
            code_block = best_code_block
    elif compile_err.startswith("ENERGY_NOT_IMPROVED") and improve_key:
        err_type    = "energy improvement insufficient"
        error_msg   = compile_err
        # Append power and duration so the model can reason about whether
        # execution time or power draw is the dominant cost.
        if opt_power_w > 0 and opt_dur_s > 0:
            error_msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        error_intro = "\n -- Energy result: "
        prompt_key  = improve_key
        # Reuse the best-performing code seen so far as the starting point
        # (matches base LASSI-EE behaviour: code_blocks[tracking_best_version+1]).
        if best_code_block:
            code_block = best_code_block
    else:
        err_type    = "compile error"
        error_msg   = compile_err
        error_intro = "\n -- Compile error: "
        prompt_key  = base_err_key

    correction_base = prompts.get_code_error_correction_prompt(prompt_key)

    # For energy corrections, re-inject Stage 0 context so the model retains
    # the knowledge summary and source analysis (matches base LASSI-EE).
    if err_type in ("energy improvement insufficient", "energy keep improving"):
        correction_base = _selfprompt_preamble(guide_summary, source_description) + correction_base

    user_prompt = (
        correction_base
        + "\n\n[START OF CODE TO FIX]\n" + code_block + "\n[/END OF CODE TO FIX]"
        + error_intro + "\n" + error_msg
    )
    return err_type, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge (functional-equivalence check after a successful compile+run)
# ─────────────────────────────────────────────────────────────────────────────

def run_judge(
    *, model: str, system_prompt: str,
    source_code: str, source_stdout: str, source_metrics: str,
    refactored_code: str, refactored_stdout: str, refactored_metrics: str,
) -> str:
    """Returns "VALID", "INVALID", or "NONE"."""
    print(f"\n  {_B}[Judge] Verifying functional equivalence via LLM...{_E}")

    content_prompt = (
        "Please compare the difference between the following two CUDA C++ codes.\n\n"
        "First, I will provide you with the SOURCE CODE followed by its STANDARD OUTPUT "
        "and METRICS measured during code runtime.\n"
        "Next, I will provide you with a REFACTORED CODE followed by its STANDARD OUTPUT "
        "and METRICS measured during code runtime.\n\n"
        "After you have reviewed both codes and their output, the REFACTORED CODE is expected "
        "to produce equivalent functionality and reasonably similar standard output in its "
        "layout and content. However, the REFACTORED CODE may not meet these expectations.\n\n"
        "Your task is to strictly evaluate whether the REFACTORED CODE (1) preserves all "
        "intended functionality of the SOURCE CODE and (2) produces printed nearly equivalent "
        "standard output, including its displayed layout and content, compared to the SOURCE CODE.\n\n"
        "Your careful evaluation should focus on the functional equivalence of the two codes "
        "and nearly exact printed standard output of the two codes. Minor differences in "
        "numerical values are acceptable in the printed standard output only if they are "
        "consistent with the REFACTORED CODE improving energy efficiency and these specific "
        "values are reasonable and consistent with the SOURCE CODE's intended purpose. "
        "Formatting differences are acceptable as long as key details are included that are "
        "seen in the standard output of the SOURCE CODE.\n\n"
        "Please explain your reasoning, and then conclude your response with VALID if the "
        "REFACTORED CODE is functionally equivalent, or INVALID if it is not."
        f"\n\n# [SOURCE CODE]\n{source_code}"
        f"\n\n## [STANDARD OUTPUT]\n{source_stdout}"
        f"\n\n## [METRICS]\n{source_metrics}"
        f"\n\n# [REFACTORED CODE]\n{refactored_code}"
        f"\n\n## [STANDARD OUTPUT]\n{refactored_stdout}"
        f"\n\n## [METRICS]\n{refactored_metrics}"
    )

    response = _llm.generate_text(
        model, system_prompt, content_prompt,
        intent="code validation judge",
        label="[LLM] Judge",
        temperature=0.2,
    )

    if "INVALID" in response:
        verdict = "INVALID"
    elif "VALID" in response:
        verdict = "VALID"
    else:
        verdict = "NONE"

    print(f"  {_B}[Judge] verdict: {verdict}{_E}")
    return verdict


# Used by pipeline._write_metadata when an energy improvement succeeds.
CODE_COMPARISON_PROMPT = (
    "Below are two sets of parallel code written with CUDA in C++ that we want to compare their "
    "differences. The first is labeled SOURCE CODE and it is followed by its output to the stdout "
    "when it was executed as well as the total energy consumed and average power. The second is "
    "labeled REFACTORED CODE followed by its output to the stdout when executed on the same GPU "
    "and its energy measurements. Both codes are expected to perform the same function. Both codes "
    "are run on the same GPU and compute system. However, the REFACTORED CODE is a refactoring of "
    "the SOURCE CODE generated by an LLM that was prompted to improve the energy efficiency of the "
    "SOURCE CODE by reducing energy consumption and/or reducing the average power utilized by the "
    "GPU while the code is executed. Please compare the differences in the REFACTORED CODE from "
    "the SOURCE CODE by highlighting any changes in the REFACTORED CODE that might be related to "
    "improving its energy efficiency. Avoid repeating the entire code for reference. Conclude your "
    "response with a simple list of the optimization methods, if any, used in the REFACTORED CODE "
    "not included in the SOURCE CODE that improves energy efficiency. Surround this list of "
    "optimization methods with the tags [OPTIMIZATION LIST] and [/OPTIMIZATION LIST]"
)
