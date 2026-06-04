"""LLM prompt assembly + the LLM-as-judge correctness check.

Three stages of the pipeline call into this module:
  Stage 0: run_selfprompt          — summarize context, analyze source
  Stage 1: build_stage1_prompt     — initial generation prompt
  Stage 2: build_correction_prompt — per-attempt error-correction prompt

Prompt wording lives in lassi/prompt_dictionary.py (empirically tuned).
"""
import lassi.llm_client as _llm
from lassi.format_colors    import B, G, E
from lassi.prompt_dictionary import PromptDictionary

# (task, resource) → (guide_key, summarize_key, describe_key, guide_label,
#                     sys_key, usr_key, base_err_key, improve_key, block_tag)
_TASK_KEYS = {
    ("translate","polaris"): ("SYCL_guide", "summarize_context_sycl", "describe_source_translate",
        "SYCL reference guide", "CUDA_to_SYCL", "CUDA_to_SYCL", "CUDA_to_SYCL_error", None,
        "CODE TO REFACTOR"),
    ("translate","aurora"):  ("SYCL_guide", "summarize_context_sycl", "describe_source_translate",
        "SYCL reference guide", "CUDA_to_SYCL", "CUDA_to_SYCL", "CUDA_to_SYCL_error", None,
        "CODE TO REFACTOR"),
    ("energy","polaris"):    ("CUDA_energy_guide", "summarize_context_cuda_energy",
        "describe_source_cuda_energy", "CUDA energy efficiency guide",
        "CUDA_optimize_energy", "CUDA_optimize_energy",
        "CUDA_optimize_energy_error", "CUDA_optimize_energy_improve",
        "CUDA CODE TO OPTIMIZE"),
    ("energy","aurora"):     ("SYCL_energy_guide", "summarize_context_sycl_energy",
        "describe_source_sycl_energy", "SYCL energy efficiency guide",
        "SYCL_optimize_energy", "SYCL_optimize_energy",
        "SYCL_optimize_energy_error", "SYCL_optimize_energy_improve",
        "SYCL CODE TO OPTIMIZE"),
    # sycl_energy = energy sub-step of "both"; improve_key=None matches base LASSI-EE.
    ("sycl_energy","polaris"): ("SYCL_energy_guide", "summarize_context_sycl_energy",
        "describe_source_sycl_energy", "SYCL energy efficiency guide",
        "SYCL_optimize_energy", "SYCL_optimize_energy",
        "SYCL_optimize_energy_error", None, "SYCL CODE TO OPTIMIZE"),
    ("sycl_energy","aurora"):  ("SYCL_energy_guide", "summarize_context_sycl_energy",
        "describe_source_sycl_energy", "SYCL energy efficiency guide",
        "SYCL_optimize_energy", "SYCL_optimize_energy",
        "SYCL_optimize_energy_error", None, "SYCL CODE TO OPTIMIZE"),
}


def _keys(task, resource):
    """Look up the prompt-key tuple for a (task, resource) combo."""
    return _TASK_KEYS.get((task, resource), _TASK_KEYS[("sycl_energy", resource)])


def _preamble(guide_summary, source_description):
    parts = []
    if guide_summary:
        parts.append(f"[CONTEXT KNOWLEDGE]\n{guide_summary}\n[/CONTEXT KNOWLEDGE]\n\n")
    if source_description:
        parts.append(f"[YOUR ANALYSIS OF THE SOURCE CODE]\n{source_description}\n"
                     f"[/YOUR ANALYSIS OF THE SOURCE CODE]\n\n")
    return "".join(parts)


def run_selfprompt(*, model, prompts: PromptDictionary, input_code,
                   task, resource, verifyprompt, use_self_prompting):
    """Stage 0: ask the LLM to summarize the reference guide + analyze the source.
    Returns (guide_summary, source_description) — both empty if disabled."""
    if not use_self_prompting:
        return "", ""
    print(f"\n{B}----- STAGE 0: SELF-PROMPTING -----{E}\n")
    guide_key, sum_key, desc_key, label, *_ = _keys(task, resource)
    sys_prompt = prompts.get_system_prompt("selfprompt")
    summary = ""
    try: guide = prompts.get_codeknowledge(guide_key)
    except (KeyError, FileNotFoundError) as exc:
        print(f"  {B}WARN:{E} no {label}: {exc}"); guide = ""
    if guide:
        print(f"  Step 1/2: Summarizing {label}...")
        summary = _llm.generate_text(model, sys_prompt,
            prompts.get_selfprompt(sum_key) + f"\n\n[BEGIN GUIDE]\n{guide}\n[END GUIDE]",
            intent=f"{label} summarization", verifyprompt=verifyprompt,
            label=f"[LLM] Summarizing {label}")
    else:
        print(f"  Step 1/2: Skipping guide summary.")
    print("  Step 2/2: Analyzing source code...")
    desc = _llm.generate_text(model, sys_prompt,
        prompts.get_selfprompt(desc_key) + f"\n\n[BEGIN SOURCE CODE]\n{input_code}\n[END SOURCE CODE]",
        intent="source code analysis", verifyprompt=verifyprompt,
        label="[LLM] Analyzing source code")
    print(f"\n  {B}{G}✓{E}  Self-prompting complete.\n")
    return summary, desc


def build_stage1_prompt(*, prompts, task, resource, input_code,
                        guide_summary, source_description,
                        baseline_energy_j=-1.0, baseline_power_w=-1.0):
    """Stage 1: assemble (system_prompt, user_prompt) for the initial generation."""
    _, _, _, _, sys_key, usr_key, _, _, tag = _keys(task, resource)
    sys_prompt = prompts.get_system_prompt(sys_key)
    body = prompts.get_codetranslate_prompt(usr_key)
    pre = _preamble(guide_summary, source_description)
    think = "Think carefully, using your analysis above.  Now, " if source_description else ""
    user = pre + think + body + f"\n\n[START OF {tag}]\n{input_code}\n[/END OF {tag}]"
    if task in ("energy", "sycl_energy") and baseline_energy_j > 0:
        user += (f"\n\nFor reference, the {tag} provided above was executed on this system "
                 f"and consumed a total energy of {baseline_energy_j:.2f} Joules at an average power "
                 f"of {baseline_power_w:.2f} Watts, which includes the idle system energy "
                 f"consumption and average power.\n")
    return sys_prompt, user


def build_correction_prompt(*, prompts, task, resource, code_block,
                            compile_err, run_err, run_stdout, input_code, attempt,
                            guide_summary="", source_description="",
                            best_code_block="", opt_power_w=-1.0, opt_dur_s=-1.0):
    """Stage 2: classify the error and build the correction prompt.
    Returns (err_type_label, user_prompt) — err_type drives the temperature ratchet."""
    *_, base_key, improve_key, _ = _keys(task, resource)

    if compile_err == "execute_error":
        err_type, intro, key = "runtime error", "\n -- Runtime error: ", base_key
        msg = run_err + (("\n\nProgram stdout before crash:\n" + run_stdout) if run_stdout.strip() else "")
        msg += f"\n\n[ORIGINAL SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL SOURCE]\n"
    elif compile_err.startswith("VERIFICATION MISMATCH"):
        err_type, intro, key, msg = "checksum mismatch", "\n -- Verification mismatch: ", base_key, compile_err
    elif compile_err.startswith("ENERGY_KEEP_IMPROVING") and improve_key:
        err_type, intro, key = "energy keep improving", "\n -- Energy result: ", improve_key
        msg = compile_err
        if opt_power_w > 0 and opt_dur_s > 0:
            msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        if best_code_block: code_block = best_code_block
    elif compile_err.startswith("ENERGY_NOT_IMPROVED") and improve_key:
        err_type, intro, key = "energy improvement insufficient", "\n -- Energy result: ", improve_key
        msg = compile_err
        if opt_power_w > 0 and opt_dur_s > 0:
            msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        if best_code_block: code_block = best_code_block
    else:
        err_type, intro, key, msg = "compile error", "\n -- Compile error: ", base_key, compile_err

    base = prompts.get_code_error_correction_prompt(key)
    if err_type in ("energy improvement insufficient", "energy keep improving"):
        base = _preamble(guide_summary, source_description) + base
    user = (base + f"\n\n[START OF CODE TO FIX]\n{code_block}\n[/END OF CODE TO FIX]"
            + intro + "\n" + msg)
    return err_type, user


def run_judge(*, model, system_prompt, source_code, source_stdout, source_metrics,
              refactored_code, refactored_stdout, refactored_metrics):
    """LLM-as-judge: ask the model to decide functional equivalence.
    Returns "VALID" / "INVALID" / "NONE"."""
    print(f"\n  {B}[Judge] Verifying functional equivalence...{E}")
    prompt = (
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
        f"\n\n## [STANDARD OUTPUT]\n{source_stdout}\n\n## [METRICS]\n{source_metrics}"
        f"\n\n# [REFACTORED CODE]\n{refactored_code}"
        f"\n\n## [STANDARD OUTPUT]\n{refactored_stdout}\n\n## [METRICS]\n{refactored_metrics}"
    )
    response = _llm.generate_text(model, system_prompt, prompt,
                                  intent="code validation judge",
                                  label="[LLM] Judge", temperature=0.2)
    verdict = "INVALID" if "INVALID" in response else ("VALID" if "VALID" in response else "NONE")
    print(f"  {B}[Judge] verdict: {verdict}{E}")
    return verdict


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
