"""LLM prompt assembly + the LLM-as-judge correctness check.

Three stages of the pipeline call into this module:
  Stage 0: run_selfprompt          — summarize context, analyze source
  Stage 1: build_stage1_prompt     — initial generation prompt
  Stage 2: build_correction_prompt — per-attempt error-correction prompt

System prompts and Stage 1/2 user prompts come from lassi/prompt_dictionary.py.
Stage 0 user-prompt wording is implemented as inline f-strings here (mirroring
base lassi/llm_methods.py).
"""
import lassi.llm_client as _llm
from lassi.format_colors    import B, G, E
from lassi.prompt_dictionary import PromptDictionary

# (task, resource) → keys + accessor info.  Energy task is Polaris-only.
_TASK_KEYS = {
    ("translate", "polaris"): dict(
        guide_key="sycl_programming", guide_label="SYCL Programming Guide",
        sys_key="CUDA_to_SYCL", usr_dict="translate", usr_key="CUDA_to_SYCL",
        err_key="SYCL_error", block_tag="CODE TO TRANSLATE",
        action_verb="translating"),
    ("translate", "aurora"):  dict(
        guide_key="sycl_programming", guide_label="SYCL Programming Guide",
        sys_key="CUDA_to_SYCL", usr_dict="translate", usr_key="CUDA_to_SYCL",
        err_key="SYCL_error", block_tag="CODE TO TRANSLATE",
        action_verb="translating"),
    ("energy", "polaris"): dict(
        guide_key="cuda_optimization_summary", guide_label="CUDA Optimization Guide",
        sys_key="CUDA_ee", usr_dict="energy", usr_key="CUDA_baseline_nocontext",
        err_key="CUDA_error", block_tag="CODE TO REFACTOR",
        action_verb="refactoring"),
}


def _keys(task, resource):
    """Look up the prompt-key dict for a (task, resource) combo."""
    return _TASK_KEYS[(task, resource)]


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
    User-prompt wording mirrors base lassi/llm_methods.py inline templates.
    Returns (guide_summary, source_description) — both empty if disabled."""
    if not use_self_prompting:
        return "", ""
    print(f"\n{B}----- STAGE 0: SELF-PROMPTING -----{E}\n")
    keys = _keys(task, resource)
    sys_prompt = prompts.get_system_prompt(keys["sys_key"])
    label, verb = keys["guide_label"], keys["action_verb"]
    summary = ""
    try: guide = prompts.get_codeknowledge(keys["guide_key"])
    except (KeyError, FileNotFoundError) as exc:
        print(f"  {B}WARN:{E} no {label}: {exc}"); guide = ""
    if guide and guide != "Key not found":
        print(f"  Step 1/2: Summarizing {label}...")
        sum_user = (f"Read and Analyze: First, read the following background knowledge. "
                    f"Then, summarize its key relevant points that we should apply to "
                    f"{verb} code for improving energy efficiency.\n\n{guide}")
        summary = _llm.generate_text(model, sys_prompt, sum_user,
            intent=f"{label} summarization", verifyprompt=verifyprompt,
            label=f"[LLM] Summarizing {label}")
    else:
        print(f"  Step 1/2: Skipping guide summary.")
    print("  Step 2/2: Analyzing source code...")
    desc_user = (
        "Read and Analyze: Carefully read the entire code provided below. "
        "Describe its complete functionality and summarize the functional blocks: "
        "list out all major classes, functions, and data structures, and briefly "
        "describe what they do.  Only respond with your description and analysis. "
        "Avoid responding with any code changes or repeating the source code."
        f"\n\n[START OF CODE TO DESCRIBE AND ANALYZE]\n{input_code}\n"
        "[/END OF CODE TO DESCRIBE AND ANALYZE]"
    )
    desc = _llm.generate_text(model, sys_prompt, desc_user,
        intent="source code analysis", verifyprompt=verifyprompt,
        label="[LLM] Analyzing source code")
    print(f"\n  {B}{G}✓{E}  Self-prompting complete.\n")
    return summary, desc


def build_stage1_prompt(*, prompts, task, resource, input_code,
                        guide_summary, source_description,
                        baseline_energy_j=-1.0, baseline_power_w=-1.0):
    """Stage 1: assemble (system_prompt, user_prompt) for the initial generation."""
    keys = _keys(task, resource)
    sys_prompt = prompts.get_system_prompt(keys["sys_key"])
    body = (prompts.get_codetranslate_prompt(keys["usr_key"])
            if keys["usr_dict"] == "translate"
            else prompts.get_code_ee_refactoring_prompt(keys["usr_key"]))
    pre = _preamble(guide_summary, source_description)
    think = "Think carefully, using your analysis above.  Now, " if source_description else ""
    tag = keys["block_tag"]
    user = pre + think + body + f"\n\n[START OF {tag}]\n{input_code}\n[/END OF {tag}]"
    if task == "energy" and baseline_energy_j > 0:
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
    Returns (err_type_label, user_prompt) — err_type drives the temperature ratchet.
    Base has no dedicated 'energy improve' prompt — stagnation reuses err_key."""
    err_key = _keys(task, resource)["err_key"]

    if compile_err == "execute_error":
        err_type, intro = "runtime error", "\n -- Runtime error: "
        msg = run_err + (("\n\nProgram stdout before crash:\n" + run_stdout) if run_stdout.strip() else "")
        msg += f"\n\n[ORIGINAL SOURCE FOR REFERENCE]\n{input_code}\n[/ORIGINAL SOURCE]\n"
    elif compile_err.startswith("VERIFICATION MISMATCH"):
        err_type, intro, msg = "checksum mismatch", "\n -- Verification mismatch: ", compile_err
    elif compile_err.startswith("ENERGY_KEEP_IMPROVING"):
        err_type, intro, msg = "energy keep improving", "\n -- Energy result: ", compile_err
        if opt_power_w > 0 and opt_dur_s > 0:
            msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        if best_code_block: code_block = best_code_block
    elif compile_err.startswith("ENERGY_NOT_IMPROVED"):
        err_type, intro, msg = "energy improvement insufficient", "\n -- Energy result: ", compile_err
        if opt_power_w > 0 and opt_dur_s > 0:
            msg += f"\n(avg power: {opt_power_w:.1f} W, duration: {opt_dur_s:.1f} s)"
        if best_code_block: code_block = best_code_block
    else:
        err_type, intro, msg = "compile error", "\n -- Compile error: ", compile_err

    base = prompts.get_code_error_correction_prompt(err_key)
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
