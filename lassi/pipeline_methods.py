import lassi.code_manager
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary

prompts = PromptDictionary()


def test_source_code(source_file_path,
                     build_script,
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
                     valid_code_history
                    ):
    """
    Compile, execute, and measure power profile of source code.
    """
    code_source = "source"  # generated
    print("\n\n----- " + color.BLUE + "SOURCE CODE: Test compile and execute" + color.END + " -----")
    print("\nTest compile the original source code: " + str(source_file_path))
    if build_script:
        compile_result = lassi.code_manager.build_script_code("source_codes", code_build_script)
        execute_file = "/home/cc/lassi/LASSI_dev/source_codes/altis/build/bin/level1/" + file_nameexe_input_parameters
    else:
        compile_result, execute_file = lassi.code_manager.compile_code(source_file_path, source_lang, code_compiler, code_compiler_kwds)
    
    if compile_result != "":
        print("\n\nFailed compile result: " + str(compile_result))
        return False
    if yes_execute:
        print("\nTest executing the compiled original source code: " + str(execute_file))
        exe_file_path = execute_file  # directory + "/" + execute_file
        execute_result, exe_output_source, total_energy_joules, ave_power, duration, metrics_results_string = lassi.code_manager.execute_code(exe_file_path, exe_input_parameters, power_measure, code_source, target_lang, build_script, wait=1)
        input_content_llmjudge_dict["source_stdout"] = exe_output_source
        input_content_llmjudge_dict["source_metrics"] = metrics_results_string
        # Save source code energy metrics:
        sourcecode_total_energy_joules = total_energy_joules
        sourcecode_ave_power = ave_power
        if execute_result != "0":
            return False
        print("\n\nMeasure system idle power...")
        idle_energy_joules, idle_ave_power = lassi.get_power.idlepower(duration)
        actual_energy = sourcecode_total_energy_joules - idle_energy_joules
        energy_history.append(round(sourcecode_total_energy_joules, 2))
        actual_ave_power = sourcecode_ave_power - idle_ave_power
        power_history.append(round(float(sourcecode_ave_power), 2))
        runtime_history.append(round(float(duration),2))
        valid_code_history.append("SOURCE")

    return (idle_energy_joules, idle_ave_power, input_content_llmjudge_dict, sourcecode_total_energy_joules, sourcecode_ave_power,
            duration, energy_history, power_history, runtime_history, valid_code_history)


def selfprompting_context_builder(generate_context_flag,
                                  source_lang=None,
                                  this_llm_prompt=None,
                                  input_context=None,
                                  llm_source=None,
                                  model=None,
                                  num_ctx=None,
                                  target_lang=None,
                                  display_response=None,
                                  use_context_summaries_only=None,
                                  input_code=None,
                                  total_energy_joules=None,
                                  ave_power=None
                                 ):
    """
    Update the system and user content prompts to include
    background context and self-prompted context.
    """
    content = ""
    original_context_knowledge = ""
   
    # Augmented prompting to handle processing of provided knowledge context
    if generate_context_flag:
        # Fetch system prompt when using context
        general_system_prompt = prompts.get_system_prompt("general_system_ee")
        # Inject provided knowledge context--must be provided from top-level LASSI framework script.
        content = "" + input_context + "\n\n"  # [BEGIN CONTEXT KNOWLEDGE]  [/END CONTEXT KNOWLEDGE]

        original_context_knowledge = content

        # Request LLM summarization of provided context knowledge
        # Note, the input_context object includes all context requested by the user in the LASSI framework.
        print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to summarize context knowledge." + color.END + "\n")
        llm_code_knowledge_summary_response = lassi.llm_methods.code_knowledge_llm(llm_source, model, num_ctx, general_system_prompt, input_context, target_lang, display_response, verifyprompt=True)
        
        if use_context_summaries_only:
            # Save only LLM summary as context
            # NOTE: this resets the content to not include the original context
            content = "[YOUR SUMMARY OF BACKGROUND KNOWLEDGE]\n" + llm_code_knowledge_summary_response + "\n\n"
        else:
            # Keep the original context in the content to be included in the LLM prompting
            content = content + "[YOUR SUMMARY OF BACKGROUND KNOWLEDGE]\n" + llm_code_knowledge_summary_response + "\n\n"  # [/END CONTEXT KNOWLEDGE SUMMARY]

        # Requet LLM summary description of the provided SOURCE code
        print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to describe the source code." + color.END + "\n")
        llm_code_description_response = lassi.llm_methods.code_description_llm(llm_source, model, num_ctx, general_system_prompt, input_context, input_code, display_response, verifyprompt=True)
        content = content + "[YOUR ANALYSIS OF THE SOURCE CODE TO REFACTOR]\n" + llm_code_description_response + "\n\n"
        # refactor_instructions = prompts.get_code_ee_refactoring_prompt(str(source_lang + "_refactor"))  # Includes prompt to generate a refactoring plan
          
    else:
        # For initial zero-shot attempt
        # Fetch system prompt when not using context
        general_system_prompt = prompts.get_system_prompt("general_system_nocontext_ee")
        content = ""  # "Think carefully before developing the code following these instructions."
        content = content + "\n\n" + prompts.get_code_ee_refactoring_prompt(str(source_lang + "_baseline_nocontext"))
        
    
    # Save the original context prompt for re-use during error correction.
    # original_context_content_prompt = content
    # This token count does not include the source code
    # original_prompt_tokens = lassi.token_analysis.token_count(original_context_content_prompt)
    
    return original_context_knowledge, content, general_system_prompt


def generate_refactoring_plan(source_lang,
                              input_code,
                              total_energy_joules,
                              ave_power,
                              llm_source,
                              model,
                              num_ctx,
                              system_prompt,
                              context_prompt,
                              display_response,
                              few_shot_learning=False
                             ):

    # Prompt structure: GOAL - RETURN FORMAT - WARNINGS - CONTEXT
    refactoring_plan_instructions = prompts.get_code_ee_refactoring_prompt(source_lang)
    # Prepend the Context:
    refactoring_plan_instructions = refactoring_plan_instructions + "\n\n" + context_prompt 
    code_to_refactor = "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]"
    code_to_refactor = code_to_refactor + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(total_energy_joules, 2)) + " Joules at an average power of " + str(round(ave_power, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
    refactoring_plan_instructions = refactoring_plan_instructions + code_to_refactor
    # Generate the refactoring plan
    refactoring_plan = "\n\n" + "[YOUR REFACTORING PLAN]\n" + lassi.llm_methods.llm_call("refactoring plan", llm_source, model, num_ctx, system_prompt, refactoring_plan_instructions, display_response, verifyprompt=True)
    # Append few-shot learning examples
    if few_shot_learning:
        refactoring_plan = refactoring_plan + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
    # refactoring_plan_original = refactoring_plan  # Save for self-correction iterations.
    # Append code to refactor
    refactoring_plan_full = refactoring_plan + code_to_refactor
        
    refactoring_plan_tokens = lassi.token_analysis.token_count(refactoring_plan_full)
    
    print("\n\n----- " + color.GREEN + "THE REFACTORING PLAN PROMPT " + color.END + " (prompt with code, token count: " + str(refactoring_plan_tokens) + "): \n")  # + refactoring_plan + "\n")

    return refactoring_plan, code_to_refactor