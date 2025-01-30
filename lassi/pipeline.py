import os
import re
import time
from datetime import datetime
import lassi.code_manager
import lassi.llm_methods
import lassi.token_analysis
import lassi.llm_config
import lassi.get_power
from lassi.format_colors import color


def auto_code_llmgeneration_pipeline(prompts,
                                     this_llm_systemcode_prompt,
                                     this_llm_prompt,  # The refactoring instruction prompt
                                     llm_source, 
                                     model,
                                     model_coder,
                                     num_ctx,
                                     num_ctx_coder,
                                     # arrayormatrix_size_range,
                                     # thread_count_range,
                                     use_context,
                                     input_context,
                                     input_code,
                                     # target_code,
                                     target_lang,
                                     refactored_code_extension,
                                     source_lang,
                                     code_compiler,
                                     code_compiler_kwds,
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
                                     yes_power_feedback
                                    ):
    
    # Test compile and execute the source code to validate compiler and execution environment
    actual_energy = 0
    actual_ave_power = 0
    idle_energy_joules = 0
    idle_ave_power = 0
    duration_refactor = 0
    
    if test_code_first:
        code_source = "source"  # generated
        print("\n\n----- " + color.BLUE + "SOURCE CODE: Test compile and execute" + color.END + " -----")
        print("\nTest compile the original source code: " + str(source_file_path))
        compile_result, execute_file = lassi.code_manager.compile_code(source_file_path, source_lang, code_compiler, code_compiler_kwds)
        if compile_result != "":
            return False
        if yes_execute:
            print("\nTest executing the compiled original source code: " + str(execute_file))
            exe_file_path = execute_file  # directory + "/" + execute_file
            execute_result, exe_output_source, total_energy_joules, ave_power, duration = lassi.code_manager.execute_code(exe_file_path, exe_input_parameters, power_measure, code_source, target_lang)
            # Save source code energy metrics:
            sourcecode_total_energy_joules = total_energy_joules
            sourcecode_ave_power = ave_power
            if execute_result != "0":
                return False
            print("\n\nMeasure system idle power...")
            idle_energy_joules, idle_ave_power = lassi.get_power.idlepower(duration)
            actual_energy = sourcecode_total_energy_joules - idle_energy_joules
            actual_ave_power = sourcecode_ave_power - idle_ave_power
    
    print("\n\n")

    compile_result = "start"
    content = ""
    execute_result = ""
    error_message = ""
    error_source = ""  # "compiled" or "executed"
    code_block = ""
    self_correction_counter = -1
    self_correction_power_counter = -1

    general_system_prompt = prompts.get_system_prompt("general_system_ee")

    print("\n\n----- " + color.BLUE + "STARTING LLM CONTEXT BUILDING WITH SELF-PROMPTING" + color.END + " -----\n\n")
    
    # Augmented prompting to handle processing of provided knowledge context
    if use_context:
        
        # Inject provided knowledge context--must be provided.
        content = "[BEGIN CONTEXT KNOWLEDGE] " + input_context + " [/END CONTEXT KNOWLEDGE]\n\n "
    
        # Request LLM summarization of provided context knowledge
        print("\n--> ... prompt LLM to summarize context knowledge.")
        llm_code_knowledge_summary_response = lassi.llm_methods.code_knowledge_llm(llm_source, model, num_ctx, general_system_prompt, input_context, target_lang, display_response)
        content = content + "[BEGIN CONTEXT KNOWLEDGE SUMMARY] " + llm_code_knowledge_summary_response + " [/END CONTEXT KNOWLEDGE SUMMARY]\n "
        
        content = content + "Using the above background context knowledge along with your summarization of this content, "

        # Requet LLM summary description of the provided source code
        print("\n--> ... prompt LLM to describe the source code.")
        llm_code_description_response = lassi.llm_methods.code_description_llm(llm_source, model, num_ctx, general_system_prompt, input_code, display_response)
    
        content = content + "think carefully before refactoring the source code that you describe as: \n"

        content = content + llm_code_description_response
    else:
        # Not using context
        content = general_system_prompt + "Think carefully before developing the following request: "

    # Save the original context prompt for re-use during error correction.
    original_context_content_prompt = content
    original_prompt_tokens = lassi.token_analysis.token_count(original_context_content_prompt)
    
    # Append the source code to the prompt
    content = content + " \n\n [BEGIN SOURCE CODE]\n" + input_code + "\n[/END SOURCE CODE]"
    
    content = content + "\n\n [BEGIN ENERGY EFFICIENCY REFACTORING TASK] Now, " + this_llm_prompt
    content = content + "\n\nFor reference, the source code provided above was executed on this system and consumed a total energy of " + str(round(actual_energy, 2)) + " Joules at an average power of " + str(round(actual_ave_power, 2)) + " Watts.\n"
    content = content + "[/END ENERGY EFFICIENCY REFACTORING TASK]\n"
     
    # Save the original context + energy efficiency context for re-use during energy efficiency correction
    original_ee_context_content_prompt = content
    original_ee_prompt_tokens = lassi.token_analysis.token_count(original_ee_context_content_prompt)

    print("\n\n----- " + color.BOLD + "REVIEW THE FULL TASK PROMPT " + color.END + " (initial, token count: " + str(original_ee_prompt_tokens) + "): \n\n" + content + "\n")
    
    print("\n\n----- " + color.BLUE + "STARTING CODE GENERATION PIPELINE" + color.END + " -----\n\n")

    # CHANGE MODEL FOR CODE GENERATION
    print("(... changing LLM for code gen)")
    
    # Unload model first if we want a different coding model
    if model_coder != model:
        lassi.llm_config.unload_ollama_model(model)
    
    start_time = time.time()
    
    # INITIATE LLM CODE GENERATION ITERATIVE PIPELINE
    while compile_result != "" and self_correction_counter <= 7 and self_correction_power_counter <= 7:
        self_correction_counter = self_correction_counter + 1
        
        if compile_result == "start":
            # CALL THE LLM
            content = re.sub(' +', ' ', content)
            # DONT PRINT THE FULL PROMPT: print("\n\n----- " + color.BOLD + "PROMPT" + color.END + " (initial, token count: " + str(original_prompt_tokens) + "): \n" + content + "\n")
            print("\n--> ... prompt the LLM to generate code.")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, this_llm_systemcode_prompt, content, display_response)
        elif compile_result != "energy_efficiency":
            # CALL THE LLM WITH ERROR CORRECTION
            print("\n\n----- " + color.BOLD + color.RED + "SELF-CORRECTION: " + str(self_correction_counter) + color.END + " -----")
            if compile_result == "execute_error":
                error_correction_prompt_intro = "\n -- The above code was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and produced the following execution error: "
                error_message = execute_result
            else:  # Must be a compiler error
                # NOTE: Maybe do not provide the original code in the other language when trying to fix code in the target language.
                error_correction_prompt_intro = "\n -- The above code was compiled with " + code_compiler + " " + code_compiler_kwds + " and produced the following compile error: "
                error_message = compile_result
            
            # error_correction_prompt_outro = " Eliminate this specific error by re-factoring the above code with an appropriate fix while also using the following background context: " + original_context_content_prompt
            # error_prompt = code_block + error_correction_prompt_intro + error_message + error_correction_prompt_outro
            # error_correction_prompt_outro = " Eliminate this specific error by re-factoring the above code with an appropriate fix."
            error_correction_prompt_outro = " Re-factor the above code with a fix to eliminate the stated error. Only provide the final code in your response."
            error_prompt = original_context_content_prompt + " [BEGIN CODE BLOCK TO REFACTOR] " + code_block + " [END CODE BLOCK TO REFACTOR] " + "\n\n" + error_correction_prompt_intro + "\n\n" + error_message + "\n\n" + error_correction_prompt_outro
            error_prompt_tokens = lassi.token_analysis.token_count(error_prompt)
            print("\n\n----- " + color.BOLD + color.RED + "ERROR SELF-CORRECTION" + color.END + " -----")
            print("\nPROMPT (error correction, token count: " + str(error_prompt_tokens) + "): " + error_prompt[0:1000] + "... \n")
            error_prompt = re.sub('\n', '', error_prompt)
            print("\n--> ... prompt LLM to generate code following an error.")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, this_llm_systemcode_prompt, error_prompt, display_response)

        elif compile_result == "energy_efficiency":
            '''Self-correction feedback for improving energy efficiency.'''
            print("\n\n----- " + color.BOLD + color.RED + "SELF-CORRECTION (EE): " + str(self_correction_power_counter) + color.END + " -----")
            ee_correction_prompt_intro = "\n -- The above code was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and consumed " + str(round(total_energy_joules, 2)) + " joules of energy with an average power of " + str(round(ave_power, 2)) + " Watts."
            ee_correction_prompt_outro = " Please re-factor the above code with optimization techniques to reduce total energy consumption during execution. Only provide the final code in your response."
            ee_prompt = original_ee_context_content_prompt + " [BEGIN CODE BLOCK TO REFACTOR] " + code_block + " [END CODE BLOCK TO REFACTOR] " + "\n\n" + ee_correction_prompt_intro + "\n\n" + ee_correction_prompt_outro
            ee_prompt_tokens = lassi.token_analysis.token_count(ee_prompt)
            print("\n\n----- " + color.BOLD + color.RED + "ENERGY EFFICIENCY SELF-CORRECTION" + color.END + " -----")
            print("\nPROMPT (energy efficiency correction, token count: " + str(ee_prompt_tokens) + "): " + ee_prompt[0:1000] + "... \n")
            ee_prompt = re.sub('\n', '', ee_prompt)
            print("\n--> ... prompt LLM to generage code for energy efficiency.")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, this_llm_systemcode_prompt, ee_prompt, display_response)        

        if code_block.startswith('c++') or code_block.startswith('cpp'):
            code_block = code_block[3:]
        if code_block.startswith('c'):
            code_block = code_block[1:]    
    
        # SAVE THE GENERATIVE CODE -- before compiling
        file_path = directory + "/" + file_name + "." + refactored_code_extension
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(code_block)
        file.close()
    
        # Compile the generated code, compile_result == "" if no error
        compile_result, execute_file = lassi.code_manager.compile_code(file_path, target_lang, code_compiler, code_compiler_kwds)

        print("\nExecute file: " + str(execute_file))

        if yes_execute and compile_result == "" and self_correction_counter <= 7:
            code_source = "generated"  # target
            # Unload any previously loaded LLMs in the Ollama server 
            # (clear out the memory)
            if llm_source == "ollama":
                lassi.llm_config.unload_ollama_model(model)
            
            exe_file_path = execute_file  # directory + "/" + execute_file
            execute_result, exe_output_generated, total_energy_joules, ave_power, duration_refactor = lassi.code_manager.execute_code(exe_file_path, exe_input_parameters, power_measure, code_source, target_lang)
            if execute_result != "0":
                compile_result = "execute_error"  # just a flag

        # Begin POWER METRICS feedback loop
        if yes_power_feedback and yes_execute and compile_result == "" and execute_result == "0" and self_correction_power_counter <= 7:
            self_correction_power_counter = self_correction_power_counter + 1
            # Substract out idle metrics
            total_energy_joules = total_energy_joules - idle_energy_joules
            ave_power = ave_power - idle_ave_power
            print("Power and Energy Feedback data: ")
            print("  sourcecode_total_energy_joules: " + str(sourcecode_total_energy_joules))
            print("  total_energy_joules: " + str(total_energy_joules))
            print("  sourcecode_ave_power: " + str(sourcecode_ave_power))
            print("  ave_power: " + str(ave_power))
            print("  runtime_sourcecode: " + str(duration))
            print("  runtime_refactored: " + str(duraction_refactor))
                                    
            if total_energy_joules < sourcecode_total_energy_joules and ave_power < sourcecode_ave_power:
                print("\n\n***** Generated refactored code is more energy efficient than source code! *****\n\n")
                break
            else:
                print("\n\n -- ENERGY EFFICIENCY REVIEW: Let's provide feedback on the energy consumption. --")
                compile_result = "energy_efficiency"
            '''
            What's the strategy? The LLM was already provided the original code power.
            So, if the generated code power is < than original code power, then exit this loop and finish.
            Else, prompt the LLM again with the original code power, its generated code power, and a refactor again,
            restating the goal that it is should try to reduce the power consumption on the system.
            ... Source code energy variable: sourcecode_total_energy_joules
            ... Source code average power: sourcecode_ave_power
            ... Generated code energy varaible: total_energy_joules
            ... Generated code average power: total_energy_joules
            + We also need to include details of the system in the context.
            + We also need to include better context for coding best practices.
            + Continue this self-feedback loop incrementing self_correction_power_counter until maxing at 7 tries.
            '''
                
    # end of while pipeline loop

    end_time = time.time() - start_time

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y_t%H_%M_%S")
    
    print("\n-----" + color.GREEN + "CODE GENERATION COMPLETE" + color.END + "-----\n")
    end_time_minutes = round(end_time/60, 2)
    print("CODE GENERATION TIME: " + str(end_time_minutes) + " minutes.\n")
    print("Self-correction count: " + str(self_correction_counter) + "\n")

    # MEASURE SIMILARITY BETWEEN GENERATED TRANSLATION CODE AND ORIGINAL TARGET CODE
    # similarity_tokens = lassi.token_analysis.token_similarity(target_code, code_block, "tokenize")
    # print(f"Token similarity (tokenize): {similarity_tokens:.2f}")

    # similarity_tokens_tiktoken = lassi.token_analysis.token_similarity(target_code, code_block, "tiktoken")
    # print(f"Token similarity (tiktoken): {similarity_tokens_tiktoken:.2f}")
    
    # similarity_lines = lassi.token_analysis.compare_lines_with_reordering(target_code, code_block)
    # print(f"Line-based similarity with reordering: {similarity_lines:.2f}")

    meta_data_file += "\n\n"
    meta_data_file += "----- METRICS -----"
    meta_data_file += "\n"
    meta_data_file += "Experiment date-time: " + str(dt_string)
    meta_data_file += "\n"
    meta_data_file += "Time (sec): " + str(round(end_time, 2))
    meta_data_file += "\n"
    # meta_data_file += f"Token similarity (tokenize): {similarity_tokens:.2f}"
    # meta_data_file += "\n"
    # meta_data_file += f"Token similarity (tiktoken): {similarity_tokens_tiktoken:.2f}"
    # meta_data_file += "\n"
    # meta_data_file += f"Line-based similarity with reordering: {similarity_lines:.2f}"
    # meta_data_file += "\n"
    meta_data_file += "Self-correction iteration count: " + str(self_correction_counter)
    if yes_execute:
        meta_data_file += "\n\n\n----- SOURCE: EXECUTION OUTPUT -----"
        meta_data_file += exe_output_source
        meta_data_file += "\n\n\n----- GENERATED: EXECUTION OUTPUT -----"
        meta_data_file += exe_output_generated
    if yes_power_feedback:
        meta_data_file += "\n\n\n----- ENERGY EFFICIENCY METRICS -----"
        meta_data_file += "\n"
        meta_data_file += "Total Energy (J): "
        meta_data_file += "  Source code: " + str(round(sourcecode_total_energy_joules, 2))
        meta_data_file += "  Refactored code: " + str(round(total_energy_joules, 2))
        meta_data_file += "\n"
        meta_data_file += "Average Power (W): "
        meta_data_file += "  Source code: " + str(round(sourcecode_ave_power, 2))
        meta_data_file += "  Refactored code: " + str(round(ave_power, 2))

    # Write meta data for this experiment
    filename = directory + "/" + "metadata" + "_" + file_name + "__" + str(dt_string) + ".txt"
    with open(filename, 'w') as file:
        file.write(meta_data_file)
    file.close()

    print("Metadata file: " + str(filename))

    return True