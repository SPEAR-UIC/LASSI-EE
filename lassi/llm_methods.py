from IPython.display import display, Markdown
import re
from lassi.llm_config import argo_llm, ollama_llm, ollama_llm_nostream
from lassi.format_colors import color


def llm_code_compare(prompts, input_content_llmjudge_dict, source_duration, source_total_energy_joules, source_ave_power, refactored_duration, refactored_total_energy_joules, refactored_ave_power, llm_source, model, num_ctx, system_prompt, display_response, verifyprompt, uri):
    print("\n\n----- " + color.BOLD + color.BLUE + "FINAL CODE COMPARISON: Source Code vs. Best Refactored Code" + color.END + " -----\n")
            
    compare_code = prompts.get_code_comparison("cuda_cpp")
    compare_code = compare_code + "\n\n" + "[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_content_llmjudge_dict["source_code"]
    compare_code = compare_code + "\n\nOUTPUT: \n\n" + input_content_llmjudge_dict["source_stdout"]
    compare_code = compare_code + "\n\nCode execution duration: " + str(round(source_duration, 2)) + " s"
    compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(source_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(source_ave_power,2)) + " W"
    compare_code = compare_code + "\n\n[/END SOURCE CODE AND RESULTS]"
            
    compare_code = compare_code + "\n\n" + "[BEGIN REFACTORED CODE AND RESULTS]\n\n" + input_content_llmjudge_dict["refactored_code"]
    compare_code = compare_code + "\n\nOUTPUT: \n\n" + input_content_llmjudge_dict["refactored_stdout"]
    compare_code = compare_code + "\n\nCode execution duration: " + str(round(refactored_duration, 2)) + " s"
    compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(refactored_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(refactored_ave_power,2)) + " W"
    compare_code = compare_code + "\n\n[/END REFACTORED CODE AND RESULTS]"
    
    # Send prompt to LLM endpoint
    print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "LLM prompt to compare codes." + color.END + "\n\n")
    code_comparison_response = llm_call("code comparison", llm_source, model, num_ctx, system_prompt, compare_code, display_response, verifyprompt, uri)

    return code_comparison_response

    
def llm_call(intent, llm_source, model, num_ctx, system_prompt, content_prompt, display_response, verifyprompt=False, uri="remote"):
   
    if intent == "refactoring plan":
        print("\n\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to write a refactoring plan." + color.END + "\n")
        
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, 0.2, verifyprompt, uri)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm(model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("\n\n----- " + color.BOLD + color.BLUE + "LLM RESPONSE: " + intent + color.END + " -----\n")
    if display_response:
        display(Markdown(response_text))

    return response_text


def llm_judge(llm_source, model, num_ctx, system_prompt, input_content_dict, power_measure, display_response, verifyprompt=False, uri="remote"):

    intent = "code validation judge"

    print("\n\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to judge the validity of the refactored code." + color.END + "\n")
    
    # Use same system prompt.
    # Build user prompt:
    # Fixed instructions first
    # "SOURCE CODE"
    # <insert source code>
    # "STANDARD OUTPUT
    # <insert runtime output>
    # "METRICS"
    # <insert measurements>
    # "REFACTORED CODE"
    # <insert refactored code>
    # "STANDARD OUTPUT
    # <insert runtime output>
    # "METRICS"
    # <insert measurements>
    content_prompt = """
    Please compare the difference between the following two codes written in CUDA C++.\n
    First, I will provide you first with the SOURCE CODE followed by its STANDARD OUTPUT and METRICS measured during code runtime.\n
    Next, I will provide you with a REFACTORED CODE followed by its STANDARD OUTPUT and METRICS measured during code runtime.\n
    After you have reviewed both codes and their output, while the REFACTORED CODE may run faster and potentially use less energy than the SOURCE CODE, it should still produce equivalent functionality and identical standard output in layout and content.\n
    Your task is to strictly evaluate whether the REFACTORED CODE preserves all intended functionality of the SOURCE CODE and produces equivalent standard output, including its layout and content, compared to the SOURCE CODE.\n
    Your careful evaluation should focus on functional equivalence of the two codes and nearly exact standard output of the two codes. Minor differences in numerical values are acceptable in the standard output only if they are consistent with the REFACTORED CODE improving energy efficiency and these specific values are reasonable and consistent with the SOURCE CODE's intended purpose.\n
    Please explain your reasoning, and then conclude your response with VALID if the refactored code is functionally equivalent, or INVALID if it is not.
    """
    # Please evaluate if you think the refactored code should be considered valid in terms of providing the same code functionality while also having the exact same standard output as compared with the the source code, other than the specific numerical values. However, these numerical values must still make sense for the given functionality and intent of the code for the refactored code to be considered valud. \n
    # If you think the refactored code is valid, then please conclude your response with VALID. Otherwise, conclude with INVALID.
    #"""

    content_prompt += f"\n\n# [SOURCE CODE]\n {input_content_dict['source_code']}"
    content_prompt += f"\n\n## [STANDARD OUTPUT]\n {input_content_dict['source_stdout']}"
    content_prompt += f"\n\n## [METRICS]\n {input_content_dict['source_metrics']}"
    content_prompt += f"\n\n# [REFACTORED CODE]\n {input_content_dict['refactored_code']}"
    content_prompt += f"\n\n## [STANDARD OUTPUT]\n {input_content_dict['refactored_stdout']}"
    content_prompt += f"\n\n## [METRICS]\n {input_content_dict['refactored_metrics']}"

    if power_measure:
        if input_content_dict['refactored_metrics'] == "":
        # No energy metrics measured. This refactored code should be flagged INVALID.
            return "INVALID"
    
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm(model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("\n\n----- " + color.BOLD + color.BLUE + "LLM RESPONSE: " + intent + color.END + " -----\n")
    if display_response:
        display(Markdown(response_text))

    print("\n\n----- " + color.BOLD + color.BLUE + "END LLM RESPONSE" + color.END + " -----\n")

    # look for "VALID" or "INVALID" in the LLM response
    # only return this value
    # NOTE: check "INVALID" first, because "VALID" is a substring
    if "INVALID" in response_text:
        response_validation = "INVALID"
    elif "VALID" in response_text:
        response_validation = "VALID"
    else:
        response_validation = "NONE"
    
    return response_validation

    
def code_knowledge_llm(llm_source, model, num_ctx, system_prompt, input_context, target_lang, display_response, verifyprompt=False):

    content_prompt = """
    Read and Analyze: First, read the following background knowledge.
    Then, summarize its key relevant points that we should apply to refactoring code for improving energy efficiency.
    """

    if target_lang == "OMP":
        content_prompt = content_prompt + """ 
        Specifically, describe how to implement the omp pragma directive 'target teams' for distributing for loops on a GPU.
        """
        
    # content_prompt = content_prompt + """ 
    # Only respond with your summary details and avoid repeating the provided source code: 
    # """

    content_prompt = content_prompt + "\n\n" + input_context
    
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm(model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("\n\n----- " + color.BOLD + color.BLUE + "CONTEXT KNOWLEDGE SUMMARY" + color.END + " -----")
    print(" (generated by the LLM)\n")
    if display_response:
        display(Markdown(response_text))

    return response_text


def code_description_llm(llm_source, model, num_ctx, system_prompt, input_context, input_code, display_response, verifyprompt=False):

    # content_prompt = """
    # Please provide a detailed summary describing the functionality of the following source code.  
    # Outline the function of the source code step-by-step to include all details so that 
    # you can later refactor this code to include all its features and functions. 
    # Only respond with your summary description and avoid repeating the provided source code: \n\n
    # [BEGIN SOURCE CODE] \n\n
    # """

    content_prompt = """
    Read and Analyze: First, read the following background knowledge.
    Then, carefully read the entire code provided below.
    Describe its complete functionality and summarize the functional blocks: list out all major classes, functions, and data structures, and briefly describe what they do.
    Only respond with your description and analysis. Avoid responding with any code changes or repeating the source code. 
    """
    
    content_prompt = content_prompt + "\n\n" + input_context
    content_prompt = content_prompt + "\n\n[START OF CODE TO DESCRIBE AND ANALYZE]\n" + input_code + "\n\n"  # [/END SOURCE CODE]"
    
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm(model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("\n\n----- " + color.BOLD + color.BLUE + "CODE DESCRIPTION and ANALYSIS" + color.END + " -----")
    print(" (generated by the LLM)\n")
    if display_response:
        display(Markdown(response_text))

    return response_text


def system_description_llm(llm_source, model, num_ctx, system_prompt, gpu_system_details, display_response, verifyprompt=False):

    # content_prompt = """
    # Please provide a detailed summary describing the functionality of the following source code.  
    # Outline the function of the source code step-by-step to include all details so that 
    # you can later refactor this code to include all its features and functions. 
    # Only respond with your summary description and avoid repeating the provided source code: \n\n
    # [BEGIN SOURCE CODE] \n\n
    # """

    content_prompt = """
    Below are the detailed system metrics of the GPU and related hardware used for executing CUDA C++ code. Your task is to analyze this information and provide a complete, clear description of the system that highlights any details relevant to assessing and improving energy efficiency. In your description, focus on key aspects such as power consumption, clock speeds, utilization, thermal metrics, PCIe configuration, and any CPU/memory details provided. Your description should included specific system details important as background context for the task of identifying inefficiencies and developing a refactoring plan for energy efficient CUDA C++ code.\n\nPlease respond with only your detailed descriptive summary of the system. Do not include any code examples or additional extraneous information.
    """
    
    content_prompt = content_prompt + "\n[CPU/GPU SYSTEM DETAILS TO DESCRIBE]\n\n" + gpu_system_details + "\n"
    
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, 0.2, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm(model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("\n\n----- " + color.BOLD + color.BLUE + "GPU SYSTEM DESCRIPTION" + color.END + " -----")
    print(" (generated by the LLM)\n")
    if display_response:
        display(Markdown(response_text))

    return response_text


def code_generator_llm(llm_source, model, num_ctx, temp, system_prompt, content_prompt, display_response, verifyprompt=False):

    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE GENERATION" + color.END + " -----\n")
    
    # print("\n\nDEBUG input parameter values:")
    # print("  llm_source: " + str(llm_source))
    # print("  model: " + str(model))
    # print("  num_ctx: " + str(num_ctx))
    # print("  system_prompt: " + str(system_prompt))
    # print("  content_prompt: " + str(content_prompt))
    # print("  display_response: " + str(display_response))
    
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
        
    print("\n\n----- " + color.BOLD + color.BLUE + "THE GENERATED CODE" + color.END + " -----\n")
    if display_response:
        print("(hidden)\n\n")
        # display(Markdown(response_text))

    # Extract only the code block - 
    # Search for the ``` pattern in the text
    #   Regular expression pattern to match text between triple backticks 
    #   that represent code blocks within generated responses from the LLM.
    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    # print("RESPONSE TEXT -- ", response_text)
    # print("MATCH -- ", match)
    if match:
        extracted_text = match.group(1)
    else:
        extracted_text = ""
    
    return response_text, extracted_text