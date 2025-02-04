import ollama
from ollama import ResponseError
import json
import subprocess
import re
import html
from IPython.display import display, HTML, clear_output

"""
Local hosted LLM via OLLAMA API, GGUF quantized models

Requirement: ollama serve [from server command line]
Requirement: ollama.pull('modelname') [from Python script to pre-load selected model at the server]

ollama.list() [preview what is available to use on the server]

Example: model='codestral:latest'
"""


def ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt):
    response_text = ""

    # Ollama serve must already be running on a port on this server 
    response = ollama.chat(
        model=model, 
        messages=[
            {"role": "system", "content": system_prompt},
            {'role': 'user', 'content': content_prompt}
        ],
        stream=False,
        options=dict(
            temperature=0.2,
            top_p=0.9,
            num_ctx=num_ctx
        )
    )

    response_text = response['message']['content']

    print(response_text)
    
    # Final cleaning of the complete response
    # Conditional cleaning only if <think> tags exist
    if re.search(r'<think>.*?</think>', response_text, flags=re.DOTALL):
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    else:
        cleaned_response = response_text.strip()
    
    return cleaned_response  # response_text


def ollama_llm(model, num_ctx, system_prompt, content_prompt):
    # Create an HTML display object with a <pre> tag for formatted text
    html_display = display(HTML(""), display_id=True)

    response_text = ""

    # Ollama serve must already be running on a port on this server 
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt}
        ],
        stream=True,
        options=dict(
            temperature=0.2,
            top_p=0.9
            # num_ctx=num_ctx  # including this parameter seems to slow down the streaming response
        ),
        keep_alive=0
    )

    # Process each chunk in the stream
    try:
        for chunk in stream:
            chunk_content = chunk['message']['content']
            response_text += chunk_content
    
            # Escape HTML and highlight <think> tags
            escaped_text = html.escape(response_text)
    
            # Style opening tags
            styled_text = re.sub(
                r'&lt;think&gt;',
                '<span style="color: #d32f2f; font-weight: bold;">&lt;think&gt;</span>',
                escaped_text
            )
            # Style closing tags
            styled_text = re.sub(
                r'&lt;/think&gt;',
                '<span style="color: #d32f2f; font-weight: bold;">&lt;/think&gt;</span>',
                styled_text
            )
            
            # Create typewriter effect with cursor
            display_text = f"{styled_text}<span style='border-right:2px solid;animation:blink 1s infinite'> </span>"  
            html_display.update(HTML(f"""
                <pre style='background:#f8f8f8;padding:10px;border-radius:5px;border:1px solid #ddd;
                              white-space:pre-wrap;font-family:monospace'>
                {display_text}
                </pre>
                <style>
                @keyframes blink {{
                    0% {{ opacity:1 }}
                    50% {{ opacity:0 }}
                    100% {{ opacity:1 }}
                }}
                </style>
            """))
    except ResponseError as e:
        if "unexpected EOF" in str(e):
            # Log the event, check if the output seems complete, and continue
            print("Stream ended unexpectedly but continuing with received output.")
        else:
            raise
    
    # Clear the streaming display after completion
    # clear_output(wait=True)

    # Final cleaning of the complete response
    # Conditional cleaning only if <think> tags exist
    if re.search(r'<think>.*?</think>', response_text, flags=re.DOTALL):
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    else:
        cleaned_response = response_text.strip()
    
    return cleaned_response  # response_text


def unload_ollama_model(model):
    print("\nUnloading Ollama server...")
    data_curl = {
        "model": model,
        "keep_alive": 0
    }
    data_curl_json = json.dumps(data_curl)
    curl_command_unload_LLM = [
        'curl',
        'http://localhost:11434/api/generate',
        '-d',
        data_curl_json
    ]
    result_unload_LLM = subprocess.run(curl_command_unload_LLM, capture_output=True, text=True)
    print("\nUnload LLM:\n", result_unload_LLM.stdout)
    print(result_unload_LLM.stderr)
