"""ALCF Inference Endpoints backend.

This module provides access to ALCF's inference endpoints using the OpenAI-compatible API.
Requires Globus authentication token.

Documentation: https://docs.alcf.anl.gov/ai-testbed/inference-endpoints/
"""

import json
import os
import subprocess
import sys


def get_alcf_token():
    """Get ALCF access token using inference_auth_token.py helper."""
    # Check if token helper exists
    token_helper_path = os.path.expanduser("~/inference_auth_token.py")

    if not os.path.exists(token_helper_path):
        raise RuntimeError(
            "ALCF token helper not found. Please download it:\n"
            "wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py\n"
            "Then authenticate: python inference_auth_token.py authenticate"
        )

    # Import the authentication module directly instead of subprocess
    # This ensures it uses the same Python environment (venv)
    try:
        # Add the helper's directory to Python path
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference_auth_token", token_helper_path)
        auth_module = importlib.util.module_from_spec(spec)
        sys.modules["inference_auth_token"] = auth_module
        spec.loader.exec_module(auth_module)

        # Get the access token
        token = auth_module.get_access_token()

        if token:
            return token
        else:
            raise RuntimeError("Empty token returned")

    except ImportError as e:
        raise RuntimeError(f"Failed to import authentication module: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to get ALCF token: {e}")


def alcf_llm(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
    cluster: str = "sophia",
) -> str:
    """Call ALCF Inference Endpoints.

    Args:
        model: Model name (e.g., "meta-llama/Llama-3.3-70B-Instruct")
        system_prompt: System prompt
        content_prompt: User prompt
        temp: Temperature (0.0-1.0)
        verifyprompt: Whether to print the prompt
        cluster: Which cluster to use ("sophia" or "metis")

    Returns:
        str: Model response
    """

    # Cluster endpoints
    endpoints = {
        "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1/chat/completions",
        "metis": "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions"
    }

    if cluster not in endpoints:
        raise ValueError(f"Unknown cluster: {cluster}. Choose 'sophia' or 'metis'")

    url = endpoints[cluster]

    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"CLUSTER: {cluster}")
        print(f"MODEL: {model}")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")

    # Get authentication token
    try:
        token = get_alcf_token()
    except Exception as e:
        raise RuntimeError(f"Authentication failed: {e}")

    # Build request payload (OpenAI format)
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt}
        ],
        "temperature": round(temp, 1),
        "max_tokens": 4096,
    }

    payload = json.dumps(data)

    # Build curl command string (used both directly and via SSH)
    # Shell-escape the payload for safe passing through SSH
    payload_escaped = payload.replace("'", "'\\''")
    curl_cmd = (
        f"curl -sS --connect-timeout 10 --max-time 60"
        f" -X POST '{url}'"
        f" -H 'Authorization: Bearer {token}'"
        f" -H 'Content-Type: application/json'"
        f" -d '{payload_escaped}'"
    )

    # Detect if on compute node (hostname starts with 'x' on Aurora)
    hostname = subprocess.run(["hostname"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True).stdout.strip()
    on_compute_node = hostname.startswith('x')

    if on_compute_node:
        # SSH to login node to execute the curl command
        # PBS_O_HOST is the node the job was submitted from (a login node)
        login_node = os.environ.get("PBS_O_HOST", "aurora-uan-0009.hostmgmt.cm.aurora.alcf.anl.gov")
        print(f"[Compute node detected] Routing API call via login node: {login_node}")
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",  # Fail immediately if password needed
            login_node,
            curl_cmd
        ]
        timeout = 80
    else:
        cmd = ["bash", "-c", curl_cmd]
        timeout = 70

    # Execute request
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ALCF API call timed out after {timeout} seconds")

    if proc.returncode != 0:
        raise RuntimeError(f"ALCF API call failed (rc={proc.returncode}): {proc.stderr.strip()}")

    # Parse JSON response
    try:
        response = json.loads(proc.stdout)

        # Extract message content
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            return content
        else:
            raise RuntimeError(f"Unexpected response format: {proc.stdout[:200]}")

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON response: {e}\nResponse: {proc.stdout[:500]}")


# Wrapper to match the argo_llm interface
def argo_llm(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
    uri: str = "remote",
) -> str:
    """Drop-in replacement for argo_llm that uses ALCF endpoints.

    Set ALCF_CLUSTER environment variable to choose cluster (default: sophia)
    Set ALCF_MODEL to override the model (default: meta-llama/Llama-3.3-70B-Instruct)
    """
    cluster = os.environ.get("ALCF_CLUSTER", "sophia")

    # Map common model names to ALCF models
    model_mapping = {
        "gpt4o": "meta-llama/Llama-3.3-70B-Instruct",
        "gpt52": "meta-llama/Llama-3.3-70B-Instruct",
        "gpt4turbo": "meta-llama/Llama-3.1-70B-Instruct",
        "gpt35": "meta-llama/Llama-3.1-8B-Instruct",
    }

    # Allow override via environment
    alcf_model = os.environ.get("ALCF_MODEL")
    if alcf_model:
        actual_model = alcf_model
    elif model in model_mapping:
        actual_model = model_mapping[model]
    else:
        # Assume it's already an ALCF model name
        actual_model = model

    return alcf_llm(actual_model, system_prompt, content_prompt, temp, verifyprompt, cluster)
