"""ALCF Inference Endpoints backend.

This module provides access to ALCF's inference endpoints using the OpenAI-compatible API.
Requires Globus authentication token via the globus_env conda environment.

Documentation: https://docs.alcf.anl.gov/ai-testbed/inference-endpoints/
"""

import json
import os
import subprocess


def alcf_llm(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
    cluster: str = "metis",
) -> str:
    """Call ALCF Inference Endpoints.

    Args:
        model: Model name (e.g., "gpt-oss-120b")
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

    # Shell-escape the payload for safe passing through bash/SSH
    payload_escaped = payload.replace("'", "'\\''")

    # The curl command uses $access_token (set by the shell command below)
    curl_part = (
        f"curl -sS --connect-timeout 10 --max-time 60"
        f" -X POST '{url}'"
        f" -H \"Authorization: Bearer $access_token\""
        f" -H 'Content-Type: application/json'"
        f" -d '{payload_escaped}'"
    )

    # Conda env to use (can be overridden via ALCF_CONDA_ENV env var)
    conda_env = os.environ.get("ALCF_CONDA_ENV", "globus_env")

    # Shell script to run on the login node.
    # Uses 'bash -l' (login shell) so .bash_profile is sourced, which initializes
    # conda - cannot use 'source ~/.bashrc' in non-interactive SSH sessions because
    # .bashrc has an early-return guard for non-interactive shells.
    # The script is passed via stdin to avoid quoting issues with the JSON payload.
    shell_script = "\n".join([
        f"conda activate {conda_env} 2>/dev/null || true",
        # Try 'python3' first, fall back to 'python'
        "PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)",
        'if [ -z "$PYTHON_CMD" ]; then echo "ERROR: no python found" >&2; exit 1; fi',
        'access_token=$($PYTHON_CMD $HOME/inference_auth_token.py get_access_token)',
        'if [ -z "$access_token" ]; then echo "ERROR: empty token from inference_auth_token.py" >&2; exit 1; fi',
        curl_part,
    ])

    # Detect if on compute node (hostname starts with 'x' on Aurora)
    hostname = subprocess.run(
        ["hostname"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    ).stdout.strip()
    on_compute_node = hostname.startswith('x')

    if on_compute_node:
        # SSH to login node and run a login shell there, feeding the script via stdin.
        # Login shell (bash -l) reads .bash_profile which initializes conda.
        # BatchMode=yes ensures we fail immediately if SSH keys aren't set up.
        login_node = os.environ.get("PBS_O_HOST", "aurora-uan-0009.hostmgmt.cm.aurora.alcf.anl.gov")
        print(f"[Compute node detected] Routing API call via login node: {login_node}")
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            login_node,
            "bash", "-l",
        ]
        timeout = 90
    else:
        # Already on login node - run login shell directly
        cmd = ["bash", "-l"]
        timeout = 80

    # Execute request, passing the script via stdin to avoid quoting issues
    try:
        proc = subprocess.run(
            cmd,
            input=shell_script,
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

    Environment variable overrides:
      ALCF_CLUSTER   - cluster to use (default: metis)
      ALCF_MODEL     - override model name
      ALCF_CONDA_ENV - conda env with globus_sdk (default: globus_env)
    """
    cluster = os.environ.get("ALCF_CLUSTER", "metis")

    # Map common Argo model names to ALCF models
    # gpt-oss-120b on metis is confirmed working
    model_mapping = {
        "gpt4o": "gpt-oss-120b",
        "gpt52": "gpt-oss-120b",
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
