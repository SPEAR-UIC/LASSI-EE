import json
import os
import subprocess
import time


_MAX_RETRIES = 3
_RETRY_DELAYS = [30, 60]  # seconds between retries (exponential backoff)


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
    max_tokens: int = 32768,
) -> str:
    cluster = os.environ.get("ALCF_CLUSTER", "metis")
    endpoints = {
        "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1/chat/completions",
        "metis":  "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions",
    }
    if cluster not in endpoints:
        raise ValueError(f"Unknown cluster: {cluster!r}. Choose 'sophia' or 'metis'")
    url = endpoints[cluster]

    if verifyprompt:
        print(f"\n=====> PROMPT REVIEW <=====")
        print(f"CLUSTER: {cluster}  MODEL: {model}  TEMP: {round(temp, 1)}")
        print(f"\nSYSTEM:\n{system_prompt}")
        print(f"\nUSER:\n{user_prompt}")
        print("=====^^^^^^^^^^^^^^^^^=====\n")

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": round(temp, 1),
        "max_tokens": max_tokens,
    }
    payload = json.dumps(data)

    conda_env = os.environ.get("ALCF_CONDA_ENV", "globus_env")

    # Write the JSON payload to a temp file so we never hit ARG_MAX limits
    # with large prompts, and so no shell escaping of the payload is needed.
    shell_script = "\n".join([
        "module load frameworks 2>/dev/null",
        f'access_token=$(conda run -n {conda_env} python3 $HOME/inference_auth_token.py get_access_token)',
        'if [ -z "$access_token" ]; then echo "ERROR: empty token from inference_auth_token.py" >&2; exit 1; fi',
        'LASSI_TMP=$(mktemp /tmp/lassi_payload_XXXXXX.json)',
        "cat > \"$LASSI_TMP\" << 'LASSI_PAYLOAD_EOF'",
        payload,
        'LASSI_PAYLOAD_EOF',
        (
            f"curl -sS --connect-timeout 10 --max-time 300"
            f" -X POST '{url}'"
            f" -H \"Authorization: Bearer $access_token\""
            f" -H 'Content-Type: application/json'"
            f" -d @\"$LASSI_TMP\""
        ),
        'rm -f "$LASSI_TMP"',
    ])

    hostname = subprocess.run(
        ["hostname"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).stdout.strip()

    if hostname.startswith("x"):
        # On compute node — SSH to login node to access the internet.
        login_node = os.environ.get("PBS_O_HOST", "aurora-uan-0009.hostmgmt.cm.aurora.alcf.anl.gov")
        print(f"[Compute node detected] Routing API call via login node: {login_node}")
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
               login_node, "bash", "-l"]
        timeout = 360
    else:
        cmd = ["bash", "-l"]
        timeout = 340

    last_error: Exception = RuntimeError("No attempt made")
    for attempt in range(_MAX_RETRIES):
        try:
            proc = subprocess.run(
                cmd,
                input=shell_script,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_error = RuntimeError(f"ALCF API call timed out after {timeout} seconds")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Timeout on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

        if proc.returncode != 0:
            last_error = RuntimeError(f"ALCF API call failed (rc={proc.returncode}):\n  stderr: {proc.stderr.strip()[:500]}\n  stdout: {proc.stdout.strip()[:200]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Non-zero exit on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

        # Check for transient HTTP errors in response (e.g., "error code: 504")
        stdout_stripped = proc.stdout.strip()
        if not stdout_stripped or (stdout_stripped.startswith("error code:") and not stdout_stripped.startswith("{")):
            last_error = RuntimeError(f"ALCF endpoint returned transient error: {stdout_stripped[:200]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Transient error on attempt {attempt + 1}/{_MAX_RETRIES}: {stdout_stripped[:80]}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

        try:
            response = json.loads(proc.stdout)
            if "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"]
            last_error = RuntimeError(f"Unexpected response format: {proc.stdout[:200]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Unexpected format on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error
        except json.JSONDecodeError as e:
            last_error = RuntimeError(f"Failed to parse JSON response: {e}\nResponse: {proc.stdout[:500]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] JSON parse error on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

    raise last_error
