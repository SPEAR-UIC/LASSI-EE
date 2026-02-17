"""Argo-only LLM backend.

This standalone bundle intentionally supports **only** the Argo gateway.

Implementation is based on the original project's
[`lassi.llm_config.argo_llm()`](../../lassi/llm_config.py:368) behavior:
- POST to `https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/`
- JSON payload with keys: `user`, `model`, `system`, `prompt` (list), `stop`, `temperature`, `top_p`
- Response is plain text (streamed by curl in the original); we return the full text.

Auth:
- The original code relies on being on the ANL network and does not include an API key.
- If your deployment requires auth, set `ARGO_API_KEY` and it will be sent as a Bearer token.
"""

import json
import os
import subprocess


def argo_llm(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
) -> str:
    # Match original behavior: always use the ANL gateway streamchat endpoint.
    url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"

    # Original special-case.
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt

    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")

    data = {
        "user": os.environ.get("ARGO_USER", "vchirio"),
        "model": model,
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop": [],
        "temperature": round(temp, 1),
        "top_p": 0.9,
    }

    payload = json.dumps(data)

    # Use curl to match the original streaming gateway behavior.
    cmd = [
        "curl",
        "-k",
        "-sS",
        "--connect-timeout", "30",  # 30 seconds to establish connection
        "--max-time", "300",  # 5 minutes total timeout for response
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-H",
        "Accept: text/plain",
        "-H",
        "Accept-Encoding: identity",
        "-H",
        "Cache-Control: no-cache",
        "-H",
        "Host: apps-dev.inside.anl.gov",
    ]

    api_key = os.environ.get("ARGO_API_KEY")
    if api_key:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]

    # Proxy configuration: only use if explicitly set
    # By default, NO proxy - apps-dev.inside.anl.gov is internal-only
    # and not reachable through the external proxy
    proxy = os.environ.get("ARGO_PROXY")
    if proxy:
        cmd += ["-x", proxy]

    cmd += ["-d", payload]

    # Python 3.6 compatible: use stdout/stderr instead of capture_output
    # Set a long timeout (300s = 5 minutes) for LLM responses which can be slow
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Argo API call timed out after 300 seconds")

    if proc.returncode != 0:
        raise RuntimeError(f"Argo curl call failed (rc={proc.returncode}): {proc.stderr.strip()}")

    return proc.stdout
