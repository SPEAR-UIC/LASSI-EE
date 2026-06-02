"""Argo LLM gateway.

Posts to the ANL internal Argo streamchat endpoint:
  https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/

Accessible from any machine on the ANL network (no ALCF login required).
No API key is needed by default; set ARGO_API_KEY if your deployment requires one.
Set ARGO_USER to override the username sent in the request body (default: $USER).

Based on the original project's lassi.llm_config.argo_llm() behavior.
"""
from __future__ import annotations

import json
import os
import subprocess


_ARGO_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"


def argo_llm(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
    uri: str = "remote",
) -> str:
    """Call the Argo gateway and return the full response text.

    The call is blocking — curl waits for the complete response before returning.
    """
    # gpto1preview is a special case in the original: system merged into user prompt.
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt

    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")

    data = {
        "user":   os.environ.get("ARGO_USER", os.environ.get("USER", "user")),
        "model":  model,
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop":   [],
    }
    payload = json.dumps(data)

    # Pass the payload via stdin (--data @-) to avoid hitting the OS ARG_MAX
    # limit when prompts contain large source files.
    cmd = [
        "curl", "-k", "-sS", "-X", "POST", _ARGO_URL,
        "-H", "Content-Type: application/json",
        "-H", "Accept: text/plain",
        "-H", "Accept-Encoding: identity",
        "-H", "Cache-Control: no-cache",
        "-H", "Host: apps-dev.inside.anl.gov",
        "--data", "@-",
    ]
    api_key = os.environ.get("ARGO_API_KEY")
    if api_key:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]

    proc = subprocess.run(cmd, input=payload, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Argo curl failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout
