"""ALCF Inference Endpoints client — OpenAI-SDK-compatible streaming chat.

Used as the LLM backend when running on-resource (Polaris/Aurora compute node),
where Argo is unreachable.  The auth helper `inference_auth_token.py` is
vendored into the project root; it must have been authenticated at least once
via `python inference_auth_token.py authenticate` before this module can run.

API documentation: https://docs.alcf.anl.gov/services/inference-endpoints/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Base URLs per cluster.  Sophia is vLLM (full OpenAI surface); Metis is
# SambaNova's API (chat-completions only).  See ALCF docs for the full list.
_BASE_URLS = {
    "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    "metis":  "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
}

DEFAULT_CLUSTER = "sophia"
DEFAULT_MODEL   = "meta-llama/Meta-Llama-3.1-70B-Instruct"


def _load_auth_helper(script_dir: Path):
    """Import the vendored inference_auth_token.py as a module."""
    auth_path = script_dir / "inference_auth_token.py"
    if not auth_path.is_file():
        raise FileNotFoundError(
            f"inference_auth_token.py not found at {auth_path}.\n"
            "Re-vendor it from "
            "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"
        )
    import importlib.util
    spec   = importlib.util.spec_from_file_location("inference_auth_token", auth_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def get_access_token(script_dir: Path) -> str:
    """Return a valid Globus access token, refreshing it if needed.

    Raises SystemExit with a clear message if the user has never authenticated.
    """
    try:
        auth_mod = _load_auth_helper(script_dir)
    except Exception as exc:
        raise SystemExit(f"ALCF Inference auth helper error: {exc}")

    try:
        return auth_mod.get_access_token()
    except Exception as exc:
        # Most likely: tokens.json missing → first-time auth required.
        msg = str(exc)
        if "Access token does not exist" in msg or "tokens" in msg.lower():
            raise SystemExit(
                "ALCF Inference: not authenticated yet.\n"
                "Run once:  python lassi_translate/inference_auth_token.py authenticate\n"
                "Then re-run this script."
            )
        raise SystemExit(f"ALCF Inference auth failure: {exc}")


def stream_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    token: str,
    cluster: str = DEFAULT_CLUSTER,
    temperature: float | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Stream a chat-completion from the ALCF Inference Endpoints.

    Returns the full assistant response as a string.  Calls `on_chunk` with
    each incremental text delta as it arrives (enables sliding-window display).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit(
            "openai package not installed.\n"
            "Run:  pip install openai globus_sdk"
        )

    base_url = _BASE_URLS.get(cluster)
    if base_url is None:
        raise ValueError(
            f"Unknown ALCF Inference cluster {cluster!r}; "
            f"valid: {list(_BASE_URLS)}"
        )

    client = OpenAI(base_url=base_url, api_key=token)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs: dict = {"model": model, "messages": messages, "stream": True}
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        stream = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Surface HTTP errors with body if the SDK includes one.
        raise RuntimeError(
            f"ALCF Inference call to {base_url} failed: {exc}"
        ) from exc

    response_text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # The OpenAI SDK exposes content via delta.content; vLLM returns
        # incremental text strings.  Tool/function-call deltas are ignored
        # (we never request them).
        content = getattr(delta, "content", None) or ""
        if content:
            response_text += content
            if on_chunk is not None:
                try:
                    on_chunk(content)
                except Exception:
                    pass

    return response_text


def list_endpoints(token: str) -> dict:
    """Query available models/endpoints (for `--inference-list-models`)."""
    import urllib.request
    import json
    req = urllib.request.Request(
        "https://inference-api.alcf.anl.gov/resource_server/list-endpoints",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())
