"""ALCF Inference Endpoints client (stdlib HTTPS + SSE streaming) — used by
--on-resource / --submit (compute nodes can't reach Argo).  Reads cached tokens
from ~/.globus/.../inference_app/tokens.json with stdlib JSON (no globus_sdk)."""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

_BASE_URLS = {
    "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    "metis":  "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
}
DEFAULT_CLUSTER = "sophia"
DEFAULT_MODEL   = "meta-llama/Meta-Llama-3.1-70B-Instruct"

_AUTH_CLIENT_ID    = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"
_APP_NAME          = "inference_app"
_GATEWAY_CLIENT_ID = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"
_TOKENS_PATH       = Path.home() / ".globus" / "app" / _AUTH_CLIENT_ID / _APP_NAME / "tokens.json"
_USER_AGENT        = "curl/8.4.0"


def _load_auth_helper(script_dir):
    path = script_dir / "inference_auth_token.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference_auth_token", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_inference_entry(obj):
    """Walk tokens.json (globus_sdk 3.x flat / 4.x nested) for the gateway entry."""
    if not isinstance(obj, dict): return None
    if obj.get("resource_server") == _GATEWAY_CLIENT_ID and "access_token" in obj:
        return obj
    for k, v in obj.items():
        if k == _GATEWAY_CLIENT_ID and isinstance(v, dict) and "access_token" in v:
            return v
        r = _find_inference_entry(v)
        if r is not None: return r
    return None


def _read_cached_access_token(safety_margin_s=60):
    """Read a still-valid access token from tokens.json (no globus_sdk needed)."""
    if not _TOKENS_PATH.is_file(): return None
    try: data = json.loads(_TOKENS_PATH.read_text(encoding="utf-8"))
    except Exception: return None
    entry = _find_inference_entry(data)
    if not entry: return None
    access = entry.get("access_token")
    try: expires = float(entry.get("expires_at_seconds", 0))
    except (TypeError, ValueError): return None
    if not access or expires <= time.time() + safety_margin_s: return None
    return access


def get_access_token(script_dir):
    """Return a valid access token (cached if fresh, else refresh via helper)."""
    cached = _read_cached_access_token()
    if cached: return cached
    try: helper = _load_auth_helper(script_dir)
    except FileNotFoundError as exc: raise SystemExit(f"auth helper: {exc}")
    except (ImportError, ModuleNotFoundError):
        raise SystemExit(
            f"No valid token at {_TOKENS_PATH} and globus_sdk unavailable. "
            f"Re-auth: python {script_dir}/inference_auth_token.py authenticate")
    try: return helper.get_access_token()
    except Exception as exc:
        msg = str(exc)
        if "Access token does not exist" in msg or "tokens" in msg.lower():
            raise SystemExit(
                f"Not authenticated. Run: python {script_dir}/inference_auth_token.py authenticate")
        raise SystemExit(f"auth failure: {exc}")


def stream_chat(*, model, system_prompt, user_prompt, token,
                cluster=DEFAULT_CLUSTER, temperature=None,
                on_chunk: Callable[[str], None] | None = None, timeout_s=600.0):
    """POST a streaming chat completion; return the assembled response.
    `on_chunk(delta)` fires per fragment.  UA spoofed to curl (Cloudflare WAF)."""
    base = _BASE_URLS.get(cluster)
    if not base: raise ValueError(f"unknown cluster {cluster!r}")
    url = f"{base}/chat/completions"

    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    body = {"model": model, "messages": messages, "stream": True}
    if temperature is not None: body["temperature"] = temperature

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "Accept": "text/event-stream",
                 "User-Agent": _USER_AGENT},
        method="POST")

    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        body_text = ""
        try: body_text = exc.read().decode("utf-8", errors="replace")
        except Exception: pass
        summary = body_text
        try:
            j = json.loads(body_text)
            if isinstance(j, dict):
                summary = j.get("detail") or j.get("error") or j.get("message") or body_text
        except Exception:
            m = re.search(r"Ray ID:?\s*([0-9a-f]+)", body_text, re.I)
            summary = f"Cloudflare block (Ray ID: {m.group(1)})" if m else "HTML response (WAF block?)"
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} from {cluster}/{model}: {str(summary)[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"connection to {url} failed: {exc.reason}") from exc

    out = ""
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line or not line.startswith("data:"): continue
            payload = line[5:].lstrip()
            if payload == "[DONE]": break
            try: chunk = json.loads(payload)
            except json.JSONDecodeError: continue
            choices = chunk.get("choices") or []
            if not choices: continue
            content = (choices[0].get("delta") or {}).get("content") or ""
            if content:
                out += content
                if on_chunk:
                    try: on_chunk(content)
                    except Exception: pass
    return out


def list_endpoints(token):
    """GET the catalog of available models per cluster."""
    req = urllib.request.Request(
        "https://inference-api.alcf.anl.gov/resource_server/list-endpoints",
        headers={"Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def verify_token(token, timeout_s=15.0):
    """Pre-flight GET to catch 30-day high-assurance timeout; returns (ok, err_msg)."""
    try:
        list_endpoints(token)
        return True, ""
    except urllib.error.HTTPError as exc:
        body = ""
        try: body = exc.read().decode("utf-8", errors="replace")
        except Exception: pass
        return False, f"HTTP {exc.code}: {body[:500]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
