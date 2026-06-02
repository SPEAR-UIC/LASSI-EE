"""Unified LLM client — Argo streaming backend.

LLMClient encapsulates the Argo HTTP transport, call logging, and both
public call types (code generation and free-text generation).

Module-level convenience API delegates to a shared LLMClient instance:
    configure(log_dir, truncate_chars, debug)
    generate_code(model, system_prompt, user_prompt, ...)
    generate_text(model, system_prompt, user_prompt, ...)

Set ARGO_USER to override the username sent in the request body (default: $USER).
Set ARGO_API_KEY if your deployment requires a Bearer token.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone

from lassi.format_colors import color  # noqa: F401 (re-exported for callers)


_ARGO_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"

_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)


# ── Argo SSE transport ────────────────────────────────────────────────────────

def _stream_argo(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float | None = None,
) -> str:
    """POST to Argo streamchat via curl and return the accumulated response string."""
    import subprocess
    import tempfile

    # o1-preview has no separate system role; merge into user turn.
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt

    data: dict = {
        "user":   os.environ.get("ARGO_USER", os.environ.get("USER", "unknown")),
        "model":  model,
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop":   [],
    }
    if temp is not None:
        data["temperature"] = temp

    payload = json.dumps(data)

    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Accept-Encoding: identity",
        "-H", "Accept: text/plain",
        "-H", "Cache-Control: no-cache",
    ]
    api_key = os.environ.get("ARGO_API_KEY")
    if api_key:
        headers += ["-H", f"Authorization: Bearer {api_key}"]

    temp_file = None
    try:
        if len(payload) > 50000:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(payload)
                temp_file = f.name
            data_arg = ["-d", f"@{temp_file}"]
        else:
            data_arg = ["-d", payload]

        process = subprocess.Popen(
            ["curl", "-k", "-X", "POST", _ARGO_URL, *headers, *data_arg],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        response_text = ""
        for line in iter(process.stdout.readline, ""):
            response_text += line

        process.stdout.close()
        process.wait()

        # Code 18 (CURLE_PARTIAL_FILE) is expected for streaming endpoints that close
        # the connection before curl's internal bookkeeping thinks it is done.
        if process.returncode not in (0, 18):
            stderr_output = process.stderr.read()
            raise RuntimeError(f"curl error (code {process.returncode}): {stderr_output}")

        return response_text

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


# ── Code extraction ───────────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    """Extract the first fenced code block; fall back to the full response."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        content = m.group(1)
        # Strip the optional language tag from the start.
        # Handles "cuda\n#include..." (normal) and "cuda#include..." (no newline).
        content = re.sub(r"^\w+[ \t]*\n?", "", content)
        return content.strip() + "\n"
    # Incomplete fence (max_tokens cutoff — closing ``` missing).
    # Strip the fence marker but preserve any code on the same line as the fence.
    lines = text.split("\n")
    if lines and re.match(r"^```", lines[0].strip()):
        rest_of_first = re.sub(r"^```\w*[ \t]*", "", lines[0])
        lines = ([rest_of_first] if rest_of_first.strip() else []) + lines[1:]
        while lines and re.match(r"^```\s*$", lines[-1].strip()):
            lines.pop()
        result = "\n".join(lines).strip()
        return result + "\n" if result else text.strip() + "\n"
    return text.strip() + "\n"


# ── Logging helpers ───────────────────────────────────────────────────────────

def _truncate(s: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    head = s[: max(0, limit - 200)]
    tail = s[-200:] if limit >= 400 else ""
    return f"{head}\n\n... [TRUNCATED {len(s) - len(head) - len(tail)} chars] ...\n\n{tail}"


# ── Terminal spinner ──────────────────────────────────────────────────────────

class Spinner:
    """Animated terminal spinner shown while a blocking LLM call is in flight."""
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str) -> None:
        self._label  = label
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _spin(self) -> None:
        start = time.time()
        i = 0
        while not self._stop.wait(0.1):
            elapsed = int(time.time() - start)
            if self._is_tty:
                sys.stdout.write(
                    f"\r  {self._FRAMES[i % len(self._FRAMES)]}  {self._label}... ({elapsed}s)"
                )
                sys.stdout.flush()
            elif elapsed > 0 and elapsed % 20 == 0:
                sys.stdout.write(f"  [{self._label}] {elapsed}s elapsed\n")
                sys.stdout.flush()
            i += 1

    def start(self) -> "Spinner":
        sys.stdout.write(f"  ◎  {self._label}...\n")
        sys.stdout.flush()
        self._thread.start()
        return self

    def stop(self, elapsed_s: float, label_done: str = "") -> None:
        self._stop.set()
        self._thread.join()
        if self._is_tty:
            sys.stdout.write("\r" + " " * 80 + "\r")
        label = label_done or self._label
        sys.stdout.write(f"  ✓  {label} ({int(elapsed_s)}s)\n")
        sys.stdout.flush()


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    """Stateful LLM client for the Argo gateway.

    Holds logging config and a per-session call counter.
    model is passed per-call so one client can serve multiple models.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        truncate_chars: int = 2000,
        debug: bool = False,
    ) -> None:
        self.log_dir        = log_dir
        self.truncate_chars = truncate_chars
        self.debug          = debug
        self._call_id       = 0
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def generate_code(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        verifyprompt: bool = False,
        label: str = "[LLM] Generating code",
        temperature: float | None = None,
    ) -> str:
        """Call Argo and return the first fenced code block."""
        response = self._call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            intent="code generation",
            verifyprompt=verifyprompt,
            label=label,
            temperature=temperature,
        )
        code = _extract_code(response)
        if code.strip().startswith("{") and '"error"' in code:
            raise RuntimeError(
                f"[LLM] Argo returned an error response (call #{self._call_id}):\n  {code.strip()[:200]}"
            )
        return code

    def generate_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        intent: str = "self-prompting",
        verifyprompt: bool = False,
        label: str = "[LLM]",
        temperature: float | None = None,
    ) -> str:
        """Call Argo and return the raw prose response."""
        text = self._call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            intent=intent,
            verifyprompt=verifyprompt,
            label=label,
            temperature=temperature,
        ).strip()
        if text.startswith("{") and '"error"' in text:
            raise RuntimeError(
                f"[LLM] Argo returned an error response (call #{self._call_id}):\n  {text[:200]}"
            )
        return text

    # ── internals ─────────────────────────────────────────────────────────────

    def _call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        intent: str,
        verifyprompt: bool,
        label: str,
        temperature: float | None = None,
    ) -> str:
        self._call_id += 1

        if self.debug:
            sys_chars = len(system_prompt)
            usr_chars = len(user_prompt)
            print(
                f"[DEBUG] LLM call #{self._call_id:04d} | system={sys_chars:,} chars | "
                f"user={usr_chars:,} chars | ~{(sys_chars + usr_chars) // 4:,} input tokens"
            )

        if verifyprompt:
            print("\n=====> VERIFY THE PROMPTING <=====\n")
            print(f"\nSYSTEM: {system_prompt}")
            print(f"\nPROMPT: {user_prompt}")
            print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")

        spinner = Spinner(label).start()
        t0 = time.time()
        response = _stream_argo(model, system_prompt, user_prompt, temp=temperature)
        elapsed = time.time() - t0
        spinner.stop(elapsed)

        self._log_call(
            intent=intent,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response,
        )
        return response

    def _log_call(
        self,
        *,
        intent: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
    ) -> None:
        if not self.log_dir:
            return
        call_dir = os.path.join(self.log_dir, f"call_{self._call_id:04d}")
        os.makedirs(call_dir, exist_ok=True)

        meta: dict = {
            "timestamp_utc":         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "call_id":               self._call_id,
            "intent":                intent,
            "model":                 model,
            "system_prompt_chars":   len(system_prompt),
            "user_prompt_chars":     len(user_prompt),
            "response_chars":        len(response_text),
            "system_prompt_preview": _truncate(system_prompt, self.truncate_chars),
            "user_prompt_preview":   _truncate(user_prompt,   self.truncate_chars),
            "response_preview":      _truncate(response_text, self.truncate_chars),
        }
        with open(os.path.join(call_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(call_dir, "system.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(call_dir, "user.txt"), "w", encoding="utf-8") as f:
            f.write(user_prompt)
        with open(os.path.join(call_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response_text)

        if self.debug:
            print(f"[LLM] call #{self._call_id:04d} complete ({intent})")


# ── Module-level convenience API ──────────────────────────────────────────────
# Holds a shared client instance configured by the pipeline runner at startup.

_client: LLMClient = LLMClient()


def configure(
    *,
    log_dir: str | None = None,
    truncate_chars: int = 2000,
    debug: bool = False,
) -> None:
    """(Re)configure the shared LLMClient instance."""
    global _client
    _client = LLMClient(log_dir=log_dir, truncate_chars=truncate_chars, debug=debug)


def generate_code(model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
    return _client.generate_code(model, system_prompt, user_prompt, **kwargs)


def generate_text(model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
    return _client.generate_text(model, system_prompt, user_prompt, **kwargs)
