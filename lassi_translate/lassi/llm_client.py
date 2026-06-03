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
from collections import deque
from datetime import datetime, timezone
from typing import Callable

_ARGO_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"

_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)


# ── Argo SSE transport ────────────────────────────────────────────────────────

def _stream_argo(
    model: str,
    system_prompt: str,
    content_prompt: str,
    temp: float | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """POST to Argo streamchat via curl and return the accumulated response string.

    `on_chunk` is called once per stdout line as the response streams in,
    enabling live terminal display.
    """
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
            if on_chunk is not None:
                try:
                    on_chunk(line)
                except Exception:
                    pass  # display errors must never break the LLM call

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


# ── Sliding-window terminal display ───────────────────────────────────────────

class SlidingWindow:
    """Live N-line sliding view of streaming LLM output for terminal demos.

    Reserves N lines under a header on start(), then each feed() call parses
    incoming chunks into complete lines and enqueues them.  A background
    renderer thread pops one line per tick (~80 ms) and redraws the window
    in place using ANSI cursor controls.  This produces a smooth line-by-line
    scroll even when many newlines arrive in a single SSE chunk.

    On a non-TTY (file redirect, CI), falls back to silent accumulation —
    the full response is still saved to the LLM call log.
    """

    _LINE_WIDTH      = 110   # safe upper bound; covers most terminals without wrapping
    _FRAME_DELAY_S   = 0.08  # cadence between line additions during normal streaming
    _FAST_DELAY_S    = 0.01  # used when backlog is large so we don't fall behind
    _BACKLOG_FAST_AT = 20    # > this many queued lines → fast-drain mode
    _STOP_DRAIN_S    = 1.5   # cap on post-stream drain time

    def __init__(self, label: str, window_size: int = 5) -> None:
        self._label       = label
        self._window_size = window_size
        self._buffer      = ""
        self._lines: deque[str] = deque(maxlen=window_size)
        self._pending: deque[str] = deque()      # parsed lines awaiting display
        self._lock        = threading.Lock()     # protects _pending
        self._stop_event  = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._t0          = time.time()
        self._is_tty      = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._drawn       = False

    def start(self) -> "SlidingWindow":
        if self._is_tty:
            sys.stdout.write(f"  ◎  {self._label}...\n")
            for _ in range(self._window_size):
                sys.stdout.write("\033[2m    │\033[0m\n")  # dim left border
            sys.stdout.flush()
            self._drawn = True
            self._render_thread = threading.Thread(
                target=self._render_loop, daemon=True,
            )
            self._render_thread.start()
        else:
            sys.stdout.write(f"  [{self._label}]...\n")
            sys.stdout.flush()
        return self

    def feed(self, chunk: str) -> None:
        if not self._is_tty:
            return  # full response logged to disk; no progress prints in dumb mode
        # Parsing runs on the caller's thread (the curl-stdout loop) — cheap.
        # Display happens on the renderer thread, paced for smooth scrolling.
        self._buffer += chunk
        new_lines = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            new_lines.append(line.rstrip("\r"))
        if new_lines:
            with self._lock:
                self._pending.extend(new_lines)

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                line = self._pending.popleft() if self._pending else None
                backlog = len(self._pending)
            if line is not None:
                self._lines.append(line)
                self._redraw()
            # Adaptive cadence: speed up when the backlog grows so we don't
            # accumulate a multi-second lag behind the real LLM stream.
            time.sleep(self._FAST_DELAY_S if backlog > self._BACKLOG_FAST_AT
                       else self._FRAME_DELAY_S)

    def _redraw(self) -> None:
        if not self._drawn:
            return
        # Move cursor back to the first window line and rewrite all N lines.
        sys.stdout.write(f"\033[{self._window_size}A")
        padded = list(self._lines) + [""] * (self._window_size - len(self._lines))
        for line in padded:
            display = line[: self._LINE_WIDTH] if len(line) > self._LINE_WIDTH \
                      else line
            sys.stdout.write(f"\r\033[2K\033[2m    │\033[0m {display}\n")
        sys.stdout.flush()

    def stop(self, elapsed_s: float | None = None) -> None:
        elapsed = elapsed_s if elapsed_s is not None else time.time() - self._t0

        if self._is_tty and self._drawn:
            # Flush any trailing partial line (no closing newline) into the queue.
            if self._buffer.strip():
                with self._lock:
                    self._pending.append(self._buffer.rstrip())
                self._buffer = ""

            # Let the renderer drain so the user sees the last lines scroll past,
            # but cap total drain time so a huge backlog can't stall the pipeline.
            drain_deadline = time.time() + self._STOP_DRAIN_S
            while time.time() < drain_deadline:
                with self._lock:
                    if not self._pending:
                        break
                time.sleep(0.04)

            # Tear down the renderer thread.
            self._stop_event.set()
            if self._render_thread is not None:
                self._render_thread.join(timeout=0.5)

            # Erase the window block + header line, then print the done line.
            sys.stdout.write(f"\033[{self._window_size + 1}A")
            for _ in range(self._window_size + 1):
                sys.stdout.write("\r\033[2K\n")
            sys.stdout.write(f"\033[{self._window_size + 1}A")

        sys.stdout.write(f"  ✓  {self._label} ({int(elapsed)}s)\n")
        sys.stdout.flush()


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    """Stateful LLM client.

    Two backends:
      - "argo"             — Argonne's Argo streamchat gateway (works from any
                             host with access to apps-dev.inside.anl.gov).
      - "alcf_inference"   — ALCF Inference Endpoints (OpenAI-compatible API,
                             used when running on-resource where Argo is
                             unreachable).  Requires `inference_token_provider`
                             and `inference_cluster`.

    Model name is passed per-call so one client can serve multiple models.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        truncate_chars: int = 2000,
        debug: bool = False,
        stream_display: bool = False,
        backend: str = "argo",
        inference_cluster: str = "sophia",
        inference_token_provider: Callable[[], str] | None = None,
    ) -> None:
        self.log_dir        = log_dir
        self.truncate_chars = truncate_chars
        self.debug          = debug
        self.stream_display = stream_display   # if True, show SlidingWindow instead of Spinner
        self.backend        = backend
        self.inference_cluster        = inference_cluster
        self.inference_token_provider = inference_token_provider
        self._call_id       = 0
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if backend not in ("argo", "alcf_inference"):
            raise ValueError(f"Unknown LLM backend: {backend!r}")
        if backend == "alcf_inference" and inference_token_provider is None:
            raise ValueError(
                "backend='alcf_inference' requires inference_token_provider"
            )

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

        t0 = time.time()
        if self.stream_display:
            window = SlidingWindow(label).start()
            response = self._stream(model, system_prompt, user_prompt,
                                    temperature, on_chunk=window.feed)
            window.stop(time.time() - t0)
        else:
            spinner = Spinner(label).start()
            response = self._stream(model, system_prompt, user_prompt,
                                    temperature, on_chunk=None)
            spinner.stop(time.time() - t0)

        self._log_call(
            intent=intent,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response,
        )
        return response

    def _stream(
        self,
        model: str, system_prompt: str, user_prompt: str,
        temperature: float | None,
        on_chunk: Callable[[str], None] | None,
    ) -> str:
        """Backend-dispatch wrapper around streaming chat completion."""
        if self.backend == "argo":
            return _stream_argo(model, system_prompt, user_prompt,
                                temp=temperature, on_chunk=on_chunk)
        # backend == "alcf_inference"
        from lassi.alcf_inference import stream_chat
        return stream_chat(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            token=self.inference_token_provider(),
            cluster=self.inference_cluster,
            temperature=temperature,
            on_chunk=on_chunk,
        )

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
    stream_display: bool = False,
    backend: str = "argo",
    inference_cluster: str = "sophia",
    inference_token_provider: Callable[[], str] | None = None,
) -> None:
    """(Re)configure the shared LLMClient instance.

    `stream_display=True` enables the demo-mode sliding-window display.
    `backend='alcf_inference'` switches to the ALCF Inference Endpoints API
    (used when running on-resource).  Requires `inference_token_provider`,
    a zero-arg callable returning a fresh Globus access token.
    """
    global _client
    _client = LLMClient(
        log_dir=log_dir, truncate_chars=truncate_chars,
        debug=debug, stream_display=stream_display,
        backend=backend,
        inference_cluster=inference_cluster,
        inference_token_provider=inference_token_provider,
    )


def generate_code(model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
    return _client.generate_code(model, system_prompt, user_prompt, **kwargs)


def generate_text(model: str, system_prompt: str, user_prompt: str, **kwargs) -> str:
    return _client.generate_text(model, system_prompt, user_prompt, **kwargs)
