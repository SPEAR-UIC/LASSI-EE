"""LLM client with two backends (Argo + ALCF Inference) and a live SlidingWindow display.

LLMClient is the public surface; `configure()` swaps the module-global instance
so callers use `generate_code(...)` / `generate_text(...)` regardless of backend.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable

_ARGO_URL      = "https://apps.inside.anl.gov/argoapi/api/v1/resource/streamchat/"
_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)


def _stream_argo(model, system_prompt, content_prompt, temp=None, on_chunk=None):
    """POST to Argo streamchat via curl; return the accumulated response."""
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt
    data = {"user": os.environ.get("ARGO_USER", os.environ.get("USER", "unknown")),
            "model": model, "system": system_prompt, "prompt": [content_prompt], "stop": []}
    if temp is not None: data["temperature"] = temp
    payload = json.dumps(data)
    headers = ["-H", "Content-Type: application/json", "-H", "Accept-Encoding: identity",
               "-H", "Accept: text/plain", "-H", "Cache-Control: no-cache"]
    key = os.environ.get("ARGO_API_KEY")
    if key: headers += ["-H", f"Authorization: Bearer {key}"]

    tmp = None
    try:
        if len(payload) > 50000:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(payload); tmp = f.name
            data_arg = ["-d", f"@{tmp}"]
        else:
            data_arg = ["-d", payload]
        p = subprocess.Popen(["curl", "-k", "-X", "POST", _ARGO_URL, *headers, *data_arg],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        out = ""
        for line in iter(p.stdout.readline, ""):
            out += line
            if on_chunk:
                try: on_chunk(line)
                except Exception: pass
        p.stdout.close(); p.wait()
        if p.returncode not in (0, 18):
            raise RuntimeError(f"curl error ({p.returncode}): {p.stderr.read()}")
        return out
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)


def _extract_code(text):
    """Pull the first fenced code block out of `text`; fall back to raw text."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        c = m.group(1)
        c = re.sub(r"^\w+[ \t]*\n?", "", c)
        return c.strip() + "\n"
    lines = text.split("\n")
    if lines and re.match(r"^```", lines[0].strip()):
        rest = re.sub(r"^```\w*[ \t]*", "", lines[0])
        lines = ([rest] if rest.strip() else []) + lines[1:]
        while lines and re.match(r"^```\s*$", lines[-1].strip()):
            lines.pop()
        result = "\n".join(lines).strip()
        return result + "\n" if result else text.strip() + "\n"
    return text.strip() + "\n"


class SlidingWindow:
    """Live N-line view of streaming LLM output (ANSI cursor controls).

    feed() enqueues parsed lines onto a deque; a background thread pops one
    per tick and redraws — yields a smooth scroll even when many newlines
    arrive in a single SSE chunk.  Falls back to silent mode on a non-TTY,
    unless LASSI_FORCE_TTY=1 (set by --submit so output renders in tail -F).
    """
    _WIDTH = 110
    _FRAME = 0.08
    _FAST  = 0.01
    _FAST_AT = 20
    _DRAIN = 1.5

    def __init__(self, label, n=5):
        self._label, self._n = label, n
        self._buf, self._lines = "", deque(maxlen=n)
        self._pending: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.time()
        self._is_tty = (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()) \
                       or os.environ.get("LASSI_FORCE_TTY") == "1"
        self._drawn = False

    def start(self):
        if self._is_tty:
            sys.stdout.write(f"  ◎  {self._label}...\n")
            for _ in range(self._n): sys.stdout.write("\033[2m    │\033[0m\n")
            sys.stdout.flush()
            self._drawn = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        else:
            sys.stdout.write(f"  [{self._label}]...\n"); sys.stdout.flush()
        return self

    def feed(self, chunk):
        if not self._is_tty: return
        self._buf += chunk
        new = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            new.append(line.rstrip("\r"))
        if new:
            with self._lock: self._pending.extend(new)

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                line = self._pending.popleft() if self._pending else None
                back = len(self._pending)
            if line is not None:
                self._lines.append(line); self._redraw()
            time.sleep(self._FAST if back > self._FAST_AT else self._FRAME)

    def _redraw(self):
        if not self._drawn: return
        sys.stdout.write(f"\033[{self._n}A")
        padded = list(self._lines) + [""] * (self._n - len(self._lines))
        for ln in padded:
            d = ln[:self._WIDTH] if len(ln) > self._WIDTH else ln
            sys.stdout.write(f"\r\033[2K\033[2m    │\033[0m {d}\n")
        sys.stdout.flush()

    def stop(self, elapsed_s=None):
        elapsed = elapsed_s if elapsed_s is not None else time.time() - self._t0
        if self._is_tty and self._drawn:
            if self._buf.strip():
                with self._lock: self._pending.append(self._buf.rstrip())
                self._buf = ""
            deadline = time.time() + self._DRAIN
            while time.time() < deadline:
                with self._lock:
                    if not self._pending: break
                time.sleep(0.04)
            self._stop.set()
            if self._thread: self._thread.join(timeout=0.5)
            sys.stdout.write(f"\033[{self._n + 1}A")
            for _ in range(self._n + 1): sys.stdout.write("\r\033[2K\n")
            sys.stdout.write(f"\033[{self._n + 1}A")
        sys.stdout.write(f"  ✓  {self._label} ({int(elapsed)}s)\n")
        sys.stdout.flush()


class LLMClient:
    """Backend-agnostic LLM call surface.

    backend="argo"            → Argo gateway via curl (remote mode)
    backend="alcf_inference"  → ALCF Inference Endpoints via urllib (on-resource/submit)

    Logs every call's prompts + response to <log_dir>/call_NNNN/ for replay.
    """

    def __init__(self, log_dir=None, debug=False, stream_display=False,
                 backend="argo", inference_cluster="sophia",
                 inference_token_provider: Callable[[], str] | None = None):
        self.log_dir, self.debug, self.stream_display = log_dir, debug, stream_display
        self.backend, self.inference_cluster = backend, inference_cluster
        self.inference_token_provider = inference_token_provider
        self._call_id = 0
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        if backend not in ("argo", "alcf_inference"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "alcf_inference" and inference_token_provider is None:
            raise ValueError("alcf_inference requires inference_token_provider")

    def generate_code(self, model, system_prompt, user_prompt, *,
                      verifyprompt=False, label="[LLM] Generating code", temperature=None):
        """Call the LLM and return the first fenced code block from the response."""
        text = self._call(model, system_prompt, user_prompt,
                          "code generation", verifyprompt, label, temperature)
        code = _extract_code(text)
        if code.strip().startswith("{") and '"error"' in code:
            raise RuntimeError(f"backend error (call #{self._call_id}): {code.strip()[:200]}")
        return code

    def generate_text(self, model, system_prompt, user_prompt, *,
                      intent="self-prompting", verifyprompt=False, label="[LLM]", temperature=None):
        """Call the LLM and return the raw response text (no fence extraction)."""
        text = self._call(model, system_prompt, user_prompt,
                          intent, verifyprompt, label, temperature).strip()
        if text.startswith("{") and '"error"' in text:
            raise RuntimeError(f"backend error (call #{self._call_id}): {text[:200]}")
        return text

    def _call(self, model, system_prompt, user_prompt, intent, verifyprompt, label, temperature):
        self._call_id += 1
        if self.debug:
            print(f"[DEBUG] LLM #{self._call_id:04d} | system={len(system_prompt):,} | "
                  f"user={len(user_prompt):,} chars")
        if verifyprompt:
            print(f"\nSYSTEM: {system_prompt}\nPROMPT: {user_prompt}\n")

        t0 = time.time()
        window = SlidingWindow(label).start() if self.stream_display else None
        try:
            response = self._stream(model, system_prompt, user_prompt, temperature,
                                    on_chunk=(window.feed if window else None))
        except Exception as exc:
            if window:
                try: window.stop(time.time() - t0)
                except Exception: pass
            print(f"\n  [LLM ERROR] call #{self._call_id:04d} ({intent}) "
                  f"backend={self.backend} model={model}: {exc}")
            raise
        if window: window.stop(time.time() - t0)

        self._log_call(intent, model, system_prompt, user_prompt, response)
        return response

    def _stream(self, model, system_prompt, user_prompt, temperature, on_chunk):
        if self.backend == "argo":
            return _stream_argo(model, system_prompt, user_prompt,
                                temp=temperature, on_chunk=on_chunk)
        from lassi.alcf_inference import stream_chat
        return stream_chat(model=model, system_prompt=system_prompt,
                           user_prompt=user_prompt, token=self.inference_token_provider(),
                           cluster=self.inference_cluster, temperature=temperature,
                           on_chunk=on_chunk)

    def _log_call(self, intent, model, system_prompt, user_prompt, response_text):
        if not self.log_dir: return
        d = os.path.join(self.log_dir, f"call_{self._call_id:04d}")
        os.makedirs(d, exist_ok=True)
        meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "call_id": self._call_id, "intent": intent, "model": model,
                "system_chars": len(system_prompt), "user_chars": len(user_prompt),
                "response_chars": len(response_text)}
        with open(os.path.join(d, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
        for name, txt in [("system.txt", system_prompt), ("user.txt", user_prompt),
                          ("response.txt", response_text)]:
            with open(os.path.join(d, name), "w", encoding="utf-8") as f: f.write(txt)


_client = LLMClient()


def configure(*, log_dir=None, debug=False, stream_display=False,
              backend="argo", inference_cluster="sophia",
              inference_token_provider=None):
    """Replace the module-global LLMClient.  Called once per experiment so
    LLM-call logs land in the right per-experiment directory."""
    global _client
    _client = LLMClient(log_dir=log_dir, debug=debug, stream_display=stream_display,
                        backend=backend, inference_cluster=inference_cluster,
                        inference_token_provider=inference_token_provider)


def generate_code(model, system_prompt, user_prompt, **kw):
    """Module-level shortcut: dispatch to the configured LLMClient."""
    return _client.generate_code(model, system_prompt, user_prompt, **kw)


def generate_text(model, system_prompt, user_prompt, **kw):
    """Module-level shortcut: dispatch to the configured LLMClient."""
    return _client.generate_text(model, system_prompt, user_prompt, **kw)
