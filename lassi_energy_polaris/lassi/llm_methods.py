"""LLM call helpers — Argo backend.

Public interface:
  generate_code(model, system_prompt, user_prompt, ...) -> str   # fenced code block extracted
  generate_text(model, system_prompt, user_prompt, ...) -> str   # plain prose

Both functions:
  - Call argo_llm() (blocking curl to the ANL Argo gateway)
  - Show a terminal spinner while waiting (Argo does not stream)
  - Log the call via LLM_CALL_LOGGER if set
  - Respect DEBUG_MODE for prompt-size diagnostics
"""
from __future__ import annotations

import re
import sys
import threading
import time
from typing import Optional

from lassi.format_colors import color
from lassi.llm_call_logger import LLMCallLogger
from lassi.llm_config import argo_llm

# ── Module-level state set by the pipeline runner ────────────────────────────
LLM_CALL_LOGGER: Optional[LLMCallLogger] = None
DEBUG_MODE: bool = False   # set to True via --debug flag

_CALL_ID = 0

# Regex to extract the first fenced C/C++/CUDA code block from a response.
_CODE_BLOCK_RE = re.compile(r"```(?:cuda|c\+\+|cpp|c|\w+)?\s*\n(.*?)\n```", re.DOTALL)


# ── Code extraction ───────────────────────────────────────────────────────────

def _extract_code(text: str) -> str:
    """Extract the first fenced code block; fall back to the full response."""
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    # Handle incomplete fence (e.g., max_tokens cutoff — closing ``` missing).
    lines = text.split("\n")
    if lines and re.match(r"^```", lines[0].strip()):
        lines = lines[1:]
        while lines and re.match(r"^```\s*$", lines[-1].strip()):
            lines.pop()
        result = "\n".join(lines).strip()
        return result + "\n" if result else text.strip() + "\n"
    return text.strip() + "\n"


# ── Terminal spinner ──────────────────────────────────────────────────────────
# Argo returns the full response in one blocking curl call, so we cannot push
# streaming tokens.  Instead we run a spinner on a daemon thread.

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

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop(0.0)


# ── Public LLM call functions ─────────────────────────────────────────────────

def generate_code(
    model: str,
    system_prompt: str,
    user_prompt: str,
    verifyprompt: bool = False,
    label: str = "[LLM] Generating code",
) -> str:
    """Call Argo and return extracted code (strips the fenced code block markers)."""
    global _CALL_ID
    _CALL_ID += 1

    if DEBUG_MODE:
        sys_chars = len(system_prompt)
        usr_chars = len(user_prompt)
        est_tok   = (sys_chars + usr_chars) // 4
        print(
            f"[DEBUG] LLM call #{_CALL_ID:04d} | system={sys_chars:,} chars | "
            f"user={usr_chars:,} chars | ~{est_tok:,} input tokens"
        )

    spinner = Spinner(label).start()
    t0 = time.time()
    response_text = argo_llm(model, system_prompt, user_prompt, temp=0.2, verifyprompt=verifyprompt)
    elapsed = time.time() - t0
    spinner.stop(elapsed)

    if LLM_CALL_LOGGER is not None:
        LLM_CALL_LOGGER.log_call(
            call_id=_CALL_ID,
            intent="code generation",
            llm_source="argo",
            model=model,
            num_ctx=0,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )
        if DEBUG_MODE:
            LLM_CALL_LOGGER.print_call_summary(call_id=_CALL_ID, intent="code generation")

    return _extract_code(response_text)


def generate_text(
    model: str,
    system_prompt: str,
    user_prompt: str,
    intent: str = "self-prompting",
    verifyprompt: bool = False,
    max_tokens: int = 4096,   # informational only — Argo uses its own default
    label: str = "[LLM]",
) -> str:
    """Call Argo and return the raw prose response (no code-block extraction).

    Used for Stage 0 self-prompting calls (context summarization, source
    code description) where the output is plain text, not a code block.
    """
    global _CALL_ID
    _CALL_ID += 1

    if DEBUG_MODE:
        sys_chars = len(system_prompt)
        usr_chars = len(user_prompt)
        est_tok   = (sys_chars + usr_chars) // 4
        print(
            f"[DEBUG] LLM call #{_CALL_ID:04d} | system={sys_chars:,} chars | "
            f"user={usr_chars:,} chars | ~{est_tok:,} input tokens"
        )

    spinner = Spinner(label).start()
    t0 = time.time()
    response_text = argo_llm(model, system_prompt, user_prompt, temp=0.2, verifyprompt=verifyprompt)
    elapsed = time.time() - t0
    spinner.stop(elapsed)

    if LLM_CALL_LOGGER is not None:
        LLM_CALL_LOGGER.log_call(
            call_id=_CALL_ID,
            intent=intent,
            llm_source="argo",
            model=model,
            num_ctx=0,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )
        if DEBUG_MODE:
            LLM_CALL_LOGGER.print_call_summary(call_id=_CALL_ID, intent=intent)

    return response_text.strip()
