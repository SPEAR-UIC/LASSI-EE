"""LLM helpers supporting Argo and ALCF inference backends."""

from __future__ import annotations

import re
import sys
import threading
import time
from typing import Optional, Tuple

from lassi.format_colors import color
from lassi.llm_call_logger import LLMCallLogger
import lassi.llm_config as _argo_mod
import lassi.llm_config_alcf as _alcf_mod

# Set by the runner (optional).
LLM_CALL_LOGGER: Optional[LLMCallLogger] = None
_CALL_ID = 0


_CODE_BLOCK_RE = re.compile(r"```(?:c\+\+|cpp|c|\w+)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_code_block(text: str) -> str:
    """Extract the first fenced code block if present; otherwise return the full text.

    Handles both complete fenced blocks (opening + closing ```) and truncated
    responses where the closing fence is missing (e.g., due to max_tokens cutoff).
    """
    # Complete fenced block — best case.
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"

    # Incomplete fence: response starts with ```lang but closing fence is absent
    # (truncated response). Strip the opening fence line and any trailing fence.
    lines = text.split("\n")
    if lines and re.match(r"^```", lines[0].strip()):
        lines = lines[1:]
        while lines and re.match(r"^```\s*$", lines[-1].strip()):
            lines.pop()
        result = "\n".join(lines).strip()
        return result + "\n" if result else text.strip() + "\n"

    return text.strip() + "\n"


class Spinner:
    """Animated 'thinking' indicator printed to stdout while an LLM call runs.

    Usage (as context manager):
        with Spinner("Generating SYCL translation"):
            response = call_llm(...)
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "Working") -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _spin(self) -> None:
        start = time.time()
        i = 0
        while not self._stop.wait(0.1):
            elapsed = int(time.time() - start)
            if self._is_tty:
                sys.stdout.write(f"\r  {self._FRAMES[i % len(self._FRAMES)]}  {self._label}... ({elapsed}s)")
                sys.stdout.flush()
            elif elapsed > 0 and elapsed % 10 == 0:
                # Non-TTY fallback: one dot every 10 s so logs show progress.
                sys.stdout.write(f"  [{self._label}] {elapsed}s elapsed\n")
                sys.stdout.flush()
            i += 1

    def start(self) -> "Spinner":
        self._thread.start()
        return self

    def stop(self, final_msg: str = "") -> None:
        self._stop.set()
        self._thread.join()
        if self._is_tty:
            clearline = "\r" + " " * 72 + "\r"
            sys.stdout.write(clearline)
            if final_msg:
                sys.stdout.write(final_msg + "\n")
            sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


def show_code_preview(code: str, title: str = "Generated code preview", max_lines: int = 30) -> None:
    """Print the first `max_lines` lines of `code` with per-line animation.

    Gives the viewer a 'sliding window' feel of the code being written.
    """
    lines = code.splitlines()
    total = len(lines)
    visible = lines[:max_lines]

    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    delay = 0.025 if is_tty else 0.0  # no delay when piped / captured

    sep = "─" * 60
    print(f"\n┌{sep}")
    print(f"│  {title}  ({total} lines total)")
    print(f"│{sep}")
    for line in visible:
        print(f"│  {line}")
        if delay:
            time.sleep(delay)
    if total > max_lines:
        print(f"│  ···  ({total - max_lines} more lines)")
    print(f"└{sep}\n")


def llm_call(
    intent: str,
    llm_source: str,
    model: str,
    num_ctx: int,
    system_prompt: str,
    user_prompt: str,
    display_response: bool,
    verifyprompt: bool,
) -> str:
    global _CALL_ID

    if llm_source == "argo":
        _CALL_ID += 1
        response_text = _argo_mod.argo_llm(model, system_prompt, user_prompt, temp=0.2, verifyprompt=verifyprompt)
    elif llm_source == "alcf":
        _CALL_ID += 1
        response_text = _alcf_mod.argo_llm(model, system_prompt, user_prompt, temp=0.2, verifyprompt=verifyprompt)
    else:
        raise ValueError(f"Unsupported llm_source: {llm_source!r}. Use 'argo' or 'alcf'")

    if LLM_CALL_LOGGER is not None:
        LLM_CALL_LOGGER.print_call_summary(call_id=_CALL_ID, intent=intent, context_components={"system_prompt": True, "user_prompt": True})
        LLM_CALL_LOGGER.log_call(
            call_id=_CALL_ID,
            intent=intent,
            llm_source=llm_source,
            model=model,
            num_ctx=num_ctx,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )

    if display_response:
        print("\n" + color.BOLD + color.PURPLE + "[LLM] response" + color.END)
        print(response_text)

    return response_text


def code_generator_llm(
    llm_source: str,
    model: str,
    num_ctx: int,
    temp: float,
    system_prompt: str,
    content_prompt: str,
    display_response: bool,
    verifyprompt: bool,
) -> Tuple[str, str]:
    """Generate code and return (raw_response, extracted_code)."""

    response_text = llm_call(
        "code generation",
        llm_source,
        model,
        num_ctx,
        system_prompt,
        content_prompt,
        display_response,
        verifyprompt,
    )
    return response_text, _extract_code_block(response_text)
