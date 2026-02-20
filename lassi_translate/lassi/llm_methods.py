from __future__ import annotations

import re
import sys
import threading
import time
from typing import Optional

from lassi.llm_call_logger import LLMCallLogger
import lassi.llm_config_alcf as _alcf

LLM_CALL_LOGGER: Optional[LLMCallLogger] = None
_CALL_ID = 0

_CODE_BLOCK_RE = re.compile(r"```(?:c\+\+|cpp|c|\w+)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    # Incomplete fence: closing ``` missing (e.g., max_tokens cutoff).
    lines = text.split("\n")
    if lines and re.match(r"^```", lines[0].strip()):
        lines = lines[1:]
        while lines and re.match(r"^```\s*$", lines[-1].strip()):
            lines.pop()
        result = "\n".join(lines).strip()
        return result + "\n" if result else text.strip() + "\n"
    return text.strip() + "\n"


class Spinner:
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
                sys.stdout.write(f"  [{self._label}] {elapsed}s elapsed\n")
                sys.stdout.flush()
            i += 1

    def start(self) -> "Spinner":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self._is_tty:
            sys.stdout.write("\r" + " " * 72 + "\r")
            sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


def show_code_preview(code: str, title: str = "Generated code", max_lines: int = 30) -> None:
    lines = code.splitlines()
    total = len(lines)
    sep = "─" * 60
    delay = 0.025 if (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()) else 0.0
    print(f"\n┌{sep}")
    print(f"│  {title}  ({total} lines total)")
    print(f"│{sep}")
    for line in lines[:max_lines]:
        print(f"│  {line}")
        if delay:
            time.sleep(delay)
    if total > max_lines:
        print(f"│  ···  ({total - max_lines} more lines)")
    print(f"└{sep}\n")


def generate_code(model: str, system_prompt: str, user_prompt: str, verifyprompt: bool = False) -> str:
    global _CALL_ID
    _CALL_ID += 1
    response_text = _alcf.call_llm(model, system_prompt, user_prompt, verifyprompt=verifyprompt)
    if LLM_CALL_LOGGER is not None:
        LLM_CALL_LOGGER.log_call(
            call_id=_CALL_ID,
            intent="code generation",
            llm_source="alcf",
            model=model,
            num_ctx=0,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )
        LLM_CALL_LOGGER.print_call_summary(call_id=_CALL_ID, intent="code generation")
    return _extract_code(response_text)
