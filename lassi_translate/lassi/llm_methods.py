from __future__ import annotations

import re
import shutil
import sys
import threading
import time
from typing import Optional

from lassi.llm_call_logger import LLMCallLogger
import lassi.llm_config_alcf as _alcf

LLM_CALL_LOGGER: Optional[LLMCallLogger] = None
_CALL_ID = 0
DEBUG_MODE: bool = False  # Set True via --debug flag to enable prompt size diagnostics

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


class StreamingWindow:
    """5-line in-place terminal window that shows tokens as they arrive.

    In TTY mode, uses ANSI escape sequences to maintain a scrolling window:
      ◎  Label... (Ns, N tokens)
      │  last line of generated content
      │  ...
      │  (5 content lines total)

    On completion, collapses to a single summary line:
      ✓  Label (Ns, N tokens)

    In non-TTY mode (piped output / SLURM logs), falls back to periodic
    single-line progress messages.
    """
    CONTENT_LINES = 5
    _REDRAW_INTERVAL = 0.12  # seconds between redraws (~8 Hz)

    def __init__(self, label: str) -> None:
        self._label = label
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self._completed: list[str] = []   # fully terminated lines
        self._pending = ""                 # current incomplete line
        self._token_count = 0
        self._start = time.time()
        self._last_redraw = 0.0
        self._last_nontty_bucket = -1
        # Total lines drawn: 1 header + CONTENT_LINES content
        self._window_h = 1 + self.CONTENT_LINES

    def _elapsed(self) -> int:
        return int(time.time() - self._start)

    def _col_width(self) -> int:
        try:
            return max(40, shutil.get_terminal_size().columns - 6)
        except Exception:
            return 100

    def start(self) -> "StreamingWindow":
        if self._is_tty:
            sys.stdout.write(f"  ◎  {self._label}...\n")
            for _ in range(self.CONTENT_LINES):
                sys.stdout.write("  │\n")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"  ◎  {self._label}...\n")
            sys.stdout.flush()
        return self

    def push(self, token: str) -> None:
        """Receive one streaming token delta and update the display."""
        self._token_count += 1
        self._pending += token
        # Split completed lines out of pending
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._completed.append(line)
        if self._is_tty:
            now = time.time()
            if now - self._last_redraw >= self._REDRAW_INTERVAL:
                self._redraw()
                self._last_redraw = now
        else:
            bucket = self._elapsed() // 30
            if bucket > self._last_nontty_bucket and self._elapsed() > 0:
                self._last_nontty_bucket = bucket
                sys.stdout.write(f"  ◎  {self._label}... ({self._elapsed()}s)\n")
                sys.stdout.flush()

    def _content_lines(self) -> list[str]:
        """Return the CONTENT_LINES most recent lines (padded with empty strings)."""
        all_lines = self._completed + ([self._pending] if self._pending else [])
        content = all_lines[-self.CONTENT_LINES:]
        while len(content) < self.CONTENT_LINES:
            content.append("")
        return content

    def _redraw(self) -> None:
        w = self._col_width()
        # Move cursor to top of window
        sys.stdout.write(f"\033[{self._window_h}A")
        # Overwrite header
        sys.stdout.write(
            f"\033[2K\r  ◎  {self._label}... ({self._elapsed()}s)\n"
        )
        # Overwrite content lines
        for line in self._content_lines():
            sys.stdout.write(f"\033[2K\r  │  {line[:w]}\n")
        sys.stdout.flush()

    def stop(self) -> None:
        if self._is_tty:
            # Final content update
            self._redraw()
            elapsed = self._elapsed()
            # Move back to top and replace header with completion marker
            sys.stdout.write(f"\033[{self._window_h}A")
            sys.stdout.write(
                f"\033[2K\r  ✓  {self._label} ({elapsed}s)\n"
            )
            # Clear content lines, then reposition cursor right after the ✓ line
            for _ in range(self.CONTENT_LINES):
                sys.stdout.write("\033[2K\r\n")
            sys.stdout.write(f"\033[{self.CONTENT_LINES}A")
            sys.stdout.flush()
        else:
            elapsed = self._elapsed()
            sys.stdout.write(
                f"  ✓  {self._label} ({elapsed}s)\n"
            )
            sys.stdout.flush()

    def __enter__(self) -> "StreamingWindow":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


# Kept for backwards compatibility — no longer used internally.
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


def generate_code(
    model: str,
    system_prompt: str,
    user_prompt: str,
    verifyprompt: bool = False,
    label: str = "[LLM] Generating code",
) -> str:
    global _CALL_ID
    _CALL_ID += 1
    if DEBUG_MODE:
        sys_chars = len(system_prompt)
        usr_chars = len(user_prompt)
        est_tokens = (sys_chars + usr_chars) // 4
        print(f"[DEBUG] LLM call #{_CALL_ID:04d} | system={sys_chars:,} chars | user={usr_chars:,} chars | ~{est_tokens:,} input tokens | max_tokens=32768")
    win = StreamingWindow(label)
    win.start()
    response_text = _alcf.call_llm(
        model, system_prompt, user_prompt,
        verifyprompt=verifyprompt,
        on_token=win.push,
        debug=DEBUG_MODE,
    )
    win.stop()
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
        if DEBUG_MODE:
            LLM_CALL_LOGGER.print_call_summary(call_id=_CALL_ID, intent="code generation")
    return _extract_code(response_text)


def generate_text(
    model: str,
    system_prompt: str,
    user_prompt: str,
    intent: str = "self-prompting",
    verifyprompt: bool = False,
    max_tokens: int = 4096,
    label: str = "[LLM]",
) -> str:
    """Call the LLM and return the raw text response (no code-block extraction).

    Used for self-prompting calls (context summarization, source code description)
    where the response is plain prose, not a fenced code block.
    max_tokens defaults to 4096 since self-prompting outputs are short prose.
    """
    global _CALL_ID
    _CALL_ID += 1
    if DEBUG_MODE:
        sys_chars = len(system_prompt)
        usr_chars = len(user_prompt)
        est_tokens = (sys_chars + usr_chars) // 4
        print(f"[DEBUG] LLM call #{_CALL_ID:04d} | system={sys_chars:,} chars | user={usr_chars:,} chars | ~{est_tokens:,} input tokens | max_tokens={max_tokens}")
    win = StreamingWindow(label)
    win.start()
    response_text = _alcf.call_llm(
        model, system_prompt, user_prompt,
        verifyprompt=verifyprompt,
        max_tokens=max_tokens,
        on_token=win.push,
        debug=DEBUG_MODE,
    )
    win.stop()
    if LLM_CALL_LOGGER is not None:
        LLM_CALL_LOGGER.log_call(
            call_id=_CALL_ID,
            intent=intent,
            llm_source="alcf",
            model=model,
            num_ctx=0,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
        )
        if DEBUG_MODE:
            LLM_CALL_LOGGER.print_call_summary(call_id=_CALL_ID, intent=intent)
    return response_text.strip()
