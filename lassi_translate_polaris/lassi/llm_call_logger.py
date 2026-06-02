from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _truncate(s: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    head = s[: max(0, limit - 200)]
    tail = s[-200:] if limit >= 400 else ""
    return f"{head}\n\n... [TRUNCATED {len(s) - len(head) - len(tail)} chars] ...\n\n{tail}"


@dataclass
class LLMCallLogger:
    root_dir: str
    truncate_chars: int = 2000

    def __post_init__(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)

    def log_call(
        self,
        *,
        call_id: int,
        intent: str,
        llm_source: str,
        model: str,
        num_ctx: int,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        call_dir = os.path.join(self.root_dir, f"call_{call_id:04d}")
        os.makedirs(call_dir, exist_ok=True)

        meta_obj: Dict[str, Any] = {
            "timestamp_utc":       _utc_ts(),
            "call_id":             call_id,
            "intent":              intent,
            "llm_source":          llm_source,
            "model":               model,
            "num_ctx":             num_ctx,
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars":   len(user_prompt),
            "response_chars":      len(response_text),
            "system_prompt_preview": _truncate(system_prompt, self.truncate_chars),
            "user_prompt_preview":   _truncate(user_prompt,   self.truncate_chars),
            "response_preview":      _truncate(response_text, self.truncate_chars),
        }
        if meta:
            meta_obj["meta"] = meta

        with open(os.path.join(call_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_obj, f, indent=2)
        with open(os.path.join(call_dir, "system.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(call_dir, "user.txt"), "w", encoding="utf-8") as f:
            f.write(user_prompt)
        with open(os.path.join(call_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response_text)

    def print_call_summary(self, *, call_id: int, intent: str, **_kwargs: Any) -> None:
        print(f"[LLM] call #{call_id:04d} complete ({intent})")
