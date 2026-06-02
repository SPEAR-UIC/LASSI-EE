"""Argo-only LLM helpers.

This standalone bundle only supports the Argo backend.
"""

from __future__ import annotations

import re
from typing import Optional

from lassi.format_colors import color
from lassi.llm_call_logger import LLMCallLogger
from lassi.llm_config import argo_llm

# Set by the runner (optional).
LLM_CALL_LOGGER: Optional[LLMCallLogger] = None
_LLM_CALL_ID = 0


def _extract_code_block(text: str) -> str:
    """Extract the first fenced code block if present; otherwise return the full text."""

    m = re.search(r"```(?:c\+\+|cpp|c|\w+)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def llm_call(
    intent: str,
    llm_source: str,
    model: str,
    num_ctx: int,
    system_prompt: str,
    content_prompt: str,
    display_response: bool,
    verifyprompt: bool,
    uri: str = "",
) -> str:
    global _LLM_CALL_ID

    if llm_source != "argo":
        raise ValueError("standalone bundle supports llm_source='argo' only")

    _LLM_CALL_ID += 1

    response_text = argo_llm(model, system_prompt, content_prompt, 0.2, verifyprompt, uri)

    if LLM_CALL_LOGGER is not None:
        LLM_CALL_LOGGER.print_call_summary(call_id=_LLM_CALL_ID, intent=intent, context_components={"system_prompt": True, "user_prompt": True})
        LLM_CALL_LOGGER.log_call(
            call_id=_LLM_CALL_ID,
            intent=intent,
            llm_source=llm_source,
            model=model,
            num_ctx=num_ctx,
            system_prompt=system_prompt,
            user_prompt=content_prompt,
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
):
    """Generate code and return (raw_response, extracted_code)."""

    response_text = llm_call("code generation", llm_source, model, num_ctx, system_prompt, content_prompt, display_response, verifyprompt)
    return response_text, _extract_code_block(response_text)
