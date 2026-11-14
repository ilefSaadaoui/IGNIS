"""Token estimation and budget management for IGNIS.

Tokenizer: tiktoken 'cl100k_base' is used as a *documented approximation*.
Groq-hosted open-weight models (Llama, GPT-OSS, Qwen, Kimi) use their own
tokenizers, and Groq does not publish per-model encoders through a public
python package — exact counts are therefore not guaranteed. Budgets are
deliberately conservative (estimates + safety margin).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import tiktoken

from ignis.config.settings import get_settings
from ignis.config.model_config import AGENT_TOKEN_BUDGETS, get_model_context_window

logger = logging.getLogger("ignis.context.token_manager")

_TOKENIZER = None
TRUNCATION_MARKER = "\n[... context truncated to fit token budget; omitted content NOT silently dropped — see limitation ...]\n"


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # offline environments without the BPE file
            logger.warning("tiktoken unavailable (%s); falling back to heuristic token estimate", exc)
            _TOKENIZER = False
    return _TOKENIZER


def estimate_tokens(text: str) -> int:
    """Approximate token count for text (cl100k_base proxy; documented approximation)."""
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    # Heuristic: ~4 chars/token for English prose, with word-boundary sanity floor.
    return max(1, (len(text) + 3) // 4)


@dataclass
class TokenBudget:
    """Resolved budgets for one agent/model combination."""

    model_name: str
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    reserved_output_tokens: int
    agent_budget: int | None = None

    @property
    def usable_input_tokens(self) -> int:
        return max(1, min(self.max_input_tokens, self.context_window - self.reserved_output_tokens))


@dataclass
class ContextCheck:
    fits: bool
    estimated_tokens: int
    budget_tokens: int
    overage: int
    message: str


class TokenManager:
    def __init__(self) -> None:
        self.settings = get_settings()

    def budget_for(self, agent_name: str | None = None, model_name: str | None = None) -> TokenBudget:
        settings = self.settings
        window = get_model_context_window(model_name or settings.GROQ_MODEL)
        agent_budget = AGENT_TOKEN_BUDGETS.get(agent_name) if agent_name else None
        max_input = agent_budget or settings.MAX_INPUT_TOKENS
        max_input = min(max_input, window - settings.MAX_OUTPUT_TOKENS)
        return TokenBudget(
            model_name=model_name or settings.GROQ_MODEL,
            context_window=window,
            max_input_tokens=max_input,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            reserved_output_tokens=settings.MAX_OUTPUT_TOKENS,
            agent_budget=agent_budget,
        )

    def check(self, text: str, agent_name: str | None = None) -> ContextCheck:
        budget = self.budget_for(agent_name)
        est = estimate_tokens(text)
        allowed = budget.usable_input_tokens
        fits = est <= allowed
        return ContextCheck(
            fits=fits,
            estimated_tokens=est,
            budget_tokens=allowed,
            overage=max(0, est - allowed),
            message=(
                f"input fits budget ({est}/{allowed} tokens)"
                if fits
                else f"input OVER budget by {est - allowed} tokens ({est}/{allowed}); reduce context or raise MAX_INPUT_TOKENS"
            ),
        )

    def truncate_to_budget(self, text: str, agent_name: str | None = None, *, head_ratio: float = 0.6) -> tuple[str, bool]:
        """Truncate text to the agent budget, keeping head and tail and an explicit marker.

        Returns (text, was_truncated). Evidence loss is signalled via the marker and the
        returned flag — callers must surface it in limitations, never hide it.
        """
        budget = self.budget_for(agent_name)
        if estimate_tokens(text) <= budget.usable_input_tokens:
            return text, False
        # Work in characters as a proxy, then verify by token estimate.
        target_chars = max(200, int(budget.usable_input_tokens * 3.5))
        head = int(target_chars * head_ratio)
        tail = target_chars - head
        if tail <= 0:
            truncated = text[:head] + TRUNCATION_MARKER
        else:
            truncated = text[:head] + TRUNCATION_MARKER + text[-tail:]
        # Safety loop: shrink until within budget (bounded iterations).
        for _ in range(6):
            if estimate_tokens(truncated) <= budget.usable_input_tokens:
                break
            target_chars = int(len(truncated) * 0.8)
            head = int(target_chars * head_ratio)
            tail = target_chars - head
            truncated = text[:head] + TRUNCATION_MARKER + text[-tail:]
        return truncated, True

    def enforce_output_limit(self, text: str) -> tuple[str, bool]:
        settings = self.settings
        limit = settings.MAX_AGENT_OUTPUT_CHARS
        if len(text) <= limit:
            return text, False
        return text[:limit] + "\n[... output truncated to MAX_AGENT_OUTPUT_CHARS ...]", True


_manager: TokenManager | None = None


def get_token_manager() -> TokenManager:
    global _manager
    if _manager is None:
        _manager = TokenManager()
    return _manager
