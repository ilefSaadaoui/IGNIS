"""Per-agent context construction with duplicate removal and evidence preservation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ignis.context.summarizer import ConversationSummarizer
from ignis.context.token_manager import get_token_manager

logger = logging.getLogger("ignis.context.context_builder")

_URL_RE = re.compile(r"https?://[^\s)\]}>'\"]+")


@dataclass
class ContextBuildResult:
    agent_name: str
    text: str
    estimated_tokens: int
    truncated: bool
    dropped_duplicates: int
    preserved_sources: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


class ContextBuilder:
    """Builds bounded, deduplicated context for a specialist agent."""

    def __init__(self) -> None:
        self.token_manager = get_token_manager()

    @staticmethod
    def _dedupe_blocks(blocks: list[str]) -> tuple[list[str], int]:
        seen: set[str] = set()
        out: list[str] = []
        dropped = 0
        for block in blocks:
            key = re.sub(r"\s+", " ", block.strip().lower())
            if not key:
                continue
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.append(block.strip())
        return out, dropped

    @staticmethod
    def _extract_sources(text: str) -> list[str]:
        return list(dict.fromkeys(_URL_RE.findall(text)))

    def build_for_agent(
        self,
        agent_name: str,
        *,
        question: str,
        summarizer: ConversationSummarizer | None = None,
        history: list[str] | None = None,
        evidence_blocks: list[str] | None = None,
        include_summary: bool = True,
    ) -> ContextBuildResult:
        """Assemble agent-specific context.

        - Duplicates removed.
        - Source URLs are extracted and echoed back so they survive truncation.
        - Enforces the per-agent token budget; if content cannot fit, the
          limitation is reported explicitly (never silently dropped).
        """
        blocks: list[str] = [f"User research question: {question.strip()}"]
        if include_summary and summarizer is not None:
            rendered = summarizer.render(max_chars=2500)
            if rendered.strip():
                blocks.append("Session research summary:\n" + rendered)
        if history:
            from ignis.config.settings import get_settings

            max_items = get_settings().MAX_CONVERSATION_HISTORY
            recent = history[-max_items:]
            blocks.append("Recent relevant history:\n" + "\n".join(f"- {h}" for h in recent))
        if evidence_blocks:
            blocks.extend(evidence_blocks)

        unique, dropped = self._dedupe_blocks(blocks)
        joined = "\n\n".join(unique)
        sources = self._extract_sources(joined)

        truncated_text, was_truncated = self.token_manager.truncate_to_budget(joined, agent_name)
        # Echo preserved sources so truncation never silently loses provenance.
        if sources and was_truncated:
            echo = "\n\n[Preserved source references (kept through truncation):]\n" + "\n".join(
                f"- {s}" for s in sources[:20]
            )
            truncated_text = truncated_text + echo  # budget check below reports if echo overflows

        check = self.token_manager.check(truncated_text, agent_name)
        limitations: list[str] = []
        if was_truncated:
            limitations.append(
                "Context was truncated to fit the agent token budget; omitted material is NOT available to this agent."
            )
        if dropped:
            limitations.append(f"{dropped} duplicate context block(s) removed before sending to the agent.")
        if not check.fits:
            limitations.append(
                f"Context still exceeds the agent budget by {check.overage} tokens after compression; "
                "return partial results and state what could not be included."
            )

        return ContextBuildResult(
            agent_name=agent_name,
            text=truncated_text,
            estimated_tokens=check.estimated_tokens,
            truncated=was_truncated,
            dropped_duplicates=dropped,
            preserved_sources=sources,
            limitations=limitations,
        )
