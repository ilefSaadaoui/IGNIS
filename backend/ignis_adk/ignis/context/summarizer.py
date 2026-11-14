"""Structured conversation summarization for IGNIS sessions.

The summarizer is deterministic (no LLM required) so it works offline and in
tests. It maintains a structured ResearchSummary with a version counter and
explicit verified/unverified separation. Previous AI statements are NEVER
promoted to verified evidence automatically — promotion requires an explicit
source reference via `add_verified_finding(..., sources=[...])`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("ignis.context.summarizer")


class ResearchSummary(BaseModel):
    """Structured, versioned research state for one session."""

    current_question: str = ""
    research_context: str = ""
    verified_findings: list[dict[str, Any]] = Field(default_factory=list)  # {statement, sources[]}
    sources: list[str] = Field(default_factory=list)  # deduplicated source refs (url or identifier)
    unverified_claims: list[str] = Field(default_factory=list)
    previous_agent_actions: list[dict[str, str]] = Field(default_factory=list)  # {agent, action, status}
    unresolved_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    version: int = 0
    updated_at: str | None = None


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


class ConversationSummarizer:
    """Maintains and compresses session research state."""

    MAX_ACTIONS = 30
    MAX_UNVERIFIED = 30
    MAX_LIMITATIONS = 30

    def __init__(self, summary: ResearchSummary | None = None) -> None:
        self.summary = summary or ResearchSummary()

    # --- mutations ----------------------------------------------------------

    def set_question(self, question: str, context: str | None = None) -> ResearchSummary:
        self.summary.current_question = question.strip()
        if context:
            self.summary.research_context = context.strip()
        self._bump()
        return self.summary

    def record_agent_action(self, agent: str, action: str, status: str) -> None:
        self.summary.previous_agent_actions.append(
            {"agent": agent, "action": action[:300], "status": status,
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        )
        self.summary.previous_agent_actions = self.summary.previous_agent_actions[-self.MAX_ACTIONS:]
        self._bump()

    def add_verified_finding(self, statement: str, sources: list[str]) -> None:
        stmt = statement.strip()
        srcs = [s.strip() for s in sources if s and s.strip()]
        if not stmt:
            return
        if not srcs:
            # No source -> cannot be verified; keep as unverified instead of inventing support.
            self.add_unverified_claim(stmt + "  [unverified: no source supplied]")
            return
        existing = {f["statement"].strip().lower() for f in self.summary.verified_findings}
        if stmt.lower() not in existing:
            self.summary.verified_findings.append({"statement": stmt, "sources": srcs})
        merged = list(self.summary.sources) + srcs
        self.summary.sources = _dedupe_keep_order(merged)
        self._bump()

    def add_unverified_claim(self, claim: str) -> None:
        claim = claim.strip()
        if claim:
            self.summary.unverified_claims = _dedupe_keep_order(self.summary.unverified_claims + [claim])[-self.MAX_UNVERIFIED:]
            self._bump()

    def add_unresolved_question(self, question: str) -> None:
        q = question.strip()
        if q:
            self.summary.unresolved_questions = _dedupe_keep_order(self.summary.unresolved_questions + [q])
            self._bump()

    def resolve_unresolved_question(self, question: str) -> None:
        key = question.strip().lower()
        self.summary.unresolved_questions = [q for q in self.summary.unresolved_questions if q.strip().lower() != key]
        self._bump()

    def add_limitation(self, limitation: str) -> None:
        lim = limitation.strip()
        if lim:
            self.summary.limitations = _dedupe_keep_order(self.summary.limitations + [lim])[-self.MAX_LIMITATIONS:]
            self._bump()

    def _bump(self) -> None:
        self.summary.version += 1
        self.summary.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- rendering -----------------------------------------------------------

    def render(self, max_chars: int = 4000) -> str:
        s = self.summary
        lines = [
            f"[Research summary v{s.version} | updated {s.updated_at or 'n/a'}]",
            f"Question: {s.current_question or '(not set)'}",
        ]
        if s.research_context:
            lines.append(f"Context: {s.research_context}")
        if s.verified_findings:
            lines.append("Verified findings (source-backed):")
            for f in s.verified_findings[-12:]:
                srcs = "; ".join(f["sources"][:3])
                lines.append(f"  - {f['statement']}  [sources: {srcs}]")
        else:
            lines.append("Verified findings: none yet.")
        if s.unverified_claims:
            lines.append("Unverified claims (do NOT present as fact):")
            for c in s.unverified_claims[-10:]:
                lines.append(f"  - {c}")
        if s.unresolved_questions:
            lines.append("Unresolved questions:")
            for q in s.unresolved_questions[-8:]:
                lines.append(f"  - {q}")
        if s.limitations:
            lines.append("Limitations:")
            for lim in s.limitations[-8:]:
                lines.append(f"  - {lim}")
        if s.sources:
            lines.append("Sources: " + " | ".join(s.sources[-15:]))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... summary truncated for context budget ...]"
        return text

    def to_dict(self) -> dict[str, Any]:
        return self.summary.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ConversationSummarizer":
        if not data:
            return cls()
        try:
            return cls(ResearchSummary.model_validate(data))
        except Exception as exc:
            logger.warning("could not restore summary (%s); starting fresh", exc)
            fresh = cls()
            fresh.add_limitation(f"Prior session summary failed validation and was reset ({exc.__class__.__name__}).")
            return fresh

    # --- history compression --------------------------------------------------

    def compress_history(self, events: list[str], max_items: int = 8) -> str:
        """Compress raw conversation events into a bounded, deduplicated digest."""
        cleaned = _dedupe_keep_order([e for e in events if e and e.strip()])
        if len(cleaned) <= max_items:
            body = "\n".join(f"- {e}" for e in cleaned)
        else:
            head = cleaned[: max_items // 2]
            tail = cleaned[-max_items // 2:]
            omitted = len(cleaned) - len(head) - len(tail)
            body = "\n".join(
                [f"- {e}" for e in head]
                + [f"- [... {omitted} earlier events compressed; key evidence preserved in verified findings/sources ...]"]
                + [f"- {e}" for e in tail]
            )
        return f"Compressed history ({len(cleaned)} unique events):\n{body}"
