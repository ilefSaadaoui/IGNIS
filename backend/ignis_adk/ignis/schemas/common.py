"""Shared schema primitives for IGNIS structured agent results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Common status convention used across IGNIS agents and tools."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class SourceRef(BaseModel):
    """A provenance record for one source."""

    title: str | None = None
    url: str | None = None
    source: str | None = None  # e.g. "NASA NTRS", "PSI", "document"
    identifier: str | None = None  # e.g. NTRS id, report number
    published: str | None = None
    accessed: str | None = None


class AgentResult(BaseModel):
    """Base structured result envelope shared by IGNIS agents."""

    agent_name: str
    task_status: TaskStatus
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence_status: str = "unknown"  # high | moderate | low | unknown
    errors: list[str] = Field(default_factory=list)


def error_result(agent_name: str, message: str, *, status: TaskStatus = TaskStatus.FAILED) -> dict[str, Any]:
    """Build a structured failure result instead of fabricating success."""
    return AgentResult(
        agent_name=agent_name,
        task_status=status,
        summary=f"{agent_name} could not complete the task.",
        errors=[message],
        confidence_status="unknown",
    ).model_dump(mode="json")
