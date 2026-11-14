"""Schemas for space fire-safety mission relevance and evidence gaps."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ignis.schemas.common import TaskStatus


class MissionRelevanceAssessment(BaseModel):
    agent_name: str = "fire_safety_agent"
    task_status: TaskStatus
    mission_context: str = ""
    relevance_score: float | None = None  # 0..1, rubric-based, NOT a certification
    relevance_rationale: list[str] = Field(default_factory=list)
    implications: list[str] = Field(default_factory=list)
    translation_limits: list[str] = Field(default_factory=list)
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)


class SafetyEvidenceGap(BaseModel):
    dimension: str
    status: str  # covered | partial | missing
    note: str


class SafetyGapReport(BaseModel):
    agent_name: str = "fire_safety_agent"
    task_status: TaskStatus
    investigation: str = ""
    gaps: list[SafetyEvidenceGap] = Field(default_factory=list)
    missing_count: int = 0
    partial_count: int = 0
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
