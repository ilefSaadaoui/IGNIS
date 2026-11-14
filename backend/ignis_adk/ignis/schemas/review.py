"""Schemas for scientific review and quality control."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ignis.schemas.common import TaskStatus


class ReviewVerdict(str, Enum):
    APPROVED = "approved"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    REQUIRES_EVIDENCE = "requires_review"


class CitationCheck(BaseModel):
    citation: str
    url: str | None = None
    status: str  # verified_reachable | unreachable | missing_url | unchecked
    detail: str = ""


class ClaimAssessment(BaseModel):
    claim: str
    verdict: str  # supported | unsupported | insufficient_evidence
    matched_evidence: list[str] = Field(default_factory=list)
    reason: str = ""


class ReviewReport(BaseModel):
    agent_name: str = "scientific_reviewer_agent"
    task_status: TaskStatus
    verdict: ReviewVerdict
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    unsupported_findings: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.verdict == ReviewVerdict.APPROVED
