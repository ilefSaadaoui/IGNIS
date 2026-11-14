"""Schemas for NASA research discovery and investigation metadata."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ignis.schemas.common import SourceRef, TaskStatus


class InvestigationRecord(BaseModel):
    """One discovered NASA record (paper, report, presentation, dataset)."""

    title: str
    identifier: str | None = None
    url: str | None = None
    source: str = "NASA NTRS"
    record_type: str | None = None
    published: str | None = None
    center: str | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    distribution: str | None = None  # PUBLIC / restricted, as reported by source
    authors: list[str] = Field(default_factory=list)


class DiscoveryResult(BaseModel):
    agent_name: str = "nasa_research_agent"
    task_status: TaskStatus
    query: str
    source: str
    total_matched: int | None = None
    records: list[InvestigationRecord] = Field(default_factory=list)
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    retrieve_status: str = ""  # machine-readable note, e.g. 'ntrs_ok', 'psi_portal_only'
    errors: list[str] = Field(default_factory=list)


class InvestigationMetadata(BaseModel):
    agent_name: str = "nasa_research_agent"
    task_status: TaskStatus
    investigation: str
    expansion: str | None = None
    expansion_verified: bool = False
    background: list[str] = Field(default_factory=list)
    records: list[InvestigationRecord] = Field(default_factory=list)
    total_matched: int | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


KNOWN_INVESTIGATIONS: dict[str, dict[str, Any]] = {
    # Expansions verified against public NASA pages (see README references).
    "BASS": {"expansion": "Burning and Suppression of Solids", "verified": True,
             "terms": "Burning and Suppression of Solids BASS"},
    "BASS-II": {"expansion": "Burning and Suppression of Solids II (BASS-II)", "verified": True,
                "terms": "BASS-II Burning and Suppression of Solids"},
    "FLEX": {"expansion": "Flame Extinguishment Experiment", "verified": True,
             "terms": "Flame Extinguishment Experiment FLEX"},
    "ACME": {"expansion": "Advanced Combustion via Microgravity Experiments", "verified": True,
             "terms": "Advanced Combustion via Microgravity Experiments ACME"},
    "SAFFIRE": {"expansion": "SAFFIRE (Spacecraft Fire Experiment)", "verified": True,
                "terms": "SAFFIRE Spacecraft Fire Experiment"},
    "SAME": {"expansion": "Smoke Aerosol Measurement Experiment", "verified": True,
             "terms": "Smoke Aerosol Measurement Experiment SAME"},
    "SAME-R": {"expansion": "Smoke Aerosol Measurement Experiment - Reflight", "verified": True,
               "terms": "Smoke Aerosol Measurement Experiment SAME-R reflight"},
}


def normalize_investigation_name(name: str) -> str:
    key = name.strip().upper().replace("_", "-").replace(" ", "-")
    aliases = {"SAFFIRE-I": "SAFFIRE", "SAFFIRE-II": "SAFFIRE", "SAFFIRE-III": "SAFFIRE",
               "SAFFIRE-IV": "SAFFIRE", "SAFFIRE1": "SAFFIRE", "SAFFIRE2": "SAFFIRE",
               "SAFFIRE3": "SAFFIRE", "SAFFIRE4": "SAFFIRE", "SAME2": "SAME-R"}
    return aliases.get(key, key)
