"""Schemas for document extraction, scientific findings and knowledge graphs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ignis.schemas.common import SourceRef, TaskStatus


class ExperimentalConditions(BaseModel):
    task_status: TaskStatus
    pressure: list[str] = Field(default_factory=list)
    temperature: list[str] = Field(default_factory=list)
    oxygen_concentration: list[str] = Field(default_factory=list)
    gravity_environment: list[str] = Field(default_factory=list)
    fuel_or_material: list[str] = Field(default_factory=list)
    flow_velocity: list[str] = Field(default_factory=list)
    atmosphere: list[str] = Field(default_factory=list)
    raw_matches: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ScientificFinding(BaseModel):
    statement: str
    evidence_sentence: str
    cue: str
    source: str | None = None


class DocumentExtraction(BaseModel):
    agent_name: str = "document_intelligence_agent"
    task_status: TaskStatus
    summary: str = ""
    char_count: int = 0
    word_count: int = 0
    sections_detected: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    figure_refs: list[str] = Field(default_factory=list)
    conditions: ExperimentalConditions | None = None
    findings: list[ScientificFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    id: str
    name: str
    type: str  # INVESTIGATION | MATERIAL | FACILITY | INSTRUMENT | CONDITION | PHENOMENON | ORGANIZATION | MEASUREMENT
    evidence: str = ""
    occurrences: int = 1


class ExtractedRelationship(BaseModel):
    id: str
    source: str  # entity id
    target: str  # entity id
    type: str  # INVESTIGATED_IN | USED_IN | MEASURED_IN | OBSERVED_IN | PART_OF | SUPPLIES
    evidence: str = ""
    confidence: str = "low"  # rule-based extraction is low/moderate confidence by default


class GraphRepresentation(BaseModel):
    task_status: TaskStatus
    nodes: list[ExtractedEntity] = Field(default_factory=list)
    edges: list[ExtractedRelationship] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    provenance: str = "derived from supplied text; rule-based extraction"
    limitations: list[str] = Field(default_factory=list)
