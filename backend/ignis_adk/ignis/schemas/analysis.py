"""Schemas for statistical analysis and ML suitability/evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ignis.schemas.common import TaskStatus


class DescriptiveStatistics(BaseModel):
    task_status: TaskStatus
    count: int
    missing: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    unit: str | None = None
    notes: list[str] = Field(default_factory=list)


class MissingValueReport(BaseModel):
    task_status: TaskStatus
    total_cells: int
    total_missing: int
    missing_fraction: float
    per_column: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recommendation: str = ""


class GroupStats(BaseModel):
    name: str
    n: int
    mean: float | None
    std: float | None
    median: float | None
    min: float | None
    max: float | None
    missing: int


class ComparisonResult(BaseModel):
    task_status: TaskStatus
    groups: list[GroupStats] = Field(default_factory=list)
    mean_difference: float | None = None
    cohens_d: float | None = None
    relative_difference_pct: float | None = None
    unit: str | None = None
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)


class MLSuitabilityAssessment(BaseModel):
    task_status: TaskStatus
    suitable: bool
    recommended_approaches: list[str] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)
    feature_engineering: list[str] = Field(default_factory=list)
    leakage_risks: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    summary: str = ""


class ModelEvaluation(BaseModel):
    task_status: TaskStatus
    task_type: str  # classification | regression
    metrics: dict[str, float] = Field(default_factory=dict)
    uncertainty: list[str] = Field(default_factory=list)
    overfitting_indicators: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    summary: str = ""
