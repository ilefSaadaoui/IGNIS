"""Schemas for multimodal (figure/chart) analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ignis.schemas.common import TaskStatus


class FigureAnalysis(BaseModel):
    agent_name: str = "multimodal_agent"
    task_status: TaskStatus
    image_path: str | None = None
    image_format: str | None = None
    image_bytes: int | None = None
    visible_labels: list[str] = Field(default_factory=list)
    axes: list[str] = Field(default_factory=list)
    legend: list[str] = Field(default_factory=list)
    ocr_text: list[str] = Field(default_factory=list)
    interpretation: str = ""
    uncertainty: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    vision_model: str | None = None
