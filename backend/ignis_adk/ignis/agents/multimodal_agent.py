"""Multimodal Analysis Agent — scientific figure analysis (vision-gated)."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.agents._common import SHARED_GROUNDING_RULES, content_config
from ignis.config.model_config import build_groq_model
from ignis.tools.multimodal_tools import analyze_scientific_figure, vision_model_supported

INSTRUCTION = f"""You are the Multimodal Analysis Agent for IGNIS.

RESPONSIBILITY
- Analyse supplied scientific figures: charts, diagrams, plots.
- Extract visible labels, axes, legends; explain observable patterns; report uncertainty.

HARD RULE
- Not every Groq model supports image input. The analyze_scientific_figure tool checks
  the configured GROQ_VISION_MODEL; if it is not image-capable or no key is set, the tool
  returns status='unavailable'. In that case you MUST report 'unavailable — no vision model
  configured' and you must NOT describe image contents you never saw.

WORKFLOW
1. Call analyze_scientific_figure with the image path and the user's question.
2. Relay the tool's structured output faithfully, including limitations.

{SHARED_GROUNDING_RULES}
"""

multimodal_agent = LlmAgent(
    name="multimodal_agent",
    description=(
        "Analyses supplied scientific figures (charts/diagrams) with the configured Groq vision model. "
        "Returns 'unavailable' honestly when no vision-capable model is configured. "
        "Use for: figure interpretation. Not for: text-only questions."
    ),
    model=build_groq_model("fast"),
    instruction=INSTRUCTION,
    tools=[FunctionTool(analyze_scientific_figure), FunctionTool(vision_model_supported)],
    generate_content_config=content_config(temperature=0.1),
    timeout=120,
)
