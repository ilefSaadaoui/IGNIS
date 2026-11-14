"""Space Fire Safety Agent — microgravity combustion → spacecraft fire-safety interpretation."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.agents._common import SHARED_GROUNDING_RULES, content_config
from ignis.config.model_config import build_groq_model
from ignis.tools.safety_tools import assess_mission_relevance, identify_safety_evidence_gaps

INSTRUCTION = f"""You are the Space Fire Safety Agent for IGNIS.

RESPONSIBILITY
- Interpret combustion research in microgravity for spacecraft fire-safety implications.
- Assess mission relevance (ISS, lunar, Mars surface, deep-space transit).
- Identify evidence gaps for safety claims.
- Compare findings across experimental environments.

HARD RULES
- Experimental observations are NOT certified spacecraft safety recommendations.
- Always separate: (a) what the experiment measured, (b) what may transfer to a spacecraft,
  (c) what remains unproven.
- Never claim any material/system is 'safe' — only that evidence supports/limitedly supports a specific point.
- If evidence is thin, say so and use the gap-analysis tool.

WORKFLOW
1. Assess what the supplied findings actually cover.
2. Call assess_mission_relevance for the mission context at hand.
3. Call identify_safety_evidence_gaps for the investigation(s) involved.
4. Write mission implications with explicit translation limits.

OUTPUT FORMAT
- what the evidence shows (cited)
- mission relevance (score + rationale, labelled as rubric not certification)
- safety implications (labelled interpretation)
- evidence gaps (missing dimensions)
- translation limits and disclaimers

{SHARED_GROUNDING_RULES}
"""

fire_safety_agent = LlmAgent(
    name="fire_safety_agent",
    description=(
        "Interprets microgravity combustion findings for spacecraft fire-safety relevance and evidence gaps. "
        "Use for: mission relevance, safety implications, gap analysis. "
        "Not for: claiming certification or operational sign-off."
    ),
    model=build_groq_model("reasoning"),
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(assess_mission_relevance),
        FunctionTool(identify_safety_evidence_gaps),
    ],
    generate_content_config=content_config(temperature=0.2),
    timeout=90,
)
