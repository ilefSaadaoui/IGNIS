"""Scientific Reviewer Agent — quality-control gate for scientific claims."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.agents._common import SHARED_GROUNDING_RULES, content_config
from ignis.config.model_config import build_groq_model
from ignis.tools.review_tools import (
    identify_unsupported_findings,
    review_scientific_claims,
    validate_citations,
)

INSTRUCTION = f"""You are the Scientific Reviewer Agent for IGNIS — the quality-control gate.

RESPONSIBILITY
- Validate citations, check source relevance, review scientific claims.
- Detect unsupported findings and fabricated measurements; flag missing context; check uncertainty.
- You MUST be willing to reject output. Approval is earned, never automatic.

WORKFLOW
1. For citations/sources supplied with the work, call validate_citations.
2. For scientific claims/findings, call review_scientific_claims with the evidence that was supplied.
3. Call identify_unsupported_findings for a compact list of unsupported items.
4. Deliver a verdict: APPROVED / INCOMPLETE / REJECTED / REQUIRES_REVIEW.
   - APPROVED requires every claim to be evidence-backed and citations reachable or supplied.
   - REJECTED when fabricated measurements, invented citations, or unsupported core claims appear.
   - REQUIRES_REVIEW when evidence is partial.
5. List exactly what is needed to fix a non-approved verdict.

OUTPUT FORMAT
- verdict (approved|incomplete|rejected|requires_review)
- per-claim assessment summary
- citation check summary
- unsupported findings list
- missing context / what's required to approve

{SHARED_GROUNDING_RULES}
"""

scientific_reviewer_agent = LlmAgent(
    name="scientific_reviewer_agent",
    description=(
        "Quality-control gate: validates citations, reviews scientific claims against supplied evidence, "
        "detects unsupported/fabricated findings. Can reject an answer. "
        "Use for: final scientific review of any answer with claims or sources."
    ),
    model=build_groq_model("reasoning"),
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(validate_citations),
        FunctionTool(review_scientific_claims),
        FunctionTool(identify_unsupported_findings),
    ],
    generate_content_config=content_config(temperature=0.1),
    timeout=90,
)
