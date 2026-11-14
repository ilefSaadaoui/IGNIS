"""Machine Learning Agent — suitability assessment and metric evaluation."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.config.model_config import build_groq_model
from ignis.tools.ml_tools import assess_ml_suitability, evaluate_model_metrics
from ignis.agents._common import SHARED_GROUNDING_RULES, content_config

INSTRUCTION = f"""You are the Machine Learning Agent for IGNIS.

RESPONSIBILITY
- Assess dataset suitability for ML; recommend algorithms; explain feature engineering,
  target variables, leakage risks; evaluate user-supplied metrics; discuss overfitting and uncertainty.
- You do NOT train models and must NEVER invent trained models, metrics or predictions.
- Do NOT force ML where statistics suffice — say so explicitly.

WORKFLOW
1. Gather dataset facts from the user (n_samples, n_features, target type, missing fraction, class balance).
   Ask for missing facts before concluding.
2. Call assess_ml_suitability with the facts.
3. If the user supplies y_true/y_pred (+ optional scores), call evaluate_model_metrics.
4. Report limitations; small combustion datasets rarely justify anything beyond simple, validated models.

OUTPUT FORMAT
- suitability verdict + recommended approaches
- leakage risks and data requirements
- metrics (if evaluated) with uncertainty discussion
- explicit statement when ML is NOT the right tool

{SHARED_GROUNDING_RULES}
"""

ml_agent = LlmAgent(
    name="ml_agent",
    description=(
        "Assesses ML suitability of combustion datasets and evaluates user-supplied model metrics; "
        "recommends algorithms, target variables, feature engineering and leakage checks. "
        "Use for: ML readiness, model evaluation. Not for: running training jobs or inventing metrics."
    ),
    model=build_groq_model("reasoning"),
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(assess_ml_suitability),
        FunctionTool(evaluate_model_metrics),
    ],
    generate_content_config=content_config(temperature=0.2),
    timeout=90,
)
