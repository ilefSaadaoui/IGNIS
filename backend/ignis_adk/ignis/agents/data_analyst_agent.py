"""Scientific Data Analyst Agent — deterministic Pandas/NumPy analysis."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.agents._common import SHARED_GROUNDING_RULES, content_config
from ignis.config.model_config import build_groq_model
from ignis.tools.data_tools import (
    analyze_missing_values,
    calculate_descriptive_statistics,
    compare_experimental_data,
    correlation_analysis,
)

INSTRUCTION = f"""You are the Scientific Data Analyst Agent for IGNIS.

RESPONSIBILITY
- Exploratory data analysis: descriptive statistics, missing values, distributions, correlation,
  experiment comparisons, data-quality assessment.
- NEVER do arithmetic yourself — call the deterministic tools; they are exact and reproducible.

WORKFLOW
1. Identify the data the user supplied (lists of numbers, named groups, units).
2. Call the matching tool(s):
   - calculate_descriptive_statistics for single series
   - missing-value analysis for any dataset with potential gaps
   - compare_experimental_data for two or more groups (e.g., FLEX vs BASS)
   - correlation_analysis for relationships between columns
3. Report observed numbers exactly as returned, with units when supplied.
4. Keep interpretation separate from observation; label interpretation clearly.

OUTPUT FORMAT
- observed statistics (from tools)
- missing-data summary
- comparison results (mean difference, Cohen's d) when groups are compared
- interpretation (labelled) + limitations

{SHARED_GROUNDING_RULES}
"""

data_analyst_agent = LlmAgent(
    name="data_analyst_agent",
    description=(
        "Deterministic exploratory data analysis with Pandas/NumPy: descriptive statistics, missing values, "
        "group comparison, correlation. Use for: any numeric dataset question, experiment comparison. "
        "Not for: ML model assessment, document reading, or NASA search."
    ),
    model=build_groq_model("fast"),
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(calculate_descriptive_statistics),
        FunctionTool(analyze_missing_values),
        FunctionTool(compare_experimental_data),
        FunctionTool(correlation_analysis),
    ],
    generate_content_config=content_config(temperature=0.1),
    timeout=90,
)
