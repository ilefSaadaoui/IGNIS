"""Deterministic task classification and workflow routing (no LLM required).

The interactive root agent routes via ADK transfer; this module powers the
programmatic workflows (main.py --workflow ...) and is fully unit-testable.
"""

from __future__ import annotations

import re
from enum import Enum


class WorkflowName(str, Enum):
    RESEARCH_DISCOVERY = "A"
    EXPERIMENT_COMPARISON = "B"
    ML_SUITABILITY = "C"
    SCIENTIFIC_QUESTION = "D"
    KNOWLEDGE_GRAPH = "E"
    FIGURE_ANALYSIS = "F"
    STATISTICAL_ANALYSIS = "G"


# Layered specialist invocation per workflow. Agents WITHIN one layer run in
# parallel (they only need the question + prior layers); layers run in order.
# The reviewer is always its own final layer.
WORKFLOW_LAYERS: dict[WorkflowName, list[list[str]]] = {
    WorkflowName.RESEARCH_DISCOVERY: [["nasa_research_agent"], ["scientific_reviewer_agent"]],
    WorkflowName.EXPERIMENT_COMPARISON: [["nasa_research_agent", "document_intelligence_agent", "data_analyst_agent"],
                                         ["scientific_reviewer_agent"]],
    WorkflowName.ML_SUITABILITY: [["data_analyst_agent"], ["ml_agent"], ["scientific_reviewer_agent"]],
    WorkflowName.SCIENTIFIC_QUESTION: [["nasa_research_agent", "document_intelligence_agent"],
                                       ["fire_safety_agent"],
                                       ["scientific_reviewer_agent"]],
    WorkflowName.KNOWLEDGE_GRAPH: [["document_intelligence_agent"], ["knowledge_graph_agent"],
                                   ["scientific_reviewer_agent"]],
    WorkflowName.FIGURE_ANALYSIS: [["multimodal_agent"], ["document_intelligence_agent"],
                                   ["scientific_reviewer_agent"]],
    WorkflowName.STATISTICAL_ANALYSIS: [["data_analyst_agent"], ["scientific_reviewer_agent"]],
}

# Flat ordered view (stable, used for logs/back-compat).
WORKFLOW_DEFINITIONS: dict[WorkflowName, list[str]] = {
    wf: [agent for layer in layers for agent in layer] for wf, layers in WORKFLOW_LAYERS.items()
}

WORKFLOW_DESCRIPTIONS: dict[WorkflowName, str] = {
    WorkflowName.RESEARCH_DISCOVERY: "Research discovery: NASA search → scientific review",
    WorkflowName.EXPERIMENT_COMPARISON: "Experiment comparison: NASA → documents → data analysis → review",
    WorkflowName.ML_SUITABILITY: "ML suitability: data analysis → ML assessment → review",
    WorkflowName.SCIENTIFIC_QUESTION: "Scientific question: NASA → documents → fire-safety implications → review",
    WorkflowName.KNOWLEDGE_GRAPH: "Knowledge graph: document extraction → graph build → review",
    WorkflowName.FIGURE_ANALYSIS: "Figure analysis: multimodal reading → document context → review",
    WorkflowName.STATISTICAL_ANALYSIS: "Statistical analysis: deterministic stats → (optional) review",
}

_KEYWORDS: list[tuple[WorkflowName, list[str]]] = [
    (WorkflowName.FIGURE_ANALYSIS, [r"\bfigure\b", r"\bchart\b", r"\bplot\b", r"\bimage\b", r"\bgraph\.(png|jpg|jpeg)\b"]),
    (WorkflowName.KNOWLEDGE_GRAPH, [r"\bknowledge graph\b", r"\bentit(y|ies)\b", r"\brelationship", r"\bnodes?\b", r"\bedges?\b"]),
    (WorkflowName.ML_SUITABILITY, [r"\bmachine learning\b", r"\bml\b", r"\btrain(ing)?\b", r"\bmodel\b", r"\bclassifier\b", r"\bregression model\b"]),
    (WorkflowName.STATISTICAL_ANALYSIS, [r"\bstatistic", r"\bmean\b", r"\bstd\b", r"\bstandard deviation\b", r"\bmedian\b", r"\bcorrelation\b", r"\bdescriptive\b", r"\bmissing values?\b"]),
    (WorkflowName.EXPERIMENT_COMPARISON, [r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\.?\b", r"\bdiffer"]),
    (WorkflowName.RESEARCH_DISCOVERY, [r"\bfind\b", r"\bsearch\b", r"\bdiscover\b", r"\bpublication", r"\bpaper", r"\binvestigation", r"\bwhat (nasa )?research", r"\blist (the )?(nasa )?"]),
    (WorkflowName.SCIENTIFIC_QUESTION, [r"\bfire safety\b", r"\bspacecraft\b", r"\bmission\b", r"\biss\b", r"\bsafety\b", r"\bimplication", r"\bmoon\b", r"\bmars\b"]),
]


def classify_task(question: str) -> WorkflowName:
    """Best-effort deterministic classification; defaults to scientific question (D)."""
    q = (question or "").lower()
    if not q.strip():
        return WorkflowName.SCIENTIFIC_QUESTION
    scores: dict[WorkflowName, int] = {}
    for workflow, patterns in _KEYWORDS:
        for pat in patterns:
            if re.search(pat, q):
                scores[workflow] = scores.get(workflow, 0) + 1
    if not scores:
        return WorkflowName.SCIENTIFIC_QUESTION
    return max(scores.items(), key=lambda kv: kv[1])[0]


def workflow_for(question: str) -> list[str]:
    """Return the ordered specialist list for a natural-language question."""
    return WORKFLOW_DEFINITIONS[classify_task(question)]
