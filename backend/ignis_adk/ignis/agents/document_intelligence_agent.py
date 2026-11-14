"""Document Intelligence Agent — analyses scientific text and supplied documents."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.agents._common import SHARED_GROUNDING_RULES, content_config
from ignis.config.model_config import build_groq_model
from ignis.tools.literature_tools import search_scientific_literature
from ignis.tools.document_tools import (
    analyze_document_text,
    extract_experimental_conditions,
    extract_research_findings,
    load_document,
)

INSTRUCTION = f"""You are the Document Intelligence Agent for IGNIS.

RESPONSIBILITY
- Analyse scientific text and supplied documents (PDF, DOCX, TXT, CSV, JSON).
- Extract experimental conditions, methods, findings, limitations, measurements when present.
- Preserve page/section references and distinguish direct evidence from interpretation.

WORKFLOW
1. If given a file path, call load_document first; if it fails, report the tool's error verbatim.
   To ground answers in the ingested NASA corpus, call search_scientific_literature and cite
   passages as [title | NTRS URL | p.N].
2. Call analyze_document_text for structure; extract_experimental_conditions and
   extract_research_findings for content. You may call all three on the same text.
3. Report ONLY what the tools found; do not add measurements that were not matched.
4. If the document lacks a section or values, list them as missing.

OUTPUT FORMAT
- document/overview (chars, words, detected sections)
- experimental conditions found (with matched spans) and missing categories
- candidate findings (with evidence sentences and cue phrases)
- limitations from tools, verbatim where applicable

{SHARED_GROUNDING_RULES}
"""

document_intelligence_agent = LlmAgent(
    name="document_intelligence_agent",
    description=(
        "Analyses scientific documents and text (PDF/DOCX/TXT/CSV/JSON): structure, experimental conditions, "
        "findings with evidence sentences, limitations. Use for: document analysis, condition/finding extraction. "
        "Not for: NASA web search, numeric statistics, or graph building."
    ),
    model=build_groq_model("default"),
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(search_scientific_literature),
        FunctionTool(load_document),
        FunctionTool(analyze_document_text),
        FunctionTool(extract_experimental_conditions),
        FunctionTool(extract_research_findings),
    ],
    generate_content_config=content_config(temperature=0.1),
    timeout=90,
)
