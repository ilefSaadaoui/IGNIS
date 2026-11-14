"""Scripted fake Groq-free model for offline end-to-end tests.

NOT a real LLM: deterministic, rule-based responses. Used to exercise the full
ADK Runner stack (delegation, tools, sessions) without network or API keys.
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmResponse
from google.genai import types
from typing_extensions import override


def _last_user_text(contents) -> str:
    text = ""
    for content in reversed(contents or []):
        if getattr(content, "role", None) != "user":
            continue
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "text", None):
                text += part.text + "\n"
            fr = getattr(part, "function_response", None)
            if fr is not None:
                resp = getattr(fr, "response", None)
                text += json.dumps(resp, default=str)[:500] + "\n"
        if text:
            break
    return text


def _has_function_response(contents) -> bool:
    for content in contents or []:
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "function_response", None) is not None:
                return True
    return False


class FakeLlm(BaseLlm):
    """Rule-based stand-in for Groq inference in tests."""

    model: str = "fake/groq-test-model"

    @override
    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        contents = llm_request.contents or []
        user_text = _last_user_text(contents)
        low = user_text.lower()
        tool_names = set(getattr(llm_request, "tools_dict", {}) or {})

        def has(name: str) -> bool:
            return name in tool_names

        def emit_text(text: str):
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                partial=False,
            )

        def emit_call(name: str, args: dict):
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
                ),
                partial=False,
            )

        # --- Root orchestrator: transfer to the right specialist -----------
        # The root is the only agent whose tool set is *just* transfer_to_agent.
        if tool_names and tool_names <= {"transfer_to_agent"}:
            if any(w in low for w in ["find", "search", "discover", "paper", "publication", "investigation", "research"]):
                yield emit_call("transfer_to_agent", {"agent_name": "nasa_research_agent"})
            elif any(w in low for w in ["statistic", "mean", "average", "std", "compare"]):
                yield emit_call("transfer_to_agent", {"agent_name": "data_analyst_agent"})
            else:
                yield emit_text(
                    "I need a more specific research question (discovery, comparison, ML suitability, "
                    "fire-safety implications, graph extraction, or figure analysis)."
                )
            return

        # --- NASA specialist: call the real tool once, then summarise ------
        if has("search_nasa_research"):
            if not _has_function_response(contents):
                m = re.search(r"(BASS(?:-II)?|FLEX|ACME|SAME(?:-R)?|SAFFIRE)", user_text.upper())
                query = f"{m.group(1)} microgravity combustion" if m else "microgravity combustion"
                yield emit_call("search_nasa_research", {"query": query, "source": "ntrs", "limit": 2})
            else:
                yield emit_text(
                    "NASA research step complete. Records were returned by the NTRS tool "
                    "(see tool output above); PSI has no JSON API and was not queried. "
                    "task_status=completed; figures below are tool-reported, not invented."
                )
            return

        # --- Document specialist -------------------------------------------
        if has("analyze_document_text"):
            if not _has_function_response(contents):
                doc_match = re.search(r"DOC_TEXT:\s*(.+?)(?:\nEVIDENCE_BLOCK|\Z)", user_text, flags=re.DOTALL)
                text = doc_match.group(1).strip() if doc_match else user_text.strip()
                yield emit_call("analyze_document_text", {"text": text[:3000]})
            else:
                yield emit_text("Document analysis complete; sections, references and mentions reported by tools.")
            return

        # --- Data analyst ----------------------------------------------------
        if has("calculate_descriptive_statistics"):
            if not _has_function_response(contents):
                g = re.search(r"GROUPS:\s*(\{.*?\})\s*(?:UNIT:\s*(\w+))?", user_text, flags=re.DOTALL)
                if g:
                    groups = json.loads(g.group(1))
                    yield emit_call("compare_experimental_data",
                                    {"groups": groups, "unit": g.group(2) or None})
                else:
                    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", user_text)]
                    yield emit_call("calculate_descriptive_statistics", {"values": nums[:50] or [1.0], "unit": "n/a"})
            else:
                yield emit_text("Deterministic statistics computed by tools; see numeric output above.")
            return

        # --- ML agent ---------------------------------------------------------
        if has("assess_ml_suitability"):
            if not _has_function_response(contents):
                n = int((re.search(r"N_SAMPLES:\s*(\d+)", user_text) or [0, 120])[1])
                feats = int((re.search(r"N_FEATURES:\s*(\d+)", user_text) or [0, 6])[1])
                target = (re.search(r"TARGET:\s*([a-z_]+)", user_text) or [0, "unknown"])[1]
                yield emit_call("assess_ml_suitability",
                                {"n_samples": n, "n_features": feats, "target_type": target,
                                 "missing_fraction": 0.05})
            else:
                yield emit_text("ML suitability assessed by tools; see verdict above. No model was trained.")
            return

        # --- Knowledge graph ---------------------------------------------------
        if has("extract_entities"):
            if not _has_function_response(contents):
                text = re.sub(r"^.*?(?:TEXT:)?\s*", "", user_text, count=0)[:2000]
                yield emit_call("extract_entities", {"text": text})
            else:
                yield emit_text("Knowledge graph extraction complete; see nodes/edges above.")
            return

        # --- Multimodal --------------------------------------------------------
        if has("analyze_scientific_figure"):
            m = re.search(r"IMAGE:\s*(\S+)", user_text)
            path = m.group(1) if m else "nonexistent.png"
            if not _has_function_response(contents):
                yield emit_call("analyze_scientific_figure", {"image_path": path})
            else:
                yield emit_text("Figure analysis step done (status reported by the tool).")
            return

        # --- Fire safety --------------------------------------------------------
        if has("assess_mission_relevance"):
            if not _has_function_response(contents):
                yield emit_call("assess_mission_relevance",
                                {"findings": [user_text[:500]], "mission_context": "iss"})
            else:
                yield emit_text("Fire-safety interpretation complete with explicit translation limits.")
            return

        # --- Reviewer ------------------------------------------------------------
        if has("review_scientific_claims"):
            if not _has_function_response(contents):
                evidence = user_text[:800]
                yield emit_call("review_scientific_claims",
                                {"claims": ["The supplied records describe NASA combustion research."],
                                 "evidence": [evidence]})
            else:
                yield emit_text(
                    "Review verdict: requires_review. Claims were checked against supplied evidence only; "
                    "no external verification was performed. Reject anything beyond the evidence."
                )
            return

        # --- Fallback -----------------------------------------------------------
        yield emit_text("FakeLlm: no rule matched for this agent/turn.")
