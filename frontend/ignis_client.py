from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent            # IGNIS_Final\frontend\
CANDIDATE_PATHS = [
    ROOT,
    ROOT.parent / "backend" / "ignis_adk",        # IGNIS_Final\backend\ignis_adk
    ROOT.parent / "backend",
    ROOT.parent / "ignis_adk_final" / "ignis_adk",# legacy fallback
    ROOT.parent,
]
for p in CANDIDATE_PATHS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


AGENT_LABELS: dict[str, str] = {
    "nasa_research_agent": "NASA Research",
    "document_intelligence_agent": "Document Intelligence",
    "data_analyst_agent": "Data Analyst",
    "ml_agent": "Machine Learning",
    "knowledge_graph_agent": "Knowledge Graph",
    "multimodal_agent": "Multimodal",
    "fire_safety_agent": "Fire Safety",
    "scientific_reviewer_agent": "Scientific Reviewer",
}

AGENT_ICONS: dict[str, str] = {
    "nasa_research_agent": "satellite",
    "document_intelligence_agent": "document",
    "data_analyst_agent": "chart",
    "ml_agent": "brain",
    "knowledge_graph_agent": "graph",
    "multimodal_agent": "image",
    "fire_safety_agent": "flame",
    "scientific_reviewer_agent": "shield",
}

WORKFLOW_LABELS: dict[str, str] = {
    "A": "Research discovery",
    "B": "Experiment comparison",
    "C": "ML suitability",
    "D": "Scientific question",
    "E": "Knowledge graph",
    "F": "Figure analysis",
    "G": "Statistical analysis",
}

WORKFLOW_USE_CASES: dict[str, str] = {
    "A": "Find NASA publications, reports, or investigations.",
    "B": "Compare investigations, documents, or measurements.",
    "C": "Assess data readiness and machine-learning approaches.",
    "D": "Answer a scientific question and assess fire-safety relevance.",
    "E": "Extract entities and relationships from supplied text.",
    "F": "Analyze a supplied figure, chart, or image.",
    "G": "Compute descriptive statistics or compare groups.",
}

MISSIONS: dict[str, str] = {
    "iss": "Low Earth Orbit / ISS",
    "lunar": "Lunar surface / Gateway",
    "mars": "Mars surface",
    "transit": "Deep-space transit",
}

INVESTIGATIONS: list[str] = [
    "BASS",
    "BASS-II",
    "FLEX",
    "ACME",
    "SAFFIRE",
    "SAME",
    "SAME-R",
]

_CITATION_RE = re.compile(
    r"\[([^\[\]\n]{1,400}?)\|\s*(https?://[^\s\]\)>'\"]+)\s*\|\s*(p\.\s*\d+[^\]]*?)\]",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s)\]}>'\"]+")
_SECTION_RE = re.compile(r"\[([a-zA-Z0-9_]+) output\]\s*", re.IGNORECASE)
_VERDICT_ORDER = ("rejected", "requires_review", "incomplete", "approved")


class BackendUnavailable(RuntimeError):
    pass


def now_label() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def backend_status() -> tuple[bool, str | None]:
    try:
        import ignis.config.settings
        import ignis.orchestration.routing
        import ignis.orchestration.workflows
        import ignis.tools
        import ignis.tools.document_tools
        import ignis.tools.graph_tools
        import ignis.tools.literature_tools
        import ignis.tools.ml_tools
        import ignis.tools.multimodal_tools
        import ignis.tools.nasa_tools
        import ignis.tools.review_tools
        import ignis.tools.safety_tools

        return True, None
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _require_backend() -> None:
    ok, error = backend_status()
    if not ok:
        raise BackendUnavailable(error or "backend import failed")


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _failure(where: str, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "errors": [f"{where}: {exc.__class__.__name__}: {exc}"],
        "_error": True,
    }


def _guard(where: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        _require_backend()
        return fn(*args, **kwargs)
    except BackendUnavailable:
        raise
    except Exception as exc:
        return _failure(where, exc)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def settings_snapshot() -> dict[str, Any]:
    _require_backend()
    from ignis.config.settings import get_settings

    settings = get_settings()
    return {
        "project": "IGNIS — Space Combustion Intelligence",
        "team": "RaoSpace Labs",
        "models": {
            "default": settings.GROQ_MODEL,
            "reasoning": settings.GROQ_REASONING_MODEL,
            "fast": settings.GROQ_FAST_MODEL,
            "vision": settings.GROQ_VISION_MODEL,
        },
        "budgets": {
            "MAX_INPUT_TOKENS": settings.MAX_INPUT_TOKENS,
            "MAX_OUTPUT_TOKENS": settings.MAX_OUTPUT_TOKENS,
            "MAX_CONTEXT_TOKENS": settings.MAX_CONTEXT_TOKENS,
            "MAX_AGENT_ITERATIONS": settings.MAX_AGENT_ITERATIONS,
            "MAX_CONVERSATION_HISTORY": settings.MAX_CONVERSATION_HISTORY,
            "MAX_DOCUMENT_CHARS": settings.MAX_DOCUMENT_CHARS,
            "MAX_AGENT_OUTPUT_CHARS": settings.MAX_AGENT_OUTPUT_CHARS,
            "REQUEST_TIMEOUT": settings.REQUEST_TIMEOUT,
            "MAX_RETRIES": settings.MAX_RETRIES,
        },
        "provider_order": settings.provider_order,
        "keys": {
            "groq": settings.groq_api_key_configured,
            "openrouter": settings.openrouter_api_key_configured,
            "google": settings.google_api_key_configured,
            "core": settings.core_api_key_configured,
            "nasa": bool(settings.NASA_API_KEY),
        },
        "missing": settings.missing_credentials(),
        "log_level": settings.LOG_LEVEL,
    }


def workflow_catalog() -> list[dict[str, Any]]:
    _require_backend()
    from ignis.orchestration.routing import WORKFLOW_DESCRIPTIONS, WORKFLOW_LAYERS, WorkflowName

    catalog: list[dict[str, Any]] = []
    for workflow in WorkflowName:
        catalog.append(
            {
                "code": workflow.value,
                "name": WORKFLOW_LABELS.get(workflow.value, workflow.name),
                "description": WORKFLOW_DESCRIPTIONS[workflow],
                "use_case": WORKFLOW_USE_CASES.get(workflow.value, ""),
                "layers": [list(layer) for layer in WORKFLOW_LAYERS[workflow]],
            }
        )
    return catalog


def classify_question(question: str) -> str:
    _require_backend()
    from ignis.orchestration.routing import classify_task

    return classify_task(question).value


def corpus_stats() -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.literature_tools import get_corpus_status

        payload = get_corpus_status()
        return payload.get("corpus", payload)

    result = _guard("corpus status", _inner)
    return result if isinstance(result, dict) else {"status": "failed", "errors": [str(result)]}


def run_workflow(
    question: str,
    workflow_code: str | None = None,
    user_id: str = "streamlit_user",
    session_id: str | None = None,
    include_reviewer: bool = True,
) -> dict[str, Any]:
    from ignis.orchestration.workflows import run_workflow as backend_run_workflow

    text = (question or "").strip()
    if not text:
        return {"status": "failed", "errors": ["Question must not be empty."], "_error": True, "question": ""}
    code = (workflow_code or "").strip().upper()
    selected = code if code in WORKFLOW_LABELS else None
    try:
        _require_backend()
        result = _run_async(
            backend_run_workflow(
                text,
                selected,
                user_id=user_id or "streamlit_user",
                session_id=(session_id or "").strip() or None,
                include_reviewer=include_reviewer,
            )
        )
        data = result.as_dict()
        data["steps"] = [_as_mapping(step) for step in data.get("steps", [])]
        failed = any(step.get("status") == "failed" for step in data["steps"])
        data["status"] = "partial" if failed else "completed"
        data["_meta"] = {
            "created_at": now_label(),
            "user_id": user_id or "streamlit_user",
            "include_reviewer": include_reviewer,
            "requested_workflow": selected or "auto",
        }
        return data
    except BackendUnavailable:
        raise
    except Exception as exc:
        return _failure("workflow", exc)


def search_nasa(query: str, source: str = "all", limit: int = 5) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.nasa_tools import search_nasa_research

        return search_nasa_research(query, source=source, limit=limit)

    return _guard("NASA search", _inner)


def search_corpus(query: str, top_k: int = 5, investigation: str | None = None) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.literature_tools import search_scientific_literature

        return search_scientific_literature(query, top_k=top_k, investigation=investigation or None)

    return _guard("corpus search", _inner)


def investigation_metadata(name: str, limit: int = 5) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.nasa_tools import retrieve_investigation_metadata

        return retrieve_investigation_metadata(name, limit=limit)

    return _guard("investigation metadata", _inner)


def search_images(query: str, media_type: str = "image", page_size: int = 5) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.nasa_tools import search_nasa_images

        return search_nasa_images(query, media_type=media_type, page_size=page_size)

    return _guard("NASA images", _inner)


def analyze_document_file(path: str) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.document_tools import (
            analyze_document_text,
            extract_experimental_conditions,
            extract_research_findings,
            load_document,
        )

        loaded = load_document(path)
        if loaded.get("status") != "completed":
            return {
                "status": "failed",
                "load": loaded,
                "errors": loaded.get("errors", ["document could not be loaded"]),
            }
        text = loaded.get("text", "")
        return {
            "status": "completed",
            "load": loaded,
            "analysis": analyze_document_text(text),
            "conditions": extract_experimental_conditions(text),
            "findings": extract_research_findings(text),
        }

    return _guard("document analysis", _inner)


def analyze_document_text_only(text: str) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.document_tools import (
            analyze_document_text,
            extract_experimental_conditions,
            extract_research_findings,
        )

        return {
            "status": "completed",
            "load": {"status": "completed", "char_count": len(text)},
            "analysis": analyze_document_text(text),
            "conditions": extract_experimental_conditions(text),
            "findings": extract_research_findings(text),
        }

    return _guard("text analysis", _inner)


def descriptive_statistics(values: list[float | None], unit: str | None = None) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.data_tools import calculate_descriptive_statistics

        return calculate_descriptive_statistics(values, unit=unit or None)

    return _guard("descriptive statistics", _inner)


def missing_values(data: dict[str, list[float | None]]) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.data_tools import analyze_missing_values

        return analyze_missing_values(data)

    return _guard("missing values", _inner)


def compare_groups(
    groups: dict[str, list[float | None]],
    unit: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.data_tools import compare_experimental_data

        return compare_experimental_data(groups, unit=unit or None, label_a=label_a or None, label_b=label_b or None)

    return _guard("group comparison", _inner)


def correlation(data: dict[str, list[float | None]]) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.data_tools import correlation_analysis

        return correlation_analysis(data)

    return _guard("correlation", _inner)


def ml_suitability(
    n_samples: int,
    n_features: int,
    target_type: str,
    missing_fraction: float = 0.0,
    class_balance_ratio: float | None = None,
    has_leakage_indicator: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.ml_tools import assess_ml_suitability

        return assess_ml_suitability(
            n_samples=n_samples,
            n_features=n_features,
            target_type=target_type,
            missing_fraction=missing_fraction,
            class_balance_ratio=class_balance_ratio,
            has_leakage_indicator=has_leakage_indicator,
            notes=notes,
        )

    return _guard("ML suitability", _inner)


def model_metrics(
    y_true: list[float],
    y_pred: list[float],
    task_type: str = "regression",
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.ml_tools import evaluate_model_metrics

        return evaluate_model_metrics(y_true, y_pred, task_type=task_type, y_score=y_score)

    return _guard("model metrics", _inner)


def build_knowledge_graph(text: str) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.graph_tools import (
            build_graph_representation,
            extract_entities,
            extract_relationships,
        )

        entities = extract_entities(text)
        relationships = extract_relationships(text, entities.get("entities", []))
        graph = build_graph_representation(
            entities.get("entities", []), relationships.get("relationships", [])
        )
        return {
            "status": graph.get("status", "completed"),
            "entities": entities,
            "relationships": relationships,
            "graph": graph,
        }

    return _guard("knowledge graph", _inner)


def vision_supported() -> tuple[bool, str]:
    def _inner() -> tuple[bool, str]:
        from ignis.tools.multimodal_tools import vision_model_supported

        return vision_model_supported()

    try:
        _require_backend()
        return _inner()
    except Exception as exc:
        return False, f"unavailable ({exc.__class__.__name__})"


def analyze_figure(path: str, question: str, ocr_text: str | None = None) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.multimodal_tools import analyze_scientific_figure

        return analyze_scientific_figure(path, question=question, ocr_text=ocr_text or None)

    return _guard("figure analysis", _inner)


def mission_relevance(findings: list[str] | str, mission_context: str = "iss") -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.safety_tools import assess_mission_relevance

        return assess_mission_relevance(findings, mission_context=mission_context)

    return _guard("mission relevance", _inner)


def safety_gaps(investigation: str, findings: list[str] | str | None = None) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.safety_tools import identify_safety_evidence_gaps

        return identify_safety_evidence_gaps(investigation, findings)

    return _guard("safety gaps", _inner)


def review_claims(
    claims: list[str],
    evidence: list[str],
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.review_tools import review_scientific_claims

        return review_scientific_claims(claims, evidence, sources)

    return _guard("claim review", _inner)


def unsupported_findings(findings: list[str], evidence: list[str]) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.review_tools import identify_unsupported_findings

        return identify_unsupported_findings(findings, evidence)

    return _guard("unsupported findings", _inner)


def citation_checks(citations: list[dict[str, str]] | list[str], reachability: bool = True) -> dict[str, Any]:
    def _inner() -> dict[str, Any]:
        from ignis.tools.review_tools import validate_citations

        return validate_citations(citations, check_reachability=reachability)

    return _guard("citation check", _inner)


def extract_citations(text: str) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    citations: list[dict[str, str]] = []
    for match in _CITATION_RE.finditer(text or ""):
        title = match.group(1).strip()
        url = match.group(2).strip()
        page = match.group(3).strip()
        key = (title.lower(), url.lower(), page.lower())
        if key in seen:
            continue
        seen.add(key)
        citations.append({"title": title, "url": url, "page": page, "raw": match.group(0)})
    return citations


def extract_urls(text: str) -> list[str]:
    return list(dict.fromkeys(_URL_RE.findall(text or "")))


def linkify_citations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        url = match.group(2).strip()
        page = match.group(3).strip()
        safe_title = title.replace("[", "(").replace("]", ")")
        return f"[{safe_title} · {page}]({url})"

    return _CITATION_RE.sub(replace, text or "")


def split_final_answer(text: str, question: str = "") -> dict[str, Any]:
    raw = text or ""
    matches = list(_SECTION_RE.finditer(raw))
    if not matches:
        return {"body": raw.strip(), "sections": {}, "lead": raw.strip()}
    lead = raw[: matches[0].start()].strip()
    if question and lead.startswith(question.strip()):
        lead = lead[len(question.strip()) :].strip()
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        sections[match.group(1)] = raw[match.end() : end].strip()
    body_parts = [
        value
        for key, value in sections.items()
        if key != "scientific_reviewer_agent" and value.strip()
    ]
    body = "\n\n".join(body_parts).strip()
    if not body:
        body = sections.get("scientific_reviewer_agent", lead).strip()
    return {"body": body, "sections": sections, "lead": lead}


def extract_verdict(text: str) -> str | None:
    lowered = (text or "").lower()
    for verdict in _VERDICT_ORDER:
        if verdict in lowered or verdict.replace("_", " ") in lowered:
            return verdict
    return None


def parse_numbers(raw: str) -> list[float | None]:
    if not raw or not raw.strip():
        return []
    parts = re.split(r"[\s,;]+", raw.strip())
    values: list[float | None] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            values.append(None)
    return values


def parse_lines(raw: str) -> list[str]:
    return [line.strip() for line in (raw or "").splitlines() if line.strip()]


def parse_named_groups(raw: str) -> dict[str, list[float | None]]:
    groups: dict[str, list[float | None]] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        for separator in (":", "="):
            if separator in line:
                name, values = line.split(separator, 1)
                groups[name.strip()] = parse_numbers(values)
                break
    return groups


def parse_csv_columns(raw: bytes) -> dict[str, list[float | None]]:
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    columns: dict[str, list[float | None]] = {}
    for row in reader:
        for key, value in row.items():
            if key is None:
                continue
            cell = (value or "").strip()
            try:
                parsed = float(cell)
            except (TypeError, ValueError):
                parsed = None
            columns.setdefault(key, []).append(parsed)
    return columns


def numeric_columns(data: dict[str, list[float | None]]) -> list[str]:
    return [key for key, values in data.items() if any(value is not None for value in values)]


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def method_names(catalog: list[dict[str, Any]]) -> dict[str, str]:
    return {entry["code"]: entry["name"] for entry in catalog}


def summarize_layers(layers: list[list[str]]) -> str:
    rendered = []
    for index, layer in enumerate(layers, start=1):
        names = ", ".join(AGENT_LABELS.get(agent, agent) for agent in layer)
        rendered.append(f"L{index}: {names}")
    return " → ".join(rendered)
