"""IGNIS — Production FastAPI Server for Cloud Deployment (Render / Docker / VM).

Exposes the Multi-Agent Scientific Research Engine (Google ADK + Groq)
via standard REST endpoints for web and mobile frontends.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ignis.config.settings import get_settings
from ignis.config.model_config import is_groq_configured
from ignis.orchestration.routing import WORKFLOW_DESCRIPTIONS, WorkflowName
from ignis.orchestration.workflows import run_workflow
from ignis.tools.literature_tools import get_corpus_status, search_scientific_literature
from ignis.tools.nasa_tools import search_nasa_research, search_nasa_images, retrieve_investigation_metadata
from ignis.tools.data_tools import calculate_descriptive_statistics, compare_experimental_data
from ignis.tools.graph_tools import build_graph_representation, extract_entities, extract_relationships

logger = logging.getLogger("ignis.server")

app = FastAPI(
    title="IGNIS — Space Combustion Intelligence API",
    description="Evidence-grounded multi-agent research system for NASA microgravity combustion data.",
    version="1.0.0",
)

# Enable CORS for frontend integration (Streamlit, React, or mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Models ---

class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language research question")
    workflow: str | None = Field(default=None, description="Workflow code 'A'..'G' or None for auto-classification")
    user_id: str = Field(default="api_user", description="Identifier for session tracking")
    session_id: str | None = Field(default=None, description="Optional persistent session id")
    include_reviewer: bool = Field(default=True, description="Enforce Scientific Reviewer validation")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    investigation: str | None = Field(default=None, description="e.g. 'BASS', 'FLEX', 'ACME', 'SAFFIRE'")


class CompareRequest(BaseModel):
    groups: dict[str, list[float | None]]
    unit: str | None = None
    label_a: str | None = None
    label_b: str | None = None


class GraphRequest(BaseModel):
    text: str


# --- Endpoints ---

@app.get("/")
@app.get("/health")
def health_check() -> dict[str, Any]:
    """Healthcheck endpoint for Render / Kubernetes."""
    return {
        "status": "healthy",
        "service": "IGNIS Space Combustion Intelligence",
        "team": "RaoSpace Labs (NASA Space Apps 2026)",
        "groq_configured": is_groq_configured(),
    }


@app.get("/api/info")
def system_info() -> dict[str, Any]:
    """Return backend status, active models, token budgets, and workflow definitions."""
    s = get_settings()
    try:
        from ignis.corpus.store import get_corpus_index
        corpus = get_corpus_index().stats()
    except Exception as exc:
        corpus = {"status": "unavailable", "error": str(exc)}

    return {
        "project": "IGNIS — Space Combustion Intelligence",
        "models": {
            "default": s.GROQ_MODEL,
            "reasoning": s.GROQ_REASONING_MODEL,
            "fast": s.GROQ_FAST_MODEL,
            "vision": s.GROQ_VISION_MODEL,
        },
        "budgets": {
            "MAX_INPUT_TOKENS": s.MAX_INPUT_TOKENS,
            "MAX_OUTPUT_TOKENS": s.MAX_OUTPUT_TOKENS,
            "MAX_CONTEXT_TOKENS": s.MAX_CONTEXT_TOKENS,
        },
        "corpus": corpus,
        "workflows": {k.value: v for k, v in WORKFLOW_DESCRIPTIONS.items()},
    }


@app.get("/api/corpus/stats")
def corpus_stats() -> dict[str, Any]:
    """Return stats on the real local NASA NTRS literature corpus."""
    try:
        payload = get_corpus_status()
        return payload.get("corpus", payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ask")
async def ask_workflow(req: AskRequest) -> dict[str, Any]:
    """Execute an end-to-end multi-agent research workflow (Workflows A to G)."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = await run_workflow(
            question,
            workflow=req.workflow,
            user_id=req.user_id,
            session_id=req.session_id,
            include_reviewer=req.include_reviewer,
        )
        return result.as_dict()
    except Exception as exc:
        logger.exception("Error executing workflow")
        raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}")


@app.post("/api/search")
def search_literature(req: SearchRequest) -> dict[str, Any]:
    """Search the 102 NASA NTRS papers via hybrid retrieval (TF-IDF + MiniLM dense)."""
    try:
        return search_scientific_literature(req.query, top_k=req.top_k, investigation=req.investigation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/nasa/search")
def search_live_nasa(query: str, source: str = "all", limit: int = 5) -> dict[str, Any]:
    """Live search on NASA NTRS public servers."""
    try:
        return search_nasa_research(query, source=source, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/nasa/images")
def search_nasa_image_library(query: str, media_type: str = "image", page_size: int = 5) -> dict[str, Any]:
    """Search the NASA Image and Video Library."""
    try:
        return search_nasa_images(query, media_type=media_type, page_size=page_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/compare")
def compare_data(req: CompareRequest) -> dict[str, Any]:
    """Compare experimental combustion datasets deterministically."""
    try:
        return compare_experimental_data(
            req.groups,
            unit=req.unit,
            label_a=req.label_a,
            label_b=req.label_b,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/graph")
def extract_graph(req: GraphRequest) -> dict[str, Any]:
    """Extract entities and build graph representation from scientific text."""
    try:
        return build_graph_representation(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

