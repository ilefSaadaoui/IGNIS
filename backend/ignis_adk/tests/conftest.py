"""Shared pytest fixtures. No test here needs a live Groq key or network,
except those explicitly marked with @pytest.mark.integration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.fakes import FakeLlm


@pytest.fixture()
def fake_llm() -> FakeLlm:
    return FakeLlm()


def patch_models_with_fake(monkeypatch: pytest.MonkeyPatch):
    """Swap every agent's model onto the deterministic FakeLlm (offline E2E)."""
    from ignis.agents import root_agent as root_mod
    import ignis.agents as agents_pkg

    fake = FakeLlm()
    for module_name in [
        "root_agent", "nasa_research_agent", "document_intelligence_agent", "data_analyst_agent",
        "ml_agent", "knowledge_graph_agent", "multimodal_agent", "fire_safety_agent", "scientific_reviewer_agent",
    ]:
        module = getattr(__import__(f"ignis.agents.{module_name}", fromlist=[module_name]), module_name)
        monkeypatch.setattr(module, "model", fake, raising=False)
    # Also patch the already-assembled root (it captured sub-agents at construction).
    monkeypatch.setattr(root_mod.root_agent, "model", fake, raising=False)
    for sub in root_mod.root_agent.sub_agents:
        monkeypatch.setattr(sub, "model", fake, raising=False)
    return fake


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


NTRS_SAMPLE_PAYLOAD = {
    "stats": {"took": 5, "total": 2, "estimate": False, "maxScore": 100.0},
    "results": [
        {
            "id": 20160000593,
            "title": "Combustion of Solids in Microgravity: Results from the BASS-II Experiment",
            "stiType": "JOURNAL_ARTICLE",
            "distributionDate": "2016-03-01T00:00:00.0000000+00:00",
            "abstract": "BASS-II investigated burning rates of solid fuels in microgravity aboard the ISS.",
            "keywords": ["combustion", "microgravity", "BASS-II"],
            "distribution": "PUBLIC",
            "authorAffiliations": [{"meta": {"author": {"name": "Ferkul, Paul V."}}}],
            "center": {"code": "GRC", "name": "Glenn Research Center"},
            "downloads": [{
                "links": {
                    "pdf": "/api/citations/20160000593/downloads/20160000593.pdf",
                    "fulltext": "/api/citations/20160000593/downloads/20160000593.txt",
                }
            }],
        },
        {
            "id": 20140011099,
            "title": "Thickness and Fuel Preheating Effects on Material Flammability in Microgravity",
            "stiType": "CONFERENCE_PAPER",
            "distributionDate": "2014-06-01T00:00:00.0000000+00:00",
            "abstract": "Laminar flame spread and flammability limits of solid fuels in microgravity.",
            "keywords": ["flammability", "microgravity"],
            "distribution": "PUBLIC",
            "authorAffiliations": [],
            "center": {"code": "GRC", "name": "Glenn Research Center"},
            "downloads": [],
        },
    ],
}


@pytest.fixture()
def mock_ntrs_http(monkeypatch: pytest.MonkeyPatch):
    """Mock NTRS HTTP so NASA tools run offline deterministically."""
    import ignis.tools.nasa_tools as nasa_tools

    def fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        assert "ntrs.nasa.gov" in url
        return _FakeResponse(200, NTRS_SAMPLE_PAYLOAD)

    monkeypatch.setattr(nasa_tools.requests, "get", fake_get)
    return NTRS_SAMPLE_PAYLOAD


@pytest.fixture()
def mock_citation_head(monkeypatch: pytest.MonkeyPatch):
    import ignis.tools.review_tools as review_tools

    def fake_head(url, timeout=None, allow_redirects=True, headers=None):  # noqa: ARG001
        if "broken" in url:
            return _FakeResponse(404, None, "not found")
        return _FakeResponse(200, None, "ok")

    monkeypatch.setattr(review_tools.requests, "head", fake_head)


@pytest.fixture(autouse=True)
def _dense_off_by_default(monkeypatch):
    """Keep the suite offline and deterministic: dense retrieval is opt-in per test."""
    monkeypatch.setenv("IGNIS_DENSE", "off")
