"""End-to-end workflow tests (offline, deterministic FakeLlm + mocked NASA/citation HTTP).

Live integration tests live in test_live_integration.py and are skipped without
an explicit GROQ_API_KEY.
"""

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from ignis.orchestration.workflows import run_workflow
from ignis.orchestration.routing import WorkflowName
from tests.conftest import patch_models_with_fake


def test_workflow_a_research_discovery_e2e(monkeypatch, mock_ntrs_http, mock_citation_head):
    patch_models_with_fake(monkeypatch)

    async def scenario():
        return await run_workflow(
            "Find NASA research on BASS-II combustion experiments",
            workflow=WorkflowName.RESEARCH_DISCOVERY, user_id="e2e_user",
        )

    result = asyncio.run(scenario())
    nasa_step = next(s for s in result.steps if s.agent == "nasa_research_agent")
    assert nasa_step.status == "completed"
    assert "BASS" in nasa_step.output_excerpt or "NTRS" in nasa_step.output_excerpt
    assert result.specialists_invoked == ["nasa_research_agent", "scientific_reviewer_agent"]
    assert result.final_answer.strip()


def test_workflow_b_experiment_comparison_e2e(monkeypatch, mock_ntrs_http, mock_citation_head):
    patch_models_with_fake(monkeypatch)

    async def scenario():
        return await run_workflow(
            'Compare burning rates GROUPS:{"FLEX":[1.0,2.0,3.0],"BASS":[2.0,3.0,4.0]} UNIT:mm/s',
            workflow=WorkflowName.EXPERIMENT_COMPARISON, user_id="e2e_user",
        )

    result = asyncio.run(scenario())
    assert result.specialists_invoked == [
        "nasa_research_agent", "document_intelligence_agent", "data_analyst_agent", "scientific_reviewer_agent",
    ]
    statuses = {s.agent: s.status for s in result.steps}
    assert statuses["nasa_research_agent"] == "completed"
    assert statuses["data_analyst_agent"] == "completed"
    assert result.final_answer.strip()


def test_workflow_c_ml_suitability_e2e(monkeypatch, mock_ntrs_http, mock_citation_head):
    patch_models_with_fake(monkeypatch)

    async def scenario():
        return await run_workflow(
            "Is this dataset suitable for ML? N_SAMPLES:150 N_FEATURES:6 TARGET:binary_classification",
            workflow=WorkflowName.ML_SUITABILITY, user_id="e2e_user",
        )

    result = asyncio.run(scenario())
    assert result.specialists_invoked == ["data_analyst_agent", "ml_agent", "scientific_reviewer_agent"]
    ml_step = next(s for s in result.steps if s.agent == "ml_agent")
    assert ml_step.status == "completed"
    assert result.final_answer.strip()


def test_root_agent_transfer_path_offline(monkeypatch, mock_ntrs_http):
    """Exercise ADK's real transfer_to_agent delegation from the ROOT agent."""
    patch_models_with_fake(monkeypatch)

    from ignis.agents.root_agent import root_agent

    async def scenario():
        session_service = InMemorySessionService()
        sid = "root-e2e-session"
        await session_service.create_session(app_name="ignis", user_id="e2e_user", session_id=sid)
        runner = Runner(agent=root_agent, app_name="ignis", session_service=session_service)
        message = genai_types.Content(
            role="user", parts=[genai_types.Part(text="Find NASA research on BASS combustion experiments")]
        )
        authors = []
        text_out = ""
        async for event in runner.run_async(user_id="e2e_user", session_id=sid, new_message=message):
            authors.append(getattr(event, "author", ""))
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    if getattr(part, "text", None):
                        text_out += part.text
        return authors, text_out

    authors, text_out = asyncio.run(scenario())
    assert "nasa_research_agent" in authors, f"expected transfer to nasa_research_agent; authors={authors}"
    assert text_out.strip()


def test_specialist_failure_is_structured_not_fabricated(monkeypatch, mock_ntrs_http):
    """If a specialist raises, the workflow records a structured failure."""
    patch_models_with_fake(monkeypatch)
    import ignis.orchestration.workflows as wf_mod

    class ExplodingRunner:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(wf_mod, "Runner", ExplodingRunner)

    async def scenario():
        return await run_workflow(
            "Find NASA research on BASS", workflow=WorkflowName.RESEARCH_DISCOVERY, user_id="e2e_user"
        )

    result = asyncio.run(scenario())
    assert all(s.status == "failed" for s in result.steps)
    assert all(s.error and "simulated provider outage" in s.error for s in result.steps)
    assert any("failed" in lim.lower() for lim in result.limitations)
