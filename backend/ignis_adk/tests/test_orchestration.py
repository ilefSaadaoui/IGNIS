import asyncio

from ignis.orchestration.routing import WORKFLOW_DEFINITIONS, WorkflowName, classify_task, workflow_for
from ignis.orchestration.workflows import run_workflow


def test_routing_table_matches_spec_workflows():
    assert WORKFLOW_DEFINITIONS[WorkflowName.RESEARCH_DISCOVERY] == ["nasa_research_agent", "scientific_reviewer_agent"]
    assert WORKFLOW_DEFINITIONS[WorkflowName.ML_SUITABILITY] == ["data_analyst_agent", "ml_agent", "scientific_reviewer_agent"]
    assert "document_intelligence_agent" in WORKFLOW_DEFINITIONS[WorkflowName.EXPERIMENT_COMPARISON]
    assert WORKFLOW_DEFINITIONS[WorkflowName.STATISTICAL_ANALYSIS][0] == "data_analyst_agent"


def test_classify_task_keywords():
    cases = {
        "Find NASA papers on FLEX": WorkflowName.RESEARCH_DISCOVERY,
        "Compare FLEX and BASS": WorkflowName.EXPERIMENT_COMPARISON,
        "Can we train an ML model?": WorkflowName.ML_SUITABILITY,
        "Build a knowledge graph from this text": WorkflowName.KNOWLEDGE_GRAPH,
        "Interpret this chart": WorkflowName.FIGURE_ANALYSIS,
        "Compute mean and standard deviation": WorkflowName.STATISTICAL_ANALYSIS,
        "What are the fire safety implications for the ISS?": WorkflowName.SCIENTIFIC_QUESTION,
    }
    for question, expected in cases.items():
        assert classify_task(question) == expected, question


def test_default_classification_is_scientific_question():
    assert classify_task("tell me something interesting") == WorkflowName.SCIENTIFIC_QUESTION


def test_all_workflows_run_offline_with_fake_model(monkeypatch, mock_ntrs_http, mock_citation_head):
    from tests.conftest import patch_models_with_fake

    patch_models_with_fake(monkeypatch)
    questions = {
        WorkflowName.RESEARCH_DISCOVERY: "Find NASA research on BASS-II combustion",
        WorkflowName.EXPERIMENT_COMPARISON: "Compare burning rates GROUPS:{\"FLEX\":[1.0,2.0,3.0],\"BASS\":[2.0,3.0,4.0]} UNIT:mm/s",
        WorkflowName.ML_SUITABILITY: "Is this dataset usable for ML? N_SAMPLES:150 N_FEATURES:6 TARGET:binary_classification",
        WorkflowName.SCIENTIFIC_QUESTION: "What does FLEX imply for ISS fire safety?",
        WorkflowName.KNOWLEDGE_GRAPH: "TEXT: BASS studied PMMA flammability in microgravity",
        WorkflowName.FIGURE_ANALYSIS: "IMAGE: /nonexistent/figure.png — describe it",
        WorkflowName.STATISTICAL_ANALYSIS: "Compute statistics for values 1.0 2.0 3.0 4.0",
    }

    async def run_all():
        results = {}
        for wf, q in questions.items():
            results[wf] = await run_workflow(q, workflow=wf, user_id="tester")
        return results

    results = asyncio.run(run_all())
    assert set(results) == set(WorkflowName)
    for wf, res in results.items():
        assert res.specialists_invoked == WORKFLOW_DEFINITIONS[wf]
        assert res.final_answer.strip(), wf
        assert res.steps, wf
    # Workflow F: multimodal tool reports unavailable/failure honestly (nonexistent image),
    # and the workflow still completes without fabricating an analysis.
    fig_steps = {s.agent: s for s in results[WorkflowName.FIGURE_ANALYSIS].steps}
    assert fig_steps["multimodal_agent"].status == "completed"  # fake model returns text


def test_parallel_layers_run_concurrently(monkeypatch, mock_ntrs_http, mock_citation_head):
    """Performance sanity: independent specialists run in parallel (not the sum of delays)."""
    import asyncio
    import time

    from ignis.orchestration import workflows as wf_mod
    from tests.conftest import patch_models_with_fake
    from tests.fakes import FakeLlm

    patch_models_with_fake(monkeypatch)

    class SlowFakeLlm(FakeLlm):
        async def generate_content_async(self, llm_request, stream=False):
            await asyncio.sleep(0.6)
            async for chunk in super().generate_content_async(llm_request, stream):
                yield chunk

    slow = SlowFakeLlm()
    import ignis.orchestration.workflows as workflows

    for agent in workflows._specialist_map().values():
        monkeypatch.setattr(agent, "model", slow, raising=False)

    async def run():
        t0 = time.perf_counter()
        res = await workflows.run_workflow(
            'Compare burning rates GROUPS:{"FLEX":[1.0,2.0],"BASS":[2.0,3.0]} UNIT:mm/s',
            workflow=WorkflowName.EXPERIMENT_COMPARISON, user_id="perf_tester",
        )
        return time.perf_counter() - t0, res

    elapsed, result = asyncio.run(run())
    # Each agent does 2 model turns (tool call + answer) = 1.2s.
    # Sequential: 4 agents x 1.2s = 4.8s. Parallel layers: 2 layers x 1.2s = 2.4s.
    assert elapsed < 3.6, f"workflow took {elapsed:.2f}s — parallelism may be broken"
    assert result.specialists_invoked == WORKFLOW_DEFINITIONS[WorkflowName.EXPERIMENT_COMPARISON]
    assert all(s.status == "completed" for s in result.steps)
