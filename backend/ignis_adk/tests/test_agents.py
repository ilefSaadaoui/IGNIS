from ignis.agents import root_agent
import ignis.agents as agents_pkg

EXPECTED_SPECIALISTS = {
    "nasa_research_agent", "document_intelligence_agent", "data_analyst_agent", "ml_agent",
    "knowledge_graph_agent", "multimodal_agent", "fire_safety_agent", "scientific_reviewer_agent",
}


def test_all_specialists_implemented_and_named():
    subs = {a.name for a in root_agent.sub_agents}
    assert EXPECTED_SPECIALISTS == subs


def test_specialists_have_groq_models_and_tools():
    for agent in root_agent.sub_agents:
        model = agent.model
        model_name = model if isinstance(model, str) else getattr(model, "model", "")
        assert "groq/" in model_name, f"{agent.name} must use Groq via LiteLLM, got {model_name}"
        assert agent.tools, f"{agent.name} must have functional tools"
        assert agent.instruction.strip(), f"{agent.name} must have instructions"


def test_root_has_no_direct_tools_only_subagents():
    assert not root_agent.tools, "root orchestrator delegates via sub-agents, not direct tools"
    assert root_agent.name == "ignis"


def test_no_gemini_anywhere():
    import ignis
    for agent in [root_agent, *root_agent.sub_agents]:
        model = agent.model
        model_name = model if isinstance(model, str) else getattr(model, "model", "")
        assert "gemini" not in model_name.lower()


def test_adk_app_entry_exposes_root_agent():
    from ignis.agent import root_agent as entry
    assert entry is root_agent
