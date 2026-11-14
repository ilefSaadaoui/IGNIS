"""Live integration tests — require a real GROQ_API_KEY and network.

Run explicitly with:  pytest -m integration -v
They are skipped automatically when GROQ_API_KEY is missing/placeholder.
"""

import os

import pytest

from ignis.config.settings import get_settings

pytestmark = pytest.mark.integration


def _key_ready() -> bool:
    return get_settings().groq_api_key_configured


requires_groq = pytest.mark.skipif(not _key_ready(), reason="GROQ_API_KEY missing or placeholder")


@requires_groq
def test_live_groq_inference_single_agent():
    import asyncio

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

    from ignis.agents.data_analyst_agent import data_analyst_agent

    async def scenario():
        session_service = InMemorySessionService()
        await session_service.create_session(app_name="ignis", user_id="live", session_id="live-1")
        runner = Runner(agent=data_analyst_agent, app_name="ignis", session_service=session_service)
        msg = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="Compute descriptive statistics for values [1.0, 2.0, 3.0, 4.0, 5.0] unit=kPa using your tool.")],
        )
        texts = []
        async for event in runner.run_async(user_id="live", session_id="live-1", new_message=msg):
            content = getattr(event, "content", None)
            if content and content.parts:
                texts.extend(p.text for p in content.parts if getattr(p, "text", None))
        return "".join(texts)

    out = asyncio.run(scenario())
    assert out.strip()
    print("\nLIVE GROQ OUTPUT:\n", out)


@pytest.mark.skipif(os.environ.get("IGNIS_LIVE_NASA", "0") != "1", reason="set IGNIS_LIVE_NASA=1 to run")
def test_live_nasa_ntrs_retrieval():
    from ignis.tools.nasa_tools import search_nasa_research

    r = search_nasa_research("microgravity combustion BASS", source="ntrs", limit=2)
    assert r["status"] == "completed"
    assert r["records"], r
