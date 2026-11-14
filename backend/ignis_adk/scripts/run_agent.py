"""Minimal programmatic IGNIS example (offline-safe: builds agents, no live calls)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ignis.orchestration.workflows import run_workflow  # noqa: E402


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "Find NASA research on the Smoke Aerosol Measurement Experiment (SAME)."
    result = await run_workflow(question, workflow=None, user_id="local_user")
    print(f"Workflow {result.workflow} — specialists: {', '.join(result.specialists_invoked)}")
    print(result.final_answer)
    for lim in result.limitations:
        print(f"limitation: {lim}")


if __name__ == "__main__":
    asyncio.run(main())
