"""Programmatic workflow execution for IGNIS.

Each specialist runs as its own ADK Runner step against a SHARED in-memory
session, so the execution log shows exactly which specialists were invoked.
The interactive path (`adk web`) instead uses the root agent's ADK transfer
routing — both paths are supported and documented.

No specialist output is fabricated: if a step raises, the failure is recorded
in the execution log and passed as context (clearly marked) to later steps;
the final answer must report the failure.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from ignis.config.settings import get_settings
from ignis.context.context_builder import ContextBuilder
from ignis.context.summarizer import ConversationSummarizer
from ignis.context.token_manager import get_token_manager
from ignis.orchestration.routing import WORKFLOW_DEFINITIONS, WORKFLOW_LAYERS, WorkflowName, classify_task, WORKFLOW_DESCRIPTIONS

logger = logging.getLogger("ignis.orchestration.workflows")

APP_NAME = "ignis"


def _specialist_map() -> dict[str, Any]:
    from ignis.agents.data_analyst_agent import data_analyst_agent
    from ignis.agents.document_intelligence_agent import document_intelligence_agent
    from ignis.agents.fire_safety_agent import fire_safety_agent
    from ignis.agents.knowledge_graph_agent import knowledge_graph_agent
    from ignis.agents.ml_agent import ml_agent
    from ignis.agents.multimodal_agent import multimodal_agent
    from ignis.agents.nasa_research_agent import nasa_research_agent
    from ignis.agents.scientific_reviewer_agent import scientific_reviewer_agent

    return {
        "nasa_research_agent": nasa_research_agent,
        "document_intelligence_agent": document_intelligence_agent,
        "data_analyst_agent": data_analyst_agent,
        "ml_agent": ml_agent,
        "knowledge_graph_agent": knowledge_graph_agent,
        "multimodal_agent": multimodal_agent,
        "fire_safety_agent": fire_safety_agent,
        "scientific_reviewer_agent": scientific_reviewer_agent,
    }


@dataclass
class WorkflowStepLog:
    step: int
    agent: str
    status: str  # completed | failed | skipped
    output_excerpt: str
    elapsed_seconds: float
    error: str | None = None


@dataclass
class WorkflowResult:
    workflow: str
    workflow_description: str
    question: str
    session_id: str
    final_answer: str
    steps: list[WorkflowStepLog] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    @property
    def specialists_invoked(self) -> list[str]:
        return [s.agent for s in self.steps]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "workflow_description": self.workflow_description,
            "question": self.question,
            "session_id": self.session_id,
            "specialists_invoked": self.specialists_invoked,
            "final_answer": self.final_answer,
            "steps": [s.__dict__ for s in self.steps],
            "summary": self.summary,
            "limitations": self.limitations,
        }


async def run_workflow(
    question: str,
    workflow: str | WorkflowName | None = None,
    *,
    user_id: str = "local_user",
    session_id: str | None = None,
    include_reviewer: bool = True,
) -> WorkflowResult:
    """Run an IGNIS workflow against a shared in-memory session.

    Args:
        question: The research question (optionally prefixed with document text path etc.).
        workflow: 'A'..'G', a WorkflowName, or None to auto-classify from the question.
        user_id / session_id: session identity; pass session_id to continue a session.
        include_reviewer: set False only for cheap G-class statistics; reviewer otherwise kept.
    """
    settings = get_settings()
    tm = get_token_manager()
    builder = ContextBuilder()

    if isinstance(workflow, WorkflowName):
        wf = workflow
    elif isinstance(workflow, str) and workflow.strip().upper() in WorkflowName.__members__.values():
        # 'A'..'G'
        wf = WorkflowName(workflow.strip().upper())
    elif isinstance(workflow, str) and workflow.strip():
        try:
            wf = WorkflowName[workflow.strip().upper()]
        except KeyError:
            wf = classify_task(question)
    else:
        wf = classify_task(question)

    layers = [list(layer) for layer in WORKFLOW_LAYERS[wf]]
    if not include_reviewer:
        layers = [layer for layer in layers if layer != ["scientific_reviewer_agent"]]

    session_service = InMemorySessionService()
    sid = session_id or f"ignis-wf-{int(time.time() * 1000)}"
    await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=sid,
                                         state={"summary": ConversationSummarizer().to_dict()})

    agents = _specialist_map()
    summarizer = ConversationSummarizer()
    summarizer.set_question(question)

    result = WorkflowResult(
        workflow=wf.value,
        workflow_description=WORKFLOW_DESCRIPTIONS[wf],
        question=question,
        session_id=sid,
        final_answer="",
    )

    context_so_far = question
    step_counter = 0

    async def _run_one(name: str, step_idx: int, context_input: str) -> tuple[WorkflowStepLog, str]:
        agent = agents.get(name)
        if agent is None:
            return (WorkflowStepLog(step_idx, name, "skipped", "", 0.0, "unknown specialist"), "")
        built = None
        started = time.perf_counter()
        output_text = ""
        status = "completed"
        error: str | None = None
        try:
            built = builder.build_for_agent(
                name,
                question=question,
                summarizer=summarizer,
                evidence_blocks=[f"Context accumulated from previous layers (may be AI-generated; treat with care):\n{context_input}"]
                if context_input != question else None,
            )
            # Isolated branch session per parallel agent — no interleaved state.
            branch_sid = f"{sid}-{name}-{step_idx}"
            await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=branch_sid,
                                                 state={"summary": summarizer.to_dict()})
            app = None
            try:
                from google.adk.apps import App
                from google.adk.agents.context_cache_config import ContextCacheConfig

                app = App(name=APP_NAME, root_agent=agent,
                          context_cache_config=ContextCacheConfig(cache_intervals=8, ttl_seconds=1800, min_tokens=1024))
            except Exception:
                app = None  # caching is an optimization; never required for correctness
            runner = Runner(app=app, agent=None if app else agent, app_name=None if app else APP_NAME,
                            session_service=session_service)
            message = genai_types.Content(role="user", parts=[genai_types.Part(text=built.text)])
            async for event in runner.run_async(user_id=user_id, session_id=branch_sid, new_message=message):
                content = getattr(event, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            output_text += text
            if not output_text.strip():
                status, error = "failed", "agent produced no text output"
        except Exception as exc:  # provider errors, auth failures, tool crashes — reported, not hidden
            logger.exception("specialist %s failed", name)
            status, error = "failed", f"{exc.__class__.__name__}: {exc}"
            output_text = ""
        elapsed = time.perf_counter() - started
        output_text, _ = tm.enforce_output_limit(output_text)
        if built is not None and built.truncated:
            result.limitations.append(f"Input to {name} was truncated to fit the token budget.")
        return (WorkflowStepLog(step_idx, name, status, output_text[:600], round(elapsed, 2), error), output_text)

    for layer in layers:
        # stable step numbering across layers, preserving definition order
        planned = []
        for name in layer:
            step_counter += 1
            planned.append((step_counter, name))
        logs, outputs = [], []
        results_layer = await asyncio.gather(*[_run_one(name, idx, context_so_far) for idx, name in planned],
                                             return_exceptions=True)
        for (idx, name), outcome in zip(planned, results_layer):
            if isinstance(outcome, Exception):
                log = WorkflowStepLog(idx, name, "failed", "", 0.0, f"{outcome.__class__.__name__}: {outcome}")
                out_text = ""
            else:
                log, out_text = outcome
            logs.append(log)
            outputs.append((name, log.status, out_text, log.error))
        # stable order = definition order within the layer
        logs.sort(key=lambda l: l.step)
        result.steps.extend(logs)
        merged_parts = []
        for name, status, out_text, error in outputs:
            summarizer.record_agent_action(name, f"workflow {wf.value} layer step", status)
            if status == "completed" and out_text.strip():
                merged_parts.append(f"[{name} output]\n{out_text}")
                summarizer.add_unverified_claim(f"[{name}] {out_text[:400]}")
            else:
                result.limitations.append(f"Specialist '{name}' failed: {error}")
                summarizer.add_limitation(f"{name} failed during workflow {wf.value}: {error}")
        if merged_parts:
            context_so_far = context_so_far + "\n\n" + "\n\n".join(merged_parts)

    result.final_answer = context_so_far
    result.summary = summarizer.to_dict()
    if not settings.groq_api_key_configured:
        result.limitations.append(
            "GROQ_API_KEY is missing or a placeholder — live specialist inference cannot succeed; "
            "set a real key in .env to execute this workflow end-to-end."
        )
    return result
