"""IGNIS — developer CLI.

Examples (Windows PowerShell):
  python main.py info
  python main.py ask "Find NASA research on the FLEX investigation"
  python main.py ask "Compare burning rates" --workflow B --session-id my-session
  python main.py repl
  python main.py sessions --user local_user
  python main.py reset --session-id my-session
  adk web          # official ADK developer UI (discovers the 'ignis' app)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ignis.config.settings import get_settings
from ignis.config.model_config import is_groq_configured
from ignis.orchestration.routing import WORKFLOW_DESCRIPTIONS
from ignis.orchestration.workflows import run_workflow
from ignis.sessions import SessionManager


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_info(_args) -> int:
    s = get_settings()
    try:
        from ignis.corpus.store import get_corpus_index

        corpus = get_corpus_index().stats()
    except Exception as exc:
        corpus = {"status": "unavailable", "error": str(exc)}
    _print_json({
        "corpus": corpus,
        "project": "IGNIS — Space Combustion Intelligence (RaoSpace Labs)",
        "python_requires": ">=3.11",
        "groq_api_key_configured": is_groq_configured(),
        "groq_api_key_note": "(key value is never printed)",
        "models": {
            "default": s.GROQ_MODEL,
            "reasoning": s.GROQ_REASONING_MODEL,
            "vision": s.GROQ_VISION_MODEL,
        },
        "budgets": {
            "MAX_INPUT_TOKENS": s.MAX_INPUT_TOKENS,
            "MAX_OUTPUT_TOKENS": s.MAX_OUTPUT_TOKENS,
            "MAX_CONTEXT_TOKENS": s.MAX_CONTEXT_TOKENS,
            "MAX_AGENT_ITERATIONS": s.MAX_AGENT_ITERATIONS,
            "MAX_CONVERSATION_HISTORY": s.MAX_CONVERSATION_HISTORY,
            "MAX_DOCUMENT_CHARS": s.MAX_DOCUMENT_CHARS,
            "MAX_AGENT_OUTPUT_CHARS": s.MAX_AGENT_OUTPUT_CHARS,
            "REQUEST_TIMEOUT": s.REQUEST_TIMEOUT,
            "MAX_RETRIES": s.MAX_RETRIES,
        },
        "issues": s.missing_credentials(),
        "workflows": {k.value: v for k, v in WORKFLOW_DESCRIPTIONS.items()},
        "sessions": "in-memory (lost on process restart; not durable)",
    })
    return 0


async def cmd_ask(args) -> int:
    result = await run_workflow(
        args.question,
        workflow=args.workflow,
        user_id=args.user,
        session_id=args.session_id,
        include_reviewer=not args.skip_reviewer,
    )
    print("=" * 72)
    print(f"Workflow {result.workflow} — {result.workflow_description}")
    print(f"Specialists invoked: {', '.join(result.specialists_invoked)}")
    print("=" * 72)
    for step in result.steps:
        marker = "OK " if step.status == "completed" else "ERR"
        print(f"[{marker}] step {step.step}: {step.agent} ({step.elapsed_seconds}s)"
              + (f" — {step.error}" if step.error else ""))
    print("=" * 72)
    print(result.final_answer)
    if result.limitations:
        print("-" * 72)
        print("Limitations:")
        for lim in result.limitations:
            print(f"  - {lim}")
    if args.json:
        _print_json(result.as_dict())
    return 0 if all(s.status == "completed" for s in result.steps) else 2


async def cmd_repl(args) -> int:
    s = get_settings()
    print("IGNIS interactive session (in-memory; state lost on exit). Ctrl+C or empty line to quit.")
    if not s.groq_api_key_configured:
        print("WARNING: GROQ_API_KEY is missing/placeholder — live inference will fail. Run 'python main.py info'.")
    sm = SessionManager()
    sid = args.session_id or await sm.create_session(user_id=args.user)
    print(f"Session: {sid} (user={args.user})\n")
    while True:
        try:
            question = input("ignis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        class _A:  # minimal arg shim
            pass
        a = _A()
        a.question, a.workflow, a.user, a.session_id, a.skip_reviewer, a.json = question, args.workflow, args.user, sid, False, False
        await cmd_ask(a)
    return 0


async def cmd_sessions(args) -> int:
    sm = SessionManager()
    listing = await sm.list_sessions(args.user)
    sessions = getattr(listing, "sessions", []) or []
    _print_json({"user": args.user, "sessions": [getattr(s, "id", str(s)) for s in sessions]})
    return 0


async def cmd_reset(args) -> int:
    sm = SessionManager()
    sid = await sm.reset_session(args.user, args.session_id)
    print(f"Session reset: {sid}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="IGNIS — Space Combustion Intelligence developer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show configuration (never prints secrets)").set_defaults(func=cmd_info)

    p_ask = sub.add_parser("ask", help="Ask a research question (single workflow run)")
    p_ask.add_argument("question")
    p_ask.add_argument("--workflow", default=None, help="A..G; default: auto-classify")
    p_ask.add_argument("--user", default="local_user")
    p_ask.add_argument("--session-id", default=None)
    p_ask.add_argument("--skip-reviewer", action="store_true", help="Skip the reviewer step (not recommended)")
    p_ask.add_argument("--json", action="store_true", help="Also print full structured result as JSON")
    p_ask.set_defaults(func=lambda a: asyncio.run(cmd_ask(a)))

    p_repl = sub.add_parser("repl", help="Interactive Q&A against one session")
    p_repl.add_argument("--workflow", default=None)
    p_repl.add_argument("--user", default="local_user")
    p_repl.add_argument("--session-id", default=None)
    p_repl.set_defaults(func=lambda a: asyncio.run(cmd_repl(a)))

    p_ls = sub.add_parser("sessions", help="List sessions for a user (this process only)")
    p_ls.add_argument("--user", default="local_user")
    p_ls.set_defaults(func=lambda a: asyncio.run(cmd_sessions(a)))

    p_reset = sub.add_parser("reset", help="Reset (delete + recreate) a session")
    p_reset.add_argument("--session-id", required=True)
    p_reset.add_argument("--user", default="local_user")
    p_reset.set_defaults(func=lambda a: asyncio.run(cmd_reset(a)))

    args = parser.parse_args()
    get_settings().configure_logging()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
