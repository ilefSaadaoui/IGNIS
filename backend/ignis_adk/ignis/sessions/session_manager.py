"""Session management on top of Google ADK's InMemorySessionService.

ADK's InMemorySessionService keeps sessions in process memory only: sessions
are LOST when the process restarts. This is documented and accepted for this
development phase — no durable persistence is claimed or implied.

State layout under session.state:
  summary      -> dict from ConversationSummarizer.to_dict()
  ignis_meta   -> small metadata dict (created/updated, agent calls)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService

from ignis.context.summarizer import ConversationSummarizer

logger = logging.getLogger("ignis.sessions.session_manager")

APP_NAME = "ignis"


class SessionManager:
    """Create / resume / reset isolated IGNIS sessions (in-memory)."""

    def __init__(self, app_name: str = APP_NAME) -> None:
        self.app_name = app_name
        self.service = InMemorySessionService()

    async def create_session(
        self,
        user_id: str = "local_user",
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create a new session; returns its session_id."""
        sid = session_id or f"ignis-{uuid.uuid4().hex[:12]}"
        base_state = {
            "summary": ConversationSummarizer().to_dict(),
            "ignis_meta": {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "agent_calls": 0,
            },
        }
        if state:
            base_state.update(state)
        session = await self.service.create_session(
            app_name=self.app_name, user_id=user_id, state=base_state, session_id=sid
        )
        logger.info("created session %s for user %s", session.id, user_id)
        return session.id

    async def get_session(self, user_id: str, session_id: str):
        return await self.service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )

    async def list_sessions(self, user_id: str):
        return await self.service.list_sessions(app_name=self.app_name, user_id=user_id)

    async def delete_session(self, user_id: str, session_id: str) -> None:
        await self.service.delete_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )

    async def reset_session(self, user_id: str, session_id: str) -> str:
        """Delete and recreate a session with the same id and fresh state."""
        try:
            await self.delete_session(user_id, session_id)
        except Exception as exc:
            logger.warning("reset: could not delete old session %s: %s", session_id, exc)
        return await self.create_session(user_id=user_id, session_id=session_id)

    async def store_summary(self, user_id: str, session_id: str, summarizer: ConversationSummarizer) -> None:
        """Persist the summarizer state into the ADK session state.

        Uses an ADK Event with a state_delta (the canonical ADK mechanism),
        not a direct mutation of the returned Session object.
        """
        session = await self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        meta = dict(session.state.get("ignis_meta", {}))
        meta["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        event = Event(
            author="ignis_session_manager",
            invocation_id=f"ignis-meta-{uuid.uuid4().hex[:8]}",
            actions=EventActions(state_delta={
                "summary": summarizer.to_dict(),
                "ignis_meta": meta,
            }),
        )
        await self.service.append_event(session, event)

    async def load_summarizer(self, user_id: str, session_id: str) -> ConversationSummarizer:
        session = await self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return ConversationSummarizer.from_dict(session.state.get("summary"))
