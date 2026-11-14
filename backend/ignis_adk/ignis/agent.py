"""ADK app entry point. `adk web` discovers the 'ignis' app here."""

from google.adk.apps import App
from google.adk.agents.context_cache_config import ContextCacheConfig

from ignis.agents.root_agent import root_agent as root_agent

# ADK 2.x App wrapper: enables per-agent prefix caching when the provider
# supports it (falls back to normal requests otherwise — never required).
app = App(
    name="ignis",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(cache_intervals=8, ttl_seconds=1800, min_tokens=1024),
)

__all__ = ["app", "root_agent"]
