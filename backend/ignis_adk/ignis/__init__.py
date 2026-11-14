"""IGNIS — Space Combustion Intelligence.

The package is lazily loaded: importing `ignis.tools` or `ignis.corpus` does NOT
import Google ADK. ADK's agent loader resolves `ignis.root_agent` via this
__getattr__ (PEP 562), so `adk web` still discovers the app.
"""

__version__ = "1.0.0"

__all__ = ["agent", "app", "root_agent", "__version__"]


def __getattr__(name: str):
    if name == "agent":
        import importlib

        return importlib.import_module(".agent", __name__)
    if name == "app":
        from ignis.agent import app

        return app
    if name == "root_agent":
        from ignis.agents.root_agent import root_agent

        return root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
