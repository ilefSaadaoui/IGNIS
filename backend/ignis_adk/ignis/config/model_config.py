"""Model configuration and multi-provider model wiring for IGNIS.

Google ADK is the orchestration framework. Groq and OpenRouter use ADK's
LiteLLM integration, while Gemini uses ADK's native Gemini integration.
Configured providers are tried in ``IGNIS_PROVIDER_ORDER`` order. The default
chain is Groq -> OpenRouter because ADK 2.14 cannot safely resume a tool-call
conversation by switching from LiteLLM to Gemini; Gemini remains configurable
for standalone/non-tool agents.

Groq does NOT expose Gemini models here; no Gemini API is used anywhere in
this project. Model names are configurable through the environment.

Transient provider failures (rate limits, overloads, and server errors) move to
the next configured provider while preserving the ADK request context.
"""

from __future__ import annotations

import logging
import os

from ignis.config.settings import get_settings

logger = logging.getLogger("ignis.config.model_config")

# Known 131k-class context windows on Groq (Llama 3.x / 4 / GPT-OSS family).
# Approximate; Groq publishes per-model limits and they change over time.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-90b-vision-preview": 8192,
    "meta-llama/llama-4-scout-17b-16e-instruct": 131072,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 131072,
    "openai/gpt-oss-120b": 131072,
    "openai/gpt-oss-20b": 131072,
    "qwen/qwen3-32b": 131072,
    "moonshotai/kimi-k2-instruct": 131072,
    "deepseek-r1-distill-llama-70b": 131072,
}
_DEFAULT_CONTEXT_WINDOW = 128000

# Per-agent input-token budgets (share of MAX_INPUT_TOKENS each agent may see).
AGENT_TOKEN_BUDGETS: dict[str, int] = {
    "root_orchestrator": 6000,
    "nasa_research_agent": 4000,
    "document_intelligence_agent": 6000,
    "data_analyst_agent": 4000,
    "ml_agent": 4000,
    "knowledge_graph_agent": 5000,
    "multimodal_agent": 3000,
    "fire_safety_agent": 5000,
    "scientific_reviewer_agent": 6000,
}


def _strip_prefix(model_name: str) -> str:
    name = model_name.strip()
    return name[len("groq/"):] if name.startswith("groq/") else name


def _configured(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return not any(
        marker in lowered for marker in ("your_", "placeholder", "changeme", "xxx", "<")
    )


def get_model_context_window(model_name: str | None = None) -> int:
    """Return the approximate context window for a configured model name."""
    settings = get_settings()
    name = _strip_prefix(model_name or settings.GROQ_MODEL)
    return _MODEL_CONTEXT_WINDOWS.get(name, _DEFAULT_CONTEXT_WINDOW)


def is_groq_configured() -> bool:
    """True when a non-placeholder GROQ_API_KEY is available."""
    return get_settings().groq_api_key_configured


def get_groq_model_name(kind: str = "default") -> str:
    """Return a configured Groq model name.

    kind: 'default' (general), 'reasoning' (harder analysis), 'vision' (multimodal),
          'fast' (lowest latency — for tool-heavy deterministic steps).
    """
    settings = get_settings()
    return {
        "default": settings.GROQ_MODEL,
        "reasoning": settings.GROQ_REASONING_MODEL,
        "vision": settings.GROQ_VISION_MODEL,
        "fast": settings.GROQ_FAST_MODEL,
    }.get(kind, settings.GROQ_MODEL)


def _model_name(settings, provider: str, kind: str) -> str:
    if provider == "groq":
        return {
            "default": settings.GROQ_MODEL,
            "reasoning": settings.GROQ_REASONING_MODEL,
            "vision": settings.GROQ_VISION_MODEL,
            "fast": settings.GROQ_FAST_MODEL,
        }.get(kind, settings.GROQ_MODEL)
    if provider == "openrouter":
        name = settings.OPENROUTER_FAST_MODEL if kind == "fast" else settings.OPENROUTER_MODEL
        return name if name.startswith("openrouter/") else f"openrouter/{name}"
    return {
        "vision": settings.GEMINI_VISION_MODEL,
        "fast": settings.GEMINI_FAST_MODEL,
    }.get(kind, settings.GEMINI_MODEL)


def build_groq_model(kind: str = "default"):
    """Build the configured multi-provider fallback model.

    The historical function name is retained so existing agent imports remain
    stable. Each provider is included only when its key is configured, except
    Groq, which remains available for offline construction and diagnostics.
    """
    from google.adk.models import FallbackModel, Gemini
    from google.adk.models.lite_llm import LiteLlm

    settings = get_settings()
    models = []
    providers = settings.provider_order or ["groq"]
    provider_timeout = min(settings.REQUEST_TIMEOUT, 20.0) if len(providers) > 1 else settings.REQUEST_TIMEOUT
    provider_retries = 0 if len(providers) > 1 else settings.MAX_RETRIES
    for provider in providers:
        model_name = _model_name(settings, provider, kind)
        if provider == "groq":
            if settings.groq_api_key_configured:
                kwargs = {"num_retries": provider_retries, "timeout": provider_timeout}
                if settings.GROQ_API_KEY:
                    kwargs["api_key"] = settings.GROQ_API_KEY
                models.append(LiteLlm(model=model_name, **kwargs))
        elif provider == "openrouter" and settings.openrouter_api_key_configured:
            openrouter_names = [model_name]
            if settings.OPENROUTER_FALLBACK_MODEL:
                fallback_name = settings.OPENROUTER_FALLBACK_MODEL
                if not fallback_name.startswith("openrouter/"):
                    fallback_name = f"openrouter/{fallback_name}"
                if fallback_name not in openrouter_names:
                    openrouter_names.append(fallback_name)
            models.extend(
                LiteLlm(
                    model=name,
                    api_key=settings.OPENROUTER_API_KEY,
                    num_retries=provider_retries,
                    timeout=provider_timeout,
                )
                for name in openrouter_names
            )
        elif provider == "gemini" and settings.google_api_key_configured:
            models.append(Gemini(model=model_name, client_kwargs={"api_key": settings.GOOGLE_API_KEY}))

    if not models:
        logger.warning("No configured provider keys; constructing Groq model for offline setup.")
        models.append(LiteLlm(model=_model_name(settings, "groq", kind),
                              num_retries=settings.MAX_RETRIES, timeout=settings.REQUEST_TIMEOUT))
    if len(models) == 1:
        return models[0]
    logger.info("Configured model fallback chain for %s: %s", kind,
                " -> ".join(model.model for model in models))
    return FallbackModel(
        models=models,
        retriable_status_codes=FallbackModel.DEFAULT_STATUS_CODES | {404},
    )
