"""Central configuration for IGNIS.

All values are read from environment variables (optionally via a local .env
file). No secrets are hardcoded. Missing or placeholder credentials are
reported explicitly and never printed.

Token budgets are configurable through the environment; see .env.example.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(Path.cwd() / ".env")  # allow running from the project root too

logger = logging.getLogger("ignis.config")

_PLACEHOLDER_MARKERS = ("your", "placeholder", "changeme", "xxx", "<", "insert")


def _looks_like_placeholder(value: str) -> bool:
    v = value.strip().lower()
    if not v:
        return True
    return any(marker in v for marker in _PLACEHOLDER_MARKERS)


class Settings(BaseSettings):
    """IGNIS runtime settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Credentials -----------------------------------------------------
    GROQ_API_KEY: str | None = Field(default=None, description="Groq API key (never logged).")
    OPENROUTER_API_KEY: str | None = Field(default=None, description="OpenRouter API key (never logged).")
    GOOGLE_API_KEY: str | None = Field(default=None, description="Google AI API key (never logged).")
    NASA_API_KEY: str | None = Field(default=None, description="NASA API key (never logged).")
    CORE_API_KEY: str | None = Field(default=None, description="CORE API key (never logged).")

    # --- Model selection (LiteLLM naming, e.g. 'groq/openai/gpt-oss-120b') ---
    GROQ_MODEL: str = Field(default="groq/openai/gpt-oss-120b")
    GROQ_REASONING_MODEL: str = Field(default="groq/openai/gpt-oss-120b")
    GROQ_FAST_MODEL: str = Field(default="groq/openai/gpt-oss-20b")
    GROQ_VISION_MODEL: str = Field(default="groq/openai/gpt-oss-20b")
    OPENROUTER_MODEL: str = Field(default="baidu/ernie-5.5-flash-preview:free")
    OPENROUTER_FAST_MODEL: str = Field(default="baidu/ernie-5.5-flash-preview:free")
    OPENROUTER_FALLBACK_MODEL: str = Field(default="nvidia/nemotron-3.5-super-vl:free")
    GEMINI_MODEL: str = Field(default="gemini-3.1-flash-lite")
    GEMINI_FAST_MODEL: str = Field(default="gemini-3.1-flash-lite")
    GEMINI_VISION_MODEL: str = Field(default="gemini-3.1-flash-lite")
    IGNIS_PROVIDER_ORDER: str = Field(default="groq,openrouter")

    # --- Token / context budgets ------------------------------------------
    MAX_INPUT_TOKENS: int = Field(default=6000)
    MAX_OUTPUT_TOKENS: int = Field(default=4096)
    MAX_CONTEXT_TOKENS: int = Field(default=128000)
    MAX_AGENT_ITERATIONS: int = Field(default=5)
    MAX_CONVERSATION_HISTORY: int = Field(default=12)
    MAX_DOCUMENT_CHARS: int = Field(default=20000)
    MAX_AGENT_OUTPUT_CHARS: int = Field(default=8000)

    # --- Execution controls ------------------------------------------------
    REQUEST_TIMEOUT: float = Field(default=60.0)
    MAX_RETRIES: int = Field(default=2)
    LOG_LEVEL: str = Field(default="INFO")

    # --- Network -----------------------------------------------------------
    NASA_REQUEST_TIMEOUT: float = Field(default=20.0)
    NASA_MAX_RESULTS: int = Field(default=5)
    NASA_CACHE_TTL_SECONDS: int = Field(default=300)
    CORE_REQUEST_TIMEOUT: float = Field(default=20.0)
    CORE_MAX_RESULTS: int = Field(default=5)

    @field_validator("GROQ_MODEL", "GROQ_REASONING_MODEL", "GROQ_VISION_MODEL", "GROQ_FAST_MODEL")
    @classmethod
    def _normalize_groq_model(cls, v: str) -> str:
        v = v.strip()
        if v and not v.startswith("groq/"):
            v = f"groq/{v}"
        return v

    @field_validator(
        "OPENROUTER_MODEL", "OPENROUTER_FAST_MODEL", "GEMINI_MODEL",
        "OPENROUTER_FALLBACK_MODEL", "GEMINI_FAST_MODEL", "GEMINI_VISION_MODEL",
    )
    @classmethod
    def _normalize_external_model(cls, v: str) -> str:
        return v.strip()

    @field_validator("LOG_LEVEL")
    @classmethod
    def _upper_log(cls, v: str) -> str:
        return v.strip().upper() or "INFO"

    @property
    def groq_api_key_configured(self) -> bool:
        """True only when a non-placeholder key is present. Never exposes the key."""
        if not self.GROQ_API_KEY:
            return False
        return not _looks_like_placeholder(self.GROQ_API_KEY)

    @staticmethod
    def _key_configured(value: str | None) -> bool:
        return bool(value and not _looks_like_placeholder(value))

    @property
    def openrouter_api_key_configured(self) -> bool:
        return self._key_configured(self.OPENROUTER_API_KEY)

    @property
    def google_api_key_configured(self) -> bool:
        return self._key_configured(self.GOOGLE_API_KEY)

    @property
    def core_api_key_configured(self) -> bool:
        return self._key_configured(self.CORE_API_KEY)

    @property
    def provider_order(self) -> list[str]:
        supported = {"groq", "openrouter", "gemini"}
        return [
            provider.strip() for provider in self.IGNIS_PROVIDER_ORDER.lower().split(",")
            if provider.strip() in supported
        ]

    def missing_credentials(self) -> list[str]:
        """Report which credentials are missing/placeholder (names only, never values)."""
        issues: list[str] = []
        if not self.groq_api_key_configured:
            issues.append(
                "GROQ_API_KEY is missing or still set to a placeholder. "
                "Set a real key in .env or the environment before live inference."
            )
        if "openrouter" in self.provider_order and not self.openrouter_api_key_configured:
            issues.append("OPENROUTER_API_KEY is missing or still a placeholder.")
        if "gemini" in self.provider_order and not self.google_api_key_configured:
            issues.append("GOOGLE_API_KEY is missing or still a placeholder.")
        return issues

    def configure_logging(self) -> None:
        level = getattr(logging, self.LOG_LEVEL, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
