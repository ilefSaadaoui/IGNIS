"""Open-access research retrieval through the CORE API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from ignis.config.settings import get_settings

CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"
USER_AGENT = "IGNIS-SpaceCombustionIntelligence/1.0 (CORE research tool)"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_work(raw: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for author in raw.get("authors") or []:
        if isinstance(author, dict) and author.get("name"):
            authors.append(author["name"])
        elif isinstance(author, str):
            authors.append(author)
    urls = raw.get("urls") or []
    url = urls[0] if urls and isinstance(urls[0], str) else None
    return {
        "title": (raw.get("title") or "").strip() or "(untitled)",
        "identifier": str(raw.get("id")) if raw.get("id") is not None else None,
        "url": url or raw.get("downloadUrl") or raw.get("sourceFulltextUrls"),
        "pdf_url": raw.get("downloadUrl"),
        "source": "CORE",
        "published": raw.get("publishedDate") or raw.get("yearPublished"),
        "authors": authors,
        "abstract": (raw.get("abstract") or "").strip() or None,
        "open_access": raw.get("isOpenAccess"),
    }


def search_core_research(query: str, limit: int = 5) -> dict[str, Any]:
    """Search CORE open-access works and return citable metadata and PDF links."""
    if not isinstance(query, str) or not query.strip():
        return {"status": "failed", "errors": ["query must be a non-empty string"], "records": []}
    settings = get_settings()
    if not settings.core_api_key_configured:
        return {
            "status": "unavailable",
            "query": query.strip(),
            "records": [],
            "errors": ["CORE_API_KEY is missing or still a placeholder."],
        }
    try:
        requested = max(1, min(int(limit), settings.CORE_MAX_RESULTS, 20))
    except (TypeError, ValueError):
        requested = settings.CORE_MAX_RESULTS
    try:
        response = requests.get(
            CORE_SEARCH_URL,
            params={"q": query.strip(), "limit": requested},
            headers={"Authorization": f"Bearer {settings.CORE_API_KEY}", "User-Agent": USER_AGENT},
            timeout=settings.CORE_REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {"status": "unavailable", "query": query.strip(), "records": [],
                "errors": [f"CORE request timed out after {settings.CORE_REQUEST_TIMEOUT}s"]}
    except requests.RequestException as exc:
        return {"status": "unavailable", "query": query.strip(), "records": [],
                "errors": [f"CORE request failed: {exc.__class__.__name__}: {exc}"]}
    if response.status_code == 429:
        return {"status": "unavailable", "query": query.strip(), "records": [],
                "errors": ["CORE rate limit reached (HTTP 429). Try again later."]}
    if response.status_code in {401, 403}:
        return {"status": "unavailable", "query": query.strip(), "records": [],
                "errors": [f"CORE authentication failed (HTTP {response.status_code})."]}
    if response.status_code >= 500:
        return {"status": "unavailable", "query": query.strip(), "records": [],
                "errors": [f"CORE server error (HTTP {response.status_code})."]}
    if response.status_code != 200:
        return {"status": "partial", "query": query.strip(), "records": [],
                "errors": [f"CORE unexpected HTTP {response.status_code}: {response.text[:200]}"]}
    try:
        payload = response.json()
    except ValueError:
        return {"status": "unavailable", "query": query.strip(), "records": [],
                "errors": ["CORE returned a non-JSON response."]}
    records = [_parse_work(item) for item in (payload.get("results") or []) if isinstance(item, dict)]
    return {
        "status": "completed" if records else "partial",
        "query": query.strip(),
        "retrieved_at": _now_iso(),
        "records": records,
        "total_matched": payload.get("totalHits") or payload.get("total"),
        "source": "CORE",
        "limitations": [
            "CORE provides open-access discovery and full-text links; verify the publisher or repository record before treating a result as authoritative.",
        ],
        "errors": [],
    }
