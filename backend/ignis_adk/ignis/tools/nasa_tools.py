"""Real NASA research retrieval tools for IGNIS.

Sources:
- NASA NTRS (Technical Reports Server): public JSON search API, queried live.
  Docs: https://ntrs.nasa.gov/api/citations/search?q=...
- NASA PSI (Physical Sciences Informatics): psi.nasa.gov is a curated web
  portal; there is no public JSON search API (the host redirects to
  science.nasa.gov/biological-physical). We therefore report PSI queries as
  'unavailable via API' and return the canonical portal URL instead of
  fabricating records. This is a deliberate, documented limitation.

No record is ever invented. When a source is unreachable or rate limits us,
the tools say so explicitly.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests

from ignis.config.settings import get_settings
from ignis.schemas.research import KNOWN_INVESTIGATIONS, normalize_investigation_name

logger = logging.getLogger("ignis.tools.nasa")

NTRS_SEARCH_URL = "https://ntrs.nasa.gov/api/citations/search"
NTRS_CITATION_URL = "https://ntrs.nasa.gov/citations/{id}"
PSI_PORTAL_URL = "https://psi.nasa.gov/"
PSI_SCIENCE_URL = "https://science.nasa.gov/biological-physical/data/"
USER_AGENT = "IGNIS-SpaceCombustionIntelligence/1.0 (research tool; contact: RaoSpace Labs)"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ntrs_record(raw: dict[str, Any]) -> dict[str, Any]:
    rid = raw.get("id")
    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()
    if len(abstract) > 1200:
        abstract = abstract[:1200].rsplit(" ", 1)[0] + " …"
    authors: list[str] = []
    for aff in raw.get("authorAffiliations", []) or []:
        meta = (aff or {}).get("meta", {}) or {}
        author = (meta.get("author", {}) or {}).get("name")
        if author:
            authors.append(author)
    center = (raw.get("center") or {}).get("name") if isinstance(raw.get("center"), dict) else None
    downloads = raw.get("downloads") or []
    download = downloads[0] if downloads and isinstance(downloads[0], dict) else {}
    links = download.get("links") or {}
    pdf_path = links.get("pdf") or links.get("original")
    text_path = links.get("fulltext")
    base_url = "https://ntrs.nasa.gov"
    return {
        "title": title or "(untitled)",
        "identifier": str(rid) if rid is not None else None,
        "url": NTRS_CITATION_URL.format(id=rid) if rid is not None else None,
        "pdf_url": f"{base_url}{pdf_path}" if pdf_path else None,
        "fulltext_url": f"{base_url}{text_path}" if text_path else None,
        "source": "NASA NTRS",
        "record_type": raw.get("stiType"),
        "published": raw.get("distributionDate"),
        "center": center,
        "abstract": abstract or None,
        "keywords": [k for k in (raw.get("keywords") or []) if isinstance(k, str)],
        "distribution": raw.get("distribution"),
        "authors": authors,
    }


_NTRS_CACHE: dict[tuple[str, int], tuple[float, list[dict[str, Any]], int | None, str | None]] = {}


def _ntrs_search(query: str, limit: int) -> tuple[list[dict[str, Any]], int | None, str | None]:
    """Query NTRS (TTL-cached in-process to cut repeat latency)."""
    settings = get_settings()
    ttl = getattr(settings, "NASA_CACHE_TTL_SECONDS", 300)
    key = (query.strip().lower(), int(limit))
    cached = _NTRS_CACHE.get(key)
    now = time.monotonic()
    if cached and ttl > 0 and (now - cached[0]) < ttl:
        return cached[1], cached[2], cached[3]
    try:
        resp = requests.get(
            NTRS_SEARCH_URL,
            params={"q": query, "page.size": limit},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=settings.NASA_REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return [], None, f"NTRS request timed out after {settings.NASA_REQUEST_TIMEOUT}s"
    except requests.RequestException as exc:
        return [], None, f"NTRS request failed: {exc.__class__.__name__}: {exc}"

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "unknown")
        return [], None, f"NTRS rate limit (HTTP 429); Retry-After={retry_after}. Try again later."
    if resp.status_code >= 500:
        return [], None, f"NTRS server error (HTTP {resp.status_code})"
    if resp.status_code != 200:
        return [], None, f"NTRS unexpected HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        payload = resp.json()
    except ValueError:
        return [], None, "NTRS returned non-JSON response"

    stats = payload.get("stats", {}) or {}
    total = stats.get("total")
    records = [_parse_ntrs_record(r) for r in (payload.get("results") or []) if isinstance(r, dict)]
    _NTRS_CACHE[key] = (now, records, total, None)
    return records, total, None


def search_nasa_research(query: str, source: str = "all", limit: int = 5) -> dict[str, Any]:
    """Search NASA public research sources for combustion-related records.

    Args:
        query: Free-text search, e.g. "microgravity combustion flame".
        source: 'ntrs', 'psi' or 'all' (default 'all'). 'psi' is reported as
            unavailable via API because psi.nasa.gov has no public JSON API.
        limit: Max records to return (1-10).

    Returns:
        Structured dict with records, provenance, status and limitations.
    """
    if not isinstance(query, str) or not query.strip():
        return {"status": "failed", "errors": ["query must be a non-empty string"], "records": []}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(10, limit))
    src = (source or "all").strip().lower()
    if src not in {"ntrs", "psi", "all"}:
        return {"status": "failed", "errors": [f"unknown source '{source}'; use ntrs|psi|all"], "records": []}

    result: dict[str, Any] = {
        "status": "completed",
        "query": query.strip(),
        "source": src,
        "retrieved_at": _now_iso(),
        "records": [],
        "total_matched": None,
        "limitations": [],
        "errors": [],
    }

    if src in {"all", "psi"}:
        result["limitations"].append(
            "PSI (psi.nasa.gov) provides no public JSON search API; it is a curated portal "
            f"(redirects to {PSI_SCIENCE_URL}). No PSI records are fabricated — use the portal URL "
            "for interactive dataset browsing. NTRS was queried instead for machine-readable metadata."
        )
        result["psi_portal_url"] = PSI_PORTAL_URL
        if src == "psi":
            result["status"] = "unavailable"
            result["retrieve_status"] = "psi_portal_only"
            return result

    records, total, error = _ntrs_search(query.strip(), limit)
    if error:
        result["status"] = "unavailable" if "rate limit" in error or "timed out" in error else "partial"
        result["errors"].append(error)
        result["records"] = []
        result["total_matched"] = None
        result["retrieve_status"] = "ntrs_error"
        return result

    # Polite inter-call spacing is handled by callers; NTRS is a single GET.
    result["records"] = records
    result["total_matched"] = total
    result["retrieve_status"] = "ntrs_ok"
    if not records:
        result["status"] = "completed"
        result["limitations"].append("NTRS returned zero records for this query.")
    public_only = [r for r in records if (r.get("distribution") or "PUBLIC") == "PUBLIC"]
    if len(public_only) != len(records):
        result["limitations"].append(
            "Some records are not marked PUBLIC by NTRS; review distribution rights before redistribution."
        )
    return result


def retrieve_investigation_metadata(investigation: str, limit: int = 5) -> dict[str, Any]:
    """Retrieve published metadata for a known microgravity combustion investigation.

    Supports (verified public expansions): BASS, BASS-II, FLEX, ACME,
    SAFFIRE, SAME, SAME-R. Unknown names are searched generically against NTRS;
    expansion strings are only returned for known campaigns.

    Args:
        investigation: Investigation name, e.g. 'FLEX' or 'SAME-R'.
        limit: Max NTRS records to return (1-10).
    """
    if not isinstance(investigation, str) or not investigation.strip():
        return {"status": "failed", "errors": ["investigation must be a non-empty string"], "records": []}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(10, limit))

    key = normalize_investigation_name(investigation)
    known = KNOWN_INVESTIGATIONS.get(key)
    search_terms = known["terms"] if known else f"{investigation.strip()} microgravity combustion"

    ntrs = search_nasa_research(search_terms, source="ntrs", limit=limit)

    meta: dict[str, Any] = {
        "status": ntrs.get("status", "failed"),
        "investigation": key,
        "expansion": known["expansion"] if known else None,
        "expansion_verified": bool(known and known.get("verified")),
        "background": [],
        "retrieved_at": _now_iso(),
        "records": ntrs.get("records", []),
        "total_matched": ntrs.get("total_matched"),
        "limitations": list(ntrs.get("limitations", [])),
        "errors": list(ntrs.get("errors", [])),
        "sources": [
            {"source": "NASA NTRS", "url": "https://ntrs.nasa.gov/", "identifier": None},
            {"source": "NASA PSI", "url": PSI_PORTAL_URL, "identifier": None},
        ],
    }
    if known:
        meta["background"] = [
            f"{key} refers to the NASA investigation '{known['expansion']}'.",
            "Metadata below comes live from NTRS; PSI holds the curated data products but has no public JSON API.",
        ]
    else:
        meta["limitations"].append(
            f"'{investigation}' is not in the verified investigation registry; NTRS was searched generically. "
            "No expansion name or background is claimed for unregistered campaigns."
        )
    return meta


NASA_IMAGES_SEARCH_URL = "https://images-api.nasa.gov/search"
_NASA_IMAGES_CACHE: dict[tuple[str, str, int], tuple[float, list[dict[str, Any]], int | None]] = {}


def search_nasa_images(query: str, media_type: str = "image", page_size: int = 5) -> dict[str, Any]:
    """Search the NASA Image and Video Library for official NASA media (no key required).

    Use this to find real imagery of microgravity combustion experiments — BASS-II/FLEX/
    ACME/SAFFIRE rigs, flame configurations, ISS smoke-detector tests — for briefings,
    reports and challenge demos.

    Args:
        query: Free text, e.g. "BASS-II combustion experiment" or "FLEX droplet flame".
        media_type: 'image' (default), 'video' or 'audio'.
        page_size: Max results (1-10).

    Returns:
        Structured dict with status, items (title, date_created, center, description,
        keywords, preview_url, collection_url, nasa_id), total_hits, provenance and
        limitations. Preview URLs come verbatim from the API — never constructed.
        Nothing is invented; when the service fails, it says so.
    """
    if not isinstance(query, str) or not query.strip():
        return {"status": "failed", "errors": ["query must be a non-empty string"], "items": []}
    media = (media_type or "image").strip().lower()
    if media not in {"image", "video", "audio"}:
        return {"status": "failed", "errors": [f"unknown media_type '{media_type}'; use image|video|audio"], "items": []}
    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = 5
    page_size = max(1, min(10, page_size))

    settings = get_settings()
    ttl = getattr(settings, "NASA_CACHE_TTL_SECONDS", 300)
    key = (query.strip().lower(), media, page_size)
    now = time.monotonic()
    cached = _NASA_IMAGES_CACHE.get(key)
    if cached and ttl > 0 and (now - cached[0]) < ttl:
        items, total = cached[1], cached[2]
        cached = True
    else:
        try:
            resp = requests.get(
                NASA_IMAGES_SEARCH_URL,
                params={"q": query.strip(), "media_type": media, "page_size": page_size},
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=settings.NASA_REQUEST_TIMEOUT,
            )
        except requests.Timeout:
            return {"status": "failed", "errors": [f"NASA Images request timed out after {settings.NASA_REQUEST_TIMEOUT}s"], "items": []}
        except requests.RequestException as exc:
            return {"status": "failed", "errors": [f"NASA Images request failed: {exc.__class__.__name__}: {exc}"], "items": []}
        if resp.status_code == 429:
            return {"status": "failed", "errors": ["NASA Images rate limit (HTTP 429); try again later."], "items": []}
        if resp.status_code != 200:
            return {"status": "failed", "errors": [f"NASA Images unexpected HTTP {resp.status_code}: {resp.text[:200]}"], "items": []}
        try:
            payload = resp.json()
        except ValueError:
            return {"status": "failed", "errors": ["NASA Images returned non-JSON response"], "items": []}

        collection = payload.get("collection") or {}
        total = (collection.get("metadata") or {}).get("total_hits")
        items = []
        for raw in collection.get("items") or []:
            if not isinstance(raw, dict):
                continue
            data = (raw.get("data") or [{}])[0]
            links = raw.get("links") or []
            preview = next((l.get("href") for l in links if isinstance(l, dict) and l.get("rel") == "preview"), None)
            description = (data.get("description") or "").strip()
            if len(description) > 600:
                description = description[:600].rsplit(" ", 1)[0] + " …"
            items.append({
                "nasa_id": data.get("nasa_id"),
                "title": (data.get("title") or "(untitled)").strip(),
                "description": description or None,
                "date_created": data.get("date_created"),
                "center": data.get("center"),
                "keywords": [k for k in (data.get("keywords") or []) if isinstance(k, str)][:12],
                "preview_url": preview,
                "collection_url": raw.get("href"),
            })
        _NASA_IMAGES_CACHE[key] = (now, items, total)
        cached = False

    return {
        "status": "completed" if items else "partial",
        "query": query.strip(),
        "media_type": media,
        "total_hits": total,
        "items": items,
        "count": len(items),
        "cached": bool(cached),
        "retrieved_at": _now_iso(),
        "provenance": "https://images.nasa.gov/ — NASA Image and Video Library (free, no API key). Media courtesy NASA; public domain unless otherwise noted.",
        "limitations": [
            "Titles/descriptions are NASA's own metadata, not our verification — open preview_url before using in print.",
            "Preview URLs point to JPEG renditions; higher-resolution assets are via collection_url (JSON).",
            "Relevance ranking is NASA's, not ours; refine the query or use media_type filters.",
        ],
    }
