"""Download and extract evidence from research URLs."""

from __future__ import annotations

import io
from typing import Any
from urllib.parse import urlparse

import requests

from ignis.config.settings import get_settings
from ignis.tools.document_tools import extract_experimental_conditions, extract_research_findings

MAX_SOURCE_BYTES = 50 * 1024 * 1024


def extract_research_source(
    url: str,
    source: str = "research URL",
    max_findings: int = 8,
) -> dict[str, Any]:
    """Fetch a public PDF/text research source and extract citable evidence.

    The returned text is bounded and every result preserves the original URL
    and source label. HTML landing pages are reported as unsupported rather
    than silently treated as paper text.
    """
    if not isinstance(url, str) or not url.strip():
        return {"status": "failed", "errors": ["url must be a non-empty string"]}
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"status": "failed", "url": url, "errors": ["url must be an absolute HTTP(S) URL"]}

    settings = get_settings()
    try:
        response = requests.get(
            url.strip(),
            headers={"User-Agent": "IGNIS-SpaceCombustionIntelligence/1.0", "Accept": "*/*"},
            timeout=settings.NASA_REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {"status": "unavailable", "url": url, "source": source,
                "errors": [f"source request timed out after {settings.NASA_REQUEST_TIMEOUT}s"]}
    except requests.RequestException as exc:
        return {"status": "unavailable", "url": url, "source": source,
                "errors": [f"source request failed: {exc.__class__.__name__}: {exc}"]}

    if response.status_code != 200:
        return {"status": "unavailable", "url": url, "source": source,
                "errors": [f"source returned HTTP {response.status_code}"]}
    if len(response.content) > MAX_SOURCE_BYTES:
        return {"status": "failed", "url": url, "source": source,
                "errors": [f"source exceeds {MAX_SOURCE_BYTES} byte safety limit"]}

    content_type = response.headers.get("Content-Type", "").lower()
    is_pdf = urlparse(url).path.lower().endswith(".pdf") or "application/pdf" in content_type
    if is_pdf:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(response.content))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(f"[page {i + 1}]\n{page}" for i, page in enumerate(pages))
            page_count = len(reader.pages)
        except Exception as exc:
            return {"status": "failed", "url": url, "source": source,
                    "errors": [f"PDF extraction failed: {exc.__class__.__name__}: {exc}"]}
    elif "text/plain" in content_type or urlparse(url).path.lower().endswith((".txt", ".text")):
        text = response.content.decode(response.encoding or "utf-8", errors="replace")
        page_count = None
    else:
        return {"status": "failed", "url": url, "source": source,
                "errors": ["source is not a PDF or plain-text full-text URL"]}

    truncated = len(text) > settings.MAX_DOCUMENT_CHARS
    if truncated:
        text = text[: settings.MAX_DOCUMENT_CHARS] + "\n[... source text truncated ...]"
    findings = extract_research_findings(text, max_findings=max_findings)
    conditions = extract_experimental_conditions(text)
    return {
        "status": "completed",
        "source": source,
        "url": url,
        "format": "pdf" if is_pdf else "text",
        "pages": page_count,
        "truncated": truncated,
        "char_count": len(text),
        "text": text,
        "findings": findings.get("findings", []),
        "conditions": conditions,
        "limitations": [
            "Extracted text is evidence from the linked source; verify page layout, tables, and figures in the original.",
            "Finding candidates are cue-based excerpts, not independent scientific validation.",
        ],
        "errors": [],
    }
