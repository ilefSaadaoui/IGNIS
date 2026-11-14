"""Scientific review / quality-control tools.

These are deterministic gates used by the Scientific Reviewer agent.
A claim is supported ONLY when overlapping evidence text is supplied;
otherwise it is 'insufficient_evidence' — never 'assumed true'.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import requests

from ignis.config.settings import get_settings

logger = logging.getLogger("ignis.tools.review")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are",
    "was", "were", "that", "this", "these", "those", "from", "as", "at", "be", "it", "its",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower()) if t not in _STOPWORDS}


def validate_citations(
    citations: list[dict[str, str]] | list[str],
    check_reachability: bool = True,
) -> dict[str, Any]:
    """Validate citations: presence of URL/source and optional HTTP reachability.

    Args:
        citations: Either dicts with keys url/title/source, or plain URL strings.
        check_reachability: If True, HEAD/GET each URL with a short timeout.
    """
    if not isinstance(citations, list) or not citations:
        return {"status": "failed", "errors": ["citations must be a non-empty list"], "checks": []}
    settings = get_settings()
    checks: list[dict[str, Any]] = []
    reachable = 0
    for raw in citations:
        if isinstance(raw, str):
            url, title, source = (raw if raw.startswith("http") else None), raw, None
        elif isinstance(raw, dict):
            url = raw.get("url")
            title = raw.get("title")
            source = raw.get("source")
        else:
            checks.append({"citation": str(raw)[:200], "url": None, "status": "invalid", "detail": "unsupported citation type"})
            continue
        label = (title or url or source or "unnamed citation")
        if not url:
            checks.append({"citation": label, "url": None, "status": "missing_url",
                           "detail": "No URL supplied; cannot verify reachability. Do not treat as verified."})
            continue
        if not check_reachability:
            checks.append({"citation": label, "url": url, "status": "unchecked",
                           "detail": "Reachability check disabled by caller."})
            continue
        try:
            resp = requests.head(url, timeout=min(settings.NASA_REQUEST_TIMEOUT, 10), allow_redirects=True,
                                 headers={"User-Agent": "IGNIS-CitationValidator/1.0"})
            status = "verified_reachable" if resp.status_code < 400 else "unreachable"
            detail = f"HTTP {resp.status_code}"
            if status == "verified_reachable":
                reachable += 1
        except requests.RequestException as exc:
            status, detail = "unreachable", f"{exc.__class__.__name__}: {exc}"
        checks.append({"citation": label, "url": url, "status": status, "detail": detail})

    total = len(checks)
    if any(c["status"] == "missing_url" for c in checks):
        verdict = "incomplete"
    elif reachable == total and total > 0:
        verdict = "approved"
    elif reachable == 0:
        verdict = "requires_review"
    else:
        verdict = "partial"
    return {
        "status": "completed",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reachable": reachable,
        "total": total,
        "verdict": verdict,
        "checks": checks,
        "limitations": ["Reachability (HTTP <400) proves the page exists, NOT that it supports the claim made about it."],
    }


def review_scientific_claims(
    claims: list[str],
    evidence: list[str],
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Map claims to supplied evidence; mark supported / unsupported / insufficient.

    Overlap heuristic: a claim is 'supported' only when an evidence passage
    shares enough content tokens (>= 45% of the claim's content tokens, or an
    exact investigation/material co-mention) — otherwise insufficient/unsupported.
    """
    if not isinstance(claims, list) or not claims:
        return {"status": "failed", "errors": ["claims must be a non-empty list of strings"]}
    evidence = [str(e) for e in (evidence or []) if str(e).strip()]
    evidence_tokens = [_tokens(e) for e in evidence]

    assessments: list[dict[str, Any]] = []
    unsupported = 0
    for claim in claims:
        claim_str = str(claim).strip()
        if not claim_str:
            assessments.append({"claim": "", "verdict": "unsupported", "matched_evidence": [],
                                "reason": "empty claim"})
            unsupported += 1
            continue
        ct = _tokens(claim_str)
        if not ct:
            assessments.append({"claim": claim_str, "verdict": "insufficient_evidence",
                                "matched_evidence": [], "reason": "claim has no content tokens"})
            continue
        best_overlap = 0.0
        matched: list[str] = []
        for ev_text, ev_tok in zip(evidence, evidence_tokens):
            if not ev_tok:
                continue
            overlap = len(ct & ev_tok) / len(ct)
            best_overlap = max(best_overlap, overlap)
            if overlap >= 0.45:
                matched.append(ev_text[:200])
        if matched:
            assessments.append({"claim": claim_str, "verdict": "supported", "matched_evidence": matched,
                                "reason": f"overlapping evidence ({best_overlap:.0%} token coverage)"})
        elif best_overlap >= 0.2:
            assessments.append({"claim": claim_str, "verdict": "insufficient_evidence",
                                "matched_evidence": [], "reason": f"weak overlap ({best_overlap:.0%}); cannot confirm"})
        else:
            assessments.append({"claim": claim_str, "verdict": "unsupported",
                                "matched_evidence": [], "reason": f"no overlapping evidence ({best_overlap:.0%})"})
            unsupported += 1

    verdict = "approved" if all(a["verdict"] == "supported" for a in assessments) else (
        "rejected" if unsupported else "requires_review")
    return {
        "status": "completed",
        "verdict": verdict,
        "assessments": assessments,
        "unsupported_count": unsupported,
        "limitations": [
            "Token overlap is a triage heuristic, not semantic proof; a qualified reviewer must confirm.",
            "'supported' means consistent with supplied evidence — not independently verified truth.",
        ],
    }


def identify_unsupported_findings(
    findings: list[str],
    evidence: list[str],
) -> dict[str, Any]:
    """List findings that lack overlapping evidence (fabricated/unsupported detection)."""
    review = review_scientific_claims(findings, evidence)
    if review.get("status") == "failed":
        return review
    unsupported = [a["claim"] for a in review["assessments"] if a["verdict"] != "supported"]
    return {
        "status": "completed",
        "unsupported_findings": unsupported,
        "unsupported_count": len(unsupported),
        "total_findings": len(review["assessments"]),
        "verdict": review["verdict"],
        "limitations": review.get("limitations", []),
    }
