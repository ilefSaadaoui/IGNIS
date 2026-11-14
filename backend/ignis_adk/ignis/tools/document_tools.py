"""Document intelligence tools: text loading, conditions and findings extraction.

Supported inputs: PDF (pypdf), DOCX (python-docx), TXT, CSV, JSON.
Files are treated as untrusted data: no instructions inside a document are
ever executed; content is only analysed as text.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from ignis.config.settings import get_settings

logger = logging.getLogger("ignis.tools.document")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}

# --- Loading ---------------------------------------------------------------


def load_document(path: str) -> dict[str, Any]:
    """Load a document from disk into structured text. Never raises."""
    settings = get_settings()
    p = Path(path).expanduser()
    if not p.exists():
        return {"status": "failed", "errors": [f"file not found: {p}"], "text": ""}
    if not p.is_file():
        return {"status": "failed", "errors": [f"not a regular file: {p}"], "text": ""}
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return {
            "status": "failed",
            "errors": [f"unsupported extension '{suffix}'; supported: {sorted(SUPPORTED_EXTENSIONS)}"],
            "text": "",
        }
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(p))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(f"[page {i+1}]\n{t}" for i, t in enumerate(pages))
            meta_pages = len(reader.pages)
        elif suffix == ".docx":
            import docx

            doc = docx.Document(str(p))
            text = "\n".join(par.text for par in doc.paragraphs)
            meta_pages = None
        elif suffix in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="replace")
            meta_pages = None
        elif suffix == ".csv":
            buf = io.StringIO()
            raw = p.read_text(encoding="utf-8", errors="replace")
            rows = list(csv.reader(io.StringIO(raw)))
            for row in rows[:500]:
                buf.write(" | ".join(row) + "\n")
            text = f"CSV with {len(rows)} rows; first 500 shown:\n" + buf.getvalue()
            meta_pages = None
        else:  # .json
            raw = p.read_text(encoding="utf-8", errors="replace")
            try:
                obj = json.loads(raw)
                text = json.dumps(obj, indent=1, ensure_ascii=False)[: settings.MAX_DOCUMENT_CHARS]
            except json.JSONDecodeError as exc:
                return {"status": "failed", "errors": [f"invalid JSON: {exc}"], "text": ""}
            meta_pages = None
    except Exception as exc:  # corrupted file, parser failure — reported, not hidden
        logger.exception("document load failed")
        return {"status": "failed", "errors": [f"{exc.__class__.__name__}: {exc}"], "text": ""}

    truncated = len(text) > settings.MAX_DOCUMENT_CHARS
    if truncated:
        text = text[: settings.MAX_DOCUMENT_CHARS] + "\n[... truncated to MAX_DOCUMENT_CHARS ...]"
    return {
        "status": "completed",
        "path": str(p),
        "extension": suffix,
        "pages": meta_pages,
        "truncated": truncated,
        "text": text,
        "char_count": len(text),
    }


# --- Text analysis ----------------------------------------------------------

_SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "methods": r"\b(methods?|materials and methods|experimental setup|apparatus)\b",
    "results": r"\bresults\b",
    "discussion": r"\bdiscussion\b",
    "conclusion": r"\bconclusions?\b",
    "references": r"\breferences\b",
    "acknowledg": r"\backnowledg",
}


def analyze_document_text(text: str) -> dict[str, Any]:
    """Structural analysis of scientific text (no LLM call required).

    Detects sections, references, table/figure mentions; reports counts.
    Returns structured dict; never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return {"status": "failed", "errors": ["text must be a non-empty string"]}
    lowered = text.lower()
    sections = [name for name, pat in _SECTION_PATTERNS.items() if re.search(pat, lowered)]
    ref_match = re.search(r"\breferences\b", lowered)
    references: list[str] = []
    if ref_match:
        ref_block = text[ref_match.end():]
        for line in ref_block.strip().splitlines():
            line = line.strip()
            if re.match(r"^\s*(\[\d+\]|\d+\.)", line):
                references.append(line[:300])
            if len(references) >= 25:
                break
    table_refs = sorted(set(re.findall(r"\btables?\s+\d+[A-Za-z]?", text, flags=re.IGNORECASE)))
    figure_refs = sorted(set(re.findall(r"\bfig(?:ure|\.)?\s*\.?\s*\d+[A-Za-z]?", text, flags=re.IGNORECASE)))
    return {
        "status": "completed",
        "char_count": len(text),
        "word_count": len(text.split()),
        "sections_detected": sections,
        "references": references,
        "reference_count_detected": len(references),
        "table_refs": table_refs[:25],
        "figure_refs": figure_refs[:25],
        "limitations": (
            ["Section detection is heuristic (keyword-based); absence of a section label does not prove the section is absent."]
        ),
    }


# --- Experimental conditions extraction --------------------------------------

_CONDITION_PATTERNS: dict[str, list[str]] = {
    "pressure": [r"\b\d+(?:\.\d+)?\s*(?:kPa|MPa|Pa|bar|atm|torr|mmHg|psia)\b"],
    "temperature": [r"\b\d+(?:\.\d+)?\s*(?:°\s?C|°\s?F|K|kelvin|degC)\b",
                    r"\b\d+\s*degrees?\s*(?:C|Celsius|F|Fahrenheit|K)\b"],
    "oxygen_concentration": [r"\b\d+(?:\.\d+)?\s*(?:%|percent|vol%|mol%)\s*(?:O2|O₂|oxygen)\b",
                             r"\b(?:O2|O₂|oxygen)\s*(?:concentration|level|fraction)?\s*(?:of\s*)?\d+(?:\.\d+)?\s*(?:%|percent|vol%)\b",
                             r"\b\d+(?:\.\d+)?\s*(?:vol%|mol%)\b"],
    "gravity_environment": [r"\bmicrogravity\b", r"\bmicro-g\b", r"\b0[- ]?g\b", r"\b1[- ]?g\b",
                            r"\bnormal gravity\b", r"\bpartial gravity\b", r"\bdrop tower\b",
                            r"\bparabolic flight", r"\bInternational Space Station\b", r"\bISS\b",
                            r"\bfree ?fall"],
    "fuel_or_material": [r"\b(?:PMMA|polyethylene|polypropylene|cotton|paper|cellulose|methane|propane|hydrogen|ethylene|heptane|hexane|ethanol|methanol|composite|fabric|foam|nomex|lucite)\b"],
    "flow_velocity": [r"\b\d+(?:\.\d+)?\s*(?:cm/s|mm/s|m/s|cm s-1|mm s-1|m s-1)\b"],
    "atmosphere": [r"\b(?:nitrogen|helium|CO2|carbon dioxide|air|inert|diluent)\b"],
}


def extract_experimental_conditions(text: str) -> dict[str, Any]:
    """Extract experimental-condition mentions from scientific text.

    Rule-based, transparent and reproducible; returns matched spans with
    context so a reviewer can verify. No values are invented — absent
    categories are reported as empty.
    """
    if not isinstance(text, str) or not text.strip():
        return {"status": "failed", "errors": ["text must be a non-empty string"]}

    found: dict[str, list[str]] = {k: [] for k in _CONDITION_PATTERNS}
    raw: list[dict[str, str]] = []
    for category, patterns in _CONDITION_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                span = m.group(0)
                if span not in found[category]:
                    found[category].append(span)
                start, end = m.start(), m.end()
                ctx = text[max(0, start - 60): min(len(text), end + 60)].replace("\n", " ")
                raw.append({"category": category, "match": span, "context": ctx.strip()})
    raw = raw[:60]
    missing = [k for k, v in found.items() if not v]
    return {
        "status": "completed" if any(found.values()) else "partial",
        **found,
        "raw_matches": raw,
        "missing_categories": missing,
        "limitations": [
            "Extraction is pattern-based; it reports mentions, not validated physical values.",
            "Units are preserved as written; convert before quantitative comparison.",
        ],
    }


# --- Findings extraction ------------------------------------------------------

_FINDING_CUES = [
    "found that", "showed that", "demonstrated that", "we report", "results indicate",
    "results show", "observed that", "measurements show", "data show", "data indicate",
    "we observe", "it was found", "reveals that", "conclude that", "suggests that",
    "indicates that", "we measured", "we found", "are consistent with",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def extract_research_findings(text: str, max_findings: int = 15) -> dict[str, Any]:
    """Extract candidate finding statements using cue phrases.

    Every returned finding carries its source sentence and cue so a reviewer
    can verify it — nothing is summarised away or invented.
    """
    if not isinstance(text, str) or not text.strip():
        return {"status": "failed", "errors": ["text must be a non-empty string"]}
    try:
        max_findings = max(1, min(int(max_findings), 40))
    except (TypeError, ValueError):
        max_findings = 15

    findings: list[dict[str, str]] = []
    for sent in _SENTENCE_SPLIT.split(text):
        s = sent.strip()
        if len(s) < 40 or len(s) > 1200:
            continue
        low = s.lower()
        for cue in _FINDING_CUES:
            if cue in low:
                findings.append({"statement": s, "evidence_sentence": s, "cue": cue})
                break
        if len(findings) >= max_findings:
            break

    return {
        "status": "completed" if findings else "partial",
        "findings": findings,
        "finding_count": len(findings),
        "limitations": [
            "Cue-phrase extraction yields *candidate* findings; each must be verified against the source.",
            "Absence of cues does not imply absence of results.",
        ]
        if findings
        else ["No cue phrases matched; the document may use different phrasing or contain no results section."],
    }
