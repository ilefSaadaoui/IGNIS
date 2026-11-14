"""Knowledge-graph tools: rule-based entity/relationship extraction.

No Neo4j is required. Output is graph-ready JSON (nodes/edges with provenance).
Extraction is deterministic and conservative — low-confidence, review-required.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_INVESTIGATIONS = [
    "BASS-II", "BASS", "FLEX", "ACME", "SAFFIRE-IV", "SAFFIRE-III", "SAFFIRE-II", "SAFFIRE-I",
    "SAFFIRE", "SAME-R", "SAME", "SLICE", "SPICE", "DAFT-2", "DAFT", "CIR", "ZIRKKA",
]
_MATERIALS = [
    "PMMA", "polyethylene", "polypropylene", "cotton", "paper", "cellulose", "methane", "propane",
    "hydrogen", "ethylene", "methanol", "ethanol", "heptane", "hexane", "Nomex", "lucite", "foam",
    "fabric", "composite", "aluminum", "aluminium", "steel", "titanium", "magnesium", "lithium",
]
_FACILITIES = [
    "International Space Station", "ISS", "Space Station", "Microgravity Science Glovebox", "MSG",
    "Zero Gravity Research Facility", "drop tower", "parabolic flight", "KC-135", "Glenn Research Center",
]
_INSTRUMENTS = [
    "smoke detector", "photoelectric detector", "ionization detector", "radiometer", "thermocouple",
    "camera", "spectrometer", "gas chromatograph", "particle sizer", "scattering", "anemometer",
]
_PHENOMENA = [
    "flammability", "ignition", "extinction", "burning rate", "flame spread", "soot", "smoke",
    "smoldering", "suppression", "flame extinction", "limiting oxygen concentration", "LOI",
    "droplet combustion", "pool fire", "ember", "heat flux", "mass loss rate",
]
_CONDITIONS = [r"\b\d+(?:\.\d+)?\s*(?:kPa|MPa|Pa|atm|torr)\b", r"\b\d+(?:\.\d+)?\s*%\s*(?:O2|O₂|oxygen)\b", r"\bmicrogravity\b", r"\b\d+[- ]?g\b"]

_RELATION_PATTERNS = [
    (r"(?P<a>[A-Za-z0-9\- ]{2,40})\s+(?:was|were)\s+(?:conducted|performed|flown|tested)\s+(?:in|on|aboard)\s+(?P<b>[A-Za-z0-9\- ]{2,40})", "INVESTIGATED_IN"),
    (r"(?P<a>[A-Za-z0-9\- ]{2,40})\s+(?:used|burned|burning)\s+(?P<b>[A-Za-z0-9\- ]{2,40})", "USED"),
    (r"(?P<a>[A-Za-z0-9\- ]{2,40})\s+measured\s+(?P<b>[A-Za-z0-9\- ]{2,40})", "MEASURED"),
    (r"(?P<a>[A-Za-z0-9\- ]{2,40})\s+(?:extends|extension|follow[- ]?on|reflight of|re-flight of)\s+(?:the\s+)?(?P<b>[A-Za-z0-9\- ]{2,40})", "PART_OF"),
    (r"(?P<a>[A-Za-z0-9\- ]{2,40})\s+(?:investigated|studied|examined)\s+(?P<b>[A-Za-z0-9\- ]{2,40})", "INVESTIGATED"),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _eid(kind: str, name: str) -> str:
    return f"{kind.lower()}-{hashlib.sha1(name.lower().encode()).hexdigest()[:10]}"


def extract_entities(text: str) -> dict[str, Any]:
    """Extract known combustion-research entities from text with provenance."""
    if not isinstance(text, str) or not text.strip():
        return {"status": "failed", "errors": ["text must be a non-empty string"], "entities": []}

    entities: dict[str, dict[str, Any]] = {}

    def add(name: str, etype: str, evidence: str) -> None:
        key = f"{etype}:{name.lower()}"
        if key in entities:
            entities[key]["occurrences"] += 1
            return
        entities[key] = {
            "id": _eid(etype, name), "name": name, "type": etype,
            "evidence": evidence[:160], "occurrences": 1,
        }

    sentences = _SENTENCE_SPLIT.split(text)
    for sent in sentences:
        low = sent.lower()
        taken_spans: list[tuple[int, int]] = []
        for term in sorted(_INVESTIGATIONS, key=len, reverse=True):
            for m in re.finditer(rf"\b{re.escape(term)}\b", sent, flags=re.IGNORECASE):
                span = m.span()
                if any(not (span[1] <= s or span[0] >= e) for s, e in taken_spans):
                    continue  # a longer investigation name already covers this span
                taken_spans.append(span)
                add(term, "INVESTIGATION", sent.strip())
        for term in _MATERIALS:
            if re.search(rf"\b{re.escape(term)}\b", sent, flags=re.IGNORECASE):
                add(term, "MATERIAL", sent.strip())
        for term in _FACILITIES:
            if term.lower() in low:
                add(term, "FACILITY", sent.strip())
        for term in _INSTRUMENTS:
            if term.lower() in low:
                add(term, "INSTRUMENT", sent.strip())
        for term in _PHENOMENA:
            if term.lower() in low:
                add(term, "PHENOMENON", sent.strip())
        for pat in _CONDITIONS:
            for m in re.finditer(pat, sent, flags=re.IGNORECASE):
                add(m.group(0), "CONDITION", sent.strip())

    result = list(entities.values())
    return {
        "status": "completed" if result else "partial",
        "entities": result,
        "entity_count": len(result),
        "limitations": [
            "Extraction is lexicon/pattern-based and English-only; novel or misspelled terms are missed.",
            "Types may need manual curation before use in a production knowledge base.",
        ],
    }


def extract_relationships(text: str, entities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Extract typed relationships between known entities within sentences."""
    if not isinstance(text, str) or not text.strip():
        return {"status": "failed", "errors": ["text must be a non-empty string"], "relationships": []}

    ent = entities if entities is not None else extract_entities(text).get("entities", [])
    by_name: dict[str, dict[str, Any]] = {e["name"].lower(): e for e in ent}

    relationships: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for sent in _SENTENCE_SPLIT.split(text):
        low = sent.lower()
        present = [e for name, e in by_name.items() if name in low]
        for pat, rtype in _RELATION_PATTERNS:
            for m in re.finditer(pat, sent, flags=re.IGNORECASE):
                a_raw, b_raw = m.group("a").strip(), m.group("b").strip()
                a = by_name.get(a_raw.lower()) or _closest_entity(a_raw, present)
                b = by_name.get(b_raw.lower()) or _closest_entity(b_raw, present)
                if not a or not b or a["id"] == b["id"]:
                    continue
                key = (a["id"], b["id"], rtype)
                if key in seen:
                    continue
                seen.add(key)
                relationships.append({
                    "id": _eid("REL", f"{a['id']}-{b['id']}-{rtype}"),
                    "source": a["id"], "target": b["id"], "type": rtype,
                    "evidence": sent.strip()[:200], "confidence": "low",
                })
        # Co-occurrence heuristic: investigation + material in one sentence.
        investigations = [e for e in present if e["type"] == "INVESTIGATION"]
        materials = [e for e in present if e["type"] == "MATERIAL"]
        for inv in investigations:
            for mat in materials:
                key = (inv["id"], mat["id"], "STUDIED_IN")
                if key in seen:
                    continue
                seen.add(key)
                relationships.append({
                    "id": _eid("REL", f"{inv['id']}-{mat['id']}-STUDIED_IN"),
                    "source": mat["id"], "target": inv["id"], "type": "STUDIED_IN",
                    "evidence": sent.strip()[:200], "confidence": "low",
                })
    return {
        "status": "completed" if relationships else "partial",
        "relationships": relationships,
        "relationship_count": len(relationships),
        "limitations": [
            "Relationships are pattern/co-occurrence based; confidence is 'low' and requires human/agent review.",
            "Direction and type may be wrong for complex sentences.",
        ],
    }


def _closest_entity(fragment: str, present: list[dict[str, Any]]) -> dict[str, Any] | None:
    frag = fragment.lower()
    for e in present:
        if e["name"].lower() in frag or frag in e["name"].lower():
            return e
    return None


def build_graph_representation(
    entities: list[dict[str, Any]], relationships: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build graph-ready JSON (deduplicated nodes/edges) from extraction output."""
    if not isinstance(entities, list):
        return {"status": "failed", "errors": ["entities must be a list"]}
    if not isinstance(relationships, list):
        return {"status": "failed", "errors": ["relationships must be a list"]}

    nodes: dict[str, dict[str, Any]] = {}
    for e in entities:
        if not isinstance(e, dict) or not e.get("id") or not e.get("name"):
            continue
        nodes[e["id"]] = {
            "id": e["id"], "name": e["name"], "type": e.get("type", "UNKNOWN"),
            "occurrences": int(e.get("occurrences", 1)),
            "properties": {"evidence": e.get("evidence", "")},
        }
    edges: dict[str, dict[str, Any]] = {}
    dropped = 0
    for r in relationships:
        if not isinstance(r, dict):
            dropped += 1
            continue
        s, t = r.get("source"), r.get("target")
        if s not in nodes or t not in nodes:
            dropped += 1
            continue
        edges[r.get("id") or _eid("REL", f"{s}-{t}-{r.get('type')}")] = {
            "id": r.get("id"), "source": s, "target": t, "type": r.get("type", "RELATED_TO"),
            "properties": {"evidence": r.get("evidence", ""), "confidence": r.get("confidence", "low")},
        }
    types: dict[str, int] = {}
    for n in nodes.values():
        types[n["type"]] = types.get(n["type"], 0) + 1
    return {
        "status": "completed" if nodes else "partial",
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": types,
        "dropped_edges": dropped,
        "provenance": "derived from supplied text; rule-based extraction; no external database used",
        "limitations": ["Rule-based graph; validate entities/relationships before downstream reasoning."],
    }
