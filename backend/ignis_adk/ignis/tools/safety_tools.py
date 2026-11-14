"""Space fire-safety interpretation tools.

These tools produce *evidence-grounded interpretation* — never certified
operational recommendations for spacecraft. Every output carries explicit
limitations and distinguishes experimental observation from safety inference.
"""

from __future__ import annotations

import re
from typing import Any

from ignis.config.settings import get_settings

_MISSION_PROFILES = {
    "iss": {"label": "Low Earth Orbit (ISS)", "weight": 1.0,
            "notes": "Microgravity ventilation-driven flames; crewed; existing smoke-detection heritage (SAME/SAME-R)."},
    "lunar": {"label": "Lunar surface / Gateway", "weight": 0.8,
              "notes": "Partial gravity (1/6 g); fire behaviour differs from both 1 g and 0 g; limited egress and resupply."},
    "mars": {"label": "Mars surface", "weight": 0.8,
             "notes": "~3/8 g; CO2-rich atmosphere options vs habitat air; long-duration autonomy, no quick return."},
    "transit": {"label": "Deep-space transit", "weight": 1.0,
                "notes": "Microgravity for months; delayed support; fire must be detected and suppressed autonomously."},
}

_RELEVANCE_KEYWORDS = {
    "high": ["flammability", "ignition", "extinction", "smoke", "detection", "suppression",
             "material", "fire", "flame", "smolder", "ember"],
    "medium": ["soot", "heat flux", "mass loss", "ventilation", "oxygen", "extinguish"],
    "low": ["droplet", "combustion", "microgravity"],
}


def assess_mission_relevance(
    findings: list[str] | str,
    mission_context: str = "iss",
) -> dict[str, Any]:
    """Score the relevance of supplied findings to a mission context.

    Rubric-based (keyword coverage x profile weight) — an aid to triage,
    NOT a certification of safety.
    """
    if isinstance(findings, str):
        findings_list = [findings]
    elif isinstance(findings, list):
        findings_list = [str(f) for f in findings if str(f).strip()]
    else:
        return {"status": "failed", "errors": ["findings must be a string or list of strings"]}
    if not findings_list:
        return {"status": "failed", "errors": ["at least one finding is required"]}

    ctx = (mission_context or "iss").strip().lower()
    profile = _MISSION_PROFILES.get(ctx)
    if profile is None:
        profile = {"label": mission_context, "weight": 0.7,
                   "notes": "Unrecognised mission context; generic weight applied."}

    text = " ".join(findings_list).lower()
    hits: dict[str, list[str]] = {level: [k for k in kws if k in text] for level, kws in _RELEVANCE_KEYWORDS.items()}
    raw = 2 * len(hits["high"]) + 1 * len(hits["medium"]) + 0.5 * len(hits["low"])
    normalised = min(1.0, raw / 10.0) * profile["weight"]

    implications = []
    if hits["high"]:
        implications.append(
            "Findings touch high-impact safety areas: " + ", ".join(sorted(set(hits["high"]))) + "."
        )
    if "smoke" in text or "detection" in text:
        implications.append("Direct bearing on fire-detection requirements for the stated environment.")
    if "suppression" in text or "extinguish" in text:
        implications.append("Bears on suppression strategy; note microgravity suppression behaves differently from 1 g.")
    if not implications:
        implications.append("Findings are combustion-relevant but transfer to crew safety is indirect; review with domain experts.")

    return {
        "status": "completed",
        "mission_context": profile["label"],
        "relevance_score": round(normalised, 3),
        "relevance_rationale": [
            f"Keyword rubric: high={hits['high'] or 'none'}; medium={hits['medium'] or 'none'}; low={hits['low'] or 'none'}.",
            f"Mission profile weight: {profile['weight']}. {profile['notes']}",
            "Score is a triage signal, not a safety metric.",
        ],
        "implications": implications,
        "translation_limits": [
            "Microgravity experiments (droplet/exacted samples, specific atmospheres) may not scale to compartment fires.",
            "Material formulations tested in orbit may differ from as-built spacecraft materials.",
            "This is research interpretation, NOT a certified spacecraft safety recommendation; NASA verification standards apply for flight hardware.",
        ],
        "summary": f"Relevance to {profile['label']}: {normalised:.2f} (rubric 0-1). Review with fire-safety authority before operational use.",
        "limitations": ["Keyword rubric is heuristic; domain review required."],
    }


_SAFETY_DIMENSIONS = [
    ("ignition_sources", "Ignition sources characterised", ["ignition", "spark", "heat flux", "autoignition", "piloted"]),
    ("material_flammability", "Material flammability measured", ["flammab", "burning rate", "flame spread", "mass loss", "loi"]),
    ("detection", "Detection capability addressed", ["smoke", "detector", "detection", "alarm"]),
    ("suppression", "Suppression effectiveness addressed", ["suppress", "extinguish", "CO2", "helium", "water"]),
    ("atmosphere", "Atmosphere/oxygen dependence measured", ["oxygen", "o2", "atmosphere", "pressure", "kPa", "concentration"]),
    ("gravity_transfer", "Gravity dependence characterised", ["microgravity", "0-g", "1-g", "partial gravity", "gravity"]),
    ("post_fire", "Post-fire / toxicology addressed", ["toxic", "CO", "effluent", "smoke particle", "post-fire"]),
    ("human_factors", "Crew response / procedures addressed", ["crew", "procedure", "egress", "response", "training"]),
]


def identify_safety_evidence_gaps(
    investigation: str,
    findings: list[str] | str | None = None,
) -> dict[str, Any]:
    """Check which fire-safety evidence dimensions are covered by supplied findings.

    Dimensions are assessed against the supplied text only — silence means
    'missing' for this evidence set, NOT that NASA never studied it.
    """
    if not isinstance(investigation, str) or not investigation.strip():
        return {"status": "failed", "errors": ["investigation must be a non-empty string"]}
    if findings is None:
        findings_text = ""
    elif isinstance(findings, str):
        findings_text = findings
    else:
        findings_text = " ".join(str(f) for f in findings)
    low = findings_text.lower()

    gaps: list[dict[str, str]] = []
    missing_count = 0
    partial_count = 0
    for dim_id, label, keywords in _SAFETY_DIMENSIONS:
        hits = [k for k in keywords if k.lower() in low]
        if len(hits) >= 2:
            status, note = "covered", f"Evidence text matches: {', '.join(hits)}"
        elif len(hits) == 1:
            status, note = "partial", f"Only indirect mention: {hits[0]}"
            partial_count += 1
        else:
            status, note = "missing", "No mention in supplied evidence text."
            missing_count += 1
        gaps.append({"dimension": dim_id, "label": label, "status": status, "note": note})

    if missing_count >= 5:
        summary = f"Evidence set for {investigation} is thin: {missing_count}/{len(_SAFETY_DIMENSIONS)} safety dimensions uncovered. Do not draw operational conclusions."
    elif missing_count:
        summary = f"{investigation}: {missing_count} missing and {partial_count} partial safety dimensions in the supplied evidence."
    else:
        summary = f"Supplied evidence covers all tracked safety dimensions for {investigation} (keyword coverage)."

    return {
        "status": "completed",
        "investigation": investigation.strip(),
        "gaps": gaps,
        "missing_count": missing_count,
        "partial_count": partial_count,
        "summary": summary,
        "limitations": [
            "Assessment covers ONLY the supplied findings text, not the full NASA corpus.",
            "Keyword coverage is not proof of adequacy — a domain expert must review covered dimensions too.",
            "Not a spacecraft certification artifact.",
        ],
    }
