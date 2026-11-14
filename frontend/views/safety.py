from __future__ import annotations

import streamlit as st

import components as ui
import ignis_client as api
import theme


def _render_mission(payload: dict) -> None:
    ui.render_tool_status(payload, "Mission relevance")
    score = float(payload.get("relevance_score") or 0.0)
    ui.render_metrics_row(
        [
            {"label": "Mission", "value": payload.get("mission_context", "-"), "sub": "context profile", "accent": True},
            {"label": "Relevance", "value": f"{score:.2f}", "sub": "rubric 0-1"},
            {
                "label": "High-impact hits",
                "value": sum(1 for item in payload.get("relevance_rationale") or [] if "high=" in item),
                "sub": "keyword rubric",
            },
        ]
    )
    st.progress(min(max(score, 0.0), 1.0))
    if payload.get("summary"):
        st.info(payload["summary"])
    for key, label in (
        ("relevance_rationale", "Relevance rationale"),
        ("implications", "Implications"),
        ("translation_limits", "Translation limits"),
    ):
        values = payload.get(key) or []
        if values:
            theme.section_title(label)
            for value in values:
                st.markdown(f"- {value}")
    ui.render_limitations(payload.get("limitations") or [])
    ui.render_json_block(payload, "mission_relevance.json")


def render() -> None:
    theme.page_header(
        "FIRE SAFETY",
        "Triage relevance, expose the gaps.",
        "Assess how findings map to mission safety and which evidence dimensions are missing. This is research triage, not flight certification.",
    )

    tabs = st.tabs(["Mission relevance", "Evidence gaps"])

    with tabs[0]:
        theme.section_title("Mission relevance")
        with st.form("mission_form"):
            findings_raw = st.text_area(
                "Findings (one per line)",
                value=(
                    "Microgravity flame spread over thin fuels is slower than in 1 g for the tested conditions.\n"
                    "Smoke detection thresholds depend on the atmosphere and ventilation field.\n"
                    "Suppression effectiveness of CO2 differs in low gravity."
                ),
                height=150,
                key="mission_findings",
            )
            mission = st.selectbox(
                "Mission context",
                options=list(api.MISSIONS),
                format_func=lambda key: api.MISSIONS.get(key, key),
                key="mission_context",
            )
            submitted = st.form_submit_button("Assess relevance", use_container_width=True)
        if submitted:
            findings = [line.strip() for line in findings_raw.splitlines() if line.strip()]
            if not findings:
                st.warning("Provide at least one finding.")
            else:
                st.session_state["mission_result"] = api.mission_relevance(findings, mission)
        payload = st.session_state.get("mission_result")
        if payload:
            _render_mission(payload)
        else:
            theme.empty_state("Provide findings to triage against a mission profile.")

    with tabs[1]:
        theme.section_title("Evidence gaps")
        with st.form("gaps_form"):
            investigation = st.text_input("Investigation", value="BASS", key="gaps_investigation")
            findings_raw = st.text_area(
                "Findings / evidence text",
                value=(
                    "The experiment measured ignition delay and flame spread rate at reduced pressure "
                    "and low oxygen concentration in microgravity. Smoke detectors were not evaluated."
                ),
                height=150,
                key="gaps_findings",
            )
            submitted = st.form_submit_button("Identify gaps", use_container_width=True)
        if submitted:
            if not investigation.strip():
                st.warning("Enter an investigation name.")
            else:
                st.session_state["gaps_result"] = api.safety_gaps(
                    investigation.strip(),
                    findings_raw.strip() or None,
                )
        payload = st.session_state.get("gaps_result")
        if payload:
            ui.render_safety_gaps(payload)
            ui.render_json_block(payload, "safety_gaps.json")
        else:
            theme.empty_state("Assess which safety dimensions the supplied evidence does not cover.")

    ui.render_callout(
        "Certification boundary",
        [
            "Keyword coverage is not proof of adequacy; covered dimensions still need expert review.",
            "Microgravity experiments may not scale to compartment fires or as-built spacecraft materials.",
            "This tool does not produce a certified spacecraft safety recommendation.",
        ],
        "warn",
    )
