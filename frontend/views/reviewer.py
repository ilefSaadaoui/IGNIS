from __future__ import annotations

import streamlit as st

import components as ui
import ignis_client as api
import theme


def render() -> None:
    theme.page_header(
        "REVIEWER",
        "Check the claims, not the confidence.",
        "Map claims to evidence, validate citation reachability and surface unsupported findings. The reviewer is a triage signal, not a certification.",
    )

    tabs = st.tabs(["Claim review", "Citation checks", "Unsupported findings"])

    with tabs[0]:
        theme.section_title("Claim review")
        with st.form("claim_form"):
            claims_raw = st.text_area(
                "Claims (one per line)",
                value=(
                    "Microgravity always increases flame spread rate.\n"
                    "Smoke detection thresholds depend on ventilation."
                ),
                height=120,
                key="claims_raw",
            )
            evidence_raw = st.text_area(
                "Evidence passages (one per line)",
                value=(
                    "Flame spread over thin fuels was slower in microgravity than in 1 g for the tested conditions.\n"
                    "Smoke detection thresholds depend on the atmosphere and ventilation field."
                ),
                height=150,
                key="claims_evidence",
            )
            submitted = st.form_submit_button("Review claims", use_container_width=True)
        if submitted:
            claims = api.parse_lines(claims_raw)
            evidence = api.parse_lines(evidence_raw)
            if not claims:
                st.warning("Provide at least one claim.")
            else:
                st.session_state["claim_result"] = api.review_claims(claims, evidence)
        payload = st.session_state.get("claim_result")
        if payload:
            ui.render_claim_review(payload)
            ui.render_json_block(payload, "claim_review.json")
        else:
            theme.empty_state("Map each claim to the evidence that is supposed to support it.")

    with tabs[1]:
        theme.section_title("Citation validation")
        with st.form("citation_form"):
            citations_raw = st.text_area(
                "Citations (URL per line, or Title | URL)",
                value="NASA NTRS | https://ntrs.nasa.gov/",
                height=120,
                key="citations_raw",
            )
            reachability = st.checkbox(
                "Check HTTP reachability (network calls)",
                value=False,
                key="citation_reachability",
            )
            submitted = st.form_submit_button("Validate citations", use_container_width=True)
        if submitted:
            citations = []
            for line in api.parse_lines(citations_raw):
                if "|" in line:
                    title, url = line.split("|", 1)
                    citations.append({"title": title.strip(), "url": url.strip()})
                else:
                    citations.append(line.strip())
            if not citations:
                st.warning("Provide at least one citation.")
            else:
                st.session_state["citation_result"] = api.citation_checks(
                    citations,
                    reachability=reachability,
                )
        payload = st.session_state.get("citation_result")
        if payload:
            ui.render_citation_checks(payload)
            ui.render_json_block(payload, "citation_checks.json")
        else:
            theme.empty_state("Validate that cited URLs exist and are reachable - not that they support the claim.")

    with tabs[2]:
        theme.section_title("Unsupported findings")
        with st.form("unsupported_form"):
            findings_raw = st.text_area(
                "Findings (one per line)",
                value=(
                    "Microgravity always increases flame spread rate.\n"
                    "Smoke detection thresholds depend on ventilation."
                ),
                height=120,
                key="unsupported_findings_raw",
            )
            evidence_raw = st.text_area(
                "Evidence (one per line)",
                value="Smoke detection thresholds depend on the atmosphere and ventilation field.",
                height=120,
                key="unsupported_evidence",
            )
            submitted = st.form_submit_button("Find unsupported findings", use_container_width=True)
        if submitted:
            findings = api.parse_lines(findings_raw)
            evidence = api.parse_lines(evidence_raw)
            if not findings:
                st.warning("Provide at least one finding.")
            else:
                st.session_state["unsupported_result"] = api.unsupported_findings(findings, evidence)
        payload = st.session_state.get("unsupported_result")
        if payload:
            ui.render_tool_status(payload, "Unsupported findings")
            ui.render_metrics_row(
                [
                    {
                        "label": "Unsupported",
                        "value": payload.get("unsupported_count", 0),
                        "sub": "findings without evidence",
                        "accent": True,
                    },
                    {
                        "label": "Total",
                        "value": payload.get("total_findings", 0),
                        "sub": "checked findings",
                    },
                    {
                        "label": "Verdict",
                        "value": payload.get("verdict", "-"),
                        "sub": "reviewer triage",
                    },
                ]
            )
            for finding in payload.get("unsupported_findings") or []:
                st.markdown(f"- {finding}")
            ui.render_limitations(payload.get("limitations") or [])
            ui.render_json_block(payload, "unsupported_findings.json")
        else:
            theme.empty_state("Find statements that have no overlapping evidence.")

    ui.render_callout(
        "What a verdict means",
        [
            "Supported means consistent with the supplied evidence, not independently verified truth.",
            "Reachable means the page exists, not that it supports the claim.",
            "Token overlap and keyword rubrics are triage heuristics; a qualified reviewer must confirm.",
        ],
        "warn",
    )
