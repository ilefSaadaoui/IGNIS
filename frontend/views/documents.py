from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import components as ui
import ignis_client as api
import theme


def _upload_path(uploaded) -> str:
    if "upload_dir" not in st.session_state:
        st.session_state["upload_dir"] = tempfile.mkdtemp(prefix="ignis_ui_")
    safe_name = Path(uploaded.name).name
    path = Path(st.session_state["upload_dir"]) / safe_name
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def _ask_findings(question: str) -> None:
    st.session_state["pending_question"] = question
    st.session_state["nav"] = "Ask IGNIS"


def _render_document(payload: dict) -> None:
    if payload.get("status") == "failed":
        ui.render_errors(payload.get("errors") or ["Document analysis failed."], "Document analysis failed")
        return

    load = payload.get("load") or {}
    ui.render_metrics_row(
        [
            {"label": "Format", "value": load.get("extension", "-"), "sub": "Loaded locally"},
            {"label": "Pages", "value": load.get("pages") or "-", "sub": "PDF page count"},
            {"label": "Characters", "value": f"{load.get('char_count', 0):,}", "sub": "Extracted text"},
            {
                "label": "Truncated",
                "value": "yes" if load.get("truncated") else "no",
                "sub": "MAX_DOCUMENT_CHARS limit",
            },
        ]
    )

    analysis = payload.get("analysis") or {}
    if analysis.get("status") != "failed":
        sections = analysis.get("sections_detected") or []
        section_html = "".join(ui.badge(section, "info") for section in sections)
        theme.section_title("Structure", section_html)
        st.caption(
            f"{analysis.get('reference_count_detected', 0)} references detected; "
            f"{len(analysis.get('table_refs') or [])} table mentions; "
            f"{len(analysis.get('figure_refs') or [])} figure mentions."
        )
        references = analysis.get("references") or []
        if references:
            with st.expander(f"References ({len(references)})"):
                for reference in references:
                    st.markdown(f"- {reference}")
        if analysis.get("limitations"):
            ui.render_limitations(analysis["limitations"])

    theme.section_title("Experimental conditions")
    ui.render_conditions(payload.get("conditions") or {})

    theme.section_title("Candidate findings")
    ui.render_findings(payload.get("findings") or {})

    findings = (payload.get("findings") or {}).get("findings") or []
    statements = [finding.get("statement", "") for finding in findings if finding.get("statement")]
    if statements:
        question = (
            "Using only the following extracted findings, identify fire-safety evidence gaps "
            "and assess mission relevance. Findings: " + " | ".join(statements[:8])
        )
        st.button(
            "Send findings to the swarm",
            key="doc_send_findings",
            use_container_width=True,
            on_click=_ask_findings,
            args=(question,),
        )

    ui.render_json_block(payload, "document_analysis.json")


def render() -> None:
    theme.page_header(
        "DOCUMENTS",
        "Read the paper, keep the receipts.",
        "Load a PDF, DOCX, text, CSV or JSON file and extract structure, experimental conditions and candidate findings without inventing values.",
    )

    uploaded = st.file_uploader(
        "Upload a scientific document",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        key="doc_upload",
    )
    pasted = st.text_area(
        "Or paste document text",
        height=180,
        key="doc_text",
        placeholder="Paste an abstract, methods section or full paper text.",
    )

    ui.render_callout(
        "Local, rule-based extraction",
        [
            "Document analysis runs locally with transparent rules; it does not call an LLM.",
            "Absent conditions are reported as not found - values are never invented.",
            "Only the first MAX_DOCUMENT_CHARS characters are analyzed to bound cost and latency.",
        ],
        "warn",
    )

    if st.button("Analyze document", type="primary", use_container_width=True):
        if uploaded is not None:
            path = _upload_path(uploaded)
            with st.spinner("Parsing and analyzing the document..."):
                st.session_state["doc_result"] = api.analyze_document_file(path)
        elif pasted.strip():
            with st.spinner("Analyzing pasted text..."):
                st.session_state["doc_result"] = api.analyze_document_text_only(pasted)
        else:
            st.warning("Upload a file or paste text first.")

    payload = st.session_state.get("doc_result")
    if payload:
        theme.divider()
        _render_document(payload)
        load = payload.get("load") or {}
        text = load.get("text") or ""
        if text:
            with st.expander("Extracted text"):
                st.text_area("Text", value=text, height=280, disabled=True)
    else:
        theme.empty_state("Upload or paste a document, then run the analyzer.")
