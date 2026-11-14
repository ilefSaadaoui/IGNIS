from __future__ import annotations

import streamlit as st

import components as ui
import ignis_client as api
import theme

SAMPLE = (
    "The BASS investigation studied flame spread over thin solid fuels in microgravity aboard the "
    "International Space Station. FLEX examined flame spread over liquid fuels in the Fluids and "
    "Combustion Facility. ACME measured smoldering combustion and extinction limits under low "
    "oxygen concentration. SAFFIRE studied large-scale spacecraft fire safety. SAME and SAME-R "
    "investigated smoke aerosol measurement and detection. These investigations provide evidence "
    "for ignition, flammability, detection, suppression, atmosphere and gravity-transfer safety "
    "dimensions, but they do not certify flight materials."
)


def _set_text() -> None:
    st.session_state["graph_text"] = SAMPLE


def render() -> None:
    theme.page_header(
        "KNOWLEDGE GRAPH",
        "Turn a paper into a map.",
        "Rule-based entity and relationship extraction over supplied text. Every node and edge is traceable; confidence is low until reviewed.",
    )

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        text = st.text_area(
            "Text to extract",
            height=260,
            key="graph_text",
            placeholder="Paste an abstract, methods section or paragraph.",
        )
        columns = st.columns(2)
        with columns[0]:
            st.button("Load sample", key="graph_sample", use_container_width=True, on_click=_set_text)
        with columns[1]:
            run = st.button("Extract graph", key="graph_run", use_container_width=True, type="primary")
    with right:
        theme.section_title("What is extracted")
        st.markdown(
            "- Investigations (BASS, FLEX, ACME, SAFFIRE, SAME)\n"
            "- Materials, facilities, instruments and phenomena\n"
            "- Conditions, organizations and measurements\n"
            "- Typed relationships when entities co-occur in a sentence"
        )
        ui.render_callout(
            "Not a database",
            [
                "Extraction is lexicon and pattern based; novel terms are missed.",
                "Relationship direction and type can be wrong.",
                "Use this as a first pass, then curate before downstream reasoning.",
            ],
            "warn",
        )

    if run:
        if not text.strip():
            st.warning("Paste text first.")
        else:
            with st.spinner("Extracting entities and relationships..."):
                st.session_state["graph_result"] = api.build_knowledge_graph(text)

    payload = st.session_state.get("graph_result")
    if payload:
        theme.divider()
        ui.render_tool_status(payload, "Knowledge graph")
        graph = payload.get("graph") or {}
        ui.render_metrics_row(
            [
                {"label": "Nodes", "value": graph.get("node_count", 0), "sub": "entities", "accent": True},
                {"label": "Edges", "value": graph.get("edge_count", 0), "sub": "relationships"},
                {
                    "label": "Dropped edges",
                    "value": graph.get("dropped_edges", 0),
                    "sub": "missing endpoints",
                },
                {
                    "label": "Types",
                    "value": len(graph.get("node_types") or {}),
                    "sub": "distinct entity types",
                },
            ]
        )
        ui.render_graph(payload)
        ui.render_json_block(payload, "knowledge_graph.json")
    else:
        theme.empty_state("Paste text and extract a graph.")
