from __future__ import annotations

import streamlit as st

import components as ui
import data
import ignis_client as api
import theme


def _render_nasa(payload: dict) -> None:
    ui.render_tool_status(payload, "NASA NTRS search")
    total = payload.get("total_matched")
    if total is not None:
        st.metric("Records matched", total)
    if payload.get("psi_portal_url"):
        st.markdown(f"[Open the NASA PSI portal]({payload['psi_portal_url']})")
    ui.render_records(payload.get("records") or [])
    ui.render_json_block(payload, "nasa_search.json")


def _render_corpus(payload: dict) -> None:
    ui.render_tool_status(payload, "Corpus search")
    retrieval = payload.get("retrieval")
    if retrieval:
        st.caption(f"Retrieval mode: {retrieval}")
    ui.render_passages(payload.get("passages") or [])
    if payload.get("corpus"):
        ui.render_corpus_banner(payload["corpus"])
    ui.render_json_block(payload, "corpus_search.json")


def _render_investigation(payload: dict) -> None:
    ui.render_tool_status(payload, "Investigation metadata")
    expansion = payload.get("expansion")
    if expansion:
        verified = "verified" if payload.get("expansion_verified") else "unverified"
        st.markdown(f"**Expansion:** {expansion} ({verified})")
    total = payload.get("total_matched")
    if total is not None:
        st.metric("Records matched", total)
    background = payload.get("background") or []
    if background:
        st.markdown("**Background**")
        for item in background:
            st.markdown(f"- {item}")
    ui.render_records(payload.get("records") or [])
    ui.render_json_block(payload, "investigation_metadata.json")


def _render_media(payload: dict) -> None:
    ui.render_tool_status(payload, "NASA media search")
    count = payload.get("count", 0)
    total = payload.get("total_hits")
    st.caption(f"{count} items returned; library total hits: {total if total is not None else 'unknown'}")
    ui.render_media_items(payload.get("items") or [])
    ui.render_json_block(payload, "nasa_media.json")


def render() -> None:
    theme.page_header(
        "RESEARCH",
        "Search the evidence, not the internet.",
        "NASA NTRS records, the local page-anchored corpus, investigation metadata and NASA media - each with provenance and limitations.",
    )

    tabs = st.tabs(["NASA NTRS", "Local corpus", "Investigations", "NASA media"])

    with tabs[0]:
        theme.section_title("NASA Technical Reports Server")
        with st.form("nasa_form"):
            query = st.text_input("Query", value="microgravity flame spread", key="nasa_query")
            columns = st.columns(2)
            with columns[0]:
                source = st.selectbox("Source", options=["all", "ntrs", "psi"], key="nasa_source")
            with columns[1]:
                limit = st.slider("Records", min_value=1, max_value=10, value=5, key="nasa_limit")
            submitted = st.form_submit_button("Search NTRS", use_container_width=True)
        if submitted:
            if not query.strip():
                st.warning("Enter a query first.")
            else:
                with st.spinner("Querying NASA NTRS..."):
                    st.session_state["research_nasa"] = data.nasa_search(query.strip(), source, limit)
        payload = st.session_state.get("research_nasa")
        if payload:
            _render_nasa(payload)
        else:
            theme.empty_state("Search NASA NTRS for reports, conference papers and metadata.")

    with tabs[1]:
        theme.section_title("Local page-anchored corpus")
        with st.form("corpus_form"):
            query = st.text_input(
                "Query",
                value="flame spread rate microgravity",
                key="corpus_query",
            )
            columns = st.columns(2)
            with columns[0]:
                investigation = st.selectbox(
                    "Investigation filter",
                    options=["All"] + api.INVESTIGATIONS,
                    key="corpus_investigation",
                )
            with columns[1]:
                top_k = st.slider("Passages", min_value=1, max_value=12, value=5, key="corpus_top_k")
            submitted = st.form_submit_button("Search corpus", use_container_width=True)
        if submitted:
            if not query.strip():
                st.warning("Enter a query first.")
            else:
                with st.spinner("Searching the local corpus..."):
                    filter_value = None if investigation == "All" else investigation
                    st.session_state["research_corpus"] = data.corpus_search(
                        query.strip(), top_k, filter_value
                    )
        payload = st.session_state.get("research_corpus")
        if payload:
            _render_corpus(payload)
        else:
            theme.empty_state("Search the ingested NASA literature with page-level provenance.")

    with tabs[2]:
        theme.section_title("Investigation metadata")
        with st.form("investigation_form"):
            columns = st.columns(2)
            with columns[0]:
                known = st.selectbox(
                    "Known investigation",
                    options=["Custom"] + api.INVESTIGATIONS,
                    key="investigation_known",
                )
            with columns[1]:
                custom = st.text_input("Custom name", value="", key="investigation_custom")
            limit = st.slider("Records", min_value=1, max_value=10, value=5, key="investigation_limit")
            submitted = st.form_submit_button("Retrieve metadata", use_container_width=True)
        if submitted:
            name = custom.strip() or ("" if known == "Custom" else known)
            if not name:
                st.warning("Choose a known investigation or type a custom name.")
            else:
                with st.spinner("Retrieving investigation metadata..."):
                    st.session_state["research_investigation"] = data.investigation_metadata(name, limit)
        payload = st.session_state.get("research_investigation")
        if payload:
            _render_investigation(payload)
        else:
            theme.empty_state("Look up an investigation expansion, background and NASA records.")

    with tabs[3]:
        theme.section_title("NASA Image and Video Library")
        with st.form("media_form"):
            query = st.text_input("Query", value="combustion microgravity", key="media_query")
            columns = st.columns(2)
            with columns[0]:
                media_type = st.selectbox(
                    "Media type",
                    options=["image", "video", "audio"],
                    key="media_type",
                )
            with columns[1]:
                page_size = st.slider("Items", min_value=1, max_value=10, value=4, key="media_page_size")
            submitted = st.form_submit_button("Search media", use_container_width=True)
        if submitted:
            if not query.strip():
                st.warning("Enter a query first.")
            else:
                with st.spinner("Searching NASA media..."):
                    st.session_state["research_media"] = data.nasa_images(query.strip(), media_type, page_size)
        payload = st.session_state.get("research_media")
        if payload:
            _render_media(payload)
        else:
            theme.empty_state("Find NASA imagery and video for briefings and reports. Verify licensing before print.")
