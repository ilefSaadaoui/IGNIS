from __future__ import annotations

from typing import Any

import streamlit as st

import ignis_client as api


@st.cache_data(ttl=300, show_spinner=False)
def corpus_stats() -> dict[str, Any]:
    return api.corpus_stats()


@st.cache_data(ttl=600, show_spinner=False)
def workflow_catalog() -> list[dict[str, Any]]:
    return api.workflow_catalog()


@st.cache_data(ttl=600, show_spinner=False)
def settings_snapshot() -> dict[str, Any]:
    return api.settings_snapshot()


@st.cache_data(ttl=300, show_spinner=False)
def nasa_search(query: str, source: str, limit: int) -> dict[str, Any]:
    return api.search_nasa(query, source=source, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def corpus_search(query: str, top_k: int, investigation: str | None) -> dict[str, Any]:
    return api.search_corpus(query, top_k=top_k, investigation=investigation or None)


@st.cache_data(ttl=600, show_spinner=False)
def investigation_metadata(name: str, limit: int) -> dict[str, Any]:
    return api.investigation_metadata(name, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def nasa_images(query: str, media_type: str, page_size: int) -> dict[str, Any]:
    return api.search_images(query, media_type=media_type, page_size=page_size)


def clear_caches() -> None:
    st.cache_data.clear()
