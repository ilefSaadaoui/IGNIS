from __future__ import annotations

import streamlit as st

import components as ui
import data
import ignis_client as api
import theme

EXAMPLES = [
    (
        "Flame spread",
        "How does flame spread over thin solid fuels in microgravity, and which NASA experiments support it?",
    ),
    (
        "Fire safety gaps",
        "What are the main fire-safety evidence gaps for crewed spacecraft, and which investigations address them?",
    ),
    (
        "Compare experiments",
        "Compare the BASS and FLEX microgravity flame spread investigations and cite the evidence.",
    ),
    (
        "Challenge a claim",
        "Is it established that microgravity always increases flame spread rate? Review the evidence.",
    ),
]


def _goto(question: str) -> None:
    st.session_state["pending_question"] = question
    st.session_state["nav"] = "Ask IGNIS"


def render() -> None:
    theme.page_header(
        "IGNIS / SPACE COMBUSTION INTELLIGENCE",
        "Microgravity fire research, answered with evidence.",
        "Eight specialist agents over NASA NTRS, a page-anchored literature corpus, documents, datasets, figures and a scientific reviewer.",
    )

    stats = data.corpus_stats()
    if stats.get("status") != "failed":
        ui.render_corpus_banner(stats)

    catalog = data.workflow_catalog()

    theme.section_title("Start with a question")
    st.caption("These route into the full swarm with evidence, citations, limitations and a reviewer verdict.")
    columns = st.columns(2)
    for index, (label, question) in enumerate(EXAMPLES):
        with columns[index % 2]:
            st.button(
                label,
                key=f"home_example_{index}",
                use_container_width=True,
                on_click=_goto,
                args=(question,),
            )

    theme.divider()
    theme.section_title("Seven workflows, one reviewer")
    card_columns = st.columns(2)
    for index, item in enumerate(catalog):
        layers = api.summarize_layers(item.get("layers") or [])
        card = f"""
        <div class="ignis-card">
          <div class="ignis-card__label">Workflow {ui.esc(item.get('code'))}</div>
          <div class="ignis-card__value" style="font-size:1.1rem">{ui.esc(item.get('name'))}</div>
          <div class="ignis-card__sub">{ui.esc(item.get('description'))}</div>
          <div class="ignis-card__sub"><strong>Use when:</strong> {ui.esc(item.get('use_case'))}</div>
          <div class="ignis-card__sub"><strong>Specialists:</strong> {ui.esc(layers)}</div>
        </div>
        """
        with card_columns[index % 2]:
            st.markdown(card, unsafe_allow_html=True)
            st.write("")

    theme.divider()
    theme.section_title("Built for scientific honesty")
    pillar_columns = st.columns(3)
    pillars = [
        (
            "Evidence first",
            "Every answer is grounded in retrieved passages, NASA records or supplied data. Missing evidence is reported as missing.",
        ),
        (
            "Instrumented",
            "Each specialist step is timed and audited. Partial failures are surfaced, not hidden.",
        ),
        (
            "Reviewed",
            "A scientific reviewer checks claims against evidence and returns a verdict before you rely on the answer.",
        ),
    ]
    for column, (title, body) in zip(pillar_columns, pillars):
        with column:
            st.markdown(
                f"""
                <div class="ignis-card">
                  <div class="ignis-card__label">{ui.esc(title)}</div>
                  <div class="ignis-card__sub">{ui.esc(body)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    ui.render_callout(
        "What IGNIS will not do",
        [
            "It will not invent citations, values or experiments; absent evidence is reported as absent.",
            "It will not certify flight hardware or replace a fire-safety authority.",
            "It will not persist conversations; every workflow run starts a fresh in-memory session.",
        ],
        "warn",
    )
