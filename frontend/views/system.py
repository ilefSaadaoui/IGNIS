from __future__ import annotations

import streamlit as st

import components as ui
import data
import ignis_client as api
import theme


def render() -> None:
    theme.page_header(
        "SYSTEM",
        "Know what is running.",
        "Backend status, models, budgets, credentials, corpus coverage and workflow routing - no secret values are shown.",
    )

    ready, error = api.backend_status()
    if ready:
        ui.render_callout(
            "Backend ready",
            ["All required IGNIS modules imported successfully in this process."],
            "good",
        )
    else:
        ui.render_callout("Backend unavailable", [error or "Unknown import failure."], "bad")

    if st.button("Refresh system data", use_container_width=True):
        data.clear_caches()
        st.rerun()

    snapshot = data.settings_snapshot()
    models = snapshot.get("models") or {}
    budgets = snapshot.get("budgets") or {}
    keys = snapshot.get("keys") or {}
    provider_order = snapshot.get("provider_order") or []
    missing = snapshot.get("missing") or []

    theme.section_title("Models")
    st.dataframe(
        [{"role": role, "model": model} for role, model in models.items()],
        use_container_width=True,
        hide_index=True,
    )
    if provider_order:
        st.caption("Provider order: " + " -> ".join(str(provider) for provider in provider_order))

    theme.section_title("Budgets")
    st.dataframe(
        [{"setting": key, "value": value} for key, value in budgets.items()],
        use_container_width=True,
        hide_index=True,
    )

    theme.section_title("Credentials")
    key_badges = []
    for name, configured in keys.items():
        state = "configured" if configured else "missing"
        key_badges.append(ui.badge(f"{name}: {state}", "good" if configured else "warn"))
    st.markdown("".join(key_badges), unsafe_allow_html=True)
    if missing:
        ui.render_callout("Missing or placeholder credentials", [str(item) for item in missing], "warn")
    else:
        st.caption("No missing required credentials were reported.")
    st.caption(f"Log level: {snapshot.get('log_level', '-')}")

    stats = data.corpus_stats()
    theme.section_title("Corpus coverage")
    ui.render_metrics_row(
        [
            {"label": "Documents", "value": stats.get("documents", 0), "sub": "ingested", "accent": True},
            {"label": "Chunks", "value": stats.get("chunks", 0), "sub": "page-anchored"},
            {
                "label": "Dense retrieval",
                "value": "ready" if (stats.get("dense") or {}).get("embeddings_ready") else "unavailable",
                "sub": (stats.get("dense") or {}).get("model", "-"),
            },
        ]
    )
    investigations = stats.get("investigations") or {}
    if investigations:
        rows = sorted(
            [{"investigation": name, "documents": count} for name, count in investigations.items()],
            key=lambda row: (-row["documents"], row["investigation"]),
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        theme.empty_state("No investigation coverage reported.")
    st.caption(f"Corpus directory: {stats.get('corpus_dir', '-')}")

    theme.section_title("Workflow routing")
    catalog = data.workflow_catalog()
    st.dataframe(
        [
            {
                "code": item.get("code"),
                "workflow": item.get("name"),
                "specialists": api.summarize_layers(item.get("layers") or []),
                "use case": item.get("use_case"),
            }
            for item in catalog
        ],
        use_container_width=True,
        hide_index=True,
    )

    supported, vision_name = api.vision_supported()
    theme.section_title("Vision")
    if supported:
        st.caption(f"Vision-capable model configured: {vision_name}")
    else:
        st.caption(f"Vision model not supported by the current configuration: {vision_name}")

    theme.section_title("Session")
    history = st.session_state.get("run_history", [])
    st.metric("Workflow runs in this browser session", len(history))
    for entry in history:
        st.caption(entry["label"])
    if history:
        if st.button("Clear run history", key="system_clear_history", use_container_width=True):
            st.session_state["run_history"] = []
            if "last_result" in st.session_state:
                del st.session_state["last_result"]
            st.rerun()

    with st.expander("Local setup and run commands"):
        st.code(
            "py -3.13 -m venv .venv\n"
            ".venv\\Scripts\\Activate.ps1\n"
            "pip install -r requirements.txt\n"
            "pip install -r frontend/requirements.txt",
            language="powershell",
        )
        st.code("streamlit run frontend/app.py", language="powershell")
        st.caption("Set GROQ_API_KEY in .env before running. The frontend calls the backend in-process.")

    ui.render_callout(
        "Operational boundaries",
        [
            "Sessions are in-memory only; conversation state does not survive across runs or processes.",
            "The frontend calls the ADK backend directly; there is no HTTP API in this build.",
            "Corpus retrieval uses only documents ingested by scripts/ingest_corpus.py.",
            "Terms, model names and budgets are read from the backend settings; secrets are never displayed.",
        ],
        "warn",
    )
