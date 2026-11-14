from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import components as ui
import ignis_client as api
import theme


def _save_image(uploaded) -> str:
    if "image_dir" not in st.session_state:
        st.session_state["image_dir"] = tempfile.mkdtemp(prefix="ignis_images_")
    safe_name = Path(uploaded.name).name
    path = Path(st.session_state["image_dir"]) / safe_name
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def _render_figure(payload: dict, path: str) -> None:
    ui.render_tool_status(payload, "Figure analysis")
    status = payload.get("status")
    if status == "unavailable":
        ui.render_callout(
            "Vision model unavailable",
            payload.get("uncertainty") or ["No vision-capable model is configured."],
            "warn",
        )
    image_columns = st.columns([1, 1])
    with image_columns[0]:
        try:
            st.image(path, use_container_width=True)
        except Exception:
            st.caption("Preview unavailable.")
    with image_columns[1]:
        ui.render_metrics_row(
            [
                {"label": "Format", "value": payload.get("image_format", "-"), "sub": "source image"},
                {
                    "label": "Size",
                    "value": f"{int(payload.get('image_bytes') or 0) / 1024:.0f} KB",
                    "sub": "upload",
                },
                {"label": "Model", "value": payload.get("vision_model", "-"), "sub": "vision backend"},
            ]
        )
        if payload.get("interpretation"):
            st.markdown("**Interpretation**")
            st.markdown(payload["interpretation"])
    for key, label in (
        ("visible_labels", "Visible labels"),
        ("axes", "Axes"),
        ("legend", "Legend"),
    ):
        values = payload.get(key) or []
        if values:
            theme.section_title(label)
            st.markdown("".join(ui.badge(str(value), "muted") for value in values), unsafe_allow_html=True)
    ui.render_limitations(payload.get("uncertainty") or [])
    ui.render_limitations(payload.get("limitations") or [])
    if payload.get("raw_model_output"):
        with st.expander("Raw model output"):
            st.code(payload["raw_model_output"])
    ui.render_json_block(payload, "figure_analysis.json")


def render() -> None:
    theme.page_header(
        "FIGURES",
        "Read the chart, not a hallucination.",
        "Send a scientific figure to the configured vision model and get labels, axes, legend, interpretation and uncertainty - clearly marked as AI-generated.",
    )

    supported, model_name = api.vision_supported()
    if supported:
        st.caption(f"Vision model configured: {model_name}")
    else:
        st.caption(f"Vision model unavailable: {model_name}")

    uploaded = st.file_uploader(
        "Upload a figure",
        type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
        key="figure_upload",
    )
    question = st.text_area(
        "What should be extracted?",
        value="Describe the figure: type, axes, labels, legend, trend and any uncertainty shown.",
        height=100,
        key="figure_question",
    )
    ocr_text = st.text_area(
        "Optional OCR text",
        height=90,
        key="figure_ocr",
        placeholder="Paste OCR output if you have it; it is passed as context, not truth.",
    )

    ui.render_callout(
        "AI reading is not measurement",
        [
            "Vision output may misread small text, dense legends or overlapping series.",
            "Nothing is invented when the model is unavailable; the failure is reported.",
            "Verify every extracted number against the source figure before use.",
        ],
        "warn",
    )

    if st.button("Analyze figure", type="primary", use_container_width=True):
        if uploaded is None:
            st.warning("Upload an image first.")
        elif not question.strip():
            st.warning("Enter what should be extracted.")
        else:
            path = _save_image(uploaded)
            st.session_state["figure_path"] = path
            with st.spinner("Sending the figure to the vision model..."):
                st.session_state["figure_result"] = api.analyze_figure(
                    path,
                    question.strip(),
                    ocr_text.strip() or None,
                )

    payload = st.session_state.get("figure_result")
    path = st.session_state.get("figure_path")
    if payload and path:
        theme.divider()
        _render_figure(payload, path)
    else:
        theme.empty_state("Upload a figure and request an evidence-marked reading.")
