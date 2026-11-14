from __future__ import annotations

import streamlit as st

import components as ui
import ignis_client as api
import theme


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _csv_input(label: str, key: str) -> dict | None:
    uploaded = st.file_uploader(label, type=["csv"], key=f"{key}_upload")
    pasted = st.text_area("Or paste CSV", height=130, key=f"{key}_paste")
    raw = None
    if uploaded is not None:
        raw = uploaded.getvalue()
    elif pasted.strip():
        raw = pasted.encode("utf-8")
    if raw is None:
        return None
    try:
        parsed = api.parse_csv_columns(raw)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return None
    if not parsed:
        st.warning("No columns found in the CSV.")
        return None
    return parsed


def _histogram(values: list[float], unit: str) -> None:
    import altair as alt
    import pandas as pd

    frame = pd.DataFrame({"value": values})
    chart = alt.Chart(frame).mark_bar(color="#ff6b2c").encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=18), title=unit or "value"),
        y=alt.Y("count():Q", title="count"),
    )
    st.altair_chart(chart, use_container_width=True)


def _bar_means(rows: list[dict], unit: str) -> None:
    import altair as alt
    import pandas as pd

    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_bar(color="#60a5fa").encode(
        x=alt.X("group:N", title=""),
        y=alt.Y("mean:Q", title=unit or "mean"),
        tooltip=["group", "mean", "n"],
    )
    st.altair_chart(chart, use_container_width=True)


def _heatmap(pearson: dict, columns: list[str]) -> None:
    import altair as alt
    import pandas as pd

    rows = []
    for column_a in columns:
        for column_b in columns:
            value = (pearson.get(column_a) or {}).get(column_b)
            rows.append({"a": column_a, "b": column_b, "r": value})
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(
        x=alt.X("a:N", title=""),
        y=alt.Y("b:N", title=""),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Pearson r"),
        tooltip=["a", "b", "r"],
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.value("#e9eef7"),
    )
    st.altair_chart(heat + text, use_container_width=True)


def _render_stats(payload: dict, parsed: list[float], unit: str) -> None:
    ui.render_tool_status(payload, "Descriptive statistics")
    ui.render_metrics_row(
        [
            {"label": "n", "value": payload.get("count", 0), "sub": "complete observations", "accent": True},
            {"label": "Missing", "value": payload.get("missing", 0), "sub": "excluded"},
            {"label": "Mean", "value": _fmt(payload.get("mean")), "sub": unit},
            {"label": "Std", "value": _fmt(payload.get("std")), "sub": unit},
        ]
    )
    ui.render_metrics_row(
        [
            {"label": "Min", "value": _fmt(payload.get("min")), "sub": unit},
            {"label": "Q1", "value": _fmt(payload.get("q1")), "sub": unit},
            {"label": "Median", "value": _fmt(payload.get("median")), "sub": unit},
            {"label": "Q3", "value": _fmt(payload.get("q3")), "sub": unit},
            {"label": "Max", "value": _fmt(payload.get("max")), "sub": unit},
        ]
    )
    clean = [value for value in parsed if value is not None]
    if len(clean) >= 2:
        _histogram(clean, unit)
    for note in payload.get("notes") or []:
        st.caption(note)
    ui.render_json_block(payload, "descriptive_statistics.json")


def render() -> None:
    theme.page_header(
        "DATA & ML",
        "Compute the numbers, keep the caveats.",
        "Descriptive statistics, missingness, group comparisons, correlation, ML suitability and metric evaluation - deterministic and fully reported.",
    )

    tabs = st.tabs(
        [
            "Descriptive stats",
            "Missing values",
            "Compare groups",
            "Correlation",
            "ML suitability",
            "Model metrics",
        ]
    )

    with tabs[0]:
        theme.section_title("Descriptive statistics")
        values_raw = st.text_area(
            "Values (comma, space or newline separated)",
            value="1.2, 1.4, 1.1, 1.3, 1.25, 1.22, 1.31",
            height=110,
            key="stats_values",
        )
        unit = st.text_input("Unit (optional)", value="mm/s", key="stats_unit")
        if st.button("Compute statistics", key="stats_run", use_container_width=True):
            parsed = api.parse_numbers(values_raw)
            st.session_state["stats_parsed"] = parsed
            with st.spinner("Computing statistics..."):
                st.session_state["stats_result"] = api.descriptive_statistics(parsed, unit or None)
        payload = st.session_state.get("stats_result")
        if payload:
            parsed = st.session_state.get("stats_parsed") or []
            _render_stats(payload, parsed, unit)
        else:
            theme.empty_state("Enter observations to compute descriptive statistics.")

    with tabs[1]:
        theme.section_title("Missing values")
        data_frame = _csv_input("CSV with a header row", "missing")
        if data_frame is not None:
            row_count = max((len(values) for values in data_frame.values()), default=0)
            st.caption(f"{len(data_frame)} columns, up to {row_count} rows.")
            st.dataframe(
                [{"column": name, "rows": len(values)} for name, values in data_frame.items()],
                use_container_width=True,
                hide_index=True,
            )
        if st.button("Analyze missingness", key="missing_run", use_container_width=True):
            if data_frame is None:
                st.warning("Upload or paste a CSV first.")
            else:
                st.session_state["missing_result"] = api.missing_values(data_frame)
        payload = st.session_state.get("missing_result")
        if payload:
            ui.render_tool_status(payload, "Missing-value analysis")
            ui.render_metrics_row(
                [
                    {"label": "Cells", "value": payload.get("total_cells", 0), "sub": "across columns", "accent": True},
                    {"label": "Missing", "value": payload.get("total_missing", 0), "sub": "cells"},
                    {
                        "label": "Fraction",
                        "value": f"{float(payload.get('missing_fraction') or 0):.1%}",
                        "sub": "of all cells",
                    },
                    {"label": "Max rows", "value": payload.get("max_rows", 0), "sub": "longest column"},
                ]
            )
            per_column = payload.get("per_column") or {}
            rows = []
            for column, info in per_column.items():
                rows.append(
                    {
                        "column": column,
                        "n": info.get("n"),
                        "missing": info.get("missing"),
                        "missing_fraction": info.get("missing_fraction"),
                    }
                )
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            if payload.get("recommendation"):
                st.info(payload["recommendation"])
            ui.render_json_block(payload, "missing_values.json")
        else:
            theme.empty_state("Upload a CSV to quantify missingness column by column.")

    with tabs[2]:
        theme.section_title("Compare experimental groups")
        groups_raw = st.text_area(
            "Groups, one per line as Name: values",
            value="1g: 12.1, 12.4, 11.9, 12.6, 12.2\nmicrogravity: 8.4, 8.8, 8.1, 8.6, 8.3",
            height=150,
            key="groups_raw",
        )
        unit = st.text_input("Unit (optional)", value="mm/s", key="groups_unit")
        if st.button("Compare groups", key="groups_run", use_container_width=True):
            groups = api.parse_named_groups(groups_raw)
            st.session_state["groups_parsed"] = groups
            st.session_state["groups_result"] = api.compare_groups(groups, unit or None)
        payload = st.session_state.get("groups_result")
        if payload:
            ui.render_tool_status(payload, "Group comparison")
            ui.render_metrics_row(
                [
                    {
                        "label": "Mean difference",
                        "value": _fmt(payload.get("mean_difference")),
                        "sub": unit,
                        "accent": True,
                    },
                    {"label": "Cohen's d", "value": _fmt(payload.get("cohens_d")), "sub": "effect size"},
                    {
                        "label": "Relative diff",
                        "value": _fmt(payload.get("relative_difference_pct")),
                        "sub": "%",
                    },
                    {"label": "Groups", "value": len(payload.get("groups") or []), "sub": "compared"},
                ]
            )
            rows = payload.get("groups") or []
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                chart_rows = [
                    {"group": row.get("name"), "mean": row.get("mean"), "n": row.get("n")}
                    for row in rows
                    if row.get("mean") is not None
                ]
                if chart_rows:
                    _bar_means(chart_rows, unit)
            if payload.get("interpretation"):
                ui.render_callout("Interpretation", [payload["interpretation"]], "info")
            ui.render_limitations(payload.get("limitations") or [])
            ui.render_json_block(payload, "group_comparison.json")
        else:
            theme.empty_state("Provide at least two named groups to compare means and effect size.")

    with tabs[3]:
        theme.section_title("Correlation")
        data_frame = _csv_input("CSV with numeric columns", "corr")
        if data_frame is not None:
            numeric = api.numeric_columns(data_frame)
            if len(numeric) < 2:
                st.warning("Need at least two numeric columns.")
            else:
                selected = st.multiselect(
                    "Columns",
                    options=numeric,
                    default=numeric[: min(4, len(numeric))],
                    key="corr_columns",
                )
                if st.button("Compute correlation", key="corr_run", use_container_width=True):
                    if len(selected) < 2:
                        st.warning("Select at least two columns.")
                    else:
                        filtered = {column: data_frame[column] for column in selected}
                        st.session_state["corr_result"] = api.correlation(filtered)
        payload = st.session_state.get("corr_result")
        if payload:
            ui.render_tool_status(payload, "Correlation analysis")
            pearson = payload.get("pearson") or {}
            columns = payload.get("columns") or []
            if columns:
                _heatmap(pearson, columns)
                rows = []
                for column_a in columns:
                    row = {"column": column_a}
                    for column_b in columns:
                        row[column_b] = (pearson.get(column_a) or {}).get(column_b)
                    rows.append(row)
                st.dataframe(rows, use_container_width=True, hide_index=True)
            ui.render_limitations(payload.get("limitations") or [])
            ui.render_json_block(payload, "correlation.json")
        else:
            theme.empty_state("Upload numeric columns to compute a Pearson correlation matrix.")

    with tabs[4]:
        theme.section_title("ML suitability")
        with st.form("ml_form"):
            columns = st.columns(3)
            with columns[0]:
                n_samples = st.number_input("Samples", min_value=1, max_value=1000000, value=120)
            with columns[1]:
                n_features = st.number_input("Features", min_value=1, max_value=10000, value=8)
            with columns[2]:
                target_type = st.selectbox("Target type", options=["regression", "classification", "unknown"])
            missing_fraction = st.slider("Missing fraction", min_value=0.0, max_value=1.0, value=0.05)
            columns = st.columns(2)
            with columns[0]:
                class_balance = st.number_input(
                    "Class balance ratio (0 = not supplied)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                )
            with columns[1]:
                leakage = st.checkbox("Leakage indicator present")
            notes = st.text_input("Notes (optional)", value="")
            submitted = st.form_submit_button("Assess ML suitability", use_container_width=True)
        if submitted:
            st.session_state["ml_result"] = api.ml_suitability(
                int(n_samples),
                int(n_features),
                target_type,
                missing_fraction=float(missing_fraction),
                class_balance_ratio=float(class_balance) if class_balance else None,
                has_leakage_indicator=leakage,
                notes=notes or None,
            )
        payload = st.session_state.get("ml_result")
        if payload:
            ui.render_ml_suitability(payload)
            ui.render_json_block(payload, "ml_suitability.json")
        else:
            theme.empty_state("Describe the dataset to get a suitability assessment, not a model.")

    with tabs[5]:
        theme.section_title("Model metrics")
        with st.form("metrics_form"):
            columns = st.columns(2)
            with columns[0]:
                y_true_raw = st.text_area("y_true", value="1, 2, 3, 4, 5", height=110)
            with columns[1]:
                y_pred_raw = st.text_area("y_pred", value="1.1, 1.9, 3.2, 3.8, 5.1", height=110)
            task_type = st.selectbox("Task type", options=["regression", "classification"])
            y_score_raw = st.text_area("y_score (optional, classification only)", height=90)
            submitted = st.form_submit_button("Evaluate metrics", use_container_width=True)
        if submitted:
            y_true = [value for value in api.parse_numbers(y_true_raw) if value is not None]
            y_pred = [value for value in api.parse_numbers(y_pred_raw) if value is not None]
            y_score = [value for value in api.parse_numbers(y_score_raw) if value is not None]
            st.session_state["metrics_result"] = api.model_metrics(
                y_true,
                y_pred,
                task_type=task_type,
                y_score=y_score or None,
            )
        payload = st.session_state.get("metrics_result")
        if payload:
            ui.render_model_metrics(payload)
            ui.render_json_block(payload, "model_metrics.json")
        else:
            theme.empty_state("Supply your own predictions; no predictions are generated.")
