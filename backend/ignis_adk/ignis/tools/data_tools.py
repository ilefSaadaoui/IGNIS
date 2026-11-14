"""Deterministic scientific data tools (Pandas/NumPy — no LLM arithmetic)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _clean_series(values: Any, name: str = "values") -> pd.Series:
    if isinstance(values, dict) and "values" in values:
        values = values["values"]
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of numbers (null/NaN allowed for missing)")
    arr = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    return arr.astype("float64")


def calculate_descriptive_statistics(values: list[float | None], unit: str | None = None) -> dict[str, Any]:
    """Descriptive statistics for one numeric series.

    Missing entries (None/NaN/unparseable) are excluded from statistics and
    reported. Returns reproducible dict; never fabricates values — empty input
    yields status='partial' with null stats.
    """
    try:
        series = _clean_series(values)
    except ValueError as exc:
        return {"status": "failed", "errors": [str(exc)]}
    total = int(series.shape[0])
    clean = series.dropna()
    missing = total - int(clean.shape[0])
    if clean.empty:
        return {
            "status": "partial", "count": 0, "missing": missing, "total_input": total,
            "mean": None, "std": None, "min": None, "max": None, "median": None,
            "q1": None, "q3": None, "unit": unit,
            "notes": ["All entries were missing or non-numeric; no statistics computed."],
        }
    return {
        "status": "completed",
        "count": int(clean.shape[0]),
        "missing": missing,
        "total_input": total,
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.shape[0] > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "median": float(clean.median()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "unit": unit,
        "notes": [f"n={clean.shape[0]} complete observations; {missing} missing excluded."],
    }


def analyze_missing_values(data: dict[str, list[float | None]]) -> dict[str, Any]:
    """Missing-value analysis for a dict of column -> values (lists may contain nulls)."""
    if not isinstance(data, dict) or not data:
        return {"status": "failed", "errors": ["data must be a non-empty dict of column -> list"]}
    per_column: dict[str, dict[str, Any]] = {}
    total_cells = 0
    total_missing = 0
    max_len = max(len(v) for v in data.values() if isinstance(v, (list, tuple))) if data else 0
    for col, values in data.items():
        try:
            series = _clean_series(values, name=col)
        except ValueError as exc:
            per_column[str(col)] = {"status": "failed", "error": str(exc)}
            continue
        n = int(series.shape[0])
        miss = int(series.isna().sum())
        total_cells += n
        total_missing += miss
        per_column[str(col)] = {
            "n": n,
            "missing": miss,
            "missing_fraction": round(miss / n, 6) if n else 1.0,
        }
    frac = (total_missing / total_cells) if total_cells else 1.0
    if frac == 0:
        rec = "No missing values detected."
    elif frac < 0.05:
        rec = "Minimal missingness (<5%); complete-case analysis is usually defensible — verify missingness mechanism."
    elif frac < 0.20:
        rec = "Moderate missingness (5–20%); consider multiple imputation and sensitivity analysis; do not silently drop rows."
    else:
        rec = "High missingness (>=20%); dataset may not support reliable inference without principled imputation and domain review."
    return {
        "status": "completed",
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_fraction": round(frac, 6),
        "max_rows": max_len,
        "per_column": per_column,
        "recommendation": rec,
    }


def compare_experimental_data(
    groups: dict[str, list[float | None]],
    unit: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> dict[str, Any]:
    """Compare two or more experimental groups on a numeric outcome.

    Deterministic: per-group stats, mean difference and Cohen's d for the
    first two groups. Interpretation is descriptive, not causal.
    """
    if not isinstance(groups, dict) or len(groups) < 2:
        return {"status": "failed", "errors": ["groups must be a dict with at least two named series"]}
    cleaned: dict[str, pd.Series] = {}
    group_stats: list[dict[str, Any]] = []
    for name, values in groups.items():
        try:
            series = _clean_series(values, name=str(name))
        except ValueError as exc:
            return {"status": "failed", "errors": [str(exc)]}
        clean = series.dropna()
        cleaned[str(name)] = clean
        group_stats.append(
            {
                "name": str(name),
                "n": int(clean.shape[0]),
                "mean": float(clean.mean()) if not clean.empty else None,
                "std": float(clean.std(ddof=1)) if clean.shape[0] > 1 else (0.0 if clean.shape[0] == 1 else None),
                "median": float(clean.median()) if not clean.empty else None,
                "min": float(clean.min()) if not clean.empty else None,
                "max": float(clean.max()) if not clean.empty else None,
                "missing": int(series.isna().sum()),
            }
        )

    names = list(cleaned)
    a_name = label_a or names[0]
    b_name = label_b or names[1]
    a, b = cleaned[a_name], cleaned[b_name]
    mean_diff = None
    rel_diff = None
    cohens_d = None
    if not a.empty and not b.empty:
        mean_diff = float(b.mean() - a.mean())
        if a.mean() != 0:
            rel_diff = float(100.0 * (b.mean() - a.mean()) / abs(a.mean()))
        pooled = math.sqrt(
            (((a.std(ddof=1) ** 2) if a.shape[0] > 1 else 0.0) * max(a.shape[0] - 1, 0)
             + ((b.std(ddof=1) ** 2) if b.shape[0] > 1 else 0.0) * max(b.shape[0] - 1, 0))
            / max(a.shape[0] + b.shape[0] - 2, 1)
        ) if (a.shape[0] + b.shape[0] - 2) > 0 else 0.0
        if pooled and pooled > 0:
            cohens_d = float((b.mean() - a.mean()) / pooled)

    interpretation_parts = [
        f"Group '{b_name}' mean minus group '{a_name}' mean = {mean_diff:.4g}{(' ' + unit) if unit else ''}."
        if mean_diff is not None else "Insufficient data for a mean difference.",
    ]
    if cohens_d is not None:
        mag = abs(cohens_d)
        size = "negligible" if mag < 0.2 else "small" if mag < 0.5 else "medium" if mag < 0.8 else "large"
        interpretation_parts.append(f"Cohen's d = {cohens_d:.3f} (effect size: {size}).")

    return {
        "status": "completed",
        "groups": group_stats,
        "comparison": {"a": a_name, "b": b_name},
        "mean_difference": mean_diff,
        "cohens_d": cohens_d,
        "relative_difference_pct": rel_diff,
        "unit": unit,
        "interpretation": " ".join(interpretation_parts)
        + " Descriptive only — no causal claim is implied; check confounders and measurement conditions.",
        "limitations": [
            "No hypothesis test is applied unless requested; significance depends on design, not just n.",
            "Missing values were excluded pairwise per group.",
        ],
    }


def correlation_analysis(data: dict[str, list[float | None]]) -> dict[str, Any]:
    """Pearson correlation matrix for numeric columns (listwise, documented)."""
    if not isinstance(data, dict) or len(data) < 2:
        return {"status": "failed", "errors": ["need at least two columns"]}
    try:
        frame = pd.DataFrame({str(k): _clean_series(v, name=str(k)) for k, v in data.items()})
    except ValueError as exc:
        return {"status": "failed", "errors": [str(exc)]}
    corr = frame.corr(method="pearson", min_periods=3).round(4)
    return {
        "status": "completed",
        "columns": list(corr.columns),
        "pearson": corr.where(pd.notna(corr), None).to_dict(),
        "n_pairwise_min": 3,
        "limitations": ["Pearson assumes linearity; inspect scatter behaviour before relying on r."],
    }
