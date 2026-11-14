"""Machine-learning assessment tools. No model is ever 'trained' or invented here;
these tools reason about suitability and evaluate metrics the user supplies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def assess_ml_suitability(
    n_samples: int,
    n_features: int,
    target_type: str,
    missing_fraction: float = 0.0,
    class_balance_ratio: float | None = None,
    has_leakage_indicator: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    """Assess whether a dataset is suitable for ML and recommend approaches.

    Args:
        n_samples: Number of rows.
        n_features: Number of input features.
        target_type: 'regression' | 'binary_classification' | 'multiclass_classification' | 'unknown'.
        missing_fraction: Fraction of missing cells, 0..1.
        class_balance_ratio: minority/majority class ratio for classification (0..1).
        has_leakage_indicator: True if a feature suspiciously encodes the target.
        notes: Free-text context (e.g. 'repeated runs per fuel at varying O2').
    """
    errors: list[str] = []
    try:
        n_samples = int(n_samples)
        n_features = int(n_features)
        missing_fraction = float(missing_fraction)
    except (TypeError, ValueError):
        return {"status": "failed", "errors": ["n_samples, n_features and missing_fraction must be numeric"]}
    if n_samples <= 0 or n_features <= 0:
        return {"status": "failed", "errors": ["n_samples and n_features must be positive"]}
    if not 0 <= missing_fraction <= 1:
        errors.append("missing_fraction outside [0,1]; clamped")
        missing_fraction = min(max(missing_fraction, 0.0), 1.0)

    target = (target_type or "unknown").strip().lower()
    limitations: list[str] = []
    recommended: list[str] = []
    leakage: list[str] = []
    feature_eng: list[str] = []
    requirements: list[str] = []

    if n_samples < 50:
        limitations.append(
            f"Only {n_samples} samples — far too few for reliable ML; prefer statistical analysis and uncertainty quantification."
        )
        suitable = False
    elif n_samples < 200:
        limitations.append(f"Small dataset (n={n_samples}); use simple models with cross-validation and report wide intervals.")
        suitable = True
    else:
        suitable = True

    if missing_fraction > 0.2:
        limitations.append(f"Missing fraction {missing_fraction:.0%} — imputation must be principled; otherwise conclusions will not generalise.")
    if n_features > 50 and n_samples < 10 * n_features:
        limitations.append("High feature count relative to samples; apply feature selection / regularisation.")

    if target in {"regression"}:
        recommended += ["Ridge/Lasso regression", "Gradient boosting (XGBoost/LightGBM) with cross-validation", "Gaussian process regression for smooth physical responses"]
        feature_eng += ["Physically meaningful ratios (e.g., oxygen fraction, Damköhler-style groups)", "Log transforms for rates", "Categorical one-hot encoding for fuel type"]
    elif target in {"binary_classification", "multiclass_classification"}:
        recommended += ["Logistic regression (interpretable baseline)", "Random forest / gradient boosting", "SVM with RBF kernel for small n"]
        feature_eng += ["Class-weighting or resampling if imbalanced", "Threshold selection via precision/recall trade-off"]
        if class_balance_ratio is not None:
            if class_balance_ratio < 0.2:
                limitations.append(f"Class imbalance (minority/majority={class_balance_ratio:.2f}); accuracy is misleading — report PR-AUC and recall.")
    else:
        target = "unknown"
        recommended += ["Define the target variable first (e.g. extinction strain, burning rate, ignition outcome) before choosing an algorithm."]
        limitations.append("Target type unknown — ML cannot be specified until the prediction goal is declared.")

    if has_leakage_indicator:
        leakage.append("A feature appears to encode the target (leakage). Remove it or re-define the pipeline before any training.")
        suitable = False
    leakage += [
        "Group-aware splitting: multiple runs of the same experiment/fuel must stay in one split to avoid leakage.",
        "Do not scale/impute using full-data statistics before splitting — fit on train folds only.",
    ]
    if notes:
        requirements.append(f"Context noted: {notes.strip()[:300]}")
    requirements += [
        "Document train/validation/test protocol before training.",
        "Hold out at least one investigation campaign (e.g., BASS-II vs BASS) as an external test set where feasible.",
        "Report uncertainty (CV intervals), not single-point metrics.",
    ]

    return {
        "status": "completed" if suitable else "partial",
        "suitable": suitable,
        "recommended_approaches": recommended,
        "target_candidates": (
            ["burning rate", "extinction condition", "ignition outcome"] if target == "unknown" else []
        ),
        "feature_engineering": feature_eng,
        "leakage_risks": leakage,
        "data_requirements": requirements,
        "limitations": limitations,
        "errors": errors,
        "summary": (
            "ML is appropriate with caution." if suitable
            else "ML is not recommended in the current state — address the listed blockers first."
        ),
    }


def evaluate_model_metrics(
    y_true: list[float],
    y_pred: list[float],
    task_type: str = "regression",
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate user-supplied predictions. No predictions are invented.

    Args:
        y_true: Ground-truth labels/values.
        y_pred: Predicted labels/values.
        task_type: 'regression' | 'classification'.
        y_score: Optional scores/probabilities for binary classification (ROC-AUC).
    """
    if not isinstance(y_true, list) or not isinstance(y_pred, list) or not y_true:
        return {"status": "failed", "errors": ["y_true and y_pred must be non-empty lists"]}
    if len(y_true) != len(y_pred):
        return {"status": "failed", "errors": [f"length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)}"]}
    try:
        yt = np.asarray(y_true, dtype="float64")
        yp = np.asarray(y_pred, dtype="float64")
    except (ValueError, TypeError):
        return {"status": "failed", "errors": ["y_true/y_pred must be numeric (encode labels as numbers)"]}
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        return {"status": "failed", "errors": ["non-finite values in y_true/y_pred"]}

    task = (task_type or "regression").strip().lower()
    metrics: dict[str, float] = {}
    uncertainty: list[str] = []
    overfit: list[str] = []
    limitations: list[str] = []
    n = int(yt.shape[0])

    if task == "regression":
        residuals = yt - yp
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(math.sqrt(np.mean(residuals**2)))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        metrics.update({"mae": mae, "rmse": rmse, "r2": r2, "n": float(n)})
        if r2 < 0:
            overfit.append("Negative R² — the model is worse than predicting the mean.")
        if n < 30:
            uncertainty.append(f"n={n} is small; metric variance is large, report CV intervals.")
        summary = f"Regression on n={n}: MAE={mae:.4g}, RMSE={rmse:.4g}, R²={r2:.4g}."
    elif task == "classification":
        correct = (yt == yp)
        acc = float(correct.mean())
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
        per_class: dict[str, float] = {}
        f1s: list[float] = []
        for lab in labels:
            tp = float(np.sum((yt == lab) & (yp == lab)))
            fp = float(np.sum((yt != lab) & (yp == lab)))
            fn = float(np.sum((yt == lab) & (yp != lab)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[f"f1_class_{lab:g}"] = f1
            f1s.append(f1)
        metrics.update({"accuracy": acc, "macro_f1": float(np.mean(f1s)), "n": float(n), **per_class})
        if y_score is not None:
            try:
                from sklearn.metrics import roc_auc_score

                ys = np.asarray(y_score, dtype="float64")
                if ys.shape[0] == n and len(set(yt.tolist())) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(yt, ys))
                else:
                    limitations.append("roc_auc requires n matching scores and exactly two classes; skipped.")
            except Exception as exc:  # sklearn optional for this metric
                limitations.append(f"roc_auc unavailable: {exc}")
        uncertainty.append(f"Point metrics on n={n}; bootstrap or CV confidence intervals recommended.")
        summary = f"Classification on n={n}: accuracy={acc:.4g}, macro-F1={metrics['macro_f1']:.4g}."
    else:
        return {"status": "failed", "errors": ["task_type must be 'regression' or 'classification'"]}

    limitations += [
        "Metrics describe this y_true/y_pred pair only; they say nothing about unseen conditions.",
        "No comparison against a baseline is included unless supplied explicitly.",
    ]
    return {
        "status": "completed",
        "task_type": task,
        "metrics": metrics,
        "uncertainty": uncertainty,
        "overfitting_indicators": overfit,
        "limitations": limitations,
        "summary": summary,
    }
