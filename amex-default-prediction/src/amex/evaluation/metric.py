"""The AMEX competition metric.

    M = 0.5 * (G + D)

G = normalized weighted Gini coefficient.
    "How well does the model rank defaulters above non-defaulters overall?"
    Normalized Gini equals 2*AUC - 1 when there are no weights.

D = default rate captured at 4%.
    "If the bank reviews only its riskiest 4% of customers, what share of all
    defaulters does it catch?" This mirrors a real business action: a review team
    has limited capacity.

Weights: every non-defaulter (target = 0) counts 20 times. AmEx kept only 5% of
non-defaulters in the training data, so the weight of 20 restores the real-world
proportions when scoring.

Why write our own vectorised version: the official pandas version is slow (it sorts
the data three times and uses .apply). We call the metric thousands of times
(every fold, every boosting round for early stopping), so speed matters.
tests/test_metric.py checks that both versions give the same answer.
"""

from __future__ import annotations

import numpy as np

NEGATIVE_WEIGHT = 20
TOP_FRACTION = 0.04


def _weighted_gini(y_true: np.ndarray, y_score: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Gini with rows ranked from highest score to lowest."""
    order = np.argsort(-y_score, kind="mergesort")  # stable sort: ties keep input order
    y, w = y_true[order], weights[order]
    random_line = np.cumsum(w / w.sum())
    lorentz = np.cumsum(y * w) / np.sum(y * w)
    return float(np.sum((lorentz - random_line) * w))


def amex_metric(y_true, y_pred) -> float:
    """AMEX score in [-1, 1]; higher is better. Inputs are array-likes of equal length."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("y_true needs both classes")

    weights = np.where(y_true == 0, NEGATIVE_WEIGHT, 1)

    # D: share of all defaulters found inside the top 4% of (weighted) customers
    order = np.argsort(-y_pred, kind="mergesort")
    cutoff = int(TOP_FRACTION * weights.sum())
    in_top = np.cumsum(weights[order]) <= cutoff
    top_four = y_true[order][in_top].sum() / y_true.sum()

    # G: model's Gini divided by the best possible Gini (ranking by the true labels)
    gini = _weighted_gini(y_true, y_pred, weights) / _weighted_gini(y_true, y_true, weights)

    return 0.5 * (gini + top_four)


def lgb_amex_metric(y_pred, data):
    """LightGBM custom eval: returns (name, value, is_higher_better)."""
    return "amex", amex_metric(data.get_label(), y_pred), True


def xgb_amex_metric(y_pred, dmatrix):
    """XGBoost custom eval (for `custom_metric=`): returns (name, value)."""
    return "amex", amex_metric(dmatrix.get_label(), y_pred)
