"""Check our fast AMEX metric against the official competition implementation."""

import numpy as np
import pandas as pd
import pytest

from amex.evaluation.metric import amex_metric


def official_amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Reference code published by AmEx on the Kaggle competition's evaluation page."""

    def top_four_percent_captured(y_true, y_pred):
        df = pd.concat([y_true, y_pred], axis="columns").sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df["weight"].sum())
        df["weight_cumsum"] = df["weight"].cumsum()
        df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
        return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true, y_pred):
        df = pd.concat([y_true, y_pred], axis="columns").sort_values("prediction", ascending=False)
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos_found"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(y_true, y_pred):
        y_true_pred = y_true.rename(columns={"target": "prediction"})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    return 0.5 * (g + d)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_official(seed):
    rng = np.random.default_rng(seed)
    n = 5_000
    y = (rng.random(n) < 0.26).astype(int)  # ~26% defaulters, like the real data
    p = np.clip(0.3 * y + rng.random(n), 0, None)  # a noisy but useful model
    official = official_amex_metric(pd.DataFrame({"target": y}), pd.DataFrame({"prediction": p}))
    assert amex_metric(y, p) == pytest.approx(official, abs=1e-9)


def test_perfect_model_scores_one():
    y = np.array([0, 0, 0, 1, 1] * 200)
    assert amex_metric(y, y.astype(float)) == pytest.approx(1.0)


def test_random_model_scores_near_zero_gini():
    rng = np.random.default_rng(7)
    y = (rng.random(50_000) < 0.26).astype(int)
    score = amex_metric(y, rng.random(50_000))
    # Gini ~ 0 and the top 4% catches only a small share of defaulters
    assert score < 0.1


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        amex_metric([0, 1], [0.1])
    with pytest.raises(ValueError):
        amex_metric([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        amex_metric([0, 2], [0.1, 0.2])
