import pandas as pd
import numpy as np
import pytest

from high_beta_analysis.data.demo import make_demo_prices
from high_beta_analysis.features.market import build_market_panel
from high_beta_analysis.evaluation.targets import (
    fit_extreme_thresholds, label_with_training_thresholds,
)
from high_beta_analysis.evaluation.walkforward import WalkForward
from high_beta_analysis.evaluation.experiment import run_experiments
from high_beta_analysis.config import load_config


def test_threshold_fit_on_train_only():
    train = pd.DataFrame({
        "ticker": ["IREN"] * 3,
        "next_abs_return": [0.01, 0.02, 0.03],
    })
    future = pd.DataFrame({
        "ticker": ["IREN"], "next_abs_return": [0.8],
    })
    t = fit_extreme_thresholds(train, 90)
    assert 0.02 < t["IREN"] < 0.03
    assert label_with_training_thresholds(future, t).tolist() == [1]
    with pytest.raises(ValueError, match="Unseen ticker"):
        label_with_training_thresholds(
            pd.DataFrame({"ticker": ["OTHER"], "next_abs_return": [0.3]}), t
        )


def test_splits_date_disjoint_and_gap():
    dates = pd.bdate_range("2026-01-01", periods=33)
    frame = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["IREN"] * 33 + ["NBIS"] * 33,
    })
    folds = list(WalkForward(
        min_train_dates=15, test_dates=5, step_dates=5, gap_dates=1
    ).split(frame))
    assert folds
    for tr, te in folds:
        train_dates = set(frame.iloc[tr].date)
        test_dates = set(frame.iloc[te].date)
        assert not train_dates & test_dates
        assert max(train_dates) < min(test_dates)
        assert len(tr) % 2 == 0 and len(te) % 2 == 0


def test_end_to_end_classical_models_on_synthetic_fixture():
    cfg = load_config()
    cfg["data"]["equities"] = ["IREN", "NBIS"]
    cfg["evaluation"].update(
        min_train_dates=100, test_dates=15, step_dates=15,
        gap_dates=1, max_splits=2,
    )
    cfg["models"] = [
        "dummy", "logistic_regression", "decision_tree",
        "random_forest", "svm_rbf",
    ]
    panel = build_market_panel(
        make_demo_prices(["IREN", "NBIS"], sessions=155), ["IREN", "NBIS"]
    )
    result = run_experiments(panel, cfg)
    assert len(result) == 2 * 6  # 5 classifiers plus regression in each fold
    assert set(result.ablation) == {"market-only"}
    assert result.loc[
        result.model.eq("logistic_regression"), "pr_auc"
    ].notna().any()
    assert result.loc[result.model.eq("svm_rbf"), "brier"].isna().all()
