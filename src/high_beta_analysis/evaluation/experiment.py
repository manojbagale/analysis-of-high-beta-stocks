"""Shared walk-forward folds and training-only thresholds for both data ablations."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from high_beta_analysis.evaluation.walkforward import WalkForward
from high_beta_analysis.evaluation.targets import (
    fit_extreme_thresholds, label_with_training_thresholds,
)
from high_beta_analysis.evaluation.metrics import classifier_metrics
from high_beta_analysis.models.baselines import make_classifier, make_regression
from high_beta_analysis.features.market import MARKET_FEATURES
from high_beta_analysis.features.text import TEXT_FEATURES


def cfg_to_splitter(cfg: dict) -> dict:
    return {
        "min_train_dates": cfg["min_train_dates"],
        "test_dates": cfg["test_dates"],
        "step_dates": cfg["step_dates"],
        "gap_dates": cfg["gap_dates"],
        "max_splits": cfg.get("max_splits"),
    }


def run_experiments(panel: pd.DataFrame, settings: dict,
                    *, include_text: bool = False) -> pd.DataFrame:
    """Run a comparison without full-data feature transforms or target thresholds."""
    cfg = settings["evaluation"]
    splitter = WalkForward(**cfg_to_splitter(cfg))
    equities = set(settings["data"]["equities"])
    features = [name for name in MARKET_FEATURES if name in panel.columns]
    if include_text:
        if not set(TEXT_FEATURES).issubset(panel):
            raise ValueError("Missing FinBERT session features for the multimodal ablation")
        features += TEXT_FEATURES
    if not features:
        raise ValueError("No model features available")
    data = panel.loc[
        panel.ticker.isin(equities) & panel.next_abs_return.notna()
    ].copy()
    data = data.replace([np.inf, -np.inf], np.nan).sort_values(["date", "ticker"])
    if data.empty:
        raise ValueError("No labeled stock-day observations")
    results = []
    fold_count = 0
    for fold, (train_ix, test_ix) in enumerate(splitter.split(data), start=1):
        fold_count += 1
        train, test = data.iloc[train_ix].copy(), data.iloc[test_ix].copy()
        thresholds = fit_extreme_thresholds(
            train, settings["target"]["extreme_abs_return_percentile"]
        )
        y_train = label_with_training_thresholds(train, thresholds)
        y_test = label_with_training_thresholds(test, thresholds)
        if len(y_train.unique()) < 2:
            warnings.warn(f"Fold {fold}: single-class training target; skipped")
            continue
        x_train, x_test = train[["ticker", *features]], test[["ticker", *features]]
        for model_name in settings["models"]:
            model = make_classifier(
                model_name, features, seed=settings["project"]["seed"]
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            has_proba = hasattr(model, "predict_proba")
            scores = (
                model.predict_proba(x_test)[:, 1] if has_proba
                else model.decision_function(x_test)
            )
            stats = classifier_metrics(
                y_test, predictions, scores, probabilities=has_proba
            )
            results.append({
                "fold": fold,
                "ablation": "market+finbert" if include_text else "market-only",
                "model": model_name,
                "train_last_date": str(train.date.max().date()),
                "test_first_date": str(test.date.min().date()),
                "test_last_date": str(test.date.max().date()),
                **stats,
            })
        # Regression is a secondary task, not a binary risk probability.
        regression = make_regression(features).fit(
            x_train, train["next_abs_return"]
        )
        pred_abs = np.maximum(0, regression.predict(x_test))
        results.append({
            "fold": fold,
            "ablation": "market+finbert" if include_text else "market-only",
            "model": "linear_regression_abs_return",
            "train_last_date": str(train.date.max().date()),
            "test_first_date": str(test.date.min().date()),
            "test_last_date": str(test.date.max().date()),
            "samples": len(test),
            "mae_abs_return": float(mean_absolute_error(
                test["next_abs_return"], pred_abs
            )),
        })
    if fold_count == 0:
        raise ValueError("Insufficient dates for configured walk-forward evaluation")
    return pd.DataFrame(results)
