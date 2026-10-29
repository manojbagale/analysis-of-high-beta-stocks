"""Fold-specific, per-stock targets; thresholds are fitted on training data only."""
import numpy as np
import pandas as pd


def fit_extreme_thresholds(train: pd.DataFrame, percentile: float = 90) -> dict[str, float]:
    if not 0 < percentile < 100:
        raise ValueError("Percentile must be between 0 and 100")
    if "next_abs_return" not in train:
        raise ValueError("Missing future absolute returns")
    result = train.groupby("ticker")["next_abs_return"].quantile(percentile / 100).to_dict()
    if not result or any(not np.isfinite(x) for x in result.values()):
        raise ValueError("Each training ticker needs valid future-return observations")
    return {str(k): float(v) for k, v in result.items()}


def label_with_training_thresholds(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    bounds = frame["ticker"].map(thresholds)
    if bounds.isna().any():
        raise ValueError("Unseen ticker: cannot derive a threshold from the test set")
    if frame["next_abs_return"].isna().any():
        raise ValueError("Unknown target return; drop the last session before evaluation")
    return (frame["next_abs_return"] > bounds).astype(int)
