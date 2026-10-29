import numpy as np
import pandas as pd
import pytest

from high_beta_analysis.features.text import (
    aggregate_news_to_sessions, scores_from_finbert_output,
)


def example_sessions():
    days = pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-26"])
    return pd.DataFrame({
        "ticker": ["IREN"] * 3,
        "date": days,
        "prediction_cutoff_utc": (
            days + pd.Timedelta(hours=16)
        ).tz_localize("America/New_York").tz_convert("UTC"),
    })


def test_news_after_close_only_used_next_session():
    rows = pd.DataFrame({
        "ticker": ["IREN", "IREN"],
        "published_at": [
            "2026-10-22T15:30:00-04:00",
            "2026-10-22T16:30:00-04:00",
        ],
        "headline": ["First", "Second"],
        "p_positive": [0.8, 0.2],
        "p_negative": [0.1, 0.6],
        "p_neutral": [0.1, 0.2],
    })
    out = aggregate_news_to_sessions(rows, example_sessions()).sort_values("date")
    assert out.news_count.tolist() == [1, 1, 0]
    assert np.isclose(out.finbert_sentiment_mean.iloc[0], 0.7)
    assert np.isclose(out.finbert_sentiment_mean.iloc[1], -0.4)
    assert out.news_missing.tolist() == [0, 0, 1]


def test_finbert_probabilities_checked():
    result = scores_from_finbert_output([[
        {"label": "positive", "score": 0.5},
        {"label": "negative", "score": 0.2},
        {"label": "neutral", "score": 0.3},
    ]])
    assert np.isclose(result.p_positive.iloc[0], 0.5)
    with pytest.raises(ValueError):
        scores_from_finbert_output([[
            {"label": "positive", "score": 0.8}
        ]])
