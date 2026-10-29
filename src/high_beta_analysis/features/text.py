"""FinBERT scoring and publication-time-safe per-session news aggregation.

Each article goes to the FIRST available session whose cutoff is at or after
publication, never to a prediction made earlier. Missing news is a distinct feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEXT_FEATURES = [
    "news_count", "finbert_positive_mean", "finbert_negative_mean",
    "finbert_neutral_mean", "finbert_sentiment_mean", "finbert_sentiment_std",
    "finbert_sentiment_abs_max", "news_missing",
]
PROBS = ("positive", "negative", "neutral")


def normalize_news(news: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "published_at", "headline"}
    if not required.issubset(news.columns):
        raise ValueError(f"News must include {sorted(required)}")
    result = news.copy()
    result["published_at"] = pd.to_datetime(
        result["published_at"], utc=True, errors="raise"
    )
    if result["published_at"].isna().any():
        raise ValueError("News publication timestamps cannot be null")
    if result["headline"].isna().any() or result["ticker"].isna().any():
        raise ValueError("News ticker/headline cannot be null")
    return result.drop_duplicates(
        ["ticker", "published_at", "headline"]
    ).sort_values(["published_at", "ticker"]).reset_index(drop=True)


def scores_from_finbert_output(rows: list[list[dict]]) -> pd.DataFrame:
    """Retain all three FinBERT probabilities, not just the winning label."""
    values = []
    for item in rows:
        scores = {
            str(entry["label"]).lower(): float(entry["score"]) for entry in item
        }
        if set(scores) != set(PROBS):
            raise ValueError(f"Unexpected FinBERT classes: {sorted(scores)}")
        if not np.isclose(sum(scores.values()), 1, atol=1e-3):
            raise ValueError("FinBERT class probabilities do not sum to one")
        values.append({f"p_{name}": scores[name] for name in PROBS})
    return pd.DataFrame(values, columns=[f"p_{name}" for name in PROBS])


def score_finbert(news: pd.DataFrame, model_name: str = "ProsusAI/finbert",
                  batch_size: int = 8) -> pd.DataFrame:
    """Run the actual Hugging Face FinBERT model on headlines; CPU works."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install FinBERT: pip install -e '.[nlp]'") from exc
    articles = normalize_news(news)
    if articles.empty:
        return articles.assign(
            p_positive=pd.Series(dtype=float),
            p_negative=pd.Series(dtype=float),
            p_neutral=pd.Series(dtype=float),
        )
    classifier = pipeline("text-classification", model=model_name, device=-1)
    probabilities = scores_from_finbert_output(classifier(
        articles["headline"].tolist(), top_k=None,
        batch_size=batch_size, truncation=True, max_length=512,
    ))
    return pd.concat([articles, probabilities], axis=1)


def aggregate_news_to_sessions(news: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate articles at their first eligible regular-session cutoff.

    Require timezone-aware published_at timestamps; date-only news cannot
    establish whether a headline was public before the close.
    """
    required = {"ticker", "date", "prediction_cutoff_utc"}
    if not required.issubset(sessions.columns):
        raise ValueError(f"Sessions must include {sorted(required)}")
    articles = normalize_news(news)
    pcols = [f"p_{label}" for label in PROBS]
    if not set(pcols).issubset(articles):
        raise ValueError("Score articles with FinBERT before joining stock sessions")
    if articles[pcols].isna().any().any():
        raise ValueError("Missing sentiment probabilities")
    if ((articles[pcols] < 0) | (articles[pcols] > 1)).any().any():
        raise ValueError("Invalid sentiment probabilities")
    if not np.allclose(articles[pcols].sum(axis=1), 1, atol=1e-3):
        raise ValueError("Sentiment probabilities must sum to 1")

    cuts = sessions[["ticker", "date", "prediction_cutoff_utc"]].drop_duplicates().copy()
    cuts["prediction_cutoff_utc"] = pd.to_datetime(
        cuts["prediction_cutoff_utc"], utc=True
    )
    if cuts.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date session cutoff")
    if articles.empty:
        return _fill_news_features(cuts[["ticker", "date"]].copy())

    # merge_asof requires global ordering by the time key.
    mapped = pd.merge_asof(
        articles.sort_values("published_at"),
        cuts.sort_values("prediction_cutoff_utc"),
        by="ticker", left_on="published_at", right_on="prediction_cutoff_utc",
        direction="forward", allow_exact_matches=True,
    ).dropna(subset=["date"])
    if (mapped["published_at"] > mapped["prediction_cutoff_utc"]).any():
        raise ValueError("News after the prediction cutoff entered the features")
    mapped["sentiment"] = mapped["p_positive"] - mapped["p_negative"]
    mapped["abs_sentiment"] = mapped["sentiment"].abs()
    out = mapped.groupby(["ticker", "date"], sort=False).agg(
        news_count=("sentiment", "size"),
        finbert_positive_mean=("p_positive", "mean"),
        finbert_negative_mean=("p_negative", "mean"),
        finbert_neutral_mean=("p_neutral", "mean"),
        finbert_sentiment_mean=("sentiment", "mean"),
        finbert_sentiment_std=("sentiment", lambda v: float(v.std(ddof=0))),
        finbert_sentiment_abs_max=("abs_sentiment", "max"),
    ).reset_index()
    return _fill_news_features(cuts[["ticker", "date"]].merge(
        out, how="left", on=["ticker", "date"], validate="one_to_one",
    ))


def _fill_news_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "news_count" not in frame:
        frame["news_count"] = 0
    frame["news_missing"] = (frame["news_count"].fillna(0) == 0).astype(int)
    for col in TEXT_FEATURES:
        if col not in frame:
            frame[col] = 0.0
        frame[col] = frame[col].fillna(0.0)
    return frame[["ticker", "date", *TEXT_FEATURES]]
