"""Create a recent, clearly scoped GDELT + FinBERT research pilot.

Do not claim that a 45-day headline sample supports the 180-day training /
30-day testing regime specified for historical market-only experiments.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from high_beta_analysis.data.news_gdelt import fetch_recent_gdelt
from high_beta_analysis.features.text import score_finbert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["IREN", "NBIS"])
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--maxrecords", type=int, default=75)
    parser.add_argument("--out", default="reports/recent-news-pilot")
    parser.add_argument("--no-model", action="store_true",
                        help="Download headlines without running FinBERT")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    articles, errors = fetch_recent_gdelt(
        args.tickers, args.days, args.maxrecords
    )
    articles.to_csv(out / "article_index.csv", index=False)
    counts = articles.groupby("ticker").size().to_dict() if len(articles) else {}
    meta = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT DOC 2.0 ArticleList",
        "date_field": "GDELT seendate, not publisher publication timestamp",
        "window_days": args.days,
        "limit_per_ticker": args.maxrecords,
        "counts_per_ticker": counts,
        "errors": errors,
        "selection_warning": (
            "ArticleList is capped and sorted newest-first; this is an "
            "incomplete, sampling-biased news corpus, not a historical news archive."
        ),
        "evaluation_warning": (
            "This run is a FinBERT/coverage pilot only; the news and market "
            "comparison requires the same historical training/test period."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if articles.empty:
        raise RuntimeError(f"No articles retrieved. Provider errors: {errors}")
    print(f"Found {len(articles)} recent headlines: {counts}", flush=True)
    if not args.no_model:
        scores = score_finbert(articles)
        scores.to_csv(out / "scored_articles.csv", index=False)
        print(f"FinBERT scored {len(scores)} headlines", flush=True)
    print(f"Provenance and pilot outputs saved in {out}", flush=True)
    print("Not a validated market+news forecast benchmark.", flush=True)


if __name__ == "__main__":
    main()
