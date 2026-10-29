"""Command line entry points for fixtures, acquisition, FinBERT and evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from high_beta_analysis.config import load_config
from high_beta_analysis.data.demo import make_demo_prices, make_demo_news
from high_beta_analysis.data.market import download_prices
from high_beta_analysis.features.market import build_market_panel
from high_beta_analysis.features.text import aggregate_news_to_sessions, score_finbert
from high_beta_analysis.evaluation.experiment import run_experiments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="High-beta infrastructure equity ML research"
    )
    parser.add_argument("command", choices=["demo", "download", "finbert", "evaluate"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prices", default="data/raw/prices.csv")
    parser.add_argument("--news", default=None, help="Raw news for finbert, scored news for evaluate")
    parser.add_argument("--output", default=None)
    parser.add_argument("--demo-days", type=int, default=340)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    equities = config["data"]["equities"]

    if args.command == "demo":
        output = Path(args.output or "data/raw")
        output.mkdir(parents=True, exist_ok=True)
        prices = make_demo_prices(
            equities, args.demo_days, config["project"]["seed"]
        )
        dates = pd.to_datetime(prices.loc[
            prices.ticker.eq(equities[0]), "date"
        ])
        news = make_demo_news(equities, dates, config["project"]["seed"])
        prices.to_csv(output / "demo_prices.csv", index=False)
        news.to_csv(output / "demo_news.csv", index=False)
        print(f"Synthetic only: {output}/demo_prices.csv and demo_news.csv")
        print("These fixtures are not financial evidence.")
        return

    if args.command == "download":
        prices = download_prices(
            equities + config["data"]["benchmarks"],
            start=config["data"]["start"], end=config["data"]["end"],
        )
        path = Path(args.output or args.prices)
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path, index=False)
        print(f"Downloaded {len(prices)} OHLCV observations to {path}")
        print("Audit ticker history, provider revisions, and daily cutoff assumptions.")
        return

    if args.command == "finbert":
        if args.news is None:
            parser.error("finbert requires --news path/to/news.csv")
        scored = score_finbert(
            pd.read_csv(args.news), model_name=config["features"]["news_model"]
        )
        output = Path(args.output or "data/processed/scored_news.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output, index=False)
        print(f"Scored {len(scored)} articles with FinBERT: {output}")
        return

    prices = pd.read_csv(args.prices)
    panel = build_market_panel(prices, equities)
    outputs = [run_experiments(panel, config, include_text=False)]
    if args.news:
        scored_news = pd.read_csv(args.news)
        aggregated = aggregate_news_to_sessions(scored_news, panel)
        panel = panel.merge(
            aggregated, on=["date", "ticker"], validate="one_to_one"
        )
        outputs.append(run_experiments(panel, config, include_text=True))
    result = pd.concat(outputs, ignore_index=True)
    output = Path(args.output or "reports/fold_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} fold-level measurements to {output}")
    print("Research evaluation only; validate provenance and time boundaries.")
    print(
        result.groupby(["ablation", "model"])["pr_auc"]
        .mean().round(4).to_string()
    )


if __name__ == "__main__":
    main()
