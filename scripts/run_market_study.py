"""Download a research panel, audit coverage, and run real-data market-only models.

Downloads are local and artifacts are not committed or publicly redistributed.
An unavailable stock is reported and excluded, never silently fabricated.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from high_beta_analysis.config import load_config
from high_beta_analysis.data.market import download_prices
from high_beta_analysis.data.audit import audit_price_coverage
from high_beta_analysis.features.market import build_market_panel
from high_beta_analysis.evaluation.experiment import run_experiments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="reports/real-market-pilot")
    p.add_argument("--min-equities", type=int, default=2)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    universe = list(dict.fromkeys(cfg["data"]["equities"] + cfg["data"]["benchmarks"]))
    frames, errors = [], {}
    for ticker in universe:
        try:
            frame = download_prices([ticker], cfg["data"]["start"], cfg["data"]["end"])
            frames.append(frame)
            print(f"Retrieved {ticker}: {len(frame)} rows", flush=True)
        except Exception as exc:
            errors[ticker] = str(exc)[:500]
            print(f"Unavailable {ticker}: {errors[ticker]}", flush=True)
    if not frames:
        raise RuntimeError("No real market price data could be obtained. See provider logs.")
    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(out / "prices.csv", index=False)
    audit = audit_price_coverage(prices)
    audit.to_csv(out / "coverage.csv", index=False)

    available = sorted(set(prices.ticker))
    equities = [s for s in cfg["data"]["equities"] if s in available]
    if "SPY" not in available:
        raise RuntimeError("SPY missing: cannot construct valid market-beta benchmark")
    if len(equities) < args.min_equities:
        raise RuntimeError(
            f"Only {len(equities)} available stocks; minimum {args.min_equities}"
        )
    # Do not silently keep the originally configured tickers during modeling.
    cfg["data"]["equities"] = equities
    panel = build_market_panel(prices, equities)
    labeled = panel[panel.next_abs_return.notna()]
    counts = labeled.groupby("ticker").date.nunique()
    eligible = counts[counts >= cfg["evaluation"]["min_train_dates"] + (
        cfg["evaluation"]["gap_dates"] + cfg["evaluation"]["test_dates"]
    )].index
    excluded = sorted(set(equities).difference(eligible))
    cfg["data"]["equities"] = [t for t in equities if t in set(eligible)]
    if len(cfg["data"]["equities"]) < args.min_equities:
        raise RuntimeError("Too few eligible tickers have enough training/test dates")
    scores = run_experiments(panel, cfg)
    scores.to_csv(out / "market_only_metrics.csv", index=False)
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance Yahoo historical daily OHLCV; unverified point-in-time",
        "python": platform.python_version(),
        "data_start": cfg["data"]["start"],
        "data_end": cfg["data"]["end"],
        "eligible_equities": cfg["data"]["equities"],
        "short_history_excluded": excluded,
        "missing_downloads": errors,
        "benchmark": "SPY",
        "prediction_cutoff_assumption": cfg["project"]["prediction_cutoff"],
        "limitations": [
            "Point-in-time adjustment, survivorship and corporate-action history not audited",
            "Vendor delivery latency and early market closes not modeled",
            "This is not a trading strategy or a statistical significance claim",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nCOVERAGE\n" + audit.to_string(index=False), flush=True)
    print("\nMODEL FOLD RESULTS\n" + scores.to_string(index=False), flush=True)
    print(f"\nResults saved in {out}", flush=True)


if __name__ == "__main__":
    main()
