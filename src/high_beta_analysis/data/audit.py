"""Coverage diagnostics for a retrieved historical price panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from high_beta_analysis.data.market import validate_prices


def audit_price_coverage(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    prices = validate_prices(prices)
    dates = set(prices.loc[prices.ticker.eq(benchmark), "date"])
    if not dates:
        raise ValueError(f"Reference benchmark {benchmark} is absent")
    records = []
    for ticker, group in prices.groupby("ticker", sort=True):
        observed = set(group["date"])
        expected = {d for d in dates if group.date.min() <= d <= group.date.max()}
        missing = len(expected.difference(observed))
        first, last = group.date.min(), group.date.max()
        records.append({
            "ticker": ticker,
            "rows": int(len(group)),
            "first_date": str(first.date()),
            "last_date": str(last.date()),
            "missing_spy_sessions_within_coverage": missing,
            "spy_session_coverage_ratio": float(
                (len(expected) - missing) / len(expected)
            ) if expected else np.nan,
            "zero_volume_rows": int((group.volume == 0).sum()),
            "unusually_large_single_day_return_count": int(
                (group.close.pct_change(fill_method=None).abs() > 0.4).sum()
            ),
        })
    return pd.DataFrame(records)
