"""Normalize daily adjusted OHLCV from a research data provider.

Historical revisions, adjusted OHLCV and point-in-time availability require
separate data-quality checks before making claims about live performance.
"""
from __future__ import annotations

import pandas as pd

REQUIRED = ("date", "ticker", "open", "high", "low", "close", "volume")


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Market download requires: pip install -e '.[market]'") from exc
    if not tickers:
        raise ValueError("At least one ticker is required")
    frames = []
    for ticker in dict.fromkeys(tickers):
        history = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d", auto_adjust=True, actions=False
        )
        if history.empty:
            raise ValueError(f"No daily history returned for {ticker}; review ticker/date range")
        history = history.reset_index()
        date_col = "Date" if "Date" in history else "Datetime"
        history = history.rename(columns={date_col: "date", **{
            col: col.lower() for col in ("Open", "High", "Low", "Close", "Volume")
        }})
        history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.normalize()
        history["ticker"] = ticker
        frames.append(history[list(REQUIRED)])
    return validate_prices(pd.concat(frames, ignore_index=True))


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    result = prices.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None).dt.normalize()
    result["ticker"] = result["ticker"].astype(str)
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if result.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date price observations")
    if (result["close"].dropna() <= 0).any() or (result["volume"].dropna() < 0).any():
        raise ValueError("Nonpositive close or negative volume")
    if result["close"].isna().any():
        raise ValueError("Missing close prices; inspect provider coverage")
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)
