"""Synthetic test fixtures only; not observations about any real company."""
import numpy as np
import pandas as pd


def make_demo_prices(equities: list[str], sessions: int = 340, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=sessions)
    market = rng.normal(0.0003, 0.012, sessions)
    btc = rng.normal(0.0006, 0.026, sessions)
    context = {
        "SPY": (market, 475),
        "QQQ": (market * 1.35 + rng.normal(0, 0.004, sessions), 390),
        "SMH": (market * 1.6 + rng.normal(0, 0.01, sessions), 180),
        "BTC-USD": (btc, 45000),
    }
    tickers = {**context}
    for i, ticker in enumerate(equities):
        jump = rng.choice([0, -1, 1], p=[0.955, 0.0225, 0.0225], size=sessions)
        daily = market * (1.25 + 0.12 * i) + rng.normal(0, 0.025, sessions) + jump * 0.11
        tickers[ticker] = (daily, 10 + 4 * i)
    rows = []
    for ticker, (r, first) in tickers.items():
        close = first * np.cumprod(1 + r)
        previous = np.r_[first, close[:-1]]
        opening = previous * (1 + rng.normal(0, 0.003, sessions))
        high = np.maximum(opening, close) * (1 + rng.uniform(0.001, 0.014, sessions))
        low = np.minimum(opening, close) * (1 - rng.uniform(0.001, 0.014, sessions))
        volume = rng.integers(200_000, 1_600_000, sessions)
        for d, o, h, l, c, v in zip(dates, opening, high, low, close, volume):
            rows.append((d, ticker, o, h, l, c, v))
    for d, m in zip(dates, market):
        vix = max(10, 21 + 400 * abs(m))
        rows.append((d, "^VIX", vix, vix, vix, vix, 0))
    return pd.DataFrame(rows, columns=[
        "date", "ticker", "open", "high", "low", "close", "volume"
    ])


def make_demo_news(equities: list[str], dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    choices = [
        "New data center capacity agreement announced",
        "Company reports higher operating costs",
        "Quarterly filing discusses capacity expansion",
        "Management reiterates its capital investment plan",
    ]
    rows = []
    for d in dates[::6]:
        for ticker in equities:
            if rng.random() < 0.75:
                published = (pd.Timestamp(d) + pd.Timedelta(hours=11)).tz_localize(
                    "America/New_York"
                ).tz_convert("UTC")
                rows.append({
                    "ticker": ticker, "published_at": published,
                    "headline": rng.choice(choices), "source": "synthetic-fixture"
                })
    return pd.DataFrame(rows, columns=[
        "ticker", "published_at", "headline", "source"
    ])
