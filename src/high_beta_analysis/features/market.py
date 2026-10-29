"""Build an end-of-session panel from information known by the close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from high_beta_analysis.data.market import validate_prices

MARKET_FEATURES = [
    "ret_1d", "abs_ret_1d", "momentum_5d", "momentum_20d",
    "volatility_5d", "volatility_20d", "volume_ratio_20d",
    "intraday_range", "overnight_gap", "beta_spy_60d",
    "spy_ret_1d", "qqq_ret_1d", "smh_ret_1d", "vix_level",
    "vix_change_1d", "btc_ret_1d_lag1", "relative_ret_spy",
]


def build_market_panel(raw_prices: pd.DataFrame, equities: list[str]) -> pd.DataFrame:
    prices = validate_prices(raw_prices)
    if "SPY" not in set(prices.ticker):
        raise ValueError("SPY is required for beta and the market baseline")
    benchmark = {}
    for symbol, prefix in [("SPY", "spy"), ("QQQ", "qqq"), ("SMH", "smh")]:
        part = prices[prices.ticker.eq(symbol)].set_index("date")["close"]
        if not part.empty:
            benchmark[f"{prefix}_ret_1d"] = part.pct_change(fill_method=None)
    vix = prices[prices.ticker.eq("^VIX")].set_index("date")["close"]
    if not vix.empty:
        benchmark["vix_level"] = vix
        benchmark["vix_change_1d"] = vix.diff()
    btc = prices[prices.ticker.eq("BTC-USD")].set_index("date")["close"]
    if not btc.empty:
        # UTC daily BTC bars may close after U.S. equities: use prior BTC bar.
        benchmark["btc_ret_1d_lag1"] = btc.pct_change(fill_method=None).shift(1)
    context = pd.DataFrame(benchmark).rename_axis("date").reset_index()

    stocks = prices[prices.ticker.isin(equities)].copy()
    if stocks.empty:
        raise ValueError("No configured equity ticker found in price data")
    g = stocks.groupby("ticker", group_keys=False, sort=False)
    stocks["ret_1d"] = g["close"].pct_change(fill_method=None)
    stocks["abs_ret_1d"] = stocks["ret_1d"].abs()
    for window in (5, 20):
        stocks[f"momentum_{window}d"] = g["close"].transform(
            lambda v: v / v.shift(window) - 1
        )
        stocks[f"volatility_{window}d"] = g["ret_1d"].transform(
            lambda v: v.rolling(window, min_periods=window).std()
        )
    median_volume = g["volume"].transform(
        lambda v: v.shift(1).rolling(20, min_periods=20).median()
    )
    stocks["volume_ratio_20d"] = stocks["volume"] / median_volume.replace(0, np.nan)
    stocks["intraday_range"] = (stocks["high"] - stocks["low"]) / stocks["close"]
    stocks["overnight_gap"] = stocks["open"] / g["close"].shift(1) - 1
    stocks = stocks.merge(context, on="date", how="left", validate="many_to_one")
    stocks["relative_ret_spy"] = stocks["ret_1d"] - stocks["spy_ret_1d"]

    stocks["beta_spy_60d"] = np.nan
    for _, indices in stocks.groupby("ticker").groups.items():
        subset = stocks.loc[indices]
        cov = subset["ret_1d"].rolling(60, min_periods=60).cov(subset["spy_ret_1d"])
        var = subset["spy_ret_1d"].rolling(60, min_periods=60).var().replace(0, np.nan)
        stocks.loc[indices, "beta_spy_60d"] = (cov / var).to_numpy()

    stocks["next_return"] = stocks.groupby("ticker")["close"].shift(-1) / stocks["close"] - 1
    stocks["next_abs_return"] = stocks["next_return"].abs()
    stocks["prediction_cutoff_utc"] = (
        stocks["date"] + pd.Timedelta(hours=16)
    ).dt.tz_localize(
        "America/New_York", ambiguous="raise", nonexistent="raise"
    ).dt.tz_convert("UTC")
    return stocks.sort_values(["date", "ticker"]).reset_index(drop=True)
