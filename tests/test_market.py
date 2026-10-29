import numpy as np
import pandas as pd
import pytest

from high_beta_analysis.data.demo import make_demo_prices
from high_beta_analysis.data.market import validate_prices
from high_beta_analysis.features.market import build_market_panel


def test_panel_has_correct_future_target_and_cutoff():
    raw = make_demo_prices(["IREN", "NBIS"], sessions=85)
    result = build_market_panel(raw, ["IREN", "NBIS"])
    iren = result[result.ticker.eq("IREN")].sort_values("date").reset_index(drop=True)
    assert np.isnan(iren.next_abs_return.iloc[-1])
    assert np.isclose(
        iren.next_return.iloc[0],
        iren.close.iloc[1] / iren.close.iloc[0] - 1,
    )
    assert iren.prediction_cutoff_utc.dt.tz is not None
    assert iren["beta_spy_60d"].notna().sum() > 0


def test_volume_baseline_excludes_current_day():
    raw = make_demo_prices(["IREN"], sessions=85)
    panel = build_market_panel(raw, ["IREN"]).sort_values("date").reset_index(drop=True)
    assert pd.isna(panel.loc[19, "volume_ratio_20d"])
    expected = panel.loc[20, "volume"] / panel.loc[:19, "volume"].median()
    assert np.isclose(panel.loc[20, "volume_ratio_20d"], expected)


def test_duplicate_price_session_rejected():
    raw = make_demo_prices(["IREN"], sessions=75)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prices(pd.concat([raw, raw.iloc[[0]]]))
