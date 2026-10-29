"""Date-based expanding windows; all tickers in a session stay in the same fold."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForward:
    min_train_dates: int = 180
    test_dates: int = 30
    step_dates: int = 30
    gap_dates: int = 1
    max_splits: int | None = 6

    def split(self, frame: pd.DataFrame):
        dates = np.sort(pd.to_datetime(frame["date"]).unique())
        if self.min_train_dates < 1 or self.test_dates < 1 or self.step_dates < 1:
            raise ValueError("train/test/step must each be positive")
        if self.gap_dates < 1:
            raise ValueError("Use at least one session gap for next-session targets")
        emitted = 0
        for start in range(
            self.min_train_dates + self.gap_dates,
            len(dates) - self.test_dates + 1,
            self.step_dates,
        ):
            train_max_idx = start - self.gap_dates - 1
            train_dates = dates[:train_max_idx + 1]
            test_dates = dates[start:start + self.test_dates]
            train_idx = np.flatnonzero(pd.to_datetime(frame["date"]).isin(train_dates))
            test_idx = np.flatnonzero(pd.to_datetime(frame["date"]).isin(test_dates))
            if not len(train_idx) or not len(test_idx):
                continue
            yield train_idx, test_idx
            emitted += 1
            if self.max_splits is not None and emitted >= self.max_splits:
                return
