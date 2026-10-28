import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import kronos_validate


def make_history(rows: int = 4_333) -> pd.DataFrame:
    ts = 1_700_000_000_000 + np.arange(rows, dtype=np.int64) * 60_000
    close = 100.0 + np.arange(rows, dtype=np.float64) / 100
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(rows, 10.0),
        },
    )


class KronosValidationTests(unittest.TestCase):
    def test_select_origins_returns_latest_and_two_daily_origins(self) -> None:
        history = make_history()
        horizon = 12

        origins = kronos_validate.select_origins(history, horizon)

        expected_latest = len(history) - horizon - 1
        self.assertEqual(origins, [expected_latest, expected_latest - 1_440, expected_latest - 2_880])
        for origin in origins:
            actual = history.iloc[origin + 1 : origin + horizon + 1]
            self.assertEqual(len(actual), horizon)

    def test_run_validation_passes_history_only_through_origin(self) -> None:
        history = make_history()
        origins = kronos_validate.select_origins(history, horizon=12)
        seen_last_timestamps: list[int] = []

        def fake_forecast_paths(
            past: pd.DataFrame,
            pred_len: int,
            sample_count: int,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            seen_last_timestamps.append(int(past["ts"].iloc[-1]))
            future_ts = pd.to_datetime(
                int(past["ts"].iloc[-1]) + np.arange(1, pred_len + 1) * 60_000,
                unit="ms",
                utc=True,
            )
            spread = pd.DataFrame(
                {
                    "ts": future_ts,
                    "step": np.arange(1, pred_len + 1),
                    "close_median": np.full(pred_len, 101.0),
                    "close_p10": np.full(pred_len, 100.0),
                    "close_p90": np.full(pred_len, 102.0),
                    "close_std": np.full(pred_len, 0.5),
                },
            )
            paths = pd.DataFrame(
                {
                    "sample_id": np.repeat(np.arange(sample_count), pred_len),
                    "step": np.tile(np.arange(1, pred_len + 1), sample_count),
                    "close": np.full(sample_count * pred_len, 101.0),
                },
            )
            return paths, spread

        output = io.StringIO()
        with patch.object(kronos_validate, "forecast_paths", side_effect=fake_forecast_paths):
            with contextlib.redirect_stdout(output):
                kronos_validate.run_validation(history, horizon=12, sample_count=20)

        expected_timestamps = [int(history["ts"].iloc[origin]) for origin in origins]
        self.assertEqual(seen_last_timestamps, expected_timestamps)
        self.assertIn("actual_close", output.getvalue())

    def test_run_validation_rejects_missing_minute_bars(self) -> None:
        history = make_history()
        history = history.drop(index=100).reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "continuous 1-minute"):
            kronos_validate.run_validation(history, horizon=12, sample_count=20)


if __name__ == "__main__":
    unittest.main()
