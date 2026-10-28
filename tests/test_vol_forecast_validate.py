from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MINUTE_MS = 60_000
DAY_BARS = 1_440
WINDOW_BARS = 30 * DAY_BARS


def make_history(days: int = 32) -> pd.DataFrame:
    count = days * DAY_BARS
    close = np.exp(np.arange(count, dtype=np.float64) * 0.0001)
    return pd.DataFrame(
        {
            "ts": 1_700_000_000_000 + np.arange(count, dtype=np.int64) * MINUTE_MS,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones(count, dtype=np.float64),
        }
    )


def fake_forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    timestamps = int(history["ts"].iloc[-1]) + np.arange(1, horizon + 1) * MINUTE_MS
    return pd.DataFrame(
        {
            "ts": timestamps,
            "variance": np.full(horizon, 1e-8),
            "volatility": np.full(horizon, 1e-4),
        }
    )


class VolatilityValidationTests(unittest.TestCase):
    def test_rejects_missing_columns_and_invalid_or_gapped_candles(self) -> None:
        from research.vol_forecast_validate import _validate_frame

        valid = make_history(days=1)
        invalid_frames = [
            valid.drop(columns="volume"),
            valid.assign(ts=valid["ts"].astype(np.float64) + 0.5),
            valid.assign(close=valid["close"].mask(valid.index == 2, np.nan)),
            valid.assign(close=valid["close"].mask(valid.index == 2, 0.0)),
            valid.assign(high=valid["high"].mask(valid.index == 2, np.inf)),
            valid.assign(ts=valid["ts"].mask(valid.index == 2, valid["ts"].iloc[1])),
            valid.assign(ts=valid["ts"] + pd.Series([0, 0, MINUTE_MS] + [0] * (len(valid) - 3))),
        ]
        for frame in invalid_frames:
            with self.subTest(columns=list(frame.columns), rows=len(frame)):
                with self.assertRaises(ValueError):
                    _validate_frame(frame)

    def test_uses_exact_trailing_window_and_excludes_test_day(self) -> None:
        from research.vol_forecast_validate import run_validation

        history = make_history(days=32)
        with patch("research.vol_forecast_validate.forecast_volatility", side_effect=fake_forecast) as forecast:
            run_validation(history)

        self.assertEqual(forecast.call_count, 2)
        for fold_index, call in enumerate(forecast.call_args_list):
            training = call.args[0]
            test_start_index = WINDOW_BARS + fold_index * DAY_BARS
            self.assertEqual(len(training), WINDOW_BARS)
            self.assertEqual(int(training["ts"].iloc[0]), int(history["ts"].iloc[test_start_index - WINDOW_BARS]))
            self.assertEqual(int(training["ts"].iloc[-1]), int(history["ts"].iloc[test_start_index - 1]))
            self.assertLess(int(training["ts"].iloc[-1]), int(history["ts"].iloc[test_start_index]))
            self.assertEqual(call.kwargs["horizon"], DAY_BARS)

    def test_scores_daily_and_overall_metrics(self) -> None:
        from research.vol_forecast_validate import run_validation

        history = make_history(days=32)
        output = []

        with patch("research.vol_forecast_validate.forecast_volatility", side_effect=fake_forecast):
            with patch("builtins.print", side_effect=lambda *values: output.append(" ".join(map(str, values)))):
                run_validation(history)

        self.assertEqual(len(output), 3)
        self.assertIn("fold=1", output[0])
        self.assertIn("fold=2", output[1])
        self.assertIn("daily_folds=2", output[2])
        self.assertIn("forecast_bars=2880", output[2])
        realized_returns = np.diff(np.log(history["close"].to_numpy()))[
            WINDOW_BARS - 1 : WINDOW_BARS - 1 + 2 * DAY_BARS
        ]
        expected_rmse = np.sqrt(np.mean((1e-4 - np.abs(realized_returns)) ** 2))
        expected_qlike = np.mean(np.log(1e-8) + realized_returns**2 / 1e-8)
        self.assertIn(f"rmse={expected_rmse:.8g}", output[2])
        self.assertIn(f"qlike={expected_qlike:.8g}", output[2])

    def test_help_runs_without_fitting_model(self) -> None:
        script = REPOSITORY_ROOT / "research" / "vol_forecast_validate.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--csv", result.stdout)
        self.assertIn("walk-forward", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
