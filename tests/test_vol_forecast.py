from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


class VolatilityForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = 1_700_000_000_000 + np.arange(100, dtype=np.int64) * 60_000
        self.history = pd.DataFrame(
            {
                "ts": self.timestamps,
                "close": np.exp(np.arange(100, dtype=np.float64) * 0.001),
            }
        )

    def _fit_result(self, values: list[list[float]], convergence_flag: int = 0):
        return SimpleNamespace(
            convergence_flag=convergence_flag,
            forecast=lambda *, horizon, reindex: SimpleNamespace(
                variance=pd.DataFrame(
                    values,
                    columns=[f"h.{step}" for step in range(1, len(values[0]) + 1)],
                )
            ),
        )

    def test_returns_decimal_variance_and_volatility_with_future_timestamps(self) -> None:
        from shadow_bot.signals.vol_forecast import forecast_volatility

        with patch("arch.arch_model") as arch_model:
            arch_model.return_value.fit.return_value = self._fit_result([[4.0, 9.0]])

            forecasts = forecast_volatility(self.history, horizon=2)

        expected_returns = np.diff(np.log(self.history["close"].to_numpy())) * 100
        np.testing.assert_allclose(arch_model.call_args.args[0], expected_returns)
        arch_model.assert_called_once_with(
            arch_model.call_args.args[0],
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        self.assertEqual(list(forecasts.columns), ["ts", "variance", "volatility"])
        self.assertEqual(
            forecasts["ts"].tolist(),
            [self.timestamps[-1] + 60_000, self.timestamps[-1] + 120_000],
        )
        np.testing.assert_allclose(forecasts["variance"], [0.0004, 0.0009])
        np.testing.assert_allclose(forecasts["volatility"], [0.02, 0.03])

    def test_rejects_invalid_timestamps(self) -> None:
        from shadow_bot.signals.vol_forecast import forecast_volatility

        invalid_timestamps = [
            self.timestamps.astype(np.float64) + 0.5,
            self.timestamps.copy(),
            self.timestamps[::-1].copy(),
            self.timestamps.copy(),
            self.timestamps.astype(np.float64),
        ]
        invalid_timestamps[1][1] = invalid_timestamps[1][0]
        invalid_timestamps[3][1] += 60_000
        invalid_timestamps[4][1] = np.nan

        for timestamps in invalid_timestamps:
            with self.subTest(timestamps=timestamps[:3]):
                history = self.history.copy()
                history["ts"] = timestamps
                with self.assertRaises(ValueError):
                    forecast_volatility(history)

        with self.assertRaises(ValueError):
            forecast_volatility(self.history.drop(columns="ts"))

    def test_rejects_nonpositive_or_nonfinite_closes(self) -> None:
        from shadow_bot.signals.vol_forecast import forecast_volatility

        for close in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(close=close):
                history = self.history.copy()
                history.loc[50, "close"] = close
                with self.assertRaises(ValueError):
                    forecast_volatility(history)

        with self.assertRaises(ValueError):
            forecast_volatility(self.history.drop(columns="close"))
        with self.assertRaises(ValueError):
            forecast_volatility(list(self.history.itertuples(index=False)))

    def test_rejects_invalid_horizon_and_short_history(self) -> None:
        from shadow_bot.signals.vol_forecast import forecast_volatility

        for horizon in (0, -1, 1.5, True):
            with self.subTest(horizon=horizon):
                with self.assertRaises(ValueError):
                    forecast_volatility(self.history, horizon=horizon)

        with self.assertRaises(ValueError):
            forecast_volatility(self.history.iloc[:99])

    def test_surfaces_fit_failure_and_nonconvergence(self) -> None:
        from shadow_bot.signals.vol_forecast import forecast_volatility

        with patch("arch.arch_model") as arch_model:
            arch_model.return_value.fit.side_effect = RuntimeError("optimizer failed")
            with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
                forecast_volatility(self.history)

        with patch("arch.arch_model") as arch_model:
            arch_model.return_value.fit.return_value = self._fit_result(
                [[4.0]],
                convergence_flag=1,
            )
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                forecast_volatility(self.history)

    def test_rejects_invalid_forecast_variance(self) -> None:
        from shadow_bot.signals.vol_forecast import forecast_volatility

        invalid_outputs = [
            [[4.0]],
            [[4.0, np.nan]],
            [[4.0, 0.0]],
            [[4.0, -1.0]],
        ]
        for values in invalid_outputs:
            with self.subTest(values=values):
                with patch("arch.arch_model") as arch_model:
                    arch_model.return_value.fit.return_value = self._fit_result(values)
                    with self.assertRaises(ValueError):
                        forecast_volatility(self.history, horizon=2)


if __name__ == "__main__":
    unittest.main()
