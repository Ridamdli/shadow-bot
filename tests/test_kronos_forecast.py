import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from shadow_bot.signals import kronos_forecast


def make_history(rows: int = 512) -> pd.DataFrame:
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


class DeterministicPredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        *,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float,
        top_p: float,
        sample_count: int,
        verbose: bool,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "df": df.copy(),
                "x_timestamp": x_timestamp.copy(),
                "y_timestamp": y_timestamp.copy(),
                "pred_len": pred_len,
                "sample_count": sample_count,
            },
        )
        path_offset = len(self.calls) - 1
        close = 101.0 + path_offset
        return pd.DataFrame(
            {
                "open": np.full(pred_len, close - 0.5),
                "high": np.full(pred_len, close + 1),
                "low": np.full(pred_len, close - 1),
                "close": np.full(pred_len, close),
                "volume": np.full(pred_len, 10.0),
            },
            index=y_timestamp,
        )


class KronosForecastTests(unittest.TestCase):
    def test_returns_paths_and_close_spread(self) -> None:
        predictor = DeterministicPredictor()
        with patch.object(kronos_forecast, "_load_predictor", return_value=predictor):
            paths, spread = kronos_forecast.forecast_paths(make_history())

        self.assertEqual(len(paths), 20 * 12)
        self.assertEqual(
            list(paths.columns),
            ["sample_id", "step", "ts", "open", "high", "low", "close", "volume"],
        )
        self.assertEqual(
            list(spread.columns),
            ["ts", "step", "close_median", "close_p10", "close_p90", "close_std"],
        )
        self.assertEqual(paths["sample_id"].nunique(), 20)
        self.assertEqual(paths["step"].nunique(), 12)
        self.assertEqual(len(spread), 12)
        self.assertEqual(len(predictor.calls), 20)
        self.assertTrue(all(call["sample_count"] == 1 for call in predictor.calls))

        expected_close = 101.0 + np.arange(20)
        self.assertTrue(np.allclose(spread["close_median"], np.median(expected_close)))
        self.assertTrue(np.allclose(spread["close_p10"], np.percentile(expected_close, 10)))
        self.assertTrue(np.allclose(spread["close_p90"], np.percentile(expected_close, 90)))
        self.assertTrue(np.allclose(spread["close_std"], np.std(expected_close)))

    def test_uses_only_latest_512_rows(self) -> None:
        predictor = DeterministicPredictor()
        history = make_history(600)
        with patch.object(kronos_forecast, "_load_predictor", return_value=predictor):
            kronos_forecast.forecast_paths(history, pred_len=2, sample_count=20)

        context = predictor.calls[0]["df"]
        self.assertEqual(len(context), 512)
        pd.testing.assert_frame_equal(
            context.reset_index(drop=True),
            history.loc[:, ["open", "high", "low", "close", "volume"]]
            .tail(512)
            .reset_index(drop=True),
        )
        self.assertEqual(
            predictor.calls[0]["x_timestamp"].iloc[0],
            pd.to_datetime(history["ts"].iloc[-512], unit="ms", utc=True),
        )

    def test_generates_future_timestamps_from_input_cadence(self) -> None:
        predictor = DeterministicPredictor()
        history = make_history()
        with patch.object(kronos_forecast, "_load_predictor", return_value=predictor):
            paths, spread = kronos_forecast.forecast_paths(
                history,
                pred_len=3,
                sample_count=20,
            )

        expected = pd.to_datetime(
            history["ts"].tail(1).iloc[0] + np.arange(1, 4) * 60_000,
            unit="ms",
            utc=True,
        )
        self.assertEqual(
            list(pd.DatetimeIndex(spread["ts"].drop_duplicates())),
            list(expected),
        )
        self.assertEqual(paths["ts"].nunique(), 3)

    def test_rejects_invalid_history_and_forecast_parameters(self) -> None:
        history = make_history(3)
        invalid_cases: list[tuple[pd.DataFrame, dict[str, int], str]] = [
            (history.drop(columns="ts"), {}, "ts"),
            (history.assign(ts=["bad", "timestamp", "values"]), {}, "ts"),
            (history.iloc[::-1].reset_index(drop=True), {}, "increasing"),
            (history.assign(ts=[history.ts.iloc[0], history.ts.iloc[0], history.ts.iloc[2]]), {}, "unique"),
            (history.assign(close=[1.0, np.nan, 3.0]), {}, "finite"),
            (history.assign(volume=[1.0, np.inf, 3.0]), {}, "finite"),
            (history.iloc[:1], {}, "two"),
            (history, {"pred_len": 0}, "pred_len"),
            (history, {"sample_count": 19}, "sample_count"),
        ]
        predictor = DeterministicPredictor()
        with patch.object(kronos_forecast, "_load_predictor", return_value=predictor):
            for frame, kwargs, message in invalid_cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        kronos_forecast.forecast_paths(frame, **kwargs)

        self.assertEqual(predictor.calls, [])

    def test_surfaces_predictor_and_missing_dependency_errors(self) -> None:
        history = make_history(2)
        with patch.object(
            kronos_forecast,
            "_load_predictor",
            side_effect=ImportError("Kronos dependency unavailable"),
        ):
            with self.assertRaisesRegex(ImportError, "Kronos dependency unavailable"):
                kronos_forecast.forecast_paths(history)

        class FailingPredictor:
            def predict(self, **kwargs: object) -> pd.DataFrame:
                raise RuntimeError("Kronos inference failed")

        with patch.object(kronos_forecast, "_load_predictor", return_value=FailingPredictor()):
            with self.assertRaisesRegex(RuntimeError, "Kronos inference failed"):
                kronos_forecast.forecast_paths(history)


if __name__ == "__main__":
    unittest.main()
