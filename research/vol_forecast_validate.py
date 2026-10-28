from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shadow_bot.signals.vol_forecast import forecast_volatility

DATA = REPOSITORY_ROOT / "data" / "BTCUSDT_1m.csv"
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_MINUTE_MS = 60_000
_BARS_PER_DAY = 1_440
_WINDOW_DAYS = 30
_WINDOW_BARS = _WINDOW_DAYS * _BARS_PER_DAY
_MAX_INT64 = np.iinfo(np.int64).max


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")
    required = ("ts", *_OHLCV_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {', '.join(missing)}")
    if len(frame) < 2:
        raise ValueError("input must contain at least two rows")
    if pd.api.types.is_bool_dtype(frame["ts"].dtype):
        raise ValueError("ts must contain epoch-millisecond timestamps")

    try:
        timestamps = pd.to_numeric(frame["ts"], errors="raise").to_numpy(dtype=np.float64)
        values = frame.loc[:, list(_OHLCV_COLUMNS)].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamps and OHLCV values must be numeric") from exc

    if (
        not np.isfinite(timestamps).all()
        or not np.equal(timestamps, np.floor(timestamps)).all()
        or (timestamps < np.iinfo(np.int64).min).any()
        or (timestamps > _MAX_INT64).any()
    ):
        raise ValueError("ts must contain finite integer epoch-millisecond timestamps")
    timestamps = timestamps.astype(np.int64)
    if not np.all(np.diff(timestamps) == _MINUTE_MS):
        raise ValueError("input must have continuous 1-minute timestamps without gaps")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("OHLCV values must be finite numbers")
    if (values["close"].to_numpy(dtype=np.float64) <= 0).any():
        raise ValueError("close prices must be positive")

    validated = frame.copy()
    validated["ts"] = timestamps
    validated.loc[:, list(_OHLCV_COLUMNS)] = values
    return validated


def _metrics(forecast_variance: np.ndarray, forecast_volatility: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((forecast_volatility - np.abs(realized)) ** 2)))
    qlike = float(np.mean(np.log(forecast_variance) + realized**2 / forecast_variance))
    return rmse, qlike


def _timestamp(timestamp_ms: int) -> str:
    return pd.to_datetime(timestamp_ms, unit="ms", utc=True).isoformat()


def run_validation(frame: pd.DataFrame) -> None:
    history = _validate_frame(frame)
    if len(history) < _WINDOW_BARS + _BARS_PER_DAY:
        raise ValueError("input must contain 30 training days and at least one complete test day")

    timestamps = history["ts"].to_numpy(dtype=np.int64)
    closes = history["close"].to_numpy(dtype=np.float64)
    forecasts_variance: list[np.ndarray] = []
    forecasts_volatility: list[np.ndarray] = []
    realized_returns: list[np.ndarray] = []
    fold_count = 0

    for test_start in range(_WINDOW_BARS, len(history) - _BARS_PER_DAY + 1, _BARS_PER_DAY):
        test_end = test_start + _BARS_PER_DAY
        training = history.iloc[test_start - _WINDOW_BARS : test_start]
        actual_timestamps = timestamps[test_start:test_end]
        forecast = forecast_volatility(training, horizon=_BARS_PER_DAY)

        required_forecast = ("ts", "variance", "volatility")
        if not isinstance(forecast, pd.DataFrame):
            raise ValueError("volatility forecaster must return a pandas DataFrame")
        missing = [column for column in required_forecast if column not in forecast.columns]
        if missing:
            raise ValueError(f"volatility forecast is missing columns: {', '.join(missing)}")
        if len(forecast) != _BARS_PER_DAY:
            raise ValueError("volatility forecast must contain one row per test-day bar")

        try:
            forecast_timestamps = pd.to_numeric(forecast["ts"], errors="raise").to_numpy(dtype=np.float64)
            variance = pd.to_numeric(forecast["variance"], errors="raise").to_numpy(dtype=np.float64)
            volatility = pd.to_numeric(forecast["volatility"], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("forecast timestamps, variances, and volatilities must be numeric") from exc
        if (
            not np.isfinite(forecast_timestamps).all()
            or not np.equal(forecast_timestamps, np.floor(forecast_timestamps)).all()
            or not np.array_equal(forecast_timestamps.astype(np.int64), actual_timestamps)
        ):
            raise ValueError("forecast timestamps must align with the following actual candles")
        if (
            not np.isfinite(variance).all()
            or not np.isfinite(volatility).all()
            or (variance <= 0).any()
            or (volatility <= 0).any()
        ):
            raise ValueError("forecast variances and volatilities must be finite positive numbers")

        realized = np.diff(np.log(np.concatenate((closes[test_start - 1 : test_start], closes[test_start:test_end]))))
        rmse, qlike = _metrics(variance, volatility, realized)
        fold_count += 1
        forecasts_variance.append(variance)
        forecasts_volatility.append(volatility)
        realized_returns.append(realized)
        print(
            f"fold={fold_count} "
            f"train_start={_timestamp(int(timestamps[test_start - _WINDOW_BARS]))} "
            f"train_end={_timestamp(int(timestamps[test_start - 1]))} "
            f"test_start={_timestamp(int(actual_timestamps[0]))} "
            f"test_end={_timestamp(int(actual_timestamps[-1]))} "
            f"forecast_bars={_BARS_PER_DAY} rmse={rmse:.8g} qlike={qlike:.8g}"
        )

    all_variance = np.concatenate(forecasts_variance)
    all_volatility = np.concatenate(forecasts_volatility)
    all_realized = np.concatenate(realized_returns)
    overall_rmse, overall_qlike = _metrics(all_variance, all_volatility, all_realized)
    print(
        f"overall training_window_days={_WINDOW_DAYS} "
        f"heldout_start={_timestamp(int(timestamps[_WINDOW_BARS]))} "
        f"heldout_end={_timestamp(int(timestamps[_WINDOW_BARS + fold_count * _BARS_PER_DAY - 1]))} "
        f"daily_folds={fold_count} forecast_bars={len(all_realized)} "
        f"rmse={overall_rmse:.8g} qlike={overall_qlike:.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a daily walk-forward GARCH volatility forecast validation.")
    parser.add_argument("--csv", default=str(DATA))
    args = parser.parse_args()
    run_validation(pd.read_csv(args.csv))


if __name__ == "__main__":
    main()
