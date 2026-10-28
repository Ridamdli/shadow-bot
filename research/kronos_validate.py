from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_bot.signals.kronos_forecast import forecast_paths

DATA = Path(__file__).resolve().parent.parent / "data" / "BTCUSDT_1m.csv"
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_MINUTES_PER_DAY = 1_440


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")
    required = ("ts", *_OHLCV_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {', '.join(missing)}")
    if len(frame) < 2:
        raise ValueError("input must contain at least two rows")

    try:
        timestamps = pd.to_numeric(frame["ts"], errors="raise").to_numpy(dtype=np.float64)
        values = frame.loc[:, list(_OHLCV_COLUMNS)].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps and OHLCV values must be numeric") from exc
    if not np.isfinite(timestamps).all() or not np.equal(timestamps, np.floor(timestamps)).all():
        raise ValueError("ts must contain finite integer epoch-millisecond timestamps")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("OHLCV values must be finite numbers")

    intervals = np.diff(timestamps.astype(np.int64))
    if (intervals != 60_000).any():
        raise ValueError("input must have continuous 1-minute timestamps without gaps")

    validated = frame.copy()
    validated["ts"] = timestamps.astype(np.int64)
    validated.loc[:, list(_OHLCV_COLUMNS)] = values
    return validated


def select_origins(frame: pd.DataFrame, horizon: int = 12) -> list[int]:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    latest = len(frame) - horizon - 1
    origins = [latest - _MINUTES_PER_DAY * offset for offset in range(3)]
    if origins[-1] < 1:
        raise ValueError("input must contain enough rows for three origins and complete horizons")
    return origins


def run_validation(
    frame: pd.DataFrame,
    horizon: int = 12,
    sample_count: int = 20,
) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or sample_count < 20:
        raise ValueError("sample_count must be an integer of at least 20")
    history = _validate_frame(frame)
    origins = select_origins(history, horizon)

    for origin in origins:
        past = history.iloc[: origin + 1]
        actual = history.iloc[origin + 1 : origin + horizon + 1]
        paths, spread = forecast_paths(past, pred_len=horizon, sample_count=sample_count)

        if not isinstance(paths, pd.DataFrame) or len(paths) != sample_count * horizon:
            raise ValueError("forecast paths do not match the requested sample count and horizon")
        if "sample_id" not in paths or paths["sample_id"].nunique() != sample_count:
            raise ValueError("forecast paths do not contain the requested number of samples")
        spread_columns = {"ts", "close_median", "close_p10", "close_p90", "close_std"}
        if not isinstance(spread, pd.DataFrame) or not spread_columns.issubset(spread.columns):
            raise ValueError("forecast spread is missing required columns")
        if len(spread) != horizon or len(actual) != horizon:
            raise ValueError("forecast and actual data must contain the complete horizon")

        actual_timestamps = pd.to_datetime(actual["ts"], unit="ms", utc=True).reset_index(drop=True)
        forecast_timestamps = pd.to_datetime(spread["ts"], utc=True).reset_index(drop=True)
        if not forecast_timestamps.equals(actual_timestamps):
            raise ValueError("forecast timestamps do not align with the following actual candles")

        origin_ts = pd.to_datetime(int(past["ts"].iloc[-1]), unit="ms", utc=True)
        print(f"origin={origin_ts.isoformat()} sample_count={sample_count} horizon={horizon}")
        for step, (_, actual_row) in enumerate(actual.iterrows(), start=1):
            forecast_row = spread.iloc[step - 1]
            actual_ts = actual_timestamps.iloc[step - 1]
            print(
                f"step={step} ts={actual_ts.isoformat()} "
                f"forecast_close_median={forecast_row['close_median']:.8g} "
                f"close_p10={forecast_row['close_p10']:.8g} "
                f"close_p90={forecast_row['close_p90']:.8g} "
                f"close_std={forecast_row['close_std']:.8g} "
                f"actual_open={actual_row['open']:.8g} "
                f"actual_high={actual_row['high']:.8g} "
                f"actual_low={actual_row['low']:.8g} "
                f"actual_close={actual_row['close']:.8g} "
                f"actual_volume={actual_row['volume']:.8g}",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare historical Kronos forecasts with actual candles.")
    parser.add_argument("--csv", default=str(DATA))
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    run_validation(pd.read_csv(args.csv), horizon=args.horizon, sample_count=args.sample_count)


if __name__ == "__main__":
    main()
