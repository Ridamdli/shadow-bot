from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from model import KronosPredictor

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_PATH_COLUMNS = ("sample_id", "step", "ts", *_OHLCV_COLUMNS)
_SPREAD_COLUMNS = ("ts", "step", "close_median", "close_p10", "close_p90", "close_std")
_MIN_SAMPLE_COUNT = 20
_MAX_CONTEXT = 512


def _load_predictor(device: str | None) -> KronosPredictor:
    kronos_root = Path(__file__).resolve().parents[2] / "vendor" / "Kronos"
    if not (kronos_root / "model" / "__init__.py").is_file():
        raise ImportError(f"Kronos source is missing from {kronos_root}")

    kronos_path = str(kronos_root)
    added_path = kronos_path not in sys.path
    if added_path:
        sys.path.insert(0, kronos_path)

    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    finally:
        if added_path:
            sys.path.remove(kronos_path)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    model.eval()

    if device is None:
        return KronosPredictor(model, tokenizer, max_context=_MAX_CONTEXT)
    return KronosPredictor(model, tokenizer, device=device, max_context=_MAX_CONTEXT)


def _validated_history(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    if not isinstance(history, pd.DataFrame):
        raise ValueError("history must be a pandas DataFrame")
    if "ts" not in history.columns:
        raise ValueError("history must include a ts column")
    missing = [column for column in _OHLCV_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"history is missing OHLCV columns: {', '.join(missing)}")
    if len(history) < 2:
        raise ValueError("history must contain at least two rows")

    try:
        ts_values = pd.to_numeric(history["ts"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ts must contain epoch-millisecond timestamps") from exc
    if not np.isfinite(ts_values).all() or not np.equal(ts_values, np.floor(ts_values)).all():
        raise ValueError("ts must contain finite integer epoch-millisecond timestamps")

    timestamps = pd.DatetimeIndex(pd.to_datetime(ts_values.astype(np.int64), unit="ms", utc=True))
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("ts timestamps must be strictly increasing and unique")

    try:
        values = history.loc[:, _OHLCV_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLCV values must be finite numbers") from exc
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError("OHLCV values must be finite numbers")

    context = values.tail(_MAX_CONTEXT).reset_index(drop=True)
    return context, timestamps[-len(context) :]


def forecast_paths(
    history: pd.DataFrame,
    pred_len: int = 12,
    sample_count: int = 20,
    *,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(pred_len, bool) or not isinstance(pred_len, (int, np.integer)) or pred_len <= 0:
        raise ValueError("pred_len must be a positive integer")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, (int, np.integer))
        or sample_count < _MIN_SAMPLE_COUNT
    ):
        raise ValueError(f"sample_count must be an integer of at least {_MIN_SAMPLE_COUNT}")

    context, context_timestamps = _validated_history(history)
    deltas_ms = np.diff(context_timestamps.asi8) // 1_000_000
    cadence_ms = int(np.median(deltas_ms))
    if cadence_ms <= 0:
        raise ValueError("history timestamps must have a positive cadence")

    future_timestamps = pd.DatetimeIndex(
        context_timestamps[-1]
        + pd.to_timedelta(cadence_ms * np.arange(1, pred_len + 1), unit="ms"),
    )
    predictor = _load_predictor(device)
    x_timestamp = pd.Series(context_timestamps)
    y_timestamp = pd.Series(future_timestamps)
    path_frames: list[pd.DataFrame] = []

    for sample_id in range(sample_count):
        prediction = predictor.predict(
            df=context,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        if not isinstance(prediction, pd.DataFrame) or len(prediction) != pred_len:
            raise ValueError("Kronos predictor must return one DataFrame row per forecast step")
        missing = [column for column in _OHLCV_COLUMNS if column not in prediction.columns]
        if missing:
            raise ValueError(f"Kronos prediction is missing OHLCV columns: {', '.join(missing)}")

        forecast_values = prediction.loc[:, _OHLCV_COLUMNS].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(forecast_values.to_numpy(dtype=np.float64)).all():
            raise ValueError("Kronos prediction contains non-finite OHLCV values")

        path_frame = forecast_values.reset_index(drop=True)
        path_frame.insert(0, "ts", future_timestamps)
        path_frame.insert(0, "step", np.arange(1, pred_len + 1))
        path_frame.insert(0, "sample_id", sample_id)
        path_frames.append(path_frame.loc[:, _PATH_COLUMNS])

    paths = pd.concat(path_frames, ignore_index=True)
    close_samples = paths.pivot(index="sample_id", columns="step", values="close").to_numpy()
    spread = pd.DataFrame(
        {
            "ts": future_timestamps,
            "step": np.arange(1, pred_len + 1),
            "close_median": np.median(close_samples, axis=0),
            "close_p10": np.percentile(close_samples, 10, axis=0),
            "close_p90": np.percentile(close_samples, 90, axis=0),
            "close_std": np.std(close_samples, axis=0),
        },
        columns=_SPREAD_COLUMNS,
    )
    return paths, spread
