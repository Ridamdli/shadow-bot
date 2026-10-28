from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_HISTORY = 100
_MINUTE_MS = 60_000
_MAX_INT64 = np.iinfo(np.int64).max


def _validated_history(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(history, pd.DataFrame):
        raise ValueError("history must be a pandas DataFrame")
    missing = [column for column in ("ts", "close") if column not in history.columns]
    if missing:
        raise ValueError(f"history is missing required columns: {', '.join(missing)}")
    if len(history) < _MIN_HISTORY:
        raise ValueError(f"history must contain at least {_MIN_HISTORY} candles")
    if pd.api.types.is_bool_dtype(history["ts"].dtype):
        raise ValueError("ts must contain epoch-millisecond timestamps")

    try:
        timestamps = pd.to_numeric(history["ts"], errors="raise").to_numpy(dtype=np.float64)
        closes = pd.to_numeric(history["close"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ts and close must contain numeric values") from exc

    if (
        not np.isfinite(timestamps).all()
        or not np.equal(timestamps, np.floor(timestamps)).all()
        or (timestamps < np.iinfo(np.int64).min).any()
        or (timestamps > _MAX_INT64).any()
    ):
        raise ValueError("ts must contain finite integer epoch-millisecond timestamps")
    timestamps = timestamps.astype(np.int64)
    if not np.all(np.diff(timestamps) == _MINUTE_MS):
        raise ValueError("ts timestamps must be strictly increasing at 1-minute intervals")

    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("close prices must be finite positive numbers")

    return timestamps, closes


def forecast_volatility(history: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    timestamps, closes = _validated_history(history)
    if timestamps[-1] > _MAX_INT64 - horizon * _MINUTE_MS:
        raise ValueError("forecast timestamps exceed the epoch-millisecond range")
    scaled_returns = np.diff(np.log(closes)) * 100

    from arch import arch_model

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    fitted = model.fit(disp="off")
    if not hasattr(fitted, "convergence_flag"):
        raise RuntimeError("GARCH fit result does not report convergence status")
    if fitted.convergence_flag != 0:
        raise RuntimeError(f"GARCH optimizer did not converge (flag={fitted.convergence_flag})")

    variance_frame = fitted.forecast(horizon=int(horizon), reindex=False).variance
    if not isinstance(variance_frame, pd.DataFrame) or variance_frame.empty:
        raise ValueError("GARCH forecast must return a non-empty variance DataFrame")
    if len(variance_frame.columns) != horizon:
        raise ValueError("GARCH forecast returned the wrong number of horizons")
    try:
        scaled_variance = variance_frame.iloc[-1].to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("GARCH forecast variances must be numeric") from exc
    if not np.isfinite(scaled_variance).all() or (scaled_variance <= 0).any():
        raise ValueError("GARCH forecast variances must be finite positive numbers")

    variance = scaled_variance / 10_000
    forecast_timestamps = timestamps[-1] + np.arange(1, horizon + 1, dtype=np.int64) * _MINUTE_MS
    return pd.DataFrame(
        {
            "ts": forecast_timestamps,
            "variance": variance,
            "volatility": np.sqrt(variance),
        }
    )
