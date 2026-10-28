# Kronos Forecast Module Design

## Purpose and scope

Add an offline probabilistic OHLCV forecasting module for the next Shadow
Timemachine stage. It will use the pretrained `NeoQuasar/Kronos-small` model
and `NeoQuasar/Kronos-Tokenizer-base`, forecast from no more than the most
recent 512 candles, retain at least 20 sampled paths, and expose their
cross-sample dispersion as a crude uncertainty proxy.

This task does not connect forecasts to `ShadowStrategy`, live execution,
Nautilus, the dashboard, or any trading decision. The historical comparison is
a demonstration of model output versus subsequent observed candles; it is not
a profitability or generalization claim.

## Upstream dependency and model loading

- Include `shiyu-coder/Kronos` as a Git submodule at
  `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (MIT license).
- Keep Kronos-only Python dependencies in `requirements-kronos.txt` so users
  who do not run forecasting do not acquire the additional model dependencies.
  Pin the upstream runtime packages `einops==0.8.1`,
  `huggingface_hub==0.33.1`, `safetensors==0.6.2`, and `tqdm==4.67.1`, and
  require `torch>=2.0.0`. Reuse the project's existing NumPy and pandas
  dependencies rather than adopting the upstream requirements file's older
  pandas pin.
- Load model code and weights only when forecasting is requested. Load
  `KronosTokenizer` from `NeoQuasar/Kronos-Tokenizer-base` and `Kronos` from
  `NeoQuasar/Kronos-small`; use `KronosPredictor` with `max_context=512`.
- Allow an explicit device override and otherwise use the predictor's default
  device selection. Hugging Face model files are downloaded and cached on first
  use. Do not silently substitute another model, tokenizer, or forecast path
  when loading or inference fails.
- Update setup documentation with submodule initialization and the optional
  requirements installation command.

## Forecast interface and data contract

Add `shadow_bot/signals/kronos_forecast.py` with a public
`forecast_paths(history, pred_len=12, sample_count=20, ...)` function.

Input is a pandas DataFrame with:

- `ts`: strictly increasing, unique epoch-millisecond candle timestamps.
- `open`, `high`, `low`, `close`, and `volume`: finite numeric OHLCV columns.

Require at least two rows and a positive `pred_len`; reject `sample_count`
below 20. Use no more than the latest 512 rows as model context. Convert
timestamps to UTC datetimes and infer the future cadence from the input
timestamps; do not read future rows as model input.

Return two pandas DataFrames:

1. A long-form paths frame with `sample_id`, `step`, `ts`, and forecasted
   `open`, `high`, `low`, `close`, and `volume` columns. It contains one row
   per sampled path and forecast step.
2. A per-step spread frame with the timestamp, close median, 10th and 90th
   percentiles, and standard deviation across sampled paths. This is descriptive
   sample dispersion only; it is not a calibrated prediction interval or a
   confidence score.

The documented Kronos `predict` API averages its requested `sample_count`
paths into one DataFrame. To retain individual paths through the public API,
invoke it `sample_count` times with `sample_count=1`, once per path. Do not
reimplement or monkey-patch Kronos internals.

## Validation

Add deterministic unit tests with an injected fake predictor. Cover input
validation, truncation to the last 512 rows, future timestamp generation,
minimum sample count, path/spread output shapes, and surfaced predictor errors.
These tests must not download weights or require a GPU.

Add an offline research validation command that:

- Reads the fetched CSV (`data/BTCUSDT_1m.csv` by default).
- Uses the latest origin with a complete future horizon, then two earlier
  origins exactly 1,440 rows apart. For the validated 1-minute dataset, this
  yields three origins spaced one day apart.
- At each origin, passes only candles at or before that origin to the module;
  it compares the forecast median and spread for the next 12 bars with those
  actual bars.
- Uses the default 20 sampled paths and prints the real forecast-versus-actual
  values for each origin. This small-sample comparison is illustrative only.

Report actual dependency installation output, command output, and failures.
Do not claim the model or validation works unless they are executed. Do not
infer strategy profitability from these three historical examples.

## Error behavior and safety boundary

Invalid timestamps, malformed OHLCV values, inadequate context, and invalid
forecast parameters raise explicit errors. Missing Kronos source or Python
dependencies, model download failures, and inference failures must be surfaced
without a synthetic fallback or success-shaped result.

All work in this stage is research-only. No API credentials, exchange
connectivity, order placement, or automatic strategy changes are in scope.
