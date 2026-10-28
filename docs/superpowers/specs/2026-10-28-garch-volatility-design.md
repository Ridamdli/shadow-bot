# GARCH Volatility Forecast Design

## Purpose and scope

Add a research-only next-period volatility forecaster using
`arch.arch_model` with a GARCH(1,1) specification. The module will operate on
BTCUSDT 1-minute close prices and forecast per-minute conditional variance and
volatility from close-to-close log returns.

Validate the model with a chronological walk-forward over the fetched
BTCUSDT 1-minute data. Use a 30-day trailing estimation window, refit once per
day, and forecast the following complete day. Report held-out forecast errors
against realized returns; this is not a trading-profitability claim.

This task does not wire forecasts into the strategy, live runner, dashboard,
or any exchange execution.

## Model and module interface

Add `shadow_bot/signals/vol_forecast.py` with a public
`forecast_volatility(history, horizon=1)` function.

Input is a pandas DataFrame with:

- `ts`: strictly increasing, unique epoch-millisecond timestamps with exact
  1-minute spacing.
- `close`: finite, strictly positive numeric close prices.

Require at least 100 input candles. Calculate close-to-close log returns,
scale returns by 100 for numerical stability while fitting, and use an explicit
normal-innovation GARCH(1,1) model (`mean="Constant"`, `vol="GARCH"`, `p=1`,
`q=1`, `dist="normal"`, `rescale=False`). The caller supplies the history
window; the forecasting module does not select or persist a training window.

Require a positive integer horizon. Return one row per forecast step with:

- `ts`: each future one-minute timestamp.
- `variance`: forecast conditional variance in squared decimal log-return
  units.
- `volatility`: square root of that variance in decimal log-return units per
  minute (not annualized).

Convert the ARCH forecast back from percent-return units by dividing variance
by 10,000 and volatility by 100. The function must reject invalid inputs and
must not return a forecast if fitting fails, the optimizer does not converge,
or output values are non-finite or non-positive. The optional `arch`
dependency is imported only when forecasting is requested so non-forecasting
parts of the repository remain usable without it.

## Walk-forward validation

Add `research/vol_forecast_validate.py`, defaulting to
`data/BTCUSDT_1m.csv`. Validate continuous, complete 1-minute OHLCV input
before fitting.

For each complete test day after the initial 30 days:

1. Select exactly the immediately preceding 30 days of candles; do not include
   any test-day candle in model fitting.
2. Fit GARCH(1,1) on the resulting log returns and forecast the next 1,440
   one-minute variances and volatilities.
3. Align those forecasts to the test-day returns, calculated using the prior
   training close for the first test minute.
4. Record daily forecast metrics and accumulate the held-out results.

Report the training window, held-out date span, daily fold count, held-out bar
count, and metrics for every daily fold and overall:

- RMSE between forecast volatility and the absolute realized log return. The
  absolute one-minute return is a noisy realized-volatility proxy.
- QLIKE loss, calculated as the mean of
  `log(forecast_variance) + realized_squared_return / forecast_variance`;
  lower is better.

No observations may be shuffled, and each fold must be scored only against
returns after its forecast origin. Do not infer strategy profitability from
these metrics.

## Dependency and documentation

Keep `arch` optional in a separate `requirements-volatility.txt`, following
the existing optional Kronos dependency pattern. Pin the version actually
installed and validated for this task. Update the README with the optional
installation command and the research validation command.

## Testing and error behavior

Add deterministic tests that mock ARCH fitting and cover:

- Input schema, timestamp cadence, positive closes, minimum history, and
  horizon validation.
- Decimal-unit conversion and future timestamp alignment.
- The validator's rolling-window boundaries and absence of look-ahead.
- Held-out metric calculations.
- Surfaced fitting failures, non-convergence, and malformed/non-finite
  forecasts.

Tests must not perform expensive ARCH optimization. Separately run the real
walk-forward validator against the fetched CSV and record its actual stdout,
exit code, installed `arch` version, and any errors in the PR description. Do
not claim successful execution or model quality without that run.

All work in this task is offline and research-only. No API credentials, network
exchange access, order placement, or automatic strategy changes are in scope.
