# GARCH Volatility Forecaster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional GARCH(1,1) module and a leakage-free daily walk-forward validator for BTCUSDT 1-minute volatility forecasts.

**Architecture:** `shadow_bot/signals/vol_forecast.py` will validate close history, lazily load `arch`, fit GARCH(1,1), and return decimal-unit variance and volatility forecasts. A separate research CLI will prepare trailing 30-day windows, refit daily, and report held-out RMSE and QLIKE without changing trading behavior.

**Tech Stack:** Python, pandas, NumPy, `arch`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-28-garch-volatility-design.md`

## Global Constraints

- Use `arch.arch_model` with `mean="Constant"`, `vol="GARCH"`, `p=1`, `q=1`, `dist="normal"`, and `rescale=False`.
- Scale decimal log returns by 100 for fitting; divide forecast variance by 10,000 and volatility by 100 before returning.
- Use exact 1-minute epoch-millisecond timestamps and strictly positive finite close prices; require at least 100 candles and a positive integer horizon.
- Validation uses exactly 30 days / 43,200 candles of trailing history, refits daily, and forecasts 1,440 one-minute bars.
- `arch` stays an optional dependency pinned to the version actually installed and validated.
- No strategy, dashboard, live-runner, exchange, or order-execution changes.

## Review Focus

- Fractional, duplicate, non-monotonic, or gapped timestamps must be rejected; test each in `tests/test_vol_forecast.py::test_rejects_invalid_timestamps`.
- Zero, negative, NaN, or infinite close values must be rejected; test in `tests/test_vol_forecast.py::test_rejects_nonpositive_or_nonfinite_closes`.
- Boolean/non-integer/nonpositive horizons and fewer than 100 candles must be rejected; test in `tests/test_vol_forecast.py::test_rejects_invalid_horizon_and_short_history`.
- Fit exceptions and nonzero ARCH convergence flags must surface as failures; test in `tests/test_vol_forecast.py::test_surfaces_fit_failure_and_nonconvergence`.
- Wrong-length, non-finite, zero, or negative forecast variances must not produce successful output; test in `tests/test_vol_forecast.py::test_rejects_invalid_forecast_variance`.

---

### Task 1: Add the GARCH forecast function

**Files:**
- Create: `requirements-volatility.txt`
- Create: `shadow_bot/signals/vol_forecast.py`
- Test: `tests/test_vol_forecast.py`

**Interfaces:**
- Consumes: pandas DataFrame with `ts` epoch milliseconds and `close` prices.
- Produces: `forecast_volatility(history: pd.DataFrame, horizon: int = 1) -> pd.DataFrame` with `ts`, `variance`, and `volatility` columns.

- [ ] **Step 1: Install `arch` and record its actual version**

Run: `.venv-task1-20261028\Scripts\python.exe -m pip install arch`

If the installation fails, make one further honest attempt. If both attempts fail, stop and report both exact errors; do not substitute another volatility library.

Run: `.venv-task1-20261028\Scripts\python.exe -m pip show arch`

Expected: successful install and an exact version value to pin in `requirements-volatility.txt`.

- [ ] **Step 2: Write failing tests for the forecast contract**

In `tests/test_vol_forecast.py`, add `unittest` cases named:

`test_returns_decimal_variance_and_volatility_with_future_timestamps`,
`test_rejects_invalid_timestamps`,
`test_rejects_nonpositive_or_nonfinite_closes`,
`test_rejects_invalid_horizon_and_short_history`,
`test_surfaces_fit_failure_and_nonconvergence`, and
`test_rejects_invalid_forecast_variance`.

Mock `arch.arch_model` and its result; assert exact ARCH arguments, return units, one-minute timestamps, explicit failure on non-convergence, and rejection of malformed model output. Add one exact `arch==` requirement by copying the version following `Version:` from the `pip show arch` output in Step 1.

- [ ] **Step 3: Run the focused tests and verify they fail**

Run: `.venv-task1-20261028\Scripts\python.exe -m unittest discover -s tests -p test_vol_forecast.py -v`

Expected: FAIL because `shadow_bot.signals.vol_forecast` does not yet exist.

- [ ] **Step 4: Implement `forecast_volatility`**

In `shadow_bot/signals/vol_forecast.py`, validate the specified input contract, calculate `100 * diff(log(close))`, lazily import `arch.arch_model`, fit the specified model, check `convergence_flag == 0`, and forecast `horizon` steps with `reindex=False`. Divide variances by 10,000 and calculate volatility as their square root. Build future timestamps at 60,000 ms intervals and raise explicit errors for invalid data, fitting failures, non-convergence, or invalid output.

- [ ] **Step 5: Run the focused tests**

Run: `.venv-task1-20261028\Scripts\python.exe -m unittest discover -s tests -p test_vol_forecast.py -v`

Expected: all six forecast-function tests pass without running an actual optimizer fit.

- [ ] **Step 6: Commit the forecast function**

```powershell
git add requirements-volatility.txt shadow_bot/signals/vol_forecast.py tests/test_vol_forecast.py
git commit -m "feat: add GARCH volatility forecast module" -m "Keep volatility forecasting optional and return one-minute forecasts in decimal log-return units." -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Add leakage-free walk-forward validation

**Files:**
- Create: `research/vol_forecast_validate.py`
- Test: `tests/test_vol_forecast_validate.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: validated OHLCV DataFrame with continuous 1-minute epoch-millisecond timestamps and `close` prices.
- Uses: `forecast_volatility(history, horizon=1440)` from Task 1, with exactly the 43,200 candles immediately before each test day.
- Produces: `run_validation(frame: pd.DataFrame) -> None`, which prints one row per daily fold and an overall held-out summary; CLI defaults to `data/BTCUSDT_1m.csv`.

- [ ] **Step 1: Write failing validator tests**

In `tests/test_vol_forecast_validate.py`, add tests named:

`test_rejects_missing_columns_and_invalid_or_gapped_candles`,
`test_uses_exact_trailing_window_and_excludes_test_day`,
`test_scores_daily_and_overall_metrics`,
and `test_help_runs_without_fitting_model`.

Use synthetic continuous 1-minute candles and patch `forecast_volatility`; assert each fit receives exactly the preceding 43,200 candles, ends before the test-day start, requests a 1,440-step forecast, and produces aligned daily and overall RMSE/QLIKE metrics.

- [ ] **Step 2: Run the focused validator tests and verify they fail**

Run: `.venv-task1-20261028\Scripts\python.exe -m unittest discover -s tests -p test_vol_forecast_validate.py -v`

Expected: FAIL because `research.vol_forecast_validate` does not yet exist.

- [ ] **Step 3: Implement CSV validation, rolling folds, metrics, and CLI**

In `research/vol_forecast_validate.py`, validate required OHLCV values and exact 1-minute spacing. For every complete test day after the initial 30 days, pass only the immediately preceding 43,200 candles to `forecast_volatility(..., horizon=1440)`. Calculate actual log returns using the prior training close for the first test minute. Print per-fold and overall bar counts, RMSE of predicted volatility against absolute realized return, and QLIKE of forecast variance against squared return. Provide `main()` with `--csv`, defaulting to `data/BTCUSDT_1m.csv`; direct script execution must resolve the repository root before importing `shadow_bot`.

- [ ] **Step 4: Run focused validator tests**

Run: `.venv-task1-20261028\Scripts\python.exe -m unittest discover -s tests -p test_vol_forecast_validate.py -v`

Expected: all four validator tests pass; the mocked forecast confirms no look-ahead.

- [ ] **Step 5: Document optional setup and validation**

Update `README.md` to show `python -m pip install -r requirements-volatility.txt` and `python research/vol_forecast_validate.py --csv data/BTCUSDT_1m.csv`. State that the forecast module and validation are research-only and are not connected to trading.

- [ ] **Step 6: Run the real walk-forward validation and capture exact output**

Run: `.venv-task1-20261028\Scripts\python.exe research\vol_forecast_validate.py --csv data\BTCUSDT_1m.csv`

Expected: exit code 0 with per-fold and overall held-out metrics, training/test date ranges, fold count, and forecast-bar count. Record stdout and the exit code verbatim. If execution fails, report the exact failure; do not present invented metrics.

- [ ] **Step 7: Run the complete test suite and static checks**

Run: `.venv-task1-20261028\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all repository tests pass. Also run `.venv-task1-20261028\Scripts\python.exe -m py_compile shadow_bot/signals/vol_forecast.py research/vol_forecast_validate.py tests/test_vol_forecast.py tests/test_vol_forecast_validate.py` and `git diff --check`; both must exit 0.

- [ ] **Step 8: Commit the validator and documentation**

```powershell
git add research/vol_forecast_validate.py tests/test_vol_forecast_validate.py README.md
git commit -m "feat: validate rolling GARCH forecasts" -m "Measure held-out daily volatility forecasts without connecting them to trading." -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Review and submit the Task 4 pull request

**Files:**
- Review: all files in this branch relative to `main`.

**Interfaces:**
- Consumes: committed implementation and actual validation logs from Tasks 1 and 2.
- Produces: one PR against `main` containing the installed `arch` version, exact test output, and exact real-validation stdout/exit code.

- [ ] **Step 1: Review branch scope and run final checks**

Confirm no strategy, live runner, dashboard, credentials, or unrelated files are included. Run the full tests and `git diff --check` against the final tree.

- [ ] **Step 2: Push the feature branch and create the PR**

Push `feature/task4-garch-volatility` and open one PR against `main`. Include the actual package version, all relevant command outputs, held-out date span, fold and forecast-bar counts, metrics, and the explicit statement that these are volatility forecast errors, not profitability results.

- [ ] **Step 3: Stop for review**

Do not begin the next signal module or wire this forecast into execution until this PR is reviewed and merged.
