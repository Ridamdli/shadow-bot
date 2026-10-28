# Kronos Forecast Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a research-only Kronos module that generates and reports at least 20 sampled OHLCV forecast paths from up to 512 recent candles, with a historical forecast-versus-actual demonstration.

**Architecture:** Pin the upstream Kronos source as a Git submodule and keep its runtime packages optional. A lazy-loaded adapter validates candle history and calls Kronos once per requested path; a separate research command evaluates three leak-free historical origins without connecting forecasts to strategy execution.

**Tech Stack:** Python, pandas, PyTorch, Kronos, Hugging Face Hub, standard-library `unittest`.

**Spec:** [2026-10-28-kronos-forecast-design.md](../specs/2026-10-28-kronos-forecast-design.md)

## Global Constraints

- Use `NeoQuasar/Kronos-small` and `NeoQuasar/Kronos-Tokenizer-base`.
- Pin `shiyu-coder/Kronos` at `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
- Keep Kronos-only Python dependencies in `requirements-kronos.txt`.
- Use `max_context=512`; default `pred_len=12`; reject `sample_count < 20`.
- Call the public Kronos predictor with `sample_count=1` once for each retained path.
- Label cross-path spread as descriptive dispersion, not calibrated confidence.
- Do not connect this module to Nautilus, live trading, the dashboard, or strategy decisions.
- Surface dependency, model-download, validation, and inference failures; do not silently substitute results.
- Report only commands and model validations that actually ran, with their real output and exit status.
- If an installation or model API fails after two honest attempts, stop and report the exact blocker.

## Review Focus

- Missing, malformed, or non-finite timestamp/OHLCV inputs must fail before inference; test missing `ts`, invalid `ts`, and NaN/infinite values.
- Unsorted or duplicate timestamps must fail; test both ordering violations.
- Too-short history and invalid horizons must fail; test fewer than two rows and `pred_len <= 0`.
- Fewer than 20 samples must fail; test `sample_count=19`.
- Missing optional dependencies and predictor failures must surface; test a failing lazy loader and a fake predictor that raises.

---

### Task 1: Pin Kronos and document optional setup

**Files:**
- Create: `.gitmodules`
- Create: `vendor/Kronos` Git submodule at the pinned revision
- Create: `requirements-kronos.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: the upstream Kronos Git repository and its documented model/tokenizer identifiers.
- Produces: a reproducible local import root at `vendor/Kronos`, optional model dependencies, and setup instructions for submodule initialization.

- [ ] **Step 1: Add the upstream submodule**

Run:
```powershell
git submodule add https://github.com/shiyu-coder/Kronos.git vendor/Kronos
git -C vendor/Kronos checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
```

Expected: `.gitmodules` points to the upstream repository and `git -C vendor/Kronos rev-parse HEAD` prints the pinned SHA.

- [ ] **Step 2: Add optional Kronos requirements**

Create `requirements-kronos.txt` with `torch>=2.0.0`, `einops==0.8.1`, `huggingface_hub==0.33.1`, `safetensors==0.6.2`, and `tqdm==4.67.1`. Do not pin or downgrade NumPy/pandas here.

- [ ] **Step 3: Document setup and the optional boundary**

Update the README setup section with `git clone --recurse-submodules` and, for an existing clone, `git submodule update --init --recursive`, followed by `python -m pip install -r requirements-kronos.txt`. State that weights are fetched and cached on first forecast use and that forecasts are not connected to trading.

- [ ] **Step 4: Verify the dependency source and documentation**

Run:
```powershell
git -C vendor/Kronos rev-parse HEAD
git diff --check
```

Expected: the pinned SHA is printed and `git diff --check` exits 0.

- [ ] **Step 5: Commit the dependency setup**

```powershell
git add .gitmodules vendor/Kronos requirements-kronos.txt README.md
git commit -m "build: pin optional Kronos runtime"
```

### Task 2: Implement and unit-test sampled forecast paths

**Files:**
- Create: `shadow_bot/signals/__init__.py`
- Create: `shadow_bot/signals/kronos_forecast.py`
- Create: `tests/test_kronos_forecast.py`

**Interfaces:**
- Consumes: `vendor/Kronos/model` and the optional runtime packages from Task 1.
- Produces: `forecast_paths(history: pd.DataFrame, pred_len: int = 12, sample_count: int = 20, *, device: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`.

- [ ] **Step 1: Write the failing unit tests**

In `tests/test_kronos_forecast.py`, monkeypatch the private lazy predictor loader to return a deterministic fake predictor. Add tests for:
  - `test_returns_paths_and_close_spread` — 20 paths by 12 steps, stable path columns, and expected median/quantile/std values.
  - `test_uses_only_latest_512_rows` — fake predictor receives exactly the last 512 rows from longer history.
  - `test_generates_future_timestamps_from_input_cadence` — expected UTC timestamps advance by the observed interval.
  - `test_rejects_invalid_history_and_forecast_parameters` — missing/invalid `ts`, non-monotonic/duplicate timestamps, non-finite OHLCV, fewer than two rows, non-positive horizon, and sample count below 20 raise `ValueError`.
  - `test_surfaces_predictor_and_missing_dependency_errors` — inference and lazy-loader failures propagate unchanged.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m unittest discover -s tests -p "test_kronos_forecast.py" -v`

Expected: tests fail because `shadow_bot.signals.kronos_forecast` does not yet exist.

- [ ] **Step 3: Implement the forecast adapter**

Implement `_load_predictor(device: str | None) -> KronosPredictor` and the specified public `forecast_paths` signature. Resolve the repository root, temporarily prepend `vendor/Kronos` to `sys.path` to import `model`, and remove the path entry after import. Build `KronosPredictor` with `max_context=512`, passing `device` only when explicitly supplied. Validate input, clip context to the last 512 rows, create UTC `x_timestamp`/future `y_timestamp`, and request one path per predictor call using `sample_count=1`. Normalize each result into long-form OHLCV rows with `sample_id`, one-based `step`, and `ts`. Return spread columns `ts`, `step`, `close_median`, `close_p10`, `close_p90`, and `close_std` alongside the paths. Keep the loader private and patchable for tests.

- [ ] **Step 4: Run the unit tests and confirm they pass**

Run: `python -m unittest discover -s tests -p "test_kronos_forecast.py" -v`

Expected: all forecast adapter tests pass without loading model weights or requiring a GPU.

- [ ] **Step 5: Commit the forecast adapter**

```powershell
git add shadow_bot/signals tests/test_kronos_forecast.py
git commit -m "feat: add sampled Kronos forecasts"
```

### Task 3: Add leak-free historical validation command

**Files:**
- Create: `research/kronos_validate.py`
- Create: `tests/test_kronos_validate.py`

**Interfaces:**
- Consumes: `forecast_paths` from Task 2 and a timestamped OHLCV CSV.
- Produces: `python research/kronos_validate.py --csv data/BTCUSDT_1m.csv --horizon 12 --sample-count 20`, printing forecast median/spread beside actual values at three historical origins.

- [ ] **Step 1: Write failing tests for historical origin selection**

Add tests in `tests/test_kronos_validate.py`:
  - `test_select_origins_returns_latest_and_two_daily_origins` — assert indices are `N - horizon - 1`, then 1,440 and 2,880 rows earlier, each with a complete future horizon.
  - `test_run_validation_passes_history_only_through_origin` — patch `forecast_paths`, run against synthetic continuous 1-minute data, and assert every call ends at its origin timestamp with no future rows in its history.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `python -m unittest discover -s tests -p "test_kronos_validate.py" -v`

Expected: fail because the validation module and origin selector do not exist.

- [ ] **Step 3: Implement the offline validation command**

Implement `select_origins(frame: pd.DataFrame, horizon: int = 12) -> list[int]` and `run_validation(frame: pd.DataFrame, horizon: int = 12, sample_count: int = 20) -> None` in `research/kronos_validate.py`. Load the CSV, validate its required columns and 1-minute continuity, select the three origins, and for each call `forecast_paths` with only rows through that origin. Print each next-horizon forecast median and spread beside the corresponding actual OHLCV values. Expose `--csv`, `--horizon`, and `--sample-count`; defaults are the specified BTCUSDT path, 12, and 20.

- [ ] **Step 4: Run validation-selector tests**

Run: `python -m unittest discover -s tests -p "test_kronos_validate.py" -v`

Expected: origin selection tests pass without loading Kronos.

- [ ] **Step 5: Commit the validation command**

```powershell
git add research/kronos_validate.py tests/test_kronos_validate.py
git commit -m "test: add historical Kronos forecast comparison"
```

### Task 4: Execute end-to-end verification

**Files:**
- No new source files; capture actual command output for the Task 3 PR description.

**Interfaces:**
- Consumes: pinned submodule, optional dependency file, unit tests, validation command, and the locally fetched CSV.
- Produces: recorded versions and actual three-origin Kronos-versus-real-candle output, or exact install/download/runtime errors.

- [ ] **Step 1: Install optional model dependencies**

Run: `python -m pip install -r requirements-kronos.txt`

Expected: command exits 0; record actual versions from `python -m pip show torch einops huggingface_hub safetensors tqdm`.

- [ ] **Step 2: Run all deterministic tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass without loading remote model weights.

- [ ] **Step 3: Run the real historical model comparison**

Run: `python research/kronos_validate.py --csv data/BTCUSDT_1m.csv --horizon 12 --sample-count 20`

Expected: exit 0 and real forecast/spread versus actual candle values for each of the three origins. Capture stdout and exit code exactly. If installation, download, or inference fails, make at most one second honest attempt, then stop and report the exact error.

- [ ] **Step 4: Run repository checks and commit any final fixes**

Run:
```powershell
git diff --check
python -m pre_commit run --all-files
```

Expected: both exit 0. Do not report a model validation success unless Step 3 actually completes.

- [ ] **Step 5: Commit final validation-related fixes**

If Step 4 required source fixes, stage only Task 3 files and commit with an appropriate `fix:` message. Do not add fetched CSVs, model weights, secrets, or the pre-existing `.gitignore` change.
