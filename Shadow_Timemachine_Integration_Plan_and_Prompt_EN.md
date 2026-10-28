# Shadow Timemachine — Integration Plan and Prompt

Oct 25, 2026 · @tyker

This is an engineering document: an honest framework for what these tools can actually do, an architecture for integrating them, and, at the end, a ready-to-use prompt you can give to an agent with full internet access (your machine or a local Claude Code session) to build and verify the system in practice.

## Why “Predicting the Market Like an LLM” Is a Partly Misleading Analogy

LLMs work well because language is statistically structured: context determines the next word with relatively high accuracy, and the rules do not change. The market is fundamentally different:

- **More noise than signal.** Most short-term price movement is approximately random (a random walk), while genuine signals are weak and sparse.
- **Regimes change (regime shifts).** A pattern that is profitable today may fail next month as market participants adapt to it.
- **Competition is reflexive.** If the market discovers a profitable pattern, traders act against it until it disappears (unlike the rules of language).

**The realistic goal, then, is not to “accurately predict the next chart,” but to:**

1. Estimate a **probability distribution** over possible price paths (rather than one supposedly certain path).
2. Measure the **confidence level** of each estimate (calibrated prediction intervals) and refrain from trading when confidence is low.
3. Convert this distribution into an **adaptive position size and stop-loss**, rather than literally predicting the shape of the next candle.

This is the framework behind the rest of the plan: a **calibrated probabilistic forecasting** system that feeds the previously built `Shadow` bot—not an “absolute prediction machine.”

## Table: The Role of Each Tool

Do not combine tools arbitrarily. Freqtrade and Lean are alternative execution engines to Nautilus (the current choice), so they are excluded from this plan to avoid splitting the system across multiple engines.

| Tool | Role in the system | Layer |
| --- | --- | --- |
| [Kronos](https://github.com/shiyu-coder/Kronos) | A pretrained foundation model that probabilistically forecasts future OHLCV candle sequences (multiple samples per candle). This is arguably the closest thing to an actual “LLM for candles.” | Probabilistic signal |
| [arch](https://github.com/bashtage/arch) | GARCH models forecast **the magnitude of future volatility**, not its direction. Their outputs inform position sizing and stop-loss distance. | Regime |
| [tick](https://github.com/X-DataInitiative/tick) | Hawkes models for self-exciting processes estimate the likelihood of an imminent volatility burst. A short-term microstructure signal. | Signal |
| [PySR](https://github.com/MilesCranmer/PySR) | Symbolic regression searches for simple mathematical formulas connecting features to future returns. It runs offline to discover new, interpretable rules. | Research (offline) |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | Wraps a model with conformal prediction to produce statistically calibrated prediction intervals (not guesses). This is the mechanism that keeps the bot from trading when confidence is insufficient. | Calibration |
| [VectorBT](https://github.com/polakowo/vectorbt) | Quickly evaluates thousands of parameter combinations (it was already used in your project). | Research |
| [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) | Live execution engine for Binance (already present). | Execution |
| [day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill) | A Claude agent skill package for human oversight: checks, PASS/BLOCK risk decisions, and scenario simulation (`trader-timemachine`). It does not replace Nautilus; it provides an oversight layer on top of it. | Oversight |
| Freqtrade / backtesting.py / Lean | Alternatives to Nautilus as execution engines. **Do not add them** unless there is a specific reason (for example, Freqtrade's hyperopt for discovering additional rules). | Not currently used |
| [Laya](https://github.com/NandhaKishorM/laya) (and [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), its hosted version) | A fast (33 ms), non-generative “System 1” decision engine that converts unstructured text (news headlines, summaries) into a structured decision (choice/score/yes-no) with calibrated probabilities. **It is not for price forecasting**; it is a fast, inexpensive final decision gate—for example, “Does this news justify halting trading?”—instead of calling a full LLM for every trade. | Decision gate |

## Layer Architecture

There are 6 layers, each feeding into the next:

1. **Data** — Binance candles (already implemented in `research/fetch_data.py`).
2. **Signal generators** (run in parallel) — Kronos (probabilistic distribution of future candles), arch (volatility forecast), tick (burst probability), and previously discovered PySR rules.
3. **Ensemble layer** — A simple model (weighted voting / gradient boosting) combines the outputs from Layer 2 into a single distribution: probability of an upward move, expected magnitude, and volatility.
4. **Calibration layer** — MAPIE wraps the ensemble model in a calibrated prediction interval. If confidence is low, the system abstains from trading.
5. **Decision and risk layer** — Converts confidence into entry distance, position size (Kelly sizing capped by GARCH volatility), and trailing-stop distance. These values replace the current constants in `ShadowStrategy` (Nautilus).
6. **Execution** — The same previously built Shadow bot based on Nautilus / Binance testnet, with an optional `day-trader-skill` oversight layer for human review before execution.

Addition to Layer 5: before any execution, the trade passes through **Laya** (a fast, calibrated decision gate) as the final filter. It receives the numerical confidence score plus any available textual news and returns a final PASS/BLOCK decision in under 40 ms, rather than calling a full conversational model for every trade.

**Research loop (nightly):** A scheduled job reruns PySR, performs a VectorBT sweep, and recalibrates Kronos/arch on new data. It records the results and waits for human review before promoting any new parameter to the live configuration.

## Phased Implementation Plan

0. **Existing foundation.** The `shadow-bot` project (strategy and dashboard) remains as it is; it has not yet been installed or run.
1. **Installation and initial smoke test.** Install the current package and run `python backtest.py --synthetic` to confirm that it actually works.
2. **Add the generators one at a time.** Start with Kronos (highest expected impact), then arch, then tick. Test each independently against a random baseline.
3. **Ensemble + MAPIE.** Build a simple model that combines the outputs, wrap it with MAPIE, and evaluate it using time-ordered walk-forward validation rather than random splits.
4. **Integrate with `ShadowStrategy`.** Replace the fixed `entry_offset_pct` / `trail_pct` values with outputs from Layer 5, using a threshold to prevent trading when confidence is insufficient.
5. **Dual backtest.** Test the complete system in Nautilus backtesting on a period that was not used for training, with realistic fees and slippage.
6. **Run PySR as a recurring background process.** Discover new rules without interrupting live execution; discover them weekly and have them reviewed by a human before integration.
7. **Run on testnet** for several weeks, monitoring the dashboard daily.
8. **Use `day-trader-skill` as an optional review layer**—it does not execute trades and is used only for human review when desired.
9. **Real money** only after several weeks of stable testnet results and two explicit confirmations from you.

## Ready-to-Use Prompt

Paste this into an agent with full internet access (such as Claude Code on your machine or in a CI environment), because pypi.org is blocked in the current chat environment and Nautilus/VectorBT could not be installed here.

```markdown
You are extending an existing crypto trading project called "shadow-bot" (Nautilus Trader strategy for Binance USDT-M futures, already built and attached as shadow-bot.zip). Unzip it and treat shadow_bot/strategy.py, backtest.py, run_live.py, research/*, dashboard/* as the current baseline — NONE of it has been executed yet, only syntax-checked, so verify everything from scratch.

GOAL
Build "Shadow Timemachine": a probabilistic forecasting layer that feeds the existing Shadow breakout strategy instead of its fixed parameters, combining:
- Kronos (github.com/shiyu-coder/Kronos) for probabilistic multi-step OHLCV forecasts
- arch (github.com/bashtage/arch) for GARCH volatility forecasts
- tick (github.com/X-DataInitiative/tick) for Hawkes-process burst/event-clustering signals
- PySR (github.com/MilesCranmer/PySR) run offline to discover candidate symbolic alpha rules
- MAPIE (github.com/scikit-learn-contrib/MAPIE) to wrap the combined meta-model and produce calibrated confidence intervals
- VectorBT (already used in research/sweep.py) to backtest candidate rules fast
- Nautilus Trader (already used in shadow_bot/strategy.py) for the live/backtest execution engine
- Laya (github.com/NandhaKishorM/laya, optional) as a fast calibrated text-to-decision gate for news/risk, NOT for price forecasting

Do NOT add Freqtrade, backtesting.py or Lean — they are redundant execution engines; Nautilus is the chosen one.

HONESTY CONSTRAINTS (do not violate these)
1. Never claim a package, script or backtest "works" without actually running it and showing the real output (stdout/exit code). If something fails, report the exact error, do not paraphrase it as success.
2. Never report a backtest's profitability without printing trade count, total return, max drawdown, Sharpe, and whether it was tested out-of-sample. A strategy profitable only in-sample is NOT profitable — say so explicitly.
3. Treat all live trading as Binance Futures TESTNET only. Mainnet is out of scope for this task.
4. If an installation or API fails after two honest attempts, stop and report the blocker instead of inventing a workaround that silently changes behavior.

TASKS, IN ORDER
1. Environment: create a fresh venv, install everything from requirements.txt plus nautilus_trader, vectorbt, pysr, arch, tick, mapie, torch (Kronos needs it). Record exact versions that actually installed.
2. Smoke test: run `python backtest.py --synthetic`. Fix any import/API mismatch against the installed Nautilus version until it completes and prints an account report. Do this before touching anything else.
3. Data: run `python research/fetch_data.py --symbol BTCUSDT --days 180` and confirm the CSV has the expected row count and no gaps.
4. Kronos module: write shadow_bot/signals/kronos_forecast.py that loads NeoQuasar/Kronos-small + its tokenizer, takes the last 512 bars, and returns a DataFrame of sampled future paths (use sample_count>=20) plus their spread as a crude confidence proxy. Test it on the fetched data and print a few real forecasts vs what actually happened next, for a handful of historical points.
5. arch module: write shadow_bot/signals/vol_forecast.py using arch.arch_model (GARCH(1,1)) on log returns, refit on a rolling window, output next-period volatility forecast. Validate against realized volatility on held-out data.
6. tick module: write shadow_bot/signals/burst_signal.py fitting a Hawkes process (tick.hawkes) on a proxy event series (e.g. bars where |return| exceeds a threshold), output current excitation intensity as a 0-1 burst score.
7. PySR discovery (offline, research/pysr_discover.py): engineer a feature set (returns, volatility, burst score, volume) over a rolling window, run PySR to search for symbolic formulas predicting forward N-bar return, save the best few equations with their out-of-sample correlation to data/pysr_formulas.json. Do not wire these into live trading automatically — they are candidates for human review.
8. Meta-model + MAPIE (shadow_bot/signals/ensemble.py): combine Kronos path stats + GARCH volatility + Hawkes burst score (+ any PySR formula a human has approved) into one simple model (logistic regression or gradient boosting is enough) predicting P(up over next K bars). Wrap it with MAPIE to get a calibrated prediction interval. Validate calibration explicitly (coverage should roughly match the target confidence on held-out data) — report the actual coverage number, do not assume it's correct.
9. Backtest the full pipeline with VectorBT first (fast iteration), using walk-forward splits (train on month N, test on month N+1, roll forward), not a single train/test split. Report results per fold, not just averaged.
10. Wire the ensemble into shadow_bot/strategy.py: before placing the straddle, call the ensemble; only trade if confidence passes a configurable threshold; set entry_offset_pct/trail_pct from the GARCH volatility forecast and MAPIE interval width instead of fixed constants. Keep this behind a flag so the original fixed-parameter strategy still runs if the ensemble is disabled.
11. Re-run the full Nautilus backtest (backtest.py) with the ensemble-driven strategy on the real fetched data, walk-forward, and compare head-to-head against the original fixed-parameter version. Report both, honestly, even if the ensemble version is worse.
12. Only if the ensemble version is genuinely better out-of-sample after costs: wire it into run_live.py and dashboard/app.py (surface the current confidence score and which signal is driving the decision in the dashboard UI), run on Binance Futures TESTNET, and monitor for at least 2 weeks before suggesting anything beyond testnet.
13. Laya gate (shadow_bot/signals/laya_gate.py, optional): self-host Laya (Apache-2.0) as a fast pre-trade gate. Feed it the day's news headlines plus a short structured summary of the ensemble's current confidence (step 8) as a 'noul' (yes/no probability) question such as "does this news justify blocking the next trade?". Use it to double-check or replace the manual trader-risk skill in step 14, not to replace the Kronos/arch/tick ensemble — Laya classifies text, it does not forecast prices.
14. Leave day-trader-skill's Claude skills (trader-risk, trader-timemachine, trader-broker) as an optional, separate, human-in-the-loop review layer — do not auto-wire its broker execution into the automated Nautilus bot; it's for discretionary oversight, not full automation.

DELIVERABLE
A short report: what was actually installed and ran, each module's real validation numbers (not estimates), the walk-forward comparison table (baseline vs ensemble), and a clear verdict on whether the ensemble is worth running on testnet — or why it isn't yet.
```

## Checklist Before Trusting Any Result

- [ ] Did each script actually return exit code zero and print its output, rather than merely being claimed to have succeeded?
- [ ] Were results split into separate training and test periods using walk-forward validation, rather than a single test?
- [ ] Was every profit figure accompanied by trade count, fees, and slippage, rather than theoretical profit alone?
- [ ] Was MAPIE coverage actually verified (actual coverage approximately matching the target)?
- [ ] Was the system compared with the fixed-parameter baseline using the same data?
- [ ] Was it actually run on Binance Futures testnet (not mainnet) for a sufficient period before any discussion of real money?
- [ ] Is every error or failure reported verbatim, rather than hidden or glossed over?

**Golden rule:** Do not believe any agent (me or anyone else) who says “the system works” or “the strategy is profitable” unless you can see the numbers yourself (trades, chart, and coverage rate).

## Laya / Jev — Explained

[Laya](https://github.com/NandhaKishorM/laya) is an open-source library for fast, calibrated structured decision-making (“System 1”), while [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is the hosted version from the same company (TypeSafe AI). Neither **predicts text nor forecasts prices**—they are fast decision classifiers that convert unstructured text into a limited set of choices with calibrated probabilities. Laya (the open-source option) is included in the architecture as the final decision gate: it takes news (from `trader-news` in day-trader-skill) and the clarification/confidence values from Layer 7, then produces a final calibrated decision: should the trade be executed now or not? This is much faster and cheaper than calling a full LLM for every trade on testnet.
