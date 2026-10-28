# Shadow Bot (Nautilus Trader + VectorBT + Dashboard)

Shadow-style breakout bot for **Binance USDT-M futures**:
straddle of stop orders -> one fills, the other is cancelled -> trailing stop -> grid-forward adds.

## Layout
- `shadow_bot/strategy.py`  the strategy (Nautilus)
- `backtest.py`             run the real strategy in Nautilus on CSV or synthetic data
- `research/fetch_data.py`  download Binance candles
- `research/sweep.py`       VectorBT parameter sweep (in-sample / out-of-sample)
- `run_live.py`             live runner, testnet by default
- `dashboard/`              FastAPI + web UI to control everything

## Setup (Python 3.12+)
```
git clone --recurse-submodules https://github.com/Ridamdli/shadow-bot.git
cd shadow-bot
pip install -r requirements.txt
cp .env.example .env      # add TESTNET keys from https://testnet.binancefuture.com
```

For an existing checkout, initialize the pinned Kronos source with
`git submodule update --init --recursive`. To run the optional research
forecaster, install its dependencies with
`python -m pip install -r requirements-kronos.txt`. The first forecast downloads
and caches `NeoQuasar/Kronos-small` and its tokenizer from Hugging Face.
Forecasts are research-only and are not connected to the strategy or live
trading.

## Workflow (do it in this order)
1. `python research/fetch_data.py --symbol BTCUSDT --days 90`
2. `python research/sweep.py --csv data/BTCUSDT_1m.csv`
3. `python backtest.py --csv data/BTCUSDT_1m.csv --offset X --trail Y`  (use sweep winners)
4. `python backtest.py --synthetic`  quick smoke test that everything runs
5. `uvicorn dashboard.app:app --host 127.0.0.1 --port 8000`  then open http://127.0.0.1:8000
6. Run on **testnet for several weeks**. Only then consider mainnet (`--mainnet` + `CONFIRM_MAINNET=YES`).

## Safety built in
- Daily realized-loss limit stops new entries until the next UTC day.
- New stop is placed before the old one is cancelled; a rejected stop triggers flatten.
- Strategy flattens on stop. Dashboard: pause / resume / emergency flatten.
- Mainnet needs two explicit confirmations. Dashboard listens on localhost only.
- API keys: Futures trading ON, Withdrawals OFF, restrict by IP.

## Known limits
- Written without being able to install Nautilus/VectorBT in the authoring sandbox:
  syntax-checked only, NOT executed. Nautilus APIs change between versions; if an import or
  signature fails, run `python backtest.py --synthetic` first and fix against your installed version.
- VectorBT sweep is an approximation (no grid-forward, trigger-price fills).
- Trailing stop moves on 1-minute bars, not tick by tick.
- Backtests ignore funding rates and real order-book slippage. A profitable backtest is not a promise.
