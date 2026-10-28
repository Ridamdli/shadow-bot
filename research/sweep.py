"""VectorBT parameter sweep for the Shadow-style breakout idea.

This is a FAST, APPROXIMATE screen (single position, no grid-forward, fills at
trigger price). Use it to find robust parameter regions, then confirm the
survivors with backtest.py, which runs the real strategy in Nautilus.

Parameters are optimised on the first 70% of the data (in-sample) and every
combo is also scored on the last 30% (out-of-sample). Trust only combos that
are good in BOTH.

    python research/sweep.py --csv data/BTCUSDT_1m.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

ROOT = Path(__file__).resolve().parent.parent
FEE = 0.0004       # Binance futures taker 0.04%
SLIPPAGE = 0.0002  # 0.02%


def load(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df.index = pd.to_datetime(df["ts"], unit="ms")
    return df[["open", "high", "low", "close"]]


def run_one(df: pd.DataFrame, offset: float, trail: float) -> dict:
    ref = df["close"].shift(1)
    long_e = df["high"] >= ref * (1 + offset)
    short_e = df["low"] <= ref * (1 - offset)
    both = long_e & short_e
    long_e, short_e = long_e & ~both, short_e & ~both  # ambiguous bars skipped

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_e,
        exits=pd.Series(False, index=df.index),
        short_entries=short_e,
        short_exits=pd.Series(False, index=df.index),
        sl_stop=trail,
        sl_trail=True,
        fees=FEE,
        slippage=SLIPPAGE,
        init_cash=10_000,
        freq="1min",
    )
    return {
        "return_pct": float(pf.total_return() * 100),
        "sharpe": float(pf.sharpe_ratio()),
        "max_dd_pct": float(pf.max_drawdown() * 100),
        "trades": int(pf.trades.count()),
        "win_rate_pct": float(pf.trades.win_rate() * 100) if pf.trades.count() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min-trades", type=int, default=30)
    a = ap.parse_args()

    df = load(a.csv)
    cut = int(len(df) * 0.7)
    ins, oos = df.iloc[:cut], df.iloc[cut:]

    offsets = np.round(np.arange(0.001, 0.0101, 0.0005), 5)   # 0.10% .. 1.00%
    trails = np.round(np.arange(0.002, 0.0201, 0.001), 5)      # 0.20% .. 2.00%

    rows = []
    combos = list(itertools.product(offsets, trails))
    for i, (o, t) in enumerate(combos, 1):
        a_in, a_out = run_one(ins, o, t), run_one(oos, o, t)
        rows.append({
            "offset_pct": round(o * 100, 3),
            "trail_pct": round(t * 100, 3),
            **{f"in_{k}": v for k, v in a_in.items()},
            **{f"out_{k}": v for k, v in a_out.items()},
        })
        if i % 20 == 0:
            print(f"{i}/{len(combos)}", flush=True)

    res = pd.DataFrame(rows)
    ok = res[(res.in_trades >= a.min_trades) & (res.out_trades >= a.min_trades // 3)]
    ok = ok[(ok.in_return_pct > 0) & (ok.out_return_pct > 0)]  # profitable in BOTH
    ok = ok.sort_values("out_sharpe", ascending=False)

    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    res.to_csv(out_dir / "sweep_all.csv", index=False)
    top = ok.head(15)
    top.to_csv(out_dir / "sweep_top.csv", index=False)
    (out_dir / "sweep_top.json").write_text(top.to_json(orient="records"))

    print(f"\n{len(res)} combos tested, {len(ok)} profitable in-sample AND out-of-sample")
    print(top.to_string(index=False) if len(top) else "No robust combo found - do not trade this idea as is.")


if __name__ == "__main__":
    main()
