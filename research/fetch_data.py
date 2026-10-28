"""Download Binance USDT-M futures candles (public endpoint, no API key).

    python research/fetch_data.py --symbol BTCUSDT --interval 1m --days 90
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://fapi.binance.com/fapi/v1/klines"
DATA = Path(__file__).resolve().parent.parent / "data"


def fetch(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    rows: list[list] = []
    cur = start
    while cur < end:
        r = requests.get(
            URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": end - 1,
                "limit": 1500,
            },
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(row for row in batch if start <= row[0] < end)
        cur = batch[-1][0] + 1
        time.sleep(0.2)  # stay far below rate limits
    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume", "close_ts", "qv", "n", "tb", "tq", "ig"],
    )
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(float)
    df["ts"] = df["ts"].astype("int64")
    return df.drop_duplicates("ts").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=90)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    out = DATA / f"{a.symbol}_{a.interval}.csv"
    d = fetch(a.symbol, a.interval, a.days)
    d.to_csv(out, index=False)
    print(f"saved {len(d):,} candles -> {out}")
