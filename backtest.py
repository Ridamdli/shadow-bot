"""Run the real ShadowStrategy in Nautilus' backtest engine.

    python backtest.py --csv data/BTCUSDT_1m.csv
    python backtest.py --synthetic          # random-walk data, just a smoke test
"""
from __future__ import annotations

import argparse
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from shadow_bot.strategy import ShadowConfig, ShadowStrategy

BAR_TYPE = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"


def synthetic(n: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.0007, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0004, n))
    return pd.DataFrame({
        "ts": (1_700_000_000_000 + np.arange(n) * 60_000).astype("int64"),
        "open": open_, "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread), "close": close, "volume": 10.0,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--offset", type=float, default=0.30)
    ap.add_argument("--trail", type=float, default=0.40)
    ap.add_argument("--grid", type=float, default=0.30)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--size", type=str, default="0.01")
    a = ap.parse_args()
    if not a.csv and not a.synthetic:
        ap.error("pass --csv or --synthetic")

    df = synthetic() if a.synthetic else pd.read_csv(a.csv)

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(10_000, USDT)],
    )
    inst = TestInstrumentProvider.btcusdt_perp_binance()
    engine.add_instrument(inst)

    bt = BarType.from_str(BAR_TYPE)
    bars = [
        Bar(
            bt,
            inst.make_price(r.open), inst.make_price(r.high),
            inst.make_price(r.low), inst.make_price(r.close),
            Quantity.from_str(f"{r.volume:.3f}"),
            int(r.ts) * 1_000_000, int(r.ts) * 1_000_000,
        )
        for r in df.itertuples()
    ]
    engine.add_data(bars)

    engine.add_strategy(ShadowStrategy(ShadowConfig(
        instrument_id=str(inst.id),
        bar_type=BAR_TYPE,
        trade_size=Decimal(a.size),
        entry_offset_pct=a.offset,
        trail_pct=a.trail,
        grid_step_pct=a.grid,
        max_levels=a.levels,
        publish_state=False,
    )))
    engine.run()

    print(engine.trader.generate_account_report(venue))
    print(engine.trader.generate_positions_report())
    engine.dispose()


if __name__ == "__main__":
    main()
