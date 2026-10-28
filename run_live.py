"""Run ShadowStrategy on Binance USDT-M futures.

Default is TESTNET. Mainnet needs  --mainnet  AND  CONFIRM_MAINNET=YES  in .env.

    python run_live.py --symbol BTCUSDT --size 0.002
"""
from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal

from dotenv import load_dotenv
from nautilus_trader.adapters.binance import (
    BINANCE,
    BinanceAccountType,
    BinanceDataClientConfig,
    BinanceExecClientConfig,
    BinanceInstrumentProviderConfig,
    BinanceLiveDataClientFactory,
    BinanceLiveExecClientFactory,
)
from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
from nautilus_trader.config import LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode

from shadow_bot.strategy import ShadowConfig, ShadowStrategy


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--size", default="0.002", help="contracts per entry (BTC)")
    ap.add_argument("--offset", type=float, default=0.30)
    ap.add_argument("--trail", type=float, default=0.40)
    ap.add_argument("--grid", type=float, default=0.30)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--max-daily-loss", type=float, default=20.0)
    ap.add_argument("--mainnet", action="store_true")
    a = ap.parse_args()

    testnet = not a.mainnet
    if a.mainnet:
        if os.getenv("CONFIRM_MAINNET") != "YES":
            sys.exit("Refusing mainnet: set CONFIRM_MAINNET=YES in .env first.")
        key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    else:
        key = os.getenv("BINANCE_FUTURES_TESTNET_API_KEY")
        secret = os.getenv("BINANCE_FUTURES_TESTNET_API_SECRET")
    if not key or not secret:
        sys.exit("Missing API key/secret in .env")

    instrument_id = f"{a.symbol}-PERP.BINANCE"
    common = dict(
        api_key=key,
        api_secret=secret,
        account_type=BinanceAccountType.USDT_FUTURES,
        environment=BinanceEnvironment.TESTNET if testnet else BinanceEnvironment.LIVE,
        instrument_provider=BinanceInstrumentProviderConfig(load_all=True),
    )
    config = TradingNodeConfig(
        trader_id="SHADOW-001",
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=True),
        data_clients={BINANCE: BinanceDataClientConfig(**common)},
        exec_clients={BINANCE: BinanceExecClientConfig(**common)},
        timeout_connection=60.0,
        timeout_reconciliation=15.0,
        timeout_portfolio=15.0,
        timeout_disconnection=10.0,
    )

    node = TradingNode(config=config)
    node.trader.add_strategy(ShadowStrategy(ShadowConfig(
        instrument_id=instrument_id,
        bar_type=f"{instrument_id}-1-MINUTE-LAST-EXTERNAL",
        trade_size=Decimal(a.size),
        entry_offset_pct=a.offset,
        trail_pct=a.trail,
        grid_step_pct=a.grid,
        max_levels=a.levels,
        max_daily_loss_usdt=a.max_daily_loss,
    )))
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
    node.build()
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()
