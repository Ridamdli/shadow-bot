"""Shadow-style breakout strategy for Nautilus Trader (Binance USDT-M futures).

Logic
-----
1. Flat: place a BUY STOP above and a SELL STOP below the current price
   (a "straddle"). If neither triggers within `straddle_ttl_bars`, re-centre it.
2. One side fills: cancel the other, open a reduce-only trailing stop.
3. Grid forward: every time price moves `grid_step_pct` further in our favour,
   add one more position of `trade_size` (max `max_levels`). The trailing stop
   always covers the whole net position.
4. Safety: daily loss limit, pause/flatten commands from the dashboard,
   flatten if the exchange rejects our protective stop.

The trailing stop is managed by the strategy on bar highs/lows (not the
exchange-native trailing order), so behaviour is identical in backtest and live.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled, OrderRejected, PositionClosed
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from shadow_bot import control


class ShadowConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    trade_size: Decimal = Decimal("0.001")
    entry_offset_pct: float = 0.30      # straddle distance from price
    trail_pct: float = 0.40             # trailing stop distance from best price
    grid_step_pct: float = 0.30         # add a position every X% in our favour
    max_levels: int = 3                 # max stacked positions
    straddle_ttl_bars: int = 30         # re-centre straddle after N bars
    min_stop_move_pct: float = 0.02     # don't re-send stop for tiny moves
    max_daily_loss_usdt: float = 20.0   # kill switch (realized PnL, UTC day)
    publish_state: bool = True          # write data/state.json for the dashboard


class ShadowStrategy(Strategy):
    def __init__(self, config: ShadowConfig) -> None:
        super().__init__(config)
        self._iid = InstrumentId.from_str(config.instrument_id)
        self._bar_type = BarType.from_str(config.bar_type)
        self._inst = None

        self._buy = None
        self._sell = None
        self._trail_order = None
        self._stop_px: float | None = None
        self._straddle_age = 0

        self._side: OrderSide | None = None
        self._qty = Decimal(0)
        self._level = 0
        self._last_add = 0.0
        self._best = 0.0
        self._add_ids: set = set()
        self._add_pending = False

        self._last_price = 0.0
        self._paused = False
        self._killed = False
        self._kill_reason = ""
        self._day = datetime.now(timezone.utc).date()
        self._daily_pnl = 0.0
        self._total_pnl = 0.0
        self._closed_trades = 0
        self._last_cmd_id = 0

    # ------------------------------------------------------------------ life
    def on_start(self) -> None:
        self._inst = self.cache.instrument(self._iid)
        if self._inst is None:
            self.log.error(f"Instrument {self._iid} not found")
            self.stop()
            return
        _, self._last_cmd_id = control.read_command()  # ignore stale commands
        self.subscribe_bars(self._bar_type)
        self.clock.set_timer(
            name="shadow-control",
            interval=timedelta(seconds=2),
            callback=self._on_timer,
        )
        self._publish("started")

    def on_stop(self) -> None:
        # Never leave a position without its protective stop.
        self._flatten("strategy stopped")
        self._publish("stopped")

    # ------------------------------------------------------------------ data
    def on_bar(self, bar: Bar) -> None:
        self._last_price = float(bar.close)
        self._roll_day(bar)

        if self._side is not None:
            self._manage_position(bar)
            return
        if self._paused or self._killed:
            self._cancel_straddle()
            return
        self._manage_straddle()

    # ----------------------------------------------------------- straddle
    def _manage_straddle(self) -> None:
        live = [o for o in (self._buy, self._sell) if o is not None and not o.is_closed]
        if len(live) == 2:
            self._straddle_age += 1
            if self._straddle_age >= self.config.straddle_ttl_bars:
                self._cancel_straddle()
            return
        if live:  # half-open straddle, reset
            self._cancel_straddle()
            return
        self._place_straddle(self._last_price)

    def _place_straddle(self, price: float) -> None:
        off = self.config.entry_offset_pct / 100.0
        qty = self._inst.make_qty(self.config.trade_size)
        self._buy = self.order_factory.stop_market(
            instrument_id=self._iid,
            order_side=OrderSide.BUY,
            quantity=qty,
            trigger_price=self._inst.make_price(price * (1 + off)),
            time_in_force=TimeInForce.GTC,
        )
        self._sell = self.order_factory.stop_market(
            instrument_id=self._iid,
            order_side=OrderSide.SELL,
            quantity=qty,
            trigger_price=self._inst.make_price(price * (1 - off)),
            time_in_force=TimeInForce.GTC,
        )
        self._straddle_age = 0
        self.submit_order(self._buy)
        self.submit_order(self._sell)

    def _cancel_straddle(self) -> None:
        for o in (self._buy, self._sell):
            if o is not None and not o.is_closed:
                self.cancel_order(o)
        self._buy = self._sell = None
        self._straddle_age = 0

    # ----------------------------------------------------------- position
    def _manage_position(self, bar: Bar) -> None:
        high, low, close = float(bar.high), float(bar.low), float(bar.close)
        trail = self.config.trail_pct / 100.0
        min_move = self.config.min_stop_move_pct / 100.0

        if self._stop_px is None:
            return
        if self._side == OrderSide.BUY:
            self._best = max(self._best, high)
            cand = self._best * (1 - trail)
            if cand > self._stop_px * (1 + min_move):
                self._set_stop(cand)
        else:
            self._best = min(self._best, low)
            cand = self._best * (1 + trail)
            if cand < self._stop_px * (1 - min_move):
                self._set_stop(cand)

        self._maybe_add(close)

    def _maybe_add(self, price: float) -> None:
        if self._paused or self._killed or self._add_pending:
            return
        if self._level >= self.config.max_levels:
            return
        step = self.config.grid_step_pct / 100.0
        if self._side == OrderSide.BUY:
            hit = price >= self._last_add * (1 + step)
        else:
            hit = price <= self._last_add * (1 - step)
        if not hit:
            return
        order = self.order_factory.market(
            instrument_id=self._iid,
            order_side=self._side,
            quantity=self._inst.make_qty(self.config.trade_size),
        )
        self._add_ids.add(order.client_order_id)
        self._add_pending = True
        self.submit_order(order)

    def _set_stop(self, price: float) -> None:
        """Place the new stop first, then cancel the old one (never unprotected)."""
        old = self._trail_order
        close_side = OrderSide.SELL if self._side == OrderSide.BUY else OrderSide.BUY
        new = self.order_factory.stop_market(
            instrument_id=self._iid,
            order_side=close_side,
            quantity=self._inst.make_qty(self._qty),
            trigger_price=self._inst.make_price(price),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        self._trail_order = new
        self._stop_px = price
        self.submit_order(new)
        if old is not None and not old.is_closed:
            self.cancel_order(old)

    # ------------------------------------------------------------- events
    def on_order_filled(self, event: OrderFilled) -> None:
        cid = event.client_order_id
        px = float(event.last_px)

        if self._buy is not None and cid == self._buy.client_order_id:
            self._on_entry(OrderSide.BUY, px)
        elif self._sell is not None and cid == self._sell.client_order_id:
            self._on_entry(OrderSide.SELL, px)
        elif cid in self._add_ids:
            self._add_ids.discard(cid)
            self._add_pending = False
            self._level += 1
            self._qty += self.config.trade_size
            self._last_add = px
            if self._stop_px is not None:
                self._set_stop(self._stop_px)  # resize stop to the new net qty

    def _on_entry(self, side: OrderSide, px: float) -> None:
        other = self._sell if side == OrderSide.BUY else self._buy
        if other is not None and not other.is_closed:
            self.cancel_order(other)
        self._buy = self._sell = None

        self._side = side
        self._qty = self.config.trade_size
        self._level = 1
        self._last_add = px
        self._best = px
        trail = self.config.trail_pct / 100.0
        self._set_stop(px * (1 - trail) if side == OrderSide.BUY else px * (1 + trail))

    def on_order_rejected(self, event: OrderRejected) -> None:
        if self._trail_order is not None and event.client_order_id == self._trail_order.client_order_id:
            self.log.error(f"Protective stop rejected ({event.reason}) -> flatten")
            self._flatten("stop rejected")

    def on_position_closed(self, event: PositionClosed) -> None:
        pnl = float(event.realized_pnl.as_double())
        self._daily_pnl += pnl
        self._total_pnl += pnl
        self._closed_trades += 1

        self.cancel_all_orders(self._iid)  # drop leftover stop / add orders
        self._side = None
        self._qty = Decimal(0)
        self._level = 0
        self._trail_order = None
        self._stop_px = None
        self._add_ids.clear()
        self._add_pending = False

        if self._daily_pnl <= -abs(self.config.max_daily_loss_usdt) and not self._killed:
            self._killed = True
            self._kill_reason = f"daily loss limit ({self._daily_pnl:.2f} USDT)"
            self.log.warning(f"KILL SWITCH: {self._kill_reason}")

    # ----------------------------------------------------- control / state
    def _on_timer(self, _event) -> None:
        cmd, cid = control.read_command()
        if cid != self._last_cmd_id:
            self._last_cmd_id = cid
            if cmd == "pause":
                self._paused = True
                self._cancel_straddle()
            elif cmd == "resume":
                self._paused = False
                if not self._killed:
                    self._kill_reason = ""
            elif cmd == "flatten":
                self._paused = True
                self._flatten("manual flatten")
        self._publish("running")

    def _flatten(self, reason: str) -> None:
        self.log.warning(f"Flatten: {reason}")
        self._buy = self._sell = None
        self.cancel_all_orders(self._iid)
        self.close_all_positions(self._iid)

    def _roll_day(self, bar: Bar) -> None:
        day = datetime.fromtimestamp(bar.ts_event / 1e9, tz=timezone.utc).date()
        if day != self._day:
            self._day = day
            self._daily_pnl = 0.0
            if self._killed:
                self._killed = False
                self._kill_reason = ""

    def _publish(self, status: str) -> None:
        if not self.config.publish_state:
            return
        upnl = 0.0
        if self._side is not None and self._last_price and self._last_add:
            pass  # exact unrealized PnL comes from the portfolio below
        try:
            u = self.portfolio.unrealized_pnl(self._iid)
            upnl = float(u.as_double()) if u is not None else 0.0
        except Exception:
            pass
        control.write_state({
            "status": status,
            "instrument": self.config.instrument_id,
            "paused": self._paused,
            "killed": self._killed,
            "kill_reason": self._kill_reason,
            "price": self._last_price,
            "side": None if self._side is None else self._side.name,
            "qty": float(self._qty),
            "level": self._level,
            "stop_price": self._stop_px,
            "unrealized_pnl": upnl,
            "daily_pnl": self._daily_pnl,
            "total_pnl": self._total_pnl,
            "closed_trades": self._closed_trades,
            "params": {
                "trade_size": float(self.config.trade_size),
                "entry_offset_pct": self.config.entry_offset_pct,
                "trail_pct": self.config.trail_pct,
                "grid_step_pct": self.config.grid_step_pct,
                "max_levels": self.config.max_levels,
                "max_daily_loss_usdt": self.config.max_daily_loss_usdt,
            },
        })
