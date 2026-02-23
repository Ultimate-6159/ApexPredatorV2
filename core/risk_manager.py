"""
Layer 4a — Risk Manager  (Reality Shield)
Handles position sizing, time stops, regime-shift exits, and the circuit breaker.
Strictly NO Martingale.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import (
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    CONSECUTIVE_LOSS_LIMIT,
    DEFAULT_SL_PIPS,
    DEFAULT_TP_PIPS,
    HALT_MINUTES,
    MAX_DRAWDOWN_PCT,
    MAX_HOLDING_BARS,
    RISK_PER_TRADE_PCT,
    Regime,
)

logger = logging.getLogger("apex_live")


@dataclass
class OpenTrade:
    ticket: int
    direction: str          # "BUY" or "SELL"
    regime: Regime
    entry_price: float
    sl: float
    tp: float
    lot: float
    open_bar: int           # bar index when opened
    open_time: datetime = field(default_factory=datetime.utcnow)


class RiskManager:
    """Central risk authority — every trade decision goes through here."""

    def __init__(self) -> None:
        self._consecutive_losses: int = 0
        self._halt_until: datetime | None = None
        self._starting_balance: float | None = None
        self._hard_stop: bool = False
        self._open_trade: OpenTrade | None = None
        self._current_bar: int = 0

    # ── Status ────────────────────────────────────
    @property
    def is_halted(self) -> bool:
        if self._hard_stop:
            return True
        if self._halt_until and datetime.utcnow() < self._halt_until:
            remaining = (self._halt_until - datetime.utcnow()).seconds
            logger.warning("Circuit breaker active — %d s remaining", remaining)
            return True
        return False

    @property
    def has_open_trade(self) -> bool:
        return self._open_trade is not None

    @property
    def open_trade(self) -> OpenTrade | None:
        return self._open_trade

    # ── Position Sizing (Anti-Martingale) ─────────
    @staticmethod
    def calculate_lot_size(
        equity: float,
        sl_distance: float,
        tick_value: float,
        tick_size: float,
        volume_min: float,
        volume_max: float,
        volume_step: float,
        point: float,
    ) -> float:
        """Tick-value position sizing — risk % of equity per trade.

        Formula:
          value_per_point = tick_value × (point / tick_size)
          raw_lot = risk_amount / (sl_points × value_per_point)
        """
        risk_amount = equity * (RISK_PER_TRADE_PCT / 100.0)

        if sl_distance <= 0 or point <= 0:
            return volume_min

        sl_points = sl_distance / point

        if tick_value > 0 and tick_size > 0:
            value_per_point = tick_value * point / tick_size
        else:
            logger.warning("tick_value/tick_size unavailable — falling back to point")
            value_per_point = point

        if value_per_point <= 0:
            return volume_min

        raw_lot = risk_amount / (sl_points * value_per_point)

        # Snap to broker grid
        raw_lot = max(raw_lot, volume_min)
        raw_lot = min(raw_lot, volume_max)
        raw_lot = math.floor(raw_lot / volume_step) * volume_step
        raw_lot = round(raw_lot, 2)
        return raw_lot

    # ── SL / TP Calculation ───────────────────────
    @staticmethod
    def calculate_sl_tp(
        direction: str,
        price: float,
        atr: float,
        regime: Regime,
        point: float,
    ) -> tuple[float, float]:
        """ATR-based SL and regime-specific TP multiplier."""
        sl_dist = atr * ATR_SL_MULTIPLIER if atr > 0 else DEFAULT_SL_PIPS * point
        tp_mult = ATR_TP_MULTIPLIER.get(regime, 2.0)
        tp_dist = atr * tp_mult if atr > 0 else DEFAULT_TP_PIPS * point

        if direction == "BUY":
            sl = round(price - sl_dist, 2)
            tp = round(price + tp_dist, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp = round(price - tp_dist, 2)
        return sl, tp

    # ── Trade Lifecycle ───────────────────────────
    def register_open(self, trade: OpenTrade) -> None:
        self._open_trade = trade
        logger.info(
            "Registered trade #%d  %s %.2f lots @ %.2f  SL=%.2f TP=%.2f  [%s]",
            trade.ticket,
            trade.direction,
            trade.lot,
            trade.entry_price,
            trade.sl,
            trade.tp,
            trade.regime.value,
        )

    def register_close(self, profit: float) -> None:
        if profit < 0:
            self._consecutive_losses += 1
            logger.info(
                "Loss recorded — consecutive losses: %d / %d",
                self._consecutive_losses,
                CONSECUTIVE_LOSS_LIMIT,
            )
            if self._consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                self._halt_until = datetime.utcnow() + timedelta(minutes=HALT_MINUTES)
                logger.warning(
                    "CIRCUIT BREAKER triggered — halting until %s",
                    self._halt_until.isoformat(),
                )
                self._consecutive_losses = 0
        else:
            self._consecutive_losses = 0

        self._open_trade = None

    def update_sl(self, new_sl: float) -> None:
        """Update tracked SL after modification (e.g. break-even)."""
        if self._open_trade is not None:
            self._open_trade.sl = new_sl

    def update_lot(self, new_lot: float) -> None:
        """Update tracked lot after partial close."""
        if self._open_trade is not None:
            self._open_trade.lot = new_lot

    def update_ticket(self, new_ticket: int) -> None:
        """Update tracked position ticket (may change after partial close)."""
        if self._open_trade is not None:
            self._open_trade.ticket = new_ticket

    # ── Drawdown Check ────────────────────────
    def check_drawdown(self, current_balance: float) -> bool:
        """Return True if max drawdown breached → full stop."""
        if self._starting_balance is None:
            self._starting_balance = current_balance
            return False

        peak = max(self._starting_balance, current_balance)
        dd_pct = (peak - current_balance) / peak * 100

        if dd_pct >= MAX_DRAWDOWN_PCT:
            logger.critical(
                "MAX DRAWDOWN %.1f%% breached (limit %.1f%%) — FULL STOP",
                dd_pct,
                MAX_DRAWDOWN_PCT,
            )
            self._hard_stop = True
            return True
        return False

    # ── Time Stop (Guillotine) ────────────────────
    def should_time_stop(self, current_bar: int) -> bool:
        if self._open_trade is None:
            return False
        bars_held = current_bar - self._open_trade.open_bar
        limit = MAX_HOLDING_BARS.get(self._open_trade.regime, 20)
        if bars_held >= limit:
            logger.warning(
                "TIME STOP — held %d bars (limit %d) for regime %s",
                bars_held,
                limit,
                self._open_trade.regime.value,
            )
            return True
        return False

    def update_bar(self, bar_index: int) -> None:
        self._current_bar = bar_index
