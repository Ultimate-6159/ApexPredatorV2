"""
Layer 4b — Execution Engine
Converts agent actions into real MT5 orders with SL/TP.
"""

from __future__ import annotations

import logging
from datetime import datetime

import MetaTrader5 as mt5

from config import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    AGENT_ACTION_MAP,
    MAGIC_NUMBER,
    ORDER_COMMENT,
    SLIPPAGE_POINTS,
    SYMBOL,
    Regime,
)
from core.risk_manager import OpenTrade, RiskManager

logger = logging.getLogger("apex_live")


class ExecutionEngine:
    """Bridges AI actions and real MT5 order execution."""

    def __init__(
        self,
        risk_manager: RiskManager,
        symbol: str = SYMBOL,
    ) -> None:
        self.risk = risk_manager
        self.symbol = symbol

    # ── Filling Mode Detection ────────────────────
    def _get_filling_type(self) -> int:
        """Detect broker-supported order filling mode."""
        info = mt5.symbol_info(self.symbol)
        if info is not None:
            if info.filling_mode & 2:  # SYMBOL_FILLING_IOC
                return mt5.ORDER_FILLING_IOC
            if info.filling_mode & 1:  # SYMBOL_FILLING_FOK
                return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    # ── Public API ────────────────────────────────
    def execute_action(
        self,
        raw_action: int,
        regime: Regime,
        price_bid: float,
        price_ask: float,
        atr: float,
        point: float,
        equity: float,
        tick_value: float,
        tick_size: float,
        volume_min: float,
        volume_max: float,
        volume_step: float,
        current_bar: int,
    ) -> bool:
        """Map the agent action to a real order. Returns True if an order was sent."""

        allowed = AGENT_ACTION_MAP[regime]
        action = allowed[raw_action]

        if action == ACTION_HOLD:
            logger.debug("Action=HOLD — no order")
            return False

        direction = "BUY" if action == ACTION_BUY else "SELL"
        entry_price = price_ask if direction == "BUY" else price_bid

        sl, tp = self.risk.calculate_sl_tp(direction, entry_price, atr, regime, point)
        sl_distance = abs(entry_price - sl)
        lot = self.risk.calculate_lot_size(
            equity, sl_distance, tick_value, tick_size,
            volume_min, volume_max, volume_step, point,
        )

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": SLIPPAGE_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(),
        }

        result = mt5.order_send(request)

        # Retry with alternative filling modes on Invalid Request
        if result is not None and result.retcode == 10013:
            logger.warning("Open order retcode=10013 — trying alternative filling modes")
            for alt in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                if alt == request["type_filling"]:
                    continue
                request["type_filling"] = alt
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    break

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order FAILED: %s", result)
            return False

        # Sync actual position ticket + fill price from broker
        actual_ticket = result.order
        actual_price = result.price if result.price > 0 else entry_price
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    actual_ticket = pos.ticket
                    break

        trade = OpenTrade(
            ticket=actual_ticket,
            direction=direction,
            regime=regime,
            entry_price=actual_price,
            sl=sl,
            tp=tp,
            lot=lot,
            open_bar=current_bar,
            open_time=datetime.utcnow(),
        )
        self.risk.register_open(trade)
        logger.info("Order FILLED: ticket=%d  %s %.2f @ %.2f", trade.ticket, direction, lot, actual_price)
        return True

    # ── Close open position ───────────────────────
    def close_open_trade(self, reason: str) -> bool:
        trade = self.risk.open_trade
        if trade is None:
            return False

        # Fetch live position for accurate volume & ticket (MAGIC-only match)
        actual_volume = trade.lot
        actual_ticket = trade.ticket
        position_found = False
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    actual_volume = pos.volume
                    actual_ticket = pos.ticket
                    position_found = True
                    break

        if not position_found:
            logger.warning(
                "Position #%d not found on broker — already closed. Syncing state.",
                trade.ticket,
            )
            self.risk.register_close(0.0)
            return True

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Cannot get tick to close trade")
            return False

        close_price = tick.bid if trade.direction == "BUY" else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if trade.direction == "BUY" else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": actual_volume,
            "type": close_type,
            "position": actual_ticket,
            "price": close_price,
            "deviation": SLIPPAGE_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": f"{ORDER_COMMENT}_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(),
        }

        result = mt5.order_send(request)

        # Retry with alternative filling modes on Invalid Request
        if result is not None and result.retcode == 10013:
            logger.warning("Close retcode=10013 — trying alternative filling modes")
            for alt in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                if alt == request["type_filling"]:
                    continue
                request["type_filling"] = alt
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    break

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Close FAILED (%s): %s", reason, result)
            return False

        pnl = (close_price - trade.entry_price) if trade.direction == "BUY" else (trade.entry_price - close_price)
        pnl_money = pnl * actual_volume * (mt5.symbol_info(self.symbol).trade_contract_size if mt5.symbol_info(self.symbol) else 1)
        logger.info(
            "Trade #%d CLOSED (%s) [%s %s] @ %.2f  PnL=%.2f",
            trade.ticket,
            reason,
            trade.direction,
            trade.regime.value,
            close_price,
            pnl_money,
        )
        self.risk.register_close(pnl_money)
        return True

    # ── Modify SL (Break-Even) ───────────────────
    def modify_sl(self, new_sl: float) -> bool:
        """Move the SL of the current position (e.g. break-even)."""
        trade = self.risk.open_trade
        if trade is None:
            return False

        # Sync position ticket from broker (may differ after partial close)
        actual_ticket = trade.ticket
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    actual_ticket = pos.ticket
                    break

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": actual_ticket,
            "sl": new_sl,
            "tp": trade.tp,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "Modify SL FAILED (retcode=%s): %s",
                result.retcode if result else "None",
                result,
            )
            return False

        self.risk.update_sl(new_sl)
        return True

    # ── Partial Close (Scale-Out) ────────────────
    def partial_close(self, volume_to_close: float) -> bool:
        """Close a portion of the open position."""
        trade = self.risk.open_trade
        if trade is None:
            return False

        # Sync position ticket from broker
        actual_ticket = trade.ticket
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    actual_ticket = pos.ticket
                    break

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("Cannot get tick for partial close")
            return False

        close_price = tick.bid if trade.direction == "BUY" else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if trade.direction == "BUY" else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume_to_close,
            "type": close_type,
            "position": actual_ticket,
            "price": close_price,
            "deviation": SLIPPAGE_POINTS,
            "magic": MAGIC_NUMBER,
            "comment": f"{ORDER_COMMENT}_PARTIAL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(),
        }

        result = mt5.order_send(request)

        # Retry with alternative filling modes on Invalid Request
        if result is not None and result.retcode == 10013:
            logger.warning("Partial close retcode=10013 — trying alternative filling modes")
            for alt in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                if alt == request["type_filling"]:
                    continue
                request["type_filling"] = alt
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    break

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Partial close FAILED: %s", result)
            return False

        remaining = round(trade.lot - volume_to_close, 2)
        self.risk.update_lot(remaining)

        # Sync ticket after partial close (may change on some brokers)
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    if pos.ticket != actual_ticket:
                        logger.info(
                            "Position ticket updated: %d → %d (after partial close)",
                            actual_ticket,
                            pos.ticket,
                        )
                    self.risk.update_ticket(pos.ticket)
                    break

        logger.info(
            "Partial close FILLED: ticket=%d  closed %.2f lots  remaining %.2f lots",
            trade.ticket,
            volume_to_close,
            remaining,
        )
        return True
