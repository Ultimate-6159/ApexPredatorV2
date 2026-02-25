"""
Layer 4b — Execution Engine
Converts agent actions into real MT5 orders with SL/TP.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import MetaTrader5 as mt5

from config import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    AGENT_ACTION_MAP,
    LIMIT_ORDER_EXPIRY_SEC,
    MAGIC_NUMBER,
    MODIFY_THRESHOLD_POINTS,
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

        # V5.1: Throttle — skip if price delta is negligible
        info = mt5.symbol_info(self.symbol)
        if info is not None:
            _pt = info.point
            if _pt > 0 and abs(trade.sl - new_sl) < MODIFY_THRESHOLD_POINTS * _pt:
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

    # ── Modify TP (Elastic TP Expansion) ─────────
    def modify_tp(self, new_tp: float) -> bool:
        """Move the TP of the current position (e.g. elastic TP expansion)."""
        trade = self.risk.open_trade
        if trade is None:
            return False

        # V5.1: Throttle — skip if price delta is negligible
        info = mt5.symbol_info(self.symbol)
        if info is not None:
            _pt = info.point
            if _pt > 0 and abs(trade.tp - new_tp) < MODIFY_THRESHOLD_POINTS * _pt:
                return False

        # Sync position ticket from broker
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
            "sl": trade.sl,
            "tp": new_tp,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "Modify TP FAILED (retcode=%s): %s",
                result.retcode if result else "None",
                result,
            )
            return False

        self.risk.update_tp(new_tp)
        return True

    # ── Partial Close (Scale-Out) ────────────────
    def partial_close(self, volume_to_close: float) -> bool:
        """Close a portion of the open position."""
        trade = self.risk.open_trade
        if trade is None:
            return False

        # V3.7: Guard — skip if lot already at broker minimum
        info = mt5.symbol_info(self.symbol)
        if info is not None and trade.lot <= info.volume_min:
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

    # ── V3.0: Pyramid Position (fire-and-forget) ─────
    def open_pyramid_position(
        self,
        direction: str,
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
    ) -> int | None:
        """Open a 2nd position without registering in RiskManager.

        Returns the ticket number on success, or None on failure.
        The caller (LiveEngine) tracks the pyramid ticket separately.
        """
        entry_price = price_bid if direction == "SELL" else price_ask
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
            "comment": f"{ORDER_COMMENT}_PYR",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(),
        }

        result = mt5.order_send(request)

        # Retry with alternative filling modes on Invalid Request
        if result is not None and result.retcode == 10013:
            logger.warning("Pyramid open retcode=10013 — trying alternative filling modes")
            for alt in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                if alt == request["type_filling"]:
                    continue
                request["type_filling"] = alt
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    break

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Pyramid order FAILED: %s", result)
            return None

        # Resolve actual ticket from broker
        actual_ticket = result.order
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            magic_tickets = {pos.ticket for pos in positions if pos.magic == MAGIC_NUMBER}
            # The new ticket is any MAGIC ticket that isn't the primary trade
            primary = self.risk.open_trade
            primary_ticket = primary.ticket if primary else None
            for t in magic_tickets:
                if t != primary_ticket:
                    actual_ticket = t
                    break

        actual_price = result.price if result.price > 0 else entry_price
        logger.info(
            "🔥 PYRAMID FILLED: ticket=%d  %s %.2f @ %.2f  SL=%.2f TP=%.2f",
            actual_ticket, direction, lot, actual_price, sl, tp,
        )
        return actual_ticket

    # ── V3.0: Close ALL positions (regime shift / shutdown) ─
    def close_all_positions(self, reason: str) -> int:
        """Close every MAGIC_NUMBER position on this symbol. Returns count closed."""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logger.error("Cannot get tick to close position #%d", pos.ticket)
                continue

            direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            close_price = tick.bid if direction == "BUY" else tick.ask
            close_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": SLIPPAGE_POINTS,
                "magic": MAGIC_NUMBER,
                "comment": f"{ORDER_COMMENT}_{reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._get_filling_type(),
            }

            result = mt5.order_send(request)

            # Retry with alternative filling modes
            if result is not None and result.retcode == 10013:
                for alt in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                    if alt == request["type_filling"]:
                        continue
                    request["type_filling"] = alt
                    result = mt5.order_send(request)
                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                        break

            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    "CLOSED #%d (%s) [%s] vol=%.2f @ %.2f",
                    pos.ticket, reason, direction, pos.volume, close_price,
                )
                closed += 1
            else:
                logger.error("Failed to close #%d (%s): %s", pos.ticket, reason, result)

        return closed

    # ── V5.0: Limit Order Spoofer ────────────────────
    def place_limit_order(
        self,
        direction: str,
        limit_price: float,
        regime: Regime,
        atr: float,
        point: float,
        equity: float,
        tick_value: float,
        tick_size: float,
        volume_min: float,
        volume_max: float,
        volume_step: float,
    ) -> int | None:
        """Place a Limit Order at *limit_price* with auto-expiry.

        Returns the pending order ticket on success, or None on failure.
        The order expires after LIMIT_ORDER_EXPIRY_SEC (one M5 bar).
        """
        sl, tp = self.risk.calculate_sl_tp(direction, limit_price, atr, regime, point)
        sl_distance = abs(limit_price - sl)
        lot = self.risk.calculate_lot_size(
            equity, sl_distance, tick_value, tick_size,
            volume_min, volume_max, volume_step, point,
        )

        order_type = (
            mt5.ORDER_TYPE_BUY_LIMIT
            if direction == "BUY"
            else mt5.ORDER_TYPE_SELL_LIMIT
        )

        expiry_time = datetime.utcnow() + timedelta(seconds=LIMIT_ORDER_EXPIRY_SEC)

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": limit_price,
            "sl": sl,
            "tp": tp,
            "magic": MAGIC_NUMBER,
            "comment": f"{ORDER_COMMENT}_LIMIT",
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": int(expiry_time.timestamp()),
            "type_filling": self._get_filling_type(),
        }

        result = mt5.order_send(request)

        # Fallback: if ORDER_TIME_SPECIFIED not supported, use GTC
        if result is not None and result.retcode in (10013, 10030):
            logger.warning(
                "Limit order TIME_SPECIFIED rejected (retcode=%d) — "
                "falling back to GTC",
                result.retcode,
            )
            request["type_time"] = mt5.ORDER_TIME_GTC
            request.pop("expiration", None)
            result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Limit order FAILED: %s", result)
            return None

        ticket = result.order
        logger.info(
            "📌 LIMIT ORDER PLACED: ticket=%d  %s_LIMIT %.2f @ %.5f  "
            "SL=%.5f TP=%.5f (expires %ds)",
            ticket,
            direction,
            lot,
            limit_price,
            sl,
            tp,
            LIMIT_ORDER_EXPIRY_SEC,
        )
        return ticket

    def cancel_pending_orders(self) -> int:
        """Cancel all MAGIC_NUMBER pending orders on this symbol. Returns count."""
        orders = mt5.orders_get(symbol=self.symbol)
        if not orders:
            return 0

        cancelled = 0
        for order in orders:
            if order.magic != MAGIC_NUMBER:
                continue
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("❌ LIMIT ORDER CANCELLED: ticket=%d", order.ticket)
                cancelled += 1
            else:
                logger.error(
                    "Failed to cancel pending #%d: %s", order.ticket, result
                )
        return cancelled
