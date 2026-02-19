"""
Backtest Engine — Simulates trading on historical data without real orders.
Used to evaluate agent performance before deploying to live trading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    AGENT_ACTION_MAP,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    MAX_HOLDING_BARS,
    RISK_PER_TRADE_PCT,
    Regime,
)
from core.agents.base_agent import BaseAgent
from core.meta_router import MetaRouter
from core.perception_engine import PerceptionEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    entry_bar: int
    exit_bar: int
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    regime: Regime
    exit_reason: str  # TP_HIT, SL_HIT, TIME_STOP, REGIME_SHIFT, END_OF_DATA
    pnl_pct: float


@dataclass
class BacktestResult:
    trades: list[BacktestTrade]
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float


class BacktestEngine:
    """Run historical backtests for the MoE trading system."""

    def __init__(
        self,
        agents: dict[Regime, BaseAgent],
        perception: PerceptionEngine,
    ) -> None:
        self.agents = agents
        self.perception = perception
        self.router = MetaRouter()

    def run(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        initial_balance: float = 10_000.0,
    ) -> BacktestResult:
        """Run backtest on provided historical data."""
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = [initial_balance]
        balance = initial_balance

        position: dict[str, Any] | None = None
        hold_counter = 0

        for i in range(len(features)):
            row = features.iloc[i]
            obs = row.values.astype(np.float32)
            regime = self.router.detect_regime(row)

            close = float(ohlcv.iloc[i]["Close"])
            high = float(ohlcv.iloc[i]["High"])
            low = float(ohlcv.iloc[i]["Low"])
            atr = float((ohlcv["High"] - ohlcv["Low"]).rolling(14).mean().iloc[i]) if i >= 14 else close * 0.01

            # Manage open position
            if position is not None:
                hold_counter += 1
                exit_reason: str | None = None
                exit_price = close

                # Check SL/TP hits
                if position["direction"] == "BUY":
                    if low <= position["sl"]:
                        exit_reason = "SL_HIT"
                        exit_price = position["sl"]
                    elif high >= position["tp"]:
                        exit_reason = "TP_HIT"
                        exit_price = position["tp"]
                else:  # SELL
                    if high >= position["sl"]:
                        exit_reason = "SL_HIT"
                        exit_price = position["sl"]
                    elif low <= position["tp"]:
                        exit_reason = "TP_HIT"
                        exit_price = position["tp"]

                # Time stop
                max_bars = MAX_HOLDING_BARS.get(position["regime"], 20)
                if exit_reason is None and hold_counter >= max_bars:
                    exit_reason = "TIME_STOP"

                # Regime shift
                if exit_reason is None:
                    if self.router.should_emergency_exit(
                        position["regime"], regime, position["direction"]
                    ):
                        exit_reason = "REGIME_SHIFT"

                if exit_reason:
                    pnl = (exit_price - position["entry_price"]) if position["direction"] == "BUY" else (position["entry_price"] - exit_price)
                    pnl_pct = (pnl / position["entry_price"]) * 100

                    trades.append(
                        BacktestTrade(
                            entry_bar=position["entry_bar"],
                            exit_bar=i,
                            direction=position["direction"],
                            entry_price=position["entry_price"],
                            exit_price=exit_price,
                            sl=position["sl"],
                            tp=position["tp"],
                            regime=position["regime"],
                            exit_reason=exit_reason,
                            pnl_pct=pnl_pct,
                        )
                    )

                    risk_amount = balance * (RISK_PER_TRADE_PCT / 100)
                    sl_dist = abs(position["entry_price"] - position["sl"])
                    actual_dist = abs(exit_price - position["entry_price"])
                    if sl_dist > 0:
                        pnl_dollar = risk_amount * (actual_dist / sl_dist) * (1 if pnl > 0 else -1)
                    else:
                        pnl_dollar = 0
                    balance += pnl_dollar

                    position = None
                    hold_counter = 0

            # Open new position if flat
            if position is None:
                agent = self.agents[regime]
                raw_action = agent.predict(obs)
                allowed = AGENT_ACTION_MAP[regime]
                action = allowed[raw_action]

                if action != 0:  # Not HOLD
                    direction = "BUY" if action == 1 else "SELL"
                    sl_mult = ATR_SL_MULTIPLIER
                    tp_mult = ATR_TP_MULTIPLIER.get(regime, 2.0)

                    if direction == "BUY":
                        sl = close - atr * sl_mult
                        tp = close + atr * tp_mult
                    else:
                        sl = close + atr * sl_mult
                        tp = close - atr * tp_mult

                    position = {
                        "entry_bar": i,
                        "direction": direction,
                        "entry_price": close,
                        "sl": sl,
                        "tp": tp,
                        "regime": regime,
                    }
                    hold_counter = 0

            equity_curve.append(balance)

        # Close any remaining position at end
        if position is not None:
            close = float(ohlcv.iloc[-1]["Close"])
            pnl = (close - position["entry_price"]) if position["direction"] == "BUY" else (position["entry_price"] - close)
            pnl_pct = (pnl / position["entry_price"]) * 100
            trades.append(
                BacktestTrade(
                    entry_bar=position["entry_bar"],
                    exit_bar=len(features) - 1,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=close,
                    sl=position["sl"],
                    tp=position["tp"],
                    regime=position["regime"],
                    exit_reason="END_OF_DATA",
                    pnl_pct=pnl_pct,
                )
            )

        return self._calculate_metrics(trades, equity_curve, initial_balance)

    @staticmethod
    def _calculate_metrics(
        trades: list[BacktestTrade],
        equity_curve: list[float],
        initial_balance: float,
    ) -> BacktestResult:
        if not trades:
            return BacktestResult(
                trades=[],
                total_pnl_pct=0.0,
                win_rate=0.0,
                total_trades=0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
            )

        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct < 0]

        total_pnl_pct = sum(t.pnl_pct for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # Max drawdown
        peak = initial_balance
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        return BacktestResult(
            trades=trades,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            total_trades=len(trades),
            max_drawdown_pct=max_dd,
            sharpe_ratio=float(sharpe),
            profit_factor=profit_factor,
        )


def print_backtest_report(result: BacktestResult) -> None:
    """Pretty-print backtest results."""
    print("\n" + "═" * 50)
    print("        📊 BACKTEST REPORT")
    print("═" * 50)
    print(f"  Total Trades      : {result.total_trades}")
    print(f"  Win Rate          : {result.win_rate:.1f}%")
    print(f"  Total Return      : {result.total_pnl_pct:.2f}%")
    print(f"  Max Drawdown      : {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio      : {result.sharpe_ratio:.2f}")
    print(f"  Profit Factor     : {result.profit_factor:.2f}")
    print("═" * 50)

    # Breakdown by regime
    if result.trades:
        print("\n  📈 Breakdown by Regime:")
        for regime in Regime:
            regime_trades = [t for t in result.trades if t.regime == regime]
            if regime_trades:
                regime_wins = len([t for t in regime_trades if t.pnl_pct > 0])
                regime_wr = regime_wins / len(regime_trades) * 100
                print(f"    {regime.value:18} : {len(regime_trades):3} trades | WR: {regime_wr:.1f}%")

    # Breakdown by exit reason
    print("\n  🚪 Exit Reasons:")
    for reason in ["TP_HIT", "SL_HIT", "TIME_STOP", "REGIME_SHIFT", "END_OF_DATA"]:
        count = len([t for t in result.trades if t.exit_reason == reason])
        if count > 0:
            print(f"    {reason:18} : {count}")

    print("═" * 50 + "\n")
