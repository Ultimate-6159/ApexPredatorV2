"""
Analyze Live Trading Logs — Performance Dashboard.

Parses daily rotating logs produced by run_live.py and the underlying
ExecutionEngine / RiskManager loggers, then computes:

  • Win Rate (overall + per-regime agent)
  • Profit Factor  (gross profit / gross loss)
  • Max Drawdown   (peak-to-trough on running balance)
  • Sharpe Ratio / Sortino Ratio / Calmar Ratio  (institutional risk metrics)
  • Expectancy & Payoff Ratio
  • Trade breakdown (TP/SL hit, voluntary close, regime shift, time stop, …)
  • Regime distribution & per-regime P&L

Usage:
    python -m scripts.analyze_live_logs                     # all logs
    python -m scripts.analyze_live_logs --date 2026-02-20   # single day
    python -m scripts.analyze_live_logs --last 7            # last 7 days
    python -m scripts.analyze_live_logs --csv trades.csv    # export trades
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

# ══════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════
_LOG_DIR = Path("logs/live")
_LOG_FILE = "live_trading.log"

# Regex patterns for the file-handler format:
#   2026-02-20 01:45:08,171 | INFO | <message>
_TS_RE = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
_LINE_RE = re.compile(rf"^{_TS_RE} \| (\w+) \| (.+)$")

# Message-level patterns
_BAR_RE = re.compile(
    r"Bar #(\d+) \| Regime: (\S+) \| Action: (\d+) .+ (\S+) \| "
    r"PnL: (-?[\d.]+) \| Balance: (-?[\d.]+)"
)

_OPEN_RE = re.compile(
    r"Registered trade #(\d+)\s+(\w+)\s+([\d.]+) lots? @ ([\d.]+)"
    r"\s+SL=([\d.]+)\s+TP=([\d.]+)\s+\[(\w+)]"
)

_CLOSE_ENGINE_RE = re.compile(
    r"Trade #(\d+) CLOSED \((\w+)\)\s+(?:\[(\w+) (\w+)\] )?@ ([\d.]+)\s+PnL=(-?[\d.]+)"
)

_CLOSE_BROKER_RE = re.compile(
    r"Trade #(\d+) closed by broker \(([^)]+)\)\s*(?:\[(\w+) (\w+)\] )?.+ profit=(-?[\d.]+)"
)

_ACCOUNT_RE = re.compile(
    r"Account: balance=([\d.]+)\s+equity=([\d.]+)"
)

_REGIME_SHIFT_RE = re.compile(
    r"REGIME SHIFT: (\S+) .+ (\S+)"
)

_TELEMETRY_RE = re.compile(
    r"TELEMETRY \| Regime: (\S+) \| "
    r"Critic Value: ([\d.eE+\-]+|N/A) \| "
    r"Confidence: ([\d.]+)% \((.+)\)"
)

_ANOMALY_RE = re.compile(
    r"ANOMALY DETECTED: Extreme feature value \(([\d.]+) STD\)"
)

_CONFIDENCE_GATE_RE = re.compile(
    r"CONFIDENCE GATE: ([\d.]+)% < ([\d.]+)% threshold"
)

# V3.x defense system patterns
_VKR_GATE_RE = re.compile(r"VKR GATE: (\w+) blocked")
_GRACE_PERIOD_RE = re.compile(r"GRACE PERIOD: Shielding trade #(\d+)")
_PHANTOM_FIRE_RE = re.compile(r"(PHANTOM_SWEEP|MOMENTUM_BOUNCE) FIRE: (\w+)")
_PYRAMID_RE = re.compile(r"RISK-FREE PYRAMIDING:")
_BREAK_EVEN_RE = re.compile(r"BREAK-EVEN ACTIVATED for Ticket #(\d+)")
_PARTIAL_CLOSE_RE = re.compile(r"PARTIAL CLOSE:")
_TIME_DECAY_RE = re.compile(r"TIME-DECAY (SHIELD|SQUEEZE): Trade #(\d+)")
_SPREAD_GATE_RE = re.compile(r"SPREAD GATE:")
_COOLDOWN_RE = re.compile(r"COOLDOWN: Waiting for next bar")

# V4.0 defense system patterns
_PENALTY_BOX_RE = re.compile(r"PENALTY BOX: (\w+) (locked|blocked)")
_VOLUME_GATE_RE = re.compile(r"VOLUME GATE: (\w+) rejected")

# V5.0 HFT system patterns
_LIMIT_ORDER_PLACED_RE = re.compile(r"LIMIT SPOOFER: (\w+)_LIMIT placed")
_LIMIT_ORDER_FILLED_RE = re.compile(r"LIMIT ORDER FILLED: ticket=(\d+)")
_LIMIT_ORDER_EXPIRED_RE = re.compile(r"LIMIT ORDER EXPIRED/CANCELLED: ticket=(\d+)")
_ELASTIC_TP_RE = re.compile(r"ELASTIC TP EXPANSION #(\d+)")
_SUBBAR_SCAN_RE = re.compile(r"NEW BAR: Cancelled (\d+) stale limit")


# ══════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════
@dataclass
class Trade:
    ticket: int
    direction: str
    regime: str
    lot: float
    entry_price: float
    sl: float
    tp: float
    open_time: datetime | None = None
    close_price: float = 0.0
    close_reason: str = ""
    pnl: float = 0.0
    close_time: datetime | None = None


@dataclass
class DashboardData:
    trades: list[Trade] = field(default_factory=list)
    bar_count: int = 0
    regime_bars: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    balance_series: list[tuple[datetime, float]] = field(default_factory=list)
    regime_shifts: int = 0
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    # Telemetry
    confidences: list[float] = field(default_factory=list)
    critic_values: list[float] = field(default_factory=list)
    anomaly_count: int = 0
    confidence_gate_count: int = 0
    regime_confidences: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # V3.x defense systems
    vkr_gate_count: int = 0
    grace_period_count: int = 0
    phantom_fire_count: int = 0
    pyramid_count: int = 0
    break_even_count: int = 0
    partial_close_count: int = 0
    time_decay_count: int = 0
    spread_gate_count: int = 0
    cooldown_count: int = 0
    # V4.0 defense systems
    penalty_box_count: int = 0
    volume_gate_count: int = 0
    time_decay_squeeze_count: int = 0
    # V5.0 HFT systems
    limit_order_placed_count: int = 0
    limit_order_filled_count: int = 0
    limit_order_expired_count: int = 0
    elastic_tp_count: int = 0
    stale_limit_cancelled_count: int = 0


# ══
# Log Parser
# ══════════════════════════════════════════════
def _parse_timestamp(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")


def _collect_log_files(
    date_filter: str | None = None,
    last_n_days: int | None = None,
) -> list[Path]:
    """Return matching log files sorted chronologically."""
    if not _LOG_DIR.exists():
        return []

    candidates: list[Path] = []

    main_log = _LOG_DIR / _LOG_FILE
    if main_log.exists():
        candidates.append(main_log)

    for f in sorted(_LOG_DIR.iterdir()):
        if f.name.startswith(_LOG_FILE + ".") and f.is_file():
            candidates.append(f)

    if date_filter:
        filtered = []
        for f in candidates:
            try:
                with open(f, encoding="utf-8") as fh:
                    first = fh.readline()
                    if date_filter in first:
                        filtered.append(f)
                        continue
                    fh.seek(0)
                    for line in fh:
                        if date_filter in line:
                            filtered.append(f)
                            break
            except OSError:
                continue
        candidates = filtered

    if last_n_days:
        cutoff = datetime.utcnow() - timedelta(days=last_n_days)
        filtered = []
        for f in candidates:
            try:
                with open(f, encoding="utf-8") as fh:
                    first = fh.readline()
                    m = _LINE_RE.match(first)
                    if m and _parse_timestamp(m.group(1)) >= cutoff:
                        filtered.append(f)
                        continue
                    fh.seek(0)
                    for line in fh:
                        m = _LINE_RE.match(line)
                        if m and _parse_timestamp(m.group(1)) >= cutoff:
                            filtered.append(f)
                            break
            except OSError:
                continue
        candidates = filtered

    return candidates


def parse_logs(log_files: Sequence[Path]) -> DashboardData:
    """Parse all log files into structured dashboard data."""
    data = DashboardData()
    pending: dict[int, Trade] = {}  # ticket → Trade (open, awaiting close)

    for path in log_files:
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.rstrip()
                m_line = _LINE_RE.match(line)
                if not m_line:
                    continue

                ts = _parse_timestamp(m_line.group(1))
                msg = m_line.group(3)

                # ── Bar log ──
                m = _BAR_RE.search(msg)
                if m:
                    data.bar_count = max(data.bar_count, int(m.group(1)))
                    regime = m.group(2)
                    action_name = m.group(4)
                    balance = float(m.group(6))

                    data.regime_bars[regime] += 1
                    data.action_counts[action_name] += 1
                    data.balance_series.append((ts, balance))

                    if data.starting_balance == 0.0:
                        data.starting_balance = balance
                    data.ending_balance = balance
                    continue

                # ── Trade opened (from RiskManager) ──
                m = _OPEN_RE.search(msg)
                if m:
                    ticket = int(m.group(1))
                    trade = Trade(
                        ticket=ticket,
                        direction=m.group(2),
                        lot=float(m.group(3)),
                        entry_price=float(m.group(4)),
                        sl=float(m.group(5)),
                        tp=float(m.group(6)),
                        regime=m.group(7),
                        open_time=ts,
                    )
                    pending[ticket] = trade
                    continue

                # ── Trade closed by ExecutionEngine ──
                m = _CLOSE_ENGINE_RE.search(msg)
                if m:
                    ticket = int(m.group(1))
                    reason = m.group(2)
                    direction = m.group(3)  # None for old logs
                    regime = m.group(4)     # None for old logs
                    close_price = float(m.group(5))
                    pnl = float(m.group(6))

                    if ticket in pending:
                        trade = pending.pop(ticket)
                        trade.close_reason = reason
                        trade.close_price = close_price
                        trade.pnl = pnl
                        trade.close_time = ts
                        data.trades.append(trade)
                    else:
                        data.trades.append(Trade(
                            ticket=ticket,
                            direction=direction or "?",
                            regime=regime or "?",
                            lot=0.0,
                            entry_price=0.0,
                            sl=0.0,
                            tp=0.0,
                            close_reason=reason,
                            close_price=close_price,
                            pnl=pnl,
                            close_time=ts,
                        ))
                    continue

                # ── Trade closed by broker (TP/SL hit) ──
                m = _CLOSE_BROKER_RE.search(msg)
                if m:
                    ticket = int(m.group(1))
                    reason = m.group(2)
                    direction = m.group(3)  # None for old logs
                    regime = m.group(4)     # None for old logs
                    pnl = float(m.group(5))

                    if ticket in pending:
                        trade = pending.pop(ticket)
                        trade.close_reason = reason
                        trade.pnl = pnl
                        trade.close_time = ts
                        data.trades.append(trade)
                    else:
                        data.trades.append(Trade(
                            ticket=ticket,
                            direction=direction or "?",
                            regime=regime or "?",
                            lot=0.0,
                            entry_price=0.0,
                            sl=0.0,
                            tp=0.0,
                            close_reason=reason,
                            pnl=pnl,
                            close_time=ts,
                        ))
                    continue

                # ── Account snapshot (startup) ──
                m = _ACCOUNT_RE.search(msg)
                if m:
                    bal = float(m.group(1))
                    if data.starting_balance == 0.0:
                        data.starting_balance = bal
                    data.balance_series.append((ts, bal))
                    continue

                # ── Regime shift ──
                if _REGIME_SHIFT_RE.search(msg):
                    data.regime_shifts += 1
                    continue

                # ── Telemetry (inference probabilities + critic value) ──
                m = _TELEMETRY_RE.search(msg)
                if m:
                    regime_t = m.group(1)
                    cv_str = m.group(2)
                    conf = float(m.group(3))
                    data.confidences.append(conf)
                    data.regime_confidences[regime_t].append(conf)
                    if cv_str != "N/A":
                        try:
                            data.critic_values.append(float(cv_str))
                        except ValueError:
                            pass
                    continue

                # ── Anomaly detection ──
                if _ANOMALY_RE.search(msg):
                    data.anomaly_count += 1
                    continue

                # ── Confidence gate override ──
                if _CONFIDENCE_GATE_RE.search(msg):
                    data.confidence_gate_count += 1
                    continue

                # ── V3.x defense systems ──
                if _VKR_GATE_RE.search(msg):
                    data.vkr_gate_count += 1
                    continue
                if _GRACE_PERIOD_RE.search(msg):
                    data.grace_period_count += 1
                    continue
                if _PHANTOM_FIRE_RE.search(msg):
                    data.phantom_fire_count += 1
                    continue
                if _PYRAMID_RE.search(msg):
                    data.pyramid_count += 1
                    continue
                if _BREAK_EVEN_RE.search(msg):
                    data.break_even_count += 1
                    continue
                if _PARTIAL_CLOSE_RE.search(msg):
                    data.partial_close_count += 1
                    continue
                m = _TIME_DECAY_RE.search(msg)
                if m:
                    data.time_decay_count += 1
                    if m.group(1) == "SQUEEZE":
                        data.time_decay_squeeze_count += 1
                    continue
                if _SPREAD_GATE_RE.search(msg):
                    data.spread_gate_count += 1
                    continue
                if _COOLDOWN_RE.search(msg):
                    data.cooldown_count += 1
                    continue

                # ── V4.0 defense systems ──
                if _PENALTY_BOX_RE.search(msg):
                    data.penalty_box_count += 1
                    continue
                if _VOLUME_GATE_RE.search(msg):
                    data.volume_gate_count += 1
                    continue

                # ── V5.0 HFT systems ──
                if _LIMIT_ORDER_PLACED_RE.search(msg):
                    data.limit_order_placed_count += 1
                    continue
                if _LIMIT_ORDER_FILLED_RE.search(msg):
                    data.limit_order_filled_count += 1
                    continue
                if _LIMIT_ORDER_EXPIRED_RE.search(msg):
                    data.limit_order_expired_count += 1
                    continue
                if _ELASTIC_TP_RE.search(msg):
                    data.elastic_tp_count += 1
                    continue
                if _SUBBAR_SCAN_RE.search(msg):
                    data.stale_limit_cancelled_count += 1
                    continue

    return data


# ══════════════════════════════════════════════
# Metric Calculations
# ══════════════════════════════════════════════
def _win_rate(trades: Sequence[Trade]) -> tuple[float, int, int]:
    """Return (win_rate_pct, wins, total)."""
    if not trades:
        return 0.0, 0, 0
    wins = sum(1 for t in trades if t.pnl > 0)
    return (wins / len(trades)) * 100, wins, len(trades)


def _profit_factor(trades: Sequence[Trade]) -> float:
    """Gross Profit / Gross Loss.  Returns inf if no losses."""
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _max_drawdown(balance_series: Sequence[tuple[datetime, float]]) -> tuple[float, float]:
    """Return (max_dd_amount, max_dd_pct) from the balance curve."""
    if not balance_series:
        return 0.0, 0.0
    peak = balance_series[0][1]
    max_dd = 0.0
    max_dd_pct = 0.0
    for _, bal in balance_series:
        if bal > peak:
            peak = bal
        dd = peak - bal
        dd_pct = (dd / peak * 100) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


def _expectancy(trades: Sequence[Trade]) -> float:
    """Average PnL per trade."""
    if not trades:
        return 0.0
    return sum(t.pnl for t in trades) / len(trades)


def _avg_win_loss(trades: Sequence[Trade]) -> tuple[float, float]:
    """Return (avg_win, avg_loss)."""
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    return avg_w, avg_l


def _payoff_ratio(avg_win: float, avg_loss: float) -> float:
    """Avg Win / |Avg Loss|.  Higher = better risk/reward per trade."""
    if avg_loss == 0:
        return float("inf") if avg_win > 0 else 0.0
    return avg_win / abs(avg_loss)


def _sharpe_ratio(
    trades: Sequence[Trade],
    risk_free_annual: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sharpe Ratio from per-trade PnL returns.

    ``SR = (mean_return - Rf_per_trade) / std_return * sqrt(N_per_year)``

    Uses trade count as proxy for periods.  Returns 0.0 when data is
    insufficient (< 2 trades) or zero variance.
    """
    if len(trades) < 2:
        return 0.0
    returns = [t.pnl for t in trades]
    n = len(returns)
    mean_r = sum(returns) / n
    var = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(var) if var > 0 else 0.0
    if std_r == 0:
        return 0.0
    rf_per_trade = risk_free_annual / periods_per_year
    return ((mean_r - rf_per_trade) / std_r) * math.sqrt(periods_per_year)


def _sortino_ratio(
    trades: Sequence[Trade],
    risk_free_annual: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sortino Ratio — penalizes downside deviation only.

    ``Sortino = (mean_return - Rf) / downside_std * sqrt(N)``
    """
    if len(trades) < 2:
        return 0.0
    returns = [t.pnl for t in trades]
    n = len(returns)
    mean_r = sum(returns) / n
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf") if mean_r > 0 else 0.0
    down_var = sum(r ** 2 for r in downside) / len(downside)
    down_std = math.sqrt(down_var) if down_var > 0 else 0.0
    if down_std == 0:
        return 0.0
    rf_per_trade = risk_free_annual / periods_per_year
    return ((mean_r - rf_per_trade) / down_std) * math.sqrt(periods_per_year)


def _calmar_ratio(
    balance_series: Sequence[tuple[datetime, float]],
) -> float:
    """Calmar Ratio = Annualized Return % / Max Drawdown %.

    Uses first/last balance + elapsed time to annualize.
    Returns 0.0 if period < 1 day or MDD = 0.
    """
    if len(balance_series) < 2:
        return 0.0
    first_ts, first_bal = balance_series[0]
    last_ts, last_bal = balance_series[-1]
    elapsed_days = (last_ts - first_ts).total_seconds() / 86400
    if elapsed_days < 1 or first_bal <= 0:
        return 0.0
    total_return = (last_bal - first_bal) / first_bal
    annual_return = total_return * (365.0 / elapsed_days)
    _, mdd_pct = _max_drawdown(balance_series)
    if mdd_pct == 0:
        return float("inf") if annual_return > 0 else 0.0
    return (annual_return * 100) / mdd_pct


# ══════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════
_W = 80


def _header(title: str) -> None:
    print(f"\n{'═' * _W}")
    print(f"  {title}")
    print(f"{'═' * _W}")


def _section(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(1, _W - len(title) - 6)}")


def _kv(label: str, value: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{label:<28} {value}")


def display_dashboard(data: DashboardData) -> None:
    """Print the full performance dashboard."""

    trades = data.trades
    total_pnl = sum(t.pnl for t in trades)
    wr, wins, total = _win_rate(trades)
    pf = _profit_factor(trades)
    dd_amt, dd_pct = _max_drawdown(data.balance_series)
    avg_w, avg_l = _avg_win_loss(trades)
    exp = _expectancy(trades)

    # ── Header ──
    _header("APEX PREDATOR V5.0 — LIVE PERFORMANCE DASHBOARD")

    if data.balance_series:
        first_ts = data.balance_series[0][0].strftime("%Y-%m-%d %H:%M")
        last_ts = data.balance_series[-1][0].strftime("%Y-%m-%d %H:%M")
        _kv("Period", f"{first_ts}  →  {last_ts}")

    _kv("Bars Processed", f"{data.bar_count:,}")
    _kv("Regime Shifts", f"{data.regime_shifts:,}")

    # ── Account ──
    _section("ACCOUNT")
    _kv("Starting Balance", f"${data.starting_balance:,.2f}")
    _kv("Ending Balance", f"${data.ending_balance:,.2f}")
    _kv("Net P&L", f"${total_pnl:+,.2f}")
    _kv("Return",
        f"{((data.ending_balance - data.starting_balance) / data.starting_balance * 100):+.2f}%"
        if data.starting_balance > 0 else "N/A")

    # ── Overall Trade Metrics ──
    _section("OVERALL TRADE METRICS")
    _kv("Total Trades", f"{total}")
    _kv("Wins / Losses", f"{wins} / {total - wins}")
    _kv("Win Rate", f"{wr:.1f}%")
    _kv("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")
    _kv("Expectancy (avg PnL/trade)", f"${exp:+,.2f}")
    _kv("Avg Win", f"${avg_w:+,.2f}")
    _kv("Avg Loss", f"${avg_l:+,.2f}")
    _kv("Max Drawdown", f"${dd_amt:,.2f}  ({dd_pct:.2f}%)")

    # ── Institutional Risk Metrics ──
    _section("INSTITUTIONAL RISK METRICS")

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pr = _payoff_ratio(avg_w, avg_l)
    sharpe = _sharpe_ratio(trades)
    sortino = _sortino_ratio(trades)
    calmar = _calmar_ratio(data.balance_series)

    _kv("Gross Profit", f"${gross_profit:,.2f}")
    _kv("Gross Loss", f"${gross_loss:,.2f}")
    _kv("Profit Factor",
        f"{pf:.2f}" if pf != float("inf") else "∞  (no losses)")
    _kv("Payoff Ratio (Win/Loss)",
        f"{pr:.2f}" if pr != float("inf") else "∞  (no losses)")
    _kv("Expectancy (per trade)", f"${exp:+,.2f}")
    _kv("Sharpe Ratio (ann.)",
        f"{sharpe:+.2f}" if trades else "N/A")
    _kv("Sortino Ratio (ann.)",
        f"{sortino:+.2f}" if sortino != float("inf") else "∞  (no downside)"
        if trades else "N/A")
    _kv("Calmar Ratio (ann.)",
        f"{calmar:.2f}" if calmar != float("inf") else "∞  (no drawdown)"
        if data.balance_series else "N/A")

    print()
    _kv("Interpretation Guide", "")
    print(f"      {'Metric':<22} {'Poor':>8} {'OK':>8} {'Good':>8} {'Elite':>8}")
    print(f"      {'─' * 54}")
    print(f"      {'Profit Factor':<22} {'< 1.0':>8} {'1.0-1.4':>8} {'1.5-2.0':>8} {'> 2.0':>8}")
    print(f"      {'Sharpe Ratio':<22} {'< 0.5':>8} {'0.5-1.0':>8} {'1.0-2.0':>8} {'> 2.0':>8}")
    print(f"      {'Sortino Ratio':<22} {'< 1.0':>8} {'1.0-1.5':>8} {'1.5-3.0':>8} {'> 3.0':>8}")
    print(f"      {'Calmar Ratio':<22} {'< 0.5':>8} {'0.5-1.0':>8} {'1.0-3.0':>8} {'> 3.0':>8}")
    print(f"      {'Payoff Ratio':<22} {'< 1.0':>8} {'1.0-1.5':>8} {'1.5-2.5':>8} {'> 2.5':>8}")

    # ── Win Rate by Agent (Regime) ──
    _section("WIN RATE BY AGENT")

    _REGIME_AGENT: dict[str, str] = {
        "TRENDING_UP": "Bull Rider",
        "TRENDING_DOWN": "Bear Hunter",
        "MEAN_REVERTING": "Range Sniper",
        "HIGH_VOLATILITY": "Vol Assassin",
    }

    regime_trades: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        regime_trades[t.regime].append(t)

    print()
    print(f"    {'Agent':<22} {'Trades':>7} {'Wins':>6} {'WR%':>7} "
          f"{'PF':>7} {'Net P&L':>12} {'Avg P&L':>10}")
    print(f"    {'─' * 72}")

    for regime in sorted(regime_trades.keys()):
        rt = regime_trades[regime]
        r_wr, r_w, r_t = _win_rate(rt)
        r_pf = _profit_factor(rt)
        r_pnl = sum(t.pnl for t in rt)
        r_exp = _expectancy(rt)
        agent_name = _REGIME_AGENT.get(regime, regime)

        pf_str = f"{r_pf:.2f}" if r_pf != float("inf") else "∞"
        print(
            f"    {agent_name:<22} {r_t:>7} {r_w:>6} {r_wr:>6.1f}% "
            f"{pf_str:>7} ${r_pnl:>+10,.2f} ${r_exp:>+8,.2f}"
        )

    # ── Close Reason Breakdown ──
    _section("CLOSE REASON BREAKDOWN")

    reason_counts: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        reason_counts[t.close_reason or "UNKNOWN"].append(t)

    print()
    print(f"    {'Reason':<24} {'Count':>7} {'Wins':>6} {'WR%':>7} {'Net P&L':>12}")
    print(f"    {'─' * 56}")

    for reason in sorted(reason_counts.keys()):
        rt = reason_counts[reason]
        r_wr, r_w, r_t = _win_rate(rt)
        r_pnl = sum(t.pnl for t in rt)
        print(f"    {reason:<24} {r_t:>7} {r_w:>6} {r_wr:>6.1f}% ${r_pnl:>+10,.2f}")

    # ── Action Distribution ──
    _section("ACTION DISTRIBUTION")
    total_actions = sum(data.action_counts.values()) or 1
    for action in sorted(data.action_counts.keys()):
        cnt = data.action_counts[action]
        pct = cnt / total_actions * 100
        bar = "█" * int(pct / 2)
        _kv(action, f"{cnt:>6}  ({pct:5.1f}%)  {bar}")

    # ── Regime Distribution ──
    _section("REGIME BAR DISTRIBUTION")
    total_bars_tracked = sum(data.regime_bars.values()) or 1
    for regime in sorted(data.regime_bars.keys()):
        cnt = data.regime_bars[regime]
        pct = cnt / total_bars_tracked * 100
        agent_name = _REGIME_AGENT.get(regime, regime)
        bar = "█" * int(pct / 2)
        _kv(f"{agent_name} ({regime})", f"{cnt:>6}  ({pct:5.1f}%)  {bar}")

    # ── AI Inference Telemetry ──
    if data.confidences:
        _section("AI INFERENCE TELEMETRY")
        n_samples = len(data.confidences)
        avg_conf = sum(data.confidences) / n_samples
        min_conf = min(data.confidences)
        max_conf = max(data.confidences)

        _kv("Telemetry Samples", f"{n_samples:,}")
        _kv("Avg Confidence", f"{avg_conf:.1f}%")
        _kv("Min / Max Confidence", f"{min_conf:.1f}% / {max_conf:.1f}%")

        if data.critic_values:
            avg_cv = sum(data.critic_values) / len(data.critic_values)
            min_cv = min(data.critic_values)
            max_cv = max(data.critic_values)
            _kv("Avg Critic Value", f"{avg_cv:+.3f}")
            _kv("Min / Max Critic Value", f"{min_cv:+.3f} / {max_cv:+.3f}")

        _kv("Anomalies Detected", f"{data.anomaly_count}")
        _kv("Confidence Gate Overrides", f"{data.confidence_gate_count}")

        # Per-regime average confidence
        if data.regime_confidences:
            print()
            print(f"    {'Agent':<22} {'Samples':>8} {'Avg Conf':>10} "
                  f"{'Min':>8} {'Max':>8}")
            print(f"    {'─' * 56}")
            for regime in sorted(data.regime_confidences.keys()):
                rc = data.regime_confidences[regime]
                agent_name = _REGIME_AGENT.get(regime, regime)
                r_avg = sum(rc) / len(rc)
                r_min = min(rc)
                r_max = max(rc)
                print(
                    f"    {agent_name:<22} {len(rc):>8} "
                    f"{r_avg:>9.1f}% {r_min:>7.1f}% {r_max:>7.1f}%"
                )

    # ── V3.x Defense Systems ──
    v3_total = (
        data.break_even_count + data.partial_close_count
        + data.phantom_fire_count + data.pyramid_count
        + data.time_decay_count + data.vkr_gate_count
        + data.grace_period_count + data.spread_gate_count
        + data.cooldown_count + data.penalty_box_count
        + data.volume_gate_count
        + data.limit_order_placed_count + data.limit_order_filled_count
        + data.limit_order_expired_count + data.elastic_tp_count
    )
    if v3_total > 0:
        _section("V3.x / V4.0 / V5.0 DEFENSE & HFT SYSTEMS")
        _kv("Break-Even Activations", f"{data.break_even_count}")
        _kv("Partial Closes (50%)", f"{data.partial_close_count}")
        _kv("Phantom Spoofer Fires", f"{data.phantom_fire_count}")
        _kv("Pyramid Positions", f"{data.pyramid_count}")
        _kv("Time-Decay (total)", f"{data.time_decay_count}")
        _kv("  └─ SL Squeeze (V4.0)", f"{data.time_decay_squeeze_count}")
        _kv("VKR Gate Blocks", f"{data.vkr_gate_count}")
        _kv("Grace Period Shields", f"{data.grace_period_count}")
        _kv("Spread Gate Skips", f"{data.spread_gate_count}")
        _kv("Bar Cooldown Blocks", f"{data.cooldown_count}")
        _kv("Penalty Box Lockouts", f"{data.penalty_box_count}")
        _kv("Volume Gate Rejects", f"{data.volume_gate_count}")
        # V5.0 HFT
        _kv("Limit Orders Placed", f"{data.limit_order_placed_count}")
        _kv("  └─ Filled (0 slip)", f"{data.limit_order_filled_count}")
        _kv("  └─ Expired/Cancelled", f"{data.limit_order_expired_count}")
        _kv("Elastic TP Expansions", f"{data.elastic_tp_count}")
        _kv("Stale Limits Cancelled", f"{data.stale_limit_cancelled_count}")

    # ── Recent Trades (last 10) ──
    if trades:
        _section(f"RECENT TRADES (last {min(10, len(trades))})")
        print()
        print(f"    {'Ticket':>10} {'Dir':<5} {'Regime':<18} {'Lot':>5} "
              f"{'Entry':>10} {'Close':>10} {'PnL':>10} {'Reason':<16}")
        print(f"    {'─' * 90}")

        for t in trades[-10:]:
            print(
                f"    {t.ticket:>10} {t.direction:<5} {t.regime:<18} "
                f"{t.lot:>5.2f} {t.entry_price:>10.2f} {t.close_price:>10.2f} "
                f"${t.pnl:>+8.2f} {t.close_reason:<16}"
            )

    print(f"\n{'═' * _W}")
    print(f"  Log directory: {_LOG_DIR.resolve()}")
    print(f"{'═' * _W}\n")


# ══════════════════════════════════════════════
# CSV Export
# ══════════════════════════════════════════════
def export_csv(trades: Sequence[Trade], path: str) -> None:
    """Write all parsed trades to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticket", "direction", "regime", "lot",
            "entry_price", "sl", "tp",
            "close_price", "close_reason", "pnl",
            "open_time", "close_time",
        ])
        for t in trades:
            writer.writerow([
                t.ticket, t.direction, t.regime, t.lot,
                t.entry_price, t.sl, t.tp,
                t.close_price, t.close_reason, t.pnl,
                t.open_time.isoformat() if t.open_time else "",
                t.close_time.isoformat() if t.close_time else "",
            ])
    print(f"Exported {len(trades)} trades → {path}")


# ══════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apex Predator V3.5 — Live Performance Dashboard",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Filter by date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Show only last N days",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export trades to CSV file",
    )
    args = parser.parse_args()

    log_files = _collect_log_files(date_filter=args.date, last_n_days=args.last)
    if not log_files:
        print(f"No log files found in {_LOG_DIR.resolve()}")
        sys.exit(1)

    print(f"Parsing {len(log_files)} log file(s)...")
    data = parse_logs(log_files)

    if not data.trades and data.bar_count == 0:
        print("No trading data found in logs.")
        sys.exit(0)

    display_dashboard(data)

    if args.csv:
        export_csv(data.trades, args.csv)


if __name__ == "__main__":
    main()
