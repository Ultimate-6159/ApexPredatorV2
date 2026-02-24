"""
Apex Predator V3.0 — Live Execution Engine (The 4D Paradigm)
Real-time automated trading on MetaTrader 5 using ensemble RL agents.

Architecture:
  PerceptionEngine → MetaRouter → Agent Dispatch → ExecutionEngine → MT5

SRS Requirements Implemented:
  1. System Initialization — Load 4 PPO models + obs_stats into RAM at startup
  2. State Normalization — Z-Score before predict (agent-specific mean/std)
  3. Meta-Router Loop — On-bar-close trigger, regime detection, agent dispatch
  4. Regime-Shift Protocol — Force close all positions on regime change
  5. Action Translation — HOLD=close/pass, BUY/SELL=open (position-aware)
  6. Risk Management — Max 1 position, ATR SL/TP, slippage protection
  7. Logging & Fail-Safes — MT5 auto-reconnect, daily rotating log

V3 Upgrades:
   8. ATR-Based Dynamic Trailing Stop — adapts to volatility
   9. News Filter — Forex Factory calendar, forces HIGH_VOLATILITY before red events
  10. Inference Telemetry — action probabilities, critic value, anomaly detection
  11. Infinite Radar (V2.17) — intra-bar re-prediction keeps cache alive
  12. HOLD Semantics Fix (V2.18) — HOLD preserves open position

V3.0 — The 4D Paradigm:
  13. Virtual Time-Decay Shield — force close after 90s without break-even (no broker spam)
  14. Smart Phantom Spoofer — dual-trigger: Phantom Sweep + Momentum Bounce
  15. Risk-Free Pyramiding — 2nd position only when 1st at break-even (zero risk)

Usage:
    python -m scripts.run_live [--timeframe M5] [--symbol XAUUSDm]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from config import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    AGENT_ACTION_MAP,
    ATR_PERIOD,
    BREAK_EVEN_ACTIVATION_ATR,
    BREAK_EVEN_BUFFER_POINTS,
    CONFIDENCE_GATE_PCT,
    ENABLE_BREAK_EVEN,
    ENABLE_PARTIAL_CLOSE,
    ENABLE_PYRAMIDING,
    MAGIC_NUMBER,
    MAX_POSITIONS,
    MODEL_DIR,
    MOMENTUM_BOUNCE_ATR,
    MOMENTUM_WINDOW_SEC,
    NEWS_BLACKOUT_MINUTES,
    NEWS_CACHE_HOURS,
    NEWS_CURRENCIES,
    NEWS_FILTER_ENABLED,
    OBS_CLIP_RANGE,
    PARTIAL_CLOSE_ACTIVATION_ATR,
    PARTIAL_CLOSE_VOLUME_PCT,
    PHANTOM_SWEEP_ATR,
    SYMBOL,
    TIMEFRAME_NAME,
    TRADE_LIFESPAN_SEC,
    TRAILING_ACTIVATION_ATR,
    TRAILING_DRAWDOWN_ATR,
    TRAINING_LOG_DIR,
    Regime,
)
from core.execution_engine import ExecutionEngine
from core.meta_router import MetaRouter
from core.news_filter import NewsFilter
from core.perception_engine import PerceptionEngine
from core.risk_manager import RiskManager

# ══════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════
_LIVE_LOG_DIR = Path("logs/live")
_POLL_INTERVAL_SEC: int = 10
_MT5_RECONNECT_SEC: int = 5
_MT5_MAX_RETRIES: int = 5

_ACTION_NAMES: dict[int, str] = {
    ACTION_HOLD: "HOLD",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL",
}


# ══════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════
def _setup_logging() -> logging.Logger:
    """Configure console + daily rotating log file."""
    _LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("apex_live")
    log.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    log.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        _LIVE_LOG_DIR / "live_trading.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    log.addHandler(file_handler)

    return log


# ══════════════════════════════════════════════
# Agent Loader  (SRS §1 — System Initialization)
# ══════════════════════════════════════════════
def _find_obs_stats(regime: Regime) -> dict[str, list[float]] | None:
    """Locate obs_stats.json for a regime.

    Search order:
      1. models/{regime_lower}_obs_stats.json  (explicit placement)
      2. Latest session in logs/training/{regime_lower}/*/obs_stats.json
    """
    regime_lower = regime.value.lower()

    # 1. Explicit file next to model
    explicit = Path(MODEL_DIR) / f"{regime_lower}_obs_stats.json"
    if explicit.exists():
        with open(explicit, encoding="utf-8") as f:
            return json.load(f)

    # 2. Latest training session
    log_base = Path(TRAINING_LOG_DIR) / regime_lower
    if log_base.exists():
        sessions = sorted(
            (d for d in log_base.iterdir() if d.is_dir()),
            key=lambda p: p.name,
        )
        for session_dir in reversed(sessions):
            stats_file = session_dir / "obs_stats.json"
            if stats_file.exists():
                with open(stats_file, encoding="utf-8") as f:
                    return json.load(f)

    return None


def _load_all_agents(
    log: logging.Logger,
) -> dict[Regime, dict[str, Any]]:
    """Load all 4 PPO models and their obs_stats into RAM (SRS §1)."""
    agents: dict[Regime, dict[str, Any]] = {}

    for regime in Regime:
        model_path = os.path.join(MODEL_DIR, f"{regime.value.lower()}.zip")

        if not os.path.exists(model_path):
            log.critical("Model not found: %s — train first!", model_path)
            sys.exit(1)

        model = PPO.load(model_path)
        stats = _find_obs_stats(regime)

        if stats is None:
            log.warning(
                "obs_stats.json not found for %s — Z-Score normalization disabled",
                regime.value,
            )
        else:
            log.info(
                "Loaded obs_stats for %s (features=%d)",
                regime.value,
                len(stats.get("mean", [])),
            )

        agents[regime] = {"model": model, "stats": stats}
        log.info("✓ Loaded %s ← %s", regime.value, model_path)

    return agents


# ══════════════════════════════════════════════
# State Normalization  (SRS §2)
# ══════════════════════════════════════════════
def _normalize_obs(
    raw_obs: np.ndarray,
    stats: dict[str, list[float]] | None,
) -> np.ndarray:
    """Z-Score normalize: ``obs_scaled = (raw - mean) / (std + 1e-8)``."""
    if stats is None:
        return raw_obs

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    std = np.maximum(std, 0.01)
    return (raw_obs - mean) / (std + 1e-8)


# ══════════════════════════════════════════════
# Live Engine
# ══════════════════════════════════════════════
class LiveEngine:
    """Core live trading loop implementing all SRS requirements."""

    def __init__(
        self,
        timeframe: str = TIMEFRAME_NAME,
        symbol: str = SYMBOL,
    ) -> None:
        self.log = _setup_logging()
        self.symbol = symbol
        self.timeframe = timeframe

        # Core components
        self.perception = PerceptionEngine(symbol=symbol, timeframe=timeframe)
        self.risk = RiskManager()
        self.executor = ExecutionEngine(risk_manager=self.risk, symbol=symbol)
        self.router = MetaRouter()

        # State
        self.agents: dict[Regime, dict[str, Any]] = {}
        self._prev_regime: Regime | None = None
        self._bar_count: int = 0
        self._last_bar_time: int = 0
        self._current_atr: float = 0.0

        # Trailing stop state (ATR-based dynamic)
        self._trailing_ticket: int | None = None
        self._highest_profit_points: float = 0.0

        # Profit locking state (Break-Even + Partial Close)
        self._break_even_done: bool = False
        self._partial_close_done: bool = False

        # Predictive Cache (V2.14 — Velocity-Aware intra-bar trigger)
        self._predictive_cache: dict[str, Any] | None = None

        # Infinite Radar (V2.17 — continuous intra-bar re-prediction)
        self._radar_active: bool = False
        self._last_repredict_time: float = 0.0

        # Elastic Cooldown Reload (V2.14 — replaces Win-Streak)
        self._swing_extended: bool = True  # True at startup to allow first trade
        self._last_ema7: float = 0.0

        # V3.0 — Pyramid state
        self._pyramid_ticket: int | None = None

        # V3.0 — Tick history for velocity measurement
        self._tick_history: list[tuple[float, float]] = []  # [(timestamp, price), ...]

        # News filter
        self.news_filter: NewsFilter | None = None
        if NEWS_FILTER_ENABLED:
            self.news_filter = NewsFilter(
                currencies=NEWS_CURRENCIES,
                blackout_minutes=NEWS_BLACKOUT_MINUTES,
                cache_hours=NEWS_CACHE_HOURS,
            )

    # ── Public Entry Point ────────────────────
    def start(self) -> None:
        """Initialize everything and enter the main loop."""
        self.log.info("═══ Apex Predator V2 — Live Engine Starting ═══")

        # §1  Load all 4 agents into RAM
        self.agents = _load_all_agents(self.log)
        self.log.info("All 4 agents loaded into RAM")

        # Connect to MT5
        if not self._ensure_connection():
            self.log.critical("Cannot establish MT5 connection — aborting")
            sys.exit(1)

        # Validate symbol
        info = mt5.symbol_info(self.symbol)
        if info is None or not info.visible:
            self.log.critical("Symbol %s not available on broker", self.symbol)
            sys.exit(1)
        self.log.info(
            "Symbol %s ready  (spread=%d, digits=%d)",
            self.symbol,
            info.spread,
            info.digits,
        )

        # Account snapshot
        acct = self.perception.get_account_info()
        self.log.info(
            "Account: balance=%.2f  equity=%.2f  margin_free=%.2f",
            acct["balance"],
            acct["equity"],
            acct["margin_free"],
        )

        # Enter main loop
        self.log.info(
            "Entering main loop (TF=%s, poll=%ds)",
            self.timeframe,
            _POLL_INTERVAL_SEC,
        )
        self._main_loop()

    # ── MT5 Connection  (SRS §7) ──────────────
    def _ensure_connection(self) -> bool:
        """Connect to MT5 with retry logic (5-second interval)."""
        for attempt in range(1, _MT5_MAX_RETRIES + 1):
            if self.perception.connect():
                return True
            self.log.warning(
                "MT5 connection attempt %d/%d failed — retrying in %ds",
                attempt,
                _MT5_MAX_RETRIES,
                _MT5_RECONNECT_SEC,
            )
            time.sleep(_MT5_RECONNECT_SEC)
        return False

    def _check_connection(self) -> bool:
        """Verify MT5 is alive; auto-reconnect if lost."""
        term = mt5.terminal_info()
        if term is not None and term.connected:
            return True
        self.log.warning("MT5 connection lost — attempting reconnect")
        self.perception._connected = False
        return self._ensure_connection()

    # ── Bar Detection  (SRS §3) ───────────────
    def _wait_for_new_bar(self) -> bool:
        """Block until a new bar closes. Returns False on unrecoverable error.

        V2.15: Tick-level risk management (Break-Even / Trailing / Sync)
               runs every 50 ms while a position is open or cache is active.
               HFT Re-entry: if a position closes intra-bar, force an
               immediate AI re-evaluation by returning True.
        """
        while True:
            if not self._check_connection():
                return False

            # --- V2.15: Intra-bar tick-level risk management ---
            was_open = self.risk.has_open_trade
            self._sync_position_state()
            self._check_profit_locking()
            self._check_trailing_stop()
            self._check_time_decay()  # V3.0: Virtual Time-Decay Shield

            # V3.0: Record tick for velocity measurement
            _tick = mt5.symbol_info_tick(self.symbol)
            if _tick is not None:
                self._record_tick(float(_tick.bid))

            # HFT Re-entry: position just closed intra-bar → force AI now
            if was_open and not self.risk.has_open_trade:
                self.log.info(
                    "⚡ INTRA-BAR CLOSE: Forcing AI re-evaluation!"
                )
                return True
            # --------------------------------------------------

            rates = mt5.copy_rates_from_pos(
                self.symbol, self.perception.timeframe, 0, 1
            )
            if rates is None or len(rates) == 0:
                time.sleep(1)
                continue

            bar_time = int(rates[0]["time"])
            if bar_time != self._last_bar_time:
                self._last_bar_time = bar_time
                self._predictive_cache = None  # Clear stale cache on new bar
                self._radar_active = False     # V2.17: reset radar on new bar
                return True

            # Intra-bar monitoring (V2.14)
            self._update_swing_tracking()

            if self._predictive_cache is not None:
                self._check_predictive_cache()

            # V2.17 Infinite Radar: re-predict when cache expired and flat
            if (
                self._predictive_cache is None
                and self._radar_active
                and not self.risk.has_open_trade
            ):
                self._intra_bar_re_predict()

            # Dynamic polling: 50 ms HFT when active, 1 s when idle
            if self._predictive_cache is not None or self.risk.has_open_trade:
                time.sleep(0.05)
            else:
                time.sleep(1)

    # ── Main Loop ─────────────────────────────
    def _main_loop(self) -> None:
        """Core event loop — fires once per bar close."""
        try:
            # Initialize bar timestamp (skip the first detection)
            rates = mt5.copy_rates_from_pos(
                self.symbol, self.perception.timeframe, 0, 1
            )
            if rates is not None and len(rates) > 0:
                self._last_bar_time = int(rates[0]["time"])

            self.log.info("Waiting for first bar close...")

            while True:
                if not self._wait_for_new_bar():
                    continue

                self._bar_count += 1
                self._on_bar_close()

        except KeyboardInterrupt:
            self.log.info("Shutdown signal received (Ctrl+C)")
        finally:
            self._shutdown()

    # ── Per-Bar Processing ────────────────────
    def _on_bar_close(self) -> None:
        """Process a single bar close event (SRS §3)."""
        try:
            # 1. Sync position state with broker (detect TP/SL hits)
            self._sync_position_state()

            # 2. Fetch fresh OHLCV + features (needed for ATR)
            ohlcv = self.perception.fetch_ohlcv()
            features = self.perception.compute_features(ohlcv)
            latest_row = features.iloc[-1]

            # 3. Compute & store ATR (used by trailing stop + dispatch)
            hl_range = (ohlcv["High"] - ohlcv["Low"]).rolling(ATR_PERIOD).mean()
            self._current_atr = float(hl_range.iloc[-1]) if not hl_range.empty else 0.0

            # 3c. Cache EMA7 for ribbon filter + elastic cooldown
            if len(ohlcv) >= 7:
                self._last_ema7 = float(
                    ohlcv["Close"].ewm(span=7, adjust=False).mean().iloc[-1]
                )

            # 3a. Spread/ATR Normalization Gate — skip bar if ATR too small vs spread
            sym_info = self.perception.get_symbol_info()
            _point = sym_info["point"]
            if _point > 0 and self._current_atr > 0:
                current_atr_points = self._current_atr / _point
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is not None:
                    current_spread_points = (tick.ask - tick.bid) / _point
                    if current_atr_points < current_spread_points * 1.5:
                        self.log.info(
                            "⏸️ SPREAD GATE: ATR=%.0f pts < 1.5×Spread=%.0f pts "
                            "— skipping bar",
                            current_atr_points,
                            current_spread_points,
                        )
                        return

            # 3b. Profit locking (Break-Even + Partial Close)
            self._check_profit_locking()

            # 4. Trailing stop check (ATR-based, overrides AI)
            self._check_trailing_stop()

            # 5. Detect regime  (SRS §3 — Perception)
            regime = self.router.detect_regime(latest_row)

            # 5b. News filter — override regime before high-impact events
            if self.news_filter is not None:
                blackout, event_title = self.news_filter.is_blackout()
                if blackout and regime != Regime.HIGH_VOLATILITY:
                    self.log.warning(
                        "NEWS OVERRIDE: %s → HIGH_VOLATILITY (event: %s)",
                        regime.value,
                        event_title,
                    )
                    regime = Regime.HIGH_VOLATILITY

            # 6. Regime-shift protocol  (SRS §4 — The Clean Slate)
            self._handle_regime_shift(regime)

            # 7. Risk checks
            acct = self.perception.get_account_info()
            balance = acct["balance"]
            equity = acct["equity"]

            if self.risk.check_drawdown(balance):
                self.log.critical("MAX DRAWDOWN breached — FULL STOP")
                return

            if self.risk.is_halted:
                self.log.warning("Circuit breaker active — skipping bar")
                return

            # 8. Time stop check  (SRS §6)
            self.risk.update_bar(self._bar_count)
            if self.risk.should_time_stop(self._bar_count):
                self._close_and_track("TIME_STOP")

            # 9. Normalize observation  (SRS §2)
            raw_obs = latest_row.values.astype(np.float32)
            agent_data = self.agents[regime]
            obs_normalized = _normalize_obs(raw_obs, agent_data["stats"])
            obs_ready = obs_normalized.reshape(1, -1)

            # 9b. Sanitize NaN/Inf THEN hard clip (3-layer defense)
            obs_ready = np.nan_to_num(
                obs_ready, nan=0.0,
                posinf=OBS_CLIP_RANGE, neginf=-OBS_CLIP_RANGE,
            )
            obs_ready = np.clip(obs_ready, -OBS_CLIP_RANGE, OBS_CLIP_RANGE)

            # 9c. Post-clip safety net (should never trigger after nan_to_num + clip)
            if np.any(np.isnan(obs_ready)) or np.any(np.isinf(obs_ready)):
                self.log.warning(
                    "⚠️ ANOMALY: NaN/Inf survived sanitization — forcing zeros"
                )
                obs_ready = np.nan_to_num(obs_ready, nan=0.0, posinf=0.0, neginf=0.0)

            # 10. Inference Telemetry — extract probabilities + critic value
            agent_model: PPO = agent_data["model"]
            probs: np.ndarray | None = None
            critic_value: float | None = None

            try:
                obs_tensor, _ = agent_model.policy.obs_to_tensor(obs_ready)
                with torch.no_grad():
                    dist = agent_model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.cpu().numpy()[0]
                    critic_value = float(
                        agent_model.policy.predict_values(obs_tensor)
                        .cpu().numpy()[0][0]
                    )
            except Exception:
                self.log.exception("Telemetry extraction failed")

            # 11. Predict action  (SRS §5)
            action_idx, _ = agent_model.predict(
                obs_ready, deterministic=True
            )
            raw_action = int(action_idx.item())

            # 12. Map to actual action
            allowed = AGENT_ACTION_MAP[regime]
            actual_action = allowed[raw_action]

            # 12c. Confidence gate — force HOLD if AI is uncertain
            if actual_action != ACTION_HOLD and probs is not None:
                action_conf = float(probs[raw_action]) * 100
                if action_conf < CONFIDENCE_GATE_PCT:
                    self.log.warning(
                        "🛡️ CONFIDENCE GATE: %.1f%% < %.0f%% threshold "
                        "— forcing HOLD",
                        action_conf,
                        CONFIDENCE_GATE_PCT,
                    )
                    actual_action = ACTION_HOLD

            # 12b. Log telemetry (before dispatch)
            if probs is not None:
                prob_str = " | ".join(
                    f"Act{i}:{p * 100:.1f}%" for i, p in enumerate(probs)
                )
                max_prob = float(np.max(probs)) * 100
                cv_str = f"{critic_value:+.3f}" if critic_value is not None else "N/A"
                self.log.info(
                    "🧠 TELEMETRY | Regime: %s | "
                    "Critic Value: %s | Confidence: %.1f%% (%s)",
                    regime.value,
                    cv_str,
                    max_prob,
                    prob_str,
                )

            # 12d. Live-Tick Precision with Momentum Confirmation (EMA7 + EMA20)
            #   Uses real-time tick price for bounce confirmation
            #   V2.14: Deferred actions cached for intra-bar Predictive Cache
            if (
                actual_action in (ACTION_BUY, ACTION_SELL)
                and regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)
                and self._current_atr > 0
                and len(ohlcv) >= 20
            ):
                close_series = ohlcv["Close"]
                current_high = float(ohlcv["High"].iloc[-1])
                current_low = float(ohlcv["Low"].iloc[-1])
                current_ema7 = float(
                    close_series.ewm(span=7, adjust=False).mean().iloc[-1]
                )
                current_ema20 = float(
                    close_series.ewm(span=20, adjust=False).mean().iloc[-1]
                )
                ribbon_gap = abs(current_ema7 - current_ema20)
                rsi_fast_val = float(latest_row.get("rsi_fast", 50.0))

                # Live tick price for real-time bounce confirmation
                _tick = mt5.symbol_info_tick(self.symbol)
                tick_price = float(_tick.bid) if _tick is not None else float(close_series.iloc[-1])

                if actual_action == ACTION_BUY and regime == Regime.TRENDING_UP:
                    ribbon_ok = (
                        current_ema7 > current_ema20
                        and ribbon_gap >= 0.1 * self._current_atr
                        and current_low <= current_ema7 + 0.2 * self._current_atr
                        and tick_price > current_ema7
                    )
                    if rsi_fast_val >= 85.0:
                        self.log.info(
                            "🚨 RSI FILTER: BUY blocked (RSI=%.1f >= 85)",
                            rsi_fast_val,
                        )
                        actual_action = ACTION_HOLD
                    elif not ribbon_ok:
                        self.log.info(
                            "⏳ RIBBON: BUY deferred → PREDICTIVE CACHE "
                            "(EMA7=%.2f, EMA20=%.2f, gap=%.2f, "
                            "low=%.2f, tick=%.2f, ATR=%.2f)",
                            current_ema7,
                            current_ema20,
                            ribbon_gap,
                            current_low,
                            tick_price,
                            self._current_atr,
                        )
                        self._predictive_cache = {
                            "action": ACTION_BUY,
                            "regime": regime,
                            "target_price": current_ema7,
                            "timestamp": time.time(),
                            "atr": self._current_atr,
                        }
                        self._radar_active = True  # V2.17: keep radar spinning
                        actual_action = ACTION_HOLD

                elif actual_action == ACTION_SELL and regime == Regime.TRENDING_DOWN:
                    ribbon_ok = (
                        current_ema7 < current_ema20
                        and ribbon_gap >= 0.1 * self._current_atr
                        and current_high >= current_ema7 - 0.2 * self._current_atr
                        and tick_price < current_ema7
                    )
                    if rsi_fast_val <= 15.0:
                        self.log.info(
                            "🚨 RSI FILTER: SELL blocked (RSI=%.1f <= 15)",
                            rsi_fast_val,
                        )
                        actual_action = ACTION_HOLD
                    elif not ribbon_ok:
                        self.log.info(
                            "⏳ RIBBON: SELL deferred → PREDICTIVE CACHE "
                            "(EMA7=%.2f, EMA20=%.2f, gap=%.2f, "
                            "high=%.2f, tick=%.2f, ATR=%.2f)",
                            current_ema7,
                            current_ema20,
                            ribbon_gap,
                            current_high,
                            tick_price,
                            self._current_atr,
                        )
                        self._predictive_cache = {
                            "action": ACTION_SELL,
                            "regime": regime,
                            "target_price": current_ema7,
                            "timestamp": time.time(),
                            "atr": self._current_atr,
                        }
                        self._radar_active = True  # V2.17: keep radar spinning
                        actual_action = ACTION_HOLD

            # 12e. Elastic Cooldown Reload — step-trend re-entry gate (V2.14)
            #   Requires: (1) swing extension > 0.5×ATR from EMA7, then
            #             (2) pullback to < 0.2×ATR from EMA7
            if (
                actual_action in (ACTION_BUY, ACTION_SELL)
                and not self.risk.has_open_trade
                and regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)
                and self._last_ema7 > 0
                and self._current_atr > 0
            ):
                _tick_ec = mt5.symbol_info_tick(self.symbol)
                if _tick_ec is not None:
                    _tp_ec = float(_tick_ec.bid)
                    _dist_ec = abs(_tp_ec - self._last_ema7)

                    # Track swing extension
                    if _dist_ec > 0.5 * self._current_atr:
                        self._swing_extended = True

                    if not self._swing_extended:
                        self.log.info(
                            "🔒 ELASTIC: Swing not extended yet "
                            "(dist=%.2f < 0.5×ATR=%.2f)",
                            _dist_ec,
                            0.5 * self._current_atr,
                        )
                        actual_action = ACTION_HOLD
                    elif _dist_ec > 0.2 * self._current_atr:
                        self.log.info(
                            "🔒 ELASTIC: Awaiting pullback "
                            "(dist=%.2f > 0.2×ATR=%.2f)",
                            _dist_ec,
                            0.2 * self._current_atr,
                        )
                        actual_action = ACTION_HOLD
                    else:
                        self.log.info(
                            "🔄 ELASTIC RELOAD: Entry cleared "
                            "(swing OK, pullback dist=%.2f < 0.2×ATR=%.2f)",
                            _dist_ec,
                            0.2 * self._current_atr,
                        )

            # 13. Dispatch with position-aware logic  (SRS §5 + §6)
            self._dispatch_action(actual_action, regime, ohlcv, equity)

            # 14. Log  (SRS §7)
            unrealised = acct.get("profit", 0.0)
            self.log.info(
                "Bar #%d | Regime: %s | Action: %d → %s | "
                "PnL: %.2f | Balance: %.2f",
                self._bar_count,
                regime.value,
                raw_action,
                _ACTION_NAMES.get(actual_action, "?"),
                unrealised,
                balance,
            )

        except Exception:
            self.log.exception("Error processing bar #%d", self._bar_count)

    # ── Close with Elastic Cooldown reset ─────
    def _close_and_track(self, reason: str) -> bool:
        """Close trade and reset Elastic Cooldown state (V2.14).

        V3.0: Also closes pyramid position if primary is being closed
        (primary gone → pyramid should go too for safety).
        """
        trade = self.risk.open_trade
        if trade is None:
            return False

        # V3.0: If primary is closing, also close pyramid
        if self._pyramid_ticket is not None:
            self.log.info(
                "Closing pyramid #%d along with primary (%s)",
                self._pyramid_ticket, reason,
            )
            self.executor.close_all_positions(reason)
            self.risk.register_close(0.0)
            self._pyramid_ticket = None
            self._swing_extended = False
            return True

        result = self.executor.close_open_trade(reason)
        if result:
            self._swing_extended = False  # Reset for Elastic Cooldown
        return result

    # ── Predictive Cache (V3.0 — Smart Phantom Spoofer) ─
    def _check_predictive_cache(self) -> bool:
        """Dual-trigger sniper system for cached predictions.

        Trigger 1 — Phantom Sweep:
          Price overshoots target by PHANTOM_SWEEP_ATR × ATR (SL sweep),
          then returns within 0.2 × ATR of target → FIRE.

        Trigger 2 — Momentum Bounce:
          Price is within 0.5 × ATR of target and bounces back in the
          trade direction with velocity > MOMENTUM_BOUNCE_ATR × ATR
          within MOMENTUM_WINDOW_SEC seconds → FIRE (prevents missing
          strong trends).

        Guards:
          - Cache expires after 10 s
          - Cannot fire while a position is open
        """
        if self._predictive_cache is None:
            return False
        if self.risk.has_open_trade:
            self._predictive_cache = None
            return False

        cache = self._predictive_cache
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False

        tick_price = float(tick.bid)
        target = cache["target_price"]
        atr = cache["atr"]
        is_buy = cache["action"] == ACTION_BUY

        now = time.time()
        time_elapsed = now - cache["timestamp"]

        # Gate: cache freshness (expire after 10 s)
        if time_elapsed > 10.0:
            self.log.info(
                "⏰ CACHE EXPIRED: %.1fs > 10s — discarding", time_elapsed,
            )
            self._predictive_cache = None
            return False

        # ── Track Phantom Sweep state ──
        # Signed overshoot: for BUY, sweep is price dipping BELOW target;
        # for SELL, sweep is price spiking ABOVE target.
        sweep_threshold = PHANTOM_SWEEP_ATR * atr
        if is_buy:
            overshoot = target - tick_price  # positive when price below target
        else:
            overshoot = tick_price - target  # positive when price above target

        if overshoot >= sweep_threshold:
            if not cache.get("sweep_detected"):
                cache["sweep_detected"] = True
                cache["sweep_peak"] = tick_price
                self.log.info(
                    "👻 PHANTOM SWEEP detected: tick=%.2f  target=%.2f  "
                    "overshoot=%.2f (threshold=%.2f)",
                    tick_price, target, overshoot, sweep_threshold,
                )
        # Update sweep peak (deepest overshoot)
        if cache.get("sweep_detected"):
            if is_buy and tick_price < cache.get("sweep_peak", tick_price):
                cache["sweep_peak"] = tick_price
            elif not is_buy and tick_price > cache.get("sweep_peak", tick_price):
                cache["sweep_peak"] = tick_price

        fired = False
        trigger_name = ""

        # ── Trigger 1: Phantom Sweep Return ──
        if cache.get("sweep_detected"):
            tolerance = 0.2 * atr
            dist_to_target = abs(tick_price - target)
            if dist_to_target <= tolerance:
                fired = True
                trigger_name = "PHANTOM_SWEEP"

        # ── Trigger 2: Momentum Bounce ──
        if not fired:
            dist_to_target = abs(tick_price - target)
            if dist_to_target <= 0.5 * atr and time_elapsed >= 1.0:
                velocity = self._compute_tick_velocity(MOMENTUM_WINDOW_SEC)
                velocity_threshold = MOMENTUM_BOUNCE_ATR * atr
                # Velocity must be in trade direction
                if is_buy and velocity >= velocity_threshold:
                    fired = True
                    trigger_name = "MOMENTUM_BOUNCE"
                elif not is_buy and velocity <= -velocity_threshold:
                    fired = True
                    trigger_name = "MOMENTUM_BOUNCE"

        if not fired:
            return False

        # ── FIRE ──
        action_name = "BUY" if is_buy else "SELL"
        self.log.info(
            "🎯 %s FIRE: %s at tick=%.2f "
            "(target=%.2f, elapsed=%.1fs, ATR=%.2f)",
            trigger_name,
            action_name,
            tick_price,
            target,
            time_elapsed,
            atr,
        )

        acct = self.perception.get_account_info()
        self._dispatch_action(
            cache["action"], cache["regime"], pd.DataFrame(), acct["equity"],
        )
        self._predictive_cache = None
        self._radar_active = False  # V2.17: mission accomplished
        return True

    # ── Infinite Radar (V2.17 — Intra-Bar Re-Prediction) ─
    def _intra_bar_re_predict(self) -> bool:
        """Re-run AI prediction mid-candle after a cache expires.

        V2.17 Infinite Radar: When a deferred signal's cache expires (>10 s),
        fetch fresh OHLCV, re-detect regime, re-run inference, and create a
        new Predictive Cache if the AI still signals BUY/SELL with ribbon
        deferral.  The radar keeps spinning until the bar closes or the
        signal fires.

        Throttle: max 1 re-prediction per 10 s (natural cache cycle).
        Only runs when flat (no open position).
        """
        if self.risk.has_open_trade:
            return False

        now = time.time()
        if now - self._last_repredict_time < 10.0:
            return False
        self._last_repredict_time = now

        try:
            # ── Fresh data ──
            ohlcv = self.perception.fetch_ohlcv()
            features = self.perception.compute_features(ohlcv)
            latest_row = features.iloc[-1]

            # ATR
            hl_range = (ohlcv["High"] - ohlcv["Low"]).rolling(ATR_PERIOD).mean()
            self._current_atr = (
                float(hl_range.iloc[-1]) if not hl_range.empty else 0.0
            )
            if self._current_atr <= 0 or len(ohlcv) < 20:
                return False

            # EMA7
            self._last_ema7 = float(
                ohlcv["Close"].ewm(span=7, adjust=False).mean().iloc[-1]
            )

            # ── Regime detection ──
            regime = self.router.detect_regime(latest_row)
            if self.news_filter is not None:
                blackout, _ = self.news_filter.is_blackout()
                if blackout and regime != Regime.HIGH_VOLATILITY:
                    regime = Regime.HIGH_VOLATILITY

            # Only re-predict for trending regimes (where ribbon deferral occurs)
            if regime not in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
                self.log.info(
                    "🔄 RADAR: Regime=%s (non-trending) — radar off",
                    regime.value,
                )
                self._radar_active = False
                return False

            # Safety: drawdown + circuit breaker
            acct = self.perception.get_account_info()
            if self.risk.check_drawdown(acct["balance"]) or self.risk.is_halted:
                return False

            # ── Normalize + sanitize ──
            raw_obs = latest_row.values.astype(np.float32)
            agent_data = self.agents[regime]
            obs_normalized = _normalize_obs(raw_obs, agent_data["stats"])
            obs_ready = obs_normalized.reshape(1, -1)
            obs_ready = np.nan_to_num(
                obs_ready, nan=0.0,
                posinf=OBS_CLIP_RANGE, neginf=-OBS_CLIP_RANGE,
            )
            obs_ready = np.clip(obs_ready, -OBS_CLIP_RANGE, OBS_CLIP_RANGE)
            if np.any(np.isnan(obs_ready)) or np.any(np.isinf(obs_ready)):
                obs_ready = np.nan_to_num(
                    obs_ready, nan=0.0, posinf=0.0, neginf=0.0,
                )

            # ── Inference ──
            agent_model: PPO = agent_data["model"]
            probs: np.ndarray | None = None
            try:
                obs_tensor, _ = agent_model.policy.obs_to_tensor(obs_ready)
                with torch.no_grad():
                    dist = agent_model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.cpu().numpy()[0]
            except Exception:
                pass

            action_idx, _ = agent_model.predict(obs_ready, deterministic=True)
            raw_action = int(action_idx.item())
            allowed = AGENT_ACTION_MAP[regime]
            actual_action = allowed[raw_action]

            # Confidence gate
            if actual_action != ACTION_HOLD and probs is not None:
                action_conf = float(probs[raw_action]) * 100
                if action_conf < CONFIDENCE_GATE_PCT:
                    self.log.info(
                        "🔄 RADAR: Confidence %.1f%% < %.0f%% — no cache",
                        action_conf, CONFIDENCE_GATE_PCT,
                    )
                    return False

            if actual_action == ACTION_HOLD:
                self.log.info("🔄 RADAR: AI says HOLD — waiting")
                return False

            # ── Ribbon filter (same logic as step 12d) ──
            close_series = ohlcv["Close"]
            current_high = float(ohlcv["High"].iloc[-1])
            current_low = float(ohlcv["Low"].iloc[-1])
            current_ema7 = float(
                close_series.ewm(span=7, adjust=False).mean().iloc[-1]
            )
            current_ema20 = float(
                close_series.ewm(span=20, adjust=False).mean().iloc[-1]
            )
            ribbon_gap = abs(current_ema7 - current_ema20)
            rsi_fast_val = float(latest_row.get("rsi_fast", 50.0))

            _tick = mt5.symbol_info_tick(self.symbol)
            tick_price = (
                float(_tick.bid) if _tick is not None
                else float(close_series.iloc[-1])
            )

            if actual_action == ACTION_BUY and regime == Regime.TRENDING_UP:
                if rsi_fast_val >= 85.0:
                    return False
                ribbon_ok = (
                    current_ema7 > current_ema20
                    and ribbon_gap >= 0.1 * self._current_atr
                    and current_low <= current_ema7 + 0.2 * self._current_atr
                    and tick_price > current_ema7
                )
                if not ribbon_ok:
                    self._predictive_cache = {
                        "action": ACTION_BUY,
                        "regime": regime,
                        "target_price": current_ema7,
                        "timestamp": time.time(),
                        "atr": self._current_atr,
                    }
                    self.log.info(
                        "🔄 RADAR CACHE: BUY "
                        "(EMA7=%.2f, ATR=%.2f, tick=%.2f)",
                        current_ema7, self._current_atr, tick_price,
                    )
                    return True
                # Ribbon passed — dispatch directly
                self.log.info(
                    "🔄 RADAR FIRE: BUY (ribbon OK, tick=%.2f)", tick_price,
                )
                self._dispatch_action(
                    ACTION_BUY, regime, ohlcv, acct["equity"],
                )
                self._radar_active = False
                return True

            if actual_action == ACTION_SELL and regime == Regime.TRENDING_DOWN:
                if rsi_fast_val <= 15.0:
                    return False
                ribbon_ok = (
                    current_ema7 < current_ema20
                    and ribbon_gap >= 0.1 * self._current_atr
                    and current_high >= current_ema7 - 0.2 * self._current_atr
                    and tick_price < current_ema7
                )
                if not ribbon_ok:
                    self._predictive_cache = {
                        "action": ACTION_SELL,
                        "regime": regime,
                        "target_price": current_ema7,
                        "timestamp": time.time(),
                        "atr": self._current_atr,
                    }
                    self.log.info(
                        "🔄 RADAR CACHE: SELL "
                        "(EMA7=%.2f, ATR=%.2f, tick=%.2f)",
                        current_ema7, self._current_atr, tick_price,
                    )
                    return True
                # Ribbon passed — dispatch directly
                self.log.info(
                    "🔄 RADAR FIRE: SELL (ribbon OK, tick=%.2f)", tick_price,
                )
                self._dispatch_action(
                    ACTION_SELL, regime, ohlcv, acct["equity"],
                )
                self._radar_active = False
                return True

            return False

        except Exception:
            self.log.exception("Intra-bar re-prediction failed")
            return False

    # ── Elastic Cooldown — intra-bar swing tracking ─
    def _update_swing_tracking(self) -> None:
        """Track intra-bar swing extension for Elastic Cooldown.

        Sets _swing_extended = True when |tick - EMA7| > 0.5 × ATR,
        indicating the trend has pushed far enough for a meaningful pullback entry.
        """
        if self._swing_extended or self.risk.has_open_trade:
            return
        if self._last_ema7 <= 0 or self._current_atr <= 0:
            return

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        tick_price = float(tick.bid)
        if abs(tick_price - self._last_ema7) > 0.5 * self._current_atr:
            self._swing_extended = True
            self.log.info(
                "🔄 ELASTIC: Swing extended "
                "(|%.2f - EMA7 %.2f| > 0.5×ATR %.2f)",
                tick_price,
                self._last_ema7,
                self._current_atr,
            )

    # ── Profit Locking  (Break-Even + Partial Close) ─
    def _check_profit_locking(self) -> None:
        """Auto Break-Even + Partial Close profit protection.

        Journey:
          1.0 × ATR profit → move SL to entry (risk-free trade)
          1.5 × ATR profit → close 50% of position (cash in pocket)
          2.0 × ATR profit → trailing stop takes over (tight lock)
        """
        if not self.risk.has_open_trade:
            self._break_even_done = False
            self._partial_close_done = False
            return

        if self._current_atr <= 0:
            return

        trade = self.risk.open_trade
        sym = self.perception.get_symbol_info()
        point = sym["point"]

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        # Profit in price distance (direction-aware)
        if trade.direction == "BUY":
            profit_dist = tick.bid - trade.entry_price
        else:
            profit_dist = trade.entry_price - tick.ask

        # 1. Break-Even: Move SL to entry + buffer
        if (
            ENABLE_BREAK_EVEN
            and not self._break_even_done
            and profit_dist >= BREAK_EVEN_ACTIVATION_ATR * self._current_atr
        ):
            if trade.direction == "BUY":
                new_sl = trade.entry_price + (BREAK_EVEN_BUFFER_POINTS * point)
            else:
                new_sl = trade.entry_price - (BREAK_EVEN_BUFFER_POINTS * point)
            new_sl = round(new_sl, 2)

            # Only move if it improves the position
            should_move = (
                (trade.direction == "BUY" and new_sl > trade.sl)
                or (trade.direction == "SELL" and new_sl < trade.sl)
            )

            if should_move and self.executor.modify_sl(new_sl):
                self.log.info(
                    "🛡️ BREAK-EVEN ACTIVATED for Ticket #%d "
                    "(SL moved to %.2f — Risk-Free Trade)",
                    trade.ticket,
                    new_sl,
                )
                self._break_even_done = True

        # 2. Partial Close: Close 50% of position
        if (
            ENABLE_PARTIAL_CLOSE
            and not self._partial_close_done
            and profit_dist >= PARTIAL_CLOSE_ACTIVATION_ATR * self._current_atr
        ):
            vol_step = sym["volume_step"]
            vol_min = sym["volume_min"]

            volume_to_close = trade.lot * PARTIAL_CLOSE_VOLUME_PCT
            volume_to_close = math.floor(volume_to_close / vol_step) * vol_step
            volume_to_close = round(volume_to_close, 2)
            remaining = round(trade.lot - volume_to_close, 2)

            if volume_to_close >= vol_min and remaining >= vol_min:
                if self.executor.partial_close(volume_to_close):
                    self.log.info(
                        "💰 PARTIAL CLOSE: Secured 50%% profit for Ticket #%d "
                        "(closed %.2f lots, remaining %.2f lots)",
                        trade.ticket,
                        volume_to_close,
                        remaining,
                    )
                    self._partial_close_done = True
            else:
                self.log.warning(
                    "Partial close skipped — lot too small "
                    "(total=%.2f, to_close=%.2f, min=%.2f)",
                    trade.lot,
                    volume_to_close,
                    vol_min,
                )

    # ── Trailing Stop  (ATR-Based Dynamic) ────
    def _check_trailing_stop(self) -> None:
        """Monitor unrealized profit; force close on retrace.

        Thresholds adapt to current volatility:
          activation = TRAILING_ACTIVATION_ATR × ATR  (in points)
          drawdown   = TRAILING_DRAWDOWN_ATR  × ATR  (in points)
        """
        if not self.risk.has_open_trade:
            self._trailing_ticket = None
            self._highest_profit_points = 0.0
            return

        if self._current_atr <= 0:
            return

        trade = self.risk.open_trade

        # Reset if tracking a stale ticket
        if self._trailing_ticket is not None and self._trailing_ticket != trade.ticket:
            self._trailing_ticket = None
            self._highest_profit_points = 0.0

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        sym = self.perception.get_symbol_info()
        point = sym["point"]

        # Dynamic thresholds: ATR multiplier → points
        activation_pts = (TRAILING_ACTIVATION_ATR * self._current_atr) / point
        drawdown_pts_threshold = (TRAILING_DRAWDOWN_ATR * self._current_atr) / point

        # Profit in points (direction-aware, uses close-side price)
        if trade.direction == "BUY":
            profit_pts = (tick.bid - trade.entry_price) / point
        else:
            profit_pts = (trade.entry_price - tick.ask) / point

        # 1. Activation
        if profit_pts >= activation_pts:
            if self._trailing_ticket != trade.ticket:
                self.log.info(
                    "TRAILING ACTIVATED #%d — profit %.0f pts "
                    "(ATR=%.2f, threshold=%.0f pts)",
                    trade.ticket,
                    profit_pts,
                    self._current_atr,
                    activation_pts,
                )
                self._trailing_ticket = trade.ticket
                self._highest_profit_points = profit_pts

        # 2. Execution
        if self._trailing_ticket == trade.ticket:
            if profit_pts > self._highest_profit_points:
                self._highest_profit_points = profit_pts

            drawdown_pts = self._highest_profit_points - profit_pts
            if drawdown_pts >= drawdown_pts_threshold:
                self.log.warning(
                    "TRAILING STOP HIT #%d — peak=%.0f, now=%.0f, "
                    "dd=%.0f pts (threshold=%.0f)",
                    trade.ticket,
                    self._highest_profit_points,
                    profit_pts,
                    drawdown_pts,
                    drawdown_pts_threshold,
                )
                self._close_and_track("TRAILING_STOP")
                self._trailing_ticket = None
                self._highest_profit_points = 0.0

    # ── V3.0: Virtual Time-Decay Shield ───────────
    def _check_time_decay(self) -> None:
        """Force close if trade exceeds TRADE_LIFESPAN_SEC without reaching break-even.

        Virtual SL — computed in Python, fires market close.
        No SL modification spam to broker.
        """
        if not self.risk.has_open_trade:
            return
        if self._break_even_done:
            return  # Trade is safe (break-even reached) — no time pressure

        trade = self.risk.open_trade
        elapsed = (datetime.utcnow() - trade.open_time).total_seconds()

        if elapsed > TRADE_LIFESPAN_SEC:
            self.log.warning(
                "⏱️ TIME-DECAY SHIELD: Trade #%d open %.0fs > %ds "
                "without break-even — FORCE CLOSE",
                trade.ticket,
                elapsed,
                TRADE_LIFESPAN_SEC,
            )
            self._close_and_track("TIME_DECAY")

    # ── V3.0: Tick Recording & Velocity ───────────
    def _record_tick(self, price: float) -> None:
        """Append current tick to history and prune stale entries (>10 s)."""
        now = time.time()
        self._tick_history.append((now, price))
        # Keep only the last 10 seconds of ticks
        cutoff = now - 10.0
        self._tick_history = [
            (t, p) for t, p in self._tick_history if t >= cutoff
        ]

    def _compute_tick_velocity(self, window_sec: float) -> float:
        """Compute signed price movement over the last *window_sec* seconds.

        Returns price_now - price_at_(now - window_sec).
        If insufficient data, returns 0.0.
        """
        if len(self._tick_history) < 2:
            return 0.0
        now = self._tick_history[-1][0]
        target_time = now - window_sec
        # Find the tick closest to target_time
        best_tick = self._tick_history[0]
        for t, p in self._tick_history:
            if t <= target_time:
                best_tick = (t, p)
            else:
                break
        # Need at least half the window covered
        if now - best_tick[0] < window_sec * 0.5:
            return 0.0
        return self._tick_history[-1][1] - best_tick[1]

    # ── Regime Shift  (SRS §4 — The Clean Slate) ─
    def _handle_regime_shift(self, current_regime: Regime) -> None:
        """Force close all positions when regime changes.

        V3.0: Uses close_all_positions to handle both primary + pyramid.
        """
        if self._prev_regime is not None and current_regime != self._prev_regime:
            self.log.warning(
                "REGIME SHIFT: %s → %s",
                self._prev_regime.value,
                current_regime.value,
            )
            if self.risk.has_open_trade or self._pyramid_ticket is not None:
                self.log.warning("Clean Slate protocol — closing ALL positions")
                closed = self.executor.close_all_positions("REGIME_SHIFT")
                self.log.info("Regime shift closed %d position(s)", closed)
                # Sync internal state
                if self.risk.has_open_trade:
                    self.risk.register_close(0.0)
                self._pyramid_ticket = None
                self._break_even_done = False
                self._partial_close_done = False
                self._swing_extended = False

        self._prev_regime = current_regime

    # ── Action Dispatch  (SRS §5 + §6 + V3.0 Pyramiding) ─
    def _dispatch_action(
        self,
        action: int,
        regime: Regime,
        ohlcv: pd.DataFrame,
        equity: float,
    ) -> None:
        """Translate AI action to MT5 order with position-aware logic.

        Rules:
          HOLD + flat       → do nothing (wait)
          HOLD + in position→ maintain position (let TP/SL/Trailing close it)
          BUY/SELL + flat   → open new position
          BUY/SELL + same   → Risk-Free Pyramid (V3.0) if break-even done, else PASS
          BUY/SELL + opposite → close existing (don't reopen this bar)
        """
        has_trade = self.risk.has_open_trade
        trade = self.risk.open_trade

        # -- HOLD action --
        if action == ACTION_HOLD:
            if has_trade:
                # V2.18: HOLD means "no new entry" — NOT "exit current trade".
                # Let TP/SL, Break-Even, Trailing Stop, or Regime Shift
                # handle the exit instead of prematurely closing a runner.
                self.log.info(
                    "HOLD signal — maintaining %s position #%d",
                    trade.direction,
                    trade.ticket,
                )
            return

        # -- BUY or SELL --
        direction = "BUY" if action == ACTION_BUY else "SELL"

        if has_trade:
            if trade.direction == direction:
                # V3.0 Risk-Free Pyramiding: allow 2nd position when
                # primary is at break-even (zero additional risk)
                if (
                    ENABLE_PYRAMIDING
                    and self._break_even_done
                    and self._pyramid_ticket is None
                ):
                    positions = mt5.positions_get(symbol=self.symbol)
                    our_count = sum(
                        1 for p in (positions or []) if p.magic == MAGIC_NUMBER
                    )
                    if our_count < MAX_POSITIONS:
                        self.log.info(
                            "🔥 RISK-FREE PYRAMIDING: Wood #1 (#%d) is safe, "
                            "firing Wood #2!",
                            trade.ticket,
                        )
                        tick = mt5.symbol_info_tick(self.symbol)
                        if tick is not None:
                            sym = self.perception.get_symbol_info()
                            ticket = self.executor.open_pyramid_position(
                                direction=direction,
                                regime=regime,
                                price_bid=tick.bid,
                                price_ask=tick.ask,
                                atr=self._current_atr,
                                point=sym["point"],
                                equity=equity,
                                tick_value=sym["trade_tick_value"],
                                tick_size=sym["trade_tick_size"],
                                volume_min=sym["volume_min"],
                                volume_max=sym["volume_max"],
                                volume_step=sym["volume_step"],
                            )
                            if ticket is not None:
                                self._pyramid_ticket = ticket
                        return
                # Anti-Martingale: same direction + no pyramid conditions → PASS
                return
            else:
                # Opposite direction → close only (next bar may reopen)
                self.log.info(
                    "Closing %s (opposite signal: %s)",
                    trade.direction,
                    direction,
                )
                self._close_and_track("OPPOSITE_SIGNAL")
                return

        # Flat → open new position
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.log.error("Cannot get tick — skipping trade")
            return

        sym = self.perception.get_symbol_info()

        # Find the raw_action index for ExecutionEngine
        raw_action_idx = AGENT_ACTION_MAP[regime].index(action)

        self.executor.execute_action(
            raw_action=raw_action_idx,
            regime=regime,
            price_bid=tick.bid,
            price_ask=tick.ask,
            atr=self._current_atr,
            point=sym["point"],
            equity=equity,
            tick_value=sym["trade_tick_value"],
            tick_size=sym["trade_tick_size"],
            volume_min=sym["volume_min"],
            volume_max=sym["volume_max"],
            volume_step=sym["volume_step"],
            current_bar=self._bar_count,
        )

    # ── Position Sync  (detect TP/SL hits by broker) ─
    def _sync_position_state(self) -> None:
        """Detect positions closed by the broker (TP/SL hit on server).

        V3.0: Also monitors the pyramid ticket — if it disappears from
        the broker, reset _pyramid_ticket (TP/SL hit on Wood #2).
        """
        if not self.risk.has_open_trade and self._pyramid_ticket is None:
            return

        positions = mt5.positions_get(symbol=self.symbol)
        magic_tickets = set()
        if positions:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER:
                    magic_tickets.add(pos.ticket)

        # --- Primary trade sync ---
        if self.risk.has_open_trade:
            trade = self.risk.open_trade
            if trade.ticket in magic_tickets:
                # Sync ticket if it drifted (e.g. after partial close)
                for pos in positions:
                    if pos.magic == MAGIC_NUMBER and pos.ticket != trade.ticket:
                        # Check if the original ticket is gone but another exists
                        pass
                    if pos.magic == MAGIC_NUMBER and pos.ticket == trade.ticket:
                        break
            else:
                # Primary trade not found — closed by broker
                profit = self._get_closed_trade_profit(trade.ticket, trade.open_time)
                reason = "TP/SL_HIT" if profit != 0.0 else "BROKER_CLOSE"
                self.log.info(
                    "Trade #%d closed by broker (%s) [%s %s] — profit=%.2f",
                    trade.ticket,
                    reason,
                    trade.direction,
                    trade.regime.value,
                    profit,
                )
                self.risk.register_close(profit)
                # Elastic Cooldown: reset swing tracking for next entry
                self._swing_extended = False

        # --- V3.0: Pyramid ticket sync ---
        if self._pyramid_ticket is not None:
            if self._pyramid_ticket not in magic_tickets:
                self.log.info(
                    "🔥 Pyramid #%d closed by broker (TP/SL hit)",
                    self._pyramid_ticket,
                )
                self._pyramid_ticket = None

    def _get_closed_trade_profit(
        self, ticket: int, open_time: datetime
    ) -> float:
        """Retrieve PnL for a position closed by the broker."""
        try:
            now = datetime.utcnow()
            start = open_time - timedelta(minutes=1)
            end = now + timedelta(hours=1)
            deals = mt5.history_deals_get(start, end)
            if deals:
                for deal in reversed(deals):
                    if (
                        deal.position_id == ticket
                        and deal.entry == mt5.DEAL_ENTRY_OUT
                    ):
                        return deal.profit + deal.swap + deal.commission
        except Exception:
            self.log.exception("Failed to retrieve deal history for #%d", ticket)
        return 0.0

    # ── Shutdown ──────────────────────────────
    def _shutdown(self) -> None:
        """Graceful shutdown — close ALL open trades and disconnect.

        V3.0: Uses close_all_positions to handle primary + pyramid.
        """
        self.log.info("Shutting down...")

        if self.risk.has_open_trade or self._pyramid_ticket is not None:
            self.log.warning("Closing ALL positions before shutdown")
            closed = self.executor.close_all_positions("SHUTDOWN")
            self.log.info("Shutdown closed %d position(s)", closed)
            if self.risk.has_open_trade:
                self.risk.register_close(0.0)
            self._pyramid_ticket = None

        self.perception.disconnect()
        self.log.info("═══ Apex Predator V2 — Live Engine Stopped ═══")


# ══════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apex Predator V2 — Live Execution Engine"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=TIMEFRAME_NAME,
        help="Timeframe (M1/M5/M15/H1)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=SYMBOL,
        help="MT5 symbol to trade",
    )
    args = parser.parse_args()

    engine = LiveEngine(timeframe=args.timeframe, symbol=args.symbol)
    engine.start()


if __name__ == "__main__":
    main()
