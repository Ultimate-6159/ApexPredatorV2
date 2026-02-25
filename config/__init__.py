"""
Apex Predator V2 — Global Configuration & Settings
All tunable parameters are defined here as constants.
"""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Market / Symbol
# ──────────────────────────────────────────────
SYMBOL: str = os.getenv("MT5_SYMBOL", "XAUUSD")
TIMEFRAME_NAME: str = "M5"       # Primary timeframe label
LOOKBACK_BARS: int = 500         # Bars to fetch for feature calculation (V5.2: EMA200 warm-up)

# ──────────────────────────────────────────────
# Regime Detection Thresholds  (Meta-Router)
# ──────────────────────────────────────────────
ADX_TREND_THRESHOLD: float = 23.0
ADX_TREND_ENTER: float = 25.0           # V5.2: Hysteresis — ADX must exceed this to enter trending
ADX_TREND_EXIT: float = 20.0            # V5.2: Hysteresis — ADX must drop below this to exit trending
VOLATILITY_RATIO_THRESHOLD: float = 1.5
ADX_PERIOD: int = 10
ATR_PERIOD: int = 14
RSI_FAST_PERIOD: int = 7
RSI_SLOW_PERIOD: int = 14
BB_PERIOD: int = 20
BB_STD: float = 2.0
EMA_FAST: int = 50
EMA_SLOW: int = 200


class Regime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    MEAN_REVERTING = "MEAN_REVERTING"


# ──────────────────────────────────────────────
# Agent Action Spaces
# ──────────────────────────────────────────────
# 0 = HOLD, 1 = BUY, 2 = SELL  (superset)
ACTION_HOLD: int = 0
ACTION_BUY: int = 1
ACTION_SELL: int = 2

AGENT_ACTION_MAP: dict[Regime, list[int]] = {
    Regime.TRENDING_UP:      [ACTION_HOLD, ACTION_BUY],                # Bull Rider  (idx1=BUY only)
    Regime.TRENDING_DOWN:    [ACTION_HOLD, ACTION_SELL],               # Bear Hunter (idx1=SELL only)
    Regime.MEAN_REVERTING:   [ACTION_HOLD, ACTION_BUY,  ACTION_SELL],  # Range Sniper
    Regime.HIGH_VOLATILITY:  [ACTION_HOLD, ACTION_BUY,  ACTION_SELL],  # Vol Assassin
}

# ──────────────────────────────────────────────
# Risk Management
# ──────────────────────────────────────────────
RISK_PER_TRADE_PCT: float = 3.0   # 3 % of equity (V5.2: balanced risk for sustainable growth)
MAX_DRAWDOWN_PCT: float = 60.0    # 60 % hard stop (accommodates aggressive risk per trade)
CONSECUTIVE_LOSS_LIMIT: int = 5
HALT_MINUTES: int = 30            # Cool-off after consecutive losses

# Maximum holding bars per regime (Time Stop / Guillotine)
MAX_HOLDING_BARS: dict[Regime, int] = {
    Regime.TRENDING_UP:     20,
    Regime.TRENDING_DOWN:   20,
    Regime.MEAN_REVERTING:  5,
    Regime.HIGH_VOLATILITY: 10,
}

# ──────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────
SLIPPAGE_POINTS: int = 35
MAGIC_NUMBER: int = 615900
ORDER_COMMENT: str = "ApexV2"

# SL / TP in pips (fallback; prefer ATR-based dynamic values)
DEFAULT_SL_PIPS: float = 30.0
DEFAULT_TP_PIPS: float = 60.0
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: dict[Regime, float] = {
    Regime.TRENDING_UP:     2.50,   # Trend-riding R:R = 1.67:1 (Elastic TP expands further)
    Regime.TRENDING_DOWN:   2.50,   # Trend-riding R:R = 1.67:1
    Regime.MEAN_REVERTING:  1.80,   # Balanced scalp R:R = 1.20:1
    Regime.HIGH_VOLATILITY: 2.50,   # Capture volatility R:R = 1.67:1
}

# Trailing Stop (ATR-based — adapts to volatility)
TRAILING_ACTIVATION_ATR: float = 1.5    # Activate after 1.5 × ATR profit (V5.2: let trend run)
TRAILING_DRAWDOWN_ATR: float = 0.8      # Force close if retraces 0.8 × ATR from peak

# Profit Locking Strategy (Break-Even + Partial Close)
ENABLE_BREAK_EVEN: bool = True
BREAK_EVEN_ACTIVATION_ATR: float = 0.5  # Move SL to entry when profit >= 0.5 × ATR (avoids gold spread noise)
BREAK_EVEN_BUFFER_POINTS: int = 20      # Extra points above entry to cover commission/swap

ENABLE_PARTIAL_CLOSE: bool = True
PARTIAL_CLOSE_ACTIVATION_ATR: float = 1.0  # Close 50% when profit >= 1.0 × ATR
PARTIAL_CLOSE_VOLUME_PCT: float = 0.5      # Fraction of lot to close (0.5 = 50%)

# V3.0 — Virtual Time-Decay Shield (V3.7: regime-aware)
TRADE_LIFESPAN_NORMAL_SEC: int = 600     # TRENDING_UP / TRENDING_DOWN (V5.2: 2× M5 bar)
TRADE_LIFESPAN_VOLATILE_SEC: int = 900   # HIGH_VOLATILITY / MEAN_REVERTING (V5.2: 3× M5 bar)

# V3.0 — Risk-Free Pyramiding (V5.1: unlimited positions when exposed risk = 0)
ENABLE_PYRAMIDING: bool = True           # Allow pyramiding when all prior positions are at break-even

# V3.0 — Phantom Spoofer Thresholds
PHANTOM_SWEEP_ATR: float = 0.3          # Overshoot distance to detect SL sweep
MOMENTUM_BOUNCE_ATR: float = 0.2        # Velocity threshold (ATR in 3 s) for bounce trigger
MOMENTUM_WINDOW_SEC: float = 3.0        # Time window for velocity measurement

# V3.5 — The 5th Dimension Patch
REGIME_SHIFT_GRACE_SEC: int = 180        # Shield fresh trades from regime shift (3 minutes)
VOLUME_ACCEL_MULTIPLIER: float = 1.5    # Tick volume must be > multiplier × avg to confirm trigger

# V4.0 — Volume-Kinetic Sizing & Filters
VOLUME_SMA_PERIOD: int = 10              # Bars for rolling volume average
VOLUME_SPIKE_MULTIPLIER: float = 1.2    # Entry gate: tick_vol must be > avg × this
MAX_CONSECUTIVE_LOSSES_DIR: int = 2      # Consecutive same-direction losses before penalty
PENALTY_BOX_MINUTES: int = 30           # Lockout duration for penalized direction
TIME_DECAY_SL_BUMP_RATIO: float = 0.5   # Squeeze SL by this fraction of (entry - SL) distance

# V5.0 — The Alpha HFT Paradigm
LIMIT_ORDER_ENABLED: bool = True          # Use Limit Orders instead of Market for Phantom fires
LIMIT_ORDER_EXPIRY_SEC: int = 10         # V5.2: Limit order auto-cancel after 10s
SUBBAR_TICK_INTERVAL: int = 20           # Re-evaluate every N ticks within a bar
SUBBAR_EVAL_TICKS: int = SUBBAR_TICK_INTERVAL  # V5.2: Alias for clarity
SUBBAR_RATE_LIMIT_SEC: float = 0.5      # V5.2: Min seconds between sub-bar scans (~2/sec)
DYNAMIC_TP_ENABLED: bool = True          # Enable ADX-driven TP expansion
DYNAMIC_TP_ACTIVATION_ADX: float = 35.0 # ADX must exceed this to expand TP
TP_EXPANSION_MULTIPLIER: float = 0.5    # Expand TP by this × ATR per step
TP_MAX_EXPANSIONS: int = 5              # Cap total expansions per trade

# V5.1 — The HFT Optimization Patch
MODIFY_THRESHOLD_POINTS: int = 15       # Skip SL/TP modify if delta < this × point (1.5 pips)
MAX_ALLOWED_SPREAD_POINTS: int = 300    # Block entry when spread > this (30 pips for gold)

# V5.3 — The Cache Optimization
PREDICTIVE_CACHE_TTL_SEC: int = 30       # Cache lifespan (covers M1/M5 candle swings)

# Inference Safety Guards
OBS_CLIP_RANGE: float = 10.0             # Hard clip Z-Score features to ± this value
CONFIDENCE_GATE_PCT: dict[Regime, float] = {   # V5.2: Regime-aware (2-action=55%, 3-action=65%)
    Regime.TRENDING_UP:     55.0,               # 2-action agent: random baseline 50%
    Regime.TRENDING_DOWN:   55.0,               # 2-action agent: random baseline 50%
    Regime.MEAN_REVERTING:  65.0,               # 3-action agent: random baseline 33%
    Regime.HIGH_VOLATILITY: 65.0,               # 3-action agent: random baseline 33%
}

# News Filter (Forex Factory calendar — forces HIGH_VOLATILITY before red news)
NEWS_FILTER_ENABLED: bool = True
NEWS_BLACKOUT_MINUTES: int = 15          # Minutes before event to activate
NEWS_CURRENCIES: list[str] = ["USD"]     # Currencies to watch
NEWS_CACHE_HOURS: int = 4                # Re-fetch calendar interval

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TRAINING_TIMESTEPS: int = 200_000
MODEL_DIR: str = "models"

# ──────────────────────────────────────────────
# Training Logging
# ──────────────────────────────────────────────
TRAINING_LOG_DIR: str = "logs/training"
TRAINING_LOG_FREQ: int = 1000        # Log every N steps
TRAINING_SAVE_FREQ: int = 10000      # Save logs every N steps
ENABLE_DETAILED_LOGGING: bool = True # Enable comprehensive logging
