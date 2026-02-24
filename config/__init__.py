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
LOOKBACK_BARS: int = 300         # Bars to fetch for feature calculation

# ──────────────────────────────────────────────
# Regime Detection Thresholds  (Meta-Router)
# ──────────────────────────────────────────────
ADX_TREND_THRESHOLD: float = 23.0
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
RISK_PER_TRADE_PCT: float = 8.0   # 8 % of equity (reduced for high-frequency scalping)
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
    Regime.TRENDING_UP:     1.20,   # Hit & Run — fast TP for high-frequency re-entry
    Regime.TRENDING_DOWN:   1.20,
    Regime.MEAN_REVERTING:  0.80,   # Tight scalp in ranging markets
    Regime.HIGH_VOLATILITY: 1.50,
}

# Trailing Stop (ATR-based — adapts to volatility)
TRAILING_ACTIVATION_ATR: float = 0.8    # Activate after 0.8 × ATR profit
TRAILING_DRAWDOWN_ATR: float = 0.4      # Force close if retraces 0.4 × ATR from peak

# Profit Locking Strategy (Break-Even + Partial Close)
ENABLE_BREAK_EVEN: bool = True
BREAK_EVEN_ACTIVATION_ATR: float = 0.5  # Move SL to entry when profit >= 0.5 × ATR (avoids gold spread noise)
BREAK_EVEN_BUFFER_POINTS: int = 20      # Extra points above entry to cover commission/swap

ENABLE_PARTIAL_CLOSE: bool = True
PARTIAL_CLOSE_ACTIVATION_ATR: float = 1.0  # Close 50% when profit >= 1.0 × ATR
PARTIAL_CLOSE_VOLUME_PCT: float = 0.5      # Fraction of lot to close (0.5 = 50%)

# V3.0 — Virtual Time-Decay Shield
TRADE_LIFESPAN_SEC: int = 90             # Force close if not break-even after this many seconds

# V3.0 — Risk-Free Pyramiding
ENABLE_PYRAMIDING: bool = True           # Allow 2nd position when 1st is at break-even
MAX_POSITIONS: int = 2                   # Maximum concurrent positions per symbol

# V3.0 — Phantom Spoofer Thresholds
PHANTOM_SWEEP_ATR: float = 0.3          # Overshoot distance to detect SL sweep
MOMENTUM_BOUNCE_ATR: float = 0.2        # Velocity threshold (ATR in 3 s) for bounce trigger
MOMENTUM_WINDOW_SEC: float = 3.0        # Time window for velocity measurement

# V3.5 — The 5th Dimension Patch
REGIME_SHIFT_GRACE_SEC: int = 180        # Shield fresh trades from regime shift (3 minutes)
VOLUME_ACCEL_MULTIPLIER: float = 1.5    # Tick volume must be > multiplier × avg to confirm trigger

# Inference Safety Guards
OBS_CLIP_RANGE: float = 10.0             # Hard clip Z-Score features to ± this value
CONFIDENCE_GATE_PCT: float = 65.0        # Force HOLD if AI confidence < this % (lowered for high-freq)

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
