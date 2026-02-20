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
ADX_TREND_THRESHOLD: float = 25.0
VOLATILITY_RATIO_THRESHOLD: float = 1.5
ADX_PERIOD: int = 14
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
    Regime.TRENDING_UP:      [ACTION_HOLD, ACTION_BUY],               # Bull Rider
    Regime.TRENDING_DOWN:    [ACTION_HOLD, ACTION_SELL],              # Bear Hunter
    Regime.MEAN_REVERTING:   [ACTION_HOLD, ACTION_BUY, ACTION_SELL],  # Range Sniper
    Regime.HIGH_VOLATILITY:  [ACTION_HOLD, ACTION_BUY, ACTION_SELL],  # Vol Assassin
}

# ──────────────────────────────────────────────
# Risk Management
# ──────────────────────────────────────────────
RISK_PER_TRADE_PCT: float = 0.5   # 0.5 % of balance
MAX_DRAWDOWN_PCT: float = 15.0    # 15 % hard stop
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
SLIPPAGE_POINTS: int = 30
MAGIC_NUMBER: int = 615900
ORDER_COMMENT: str = "ApexV2"

# SL / TP in pips (fallback; prefer ATR-based dynamic values)
DEFAULT_SL_PIPS: float = 30.0
DEFAULT_TP_PIPS: float = 60.0
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: dict[Regime, float] = {
    Regime.TRENDING_UP:     3.0,
    Regime.TRENDING_DOWN:   3.0,
    Regime.MEAN_REVERTING:  1.5,
    Regime.HIGH_VOLATILITY: 2.0,
}

# Trailing Stop (ATR-based — adapts to volatility)
TRAILING_ACTIVATION_ATR: float = 1.0    # Activate after 1.0 × ATR profit
TRAILING_DRAWDOWN_ATR: float = 0.5      # Force close if retraces 0.5 × ATR from peak

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
