"""
Custom Gymnasium environment for training specialised RL agents.
Each agent type gets its own action-space constraints and reward shaping.

The environment simulates ATR-based TP/SL mechanics that mirror the
backtest engine so the agent learns under realistic conditions.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    AGENT_ACTION_MAP,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    MAX_HOLDING_BARS,
    RISK_PER_TRADE_PCT,
    Regime,
)

logger = logging.getLogger(__name__)

# ── Reward constants ─────────────────────────────
_TP_REWARD: float = 1.0
_SL_PENALTY: float = -1.0
_TIMESTOP_PENALTY: float = -0.5
_HOLD_FLAT_PENALTY: float = -0.001
_ATR_PERIOD: int = 14

# Exit-strategy rewards (v2)
_TRAILING_PENALTY: float = -2.0            # Penalise letting profit evaporate
_TRAILING_DD_THRESHOLD: float = 0.3        # Trigger when 30 % of peak profit lost
_PEAK_BONUS: float = 0.1                   # Small bonus each time profit makes new high
_CLOSE_PROFIT_BONUS: dict[Regime, float] = {
    Regime.TRENDING_UP:     2.0,            # Bull Rider: 2× for profitable close
    Regime.TRENDING_DOWN:   2.5,            # Bear Hunter: 2.5× (shorts must exit faster)
    Regime.MEAN_REVERTING:  3.0,            # Range Sniper: 3× (scalp → take profit ASAP)
    Regime.HIGH_VOLATILITY: 2.0,            # Vol Assassin: 2×
}


class TradingEnv(gym.Env):
    """A vectorised, single-asset trading environment for one regime.

    The environment now simulates TP/SL using ATR — identical logic to
    ``BacktestEngine`` — so the agent learns entry timing that produces
    real TP hits rather than aimless trades.

    Parameters
    ----------
    features : pd.DataFrame
        Pre-computed feature matrix (from PerceptionEngine).
    ohlcv : pd.DataFrame
        Corresponding OHLCV data aligned to *features* index.
    regime : Regime
        The regime this environment is specialised for.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
        regime: Regime,
    ) -> None:
        super().__init__()
        self.features = features.values.astype(np.float32)
        self.ohlcv = ohlcv.loc[features.index].reset_index(drop=True)
        self.regime = regime

        allowed_actions = AGENT_ACTION_MAP[regime]
        self.n_actions = len(allowed_actions)
        self.allowed_actions = allowed_actions

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.features.shape[1],),
            dtype=np.float32,
        )

        self.max_bars = MAX_HOLDING_BARS[regime]
        self.sl_mult = ATR_SL_MULTIPLIER
        self.tp_mult = ATR_TP_MULTIPLIER[regime]
        self.close_bonus = _CLOSE_PROFIT_BONUS[regime]

        # Pre-compute rolling ATR (same as backtest engine)
        hl_range = (self.ohlcv["High"] - self.ohlcv["Low"]).rolling(_ATR_PERIOD).mean()
        self._atr_series: np.ndarray = hl_range.fillna(
            self.ohlcv["Close"] * 0.01
        ).values.astype(np.float64)

        # Episode state
        self._current_step: int = 0
        self._position: int = 0          # 0=flat, 1=long, -1=short
        self._entry_price: float = 0.0
        self._entry_atr: float = 0.0
        self._sl: float = 0.0
        self._tp: float = 0.0
        self._hold_counter: int = 0
        self._peak_unrealised: float = 0.0  # Track max profit for trailing logic
        self._done: bool = False

    # ── Gym API ───────────────────────────────────
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._current_step = 0
        self._position = 0
        self._entry_price = 0.0
        self._entry_atr = 0.0
        self._sl = 0.0
        self._tp = 0.0
        self._hold_counter = 0
        self._peak_unrealised = 0.0
        self._done = False
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        raw_action = self.allowed_actions[action]
        reward = self._execute_action(raw_action)

        self._current_step += 1
        terminated = self._done
        truncated = self._current_step >= len(self.features) - 1

        if truncated and self._position != 0:
            reward += self._close_position()

        return self._get_obs(), reward, terminated, truncated, {}

    # ── Internal Logic ────────────────────────────
    def _get_obs(self) -> np.ndarray:
        idx = min(self._current_step, len(self.features) - 1)
        return self.features[idx]

    def _current_close(self) -> float:
        idx = min(self._current_step, len(self.ohlcv) - 1)
        return float(self.ohlcv.loc[idx, "Close"])

    def _current_high_low(self) -> tuple[float, float]:
        idx = min(self._current_step, len(self.ohlcv) - 1)
        return float(self.ohlcv.loc[idx, "High"]), float(self.ohlcv.loc[idx, "Low"])

    def _current_atr(self) -> float:
        idx = min(self._current_step, len(self._atr_series) - 1)
        return float(self._atr_series[idx])

    def _reset_position(self) -> None:
        self._position = 0
        self._entry_price = 0.0
        self._entry_atr = 0.0
        self._sl = 0.0
        self._tp = 0.0
        self._hold_counter = 0
        self._peak_unrealised = 0.0

    def _check_tp_sl(self, high: float, low: float) -> str | None:
        """Check whether the current bar's high/low triggers TP or SL."""
        if self._position == 1:  # Long
            if low <= self._sl:
                return "SL_HIT"
            if high >= self._tp:
                return "TP_HIT"
        elif self._position == -1:  # Short
            if high >= self._sl:
                return "SL_HIT"
            if low <= self._tp:
                return "TP_HIT"
        return None

    def _execute_action(self, action: int) -> float:
        price = self._current_close()
        atr = self._current_atr()
        reward = 0.0

        if self._position != 0:
            self._hold_counter += 1
            high, low = self._current_high_low()

            # ── 1. Check TP/SL hit (mirrors BacktestEngine) ──
            hit = self._check_tp_sl(high, low)
            if hit == "TP_HIT":
                reward = _TP_REWARD
                self._reset_position()
                return reward
            if hit == "SL_HIT":
                reward = _SL_PENALTY
                self._reset_position()
                return reward

            # ── 2. Track peak unrealised PnL ──
            unrealised = self._normalised_pnl(price)
            peak_bonus = 0.0
            if unrealised > self._peak_unrealised:
                peak_bonus = _PEAK_BONUS
            self._peak_unrealised = max(self._peak_unrealised, unrealised)

            # ── 3. Trailing penalty: profit dropped >30% from peak ──
            if self._peak_unrealised > 0.1:
                dd = (self._peak_unrealised - unrealised) / self._peak_unrealised
                if dd > _TRAILING_DD_THRESHOLD:
                    reward = _TRAILING_PENALTY
                    self._reset_position()
                    return reward

            # ── 4. Time stop ──
            if self._hold_counter >= self.max_bars:
                reward = unrealised + _TIMESTOP_PENALTY
                self._reset_position()
                return reward

            # ── 5. Stepped time-decay penalty ──
            time_penalty = 0.0
            half = self.max_bars * 0.5
            if self._hold_counter > half:
                progress = (self._hold_counter - half) / max(half, 1)
                time_penalty = -0.05 * (1.0 + progress)  # -0.05 → -0.10

            # ── 6. Voluntary close: HOLD while in position = exit ──
            if action == ACTION_HOLD:
                pnl_norm = self._normalised_pnl(price)
                if pnl_norm > 0:
                    reward = pnl_norm * self.close_bonus
                else:
                    reward = pnl_norm
                self._reset_position()
                return reward

            # ── 7. Agent closes via opposite signal ──
            if (action == ACTION_BUY and self._position == -1) or (
                action == ACTION_SELL and self._position == 1
            ):
                pnl_norm = self._normalised_pnl(price)
                if pnl_norm > 0:
                    reward = pnl_norm * self.close_bonus
                else:
                    reward = pnl_norm
                self._reset_position()
                return reward

            # ── 8. Still holding — tiny unrealised PnL + time decay + peak bonus ──
            reward = unrealised * 0.01 + time_penalty + peak_bonus
            return reward

        # ── Flat — open new position ──
        if action == ACTION_BUY:
            self._position = 1
            self._entry_price = price
            self._entry_atr = atr
            self._hold_counter = 0
            self._sl = price - atr * self.sl_mult
            self._tp = price + atr * self.tp_mult
        elif action == ACTION_SELL:
            self._position = -1
            self._entry_price = price
            self._entry_atr = atr
            self._hold_counter = 0
            self._sl = price + atr * self.sl_mult
            self._tp = price - atr * self.tp_mult
        else:
            reward = _HOLD_FLAT_PENALTY

        return reward

    def _normalised_pnl(self, current_price: float) -> float:
        """Return PnL normalised by entry ATR so rewards are scale-free."""
        pnl = (current_price - self._entry_price) * self._position
        if self._entry_atr > 0:
            return pnl / self._entry_atr
        return 0.0

    def _close_position(self) -> float:
        if self._position == 0:
            return 0.0
        reward = self._normalised_pnl(self._current_close())
        self._reset_position()
        return reward
