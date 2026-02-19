"""
Custom Gymnasium environment for training specialised RL agents.
Each agent type gets its own action-space constraints and reward shaping.
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


class TradingEnv(gym.Env):
    """A vectorised, single-asset trading environment for one regime.

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
        self.tp_mult = ATR_TP_MULTIPLIER[regime]

        # Episode state
        self._current_step: int = 0
        self._position: int = 0          # 0=flat, 1=long, -1=short
        self._entry_price: float = 0.0
        self._hold_counter: int = 0
        self._done: bool = False

    # ── Gym API ───────────────────────────────────
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._current_step = 0
        self._position = 0
        self._entry_price = 0.0
        self._hold_counter = 0
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

    def _current_atr(self) -> float:
        """Rough ATR proxy from OHLCV high-low of current bar."""
        idx = min(self._current_step, len(self.ohlcv) - 1)
        return float(self.ohlcv.loc[idx, "High"] - self.ohlcv.loc[idx, "Low"])

    def _execute_action(self, action: int) -> float:
        price = self._current_close()
        reward = 0.0

        if self._position != 0:
            # Already in a trade — check time stop
            self._hold_counter += 1
            pnl = (price - self._entry_price) * self._position

            if self._hold_counter >= self.max_bars:
                reward = pnl - abs(pnl) * 0.1  # penalty for time-out
                self._position = 0
                self._entry_price = 0.0
                self._hold_counter = 0
                return reward

            # Agent chooses HOLD → small time-decay penalty (Range Sniper effect)
            if action == ACTION_HOLD:
                reward = -0.01 * self._hold_counter
                return reward

            # Agent tries to open opposite or close (we treat new signal as close)
            if (action == ACTION_BUY and self._position == -1) or (
                action == ACTION_SELL and self._position == 1
            ):
                reward = pnl
                self._position = 0
                self._entry_price = 0.0
                self._hold_counter = 0
                return reward

            # Holding same direction — small reward for trend continuation
            reward = pnl * 0.01
            return reward

        # Flat — open new position
        if action == ACTION_BUY:
            self._position = 1
            self._entry_price = price
            self._hold_counter = 0
        elif action == ACTION_SELL:
            self._position = -1
            self._entry_price = price
            self._hold_counter = 0
        else:
            reward = -0.001  # tiny penalty for doing nothing to encourage engagement

        return reward

    def _close_position(self) -> float:
        if self._position == 0:
            return 0.0
        pnl = (self._current_close() - self._entry_price) * self._position
        self._position = 0
        self._entry_price = 0.0
        self._hold_counter = 0
        return pnl
