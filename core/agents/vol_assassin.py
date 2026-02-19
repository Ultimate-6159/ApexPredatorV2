"""Agent 4 — The Volatility Assassin  (HIGH_VOLATILITY)
Action Space: [HOLD, BUY, SELL]  — Breakout / squeeze specialist with tight SLs.
"""

from __future__ import annotations

from config import Regime
from core.agents.base_agent import BaseAgent


class VolAssassin(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            regime=Regime.HIGH_VOLATILITY,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            ent_coef=0.02,
        )
