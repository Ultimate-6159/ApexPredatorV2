"""Agent 2 — The Bear Hunter  (TRENDING_DOWN)
Action Space: [HOLD, SELL]  — BUY is strictly forbidden.
"""

from __future__ import annotations

from config import Regime
from core.agents.base_agent import BaseAgent


class BearHunter(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            regime=Regime.TRENDING_DOWN,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
        )
