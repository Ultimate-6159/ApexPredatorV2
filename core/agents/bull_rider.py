"""Agent 1 — The Bull Rider  (TRENDING_UP)
Action Space: [HOLD, BUY]  — SELL is strictly forbidden.
"""

from __future__ import annotations

from config import Regime
from core.agents.base_agent import BaseAgent


class BullRider(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            regime=Regime.TRENDING_UP,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
        )
