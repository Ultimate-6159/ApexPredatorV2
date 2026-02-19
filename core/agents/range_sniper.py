"""Agent 3 — The Range Sniper  (MEAN_REVERTING)
Action Space: [HOLD, BUY, SELL]  — Optimised for high win-rate, short holds.
"""

from __future__ import annotations

from config import Regime
from core.agents.base_agent import BaseAgent


class RangeSniper(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            regime=Regime.MEAN_REVERTING,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            ent_coef=0.02,
        )
