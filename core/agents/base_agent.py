"""
Base Agent — shared logic for all 4 specialised RL agents.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from config import MODEL_DIR, TRAINING_TIMESTEPS, Regime

logger = logging.getLogger(__name__)


class BaseAgent:
    """Wraps a Stable-Baselines3 model for a specific regime.

    Parameters
    ----------
    regime : Regime
        The market regime this agent is responsible for.
    algo_cls :
        SB3 algorithm class (default ``PPO``).
    """

    def __init__(
        self,
        regime: Regime,
        algo_cls: type[BaseAlgorithm] = PPO,
        **algo_kwargs: Any,
    ) -> None:
        self.regime = regime
        self.algo_cls = algo_cls
        self.algo_kwargs = algo_kwargs
        self.model: BaseAlgorithm | None = None

    @property
    def model_path(self) -> str:
        return os.path.join(MODEL_DIR, f"{self.regime.value.lower()}.zip")

    # ── Training ──────────────────────────────────
    def train(self, env: Any, timesteps: int = TRAINING_TIMESTEPS) -> None:
        logger.info("Training %s for %d timesteps …", self.regime.value, timesteps)
        self.model = self.algo_cls(
            "MlpPolicy",
            env,
            verbose=1,
            **self.algo_kwargs,
        )
        self.model.learn(total_timesteps=timesteps)
        self.save()

    # ── Persistence ───────────────────────────────
    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        logger.info("Model saved → %s", self.model_path)

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model at {self.model_path}")
        self.model = self.algo_cls.load(self.model_path)
        logger.info("Model loaded ← %s", self.model_path)

    # ── Prediction ────────────────────────────────
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        if self.model is None:
            raise RuntimeError(f"Agent {self.regime.value} has no loaded model")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
