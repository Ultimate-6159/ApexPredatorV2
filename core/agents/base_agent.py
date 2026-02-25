"""
Base Agent — shared logic for all 4 specialised RL agents.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList

from config import MODEL_DIR, TRAINING_TIMESTEPS, Regime

if TYPE_CHECKING:
    from training.training_logger import TrainingLogger

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
    def train(
        self,
        env: Any,
        timesteps: int = TRAINING_TIMESTEPS,
        enable_logging: bool = True,
        log_freq: int = 1000,
        save_freq: int = 10000,
    ) -> None:
        """Train the agent on the given environment.

        Parameters
        ----------
        env : gym.Env
            The training environment.
        timesteps : int
            Total training timesteps.
        enable_logging : bool
            Whether to enable detailed training logging.
        log_freq : int
            Frequency of logging (every N steps).
        save_freq : int
            Frequency of saving logs to file (every N steps).
        """
        logger.info("Training %s for %d timesteps …", self.regime.value, timesteps)

        # V8.0: Tuned defaults for financial RL (overridable via algo_kwargs)
        default_kwargs: dict[str, Any] = {
            "learning_rate": 1e-4,      # Slower learning — prevents overshooting
            "n_steps": 512,             # Shorter rollouts — faster regime adaptation
            "batch_size": 128,          # Larger batches — smoother gradients
            "ent_coef": 0.01,           # Exploration bonus — avoids action collapse
            "gamma": 0.95,              # Shorter discount — matches 5–20 bar trades
        }
        default_kwargs.update(self.algo_kwargs)

        self.model = self.algo_cls(
            "MlpPolicy",
            env,
            verbose=1,
            **default_kwargs,
        )

        # Setup callbacks
        callbacks = []
        if enable_logging:
            # Lazy import to avoid circular imports
            from training.training_logger import TrainingLogger

            training_logger = TrainingLogger(
                regime=self.regime,
                log_freq=log_freq,
                save_freq=save_freq,
                verbose=1,
            )
            callbacks.append(training_logger)

        callback_list = CallbackList(callbacks) if callbacks else None

        self.model.learn(total_timesteps=timesteps, callback=callback_list)
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
