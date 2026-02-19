"""
Training Logger — Callback system for logging training history, states, and learning factors.
Saves comprehensive logs for retrospective analysis.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from config import Regime

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Log Directory Configuration
# ──────────────────────────────────────────────
TRAINING_LOG_DIR: str = "logs/training"


class _MetricsCaptureWriter(KVWriter):
    """
    Custom SB3 logger output that captures all training metrics
    (rollout/*, time/*, train/*) emitted by the algorithm at each dump.
    """

    def __init__(self) -> None:
        self.metrics_history: list[dict[str, Any]] = []

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int,
    ) -> None:
        entry: dict[str, Any] = {"_step": step}
        for key, value in key_values.items():
            try:
                entry[key] = float(value) if isinstance(value, (int, float, np.number)) else value
            except (TypeError, ValueError):
                entry[key] = str(value)
        self.metrics_history.append(entry)

    def close(self) -> None:
        pass


class TrainingLogger(BaseCallback):
    """
    Comprehensive training logger callback for Stable Baselines3.
    
    Logs the following information:
    - Episode rewards and lengths
    - Action distribution per episode
    - Policy loss, value loss, entropy loss
    - Learning progress over time
    - State/observation statistics
    
    Parameters
    ----------
    regime : Regime
        The market regime being trained.
    log_freq : int
        Frequency of logging (every N steps).
    save_freq : int
        Frequency of saving logs to file (every N steps).
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        regime: Regime,
        log_freq: int = 1000,
        save_freq: int = 10000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.regime = regime
        self.log_freq = log_freq
        self.save_freq = save_freq
        
        # Training session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(TRAINING_LOG_DIR) / self.regime.value.lower() / self.session_id
        
        # Storage for metrics
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_actions: list[dict[int, int]] = []
        self.timestep_logs: list[dict] = []
        
        # Current episode tracking
        self._current_ep_reward: float = 0.0
        self._current_ep_length: int = 0
        self._current_ep_actions: dict[int, int] = {}
        
        # Observation statistics
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.obs_min: np.ndarray | None = None
        self.obs_max: np.ndarray | None = None
        
        # Training start time
        self.start_time: datetime | None = None

        # SB3 metrics capture writer (attached in _init_callback)
        self._metrics_writer: _MetricsCaptureWriter | None = None

    def _init_callback(self) -> bool:
        """Initialize callback - create log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        
        # Save initial config
        config_info = {
            "regime": self.regime.value,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "log_freq": self.log_freq,
            "save_freq": self.save_freq,
        }
        
        with open(self.log_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        # Attach custom metrics writer to SB3 logger
        self._metrics_writer = _MetricsCaptureWriter()
        if self.model is not None and hasattr(self.model, "logger"):
            self.model.logger.output_formats.append(self._metrics_writer)

        logger.info("Training logger initialized → %s", self.log_dir)
        return True

    def _on_step(self) -> bool:
        """Called after every step in the environment."""
        # Track current episode
        if self.locals.get("rewards") is not None:
            reward = self.locals["rewards"][0] if hasattr(self.locals["rewards"], "__len__") else self.locals["rewards"]
            self._current_ep_reward += float(reward)
        
        self._current_ep_length += 1
        
        # Track action distribution
        if self.locals.get("actions") is not None:
            action = self.locals["actions"][0] if hasattr(self.locals["actions"], "__len__") else self.locals["actions"]
            action = int(action)
            self._current_ep_actions[action] = self._current_ep_actions.get(action, 0) + 1
        
        # Update observation statistics
        if self.locals.get("obs_tensor") is not None:
            obs = self.locals["obs_tensor"].cpu().numpy() if hasattr(self.locals["obs_tensor"], "cpu") else self.locals["obs_tensor"]
            self._update_obs_stats(obs)
        
        # Check for episode end
        if self.locals.get("dones") is not None:
            done = self.locals["dones"][0] if hasattr(self.locals["dones"], "__len__") else self.locals["dones"]
            if done:
                self._on_episode_end()
        
        # Log at specified frequency
        if self.n_calls % self.log_freq == 0:
            self._log_timestep()
        
        # Save at specified frequency
        if self.n_calls % self.save_freq == 0:
            self._save_logs()

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout — save logs to prevent data loss."""
        self._save_logs()

    def _on_episode_end(self) -> None:
        """Called when an episode ends."""
        self.episode_rewards.append(self._current_ep_reward)
        self.episode_lengths.append(self._current_ep_length)
        self.episode_actions.append(self._current_ep_actions.copy())
        
        # Reset episode tracking
        self._current_ep_reward = 0.0
        self._current_ep_length = 0
        self._current_ep_actions = {}

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """Update running observation statistics."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        if self.obs_mean is None:
            self.obs_mean = obs.mean(axis=0)
            self.obs_std = obs.std(axis=0)
            self.obs_min = obs.min(axis=0)
            self.obs_max = obs.max(axis=0)
        else:
            # Exponential moving average for statistics
            alpha = 0.01
            self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs.mean(axis=0)
            self.obs_std = (1 - alpha) * self.obs_std + alpha * obs.std(axis=0)
            self.obs_min = np.minimum(self.obs_min, obs.min(axis=0))
            self.obs_max = np.maximum(self.obs_max, obs.max(axis=0))

    def _log_timestep(self) -> None:
        """Log metrics at current timestep."""
        log_entry: dict[str, Any] = {
            "timestep": self.num_timesteps,
            "n_calls": self.n_calls,
            "n_episodes": len(self.episode_rewards),
        }
        
        # Get training metrics from model if available
        if self.model is not None and hasattr(self.model, "logger") and self.model.logger is not None:
            # Try to get recent metrics
            try:
                if hasattr(self.model.logger, "name_to_value"):
                    for key, value in self.model.logger.name_to_value.items():
                        log_entry[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
            except Exception:
                pass
        
        # Recent episode statistics
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
            log_entry["ep_reward_mean"] = float(np.mean(recent_rewards))
            log_entry["ep_reward_std"] = float(np.std(recent_rewards))
            log_entry["ep_reward_min"] = float(np.min(recent_rewards))
            log_entry["ep_reward_max"] = float(np.max(recent_rewards))
            
            recent_lengths = self.episode_lengths[-100:]
            log_entry["ep_length_mean"] = float(np.mean(recent_lengths))
        
        # Action distribution (recent)
        if len(self.episode_actions) > 0:
            recent_actions = self.episode_actions[-100:]
            total_actions: dict[int, int] = {}
            for action_dist in recent_actions:
                for action, count in action_dist.items():
                    total_actions[action] = total_actions.get(action, 0) + count
            
            total = sum(total_actions.values())
            if total > 0:
                log_entry["action_distribution"] = {
                    str(k): round(v / total, 4) for k, v in sorted(total_actions.items())
                }
        
        self.timestep_logs.append(log_entry)
        
        if self.verbose > 0:
            logger.info("[Step %d] Episodes: %d, Reward Mean: %.4f", 
                       self.num_timesteps, 
                       len(self.episode_rewards),
                       log_entry.get("ep_reward_mean", 0))

    def _save_logs(self) -> None:
        """Save all logs to files."""
        logger.info("Saving training logs at step %d...", self.num_timesteps)
        
        # Save episode data
        if len(self.episode_rewards) > 0:
            episode_df = pd.DataFrame({
                "episode": range(len(self.episode_rewards)),
                "reward": self.episode_rewards,
                "length": self.episode_lengths,
            })
            episode_df.to_parquet(self.log_dir / "episodes.parquet", index=False)
            episode_df.to_csv(self.log_dir / "episodes.csv", index=False)
        
        # Save timestep logs
        if len(self.timestep_logs) > 0:
            # Flatten action_distribution for CSV compatibility
            flat_logs = []
            for log in self.timestep_logs:
                flat_log = {k: v for k, v in log.items() if k != "action_distribution"}
                if "action_distribution" in log:
                    for action, pct in log["action_distribution"].items():
                        flat_log[f"action_{action}_pct"] = pct
                flat_logs.append(flat_log)
            
            timestep_df = pd.DataFrame(flat_logs)
            timestep_df.to_parquet(self.log_dir / "timesteps.parquet", index=False)
            timestep_df.to_csv(self.log_dir / "timesteps.csv", index=False)
        
        # Save observation statistics
        if self.obs_mean is not None:
            obs_stats = {
                "mean": self.obs_mean.tolist(),
                "std": self.obs_std.tolist(),
                "min": self.obs_min.tolist(),
                "max": self.obs_max.tolist(),
            }
            with open(self.log_dir / "obs_stats.json", "w", encoding="utf-8") as f:
                json.dump(obs_stats, f, indent=2)
        
        # Save action distribution per episode (detailed)
        if len(self.episode_actions) > 0:
            with open(self.log_dir / "episode_actions.json", "w", encoding="utf-8") as f:
                json.dump(self.episode_actions, f, indent=2)

        # Save SB3 training metrics (rollout/*, time/*, train/*)
        if self._metrics_writer and len(self._metrics_writer.metrics_history) > 0:
            metrics_df = pd.DataFrame(self._metrics_writer.metrics_history)
            metrics_df.to_parquet(self.log_dir / "training_metrics.parquet", index=False)
            metrics_df.to_csv(self.log_dir / "training_metrics.csv", index=False)

        logger.info("Logs saved → %s", self.log_dir)

    def _on_training_end(self) -> None:
        """Called when training ends - save final logs."""
        # Final episode handling
        if self._current_ep_length > 0:
            self._on_episode_end()
        
        # Final log entry
        self._log_timestep()
        
        # Save all logs
        self._save_logs()
        
        # Save training summary
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        summary: dict[str, Any] = {
            "regime": self.regime.value,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_timesteps": self.num_timesteps,
            "total_episodes": len(self.episode_rewards),
            "final_reward_mean": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
            "final_reward_std": float(np.std(self.episode_rewards[-100:])) if self.episode_rewards else 0,
            "best_episode_reward": float(max(self.episode_rewards)) if self.episode_rewards else 0,
            "worst_episode_reward": float(min(self.episode_rewards)) if self.episode_rewards else 0,
        }

        # Include last captured SB3 training metrics in summary
        if self._metrics_writer and self._metrics_writer.metrics_history:
            last = self._metrics_writer.metrics_history[-1]
            summary["final_training_metrics"] = {
                k: v for k, v in last.items() if k != "_step"
            }
        
        with open(self.log_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("═══ Training Complete ═══")
        logger.info("Regime: %s", self.regime.value)
        logger.info("Duration: %.1f minutes", duration / 60)
        logger.info("Total Episodes: %d", len(self.episode_rewards))
        logger.info("Final Reward Mean: %.4f", summary["final_reward_mean"])
        logger.info("Logs saved → %s", self.log_dir)


class DetailedEpisodeLogger(BaseCallback):
    """
    Logs detailed per-step information within episodes.
    Use this for deep analysis of agent behavior.
    
    Warning: This creates large log files. Use only for debugging.
    """

    def __init__(
        self,
        regime: Regime,
        max_episodes: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.regime = regime
        self.max_episodes = max_episodes
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(TRAINING_LOG_DIR) / self.regime.value.lower() / self.session_id / "detailed"
        
        self.episode_data: list[dict] = []
        self._current_episode: list[dict] = []
        self._episode_count: int = 0

    def _init_callback(self) -> bool:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _on_step(self) -> bool:
        if self._episode_count >= self.max_episodes:
            return True
        
        step_data: dict[str, Any] = {
            "step": len(self._current_episode),
            "timestep": self.num_timesteps,
        }
        
        # Record observation
        if self.locals.get("obs_tensor") is not None:
            obs = self.locals["obs_tensor"]
            if hasattr(obs, "cpu"):
                obs = obs.cpu().numpy()
            step_data["observation"] = obs[0].tolist() if obs.ndim > 1 else obs.tolist()
        
        # Record action
        if self.locals.get("actions") is not None:
            action = self.locals["actions"]
            step_data["action"] = int(action[0]) if hasattr(action, "__len__") else int(action)
        
        # Record reward
        if self.locals.get("rewards") is not None:
            reward = self.locals["rewards"]
            step_data["reward"] = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
        
        self._current_episode.append(step_data)
        
        # Check for episode end
        if self.locals.get("dones") is not None:
            done = self.locals["dones"]
            done = done[0] if hasattr(done, "__len__") else done
            if done:
                self._save_episode()
        
        return True

    def _save_episode(self) -> None:
        if len(self._current_episode) == 0:
            return
        
        self._episode_count += 1
        
        # Save episode to file
        episode_file = self.log_dir / f"episode_{self._episode_count:04d}.json"
        with open(episode_file, "w", encoding="utf-8") as f:
            json.dump(self._current_episode, f, indent=2)
        
        self._current_episode = []
        
        if self.verbose > 0:
            logger.info("Saved detailed episode %d", self._episode_count)

    def _on_training_end(self) -> None:
        if len(self._current_episode) > 0:
            self._save_episode()
        logger.info("Detailed episode logs saved → %s", self.log_dir)
