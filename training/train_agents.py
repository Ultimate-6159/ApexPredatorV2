"""
Training Pipeline — collects regime-filtered data and trains each agent.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from config import LOOKBACK_BARS, TRAINING_TIMESTEPS, Regime
from core.agents.bear_hunter import BearHunter
from core.agents.bull_rider import BullRider
from core.agents.range_sniper import RangeSniper
from core.agents.vol_assassin import VolAssassin
from core.environments.trading_env import TradingEnv
from core.meta_router import MetaRouter
from core.perception_engine import PerceptionEngine

logger = logging.getLogger(__name__)


def _filter_regime_data(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    regime: Regime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return contiguous blocks where the detected regime matches."""
    router = MetaRouter()
    mask = features.apply(lambda row: router.detect_regime(row) == regime, axis=1)
    filtered_feat = features.loc[mask]
    filtered_ohlcv = ohlcv.loc[filtered_feat.index]

    if len(filtered_feat) < 50:
        logger.warning(
            "Regime %s has only %d rows — may be insufficient for training",
            regime.value,
            len(filtered_feat),
        )

    # Re-index for the environment
    filtered_feat = filtered_feat.reset_index(drop=True)
    filtered_ohlcv = filtered_ohlcv.reset_index(drop=True)
    return filtered_feat, filtered_ohlcv


def train_all(
    timeframe: str = "M5",
    bars: int = 5000,
    timesteps: int = TRAINING_TIMESTEPS,
) -> None:
    """Connect to MT5, pull data, filter per regime, train each agent."""
    engine = PerceptionEngine(timeframe=timeframe)
    if not engine.connect():
        logger.error("Cannot connect to MT5 — aborting training")
        sys.exit(1)

    try:
        ohlcv = engine.fetch_ohlcv(bars=bars)
        features = engine.compute_features(ohlcv)
        # Align ohlcv to features index
        ohlcv = ohlcv.loc[features.index]

        agents = [BullRider(), BearHunter(), RangeSniper(), VolAssassin()]

        for agent in agents:
            regime = agent.regime
            logger.info("═══ Training %s ═══", regime.value)

            feat, price = _filter_regime_data(features, ohlcv, regime)
            if len(feat) < 50:
                logger.warning("Skipping %s — not enough data", regime.value)
                continue

            env = TradingEnv(features=feat, ohlcv=price, regime=regime)
            agent.train(env, timesteps=timesteps)
            logger.info("✓ %s training complete", regime.value)

    finally:
        engine.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_all()
