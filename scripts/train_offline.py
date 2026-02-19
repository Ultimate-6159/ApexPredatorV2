"""
Offline Training — Train agents from pre-saved historical data.
Use this when you don't have live MT5 connection.

Usage:
    python -m scripts.train_offline --data data/ohlcv.parquet --timesteps 200000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import TRAINING_TIMESTEPS, Regime
from core.agents.bear_hunter import BearHunter
from core.agents.bull_rider import BullRider
from core.agents.range_sniper import RangeSniper
from core.agents.vol_assassin import VolAssassin
from core.environments.trading_env import TradingEnv
from core.meta_router import MetaRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Feature columns expected from perception engine
FEATURE_COLS = [
    "rsi_fast", "rsi_slow", "bb_width",
    "dist_ema50", "dist_ema200",
    "adx", "plus_di", "minus_di",
    "atr_norm", "volatility_ratio", "volume_zscore",
    "close_return", "ema_cross",
]

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _filter_regime_data(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    regime: Regime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return rows where detected regime matches."""
    router = MetaRouter()
    mask = features.apply(lambda row: router.detect_regime(row) == regime, axis=1)
    filtered_feat = features.loc[mask].reset_index(drop=True)
    filtered_ohlcv = ohlcv.loc[mask].reset_index(drop=True)

    logger.info("  %s: %d rows (%.1f%%)", regime.value, len(filtered_feat), len(filtered_feat) / len(features) * 100)
    return filtered_feat, filtered_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train agents from offline data")
    parser.add_argument("--data", type=str, required=True, help="Path to parquet/csv data file")
    parser.add_argument("--timesteps", type=int, default=TRAINING_TIMESTEPS, help="Training timesteps per agent")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    # Load data
    logger.info("Loading data from %s...", data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Split into OHLCV and features
    ohlcv = df[OHLCV_COLS].copy()
    features = df[FEATURE_COLS].copy()

    logger.info("Loaded %d rows with %d features", len(features), len(FEATURE_COLS))

    # Regime distribution
    logger.info("\nRegime distribution:")

    agents = [BullRider(), BearHunter(), RangeSniper(), VolAssassin()]

    for agent in agents:
        regime = agent.regime
        logger.info("\n═══ Training %s ═══", regime.value)

        feat, price = _filter_regime_data(features, ohlcv, regime)
        if len(feat) < 100:
            logger.warning("Skipping %s — only %d rows (need at least 100)", regime.value, len(feat))
            continue

        env = TradingEnv(features=feat, ohlcv=price, regime=regime)
        agent.train(env, timesteps=args.timesteps)
        logger.info("✓ %s training complete", regime.value)

    logger.info("\n✓ All training complete!")


if __name__ == "__main__":
    main()
