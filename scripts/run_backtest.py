"""
Backtest Runner Script — Run historical simulation on trained models.

Usage:
    python -m scripts.run_backtest --bars 5000 --balance 10000
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import Regime, SYMBOL
from core.agents.bear_hunter import BearHunter
from core.agents.bull_rider import BullRider
from core.agents.range_sniper import RangeSniper
from core.agents.vol_assassin import VolAssassin
from core.backtest_engine import BacktestEngine, print_backtest_report
from core.perception_engine import PerceptionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on trained agents")
    parser.add_argument("--bars", type=int, default=5000, help="Number of historical bars")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (M1/M5/M15/H1)")
    args = parser.parse_args()

    # Load agents
    agents = {
        Regime.TRENDING_UP: BullRider(),
        Regime.TRENDING_DOWN: BearHunter(),
        Regime.MEAN_REVERTING: RangeSniper(),
        Regime.HIGH_VOLATILITY: VolAssassin(),
    }

    for agent in agents.values():
        try:
            agent.load()
            logger.info("Loaded agent: %s", agent.regime.value)
        except FileNotFoundError:
            logger.error("Model not found for %s — train first!", agent.regime.value)
            sys.exit(1)

    # Connect to MT5 and fetch data
    perception = PerceptionEngine(symbol=SYMBOL, timeframe=args.timeframe)
    if not perception.connect():
        logger.error("Cannot connect to MT5")
        sys.exit(1)

    try:
        logger.info("Fetching %d bars of historical data...", args.bars)
        ohlcv = perception.fetch_ohlcv(bars=args.bars)
        features = perception.compute_features(ohlcv)
        ohlcv = ohlcv.loc[features.index]  # Align

        logger.info("Running backtest on %d bars...", len(features))
        engine = BacktestEngine(agents=agents, perception=perception)
        result = engine.run(ohlcv=ohlcv, features=features, initial_balance=args.balance)

        print_backtest_report(result)

    finally:
        perception.disconnect()


if __name__ == "__main__":
    main()
