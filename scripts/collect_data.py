"""
Data Collector — Download and save historical OHLCV data for offline training.

Usage:
    python -m scripts.collect_data --bars 50000 --output data/xauusd_m5.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import SYMBOL
from core.perception_engine import PerceptionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect historical OHLCV data from MT5")
    parser.add_argument("--bars", type=int, default=50000, help="Number of bars to fetch")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (M1/M5/M15/H1)")
    parser.add_argument("--output", type=str, default="data/ohlcv.parquet", help="Output file path")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Trading symbol")
    args = parser.parse_args()

    perception = PerceptionEngine(symbol=args.symbol, timeframe=args.timeframe)
    if not perception.connect():
        logger.error("Cannot connect to MT5")
        sys.exit(1)

    try:
        logger.info("Fetching %d bars of %s %s data...", args.bars, args.symbol, args.timeframe)
        ohlcv = perception.fetch_ohlcv(bars=args.bars)
        features = perception.compute_features(ohlcv)

        # Align OHLCV to features
        ohlcv_aligned = ohlcv.loc[features.index]

        # Combine for storage
        combined = pd.concat([ohlcv_aligned, features], axis=1)

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.output.endswith(".parquet"):
            combined.to_parquet(output_path)
        else:
            combined.to_csv(output_path)

        logger.info("✓ Saved %d rows to %s", len(combined), output_path)
        logger.info("  Columns: %s", list(combined.columns))
        logger.info("  Date range: %s to %s", combined.index[0], combined.index[-1])

    finally:
        perception.disconnect()


if __name__ == "__main__":
    main()
