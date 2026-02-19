"""
Apex Predator V2 — Live Trading Entry Point

Loop:
  1. Wait for new candle close
  2. Perception Engine → compute features
  3. Meta-Router → detect regime
  4. Risk checks (drawdown, circuit breaker, time-stop, regime-shift exit)
  5. Dispatch to the correct specialised agent → get action
  6. Execution Engine → place / manage MT5 orders
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    AGENT_ACTION_MAP,
    Regime,
    SYMBOL,
)
from core.agents.bear_hunter import BearHunter
from core.agents.bull_rider import BullRider
from core.agents.range_sniper import RangeSniper
from core.agents.vol_assassin import VolAssassin
from core.agents.base_agent import BaseAgent
from core.execution_engine import ExecutionEngine
from core.meta_router import MetaRouter
from core.perception_engine import PerceptionEngine
from core.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ApexPredatorV2")

# ── Graceful shutdown ─────────────────────────
_running = True


def _signal_handler(sig: int, frame) -> None:  # type: ignore[override]
    global _running
    logger.info("Shutdown signal received — cleaning up …")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _load_agents() -> dict[Regime, BaseAgent]:
    agents: dict[Regime, BaseAgent] = {
        Regime.TRENDING_UP: BullRider(),
        Regime.TRENDING_DOWN: BearHunter(),
        Regime.MEAN_REVERTING: RangeSniper(),
        Regime.HIGH_VOLATILITY: VolAssassin(),
    }
    for agent in agents.values():
        agent.load()
        logger.info("Loaded agent: %s", agent.regime.value)
    return agents


def main() -> None:
    # ── Initialise components ─────────────────
    perception = PerceptionEngine(symbol=SYMBOL, timeframe="M5")
    if not perception.connect():
        logger.critical("MT5 connection failed — exiting")
        sys.exit(1)

    router = MetaRouter()
    risk = RiskManager()
    executor = ExecutionEngine(risk_manager=risk, symbol=SYMBOL)
    agents = _load_agents()

    symbol_info = perception.get_symbol_info()
    bar_index = 0
    last_bar_time: datetime | None = None

    logger.info("══════════════════════════════════════")
    logger.info("   Apex Predator V2 — LIVE MODE")
    logger.info("══════════════════════════════════════")

    try:
        while _running:
            # ── 1. Wait for new candle ────────────
            try:
                features = perception.get_latest_features()
            except RuntimeError as exc:
                logger.error("Data fetch error: %s — retrying in 10 s", exc)
                time.sleep(10)
                continue

            current_bar_time = features.index[-1]
            if last_bar_time is not None and current_bar_time <= last_bar_time:
                time.sleep(5)
                continue

            last_bar_time = current_bar_time
            bar_index += 1
            risk.update_bar(bar_index)

            latest = features.iloc[-1]
            obs = latest.values.astype(np.float32)

            # ── 2. Regime detection ───────────────
            regime = router.detect_regime(latest)

            # ── 3. Risk pre-checks ────────────────
            account = perception.get_account_info()
            if risk.check_drawdown(account["balance"]):
                logger.critical("HARD STOP — max drawdown breached. Manual reset required.")
                break

            if risk.is_halted:
                time.sleep(30)
                continue

            # ── 4. Manage open trade ──────────────
            if risk.has_open_trade:
                trade = risk.open_trade
                assert trade is not None

                # Time stop
                if risk.should_time_stop(bar_index):
                    executor.close_open_trade(reason="TIME_STOP")
                    continue

                # Regime-shift emergency exit
                if router.should_emergency_exit(trade.regime, regime, trade.direction):
                    logger.warning("REGIME SHIFT EXIT — %s → %s", trade.regime.value, regime.value)
                    executor.close_open_trade(reason="REGIME_SHIFT")
                    continue

                # Already in a position — skip new entry
                continue

            # ── 5. Agent prediction ───────────────
            agent = agents[regime]
            raw_action = agent.predict(obs)

            # ── 6. Execute ────────────────────────
            tick = perception.get_current_tick()
            ohlcv = perception.fetch_ohlcv(bars=20)
            current_atr = float((ohlcv["High"] - ohlcv["Low"]).rolling(14).mean().iloc[-1])

            executor.execute_action(
                raw_action=raw_action,
                regime=regime,
                price_bid=tick["bid"],
                price_ask=tick["ask"],
                atr=current_atr,
                point=symbol_info["point"],
                balance=account["balance"],
                contract_size=symbol_info["trade_contract_size"],
                volume_min=symbol_info["volume_min"],
                volume_max=symbol_info["volume_max"],
                volume_step=symbol_info["volume_step"],
                current_bar=bar_index,
            )

            time.sleep(5)

    except Exception:
        logger.exception("Unhandled error in main loop")
    finally:
        # Close any open trade before shutdown
        if risk.has_open_trade:
            executor.close_open_trade(reason="SHUTDOWN")
        perception.disconnect()
        logger.info("Apex Predator V2 shut down cleanly.")


if __name__ == "__main__":
    main()
