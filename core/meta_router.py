"""
Layer 2 — Deterministic Meta-Router
Hard-coded logic (strictly NO ML) that classifies the current market regime
and routes the state to the appropriate specialised agent.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import (
    ADX_TREND_THRESHOLD,
    VOLATILITY_RATIO_THRESHOLD,
    Regime,
)

logger = logging.getLogger(__name__)


class MetaRouter:
    """Classifies market regime from the latest feature row."""

    @staticmethod
    def detect_regime(features: pd.Series) -> Regime:
        """Determine regime from a single feature row.

        Priority order (first match wins):
        1. HIGH_VOLATILITY  — Volatility Ratio > threshold
        2. TRENDING_UP      — ADX > 25 *and* +DI > -DI
        3. TRENDING_DOWN    — ADX > 25 *and* -DI > +DI
        4. MEAN_REVERTING   — fallback (ADX < 25)
        """
        adx: float = float(features.get("adx", 0))
        plus_di: float = float(features.get("plus_di", 0))
        minus_di: float = float(features.get("minus_di", 0))
        vol_ratio: float = float(features.get("volatility_ratio", 1.0))

        # 1. High Volatility check first (takes priority)
        if vol_ratio > VOLATILITY_RATIO_THRESHOLD:
            regime = Regime.HIGH_VOLATILITY

        # 2-3. Trending (ADX strong enough)
        elif adx > ADX_TREND_THRESHOLD:
            if plus_di > minus_di:
                regime = Regime.TRENDING_UP
            else:
                regime = Regime.TRENDING_DOWN

        # 4. Fallback — mean-reverting / range
        else:
            regime = Regime.MEAN_REVERTING

        logger.info(
            "Regime=%s  (ADX=%.1f, +DI=%.1f, -DI=%.1f, VolRatio=%.2f)",
            regime.value,
            adx,
            plus_di,
            minus_di,
            vol_ratio,
        )
        return regime

    @staticmethod
    def should_emergency_exit(
        open_regime: Regime,
        current_regime: Regime,
        trade_direction: str,
    ) -> bool:
        """Return True if a regime shift invalidates the open position.

        Rules:
        - Bull Rider trade open + regime flips to TRENDING_DOWN → exit
        - Bear Hunter trade open + regime flips to TRENDING_UP  → exit
        - Any trade open + regime changes to a completely different one → exit
        """
        if open_regime == current_regime:
            return False

        # Direct contradiction — always exit
        if open_regime == Regime.TRENDING_UP and current_regime == Regime.TRENDING_DOWN:
            return True
        if open_regime == Regime.TRENDING_DOWN and current_regime == Regime.TRENDING_UP:
            return True

        # Direction contradiction under HIGH_VOLATILITY
        if current_regime == Regime.HIGH_VOLATILITY:
            return True  # vol spike while in a calm-regime trade → exit

        # Mean-reverting → trending against position
        if open_regime == Regime.MEAN_REVERTING:
            if trade_direction == "BUY" and current_regime == Regime.TRENDING_DOWN:
                return True
            if trade_direction == "SELL" and current_regime == Regime.TRENDING_UP:
                return True

        return False
