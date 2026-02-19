"""
Layer 1 — Perception Engine
Ingests OHLCV data from MT5 and computes 12-15 noise-free features.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

from config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    EMA_FAST,
    EMA_SLOW,
    LOOKBACK_BARS,
    RSI_FAST_PERIOD,
    RSI_SLOW_PERIOD,
    SYMBOL,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Map readable timeframe name → MT5 constant
_TF_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class PerceptionEngine:
    """Connects to MT5 and produces a feature DataFrame."""

    def __init__(self, symbol: str = SYMBOL, timeframe: str = "M5") -> None:
        self.symbol = symbol
        self.timeframe_name = timeframe
        self.timeframe = _TF_MAP.get(timeframe, mt5.TIMEFRAME_M5)
        self._connected = False

    # ── MT5 Connection ────────────────────────────
    def connect(self) -> bool:
        if self._connected:
            return True

        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        path = os.getenv("MT5_PATH")

        if not all([login, password, server]):
            logger.error("MT5 credentials missing in .env")
            return False

        init_kwargs: dict = {}
        if path:
            init_kwargs["path"] = path

        if not mt5.initialize(**init_kwargs):
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        authorized = mt5.login(int(login), password=password, server=server)  # type: ignore[arg-type]
        if not authorized:
            logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return False

        logger.info("MT5 connected — account %s on %s", login, server)
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    # ── Data Retrieval ────────────────────────────
    def fetch_ohlcv(self, bars: int = LOOKBACK_BARS) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Failed to fetch rates for {self.symbol}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
            },
            inplace=True,
        )
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ── Feature Engineering ───────────────────────
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute 12-15 normalised, noise-free features.

        Returns a new DataFrame containing only the feature columns
        (rows with NaN from warm-up periods are dropped).
        """
        feat = pd.DataFrame(index=df.index)

        # 1. RSI Fast & Slow
        feat["rsi_fast"] = ta.rsi(df["Close"], length=RSI_FAST_PERIOD)
        feat["rsi_slow"] = ta.rsi(df["Close"], length=RSI_SLOW_PERIOD)

        # 2. Bollinger Band Width (normalised)
        bb = ta.bbands(df["Close"], length=BB_PERIOD, std=BB_STD)
        if bb is not None:
            upper_col = f"BBU_{BB_PERIOD}_{BB_STD}"
            lower_col = f"BBL_{BB_PERIOD}_{BB_STD}"
            mid_col = f"BBM_{BB_PERIOD}_{BB_STD}"
            feat["bb_width"] = (bb[upper_col] - bb[lower_col]) / bb[mid_col]
        else:
            feat["bb_width"] = 0.0

        # 3. Distance to EMA 50 / 200 (% from close)
        ema50 = ta.ema(df["Close"], length=EMA_FAST)
        ema200 = ta.ema(df["Close"], length=EMA_SLOW)
        feat["dist_ema50"] = (df["Close"] - ema50) / ema50 * 100
        feat["dist_ema200"] = (df["Close"] - ema200) / ema200 * 100

        # 4. ADX, +DI, -DI
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=ADX_PERIOD)
        if adx_df is not None:
            feat["adx"] = adx_df[f"ADX_{ADX_PERIOD}"]
            feat["plus_di"] = adx_df[f"DMP_{ADX_PERIOD}"]
            feat["minus_di"] = adx_df[f"DMN_{ADX_PERIOD}"]
        else:
            feat["adx"] = 0.0
            feat["plus_di"] = 0.0
            feat["minus_di"] = 0.0

        # 5. ATR (normalised by close)
        atr = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_PERIOD)
        feat["atr_norm"] = atr / df["Close"] * 100 if atr is not None else 0.0

        # 6. Volatility Ratio = current ATR / 50-bar rolling mean ATR
        if atr is not None:
            atr_ma = atr.rolling(50).mean()
            feat["volatility_ratio"] = atr / atr_ma
        else:
            feat["volatility_ratio"] = 1.0

        # 7. Tick Volume normalised (z-score over rolling 50 bars)
        vol_mean = df["Volume"].rolling(50).mean()
        vol_std = df["Volume"].rolling(50).std().replace(0, 1)
        feat["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std

        # 8. Close change (returns %)
        feat["close_return"] = df["Close"].pct_change() * 100

        # 9. EMA 50 / 200 crossover signal (binary)
        if ema50 is not None and ema200 is not None:
            feat["ema_cross"] = np.where(ema50 > ema200, 1.0, -1.0)
        else:
            feat["ema_cross"] = 0.0

        feat.dropna(inplace=True)
        return feat

    # ── Convenience ───────────────────────────────
    def get_latest_features(self, bars: int = LOOKBACK_BARS) -> pd.DataFrame:
        """Fetch fresh OHLCV and return computed features."""
        ohlcv = self.fetch_ohlcv(bars)
        return self.compute_features(ohlcv)

    def get_account_info(self) -> dict:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError("Cannot retrieve account info")
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin_free": info.margin_free,
            "profit": info.profit,
        }

    def get_symbol_info(self) -> dict:
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"Symbol info unavailable for {self.symbol}")
        return {
            "point": info.point,
            "digits": info.digits,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def get_current_tick(self) -> dict:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"Tick data unavailable for {self.symbol}")
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}
