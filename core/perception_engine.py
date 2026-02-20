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
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, ADXIndicator
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
logger = logging.getLogger("apex_live")

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
        feat["rsi_fast"] = RSIIndicator(df["Close"], window=RSI_FAST_PERIOD).rsi()
        feat["rsi_slow"] = RSIIndicator(df["Close"], window=RSI_SLOW_PERIOD).rsi()

        # 2. Bollinger Band Width (normalised)
        bb = BollingerBands(df["Close"], window=BB_PERIOD, window_dev=int(BB_STD))
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        feat["bb_width"] = (bb_upper - bb_lower) / bb_mid

        # 3. Distance to EMA 50 / 200 (% from close)
        ema50 = EMAIndicator(df["Close"], window=EMA_FAST).ema_indicator()
        ema200 = EMAIndicator(df["Close"], window=EMA_SLOW).ema_indicator()
        feat["dist_ema50"] = (df["Close"] - ema50) / ema50 * 100
        feat["dist_ema200"] = (df["Close"] - ema200) / ema200 * 100

        # 4. ADX, +DI, -DI
        adx_ind = ADXIndicator(df["High"], df["Low"], df["Close"], window=ADX_PERIOD)
        feat["adx"] = adx_ind.adx()
        feat["plus_di"] = adx_ind.adx_pos()
        feat["minus_di"] = adx_ind.adx_neg()

        # 5. ATR (normalised by close)
        atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=ATR_PERIOD).average_true_range()
        feat["atr_norm"] = atr / df["Close"] * 100

        # 6. Volatility Ratio = current ATR / 50-bar rolling mean ATR
        atr_ma = atr.rolling(50).mean()
        feat["volatility_ratio"] = atr / atr_ma

        # 7. Tick Volume normalised (z-score over rolling 50 bars)
        vol_mean = df["Volume"].rolling(50).mean()
        vol_std = df["Volume"].rolling(50).std().replace(0, 1)
        feat["volume_zscore"] = (df["Volume"] - vol_mean) / vol_std

        # 8. Close change (returns %)
        feat["close_return"] = df["Close"].pct_change() * 100

        # 9. EMA 50 / 200 crossover signal (binary)
        feat["ema_cross"] = np.where(ema50 > ema200, 1.0, -1.0)

        feat.dropna(inplace=True)

        # Sanitize: replace infinities and clip extreme raw values
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.fillna(0.0, inplace=True)

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
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def get_current_tick(self) -> dict:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"Tick data unavailable for {self.symbol}")
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}
