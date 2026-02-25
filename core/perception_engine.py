"""
Layer 1 — Perception Engine
Ingests OHLCV data from MT5 and computes 12-15 noise-free features.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    EMA_FAST,
    EMA_SLOW,
    HTF_TREND_EMA_PERIOD,
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


# ═══════════════════════════════════════════════
# V6.0: Streaming Indicators (Zero-Latency Perception)
# ═══════════════════════════════════════════════
class StreamingIndicators:
    """Stateful streaming indicator calculator (numpy-based).

    Replaces pandas/ta library computations with pure numpy.
    - First call: full vectorised computation, saves state (~1-2 ms)
    - Subsequent bar: single-bar incremental update (~0.01 ms)
    - Same bar count (re-prediction): full recompute (still fast numpy)
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._n_bars: int = 0

        # Previous bar
        self._prev_close: float = 0.0
        self._prev_high: float = 0.0
        self._prev_low: float = 0.0

        # RSI (Wilder EMA: alpha = 1/period)
        self._rsi_fast_ag: float = 0.0
        self._rsi_fast_al: float = 0.0
        self._rsi_slow_ag: float = 0.0
        self._rsi_slow_al: float = 0.0

        # EMA
        self._ema_fast: float = 0.0
        self._ema_slow: float = 0.0

        # Bollinger Bands rolling window
        self._bb_closes: deque[float] = deque(maxlen=BB_PERIOD)

        # ATR (Wilder smoothing)
        self._atr_val: float = 0.0
        self._atr_buf: deque[float] = deque(maxlen=50)

        # ADX (Wilder smoothing)
        self._sm_tr: float = 0.0
        self._sm_pdm: float = 0.0
        self._sm_mdm: float = 0.0
        self._adx_val: float = 0.0
        self._adx_ready: bool = False
        self._dx_buf: deque[float] = deque(maxlen=ADX_PERIOD)

        # Volume rolling window
        self._vol_buf: deque[float] = deque(maxlen=50)

    # ── Smart Dispatch ────────────────────────
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full numpy or incremental update based on state."""
        n = len(df)
        if not self._initialized:
            return self._full_compute(df)
        if n > self._n_bars:
            return self._incremental(df)
        # Same bar count (re-prediction) — full recompute
        return self._full_compute(df)

    # ── Full Vectorised Computation ───────────
    def _full_compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full numpy computation replacing ta library."""
        closes = df["Close"].values.astype(np.float64)
        highs = df["High"].values.astype(np.float64)
        lows = df["Low"].values.astype(np.float64)
        volumes = df["Volume"].values.astype(np.float64)
        n = len(closes)

        # 1. RSI Fast & Slow (Wilder EMA, alpha = 1/period, adjust=False)
        rsi_fast = np.full(n, np.nan)
        rsi_slow = np.full(n, np.nan)
        alpha_f = 1.0 / RSI_FAST_PERIOD
        alpha_s = 1.0 / RSI_SLOW_PERIOD
        f_ag, f_al = 0.0, 0.0
        s_ag, s_al = 0.0, 0.0
        for i in range(1, n):
            change = closes[i] - closes[i - 1]
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            f_ag = alpha_f * gain + (1 - alpha_f) * f_ag
            f_al = alpha_f * loss + (1 - alpha_f) * f_al
            rsi_fast[i] = 100.0 if f_al == 0 else 100.0 - 100.0 / (1.0 + f_ag / f_al)
            s_ag = alpha_s * gain + (1 - alpha_s) * s_ag
            s_al = alpha_s * loss + (1 - alpha_s) * s_al
            rsi_slow[i] = 100.0 if s_al == 0 else 100.0 - 100.0 / (1.0 + s_ag / s_al)

        # 2. EMA 50 / 200 (alpha = 2/(span+1), adjust=False)
        k_f = 2.0 / (EMA_FAST + 1)
        k_s = 2.0 / (EMA_SLOW + 1)
        ema_fast_arr = np.empty(n)
        ema_slow_arr = np.empty(n)
        ema_fast_arr[0] = closes[0]
        ema_slow_arr[0] = closes[0]
        for i in range(1, n):
            ema_fast_arr[i] = closes[i] * k_f + ema_fast_arr[i - 1] * (1 - k_f)
            ema_slow_arr[i] = closes[i] * k_s + ema_slow_arr[i - 1] * (1 - k_s)

        # 3. Bollinger Band Width (SMA + std, ddof=0)
        bb_width = np.full(n, np.nan)
        for i in range(BB_PERIOD - 1, n):
            w = closes[i - BB_PERIOD + 1 : i + 1]
            m = np.mean(w)
            s = np.std(w, ddof=0)
            if m > 0:
                bb_width[i] = 2 * BB_STD * s / m

        # 4. Distance to EMA (%)
        dist_ema50 = np.where(
            ema_fast_arr > 0,
            (closes - ema_fast_arr) / ema_fast_arr * 100,
            np.nan,
        )
        dist_ema200 = np.where(
            ema_slow_arr > 0,
            (closes - ema_slow_arr) / ema_slow_arr * 100,
            np.nan,
        )

        # 5. True Range + ATR (Wilder EMA, alpha = 1/period)
        tr = np.empty(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.full(n, np.nan)
        alpha_atr = 1.0 / ATR_PERIOD
        running_atr = 0.0
        count_atr = 0
        for i in range(1, n):
            running_atr = (
                tr[i] if count_atr == 0
                else alpha_atr * tr[i] + (1 - alpha_atr) * running_atr
            )
            count_atr += 1
            if count_atr >= ATR_PERIOD:
                atr[i] = running_atr

        atr_norm = np.where(
            (~np.isnan(atr)) & (closes > 0), atr / closes * 100, np.nan
        )

        # Volatility ratio: ATR / rolling 50-bar mean ATR
        vol_ratio = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(atr[i]):
                continue
            start = max(0, i - 49)
            win = atr[start : i + 1]
            valid = win[~np.isnan(win)]
            if len(valid) >= 50:
                ma = np.mean(valid)
                if ma > 0:
                    vol_ratio[i] = atr[i] / ma

        # 6. ADX, +DI, -DI (Wilder smoothing)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            if up > down and up > 0:
                plus_dm[i] = up
            if down > up and down > 0:
                minus_dm[i] = down

        p = ADX_PERIOD
        sm_tr_arr = np.full(n, np.nan)
        sm_pdm_arr = np.full(n, np.nan)
        sm_mdm_arr = np.full(n, np.nan)
        if n > p:
            sm_tr_arr[p - 1] = np.sum(tr[:p])
            sm_pdm_arr[p - 1] = np.sum(plus_dm[:p])
            sm_mdm_arr[p - 1] = np.sum(minus_dm[:p])
            for i in range(p, n):
                sm_tr_arr[i] = sm_tr_arr[i - 1] - sm_tr_arr[i - 1] / p + tr[i]
                sm_pdm_arr[i] = sm_pdm_arr[i - 1] - sm_pdm_arr[i - 1] / p + plus_dm[i]
                sm_mdm_arr[i] = sm_mdm_arr[i - 1] - sm_mdm_arr[i - 1] / p + minus_dm[i]

        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        for i in range(p - 1, n):
            if not np.isnan(sm_tr_arr[i]) and sm_tr_arr[i] > 0:
                plus_di[i] = 100 * sm_pdm_arr[i] / sm_tr_arr[i]
                minus_di[i] = 100 * sm_mdm_arr[i] / sm_tr_arr[i]
                di_sum = plus_di[i] + minus_di[i]
                dx[i] = (
                    100 * abs(plus_di[i] - minus_di[i]) / di_sum
                    if di_sum > 0 else 0.0
                )

        adx_arr = np.full(n, np.nan)
        adx_start = 2 * p - 2
        if n > adx_start:
            valid_dx = dx[p - 1 : adx_start + 1]
            valid_dx = valid_dx[~np.isnan(valid_dx)]
            if len(valid_dx) >= p:
                adx_arr[adx_start] = np.mean(valid_dx)
                for i in range(adx_start + 1, n):
                    if not np.isnan(dx[i]) and not np.isnan(adx_arr[i - 1]):
                        adx_arr[i] = (adx_arr[i - 1] * (p - 1) + dx[i]) / p

        # 7. Volume Z-Score (rolling 50, ddof=1)
        vol_zscore = np.full(n, np.nan)
        for i in range(49, n):
            w = volumes[i - 49 : i + 1]
            m = np.mean(w)
            s = np.std(w, ddof=1)
            vol_zscore[i] = (volumes[i] - m) / s if s > 0 else 0.0

        # 8. Close return (%)
        close_return = np.empty(n)
        close_return[0] = np.nan
        close_return[1:] = np.diff(closes) / closes[:-1] * 100

        # 9. EMA Cross
        ema_cross = np.where(ema_fast_arr > ema_slow_arr, 1.0, -1.0)

        # Build DataFrame
        feat = pd.DataFrame(
            {
                "rsi_fast": rsi_fast,
                "rsi_slow": rsi_slow,
                "bb_width": bb_width,
                "dist_ema50": dist_ema50,
                "dist_ema200": dist_ema200,
                "adx": adx_arr,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "atr_norm": atr_norm,
                "volatility_ratio": vol_ratio,
                "volume_zscore": vol_zscore,
                "close_return": close_return,
                "ema_cross": ema_cross,
            },
            index=df.index,
        )
        feat.dropna(inplace=True)
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.fillna(0.0, inplace=True)

        # ── Save state for incremental updates ──
        self._prev_close = float(closes[-1])
        self._prev_high = float(highs[-1])
        self._prev_low = float(lows[-1])

        self._rsi_fast_ag = f_ag
        self._rsi_fast_al = f_al
        self._rsi_slow_ag = s_ag
        self._rsi_slow_al = s_al

        self._ema_fast = float(ema_fast_arr[-1])
        self._ema_slow = float(ema_slow_arr[-1])

        bb_start = max(0, n - BB_PERIOD)
        self._bb_closes = deque(closes[bb_start:].tolist(), maxlen=BB_PERIOD)

        self._atr_val = running_atr
        valid_atr = atr[~np.isnan(atr)]
        self._atr_buf = deque(valid_atr[-50:].tolist(), maxlen=50)

        if n > p and not np.isnan(sm_tr_arr[-1]):
            self._sm_tr = float(sm_tr_arr[-1])
            self._sm_pdm = float(sm_pdm_arr[-1])
            self._sm_mdm = float(sm_mdm_arr[-1])

        valid_adx = adx_arr[~np.isnan(adx_arr)]
        if len(valid_adx) > 0:
            self._adx_val = float(valid_adx[-1])
            self._adx_ready = True

        vol_start = max(0, n - 50)
        self._vol_buf = deque(volumes[vol_start:].tolist(), maxlen=50)

        self._initialized = True
        self._n_bars = n
        return feat

    # ── Single-Bar Incremental Update ─────────
    def _incremental(self, df: pd.DataFrame) -> pd.DataFrame:
        """Update with a single new bar — O(1) streaming."""
        latest = df.iloc[-1]
        h = float(latest["High"])
        l = float(latest["Low"])
        c = float(latest["Close"])
        v = float(latest["Volume"])
        change = c - self._prev_close

        # RSI
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        alpha_f = 1.0 / RSI_FAST_PERIOD
        alpha_s = 1.0 / RSI_SLOW_PERIOD
        self._rsi_fast_ag = alpha_f * gain + (1 - alpha_f) * self._rsi_fast_ag
        self._rsi_fast_al = alpha_f * loss + (1 - alpha_f) * self._rsi_fast_al
        rsi_fast = (
            100.0 if self._rsi_fast_al == 0
            else 100.0 - 100.0 / (1.0 + self._rsi_fast_ag / self._rsi_fast_al)
        )
        self._rsi_slow_ag = alpha_s * gain + (1 - alpha_s) * self._rsi_slow_ag
        self._rsi_slow_al = alpha_s * loss + (1 - alpha_s) * self._rsi_slow_al
        rsi_slow = (
            100.0 if self._rsi_slow_al == 0
            else 100.0 - 100.0 / (1.0 + self._rsi_slow_ag / self._rsi_slow_al)
        )

        # EMA
        k_f = 2.0 / (EMA_FAST + 1)
        k_s = 2.0 / (EMA_SLOW + 1)
        self._ema_fast = c * k_f + self._ema_fast * (1 - k_f)
        self._ema_slow = c * k_s + self._ema_slow * (1 - k_s)

        # Bollinger Band Width
        self._bb_closes.append(c)
        bb_arr = np.array(self._bb_closes)
        bb_mid = float(np.mean(bb_arr))
        bb_std = float(np.std(bb_arr, ddof=0))
        bb_width = 2 * BB_STD * bb_std / bb_mid if bb_mid > 0 else 0.0

        # Distance to EMA
        dist_ema50 = (
            (c - self._ema_fast) / self._ema_fast * 100
            if self._ema_fast > 0 else 0.0
        )
        dist_ema200 = (
            (c - self._ema_slow) / self._ema_slow * 100
            if self._ema_slow > 0 else 0.0
        )

        # ATR
        tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        alpha_atr = 1.0 / ATR_PERIOD
        self._atr_val = alpha_atr * tr + (1 - alpha_atr) * self._atr_val
        self._atr_buf.append(self._atr_val)
        atr_norm = self._atr_val / c * 100 if c > 0 else 0.0
        vol_ratio = 1.0
        if len(self._atr_buf) >= 50:
            atr_ma = float(np.mean(list(self._atr_buf)))
            if atr_ma > 0:
                vol_ratio = self._atr_val / atr_ma

        # ADX
        up = h - self._prev_high
        down = self._prev_low - l
        pdm = up if (up > down and up > 0) else 0.0
        mdm = down if (down > up and down > 0) else 0.0
        ap = ADX_PERIOD
        self._sm_tr = self._sm_tr - self._sm_tr / ap + tr
        self._sm_pdm = self._sm_pdm - self._sm_pdm / ap + pdm
        self._sm_mdm = self._sm_mdm - self._sm_mdm / ap + mdm
        plus_di_v = 100 * self._sm_pdm / self._sm_tr if self._sm_tr > 0 else 0.0
        minus_di_v = 100 * self._sm_mdm / self._sm_tr if self._sm_tr > 0 else 0.0
        di_sum = plus_di_v + minus_di_v
        dx_v = 100 * abs(plus_di_v - minus_di_v) / di_sum if di_sum > 0 else 0.0
        if self._adx_ready:
            self._adx_val = (self._adx_val * (ap - 1) + dx_v) / ap
        else:
            self._dx_buf.append(dx_v)
            if len(self._dx_buf) >= ap:
                self._adx_val = float(np.mean(list(self._dx_buf)))
                self._adx_ready = True

        # Volume Z-Score
        self._vol_buf.append(v)
        vol_arr = np.array(self._vol_buf)
        vol_m = float(np.mean(vol_arr))
        vol_s = float(np.std(vol_arr, ddof=1)) if len(vol_arr) > 1 else 1.0
        vol_zscore = (v - vol_m) / vol_s if vol_s > 0 else 0.0

        # Close return
        close_return = (
            change / self._prev_close * 100
            if self._prev_close > 0 else 0.0
        )

        # EMA cross
        ema_cross = 1.0 if self._ema_fast > self._ema_slow else -1.0

        # Update previous bar
        self._prev_close = c
        self._prev_high = h
        self._prev_low = l
        self._n_bars += 1

        feat = {
            "rsi_fast": rsi_fast,
            "rsi_slow": rsi_slow,
            "bb_width": bb_width,
            "dist_ema50": dist_ema50,
            "dist_ema200": dist_ema200,
            "adx": self._adx_val,
            "plus_di": plus_di_v,
            "minus_di": minus_di_v,
            "atr_norm": atr_norm,
            "volatility_ratio": vol_ratio,
            "volume_zscore": vol_zscore,
            "close_return": close_return,
            "ema_cross": ema_cross,
        }
        # Sanitize
        for k, val in feat.items():
            if not np.isfinite(val):
                feat[k] = 0.0

        return pd.DataFrame([feat], index=[df.index[-1]])

    # ── V7.0: Expose Raw ATR ──────────────────
    @property
    def last_atr(self) -> float:
        """Return the last computed raw ATR value (Wilder smoothing)."""
        return self._atr_val


class PerceptionEngine:
    """Connects to MT5 and produces a feature DataFrame."""

    def __init__(self, symbol: str = SYMBOL, timeframe: str = "M5") -> None:
        self.symbol = symbol
        self.timeframe_name = timeframe
        self.timeframe = _TF_MAP.get(timeframe, mt5.TIMEFRAME_M5)
        self._connected = False
        self._streaming = StreamingIndicators()  # V6.0

    # ── V7.0: True ATR Accessor ───────────────────
    @property
    def current_atr(self) -> float:
        """Return the current raw ATR from streaming indicators.

        This is the proper Wilder-smoothed True Range, NOT the simple
        (High-Low) average that was previously used.  Fixes SL/TP
        accuracy for all ATR-dependent calculations.
        """
        return self._streaming.last_atr

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

    # ── Feature Engineering (V6.0: streaming) ─────────
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 13 normalised, noise-free features.

        V6.0: Delegates to StreamingIndicators for zero-latency
        numpy-based computation with incremental updates.

        Returns a new DataFrame containing only the feature columns
        (rows with NaN from warm-up periods are dropped).
        """
        return self._streaming.compute(df)

    # ── V7.0: Multi-Timeframe Confluence ──────────
    def get_htf_trend_bias(self) -> int:
        """Return higher-timeframe (H1) trend direction.

        Computes EMA on H1 close prices to determine overall trend:
          +1 = bullish (price > H1 EMA)
          -1 = bearish (price < H1 EMA)
           0 = neutral / insufficient data
        """
        try:
            n_bars = HTF_TREND_EMA_PERIOD + 5
            rates = mt5.copy_rates_from_pos(
                self.symbol, mt5.TIMEFRAME_H1, 0, n_bars,
            )
            if rates is None or len(rates) < HTF_TREND_EMA_PERIOD:
                return 0

            closes = rates["close"].astype(np.float64)
            # EMA (alpha = 2 / (period + 1), adjust=False)
            k = 2.0 / (HTF_TREND_EMA_PERIOD + 1)
            ema = float(closes[0])
            for i in range(1, len(closes)):
                ema = float(closes[i]) * k + ema * (1 - k)

            if float(closes[-1]) > ema:
                return 1
            elif float(closes[-1]) < ema:
                return -1
        except Exception:
            logger.debug("HTF trend bias calculation failed", exc_info=True)
        return 0

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

    # ── V6.0: DOM Order Flow ──────────────────
    def get_market_depth(self) -> dict | None:
        """Return aggregated DOM bid/ask volume (V6.0).

        Uses ``mt5.market_book_add`` / ``mt5.market_book_get`` to read
        Level 2 data.  Returns ``None`` if DOM is unavailable for the
        current symbol.
        """
        if not mt5.market_book_add(self.symbol):
            return None
        book = mt5.market_book_get(self.symbol)
        if book is None or len(book) == 0:
            return None

        bid_volume = 0.0
        ask_volume = 0.0
        for entry in book:
            if entry.type == mt5.BOOK_TYPE_BUY:
                bid_volume += entry.volume
            elif entry.type == mt5.BOOK_TYPE_SELL:
                ask_volume += entry.volume

        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
