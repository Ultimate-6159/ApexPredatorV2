# 🦅 Apex Predator V2 — Mixture of Experts (MoE) Algorithmic Trading System

> Institutional-grade XAUUSD trading on MetaTrader 5 powered by 4 regime-specific Reinforcement Learning agents, ATR-adaptive risk management, and a Forex Factory news filter.

---

## 📌 System Overview

Apex Predator V2 solves **Catastrophic Forgetting** — the #1 failure mode of single-model RL traders — by splitting the market into 4 regimes and training a dedicated PPO agent for each one. A deterministic Meta-Router (zero ML, zero hallucination) detects the current regime on every bar close and dispatches the observation to the appropriate specialist.

### Key Capabilities

| Capability | Description |
|---|---|
| **Mixture of Experts** | 4 PPO agents, each mastering one market regime |
| **13 Noise-Free Features** | RSI, BB, EMA, ADX, ATR, Volume Z-Score, etc. |
| **ATR-Based Dynamic SL/TP** | Per-regime multipliers adapt to volatility |
| **ATR-Based Trailing Stop** | Activation & drawdown thresholds scale with ATR *(V3)* |
| **News Filter** | Forex Factory calendar forces HIGH_VOLATILITY before red events *(V3)* |
| **Dynamic Position Sizing** | `tick_value`-based formula using equity (compound growth) |
| **Regime-Shift Protocol** | Force-close all positions on regime change |
| **Anti-Martingale** | Max 1 position, fixed 0.5% risk, circuit breaker |
| **Live Performance Dashboard** | Parses live logs → Win Rate, Profit Factor, Max Drawdown |

---

## 🏛️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          APEX PREDATOR V2                                │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Perception Engine            (core/perception_engine.py)       │
│  ├── MT5 Connection (OHLCV + Tick Volume, 300-bar lookback)              │
│  └── 13 Noise-Free Features → Z-Score Normalized per agent              │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Meta-Router                  (core/meta_router.py)            │
│  ├── Deterministic regime detection (ADX / DI / Volatility Ratio)       │
│  └── News Filter override → forces HIGH_VOLATILITY before red events    │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Specialized RL Agents        (core/agents/)                   │
│  ├── 🐂 Bull Rider    (TRENDING_UP)      → [HOLD, BUY]                  │
│  ├── 🐻 Bear Hunter   (TRENDING_DOWN)    → [HOLD, SELL]                 │
│  ├── 🎯 Range Sniper  (MEAN_REVERTING)   → [HOLD, BUY, SELL]            │
│  └── ⚡ Vol Assassin  (HIGH_VOLATILITY)  → [HOLD, BUY, SELL]            │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Reality Shield & Execution                                     │
│  ├── Risk Manager      (core/risk_manager.py)                           │
│  │   ├── tick_value position sizing (equity-based compound growth)       │
│  │   ├── ATR trailing stop (activation 1×ATR, drawdown 0.5×ATR)         │
│  │   ├── Time stop (5-20 bars per regime) + Circuit breaker              │
│  │   └── Max drawdown 15% hard stop                                     │
│  └── Execution Engine  (core/execution_engine.py)                       │
│      ├── MT5 orders with ATR SL/TP + slippage protection (30 pts)       │
│      └── Anti-Martingale: max 1 position at any time                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Layer Details

### Layer 1 — Perception Engine (`core/perception_engine.py`)

Fetches live OHLCV + tick volume from MT5 and computes 13 noise-free features:

| Feature | Description |
|---|---|
| `rsi_fast` | RSI (7 periods) |
| `rsi_slow` | RSI (14 periods) |
| `bb_width` | Bollinger Band Width (normalized) |
| `dist_ema50` | Distance to EMA 50 (%) |
| `dist_ema200` | Distance to EMA 200 (%) |
| `adx` | Average Directional Index |
| `plus_di` | +DI (Directional Indicator) |
| `minus_di` | −DI (Directional Indicator) |
| `atr_norm` | ATR normalized by close price |
| `volatility_ratio` | ATR / 50-bar rolling mean ATR |
| `volume_zscore` | Volume Z-score (rolling 50 bars) |
| `close_return` | Price return (%) |
| `ema_cross` | EMA 50/200 crossover signal (+1/−1) |

Also exposes `get_symbol_info()` returning `point`, `trade_tick_value`, `trade_tick_size`, `volume_min/max/step` for dynamic position sizing.

### Layer 2 — Deterministic Meta-Router (`core/meta_router.py`)

Hard-coded logic (strictly NO ML) classifies the current market regime:

| Regime | Condition | Priority |
|---|---|---|
| `HIGH_VOLATILITY` | Volatility Ratio > 1.5 | 1st |
| `TRENDING_UP` | ADX > 25 & +DI > −DI | 2nd |
| `TRENDING_DOWN` | ADX > 25 & −DI > +DI | 3rd |
| `MEAN_REVERTING` | ADX < 25 (fallback) | 4th |

### Layer 3 — The 4 Specialized Agents (`core/agents/`)

Each agent is a PPO model trained in a custom Gymnasium environment with regime-specific reward shaping (entry cost, cooldown penalty, hold flat reward, trailing penalty, peak bonus, close profit bonus).

| Agent | Regime | Action Space | Strategy |
|---|---|---|---|
| 🐂 **Bull Rider** | `TRENDING_UP` | `[HOLD, BUY]` | Let profits run in uptrends |
| 🐻 **Bear Hunter** | `TRENDING_DOWN` | `[HOLD, SELL]` | Momentum shorting |
| 🎯 **Range Sniper** | `MEAN_REVERTING` | `[HOLD, BUY, SELL]` | Mean reversion, quick exits |
| ⚡ **Vol Assassin** | `HIGH_VOLATILITY` | `[HOLD, BUY, SELL]` | Breakout/squeeze trading |

### Layer 4 — Reality Shield

- **Risk Manager** (`core/risk_manager.py`): Position sizing, time stops, circuit breaker, max drawdown
- **Execution Engine** (`core/execution_engine.py`): MT5 order execution with ATR SL/TP, `ORDER_FILLING_IOC`, 30-point slippage deviation

---

## 🛡️ Risk Management (Strictly NO Martingale)

| Feature | Parameter | Description |
|---|---|---|
| **Position Sizing** | `tick_value * point / tick_size` | Calculates lot from equity (compound growth) |
| **Risk Per Trade** | `0.5%` | Fixed percentage of equity at risk |
| **ATR SL** | `1.5 × ATR` | Dynamic stop-loss adapts to volatility |
| **ATR TP** | `1.5–3.0 × ATR` | Per-regime take-profit multiplier |
| **ATR Trailing Stop** | `1.0 × ATR` / `0.5 × ATR` | Activation / drawdown thresholds *(V3)* |
| **Time Stop** | 5–20 bars | Force-close after N bars (per regime) |
| **Regime-Shift Exit** | Immediate | Close all on regime change |
| **Circuit Breaker** | 5 losses → 30 min | Halt trading after consecutive losses |
| **Max Drawdown** | `15%` | Full stop — no more trades |
| **Anti-Martingale** | Max 1 position | Never adds to a losing position |
| **Slippage Protection** | 30 points | `ORDER_FILLING_IOC` + deviation cap |

---

## 📰 V3: News Filter (`core/news_filter.py`)

Fetches the Forex Factory economic calendar (weekly JSON endpoint) and detects imminent high-impact events.

```
Flow:
  Every bar → NewsFilter.is_blackout()
           → If red event within 15 min:
                Force regime → HIGH_VOLATILITY (Vol Assassin takes over)
```

| Parameter | Default | Description |
|---|---|---|
| `NEWS_FILTER_ENABLED` | `True` | Master switch |
| `NEWS_BLACKOUT_MINUTES` | `15` | Minutes before event to activate |
| `NEWS_CURRENCIES` | `["USD"]` | Currencies to watch |
| `NEWS_CACHE_HOURS` | `4` | Re-fetch interval |

- Uses `urllib` only (stdlib, no `requests` dependency)
- Caches calendar in memory — re-fetches every 4 hours
- Logs `NEWS OVERRIDE: {regime} → HIGH_VOLATILITY (event: {title})`

---

## 🔁 V3: ATR-Based Dynamic Trailing Stop

Replaces fixed-point trailing (V2: 300/200 points) with ATR-adaptive thresholds.

| Parameter | Default | Formula |
|---|---|---|
| `TRAILING_ACTIVATION_ATR` | `1.0` | Activate at `1.0 × ATR` profit (in points) |
| `TRAILING_DRAWDOWN_ATR` | `0.5` | Close if retraces `0.5 × ATR` from peak |

**Behavior:**
- **Ranging market** (low ATR): Tight trailing — locks in small profits quickly
- **Trending market** (high ATR): Wide trailing — lets winners run to full potential
- Logged as `TRAILING_STOP` close reason in dashboard

---

## ⚙️ Live Execution Pipeline (`scripts/run_live.py`)

The `LiveEngine` fires once per bar close and executes a 13-step pipeline:

```
 1.  Sync position state with broker (detect TP/SL hits)
 2.  Fetch fresh OHLCV + compute features
 3.  Compute & store ATR (used by trailing stop + dispatch)
 4.  ATR trailing stop check (overrides AI)
 5.  Detect regime (Meta-Router)
 5b. News filter override (force HIGH_VOLATILITY if blackout)
 6.  Regime-shift protocol (Clean Slate — force close)
 7.  Risk checks (drawdown, circuit breaker)
 8.  Time stop check
 9.  Z-Score normalize observation
10.  Predict action (PPO model)
11.  Map to actual action (regime-specific action space)
12.  Dispatch with position-aware logic (Anti-Martingale)
13.  Log bar result
```

**Additional features:**
- MT5 auto-reconnect (5 attempts × 5s interval)
- Daily rotating log (`logs/live/live_trading.log`, 30-day retention)
- All `core/` modules log to unified `"apex_live"` logger
- Graceful shutdown on Ctrl+C (closes open trades)

---

## 📊 Live Performance Dashboard (`scripts/analyze_live_logs.py`)

Parses live trading logs and produces a comprehensive performance report:

```bash
python -m scripts.analyze_live_logs                     # all logs
python -m scripts.analyze_live_logs --date 2026-02-20   # single day
python -m scripts.analyze_live_logs --last 7            # last 7 days
python -m scripts.analyze_live_logs --csv trades.csv    # export to CSV
```

**Dashboard Metrics:**

| Section | Metrics |
|---|---|
| **Account** | Starting/Ending Balance, Net P&L, Return % |
| **Overall Trade Metrics** | Win Rate, Profit Factor, Expectancy, Avg Win/Loss, Max Drawdown |
| **Institutional Risk Metrics** | Sharpe Ratio, Sortino Ratio, Calmar Ratio, Payoff Ratio, Gross P/L |
| **Win Rate by Agent** | Per-regime: Trades, Wins, WR%, PF, Net P&L, Avg P&L |
| **Close Reason Breakdown** | TP/SL_HIT, TRAILING_STOP, VOLUNTARY_CLOSE, REGIME_SHIFT, TIME_STOP, … |
| **Action Distribution** | HOLD / BUY / SELL counts with bar chart |
| **Regime Distribution** | Bars per regime with bar chart |
| **Recent Trades** | Last 10 trades: ticket, direction, regime, lot, entry, close, PnL, reason |

**Institutional Metric Thresholds:**

| Metric | Poor | OK | Good | Elite |
|---|---|---|---|---|
| Profit Factor | < 1.0 | 1.0–1.4 | 1.5–2.0 | > 2.0 |
| Sharpe Ratio | < 0.5 | 0.5–1.0 | 1.0–2.0 | > 2.0 |
| Sortino Ratio | < 1.0 | 1.0–1.5 | 1.5–3.0 | > 3.0 |
| Calmar Ratio | < 0.5 | 0.5–1.0 | 1.0–3.0 | > 3.0 |
| Payoff Ratio | < 1.0 | 1.0–1.5 | 1.5–2.5 | > 2.5 |

---

## 📁 Project Structure

```
ApexPredatorV2/
├── config/
│   └── __init__.py              # All tunable parameters & constants
├── core/
│   ├── __init__.py
│   ├── perception_engine.py     # Layer 1 — MT5 + 13 features + symbol info
│   ├── meta_router.py           # Layer 2 — Deterministic regime detection
│   ├── news_filter.py           # V3 — Forex Factory calendar integration
│   ├── risk_manager.py          # Layer 4a — Position sizing, time stop, circuit breaker
│   ├── execution_engine.py      # Layer 4b — MT5 order execution with SL/TP
│   ├── backtest_engine.py       # Historical backtesting
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Base RL agent class (train/load/predict)
│   │   ├── bull_rider.py        # 🐂 TRENDING_UP specialist
│   │   ├── bear_hunter.py       # 🐻 TRENDING_DOWN specialist
│   │   ├── range_sniper.py      # 🎯 MEAN_REVERTING specialist
│   │   └── vol_assassin.py      # ⚡ HIGH_VOLATILITY specialist
│   └── environments/
│       ├── __init__.py
│       └── trading_env.py       # Gymnasium env with ATR TP/SL simulation
├── training/
│   ├── __init__.py
│   ├── train_agents.py          # Training pipeline (live MT5 data)
│   └── training_logger.py       # Metrics capture (obs_stats, episodes, actions)
├── scripts/
│   ├── __init__.py
│   ├── run_live.py              # 🔴 Live execution engine (13-step pipeline)
│   ├── analyze_live_logs.py     # 📊 Live performance dashboard
│   ├── analyze_training.py      # Training session analysis & comparison
│   ├── run_backtest.py          # Historical backtest runner
│   ├── collect_data.py          # Download data for offline training
│   └── train_offline.py         # Train from saved parquet data
├── logs/
│   ├── live/                    # Daily rotating live trading logs
│   └── training/                # Per-regime training sessions
│       ├── trending_up/         #   └── {session_id}/
│       ├── trending_down/       #       ├── config.json
│       ├── mean_reverting/      #       ├── obs_stats.json
│       └── high_volatility/     #       ├── episodes.parquet / .csv
│                                #       ├── timesteps.parquet / .csv
│                                #       ├── training_metrics.parquet / .csv
│                                #       ├── episode_actions.json
│                                #       └── summary.json
├── models/                      # Saved PPO weights (.zip) per regime
├── data/                        # Historical data storage (.parquet)
├── main.py                      # Legacy entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 💻 Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| RL Framework | Stable Baselines3 (PPO) | ≥ 2.1.0 |
| Broker API | MetaTrader5 | ≥ 5.0.45 |
| Gym Environment | Gymnasium | ≥ 0.29.0 |
| Technical Analysis | `ta` library | ≥ 0.11.0 |
| Data Processing | pandas + numpy | ≥ 2.0 / ≥ 1.24 |
| Serialization | pyarrow (parquet) | ≥ 14.0 |
| Environment Vars | python-dotenv | ≥ 1.0 |
| Gym Compatibility | shimmy | ≥ 1.3.0 |
| News Calendar | urllib (stdlib) | — |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/Ultimate-6159/ApexPredatorV2.git
cd ApexPredatorV2

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```ini
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_SYMBOL=XAUUSDm           # Broker-specific suffix (e.g., Exness Trial)
```

### 3. Train All 4 Agents

```bash
# Option A: Live MT5 data
python -m training.train_agents

# Option B: Offline (collect first)
python -m scripts.collect_data --bars 50000 --output data/xauusd.parquet
python -m scripts.train_offline --data data/xauusd.parquet --timesteps 200000
```

### 4. Analyze Training

```bash
python -m scripts.analyze_training --all
python -m scripts.analyze_training --regime trending_up --compare
python -m scripts.analyze_training --regime trending_up --session 20260219_214033
```

### 5. Backtest

```bash
python -m scripts.run_backtest --bars 5000 --balance 10000
```

### 6. Go Live 🔴

```bash
python -m scripts.run_live --timeframe M5 --symbol XAUUSDm
```

### 7. Monitor Performance

```bash
python -m scripts.analyze_live_logs --last 7
python -m scripts.analyze_live_logs --csv trades.csv
```

---

## ⚙️ Configuration Reference (`config/__init__.py`)

### Market

| Parameter | Default | Description |
|---|---|---|
| `SYMBOL` | `"XAUUSD"` (from env) | Trading symbol |
| `TIMEFRAME_NAME` | `"M5"` | Primary timeframe |
| `LOOKBACK_BARS` | `300` | Bars to fetch for feature calculation |

### Regime Detection

| Parameter | Default | Description |
|---|---|---|
| `ADX_TREND_THRESHOLD` | `25.0` | ADX threshold for trending regime |
| `VOLATILITY_RATIO_THRESHOLD` | `1.5` | ATR ratio threshold for HIGH_VOLATILITY |
| `ADX_PERIOD` | `14` | ADX calculation period |
| `ATR_PERIOD` | `14` | ATR calculation period |

### Risk Management

| Parameter | Default | Description |
|---|---|---|
| `RISK_PER_TRADE_PCT` | `0.5` | % of equity at risk per trade |
| `MAX_DRAWDOWN_PCT` | `15.0` | Hard stop — halts all trading |
| `CONSECUTIVE_LOSS_LIMIT` | `5` | Losses before circuit breaker |
| `HALT_MINUTES` | `30` | Circuit breaker cool-off |
| `MAX_HOLDING_BARS` | 5–20 | Per-regime time stop (bars) |

### Execution & SL/TP

| Parameter | Default | Description |
|---|---|---|
| `SLIPPAGE_POINTS` | `30` | Max slippage deviation |
| `ATR_SL_MULTIPLIER` | `1.5` | SL = 1.5 × ATR |
| `ATR_TP_MULTIPLIER` | 1.5–3.0 | Per-regime TP multiplier |
| `TRAILING_ACTIVATION_ATR` | `1.0` | Trailing activates at 1.0 × ATR profit |
| `TRAILING_DRAWDOWN_ATR` | `0.5` | Trailing closes on 0.5 × ATR retrace |

### News Filter

| Parameter | Default | Description |
|---|---|---|
| `NEWS_FILTER_ENABLED` | `True` | Enable Forex Factory integration |
| `NEWS_BLACKOUT_MINUTES` | `15` | Pre-event blackout window |
| `NEWS_CURRENCIES` | `["USD"]` | Currencies to watch |
| `NEWS_CACHE_HOURS` | `4` | Calendar re-fetch interval |

### Training

| Parameter | Default | Description |
|---|---|---|
| `TRAINING_TIMESTEPS` | `200,000` | PPO training steps per agent |
| `TRAINING_LOG_FREQ` | `1,000` | Log every N steps |
| `TRAINING_SAVE_FREQ` | `10,000` | Save checkpoint every N steps |

---

## 📊 Training Logs

```
logs/training/{regime_lower}/{session_id}/
├── config.json              # Training configuration snapshot
├── obs_stats.json           # Feature mean/std for Z-Score normalization
├── episodes.parquet/.csv    # Episode rewards & lengths
├── timesteps.parquet/.csv   # Per-timestep detailed metrics
├── training_metrics.parquet/.csv  # SB3 policy/value loss, entropy
├── episode_actions.json     # Action distribution per episode
└── summary.json             # Final summary (duration, best/worst reward)
```

---

## 🤖 Development Rules

1. **NO Martingale** — Never grid, never average down, never scale into losers
2. **Meta-Router is deterministic** — Regime detection must stay hard-coded, no ML
3. **Credentials via `os.getenv()`** — Never hardcode secrets
4. **Type hints everywhere** — `from __future__ import annotations` in every file
5. **All loggers → `"apex_live"`** — Unified log routing for live engine
6. **One position max** — Anti-Martingale enforced at execution layer

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
