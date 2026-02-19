# 🦅 Apex Predator V2: Mixture of Experts (MoE) Trading System

## 📌 System Overview
An institutional-grade algorithmic trading system for XAUUSD (M5/M15) utilizing MetaTrader 5 and Reinforcement Learning (Stable Baselines3). It implements a Mixture of Experts (MoE) architecture to solve "Catastrophic Forgetting" by routing market states to 4 highly specialized RL agents based on real-time regime detection.

## 🏛️ 4-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     APEX PREDATOR V2                            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Perception Engine                                     │
│  ├── MT5 Connection (OHLCV + Tick Volume)                       │
│  └── 14 Noise-Free Features (RSI, BB, EMA, ADX, ATR, etc.)      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Meta-Router (Deterministic - NO ML)                   │
│  └── Regime Detection: TRENDING_UP | TRENDING_DOWN |            │
│                        HIGH_VOLATILITY | MEAN_REVERTING         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Specialized RL Agents (PPO)                           │
│  ├── 🐂 Bull Rider    (TRENDING_UP)     → [HOLD, BUY]           │
│  ├── 🐻 Bear Hunter   (TRENDING_DOWN)   → [HOLD, SELL]          │
│  ├── 🎯 Range Sniper  (MEAN_REVERTING)  → [HOLD, BUY, SELL]     │
│  └── ⚡ Vol Assassin  (HIGH_VOLATILITY) → [HOLD, BUY, SELL]     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Reality Shield & Execution                            │
│  ├── Risk Manager (0.5% risk, Circuit Breaker, Time Stop)       │
│  └── Execution Engine (MT5 Orders with SL/TP)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Perception Engine (`core/perception_engine.py`)
Ingests live OHLCV and Tick Volume data via MT5. Computes 14 noise-free features:

| Feature | Description |
|---------|-------------|
| `rsi_fast` | RSI (7 periods) |
| `rsi_slow` | RSI (14 periods) |
| `bb_width` | Bollinger Band Width (normalized) |
| `dist_ema50` | Distance to EMA 50 (%) |
| `dist_ema200` | Distance to EMA 200 (%) |
| `adx` | Average Directional Index |
| `plus_di` | +DI (Directional Indicator) |
| `minus_di` | -DI (Directional Indicator) |
| `atr_norm` | ATR normalized by close price |
| `volatility_ratio` | ATR / 50-bar rolling mean ATR |
| `volume_zscore` | Volume Z-score (rolling 50 bars) |
| `close_return` | Price return (%) |
| `ema_cross` | EMA 50/200 crossover signal (+1/-1) |

### 2. Deterministic Meta-Router (`core/meta_router.py`)
A hard-coded logic layer (Strictly NO Machine Learning to avoid hallucination) that classifies the current market regime:

| Regime | Condition | Priority |
|--------|-----------|----------|
| `HIGH_VOLATILITY` | Volatility Ratio > 1.5 | 1st |
| `TRENDING_UP` | ADX > 25 & +DI > -DI | 2nd |
| `TRENDING_DOWN` | ADX > 25 & -DI > +DI | 3rd |
| `MEAN_REVERTING` | ADX < 25 (Fallback) | 4th |

### 3. The 4 Specialized Agents (`core/agents/`)
The Meta-Router dispatches the state to **ONLY ONE** of the following agents:

| Agent | Regime | Action Space | Strategy |
|-------|--------|--------------|----------|
| 🐂 **Bull Rider** | `TRENDING_UP` | `[HOLD, BUY]` | Let profits run |
| 🐻 **Bear Hunter** | `TRENDING_DOWN` | `[HOLD, SELL]` | Momentum shorting |
| 🎯 **Range Sniper** | `MEAN_REVERTING` | `[HOLD, BUY, SELL]` | Mean reversion, quick exits |
| ⚡ **Vol Assassin** | `HIGH_VOLATILITY` | `[HOLD, BUY, SELL]` | Breakout/squeeze trading |

### 4. Reality Shield & Execution
- **Risk Manager** (`core/risk_manager.py`): Position sizing, time stops, circuit breaker
- **Execution Engine** (`core/execution_engine.py`): MT5 order execution with SL/TP

## 🛡️ Institutional Risk Management (Strictly NO Martingale)

| Feature | Description |
|---------|-------------|
| **Anti-Martingale** | Fixed 0.5% risk per trade |
| **Time Stop** | Force-close after N bars (5-20 depending on regime) |
| **Regime-Shift Exit** | Immediate exit on regime contradiction |
| **Circuit Breaker** | 30-min halt after 5 consecutive losses |
| **Max Drawdown** | Full stop at 15% drawdown |

## 📁 Project Structure

```
ApexPredatorV2/
├── config/
│   └── __init__.py          # Global configurations & constants
├── core/
│   ├── __init__.py
│   ├── perception_engine.py # Layer 1 - MT5 + Feature extraction
│   ├── meta_router.py       # Layer 2 - Regime detection
│   ├── risk_manager.py      # Layer 4a - Risk management
│   ├── execution_engine.py  # Layer 4b - Order execution
│   ├── backtest_engine.py   # Historical backtesting
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py    # Base RL agent class
│   │   ├── bull_rider.py    # TRENDING_UP specialist
│   │   ├── bear_hunter.py   # TRENDING_DOWN specialist
│   │   ├── range_sniper.py  # MEAN_REVERTING specialist
│   │   └── vol_assassin.py  # HIGH_VOLATILITY specialist
│   └── environments/
│       ├── __init__.py
│       └── trading_env.py   # Gymnasium environment
├── training/
│   ├── __init__.py
│   └── train_agents.py      # Training pipeline (live MT5)
├── scripts/
│   ├── __init__.py
│   ├── run_backtest.py      # Run historical backtest
│   ├── collect_data.py      # Download data for offline use
│   └── train_offline.py     # Train from saved data
├── data/                    # Historical data storage
├── models/                  # Saved model weights (.zip)
├── main.py                  # Live trading entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 💻 Tech Stack

| Component | Technology |
|-----------|------------|
| Core Engine | Python 3.10+ |
| RL Framework | Stable Baselines3 (PPO) |
| Broker API | MetaTrader5 |
| Technical Analysis | `ta` library |
| Data Processing | pandas, numpy |
| Environment | Gymnasium |
| Security | python-dotenv |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Ultimate-6159/ApexPredatorV2.git
cd ApexPredatorV2

# Create virtual environment (recommended: Python 3.11)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your MT5 credentials
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

### 3. Training Agents

```bash
# Option A: Train with live MT5 connection
python -m training.train_agents

# Option B: Collect data first, then train offline
python -m scripts.collect_data --bars 50000 --output data/xauusd.parquet
python -m scripts.train_offline --data data/xauusd.parquet --timesteps 200000
```

### 4. Run Backtest

```bash
python -m scripts.run_backtest --bars 5000 --balance 10000
```

### 5. Live Trading

```bash
python main.py
```

## ⚙️ Configuration Parameters (`config/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL` | `"XAUUSD"` | Trading symbol |
| `TIMEFRAME_NAME` | `"M5"` | Primary timeframe |
| `LOOKBACK_BARS` | `300` | Bars for feature calculation |
| `ADX_TREND_THRESHOLD` | `25.0` | ADX threshold for trend |
| `VOLATILITY_RATIO_THRESHOLD` | `1.5` | High volatility threshold |
| `RISK_PER_TRADE_PCT` | `0.5` | Risk per trade (%) |
| `MAX_DRAWDOWN_PCT` | `15.0` | Max drawdown before halt |
| `CONSECUTIVE_LOSS_LIMIT` | `5` | Losses before circuit breaker |
| `HALT_MINUTES` | `30` | Circuit breaker duration |
| `TRAINING_TIMESTEPS` | `200,000` | RL training steps |

## 🤖 Development Guidelines

1. **Rule 1:** NEVER implement Martingale or grid strategies
2. **Rule 2:** Keep Meta-Router logic separate from RL training
3. **Rule 3:** Always use `os.getenv()` for credentials
4. **Rule 4:** Use explicit type hints and OOP patterns
5. **Rule 5:** Handle MT5 disconnections gracefully

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
