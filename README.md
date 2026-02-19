# 🦅 Apex Predator V2: Mixture of Experts (MoE) Trading System

## 📌 System Overview
An institutional-grade algorithmic trading system for XAUUSD (M5/M15) utilizing MetaTrader 5 and Reinforcement Learning (Stable Baselines3). It implements a Mixture of Experts (MoE) architecture to solve "Catastrophic Forgetting" by routing market states to 4 highly specialized RL agents based on real-time regime detection.

## 🏛️ 4-Layer Architecture

### 1. Perception Engine
Ingests live OHLCV and Tick Volume data via MT5. Computes 12-15 noise-free features (e.g., RSI Fast, BB Width, Distance to EMA50/200) and Max Pain sentiment.

### 2. Deterministic Meta-Router
A hard-coded logic layer (Strictly NO Machine Learning to avoid hallucination) that classifies the current market regime using ADX, DMI, and Volatility Ratio:
- `TRENDING_UP`: ADX > 25 & +DI > -DI
- `TRENDING_DOWN`: ADX > 25 & -DI > +DI
- `HIGH_VOLATILITY`: Volatility Ratio > 1.5 or during major news events
- `MEAN_REVERTING`: ADX < 25 (Fallback condition)

### 3. The 4 Specialized Agents (RL)
The Meta-Router dispatches the state to ONLY ONE of the following agents:
- 🐂 **Agent 1 (The Bull Rider):** Regime = `TRENDING_UP`. Action Space = `[BUY, HOLD]`. Trained to let profits run.
- 🐻 **Agent 2 (The Bear Hunter):** Regime = `TRENDING_DOWN`. Action Space = `[SELL, HOLD]`. Trained for momentum shorting.
- 🎯 **Agent 3 (The Range Sniper):** Regime = `MEAN_REVERTING`. Action Space = `[BUY, SELL, HOLD]`. Optimized for >85% Win Rate with severe time-decay penalties for holding too long.
- ⚡ **Agent 4 (The VolAssassin):** Regime = `HIGH_VOLATILITY`. Action Space = `[BUY, SELL, HOLD]`. Specializes in breakout/squeeze trading with strict, tight stop-losses.

### 4. Reality Shield & Execution
The risk management and order execution layer. Converts AI actions to MT5 orders with strict safety nets.

## 🛡️ Institutional Risk Management (Strictly NO Martingale)
- **Dynamic Anti-Martingale:** Position sizing is strictly calculated based on a fixed 0.5% risk of the current account balance.
- **Time Stop (Guillotine):** Force-closes trades that exceed their regime's maximum holding bars (e.g., 5-10 bars for Range Sniper) regardless of PnL.
- **Regime-Shift Emergency Exit:** Immediately cuts losses if the market regime flips entirely while an order is open.
- **Circuit Breaker:** Halts trading for 30 minutes after 5 consecutive losses. Halts the system completely if Max Total Drawdown reaches 15%.

## 💻 Tech Stack
- **Core Engine:** Python 3.10+
- **RL Framework:** Stable Baselines3 (PPO/SAC)
- **Broker Integration:** `MetaTrader5` library
- **Data & Features:** `pandas`, `numpy`, `TA-Lib` (or `pandas-ta`)
- **Security:** `python-dotenv` (Credentials must be loaded from `.env`)

## 🤖 Instructions for GitHub Copilot Workspace
As an AI coding assistant, your objective is to build this project strictly following the architecture defined above. 
1. **Rule 1:** NEVER suggest or implement Martingale or grid strategies.
2. **Rule 2:** Enforce the separation of concerns. Do not mix the Meta-Router logic with the RL Agent training logic.
3. **Rule 3:** Always use `os.getenv()` for MT5 credentials.
4. **Rule 4:** When generating Python code, prioritize clean architecture (OOP), explicit type hinting, and robust error handling for API disconnects.
