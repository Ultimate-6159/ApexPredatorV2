# 🦅 Apex Predator V3.5 — The 5th Dimension (MoE Algorithmic Trading)

> Institutional-grade XAUUSD trading on MetaTrader 5 powered by 4 regime-specific RL agents, 5-dimensional trade management (Price × Space × Time × Liquidity × Volume), 3-stage profit locking, and risk-free pyramiding.

---

## 📌 System Overview

Apex Predator V2 solves **Catastrophic Forgetting** — the #1 failure mode of single-model RL traders — by splitting the market into 4 regimes and training a dedicated PPO agent for each one. A deterministic Meta-Router (zero ML, zero hallucination) detects the current regime on every bar close and dispatches the observation to the appropriate specialist.

### Key Capabilities

| Capability | Description |
|---|---|
| **Mixture of Experts** | 4 PPO agents, each mastering one market regime |
| **13 Noise-Free Features** | RSI, BB, EMA, ADX, ATR, Volume Z-Score, etc. |
| **ATR-Based Dynamic SL/TP** | Per-regime multipliers adapt to volatility |
| **3-Stage Profit Locking** | Break-Even (0.5×ATR) → Partial Close 50% (1.0×ATR) → Trailing Stop (0.8×ATR) |
| **News Filter** | Forex Factory calendar forces HIGH_VOLATILITY before red events |
| **Dynamic Position Sizing** | `tick_value`-based formula using equity (compound growth) |
| **Dynamic Filling Mode** | Auto-detects broker-supported IOC/FOK/RETURN + retry on error 10013 |
| **Regime-Shift Protocol** | Force-close all positions on regime change |
| **Anti-Martingale** | Max 1 position, 8% risk, circuit breaker |
| **Inference Safety Guards** | Z-Score clip ±10.0, confidence gate 65%, anomaly detection |
| **Predictive Cache** | Velocity-aware intra-bar trigger with 3-gate system (zone/time/velocity) |
| **Infinite Radar** | Cache expired → re-predict mid-candle → new cache (never blind intra-bar) |
| **Stealth Execution** | 15-point pre-fire buffer + 35pt deviation (latency compensation) |
| **Elastic Cooldown** | Step-trend reload: swing extension > 0.5×ATR + pullback < 0.2×ATR |
| **Virtual Time-Decay** | Force close after 90s without break-even (no broker SL spam) |
| **Smart Phantom Spoofer** | Dual-trigger: Phantom Sweep (0.3×ATR overshoot) + Momentum Bounce (velocity) |
| **Risk-Free Pyramiding** | 2nd position only when 1st at break-even — zero additional portfolio risk |
| **Grace Period Shield** | Protect trades <180s from regime shift whiplash (V3.5) |
| **Volume-Kinetic Resonance** | Tick volume acceleration gate — blocks fake bounces without real money flow (V3.5) |
| **Live Performance Dashboard** | Parses live logs → Win Rate, Profit Factor, Sharpe, Sortino, Calmar |

---

## 🏛️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       APEX PREDATOR V2 (V3)                              │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Perception Engine            (core/perception_engine.py)       │
│  ├── MT5 Connection (OHLCV + Tick Volume, 300-bar lookback)              │
│  ├── 13 Noise-Free Features (inf/NaN sanitized)                          │
│  └── Z-Score Normalized per agent (min std=0.01, clip ±10.0)            │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Meta-Router                  (core/meta_router.py)            │
│  ├── Deterministic regime detection (ADX / DI / Volatility Ratio)       │
│  └── News Filter override → forces HIGH_VOLATILITY before red events    │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Specialized RL Agents        (core/agents/)                   │
│  ├── 🐂 Bull Rider    (TRENDING_UP)      → [HOLD, BUY]                 │
│  ├── 🐻 Bear Hunter   (TRENDING_DOWN)    → [HOLD, SELL]                │
│  ├── 🎯 Range Sniper  (MEAN_REVERTING)   → [HOLD, BUY, SELL]            │
│  └── ⚡ Vol Assassin  (HIGH_VOLATILITY)  → [HOLD, BUY, SELL]            │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Reality Shield & Execution                                     │
│  ├── Risk Manager      (core/risk_manager.py)                           │
│  │   ├── tick_value position sizing (equity-based compound growth)       │
│  │   ├── 3-stage profit locking (Break-Even → Partial Close → Trailing) │
│  │   ├── Time stop (5-20 bars per regime) + Circuit breaker              │
│  │   └── Max drawdown 60% hard stop                                     │
│  └── Execution Engine  (core/execution_engine.py)                       │
│      ├── Dynamic filling mode detection (IOC/FOK/RETURN)                │
│      ├── Error 10013 auto-retry with alternative filling modes           │
│      ├── Live position sync before close (accurate volume/ticket)        │
│      ├── Risk-Free Pyramiding: open_pyramid_position() (V3.0)           │
│      └── Bulk close: close_all_positions() for regime shift/shutdown    │
├──────────────────────────────────────────────────────────────────────────┤
│  V3.0 — The 4D Paradigm (Time + Liquidity Dimensions)                   │
│  ├── ⏱️ Virtual Time-Decay Shield (90s kill switch, no SL modify spam)   │
│  ├── 👻 Smart Phantom Spoofer (sweep 0.3×ATR + bounce velocity trigger) │
│  └── 🔥 Risk-Free Pyramiding (Wood #2 only when Wood #1 at break-even) │
├──────────────────────────────────────────────────────────────────────────┤
│  V3.5 — The 5th Dimension (Volume-Kinetic Resonance)                    │
│  ├── 🛡️ Grace Period Shield (180s immunity from regime shift whiplash)   │
│  └── 📊 Volume-Kinetic Resonance (tick_vol > 1.5× avg to fire)         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Layer Details

### Layer 1 — Perception Engine (`core/perception_engine.py`)

Fetches live OHLCV + tick volume from MT5 and computes 13 noise-free features:

| # | Feature | Description |
|---|---|---|
| 1 | `rsi_fast` | RSI (7 periods) |
| 2 | `rsi_slow` | RSI (14 periods) |
| 3 | `bb_width` | Bollinger Band Width (normalized) |
| 4 | `dist_ema50` | Distance to EMA 50 (%) |
| 5 | `dist_ema200` | Distance to EMA 200 (%) |
| 6 | `adx` | Average Directional Index (10) |
| 7 | `plus_di` | +DI (Directional Indicator) |
| 8 | `minus_di` | −DI (Directional Indicator) |
| 9 | `atr_norm` | ATR(14) normalized by close price |
| 10 | `volatility_ratio` | ATR / 50-bar rolling mean ATR |
| 11 | `volume_zscore` | Volume Z-score (rolling 50 bars) |
| 12 | `close_return` | Price return (%) |
| 13 | `ema_cross` | EMA 50/200 crossover signal (+1/−1) |

**Data Sanitization (3-layer defense):**
1. `inf` / `-inf` → `NaN` → `0.0` (in `compute_features`)
2. Z-Score: `std = max(std, 0.01)` prevents near-zero division explosion
3. Hard clip `±10.0` before model inference (blocks billion-STD hallucinations)

Also exposes `get_symbol_info()` returning `point`, `trade_tick_value`, `trade_tick_size`, `volume_min/max/step` for dynamic position sizing.

### Layer 2 — Deterministic Meta-Router (`core/meta_router.py`)

Hard-coded logic (strictly NO ML) classifies the current market regime:

| Regime | Condition | Priority |
|---|---|---|
| `HIGH_VOLATILITY` | Volatility Ratio > 1.5 | 1st |
| `TRENDING_UP` | ADX > 23 & +DI > −DI | 2nd |
| `TRENDING_DOWN` | ADX > 23 & −DI > +DI | 3rd |
| `MEAN_REVERTING` | ADX < 23 (fallback) | 4th |

### Layer 3 — The 4 Specialized Agents (`core/agents/`)

Each agent is a PPO model trained in a custom Gymnasium environment with regime-specific reward shaping (entry cost, cooldown penalty, hold flat reward, trailing penalty, peak bonus, close profit bonus).

| Agent | Regime | Action Space | Strategy |
|---|---|---|---|
| 🐂 **Bull Rider** | `TRENDING_UP` | `[HOLD, BUY]` | Let profits run in uptrends |
| 🐻 **Bear Hunter** | `TRENDING_DOWN` | `[HOLD, SELL]` | Momentum shorting |
| 🎯 **Range Sniper** | `MEAN_REVERTING` | `[HOLD, BUY, SELL]` | Mean reversion, quick exits |
| ⚡ **Vol Assassin** | `HIGH_VOLATILITY` | `[HOLD, BUY, SELL]` | Breakout/squeeze trading |

**Inference Safety Guards:**
- **Confidence Gate:** If AI probability < 65% → forced HOLD (prevents noisy trades)
- **Post-Clip Verification:** After clipping to ±10.0, verify max ≤ 10.0 (detects NaN/Inf leaks)
- **Telemetry Logging:** Per-bar action probabilities, critic value, and anomaly detection

### Layer 4 — Reality Shield

#### Risk Manager (`core/risk_manager.py`)
- **Position Sizing:** `lot = risk_amount / (SL_points × value_per_point)` (tick_value-based)
- **Time Stop:** Force-close after N bars per regime (MR=5, HV=10, TU/TD=20)
- **Circuit Breaker:** 5 consecutive losses → 30-minute halt
- **Max Drawdown:** 60% hard stop — kills all trading permanently
- **Trade Lifecycle:** `register_open()` → `update_sl()` / `update_lot()` → `register_close()`

#### Execution Engine (`core/execution_engine.py`)
- **Dynamic Filling Mode:** `_get_filling_type()` queries `symbol_info().filling_mode` bitmask
- **Error 10013 Retry:** If order fails → cycles through IOC/FOK/RETURN automatically
- **Position Sync:** Before close → queries live MT5 position for real `volume` + `ticket`
- **Operations:** `execute_action()`, `close_open_trade()`, `modify_sl()`, `partial_close()`
- **V3.0:** `open_pyramid_position()` (fire-and-forget, no risk_manager), `close_all_positions(reason)` (bulk close all MAGIC positions)

---

## 🛡️ Risk Management (Strictly NO Martingale)

| Feature | Parameter | Description |
|---|---|---|
| **Position Sizing** | `tick_value × point / tick_size` | Calculates lot from equity (compound growth) |
| **Risk Per Trade** | `8%` | Reduced sizing for high-frequency scalping |
| **ATR SL** | `1.5 × ATR` | Dynamic stop-loss adapts to volatility |
| **ATR TP** | `0.80–1.50 × ATR` | Per-regime take-profit (MR=0.80, TU/TD=1.20, HV=1.50) |
| **Break-Even** | `0.5 × ATR` | Move SL to entry + 20pts buffer |
| **Partial Close** | `1.0 × ATR` | Close 50% of position to lock profit |
| **Trailing Stop** | `0.8 × ATR` / `0.4 × ATR` | Activation / drawdown thresholds |
| **Time Stop** | 5–20 bars | Force-close after N bars (per regime) |
| **Regime-Shift Exit** | Immediate | Close all on regime change (Clean Slate) |
| **Circuit Breaker** | 5 losses → 30 min | Halt trading after consecutive losses |
| **Max Drawdown** | `60%` | Full stop — no more trades |
| **Anti-Martingale** | Max 1 position | Never adds to a losing position |
| **Slippage Protection** | 35 points | Dynamic filling mode + deviation cap (latency-compensated) |
| **Confidence Gate** | 65% | Force HOLD if AI is uncertain |
| **Spread Gate** | ATR > 1.5×Spread | Skip bar if volatility too low for spread |
| **Live-Tick Precision** | EMA7 + EMA20 | Real-time tick bounce: gap ≥ 0.1×ATR + tick above/below EMA7 |
| **RSI Anti-Chasing** | RSI(7) 15–85 | Block BUY if RSI≥85, block SELL if RSI≤15 |
| **Predictive Cache** | 3-gate intra-bar | Velocity (≥3s) + Time (<10s) + Zone (0.2×ATR) |
| **Infinite Radar** | 10s re-predict | Cache expired → re-run AI mid-candle → new cache (V2.17) |
| **Stealth Trigger** | 15-point buffer | Fire 15pts before target (latency compensation) |
| **Elastic Cooldown** | Swing + Pullback | Swing >0.5×ATR then pullback <0.2×ATR for re-entry |
| **Time-Decay Shield** | 90 seconds | Force close if not break-even after 90s (virtual SL) |
| **Phantom Spoofer** | 0.3×ATR sweep | Dual-trigger: sweep overshoot + momentum bounce velocity |
| **Risk-Free Pyramiding** | Break-even gate | 2nd position only when 1st at break-even (max 2 total) |
| **Grace Period Shield** | 180 seconds | Shield fresh trades from regime shift whiplash (V3.5) |
| **Volume-Kinetic Resonance** | 1.5× avg tick_vol | Block Phantom Spoofer triggers without volume confirmation (V3.5) |

---

## 🔒 3-Stage Profit Locking System

The profit locking system protects unrealized profit in 3 progressive stages:

```
Entry ─────── 0.5×ATR ──────── 1.0×ATR ──────── 0.8×ATR+ ───────── TP
              🛡️ Break-Even    💰 Partial Close   📈 Trailing Stop
              SL → Entry+20pts  Close 50% lots    Lock peak, close
                                                   on 0.4×ATR retrace
```

| Stage | Trigger | Action | Effect |
|---|---|---|---|
| 🛡️ **Break-Even** | Profit ≥ 0.5 × ATR | Move SL to entry + 20 points | Risk-free trade |
| 💰 **Partial Close** | Profit ≥ 1.0 × ATR | Close 50% of position | Cash in pocket |
| 📈 **Trailing Stop** | Profit ≥ 0.8 × ATR | Track peak, close on 0.4×ATR retrace | Let winners run |

**Config:**

| Parameter | Default | Description |
|---|---|---|
| `ENABLE_BREAK_EVEN` | `True` | Master switch for break-even |
| `BREAK_EVEN_ACTIVATION_ATR` | `0.5` | ATR multiplier to activate |
| `BREAK_EVEN_BUFFER_POINTS` | `20` | Points above entry (covers commission) |
| `ENABLE_PARTIAL_CLOSE` | `True` | Master switch for partial close |
| `PARTIAL_CLOSE_ACTIVATION_ATR` | `1.0` | ATR multiplier to activate |
| `PARTIAL_CLOSE_VOLUME_PCT` | `0.5` | Fraction of lot to close (50%) |
| `TRAILING_ACTIVATION_ATR` | `0.8` | ATR multiplier to activate trailing |
| `TRAILING_DRAWDOWN_ATR` | `0.4` | ATR multiplier for max retrace |

---

## ⏱️ V3.0 — The 4D Paradigm

V3.0 adds three new dimensions to trade management beyond the original Price (AI entry) and Space (ATR SL/TP):

### Dimension 3: Time — Virtual Time-Decay Shield

Trades that fail to reach break-even within 90 seconds are force-closed via market order. The SL is computed **in Python** (virtual) — no `TRADE_ACTION_SLTP` spam to the broker. The broker only sees "open" and "close".

```
Trade Opens ────── 90s elapsed ──────── Break-Even NOT reached?
                                         → FORCE MARKET CLOSE
                                         "We don't play the hope game"
```

### Dimension 4: Liquidity — Smart Phantom Spoofer (Dual-Trigger)

Upgraded predictive cache with two independent fire triggers:

| Trigger | Condition | Purpose |
|---|---|---|
| 👻 **Phantom Sweep** | Price overshoots target by 0.3×ATR (sweep), then returns within 0.2×ATR | Catches the post-sweep reversal |
| 🏎️ **Momentum Bounce** | Price near target (0.5×ATR) + velocity > 0.2×ATR in 3s | Prevents missing strong trends |

Either trigger fires the order. Tick velocity is computed from a 10-second rolling tick history recorded every 50ms.

### Risk-Free Pyramiding (Wood #2)

When the primary position reaches break-even (SL moved to entry), a 2nd position can be opened in the same direction:

```
Wood #1: BUY @ 2000  SL→2000 (break-even) → Risk = $0
Wood #2: BUY @ 2005  SL=1995            → Risk = normal
Portfolio Risk = Same as a single trade (Wood #1 is free)
```

**Safety Gates:**
- `ENABLE_PYRAMIDING = True` must be on
- `_break_even_done = True` (primary is risk-free)
- `_pyramid_ticket is None` (no existing pyramid)
- MT5 position count < `MAX_POSITIONS` (2)
- If primary closes → pyramid closes too (`close_all_positions`)

---

## 🌌 V3.5 — The 5th Dimension Patch

V3.5 adds two surgical fixes identified from live trading log analysis to eliminate spread bleed and regime shift whiplash:

### Dimension 5a: Grace Period Shield

Prevents regime shift from killing trades that just opened. A trade younger than 180 seconds is immune to the Clean Slate protocol — its own Time-Decay Shield (90s) handles the exit instead.

```
Trade Opens ───── 5s later: Regime Shift detected!
                   └── Age=5s < 180s → SHIELDED ✔️
                       (Time-Decay will close at 90s if needed)

Trade Opens ───── 200s later: Regime Shift detected!
                   └── Age=200s > 180s → Clean Slate closes it ✔️
```

### Dimension 5b: Volume-Kinetic Resonance (VKR)

Adds a volume confirmation gate to the Phantom Spoofer triggers. Price can be manipulated (fake sweeps, fake bounces) but volume cannot be faked — real institutional flow always leaves a volume footprint.

| Gate | Condition | Purpose |
|---|---|---|
| 📊 **VKR Gate** | Current bar tick_volume > 1.5× rolling 50-bar average | Confirms real money flow behind the trigger |

```
Phantom Sweep detected + Price returns to target
  └── tick_volume = 1200 vs avg = 600
      └── 1200 > 600 × 1.5 = 900 → VOLUME CONFIRMED → FIRE! ✔️

Momentum Bounce detected + Velocity OK
  └── tick_volume = 400 vs avg = 600
      └── 400 < 600 × 1.5 = 900 → FAKE BOUNCE → BLOCKED ❌
```

**Config:**

| Parameter | Default | Description |
|---|---|---|
| `REGIME_SHIFT_GRACE_SEC` | `180` | Seconds of immunity for fresh trades |
| `VOLUME_ACCEL_MULTIPLIER` | `1.5` | Tick volume must exceed multiplier × avg to fire |

---

## 📰 News Filter (`core/news_filter.py`)

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

## ⚙️ Live Execution Pipeline (`scripts/run_live.py`)

The `LiveEngine` fires once per bar close and executes a **15-step pipeline**:

```
 1.  Sync position state with broker (detect TP/SL hits)
 2.  Fetch fresh OHLCV + compute features
 3.  Compute & store ATR (used by profit locking + dispatch)
 3a. Spread/ATR Normalization Gate — skip bar if ATR < 1.5×Spread
 3b. Profit locking check (Break-Even + Partial Close)
 3c. Cache EMA7 for ribbon filter + elastic cooldown
 4.  ATR trailing stop check (overrides AI)
 5.  Detect regime (Meta-Router)
 5b. News filter override (force HIGH_VOLATILITY if blackout)
 6.  Regime-shift protocol (Clean Slate — force close)
 7.  Risk checks (drawdown, circuit breaker)
 8.  Time stop check
 9.  Z-Score normalize observation (min std=0.01)
 9b. Hard clip features to ±10.0 (sanitize extreme values)
 9c. Post-clip verification (confirm data ≤ ±10.0, detect NaN/Inf leaks)
10.  Inference Telemetry — extract action probs + critic value
11.  Predict action (PPO model, deterministic)
12.  Map to actual action (regime-specific action space)
12c. Confidence gate — force HOLD if confidence < 65%
12b. Log telemetry (confidence %, critic value, per-action probs)
12d. Live-Tick Precision — EMA7/EMA20 bounce + Predictive Cache on defer
12e. Elastic Cooldown — swing extension + pullback gate (replaces Win-Streak)
13.  Dispatch with position-aware logic (Anti-Martingale)
14.  Log bar result

Intra-bar (V2.15 — 50 ms HFT polling when active):
•  Tick-level Risk: Sync + Break-Even + Trailing every 50 ms
•  Virtual Time-Decay: force close after 90s without break-even (V3.0)
•  Tick Recording: price history for velocity measurement (V3.0)
•  HFT Re-entry: intra-bar close → force AI re-evaluation instantly
•  Phantom Spoofer: dual-trigger (Sweep 0.3×ATR + Bounce velocity) + VKR gate (V3.5)
•  Elastic Cooldown: swing tracking (|tick − EMA7| > 0.5×ATR)
•  Infinite Radar (V2.17): cache expired → re-predict mid-candle → new cache
   Throttle: max 1 re-prediction per 10 s.  Radar stays active until bar close
   or signal fires.  Non-trending regime → radar off automatically.
```

### Position-Aware Dispatch Logic (Step 13)

| State | AI Signal | Action |
|---|---|---|
| Flat (no position) | HOLD | Do nothing (wait) |
| In position | HOLD | Maintain position (V2.18 — let TP/SL/Trailing close it) |
| Flat | BUY or SELL | Open new position |
| In BUY position | BUY | Risk-Free Pyramid (V3.0) if break-even done, else PASS |
| In BUY position | SELL | Close BUY + pyramid (don't reopen this bar) |
| In SELL position | SELL | Risk-Free Pyramid (V3.0) if break-even done, else PASS |
| In SELL position | BUY | Close SELL + pyramid (don't reopen this bar) |

### Additional Features

- MT5 auto-reconnect (5 attempts × 5s interval)
- Daily rotating log (`logs/live/live_trading.log`, 30-day retention)
- All `core/` modules log to unified `"apex_live"` logger
- Graceful shutdown on Ctrl+C (closes open trades before exit)

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
| **Close Reason Breakdown** | TP/SL_HIT, TRAILING_STOP, VOLUNTARY_CLOSE, REGIME_SHIFT, BREAK_EVEN, PARTIAL_CLOSE, … |
| **Action Distribution** | HOLD / BUY / SELL counts with bar chart |
| **Regime Distribution** | Bars per regime with bar chart |
| **AI Inference Telemetry** | Avg/Min/Max Confidence %, Critic Value stats, per-regime confidence, anomaly count |
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
│   ├── perception_engine.py     # Layer 1 — MT5 + 13 features + data sanitization
│   ├── meta_router.py           # Layer 2 — Deterministic regime detection (ADX/DI)
│   ├── news_filter.py           # News — Forex Factory calendar integration
│   ├── risk_manager.py          # Layer 4a — Position sizing, time stop, circuit breaker
│   ├── execution_engine.py      # Layer 4b — MT5 orders + dynamic filling + retry
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
│   ├── run_live.py              # 🔴 Live execution engine (15-step pipeline)
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
| Deep Learning | PyTorch (via SB3) | ≥ 2.0 |
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
| `ADX_TREND_THRESHOLD` | `23.0` | ADX threshold for trending regime |
| `VOLATILITY_RATIO_THRESHOLD` | `1.5` | ATR ratio threshold for HIGH_VOLATILITY |
| `ADX_PERIOD` | `10` | ADX calculation period |
| `ATR_PERIOD` | `14` | ATR calculation period |

### Risk Management

| Parameter | Default | Description |
|---|---|---|
| `RISK_PER_TRADE_PCT` | `8.0` | % of equity at risk per trade |
| `MAX_DRAWDOWN_PCT` | `60.0` | Hard stop — halts all trading |
| `CONSECUTIVE_LOSS_LIMIT` | `5` | Losses before circuit breaker |
| `HALT_MINUTES` | `30` | Circuit breaker cool-off |
| `MAX_HOLDING_BARS` | 5–20 | Per-regime time stop (bars) |

### Execution & SL/TP

| Parameter | Default | Description |
|---|---|---|
| `SLIPPAGE_POINTS` | `35` | Max slippage deviation (latency-compensated) |
| `ATR_SL_MULTIPLIER` | `1.5` | SL = 1.5 × ATR |
| `ATR_TP_MULTIPLIER` | 0.80–1.50 | Per-regime TP (MR=0.80, TU/TD=1.20, HV=1.50) |

### Profit Locking

| Parameter | Default | Description |
|---|---|---|
| `ENABLE_BREAK_EVEN` | `True` | Move SL to entry when profitable |
| `BREAK_EVEN_ACTIVATION_ATR` | `0.5` | Profit threshold (ATR multiplier) |
| `BREAK_EVEN_BUFFER_POINTS` | `20` | Points above entry (covers commission) |
| `ENABLE_PARTIAL_CLOSE` | `True` | Close 50% at profit target |
| `PARTIAL_CLOSE_ACTIVATION_ATR` | `1.0` | Profit threshold (ATR multiplier) |
| `PARTIAL_CLOSE_VOLUME_PCT` | `0.5` | Fraction of lot to close |
| `TRAILING_ACTIVATION_ATR` | `0.8` | Trailing activates at 0.8 × ATR profit |
| `TRAILING_DRAWDOWN_ATR` | `0.4` | Trailing closes on 0.4 × ATR retrace |

### V3.0 — The 4D Paradigm

| Parameter | Default | Description |
|---|---|---|
| `TRADE_LIFESPAN_SEC` | `90` | Force close if not break-even after N seconds |
| `ENABLE_PYRAMIDING` | `True` | Allow 2nd position when 1st at break-even |
| `MAX_POSITIONS` | `2` | Max concurrent positions per symbol |
| `PHANTOM_SWEEP_ATR` | `0.3` | Overshoot distance to detect SL sweep |
| `MOMENTUM_BOUNCE_ATR` | `0.2` | Velocity threshold (ATR in 3s) for bounce trigger |
| `MOMENTUM_WINDOW_SEC` | `3.0` | Time window for velocity measurement |

### Inference Safety Guards

| Parameter | Default | Description |
|---|---|---|
| `OBS_CLIP_RANGE` | `10.0` | Hard clip Z-Score features to ± this value |
| `CONFIDENCE_GATE_PCT` | `65.0` | Force HOLD if AI confidence < this % |

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
6. **Max 2 positions** — Anti-Martingale enforced; 2nd only when 1st at break-even (V3.0)
7. **Clip before predict** — Data sanitization must happen before model inference
8. **Dynamic filling** — Never hardcode `ORDER_FILLING_IOC`; always auto-detect

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
