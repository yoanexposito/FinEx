# FinEx — Algorithmic Trading Engine

A production-style intraday trading system built in Python. Implements two price-action strategies, a vectorised backtesting engine, a real-time Streamlit dashboard, and live order execution through the Alpaca API.

> **Disclaimer:** For educational purposes only. Paper-trade before risking real capital.

---

## What this demonstrates

| Skill | Where |
|---|---|
| Strategy design pattern — pluggable strategies behind a common interface | `orb_trader/strategy.py`, `orb_trader/strategy_vwap.py` |
| Clean layered architecture (models → strategy → risk → backtest → live → UI) | `orb_trader/` package |
| Real broker API integration with proper error handling | `orb_trader/adapters/alpaca.py` |
| Financial metrics from scratch — Sharpe, profit factor, drawdown, alpha | `orb_trader/backtest.py` |
| Risk management — fractional position sizing, daily loss circuit breaker | `orb_trader/risk.py` |
| Secure credential handling — env vars only, never in source or UI | `.env.example`, `dashboard/app.py` |
| 67-test pytest suite covering unit and integration scenarios | `tests/` |
| CI via GitHub Actions | `.github/workflows/tests.yml` |

---

## Strategies

### Opening Range Breakout (ORB)

The first *N* minutes after market open define a high/low range. Once that window closes, a breakout above the high triggers a long entry; a breakout below the low triggers a short. One trade per symbol per day.

```
09:30 ────────────────── range window ──────── 10:00 ──────────────────▶
         [builds high/low]                          [breakout or nothing]

          range high ─────────────────────────────── ↑ LONG entry
          ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
          range low  ─────────────────────────────── ↓ SHORT entry
```

**Key parameters**

| Parameter | Default | Effect |
|---|---|---|
| Opening Range (min) | 30 | Width of the range-building window |
| Breakout Buffer (bps) | 5 | Extra distance above/below range before entry |
| Stop Loss Buffer (bps) | 0 | Additional padding on the stop |
| Target R:R | 1.5 | Take-profit set at 1.5× the risk per share |

---

### VWAP Breakout

VWAP (Volume-Weighted Average Price) is the intraday fair-value benchmark used by institutional desks. Price crossing above VWAP signals buyers are in control; crossing below signals sellers.

```
Price ─┐
       │    ╭──────────────  VWAP (resets each day)
       │   ╭╯
  ─────┼──╳─────────────────  ← crossover → LONG entry
       │  ╱
       │ ╱  (price was below VWAP, now confirms above)
       ╰╯
```

The strategy blocks entries during a configurable warmup period (VWAP is unreliable on only 1–2 bars) and after an entry cutoff time (avoids low-liquidity afternoon chop).

**Key parameters**

| Parameter | Default | Effect |
|---|---|---|
| VWAP Buffer (bps) | 5 | Minimum distance above/below VWAP to trigger |
| Stop Distance (bps from VWAP) | 20 | Stop placed this far beyond VWAP at entry |
| Warmup Bars | 6 | Bars after open before first entry is allowed |
| Entry Cutoff (hour ET) | 14 | No new entries after 2 pm |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   dashboard/app.py                       │
│              (Streamlit — backtest + live UI)            │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
┌─────────▼──────────┐    ┌────────────▼───────────┐
│  orb_trader/       │    │  orb_trader/            │
│  backtest.py       │    │  live.py                │
│  (simulation)      │    │  (polling loop)         │
└─────────┬──────────┘    └────────────┬────────────┘
          │                            │
          └──────────┬─────────────────┘
                     │
        ┌────────────┴────────────┐
        │                        │
┌───────▼───────┐    ┌───────────▼────────┐
│ strategy.py   │    │ strategy_vwap.py   │
│ (ORB)         │    │ (VWAP Breakout)    │
└───────┬───────┘    └───────────┬────────┘
        │                        │
        └────────────┬───────────┘
                     │
            ┌────────▼────────┐
            │   risk.py       │   ← position sizing
            │   models.py     │   ← Bar, Signal, Position, Trade
            │   config.py     │   ← EngineConfig dataclass
            └────────┬────────┘
                     │
        ┌────────────▼────────────┐
        │  adapters/alpaca.py     │
        │  (Alpaca market data    │
        │   + broker)             │
        └─────────────────────────┘
```

---

## Project structure

```
FinEx/
├── dashboard/
│   └── app.py               Streamlit dashboard (backtest + live trading)
├── orb_trader/
│   ├── adapters/
│   │   └── alpaca.py        Alpaca market-data provider and broker adapter
│   ├── backtest.py          Backtesting engine + metrics (Sharpe, drawdown, etc.)
│   ├── config.py            EngineConfig dataclass — all tunable parameters
│   ├── interfaces.py        Abstract Broker / MarketDataProvider interfaces
│   ├── live.py              Polling-based live trading loop
│   ├── models.py            Core data models: Bar, Signal, Position, Trade
│   ├── risk.py              Position sizing and daily loss gate
│   ├── strategy.py          Opening Range Breakout strategy
│   └── strategy_vwap.py     VWAP Breakout strategy
├── tests/
│   ├── conftest.py          Shared test helpers (make_bar, drive)
│   ├── test_backtest_engine.py
│   ├── test_risk.py
│   ├── test_strategy_orb.py
│   └── test_strategy_vwap.py
├── data/
│   └── sample.csv           Minimal sample bar data for offline testing
├── main.py                  CLI entrypoint (backtest / live modes)
├── requirements.txt
└── requirements-dev.txt
```

---

## Quick start

### 1. Clone and set up the environment

```bash
git clone https://github.com/yoanexposito/FinEx.git
cd FinEx
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Alpaca credentials

```bash
cp .env.example .env
# Open .env and paste your keys from alpaca.markets → Paper Trading → API Keys
```

```env
export ALPACA_API_KEY=your_key_here
export ALPACA_SECRET_KEY=your_secret_here
```

> Alpaca paper-trading accounts are **free** — sign up at [alpaca.markets](https://alpaca.markets) to get keys immediately.

### 3. Launch the dashboard

```bash
source .env
streamlit run dashboard/app.py
```

Navigate to **http://localhost:8501**.

---

## Dashboard walkthrough

### Backtest tab

1. Open **⚙️ Strategy Config** in the sidebar and choose a strategy (ORB or VWAP).
2. Tune parameters. Risk management settings (R:R, position size %, daily loss limit) apply to both strategies.
3. In the **Backtest** tab, choose **Alpaca Historical Data** as the data source.
4. Pick a date range, bar timeframe (1–5 min works best for intraday strategies), and one or more symbols from the ticker browser.
5. Click **▶ Run Backtest**.

The results include:

| Metric | What it tells you |
|---|---|
| **Total Return** | Overall % gain/loss over the period |
| **α vs B&H** | Alpha over an equal-weight buy-and-hold benchmark |
| **Max Drawdown** | Largest peak-to-trough equity decline |
| **Sharpe Ratio** | Risk-adjusted return (daily, annualised ×√252). > 1 is solid |
| **Profit Factor** | Gross profit ÷ gross loss. > 1.5 indicates an edge |
| **Win Rate** | % of trades closed profitably |
| **Equity Curve** | Strategy (green) vs Buy & Hold (dashed amber) over time |
| **Monthly Returns** | Heatmap — spot seasonality or regime changes at a glance |

### Live Trading tab

Requires a connected Alpaca account (paper or live — toggle in the sidebar).

1. Enter comma-separated symbols and a poll interval.
2. Click **▶ Start Trader** — the ORB loop runs in a background thread.
3. Open positions and recent orders refresh on each page load.
4. Click **⏹ Stop Trader** for a clean shutdown.

---

## CLI usage

```bash
# Backtest from a local CSV
python main.py backtest --csv data/sample.csv --initial-equity 50000

# Live paper trading
source .env
python main.py live --symbols SPY,QQQ --poll-seconds 60 --paper
```

CSV format required for `--csv`:

```
symbol,timestamp,open,high,low,close,volume
SPY,2026-01-05T09:30:00-05:00,500.00,500.40,499.80,500.10,1200000
```

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

```
67 passed in 0.04s
```

The suite covers:
- Risk manager — position sizing, daily loss gate boundary conditions
- Backtesting primitives — slippage, commission, P&L, profit factor, streaks, buy-and-hold
- ORB strategy — range construction, signal generation, exit logic, force-close
- VWAP strategy — VWAP calculation, crossover detection, warmup/cutoff guards, exit logic

---

## Tech stack

| Layer | Library |
|---|---|
| Dashboard | [Streamlit](https://streamlit.io) |
| Charts | [Plotly](https://plotly.com/python/) |
| Broker / market data | [alpaca-py](https://github.com/alpacahq/alpaca-py) |
| Data manipulation | [pandas](https://pandas.pydata.org) |
| Testing | [pytest](https://pytest.org) |
| Environment config | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## Security

- API keys are loaded **exclusively from environment variables** — never entered in the UI, never stored in source code.
- `.env` is listed in `.gitignore`. Only `.env.example` (with placeholder values) is tracked.
- The dashboard has no credential input fields by design.

---

## Roadmap

- [ ] WebSocket-based live data feed (replace polling)
- [ ] Walk-forward validation tooling
- [ ] Additional strategies (momentum, mean reversion)
- [ ] Persistent trade journal with SQLite
- [ ] Market-hours and holiday guard for the live trader
