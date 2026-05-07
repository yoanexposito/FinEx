# Opening Range Breakout Trading Engine (Starter)

This project is a beginner-friendly algorithmic trading engine focused on an Opening Range Breakout (ORB) strategy.

It includes:

- Backtesting on historical intraday bar data (CSV)
- Strategy and risk management modules
- A live trading runtime with pluggable market-data and broker adapters
- A CLI for running backtests or live mode

## Strategy Overview

The ORB strategy:

1. Builds an opening range from the first N minutes after market open.
2. Looks for breakouts above or below that range after it is set.
3. Takes at most one trade per symbol per day.
4. Uses stop loss, optional take profit, and end-of-day exit.

## Project Structure

- `main.py` - CLI entrypoint
- `orb_trader/config.py` - settings dataclass
- `orb_trader/models.py` - core data models (bars, positions, trades)
- `orb_trader/strategy.py` - ORB signal and position management logic
- `orb_trader/risk.py` - position sizing and daily risk gates
- `orb_trader/backtest.py` - historical simulation engine
- `orb_trader/live.py` - real-time loop using provider/broker interfaces
- `orb_trader/interfaces.py` - abstract interfaces for API integration

## Quick Start

### 1) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a backtest

Input CSV must have columns:

- `symbol`
- `timestamp` (ISO-8601, e.g. `2026-01-15T09:35:00-05:00`)
- `open`
- `high`
- `low`
- `close`
- `volume`

Run:

```bash
python main.py backtest --csv data/sample.csv
```

Optional args:

- `--initial-equity`
- `--risk-per-trade-pct`
- `--opening-range-minutes`
- `--market-open`
- `--market-close`

### 3) Run live mode (paper/live wiring later)

```bash
python main.py live
```

The default live runtime uses placeholder adapters. Replace them with real implementations in `orb_trader/interfaces.py` + your concrete classes.

## Notes

- This is a learning-oriented starter, not financial advice.
- Backtests are simplified and may differ from real fills.
- Always paper trade before real capital.
