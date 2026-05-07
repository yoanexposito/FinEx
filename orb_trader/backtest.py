import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

from orb_trader.config import EngineConfig
from orb_trader.models import Bar, Position, Side, Trade
from orb_trader.risk import RiskManager
from orb_trader.strategy import OpeningRangeBreakoutStrategy


def _compute_sharpe(snapshots: List[Tuple[datetime, float]], initial_equity: float) -> float:
    if len(snapshots) < 2:
        return 0.0
    daily: Dict[date, float] = {}
    for ts, eq in snapshots:
        daily[ts.date()] = eq
    sorted_eq = [initial_equity] + [daily[d] for d in sorted(daily)]
    returns = [
        (sorted_eq[i] - sorted_eq[i - 1]) / max(sorted_eq[i - 1], 1e-9)
        for i in range(1, len(sorted_eq))
    ]
    if len(returns) < 2:
        return 0.0
    mu = mean(returns)
    sigma = stdev(returns)
    return (mu / sigma) * math.sqrt(252) if sigma > 0 else 0.0


def _apply_slippage(price: float, side: Side, slippage_bps: float, is_entry: bool) -> float:
    slip = slippage_bps / 10_000.0
    if side == Side.LONG:
        return price * (1 + slip) if is_entry else price * (1 - slip)
    return price * (1 - slip) if is_entry else price * (1 + slip)


def _commission(qty: int, per_share: float, minimum: float) -> float:
    return max(minimum, qty * per_share)


def _signed_pnl(side: Side, entry_price: float, exit_price: float, qty: int) -> float:
    if side == Side.LONG:
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def load_bars_from_csv(path: Path) -> List[Bar]:
    bars: List[Bar] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            bars.append(
                Bar(
                    symbol=row["symbol"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    bars.sort(key=lambda b: (b.timestamp, b.symbol))
    return bars


@dataclass
class BacktestResult:
    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade]
    win_rate_pct: float
    avg_trade_pnl: float
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class Backtester:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.strategy = OpeningRangeBreakoutStrategy(config)
        self.risk = RiskManager(config)

    def run(self, bars: Iterable[Bar], initial_equity: float = 100_000.0) -> BacktestResult:
        bars = list(bars)
        equity = initial_equity
        peak_equity = equity
        max_dd = 0.0
        last_bar_by_symbol: Dict[str, Bar] = {}

        open_positions: Dict[str, Position] = {}
        trades: List[Trade] = []
        day_start_equity: Dict[date, float] = {}
        equity_snapshots: List[Tuple[datetime, float]] = [
            (bars[0].timestamp, initial_equity)
        ] if bars else []

        for bar in bars:
            last_bar_by_symbol[bar.symbol] = bar
            day = bar.timestamp.date()
            if day not in day_start_equity:
                day_start_equity[day] = equity

            self.strategy.update_opening_range(bar)
            position = open_positions.get(bar.symbol)

            if position:
                exit_hit = self.strategy.exit_check(bar, position)
                force_close = self.strategy.should_force_close(bar, position)
                if exit_hit or force_close:
                    if exit_hit:
                        raw_exit_price, reason = exit_hit
                    else:
                        raw_exit_price, reason = bar.close, "end_of_day"

                    exit_price = _apply_slippage(
                        raw_exit_price,
                        position.side,
                        self.config.slippage_bps,
                        is_entry=False,
                    )
                    gross_pnl = _signed_pnl(
                        position.side,
                        position.entry_price,
                        exit_price,
                        position.qty,
                    )
                    fees = _commission(
                        position.qty,
                        self.config.commission_per_share,
                        self.config.min_commission,
                    )
                    net_pnl = gross_pnl - fees
                    equity += net_pnl
                    trades.append(
                        Trade(
                            symbol=position.symbol,
                            side=position.side,
                            qty=position.qty,
                            entry_time=position.entry_time,
                            exit_time=bar.timestamp,
                            entry_price=position.entry_price,
                            exit_price=exit_price,
                            gross_pnl=gross_pnl,
                            net_pnl=net_pnl,
                            exit_reason=reason,
                        )
                    )
                    del open_positions[bar.symbol]
                    equity_snapshots.append((bar.timestamp, equity))
                    peak_equity = max(peak_equity, equity)
                    dd = (peak_equity - equity) / max(peak_equity, 1e-9)
                    max_dd = max(max_dd, dd)
                    continue

            if open_positions.get(bar.symbol):
                continue

            can_trade = self.risk.can_trade(day_start_equity[day], equity)
            if not can_trade:
                continue

            signal = self.strategy.maybe_generate_signal(bar, has_open_position=False)
            if not signal:
                continue

            entry_price = _apply_slippage(
                signal.trigger_price,
                signal.side,
                self.config.slippage_bps,
                is_entry=True,
            )
            qty = self.risk.position_size(signal.side, equity, entry_price, signal.stop_price)
            if qty <= 0:
                continue

            entry_fee = _commission(qty, self.config.commission_per_share, self.config.min_commission)
            equity -= entry_fee
            target_price = self.strategy.target_price(entry_price, signal.stop_price, signal.side)

            open_positions[bar.symbol] = Position(
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                entry_time=signal.timestamp,
                entry_price=entry_price,
                stop_price=signal.stop_price,
                target_price=target_price,
            )
            self.strategy.register_trade(signal.symbol, signal.timestamp)

            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / max(peak_equity, 1e-9)
            max_dd = max(max_dd, dd)

        for symbol, position in list(open_positions.items()):
            # Fallback close for any positions still open after data ends.
            last_bar: Optional[Bar] = last_bar_by_symbol.get(symbol)
            if not last_bar:
                continue

            exit_price = _apply_slippage(
                last_bar.close,
                position.side,
                self.config.slippage_bps,
                is_entry=False,
            )
            gross_pnl = _signed_pnl(position.side, position.entry_price, exit_price, position.qty)
            fees = _commission(
                position.qty,
                self.config.commission_per_share,
                self.config.min_commission,
            )
            net_pnl = gross_pnl - fees
            equity += net_pnl
            equity_snapshots.append((last_bar.timestamp, equity))
            trades.append(
                Trade(
                    symbol=position.symbol,
                    side=position.side,
                    qty=position.qty,
                    entry_time=position.entry_time,
                    exit_time=last_bar.timestamp,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    exit_reason="end_of_data",
                )
            )

        wins = [t for t in trades if t.net_pnl > 0]
        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
        avg_trade = mean([t.net_pnl for t in trades]) if trades else 0.0
        total_return_pct = ((equity / initial_equity) - 1.0) * 100.0
        sharpe = _compute_sharpe(equity_snapshots, initial_equity)

        return BacktestResult(
            initial_equity=initial_equity,
            final_equity=equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd * 100.0,
            sharpe_ratio=sharpe,
            trades=trades,
            win_rate_pct=win_rate,
            avg_trade_pnl=avg_trade,
            equity_curve=equity_snapshots,
        )
