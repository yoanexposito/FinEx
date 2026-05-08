import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

from orb_trader.config import EngineConfig
from orb_trader.models import Bar, Position, Side, Trade
from orb_trader.risk import RiskManager
from orb_trader.strategy import OpeningRangeBreakoutStrategy
from orb_trader.strategy_vwap import VWAPBreakoutStrategy


def compute_buy_and_hold(
    bars: List[Bar], initial_equity: float
) -> Tuple[List[Tuple[datetime, float]], float]:
    """Equal-weight buy-and-hold benchmark over the same bar set.

    Returns (equity_curve, total_return_pct).
    """
    if not bars:
        return [], 0.0

    # First close price seen for each symbol
    first_price: Dict[str, float] = {}
    # Last close price seen for each symbol on each calendar day
    daily_close: Dict[Tuple[date, str], float] = {}
    last_ts_per_day: Dict[date, datetime] = {}

    for bar in bars:
        if bar.symbol not in first_price:
            first_price[bar.symbol] = bar.close
        d = bar.timestamp.date()
        daily_close[(d, bar.symbol)] = bar.close
        if d not in last_ts_per_day or bar.timestamp > last_ts_per_day[d]:
            last_ts_per_day[d] = bar.timestamp

    symbols = list(first_price.keys())
    n = max(len(symbols), 1)
    alloc = initial_equity / n
    prev_price = dict(first_price)

    curve: List[Tuple[datetime, float]] = [(bars[0].timestamp, initial_equity)]
    for day in sorted(last_ts_per_day):
        ts = last_ts_per_day[day]
        value = 0.0
        for sym in symbols:
            price = daily_close.get((day, sym), prev_price[sym])
            value += alloc * price / max(first_price[sym], 1e-9)
            prev_price[sym] = price
        curve.append((ts, value))

    final_value = curve[-1][1]
    return_pct = (final_value / initial_equity - 1.0) * 100.0
    return curve, return_pct


def _compute_profit_factor(trades: List[Trade]) -> float:
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _compute_streaks(trades: List[Trade]) -> Tuple[int, int]:
    max_wins = max_losses = cur_wins = cur_losses = 0
    for t in trades:
        if t.net_pnl > 0:
            cur_wins += 1
            cur_losses = 0
        else:
            cur_losses += 1
            cur_wins = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def _compute_monthly_returns(
    snapshots: List[Tuple[datetime, float]], initial_equity: float
) -> Dict[str, float]:
    monthly: Dict[str, float] = {}
    for ts, eq in snapshots:
        monthly[ts.strftime("%Y-%m")] = eq
    prev_eq = initial_equity
    result: Dict[str, float] = {}
    for key in sorted(monthly):
        end_eq = monthly[key]
        result[key] = (end_eq - prev_eq) / max(prev_eq, 1e-9) * 100
        prev_eq = end_eq
    return result


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
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_hold_minutes: float
    trades: List[Trade]
    win_rate_pct: float
    avg_trade_pnl: float
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    benchmark_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    benchmark_return_pct: float = 0.0


class Backtester:
    def __init__(self, config: EngineConfig):
        self.config = config
        if config.strategy_type == "vwap":
            self.strategy = VWAPBreakoutStrategy(config)
        else:
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
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
        avg_trade = mean([t.net_pnl for t in trades]) if trades else 0.0
        total_return_pct = ((equity / initial_equity) - 1.0) * 100.0
        sharpe = _compute_sharpe(equity_snapshots, initial_equity)
        max_wins, max_losses = _compute_streaks(trades)
        hold_minutes = [
            (t.exit_time - t.entry_time).total_seconds() / 60 for t in trades
        ]
        benchmark_curve, benchmark_return_pct = compute_buy_and_hold(bars, initial_equity)

        return BacktestResult(
            initial_equity=initial_equity,
            final_equity=equity,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd * 100.0,
            sharpe_ratio=sharpe,
            profit_factor=_compute_profit_factor(trades),
            avg_win=mean([t.net_pnl for t in wins]) if wins else 0.0,
            avg_loss=mean([t.net_pnl for t in losses]) if losses else 0.0,
            best_trade=max((t.net_pnl for t in trades), default=0.0),
            worst_trade=min((t.net_pnl for t in trades), default=0.0),
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            avg_hold_minutes=mean(hold_minutes) if hold_minutes else 0.0,
            trades=trades,
            win_rate_pct=win_rate,
            avg_trade_pnl=avg_trade,
            equity_curve=equity_snapshots,
            monthly_returns=_compute_monthly_returns(equity_snapshots, initial_equity),
            benchmark_curve=benchmark_curve,
            benchmark_return_pct=benchmark_return_pct,
        )
