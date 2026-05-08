"""Tests for the core backtest primitives and the Backtester integration."""
from datetime import datetime

import pytest

from orb_trader.backtest import (
    Backtester,
    _apply_slippage,
    _commission,
    _compute_profit_factor,
    _compute_streaks,
    _signed_pnl,
    compute_buy_and_hold,
)
from orb_trader.config import EngineConfig
from orb_trader.models import Bar, Side, Trade


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trade(net_pnl: float, symbol: str = "SPY") -> Trade:
    return Trade(
        symbol=symbol, side=Side.LONG, qty=1,
        entry_time=datetime(2026, 1, 5, 9, 30),
        exit_time=datetime(2026, 1, 5, 10, 0),
        entry_price=100.0, exit_price=100.0 + net_pnl,
        gross_pnl=net_pnl, net_pnl=net_pnl, exit_reason="test",
    )


def _bar(ts: datetime, close: float, volume: float = 1_000, symbol: str = "SPY") -> Bar:
    return Bar(symbol=symbol, timestamp=ts, open=close,
               high=close * 1.001, low=close * 0.999, close=close, volume=volume)


# ── Slippage ───────────────────────────────────────────────────────────────────

class TestSlippage:
    def test_long_entry_increases_price(self):
        result = _apply_slippage(100.0, Side.LONG, 10.0, is_entry=True)
        assert result > 100.0

    def test_long_exit_decreases_price(self):
        result = _apply_slippage(100.0, Side.LONG, 10.0, is_entry=False)
        assert result < 100.0

    def test_short_entry_decreases_price(self):
        result = _apply_slippage(100.0, Side.SHORT, 10.0, is_entry=True)
        assert result < 100.0

    def test_short_exit_increases_price(self):
        result = _apply_slippage(100.0, Side.SHORT, 10.0, is_entry=False)
        assert result > 100.0

    def test_zero_slippage_is_identity(self):
        assert _apply_slippage(100.0, Side.LONG, 0.0, is_entry=True) == 100.0


# ── Commission ────────────────────────────────────────────────────────────────

class TestCommission:
    def test_minimum_applied_for_small_orders(self):
        assert _commission(qty=1, per_share=0.005, minimum=1.0) == 1.0

    def test_per_share_applied_for_large_orders(self):
        assert _commission(qty=300, per_share=0.005, minimum=1.0) == pytest.approx(1.5)

    def test_boundary_exactly_at_minimum(self):
        # 200 shares * $0.005 = $1.00 = minimum
        assert _commission(qty=200, per_share=0.005, minimum=1.0) == pytest.approx(1.0)


# ── Signed P&L ────────────────────────────────────────────────────────────────

class TestSignedPnl:
    def test_long_profit(self):
        assert _signed_pnl(Side.LONG, 100.0, 110.0, 10) == pytest.approx(100.0)

    def test_long_loss(self):
        assert _signed_pnl(Side.LONG, 100.0, 90.0, 10) == pytest.approx(-100.0)

    def test_short_profit(self):
        assert _signed_pnl(Side.SHORT, 100.0, 90.0, 10) == pytest.approx(100.0)

    def test_short_loss(self):
        assert _signed_pnl(Side.SHORT, 100.0, 110.0, 10) == pytest.approx(-100.0)


# ── Profit factor ─────────────────────────────────────────────────────────────

class TestProfitFactor:
    def test_basic_calculation(self):
        trades = [_trade(100), _trade(100), _trade(-50)]
        # gross profit = 200, gross loss = 50 → PF = 4.0
        assert _compute_profit_factor(trades) == pytest.approx(4.0)

    def test_infinity_when_no_losses(self):
        trades = [_trade(100), _trade(50)]
        assert _compute_profit_factor(trades) == float("inf")

    def test_zero_when_no_wins(self):
        trades = [_trade(-100), _trade(-50)]
        assert _compute_profit_factor(trades) == pytest.approx(0.0)


# ── Streaks ───────────────────────────────────────────────────────────────────

class TestStreaks:
    def test_basic_streaks(self):
        #  W W L L L W  →  max_wins=2, max_losses=3
        trades = [_trade(10), _trade(10), _trade(-5), _trade(-5), _trade(-5), _trade(10)]
        max_wins, max_losses = _compute_streaks(trades)
        assert max_wins == 2
        assert max_losses == 3

    def test_all_wins(self):
        trades = [_trade(10)] * 5
        max_wins, max_losses = _compute_streaks(trades)
        assert max_wins == 5
        assert max_losses == 0

    def test_single_trade_win(self):
        max_wins, max_losses = _compute_streaks([_trade(10)])
        assert max_wins == 1
        assert max_losses == 0


# ── Buy-and-hold benchmark ────────────────────────────────────────────────────

class TestBuyAndHold:
    def test_single_symbol_ten_percent_gain(self):
        bars = [
            _bar(datetime(2026, 1, 5, 9, 30), close=100.0),
            _bar(datetime(2026, 1, 5, 16, 0), close=110.0),  # +10%
        ]
        curve, ret_pct = compute_buy_and_hold(bars, 10_000)
        assert abs(ret_pct - 10.0) < 0.01

    def test_empty_bars_returns_zero(self):
        curve, ret_pct = compute_buy_and_hold([], 10_000)
        assert curve == []
        assert ret_pct == 0.0

    def test_flat_market_zero_return(self):
        bars = [
            _bar(datetime(2026, 1, 5, 9, 30), close=100.0),
            _bar(datetime(2026, 1, 5, 16, 0), close=100.0),
        ]
        _, ret_pct = compute_buy_and_hold(bars, 10_000)
        assert abs(ret_pct) < 0.01

    def test_curve_starts_at_initial_equity(self):
        bars = [_bar(datetime(2026, 1, 5, 9, 30), close=100.0)]
        curve, _ = compute_buy_and_hold(bars, 50_000)
        assert curve[0][1] == pytest.approx(50_000)


# ── Backtester integration smoke test ─────────────────────────────────────────

class TestBacktesterIntegration:
    def _two_day_bars(self):
        """Minimal two-day bar set: opening range then breakout each day."""
        bars = []
        for day in (5, 6):
            # Opening range bars (09:30–09:55)
            for minute in (30, 35, 40, 45, 50, 55):
                bars.append(_bar(datetime(2026, 1, day, 9, minute), close=100.0))
            # Post-range bars — breakout up
            bars.append(_bar(datetime(2026, 1, day, 10, 1), close=101.5,
                             symbol="SPY"))
            # Bars until EOD
            for minute in (30,):
                bars.append(_bar(datetime(2026, 1, day, 15, minute), close=101.0))
            bars.append(_bar(datetime(2026, 1, day, 16, 0), close=101.0))
        bars.sort(key=lambda b: b.timestamp)
        return bars

    def test_result_fields_populated(self):
        result = Backtester(EngineConfig()).run(self._two_day_bars(), 100_000)
        assert result.initial_equity == 100_000
        assert isinstance(result.final_equity, float)
        assert 0.0 <= result.win_rate_pct <= 100.0
        assert result.max_drawdown_pct >= 0.0

    def test_benchmark_curve_populated(self):
        result = Backtester(EngineConfig()).run(self._two_day_bars(), 100_000)
        assert len(result.benchmark_curve) > 0

    def test_no_trades_on_empty_bars(self):
        result = Backtester(EngineConfig()).run([], 100_000)
        assert result.trades == []
        assert result.final_equity == 100_000
