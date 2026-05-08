"""Tests for the VWAP Breakout strategy."""
from datetime import datetime

import pytest

from orb_trader.config import EngineConfig
from orb_trader.models import Position, Side
from orb_trader.strategy_vwap import VWAPBreakoutStrategy
from tests.conftest import drive, make_bar


def _cfg(**overrides) -> EngineConfig:
    defaults = dict(
        strategy_type="vwap",
        vwap_min_warmup_bars=2,
        vwap_buffer_bps=0.0,
        vwap_stop_bps=20.0,
        vwap_entry_cutoff_hour=14,
        allow_short=True,
        one_trade_per_day=False,   # make signal testing easier
    )
    defaults.update(overrides)
    return EngineConfig(**defaults)


def _pos(side: Side = Side.LONG, stop: float = 95.0, target: float = 110.0) -> Position:
    return Position(
        symbol="SPY", side=side, qty=10,
        entry_time=datetime(2026, 1, 5, 10, 0),
        entry_price=100.0, stop_price=stop, target_price=target,
    )


# ── VWAP calculation ──────────────────────────────────────────────────────────

class TestVWAPCalculation:
    def test_vwap_equals_typical_price_for_first_bar(self):
        strat = VWAPBreakoutStrategy(_cfg())
        bar = make_bar(9, 30, close=100, high=102, low=98, volume=1_000)
        strat.update_opening_range(bar)
        from datetime import date
        state = strat._states[("SPY", date(2026, 1, 5))]
        # typical = (102 + 98 + 100) / 3 = 100
        assert state.vwap == pytest.approx(100.0)

    def test_vwap_weighted_by_volume(self):
        strat = VWAPBreakoutStrategy(_cfg())
        # Bar 1: typical=100, volume=1000 → pv=100_000
        # Bar 2: typical=110, volume=1000 → pv=110_000 → VWAP = 210_000/2000 = 105
        strat.update_opening_range(make_bar(9, 30, close=100, high=100, low=100, volume=1_000))
        strat.update_opening_range(make_bar(9, 35, close=110, high=110, low=110, volume=1_000))
        from datetime import date
        state = strat._states[("SPY", date(2026, 1, 5))]
        assert state.vwap == pytest.approx(105.0)

    def test_vwap_resets_each_day(self):
        strat = VWAPBreakoutStrategy(_cfg())
        strat.update_opening_range(make_bar(9, 30, close=200, day=5))
        strat.update_opening_range(make_bar(9, 30, close=100, day=6))
        from datetime import date
        assert strat._states[("SPY", date(2026, 1, 6))].vwap == pytest.approx(100.0, rel=0.01)


# ── Signal generation ─────────────────────────────────────────────────────────

class TestSignalGeneration:
    def _crossover_bars(self):
        """
        Bar 1 (warmup): close=90, high=100, low=90 → typical≈93.3, VWAP=93.3
                         State after signal check: prev_close=90, prev_vwap=93.3
        Bar 2 (warmup ends): close=90, high=91, low=89 → VWAP≈91.7
                         prev_close(90) < prev_vwap(93.3) ✓ but close(90) < VWAP(91.7) → no long
        Bar 3: close=95, high=96, low=94 → VWAP≈92.8
                         prev_close(90) < prev_vwap(91.7) ✓  AND close(95) > VWAP(92.8) → LONG
        """
        return [
            make_bar(9, 30, close=90, high=100, low=90),
            make_bar(9, 35, close=90, high=91,  low=89),
            make_bar(9, 40, close=95, high=96,  low=94),
        ]

    def test_long_signal_on_vwap_crossover(self):
        strat = VWAPBreakoutStrategy(_cfg())
        sig = drive(strat, self._crossover_bars())
        assert sig is not None
        assert sig.side == Side.LONG

    def test_no_signal_during_warmup(self):
        strat = VWAPBreakoutStrategy(_cfg(vwap_min_warmup_bars=5))
        # Only feed 3 bars — never clears the warmup gate
        sig = drive(strat, self._crossover_bars())
        assert sig is None

    def test_no_signal_with_open_position(self):
        strat = VWAPBreakoutStrategy(_cfg())
        sig = drive(strat, self._crossover_bars(), has_open_position=True)
        assert sig is None

    def test_no_signal_after_entry_cutoff(self):
        strat = VWAPBreakoutStrategy(_cfg(vwap_entry_cutoff_hour=14))
        # Replay the same crossover scenario but in the afternoon
        bars = [
            make_bar(14, 0, close=90, high=100, low=90),
            make_bar(14, 5, close=90, high=91,  low=89),
            make_bar(14, 10, close=95, high=96, low=94),
        ]
        assert drive(strat, bars) is None

    def test_short_signal_on_vwap_crossdown(self):
        """
        Start above VWAP, then drop below it.
        Bar 1 (warmup): close=100, high=100, low=90 → typical=96.7, VWAP=96.7
                         prev_close=100, prev_vwap=96.7
        Bar 2: close=100, high=101, low=99 → VWAP≈98.2
                         prev_close(100) > prev_vwap(96.7) — above VWAP
        Bar 3: close=95, high=96, low=94 → VWAP≈97.0
                         prev_close(100) > prev_vwap(98.2) ✓  AND close(95) < VWAP(97.0) → SHORT
        """
        strat = VWAPBreakoutStrategy(_cfg(allow_short=True))
        bars = [
            make_bar(9, 30, close=100, high=100, low=90),
            make_bar(9, 35, close=100, high=101, low=99),
            make_bar(9, 40, close=95,  high=96,  low=94),
        ]
        sig = drive(strat, bars)
        assert sig is not None
        assert sig.side == Side.SHORT

    def test_no_short_when_disabled(self):
        strat = VWAPBreakoutStrategy(_cfg(allow_short=False))
        bars = [
            make_bar(9, 30, close=100, high=100, low=90),
            make_bar(9, 35, close=100, high=101, low=99),
            make_bar(9, 40, close=95,  high=96,  low=94),
        ]
        sig = drive(strat, bars)
        assert sig is None or sig.side == Side.LONG

    def test_buffer_prevents_marginal_entries(self):
        """With a large buffer, a tiny crossover should not trigger."""
        strat = VWAPBreakoutStrategy(_cfg(vwap_buffer_bps=200.0))   # 2% buffer
        # Bar 3 close=95 is only ~2.3% above VWAP≈92.8 — borderline
        # With a 2% buffer requirement this should barely fire or not at all
        sig = drive(strat, [
            make_bar(9, 30, close=90, high=100, low=90),
            make_bar(9, 35, close=90, high=91,  low=89),
            make_bar(9, 40, close=92, high=93,  low=91),   # tiny cross — below buffer
        ])
        assert sig is None


# ── Exit logic (shared with ORB) ─────────────────────────────────────────────

class TestExitCheck:
    def _strat(self):
        return VWAPBreakoutStrategy(_cfg())

    def test_stop_hit_long(self):
        result = self._strat().exit_check(make_bar(10, 5, 94, low=93), _pos(Side.LONG, stop=95))
        assert result is not None
        _, reason = result
        assert reason == "stop_hit"

    def test_target_hit_long(self):
        result = self._strat().exit_check(make_bar(10, 5, 111, high=112), _pos(Side.LONG, target=110))
        assert result is not None
        _, reason = result
        assert reason == "target_hit"

    def test_no_exit_mid_range(self):
        result = self._strat().exit_check(
            make_bar(10, 5, 102, high=104, low=98),
            _pos(Side.LONG, stop=95, target=110),
        )
        assert result is None


# ── Force close ───────────────────────────────────────────────────────────────

class TestForceClose:
    def test_force_close_at_market_close(self):
        strat = VWAPBreakoutStrategy(_cfg())
        assert strat.should_force_close(make_bar(16, 0, 100), _pos()) is True

    def test_no_force_close_before_market_close(self):
        strat = VWAPBreakoutStrategy(_cfg())
        assert strat.should_force_close(make_bar(15, 59, 100), _pos()) is False
