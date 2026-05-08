"""Tests for the Opening Range Breakout strategy."""
from datetime import datetime

import pytest

from orb_trader.config import EngineConfig
from orb_trader.models import Position, Side
from orb_trader.strategy import OpeningRangeBreakoutStrategy
from tests.conftest import drive, make_bar


def _cfg(**overrides) -> EngineConfig:
    defaults = dict(opening_range_minutes=30, breakout_buffer_bps=0.0,
                    stop_loss_buffer_bps=0.0, allow_short=True)
    defaults.update(overrides)
    return EngineConfig(**defaults)


def _pos(side: Side = Side.LONG, stop: float = 95.0, target: float = 110.0) -> Position:
    return Position(
        symbol="SPY", side=side, qty=10,
        entry_time=datetime(2026, 1, 5, 10, 0),
        entry_price=100.0, stop_price=stop, target_price=target,
    )


# ── Opening range construction ────────────────────────────────────────────────

class TestOpeningRange:
    def test_range_tracks_high_and_low(self):
        strat = OpeningRangeBreakoutStrategy(_cfg())
        strat.update_opening_range(make_bar(9, 30, 100, high=102, low=98))
        strat.update_opening_range(make_bar(9, 45, 101, high=104, low=97))
        from datetime import date
        state = strat._ranges[("SPY", date(2026, 1, 5))]
        assert state.high == 104
        assert state.low == 97

    def test_range_ignores_bars_after_window_closes(self):
        strat = OpeningRangeBreakoutStrategy(_cfg())
        strat.update_opening_range(make_bar(9, 30, 100, high=102, low=98))
        # Range ends at 10:00; this 10:05 bar should not expand the range
        strat.update_opening_range(make_bar(10, 5, 120, high=125, low=115))
        from datetime import date
        state = strat._ranges[("SPY", date(2026, 1, 5))]
        assert state.high == 102
        assert state.low == 98


# ── Signal generation ─────────────────────────────────────────────────────────

class TestSignalGeneration:
    def _setup_range(self, **cfg_kwargs):
        """Build a range high=103, low=97, then return the ready strategy."""
        strat = OpeningRangeBreakoutStrategy(_cfg(**cfg_kwargs))
        strat.update_opening_range(make_bar(9, 30, 100, high=102, low=98))
        strat.update_opening_range(make_bar(9, 55, 100, high=103, low=97))
        return strat

    def test_no_signal_before_range_closes(self):
        strat = self._setup_range()
        bar = make_bar(9, 45, 105)   # still inside the window
        strat.update_opening_range(bar)
        assert strat.maybe_generate_signal(bar, False) is None

    def test_long_signal_on_upward_breakout(self):
        strat = self._setup_range()
        bar = make_bar(10, 1, 104)   # above range high of 103
        strat.update_opening_range(bar)
        sig = strat.maybe_generate_signal(bar, False)
        assert sig is not None
        assert sig.side == Side.LONG
        assert sig.symbol == "SPY"

    def test_short_signal_on_downward_breakout(self):
        strat = self._setup_range(allow_short=True)
        bar = make_bar(10, 1, 96)    # below range low of 97
        strat.update_opening_range(bar)
        sig = strat.maybe_generate_signal(bar, False)
        assert sig is not None
        assert sig.side == Side.SHORT

    def test_no_short_when_shorts_disabled(self):
        strat = self._setup_range(allow_short=False)
        bar = make_bar(10, 1, 96)
        strat.update_opening_range(bar)
        assert strat.maybe_generate_signal(bar, False) is None

    def test_no_signal_inside_range(self):
        strat = self._setup_range()
        bar = make_bar(10, 1, 100)   # between high and low — no breakout
        strat.update_opening_range(bar)
        assert strat.maybe_generate_signal(bar, False) is None

    def test_no_signal_with_existing_position(self):
        strat = self._setup_range()
        bar = make_bar(10, 1, 104)
        strat.update_opening_range(bar)
        assert strat.maybe_generate_signal(bar, has_open_position=True) is None

    def test_one_trade_per_day_enforced(self):
        strat = self._setup_range()
        bar = make_bar(10, 1, 104)
        strat.update_opening_range(bar)
        sig = strat.maybe_generate_signal(bar, False)
        assert sig is not None
        strat.register_trade(sig.symbol, sig.timestamp)
        # Second bar same day — should be blocked
        bar2 = make_bar(10, 5, 106)
        strat.update_opening_range(bar2)
        assert strat.maybe_generate_signal(bar2, False) is None

    def test_breakout_buffer_requires_extra_distance(self):
        strat = self._setup_range(breakout_buffer_bps=100.0)   # 1% buffer
        # Range high=103; with 1% buffer, up_break = 103 * 1.01 = 104.03
        bar = make_bar(10, 1, 103.5)   # above 103 but below 104.03
        strat.update_opening_range(bar)
        assert strat.maybe_generate_signal(bar, False) is None


# ── Exit logic ────────────────────────────────────────────────────────────────

class TestExitCheck:
    def _strat(self):
        return OpeningRangeBreakoutStrategy(_cfg())

    def test_stop_hit_long(self):
        result = self._strat().exit_check(make_bar(10, 5, 94, low=93), _pos(Side.LONG, stop=95))
        assert result is not None
        price, reason = result
        assert reason == "stop_hit"
        assert price == pytest.approx(95.0)

    def test_target_hit_long(self):
        result = self._strat().exit_check(make_bar(10, 5, 111, high=112), _pos(Side.LONG, target=110))
        assert result is not None
        _, reason = result
        assert reason == "target_hit"

    def test_both_hit_conservative_long(self):
        # Same bar hits stop low and target high — conservative exit at stop
        result = self._strat().exit_check(
            make_bar(10, 5, 100, high=115, low=93),
            _pos(Side.LONG, stop=95, target=110),
        )
        assert result is not None
        _, reason = result
        assert reason == "stop_hit_both_hit_conservative"

    def test_no_exit_in_mid_range(self):
        result = self._strat().exit_check(
            make_bar(10, 5, 102, high=104, low=98),
            _pos(Side.LONG, stop=95, target=110),
        )
        assert result is None

    def test_stop_hit_short(self):
        result = self._strat().exit_check(
            make_bar(10, 5, 106, high=107),
            _pos(Side.SHORT, stop=105, target=90),
        )
        assert result is not None
        _, reason = result
        assert reason == "stop_hit"


# ── Force close ───────────────────────────────────────────────────────────────

class TestForceClose:
    def test_force_close_exactly_at_market_close(self):
        strat = OpeningRangeBreakoutStrategy(_cfg())
        bar = make_bar(16, 0, 100)
        assert strat.should_force_close(bar, _pos()) is True

    def test_no_force_close_one_minute_before(self):
        strat = OpeningRangeBreakoutStrategy(_cfg())
        bar = make_bar(15, 59, 100)
        assert strat.should_force_close(bar, _pos()) is False
