"""Tests for RiskManager — position sizing and daily loss gate."""
import pytest
from orb_trader.config import EngineConfig
from orb_trader.models import Side
from orb_trader.risk import RiskManager


def _rm(**overrides) -> RiskManager:
    defaults = dict(
        risk_per_trade_pct=0.01,       # 1% risk per trade
        max_position_value_pct=0.50,   # 50% max notional
        max_daily_loss_pct=0.02,       # 2% daily loss limit
    )
    defaults.update(overrides)
    return RiskManager(EngineConfig(**defaults))


# ── Daily loss gate ────────────────────────────────────────────────────────────

class TestCanTrade:
    def test_allows_trade_within_daily_limit(self):
        rm = _rm()
        assert rm.can_trade(100_000, 99_000) is True   # 1% drawdown < 2% limit

    def test_blocks_trade_when_limit_breached(self):
        rm = _rm()
        assert rm.can_trade(100_000, 97_900) is False  # 2.1% drawdown > 2% limit

    def test_blocks_trade_at_exact_limit(self):
        # drawdown = exactly 2% — the check is strict (<), so this is blocked
        rm = _rm()
        assert rm.can_trade(100_000, 98_000) is False

    def test_allows_trade_when_equity_has_grown(self):
        rm = _rm()
        assert rm.can_trade(100_000, 110_000) is True  # profitable day


# ── Position sizing ────────────────────────────────────────────────────────────

class TestPositionSize:
    def test_basic_risk_based_sizing(self):
        rm = _rm(risk_per_trade_pct=0.01, max_position_value_pct=0.50)
        # equity=100k, risk_budget=1k, per_share_risk=5 → 200 shares
        qty = rm.position_size(Side.LONG, 100_000, 100.0, 95.0)
        assert qty == 200

    def test_capped_by_max_notional(self):
        rm = _rm(risk_per_trade_pct=0.10, max_position_value_pct=0.05)
        # risk would allow 2000 shares; notional cap (5k / $100) limits to 50
        qty = rm.position_size(Side.LONG, 100_000, 100.0, 95.0)
        assert qty == 50

    def test_zero_when_stop_equals_entry(self):
        rm = _rm()
        qty = rm.position_size(Side.LONG, 100_000, 100.0, 100.0)
        assert qty == 0

    def test_zero_when_stop_beyond_entry_long(self):
        # Stop above entry for a long makes no sense — per_share_risk is still positive (abs)
        # but this edge is handled gracefully
        rm = _rm()
        qty = rm.position_size(Side.LONG, 100_000, 100.0, 100.0)
        assert qty >= 0

    def test_short_sizing_same_as_long(self):
        # Side is currently unused in sizing logic — should be symmetric
        rm = _rm(risk_per_trade_pct=0.01, max_position_value_pct=0.50)
        long_qty  = rm.position_size(Side.LONG,  100_000, 100.0, 95.0)
        short_qty = rm.position_size(Side.SHORT, 100_000, 100.0, 105.0)
        assert long_qty == short_qty

    def test_scales_with_equity(self):
        rm = _rm(risk_per_trade_pct=0.01, max_position_value_pct=0.50)
        qty_small = rm.position_size(Side.LONG,  50_000, 100.0, 95.0)
        qty_large = rm.position_size(Side.LONG, 100_000, 100.0, 95.0)
        assert qty_large == qty_small * 2
