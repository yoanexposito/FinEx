"""
VWAP Breakout Strategy
======================
Uses the intraday Volume-Weighted Average Price (VWAP) as the key reference
level for price action entries.

Rationale
---------
VWAP is the average price weighted by volume since market open.  Institutional
desks use it as a benchmark, so a sustained break above/below VWAP signals a
genuine shift in control rather than random noise.

Entry logic
-----------
- LONG  : previous bar closed *below* VWAP; current bar closes *above* VWAP
          by at least `vwap_buffer_bps`.
- SHORT : previous bar closed *above* VWAP; current bar closes *below* VWAP
          by at least `vwap_buffer_bps`.
- Entries are blocked during the first `vwap_min_warmup_bars` bars of the day
  (price needs time to establish a meaningful VWAP).
- Entries are blocked after `vwap_entry_cutoff_hour` (default 14:00) to avoid
  low-liquidity afternoon chop.

Stop / target
-------------
- Stop  : placed `vwap_stop_bps` beyond the VWAP value at entry time.
          If price retreats through VWAP it has invalidated the breakout.
- Target: entry ± risk × target_rr  (same R:R framework as ORB).

VWAP formula
------------
  typical_price = (H + L + C) / 3          (per bar)
  VWAP = Σ(typical_price × volume) / Σ(volume)   (reset each calendar day)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, Set, Tuple

from orb_trader.config import EngineConfig
from orb_trader.models import Bar, Position, Side, Signal


@dataclass
class _VWAPDayState:
    """Per-symbol, per-day running state."""
    day: date
    cum_pv: float = 0.0     # cumulative (typical_price × volume)
    cum_v: float = 0.0      # cumulative volume
    vwap: float = 0.0
    prev_close: Optional[float] = None
    prev_vwap: Optional[float] = None
    bar_count: int = 0


class VWAPBreakoutStrategy:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._states: Dict[Tuple[str, date], _VWAPDayState] = {}
        self._day_traded_symbols: Dict[date, Set[str]] = defaultdict(set)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _key(self, bar: Bar) -> Tuple[str, date]:
        return (bar.symbol, bar.timestamp.date())

    def _market_close_dt(self, ts: datetime) -> datetime:
        return datetime.combine(ts.date(), self.config.market_close, tzinfo=ts.tzinfo)

    def _get_or_create_state(self, bar: Bar) -> _VWAPDayState:
        key = self._key(bar)
        if key not in self._states:
            self._states[key] = _VWAPDayState(day=bar.timestamp.date())
        return self._states[key]

    # ── Strategy interface (same method names as ORB for Backtester) ──────────

    def update_opening_range(self, bar: Bar) -> None:
        """Update the running VWAP with the new bar."""
        state = self._get_or_create_state(bar)
        typical = (bar.high + bar.low + bar.close) / 3.0
        state.cum_pv += typical * bar.volume
        state.cum_v  += bar.volume
        if state.cum_v > 0:
            state.vwap = state.cum_pv / state.cum_v
        state.bar_count += 1

    def maybe_generate_signal(self, bar: Bar, has_open_position: bool) -> Optional[Signal]:
        if has_open_position:
            return None

        day = bar.timestamp.date()
        if self.config.one_trade_per_day and bar.symbol in self._day_traded_symbols[day]:
            return None

        state = self._states.get(self._key(bar))
        if state is None or state.vwap == 0.0:
            return None

        # Warmup guard — need enough bars for a meaningful VWAP
        if state.bar_count < self.config.vwap_min_warmup_bars:
            state.prev_close = bar.close
            state.prev_vwap  = state.vwap
            return None

        # Time filter — no new entries in the final session hours
        if bar.timestamp.hour >= self.config.vwap_entry_cutoff_hour:
            state.prev_close = bar.close
            state.prev_vwap  = state.vwap
            return None

        prev_close = state.prev_close
        prev_vwap  = state.prev_vwap

        # Update state for next bar before potentially returning early
        state.prev_close = bar.close
        state.prev_vwap  = state.vwap

        if prev_close is None or prev_vwap is None:
            return None

        buf  = self.config.vwap_buffer_bps / 10_000.0
        stop_buf = self.config.vwap_stop_bps / 10_000.0
        vwap = state.vwap

        # LONG crossover: was below, now above VWAP by buffer
        long_cross = (prev_close < prev_vwap) and (bar.close > vwap * (1 + buf))
        # SHORT crossover: was above, now below VWAP by buffer
        short_cross = (prev_close > prev_vwap) and (bar.close < vwap * (1 - buf))

        if long_cross:
            stop_price = vwap * (1 - stop_buf)
            return Signal(
                symbol=bar.symbol,
                side=Side.LONG,
                timestamp=bar.timestamp,
                trigger_price=bar.close,
                stop_price=stop_price,
            )

        if self.config.allow_short and short_cross:
            stop_price = vwap * (1 + stop_buf)
            return Signal(
                symbol=bar.symbol,
                side=Side.SHORT,
                timestamp=bar.timestamp,
                trigger_price=bar.close,
                stop_price=stop_price,
            )

        return None

    def register_trade(self, symbol: str, ts: datetime) -> None:
        self._day_traded_symbols[ts.date()].add(symbol)

    def should_force_close(self, bar: Bar, position: Position) -> bool:
        del position
        return bar.timestamp >= self._market_close_dt(bar.timestamp)

    def exit_check(self, bar: Bar, position: Position) -> Optional[Tuple[float, str]]:
        """Shared stop/target logic — identical to ORB."""
        if position.side == Side.LONG:
            stop_hit   = bar.low  <= position.stop_price
            target_hit = position.target_price is not None and bar.high >= position.target_price
            if stop_hit and target_hit:
                return position.stop_price, "stop_hit_both_hit_conservative"
            if stop_hit:
                return position.stop_price, "stop_hit"
            if target_hit:
                return position.target_price, "target_hit"
            return None

        stop_hit   = bar.high >= position.stop_price
        target_hit = position.target_price is not None and bar.low <= position.target_price
        if stop_hit and target_hit:
            return position.stop_price, "stop_hit_both_hit_conservative"
        if stop_hit:
            return position.stop_price, "stop_hit"
        if target_hit:
            return position.target_price, "target_hit"
        return None

    def target_price(self, entry_price: float, stop_price: float, side: Side) -> Optional[float]:
        if self.config.target_rr <= 0:
            return None
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return None
        if side == Side.LONG:
            return entry_price + self.config.target_rr * risk
        return entry_price - self.config.target_rr * risk
