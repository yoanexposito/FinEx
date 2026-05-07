from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Set, Tuple

from orb_trader.config import EngineConfig
from orb_trader.models import Bar, Position, Side, Signal


@dataclass
class OpeningRangeState:
    day: date
    range_end: datetime
    high: float
    low: float
    finalized: bool = False


class OpeningRangeBreakoutStrategy:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._ranges: Dict[Tuple[str, date], OpeningRangeState] = {}
        self._day_traded_symbols: Dict[date, Set[str]] = defaultdict(set)

    def _get_day_key(self, bar: Bar) -> Tuple[str, date]:
        return (bar.symbol, bar.timestamp.date())

    def _market_datetimes(self, ts: datetime) -> Tuple[datetime, datetime]:
        day = ts.date()
        tzinfo = ts.tzinfo
        market_open = datetime.combine(day, self.config.market_open, tzinfo=tzinfo)
        market_close = datetime.combine(day, self.config.market_close, tzinfo=tzinfo)
        return market_open, market_close

    def update_opening_range(self, bar: Bar) -> None:
        key = self._get_day_key(bar)
        state = self._ranges.get(key)
        if state is None:
            market_open, _ = self._market_datetimes(bar.timestamp)
            range_end = market_open + timedelta(minutes=self.config.opening_range_minutes)
            self._ranges[key] = OpeningRangeState(
                day=bar.timestamp.date(),
                range_end=range_end,
                high=bar.high,
                low=bar.low,
            )
            return

        if bar.timestamp <= state.range_end and not state.finalized:
            state.high = max(state.high, bar.high)
            state.low = min(state.low, bar.low)
        elif bar.timestamp > state.range_end:
            state.finalized = True

    def _breakout_prices(self, state: OpeningRangeState) -> Tuple[float, float]:
        buffer = self.config.breakout_buffer_bps / 10_000.0
        up_break = state.high * (1 + buffer)
        down_break = state.low * (1 - buffer)
        return up_break, down_break

    def maybe_generate_signal(self, bar: Bar, has_open_position: bool) -> Optional[Signal]:
        day = bar.timestamp.date()
        if has_open_position:
            return None
        if self.config.one_trade_per_day and bar.symbol in self._day_traded_symbols[day]:
            return None

        key = self._get_day_key(bar)
        state = self._ranges.get(key)
        if not state:
            return None
        if bar.timestamp <= state.range_end:
            return None

        state.finalized = True
        up_break, down_break = self._breakout_prices(state)
        stop_buffer = self.config.stop_loss_buffer_bps / 10_000.0

        if bar.close >= up_break:
            stop_price = state.low * (1 - stop_buffer)
            return Signal(
                symbol=bar.symbol,
                side=Side.LONG,
                timestamp=bar.timestamp,
                trigger_price=bar.close,
                stop_price=stop_price,
            )

        if self.config.allow_short and bar.close <= down_break:
            stop_price = state.high * (1 + stop_buffer)
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
        _, market_close = self._market_datetimes(bar.timestamp)
        return bar.timestamp >= market_close

    def exit_check(self, bar: Bar, position: Position) -> Optional[Tuple[float, str]]:
        if position.side == Side.LONG:
            stop_hit = bar.low <= position.stop_price
            target_hit = position.target_price is not None and bar.high >= position.target_price
            if stop_hit and target_hit:
                return position.stop_price, "stop_hit_both_hit_conservative"
            if stop_hit:
                return position.stop_price, "stop_hit"
            if target_hit:
                return position.target_price, "target_hit"
            return None

        stop_hit = bar.high >= position.stop_price
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
