"""Shared test helpers and fixtures."""
from datetime import datetime

from orb_trader.models import Bar


def make_bar(
    hour: int,
    minute: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100_000,
    symbol: str = "SPY",
    day: int = 5,
) -> Bar:
    """Create a Bar for Jan <day> 2026 at the given time."""
    return Bar(
        symbol=symbol,
        timestamp=datetime(2026, 1, day, hour, minute, 0),
        open=close,
        high=high if high is not None else close * 1.001,
        low=low if low is not None else close * 0.999,
        close=close,
        volume=volume,
    )


def drive(strategy, bars, *, has_open_position: bool = False):
    """Drive a strategy through a list of bars and return the last signal emitted."""
    last_signal = None
    for bar in bars:
        strategy.update_opening_range(bar)
        sig = strategy.maybe_generate_signal(bar, has_open_position)
        if sig:
            last_signal = sig
    return last_signal
