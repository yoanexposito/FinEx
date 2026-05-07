from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    symbol: str
    side: Side
    timestamp: datetime
    trigger_price: float
    stop_price: float


@dataclass
class Position:
    symbol: str
    side: Side
    qty: int
    entry_time: datetime
    entry_price: float
    stop_price: float
    target_price: Optional[float]


@dataclass
class Trade:
    symbol: str
    side: Side
    qty: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    exit_reason: str
