from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List

from orb_trader.models import Bar, Position, Side


class MarketDataProvider(ABC):
    @abstractmethod
    def get_latest_bars(self, symbols: List[str]) -> Iterable[Bar]:
        """Return latest bars for each symbol."""


class Broker(ABC):
    @abstractmethod
    def place_market_order(self, symbol: str, side: Side, qty: int) -> str:
        """Place market order and return order id."""

    @abstractmethod
    def close_position(self, position: Position) -> str:
        """Close an existing position and return order id."""

    @abstractmethod
    def get_equity(self) -> float:
        """Return current account equity."""


class PlaceholderMarketDataProvider(MarketDataProvider):
    def get_latest_bars(self, symbols: List[str]) -> Iterable[Bar]:
        raise NotImplementedError(
            "Implement a real MarketDataProvider using your market data API."
        )


class PlaceholderBroker(Broker):
    def place_market_order(self, symbol: str, side: Side, qty: int) -> str:
        raise NotImplementedError("Implement broker order routing with your broker API.")

    def close_position(self, position: Position) -> str:
        raise NotImplementedError("Implement position close with your broker API.")

    def get_equity(self) -> float:
        raise NotImplementedError("Implement account equity lookup with your broker API.")


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)
