import logging
from datetime import datetime
from typing import Iterable, List

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from orb_trader.interfaces import Broker, MarketDataProvider
from orb_trader.models import Bar, Position, Side

LOGGER = logging.getLogger(__name__)


def fetch_historical_bars(
    api_key: str,
    secret_key: str,
    symbols: List[str],
    start: datetime,
    end: datetime,
    timeframe: TimeFrame,
) -> List[Bar]:
    """Fetch split/dividend-adjusted historical bars from Alpaca."""
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment="all",
    )
    response = client.get_stock_bars(request)
    bars: List[Bar] = []
    for symbol, bar_list in response.data.items():
        for bar in bar_list:
            bars.append(Bar(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            ))
    bars.sort(key=lambda b: (b.timestamp, b.symbol))
    LOGGER.info("Fetched %d historical bars for %s", len(bars), symbols)
    return bars


class AlpacaMarketDataProvider(MarketDataProvider):
    def __init__(self, api_key: str, secret_key: str) -> None:
        self._client = StockHistoricalDataClient(api_key, secret_key)

    def get_latest_bars(self, symbols: List[str]) -> Iterable[Bar]:
        request = StockLatestBarRequest(symbol_or_symbols=symbols)
        response = self._client.get_stock_latest_bar(request)
        bars: List[Bar] = []
        for symbol, bar in response.items():
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                )
            )
        LOGGER.debug("Fetched %d bars from Alpaca", len(bars))
        return bars


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        self._client = TradingClient(api_key, secret_key, paper=paper)
        mode = "paper" if paper else "live"
        LOGGER.info("AlpacaBroker initialised in %s mode", mode)

    def place_market_order(self, symbol: str, side: Side, qty: int) -> str:
        order_side = OrderSide.BUY if side == Side.LONG else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(request)
        LOGGER.info("Submitted %s order for %s qty=%d id=%s", order_side.value, symbol, qty, order.id)
        return str(order.id)

    def close_position(self, position: Position) -> str:
        order = self._client.close_position(position.symbol)
        LOGGER.info("Closed position %s id=%s", position.symbol, order.id)
        return str(order.id)

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.equity)
