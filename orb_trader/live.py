import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

from orb_trader.config import EngineConfig
from orb_trader.interfaces import Broker, MarketDataProvider
from orb_trader.models import Bar, Position, Side
from orb_trader.risk import RiskManager
from orb_trader.strategy import OpeningRangeBreakoutStrategy

LOGGER = logging.getLogger(__name__)


@dataclass
class LiveState:
    day_start_equity: float
    positions: Dict[str, Position]


class LiveTrader:
    def __init__(
        self,
        symbols: List[str],
        config: EngineConfig,
        market_data: MarketDataProvider,
        broker: Broker,
        poll_seconds: float = 5.0,
    ):
        self.symbols = symbols
        self.config = config
        self.market_data = market_data
        self.broker = broker
        self.poll_seconds = poll_seconds
        self.strategy = OpeningRangeBreakoutStrategy(config)
        self.risk = RiskManager(config)

    def _entry_price_estimate(self, side: Side, bar: Bar) -> float:
        slip = self.config.slippage_bps / 10_000.0
        if side == Side.LONG:
            return bar.close * (1 + slip)
        return bar.close * (1 - slip)

    def _process_bar(self, bar: Bar, state: LiveState) -> None:
        self.strategy.update_opening_range(bar)
        position = state.positions.get(bar.symbol)

        if position:
            exit_hit = self.strategy.exit_check(bar, position)
            should_close = exit_hit is not None or self.strategy.should_force_close(bar, position)
            if should_close:
                order_id = self.broker.close_position(position)
                reason = exit_hit[1] if exit_hit else "end_of_day"
                LOGGER.info(
                    "Closed %s %s qty=%s order_id=%s reason=%s",
                    position.side.value,
                    position.symbol,
                    position.qty,
                    order_id,
                    reason,
                )
                del state.positions[bar.symbol]
            return

        equity_now = self.broker.get_equity()
        if not self.risk.can_trade(state.day_start_equity, equity_now):
            return

        signal = self.strategy.maybe_generate_signal(bar, has_open_position=False)
        if not signal:
            return

        est_entry = self._entry_price_estimate(signal.side, bar)
        qty = self.risk.position_size(signal.side, equity_now, est_entry, signal.stop_price)
        if qty <= 0:
            return

        order_id = self.broker.place_market_order(signal.symbol, signal.side, qty)
        target = self.strategy.target_price(est_entry, signal.stop_price, signal.side)
        state.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            side=signal.side,
            qty=qty,
            entry_time=signal.timestamp,
            entry_price=est_entry,
            stop_price=signal.stop_price,
            target_price=target,
        )
        self.strategy.register_trade(signal.symbol, signal.timestamp)

        LOGGER.info(
            "Entered %s %s qty=%s order_id=%s est_entry=%.4f stop=%.4f target=%s",
            signal.side.value,
            signal.symbol,
            qty,
            order_id,
            est_entry,
            signal.stop_price,
            f"{target:.4f}" if target is not None else "None",
        )

    def run_forever(self) -> None:
        starting_equity = self.broker.get_equity()
        state = LiveState(day_start_equity=starting_equity, positions={})
        LOGGER.info("Live trader started with equity=%.2f", starting_equity)

        while True:
            bars: Iterable[Bar] = self.market_data.get_latest_bars(self.symbols)
            for bar in bars:
                self._process_bar(bar, state)
            time.sleep(self.poll_seconds)
