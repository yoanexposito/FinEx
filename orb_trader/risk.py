from orb_trader.config import EngineConfig
from orb_trader.models import Side


class RiskManager:
    def __init__(self, config: EngineConfig):
        self.config = config

    def can_trade(self, day_start_equity: float, equity_now: float) -> bool:
        daily_drawdown_pct = (day_start_equity - equity_now) / max(day_start_equity, 1e-9)
        return daily_drawdown_pct < self.config.max_daily_loss_pct

    def position_size(
        self,
        side: Side,
        equity: float,
        entry_price: float,
        stop_price: float,
    ) -> int:
        del side  # Side is currently unused but kept for future asymmetric sizing.
        risk_budget = equity * self.config.risk_per_trade_pct
        per_share_risk = abs(entry_price - stop_price)
        if per_share_risk <= 0:
            return 0

        size_by_risk = int(risk_budget // per_share_risk)
        max_notional = equity * self.config.max_position_value_pct
        size_by_notional = int(max_notional // max(entry_price, 1e-9))
        qty = min(size_by_risk, size_by_notional)
        return max(0, qty)
