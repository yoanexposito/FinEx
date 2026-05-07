from dataclasses import dataclass
from datetime import time


@dataclass
class EngineConfig:
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    opening_range_minutes: int = 30
    breakout_buffer_bps: float = 5.0
    stop_loss_buffer_bps: float = 0.0
    target_rr: float = 1.5
    risk_per_trade_pct: float = 0.005
    max_position_value_pct: float = 0.2
    max_daily_loss_pct: float = 0.02
    slippage_bps: float = 1.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    allow_short: bool = True
    one_trade_per_day: bool = True
