import argparse
import logging
from datetime import time
from pathlib import Path

from orb_trader.backtest import Backtester, load_bars_from_csv
from orb_trader.config import EngineConfig
from orb_trader.interfaces import PlaceholderBroker, PlaceholderMarketDataProvider
from orb_trader.live import LiveTrader


def _parse_hhmm(value: str) -> time:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time must be HH:MM")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        return time(hour=hour, minute=minute)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Invalid HH:MM value") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Opening Range Breakout trading engine")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--market-open", type=_parse_hhmm, default=time(9, 30))
    common.add_argument("--market-close", type=_parse_hhmm, default=time(16, 0))
    common.add_argument("--opening-range-minutes", type=int, default=30)
    common.add_argument("--breakout-buffer-bps", type=float, default=5.0)
    common.add_argument("--stop-loss-buffer-bps", type=float, default=0.0)
    common.add_argument("--target-rr", type=float, default=1.5)
    common.add_argument("--risk-per-trade-pct", type=float, default=0.005)
    common.add_argument("--max-position-value-pct", type=float, default=0.2)
    common.add_argument("--max-daily-loss-pct", type=float, default=0.02)
    common.add_argument("--slippage-bps", type=float, default=1.0)
    common.add_argument("--commission-per-share", type=float, default=0.005)
    common.add_argument("--min-commission", type=float, default=1.0)
    common.add_argument("--allow-short", action="store_true", default=False)

    bt = subparsers.add_parser("backtest", parents=[common], help="Run backtest from CSV")
    bt.add_argument("--csv", type=Path, required=True)
    bt.add_argument("--initial-equity", type=float, default=100_000.0)

    live = subparsers.add_parser("live", parents=[common], help="Run live loop")
    live.add_argument("--symbols", type=str, default="SPY")
    live.add_argument("--poll-seconds", type=float, default=5.0)

    return parser


def build_config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        market_open=args.market_open,
        market_close=args.market_close,
        opening_range_minutes=args.opening_range_minutes,
        breakout_buffer_bps=args.breakout_buffer_bps,
        stop_loss_buffer_bps=args.stop_loss_buffer_bps,
        target_rr=args.target_rr,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_position_value_pct=args.max_position_value_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
        slippage_bps=args.slippage_bps,
        commission_per_share=args.commission_per_share,
        min_commission=args.min_commission,
        allow_short=args.allow_short,
    )


def run_backtest(args: argparse.Namespace) -> None:
    config = build_config(args)
    bars = load_bars_from_csv(args.csv)
    result = Backtester(config).run(bars, initial_equity=args.initial_equity)

    print("=== Backtest Results ===")
    print(f"Initial equity: ${result.initial_equity:,.2f}")
    print(f"Final equity:   ${result.final_equity:,.2f}")
    print(f"Total return:   {result.total_return_pct:.2f}%")
    print(f"Max drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Trades:         {len(result.trades)}")
    print(f"Win rate:       {result.win_rate_pct:.2f}%")
    print(f"Avg trade PnL:  ${result.avg_trade_pnl:,.2f}")


def run_live(args: argparse.Namespace) -> None:
    config = build_config(args)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")

    market_data = PlaceholderMarketDataProvider()
    broker = PlaceholderBroker()
    trader = LiveTrader(
        symbols=symbols,
        config=config,
        market_data=market_data,
        broker=broker,
        poll_seconds=args.poll_seconds,
    )
    trader.run_forever()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args)
        return
    if args.mode == "live":
        run_live(args)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
