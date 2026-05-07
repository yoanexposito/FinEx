import sys
import threading
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

import os
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

from orb_trader.adapters.alpaca import AlpacaBroker, AlpacaMarketDataProvider, fetch_historical_bars
from orb_trader.backtest import BacktestResult, Backtester, load_bars_from_csv
from orb_trader.config import EngineConfig
from orb_trader.live import LiveTrader
from orb_trader.models import Side

_SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample.csv"
_GREEN = "#00C805"
_RED = "#FF3B30"


# ── Chart builders ────────────────────────────────────────────────────────────

def _equity_chart(result: BacktestResult) -> go.Figure:
    dates = [p[0] for p in result.equity_curve]
    equity = [p[1] for p in result.equity_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        mode="lines",
        name="Portfolio",
        line=dict(color=_GREEN, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,200,5,0.07)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title=None,
        yaxis=dict(tickformat="$,.0f"),
        height=360,
        margin=dict(l=0, r=0, t=36, b=0),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _drawdown_chart(result: BacktestResult) -> go.Figure:
    equity = [p[1] for p in result.equity_curve]
    dates = [p[0] for p in result.equity_curve]
    peak = equity[0]
    drawdowns = []
    for eq in equity:
        peak = max(peak, eq)
        drawdowns.append((eq - peak) / peak * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdowns,
        mode="lines",
        name="Drawdown",
        line=dict(color=_RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,59,48,0.12)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.2f}%</b><extra></extra>",
    ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title=None,
        yaxis=dict(tickformat=".1f", ticksuffix="%"),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _pnl_bar_chart(result: BacktestResult) -> go.Figure:
    pnls = [t.net_pnl for t in result.trades]
    colors = [_GREEN if p > 0 else _RED for p in pnls]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(pnls) + 1)),
        y=pnls,
        marker_color=colors,
        hovertemplate="Trade %{x}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="gray", line_width=0.8)
    fig.update_layout(
        title="Trade P&L",
        xaxis_title="Trade #",
        yaxis=dict(tickformat="$,.0f"),
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Alpaca helpers ────────────────────────────────────────────────────────────

def _trading_client(api_key: str, secret_key: str, paper: bool) -> TradingClient:
    return TradingClient(api_key, secret_key, paper=paper)


def _fetch_account(api_key: str, secret_key: str, paper: bool) -> dict:
    acct = _trading_client(api_key, secret_key, paper).get_account()
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "portfolio_value": float(acct.portfolio_value),
    }


def _fetch_positions(api_key: str, secret_key: str, paper: bool) -> pd.DataFrame:
    positions = _trading_client(api_key, secret_key, paper).get_all_positions()
    if not positions:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Symbol": p.symbol,
        "Side": p.side.value.upper(),
        "Qty": int(float(p.qty)),
        "Avg Entry": float(p.avg_entry_price),
        "Current Price": float(p.current_price),
        "Market Value": float(p.market_value),
        "Unrealized P&L": float(p.unrealized_pl),
        "Unrealized P&L %": float(p.unrealized_plpc) * 100,
    } for p in positions])


def _fetch_orders(api_key: str, secret_key: str, paper: bool, limit: int = 25) -> pd.DataFrame:
    orders = _trading_client(api_key, secret_key, paper).get_orders(
        GetOrdersRequest(limit=limit)
    )
    if not orders:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Symbol": o.symbol,
        "Side": o.side.value.upper(),
        "Qty": str(o.qty),
        "Type": o.order_type.value,
        "Status": o.status.value,
        "Filled Qty": str(o.filled_qty or "—"),
        "Fill Price": f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "—",
        "Submitted": o.submitted_at.strftime("%Y-%m-%d %H:%M") if o.submitted_at else "—",
    } for o in orders])


# ── Trader thread management ──────────────────────────────────────────────────

def _start_trader_thread(
    symbols: list,
    config: EngineConfig,
    api_key: str,
    secret_key: str,
    paper: bool,
    poll_seconds: float,
) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()

    def _run() -> None:
        try:
            market_data = AlpacaMarketDataProvider(api_key, secret_key)
            broker = AlpacaBroker(api_key, secret_key, paper)
            LiveTrader(symbols, config, market_data, broker, poll_seconds).run_forever(stop_event)
        except Exception as exc:
            st.session_state["trader_error"] = str(exc)
            st.session_state["trader_running"] = False

    thread = threading.Thread(target=_run, daemon=True, name="orb-live-trader")
    thread.start()
    return thread, stop_event


# ── Config builder ────────────────────────────────────────────────────────────

def _build_config(
    opening_range_minutes: int,
    breakout_buffer_bps: float,
    stop_loss_buffer_bps: float,
    target_rr: float,
    risk_per_trade_pct: float,
    max_position_value_pct: float,
    max_daily_loss_pct: float,
    allow_short: bool,
) -> EngineConfig:
    return EngineConfig(
        opening_range_minutes=opening_range_minutes,
        breakout_buffer_bps=breakout_buffer_bps,
        stop_loss_buffer_bps=stop_loss_buffer_bps,
        target_rr=target_rr,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_value_pct=max_position_value_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        allow_short=allow_short,
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="FinEx | Algorithmic Trading",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load credentials from environment — never from the UI
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    creds_present = bool(api_key and secret_key)

    # Session state defaults
    for key, default in {
        "paper": True,
        "connected": False,
        "trader_running": False,
        "trader_thread": None,
        "stop_event": None,
        "trader_error": None,
        "backtest_result": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Auto-connect on first load if credentials are present
    if creds_present and not st.session_state.connected:
        try:
            _fetch_account(api_key, secret_key, st.session_state.paper)
            st.session_state.connected = True
        except Exception:
            st.session_state.connected = False

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 FinEx")
        st.caption("Algorithmic Trading Engine")
        st.divider()

        st.markdown("### 🔗 Alpaca Connection")
        if not creds_present:
            st.warning(
                "Credentials not found. Set `ALPACA_API_KEY` and "
                "`ALPACA_SECRET_KEY` in your `.env` file and restart."
            )
        elif st.session_state.connected:
            st.success("Connected")
        else:
            st.error("Credentials found but connection failed.")

        paper_mode = st.toggle("Paper Trading", value=st.session_state.paper)
        if paper_mode != st.session_state.paper:
            st.session_state.paper = paper_mode
            st.session_state.connected = False
            st.rerun()

        st.divider()

        with st.expander("⚙️ Strategy Config", expanded=False):
            orm = st.number_input("Opening Range (min)", 1, 120, 30)
            bbb = st.number_input("Breakout Buffer (bps)", 0.0, 50.0, 5.0)
            slb = st.number_input("Stop Loss Buffer (bps)", 0.0, 50.0, 0.0)
            rr = st.number_input("Target R:R", 0.0, 10.0, 1.5)
            rpt = st.number_input("Risk per Trade (%)", 0.1, 5.0, 0.5) / 100
            mpp = st.number_input("Max Position Size (%)", 1.0, 50.0, 20.0) / 100
            mdl = st.number_input("Max Daily Loss (%)", 0.1, 10.0, 2.0) / 100
            allow_short = st.checkbox("Allow Short", value=False)

        config = _build_config(orm, bbb, slb, rr, rpt, mpp, mdl, allow_short)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_bt, tab_live = st.tabs(["📊 Backtest", "⚡ Live Trading"])

    # ── Backtest tab ──────────────────────────────────────────────────────────
    with tab_bt:
        st.header("Backtest")

        _TIMEFRAME_OPTIONS = {
            "1 min":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5 min":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15 min": TimeFrame(15, TimeFrameUnit.Minute),
            "30 min": TimeFrame(30, TimeFrameUnit.Minute),
            "1 hour": TimeFrame(1,  TimeFrameUnit.Hour),
        }

        ctrl_col, _ = st.columns([1, 2])
        with ctrl_col:
            source = st.radio(
                "Data Source",
                ["Alpaca Historical Data", "Upload CSV"],
                horizontal=True,
            )

            uploaded_file = None
            alpaca_symbols = []
            alpaca_start = alpaca_end = None
            alpaca_tf_label = "1 min"

            if source == "Upload CSV":
                uploaded_file = st.file_uploader("Bar data CSV", type="csv")
            else:
                if not creds_present:
                    st.warning("Connect Alpaca in the sidebar to fetch historical data.")
                alpaca_symbols_raw = st.text_input("Symbols", "SPY,QQQ")
                alpaca_symbols = [s.strip().upper() for s in alpaca_symbols_raw.split(",") if s.strip()]
                col_start, col_end = st.columns(2)
                with col_start:
                    alpaca_start = st.date_input("Start Date", value=date.today() - timedelta(days=90))
                with col_end:
                    alpaca_end = st.date_input("End Date", value=date.today() - timedelta(days=1))
                alpaca_tf_label = st.selectbox("Bar Timeframe", list(_TIMEFRAME_OPTIONS.keys()))

                if alpaca_start and alpaca_end and alpaca_start >= alpaca_end:
                    st.error("Start date must be before end date.")

            initial_equity = st.number_input(
                "Initial Equity ($)", 10_000, 10_000_000, 100_000, step=10_000
            )
            run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

        if run_bt:
            try:
                if source == "Upload CSV":
                    if not uploaded_file:
                        st.error("Upload a CSV file first.")
                        st.stop()
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        csv_path = Path(tmp.name)
                    with st.spinner("Running backtest…"):
                        bars = load_bars_from_csv(csv_path)
                        result = Backtester(config).run(bars, initial_equity=float(initial_equity))
                    st.session_state.backtest_result = result
                else:
                    if not creds_present:
                        st.error("Alpaca credentials required to fetch historical data.")
                        st.stop()
                    if not alpaca_symbols:
                        st.error("Enter at least one symbol.")
                        st.stop()
                    if alpaca_start >= alpaca_end:
                        st.error("Start date must be before end date.")
                        st.stop()
                    from datetime import datetime as dt
                    start_dt = dt.combine(alpaca_start, dt.min.time())
                    end_dt = dt.combine(alpaca_end, dt.max.time().replace(microsecond=0))
                    tf = _TIMEFRAME_OPTIONS[alpaca_tf_label]
                    with st.spinner(f"Fetching {alpaca_tf_label} bars for {', '.join(alpaca_symbols)}…"):
                        bars = fetch_historical_bars(api_key, secret_key, alpaca_symbols, start_dt, end_dt, tf)
                    if not bars:
                        st.error("No bars returned. Check your symbols and date range.")
                        st.stop()
                    st.caption(f"Fetched {len(bars):,} bars across {len(alpaca_symbols)} symbol(s).")
                    with st.spinner("Running backtest…"):
                        result = Backtester(config).run(bars, initial_equity=float(initial_equity))
                    st.session_state.backtest_result = result
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")

        result: Optional[BacktestResult] = st.session_state.backtest_result
        if result:
            st.divider()

            # KPI metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            pnl = result.final_equity - result.initial_equity
            m1.metric("Total Return", f"{result.total_return_pct:.2f}%", delta=f"${pnl:,.0f}")
            m2.metric("Final Equity", f"${result.final_equity:,.0f}")
            m3.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
            m4.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            m5.metric("Win Rate", f"{result.win_rate_pct:.1f}%")
            m6.metric("Total Trades", len(result.trades))

            st.plotly_chart(_equity_chart(result), use_container_width=True)

            dd_col, pnl_col = st.columns(2)
            with dd_col:
                st.plotly_chart(_drawdown_chart(result), use_container_width=True)
            with pnl_col:
                st.plotly_chart(_pnl_bar_chart(result), use_container_width=True)

            with st.expander("📋 Trade Log", expanded=False):
                trade_df = pd.DataFrame([{
                    "Symbol": t.symbol,
                    "Side": t.side.value.upper(),
                    "Qty": t.qty,
                    "Entry Time": t.entry_time,
                    "Exit Time": t.exit_time,
                    "Entry $": round(t.entry_price, 2),
                    "Exit $": round(t.exit_price, 2),
                    "Gross P&L": round(t.gross_pnl, 2),
                    "Net P&L": round(t.net_pnl, 2),
                    "Exit Reason": t.exit_reason,
                } for t in result.trades])
                st.dataframe(
                    trade_df,
                    column_config={
                        "Gross P&L": st.column_config.NumberColumn(format="$%.2f"),
                        "Net P&L": st.column_config.NumberColumn(format="$%.2f"),
                        "Entry $": st.column_config.NumberColumn(format="$%.2f"),
                        "Exit $": st.column_config.NumberColumn(format="$%.2f"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Live Trading tab ──────────────────────────────────────────────────────
    with tab_live:
        st.header("Live Trading")

        if not st.session_state.connected:
            st.info("Connect your Alpaca account in the sidebar to enable live trading.")
            st.stop()

        # Account overview
        try:
            acct = _fetch_account(
                api_key, secret_key, st.session_state.paper
            )
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
            a2.metric("Equity", f"${acct['equity']:,.2f}")
            a3.metric("Cash", f"${acct['cash']:,.2f}")
            a4.metric("Buying Power", f"${acct['buying_power']:,.2f}")
        except Exception as exc:
            st.error(f"Could not fetch account data: {exc}")

        mode_label = "🟡 Paper Trading" if st.session_state.paper else "🔴 Live Trading"
        st.caption(mode_label)
        st.divider()

        ctrl_col, pos_col = st.columns([1, 2])

        with ctrl_col:
            st.subheader("Trader Controls")
            symbols_input = st.text_input("Symbols (comma-separated)", "SPY,QQQ")
            poll_secs = st.slider("Poll Interval (s)", 10, 300, 60)
            is_running = st.session_state.trader_running

            if not is_running:
                if st.button("▶  Start Trader", type="primary", use_container_width=True):
                    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
                    if symbols:
                        thread, stop_event = _start_trader_thread(
                            symbols, config,
                            api_key,
                            secret_key,
                            st.session_state.paper,
                            float(poll_secs),
                        )
                        st.session_state.update({
                            "trader_running": True,
                            "trader_thread": thread,
                            "stop_event": stop_event,
                            "trader_error": None,
                        })
                        st.rerun()
                    else:
                        st.warning("Enter at least one symbol.")
            else:
                st.success("🟢 Trader is running")
                watching = symbols_input
                st.caption(f"Watching: {watching} · polling every {poll_secs}s")
                if st.button("⏹  Stop Trader", use_container_width=True):
                    if st.session_state.stop_event:
                        st.session_state.stop_event.set()
                    st.session_state.update({
                        "trader_running": False,
                        "trader_thread": None,
                        "stop_event": None,
                    })
                    st.rerun()

            if st.session_state.trader_error:
                st.error(f"Trader error: {st.session_state.trader_error}")

        with pos_col:
            st.subheader("Open Positions")
            try:
                pos_df = _fetch_positions(
                    api_key, secret_key, st.session_state.paper
                )
                if pos_df.empty:
                    st.info("No open positions.")
                else:
                    st.dataframe(
                        pos_df,
                        column_config={
                            "Avg Entry": st.column_config.NumberColumn(format="$%.2f"),
                            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                            "Market Value": st.column_config.NumberColumn(format="$%.2f"),
                            "Unrealized P&L": st.column_config.NumberColumn(format="$%.2f"),
                            "Unrealized P&L %": st.column_config.NumberColumn(format="%.2f%%"),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(f"Could not fetch positions: {exc}")

        st.divider()

        hdr_col, refresh_col = st.columns([4, 1])
        with hdr_col:
            st.subheader("Recent Orders")
        with refresh_col:
            st.write("")
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        try:
            orders_df = _fetch_orders(
                api_key, secret_key, st.session_state.paper
            )
            if orders_df.empty:
                st.info("No orders found.")
            else:
                st.dataframe(orders_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not fetch orders: {exc}")


if __name__ == "__main__":
    main()
