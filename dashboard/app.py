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
_GREEN  = "#00E676"
_RED    = "#FF4757"
_CYAN   = "#00D4FF"
_AMBER  = "#FFA502"
_BG0    = "#060A12"
_BG1    = "#0B1120"
_BG2    = "#101828"
_TEXT0  = "#EEF2F8"
_TEXT1  = "#8899BB"


def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Syne:wght@400;500;600;700;800&display=swap');

    :root {
      --bg-0:#060A12; --bg-1:#0B1120; --bg-2:#101828; --bg-3:#162033;
      --bd-dim:rgba(255,255,255,0.05); --bd-mid:rgba(255,255,255,0.10);
      --bd-accent:rgba(0,212,255,0.35);
      --t0:#EEF2F8; --t1:#8899BB; --t2:#4A5A72;
      --cyan:#00D4FF; --green:#00E676; --red:#FF4757; --amber:#FFA502;
      --mono:'IBM Plex Mono',monospace; --display:'Syne',sans-serif;
      --r:6px; --ease:0.18s cubic-bezier(.4,0,.2,1);
    }

    /* ── Shell ── */
    .stApp {
      background: var(--bg-0) !important;
      background-image:
        radial-gradient(ellipse 70% 50% at 5% 90%, rgba(0,80,180,.07) 0%, transparent 65%),
        radial-gradient(ellipse 55% 35% at 95% 5%, rgba(0,212,255,.04) 0%, transparent 55%);
      background-attachment: fixed;
    }
    .main .block-container { padding-top:1.5rem !important; max-width:1400px; }
    * { box-sizing: border-box; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
      background: var(--bg-1) !important;
      border-right: 1px solid var(--bd-dim) !important;
    }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1.25rem !important; }

    /* ── Typography ── */
    h1,h2,h3,h4 { font-family:var(--display) !important; letter-spacing:-.02em !important; color:var(--t0) !important; }
    h1 { font-size:1.85rem !important; font-weight:800 !important; }
    h2 { font-size:1.25rem !important; font-weight:700 !important; }
    h3 { font-size:1.05rem !important; font-weight:600 !important; }
    p, .stMarkdown p { font-family:var(--display) !important; color:var(--t1) !important; font-size:.88rem !important; }
    label, .stMarkdown label { font-family:var(--display) !important; }
    [data-testid="stCaptionContainer"] p {
      font-size:.7rem !important; color:var(--t2) !important; font-family:var(--display) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
      background:transparent !important;
      border-bottom:1px solid var(--bd-dim) !important;
      gap:0 !important; padding-bottom:0 !important;
    }
    .stTabs [data-baseweb="tab"] {
      font-family:var(--display) !important; font-weight:600 !important;
      font-size:.78rem !important; letter-spacing:.08em !important;
      text-transform:uppercase !important; color:var(--t2) !important;
      background:transparent !important; border:none !important;
      border-bottom:2px solid transparent !important;
      padding:.7rem 1.4rem !important;
      transition:color var(--ease), border-color var(--ease) !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color:var(--t1) !important; }
    .stTabs [aria-selected="true"] {
      color:var(--cyan) !important;
      border-bottom-color:var(--cyan) !important;
      background:transparent !important;
    }
    .stTabs [data-baseweb="tab-panel"] { background:transparent !important; padding-top:1.5rem !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
      background:var(--bg-2) !important; border:1px solid var(--bd-dim) !important;
      border-radius:var(--r) !important; padding:1rem 1.1rem !important;
      transition:background var(--ease), border-color var(--ease), box-shadow var(--ease) !important;
    }
    [data-testid="metric-container"]:hover {
      background:var(--bg-3) !important; border-color:var(--bd-mid) !important;
      box-shadow:0 0 0 1px var(--bd-accent) !important;
    }
    [data-testid="stMetricLabel"] > div {
      font-family:var(--display) !important; font-size:.62rem !important;
      font-weight:700 !important; text-transform:uppercase !important;
      letter-spacing:.14em !important; color:var(--t2) !important;
    }
    [data-testid="stMetricValue"] > div {
      font-family:var(--mono) !important; font-size:1.35rem !important;
      font-weight:500 !important; color:var(--t0) !important; letter-spacing:-.02em !important;
    }
    [data-testid="stMetricDelta"] > div {
      font-family:var(--mono) !important; font-size:.72rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
      font-family:var(--display) !important; font-weight:700 !important;
      font-size:.78rem !important; letter-spacing:.07em !important;
      text-transform:uppercase !important; border-radius:var(--r) !important;
      padding:.52rem 1.1rem !important;
      transition:all var(--ease) !important;
    }
    .stButton > button[kind="primary"] {
      background:rgba(0,180,220,0.10) !important;
      color:var(--cyan) !important;
      border:1px solid rgba(0,212,255,0.30) !important;
    }
    .stButton > button[kind="primary"]:hover {
      background:rgba(0,180,220,0.18) !important;
      border-color:rgba(0,212,255,0.55) !important;
      box-shadow:0 0 20px rgba(0,212,255,0.15) !important;
      transform:translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
      background:var(--bg-2) !important; color:var(--t1) !important;
      border:1px solid var(--bd-mid) !important;
    }
    .stButton > button[kind="secondary"]:hover {
      background:var(--bg-3) !important; color:var(--t0) !important;
      border-color:var(--bd-accent) !important;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
      background:var(--bg-2) !important; border:1px solid var(--bd-mid) !important;
      border-radius:var(--r) !important; color:var(--t0) !important;
      font-family:var(--mono) !important; font-size:.84rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
      border-color:var(--cyan) !important;
      box-shadow:0 0 0 3px rgba(0,212,255,.12) !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stMultiSelect label, .stRadio > label, .stSlider label, .stDateInput label {
      font-family:var(--display) !important; font-size:.65rem !important;
      font-weight:700 !important; letter-spacing:.12em !important;
      text-transform:uppercase !important; color:var(--t2) !important;
    }

    /* ── Select / Multi ── */
    [data-baseweb="select"] > div {
      background:var(--bg-2) !important; border:1px solid var(--bd-mid) !important;
      border-radius:var(--r) !important;
    }
    [data-baseweb="select"] span { font-family:var(--mono) !important; color:var(--t0) !important; font-size:.84rem !important; }
    [data-baseweb="tag"] {
      background:rgba(0,212,255,.15) !important;
      border:1px solid rgba(0,212,255,.3) !important;
      border-radius:3px !important;
    }
    [data-baseweb="tag"] span { font-family:var(--mono) !important; color:var(--cyan) !important; font-size:.75rem !important; }

    /* ── Radio ── */
    .stRadio > div > label > div { font-family:var(--mono) !important; color:var(--t1) !important; font-size:.82rem !important; }

    /* ── Date input ── */
    .stDateInput > div > div > input {
      background:var(--bg-2) !important; border:1px solid var(--bd-mid) !important;
      border-radius:var(--r) !important; color:var(--t0) !important;
      font-family:var(--mono) !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
      background:var(--cyan) !important;
    }

    /* ── Toggle ── */
    [data-testid="stToggle"] span { font-family:var(--display) !important; font-size:.84rem !important; color:var(--t1) !important; }

    /* ── Divider ── */
    hr { border-color:var(--bd-dim) !important; margin:.75rem 0 !important; }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
      border-radius:var(--r) !important; border-left-width:3px !important;
      font-family:var(--display) !important; font-size:.82rem !important;
      background:rgba(255,255,255,.03) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
      border:1px solid var(--bd-dim) !important;
      border-radius:var(--r) !important; background:var(--bg-2) !important;
    }
    [data-testid="stExpander"] summary {
      font-family:var(--display) !important; font-weight:600 !important;
      color:var(--t1) !important; font-size:.83rem !important; letter-spacing:.02em !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
      border:1px solid var(--bd-dim) !important; border-radius:var(--r) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width:4px; height:4px; }
    ::-webkit-scrollbar-track { background:var(--bg-1); }
    ::-webkit-scrollbar-thumb { background:var(--bd-mid); border-radius:2px; }
    ::-webkit-scrollbar-thumb:hover { background:var(--t2); }

    /* ── Custom components ── */
    .finex-brand {
      font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
      letter-spacing:-.04em; line-height:1;
      background:linear-gradient(135deg,#00D4FF 0%,#0077FF 55%,#00E676 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    }
    .finex-sub {
      font-family:'Syne',sans-serif; font-size:.6rem; font-weight:700;
      letter-spacing:.22em; text-transform:uppercase; color:var(--t2); margin-top:.2rem;
    }
    .status-row {
      display:flex; align-items:center; gap:.5rem;
      font-family:'Syne',sans-serif; font-size:.75rem; font-weight:600;
      color:var(--t1); margin:.6rem 0;
    }
    .dot {
      width:7px; height:7px; border-radius:50%; flex-shrink:0;
      animation:pulse-dot 2.4s ease-in-out infinite;
    }
    .dot-green { background:var(--green); box-shadow:0 0 6px var(--green); }
    .dot-red   { background:var(--red);   box-shadow:0 0 6px var(--red); }
    .dot-amber { background:var(--amber); box-shadow:0 0 6px var(--amber); }
    @keyframes pulse-dot {
      0%,100% { opacity:1; transform:scale(1); }
      50%      { opacity:.5; transform:scale(.85); }
    }
    .section-label {
      font-family:'Syne',sans-serif; font-size:.6rem; font-weight:700;
      letter-spacing:.18em; text-transform:uppercase; color:var(--t2);
      margin-bottom:.5rem; display:block;
    }
    .mode-badge {
      display:inline-block; padding:.15rem .5rem; border-radius:3px;
      font-family:'IBM Plex Mono',monospace; font-size:.65rem; font-weight:600;
      letter-spacing:.06em; text-transform:uppercase;
    }
    .mode-paper { background:rgba(255,165,2,.12); color:var(--amber); border:1px solid rgba(255,165,2,.3); }
    .mode-live  { background:rgba(255,71,87,.12);  color:var(--red);   border:1px solid rgba(255,71,87,.3); }
    </style>
    """, unsafe_allow_html=True)

# ── Ticker catalog ────────────────────────────────────────────────────────────
# All symbols verified available on Alpaca IEX feed.
# Format: (TICKER, Company Name)
_TICKER_CATALOG: dict[str, list[tuple[str, str]]] = {
    "ETFs": [
        ("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000"),
        ("DIA", "Dow Jones Industrial"), ("VTI", "Total US Market"),
        ("GLD", "Gold"), ("SLV", "Silver"), ("TLT", "20yr Treasury"),
        ("XLF", "Financials Sector"), ("XLK", "Technology Sector"),
        ("XLE", "Energy Sector"), ("XLV", "Healthcare Sector"),
        ("XLI", "Industrials Sector"), ("XLY", "Consumer Discretionary"),
        ("ARKK", "ARK Innovation"),
    ],
    "Tech": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"),
        ("GOOGL", "Alphabet"), ("META", "Meta Platforms"), ("AMZN", "Amazon"),
        ("TSLA", "Tesla"), ("AMD", "Advanced Micro Devices"), ("AVGO", "Broadcom"),
        ("CRM", "Salesforce"), ("ORCL", "Oracle"), ("ADBE", "Adobe"),
        ("NFLX", "Netflix"), ("UBER", "Uber"), ("SNOW", "Snowflake"),
        ("PLTR", "Palantir"), ("COIN", "Coinbase"),
    ],
    "Finance": [
        ("JPM", "JPMorgan Chase"), ("BAC", "Bank of America"),
        ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"),
        ("WFC", "Wells Fargo"), ("C", "Citigroup"),
        ("V", "Visa"), ("MA", "Mastercard"), ("AXP", "American Express"),
        ("SCHW", "Charles Schwab"), ("KKR", "KKR & Co"),
    ],
    "Healthcare": [
        ("UNH", "UnitedHealth Group"), ("LLY", "Eli Lilly"),
        ("JNJ", "Johnson & Johnson"), ("ABBV", "AbbVie"),
        ("MRK", "Merck"), ("PFE", "Pfizer"), ("MRNA", "Moderna"),
        ("AMGN", "Amgen"), ("BMY", "Bristol-Myers Squibb"),
    ],
    "Energy": [
        ("XOM", "ExxonMobil"), ("CVX", "Chevron"),
        ("COP", "ConocoPhillips"), ("OXY", "Occidental Petroleum"),
        ("MPC", "Marathon Petroleum"), ("PSX", "Phillips 66"),
    ],
    "Consumer": [
        ("WMT", "Walmart"), ("AMZN", "Amazon"), ("COST", "Costco"),
        ("HD", "Home Depot"), ("LOW", "Lowe's"), ("TGT", "Target"),
        ("MCD", "McDonald's"), ("SBUX", "Starbucks"), ("NKE", "Nike"),
        ("LULU", "Lululemon"),
    ],
}


# ── Chart builders ────────────────────────────────────────────────────────────

def _base_layout(title: str, height: int, **extra) -> dict:
    """Shared dark theme for all Plotly charts."""
    axis = dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="IBM Plex Mono", color="#4A5A72", size=10),
        linecolor="rgba(255,255,255,0.05)",
    )
    layout = dict(
        title=dict(text=title, font=dict(family="Syne", color="#8899BB", size=12), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,30,0.5)",
        font=dict(family="IBM Plex Mono", color="#4A5A72", size=10),
        xaxis={**axis},
        yaxis={**axis},
        height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0B1120",
            bordercolor="rgba(0,212,255,0.4)",
            font=dict(family="IBM Plex Mono", color="#EEF2F8", size=11),
        ),
        legend=dict(
            font=dict(family="IBM Plex Mono", color="#8899BB", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    layout.update(extra)
    return layout


def _equity_chart(result: BacktestResult) -> go.Figure:
    dates  = [p[0] for p in result.equity_curve]
    equity = [p[1] for p in result.equity_curve]
    fig = go.Figure()

    # Buy-and-hold benchmark overlay (drawn first so it renders beneath strategy)
    if getattr(result, "benchmark_curve", None):
        bm_dates  = [p[0] for p in result.benchmark_curve]  # type: ignore[attr-defined]
        bm_equity = [p[1] for p in result.benchmark_curve]  # type: ignore[attr-defined]
        fig.add_trace(go.Scatter(
            x=bm_dates, y=bm_equity, mode="lines", name="Buy & Hold",
            line=dict(color=_AMBER, width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>B&H <b>$%{y:,.2f}</b><extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines", name="ORB Strategy",
        line=dict(color=_GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.06)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>ORB <b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("EQUITY CURVE", 340,
        yaxis=dict(tickformat="$,.0f", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="IBM Plex Mono", color="#4A5A72", size=10))))
    return fig


def _drawdown_chart(result: BacktestResult) -> go.Figure:
    equity = [p[1] for p in result.equity_curve]
    dates  = [p[0] for p in result.equity_curve]
    peak, drawdowns = equity[0], []
    for eq in equity:
        peak = max(peak, eq)
        drawdowns.append((eq - peak) / peak * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdowns, mode="lines", name="Drawdown",
        line=dict(color=_RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,71,87,0.10)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.2f}%</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("DRAWDOWN", 240,
        yaxis=dict(tickformat=".1f", ticksuffix="%", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="IBM Plex Mono", color="#4A5A72", size=10))))
    return fig


def _pnl_bar_chart(result: BacktestResult) -> go.Figure:
    pnls   = [t.net_pnl for t in result.trades]
    colors = [_GREEN if p > 0 else _RED for p in pnls]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(pnls) + 1)), y=pnls,
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="Trade %{x}<br><b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
    fig.update_layout(**_base_layout("TRADE P&L", 240,
        xaxis=dict(title="Trade #", tickfont=dict(family="IBM Plex Mono", color="#4A5A72", size=10),
                   gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(tickformat="$,.0f", gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(family="IBM Plex Mono", color="#4A5A72", size=10)),
        showlegend=False))
    return fig


def _monthly_heatmap(monthly_returns: dict) -> go.Figure:
    import calendar
    year_data: dict[int, dict[int, float]] = {}
    for key, ret in monthly_returns.items():
        y, m = int(key[:4]), int(key[5:])
        year_data.setdefault(y, {})[m] = ret
    years = sorted(year_data)
    month_names = [calendar.month_abbr[m] for m in range(1, 13)]
    z    = [[year_data[y].get(m) for m in range(1, 13)] for y in years]
    text = [[f"{v:.1f}%" if v is not None else "" for v in row] for row in z]
    colorscale = [
        [0.0,  _RED],
        [0.45, "rgba(255,71,87,0.15)"],
        [0.5,  "rgba(255,255,255,0.05)"],
        [0.55, "rgba(0,230,118,0.15)"],
        [1.0,  _GREEN],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=month_names, y=[str(y) for y in years],
        colorscale=colorscale, zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono", size=11),
        showscale=True,
        colorbar=dict(
            tickfont=dict(family="IBM Plex Mono", color="#8899BB", size=9),
            outlinecolor="rgba(0,0,0,0)", thickness=8,
        ),
        hovertemplate="%{y} %{x}: <b>%{z:.2f}%</b><extra></extra>",
    ))
    fig.update_layout(**_base_layout("MONTHLY RETURNS", max(200, 80 * len(years) + 100),
        xaxis=dict(tickfont=dict(family="IBM Plex Mono", color="#8899BB", size=11),
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono", color="#8899BB", size=11),
                   gridcolor="rgba(0,0,0,0)")))
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
    _inject_css()

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
        st.markdown(
            '<div class="finex-brand">FinEx</div>'
            '<div class="finex-sub">Algorithmic Trading Engine</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Connection status
        if not creds_present:
            st.markdown(
                '<div class="status-row"><span class="dot dot-red"></span>No credentials — set <code>ALPACA_API_KEY</code> and <code>ALPACA_SECRET_KEY</code> in <code>.env</code></div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.connected:
            st.markdown(
                '<div class="status-row"><span class="dot dot-green"></span>Alpaca connected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-row"><span class="dot dot-amber"></span>Credentials found — connection failed</div>',
                unsafe_allow_html=True,
            )

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

        # Resolve symbol selection from session state before columns render
        _selected_opts: list[str] = st.session_state.get("bt_ticker_selection", [])
        _custom_raw: str = st.session_state.get("bt_custom_ticker", "")
        alpaca_symbols: list[str] = (
            [o.split(" — ")[0] for o in _selected_opts]
            + [s.strip().upper() for s in _custom_raw.split(",") if s.strip()]
        )

        ctrl_col, ticker_col = st.columns([1, 1])

        with ctrl_col:
            source = st.radio(
                "Data Source",
                ["Alpaca Historical Data", "Upload CSV"],
                horizontal=True,
            )

            uploaded_file = None
            alpaca_start = alpaca_end = None
            alpaca_tf_label = "1 min"

            if source == "Upload CSV":
                uploaded_file = st.file_uploader("Bar data CSV", type="csv")
            else:
                if not creds_present:
                    st.warning("Connect Alpaca in the sidebar to fetch historical data.")

                col_start, col_end = st.columns(2)
                with col_start:
                    alpaca_start = st.date_input("Start Date", value=date.today() - timedelta(days=90))
                with col_end:
                    alpaca_end = st.date_input("End Date", value=date.today() - timedelta(days=1))
                alpaca_tf_label = st.selectbox("Bar Timeframe", list(_TIMEFRAME_OPTIONS.keys()))
                st.caption("💡 1–5 min bars work best for ORB. Longer bars = fewer signals.")

                if alpaca_start and alpaca_end and alpaca_start >= alpaca_end:
                    st.error("Start date must be before end date.")

                if alpaca_symbols:
                    st.success(f"Symbols: **{', '.join(alpaca_symbols)}**")
                else:
                    st.warning("Select at least one symbol from the panel →")

            initial_equity = st.number_input(
                "Initial Equity ($)", 10_000, 10_000_000, 100_000, step=10_000
            )
            run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

        with ticker_col:
            if source == "Alpaca Historical Data":
                st.markdown("#### 🔍 Symbol Search")
                st.caption("Search by company name or ticker — all symbols verified on Alpaca.")

                all_opts = [
                    f"{sym} — {name}"
                    for tickers in _TICKER_CATALOG.values()
                    for sym, name in tickers
                ]
                sector_filter = st.selectbox(
                    "Filter by sector",
                    ["All Sectors"] + list(_TICKER_CATALOG.keys()),
                )
                filtered_opts = (
                    all_opts if sector_filter == "All Sectors"
                    else [f"{sym} — {name}" for sym, name in _TICKER_CATALOG[sector_filter]]
                )

                # Preserve selections that are still in the filtered list
                valid_defaults = [o for o in _selected_opts if o in filtered_opts]

                st.multiselect(
                    "Search tickers",
                    options=filtered_opts,
                    default=valid_defaults,
                    placeholder="Type 'Apple', 'energy', 'SPY'…",
                    label_visibility="collapsed",
                    key="bt_ticker_selection",
                )
                st.text_input(
                    "Unlisted symbol",
                    placeholder="e.g. HOOD, RIVN, MSTR  (comma-separated)",
                    label_visibility="collapsed",
                    key="bt_custom_ticker",
                )

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

            # Row 1: return overview
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            pnl = result.final_equity - result.initial_equity
            bm_return = getattr(result, "benchmark_return_pct", None)
            bm_curve  = getattr(result, "benchmark_curve", [])
            alpha = result.total_return_pct - bm_return if bm_return is not None else None
            alpha_label = (
                f"α {alpha:+.2f}% vs B&H ({bm_return:.2f}%)"
                if alpha is not None and bm_curve else f"${pnl:,.0f}"
            )
            m1.metric("Total Return", f"{result.total_return_pct:.2f}%", delta=alpha_label)
            m2.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
            m3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            m4.metric("Profit Factor", f"{result.profit_factor:.2f}" if result.profit_factor != float('inf') else "∞")
            m5.metric("Win Rate", f"{result.win_rate_pct:.1f}%")
            m6.metric("Total Trades", len(result.trades))

            # Row 2: trade-level breakdown
            n1, n2, n3, n4, n5, n6 = st.columns(6)
            n1.metric("Avg Win", f"${result.avg_win:,.0f}")
            n2.metric("Avg Loss", f"${result.avg_loss:,.0f}")
            n3.metric("Best Trade", f"${result.best_trade:,.0f}")
            n4.metric("Worst Trade", f"${result.worst_trade:,.0f}")
            n5.metric("Max Win Streak", result.max_consecutive_wins)
            n6.metric("Max Loss Streak", result.max_consecutive_losses)

            st.plotly_chart(_equity_chart(result), use_container_width=True)

            dd_col, pnl_col = st.columns(2)
            with dd_col:
                st.plotly_chart(_drawdown_chart(result), use_container_width=True)
            with pnl_col:
                st.plotly_chart(_pnl_bar_chart(result), use_container_width=True)

            if result.monthly_returns:
                st.plotly_chart(_monthly_heatmap(result.monthly_returns), use_container_width=True)

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

        badge_cls = "mode-paper" if st.session_state.paper else "mode-live"
        badge_txt = "Paper Trading" if st.session_state.paper else "⚠ Live Trading"
        st.markdown(f'<span class="mode-badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
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
