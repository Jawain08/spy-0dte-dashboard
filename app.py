"""
Jawain's Trading Strategy — 0DTE SPY Dashboard & Signal Engine
===============================================================
A Streamlit application that ingests intraday 1-minute SPY data (real-time
via Alpaca, with a yfinance fallback), computes technical indicators
(session-anchored VWAP, RSI, fast/slow SMA), and flags rules-based
'Ideal Call' / 'Ideal Put' entry scenarios with a midday chop-zone filter.

DISCLAIMER: Educational tool only. Not financial advice. 0DTE options carry
extreme risk, including total loss of premium.

Run locally:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

from data_sources import (fetch_intraday_data_alpaca, fetch_news_alpaca,
                          fetch_atm_0dte_contract)
from glossary import GLOSSARY

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Jawain's Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #161B26 0%, #12172200 100%);
    border: 1px solid #262D3D;
    border-left: 3px solid #F5A623;
    border-radius: 10px;
    padding: 12px 14px;
}
[data-testid="stMetricLabel"] { color: #8B93A7; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
h1 {
    background: linear-gradient(90deg, #F5A623 0%, #26A69A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 8px 18px; }
.stTabs [aria-selected="true"] { border-bottom: 3px solid #F5A623 !important; }
div[data-testid="stSidebarContent"] { border-right: 1px solid #262D3D; }
</style>
"""

EASTERN = "America/New_York"  # canonical IANA key ("US/Eastern" is a legacy alias)
CHOP_START = dt.time(11, 30)   # Midday chop zone start (ET)
CHOP_END = dt.time(13, 30)     # Midday chop zone end (ET)
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)


# ---------------------------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner="Fetching intraday SPY data…")
def fetch_intraday_data(ticker: str = "SPY", lookback_days: int = 1) -> pd.DataFrame:
    """
    Fetch near-live 1-minute candlestick data for `ticker` via yfinance.
    """
    period = f"{max(1, min(lookback_days, 7))}d"
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1m",
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EASTERN)
    df.index.name = "Timestamp"

    df = df.between_time(MARKET_OPEN, MARKET_CLOSE)
    df = df.dropna(subset=["Close", "Volume"])
    return df


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes <= 1 or df.empty:
        return df
    out = df.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])
    return out[out["Volume"] > 0]


# ---------------------------------------------------------------------------
# 2. TECHNICAL INDICATOR ENGINE
# ---------------------------------------------------------------------------
def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int,
    fast_sma: int,
    slow_sma: int,
) -> pd.DataFrame:
    out = df.copy()

    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    session = out.index.date
    cum_pv = (typical_price * out["Volume"]).groupby(session).cumsum()
    cum_vol = out["Volume"].groupby(session).cumsum().replace(0, np.nan)
    out["VWAP"] = cum_pv / cum_vol

    out["RSI"] = RSIIndicator(close=out["Close"], window=rsi_period).rsi()
    out["SMA_Fast"] = SMAIndicator(close=out["Close"], window=fast_sma).sma_indicator()
    out["SMA_Slow"] = SMAIndicator(close=out["Close"], window=slow_sma).sma_indicator()

    return out


# ---------------------------------------------------------------------------
# 3. AUTOMATED SIGNALING LOGIC
# ---------------------------------------------------------------------------
def generate_signals(
    df: pd.DataFrame,
    call_rsi_min: float,
    call_rsi_max: float,
    put_rsi_min: float,
    put_rsi_max: float,
    require_sma_alignment: bool,
    apply_chop_filter: bool,
    use_volume_filter: bool = False,
    vol_mult: float = 1.5,
) -> pd.DataFrame:
    out = df.copy()

    prev_close = out["Close"].shift(1)
    prev_vwap = out["VWAP"].shift(1)
    prev_rsi = out["RSI"].shift(1)

    cross_above_vwap = (prev_close <= prev_vwap) & (out["Close"] > out["VWAP"])
    cross_below_vwap = (prev_close >= prev_vwap) & (out["Close"] < out["VWAP"])

    rsi_rising = out["RSI"] > prev_rsi
    rsi_falling = out["RSI"] < prev_rsi

    call_rsi_ok = out["RSI"].between(call_rsi_min, call_rsi_max)
    put_rsi_ok = out["RSI"].between(put_rsi_min, put_rsi_max)

    call_signal = cross_above_vwap & rsi_rising & call_rsi_ok
    put_signal = cross_below_vwap & rsi_falling & put_rsi_ok

    if require_sma_alignment:
        call_signal &= out["SMA_Fast"] > out["SMA_Slow"]
        put_signal &= out["SMA_Fast"] < out["SMA_Slow"]

    if use_volume_filter:
        vol_avg = out["Volume"].rolling(20).mean().shift(1)
        vol_ok = out["Volume"] > vol_mult * vol_avg
        call_signal &= vol_ok
        put_signal &= vol_ok

    if apply_chop_filter:
        bar_times = pd.Series(out.index.time, index=out.index)
        in_chop_zone = (bar_times >= CHOP_START) & (bar_times < CHOP_END)
        call_signal &= ~in_chop_zone
        put_signal &= ~in_chop_zone
        out["In_Chop_Zone"] = in_chop_zone
    else:
        out["In_Chop_Zone"] = False

    out["Call_Signal"] = call_signal.fillna(False)
    out["Put_Signal"] = put_signal.fillna(False)
    return out


def compute_signal_outcomes(
    df: pd.DataFrame, tf_minutes: int = 1
) -> pd.DataFrame:
    horizon_map = {1: (5, 10, 15), 5: (5, 15, 30), 30: (30, 60, 90)}
    horizons_min = horizon_map.get(tf_minutes, (5, 10, 15))
    rows = []
    for label, mask, direction in (
        ("🟢 Call", df["Call_Signal"], 1),
        ("🔴 Put", df["Put_Signal"], -1),
    ):
        if not mask.any():
            continue
        for h_min in horizons_min:
            h_bars = max(1, h_min // tf_minutes)
            fwd = (df["Close"].shift(-h_bars) - df["Close"]) * direction
            moves = fwd[mask].dropna()
            if moves.empty:
                continue
            rows.append(
                {
                    "Signal": label,
                    "Horizon": f"+{h_min} min",
                    "Signals": len(moves),
                    "Favorable %": round(100 * (moves > 0).mean(), 1),
                    "Avg move ($)": round(moves.mean(), 3),
                }
            )
    return pd.DataFrame(rows)


def opening_range(df: pd.DataFrame):
    today = df.index[-1].date()
    session = df[df.index.date == today]
    orb = session.between_time(MARKET_OPEN, dt.time(10, 0))
    if orb.empty:
        return None, None
    return float(orb["High"].max()), float(orb["Low"].min())


@st.cache_data(ttl=300, show_spinner=False)
def fetch_vix1d() -> float | None:
    try:
        hist = yf.Ticker("^VIX1D").history(period="5d")
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def expected_move(session_open: float, vix1d: float) -> float:
    return session_open * (vix1d / 100.0) / np.sqrt(252)


def prior_day_levels(df: pd.DataFrame) -> dict | None:
    dates = sorted(set(df.index.date))
    if len(dates) < 2:
        return None
    prev = df[df.index.date == dates[-2]]
    return {
        "PD high": float(prev["High"].max()),
        "PD low": float(prev["Low"].min()),
        "PD close": float(prev["Close"].iloc[-1]),
    }


def relative_volume(df: pd.DataFrame) -> float | None:
    dates = sorted(set(df.index.date))
    if len(dates) < 2:
        return None
    today = df[df.index.date == dates[-1]]
    n = len(today)
    prior = [
        float(df[df.index.date == d]["Volume"].iloc[:n].sum())
        for d in dates[:-1]
        if len(df[df.index.date == d]) >= n
    ]
    if not prior or np.mean(prior) == 0:
        return None
    return float(today["Volume"].sum()) / float(np.mean(prior))


# ---------------------------------------------------------------------------
# 4. DASHBOARD USER INTERFACE
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, show_smas: bool, shade_chop: bool,
                orh: float = None, orl: float = None,
                levels: dict = None, height: int = 760,
                tf_label: str = "1 min") -> go.Figure:
    
    last_close = float(df["Close"].iloc[-1])
    session_open = float(df[df.index.date == df.index[-1].date()]["Open"].iloc[0])
    day_chg = last_close - session_open
    px_color = "#26a69a" if day_chg >= 0 else "#ef5350"
    arrow = "▲" if day_chg >= 0 else "▼"
    header = (f"<b>SPY  ${last_close:,.2f}</b>   {arrow} {day_chg:+.2f} "
              f"({day_chg / session_open * 100:+.2f}%)  ·  {tf_label}")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=(header, "RSI"),
    )
    fig.layout.annotations[0].font = dict(size=24, color=px_color)
    fig.layout.annotations[0].x = 0
    fig.layout.annotations[0].xanchor = "left"

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="SPY",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            mode="lines", line=dict(color="#f5a623", width=2, dash="dot"),
        ), row=1, col=1,
    )

    if show_smas:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_Fast"], name="Fast SMA", mode="lines", line=dict(color="#42a5f5", width=1.3)), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_Slow"], name="Slow SMA", mode="lines", line=dict(color="#ab47bc", width=1.3)), row=1, col=1,
        )

    calls = df[df["Call_Signal"]]
    puts = df[df["Put_Signal"]]

    fig.add_trace(
        go.Scatter(
            x=calls.index, y=calls["Low"] * 0.9985, name="Ideal CALL", mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#00e676", line=dict(width=1, color="#004d26")),
            hovertemplate="CALL @ %{x|%H:%M} — $%{customdata:.2f}<extra></extra>", customdata=calls["Close"],
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=puts.index, y=puts["High"] * 1.0015, name="Ideal PUT", mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#ff1744", line=dict(width=1, color="#5c0011")),
            hovertemplate="PUT @ %{x|%H:%M} — $%{customdata:.2f}<extra></extra>", customdata=puts["Close"],
        ), row=1, col=1,
    )

    fig.add_hline(
        y=last_close, line=dict(color=px_color, width=1, dash="dot"),
        annotation_text=f"  {last_close:,.2f}", annotation_position="right",
        annotation_font=dict(color=px_color, size=12), row=1, col=1,
    )

    if orh is not None and orl is not None:
        fig.add_hline(y=orh, line=dict(color="#5DCAA5", dash="dash", width=1), annotation_text="OR high", annotation_font_size=10, row=1, col=1)
        fig.add_hline(y=orl, line=dict(color="#F0997B", dash="dash", width=1), annotation_text="OR low", annotation_font_size=10, row=1, col=1)

    if levels:
        styles = {
            "EM high": dict(color="#378ADD", dash="dashdot", width=1.2), "EM low": dict(color="#378ADD", dash="dashdot", width=1.2),
            "PD high": dict(color="#AFA9EC", dash="dot", width=1), "PD low": dict(color="#AFA9EC", dash="dot", width=1),
            "PD close": dict(color="#B4B2A9", dash="dot", width=1),
        }
        for name, value in levels.items():
            if value is None: continue
            fig.add_hline(y=value, line=styles.get(name, dict(width=1)), annotation_text=name, annotation_font_size=10, row=1, col=1)

    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines", line=dict(color="#90caf9", width=1.5)), row=2, col=1,
    )
    fig.add_hline(y=70, line=dict(color="#ef5350", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=50, line=dict(color="#9e9e9e", dash="dot", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#26a69a", dash="dash", width=1), row=2, col=1)

    if shade_chop:
        for session_date in sorted(set(df.index.date)):
            start = pd.Timestamp.combine(session_date, CHOP_START).tz_localize(EASTERN)
            end = pd.Timestamp.combine(session_date, CHOP_END).tz_localize(EASTERN)
            fig.add_vrect(
                x0=start, x1=end, fillcolor="rgba(158,158,158,0.12)", line_width=0,
                annotation_text="chop zone", annotation_position="top left", annotation_font_size=10, row=1, col=1,
            )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="top", y=-0.07, x=0),
        margin=dict(l=40, r=78, t=76, b=80), hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("📈 Jawain's Trading Strategy")
    st.caption(
        "0DTE SPY signal engine — VWAP/RSI trend-following on 1-minute bars. "
        "**Educational tool — not financial advice.**"
    )

    # -----------------------------------------------------------------------
    # 🆕 Massive Top Price Banner Placeholder 
    # (Gets filled out later in the code once data is downloaded)
    # -----------------------------------------------------------------------
    top_price_banner = st.empty()

    # ---------------------- Sidebar controls -------------------------------
    with st.sidebar:
        st.header("⚙️ Strategy Controls")

        st.subheader("Data Source")
        data_source = st.radio(
            "Feed",
            ["Alpaca — IEX (free, real-time)", "Alpaca — SIP (paid, full tape)", "Yahoo Finance (15-min delayed)"],
            index=0,
        )

        tf_label = st.radio("Chart timeframe", ["1 min", "5 min", "30 min"], index=0, horizontal=True)
        tf_minutes = int(tf_label.split()[0])

        lookback_days = st.slider("Lookback (trading days)", 1, 5, 1)

        st.subheader("Moving Averages")
        fast_sma = st.slider("Fast SMA period", 3, 50, 9)
        slow_sma = st.slider("Slow SMA period", 10, 200, 21)
        if fast_sma >= slow_sma:
            st.warning("Fast SMA should be shorter than Slow SMA.")
        require_sma_alignment = st.checkbox("Require SMA trend alignment", value=False)

        st.subheader("RSI")
        rsi_period = st.slider("RSI period", 5, 30, 14)
        call_rsi_min, call_rsi_max = st.slider("CALL RSI band (rising)", 0, 100, (50, 65))
        put_rsi_min, put_rsi_max = st.slider("PUT RSI band (falling)", 0, 100, (35, 50))

        st.subheader("Filters")
        apply_chop_filter = st.checkbox("Block midday chop zone (11:30–1:30 ET)", value=True)
        use_volume_filter = st.checkbox("Require volume confirmation", value=False)
        vol_mult = st.slider("Volume multiple", 1.0, 3.0, 1.5, 0.1, disabled=not use_volume_filter)
        show_orb = st.checkbox("Show opening range (9:30–10:00)", value=True)
        show_context = st.checkbox("Show expected move & key levels", value=True)
        show_atm = st.checkbox("Show ATM 0DTE contract quotes", value=True)
        shade_chop = st.checkbox("Shade chop zone on chart", value=True)
        show_smas = st.checkbox("Show SMAs on chart", value=True)

        st.subheader("Layout")
        layout_mode = st.radio("Screen", ["Phone (folded)", "Unfolded (10-inch)", "Desktop"], index=0)
        compact = layout_mode == "Phone (folded)"
        chart_height = {"Phone (folded)": 460, "Unfolded (10-inch)": 620, "Desktop": 760}[layout_mode]

        st.divider()
        if st.button("🔄 Refresh data", use_container_width=True):
            fetch_intraday_data.clear()
            fetch_intraday_data_alpaca.clear()
            st.rerun()

    # ---------------------- Pipeline ---------------------------------------
    if data_source.startswith("Alpaca"):
        feed = "iex" if "IEX" in data_source else "sip"
        raw = fetch_intraday_data_alpaca("SPY", lookback_days, feed=feed)
        if raw.empty and "ALPACA_API_KEY" not in st.secrets:
            st.warning("Falling back to Yahoo Finance (15-min delayed) until Alpaca keys are added.")
            raw = fetch_intraday_data("SPY", lookback_days)
    else:
        raw = fetch_intraday_data("SPY", lookback_days)
    if raw.empty:
        st.error("No intraday data returned. Check market status or API keys.")
        st.stop()

    raw = resample_bars(raw, tf_minutes)
    min_bars = slow_sma + rsi_period
    if len(raw) < min_bars:
        st.warning(f"Only {len(raw)} bars — indicators need ~{min_bars} to warm up.")

    df = compute_indicators(raw, rsi_period, fast_sma, slow_sma)
    df = generate_signals(
        df, call_rsi_min, call_rsi_max, put_rsi_min, put_rsi_max,
        require_sma_alignment, apply_chop_filter, use_volume_filter, vol_mult,
    )

    # ---------------------- KPI row & Top Banner Injection -----------------
    last = df.iloc[-1]
    session_df = df[df.index.date == df.index[-1].date()]
    session_open = session_df["Open"].iloc[0]
    chg = last["Close"] - session_open
    pct_chg = (chg / session_open) * 100
    above_vwap = last["Close"] > last["VWAP"]

    # -----------------------------------------------------------------------
    # Injecting the data into the Top Price Banner we created earlier
    # -----------------------------------------------------------------------
    p_color = "#00e676" if chg >= 0 else "#ff1744"
    p_arrow = "▲" if chg >= 0 else "▼"
    
    top_price_banner.markdown(
        f"""
        <div style="text-align: center; padding: 15px; background-color: #161B26; 
                    border-radius: 10px; margin-bottom: 25px; 
                    border-left: 5px solid {p_color}; border-right: 5px solid {p_color};
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <span style="font-size: 42px; font-weight: bold; color: {p_color};">
                SPY ${last['Close']:.2f}
            </span>
            <span style="font-size: 24px; color: {p_color}; margin-left: 15px; font-weight: 500;">
                {p_arrow} {chg:+.2f} ({pct_chg:+.2f}%)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if compact:
        c1, c2, c3 = st.columns(3)
        c1.metric("SPY", f"${last['Close']:.2f}", f"{chg:+.2f}")
        c2.metric("VWAP", f"${last['VWAP']:.2f}", "Above ✅" if above_vwap else "Below ❌", delta_color="off")
        c3.metric("RSI", f"{last['RSI']:.1f}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Price", f"${last['Close']:.2f}", f"{chg:+.2f} vs open")
        c2.metric("VWAP", f"${last['VWAP']:.2f}", "Above ✅" if above_vwap else "Below ❌", delta_color="off")
        c3.metric(f"RSI ({rsi_period})", f"{last['RSI']:.1f}")
        c4.metric("Call signals", int(df["Call_Signal"].sum()))
        c5.metric("Put signals", int(df["Put_Signal"].sum()))

    # ---------------------- Market context row -----------------------------
    levels = {}
    if show_context:
        vix1d = fetch_vix1d()
        em = expected_move(session_open, vix1d) if vix1d else None
        if em:
            levels["EM high"] = session_open + em
            levels["EM low"] = session_open - em
        pd_lvls = prior_day_levels(df)
        if pd_lvls:
            levels.update(pd_lvls)
        rvol = relative_volume(df)

        now_et = pd.Timestamp.now(tz=EASTERN)
        close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        mins_left = int((close_dt - now_et).total_seconds() // 60)
        clock = (f"{mins_left // 60}h {mins_left % 60}m" if 0 < mins_left <= 390 else "Closed")

        if not compact:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("VIX1D", f"{vix1d:.1f}" if vix1d else "—", "1-day implied vol", delta_color="off")
            k2.metric("Expected move", f"±${em:.2f}" if em else "—", "VIX1D-implied", delta_color="off")
            k3.metric("Relative volume", f"{rvol:.2f}×" if rvol else "—", "running hot" if rvol and rvol > 1.1 else "normal", delta_color="off")
            k4.metric("Time to close", clock)

        if 0 < mins_left <= 60:
            st.info("🕒 **Final hour** — 0DTE theta decay is steepest now; long premium needs the move to happen immediately.")

    # Current-bar bias banner
    if last["Call_Signal"]:
        st.success("🟢 **IDEAL CALL scenario on the latest bar** — VWAP breakout with rising RSI.")
    elif last["Put_Signal"]:
        st.error("🔴 **IDEAL PUT scenario on the latest bar** — VWAP breakdown with falling RSI.")
    elif last.get("In_Chop_Zone", False):
        st.info("⏸️ Inside the midday chop zone — signals suppressed until 1:30 PM ET.")

    now_et = pd.Timestamp.now(tz=EASTERN)
    if now_et.weekday() < 5:
        for event_t in (dt.time(8, 30), dt.time(10, 0), dt.time(14, 0)):
            event_dt = now_et.replace(hour=event_t.hour, minute=event_t.minute, second=0, microsecond=0)
            if abs((now_et - event_dt).total_seconds()) <= 15 * 60:
                st.warning(f"⚠️ Within 15 min of the {event_t.strftime('%-I:%M %p')} ET window — technical signals may be unreliable.")
                break

    # ---------------------- ATM 0DTE contract quotes -----------------------
    if show_atm:
        now_et = pd.Timestamp.now(tz=EASTERN)
        if now_et.weekday() >= 5:
            st.caption("💤 ATM 0DTE quotes: contracts expiring today only exist on trading days.")
        else:
            spot = float(df["Close"].iloc[-1])
            call_q = fetch_atm_0dte_contract(spot, "C")
            put_q = fetch_atm_0dte_contract(spot, "P")

            def _quote_line(label: str, q: dict) -> str:
                if q.get("error"):
                    return f"**{label}** — unavailable ({q['error']})"
                bid = q.get("bid")
                ask = q.get("ask")
                if bid is None or ask is None:
                    return f"**{label} ${q['strike']}** — no live quote"
                liq = "🟩 liquid" if q.get("is_liquid") else "🟥 wide spread"
                return (f"**{label} ${q['strike']}** — bid **${bid:.2f}** / ask **${ask:.2f}** · spread ${q['spread']:.2f} {liq}")

            qc1, qc2 = st.columns(2)
            qc1.markdown(_quote_line("🟢 ATM Call", call_q))
            qc2.markdown(_quote_line("🔴 ATM Put", put_q))

    # ---------------------- Tabbed content ---------------------------------
    orh, orl = opening_range(df) if show_orb else (None, None)
    tab_chart, tab_outcomes, tab_news, tab_dict = st.tabs(["📊 Chart", "🎯 Outcomes", "📰 News", "📖 Dictionary"])

    with tab_chart:
        st.plotly_chart(
            build_chart(df, show_smas, shade_chop, orh, orl, levels=levels or None, height=chart_height, tf_label=tf_label),
            use_container_width=True,
        )

        st.subheader("📋 Signal Log")
        signal_rows = df[df["Call_Signal"] | df["Put_Signal"]].copy()
        if signal_rows.empty:
            st.write("No signals triggered under the current parameters.")
        else:
            log = pd.DataFrame({
                "Time (ET)": signal_rows.index.strftime("%Y-%m-%d %H:%M"),
                "Signal": np.where(signal_rows["Call_Signal"], "🟢 CALL", "🔴 PUT"),
                "Price": signal_rows["Close"].round(2),
                "VWAP": signal_rows["VWAP"].round(2),
                "RSI": signal_rows["RSI"].round(1),
                "Fast SMA": signal_rows["SMA_Fast"].round(2),
                "Slow SMA": signal_rows["SMA_Slow"].round(2),
            }).iloc[::-1]
            st.dataframe(log, use_container_width=True, hide_index=True)

        if not compact:
            with st.expander("🔍 Raw data (last 50 bars)"):
                st.dataframe(df.tail(50), use_container_width=True)

    with tab_outcomes:
        st.subheader("🎯 Signal Outcomes")
        outcomes = compute_signal_outcomes(df, tf_minutes)
        if outcomes.empty:
            st.write("No completed signals to evaluate yet.")
        else:
            st.dataframe(outcomes, use_container_width=True, hide_index=True)

    with tab_news:
        st.subheader("📰 SPY Market News")
        news = fetch_news_alpaca("SPY", limit=8)
        if not news:
            st.write("No headlines available.")
        else:
            for item in news:
                line = (f"**{item['time']}** — [{item['headline']}]({item['url']})" if item["url"] else f"**{item['time']}** — {item['headline']}")
                st.markdown(f"{line}  \n<span style='color:gray;font-size:0.8em'>{item['source']}</span>", unsafe_allow_html=True)

    with tab_dict:
        st.subheader("📖 Options Dictionary")
        query = st.text_input("Search terms", placeholder="e.g. theta, VWAP, chop", label_visibility="collapsed")
        q = query.strip().lower()
        matches = {term: (cat, definition) for term, (cat, definition) in GLOSSARY.items()
                   if not q or q in term.lower() or q in definition.lower() or q in cat.lower()}
        if matches:
            for cat in ["Indicators", "Greeks", "Options basics", "Execution & risk", "Market structure"]:
                cat_terms = {t: d for t, (c, d) in matches.items() if c == cat}
                if cat_terms:
                    st.markdown(f"**{cat}**")
                    for term in sorted(cat_terms):
                        with st.expander(term, expanded=bool(q)):
                            st.write(cat_terms[term])


if __name__ == "__main__":
    main()
