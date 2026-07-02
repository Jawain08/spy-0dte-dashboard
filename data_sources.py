"""
data_sources.py — Real-time data ingestion via Alpaca Market Data API.

Drop-in replacement for the yfinance fetcher in app.py.

Setup
-----
1. Create a free account at https://alpaca.markets (no funding required).
2. Generate API keys from the dashboard.
3. Add them to Streamlit secrets:

   Local:  create `.streamlit/secrets.toml` in the project folder:
       ALPACA_API_KEY = "PK..."
       ALPACA_SECRET_KEY = "..."
   Cloud:  Streamlit Community Cloud → Manage app → Settings → Secrets,
           paste the same two lines. NEVER commit secrets.toml to GitHub
           (add `.streamlit/secrets.toml` to .gitignore).

4. Add `alpaca-py>=0.30.0` to requirements.txt.

5. In app.py, replace the import/call:
       from data_sources import fetch_intraday_data_alpaca as fetch_intraday_data

Feeds
-----
- feed="iex"  → FREE, truly real-time, but IEX exchange only (~2% of tape
                volume). Prices track SPY closely; VWAP uses IEX volume only.
- feed="sip"  → Full consolidated tape (100% of volume), requires the paid
                Algo Trader Plus subscription (~$99/mo). Exact tape VWAP.
"""

import datetime as dt
import pandas as pd
import streamlit as st
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# New imports for the 0DTE Option Engine
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest

EASTERN = "America/New_York"
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)


@st.cache_data(ttl=30, show_spinner="Fetching real-time SPY data…")
def fetch_intraday_data_alpaca(
    ticker: str = "SPY",
    lookback_days: int = 1,
    feed: str = "iex",  # "iex" = free real-time; "sip" = paid full tape
) -> pd.DataFrame:
    """
    Fetch 1-minute bars from Alpaca and return them in the exact schema
    app.py expects: ET-indexed OHLCV, regular trading hours only.
    """
    try:
        client = StockHistoricalDataClient(
            api_key=st.secrets["ALPACA_API_KEY"],
            secret_key=st.secrets["ALPACA_SECRET_KEY"],
        )
    except KeyError:
        st.error(
            "Alpaca API keys not found. Add ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY to your Streamlit secrets."
        )
        return pd.DataFrame()

    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, lookback_days))
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start.to_pydatetime(),
        feed=feed,  # critical: free accounts must request "iex" for live bars
    )

    try:
        bars = client.get_stock_bars(request)
        df = bars.df
    except Exception as exc:  # auth errors, rate limits, feed-permission errors
        st.error(f"Alpaca data request failed: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # alpaca-py returns a MultiIndex (symbol, timestamp) — drop the symbol level.
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")

    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )[["Open", "High", "Low", "Close", "Volume"]]

    # Alpaca timestamps are UTC → convert to ET, keep regular session only.
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN)
    df.index.name = "Timestamp"
    df = df.between_time(MARKET_OPEN, MARKET_CLOSE)
    df = df.dropna(subset=["Close", "Volume"])
    return df


@st.cache_data(ttl=120, show_spinner=False)
def fetch_news_alpaca(symbol: str = "SPY", limit: int = 8) -> list:
    """
    Latest Benzinga-sourced headlines for `symbol` via Alpaca's free News API.
    Returns a list of dicts: {time, headline, source, url}. Empty list on any
    failure so the dashboard never breaks because of the news panel.
    """
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(
            api_key=st.secrets["ALPACA_API_KEY"],
            secret_key=st.secrets["ALPACA_SECRET_KEY"],
        )
        result = client.get_news(NewsRequest(symbols=symbol, limit=limit))
        articles = getattr(result, "data", {}).get("news", []) or getattr(
            result, "news", []
        )
        items = []
        for a in articles:
            created = pd.Timestamp(getattr(a, "created_at")).tz_convert(EASTERN)
            items.append(
                {
                    "time": created.strftime("%b %d, %H:%M ET"),
                    "headline": getattr(a, "headline", ""),
                    "source": getattr(a, "source", "benzinga"),
                    "url": getattr(a, "url", "") or "",
                }
            )
        return items
    except Exception:
        return []


@st.cache_data(ttl=2, show_spinner=False)
def fetch_atm_0dte_contract(
    underlying_price: float, 
    option_type: str = "C",
    ticker: str = "SPY"
) -> dict:
    """
    Calculates the closest At-The-Money strike (0.50 Delta proxy) for today's 0DTE expiration,
    constructs the OCC format string, and queries Alpaca for the live bid, ask, and spread.
    
    Parameters:
    -----------
    underlying_price : float
        The most recent close/last price of the underlying asset (e.g., SPY).
    option_type : str
        'C' for Calls (breakout strategy), 'P' for Puts (breakdown strategy).
    ticker : str
        The underlying asset symbol. Defaults to 'SPY'.
    """
    try:
        options_client = OptionHistoricalDataClient(
            api_key=st.secrets["ALPACA_API_KEY"],
            secret_key=st.secrets["ALPACA_SECRET_KEY"],
        )
    except KeyError:
        return {"error": "Alpaca keys missing from Streamlit secrets."}

    try:
        # 1. Format today's date structure for standard OCC formatting (YYMMDD)
        today_str = dt.datetime.now(tz=pd.Timestamp.now(tz=EASTERN).tz).strftime("%y%m%d")
        
        # 2. Derive the closest ATM strike price (SPY trades in $1 increments)
        atm_strike = round(underlying_price)
        
        # 3. Format strike to explicit 8-character OCC specification (e.g., 545 -> 00545000)
        strike_formatted = f"{int(atm_strike * 1000):08d}"
        
        # 4. Assemble standard OCC option symbol string
        occ_symbol = f"{ticker.ljust(6)}{today_str}{option_type}{strike_formatted}".replace(" ", "")
        
        # 5. Execute snapshot request for the target contract
        request = OptionSnapshotRequest(symbol_or_symbols=occ_symbol)
        snapshot = options_client.get_option_snapshot(request)
        
        if not snapshot or occ_symbol not in snapshot:
            return {"error": f"No active data feed found for contract {occ_symbol}"}
            
        contract_data = snapshot[occ_symbol]
        
        # Verify active quote packet exists
        if contract_data.latest_quote is None:
            return {"error": f"Contract {occ_symbol} has no live quote details."}
            
        bid = contract_data.latest_quote.bid_price
        ask = contract_data.latest_quote.ask_price
        spread = ask - bid
        
        return {
            "symbol": occ_symbol,
            "strike": atm_strike,
            "bid": bid,
            "ask": ask,
            "spread": round(spread, 2),
            "is_liquid": spread <= 0.05,  # Liquid threshold check
            "error": None
        }
        
    except Exception as err:
        return {"error": f"Option retrieval failure: {str(err)}"}
