# 📈 Jawain's Trading Strategy — 0DTE SPY Dashboard

A Streamlit app that pulls **real-time 1-minute SPY bars via Alpaca** (free
IEX feed, or the paid full-tape SIP feed), computes session-anchored **VWAP**,
**RSI**, and **fast/slow SMAs**, and flags rules-based **Ideal Call / Ideal
Put** scenarios:

- **CALL** — price breaks *above* VWAP while RSI is *rising* inside the 50–65 band
- **PUT** — price breaks *below* VWAP while RSI is *falling* inside the 35–50 band
- **Time filter** — signals suppressed during the midday chop zone (11:30 AM – 1:30 PM ET)

A Yahoo Finance fallback (15-min delayed, no keys needed) is selectable from
the sidebar, and the app auto-falls-back to it if Alpaca keys are missing.

> ⚠️ **Disclaimer:** Educational tool only. Not financial advice. 0DTE options
> carry extreme risk, including total loss of premium.

---

## 🔑 Get Alpaca API Keys (free)

1. Sign up at [alpaca.markets](https://alpaca.markets) — no funding required.
2. In the dashboard, generate an **API Key ID** and **Secret Key**.
3. These keys are data/paper keys with no money behind them — but still never
   commit them to GitHub.

## 🖥️ Run Locally

```bash
cd spy-0dte-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Add your keys:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   ...then edit .streamlit/secrets.toml with your real keys

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🚀 Deploy to Streamlit Community Cloud (free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Jawain's Trading Strategy: 0DTE SPY dashboard"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/spy-0dte-dashboard.git
git push -u origin main
```

The included `.gitignore` prevents `secrets.toml` from ever being committed —
verify with `git status` that it does **not** appear before pushing.

### Step 2 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Repository: `<YOUR_USERNAME>/spy-0dte-dashboard` · Branch: `main` · Main file: `app.py`
4. Click **Deploy**.

### Step 3 — Add your Alpaca keys to the cloud app

1. Open **Manage app → Settings → Secrets** (bottom-right panel of your deployed app).
2. Paste:
   ```
   ALPACA_API_KEY = "PK..."
   ALPACA_SECRET_KEY = "..."
   ```
3. Save — the app reboots with real-time data enabled.

### Updating the live app

Every `git push` to `main` auto-redeploys.

---

## 🧠 Strategy Logic Reference

| Component | Rule |
|---|---|
| VWAP | Session-anchored (resets each trading day at 9:30 ET) |
| Call trigger | Prev close ≤ VWAP → close > VWAP, RSI rising, RSI ∈ [50, 65] |
| Put trigger | Prev close ≥ VWAP → close < VWAP, RSI falling, RSI ∈ [35, 50] |
| Optional filter | Fast SMA > Slow SMA for calls (inverse for puts) |
| Chop zone | 11:30 AM – 1:30 PM ET — all signals blocked |


## 🆕 Signal-quality & news features

| Feature | What it does |
|---|---|
| Volume confirmation | Optional filter: signal bar volume must exceed N× the 20-bar average — screens out thin, failure-prone VWAP breaks |
| Opening range | First-30-min high/low drawn on the chart for confluence with VWAP signals |
| Signal outcomes | Measures SPY's move 5/10/15 min after every historical signal — favorable %, average move. Proxy for signal quality, NOT option P&L |
| SPY news panel | Live Benzinga headlines via Alpaca's free News API (same keys), ~2-min refresh |
| Event-window caution | Warns when within ±15 min of common release slots (8:30 / 10:00 / 2:00 ET) |

## 📡 Data Feeds

| Feed | Cost | Latency | Coverage |
|---|---|---|---|
| Alpaca IEX | Free | Real-time | IEX exchange only (~2% of tape; fine for SPY signals) |
| Alpaca SIP | ~$99/mo (Algo Trader Plus) | Real-time | 100% consolidated tape |
| Yahoo Finance | Free, no keys | ~15 min delayed | Consolidated |

### Troubleshooting

| Symptom | Fix |
|---|---|
| "Alpaca API keys not found" | Add both keys to secrets (local file or Cloud Secrets panel). |
| `subscription does not permit querying recent SIP data` | You selected the SIP feed without Algo Trader Plus — switch to IEX. |
| No data, market closed | Raise the lookback slider to load prior sessions. |
| Blank chart after deploy | Check app logs (Manage app) for a missing package; confirm requirements.txt was committed. |
