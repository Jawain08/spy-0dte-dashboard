"""
glossary.py — Options & day-trading dictionary for the dashboard.

Each entry: term -> (category, definition written for a 0DTE SPY trader).
Rendered by the Dictionary tab in app.py with live search.
"""

GLOSSARY = {
    # ---- Indicators on this dashboard ------------------------------------
    "VWAP": ("Indicators", "Volume Weighted Average Price — the average price paid today, weighted by volume, resetting at the 9:30 open. Institutions benchmark fills against it, so price above VWAP = buyers in control, below = sellers. This app's core signal line: calls trigger on breaks above, puts on breaks below."),
    "RSI": ("Indicators", "Relative Strength Index — momentum oscillator from 0–100 measuring the speed of recent gains vs. losses. Above 70 = overbought, below 30 = oversold. This app uses the middle bands (rising 50–65 for calls, falling 35–50 for puts) to catch trends early rather than fading extremes."),
    "SMA": ("Indicators", "Simple Moving Average — the plain average of the last N closes. The fast/slow pair here (9/21 default) defines short-term trend: fast above slow = uptrend bias. Every bar counts equally, so it lags fast markets more than an EMA."),
    "EMA": ("Indicators", "Exponential Moving Average — like an SMA but recent bars count more, so it turns faster. Day traders often prefer the 9 EMA for momentum. This app uses SMAs by default; swapping to EMAs is a one-line change if you want quicker (but noisier) trend reads."),
    "Opening range": ("Indicators", "The high and low of the first 30 minutes (9:30–10:00 ET). The day's first battle lines — breakouts above/below them are classic 0DTE entries, and this app draws both on the chart for confluence with VWAP signals."),
    "Expected move": ("Indicators", "The market's own estimate of how far SPY should travel today, derived from option prices (here via VIX1D). If price is already at the expected-move edge, further extension is a fight against what's priced in — a poor spot to buy a breakout."),
    "Relative volume": ("Indicators", "Today's volume vs. the average at the same time of day. Above ~1.2× = active participation that supports trends; below ~0.9× = a sleepy tape where breakouts tend to fail. Shown in the context row."),
    "MACD": ("Indicators", "Moving Average Convergence Divergence — the gap between a fast and slow EMA plus a signal line; crossovers flag momentum shifts. Not on this dashboard, but a common companion to RSI."),

    # ---- The Greeks --------------------------------------------------------
    "Delta": ("Greeks", "How much the option's price moves per $1 move in SPY. A 0.50-delta call gains ~$0.50 if SPY rises $1. Also a rough probability of expiring in the money. Higher delta = more stock-like, more expensive; low delta = cheap lottery tickets."),
    "Theta": ("Greeks", "Time decay — how much value the option loses per day just from the clock. THE defining force of 0DTE: your entire time value evaporates by 4 PM, fastest in the final hours. Long premium is a race between the move (delta) and the melt (theta)."),
    "Gamma": ("Greeks", "How fast delta itself changes as SPY moves. 0DTE at-the-money options have enormous gamma — small SPY moves cause explosive option-price swings in both directions. It's why 0DTE wins feel huge and losses arrive just as fast."),
    "Vega": ("Greeks", "Sensitivity to implied volatility. If IV drops after an event (see IV crush), the option loses value even if SPY doesn't move. Small in dollar terms on 0DTE but brutal in percentage terms around scheduled releases."),
    "IV": ("Greeks", "Implied volatility — the market's forecast of movement, baked into the option's price. High IV = expensive options that need bigger moves to profit. VIX1D on the context row is essentially SPX's 1-day IV."),
    "IV crush": ("Greeks", "The collapse in implied volatility right after a known event (CPI, FOMC). Options bought before the event can lose value even when the trader guessed direction correctly, because the uncertainty premium deflates instantly."),

    # ---- Options basics ----------------------------------------------------
    "0DTE": ("Options basics", "Zero Days To Expiration — options expiring today. Maximum gamma, maximum theta: the cheapest way to bet on today's move and the fastest way to lose the entire premium. Most 0DTE buyers lose over time; discipline and selectivity are the whole game."),
    "Call": ("Options basics", "The right to buy 100 shares at the strike price until expiration. Bought when you expect price to rise. This app's green arrows flag conditions that historically precede upward moves."),
    "Put": ("Options basics", "The right to sell 100 shares at the strike price until expiration. Bought when you expect price to fall. Red arrows on the chart."),
    "Strike": ("Options basics", "The price at which the option converts to shares. Distance from SPY's current price determines cost and odds: near-the-money = expensive but responsive; far out-of-the-money = cheap and usually worthless at the close."),
    "Premium": ("Options basics", "The price you pay for the option. For a 0DTE buyer this is the maximum possible loss — and by 4 PM, out-of-the-money premium goes to exactly zero."),
    "ATM": ("Options basics", "At the money — strike ≈ current price. Highest gamma and theta, ~0.50 delta. The standard 0DTE momentum vehicle."),
    "ITM": ("Options basics", "In the money — a call below / put above the current price. Has real intrinsic value, higher delta, less theta risk, but costs more."),
    "OTM": ("Options basics", "Out of the money — no intrinsic value, only time value. All of an OTM option's price melts to zero at expiry unless price crosses the strike. The cheap 0DTE strikes everyone buys and mostly loses on."),
    "Intrinsic value": ("Options basics", "The 'real' portion of an option's price: how far it's in the money. Everything above that is time value — the part theta destroys."),
    "Exercise / assignment": ("Options basics", "Converting the option into shares (exercise) or being forced to deliver (assignment). Rare concern for 0DTE traders who close positions before the bell — but ITM options held through expiration auto-exercise, which can surprise small accounts."),
    "Open interest": ("Options basics", "Total contracts outstanding at a strike. High-OI strikes act like magnets/pins near expiry as dealers hedge. Not shown in this app (needs paid options data)."),

    # ---- Execution & risk ---------------------------------------------------
    "Bid-ask spread": ("Execution & risk", "The gap between what buyers pay and sellers ask. You lose the spread on every round trip — on a $0.50 option a $0.05 spread is 10% gone instantly. Trade liquid strikes; SPY 0DTE near the money is among the tightest anywhere."),
    "Slippage": ("Execution & risk", "The difference between the price you expected and the fill you got. Grows with fast markets and market orders — one reason a 'favorable' signal in the outcomes table doesn't map 1:1 to option profit."),
    "Stop loss": ("Execution & risk", "A predefined exit if the trade goes against you. On 0DTE, mental stops fail fast — decide the max loss before entry. Note this dashboard signals entries only; exits are entirely your process."),
    "Position sizing": ("Execution & risk", "How much you risk per trade. The most common account-killer isn't bad signals — it's oversized ones. Many day traders cap risk at 1–2% of account per trade; with 0DTE, assume the full premium can vanish."),
    "Risk/reward": ("Execution & risk", "Expected gain vs. accepted loss on a trade. A 60% win rate with 1:2 risk/reward loses money; a 40% win rate at 3:1 prints. Read the outcomes table's Favorable % and Avg move together for exactly this reason."),
    "Scalping": ("Execution & risk", "Very short holds capturing small moves. If the outcomes table shows edge at +5 min that fades by +15, your signals are scalps — hold accordingly."),
    "Paper trading": ("Execution & risk", "Simulated trading with fake money. The zero-cost way to test this dashboard's signals against your execution before risking premium."),

    # ---- Market structure & context ----------------------------------------
    "Chop": ("Market structure", "Directionless back-and-forth price action that stops out both bulls and bears. Worst environment for trend-following. This app blocks signals during the classic 11:30–1:30 lunch chop by default."),
    "Breakout": ("Market structure", "Price escaping a defined level or range (VWAP, opening range, prior-day high). Real breakouts come with volume — hence the volume-confirmation filter."),
    "Support / resistance": ("Market structure", "Price zones where buying (support) or selling (resistance) has repeatedly appeared. Prior-day high/low/close and the opening range on this chart are the intraday versions."),
    "Trend day": ("Market structure", "A session that opens, picks a direction, and never looks back — usually on hot relative volume. Trend-following signals shine here and struggle everywhere else."),
    "VIX": ("Market structure", "The Cboe Volatility Index — 30-day implied volatility on SPX, the market's 'fear gauge.' Cousin of the VIX1D shown on the context row."),
    "VIX1D": ("Market structure", "Cboe's 1-Day Volatility Index — implied volatility for just today, built largely from 0DTE option prices. The purest single number for 'how wild should today be,' and the source of this app's expected-move band."),
    "FOMC": ("Market structure", "The Fed's rate-setting committee. Statement days (2:00 PM ET) produce violent SPY whipsaws that shred technical signals — the reason for this app's 2:00 PM event-window warning."),
    "CPI": ("Market structure", "Consumer Price Index — the monthly inflation print, released 8:30 AM ET. One of the biggest scheduled movers of SPY; 0DTE straddle prices double on CPI mornings."),
    "SIP / IEX feeds": ("Market structure", "SIP = the consolidated tape of all US exchanges (100% of volume); IEX = one exchange (~2%). This app's free Alpaca feed is real-time IEX — prices track closely on SPY, but volume-based stats are IEX-only unless you upgrade."),
    "Liquidity": ("Market structure", "How easily you can enter/exit near the quoted price. SPY is the most liquid equity product in the world — a key reason 0DTE traders concentrate there."),
}
