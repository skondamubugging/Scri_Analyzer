# Integration guide: wiring analysis_core / backtest_engine / universe_scanner into app.py

This does NOT require a full rewrite of app.py. Three surgical changes:

--------------------------------------------------------------------------------
## 1. Replace the in-file agent classes with an import

In app-2.py, delete the `TechnicalResult`, `SentimentResult`, `DecisionResult`
dataclasses and the `TechnicalDataAgent` / `SentimentAgent` / `DecisionMatrixAgent`
class bodies (they now live in analysis_core.py, byte-for-byte identical logic).
Replace with, near the top of app.py:

    from analysis_core import (
        TechnicalResult, SentimentResult, DecisionResult,
        TechnicalDataAgent, SentimentAgent, DecisionMatrixAgent,
        HIST_PERIOD, HIST_INTERVAL, MAX_WORKERS,
    )

Everything downstream (your `@st.cache_data`-wrapped `fetch_technical`,
`fetch_sentiment`, the batch runner, the UI rendering) keeps working exactly
as before -- you're only changing where the class bodies live, not their
behaviour. This is also what makes them testable now:

    # tests/test_analysis_core.py
    from analysis_core import TechnicalDataAgent, TechnicalResult
    import pandas as pd, numpy as np

    def test_minervini_fails_when_price_below_smas():
        ...

--------------------------------------------------------------------------------
## 2. Fix the two bugs flagged in review while you're in there

a) Cache invalidation on exchange switch -- change:

    if run_button or st.session_state.pipeline_results is None:

   to also invalidate when the exchange or ticker list changed since the
   last run, e.g.:

    current_inputs = (exchange, tuple(sorted(tickers)))
    if (run_button
            or st.session_state.pipeline_results is None
            or st.session_state.get("last_inputs") != current_inputs):
        st.session_state.last_inputs = current_inputs
        ... run pipeline ...

b) Stamp every result with its data timestamp in the UI -- since
   TechnicalResult.as_of now exists, show it next to each decision, e.g.:

    st.caption(f"Signal as of: {tech.as_of:%d %b %Y} (last confirmed daily close)")

   This alone fixes the "is this signal from today or is the market still
   open and this is stale" ambiguity called out in review.

--------------------------------------------------------------------------------
## 3. Add a new "Daily Shortlist" tab

    import universe_scanner
    from backtest_engine import summarize_events  # for showing historical context

    tab_manual, tab_shortlist = st.tabs(["🔍 Manual Watchlist", "🗓️ Daily Shortlist"])

    with tab_shortlist:
        st.caption(
            "Auto-scanned from the Nifty 500, filtered to liquid names "
            "(≥ ₹5 Cr avg. daily turnover). This is a ranked probability "
            "list, not a profit guarantee — see the historical scorecard "
            "below each signal type before sizing any position."
        )
        index_choice = st.selectbox("Universe", ["nifty500", "niftysmallcap250", "niftymidcap150"])
        min_turnover = st.slider("Min. avg daily turnover (₹ Cr)", 1.0, 50.0, 5.0)

        if st.button("Scan today's universe"):
            with st.spinner("Fetching universe + filtering for liquidity..."):
                tickers, liquidity_report = universe_scanner.get_daily_universe(
                    index=index_choice, min_avg_turnover_cr=min_turnover,
                )
            st.info(f"{len(tickers)} liquid names to scan out of "
                    f"{len(liquidity_report)} in the index.")
            with st.expander("Liquidity report (why a name was excluded)"):
                st.dataframe(liquidity_report)

            # reuse the EXACT same pipeline the manual tab uses
            results_df = run_batch(tickers, index_choice_exchange="NSE")  # your existing batch runner
            shortlist = results_df[results_df["Action"].isin(["STRONG BUY", "SPECULATIVE BUY"])]
            shortlist = shortlist.sort_values("Stage2Score", ascending=False)
            st.dataframe(style_action_table(shortlist))

        st.divider()
        st.subheader("Historical scorecard (technical signals only)")
        st.caption(
            "Backtested win-rate / avg return per signal type. Sentiment "
            "can't be backtested (Google News RSS has no historical "
            "archive) so this scorecard reflects the technical rules only "
            "— treat it as a floor, not the full picture. Regenerate "
            "weekly via `python backtest_engine.py` against a real "
            "universe, save to CSV, and load that CSV here rather than "
            "re-running the backtest on every page load (500 tickers x "
            "5 years is NOT something you want happening on a Streamlit "
            "button click)."
        )
        try:
            scorecard = pd.read_csv("backtest_results/latest_scorecard.csv")
            st.dataframe(scorecard)
        except FileNotFoundError:
            st.warning("No backtest scorecard found yet — run backtest_engine.py "
                       "against a real universe and save results to "
                       "backtest_results/latest_scorecard.csv.")

--------------------------------------------------------------------------------
## 4. Running the backtest for real (needs network + yfinance installed)

    from backtest_engine import run_universe_backtest, summarize_events
    from universe_scanner import get_daily_universe

    tickers, _ = get_daily_universe(index="nifty500", min_avg_turnover_cr=5.0)
    events = run_universe_backtest(tickers, exchange="NSE", period="5y", stride=3)
    scorecard = summarize_events(events)
    scorecard.to_csv("backtest_results/latest_scorecard.csv", index=False)

Run this as a scheduled weekly job (cron, GitHub Actions on a schedule
trigger, or Render/Railway cron), NOT inside the Streamlit request path —
a 500-ticker x 5-year backtest takes real minutes even with stride=3, and
blocking a UI click on it will time out / feel broken.
