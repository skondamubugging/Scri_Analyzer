"""
================================================================================
 universe_scanner.py
================================================================================
Replaces "manually type in tickers" with a daily auto-scanned universe:

    1. fetch_nse_index_constituents() -- pulls the current Nifty 500 /
       Smallcap 250 constituent list from NSE's public archives.
    2. compute_liquidity() -- filters that list down to names that are
       actually tradable in size (min. average daily turnover), so the
       scanner never surfaces a "burst" that was really just a wide print
       on a stock nobody could have filled.
    3. get_daily_universe() -- orchestrates both and hands back a ready-to
       -scan ticker list for the pipeline in app.py.

NSE'S ANTI-SCRAPING BEHAVIOUR (read before deploying)
-------------------------------------------------------
archives.nseindia.com and nseindia.com require a "warm-up" request against
the homepage first to receive cookies + a realistic browser User-Agent
before the CSV endpoint will respond -- hitting the CSV URL cold usually
gets you a 401/403. This module does that warm-up, but NSE changes this
behaviour periodically without notice. If the fetch starts failing in
production:
    - the function degrades to FALLBACK_LIQUID_UNIVERSE (a small hardcoded
      list of large, highly liquid names) rather than crashing the app,
      exactly like the news-sentiment agent degrades to "Neutral" -- but
      you should surface that degradation in the UI (see app.py
      integration note) rather than let it fail silently, since a stale
      fallback list quietly replacing the real universe is a worse bug
      than an outright error.
    - consider caching the fetched list to disk once a day rather than
      re-fetching on every scan, both to reduce load on NSE and reduce your
      exposure to transient failures.
================================================================================
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

import pandas as pd

NSE_INDEX_CSV_URLS = {
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "niftysmallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "niftymidcap150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
}

# Fallback used ONLY if the live NSE fetch fails -- deliberately small and
# large-cap/liquid so a degraded run never recommends something unfillable.
FALLBACK_LIQUID_UNIVERSE: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "ITC", "LT", "KOTAKBANK", "SBIN", "HINDUNILVR", "AXISBANK",
    "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN", "ASIANPAINT",
    "ULTRACEMCO", "NESTLEIND", "WIPRO", "ADANIENT", "ADANIPORTS",
    "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "HCLTECH",
    "M&M", "BAJAJFINSV", "ONGC", "COALINDIA", "JSWSTEEL", "GRASIM",
    "TECHM", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BPCL",
    "DIVISLAB", "BRITANNIA", "APOLLOHOSP", "INDUSINDBK", "SBILIFE",
    "HDFCLIFE", "TATACONSUM", "UPL", "SHREECEM", "HINDALCO", "VEDL",
]


def fetch_nse_index_constituents(index: str = "nifty500") -> List[str]:
    """Returns the list of NSE trading symbols for the given index. Falls
    back to FALLBACK_LIQUID_UNIVERSE (with a printed warning) if the live
    fetch fails for any reason -- network, NSE anti-bot changes, schema
    change in the CSV, etc."""
    url = NSE_INDEX_CSV_URLS.get(index.lower())
    if url is None:
        raise ValueError(f"Unknown index '{index}'. Choose from {list(NSE_INDEX_CSV_URLS)}.")

    try:
        import requests  # lazy import, see analysis_core.py for rationale
        import io

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/csv,application/csv,*/*",
        }
        session = requests.Session()
        session.headers.update(headers)
        # warm-up request: NSE only issues valid cookies after a homepage hit
        session.get("https://www.nseindia.com", timeout=6)
        resp = session.get(url, timeout=8)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        symbol_col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
        if symbol_col is None:
            raise ValueError(f"Unexpected CSV schema, columns were: {list(df.columns)}")

        symbols = df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
        if len(symbols) < 50:
            raise ValueError(f"Suspiciously few symbols returned ({len(symbols)}) -- treating as a failed fetch.")
        return symbols

    except Exception as exc:
        print(f"[universe_scanner] Live NSE fetch failed ({exc}); "
              f"falling back to a small hardcoded liquid-universe list. "
              f"Signals below are only as broad as this fallback list.")
        return list(FALLBACK_LIQUID_UNIVERSE)


def _default_turnover_fetcher(ticker_yf: str, lookback_days: int) -> Optional[float]:
    """Real data path: pulls recent daily bars via yfinance and returns the
    average (Close * Volume) in INR crore. Returns None on any failure so
    callers can filter it out rather than crash the batch."""
    try:
        import yfinance as yf
        df = yf.download(ticker_yf, period=f"{lookback_days + 10}d", interval="1d",
                          progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        window = df.dropna(subset=["Close", "Volume"]).tail(lookback_days)
        if window.empty:
            return None
        turnover_inr = (window["Close"] * window["Volume"]).mean()
        return float(turnover_inr / 1e7)  # INR -> crore
    except Exception:
        return None


def compute_liquidity(
    tickers: List[str],
    exchange: str = "NSE",
    lookback_days: int = 20,
    min_avg_turnover_cr: float = 5.0,
    max_workers: int = 8,
    turnover_fetcher: Callable[[str, int], Optional[float]] = _default_turnover_fetcher,
) -> pd.DataFrame:
    """Filters `tickers` down to those with at least `min_avg_turnover_cr`
    crore of average daily traded value over `lookback_days`. Returns a
    DataFrame with every ticker's measured turnover and a pass/fail flag,
    so failures are visible rather than just silently dropped.

    `turnover_fetcher` is injectable so this can be unit-tested offline
    without hitting yfinance -- pass a stub that returns synthetic values.
    """
    suffix = {"NSE": ".NS", "BSE": ".BO"}.get(exchange, ".NS")
    rows = []

    def _job(raw_ticker: str):
        ticker_yf = raw_ticker if raw_ticker.endswith((".NS", ".BO")) else f"{raw_ticker}{suffix}"
        turnover = turnover_fetcher(ticker_yf, lookback_days)
        return raw_ticker, turnover

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_job, t): t for t in tickers}
        for fut in as_completed(futures):
            raw_ticker, turnover = fut.result()
            rows.append({
                "ticker": raw_ticker,
                "avg_turnover_20d_cr": turnover,
                "liquid_enough": bool(turnover is not None and turnover >= min_avg_turnover_cr),
            })

    return pd.DataFrame(rows).sort_values("avg_turnover_20d_cr", ascending=False, na_position="last").reset_index(drop=True)


def get_daily_universe(
    index: str = "nifty500",
    exchange: str = "NSE",
    min_avg_turnover_cr: float = 5.0,
    lookback_days: int = 20,
) -> tuple[List[str], pd.DataFrame]:
    """One-call orchestration: fetch the index constituent list, apply the
    liquidity filter, return (final_ticker_list, full_liquidity_report).
    Keep the full report around in the UI/logs -- it's what lets you
    explain to a user *why* a stock they expected to see isn't on today's
    shortlist (illiquid, not just 'failed the technicals')."""
    raw_universe = fetch_nse_index_constituents(index=index)
    liquidity_df = compute_liquidity(
        raw_universe, exchange=exchange, lookback_days=lookback_days, min_avg_turnover_cr=min_avg_turnover_cr,
    )
    final_list = liquidity_df.loc[liquidity_df["liquid_enough"], "ticker"].tolist()
    return final_list, liquidity_df


# ==============================================================================
# OFFLINE SELF-TEST (no network required -- uses an injected fake fetcher)
# ==============================================================================

if __name__ == "__main__":
    import random

    print("Running OFFLINE self-test of compute_liquidity() with a fake turnover fetcher...\n")

    fake_universe = [f"FAKE{i:03d}" for i in range(30)]

    def _fake_fetcher(ticker_yf: str, lookback_days: int) -> Optional[float]:
        rng = random.Random(ticker_yf)
        if rng.random() < 0.1:
            return None  # simulate an occasional data-fetch failure
        return rng.uniform(0.5, 40.0)  # crore

    report = compute_liquidity(fake_universe, min_avg_turnover_cr=5.0, turnover_fetcher=_fake_fetcher)
    print(report.to_string(index=False))
    print(f"\n{report['liquid_enough'].sum()} / {len(report)} names cleared the liquidity bar.")
