"""
================================================================================
 backtest_engine.py
================================================================================
Walk-forward backtester for the signals produced by analysis_core.py
(Minervini Stage 2 pass, Bonde Momentum Burst, Stage 2 Entry Screener).

WHY THIS EXISTS
----------------
The live app can tell you a stock passes a rule TODAY. It cannot tell you
whether stocks that passed that same rule in the past actually went on to
make money, how often, by how much, or how much pain (drawdown) you'd have
sat through first. This module answers that, honestly, using only
information that would have been available on each historical day (no
lookahead).

IMPORTANT HONESTY NOTE ON SENTIMENT
------------------------------------
Google News RSS only returns CURRENT headlines -- there is no way to
reconstruct what the news sentiment score would have been on, say,
14 March 2023 for a given ticker without a paid historical news archive.
So this backtester tests a "Stage 2 Entry -- technical only" variant that
applies every check EXCEPT `sentiment_not_bearish`. Treat the live app's
sentiment gate as an extra filter layered on top of what's validated here,
not as something whose historical edge has been proven.

WHAT THIS DOES NOT DO
----------------------
- Does not model slippage, brokerage, STT, or the fact that a % move on a
  candle isn't the price you'd actually get filled at.
- Does not account for NSE circuit filters (a stock frozen at upper circuit
  can show a "burst" you literally could not have bought).
- Does not de-duplicate overlapping signals on the same stock (e.g. a stock
  that stays "Stage 2 Entry Ready" for 10 straight days produces 10 events,
  not 1) -- results are per SIGNAL DAY, not per unique trade. Treat the
  win-rate as "if you'd bought on every qualifying day" not "expected trades
  per year."
- Backtests one ticker's history against ONLY that ticker's own indicators
  -- it does not risk-adjust vs. Nifty, so results can look artificially
  good/bad depending on the broad market regime over the test window.

Use this to sanity-check whether a rule has ANY historical edge before
trusting it, not as a promise of forward returns.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis_core import (
    TechnicalDataAgent,
    TechnicalResult,
    SentimentResult,
    EXCHANGE_SUFFIX,
)

DEFAULT_HORIZONS = (5, 10, 20)   # trading days forward
MIN_LOOKBACK_BARS = 320          # bars needed before a signal can even be evaluated
                                  # (200-SMA + 20-bar trend check + buffer)


@dataclass
class SignalEvent:
    ticker: str
    signal: str                  # "momentum_burst" | "stage2_technical"
    date: pd.Timestamp
    entry_price: float
    fwd_return: Dict[int, float] = field(default_factory=dict)   # horizon -> % return
    mae: Dict[int, float] = field(default_factory=dict)          # horizon -> max adverse excursion %


def _neutral_sentiment() -> SentimentResult:
    """Stand-in for sentiment we cannot reconstruct historically. Labeling
    it Neutral (rather than Bullish) keeps the `stage2_entry_ready` gate's
    sentiment check trivially passing without pretending we know the news."""
    return SentimentResult(label="Neutral", score=0.0, n_headlines=0)


def _stage2_technical_gate(tech: TechnicalResult) -> bool:
    """Reproduces TechnicalDataAgent.run_stage2_entry_screen's core gate but
    EXCLUDING the sentiment check, since historical sentiment can't be
    reconstructed. See module docstring."""
    c = tech.stage2_entry_checks
    if not c:
        return False
    return bool(
        c.get("minervini_stage2")
        and c.get("macd_bullish")
        and c.get("rsi_healthy_zone")
        and c.get("not_overbought")
        and c.get("not_already_extended")
        and c.get("no_active_exit_signal")
        # sentiment_not_bearish deliberately excluded
    )


def _evaluate_window(window: pd.DataFrame) -> TechnicalResult:
    """Runs the exact same indicator/rule pipeline the live app uses, on a
    bounded trailing window ending at the day being evaluated -- this is
    what the live app would have seen if it had been run on that day."""
    result = TechnicalResult(ticker_display="BT", ticker_yf="BT")
    result.ok = True
    result.as_of = window.index[-1]
    df = window.copy()
    TechnicalDataAgent.compute_all(result, df)
    TechnicalDataAgent.run_stage2_entry_screen(result, _neutral_sentiment())
    return result


def backtest_dataframe(
    df: pd.DataFrame,
    ticker_label: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_lookback: int = MIN_LOOKBACK_BARS,
    stride: int = 1,
) -> List[SignalEvent]:
    """Walks forward through `df` (ascending-date OHLCV with columns
    Open/High/Low/Close/Volume) one day at a time, evaluates signals using
    only data up to and including that day, and records forward returns.

    `stride` > 1 skips days between evaluations (e.g. stride=5 -> weekly
    evaluation) to speed up large universe backtests at the cost of
    resolution -- use stride=1 for a handful of tickers, stride=5+ for a
    500-stock universe.
    """
    events: List[SignalEvent] = []
    n = len(df)
    max_h = max(horizons)

    last_evaluable = n - max_h - 1
    if last_evaluable <= min_lookback:
        return events  # not enough history to backtest at all

    for t in range(min_lookback, last_evaluable, stride):
        window = df.iloc[max(0, t - min_lookback): t + 1]
        try:
            tech = _evaluate_window(window)
        except Exception:
            continue  # skip malformed windows rather than crash the whole run

        entry_date = df.index[t]
        entry_price = float(df["Close"].iloc[t])
        if entry_price <= 0:
            continue

        fired = []
        if tech.momentum_burst:
            fired.append("momentum_burst")
        if _stage2_technical_gate(tech):
            fired.append("stage2_technical")

        if not fired:
            continue

        future_slice = df.iloc[t: t + max_h + 1]
        fwd_return: Dict[int, float] = {}
        mae: Dict[int, float] = {}
        for h in horizons:
            if len(future_slice) <= h:
                continue
            path = future_slice["Close"].iloc[: h + 1]
            fwd_return[h] = float((path.iloc[-1] / entry_price - 1.0) * 100.0)
            mae[h] = float((path.min() / entry_price - 1.0) * 100.0)  # worst dip, negative %

        for signal_name in fired:
            events.append(SignalEvent(
                ticker=ticker_label, signal=signal_name, date=entry_date,
                entry_price=entry_price, fwd_return=fwd_return, mae=mae,
            ))

    return events


def backtest_ticker(
    raw_ticker: str,
    exchange: str = "NSE",
    period: str = "5y",
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    stride: int = 1,
) -> List[SignalEvent]:
    """Thin wrapper: downloads history via yfinance, then backtests it.
    Requires network access (not used in the offline __main__ demo below)."""
    import yfinance as yf

    ticker_yf = TechnicalDataAgent.build_yf_ticker(raw_ticker, exchange)
    df = yf.download(ticker_yf, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close", "Volume"]).copy()
    return backtest_dataframe(df, raw_ticker.upper(), horizons=horizons, stride=stride)


def run_universe_backtest(
    tickers: Sequence[str],
    exchange: str = "NSE",
    period: str = "5y",
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    stride: int = 3,
) -> List[SignalEvent]:
    """Backtests a whole ticker universe. `stride=3` by default to keep a
    500-stock universe backtest tractable; drop to 1 for a small watchlist
    where you want maximum event resolution."""
    all_events: List[SignalEvent] = []
    for raw_ticker in tickers:
        try:
            all_events.extend(
                backtest_ticker(raw_ticker, exchange=exchange, period=period, horizons=horizons, stride=stride)
            )
        except Exception as exc:
            print(f"[backtest] skipped {raw_ticker}: {exc}")
    return all_events


def summarize_events(events: List[SignalEvent], horizons: Sequence[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Aggregates raw signal events into the honest scorecard: per signal
    type and horizon, sample size, win rate, average / median return, and
    average worst-case drawdown before that horizon."""
    rows = []
    for signal_name in sorted(set(e.signal for e in events)):
        sig_events = [e for e in events if e.signal == signal_name]
        for h in horizons:
            rets = [e.fwd_return[h] for e in sig_events if h in e.fwd_return]
            maes = [e.mae[h] for e in sig_events if h in e.mae]
            if not rets:
                continue
            rets_arr = np.array(rets)
            rows.append({
                "signal": signal_name,
                "horizon_days": h,
                "n_signals": len(rets_arr),
                "win_rate_%": round(float((rets_arr > 0).mean() * 100), 1),
                "avg_return_%": round(float(rets_arr.mean()), 2),
                "median_return_%": round(float(np.median(rets_arr)), 2),
                "avg_worst_drawdown_%": round(float(np.mean(maes)), 2) if maes else np.nan,
                "best_%": round(float(rets_arr.max()), 2),
                "worst_%": round(float(rets_arr.min()), 2),
            })
    return pd.DataFrame(rows).sort_values(["signal", "horizon_days"]).reset_index(drop=True)


# ==============================================================================
# OFFLINE DEMO / SELF-TEST (no network required)
# ==============================================================================

def _generate_synthetic_ohlcv(n_days: int = 900, seed: int = 42) -> pd.DataFrame:
    """Generates a synthetic price series with randomly injected 'burst'
    days (large up-move + volume spike after a tight consolidation) so the
    pipeline can be exercised end-to-end without network access. This is
    ONLY for verifying the backtester's plumbing -- it says nothing about
    real market behaviour."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)

    price = 100.0
    closes, highs, lows, opens, vols = [], [], [], [], []
    base_vol = 500_000
    consolidating = False
    consolidation_days = 0

    for i in range(n_days):
        # occasionally enter a tight consolidation, then break out of it
        if not consolidating and rng.random() < 0.02:
            consolidating = True
            consolidation_days = 0

        if consolidating:
            consolidation_days += 1
            drift = rng.normal(0.0003, 0.004)  # tight range
            vol_mult = rng.uniform(0.7, 1.1)
            if consolidation_days >= 5 and rng.random() < 0.25:
                drift = rng.uniform(0.045, 0.07)  # burst breakout
                vol_mult = rng.uniform(2.0, 3.5)
                consolidating = False
        else:
            drift = rng.normal(0.0005, 0.018)
            vol_mult = rng.uniform(0.6, 1.6)

        prev_close = price
        price = max(1.0, price * (1 + drift))
        day_vol = abs(rng.normal(0.01, 0.006))
        o = prev_close * (1 + rng.normal(0, 0.003))
        h = max(o, price) * (1 + day_vol)
        l = min(o, price) * (1 - day_vol)
        v = max(1000, base_vol * vol_mult * rng.uniform(0.8, 1.2))

        opens.append(o); highs.append(h); lows.append(l); closes.append(price); vols.append(v)

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols,
    }, index=dates)


if __name__ == "__main__":
    print("Running OFFLINE synthetic self-test (no network / no real market data)...\n")
    all_events: List[SignalEvent] = []
    for label, seed in [("SYNTH_A", 1), ("SYNTH_B", 2), ("SYNTH_C", 3), ("SYNTH_D", 4), ("SYNTH_E", 5)]:
        df = _generate_synthetic_ohlcv(n_days=900, seed=seed)
        events = backtest_dataframe(df, label, stride=1)
        print(f"  {label}: {len(events)} signal events")
        all_events.extend(events)

    print(f"\nTotal events across synthetic universe: {len(all_events)}")
    summary = summarize_events(all_events)
    print("\nScorecard (SYNTHETIC DATA -- structural self-test only):\n")
    print(summary.to_string(index=False))
