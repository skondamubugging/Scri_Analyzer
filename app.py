"""
================================================================================
 NSE/BSE Quantitative Script Analyzer & Momentum Burst Engine
================================================================================
A production-ready Streamlit dashboard for Indian equities (NSE/BSE) that
implements:
    1. Mark Minervini's Trend Template (Stage 2 structural uptrend filter)
    2. Pradeep Bonde's (Stockbee) Momentum Burst Engine
    3. A News Sentiment Agent (Google News RSS + VADER sentiment scoring)
    4. A Decision Matrix Agent that fuses technicals + sentiment into a
       single actionable call: STRONG BUY / SPECULATIVE BUY / HOLD /
       TAKE PROFIT / SELL / AVOID

Architecture
------------
This app is deliberately organized as a set of cooperating "agents"
(plain Python classes, each with a single responsibility):

    TechnicalDataAgent   -> fetches OHLCV data via yfinance
    MinerviniAnalyzer    -> Trend Template + VCP squeeze detection
    BondeMomentumAgent   -> Momentum Burst detection + exit mechanics
    SentimentAgent       -> Google News RSS scraping + VADER scoring
    DecisionMatrixAgent  -> merges technical + sentiment signals

The Streamlit UI layer at the bottom wires these agents together, renders
a scannable, conditionally-formatted table, and offers a per-stock detail
drill-down with charts and a plain-English rationale.

Run with:
    streamlit run nse_bse_momentum_dashboard.py

Dependencies:
    pip install streamlit yfinance pandas numpy plotly vaderSentiment \
                requests beautifulsoup4 lxml
================================================================================
"""

from __future__ import annotations

import concurrent.futures
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================

st.set_page_config(
    page_title="NSE/BSE Momentum & Trend Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

HIST_PERIOD = "1y"          # 1 year of daily data, per spec
HIST_INTERVAL = "1d"
NEWS_HEADLINE_LIMIT = 12    # headlines scored per ticker
REQUEST_TIMEOUT = 6         # seconds, for news scraping
MAX_WORKERS = 8             # thread pool size for parallel fetches

EXCHANGE_SUFFIX = {"NSE": ".NS", "BSE": ".BO"}


# ==============================================================================
# DATA CONTAINERS
# ==============================================================================

@dataclass
class TechnicalResult:
    ticker_display: str
    ticker_yf: str
    ok: bool = False
    error: Optional[str] = None

    current_price: float = np.nan
    prev_close: float = np.nan
    daily_pct: float = np.nan

    sma50: float = np.nan
    sma150: float = np.nan
    sma200: float = np.nan
    sma5: float = np.nan

    high_52w: float = np.nan
    low_52w: float = np.nan

    minervini_pass: bool = False
    minervini_checks: Dict[str, bool] = field(default_factory=dict)

    vcp_squeeze: bool = False
    vcp_range_pct: float = np.nan

    momentum_burst: bool = False
    burst_checks: Dict[str, bool] = field(default_factory=dict)

    consecutive_up_days: int = 0
    is_up_day: bool = False
    exhaustion_exit: bool = False
    below_5sma_exit: bool = False
    already_extended: bool = False  # Day 3+ of continuous rally -> no new BUY

    # --- Supplementary momentum/oscillator indicators ---
    rsi14: float = np.nan
    rsi_status: str = "N/A"          # Overbought / Oversold / Healthy / Neutral

    macd_line: float = np.nan
    macd_signal: float = np.nan
    macd_hist: float = np.nan
    macd_prev_hist: float = np.nan
    macd_bullish: bool = False       # MACD line above signal line
    macd_bullish_cross: bool = False # MACD crossed above signal in the last session
    macd_above_zero: bool = False

    adx14: float = np.nan
    adx_trending: bool = False       # ADX > 20 -> trend has real strength

    # --- Stage 2 Entry Screener (Minervini structure + indicators + sentiment) ---
    stage2_entry_ready: bool = False
    stage2_entry_score: float = 0.0        # 0-100 composite confluence score
    stage2_entry_checks: Dict[str, bool] = field(default_factory=dict)

    history: Optional[pd.DataFrame] = None


@dataclass
class SentimentResult:
    label: str = "Neutral"
    score: float = 0.0          # compound score, -1..1
    n_headlines: int = 0
    headlines: List[Tuple[str, float]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DecisionResult:
    action: str
    rationale: List[str]
    color: str  # hex color for row highlighting


# ==============================================================================
# AGENT 1: TECHNICAL DATA AGENT  (yfinance ingestion + indicator engineering)
# ==============================================================================

class TechnicalDataAgent:
    """Downloads OHLCV history from yfinance and engineers all indicators
    required by the Minervini Trend Template and the Bonde Momentum Burst
    Engine."""

    def __init__(self, period: str = HIST_PERIOD, interval: str = HIST_INTERVAL):
        self.period = period
        self.interval = interval

    @staticmethod
    def build_yf_ticker(raw_ticker: str, exchange: str) -> str:
        """Appends the correct exchange suffix. Leaves the ticker untouched
        if the user has already supplied a suffix."""
        raw = raw_ticker.strip().upper()
        if raw.endswith(".NS") or raw.endswith(".BO"):
            return raw
        suffix = EXCHANGE_SUFFIX.get(exchange, ".NS")
        return f"{raw}{suffix}"

    def fetch(self, raw_ticker: str, exchange: str) -> TechnicalResult:
        ticker_yf = self.build_yf_ticker(raw_ticker, exchange)
        result = TechnicalResult(ticker_display=raw_ticker.strip().upper(), ticker_yf=ticker_yf)

        try:
            df = yf.download(
                ticker_yf,
                period=self.period,
                interval=self.interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as exc:  # network / API failure
            result.error = f"Download failed: {exc}"
            return result

        if df is None or df.empty or len(df) < 60:
            result.error = "Insufficient or no data returned for this ticker."
            return result

        # yfinance sometimes returns a MultiIndex column frame for single
        # tickers depending on version -- normalize defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Close", "Volume"]).copy()
        result.history = df
        result.ok = True

        self._engineer_indicators(result, df)
        self._run_minervini(result, df)
        self._run_vcp_squeeze(result, df)
        self._run_bonde_burst(result, df)
        self._run_exit_mechanics(result, df)
        self._engineer_oscillators(result, df)

        return result

    # ---- indicator engineering -------------------------------------------------
    @staticmethod
    def _engineer_indicators(result: TechnicalResult, df: pd.DataFrame) -> None:
        df["SMA5"] = df["Close"].rolling(5).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA150"] = df["Close"].rolling(150).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["Vol20Avg"] = df["Volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        result.current_price = float(last["Close"])
        result.prev_close = float(prev["Close"])
        result.daily_pct = (
            (result.current_price - result.prev_close) / result.prev_close * 100.0
            if result.prev_close
            else np.nan
        )

        result.sma5 = float(last["SMA5"]) if not pd.isna(last["SMA5"]) else np.nan
        result.sma50 = float(last["SMA50"]) if not pd.isna(last["SMA50"]) else np.nan
        result.sma150 = float(last["SMA150"]) if not pd.isna(last["SMA150"]) else np.nan
        result.sma200 = float(last["SMA200"]) if not pd.isna(last["SMA200"]) else np.nan

        # 52-week (trailing ~252 trading day) high/low
        window = df.tail(252)
        result.high_52w = float(window["High"].max())
        result.low_52w = float(window["Low"].min())

    # ---- A. Minervini Trend Template -------------------------------------------
    @staticmethod
    def _run_minervini(result: TechnicalResult, df: pd.DataFrame) -> None:
        checks: Dict[str, bool] = {}

        price = result.current_price
        sma50, sma150, sma200 = result.sma50, result.sma150, result.sma200

        has_all_smas = not any(pd.isna(x) for x in (sma50, sma150, sma200))

        # 1. Price above all three key moving averages
        checks["price_above_all_smas"] = bool(
            has_all_smas and price > sma50 > 0 and price > sma150 > 0 and price > sma200 > 0
        )

        # 2. Moving average stacking order: 50 > 150 > 200
        checks["ma_stack_aligned"] = bool(has_all_smas and sma50 > sma150 > sma200)

        # 3. 200-day SMA trending up (current > value 20 trading days ago)
        sma200_trending_up = False
        if len(df) >= 220 and not pd.isna(df["SMA200"].iloc[-1]) and not pd.isna(df["SMA200"].iloc[-21]):
            sma200_trending_up = bool(df["SMA200"].iloc[-1] > df["SMA200"].iloc[-21])
        checks["sma200_trending_up"] = sma200_trending_up

        # 4. Within 25% of 52-week high
        checks["near_52w_high"] = bool(
            not pd.isna(result.high_52w) and result.high_52w > 0
            and price >= result.high_52w * 0.75
        )

        # 5. At least 30% above 52-week low
        checks["above_52w_low"] = bool(
            not pd.isna(result.low_52w) and result.low_52w > 0
            and price >= result.low_52w * 1.30
        )

        result.minervini_checks = checks
        result.minervini_pass = all(checks.values())

    # ---- VCP Squeeze filter ------------------------------------------------
    @staticmethod
    def _run_vcp_squeeze(result: TechnicalResult, df: pd.DataFrame) -> None:
        last5 = df.tail(5)
        if len(last5) < 5:
            return
        hi, lo, mean_close = last5["High"].max(), last5["Low"].min(), last5["Close"].mean()
        if mean_close <= 0:
            return
        range_pct = (hi - lo) / mean_close * 100.0
        result.vcp_range_pct = float(range_pct)
        result.vcp_squeeze = bool(range_pct < 3.0)

    # ---- B. Bonde (Stockbee) Momentum Burst --------------------------------
    @staticmethod
    def _run_bonde_burst(result: TechnicalResult, df: pd.DataFrame) -> None:
        checks: Dict[str, bool] = {}
        if len(df) < 26:
            result.burst_checks = checks
            return

        last, prev = df.iloc[-1], df.iloc[-2]

        # 1. Daily range expansion: close >= 4% higher than previous close
        pct_change = (last["Close"] - prev["Close"]) / prev["Close"] * 100.0 if prev["Close"] else np.nan
        checks["range_expansion_4pct"] = bool(not pd.isna(pct_change) and pct_change >= 4.0)

        # 2. Volume expansion: today's volume > yesterday's AND > 20d average
        vol20 = last["Vol20Avg"]
        checks["volume_vs_prev_day"] = bool(last["Volume"] > prev["Volume"])
        checks["volume_vs_20d_avg"] = bool(not pd.isna(vol20) and last["Volume"] > vol20)

        # 3. Pivot tightness: the 5 days PRIOR to the burst day were tight
        #    (< 2% standard deviation relative to mean close, using the
        #    5 sessions before today's burst candle).
        pre_burst_window = df.iloc[-6:-1]  # 5 sessions preceding the burst day
        tight = False
        if len(pre_burst_window) == 5:
            mean_c = pre_burst_window["Close"].mean()
            std_c = pre_burst_window["Close"].std()
            if mean_c > 0:
                rel_std_pct = (std_c / mean_c) * 100.0
                tight = bool(rel_std_pct < 2.0)
        checks["pivot_tightness_prior_5d"] = tight

        result.burst_checks = checks
        result.momentum_burst = all(checks.values()) if checks else False

    # ---- Execution / Risk / Exit mechanics ---------------------------------
    @staticmethod
    def _run_exit_mechanics(result: TechnicalResult, df: pd.DataFrame) -> None:
        closes = df["Close"]
        if len(closes) < 6:
            return

        # Determine consecutive up-day streak ending on the PRIOR day,
        # then evaluate whether today is a down day (Exhaustion Pivot).
        diffs = closes.diff()
        is_up = diffs > 0

        # Consecutive up days counted up to and including yesterday
        streak = 0
        for val in is_up.iloc[-2::-1]:  # walk backward starting at yesterday
            if val:
                streak += 1
            else:
                break
        result.consecutive_up_days = int(streak)

        result.is_up_day = bool(is_up.iloc[-1]) if not pd.isna(diffs.iloc[-1]) else False

        # Exhaustion Exit: today is a DOWN day immediately after 3+ up days
        result.exhaustion_exit = bool((not result.is_up_day) and streak >= 3)

        # Below-5SMA exit
        result.below_5sma_exit = bool(
            not pd.isna(result.sma5) and result.current_price < result.sma5
        )

        # "Already extended" guard: Day 3+ of a continuous rally counting
        # TODAY as part of the up-streak -> suppress new BUY signals.
        # (i.e. today is an up day and, including today, the rally is
        # already 3 or more days old.)
        rally_len_incl_today = streak + (1 if result.is_up_day else 0)
        result.already_extended = bool(result.is_up_day and rally_len_incl_today >= 3)

    # ---- D. Supplementary oscillators: RSI(14), MACD(12,26,9), ADX(14) ----
    @staticmethod
    def _engineer_oscillators(result: TechnicalResult, df: pd.DataFrame) -> None:
        if len(df) < 35:
            return  # not enough history for a stable MACD/ADX read

        close = df["Close"]

        # --- RSI(14), Wilder's smoothing method ---
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)  # avg_loss == 0 -> maximally overbought
        df["RSI14"] = rsi

        last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else np.nan
        result.rsi14 = last_rsi
        if not pd.isna(last_rsi):
            if last_rsi >= 70:
                result.rsi_status = "Overbought"
            elif last_rsi <= 30:
                result.rsi_status = "Oversold"
            elif 45 <= last_rsi <= 70:
                result.rsi_status = "Healthy"
            else:
                result.rsi_status = "Neutral"

        # --- MACD(12, 26, 9) ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd_line, signal_line, hist

        if len(hist) >= 2 and not pd.isna(hist.iloc[-1]) and not pd.isna(hist.iloc[-2]):
            result.macd_line = float(macd_line.iloc[-1])
            result.macd_signal = float(signal_line.iloc[-1])
            result.macd_hist = float(hist.iloc[-1])
            result.macd_prev_hist = float(hist.iloc[-2])
            result.macd_bullish = bool(result.macd_line > result.macd_signal)
            result.macd_above_zero = bool(result.macd_line > 0)
            # Bullish crossover: histogram flips from <=0 to >0 in the latest session
            result.macd_bullish_cross = bool(result.macd_prev_hist <= 0 and result.macd_hist > 0)

        # --- ADX(14) — Average Directional Index (trend strength, Wilder's method) ---
        high, low = df["High"], df["Low"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

        if not adx.empty and not pd.isna(adx.iloc[-1]):
            result.adx14 = float(adx.iloc[-1])
            result.adx_trending = bool(result.adx14 >= 20)

    # ---- E. Stage 2 Entry Screener (structure + indicators + sentiment) ---
    @staticmethod
    def run_stage2_entry_screen(result: TechnicalResult, sent: "SentimentResult") -> None:
        """Flags scripts that are structurally in a Minervini Stage 2 uptrend
        AND are showing fresh, indicator-confirmed strength (MACD + RSI),
        AND are not contradicted by negative news sentiment -- independent of
        whether a Bonde burst day has fired yet. This surfaces stocks that
        are *building* toward a high-probability entry, not just the ones
        already bursting today."""
        checks: Dict[str, bool] = {}

        checks["minervini_stage2"] = bool(result.minervini_pass)
        checks["macd_bullish"] = bool(result.macd_bullish)
        checks["macd_above_zero"] = bool(result.macd_above_zero)
        checks["rsi_healthy_zone"] = bool(result.rsi_status in ("Healthy", "Neutral") and not pd.isna(result.rsi14))
        checks["not_overbought"] = bool(pd.isna(result.rsi14) or result.rsi14 < 75)
        checks["trend_has_strength"] = bool(result.adx_trending)
        checks["not_already_extended"] = bool(not result.already_extended)
        checks["no_active_exit_signal"] = bool(not result.exhaustion_exit and not result.below_5sma_exit)
        checks["sentiment_not_bearish"] = bool(sent.label != "Bearish")

        result.stage2_entry_checks = checks

        # Weighted composite score (0-100). Structure + MACD carry the most
        # weight since they are the primary trend-confirmation signals;
        # sentiment and ADX act as amplifiers/tiebreakers.
        weights = {
            "minervini_stage2": 30,
            "macd_bullish": 20,
            "macd_above_zero": 10,
            "rsi_healthy_zone": 10,
            "not_overbought": 5,
            "trend_has_strength": 10,
            "not_already_extended": 5,
            "no_active_exit_signal": 5,
            "sentiment_not_bearish": 5,
        }
        score = sum(weights[k] for k, passed in checks.items() if passed)
        # Sentiment amplifier: bullish news nudges the score up further, on top
        # of the base pass/fail credit above (Confluence Rule).
        if sent.label == "Bullish":
            score = min(100, score + 5)
        result.stage2_entry_score = float(score)

        # A script only qualifies as "Stage 2 Entry Ready" if the core
        # structural + momentum-confirmation checks all pass -- sentiment and
        # ADX are amplifiers, not gatekeepers, consistent with the Confluence Rule.
        core_gate = (
            checks["minervini_stage2"]
            and checks["macd_bullish"]
            and checks["rsi_healthy_zone"]
            and checks["not_overbought"]
            and checks["not_already_extended"]
            and checks["no_active_exit_signal"]
            and checks["sentiment_not_bearish"]
        )
        result.stage2_entry_ready = bool(core_gate)


# ==============================================================================
# AGENT 2: SENTIMENT ANALYSIS AGENT (Google News RSS + VADER)
# ==============================================================================

class SentimentAgent:
    """Scrapes recent global headlines for a given script via Google News
    RSS and scores aggregate sentiment using VADER (rule-based, works well
    on short headline text without needing a model download)."""

    def __init__(self, headline_limit: int = NEWS_HEADLINE_LIMIT):
        self.headline_limit = headline_limit
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, company_query: str) -> SentimentResult:
        result = SentimentResult()
        try:
            headlines = self._fetch_headlines(company_query)
        except Exception as exc:
            result.error = f"News fetch failed: {exc}"
            return result

        if not headlines:
            result.error = "No recent headlines found."
            return result

        scored: List[Tuple[str, float]] = []
        for headline in headlines[: self.headline_limit]:
            compound = self.analyzer.polarity_scores(headline)["compound"]
            scored.append((headline, compound))

        avg_compound = float(np.mean([c for _, c in scored])) if scored else 0.0
        result.headlines = scored
        result.n_headlines = len(scored)
        result.score = avg_compound
        result.label = self._classify(avg_compound)
        return result

    @staticmethod
    def _classify(compound: float) -> str:
        if compound >= 0.15:
            return "Bullish"
        if compound <= -0.15:
            return "Bearish"
        return "Neutral"

    def _fetch_headlines(self, company_query: str) -> List[str]:
        """Pulls headlines from Google News RSS for the given search query."""
        query = quote_plus(f"{company_query} stock NSE OR BSE")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "xml") if self._has_lxml() else BeautifulSoup(resp.content, "html.parser")
        items = soup.find_all("item")

        headlines = []
        for item in items:
            title_tag = item.find("title")
            if title_tag and title_tag.text:
                # Google News often appends " - Source Name"; strip it for cleaner scoring.
                clean = re.sub(r"\s+-\s+[^-]+$", "", title_tag.text).strip()
                if clean:
                    headlines.append(clean)
        return headlines

    @staticmethod
    def _has_lxml() -> bool:
        try:
            import lxml  # noqa: F401
            return True
        except ImportError:
            return False


# ==============================================================================
# AGENT 3: DECISION MATRIX AGENT
# ==============================================================================

class DecisionMatrixAgent:
    """Fuses the TechnicalResult and SentimentResult into one definitive
    call. Sentiment is used strictly as a directional AMPLIFIER, never as
    a standalone trigger, per the Confluence Rule."""

    COLOR_STRONG_BUY = "#1e8f3e"       # vibrant green
    COLOR_SPECULATIVE_BUY = "#a9d18e"  # soft green
    COLOR_HOLD = "#f4d35e"             # amber
    COLOR_SELL = "#f4a3a3"             # light red
    COLOR_AVOID = "#e0e0e0"            # neutral grey

    def decide(self, tech: TechnicalResult, sent: SentimentResult) -> DecisionResult:
        rationale: List[str] = []

        if not tech.ok:
            return DecisionResult(
                action="NO DATA",
                rationale=[tech.error or "Data unavailable."],
                color=self.COLOR_AVOID,
            )

        # --- Highest priority: exit signals override everything else ---
        if tech.exhaustion_exit:
            rationale.append(
                f"Exhaustion Pivot: today is a down day after "
                f"{tech.consecutive_up_days} consecutive up days -> Stockbee exit rule triggered."
            )
        if tech.below_5sma_exit:
            rationale.append(
                f"Price ({tech.current_price:.2f}) has closed below the 5-day SMA "
                f"({tech.sma5:.2f}) -> trailing stop violated."
            )
        if tech.exhaustion_exit or tech.below_5sma_exit:
            rationale.append("HOLD rule broken: position should be closed to lock in gains / limit loss.")
            return DecisionResult(action="TAKE PROFIT / SELL", rationale=rationale, color=self.COLOR_SELL)

        # --- Build the BUY case ---
        minervini_ok = tech.minervini_pass
        burst_ok = tech.momentum_burst
        blocked_by_extension = tech.already_extended

        if minervini_ok:
            rationale.append("Minervini Trend Template: PASSED (Stage 2 structural uptrend confirmed).")
        else:
            failed = [k for k, v in tech.minervini_checks.items() if not v]
            if failed:
                rationale.append(f"Minervini Trend Template: FAILED checks -> {', '.join(failed)}.")

        if tech.vcp_squeeze:
            rationale.append(f"VCP Squeeze detected: 5-day range at {tech.vcp_range_pct:.2f}% (tight base).")

        if burst_ok:
            rationale.append("Stockbee Momentum Burst: Day-1 breakout confirmed (range + volume + prior tightness).")
        elif tech.burst_checks:
            failed_b = [k for k, v in tech.burst_checks.items() if not v]
            if failed_b:
                rationale.append(f"Momentum Burst: not yet confirmed -> missing {', '.join(failed_b)}.")

        if blocked_by_extension:
            rationale.append(
                f"CRITICAL RESTRICTION: stock is already on Day {tech.consecutive_up_days + 1}+ of a "
                f"continuous rally -> new BUY signal suppressed regardless of setup quality."
            )

        # --- Supplementary oscillator readout (RSI / MACD / ADX) ---
        if not pd.isna(tech.rsi14):
            rationale.append(f"RSI(14) = {tech.rsi14:.1f} -> {tech.rsi_status}.")
        if not pd.isna(tech.macd_line):
            macd_desc = "bullish (line above signal)" if tech.macd_bullish else "bearish (line below signal)"
            cross_note = " — fresh bullish crossover today." if tech.macd_bullish_cross else ""
            rationale.append(f"MACD: {macd_desc}{cross_note} (MACD {tech.macd_line:.2f} vs Signal {tech.macd_signal:.2f}).")
        if not pd.isna(tech.adx14):
            rationale.append(f"ADX(14) = {tech.adx14:.1f} -> {'trend has real strength' if tech.adx_trending else 'trend is weak/choppy'}.")
        if tech.stage2_entry_ready:
            rationale.append(
                f"🎯 Stage 2 Entry Screener: QUALIFIES (confluence score {tech.stage2_entry_score:.0f}/100) -- "
                f"structure, MACD and RSI are all aligned for a Stage 2 entry."
            )

        rationale.append(f"News Sentiment: {sent.label} (compound score {sent.score:+.2f} over {sent.n_headlines} headlines).")

        # --- Decision logic ---
        if minervini_ok and burst_ok and not blocked_by_extension:
            if sent.label in ("Bullish", "Neutral"):
                rationale.append("Confluence Rule: technical breakout confirmed" +
                                  (" and reinforced by a bullish news catalyst." if sent.label == "Bullish" else " with no contradicting news."))
                return DecisionResult(action="STRONG BUY", rationale=rationale, color=self.COLOR_STRONG_BUY)
            else:  # Bearish news against a technical breakout
                rationale.append("Confluence Rule: technical breakout present, but bearish sentiment introduces risk -> downgraded to speculative.")
                return DecisionResult(action="SPECULATIVE BUY", rationale=rationale, color=self.COLOR_SPECULATIVE_BUY)

        if minervini_ok and not burst_ok and not blocked_by_extension:
            if tech.stage2_entry_ready:
                rationale.append(
                    "Structure is sound and MACD/RSI/sentiment confluence confirms building strength, but "
                    "no confirmed Bonde burst day yet -> speculative entry only, watching for a trigger day."
                )
                return DecisionResult(action="SPECULATIVE BUY", rationale=rationale, color=self.COLOR_SPECULATIVE_BUY)
            if sent.label == "Bullish":
                rationale.append("Structure is sound and news is supportive, but no confirmed burst yet -> speculative entry only on a subsequent trigger day.")
                return DecisionResult(action="SPECULATIVE BUY", rationale=rationale, color=self.COLOR_SPECULATIVE_BUY)
            rationale.append("Structure sound but no burst trigger yet -> wait and watch.")
            return DecisionResult(action="HOLD", rationale=rationale, color=self.COLOR_HOLD)

        if blocked_by_extension and minervini_ok:
            rationale.append("Setup otherwise qualifies, but the extension restriction blocks fresh entries -> monitor for a pullback to the 5/10-day SMA.")
            return DecisionResult(action="HOLD", rationale=rationale, color=self.COLOR_HOLD)

        rationale.append("Structural (Minervini) criteria not met -> not in a qualifying Stage 2 uptrend.")
        return DecisionResult(action="AVOID", rationale=rationale, color=self.COLOR_AVOID)


# ==============================================================================
# ORCHESTRATION HELPERS
# ==============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def _cached_technical_fetch(raw_ticker: str, exchange: str) -> TechnicalResult:
    agent = TechnicalDataAgent()
    return agent.fetch(raw_ticker, exchange)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_sentiment_fetch(company_query: str) -> SentimentResult:
    agent = SentimentAgent()
    return agent.analyze(company_query)


def run_pipeline_for_ticker(raw_ticker: str, exchange: str) -> Tuple[TechnicalResult, SentimentResult, DecisionResult]:
    tech = _cached_technical_fetch(raw_ticker, exchange)
    sent = _cached_sentiment_fetch(raw_ticker.strip().upper())
    if tech.ok:
        TechnicalDataAgent.run_stage2_entry_screen(tech, sent)
    decision_agent = DecisionMatrixAgent()
    decision = decision_agent.decide(tech, sent)
    return tech, sent, decision


def run_batch(tickers: List[str], exchange: str) -> Dict[str, Tuple[TechnicalResult, SentimentResult, DecisionResult]]:
    results: Dict[str, Tuple[TechnicalResult, SentimentResult, DecisionResult]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(run_pipeline_for_ticker, t, exchange): t for t in tickers}
        for future in concurrent.futures.as_completed(future_map):
            raw_t = future_map[future]
            try:
                results[raw_t] = future.result()
            except Exception as exc:
                blank_tech = TechnicalResult(ticker_display=raw_t.upper(), ticker_yf=raw_t.upper(), error=str(exc))
                results[raw_t] = (blank_tech, SentimentResult(error=str(exc)),
                                   DecisionResult(action="ERROR", rationale=[str(exc)], color="#e0e0e0"))
    # Preserve user-entered order
    return {t: results[t] for t in tickers if t in results}


# ==============================================================================
# UI HELPERS
# ==============================================================================

def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2f}%"


def fmt_price(x: float) -> str:
    return "—" if pd.isna(x) else f"₹{x:,.2f}"


def build_summary_dataframe(pipeline_results: Dict[str, Tuple[TechnicalResult, SentimentResult, DecisionResult]]) -> pd.DataFrame:
    rows = []
    for raw_ticker, (tech, sent, decision) in pipeline_results.items():
        rows.append(
            {
                "Ticker": tech.ticker_display,
                "Symbol (Yahoo)": tech.ticker_yf,
                "Current Price": fmt_price(tech.current_price) if tech.ok else "N/A",
                "Daily %": fmt_pct(tech.daily_pct) if tech.ok else "N/A",
                "Minervini Status": "✅ PASS" if tech.minervini_pass else ("❌ FAIL" if tech.ok else "—"),
                "VCP Squeeze": "🟢 Tight" if tech.vcp_squeeze else ("⚪ No" if tech.ok else "—"),
                "Momentum Burst": "🚀 BURST" if tech.momentum_burst else ("— " if tech.ok else "—"),
                "RSI(14)": f"{tech.rsi14:.1f} ({tech.rsi_status})" if tech.ok and not pd.isna(tech.rsi14) else "—",
                "MACD": ("🟢 Bull" if tech.macd_bullish else "🔴 Bear") + (" ↗️" if tech.macd_bullish_cross else "") if tech.ok and not pd.isna(tech.macd_line) else "—",
                "ADX(14)": f"{tech.adx14:.1f}" if tech.ok and not pd.isna(tech.adx14) else "—",
                "News Sentiment": f"{sent.label} ({sent.score:+.2f})" if not sent.error else "N/A",
                "Stage 2 Entry": f"🎯 {tech.stage2_entry_score:.0f}/100" if tech.stage2_entry_ready else (f"{tech.stage2_entry_score:.0f}/100" if tech.ok else "—"),
                "Final Action": decision.action,
                "_color": decision.color,
            }
        )
    return pd.DataFrame(rows)


def build_stage2_screener_dataframe(pipeline_results: Dict[str, Tuple[TechnicalResult, SentimentResult, DecisionResult]]) -> pd.DataFrame:
    """Builds a focused table of scripts that qualify as Stage 2 Entry
    candidates: Minervini structure + MACD/RSI/ADX confirmation + sentiment
    not bearish -- ranked by confluence score, highest first."""
    rows = []
    for raw_ticker, (tech, sent, decision) in pipeline_results.items():
        if not tech.ok or not tech.stage2_entry_ready:
            continue
        rows.append(
            {
                "Ticker": tech.ticker_display,
                "Price": fmt_price(tech.current_price),
                "Confluence Score": tech.stage2_entry_score,
                "RSI(14)": f"{tech.rsi14:.1f}",
                "MACD": "Bullish Crossover" if tech.macd_bullish_cross else "Bullish",
                "ADX(14)": f"{tech.adx14:.1f}",
                "% from 52W High": f"{((tech.current_price / tech.high_52w) - 1) * 100:+.1f}%" if not pd.isna(tech.high_52w) and tech.high_52w else "—",
                "News Sentiment": sent.label,
                "Burst Fired Today?": "Yes 🚀" if tech.momentum_burst else "Not yet",
                "Final Action": decision.action,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Confluence Score", ascending=False).reset_index(drop=True)
        df["Confluence Score"] = df["Confluence Score"].map(lambda x: f"{x:.0f}/100")
    return df


def style_action_table(df: pd.DataFrame):
    color_map = dict(zip(df["Ticker"], df["_color"]))
    display_df = df.drop(columns=["_color"])

    def highlight_row(row: pd.Series):
        color = color_map.get(row["Ticker"], "#ffffff")
        return [f"background-color: {color}; color: #10141a; font-weight: 600" if col == "Final Action"
                else f"background-color: {color}22" for col in row.index]

    styled = display_df.style.apply(highlight_row, axis=1)
    return styled


def render_price_chart(tech: TechnicalResult):
    import plotly.graph_objects as go

    df = tech.history.tail(150).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#1e8f3e", decreasing_line_color="#c0392b",
    ))
    for col, color, label in [
        ("SMA5", "#f39c12", "5 SMA"),
        ("SMA50", "#3498db", "50 SMA"),
        ("SMA150", "#9b59b6", "150 SMA"),
        ("SMA200", "#2c3e50", "200 SMA"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=label,
                                      line=dict(width=1.3, color=color)))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#7f8c8d"))
    if "Vol20Avg" in df.columns:
        vol_fig.add_trace(go.Scatter(x=df.index, y=df["Vol20Avg"], mode="lines",
                                      name="20D Avg Vol", line=dict(color="#e74c3c", width=1.5)))
    vol_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white")
    st.plotly_chart(vol_fig, use_container_width=True)

    # --- RSI(14) panel ---
    if "RSI14" in df.columns:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI(14)",
                                      line=dict(color="#8e44ad", width=1.5)))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="#c0392b", annotation_text="Overbought (70)")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="#1e8f3e", annotation_text="Oversold (30)")
        rsi_fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                               yaxis=dict(range=[0, 100]))
        st.plotly_chart(rsi_fig, use_container_width=True)

    # --- MACD panel ---
    if "MACD" in df.columns:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD",
                                       line=dict(color="#2980b9", width=1.5)))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal",
                                       line=dict(color="#e67e22", width=1.5)))
        hist_colors = ["#1e8f3e" if v >= 0 else "#c0392b" for v in df["MACD_HIST"]]
        macd_fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram", marker_color=hist_colors))
        macd_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(macd_fig, use_container_width=True)


def render_detail_expander(raw_ticker: str, tech: TechnicalResult, sent: SentimentResult, decision: DecisionResult):
    with st.expander(f"🔎 {tech.ticker_display}  —  {decision.action}", expanded=False):
        if not tech.ok:
            st.error(tech.error or "No data available for this ticker.")
            return

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", fmt_price(tech.current_price), fmt_pct(tech.daily_pct))
        col2.metric("52W High", fmt_price(tech.high_52w))
        col3.metric("52W Low", fmt_price(tech.low_52w))
        col4.metric("5-Day SMA", fmt_price(tech.sma5))

        st.markdown(f"**Final Action Strategy:** :orange[{decision.action}]" if decision.action in ("HOLD", "SPECULATIVE BUY")
                    else f"**Final Action Strategy:** {decision.action}")

        st.markdown("##### Why the system suggested this action")
        for line in decision.rationale:
            st.markdown(f"- {line}")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Chart", "🧮 Framework Checklist", "🎯 Stage 2 Entry Screener", "📰 News Headlines"]
        )

        with tab1:
            render_price_chart(tech)

        with tab2:
            st.markdown("**A. Minervini Trend Template**")
            for check, passed in tech.minervini_checks.items():
                st.markdown(f"{'✅' if passed else '❌'} {check.replace('_', ' ').title()}")
            st.markdown(f"{'✅' if tech.vcp_squeeze else '⚪'} VCP Squeeze "
                        f"(5-day range = {tech.vcp_range_pct:.2f}%, threshold < 3%)" if not pd.isna(tech.vcp_range_pct) else "⚪ VCP Squeeze: insufficient data")

            st.markdown("---")
            st.markdown("**B. Stockbee Momentum Burst Engine**")
            if tech.burst_checks:
                for check, passed in tech.burst_checks.items():
                    st.markdown(f"{'✅' if passed else '❌'} {check.replace('_', ' ').title()}")
            else:
                st.markdown("_Insufficient history to evaluate burst criteria._")

            st.markdown("---")
            st.markdown("**C. Execution & Risk Protocols**")
            st.markdown(f"{'⚠️' if tech.exhaustion_exit else '✅'} Exhaustion Exit "
                        f"({tech.consecutive_up_days} consecutive up days prior to today, today is "
                        f"{'DOWN' if not tech.is_up_day else 'UP'})")
            st.markdown(f"{'⚠️' if tech.below_5sma_exit else '✅'} Price vs 5-Day SMA "
                        f"({fmt_price(tech.current_price)} vs {fmt_price(tech.sma5)})")
            st.markdown(f"{'🚫' if tech.already_extended else '✅'} Extension Guard "
                        f"(Day 3+ continuous rally restriction)")

        with tab3:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("RSI(14)", f"{tech.rsi14:.1f}" if not pd.isna(tech.rsi14) else "N/A", tech.rsi_status)
            col_b.metric("MACD", f"{tech.macd_line:.2f}" if not pd.isna(tech.macd_line) else "N/A",
                         "Bullish" if tech.macd_bullish else "Bearish")
            col_c.metric("ADX(14)", f"{tech.adx14:.1f}" if not pd.isna(tech.adx14) else "N/A",
                         "Trending" if tech.adx_trending else "Weak/Choppy")

            st.markdown(
                f"### {'🎯 Qualifies for Stage 2 Entry' if tech.stage2_entry_ready else '⏳ Not Yet a Stage 2 Entry'}"
                f"  —  Confluence Score: **{tech.stage2_entry_score:.0f}/100**"
            )
            st.progress(min(1.0, tech.stage2_entry_score / 100))

            st.markdown("**Confluence Checklist** _(structure + MACD/RSI must all pass; sentiment/ADX amplify)_")
            labels = {
                "minervini_stage2": "Minervini Stage 2 structure confirmed",
                "macd_bullish": "MACD line above signal line",
                "macd_above_zero": "MACD above the zero line (net bullish momentum)",
                "rsi_healthy_zone": "RSI(14) in a healthy momentum zone (45-70)",
                "not_overbought": "RSI(14) not extremely overbought (< 75)",
                "trend_has_strength": "ADX(14) ≥ 20 (trend has real strength)",
                "not_already_extended": "Not already extended (no Day 3+ rally block)",
                "no_active_exit_signal": "No active exit signal (exhaustion / 5-SMA break)",
                "sentiment_not_bearish": "News sentiment is not bearish",
            }
            for key, passed in tech.stage2_entry_checks.items():
                st.markdown(f"{'✅' if passed else '❌'} {labels.get(key, key)}")

        with tab4:
            if sent.error:
                st.info(sent.error)
            else:
                st.markdown(f"**Aggregate Sentiment:** {sent.label}  (compound score: {sent.score:+.2f} "
                            f"across {sent.n_headlines} headlines)")
                for headline, score in sent.headlines:
                    tag = "🟢" if score >= 0.15 else ("🔴" if score <= -0.15 else "⚪")
                    st.markdown(f"{tag} {headline}  `({score:+.2f})`")


# ==============================================================================
# STREAMLIT MAIN APP
# ==============================================================================

def main():
    st.title("Quantitative Script Analyzer")
    st.caption(" Developed by K.Srichandan")
    st.caption(
        "Minervini Trend Template × Stockbee Momentum Burst Engine × Live News Sentiment — "
        "a multi-agent scanner for high-probability, low-risk breakout entries."
    )

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("⚙️ Scanner Configuration")
        exchange = st.radio("Exchange", options=["NSE", "BSE"], horizontal=True, index=0)
        default_watchlist = "TATAMOTORS, RELIANCE, TCS, INFY, HDFCBANK"
        tickers_raw = st.text_area(
            "Stocks (comma-separated)",
            value=default_watchlist,
            height=100,
            help="Enter NSE/BSE tradingsymbols without the exchange suffix — it is added automatically "
                 "(e.g. TATAMOTORS -> TATAMOTORS.NS on NSE, TATAMOTORS.BO on BSE).",
        )
        st.markdown("---")
        st.markdown("**Framework Thresholds** _(fixed per skill spec)_")
        st.caption("• Burst: Close ≥ +4% • Vol > prev day & 20D avg\n\n"
                    "• Prior 5D pivot std-dev < 2%\n\n"
                    "• VCP squeeze: 5D range < 3%\n\n"
                    "• Exit: below 5-SMA or exhaustion pivot\n\n"
                    "• Stage 2 Entry: RSI(14) 45–70, MACD bullish, ADX ≥ 20, sentiment not bearish")
        st.markdown("---")
        run_button = st.button("🚀 Run Scanner", type="primary", use_container_width=True)
        st.caption(f"Last run: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}" if run_button else " ")
        st.markdown("---")
        st.caption(
            "⚠️ Educational/analytical tool only. Signals are generated purely from historical price, "
            "volume and public news headlines using fixed rule sets. This is not investment advice; "
            "verify independently and consult a registered advisor before trading."
        )

    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]

    if not tickers:
        st.info("Add at least one ticker in the sidebar to begin scanning.")
        return

    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    if run_button or st.session_state.pipeline_results is None:
        with st.spinner(f"Running Technical, Sentiment & Decision agents on {len(tickers)} script(s)..."):
            st.session_state.pipeline_results = run_batch(tickers, exchange)
            st.session_state.last_exchange = exchange

    pipeline_results = st.session_state.pipeline_results

    # ---------------- Summary metrics ----------------
    df_summary = build_summary_dataframe(pipeline_results)
    n_strong_buy = (df_summary["Final Action"] == "STRONG BUY").sum()
    n_spec_buy = (df_summary["Final Action"] == "SPECULATIVE BUY").sum()
    n_sell = (df_summary["Final Action"] == "TAKE PROFIT / SELL").sum()
    n_hold = (df_summary["Final Action"] == "HOLD").sum()

    df_stage2 = build_stage2_screener_dataframe(pipeline_results)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Scripts Scanned", len(df_summary))
    m2.metric("🟢 Strong Buy", int(n_strong_buy))
    m3.metric("🟡 Speculative Buy", int(n_spec_buy))
    m4.metric("⏸️ Hold", int(n_hold))
    m5.metric("🔴 Take Profit / Sell", int(n_sell))
    m6.metric("🎯 Stage 2 Ready", len(df_stage2))

    st.markdown("### 📊 Scanner Results")
    st.dataframe(style_action_table(df_summary), use_container_width=True, hide_index=True)

    st.markdown("### 🎯 Stage 2 Entry Screener")
    st.caption(
        "Scripts that are structurally in a confirmed Minervini Stage 2 uptrend AND show MACD + RSI "
        "momentum confirmation AND aren't contradicted by bearish news — ranked by confluence score. "
        "This surfaces stocks *building* toward a high-probability entry, even before a Bonde burst day fires."
    )
    if df_stage2.empty:
        st.info("No scripts in the current watchlist currently qualify as a Stage 2 entry candidate on all confluence checks.")
    else:
        st.dataframe(df_stage2, use_container_width=True, hide_index=True)

    st.markdown("### 🔍 Per-Stock Deep Dive")
    for raw_ticker, (tech, sent, decision) in pipeline_results.items():
        render_detail_expander(raw_ticker, tech, sent, decision)


if __name__ == "__main__":
    main()
