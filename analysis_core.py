"""
================================================================================
 analysis_core.py
================================================================================
Streamlit-FREE extraction of the analysis agents from app-2.py.

Why this exists
----------------
In app-2.py, TechnicalDataAgent / SentimentAgent / DecisionMatrixAgent never
actually touch `st.*` -- only the top-level config (`st.set_page_config`) and
the `@st.cache_data` wrappers around them do. That means the agents were
already pure/importable; they just lived inside a file that hard-crashes on
import outside a Streamlit runtime (`st.set_page_config()` runs at import
time). Splitting them out here means:

    1. backtest_engine.py can import and reuse the EXACT same rule logic
       the live app uses -- no risk of the backtest silently drifting from
       production behaviour.
    2. You can finally write real pytest unit tests against these classes
       with synthetic OHLCV, per the README's (currently unfulfilled) claim.
    3. app.py becomes a thin UI layer: `from analysis_core import *`.

Nothing in the logic below has been changed from app-2.py -- this is a
lift-and-shift, not a rewrite, so behaviour stays identical.
================================================================================
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# yfinance / requests / bs4 are imported lazily inside the methods that need
# them (TechnicalDataAgent.fetch, SentimentAgent._fetch_headlines) rather
# than at module load time. This means analysis_core -- and therefore
# backtest_engine, which only needs the pure indicator math -- stays
# importable and unit-testable in any environment, even one without network
# libraries installed (e.g. CI, or this sandbox).

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # keep analysis_core importable even if vader isn't installed
    SentimentIntensityAnalyzer = None

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================

HIST_PERIOD = "2y"          # bumped from 1y -> 2y: gives safety margin around
                             # holidays / newly-listed stocks for the 200-SMA
                             # trend check, which also needs a 20-bar lookback
HIST_INTERVAL = "1d"
NEWS_HEADLINE_LIMIT = 12
REQUEST_TIMEOUT = 6
MAX_WORKERS = 8

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

    as_of: Optional[pd.Timestamp] = None   # timestamp of the LAST bar used --
                                            # always show this in any UI so a
                                            # user knows if a signal is based
                                            # on a confirmed close or is stale

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
    already_extended: bool = False

    rsi14: float = np.nan
    rsi_status: str = "N/A"

    macd_line: float = np.nan
    macd_signal: float = np.nan
    macd_hist: float = np.nan
    macd_prev_hist: float = np.nan
    macd_bullish: bool = False
    macd_bullish_cross: bool = False
    macd_above_zero: bool = False

    adx14: float = np.nan
    adx_trending: bool = False

    stage2_entry_ready: bool = False
    stage2_entry_score: float = 0.0
    stage2_entry_checks: Dict[str, bool] = field(default_factory=dict)

    # Liquidity -- NEW: needed to filter out unfillable small-caps before
    # a signal is ever surfaced to a user (see universe_scanner.py)
    avg_turnover_20d_cr: float = np.nan   # 20D avg (Close * Volume), in INR crore

    history: Optional[pd.DataFrame] = None


@dataclass
class SentimentResult:
    label: str = "Neutral"
    score: float = 0.0
    n_headlines: int = 0
    headlines: List[Tuple[str, float]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DecisionResult:
    action: str
    rationale: List[str]
    color: str


# ==============================================================================
# AGENT 1: TECHNICAL DATA AGENT
# ==============================================================================

class TechnicalDataAgent:
    """Downloads OHLCV history from yfinance and engineers all indicators
    required by the Minervini Trend Template and the Bonde Momentum Burst
    Engine. Logic is unchanged from app-2.py."""

    def __init__(self, period: str = HIST_PERIOD, interval: str = HIST_INTERVAL):
        self.period = period
        self.interval = interval

    @staticmethod
    def build_yf_ticker(raw_ticker: str, exchange: str) -> str:
        raw = raw_ticker.strip().upper()
        if raw.endswith(".NS") or raw.endswith(".BO"):
            return raw
        suffix = EXCHANGE_SUFFIX.get(exchange, ".NS")
        return f"{raw}{suffix}"

    def fetch(self, raw_ticker: str, exchange: str) -> TechnicalResult:
        import yfinance as yf  # lazy import -- see module docstring

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
        except Exception as exc:
            result.error = f"Download failed: {exc}"
            return result

        if df is None or df.empty or len(df) < 60:
            result.error = "Insufficient or no data returned for this ticker."
            return result

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Close", "Volume"]).copy()
        result.history = df
        result.ok = True
        result.as_of = df.index[-1]

        self.compute_all(result, df)
        return result

    @classmethod
    def compute_all(cls, result: TechnicalResult, df: pd.DataFrame) -> None:
        """Runs the full indicator/rule pipeline on an already-fetched
        dataframe. Split out from `fetch()` so the backtester can call this
        directly on a windowed slice of historical data without re-hitting
        yfinance per day."""
        cls._engineer_indicators(result, df)
        cls._run_minervini(result, df)
        cls._run_vcp_squeeze(result, df)
        cls._run_bonde_burst(result, df)
        cls._run_exit_mechanics(result, df)
        cls._engineer_oscillators(result, df)
        cls._engineer_liquidity(result, df)

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

        checks["price_above_all_smas"] = bool(
            has_all_smas and price > sma50 > 0 and price > sma150 > 0 and price > sma200 > 0
        )
        checks["ma_stack_aligned"] = bool(has_all_smas and sma50 > sma150 > sma200)

        sma200_trending_up = False
        if len(df) >= 220 and not pd.isna(df["SMA200"].iloc[-1]) and not pd.isna(df["SMA200"].iloc[-21]):
            sma200_trending_up = bool(df["SMA200"].iloc[-1] > df["SMA200"].iloc[-21])
        checks["sma200_trending_up"] = sma200_trending_up

        checks["near_52w_high"] = bool(
            not pd.isna(result.high_52w) and result.high_52w > 0
            and price >= result.high_52w * 0.75
        )
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
        pct_change = (last["Close"] - prev["Close"]) / prev["Close"] * 100.0 if prev["Close"] else np.nan
        checks["range_expansion_4pct"] = bool(not pd.isna(pct_change) and pct_change >= 4.0)

        vol20 = last["Vol20Avg"]
        checks["volume_vs_prev_day"] = bool(last["Volume"] > prev["Volume"])
        checks["volume_vs_20d_avg"] = bool(not pd.isna(vol20) and last["Volume"] > vol20)

        pre_burst_window = df.iloc[-6:-1]
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

        diffs = closes.diff()
        is_up = diffs > 0

        streak = 0
        for val in is_up.iloc[-2::-1]:
            if val:
                streak += 1
            else:
                break
        result.consecutive_up_days = int(streak)
        result.is_up_day = bool(is_up.iloc[-1]) if not pd.isna(diffs.iloc[-1]) else False
        result.exhaustion_exit = bool((not result.is_up_day) and streak >= 3)
        result.below_5sma_exit = bool(
            not pd.isna(result.sma5) and result.current_price < result.sma5
        )

        rally_len_incl_today = streak + (1 if result.is_up_day else 0)
        result.already_extended = bool(result.is_up_day and rally_len_incl_today >= 3)

    # ---- D. Supplementary oscillators: RSI(14), MACD(12,26,9), ADX(14) ----
    @staticmethod
    def _engineer_oscillators(result: TechnicalResult, df: pd.DataFrame) -> None:
        if len(df) < 35:
            return

        close = df["Close"]

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)
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
            result.macd_bullish_cross = bool(result.macd_prev_hist <= 0 and result.macd_hist > 0)

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

    # ---- Liquidity engineering (NEW) ---------------------------------------
    @staticmethod
    def _engineer_liquidity(result: TechnicalResult, df: pd.DataFrame) -> None:
        """20-day average turnover in INR crore (Close * Volume). Used to
        filter out stocks that show a 'burst' on paper but can't actually be
        filled at the signal price in real trading size."""
        window = df.tail(20)
        if window.empty:
            return
        turnover = (window["Close"] * window["Volume"]).mean()
        result.avg_turnover_20d_cr = float(turnover / 1e7)  # INR -> crore

    # ---- E. Stage 2 Entry Screener (structure + indicators + sentiment) ---
    @staticmethod
    def run_stage2_entry_screen(result: TechnicalResult, sent: "SentimentResult") -> None:
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
        if sent.label == "Bullish":
            score = min(100, score + 5)
        result.stage2_entry_score = float(score)

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
    def __init__(self, headline_limit: int = NEWS_HEADLINE_LIMIT):
        self.headline_limit = headline_limit
        self.analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

    def analyze(self, company_query: str) -> SentimentResult:
        result = SentimentResult()
        if self.analyzer is None:
            result.error = "vaderSentiment not installed."
            return result
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
        import requests  # lazy import -- see module docstring
        from urllib.parse import quote_plus

        query = quote_plus(f"{company_query} stock NSE OR BSE")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        from bs4 import BeautifulSoup  # lazy import -- see module docstring
        soup = BeautifulSoup(resp.content, "xml") if self._has_lxml() else BeautifulSoup(resp.content, "html.parser")
        items = soup.find_all("item")

        headlines = []
        for item in items:
            title_tag = item.find("title")
            if title_tag and title_tag.text:
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
    COLOR_STRONG_BUY = "#1e8f3e"
    COLOR_SPECULATIVE_BUY = "#a9d18e"
    COLOR_HOLD = "#f4d35e"
    COLOR_SELL = "#f4a3a3"
    COLOR_AVOID = "#e0e0e0"

    def decide(self, tech: TechnicalResult, sent: SentimentResult) -> DecisionResult:
        rationale: List[str] = []

        if not tech.ok:
            return DecisionResult(
                action="NO DATA",
                rationale=[tech.error or "Data unavailable."],
                color=self.COLOR_AVOID,
            )

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

        if minervini_ok and burst_ok and not blocked_by_extension:
            if sent.label in ("Bullish", "Neutral"):
                rationale.append("Confluence Rule: technical breakout confirmed" +
                                  (" and reinforced by a bullish news catalyst." if sent.label == "Bullish" else " with no contradicting news."))
                return DecisionResult(action="STRONG BUY", rationale=rationale, color=self.COLOR_STRONG_BUY)
            else:
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
