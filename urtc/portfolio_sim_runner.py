#!/usr/bin/env python3
"""
portfolio_sim_runner.py — Cash-aware, stateful portfolio simulator with LLM buy/sell decisions
==============================================================================================

A more realistic evolution of reinvestment_runner.py. Instead of liquidating and
fully reselecting each period, this maintains a live portfolio (cash + fractional
share positions with cost basis) and lets the LLM trade it over time under a hard
budget constraint.

Flow
----
  Period 0  (BUY-ONLY):
     news step  -> candidate stocks (Stage One)
     then the LLM is given each candidate's financial report AND current share price,
     and a fixed cash budget (default $10,000). It buys UP TO Y stocks (Y is a cap,
     not a target) and may keep some cash uninvested for later.

  Period i>0  (BUY / SELL):
     news step  -> candidate stocks
     then, AT THE FINANCIAL-REPORT STEP, the LLM is also given the realized
     performance of every stock it currently holds (price now vs. cost, vs. last
     period) plus current prices and available cash. It issues a set of buy AND
     sell orders. A decision is ATOMIC: total buys may not exceed cash unless sells
     cover the difference. If they do, the whole response is REJECTED and the LLM is
     RE-PROMPTED with the overdraw amount.

Order / share model
-------------------
  Orders are dollar-denominated: {"action":"buy"|"sell", "ticker": "...", "dollars": N}.
  • Default (fractional): assumes commission-free fractional trading (as offered by
    major brokers — Fidelity, Schwab, Robinhood, etc.). Dollars convert exactly to
    fractional shares, so cash is used to the cent.
  • --whole-shares: conservative fallback. A buy of $D buys the most WHOLE shares
    that fit under $D at the current price; leftover stays as cash.

Budget rule
-----------
  Sells are applied first (freeing cash); then buys in order. If, after sells, total
  buys still exceed available cash, the response is rejected and re-prompted (up to
  --max-retries). If retries are exhausted, a safe fallback applies sells + only the
  buys that fit (in order), logging what was skipped.

Run modes
---------
  --dry-run : MockBackend, NO OpenAI calls. Validates the simulation engine
              (cash ledger, atomic reject/re-prompt, fractional vs whole-share,
              <=Y cap, cash retention, multi-period mark-to-market).
  (live)    : RealBackend wraps InvestmentStrategyGenerator's data helpers with new
              cash-aware order prompts. Requires OPENAI_API_KEY in the environment
              (never reads keys hard-coded in source).

Usage
-----
    python portfolio_sim_runner.py --dry-run --sector technology \
        --start 2024-07-01 --end 2025-07-01 --segments 4 \
        --budget 10000 --max-holdings 5

    SECTOR=financials OPENAI_API_KEY=sk-... python portfolio_sim_runner.py \
        --sector financials --start 2024-07-01 --end 2025-07-01 \
        --interval-months 3 --budget 10000 --max-holdings 5
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil.relativedelta import relativedelta

# --- load API keys from .env (kept out of git); safe no-op if .env is absent ---
import os as _os, pathlib as _pl
for _d in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents]:
    _env = _d / ".env"
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _os.environ.setdefault(_k.strip(), _v.strip().strip(chr(34)).strip(chr(39)))
        break
# --- end .env loader ---

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SCCUR_TESTS"))

import sector_config                      # noqa: E402  sector registry (SECTOR env var)
import risk_metrics                        # noqa: E402  risk-adjusted performance metrics
from reinvestment_runner import build_segments, _d, _s   # noqa: E402  reuse scheduler

EPS = 1e-6


def business_days(start: str, end: str, inclusive_end: bool = True) -> List[str]:
    """List of weekday date strings in [start, end] (skips Sat/Sun; no holiday calendar)."""
    out, cur, end_dt = [], _d(start), _d(end)
    while cur < end_dt or (inclusive_end and cur == end_dt):
        if cur.weekday() < 5:
            out.append(_s(cur))
        cur += relativedelta(days=1)
    return out


# =========================================================================== #
#  Portfolio state                                                            #
# =========================================================================== #
@dataclass
class Position:
    shares: float = 0.0
    avg_cost: float = 0.0      # cost basis per share


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    # ---- valuation -------------------------------------------------------- #
    def holdings_value(self, prices: Dict[str, float]) -> float:
        return sum(p.shares * prices.get(t, p.avg_cost) for t, p in self.positions.items()
                   if p.shares > EPS)

    def total_value(self, prices: Dict[str, float]) -> float:
        return self.cash + self.holdings_value(prices)

    def num_holdings(self) -> int:
        return sum(1 for p in self.positions.values() if p.shares > EPS)

    def active_tickers(self) -> List[str]:
        return [t for t, p in self.positions.items() if p.shares > EPS]

    # ---- mutation (assumes validation already passed) --------------------- #
    def buy(self, ticker: str, dollars: float, price: float, whole_shares: bool) -> float:
        """Execute a buy; returns actual dollars spent."""
        if price <= 0 or dollars <= 0:
            return 0.0
        shares = math.floor(dollars / price) if whole_shares else dollars / price
        if shares <= 0:
            return 0.0
        spent = shares * price
        pos = self.positions.setdefault(ticker, Position())
        new_total = pos.shares + shares
        pos.avg_cost = ((pos.avg_cost * pos.shares) + spent) / new_total if new_total > 0 else price
        pos.shares = new_total
        self.cash -= spent
        return spent

    def sell(self, ticker: str, dollars: float, price: float, whole_shares: bool) -> float:
        """Execute a sell (dollars clamped to position value); returns proceeds."""
        pos = self.positions.get(ticker)
        if not pos or pos.shares <= EPS or price <= 0:
            return 0.0
        max_val = pos.shares * price
        sell_val = min(dollars, max_val)
        shares_sold = (math.floor(sell_val / price) if whole_shares else sell_val / price)
        shares_sold = min(shares_sold, pos.shares)
        if shares_sold <= 0:
            return 0.0
        proceeds = shares_sold * price
        pos.shares -= shares_sold
        if pos.shares <= EPS:
            pos.shares = 0.0
        self.cash += proceeds
        return proceeds


# =========================================================================== #
#  Atomic order execution                                                     #
# =========================================================================== #
def _norm_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for o in orders or []:
        action = str(o.get("action", "")).lower().strip()
        ticker = str(o.get("ticker", "")).upper().strip()
        try:
            dollars = float(o.get("dollars", 0) or 0)
        except (TypeError, ValueError):
            dollars = 0.0
        if action in ("buy", "sell") and ticker and dollars > 0:
            out.append({"action": action, "ticker": ticker, "dollars": dollars,
                        "rationale": o.get("rationale", "")})
    return out


# Abort a run if the model overspends this many times in a row within one period.
OVERSPEND_STREAK_LIMIT = 3


class OverspendError(RuntimeError):
    """Raised when the model overspends OVERSPEND_STREAK_LIMIT times in a row in one
    period. Signals the run should be marked an ERROR rather than silently falling
    back and completing."""


def _is_overspend_reason(reason: str) -> bool:
    """True when a rejection reason is an overspend (the model tried to buy more than
    the available cash), as opposed to a holdings-cap / bad-ticker / no-price reject."""
    r = (reason or "").lower()
    return ("exceed available cash" in r or "overdraw" in r or "cash went negative" in r)


def validate_and_execute(portfolio: Portfolio, orders: List[Dict[str, Any]],
                         prices: Dict[str, float], *, max_holdings: int,
                         candidates: List[str], whole_shares: bool
                         ) -> Tuple[bool, str, Optional[Portfolio]]:
    """
    Atomically validate a full buy/sell response on a COPY. Returns
    (ok, reason, new_portfolio). The caller commits new_portfolio only if ok.

    Rules:
      * sells applied first, then buys in order
      * cash may never go negative (sells must cover buys)
      * final holdings count <= max_holdings
      * buys only into candidate tickers (sells only from current holdings)
    """
    orders = _norm_orders(orders)
    work = copy.deepcopy(portfolio)

    sells = [o for o in orders if o["action"] == "sell"]
    buys = [o for o in orders if o["action"] == "buy"]

    # --- sells first --------------------------------------------------------
    proceeds = 0.0
    for o in sells:
        if o["ticker"] not in work.positions:
            return False, f"sell order for {o['ticker']} not currently held", None
        proceeds += work.sell(o["ticker"], o["dollars"], prices.get(o["ticker"], 0.0), whole_shares)

    available = work.cash  # already includes proceeds after sells

    # --- pre-check the budget (the binding atomic constraint) ---------------
    total_buys = sum(min(o["dollars"], _buy_cost_ceiling(o, prices, whole_shares)) for o in buys)
    if total_buys > available + 1e-4:
        overdraw = total_buys - available
        return (False,
                f"buy orders total ${total_buys:,.2f} exceed available cash "
                f"${available:,.2f} after sells (overdraw ${overdraw:,.2f})",
                None)

    # --- buys in order ------------------------------------------------------
    for o in buys:
        price = prices.get(o["ticker"], 0.0)
        if price <= 0:
            return False, f"no price available for buy of {o['ticker']}", None
        if o["ticker"] not in candidates and o["ticker"] not in work.positions:
            return False, f"buy of {o['ticker']} is not in this period's candidate set", None
        spend = min(o["dollars"], work.cash)  # guard rounding
        work.buy(o["ticker"], spend, price, whole_shares)

    if work.cash < -1e-4:
        return False, f"cash went negative (${work.cash:,.2f})", None

    if work.num_holdings() > max_holdings:
        return (False,
                f"resulting portfolio has {work.num_holdings()} holdings, "
                f"exceeds the cap of {max_holdings}", None)

    return True, "ok", work


def _buy_cost_ceiling(order: Dict[str, Any], prices: Dict[str, float], whole_shares: bool) -> float:
    """Max cash a buy can actually consume (whole-share buys may cost less than asked)."""
    price = prices.get(order["ticker"], 0.0)
    if price <= 0:
        return 0.0
    if whole_shares:
        return math.floor(order["dollars"] / price) * price
    return order["dollars"]


def safe_fallback_execute(portfolio: Portfolio, orders: List[Dict[str, Any]],
                          prices: Dict[str, float], *, max_holdings: int,
                          candidates: List[str], whole_shares: bool
                          ) -> Tuple[Portfolio, List[str]]:
    """
    Last resort after retries are exhausted: apply all sells, then buys in order,
    skipping any buy that would overdraw or breach the holdings cap. Returns the
    committed portfolio and a log of skipped orders.
    """
    orders = _norm_orders(orders)
    work = copy.deepcopy(portfolio)
    skipped: List[str] = []
    for o in [o for o in orders if o["action"] == "sell"]:
        work.sell(o["ticker"], o["dollars"], prices.get(o["ticker"], 0.0), whole_shares)
    for o in [o for o in orders if o["action"] == "buy"]:
        price = prices.get(o["ticker"], 0.0)
        cost = min(o["dollars"], _buy_cost_ceiling(o, prices, whole_shares))
        would_be_new = o["ticker"] not in work.active_tickers()
        if price <= 0 or cost > work.cash + 1e-4:
            skipped.append(f"buy {o['ticker']} ${o['dollars']:.0f} (insufficient cash)")
            continue
        if would_be_new and work.num_holdings() >= max_holdings:
            skipped.append(f"buy {o['ticker']} ${o['dollars']:.0f} (holdings cap)")
            continue
        work.buy(o["ticker"], min(o["dollars"], work.cash), price, whole_shares)
    return work, skipped


# =========================================================================== #
#  Backends                                                                    #
# =========================================================================== #
class SimBackend:
    """Interface for the data + decision provider."""

    def get_prices(self, tickers: List[str], as_of: str) -> Dict[str, float]:
        raise NotImplementedError

    def get_daily_prices(self, tickers: List[str], start: str, end: str
                         ) -> Dict[str, Dict[str, float]]:
        """{ticker: {date: price}} of daily (weekday) closes over [start, end]."""
        raise NotImplementedError

    def get_candidates(self, sector: str, as_of: str, max_holdings: int) -> List[str]:
        raise NotImplementedError

    def decide(self, *, stage: str, sector: str, as_of: str, candidates: List[str],
               prices: Dict[str, float], cash: float, total_value: float,
               max_holdings: int, holdings_report: str,
               retry_feedback: Optional[str], feedback: bool = True,
               book_report: str = "", horizon: str = "",
               prompt_framing: str = "reset") -> List[Dict[str, Any]]:
        raise NotImplementedError


# ----------------------------- Mock (dry-run) ------------------------------ #
class MockBackend(SimBackend):
    """Deterministic, no API. Exercises every engine path including reject/re-prompt."""

    def __init__(self, cash_retain: float = 0.12, force_overdraw_once: bool = True,
                 order_mode: str = "dollars"):
        self.cash_retain = cash_retain
        self.order_mode = order_mode
        # In weights mode an overdraw cannot be expressed, so the deliberate
        # overdraw test is disabled — there is no reject path left to exercise.
        if order_mode == "weights":
            force_overdraw_once = False
        self.force_overdraw_once = force_overdraw_once
        self._overdrew = False
        self.decision_log: List[Dict[str, Any]] = []

    def _base_price(self, ticker: str) -> float:
        h = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
        return 20 + (h % 9800) / 20.0          # ~$20..$510

    def _price(self, ticker: str, date: str) -> float:
        """
        Continuous, deterministic daily price: a per-ticker exponential drift plus a
        smooth seasonal wave and a small day-level wiggle. Same function backs both
        the as-of price and the daily series, so they are mutually consistent.
        """
        base = self._base_price(ticker)
        day = (_d(date) - _d("2024-01-01")).days
        h = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
        mu = ((h % 40) - 15) / 100.0 / 365.0           # annualized drift / 365, +/-
        wave = 0.05 * math.sin((day + (h % 90)) / 30.0)  # smooth seasonal component
        wig = int(hashlib.md5(f"{ticker}{date}".encode()).hexdigest(), 16) % 200
        noise = (wig - 100) / 100.0 * 0.012             # +/-1.2% daily wiggle
        price = base * math.exp(mu * day) * (1 + wave + noise)
        return round(max(1.0, price), 2)

    def get_prices(self, tickers, as_of):
        return {t: self._price(t, as_of) for t in tickers}

    def get_daily_prices(self, tickers, start, end):
        days = business_days(start, end)
        return {t: {d: self._price(t, d) for d in days} for t in tickers}

    def get_candidates(self, sector, as_of, max_holdings):
        universe = sector_config.universe()
        h = int(hashlib.md5(f"cand{as_of}".encode()).hexdigest(), 16)
        k = min(len(universe), max_holdings * 2 + 2)
        return [universe[(h + i * 5) % len(universe)] for i in range(k)]

    def decide(self, *, stage, sector, as_of, candidates, prices, cash, total_value,
               max_holdings, holdings_report, retry_feedback, feedback=True,
               book_report="", horizon="", prompt_framing="reset"):
        self.decision_log.append({"stage": stage, "as_of": as_of,
                                   "retry": retry_feedback is not None,
                                   "cash_in": round(cash, 2)})
        # BUY-ONLY: the engine already liquidated all holdings, so `cash` is the exact
        # budget. The model just chooses what to buy (up to `max_holdings`).
        picks = candidates[:max_holdings]
        if self.order_mode == "weights":
            # conviction-tilted weights that deliberately leave cash_retain uninvested
            raw = {t: (len(picks) - i) for i, t in enumerate(picks)}
            s = sum(raw.values()) or 1
            deploy = 1.0 - self.cash_retain
            return weights_to_orders({t: v / s * deploy for t, v in raw.items()},
                                     cash, max_holdings, candidates)
        if (self.force_overdraw_once and not self._overdrew
                and retry_feedback is None and stage != "initial"):
            # deliberately overspend once to exercise the reject-and-reprompt path
            self._overdrew = True
            return [{"action": "buy", "ticker": picks[0], "dollars": round(cash * 2, 2),
                     "rationale": "INTENTIONAL OVERDRAW (test)"}]
        deploy = cash * (1 - self.cash_retain)
        per = deploy / len(picks) if picks else 0
        return [{"action": "buy", "ticker": t, "dollars": round(per, 2),
                 "rationale": "buy from liquidated cash"} for t in picks]

    # the engine feeds current state to the mock so it can craft valid orders
    def sync_state(self, portfolio: Portfolio):
        self._held_set = set(portfolio.active_tickers())
        self._shares = {t: portfolio.positions[t].shares for t in self._held_set}


# ----------------------------- Real (live) --------------------------------- #
class RealBackend(SimBackend):
    """
    Cash-aware decisions on top of InvestmentStrategyGenerator's data helpers.
    Makes OpenAI calls. Requires OPENAI_API_KEY (never reads source-embedded keys).
    """

    # Paper B only: when screened_news is on, Stage One reads the LLM-relevance-
    # screened corpus (screen_news_relevance.py) instead of the raw merged file,
    # and skips the sector-keyword substring filter that would otherwise throw
    # away ~97% of it. Paper A's permutation_runner is untouched by this.
    SCREENED_CORPUS = os.path.join(ROOT, "tech_screened_corpus.csv")

    def __init__(self, model: str = "gpt-5.1", news_lookback_years: int = 2,
                 screened_news: bool = False, order_mode: str = "dollars",
                 screened_corpus: Optional[str] = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("RealBackend needs OPENAI_API_KEY in the environment. "
                               "Use --dry-run to validate with no API calls.")
        from investment_strategy_generator import InvestmentStrategyGenerator
        self.gen = InvestmentStrategyGenerator(
            api_key_openai=api_key, nyt_api_key=os.environ.get("NYT_API_KEY"), model=model)
        self.model = model
        self.news_lookback_years = news_lookback_years
        self.screened_news = bool(screened_news)
        if order_mode not in ("dollars", "weights"):
            raise ValueError(f"order_mode must be 'dollars' or 'weights', got {order_mode!r}")
        self.order_mode = order_mode
        # Per-run override so a threshold sweep can point Stage One at any scored
        # corpus (tech_scored_ge4.csv ... ge8.csv from score_news_relevance.py)
        # without touching the class default. Shadows the class attribute.
        if screened_corpus:
            self.SCREENED_CORPUS = (screened_corpus if os.path.isabs(screened_corpus)
                                    else os.path.join(ROOT, screened_corpus))
        if self.screened_news and not os.path.exists(self.SCREENED_CORPUS):
            raise RuntimeError(
                f"screened_news=on but {os.path.basename(self.SCREENED_CORPUS)} is missing. "
                f"Run `python screen_news_relevance.py` (or score_news_relevance.py "
                f"--emit N) first.")
        self._chat: List[Dict[str, str]] = []      # running chat for the CURRENT period
        self._last_assistant_content = ""          # raw text of the model's last reply

    # ---- data ------------------------------------------------------------- #
    def get_prices(self, tickers, as_of):
        """As-of price per ticker = last close on/before as_of. Uses the same reliable
        chart-fetch path as get_daily_prices (the temporal helper intermittently returns
        empty, which previously erased gains at rebalances)."""
        from datetime import timedelta
        prices = {}
        e_dt = _d(as_of)
        s_dt = e_dt - relativedelta(days=14)     # small window to grab the last close on/before as_of
        for t in tickers:
            px = None
            try:
                hist = self.gen._fetch_yahoo_chart_history(t, s_dt, e_dt + timedelta(days=1))
                if hist is not None and not hist.empty:
                    px = float(hist["Close"].iloc[-1])
            except Exception as e:        # noqa: BLE001
                print(f"   price fetch (chart) failed for {t}: {e}")
            if px is None:                # fallback to the temporal helper
                try:
                    hist_1y, _, _ = self.gen._load_temporal_price_histories(t, as_of)
                    if hist_1y is not None and not hist_1y.empty:
                        px = float(hist_1y["Close"].iloc[-1])
                except Exception as e:    # noqa: BLE001
                    print(f"   price fetch (temporal) failed for {t}: {e}")
            if px is not None and px > 0:
                prices[t] = round(px, 2)
        return prices

    def get_daily_prices(self, tickers, start, end):
        """Daily adjusted closes per ticker over [start, end] via the generator's chart history."""
        from datetime import timedelta
        out: Dict[str, Dict[str, float]] = {}
        s_dt, e_dt = _d(start), _d(end)
        for t in tickers:
            series: Dict[str, float] = {}
            try:
                hist = self.gen._fetch_yahoo_chart_history(t, s_dt, e_dt + timedelta(days=1))
                if hist is not None and not hist.empty:
                    for idx, row in hist.iterrows():
                        d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                        series[d] = round(float(row["Close"]), 4)
            except Exception as e:        # noqa: BLE001
                print(f"   daily price fetch failed for {t}: {e}")
            out[t] = series
        return out

    def _screened_window(self, news_start: str, as_of: str):
        """Screened articles published in [news_start, as_of]. Read directly rather
        than through _load_news_data, whose workspace walk would pull the raw
        corpus back in alongside the screened one."""
        import pandas as pd
        df = pd.read_csv(self.SCREENED_CORPUS)
        dt = pd.to_datetime(df["pub_date"], errors="coerce")
        win = df[(dt >= pd.Timestamp(news_start)) & (dt <= pd.Timestamp(as_of))]
        return win.reset_index(drop=True)

    def get_candidates(self, sector, as_of, max_holdings):
        news_start = _s(_d(as_of) - relativedelta(years=self.news_lookback_years))
        if self.screened_news:
            news_df = self._screened_window(news_start, as_of)
            print(f"   [screened news] {len(news_df)} articles in {news_start}..{as_of}")
        else:
            news_df = self.gen._load_news_data(sector_config.news_file(ROOT),
                                               news_start, as_of, sector)
        if news_df is None or len(news_df) == 0:
            return []
        # Screened articles already passed a relevance test, so the sector-keyword
        # substring filter is skipped (passing "" keeps every article).
        articles = self.gen._process_news_data(news_df, "" if self.screened_news else sector)
        model = self.gen._validate_model_for_date_range(as_of)
        prompt = f"""
You are screening the {sector} sector for a long-term portfolio as of {as_of}.
Using ONLY the news below (information available on/before {as_of}), list the
{max(8, max_holdings * 2)} strongest candidate companies.

Ticker rules (STRICT):
- Only common stocks listed on a U.S. exchange (NYSE, NASDAQ, or NYSE American).
- Use the exact U.S. root ticker (e.g. AAPL, MSFT, JPM); no exchange suffixes
  (no ".L", ".MI", ".DE", ".SW", ".PA", etc.).
- EXCLUDE ETFs and funds (including UCITS/foreign-domiciled funds), ADRs of
  foreign companies, OTC/pink-sheet names, indices, and any non-U.S. listing.
- If a company is not directly investable as a U.S.-listed common stock, skip it.

NEWS:
{articles}

Return ONLY JSON: a plain list of ticker symbols and nothing else (no rationale,
no other fields): {{"candidates": ["AAPL", "MSFT"]}}
"""
        from investment_strategy_generator import _oai_create_with_retry, extract_candidate_tickers
        # Retry a few times: a malformed/truncated screening response is usually
        # transient. If it still fails, skip this period (hold cash) rather than
        # crashing the whole multi-period run.
        for attempt in range(3):
            try:
                resp = _oai_create_with_retry(self.gen.openai_client, model=model,
                                              messages=[{"role": "user", "content": prompt}],
                                              **self.gen._temp_kwargs(model), max_completion_tokens=1500)
                data = self.gen._load_json_response(resp.choices[0].message.content,
                                                    "candidate screening", model, max_completion_tokens=1500)
                return extract_candidate_tickers(data)
            except Exception as e:      # noqa: BLE001
                print(f"   candidate screening attempt {attempt + 1}/3 failed: {e}")
        print("   candidate screening failed after 3 attempts; skipping this period (holding cash)")
        return []

    def _financials_block(self, tickers, as_of, prices):
        raw = self.gen._fetch_financial_data_for_tickers(tickers, max_tickers=None)
        fin = {fd.get("ticker"): self.gen._filter_financial_data_as_of(fd, as_of) for fd in raw if fd}
        lines = []
        for t in tickers:
            try:
                h1, h3, h5 = self.gen._load_temporal_price_histories(t, as_of)
                perf = self.gen._calculate_performance_metrics(h1, h3, h5) if h1 is not None else {}
                summary = self.gen._generate_stock_summary(t, {
                    "name": t, "performance_data": perf, "financial_data": fin.get(t),
                    "info": {}, "suppress_earnings_history": True})
                lines.append(f"{t} | current price ${prices.get(t, 'n/a')}\n{summary}")
            except Exception as e:        # noqa: BLE001
                lines.append(f"{t} | current price ${prices.get(t, 'n/a')} (data error: {e})")
        return "\n" + ("\n" + "-" * 30 + "\n").join(lines)

    # ---- decision --------------------------------------------------------- #
    def decide(self, *, stage, sector, as_of, candidates, prices, cash, total_value,
               max_holdings, holdings_report, retry_feedback, feedback=True,
               book_report="", horizon="", prompt_framing="reset"):
        model = self.gen._validate_model_for_date_range(as_of)
        from investment_strategy_generator import _oai_create_with_retry

        # RETRY = continue the SAME chat: show the model its own failing reply, then a
        # correction quoting exactly what was wrong (incl. the overdraw amount).
        if retry_feedback is not None and getattr(self, "_chat", None):
            self._chat.append({"role": "assistant",
                               "content": self._last_assistant_content or "(no content returned)"})
            self._chat.append({"role": "user", "content": (
                f"That response was REJECTED: {retry_feedback}\n"
                f"You have EXACTLY ${cash:,.2f} to spend and NOT ONE DOLLAR MORE. Revise your\n"
                f"orders so the SUM of every order's 'dollars' is <= ${cash:,.2f}, holding at most\n"
                f"{max_holdings} stocks. Keep the picks you still want — just cut or shrink orders\n"
                f"until the total fits. Return the corrected JSON only.")})
            resp = _oai_create_with_retry(self.gen.openai_client, model=model,
                                          messages=self._chat,
                                          **self.gen._temp_kwargs(model), max_completion_tokens=4000)
            content = resp.choices[0].message.content
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                print(f"  WARNING: model reply truncated (finish_reason=length) at {as_of} [retry]")
            self._last_assistant_content = content
            data = self.gen._load_json_response(content, "buy/sell orders (retry)", model,
                                                max_completion_tokens=4000)
            return data.get("orders", [])

        # FIRST attempt this period: build the initial prompt and start a new chat.
        fin_block = self._financials_block(candidates, as_of, prices)
        if stage == "initial":
            task = f"""
This is the INITIAL BUY (as of {as_of}). You have EXACTLY ${cash:,.2f} in cash and no holdings.
Buy UP TO {max_holdings} stocks (fewer is fine). You need NOT spend all the cash —
keeping some uninvested is fine.
"""
        else:
            # feedback ON: show realized performance of the just-sold holdings (the
            # self-reflection signal). feedback OFF (control): omit it entirely so the
            # rebalance is a fresh reselection with no performance memory.
            if prompt_framing in ("incumbent", "incumbent_invested"):
                # DEFAULT-KEEP framing. The reset wording below tells the model its
                # book "has already been SOLD" and asks it to "rebuild from scratch",
                # which removes any incumbency and structurally invites rotation —
                # the measured swap edge is -1.6 to -2.6pp (dropped names outrun
                # added ones). It is also simply untrue under rebalance_mode="hold",
                # where only the delta is traded. This branch states the real book.
                #
                # feedback ON  -> book shown WITH per-name performance
                # feedback OFF -> book shown as STATE ONLY, so the control arm still
                #                 gets no performance signal.
                if feedback:
                    book_block = (f"\nYOUR CURRENT HOLDINGS (NOT sold — you own these "
                                  f"right now), with how they have done:\n{holdings_report}\n")
                else:
                    book_block = (f"\nYOUR CURRENT HOLDINGS (NOT sold — you own these "
                                  f"right now):\n{book_report}\n")
                horizon_block = (f"\nYour next opportunity to change this portfolio is in "
                                 f"{horizon}. A name you drop today you cannot re-buy "
                                 f"until then, and a name you keep compounds for that "
                                 f"whole span.\n") if horizon else ""
                # "incumbent" alone produced a 7.06% average cash weight (vs 1.52%
                # under reset) because the switching hurdle discouraged BUYING as
                # much as selling: adds fell 76% while drops fell only 59%, so the
                # model sold winners straight to cash. That drag cost ~5pp of ROI
                # and swamped the framing's real gains (best selection skill of any
                # arm, rotation halved). "incumbent_invested" adds an explicit
                # stay-invested instruction and withdraws the cash blessing in the
                # weights rules below.
                invested_block = ("""
Stay fully invested. Your target weights should sum to approximately 1.0. Cash
earns nothing here, so do NOT use it to avoid a decision: if you drop a name,
redeploy that weight into the names you are keeping or into a replacement you
believe in. Holding a name is a decision; parking its weight in cash is not.
""" if prompt_framing == "incumbent_invested" else "")
                task = f"""
This is a REBALANCE (as of {as_of}). You are MANAGING AN EXISTING PORTFOLIO worth
${cash:,.2f} — you are not starting from cash.
{book_block}{horizon_block}
State the portfolio you want to hold going forward, as target weights over at most
{max_holdings} names. Anything you leave unchanged simply stays as it is.

Switching is not free: replacing a holding forfeits whatever it would have gone on
to earn. Only swap a name out if your conviction in the replacement is MATERIALLY
higher — not merely equal, and not because the incumbent has already risen. If in
doubt, keep what you own.
{invested_block}"""
            else:
                perf_block = (f"\nHOW YOUR PREVIOUS HOLDINGS PERFORMED (for context; they are "
                              f"already sold):\n{holdings_report}\n") if feedback else ""
                task = f"""
This is a REBALANCE (as of {as_of}). Your entire previous portfolio has already been
SOLD, so you now hold EXACTLY ${cash:,.2f} in cash and no positions. Rebuild the
portfolio from scratch: buy UP TO {max_holdings} stocks. Keeping some cash uninvested is fine.
{perf_block}"""
        # ---- weights mode: the model allocates fractions, never dollars ------ #
        # No budget rule, no arithmetic, no reject/re-prompt loop: weights_to_orders
        # converts intent into orders that cannot overdraw by construction.
        if self.order_mode == "weights":
            # The cash rule is framing-dependent. The default wording explicitly
            # blesses leaving weight uninvested ("perfectly acceptable"), which is
            # fine on its own but interacts badly with the incumbent switching
            # hurdle — together they drove average cash from 1.52% to 7.06%.
            cash_rule = ("""  - The weights must sum to AT MOST 1.0, and should sum to APPROXIMATELY 1.0.
    Any shortfall is held as cash, which earns nothing — leave weight uninvested
    only if you have a specific reason, never as a way of avoiding a choice."""
                         if prompt_framing == "incumbent_invested" else
                         """  - The weights must sum to AT MOST 1.0. If they sum to less than 1.0, the
    remainder is simply held as cash, which is perfectly acceptable.""")
            wprompt = f"""
You are managing a {sector}-sector portfolio as of {as_of}.
Use ONLY information available on/before {as_of}.
{task}
CANDIDATES (with current share price and financial reports):
{fin_block}

Allocate the portfolio by TARGET WEIGHT — fractions of the total portfolio, not
dollars. Do NOT compute any dollar amounts; the execution system handles that.

Rules:
  - Pick AT MOST {max_holdings} tickers, all from the candidate list above.
  - Each weight is a decimal fraction between 0 and 1.
{cash_rule}
  - Size positions by conviction: a stronger view deserves a larger weight.

Return ONLY JSON:
{{"weights": {{"TICKER": 0.15, "TICKER2": 0.10}},
  "rationale": "one or two sentences on the sizing logic"}}
"""
            self._chat = [{"role": "user", "content": wprompt}]
            resp = _oai_create_with_retry(self.gen.openai_client, model=model,
                                          messages=self._chat,
                                          **self.gen._temp_kwargs(model),
                                          max_completion_tokens=4000)
            content = resp.choices[0].message.content
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                print(f"  WARNING: model reply truncated (finish_reason=length) at {as_of}")
            self._last_assistant_content = content
            data = self.gen._load_json_response(content, "target weights", model,
                                                max_completion_tokens=4000)
            # Stash the raw weights: rebalance_mode="hold" needs the model's INTENT
            # (which names, at what conviction) without the dollar resizing that
            # weights_to_orders bakes in.
            self.last_weights = dict(data.get("weights") or {})
            orders = weights_to_orders(data.get("weights") or {}, cash,
                                       max_holdings, candidates)
            spend = sum(o["dollars"] for o in orders)
            print(f"   [weights] {len(orders)} positions, deploying "
                  f"${spend:,.2f} of ${cash:,.2f} ({spend / cash * 100:.1f}%)"
                  if cash else "   [weights] no cash")
            return orders

        budget_rule = f"""
============================ HARD BUDGET RULE ============================
Your budget is EXACTLY ${cash:,.2f}. This is the ONLY money you have.
  1. The SUM of the "dollars" fields across ALL of your buy orders MUST be
     LESS THAN OR EQUAL TO ${cash:,.2f}.
  2. You may NOT borrow, use margin, or spend even one dollar over ${cash:,.2f}.
  3. Before you answer, ADD UP the dollars of every order and CONFIRM the
     total is <= ${cash:,.2f}. If it is over, remove or shrink orders first.
  4. If the total exceeds ${cash:,.2f}, the ENTIRE response is REJECTED and you
     will have to redo it — so do not exceed it.
  5. Leaving some cash uninvested is completely acceptable. Going over budget is NOT.
=========================================================================
"""
        prompt = f"""
You are managing a real ${cash:,.2f} cash budget in the {sector} sector, as of {as_of}.
Use ONLY information available on/before {as_of}.
{budget_rule}{task}
CANDIDATES (with current share price and financial reports):
{fin_block}

Issue BUY orders only, and make sure their dollar amounts SUM TO <= ${cash:,.2f}.
Return ONLY JSON:
{{"orders": [{{"action": "buy", "ticker": "TICKER", "dollars": 1500, "rationale": "..."}}],
  "cash_plan": "confirm the order dollars sum to <= your budget, and why you kept any cash uninvested"}}
"""
        self._chat = [{"role": "user", "content": prompt}]
        resp = _oai_create_with_retry(self.gen.openai_client, model=model,
                                      messages=self._chat,
                                      **self.gen._temp_kwargs(model), max_completion_tokens=4000)
        content = resp.choices[0].message.content
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            print(f"  WARNING: model reply truncated (finish_reason=length) at {as_of} — raise max_completion_tokens")
        self._last_assistant_content = content
        data = self.gen._load_json_response(content, "buy/sell orders", model,
                                            max_completion_tokens=4000)
        return data.get("orders", [])

    def sync_state(self, portfolio: Portfolio):
        pass  # real backend reads state from the prompt; nothing to cache


# =========================================================================== #
#  Holdings performance report (injected at the financial step)               #
# =========================================================================== #
def weights_to_orders(weights: Dict[str, Any], cash: float, max_holdings: int,
                      candidates: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Turn model-emitted target weights into dollar buy orders. Deterministic.

    This is the whole point of order_mode="weights": the LLM never handles money.
    It expresses intent as fractions; the arithmetic happens here, where it cannot
    be wrong. Overdraw is structurally impossible because the weights are scaled
    to at most 1.0 before being multiplied by the available cash.

    Rules, in order:
      * drop non-numeric, non-positive, and non-candidate tickers
      * keep the `max_holdings` largest weights (the model may over-list)
      * if the weights sum to more than 1, scale them DOWN to 1
        (if they sum to less, the remainder is deliberate cash — honored as-is,
        preserving the model's ability to stay partly uninvested)
    """
    cand = set(candidates) if candidates else None
    clean: Dict[str, float] = {}
    for t, w in (weights or {}).items():
        if not isinstance(t, str):
            continue
        tick = t.strip().upper()
        if tick in ("CASH", "USD", ""):        # explicit cash sleeve: just omit it
            continue
        if cand is not None and tick not in cand:
            continue
        try:
            wv = float(w)
        except (TypeError, ValueError):
            continue
        if wv > 0 and math.isfinite(wv):
            clean[tick] = clean.get(tick, 0.0) + wv

    if not clean:
        return []
    # If the model emitted percentages (0-100) rather than fractions, rescale.
    total = sum(clean.values())
    if total > 1.5:
        clean = {t: w / total for t, w in clean.items()}
        total = 1.0

    top = dict(sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:max_holdings])
    total = sum(top.values())
    if total > 1.0:
        top = {t: w / total for t, w in top.items()}

    # 1e-6 haircut absorbs float error so the sum can never round above cash.
    budget = float(cash) * (1.0 - 1e-6)
    return [{"action": "buy", "ticker": t, "dollars": round(w * budget, 2),
             "rationale": f"target weight {w:.3f}"}
            for t, w in top.items() if w * budget >= 0.01]


def holdings_book_report(portfolio: Portfolio, prices_now: Dict[str, float]) -> str:
    """Current book as STATE only — names, shares, value, portfolio weight.

    Deliberately reports NO return figures. `prompt_framing="incumbent"` shows the
    model what it owns even when feedback is OFF; if that block carried performance
    it would hand the control arm the very signal feedback is supposed to provide
    and destroy the ON/OFF comparison. Performance lives in
    `holdings_performance_report` and is shown only when feedback is on.
    """
    if portfolio.num_holdings() == 0:
        return "(no current holdings)"
    total = portfolio.total_value(prices_now) or 1.0
    lines = []
    for t in portfolio.active_tickers():
        pos = portfolio.positions[t]
        pnow = prices_now.get(t, pos.avg_cost)
        val = pos.shares * pnow
        lines.append(f"  - {t}: {pos.shares:.3f} sh @ ${pnow:.2f} "
                     f"= ${val:,.2f}  ({val / total * 100:.1f}% of portfolio)")
    return "\n".join(lines)


def horizon_label(as_of: str, next_as_of: Optional[str]) -> str:
    """Human-readable gap to the next decision, for prompt_framing='incumbent'.

    The model is otherwise blind to cadence: a semiannual and a monthly decision
    read identically to it, yet the measured cost of a bad swap differs by roughly
    6x (2x cadence ~ -4 to -6.5pp per decision, 12x ~ 0).
    """
    if not next_as_of:
        return ""
    try:
        d0 = datetime.strptime(as_of[:10], "%Y-%m-%d")
        d1 = datetime.strptime(next_as_of[:10], "%Y-%m-%d")
    except Exception:
        return ""
    days = (d1 - d0).days
    if days <= 0:
        return ""
    months = days / 30.44
    span = f"{months:.0f} months" if months >= 1.5 else f"{days} days"
    return f"{span} (next decision {next_as_of[:10]})"


def holdings_performance_report(portfolio: Portfolio, prices_now: Dict[str, float],
                                prices_prev: Dict[str, float]) -> str:
    if portfolio.num_holdings() == 0:
        return "(no current holdings)"
    lines = []
    for t in portfolio.active_tickers():
        pos = portfolio.positions[t]
        pnow = prices_now.get(t, pos.avg_cost)
        val = pos.shares * pnow
        unreal = (pnow / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0.0
        if t in prices_prev and prices_prev[t] > 0:
            period = (pnow / prices_prev[t] - 1) * 100
            ptxt = f"{period:+.1f}% since last period"
        else:
            ptxt = "new this period"
        lines.append(f"  - {t}: {pos.shares:.3f} sh, avg cost ${pos.avg_cost:.2f}, "
                     f"now ${pnow:.2f} (value ${val:,.2f}), "
                     f"unrealized {unreal:+.1f}%, {ptxt}")
    return "\n".join(lines)


# =========================================================================== #
#  Daily value-series reconstruction (for risk metrics)                       #
# =========================================================================== #
def _forward_filled(daily_map: Dict[str, float], dates: List[str]) -> List[Optional[float]]:
    """For each target date, the most recent available price <= date (leading gap
    back-filled with the first available price)."""
    avail = sorted(daily_map.items())
    if not avail:
        return [None] * len(dates)
    out, j, last = [], 0, None
    for d in dates:
        while j < len(avail) and avail[j][0] <= d:
            last = avail[j][1]
            j += 1
        out.append(last if last is not None else avail[0][1])
    return out


def reconstruct_daily_series(backend: SimBackend,
                             segment_states: List[Dict[str, Any]]
                             ) -> Tuple[List[str], List[float]]:
    """
    Build a daily portfolio value series. Within each segment the share counts are
    fixed, so value(d) = cash + Σ shares_t × adj_close_t(d). Segments are stitched
    end-to-end (each segment owns [start, end]; the final segment includes end_date).
    """
    dates_out: List[str] = []
    values_out: List[float] = []
    seen = set()
    for k, st in enumerate(segment_states):
        last_seg = (k == len(segment_states) - 1)
        dates = business_days(st["start"], st["end"], inclusive_end=last_seg)
        tickers = list(st["positions"].keys())
        daily = backend.get_daily_prices(tickers, st["start"], st["end"]) if tickers else {}
        ff = {t: _forward_filled(daily.get(t, {}), dates) for t in tickers}
        for di, d in enumerate(dates):
            if d in seen:
                continue
            seen.add(d)
            val = st["cash"]
            for t in tickers:
                px = ff[t][di]
                if px is not None:
                    val += st["positions"][t] * px
            dates_out.append(d)
            values_out.append(round(val, 4))
    return dates_out, values_out


def benchmark_series_for(backend: SimBackend, benchmark_ticker: str,
                         dates: List[str]) -> List[float]:
    """Benchmark close aligned (forward-filled) to the portfolio's daily dates."""
    if not dates:
        return []
    daily = backend.get_daily_prices([benchmark_ticker], dates[0], dates[-1])
    ff = _forward_filled(daily.get(benchmark_ticker, {}), dates)
    return [p for p in ff if p is not None]


# =========================================================================== #
#  Engine                                                                     #
# =========================================================================== #
def _liquidate_all(portfolio: "Portfolio", prices: Dict[str, float]) -> float:
    """Sell every holding at current prices so cash == full portfolio value. Returns proceeds.
    This makes each period a clean BUY-only decision from a known, exact budget."""
    proceeds = 0.0
    for t in list(portfolio.active_tickers()):
        pos = portfolio.positions[t]
        price = prices.get(t, pos.avg_cost)
        proceeds += pos.shares * price
        portfolio.cash += pos.shares * price
        pos.shares = 0.0
    return proceeds


def normalize_target_book(weights: Dict[str, Any], max_holdings: int,
                          candidates: Optional[List[str]] = None) -> Dict[str, float]:
    """Clean model weights into a target book {ticker: weight}. Same cleaning rules
    as weights_to_orders, but returns intent rather than dollar orders."""
    cand = set(candidates) if candidates else None
    clean: Dict[str, float] = {}
    for t, w in (weights or {}).items():
        if not isinstance(t, str):
            continue
        tick = t.strip().upper()
        if tick in ("CASH", "USD", ""):
            continue
        if cand is not None and tick not in cand:
            continue
        try:
            wv = float(w)
        except (TypeError, ValueError):
            continue
        if wv > 0 and math.isfinite(wv):
            clean[tick] = clean.get(tick, 0.0) + wv
    if not clean:
        return {}
    return dict(sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:max_holdings])


def execute_hold_mode(portfolio: "Portfolio", weights: Dict[str, Any],
                      prices: Dict[str, float], max_holdings: int,
                      candidates: Optional[List[str]], whole_shares: bool,
                      redeploy: bool = False
                      ) -> Tuple["Portfolio", Dict[str, Any]]:
    """Rebalance WITHOUT resizing the positions the model chose to keep.

    The default path liquidates 100% and re-buys to fresh target weights, so a name
    that compounded to 25% of the book is reset to ~10% every period. That truncates
    exactly the right tail that produces the returns (strategy skew +0.72 vs INIT
    +1.88; 71% upside capture when INIT lands in its best quartile).

    Here the model's weights are read as INTENT — which names to own — and only the
    delta is traded:
      * names it dropped      -> sold in full
      * names it kept         -> untouched, left to keep compounding
      * names it added        -> bought with the proceeds + free cash, pro-rata to weight

    Overdraw is impossible: entry weights are normalized to 1 and multiplied by cash
    already in hand after the exits settle.

    `redeploy` (rebalance_mode="hold_redeploy") fixes a real defect in the plain
    path: exit proceeds are only ever spent on names NOT already held, so a decision
    that drops a name without adding one strands its proceeds as idle cash forever.
    Measured over 540 decisions per arm: 37% of hold_incumbent's decisions took that
    path and left 15.4% average cash against 2.1% elsewhere; even hold_winners hit it
    5% of the time (6.6% cash vs 0.01%). No prompt wording can reach this — the model
    has no way to say "put it into what I already own".

    With redeploy on, leftover cash tops up KEPT positions that sit BELOW their target
    weight, pro-rata to the shortfall. It never trims, so the let-winners-run property
    that hold mode exists for is preserved: a position that compounded past its target
    is simply left alone. Cash the model deliberately asked for (weights summing to
    less than 1) still survives, because a book already at target has no shortfall.
    """
    top = normalize_target_book(weights, max_holdings, candidates)
    target = set(top)
    held = set(portfolio.active_tickers())

    exits = sorted(held - target)
    for t in exits:
        pos = portfolio.positions[t]
        px = prices.get(t, pos.avg_cost)
        portfolio.sell(t, pos.shares * px, px, whole_shares)

    entries = [t for t in top if t not in held and prices.get(t, 0) > 0]
    cash_avail = max(portfolio.cash, 0.0)
    wsum = sum(top[t] for t in entries)
    spent = 0.0
    for t in entries:
        share = (top[t] / wsum) if wsum > 0 else (1.0 / len(entries))
        spent += portfolio.buy(t, cash_avail * share, prices[t], whole_shares)

    topped = 0.0
    topped_names: List[str] = []
    if redeploy:
        cash_left = max(portfolio.cash, 0.0)
        total_value = portfolio.total_value(prices)
        kept_now = [t for t in top if t in portfolio.active_tickers()
                    and prices.get(t, 0) > 0]
        # dollar shortfall of each kept name against its target weight
        short = {}
        for t in kept_now:
            cur = portfolio.positions[t].shares * prices[t]
            want = top[t] * total_value
            if want > cur:
                short[t] = want - cur
        ssum = sum(short.values())
        if cash_left > 0.01 and ssum > 0:
            for t, need in sorted(short.items(), key=lambda kv: -kv[1]):
                amt = min(need, cash_left * (need / ssum))
                if amt > 0.01:
                    got = portfolio.buy(t, amt, prices[t], whole_shares)
                    if got > 0:
                        topped += got
                        topped_names.append(t)

    return portfolio, {
        "mode": "hold_redeploy" if redeploy else "hold",
        "kept": sorted(held & target),
        "sold": exits,
        "bought": entries,
        "redeployed": round(spent, 2),
        "topped_up": sorted(topped_names),
        "topped_up_amount": round(topped, 2),
    }


def run_simulation(backend: SimBackend, *, sector: str, budget: float,
                   segments: List[Dict[str, str]], max_holdings: int,
                   whole_shares: bool, max_retries: int = 2,
                   rf_annual: float = 0.0, compute_risk: bool = True,
                   feedback: bool = True, txn_cost_bps: float = 0.0,
                   rebalance_mode: str = "reset",
                   prompt_framing: str = "reset",
                   weighting: str = "model") -> Dict[str, Any]:
    portfolio = Portfolio(cash=float(budget))
    prices_prev: Dict[str, float] = {}
    last_prices: Dict[str, float] = {}          # last KNOWN price per ticker (carry-forward)
    period_log: List[Dict[str, Any]] = []
    segment_states: List[Dict[str, Any]] = []   # exact post-trade state per segment

    for i, seg in enumerate(segments):
        as_of = seg["start"]
        stage = "initial" if i == 0 else "rebalance"

        candidates = backend.get_candidates(sector, as_of, max_holdings)
        price_tickers = sorted(set(candidates) | set(portfolio.active_tickers()))
        prices = backend.get_prices(price_tickers, as_of)

        # Carry forward the last known price for any HELD ticker we could not price now,
        # so a transient fetch miss never liquidates the portfolio at cost basis
        # (which silently erases gains/losses).
        for t in portfolio.active_tickers():
            if (t not in prices or prices[t] <= 0) and t in last_prices:
                prices[t] = last_prices[t]
        last_prices.update({t: pv for t, pv in prices.items() if pv > 0})

        # Only offer the model candidates we can actually price (avoids "no price" rejects).
        candidates = [c for c in candidates if prices.get(c, 0) > 0]

        # value BEFORE trading (mark current holdings at today's prices)
        value_before = portfolio.total_value(prices)
        holdings_report = holdings_performance_report(portfolio, prices, prices_prev)
        # State-only book + gap to the next decision, for prompt_framing="incumbent".
        # Built BEFORE any liquidation, so they describe what the model actually owns.
        book_report = holdings_book_report(portfolio, prices)
        next_as_of = (segments[i + 1]["start"] if i + 1 < len(segments)
                      else seg.get("end"))
        horizon = horizon_label(as_of, next_as_of)
        # structured per-name realized performance of the just-ended period (for auditing
        # the feedback loop: did the model chase winners / drop losers?). Captured BEFORE
        # liquidation empties the positions.
        prior_holdings_perf = {t: {
            "ret_since_prev_pct": round((prices.get(t, portfolio.positions[t].avg_cost) / prices_prev[t] - 1) * 100, 2)
                                  if prices_prev.get(t, 0) > 0 else None,
            "unrealized_pct": round((prices.get(t, portfolio.positions[t].avg_cost) / portfolio.positions[t].avg_cost - 1) * 100, 2)
                              if portfolio.positions[t].avg_cost > 0 else None,
        } for t in portfolio.active_tickers()}

        # rebalance_mode="hold" only makes sense when the model expresses intent as
        # weights; dollar orders carry no reusable notion of a target book.
        hold_mode = (rebalance_mode in ("hold", "hold_redeploy") and stage == "rebalance"
                     and getattr(backend, "order_mode", "dollars") == "weights")

        # SELL EVERYTHING FIRST: liquidate all holdings so the model faces a clean,
        # exact cash budget and only decides what to BUY (no mixed buy/sell budget math).
        # Skipped in hold mode, where kept positions are deliberately left to compound.
        if not hold_mode:
            _liquidate_all(portfolio, prices)   # cash now == value_before, no positions

        if hasattr(backend, "sync_state"):
            backend.sync_state(portfolio)

        # ---- decision with atomic validation + reject/re-prompt ----------- #
        retry_feedback = None
        committed = None
        attempts = 0
        rejects: List[str] = []
        overspend_streak = 0
        hold_info: Optional[Dict[str, Any]] = None
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            # In hold mode the book is NOT liquidated, so portfolio.cash is only the
            # idle sleeve. The model still allocates over the whole portfolio, so it
            # is shown value_before — the same notional it sees in reset mode.
            orders = backend.decide(
                stage=stage, sector=sector, as_of=as_of, candidates=candidates,
                prices=prices, cash=(value_before if hold_mode else portfolio.cash),
                total_value=value_before,
                max_holdings=max_holdings, holdings_report=holdings_report,
                retry_feedback=retry_feedback, feedback=feedback,
                book_report=book_report, horizon=horizon,
                prompt_framing=prompt_framing)
            # ---- position sizing: model | equal | black_litterman ------------
            # The model's own conviction weights measurably destroy value versus
            # 1/N on the SAME picks (+4.08pp Paper A p=5e-10, +11.42pp Paper B
            # p=3e-24). This re-sizes the book the model chose, without touching
            # WHICH names it chose. black_litterman follows arXiv:2504.14345:
            # LLM supplies views, an optimiser supplies weights.
            if (weighting != "model"
                    and getattr(backend, "order_mode", "dollars") == "weights"
                    and getattr(backend, "last_weights", None)):
                import black_litterman as _bl
                hist = {}
                if weighting == "black_litterman":
                    # Covariance from prices ending STRICTLY BEFORE as_of.
                    try:
                        lb = _s(_d(as_of) - relativedelta(months=15))
                        prior_day = _s(_d(as_of) - relativedelta(days=1))
                        daily = backend.get_daily_prices(
                            sorted(backend.last_weights), lb, prior_day)
                        hist = {t: [v for _, v in sorted(series.items())]
                                for t, series in (daily or {}).items()}
                    except Exception as e:              # noqa: BLE001
                        print(f"   [weighting] history fetch failed ({e}); "
                              f"falling back to equal weight")
                new_w = _bl.apply_weighting(weighting, backend.last_weights, hist)
                if new_w:
                    backend.last_weights = new_w
                    if not hold_mode:
                        orders = weights_to_orders(new_w, portfolio.cash,
                                                   max_holdings, candidates)
                    print(f"   [weighting={weighting}] {len(new_w)} names, "
                          f"max weight {max(new_w.values()):.3f}")

            if hold_mode:
                # Deterministic delta execution: no budget validation needed because
                # only cash already in hand is ever deployed.
                committed, hold_info = execute_hold_mode(
                    portfolio, getattr(backend, "last_weights", {}) or {}, prices,
                    max_holdings, candidates, whole_shares,
                    redeploy=(rebalance_mode == "hold_redeploy"))
                extra = (f" topped-up {hold_info['topped_up']} "
                         f"${hold_info['topped_up_amount']:,.2f}"
                         if hold_info.get("topped_up") else "")
                print(f"   [{hold_info['mode']}] kept {len(hold_info['kept'])} "
                      f"sold {hold_info['sold']} bought {hold_info['bought']} "
                      f"redeployed ${hold_info['redeployed']:,.2f}{extra}")
                break
            ok, reason, new_pf = validate_and_execute(
                portfolio, orders, prices, max_holdings=max_holdings,
                candidates=candidates, whole_shares=whole_shares)
            if ok:
                committed = new_pf
                break
            rejects.append(reason)
            retry_feedback = reason
            # Abort the whole run if the model overspends OVERSPEND_STREAK_LIMIT times
            # in a row: it never respects the cash budget, so mark the run an ERROR
            # instead of silently falling back to a safe (skipped-orders) execution.
            if _is_overspend_reason(reason):
                overspend_streak += 1
                if overspend_streak >= OVERSPEND_STREAK_LIMIT:
                    raise OverspendError(
                        f"period {i} ({as_of}): model overspent {overspend_streak} "
                        f"times in a row (last: {reason})")
            else:
                overspend_streak = 0
            if hasattr(backend, "sync_state"):
                backend.sync_state(portfolio)

        skipped: List[str] = []
        if committed is None:   # retries exhausted -> safe fallback
            committed, skipped = safe_fallback_execute(
                portfolio, orders, prices, max_holdings=max_holdings,
                candidates=candidates, whole_shares=whole_shares)
        portfolio = committed

        # transaction costs on round-trip notional: everything was sold at a rebalance
        # (== value_before) plus whatever was just bought. Charged to cash so it flows
        # into value_after and the daily series. 0 bps => no change (default).
        # In reset mode the whole book is sold and re-bought, so round-trip notional is
        # the full portfolio. In hold mode only the exits trade, so charging the full
        # value would invent a cost the strategy never paid.
        if hold_mode and hold_info is not None:
            sold_notional = float(hold_info.get("redeployed", 0.0))
            bought_notional = float(hold_info.get("redeployed", 0.0))
        else:
            sold_notional = value_before if stage == "rebalance" else 0.0
            bought_notional = portfolio.holdings_value(prices)
        txn_cost = round((txn_cost_bps / 10000.0) * (sold_notional + bought_notional), 2)
        if txn_cost:
            portfolio.cash -= txn_cost

        value_after = portfolio.total_value(prices)
        period_log.append({
            "period": i, "stage": stage, "as_of": as_of,
            "window": f"{seg['start']}->{seg['end']}",
            "value_before": round(value_before, 2),
            "value_after": round(value_after, 2),
            "cash_after": round(portfolio.cash, 2),
            "invested_after": round(portfolio.holdings_value(prices), 2),
            "txn_cost": txn_cost,
            "num_holdings": portfolio.num_holdings(),
            "candidates": candidates,          # opportunity set the model chose from (selection null)
            "prior_holdings_perf": prior_holdings_perf,   # per-name realized perf feeding the reflection
            "holdings": {t: {"shares": round(portfolio.positions[t].shares, 4),
                             "avg_cost": round(portfolio.positions[t].avg_cost, 2),
                             "price": prices.get(t)} for t in portfolio.active_tickers()},
            "decision_attempts": attempts,
            "rejected_reasons": rejects,
            "skipped_orders": skipped,
            "hold_delta": hold_info,           # None unless rebalance_mode="hold"
        })
        # exact (unrounded) post-trade state for daily reconstruction
        segment_states.append({
            "start": seg["start"], "end": seg["end"], "cash": portfolio.cash,
            "positions": {t: portfolio.positions[t].shares for t in portfolio.active_tickers()},
        })
        prices_prev = prices

    # ---- final mark-to-market at window end ------------------------------- #
    end_date = segments[-1]["end"]
    end_prices = backend.get_prices(sorted(portfolio.active_tickers()), end_date)
    for t, p in prices_prev.items():
        end_prices.setdefault(t, p)
    final_value = portfolio.total_value(end_prices)
    total_roi = (final_value - budget) / budget * 100.0

    # ---- risk metrics from the reconstructed daily value series ----------- #
    risk: Optional[Dict[str, Any]] = None
    daily_dates: List[str] = []
    daily_values: List[float] = []
    bench_ticker = sector_config.benchmark()
    bench_metrics: Optional[Dict[str, Any]] = None
    if compute_risk:
        daily_dates, daily_values = reconstruct_daily_series(backend, segment_states)
        bench_values = benchmark_series_for(backend, bench_ticker, daily_dates)
        bench_aligned = bench_values if len(bench_values) == len(daily_values) else None
        risk = risk_metrics.compute_all(daily_values, bench_aligned, rf_annual=rf_annual)
        if bench_aligned:
            # standalone risk profile of the benchmark over the same window (context)
            bench_metrics = risk_metrics.compute_all(bench_aligned, None, rf_annual=rf_annual)

    # ---- INIT: the same run with reinvestment switched off ---------------- #
    # Buy the model's period-0 picks, then never trade again. Same window, same
    # backend, same price path as the strategy above — the ONLY difference is
    # that the rebalance decisions never happen. So `vs_no_reinvest_pp` isolates
    # what the reinvestment layer added or destroyed, with selection held fixed.
    #
    # Built by collapsing segment_states[0] into one segment spanning the whole
    # window: share counts and cash stay frozen at their post-period-0 values,
    # which is exactly buy-and-hold of the initial book.
    no_reinvest: Optional[Dict[str, Any]] = None
    if segment_states:
        s0 = segment_states[0]
        init_state = [{"start": s0["start"], "end": end_date,
                       "cash": s0["cash"], "positions": dict(s0["positions"])}]
        try:
            init_dates, init_values = reconstruct_daily_series(backend, init_state)
        except Exception:                                       # noqa: BLE001
            init_dates, init_values = [], []
        if init_values:
            init_final = init_values[-1]
            init_roi = (init_final - budget) / budget * 100.0
            init_risk = None
            if compute_risk:
                ib = bench_aligned if (bench_aligned and
                                       len(bench_aligned) == len(init_values)) else None
                init_risk = risk_metrics.compute_all(init_values, ib, rf_annual=rf_annual)
            no_reinvest = {
                "label": "INIT — hold period-0 picks, no rebalancing",
                "final_value": round(init_final, 2),
                "total_roi_pct": round(init_roi, 2),
                "holdings": {t: round(sh, 4) for t, sh in s0["positions"].items()},
                "cash": round(s0["cash"], 2),
                "risk_metrics": init_risk,
            }

    return {
        "sector": sector, "budget": budget, "max_holdings": max_holdings,
        "whole_shares": whole_shares, "end_date": end_date,
        "feedback": feedback, "txn_cost_bps": txn_cost_bps, "risk_free": rf_annual,
        "screened_news": bool(getattr(backend, "screened_news", False)),
        "order_mode": getattr(backend, "order_mode", "dollars"),
        "rebalance_mode": rebalance_mode,
        "prompt_framing": prompt_framing,
        "final_value": round(final_value, 2),
        "final_cash": round(portfolio.cash, 2),
        "total_roi_pct": round(total_roi, 2),
        # reinvestment attribution: strategy ROI minus the no-rebalance counterfactual.
        # positive => rebalancing added value; negative => it destroyed value.
        "no_reinvest_baseline": no_reinvest,
        "vs_no_reinvest_pp": (round(total_roi - no_reinvest["total_roi_pct"], 2)
                              if no_reinvest else None),
        "num_rebalances": len(segments) - 1,
        "periods": period_log,
        "final_holdings": {t: {"shares": round(portfolio.positions[t].shares, 4),
                               "avg_cost": round(portfolio.positions[t].avg_cost, 2),
                               "end_price": end_prices.get(t)}
                           for t in portfolio.active_tickers()},
        "benchmark_ticker": bench_ticker,
        "risk_metrics": risk,
        "benchmark_risk_metrics": bench_metrics,
        "daily_value_series": [[d, v] for d, v in zip(daily_dates, daily_values)],
    }


# =========================================================================== #
#  Reporting + CLI                                                            #
# =========================================================================== #
def print_report(run: Dict[str, Any]) -> None:
    print(f"\n{'='*78}")
    print(f" PORTFOLIO SIM — sector={run['sector']}  budget=${run['budget']:,.0f}  "
          f"max_holdings={run['max_holdings']}  "
          f"{'whole-shares' if run['whole_shares'] else 'fractional'}")
    print(f"{'='*78}")
    print(f" {'per':>3} {'stage':>9} {'as_of':>11} {'val_before':>11} {'val_after':>11} "
          f"{'cash':>10} {'hold':>4} {'try':>3}")
    for p in run["periods"]:
        print(f" {p['period']:>3} {p['stage']:>9} {p['as_of']:>11} "
              f"${p['value_before']:>9,.0f} ${p['value_after']:>9,.0f} "
              f"${p['cash_after']:>8,.0f} {p['num_holdings']:>4} {p['decision_attempts']:>3}")
        if p["rejected_reasons"]:
            for r in p["rejected_reasons"]:
                print(f"        ! rejected & re-prompted: {r}")
        if p["skipped_orders"]:
            for s in p["skipped_orders"]:
                print(f"        ~ skipped (fallback): {s}")
    print(f"{'-'*78}")
    print(f" FINAL VALUE  : ${run['final_value']:,.2f}  (cash ${run['final_cash']:,.2f})")
    print(f" TOTAL ROI    : {run['total_roi_pct']:+.2f}%   over {run['num_rebalances']} rebalances")
    nri = run.get("no_reinvest_baseline")
    if nri:
        d = run.get("vs_no_reinvest_pp")
        verdict = ("reinvestment ADDED value" if isinstance(d, (int, float)) and d > 0
                   else "reinvestment DESTROYED value" if isinstance(d, (int, float)) and d < 0
                   else "no difference")
        print(f" NO-REINVEST  : {nri['total_roi_pct']:+.2f}%  "
              f"(${nri['final_value']:,.2f}) — hold period-0 picks, never rebalance")
        print(f" vs NO-REINVEST: {d:+.2f}pp   {verdict}")
    print(f"{'='*78}")

    rmx = run.get("risk_metrics")
    if rmx:
        print(f"\n RISK METRICS (daily series, {rmx.get('n_observations')} obs, "
              f"benchmark {run.get('benchmark_ticker')}, "
              f"rf {rmx.get('risk_free_annual', 0)*100:.1f}%)")
        print(risk_metrics.format_summary(rmx))
        bm = run.get("benchmark_risk_metrics")
        if bm:
            def f(x, p=False):
                if x is None: return "n/a"
                return f"{x*100:+.2f}%" if p else f"{x:.2f}"
            print(f"   --- {run.get('benchmark_ticker')} over same window: "
                  f"return {f(bm.get('total_return'), True)}, "
                  f"vol {f(bm.get('annualized_volatility'), True)}, "
                  f"Sharpe {f(bm.get('sharpe'))}, MaxDD {f(bm.get('max_drawdown'), True)}")
        print(f"{'='*78}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cash-aware LLM portfolio simulator (buy/sell, fractional shares)")
    p.add_argument("--sector", default=sector_config.active_sector())
    p.add_argument("--budget", type=float, default=10000.0)
    p.add_argument("--max-holdings", type=int, default=5, help="upper cap Y on number of holdings")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    sched = p.add_mutually_exclusive_group(required=True)
    sched.add_argument("--segments", type=int)
    sched.add_argument("--interval-months", type=int)
    sched.add_argument("--dates")
    p.add_argument("--whole-shares", action="store_true",
                   help="conservative dollars->whole-shares execution (default: fractional)")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--risk-free", type=float, default=0.0,
                   help="annual risk-free rate as a decimal (e.g. 0.04 for 4%%)")
    p.add_argument("--feedback", choices=["on", "off"], default="on",
                   help="on (default): show prior-holdings performance at each rebalance "
                        "(self-reflection). off: control arm with no performance memory.")
    p.add_argument("--order-mode", choices=["dollars", "weights"], default="dollars",
                   dest="order_mode",
                   help="dollars (default): model emits dollar orders against a hard budget. "
                        "weights: model emits target fractions and the engine does the "
                        "arithmetic — overdraw becomes impossible, no re-prompt loop.")
    p.add_argument("--screened-news", choices=["on", "off"], default="off",
                   dest="screened_news",
                   help="on: Stage One reads tech_screened_corpus.csv (the LLM relevance "
                        "screen) instead of the raw merged corpus. Paper B only.")
    p.add_argument("--txn-cost-bps", type=float, default=0.0,
                   help="transaction cost in basis points on round-trip notional per rebalance "
                        "(default 0 = frictionless). Try 5/10/25 for sensitivity.")
    p.add_argument("--no-risk", action="store_true", help="skip risk-metric computation")
    p.add_argument("--model", default="gpt-5.1")
    p.add_argument("--dry-run", action="store_true", help="MockBackend; NO API calls")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    os.environ["SECTOR"] = args.sector
    dates = [d.strip() for d in args.dates.split(",")] if args.dates else None
    segments = build_segments(args.start, args.end, n_segments=args.segments,
                              interval_months=args.interval_months, dates=dates)
    print(f" Schedule: {len(segments)} periods ({len(segments)-1} rebalances), "
          f"{args.start} -> {args.end}")

    backend: SimBackend
    if args.dry_run:
        backend = MockBackend(order_mode=args.order_mode)
        print(f" Backend : MockBackend (dry-run, no API calls, order_mode={args.order_mode})")
    else:
        backend = RealBackend(model=args.model, screened_news=(args.screened_news == "on"),
                              order_mode=args.order_mode)
        print(f" Backend : RealBackend (live, model={args.model}, "
              f"screened_news={args.screened_news}, order_mode={args.order_mode})")

    run = run_simulation(backend, sector=args.sector, budget=args.budget,
                         segments=segments, max_holdings=args.max_holdings,
                         whole_shares=args.whole_shares, max_retries=args.max_retries,
                         rf_annual=args.risk_free, compute_risk=not args.no_risk,
                         feedback=(args.feedback == "on"), txn_cost_bps=args.txn_cost_bps)
    print_report(run)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"schedule": segments, "run": run}, f, indent=2)
        print(f"\n Wrote results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
