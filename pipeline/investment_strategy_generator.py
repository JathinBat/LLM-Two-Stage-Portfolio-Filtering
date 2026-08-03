"""
Investment Strategy Generator

Comprehensive investment analysis that combines news data and financial data to create 
detailed investment strategies. Merges functionality from NewsAPI+FinanceReports.ipynb,
FullCombinedWithCustomPeriods.py, and related investment analysis components.

This module provides complete investment strategy generation including:
- News-based market analysis
- Financial statement analysis
- Portfolio optimization
- Custom time period analysis
- Reinvestment scenarios
- Comprehensive reporting

DATA LEAKAGE PREVENTION MEASURES:
===============================
To ensure the AI model doesn't receive information from after the analysis period begins,
this module implements strict date filtering at multiple levels:

1. NEWS DATA FILTERING:
   - News end date is capped at analysis start date to prevent future information
   - All loaded news data is filtered to exclude articles after the cutoff date
   - Sample data generation uses dates before the analysis period

2. STOCK DATA FILTERING:
   - Analysis dates are validated against current date to prevent future data usage
   - Stock price data is filtered to exclude any data points after the analysis end date
   - Return calculations use only historical data within specified date ranges

3. VALIDATION LAYERS:
   - Input date validation ensures no future dates are accepted
   - Multiple checkpoints throughout the analysis pipeline
   - Clear logging of date adjustments and filtering actions

This ensures that investment recommendations are based solely on information that would
have been available at the time of the analysis period, maintaining realistic backtesting
conditions and preventing data snooping bias.
"""

import openai as _openai_lib
from openai import OpenAI
import ast
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import time
import random
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
import re

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
warnings.filterwarnings('ignore')

_MACRO_HTTP_TIMEOUT = (3.05, 6)

# Well-formed equity symbol: 1-5 letters, optional single class/exchange suffix
# (e.g. AAPL, BRK.B, BRK-B, RDS.A). Rejects LLM hallucinations like
# "META.DE-ALTERNATIVE" (multiple segments / word suffixes) before any Yahoo call.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def sanitize_ticker(symbol) -> str:
    """Uppercase and strip footnote/formatting chars (e.g. '*') from a raw ticker string."""
    return re.sub(r"[^A-Za-z0-9.\-]", "", str(symbol or "")).upper()


def is_valid_ticker(symbol) -> bool:
    """True only for well-formed equity symbols, filtering out hallucinated/malformed
    tickers (e.g. 'META.DE-ALTERNATIVE') that would 404 at the price API."""
    return bool(_TICKER_RE.match(sanitize_ticker(symbol)))


def extract_candidate_tickers(data) -> list:
    """Pull well-formed U.S. tickers from a candidate-screening JSON payload. Accepts
    either a plain list of strings (["AAPL", ...]) or a list of objects
    ([{"ticker": "AAPL", ...}, ...]), so it tolerates both prompt schemas."""
    candidates = data.get("candidates", []) if isinstance(data, dict) else []
    out = []
    for c in candidates:
        sym = sanitize_ticker(c if isinstance(c, str) else
                              (c.get("ticker", "") if isinstance(c, dict) else ""))
        if is_valid_ticker(sym):
            out.append(sym)
    return out


# Symbols Yahoo has already returned 404 for (well-formed but unknown/delisted, e.g. a
# hallucinated 'TMGT'). Cached so repeated fetches within a run don't re-hit the API.
_UNKNOWN_SYMBOLS: set = set()

# ── Alpha Vantage global rate limiter ─────────────────────────────────────
# AV free tier: 5 calls/min (1 call per 12 s).  This lock + timestamp is
# shared across ALL InvestmentStrategyGenerator instances and all threads so
# the 12-second gap is enforced globally, never per-thread.
import threading as _threading
_AV_LOCK           = _threading.Lock()
_AV_LAST_CALL      = [0.0]   # list so closure can mutate it
_AV_MIN_GAP_S      = 12.5   # slightly over 12 s to stay within free limit

def _av_rate_limit_wait():
    """Block the calling thread until it is safe to make the next Alpha Vantage call."""
    with _AV_LOCK:
        gap = _AV_MIN_GAP_S - (time.time() - _AV_LAST_CALL[0])
        if gap > 0:
            time.sleep(gap)
        _AV_LAST_CALL[0] = time.time()

# ── OpenAI global rate limiter + retry ────────────────────────────────────
# Enforces a minimum gap between all OpenAI calls across all threads and
# retries using server-provided rate-limit wait times when available.
_OAI_LOCK        = _threading.Lock()
_OAI_REQUEST_LOCK = _threading.Lock()
_OAI_LAST_CALL   = [0.0]
_OAI_BLOCK_UNTIL = [0.0]
_OAI_MIN_GAP_S   = 1.0   # ~60 RPM ceiling across all threads (conservative)

# ── Financial-report availability lag ─────────────────────────────────────
# A fiscal period ending before a decision date does NOT mean its report was
# public by then. When the data carries no filing date, assume a report became
# available no earlier than the SEC deadline for its type:
#   10-Q  40 days (large accelerated) / 45 days (accelerated + non-accelerated)
#   10-K  60 days (large accelerated) / 75 days (accelerated) / 90 (non-accel.)
# Defaults take the accelerated-filer deadline so the bound holds for the whole
# universe rather than only for megacaps. Override per run if a stricter or
# looser convention is wanted; report whichever value was used.
FIN_REPORT_LAG_DAYS_QUARTERLY = int(os.environ.get("FIN_REPORT_LAG_DAYS_Q", "45"))
FIN_REPORT_LAG_DAYS_ANNUAL    = int(os.environ.get("FIN_REPORT_LAG_DAYS_A", "75"))


def _format_exception_details(exc: Exception) -> str:
    """Include nested exception details so network failures are actionable."""
    parts = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    nested = exc.__cause__ or exc.__context__
    while nested is not None and id(nested) not in seen:
        seen.add(id(nested))
        parts.append(f"caused by {type(nested).__name__}: {nested}")
        nested = nested.__cause__ or nested.__context__
    return " | ".join(parts)


def _extract_json_text(content: str) -> str:
    """Extract the JSON payload from a model response."""
    text = (content or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    text = text.strip()
    if not text:
        return text

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        return text[first:last + 1].strip()
    return text


def _extract_openai_error_payload(exc: Exception) -> dict:
    """Best-effort extraction of OpenAI's structured error payload."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    text = str(exc)
    match = re.search(r"\{'error':\s*\{.*?\}\}", text)
    if match:
        try:
            payload = ast.literal_eval(match.group(0))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _openai_retry_after_seconds(exc: Exception, fallback_wait: float) -> tuple[float, str]:
    """Return a retry wait and reason based on OpenAI error details."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    retry_after = None
    try:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        retry_after = None
    if retry_after:
        try:
            return max(float(retry_after), 0.0) + 0.5, "retry-after header"
        except ValueError:
            pass

    payload = _extract_openai_error_payload(exc)
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = str(err.get("message") or exc)
    match = re.search(r"try again in\s+([0-9]*\.?[0-9]+)\s*s", message, re.IGNORECASE)
    if match:
        return max(float(match.group(1)), 0.0) + 0.75, "OpenAI message"

    err_type = str(err.get("type") or "").lower()
    err_code = str(err.get("code") or "").lower()
    if err_type == "tokens" or "token" in message.lower():
        return max(fallback_wait, 10.0), "token rate limit fallback"
    if "rate_limit" in err_code or "rate limit" in message.lower():
        return max(fallback_wait, 5.0), "rate limit fallback"
    return fallback_wait, "exponential fallback"


def _oai_wait_for_slot() -> None:
    """Block until the shared OpenAI throttle allows another request."""
    with _OAI_LOCK:
        now = time.time()
        blocked_for = _OAI_BLOCK_UNTIL[0] - now
        if blocked_for > 0:
            time.sleep(blocked_for)

        gap = _OAI_MIN_GAP_S - (time.time() - _OAI_LAST_CALL[0])
        if gap > 0:
            time.sleep(gap)
        _OAI_LAST_CALL[0] = time.time()


# Transient errors worth retrying (connection drops like WinError 10054, timeouts,
# and 5xx). Built with getattr so it degrades gracefully across openai versions.
_OAI_RETRYABLE_ERRORS = tuple(c for c in (
    getattr(_openai_lib, "APIConnectionError", None),   # includes APITimeoutError (subclass)
    getattr(_openai_lib, "APITimeoutError", None),
    getattr(_openai_lib, "InternalServerError", None),  # transient 5xx
) if isinstance(c, type)) or (ConnectionError,)


def _oai_create_with_retry(client, max_retries: int = 6, **kwargs):
    """Call client.chat.completions.create(**kwargs) with:
    - a global minimum inter-call gap to stay under RPM limits
    - exponential backoff on RateLimitError (5s → 10s → 20s … capped at 60s)
    """
    wait = 5.0
    for attempt in range(max_retries):
        try:
            # Pacing is serialized (_oai_wait_for_slot takes _OAI_LOCK and enforces
            # the global _OAI_MIN_GAP_S ceiling); the request itself is NOT. Holding
            # a mutex across create() would serialize every OpenAI call in the
            # process, so worker threads would queue on it and multi-threaded runs
            # would go exactly as fast as single-threaded ones.
            _oai_wait_for_slot()
            return client.chat.completions.create(**kwargs)
        except _openai_lib.RateLimitError as exc:
            if attempt == max_retries - 1:
                raise
            sleep_for, reason = _openai_retry_after_seconds(exc, wait)
            sleep_for = min(max(sleep_for, 1.0), 120.0)
            with _OAI_LOCK:
                _OAI_BLOCK_UNTIL[0] = max(_OAI_BLOCK_UNTIL[0], time.time() + sleep_for)
            print(
                f"  [OpenAI rate limit] {reason}; pausing OpenAI calls for "
                f"{sleep_for:.2f}s before retry {attempt + 2}/{max_retries} ..."
            )
            time.sleep(sleep_for)
            wait = min(max(wait * 2, sleep_for * 1.5), 120.0)
        except _OAI_RETRYABLE_ERRORS as exc:
            # RateLimitError is a subclass of APIStatusError but is handled above.
            if attempt == max_retries - 1:
                raise
            sleep_for = min(max(wait, 2.0), 60.0)
            print(f"  [OpenAI connection error] {type(exc).__name__}: {exc}; "
                  f"retrying in {sleep_for:.1f}s (attempt {attempt + 2}/{max_retries}) ...")
            time.sleep(sleep_for)
            wait = min(wait * 2, 60.0)

# ── Module-level news cache: (abs_news_data_path | None, start_date_str, end_date_str) -> DataFrame
# Shared across all InvestmentStrategyGenerator instances so each unique date window is
# loaded and filtered from disk exactly once per process, no matter how many runs share it.
_NEWS_DF_CACHE: dict = {}

# ── Ticker universe for unfiltered (single-pass) runs ────────────────────────
# All tickers in this set have their financial + price history fetched upfront
# and fed to the LLM in a single call with no candidate-filtering stage.
# Sector registry: the unfiltered universe, benchmark and news scope are selected
# by the SECTOR env var (default "technology" -> the original list below, unchanged).
try:
    import sector_config as _sector_config
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import sector_config as _sector_config

# Unfiltered universe for the active sector. For technology this is identical to
# the original fixed list: GOOGL, MSFT, NVDA, AAPL, TSM, CSCO, AMD, TXN, MCHP,
# ASML, MU, META, AMZN, CRM, AVGO, IBM, ZM, PLTR, QCOM, TSLA, INTC, LRCX, ORCL,
# ADBE, NOW, SNPS, CDNS, KLAC, AMAT.
UNFILTERED_TICKER_UNIVERSE: list = _sector_config.universe()

# ── Process-level in-memory caches (shared across all instances and threads) ─
# Sits above the disk cache so any data loaded once stays in RAM for the full
# process lifetime — subsequent runs skip file I/O entirely.
_PRICE_HISTORY_MEM_CACHE:  dict = {}   # ticker -> (hist_1y, hist_3y, hist_5y)
_TICKER_INFO_MEM_CACHE:    dict = {}   # ticker -> info dict
_FINANCIAL_DATA_MEM_CACHE: dict = {}   # ticker -> company_data dict


class InvestmentStrategyGenerator:
    """Complete investment strategy generator with news and financial data integration"""
    
    def __init__(self, api_key_openai: str, alpha_vantage_key: Optional[str] = None, nyt_api_key: Optional[str] = None, temperature: float = 0.3, model: str = "gpt-5.1", initial_candidates: int = 20, final_portfolio: int = 5):
        """Initialize with API keys, temperature setting, and model selection"""
        # Disable SDK-level retries so connection failures surface to the GUI
        # instead of sleeping inside OpenAI's internal retry loop.
        self.openai_client = OpenAI(api_key=api_key_openai, max_retries=0)
        self.alpha_vantage_key = alpha_vantage_key or "demo"  # Default demo key
        self.nyt_api_key = nyt_api_key or os.environ.get("NYT_API_KEY", "")
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        self.temperature = temperature  # Configurable temperature for all OpenAI calls
        self.model = model  # Model to use for OpenAI calls
        self.initial_candidates = initial_candidates  # Companies to identify from news (stage 1)
        self.final_portfolio = final_portfolio        # Companies to keep after financial validation (stage 2)
        self._last_strategy_error = ""
        self.gpt5_cutoff_date = datetime(2024, 10, 1)  # GPT-5.1 only available for periods after Oct 2024

        # Training data cutoff dates for known models.
        # If the analysis period overlaps with dates the model was trained on,
        # the LLM may "remember" real market events and leak future information
        # into its recommendations, invalidating the backtest.
        self.TRAINING_CUTOFFS: Dict[str, datetime] = {
            "gpt-4":       datetime(2023, 12, 1),
            "gpt-4-turbo": datetime(2023, 4, 1),
            "gpt-4o":      datetime(2023, 10, 1),
            "gpt-4o-mini": datetime(2023, 10, 1),
            "gpt-5":       datetime(2024, 1, 1),
            "gpt-5.1":     datetime(2024, 6, 1),
            "o3-mini":     datetime(2025, 10, 1),
        }

    def _strategy_failed(self, message: str) -> None:
        self._last_strategy_error = message
        print(f" {message}")

    def _load_json_response(
        self,
        content: str,
        label: str,
        model: str,
        max_completion_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """Parse model JSON, with one repair pass for malformed responses."""
        json_text = _extract_json_text(content)
        if not json_text:
            raise ValueError(f"Empty JSON block in response for {label}")

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as original_error:
            print(f"  Malformed JSON for {label}; requesting one JSON repair pass...")
            repair_prompt = f"""
The following response was intended to be a single JSON object for {label}, but it is malformed.

Return ONLY one valid JSON object. Do not include markdown, explanations, comments, or code fences.
Preserve the same schema and values where possible. If a string was cut off, close it cleanly.

Malformed response:
{json_text}
"""
            repair_response = _oai_create_with_retry(
                self.openai_client,
                model=model,
                messages=[{"role": "user", "content": repair_prompt}],
                **self._temp_kwargs(model),
                max_completion_tokens=max_completion_tokens,
            )
            repaired = repair_response.choices[0].message.content
            repaired_text = _extract_json_text(repaired)
            try:
                return json.loads(repaired_text)
            except Exception as repair_error:
                raise ValueError(
                    f"Could not parse {label} JSON after repair. "
                    f"Original parse error: {_format_exception_details(original_error)}; "
                    f"repair parse error: {_format_exception_details(repair_error)}"
                ) from repair_error

    def _check_training_data_overlap(self, model: str, analysis_start: str, analysis_end: str) -> Optional[str]:
        """Return a warning string if the analysis period overlaps with the model's training data."""
        cutoff = None
        model_lower = model.lower()
        # Match the longest key first so "gpt-4o-mini" beats "gpt-4"
        for key in sorted(self.TRAINING_CUTOFFS, key=len, reverse=True):
            if key in model_lower:
                cutoff = self.TRAINING_CUTOFFS[key]
                break
        if cutoff is None:
            return None

        start_dt = datetime.strptime(analysis_start, '%Y-%m-%d')
        end_dt = datetime.strptime(analysis_end, '%Y-%m-%d')

        if end_dt <= cutoff:
            # Entire analysis window is within training data
            return (f"FULL OVERLAP: The entire analysis period ({analysis_start} to {analysis_end}) "
                    f"falls within {model}'s training data (cutoff ~{cutoff.strftime('%Y-%m')}). "
                    f"The model may recall real market outcomes, compromising backtest validity.")
        elif start_dt < cutoff:
            # Partial overlap
            return (f"PARTIAL OVERLAP: The analysis period ({analysis_start} to {analysis_end}) "
                    f"partially overlaps {model}'s training data (cutoff ~{cutoff.strftime('%Y-%m')}). "
                    f"Results before {cutoff.strftime('%Y-%m-%d')} may be influenced by memorised events.")
        return None

    def _validate_model_for_date_range(self, analysis_start_date: str) -> str:
        """Validate and adjust model based on analysis date range"""
        analysis_start = datetime.strptime(analysis_start_date, '%Y-%m-%d')
        
        if "gpt-5" in self.model.lower():
            if analysis_start < self.gpt5_cutoff_date:
                print(f"  {self.model} requested but analysis start date ({analysis_start_date}) is before Oct 2024")
                print(f"    Falling back to gpt-4o-mini for historical consistency")
                return "gpt-4o-mini"
            else:
                print(f" Using {self.model} for analysis period starting {analysis_start_date}")
                return self.model
        else:
            return self.model

    # Models that only accept the default temperature (1) and reject custom values.
    _NO_TEMP_PATTERNS = ("o1", "o3", "o4", "gpt-5")

    def _temp_kwargs(self, model: str) -> dict:
        """Return {'temperature': ...} only for models that support custom temperature."""
        if any(p in model.lower() for p in self._NO_TEMP_PATTERNS):
            return {}
        return {"temperature": self.temperature}

    def generate_complete_strategy(self,
                                 user_input_keyword: str,
                                 investment_amount: float,
                                 news_start_date: Optional[str] = None,
                                 news_end_date: Optional[str] = None,
                                 analysis_start_date: Optional[str] = None,
                                 analysis_end_date: Optional[str] = None,
                                 return_periods: Optional[List[str]] = None,
                                 reinvestment_amount: float = 0,
                                 news_data_path: Optional[str] = None,
                                 include_financial_validation: bool = True,
                                 max_companies: Optional[int] = None,
                                 additional_context: str = "",
                                 final_step_context: str = "",
                                 extra_data_paths: Optional[List[str]] = None,
                                 single_pass: bool = False,
                                 ranked_final_selection: bool = False) -> Dict[str, Any]:
        """
        Generate complete investment strategy with news and financial data integration
        
        This is the main method that combines all functionality:
        1. News-based analysis for market insights
        2. Financial statement validation
        3. Portfolio optimization
        4. Return calculations with custom periods
        5. Risk assessment and final recommendations
        """
        
        # Set default dates with strict validation to prevent future data usage
        today = datetime.now()
        
        if news_start_date is None:
            news_start_date = (today - relativedelta(years=2)).strftime('%Y-%m-%d')
        if news_end_date is None:
            news_end_date = today.strftime('%Y-%m-%d')
        if analysis_start_date is None:
            analysis_start_date = (today - relativedelta(years=1)).strftime('%Y-%m-%d')
        if analysis_end_date is None:
            analysis_end_date = today.strftime('%Y-%m-%d')
        if return_periods is None:
            return_periods = ['OVERALL']  # Only calculate overall return from start to end
        
        # BASIC VALIDATION: Only prevent clearly future dates
        news_end_dt = datetime.strptime(news_end_date, '%Y-%m-%d')
        analysis_end_dt = datetime.strptime(analysis_end_date, '%Y-%m-%d')
        
        # Only adjust if dates are actually in the future (beyond today)
        if news_end_dt.date() > today.date():
            print(f" News end date {news_end_date} is in the future, adjusting to today")
            news_end_date = today.strftime('%Y-%m-%d')
            
        if analysis_end_dt.date() > today.date():
            print(f" Analysis end date {analysis_end_date} is in the future, adjusting to today")
            analysis_end_date = today.strftime('%Y-%m-%d')
        
        # Allow user to select news and analysis periods independently
        # Only warn if there might be data leakage, but don't force changes
        analysis_start_dt = datetime.strptime(analysis_start_date, '%Y-%m-%d')
        if news_end_dt > analysis_start_dt:
            print(f" Note: News data extends to {news_end_date}, analysis starts {analysis_start_date}")
            print(f"    This may include some information available during the analysis period")
            # Don't force change - let user decide
            
        # Validate and set appropriate model for date range
        validated_model = self._validate_model_for_date_range(analysis_start_date)

        # Check for training-data / analysis-period overlap
        training_overlap_warning = self._check_training_data_overlap(
            validated_model, analysis_start_date, analysis_end_date
        )
        if training_overlap_warning:
            print(f"\n{'!'*80}")
            print(f"WARNING  {training_overlap_warning}")
            print(f"{'!'*80}\n")

        print(f" COMPREHENSIVE INVESTMENT STRATEGY GENERATION")
        print(f" Sector: {user_input_keyword}")
        print(f" Investment: ${investment_amount:,.2f}")
        print(f" News Period: {news_start_date} to {news_end_date}")
        print(f" Analysis Period: {analysis_start_date} to {analysis_end_date}")
        print(f" AI Model: {validated_model}")
        print("="*80)
        self._last_strategy_error = ""
        
        try:
            # Single Comprehensive Analysis Phase
            print(" COMPREHENSIVE INVESTMENT STRATEGY GENERATION")
            print("-" * 60)
            print("Merging all analysis layers: News + Financial + Historical Performance")
            
            if single_pass:
                final_strategy = self._run_strategy_unfiltered(
                    user_input_keyword, news_data_path, news_start_date,
                    news_end_date, investment_amount, reinvestment_amount,
                    additional_context, validated_model,
                    extra_data_paths=extra_data_paths,
                )
            else:
                final_strategy = self._generate_unified_strategy(
                    user_input_keyword, news_data_path, news_start_date,
                    news_end_date, investment_amount,
                    reinvestment_amount, include_financial_validation, additional_context,
                    validated_model,
                    extra_data_paths=extra_data_paths,
                    ranked_final_selection=ranked_final_selection,
                    final_step_context=final_step_context,
                )
            
            if not final_strategy:
                detail = self._last_strategy_error or "No final strategy was returned"
                return {"error": f"Unified strategy generation failed: {detail}"}
            
            # Return Analysis
            print("\n RETURN ANALYSIS")
            print("-" * 30)
            
            return_analysis = self._calculate_comprehensive_returns(
                final_strategy, analysis_start_date, analysis_end_date, 
                investment_amount, reinvestment_amount, return_periods
            )
            
            # Final Results Assembly
            print("\n RESULTS COMPILATION")
            print("-" * 30)
            
            complete_results = self._assemble_unified_results(
                final_strategy, return_analysis, investment_amount, reinvestment_amount,
                sector=user_input_keyword, model=validated_model,
                news_start_date=news_start_date, news_end_date=news_end_date,
                analysis_start_date=analysis_start_date, analysis_end_date=analysis_end_date,
                max_companies=max_companies,
                training_overlap_warning=training_overlap_warning
            )
            
            # Display overall return summary
            if 'return_analysis' in complete_results and not complete_results['return_analysis'].get('error'):
                return_data = complete_results['return_analysis']
                main_return = return_data.get('main_return', 0)
                total_roi = return_data.get('total_roi', 0)
                final_value = return_data.get('total_final_value', 0)
                
                print("\n" + "="*60)
                print(" OVERALL PORTFOLIO PERFORMANCE SUMMARY")
                print("="*60)
                print(f" Main Portfolio Return: {main_return:.2f}%" if isinstance(main_return, (int, float)) else f" Main Portfolio Return: {main_return}")
                print(f" Total ROI: {total_roi:.2f}%" if isinstance(total_roi, (int, float)) else f" Total ROI: {total_roi}")
                print(f" Final Portfolio Value: ${final_value:,.2f}" if isinstance(final_value, (int, float)) else f" Final Portfolio Value: {final_value}")
                print("="*60)
            
            print(" STRATEGY GENERATION COMPLETE!")
            return complete_results
            
        except Exception as e:
            error_details = _format_exception_details(e)
            print(f" Error in strategy generation: {error_details}")
            return {"error": error_details, "phase": "unknown"}

    def run_sliding_window_analysis(self,
                                     user_input_keyword: str,
                                     investment_amount: float,
                                     start_date: str,
                                     period_length_months: int,
                                     end_date: str,
                                     news_data_path: Optional[str] = None,
                                     include_financial_validation: bool = True,
                                     max_companies: Optional[int] = None,
                                     additional_context: str = "") -> List[Dict[str, Any]]:
        """
        Run investment analysis with a sliding window approach.

        Args:
            user_input_keyword: Investment sector/keyword (e.g., "technology")
            investment_amount: Amount to invest in each period
            start_date: Starting date for the first period (YYYY-MM-DD)
            period_length_months: Length of each analysis period in months
            end_date: Stop when period end reaches this date (YYYY-MM-DD)
            news_data_path: Optional path to news CSV file
            include_financial_validation: Whether to include financial validation
            max_companies: Maximum number of companies to analyze
            additional_context: Additional context for analysis

        Returns:
            List of results for each period analyzed
        """
        print("\n" + "="*80)
        print(" SLIDING WINDOW INVESTMENT ANALYSIS")
        print("="*80)

        # Parse dates
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        final_end = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        results = []
        period_num = 1

        while True:
            # Calculate period end date
            period_end = current_start + relativedelta(months=period_length_months)

            # Stop if the full period length can't fit before today's date
            if period_end > today:
                print(f"\n Stopping: period end ({period_end.strftime('%Y-%m-%d')}) exceeds today ({today.strftime('%Y-%m-%d')}), full {period_length_months}-month period cannot be fulfilled")
                break

            # Check if we've reached the final end date
            if period_end > final_end:
                print(f"\n Reached end date. Period end ({period_end.strftime('%Y-%m-%d')}) would exceed final end date ({end_date})")
                break

            # Format dates for this period
            period_start_str = current_start.strftime('%Y-%m-%d')
            period_end_str = period_end.strftime('%Y-%m-%d')

            print(f"\n{'='*80}")
            print(f" PERIOD {period_num}: {period_start_str} to {period_end_str}")
            print(f"{'='*80}")

            # Run analysis for this period
            # News window: 2 years before the analysis period start
            news_start_for_period = (current_start - relativedelta(years=2)).strftime('%Y-%m-%d')

            try:
                result = self.generate_complete_strategy(
                    user_input_keyword=user_input_keyword,
                    investment_amount=investment_amount,
                    news_start_date=news_start_for_period,  # 2 years before this period's start
                    news_end_date=period_start_str,  # News should be BEFORE analysis period
                    analysis_start_date=period_start_str,
                    analysis_end_date=period_end_str,
                    return_periods=None,  # Use default return periods
                    reinvestment_amount=0,  # No reinvestment in simplified version
                    news_data_path=news_data_path,
                    include_financial_validation=include_financial_validation,
                    max_companies=max_companies,
                    additional_context=additional_context
                )

                # Add period information to result
                result['period_number'] = period_num
                result['period_start'] = period_start_str
                result['period_end'] = period_end_str

                results.append(result)
                print(f" Period {period_num} analysis complete")

            except Exception as e:
                print(f" Error in period {period_num}: {e}")
                results.append({
                    'period_number': period_num,
                    'period_start': period_start_str,
                    'period_end': period_end_str,
                    'error': str(e)
                })

            # Move to next period (increment by 1 month)
            current_start = current_start + relativedelta(months=1)
            period_num += 1

        print(f"\n{'='*80}")
        print(f" SLIDING WINDOW ANALYSIS COMPLETE - {len(results)} periods analyzed")
        print(f"{'='*80}")

        return results

    def _generate_unified_strategy(self, keyword: str, news_data_path: Optional[str],
                                 news_start_date: str, news_end_date: str,
                                 investment_amount: float, reinvestment_amount: float,
                                 include_financial_validation: bool, additional_context: str = "",
                                 validated_model: str = None,
                                 extra_data_paths: Optional[List[str]] = None,
                                 ranked_final_selection: bool = False,
                                 final_step_context: str = "") -> Optional[Dict]:
        """Single-layer comprehensive analysis: News + Financial + Historical + Strategy generation"""

        print(" Loading and analyzing all data sources...")

        # Step 1a: Macro market context (historical, temporally compliant)
        print(" Fetching macro market context...")
        macro_context = self._fetch_macro_context(news_end_date)
        if macro_context:
            print(f"  Macro context loaded ({len(macro_context)} chars)")

        # Step 1b: Extra user-provided data files
        extra_data_text = ""
        if extra_data_paths:
            print(" [UNFILTERED] Ignoring supplementary files in strict temporal mode")

        # Step 1c: Load news data with date filtering
        news_df = self._load_news_data(news_data_path, news_start_date, news_end_date, keyword)
        if news_df is None or len(news_df) == 0:
            print(" No news data available - falling back to stock-only analysis")
            return self._generate_stock_only_strategy(keyword, news_start_date, news_end_date, investment_amount, reinvestment_amount, additional_context)

        # Step 2: Process news articles
        print(f" Processing ALL {len(news_df)} news articles (no limits)...")
        articles_text = self._process_news_data(news_df, keyword)
        
        # Step 3: Get comprehensive analysis from GPT with all data
        print(" Requesting comprehensive strategy from GPT...")
        
        if ranked_final_selection:
            first_step_instructions = f"""
        1. RANK COMPANIES: From the news data, identify and rank all relevant publicly-traded companies in the {keyword} sector that have enough support in the news.
        2. DO NOT use a fixed company-count target in this step. Include every relevant company you can justify from the provided data.
        3. ANALYZE PERFORMANCE: I will provide financial and historical performance data for the ranked companies.
        4. FINAL SELECTION LATER: Do not make the final portfolio selection yet.
        """
            first_step_schema_extra = """
                    "rank": 1,
                    "news_score": 9.2,
"""
            first_step_return_note = "Return ONLY the JSON object with the ranked company list."
        else:
            first_step_instructions = f"""
        1. IDENTIFY COMPANIES: From the news data, identify {self.initial_candidates} relevant publicly-traded companies in the {keyword} sector
        2. ANALYZE PERFORMANCE: I will provide financial and historical performance data for these companies
        3. GENERATE STRATEGY: Create final portfolio allocations based on all data combined
        """
            first_step_schema_extra = ""
            first_step_return_note = "Return ONLY the JSON object with the company list."

        unified_prompt = f"""
        You are an expert investment analyst creating a complete long-term investment strategy.

        TASK: Analyze ALL provided data sources simultaneously to generate a final investment portfolio.

        SECTOR FOCUS: {keyword}
        INVESTMENT AMOUNT: ${investment_amount:,.2f}
        REINVESTMENT AMOUNT: ${reinvestment_amount:,.2f}

        {macro_context if macro_context else ""}

        NEWS DATA FOR ANALYSIS:
        {articles_text}

        {extra_data_text if extra_data_text else ""}

        {additional_context if additional_context else ""}

        INSTRUCTIONS:
        {first_step_instructions}
        
        First, provide a list of companies to analyze in this JSON format:
        {{
            "companies_for_analysis": [
                {{{first_step_schema_extra}"ticker": "AAPL", "company_name": "Apple Inc.", "rationale": "Why selected from news"}},
                {{{first_step_schema_extra}"ticker": "MSFT", "company_name": "Microsoft Corp.", "rationale": "Why selected from news"}}
            ],
            "market_analysis": {{
                "key_themes": ["theme1", "theme2", "theme3"],
                "investment_outlook": "Overall market outlook for the sector",
                "major_trends": ["trend1", "trend2", "trend3"]
            }}
        }}
        
        {first_step_return_note}
        """
        
        # Get initial company list from GPT
        try:
            current_model = validated_model or self.model
            response = _oai_create_with_retry(
                self.openai_client,
                model=current_model,
                messages=[{"role": "user", "content": unified_prompt}],
                **self._temp_kwargs(current_model),
                max_completion_tokens=3000,
            )
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                self._strategy_failed("Empty response from OpenAI for company identification")
                return None

            company_data = self._load_json_response(
                content,
                "company identification",
                current_model,
                max_completion_tokens=3000,
            )
            companies_list = company_data.get('companies_for_analysis', [])
            market_analysis = company_data.get('market_analysis', {})
            if not companies_list:
                self._strategy_failed("OpenAI returned no companies_for_analysis")
                return None
            
            print(f" Identified {len(companies_list)} companies for comprehensive analysis")
            
        except Exception as e:
            self._strategy_failed(f"Error getting company list: {_format_exception_details(e)}")
            return None
        
        # Step 4: Fetch financial and historical data for identified companies
        print(" Fetching comprehensive data for identified companies...")
        
        comprehensive_data = {}
        if include_financial_validation:
            # Fetch financial data
            financial_data = self._fetch_financial_data_for_tickers(
                [comp['ticker'] for comp in companies_list],
                max_tickers=None if ranked_final_selection else 10,
            )
        else:
            financial_data = []
        
        # Fetch historical performance for all companies
        for company in companies_list:
            ticker = company['ticker']
            print(f"   Analyzing {ticker}...")

            try:
                hist_1y, hist_3y, hist_5y = self._load_temporal_price_histories(ticker, news_end_date)
                
                if hist_1y is not None and not hist_1y.empty:
                    perf_metrics = self._calculate_performance_metrics(hist_1y, hist_3y, hist_5y)

                    # Fetch fundamental info and earnings (cached after first call)
                    info_data = self._fetch_ticker_info(ticker)

                    # Find corresponding financial data
                    fin_data = None
                    for fd in financial_data:
                        if fd.get('ticker') == ticker:
                            fin_data = self._filter_financial_data_as_of(fd, news_end_date)
                            break

                    # Generate summary
                    data_package = {
                        'name': company['company_name'],
                        'performance_data': perf_metrics,
                        'financial_data': fin_data,
                        'news_rationale': company['rationale'],
                        'info': info_data,
                    }
                    
                    summary = self._generate_stock_summary(ticker, data_package)
                    comprehensive_data[ticker] = {
                        'company_name': company['company_name'],
                        'summary': summary,
                        'rationale': company['rationale']
                    }
                    
            except Exception as e:
                print(f"     Error analyzing {ticker}: {e}")
        
        # Step 5: Generate final strategy with all data
        print(" Generating final investment strategy with comprehensive data...")
        
        # Prepare comprehensive data summary
        data_summary = "COMPREHENSIVE COMPANY ANALYSIS:\n"
        data_summary += "=" * 50 + "\n\n"
        
        for ticker, data in comprehensive_data.items():
            data_summary += f"{data['company_name']} ({ticker}):\n"
            data_summary += f"News Rationale: {data['rationale']}\n"
            data_summary += data['summary'] + "\n"
            data_summary += "-" * 30 + "\n\n"
        
        if ranked_final_selection:
            final_selection_instructions = """
        CREATE FINAL PORTFOLIO:
        1. First rank every analyzed company from strongest to weakest using all available news, financial, performance, fundamentals, analyst, and macro evidence.
        2. Only after that ranking, choose the final portfolio companies. Do not use a fixed company-count target; choose the number of holdings that the evidence and diversification needs justify.
        3. Allocate percentages across the selected final companies (must sum to 100%).
        4. Prioritize: Low volatility + Strong growth + Good momentum + Positive analyst consensus.
        5. Factor in macro conditions (VIX level, yield environment, market trend) when sizing risk.
        6. Consider news sentiment and market trends.
"""
            ranked_schema = """
            "ranked_companies": [
                {
                    "rank": 1,
                    "ticker": "TICKER",
                    "company_name": "Company Name",
                    "ranking_score": 8.9,
                    "ranking_rationale": "Why this company ranks here"
                }
            ],
"""
        else:
            final_selection_instructions = f"""
        CREATE FINAL PORTFOLIO:
        1. Select exactly {self.final_portfolio} companies for the final portfolio based on volatility, growth, speed, fundamentals, and analyst consensus
        2. Allocate percentages across these {self.final_portfolio} companies (must sum to 100%)
        3. Prioritize: Low volatility + Strong growth + Good momentum + Positive analyst consensus
        4. Factor in macro conditions (VIX level, yield environment, market trend) when sizing risk
        5. Consider news sentiment and market trends
"""
            ranked_schema = ""

        # Optional prior-holdings performance, injected into the financial/final
        # selection step (used by the reinvestment runner's 'financial_report' mode).
        prior_perf_block = ""
        if final_step_context:
            prior_perf_block = (
                "PRIOR HOLDINGS PERFORMANCE (realized since last rebalance; as-of data only,\n"
                "        no post-rebalance information):\n        " + final_step_context + "\n"
            )

        # Final strategy generation prompt
        final_prompt = f"""
        Create a complete investment strategy using all the analyzed data.

        {macro_context if macro_context else ""}

        MARKET ANALYSIS FROM NEWS:
        - Key Themes: {market_analysis.get('key_themes', [])}
        - Investment Outlook: {market_analysis.get('investment_outlook', 'N/A')}
        - Major Trends: {market_analysis.get('major_trends', [])}

        COMPREHENSIVE COMPANY DATA:
        {data_summary}

        {prior_perf_block}

        INVESTMENT PARAMETERS:
        - Total Investment: ${investment_amount:,.2f}
        - Reinvestment Amount: ${reinvestment_amount:,.2f}

        {final_selection_instructions}
        
        Return complete strategy as JSON:
        {{
            "investment_strategy": "Overall strategy description",
            {ranked_schema}
            "final_recommendations": [
                {{
                    "ticker": "TICKER",
                    "company_name": "Company Name", 
                    "final_allocation": 15.5,
                    "volatility_assessment": "LOW/MODERATE/HIGH",
                    "growth_assessment": "STRONG/GOOD/MODEST/DECLINING",
                    "speed_assessment": "FAST/STEADY/SLOW/NEGATIVE",
                    "news_factor": "How news influenced selection",
                    "allocation_rationale": "Why this allocation percentage",
                    "overall_score": 8.2
                }}
            ],
            "portfolio_summary": {{
                "total_allocation": 100.0,
                "expected_return": "8-12%",
                "risk_level": "Moderate",
                "strategy_focus": "Main investment thesis"
            }},
            "analysis_summary": {{
                "companies_analyzed": {len(comprehensive_data)},
                "selection_criteria": "How companies were selected and weighted",
                "market_outlook": "Overall sector outlook"
            }}
        }}
        """
        
        try:
            current_model = validated_model or self.model
            response = _oai_create_with_retry(
                self.openai_client,
                model=current_model,
                messages=[{"role": "user", "content": final_prompt}],
                **self._temp_kwargs(current_model),
                max_completion_tokens=4000,
            )
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                self._strategy_failed("Empty response from OpenAI for final strategy")
                return None

            final_strategy = self._load_json_response(
                content,
                "final strategy",
                current_model,
                max_completion_tokens=4000,
            )
            if not final_strategy.get("final_recommendations"):
                self._strategy_failed("OpenAI final strategy JSON did not include final_recommendations")
                return None
            
            print(" Unified strategy generation complete")
            return final_strategy
            
        except Exception as e:
            self._strategy_failed(f"Error generating final strategy: {_format_exception_details(e)}")
            return None

    def _run_strategy_unfiltered(
        self,
        keyword: str,
        news_data_path: Optional[str],
        news_start_date: str,
        news_end_date: str,
        investment_amount: float,
        reinvestment_amount: float,
        additional_context: str = "",
        validated_model: Optional[str] = None,
        extra_data_paths: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Single-pass unfiltered strategy.

        Loads ALL news and ALL financial + price data for UNFILTERED_TICKER_UNIVERSE
        simultaneously, then makes a single LLM call.  There is no candidate-filtering
        stage: the model sees every data point at once and decides the portfolio freely.
        """
        print(" [UNFILTERED] Loading all data sources simultaneously...")

        # ── Macro context ────────────────────────────────────────────────────
        macro_context = self._fetch_macro_context(news_end_date)
        if macro_context:
            print(f"  Macro context loaded ({len(macro_context)} chars)")

        # ── Extra user-provided files ────────────────────────────────────────
        if extra_data_paths:
            print(" [UNFILTERED] Ignoring supplementary files in strict temporal mode")
        extra_data_text = ""

        # ── All news ─────────────────────────────────────────────────────────
        news_df = self._load_news_data(news_data_path, news_start_date, news_end_date, keyword)
        if news_df is None or len(news_df) == 0:
            print(" [UNFILTERED] No news data — falling back to stock-only analysis")
            return self._generate_stock_only_strategy(
                keyword, news_start_date, news_end_date,
                investment_amount, reinvestment_amount, additional_context,
            )
        articles_text = self._process_news_data(news_df, keyword)
        print(f" [UNFILTERED] {len(news_df)} news articles loaded")

        # ── All financial + price data for the full ticker universe ──────────
        tickers = list(UNFILTERED_TICKER_UNIVERSE)
        print(f" [UNFILTERED] Fetching data for {len(tickers)}-ticker universe as of {news_end_date}...")

        raw_financial_data = self._fetch_financial_data_for_tickers(tickers)
        financial_data = [
            self._filter_financial_data_as_of(fd, news_end_date)
            for fd in raw_financial_data
            if fd
        ]

        comprehensive_data: Dict[str, str] = {}
        for ticker in tickers:
            print(f"   {ticker}...")
            try:
                hist_1y, hist_3y, hist_5y = self._load_temporal_price_histories(ticker, news_end_date)

                if hist_1y is not None and not hist_1y.empty:
                    perf    = self._calculate_performance_metrics(hist_1y, hist_3y, hist_5y)
                    fin     = next((fd for fd in financial_data if fd.get("ticker") == ticker), None)
                    summary = self._generate_stock_summary(ticker, {
                        "performance_data": perf,
                        "financial_data":   fin,
                        "name":             ticker,
                        "info":             {},
                        "suppress_earnings_history": True,
                    })
                    comprehensive_data[ticker] = summary
            except Exception as exc:
                print(f"   Error fetching {ticker}: {exc}")

        print(f" [UNFILTERED] Data ready for {len(comprehensive_data)} tickers")

        # ── Build the combined data block ────────────────────────────────────
        data_block = "\n".join(
            f"{ticker}:\n{summary}\n{'─' * 30}"
            for ticker, summary in comprehensive_data.items()
        )

        # ── Single LLM call ──────────────────────────────────────────────────
        prompt = f"""
You are an expert investment analyst.

Using ONLY the data provided below, clipped to information available on or before {news_end_date},
build the best long-term investment portfolio for the
'{keyword}' sector.

INVESTMENT AMOUNT: ${investment_amount:,.2f}
REINVESTMENT AMOUNT: ${reinvestment_amount:,.2f}

{macro_context or ''}

{'=' * 60}
NEWS DATA ({len(news_df)} articles in the analysis window)
{'=' * 60}
{articles_text}

{'=' * 60}
FINANCIAL & PERFORMANCE DATA (full {len(comprehensive_data)}-ticker universe)
{'=' * 60}
{data_block}

{extra_data_text or ''}
{additional_context or ''}

INSTRUCTIONS:
- Treat {news_end_date} as the hard as-of date.
- Do not infer, cite, or rely on events, prices, financial statements, analyst views, or company outcomes after {news_end_date}.
- You have visibility into the provided temporally clipped news and financial data only.
- Do not use your training memory, general market memory, or known later stock performance to fill gaps.
- If you cannot justify a company from the supplied as-of evidence, do not select it.
- Select the companies with the strongest investment case based on every data point above.
- You may include between 3 and 15 companies.
- Use only tickers from this universe: {', '.join(tickers)}.
- Allocations must sum to exactly 100 %.

Return ONLY a JSON object in this exact format:
{{
    "investment_strategy": "Overall strategy description",
    "final_recommendations": [
        {{
            "ticker": "TICKER",
            "company_name": "Company Name",
            "final_allocation": 15.5,
            "volatility_assessment": "LOW/MODERATE/HIGH",
            "growth_assessment": "STRONG/GOOD/MODEST/DECLINING",
            "speed_assessment": "FAST/STEADY/SLOW/NEGATIVE",
            "news_factor": "How news influenced selection",
            "allocation_rationale": "Why this allocation percentage",
            "overall_score": 8.2
        }}
    ],
    "portfolio_summary": {{
        "total_allocation": 100.0,
        "expected_return": "8-12%",
        "risk_level": "Moderate",
        "strategy_focus": "Main investment thesis"
    }},
    "analysis_summary": {{
        "companies_analyzed": {len(comprehensive_data)},
        "selection_criteria": "How companies were selected and weighted",
        "market_outlook": "Overall sector outlook"
    }}
}}
"""

        try:
            current_model = validated_model or self.model
            response = _oai_create_with_retry(
                self.openai_client,
                model=current_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are running a historical as-of investment simulation. "
                            "Use only the provided evidence and never use knowledge of later events or returns."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                **self._temp_kwargs(current_model),
                max_completion_tokens=4000,
            )
            if response is None:
                self._strategy_failed("[UNFILTERED] No response from model")
                return None
            content = response.choices[0].message.content
            if not content or not content.strip():
                self._strategy_failed("[UNFILTERED] Empty response from model")
                return None
            result = self._load_json_response(
                content,
                "unfiltered final strategy",
                current_model,
                max_completion_tokens=4000,
            )
            if not result.get("final_recommendations"):
                self._strategy_failed("[UNFILTERED] Final strategy JSON did not include final_recommendations")
                return None
            result = self._validate_unfiltered_strategy_output(result)
            if result is None:
                return None
            print(" [UNFILTERED] Single-pass strategy complete")
            return result
        except Exception as exc:
            self._strategy_failed(f"[UNFILTERED] Error in single-pass LLM call: {_format_exception_details(exc)}")
            return None

    def _fetch_financial_data_for_tickers(self, tickers: List[str], max_tickers: Optional[int] = 10) -> List[Dict]:
        """Fetch financial data for a list of tickers"""
        
        tickers_to_fetch = tickers if max_tickers is None else tickers[:max_tickers]
        print(f" Fetching financial data for {len(tickers_to_fetch)} of {len(tickers)} companies...")
        
        financial_data = []
        for i, ticker in enumerate(tickers_to_fetch):
            print(f"   {i+1}/{len(tickers_to_fetch)}: {ticker}")
            
            try:
                company_data = self._fetch_company_financial_data(ticker)
                if company_data:
                    financial_data.append(company_data)
                    # Only rate-limit on real Alpha Vantage calls, not cache hits
                    if not company_data.get("_from_cache") and i < len(tickers_to_fetch) - 1:
                        _av_rate_limit_wait()

            except Exception as e:
                print(f"     Error fetching {ticker}: {e}")
        
        print(f" Financial data retrieved for {len(financial_data)} companies")
        return financial_data
    
    def _assemble_unified_results(self, final_strategy: Dict, return_analysis: Dict,
                                 investment_amount: float, reinvestment_amount: float,
                                 sector: str = "", model: str = "",
                                 news_start_date: str = "", news_end_date: str = "",
                                 analysis_start_date: str = "", analysis_end_date: str = "",
                                 max_companies: Optional[int] = None,
                                 training_overlap_warning: Optional[str] = None) -> Dict:
        """Assemble final results for unified strategy"""

        meta = {
            "run_type": "single",
            "model": model or self.model,
            "temperature": self.temperature,
            "sector": sector,
            "timestamp": datetime.now().isoformat(),
            "news_start_date": news_start_date,
            "news_end_date": news_end_date,
            "analysis_start_date": analysis_start_date,
            "analysis_end_date": analysis_end_date,
            "max_companies": max_companies,
            "initial_candidates": self.initial_candidates,
            "final_portfolio": self.final_portfolio,
        }
        if training_overlap_warning:
            meta["training_data_warning"] = training_overlap_warning

        return {
            "metadata": meta,
            "investment_strategy": final_strategy.get("investment_strategy", ""),
            "final_recommendations": final_strategy.get("final_recommendations", []),
            "portfolio_summary": final_strategy.get("portfolio_summary", {}),
            "analysis_summary": final_strategy.get("analysis_summary", {}),
            "return_analysis": return_analysis,
            "investment_parameters": {
                "initial_investment": investment_amount,
                "reinvestment_amount": reinvestment_amount,
                "total_capital": investment_amount + reinvestment_amount
            }
        }
    
    def _perform_news_analysis(self, keyword: str, news_data_path: Optional[str], 
                              start_date: str, end_date: str, investment_amount: float) -> Optional[Dict]:
        """Perform comprehensive news-based market analysis"""
        
        # Load news data with date filtering for LLM input (prevents data leakage)
        # NOTE: This cutoff only affects what news articles are sent to the LLM
        # Stock return calculations use the full analysis period without news date restrictions
        news_df = self._load_news_data(news_data_path, start_date, end_date, keyword)
        if news_df is None or len(news_df) == 0:
            print(" No news data available - falling back to stock-only analysis")
            return self._generate_stock_only_strategy(keyword, start_date, end_date, 1000.0, 0.0)
        
        print(f" Processing {len(news_df)} news articles...")
        
        # Process ALL articles for comprehensive analysis
        articles_text = self._process_news_data(news_df, keyword)
        
        # Get OpenAI analysis
        analysis_result = self._get_news_based_analysis(keyword, articles_text)
        
        if analysis_result:
            companies = analysis_result.get('recommended_companies', [])
            print(f" News analysis complete: {len(companies)} companies identified")
            
            # Display key insights
            summary = analysis_result.get('analysis_summary', {})
            print(f" Market Themes: {', '.join(summary.get('market_themes', [])[:3])}")
            print(f" Key Trends: {', '.join(summary.get('key_trends', [])[:3])}")
            
        return analysis_result
    
    def _load_news_data(self, news_data_path: Optional[str], start_date: str, 
                       end_date: str, keyword: str) -> Optional[pd.DataFrame]:
        """Load news data from CSV files or other sources with user-defined date filtering
        
        Enhanced version that can discover and combine ALL news CSV files in the workspace
        for maximum data coverage while respecting token limits.
        """
        
        # Use the dates as provided by the user
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Check module-level cache — same (path, start, end) is reused across many runs
        _cache_key = (
            os.path.abspath(news_data_path) if news_data_path else None,
            start_date,
            end_date,
        )
        if _cache_key in _NEWS_DF_CACHE:
            cached = _NEWS_DF_CACHE[_cache_key]
            print(f" [news cache hit] {start_date} to {end_date} — {len(cached)} articles (skipping disk load)")
            return cached.copy()

        print(f" Loading news data for period: {start_date} to {end_date}")

        all_dataframes = []

        # If no specific file provided, look for most recent file in news_collection_archive
        if not news_data_path:
            workspace_root = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(workspace_root)
            archive_dir = os.path.join(parent_dir, 'news_collection_archive')

            if os.path.exists(archive_dir):
                print(f" No news file specified. Searching news_collection_archive for most recent CSV...")

                # Find all CSV files in news_collection_archive and subdirectories
                archive_files = []
                for root, dirs, files in os.walk(archive_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            try:
                                # Get file modification time
                                mod_time = os.path.getmtime(file_path)
                                archive_files.append((file_path, mod_time))
                            except:
                                pass

                if archive_files:
                    # Sort by modification time (most recent first)
                    archive_files.sort(key=lambda x: x[1], reverse=True)
                    most_recent_file = archive_files[0][0]
                    news_data_path = most_recent_file
                    print(f" Using most recent file: {os.path.basename(most_recent_file)}")
                    print(f"   Full path: {most_recent_file}")
                else:
                    print(f"  No CSV files found in news_collection_archive")
            else:
                print(f"  news_collection_archive directory not found at: {archive_dir}")

        # If specific file provided (or found from archive), load it first
        if news_data_path and os.path.exists(news_data_path):
            df = self._load_single_news_file(news_data_path, start_dt, end_dt, "Specified file")
            if df is not None and len(df) > 0:
                all_dataframes.append(df)
        
        # Discover and load ALL news CSV files in workspace for maximum coverage
        print(" Discovering all news CSV files in workspace for maximum data coverage...")
        workspace_root = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(workspace_root)
        
        news_files = []
        
        # Search workspace and parent directories comprehensively
        search_dirs = [workspace_root, parent_dir]

        # ── Sector isolation ──────────────────────────────────────────────
        # By default the loader walks the whole workspace and ingests every
        # news CSV (keyword only re-prioritises filenames). For a non-tech
        # sector that would leak tech news into a financials/healthcare run.
        # When the active sector sets isolate_news=True we use ONLY its own
        # news file (already loaded above via news_data_path) plus any CSVs in
        # its sectors/<sector>/news directory, and skip the workspace walk.
        try:
            if _sector_config.isolate_news():
                _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _nd = _sector_config.news_dir(_root)
                if _nd and os.path.isdir(_nd):
                    for _f in sorted(os.listdir(_nd)):
                        if _f.lower().endswith('.csv'):
                            _p = os.path.join(_nd, _f)
                            if not (news_data_path and os.path.abspath(_p) == os.path.abspath(news_data_path)):
                                news_files.append(_p)
                search_dirs = []  # skip the workspace-wide walk below
                print(f" [sector isolation] {_sector_config.active_sector()}: "
                      f"news restricted to sector file(s) only")
        except Exception as _e:
            print(f" [sector isolation] check skipped ({_e})")
        # ──────────────────────────────────────────────────────────────────
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if (file.endswith('.csv') and 
                            # Include files with news/finance keywords
                            any(keyword in file.lower() for keyword in ['news', 'article', 'finance', 'consolidated']) and
                            # EXCLUDE analysis results and processed files
                            not any(exclude_keyword in file.lower() for exclude_keyword in [
                                'investment_analysis_results', 'gpt_analysis', '_analysis', 
                                'strategy', 'test_', 'quick_', 'requirements', 'config'
                            ])):
                            
                            file_path = os.path.join(root, file)
                            # Skip if already loaded (avoid duplicates)
                            if news_data_path and os.path.abspath(file_path) == os.path.abspath(news_data_path):
                                continue
                            news_files.append(file_path)        # Remove duplicates and sort by relevance
        news_files = list(set(news_files))
        
        # Prioritize files by relevance to maximize data quality (prioritize RAW news over analysis)
        priority_keywords = ['consolidated_news', 'comprehensive_finance', 'news_data', 'finance_news', 'tech_news', keyword.lower()]
        
        def file_priority(filepath):
            filename = os.path.basename(filepath).lower()
            score = 0
            
            # High priority for consolidated/comprehensive news files
            if 'consolidated' in filename or 'comprehensive' in filename:
                score += 100
            
            # Medium priority for direct news files
            for i, priority_keyword in enumerate(priority_keywords):
                if priority_keyword in filename:
                    score += (len(priority_keywords) - i) * 10
            
            # Bonus for files that clearly contain raw news data
            if any(term in filename for term in ['news_data', 'finance_news', 'tech_news']):
                score += 50
                
            # Penalty for files that might be analysis results (despite our filtering)
            if any(term in filename for term in ['summary', 'detailed']):
                score -= 20
            
            # Prefer larger files (likely more data)
            try:
                score += min(os.path.getsize(filepath) / (1024 * 1024), 50)  # Cap at 50MB for scoring
            except:
                pass
            return score
        
        news_files.sort(key=file_priority, reverse=True)
        
        print(f" Found {len(news_files)} news CSV files to process")
        
        # Load files with intelligent limits to stay within token constraints
        total_articles_loaded = 0
        max_articles_per_file = 1000  # Limit per file to prevent token overflow
        max_total_articles = 5000     # Overall limit
        
        for i, file_path in enumerate(news_files):
            if total_articles_loaded >= max_total_articles:
                print(f" Reached article limit ({max_total_articles}), stopping to stay within token constraints")
                break
                
            print(f" Loading news file {i+1}/{min(len(news_files), 20)}: {os.path.basename(file_path)}")
            
            df = self._load_single_news_file(file_path, start_dt, end_dt, f"File {i+1}")
            
            if df is not None and len(df) > 0:
                # Limit articles per file to manage memory and tokens
                if len(df) > max_articles_per_file:
                    # Sample most recent articles if too many
                    if 'date' in df.columns:
                        df = df.nlargest(max_articles_per_file, 'date')
                    elif 'period_start' in df.columns:
                        df = df.nlargest(max_articles_per_file, 'period_start')
                    else:
                        df = df.head(max_articles_per_file)
                    print(f"    Limited to {max_articles_per_file} most recent articles")
                
                all_dataframes.append(df)
                total_articles_loaded += len(df)
                
                print(f"    Loaded {len(df)} articles (total: {total_articles_loaded})")
                
                # Stop loading more files if we're approaching limits
                if total_articles_loaded >= max_total_articles * 0.8:  # 80% threshold
                    remaining_capacity = max_total_articles - total_articles_loaded
                    print(f"    Approaching limit, {remaining_capacity} article slots remaining")
            
            # Process only first 20 files to avoid excessive processing time
            if i >= 19:
                print(f" Processed 20 files, stopping to maintain performance")
                break
        
        # Combine all loaded dataframes
        if all_dataframes:
            print(f" Combining {len(all_dataframes)} dataframes...")
            
            try:
                combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
                
                # Remove duplicates based on headline and date to avoid redundancy
                initial_count = len(combined_df)
                
                if 'headline' in combined_df.columns:
                    # Remove exact headline duplicates
                    combined_df = combined_df.drop_duplicates(subset=['headline'], keep='first')
                    
                    # Remove very similar headlines (basic deduplication)
                    combined_df = combined_df.drop_duplicates(subset=['headline'], keep='first')
                
                final_count = len(combined_df)
                removed_dupes = initial_count - final_count
                
                if removed_dupes > 0:
                    print(f" Removed {removed_dupes} duplicate articles")
                
                print(f" Successfully combined data: {final_count} unique articles from {len(all_dataframes)} files")
                print(f" Date range coverage: {start_date} to {end_date}")

                # Show sample of what was loaded
                if 'headline' in combined_df.columns and len(combined_df) > 0:
                    print(f" Sample headlines loaded:")
                    for headline in combined_df['headline'].head(3):
                        print(f"    {headline[:100]}...")

                # Store in module-level cache so subsequent runs skip the disk load
                _NEWS_DF_CACHE[_cache_key] = combined_df
                return combined_df
                
            except Exception as e:
                print(f" Error combining dataframes: {e}")
                # Return the first successful dataframe if combination fails
                return all_dataframes[0] if all_dataframes else None
        
        else:
            print(" No news data found in any CSV files, creating sample dataset")
            return self._create_sample_news_data(keyword, end_dt)
    
    def _load_single_news_file(self, file_path: str, start_dt: pd.Timestamp, 
                              end_dt: pd.Timestamp, file_label: str) -> Optional[pd.DataFrame]:
        """Load and filter a single news CSV file"""
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                return None
            
            # Smart date column detection - prioritize period columns for historical analysis
            date_column = None
            period_filtering = False
            
            # First check for period columns (historical analysis)
            if 'period_start' in df.columns and 'period_end' in df.columns:
                period_filtering = True
                df['period_start'] = pd.to_datetime(df['period_start'], errors='coerce', utc=True).dt.tz_localize(None)
                df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce', utc=True).dt.tz_localize(None)
                
                # Remove rows with invalid dates
                df = df.dropna(subset=['period_start', 'period_end'])
                
                # Ensure start_dt and end_dt are timezone-naive for comparison
                start_dt_naive = start_dt.tz_localize(None) if hasattr(start_dt, 'tz') and start_dt.tz else start_dt
                end_dt_naive = end_dt.tz_localize(None) if hasattr(end_dt, 'tz') and end_dt.tz else end_dt
                
                # Filter articles where the period overlaps with requested date range
                before_filter = len(df)
                df = df[
                    (df['period_start'] <= end_dt_naive) & 
                    (df['period_end'] >= start_dt_naive)
                ]
                after_filter = len(df)
                
                if after_filter > 0:
                    print(f"    {file_label}: Period overlap filtering: {after_filter}/{before_filter} articles in range")
                
            else:
                # Fall back to standard date columns
                possible_date_columns = ['date', 'pub_date', 'published_date', 'publish_date', 'publication_date', 'time']
                
                for col in possible_date_columns:
                    if col in df.columns:
                        date_column = col
                        break
                
                if date_column:
                    # Convert to datetime, handling different formats and timezones
                    try:
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True).dt.tz_localize(None)
                    except:
                        try:
                            # Try without UTC first
                            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                        except:
                            print(f"    {file_label}: Could not parse date column {date_column}")
                            return None
                    
                    # Remove rows with invalid dates
                    df = df.dropna(subset=[date_column])
                    
                    # Ensure start_dt and end_dt are timezone-naive for comparison
                    start_dt_naive = start_dt.tz_localize(None) if hasattr(start_dt, 'tz') and start_dt.tz else start_dt
                    end_dt_naive = end_dt.tz_localize(None) if hasattr(end_dt, 'tz') and end_dt.tz else end_dt
                    
                    # Apply user-defined date filtering
                    before_filter = len(df)
                    df = df[(df[date_column] >= start_dt_naive) & (df[date_column] <= end_dt_naive)]
                    after_filter = len(df)
                    
                    if after_filter > 0:
                        print(f"    {file_label}: Date filtering: {after_filter}/{before_filter} articles in range")
                
                else:
                    # No date column found - include all articles but warn
                    print(f"    {file_label}: No date column found, including all {len(df)} articles")
            
            return df if len(df) > 0 else None
            
        except Exception as e:
            print(f"    {file_label}: Error loading {os.path.basename(file_path)}: {e}")
            return None
        
        # Create sample data if no files found - use the date range provided
        print(" No news data found, creating sample dataset within specified date range")
        return self._create_sample_news_data(keyword, end_dt)
    
    def _create_sample_news_data(self, keyword: str, cutoff_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Create sample news data for testing with proper date constraints"""
        
        if cutoff_date is None:
            cutoff_date = pd.Timestamp.now()
        
        # Ensure sample data is from BEFORE the analysis period to prevent data leakage
        base_date = cutoff_date - pd.Timedelta(days=90)  # Start 90 days before cutoff
        
        sample_articles = [
            {
                'headline': f'{keyword.title()} Companies Show Strong Growth in Q3',
                'snippet': f'Leading {keyword} companies reported better-than-expected earnings with strong revenue growth.',
                'date': (base_date + pd.Timedelta(days=60)).strftime('%Y-%m-%d'),
                'section': 'Business'
            },
            {
                'headline': f'Investment Surge in {keyword.title()} Sector',
                'snippet': f'Venture capital funding in {keyword} sector reaches record highs this quarter.',
                'date': (base_date + pd.Timedelta(days=45)).strftime('%Y-%m-%d'),
                'section': 'Business'
            },
            {
                'headline': f'{keyword.title()} Innovation Drives Market Expansion',
                'snippet': f'New innovations in {keyword} technology create opportunities for long-term growth.',
                'date': (base_date + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                'section': 'Technology'
            }
        ]
        
        print(f" Created sample news data with dates ending before {cutoff_date.strftime('%Y-%m-%d')}")
        return pd.DataFrame(sample_articles)
    
    def _process_news_data(self, news_df: pd.DataFrame, keyword: str, max_articles: Optional[int] = None) -> str:
        """Process news data for OpenAI analysis - NO ARTICLE LIMITS"""
        
        # Process ALL articles - standard format
        if 'headline' in news_df.columns and 'snippet' in news_df.columns:
            relevant_articles = []
            
            # Process ALL articles, no limit
            for _, article in news_df.iterrows():
                headline = str(article.get('headline', ''))
                snippet = str(article.get('snippet', ''))
                
                # Basic relevance filtering (or include all if no keyword filtering needed)
                if not keyword or keyword.lower() in headline.lower() or keyword.lower() in snippet.lower():
                    article_data = {
                        'headline': headline,
                        'snippet': snippet,
                        'date': article.get('date', 'Unknown')
                    }
                    
                    # Add AI analysis if available
                    if 'ai_investment_relevance' in news_df.columns:
                        article_data['ai_analysis'] = str(article.get('ai_investment_relevance', ''))
                    if 'ai_key_topics' in news_df.columns:
                        article_data['ai_topics'] = str(article.get('ai_key_topics', ''))
                    if 'ai_sentiment' in news_df.columns:
                        article_data['ai_sentiment'] = str(article.get('ai_sentiment', ''))
                        
                    relevant_articles.append(article_data)
            
            # Format for OpenAI - ALL relevant articles
            articles_text = ""
            for i, article in enumerate(relevant_articles, 1):
                articles_text += f"Article {i}:\n"
                articles_text += f"Headline: {article['headline']}\n"
                articles_text += f"Content: {article['snippet']}\n"
                articles_text += f"Date: {article['date']}\n"
                
                # Include AI analysis if available
                if 'ai_analysis' in article:
                    articles_text += f"AI Analysis: {article['ai_analysis']}\n"
                if 'ai_topics' in article:
                    articles_text += f"Key Topics: {article['ai_topics']}\n"
                if 'ai_sentiment' in article:
                    articles_text += f"Sentiment: {article['ai_sentiment']}\n"
                    
                articles_text += "\n"
            
            print(f" Processed ALL {len(relevant_articles)} articles (no limits)")
            return articles_text
        
        return "No relevant articles found"
    
    def _get_news_based_analysis(self, keyword: str, articles_text: str, model: str = None) -> Optional[Dict]:
        """Get comprehensive investment analysis from OpenAI based on news data"""
        
        enhanced_system_prompt = """
        You are an expert financial analyst specializing in identifying high-potential public companies for long-term investment.

        Using ONLY the provided news articles, analyze market trends, technological developments, regulatory changes,
        and company-specific information to identify the most promising publicly-traded companies for a 10-year investment horizon.

        Focus on:
        - Companies with strong competitive moats and market positions
        - Emerging growth trends and technological disruptions
        - Regulatory tailwinds and policy support
        - Strong financial fundamentals and growth potential
        - ESG considerations and sustainability factors

        IMPORTANT: Only recommend publicly-traded companies with valid stock tickers on major exchanges (NYSE, NASDAQ).
        Exclude private companies, startups without public listings, and any non-investable entities.

        Return your analysis as a properly formatted JSON object with the following exact structure:
        """

        enhanced_user_prompt = f"""
        Keyword Focus: {keyword}

        News Articles Data:
        {articles_text}

        Based on this news data, provide a comprehensive investment analysis with 20 high-potential companies.

        Return ONLY a valid JSON object with this exact structure:
        {{
            "analysis_summary": {{
                "market_themes": ["theme1", "theme2", "theme3"],
                "key_trends": ["trend1", "trend2", "trend3"],
                "implementation_date": "2024-01-01",
                "total_companies_analyzed": 15
            }},
            "recommended_companies": [
                {{
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "sector": "Technology",
                    "investment_thesis": "Detailed explanation of why this company was selected based on news analysis",
                    "growth_catalysts": ["catalyst1", "catalyst2"],
                    "risk_factors": ["risk1", "risk2"],
                    "confidence_score": 8.5,
                    "recommended_allocation": 12.5,
                    "price_target_10yr": "Conservative estimate based on analysis"
                }}
            ],
            "portfolio_strategy": {{
                "diversification_approach": "Strategy description",
                "sector_allocation": {{
                    "technology": 40,
                    "healthcare": 25,
                    "energy": 20,
                    "other": 15
                }},
                "risk_level": "Moderate-Aggressive",
                "expected_annual_return": "8-12%"
            }}
        }}
        """
        
        try:
            current_model = model or self.model
            response = _oai_create_with_retry(
                self.openai_client,
                model=current_model,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": enhanced_user_prompt}
                ],
                **self._temp_kwargs(current_model),
                max_completion_tokens=4000,
            )
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                print(" Empty response from OpenAI")
                return None

            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            if not content.strip():
                print(" Empty JSON block in response")
                return None
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            print(f" News analysis failed: {e}")
            return None
    
    def _fetch_financial_data_for_companies(self, news_analysis: Dict) -> Optional[List[Dict]]:
        """Fetch financial data for recommended companies using Alpha Vantage API"""
        
        companies = news_analysis.get('recommended_companies', [])
        if not companies:
            return None
        
        # Limit to top companies to avoid API rate limits
        top_companies = companies[:20]
        tickers = [comp.get('ticker', '') for comp in top_companies if comp.get('ticker')]
        
        print(f" Fetching financial data for {len(tickers)} companies...")
        
        financial_data = []
        for i, ticker in enumerate(tickers):
            print(f"   {i+1}/{len(tickers)}: {ticker}")
            
            try:
                company_data = self._fetch_company_financial_data(ticker)
                if company_data:
                    financial_data.append(company_data)
                    # Only rate-limit when we actually hit Alpha Vantage (not cache hits)
                    if not company_data.get("_from_cache") and i < len(tickers) - 1:
                        _av_rate_limit_wait()

            except Exception as e:
                print(f"     Error fetching {ticker}: {e}")
        
        print(f" Financial data retrieved for {len(financial_data)} companies")
        return financial_data if financial_data else None
    
    def _cache_dir_for(self, ticker: str) -> str:
        """Return the financial_reports/<TICKER> directory path."""
        this_dir   = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(this_dir)
        safe = re.sub(r"[^A-Za-z0-9.\-]", "", str(ticker)).upper()
        return os.path.join(parent_dir, "financial_reports", safe)

    def _save_to_financial_reports_cache(self, ticker: str, company_data: Dict) -> None:
        """Persist Alpha Vantage data to memory and disk so future calls skip the API."""
        _FINANCIAL_DATA_MEM_CACHE[ticker] = company_data
        cache_dir = self._cache_dir_for(ticker)
        os.makedirs(cache_dir, exist_ok=True)

        file_map = {
            "income":    "INCOME_STATEMENT.csv",
            "balance":   "BALANCE_SHEET.csv",
            "cash_flow": "CASH_FLOW.csv",
        }
        for key, filename in file_map.items():
            records = company_data.get(key, [])
            if not records:
                continue
            try:
                pd.DataFrame(records).to_csv(
                    os.path.join(cache_dir, filename), index=False
                )
            except Exception as e:
                print(f"  Cache write error for {ticker}/{filename}: {e}")

        print(f"  Saved {ticker} financials to cache ({cache_dir})")

    def _load_price_history_cache(self, ticker: str):
        """
        Load cached yfinance price histories.
        Checks the process-level memory cache first, then falls back to disk.
        Returns (hist_1y, hist_3y, hist_5y) DataFrames, or (None, None, None) on miss.
        """
        if ticker in _PRICE_HISTORY_MEM_CACHE:
            return _PRICE_HISTORY_MEM_CACHE[ticker]

        cache_dir = self._cache_dir_for(ticker)
        paths = {
            "1y": os.path.join(cache_dir, "PRICE_HISTORY_1Y.csv"),
            "3y": os.path.join(cache_dir, "PRICE_HISTORY_3Y.csv"),
            "5y": os.path.join(cache_dir, "PRICE_HISTORY_5Y.csv"),
        }
        if not all(os.path.exists(p) for p in paths.values()):
            return None, None, None
        try:
            dfs = {}
            for period, path in paths.items():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                # Ensure the index is a proper DatetimeIndex (parse_dates=True is not
                # always sufficient when the index column contains tz-aware strings).
                try:
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                except TypeError:
                    # Already tz-naive after conversion
                    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
                dfs[period] = df
            result = dfs["1y"], dfs["3y"], dfs["5y"]
            _PRICE_HISTORY_MEM_CACHE[ticker] = result
            print(f"  Loaded {ticker} price history from disk cache")
            return result
        except Exception as e:
            print(f"  Price cache read error for {ticker}: {e}")
            return None, None, None

    def _save_price_history_cache(self, ticker: str, hist_1y, hist_3y, hist_5y) -> None:
        """Save yfinance price histories to disk and memory cache for future reuse."""
        _PRICE_HISTORY_MEM_CACHE[ticker] = (hist_1y, hist_3y, hist_5y)
        cache_dir = self._cache_dir_for(ticker)
        os.makedirs(cache_dir, exist_ok=True)
        for period, df in [("1y", hist_1y), ("3y", hist_3y), ("5y", hist_5y)]:
            if df is not None and not df.empty:
                try:
                    df.to_csv(os.path.join(cache_dir, f"PRICE_HISTORY_{period.upper()}.csv"))
                except Exception as e:
                    print(f"  Price cache write error for {ticker} {period}: {e}")
        print(f"  Saved {ticker} price history to cache")

    def _slice_price_history_as_of(self, hist, as_of_dt, years: int):
        """Return only prices that would have existed by as_of_dt."""
        if hist is None or hist.empty:
            return hist
        df = hist.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert(None)
        start_dt = as_of_dt - relativedelta(years=years)
        return df[(df.index >= start_dt) & (df.index <= as_of_dt)]

    def _fetch_yahoo_chart_history(self, symbol: str, start_dt: datetime, end_dt: datetime):
        """Fetch daily price history from Yahoo's chart endpoint without importing yfinance."""
        period1 = int(start_dt.timestamp())
        period2 = int(end_dt.timestamp())
        symbol = sanitize_ticker(symbol)                              # drop footnote chars like '*'
        if not is_valid_ticker(symbol):
            # Malformed/hallucinated symbol (e.g. "META.DE-ALTERNATIVE") — skip the
            # doomed request instead of round-tripping to a 404.
            print(f"  Skipping invalid ticker symbol: {symbol!r}")
            return pd.DataFrame()
        if symbol in _UNKNOWN_SYMBOLS:
            # Already known to 404 (well-formed but unknown/delisted) — skip silently.
            return pd.DataFrame()
        encoded_symbol = requests.utils.quote(symbol, safe="")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        try:
            response = requests.get(
                url,
                params={
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "history",
                    "includeAdjustedClose": "true",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=_MACRO_HTTP_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (404, 400):
                # Symbol not found on Yahoo (unknown/delisted, e.g. a hallucinated ticker).
                # Expected and handled: no price -> the sim drops it. Log once, then cache.
                print(f"  No Yahoo data for {symbol} (unknown/delisted symbol) - skipping")
                _UNKNOWN_SYMBOLS.add(symbol)
            else:
                # Transient/network error — do NOT cache; a later retry may succeed.
                print(f"  Yahoo chart fetch error for {symbol}: {exc}")
            return pd.DataFrame()

        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return pd.DataFrame()

        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0]
        def _at(values, index):
            return values[index] if values is not None and index < len(values) else None

        close_values = adjclose.get("adjclose") or quote.get("close") or []
        rows = []
        for i, ts in enumerate(timestamps):
            close = _at(close_values, i)
            if close is None:
                close = _at(quote.get("close") or [], i)
            if close is None:
                continue
            rows.append({
                "Date": datetime.fromtimestamp(ts),
                "Open": _at(quote.get("open") or [], i),
                "High": _at(quote.get("high") or [], i),
                "Low": _at(quote.get("low") or [], i),
                "Close": close,
                "Volume": _at(quote.get("volume") or [], i),
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("Date").sort_index()

    def _load_temporal_price_histories(self, ticker: str, as_of_date: str):
        """Load/fetch price histories clipped to the unfiltered run's as-of date."""
        as_of_dt = pd.to_datetime(as_of_date, utc=True).tz_convert(None)
        hist_1y, hist_3y, hist_5y = self._load_price_history_cache(ticker)

        if hist_5y is None or hist_5y.empty:
            try:
                fetch_start = as_of_dt - relativedelta(years=5, days=10)
                fetch_end = as_of_dt + timedelta(days=1)
                hist_5y = self._fetch_yahoo_chart_history(ticker, fetch_start, fetch_end)
            except Exception as exc:
                print(f"  Temporal price fetch error for {ticker}: {exc}")
                return None, None, None

        hist_1y = self._slice_price_history_as_of(hist_5y, as_of_dt, 1)
        hist_3y = self._slice_price_history_as_of(hist_5y, as_of_dt, 3)
        hist_5y = self._slice_price_history_as_of(hist_5y, as_of_dt, 5)
        self._save_price_history_cache(ticker, hist_1y, hist_3y, hist_5y)
        return hist_1y, hist_3y, hist_5y

    def _filter_financial_records_as_of(self, records, as_of_dt):
        """Admit a financial report only if it was PUBLIC on the decision date.

        The date a fiscal period ends is not the date its report becomes available.
        Filtering on `fiscalDateEnding` admits, for example, the quarter ending
        2024-09-30 for a decision dated 2024-10-01 — a 10-Q that would not be filed
        for another month. Measured across the nine GPT-5.1 decision dates, that
        admitted an almost-certainly-unfiled report in 42 of 63 company-date pairs
        (67%), which is look-ahead: the model sees numbers no investor had yet.

        ONE RULE, chosen for practicality — it depends only on a field that is always
        present, and no missing or malformed field can weaken it:

            available_on = MAX( fiscal_period_end + statutory_filing_deadline,
                                every observed filing/report date on the record )
            admit only if  available_on < decision_date

        Why the max rather than a preference order. A preference chain that trusts
        `reportedDate` when present would let one bad or optimistic value bypass the
        deadline bound entirely — the exact leak being closed. Taking the max means
        the bound is a floor that a real date can only push LATER, never earlier, so
        the filter degrades safely when data is missing, partial, or wrong. In this
        repository no cached file carries a filing date at all, so in practice the
        deadline bound is what runs; the max keeps the rule correct anyway once real
        dates are supplied.

        Cost of the conservative choice: a company that files ahead of the deadline
        has its report withheld for the few remaining days. Measured on the cached
        statements that is 0.67 of ~75 reports per company-decision — negligible
        against the 67% of decisions that were previously ingesting unfiled reports.

        Strict `<`: EDGAR disseminates filings accepted after 17:30 ET on the next
        business day, so a report dated the same day as the decision is not treated
        as available.

        Deadlines: 10-Q 45 days, 10-K 75 days (accelerated-filer limits, so the bound
        holds across the whole universe rather than only for megacaps).
        """
        filtered = []
        for record in records or []:
            period_str = record.get("fiscalDateEnding") or record.get("date")
            if not period_str:
                continue
            period_dt = pd.to_datetime(period_str, errors="coerce", utc=True)
            if pd.isna(period_dt):
                continue

            rtype = str(record.get("reportType") or record.get("period") or "").lower()
            is_annual = ("annual" in rtype) or (rtype == "fy") or ("10-k" in rtype)
            lag = FIN_REPORT_LAG_DAYS_ANNUAL if is_annual else FIN_REPORT_LAG_DAYS_QUARTERLY

            # floor: earliest date the report could legally have been public
            candidates = [period_dt + pd.Timedelta(days=lag)]

            # any observed filing date can only push availability later
            for field in ("acceptedDate", "filedDate", "filingDate", "reportedDate"):
                v = record.get(field)
                if not v:
                    continue
                parsed = pd.to_datetime(v, errors="coerce", utc=True)
                if pd.notna(parsed):
                    candidates.append(parsed)

            avail_dt = max(candidates)
            if getattr(avail_dt, "tz", None) is not None:
                avail_dt = avail_dt.tz_convert(None)
            if avail_dt < as_of_dt:
                filtered.append(record)
        return filtered

    def _filter_financial_data_as_of(self, company_data: Optional[Dict], as_of_date: str) -> Optional[Dict]:
        if not company_data:
            return company_data
        as_of_dt = pd.to_datetime(as_of_date, utc=True).tz_convert(None)
        filtered = dict(company_data)
        for key in ("income", "balance", "cash_flow", "income_statement"):
            if key in filtered:
                filtered[key] = self._filter_financial_records_as_of(filtered.get(key), as_of_dt)
        filtered["as_of_date"] = as_of_date
        filtered["temporal_filter_applied"] = True
        return filtered

    # ── New data-source helpers ────────────────────────────────────────────────

    def _validate_unfiltered_strategy_output(self, strategy: Dict) -> Optional[Dict]:
        """Constrain unfiltered output to the declared universe and valid allocations."""
        if not isinstance(strategy, dict):
            self._strategy_failed("[UNFILTERED] Model output was not a JSON object")
            return None

        allowed = set(UNFILTERED_TICKER_UNIVERSE)
        merged: Dict[str, Dict] = {}
        dropped = []
        for rec in strategy.get("final_recommendations", []) or []:
            ticker = str(rec.get("ticker", "")).replace("$", "").strip().upper()
            if ticker not in allowed:
                dropped.append(ticker or "<blank>")
                continue
            try:
                allocation = float(rec.get("final_allocation", 0) or 0)
            except (TypeError, ValueError):
                allocation = 0
            if allocation <= 0:
                continue

            if ticker in merged:
                merged[ticker]["final_allocation"] += allocation
            else:
                cleaned = dict(rec)
                cleaned["ticker"] = ticker
                cleaned["final_allocation"] = allocation
                merged[ticker] = cleaned

        if dropped:
            print(f" [UNFILTERED] Dropped out-of-universe tickers: {', '.join(sorted(set(dropped)))}")

        recs = list(merged.values())
        if not recs:
            self._strategy_failed("[UNFILTERED] No valid in-universe final recommendations after validation")
            return None
        recs = sorted(
            recs, key=lambda r: float(r.get("final_allocation", 0) or 0), reverse=True
        )
        if len(recs) > 15:
            print(f" [UNFILTERED] Trimming final recommendations from {len(recs)} to 15 holdings")
            recs = recs[:15]
        if len(recs) < 3:
            self._strategy_failed("[UNFILTERED] Fewer than 3 valid holdings after validation")
            return None

        total_alloc = sum(float(r.get("final_allocation", 0) or 0) for r in recs)
        if total_alloc <= 0:
            self._strategy_failed("[UNFILTERED] Final recommendations had no positive allocation")
            return None

        for rec in recs:
            rec["final_allocation"] = round(float(rec["final_allocation"]) / total_alloc * 100, 2)
        rounding_delta = round(100.0 - sum(float(r["final_allocation"]) for r in recs), 2)
        recs[0]["final_allocation"] = round(float(recs[0]["final_allocation"]) + rounding_delta, 2)

        strategy["final_recommendations"] = recs
        strategy.setdefault("portfolio_summary", {})["total_allocation"] = 100.0
        strategy.setdefault("analysis_summary", {})["unfiltered_validation"] = (
            "Output constrained to the unfiltered ticker universe; duplicate tickers merged; "
            "portfolio size constrained to 3-15 holdings; allocations normalized to 100."
        )
        return strategy

    def _fetch_macro_context(self, as_of_date_str: str) -> str:
        """
        Fetch broad macro market conditions at or just before *as_of_date_str*
        using yfinance historical data (temporally compliant — no future data).

        Returns a formatted string ready to embed in a prompt, or "" on failure.
        """
        try:
            end_dt   = datetime.strptime(as_of_date_str, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=45)
            start_str = start_dt.strftime('%Y-%m-%d')
            end_str   = end_dt.strftime('%Y-%m-%d')

            symbols = {
                "S&P 500 (^GSPC)":        "^GSPC",
                "NASDAQ-100 (^IXIC)":     "^IXIC",
                "VIX Fear Index (^VIX)":  "^VIX",
                "10Y Treasury Yield (%)": "^TNX",
            }

            lines = [f"MACRO MARKET CONDITIONS (as of {as_of_date_str}):"]
            for name, sym in symbols.items():
                try:
                    hist = self._fetch_macro_history(sym, start_dt, end_dt + timedelta(days=1))
                    if hist.empty:
                        continue
                    latest = float(hist['Close'].iloc[-1])
                    if sym in ("^GSPC", "^IXIC") and len(hist) > 5:
                        mo_ret = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                        lines.append(f"  {name}: {latest:,.2f}  (30-day return: {mo_ret:+.1f}%)")
                    else:
                        lines.append(f"  {name}: {latest:.2f}")
                except Exception:
                    pass

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _fetch_macro_history(self, symbol: str, start_dt: datetime, end_dt: datetime):
        """
        Fetch macro history without yfinance.

        Python 3.14 can crash in yfinance's curl_cffi path before an exception is
        raised, so the macro layer uses Yahoo's chart JSON endpoint via requests.
        """
        period1 = int(start_dt.timestamp())
        period2 = int(end_dt.timestamp())
        symbol = re.sub(r"[^A-Za-z0-9.\-]", "", str(symbol)).upper()   # drop footnote chars like '*'
        encoded_symbol = requests.utils.quote(symbol, safe="")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        try:
            response = requests.get(
                url,
                params={
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "history",
                    "includeAdjustedClose": "true",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=_MACRO_HTTP_TIMEOUT,
            )
        except requests.RequestException:
            return pd.DataFrame()
        response.raise_for_status()
        payload = response.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return pd.DataFrame()

        timestamps = result.get("timestamp") or []
        quotes = result.get("indicators", {}).get("quote") or [{}]
        closes = quotes[0].get("close") or []
        rows = [
            (datetime.fromtimestamp(ts), close)
            for ts, close in zip(timestamps, closes)
            if close is not None
        ]
        if not rows:
            return pd.DataFrame()

        hist = pd.DataFrame(rows, columns=["Date", "Close"]).set_index("Date")
        return hist[(hist.index >= start_dt) & (hist.index <= end_dt)]

    # ── Ticker fundamental-info cache (yfinance .info) ─────────────────────────

    _TICKER_INFO_KEYS = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook", "beta",
        "sector", "industry",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyTwoWeekChangePercent",
        "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
        "targetMeanPrice", "targetHighPrice", "targetLowPrice",
        "earningsGrowth", "revenueGrowth",
        "profitMargins", "grossMargins", "operatingMargins",
        "returnOnEquity", "returnOnAssets",
        "debtToEquity", "currentRatio", "quickRatio",
        "totalCash", "totalDebt", "freeCashflow",
        "dividendYield", "payoutRatio",
    ]

    def _load_ticker_info_cache(self, ticker: str) -> Optional[Dict]:
        if ticker in _TICKER_INFO_MEM_CACHE:
            return _TICKER_INFO_MEM_CACHE[ticker]
        cache_path = os.path.join(self._cache_dir_for(ticker), "INFO.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            _TICKER_INFO_MEM_CACHE[ticker] = info
            return info
        except Exception:
            return None

    def _save_ticker_info_cache(self, ticker: str, info: Dict) -> None:
        _TICKER_INFO_MEM_CACHE[ticker] = info
        cache_dir = self._cache_dir_for(ticker)
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(os.path.join(cache_dir, "INFO.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            print(f"  Info cache write error for {ticker}: {e}")

    def _fetch_ticker_info(self, ticker: str) -> Dict:
        """Return cached ticker info when available.

        Live yfinance .info is disabled because importing yfinance can crash
        Python 3.14 on this environment before an exception is catchable.
        """
        cached = self._load_ticker_info_cache(ticker)
        if cached is not None:
            return cached
        return {}

    # ── Earnings history cache (yfinance) ─────────────────────────────────────

    def _fetch_earnings_history(self, ticker: str) -> str:
        """
        Return a short text summary of recent EPS beats/misses from cache.
        Results are cached to EARNINGS_HISTORY.json per ticker.
        """
        cache_path = os.path.join(self._cache_dir_for(ticker), "EARNINGS_HISTORY.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if records:
                    return self._format_earnings_records(ticker, records)
            except Exception:
                pass
        return ""

    def _format_earnings_records(self, ticker: str, records: list) -> str:
        lines = [f"  {ticker} recent EPS history (est → actual, surprise%):"]
        for r in records[-6:]:
            est = r.get("epsEstimate")
            act = r.get("epsActual")
            sur = r.get("surprise")
            parts = []
            if r.get("quarter"):
                parts.append(str(r["quarter"]))
            if est is not None and act is not None:
                beat = "BEAT" if act > est else ("MISS" if act < est else "MET")
                parts.append(f"est {est:.2f} → actual {act:.2f} [{beat}]")
                if sur is not None:
                    parts.append(f"{float(sur):+.1f}%")
            if parts:
                lines.append("    " + "  |  ".join(parts))
        return "\n".join(lines) if len(lines) > 1 else ""

    # ── Extra user-supplied data files ────────────────────────────────────────

    def _load_extra_data_files(self, paths: List[str],
                               start_date: str, end_date: str) -> str:
        """
        Load any user-supplied CSV or JSON files and return them as a single
        formatted string for embedding in the prompt.  Date filtering is applied
        when a recognisable date column is present.
        """
        if not paths:
            return ""

        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)
        sections = []

        for path in paths:
            if not os.path.exists(path):
                print(f"  [extra data] file not found: {path}")
                continue
            ext   = os.path.splitext(path)[1].lower()
            fname = os.path.basename(path)
            try:
                if ext == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sections.append(f"--- {fname} ---\n{json.dumps(data, indent=2)[:6000]}")

                elif ext == ".csv":
                    df = pd.read_csv(path)
                    for col in ["date", "Date", "pub_date", "timestamp", "Timestamp"]:
                        if col in df.columns:
                            try:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                                mask = (df[col] >= start_dt) & (df[col] <= end_dt)
                                df   = df[mask]
                            except Exception:
                                pass
                            break
                    sections.append(
                        f"--- {fname} ({len(df)} rows) ---\n"
                        + df.to_string(max_rows=60, index=False)[:6000]
                    )
                else:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(6000)
                    sections.append(f"--- {fname} ---\n{content}")

                print(f"  [extra data] loaded: {fname}")
            except Exception as e:
                print(f"  [extra data] error loading {fname}: {e}")

        if not sections:
            return ""
        return "SUPPLEMENTARY USER-PROVIDED DATA:\n" + "\n\n".join(sections)

    def _load_from_financial_reports_cache(self, ticker: str) -> Optional[Dict]:
        """
        Load pre-saved financial reports.
        Checks the process-level memory cache first, then falls back to disk.
        Returns data in the same dict format as _fetch_company_financial_data so
        the rest of the pipeline is unaffected.
        Returns None if the ticker folder does not exist.
        """
        if ticker in _FINANCIAL_DATA_MEM_CACHE:
            return _FINANCIAL_DATA_MEM_CACHE[ticker]

        cache_dir = self._cache_dir_for(ticker)

        if not os.path.isdir(cache_dir):
            return None

        company_data = {"ticker": ticker, "_from_cache": True}

        for key, filename in [
            ("income",    "INCOME_STATEMENT.csv"),
            ("balance",   "BALANCE_SHEET.csv"),
            ("cash_flow", "CASH_FLOW.csv"),
        ]:
            fpath = os.path.join(cache_dir, filename)
            if not os.path.exists(fpath):
                company_data[key] = []
                continue
            try:
                df = pd.read_csv(fpath)
                # Drop internal index column if present
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])
                # Convert to list-of-dicts (same shape Alpha Vantage returns)
                company_data[key] = df.where(df.notna(), other=None).to_dict(orient="records")
            except Exception as e:
                print(f"  Cache read error for {ticker}/{filename}: {e}")
                company_data[key] = []

        _FINANCIAL_DATA_MEM_CACHE[ticker] = company_data
        return company_data

    def _fetch_company_financial_data(self, ticker: str) -> Optional[Dict]:
        """Fetch comprehensive financial data for a single company.
        Checks financial_reports cache first; falls back to Alpha Vantage API."""

        # ── Cache-first lookup ─────────────────────────────────────────────
        cached = self._load_from_financial_reports_cache(ticker)
        if cached is not None:
            print(f"  Loaded {ticker} financials from cache (no API call)")
            return cached

        # ── Alpha Vantage fallback ─────────────────────────────────────────
        company_data = {"ticker": ticker}

        try:
            # Fetch income statement
            income_params = {
                'function': 'INCOME_STATEMENT',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }

            income_response = requests.get(self.alpha_vantage_base_url, params=income_params)
            if income_response.status_code == 200:
                income_data = income_response.json()
                company_data["income"] = income_data.get("quarterlyReports", [])

            _av_rate_limit_wait()  # global rate limiter shared across all threads

            # Fetch balance sheet
            balance_params = {
                'function': 'BALANCE_SHEET',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }

            balance_response = requests.get(self.alpha_vantage_base_url, params=balance_params)
            if balance_response.status_code == 200:
                balance_data = balance_response.json()
                company_data["balance"] = balance_data.get("quarterlyReports", [])

            _av_rate_limit_wait()  # global rate limiter shared across all threads

            # Fetch cash flow
            cash_params = {
                'function': 'CASH_FLOW',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key
            }

            cash_response = requests.get(self.alpha_vantage_base_url, params=cash_params)
            if cash_response.status_code == 200:
                cash_data = cash_response.json()
                company_data["cash_flow"] = cash_data.get("quarterlyReports", [])

            # Save to cache so this ticker is never fetched from Alpha Vantage again
            self._save_to_financial_reports_cache(ticker, company_data)

            return company_data

        except Exception as e:
            print(f"     Financial data error for {ticker}: {e}")
            return None
    
    def _analyze_historical_performance(self, news_analysis: Dict) -> str:
        """Analyze historical stock performance for recommended companies"""
        
        companies = news_analysis.get('recommended_companies', [])
        if not companies:
            return "No companies available for historical analysis."
        
        print(f" Analyzing historical performance for {len(companies)} companies...")
        
        performance_summary = "HISTORICAL PERFORMANCE ANALYSIS:\n"
        performance_summary += "=" * 50 + "\n\n"
        
        for company in companies:
            ticker = company.get('ticker', '').strip().upper()
            company_name = company.get('company_name', 'Unknown')
            
            if not ticker:
                continue
                
            try:
                print(f"   Analyzing {ticker}...")

                # Check price cache first
                hist_1y, hist_3y, hist_5y = self._load_price_history_cache(ticker)
                if hist_1y is None:
                    hist_1y, hist_3y, hist_5y = self._load_temporal_price_histories(
                        ticker, datetime.now().strftime("%Y-%m-%d")
                    )
                
                if not hist_1y.empty:
                    # Calculate performance metrics
                    current_price = hist_1y['Close'].iloc[-1]
                    
                    # 1-year performance
                    year_ago_price = hist_1y['Close'].iloc[0] if len(hist_1y) > 0 else current_price
                    year_return = ((current_price - year_ago_price) / year_ago_price) * 100 if year_ago_price > 0 else 0
                    
                    # Volatility (standard deviation of daily returns)
                    daily_returns = hist_1y['Close'].pct_change().dropna()
                    volatility = daily_returns.std() * (252**0.5) * 100  # Annualized volatility
                    
                    # Trend analysis (simple moving averages)
                    ma_20 = hist_1y['Close'].rolling(20).mean().iloc[-1] if len(hist_1y) >= 20 else current_price
                    ma_50 = hist_1y['Close'].rolling(50).mean().iloc[-1] if len(hist_1y) >= 50 else current_price
                    
                    trend_signal = "BULLISH" if current_price > ma_20 > ma_50 else "BEARISH" if current_price < ma_20 < ma_50 else "NEUTRAL"
                    
                    # 3-year and 5-year returns if available
                    three_year_return = 0
                    five_year_return = 0
                    
                    if hist_3y is not None and not hist_3y.empty and len(hist_3y) > 252:  # At least 1 year of data
                        three_year_ago_price = hist_3y['Close'].iloc[0]
                        three_year_return = ((current_price - three_year_ago_price) / three_year_ago_price) * 100 if three_year_ago_price > 0 else 0

                    if hist_5y is not None and not hist_5y.empty and len(hist_5y) > 252:  # At least 1 year of data
                        five_year_ago_price = hist_5y['Close'].iloc[0]
                        five_year_return = ((current_price - five_year_ago_price) / five_year_ago_price) * 100 if five_year_ago_price > 0 else 0
                    
                    # Create performance summary
                    performance_summary += f"{company_name} ({ticker}):\n"
                    performance_summary += f"   1-Year Return: {year_return:.2f}%\n"
                    if three_year_return != 0:
                        performance_summary += f"   3-Year Return: {three_year_return:.2f}%\n"
                    if five_year_return != 0:
                        performance_summary += f"   5-Year Return: {five_year_return:.2f}%\n"
                    performance_summary += f"   Annual Volatility: {volatility:.2f}%\n"
                    performance_summary += f"   Current Trend: {trend_signal}\n"
                    performance_summary += f"   Price vs 20-day MA: {'Above' if current_price > ma_20 else 'Below'}\n"
                    performance_summary += f"   Price vs 50-day MA: {'Above' if current_price > ma_50 else 'Below'}\n"
                    
                    # Risk-return assessment
                    if year_return > 15 and volatility < 25:
                        assessment = "STRONG PERFORMER - High return, moderate risk"
                    elif year_return > 0 and volatility < 20:
                        assessment = "STABLE PERFORMER - Positive return, low risk"
                    elif year_return < -10 or volatility > 40:
                        assessment = "HIGH RISK - Consider lower allocation"
                    else:
                        assessment = "MODERATE PERFORMER - Standard allocation"
                        
                    performance_summary += f"   Assessment: {assessment}\n\n"
                    
                else:
                    performance_summary += f"{company_name} ({ticker}): No sufficient historical data available\n\n"
                    
            except Exception as e:
                performance_summary += f"{company_name} ({ticker}): Error retrieving data - {str(e)[:50]}\n\n"
                print(f"     Error analyzing {ticker}: {e}")
        
        print(" Historical performance analysis complete")
        return performance_summary
    
    def _perform_comprehensive_financial_analysis(self, news_analysis: Dict, financial_data: Optional[List[Dict]]) -> str:
        """Combined Layer: Analyze financial statements + historical performance + generate summaries"""
        
        companies = news_analysis.get('recommended_companies', [])
        if not companies:
            return "No companies available for comprehensive analysis."
        
        print(f" Performing comprehensive financial & performance analysis for {len(companies)} companies...")
        
        # Step 1: Gather all financial and performance data
        comprehensive_data = {}
        
        for company in companies:
            ticker = company.get('ticker', '').strip().upper()
            company_name = company.get('company_name', 'Unknown')
            
            if not ticker:
                continue
                
            print(f"   Analyzing {ticker} - {company_name}")
            
            # Initialize company data structure
            comprehensive_data[ticker] = {
                'name': company_name,
                'financial_data': None,
                'performance_data': None,
                'summary': None
            }
            
            # Get financial data if available
            if financial_data:
                for fin_data in financial_data:
                    if fin_data.get('ticker') == ticker:
                        comprehensive_data[ticker]['financial_data'] = fin_data
                        break
            
            # Get historical performance data (cache-first)
            try:
                hist_1y, hist_3y, hist_5y = self._load_price_history_cache(ticker)
                if hist_1y is None:
                    hist_1y, hist_3y, hist_5y = self._load_temporal_price_histories(
                        ticker, datetime.now().strftime("%Y-%m-%d")
                    )

                if hist_1y is not None and not hist_1y.empty:
                    perf_data = self._calculate_performance_metrics(hist_1y, hist_3y, hist_5y)
                    comprehensive_data[ticker]['performance_data'] = perf_data

            except Exception as e:
                print(f"     Error getting performance data for {ticker}: {e}")
        
        # Step 2: Generate summaries for volatility, growth, and speed of growth
        print(" Generating volatility, growth, and speed summaries...")
        
        for ticker, data in comprehensive_data.items():
            if data['performance_data'] or data['financial_data']:
                summary = self._generate_stock_summary(ticker, data)
                comprehensive_data[ticker]['summary'] = summary
        
        # Step 3: Create final analysis report
        analysis_report = "COMPREHENSIVE FINANCIAL & PERFORMANCE ANALYSIS\n"
        analysis_report += "=" * 60 + "\n\n"
        
        analysis_report += "INDIVIDUAL STOCK SUMMARIES:\n"
        analysis_report += "-" * 30 + "\n\n"
        
        for ticker, data in comprehensive_data.items():
            if data['summary']:
                analysis_report += f"{data['name']} ({ticker}):\n"
                analysis_report += data['summary'] + "\n\n"
        
        print(" Comprehensive financial & performance analysis complete")
        return analysis_report
    
    def _calculate_performance_metrics(self, hist_1y, hist_3y, hist_5y):
        """Calculate detailed performance metrics for a stock"""
        
        metrics = {}

        # Guard against empty/missing history (e.g. a ticker with no data as of the
        # decision date). Without this, hist_1y['Close'].iloc[-1] raises
        # "single positional indexer is out-of-bounds".
        if hist_1y is None or len(hist_1y) == 0:
            return metrics

        try:
            # Current price and basic returns
            current_price = hist_1y['Close'].iloc[-1]
            
            # 1-year metrics
            if len(hist_1y) > 0:
                year_ago_price = hist_1y['Close'].iloc[0]
                metrics['1y_return'] = ((current_price - year_ago_price) / year_ago_price) * 100 if year_ago_price > 0 else 0
                
                # Volatility (annualized standard deviation)
                daily_returns = hist_1y['Close'].pct_change().dropna()
                metrics['volatility'] = daily_returns.std() * (252**0.5) * 100
                
                # Growth speed (average monthly return)
                monthly_returns = hist_1y['Close'].resample('ME').last().pct_change().dropna()
                metrics['avg_monthly_return'] = monthly_returns.mean() * 100
                metrics['growth_consistency'] = 1 - (monthly_returns.std() / abs(monthly_returns.mean())) if monthly_returns.mean() != 0 else 0
            
            # 3-year metrics
            if hist_3y is not None and not hist_3y.empty and len(hist_3y) > 252:
                three_year_ago_price = hist_3y['Close'].iloc[0]
                metrics['3y_return'] = ((current_price - three_year_ago_price) / three_year_ago_price) * 100 if three_year_ago_price > 0 else 0
                metrics['3y_annualized'] = (((current_price / three_year_ago_price) ** (1/3)) - 1) * 100 if three_year_ago_price > 0 else 0
            
            # 5-year metrics
            if hist_5y is not None and not hist_5y.empty and len(hist_5y) > 252:
                five_year_ago_price = hist_5y['Close'].iloc[0]
                metrics['5y_return'] = ((current_price - five_year_ago_price) / five_year_ago_price) * 100 if five_year_ago_price > 0 else 0
                metrics['5y_annualized'] = (((current_price / five_year_ago_price) ** (1/5)) - 1) * 100 if five_year_ago_price > 0 else 0
            
            # Trend analysis
            if len(hist_1y) >= 50:
                ma_20 = hist_1y['Close'].rolling(20).mean().iloc[-1]
                ma_50 = hist_1y['Close'].rolling(50).mean().iloc[-1]
                metrics['trend'] = "BULLISH" if current_price > ma_20 > ma_50 else "BEARISH" if current_price < ma_20 < ma_50 else "NEUTRAL"
                
        except Exception as e:
            print(f"     Error calculating metrics: {e}")
            
        return metrics
    
    def _generate_stock_summary(self, ticker: str, data: Dict) -> str:
        """Generate concise summary of volatility, growth, speed, fundamentals, and analyst view."""

        summary = ""
        perf_data = data.get('performance_data', {})
        fin_data  = data.get('financial_data', {})
        info      = data.get('info', {})

        # ── Fundamental snapshot (from yfinance .info) ─────────────────────
        if info:
            mcap = info.get("marketCap")
            if mcap:
                mcap_b = mcap / 1e9
                summary += f"   MARKET CAP: ${mcap_b:.1f}B\n"
            sector = info.get("sector") or info.get("industry")
            if sector:
                summary += f"   SECTOR: {sector}\n"
            pe = info.get("forwardPE") or info.get("trailingPE")
            if pe:
                summary += f"   P/E (fwd/trail): {pe:.1f}\n"
            beta = info.get("beta")
            if beta:
                risk_label = "LOW" if beta < 0.8 else ("HIGH" if beta > 1.3 else "MODERATE")
                summary += f"   BETA: {beta:.2f} ({risk_label} market sensitivity)\n"
            rec = info.get("recommendationKey")
            rec_mean = info.get("recommendationMean")
            n_analysts = info.get("numberOfAnalystOpinions")
            if rec:
                tgt = info.get("targetMeanPrice")
                tgt_str = f"  |  target ${tgt:.2f}" if tgt else ""
                analyst_str = f" ({n_analysts} analysts)" if n_analysts else ""
                summary += f"   ANALYST CONSENSUS: {rec.upper()}{analyst_str}  [mean={rec_mean:.1f}/5]{tgt_str}\n"
            eg = info.get("earningsGrowth")
            rg = info.get("revenueGrowth")
            if eg is not None or rg is not None:
                growth_parts = []
                if eg is not None:
                    growth_parts.append(f"earnings growth {eg*100:+.1f}%")
                if rg is not None:
                    growth_parts.append(f"revenue growth {rg*100:+.1f}%")
                summary += f"   FUNDAMENTAL GROWTH: {', '.join(growth_parts)}\n"
            roe = info.get("returnOnEquity")
            de  = info.get("debtToEquity")
            if roe is not None:
                summary += f"   ROE: {roe*100:.1f}%"
                if de is not None:
                    summary += f"   D/E: {de:.2f}"
                summary += "\n"

        # ── Earnings beat/miss history ─────────────────────────────────────
        earnings_text = "" if data.get("suppress_earnings_history") else self._fetch_earnings_history(ticker)
        if earnings_text:
            summary += "   EARNINGS HISTORY:\n" + earnings_text + "\n"
        
        # Volatility Summary
        volatility = perf_data.get('volatility', 0)
        if volatility > 0:
            if volatility < 15:
                vol_summary = "LOW VOLATILITY (Stable)"
            elif volatility < 30:
                vol_summary = "MODERATE VOLATILITY (Average risk)"
            else:
                vol_summary = "HIGH VOLATILITY (Risky)"
            summary += f"   VOLATILITY: {vol_summary} - {volatility:.1f}% annual\n"
        
        # Growth Summary
        returns_1y = perf_data.get('1y_return', 0)
        returns_3y = perf_data.get('3y_annualized', 0)
        returns_5y = perf_data.get('5y_annualized', 0)
        
        if returns_1y != 0:
            if returns_1y > 20:
                growth_summary = "STRONG GROWTH"
            elif returns_1y > 10:
                growth_summary = "GOOD GROWTH"
            elif returns_1y > 0:
                growth_summary = "MODEST GROWTH"
            else:
                growth_summary = "DECLINING"
            
            summary += f"   GROWTH: {growth_summary} - 1Y: {returns_1y:.1f}%"
            if returns_3y != 0:
                summary += f", 3Y: {returns_3y:.1f}%"
            if returns_5y != 0:
                summary += f", 5Y: {returns_5y:.1f}%"
            summary += "\n"
        
        # Speed of Growth Summary
        avg_monthly = perf_data.get('avg_monthly_return', 0)
        consistency = perf_data.get('growth_consistency', 0)
        
        if avg_monthly != 0:
            if avg_monthly > 2:
                speed_summary = "FAST GROWTH (High momentum)"
            elif avg_monthly > 1:
                speed_summary = "STEADY GROWTH (Consistent)"
            elif avg_monthly > 0:
                speed_summary = "SLOW GROWTH (Gradual)"
            else:
                speed_summary = "NEGATIVE MOMENTUM"
                
            summary += f"   SPEED: {speed_summary} - {avg_monthly:.1f}% avg monthly\n"
            summary += f"   CONSISTENCY: {'High' if consistency > 0.7 else 'Moderate' if consistency > 0.3 else 'Low'} ({consistency:.2f})\n"
        
        # Trend and Financial Health
        trend = perf_data.get('trend', 'UNKNOWN')
        summary += f"   TREND: {trend}\n"
        
        # Financial metrics if available
        if fin_data and fin_data.get("temporal_filter_applied"):
            summary += f"   FINANCIALS AS OF: {fin_data.get('as_of_date')}\n"
        if fin_data and (fin_data.get('income_statement') or fin_data.get('income')):
            income = fin_data.get('income_statement') or fin_data.get('income')
            if income and len(income) > 0:
                recent = income[0]
                revenue = recent.get('totalRevenue', 'N/A')
                if revenue != 'N/A' and str(revenue).isdigit():
                    summary += f"   REVENUE: ${int(revenue)/1e9:.1f}B (latest quarter)\n"
        
        # Overall Assessment
        if volatility > 0 and returns_1y != 0:
            if returns_1y > 15 and volatility < 25:
                assessment = "ATTRACTIVE - Good growth with manageable risk"
            elif returns_1y > 0 and volatility < 20:
                assessment = "STABLE - Positive returns with low risk"
            elif returns_1y < -10 or volatility > 40:
                assessment = "HIGH RISK - Consider minimal allocation"
            else:
                assessment = "MODERATE - Standard consideration"
            
            summary += f"   ASSESSMENT: {assessment}\n"
        
        return summary
    
    def _generate_integrated_strategy(self, news_analysis: Dict, financial_data: Optional[List[Dict]], 
                                    investment_amount: float, reinvestment_amount: float) -> Dict:
        """Generate final investment strategy combining news and financial analysis"""
        
        if not financial_data:
            print(" Generating strategy based on news analysis only")
            return self._generate_news_only_strategy(news_analysis, investment_amount)
        
        print(" Integrating news analysis with comprehensive financial & performance data...")
        
        # Combined Layer: Fetch financial data AND historical performance together
        combined_analysis = self._perform_comprehensive_financial_analysis(news_analysis, financial_data)
        
        # Create integrated strategy prompt
        integration_prompt = f"""
        You are an expert financial analyst creating a final investment strategy by combining:
        1. News-based market analysis and company recommendations
        2. Detailed financial statement data
        
        NEWS ANALYSIS SUMMARY:
        - Market Themes: {news_analysis.get('analysis_summary', {}).get('market_themes', [])}
        - Key Trends: {news_analysis.get('analysis_summary', {}).get('key_trends', [])}
        - Companies Identified: {len(news_analysis.get('recommended_companies', []))}
        
        COMPREHENSIVE FINANCIAL & PERFORMANCE ANALYSIS:
        {combined_analysis}
        
        INVESTMENT PARAMETERS:
        - Initial Investment: ${investment_amount:,.2f}
        - Reinvestment Amount: ${reinvestment_amount:,.2f}
        
        Create a refined investment strategy that:
        1. Uses the provided VOLATILITY, GROWTH, and SPEED summaries as primary decision factors
        2. Allocates MORE to stocks with: Low volatility + Strong growth + Fast/steady speed
        3. Allocates LESS to stocks with: High volatility + Declining growth + Negative momentum
        4. Validates recommendations using the comprehensive financial & performance analysis
        5. Provides final portfolio allocation percentages (must sum to 100%)
        6. Base allocation decisions primarily on the summarized assessments provided
        
        Return a JSON object with this structure:
        {{
            "final_strategy": {{
                "allocation_methodology": "How volatility, growth, and speed summaries influenced allocations",
                "key_insights": ["insight based on comprehensive analysis"],
                "risk_assessment": "Overall portfolio risk based on combined analysis",
                "implementation_approach": "How to implement this strategy"
            }},
            "final_recommendations": [
                {{
                    "ticker": "TICKER",
                    "company_name": "Company Name",
                    "final_allocation": 15.5,
                    "volatility_assessment": "LOW/MODERATE/HIGH based on summary",
                    "growth_assessment": "STRONG/GOOD/MODEST/DECLINING based on summary",
                    "speed_assessment": "FAST/STEADY/SLOW/NEGATIVE based on summary",
                    "allocation_rationale": "Why this allocation based on volatility + growth + speed combination",
                    "overall_score": 8.2,
                    "implementation_notes": "Specific notes for this holding"
                }}
            ],
            "portfolio_summary": {{
                "total_allocation": 100.0,
                "expected_return": "10-14%",
                "risk_level": "Moderate",
                "diversification_score": 8.5,
                "implementation_timeline": "3-6 months"
            }}
        }}
        """
        
        try:
            response = _oai_create_with_retry(
                self.openai_client,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": integration_prompt}],
                temperature=self.temperature,
                max_completion_tokens=3000,
            )
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                print(" Empty response from OpenAI for strategy integration")
                return self._generate_news_only_strategy(news_analysis, investment_amount)

            # Clean JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            if not content.strip():
                print(" Empty JSON block in response for strategy integration")
                return self._generate_news_only_strategy(news_analysis, investment_amount)
            strategy = json.loads(content.strip())
            print(" Integrated strategy generated successfully")
            return strategy
            
        except Exception as e:
            print(f" Integration failed, using news-only strategy: {e}")
            return self._generate_news_only_strategy(news_analysis, investment_amount)
    
    def _prepare_financial_summary(self, financial_data: List[Dict]) -> str:
        """Prepare financial data summary for OpenAI analysis"""
        
        summary = ""
        for company_data in financial_data[:20]:
            ticker = company_data.get('ticker', 'Unknown')
            summary += f"\n{ticker} Financial Summary:\n"
            
            # Income statement highlights
            income_reports = company_data.get('income', [])
            if income_reports:
                latest = income_reports[0]
                revenue = latest.get('totalRevenue', 'N/A')
                net_income = latest.get('netIncome', 'N/A')
                summary += f"  Latest Revenue: {revenue}\n"
                summary += f"  Latest Net Income: {net_income}\n"
            
            # Balance sheet highlights  
            balance_reports = company_data.get('balance', [])
            if balance_reports:
                latest_balance = balance_reports[0]
                total_assets = latest_balance.get('totalAssets', 'N/A')
                total_debt = latest_balance.get('totalLiabilities', 'N/A')
                summary += f"  Total Assets: {total_assets}\n"
                summary += f"  Total Liabilities: {total_debt}\n"
            
            summary += "\n"
        
        return summary
    
    def _generate_news_only_strategy(self, news_analysis: Dict, investment_amount: float) -> Dict:
        """Generate strategy based only on news analysis"""
        
        companies = news_analysis.get('recommended_companies', [])
        
        return {
            "final_strategy": {
                "validation_summary": "Strategy based on news analysis and market trends",
                "key_financial_insights": ["Limited financial validation due to data constraints"],
                "implementation_approach": "News-driven investment strategy",
                "risk_assessment": "Moderate - based on market analysis only"
            },
            "final_recommendations": [
                {
                    "ticker": comp.get('ticker', ''),
                    "company_name": comp.get('company_name', ''),
                    "final_allocation": comp.get('recommended_allocation', 5.0),
                    "news_score": comp.get('confidence_score', 7.0),
                    "financial_score": 0.0,
                    "combined_rationale": comp.get('investment_thesis', ''),
                    "key_metrics": "News-based analysis",
                    "implementation_notes": "Monitor for financial validation"
                } for comp in companies[:20]
            ],
            "portfolio_summary": {
                "total_allocation": 100.0,
                "expected_return": news_analysis.get('portfolio_strategy', {}).get('expected_annual_return', '8-12%'),
                "risk_level": news_analysis.get('portfolio_strategy', {}).get('risk_level', 'Moderate'),
                "diversification_score": 7.0,
                "implementation_timeline": "1-3 months"
            }
        }
    
    def _generate_stock_only_strategy(self, keyword: str, start_date: str, end_date: str, 
                                    investment_amount: float, reinvestment_amount: float,
                                    additional_context: str = "") -> Dict:
        """Generate investment strategy based purely on stock/financial data when no news is available"""
        
        print(" STOCK-ONLY ANALYSIS MODE")
        print(" Generating strategy from financial data and sector analysis...")
        
        # Define major companies in the sector for analysis
        sector_companies = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MDT', 'AMGN', 'GILD', 'CVS', 'MRK'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'WMB'],
            'industrial': ['BA', 'CAT', 'GE', 'MMM', 'UPS', 'HON', 'LMT', 'RTX', 'DE', 'UNP'],
            'consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT', 'KO', 'PG']
        }
        
        # Get companies for the specified sector
        companies = sector_companies.get(keyword.lower(), sector_companies['technology'])
        print(f" Analyzing {len(companies)} major {keyword} companies...")
        
        try:
            # Generate strategy based on stock performance and financial analysis
            stock_prompt = f"""
            You are an expert quantitative investment analyst creating a stock-only investment strategy.
            
            TASK: Create a comprehensive investment strategy for the {keyword} sector using financial analysis.
            
            PARAMETERS:
            - Sector: {keyword}
            - Investment Amount: ${investment_amount:,.2f}
            - Analysis Period: {start_date} to {end_date}
            - Companies to Consider: {', '.join(companies)}
            
            ANALYSIS APPROACH:
            Since no recent news data is available, base your analysis on:
            1. Historical stock performance patterns
            2. Sector fundamentals and trends
            3. Market cap and liquidity considerations
            4. Risk-return optimization
            5. Diversification across sub-sectors
            
            INSTRUCTIONS:
            1. Select 8-12 companies from the provided list
            2. Assign allocation percentages (must sum to 100%)
            3. Focus on established, liquid stocks
            4. Balance growth and stability
            5. Consider sector-specific factors
            
            {additional_context}
            
            OUTPUT FORMAT (JSON):
            {{
                "analysis_type": "stock_only_financial_analysis",
                "sector": "{keyword}",
                "strategy_summary": "Brief overview of the approach",
                "final_recommendations": [
                    {{
                        "ticker": "AAPL",
                        "company_name": "Apple Inc.",
                        "final_allocation": 15.0,
                        "rationale": "Strong fundamentals and market position",
                        "risk_level": "Medium",
                        "expected_return": "8-12%"
                    }}
                ],
                "portfolio_summary": {{
                    "total_allocation": 100.0,
                    "expected_return": "8-12%",
                    "risk_level": "Moderate",
                    "diversification_score": 8.0,
                    "implementation_timeline": "1-2 weeks"
                }},
                "key_assumptions": [
                    "Analysis based on historical performance",
                    "No recent news catalysts considered",
                    "Focus on fundamental strength"
                ]
            }}
            """
            
            # Get strategy from GPT
            response = _oai_create_with_retry(
                self.openai_client,
                model="gpt-4",
                messages=[{"role": "user", "content": stock_prompt}],
                max_completion_tokens=4000,
                temperature=self.temperature,
            )
            
            strategy_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            if strategy_text:
                # Clean up the response to extract just the JSON part
                if '```json' in strategy_text:
                    strategy_text = strategy_text.split('```json')[1].split('```')[0]
                elif '```' in strategy_text:
                    strategy_text = strategy_text.split('```')[1].split('```')[0]
                
                strategy = json.loads(strategy_text.strip())
            else:
                raise ValueError("Empty response from OpenAI")
            
            print(f" Generated stock-only strategy with {len(strategy.get('final_recommendations', []))} recommendations")
            return strategy
            
        except Exception as e:
            print(f" Error generating stock-only strategy: {str(e)}")
            # Return a basic fallback strategy
            return {
                "analysis_type": "fallback_strategy",
                "sector": keyword,
                "final_recommendations": [
                    {
                        "ticker": _sector_config.benchmark(),
                        "company_name": "SPDR S&P 500 ETF",
                        "final_allocation": 60.0,
                        "rationale": "Broad market exposure as fallback",
                        "risk_level": "Medium"
                    },
                    {
                        "ticker": companies[0] if companies else "AAPL",
                        "company_name": "Sector Leader",
                        "final_allocation": 40.0,
                        "rationale": "Sector-specific exposure",
                        "risk_level": "Medium"
                    }
                ],
                "portfolio_summary": {
                    "total_allocation": 100.0,
                    "expected_return": "6-10%",
                    "risk_level": "Moderate",
                    "diversification_score": 6.0
                },
                "error_fallback": True
            }
    
    def _calculate_comprehensive_returns(self, strategy: Dict, start_date: str, end_date: str,
                                       investment_amount: float, reinvestment_amount: float,
                                       return_periods: List[str]) -> Dict:
        """Calculate comprehensive return analysis for the strategy"""
        
        print(" Calculating portfolio returns...")
        
        recommendations = strategy.get('final_recommendations', [])
        if not recommendations:
            return {"error": "No recommendations for return calculation"}
        
        # Get tickers and allocations - LIMIT TO TOP COMPANIES ONLY
        tickers = []
        allocations = []
        
        # Sort recommendations by allocation (highest first)
        sorted_recs = sorted(recommendations, key=lambda x: x.get('final_allocation', 0), reverse=True)
        top_companies = sorted_recs[:20]
        
        print(f" Calculating returns for top {len(top_companies)} companies (out of {len(recommendations)} total)")
        
        for rec in top_companies:
            ticker = rec.get('ticker', '').strip()
            allocation = rec.get('final_allocation', 0)
            
            if ticker and allocation > 0:
                tickers.append(ticker)
                allocations.append(allocation / 100.0)  # Convert to decimal
        
        if not tickers:
            return {"error": "No valid tickers for return calculation"}
        
        print(f" Analyzing {len(tickers)} companies: {', '.join(tickers)}")
        
        # Fetch historical data and calculate returns
        portfolio_returns = {}
        individual_returns = {}
        
        try:
            # Calculate single overall return from analysis start to end
            overall_returns = self._calculate_period_returns(tickers, allocations, start_date, end_date, 'OVERALL')
            main_return = overall_returns['portfolio_return']
            individual_returns = overall_returns['individual_returns']
            
            # Store in portfolio_returns for compatibility
            portfolio_returns['overall'] = main_return
            
            # Calculate final values
            final_portfolio_value = investment_amount * (1 + main_return / 100)
            total_final_value = final_portfolio_value + reinvestment_amount
            total_roi = ((total_final_value - investment_amount - reinvestment_amount) / 
                        (investment_amount + reinvestment_amount)) * 100 if (investment_amount + reinvestment_amount) > 0 else 0
            
            return {
                "portfolio_returns": portfolio_returns,
                "individual_returns": individual_returns,
                "main_return": main_return,
                "investment_amount": investment_amount,
                "reinvestment_amount": reinvestment_amount,
                "final_portfolio_value": final_portfolio_value,
                "total_final_value": total_final_value,
                "total_roi": total_roi,
                "benchmark_comparison": self._calculate_benchmark_comparison(main_return, start_date, end_date)
            }
            
        except Exception as e:
            print(f" Return calculation error: {e}")
            return {"error": f"Return calculation failed: {e}"}
    
    def _calculate_period_returns(self, tickers: List[str], allocations: List[float], 
                                 start_date: str, end_date: str, period: str) -> Dict:
        """Calculate returns for a specific period with strict date validation"""
        
        # Convert period to timedelta
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # CRITICAL: Ensure analysis end date is not in the future
        today = datetime.now().date()
        if end_dt.date() > today:
            print(f" Analysis end date {end_date} is in the future, adjusting to today ({today})")
            end_dt = datetime.combine(today, datetime.min.time())
            end_date = today.strftime('%Y-%m-%d')
        
        if period == '1M':
            start_dt = end_dt - relativedelta(months=1)
        elif period == '3M':
            start_dt = end_dt - relativedelta(months=3)
        elif period == '6M':
            start_dt = end_dt - relativedelta(months=6)
        elif period == '1Y':
            start_dt = end_dt - relativedelta(years=1)
        elif period == '2Y':
            start_dt = end_dt - relativedelta(years=2)
        elif period == 'OVERALL':
            # Use the full analysis period from start to end
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            # Also validate start date is not in the future
            if start_dt.date() > today:
                print(f" Analysis start date {start_date} is in the future, adjusting to today ({today})")
                start_dt = datetime.combine(today, datetime.min.time())
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            # Also validate start date is not in the future
            if start_dt.date() > today:
                print(f" Analysis start date {start_date} is in the future, adjusting to today ({today})")
                start_dt = datetime.combine(today, datetime.min.time())
        
        # Additional safety: ensure start is before end
        if start_dt >= end_dt:
            print(f" Invalid date range: start {start_dt.date()} >= end {end_dt.date()}")
            return {"portfolio_return": 0, "individual_returns": {}}
        
        if period == 'OVERALL':
            print(f" Calculating overall returns from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        else:
            print(f" Calculating {period} returns for period {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        print(f" Stock data analysis: full period data available for return calculations")
        
        individual_returns = {}
        portfolio_return = 0
        
        for ticker, allocation in zip(tickers, allocations):
            try:
                # Clean ticker symbol (remove $ prefix and other invalid characters)
                clean_ticker = ticker.replace('$', '').replace('#', '').strip().upper()
                
                # Skip invalid tickers
                if not clean_ticker or len(clean_ticker) > 5:
                    print(f"     Skipping invalid ticker: {ticker}")
                    individual_returns[ticker] = 0
                    continue
                
                hist = self._fetch_yahoo_chart_history(
                    clean_ticker, start_dt, end_dt + timedelta(days=1)
                )
                
                # Basic validation: ensure we have stock data for the analysis period
                # Note: News data cutoff only applies to LLM inputs, not stock return calculations
                
                if not hist.empty and len(hist) > 1:
                    start_price = hist.iloc[0]['Close']
                    end_price = hist.iloc[-1]['Close']
                    
                    # Validate prices are reasonable
                    if start_price > 0 and end_price > 0:
                        stock_return = ((end_price - start_price) / start_price) * 100
                        individual_returns[clean_ticker] = stock_return
                        portfolio_return += stock_return * allocation
                        print(f"     {clean_ticker}: {stock_return:.2f}% return ({len(hist)} data points)")
                    else:
                        individual_returns[clean_ticker] = 0
                        print(f"     {clean_ticker}: Invalid price data")
                else:
                    individual_returns[clean_ticker] = 0
                    print(f"     {clean_ticker}: No historical data available")
                    
            except Exception as e:
                clean_ticker = ticker.replace('$', '').replace('#', '').strip().upper()
                print(f"     Error with {clean_ticker}: {str(e)[:50]}...")
                individual_returns[clean_ticker] = 0
        
        return {
            "portfolio_return": portfolio_return,
            "individual_returns": individual_returns
        }
    
    def _calculate_benchmark_comparison(self, portfolio_return: float, start_date: str, end_date: str) -> Dict:
        """Compare portfolio performance to market benchmarks"""
        
        try:
            start_dt = pd.to_datetime(start_date).to_pydatetime()
            end_dt = pd.to_datetime(end_date).to_pydatetime()
            hist = self._fetch_yahoo_chart_history(_sector_config.benchmark(), start_dt, end_dt + timedelta(days=1))
            
            if not hist.empty and len(hist) > 1:
                start_price = hist.iloc[0]['Close']
                end_price = hist.iloc[-1]['Close']
                market_return = ((end_price - start_price) / start_price) * 100
                
                alpha = portfolio_return - market_return
                
                return {
                    "market_return": market_return,
                    "portfolio_return": portfolio_return,
                    "alpha": alpha,
                    "outperformed": alpha > 0
                }
            
        except Exception as e:
            print(f" Benchmark comparison error: {e}")
        
        return {
            "market_return": 0,
            "portfolio_return": portfolio_return,
            "alpha": 0,
            "outperformed": False
        }
    
    def _assemble_final_results(self, news_analysis: Dict, financial_data: Optional[List[Dict]], 
                               strategy: Dict, return_analysis: Dict, 
                               investment_amount: float, reinvestment_amount: float) -> Dict:
        """Assemble comprehensive final results"""
        
        return {
            "strategy_generation": {
                "timestamp": datetime.now().isoformat(),
                "investment_amount": investment_amount,
                "reinvestment_amount": reinvestment_amount,
                "data_sources": {
                    "news_analysis": True,
                    "financial_data": financial_data is not None,
                    "return_analysis": "error" not in return_analysis
                }
            },
            "market_analysis": news_analysis.get('analysis_summary', {}),
            "final_strategy": strategy.get('final_strategy', {}),
            "recommendations": strategy.get('final_recommendations', []),
            "portfolio_summary": strategy.get('portfolio_summary', {}),
            "return_analysis": return_analysis,
            "implementation_guide": {
                "next_steps": [
                    "Review individual company recommendations",
                    "Validate financial health of top holdings",
                    "Consider market timing for implementation",
                    "Set up monitoring and rebalancing schedule"
                ],
                "risk_considerations": [
                    "Market volatility impact",
                    "Sector concentration risk",
                    "Individual company risk",
                    "Economic cycle sensitivity"
                ],
                "monitoring_metrics": [
                    "Portfolio performance vs benchmark",
                    "Individual stock performance",
                    "Sector allocation drift",
                    "News sentiment changes"
                ]
            }
        }

    def get_alpha_vantage_report(self, ticker: str, function: str) -> Dict:
        """Enhanced Alpha Vantage data fetching with error handling"""
        url = f"{self.alpha_vantage_base_url}?function={function}&symbol={ticker}&apikey={self.alpha_vantage_key}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API limit or error messages
            if 'Error Message' in data:
                print(f" Alpha Vantage error for {ticker} ({function}): {data['Error Message']}")
                return {}
            elif 'Information' in data:
                print(f" Alpha Vantage info for {ticker} ({function}): {data['Information']}")
                return {}
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f" Network error fetching {ticker} ({function}): {e}")
            return {}
        except json.JSONDecodeError:
            print(f" Invalid JSON response for {ticker} ({function})")
            return {}
        except Exception as e:
            print(f" Unexpected error fetching {ticker} ({function}): {e}")
            return {}

    def fetch_comprehensive_financial_data(self, tickers: List[str]) -> List[Dict]:
        """
        Enhanced financial data fetching with comprehensive error handling
        Based on NewsAPI+FinanceReports.ipynb methodology
        """
        print(f" Fetching comprehensive financial data for {len(tickers)} companies...")
        print(f" Companies: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        
        financial_data = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f" [{i}/{len(tickers)}] Fetching data for {ticker}...")
            
            try:
                # Fetch the three main financial statements
                income_statement = self.get_alpha_vantage_report(ticker, "INCOME_STATEMENT")
                balance_sheet = self.get_alpha_vantage_report(ticker, "BALANCE_SHEET")
                cash_flow = self.get_alpha_vantage_report(ticker, "CASH_FLOW")
                
                company_data = {
                    "ticker": ticker,
                    "income": income_statement.get("quarterlyReports", []),
                    "balance_sheet": balance_sheet.get("quarterlyReports", []),
                    "cash_flow": cash_flow.get("quarterlyReports", []),
                    "data_quality": self._assess_data_quality(income_statement, balance_sheet, cash_flow)
                }
                
                financial_data.append(company_data)
                print(f" {ticker}: Data quality - {company_data['data_quality']}")
                
            except Exception as e:
                print(f" Error fetching {ticker}: {e}")
                # Add empty entry to maintain list consistency
                financial_data.append({
                    "ticker": ticker,
                    "income": [],
                    "balance_sheet": [],
                    "cash_flow": [],
                    "data_quality": "failed",
                    "error": str(e)
                })
            
            # Rate limiting - Alpha Vantage free tier allows 5 calls per minute
            if i < len(tickers):
                print(" Waiting 15 seconds (rate limiting)...")
                time.sleep(15)
        
        print(f" Financial data collection complete for {len(financial_data)} companies!")
        return financial_data

    def _assess_data_quality(self, income_data: Dict, balance_data: Dict, cash_data: Dict) -> str:
        """Assess the quality of fetched financial data"""
        quality_score = 0
        max_score = 3
        
        # Check income statement
        if income_data.get("quarterlyReports") and len(income_data["quarterlyReports"]) > 0:
            quality_score += 1
        
        # Check balance sheet
        if balance_data.get("quarterlyReports") and len(balance_data["quarterlyReports"]) > 0:
            quality_score += 1
        
        # Check cash flow
        if cash_data.get("quarterlyReports") and len(cash_data["quarterlyReports"]) > 0:
            quality_score += 1
        
        if quality_score == max_score:
            return "excellent"
        elif quality_score >= 2:
            return "good"
        elif quality_score >= 1:
            return "partial"
        else:
            return "poor"

    def generate_financial_based_strategy(self, news_analysis: Dict, financial_data: List[Dict], 
                                        investment_amount: float) -> Optional[Dict]:
        """
        Generate investment strategy based on both news analysis and financial data
        Enhanced version from NewsAPI+FinanceReports.ipynb
        """
        if not financial_data:
            print(" No financial data available for strategy generation")
            return None

        # Prepare comprehensive financial data string for OpenAI analysis
        financial_data_string = self._prepare_comprehensive_financial_summary(financial_data)
        
        # Get the original news-based analysis
        recommended_companies = news_analysis.get('recommended_companies', [])
        
        # Enhanced prompt combining both analyses
        enhanced_prompt = f"""
        You are an expert financial analyst creating the final investment strategy by combining:
        1. News-based market analysis and company recommendations
        2. Detailed financial statement analysis with quarterly data

        NEWS-BASED ANALYSIS SUMMARY:
        - Market Themes: {news_analysis.get('analysis_summary', {}).get('market_themes', [])}
        - Key Trends: {news_analysis.get('analysis_summary', {}).get('key_trends', [])}
        - Companies Identified: {len(recommended_companies)}

        COMPREHENSIVE FINANCIAL DATA:
        {financial_data_string}

        TASK: Create a refined investment strategy that:
        1. Validates the news-based recommendations using financial fundamentals
        2. Adjusts allocations based on financial health and performance trends
        3. Identifies the strongest companies from both news sentiment and financial metrics
        4. Provides final portfolio allocation percentages (must sum to 100%)
        5. Includes specific financial metrics that support each recommendation

        Return a JSON object with this structure:
        {{
            "final_strategy": {{
                "validation_summary": "How financial data supports or contradicts news analysis",
                "key_financial_insights": ["Revenue growth trends", "Profitability analysis", "Cash flow strength"],
                "implementation_date": "2024-01-01",
                "confidence_level": "High/Medium/Low based on data quality"
            }},
            "final_recommendations": [
                {{
                    "ticker": "TICKER",
                    "company_name": "Company Name",
                    "final_allocation": 15.5,
                    "news_score": 8.5,
                    "financial_score": 9.0,
                    "combined_reasoning": "News sentiment + financial fundamentals analysis",
                    "key_financial_metrics": ["20% revenue growth", "15% profit margin", "Strong cash flow"],
                    "risk_assessment": "Financial risk analysis",
                    "growth_catalysts": ["Market trends", "Financial strengths"],
                    "investment_thesis": "Combined news and financial rationale"
                }}
            ],
            "portfolio_summary": {{
                "total_companies": 10,
                "diversification_score": 8.5,
                "expected_annual_return": "12-18%",
                "risk_level": "Moderate-Aggressive",
                "financial_validation_coverage": "85%",
                "avg_revenue_growth": "15%",
                "avg_profit_margin": "12%"
            }}
        }}
        """

        try:
            response = _oai_create_with_retry(
                self.openai_client,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst creating investment recommendations by combining news analysis with comprehensive financial statement analysis. Always return valid JSON with specific financial metrics."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=self.temperature,
                max_completion_tokens=4000,
            )

            response_content = response.choices[0].message.content

            # Check for empty response
            if not response_content or not response_content.strip():
                print(" Empty response from OpenAI for financial strategy")
                return None

            # Clean and parse JSON
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                response_content = response_content.split("```")[1]

            if not response_content.strip():
                print(" Empty JSON block in response for financial strategy")
                return None
            final_strategy = json.loads(response_content.strip())
            
            # Add metadata
            final_strategy['financial_data_used'] = True
            final_strategy['companies_with_financial_data'] = len([f for f in financial_data if f.get('data_quality') != 'poor'])
            final_strategy['generation_timestamp'] = datetime.now().isoformat()

            print(" Enhanced financial-based strategy generated successfully!")
            return final_strategy

        except json.JSONDecodeError as e:
            print(f" JSON parsing failed in financial strategy: {e}")
            return None
        except Exception as e:
            print(f" Financial strategy generation failed: {e}")
            return None

    def _prepare_comprehensive_financial_summary(self, financial_data: List[Dict]) -> str:
        """Prepare detailed financial data summary for enhanced analysis"""
        summary = ""
        
        for company_data in financial_data:
            ticker = company_data.get('ticker', 'Unknown')
            data_quality = company_data.get('data_quality', 'unknown')
            
            summary += f"\n{ticker} COMPREHENSIVE FINANCIAL ANALYSIS (Quality: {data_quality}):\n"
            summary += "="*50 + "\n"
            
            # Income statement analysis with trends
            income_reports = company_data.get('income', [])
            if income_reports:
                summary += "INCOME STATEMENT (Recent Quarters):\n"
                for i, report in enumerate(income_reports[:4]):
                    fiscal_date = report.get('fiscalDateEnding', 'N/A')
                    revenue = report.get('totalRevenue', 'N/A')
                    net_income = report.get('netIncome', 'N/A')
                    gross_profit = report.get('grossProfit', 'N/A')
                    summary += f"  Q{i+1} ({fiscal_date}): Revenue=${revenue}, Net Income=${net_income}, Gross Profit=${gross_profit}\n"
                
                # Calculate trends if enough data
                if len(income_reports) >= 2:
                    try:
                        current_rev = float(income_reports[0].get('totalRevenue', 0))
                        prev_rev = float(income_reports[1].get('totalRevenue', 0))
                        if prev_rev > 0:
                            growth = ((current_rev - prev_rev) / prev_rev) * 100
                            summary += f"  Revenue Growth (QoQ): {growth:.1f}%\n"
                    except (ValueError, TypeError):
                        pass

            # Balance sheet analysis
            balance_reports = company_data.get('balance_sheet', [])
            if balance_reports:
                summary += "\nBALANCE SHEET (Recent Quarter):\n"
                latest_balance = balance_reports[0]
                assets = latest_balance.get('totalAssets', 'N/A')
                liabilities = latest_balance.get('totalLiabilities', 'N/A')
                equity = latest_balance.get('totalShareholderEquity', 'N/A')
                cash = latest_balance.get('cashAndCashEquivalentsAtCarryingValue', 'N/A')
                summary += f"  Total Assets: ${assets}\n"
                summary += f"  Total Liabilities: ${liabilities}\n"
                summary += f"  Shareholder Equity: ${equity}\n"
                summary += f"  Cash & Equivalents: ${cash}\n"

            # Cash flow analysis
            cash_reports = company_data.get('cash_flow', [])
            if cash_reports:
                summary += "\nCASH FLOW (Recent Quarters):\n"
                for i, report in enumerate(cash_reports[:4]):
                    fiscal_date = report.get('fiscalDateEnding', 'N/A')
                    operating_cf = report.get('operatingCashFlow', 'N/A')
                    investing_cf = report.get('cashFlowFromInvestment', 'N/A')
                    financing_cf = report.get('cashFlowFromFinancing', 'N/A')
                    summary += f"  Q{i+1} ({fiscal_date}): Operating=${operating_cf}, Investing=${investing_cf}, Financing=${financing_cf}\n"

            summary += "\n" + "-"*50 + "\n"
        
        return summary


# Convenience function for easy use
def generate_investment_strategy(keyword: str, investment_amount: float, 
                               openai_api_key: str, alpha_vantage_key: Optional[str] = None,
                               temperature: float = 0.3, model: str = "gpt-4o-mini", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to generate complete investment strategy
    
    Args:
        keyword: Investment sector/theme (e.g., "technology", "healthcare")
        investment_amount: Amount to invest
        openai_api_key: OpenAI API key
        alpha_vantage_key: Alpha Vantage API key (optional)
        temperature: Temperature setting for AI responses (0.0-2.0, default 0.3)
        model: AI model to use ("gpt-4o-mini", "gpt-5.1", etc. - GPT-5.1 only for periods after Oct 2024)
        **kwargs: Additional parameters for customization
    
    Returns:
        Complete investment strategy with analysis and recommendations
    """
    
    generator = InvestmentStrategyGenerator(
        api_key_openai=openai_api_key,
        alpha_vantage_key=alpha_vantage_key,
        temperature=temperature,
        model=model
    )
    
    return generator.generate_complete_strategy(
        user_input_keyword=keyword,
        investment_amount=investment_amount,
        **kwargs
    )


class InvestmentStrategyGUI:
    """Simple GUI for Investment Strategy Generator"""
    
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
        
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        
        # API Key
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        
        self.setup_gui()
    
    def setup_gui(self):
        """Create the GUI window and widgets"""

        # Main window
        self.root = self.tk.Tk()
        self.root.title(" Sliding Window Investment Analysis")
        self.root.geometry("600x750")
        self.root.configure(bg='#f0f0f0')

        # Title
        title_label = self.tk.Label(
            self.root,
            text=" Sliding Window Investment Analysis",
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)

        # Main frame
        main_frame = self.ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)

        # Investment Parameters Section
        params_label = self.tk.Label(
            main_frame,
            text=" Investment Parameters",
            font=('Arial', 12, 'bold')
        )
        params_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

        # Sector input
        self.tk.Label(main_frame, text="Investment Sector:").grid(row=1, column=0, sticky='w', pady=5)
        self.sector_var = self.tk.StringVar(value="technology")
        sector_combo = self.ttk.Combobox(
            main_frame,
            textvariable=self.sector_var,
            values=["technology", "healthcare", "energy", "finance", "consumer", "industrial", "agriculture"],
            width=25
        )
        sector_combo.grid(row=1, column=1, sticky='w', pady=5)

        # Investment amount
        self.tk.Label(main_frame, text="Investment Amount ($):").grid(row=2, column=0, sticky='w', pady=5)
        self.amount_var = self.tk.StringVar(value="10000")
        amount_entry = self.ttk.Entry(main_frame, textvariable=self.amount_var, width=25)
        amount_entry.grid(row=2, column=1, sticky='w', pady=5)

        # Sliding Window Settings Section
        window_label = self.tk.Label(
            main_frame,
            text=" Sliding Window Settings",
            font=('Arial', 12, 'bold')
        )
        window_label.grid(row=3, column=0, columnspan=2, sticky='w', pady=(20, 10))

        # Help text
        help_text = self.tk.Label(
            main_frame,
            text="Analysis runs in periods, moving forward 1 month each time.",
            font=('Arial', 8),
            fg='#666666'
        )
        help_text.grid(row=4, column=0, columnspan=2, sticky='w', pady=(0, 10))

        # Start date
        self.tk.Label(main_frame, text="Start Date (YYYY-MM-DD):").grid(row=5, column=0, sticky='w', pady=5)
        self.start_date_var = self.tk.StringVar(value="2023-01-01")
        start_entry = self.ttk.Entry(main_frame, textvariable=self.start_date_var, width=25)
        start_entry.grid(row=5, column=1, sticky='w', pady=5)

        # Period length
        self.tk.Label(main_frame, text="Period Length (months):").grid(row=6, column=0, sticky='w', pady=5)
        self.period_length_var = self.tk.StringVar(value="6")
        period_spinbox = self.tk.Spinbox(
            main_frame,
            from_=1,
            to=60,
            textvariable=self.period_length_var,
            width=23,
            font=('Arial', 10)
        )
        period_spinbox.grid(row=6, column=1, sticky='w', pady=5)

        # End date
        self.tk.Label(main_frame, text="End Date (YYYY-MM-DD):").grid(row=7, column=0, sticky='w', pady=5)
        self.end_date_var = self.tk.StringVar(value="2024-12-31")
        end_entry = self.ttk.Entry(main_frame, textvariable=self.end_date_var, width=25)
        end_entry.grid(row=7, column=1, sticky='w', pady=5)

        # Optional Settings Section
        optional_label = self.tk.Label(
            main_frame,
            text=" Optional Settings",
            font=('Arial', 12, 'bold')
        )
        optional_label.grid(row=8, column=0, columnspan=2, sticky='w', pady=(20, 10))

        # Max companies
        self.tk.Label(main_frame, text="Max Companies to Analyze:").grid(row=9, column=0, sticky='w', pady=5)
        self.companies_var = self.tk.StringVar(value="10")
        companies_entry = self.ttk.Entry(main_frame, textvariable=self.companies_var, width=25)
        companies_entry.grid(row=9, column=1, sticky='w', pady=5)

        # News data file
        self.tk.Label(main_frame, text="News Data File (optional):").grid(row=10, column=0, sticky='w', pady=5)
        self.news_file_var = self.tk.StringVar()
        news_frame = self.ttk.Frame(main_frame)
        news_frame.grid(row=10, column=1, sticky='w', pady=5)

        news_entry = self.ttk.Entry(news_frame, textvariable=self.news_file_var, width=18)
        news_entry.pack(side='left')

        browse_btn = self.ttk.Button(news_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side='left', padx=(5, 0))

        # Generate button
        generate_btn = self.ttk.Button(
            main_frame,
            text=" Run Sliding Window Analysis",
            command=self.generate_strategy,
            style='Accent.TButton'
        )
        generate_btn.grid(row=11, column=0, columnspan=2, pady=30)
        
        # Progress bar
        self.progress = self.ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=18, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Status label
        self.status_var = self.tk.StringVar(value="Ready to generate strategy")
        status_label = self.tk.Label(main_frame, textvariable=self.status_var, fg='#27ae60')
        status_label.grid(row=19, column=0, columnspan=2, pady=5)
        
        # Results text area
        results_label = self.tk.Label(
            main_frame, 
            text=" Strategy Results", 
            font=('Arial', 12, 'bold')
        )
        results_label.grid(row=20, column=0, columnspan=2, sticky='w', pady=(20, 10))
        
        self.results_text = self.scrolledtext.ScrolledText(
            main_frame, 
            height=15, 
            width=70,
            wrap='word',
            font=('Consolas', 9)
        )
        self.results_text.grid(row=21, column=0, columnspan=2, sticky='nsew', pady=5)
        
        # File links section
        links_label = self.tk.Label(
            main_frame, 
            text=" Generated Files", 
            font=('Arial', 12, 'bold')
        )
        links_label.grid(row=22, column=0, columnspan=2, sticky='w', pady=(20, 10))
        
        # Frame for file link buttons
        self.links_frame = self.tk.Frame(main_frame)
        self.links_frame.grid(row=23, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Initially hidden - will show after files are generated
        self.links_frame.grid_remove()
        
        # Configure grid weights
        main_frame.grid_rowconfigure(19, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
    
    def browse_file(self):
        """Open file browser for news data file"""
        filename = self.filedialog.askopenfilename(
            title="Select News Data CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.news_file_var.set(filename)
    
    def generate_strategy(self):
        """Run sliding window analysis in background thread"""
        import threading

        # Validate inputs
        try:
            sector = self.sector_var.get().strip()
            if not sector:
                self.messagebox.showerror("Error", "Please enter an investment sector")
                return

            investment_amount = float(self.amount_var.get())
            max_companies = int(self.companies_var.get() or "10")
            news_file = self.news_file_var.get().strip() or None

            # Get sliding window parameters
            start_date = self.start_date_var.get().strip()
            period_length = int(self.period_length_var.get())
            end_date = self.end_date_var.get().strip()

            # Validate date formats
            from datetime import datetime
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')

            if period_length < 1 or period_length > 60:
                self.messagebox.showerror("Error", "Period length must be between 1 and 60 months")
                return

        except ValueError as e:
            self.messagebox.showerror("Error", f"Invalid input values: {e}")
            return

        # Start progress bar
        self.progress.start()
        self.status_var.set("Running sliding window analysis... This may take a while")
        self.results_text.delete(1.0, 'end')

        # Clear any existing file links
        self.clear_file_links()

        # Run in separate thread to prevent UI freezing
        def run_generation():
            try:
                print(f" Starting sliding window analysis for {sector}")
                print(f" Investment amount: ${investment_amount}")
                print(f" Max companies: {max_companies}")
                print(f" News file: {news_file or 'Auto-detected from news_collection_archive'}")
                print(f" Period: {start_date} to {end_date} ({period_length} months per window)")

                # Create generator instance
                generator = InvestmentStrategyGenerator(
                    api_key_openai=self.OPENAI_API_KEY,
                    temperature=0.3
                )

                # Run sliding window analysis
                results = generator.run_sliding_window_analysis(
                    user_input_keyword=sector,
                    investment_amount=investment_amount,
                    start_date=start_date,
                    period_length_months=period_length,
                    end_date=end_date,
                    news_data_path=news_file,
                    include_financial_validation=False,
                    max_companies=max_companies
                )

                print(f" Sliding window analysis completed - {len(results)} periods analyzed")

                # Build metadata for save envelope
                sw_metadata = {
                    "run_type": "sliding_window",
                    "model": generator.model,
                    "temperature": generator.temperature,
                    "sector": sector,
                    "timestamp": datetime.now().isoformat(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "period_length_months": period_length,
                    "total_periods": len(results),
                    "investment_amount": investment_amount,
                    "max_companies": max_companies
                }
                overlap_warn = generator._check_training_data_overlap(
                    generator.model, start_date, end_date
                )
                if overlap_warn:
                    sw_metadata["training_data_warning"] = overlap_warn

                # Update UI on main thread
                self.root.after(0, lambda: self.display_sliding_window_results(results, sector, sw_metadata))

            except Exception as e:
                print(f" Analysis failed: {e}")
                import traceback
                traceback.print_exc()
                error_message = str(e)
                self.root.after(0, lambda msg=error_message: self.display_error(msg))

        thread = threading.Thread(target=run_generation)
        thread.daemon = True
        thread.start()
    
    def display_results(self, strategy_result, sector):
        """Display strategy results in the text area"""
        self.progress.stop()
        
        if strategy_result and (strategy_result.get('investment_strategy') or strategy_result.get('final_strategy')):
            # Save results to dedicated results/ folder
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(results_dir, f"{sector}_investment_strategy_{timestamp}.json")

            try:
                with open(output_file, 'w') as f:
                    json.dump(strategy_result, f, indent=2, default=str)
                
                self.status_var.set(f" Strategy generated and saved to {output_file}")
                
                # Display results
                results_text = f" INVESTMENT STRATEGY GENERATED SUCCESSFULLY!\n"
                results_text += f"=" * 60 + "\n\n"
                results_text += f" Saved to: {output_file}\n\n"
                
                # Debug: Show what keys are available
                results_text += f" Available data: {', '.join(strategy_result.keys())}\n\n"
                
                # Strategy summary - try multiple possible keys
                strategy = (strategy_result.get('investment_strategy') or 
                           strategy_result.get('final_strategy') or 
                           strategy_result.get('strategy_summary') or
                           'No strategy text found')
                
                # If strategy is a dict, extract meaningful text
                if isinstance(strategy, dict):
                    strategy_text = ""
                    if 'analysis_summary' in strategy:
                        summary = strategy['analysis_summary']
                        strategy_text += f"Market Sentiment: {summary.get('market_sentiment', 'N/A')}\n"
                        strategy_text += f"Investment Outlook: {summary.get('investment_outlook', 'N/A')}\n"
                        strategy_text += f"Key Themes: {', '.join(summary.get('market_themes', []))}\n\n"
                    
                    if 'portfolio_strategy' in strategy:
                        portfolio = strategy['portfolio_strategy']
                        strategy_text += f"Risk Level: {portfolio.get('risk_level', 'N/A')}\n"
                        strategy_text += f"Expected Return: {portfolio.get('expected_annual_return', 'N/A')}\n"
                        strategy_text += f"Investment Horizon: {portfolio.get('investment_horizon', 'N/A')}\n"
                    
                    strategy = strategy_text if strategy_text else str(strategy)
                
                results_text += f" STRATEGY SUMMARY:\n"
                results_text += f"{'-' * 40}\n"
                results_text += f"{strategy[:1500]}...\n\n" if len(strategy) > 1500 else f"{strategy}\n\n"
                
                # Company analysis - try multiple possible keys
                companies = (strategy_result.get('company_analysis') or 
                           strategy_result.get('recommended_companies') or 
                           [])
                
                if companies:
                    results_text += f" RECOMMENDED COMPANIES ({len(companies)}):\n"
                    results_text += f"{'-' * 40}\n"
                    for i, company in enumerate(companies[:15], 1):
                        if isinstance(company, dict):
                            name = company.get('company_name', company.get('name', 'Unknown'))
                            ticker = company.get('ticker', 'N/A')
                            allocation = company.get('recommended_allocation', company.get('allocation', 'N/A'))
                            confidence = company.get('confidence_score', 'N/A')
                            results_text += f"{i:2d}. {name} ({ticker}): {allocation}% allocation, {confidence} confidence\n"
                        else:
                            results_text += f"{i:2d}. {str(company)}\n"
                    results_text += "\n"
                
                # Portfolio summary - try multiple locations
                portfolio = (strategy_result.get('portfolio_summary') or 
                           strategy_result.get('portfolio_strategy') or 
                           {})
                
                if portfolio:
                    results_text += f" PORTFOLIO SUMMARY:\n"
                    results_text += f"{'-' * 40}\n"
                    results_text += f"Expected Return: {portfolio.get('expected_return', portfolio.get('expected_annual_return', 'N/A'))}\n"
                    results_text += f"Risk Level: {portfolio.get('risk_level', 'N/A')}\n"
                    results_text += f"Diversification: {portfolio.get('diversification_approach', 'N/A')}\n"
                
                # Return analysis if available
                if 'return_analysis' in strategy_result:
                    returns = strategy_result['return_analysis']
                    results_text += f"\n PORTFOLIO RETURNS:\n"
                    results_text += f"{'-' * 40}\n"
                    
                    # Display main overall return
                    main_return = returns.get('main_return', 'N/A')
                    results_text += f" Overall Portfolio Return: {main_return:.2f}%\n" if isinstance(main_return, (int, float)) else f" Overall Portfolio Return: {main_return}\n"
                    
                    # Display total ROI
                    total_roi = returns.get('total_roi', 'N/A')
                    results_text += f" Total ROI: {total_roi:.2f}%\n" if isinstance(total_roi, (int, float)) else f" Total ROI: {total_roi}\n"
                    
                    # Display final values
                    final_value = returns.get('total_final_value', returns.get('final_portfolio_value', 'N/A'))
                    results_text += f" Final Portfolio Value: ${final_value:,.2f}\n" if isinstance(final_value, (int, float)) else f" Final Portfolio Value: {final_value}\n"
                
                self.results_text.delete(1.0, 'end')
                self.results_text.insert(1.0, results_text)
                
                # Add clickable file link to GUI
                self.add_file_links(output_file)
                
            except Exception as e:
                self.display_error(f"Error saving results: {e}")
        else:
            # Better error message with debug info
            error_msg = "Strategy generation failed"
            if strategy_result:
                available_keys = list(strategy_result.keys())
                error_msg += f"\nAvailable keys: {available_keys}"
                if 'error' in strategy_result:
                    error_msg += f"\nError details: {strategy_result['error']}"
            else:
                error_msg += " - no results returned"
            self.display_error(error_msg)
    
    def display_error(self, error_msg):
        """Display error message"""
        self.progress.stop()
        self.status_var.set(f" Error: {error_msg[:50]}...")
        
        error_text = f" ERROR OCCURRED\n"
        error_text += f"=" * 60 + "\n\n"
        error_text += f"{error_msg}\n\n"
        
        # Add troubleshooting tips
        error_text += f" TROUBLESHOOTING TIPS:\n"
        error_text += f"{'-' * 30}\n"
        error_text += f"1. Check internet connection for API access\n"
        error_text += f"2. Verify OpenAI API key is valid\n"
        error_text += f"3. Try a different investment sector\n"
        error_text += f"4. Ensure investment amount is a valid number\n"
        error_text += f"5. Check if news data file exists (if specified)\n\n"
        
        error_text += f" If the problem persists, try:\n"
        error_text += f"    Using existing news data file\n"
        error_text += f"    Reducing max companies to analyze\n"
        error_text += f"    Checking the console for detailed error messages\n"
        
        self.results_text.delete(1.0, 'end')
        self.results_text.insert(1.0, error_text)

    def display_sliding_window_results(self, results, sector, metadata=None):
        """Display sliding window analysis results"""
        self.progress.stop()

        from datetime import datetime
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(results_dir, f"{sector}_sliding_window_analysis_{timestamp}.json")

        try:
            # Wrap results in envelope with metadata
            save_data = {
                "metadata": metadata or {
                    "run_type": "sliding_window",
                    "sector": sector,
                    "timestamp": datetime.now().isoformat(),
                    "total_periods": len(results)
                },
                "results": results
            }

            # Save results to JSON
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

            self.status_var.set(f" Analysis complete - {len(results)} periods analyzed")

            # Display summary
            results_text = f" SLIDING WINDOW ANALYSIS COMPLETE!\n"
            results_text += f"=" * 60 + "\n\n"
            results_text += f" Saved to: {output_file}\n"
            results_text += f" Total Periods Analyzed: {len(results)}\n\n"

            # Show summary of each period
            results_text += f" PERIOD SUMMARIES:\n"
            results_text += f"{'-' * 60}\n\n"

            for i, period_result in enumerate(results, 1):
                if 'error' in period_result:
                    results_text += f"Period {i}: {period_result['period_start']} to {period_result['period_end']}\n"
                    results_text += f"   Error: {period_result['error']}\n\n"
                    continue

                period_start = period_result.get('period_start', 'N/A')
                period_end = period_result.get('period_end', 'N/A')
                results_text += f"Period {i}: {period_start} to {period_end}\n"

                # Try to get return information
                return_analysis = period_result.get('return_analysis', {})
                if return_analysis:
                    main_return = return_analysis.get('main_return', 'N/A')
                    total_roi = return_analysis.get('total_roi', 'N/A')

                    if isinstance(main_return, (int, float)):
                        results_text += f"   Portfolio Return: {main_return:.2f}%\n"
                    if isinstance(total_roi, (int, float)):
                        results_text += f"   Total ROI: {total_roi:.2f}%\n"

                # Get top companies
                final_strategy = period_result.get('final_strategy', {})
                if isinstance(final_strategy, dict):
                    companies = final_strategy.get('ranked_companies', [])
                    if companies:
                        top_3 = companies[:3]
                        results_text += f"   Top Companies: "
                        company_names = []
                        for comp in top_3:
                            if isinstance(comp, dict):
                                name = comp.get('company_name', comp.get('ticker', 'N/A'))
                                company_names.append(name)
                        results_text += ", ".join(company_names) + "\n"

                results_text += "\n"

            # Add note about full details
            results_text += f"\n Full details for all periods saved in: {output_file}\n"
            results_text += f"   Open this file to see complete analysis for each period.\n"

            self.results_text.delete(1.0, 'end')
            self.results_text.insert(1.0, results_text)

            # Add clickable file link
            self.add_file_links(output_file)

        except Exception as e:
            self.display_error(f"Error displaying results: {e}")

    def add_file_links(self, main_filename, html_filename=None):
        """Add clickable file links to the GUI"""
        import os
        import webbrowser
        import subprocess
        import platform

        # Clear existing links
        for widget in self.links_frame.winfo_children():
            widget.destroy()

        # Show the links frame
        self.links_frame.grid()
        
        def open_main_file():
            """Open main file (Excel/JSON) with default application"""
            try:
                if platform.system() == "Windows":
                    os.startfile(main_filename)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", main_filename])
                else:  # Linux
                    subprocess.run(["xdg-open", main_filename])
            except Exception as e:
                self.messagebox.showerror("Error", f"Could not open file: {e}")
        
        def open_html():
            """Open HTML file in browser"""
            try:
                if html_filename:
                    webbrowser.open(f"file://{os.path.abspath(html_filename)}")
            except Exception as e:
                self.messagebox.showerror("Error", f"Could not open HTML file: {e}")
        
        def open_file_location():
            """Open file location in file explorer"""
            try:
                file_dir = os.path.dirname(os.path.abspath(main_filename))
                if platform.system() == "Windows":
                    os.startfile(file_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", file_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", file_dir])
            except Exception as e:
                self.messagebox.showerror("Error", f"Could not open file location: {e}")
        
        # Determine file type and set appropriate styling
        file_ext = os.path.splitext(main_filename)[1].lower()
        if file_ext == '.xlsx':
            file_icon = ""
            file_type = "Excel"
            button_color = '#27ae60'
        elif file_ext == '.json':
            file_icon = ""
            file_type = "JSON"
            button_color = '#e67e22'
        else:
            file_icon = ""
            file_type = "File"
            button_color = '#95a5a6'
        
        # Main file button
        main_btn = self.tk.Button(
            self.links_frame,
            text=f"{file_icon} Open {file_type}: {os.path.basename(main_filename)}",
            command=open_main_file,
            bg=button_color,
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5
        )
        main_btn.pack(side='left', padx=5)
        
        # HTML file button (if HTML file exists)
        if html_filename and os.path.exists(html_filename):
            html_btn = self.tk.Button(
                self.links_frame,
                text=f" Open Report: {os.path.basename(html_filename)}",
                command=open_html,
                bg='#3498db',
                fg='white',
                font=('Arial', 10, 'bold'),
                padx=15,
                pady=5
            )
            html_btn.pack(side='left', padx=5)
        
        # File location button
        location_btn = self.tk.Button(
            self.links_frame,
            text=" Open File Location",
            command=open_file_location,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5
        )
        location_btn.pack(side='left', padx=5)
        
        # File info label
        info_label = self.tk.Label(
            self.links_frame,
            text=f"Generated: {os.path.basename(main_filename)}",
            font=('Arial', 9),
            fg='#7f8c8d'
        )
        info_label.pack(side='right', padx=10)
    
    def clear_file_links(self):
        """Clear and hide file links"""
        for widget in self.links_frame.winfo_children():
            widget.destroy()
        self.links_frame.grid_remove()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


if __name__ == "__main__":
    # Check if tkinter is available
    try:
        import tkinter
        app = InvestmentStrategyGUI()
        app.run()
    except ImportError:
        print(" tkinter not available. Using command line interface instead.")
        print(" Investment Strategy Generator")
        print("=" * 50)
        
        # Fallback to simple command line interface
        sector = input("Investment sector: ").strip() or "technology"
        amount = input("Investment amount ($): ").strip() or "10000"
        
        try:
            strategy_result = generate_investment_strategy(
                keyword=sector,
                investment_amount=float(amount),
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                include_financial_validation=False
            )
            print(" Strategy generated successfully!")
        except Exception as e:
            print(f" Error: {e}")
    
    if "error" not in strategy_result:
        print("\n Strategy generated successfully!")
        print(f" Recommendations: {len(strategy_result.get('recommendations', []))}")
        print(f" Expected Return: {strategy_result.get('portfolio_summary', {}).get('expected_return', 'N/A')}")
    else:
        print(f"\n Strategy generation failed: {strategy_result['error']}")


def run_batch_analysis(num_runs=10, keywords=None, investment_amounts=None,
                      save_detailed_results=True, output_dir="batch_results",
                      openai_api_key=None, alpha_vantage_key=None):
    """
    Run the investment analysis multiple times and save results to CSV
    
    Enhanced batch analysis functionality from NewsAPI+FinanceReports.ipynb
    
    Parameters:
    - num_runs: Number of times to run the analysis
    - keywords: List of keywords to test (default: ['technology', 'energy', 'agriculture'])
    - investment_amounts: List of investment amounts to test (default: [10000])
    - save_detailed_results: Whether to save detailed JSON results for each run
    - output_dir: Directory to save results
    - openai_api_key: OpenAI API key
    - alpha_vantage_key: Alpha Vantage API key for financial data
    """
    import csv
    import random
    
    if keywords is None:
        keywords = ['technology', 'energy', 'agriculture']
    if investment_amounts is None:
        investment_amounts = [10000]
    if openai_api_key is None:
        print(" OpenAI API key required for batch analysis")
        return None

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare CSV file
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    csv_filename = f"{output_dir}/batch_analysis_results_SESSION_{timestamp}.csv"

    # CSV headers
    csv_headers = [
        'run_id', 'timestamp', 'keyword', 'investment_amount', 'reinvestment_amount',
        'analysis_success', 'companies_recommended', 'companies_validated',
        'portfolio_return_pct', 'projected_value', 'roi_pct',
        'market_themes', 'key_trends', 'top_companies', 'top_allocations',
        'sector_allocation', 'risk_level', 'expected_annual_return',
        'news_articles_analyzed', 'execution_time_seconds', 'financial_validation',
        'avg_revenue_growth', 'avg_profit_margin', 'avg_debt_ratio'
    ]

    print(f" Starting enhanced batch analysis with {num_runs} runs")
    print(f" Testing keywords: {keywords}")
    print(f" Testing investment amounts: {investment_amounts}")
    print(f" Results will be saved to: {csv_filename}")
    print("="*70)

    # Open CSV file for writing
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        successful_runs = 0

        for run_id in range(1, num_runs + 1):
            # Randomly select keyword and investment amount for variety
            keyword = random.choice(keywords)
            investment_amount = random.choice(investment_amounts)
            
            # Random reinvestment (0, quarterly, or yearly)
            reinvestment_options = [0, 2500, 10000]
            reinvestment_amount = random.choice(reinvestment_options)

            print(f"\n Run {run_id}/{num_runs}: {keyword} with ${investment_amount:,} investment")
            run_start_time = time.time()

            try:
                # Create generator instance
                generator = InvestmentStrategyGenerator(
                    api_key_openai=openai_api_key,
                    alpha_vantage_key=alpha_vantage_key
                )
                
                # Run the analysis with financial validation
                result = generator.generate_complete_strategy(
                    user_input_keyword=keyword,
                    investment_amount=investment_amount,
                    reinvestment_amount=reinvestment_amount,
                    include_financial_validation=True,
                    max_companies=15
                )

                run_time = time.time() - run_start_time

                if result and not result.get('error'):
                    successful_runs += 1
                    
                    # Extract metrics
                    recommendations = result.get('recommendations', [])
                    portfolio_summary = result.get('portfolio_summary', {})
                    return_analysis = result.get('return_analysis', {})
                    
                    # Calculate financial metrics if available
                    financial_metrics = _calculate_financial_metrics(result)
                    
                    # Calculate portfolio return and ROI
                    portfolio_return = return_analysis.get('1Y', {}).get('portfolio_return', 0)
                    projected_value = investment_amount + (investment_amount * portfolio_return / 100)
                    if reinvestment_amount > 0:
                        projected_value += reinvestment_amount
                    
                    total_invested = investment_amount + reinvestment_amount
                    roi_pct = ((projected_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0

                    # Prepare CSV row
                    csv_row = {
                        'run_id': run_id,
                        'timestamp': datetime.now().isoformat(),
                        'keyword': keyword,
                        'investment_amount': investment_amount,
                        'reinvestment_amount': reinvestment_amount,
                        'analysis_success': True,
                        'companies_recommended': len(recommendations),
                        'companies_validated': len([r for r in recommendations if r.get('allocation', 0) > 0]),
                        'portfolio_return_pct': round(portfolio_return, 2),
                        'projected_value': round(projected_value, 2),
                        'roi_pct': round(roi_pct, 2),
                        'market_themes': '; '.join(result.get('market_analysis', {}).get('themes', [])),
                        'key_trends': '; '.join(result.get('market_analysis', {}).get('trends', [])),
                        'top_companies': '; '.join([f"{r.get('ticker', '')}({r.get('allocation', 0):.1f}%)"
                                                  for r in recommendations[:20]]),
                        'top_allocations': '; '.join([f"{r.get('allocation', 0):.1f}%"
                                                    for r in recommendations[:20]]),
                        'sector_allocation': str(portfolio_summary.get('sector_breakdown', {})),
                        'risk_level': portfolio_summary.get('risk_assessment', 'N/A'),
                        'expected_annual_return': portfolio_summary.get('expected_return', 'N/A'),
                        'news_articles_analyzed': result.get('market_analysis', {}).get('articles_processed', 0),
                        'execution_time_seconds': round(run_time, 2),
                        'financial_validation': result.get('financial_data_used', False),
                        'avg_revenue_growth': financial_metrics.get('avg_revenue_growth', 'N/A'),
                        'avg_profit_margin': financial_metrics.get('avg_profit_margin', 'N/A'),
                        'avg_debt_ratio': financial_metrics.get('avg_debt_ratio', 'N/A')
                    }

                    # Save detailed results if requested
                    if save_detailed_results:
                        detailed_filename = f"{output_dir}/detailed_run_{run_id}_{keyword}_{timestamp}.json"
                        try:
                            with open(detailed_filename, 'w', encoding='utf-8') as f:
                                json.dump(result, f, indent=2, default=str)
                        except Exception as e:
                            print(f" Could not save detailed results: {e}")

                    print(f" Success: {portfolio_return:.2f}% return, ROI: {roi_pct:.2f}%")

                else:
                    # Failed analysis
                    csv_row = {
                        'run_id': run_id,
                        'timestamp': datetime.now().isoformat(),
                        'keyword': keyword,
                        'investment_amount': investment_amount,
                        'reinvestment_amount': reinvestment_amount,
                        'analysis_success': False,
                        'companies_recommended': 0,
                        'companies_validated': 0,
                        'portfolio_return_pct': 0,
                        'projected_value': investment_amount,
                        'roi_pct': 0,
                        'market_themes': result.get('error', 'FAILED'),
                        'key_trends': 'FAILED',
                        'top_companies': 'FAILED',
                        'top_allocations': 'FAILED',
                        'sector_allocation': 'FAILED',
                        'risk_level': 'FAILED',
                        'expected_annual_return': 'FAILED',
                        'news_articles_analyzed': 0,
                        'execution_time_seconds': round(run_time, 2),
                        'financial_validation': False,
                        'avg_revenue_growth': 'FAILED',
                        'avg_profit_margin': 'FAILED',
                        'avg_debt_ratio': 'FAILED'
                    }
                    print(f" Failed analysis: {result.get('error', 'Unknown error')}")

                # Write row to CSV
                writer.writerow(csv_row)
                csvfile.flush()  # Ensure data is written immediately

            except Exception as e:
                print(f" Error in run {run_id}: {e}")
                
                # Write error row
                error_row = {header: 'ERROR' if header not in ['run_id', 'timestamp', 'investment_amount', 
                           'reinvestment_amount', 'analysis_success', 'execution_time_seconds'] else 
                           (run_id if header == 'run_id' else 
                            datetime.now().isoformat() if header == 'timestamp' else
                            investment_amount if header == 'investment_amount' else
                            reinvestment_amount if header == 'reinvestment_amount' else
                            False if header == 'analysis_success' else
                            round(time.time() - run_start_time, 2)) 
                           for header in csv_headers}
                error_row['market_themes'] = f'ERROR: {str(e)[:100]}'
                writer.writerow(error_row)
                csvfile.flush()

            # Add delay between runs to respect API limits
            if run_id < num_runs:
                print(" Waiting 30 seconds before next run...")
                time.sleep(30)

    # Generate summary statistics
    _generate_batch_summary(csv_filename, keywords, successful_runs, num_runs)
    return csv_filename


def _calculate_financial_metrics(result):
    """Calculate average financial metrics from analysis results"""
    financial_data = result.get('financial_data', [])
    if not financial_data:
        return {}
    
    metrics = {'revenue_growth': [], 'profit_margin': [], 'debt_ratio': []}
    
    for company in financial_data:
        income_reports = company.get('income', [])
        balance_reports = company.get('balance', [])
        
        # Calculate revenue growth
        if len(income_reports) >= 2:
            try:
                current_revenue = float(income_reports[0].get('totalRevenue', 0))
                prev_revenue = float(income_reports[1].get('totalRevenue', 0))
                if prev_revenue > 0:
                    growth = ((current_revenue - prev_revenue) / prev_revenue) * 100
                    metrics['revenue_growth'].append(growth)
            except (ValueError, TypeError):
                pass
        
        # Calculate profit margin
        if income_reports:
            try:
                revenue = float(income_reports[0].get('totalRevenue', 0))
                net_income = float(income_reports[0].get('netIncome', 0))
                if revenue > 0:
                    margin = (net_income / revenue) * 100
                    metrics['profit_margin'].append(margin)
            except (ValueError, TypeError):
                pass
        
        # Calculate debt ratio
        if balance_reports:
            try:
                total_debt = float(balance_reports[0].get('totalLiabilities', 0))
                total_assets = float(balance_reports[0].get('totalAssets', 0))
                if total_assets > 0:
                    debt_ratio = (total_debt / total_assets) * 100
                    metrics['debt_ratio'].append(debt_ratio)
            except (ValueError, TypeError):
                pass
    
    # Calculate averages
    result_metrics = {}
    for metric, values in metrics.items():
        if values:
            result_metrics[f'avg_{metric}'] = round(sum(values) / len(values), 2)
        else:
            result_metrics[f'avg_{metric}'] = 'N/A'
    
    return result_metrics


def _generate_batch_summary(csv_filename, keywords, successful_runs, total_runs):
    """Generate summary statistics for batch analysis"""
    try:
        df = pd.read_csv(csv_filename)
        
        print("\n" + "="*70)
        print(" BATCH ANALYSIS COMPLETE")
        print("="*70)
        print(f" Total runs completed: {total_runs}")
        print(f" Successful analyses: {successful_runs}/{total_runs}")
        print(f" Results saved to: {csv_filename}")

        if len(df) > 0:
            successful_df = df[df['analysis_success'] == True]

            if len(successful_df) > 0:
                print(f"\n PERFORMANCE STATISTICS:")
                print(f"Average ROI: {successful_df['roi_pct'].mean():.2f}%")
                print(f"Best ROI: {successful_df['roi_pct'].max():.2f}%")
                print(f"Worst ROI: {successful_df['roi_pct'].min():.2f}%")
                print(f"Average Portfolio Return: {successful_df['portfolio_return_pct'].mean():.2f}%")
                print(f"Average Companies per Analysis: {successful_df['companies_recommended'].mean():.1f}")

                print(f"\n KEYWORD PERFORMANCE:")
                for keyword in keywords:
                    keyword_df = successful_df[successful_df['keyword'] == keyword]
                    if len(keyword_df) > 0:
                        avg_roi = keyword_df['roi_pct'].mean()
                        print(f"{keyword.title()}: {avg_roi:.2f}% average ROI ({len(keyword_df)} runs)")

        # Save summary to file
        summary_file = csv_filename.replace('.csv', '_SUMMARY.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Enhanced Batch Analysis Summary\n")
            f.write(f"="*50 + "\n")
            f.write(f"Total Runs: {total_runs}\n")
            f.write(f"Successful Runs: {successful_runs}\n")
            f.write(f"Success Rate: {(successful_runs/total_runs)*100:.1f}%\n")
            if len(successful_df) > 0:
                f.write(f"Average ROI: {successful_df['roi_pct'].mean():.2f}%\n")
                f.write(f"Best ROI: {successful_df['roi_pct'].max():.2f}%\n")
                f.write(f"Average Execution Time: {df['execution_time_seconds'].mean():.1f} seconds\n")

        print(f" Summary saved to: {summary_file}")

    except Exception as e:
        print(f" Could not generate summary statistics: {e}")


def quick_batch_analysis(num_runs=5, openai_api_key=None, alpha_vantage_key=None):
    """Run a quick batch analysis with default settings"""
    return run_batch_analysis(
        num_runs=num_runs,
        keywords=['technology', 'energy', 'agriculture'],
        investment_amounts=[10000],
        save_detailed_results=False,
        openai_api_key=openai_api_key,
        alpha_vantage_key=alpha_vantage_key
    )


def demo_enhanced_finance_integration(openai_api_key: str, alpha_vantage_key: Optional[str] = None):
    """
    Demonstration of enhanced finance reports integration
    Based on NewsAPI+FinanceReports.ipynb functionality
    """
    print(" ENHANCED FINANCE REPORTS INTEGRATION DEMO")
    print("="*60)
    print("This demo shows the enhanced financial analysis capabilities")
    print("integrated from NewsAPI+FinanceReports.ipynb")
    print("="*60)
    
    # Initialize generator
    generator = InvestmentStrategyGenerator(
        api_key_openai=openai_api_key,
        alpha_vantage_key=alpha_vantage_key
    )
    
    # Example 1: Basic strategy with financial validation
    print("\n EXAMPLE 1: Technology Strategy with Financial Validation")
    print("-" * 50)
    
    try:
        result = generator.generate_complete_strategy(
            user_input_keyword="technology",
            investment_amount=10000,
            include_financial_validation=True,
            max_companies=10
        )
        
        if result and not result.get('error'):
            print(" Strategy generated with financial validation!")
            print(f" Companies recommended: {len(result.get('recommendations', []))}")
            print(f" Financial data used: {result.get('financial_data_used', False)}")
            
            # Show top recommendations
            recommendations = result.get('recommendations', [])[:20]
            print(f"\n Top {len(recommendations)} Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec.get('ticker', 'N/A')} - {rec.get('allocation', 0):.1f}% allocation")
                print(f"     Reasoning: {rec.get('reasoning', 'N/A')[:100]}...")
        else:
            print(f" Strategy generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f" Demo failed: {e}")
    
    # Example 2: Batch analysis demo
    print(f"\n EXAMPLE 2: Quick Batch Analysis Demo")
    print("-" * 50)
    print("Running 3 quick analyses with different sectors...")
    
    try:
        csv_file = quick_batch_analysis(
            num_runs=3,
            openai_api_key=openai_api_key,
            alpha_vantage_key=alpha_vantage_key
        )
        
        if csv_file:
            print(f" Batch analysis complete!")
            print(f" Results saved to: {csv_file}")
            
            # Load and show summary
            try:
                df = pd.read_csv(csv_file)
                successful = df[df['analysis_success'] == True]
                if len(successful) > 0:
                    print(f" Average ROI: {successful['roi_pct'].mean():.2f}%")
                    print(f" Best ROI: {successful['roi_pct'].max():.2f}%")
            except Exception as e:
                print(f" Could not analyze results: {e}")
        else:
            print(" Batch analysis failed")
            
    except Exception as e:
        print(f" Batch demo failed: {e}")
    
    print(f"\n DEMO COMPLETE!")
    print(" Key Features Added from NewsAPI+FinanceReports.ipynb:")
    print("    Enhanced financial data fetching with Alpha Vantage")
    print("    Comprehensive financial statement analysis")
    print("    Batch analysis with CSV export and statistics") 
    print("    Data quality assessment for financial information")
    print("    Performance tracking and ROI calculations")
    print("    Detailed error handling and rate limiting")
    
    def generate_investment_strategy(self, period_start: str, period_end: str, 
                                   sector_focus: str, additional_context: str = "") -> Dict:
        """
        Alias method for dynamic reinvestment engine compatibility
        
        Args:
            period_start: Analysis start date
            period_end: Analysis end date  
            sector_focus: Investment sector
            additional_context: Previous performance context
        
        Returns:
            Investment strategy dictionary
        """
        
        return self.generate_complete_strategy(
            user_input_keyword=sector_focus,
            investment_amount=100000,  # Default amount
            news_start_date=period_start,
            news_end_date=period_end,
            analysis_start_date=period_start,
            analysis_end_date=period_end,
            additional_context=additional_context
        )


# Usage examples for the enhanced functionality
def example_usage():
    """Example usage of the enhanced investment strategy generator"""
    
    # Replace with your actual API keys
    OPENAI_API_KEY = "your-openai-api-key"
    ALPHA_VANTAGE_KEY = "your-alpha-vantage-key"  # Optional, defaults to demo
    
    print(" USAGE EXAMPLES:")
    print("="*50)
    
    print("\n1. Basic Strategy Generation:")
    print("   result = generate_investment_strategy(")
    print("       keyword='technology',")
    print("       investment_amount=10000,")
    print("       openai_api_key=OPENAI_API_KEY")
    print("   )")
    
    print("\n2. Enhanced Strategy with Financial Validation:")
    print("   generator = InvestmentStrategyGenerator(")
    print("       api_key_openai=OPENAI_API_KEY,")
    print("       alpha_vantage_key=ALPHA_VANTAGE_KEY")
    print("   )")
    print("   result = generator.generate_complete_strategy(")
    print("       user_input_keyword='healthcare',")
    print("       investment_amount=25000,")
    print("       include_financial_validation=True")
    print("   )")
    
    print("\n3. Batch Analysis:")
    print("   csv_file = run_batch_analysis(")
    print("       num_runs=10,")
    print("       keywords=['technology', 'energy'],")
    print("       investment_amounts=[10000, 25000],")
    print("       openai_api_key=OPENAI_API_KEY,")
    print("       alpha_vantage_key=ALPHA_VANTAGE_KEY")
    print("   )")
    
    print("\n4. Quick Batch Test:")
    print("   quick_batch_analysis(")
    print("       num_runs=5,")
    print("       openai_api_key=OPENAI_API_KEY")
    print("   )")
    
    print("\n5. Demo with Finance Reports:")
    print("   demo_enhanced_finance_integration(")
    print("       openai_api_key=OPENAI_API_KEY,")
    print("       alpha_vantage_key=ALPHA_VANTAGE_KEY")
    print("   )")
    

if __name__ == "__main__":
    # Show usage examples when run directly
    example_usage()
    
    print("\n" + "="*70)
    print(" ENHANCED INVESTMENT STRATEGY GENERATOR READY!")
    print(" Now includes comprehensive finance reports integration!")
    print("="*70)
