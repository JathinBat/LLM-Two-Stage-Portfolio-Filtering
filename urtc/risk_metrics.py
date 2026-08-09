#!/usr/bin/env python3
"""
risk_metrics.py — risk-adjusted performance metrics from a portfolio value series
=================================================================================

Pure, dependency-light functions (numpy only) computed from a daily portfolio
value series. Reusable by portfolio_sim_runner.py, reinvestment_runner.py, and the
main pipeline's per-window analysis (per Paper_Split_Handoff.md §6: these metrics
need NO new model calls — they reconstruct from the daily adjusted-close path while
shares are held fixed between rebalances).

Conventions
-----------
  • Daily series, annualized with 252 trading days.
  • Risk-free rate is an ANNUAL decimal (e.g. 0.04); converted to a per-day rate.
  • Sharpe  = mean(excess daily) / std(excess daily)            × √252
  • Sortino = mean(excess daily) / downside-deviation           × √252
              downside-deviation = √(mean(min(0, excess)²))  (MAR = risk-free)
  • Max drawdown = min over t of (V_t / running-peak − 1)  (a negative number)
  • Calmar = annualized return / |max drawdown|
  • Beta/alpha = OLS of portfolio daily returns on benchmark daily returns;
                 alpha reported annualized (CAPM intercept × 252).

All functions degrade gracefully: a series too short to be meaningful returns None
for the affected metric rather than raising.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
#  Building blocks                                                            #
# --------------------------------------------------------------------------- #
def daily_returns(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return np.array([])
    v = np.where(v == 0, np.nan, v)
    r = v[1:] / v[:-1] - 1.0
    return r[np.isfinite(r)]


def annualized_return(values: Sequence[float], periods_per_year: int = TRADING_DAYS) -> Optional[float]:
    v = np.asarray(values, dtype=float)
    if v.size < 2 or v[0] <= 0:
        return None
    n_intervals = v.size - 1
    total_growth = v[-1] / v[0]
    if total_growth <= 0:
        return None
    return float(total_growth ** (periods_per_year / n_intervals) - 1.0)


def annualized_volatility(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> Optional[float]:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(values: Sequence[float]) -> Optional[float]:
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    running_peak = np.maximum.accumulate(v)
    drawdowns = v / running_peak - 1.0
    return float(drawdowns.min())


def sharpe_ratio(returns: Sequence[float], rf_annual: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> Optional[float]:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    rf_daily = rf_annual / periods_per_year
    excess = r - rf_daily
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return None
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: Sequence[float], rf_annual: float = 0.0,
                  periods_per_year: int = TRADING_DAYS) -> Optional[float]:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    rf_daily = rf_annual / periods_per_year
    excess = r - rf_daily
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return None  # no downside observed
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def calmar_ratio(ann_return: Optional[float], mdd: Optional[float]) -> Optional[float]:
    if ann_return is None or mdd is None or mdd == 0:
        return None
    return float(ann_return / abs(mdd))


def beta_alpha(port_returns: Sequence[float], bench_returns: Sequence[float],
               rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS) -> Dict[str, Optional[float]]:
    rp = np.asarray(port_returns, dtype=float)
    rm = np.asarray(bench_returns, dtype=float)
    n = min(rp.size, rm.size)
    if n < 3:
        return {"beta": None, "alpha_annual": None, "r_squared": None}
    rp, rm = rp[-n:], rm[-n:]
    var_m = np.var(rm, ddof=1)
    if var_m == 0:
        return {"beta": None, "alpha_annual": None, "r_squared": None}
    cov = np.cov(rp, rm, ddof=1)[0, 1]
    beta = cov / var_m
    rf_daily = rf_annual / periods_per_year
    alpha_daily = np.mean(rp - rf_daily) - beta * np.mean(rm - rf_daily)
    corr = np.corrcoef(rp, rm)[0, 1]
    return {"beta": float(beta),
            "alpha_annual": float(alpha_daily * periods_per_year),
            "r_squared": float(corr ** 2)}


# --------------------------------------------------------------------------- #
#  One-call summary                                                          #
# --------------------------------------------------------------------------- #
def compute_all(values: Sequence[float],
                benchmark_values: Optional[Sequence[float]] = None,
                rf_annual: float = 0.0,
                periods_per_year: int = TRADING_DAYS) -> Dict[str, Optional[float]]:
    """Full risk summary from a daily value series (+ optional benchmark series)."""
    rets = daily_returns(values)
    ann_ret = annualized_return(values, periods_per_year)
    mdd = max_drawdown(values)
    out: Dict[str, Optional[float]] = {
        "n_observations": int(len(values)),
        "total_return": (float(values[-1] / values[0] - 1.0)
                         if len(values) >= 2 and values[0] > 0 else None),
        "annualized_return": ann_ret,
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, rf_annual, periods_per_year),
        "sortino": sortino_ratio(rets, rf_annual, periods_per_year),
        "max_drawdown": mdd,
        "calmar": calmar_ratio(ann_ret, mdd),
        "risk_free_annual": rf_annual,
    }
    if benchmark_values is not None and len(benchmark_values) >= 3:
        out.update(beta_alpha(rets, daily_returns(benchmark_values), rf_annual, periods_per_year))
    else:
        out.update({"beta": None, "alpha_annual": None, "r_squared": None})
    return out


def format_summary(m: Dict[str, Optional[float]]) -> str:
    def pct(x): return "  n/a " if x is None else f"{x*100:+6.2f}%"
    def num(x): return " n/a " if x is None else f"{x:6.2f}"
    lines = [
        f"  Observations (daily)   : {m.get('n_observations')}",
        f"  Total return           : {pct(m.get('total_return'))}",
        f"  Annualized return      : {pct(m.get('annualized_return'))}",
        f"  Annualized volatility  : {pct(m.get('annualized_volatility'))}",
        f"  Sharpe ratio           : {num(m.get('sharpe'))}",
        f"  Sortino ratio          : {num(m.get('sortino'))}",
        f"  Max drawdown           : {pct(m.get('max_drawdown'))}",
        f"  Calmar ratio           : {num(m.get('calmar'))}",
        f"  Beta vs benchmark      : {num(m.get('beta'))}",
        f"  Alpha (annualized)     : {pct(m.get('alpha_annual'))}",
        f"  R² vs benchmark        : {num(m.get('r_squared'))}",
    ]
    return "\n".join(lines)
