#!/usr/bin/env python3
"""
black_litterman.py — Black-Litterman position sizing from LLM conviction views
==============================================================================

Why this exists
---------------
Across both papers the model's own allocation weights destroy value relative to
simply equal-weighting the same picks:

    Paper A (static, GPT-5.1)   +4.08pp for equal weight, 90 cells, p = 5.4e-10
    Paper B (reinvestment)     +11.42pp for equal weight, 144 cells, p = 3.1e-24

So the *selection* carries information and the *sizing* does not. The published
answer to that is not "let the LLM size anyway" — it is to use the LLM for views
and hand sizing to an optimizer. This module implements the standard
Black-Litterman construction for that split, following

    Y. Lee, Y. Kim, J. Kim, S. Kim and Y. Lee, "LLM-Enhanced Black-Litterman
    Portfolio Optimization" (arXiv:2504.14345, 2025), which rebalances on a
    fixed schedule, uses the language model only to generate return views, and
    delegates all position sizing to a Black-Litterman optimizer.

Model
-----
Equilibrium (reverse optimisation from the prior weights w_eq):

    Pi = delta * Sigma @ w_eq

Views. The LLM emits conviction weights m over the names it selected. We read a
name's over/under-weighting versus equal weight as a view on its excess return,
stated as a DEVIATION FROM EQUILIBRIUM and scaled by its own volatility so a tilt
on a volatile name is not treated as the same claim as the identical tilt on a
quiet one:

    q_i = Pi_i + view_strength * (m_i - 1/N) / (1/N) * sigma_i

    P   = I over the selected names        (absolute views)
    Omega = diag(tau * P Sigma P')         (Idzorek-style view uncertainty)

The Pi_i term is not cosmetic. Without it, a model that emits perfectly equal
weights supplies q = 0, which BL reads as "I predict exactly zero excess return
on every name" -- a strong active view -- and the posterior is dragged away from
the prior. Anchoring on Pi makes a zero tilt mean "no view", so BL then returns
the prior exactly: q = Pi gives E[R] = Pi and w* = inv(delta*Sigma) Pi = w_eq.
That identity is asserted in the unit tests.

Posterior:

    M      = inv( inv(tau*Sigma) + P' inv(Omega) P )
    E[R]   = M @ ( inv(tau*Sigma) @ Pi + P' inv(Omega) @ q )
    w_star = inv(delta * Sigma) @ E[R]

then long-only projection (clip negatives, renormalise). Long-only matters: the
simulator cannot short, so an unconstrained BL solution is not executable.

NO LOOKAHEAD
------------
Sigma must be estimated from returns ending STRICTLY BEFORE the decision date.
`covariance_from_prices` enforces that; it is the caller's job to pass a price
history that stops at the decision date.

Degrades gracefully: too few observations, a singular covariance, or fewer than
two priceable names all fall back to equal weight rather than raising, so a bad
window can never abort a run.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# Standard Black-Litterman parameters. delta is the market risk-aversion
# coefficient implied by a ~4% equity premium on ~16% vol; tau scales the
# uncertainty of the equilibrium prior and is conventionally small.
DEFAULT_DELTA = 2.5
DEFAULT_TAU = 0.05
DEFAULT_VIEW_STRENGTH = 0.02      # a 1x over-weight becomes a 2%*sigma view
MIN_OBS = 60                      # trading days needed for a usable covariance


def covariance_from_prices(prices: Dict[str, Sequence[float]],
                           names: List[str],
                           min_obs: int = MIN_OBS):
    """Annualised covariance of daily log returns, plus per-name volatility.

    `prices[t]` must END STRICTLY BEFORE the decision date. Returns
    (Sigma, sigma, usable_names) or (None, None, []) if not estimable.
    """
    series = []
    usable = []
    for t in names:
        p = np.asarray(prices.get(t, ()), dtype=float)
        p = p[np.isfinite(p) & (p > 0)]
        if p.size >= min_obs + 1:
            series.append(np.diff(np.log(p)))
            usable.append(t)
    if len(usable) < 2:
        return None, None, []
    n = min(len(s) for s in series)
    if n < min_obs:
        return None, None, []
    R = np.vstack([s[-n:] for s in series])          # names x days
    Sigma = np.cov(R) * 252.0
    if not np.all(np.isfinite(Sigma)):
        return None, None, []
    # ridge for conditioning; tiny relative to typical equity variances
    Sigma = Sigma + np.eye(len(usable)) * 1e-8
    sigma = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    return Sigma, sigma, usable


def black_litterman_weights(model_weights: Dict[str, float],
                            prices: Dict[str, Sequence[float]],
                            *,
                            delta: float = DEFAULT_DELTA,
                            tau: float = DEFAULT_TAU,
                            view_strength: float = DEFAULT_VIEW_STRENGTH,
                            prior: Optional[Dict[str, float]] = None,
                            ) -> Dict[str, float]:
    """LLM conviction weights -> Black-Litterman long-only target weights.

    Falls back to equal weight over the priceable names on any numerical
    problem, so the caller always receives a usable book.
    """
    names = [t for t, w in (model_weights or {}).items()
             if isinstance(w, (int, float)) and w > 0]
    if len(names) < 2:
        return {t: 1.0 / len(names) for t in names} if names else {}

    Sigma, sigma, usable = covariance_from_prices(prices, names)
    if Sigma is None:
        return {t: 1.0 / len(names) for t in names}

    N = len(usable)
    eq = 1.0 / N

    # equilibrium prior: equal weight unless the caller supplies one. The 29-name
    # reference universe is not a market portfolio and market caps are not stored
    # per run, so equal weight is the honest neutral prior here.
    if prior:
        w_eq = np.array([max(float(prior.get(t, eq)), 0.0) for t in usable])
        s = w_eq.sum()
        w_eq = w_eq / s if s > 0 else np.full(N, eq)
    else:
        w_eq = np.full(N, eq)

    Pi = delta * Sigma @ w_eq

    # Views from the model's tilt away from equal weight, anchored on equilibrium
    # so that "no tilt" means "no view" rather than "I predict zero return".
    m = np.array([float(model_weights[t]) for t in usable])
    m = m / m.sum() if m.sum() > 0 else np.full(N, eq)
    q = Pi + view_strength * ((m - eq) / eq) * sigma

    P = np.eye(N)
    tS = tau * Sigma
    Omega = np.diag(np.clip(np.diag(P @ tS @ P.T), 1e-10, None))

    try:
        inv_tS = np.linalg.inv(tS)
        inv_Om = np.linalg.inv(Omega)
        M = np.linalg.inv(inv_tS + P.T @ inv_Om @ P)
        er = M @ (inv_tS @ Pi + P.T @ inv_Om @ q)
        w = np.linalg.solve(delta * Sigma, er)
    except np.linalg.LinAlgError:
        return {t: eq for t in usable}

    w = np.clip(w, 0.0, None)                 # long-only: the sim cannot short
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return {t: eq for t in usable}
    w = w / s
    out = {t: float(x) for t, x in zip(usable, w) if x > 1e-6}
    return out or {t: eq for t in usable}


def equal_weights(model_weights: Dict[str, float]) -> Dict[str, float]:
    """1/N over the names the model selected. The +11.4pp baseline."""
    names = [t for t, w in (model_weights or {}).items()
             if isinstance(w, (int, float)) and w > 0]
    return {t: 1.0 / len(names) for t in names} if names else {}


def apply_weighting(scheme: str, model_weights: Dict[str, float],
                    prices: Optional[Dict[str, Sequence[float]]] = None,
                    **kw) -> Dict[str, float]:
    """Dispatch: 'model' (unchanged) | 'equal' | 'black_litterman'."""
    if scheme == "equal":
        return equal_weights(model_weights)
    if scheme == "black_litterman":
        if not prices:
            return equal_weights(model_weights)
        return black_litterman_weights(model_weights, prices, **kw)
    return dict(model_weights or {})
