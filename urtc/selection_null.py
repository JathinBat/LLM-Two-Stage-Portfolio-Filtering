"""Selection null — does Stage Two beat random picking from its OWN candidate pool?

Why this is the right null (Paper B, task B-3)
----------------------------------------------
The strategy is news-driven and unbounded: across the 90 legacy runs it held 36
distinct tickers, 10 of them outside the 29-name `sector_config.universe()` list.
So "equal-weight the 29 names" or "draw randomly from the 29 names" is NOT the
strategy's opportunity set, and beating it is not evidence of selection skill.
Those baselines are reference-only (see tier1a_baselines.py).

The honest test is: at each rebalance the model saw a specific pool of
`candidates`. Draw many random Y-name portfolios from THAT pool, hold each to the
next rebalance, and ask where the model's actual pick lands in that distribution.
Percentile 50 = no skill (random from its own shortlist does as well).

This needs the `candidates` logging added this cycle; the legacy 90 runs predate
it and are skipped automatically with a count.

Return basis: adjusted close (splits + dividends), matching tier1a_baselines.py.
Equal-weight over the Y names, held from one rebalance date to the next — the
same buy-and-hold-between-rebalances shape the simulator uses, so the model's
pick and its null are scored identically.

Usage
-----
    python selection_null.py                       # all runs under results/portfolio_sim
    python selection_null.py --dir results/portfolio_sim/feedback_on
    python selection_null.py --draws 20000 --seed 7
    python selection_null.py --by-arm              # split ON vs OFF (needs both arms)

Writes selection_null.txt (+ selection_null_periods.csv with per-period detail).
Needs yfinance and internet; run locally (the cloud sandbox 403s on Yahoo).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(ROOT, "results", "portfolio_sim")
OUT_TXT = os.path.join(ROOT, "selection_null.txt")
OUT_CSV = os.path.join(ROOT, "selection_null_periods.csv")


# --------------------------------------------------------------------------- #
#  Load runs                                                                   #
# --------------------------------------------------------------------------- #
def _is_dry(path: str, run: dict) -> bool:
    """Synthetic MockBackend output, which must never enter an analysis.

    Checked two ways because the two markers were added at different times:
    `_dry` in the filename and `metadata.dry_run`. A dry run's numbers are
    artifacts of the deterministic mock and would silently pollute any statistic
    computed over the folder.
    """
    if "_dry_" in os.path.basename(path):
        return True
    md = run.get("metadata")
    return bool(isinstance(md, dict) and md.get("dry_run"))


def load_runs(root: str) -> List[dict]:
    files = sorted(glob.glob(os.path.join(root, "**", "psperm_*.json"), recursive=True))
    runs, skipped_dry = [], 0
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                r = json.load(f)
        except Exception as e:                                   # noqa: BLE001
            print(f"  (unreadable: {os.path.basename(fp)} — {e})")
            continue
        if _is_dry(fp, r):
            skipped_dry += 1
            continue
        r["_file"] = fp
        runs.append(r)
    if skipped_dry:
        print(f"  skipped {skipped_dry} dry-run (MockBackend) files")
    return runs


def periods_with_candidates(runs: List[dict]):
    """(run, period, next_period) triples that can be scored.

    A period is scorable when it logs candidates AND a later period gives the
    exit date. The final period has no successor, so it is dropped — its holding
    interval ends at the run's end_date, which we use instead.
    """
    out, skipped_runs, skipped_periods = [], 0, 0
    for r in runs:
        ps = r.get("periods") or []
        if not any(p.get("candidates") for p in ps):
            skipped_runs += 1
            continue
        for i, p in enumerate(ps):
            if not p.get("candidates") or not p.get("holdings"):
                skipped_periods += 1
                continue
            exit_date = (ps[i + 1]["as_of"] if i + 1 < len(ps)
                         else r.get("end_date"))
            if not exit_date:
                skipped_periods += 1
                continue
            out.append((r, p, exit_date))
    return out, skipped_runs, skipped_periods


# --------------------------------------------------------------------------- #
#  Prices                                                                      #
# --------------------------------------------------------------------------- #
def fetch_prices(tickers, start, end) -> pd.DataFrame:
    import yfinance as yf
    # auto_adjust=True -> adjusted close (total-return basis), as in tier1a_baselines.py
    df = yf.download(sorted(set(tickers)), start=start, end=end,
                     auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(sorted(set(tickers))[0])
    return close.dropna(how="all")


def as_of_price(prices: pd.DataFrame, ticker: str, date: str) -> Optional[float]:
    """Last close on/before date — the same as-of convention the simulator uses."""
    if ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    s = s[s.index <= pd.Timestamp(date)]
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def basket_return(prices, names, entry, exit_) -> Optional[float]:
    """Equal-weight buy-and-hold return (%) over [entry, exit_]."""
    rets = []
    for t in names:
        p0 = as_of_price(prices, t, entry)
        p1 = as_of_price(prices, t, exit_)
        if p0 and p1:
            rets.append(p1 / p0 - 1.0)
    if not rets:
        return None
    return float(np.mean(rets) * 100.0)


# --------------------------------------------------------------------------- #
#  Null                                                                        #
# --------------------------------------------------------------------------- #
def null_percentile(prices, pool: List[str], y: int, entry: str, exit_: str,
                    actual: float, draws: int, rng) -> Optional[dict]:
    """Where does `actual` sit among random y-name draws from `pool`?"""
    priced = [t for t in pool
              if as_of_price(prices, t, entry) and as_of_price(prices, t, exit_)]
    if len(priced) <= y:
        return None            # pool no bigger than the pick -> no null to draw

    # Per-name returns once, then average random subsets (far cheaper than
    # re-pricing every draw).
    per = np.array([(as_of_price(prices, t, exit_) / as_of_price(prices, t, entry) - 1.0) * 100
                    for t in priced], dtype=float)
    idx = np.arange(len(priced))
    sims = np.empty(draws, dtype=float)
    for i in range(draws):
        sims[i] = per[rng.choice(idx, size=y, replace=False)].mean()

    pct = float((sims < actual).mean() * 100.0)
    sd = float(sims.std(ddof=1))
    return {
        "pool_priced": len(priced),
        "null_mean": float(sims.mean()),
        "null_median": float(np.median(sims)),
        "null_sd": sd,
        "percentile": pct,
        "z": float((actual - sims.mean()) / sd) if sd > 0 else np.nan,
        "beat_median": bool(actual > float(np.median(sims))),
    }


# --------------------------------------------------------------------------- #
#  Report                                                                      #
# --------------------------------------------------------------------------- #
def binom_ci(k: int, n: int):
    try:
        from scipy.stats import binomtest
        r = binomtest(k, n).proportion_ci()
        return r.low * 100, r.high * 100
    except Exception:                                            # noqa: BLE001
        return float("nan"), float("nan")


def summarize(rows: List[dict], label: str, lines: List[str]) -> None:
    if not rows:
        lines.append(f"\n{label}: no scorable periods.")
        return
    pcts = np.array([r["percentile"] for r in rows], float)
    zs = np.array([r["z"] for r in rows], float)
    zs = zs[np.isfinite(zs)]
    beat = sum(1 for r in rows if r["beat_median"])
    n = len(rows)
    lo, hi = binom_ci(beat, n)

    lines.append(f"\n{label}  (n = {n} rebalance decisions)")
    lines.append(f"  median percentile vs own candidate pool : {np.median(pcts):5.1f}"
                 f"   (50 = no selection skill)")
    lines.append(f"  mean percentile                         : {np.mean(pcts):5.1f}")
    lines.append(f"  beat its pool's median portfolio        : {beat}/{n} "
                 f"({beat / n * 100:.1f}%, 95% CI {lo:.0f}-{hi:.0f}%)")
    if len(zs):
        lines.append(f"  mean z vs null                          : {np.mean(zs):+.3f}")
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(pcts - 50.0)
        lines.append(f"  Wilcoxon percentile vs 50               : p = {p:.4f}")
    except Exception:                                            # noqa: BLE001
        pass
    q = np.percentile(pcts, [25, 50, 75])
    lines.append(f"  percentile quartiles                    : "
                 f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="results directory (searched recursively)")
    ap.add_argument("--draws", type=int, default=10000, help="random portfolios per period")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--by-arm", action="store_true",
                    help="also break results out by feedback ON/OFF")
    args = ap.parse_args()

    print(f"Loading runs from {args.dir} ...")
    runs = load_runs(args.dir)
    if not runs:
        print("No psperm_*.json found.")
        return 1
    trip, skipped_runs, skipped_periods = periods_with_candidates(runs)
    print(f"  {len(runs)} runs, {len(trip)} scorable rebalance decisions")
    print(f"  skipped {skipped_runs} runs with no candidate logging (legacy schema)")
    if skipped_periods:
        print(f"  skipped {skipped_periods} periods missing candidates/holdings")
    if not trip:
        print("\nNothing scorable: these runs predate candidate logging. "
              "Run the feedback experiment first, then re-run this.")
        return 1

    tickers = set()
    dmin, dmax = None, None
    for _, p, exit_ in trip:
        tickers.update(p["candidates"])
        tickers.update(p["holdings"])
        for d in (p["as_of"], exit_):
            dmin = d if dmin is None or d < dmin else dmin
            dmax = d if dmax is None or d > dmax else dmax
    print(f"  {len(tickers)} distinct tickers, {dmin} -> {dmax}")

    print("Fetching prices (yfinance; local only)...")
    try:
        prices = fetch_prices(sorted(tickers),
                              (pd.Timestamp(dmin) - pd.DateOffset(days=10)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(dmax) + pd.DateOffset(days=5)).strftime("%Y-%m-%d"))
    except Exception as e:                                       # noqa: BLE001
        print(f"Price fetch failed: {e}\nThis script needs local internet access to Yahoo.")
        return 1
    print(f"  got {prices.shape[1]} tickers x {prices.shape[0]} trading days")

    rng = np.random.default_rng(args.seed)
    rows, unscorable = [], 0
    for run, p, exit_ in trip:
        entry = p["as_of"]
        held = list(p["holdings"])
        actual = basket_return(prices, held, entry, exit_)
        if actual is None:
            unscorable += 1
            continue
        res = null_percentile(prices, list(p["candidates"]), len(held),
                              entry, exit_, actual, args.draws, rng)
        if res is None:
            unscorable += 1
            continue
        md = run.get("metadata") or {}
        rows.append({
            "file": os.path.basename(run["_file"]),
            "arm": "ON" if run.get("feedback", True) else "OFF",
            "screened_news": run.get("screened_news", False),
            "window_start": md.get("analysis_start_date", ""),
            "freq": md.get("freq", ""),
            "period": p.get("period"),
            "stage": p.get("stage"),
            "entry": entry, "exit": exit_,
            "y": len(held), "pool": len(p["candidates"]),
            "actual_ret_pct": round(actual, 3),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()},
        })

    if not rows:
        print("No periods could be scored (price coverage too sparse).")
        return 1
    if unscorable:
        print(f"  {unscorable} decisions unscorable (pool <= Y, or no price coverage)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    lines = ["SELECTION NULL — model's pick vs random draws from its OWN candidate pool",
             "=" * 74,
             "",
             f"runs dir      : {args.dir}",
             f"draws/period  : {args.draws:,}   seed {args.seed}",
             f"return basis  : adjusted close, equal-weight, held to the next rebalance",
             f"scored        : {len(df)} rebalance decisions "
             f"({df['file'].nunique()} runs)",
             f"skipped       : {skipped_runs} legacy runs (no candidate logging), "
             f"{unscorable} unscorable decisions",
             "",
             "A percentile of 50 means the model did no better than drawing at random",
             "from the shortlist it built. Above 50 = Stage Two adds value over its own",
             "opportunity set; below 50 = the shortlist was better than the final pick."]

    summarize(df.to_dict("records"), "ALL DECISIONS", lines)

    if args.by_arm and df["arm"].nunique() > 1:
        for arm in ("ON", "OFF"):
            sub = df[df["arm"] == arm]
            summarize(sub.to_dict("records"), f"feedback {arm}", lines)

    # Initial buy vs later rebalances — is the value in the first pick or the reselection?
    for stage in df["stage"].dropna().unique():
        sub = df[df["stage"] == stage]
        if len(sub) >= 5:
            summarize(sub.to_dict("records"), f"stage = {stage}", lines)

    lines += ["", "Per-decision detail: " + os.path.basename(OUT_CSV)]
    text = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nWrote {os.path.basename(OUT_TXT)} and {os.path.basename(OUT_CSV)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
