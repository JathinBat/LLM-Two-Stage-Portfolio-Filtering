"""Why does rebalancing lose to buy-and-hold? Per-decision attribution.

The strategy beats XLK 9/9 windows but loses to INIT (buy-and-hold of its own
period-0 picks) in 22/90 runs, mean -10.1pp. That means the reinvestment layer,
not the selection layer, is destroying value. This script localizes the damage.

For every rebalance it builds the counterfactual "do nothing" portfolio — the
prior holdings and prior cash, left untouched — and scores it over exactly the
same interval as the portfolio the model actually built:

    rebalance_alpha = return(new portfolio) - return(held portfolio)

Then it decomposes the loss into the mechanisms that can cause it:
  * cash drag        — capital parked instead of invested
  * dropped winners  — names sold that then outperformed what replaced them
  * added losers     — names bought that underperformed what they displaced
  * concentration    — weight shifted toward worse performers among kept names

Works on ALL runs (legacy included): prices come from yfinance, so it does not
need the newer prior_holdings_perf logging.

Usage
-----
    python rebalance_attribution.py
    python rebalance_attribution.py --dir results/portfolio_sim/feedback_on
    python rebalance_attribution.py --worst 15      # show the 15 costliest decisions

Writes rebalance_attribution.txt + rebalance_attribution_decisions.csv.
Needs yfinance and internet; run locally.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(ROOT, "results", "portfolio_sim")
OUT_TXT = os.path.join(ROOT, "rebalance_attribution.txt")
OUT_CSV = os.path.join(ROOT, "rebalance_attribution_decisions.csv")


def _is_dry(path: str, run: dict) -> bool:
    """Synthetic MockBackend output, which must never enter an analysis.

    Checked two ways because the two markers were added at different times:
    `_dry` in the filename and `metadata.dry_run`.
    """
    if "_dry_" in os.path.basename(path):
        return True
    md = run.get("metadata")
    return bool(isinstance(md, dict) and md.get("dry_run"))


def load_runs(root: str) -> List[dict]:
    runs, skipped_dry = [], 0
    for fp in sorted(glob.glob(os.path.join(root, "**", "psperm_*.json"), recursive=True)):
        try:
            with open(fp, encoding="utf-8") as f:
                r = json.load(f)
        except Exception:                                        # noqa: BLE001
            continue
        if _is_dry(fp, r):
            skipped_dry += 1
            continue
        r["_file"] = fp
        runs.append(r)
    if skipped_dry:
        print(f"  skipped {skipped_dry} dry-run (MockBackend) files")
    return runs


def fetch_prices(tickers, start, end) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(sorted(set(tickers)), start=start, end=end,
                     auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(sorted(set(tickers))[0])
    return close.dropna(how="all")


def px(prices: pd.DataFrame, t: str, date: str) -> Optional[float]:
    if t not in prices.columns:
        return None
    s = prices[t].dropna()
    s = s[s.index <= pd.Timestamp(date)]
    if s.empty:
        return None
    v = float(s.iloc[-1])
    return v if np.isfinite(v) and v > 0 else None


def weights_from_shares(shares: Dict[str, float], prices, date: str, cash: float):
    """Value weights including a cash sleeve (cash earns 0 over the interval)."""
    vals, total = {}, float(cash or 0.0)
    for t, sh in shares.items():
        p = px(prices, t, date)
        if p and sh:
            vals[t] = sh * p
            total += vals[t]
    if total <= 0:
        return None, 0.0
    return {t: v / total for t, v in vals.items()}, float(cash or 0.0) / total


def portfolio_return(w: Dict[str, float], prices, entry: str, exit_: str) -> Optional[float]:
    """Weighted buy-and-hold return (%); un-invested weight earns 0."""
    tot = 0.0
    for t, wt in w.items():
        p0, p1 = px(prices, t, entry), px(prices, t, exit_)
        if p0 and p1:
            tot += wt * (p1 / p0 - 1.0)
    return tot * 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--worst", type=int, default=12, help="how many worst decisions to list")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    print(f"{len(runs)} runs from {args.dir}")
    if not runs:
        return 1

    # Collect tickers + date span
    tickers, dmin, dmax = set(), None, None
    for r in runs:
        for p in (r.get("periods") or []):
            tickers.update((p.get("holdings") or {}).keys())
            d = p.get("as_of")
            if d:
                dmin = d if dmin is None or d < dmin else dmin
                dmax = d if dmax is None or d > dmax else dmax
        if r.get("end_date"):
            dmax = max(dmax or "", r["end_date"])
    print(f"{len(tickers)} tickers, {dmin} -> {dmax}\nFetching prices...")
    try:
        prices = fetch_prices(sorted(tickers),
                              (pd.Timestamp(dmin) - pd.DateOffset(days=10)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(dmax) + pd.DateOffset(days=5)).strftime("%Y-%m-%d"))
    except Exception as e:                                       # noqa: BLE001
        print(f"Price fetch failed: {e}")
        return 1
    print(f"  {prices.shape[1]} tickers x {prices.shape[0]} days")

    rows = []
    for r in runs:
        ps = r.get("periods") or []
        md = r.get("metadata") or {}
        for i in range(1, len(ps)):
            prev, cur = ps[i - 1], ps[i]
            entry = cur.get("as_of")
            exit_ = ps[i + 1]["as_of"] if i + 1 < len(ps) else r.get("end_date")
            if not entry or not exit_:
                continue
            prev_sh = {t: h.get("shares", 0) for t, h in (prev.get("holdings") or {}).items()}
            cur_sh = {t: h.get("shares", 0) for t, h in (cur.get("holdings") or {}).items()}
            if not prev_sh or not cur_sh:
                continue

            # counterfactual: keep prior shares AND prior cash, untouched
            w_hold, cash_hold = weights_from_shares(prev_sh, prices, entry,
                                                    prev.get("cash_after", 0.0))
            w_new, cash_new = weights_from_shares(cur_sh, prices, entry,
                                                  cur.get("cash_after", 0.0))
            if w_hold is None or w_new is None:
                continue
            r_hold = portfolio_return(w_hold, prices, entry, exit_)
            r_new = portfolio_return(w_new, prices, entry, exit_)
            if r_hold is None or r_new is None:
                continue
            alpha = r_new - r_hold

            # per-name forward returns over this interval
            fwd = {}
            for t in set(w_hold) | set(w_new):
                p0, p1 = px(prices, t, entry), px(prices, t, exit_)
                if p0 and p1:
                    fwd[t] = (p1 / p0 - 1.0) * 100.0
            dropped = [t for t in w_hold if t not in w_new]
            added = [t for t in w_new if t not in w_hold]
            kept = [t for t in w_new if t in w_hold]
            mean_all = np.mean(list(fwd.values())) if fwd else np.nan

            rows.append({
                "file": os.path.basename(r["_file"]),
                "window": md.get("analysis_start_date", "")[:7],
                "freq": md.get("freq"), "cfg": f"{md.get('filter_X')}-{md.get('filter_Y')}",
                "arm": "ON" if r.get("feedback", True) else "OFF",
                "period": cur.get("period"), "entry": entry, "exit": exit_,
                "ret_new": round(r_new, 3), "ret_hold": round(r_hold, 3),
                "rebalance_alpha": round(alpha, 3),
                "cash_w_new": round(cash_new * 100, 2),
                "cash_w_hold": round(cash_hold * 100, 2),
                "cash_drag_pp": round(-(cash_new - cash_hold) * mean_all, 3)
                                if np.isfinite(mean_all) else np.nan,
                "n_dropped": len(dropped), "n_added": len(added), "n_kept": len(kept),
                "dropped_fwd": round(float(np.mean([fwd[t] for t in dropped if t in fwd])), 2)
                               if any(t in fwd for t in dropped) else np.nan,
                "added_fwd": round(float(np.mean([fwd[t] for t in added if t in fwd])), 2)
                             if any(t in fwd for t in added) else np.nan,
                "dropped_names": ",".join(dropped[:6]),
                "added_names": ",".join(added[:6]),
            })

    if not rows:
        print("No scorable rebalances.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    L = []
    A = df.rebalance_alpha
    L.append("REBALANCE ATTRIBUTION — new portfolio vs simply holding the old one")
    L.append("=" * 72)
    L.append(f"\n{len(df)} rebalance decisions across {df.file.nunique()} runs")
    L.append(f"\nrebalance alpha (per decision, pp):")
    L.append(f"  mean   {A.mean():+.3f}    median {A.median():+.3f}")
    L.append(f"  helped {int((A > 0).sum())}/{len(A)} ({(A > 0).mean() * 100:.1f}%)   "
             f"hurt {int((A < 0).sum())}/{len(A)}")
    L.append(f"  worst  {A.min():+.2f}     best {A.max():+.2f}")
    L.append(f"  sum of negative alphas {A[A < 0].sum():+.1f}pp vs "
             f"positive {A[A > 0].sum():+.1f}pp")
    try:
        from scipy.stats import wilcoxon
        L.append(f"  Wilcoxon vs 0: p = {wilcoxon(A)[1]:.4f}")
    except Exception:                                            # noqa: BLE001
        pass

    L.append(f"\nis the damage concentrated or spread?")
    srt = A.sort_values()
    L.append(f"  worst 10% of decisions contribute {srt.head(max(1, len(A) // 10)).sum():+.1f}pp")
    L.append(f"  all negative decisions            {A[A < 0].sum():+.1f}pp")

    L.append(f"\nmechanisms (mean per decision):")
    L.append(f"  cash weight after rebalance : {df.cash_w_new.mean():.2f}% "
             f"(held portfolio {df.cash_w_hold.mean():.2f}%)")
    L.append(f"  est. cash drag              : {df.cash_drag_pp.mean():+.3f}pp")
    L.append(f"  names dropped / added / kept: {df.n_dropped.mean():.2f} / "
             f"{df.n_added.mean():.2f} / {df.n_kept.mean():.2f}")
    d, a = df.dropped_fwd.dropna(), df.added_fwd.dropna()
    if len(d) and len(a):
        L.append(f"  forward return of DROPPED names: {d.mean():+.2f}%")
        L.append(f"  forward return of ADDED   names: {a.mean():+.2f}%")
        L.append(f"  -> swap edge {a.mean() - d.mean():+.2f}pp "
                 f"({'buying better' if a.mean() > d.mean() else 'SOLD THE WINNERS'})")

    L.append(f"\nby cadence:")
    for fq, g in df.groupby("freq"):
        L.append(f"  freq {int(fq):>2}x: n={len(g):4d}  mean alpha {g.rebalance_alpha.mean():+.3f}pp  "
                 f"helped {(g.rebalance_alpha > 0).mean() * 100:4.1f}%  "
                 f"cash {g.cash_w_new.mean():4.1f}%")

    L.append(f"\n{args.worst} costliest single decisions:")
    for _, r_ in df.nsmallest(args.worst, "rebalance_alpha").iterrows():
        L.append(f"  {r_.rebalance_alpha:+7.2f}pp  {r_.window} {int(r_.freq)}x p{r_.period:<2} "
                 f"{r_.entry}->{r_.exit}  dropped [{r_.dropped_names}] ({r_.dropped_fwd:+.1f}%) "
                 f"added [{r_.added_names}] ({r_.added_fwd:+.1f}%)")

    text = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nWrote {os.path.basename(OUT_TXT)} + {os.path.basename(OUT_CSV)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
