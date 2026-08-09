"""Which permutations have enough news to be worth running?

Paper B's news corpus is very uneven: 2025Q1/Q2 hold ~12-14 screened articles per
QUARTER while 2025Q4 onward hold ~150+. A rebalance asked to pick X candidates
from a dozen stale articles is close to degenerate, and running it burns API
budget to produce a decision the data cannot support.

This script counts, for every permutation cell, how many articles each of its
decision dates can actually see (the same [as_of - lookback, as_of] window the
runner uses) and reports how many cells survive at various minimums. Use it to
choose the "Min articles" threshold in the permutation-runner GUI.

It makes NO API calls and reads only local CSVs.

Usage
-----
    python news_coverage_check.py                    # screened corpus, default grid
    python news_coverage_check.py --raw              # score the unscreened corpus
    python news_coverage_check.py --freqs 12 --filters 20-10
    python news_coverage_check.py --min 50           # detail for one threshold
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

import portfolio_sim_permutation_runner as R
import portfolio_sim_runner as psr

ROOT = os.path.dirname(os.path.abspath(__file__))
SCREENED = os.path.join(ROOT, "tech_screened_corpus.csv")
RAW = os.path.join(ROOT, "merged_news_data.csv")
LOOKBACK_YEARS = 2


def load_dates(screened: bool) -> pd.Series:
    """Publication dates of the corpus the runs would actually read."""
    import glob
    if screened:
        if not os.path.exists(SCREENED):
            print(f"{os.path.basename(SCREENED)} not found — run screen_news_relevance.py, "
                  f"or pass --raw.")
            sys.exit(1)
        paths = [SCREENED]
    else:
        paths = [RAW] + sorted(glob.glob(os.path.join(ROOT, "tech_news_backfill_*.csv")))
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p, usecols=["pub_date"]))
    d = pd.concat(frames, ignore_index=True)
    return pd.to_datetime(d.pub_date, errors="coerce").dropna().sort_values()


def decision_dates(p: dict) -> List[str]:
    """Exact as_of dates this permutation would decide on (runner's own scheduler)."""
    segs = psr.build_segments(p["start"].strftime("%Y-%m-%d"),
                              p["end"].strftime("%Y-%m-%d"), n_segments=p["freq"])
    return [s["start"] for s in segs]


def counts_for(dates: pd.Series, as_ofs: List[str]) -> List[int]:
    out = []
    for a in as_ofs:
        end = pd.Timestamp(a)
        start = end - pd.DateOffset(years=LOOKBACK_YEARS)
        out.append(int(((dates >= start) & (dates <= end)).sum()))
    return out


def recent_counts_for(dates: pd.Series, as_ofs: List[str], days: int = 90) -> List[int]:
    """Articles in the RECENT window before each decision — what drives a news strategy."""
    out = []
    for a in as_ofs:
        end = pd.Timestamp(a)
        out.append(int(((dates > end - pd.DateOffset(days=days)) & (dates <= end)).sum()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="score the unscreened corpus instead")
    ap.add_argument("--freqs", default="", help="comma-separated cadences, e.g. 2,12")
    ap.add_argument("--filters", default="", help="comma-separated X-Y, e.g. 20-10")
    ap.add_argument("--models", default="gpt-5.1")
    ap.add_argument("--min", type=int, default=0, help="show per-cell detail at this threshold")
    ap.add_argument("--recent-days", type=int, default=90,
                    help="window for the recency check (default 90)")
    args = ap.parse_args()

    dates = load_dates(not args.raw)
    label = "unscreened" if args.raw else "screened"
    print(f"corpus: {label}, {len(dates)} articles, "
          f"{dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}\n")

    perms = R.build_permutations()
    want_f = {int(x) for x in args.freqs.split(",") if x.strip()} if args.freqs else None
    want_c = ({tuple(int(v) for v in x.split("-"))
               for x in args.filters.split(",") if x.strip()} if args.filters else None)
    want_m = {m.strip() for m in args.models.split(",") if m.strip()} if args.models else None

    rows = []
    for p in perms:
        if p["status"] == R.ST_SKIPPED:
            continue                       # already excluded (training cutoff etc.)
        if want_f and p["freq"] not in want_f:
            continue
        if want_c and (p["X"], p["Y"]) not in want_c:
            continue
        if want_m and p["model"] not in want_m:
            continue
        as_ofs = decision_dates(p)
        lb = counts_for(dates, as_ofs)
        rc = recent_counts_for(dates, as_ofs, args.recent_days)
        rows.append({
            "id": p["id"], "model": p["model"], "window": f"{p['start']:%Y-%m}",
            "freq": p["freq"], "cfg": R.filter_label(p["X"], p["Y"]),
            "decisions": len(as_ofs),
            "min_lookback": min(lb), "median_lookback": int(pd.Series(lb).median()),
            "min_recent": min(rc), "median_recent": int(pd.Series(rc).median()),
        })

    if not rows:
        print("No permutations matched those filters.")
        return 1
    df = pd.DataFrame(rows)
    print(f"{len(df)} candidate cells (model={args.models})\n")

    print(f"cells surviving a minimum on the WORST decision's {LOOKBACK_YEARS}-year lookback:")
    for thr in (0, 25, 50, 100, 200, 400):
        k = int((df.min_lookback >= thr).sum())
        print(f"  >= {thr:4d} articles : {k:4d}/{len(df)} cells ({k / len(df) * 100:5.1f}%)")

    print(f"\ncells surviving a minimum on the worst decision's last {args.recent_days} days:")
    for thr in (0, 5, 10, 25, 50, 100):
        k = int((df.min_recent >= thr).sum())
        print(f"  >= {thr:4d} articles : {k:4d}/{len(df)} cells ({k / len(df) * 100:5.1f}%)")

    print("\nby window start (min across that window's decisions):")
    g = df.groupby("window").agg(cells=("id", "count"),
                                 worst_lookback=("min_lookback", "min"),
                                 worst_recent=("min_recent", "min"))
    for w, r in g.iterrows():
        flag = "  <-- STARVED" if r.worst_recent < 10 else ""
        print(f"  {w}: {int(r.cells):3d} cells, worst lookback {int(r.worst_lookback):5d}, "
              f"worst {args.recent_days}d {int(r.worst_recent):4d}{flag}")

    if args.min:
        keep = df[df.min_lookback >= args.min]
        print(f"\n--min {args.min}: {len(keep)}/{len(df)} cells pass. Windows kept: "
              f"{sorted(keep.window.unique())}")
        drop = df[df.min_lookback < args.min]
        if len(drop):
            print(f"  dropped windows: {sorted(drop.window.unique())}")

    out = os.path.join(ROOT, f"news_coverage_{label}.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {os.path.basename(out)}")
    print("\nSet the matching value in the runner GUI's 'Min articles' field to enforce it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
