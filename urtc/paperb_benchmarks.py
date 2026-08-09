#!/usr/bin/env python3
"""
paperb_benchmarks.py — sector / equal-weight / no-reinvest benchmarks for Paper B
================================================================================

Answers the "is it just tech beta?" question (URTC readiness §3 gap 1) using only
data already stored in the run JSONs. **No re-runs and no model calls** — each run
records a `daily_value_series`, so every benchmark is a post-hoc price lookup over
that run's own window.

Benchmarks, all adjusted-close (total-return) basis, each over the run's window:

    SPY   broad market
    XLK   S&P 500 Information Technology
    QQQ   Nasdaq-100
    EW    equal-weighted portfolio of the 29-name reference universe
    INIT  the run's own no-reinvest baseline (hold the period-0 picks)

Caveat on EW, worth repeating wherever these numbers are used: the 29-name
universe was assembled with hindsight, so equal-weighting it is a
survivorship-inflated benchmark. `CLAUDE.md` designates it reference-only. The
clean null for selection skill is `selection_null.py`, which draws from each
period's actual candidate pool.

Usage
-----
    python paperb_benchmarks.py
    python paperb_benchmarks.py --arms feedback_on,hold_winners
    python paperb_benchmarks.py --out paperb_benchmarks.txt

Requires local network access (yfinance). Per CLAUDE.md the cloud sandbox is
firewalled from Yahoo Finance; run this locally.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "portfolio_sim")
DEFAULT_ARMS = ["feedback_on", "feedback_off", "hold_winners"]
INDICES = ["SPY", "XLK", "QQQ"]
LABELS = ["SPY", "XLK", "QQQ", "EW", "INIT"]

sys.path.insert(0, ROOT)
try:
    import sector_config
    UNIVERSE = sorted(sector_config.universe())
except Exception:                                          # pragma: no cover
    UNIVERSE = sorted({"GOOGL", "MSFT", "NVDA", "AAPL", "TSM", "CSCO", "AMD",
                       "TXN", "MCHP", "ASML", "MU", "AMZN", "META", "AVGO",
                       "QCOM", "INTC", "ORCL", "CRM", "ADBE", "NOW", "IBM",
                       "NFLX", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "ANET",
                       "PANW"})


def load_runs(arms):
    runs = []
    for arm in arms:
        d = os.path.join(RESULTS, arm)
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                j = json.load(open(f, encoding="utf8"))
            except Exception:
                continue
            dvs = j.get("daily_value_series") or []
            if len(dvs) < 2:
                continue
            md = j.get("metadata", {}) or {}
            nre = j.get("no_reinvest_baseline") or {}
            runs.append(dict(
                arm=arm, file=os.path.basename(f),
                start=dvs[0][0], end=dvs[-1][0],
                roi=j.get("total_roi_pct"),
                INIT=nre.get("total_roi_pct"),
                freq=md.get("freq"),
                window=str(md.get("analysis_start_date"))[:7],
            ))
    return runs


def fetch_prices(tickers, start):
    import yfinance as yf
    d = yf.download(list(tickers), start=start, end="2026-12-31",
                    auto_adjust=True, progress=False, threads=True)
    px = d["Close"] if "Close" in d.columns.get_level_values(0) else d
    px = px.dropna(axis=1, how="all")
    return px.loc[:, ~px.columns.duplicated()]


def _total_return(px, ticker, s, e):
    sl = px.loc[(px.index >= s) & (px.index <= e)]
    if ticker not in sl.columns or len(sl) < 2:
        return None
    c = np.asarray(sl[ticker], dtype=float)
    c = c[np.isfinite(c)]
    if c.size < 2 or c[0] <= 0:
        return None
    return (c[-1] / c[0] - 1.0) * 100.0


def _equal_weight(px, s, e):
    rs = [_total_return(px, t, s, e) for t in UNIVERSE]
    rs = [r for r in rs if r is not None]
    return statistics.mean(rs) if rs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated result subfolders")
    ap.add_argument("--out", default="paperb_benchmarks.txt")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    runs = load_runs(arms)
    if not runs:
        print("no runs found under %s for arms %s" % (RESULTS, arms))
        return 1
    lo = min(r["start"] for r in runs)
    print("%d runs, %d arms, span %s .. %s"
          % (len(runs), len(arms), lo, max(r["end"] for r in runs)))
    print("universe: %d names" % len(UNIVERSE))

    print("fetching prices ...")
    px = fetch_prices(INDICES + UNIVERSE, lo)
    print("  %d tickers x %d trading days" % (len(px.columns), len(px)))

    cache = {}
    for r in runs:
        key = (r["start"], r["end"])
        if key not in cache:
            cache[key] = {b: _total_return(px, b, *key) for b in INDICES}
            cache[key]["EW"] = _equal_weight(px, *key)
        r.update(cache[key])

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("PAPER B BENCHMARKS — strategy vs sector, equal-weight and no-reinvest")
    emit("=" * 78)
    emit()
    emit("Excess return in percentage points, per run; 'wins' counts runs ahead of")
    emit("the benchmark. EW is survivorship-inflated (see module docstring).")
    emit()
    emit("%-14s %-6s %s" % ("arm", "n", "  ".join("%-21s" % b for b in LABELS)))
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        cells = []
        for b in LABELS:
            d = [r["roi"] - r[b] for r in rs
                 if isinstance(r.get(b), (int, float)) and isinstance(r["roi"], (int, float))]
            if not d:
                cells.append("%-21s" % "n/a")
                continue
            cells.append("%-21s" % ("%+6.2fpp %3d/%3d"
                                    % (statistics.mean(d), sum(1 for x in d if x > 0), len(d))))
        emit("%-14s %-6d %s" % (arm, len(rs), "  ".join(cells)))

    emit()
    emit("Paired by cell (window x cadence), cell means:")
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        bycell = collections.defaultdict(list)
        for r in rs:
            bycell[(r["window"], r["freq"])].append(r)
        emit()
        emit("  -- %s (%d cells)" % (arm, len(bycell)))
        for b in LABELS:
            diffs = []
            for _, grp in sorted(bycell.items()):
                vals = [g["roi"] - g[b] for g in grp
                        if isinstance(g.get(b), (int, float)) and isinstance(g["roi"], (int, float))]
                if vals:
                    diffs.append(statistics.mean(vals))
            if not diffs:
                continue
            line = ("     vs %-5s mean %+6.2fpp  wins %2d/%2d"
                    % (b, statistics.mean(diffs), sum(1 for x in diffs if x > 0), len(diffs)))
            try:
                from scipy.stats import wilcoxon
                line += "  Wilcoxon p = %.4f" % wilcoxon(diffs).pvalue
            except Exception:
                pass
            emit(line)

    with open(os.path.join(ROOT, args.out), "w", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")
    emit()
    emit("Wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
