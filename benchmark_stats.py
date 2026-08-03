#!/usr/bin/env python3
"""
benchmark_stats.py — pooled sector-benchmark, random-portfolio, ablation, and
equal-weight analysis for the NHSJS revision (reviewer Major #9, Rec #3).

Run from the project root, AFTER revision_stats.py:

    python benchmark_stats.py

Unlike the sandbox draft (which used one representative run per cell), this scans
EVERY results/perm_*.json so the benchmark numbers rest on the same 2,759-run
pooled set as the win rates and risk metrics.

For each 12-month run it reconstructs the portfolio from the stored per-stock
window returns and LLM allocations, then computes, per model x configuration:
  - beat rate vs the S&P 500 (from each run's benchmark_comparison),
  - beat rate vs XLK (S&P 500 IT), SMH (semis), and an equal-weighted portfolio
    of the eligible universe, each per window,
  - percentile of the run's ROI within 4,000 random equal-weight portfolios drawn
    from the eligible universe and matched to the run's holding count,
  - mean ROI with the single largest contributor removed, and with all
    semiconductor holdings removed (weights renormalized),
  - equal-weight rescoring of the same picks.

ETF window returns come from the repo price caches
(FINANCIAL_REPORTS/XLK/PRICE_HISTORY_5Y.csv, .../SMH/...); refresh those caches
first if you need windows starting after the cache end date.

Output: results/benchmark_stats.xlsx
Requires: pandas, numpy, scipy (same as revision_stats.py).
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st

from permutation_runner import (
    RESULTS_DIR, ST_SKIPPED, build_permutations, _filter_label,
    _is_valid_result, _iter_saved_result_json_files, _match_perm_by_metadata,
)

ROOT = Path(RESULTS_DIR).resolve().parent
FR = ROOT / "FINANCIAL_REPORTS"
OUT = Path(RESULTS_DIR) / "benchmark_stats.xlsx"
N_RANDOM = 4000
RNG_SEED = 7

# The eligible large-cap tech/semi comparison universe (unfiltered baseline set).
UNIVERSE = {"GOOGL","MSFT","NVDA","AAPL","TSM","CSCO","AMD","TXN","MCHP","ASML","MU",
            "META","AMZN","CRM","AVGO","IBM","ZM","PLTR","QCOM","TSLA","INTC","LRCX",
            "ORCL","ADBE","NOW","SNPS","CDNS","KLAC","AMAT"}
SEMI = {"NVDA","TSM","AMD","TXN","MCHP","ASML","MU","AVGO","QCOM","INTC","LRCX",
        "SNPS","CDNS","KLAC","AMAT","ARM","ON","NXPI","ADI","MRVL","TER","STM",
        "GFS","SWKS","CRUS","MPWR","AMBA","ACLS","SMH"}


def _safe(v):
    try: return None if v is None else float(v)
    except (TypeError, ValueError): return None


def etf_window_return(series: pd.Series, start: str, months=12):
    s0 = pd.Timestamp(start); s1 = s0 + pd.DateOffset(months=months)
    a = series[series.index >= s0]; b = series[series.index <= s1]
    if a.empty or b.empty or b.index[-1] < s1 - pd.Timedelta(days=7):
        return None
    return 100.0 * (a.iloc[0] and b.iloc[-1] / a.iloc[0] - 1.0)


def load_etf(ticker):
    p = FR / ticker / "PRICE_HISTORY_5Y.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.set_index("Date")["Close"]


def load_runs():
    perms = build_permutations(); by_id = {p["id"]: p for p in perms}
    holds, runs = [], []
    for fp in _iter_saved_result_json_files():
        try:
            r = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        if not _is_valid_result(r):
            continue
        p = _match_perm_by_metadata(r, by_id, perms)
        if p is None or p["status"] == ST_SKIPPED or p["period"] != 12:
            continue
        ra = r.get("return_analysis", {}) or {}
        ir = ra.get("individual_returns") or {}
        if isinstance(ir, list):
            ir = {t: v for t, v in ir}
        recs = {x["ticker"]: float(x.get("final_allocation") or 0)
                for x in r.get("final_recommendations", [])}
        bc = ra.get("benchmark_comparison", {}) or {}
        key = dict(model=p["model"], temp=p["temp"], start=p["start"].strftime("%Y-%m"),
                   filter=_filter_label(p["init_n"], p["fin_n"]), perm=p["id"])
        runs.append({**key, "roi": _safe(ra.get("total_roi")),
                     "market": _safe(bc.get("market_return")),
                     "alpha": _safe(bc.get("alpha"))})
        for t in recs:
            if t in ir:
                holds.append({**key, "ticker": t, "w": recs[t], "ret": ir[t]})
    return pd.DataFrame(runs), pd.DataFrame(holds)


def main():
    runs, H = load_runs()
    print(f"pooled 12mo runs: {len(runs)}; holdings rows: {len(H)}")
    xlk, smh = load_etf("XLK"), load_etf("SMH")
    starts = sorted(H["start"].unique())
    bench = pd.DataFrame({"start": starts})
    bench["XLK"] = [etf_window_return(xlk, s) if xlk is not None else None for s in starts]
    bench["SMH"] = [etf_window_return(smh, s) if smh is not None else None for s in starts]

    # universe equal-weight per window (median ticker return across runs)
    tw = H.groupby(["start", "ticker"])["ret"].median().reset_index()
    uni = tw[tw.ticker.isin(UNIVERSE)].groupby("start")["ret"].mean().rename("UNIV_EW").reset_index()

    key = ["model", "temp", "start", "filter"]
    abl = []
    for k, g in H.groupby(key):
        w, r, t = g.w.values, g.ret.values, g.ticker.values
        wn = w / w.sum()
        i = (wn * r).argmax(); m = np.ones(len(w), bool); m[i] = False
        m2 = ~np.isin(t, list(SEMI))
        abl.append(dict(zip(key, k)) | {
            "roi": float((wn * r).sum()), "equal_weight": float(r.mean()),
            "top_removed": float((w[m] / w[m].sum() * r[m]).sum()) if m.sum() else np.nan,
            "semi_excluded": float((w[m2] / w[m2].sum() * r[m2]).sum()) if m2.sum() else np.nan,
            "n_hold": len(w)})
    A = pd.DataFrame(abl).merge(runs[key + ["roi", "market"]].rename(columns={"roi": "roi_run"}), on=key)
    A = A.merge(bench, on="start", how="left").merge(uni, on="start", how="left")

    # beat rates
    beat = []
    for (mdl, f), g in A.groupby(["model", "filter"]):
        row = {"model": mdl, "filter": f, "n": len(g)}
        for b, col in [("S&P", "market"), ("XLK", "XLK"), ("SMH", "SMH"), ("UnivEW", "UNIV_EW")]:
            gg = g.dropna(subset=[col])
            if len(gg):
                wins = int((gg.roi > gg[col]).sum())
                ci = st.binomtest(wins, len(gg)).proportion_ci(0.95, "exact")
                row[f"beat_{b}_%"] = round(100 * wins / len(gg), 1)
                row[f"beat_{b}_CI"] = f"{100*ci.low:.0f}-{100*ci.high:.0f}"
                row[f"n_{b}"] = len(gg)
        beat.append(row)
    B = pd.DataFrame(beat)

    # random-portfolio percentiles
    rng = np.random.default_rng(RNG_SEED)
    uv = {s: g[g.ticker.isin(UNIVERSE)].set_index("ticker")["ret"] for s, g in tw.groupby("start")}
    pct = []
    for _, r in A.iterrows():
        v = uv.get(r.start)
        if v is None or len(v) < 10:
            continue
        k = int(min(r.n_hold, len(v)))
        draws = rng.choice(v.values, (N_RANDOM, k)).mean(axis=1)
        pct.append({"model": r.model, "filter": r["filter"], "percentile": 100 * (draws < r.roi).mean()})
    Pc = pd.DataFrame(pct).groupby(["model", "filter"])["percentile"].median().round(1).reset_index()

    cfg = A.groupby(["model", "filter"])[["roi", "equal_weight", "top_removed", "semi_excluded"]].mean().round(2).reset_index()

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        B.to_excel(xw, "Beat rates", index=False)
        Pc.to_excel(xw, "Random percentiles", index=False)
        cfg.to_excel(xw, "Ablations & equal-weight", index=False)
        bench.round(2).to_excel(xw, "Sector bench windows", index=False)
    print(f"wrote {OUT}")
    print(B[B.model == "gpt-5.1"].to_string(index=False))


if __name__ == "__main__":
    main()
