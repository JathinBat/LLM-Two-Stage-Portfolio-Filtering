#!/usr/bin/env python3
"""
Tier 1A baseline analysis for the reinvestment (Paper B) study.

For every psperm_*.json run in results/portfolio_sim/, this compares the LLM
rebalancing strategy against FAIR baselines over the same window, on a
total-return (adjusted-close) basis, and reports paired statistics.

Baselines (all buy-and-hold over the run's window, $10k start):
  - XLK   : S&P 500 Information Technology ETF (the correct sector benchmark)
  - QQQ   : Nasdaq-100
  - SPY   : broad market (for reference / beta)
  - SMH   : semiconductors (optional context)
  - EWU   : Equal-Weight of the 29-name tech Universe
  - CWU   : Cap-Weight of the universe (uses yfinance market caps; skipped if unavailable)
  - INIT  : buy-and-hold the LLM's OWN period-0 picks, no rebalancing
            (isolates the value of *rebalancing* specifically)
  - RAND  : 1,000 random equal-weight portfolios from the universe matched to
            the run's holding count -> gives the strategy a skill percentile

Outputs (written next to this script):
  - tier1a_per_run.csv       one row per run: strategy vs every baseline
  - tier1a_window_summary.csv per-window strategy(mean) vs each baseline
  - tier1a_report.txt        printed headline: win rates, paired tests, CIs
  - console summary

Requires: yfinance, pandas, numpy, scipy   (pip install -r requirements.txt)
Run from the repo root:   python tier1a_baselines.py
Needs internet (pulls prices from Yahoo). The on-disk FINANCIAL_REPORTS caches
are stale (they end mid-2024) and do NOT cover the 2024-2026 windows, so this
script deliberately fetches fresh prices instead of using them.
"""

import os, glob, json, sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "portfolio_sim")
TRADING_DAYS = 252
RF_ANNUAL = 0.0
N_RANDOM = 1000
RANDOM_SEED = 20260723   # fixed so results are reproducible

# CAVEAT — this is NOT the strategy's opportunity set.
# The live strategy is news-driven and unbounded: across the 90 runs it held
# 36 distinct tickers, 10 of them OUTSIDE this list (CRWD, SNOW, DDOG, PANW,
# NFLX, MSTR, MSI, ACN, TMUS, GOOG). sector_config.universe() is used only by
# the mock/dry-run backend and by v1's unfiltered baseline. So the equal-weight
# and random baselines built from this list are a *fixed big-tech reference
# basket*, NOT the set the model chose from — read them as context, not as a
# skill-vs-luck test. The clean selection null (random draw from each period's
# actual candidate pool) requires re-running with candidate logging; the pools
# are not stored in the result JSONs.
BIGTECH_BASKET = ["GOOGL","MSFT","NVDA","AAPL","TSM","CSCO","AMD","TXN","MCHP","ASML",
                  "MU","META","AMZN","CRM","AVGO","IBM","ZM","PLTR","QCOM","TSLA",
                  "INTC","LRCX","ORCL","ADBE","NOW","SNPS","CDNS","KLAC","AMAT"]
UNIVERSE = BIGTECH_BASKET   # alias kept for the reference-basket baselines below
ETFS = ["XLK","QQQ","SPY","SMH"]   # universe-free primary baselines

# ----------------------------------------------------------------------------- #
# Metrics (mirrors risk_metrics.py conventions: 252d annualization, rf=0)
# ----------------------------------------------------------------------------- #
def metrics_from_values(dates, values, bench_ret=None):
    """values: 1-D array of daily portfolio value. bench_ret: aligned daily
    benchmark returns for beta/alpha (optional)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {}
    r = v[1:] / v[:-1] - 1.0
    total = v[-1] / v[0] - 1.0
    ann = (v[-1] / v[0]) ** (TRADING_DAYS / (len(v) - 1)) - 1.0
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - RF_ANNUAL) / vol if vol > 0 else np.nan
    downside = r[r < 0]
    dvol = downside.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    sortino = (r.mean() * TRADING_DAYS) / dvol if dvol and dvol > 0 else np.nan
    cum = v / v[0]
    peak = np.maximum.accumulate(cum)
    maxdd = (cum / peak - 1.0).min()
    beta = alpha = np.nan
    if bench_ret is not None and len(bench_ret) == len(r):
        b = np.asarray(bench_ret, float)
        m = np.isfinite(b) & np.isfinite(r)
        if m.sum() > 2 and b[m].var() > 0:
            beta = np.cov(r[m], b[m])[0, 1] / b[m].var()
            alpha = (r[m] - beta * b[m]).mean() * TRADING_DAYS
    return dict(total_return=total*100, ann_return=ann*100, vol=vol*100,
                sharpe=sharpe, sortino=sortino, maxdd=maxdd*100, beta=beta,
                alpha=alpha*100 if np.isfinite(alpha) else np.nan)

# ----------------------------------------------------------------------------- #
# Price loading
# ----------------------------------------------------------------------------- #
def fetch_prices(tickers, start, end):
    import yfinance as yf
    # auto_adjust=True -> adjusted close (splits + dividends folded in = total return)
    df = yf.download(sorted(set(tickers)), start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")

def window_slice(prices, start, end):
    """Trading-day rows in [start, end], forward/back filled for small gaps."""
    idx = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    w = prices.loc[idx].copy()
    return w.ffill().bfill()

def bh_value(prices_w, weights, budget=10000.0):
    """Buy-and-hold value series for initial weights (dict ticker->weight)."""
    cols = [t for t in weights if t in prices_w.columns and prices_w[t].notna().any()]
    if not cols:
        return None
    w = np.array([weights[t] for t in cols], float); w = w / w.sum()
    p = prices_w[cols].to_numpy(float)
    rel = p / p[0]                         # normalized to 1 at window start
    val = budget * (rel * w).sum(axis=1)
    return pd.Series(val, index=prices_w.index)

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "psperm_*.json")))
    if not files:
        sys.exit(f"No psperm_*.json found in {RESULTS_DIR}")
    runs = []
    for f in files:
        d = json.load(open(f))
        m = d["metadata"]
        runs.append(dict(
            file=os.path.basename(f), start=m["analysis_start_date"],
            end=m["analysis_end_date"], nreb=d["num_rebalances"], freq=m["freq"],
            filt=f"{m['filter_X']}-{m['filter_Y']}", roi=d["total_roi_pct"],
            hold_cap=m["filter_Y"],
            init_holdings=d["periods"][0]["holdings"],  # ticker -> {shares,avg_cost,price}
            spy_ret=d["benchmark_risk_metrics"]["total_return"]*100,
        ))
    span_start = min(r["start"] for r in runs)
    span_end   = (pd.Timestamp(max(r["end"] for r in runs)) + pd.DateOffset(days=5)).strftime("%Y-%m-%d")
    print(f"{len(runs)} runs | span {span_start} -> {span_end} | fetching prices...")

    prices = fetch_prices(UNIVERSE + ETFS, span_start, span_end)
    missing = [t for t in UNIVERSE + ETFS if t not in prices.columns]
    if missing:
        print(f"  WARNING: no price data for {missing} (baselines will skip them)")

    # optional cap weights from current market cap (approximation; documented as such)
    capw = {}
    try:
        import yfinance as yf
        for t in UNIVERSE:
            try:
                mc = yf.Ticker(t).fast_info.get("market_cap")
                if mc: capw[t] = float(mc)
            except Exception:
                pass
    except Exception:
        pass

    rng = np.random.default_rng(RANDOM_SEED)
    windows = sorted(set(r["start"] for r in runs))
    win_baselines = {}   # start -> {baseline_name: metrics dict}
    for w0 in windows:
        w1 = next(r["end"] for r in runs if r["start"] == w0)
        pw = window_slice(prices, w0, w1)
        if len(pw) < 5:
            print(f"  window {w0}: insufficient price rows, skipping"); continue
        spy_ret_series = pw["SPY"].pct_change().dropna().to_numpy() if "SPY" in pw else None
        base = {}
        # single-ETF baselines
        for etf in ETFS:
            if etf in pw.columns:
                base[etf] = metrics_from_values(pw.index, bh_value(pw, {etf:1.0}),
                                                bench_ret=spy_ret_series)
        # equal-weight universe
        uni = [t for t in UNIVERSE if t in pw.columns]
        base["EWU"] = metrics_from_values(pw.index, bh_value(pw, {t:1.0 for t in uni}),
                                          bench_ret=spy_ret_series)
        # cap-weight universe (if caps available)
        if capw:
            cw = {t: capw[t] for t in uni if t in capw}
            if cw:
                base["CWU"] = metrics_from_values(pw.index, bh_value(pw, cw),
                                                  bench_ret=spy_ret_series)
        # random portfolios: store the ROI distribution by holding count
        base["_RAND"] = {}
        for k in sorted(set(r["hold_cap"] for r in runs)):
            rois = []
            for _ in range(N_RANDOM):
                pick = list(rng.choice(uni, size=min(k, len(uni)), replace=False))
                s = bh_value(pw, {t:1.0 for t in pick})
                if s is not None: rois.append(s.iloc[-1]/s.iloc[0]*100 - 100)
            base["_RAND"][k] = np.array(rois)
        win_baselines[w0] = base
        print(f"  window {w0}: XLK {base.get('XLK',{}).get('total_return',float('nan')):+.1f}%  "
              f"QQQ {base.get('QQQ',{}).get('total_return',float('nan')):+.1f}%  "
              f"EWU {base['EWU']['total_return']:+.1f}%")

    # per-run comparison
    rows = []
    for r in runs:
        b = win_baselines.get(r["start"])
        if not b: continue
        pw = window_slice(prices, r["start"], r["end"])
        # INIT = buy-and-hold the run's own period-0 picks (dollar weights)
        w0 = {}
        for t, h in r["init_holdings"].items():
            dollars = h.get("shares",0)*h.get("price",0)
            if dollars>0 and t in pw.columns: w0[t]=dollars
        init_roi = None
        if w0:
            s = bh_value(pw, w0)
            if s is not None: init_roi = s.iloc[-1]/s.iloc[0]*100 - 100
        row = dict(file=r["file"], window=r["start"], cadence_reb=r["nreb"],
                   config=r["filt"], strat_roi=r["roi"], spy_roi=r["spy_ret"])
        for name in ["XLK","QQQ","SMH","EWU","CWU"]:
            if name in b and b[name]:
                row[f"{name}_roi"] = b[name]["total_return"]
                row[f"vs_{name}"]  = r["roi"] - b[name]["total_return"]
        row["INIT_roi"] = init_roi
        row["vs_INIT"]  = (r["roi"] - init_roi) if init_roi is not None else None
        rd = b["_RAND"].get(r["hold_cap"])
        if rd is not None and len(rd):
            row["rand_pctile"] = float((rd < r["roi"]).mean()*100)   # strategy percentile
            row["rand_median_roi"] = float(np.median(rd))
        rows.append(row)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(HERE,"tier1a_per_run.csv"), index=False)

    # ---- headline stats ----
    from scipy.stats import wilcoxon, binomtest
    def sign_test(diffs):
        diffs=[d for d in diffs if d is not None and np.isfinite(d)]
        pos=sum(1 for d in diffs if d>0); n=len(diffs)
        p=binomtest(pos,n,0.5).pvalue if n else float('nan')
        return pos,n,p
    lines=[]
    lines.append("TIER 1A — strategy vs baselines (total-return basis)\n")
    lines.append(f"{len(per_run)} runs across {per_run['window'].nunique()} windows, "
                 f"gpt-5.1 technology.\n")
    lines.append("NOTE ON UNIVERSE: the strategy is news-driven and unbounded (it held 10 names\n"
                 "outside the 29-name basket). SPY/XLK/QQQ/SMH and INIT are the real baselines —\n"
                 "they need no universe. EWU/CWU/random are a FIXED big-tech reference basket, not\n"
                 "the strategy's opportunity set; treat them as context, not a skill test. The clean\n"
                 "selection null needs a re-run that logs each period's candidate pool.\n")
    lines.append("PRIMARY (universe-free) — per-run win rate + mean excess:\n")
    for name in ["SPY","XLK","QQQ","SMH","INIT"]:
        col=f"vs_{name}"
        if col not in per_run: continue
        d=per_run[col].dropna()
        if not len(d): continue
        wins=(d>0).sum(); n=len(d)
        ci=binomtest(int(wins),int(n),0.5).proportion_ci(confidence_level=0.95)
        tag = "  (beats own initial picks = rebalancing adds value)" if name=="INIT" else ""
        lines.append(f"  vs {name:4}: beat {wins:3d}/{n:3d}  ({100*wins/n:5.1f}%, "
                     f"95% CI {100*ci.low:.0f}-{100*ci.high:.0f}%)  mean excess {d.mean():+6.1f}pp  median {d.median():+6.1f}pp{tag}")
    lines.append("\nREFERENCE ONLY (fixed big-tech basket — NOT the opportunity set):\n")
    for name in ["EWU","CWU"]:
        col=f"vs_{name}"
        if col not in per_run: continue
        d=per_run[col].dropna()
        if not len(d): continue
        wins=(d>0).sum(); n=len(d)
        lines.append(f"  vs {name:4}: beat {wins:3d}/{n:3d}  ({100*wins/n:5.1f}%)  mean excess {d.mean():+6.1f}pp   [reference basket]")
    # window-level paired tests using the 20->10 config mean per window
    lines.append("\nWINDOW-LEVEL paired tests (20->10 config, n=windows) — the honest unit:\n")
    sub=per_run[per_run["config"]=="20-10"]
    for name in ["XLK","QQQ","INIT"]:
        col=f"vs_{name}"
        if col not in sub: continue
        per_win=sub.groupby("window")[col].mean().dropna()
        if len(per_win)<3: continue
        pos,n,sp=sign_test(list(per_win))
        try: wp=wilcoxon(per_win).pvalue
        except Exception: wp=float('nan')
        ci=binomtest(pos,n,0.5).proportion_ci(confidence_level=0.95)
        lines.append(f"  vs {name:4}: strategy wins {pos}/{n} windows "
                     f"(95% CI {100*ci.low:.0f}-{100*ci.high:.0f}%)  "
                     f"mean {per_win.mean():+.1f}pp  sign p={sp:.3f}  Wilcoxon p={wp:.3f}")
    # ---- per-cadence window-level paired tests (de-confounded within 20->10) ----
    CAD = {1:"semiannual (1 reb)", 3:"quarterly (3 reb)", 5:"bimonthly (5 reb)", 11:"monthly (11 reb)"}
    c2010 = per_run[per_run["config"]=="20-10"]
    lines.append("\nPER-CADENCE window-level paired tests (config 20->10, n=9 windows) — "
                 "does rebalance frequency change the sector verdict?\n")
    for nreb in [1,3,5,11]:
        cc = c2010[c2010["cadence_reb"]==nreb]
        if not len(cc): continue
        lines.append(f"  {CAD[nreb]}:")
        for name in ["XLK","QQQ","INIT"]:
            col=f"vs_{name}"
            if col not in cc: continue
            per_win = cc.groupby("window")[col].mean().dropna()
            if len(per_win)<3: continue
            pos,n,sp = sign_test(list(per_win))
            try: wp=wilcoxon(per_win).pvalue
            except Exception: wp=float('nan')
            ci=binomtest(pos,n,0.5).proportion_ci(confidence_level=0.95)
            lines.append(f"      vs {name:4}: wins {pos}/{n} (95% CI {100*ci.low:.0f}-{100*ci.high:.0f}%)  "
                         f"mean {per_win.mean():+.1f}pp  sign p={sp:.3f}  Wilcoxon p={wp:.3f}")

    # ---- frequency contrast: monthly vs semiannual, paired by window (20->10) ----
    lines.append("\nFREQUENCY CONTRAST (config 20->10): monthly (11 reb) minus semiannual (1 reb), paired by window:\n")
    piv = c2010.pivot_table(index="window", columns="cadence_reb", values="strat_roi", aggfunc="mean")
    if 11 in piv.columns and 1 in piv.columns:
        diff = (piv[11]-piv[1]).dropna()
        pos,n,sp = sign_test(list(diff))
        try: wp=wilcoxon(diff).pvalue
        except Exception: wp=float('nan')
        verdict = "frequency helps" if (np.isfinite(sp) and sp<0.05 and diff.mean()>0) else "no frequency benefit"
        lines.append(f"  monthly beats semiannual in {pos}/{n} windows  "
                     f"mean {diff.mean():+.1f}pp  sign p={sp:.3f}  Wilcoxon p={wp:.3f}  -> {verdict}")
    else:
        lines.append("  (need both monthly and semiannual runs at 20->10 — missing)")

    if "rand_pctile" in per_run:
        lines.append(f"\nRandom-basket percentile (REFERENCE ONLY — random draw from the fixed "
                     f"big-tech basket, not the strategy's real candidate pool): "
                     f"mean {per_run['rand_pctile'].mean():.0f}th, median {per_run['rand_pctile'].median():.0f}th. "
                     f"A true selection null requires logging the news filter's per-period candidates.")
    lines.append("\nDISCLOSURES (Tier 1D):")
    lines.append("  - Window non-independence: the 9 windows are monthly-staggered 12-month holds "
                 "overlapping ~92%; treated as n=9 effective units, not 90 runs.")
    lines.append("  - Replication depth: 1 run per (window x config x cadence) cell; the plan "
                 "targets >=3 runs/cell, so within-cell SD is not yet estimable.")
    lines.append("  - Return basis: strategy and all baselines are total-return (auto-adjusted close).")
    report="\n".join(lines)
    open(os.path.join(HERE,"tier1a_report.txt"),"w").write(report)
    print("\n"+report)
    # window summary csv
    ws=[]
    for w0,b in win_baselines.items():
        row={"window":w0}
        for name in ["SPY","XLK","QQQ","SMH","EWU","CWU"]:
            if name in b and b[name]: row[f"{name}_roi"]=round(b[name]["total_return"],2)
        sub=per_run[per_run["window"]==w0]
        row["strat_mean_roi"]=round(sub["strat_roi"].mean(),2)
        row["strat_2010_roi"]=round(sub[sub["config"]=="20-10"]["strat_roi"].mean(),2)
        ws.append(row)
    pd.DataFrame(ws).to_csv(os.path.join(HERE,"tier1a_window_summary.csv"), index=False)
    print("\nWrote: tier1a_per_run.csv, tier1a_window_summary.csv, tier1a_report.txt")

if __name__ == "__main__":
    main()
