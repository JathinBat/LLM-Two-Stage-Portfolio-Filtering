#!/usr/bin/env python3
"""
Post-hoc reanalysis of the EXISTING portfolio_sim runs — no new model calls, no
internet. Answers two reviewer points on data already in hand:

  B1  Transaction-cost sensitivity: charges bps on the round-trip notional
      (sell-all at each rebalance + new buys, reconstructed from each period's
      value_before / invested_after) and re-ranks cadences at 0/5/10/25 bps.
  B3  Risk-free sensitivity: recomputes the strategy's mean Sharpe at rf = 0/2/4/5%
      from the stored daily value series (the runs assumed rf = 0, which inflates it).

Run from the repo root:   python reanalysis_costs_rf.py
Writes: reanalysis_costs_rf.txt (+ prints to console).
"""
import os, glob, json, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "portfolio_sim")
BPS = [0, 5, 10, 25]
RFS = [0.0, 0.02, 0.04, 0.05]

def load():
    runs = []
    for f in glob.glob(os.path.join(RESULTS, "psperm_*.json")):
        d = json.load(open(f)); m = d["metadata"]; P = d["periods"]
        traded = []
        for i, p in enumerate(P):
            vb = p.get("value_before", 0.0); inv = p.get("invested_after", 0.0)
            sold = 0.0 if i == 0 else vb          # nothing to sell on the initial buy
            traded.append((sold + inv, vb if vb > 0 else 10000.0))
        runs.append(dict(nreb=d["num_rebalances"], filt=f"{m['filter_X']}-{m['filter_Y']}",
                         start=m["analysis_start_date"], roi=d["total_roi_pct"],
                         spy=d["benchmark_risk_metrics"]["total_return"] * 100,
                         dv=[x[1] for x in d["daily_value_series"]], traded=traded))
    return runs

def net_roi(r, bps):
    drag = 1.0
    for notional, vat in r["traded"]:
        drag *= (1 - (bps / 10000.0) * (notional / vat if vat > 0 else 0))
    return (1 + r["roi"] / 100.0) * drag * 100 - 100

def sharpe(vals, rf):
    v = np.asarray(vals, float); r = v[1:] / v[:-1] - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    return (r.mean() * 252 - rf) / vol if vol > 0 else float("nan")

def main():
    runs = load()
    if not runs:
        raise SystemExit(f"no runs in {RESULTS}")
    out = [f"Post-hoc reanalysis of {len(runs)} runs (no new model calls).\n"]

    out.append("=== B1  TRANSACTION-COST SENSITIVITY — mean ROI / mean vs-SPY by cadence ===")
    out.append(f"{'rebalances':>11} {'n':>3} | " + " | ".join(f"{b:>2}bps" for b in BPS))
    for nr in sorted(set(r["nreb"] for r in runs)):
        sub = [r for r in runs if r["nreb"] == nr]
        cells = [f"{st.mean([net_roi(r,b) for r in sub]):5.1f}/{st.mean([net_roi(r,b)-r['spy'] for r in sub]):+5.1f}"
                 for b in BPS]
        out.append(f"{nr:>11} {len(sub):>3} | " + " | ".join(cells))
    out.append("  (cell = net ROI% / net excess-vs-SPY pp)\n")

    out.append("=== B1  cost DRAG (gross ROI − net ROI, pp) — high-cadence pays the most ===")
    for nr in sorted(set(r["nreb"] for r in runs)):
        sub = [r for r in runs if r["nreb"] == nr]
        out.append("  " + f"{nr:>2} reb: " +
                   "  ".join(f"{b}bps -{max(0.0, st.mean([r['roi']-net_roi(r,b) for r in sub])):.1f}pp" for b in BPS))
    out.append("")

    out.append("=== B1  within 20->10 (de-confounded) — net excess vs SPY by cadence ===")
    c = [r for r in runs if r["filt"] == "20-10"]
    for nr in sorted(set(r["nreb"] for r in c)):
        sub = [r for r in c if r["nreb"] == nr]
        out.append("  " + f"{nr:>2} reb: " +
                   "  ".join(f"{b}bps {st.mean([net_roi(r,b)-r['spy'] for r in sub]):+5.1f}" for b in BPS))
    out.append("")

    out.append("=== B3  RISK-FREE SENSITIVITY — mean strategy Sharpe ===")
    for rf in RFS:
        out.append(f"  rf={rf*100:>2.0f}%: {np.mean([sharpe(r['dv'], rf) for r in runs]):.2f}")
    out.append("  (runs assumed rf=0; apply the same rf to the benchmark when reporting.)")

    report = "\n".join(out)
    open(os.path.join(HERE, "reanalysis_costs_rf.txt"), "w").write(report)
    print(report)
    print("\nWrote reanalysis_costs_rf.txt")

if __name__ == "__main__":
    main()
