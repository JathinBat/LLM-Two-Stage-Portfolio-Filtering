#!/usr/bin/env python3
"""
Analyze the feedback ON vs OFF experiment (review gap A1) once run_feedback_control.py
has produced results/portfolio_sim/feedback_on/ and .../feedback_off/.

Reports:
  1. Paired ON-vs-OFF ROI and Sharpe by window (sign test + Wilcoxon + binomial CI)
     — the direct test of whether self-reflection helps.
  2. Behavioral differences: turnover, re-prompts, cash drag by arm.
  3. Winner-chasing: does the model retain recent winners more WITH feedback than
     without? Uses the per-name `prior_holdings_perf` now logged each period.

If feedback_on/ is empty it falls back to the existing 20->10 runs in the main
results dir as the ON arm (less clean, but usable).

Run from the repo root:  python analyze_feedback.py
"""
import os, glob, json, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.join(HERE, "results", "portfolio_sim")

def sharpe(vals, rf):
    v = np.asarray(vals, float); r = v[1:] / v[:-1] - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    return (r.mean() * 252 - rf) / vol if vol > 0 else float("nan")

def load_dir(d, arm):
    out = []
    for f in glob.glob(os.path.join(d, "*.json")):
        try: dd = json.load(open(f))
        except Exception: continue
        run = dd.get("run", dd)            # tolerate {"run": ...} wrappers
        m = run.get("metadata", {})
        P = run.get("periods", [])
        hsets = [set(p.get("holdings", {}).keys()) for p in P]
        turns = [1 - len(a & b) / len(a | b) for a, b in zip(hsets, hsets[1:]) if (a | b)]
        reprompts = sum(max(0, p.get("decision_attempts", 1) - 1) for p in P)
        dv = [x[1] for x in run.get("daily_value_series", [])]
        out.append(dict(
            arm=arm, window=m.get("analysis_start_date"), freq=m.get("freq"),
            config=f"{m.get('filter_X')}-{m.get('filter_Y')}",
            roi=run.get("total_roi_pct"),
            vs_init=run.get("vs_no_reinvest_pp"),   # strategy - no-rebalance counterfactual
            sharpe0=sharpe(dv, 0.0) if len(dv) > 2 else float("nan"),
            sharpe4=sharpe(dv, 0.04) if len(dv) > 2 else float("nan"),
            turnover=st.mean(turns) if turns else 0.0,
            reprompts=reprompts,
            avgcash=st.mean([p.get("cash_after", 0)/max(p.get("value_after", 1), 1) for p in P]) if P else 0,
            periods=P))
    return out

def winner_chasing(runs):
    """mean prior-period return of RETAINED vs DROPPED names (needs prior_holdings_perf)."""
    ret_r, ret_d = [], []
    for r in runs:
        P = r["periods"]
        for i in range(1, len(P)):
            prev_perf = P[i].get("prior_holdings_perf", {})   # names held during period i-1 + their return
            now_held = set(P[i].get("holdings", {}).keys())   # what got re-bought at period i
            for t, d in prev_perf.items():
                v = d.get("ret_since_prev_pct")
                if v is None: continue
                (ret_r if t in now_held else ret_d).append(v)
    return ret_r, ret_d

def paired(on, off, key):
    """pair ON vs OFF by (window, cadence), mean per cell per arm.

    Pairing on window ALONE would average 2x and 12x runs of the same window into
    one number. The two cadences behave very differently (per-decision rebalance
    alpha is -5.7pp at 2x vs +0.1pp at 12x), so blending them cancels real signal
    and halves the number of paired units. The grid is 9 windows x 2 cadences =
    18 cells; each cell is the unit of pairing.
    """
    from collections import defaultdict
    A, B = defaultdict(list), defaultdict(list)
    for r in on:  A[(r["window"], r["freq"])].append(r[key])
    for r in off: B[(r["window"], r["freq"])].append(r[key])
    diffs = []
    for cell in sorted(set(A) & set(B)):
        a = [x for x in A[cell] if x is not None and np.isfinite(x)]
        b = [x for x in B[cell] if x is not None and np.isfinite(x)]
        if a and b:
            label = f"{cell[0]} {cell[1]}x"
            diffs.append((label, st.mean(a) - st.mean(b), st.mean(a), st.mean(b)))
    return diffs

def main():
    on = load_dir(os.path.join(PS, "feedback_on"), "on")
    off = load_dir(os.path.join(PS, "feedback_off"), "off")
    if not off:
        raise SystemExit("No feedback_off runs found. Run the control arm via the "
                         "permutation runner:\n"
                         "  python portfolio_sim_permutation_runner.py "
                         "--preset presets/fresh_news_feedback_off.json\n"
                         "(run_feedback_control.py is deprecated and refuses to run.)")
    if not on:
        print("No feedback_on runs; falling back to existing 20->10 monthly runs as the ON arm.")
        for f in glob.glob(os.path.join(PS, "psperm_*.json")):
            dd = json.load(open(f)); m = dd.get("metadata", {})
            if f"{m.get('filter_X')}-{m.get('filter_Y')}" == "20-10" and m.get("freq") == off[0]["freq"]:
                one = load_dir(os.path.dirname(f), "on")  # not efficient but simple
                break
        on = [r for r in load_dir(PS, "on")
              if r["config"] == "20-10" and r["freq"] == off[0]["freq"]]

    from scipy.stats import wilcoxon, binomtest
    out = [f"Feedback ON vs OFF — {len(on)} ON runs, {len(off)} OFF runs\n"]

    for key, label in [("roi", "ROI %"), ("sharpe4", "Sharpe (rf=4%)"),
                       ("vs_init", "vs no-reinvest (pp)")]:
        d = paired(on, off, key)
        if not d:
            out.append(f"{label}: no paired windows\n"); continue
        diffs = [x[1] for x in d]
        wins = sum(1 for x in diffs if x > 0); n = len(diffs)
        sp = binomtest(wins, n, 0.5).pvalue
        try: wp = wilcoxon([x[1] for x in d]).pvalue
        except Exception: wp = float("nan")
        ci = binomtest(wins, n, 0.5).proportion_ci(0.95)
        out.append(f"=== {label}: ON − OFF, paired by (window, cadence) (n={n}) ===")
        out.append(f"  ON wins {wins}/{n}  (95% CI {100*ci.low:.0f}-{100*ci.high:.0f}%)  "
                   f"mean diff {st.mean(diffs):+.2f}  sign p={sp:.3f}  Wilcoxon p={wp:.3f}")
        for w, dd, a, b in d:
            out.append(f"    {w}: ON {a:+.1f}  OFF {b:+.1f}  diff {dd:+.1f}")
        out.append("")

    # behavioral
    out.append("=== behavioral (mean by arm) ===")
    for arm, runs in [("ON", on), ("OFF", off)]:
        if not runs: continue
        out.append(f"  {arm}: turnover {st.mean([r['turnover'] for r in runs]):.2f}  "
                   f"reprompts {st.mean([r['reprompts'] for r in runs]):.1f}  "
                   f"cash {100*st.mean([r['avgcash'] for r in runs]):.1f}%")
    out.append("")

    # winner-chasing
    out.append("=== winner-chasing: mean prior-period return of RETAINED vs DROPPED names ===")
    for arm, runs in [("ON", on), ("OFF", off)]:
        rr, rd = winner_chasing(runs)
        if rr and rd:
            out.append(f"  {arm}: retained {st.mean(rr):+.1f}%  dropped {st.mean(rd):+.1f}%  "
                       f"gap {st.mean(rr)-st.mean(rd):+.1f}pp  (positive gap = keeps winners)")
        else:
            out.append(f"  {arm}: no prior_holdings_perf data (only present on new runs)")
    out.append("  If ON shows a bigger retained−dropped gap than OFF, the feedback is inducing "
               "momentum/winner-chasing.")

    report = "\n".join(out)
    # utf-8 explicitly: the report contains U+2212 MINUS SIGN, which the Windows
    # default cp1252 codec cannot encode.
    with open(os.path.join(HERE, "feedback_analysis.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print("\nWrote feedback_analysis.txt")

if __name__ == "__main__":
    main()
