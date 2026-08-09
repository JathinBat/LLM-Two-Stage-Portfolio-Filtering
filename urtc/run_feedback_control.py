#!/usr/bin/env python3
"""
Driver for the feedback ON vs OFF experiment (review gap A1).

Runs the headline config (gpt-5.1, technology, 20->10, monthly rebalancing) across
the 9 evaluation windows, N runs per cell, for BOTH arms:
  - feedback ON  -> results/portfolio_sim/feedback_on/
  - feedback OFF -> results/portfolio_sim/feedback_off/   (no performance memory)
Running both fresh in the same batch makes the ON-vs-OFF comparison matched (same
code, same prices, same period), which is cleaner than pairing against the older runs.

It calls through portfolio_sim_permutation_runner.simulate_permutation, so the
X-aware Stage-One screening (20 candidates) and all the correct wiring are reused.

SAFETY: without --go it does a MOCK dry-run (no API, no cost) to validate the plan and
plumbing. Add --go to spend real API budget.

Examples:
  python run_feedback_control.py                 # dry preview (mock, free)
  python run_feedback_control.py --go            # LIVE: 9 windows x 3 runs x 2 arms = 54 runs
  python run_feedback_control.py --go --arms off --runs 3   # OFF only, pair vs existing ON
Run from the repo root.
"""
import os, sys, json, time, argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

# load API key from .env if present (no-op otherwise)
try:
    import env_loader  # noqa: F401
except Exception:
    pass

import portfolio_sim_permutation_runner as ppr

# ---- experiment definition (edit here to widen scope) ----------------------- #
MODEL      = "gpt-5.1"
SECTOR     = "technology"
X, Y       = 20, 10             # 20 Stage-One candidates -> 10 holdings (headline config)
SEGMENTS   = 12                 # monthly (11 rebalances) — where the feedback fires most
PERIOD_MO  = 12
WINDOWS    = ["2024-10-01", "2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01",
              "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01"]
BUDGET     = 10000.0
RF         = 0.0
ID_BASE    = 9000               # synthetic perm ids (won't collide with existing runs)

def out_dir(arm):
    d = os.path.join("results", "portfolio_sim", f"feedback_{arm}")
    os.makedirs(d, exist_ok=True)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="LIVE run (spends API). Default: mock dry-run.")
    ap.add_argument("--arms", default="on,off", help="comma list: on,off (default both)")
    ap.add_argument("--runs", type=int, default=3, help="runs per cell (default 3)")
    ap.add_argument("--segments", type=int, default=SEGMENTS, help="rebalance segments (default 12)")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ("on", "off")]
    live = args.go
    total = len(WINDOWS) * args.runs * len(arms)
    print(f"Plan: {len(WINDOWS)} windows x {args.runs} runs x {len(arms)} arm(s) "
          f"= {total} runs | {args.segments} segments | {'LIVE (API)' if live else 'MOCK dry-run (free)'}")
    if not live:
        print("  (no --go: validating plumbing with the mock backend; add --go to run for real)")

    key = os.environ.get("OPENAI_API_KEY", "")
    if live and not key:
        sys.exit("OPENAI_API_KEY not set. Put it in .env or the environment, then rerun with --go.")

    idx = 0
    done = 0
    for arm in arms:
        odir = out_dir(arm)
        for w in WINDOWS:
            start = datetime.strptime(w, "%Y-%m-%d")
            end = start + relativedelta(months=PERIOD_MO)
            for run_i in range(args.runs):
                idx += 1
                p = {"id": ID_BASE + idx, "sector": SECTOR, "model": MODEL,
                     "period": PERIOD_MO, "freq": args.segments, "X": X, "Y": Y,
                     "start": start, "end": end}
                print(f"\n[{done+1}/{total}] arm={arm} window={w} run={run_i} "
                      f"(perm {p['id']})")
                try:
                    run = ppr.simulate_permutation(
                        p, api_key=key, dry_run=(not live), budget=BUDGET,
                        whole_shares=False, rf=RF,
                        feedback=(arm == "on"), txn_cost_bps=0.0)
                except Exception as e:  # noqa: BLE001
                    print(f"   ERROR: {e}  (skipping)")
                    done += 1
                    continue
                run.setdefault("metadata", {})["feedback_arm"] = arm
                run.setdefault("metadata", {})["run_index"] = run_i
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = (f"psperm_{p['id']:04d}_{SECTOR[:4]}_{MODEL.replace('.', '')}_"
                         f"{args.segments}x_{X}-{Y}_{start.strftime('%Y%m')}_"
                         f"{ts}_fb{arm}_w{run_i}.json")
                with open(os.path.join(odir, fname), "w", encoding="utf-8") as f:
                    json.dump(run, f, indent=2, default=str)
                roi = run.get("total_roi_pct")
                print(f"   saved {fname}  ROI {roi:+.2f}%" if roi is not None else f"   saved {fname}")
                done += 1

    print(f"\nDone: {done} runs. ON -> results/portfolio_sim/feedback_on/, "
          f"OFF -> results/portfolio_sim/feedback_off/")
    print("Next: python analyze_feedback.py")

if __name__ == "__main__":
    main()
