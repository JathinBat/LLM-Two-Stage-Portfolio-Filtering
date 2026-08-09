# Running the new controls (feedback-off, costs) + what the engine now logs

This session upgraded `portfolio_sim_runner.py` (and threaded flags through
`portfolio_sim_permutation_runner.py`) to close the two biggest review gaps —
the missing feedback control (A1) and un-modeled transaction costs (B1) — and to
log the data needed to audit the feedback loop and build the selection null (A4).

## What changed in the engine

New in every result JSON:
- top level: `feedback` (bool), `txn_cost_bps` (float), `risk_free`.
- per period: `txn_cost`, `candidates` (the opportunity set the model chose from —
  the missing piece for a real selection null), and `prior_holdings_perf`
  (per-name realized return + unrealized, the exact signal the feedback loop feeds).

New knobs:
- `--feedback on|off` — OFF omits the "how your previous holdings performed" block
  at each rebalance (the control arm; no performance memory).
- `--txn-cost-bps N` — charges N bps on round-trip notional (sell-all + buys) each
  rebalance, flowing into value and the daily series. Default 0 = frictionless.
- Token limit raised 2500 → 4000 with a truncation warning (guards against a
  10-holding JSON getting cut off and forcing a reparse).

## A1 — run the feedback control (the key experiment)

**Easiest — one command** (`run_feedback_control.py` drives the whole grid for BOTH
arms into `feedback_on/` and `feedback_off/`, then `analyze_feedback.py` pairs them):
```
python run_feedback_control.py            # MOCK dry-run first (free) — validates the plan
python run_feedback_control.py --go       # LIVE: 9 windows x 3 runs x 2 arms = 54 runs
python analyze_feedback.py                # paired ON-vs-OFF ROI/Sharpe + winner-chasing
```
`analyze_feedback.py` reports the paired per-window sign/Wilcoxon tests (does feedback
help?), behavioral differences (turnover, re-prompts), and a winner-chasing test that
uses the new per-name performance logging. Edit the constants at the top of the driver
to widen scope (more cadences, more runs). Manual alternatives below.

---

Single run via the engine CLI:
```
python portfolio_sim_runner.py --start 2024-10-01 --end 2025-10-01 --segments 12 \
  --max-holdings 10 --model gpt-5.1 --feedback off --out fb_off_202410_12x_20-10.json
```
Full grid via the permutation runner without touching the GUI — set the env var and
route to a SEPARATE results folder so filenames don't collide with the feedback-on runs:
```
set PSIM_FEEDBACK=off        # Windows (PowerShell: $env:PSIM_FEEDBACK="off")
# then launch the runner and set the results-subfolder field to e.g. "feedback_off"
python portfolio_sim_permutation_runner.py
```
Run the same cells as your headline grid (gpt-5.1, 20→10, 9 windows, ≥3 runs each).
Then compare ON vs OFF ROI/Sharpe with paired per-window tests — that is the first
direct evidence of whether the self-reflection does anything.

> Collision note: the output filename doesn't encode feedback mode, so ALWAYS put the
> OFF batch in its own results subfolder (or `--out` name). Otherwise an OFF run can
> overwrite the matching ON run.

## B1 — transaction costs

Already answered post-hoc on the existing runs (no new calls) — run:
```
python reanalysis_costs_rf.py
```
It re-ranks cadences at 0/5/10/25 bps and shows the rf-Sharpe sensitivity. Headline
from the current data: at 25 bps the monthly arm loses ~8.1pp to costs vs ~1.0pp for
semiannual, so semiannual's edge over monthly widens from +4.2pp (gross) to +11.2pp
(net) — the "frequency doesn't help" result gets stronger once trading isn't free.

For new runs you can also bake costs in directly with `--txn-cost-bps 10` (or set
`PSIM_TXN_BPS=10` for the grid).

## A4 — selection null + behavioral audit (enabled for future runs)

New runs now store `candidates` and `prior_holdings_perf`, so after your next batch
you can (1) build the real "beats random selection" null by drawing from each period's
actual candidate pool, and (2) test whether the model chases recent winners / dumps
losers in response to the feedback. Neither was computable from the old runs.

## B3 — risk-free rate

Report the rf you use and apply it to strategy AND benchmark. The runs assumed rf=0,
which lifts the mean Sharpe from 1.31 (rf=4%) to 1.47 (rf=0).
