# Adversarial review — reinvestment / feedback study (Paper B)

*Method: read the live engine (`portfolio_sim_runner.py`) and the reinvestment orchestrator, then probed the 90 runs and the run logs to confirm or kill each concern. Findings are severity-ranked; a few hypotheses were checked and disproven — those are listed too, because they're load-bearing for the paper's credibility.*

---

## A. The feedback mechanism (the requested focus)

**How it actually works (verified in code).** At each rebalance the engine (1) marks the just-ended holdings at as-of prices, (2) builds a text report of how each held name did (`holdings_performance_report`), (3) **liquidates the entire portfolio to cash**, then (4) prompts the model to rebuild from scratch, injecting that performance report as "HOW YOUR PREVIOUS HOLDINGS PERFORMED (for context; they are already sold)." So the "feedback" is realized performance of the prior holdings, handed to a fresh full-reselection.

### A1 — [HIGH] The feedback's effect is untestable from these runs (no OFF control)
Every one of the 90 runs has feedback **on**; there is no feedback-off arm. The paper's most novel claim — that showing the model its own track record helps — cannot be supported, refuted, or even estimated from the current data. This is the single most important gap: a paper *about* self-reflective rebalancing currently contains zero evidence that the reflection does anything. The feedback-off control (plan Tier 1B) is not optional; it is the experiment.

### A2 — [MED] The feedback is structurally one-sided (survivorship in the signal)
The report only covers names the model **held** (`portfolio.active_tickers()`); it never shows how the names it *passed on* performed. So the loop can, at best, teach the model to re-evaluate its own winners/losers — it is structurally incapable of surfacing missed opportunities or false negatives. Any "learning" narrative must be scoped to this: the model reflects on its picks, not on the choices it declined. A reviewer will note the feedback can reinforce the model's prior selection biases rather than correct them.

### A3 — [MED] "Feedback" informs a from-scratch re-pick, not an adjustment — don't call it "learning"
Because the portfolio is fully liquidated before each decision (see B1), the model isn't refining a position — it re-selects the whole book with a note about last period attached. Framing this as iterative learning or an "agent that adapts" overstates it; it is "re-decide from zero, given a summary of the prior quarter." Keep the language to "performance-conditioned reselection."

### A4 — [MED] The behavioral effect is unauditable — the runs don't store what's needed
To show *how* the feedback changes behavior (does the model chase recent winners? dump recent losers?), you need each name's realized per-period return and whether it was re-selected. The result JSONs store end-of-period holdings but **not** per-name realized returns or the candidate pool the model chose from, so winner-chasing / loser-dumping cannot be measured from existing outputs. This is both a behavioral-analysis gap and a reproducibility gap; fix it by logging per-name period returns and candidates on the next run.

### A5 — [checked, CLEAN — state it as a strength]
I looked for look-ahead in the feedback path and did **not** find it: as-of prices are the last close on/before the decision date, news is windowed to `[as_of − lookback, as_of]`, financials are filtered as-of, and the performance report is computed **before** liquidation using as-of prices. Combined with skipping windows that start on/before the model's training cutoff, the temporal hygiene here is genuinely careful — say so explicitly, because it pre-empts the obvious "the LLM already knew the future" objection.

---

## B. General methodology

### B1 — [HIGH] "Sell everything first" = 100% turnover every rebalance, with no costs modeled
Each rebalance liquidates the entire portfolio and rebuys. With commission-free fractional trading assumed and **no transaction costs or taxes**, that's free in the simulation but not in reality. Monthly cadence does this 11 times per year; a real taxable account would realize short-term gains on nearly the whole book each time. This undercuts both the absolute returns and — critically — the frequency comparison, since the higher-turnover arms would be penalized most once costs exist. **Add a transaction-cost sensitivity** (e.g., 5/10/25 bps per turn) and re-rank cadences; the "frequency doesn't help" finding likely gets *stronger*, which is good for you.

### B2 — [HIGH] SPY is the wrong benchmark; the excess is largely sector beta
90/90 beat SPY, but β≈1.26 and 0/90 beat SPY on drawdown — the outperformance is substantially tech exposure. (Covered by the Tier 1A baselines; flagged here because it's the headline framing risk.)

### B3 — [MED] Sharpe assumes a 0% risk-free rate — it's inflated
`rf_annual = 0.0` throughout. Recomputing at rf = 4% (roughly 2024–25 T-bills) drops the strategy's mean Sharpe from **1.47 to 1.31** (SPY's ~1.21 falls similarly). The ranking survives, but the absolute numbers are overstated; report the rate used and apply the same rf to strategy and benchmark.

### B4 — [MED] Power & independence: one model, one sector, overlapping windows, ~1 run/cell
Nine 12-month windows on monthly starts overlap ~92%, so the effective n is far below 90; most cells are a single stochastic draw. Treat the window as the unit, report the overlap, and add replicate runs on the headline cells. (Consistent with the NHSJS Major #3/#4 critique.)

### B5 — [MED] Frequency finding is confounded with config
Monthly (12×) was only run at 20→10, the strongest config, so the raw "monthly is best" is entangled. The within-20→10 paired result (semiannual ≈ best) is the clean one; run monthly at the other two configs to de-confound. (Already in the plan.)

### B6 — [MED] News-corpus survivorship, on top of the universe issue
Candidates are drawn from a news dataset assembled retrospectively (2025). If that corpus over-covers names that are prominent *now*, the opportunity set is forward-biased before the model even chooses — a survivorship channel distinct from the ticker-universe point. Document how/when the news was collected and whether coverage is point-in-time.

### B7 — [LOW] Budget compliance is assisted, not one-shot
On an over-budget response the model gets up to two in-chat corrections quoting the exact overdraw. Fine and applied uniformly, but "the model respects a hard budget" should be reported as "within a reject-and-reprompt loop," not as first-try competence.

### B8 — [LOW] Run ledger / reproducibility
26 distinct cells logged transient errors (missing `openai` package, connection drops) and were re-run; I confirmed these are **infrastructure**, not model or budget failures, so there is no behavioral selection bias — but publish the attempt/success ledger and note the one unrecovered id (0309, a pre-cutoff window that should be excluded anyway). Also verify `max_completion_tokens=2500` never truncated a 10-holding JSON into a parse-retry.

---

## Priority order

1. **A1 — run the feedback-off control.** Without it there is no feedback paper.
2. **B1 — add transaction-cost sensitivity.** Cheap recomputation; likely strengthens your frequency result and defuses the biggest realism objection.
3. **B2 / Tier 1A — fair benchmarks** (already scripted).
4. **A4 — log per-name returns + candidates** on the next run so behavior and the selection null both become analyzable.
5. **B3, B4, B5, B6, B8 — disclosures and de-confounding**, all cheap.

## Net read

The engine is carefully built where it counts most (temporal hygiene is clean — A5), and the honest-caveat instinct in your own notes is right. The two things that would most change a reviewer's verdict are both missing *experiments*, not flaws in what exists: the feedback-off control (A1) and a costed, fairly-benchmarked comparison (B1/B2). Everything else is disclosure and de-confounding you can do on the data in hand.
