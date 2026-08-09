# Reinvestment (Paper B) — URTC readiness assessment

*What the 90 live runs actually show, how strong the story is, and how the NHSJS reviewer's critiques apply to this paper.*

---

## 1. What you have

A real, fully-instrumented experiment — not a dry run. `results/portfolio_sim/` holds **90 live gpt-5.1 runs** over technology, 12-month horizon, budget $10k:

- **9 start windows**, monthly, Oct 2024 → Jun 2025 (each a 12-month hold).
- **Rebalance cadences:** 1, 3, 5, 11 rebalances (your 2×/4×/6×/12× = semiannual, quarterly, bimonthly, monthly).
- **Filter configs:** 20→10, 20→5, 10→5 (12× was run at 20→10 only).
- Each run stores per-period holdings, decision attempts, budget-rejection reasons, a 262-point daily value series, and a full risk profile (Sharpe, Sortino, vol, max drawdown, Calmar, beta, alpha, R²) with the matching SPY benchmark.

The engine itself (`portfolio_sim_runner.py`, ~950 lines) is a cash-aware simulator: real prices, cash ledger, cost basis, budget-constrained buy-only rebalancing, multi-turn reject-and-reprompt when the model overspends, and prior-period realized performance fed back into each decision. That is a genuine, distinct contribution from the v1 static selector.

---

## 2. What the data shows

### 2.1 Outperformance is universal — but it's largely sector beta

| Metric | Result (n=90) |
|---|---|
| Beat SPY on total return | **90/90** |
| Beat SPY on Sharpe | 83/90 (mean Sharpe **1.47** vs SPY 1.21) |
| Positive annual alpha vs SPY | 90/90 (mean **+10.4%**) |
| **Smaller max drawdown than SPY** | **0/90** (portfolio −23.4% vs SPY −15.4%) |
| Mean beta to SPY | **1.26** |

The headline "beats the market every time" is real but **cannot be the paper's claim**, because the portfolio is 29 tech large-caps in a tech bull market: beta 1.26 and *deeper drawdowns than SPY in every single run*. The excess return is substantially compensation for carrying more risk than the S&P 500, not proven skill. This is the exact trap the NHSJS reviewer flagged in v1 (Major #9), and it applies here with full force.

### 2.2 The genuinely interesting finding: rebalancing frequency does *not* help

De-confounded within the 20→10 config (the one run at all four cadences), paired across the 9 windows:

| Cadence | mean ROI | mean vs SPY | mean Sharpe | turnover/rebalance |
|---|---|---|---|---|
| Semiannual (1 rebalance) | **50.1%** | **+29.4pp** | **1.69** | 0.45 |
| Quarterly (3) | 40.6% | +19.9pp | 1.49 | 0.42 |
| Bimonthly (5) | 41.3% | +20.6pp | 1.47 | 0.30 |
| Monthly (11) | 45.9% | +25.2pp | 1.57 | 0.27 |

Paired monthly vs semiannual: **monthly wins only 4/9 windows, mean −4.2pp.** More frequent LLM re-decisions do **not** improve returns and, if anything, the least-active cadence is best. That is a non-obvious, defensible, publishable result: *the value is in the selection, not in active rebalancing.* It also lines up with theory (turnover without edge is a drag) and is far more interesting than "LLM beats market."

(Note: my earlier quick read said monthly looked best — that was an artifact of 12× only running on the strongest config. Paired within-config, it reverses. This is itself a lesson the reviewer would want handled correctly.)

### 2.3 Behavioral signals worth a figure

Monthly rebalancing holds more names (8.6 vs 6.0 avg) and triggers far more budget-violation reprompts (9.4 vs 0.4 per run) — the model struggles to respect the cash constraint when trading often, without any return benefit. Holdings retention across a rebalance is 77–84% (it churns ~1 in 5 names each time). Every window beat SPY on average; the weakest was Mar 2025 (+7.2pp), the strongest May 2025 (+30.1pp); single-run edge ranged +0.9 to +68pp.

---

## 3. Is it enough for a v2? Yes — with the same fixes the NHSJS reviewer already demanded

The reviewer's decision letter is effectively a free pre-review of this paper. The 16 majors cluster into themes that transfer almost one-to-one. Here is the mapping.

### Already satisfied (better than v1 was)

- **Risk numbers (Major #13).** v1 had to bolt these on. Paper B computes volatility, max drawdown, Sharpe, Sortino, Calmar, beta, alpha, and a benchmark comparison *for every run already*. Lead with them — including the honest "deeper drawdown than SPY" point.
- **Input-side look-ahead (Major #8).** The feedback loop is coded to pass only realized, as-of returns ("no look-ahead"), and gpt-5.1's cutoff (2024-09) precedes every window start (Oct 2024+). Document it and you're clean.
- **Reproducibility scaffolding (Major #5).** Per-run JSONs already store holdings, dates, decisions — the raw material for a run-decomposition table and appendix.

### Open gaps — must-fix before URTC (each maps to a real reviewer major)

1. **Fair / sector benchmarks — the make-or-break (Major #9).** SPY alone is not acceptable for a tech portfolio; the reviewer said so explicitly. Add **XLK (S&P 500 IT), QQQ/Nasdaq-100, and an equal-weight of your own 29-name universe**, plus **a buy-and-hold-of-the-initial-picks / no-rebalance control**. If the strategy still beats *XLK and equal-weight-universe*, you have a paper. If it doesn't, the honest finding becomes "LLM rebalancing tracks the sector but doesn't add alpha over it" — still publishable, but you must know which it is. This is the single highest-priority run.

2. **Isolate the mechanism — no-feedback / no-rebalance control (Major #10, the "handicapped baseline" critique).** Right now performance feedback is *always on* and there's no no-rebalance arm, so you cannot show the dynamic loop adds anything. Add (a) a no-rebalance baseline and (b) a feedback-off control at the same cadence. This is the intellectual core — without it the "self-reflective rebalancing" novelty is untested.

3. **Paired statistics + window independence (Major #3, #4).** The 9 windows are overlapping monthly starts of 12-month holds — **not independent**, exactly the issue the reviewer raised. Report it plainly; use sign test + Wilcoxon on paired per-window differences and exact binomial CIs on the win rates (e.g. 90/90 → a one-sided 95% CI), not a bare "beats 100% of the time." All scriptable on existing data.

4. **Replication (Major #3).** Most cells are a single run; LLMs are stochastic. Add 2–3 runs per cell for the headline config so within-cell spread is reported (a few cells already have w0/w1 — extend it).

5. **Universe & survivorship (Major #11).** The 29 tech names look 2025-vintage (today's winners). Disclose when/how the universe was assembled and the survivorship exposure — excluding, e.g., a name that was prominent in 2024 but faded biases results upward.

6. **Return-basis / benchmark consistency (Rec #8).** Confirm portfolio and benchmark are both on a total-return (adjusted-close) basis and state it; v1 had a price-vs-total-return mismatch.

7. **Run decomposition + errors (Major #5, #12).** `results/portfolio_sim/logs/` shows several runs errored and were retried. Report attempted/succeeded/excluded per cell and reconcile every in-text number to the workbook, so the counts are auditable.

### Inherited writing-craft fixes (cheap, do once)

Formatting (no bullets, Times New Roman, ≤ page limit), citation rigor (claim-by-claim, real DOIs not arXiv stragglers), terminology ("alpha" → "excess return" unless you mean CAPM alpha — and here you *do* have CAPM alpha, so define it), and moving causal narrative into Discussion. All carry straight over from the v1 checklist.

---

## 4. Recommended framing for URTC

Don't lead with "beats the market." Lead with the method and the mechanism question:

> *A cash-aware simulator that lets an LLM manage a real budget across periodic rebalances, used to test whether rebalancing frequency and performance-feedback improve a two-stage LLM stock selector — evaluated on risk-adjusted terms against sector benchmarks.*

Then the three honest findings: (1) it reliably beats the broad market but that is largely tech beta — against the *sector* the picture is [to be determined by the XLK/QQQ runs]; (2) **rebalancing frequency does not help and adds turnover and constraint-violations** — value is in selection, not timing; (3) [feedback on/off result, once the control is run]. That is novel, defensible, and distinct from the NHSJS paper.

---

## 5. Verdict

The experiment is **big enough and distinct enough for a URTC v2** — you have a genuine system and 90 live, richly-instrumented runs, which is more than most undergraduate submissions bring. The gating work is not "run the experiment" (done) but "make it defensible": add sector + equal-weight + no-rebalance baselines, add a feedback-off control, apply paired stats with honest window-independence caveats, and disclose survivorship. Every one of those is something the NHSJS reviewer already told you they care about, so treating their letter as this paper's rubric is the fastest path to an acceptance-grade submission.

Rough effort: the baseline/control runs are the only new compute (one focused grid on gpt-5.1, tech); everything else — stats, decomposition, survivorship writeup, benchmark alignment — is scriptable on the data you already have.
