# Reinvestment (Paper B) — URTC experiment plan

*Exact runs, baselines, statistics, and disclosures to turn the existing 90-run study into an acceptance-grade URTC submission. Ordered by priority; the deadline is ~early August, so Tier 1 is the commitment and Tiers 2–3 are upside.*

---

## Framing (what the paper claims)

Lead with the **method and the mechanism**, not "beats the market":

> A cash-aware simulator lets gpt-5.1 manage a real $10k budget across periodic rebalances of a two-stage-filtered stock portfolio. We ask whether (a) the strategy adds value over fair sector baselines on risk-adjusted terms, (b) rebalance frequency matters, and (c) feeding the model its own realized performance improves decisions.

Three honest headline results the data already supports or will support:
1. It reliably beats the broad market, but that is largely tech beta (β≈1.26, deeper drawdowns than SPY in 90/90 runs) — so the real test is the *sector* baseline.
2. **Rebalance frequency does not help** — semiannual ≈ best, monthly beats it in only 4/9 windows. Value is in selection, not timing.
3. Whether performance-feedback helps — *pending the control in Tier 1*.

---

## Tier 1 — Must-do (cheap; settles the mechanism). Maps to reviewer Majors #9, #10, #3, #13.

### 1A. Fair baselines — **no new LLM calls**, pure recomputation over prices you already fetch

**Universe caveat (important):** the strategy is news-driven and *unbounded* — across the 90 runs it held 36 distinct tickers, 10 of them outside the 29-name list (`sector_config.universe()` is used only by the mock backend and v1's unfiltered baseline, not the live strategy). So an "equal-weight / random draw from the 29 names" is **not** the strategy's opportunity set and is **not** a valid skill-vs-luck null. The baselines split into two groups:

*Primary — universe-free, valid as written:*
- **XLK** (S&P 500 IT) and **QQQ / Nasdaq-100** — the correct passive sector benchmarks.
- **SPY / SMH** — broad market and semiconductor context.
- **Buy-and-hold the LLM's own initial picks** (period-0 holdings, no rebalancing) — isolates *rebalancing* value using only the model's own choices, so no universe is assumed.

*Reference only — a fixed big-tech basket, NOT the opportunity set:*
- Equal-weight / cap-weight / random draws from the 29-name basket. Report as context ("how a naive big-tech buyer did"), never as a skill test.

*The real selection null (deferred):* equal-weight or random sampling from **each period's actual candidate pool** — the set the news filter surfaced. Those pools are not stored in the run JSONs, so this needs a small re-run with candidate logging added to the runner (a one-line dump of Stage-One/Stage-Two candidates per rebalance). A principled sourceable alternative is random portfolios drawn from **point-in-time XLK / Nasdaq-100 constituents** as of each window start (a pre-registered opportunity set), but that requires historical index membership (survivorship-clean).

Decision rule: the paper's central claim is **"beats XLK / QQQ, and beats its own buy-and-hold picks,"** with risk metrics beside every number. If it clears those → genuine sector-relative and rebalancing skill. If it only beats SPY but not XLK → the honest finding is "tracks the sector, no added alpha" — still publishable, more credible. The "beats random selection" claim is explicitly held back until candidate logging exists.

### 1B. No-feedback and no-rebalance controls — **new LLM calls, but small**
The mechanism claim needs these arms on the headline config (gpt-5.1, 20→10, all 9 windows), 3 runs each:

| Arm | Purpose | New runs |
|---|---|---|
| **No-rebalance** (initial pick, hold) | is rebalancing worth anything? | covered by 1A (no LLM) |
| **Feedback-OFF** (rebalance, but don't show prior performance) | does self-reflection help? | 9 windows × 3 = 27 |
| **Feedback-ON** (current default) | already have it | 0 (reuse) |

That is **~27 new runs** for the feedback contrast (optionally + a mid and aggressive config → ~80). This is the intellectual core: without the feedback-OFF arm, the "self-reflective rebalancing" novelty is untested.

### 1C. De-confound frequency — **new LLM calls, small**
Monthly (12×) was only run at 20→10. Run monthly at **20→5 and 10→5** too (9 windows × 2 configs × 3 = ~54 runs) so cadence isn't entangled with the best config. If time is tight, this is the first thing to cut — the within-20→10 paired result already carries the frequency finding.

### 1D. Statistics (no new runs)
- **Paired per-window tests** (sign test + Wilcoxon) on strategy − baseline, per baseline, per cadence.
- **Exact binomial CIs** on every win rate (e.g. beat-XLK k/9) via `scipy.stats.binomtest`.
- **State window non-independence explicitly**: 9 monthly-start 12-month holds overlap ~92%; treat n=9 as the effective unit and say so (this is precisely the v1 reviewer's Major #3/#4 point).
- **Replication**: extend every headline cell to ≥3 runs; report within-cell SD.

---

## Tier 2 — High value, gated on news. Maps to Majors #9 (generalization) and #11.

**One contrasting sector: healthcare.** It's the acid test because XLV was flat-to-weak over Oct 2024–Jun 2025 — beating a *non-booming* sector benchmark is the result no reviewer can dismiss as beta. `sector_config.py` already has the healthcare universe and XLV benchmark wired.

**Gating item (do this first or Tier 2 is invalid):** source real, *dated, pre-window* healthcare news for the 9 windows via the news pipeline. Without as-of news the run is either signal-free or look-ahead-contaminated. Budget this as the main cost of Tier 2.

Then replicate the Tier-1 headline grid (gpt-5.1, 20→10, feedback on/off, 9 windows, 3 runs) on healthcare: ~54 runs. Financials (XLF) is a weaker test (it ran hot post-election, same rising-tide problem as tech) — skip unless time allows a third sector.

Payoff: turns "one sector, one bull market" into a generalization claim, and directly answers the reviewer's sector-benchmark and external-validity concerns.

---

## Tier 3 — Reproducibility & disclosure (writing, no runs). Maps to Majors #5, #8, #11, #12, Rec #8.

- **Run-decomposition table**: per cell, runs attempted / succeeded / errored-and-retried / analyzed (the `logs/psperm_*_error.txt` files show several needed retries — report them).
- **Survivorship disclosure**: state when/how the 29-name universe was assembled; if 2025-vintage, name the bias (you sampled today's survivors) and its direction (upward).
- **Look-ahead statement**: document that feedback passes only realized as-of returns and that gpt-5.1's 2024-09 cutoff precedes every window start.
- **Return-basis consistency**: confirm portfolio and all benchmarks are total-return (adjusted close) and say so; quantify any price-vs-total-return gap.
- **Prompts appendix + window list + public repo** (already key-scrubbed) with the run JSONs.
- **Terminology**: you have genuine CAPM alpha here (beta-adjusted), so define "alpha" rather than renaming it — but be explicit it's measured vs SPY, and report the vs-XLK alpha too.

---

## Reviewer-major → task cross-check

| NHSJS Major | How it lands on Paper B | Tier |
|---|---|---|
| #9 stacked/sector/random benchmarks | XLK, QQQ, equal/cap-weight, random portfolios | 1A |
| #10 handicapped baseline / isolate mechanism | no-rebalance + feedback-off controls | 1B |
| #3, #4 replication, paired stats, window independence | paired tests, CIs, ≥3 runs/cell, overlap stated | 1B/1D |
| #13 risk numbers | already computed per run — lead with them | done |
| #8 input-side look-ahead | disclosure statement | 3 |
| #11 universe & survivorship | disclosure + universe listing | 3 |
| #12 numeric consistency | reconcile every number to the workbook | 3 |
| Rec #8 return basis | align total-return, state it | 3 |
| #1, citations, formatting, terminology | inherit the v1 checklist | 3 |

---

## Effort & sequence (to ~early August)

1. **Week 1:** Tier 1A + 1D — all recomputation and stats on existing data. This alone converts the paper from "beats SPY" to a defensible risk-adjusted, fairly-benchmarked study. Decide from 1A whether the claim is "beats the sector" or "tracks the sector."
2. **Week 1–2:** Tier 1B controls (~27–80 runs) — the feedback/no-rebalance mechanism arms. Small compute.
3. **Week 2 (optional):** Tier 1C monthly-at-other-configs; Tier 3 writing in parallel.
4. **Week 2–3 (upside):** Tier 2 healthcare — only if the news collection completes early; otherwise defer to a journal version and note single-sector scope as a stated limitation.

New-compute total: Tier 1 ≈ 30–130 gpt-5.1 runs (vs the 90 you already ran — very affordable); Tier 2 adds ~54 + the news-collection cost. Everything else is scripting and writing on data in hand.
