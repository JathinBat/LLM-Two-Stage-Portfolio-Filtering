# Session handoff — Paper B (MIT URTC), updated 2026-08-03

_Read `CLAUDE.md` first; it wins on any conflict. This file supersedes the
2026-07-30 version, whose one-line status ("the headline feedback ON/OFF
experiment is still not run") is **no longer true** — see §1._

**One-line status:** B-1 is **run and analysed**. Three arms, 324 live runs, all
clean. Sector benchmarks, selection-null and rebalance attribution have all been
recomputed on the new data. **The manuscript is not written** (owner is handling
that). What remains is writing, plus the housekeeping in §7.

---

## 1. What changed since the 2026-07-30 handoff

| Item | 2026-07-30 said | Actually true 2026-08-03 |
|---|---|---|
| News backfill | 5 months missing | **complete** — 2024-12→2025-03, 2025-04, 2025-05→2025-09, 2025-11→2026-06 all present |
| B-1 feedback ON/OFF | "still not run" | **run**: 108 + 108 runs |
| Hold-mode arm | specified, not run | **run**: 108 runs (`hold_winners`) |
| Sector benchmarks | "the single highest-priority run" | **done post-hoc** — no re-runs needed |
| selection_null / rebalance_attribution | computed on the legacy 90 | **recomputed per arm on the new 324** |

Two findings from the old handoff **did not survive** the larger sample — see §4.

---

## 2. Run inventory (verified 2026-08-03)

`results/portfolio_sim/` — three arms, **108 runs each**:

| arm | feedback | rebalance_mode | n |
|---|---|---|---|
| `feedback_on` | True | reset | 108 |
| `feedback_off` | False | reset | 108 |
| `hold_winners` | True | **hold** | 108 |

Every arm: 36 cells = 9 windows (2024-10 … 2025-06) × 4 cadences (2/4/6/12),
3 runs per cell, filter 20→10, `order_mode: weights`, `screened_news: True`,
`txn_cost_bps: 0`, no dry runs. Every run carries `no_reinvest_baseline`,
`daily_value_series`, `risk_metrics`, `benchmark_risk_metrics`, `periods[]`
(with the per-period `candidates` pool) and `final_holdings`.

**A schema wrinkle that looks like a bug and is not.** In the ON and OFF arms
`rebalance_mode` is absent from 54 files and `"reset"` in the other 54. The
absent group is exactly cadences **2 and 12**; the present group is **4 and 6**,
with **zero cell overlap**. The key was added to the schema between the original
2×/12× batch and the later 4×/6× batch. All 108 are reset mode. This is *not*
the config-leak contamination that hit an earlier pass.

**Still at `results/portfolio_sim/` root: the 90 legacy runs** (filters 20-10,
20-5, 10-5). Superseded, not archived — rule 2 forbids deleting and archiving is
the owner's call. **Consequence: any script with a recursive default glob will
mix legacy with new.** Always pass `--dir results/portfolio_sim/<arm>`.

---

## 3. Results (all recomputed 2026-08-03 on the new 324)

### 3.1 B-1 — feedback ON vs OFF: marginal on return, real in behaviour

Paired by (window, cadence), n = 36 cells:

| metric | ON − OFF | ON wins | sign p | Wilcoxon p |
|---|---|---|---|---|
| ROI % | **+1.32pp** | 25/36 | **0.029** | 0.062 |
| Sharpe (rf=4%) | +0.03 | 21/36 | 0.405 | 0.203 |
| vs no-reinvest (pp) | +0.28 | 21/36 | 0.405 | 0.509 |

On return it is a **borderline null** — the sign test clears 0.05, the Wilcoxon
does not. Do not claim a return benefit.

The behavioural difference is the more defensible result:

- **feedback makes the model keep its winners.** ON retains names that go on to
  return **+6.5%** while dropping ones that return **+3.7%** → gap **+2.8pp**.
  OFF's gap is **−0.5pp** — no discrimination at all.
- turnover ON **0.39** vs OFF **0.44**; cash 4.6% vs 4.0%
- budget re-prompts are **0.0 in both arms** — weights mode removes the
  violation by construction, which retires the old "52% of decisions violated
  the budget" result. That number belongs to dollars mode only.

### 3.2 Hold mode is the strongest effect in the study

`hold_winners` vs `feedback_on`, paired on the same 36 cells:

- **+4.42pp mean ROI, hold wins 27/36, Wilcoxon p = 0.0001**
- loss to INIT shrinks from **−8.90pp → −3.87pp**; beats INIT in 16/36 cells vs 9/36
- idle cash after a rebalance drops from **4.00% → 0.31%**

Trading the delta instead of liquidating 100% of notional recovers most of the
reinvestment loss. This is the design fix the old handoff listed as "specified,
not run" (§8.5 there).

### 3.3 Benchmarks — the make-or-break question, answered

Post-hoc from each run's stored daily series, adjusted-close basis, each run's
own window. Paired by cell (n=36), mean excess and cells won:

| arm | SPY | XLK | QQQ | Equal-weight universe | INIT |
|---|---|---|---|---|---|
| feedback_on | +17.88pp 36/36 | +3.07pp 26/36 **p=.014** | +10.99pp 34/36 | **−21.82pp 3/36** | −8.90pp 9/36 |
| feedback_off | +16.56pp 36/36 | +1.74pp 22/36 p=.119 | +9.66pp 36/36 | **−23.14pp 2/36** | −9.19pp 9/36 |
| hold_winners | +22.30pp 36/36 | +7.49pp 29/36 **p<.001** | +15.41pp 35/36 | **−17.40pp 4/36** | −3.87pp 16/36 p=.067 |

**It beats the sector indices and loses heavily to the equal-weighted universe.**

Read the EW column carefully before putting it in the paper. The 29-name basket
was assembled with hindsight, so equal-weighting it is a *survivorship-inflated*
benchmark; `CLAUDE.md` already designates it reference-only. It is evidence that
concentration hurt, not clean evidence of no skill. The clean null is §3.4.

### 3.4 Selection null — skill exists, and only with feedback on

Model's pick vs 10,000 random draws from **its own candidate pool** (the real
null per `CLAUDE.md`); 50 = no skill.

| arm | median pct | beat pool median | p | initial stage | rebalance stage |
|---|---|---|---|---|---|
| `hold_winners` | **54.1** | 346/648 (53.4%) | **0.0000** | 49.8, p=0.18 | 54.5, **p=0.0001** |
| `feedback_on` | **52.1** | 343/646 (53.1%) | **0.0070** | 50.3, p=0.57 | 52.4, **p=0.0079** |
| `feedback_off` | 49.5 | 316/645 (49.0%) | 0.0809 | 47.1, p=0.76 | 50.0, p=0.042 |

Two things replicate cleanly across all three arms and both sample sizes:

1. **The initial pick shows no skill anywhere** (p = 0.18 / 0.57 / 0.76). The
   edge lives entirely in *reselection*.
2. **The edge tracks feedback.** Turn feedback off and the median percentile
   falls to 49.5 and significance goes away. This is the strongest argument the
   paper has for the feedback loop — stronger than its effect on raw ROI.

Effect sizes are smaller than the legacy 10-run estimate (59.2nd percentile);
52–54 on 646 decisions is the honest number.

### 3.5 Rebalance attribution — the old headline no longer holds

| arm | mean alpha/decision | helped | Wilcoxon p | swap edge |
|---|---|---|---|---|
| `feedback_on` | −0.328pp | 257/536 (47.9%) | 0.1098 | −1.64pp |
| `feedback_off` | −0.251pp | 274/534 (51.3%) | 0.3746 | −0.22pp |
| `hold_winners` | −0.310pp | 244/540 (45.2%) | 0.2896 | −2.61pp |

The legacy 90 gave −0.533pp at **p = 0.0037**. On 1,610 decisions across the new
arms it is **−0.25 to −0.33pp and not significant in any arm.** *"Rebalancing
destroys value (p=0.004)" must not be carried into the paper unqualified.*

What does persist:

- **it still sells winners** — dropped names outperform added names in every arm
- **the cadence inversion is intact**: 2× ≈ −4.1 to −6.5pp per decision (helped
  22–37%), 12× ≈ 0 (helped 46–53%). Infrequent swaps compound; monthly ones get
  corrected.
- **MU is still the recurring casualty.** The single worst hold-arm decision
  dropped MU after **+338%** to add TSM at +50% (−27.6pp).

---

## 4. Two findings from the old handoff that did NOT survive

1. **Rebalance alpha significance** — see §3.5. Was p=0.0037 on 90 runs, now
   n.s. on 324.
2. **"52% of decisions violated the budget"** — an artifact of dollars mode.
   Weights mode reports 0.0 re-prompts. Still reportable, but only as a property
   of the dollars-mode design that motivated weights mode.

Both were headline claims in the previous handoff. Anything written from that
document needs re-checking against §3.

---

## 5. Artifacts produced this pass

| file | contents |
|---|---|
| `feedback_analysis.txt` | B-1 paired ROI/Sharpe/vs-INIT, behaviour, winner-retention |
| `analyze_feedback.txt` | duplicate of the above (tee); harmless, not deleted per rule 2 |
| `selection_null_feedback_on.txt` / `_feedback_off.txt` / `_hold_winners.txt` | per-arm selection null |
| `selection_null_periods.csv` | per-decision detail (last arm run wins — regenerate per arm if needed) |
| `selection_null.txt` | script default output; overwritten each run, use the per-arm files |
| `paperb_benchmarks.py` | **new, committed** — computes §3.3 from stored daily series; no re-runs, no model calls |
| `paperb_benchmarks.txt` | its output (the §3.3 tables) |

`paperb_benchmarks.py` needs local network access for prices; per `CLAUDE.md` the
cloud sandbox is firewalled from Yahoo Finance, so run it locally.

---

## 5b. New knob: `prompt_framing` (added 2026-08-03, not yet run)

The rebalance prompt in all 324 existing runs tells the model:

> "Your entire previous portfolio has already been **SOLD** … you now hold
> EXACTLY $X in cash and **no positions**. **Rebuild the portfolio from scratch**"

That is a **default-sell** framing: no incumbency, every name re-justified from
zero against a fresh candidate list. It is the direct structural cause of the
swap edge in §3.5. It is also **factually wrong under `rebalance_mode="hold"`**,
where only the delta is traded — which is why `hold_winners` has the *worst* swap
edge (−2.61pp) despite the best returns: execution changed, the prompt did not.

`prompt_framing="incumbent"` replaces it with three things, all targeting measured
failure modes:

1. **The real book** — "YOUR CURRENT HOLDINGS (NOT sold — you own these right
   now)", with "anything you leave unchanged simply stays as it is".
2. **The horizon** — "your next opportunity to change this portfolio is in 6
   months; a name you drop today you cannot re-buy until then." The model was
   previously blind to cadence, yet a bad swap costs −4 to −6.5pp at 2× and ≈0 at
   12×.
3. **A switching hurdle** — "only swap a name out if your conviction in the
   replacement is MATERIALLY higher — not merely equal, and not because the
   incumbent has already risen."

**Feedback isolation is preserved.** With feedback ON the book is shown *with*
per-name performance; with feedback OFF it is shown as **state only** (names,
shares, value, weight — no returns), so the control arm still receives no
performance signal. This is verified by assertion, not by inspection.

Wiring: GUI combo ("Framing"), `--prompt-framing`, preset key `prompt_framing`,
and it is in `EXPERIMENT_DEFINING_DEFAULTS` (default `"reset"`) so a preset that
omits it can never silently inherit `incumbent` — the failure that cost 108 runs
on 2026-07-31. Runs tag `_inc` in the filename and record `prompt_framing` in the
result JSON.

**Default is `reset`, so all 324 existing runs stay comparable.** No run has used
`incumbent` yet; it needs a fresh arm through the runner UI (rule 1). The natural
test is `hold_winners` × `incumbent` against the existing `hold_winners`, since
that is where the prompt/execution mismatch is largest.

---

## 6. Framing the paper (unchanged advice, now with numbers)

The three results point the same way and are mutually consistent:

- selection beats its own shortlist, but **only with feedback on** (§3.4)
- feedback's effect on raw return is a **borderline null** (§3.1)
- the reinvestment layer **still loses to just holding the period-0 picks** in
  every arm (§3.3, INIT column), though hold mode more than halves the gap

The honest headline is something like *"two-stage LLM selection adds value over
its own candidate pool, and a performance-feedback loop is what produces that
edge — but periodic reinvestment on top of the initial picks still subtracts,
and how you execute the rebalance matters more than how often you do it."*

Do not lead with "beats the market": +17 to +22pp over SPY on a 29-name tech
basket in this window is beta, and the paper already has the XLK/QQQ/EW numbers
to say so honestly.

---

## 7. What's left

1. **Write the manuscript.** Owner is handling this; no draft exists (only
   `paper_b/*.md` planning docs and two figures).
2. **Decide what to do with the 90 legacy runs** at `results/portfolio_sim/`
   root. They are superseded and they pollute recursive globs. Rule 2 says do
   not delete; moving to `archive/` needs the owner's go-ahead.
3. **Regenerate the results workbook** — `portfolio_sim_permutations.xlsx` is
   not at the repo root and any older copy predates all 324 runs.
4. **Survivorship disclosure** for the 29-name universe — still unwritten, and
   it is load-bearing for how the EW column in §3.3 is read.
5. **Rotate the API keys** before public release (unchanged).
6. **Validator `>` / `>=` off-by-epsilon** (3 of 756 periods) — deliberately
   left unfixed to preserve arm symmetry. Decide whether to fix and re-run, or
   disclose.

Resolved since the last handoff and needing no action: the Paper A / Rec #5
screened-corpus mismatch (§8.3 there). The published repo's `sector_config.py`
points technology at the gpt_filtered corpus with `isolate_news: True`, matching
the manuscript's claim. The stale `merged_news_data.csv` setting survives only on
this repo's `dev-custom-periods` branch, which is not published.

---

## 8. Key files

- `portfolio_sim_permutation_runner.py` — GUI; Feedback, Txn bps, Screened news,
  Min articles/per-days, Orders, Load preset, `--preset`, `--autostart`
- `portfolio_sim_runner.py` — engine; `order_mode`, `rebalance_mode`,
  `screened_news`, `weights_to_orders`, `normalize_target_book`, `execute_hold_mode`
- `presets/` — `today_feedback_on/off.json`, `fresh_news_feedback_on/off.json`
- `analyze_feedback.py`, `selection_null.py --dir <arm>`,
  `rebalance_attribution.py --dir <arm>`, `tier1a_baselines.py`
- `CLAUDE.md` — standing rules (authoritative)
