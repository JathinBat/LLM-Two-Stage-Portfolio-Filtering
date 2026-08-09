# Handoff — Splitting the Two-Stage LLM Filtering Paper Into Two

**Project:** Dividing the combined manuscript *"A Two-Stage Sequential Filtering Framework for LLM-Driven Long-Term Stock Selection"* into two separately publishable papers.
**Status:** Planning complete on strategy and constraints; one core decision and one data question still open before execution.
**Prepared as a working handoff — confirm publication-ethics specifics with Dr. Qin and venue organizers before submitting.**

---

## 1. The goal and the shape we landed on

The combined paper does too much for one venue. It bundles three contributions: (1) the two-stage sequential-filtering framework, (2) proof it beats both the unfiltered baseline and the market, and (3) a design-space analysis (inverted-U in candidate-pool size, the 2× rule, model dependence, sector attribution).

**Key constraint discovered along the way:** the two papers cannot be two slices of the *same* finding. The framework result and the 2× rule are two halves of one claim — splitting them weakens both, **and** journal/IEEE rules against republication independently forbid carving one finding into two papers. So the split must be by **contribution**, with each paper carrying its own thesis and its own headline result. The bar is two genuinely strong, self-justifying papers — not a flagship plus a thin satellite.

The agreed structure:

- **Paper A** — keep the framework + optimization story whole (this is the "strong" paper), and *add* new risk-adjusted metrics it doesn't currently have.
- **Paper B** — a new-experiment paper with its own thesis that Paper A does not answer. Exact thesis still open (Section 4).

---

## 2. Paper A — framework + optimization (mostly ready)

**Thesis:** Information structure is a determinant of investment performance, separate from model choice; the optimum follows the 2× rule (Stage-One pool ≈ twice the target portfolio size).

**Contents:** full Intro, Prior Work, Methods; results 4.1–4.6 (beats-baseline, inverted-U, 2× rule, model comparison, consistency, sector attribution); Appendix A; **plus newly reconstructed Sharpe, Sortino, and maximum drawdown.**

**Why it's the low-risk URTC candidate:** self-contained, runs almost entirely on existing data, and condenses to URTC's hard 5-page limit far more naturally than the combined draft. The risk metrics compete for space, so some per-model tables likely move to an appendix or to Paper B.

**What it still needs:** the risk-metric reconstruction (no new model calls — see Section 6), connective rewriting, and condensation to 5 pages.

---

## 3. Publication strategy and constraints

### IEEE (URTC)
- Only original work that has not appeared elsewhere and is not under review elsewhere.
- Authors **must disclose** all prior publications and current/concurrent submissions, and state explicitly how the new work differs.
- Self-plagiarism — including parallel submission of substantially similar manuscripts — is treated as plagiarism.
- Staged evolution of work **is** permitted *if* the prior work is cited and the new submission's substantial novel contribution is stated.
- Penalties for double-submission are real (rejection; multi-month bans in some societies).

### NHSJS
- A real, peer-reviewed, indexed journal — so an NHSJS publication **counts as a prior publication** for IEEE's purposes.
- Requires confirming the work is original / not published elsewhere.
- Format: Reports ≤ 4 figures/tables; Original Research ≥ 5 figures/tables. Submit the new-experiment paper as full Original Research, not a Report.
- Rolling submissions; review runs ~8–14 weeks.

### The ordering consequence (important)
We are submitting to **NHSJS first**. Whichever paper publishes first becomes prior art the second must cite and differentiate from. Two implications:

1. The two papers must be **genuinely distinct contributions** (handled by the Paper A / Paper B split above).
2. Shared methods text (both use the same pipeline) is handled by **cite-and-condense**: the second paper summarizes the pipeline and cites the first for full detail — acceptable under IEEE rules.
3. NHSJS review will likely still be open when URTC is submitted, so **disclose the NHSJS submission to URTC** as concurrent *related (not substantially similar)* work, cited as "under review / forthcoming."

### Deadlines
- **URTC 2026:** official date **not yet released.** Estimate **early August 2026** (2025 deadline was Aug 3; 2024 was Aug 9; conference mid-October). Hard 5-page limit. Accepts high-school students affiliated with a collegiate research program (covers our ASDRP situation). **Watch urtc.mit.edu for the official date.**
- **NHSJS:** rolling — no fixed deadline, which gives the new-experiment paper more runway.

### Placement logic
Against an ~early-August URTC deadline (~5 weeks out as of this writing), **Paper A is the safer URTC submission** (ready, self-contained, fits 5 pages) and **Paper B fits NHSJS** (rolling deadline absorbs the new runs). Note the tension: if "NHSJS first" is strict, Paper B's new runs must finish before Paper A goes to URTC. Resolve this explicitly.

---

## 4. Paper B — OPEN: two live candidate theses

Both are feasible; each has a single gating question. Pick one (or combine).

### Candidate 1 — Selection vs. management ("beyond buy-and-hold")
**Thesis:** Paper A optimized *what* the model picks; this paper asks how much the *management policy* applied to those picks changes outcomes.
- **Passive-overlay tier** (periodic rebalance-to-target, volatility-targeting, stops): deterministic re-computation on existing portfolios — **zero new model calls.**
- **Dynamic re-selection tier** (model revises holdings at interim rebalance dates): new calls, but feasible on the reduced compute budget (Section 6); architecturally supported (Section 5).
- Keep it **long-only** — short/leverage/derivatives shift the framing from "investing" to "trading," which the Intro deliberately distinguishes against.
- **Gating question:** how many rebalance points per 12-month window — quarterly (3 interim decisions, modest cost) vs. monthly (11, ~4× the inference and heavier cutoff bookkeeping)?

### Candidate 2 — Sector / market generalization (B1)
**Thesis:** Does the 2× advantage survive outside large-cap tech, and across market regimes?
- **Feasible because sector is a controllable input** (Section 5), not just an emergent output.
- Pick sectors to span the conviction gradient: financials (strong/news-dense) → consumer or industrials (medium) → energy/utilities (thin, commodity-driven). If the advantage collapses in energy, that's *mechanism confirmation*, not failure.
- Lead on **sector breadth on GPT-5.1**; treat **market regime** (down/sideways) as a scoped secondary section — GPT-5.1's post-cutoff windows are almost all a rising market, so the regime test leans on GPT-4o / 4o-mini.
- Geographic (non-US) markets are higher-ceiling but carry real overhead: benchmark is hard-coded to SPY, and non-US semi-annual reporting degrades the Stage-Two financial-report input.
- **Gating question:** can we assemble **dated, pre-window news corpora for 2–3 non-tech sectors** from the existing provider? If yes → true sector study. If no → the honest version becomes an *input/source-sensitivity* paper instead.

---

## 5. What the code supports (from `investment_strategy_generator.py` + `permutation_runner.py`)

**Sector is controllable.** The runner passes `KEYWORD` (currently `"technology"`), and the Stage-One prompt asks the model to "identify companies in the {keyword} sector." Changing the keyword retargets the pipeline — but two things are hard-coded to tech and must move with it:
- **News loader** (`_load_news_data`) grabs *every* news CSV in the workspace by date; the keyword only prioritizes filenames. → Must supply real per-sector dated news files, or a non-tech keyword still gets fed tech news.
- **`UNFILTERED_TICKER_UNIVERSE`** is a fixed ~29-ticker tech/semis list used by the unfiltered baseline. → Each sector needs its own unfiltered universe, or the matched comparison breaks.
- **Benchmark** is hard-coded `SPY` (`_calculate_benchmark_comparison`). → Fine for US sectors; must change for non-US markets.

**Returns are clean buy-and-hold.** Prices use adjusted close (`includeAdjustedClose=true`, `adjclose` fills the `Close` field), and `_calculate_period_returns` computes endpoint-to-endpoint per-stock return weighted by initial allocation — mathematically exact buy-and-hold total return. (Confirms the paper's "adjusted close" claim.)

**Risk metrics and passive overlays need NO new model calls.** The daily price path is already fetched, then discarded via `iloc[0]`/`iloc[-1]`. Reuse the same daily adjusted frame to build a daily portfolio-value series → Sharpe, Sortino, max drawdown, and all passive overlays (rebalancing, vol-targeting, stops) reconstruct from existing runs. This removes the earlier "daily series weren't retained" limitation.
- *Implementation notes:* build the value series as the sum of per-position share values (shares fixed at entry; daily value = shares × daily adjusted close) so buy-and-hold weights are allowed to drift — do **not** apply fixed weights daily (that silently assumes daily rebalancing). For Sortino, use *downside* deviation (only negative excess returns), not Sharpe's full denominator. Per Section 3.9 (non-independence), report risk metrics per-window with the existing heuristic caveat rather than pooling naively across overlapping windows.

**Dynamic re-selection is architecturally supported.** The pipeline is date-parameterized (`run_sliding_window_analysis` loops windows by `news_end_date` / analysis boundaries), and a cutoff check runs per call (`_validate_model_for_date_range`; `_check_training_data_overlap`; runner-side `_skip_reason` / `TRAINING_CUTOFFS`). Invoking at interim rebalance dates re-validates the cutoff automatically at each date — the per-decision-date leakage discipline is already in the machinery.

**Current run grid** (`permutation_runner.py`): temperature × period × start date × model × filter config, with cutoff-based skipping.
- Temperatures: 0.3 / 0.4 / 0.5
- Models: gpt-4o / gpt-4o-mini / gpt-5.1
- Start dates: 13 (Jun 2024 – Jun 2025)
- Filter configs: ranked_final, 30→15, 30→10, 30→5, 20→10, 20→5, 10→5, 10→3, 5→3 (+ unfiltered)

---

## 6. Compute plan for the new experiments

Decisions that make new runs affordable:
- **Drop the temperature sweep** (3× → 1×, e.g. fix at 0.4). Justified: temperature showed no systematic effect in Paper A. State this in methods to isolate the policy/sector variable.
- **Reduce non-flagship model runs**; GPT-5.1 already only runs post-cutoff.
- Net effect: roughly **4–6× fewer calls** than Paper A's full grid — enough to fund the dynamic-rebalancing tier and/or sector runs.

---

## 7. Open decisions (need answers before execution)

1. **Paper B thesis:** selection-vs-management (Candidate 1), sector/market generalization (Candidate 2), or a combination.
2. **Gating data — sector news:** can we get dated, pre-window news for 2–3 non-tech sectors? (Decides whether Candidate 2 is a true sector study or pivots to source-sensitivity.)
3. **Rebalancing scope** (if Candidate 1): passive-overlay tier only (no new calls) vs. include dynamic re-selection (new calls), and rebalance frequency (quarterly vs. monthly).
4. **Placement & ordering:** confirm which paper → URTC (early Aug) and which → NHSJS (rolling), reconciled with the "NHSJS first" intent and the anti-republication rules.
5. **Sign-off:** run the final two-paper plan and the concurrent-submission disclosure past Dr. Qin, and ideally confirm with URTC organizers that a same-group NHSJS paper under review is acceptable to disclose-and-differentiate.

---

## 8. Buildable now (do not wait on the decisions above)

- **Paper A full draft** to the 5-page URTC constraint, with the risk metrics slotted in.
- **Daily-series reconstruction script** for Sharpe / Sortino / max drawdown — reuses the daily adjusted frames the pipeline already fetches; runs on existing portfolios; no new model calls.

---

## 9. Recommended next actions (in order)

1. Answer the sector-news question (#2) — a one-feed, one-run pilot settles it cheaply.
2. Lock Paper B's thesis (#1) and scope (#3).
3. Start the Paper A draft and the risk-metric reconstruction in parallel (Section 8) — neither waits on the above.
4. Confirm the plan with Dr. Qin; watch urtc.mit.edu for the official 2026 deadline.
