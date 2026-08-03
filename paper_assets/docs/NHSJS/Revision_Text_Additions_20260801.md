# Text additions for the NHSJS revision — drafted 2026-08-01

Three reviewer items still needed manuscript text rather than analysis. Draft
wording is below, ready to paste. Each notes where it goes and what it answers.

---

## 1. Must-Fix 8 — financial-report availability (Methods → Data Sources)

The reviewer's concern: *"A report whose period ends before a decision but which
was filed after it is lookahead, and as written the reader cannot tell whether
that occurred."* The revised Methods documented news thoroughly but said of
reports only that they were "sourced only from dates preceding the evaluation
period" — which is exactly the ambiguity raised, since a fiscal period ending
before a decision does not mean the report was public by then.

**Add to Data Sources, replacing the existing sentence about financial reports:**

> Financial reports are quarterly income statement, balance sheet and cash-flow
> filings retrieved for each Stage-One candidate through the Alpha Vantage
> company-report endpoints. Because a fiscal period ending before a decision date
> does not imply the corresponding report was public by that date, reports are
> admitted only if they were available strictly before the decision date.
> Availability is taken as the later of (i) the fiscal period end plus the
> statutory filing deadline for that report type — 45 days for quarterly filings
> and 75 days for annual filings, the accelerated-filer limits, so the bound holds
> across the whole universe rather than only the largest constituents — and
> (ii) any filing or report date supplied by the data source. Reports dated on the
> decision date itself are excluded, since filings accepted after 17:30 ET are
> disseminated on the following business day. Under this rule the most recent
> quarter is withheld from a decision falling inside the filing window, at a cost
> of approximately 0.7 of the roughly 75 reports available per company-decision.

**Add to Limitations (restatements are not detectable in the source data):**

> The data source does not flag restatements or amendments. A restated figure
> carries the fiscal period end of the original filing, so under the availability
> rule above it is treated as having been public from the original filing date
> even though the corrected values appeared later. We therefore cannot rule out
> that a small number of report values reflect subsequent restatement.

---

## 2. Must-Fix 9 / 11 — universe definition and survivorship (Methods → Scope, and Results → benchmark subsection)

The reviewer asked that the benchmarked universe be "reconstructed as it stood at
each window start, including companies later acquired or delisted." The response
to comment 11 correctly noted the *filtered pipeline* has no fixed universe, but
the equal-weighted and random-draw benchmarks introduced for comment 9 do draw on
a fixed 29-name list, so the point still applies to them.

**Add where the benchmark comparisons are introduced:**

> The eligible comparison set is a fixed list of 29 large-capitalisation
> technology and semiconductor companies (listed in Supplementary Information),
> used only for the unfiltered baseline and as the draw pool for the
> equal-weighted and random-portfolio benchmarks; the filtered configurations
> never see it and nominate candidates freely from screened news. Every
> constituent was continuously listed on a US exchange across the full evaluation
> span, so no member of the comparison set was acquired or delisted during the
> study period and the benchmark requires no point-in-time reconstruction on that
> account. The residual survivorship exposure is one of omission — companies that
> were large-capitalisation technology names at the earliest window starts but no
> longer qualify would not appear in the list. We note that the set retains
> several constituents whose share prices declined materially over the period,
> so it is not a selection of winners only, but we cannot exclude this omission
> bias entirely and flag it as a limitation of the benchmark rather than of the
> pipeline.

---

## 3. Response-letter updates

**Comment 8** — the current response describes only the news pipeline and does not
mention financial reports. It should state the availability rule above, and say
plainly whether the reported results were produced under it or whether the rule
was adopted in revision.

**Comment 14** — the Supplementary Information now contains what the response
promises: evaluation-window list (S1), run accounting (S2), per-cell run counts
(S3), per-configuration performance and risk (S4), component ablation with paired
tests (S5), and all pipeline prompts (S6).

**Comment 5** — the per-cell accounting requested is Table S2a–S2c and S3. Note
in the response that no configuration is missing from any window (zero empty
cells for all three models), which is the matched-design condition the reviewer
asked to see demonstrated.

---

## 4. Discrepancies found while assembling the Supplementary — need author decision

These are places where the manuscript's current wording does not match the run
data in this repository. They may reflect data produced elsewhere; each should be
checked against the source of the published numbers before resubmission.

**a. Ablation window coverage.** The manuscript states the ablation ran "across
all nine evaluation windows, with three independent generations per condition."
The parsed run files give 9 windows for every arm, but between 2 and 4 generations
per cell rather than a uniform 3, and one arm (single prompt over the full
candidate pool) has only 3 window-level observations. Suggest softening to the
actual counts, which Table S5a now reports.

**b. "Outperformed every variant in seven to nine of nine windows."** Recomputed
window-paired, the full pipeline beats the shuffled-reports arm 9/9 and the
hold-all arm 7/9, but the no-elimination arm only 5/9 and the budget-matched arm
6/9. The "seven to nine" range does not hold for all arms.

**c. "A budget-matched single-pass arm ... loses significantly (Wilcoxon
p = 0.012)."** Recomputed, that comparison gives p = 0.055 over nine paired
windows — directionally the same but not significant at the 0.05 level. The
p = 0.012 figure could not be reproduced from these files.

**d. Mean ROI of the full pipeline.** The manuscript reports 52.2%; the parsed
files give 52.4% for the full two-stage arm, which is the best arm as claimed.
This one is consistent within rounding.
