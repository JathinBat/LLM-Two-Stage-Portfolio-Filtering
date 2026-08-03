# screen_validation/ — how to run the Rec #5 validation

This folder computes the relevance-screen validation numbers requested in NHSJS
Recommended comment #5 (validate the LLM news-relevance screen against a
hand-labeled sample).

## Files

- `screen_validation_metrics.py` — computes precision, false-omission rate,
  recall (re-weighted to the 8.0% population pass rate), F1, and Cohen's kappa,
  with confidence intervals, and prints a manuscript-ready sentence.
- `screen_validation_sample.xlsx` — the 180-row blind labeling sheet
  (70 screen-passed + 110 screen-rejected). **Currently lives at the repo root**;
  the script looks in this folder first, then the repo root, so either location
  works. Do not overwrite it — it holds the human labels.
- `screen_validation_key.csv` — **REQUIRED, not auto-generated.** See below.

## The key file (`screen_validation_key.csv`)

The key is the screen's *actual* PASS/REJECT decision for each blind id — the
ground truth the human labels are compared against. It comes from the original
screening run and **cannot be reconstructed after the fact** (regenerating it
would require re-running the LLM screen over the same 1,382-article pool with the
same id assignment). It is therefore **not** recreated by any script here.

To restore it, place the original `screen_validation_key.csv` into this folder
(it was included in the `ASDRP_repo_add.zip` delivery, and is produced by the
sample-builder that created `screen_validation_sample.xlsx`).

Expected format:

```
id,screen_pass
A001,0
A002,1
...
```

`screen_pass` = 1 if the screen PASSED the article, 0 if it REJECTED it.
`screen_validation_metrics.py` refuses to run without this file rather than
print meaningless numbers.

## Run

1. Finish labeling every row in `screen_validation_sample.xlsx` (the `relevant?
   (1/0)` column) and save.
2. Ensure `screen_validation_key.csv` is present (this folder or repo root).
3. `python screen_validation_metrics.py`
4. Paste the printed sentence + numbers into the manuscript (News Screening) and
   the response letter (Rec #5), clearing that [CONFIRM] marker.

## Note on re-weighting

The sample is **stratified**: it over-samples screen-passed articles so precision
is estimated on a stable base. Precision and false-omission rate condition on a
single stratum and need no adjustment. Recall and Cohen's kappa span both strata,
so they are re-weighted to the population pass rate (110/1382 = 8.0%, overridable
via an optional `_meta.json`). Confidence intervals: Wilson for precision and
false-omission rate; a stratified bootstrap for recall and kappa.
