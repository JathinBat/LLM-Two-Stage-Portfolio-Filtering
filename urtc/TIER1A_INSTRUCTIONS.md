# Tier 1A — run instructions

This computes the single most important missing number for the reinvestment paper: **does the LLM strategy beat fair sector baselines (XLK, QQQ, equal-weight universe) — or just SPY?** It also gives you the "beats its own buy-and-hold picks" test (does rebalancing add anything) and a random-portfolio skill percentile.

## Why this has to run locally (not in the cloud session)

The `FINANCIAL_REPORTS/<ticker>/PRICE_HISTORY_5Y.csv` caches on your machine are **stale** — the universe files end around May 2024, so they don't cover the Oct 2024 → 2026 evaluation windows. The script therefore pulls **fresh** adjusted-close prices from Yahoo (via `yfinance`), which needs live internet. Running it under Claude Code on your machine is the clean way to do that.

## Prerequisites

```
pip install yfinance pandas numpy scipy      # yfinance is the only new one vs requirements.txt
```

## How to run

1. Put `tier1a_baselines.py` in the **repo root** (same folder as `results/`, `sector_config.py`).
2. From the repo root:

```
python tier1a_baselines.py
```

It reads every `results/portfolio_sim/psperm_*.json`, fetches prices once for the 29-name universe + XLK/QQQ/SPY/SMH, and computes each baseline buy-and-hold over each run's exact window on a total-return (adjusted-close) basis.

## What it writes (next to the script)

- **`tier1a_report.txt`** — the headline: per-run win rates with 95% CIs, window-level paired sign + Wilcoxon tests (the honest unit, n = 9 windows), mean/median excess vs each baseline, and the random-portfolio percentile.
- `tier1a_per_run.csv` — one row per run: strategy ROI and excess vs XLK / QQQ / SMH / equal-weight / cap-weight / its own initial picks.
- `tier1a_window_summary.csv` — per-window baseline returns vs strategy.

## How to read the result (the decision this makes)

**Universe caveat:** the strategy is news-driven and unbounded (it held 10 names outside the 29-name basket), so the equal-weight / random-basket numbers are a fixed big-tech *reference*, NOT the strategy's opportunity set — don't read them as a skill test. The real baselines are the ones that need no universe: XLK, QQQ, SPY, and INIT.

- **If the strategy beats XLK and QQQ** on the window-level paired tests → genuine sector-relative outperformance. Lead with it.
- **If it beats SPY but not XLK / QQQ** → the excess was sector beta; the paper's spine becomes "within-sector: frequency and feedback effects," stated honestly. Either way it's publishable — this tells you which paper you're writing.
- **`vs_INIT`** (strategy minus buy-and-hold of its own period-0 picks) isolates whether *rebalancing* adds anything at all — no universe assumed. Given that more rebalancing didn't help, watch this closely; ≈0 or negative is itself a clean result.
- **`rand_pctile`** is REFERENCE ONLY (random draw from the fixed big-tech basket). A true "beats random selection" null needs a re-run that logs the news filter's per-period candidate pool — the pools aren't stored in the current run JSONs.

## Notes / caveats baked in

- Cap-weight uses *current* market caps from yfinance (an approximation — it's start-of-window weights that would be ideal; the script documents this and skips gracefully if caps don't fetch).
- Random seed is fixed (`RANDOM_SEED = 20260723`) so the 1,000-portfolio draw is reproducible.
- Everything is total-return (auto-adjusted) for consistency with the simulator; the script notes this in-code.

## If you'd rather have Claude Code drive it

Paste this into Claude Code from the repo root:

> Run `tier1a_baselines.py` from the repo root. If yfinance throws rate-limit or missing-ticker errors, retry with a small delay and report which tickers failed. When it finishes, open `tier1a_report.txt` and tell me: (1) does the strategy beat XLK and the equal-weight universe on the window-level paired tests, (2) what's the mean excess vs XLK, and (3) what does `vs_INIT` say about whether rebalancing adds value. Then give me a one-paragraph read on which of the two paper framings the data supports.
