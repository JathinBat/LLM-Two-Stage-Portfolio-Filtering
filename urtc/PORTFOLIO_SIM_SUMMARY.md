# Project Work Summary — Sector Generalization, Reinvestment & Cash-Aware Portfolio Simulator

This document summarizes everything built and changed across this work session for the
two-stage LLM stock-filtering project. It covers new modules, the permutation-runner GUI,
all the design decisions, and the bug fixes, plus known caveats.

---

## 1. Overview

Three threads of work were completed:

1. **Sector generalization prep** — made the existing pipeline configurable to run on
   sectors other than technology (financials, healthcare), without changing analysis logic.
2. **New experiment engines** — two new "periodic reinvestment" runners: a simple
   full-reselection version, and a more realistic **cash-aware portfolio simulator** with
   real share prices, a cash ledger, buy/sell decisions, and risk metrics.
3. **A full permutation runner + GUI** for the cash-aware simulator, with feature parity to
   the original `permutation_runner.py` plus new conveniences, followed by a series of
   correctness and robustness fixes driven by live-run feedback.

New files created:

- `sector_config.py` — sector registry (universes, benchmark, news, keyword).
- `reinvestment_runner.py` — full-reselection periodic reinvestment orchestrator.
- `portfolio_sim_runner.py` — cash-aware stateful portfolio simulator + risk.
- `risk_metrics.py` — reusable risk-adjusted performance metrics.
- `portfolio_sim_permutation_runner.py` — grid runner + Tkinter GUI for the simulator.
- `sectors/` — per-sector news scaffold + README.

Files patched:

- `permutation_runner.py` — reads sector from config (default technology, unchanged).
- `SCCUR_TESTS/investment_strategy_generator.py` — sector wiring, an optional
  Stage-Two feedback slot, ticker sanitization, price-history guard, and connection retries.

---

## 2. Sector generalization

### `sector_config.py`
A single registry that owns everything sector-specific, so the analysis code is no longer
hard-coded to technology. Sectors: **technology, financials, healthcare**. Each entry holds:

- `keyword` — the Stage-One prompt sector keyword.
- ticker universe with sub-sector tags and company names (technology preserves the exact
  original 29-ticker list; financials and healthcare are ~30 large-caps each).
- `benchmark` — kept as `SPY` for all three (so "beats the market" stays comparable),
  with each sector's ETF (`XLK`/`XLF`/`XLV`) recorded for optional "beats the sector" use.
- `news_file` / `news_dir` and an `isolate_news` flag.

The active sector is selected with the `SECTOR` environment variable (default `technology`).
Helper functions (`keyword()`, `universe()`, `benchmark()`, `news_file()`, `isolate_news()`,
`active_sector()`) are what the runner and generator call.

### Wiring (backward-compatible)
- `permutation_runner.py`: `KEYWORD` and `NEWS_DATA_PATH` now come from `sector_config`.
- `investment_strategy_generator.py`: `UNFILTERED_TICKER_UNIVERSE` and the two `SPY`
  benchmark references now come from `sector_config`.
- **News isolation:** the generator's `_load_news_data` previously globbed *every* news CSV
  in the workspace (so a non-tech run still ingested tech news). For any sector with
  `isolate_news=True`, it now uses only that sector's own file(s) and skips the workspace
  walk. Technology keeps its legacy glob, so its behavior is byte-identical.

### `sectors/` scaffold
`sectors/financials/` and `sectors/healthcare/` with header-only template CSVs and a README
documenting how to run a sector (`SECTOR=financials python …`), the news schema, and the
isolation behavior. **Gating note:** this is code prep only — the sectors have the right
universe/benchmark but need real dated, pre-window news sourced before a live run is
meaningful (that requires news-API calls and was intentionally out of scope).

---

## 3. `reinvestment_runner.py` (full-reselection)

The first, simpler reinvestment engine. At each rebalance across a 12-month window it
liquidates and fully reselects the portfolio, compounding value forward. Two feedback
injection modes for the prior holdings' performance:

- **reflection_layer** — performance is reviewed *before* Stage One (lands in the
  candidate-identification prompt).
- **financial_report** — performance rides with the Stage-Two financial validation
  (a new optional `final_step_context` slot threaded into the generator).

Rebalance schedule is fully configurable (no default): `--segments N`, `--interval-months M`,
or explicit `--dates`. A `StrategyBackend` abstraction lets it run against the real generator
or a deterministic mock; validated in dry-run (no API).

---

## 4. `portfolio_sim_runner.py` (cash-aware simulator)

The main, realistic engine. It maintains a live portfolio (cash + fractional share positions
with cost basis) and lets the LLM trade under a hard budget.

### Core mechanics
- **Portfolio state:** cash, positions (shares + average cost), mark-to-market valuation,
  holdings count.
- **Atomic order execution:** orders are dollar-denominated. Fractional shares by default
  (assumes commission-free fractional trading), with a `--whole-shares` mode. A decision is
  validated on a copy and only committed if valid.
- **Budget rule:** total buys may not exceed available cash; the holdings cap `Y` is enforced;
  buys must be into priced candidates.

### "Sell everything first" simplification
Originally each rebalance was a mixed buy/sell decision, which made budget reasoning hard for
the model. This was changed so that **at each rebalance the engine liquidates the entire
portfolio to cash first**, then the model faces a clean, exact cash budget and issues
**buy-only** orders (up to `Y` holdings, may keep cash uninvested). It still receives the
prior holdings' realized performance as context. This is effectively full reselection but with
the cash-aware buy stage and daily risk metrics preserved.

### Budget emphasis
The live decision prompt was strengthened with a prominent **HARD BUDGET RULE** block that
states the exact cash figure many times, tells the model to sum its orders and confirm they
fit before answering, and that going over gets the whole response rejected.

### Multi-turn correction (reject-and-reprompt)
Instead of re-sending a fresh prompt on an over-budget response, `RealBackend.decide` now keeps
a **stateful conversation per rebalance**: on a rejection it appends the model's own failing
reply as an assistant turn, then a user turn quoting the exact rejection (including the
overdraw amount) and asking it to fix — all in the same chat series. A new period starts a
fresh conversation. A deterministic safe-fallback still trims orders if retries are exhausted.

### Backends
- `MockBackend` — deterministic prices/decisions for dry-run validation (no API).
- `RealBackend` — wraps the generator's data helpers (news, financials, price histories) plus
  the cash-aware prompts. Requires `OPENAI_API_KEY` from the environment; never reads
  source-embedded keys.

---

## 5. `risk_metrics.py`

Reusable, dependency-light (numpy) functions computed from a **daily** portfolio value series:

- Total return, annualized return (CAGR), annualized volatility.
- **Sharpe** (excess/total vol), **Sortino** (downside deviation), **Max Drawdown**, **Calmar**.
- **Beta / alpha / R²** versus the SPY benchmark over the same window.
- Configurable risk-free rate; 252-day annualization; graceful handling of short series.

The simulator reconstructs a true daily value series between rebalances (shares are fixed, so
value = Σ shares × daily adjusted close + cash), pulls the SPY daily series over the same dates,
and reports both the portfolio's risk profile and the benchmark's. Validated with known-answer
tests (constant series → zero vol, hand-built path → exact −25% drawdown, beta-vs-itself = 1,
Calmar = annret/|maxDD|, etc.).

---

## 6. `portfolio_sim_permutation_runner.py` (grid + GUI)

A grid runner and Tkinter GUI for the cash-aware simulator, rebuilt to match the original
`permutation_runner.py` feature set and then extended.

### Grid & parameters
Cartesian product over: **sector × model × period × start-date × rebalance-frequency ×
filter-config (X→Y)**, where `X` is the Stage-One candidate count and `Y` the holdings cap.
Each dimension has include/exclude checkboxes applied at Start, plus an ID range and a
start-date range.

### Feature parity with the original runner
- OpenAI + NYT API-key fields (masked), saved to a config file and pre-loaded.
- Threads and runs-per-permutation spinboxes; budget, risk-free, share-mode, and results-folder
  settings.
- **Start / Pause / Stop / Retry-errors** controls; live progress bar and P/S/R/D/E stats;
  sortable, color-coded status table.
- **Model training-cutoff skip logic** — periods whose start is on/before a model's cutoff are
  auto-cancelled (see fixes below), computed before Start.
- Per-run JSON output to a configurable folder; Excel export; session resume by metadata;
  config persistence.
- **Folder input parity** — value sanitized and written back, locked during runs, output routed
  into `results/portfolio_sim/<folder>/`, and resume recovers recursively from all subfolders.

### New conveniences
- **Per-permutation logs:** each permutation's full simulator output is captured to its own
  buffer via a `ThreadRoutedStream` that routes each worker thread's stdout by thread id — so
  concurrent workers never interleave. Click any row to view its log.
- **Per-worker pop-out windows** streaming each worker's live log during threaded runs.
- **Queued-runs window** — a dedicated window listing only the current batch with a
  completed/total header and per-permutation run counts.
- **Queued/completed counter** shown the instant you click Start.
- **Double-click a permutation** to open its saved result JSON in the OS default app (with a
  recursive-glob fallback if the stored path is missing).
- **Retry All (replace)** — re-runs every filter-matched permutation regardless of status,
  deleting their old JSON files and clearing in-memory results first, then scheduling a full
  set of fresh runs. Skipped and non-matching permutations are left untouched.
- **S&P 500 columns** — alongside ROI, the table shows the benchmark's return for that exact
  window ("S&P 500") and the difference ("vs S&P"); the batch window and Excel export include
  the benchmark too.

Modes: dry-run (MockBackend, no API) by default; `--live` uses the RealBackend. A `--headless`
mode runs the grid without the GUI and can save a results JSON, and `--save-excel-from-json`
rebuilds the workbook from saved runs.

---

## 7. Bug fixes (driven by live-run feedback)

- **Empty price-history crash** — `_calculate_performance_metrics` called `iloc[-1]` before its
  length check, throwing "single positional indexer is out-of-bounds" when a ticker's history
  came back empty. Added an early guard for empty/None histories (1y, 3y, 5y).

- **Ticker footnote symbols (`AMD*`, `AVGO*`)** — the LLM sometimes returned footnoted symbols;
  the `*` is illegal in a Windows path and breaks the Yahoo URL. Tickers are now sanitized both
  at the candidate source (screening) and defensively in the generator's cache-path builder and
  both Yahoo chart-fetch functions. Legitimate punctuation (`BRK.B`, `BRK-B`) is preserved.

- **Prices missing at a rebalance → gains erased** — the decision-time `get_prices` used a
  different, intermittently-failing helper than the daily reconstruction, so at some rebalances
  it returned nothing; valuation/liquidation then fell back to cost basis (erasing gains) and
  buys rejected for "no price." Fixed by (a) routing `get_prices` through the same reliable
  Yahoo chart-fetch the daily reconstruction uses, (b) carrying forward last-known prices so a
  transient miss never liquidates at cost, and (c) pre-filtering candidates to only those with a
  real price.

- **OpenAI connection drops (`WinError 10054`)** — the retry wrapper only retried on
  `RateLimitError`, so connection drops killed the run. Added exponential-backoff retries for
  transient `APIConnectionError` / `APITimeoutError` / `InternalServerError`, while permanent
  4xx errors (auth/bad-request) still fail fast and rate-limit handling is unchanged.

- **Training-cutoff cancellation** — a period whose start is on/before a model's training cutoff
  is now cancelled entirely (comparison tightened from `<` to `<=`), so e.g. gpt-5.1 with a
  2024-09 start (its cutoff) is skipped before Start rather than run.

- **Stop halts the whole queue** — Stop now aborts in-flight runs at their next step (a stop
  check is wrapped around each backend call), reverting them cleanly to Pending, in addition to
  halting new runs.

- **File-integrity restoration** — the runner file had been silently truncated at one point,
  losing its `main()` / entry point (so running it did nothing); the missing tail was restored.

---

## 8. Known caveat: sandbox mount desync

During development, the isolated build/validation sandbox intermittently fell out of sync with
a stale or truncated copy of the two largest files (`portfolio_sim_runner.py`,
`portfolio_sim_permutation_runner.py`). Your actual files on disk are the authoritative,
complete versions and are what the app runs against. When the sandbox was stale, changes were
validated by (a) reproducing the exact logic in isolation and (b) reading the authoritative
file view, rather than by executing the file in place. If a fresh launch ever throws a
`SyntaxError`/`NameError` on startup, it's worth pasting the traceback so the specific line can
be fixed directly.

---

## 9. Suggested next steps

- Source real dated, pre-window news for financials and healthcare to make those sectors
  runnable end-to-end (the one remaining gating item for the sector study).
- Consider whether the model should be *required* to deploy all cash now that budgeting is a
  clean buy-only stage (currently it may retain cash).
- Optionally add a buy-and-hold-of-initial-picks baseline per run, to separate the value added
  by active rebalancing from the sector's drift.
